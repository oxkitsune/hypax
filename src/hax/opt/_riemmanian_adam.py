import jax
import jax.numpy as jnp


def riemannian_adam(lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False):
    def init_fn(params):
        def init_state(param):
            # Manifold parameter: assume dict with key "manifold" and "tensor"
            if isinstance(param, dict) and "manifold" in param:
                t = param["tensor"]
                state = {"step": 0, "m": jnp.zeros_like(t), "v": jnp.zeros_like(t)}
                if amsgrad:
                    state["v_hat"] = jnp.zeros_like(t)
                return state
            else:
                state = {"step": 0, "m": jnp.zeros_like(param)}

                # For Euclidean parameters, v is computed as a summed quantity over the last axis.
                state["v"] = (
                    jnp.zeros(param.shape[:-1] + (1,))
                    if param.ndim > 0
                    else jnp.zeros_like(param)
                )
                if amsgrad:
                    state["v_hat"] = state["v"]
                return state

        opt_state = jax.tree_util.tree_map(init_state, params)
        return {
            "params": params,
            "state": opt_state,
            "hyperparams": {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": amsgrad,
            },
        }

    def update_fn(grads, opt_state):
        hp = opt_state["hyperparams"]
        beta1, beta2 = hp["betas"]
        lr = hp["lr"]
        eps = hp["eps"]
        weight_decay = hp["weight_decay"]
        amsgrad = hp["amsgrad"]

        def update_single(param, grad, state):
            step = state["step"] + 1
            if isinstance(param, dict) and "manifold" in param:
                manifold = param["manifold"]
                # Weight decay on the Euclidean representation.
                grad = grad + weight_decay * param["tensor"]
                # Convert to tangent vector.
                g = manifold.euc_to_tangent(param, grad)
                m = beta1 * state["m"] + (1 - beta1) * g
                v_new = beta2 * state["v"] + (1 - beta2) * manifold.inner(
                    g, g, keepdim=True, safe_mode=False
                )
                if amsgrad:
                    v_hat = jnp.maximum(state["v_hat"], v_new)
                    denom = jnp.sqrt(v_hat / (1 - beta2**step)) + eps
                else:
                    denom = jnp.sqrt(v_new / (1 - beta2**step)) + eps
                direction = -lr * (m / (1 - beta1**step)) / denom
                new_tensor = manifold.expmap(param["tensor"], direction)
                new_m = manifold.transp(param["tensor"], new_tensor, m)
                new_param = dict(param, tensor=new_tensor)
                new_state = {"step": step, "m": new_m, "v": v_new}
                if amsgrad:
                    new_state["v_hat"] = v_hat
                return new_param, new_state
            else:
                grad = grad + weight_decay * param
                m = beta1 * state["m"] + (1 - beta1) * grad
                v_new = beta2 * state["v"] + (1 - beta2) * jnp.sum(
                    grad**2, axis=-1, keepdims=True
                )
                if amsgrad:
                    v_hat = jnp.maximum(state["v_hat"], v_new)
                    denom = jnp.sqrt(v_hat / (1 - beta2**step)) + eps
                else:
                    denom = jnp.sqrt(v_new / (1 - beta2**step)) + eps
                direction = m / (1 - beta1**step) / denom
                new_param = param - lr * direction
                new_state = {"step": step, "m": m, "v": v_new}
                if amsgrad:
                    new_state["v_hat"] = v_hat
                return new_param, new_state

        # Apply update_single over the pytree.
        updated = jax.tree_util.tree_map(
            lambda p, g, s: update_single(p, g, s),
            opt_state["params"],
            grads,
            opt_state["state"],
        )
        new_params = jax.tree_util.tree_map(lambda tup: tup[0], updated)
        new_state = jax.tree_util.tree_map(lambda tup: tup[1], updated)
        return {"params": new_params, "state": new_state, "hyperparams": hp}

    def get_params(opt_state):
        return opt_state["params"]

    return init_fn, update_fn, get_params
