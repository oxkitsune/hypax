import jax
import jax.numpy as jnp


def project(x: jax.Array, c: jax.Array, dim: int = -1, eps: float = -1.0) -> jax.Array:
    # Convert eps to JAX array with proper dtype
    eps = jnp.asarray(eps, dtype=x.dtype)
    eps_val = jnp.where(
        eps < 0, jnp.array(4e-3 if x.dtype == jnp.float32 else 1e-5, dtype=x.dtype), eps
    )

    maxnorm = (1 - eps_val) / jnp.sqrt(c + 1e-15)
    maxnorm = jnp.where(c > 0, maxnorm, jnp.full_like(c, 1e15))

    norm = jnp.linalg.norm(x, axis=dim, keepdims=True, ord=2)
    norm = jnp.maximum(norm, 1e-15)
    cond = norm > maxnorm
    projected = x / norm * maxnorm

    return jnp.where(cond, projected, x)


def expmap0(v: jax.Array, c: jax.Array, dim: int = -1):
    v_norm = jnp.linalg.norm(v, axis=dim, keepdims=True)
    v_norm = jnp.maximum(v_norm, 1e-15)

    v_norm_c_sqrt = v_norm * jnp.sqrt(c)

    return project(jnp.tanh(v_norm_c_sqrt) * v / v_norm_c_sqrt, c, dim=dim)
