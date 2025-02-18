import jax
import jax.numpy as jnp


def mobius_add(x: jax.Array, y: jax.Array, c: jax.Array, axis: int = -1) -> jax.Array:
    broadcast_dim = max(x.ndim, y.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis

    x2 = jnp.pow(x, 2).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True)
    y2 = jnp.pow(y, 2).sum(axis=axis - broadcast_dim + y.ndim, keepdims=True)

    xy = (x * y).sum(axis=axis, keepdims=True)

    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = jnp.maximum(1 + 2 * c * xy + c**2 * x2 * y2, 1e-15)

    return numerator / denominator


def project(x: jax.Array, c: jax.Array, axis: int = -1, eps: float = -1.0) -> jax.Array:
    eps = jnp.asarray(eps, dtype=x.dtype)
    eps_val = jnp.where(
        eps < 0, jnp.array(4e-3 if x.dtype == jnp.float32 else 1e-5, dtype=x.dtype), eps
    )

    maxnorm = (1 - eps_val) / jnp.sqrt(c + 1e-15)
    maxnorm = jnp.where(c > 0, maxnorm, jnp.full_like(c, 1e15))

    norm = jnp.linalg.norm(x, axis=axis, keepdims=True, ord=2)
    norm = jnp.maximum(norm, 1e-15)
    cond = norm > maxnorm
    projected = x / norm * maxnorm

    return jnp.where(cond, projected, x)


def expmap0(v: jax.Array, c: jax.Array, axis: int = -1):
    v_norm = jnp.linalg.norm(v, axis=axis, keepdims=True)
    v_norm = jnp.maximum(v_norm, 1e-15)

    v_norm_c_sqrt = v_norm * jnp.sqrt(c)

    return project(jnp.tanh(v_norm_c_sqrt) * v / v_norm_c_sqrt, c, axis=axis)


def logmap0(y: jax.Array, c: jax.Array, axis: int = -1):
    y_norm = jnp.linalg.norm(y, axis=axis, keepdims=True)
    y_norm = jnp.maximum(y_norm, 1e-15)

    y_norm_c_sqrt = y_norm * jnp.sqrt(c)

    return jnp.atanh(y_norm_c_sqrt) * y / y_norm_c_sqrt


def expmap(x: jax.Array, v: jax.Array, c: jax.Array, axis: int = -1):
    broadcast_dim = max(x.ndim, v.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis

    v_norm = jnp.linalg.norm(v, axis=axis - broadcast_dim + v.ndim, keepdims=True)
    v_norm = jnp.maximum(v_norm, 1e-15)

    lambda_x = 2 / jnp.maximum(
        1 - c * jnp.pow(x, 2).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True),
        1e-15,
    )

    c_sqrt = jnp.sqrt(c)
    second_term = jnp.tanh(c_sqrt * lambda_x * v_norm / 2) * v / (c_sqrt * v_norm)

    return project(mobius_add(x, second_term, c, axis=axis), c, axis=axis)


def logmap(x: jax.Array, y: jax.Array, c: jax.Array, axis: int = -1):
    broadcast_dim = max(x.ndim, y.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis

    min_x_y = mobius_add(-x, y, c, axis=axis)
    min_x_y_norm = jnp.linalg.norm(min_x_y, axis=axis, keepdims=True)
    min_x_y_norm = jnp.maximum(min_x_y_norm, 1e-15)

    lambda_x = 2 / jnp.maximum(
        1 - c * jnp.pow(x, 2).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True),
        1e-15,
    )

    c_sqrt = jnp.sqrt(c)
    return (
        2
        / (c_sqrt * lambda_x)
        * jnp.atanh(c_sqrt * min_x_y_norm)
        * min_x_y
        / min_x_y_norm
    )


def gyration(u: jax.Array, v: jax.Array, w: jax.Array, c: jax.Array, axis: int = -1):
    broadcast_dim = max(u.ndim, v.ndim, w.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis
    u2 = jnp.sum(u**2, axis=axis - broadcast_dim + u.ndim, keepdims=True)
    v2 = jnp.sum(v**2, axis=axis - broadcast_dim + v.ndim, keepdims=True)
    uv = jnp.sum(u * v, axis=axis - broadcast_dim + max(u.ndim, v.ndim), keepdims=True)
    uw = jnp.sum(u * w, axis=axis - broadcast_dim + max(u.ndim, w.ndim), keepdims=True)
    vw = jnp.sum(v * w, axis=axis - broadcast_dim + max(v.ndim, w.ndim), keepdims=True)

    K2 = c**2
    a = -K2 * uw * v2 + c * vw + 2 * K2 * uv * vw
    b = -K2 * vw * u2 - c * uw
    d = 1 + 2 * c * uv + K2 * u2 * v2

    return w + 2 * (a * u + b * v) / jnp.maximum(d, 1e-15)
