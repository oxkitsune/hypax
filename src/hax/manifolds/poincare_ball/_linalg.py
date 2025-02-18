import jax
import jax.numpy as jnp


def poincare_hyperplane_dists(
    x: jax.Array,
    z: jax.Array,
    r: jax.Array | None,
    c: jax.Array,
    axis: int = -1,
) -> jax.Array:
    """The Poincare signed distance to hyperplanes operation.

    Args:
        x (jax.Array): The input values.
        z (jax.Array): The hyperbolic vectors describing the hyperplane orientations
        r (jax.Array | None): The hyperplane offsets
        c (jax.Array): The curvature of the Poincare disk.
        dim (int, optional): The axis. Defaults to -1.

    Returns:
        jax.Array: signed distances of input w.r.t. the hyperplanes, denoted by v_k(x) in the HNN++ paper
    """

    axis_shifted_x = jnp.moveaxis(x, source=axis, destination=-1)

    c_sqrt = c.sqrt()
    lam = 2 / (1 - c * jnp.pow(axis_shifted_x, 2).sum(axis=-1, keepdim=True))
    z_norm = jnp.linalg.norm(z, axis=0)
    z_norm = jnp.maximum(z_norm, 1e-15)

    # Computation can be simplified if there is no offset
    if r is None:
        dim_shifted_output = (
            2
            * z_norm
            / c_sqrt
            * jnp.asinh(c_sqrt * lam / z_norm * jnp.matmul(axis_shifted_x, z))
        )
    else:
        two_csqrt_r = 2.0 * c_sqrt * r
        dim_shifted_output = (
            2
            * z_norm
            / c_sqrt
            * jnp.asinh(
                c_sqrt
                * lam
                / z_norm
                * jnp.matmul(axis_shifted_x, z)
                * two_csqrt_r.cosh()
                - (lam - 1) * two_csqrt_r.sinh()
            )
        )

    return jnp.moveaxis(dim_shifted_output, source=-1, destination=axis)


def poincare_fully_connected(
    x: jax.Array,
    z: jax.Array,
    bias: jax.Array | None,
    c: jax.Array,
    axis: int = -1,
) -> jax.Array:
    """The Poincare fully connected layer operation.

    Args:
        x (jax.Array): The layer inputs
        z (jax.Array): The hyperbolic vectors describing the hyperplane orientations
        bias (jax.Array | None): The layer biases (hyperplane offsets)
        c (jax.Array): The curvature of the Poincare disk.
        axis (int, optional): The axis. Defaults to -1.

    Returns:
        jax.Array: Poincare FC transformed hyperbolic array, commonly denoted by y
    """
    c_sqrt = c.sqrt()
    x = poincare_hyperplane_dists(x=x, z=z, r=bias, c=c, axis=axis)
    x = (c_sqrt * x).sinh() / c_sqrt
    return x / (1 + (1 + c * jnp.pow(x, 2).sum(axis=axis, keepdim=True)).sqrt())
