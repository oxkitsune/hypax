import torch
import jax.numpy as jnp


def project(x, c):
    from hax.manifolds.poincare_ball._diffgeom import project as hax_project
    from hypll.manifolds.poincare_ball.math.diffgeom import project as hypll_project

    torch_x = torch.tensor(x).float()
    torch_c = torch.tensor(c)
    torch_proj = hypll_project(torch_x, torch_c)

    jax_x = jnp.array(x, dtype=jnp.float32)
    jax_c = jnp.array(c)
    jax_proj = hax_project(jax_x, jax_c)

    assert jnp.all(jnp.isclose(jax_x, torch_x.numpy())), "x differs"
    assert jnp.all(jnp.isclose(jax_c, torch_c.numpy())), "c differs"
    assert jnp.all(
        jnp.isclose(jax_proj, torch_proj.numpy())
    ), "projected output differs"


project([1, 2, 3], [0.1, 0.1, 0.1])
project([1, 2, 3], [0.01, 0.41, 0.12])


def expmap0(x, c):
    from hax.manifolds.poincare_ball._diffgeom import expmap0 as hax_expmap0
    from hypll.manifolds.poincare_ball.math.diffgeom import expmap0 as hypll_expmap0

    torch_x = torch.tensor(x).float()
    torch_c = torch.tensor(c)
    torch_proj = hypll_expmap0(torch_x, torch_c)

    jax_x = jnp.array(x, dtype=jnp.float32)
    jax_c = jnp.array(c)
    jax_proj = hax_expmap0(jax_x, jax_c)

    assert jnp.all(jnp.isclose(jax_x, torch_x.numpy())), "x differs"
    assert jnp.all(jnp.isclose(jax_c, torch_c.numpy())), "c differs"
    assert jnp.all(jnp.isclose(jax_proj, torch_proj.numpy())), "expmap0 output differs"


expmap0([1, 2, 3], [0.1, 0.1, 0.1])
expmap0([1, 2, 3], [0.01, 0.41, 0.12])
