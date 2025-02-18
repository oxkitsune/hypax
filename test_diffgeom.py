from typing import Callable, Sequence
import torch
import jax.numpy as jnp
import time
import jax

from hax.manifolds.poincare_ball._diffgeom import expmap0 as hax_expmap0
from hypll.manifolds.poincare_ball.math.diffgeom import expmap0 as hypll_expmap0

from tqdm.auto import tqdm


def mobius_add(x, y, c):
    from hax.manifolds.poincare_ball._diffgeom import mobius_add as hax_mobius_add
    from hypll.manifolds.poincare_ball.math.diffgeom import (
        mobius_add as hypll_mobius_add,
    )

    torch_x = torch.tensor(x).float()
    torch_y = torch.tensor(y).float()
    torch_c = torch.tensor(y).float()
    torch_mobius_add = hypll_mobius_add(torch_x, torch_y, torch_c)

    jax_x = jnp.array(x)
    jax_y = jnp.array(y)
    jax_c = jnp.array(y)
    jax_mobius_add = hax_mobius_add(jax_x, jax_y, jax_c)

    assert jnp.all(jnp.isclose(jax_x, torch_x.numpy())), "x differs"
    assert jnp.all(jnp.isclose(jax_y, torch_y.numpy())), "y differs"
    assert jnp.all(jnp.isclose(jax_c, torch_c.numpy())), "c differs"
    assert jnp.all(jnp.isclose(jax_mobius_add, torch_mobius_add.numpy())), (
        "mobius add output differs"
    )


mobius_add([1, 2, 3], [0.1, 0.2, 0.3], [0.1, 0.1, 0.1])
mobius_add([1, 2, 3], [0.1, 0.2, 0.3], [0.01, 0.41, 0.12])


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
    assert jnp.all(jnp.isclose(jax_proj, torch_proj.numpy())), (
        "projected output differs"
    )


project([1, 2, 3], [0.1, 0.1, 0.1])
project([1, 2, 3], [0.01, 0.41, 0.12])


def expmap0(x, c):
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


def logmap0(x, c):
    from hax.manifolds.poincare_ball._diffgeom import logmap0 as hax_logmap0
    from hypll.manifolds.poincare_ball.math.diffgeom import logmap0 as hypll_logmap0

    torch_x = torch.tensor(x).float()
    torch_c = torch.tensor(c)
    torch_logmap0 = hypll_logmap0(torch_x, torch_c)

    jax_x = jnp.array(x, dtype=jnp.float32)
    jax_c = jnp.array(c)
    jax_logmap0 = hax_logmap0(jax_x, jax_c)

    assert jnp.all(jnp.isclose(jax_x, torch_x.numpy())), (
        f"x differs, expected: {torch_x.numpy()} got: {jax_x}"
    )
    assert jnp.all(jnp.isclose(jax_c, torch_c.numpy())), (
        f"c differs, expected: {torch_c.numpy()} got: {jax_c}"
    )
    assert jnp.all(jnp.isclose(jax_logmap0, torch_logmap0.numpy())), (
        f"logmap0 output differs, expected: {torch_logmap0.numpy()} got: {jax_logmap0}"
    )


logmap0([0.1, 0.1, 0.1], [3.3, 3.3, 3.3])
logmap0([0.1, 0.1, 0.1], [1.01, 1.41, 1.12])


def expmap(x, v, c):
    from hax.manifolds.poincare_ball._diffgeom import expmap as hax_expmap
    from hypll.manifolds.poincare_ball.math.diffgeom import expmap as hypll_expmap

    torch_x = torch.tensor(x, dtype=torch.float32)
    torch_v = torch.tensor(v, dtype=torch.float32)
    torch_c = torch.tensor(c, dtype=torch.float32)
    torch_expmap = hypll_expmap(torch_x, torch_v, torch_c)

    jax_x = jnp.array(x)
    jax_v = jnp.array(v)
    jax_c = jnp.array(c)
    jax_expmap = hax_expmap(jax_x, jax_v, jax_c)

    assert jnp.all(jnp.isclose(jax_x, torch_x.numpy())), "x differs"
    assert jnp.all(jnp.isclose(jax_v, torch_v.numpy())), "v differs"
    assert jnp.all(jnp.isclose(jax_c, torch_c.numpy())), "c differs"
    assert jnp.all(jnp.isclose(jax_expmap, torch_expmap.numpy())), "expmap differs"


expmap([1, 2, 3], [3.3, 3.3, 3.3], [0.1, 0.1, 0.1])
expmap([1, 2, 3], [1.01, 1.41, 1.12], [0.1, 0.1, 0.1])


def logmap(x, v, c):
    from hax.manifolds.poincare_ball._diffgeom import logmap as hax_logmap
    from hypll.manifolds.poincare_ball.math.diffgeom import logmap as hypll_logmap

    torch_x = torch.tensor(x, dtype=torch.float32)
    torch_v = torch.tensor(v, dtype=torch.float32)
    torch_c = torch.tensor(c, dtype=torch.float32)
    torch_logmap = hypll_logmap(torch_x, torch_v, torch_c)

    jax_x = jnp.array(x)
    jax_v = jnp.array(v)
    jax_c = jnp.array(c)
    jax_logmap = hax_logmap(jax_x, jax_v, jax_c)

    assert jnp.all(jnp.isclose(jax_x, torch_x.numpy())), (
        f"x differs expected: {torch_x.numpy()}, got: {jax_x}"
    )
    assert jnp.all(jnp.isclose(jax_v, torch_v.numpy())), (
        f"v differs expected: {torch_v.numpy()}, got: {jax_v}"
    )
    assert jnp.all(jnp.isclose(jax_c, torch_c.numpy())), (
        f"c differs expected: {torch_c.numpy()}, got: {jax_c}"
    )
    assert jnp.all(jnp.isclose(jax_logmap, torch_logmap.numpy())), (
        f"logmap differs expected: {torch_logmap.numpy()}, got: {jax_logmap}"
    )


logmap([0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.1, 0.1, 0.1])
logmap([0.1, 0.1, 0.1], [0.01, 0.41, 0.12], [0.1, 0.1, 0.1])


def gyration(u, v, w, c):
    from hax.manifolds.poincare_ball._diffgeom import gyration as hax_gyration
    from hypll.manifolds.poincare_ball.math.diffgeom import gyration as hypll_gyration

    torch_u = torch.tensor(u, dtype=torch.float32)
    torch_v = torch.tensor(v, dtype=torch.float32)
    torch_w = torch.tensor(w, dtype=torch.float32)
    torch_c = torch.tensor(c, dtype=torch.float32)
    torch_out = hypll_gyration(torch_u, torch_v, torch_w, torch_c)

    jax_u = jnp.array(u)
    jax_v = jnp.array(v)
    jax_w = jnp.array(w)
    jax_c = jnp.array(c)
    jax_out = hax_gyration(jax_u, jax_v, jax_w, jax_c)

    assert jnp.all(jnp.isclose(jax_u, torch_u.numpy())), (
        f"u differs expected: {torch_u.numpy()}, got: {jax_u}"
    )
    assert jnp.all(jnp.isclose(jax_v, torch_v.numpy())), (
        f"v differs expected: {torch_v.numpy()}, got: {jax_v}"
    )
    assert jnp.all(jnp.isclose(jax_w, torch_w.numpy())), (
        f"w differs expected: {torch_w.numpy()}, got: {jax_w}"
    )
    assert jnp.all(jnp.isclose(jax_c, torch_c.numpy())), (
        f"c differs expected: {torch_c.numpy()}, got: {jax_c}"
    )
    assert jnp.all(jnp.isclose(jax_out, torch_out.numpy())), (
        f"gyration differs expected: {torch_out.numpy()}, got: {jax_out}"
    )


gyration([0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.2, 0.2, 0.2], [0.1, 0.1, 0.1])
gyration([0.1, 0.1, 0.1], [0.01, 0.41, 0.12], [0.3, 0.2, 0.4], [0.1, 0.1, 0.1])

gpu_device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def speed_torch(x, c):
    return hypll_expmap0(x, c)


def make_tensor(s, num_iters, key):
    return torch.rand((num_iters, *s)).to(gpu_device)


def speed_jax(x, c):
    expmap_fn = jax.vmap(hax_expmap0, in_axes=(0, 0, None))

    return expmap_fn(x, c, -1)


def make_jax_array(s, num_iters, key):
    return jax.block_until_ready(jax.random.uniform(key, (num_iters, *s)))


def bench_function(
    func: Callable[[Sequence[int], int], None],
    make_array: Callable,
    shapes: Sequence[Sequence[int]],
    num_runs: int = 5,
    num_iters: int = 100_000,
):
    tqdm.write(f"benchmarking {func.__name__}...")
    tqdm.write(
        f"measuring execution time of {num_iters} iters (mean over {num_runs} runs)"
    )
    key = jax.random.key(0)
    results = {s: [] for s in shapes}
    for _ in tqdm(range(num_runs), desc=func.__name__):
        for _ in tqdm(range(5), desc="warming up", leave=False):
            for s in shapes:
                keyx, keyc = jax.random.split(key, 2)
                x = make_array(s, num_iters, keyx)
                c = make_array(s, num_iters, keyc)
                start = time.time()
                jax.block_until_ready(func(x, c))

        for s in shapes:
            keyx, keyc = jax.random.split(key, 2)
            x = make_array(s, num_iters, keyx)
            c = make_array(s, num_iters, keyc)
            start = time.time()
            jax.block_until_ready(func(x, c))
            results[s].append((time.time() - start) * 1000)

    tqdm.write("========")
    for k, v in results.items():
        formatted_durations = ", ".join([f"{t:.4f}ms" for t in v])
        tqdm.write(
            f"   - {k}: {sum(v) / num_runs:.4f}ms ({len(v)} unique) [{formatted_durations}]"
        )


num_runs = 5
num_iters = 1_000
shapes = [(1, 3), (1, 30), (1, 300), (1, 3000)]
torch_results = bench_function(speed_torch, make_tensor, shapes, num_runs, num_iters)
jax_results = bench_function(
    jax.jit(speed_jax), make_jax_array, shapes, num_runs, num_iters
)
