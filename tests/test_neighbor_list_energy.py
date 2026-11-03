"""Neighbor-list rigid-body energy agrees with the dense (all-pairs) one."""
import numpy as np
import pytest

from base.jax_setup import configure_jax

configure_jax()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax_md import rigid_body  # noqa: E402

from base.energy import build_pair_energy_fn, build_pair_energy_fn_neighbor_list  # noqa: E402
from base.geometry import energy_matrix  # noqa: E402
from base.initial_conditions import preassembled_rings, rsa_2d  # noqa: E402

pytestmark = pytest.mark.fast

N, L = 40, 30.0


def energies(angle_deg, start):
    thetas = jnp.array([0.0, np.deg2rad(angle_deg)])
    eng = energy_matrix(jnp.array([1.0, 5.0, 1.0]))
    kw = dict(rep_A=10_000.0)
    dense, _ = build_pair_energy_fn(thetas, eng, L, **kw)
    nl_fns, nl, _ = build_pair_energy_fn_neighbor_list(thetas, eng, L, **kw)
    if start == 'rsa':
        pos, ori = rsa_2d(N, L, seed=3, centered=False)
    else:
        pos, ori = preassembled_rings(N, L, np.deg2rad(angle_deg), seed=3)
    body = rigid_body.RigidBody(jnp.asarray(pos), jnp.asarray(ori))
    nbrs = nl_fns.allocate(body)
    assert not nbrs.did_buffer_overflow
    return dense, nl, nbrs, body


@pytest.mark.parametrize('angle_deg', [70.0, 90.0, 100.0])
@pytest.mark.parametrize('start', ['rsa', 'rings'])
def test_energy_and_forces_match_dense(angle_deg, start):
    dense, nl, nbrs, body = energies(angle_deg, start)
    assert float(nl(body, nbrs)) == pytest.approx(float(dense(body)), rel=1e-10, abs=1e-8)
    g_dense = jax.grad(dense)(body)
    g_nl = jax.grad(lambda b: nl(b, nbrs))(body)
    np.testing.assert_allclose(g_nl.center, g_dense.center, atol=1e-7)
    np.testing.assert_allclose(g_nl.orientation, g_dense.orientation, atol=1e-7)


def test_nvt_trajectory_matches_dense(tmp_path):
    """Same start, same deterministic NVT: neighbor-list and dense runs agree."""
    from base.params import generate_parameter_grid
    from base.trajectory import open_trajectory
    from simulation.src.canonical_md import run_many_particle_md

    params = generate_parameter_grid(np.deg2rad([90.0]), np.array([2.0]))[0]
    pos, ori = preassembled_rings(20, 20.0, np.deg2rad(90.0), seed=5)
    out = {}
    for name, nl in (('dense', False), ('nl', True)):
        path = tmp_path / f'{name}.h5'
        run_many_particle_md(
            20, 20.0, params, 3000, 1.0, jax.random.PRNGKey(5), 1e-3, pos, ori,
            sample_interval=500, trajectory_path=path, checkpoint_interval=1500,
            seed=5, neighbor_list=nl)
        r = open_trajectory(path)
        out[name] = (r.series('production', 'potential_energy'),
                     r.series('production', 'kinetic_energy'))
    np.testing.assert_allclose(out['nl'][0], out['dense'][0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out['nl'][1], out['dense'][1], rtol=1e-6, atol=1e-6)
