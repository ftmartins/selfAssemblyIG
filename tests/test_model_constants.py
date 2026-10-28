"""ADR-0001: the Morse bond is the full Morse out to r_onset = 1.5, a smooth
window to zero at r_cutoff = 2.0, and nothing beyond.  Checked for every
energy function of both engines (canonical MD and hybrid GCMC)."""
import numpy as np
import pytest

from base.jax_setup import configure_jax

configure_jax()

import jax.numpy as jnp  # noqa: E402
from jax_md import rigid_body  # noqa: E402

from base import config  # noqa: E402
from base.energy import (  # noqa: E402
    build_pair_energy_fn, particle_energy, total_energy, total_energy_jax,
    total_energy_pbc,
)
from base.geometry import SHAPE_CONFIGS, energy_matrix  # noqa: E402
from base.params import make_gcmc_params  # noqa: E402

pytestmark = pytest.mark.fast

E_AB = 5.0
ALPHA = 7.0
BOX = 100.0
OPENING = np.pi  # straight particle: A at +x, B at -x


def morse(r, eps=E_AB, alpha=ALPHA):
    return eps * (1.0 - np.exp(-alpha * r)) ** 2 - eps


def window(r, onset=1.5, cutoff=2.0):
    if r < onset:
        return 1.0
    if r >= cutoff:
        return 0.0
    r2, c2, o2 = r * r, cutoff ** 2, onset ** 2
    return (c2 - r2) ** 2 * (c2 + 2 * r2 - 3 * o2) / (c2 - o2) ** 3


def expected(r):
    return window(r) * morse(r)


# separations: inside the patch radius, between patch radius and onset,
# inside the smoothing window, beyond the cutoff
SEPARATIONS = [0.2, 0.5, 1.0, 1.4, 1.75, 1.95, 2.0, 2.3]


def two_particle_q(r):
    """Particle 1 at the origin, particle 2 placed so that the only patch pair
    within range is B(2)-A(1), separated by exactly ``r``.  Cores do not
    overlap (centers 2 + r apart)."""
    return np.array([[0.0, 0.0, 0.0], [2.0 + r, 0.0, 0.0]])


def gcmc_params():
    p = make_gcmc_params(OPENING, E_AB, E_AA=E_AB, E_BB=E_AB, rep_A=0.0)
    p['E_AA'] = p['E_BB'] = 0.0  # isolate the A-B pair
    return p


def test_config_constants():
    assert config.ALPHA == 7.0
    assert config.MORSE_R_ONSET == 1.5
    assert config.MORSE_R_CUTOFF == 2.0
    assert config.PATCH_RADIUS == pytest.approx(0.38, abs=0.005)


def test_no_cutoff_identifier_names_the_patch_radius():
    assert not hasattr(config, 'R_CUTOFF')
    assert config.MORSE_R_CUTOFF != config.PATCH_RADIUS


@pytest.mark.parametrize('r', SEPARATIONS)
def test_gcmc_numpy_particle_energy(r):
    q = two_particle_q(r)
    assert particle_energy(0, q, BOX, gcmc_params()) == pytest.approx(expected(r), abs=1e-12)
    assert total_energy_pbc(q, BOX, gcmc_params()) == pytest.approx(expected(r), abs=1e-12)


@pytest.mark.parametrize('r', SEPARATIONS)
def test_gcmc_jax_total_energy(r):
    q = jnp.asarray(two_particle_q(r).reshape(-1))
    assert float(total_energy_jax(q, BOX, gcmc_params())) == pytest.approx(expected(r), abs=1e-12)


@pytest.mark.parametrize('r', SEPARATIONS)
def test_free_space_total_energy(r):
    q = jnp.asarray(two_particle_q(r).reshape(-1))
    e = total_energy(q, OPENING, E_AB, ALPHA, E_AA=0.0, E_BB=0.0, rep_A=0.0)
    assert float(e) == pytest.approx(expected(r), abs=1e-12)


@pytest.mark.parametrize('r', SEPARATIONS)
def test_canonical_md_pair_energy(r):
    thetas = jnp.array([0.0, OPENING])
    eng = energy_matrix(jnp.array([0.0, E_AB, 0.0]))  # (AA, AB, BB)
    energy_fn, _ = build_pair_energy_fn(thetas, eng, BOX, rep_A=0.0, alpha_morse=ALPHA)
    q = two_particle_q(r)
    body = rigid_body.RigidBody(jnp.asarray(q[:, :2]), jnp.asarray(q[:, 2]))
    assert float(energy_fn(body)) == pytest.approx(expected(r), abs=1e-10)


def test_triangle_default_opening_angle_is_matching_angle():
    # matching angle theta_n = pi (n - 2) / n (CONTEXT.md); n = 3 -> 60 degrees
    cfg = SHAPE_CONFIGS['triangle']
    assert cfg['n_vertices'] == 3
    assert cfg['default_opening_angle'] == pytest.approx(60.0)
