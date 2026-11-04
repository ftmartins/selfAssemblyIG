"""Equilibrium ring-yield machinery (docs/ring_yield_plan.md, validation step 1)."""
import itertools

import numpy as np
import pytest

from base.jax_setup import configure_jax

configure_jax()

from analysis import fisher  # noqa: E402
from analysis.mass_action import (SpeciesTable, at_density, at_fugacity, extend_chains,  # noqa: E402
                                  finite_n, solve_ln_z)
from analysis.partition import (Model, PairProposal, estimate_ring, pair_energy,  # noqa: E402
                                ring_graph_ok)
from analysis.ring_modes import analyze_ring, jacobian_factor, reference_ring  # noqa: E402

pytestmark = pytest.mark.fast


def dimer_table(lnK):
    return SpeciesTable.from_arrays(['monomer', 'chain'], [1, 2],
                                    [np.log(2 * np.pi), np.log(2 * np.pi) + lnK])


def test_dimer_mass_action_matches_quadratic():
    lnK, rho = 3.0, 0.05
    y = at_density(dimer_table(lnK), rho)
    q1, q2 = 2 * np.pi, 2 * np.pi * np.exp(lnK)
    z = (-q1 + np.sqrt(q1 ** 2 + 8 * q2 * rho)) / (4 * q2)      # q1 z + 2 q2 z^2 = rho
    assert y.ln_z == pytest.approx(np.log(z), abs=1e-10)
    assert y.particle.sum() == pytest.approx(1.0)
    assert y.cluster.sum() == pytest.approx(1.0)


def test_relative_yield_depends_on_density():
    t = SpeciesTable.from_arrays(['monomer', 'ring', 'ring'], [1, 3, 4], [1.8, 17.0, 21.0])
    lo, hi = at_density(t, 1e-4), at_density(t, 1e-1)
    r = lambda y: y.get('ring_4') / y.get('ring_3')
    assert r(hi) / r(lo) == pytest.approx(np.exp(hi.ln_z - lo.ln_z))


def test_recursion_matches_brute_force_partition_sum():
    t = SpeciesTable.from_arrays(['monomer', 'chain', 'ring'], [1, 2, 3], [0.3, 1.1, 0.7])
    N, A = 6, 2.5
    a = {k: A * np.exp(t.lnq[t.n == k]).sum() for k in (1, 2, 3)}
    # brute force over multisets of cluster sizes: Q_N = sum prod a_k^{m_k}/m_k!
    from math import factorial
    QN, mean3 = 0.0, 0.0
    for m1, m2, m3 in itertools.product(range(N + 1), repeat=3):
        if m1 + 2 * m2 + 3 * m3 != N:
            continue
        w = a[1] ** m1 / factorial(m1) * a[2] ** m2 / factorial(m2) * a[3] ** m3 / factorial(m3)
        QN += w
        mean3 += m3 * w
    y = finite_n(t, N, A)
    assert y.meta['mean_counts'][t.index('ring_3')] == pytest.approx(mean3 / QN, rel=1e-10)


def test_recursion_tends_to_mass_action_at_large_N():
    t = SpeciesTable.from_arrays(['monomer', 'chain', 'ring'], [1, 2, 3], [1.8, 6.0, 9.0])
    rho = 0.03
    y_inf = at_density(t, rho)
    N = 4000
    y_N = finite_n(t, N, N / rho)
    assert y_N.particle == pytest.approx(y_inf.particle, abs=2e-3)


def test_extend_chains_continues_linearly():
    n = np.arange(1, 9)
    lnq = 1.0 + 2.0 * n
    t = SpeciesTable.from_arrays(['monomer'] + ['chain'] * 7, n, lnq, ess=np.full(8, 1e4))
    e = extend_chains(t, n_max=20)
    assert e.lnq[e.n == 20][0] == pytest.approx(41.0)
    assert solve_ln_z(e, 0.01) < -2.0


@pytest.mark.parametrize('n', [3, 4, 6])
def test_jacobian_of_regular_ring(n):
    q = reference_ring(Model(np.pi * (n - 2) / n, 10.0), n)
    R2 = np.sum(q.reshape(n, 3)[:, :2] ** 2)
    assert jacobian_factor(q) == pytest.approx(2 * np.pi * n * np.sqrt(R2 + n), rel=1e-12)


def test_triangle_relaxes_to_valid_ring():
    m = Model(np.deg2rad(60), 10.0)
    md = analyze_ring(m, 3)
    assert md.valid and md.n_soft == 0
    assert md.E0 == pytest.approx(-30.0, abs=0.2)


def test_saddle_and_dissociated_states_rejected():
    m = Model(np.deg2rad(60), 10.0)
    q = reference_ring(m, 3).reshape(3, 3)
    assert ring_graph_ok(q[:, :2], q[:, 2], m)
    far = q.copy()
    far[2, :2] += 5.0                                         # dissociated
    assert not ring_graph_ok(far[:, :2], far[:, 2], m)
    # a strained triangle at 130 deg relaxes away from the ring, or is flagged
    md = analyze_ring(Model(np.deg2rad(130), 50.0), 3)
    assert md.valid or md.reasons


def test_pair_energy_matches_jax_energy():
    from base.energy import total_energy
    import jax.numpy as jnp
    m = Model(np.deg2rad(80), 7.0)
    rng = np.random.default_rng(1)
    q = np.column_stack([rng.uniform(0, 4, (4, 2)), rng.uniform(-np.pi, np.pi, 4)])
    i, j = np.triu_indices(4, 1)
    e_np = pair_energy(q[i, :2], q[i, 2], q[j, :2], q[j, 2], m).sum()
    e_jx = float(total_energy(jnp.asarray(q.ravel()), m.opening_angle, m.E_AB, m.alpha,
                              E_AA=m.E_AA, E_BB=m.E_BB, rep_A=m.eps_c))
    assert e_np == pytest.approx(e_jx, rel=1e-10)


def test_triangle_ln_q_matches_direct_sampling():
    from analysis.direct_z import direct_ln_q
    m = Model(np.deg2rad(60), 10.0)
    smc = estimate_ring(m, 3, n_walkers=8000, rng=0)
    d, derr, _ = direct_ln_q(m, 3, n_samples=60_000)
    assert smc.lnq == pytest.approx(d, abs=5 * np.hypot(smc.lnq_err, derr) + 0.02)


def test_pair_proposal_weights_are_flat():
    from analysis.partition import _kish
    p = PairProposal(Model(np.deg2rad(90), 20.0))
    _, _, lw = p.sample((5000,), np.random.default_rng(0))
    assert _kish(lw) > 0.8 * 5000


def test_cluster_metric_is_psd():
    t = SpeciesTable.from_arrays(['monomer', 'ring', 'ring'], [1, 3, 4], [1.8, 17.0, 21.0],
                                 scores=[[0, 0], [-3.0, 3.0], [2.0, 4.0]])
    y = at_density(t, 0.03)
    g = fisher.cluster_metric(t, y)
    assert np.all(np.linalg.eigvalsh(g) >= -1e-12)
