"""Equilibrium-check seam: synthetic series and yields in, verdicts out."""
import json

import numpy as np
import pytest

from analysis import equilibrium as eq
from analysis.bond_graph import Yields, yields
from base.config import PATCH_RADIUS
from base.initial_conditions import preassembled_rings
from analysis import bond_graph as bg
from tests.test_bond_graph_analysis import build_chain, monomers, ring, stack, write_traj

pytestmark = pytest.mark.fast


def ar1(phi, n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_stationary_series_passes_drifting_fails():
    rng = np.random.default_rng(1)
    assert eq.stationarity(rng.normal(size=2000)).passed
    assert eq.stationarity(ar1(0.9, 20_000, seed=2)).passed
    drift = rng.normal(size=2000) + np.linspace(0, 1.5, 2000)
    res = eq.stationarity(drift)
    assert not res.passed and res.values['second_half_mean'] > res.values['first_half_mean']
    assert res.values['drift_slope'] > 0


def test_autocorrelation_time_recovers_ar1_value():
    phi = 0.9
    tau = eq.integrated_autocorrelation_time(ar1(phi, 200_000, seed=3))
    assert tau == pytest.approx((1 + phi) / (1 - phi), rel=0.25)
    assert eq.integrated_autocorrelation_time(np.random.default_rng(4).normal(size=50_000)) \
        == pytest.approx(1.0, abs=0.3)


def fake_yields(py, err=0.01):
    labels = list(py)
    return Yields(dict(py), {k: err for k in labels}, dict(py), {k: err for k in labels}, 100)


def test_agreement_verdicts_on_synthetic_distributions():
    a = fake_yields({'monomer': 0.30, 'ring_3': 0.50, 'ring_4': 0.20})
    b = fake_yields({'monomer': 0.31, 'ring_3': 0.49, 'ring_4': 0.20})
    c = fake_yields({'monomer': 0.60, 'ring_3': 0.10, 'ring_4': 0.30})
    assert eq.forward_backward(a, b).passed
    assert not eq.forward_backward(a, c).passed
    assert eq.seed_agreement([a, b, a]).passed
    assert not eq.seed_agreement([a, b, c]).passed


def test_bond_distance_sensitivity(tmp_path):
    theta = np.pi / 2
    frames = [stack(ring(4), monomers([(10, 10), (14, 10)]))]
    p = write_traj(tmp_path / 'a.h5', frames * 5, theta)
    assert eq.bond_distance_sensitivity(p, tol=0.05).passed
    # a ring whose bonds sit just inside the patch radius flips when it shrinks
    c, o = ring(4)
    c = c * (1 + 0.9 * PATCH_RADIUS / 2)          # stretch bonds toward the cutoff
    p2 = write_traj(tmp_path / 'b.h5', [(c, o)] * 5, theta)
    assert not eq.bond_distance_sensitivity(p2, factors=(0.2, 1.0), tol=0.05).passed


def test_report_round_trips_through_saved_file(tmp_path):
    rep = eq.EquilibriumReport(meta={'a': 1})
    rep.add(eq.stationarity(np.random.default_rng(0).normal(size=500)))
    rep.add(eq.agreement([fake_yields({'monomer': 1.0}), fake_yields({'monomer': 1.0})],
                         name='seed_agreement'))
    path = rep.save(tmp_path / 'r.equilibrium.json')
    back = eq.EquilibriumReport.load(path)
    assert back.meta == rep.meta and back.passed == rep.passed
    assert back.checks.keys() == rep.checks.keys()
    for k in rep.checks:
        assert back.checks[k].passed == rep.checks[k].passed
        assert back.checks[k].values == json.loads(json.dumps(rep.checks[k].values))


def test_run_checks_saves_beside_trajectory(tmp_path):
    theta = np.pi / 2
    frames = [stack(ring(4), monomers([(10, 10)]))] * 40
    p = write_traj(tmp_path / 'run.h5', frames, theta)
    rep = eq.run_checks(p, other_seeds=[write_traj(tmp_path / 's2.h5', frames, theta)],
                        preassembled=write_traj(tmp_path / 'pre.h5', frames, theta))
    assert eq.report_path(p).exists()
    assert {'forward_backward', 'seed_agreement', 'bond_distance_sensitivity',
            'stationarity:potential_energy'} <= set(rep.checks)
    assert eq.EquilibriumReport.load(eq.report_path(p)).passed == rep.passed


@pytest.mark.parametrize('theta,n', [(np.pi / 3, 3), (np.pi / 2, 4)])
def test_preassembled_start_is_rings_plus_monomers(theta, n):
    L = 40.0
    pos, ori = preassembled_rings(2 * n + 2, L, theta, seed=0)
    s = bg.classify_frame(pos, ori, L, theta)
    assert s.particle_counts[f'ring_{n}'] == 2 * n
    assert s.particle_counts['monomer'] == 2


def test_initial_condition_option_in_both_engines(tmp_path):
    from base.trajectory import open_trajectory
    from simulation.src.run import run_simulation
    theta = np.pi / 2
    common = dict(opening_angle=theta, E_AB=2.0, box_area=900.0, sample_interval=1,
                  seed=1, initial_condition='preassembled')
    run_simulation('canonical', tmp_path / 'n.h5', N=8, n_equil=0, n_prod=2, dt=1e-4,
                   checkpoint_interval=2, **common)
    run_simulation('grand_canonical', tmp_path / 'g.h5', mu=-30.0, N_init=8, n_equil=0,
                   n_prod=1, md_steps=1, md_dt=1e-4, **common)
    for f in ('n.h5', 'g.h5'):
        with open_trajectory(tmp_path / f) as r:
            fr = r.frames('production')[0]
            s = bg.classify_frame(fr.positions, fr.orientations, r.attrs['box_size'], theta)
            assert s.particle_counts.get('ring_4') == 8, f
            assert r.attrs.get('initial_condition', 'preassembled') == 'preassembled'
    with pytest.raises(ValueError, match='initial_condition'):
        run_simulation('canonical', tmp_path / 'x.h5', N=4, n_equil=0, n_prod=1, dt=1e-4,
                       **{**common, 'initial_condition': 'bogus'})
