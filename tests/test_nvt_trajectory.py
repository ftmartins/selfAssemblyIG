"""Trajectory-file seam, canonical engine: tiny NVT runs through the driver and
assertions on the HDF5 file they leave behind."""
import numpy as np
import pytest

from base.jax_setup import configure_jax

configure_jax()

from base.initial_conditions import rsa_2d  # noqa: E402
from base.params import generate_parameter_grid  # noqa: E402
from base import trajectory  # noqa: E402
from base.trajectory import open_trajectory  # noqa: E402
from simulation.src.canonical_md import run_many_particle_md  # noqa: E402

pytestmark = pytest.mark.fast

N, BOX = 6, 12.0
EQ, PROD, SI, CK = 20, 40, 10, 20


def run(path, resume=False, **kw):
    import jax
    params = generate_parameter_grid(np.deg2rad([90.0]), np.array([2.0]))[0]
    pos, ori = rsa_2d(N, BOX, seed=1, centered=False)
    args = dict(sample_interval=SI, equilibration_steps=EQ,
                equilibration_sample_interval=SI, checkpoint_interval=CK,
                trajectory_path=path, resume=resume, seed=7)
    args.update(kw)
    return run_many_particle_md(N, BOX, params, PROD, 1.0, jax.random.PRNGKey(7), 1e-3,
                                pos, ori, **args)


@pytest.fixture(scope='module')
def full_run(tmp_path_factory):
    p = tmp_path_factory.mktemp('nvt') / 'full.h5'
    run(p)
    return p


def test_required_attributes_and_datasets(full_run):
    r = open_trajectory(full_run)
    for k in ['ensemble', 'seed', 'box_size', 'kT', 'dt', 'thermostat', 'N',
              'opening_angle', 'E_AA', 'E_BB', 'E_AB', 'bond_specificity',
              'num_equilibration_steps', 'num_production_steps', 'sample_interval',
              'software_version', 'alpha', 'patch_radius', 'morse_r_onset',
              'morse_r_cutoff', 'center_radius', 'precision', 'schema_version']:
        assert k in r.attrs, k
    assert r.attrs['ensemble'] == 'canonical' and r.attrs['N'] == N
    assert r.attrs['box_size'] == BOX
    for ph in ('equilibration', 'production'):
        n = r.n_frames(ph)
        for s in ('step', 'N', 'potential_energy', 'kinetic_energy'):
            assert len(r.series(ph, s)) == n
    assert r.n_frames('equilibration') == EQ // SI + 1
    assert r.n_frames('production') == PROD // SI + 1
    assert list(r.series('production', 'step')) == list(range(0, PROD + 1, SI))


def test_frames_are_finite_in_box_with_constant_n(full_run):
    r = open_trajectory(full_run)
    for ph in ('equilibration', 'production'):
        for f in r.frames(ph):
            assert f.N == N and f.positions.shape == (N, 2)
            assert np.all(np.isfinite(f.positions)) and np.all(np.isfinite(f.orientations))
            assert f.positions.min() >= 0.0 and f.positions.max() <= BOX
            assert np.isfinite(f.potential_energy) and f.kinetic_energy >= 0


def test_final_state_is_restartable_and_marked_done(full_run):
    phase, step, leaves, _ = open_trajectory(full_run).read_restart()
    assert phase == 'done' and step == PROD and len(leaves) > 0


def test_precision_option_sets_dtype_energies_stay_double(tmp_path):
    p = tmp_path / 'f32.h5'
    run(p, precision='float32')
    r = open_trajectory(p)
    f = r.frames('production')[0]
    assert f.positions.dtype == np.float32 and f.orientations.dtype == np.float32
    assert r.series('production', 'potential_energy').dtype == np.float64


def test_killed_and_resumed_run_matches_uninterrupted(tmp_path, full_run, monkeypatch):
    p = tmp_path / 'killed.h5'
    calls = {'n': 0}
    real = trajectory.TrajectoryWriter.write_restart

    def dying(self, *a, **k):
        calls['n'] += 1
        if calls['n'] == 3:          # frames of this chunk are already on disk
            raise KeyboardInterrupt
        return real(self, *a, **k)

    monkeypatch.setattr(trajectory.TrajectoryWriter, 'write_restart', dying)
    with pytest.raises(KeyboardInterrupt):
        run(p)
    monkeypatch.undo()
    assert open_trajectory(p).n_frames('production') >= 1      # readable partial file

    run(p, resume=True)
    a, b = open_trajectory(p), open_trajectory(full_run)
    for ph in ('equilibration', 'production'):
        assert a.n_frames(ph) == b.n_frames(ph)
        assert list(a.series(ph, 'step')) == list(b.series(ph, 'step'))   # no duplicates
    np.testing.assert_allclose(a.frames('production')[-1].positions,
                               b.frames('production')[-1].positions, atol=1e-8)
    assert a.read_restart()[0] == 'done'


def test_resume_of_finished_run_adds_nothing(tmp_path):
    p = tmp_path / 'again.h5'
    run(p)
    run(p, resume=True)
    r = open_trajectory(p)
    assert r.n_frames('production') == PROD // SI + 1


# ── the canonical runners ────────────────────────────────────────────────

def test_cluster_runner_writes_one_file_per_realization_and_resumes(tmp_path, monkeypatch):
    import sys
    from simulation.runners import run_yield_cluster as ryc
    argv = ['run_yield_cluster', '0', '--output-dir', str(tmp_path), '--num-particles', '4',
            '--equilibration-steps', '20', '--num-steps', '40', '--sample-interval', '10',
            '--equilibration-sample-interval', '10', '--checkpoint-interval', '20',
            '--realizations', '2']
    monkeypatch.setattr(sys, 'argv', argv)
    ryc.main()
    files = sorted(tmp_path.glob('*_r*.h5'))
    assert len(files) == 2 and not list(tmp_path.glob('*_trajectory.npz'))
    assert not list((tmp_path).glob('checkpoints'))
    with open_trajectory(files[0]) as r:
        assert r.attrs['ensemble'] == 'canonical' and r.attrs['precision'] == 'float32'
        assert r.n_frames('production') == 5 and r.n_frames('equilibration') == 3
    ryc.main()                      # rerun: finished files untouched, no duplicates
    assert open_trajectory(files[0]).n_frames('production') == 5


def test_design_runner_streams_frames_and_resumes(tmp_path):
    import jax
    from simulation.runners.run_yield import run_nvt
    from simulation.src.canonical_md import random_IC
    v = jax.numpy.array([0.0, 1.5, 3.0, 1.0, 1.0])     # ref angle, theta, E_AA, E_AB, E_BB
    body = random_IC(v, jax.random.PRNGKey(0), 4, BOX)
    kw = dict(chunk_size=3, frame_stride=1, equilibration_frame_stride=0)
    p, q = tmp_path / 'a.h5', tmp_path / 'b.h5'
    run_nvt(v, body, BOX, 3, 6, 5, p, **kw)
    with open_trajectory(p) as r:
        assert r.n_frames('equilibration') == 0 and r.n_frames('production') == 1 + 6
        assert r.attrs['E_AB'] == 1.0 and r.attrs['opening_angle'] == 1.5
    # resuming a finished file adds nothing
    run_nvt(v, body, BOX, 3, 6, 5, p, resume=True, **kw)
    assert open_trajectory(p).n_frames('production') == 7
