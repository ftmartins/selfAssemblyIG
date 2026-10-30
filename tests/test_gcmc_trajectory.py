"""Trajectory-file seam, grand-canonical engine and the unified run entry point."""
import os
import sys

import numpy as np
import pytest

from base.jax_setup import configure_jax

configure_jax()

from base.trajectory import open_trajectory  # noqa: E402
from simulation.src.run import run_simulation  # noqa: E402

pytestmark = pytest.mark.fast

AREA, EQ, PROD, SI = 36.0, 2, 8, 2
GC = dict(opening_angle=np.deg2rad(90.0), E_AB=2.0, box_area=AREA, mu=-1.0, N_init=4,
          n_equil=EQ, n_prod=PROD, sample_interval=SI, md_steps=5, seed=3)
NVT = dict(opening_angle=np.deg2rad(90.0), E_AB=2.0, box_area=AREA, N=4,
           n_equil=20, n_prod=40, sample_interval=10, seed=3, dt=1e-3,
           checkpoint_interval=20)


@pytest.fixture(scope='module')
def gc_file(tmp_path_factory):
    p = tmp_path_factory.mktemp('gc') / 'gc.h5'
    run_simulation('grand_canonical', p, **GC)
    return p


@pytest.fixture(scope='module')
def nvt_file(tmp_path_factory):
    p = tmp_path_factory.mktemp('nvt') / 'nvt.h5'
    run_simulation('canonical', p, **NVT)
    return p


def test_gcmc_frames_have_correct_variable_n(gc_file):
    L = np.sqrt(AREA)
    with open_trajectory(gc_file) as r:
        assert r.attrs['ensemble'] == 'grand_canonical'
        assert r.n_frames('production') == PROD // SI + 1
        assert list(r.series('production', 'step')) == list(range(0, PROD + 1, SI))
        Ns = r.series('production', 'N')
        frames = r.frames('production')
        assert [f.N for f in frames] == list(Ns)
        for f in frames:
            assert f.positions.shape == (f.N, 2) and f.orientations.shape == (f.N,)
            assert np.all(np.isfinite(f.positions))
            assert f.positions.min() >= 0.0 and f.positions.max() <= L
        # a strided / ranged read returns the right particles when N varies
        sub = r.frames('production', 1, None, 2)
        assert [f.N for f in sub] == list(Ns[1::2])
        np.testing.assert_array_equal(sub[0].positions, frames[1].positions)


def test_gcmc_n_actually_varies(tmp_path):
    # guard that the variable-N assertions are meaningful
    p = tmp_path / 'x.h5'
    run_simulation('grand_canonical', p, **{**GC, 'mu': 0.0, 'n_prod': 20,
                                            'sample_interval': 1})
    with open_trajectory(p) as r:
        Ns = r.series('production', 'N')
        assert len(set(Ns)) > 1
        assert [f.N for f in r.frames('production')] == list(Ns)


def test_gcmc_acceptance_rates_in_attributes(gc_file):
    with open_trajectory(gc_file) as r:
        for k in ('acc_insertion', 'acc_deletion', 'acc_displacement'):
            assert 0.0 <= r.attrs[k] <= 1.0, k
        for k in ('mu', 'box_size', 'seed', 'kT', 'E_AB', 'opening_angle',
                  'bond_specificity', 'f_disp', 'md_steps'):
            assert k in r.attrs, k
        assert r.read_restart()[0] == 'done'


def _schema(p):
    import h5py
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and not name.startswith('restart'):
            out[name] = (obj.dtype, obj.compression, obj.chunks is not None)

    with h5py.File(p, 'r') as f:
        f.visititems(visit)
        return set(f), out


def test_schema_equal_between_ensembles(gc_file, nvt_file):
    (ga, sa), (gb, sb) = _schema(gc_file), _schema(nvt_file)
    assert ga == gb and sa == sb
    with open_trajectory(gc_file) as a, open_trajectory(nvt_file) as b:
        common = {'ensemble', 'seed', 'box_size', 'kT', 'opening_angle', 'E_AA', 'E_BB',
                  'E_AB', 'bond_specificity', 'precision', 'schema_version', 'alpha',
                  'patch_radius', 'morse_r_onset', 'morse_r_cutoff', 'center_radius',
                  'software_version', 'sample_interval', 'num_equilibration_steps',
                  'num_production_steps'}
        assert common <= set(a.attrs) and common <= set(b.attrs)


def test_equilibration_frames_optional_and_precision(tmp_path):
    p = tmp_path / 'e.h5'
    run_simulation('grand_canonical', p, **GC, equilibration_sample_interval=1,
                   precision='float32')
    with open_trajectory(p) as r:
        assert r.n_frames('equilibration') == EQ + 1
        assert r.frames('production')[0].positions.dtype == np.float32


def test_killed_gcmc_resumes_without_duplicates(tmp_path, gc_file, monkeypatch):
    from base import trajectory
    p = tmp_path / 'k.h5'
    calls = {'n': 0}
    real = trajectory.TrajectoryWriter.write_restart

    def dying(self, *a, **k):
        calls['n'] += 1
        if calls['n'] == 4:
            raise KeyboardInterrupt
        return real(self, *a, **k)

    monkeypatch.setattr(trajectory.TrajectoryWriter, 'write_restart', dying)
    with pytest.raises(KeyboardInterrupt):
        run_simulation('grand_canonical', p, **GC)
    monkeypatch.undo()
    run_simulation('grand_canonical', p, **GC, resume=True)
    with open_trajectory(p) as a, open_trajectory(gc_file) as b:
        assert list(a.series('production', 'step')) == list(b.series('production', 'step'))
        # RNG state restored, so the resumed trajectory is the uninterrupted one
        assert list(a.series('production', 'N')) == list(b.series('production', 'N'))
        np.testing.assert_allclose(a.frames('production')[-1].positions,
                                   b.frames('production')[-1].positions, atol=1e-6)


def test_nvt_through_unified_entry(nvt_file):
    with open_trajectory(nvt_file) as r:
        assert r.attrs['ensemble'] == 'canonical' and r.attrs['N'] == 4
        assert r.n_frames('production') == 5
        assert set(r.series('production', 'N')) == {4}


def test_unknown_ensemble_rejected(tmp_path):
    with pytest.raises(ValueError, match='ensemble'):
        run_simulation('microcanonical', tmp_path / 'x.h5', **GC)


def test_cli_ensemble_option(tmp_path):
    from simulation.runners import run
    cases = [('canonical', ['--N', '4', '--dt', '1e-3', '--n_equil', '10',
                            '--n_prod', '20', '--sample_interval', '10',
                            '--checkpoint_interval', '10']),
             ('grand_canonical', ['--mu', '-1', '--N_init', '3', '--n_equil', '1',
                                  '--n_prod', '2', '--sample_interval', '1',
                                  '--md_steps', '3'])]
    for ens, extra in cases:
        out = tmp_path / f'{ens}.h5'
        run.main(['--ensemble', ens, '--opening_angle', '1.5', '--E_AB', '2.0',
                  '--box_area', '36', '--output', str(out), '--seed', '1'] + extra)
        with open_trajectory(out) as r:
            assert r.attrs['ensemble'] == ens and r.n_frames('production') >= 3


def test_gcmc_runner_wrapper(tmp_path, monkeypatch):
    from simulation.runners import run_gcmc
    out = tmp_path / 'w.h5'
    monkeypatch.setattr(sys, 'argv', [
        'run_gcmc', '--opening_angle', '1.5', '--E_AB', '2', '--mu', '-1', '--box_area', '36',
        '--n_equil', '1', '--n_prod', '2', '--snapshot_interval', '1', '--md_steps', '3',
        '--N_init', '3', '--output', str(out), '--quiet', '--no_checkpoint'])
    monkeypatch.setattr(run_gcmc, 'make_gcmc_run_dir', lambda **k: str(tmp_path / 'rd'))
    run_gcmc.main()
    with open_trajectory(out) as r:
        assert r.attrs['ensemble'] == 'grand_canonical'
