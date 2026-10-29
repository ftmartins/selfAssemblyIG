"""Trajectory-file seam, store half: what the writer puts on disk is what the
reader returns, for fixed-N and variable-N frames, with ranges/strides, dtype
control, truncation-safe resume, and legacy files through the same reader."""
import numpy as np
import pytest

from base.trajectory import TrajectoryWriter, open_trajectory

pytestmark = pytest.mark.fast

ATTRS = dict(ensemble='canonical', seed=3, box_size=10.0, kT=1.0)


def frame(n, fill):
    return dict(positions=np.full((n, 2), float(fill)),
                orientations=np.full(n, float(fill) / 10),
                potential_energy=-float(fill), step=int(fill) * 10)


def write(path, ns, phase='production', **kw):
    with TrajectoryWriter(path, ATTRS, **kw) as w:
        for i, n in enumerate(ns):
            w.append_frame(phase=phase, **frame(n, i))
    return path


def test_roundtrip_attrs_and_frames(tmp_path):
    p = write(tmp_path / 'a.h5', [4, 4, 4])
    r = open_trajectory(p)
    assert r.attrs['ensemble'] == 'canonical' and r.attrs['seed'] == 3
    assert r.n_frames('production') == 3 and r.n_frames('equilibration') == 0
    f = r.frames('production')
    assert [x.step for x in f] == [0, 10, 20]
    assert f[2].positions.shape == (4, 2) and np.all(f[2].positions == 2.0)
    assert f[1].potential_energy == -1.0 and f[1].N == 4


def test_variable_n_frames_are_exact_not_padded(tmp_path):
    p = write(tmp_path / 'v.h5', [3, 5, 0, 2])
    r = open_trajectory(p)
    f = r.frames('production')
    assert [x.N for x in f] == [3, 5, 0, 2]
    assert [len(x.positions) for x in f] == [3, 5, 0, 2]
    assert np.all(f[1].positions == 1.0)
    assert list(r.series('production', 'N')) == [3, 5, 0, 2]


def test_ranges_and_strides(tmp_path):
    p = write(tmp_path / 'r.h5', [2] * 10)
    r = open_trajectory(p)
    assert [x.step for x in r.frames('production', 2, 5)] == [20, 30, 40]
    assert [x.step for x in r.frames('production', 1, None, 4)] == [10, 50, 90]
    assert [x.step for x in r.frames('production', step=3)] == [0, 30, 60, 90]


def test_precision_changes_coordinate_dtype_only(tmp_path):
    for prec, dt in [('float32', np.float32), ('float64', np.float64)]:
        p = write(tmp_path / f'{prec}.h5', [2, 2], precision=prec)
        f = open_trajectory(p).frames('production')[0]
        assert f.positions.dtype == dt and f.orientations.dtype == dt
        assert np.asarray(f.potential_energy).dtype == np.float64


def test_phases_are_separate(tmp_path):
    p = tmp_path / 'ph.h5'
    with TrajectoryWriter(p, ATTRS) as w:
        w.append_frame(phase='equilibration', **frame(2, 1))
        w.append_frame(phase='production', **frame(2, 2))
        w.append_frame(phase='production', **frame(2, 3))
    r = open_trajectory(p)
    assert r.n_frames('equilibration') == 1 and r.n_frames('production') == 2


def test_resume_truncates_frames_after_last_restart(tmp_path):
    p = tmp_path / 'res.h5'
    w = TrajectoryWriter(p, ATTRS)
    w.append_frame(phase='production', **frame(2, 0))
    w.append_frame(phase='production', **frame(2, 1))
    w.write_restart('production', 10, [np.arange(3.0), np.eye(2)])
    w.append_frame(phase='production', **frame(2, 2))   # written, then "killed"
    w.flush()
    # readable after a kill (no close)
    assert open_trajectory(p).n_frames('production') == 3
    del w
    w2 = TrajectoryWriter(p, ATTRS, resume=True)
    phase, pstep, leaves, _ = w2.restart
    assert (phase, pstep) == ('production', 10)
    np.testing.assert_array_equal(leaves[1], np.eye(2))
    w2.append_frame(phase='production', **frame(2, 2))
    w2.close()
    assert [x.step for x in open_trajectory(p).frames('production')] == [0, 10, 20]


def test_resume_without_restart_starts_empty(tmp_path):
    p = write(tmp_path / 'n.h5', [2, 2])
    w = TrajectoryWriter(p, ATTRS, resume=True)
    assert w.restart is None
    w.close()
    assert open_trajectory(p).n_frames('production') == 0


def test_legacy_cluster_npz(tmp_path):
    pos = np.zeros((4, 3, 2))
    pos[:3] = 1.0 + np.arange(3)[:, None, None]
    ori = np.zeros((4, 3))
    e = np.array([-1., -2., -3., 0.])
    p = tmp_path / 'old.npz'
    np.savez_compressed(p, box_size=12.0, alpha_deg=60.0, sample_interval=100,
                        r0_positions=pos, r0_orientations=ori, r0_energy=e)
    r = open_trajectory(p)
    assert r.attrs['box_size'] == 12.0
    assert r.n_frames('production') == 3          # unused zero rows dropped
    f = r.frames('production', 1, 3)
    assert f[0].N == 3 and f[0].potential_energy == -2.0 and f[0].positions[0, 0] == 2.0
    assert [x.step for x in f] == [100, 200]


def test_legacy_gcmc_npz_nan_padded(tmp_path):
    q = np.full((2, 4, 3), np.nan)
    q[0, :3] = 1.0
    q[1, :2] = 2.0
    p = tmp_path / 'run.npz'
    np.savez_compressed(p, snapshot_q=q, snapshot_N=np.array([3, 2]),
                        snapshot_sweeps=np.array([5, 10]), E_traj=np.array([0.1, 0.2]),
                        box_area=np.array([100.0]), mu=np.array([-4.0]))
    r = open_trajectory(p)
    f = r.frames('production')
    assert [x.N for x in f] == [3, 2] and [x.step for x in f] == [5, 10]
    assert f[1].positions.shape == (2, 2) and f[1].orientations.shape == (2,)
    assert r.attrs['box_size'] == pytest.approx(10.0)


def test_legacy_pickled_npy_md_trajectories(tmp_path):
    p = tmp_path / 'md.npy'
    np.save(p, {'MD_trajectories': {0: np.ones((3, 3)), 1: 2 * np.ones((3, 3))}},
            allow_pickle=True)
    f = open_trajectory(p).frames('production')
    assert len(f) == 2 and f[1].positions.shape == (3, 2) and f[1].orientations[0] == 2.0
