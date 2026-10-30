"""GIF seam: synthetic trajectory file in, GIF out."""
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from analysis import movie
from analysis import structure_plots as sp
from tests.test_bond_graph_analysis import BOX, monomers, ring, stack, write_traj

pytestmark = pytest.mark.fast
THETA = np.pi / 2


def frames_moving(n, origin=(10, 10)):
    """A ring drifting across the periodic boundary plus a monomer."""
    out = []
    for i in range(n):
        out.append(stack(ring(4, (origin[0] + 6 * i, origin[1])), monomers([(5, 30)])))
    return out


@pytest.fixture(scope='module')
def traj(tmp_path_factory):
    return write_traj(tmp_path_factory.mktemp('m') / 'a.h5', frames_moving(10), THETA)


def test_gif_frame_count_follows_range_and_stride(traj, tmp_path):
    out = tmp_path / 'a.gif'
    n = movie.make_gif(traj, out, start=1, stop=9, stride=2, fps=5, figsize=2)
    assert n == 4
    with Image.open(out) as im:
        assert im.n_frames == 4 and im.size == (160, 160)
        assert im.info['duration'] == 200


def test_patches_distinct_and_cluster_colours_match_snapshot(traj, tmp_path):
    out = tmp_path / 'c.gif'
    movie.make_gif(traj, out, start=0, stop=1, figsize=3, dpi=100)
    with Image.open(out) as im:
        rgb = np.asarray(im.convert('RGB'))
    colours = {tuple(c) for c in rgb.reshape(-1, 3)}

    def near(hexcol):
        t = np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)])
        return any(np.abs(np.array(c) - t).max() <= 24 for c in colours)

    assert near(sp.PATCH_A_COLOR) and near(sp.PATCH_B_COLOR)
    assert sp.PATCH_A_COLOR != sp.PATCH_B_COLOR
    assert near(sp.structure_color('ring_4')) and near(sp.structure_color('monomer'))
    assert movie.make_gif(traj, tmp_path / 'p.gif', stop=1, figsize=3, dpi=100,
                          color_clusters=False) == 1


def test_unwrapped_mode_draws_boundary_ring_contiguous(tmp_path):
    c, o = ring(4, (0.2, 0.2))
    p = write_traj(tmp_path / 'b.h5', [(c, o)] * 2, THETA)
    from base.trajectory import open_trajectory
    from analysis.bond_graph import classify_frame
    with open_trajectory(p) as r:
        fr = r.frames('production')[0]
        s = classify_frame(fr.positions, fr.orientations, BOX, THETA)
    un = sp.unwrap_clusters(fr.positions, BOX, s)
    assert np.ptp(un[:, 0]) < 4 and np.ptp(un[:, 1]) < 4
    assert movie.make_gif(p, tmp_path / 'u.gif', unwrap=True, figsize=2) >= 1


def test_missing_pillow_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, 'PIL', None)
    with pytest.raises(ImportError, match='pip install pillow'):
        movie._pillow()


def test_runs_without_display(traj, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ('DISPLAY', 'MPLBACKEND')}
    r = subprocess.run([sys.executable, '-m', 'analysis.movie', str(traj),
                        str(tmp_path / 'cli.gif'), '--stride', '3', '--figsize', '2'],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with Image.open(tmp_path / 'cli.gif') as im:
        assert im.n_frames == 4
