"""Plot seam: synthetic trajectory files in, PNGs out."""
import os
import struct

import numpy as np
import pytest

from analysis import structure_plots as sp
from analysis.bond_graph import yields
from tests.test_bond_graph_analysis import BOX, monomers, ring, stack, write_traj

pytestmark = pytest.mark.fast
THETA = np.pi / 2


def png_size(path):
    with open(path, 'rb') as f:
        head = f.read(24)
    assert head[:8] == b'\x89PNG\r\n\x1a\n'
    return struct.unpack('>II', head[16:24])


@pytest.fixture(scope='module')
def traj(tmp_path_factory):
    frames = [stack(ring(4, (10, 10)), ring(4, (25, 25)), monomers([(5, 30), (30, 5)]))] * 6
    return write_traj(tmp_path_factory.mktemp('p') / 'a.h5', frames, THETA)


def test_snapshot_png_has_expected_dimensions(traj, tmp_path):
    sp.snapshot(traj, frame=2, out_dir=tmp_path, figsize=(5, 4))
    out = tmp_path / 'a_snapshot_production_f00002.png'
    assert png_size(out) == (500, 400)
    assert out.stat().st_size > 5_000


def test_histograms_and_grid_write_nonempty_pngs(traj, tmp_path):
    sp.yield_histogram(traj, 'particle', tmp_path, figsize=(6, 4))
    sp.yield_histogram(traj, 'cluster', tmp_path)
    sp.yield_grid([traj, ('second', traj), yields(traj)], 'particle', tmp_path, ncols=2,
                  panel=(3, 2))
    assert png_size(tmp_path / 'a_yield_particle.png') == (600, 400)
    assert (tmp_path / 'a_yield_cluster.png').stat().st_size > 5_000
    assert png_size(tmp_path / 'yield_grid_particle.png') == (600, 400)


def test_same_structure_has_same_colour_in_snapshot_and_histogram(traj):
    fig = sp.snapshot(traj)
    ax = fig.axes[0]
    faces = {tuple(np.round(c, 3)) for coll in ax.collections for c in coll.get_facecolor()}
    from matplotlib.colors import to_rgba
    for label in ('ring_4', 'monomer'):
        assert tuple(np.round(to_rgba(sp.structure_color(label)), 3)) in faces
    hist = sp.yield_histogram(traj, 'particle')
    bars = {t.get_text(): b.get_facecolor() for t, b in
            zip(hist.axes[0].get_xticklabels(), hist.axes[0].patches)}
    for label, rgba in bars.items():
        assert np.allclose(rgba, to_rgba(sp.structure_color(label)))
    legend = fig.legends[0]
    assert {t.get_text() for t in legend.get_texts()} >= {'ring_4', 'monomer',
                                                          'patch A', 'patch B'}


def test_colour_map_is_distinct_per_ring_size_and_neutral_for_chains():
    cols = [sp.structure_color(f'ring_{n}') for n in range(3, 13)] + [sp.structure_color('ring_13+')]
    assert len(set(cols)) == len(cols)
    assert sp.structure_color('chain_2') == sp.structure_color('chain_7')
    assert sp.structure_color('chain_2') not in cols


def test_runs_without_display(traj, tmp_path):
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k not in ('DISPLAY', 'MPLBACKEND')}
    r = subprocess.run([sys.executable, '-m', 'analysis.structure_plots', 'snapshot',
                        str(traj), '--outdir', str(tmp_path)], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / 'a_snapshot_production_f00005.png').exists()
    r = subprocess.run([sys.executable, '-m', 'analysis.structure_plots', 'grid',
                        str(traj), str(traj), '--outdir', str(tmp_path)], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / 'yield_grid_cluster.png').exists()


def test_unwrap_keeps_boundary_ring_contiguous():
    from analysis.bond_graph import classify_frame
    c, o = ring(4, (0.2, 0.2))
    pos = c % BOX
    s = classify_frame(pos, o, BOX, THETA)
    assert s.clusters[0].kind == 'ring'
    un = sp.unwrap_clusters(pos, BOX, s)
    d = np.hypot(*(un[:, None] - un[None]).transpose(2, 0, 1))
    assert d.max() < 4.0                      # diagonal of a unit square ring, not ~BOX
