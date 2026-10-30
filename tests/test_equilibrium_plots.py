"""Equilibrium-check figures from synthetic files."""
import struct

import numpy as np
import pytest
from matplotlib.colors import to_rgba

from analysis import equilibrium_plots as ep
from analysis import structure_plots as sp
from analysis.bond_graph import yields
from tests.test_bond_graph_analysis import monomers, ring, stack, write_traj

pytestmark = pytest.mark.fast
THETA = np.pi / 2


def make(path, n_rings, n_frames=40):
    """Frames whose ring count switches half way (nonstationary) or not."""
    frames = []
    for i in range(n_frames):
        k = n_rings if isinstance(n_rings, int) else n_rings[0 if i < n_frames // 2 else 1]
        parts = [ring(4, (6 + 8 * j, 6)) for j in range(k)] + [monomers([(5, 30), (30, 5)])]
        frames.append(stack(*parts))
    return write_traj(path, frames, THETA)


def test_timeseries_figure_annotates_stationarity_verdict(tmp_path):
    steady = make(tmp_path / 'steady.h5', 2)
    drift = make(tmp_path / 'drift.h5', (1, 3))
    fig = ep.timeseries_figure(steady, tmp_path)
    out = tmp_path / 'steady_timeseries.png'
    assert out.stat().st_size > 5_000
    with open(out, 'rb') as f:
        w, h = struct.unpack('>II', f.read(24)[16:24])
    assert w > 0 and h > 0
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any(t.startswith('stationary') for t in texts)
    assert 'cluster yields stationary' in texts
    fig2 = ep.timeseries_figure(drift)
    texts2 = [t.get_text() for ax in fig2.axes for t in ax.texts]
    assert any(t.startswith('NOT stationary') for t in texts2) or \
        any(t.startswith('drifting') for t in texts2)


def test_cluster_yield_lines_use_shared_colours(tmp_path):
    fig = ep.timeseries_figure(make(tmp_path / 'a.h5', 2))
    colours = {to_rgba(l.get_color()) for l in fig.axes[-1].lines}
    assert to_rgba(sp.structure_color('ring_4')) in colours


def test_comparison_overlays_two_runs_with_shared_colours(tmp_path):
    a, b = make(tmp_path / 'a.h5', 2), make(tmp_path / 'b.h5', 3)
    fig = ep.comparison_figure({'dispersed': a, 'pre-assembled': yields(b)}, tmp_path)
    assert (tmp_path / 'comparison.png').stat().st_size > 5_000
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert len(ax.patches) == 2 * len(labels)
    ring_idx = labels.index('ring_4')
    for bar in (ax.patches[ring_idx], ax.patches[len(labels) + ring_idx]):
        assert np.allclose(bar.get_facecolor(), to_rgba(sp.structure_color('ring_4')))
    assert {t.get_text() for t in ax.get_legend().get_texts()} == {'dispersed', 'pre-assembled'}
    with pytest.raises(ValueError):
        ep.comparison_figure({'only': a})
