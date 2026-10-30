"""
Structure plots from trajectory files: snapshots coloured by cluster size and
particle/cluster yield histograms.

* One colourblind-safe map (:func:`structure_color`) is shared by every plot
  type: rings take one colour per size (3 .. 12, ``ring_13+``), open chains and
  monomers are neutral greys.
* Headless-safe: figures are built on ``matplotlib.figure.Figure`` with the Agg
  canvas, never ``pyplot``, so no display or backend is touched.  Every
  function returns the ``Figure`` (notebooks can show or save it) and, given
  ``out_dir``, also writes a PNG with a predictable name (``<stem>`` is the
  trajectory file name without extension):

  ``<stem>_snapshot_<phase>_f<frame:05d>.png``,
  ``<stem>_yield_particle.png``, ``<stem>_yield_cluster.png``,
  ``yield_grid_particle.png`` / ``yield_grid_cluster.png``.

Command line::

    python -m analysis.structure_plots snapshot FILE [--frame -1] [--outdir D]
    python -m analysis.structure_plots histogram FILE [--outdir D]
    python -m analysis.structure_plots grid FILE [FILE ...] [--outdir D]
"""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from analysis.bond_graph import classify_frame, structure_label, yields as compute_yields
from base.config import CENTER_RADIUS
from base.geometry import patch_positions
from base.trajectory import open_trajectory

# Paul Tol "muted" + two Okabe-Ito colours: ring sizes 3..11, 12, 13+
_RING_COLORS = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255',
                '#44AA99', '#999933', '#AA4499', '#E69F00', '#0072B2']
_NEUTRAL = {'monomer': '#DDDDDD', 'chain': '#999999'}
PATCH_A_COLOR, PATCH_B_COLOR = '#D55E00', '#0072B2'    # vermillion / blue (Okabe-Ito)
DPI = 100


def structure_color(label):
    """Colour of a structure label (``'monomer'``, ``'chain_k'``, ``'ring_n'``)."""
    if label == 'monomer':
        return _NEUTRAL['monomer']
    kind, _, size = label.partition('_')
    if kind == 'chain':
        return _NEUTRAL['chain']
    n = 13 if size.endswith('+') else int(size)
    return _RING_COLORS[min(n, 13) - 3]


def _label_order(label):
    if label == 'monomer':
        return (0, 0)
    kind, _, size = label.partition('_')
    return (1 if kind == 'chain' else 2, int(size.rstrip('+')))


def _finish(fig, out_dir, name):
    FigureCanvasAgg(fig)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / name, dpi=DPI)
    return fig


def _stem(source):
    return Path(source).stem if isinstance(source, (str, Path)) else 'trajectory'


def _open(source):
    return open_trajectory(source) if isinstance(source, (str, Path)) else source


def unwrap_clusters(positions, box_size, structures, link=2.5):
    """Centres moved by whole box vectors so every cluster is contiguous (a ring
    straddling the boundary is not torn apart).  Neighbours closer than ``link``
    under the minimum image are chained from each cluster's first member."""
    pos = np.asarray(positions, float)
    out = pos.copy()
    L = float(box_size)
    for c in structures.clusters:
        if c.size == 1:
            continue
        seen = {int(c.members[0])}
        queue = [int(c.members[0])]
        while queue:
            i = queue.pop()
            for j in c.members:
                j = int(j)
                if j in seen:
                    continue
                d = pos[j] - pos[i]
                d -= L * np.round(d / L)
                if np.hypot(*d) < link:
                    out[j] = out[i] + d
                    seen.add(j)
                    queue.append(j)
    return out


def draw_frame(ax, positions, orientations, box_size, opening_angle, structures=None,
               color_clusters=True, unwrap=False, patches=True):
    """Draw one frame on ``ax``: cores (coloured by structure when
    ``structures`` is given), A and B patch markers, periodic images for
    particles near the edge (wrapped mode)."""
    L = float(box_size)
    pos = np.asarray(positions, float).reshape(-1, 2)
    ori = np.asarray(orientations, float).reshape(-1)
    if unwrap and structures is not None:
        pos = unwrap_clusters(pos, L, structures)
    else:
        pos = np.mod(pos, L)
    colors = ['#CCCCCC'] * len(pos)
    if structures is not None and color_clusters:
        colors = [structure_color(structures.clusters[k].label)
                  for k in structures.cluster_id]
    shifts = [(0.0, 0.0)] if unwrap else [(sx * L, sy * L) for sx in (-1, 0, 1)
                                          for sy in (-1, 0, 1)]
    cores, core_colors, pa_pts, pb_pts = [], [], [], []
    pA, pB = patch_positions(pos, ori, opening_angle, CENTER_RADIUS)
    for sx, sy in shifts:
        shift = np.array([sx, sy])
        for i in range(len(pos)):
            c = pos[i] + shift
            if not unwrap and not (-CENTER_RADIUS <= c[0] <= L + CENTER_RADIUS
                                   and -CENTER_RADIUS <= c[1] <= L + CENTER_RADIUS):
                continue
            cores.append(Circle(c, CENTER_RADIUS))
            core_colors.append(colors[i])
            pa_pts.append(pA[i] + shift)
            pb_pts.append(pB[i] + shift)
    ax.add_collection(PatchCollection(cores, facecolors=core_colors,
                                      edgecolors='#333333', linewidths=0.6))
    if patches and cores:
        for pts, col in ((pa_pts, PATCH_A_COLOR), (pb_pts, PATCH_B_COLOR)):
            pts = np.array(pts)
            ax.scatter(pts[:, 0], pts[:, 1], s=14, c=col, zorder=3, linewidths=0)
    if unwrap:
        lo, hi = pos.min(axis=0) - 2, pos.max(axis=0) + 2
        ax.set_xlim(min(lo[0], 0), max(hi[0], L))
        ax.set_ylim(min(lo[1], 0), max(hi[1], L))
        ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], color='#888888', lw=0.8, ls=':')
    else:
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def structure_legend_handles(labels):
    labels = sorted(set(labels), key=_label_order)
    handles = [Patch(facecolor=structure_color(k), edgecolor='#333333', label=k)
               for k in labels if not k.startswith('chain_')]
    if any(k.startswith('chain_') for k in labels):
        handles.append(Patch(facecolor=_NEUTRAL['chain'], edgecolor='#333333',
                             label='open chain'))
    handles += [Line2D([], [], marker='o', ls='', color=PATCH_A_COLOR, label='patch A'),
                Line2D([], [], marker='o', ls='', color=PATCH_B_COLOR, label='patch B')]
    return handles


def snapshot(source, frame=-1, phase='production', out_dir=None, bond_distance=None,
             figsize=(6.4, 5.6), unwrap=False):
    """Snapshot of one frame, particles coloured by the size of their cluster."""
    reader = _open(source)
    n = reader.n_frames(phase)
    idx = frame if frame >= 0 else n + frame
    if not 0 <= idx < n:
        raise IndexError(f'frame {frame} out of range for {n} {phase} frames')
    fr = reader.frames(phase, idx, idx + 1)[0]
    L, theta = float(reader.attrs['box_size']), float(reader.attrs['opening_angle'])
    s = classify_frame(fr.positions, fr.orientations, L, theta, bond_distance)
    fig = Figure(figsize=figsize)
    ax = fig.add_axes([0.02, 0.02, 0.72, 0.94])
    draw_frame(ax, fr.positions, fr.orientations, L, theta, s, unwrap=unwrap)
    ax.set_title(f'{phase} frame {idx}  (N = {fr.N})', fontsize=10)
    fig.legend(handles=structure_legend_handles(c.label for c in s.clusters),
               loc='center right', frameon=False, fontsize=9)
    return _finish(fig, out_dir, f'{_stem(source)}_snapshot_{phase}_f{idx:05d}.png')


def _bars(ax, y, labels, kind):
    x = np.arange(len(labels))
    mean = [getattr(y, f'{kind}_yield').get(k, 0.0) for k in labels]
    err = [np.nan_to_num(getattr(y, f'{kind}_yield_err').get(k, 0.0)) for k in labels]
    ax.bar(x, mean, yerr=err, color=[structure_color(k) for k in labels],
           edgecolor='#333333', linewidth=0.6, capsize=3, error_kw=dict(lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel(f'{kind} yield')


def _all_labels(ys, kind):
    return sorted({k for y in ys for k in getattr(y, f'{kind}_yield')}, key=_label_order)


def yield_histogram(source, kind='particle', out_dir=None, figsize=(6.4, 4.0), **yield_kw):
    """Particle- or cluster-yield histogram by structure with error bars.
    ``source`` is a trajectory (path or reader) or a precomputed ``Yields``."""
    y = source if hasattr(source, 'particle_yield') else compute_yields(
        source, **yield_kw)
    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    _bars(ax, y, _all_labels([y], kind), kind)
    ax.set_title(f'{kind} yield  ({y.n_frames} frames, {y.n_seeds} seed'
                 f'{"s" if y.n_seeds != 1 else ""})', fontsize=10)
    fig.tight_layout()
    return _finish(fig, out_dir, f'{_stem(source)}_yield_{kind}.png')


def yield_grid(points, kind='particle', out_dir=None, ncols=3, panel=(3.2, 2.6),
               **yield_kw):
    """Grid of yield histograms, one panel per parameter point.  ``points`` is a
    list of trajectory paths / readers / ``Yields`` or ``(title, source)`` pairs;
    all panels share axes and colours."""
    items = []
    for p in points:
        title, src = p if isinstance(p, tuple) else (_stem(p), p)
        items.append((title, src if hasattr(src, 'particle_yield')
                      else compute_yields(src, **yield_kw)))
    labels = _all_labels([y for _, y in items], kind)
    ncols = min(ncols, len(items))
    nrows = -(-len(items) // ncols)
    fig = Figure(figsize=(panel[0] * ncols, panel[1] * nrows))
    axes = fig.subplots(nrows, ncols, sharey=True, squeeze=False)
    for ax in axes.flat[len(items):]:
        ax.set_visible(False)
    for ax, (title, y) in zip(axes.flat, items):
        _bars(ax, y, labels, kind)
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    return _finish(fig, out_dir, f'yield_grid_{kind}.png')


# ── command line ───────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description='Structure plots from trajectory files')
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('snapshot', 'histogram', 'grid'):
        s = sub.add_parser(name)
        s.add_argument('files', nargs='+' if name == 'grid' else 1)
        s.add_argument('--outdir', default='.')
        s.add_argument('--bond_distance', type=float, default=None)
    sub.choices['snapshot'].add_argument('--frame', type=int, default=-1)
    sub.choices['snapshot'].add_argument('--phase', default='production')
    sub.choices['snapshot'].add_argument('--unwrap', action='store_true')
    for name in ('histogram', 'grid'):
        sub.choices[name].add_argument('--kind', choices=['particle', 'cluster', 'both'],
                                       default='both')
    a = p.parse_args(argv)
    if a.cmd == 'snapshot':
        snapshot(a.files[0], a.frame, a.phase, a.outdir, a.bond_distance, unwrap=a.unwrap)
        return
    kinds = ('particle', 'cluster') if a.kind == 'both' else (a.kind,)
    for kind in kinds:
        if a.cmd == 'histogram':
            yield_histogram(a.files[0], kind, a.outdir, bond_distance=a.bond_distance)
        else:
            yield_grid(a.files, kind, a.outdir, bond_distance=a.bond_distance)


if __name__ == '__main__':
    main()
