"""
One command from a trajectory file to a report directory.

    python -m analysis.report FILE [--outdir D] [--seeds F ...] [--preassembled F]
        [--bond_distance d] [--no_gif]

``D`` defaults to ``<FILE stem>_report`` beside the file.  It contains

``job_<stem>_yields.npz``
    Equilibrium yields in the format the phase-diagram aggregation
    (:func:`analysis.yields.load_yield_grid`) reads: ``alpha_deg`` (opening
    angle), ``selectivity`` (= ``bond_specificity``, E_AB / (E_AA + E_BB)),
    ``avg_Y_<structure>`` and ``avg_count_<structure>`` for triangle, square,
    pentagon, hexagon, heptagon, octagon (the equilibrium *particle* yield /
    *cluster* yield of the ring of that size), plus the full bond-graph result:
    ``labels``, ``particle_yield``, ``particle_yield_err``, ``cluster_yield``,
    ``cluster_yield_err`` (aligned arrays), ``n_frames``, ``n_seeds``,
    ``bond_distance``.  ``--seeds`` adds further files at the same parameters,
    combined with seed-to-seed errors.
``yields.json``
    The same numbers keyed by structure label.
``equilibrium.json``
    :class:`analysis.equilibrium.EquilibriumReport` (stationarity, autocorrelation,
    bond-distance sensitivity, seed agreement with ``--seeds``, forward/backward
    with ``--preassembled``).
``<stem>_snapshot_*.png``, ``<stem>_yield_particle.png``, ``<stem>_yield_cluster.png``,
``<stem>_timeseries.png``, ``comparison.png`` (with ``--seeds`` or
``--preassembled``), ``<stem>.gif`` (at most 50 frames; skipped with a message
when Pillow is missing or ``--no_gif``).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis import equilibrium as eq
from analysis import equilibrium_plots as ep
from analysis import structure_plots as sp
from analysis.bond_graph import combine_seeds, yields as compute_yields
from base.trajectory import open_trajectory

RING_NAMES = {'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6, 'heptagon': 7,
              'octagon': 8}
MAX_GIF_FRAMES = 50


def yields_record(y):
    """Flat dict of arrays/scalars (the ``job_*_yields.npz`` content) for a
    :class:`analysis.bond_graph.Yields` with its file attributes."""
    labels = sorted(y.particle_yield, key=sp._label_order)
    a = y.attrs
    rec = dict(
        alpha_deg=float(np.rad2deg(a['opening_angle'])),
        selectivity=float(a.get('bond_specificity', np.nan)),
        labels=np.array(labels),
        particle_yield=np.array([y.particle_yield[k] for k in labels]),
        particle_yield_err=np.array([y.particle_yield_err[k] for k in labels]),
        cluster_yield=np.array([y.cluster_yield[k] for k in labels]),
        cluster_yield_err=np.array([y.cluster_yield_err[k] for k in labels]),
        n_frames=int(y.n_frames), n_seeds=int(y.n_seeds), bond_distance=float(y.bond_distance))
    for name, n in RING_NAMES.items():
        rec[f'avg_Y_{name}'] = float(y.particle_yield.get(f'ring_{n}', 0.0))
        rec[f'avg_count_{name}'] = float(y.cluster_yield.get(f'ring_{n}', 0.0))
    return rec


def build_report(path, outdir=None, seeds=(), preassembled=None, bond_distance=None,
                 gif=True, n_blocks=10):
    """Run analysis, equilibrium checks and plots; returns ``(outdir, report)``."""
    path = Path(path)
    stem = path.stem
    out = Path(outdir) if outdir is not None else path.with_name(f'{stem}_report')
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(bond_distance=bond_distance, n_blocks=n_blocks)

    main = compute_yields(path, **kw)
    combined = combine_seeds([main] + [compute_yields(p, **kw) for p in seeds]) \
        if seeds else main
    rec = yields_record(combined)
    np.savez(out / f'job_{stem}_yields.npz', **rec)
    (out / 'yields.json').write_text(json.dumps(dict(
        n_frames=combined.n_frames, n_seeds=combined.n_seeds,
        bond_distance=combined.bond_distance,
        particle_yield=combined.particle_yield, particle_yield_err=combined.particle_yield_err,
        cluster_yield=combined.cluster_yield, cluster_yield_err=combined.cluster_yield_err,
    ), indent=2))

    report = eq.run_checks(path, other_seeds=seeds, preassembled=preassembled, save=False,
                           **kw)
    report.save(out / 'equilibrium.json')

    sp.snapshot(path, -1, out_dir=out, bond_distance=bond_distance)
    for kind in ('particle', 'cluster'):
        sp.yield_histogram(combined, kind, out, stem=stem)
    ep.timeseries_figure(path, out, report=report, **kw)
    runs = {stem: combined}
    runs.update({Path(p).stem: compute_yields(p, **kw) for p in seeds})
    if preassembled is not None:
        runs['pre-assembled'] = compute_yields(preassembled, **kw)
    if len(runs) > 1:
        ep.comparison_figure(runs, out)
    if gif:
        _gif(path, out / f'{stem}.gif', bond_distance)
    return out, report


def _gif(path, out_path, bond_distance):
    from analysis import movie
    with open_trajectory(path) as r:
        n = r.n_frames('production')
    try:
        movie.make_gif(path, out_path, stride=max(1, -(-n // MAX_GIF_FRAMES)),
                       bond_distance=bond_distance)
    except ImportError as e:
        print(f'skipping GIF: {e}')


def main(argv=None):
    p = argparse.ArgumentParser(description='Analysis, equilibrium checks and plots '
                                'for a trajectory file')
    p.add_argument('file')
    p.add_argument('--outdir', default=None)
    p.add_argument('--seeds', nargs='*', default=[], help='further seeds, same parameters')
    p.add_argument('--preassembled', default=None, help='run from a pre-assembled start')
    p.add_argument('--bond_distance', type=float, default=None)
    p.add_argument('--no_gif', action='store_true')
    a = p.parse_args(argv)
    out, report = build_report(a.file, a.outdir, a.seeds, a.preassembled,
                               a.bond_distance, gif=not a.no_gif)
    for name, c in report.checks.items():
        print(f"  {'PASS' if c.passed else 'FAIL'}  {name}")
    print(f'Report: {out}')


if __name__ == '__main__':
    main()
