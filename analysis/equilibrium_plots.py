"""
Figures for the equilibrium checks (:mod:`analysis.equilibrium`).

``timeseries_figure(source, out_dir)``
    Potential energy, N (only when it varies, i.e. grand canonical),
    bonded-particle fraction and cluster yields against frame number, each panel
    annotated with its stationarity verdict.  Cluster-yield lines use the shared
    structure colour map.  Written to ``<stem>_timeseries.png``.
``comparison_figure(named_yields, out_dir)``
    Particle yield per structure for several runs (dispersed vs pre-assembled
    start, or different seeds) overlaid as grouped bars: colour = structure
    (shared map), hatch = run.  Written to ``comparison.png``.

Headless-safe like :mod:`analysis.structure_plots`; both return the ``Figure``.
"""
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from analysis import equilibrium as eq
from analysis.bond_graph import yields as compute_yields
from analysis.structure_plots import (_all_labels, _finish, _label_order, _open, _stem,
                                      structure_color)

_HATCHES = ['', '//', 'xx', '..', '\\\\', '++']


def _annotate(ax, result):
    v = result.values
    mark = 'stationary' if result.passed else 'NOT stationary'
    ax.text(0.99, 0.97, f"{mark}  (z_halves = {v['z_halves']:.1f}, z_slope = {v['z_slope']:.1f})",
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color='#117733' if result.passed else '#CC3311',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))


def timeseries_figure(source, out_dir=None, phase='production', min_yield=0.02,
                      figsize=(7.0, 2.2), report=None, **yield_kw):
    """Time-series figure with stationarity annotations.  ``report`` (an
    :class:`~analysis.equilibrium.EquilibriumReport`) supplies verdicts when
    given, otherwise they are computed here."""
    owned = isinstance(source, (str, Path))
    reader = _open(source)
    try:
        y = compute_yields(reader, phase=phase, **yield_kw)
        energy = reader.series(phase, 'potential_energy')
        n_series = reader.series(phase, 'N').astype(float)
    finally:
        if owned:
            reader.close()
    n_series = n_series[n_series > 0]
    bonded = 1.0 - y.particle_series['monomer'] if 'monomer' in y.particle_series \
        else np.ones(y.n_frames)
    panels = [('potential energy', energy, '#333333', 'stationarity:potential_energy')]
    if len(set(n_series)) > 1:
        panels.append(('N', n_series, '#333333', 'stationarity:N'))
    panels.append(('bonded-particle fraction', bonded, '#333333',
                   'stationarity:bonded_fraction'))

    labels = [k for k in sorted(y.cluster_series, key=_label_order)
              if y.cluster_yield[k] >= min_yield]
    fig = Figure(figsize=(figsize[0], figsize[1] * (len(panels) + 1)))
    axes = fig.subplots(len(panels) + 1, 1, sharex=True)
    for ax, (name, ser, col, key) in zip(axes, panels):
        ax.plot(ser, color=col, lw=0.8)
        ax.set_ylabel(name, fontsize=8)
        if report is not None and key in report.checks:
            _annotate(ax, report.checks[key])
        elif len(ser) >= eq.MIN_STATIONARITY_POINTS:
            _annotate(ax, eq.stationarity(ser, name=key))
    ax = axes[-1]
    for k in labels:
        ax.plot(y.cluster_series[k], color=structure_color(k), lw=1.0, label=k)
    ax.set_ylabel('cluster yield', fontsize=8)
    ax.set_xlabel('frame')
    if labels:
        ax.legend(fontsize=7, ncol=min(len(labels), 4), loc='upper right')
        verdicts = [eq.stationarity(y.cluster_series[k], name=k) for k in labels
                    if len(y.cluster_series[k]) >= eq.MIN_STATIONARITY_POINTS]
        bad = [r.name for r in verdicts if not r.passed]
        ax.text(0.01, 0.97, 'cluster yields stationary' if not bad
                else 'drifting: ' + ', '.join(bad), transform=ax.transAxes, ha='left',
                va='top', fontsize=8, color='#117733' if not bad else '#CC3311')
    fig.tight_layout()
    return _finish(fig, out_dir, f'{_stem(source)}_timeseries.png')


def comparison_figure(named_yields, out_dir=None, kind='particle', figsize=(7.0, 4.0)):
    """Overlay the ring-size distributions of several runs.  ``named_yields`` maps
    a run name to a :class:`~analysis.bond_graph.Yields` or a trajectory path."""
    runs = {name: (v if hasattr(v, 'particle_yield') else compute_yields(v))
            for name, v in named_yields.items()}
    if len(runs) < 2:
        raise ValueError('comparison needs at least two runs')
    labels = _all_labels(runs.values(), kind)
    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    w = 0.8 / len(runs)
    x = np.arange(len(labels))
    for i, (name, y) in enumerate(runs.items()):
        mean = [getattr(y, f'{kind}_yield').get(k, 0.0) for k in labels]
        err = [np.nan_to_num(getattr(y, f'{kind}_yield_err').get(k, 0.0)) for k in labels]
        ax.bar(x - 0.4 + w * (i + 0.5), mean, w, yerr=err,
               color=[structure_color(k) for k in labels], edgecolor='#333333',
               hatch=_HATCHES[i % len(_HATCHES)], linewidth=0.6, capsize=2,
               error_kw=dict(lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel(f'{kind} yield')
    ax.legend(handles=[Patch(facecolor='white', edgecolor='#333333',
                             hatch=_HATCHES[i % len(_HATCHES)], label=n)
                       for i, n in enumerate(runs)], fontsize=8)
    fig.tight_layout()
    return _finish(fig, out_dir, 'comparison.png')
