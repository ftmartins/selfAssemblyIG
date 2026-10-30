"""
Equilibrium checks over trajectory-derived series.

Each check returns a :class:`CheckResult` (``name``, ``passed``, ``values``);
:class:`EquilibriumReport` bundles them, round-trips through JSON and is saved
beside the trajectory (``<stem>.equilibrium.json``).

Checks
------
``stationarity(series)``
    First versus second half of the series (difference in standard errors, each
    half's error from block means), OLS drift slope with an autocorrelation-
    corrected error, and block means.  Passes if both z-scores are below
    ``z_crit``.
``integrated_autocorrelation_time(series)``
    Sokal automatic windowing; ``tau = 1 + 2 sum_t rho(t)`` so white noise has
    tau = 1 and AR(1) with coefficient phi has (1 + phi) / (1 - phi).
``agreement(yields_list)``
    Per-structure chi-squared test that particle yields (with their errors,
    floored at ``atol``) agree across the given :class:`bond_graph.Yields`.
    :func:`forward_backward` (dispersed vs pre-assembled starts) and
    :func:`seed_agreement` (independent seeds) are this test with a name.
``bond_distance_sensitivity(source)``
    Particle yields recomputed at bond distances around the patch radius; passes
    if no structure moves by more than ``tol``.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from analysis.bond_graph import yields as compute_yields
from base.config import PATCH_RADIUS
from base.trajectory import open_trajectory


@dataclass
class CheckResult:
    name: str
    passed: bool
    values: dict = field(default_factory=dict)


@dataclass
class EquilibriumReport:
    checks: dict = field(default_factory=dict)      # name -> CheckResult
    meta: dict = field(default_factory=dict)

    @property
    def passed(self):
        return all(c.passed for c in self.checks.values())

    def add(self, result):
        self.checks[result.name] = result
        return result

    def save(self, path):
        data = dict(meta=self.meta, checks={k: asdict(v) for k, v in self.checks.items()})
        Path(path).write_text(json.dumps(data, indent=2, default=_json_default))
        return Path(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls({k: CheckResult(**v) for k, v in data['checks'].items()},
                   data.get('meta', {}))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def report_path(trajectory_path):
    p = Path(trajectory_path)
    return p.with_name(p.stem + '.equilibrium.json')


# ── statistics ─────────────────────────────────────────────────────────────

def integrated_autocorrelation_time(x, c=5.0):
    """Integrated autocorrelation time (Sokal windowing, window M >= c tau)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2 or x.std() == 0:
        return 1.0
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    rho = acf / acf[0]
    taus = 2.0 * np.cumsum(rho) - 1.0
    window = np.arange(n) >= c * taus
    m = int(np.argmax(window)) if window.any() else n - 1
    return float(max(taus[m], 1.0))


def _block_stats(x, n_blocks):
    nb = max(2, min(int(n_blocks), len(x)))
    means = np.array([b.mean() for b in np.array_split(x, nb)])
    return means, float(means.std(ddof=1) / np.sqrt(nb))


def stationarity(series, n_blocks=10, z_crit=3.0, name='stationarity'):
    """Stationarity of one series (see module docstring)."""
    x = np.asarray(series, float)
    if len(x) < 8:
        raise ValueError('need at least 8 points for a stationarity check')
    h = len(x) // 2
    first, second = x[:h], x[h:]
    nb_half = max(2, n_blocks // 2)
    (_, e1), (_, e2) = _block_stats(first, nb_half), _block_stats(second, nb_half)
    diff = float(second.mean() - first.mean())
    se = float(np.hypot(e1, e2))
    z_halves = diff / se if se > 0 else (0.0 if diff == 0 else np.inf)

    t = np.arange(len(x), dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    resid = x - (slope * t + intercept)
    tau = integrated_autocorrelation_time(resid)
    slope_se = resid.std(ddof=2) / (t.std() * np.sqrt(len(x))) * np.sqrt(tau)
    z_slope = float(slope / slope_se) if slope_se > 0 else (0.0 if slope == 0 else np.inf)

    block_means, _ = _block_stats(x, n_blocks)
    passed = bool(abs(z_halves) < z_crit and abs(z_slope) < z_crit)
    return CheckResult(name, passed, dict(
        first_half_mean=float(first.mean()), second_half_mean=float(second.mean()),
        difference=diff, z_halves=float(z_halves), drift_slope=float(slope),
        z_slope=z_slope, tau_int=tau, block_means=block_means.tolist(), z_crit=z_crit))


def agreement(yields_list, name='agreement', atol=0.01, alpha=0.01):
    """Chi-squared agreement of particle yields across ``yields_list``.

    Per structure: chi2 = sum_s (y_s - mean)^2 / (err_s^2 + atol^2) with S - 1
    degrees of freedom; the check fails if any structure's p-value is below
    ``alpha`` / (number of structures)."""
    ys = list(yields_list)
    if len(ys) < 2:
        raise ValueError('need at least two sets of yields to compare')
    labels = sorted({k for y in ys for k in y.particle_yield})
    p_values, max_dev = {}, {}
    for k in labels:
        v = np.array([y.particle_yield.get(k, 0.0) for y in ys])
        e = np.array([np.nan_to_num(y.particle_yield_err.get(k, 0.0)) for y in ys])
        w = 1.0 / (e ** 2 + atol ** 2)
        mean = np.sum(w * v) / np.sum(w)
        stat = float(np.sum(w * (v - mean) ** 2))
        p_values[k] = float(chi2.sf(stat, len(ys) - 1))
        max_dev[k] = float(v.max() - v.min())
    worst = min(p_values.values())
    return CheckResult(name, bool(worst >= alpha / len(labels)), dict(
        p_values=p_values, spread=max_dev, min_p=worst, atol=atol, alpha=alpha,
        n_sets=len(ys)))


def forward_backward(dispersed, preassembled, **kw):
    """Dispersed-monomer start versus pre-assembled-ring start agree."""
    return agreement([dispersed, preassembled], name='forward_backward', **kw)


def seed_agreement(per_seed_yields, **kw):
    """Independent seeds agree on the ring-size distribution."""
    return agreement(per_seed_yields, name='seed_agreement', **kw)


def bond_distance_sensitivity(source, factors=(0.9, 1.0, 1.1), tol=0.05,
                              name='bond_distance_sensitivity', center=PATCH_RADIUS,
                              **yield_kw):
    """Yields at bond distances ``factor * center`` (default the patch radius); passes if no
    structure's particle yield moves by more than ``tol`` across ``factors``."""
    reader = open_trajectory(source) if isinstance(source, (str, Path)) else source
    ys = [compute_yields(reader, bond_distance=f * center, **yield_kw)
          for f in factors]
    labels = sorted({k for y in ys for k in y.particle_yield})
    spread = {k: float(np.ptp([y.particle_yield.get(k, 0.0) for y in ys]))
              for k in labels}
    return CheckResult(name, bool(max(spread.values()) <= tol), dict(
        factors=list(factors), spread=spread, max_spread=max(spread.values()), tol=tol))


# ── driver ─────────────────────────────────────────────────────────────────

def run_checks(trajectory_path, other_seeds=(), preassembled=None, save=True,
               min_yield=0.02, **yield_kw):
    """All checks for one trajectory file.

    Stationarity is tested for the potential energy and for every structure
    whose mean particle yield exceeds ``min_yield``.  ``other_seeds`` are
    further trajectory files at the same parameters (seed agreement) and
    ``preassembled`` is a trajectory from a pre-assembled start (forward/
    backward).  The report is saved to ``report_path(trajectory_path)``.
    """
    report = EquilibriumReport(meta=dict(trajectory=str(trajectory_path)))
    with open_trajectory(trajectory_path) as r:
        main = compute_yields(r, **yield_kw)
        energy = r.series('production', 'potential_energy')
        report.meta['n_frames'] = main.n_frames
        report.add(stationarity(energy, name='stationarity:potential_energy'))
        for k, ser in main.particle_series.items():
            if main.particle_yield[k] >= min_yield:
                report.add(stationarity(ser, name=f'stationarity:{k}'))
        tau = integrated_autocorrelation_time(energy)
        report.add(CheckResult('autocorrelation', True, dict(
            tau_int_energy=tau, n_independent=len(energy) / tau)))
        report.add(bond_distance_sensitivity(
            r, center=yield_kw.get('bond_distance') or PATCH_RADIUS,
            **{k: v for k, v in yield_kw.items() if k != 'bond_distance'}))
    if other_seeds:
        report.add(seed_agreement([main] + [compute_yields(p, **yield_kw)
                                            for p in other_seeds]))
    if preassembled is not None:
        report.add(forward_backward(main, compute_yields(preassembled, **yield_kw)))
    if save:
        report.save(report_path(trajectory_path))
    return report
