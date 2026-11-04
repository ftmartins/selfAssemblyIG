"""
Predicted equilibrium yields against MD / GCMC (docs/ring_yield_plan.md,
validation steps 3 and 4).

* :func:`compare` — bond-graph yields of one or more trajectories at a single
  design point (``analysis.bond_graph.yields``, combined over seeds) next to the
  mass-action prediction from a species table, both at fixed density and with
  the exact finite-N recursion for the simulated N and box.
* :func:`ideal_gas_test` — model-free check on simulations at several
  densities: ln rho_s - n_s ln rho_1 must not depend on density where the
  ideal cluster gas holds (equivalently ln rho_s against beta mu has slope n_s).
"""
from dataclasses import dataclass

import numpy as np

from analysis.bond_graph import combine_seeds, yields as bond_yields
from analysis.mass_action import at_density, finite_n


@dataclass
class Comparison:
    labels: list
    simulated: np.ndarray
    simulated_err: np.ndarray
    predicted_rho: np.ndarray      # mass action at the simulated density
    predicted_N: np.ndarray        # finite-N recursion
    which: str
    N: int
    area: float

    def table(self):
        rows = [f'{"species":10s} {"sim":>8s} {"err":>7s} {"rho":>8s} {"N":>8s}']
        for l, s, e, a, b in zip(self.labels, self.simulated, self.simulated_err,
                                 self.predicted_rho, self.predicted_N):
            rows.append(f'{l:10s} {s:8.4f} {e:7.4f} {a:8.4f} {b:8.4f}')
        return '\n'.join(rows)

    def chi2(self, floor=1e-3):
        err = np.maximum(self.simulated_err, floor)
        return float(np.sum(((self.simulated - self.predicted_N) / err) ** 2))


def compare(table, trajectories, N=None, which='particle', labels=None, **yield_kw):
    """Simulated vs predicted yields; ``N`` defaults to the ``N`` attribute of
    the first trajectory."""
    ys = [bond_yields(t, **yield_kw) for t in trajectories]
    y = combine_seeds(ys) if len(ys) > 1 else ys[0]
    L = float(y.attrs['box_size'])
    if N is None:
        N = int(y.attrs.get('N', y.attrs.get('n_particles')))
    area = L * L
    pr = at_density(table, N / area)
    pN = finite_n(table, N, area)
    sim = getattr(y, f'{which}_yield')
    sim_err = getattr(y, f'{which}_yield_err')
    labels = labels or sorted(set(sim) | {l for l, v in zip(pr.labels, getattr(pr, which)) if v > 1e-3},
                              key=_label_order)
    return Comparison(labels, np.array([sim.get(l, 0.0) for l in labels]),
                      np.array([sim_err.get(l, 0.0) for l in labels]),
                      np.array([pr.get(l, which) for l in labels]),
                      np.array([pN.get(l, which) for l in labels]), which, N, area)


def _label_order(l):
    if l == 'monomer':
        return (0, 0)
    kind, n = l.split('_')
    return (1 if kind == 'chain' else 2, int(n.rstrip('+')))


def ideal_gas_test(densities, rho_species):
    """``rho_species``: list (one per density) of dicts label -> number density.
    Returns label -> array of ln rho_s - n_s ln rho_1 across densities and its
    spread (max - min); a flat line is the ideal-cluster-gas regime."""
    out = {}
    labels = set().union(*[d.keys() for d in rho_species]) - {'monomer'}
    for l in labels:
        n = int(l.split('_')[1].rstrip('+'))
        vals = np.array([np.log(d[l]) - n * np.log(d['monomer']) if d.get(l, 0) > 0 and d.get('monomer', 0) > 0
                         else np.nan for d in rho_species])
        out[l] = dict(values=vals, spread=float(np.nanmax(vals) - np.nanmin(vals))
                      if np.isfinite(vals).any() else np.nan)
    return out
