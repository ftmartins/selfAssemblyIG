"""
Equilibrium yields from a table of species partition functions (ideal cluster
gas, docs/ring_yield_plan.md section 1).

A table is a list of species with sizes ``n_s`` and ``ln q_s`` (q_s = Z_s / A,
reduced units, the monomer has q_1 = 2 pi).  With fugacity z = e^{beta mu}:

    rho_s = q_s z^{n_s},        rho = sum_s n_s rho_s.

* :func:`at_fugacity` — yields at fixed ln z.
* :func:`solve_ln_z` / :func:`at_density` — fix ln z from the total number
  density rho (particle conservation) and return the yields.
* :func:`finite_n` — the exact canonical recursion for N particles in area A,
  ``N Q_N = sum_k k a_k Q_{N-k}``, ``<N_s> = A q_s Q_{N-n_s} / Q_N``, with
  ``a_k = A sum_{n_s = k} q_s``.  Needed for N of order 100 or below.

Yields are reported both ways (CONTEXT.md): ``particle`` (fraction of particles
in s) and ``cluster`` (fraction of clusters, monomers included).

Open chains can be extended past the sampled length with
:func:`extend_chains` (linear tail of ln q_n), so polymerisation near
z K -> 1 is not truncated.  GCMC: with the current insertion move,
ln z = beta mu - ln 2 pi (:func:`ln_z_from_mu`).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp


@dataclass
class SpeciesTable:
    labels: np.ndarray        # 'monomer', 'chain_k', 'ring_n'
    kind: np.ndarray
    n: np.ndarray             # particles per species
    lnq: np.ndarray
    lnq_err: np.ndarray = None
    ess: np.ndarray = None
    scores: np.ndarray = None  # (S, P) d ln q / d a

    @classmethod
    def from_arrays(cls, kind, n, lnq, lnq_err=None, ess=None, scores=None):
        kind = np.asarray(kind).astype(str)
        n = np.asarray(n, int)
        labels = np.array(['monomer' if k == 'monomer' else f'{k}_{m}' for k, m in zip(kind, n)])
        return cls(labels, kind, n, np.asarray(lnq, float),
                   None if lnq_err is None else np.asarray(lnq_err, float),
                   None if ess is None else np.asarray(ess, float),
                   None if scores is None else np.asarray(scores, float))

    def select(self, mask):
        pick = lambda a: None if a is None else a[mask]
        return SpeciesTable(self.labels[mask], self.kind[mask], self.n[mask], self.lnq[mask],
                            pick(self.lnq_err), pick(self.ess), pick(self.scores))

    def finite(self):
        return self.select(np.isfinite(self.lnq))

    def index(self, label):
        return int(np.flatnonzero(self.labels == label)[0])


def extend_chains(table, n_max=400, n_fit=4, min_ess=50.0):
    """Append chains up to ``n_max`` with ln q_n continued linearly from the
    last ``n_fit`` reliable (ESS >= ``min_ess``) sampled chains.  Extrapolated
    rows get NaN errors, zero ESS and the slope-extrapolated scores.  Nothing
    is added unless the longest sampled chain is itself reliable: past the
    point where chains curl into themselves ln q_n bends down, and a line
    through the short chains would wildly overcount long ones."""
    ch = np.flatnonzero(table.kind == 'chain')
    if ch.size < 2:
        return table
    if table.ess is not None and table.ess[ch[np.argmax(table.n[ch])]] < min_ess:
        return table
    ok = ch if table.ess is None else ch[table.ess[ch] >= min_ess]
    ok = ok[np.isfinite(table.lnq[ok])]
    if ok.size < 2:
        return table
    fit = ok[np.argsort(table.n[ok])][-n_fit:]
    slope, icpt = np.polyfit(table.n[fit], table.lnq[fit], 1)
    n_last = table.n[ch].max()
    new_n = np.arange(n_last + 1, n_max + 1)
    if new_n.size == 0:
        return table
    return SpeciesTable.from_arrays(
        np.concatenate([table.kind, np.full(new_n.size, 'chain')]),
        np.concatenate([table.n, new_n]),
        np.concatenate([table.lnq, icpt + slope * new_n]),
        None if table.lnq_err is None else np.concatenate([table.lnq_err, np.full(new_n.size, np.nan)]),
        None if table.ess is None else np.concatenate([table.ess, np.zeros(new_n.size)]),
        None if table.scores is None else np.concatenate(
            [table.scores, _extrapolate_scores(table, fit, new_n)]),
    )


def _extrapolate_scores(table, fit, new_n):
    cols = []
    for j in range(table.scores.shape[1]):
        y = table.scores[fit, j]
        if not np.all(np.isfinite(y)):
            cols.append(np.full(new_n.size, np.nan))
            continue
        b, a = np.polyfit(table.n[fit], y, 1)
        cols.append(a + b * new_n)
    return np.stack(cols, -1)


@dataclass
class Yields:
    labels: np.ndarray
    n: np.ndarray
    ln_z: float
    rho: np.ndarray           # number density of each species
    particle: np.ndarray      # n_s rho_s / rho_total
    cluster: np.ndarray       # rho_s / sum rho
    meta: dict = field(default_factory=dict)

    def as_dict(self, which='cluster'):
        return dict(zip(self.labels, getattr(self, which)))

    def get(self, label, which='cluster'):
        hit = np.flatnonzero(self.labels == label)
        return float(getattr(self, which)[hit[0]]) if hit.size else 0.0


def _log_rho(table, ln_z):
    return table.lnq + table.n * ln_z


def at_fugacity(table, ln_z):
    t = table.finite()
    lr = _log_rho(t, ln_z)
    lp = lr + np.log(t.n)
    return Yields(t.labels, t.n, float(ln_z), np.exp(lr),
                  np.exp(lp - logsumexp(lp)), np.exp(lr - logsumexp(lr)))


def ln_density(table, ln_z):
    """ln of the total number density at fugacity ln z."""
    t = table.finite()
    return float(logsumexp(_log_rho(t, ln_z) + np.log(t.n)))


def solve_ln_z(table, rho):
    """ln z giving total number density ``rho`` (monotone in ln z).  The chain
    series converges only below ln z* = -slope of ln q_n; the bracket stays
    inside it."""
    t = table.finite()
    target = np.log(rho)
    lo = target - t.lnq[t.n == 1][0] - 50.0 if (t.n == 1).any() else target - 100.0
    ch = t.kind == 'chain'
    hi = 50.0
    if ch.sum() >= 2:
        nn, lq = t.n[ch], t.lnq[ch]
        top = np.argsort(nn)[-2:]
        slope = (lq[top[1]] - lq[top[0]]) / (nn[top[1]] - nn[top[0]])
        hi = min(hi, -slope - 1e-9)
    f = lambda x: ln_density(t, x) - target
    if f(hi) < 0:
        raise ValueError('density above what the truncated chain series can hold; '
                         'extend the chains')
    return brentq(f, lo, hi, xtol=1e-12)


def at_density(table, rho):
    y = at_fugacity(table, solve_ln_z(table, rho))
    y.meta['rho'] = float(rho)
    return y


def density_from_area_fraction(phi, core_radius=1.0):
    """Number density of disks of radius ``core_radius`` at area fraction phi."""
    return phi / (np.pi * core_radius ** 2)


def ln_z_from_mu(beta_mu):
    """GCMC convention for the current insertion move: z = e^{beta mu} / 2 pi."""
    return beta_mu - np.log(2 * np.pi)


def finite_n(table, N, area):
    """Exact canonical averages <N_s> for N particles in area A.  Returns a
    :class:`Yields` whose ``rho`` holds <N_s> / A."""
    t = table.finite()
    n_max = min(int(t.n.max()), N)
    t = t.select(t.n <= n_max)
    ln_a = np.full(n_max + 1, -np.inf)
    for k in range(1, n_max + 1):
        sel = t.n == k
        if sel.any():
            ln_a[k] = np.log(area) + logsumexp(t.lnq[sel])
    lnQ = np.full(N + 1, -np.inf)
    lnQ[0] = 0.0
    ks = np.arange(1, n_max + 1)
    for M in range(1, N + 1):
        k = ks[ks <= M]
        lnQ[M] = logsumexp(np.log(k) + ln_a[k] + lnQ[M - k]) - np.log(M)
    lnN = np.log(area) + t.lnq + lnQ[N - t.n] - lnQ[N]
    Ns = np.exp(lnN)
    part = t.n * Ns / N
    clus = Ns / Ns.sum()
    y = Yields(t.labels, t.n, np.nan, Ns / area, part, clus)
    y.meta.update(N=int(N), area=float(area), mean_counts=Ns)
    return y
