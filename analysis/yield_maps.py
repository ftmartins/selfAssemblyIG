"""
Equilibrium yield maps over (theta, E_AB, rho) from the partition scan
(docs/ring_yield_plan.md section 3).

Pipeline
--------
1. :func:`load_scan` reads ``data/partition/*.npz`` (``simulation/runners/
   scan_partition.py``) into a :class:`Scan`: a regular (E_AB, theta) grid of
   :class:`analysis.mass_action.SpeciesTable` rows, with the ring mode data.
2. :func:`refine` interpolates ln q of every species onto a finer grid (PCHIP
   along theta, then along E_AB), so maps resolve the ring-size crossovers.
   ln q depends smoothly on both; ESS and errors take the nearest scanned point.
3. :func:`compute_maps` solves mass action at each point for one density
   (``phi`` area fraction or ``rho``) and returns a :class:`Maps` of particle
   and cluster yields per species, ln z, the dominant ring, the assembled
   fraction, the yield carried by unreliable species (ESS below ``min_ess`` or
   ln q error above ``max_err``), and optionally the Fisher metric.
   :func:`cached_maps` stores each result under ``data/yield_maps/``.
4. Plot functions (`plot_*`) draw the phase-space figures of the old
   analytical notebook from mass action: dominant ring, A-vs-B relative yield
   (Y_a - Y_b)/(Y_a + Y_b), per-species yields and the assembled fraction, with
   the matching angles theta_n marked.
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from scipy.interpolate import PchipInterpolator

from analysis import fisher
from analysis.mass_action import (SpeciesTable, at_density, density_from_area_fraction,
                                  extend_chains)
from base.paths import PARTITION_DIR, YIELD_MAP_DIR, ensure_dir

# Categorical slots in fixed order (dataviz reference palette); ring n -> slot n-3.
RING_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
               '#6250d6', '#e34948']
CHAIN_COLOR = '#8a8984'
MONOMER_COLOR = '#c9c8c2'
DIVERGING = LinearSegmentedColormap.from_list('blue_gray_orange', ['#1c5cab', '#86b6ef', '#f0efec',
                                                                   '#f2a07c', '#b8461a'])
SEQUENTIAL = LinearSegmentedColormap.from_list('blues', ['#f4f8fd', '#9ec5f4', '#3987e5',
                                                         '#1c5cab', '#0d366b'])


def ring_color(n):
    return RING_COLORS[(n - 3) % len(RING_COLORS)] if n - 3 < len(RING_COLORS) else CHAIN_COLOR


# ── Loading ────────────────────────────────────────────────────────────

def load_point(path):
    """One scan NPZ -> (SpeciesTable, meta dict, modes dict)."""
    with np.load(path, allow_pickle=False) as d:
        t = SpeciesTable.from_arrays(d['kind'], d['n'], d['lnq'], d['lnq_err'], d['ess'], d['scores'])
        meta = dict(theta_deg=float(d['theta_deg']), E_AB=float(d['E_AB']),
                    E_AA=float(d['E_AA']), E_BB=float(d['E_BB']), kT=float(d['kT']))
        modes = {k[5:]: d[k] for k in d.files if k.startswith('mode_')}
    return t, meta, modes


@dataclass
class Scan:
    thetas_deg: np.ndarray            # (T,)
    eabs: np.ndarray                  # (E,)
    labels: np.ndarray                # species labels, common to every point
    kind: np.ndarray
    n: np.ndarray
    lnq: np.ndarray                   # (E, T, S), NaN where missing
    lnq_err: np.ndarray
    ess: np.ndarray
    scores: np.ndarray                # (E, T, S, P)
    modes: dict = field(default_factory=dict)   # (theta, E) -> modes dict

    def table(self, i, j):
        return SpeciesTable(self.labels, self.kind, self.n, self.lnq[i, j], self.lnq_err[i, j],
                            self.ess[i, j], self.scores[i, j])

    @property
    def shape(self):
        return self.lnq.shape[:2]


def load_scan(directory=None):
    """All points of a partition scan as a :class:`Scan` (missing points NaN)."""
    files = sorted(Path(PARTITION_DIR if directory is None else directory).glob('oa*_EAB*.npz'))
    files = [f for f in files if '.tmp' not in f.name]
    if not files:
        raise FileNotFoundError(f'no scan points in {directory or PARTITION_DIR}')
    pts = [load_point(f) for f in files]
    ref = max(pts, key=lambda p: len(p[0].labels))[0]
    thetas = np.unique([m['theta_deg'] for _, m, _ in pts])
    eabs = np.unique([m['E_AB'] for _, m, _ in pts])
    S, P = len(ref.labels), ref.scores.shape[1]
    lnq = np.full((len(eabs), len(thetas), S), np.nan)
    err, ess = lnq.copy(), np.zeros_like(lnq)
    sc = np.full((len(eabs), len(thetas), S, P), np.nan)
    modes = {}
    pos = {l: k for k, l in enumerate(ref.labels)}
    for t, m, md in pts:
        i, j = np.searchsorted(eabs, m['E_AB']), np.searchsorted(thetas, m['theta_deg'])
        idx = np.array([pos[l] for l in t.labels])
        lnq[i, j, idx], err[i, j, idx], ess[i, j, idx], sc[i, j, idx] = t.lnq, t.lnq_err, t.ess, t.scores
        modes[(m['theta_deg'], m['E_AB'])] = md
    return Scan(thetas, eabs, ref.labels, ref.kind, ref.n, lnq, err, ess, sc, modes)


# ── Interpolation ─────────────────────────────────────────────────────

def _pchip_axis(x, y, x_new, axis):
    """PCHIP of y along ``axis`` per column, using only finite samples (NaN
    outside their range)."""
    y = np.moveaxis(y, axis, 0)
    out = np.full((len(x_new),) + y.shape[1:], np.nan)
    flat, fout = y.reshape(len(x), -1), out.reshape(len(x_new), -1)
    for c in range(flat.shape[1]):
        ok = np.isfinite(flat[:, c])
        if ok.sum() >= 2:
            f = PchipInterpolator(x[ok], flat[ok, c], extrapolate=False)
            fout[:, c] = f(x_new)
    return np.moveaxis(out, 0, axis)


def _nearest(x, x_new):
    return np.abs(np.asarray(x_new)[:, None] - np.asarray(x)[None]).argmin(1)


def refine(scan, thetas_deg=None, eabs=None):
    """Scan interpolated onto ``thetas_deg`` x ``eabs`` (defaults: 1 degree,
    and 4x in E_AB).  ln q is finite only where it was finite on both sides."""
    if thetas_deg is None:
        thetas_deg = np.arange(scan.thetas_deg[0], scan.thetas_deg[-1] + 1e-9, 1.0)
    if eabs is None:
        eabs = np.unique(np.concatenate([np.linspace(a, b, 5)[:-1]
                                         for a, b in zip(scan.eabs[:-1], scan.eabs[1:])] + [scan.eabs[-1:]]))
    lnq = _pchip_axis(scan.eabs, _pchip_axis(scan.thetas_deg, scan.lnq, thetas_deg, 1), eabs, 0)
    sc = _pchip_axis(scan.eabs, _pchip_axis(scan.thetas_deg, scan.scores, thetas_deg, 1), eabs, 0)
    ie, it = _nearest(scan.eabs, eabs), _nearest(scan.thetas_deg, thetas_deg)
    return Scan(np.asarray(thetas_deg, float), np.asarray(eabs, float), scan.labels, scan.kind, scan.n,
                lnq, scan.lnq_err[np.ix_(ie, it)], scan.ess[np.ix_(ie, it)], sc, scan.modes)


# ── Maps ───────────────────────────────────────────────────────────────

@dataclass
class Maps:
    thetas_deg: np.ndarray
    eabs: np.ndarray
    labels: np.ndarray
    n: np.ndarray
    kind: np.ndarray
    rho: float
    particle: np.ndarray      # (E, T, S)
    cluster: np.ndarray       # (E, T, S)
    ln_z: np.ndarray          # (E, T)
    unreliable: np.ndarray    # (E, T) particle yield in unreliable species
    metric: np.ndarray = None  # (E, T, P, P) cluster-yield Fisher metric

    def get(self, label, which='cluster'):
        hit = np.flatnonzero(self.labels == label)
        arr = getattr(self, which)
        return arr[..., hit[0]] if hit.size else np.zeros(arr.shape[:2])

    def rings(self):
        return sorted(int(n) for k, n in zip(self.kind, self.n) if k == 'ring')

    def dominant_ring(self, which='cluster'):
        """Ring size with the largest yield among rings (NaN where no ring)."""
        rs = self.rings()
        y = np.stack([self.get(f'ring_{r}', which) for r in rs], -1)
        out = np.array(rs, float)[np.nanargmax(np.nan_to_num(y, nan=-1.0), -1)]
        return np.where(np.isfinite(y).any(-1), out, np.nan)

    def assembled(self):
        """Fraction of particles not monomers."""
        return 1.0 - self.get('monomer', 'particle')

    def ring_fraction(self):
        """Fraction of particles in rings."""
        sel = self.kind == 'ring'
        return np.nansum(self.particle[..., sel], -1)

    def relative(self, a, b, which='cluster'):
        """(Y_a - Y_b) / (Y_a + Y_b) in [-1, 1]."""
        ya, yb = self.get(a, which), self.get(b, which)
        tot = ya + yb
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(tot > 0, (ya - yb) / tot, np.nan)

    def save(self, path):
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            kw = {k: d[k] for k in d.files}
        kw['rho'] = float(kw['rho'])
        return cls(**kw)


def compute_maps(scan, phi=None, rho=None, min_ess=100.0, max_err=0.3, chain_tail=400,
                 with_metric=True):
    """Mass-action yields at every grid point for one density."""
    if rho is None:
        rho = density_from_area_fraction(phi)
    E, T = scan.shape
    S = len(scan.labels)
    part = np.full((E, T, S), np.nan)
    clus = part.copy()
    lnz = np.full((E, T), np.nan)
    unrel = np.full((E, T), np.nan)
    P = scan.scores.shape[-1]
    metric = np.full((E, T, P, P), np.nan) if with_metric else None
    for i in range(E):
        for j in range(T):
            t = scan.table(i, j)
            if not np.isfinite(t.lnq[t.n == 1]).all() or np.isfinite(t.lnq).sum() < 2:
                continue
            bad = (t.ess < min_ess) | ~(t.lnq_err <= max_err)
            bad[t.n == 1] = False
            try:
                ext = extend_chains(t, chain_tail, min_ess=min_ess) if chain_tail else t
                y = at_density(ext, rho)
            except ValueError:
                continue
            idx = {l: k for k, l in enumerate(t.labels)}
            for l, p, c in zip(y.labels, y.particle, y.cluster):
                if l in idx:
                    part[i, j, idx[l]], clus[i, j, idx[l]] = p, c
            lnz[i, j] = y.ln_z
            tail = np.sum(y.particle[[l not in idx for l in y.labels]])
            unrel[i, j] = np.nansum(np.where(bad, part[i, j], 0.0)) + tail
            if with_metric and ext.scores is not None:
                metric[i, j] = fisher.cluster_metric(ext, y)
    return Maps(scan.thetas_deg, scan.eabs, scan.labels, scan.n, scan.kind, float(rho),
                part, clus, lnz, unrel, metric)


def _scan_digest(scan):
    h = hashlib.sha1()
    for a in (scan.thetas_deg, scan.eabs, np.nan_to_num(scan.lnq, nan=-1e9)):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:12]


def cached_maps(scan, phi, cache_dir=None, overwrite=False, **kw):
    """:func:`compute_maps`, stored by (scan contents, phi, options)."""
    key = json.dumps(dict(kw, phi=float(phi)), sort_keys=True)
    tag = hashlib.sha1((key + _scan_digest(scan)).encode()).hexdigest()[:12]
    path = ensure_dir(YIELD_MAP_DIR if cache_dir is None else cache_dir) / f'maps_phi{phi:.4f}_{tag}.npz'
    if path.exists() and not overwrite:
        return Maps.load(path)
    m = compute_maps(scan, phi=phi, **kw)
    m.save(path)
    return m


# ── Plots ──────────────────────────────────────────────────────────────

def _edges(x, log=False):
    x = np.log(x) if log else np.asarray(x, float)
    mid = 0.5 * (x[1:] + x[:-1])
    e = np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])
    return np.exp(e) if log else e


def _axes_style(ax, maps, matching=True, n_range=range(3, 13)):
    ax.set_yscale('log')
    ticks = [t for t in (5, 7.5, 10, 15, 20, 30, 40) if maps.eabs[0] <= t <= maps.eabs[-1]]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:g}' for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(r'opening angle $\theta$ (deg)')
    ax.set_ylabel(r'$E_{AB}/k_BT$')
    if matching:
        for n in n_range:
            th = 180.0 * (n - 2) / n
            if maps.thetas_deg[0] <= th <= maps.thetas_deg[-1]:
                ax.axvline(th, color='#52514e', lw=0.6, ls=':', alpha=0.8)
                ax.text(th, 1.0, f'{n}', transform=ax.get_xaxis_transform(), ha='center',
                        va='bottom', fontsize=7, color='#52514e')


def _mesh(ax, maps, Z, **kw):
    return ax.pcolormesh(_edges(maps.thetas_deg), _edges(maps.eabs, log=True), Z,
                         shading='flat', rasterized=True, **kw)


def plot_dominant_ring(maps, ax=None, which='cluster', min_yield=0.0):
    """Ring size with the largest yield; hatched where rings hold less than
    ``min_yield`` of the clusters (or particles)."""
    ax = ax or plt.subplots(figsize=(5.5, 4))[1]
    rs = maps.rings()
    dom = maps.dominant_ring(which)
    cmap = ListedColormap([ring_color(r) for r in rs])
    norm = BoundaryNorm(np.arange(rs[0] - 0.5, rs[-1] + 1.5), len(rs))
    m = _mesh(ax, maps, dom, cmap=cmap, norm=norm)
    if min_yield > 0:
        ring_tot = np.nansum(np.stack([maps.get(f'ring_{r}', which) for r in rs], -1), -1)
        ax.contourf(maps.thetas_deg, maps.eabs, (ring_tot < min_yield).astype(float),
                    levels=[0.5, 1.5], colors='none', hatches=['///'])
    cb = plt.colorbar(m, ax=ax, ticks=rs)
    cb.set_label('dominant ring size n')
    _axes_style(ax, maps)
    ax.set_title(f'dominant ring ({which} yield), $\\phi$={maps.rho * np.pi:.3g}', fontsize=10)
    return ax


def plot_relative(maps, a='ring_3', b='ring_4', ax=None, which='cluster'):
    """(Y_a - Y_b) / (Y_a + Y_b), diverging around 0."""
    ax = ax or plt.subplots(figsize=(5.5, 4))[1]
    m = _mesh(ax, maps, maps.relative(a, b, which), cmap=DIVERGING, vmin=-1, vmax=1)
    cb = plt.colorbar(m, ax=ax)
    cb.set_label(f'({a} − {b}) / ({a} + {b})')
    _axes_style(ax, maps)
    ax.set_title(f'{a} vs {b} ({which} yield), $\\phi$={maps.rho * np.pi:.3g}', fontsize=10)
    return ax


def plot_yield(maps, label, ax=None, which='cluster', log=False, vmin=None):
    """One species' yield (linear 0-1, or log10)."""
    ax = ax or plt.subplots(figsize=(5.5, 4))[1]
    Z = maps.get(label, which)
    if log:
        with np.errstate(divide='ignore'):
            Z = np.log10(Z)
        m = _mesh(ax, maps, Z, cmap=SEQUENTIAL, vmin=-6 if vmin is None else vmin, vmax=0)
    else:
        m = _mesh(ax, maps, Z, cmap=SEQUENTIAL, vmin=0, vmax=1)
    cb = plt.colorbar(m, ax=ax)
    cb.set_label(('log$_{10}$ ' if log else '') + f'{which} yield')
    _axes_style(ax, maps)
    ax.set_title(label, fontsize=10)
    return ax


def plot_species_grid(maps, labels=None, which='cluster', ncols=4, log=False):
    """Small multiples of per-species yields, one shared color scale."""
    labels = labels or [f'ring_{n}' for n in maps.rings()[:8]]
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    for ax, l in zip(axes.flat, labels):
        Z = maps.get(l, which)
        if log:
            with np.errstate(divide='ignore'):
                Z = np.log10(Z)
            m = _mesh(ax, maps, Z, cmap=SEQUENTIAL, vmin=-6, vmax=0)
        else:
            m = _mesh(ax, maps, Z, cmap=SEQUENTIAL, vmin=0, vmax=1)
        _axes_style(ax, maps)
        ax.set_title(l, fontsize=10)
        ax.label_outer()
    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)
    fig.colorbar(m, ax=axes, shrink=0.8, label=('log$_{10}$ ' if log else '') + f'{which} yield')
    fig.suptitle(f'{which} yields, $\\phi$={maps.rho * np.pi:.3g}')
    return fig


def plot_assembled(maps, ax=None, rings_only=False):
    """Fraction of particles assembled (not monomers), or in rings."""
    ax = ax or plt.subplots(figsize=(5.5, 4))[1]
    Z = maps.ring_fraction() if rings_only else maps.assembled()
    m = _mesh(ax, maps, Z, cmap=SEQUENTIAL, vmin=0, vmax=1)
    plt.colorbar(m, ax=ax, label='particles in rings' if rings_only else 'particles assembled')
    _axes_style(ax, maps)
    ax.set_title(('ring' if rings_only else 'assembled') + f' fraction, $\\phi$={maps.rho * np.pi:.3g}',
                 fontsize=10)
    return ax


def plot_unreliable(maps, ax=None):
    """Particle yield in species whose ln q failed the ESS / error cut."""
    ax = ax or plt.subplots(figsize=(5.5, 4))[1]
    with np.errstate(divide='ignore'):
        Z = np.log10(np.maximum(maps.unreliable, 1e-12))
    m = _mesh(ax, maps, Z, cmap=SEQUENTIAL, vmin=-6, vmax=0)
    plt.colorbar(m, ax=ax, label='log$_{10}$ particle yield in unreliable species')
    _axes_style(ax, maps)
    ax.set_title('sampling quality', fontsize=10)
    return ax


def plot_lnq_per_particle(scan, E_AB, ax=None, kinds=('ring',), n_max=10):
    """ln q_n / n against theta at one E_AB: where each ring is favoured per particle."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    i = int(np.argmin(np.abs(scan.eabs - E_AB)))
    for s, (k, n) in enumerate(zip(scan.kind, scan.n)):
        if k not in kinds or n > n_max:
            continue
        ax.plot(scan.thetas_deg, scan.lnq[i, :, s] / n, lw=2, color=ring_color(n) if k == 'ring' else CHAIN_COLOR,
                ls='-' if k == 'ring' else '--', label=f'{k} {n}')
    ax.set_xlabel(r'opening angle $\theta$ (deg)')
    ax.set_ylabel(r'$\ln q_n / n$')
    ax.set_title(f'$E_{{AB}}$ = {scan.eabs[i]:g}', fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    return ax
