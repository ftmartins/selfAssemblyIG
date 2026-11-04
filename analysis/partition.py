"""
Per-species partition functions ln q_s of open chains and rings by sequential
sampling (docs/ring_yield_plan.md, sections 1 and 3).

Conventions
-----------
A cluster of ``n`` labelled particles is a path or a cycle of A-B bonds
(ADR-0002).  Bonds run ``B_k -> A_{k+1}``.  The ``1/n!`` of identical particles
cancels against the labellings: an open chain is integrated over one fixed
ordering (sigma = 1), a ring over one fixed cyclic ordering divided by
``sigma_n = n``.  Fixing particle 0 gauges out translation (area A) and
rotation (2 pi), so

    q_s = Z_s / A = (2 pi / sigma_s) * I_s,
    I_s = integral over particles 1..n-1 of exp(-beta U) * 1[bond graph is s],

and the monomer has q_1 = 2 pi.  The energy is the full Hamiltonian of
``base.energy`` (core eps_c / 2.5 (1 - d/2)^2.5, smooth-window Morse on every
patch pair), so the result is the same model as MD and GCMC (ADR-0001).

Sampling
--------
* **Pair proposal** (:class:`PairProposal`).  Particle k+1 relative to k is
  ``(r, psi, delta)``: A_{k+1} = B_k + r e(psi) with r < a (the bond), and
  phi_{k+1} = phi_k + theta + pi + delta, so delta = 0 is a straight joint.
  The proposal is a piecewise-constant density on a fine (r, psi, delta) grid
  shaped by the pair Boltzmann factor, mixed with 1% uniform so its support
  covers the whole bond region; samples carry exact weights e^{-beta u}/p, so
  every estimate is unbiased whatever the grid.  ``tilt`` multiplies the grid
  by e^{lambda delta}; a ring of n closes when the mean joint deviation is
  ``theta_n - theta``, and :meth:`PairProposal.tilt_for_ring` picks lambda for
  that, so strained rings are reachable.
* **Chains** are grown by Rosenbluth sampling with ``n_trials`` pair draws per
  step, so chains longer than the matching ring that curl into themselves are
  still sampled.  One growth gives every chain length up to ``n_chain_max``.
* **Rings** of n: a tilted Rosenbluth chain of n-1 particles, then the last
  particle is drawn in closed form so that its A patch lies within a of
  B_{n-2} and its B patch within a of A_0 (uniform A in the bond disk, uniform
  orientation over the arc that closes the ring).

Scores u_a = d ln q / d a = <-beta dU/da> over the same weighted samples, for
``a`` in (theta, E_AB).  The theta score omits the boundary term from the bond
indicator moving with theta; the E_AB score is exact.
"""
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.special import logsumexp

from base.config import (ALPHA, CENTER_RADIUS, CORE_EPSILON, CORE_EXPONENT,
                         MORSE_R_CUTOFF, MORSE_R_ONSET, PATCH_RADIUS)
from base.energy import morse_well, smooth_cutoff_weight, soft_sphere_repulsion

SCORE_PARAMS = ('theta', 'E_AB')


@dataclass(frozen=True)
class Model:
    """One point of design space plus the fixed model constants."""
    opening_angle: float                  # theta, radians
    E_AB: float
    E_AA: float = 1.0
    E_BB: float = 1.0
    kT: float = 1.0
    alpha: float = ALPHA
    eps_c: float = CORE_EPSILON
    bond_distance: float = PATCH_RADIUS

    @property
    def beta(self):
        return 1.0 / self.kT

    def replace(self, **kw):
        d = asdict(self)
        d.update(kw)
        return Model(**d)


def matching_angle(n):
    """theta_n = pi (n - 2) / n."""
    return np.pi * (n - 2) / n


# ── Pair energy (numpy, broadcasting) ──────────────────────────────────────

def _e(a):
    return np.stack([np.cos(a), np.sin(a)], axis=-1)


def patches(c, phi, theta):
    """(A, B) patch sites of particles at centres ``c`` (..., 2), orientations
    ``phi`` (...); sites on the core surface."""
    return c + CENTER_RADIUS * _e(phi), c + CENTER_RADIUS * _e(phi + theta)


def pair_energy(ci, phii, cj, phij, model):
    """Full pair energy (core + Morse on AA, BB, AB, BA), broadcasting over
    leading dimensions.  Matches ``base.energy.total_energy`` term by term."""
    th = model.opening_angle
    Ai, Bi = patches(ci, phii, th)
    Aj, Bj = patches(cj, phij, th)
    d = np.linalg.norm(ci - cj, axis=-1)
    u = soft_sphere_repulsion(d, 2.0 * CENTER_RADIUS, model.eps_c, CORE_EXPONENT)
    for P, Q, eps in ((Ai, Aj, model.E_AA), (Bi, Bj, model.E_BB),
                      (Ai, Bj, model.E_AB), (Bi, Aj, model.E_AB)):
        if eps == 0.0:
            continue
        r = np.linalg.norm(P - Q, axis=-1)
        u = u + smooth_cutoff_weight(r, MORSE_R_ONSET, MORSE_R_CUTOFF) * morse_well(r, eps, model.alpha)
    return u


def ab_bonds(ci, phii, cj, phij, model):
    """(A_i-B_j bonded, B_i-A_j bonded) at the bond distance (ADR-0002)."""
    th, a = model.opening_angle, model.bond_distance
    Ai, Bi = patches(ci, phii, th)
    Aj, Bj = patches(cj, phij, th)
    return (np.linalg.norm(Ai - Bj, axis=-1) < a), (np.linalg.norm(Bi - Aj, axis=-1) < a)


def ring_graph_ok(c, phi, model):
    """True where configurations (..., n, 2) / (..., n) labelled in cyclic
    order are exactly the n-ring of bonds B_k -> A_{k+1} (no other A-B bond)."""
    n = phi.shape[-1]
    i, j = np.triu_indices(n, k=1)
    ab, ba = ab_bonds(c[..., i, :], phi[..., i], c[..., j, :], phi[..., j], model)
    want_ba = (j == i + 1)                      # B_i - A_{i+1}
    want_ab = (i == 0) & (j == n - 1)           # A_0 - B_{n-1}
    return np.all(ba == want_ba, -1) & np.all(ab == want_ab, -1)


def score_terms(ci, phii, cj, phij, model, h=1e-5):
    """d u / d theta (central difference) and d u / d E_AB of one pair."""
    up = pair_energy(ci, phii, cj, phij, model.replace(opening_angle=model.opening_angle + h))
    um = pair_energy(ci, phii, cj, phij, model.replace(opening_angle=model.opening_angle - h))
    shape = model.replace(E_AB=1.0, E_AA=0.0, E_BB=0.0, eps_c=0.0)
    return (up - um) / (2 * h), pair_energy(ci, phii, cj, phij, shape)


def rotate(v, ang):
    c, s = np.cos(ang)[..., None], np.sin(ang)[..., None]
    return np.concatenate([c * v[..., :1] - s * v[..., 1:], s * v[..., :1] + c * v[..., 1:]], -1)


# ── Pair proposal ─────────────────────────────────────────────────────────

class PairProposal:
    """Importance proposal for the bonded pair B_k -> A_{k+1}; see module doc."""

    def __init__(self, model, n_r=40, n_psi=32, n_delta=192, uniform_frac=0.01, tilt=0.0):
        self.model = model
        a = model.bond_distance
        self.r_edges = a * np.linspace(0.0, 1.0, n_r + 1) ** 1.5
        self.psi_edges = np.linspace(-np.pi, np.pi, n_psi + 1)
        self.delta_edges = np.linspace(-np.pi, np.pi, n_delta + 1)
        self.uniform_frac = uniform_frac
        self.volume = np.pi * a * a * 2.0 * np.pi
        rc = 0.5 * (self.r_edges[1:] + self.r_edges[:-1])
        pc = 0.5 * (self.psi_edges[1:] + self.psi_edges[:-1])
        dc = 0.5 * (self.delta_edges[1:] + self.delta_edges[:-1])
        R, P, D = np.meshgrid(rc, pc, dc, indexing='ij')
        meas_r = 0.5 * np.diff(self.r_edges ** 2)
        self.log_meas = (np.log(meas_r)[:, None, None] + np.log(np.diff(self.psi_edges))[None, :, None]
                         + np.log(np.diff(self.delta_edges))[None, None, :])
        c, phi = self.relative(R, P, D)
        u = pair_energy(np.zeros(2), 0.0, c, phi, model)
        _, rev = ab_bonds(c, phi, np.zeros(2), 0.0, model)   # B_{k+1}-A_k: a 2-cycle
        self.log_boltz = np.where(rev, -np.inf, -model.beta * u)
        self.delta_centres = D
        self.set_tilt(tilt)

    def relative(self, r, psi, delta):
        """Centre and orientation of particle k+1 in the frame of particle k."""
        th = self.model.opening_angle
        pA = _e(th) + r[..., None] * _e(psi)
        phi = th + np.pi + delta
        return pA - CENTER_RADIUS * _e(phi), phi

    def set_tilt(self, lam):
        """Use one tilt, or an equal mixture of the tilts in ``lam``."""
        lams = np.atleast_1d(np.asarray(lam, float))
        self.tilt = lams
        lg = (self.log_boltz + self.log_meas)[None] + lams[:, None, None, None] * self.delta_centres[None]
        self._log_cell = lg - logsumexp(lg, axis=(1, 2, 3), keepdims=True)   # (K, cells)
        p = np.exp(self._log_cell.reshape(len(lams), -1))
        self._cell_p = p / p.sum(1, keepdims=True)

    def mean_delta(self, lam):
        lg = self.log_boltz + self.log_meas + lam * self.delta_centres
        w = np.exp(lg - logsumexp(lg))
        return float((w * self.delta_centres).sum())

    def tilt_for_ring(self, n, lam_max=40.0):
        """lambda whose tilted mean joint deviation is theta_n - theta (clipped
        to what the grid can reach)."""
        target = matching_angle(n) - self.model.opening_angle
        lo, hi = -lam_max, lam_max
        if target <= self.mean_delta(lo):
            return lo
        if target >= self.mean_delta(hi):
            return hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.mean_delta(mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def log_density(self, r, psi, delta):
        ir = np.clip(np.searchsorted(self.r_edges, r) - 1, 0, len(self.r_edges) - 2)
        ip = np.clip(np.searchsorted(self.psi_edges, psi) - 1, 0, len(self.psi_edges) - 2)
        idl = np.clip(np.searchsorted(self.delta_edges, delta) - 1, 0, len(self.delta_edges) - 2)
        lg = self._log_cell[:, ir, ip, idl] - self.log_meas[ir, ip, idl][None]
        lg = logsumexp(lg, axis=0) - np.log(len(self.tilt))
        f = self.uniform_frac
        return np.logaddexp(np.log1p(-f) + lg, np.log(f) - np.log(self.volume))

    def sample(self, shape, rng):
        """Draw relative placements: returns (c (..., 2), phi (...), log_w (...)),
        ``log_w = -beta u - log p`` (-inf where the pair is not a single bond)."""
        size = int(np.prod(shape))
        a = self.model.bond_distance
        grid = rng.random(size) >= self.uniform_frac
        comp = rng.integers(len(self.tilt), size=size)
        cell = np.empty(size, int)
        for kc in range(len(self.tilt)):
            sel = comp == kc
            cell[sel] = rng.choice(self._cell_p.shape[1], size=sel.sum(), p=self._cell_p[kc])
        ir, ip, idl = np.unravel_index(cell, self._log_cell.shape[1:])
        u1, u2, u3 = rng.random((3, size))
        re0, re1 = self.r_edges[ir] ** 2, self.r_edges[ir + 1] ** 2
        r = np.where(grid, np.sqrt(re0 + u1 * (re1 - re0)), a * np.sqrt(u1))
        psi = np.where(grid, self.psi_edges[ip] + u2 * np.diff(self.psi_edges)[ip], -np.pi + 2 * np.pi * u2)
        delta = np.where(grid, self.delta_edges[idl] + u3 * np.diff(self.delta_edges)[idl],
                         -np.pi + 2 * np.pi * u3)
        c, phi = self.relative(r, psi, delta)
        u = pair_energy(np.zeros(2), 0.0, c, phi, self.model)
        _, rev = ab_bonds(c, phi, np.zeros(2), 0.0, self.model)
        logw = np.where(rev, -np.inf, -self.model.beta * u) - self.log_density(r, psi, delta)
        return c.reshape(*shape, 2), phi.reshape(shape), logw.reshape(shape)


# ── Sequential Monte Carlo growth ─────────────────────────────────────────

def _choose(lt, rng):
    """Rosenbluth step over trials ``lt`` (M, m): log mean trial weight per
    walker and one trial picked in proportion to its weight."""
    M, m = lt.shape
    lmean = logsumexp(lt, axis=1) - np.log(m)
    alive = np.isfinite(lmean)
    prob = np.exp(lt - np.where(alive, lmean + np.log(m), 0.0)[:, None])
    prob = np.where(alive[:, None], prob, 1.0 / m)
    prob /= prob.sum(1, keepdims=True)
    pick = (prob.cumsum(1) < rng.random(M)[:, None]).sum(1).clip(max=m - 1)
    return lmean, pick


def _kish(logw):
    fin = np.isfinite(logw)
    if not fin.any():
        return 0.0
    w = np.exp(logw[fin] - logw[fin].max())
    return float(w.sum() ** 2 / (w ** 2).sum())


def _wmean(g, logw):
    fin = np.isfinite(logw)
    if not fin.any():
        return np.full(g.shape[1:], np.nan)
    w = np.exp(logw[fin] - logw[fin].max())
    return (w[:, None] * g[fin]).sum(0) / w.sum()


def _new_particle_scores(c, phi, k, cn, phin, model):
    """-beta dU/da of the pairs between new particles (M,) and particles 0..k-1."""
    dth, dE = score_terms(c[:, :k], phi[:, :k], cn[:, None], phin[:, None], model)
    return -model.beta * np.stack([dth.sum(1), dE.sum(1)], -1)


def _nonadjacent(c, phi, k, cn, phin, model):
    """Energy and forbidden-bond flag of trial particles (cn, phin) of shape
    (M, m) against particles 0..k-2 of each walker."""
    if k < 2:
        z = np.zeros(cn.shape[:-1])
        return z, np.zeros_like(z, bool)
    ci, pi = c[:, None, :k - 1], phi[:, None, :k - 1]
    cj, pj = cn[:, :, None], phin[:, :, None]
    u = pair_energy(ci, pi, cj, pj, model).sum(-1)
    b1, b2 = ab_bonds(ci, pi, cj, pj, model)
    return u, (b1 | b2).any(-1)


class RingGuide:
    """Guiding function for ring growth (twisted SMC).  After k+1 particles the
    walker is tempered by a Gaussian in the pose of particle k relative to its
    pose in the reference n-ring (regular n-gon of edge 2, orientations stepping
    by -2 pi / n, joint strain shared symmetrically).  The width grows as the
    square root of the steps left to close.  The guide only reweights
    intermediate steps; it is divided out at closure, so ring estimates stay
    unbiased whatever its widths."""

    def __init__(self, model, n, pos_width=0.25, ang_width=0.2):
        self.n = n
        th, thn = model.opening_angle, matching_angle(n)
        c = np.zeros((n, 2))
        phi = -2 * np.pi * np.arange(n) / n
        for k in range(n - 1):
            c[k + 1] = c[k] + 2 * CENTER_RADIUS * _e(phi[k] + 0.5 * (th + thn))
        self.c_ref, self.phi_ref = c, phi
        self.pos_width, self.ang_width = pos_width, ang_width

    def __call__(self, k, c, phi):
        j = self.n - k                              # steps left to return to particle 0
        dc = np.linalg.norm(c - self.c_ref[k], axis=-1)
        dphi = np.angle(np.exp(1j * (phi - self.phi_ref[k])))
        return (-0.5 * dc ** 2 / (self.pos_width ** 2 * j)
                - 0.5 * dphi ** 2 / (self.ang_width ** 2 * j))


@dataclass
class Population:
    """Weighted walkers after a growth: configurations, log weights, running
    score sums, and per-prefix estimates (index k-1 for k particles)."""
    c: np.ndarray            # (M, n, 2)
    phi: np.ndarray          # (M, n)
    logw: np.ndarray         # (M,)
    g: np.ndarray            # (M, 2) -beta dU/da of the full configuration
    logZ: np.ndarray         # (n,) ln I_chain(k) (guided target if a guide was used)
    ess: np.ndarray          # (n,) Kish ESS just before any resampling at step k
    scores: np.ndarray       # (n, 2) <-beta dU/da> for the k-prefix
    log_psi: np.ndarray = None   # (M,) guide value of the last particle


def grow_chains(model, proposal, n_particles, n_walkers, n_trials, rng, resample_frac=0.5,
                guide=None):
    """Sequential Monte Carlo growth of open chains (Rosenbluth trials within a
    step, multinomial resampling when the ESS falls below ``resample_frac``).
    Without a guide, ``exp(logZ[k-1])`` is an unbiased estimate of I_chain(k);
    with one, it estimates the guided target and only the closure is meaningful."""
    M, m = n_walkers, n_trials
    c = np.zeros((M, n_particles, 2))
    phi = np.zeros((M, n_particles))
    logw = np.zeros(M)
    g = np.zeros((M, len(SCORE_PARAMS)))
    logZ = np.full(n_particles, -np.inf)
    ess = np.zeros(n_particles)
    scores = np.zeros((n_particles, len(SCORE_PARAMS)))
    logZ[0], ess[0] = 0.0, M
    log_psi = np.zeros(M)
    rows = np.arange(M)
    for k in range(1, n_particles):
        cr, pr, lw = proposal.sample((M, m), rng)
        cn = c[:, k - 1, None] + rotate(cr, phi[:, k - 1, None])
        phin = phi[:, k - 1, None] + pr
        u, bad = _nonadjacent(c, phi, k, cn, phin, model)
        lt = np.where(bad, -np.inf, lw - model.beta * u)
        if guide is not None:
            psi_t = guide(k, cn, phin)
            lt = lt + psi_t - log_psi[:, None]
        lmean, pick = _choose(lt, rng)
        new = logw + lmean
        if not np.isfinite(new).any():
            break
        logZ[k] = logZ[k - 1] + logsumexp(new) - logsumexp(logw)
        logw = new
        c[:, k], phi[:, k] = cn[rows, pick], phin[rows, pick]
        if guide is not None:
            log_psi = psi_t[rows, pick]
        g = g + _new_particle_scores(c, phi, k, c[:, k], phi[:, k], model)
        ess[k], scores[k] = _kish(logw), _wmean(g, logw)
        if ess[k] < resample_frac * M and k < n_particles - 1:
            p = np.exp(logw - logsumexp(logw))
            idx = rng.choice(M, size=M, p=p / p.sum())
            c, phi, g, log_psi = c[idx], phi[idx], g[idx], log_psi[idx]
            logw = np.zeros(M)
    return Population(c, phi, logw, g, logZ, ess, scores, log_psi)


def _bond_radial_table(model, n_bins=200):
    """Bins of the bond length r in [0, a) and their probabilities under
    r exp(-beta E_AB morse(r)), mixed with 5% of the uniform-area law."""
    a = model.bond_distance
    edges = a * np.linspace(0.0, 1.0, n_bins + 1) ** 1.5
    rc = 0.5 * (edges[1:] + edges[:-1])
    lw = -model.beta * smooth_cutoff_weight(rc, MORSE_R_ONSET, MORSE_R_CUTOFF) * morse_well(rc, model.E_AB, model.alpha)
    w = np.exp(lw - lw.max()) * rc * np.diff(edges)
    w = 0.95 * w / w.sum() + 0.05 * np.diff(edges ** 2) / a ** 2
    return edges, w / w.sum()


def close_rings(model, pop, n_trials, rng, n_arc=64):
    """Closure step: place particle n-1 bonded to B_{n-2} and A_0 on top of the
    (n-1)-particle chains of ``pop``.  Its A patch is drawn around B_{n-2} from
    the Morse radial law; its orientation from the arc of orientations that put
    its B patch within a of A_0, weighted by the B-A_0 Morse factor on a grid of
    ``n_arc`` bins.  Returns (ln I_ring, ESS, scores)."""
    c, phi = pop.c, pop.phi
    M, k = phi.shape                               # k = n - 1 placed particles
    th, a, beta = model.opening_angle, model.bond_distance, model.beta
    s = 2.0 * CENTER_RADIUS * np.sin(th / 2.0)     # A-B distance on one particle
    _, Bprev = patches(c[:, k - 1], phi[:, k - 1], th)
    A0, _ = patches(c[:, 0], phi[:, 0], th)
    shape = (M, n_trials)

    edges, pbin = _bond_radial_table(model)
    ib = rng.choice(pbin.size, size=shape, p=pbin)
    r1 = edges[ib] + rng.random(shape) * np.diff(edges)[ib]
    log_p_pA = np.log(pbin[ib] / np.diff(edges)[ib]) - np.log(2 * np.pi * r1)
    pA = Bprev[:, None] + r1[..., None] * _e(2 * np.pi * rng.random(shape))

    vec = A0[:, None] - pA
    D = np.linalg.norm(vec, axis=-1)
    h = np.arccos(np.clip((s * s + D * D - a * a) / (2 * s * D), -1.0, 1.0))
    ok = (np.abs(D - s) < a) & (h > 0)
    g0 = np.arctan2(vec[..., 1], vec[..., 0])
    t = (np.arange(n_arc) + 0.5) / n_arc * 2 - 1                    # bin centres in [-1, 1]
    gam_c = g0[..., None] + h[..., None] * t
    r2 = np.linalg.norm(pA[..., None, :] + s * _e(gam_c) - A0[:, None, None], axis=-1)
    lw = np.where(r2 < a, -beta * morse_well(r2, model.E_AB, model.alpha), -np.inf)
    lw = lw - np.max(np.where(np.isfinite(lw), lw, -1e300), axis=-1, keepdims=True)
    w = np.exp(lw)
    w = 0.95 * w / np.maximum(w.sum(-1, keepdims=True), 1e-300) + 0.05 / n_arc
    cdf = np.cumsum(w, -1)
    j = (cdf < rng.random(shape)[..., None] * cdf[..., -1:]).sum(-1).clip(max=n_arc - 1)
    wj = np.take_along_axis(w, j[..., None], -1)[..., 0] / cdf[..., -1]
    width = 2 * h / n_arc
    gamma = g0 + h * (2 * (j + rng.random(shape)) / n_arc - 1)
    log_p_gam = np.log(np.maximum(wj, 1e-300)) - np.log(np.maximum(width, 1e-300))
    phin = gamma - (th + np.pi) / 2.0
    cn = pA - CENTER_RADIUS * _e(phin)
    log_meas = -(log_p_pA + log_p_gam)

    cp, pp = c[:, None], phi[:, None]               # (M, 1, k[, 2])
    u = pair_energy(cp, pp, cn[:, :, None], phin[:, :, None], model).sum(-1)
    prev = ab_bonds(cp[:, :, k - 1], pp[:, :, k - 1], cn, phin, model)   # (A_prev-B_new, B_prev-A_new)
    first = ab_bonds(cn, phin, cp[:, :, 0], pp[:, :, 0], model)          # (A_new-B_0, B_new-A_0)
    good = ok & prev[1] & first[1] & ~prev[0] & ~first[0]
    if k > 2:
        b1, b2 = ab_bonds(cp[:, :, 1:k - 1], pp[:, :, 1:k - 1], cn[:, :, None], phin[:, :, None], model)
        good &= ~(b1 | b2).any(-1)
    lmean, pick = _choose(np.where(good, -beta * u + log_meas, -np.inf), rng)
    if pop.log_psi is not None:
        lmean = lmean - pop.log_psi
    new = pop.logw + lmean
    if not np.isfinite(new).any():
        return -np.inf, 0.0, np.full(len(SCORE_PARAMS), np.nan)
    rows = np.arange(M)
    g = pop.g + _new_particle_scores(c, phi, k, cn[rows, pick], phin[rows, pick], model)
    logZ = pop.logZ[k - 1] + logsumexp(new) - logsumexp(pop.logw)
    return logZ, _kish(new), _wmean(g, new)


# ── Estimates ────────────────────────────────────────────────────────────

@dataclass
class SpeciesEstimate:
    kind: str                 # 'monomer' | 'chain' | 'ring'
    n: int
    lnq: float
    lnq_err: float            # standard error of ln q across independent replicas
    ess: float                # smallest per-replica Kish ESS, summed over replicas
    scores: np.ndarray = field(default_factory=lambda: np.zeros(len(SCORE_PARAMS)))
    tilt: float = 0.0

    @property
    def label(self):
        return 'monomer' if self.kind == 'monomer' else f'{self.kind}_{self.n}'


def combine_replicas(logZ, scores=None):
    """ln of the mean of the replica estimates exp(logZ_r), its standard error,
    and the Z-weighted mean of the replica scores."""
    logZ = np.asarray(logZ, float)
    R = logZ.size
    if not np.isfinite(logZ).any():
        return -np.inf, np.inf, None if scores is None else np.full(scores.shape[1:], np.nan)
    mx = logZ[np.isfinite(logZ)].max()
    w = np.exp(logZ - mx)
    err = w.std(ddof=1) / np.sqrt(R) / w.mean() if R > 1 else np.nan
    sc = None
    if scores is not None:
        fin = np.isfinite(logZ) & np.all(np.isfinite(scores), axis=-1)
        sc = (w[fin, None] * scores[fin]).sum(0) / w[fin].sum()
    return float(np.log(w.mean()) + mx), float(err), sc


def chain_tilt_targets(model, n_chain_max, n_tilted=4):
    """Ring sizes whose matching joint the tilted chain populations aim at:
    spread between one past the matching ring size m and n_chain_max + 1, since
    a chain longer than m must open its joints to avoid itself."""
    m = 2 * np.pi / (np.pi - model.opening_angle)
    lo = int(np.floor(m)) + 1
    if lo > n_chain_max + 1:
        return []
    return sorted({int(round(x)) for x in np.geomspace(lo, n_chain_max + 1, n_tilted)})


def estimate_chains(model, n_chain_max, n_walkers=4000, n_trials=16, n_replicas=4,
                    rng=None, n_tilted=4, tilt_spread=4.0):
    """Chains 2..n_chain_max.  One untilted SMC population plus up to
    ``n_tilted`` populations whose joints are tilted open toward larger rings
    (:func:`chain_tilt_targets`); each is unbiased, and every chain length takes
    the population with the largest ESS, so chains that curl into themselves
    are sampled by a population that keeps them open."""
    rng = np.random.default_rng(rng)
    prop = PairProposal(model)
    lams = [0.0] + [prop.tilt_for_ring(n) for n in chain_tilt_targets(model, n_chain_max, n_tilted)]
    per_pop = []
    for lam in lams:
        mix = [-tilt_spread, 0.0, 0.0, tilt_spread] if lam == 0.0 else [lam - tilt_spread, lam, lam, lam + tilt_spread]
        prop.set_tilt(mix)
        pops = [grow_chains(model, prop, n_chain_max, n_walkers // n_replicas, n_trials, rng)
                for _ in range(n_replicas)]
        per_pop.append(pops)
    out = []
    for k in range(2, n_chain_max + 1):
        best = None
        for lam, pops in zip(lams, per_pop):
            ess = min(p.ess[1:k].min() for p in pops) * n_replicas
            if best is None or ess > best[0]:
                best = (ess, lam, pops)
        ess, lam, pops = best
        lnI, err, sc = combine_replicas([p.logZ[k - 1] for p in pops],
                                        np.array([p.scores[k - 1] for p in pops]))
        out.append(SpeciesEstimate('chain', k, np.log(2 * np.pi) + lnI, err, ess, sc, lam))
    return out


def estimate_ring(model, n, n_walkers=4000, n_trials=16, n_close=32, n_replicas=4,
                  rng=None, proposal=None, tilt_spread=4.0, guided=True):
    """ln q of the n-ring (sigma = n)."""
    rng = np.random.default_rng(rng)
    prop = PairProposal(model) if proposal is None else proposal
    lam = prop.tilt_for_ring(n)
    prop.set_tilt([lam - tilt_spread, lam, lam, lam + tilt_spread])
    guide = RingGuide(model, n) if guided else None
    lnZ, ess, sc = [], [], []
    for _ in range(n_replicas):
        pop = grow_chains(model, prop, n - 1, n_walkers // n_replicas, n_trials, rng,
                          guide=guide)
        z, e, s = close_rings(model, pop, n_close, rng)
        lnZ.append(z)
        ess.append(min(e, pop.ess[1:].min()) if n > 2 else e)
        sc.append(s)
    lnI, err, scores = combine_replicas(lnZ, np.array(sc))
    return SpeciesEstimate('ring', n, np.log(2 * np.pi / n) + lnI, err,
                           float(np.sum(ess)), scores, lam)


def estimate_species(model, n_chain_max=24, n_ring_max=12, n_walkers=4000, n_trials=16,
                     n_close=32, n_replicas=4, seed=0):
    """Monomer, chains 2..n_chain_max and rings 3..n_ring_max at one design point."""
    rng = np.random.default_rng(seed)
    out = [SpeciesEstimate('monomer', 1, float(np.log(2 * np.pi)), 0.0, float(n_walkers))]
    out += estimate_chains(model, n_chain_max, n_walkers, n_trials, n_replicas, rng)
    prop = PairProposal(model)
    for n in range(3, n_ring_max + 1):
        out.append(estimate_ring(model, n, n_walkers, n_trials, n_close, n_replicas, rng, prop))
    return out


def species_arrays(estimates):
    """Columns of a species list, for saving: kind, n, lnq, lnq_err, ess, scores, tilt."""
    return dict(
        kind=np.array([e.kind for e in estimates]),
        n=np.array([e.n for e in estimates]),
        lnq=np.array([e.lnq for e in estimates]),
        lnq_err=np.array([e.lnq_err for e in estimates]),
        ess=np.array([e.ess for e in estimates]),
        scores=np.array([e.scores for e in estimates]),
        tilt=np.array([e.tilt for e in estimates]),
    )
