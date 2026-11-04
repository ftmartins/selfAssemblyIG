"""
Mode picture of an n-ring (docs/ring_yield_plan.md sections 1 and 3).

Relax the ring, take out the rigid-body motions with their analytic
generators, split the internal Hessian modes into the n-3 soft ones (one per
pin-joint mechanism) and the stiff rest, and write

    Z_n = e^{-beta E_0} A (J~/sigma_n) prod_stiff sqrt(2 pi / beta lambda_i) I_soft e^{n b_n},

with J~ = 2 pi sqrt(det G) (G the Gram matrix of the generators at q0) and
I_soft the integral of e^{-beta (U - E_0)} over the soft-mode subspace through
q0.  b_n is the per-bond residual against the sampled ln q_n of
``analysis.partition``:

    n b_n = ln q_n - [-beta E_0 + ln(J~/sigma_n) + sum_stiff 1/2 ln(2 pi / beta lambda_i) + ln I_soft].

Filters (each recorded, a failing ring is marked ``valid = False``): no
negative internal mode beyond ``neg_tol`` of the largest; the relaxed state is
exactly the n-ring of the bond graph; symmetric starts are perturbed so a
saddle can relax off it.

Same Hamiltonian as MD (``base.energy.total_energy`` with the shared core
form).  State vector q = [x0, y0, phi0, x1, ...].
"""
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from base.config import CENTER_RADIUS
from base.energy import total_energy
from analysis.partition import Model, RingGuide, ring_graph_ok, matching_angle


def energy_fn(model):
    """Jitted U(q) for the free-space cluster at ``model``."""
    def U(q):
        return total_energy(q, model.opening_angle, model.E_AB, model.alpha,
                            E_AA=model.E_AA, E_BB=model.E_BB, r_patch=CENTER_RADIUS,
                            rep_A=model.eps_c)
    return jax.jit(U)


def reference_ring(model, n):
    """Regular n-gon start (edge 2, symmetric joint strain), as q (3n,)."""
    g = RingGuide(model, n)
    c = g.c_ref - g.c_ref.mean(0)
    return np.column_stack([c, g.phi_ref]).ravel()


def rigid_generators(q):
    """Rows t_x, t_y and the rotation about the origin, (3, 3n)."""
    q = np.asarray(q).reshape(-1, 3)
    n = len(q)
    T = np.zeros((3, n, 3))
    T[0, :, 0] = 1.0
    T[1, :, 1] = 1.0
    T[2, :, 0] = -q[:, 1]
    T[2, :, 1] = q[:, 0]
    T[2, :, 2] = 1.0
    return T.reshape(3, 3 * n)


def jacobian_factor(q):
    """J~ = 2 pi sqrt(det G), G = T T^T."""
    T = rigid_generators(q)
    return 2 * np.pi * np.sqrt(np.linalg.det(T @ T.T))


def internal_basis(q):
    """Orthonormal basis (3n, 3n-3) of the complement of the rigid generators."""
    T = rigid_generators(q)
    Qr, _ = np.linalg.qr(T.T)
    P = np.eye(T.shape[1]) - Qr @ Qr.T
    w, V = np.linalg.eigh(P)
    return V[:, w > 0.5]


@dataclass
class RingModes:
    n: int
    model: Model
    q0: np.ndarray
    E0: float
    J_tilde: float
    evals: np.ndarray          # internal Hessian eigenvalues, ascending (3n-3,)
    evecs: np.ndarray          # (3n, 3n-3) in full coordinates
    n_soft: int
    valid: bool
    reasons: list = field(default_factory=list)
    grad_norm: float = np.nan

    @property
    def sigma(self):
        return self.n

    @property
    def soft_evals(self):
        return self.evals[:self.n_soft]

    @property
    def stiff_evals(self):
        return self.evals[self.n_soft:]

    def ln_q_harmonic(self):
        """Full Laplace (all internal modes Gaussian), ln(Z / A)."""
        b = self.model.beta
        if self.evals[0] <= 0:
            return np.inf
        return float(-b * self.E0 + np.log(self.J_tilde / self.sigma)
                     + 0.5 * np.sum(np.log(2 * np.pi / (b * self.evals))))

    def ln_q_stiff(self):
        """Everything but the soft integral: -beta E0 + ln(J~/sigma) + stiff Gaussians."""
        b = self.model.beta
        return float(-b * self.E0 + np.log(self.J_tilde / self.sigma)
                     + 0.5 * np.sum(np.log(2 * np.pi / (b * self.stiff_evals))))


def relax_ring(model, n, perturb=1e-3, seed=0, gtol=1e-10):
    """Relax the perturbed reference n-ring; returns (q0, E0, |grad|)."""
    U = energy_fn(model)
    vg = jax.jit(jax.value_and_grad(U))
    hess = jax.jit(jax.hessian(U))
    rng = np.random.default_rng(seed)
    q = reference_ring(model, n) + perturb * rng.standard_normal(3 * n)

    def f(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g)

    res = minimize(f, q, jac=True, method='L-BFGS-B', options=dict(maxiter=20000, gtol=gtol, ftol=1e-15))
    q = res.x
    B = internal_basis(q)
    for _ in range(20):                         # Newton polish in the internal subspace
        g = np.asarray(vg(jnp.asarray(q))[1])
        if np.linalg.norm(g) < gtol:
            break
        H = B.T @ np.asarray(hess(jnp.asarray(q))) @ B
        w, V = np.linalg.eigh(H)
        if w[0] <= 0:
            break
        step = B @ (V @ ((V.T @ (B.T @ g)) / w))
        q_new = q - step
        if float(U(jnp.asarray(q_new))) > float(U(jnp.asarray(q))) + 1e-12:
            break
        q = q_new
        B = internal_basis(q)
    E0, g = vg(jnp.asarray(q))
    return q, float(E0), float(np.linalg.norm(np.asarray(g)))


def analyze_ring(model, n, neg_tol=1e-5, **relax_kw):
    """Relax and decompose one ring; see module docstring."""
    q0, E0, gnorm = relax_ring(model, n, **relax_kw)
    U = energy_fn(model)
    H = np.asarray(jax.hessian(U)(jnp.asarray(q0)))
    B = internal_basis(q0)
    w, V = np.linalg.eigh(B.T @ H @ B)
    reasons = []
    if w[0] < -neg_tol * abs(w[-1]):
        reasons.append(f'saddle: lambda_min = {w[0]:.3g}')
    qr = q0.reshape(n, 3)
    if not ring_graph_ok(qr[:, :2], qr[:, 2], model):
        reasons.append('bond graph is not the n-ring')
    return RingModes(n, model, q0, E0, jacobian_factor(q0), w, B @ V, max(n - 3, 0),
                     not reasons, reasons, gnorm)


# ── Soft-mode integral ─────────────────────────────────────────────────

def _line_extent(U, q0, v, E0, beta, rise=15.0, s_max=3.0):
    """Distance along +v and -v at which beta (U - E0) first exceeds ``rise``."""
    out = []
    for sgn in (1.0, -1.0):
        lo, hi = 0.0, s_max
        if beta * (float(U(jnp.asarray(q0 + sgn * hi * v))) - E0) < rise:
            out.append(hi)
            continue
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if beta * (float(U(jnp.asarray(q0 + sgn * mid * v))) - E0) < rise:
                lo = mid
            else:
                hi = mid
        out.append(hi)
    return np.array(out)


def soft_integral(modes, n_samples=200_000, seed=0, df=4.0):
    """ln I_soft = ln of the integral of e^{-beta (U - E0)} 1[n-ring] over the
    soft subspace q0 + V_soft s, by importance sampling with a Student-t whose
    scales come from line scans along each soft mode.  Returns
    (ln I_soft, standard error, ESS)."""
    d = modes.n_soft
    if d == 0:
        return 0.0, 0.0, float(n_samples)
    model, beta = modes.model, modes.model.beta
    U = energy_fn(model)
    Uv = jax.jit(jax.vmap(U))
    Vs = modes.evecs[:, :d]
    ext = np.array([_line_extent(U, modes.q0, Vs[:, i], modes.E0, beta) for i in range(d)])
    scale = ext.mean(1) / 2.0
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df, size=(n_samples, d))
    s = z * scale
    from scipy.stats import t as student_t
    log_p = student_t.logpdf(z, df).sum(1) - np.log(scale).sum()
    q = modes.q0[None] + s @ Vs.T
    E = np.asarray(Uv(jnp.asarray(q)))
    qr = q.reshape(n_samples, modes.n, 3)
    ok = ring_graph_ok(qr[..., :2], qr[..., 2], model)
    logw = np.where(ok & np.isfinite(E), -beta * (E - modes.E0), -np.inf) - log_p
    fin = np.isfinite(logw)
    if not fin.any():
        return -np.inf, np.inf, 0.0
    mx = logw[fin].max()
    w = np.where(fin, np.exp(logw - mx), 0.0)
    mean = w.mean()
    return (float(np.log(mean) + mx), float(w.std(ddof=1) / np.sqrt(n_samples) / mean),
            float(w.sum() ** 2 / (w ** 2).sum()))


@dataclass
class Breakdown:
    n: int
    ln_q: float                # sampled (partition.py)
    neg_beta_E0: float
    ln_J_over_sigma: float
    ln_stiff: float            # sum over stiff modes of 1/2 ln(2 pi / beta lambda)
    ln_I_soft: float
    ln_soft_gaussian: float    # what the Gaussian would have given for the soft modes
    b_n: float                 # per-bond residual

    def as_dict(self):
        return self.__dict__.copy()


def decompose(modes, ln_q_sampled, ln_I_soft):
    b = modes.model.beta
    stiff = 0.5 * float(np.sum(np.log(2 * np.pi / (b * modes.stiff_evals))))
    soft = modes.soft_evals
    soft_g = 0.5 * float(np.sum(np.log(2 * np.pi / (b * soft)))) if np.all(soft > 0) else np.inf
    parts = -b * modes.E0 + np.log(modes.J_tilde / modes.sigma) + stiff + ln_I_soft
    return Breakdown(modes.n, ln_q_sampled, -b * modes.E0, float(np.log(modes.J_tilde / modes.sigma)),
                     stiff, ln_I_soft, soft_g, float((ln_q_sampled - parts) / modes.n))


def mode_summary(modes):
    """Plain dict for saving: E0, J~, eigenvalues, validity."""
    return dict(n=modes.n, E0=modes.E0, J_tilde=modes.J_tilde, evals=modes.evals,
                n_soft=modes.n_soft, valid=modes.valid, reasons='; '.join(modes.reasons),
                grad_norm=modes.grad_norm, ln_q_harmonic=modes.ln_q_harmonic()
                if modes.valid else np.nan, matching_angle=matching_angle(modes.n))
