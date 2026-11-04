"""
Independent check of ring ln q_n by Cartesian importance sampling
(docs/ring_yield_plan.md, validation step 2).

Particle 0 is fixed at the origin with phi_0 = 0 (gauge fixing: the rigid
motions give exactly A * 2 pi), and the other 3(n-1) coordinates are sampled
from a multivariate Student-t around the relaxed ring of
``analysis.ring_modes``.  The proposal starts from the gauge-fixed Hessian
(eigenvalues floored, so soft modes get a finite width) and is refitted to the
weighted samples for a few rounds.  No code is shared with the chain-growth
sampler of ``analysis.partition`` beyond the energy and the bond-graph test.

    q_n = (2 pi / n) * integral dx e^{-beta U(x)} 1[n-ring].
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import logsumexp, gammaln

from analysis.partition import pair_energy, ring_graph_ok
from analysis.ring_modes import energy_fn, relax_ring


def gauge_fix(q):
    """Shift/rotate a ring state so particle 0 sits at the origin with phi = 0."""
    q = np.asarray(q, float).reshape(-1, 3).copy()
    c0, p0 = q[0, :2].copy(), q[0, 2]
    R = np.array([[np.cos(-p0), -np.sin(-p0)], [np.sin(-p0), np.cos(-p0)]])
    q[:, :2] = (q[:, :2] - c0) @ R.T
    q[:, 2] -= p0
    return q


def _energies(x, n, model):
    """U of gauge-fixed samples x (M, 3(n-1)), plus the n-ring indicator."""
    M = x.shape[0]
    q = np.concatenate([np.zeros((M, 1, 3)), x.reshape(M, n - 1, 3)], 1)
    c, phi = q[..., :2], q[..., 2]
    i, j = np.triu_indices(n, k=1)
    U = pair_energy(c[:, i], phi[:, i], c[:, j], phi[:, j], model).sum(1)
    return U, ring_graph_ok(c, phi, model)


def _mvt_sample(mean, cov, df, M, rng):
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((M, len(mean)))
    u = rng.chisquare(df, size=M) / df
    return mean + (z @ L.T) / np.sqrt(u)[:, None]


def _mvt_logpdf(x, mean, cov, df):
    d = len(mean)
    L = np.linalg.cholesky(cov)
    y = np.linalg.solve(L, (x - mean).T).T
    maha = np.sum(y * y, 1)
    return (gammaln((df + d) / 2) - gammaln(df / 2) - 0.5 * d * np.log(df * np.pi)
            - np.log(np.diag(L)).sum() - 0.5 * (df + d) * np.log1p(maha / df))


def direct_ln_q(model, n, n_samples=200_000, n_rounds=4, df=5.0, eig_floor=4.0,
                seed=0, batch=50_000):
    """ln q_n, its standard error and the ESS of the final round."""
    rng = np.random.default_rng(seed)
    q0, E0, _ = relax_ring(model, n, seed=seed)
    x0 = gauge_fix(q0)[1:].ravel()
    U = energy_fn(model)
    Hfull = np.asarray(jax.hessian(U)(jnp.asarray(gauge_fix(q0).ravel())))
    H = model.beta * Hfull[3:, 3:]
    w, V = np.linalg.eigh(H)
    cov = (V / np.maximum(w, eig_floor)) @ V.T
    mean = x0
    for r in range(n_rounds):
        M = n_samples if r == n_rounds - 1 else max(n_samples // 4, 20_000)
        logw = np.empty(M)
        xs = np.empty((M, len(mean)))
        for s in range(0, M, batch):
            x = _mvt_sample(mean, cov, df, min(batch, M - s), rng)
            E, ok = _energies(x, n, model)
            logw[s:s + len(x)] = np.where(ok, -model.beta * E, -np.inf) - _mvt_logpdf(x, mean, cov, df)
            xs[s:s + len(x)] = x
        if r == n_rounds - 1:
            break
        fin = np.isfinite(logw)
        p = np.exp(logw[fin] - logsumexp(logw[fin]))
        mean = p @ xs[fin]
        dx = xs[fin] - mean
        cov = 1.5 * (dx * p[:, None]).T @ dx + 1e-10 * np.eye(len(mean))
    fin = np.isfinite(logw)
    mx = logw[fin].max()
    wts = np.where(fin, np.exp(logw - mx), 0.0)
    m = wts.mean()
    ln_q = np.log(2 * np.pi / n) + np.log(m) + mx
    return float(ln_q), float(wts.std(ddof=1) / np.sqrt(M) / m), float(wts.sum() ** 2 / (wts ** 2).sum())
