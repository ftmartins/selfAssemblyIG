"""
Fisher metric of the equilibrium yields over design parameters
(docs/ring_yield_plan.md section 4).

Inputs are the species scores u_{a,s} = d ln q_s / d a (from
``analysis.partition``, ensemble averages <-beta dU/da>) and the yields of
``analysis.mass_action`` at fixed density.  At fixed rho the fugacity moves:

    d_a ln z = - sum_s n_s rho_s u_{a,s} / sum_s n_s^2 rho_s
    v_{a,s}  = u_{a,s} + n_s d_a ln z

* :func:`count_metric` — g_ab = A sum_s rho_s v_a,s v_b,s (Poisson counts at
  fixed density, per unit area unless ``area`` is given).
* :func:`cluster_metric` — g_ab = sum_s p_s d_a ln p_s d_b ln p_s for the
  cluster yields p_s = rho_s / sum rho, d_a ln p_s = v_a,s - sum_t p_t v_a,t.
"""
import numpy as np


def _aligned_scores(table, yields):
    idx = np.array([table.index(l) for l in yields.labels])
    u = np.nan_to_num(table.scores[idx], nan=0.0)
    u[yields.n == 1] = 0.0                    # q_1 = 2 pi does not depend on the design
    return u


def fugacity_response(table, yields):
    """d_a ln z at fixed density, shape (P,)."""
    u = _aligned_scores(table, yields)
    n, rho = yields.n, yields.rho
    return -(n * rho) @ u / np.sum(n * n * rho)


def total_scores(table, yields):
    """v_{a,s} = u_{a,s} + n_s d_a ln z, shape (S, P)."""
    return _aligned_scores(table, yields) + yields.n[:, None] * fugacity_response(table, yields)[None]


def count_metric(table, yields, area=1.0):
    v = total_scores(table, yields)
    return area * (v * yields.rho[:, None]).T @ v


def cluster_metric(table, yields):
    v = total_scores(table, yields)
    p = yields.cluster
    dlnp = v - p @ v
    return (dlnp * p[:, None]).T @ dlnp
