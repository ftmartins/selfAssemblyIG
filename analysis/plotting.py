"""
All matplotlib figure functions for the self-assembly analysis.

Each function takes pre-computed data and (optionally) writes a PNG under
``base.paths.PLOTS_DIR``.  No analysis logic here — computation lives in
:mod:`analysis.mu_scan`, :mod:`analysis.yields`, :mod:`analysis.gcmc_runs`.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from base.geometry import patch_positions


# ── GCMC mu-scan two-panel ────────────────────────────────────────────────

def plot_mu_scan(agg, target_phi, N_target, mu_star, mu_star_interp,
                 box_area, r_center, out_path=None):
    """Two stacked panels: <phi> vs mu (top) and <N> vs mu (bottom), with
    +/- 1 sigma bands and the mu* markers.  ``agg`` is the dict from
    :func:`analysis.mu_scan.aggregate_by_mu`."""
    mu = agg['mu']
    fig, (ax_phi, ax_N) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax_phi.errorbar(mu, agg['mean_phi'], yerr=agg['sem_phi'], fmt='o-', capsize=5,
                    color='steelblue', label=r'$\langle\phi\rangle\pm$ SEM')
    ax_phi.fill_between(mu, agg['mean_phi'] - agg['std_phi'],
                        agg['mean_phi'] + agg['std_phi'], alpha=0.15, color='steelblue')
    ax_phi.axhline(target_phi, color='red', ls='--', lw=1.5, label=f'target $\\phi$={target_phi}')
    ax_phi.axvline(mu_star, color='green', ls=':', lw=2.0, label=f'$\\mu^*$={mu_star:.3f}')
    if abs(mu_star_interp - mu_star) > 1e-4:
        ax_phi.axvline(mu_star_interp, color='limegreen', ls='--', lw=1.5,
                       label=f'$\\mu^*$ (interp)={mu_star_interp:.4f}')
    ax_phi.set_ylabel(r'Area fraction $\phi$')
    ax_phi.legend(fontsize=9)
    ax_phi.grid(alpha=0.3)
    ax_phi.set_title(f'GCMC $\\mu$-scan  |  $A={box_area:.0f}$  $r={r_center}$')

    ax_N.errorbar(mu, agg['mean_N'], yerr=agg['sem_N'], fmt='s-', capsize=5,
                  color='darkorange', label=r'$\langle N\rangle\pm$ SEM')
    ax_N.fill_between(mu, agg['mean_N'] - agg['std_N'], agg['mean_N'] + agg['std_N'],
                     alpha=0.15, color='darkorange')
    ax_N.axhline(N_target, color='red', ls='--', lw=1.5, label=f'$N_{{target}}$={N_target:.1f}')
    ax_N.axvline(mu_star, color='green', ls=':', lw=2.0)
    ax_N.set_xlabel(r'Chemical potential $\mu / k_BT$')
    ax_N.set_ylabel(r'$\langle N\rangle$')
    ax_N.legend(fontsize=9)
    ax_N.grid(alpha=0.3)

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f'Plot saved: {out_path}')
    return fig


# ── Phase-diagram heatmap ─────────────────────────────────────────────────

def plot_phase_diagram(x_vals, y_vals, Z, xlabel, ylabel, title,
                       cmap='viridis', vmin=None, vmax=None, ax=None, out_path=None):
    """Single heatmap of ``Z`` (shape ``(len(y_vals), len(x_vals))``) with real
    axis tick labels."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Z, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        ax.figure.savefig(out_path, dpi=150)
        print(f'Plot saved: {out_path}')
    return ax


# ── Configuration snapshot ───────────────────────────────────────────────

def plot_configuration(q, L, params, ax, title='', r_c=1.0):
    """Draw a patchy-particle configuration: blue cores, red patch A, green
    patch B.  ``q`` is (N, 3) = [x, y, theta]; ``params`` is a make_gcmc_params dict."""
    from matplotlib.patches import Circle

    ax.set_xlim(-L / 2 - 0.5, L / 2 + 0.5)
    ax.set_ylim(-L / 2 - 0.5, L / 2 + 0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.add_patch(plt.Rectangle((-L / 2, -L / 2), L, L, fill=False,
                               edgecolor='dimgray', lw=1.2))
    if len(q) == 0:
        ax.text(0, 0, 'N=0', ha='center', va='center', color='gray')
        ax.set_title(title, fontsize=8)
        return

    q = np.asarray(q)
    for xi, yi, _ in q:
        ax.add_patch(Circle((xi, yi), r_c, color='steelblue', alpha=0.4, zorder=2))
    pA, pB = patch_positions(q[:, :2], q[:, 2], params['opening_angle'], params['r_patch'])
    ax.scatter(pA[:, 0], pA[:, 1], s=20, c='red', zorder=3, linewidths=0)
    ax.scatter(pB[:, 0], pB[:, 1], s=20, c='green', zorder=3, linewidths=0)
    ax.set_title(f'{title}  N={len(q)}', fontsize=8, pad=2)


# ── Time series ──────────────────────────────────────────────────────────

def plot_timeseries(N_traj, E_traj=None, snapshot_sweeps=None, out_path=None):
    """N (and optionally E) vs production sweep."""
    fig, axes = plt.subplots(2 if E_traj is not None else 1, 1,
                             figsize=(8, 6), sharex=True, squeeze=False)
    axes[0][0].plot(np.arange(len(N_traj)), N_traj, color='darkorange')
    axes[0][0].set_ylabel('N')
    axes[0][0].grid(alpha=0.3)
    if E_traj is not None:
        x = snapshot_sweeps if snapshot_sweeps is not None else np.arange(len(E_traj))
        axes[1][0].plot(x, E_traj, color='steelblue')
        axes[1][0].set_ylabel('E')
        axes[1][0].grid(alpha=0.3)
    axes[-1][0].set_xlabel('production sweep')
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=150)
    return fig
