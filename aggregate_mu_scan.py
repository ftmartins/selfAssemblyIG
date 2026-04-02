"""
aggregate_mu_scan.py — aggregate results from a GCMC mu-scan SLURM array job.

For each NPZ file produced by mu_scan_gcmc.py:
  1. Equilibration check: Mann-Whitney U test comparing the first 25% vs last
     50% of the N trajectory.  If p < alpha, the run is flagged as
     not-equilibrated and excluded from the mean.
  2. Computes <N> +/- std and <phi> +/- std per mu from equilibrated runs.
  3. Finds mu* = argmin |<phi> - target_phi|, with linear interpolation.
  4. Prints a summary table and saves a two-panel plot:
       top panel:    phi vs mu with error bars
       bottom panel: <N> vs mu with error bars

Usage
-----
python aggregate_mu_scan.py \\
    --input_dir results/mu_scan \\
    --plot_output results/plots/mu_scan_aggregated.png

Defaults for box_area, target_phi, r_center, etc. come from config_gcmc.py.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from config_gcmc import (
    MU_SCAN_DIR,
    PLOTS_DIR,
    DEFAULT_BOX_AREA,
    DEFAULT_TARGET_PHI,
    DEFAULT_R_CENTER,
    DEFAULT_ALPHA_EQUIL,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Aggregate GCMC mu-scan results and find mu* for target phi.\n\n'
                    'Filters files to a fixed (E_AB, opening_angle) combination before\n'
                    'computing statistics, so mixed-parameter scan directories work correctly.'
    )
    p.add_argument('--input_dir',      type=str,
                   default=MU_SCAN_DIR,
                   help=f'Directory with NPZ files from mu_scan_gcmc.py  '
                        f'(default: {MU_SCAN_DIR})')
    # Filter to a single parameter combination
    p.add_argument('--E_AB',           type=float, default=None,
                   help='Only include files with this E_AB value  (required when '
                        'the directory contains multiple E_AB values)')
    p.add_argument('--opening_angle',  type=float, default=None,
                   help='Only include files with this opening angle in radians  '
                        '(required when directory contains multiple angles)')
    p.add_argument('--tol',            type=float, default=1e-3,
                   help='Tolerance for matching E_AB / opening_angle in NPZ metadata  '
                        '(default: 1e-3)')
    # Box / density
    p.add_argument('--box_area',    type=float, default=DEFAULT_BOX_AREA,
                   help=f'Box area A = L^2  (default: {DEFAULT_BOX_AREA})')
    p.add_argument('--target_phi',  type=float, default=DEFAULT_TARGET_PHI,
                   help=f'Target area fraction  (default: {DEFAULT_TARGET_PHI})')
    p.add_argument('--r_center',    type=float, default=DEFAULT_R_CENTER,
                   help=f'Particle core radius  (default: {DEFAULT_R_CENTER})')
    p.add_argument('--alpha_equil', type=float, default=DEFAULT_ALPHA_EQUIL,
                   help=f'p-value threshold for equilibration check  '
                        f'(default: {DEFAULT_ALPHA_EQUIL})')
    p.add_argument('--plot_output', type=str,   default=None,
                   help='Output path for the plot  '
                        '(default: auto-named from E_AB + opening_angle in PLOTS_DIR)')
    p.add_argument('--no_plot',     action='store_true',
                   help='Skip plotting (useful in headless environments)')
    return p.parse_args()


def is_equilibrated(N_traj: np.ndarray, alpha: float = 0.05):
    """
    Test whether the N trajectory has equilibrated by comparing the first 25%
    vs the last 50% of sweeps using the Mann-Whitney U test (non-parametric,
    no normality assumption needed for integer particle counts).

    Returns
    -------
    (equilibrated: bool, p_value: float)
    """
    n = len(N_traj)
    if n < 8:
        return True, 1.0   # too short to test; assume OK

    first = N_traj[:n // 4]
    last  = N_traj[n // 2:]

    _, p = mannwhitneyu(first, last, alternative='two-sided')
    return bool(p >= alpha), float(p)


def N_to_phi(N, box_area, r_center):
    """Convert particle count N to area fraction phi."""
    return N * np.pi * r_center ** 2 / box_area


def main():
    args = parse_args()

    box_area  = args.box_area
    r_center  = args.r_center
    N_target  = args.target_phi * box_area / (np.pi * r_center ** 2)
    phi_scale = np.pi * r_center ** 2 / box_area

    # Auto-name plot if not specified
    if args.plot_output is None:
        tag = ''
        if args.E_AB is not None:
            tag += f'_EAB{args.E_AB:.3f}'
        if args.opening_angle is not None:
            tag += f'_oa{np.degrees(args.opening_angle):.2f}deg'
        args.plot_output = os.path.join(PLOTS_DIR, f'mu_scan_aggregated{tag}.png')

    print('aggregate_mu_scan.py')
    print(f'  input_dir    = {args.input_dir}')
    print(f'  filter E_AB  = {args.E_AB if args.E_AB is not None else "any"}')
    if args.opening_angle is not None:
        print(f'  filter oa    = {np.degrees(args.opening_angle):.2f} deg')
    else:
        print(f'  filter oa    = any')
    print(f'  box_area     = {box_area:.0f}   r_center={r_center}')
    print(f'  target_phi   = {args.target_phi}  =>  N_target = {N_target:.2f}')
    print(f'  equil alpha  = {args.alpha_equil}')
    print()

    npz_files = sorted(Path(args.input_dir).glob('*.npz'))
    if not npz_files:
        print(f'ERROR: no NPZ files found in {args.input_dir}')
        sys.exit(1)

    print(f'Found {len(npz_files)} NPZ file(s) total.\n')

    # Group by mu value; filter to the requested (E_AB, opening_angle) combination
    mu_groups   = defaultdict(list)
    skipped     = []
    n_filtered  = 0

    for f in npz_files:
        try:
            data = np.load(f)
        except Exception as e:
            print(f'  WARNING: could not load {f.name}: {e}')
            continue

        # ── Parameter filter ──────────────────────────────────────────────────
        if args.E_AB is not None:
            file_eab = float(data['E_AB'][0]) if 'E_AB' in data.files else None
            if file_eab is None or abs(file_eab - args.E_AB) > args.tol:
                n_filtered += 1
                continue
        if args.opening_angle is not None:
            file_oa = float(data['opening_angle'][0]) if 'opening_angle' in data.files else None
            if file_oa is None or abs(file_oa - args.opening_angle) > args.tol:
                n_filtered += 1
                continue

        mu_val = float(data['mu'][0])
        N_traj = data['N_traj']

        eq, p_val = is_equilibrated(N_traj, alpha=args.alpha_equil)

        if not eq:
            skipped.append((f.name, mu_val, p_val))
            print(f'  NOT EQUILIBRATED: {f.name}  mu={mu_val:.4f}  p={p_val:.4f}')
            continue

        mu_groups[mu_val].append(N_traj)

    if n_filtered:
        print(f'{n_filtered} file(s) skipped (parameter mismatch).')
    if skipped:
        print(f'{len(skipped)} file(s) excluded (not equilibrated).')

    if not mu_groups:
        print('\nERROR: no equilibrated runs found. '
              'Consider lowering --alpha_equil or running longer simulations.')
        sys.exit(1)

    # Compute statistics per mu
    mu_sorted = sorted(mu_groups.keys())
    mean_N    = []
    std_N     = []
    sem_N     = []   # standard error of the mean (for error bars)
    mean_phi  = []
    std_phi   = []
    sem_phi   = []
    n_sweeps  = []   # total number of production sweeps pooled

    for mu_val in mu_sorted:
        all_N = np.concatenate(mu_groups[mu_val])
        n     = len(all_N)

        mn  = float(np.mean(all_N))
        sd  = float(np.std(all_N))
        se  = sd / np.sqrt(n)

        mean_N.append(mn)
        std_N.append(sd)
        sem_N.append(se)
        mean_phi.append(mn * phi_scale)
        std_phi.append(sd * phi_scale)
        sem_phi.append(se * phi_scale)
        n_sweeps.append(n)

    mean_N   = np.array(mean_N)
    std_N    = np.array(std_N)
    sem_N    = np.array(sem_N)
    mean_phi = np.array(mean_phi)
    std_phi  = np.array(std_phi)
    sem_phi  = np.array(sem_phi)

    # Find mu* (grid)
    idx_star = int(np.argmin(np.abs(mean_phi - args.target_phi)))
    mu_star  = mu_sorted[idx_star]

    # Linear interpolation for sub-grid mu*
    mu_star_interp = float(mu_star)
    if 0 < idx_star < len(mu_sorted) - 1:
        lo = idx_star - 1 if mean_phi[idx_star] > args.target_phi else idx_star
        hi = lo + 1
        if hi < len(mu_sorted):
            dphi = mean_phi[hi] - mean_phi[lo]
            if abs(dphi) > 1e-9:
                t = (args.target_phi - mean_phi[lo]) / dphi
                mu_star_interp = mu_sorted[lo] + t * (mu_sorted[hi] - mu_sorted[lo])

    # ── Summary table ──────────────────────────────────────────────────────────
    print()
    hdr = f'{"mu":>7}  {"<phi>":>7}  {"std_phi":>8}  {"<N>":>7}  {"std_N":>7}  {"sweeps":>7}'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for i, mu_val in enumerate(mu_sorted):
        marker = '  <-- mu*' if i == idx_star else ''
        print(f'  {mu_val:5.3f}  {mean_phi[i]:7.4f}  {std_phi[i]:8.4f}  '
              f'{mean_N[i]:7.2f}  {std_N[i]:7.2f}  {n_sweeps[i]:7d}{marker}')

    print()
    print(f'Target phi   = {args.target_phi}  =>  N_target = {N_target:.2f}')
    print(f'mu* (grid)   = {mu_star:.4f}   <phi> = {mean_phi[idx_star]:.4f}')
    print(f'mu* (interp) = {mu_star_interp:.4f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.no_plot:
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.plot_output)), exist_ok=True)

    fig, (ax_phi, ax_N) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # ── Top panel: phi vs mu ───────────────────────────────────────────────────
    ax_phi.errorbar(
        mu_sorted, mean_phi, yerr=sem_phi,
        fmt='o-', capsize=5, capthick=1.2, elinewidth=1.2,
        color='steelblue', label=r'$\langle \phi \rangle \pm$ SEM',
    )
    # shaded ±1 std band
    ax_phi.fill_between(
        mu_sorted,
        mean_phi - std_phi,
        mean_phi + std_phi,
        alpha=0.15, color='steelblue', label=r'$\pm 1\,\sigma$',
    )
    ax_phi.axhline(
        args.target_phi, color='red', ls='--', lw=1.5,
        label=f'target $\\phi$ = {args.target_phi}',
    )
    ax_phi.axvline(
        mu_star, color='green', ls=':', lw=2.0,
        label=f'$\\mu^*$ (grid) = {mu_star:.3f}',
    )
    if abs(mu_star_interp - mu_star) > 1e-4:
        ax_phi.axvline(
            mu_star_interp, color='limegreen', ls='--', lw=1.5,
            label=f'$\\mu^*$ (interp) = {mu_star_interp:.4f}',
        )
    ax_phi.set_ylabel(r'Area fraction $\phi$', fontsize=12)
    ax_phi.legend(fontsize=9, loc='upper left')
    ax_phi.grid(alpha=0.3)
    ax_phi.set_title(
        f'GCMC $\\mu$-scan  |  $A={box_area:.0f}$  $r={r_center}$',
        fontsize=13,
    )

    # ── Bottom panel: <N> vs mu ────────────────────────────────────────────────
    ax_N.errorbar(
        mu_sorted, mean_N, yerr=sem_N,
        fmt='s-', capsize=5, capthick=1.2, elinewidth=1.2,
        color='darkorange', label=r'$\langle N \rangle \pm$ SEM',
    )
    ax_N.fill_between(
        mu_sorted,
        mean_N - std_N,
        mean_N + std_N,
        alpha=0.15, color='darkorange', label=r'$\pm 1\,\sigma$',
    )
    ax_N.axhline(
        N_target, color='red', ls='--', lw=1.5,
        label=f'$N_{{target}}$ = {N_target:.1f}',
    )
    ax_N.axvline(mu_star,        color='green',     ls=':',  lw=2.0)
    if abs(mu_star_interp - mu_star) > 1e-4:
        ax_N.axvline(mu_star_interp, color='limegreen', ls='--', lw=1.5)
    ax_N.set_xlabel(r'Chemical potential $\mu\,/\,k_BT$', fontsize=12)
    ax_N.set_ylabel(r'$\langle N \rangle$', fontsize=12)
    ax_N.legend(fontsize=9, loc='upper left')
    ax_N.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.plot_output, dpi=150)
    print(f'\nPlot saved: {args.plot_output}')
    plt.close()


if __name__ == '__main__':
    main()
