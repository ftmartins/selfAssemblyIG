"""
aggregate_mu_scan.py — CLI wrapper around :mod:`analysis.mu_scan` +
:mod:`analysis.plotting`.

For a fixed (E_AB, opening_angle), pool the equilibrated GCMC mu-scan NPZs,
print the <phi>/<N> table, report mu* for a target area fraction, and save the
two-panel plot.

    python analysis/aggregate_mu_scan.py --input_dir data/mu_scan \\
        --E_AB 5 --opening_angle 1.5708
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from base.paths import MU_SCAN_DIR, PLOTS_DIR
from analysis.mu_scan import aggregate_by_mu, find_mu_star
from analysis.plotting import plot_mu_scan

# GCMC-run reference defaults (kept in sync with simulation.src.gcmc.GcmcDefaults)
DEFAULT_BOX_AREA = 100.0
DEFAULT_TARGET_PHI = 0.2
DEFAULT_R_CENTER = 1.0
DEFAULT_ALPHA_EQUIL = 0.05


def parse_args():
    p = argparse.ArgumentParser(description='Aggregate GCMC mu-scan; find mu* for a target phi')
    p.add_argument('--input_dir', type=str, default=str(MU_SCAN_DIR))
    p.add_argument('--E_AB', type=float, default=None)
    p.add_argument('--opening_angle', type=float, default=None)
    p.add_argument('--tol', type=float, default=1e-3)
    p.add_argument('--box_area', type=float, default=DEFAULT_BOX_AREA)
    p.add_argument('--target_phi', type=float, default=DEFAULT_TARGET_PHI)
    p.add_argument('--r_center', type=float, default=DEFAULT_R_CENTER)
    p.add_argument('--alpha_equil', type=float, default=DEFAULT_ALPHA_EQUIL)
    p.add_argument('--plot_output', type=str, default=None)
    p.add_argument('--no_plot', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    N_target = args.target_phi * args.box_area / (np.pi * args.r_center ** 2)

    agg = aggregate_by_mu(
        args.input_dir, E_AB=args.E_AB, opening_angle=args.opening_angle,
        tol=args.tol, alpha_equil=args.alpha_equil,
        box_area=args.box_area, r_center=args.r_center,
    )
    mu_star, mu_interp, idx = find_mu_star(agg['mu'], agg['mean_phi'], args.target_phi)

    if agg['n_filtered']:
        print(f"{agg['n_filtered']} file(s) skipped (parameter mismatch).")
    if agg['skipped']:
        print(f"{len(agg['skipped'])} file(s) excluded (not equilibrated).")

    hdr = f'{"mu":>7}  {"<phi>":>7}  {"std_phi":>8}  {"<N>":>7}  {"std_N":>7}  {"sweeps":>7}'
    print('\n' + hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for i, mu_val in enumerate(agg['mu']):
        marker = '  <-- mu*' if i == idx else ''
        print(f'  {mu_val:5.3f}  {agg["mean_phi"][i]:7.4f}  {agg["std_phi"][i]:8.4f}  '
              f'{agg["mean_N"][i]:7.2f}  {agg["std_N"][i]:7.2f}  {agg["n_sweeps"][i]:7d}{marker}')

    print(f'\nTarget phi   = {args.target_phi}  =>  N_target = {N_target:.2f}')
    print(f'mu* (grid)   = {mu_star:.4f}   <phi> = {agg["mean_phi"][idx]:.4f}')
    print(f'mu* (interp) = {mu_interp:.4f}')

    if args.no_plot:
        return
    out = args.plot_output
    if out is None:
        tag = ''
        if args.E_AB is not None:
            tag += f'_EAB{args.E_AB:.3f}'
        if args.opening_angle is not None:
            tag += f'_oa{np.degrees(args.opening_angle):.2f}deg'
        out = os.path.join(str(PLOTS_DIR), 'mu_scan', f'mu_scan_aggregated{tag}.png')
    plot_mu_scan(agg, args.target_phi, N_target, mu_star, mu_interp,
                 args.box_area, args.r_center, out_path=out)


if __name__ == '__main__':
    main()
