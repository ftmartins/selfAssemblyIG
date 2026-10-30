"""
run.py — one command for both ensembles.

Selects the ensemble with ``--ensemble`` and dispatches to the matching engine
(:func:`simulation.src.run.run_simulation`); both write the same HDF5 schema.

Usage
-----
python simulation/runners/run.py --ensemble canonical --opening_angle 1.5708 \\
    --E_AB 5.0 --box_area 100 --N 40 --output data/run_nvt.h5
python simulation/runners/run.py --ensemble grand_canonical --opening_angle 1.5708 \\
    --E_AB 5.0 --mu -4.0 --output data/run_gc.h5

``--n_equil`` / ``--n_prod`` / ``--sample_interval`` are MD steps (canonical) or
sweeps (grand canonical).  ``--resume`` continues an existing file.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))

from base.jax_setup import configure_jax
from simulation.src.gcmc import DEFAULTS as GC


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Run a simulation in either ensemble')
    p.add_argument('--ensemble', choices=['canonical', 'grand_canonical'], required=True)
    p.add_argument('--output', required=True, help='HDF5 trajectory file')
    p.add_argument('--opening_angle', type=float, required=True, help='Patch opening angle (rad)')
    p.add_argument('--E_AB', type=float, required=True)
    p.add_argument('--E_AA', type=float, default=1.0)
    p.add_argument('--E_BB', type=float, default=1.0)
    p.add_argument('--rep_A', type=float, default=None, help='default: per-ensemble')
    p.add_argument('--kT', type=float, default=1.0)
    p.add_argument('--box_area', type=float, default=GC.box_area)
    p.add_argument('--seed', type=int, default=GC.seed)
    p.add_argument('--n_equil', type=int, default=None)
    p.add_argument('--n_prod', type=int, default=None)
    p.add_argument('--sample_interval', type=int, default=None)
    p.add_argument('--equilibration_sample_interval', type=int, default=0)
    p.add_argument('--checkpoint_interval', type=int, default=None)
    p.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    p.add_argument('--initial_condition', choices=['dispersed', 'preassembled'],
                   default='dispersed', help='RSA monomers or pre-assembled rings')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--quiet', action='store_true')
    c = p.add_argument_group('canonical')
    c.add_argument('--N', type=int, default=None)
    c.add_argument('--dt', type=float, default=1e-3)
    g = p.add_argument_group('grand canonical')
    g.add_argument('--mu', type=float, default=None, help='chemical potential (kT units)')
    g.add_argument('--N_init', type=int, default=GC.n_init)
    g.add_argument('--md_steps', type=int, default=GC.md_steps)
    g.add_argument('--md_dt', type=float, default=GC.md_dt)
    g.add_argument('--gamma', type=float, default=GC.gamma)
    g.add_argument('--f_disp', type=float, default=GC.f_disp)
    return p.parse_args(argv)


def main(argv=None):
    configure_jax()
    from simulation.src.run import run_simulation
    a = parse_args(argv)
    canonical = a.ensemble == 'canonical'
    defaults = ((10_000, 100_000, 100) if canonical
                else (GC.n_equil_full, GC.n_prod_full, GC.save_interval))
    n_equil, n_prod, sample = (v if v is not None else d for v, d in
                               zip((a.n_equil, a.n_prod, a.sample_interval), defaults))
    run_simulation(
        a.ensemble, a.output, opening_angle=a.opening_angle, E_AB=a.E_AB,
        box_area=a.box_area, n_equil=n_equil, n_prod=n_prod, sample_interval=sample,
        seed=a.seed, kT=a.kT, E_AA=a.E_AA, E_BB=a.E_BB, rep_A=a.rep_A, N=a.N, dt=a.dt,
        checkpoint_interval=a.checkpoint_interval, mu=a.mu, N_init=a.N_init,
        md_steps=a.md_steps, md_dt=a.md_dt, gamma=a.gamma, f_disp=a.f_disp,
        equilibration_sample_interval=a.equilibration_sample_interval,
        precision=a.precision, resume=a.resume, verbose=not a.quiet,
        initial_condition=a.initial_condition)
    print(f'Saved: {a.output}')


if __name__ == '__main__':
    main()
