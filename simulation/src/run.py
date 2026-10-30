"""
Unified simulation entry point: one call selects the ensemble and dispatches to
the matching engine, which streams the shared HDF5 trajectory schema
(:mod:`base.trajectory`).

``ensemble='canonical'`` runs the NVT Nose-Hoover driver
(:func:`simulation.src.canonical_md.run_many_particle_md`); ``n_equil`` /
``n_prod`` / ``sample_interval`` are MD steps.  ``ensemble='grand_canonical'``
runs the hybrid MC/MD GCMC engine (:func:`simulation.src.gcmc.run_gcmc`);
they are sweeps.
"""

import numpy as np

ENSEMBLES = ('canonical', 'grand_canonical')


def _canonical(path, *, opening_angle, E_AB, box_area, N, n_equil, n_prod,
               sample_interval, seed, kT, E_AA, E_BB, rep_A, dt,
               checkpoint_interval, equilibration_sample_interval, precision, resume,
               initial_condition):
    import jax
    from base.initial_conditions import preassembled_rings, rsa_2d
    from base.params import SimpleParams
    from simulation.src.canonical_md import run_many_particle_md

    L = float(np.sqrt(box_area))
    params = SimpleParams(kBT=kT, E_RR=E_AA, E_BB=E_BB, E_RB=E_AB, rep_A=rep_A)
    params.alpha = float(opening_angle)
    if initial_condition == 'preassembled':
        positions, orientations = preassembled_rings(N, L, opening_angle, seed=seed)
    elif initial_condition == 'dispersed':
        positions, orientations = rsa_2d(N, L, seed=seed, centered=False)
    else:
        raise ValueError("initial_condition must be 'dispersed' or 'preassembled'")
    return run_many_particle_md(
        N, L, params, n_prod, kT, jax.random.PRNGKey(seed), dt,
        positions, orientations, sample_interval=sample_interval,
        equilibration_steps=n_equil,
        equilibration_sample_interval=equilibration_sample_interval,
        trajectory_path=path, resume=resume, precision=precision,
        checkpoint_interval=checkpoint_interval, seed=seed)


def _grand_canonical(path, *, opening_angle, E_AB, box_area, N_init, mu, n_equil,
                     n_prod, sample_interval, seed, kT, E_AA, E_BB, rep_A,
                     md_steps, md_dt, gamma, f_disp, checkpoint_interval,
                     equilibration_sample_interval, precision, resume, verbose, initial_condition):
    from simulation.src.gcmc import generate_ic, make_gcmc_params, run_gcmc

    L = float(np.sqrt(box_area))
    params = make_gcmc_params(opening_angle=opening_angle, E_AB=E_AB, E_AA=E_AA,
                              E_BB=E_BB, rep_A=rep_A)
    attrs = dict(
        ensemble='grand_canonical', engine='hybrid_mc_md', thermostat='baoab_langevin',
        seed=int(seed), box_size=L, kT=float(kT), mu=float(mu),
        opening_angle=float(opening_angle), E_AA=float(E_AA), E_BB=float(E_BB),
        E_AB=float(E_AB), bond_specificity=float(E_AB) / float(E_AA + E_BB),
        rep_A=float(rep_A), N_init=int(N_init), f_disp=float(f_disp),
        md_steps=int(md_steps), md_dt=float(md_dt), gamma=float(gamma),
        num_equilibration_steps=int(n_equil), num_production_steps=int(n_prod),
        sample_interval=int(sample_interval),
        equilibration_sample_interval=int(equilibration_sample_interval),
        initial_condition=initial_condition)
    rng = np.random.default_rng(seed)
    state = generate_ic(int(N_init), L, params, kT=kT, seed=seed,
                        initial_condition=initial_condition)
    return run_gcmc(
        state, L, kT, mu, params, rng, n_equil=n_equil, n_prod=n_prod, f_disp=f_disp,
        MD_STEPS=md_steps, MD_DT=md_dt, GAMMA=gamma, snapshot_interval=sample_interval,
        verbose=verbose, trajectory_path=path, attrs=attrs, resume=resume,
        precision=precision, equilibration_sample_interval=equilibration_sample_interval,
        checkpoint_interval=checkpoint_interval)


def run_simulation(ensemble, path, *, opening_angle, E_AB, box_area, n_equil, n_prod,
                   sample_interval, seed=42, kT=1.0, E_AA=1.0, E_BB=1.0, rep_A=None,
                   N=None, dt=1e-3, checkpoint_interval=None,
                   mu=None, N_init=16, md_steps=5000, md_dt=1e-2, gamma=1.0, f_disp=0.5,
                   equilibration_sample_interval=0, precision='float64', resume=False,
                   verbose=False, initial_condition='dispersed'):
    """Run one simulation of the given ensemble into the HDF5 file ``path``.

    Canonical needs ``N`` (and ``dt``); grand canonical needs ``mu`` (and
    ``N_init``, ``md_steps``, ``md_dt``, ``gamma``, ``f_disp``).
    ``initial_condition`` is ``'dispersed'`` (RSA monomers) or ``'preassembled'``
    (complete rings of the size matching ``opening_angle``).  ``rep_A``
    defaults to each engine's own value.  Returns the engine's result (final MD
    state for canonical, result dict for grand canonical).
    """
    if ensemble not in ENSEMBLES:
        raise ValueError(f"ensemble must be one of {ENSEMBLES}, got {ensemble!r}")
    common = dict(opening_angle=opening_angle, E_AB=E_AB, box_area=box_area,
                  n_equil=n_equil, n_prod=n_prod, sample_interval=sample_interval,
                  seed=seed, kT=kT, E_AA=E_AA, E_BB=E_BB,
                  equilibration_sample_interval=equilibration_sample_interval,
                  precision=precision, resume=resume, initial_condition=initial_condition)
    if ensemble == 'canonical':
        if N is None:
            raise ValueError("canonical ensemble requires N")
        return _canonical(path, **common, N=int(N), dt=dt,
                          rep_A=10.0 if rep_A is None else rep_A,
                          checkpoint_interval=checkpoint_interval or 200_000)
    if mu is None:
        raise ValueError("grand_canonical ensemble requires mu")
    return _grand_canonical(path, **common, mu=mu, N_init=N_init, md_steps=md_steps,
                            md_dt=md_dt, gamma=gamma, f_disp=f_disp,
                            rep_A=10_000.0 if rep_A is None else rep_A,
                            checkpoint_interval=checkpoint_interval, verbose=verbose)
