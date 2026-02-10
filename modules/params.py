"""
Parameter classes and grid generation for patchy particle simulations.

Self-contained module with no dependencies on production/ code.
"""

import numpy as np


class SimpleParams:
    """
    Parameter class for yield calculations.

    Attributes
    ----------
    kBT : float
        Thermal energy
    E_RR : float
        R-R (A-A) patch attraction strength
    E_BB : float
        B-B patch attraction strength
    E_RB : float
        R-B (A-B) cross-patch interaction strength
    morse_a : float
        Morse potential decay parameter
    morse_r0 : float
        Morse equilibrium distance
    rep_A : float
        Core-core repulsion strength
    rep_alpha : float
        Core repulsion exponent
    morse_rcut : float
        Morse cutoff distance
    r_patch : float
        Distance of patches from particle center
    alpha : float
        Opening angle (radians), set after construction
    selectivity : float
        Selectivity ratio zeta, set after construction
    """
    def __init__(self,
                 kBT=1.0,
                 E_RR=5.0,
                 E_BB=3.0,
                 E_RB=4.0,
                 morse_a=7.0,
                 morse_r0=1.0,
                 rep_A=10.0,
                 rep_alpha=2.0,
                 morse_rcut=0.5,
                 r_patch=1.0):
        self.kBT = kBT
        self.E_RR = E_RR
        self.E_BB = E_BB
        self.E_RB = E_RB
        self.morse_a = morse_a
        self.morse_r0 = morse_r0
        self.rep_A = rep_A
        self.rep_alpha = rep_alpha
        self.morse_rcut = morse_rcut
        self.r_patch = r_patch
        self.n_point_species = 3


def generate_parameter_grid(alphas, zetas, kT=1.0, morse_a=7.0):
    """
    Generate systematic parameter grid from selectivity values.

    Parameters
    ----------
    alphas : array-like
        Opening angles in radians.
    zetas : array-like
        Selectivity ratios zeta = E_AB / (E_AA + E_BB).
    kT : float
        Temperature.
    morse_a : float
        Morse potential decay parameter.

    Returns
    -------
    list of SimpleParams
        One entry per (alpha, zeta) pair, ordered as alpha-major
        (i.e. all zetas for alpha[0], then all zetas for alpha[1], ...).

    Notes
    -----
    Fixes E_AA = E_BB = 1.0 and solves E_AB = zeta * (E_AA + E_BB).
    """
    param_grid = []

    for alpha in alphas:
        for zeta in zetas:
            E_AA = 1.0
            E_BB = 1.0
            E_AB = zeta * (E_AA + E_BB)

            params = SimpleParams(
                kBT=kT,
                E_RR=E_AA,
                E_BB=E_BB,
                E_RB=E_AB,
                morse_a=morse_a,
                morse_r0=0.0,
                rep_A=10_000.0,
                rep_alpha=2.5,
                morse_rcut=0.38,
                r_patch=1.0,
            )
            params.alpha = alpha
            params.selectivity = zeta
            param_grid.append(params)

    return param_grid
