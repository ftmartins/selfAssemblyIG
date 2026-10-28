"""
Parameter containers shared by the simulation engines.

* :class:`SimpleParams` / :func:`generate_parameter_grid` — parameter objects
  for the canonical-ensemble yield sweeps (attribute access: ``.alpha``,
  ``.E_RR``, ``.E_RB``, ``.E_BB``, ``.rep_A``, ``.r_patch`` ...).
* :func:`make_gcmc_params` — the plain ``dict`` used by the hybrid GCMC engine
  (all values cast to ``float`` so ``frozenset(params.items())`` is hashable for
  the MD-runner LRU cache).
"""

from base.config import ALPHA, MORSE_R_ONSET, MORSE_R_CUTOFF


class SimpleParams:
    """Parameter bag for canonical-ensemble yield calculations."""

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
    One :class:`SimpleParams` per ``(alpha, zeta)`` pair, alpha-major order.

    Fixes ``E_AA = E_BB = 1.0`` and sets ``E_AB = zeta * (E_AA + E_BB)``.
    ``alphas`` are opening angles in radians; ``zetas`` are selectivity ratios.
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


def make_gcmc_params(
    opening_angle: float,
    E_AB: float,
    E_AA: float = 1.0,
    E_BB: float = 1.0,
    r_patch: float = 1.0,
    rep_A: float = 10_000.0,
) -> dict:
    """Assemble the ``params`` dict used by every hybrid-GCMC energy/MC function."""
    return dict(
        opening_angle=float(opening_angle),
        r_patch=float(r_patch),
        E_AA=float(E_AA),
        E_BB=float(E_BB),
        E_AB=float(E_AB),
        rep_A=float(rep_A),
        alpha=float(ALPHA),
        r_onset=float(MORSE_R_ONSET),
        r_cutoff=float(MORSE_R_CUTOFF),
    )
