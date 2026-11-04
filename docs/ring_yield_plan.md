# Equilibrium ring yields: computation plan

Plan only. Nothing here has been implemented in the repo.

## 1. What is being computed

Model constants: Morse steepness α = 7, patch sites on the core surface, core exponent 2.5, one Hamiltonian for theory, MD and GCMC (ADR-0001). Design parameters: opening angle θ, bond energy E_AB, and total density ρ (or chemical potential μ).

For species s with n_s particles, area A, and fugacity z = e^{βμ}:

```
Z_s = (1/σ_s) ∫ Π dq_i e^{−βU}      partition function over the ADR-0002 bond graph
q_s = Z_s / A
ρ_s = q_s z^{n_s}                   mass action
ρ   = Σ_s n_s ρ_s                   fixes z at given density
```

Species: monomer, open A–B chains, A–B rings n = 3…n_max. Relative yields need z, so they depend on density:

```
ρ_n / ρ_m = (q_n / q_m) z^{n−m}
```

For the mode picture, keep the Goodrich–Brenner change of coordinates, but take the soft modes out of the Gaussian:

```
Z_n = e^{−βE_0} · A · (J̃/σ_n) · Π_stiff √(2π/βλ_i) · I_soft · e^{n b_n}
```

I_soft is the true integral over the n−3 soft Hessian modes. b_n is the leftover error per bond, from Morse anharmonicity and the one-sided core. Sampling supplies the accurate ln q_n. This form explains where the flexibility enters.

## 2. Fixes

| Area | Problem | Fix |
|---|---|---|
| Yield rule | `Z/ΣZ` drops z^{n_s}, so results depend on units and not on density | Solve mass action with particle conservation |
| Monomer | Weighted 1 or A | Use Z_1 = 2πA |
| Species | Rings and monomers only | Add chains |
| J̃ | Evaluated without q0 | J̃ = 2π √det G, with G the Gram matrix of the rigid generators at q0; no Monte Carlo needed |
| Rigid modes | Removed by eigenvalue order | Project out the analytic generators |
| Filters | Saddles and dissociated structures pass | Reject negative internal modes; check the bond graph; perturb symmetric starts so saddles can relax |
| Regulariser | `1e-12` inflates Z_vib | Remove it |
| Soft modes | Gaussian width is far too large near θ_n | Integrate them outside the Gaussian |
| Temperature | Z_vib omits kT | Use √(2π/βλ) |
| Hamiltonian | Core exponent and prefactor differ between code paths | Use ε_c/2.5·(1 − d/σ)^2.5 everywhere |
| Core code | `rep_alpha` isn't passed on; `total_energy` defaults to `rep_A=10`; tests use `rep_A=0` | Pass the exponent explicitly in `build_pair_energy_fn` and `soft_sphere_repulsion`; set the default; add a test with nonzero core |
| Metric | Built on `P = w/Σw` | Rebuild on mass action, including the change in z |
| Legacy | jax-md path bugs in `compute_Zc_2d_jaxmd` | Do not port this code |

## 3. Central scripts

| Script | Role |
|---|---|
| `base/energy.py` (edit) | One core form for all engines |
| `analysis/ring_modes.py` | Relax rings, project out rigid modes, split soft and stiff modes, compute E_0, J̃ and λ_i, apply filters; read the samples from `partition.py` to get I_soft and the residual b_n |
| `analysis/partition.py` | Dimer quadrature for K; chain and ring ln q_n by closure sampling; design scores from the same samples |
| `analysis/direct_z.py` | Independent Cartesian importance sampling of ln q_n, used as a check |
| `analysis/mass_action.py` | From a table of (n_s, ln q_s), compute yields at fixed z, at fixed ρ, and at finite N |
| `analysis/fisher.py` | Build g_ab from scores and yields |
| `simulation/runners/scan_partition.py` | SLURM array over (θ, E_AB, n); writes one table of ln q_n, scores, errors and mode data |
| `analysis/yield_maps.py` | Interpolate the table and produce yield, selectivity and metric maps over (θ, E_AB, ρ) |
| `analysis/validate_yields.py` | Compare predictions with MD and GCMC |

Notes on the less obvious pieces:

- **Closure sampling:** grow an open chain from the exact bonded-pair distribution, weight chains whose end rebinds to the start, and tilt the bend distribution so strained rings are reachable.
- **Long chains:** use Rosenbluth growth. Chains longer than the matching ring curl into themselves, and naive sampling fails there.
- **Finite N:** use the exact recursion

```
N Q_N = Σ_k k a_k Q_{N−k},   ⟨N_s⟩ = A q_s Q_{N−n_s} / Q_N
```

  where a_k is A times the sum of q_s over species of size k. This is needed for small systems (N of order 100 or below).

- **GCMC convention:** with the current insertion move, z = e^{βμ}/2π.
- **Mode breakdown:** `ring_modes.py` also does the last two steps of the Z_n breakdown, reading the samples from `partition.py`. It projects them on the soft eigenvectors to get I_soft, then takes b_n as the residual:

```
n b_n = ln q_n − [−βE_0 + ln(J̃/σ_n) + Σ_stiff ½ ln(2π/βλ_i) + ln I_soft]
```

## 4. Exploring parameter space

1. **Compute q_n once per (θ, E_AB).** The q_n don't depend on density, so density sweeps then only need the mass-action solver.
2. **Start coarse.** Use θ from 60° to 140° and E_AB from about 5 to 40, with rings n = 3–8. Skip rings that are too strained to matter, and confirm that skipping them doesn't change the yields.
3. **Refine adaptively.** Add points near ring-size crossovers and near each matching angle θ_n = π(n−2)/n, where ln q_n and the metric change fastest.
4. **Check n_max.** At large θ, add n = 9 and above until yields stop changing.
5. **Set an accuracy target.** Relative error in ρ_n/ρ_m is about the error in ln q_n plus |n−m| times the error in ln z. Aim for 0.05 or better in ln q_n, with an effective-sample-size cutoff and bootstrap errors carried through to the yields.
6. **Get scores from the ensemble, not from finite differences of ln q_n:**

```
u_a,n = ∂_a ln q_n = ⟨−β ∂_a U⟩_n
```

7. **Report both yields:** cluster yield p_n and particle yield.

Flexibility and the metric:

```
∂_a ln z = −Σ_s n_s ρ_s u_a,s / Σ_s n_s² ρ_s     at fixed density
v_a,s    = u_a,s + n_s ∂_a ln z
g_ab     = A Σ_s ρ_s v_a,s v_b,s                  for counts at fixed density
```

For cluster-yield statistics, use g_ab = Σ_n p_n ∂_a ln p_n ∂_b ln p_n instead. To show the flexibility contribution, compare the sampled score with the T=0 energy score at q0, and report I_soft from the thermal distribution along the soft modes.

## 5. Validation order

1. Unit tests: J̃ for regular rings; rejection of injected saddles and dissociated states; dimer mass action against the quadratic solution; recursion against a brute-force partition sum and against mass action at large N.
2. `partition.py` against `direct_z.py` for n = 3–5 at the new exponent.
3. Model-free test on simulation data: ln ρ_s against βμ should have slope n_s, and ln ρ_s − n_s ln ρ_1 should not depend on density. This finds where the ideal cluster gas holds.
4. Predicted yields against MD and GCMC: triangle first, then square, then a crossover angle.
5. ADR-0002 sensitivity to the bond distance, mainly at low E_AB.

## 6. Open items

- **ε_c:** at 10⁴, exponent 2.5 is softer than before, with a 1 kT overlap of about 0.036 rather than 0.01. This needs a decision, because the project wants near-hard cores.
- **Dilute limit:** at φ = 0.1, the second-virial estimate of the monomer activity coefficient is already about 1.5, so step 3 of validation may limit the usable density.
- **I_soft for n ≥ 5:** this is a joint integral over several modes, so a histogram gets noisy. It may need a better density estimator.
- **Exponent 2.5 has not been tested.** Earlier checks used exponent 2. Closure matched Cartesian Z within 0.003 at seven points there, but b_n, the soft-mode numbers and all crossover angles must be redone.
- **Earlier `g_ab` numbers are provisional.** They used exponent 2 and the joint-bend approximation, so they should not be reused.
- **4th-order Laplace:** worth one test as the formalism's own correction. It is untested here.
