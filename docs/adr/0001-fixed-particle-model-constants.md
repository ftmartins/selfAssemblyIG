---
status: accepted
---

# Fix the particle model constants across simulation and theory

The equilibrium theory and the simulations must use one and the same potential. Every run and calculation therefore fixes the following: Morse steepness α = 7 (patch radius a ≈ 0.38); near-hard cores (soft-sphere overlap, ε_c = 10⁴, σ = 2R); patch sites on the core surface; and the full Morse attraction with an explicit smooth cutoff (`r_onset = 1.5`, `r_cutoff = 2.0`). Hard cores are a deliberate departure from the soft cores of Snyder & Du (ε_s = 1). With hard cores, joint bends cost energy at quartic order, which is central to the floppy-mode question. Their stiffness is set mainly by the A–B bond for E_AB ≲ 25 and mainly by the core above that.

## Consequences

- The original code passed only `r_cutoff = 0.38` to `morse_pair`. JAX-MD 0.2.27 ignores this, because `r_onset` defaults to 2.0, so the bond was effectively the full Morse out to 2.0. Newer JAX-MD would apply it as a hard cutoff at 0.38, where the attraction is still about −0.14 E_AB. Both values are now set explicitly, so behaviour no longer depends on the JAX-MD version.
- Varying the patch radius is out of scope. Results hold for α = 7 only.
