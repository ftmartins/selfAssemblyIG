---
status: accepted
---

# One bond-graph cluster definition, shared by theory and simulation analysis

A bond is an A–B patch pair closer than the patch radius a ≈ 0.38. Clusters are the connected components of the bond graph. An n-ring is a cluster of n particles in which every particle has both patches bonded. The equilibrium partition functions and the MD/GCMC analysis use this same definition and apply no shape-match filter. A shape filter would discard exactly the floppy and strained conformations that the partition functions count, so simulated and predicted yields would no longer measure the same thing.

## Consequences

- The earlier detection pipeline is replaced for yield measurements. It clustered same-type patches within a chord-distance allowance, then filtered by shape loss < 0.5.
- The ring-size distribution must be checked for insensitivity to the bond distance around a, in both theory and simulation. This matters most at low E_AB, where the bond is shallow.
