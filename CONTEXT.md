# SelfAssemblyIG

Information geometry of equilibrium self-assembly: how the equilibrium distribution of assembled structures depends on the design of two-patch particles in 2D, with rings as the target structures.

## Language

### Building blocks

**Two-patch particle**:
A rigid 2D disk with a repulsive core and two point patches, A and B, on its surface; patches on different particles attract through a Morse potential.
_Avoid_: patchy disk, monomer (when meaning the particle design rather than the free species)

**Patch site**:
The point on the core surface where a patch is centered; it lies at the core radius from the particle center.
_Avoid_: patch position (ambiguous with patch extent), patch radius

**Patch radius** (a):
The size of a patch: half the patch–patch distance at which the Morse attraction falls to 1% of its depth, set by the Morse steepness and much smaller than the core radius.
_Avoid_: patch size, cutoff radius

**Opening angle** (θ):
The angle between the A and B patch directions on a two-patch particle.
_Avoid_: α (reserved for Morse steepness), patch angle

**Morse steepness** (α):
The decay constant of the patch–patch Morse attraction; it sets the patch radius.
_Avoid_: β, ALPHA, patch width

**Bond specificity** (ζ):
The ratio ζ = E_AB / (E_AA + E_BB) of the A–B Morse depth to the like-patch depths; it measures how strongly correct (A–B) bonds are favored over A–A and B–B bonds.
_Avoid_: selectivity (reserved for size selectivity)

### Structures and outcomes

**Bond**:
An A–B patch pair on two particles that are closer than the patch radius.
_Avoid_: contact, link

**Cluster**:
A connected component of the bond graph, with no condition on its shape.
_Avoid_: aggregate, assembly

**Ring**:
A closed cluster of two-patch particles in which every particle is bonded through both of its patches; an n-ring contains n particles.
_Avoid_: n-gon (for the cluster itself; n-gon names the reference geometry), loop

**Open chain**:
A linear cluster of two-patch particles joined by A–B bonds, with one free A patch and one free B patch at its ends.
_Avoid_: polymer, filament, linear aggregate

**Equilibrium yield**:
The equilibrium abundance of a structure, computed from per-structure partition functions and mass action; the object of study in this project, reported both as a particle yield and as a cluster yield.
_Avoid_: assembly yield, yield (unqualified, when the snapshot sense could be meant)

**Particle yield**:
The fraction of all particles that belong to a given structure.
_Avoid_: mass fraction

**Cluster yield** (p_n):
The fraction of all clusters, monomers included, that are a given structure; the quantity that counting experiments sample.
_Avoid_: number fraction

**Snapshot yield**:
A yield measured from finite-time simulation snapshots (e.g. the fraction of runs containing a target at the final step); used only for validation and comparison with the literature, never as the object of study.
_Avoid_: yield (unqualified)

**Size selectivity**:
The preference of a design for one ring size over other ring sizes in the equilibrium yields.
_Avoid_: selectivity (unqualified), angle selectivity

**Matching angle** (θ_n):
The opening angle at which an n-ring closes with every bond unstrained, θ_n = π(n−2)/n, the interior angle of a regular n-gon.
_Avoid_: ideal angle, target angle

**Matching ring size** (m):
The ring size whose matching angle equals a given opening angle, m = 2π/(π − θ); generally not an integer. An open chain with straight joints traces an arc of this ring.
_Avoid_: preferred ring size, target size

**Ring conformation**:
The arrangement of a ring's particles, both center positions and orientations, at fixed ring size, e.g. the counter-rotation of neighbours along a 4-ring's gear mode or a 5-ring's departure from a regular pentagon.
_Avoid_: shape, arrangement

**Ring-size distribution**:
The equilibrium distribution over ring sizes, with all conformations of a given size counted together.
_Avoid_: structure distribution

### Mechanics

**Floppy mode**:
An internal motion of a structure, beyond rigid-body motions, with zero harmonic stiffness.
_Avoid_: zero mode (for internal motions), soft mode

**Soft mode**:
An internal motion of a structure with small but nonzero harmonic stiffness.
_Avoid_: floppy mode

**Joint bend** (δ):
The relative rotation of two bonded particles away from a straight bond, in which patch B of one points directly at patch A of the other.
_Avoid_: hinge angle, bond angle

**Joint stiffness** (C):
The coefficient of the quartic energy cost of a joint bend.
_Avoid_: bending modulus, hinge stiffness

**Gear mode**:
The floppy mode of an even ring in which neighbouring particles counter-rotate while the particle centers stay fixed.
_Avoid_: rolling mode, lozenge mode

### Design space

**Design parameters**:
The quantities varied across designs and conditions: opening angle, bond energies (chiefly E_AB), and total particle concentration or chemical potential.
_Avoid_: knobs, hyperparameters

**Model constants**:
The quantities held fixed in every simulation and calculation: Morse steepness, a near-hard core, and placement of the patches on the core surface.
_Avoid_: fixed parameters
