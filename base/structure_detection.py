"""
Cluster / polygon detection on a produced particle configuration.

Two independent detectors:

* :func:`detect_clusters` / :func:`compute_yields_from_frame` — bond-graph
  connectivity from patch–patch proximity.  Pure numpy, used by the GCMC engine
  and the canonical yield-sweep cluster runner.
* :func:`count_polygons` — freud spatial clustering plus a shape-match check
  against the ideal n-gon.  Used by the single-parameter yield runner.
  ``freud`` is imported lazily so ``base`` stays importable without it.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from base.config import CENTER_RADIUS, PATCH_SIZE
from base.geometry import (
    patch_positions, regular_ngon_reference, NGON_SIZES, patch_separation,
)

STRUCTURES = ['triangle', 'square', 'pentagon', 'hexagon', 'heptagon', 'octagon']
BOND_THRESHOLD = 0.5
PATCH_ALLOWANCE = PATCH_SIZE * 2 / 3.0
CLOSENESS_PENALTY = 0.01
CLOSENESS_PENALTY_NEIGHBORS = 1

_SIZE_TO_NAME = {
    1: 'monomer', 2: 'dimer', 3: 'triangle', 4: 'square',
    5: 'pentagon', 6: 'hexagon', 7: 'heptagon', 8: 'octagon',
}


# ── Bond-graph connectivity ────────────────────────────────────────────────

def classify_cluster(cluster_size: int) -> str:
    return _SIZE_TO_NAME.get(cluster_size, f'cluster_{cluster_size}')


def detect_clusters(positions, orientations, r_patch, opening_angle,
                    bond_threshold=BOND_THRESHOLD, box_size=None):
    """
    Connected components of the bond graph: particles ``i`` and ``j`` are bonded
    if any of their A/B patch pairs are within ``bond_threshold``.

    ``box_size`` is accepted for signature compatibility; the original
    implementation used raw (non-periodic) patch distances and that behaviour is
    preserved.

    Returns ``(clusters, bonds)`` where ``clusters`` is a list of index lists.
    """
    n = len(positions)
    positions = np.asarray(positions)
    orientations = np.asarray(orientations)

    patches_A, patches_B = patch_positions(positions, orientations, opening_angle, r_patch)

    bonds = []
    adjacency = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d_AA = np.linalg.norm(patches_A[i] - patches_A[j])
            d_BB = np.linalg.norm(patches_B[i] - patches_B[j])
            d_AB = np.linalg.norm(patches_A[i] - patches_B[j])
            d_BA = np.linalg.norm(patches_B[i] - patches_A[j])
            for d, tag in ((d_AA, ('A', 'A')), (d_BB, ('B', 'B')),
                           (d_AB, ('A', 'B')), (d_BA, ('B', 'A'))):
                if d < bond_threshold:
                    bonds.append((i, j, *tag))
                    adjacency[i].add(j)
                    adjacency[j].add(i)

    visited = set()
    clusters = []
    for start in range(n):
        if start in visited:
            continue
        cluster, queue = [], [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            queue.extend(adjacency[node] - visited)
        clusters.append(cluster)
    return clusters, bonds


def compute_yields_from_frame(positions, orientations, r_patch, opening_angle,
                              bond_threshold=BOND_THRESHOLD):
    """
    ``Y[s] = (# clusters of size s) / (# clusters of any recognised polygon size)``
    for one frame.

    Returns ``(yields, structure_counts, monomer_count, total_structures)``.
    """
    clusters, _ = detect_clusters(positions, orientations, r_patch,
                                  opening_angle, bond_threshold)
    structure_counts = {s: 0 for s in STRUCTURES}
    monomer_count = 0
    for cluster in clusters:
        name = classify_cluster(len(cluster))
        if name in structure_counts:
            structure_counts[name] += 1
        elif name == 'monomer':
            monomer_count += 1
    total = sum(structure_counts.values())
    yields = {s: (structure_counts[s] / total if total > 0 else 0.0) for s in STRUCTURES}
    return yields, structure_counts, monomer_count, total


# ── Shape-match helper ─────────────────────────────────────────────────────

def shape_match_loss(cluster_centers, ref_shape) -> float:
    """
    Mean absolute difference between the sorted pairwise-distance rows of a
    candidate cluster and those of a reference shape.  Lower = better match.
    """
    cluster_centers = np.asarray(cluster_centers)
    ref_shape = np.asarray(ref_shape)
    if len(cluster_centers) != len(ref_shape):
        return np.inf
    ref_sorted = np.sort(squareform(pdist(ref_shape)), axis=1)
    cl_sorted = np.sort(squareform(pdist(cluster_centers)), axis=1)
    return float(np.mean(np.abs(cl_sorted[:, 1:] - ref_sorted[:, 1:])))


# ── freud spatial clustering + shape validation ────────────────────────────

def freud_ngon_candidates(patch_clouds, box_size, max_dist, ref_size, r_min=0.001):
    """
    Sorted index-tuples that form a size-``ref_size`` freud cluster in **any** of
    the patch point clouds.

    ``patch_clouds`` — list of ``(M, 2)`` arrays (one per patch type), all with
    the same row order (row ``k`` = particle ``k``).  ``freud`` is imported
    lazily so importing this module never requires it.
    """
    import freud
    box = freud.box.Box(box_size, box_size, is2D=True)
    candidates = set()
    for cloud in patch_clouds:
        cloud = np.asarray(cloud)
        pts = np.hstack((cloud, np.zeros((len(cloud), 1))))
        cl = freud.cluster.Cluster()
        cl.compute((box, pts), neighbors={'r_max': max_dist, 'r_min': r_min})
        for c in cl.cluster_keys:
            if len(c) == ref_size:
                candidates.add(tuple(sorted(int(x) for x in c)))
    return candidates


def count_polygons(centers, orientations, opening_angle, box_size,
                   patch_allowance=PATCH_ALLOWANCE, cluster_check=0.5):
    """
    Count triangles … octagons in a configuration using freud clustering of the
    patch point clouds, validated by :func:`shape_match_loss` against the ideal
    n-gon.  Returns a dict ``{name: count}`` including ``'monomers'``.
    """
    centers = np.asarray(centers)
    pA, pB = patch_positions(centers, np.asarray(orientations), opening_angle, CENTER_RADIUS)
    max_dist = patch_separation(opening_angle) + patch_allowance

    counts = {name: 0 for name in STRUCTURES}
    counts['monomers'] = 0
    counts['other'] = 0
    found = set()

    for name in STRUCTURES:
        ref_cl_size = NGON_SIZES[name]
        ref_shape = regular_ngon_reference(
            ref_cl_size, R=1.0 / np.sin(np.pi / ref_cl_size),
            opening_angle_rad=opening_angle,
        )[0]
        for key in freud_ngon_candidates([np.asarray(pA), np.asarray(pB)],
                                         box_size, max_dist, ref_cl_size):
            if key in found:
                continue
            if shape_match_loss(centers[np.array(key)], ref_shape) < cluster_check:
                counts[name] += 1
                found.add(key)

    clustered = set()
    for key in found:
        clustered.update(key)
    counts['monomers'] = len(centers) - len(clustered)
    return counts
