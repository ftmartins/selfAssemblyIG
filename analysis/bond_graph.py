"""
Bond-graph clusters and equilibrium yields from a trajectory file (ADR-0002).

Definitions (CONTEXT.md)
------------------------
* Bond: an A-B patch pair on two different particles closer than the bond
  distance (default: the patch radius a).  Patch sites lie on the core surface.
* Cluster: connected component of the bond graph, no shape filter.
* Ring: cluster of n particles in which every particle has both patches
  bonded.  Every other cluster of k >= 2 particles is an open chain of length
  k; a lone particle is a monomer.
* Structure labels: ``'monomer'``, ``'chain_<k>'``, ``'ring_<n>'`` for n <= 12,
  ``'ring_13+'`` for larger rings (see :func:`structure_label`).
* Particle yield: fraction of particles in a structure.  Cluster yield:
  fraction of clusters (monomers included) that are a structure.  Both are
  equilibrium yields when averaged over a production run; they are NOT the
  snapshot yield of ``base.structure_detection`` (shape-filtered, finite-time,
  kept for validation/literature comparison only).

Public API (small; this module imports no simulation engine)
------------------------------------------------------------
``classify_frame(positions, orientations, box_size, opening_angle,
bond_distance=None) -> FrameStructures``
    One frame -> ``.clusters`` (list of :class:`Cluster`: kind, size,
    members), ``.cluster_id`` (N,) per-particle cluster index,
    ``.cluster_counts`` / ``.particle_counts`` (label -> count).  Mass
    conservation (particle counts sum to N) is checked; periodic boundaries
    via a periodic cKDTree.

``yields(source, phase='production', start=None, stop=None, step=None,
bond_distance=None, n_blocks=10, opening_angle=None, box_size=None) -> Yields``
    ``source`` is a path or an open reader (HDF5 or legacy).  Frames are read
    lazily from the store and classified; ``Yields`` holds, per label,
    ``particle_yield`` / ``cluster_yield`` (mean over frames) and
    ``*_err`` (block-averaged standard error: frames are split into
    ``n_blocks`` contiguous blocks, error = std of block means / sqrt(blocks)),
    plus per-frame series ``particle_series`` / ``cluster_series`` and
    ``n_frames``, ``n_seeds``, ``bond_distance``, ``attrs``.  Empty frames
    (N = 0) are skipped.  Opening angle and box default to the file attrs.

``combine_seeds(list_of_Yields) -> Yields``
    Equal-weight mean over seeds; error = seed-to-seed standard error
    (std ddof=1 / sqrt(S)); one seed passes through unchanged.

``structure_label(kind, size)`` builds the label strings.
"""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from base.config import CENTER_RADIUS, PATCH_RADIUS
from base.geometry import patch_positions
from base.trajectory import open_trajectory

MAX_INDIVIDUAL_RING = 12


def structure_label(kind, size):
    """'monomer' | 'chain_<k>' | 'ring_<n>' (n <= 12) | 'ring_13+'."""
    if kind == 'monomer':
        return 'monomer'
    if kind == 'ring' and size > MAX_INDIVIDUAL_RING:
        return f'ring_{MAX_INDIVIDUAL_RING + 1}+'
    return f'{kind}_{size}'


@dataclass
class Cluster:
    kind: str                 # 'monomer' | 'chain' | 'ring'
    size: int                 # number of particles
    members: np.ndarray       # particle indices

    @property
    def label(self):
        return structure_label(self.kind, self.size)


@dataclass
class FrameStructures:
    n_particles: int
    clusters: list
    cluster_id: np.ndarray    # (N,) index into ``clusters``

    @property
    def cluster_counts(self):
        return dict(Counter(c.label for c in self.clusters))

    @property
    def particle_counts(self):
        out = Counter()
        for c in self.clusters:
            out[c.label] += c.size
        return dict(out)


def classify_frame(positions, orientations, box_size, opening_angle,
                   bond_distance=None):
    """Classify every cluster of one frame (periodic square box of side
    ``box_size``).  See the module docstring for the definitions."""
    bond_distance = PATCH_RADIUS if bond_distance is None else float(bond_distance)
    pos = np.asarray(positions, float).reshape(-1, 2)
    ori = np.asarray(orientations, float).reshape(-1)
    n = len(pos)
    if n == 0:
        return FrameStructures(0, [], np.zeros(0, int))
    L = float(box_size)
    pA, pB = patch_positions(pos, ori, opening_angle, CENTER_RADIUS)
    # boxsize needs coordinates in [0, L); round-off can land exactly on L
    pA, pB = np.mod(pA, L), np.mod(pB, L)
    pA[pA >= L], pB[pB >= L] = 0.0, 0.0
    tA, tB = cKDTree(pA, boxsize=L), cKDTree(pB, boxsize=L)
    d = tA.sparse_distance_matrix(tB, bond_distance, output_type='ndarray')
    keep = (d['v'] < bond_distance) & (d['i'] != d['j'])
    ia, jb = d['i'][keep], d['j'][keep]           # A of ia bonded to B of jb
    a_bonded = np.zeros(n, bool)
    b_bonded = np.zeros(n, bool)
    a_bonded[ia], b_bonded[jb] = True, True
    adj = coo_matrix((np.ones(len(ia)), (ia, jb)), shape=(n, n))
    ncomp, comp = connected_components(adj, directed=False)
    both = a_bonded & b_bonded
    clusters = []
    for k in range(ncomp):
        members = np.flatnonzero(comp == k)
        size = len(members)
        if size == 1:
            kind = 'monomer'
        elif both[members].all():
            kind = 'ring'
        else:
            kind = 'chain'
        clusters.append(Cluster(kind, size, members))
    fs = FrameStructures(n, clusters, comp)
    if sum(fs.particle_counts.values()) != n:
        raise AssertionError('mass conservation violated in classify_frame')
    return fs


# ── yields over a trajectory ───────────────────────────────────────────────

@dataclass
class Yields:
    particle_yield: dict
    particle_yield_err: dict
    cluster_yield: dict
    cluster_yield_err: dict
    n_frames: int
    n_seeds: int = 1
    bond_distance: float = PATCH_RADIUS
    particle_series: dict = field(default_factory=dict)   # label -> (F,)
    cluster_series: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)


def _block_mean_err(x, n_blocks):
    """Mean of a per-frame series and its block-averaged standard error."""
    x = np.asarray(x, float)
    nb = max(1, min(int(n_blocks), len(x)))
    means = np.array([b.mean() for b in np.array_split(x, nb)])
    err = means.std(ddof=1) / np.sqrt(nb) if nb > 1 else np.nan
    return float(x.mean()), float(err)


def _summarise(series, n_blocks):
    mean, err = {}, {}
    for k, v in series.items():
        mean[k], err[k] = _block_mean_err(v, n_blocks)
    return mean, err


def yields(source, phase='production', start=None, stop=None, step=None,
           bond_distance=None, n_blocks=10, opening_angle=None, box_size=None):
    """Particle and cluster yields (with block-averaged errors) of a trajectory."""
    reader = open_trajectory(source) if isinstance(source, (str, bytes)) or hasattr(
        source, '__fspath__') else source
    attrs = dict(reader.attrs)
    theta = float(attrs['opening_angle'] if opening_angle is None else opening_angle)
    L = float(attrs['box_size'] if box_size is None else box_size)
    bd = PATCH_RADIUS if bond_distance is None else float(bond_distance)
    per_frame = []
    for fr in reader.frames(phase, start, stop, step):
        if fr.N == 0:
            continue
        s = classify_frame(fr.positions, fr.orientations, L, theta, bd)
        per_frame.append((fr.N, s.particle_counts, s.cluster_counts))
    if not per_frame:
        raise ValueError('no non-empty frames to analyse')
    labels = sorted({k for _, p, _ in per_frame for k in p}, key=_label_key)
    F = len(per_frame)
    ps = {k: np.array([p.get(k, 0) / N for N, p, _ in per_frame]) for k in labels}
    cs = {k: np.array([c.get(k, 0) / sum(c.values()) for _, _, c in per_frame])
          for k in labels}
    pm, pe = _summarise(ps, n_blocks)
    cm, ce = _summarise(cs, n_blocks)
    return Yields(pm, pe, cm, ce, F, 1, bd, ps, cs, attrs)


def _label_key(label):
    if label == 'monomer':
        return (0, 0)
    kind, _, size = label.partition('_')
    return (1 if kind == 'chain' else 2, int(size.rstrip('+')))


def combine_seeds(results):
    """Combine per-seed :class:`Yields`: equal-weight mean, seed-to-seed
    standard error (needs >= 2 seeds; a single seed is returned unchanged)."""
    results = list(results)
    if len(results) == 1:
        return results[0]
    S = len(results)
    labels = sorted({k for r in results for k in r.particle_yield}, key=_label_key)

    def comb(attr):
        mean, err = {}, {}
        for k in labels:
            v = np.array([getattr(r, attr).get(k, 0.0) for r in results])
            mean[k], err[k] = float(v.mean()), float(v.std(ddof=1) / np.sqrt(S))
        return mean, err

    pm, pe = comb('particle_yield')
    cm, ce = comb('cluster_yield')
    return Yields(pm, pe, cm, ce, sum(r.n_frames for r in results), S,
                  results[0].bond_distance, attrs=dict(results[0].attrs))
