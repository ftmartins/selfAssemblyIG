"""Analysis seam: hand-built synthetic HDF5 trajectories (written through the
store's writer) go in; cluster classification and yields come out.

Geometry is built here independently of the analysis module: particles touch
at their patch sites, which lie on the core surface (patch-site radius = center
radius = 1), so a bond A_j--B_i has patch gap 0 when centers are 2 apart."""
import numpy as np
import pytest

from base.config import PATCH_RADIUS as A
from base.trajectory import TrajectoryWriter
from analysis import bond_graph as bg

pytestmark = pytest.mark.fast
BOX = 40.0


def build_chain(k, theta, origin=(0.0, 0.0), phi0=0.0, gap=0.0):
    """k particles joined B_i -> A_{i+1} by straight (unstrained) A-B bonds.
    Closes into a regular ring exactly when theta = pi (k-2)/k.  ``gap`` pulls
    each successive particle further along the bond axis."""
    c = np.zeros((k, 2))
    th = np.zeros(k)
    c[0] = origin
    phi = phi0
    th[0] = phi - theta
    for i in range(k - 1):
        c[i + 1] = c[i] + (2.0 + gap) * np.array([np.cos(phi), np.sin(phi)])
        th[i + 1] = phi + np.pi
        phi = phi + theta + np.pi
    return c, th


def ring(n, origin=(0.0, 0.0)):
    return build_chain(n, np.pi * (n - 2) / n, origin)


def monomers(origins):
    o = np.asarray(origins, float)
    return o, np.zeros(len(o))


def stack(*parts):
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]))


def write_traj(path, frames, theta, box=BOX, phase='production', seed=0):
    with TrajectoryWriter(path, dict(ensemble='canonical', box_size=box,
                                     opening_angle=theta, seed=seed)) as w:
        for i, (pos, ori) in enumerate(frames):
            w.append_frame(pos % box, ori, 0.0, step=i, phase=phase)
    return path


def classify(frame, theta, **kw):
    pos, ori = frame
    return bg.classify_frame(pos % BOX, ori, BOX, theta, **kw)


def test_monomers_and_square_ring():
    th = np.pi / 2
    f = stack(monomers([[5, 5], [15, 5]]), ring(4, (25, 25)))
    s = classify(f, th)
    assert s.cluster_counts == {'monomer': 2, 'ring_4': 1}
    assert s.particle_counts == {'monomer': 2, 'ring_4': 4}


def test_triangle_ring():
    f = ring(3, (10, 10))
    assert classify(f, np.pi / 3).cluster_counts == {'ring_3': 1}


def test_open_chain_is_not_a_ring():
    th = 2.2                       # not a matching angle: 4-chain stays open
    f = build_chain(4, th, (10, 10))
    assert classify(f, th).cluster_counts == {'chain_4': 1}


def test_ring_versus_chain_same_particle_count():
    sq = classify(ring(4, (10, 10)), np.pi / 2)
    ch = classify(build_chain(4, 2.2, (10, 10)), 2.2)
    assert sq.particle_counts == {'ring_4': 4}
    assert ch.particle_counts == {'chain_4': 4}


def test_ring_straddling_periodic_boundary_is_one_ring():
    pos, ori = ring(4, (BOX - 1.0, BOX - 1.0))       # crosses both edges
    assert np.ptp(pos % BOX, axis=0).max() > BOX / 2  # really split by wrapping
    assert classify((pos, ori), np.pi / 2).cluster_counts == {'ring_4': 1}


def test_bond_distance_changes_classification_around_patch_radius():
    th = 2.2
    inside = build_chain(2, th, (10, 10), gap=0.9 * A)
    outside = build_chain(2, th, (10, 10), gap=1.1 * A)
    assert classify(inside, th).cluster_counts == {'chain_2': 1}
    assert classify(outside, th).cluster_counts == {'monomer': 2}
    assert classify(outside, th, bond_distance=1.2 * A).cluster_counts == {'chain_2': 1}
    assert classify(inside, th, bond_distance=0.8 * A).cluster_counts == {'monomer': 2}


def test_ring_with_one_bond_just_outside_is_a_chain():
    th = np.pi / 2
    pos, ori = ring(4, (10, 10))
    pos = pos.copy()
    # pull particle 3 along the 2->3 bond axis by 1.1 a: bond 2-3 breaks
    d = pos[3] - pos[2]
    pos[3] = pos[2] + d / np.linalg.norm(d) * (2 + 1.1 * A)
    s = classify((pos, ori), th)
    assert not any(k.startswith('ring') for k in s.cluster_counts)
    assert sum(s.particle_counts.values()) == 4


# ── yields over a trajectory file ───────────────────────────────────────────

def test_yields_match_hand_computed_values(tmp_path):
    th = np.pi / 3
    # frame 0: 2 monomers + 2 triangles          (N=8, 4 clusters)
    f0 = stack(monomers([[5, 5], [15, 5]]), ring(3, (25, 25)), ring(3, (5, 25)))
    # frame 1: 6 monomers + 1 triangle           (N=9, 7 clusters)
    f1 = stack(monomers([[5, 5], [15, 5], [25, 5], [35, 5], [5, 15], [15, 15]]),
               ring(3, (25, 25)))
    p = write_traj(tmp_path / 'y.h5', [f0, f1], th)
    y = bg.yields(p, n_blocks=2)
    assert y.n_frames == 2
    assert y.particle_yield['monomer'] == pytest.approx((2 / 8 + 6 / 9) / 2)
    assert y.particle_yield['ring_3'] == pytest.approx((6 / 8 + 3 / 9) / 2)
    assert y.cluster_yield['monomer'] == pytest.approx((2 / 4 + 6 / 7) / 2)
    assert y.cluster_yield['ring_3'] == pytest.approx((2 / 4 + 1 / 7) / 2)
    # frame-level mass conservation: yields sum to one
    assert sum(y.particle_yield.values()) == pytest.approx(1.0)
    assert sum(y.cluster_yield.values()) == pytest.approx(1.0)
    # two blocks of one frame: error is the standard error of the block means
    sd = np.std([2 / 8, 6 / 9], ddof=1) / np.sqrt(2)
    assert y.particle_yield_err['monomer'] == pytest.approx(sd)


def test_large_rings_are_grouped_above_twelve():
    assert bg.structure_label('ring', 12) == 'ring_12'
    assert bg.structure_label('ring', 13) == 'ring_13+'
    assert bg.structure_label('ring', 20) == 'ring_13+'
    assert bg.structure_label('chain', 20) == 'chain_20'


def test_combine_seeds_averages_and_uses_seed_to_seed_error(tmp_path):
    th = np.pi / 3
    fm = stack(monomers([[5, 5], [15, 5], [25, 5]]))        # all monomers
    fr = ring(3, (10, 10))                                   # all ring
    ys = [bg.yields(write_traj(tmp_path / f's{i}.h5', [f] * 4, th, seed=i), n_blocks=2)
          for i, f in enumerate([fm, fr])]
    c = bg.combine_seeds(ys)
    assert c.n_seeds == 2
    assert c.particle_yield['monomer'] == pytest.approx(0.5)
    assert c.particle_yield['ring_3'] == pytest.approx(0.5)
    assert c.particle_yield_err['ring_3'] == pytest.approx(0.5)   # std([0,1])/sqrt2
    assert c.cluster_yield['ring_3'] == pytest.approx(0.5)
    assert bg.combine_seeds(ys[:1]).particle_yield == ys[0].particle_yield
