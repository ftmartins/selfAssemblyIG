"""
Evaluation functions for patchy particle optimization.

This module contains functions for:
- Loss calculation (distance-based metrics)
- Cluster detection using freud library
- Polygon validation and counting
- Shape matching metrics
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax_md import space
import freud

# Import all configuration parameters
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_patchy_particle import *

# Import utility functions (for body_to_plot)
from .utility_functions import body_to_plot 


def get_desired_dists(ref_shape):
    displacement, shift = space.periodic(BOX_SIZE)
    vdisp = space.map_product(displacement)
    ds = vdisp(ref_shape, ref_shape)
    dists = jnp.sort(space.distance(ds))
    return dists

ref_dists = get_desired_dists(REF_SHAPE)

@jit
def sys_loss(R, ref_shape=None):
    """
    Calculate system loss based on distance matching to reference shape.

    Parameters
    ----------
    R : array
        Particle positions
    ref_shape : array, optional
        Reference shape positions. If None, uses global REF_SHAPE.

    Returns
    -------
    float : Loss value
    """
    # Use global if not provided
    if ref_shape is None:
        ref_shape = REF_SHAPE
        ref_shape_size = REF_SHAPE_SIZE
        ref_dists_local = ref_dists
    else:
        ref_shape_size = len(ref_shape)
        ref_dists_local = get_desired_dists(ref_shape)

    displacement, shift = space.periodic(BOX_SIZE)
    #Location Loss
    vdisp = space.map_product(displacement)
    ds = jnp.sort(space.distance(vdisp(R, R)))
    subtract = lambda R, Rref: R - Rref
    v_subtract = space.map_product(subtract)
    diffs = v_subtract(ds[:, :ref_shape_size], ref_dists_local)
    nearest_nbrs_match_ref_dist = jnp.min(jnp.linalg.norm(diffs, axis=-1), axis=0)

    #Combine Losses
    if NUM_PARTICLES > ref_shape_size:
      other_nbrs_far = ds[:, ref_shape_size:CLOSENESS_PENALTY_NEIGHBORS + ref_shape_size]
      endloss = jnp.sum(nearest_nbrs_match_ref_dist) - CLOSENESS_PENALTY * jnp.mean(other_nbrs_far)

    elif NUM_PARTICLES == ref_shape_size:
      endloss = jnp.sum(nearest_nbrs_match_ref_dist)
    return endloss

# Create vmap version that maps over R_batched but keeps ref_shape fixed
v_loss = vmap(sys_loss, in_axes=(0, None))

@jit
def avg_loss(R_batched, ref_shape=None):
    """
    Calculate average loss over batch.

    Parameters
    ----------
    R_batched : array
        Batch of particle positions
    ref_shape : array, optional
        Reference shape positions. If None, uses global REF_SHAPE.

    Returns
    -------
    float : Average loss value
    """
    losses = v_loss(R_batched, ref_shape)
    return jnp.mean(losses)
  
@jit
def loc_loss(R, ref_shape = REF_SHAPE):
    ref_dists = get_desired_dists(ref_shape)
    ref_size = len(ref_dists)
    displacement, shift = space.periodic(BOX_SIZE)
    #Location Loss
    vdisp = space.map_product(displacement)
    ds = jnp.sort(space.distance(vdisp(R, R)))
    subtract = lambda R, Rref: R - Rref
    v_subtract = space.map_product(subtract)
    diffs = v_subtract(ds[:, :len(ref_shape)], ref_dists)
    nearest_nbrs_match_ref_dist = jnp.min(jnp.linalg.norm(diffs, axis=-1), axis=0)
    if len(R) > ref_size:
      return jnp.nan
    else:
      endloss = jnp.sum(nearest_nbrs_match_ref_dist) 
      return endloss
patch_allowance= PATCH_SIZE*2/3.
def print_loc_loss(state, ref_shape=None):
    if ref_shape is None:
        ref_shape = shape_ID
    loss = loc_loss(state.center,get_shape(ref_shape))
    return loss
def patch_dist(angle_rad, radius = CENTER_RADIUS):
    # Calculate the distance using the formula
    distance = 2 * radius * np.sin(angle_rad / 2.)
    
    return distance
def get_clusters(state, params, cl_type=None, allowance=patch_allowance, cluster_check=.5, print_patch=False, mismatch=False):
    if cl_type is None:
        cl_type = shape_ID
    center_pos, patches_pos = body_to_plot(state, params[:NUM_PATCHES])
    patch_dists = abs(patch_dist(params[1]))

    # Determine reference cluster size and max distance based on shape type
    if cl_type == "Square":
        max_dist, ref_cl_size = patch_dists + allowance, 4
    elif cl_type == "Triangle":
        max_dist, ref_cl_size = patch_dists + allowance, 3
    else:
        print("Unknown Shape")
        max_dist, ref_cl_size = 8 * CENTER_RADIUS + 0.1, NUM_PARTICLES

    min_dist = 0.001  # Avoid zero minimum distance
    box = freud.box.Box(BOX_SIZE, BOX_SIZE, is2D=True)

    # Get cluster information for each patch type
    patch_clusters = []
    for k in range(NUM_PATCHES):
        points_2d = np.hstack((np.array(patches_pos[k]), np.zeros((len(patches_pos[k]), 1))))
        cl = freud.cluster.Cluster()
        cl.compute((box, points_2d), neighbors={'r_max': max_dist, 'r_min': min_dist})

        cl_props = freud.cluster.ClusterProperties()
        cl_props.compute((box, points_2d), cl.cluster_idx)

        cluster_keys = np.array(cl.cluster_keys, dtype=object)
        ref_clusters = tuple(c for c in cluster_keys if len(c) == ref_cl_size)
        patch_clusters.append(ref_clusters)

    patch_A, patch_B = patch_clusters[0], patch_clusters[1]
    set_a = set(map(tuple, patch_A))
    set_b = set(map(tuple, patch_B))

    matches = set_a.intersection(set_b)
    only_in_a = set_a.difference(set_b)
    only_in_b = set_b.difference(set_a)    

    # Helper to validate cluster shapes
    def validate_clusters(cluster_list, cl_check = cluster_check):
        valid = []
        for cluster in cluster_list:
            cluster_indices = np.array(cluster)
            cluster_points = state[cluster_indices]
            loss = print_loc_loss(cluster_points, ref_shape=cl_type)
            if loss < cl_check:
                valid.append(cluster)
        return valid
    if mismatch:
      #print(only_in_a, only_in_b)
      if only_in_a or only_in_b:
          only_in_a = [list(row) for row in set_a]
          only_in_b = [list(row) for row in set_b]
          return only_in_a, only_in_b
      else:
        return None, None
    # Add good extra clusters from A and B
    matches.update(validate_clusters(only_in_a))
    matches.update(validate_clusters(only_in_b))

    # Final filtering based on shape loss
    confirmed_shape = 0
    final_matches = set()
    for cluster in matches:
        cluster_indices = np.array(cluster)
        cluster_points = state[cluster_indices]
        loss = print_loc_loss(cluster_points, ref_shape=cl_type)
        if loss < cluster_check:
            confirmed_shape += 1
            final_matches.add(cluster)

    glow_list = [list(row) for row in final_matches]
    return len(final_matches), final_matches, glow_list

def make_cluster_list(state, params, batch_size=BATCH_SIZE, cl_type=None):
    if cl_type is None:
        cl_type = shape_ID

    squares = np.zeros(batch_size)

    for i in range(batch_size):
        ksquares, _, _ = get_clusters(state[i], params, cl_type=cl_type)

        squares[i] = ksquares
    return squares    
    
#make_params