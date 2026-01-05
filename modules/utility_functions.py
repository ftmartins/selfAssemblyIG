"""
Utility functions for patchy particle simulations.

This module contains core simulation infrastructure including:
- Shape creation from patch angles
- Energy matrix construction
- Rigid body simulation wrappers
- Initial condition generation
- File management utilities
"""

import os
import numpy as np
import jax.numpy as jnp
from jax import jit, random, lax, vmap, remat
from jax_md import space, simulate, energy, rigid_body
import jax

# Import all configuration parameters
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_patchy_particle import *

def bprint(bug_text, print_bug = bug_print):
     if print_bug:
         print(f"From Bug-Printer: {bug_text}", flush=True)
     else:
         return
     
def rename_file(filepath):
    """Check if a file exists and rename it by appending '_old' and a number if needed.
    
    Args:
        filepath (str): Path to the file to check.
        
    Returns:
        str: Path to the renamed file if renamed, otherwise original filepath.
    """
    if os.path.exists(filepath):
        directory, filename = os.path.split(filepath)
        name, ext = os.path.splitext(filename)
        
        # Start with "_old" and increment number if needed
        new_filepath = os.path.join(directory, f"{name}_old{ext}")
        counter = 1
        
        while os.path.exists(new_filepath):
            new_filepath = os.path.join(directory, f"{name}_old_{counter}{ext}")
            counter += 1
        
        os.rename(filepath, new_filepath)
        print(f"File renamed to: {new_filepath}")
        return new_filepath
    else:
        
        return filepath 

# get_shape, REF_SHAPE, and REF_SHAPE_SIZE are now imported from config_patchy_particle

def body_to_plot(body, thetas,radius=CENTER_RADIUS):
  
  shape = thetas_to_shape(thetas, radius)
  body_pos = vmap(rigid_body.transform, (0, None))(body, shape)
  bodypos = body_pos.reshape(-1, 2)
 
  species_list = jnp.array(list(jnp.arange(NUM_PATCHES+1)) * len(body.center))

  inds_at_id = lambda id: jnp.squeeze(jnp.argwhere(species_list==id))
  center_particles = bodypos[inds_at_id(0)]
  #print(center_particles)
  list_of_patch_particles = []
  
  for i in range(1, len(thetas)+1):
    list_of_patch_particles += [bodypos[inds_at_id(i)]]

  return center_particles, list_of_patch_particles
##########################################Simulation Functions##########################################################
@jit
def thetas_to_shape(thetas = "", radius=CENTER_RADIUS):
  patch_positions = jnp.zeros((len(thetas), 2), dtype = jnp.float64)
  patch_positions = patch_positions.at[:,0].set(radius*jnp.cos(thetas))
  patch_positions = patch_positions.at[:,1].set(radius*jnp.sin(thetas))
  positions = jnp.concatenate((jnp.array([[0.0, 0.0]]), patch_positions), axis=0)
  species = jnp.arange(len(thetas) + 1)
  species = jnp.array(species, dtype = jnp.int32)
  patch_mass = PATCH_MASS * jnp.ones(len(thetas))
  mass = jnp.concatenate((jnp.array([CENTER_MASS]), patch_mass), axis = 0)
  shape = rigid_body.point_union_shape(positions, mass).set(point_species=species)
  return shape

@jit
def energy_matrix(eng):
  i, j =jnp.triu_indices(NUM_PATCHES)
  eng_m = jnp.zeros((NUM_PATCHES, NUM_PATCHES))
  eng_m = eng_m.at[i, j].set(eng)
  return eng_m.at[j, i].set(eng)

def random_IC(thetas_and_energy, key):
  # generate random initial condition (random physical position of each particles)
  import config_patchy_particle
  thetas = thetas_and_energy[:NUM_PATCHES]
  shape = thetas_to_shape(thetas, radius = CENTER_RADIUS)
  # Access NUM_PARTICLES and BOX_SIZE at runtime to get updated values
  num_particles = config_patchy_particle.NUM_PARTICLES
  box_size = config_patchy_particle.BOX_SIZE
  displacement, shift = space.periodic(box_size)
  key, pos_key, angle_key = random.split(key, 3)
  R = box_size * random.uniform(pos_key, (num_particles, 2), dtype=jnp.float64) #random initial position
  angles = random.uniform(angle_key, (len(R),), dtype=jnp.float64) * jnp.pi * 2 #random initial orientation
  body = rigid_body.RigidBody(R, angles)
  return body
v_random_IC = vmap(random_IC, (None, 0))

def run_sim(thetas_and_energy,
            x0,
            num_steps,
            CENTER_RADIUS,
            key,
            kT = 1.,
            print_sim = False,
            print_progress = False):
  # Access BOX_SIZE at runtime to get updated value
  import config_patchy_particle
  box_size = config_patchy_particle.BOX_SIZE

  thetas_and_energy = thetas_and_energy.at[0].set(0.0) #fix one patch position
  thetas = thetas_and_energy[:NUM_PATCHES]
  eng = thetas_and_energy[NUM_PATCHES:]
  eng_mat = energy_matrix(eng)
  shape = thetas_to_shape(thetas, radius=CENTER_RADIUS)
  displacement, shift = space.periodic(box_size)

  morse_eps = jnp.pad(eng_mat, pad_width=(1, 0))
  soft_sphere_eps = jnp.zeros((len(thetas) + 1, len(thetas) + 1))
  soft_sphere_eps = soft_sphere_eps.at[0, 0].set(1.0)
  soft_sphere_eps = 10000*soft_sphere_eps


  ### print parameters
  # Convert JAX arrays to regular Python values for printing
  def to_float(x):
    """Convert JAX traced values to regular floats for printing."""
    try:
      return float(x)
    except:
      return x

  print("="*80)
  print("ENERGY FUNCTION PARAMETERS")
  print("="*80)
  print(f"Temperature (kT): {to_float(kT)}")
  print(f"Time Step (dt): {to_float(DT)}")
  print(f"Number of Steps: {num_steps}")
  print(f"Number of Patches: {len(thetas)}")
  print(f"Center Radius: {to_float(CENTER_RADIUS)}")
  print(f"Center Mass: {to_float(CENTER_MASS)}")
  print(f"Patch Mass: {to_float(PATCH_MASS)}")
  print(f"Morse Alpha (ALPHA): {to_float(ALPHA)}")
  print(f"Morse r_cutoff (R_CUTOFF): {to_float(R_CUTOFF)}")
  print(f"Morse sigma: 0.0")
  print(f"Soft Sphere sigma: {to_float(CENTER_RADIUS*2)}")
  print(f"\nMorse Epsilon Matrix (padded):\n{morse_eps}")
  print(f"\nSoft Sphere Epsilon Matrix:\n{soft_sphere_eps}")
  print("="*80)

  pair_energy_soft = energy.soft_sphere_pair(displacement, species=1+len(thetas), sigma = CENTER_RADIUS*2, epsilon=soft_sphere_eps)
  pair_energy_morse = energy.morse_pair(displacement, species=1+len(thetas), sigma = 0.0, epsilon=morse_eps, alpha=ALPHA, r_cutoff=R_CUTOFF) #why is r_cutoff =1?
  pair_energy_fn = lambda R, **kwargs: pair_energy_soft(R, **kwargs) + pair_energy_morse(R, **kwargs)
  energy_fn = rigid_body.point_energy(pair_energy_fn, shape) #note, this is for every particle being the SAME shape, will need to amend this for multiple types of particles

  init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, DT, kT)
  step_fn = jit(step_fn)
  state = init_fn(key, x0, mass=shape.mass())

  do_step = lambda state, t: (step_fn(state), (state.position.center, state.position.orientation))
  do_step = jit(do_step)

  # Run simulation and collect trajectory
  inner_steps = jnp.arange(num_steps)
  state, (positions, orientations) = lax.scan(do_step, state, inner_steps)

  return state, positions, orientations

my_sim = jit(run_sim, static_argnums=(2,7))
v_run_my_sim = jit(vmap(my_sim, in_axes=(None, 0, None, None, 0)), static_argnums=2)



def run_partial_sim(params, key, batch_size):
  rand_key, run_key = random.split(key, 2)
  random_IC_keys = random.split(rand_key, batch_size)
  init_positions = v_random_IC(params, random_IC_keys)
  sim_keys = random.split(run_key, batch_size)
  states, _, _ = v_run_my_sim(params, init_positions, QUICK_STEPS, CENTER_RADIUS, sim_keys)
  return states.position

@jit
def run_sim_and_get_positions(param, state, key):
    xf, _, _ = v_run_my_sim(param, state, SQRT_NUM_STEPS_TO_OPT, CENTER_RADIUS, key)
    return xf.position, xf.position.center, xf.position.orientation

def save_position(param, state, key, stepk, file_dir):
    _, center, orientation = run_sim_and_get_positions(param, state, key)
    
    pos_dir = f"{file_dir}/Position_log"
    os.makedirs(pos_dir, exist_ok=True)
  
    np.save(f"{pos_dir}/pos{stepk}.npy", center)
    np.save(f"{pos_dir}/angle{stepk}.npy", orientation)

###################################################

def make_params(opt_params, param_lim =MAX_ENERGY ):

    params = np.zeros((len(opt_params)+1))

    for zet in range(0,len(opt_params)):
        params[zet+1] = opt_params[zet]

    if params[1] < 0:
        params[1] = np.abs(params[1])
    elif params[1]> np.pi:
        params[1] = 2*np.pi - params[1]

    for M in range(NUM_PATCHES,len(params)):
        if params[M] > param_lim:
            params[M] = param_lim
    return jnp.array(params)