"""
Optimization functions for patchy particle design.

This module contains:
- Random parameter generation
- Random search initialization
- Main optimization loop with gradient descent
- Loss computation and gradient calculation
"""

import time
import numpy as np
import jax.numpy as jnp
from jax import jit, random, value_and_grad, jacfwd, jacrev, hessian
from jax.example_libraries import optimizers

# Import all configuration parameters
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_patchy_particle import *

# Import utility functions
from .utility_functions import (
    v_run_my_sim, make_params, run_partial_sim,
    run_sim_and_get_positions, save_position, bprint
)

# Import evaluation functions
from .evaluation_functions import avg_loss, make_cluster_list

print("Loading Self-Assembly Optimizer Functions")
def get_mean_loss(params, initial_position, keys):
    states, _, _ = v_run_my_sim(params, initial_position, SQRT_NUM_STEPS_TO_OPT, CENTER_RADIUS, keys)
    return avg_loss(states.position.center)

g_mean_loss = jit(value_and_grad(get_mean_loss)) #if need to, switch to jacfwd
#g_mean_loss = jit(jacfwd(get_mean_loss)) #this gives nans, no idea why

def find_like(params, num_patches=NUM_PATCHES, buffer = .9):
    likes = np.zeros(num_patches)
    mixed = np.zeros(int(num_patches * (num_patches + 1) / 2) - num_patches)
    like_pos = likes  

    for k in range(1, num_patches):
        like_pos[k] = like_pos[k - 1] + num_patches - k + 1

    l = 0  # counter for like bonds
    m = 0  # counter for mixed bonds
    for i in range(len(params)):
        if i in like_pos:
            likes[l] = params[i]
            l += 1
        else:
            mixed[m] = params[i]
            m += 1
    return all(x < y * buffer for x in likes for y in mixed)
################# Random Search #################
def generate_random_params(key, num_patches, max_energy):
    # generate random parameters for the location (thetas) of the patches on a particle  
    N_eng = int(num_patches * (num_patches + 1) / 2)
    eng_check = False

    thetas = random.uniform(key, (num_patches,), minval=0., maxval=2*jnp.pi)
    while not eng_check:
        key, split = random.split(key)
        energies = random.uniform(split, (N_eng,), minval=0.0, maxval=max_energy)
        eng_check = find_like(energies)
        if RAND_ENG_CHECK:
            eng_check = True

    thetas = thetas.at[0].set(0.)
    params = list(thetas)
    params.extend(list(energies))
    return jnp.array(params)
def random_search(key,
                  n_iterations, 
                  num_patches, 
                  max_energy,
                  batch_size):
  # generate a bunch of different initial conditions, calculate the loss function, then choose the set of parameters with the smallest loss function as the initial condition to do all the following optimization procedure
  # only used in the initial step
  bprint("start random search")
  min_params = generate_random_params(key, num_patches, max_energy)
  key, split = random.split(key)
  simulation_keys = random.split(split, batch_size)
  
  initial_positions = run_partial_sim(min_params, split, batch_size)
  min_loss = get_mean_loss(min_params, initial_positions, simulation_keys)
  for n in range(n_iterations - 1):
    bprint(f"arrived at {n}th param search")
    params = generate_random_params(key, num_patches, max_energy)
    key, split = random.split(key)
    simulation_keys = random.split(split, batch_size)
    initial_positions = run_partial_sim(min_params, split, batch_size)
    loss = get_mean_loss(params, initial_positions, simulation_keys)
    if loss < min_loss:
      min_loss = loss
      min_params = params
  bprint(min_params)
  return min_loss, min_params[1:]


def optimize(input_params,
             key,
             opt_steps,
             batch_size,
             loop_batch,
             save_every,
             tracker,
             learning_rate=0.5,
             cmd='w',
             myoptsteps = 1,
             optimizer = OPTIMIZER,
             last_step = 0,
             cl_type=None):
  
  startopttime = time.time()
  learning_rate_schedule = jnp.ones(opt_steps)*learning_rate
  ind = int(opt_steps / 3)
  learning_rate_schedule = learning_rate_schedule.at[ind:2*ind].set(learning_rate * 0.5)
  learning_rate_schedule = learning_rate_schedule.at[2*ind:].set(learning_rate * 0.1)
  learning_rate_fn = lambda i: learning_rate_schedule[i]

   
  if optimizer == "rms":
    opt_init, opt_update, get_params = optimizers.rmsprop(step_size=learning_rate_fn, gamma=0.9, eps=1e-8)
  else:
    opt_init, opt_update, get_params = optimizers.adam(step_size=learning_rate_fn)

  print(f"Optimizing using the {optimizer} algorithm.")

  loss_file = JOBID+'/loss_'+ tracker[-1] +'.txt'
  param_file = JOBID+'/params_'+ tracker[-1] +'.txt'
  grad_file = JOBID+'/grad_'+ tracker[-1] +'.txt'
  hess_file = JOBID+'/hes_'+ tracker[-1] +'.txt'

  def clip_gradient(g, clip=CLIP):
    return jnp.array(jnp.where(jnp.abs(g) > clip, jnp.sign(g)*clip, g))
  ##########################################################################################################
  def step(stepk,
          opt_state,
          key,
          batch_size=10,
          save_every=10,
          cmd='w',
          cl_type=None):
    bprint(f" Opt State :{opt_state}")
    opt_params = get_params(opt_state)
    run_params = make_params(opt_params)   
    bprint(f"Step Params are: {run_params}")
    key, split = random.split(key)
    simulation_keys = random.split(split, batch_size)
    bprint('Warm Start')
    initial_positions = run_partial_sim(run_params, split, batch_size)

    gs = []
    Hs = []
    ls = 0    
    bprint(f"Initial Positions:{initial_positions}")
    for i in range(loop_batch):
      l, g = g_mean_loss(run_params, initial_positions, simulation_keys)
      if FIND_HESSIAN:
        Hes = h_mean_loss(run_params, initial_positions, simulation_keys)
      g = clip_gradient(g)
      gs += [g]
      if FIND_HESSIAN:
        Hs += [Hes]
      ls += l          
    
    g = jnp.mean(jnp.array(gs), axis = 0)

    if FIND_HESSIAN:
      Hes = jnp.mean(jnp.array(Hs), axis = 0)

    loss = ls / loop_batch
    position,_,_= run_sim_and_get_positions(run_params, initial_positions, simulation_keys)
    batch_shapes = make_cluster_list(position, run_params, batch_size=batch_size, cl_type=cl_type)
    step_shapes = np.sum(batch_shapes)
    g_out = g

    #bprint(f" Shapes from step: {step_shapes}")
    if(stepk%save_every==0):
  
      save_position(run_params, initial_positions, simulation_keys, stepk+myoptsteps, JOBID)

      with open(loss_file, cmd) as outfile1:
        outfile1.write("{}".format(loss)+'\n')

      with open(param_file, cmd) as outfile2:
        outlist2 = np.array(run_params).tolist()
        separator=' '
        outfile2.write(separator.join(['{}'.format(temp) for temp in outlist2])+'\n')

      with open(grad_file, cmd) as outfile3:
        outlist3 = np.array(g_out).tolist()
        separator=' '
        outfile3.write(separator.join(['{}'.format(temp) for temp in outlist3])+'\n')
      if FIND_HESSIAN:
        with open(hess_file, cmd) as outfile3:
          outlist4 = np.array(Hes).tolist()
          separator=' '
          outfile3.write(separator.join(['{}'.format(temp) for temp in outlist4])+'\n')
      
    bprint("Loss: {}".format(loss))
    bprint("Parameters: {}".format(run_params))
    bprint("Gradient: {}".format(g_out))
    if FIND_HESSIAN:
      bprint(f"Hessian: {Hes.shape}") 
    return opt_update(stepk, g[1:], opt_state), loss, step_shapes
    ##########################################################################################################
  print(f"Input params are {input_params}")
  opt_state = opt_init(input_params)
  min_loss_params = max_cl_params = input_params
  min_loss = cl_loss = 1e6
  max_cl = 0 
  
  for i in range(last_step, opt_steps):
    key, split=random.split(key)

    if i == 0:
      steptime = time.time()
      new_opt_state, loss, step_cl  = step(i,
                                 opt_state,
                                 split,
                                 batch_size=batch_size,
                                 save_every=save_every,
                                 cmd=cmd,
                                 cl_type=cl_type)
      inittimefin = int(time.time()-startopttime)
      bprint(f"First step took {inittimefin} seconds to complete.")
      print(f"Step 1/{opt_steps} | Loss: {loss:.6f} | Time: {inittimefin}s")

    elif i == 1:
      steptime = time.time()
      new_opt_state, loss, step_cl = step(i,
                                 opt_state,
                                 split,
                                 batch_size=batch_size,
                                 save_every=save_every,
                                 cmd=cmd,
                                 cl_type=cl_type)
      steptimefin = int(time.time()-steptime+1)
      bprint(f"Second step took {steptimefin} seconds to complete.")
      full_opt_time = (inittimefin + steptimefin * (opt_steps-1)) //60
      print(f"Step 2/{opt_steps} | Loss: {loss:.6f} | Time: {steptimefin}s | Est. total: {full_opt_time}min")


    else:
      steptime = time.time()
      new_opt_state, loss, step_cl = step(i,
                                 opt_state,
                                 split,
                                 batch_size=batch_size,
                                 save_every=save_every,
                                 cmd=cmd,
                                 cl_type=cl_type)
      steptimefin = int(time.time()-steptime)
      bprint(f"Optimization step {i+1} took {steptimefin} seconds to complete.")
      if (i + 1) % save_every == 0:
        elapsed = int((time.time()-startopttime)//60)
        print(f"Step {i+1}/{opt_steps} | Loss: {loss:.6f} | Time: {steptimefin}s | Elapsed: {elapsed}min")
    if loss < min_loss:
      min_loss = loss
      min_loss_params = get_params(opt_state)
    if step_cl > max_cl:
      max_cl = step_cl
      cl_loss = loss
      max_cl_params = get_params(opt_state)    
    opt_state = new_opt_state
    cmd='a'
  myoptsteps = myoptsteps + opt_steps
  finopttime = int((time.time()-startopttime)//60)
  print(f"This optimization took {finopttime} minutes.")
  
  return min_loss, min_loss_params,cl_loss, max_cl_params, get_params(new_opt_state), myoptsteps