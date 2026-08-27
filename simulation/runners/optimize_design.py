#!/usr/bin/env python3
"""
optimize_design.py — optimize 2-patch particle parameters for target-shape
self-assembly.

    python simulation/runners/optimize_design.py --shape square

Runs a random-search initialization followed by multi-stage Adam/RMSProp
optimization.  The opening angle between patches is OPTIMIZED, not fixed.
Output: data/design/{shape}_params_{timestamp}.npz  (+ a text summary).
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
from jax import random

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))

from base.jax_setup import configure_jax
from base.geometry import SHAPE_CONFIGS, get_shape
from base.grids import KEY_PARAM_OPT
from base.io import rename_with_backup
from base.paths import DESIGN_DIR
from simulation.src.design import DesignConfig, random_search, optimize


def parse_arguments():
    p = argparse.ArgumentParser(description='Optimize patchy-particle parameters for assembly')
    p.add_argument('--shape', type=str, required=True,
                   choices=['square', 'triangle', 'Square', 'Triangle'])
    p.add_argument('--max_energy', type=float, default=30.0)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--opt_steps', type=int, default=100)
    p.add_argument('--rand_search_iterations', type=int, default=1)
    p.add_argument('--output_dir', type=str, default=str(DESIGN_DIR))
    return p.parse_args()


def main():
    configure_jax()
    args = parse_arguments()
    shape_name = args.shape.capitalize()
    shape_cfg = SHAPE_CONFIGS[shape_name.lower()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    job_dir = rename_with_backup(f"{shape_name}-Assembly_Optimization")
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = DesignConfig(
        batch_size=args.batch_size,
        max_energy=args.max_energy,
        opt_stage_steps=(args.opt_steps,) * 4,
        job_dir=job_dir,
    )
    ref_shape = get_shape(shape_name)

    print("=" * 80)
    print(f"Patchy-particle optimization — target: {shape_name} "
          f"({shape_cfg['description']})")
    print(f"job dir: {job_dir}   opt_steps/stage: {args.opt_steps}")
    print("=" * 80)

    keys = random.split(random.PRNGKey(KEY_PARAM_OPT), 4)

    print("\nRandom parameter search...")
    _, start_params = random_search(
        keys[0], args.rand_search_iterations,
        DesignConfig(batch_size=args.batch_size * 3, max_energy=args.max_energy,
                     job_dir=job_dir),
        ref_shape,
    )
    print(f"  start params: {start_params}")

    stage_names = ["Coarse", "Medium", "Fine"]
    lrs = cfg.learning_rates
    opt_params = start_params
    myoptsteps = 1
    stage_results = []
    for i, name in enumerate(stage_names):
        print(f"\n{'=' * 80}\nStage {i + 1}: {name}  (lr={lrs[i]})\n{'=' * 80}")
        res = optimize(
            input_params=opt_params, key=keys[i + 1], opt_steps=args.opt_steps,
            cfg=cfg, tracker=f"stage{i + 1}", learning_rate=lrs[i],
            cmd='w' if i == 0 else 'a', myoptsteps=myoptsteps, cl_type=shape_name,
        )
        stage_results.append(res)
        opt_params = res[3]        # max-cluster params
        myoptsteps = res[5]

    optimized = stage_results[-1][3]
    print(f"\n{'=' * 80}\nFINAL PARAMS: {optimized}")
    print(f"opening angle: {np.rad2deg(float(optimized[0])):.2f} deg (optimized)\n{'=' * 80}")

    out_npz = os.path.join(args.output_dir, f"{shape_name.lower()}_params_{timestamp}.npz")
    np.savez(
        out_npz,
        shape=shape_name.lower(),
        final_params=np.array(optimized),
        timestamp=timestamp,
        min_params_stage1=np.array(stage_results[0][1]),
        cluster_params_stage1=np.array(stage_results[0][3]),
        min_params_stage2=np.array(stage_results[1][1]),
        cluster_params_stage2=np.array(stage_results[1][3]),
        min_params_stage3=np.array(stage_results[2][1]),
        cluster_params_stage3=np.array(stage_results[2][3]),
        num_patches=cfg.num_patches,
        max_energy=args.max_energy,
        batch_size=args.batch_size,
        learning_rates=np.array(cfg.learning_rates),
        optimizer=cfg.optimizer,
    )
    summary = os.path.join(args.output_dir, f"{shape_name.lower()}_summary_{timestamp}.txt")
    with open(summary, 'w') as f:
        f.write(f"Patchy Particle Optimization — {shape_name}\n")
        f.write(f"Timestamp: {timestamp}\nOptimizer: {cfg.optimizer}\n")
        f.write(f"Final params: {np.array(optimized).tolist()}\n")
        f.write(f"Opening angle (optimized): {np.rad2deg(float(optimized[0])):.2f} deg\n")
    print(f"Saved: {out_npz}\n       {summary}")
    print(f"\nNext: python simulation/runners/run_yield.py --params {out_npz}")


if __name__ == "__main__":
    main()
