#!/bin/sh
#SBATCH -t 7-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5gb
#SBATCH --array=0
#SBATCH --begin=now
#SBATCH --job-name=PolygonYield
#SBATCH --output=/home1/felipetm/selfAssemblyIG/Logs/PolyYield_Test_%A_%a.out
#SBATCH --error=/home1/felipetm/selfAssemblyIG/Logs/PolyYield_Test_%A_%a.err


cd $SLURM_SUBMIT_DIR/
eval "$(conda shell.bash hook)"
conda activate /home1/felipetm/.conda/envs/selfAssemblyIG

# Run simulations with different seeds
# You can now specify custom seeds for each simulation run

# Square with seed 1273 (default)
python run_yield_simulation.py  --params=optimal_params/square_params_20251208_101306.npz --seed=1273

# Triangle with seed 1274
python run_yield_simulation.py  --params=optimal_params/triangle_params_20251208_121534.npz --seed=1274

# Example: Run with different seeds for reproducibility testing
# python run_yield_simulation.py  --params=optimal_params/square_params_20251208_101306.npz --seed=2000
# python run_yield_simulation.py  --params=optimal_params/square_params_20251208_101306.npz --seed=3000
