#!/bin/sh
#SBATCH -t 7-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5gb
#SBATCH --array=0-99
#SBATCH --begin=now
#SBATCH --job-name=PolygonYieldMD
#SBATCH --output=/home1/felipetm/selfAssemblyIG/Logs/PolyYieldMD_%A_%a.out
#SBATCH --error=/home1/felipetm/selfAssemblyIG/Logs/PolyYieldMD_%A_%a.err

cd $SLURM_SUBMIT_DIR/
eval "$(conda shell.bash hook)"
conda activate /home1/felipetm/.conda/envs/selfAssemblyIG

# Preview grid (only from first task, for the log)
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python generate_simulation_tasks.py
fi

python run_yield_simulation_cluster.py $SLURM_ARRAY_TASK_ID
