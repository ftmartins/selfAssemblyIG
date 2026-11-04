#!/bin/sh
# submit_partition.sh — SLURM array over the (theta, E_AB) grid of
# base.grids.PARTITION_ANGLES_DEG x PARTITION_EAB_VALUES (E_AB fastest).
#
# Each task runs simulation/runners/scan_partition.py --job_id for one point and
# writes data/partition/oa<deg>_EAB<E>.npz.  Points already on disk with the
# same settings are skipped, so resubmitting only fills gaps.
#
# IMPORTANT: --array upper bound = NUM_PARTITION_JOBS - 1 (currently 17 * 9 = 153).
#   python -c "from base.grids import NUM_PARTITION_JOBS; print(NUM_PARTITION_JOBS - 1)"
#
# Usage:
#   sbatch simulation/runners/slurm/submit_partition.sh
#
#SBATCH -t 0-04:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --array=0-152
#SBATCH --begin=now
#SBATCH --job-name=PartitionScan
#SBATCH --output=Logs/PartitionScan_%A_%a.out
#SBATCH --error=Logs/PartitionScan_%A_%a.err

N_WALKERS=${N_WALKERS:-16000}
N_CHAIN_MAX=${N_CHAIN_MAX:-16}
N_RING_MAX=${N_RING_MAX:-12}

cd $SLURM_SUBMIT_DIR
mkdir -p Logs

eval "$(conda shell.bash hook)"
conda activate selfAssemblyIG

echo "Job ${SLURM_JOB_ID}  task ${SLURM_ARRAY_TASK_ID}  host $(hostname)  $(date)"

python simulation/runners/scan_partition.py \
    --job_id      ${SLURM_ARRAY_TASK_ID} \
    --n_walkers   ${N_WALKERS}   \
    --n_chain_max ${N_CHAIN_MAX} \
    --n_ring_max  ${N_RING_MAX}

echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
