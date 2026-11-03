#!/bin/bash
# submit_nvt.sh — SLURM array: canonical NVT run per opening angle (one core each).
#
# Only the opening angle differs between tasks.  Box side follows from the area
# fraction:  L^2 = N * pi / PHI.  Runs resume from their checkpoint, so a requeued
# or resubmitted task continues.
#
# Usage (all overridable from the environment):
#   sbatch simulation/runners/slurm/submit_nvt.sh
#   PHI=0.1 N_PROD=1000000 E_AB=10 sbatch simulation/runners/slurm/submit_nvt.sh
# On a GPU (the Nose-Hoover chain dominates a GPU step; 3/1/1 is ~7x cheaper than
# the default 5/2/3):
#   PLAT=gpu NHC_CHAIN=3 NHC_CHAIN_STEPS=1 NHC_SY=1 \
#   sbatch --partition=liu_titan --qos=liu --gres=gpu:1 --cpus-per-task=2 \
#          simulation/runners/slurm/submit_nvt.sh
#
#SBATCH -t 2-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --job-name=NVT
#SBATCH --output=Logs/NVT_%A_%a.out
#SBATCH --error=Logs/NVT_%A_%a.err

N=${N:-1000}
PHI=${PHI:-0.2}                  # area fraction
N_PROD=${N_PROD:-10000000}       # MD steps
DT=${DT:-1e-3}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-1000}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-500000}
SEED=${SEED:-42}
E_AA=${E_AA:-1.0}
E_BB=${E_BB:-1.0}
E_AB=${E_AB:-5.0}
NHC_CHAIN=${NHC_CHAIN:-5}        # Nose-Hoover chain links / sub-steps / Suzuki-Yoshida order
NHC_CHAIN_STEPS=${NHC_CHAIN_STEPS:-2}
NHC_SY=${NHC_SY:-3}
PLAT=${PLAT:-cpu}
OUT_DIR=${OUT_DIR:-/data2/shared/felipetm/selfAssembly/nvt_N${N}_phi${PHI}_EAA${E_AA}_EBB${E_BB}_EAB${E_AB}_nhc${NHC_CHAIN}${NHC_CHAIN_STEPS}${NHC_SY}_${N_PROD}steps}

ANGLES_DEG=(70 90 100)
ANGLE_DEG=${ANGLES_DEG[$SLURM_ARRAY_TASK_ID]}
OPENING_ANGLE=$(python3 -c "import math; print(math.radians(${ANGLE_DEG}))")
BOX_AREA=$(python3 -c "import math; print(${N} * math.pi / ${PHI})")

cd $SLURM_SUBMIT_DIR
mkdir -p Logs ${OUT_DIR}

eval "$(conda shell.bash hook)"
conda activate selfAssemblyIG
if [ "$PLAT" = cpu ]; then
    export JAX_PLATFORMS=cpu
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
fi

echo "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "angle=${ANGLE_DEG} deg  N=${N}  phi=${PHI}  box_area=${BOX_AREA}  steps=${N_PROD}  dt=${DT}"
echo "E_AA=${E_AA} E_BB=${E_BB} E_AB=${E_AB}  NHC ${NHC_CHAIN}/${NHC_CHAIN_STEPS}/${NHC_SY}  platform=${PLAT}"

python simulation/runners/run.py --ensemble canonical \
    --opening_angle ${OPENING_ANGLE} \
    --E_AB ${E_AB} --E_AA ${E_AA} --E_BB ${E_BB} \
    --box_area ${BOX_AREA} --N ${N} --dt ${DT} \
    --n_equil 0 --n_prod ${N_PROD} \
    --sample_interval ${SAMPLE_INTERVAL} --checkpoint_interval ${CHECKPOINT_INTERVAL} \
    --nhc_chain_length ${NHC_CHAIN} --nhc_chain_steps ${NHC_CHAIN_STEPS} --nhc_sy_steps ${NHC_SY} \
    --seed ${SEED} --resume \
    --output ${OUT_DIR}/theta${ANGLE_DEG}.h5
