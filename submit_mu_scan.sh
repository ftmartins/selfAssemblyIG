#!/bin/sh
# submit_mu_scan.sh — SLURM 2-D array: mu × E_AB.
#
# Each task runs mu_scan_gcmc.py for one (mu, E_AB) pair in parallel.
# Opening angle is fixed per submission; submit separately for each angle.
# Aggregate afterwards with aggregate_mu_scan.py --E_AB X --opening_angle Y.
#
# Task mapping:
#   eab_idx = SLURM_ARRAY_TASK_ID % N_EAB
#   mu_idx  = SLURM_ARRAY_TASK_ID // N_EAB
#   → same mu values sweep across all E_AB values simultaneously.
#
# IMPORTANT: update --array=0-N when you change the grid sizes.
#   Total tasks = N_MU * N_EAB  (currently 20 * 6 = 120 → --array=0-119)
#
# Usage:
#   sbatch submit_mu_scan.sh
#
#SBATCH -t 7-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --array=0-119
#SBATCH --begin=now
#SBATCH --job-name=GCMC_MuScan
#SBATCH --output=Logs/GCMCMuScan_%A_%a.out
#SBATCH --error=Logs/GCMCMuScan_%A_%a.err

# ── Fixed physics parameters ──────────────────────────────────────────────────
BOX_AREA=1600           # box area A = L^2
OPENING_ANGLE=1.5708    # opening angle in radians  (pi/2 = 90 deg for square)
KT=1.0                  # thermal energy
E_AA=1.0                # A-A like-patch Morse depth (fixed)
E_BB=1.0                # B-B like-patch Morse depth (fixed)
SEED=42                 # random seed
N_EQUIL=10000           # equilibration sweeps per mu
N_PROD=5000             # production sweeps per mu
N_INIT=16               # starting particle count for IC
F_DISP=0.5              # fraction of moves that are MD displacement
MD_STEPS=5000           # Langevin steps per displacement move
MD_DT=0.01              # Langevin time step
GAMMA=1.0               # Langevin friction coefficient

# ── mu grid (N_MU = 20) ───────────────────────────────────────────────────────
# 20 values from -6.0 to -1.25, step 0.25
# Update --array upper bound when changing this list: new_upper = N_MU * N_EAB - 1
MU_VALUES=(-6.00 -5.75 -5.50 -5.25 -5.00 -4.75 -4.50 -4.25 -4.00 -3.75 \
           -3.50 -3.25 -3.00 -2.75 -2.50 -2.25 -2.00 -1.75 -1.50 -1.25)
N_MU=20

# ── E_AB grid (N_EAB = 6) ────────────────────────────────────────────────────
# 6 values from 1 to 50 on a roughly logarithmic scale
# Update --array upper bound when changing this list: new_upper = N_MU * N_EAB - 1
EAB_VALUES=(1.0 2.0 5.0 10.0 20.0 50.0)
N_EAB=6

# ── Index arithmetic ──────────────────────────────────────────────────────────
EAB_IDX=$(( SLURM_ARRAY_TASK_ID % N_EAB ))
MU_IDX=$(( SLURM_ARRAY_TASK_ID / N_EAB ))

MU=${MU_VALUES[$MU_IDX]}
E_AB=${EAB_VALUES[$EAB_IDX]}

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR=$(python3 -c "from config_gcmc import MU_SCAN_DIR; print(MU_SCAN_DIR)")

# ── Environment ───────────────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR
mkdir -p Logs
mkdir -p ${OUTPUT_DIR}

eval "$(conda shell.bash hook)"
conda activate selfAssemblyIG

echo "=========================================="
echo "Job:    ${SLURM_JOB_ID}  Task: ${SLURM_ARRAY_TASK_ID}"
echo "mu =    ${MU}  (idx ${MU_IDX})"
echo "E_AB =  ${E_AB}  (idx ${EAB_IDX})"
echo "opening_angle = ${OPENING_ANGLE} rad"
echo "Host:   $(hostname)"
echo "Time:   $(date)"
echo "=========================================="

python mu_scan_gcmc.py \
    --mu            ${MU}            \
    --box_area      ${BOX_AREA}      \
    --opening_angle ${OPENING_ANGLE} \
    --kT            ${KT}            \
    --E_AB          ${E_AB}          \
    --E_AA          ${E_AA}          \
    --E_BB          ${E_BB}          \
    --seed          ${SEED}          \
    --output_dir    ${OUTPUT_DIR}    \
    --n_equil       ${N_EQUIL}       \
    --n_prod        ${N_PROD}        \
    --N_init        ${N_INIT}        \
    --f_disp        ${F_DISP}        \
    --md_steps      ${MD_STEPS}      \
    --md_dt         ${MD_DT}         \
    --gamma         ${GAMMA}

echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
