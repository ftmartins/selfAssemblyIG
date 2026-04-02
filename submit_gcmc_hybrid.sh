#!/bin/sh
# submit_gcmc_hybrid.sh — SLURM 3-D array: opening_angle × E_AB × realization.
#
# Runs run_gcmc_hybrid.py for every combination in the parameter grid,
# with N_REAL independent realizations per combination.
#
# Task mapping (innermost index varies fastest):
#   real_idx  = SLURM_ARRAY_TASK_ID % N_REAL
#   param_idx = SLURM_ARRAY_TASK_ID // N_REAL
#   eab_idx   = param_idx % N_EAB
#   angle_idx = param_idx // N_EAB
#
# IMPORTANT: update --array=0-N when you change the grid.
#   Total tasks = N_ANGLES * N_EAB * N_REAL  (currently 10 * 6 * 5 = 300 → --array=0-299)
#
# Provide --mu for each submission; find mu* first with the mu-scan workflow:
#   sbatch submit_mu_scan.sh
#   python aggregate_mu_scan.py --E_AB X --opening_angle Y
#
# Usage:
#   MU=-4.0 sbatch submit_gcmc_hybrid.sh      # pass mu as environment variable
#   or edit MU directly below
#
#SBATCH -t 7-00:00:00
#SBATCH --qos=low
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --array=0-299
#SBATCH --begin=now
#SBATCH --job-name=GCMC_Hybrid
#SBATCH --output=Logs/GCMCHybrid_%A_%a.out
#SBATCH --error=Logs/GCMCHybrid_%A_%a.err

# ── Chemical potential (find with aggregate_mu_scan.py) ───────────────────────
# Can override at submission time:  MU=-4.5 sbatch submit_gcmc_hybrid.sh
MU=${MU:--4.0}          # default -4.0 if not set in environment

# ── Fixed simulation parameters ───────────────────────────────────────────────
BOX_AREA=1600           # box area A = L^2
KT=1.0                  # thermal energy
E_AA=1.0                # A-A like-patch Morse depth (fixed)
E_BB=1.0                # B-B like-patch Morse depth (fixed)
N_EQUIL=20000           # equilibration sweeps
N_PROD=10000            # production sweeps
SNAPSHOT_INTERVAL=100   # save snapshot + energy every N production sweeps
N_INIT=16               # starting particle count for IC
F_DISP=0.5              # fraction of moves that are MD displacement
MD_STEPS=5000           # Langevin steps per displacement move
MD_DT=0.01              # Langevin time step
GAMMA=1.0               # Langevin friction coefficient

# ── Opening angle grid  [deg → rad conversion happens below] ─────────────────
# Matches config_patchy_particle YIELD_ALPHAS_DEG = np.linspace(70, 115, 10)
# Update --array upper bound when changing: new_upper = N_ANGLES * N_EAB * N_REAL - 1
ANGLES_DEG=(70.0 75.0 80.0 85.0 90.0 95.0 100.0 105.0 110.0 115.0)
N_ANGLES=10

# ── E_AB grid (same as mu-scan) ───────────────────────────────────────────────
EAB_VALUES=(1.0 2.0 5.0 10.0 20.0 50.0)
N_EAB=6

# ── Number of independent realizations per parameter combination ──────────────
N_REAL=5

# ── Index arithmetic ──────────────────────────────────────────────────────────
REAL_IDX=$(( SLURM_ARRAY_TASK_ID % N_REAL ))
PARAM_IDX=$(( SLURM_ARRAY_TASK_ID / N_REAL ))
EAB_IDX=$(( PARAM_IDX % N_EAB ))
ANGLE_IDX=$(( PARAM_IDX / N_EAB ))

ANGLE_DEG=${ANGLES_DEG[$ANGLE_IDX]}
E_AB=${EAB_VALUES[$EAB_IDX]}

# Convert degrees to radians using python (bc lacks trig)
OPENING_ANGLE=$(python3 -c "import math; print(math.radians(${ANGLE_DEG}))")

# Seed: deterministic from (angle_idx, eab_idx, realization) so runs are reproducible
SEED=$(( ANGLE_IDX * 1000 + EAB_IDX * 10 + REAL_IDX ))

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR=$(python3 -c "from config_gcmc import GCMC_RUNS_DIR; print(GCMC_RUNS_DIR)")

# ── Environment ───────────────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR
mkdir -p Logs
mkdir -p ${OUTPUT_DIR}

eval "$(conda shell.bash hook)"
conda activate selfAssemblyIG

echo "=========================================="
echo "Job:    ${SLURM_JOB_ID}  Task: ${SLURM_ARRAY_TASK_ID}"
echo "angle = ${ANGLE_DEG} deg  (idx ${ANGLE_IDX})"
echo "E_AB =  ${E_AB}  (idx ${EAB_IDX})"
echo "real =  ${REAL_IDX}   seed = ${SEED}"
echo "mu =    ${MU}   kT = ${KT}"
echo "Host:   $(hostname)"
echo "Time:   $(date)"
echo "=========================================="

# Output file is auto-named from parameters when --output is not given
python run_gcmc_hybrid.py \
    --opening_angle    ${OPENING_ANGLE}    \
    --kT               ${KT}               \
    --E_AB             ${E_AB}             \
    --E_AA             ${E_AA}             \
    --E_BB             ${E_BB}             \
    --mu               ${MU}               \
    --box_area         ${BOX_AREA}         \
    --n_equil          ${N_EQUIL}          \
    --n_prod           ${N_PROD}           \
    --snapshot_interval ${SNAPSHOT_INTERVAL} \
    --seed             ${SEED}             \
    --N_init           ${N_INIT}           \
    --f_disp           ${F_DISP}           \
    --md_steps         ${MD_STEPS}         \
    --md_dt            ${MD_DT}            \
    --gamma            ${GAMMA}            \
    --realization      ${REAL_IDX}         \
    --verbose

echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
