#!/bin/bash
#SBATCH --job-name=stage4_mix3
#SBATCH --output=logs/stage4_%A_%a.out
#SBATCH --error=logs/stage4_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu2023
#SBATCH --array=0-149

set -euo pipefail

cd ~/projects/Battery_Analysis
mkdir -p logs outputs

source ~/venvs/jaxsys/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# -------------------------
# 3-way split: REAL, SYNTH locked, SYNTH free
# -------------------------
N_PER=50                     # trials per group
TASK=${SLURM_ARRAY_TASK_ID}

GROUP=$(( TASK / N_PER ))    # 0,1,2
TRIAL=$(( TASK % N_PER ))    # 0..N_PER-1

JOB=${SLURM_ARRAY_JOB_ID}
OUTBASE="outputs/stage4mix_${JOB}"

mkdir -p "${OUTBASE}/real" "${OUTBASE}/synth_locked" "${OUTBASE}/synth_free"

if [ "${GROUP}" -eq 0 ]; then
  # -------------------------
  # REAL
  # -------------------------
  OUTDIR="${OUTBASE}/real/task_${TASK}_trial_${TRIAL}"
  python -u run_stage4.py \
    --resample \
    --tmax -1 \
    --trial_start "${TRIAL}" \
    --trial_count 1 \
    --x0_trials 16 \
    --outdir "${OUTDIR}" \
    --seed 0 \
    --warm_adam 300 \
    --warm_lbfgs 40 \
    --stage4_adam 300 \
    --stage4_lbfgs 80 \
    --checkpoint_every 1 \
    --v_col "Ewe/V" \
    --i_col "control/mA" \
    --force_units "A" \
    --v_ref "first"

elif [ "${GROUP}" -eq 1 ]; then
  # -------------------------
  # SYNTH locked  (IMPORTANT: do NOT use --resample if you want locked)
  # -------------------------
  OUTDIR="${OUTBASE}/synth_locked/task_${TASK}_trial_${TRIAL}"
  python -u run_stage4generateddata.py \
    --mode locked \
    --trial_start "${TRIAL}" \
    --trial_count 1 \
    --x0_trials 1 \
    --outdir "${OUTDIR}" \
    --seed 1000 \
    --I_const 2.0 \
    --t_end 25.0 \
    --dt 0.1 \
    --theta_n0 0.8 \
    --theta_p0 0.4 \
    --ce0 0.0 \
    --N_series 3 \
    --warm_adam 300 \
    --warm_lbfgs 40 \
    --stage4_adam 300 \
    --stage4_lbfgs 80 \
    --checkpoint_every 1

else
  # -------------------------
  # SYNTH free (resample OK, but not required)
  # -------------------------
  OUTDIR="${OUTBASE}/synth_free/task_${TASK}_trial_${TRIAL}"
  python -u run_stage4generateddata.py \
    --mode free \
    --trial_start "${TRIAL}" \
    --trial_count 1 \
    --x0_trials 16 \
    --outdir "${OUTDIR}" \
    --seed 2000 \
    --I_const 2.0 \
    --t_end 25.0 \
    --dt 0.1 \
    --theta_n0 0.8 \
    --theta_p0 0.4 \
    --ce0 0.0 \
    --N_series 3 \
    --warm_adam 300 \
    --warm_lbfgs 40 \
    --stage4_adam 300 \
    --stage4_lbfgs 80 \
    --checkpoint_every 1
fi
