#!/usr/bin/env bash
#SBATCH --job-name=sip-task
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"benchmarking","description":"CPU ridge assay-seed experiment task."}}'

set -euo pipefail

export SLURM_CONF="${SI_SLURM_CONF:?set SI_SLURM_CONF}"
export OPENBLAS_CORETYPE=Haswell
: "${SI_REPO_ROOT:?set SI_REPO_ROOT}"
: "${SI_DATA_ROOT:?set SI_DATA_ROOT}"
: "${SI_ARTIFACT_ROOT:?set SI_ARTIFACT_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?Slurm array index is required}"
config="${SI_CONFIG:-${SI_REPO_ROOT}/configs/v0.yaml}"
manifest="${SI_MANIFEST:-${SI_ARTIFACT_ROOT}/studies/v0/data_manifest.json}"
processed_root="${SI_PROCESSED_ROOT:-${SI_DATA_ROOT}/processed/v0}"
embedding_root="${SI_EMBEDDING_ROOT:-${SI_DATA_ROOT}/embeddings/v0}"
results_root="${SI_RESULTS_ROOT:-${SI_ARTIFACT_ROOT}/studies/v0}"
mode="${SI_MODE:-development}"

cd "$SI_REPO_ROOT"
.venv/bin/python -c 'from self_improve_protein.experiment import require_openblas_coretype; require_openblas_coretype("Haswell")'
exec .venv/bin/self-improve-protein --config "$config" run-task \
    --manifest "$manifest" \
    --processed-root "$processed_root" \
    --embedding-root "$embedding_root" \
    --results-root "$results_root" \
    --mode "$mode" \
    --task-index "$SLURM_ARRAY_TASK_ID"
