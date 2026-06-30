#!/usr/bin/env bash
#SBATCH --job-name=sip-crossfit-embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-25
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"benchmarking","description":"Pinned ESM-2 caches for untouched crossfit pools."}}'

set -euo pipefail

export SLURM_CONF="${SI_SLURM_CONF:?set SI_SLURM_CONF}"
export OPENBLAS_CORETYPE=Haswell
export TOKENIZERS_PARALLELISM=false
: "${SI_REPO_ROOT:?set SI_REPO_ROOT}"
: "${SI_DATA_ROOT:?set SI_DATA_ROOT}"
: "${SI_ARTIFACT_ROOT:?set SI_ARTIFACT_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?Slurm array index is required}"
config="${SI_CONFIG:-${SI_REPO_ROOT}/configs/v0.yaml}"
base_manifest="${SI_BASE_MANIFEST:-${SI_ARTIFACT_ROOT}/studies/v0/data_manifest.json}"
pool_manifest="${SI_CROSSFIT_POOL_MANIFEST:-${SI_ARTIFACT_ROOT}/exploratory/crossfit_v1/pool_manifest.json}"
processed_root="${SI_PROCESSED_ROOT:-${SI_DATA_ROOT}/processed/crossfit_v1}"
embedding_root="${SI_EMBEDDING_ROOT:-${SI_DATA_ROOT}/embeddings/crossfit_v1}"
batch_size="${SI_EMBED_BATCH_SIZE:-128}"

cd "$SI_REPO_ROOT"
.venv/bin/python -c 'from self_improve_protein.experiment import require_openblas_coretype; require_openblas_coretype("Haswell")'
exec .venv/bin/python -m self_improve_protein.crossfit_pool_cli \
    --config "$config" embed-assay \
    --base-manifest "$base_manifest" \
    --pool-manifest "$pool_manifest" \
    --processed-root "$processed_root" \
    --embedding-root "$embedding_root" \
    --assay-index "$SLURM_ARRAY_TASK_ID" \
    --batch-size "$batch_size" \
    --device cuda
