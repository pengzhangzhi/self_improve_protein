#!/usr/bin/env bash
#SBATCH --job-name=sip-crossfit-rep-task
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

export SLURM_CONF="${SI_SLURM_CONF:?set SI_SLURM_CONF}"
export OPENBLAS_CORETYPE=Haswell
: "${SI_REPO_ROOT:?set SI_REPO_ROOT}"
: "${SI_CONFIG:?set SI_CONFIG}"
: "${SI_BASE_MANIFEST:?set SI_BASE_MANIFEST}"
: "${SI_CROSSFIT_POOL_MANIFEST:?set SI_CROSSFIT_POOL_MANIFEST}"
: "${SI_CROSSFIT_PROMOTION_GATE:?set SI_CROSSFIT_PROMOTION_GATE}"
: "${SI_EXPECTED_CROSSFIT_PROMOTION_GATE_SHA256:?set expected gate SHA}"
: "${SI_PROCESSED_ROOT:?set SI_PROCESSED_ROOT}"
: "${SI_EMBEDDING_ROOT:?set SI_EMBEDDING_ROOT}"
: "${SI_CROSSFIT_REPLICATION_RESULTS_ROOT:?set results root}"
: "${SLURM_ARRAY_TASK_ID:?Slurm array index is required}"

cd "$SI_REPO_ROOT"
.venv/bin/python -c 'from self_improve_protein.experiment import require_openblas_coretype; require_openblas_coretype("Haswell")'
exec .venv/bin/python -m self_improve_protein.crossfit_replication_cli \
    --config "$SI_CONFIG" run-task \
    --base-manifest "$SI_BASE_MANIFEST" \
    --pool-manifest "$SI_CROSSFIT_POOL_MANIFEST" \
    --promotion-gate "$SI_CROSSFIT_PROMOTION_GATE" \
    --expected-promotion-gate-sha256 \
    "$SI_EXPECTED_CROSSFIT_PROMOTION_GATE_SHA256" \
    --processed-root "$SI_PROCESSED_ROOT" \
    --embedding-root "$SI_EMBEDDING_ROOT" \
    --results-root "$SI_CROSSFIT_REPLICATION_RESULTS_ROOT" \
    --task-index "$SLURM_ARRAY_TASK_ID"
