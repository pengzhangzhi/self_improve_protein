#!/usr/bin/env bash
#SBATCH --job-name=sip-locality-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"benchmarking","description":"Exact pseudo-perturbation locality screen aggregation."}}'

set -euo pipefail

export SLURM_CONF="${SI_SLURM_CONF:?set SI_SLURM_CONF}"
export OPENBLAS_CORETYPE=Haswell
: "${SI_REPO_ROOT:?set SI_REPO_ROOT}"
: "${SI_CONFIG:?set SI_CONFIG}"
: "${SI_BASE_MANIFEST:?set SI_BASE_MANIFEST}"
: "${SI_CROSSFIT_POOL_MANIFEST:?set SI_CROSSFIT_POOL_MANIFEST}"
: "${SI_PROCESSED_ROOT:?set SI_PROCESSED_ROOT}"
: "${SI_EMBEDDING_ROOT:?set SI_EMBEDDING_ROOT}"
: "${SI_LOCALITY_RESULTS_ROOT:?set SI_LOCALITY_RESULTS_ROOT}"
: "${SI_LOCALITY_AGGREGATE:?set SI_LOCALITY_AGGREGATE}"

cd "$SI_REPO_ROOT"
.venv/bin/python -c 'from self_improve_protein.experiment import require_openblas_coretype; require_openblas_coretype("Haswell")'
exec .venv/bin/python -m self_improve_protein.locality_cli \
    --config "$SI_CONFIG" aggregate \
    --base-manifest "$SI_BASE_MANIFEST" \
    --pool-manifest "$SI_CROSSFIT_POOL_MANIFEST" \
    --processed-root "$SI_PROCESSED_ROOT" \
    --embedding-root "$SI_EMBEDDING_ROOT" \
    --results-root "$SI_LOCALITY_RESULTS_ROOT" \
    --output "$SI_LOCALITY_AGGREGATE"
