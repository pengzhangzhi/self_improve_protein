#!/usr/bin/env bash
#SBATCH --job-name=sip-exact-cv-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"benchmarking","description":"Receipt-only exact-CV screen aggregation."}}'

set -euo pipefail

export SLURM_CONF="${SI_SLURM_CONF:?set SI_SLURM_CONF}"
export OPENBLAS_CORETYPE=Haswell
: "${SI_REPO_ROOT:?set SI_REPO_ROOT}"
: "${SI_CONFIG:?set SI_CONFIG}"
: "${SI_BASE_MANIFEST:?set SI_BASE_MANIFEST}"
: "${SI_CROSSFIT_POOL_MANIFEST:?set SI_CROSSFIT_POOL_MANIFEST}"
: "${SI_EXACT_CV_RESULTS_ROOT:?set SI_EXACT_CV_RESULTS_ROOT}"
: "${SI_EXACT_CV_AGGREGATE:?set SI_EXACT_CV_AGGREGATE}"

cd "$SI_REPO_ROOT"
.venv/bin/python -c 'from self_improve_protein.experiment import require_openblas_coretype; require_openblas_coretype("Haswell")'
exec .venv/bin/python -m self_improve_protein.exact_cv_cli \
    --config "$SI_CONFIG" aggregate \
    --base-manifest "$SI_BASE_MANIFEST" \
    --pool-manifest "$SI_CROSSFIT_POOL_MANIFEST" \
    --results-root "$SI_EXACT_CV_RESULTS_ROOT" \
    --output "$SI_EXACT_CV_AGGREGATE"
