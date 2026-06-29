#!/usr/bin/env bash
#SBATCH --job-name=sip-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"benchmarking","description":"Exact-grid clustered experiment aggregation."}}'

set -euo pipefail

export SLURM_CONF="${SI_SLURM_CONF:?set SI_SLURM_CONF}"
export OPENBLAS_CORETYPE=Haswell
: "${SI_REPO_ROOT:?set SI_REPO_ROOT}"
: "${SI_DATA_ROOT:?set SI_DATA_ROOT}"
: "${SI_ARTIFACT_ROOT:?set SI_ARTIFACT_ROOT}"
config="${SI_CONFIG:-${SI_REPO_ROOT}/configs/v0.yaml}"
manifest="${SI_MANIFEST:-${SI_ARTIFACT_ROOT}/studies/v0/data_manifest.json}"
results_root="${SI_RESULTS_ROOT:-${SI_ARTIFACT_ROOT}/studies/v0}"
mode="${SI_MODE:-confirmatory}"
output="${SI_AGGREGATE_OUTPUT:-${results_root}/${mode}/aggregate.json}"

cd "$SI_REPO_ROOT"
.venv/bin/python -c 'from self_improve_protein.experiment import require_openblas_coretype; require_openblas_coretype("Haswell")'
exec .venv/bin/self-improve-protein --config "$config" aggregate \
    --manifest "$manifest" \
    --results-root "$results_root" \
    --output "$output" \
    --mode "$mode"
