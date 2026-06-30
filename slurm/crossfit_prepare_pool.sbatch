#!/usr/bin/env bash
#SBATCH --job-name=sip-crossfit-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"benchmarking","description":"Outcome-blind crossfit pool preparation."}}'

set -euo pipefail

export SLURM_CONF="${SI_SLURM_CONF:?set SI_SLURM_CONF}"
export OPENBLAS_CORETYPE=Haswell
: "${SI_REPO_ROOT:?set SI_REPO_ROOT}"
: "${SI_DATA_ROOT:?set SI_DATA_ROOT}"
: "${SI_ARTIFACT_ROOT:?set SI_ARTIFACT_ROOT}"
config="${SI_CONFIG:-${SI_REPO_ROOT}/configs/v0.yaml}"
base_manifest="${SI_BASE_MANIFEST:-${SI_ARTIFACT_ROOT}/studies/v0/data_manifest.json}"
pool_manifest="${SI_CROSSFIT_POOL_MANIFEST:-${SI_ARTIFACT_ROOT}/exploratory/crossfit_v1/pool_manifest.json}"
processed_root="${SI_PROCESSED_ROOT:-${SI_DATA_ROOT}/processed/crossfit_v1}"
dms_zip="${SI_DMS_ZIP:-${SI_DATA_ROOT}/raw/DMS_ProteinGym_substitutions.zip}"
scores_zip="${SI_SCORES_ZIP:-${SI_DATA_ROOT}/raw/zero_shot_substitutions_scores.zip}"
metadata_csv="${SI_METADATA_CSV:-${SI_DATA_ROOT}/raw/DMS_substitutions.csv}"

cd "$SI_REPO_ROOT"
.venv/bin/python -c 'from self_improve_protein.experiment import require_openblas_coretype; require_openblas_coretype("Haswell")'
exec .venv/bin/python -m self_improve_protein.crossfit_pool_cli \
    --config "$config" prepare-pool \
    --base-manifest "$base_manifest" \
    --pool-manifest "$pool_manifest" \
    --dms-zip "$dms_zip" \
    --scores-zip "$scores_zip" \
    --metadata-csv "$metadata_csv" \
    --processed-root "$processed_root"
