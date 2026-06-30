#!/usr/bin/env bash
set -euo pipefail

: "${SI_ACCOUNT:?set SI_ACCOUNT}"
: "${SI_CPU_PARTITION:?set SI_CPU_PARTITION}"
: "${SI_REPO_ROOT:?set SI_REPO_ROOT}"
: "${SI_DATA_ROOT:?set SI_DATA_ROOT}"
: "${SI_ARTIFACT_ROOT:?set SI_ARTIFACT_ROOT}"
: "${SI_SLURM_CONF:?set SI_SLURM_CONF}"

export SI_ACCOUNT SI_CPU_PARTITION SI_REPO_ROOT SI_DATA_ROOT SI_ARTIFACT_ROOT
export SI_SLURM_CONF
export SLURM_CONF="$SI_SLURM_CONF"
export OPENBLAS_CORETYPE=Haswell
export SI_CONFIG="${SI_CONFIG:-${SI_REPO_ROOT}/configs/v0.yaml}"
export SI_BASE_MANIFEST="${SI_BASE_MANIFEST:-${SI_ARTIFACT_ROOT}/studies/v0/data_manifest.json}"
export SI_CROSSFIT_POOL_MANIFEST="${SI_CROSSFIT_POOL_MANIFEST:-${SI_ARTIFACT_ROOT}/exploratory/crossfit_v1/pool_manifest.json}"
# This exploratory screen is restricted to already exposed GFP + v0 data.
# No untouched crossfit processed or embedding root is reachable here.
export SI_PROCESSED_ROOT="${SI_PROCESSED_ROOT:-${SI_DATA_ROOT}/processed/v0}"
export SI_EMBEDDING_ROOT="${SI_EMBEDDING_ROOT:-${SI_DATA_ROOT}/embeddings/v0}"
export SI_EXACT_CV_RESULTS_ROOT="${SI_EXACT_CV_RESULTS_ROOT:-${SI_ARTIFACT_ROOT}/exploratory/exact_cv_v1/screen}"
export SI_EXACT_CV_AGGREGATE="${SI_EXACT_CV_AGGREGATE:-${SI_EXACT_CV_RESULTS_ROOT}/aggregate.json}"
export SI_EXACT_CV_PROBE="${SI_EXACT_CV_PROBE:-${SI_ARTIFACT_ROOT}/verification/exact_cv_v1/real_shape_probe.json}"

cd "$SI_REPO_ROOT"
.venv/bin/python -m self_improve_protein.exact_cv_cli \
    --config "$SI_CONFIG" verify \
    --base-manifest "$SI_BASE_MANIFEST" \
    --pool-manifest "$SI_CROSSFIT_POOL_MANIFEST" \
    --processed-root "$SI_PROCESSED_ROOT" \
    --embedding-root "$SI_EMBEDDING_ROOT" \
    --probe-artifact "$SI_EXACT_CV_PROBE"

run_id="${SI_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="${SI_REPO_ROOT}/local/slurm"
run_dir="${run_root}/${run_id}"
mkdir -p "$run_root"
mkdir "$run_dir"
job_manifest="$run_dir/job_ids.json"
temporary="$run_dir/.job_ids.json.tmp"
task_job=""
verify_job=""
aggregate_job=""

json_job_id() {
    local value="$1"
    if [[ -z "$value" ]]; then
        printf 'null'
    else
        printf '"%s"' "$value"
    fi
}

write_job_manifest() {
    {
        printf '{\n  "jobs": {\n    "aggregate": '
        json_job_id "$aggregate_job"
        printf ',\n    "task": '
        json_job_id "$task_job"
        printf ',\n    "verify": '
        json_job_id "$verify_job"
        printf '\n  },\n'
        printf '  "kind": "exact_cv_screen_jobs",\n  "schema_version": 1\n}\n'
    } >"$temporary"
    mv "$temporary" "$job_manifest"
}

submit_job() {
    local raw job_id
    raw="$(sbatch --parsable "$@")"
    if [[ ! "$raw" =~ ^[0-9]+(\;[A-Za-z0-9._-]+)?$ ]]; then
        echo "sbatch returned an invalid parsable job ID" >&2
        return 2
    fi
    job_id="${raw%%;*}"
    printf '%s' "$job_id"
}

task_job="$(submit_job \
    --account="$SI_ACCOUNT" \
    --partition="$SI_CPU_PARTITION" \
    --array=0-44 \
    slurm/exact_cv_task_array.sbatch)"
write_job_manifest

verify_job="$(submit_job \
    --account="$SI_ACCOUNT" \
    --partition="$SI_CPU_PARTITION" \
    --array=0-44 \
    --dependency=afterok:"$task_job" \
    slurm/exact_cv_verify_array.sbatch)"
write_job_manifest

aggregate_job="$(submit_job \
    --account="$SI_ACCOUNT" \
    --partition="$SI_CPU_PARTITION" \
    --dependency=afterok:"$verify_job" \
    slurm/exact_cv_aggregate.sbatch)"
write_job_manifest

printf '%s\n' "$job_manifest"
