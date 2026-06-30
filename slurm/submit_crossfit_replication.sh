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
export SI_CROSSFIT_PROMOTION_GATE="${SI_CROSSFIT_PROMOTION_GATE:-${SI_ARTIFACT_ROOT}/exploratory/crossfit_v1/promotion_gate.json}"
export SI_CROSSFIT_SCREEN_PROCESSED_ROOT="${SI_CROSSFIT_SCREEN_PROCESSED_ROOT:-${SI_DATA_ROOT}/processed/v0}"
export SI_CROSSFIT_SCREEN_EMBEDDING_ROOT="${SI_CROSSFIT_SCREEN_EMBEDDING_ROOT:-${SI_DATA_ROOT}/embeddings/v0}"
export SI_CROSSFIT_SCREEN_RESULTS_ROOT="${SI_CROSSFIT_SCREEN_RESULTS_ROOT:-${SI_ARTIFACT_ROOT}/exploratory/crossfit_v1/screen}"
export SI_CROSSFIT_SCREEN_AGGREGATE="${SI_CROSSFIT_SCREEN_AGGREGATE:-${SI_CROSSFIT_SCREEN_RESULTS_ROOT}/aggregate.json}"
export SI_PROCESSED_ROOT="${SI_PROCESSED_ROOT:-${SI_DATA_ROOT}/processed/crossfit_v1}"
export SI_EMBEDDING_ROOT="${SI_EMBEDDING_ROOT:-${SI_DATA_ROOT}/embeddings/crossfit_v1}"
export SI_CROSSFIT_REPLICATION_RESULTS_ROOT="${SI_CROSSFIT_REPLICATION_RESULTS_ROOT:-${SI_ARTIFACT_ROOT}/exploratory/crossfit_v1/replication}"
export SI_CROSSFIT_REPLICATION_AGGREGATE="${SI_CROSSFIT_REPLICATION_AGGREGATE:-${SI_CROSSFIT_REPLICATION_RESULTS_ROOT}/aggregate.json}"

cd "$SI_REPO_ROOT"
.venv/bin/python -m self_improve_protein.crossfit_replication_cli \
    --config "$SI_CONFIG" create-gate \
    --base-manifest "$SI_BASE_MANIFEST" \
    --pool-manifest "$SI_CROSSFIT_POOL_MANIFEST" \
    --processed-root "$SI_CROSSFIT_SCREEN_PROCESSED_ROOT" \
    --embedding-root "$SI_CROSSFIT_SCREEN_EMBEDDING_ROOT" \
    --screen-results-root "$SI_CROSSFIT_SCREEN_RESULTS_ROOT" \
    --screen-aggregate "$SI_CROSSFIT_SCREEN_AGGREGATE" \
    --output "$SI_CROSSFIT_PROMOTION_GATE"

SI_EXPECTED_CROSSFIT_PROMOTION_GATE_SHA256="$(
    sha256sum "$SI_CROSSFIT_PROMOTION_GATE" | awk '{print $1}'
)"
if [[ ! "$SI_EXPECTED_CROSSFIT_PROMOTION_GATE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "promotion gate SHA capture failed" >&2
    exit 2
fi
export SI_EXPECTED_CROSSFIT_PROMOTION_GATE_SHA256

.venv/bin/python -m self_improve_protein.crossfit_replication_cli \
    --config "$SI_CONFIG" verify \
    --base-manifest "$SI_BASE_MANIFEST" \
    --pool-manifest "$SI_CROSSFIT_POOL_MANIFEST" \
    --promotion-gate "$SI_CROSSFIT_PROMOTION_GATE" \
    --expected-promotion-gate-sha256 \
    "$SI_EXPECTED_CROSSFIT_PROMOTION_GATE_SHA256" \
    --processed-root "$SI_PROCESSED_ROOT" \
    --embedding-root "$SI_EMBEDDING_ROOT"

run_id="${SI_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="${SI_REPO_ROOT}/local/slurm/${run_id}"
mkdir -p "${SI_REPO_ROOT}/local/slurm"
mkdir "$run_dir"
job_manifest="$run_dir/job_ids.json"
temporary="$run_dir/.job_ids.json.tmp"
task_job=""
aggregate_job=""

json_job_id() {
    if [[ -z "$1" ]]; then printf 'null'; else printf '"%s"' "$1"; fi
}

write_manifest() {
    {
        printf '{\n  "jobs": {\n    "aggregate": '
        json_job_id "$aggregate_job"
        printf ',\n    "task": '
        json_job_id "$task_job"
        printf '\n  },\n  "kind": "crossfit_replication_jobs",\n'
        printf '  "schema_version": 1\n}\n'
    } >"$temporary"
    mv "$temporary" "$job_manifest"
}

submit_job() {
    local raw
    raw="$(sbatch --parsable "$@")"
    if [[ ! "$raw" =~ ^[0-9]+(\;[A-Za-z0-9._-]+)?$ ]]; then
        echo "sbatch returned an invalid parsable job ID" >&2
        return 2
    fi
    printf '%s' "${raw%%;*}"
}

task_job="$(submit_job \
    --account="$SI_ACCOUNT" \
    --partition="$SI_CPU_PARTITION" \
    --array=0-129 \
    slurm/crossfit_replication_task_array.sbatch)"
write_manifest
aggregate_job="$(submit_job \
    --account="$SI_ACCOUNT" \
    --partition="$SI_CPU_PARTITION" \
    --dependency=afterok:"$task_job" \
    slurm/crossfit_replication_aggregate.sbatch)"
write_manifest
printf '%s\n' "$job_manifest"
