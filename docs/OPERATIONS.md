# Operations

This is the maintainer runbook for reproducing and extending the experiment. If
you want the scientific explanation first, read [Getting started](GETTING_STARTED.md).

Commands explicitly described as read-only only inspect the repository or
scheduler. `uv sync` writes the local `.venv`, and the R1--R3 script publishes
local verification artifacts, but neither submits cluster jobs. The submission
and cancellation sections do change scheduler state and are labeled accordingly.
Use them only after reviewing the site's values and storage paths. No public
default contains a real account, partition, or private filesystem path.

## Local setup and CLI discovery

Run these commands from the repository root. The first block creates or updates
the locked CPU environment, then prints the validated protocol and CLI help. The
two CLI commands are read-only; successful `--help` commands print usage and exit
with status 0.

```bash
uv sync --frozen
uv run self-improve-protein --show-config
uv run self-improve-protein --help
```

Install the developer and embedding extras before testing or embedding:

```bash
uv sync --frozen --extra dev --extra embed
uv run self-improve-protein prepare-data --help
uv run self-improve-protein embed-assay --help
uv run self-improve-protein run-task --help
uv run self-improve-protein aggregate --help
uv run self-improve-protein verify --help
```

The first embedding sync installs Torch and Transformers and may be large. It
only changes the local environment; it does not submit jobs. Every successful
stage `--help` command above prints usage and exits with status 0.

## Local R1--R3 verification

Start from a clean Git worktree with the full environment already present in the
local `uv` cache. The script deliberately runs offline and fails before testing
if the worktree is dirty or the locked dependencies cannot be resolved from the
cache.

```bash
bash scripts/verify_r1_r3.sh
```

The script checks fresh-environment resolution, imports, the locked config, CLI
help, lint, types, targeted and full tests, algebra, and synthetic learnability.
It publishes receipts under `artifacts/verification/r1/`, `r2/`, and `r3/`.
Passing establishes code-path correctness and learnability on the controlled
synthetic problem. It does **not** establish protein-task method quality.

## Inputs and stage contract

Official execution requires the pinned ProteinGym substitution archive and
metadata, the pinned zero-shot archive containing the `ESM1v_ensemble` teacher
scores, the ESM-2 checkpoint and exact revision locked in `configs/v0.yaml`, and
that same locked configuration. The processed, embedding, and results roots must
be writable. `prepare-data` creates the immutable manifest consumed by all later
stages; an existing non-identical manifest is rejected rather than overwritten.

| Stage | Required inputs | Output or check |
| --- | --- | --- |
| `prepare-data` | Pinned substitution, zero-shot-score, and metadata files; locked config; writable processed root | Checksum-verified processed assay tables and immutable data manifest |
| `embed-assay` | Manifest, one processed assay, frozen ESM-2 checkpoint, writable embedding root | Revision- and row-bound `.npy` embedding cache plus JSON metadata |
| `run-task` | Manifest, processed data, embeddings, locked mode/task grid, writable results root | One provenance-locked assay-seed task JSON |
| `aggregate` | Manifest, processed data, embeddings, and the complete expected task grid | Reconstructed tables, diagnostics, and `<results-root>/<mode>/aggregate.json` |
| `verify` | Manifest plus any processed, embedding, task, aggregate, or gate artifacts being checked | Fail-closed validation; the postflight invocation below is read-only |

By default, preparation reads
`$SI_DATA_ROOT/raw/DMS_ProteinGym_substitutions.zip`,
`$SI_DATA_ROOT/raw/zero_shot_substitutions_scores.zip`, and
`$SI_DATA_ROOT/raw/DMS_substitutions.csv`. The batch script also accepts the
explicit `SI_DMS_ZIP`, `SI_SCORES_ZIP`, and `SI_METADATA_CSV` overrides.

## Configure the cluster launcher

Replace every public example below with values approved for the target site.
Run from the repository root so that `SI_REPO_ROOT` resolves correctly.

```bash
export SI_ACCOUNT="your-cluster-account"
export SI_CPU_PARTITION="your-cpu-partition"
export SI_GPU_PARTITION="your-gpu-partition"
export SI_REPO_ROOT="$(pwd)"
export SI_DATA_ROOT="/path/to/project-data"
export SI_ARTIFACT_ROOT="/path/to/project-artifacts"
export SI_SLURM_CONF="/path/to/slurm.conf"
```

`slurm/submit_pipeline.sh` requires those seven variables. It derives and exports
the following values unless the operator overrides them:

| Variable | Launcher default |
| --- | --- |
| `SI_CONFIG` | `${SI_REPO_ROOT}/configs/v0.yaml` |
| `SI_MANIFEST` | `${SI_ARTIFACT_ROOT}/studies/v0/data_manifest.json` |
| `SI_PROCESSED_ROOT` | `${SI_DATA_ROOT}/processed/v0` |
| `SI_EMBEDDING_ROOT` | `${SI_DATA_ROOT}/embeddings/v0` |
| `SI_RESULTS_ROOT` | `${SI_ARTIFACT_ROOT}/studies/v0` |
| `SI_MODE` | `development` |

The launcher exports those defaults only inside its process and the submitted
jobs. Materialize the path defaults in the current shell as well so that the
postflight commands later in this guide use the identical locations:

```bash
export SI_CONFIG="${SI_CONFIG:-${SI_REPO_ROOT}/configs/v0.yaml}"
export SI_MANIFEST="${SI_MANIFEST:-${SI_ARTIFACT_ROOT}/studies/v0/data_manifest.json}"
export SI_PROCESSED_ROOT="${SI_PROCESSED_ROOT:-${SI_DATA_ROOT}/processed/v0}"
export SI_EMBEDDING_ROOT="${SI_EMBEDDING_ROOT:-${SI_DATA_ROOT}/embeddings/v0}"
export SI_RESULTS_ROOT="${SI_RESULTS_ROOT:-${SI_ARTIFACT_ROOT}/studies/v0}"
```

## Submit the pipeline

All stages are chained with `afterok`: a downstream job starts only after its
dependency succeeds.

### Development pilot

**SUBMITS CLUSTER JOBS**

```bash
export SI_MODE=development
export SI_RUN_ID="dev-$(date -u +%Y%m%dT%H%M%SZ)"
bash slurm/submit_pipeline.sh
```

This schedules one CPU preparation job, a nine-element one-GPU embedding array,
a two-element CPU task array, and one CPU aggregation job. Each embedding array
element requests one GPU; the scheduler may run elements concurrently. Computing
all nine full assay embeddings can be substantial and is not claimed to be
cheap. This is an R5-style development pilot, not a dedicated reduced R4
launcher.

### Confirmatory execution

Confirmatory execution is allowed only when `SI_R5_GATE` already names a
pre-authorized, validated R5 receipt. The run covers the locked eight-assay by
five-seed grid and therefore creates a 40-element CPU task array.

**SUBMITS CLUSTER JOBS**

```bash
export SI_MODE=confirmatory
export SI_RUN_ID="confirmatory-$(date -u +%Y%m%dT%H%M%SZ)"
: "${SI_R5_GATE:?set SI_R5_GATE to the pre-authorized R5 receipt}"
export SI_R5_GATE
bash slurm/submit_pipeline.sh
```

This is not authorization to access or reveal the 26 designated untouched assay
outcomes. They remain sealed, and this guide intentionally provides no substitute
or fabricated gate value.

## Monitor, cancel, and verify

Submission prints the path to and incrementally writes
`local/slurm/<run-id>/job_ids.json`. Inspect the recorded `jobs.prepare`,
`jobs.embed`, `jobs.task`, and `jobs.aggregate` values:

```bash
python -m json.tool "local/slurm/${SI_RUN_ID}/job_ids.json"
```

Substitute those four numeric values for `PREPARE_ID`, `EMBED_ID`, `TASK_ID`,
and `AGGREGATE_ID`. Both commands below are read-only scheduler queries:

```bash
squeue --jobs PREPARE_ID,EMBED_ID,TASK_ID,AGGREGATE_ID
sacct -j PREPARE_ID,EMBED_ID,TASK_ID,AGGREGATE_ID \
  --format=JobID,JobName%40,State,ExitCode,Elapsed
```

The launcher submits from `SI_REPO_ROOT`, so Slurm writes these files there (the
job-ID JSON remains under `local/slurm/<run-id>/`):

```text
slurm-sip-prepare-<job-id>.out/.err
slurm-sip-embed-<array-job-id>_<index>.out/.err
slurm-sip-task-<array-job-id>_<index>.out/.err
slurm-sip-aggregate-<job-id>.out/.err
```

To stop a submitted chain, include the downstream pending IDs as well as any
running ID.

**CANCELS CLUSTER JOBS**

```bash
scancel PREPARE_ID EMBED_ID TASK_ID AGGREGATE_ID
```

Scheduler success means every job and every array element is `COMPLETED` with
`ExitCode` `0:0`. It also requires the expected aggregate to exist and pass exact
CLI reconstruction. For a development run, verify
`${SI_RESULTS_ROOT}/development/aggregate.json` with:

```bash
OPENBLAS_CORETYPE=Haswell uv run self-improve-protein \
  --config "$SI_CONFIG" verify \
  --manifest "$SI_MANIFEST" \
  --processed-root "$SI_PROCESSED_ROOT" \
  --embedding-root "$SI_EMBEDDING_ROOT" \
  --results-root "$SI_RESULTS_ROOT" \
  --aggregate-artifact "${SI_RESULTS_ROOT}/development/aggregate.json"
```

For an authorized confirmatory run, verification must consume the same gate and
the confirmatory aggregate:

```bash
OPENBLAS_CORETYPE=Haswell uv run self-improve-protein \
  --config "$SI_CONFIG" verify \
  --manifest "$SI_MANIFEST" \
  --processed-root "$SI_PROCESSED_ROOT" \
  --embedding-root "$SI_EMBEDDING_ROOT" \
  --results-root "$SI_RESULTS_ROOT" \
  --r5-gate "$SI_R5_GATE" \
  --aggregate-artifact "${SI_RESULTS_ROOT}/confirmatory/aggregate.json"
```

The terminal CLI event must report `"status":"complete"` and include
`"aggregate"` in its verified list. Scheduler completion without this artifact
check is not a successful experiment run.

## Extend the study safely

Follow the [R0--R7 feedback ladder](research/feedback-ladder.md) rather than
tuning directly on completed outcomes:

1. Write a new experiment card and state the one factor changed from its
   baseline. Use the existing [v0](research/experiment-card-v0.md),
   [cross-fit](research/experiment-card-crossfit.md),
   [locality](research/experiment-card-locality.md),
   [no-Hessian](research/experiment-card-no-hessian.md), or
   [exact-CV](research/experiment-card-exact-cv.md) cards as concrete templates.
2. Add focused tests for the new mechanism, then run the complete local R1--R3
   contract.
3. Arrange an explicit reduced real-data R4 smoke with a maintainer. The public
   repository defines the R4 evidence bar but has no dedicated one-command R4
   launcher; do not relabel the larger development pipeline as R4.
4. Run the R5-style development pilot and review its finite metrics, diagnostics,
   reproducibility, and launcher evidence.
5. Lock the reviewed artifacts and arrange the promotion gate. Run confirmation
   only when that gate and the execution are explicitly authorized.
6. Record the result in a separate decision memo without rewriting the original
   card or its success rule.

Operational success proves that the declared code and artifact path executed; it
does not prove that the selector is scientifically useful. The completed v0 result
is negative, and the 26 untouched outcomes remain sealed.
