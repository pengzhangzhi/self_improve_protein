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

The five stages have narrow contracts:

- `prepare-data`: verify the three pinned sources; write processed tables and the
  immutable manifest.
- `embed-assay`: combine one manifest-bound table with frozen ESM-2; write its
  `.npy` cache and JSON metadata.
- `run-task`: combine the manifest, processed data, embeddings, and locked grid;
  write one assay-seed task JSON.
- `aggregate`: reconstruct the complete task grid; write tables, diagnostics,
  and `<results-root>/<mode>/aggregate.json`.
- `verify`: fail closed on the supplied manifest and artifacts; the postflight
  invocation below is read-only.

By default, preparation reads
`$SI_DATA_ROOT/raw/DMS_ProteinGym_substitutions.zip`,
`$SI_DATA_ROOT/raw/zero_shot_substitutions_scores.zip`, and
`$SI_DATA_ROOT/raw/DMS_substitutions.csv`. The batch script also accepts the
explicit `SI_DMS_ZIP`, `SI_SCORES_ZIP`, and `SI_METADATA_CSV` overrides.
Preparation does not fetch those archives. If compute nodes cannot reach the
model registry, pre-cache the exact configured ESM-2 model and revision in a
cache visible to them before submission.

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
export SLURM_CONF="$SI_SLURM_CONF"
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

Slurm client commands consume `SLURM_CONF`, not `SI_SLURM_CONF`. Exporting both
in the parent shell keeps submission, `squeue`, `sacct`, and `scancel` on the
intended controller. Repeat the site export block after opening a new shell.

## Submit the pipeline

All stages are chained with `afterok`: a downstream job starts only after its
dependency succeeds.

`slurm/submit_pipeline.sh` is not a generic experiment launcher. It enforces the
canonical v0 protocol digest and invokes the fixed v0 task implementation in its
v0 artifact namespace.

### Immutable-run preflight

Use a dedicated worktree. Immediately before either submission block, run this
preflight in the same shell that you will keep through postflight:

```bash
RUN_HEAD=''
RUN_STATUS=''
if ! RUN_HEAD="$(git rev-parse --verify 'HEAD^{commit}')" ||
   [[ ! "$RUN_HEAD" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'cannot resolve a full 40-hex HEAD commit\n' >&2
  false
elif ! RUN_STATUS="$(git status --porcelain=v1 --untracked-files=all)"; then
  printf 'cannot inspect worktree status\n' >&2
  false
elif [[ -n "$RUN_STATUS" ]]; then
  printf 'submission requires a clean dedicated worktree:\n%s\n' \
    "$RUN_STATUS" >&2
  false
else
  unset RUN_STATUS
  printf 'immutable run HEAD: %s\n' "$RUN_HEAD"
fi
```

Proceed only if the command prints the immutable head and exits 0. Do not edit
the checkout or resync, replace, or otherwise change `.venv` until postflight
verification finishes. The final comparison below assumes this same shell. If
you reconnect, record the printed SHA externally and restore that exact value to
`RUN_HEAD` before postflight.

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

### Reconnect to an in-flight run

Before leaving the original shell, print and save this non-secret context in
approved operator notes, not in the repository:

```bash
printf 'cd -- %q\nexport SI_RUN_ID=%q\nexport SI_MODE=%q\nRUN_HEAD=%q\n' \
  "$(pwd -P)" "$SI_RUN_ID" "$SI_MODE" "$RUN_HEAD"
```

In a new login shell:

1. Run the recorded `cd` line to return to the exact dedicated worktree.
2. Rerun the seven site-configuration exports and the derived-path export block
   above. This restores `SI_REPO_ROOT`, `SLURM_CONF`, and every artifact root.
3. Run the recorded `SI_RUN_ID`, `SI_MODE`, and `RUN_HEAD` assignments exactly;
   do not generate a new run ID.
4. If `SI_MODE=confirmatory`, re-enter the same authorized `SI_R5_GATE` value and
   export it. The gate is intentionally absent from the recorded context.

Validate the restored non-secret context before monitoring or postflight:

```bash
test -n "$SI_RUN_ID" &&
[[ "$SI_MODE" == development || "$SI_MODE" == confirmatory ]] &&
[[ "$RUN_HEAD" =~ ^[0-9a-f]{40}$ ]]
```

## Monitor, cancel, and verify

The launcher incrementally writes `local/slurm/<run-id>/job_ids.json`; a
successful submission also prints that path. Inspect the recorded
`jobs.prepare`, `jobs.embed`, `jobs.task`, and `jobs.aggregate` values:

```bash
python -m json.tool "local/slurm/${SI_RUN_ID}/job_ids.json"
```

The launcher is non-atomic: it submits one stage at a time. If it exits nonzero
after creating the manifest, immediately cancel every non-null recorded job ID
before diagnosing or retrying. This recovery runs in a subshell, validates IDs,
skips `null`, and lets `scancel` inherit the parent shell's `SLURM_CONF`.

**CANCELS CLUSTER JOBS**

```bash
(
  set -euo pipefail
  job_manifest="local/slurm/${SI_RUN_ID}/job_ids.json"
  test -f "$job_manifest" || {
    printf 'missing job manifest: %s\n' "$job_manifest" >&2
    exit 1
  }
  .venv/bin/python - "$job_manifest" <<'PY'
import json
import subprocess
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
jobs = payload.get("jobs")
if not isinstance(jobs, dict):
    raise SystemExit("job manifest has no jobs object")
job_ids = []
for name in ("prepare", "embed", "task", "aggregate"):
    value = jobs.get(name)
    if value is None:
        continue
    if not isinstance(value, str) or not value.isdecimal():
        raise SystemExit(f"invalid {name} job ID")
    job_ids.append(value)
if job_ids:
    subprocess.run(["scancel", *job_ids], check=True)
PY
)
```

For a fully submitted chain, all four values are non-null. Substitute them for
`PREPARE_ID`, `EMBED_ID`, `TASK_ID`, and `AGGREGATE_ID`. Both commands below are
read-only scheduler queries:

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

The concise cancellation command below applies only when all four IDs exist. It
includes downstream pending jobs as well as any running job; use the manifest
recovery above for a partial submission.

**CANCELS CLUSTER JOBS**

```bash
scancel PREPARE_ID EMBED_ID TASK_ID AGGREGATE_ID
```

After either cancellation path, rerun `squeue` with only the recorded non-null
IDs, or query them with `sacct`, until no job or array element remains active.

Scheduler success means every job and every array element is `COMPLETED` with
`ExitCode` `0:0`. It also requires the expected aggregate to exist and pass exact
CLI reconstruction. For a development run, verify
`${SI_RESULTS_ROOT}/development/aggregate.json` with:

```bash
OPENBLAS_CORETYPE=Haswell .venv/bin/self-improve-protein \
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
OPENBLAS_CORETYPE=Haswell .venv/bin/self-improve-protein \
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

Finally, check that the checkout still matches the head recorded in the same
shell. After reconnecting, first restore `RUN_HEAD` from the SHA printed by the
preflight.

```bash
POST_HEAD="$(git rev-parse --verify 'HEAD^{commit}')" &&
[[ "$POST_HEAD" =~ ^[0-9a-f]{40}$ ]] &&
test -n "${RUN_HEAD:-}" &&
test "$POST_HEAD" = "$RUN_HEAD" &&
printf 'postflight HEAD unchanged: %s\n' "$RUN_HEAD"
```

## Extend the study safely

Follow the [R0--R7 feedback ladder](research/feedback-ladder.md) rather than
tuning directly on completed outcomes. A new method cannot simply use the v0
launcher: it needs a separately reviewed task launcher and a separate artifact
namespace. The committed [cross-fit](../slurm/submit_crossfit_screen.sh),
[locality](../slurm/submit_locality_screen.sh), and
[exact-CV](../slurm/submit_exact_cv_screen.sh) screen launchers are specialized
examples for their own cards, not generic launchers for arbitrary methods.

1. Write a new experiment card and state the one factor changed from its
   baseline. Use the existing [v0](research/experiment-card-v0.md),
   [cross-fit](research/experiment-card-crossfit.md),
   [locality](research/experiment-card-locality.md),
   [no-Hessian](research/experiment-card-no-hessian.md), or
   [exact-CV](research/experiment-card-exact-cv.md) cards as concrete templates.
2. Add focused tests for the new mechanism, then run the complete local R1--R3
   contract.
3. Review the method-specific launcher and its separate artifact namespace.
4. Arrange an explicit reduced real-data R4 smoke with a maintainer. The public
   repository defines the R4 evidence bar but has no dedicated one-command R4
   launcher; do not relabel the larger development pipeline as R4.
5. Run the reviewed R5-style development pilot and inspect its metrics, diagnostics,
   reproducibility, and launcher evidence.
6. Lock the reviewed artifacts. Gate issuance is study-specific and
   maintainer-controlled; there is no generic public issuance procedure.
7. Run confirmation only when its gate and execution are explicitly authorized.
8. Record the result in a separate decision memo without rewriting the original
   card or its success rule.

Operational success proves that the declared code and artifact path executed; it
does not prove that the selector is scientifically useful. The completed v0 result
is negative, and the 26 untouched outcomes remain sealed.
