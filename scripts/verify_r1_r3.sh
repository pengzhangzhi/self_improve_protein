#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

START_HEAD="$(git rev-parse HEAD)"
START_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
if [[ -n "$START_STATUS" ]]; then
    printf 'verification requires a clean git worktree before start:\n%s\n' \
        "$START_STATUS" >&2
    exit 1
fi

ARTIFACT_ROOT="${SELF_IMPROVE_VERIFICATION_ROOT:-$REPO_ROOT/artifacts/verification}"
ARTIFACT_PARENT="$(dirname -- "$ARTIFACT_ROOT")"
ARTIFACT_NAME="$(basename -- "$ARTIFACT_ROOT")"
mkdir -p "$ARTIFACT_PARENT"
ARTIFACT_PARENT="$(cd "$ARTIFACT_PARENT" && pwd)"
ARTIFACT_ROOT="$ARTIFACT_PARENT/$ARTIFACT_NAME"
TEMP_ROOT="$(mktemp -d "$ARTIFACT_PARENT/.${ARTIFACT_NAME}.r1-r3.XXXXXX")"
STATUS_FILE="$TEMP_ROOT/commands.tsv"
STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
trap 'rm -rf -- "$TEMP_ROOT"' EXIT

mkdir -p "$TEMP_ROOT/command-logs" "$TEMP_ROOT/r1" "$TEMP_ROOT/r2" "$TEMP_ROOT/r3"
: > "$STATUS_FILE"

record_command() {
    local name="$1"
    shift
    local log_path="$TEMP_ROOT/command-logs/$name.txt"
    local command_text
    local started_at
    local completed_at
    local exit_code=0
    local output_sha256

    printf -v command_text '%q ' "$@"
    command_text="${command_text% }"
    started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    if "$@" > "$log_path" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi
    completed_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    output_sha256="$(sha256sum "$log_path" | awk '{print $1}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$started_at" "$completed_at" "$exit_code" \
        "$output_sha256" "$command_text" >> "$STATUS_FILE"
    if (( exit_code != 0 )); then
        printf 'verification command failed: %s (exit %s)\n' "$name" "$exit_code" >&2
        tail -n 200 "$log_path" >&2
        return "$exit_code"
    fi
}

UV_BASE=(uv run --frozen --offline --extra dev --extra embed)

record_command r1_fresh_environment_resolution \
    bash -c \
    'set -euo pipefail; env UV_PROJECT_ENVIRONMENT="$1" uv sync --dry-run --locked --offline --extra dev --extra embed --output-format json > "$2"' \
    _ "$TEMP_ROOT/fresh-environment" \
    "$TEMP_ROOT/r1/fresh-environment-resolution.json"

record_command r1_package_import \
    "${UV_BASE[@]}" python -c \
    'import self_improve_protein, torch, transformers; print(self_improve_protein.__version__, torch.__version__, transformers.__version__)'
record_command r1_locked_config \
    "${UV_BASE[@]}" python -c \
    'import json; from self_improve_protein.config import load_protocol; p=load_protocol("configs/v0.yaml"); assert (p.working_size,p.n_labeled,p.n_unlabeled,p.n_test,p.q,p.pseudo_weight,p.ridge_lambda,p.damping)==(6000,96,2000,1000,192,0.1,0.01,0.0001); print(json.dumps(p.model_dump(mode="json"),allow_nan=False,sort_keys=True))'
record_command r1_cli_help \
    "${UV_BASE[@]}" self-improve-protein --help
record_command r1_ruff \
    "${UV_BASE[@]}" ruff check .
record_command r1_mypy \
    "${UV_BASE[@]}" mypy src

record_command r2_targeted_pytest \
    "${UV_BASE[@]}" pytest -q \
    tests/test_ridge.py \
    tests/test_selection.py \
    tests/test_data.py \
    tests/test_embeddings.py \
    tests/test_metrics.py \
    tests/test_experiment.py \
    tests/test_synthetic_probe.py
record_command r2_full_pytest \
    "${UV_BASE[@]}" pytest -q

record_command r3_synthetic_probe \
    "${UV_BASE[@]}" python -m self_improve_protein.probes \
    --output "$TEMP_ROOT/r3/synthetic_probe.json" \
    --algebra-output "$TEMP_ROOT/r2/algebra_probe.json"

{
    printf 'R2 targeted command: '
    awk -F '\t' '$1 == "r2_targeted_pytest" {print $6 "\nexit_code=" $4}' "$STATUS_FILE"
    cat "$TEMP_ROOT/command-logs/r2_targeted_pytest.txt"
    printf '\nR2 full command: '
    awk -F '\t' '$1 == "r2_full_pytest" {print $6 "\nexit_code=" $4}' "$STATUS_FILE"
    cat "$TEMP_ROOT/command-logs/r2_full_pytest.txt"
} > "$TEMP_ROOT/r2/pytest.txt"

END_HEAD="$(git rev-parse HEAD)"
END_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
if [[ "$END_HEAD" != "$START_HEAD" ]]; then
    printf 'verification HEAD changed from %s to %s\n' "$START_HEAD" "$END_HEAD" >&2
    exit 1
fi
if [[ -n "$END_STATUS" ]]; then
    printf 'verification worktree became dirty:\n%s\n' "$END_STATUS" >&2
    exit 1
fi

"${UV_BASE[@]}" python - \
    "$STATUS_FILE" "$TEMP_ROOT" "$STARTED_AT" "$START_HEAD" \
    "$ARTIFACT_ROOT" "$SCRIPT_DIR/verify_r1_r3.sh" <<'PY'
import csv
import dataclasses
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from self_improve_protein.config import load_protocol
from self_improve_protein.experiment import (
    NUMERICAL_POLICY,
    canonical_protocol_digest,
    current_numerical_runtime_fingerprint,
)
from self_improve_protein.provenance import (
    atomic_write_json,
    sha256_bytes,
    sha256_file,
)

status_path = Path(sys.argv[1])
temporary_root = Path(sys.argv[2])
started_at = sys.argv[3]
expected_head = sys.argv[4]
artifact_root = Path(sys.argv[5]).resolve()
verification_script = Path(sys.argv[6]).resolve()
commands = []
with status_path.open(encoding="utf-8", newline="") as handle:
    for row in csv.reader(handle, delimiter="\t"):
        if len(row) != 6:
            raise RuntimeError("command status row does not have six fields")
        name, command_started, completed, exit_code, output_sha256, command = row
        log_path = temporary_root / "command-logs" / f"{name}.txt"
        output = log_path.read_text(encoding="utf-8", errors="replace")
        commands.append(
            {
                "name": name,
                "command": command,
                "started_at": command_started,
                "completed_at": completed,
                "exit_code": int(exit_code),
                "output_sha256": output_sha256,
                "output_bytes": log_path.stat().st_size,
                "output_tail": output[-4000:],
            }
        )

if not commands or any(command["exit_code"] != 0 for command in commands):
    raise RuntimeError("cannot publish R1 report with failed commands")

protocol = load_protocol("configs/v0.yaml")
package_names = (
    "self-improve-protein",
    "numpy",
    "pandas",
    "scipy",
    "pydantic",
    "PyYAML",
    "typer",
    "threadpoolctl",
    "pytest",
    "ruff",
    "mypy",
    "torch",
    "transformers",
)
versions = {name: importlib.metadata.version(name) for name in package_names}
config_path = Path("configs/v0.yaml")
pyproject_path = Path("pyproject.toml")
lock_path = Path("uv.lock")
executable_paths = {
    "python": Path(sys.executable).resolve(),
    "uv": Path(shutil.which("uv") or "").resolve(),
    "pytest": Path(shutil.which("pytest") or "").resolve(),
    "ruff": Path(shutil.which("ruff") or "").resolve(),
    "mypy": Path(shutil.which("mypy") or "").resolve(),
}
if any(not path.is_file() for path in executable_paths.values()):
    raise RuntimeError("verification toolchain executable is missing")
toolchain = {
    name: {"path": str(path), "sha256": sha256_file(path)}
    for name, path in executable_paths.items()
}
toolchain["uv"]["version"] = subprocess.run(
    [str(executable_paths["uv"]), "--version"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
toolchain["python"]["version"] = platform.python_version()
trust_root = {
    "git_head": expected_head,
    "pyproject_sha256": sha256_file(pyproject_path),
    "uv_lock_sha256": sha256_file(lock_path),
    "config_sha256": sha256_file(config_path),
    "verification_script_sha256": sha256_file(verification_script),
    "python_executable_sha256": toolchain["python"]["sha256"],
    "uv_executable_sha256": toolchain["uv"]["sha256"],
    "pytest_executable_sha256": toolchain["pytest"]["sha256"],
    "ruff_executable_sha256": toolchain["ruff"]["sha256"],
    "mypy_executable_sha256": toolchain["mypy"]["sha256"],
}
output_relative_paths = (
    "r1/fresh-environment-resolution.json",
    "r2/pytest.txt",
    "r2/algebra_probe.json",
    "r3/synthetic_probe.json",
)
output_hashes = {
    relative_path: sha256_file(temporary_root / relative_path)
    for relative_path in output_relative_paths
}
report = {
    "schema_version": 2,
    "rung": "R1",
    "status": "passed",
    "started_at": started_at,
    "completed_at": commands[-1]["completed_at"],
    "repository_root": str(Path.cwd()),
    "git_head": expected_head,
    "repository_state": {
        "start_head": expected_head,
        "end_head": expected_head,
        "start_clean": True,
        "end_clean": True,
    },
    "python": {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
    },
    "package_versions": versions,
    "toolchain": toolchain,
    "trust_root": trust_root,
    "config": {
        "path": str(config_path),
        "sha256": sha256_file(config_path),
        "protocol_digest": canonical_protocol_digest(protocol),
        "dump": protocol.model_dump(mode="json"),
    },
    "numerical_runtime": {
        "policy": NUMERICAL_POLICY,
        "fingerprint": dataclasses.asdict(
            current_numerical_runtime_fingerprint()
        ),
    },
    "commands": commands,
    "artifacts": {
        "r1_report": str(artifact_root / "r1" / "report.json"),
        "fresh_environment_resolution": str(
            artifact_root / "r1" / "fresh-environment-resolution.json"
        ),
        "r2_pytest": str(artifact_root / "r2" / "pytest.txt"),
        "r2_algebra": str(artifact_root / "r2" / "algebra_probe.json"),
        "r3_probe": str(artifact_root / "r3" / "synthetic_probe.json"),
        "completion": str(artifact_root / "completion.json"),
    },
    "output_sha256": output_hashes,
}
serialized = json.dumps(
    report,
    allow_nan=False,
    separators=(",", ":"),
    sort_keys=True,
).encode()
report["report_content_sha256"] = sha256_bytes(serialized)
atomic_write_json(temporary_root / "r1" / "report.json", report)
PY

"${UV_BASE[@]}" python - \
    "$TEMP_ROOT" "$ARTIFACT_ROOT" "$START_HEAD" "$REPO_ROOT" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from self_improve_protein.probes import (
    build_verification_completion,
    publish_verification_bundle,
    require_clean_verification_git_state,
    validate_verification_bundle,
)

temporary_root = Path(sys.argv[1]).resolve()
artifact_root = Path(sys.argv[2]).resolve()
expected_head = sys.argv[3]
repository_root = Path(sys.argv[4]).resolve()
require_clean_verification_git_state(
    repository_root,
    expected_head=expected_head,
)
report = json.loads(
    (temporary_root / "r1" / "report.json").read_text(encoding="utf-8")
)
completion = build_verification_completion(
    temporary_root,
    artifact_root=artifact_root,
    git_head=expected_head,
    trust_root=report["trust_root"],
    published_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
)
require_clean_verification_git_state(
    repository_root,
    expected_head=expected_head,
)
publish_verification_bundle(temporary_root, artifact_root, completion)
validate_verification_bundle(
    artifact_root,
    expected_git_head=expected_head,
)
PY

printf 'R1-R3 verification passed; artifacts: %s\n' "$ARTIFACT_ROOT"
