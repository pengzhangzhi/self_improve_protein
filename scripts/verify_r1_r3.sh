#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

ARTIFACT_ROOT="${SELF_IMPROVE_VERIFICATION_ROOT:-$REPO_ROOT/artifacts/verification}"
mkdir -p "$ARTIFACT_ROOT"
TEMP_ROOT="$(mktemp -d "$ARTIFACT_ROOT/.r1-r3.XXXXXX")"
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

UV_BASE=(uv run --frozen --offline --extra dev)

record_command r1_package_import \
    "${UV_BASE[@]}" python -c \
    'import self_improve_protein; print(self_improve_protein.__version__)'
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

"${UV_BASE[@]}" python - "$STATUS_FILE" "$TEMP_ROOT" "$STARTED_AT" <<'PY'
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from self_improve_protein.config import load_protocol
from self_improve_protein.experiment import (
    NUMERICAL_POLICY,
    canonical_protocol_digest,
    current_numerical_runtime_fingerprint,
)
from self_improve_protein.provenance import atomic_write_json, sha256_file

status_path = Path(sys.argv[1])
temporary_root = Path(sys.argv[2])
started_at = sys.argv[3]
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
)
versions = {name: importlib.metadata.version(name) for name in package_names}
config_path = Path("configs/v0.yaml")
git_head = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
git_status = subprocess.run(
    ["git", "status", "--short"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
report = {
    "schema_version": 1,
    "rung": "R1",
    "status": "passed",
    "started_at": started_at,
    "completed_at": commands[-1]["completed_at"],
    "repository_root": str(Path.cwd()),
    "git_head": git_head,
    "git_status_porcelain": git_status,
    "python": {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
    },
    "package_versions": versions,
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
        "r2_pytest": "artifacts/verification/r2/pytest.txt",
        "r2_algebra": "artifacts/verification/r2/algebra_probe.json",
        "r3_probe": "artifacts/verification/r3/synthetic_probe.json",
    },
}
serialized = json.dumps(report, allow_nan=False, sort_keys=True).encode()
report["report_content_sha256"] = hashlib.sha256(serialized).hexdigest()
atomic_write_json(temporary_root / "r1" / "report.json", report)
PY

"${UV_BASE[@]}" python - "$TEMP_ROOT" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

from self_improve_protein.config import load_protocol
from self_improve_protein.probes import validate_synthetic_probe

root = Path(sys.argv[1])
report = json.loads((root / "r1" / "report.json").read_text(encoding="utf-8"))
algebra = json.loads(
    (root / "r2" / "algebra_probe.json").read_text(encoding="utf-8")
)
probe = json.loads(
    (root / "r3" / "synthetic_probe.json").read_text(encoding="utf-8")
)
pytest_output = (root / "r2" / "pytest.txt").read_text(encoding="utf-8")
validate_synthetic_probe(probe)
if report.get("status") != "passed" or report.get("rung") != "R1":
    raise RuntimeError("R1 report did not pass schema validation")
unsigned_report = dict(report)
reported_content_digest = unsigned_report.pop("report_content_sha256", None)
expected_content_digest = hashlib.sha256(
    json.dumps(unsigned_report, allow_nan=False, sort_keys=True).encode()
).hexdigest()
if reported_content_digest != expected_content_digest:
    raise RuntimeError("R1 report content digest is invalid")
if not re.fullmatch(r"[0-9a-f]{40}", report.get("git_head", "")):
    raise RuntimeError("R1 report git revision is invalid")
commands = report.get("commands", [])
expected_command_names = {
    "r1_package_import",
    "r1_locked_config",
    "r1_cli_help",
    "r1_ruff",
    "r1_mypy",
    "r2_targeted_pytest",
    "r2_full_pytest",
    "r3_synthetic_probe",
}
if {command.get("name") for command in commands} != expected_command_names:
    raise RuntimeError("R1 report command set is incomplete")
if any(command.get("exit_code") != 0 for command in commands):
    raise RuntimeError("R1 report contains a failed command")
if report["config"]["dump"] != load_protocol(
    "configs/v0.yaml"
).model_dump(mode="json"):
    raise RuntimeError("R1 report config dump does not match the locked protocol")
if algebra != probe["algebra"] or algebra["finite_checks"]["all"] is not True:
    raise RuntimeError("R2 algebra artifact does not match the R3 probe")
if pytest_output.count("exit_code=0") != 2:
    raise RuntimeError("R2 pytest artifact does not record two passing commands")
PY

"${UV_BASE[@]}" python - "$TEMP_ROOT" "$ARTIFACT_ROOT" <<'PY'
import os
import sys
from pathlib import Path

temporary_root = Path(sys.argv[1])
artifact_root = Path(sys.argv[2])
relative_paths = (
    Path("r1/report.json"),
    Path("r2/pytest.txt"),
    Path("r2/algebra_probe.json"),
    Path("r3/synthetic_probe.json"),
)
for relative_path in relative_paths:
    source = temporary_root / relative_path
    destination = artifact_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
PY

printf 'R1-R3 verification passed; artifacts: %s\n' "$ARTIFACT_ROOT"
