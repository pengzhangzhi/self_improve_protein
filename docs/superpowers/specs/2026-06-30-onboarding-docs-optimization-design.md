# Onboarding Documentation Optimization Design

**Status:** Approved for autonomous implementation on 2026-06-30.

## Audience and goal

The primary reader is a mathematically sophisticated researcher who developed the
pseudo-sample selection theory but is new to protein fitness prediction, ProteinGym,
and this codebase. The documentation should let that reader answer three questions
quickly:

1. What scientific question did we test, and why is protein DMS a useful test bed?
2. What did the experiment find, including the negative result and its caveats?
3. Where should a researcher or maintainer go next to inspect, reproduce, or extend
   the work safely?

The public documentation must remain concise enough to scan, while operational
details must be complete enough that a maintainer does not need to infer Slurm job
dependencies, logs, completion criteria, or cancellation commands.

## Approaches considered

### 1. Put everything in the README

This gives one entry point, but mixes the scientific result, biology background,
code tour, and cluster operations. It would make the repository front page too long
and increase the chance that safety-critical run instructions become stale or hard
to find.

### 2. Keep the current two-document structure

The current README plus `docs/GETTING_STARTED.md` is friendly and accurate, but the
guide has two jobs: teaching the project and operating the cluster workflow. The
result is a long onboarding path with incomplete monitoring and completion guidance.

### 3. Use a three-layer documentation structure

This is the selected approach:

- `README.md` is the public front door: scientific question, answer, evidence,
  repository map, and links to the two focused guides.
- `docs/GETTING_STARTED.md` is the conceptual guide: protein/DMS background, theory
  to implementation mapping, experiment anatomy, result interpretation, and code
  tour.
- `docs/OPERATIONS.md` is the maintainer runbook: setup, verification commands,
  data and embedding prerequisites, Slurm resource footprint, submission,
  monitoring, logs, cancellation, artifacts, and extension workflow.

This separation supports both readers without duplicating command-heavy material.

## Content design

### README responsibilities

The README should stay between roughly 650 and 850 words. It should:

- state the research question and the negative v0 conclusion near the top;
- retain the main quantitative evidence and the essential caveats;
- explain in one paragraph what ProteinGym DMS assays, ESM-2 embeddings, the
  zero-shot teacher, and the ridge student contribute;
- provide a small “choose your path” navigation block;
- show the shortest local orientation commands, while linking all cluster execution
  details to `docs/OPERATIONS.md`;
- avoid presenting tests or successful execution as evidence that the method works.

### Getting-started guide responsibilities

The conceptual guide should target roughly 2,200 to 2,600 words and use plain
language before notation. It should:

- explain mutation fitness assays and why ranking metrics matter;
- map the paper notation to concrete arrays, models, and files;
- explain the four compared methods and why the teacher is non-identical;
- explain that weighted pseudo-examples change the balance between the data loss and
  a fixed ridge penalty, so random pseudo-label gains do not isolate selection;
- distinguish implementation validity from scientific evidence;
- walk through the code and result artifacts without embedding detailed Slurm
  operations;
- end with clear next steps for a theory reader and a code maintainer.

### Operations runbook responsibilities

The runbook should be command-oriented and explicit. It should include:

- environment setup and the local R1--R3 verification command;
- CLI discovery commands for `prepare-data`, `embed-assay`, `run-task`, `aggregate`,
  and `verify`;
- data, teacher-score, embedding, and configuration prerequisites;
- every required `SI_*` environment variable used by the launcher;
- an explicit warning that submission commands create cluster jobs;
- the development pipeline footprint: one data-preparation job, a nine-member
  one-GPU embedding array, a two-member task array, and one aggregation job;
- the confirmatory gate requirements and its 40-task array;
- `squeue` for live state, `sacct` for terminal state and exit codes, the Slurm log
  filename patterns, and `scancel` using the recorded job IDs;
- a success criterion: all stages and array elements are `COMPLETED` with exit code
  `0:0`, and the expected aggregate JSON exists and passes verification;
- the expected artifact paths, including `local/slurm/<run-id>/job_ids.json` and
  `<results-root>/<mode>/aggregate.json`;
- a safe extension sequence based on the experiment card and R0--R7 feedback
  ladder;
- an honest note that the public repository documents R4 criteria but does not
  currently expose a dedicated one-command reduced R4 launcher; development mode is
  the larger pilot workflow.

## Scientific interpretation constraints

All documents must preserve the following distinctions:

- The proposed full influence selector did not beat random pseudo-label selection in
  the completed v0 experiment: mean Spearman difference `-0.05526`, with `0/8`
  assay-level wins.
- Exact cross-validation selection did not reverse the result.
- The no-Hessian variant showed one small, non-significant locality-grid Spearman
  contrast (`+0.00654`, `p=0.6172`) and worse MSE; this is exploratory, not evidence
  of superiority.
- A same-student squared-loss teacher produces zero first-round pseudo-gradients at
  the fitted student and is therefore unsuitable for this test.
- Random pseudo-labeling improving on supervised-only is consistent with useful
  teacher signal, but it also changes effective regularization.
- Twenty-six untouched confirmatory outcomes remain sealed.

The authoritative interpretation remains `docs/results/overall-conclusion.md`.

## Safety and portability

- Do not publish private filesystem paths, account names, credentials, or cluster
  configuration values.
- Label commands that submit or cancel jobs before the command block.
- Do not claim a command is cheap when it computes all nine assay embeddings.
- Use placeholders such as `/path/to/...` only in user-supplied environment values,
  with each placeholder explained immediately.
- Keep all Markdown links repository-relative where possible.

## Acceptance criteria

1. The three documents have distinct responsibilities and no contradictory commands.
2. README and conceptual guide stay within their target word-count ranges.
3. Every published numerical result matches committed result artifacts.
4. Every documented CLI subcommand exists and its `--help` command succeeds.
5. Slurm commands, job counts, log patterns, and result paths match the launcher and
   batch scripts.
6. Local documentation links resolve, no private absolute paths are present, and no
   `TODO`/`TBD` placeholders remain.
7. The complete test suite and documentation checks pass after integration.
8. The optimized documents are committed to `main`, pushed to the public GitHub
   repository, and verified against the remote commit.
