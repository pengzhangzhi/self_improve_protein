# Coworker Onboarding Documentation Design

**Status:** Approved by the user's autonomous execution instruction on
2026-06-30.

## Audience

The primary reader is a mathematically sophisticated research collaborator who
developed the theory but is new to protein machine learning, Python research
code, and cluster execution. The documentation must respect that expertise
while explaining biological and software vocabulary from first principles.

## Goal

After reading the documentation, the collaborator should be able to:

1. explain the biological prediction problem and the low-label setting;
2. map the manuscript notation to the implemented experiment;
3. understand exactly what was tested, what was not tested, and why;
4. interpret the negative result without mistaking it for a software failure;
5. navigate the repository and identify the source of truth for each claim;
6. install the project, run safe inspection commands and tests, and understand
   what full cluster reproduction entails; and
7. propose a new experiment without silently changing the scientific claim or
   contaminating held-out outcomes.

## Approaches considered

### One long README

This gives a single entry point, but it makes the repository landing page too
long and mixes orientation, scientific detail, and operational instructions.

### Layered README plus coworker guide

This is the selected approach. The README becomes a friendly front door with a
plain-language summary, the result, a short repository tour, and a first-hour
path. A dedicated `docs/GETTING_STARTED.md` contains the complete conceptual
and operational handoff. Existing experiment cards, audits, and result memos
remain the authoritative technical records.

### Notebook-first tutorial

A notebook could be approachable, but it introduces another execution surface,
duplicates tested code, and encourages readers to treat an expensive,
provenance-locked study as an interactive toy. It can be added later if a
specific teaching need emerges.

## Information architecture

### README

The README will answer, in order:

1. What is this project?
2. What did we learn?
3. What does the result mean for the theory?
4. Where should a new collaborator start?
5. How is the repository organized?
6. What are the shortest safe setup and verification commands?
7. Where are the detailed guide and technical evidence?

### `docs/GETTING_STARTED.md`

The guide will contain:

1. a note to the collaborator and a recommended reading path;
2. protein, mutation, DMS, fitness, assay, and ProteinGym background;
3. the low-label regression problem in both plain language and notation;
4. a manuscript-to-code notation table;
5. the external-teacher adaptation and literal self-teacher degeneracy;
6. the locked dataset, split, teacher, representation, student, selectors,
   training objective, metrics, and inference;
7. the sequence of v0, crossfit, locality, and exact-CV studies;
8. the supported conclusion and explicit claim boundary;
9. a repository map organized by scientific responsibility;
10. local setup, first commands, expected outcomes, and cluster reproduction;
11. a safe workflow for changing the method or starting a new study;
12. troubleshooting, glossary, and pointers to authoritative documents.

## Writing style

- Define a term before using its abbreviation.
- Lead each technical section with the intuition, then show equations or code.
- Use short examples and notation tables instead of assuming protein knowledge.
- Distinguish evidence, interpretation, and speculation explicitly.
- Avoid describing a negative scientific result as a failed project.
- State when a command is cheap, when it downloads data, and when it uses GPUs.
- Never include private cluster paths, credentials, or untracked raw artifacts.

## Scientific safeguards

The guide must preserve these boundaries:

- The implementation tests an externally supplied ESM-1v teacher, not the
  manuscript's literal same-student pseudo-labeler.
- The literal squared-loss self-teacher has zero pseudo-gradients at the first
  supervised optimum and therefore no candidate-ranking signal.
- The primary v0 comparison is influence selection versus random selection.
- Random pseudo-labeling helps, while the proposed selection score does not.
- Crossfit, locality, and exact-CV are exploratory diagnosis, not post hoc
  replacements for the confirmatory result.
- The 26 untouched assay outcomes remain sealed.
- The result is specific to this teacher, representation, student, label
  budget, and benchmark slice; it is not a general impossibility theorem.

## Operational safeguards

- `configs/v0.yaml` and `uv.lock` remain the protocol and environment sources
  of truth.
- Commands shown in the guide must be copied from actual CLI help or tested in
  dry-run/read-only mode.
- Large ProteinGym inputs, embeddings, task shards, logs, and model weights are
  not committed to Git.
- A full rerun uses the Slurm launchers and requires site-specific `SI_*`
  environment variables; the guide must not invent portable defaults for them.
- New scientific variants require a new experiment card and untouched evidence
  rather than editing the interpretation of the completed v0 study.

## Verification and acceptance criteria

The documentation is complete when:

- every local Markdown link in README and the guide resolves;
- every shown lightweight command is syntactically valid and its stated
  behavior matches current CLI output;
- all reported headline numbers match the reviewed CSV tables and result memos;
- no tracked documentation contains a private cluster path or credential;
- package tests, Ruff, and strict mypy remain green;
- the worktree is committed and pushed to public `main`; and
- local HEAD equals `origin/main` after publication.
