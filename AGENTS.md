# AGENTS.md

## Purpose

This file defines persistent instructions for Codex when working in this Python research-code workspace.

The user is a PhD student in artificial intelligence and computer science. Treat this repository as an academic/research codebase where correctness, reproducibility, clarity, and minimal, well-motivated changes matter more than superficial speed.

## Workspace boundary

- Treat `/root/nas/mingjing/InherNet-Demo` as the workspace root.
- Only inspect, create, edit, move, or delete files inside this workspace root.
- Do not create, edit, move, or delete files outside the workspace root, including parent directories, sibling projects, system directories, hidden global configuration files, or unrelated datasets.
- Read-only environment discovery outside the workspace is allowed when needed to locate Python, conda, CUDA, or system package metadata for running this project. Keep this discovery narrow and do not print secrets or unrelated personal data.
- Do not follow symlinks that resolve outside the workspace root unless the user explicitly authorizes it.
- If a command would write outside the workspace, do not run it.
- If external files, credentials, datasets, checkpoints, or system resources appear necessary, ask the user or provide a safe fallback inside the workspace. Prefer workspace-local temporary files, logs, caches, and virtual environments.

## General working procedure

Before modifying code:

1. Inspect the project structure.
2. Read the most relevant local documentation and configuration files, such as:
   - `README.md`
   - `AGENTS.md`
   - `pyproject.toml`
   - `setup.py`
   - `setup.cfg`
   - `requirements.txt`
   - `environment.yml`
   - `tox.ini`
   - `.pre-commit-config.yaml`
   - relevant scripts, tests, configs, and examples
3. Identify the intended architecture, module boundaries, entry points, experiment workflow, and testing conventions.
4. Prefer changes that fit the existing design instead of introducing a new structure without need.
5. Make the smallest coherent change that solves the requested problem.
6. Consider downstream effects on training, evaluation, reproducibility, checkpoint loading, configuration parsing, logging, and tests.

Do not make broad refactors unless the user asks for them or they are required to solve the task safely.

## Python environment requirement

Every time Codex runs, tests, debugs, formats, lints, type-checks, benchmarks, or otherwise executes Python-related code, it should use a project environment with the dependencies in `requirements.txt`.

Preferred environment:

```bash
conda activate inherdemo
<command>
```

If `conda` or the `inherdemo` environment is unavailable, use a workspace-local fallback instead of modifying system Python:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
<command>
```

When installing dependencies for the fallback environment, keep caches inside the workspace when practical, for example:

```bash
PIP_CACHE_DIR="$PWD/.cache/pip" python -m pip install -r requirements.txt
```

If neither conda nor a workspace-local virtual environment can provide the needed dependencies, run non-import checks such as `python3 -m py_compile` and report the blocker clearly.

## Python coding standards

Write Python that is robust, readable, and maintainable.

Prefer:

* Clear module boundaries.
* Explicit imports.
* Type hints for public functions, complex internal functions, dataclasses, and configuration objects.
* Small functions with single responsibilities.
* Descriptive names for variables, functions, classes, and configuration fields.
* Deterministic behavior where possible.
* Explicit error messages for invalid inputs, missing files, malformed configs, shape mismatches, and unsupported modes.
* Path handling with `pathlib.Path`.
* Logging instead of ad hoc `print` statements for reusable library code.
* Tests or small validation scripts for behavior that is changed.

Avoid:

* Hidden global state.
* Unnecessary mutation.
* Overbroad exception handling.
* Silent fallback behavior that can hide experimental bugs.
* Hard-coded absolute paths.
* Unseeded randomness in tests.
* Large rewrites when a targeted fix is sufficient.
* Adding dependencies unless clearly justified.

## Research-code expectations

For AI/ML code, pay special attention to:

* Tensor shapes, dtypes, devices, and broadcasting behavior.
* Correct train/eval mode handling.
* Gradient flow and accidental `.detach()`, `.item()`, `torch.no_grad()`, or in-place operation issues.
* Random seeds and reproducibility.
* Dataset splits and leakage.
* Config compatibility.
* Checkpoint loading and backward compatibility.
* Metrics correctness.
* Numerical stability.
* Memory use and unnecessary GPU synchronization.
* Distributed-training assumptions.
* Batch-size-dependent behavior.
* Evaluation code matching the stated experimental protocol.

Do not change experimental semantics casually. If a requested code improvement may alter reported results, make that explicit.

## Testing and validation

After modifying code, run the narrowest relevant checks first, then broader checks if appropriate.

Use existing project conventions when available. Typical commands may include:

```bash
python -m pytest
python -m pytest tests/<relevant_test_file>.py
python -m ruff check .
python -m ruff format .
python -m mypy .
python -m pyright
python -m unittest
```

All such commands should be run inside the active project environment described above.

When tests cannot be run:

* Explain why.
* State what was checked instead.
* Identify the most relevant command the user should run.

Do not claim that code is tested unless a relevant command was actually run successfully.

## Editing discipline

When changing files:

* Preserve backward compatibility where reasonable.
* Update tests when behavior changes.
* Update documentation or comments when user-facing behavior changes.
* Keep diffs focused.
* Do not reformat unrelated files.
* Do not rename files, functions, classes, or configuration keys unless necessary.
* Do not delete code unless it is clearly obsolete, unused, or part of the requested change.
* Prefer local fixes over global rewrites.
* Maintain compatibility with the project’s supported Python version.

## Dependency policy

Before adding a dependency:

1. Check whether the project already has an equivalent dependency.
2. Prefer standard-library solutions when adequate.
3. Consider reproducibility and environment stability.
4. Ask the user before adding new heavy dependencies, especially ML, CUDA, distributed-computing, or data-processing packages.


## File and data safety

* Do not overwrite existing experiment outputs unless the user explicitly asks.
* Prefer writing temporary outputs to a clearly named temporary location inside the workspace.
* Do not commit, expose, or print secrets, tokens, credentials, API keys, private paths, or personal data.

## Communication style for Codex

When reporting work:

* Summarize the project context considered.
* State the files changed.
* State the behavioral effect of the change.
* State the validation commands run, including whether they passed or failed.
* Mention any remaining risks, assumptions, or follow-up checks.
* Be concise but precise.

When uncertain:

* Make a reasoned best effort based on local evidence.
* Ask a question only when ambiguity could cause an unsafe, destructive, or semantically incorrect change.
* Prefer explaining trade-offs over making hidden assumptions.

## Definition of done

A task is complete when:

* The requested behavior is implemented.
* The change is consistent with the surrounding project architecture.
* Relevant tests or checks have been run with the required conda environment activated.
* Any failures are either fixed or clearly reported.
