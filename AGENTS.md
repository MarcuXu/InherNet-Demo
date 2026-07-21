# Repository guidance

This is an ML research repository for Hetero and its InherNet baseline. Prioritize algorithmic correctness, reproducibility, and comparable experiments over broad refactoring.

## Source of truth

- Read `README.md` before changing training or model behavior; it defines the maintained workflow, methods, datasets, metrics, and CLI examples.
- Use `scripts/run.sh` / `demo_code.py` for maintained experiments. All maintained shell launchers live under `scripts/`; `scripts/search.sh` is the only hyperparameter-search entry point. `GenericInherNet` is the maintained baseline; `demo_code_org.py` alone is frozen legacy-demo compatibility. Change either method's semantics only when the task explicitly targets it.
- Discover current code and commands from the repository. Do not maintain a file inventory in this document.

## Environment and validation

Run Python commands in the project environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate inherdemo
```

Do not install packages into system Python. If the environment is unavailable, report that limitation instead of silently changing environments.

- Unit tests: `python -m unittest discover -s tests -v`
- CLI options and current experiment examples: `scripts/run.sh --help` and `README.md`
- Validate narrowly first. For an affected training path, use its documented `--smoke-test --plot-mode none` form before considering a longer run. Do not start full training or overwrite prior outputs unless explicitly requested.

## Research invariants

- Preserve experiment comparability: seeds, data splits, preprocessing, model defaults, optimizer/scheduler behavior, checkpoint compatibility, metric definitions, and structured log fields must not drift accidentally.
- For tensor code, verify shapes, dtypes, devices, train/eval state, gradient flow, and numerical stability. Avoid hidden detaches or in-place operations that alter optimization.
- For algorithmic changes, test the relevant invariant (for example reconstruction, rank budget, routing, or expert gradient flow), not only that execution completes.
- Preserve the shared `--size small|large` comparison: Hetero uses the same registered rank and exactly the same parameter count as the corresponding InherNet construction. Do not reintroduce user-facing ratio or budget-scope controls.
- Clearly report any intentional change that can alter metrics or invalidate comparison with earlier logs.

## Change discipline

- Surface material assumptions and trade-offs, choose the simplest adequate design, and keep every changed line traceable to the request.
- Match local style and preserve public/configuration compatibility unless the requested behavior requires a break.
- Do not overwrite existing logs, results, checkpoints, datasets, or caches. Keep generated artifacts out of commits unless requested.
- Add dependencies only when necessary and update tests/documentation when a user-visible or experimental contract changes.

## Completion

Run the narrowest relevant checks and report the exact commands and outcomes. Distinguish verified facts from hypotheses and note remaining experimental validation.
