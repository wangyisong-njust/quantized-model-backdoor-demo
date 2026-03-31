# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `models/`, `attacks/`, `defenses/`, `datasets/`, `eval/`, `quant/`, `deploy/`, and `utils/`. Entry points are in `scripts/` for reproducible runs and `demos/` for presentation-ready pipelines. Configs are split by domain under `configs/cls`, `configs/det`, `configs/quant`, and `configs/attack`. Generated reports, figures, and experiment outputs go to `outputs/`. Treat `third_party/` as vendored upstream code; avoid broad edits there unless the change is intentionally upstream-specific.

## Build, Test, and Development Commands
Create the local environment with `conda env create -f environment.yml` and activate `demo_adv`, or install Python deps with `pip install -r requirements.txt` after installing PyTorch separately. Run `python scripts/check_env.py` first to verify CUDA, timm, ONNX Runtime, and optional MMDetection packages. Common smoke runs:

- `python scripts/cls_baseline.py --max_batches 2`
- `python scripts/det_baseline.py`
- `python scripts/cls_ptq.py`
- `conda run -n qura python demos/final_vit_patchdrop_demo.py`

Use OmegaConf overrides instead of editing configs for one-off experiments, for example `python scripts/cls_baseline.py dataset.data_type=demo`.

## Coding Style & Naming Conventions
Use 4-space indentation and follow standard Python style. Prefer `snake_case` for files, functions, and config keys; use `PascalCase` for classes. Keep modules focused: model wrappers in `models/`, evaluation logic in `eval/`, and shared helpers in `utils/`. There is no repo-wide formatter config, so keep imports tidy, avoid dead code, and preserve the existing docstring-heavy research style.

## Testing Guidelines
This repository does not currently ship a first-party `tests/` suite outside vendored code in `third_party/qura/test/`. Validate changes with targeted smoke runs and small datasets or `--max_batches` limits. When touching data loaders, document the expected directory layout from `docs/data_setup.md`. If you modify vendored QuRA code, run the relevant upstream tests in `third_party/qura/test/` separately.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects such as `Add ...` and `Complete ...`; keep that pattern. PRs should state the problem, the configs or datasets used, and the exact output paths affected, such as `outputs/cls/quant/`. Include screenshots when changing demo panels or generated figures, and note any heavyweight artifacts that were intentionally omitted because they are ignored by `.gitignore`.
