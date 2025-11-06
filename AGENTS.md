# Agent Instructions

This repository's README contains the canonical overview of the project, including the detailed file tree and additional documentation links. Please refer to [README.md](README.md) for the complete context when exploring the codebase.

## Key conventions
- Code is formatted with **black** and **isort**.
- Prefer dataclasses for structs and other classes, and keep related logic inside classes instead of free functions when practical.
- Use single-line comments when possible; short comments may share a line with code separated by two spaces. Reserve multiline comments for substantial explanations.
- Add type annotations throughout the codebase and include shape comments for tensors when helpful.

## Testing
- Unit tests live under `test/` and should be run with `pytest`.
- Long-running tests are marked with the `slow` marker; skip them during quick iterations via `pytest -m "not slow"`.

## Ignored directories
The following paths are treated as transient artifacts and should not be committed:
`.cache/`, `ckpt/`, `inference_outputs/`, `lightning_logs/`, `multiflow_weights/`, `venv/`, `wandb/`.
