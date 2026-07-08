# Project Rules — visionMamba3

## Python / packaging

- **Never invoke `pip` directly.** Use `uv` for everything.
  - Add packages: `uv add <pkg>`
  - Run: `uv run python ...`, `uv run pytest`
  - Sync after manual edits to `pyproject.toml`: `uv sync`

## third_party layout

- `third_party/mamba-ssm` is an absolute symlink → `/home/mas/proj/study/mamba-ssm`. Never edit files inside it.
- After cloning on a new machine, recreate the symlink:
  ```bash
  ln -s /home/mas/proj/study/mamba-ssm third_party/mamba-ssm
  ```

## Sudo

- Never run `sudo` directly. If a command needs sudo, print the exact command and ask the user to run it.

## Code style

- Prefer concise expressions (comprehensions over for+append loops).
- Do not add comments that restate the code; only add comments when the *why* is non-obvious.

## Memory

- Memory files for this project live at `memory/*.md` under this repo, indexed by `MEMORY.md` at the repo root.
