# Project Rules — visionMamba3

## Python / packaging

- **Never invoke `pip` directly.** Use `uv` for everything.
  - Add packages: `uv add <pkg>`
  - Run: `uv run python ...`, `uv run pytest`
  - Sync after manual edits to `pyproject.toml`: `uv sync`

## third_party layout

- `third_party/mamba-ssm` is a **git submodule** on official upstream
  (`https://github.com/state-spaces/mamba.git`, branch `main`). Never edit files inside it.
- Clone with `git clone --recursive`, or after the fact:
  ```bash
  git submodule update --init --recursive
  ```
- Submodule, not symlink, on purpose: we call its Triton SSD kernels directly, so its version decides
  our numbers. It used to symlink a shared local clone, and that silently broke reproducibility — the
  clone sat on a stale branch whose kernel had a backward-pass dtype bug, so `Mamba3SelfAttention`
  fell back to the slow reference path and Table 1's bidirectional row could not be reproduced, with
  nothing in this repo recording which version had produced it.
- Bump it deliberately (`git -C third_party/mamba-ssm fetch && git checkout <newer>`), commit the new
  pin, and re-run any benchmark whose numbers you intend to keep quoting.
- We never import the `mamba_ssm` top-level package: its `__init__` eagerly pulls in cute/tilelang
  backends needing `tilelang`, `cutlass` and `quack`. `visionmamba3/__init__.py` registers it as a bare
  namespace package so only the Triton kernels under `mamba_ssm.ops.triton` load. Don't add those three
  as dependencies — they are not needed.

## Sudo

- Never run `sudo` directly. If a command needs sudo, print the exact command and ask the user to run it.

## Code style

- Prefer concise expressions (comprehensions over for+append loops).
- Do not add comments that restate the code; only add comments when the *why* is non-obvious.

## Memory

- Memory files for this project live at `memory/*.md` under this repo, indexed by `MEMORY.md` at the repo root.
