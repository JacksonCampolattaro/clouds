# AGENTS.md

`clouds` is a collection of PyG-compatible point-cloud datasets plus a library of `BaseTransform`-style
transforms. Library code is in `src/clouds/`; tests mirror it under `tests/clouds/`.

## Toolchain

- Python >=3.12 (`uv`-managed, backend `uv_build`).
- The dev accelerator stack (PyTorch 2.12 + CUDA 13 and prebuilt PyG extension wheels) is configured in
  `uv.toml`, deliberately **not** `pyproject.toml`, so consumers don't inherit the CUDA index.
  Run `uv sync --dev` to install it.
- `uv.lock` is gitignored and machine-specific; do not commit it or treat it as a source of truth.
- `ruff` and `ty` are not project dependencies; they resolve from `PATH`.

Commands:

- All tests: `uv run pytest`
- One test: `uv run pytest tests/clouds/transforms/test_fps.py::test_name`
- Lint / format: `uv run ruff check src tests` and `uv run ruff format src tests`
  (line length 128, `quote-style = "preserve"` — don't reflow quotes)
- Type check: `uv run ty check`

Lint and typecheck are **not clean** on `main`: `ruff` reports pre-existing violations (unused imports /
loop vars in the dataset loaders) and `ty check` reports many diagnostics. Don't assume your change caused
them.

## Transform / data conventions (non-obvious)

- `edge_index` is stored in **dense `(N, K)` kNN form**, not PyG's `(2, E)`. `clouds.data.SourceIndexedData`
  overrides `__cat_dim__` to batch this, and `select_knn_edges` remaps it. Assuming `(2, E)` breaks batching.
- Selection is a shared protocol: `*Select` transforms write `data.selection_index` (a sorted node-index
  tensor, batch-aware). `*Sample` variants are `Select` + `apply_selection`. `ApplySelection` filters node
  attributes down to the selected nodes, remaps dense edges, and **drops** edge attributes and any key whose
  name contains `index`.
- `clouds.transforms.pipeline.unpack_pipeline` is a state machine over a list where the sentinel strings
  `'collate'`, `'device'`, `'model'` mark stage boundaries (`transform` / `batch_transform` /
  `finalize_transform` / `post_transform`). See `tests/clouds/transforms/test_pipeline.py`.
- Transforms are re-exported with stable aliases from `clouds/transforms/__init__.py`;
  `test_importability.py` enforces that every imported `BaseTransform` subclass is reachable. Add new
  transforms there.
- `knn()` picks a backend by device + availability: `pykeops` (CUDA), `pynanoflann` (CPU), else a slow PyG
  fallback with a warning.

## Datasets

- One loader per file `src/clouds/<name>.py`, re-exported (with aliases) from `clouds/__init__.py`.
  `dales.py` defines `DALES` but is **not** re-exported from the package.
- Loaders subclass `InMemoryDataset` and cache processed data into the package tree at runtime
  (`src/clouds/data/<Name>/`, or `src/clouds/.data/...` for S3DIS). Caches are multi-GB and excluded from
  wheels via `[tool.uv.build-backend] source-exclude`; never commit them. `.data/`, `data.daic/`, `*.pkl`,
  and `*.ckpt` are gitignored.
- Dataset loaders have no unit tests; only `clouds.transforms` is covered.

## Tests

- `tests/conftest.py` seeds `random` and `torch` for every test (autouse fixture).
- `tests/clouds/transforms/conftest.py` provides the `make_point_cloud` fixture.
- Optional native accelerators (`pyg-lib` FPS, `torch_fpsample`, `pykeops`, `pynanoflann`) and CUDA tests are
  guarded with `pytest.mark.skipif`; CPU fallbacks keep the suite green. The transforms suite runs in ~15s.

## Version control (colocated jj + git)

This is a **colocated Jujutsu/Git** repo: `.jj/` and `.git/` share the working copy, and jj imports/exports on every `jj` command. **Agents may use `git`** for everyday work, but because jj owns the working copy, prefer read-only git and avoid commands that mutate the working copy, HEAD, or history.

- Safe: `git status`, `git diff`, `git log`, `git show`, `git grep`, `git fetch`, `git push`. You can also `git add`/`git commit`; jj will pick those commits up and they can be recovered with `jj undo` / `jj op restore`.
- Avoid (use the jj equivalent): `git reset --hard`, `git checkout` / `git switch` / `git restore`, `git stash`, `git rebase`, `git commit --amend`, `git merge`, and `git worktree` / sparse-checkout / submodules (unsupported). jj does not understand git's staging area or interrupted rebase/merge states, and rebase drops change-id headers, causing divergent jj changes.
- **Never run `git clean -x`, `-X`, or `-dfx`**: `.jj/` is gitignored, so `-x` deletes the jj workspace metadata (along with the gitignored `uv.lock`, `.venv`, `*.ini`, and dataset caches).
- jj's staging area is ignored, so `git add` has no effect on jj; all working-copy edits are automatically part of the current jj change.
- jj leaves git in a detached-HEAD state (expected). If a mutating git command causes trouble, recover with `jj undo` or `jj op restore`; when changing history, prefer `jj abandon` / `jj restore` / `jj squash` / `jj rebase` / `jj describe`.
