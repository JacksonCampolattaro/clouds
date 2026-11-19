# AGENTS.md

`clouds` is a collection of PyG-compatible point-cloud datasets plus a library of `BaseTransform`-style
transforms. Library code is in `src/clouds/`; tests mirror it under `tests/clouds/`.

## Toolchain

- Python >=3.12 (`uv`-managed, backend `uv_build`).
- The dev accelerator stack (PyTorch 2.12 + CUDA 13 and prebuilt PyG extension wheels) must be installed with `uv sync --dev`.

Commands:

- All tests: `uv run pytest`
- One test: `uv run pytest tests/clouds/transforms/test_fps.py::test_name`
- Lint / format: `uv run ruff check src tests` and `uv run ruff format src tests`
  (line length 128, `quote-style = "preserve"` — don't reflow quotes)
- Type check: `uv run ty check`

Lint is clean; `ty check` is **not** clean on `main` and reports many diagnostics (PyG reflection / private
paths) — don't assume your change caused them.

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

## Coding style

- **Annotate everything** in `src/clouds` — parameters and return types, public and private. Every
  `__init__` ends `-> None`; leave pass-through `**kwargs` unannotated (it already means `Any`).
- **Modern typing only**: `collections.abc.Callable`, builtin generics (`list[...]`, `dict[...]`), and
  `X | None` / `X | Y`. Do not use `typing.Optional` / `Union` / `List` / `Callable`; keep only
  `ClassVar` / `Any` from `typing`. Forward references to names that may not be importable at runtime
  (e.g. the optional `KDTree`) must be quoted strings.
- **Quoting**: use double quotes for human-readable output — exception messages, `__repr__`, printed
  text, docstrings — and single quotes everywhere else, especially string literals used as keys or
  identifiers (dict keys, split names, filenames, attribute names).
- **Imports within `clouds` are relative** (`from .knn import knn`, `from ..data import ...`); external
  imports stay absolute. `__init__.py` re-exports with the explicit `from .mod import Name as Name` form.
- **Transforms are thin `BaseTransform` subclasses**: `def forward(self, data: Data) -> Data:` mutates
  `data` in place and returns it. Selection transforms follow the `*Select` / `*Sample` protocol above.
- **Batch handling**: detect batching with `isinstance(data.batch, Tensor)`; iterate graphs with
  `itertools.pairwise(data.ptr)` or `data.batch == i`; pass `getattr(store, 'batch', None)` /
  `getattr(store, 'ptr', None)` into PyG aggregations; count graphs with
  `data.batch_size if hasattr(data, 'batch_size') else ...`.
- **Preconditions use `assert` / `raise`** rather than logging, and `__repr__` follows
  `f"{self.__class__.__name__}(...)"`.
- **Optional backends** use the `try: import X` / `except ImportError` + `HAS_*` flag + device check +
  fallback-with-warning pattern (see `transforms/knn.py`).

## Package layout

The package mirrors PyG's structure: data structures (`Data`/`Dataset` subclasses) live in
`clouds.data`, loaders in `clouds.loader`, dataset definitions in `clouds.datasets`, transforms in
`clouds.transforms`, neural-network layers in `clouds.nn`, and visualization in `clouds.visualization`.
`clouds/__init__.py` eagerly imports every subpackage and exposes `__version__` / `__all__` (plus the
`clouds.home` helpers).

- `clouds.nn` mirrors `torch_geometric.nn`: `resolver.py` (activation/normalization resolvers),
  `sequential.py`, `dense/linear.py` (`Linear`), `models/` (`MLP` plus the DeLA MLPs), `norm/`
  (DeLA batch/layer/RMS norms), `conv/` (generic source-indexed convs plus the DeLA/PointNeXt/spatial
  encoding layers), `functional/` (losses), `edge_scaler.py` (edge feature encoders), and `multigrid/`
  (wrappers over `clouds.data.MultiGridData`). `apply.py` holds the `Data`-dispatch helpers. The DeLA
  conv uses `pool` (a non-PyPI git dependency declared as a runtime dependency whose source is inherited
  from `pyproject.toml`).

- `clouds.home` holds the cache-root helpers `get_home_dir()` / `set_home_dir()` (env `$CLOUDS_HOME`,
  default `~/.cache/clouds`), mirroring PyG's `torch_geometric.home`, plus `get_dataset_root(name)`.
- `clouds.data` holds only containers/machinery (`SourceIndexedData`); `clouds.loader` holds
  `ThreadingDataLoader`. Keep new loaders in `clouds.loader`, not `clouds.data`.

## Datasets

- One loader per file `src/clouds/datasets/<name>.py`, re-exported (with aliases) from
  `clouds/datasets/__init__.py`. The package follows PyG: classes are reached via `clouds.datasets.*`
  (e.g. `from clouds.datasets import ModelNet40`); `clouds/__init__.py` does **not** re-export dataset
  classes. `DALES` is re-exported like the others.
- Loaders subclass `InMemoryDataset`/`Dataset` and take their data location from the caller. There is no
  default cache root inside the package; each loader's `__main__` gets its root from
  `clouds.home.get_dataset_root('<Name>')`, which uses `sys.argv[1]` when given and otherwise falls back
  to `get_home_dir()`. Never commit downloaded data. `.data/`, `data.daic/`, `*.pkl`, and `*.ckpt` are
  gitignored.
- `tests/clouds/datasets/test_dataset_importability.py` checks that every public dataset class in
  `clouds.datasets` is reachable. `SemanticWaymo` has synthetic-frame tests in `test_waymo.py`; the other loaders themselves still have no functional tests.
- `SemanticWaymo` (`datasets/waymo.py`) stores one frame per `numpy` `.npz` file,
  `processed/<split>/<segment>/<frame:04d>.npz`, written with `np.savez_compressed` (deflate; the zip CRC32 provides
  per-frame integrity). Arrays are `pos` (float32 `[n,3]`), `intensity` (float32 `[n,1]`, post-`tanh`), and `y`
  (int8, −1…21) only for labeled frames. Per-split JSON indexes (`processed/index_{training,validation,testing}.json`,
  `{"version": 2, ...}`) map each segment to its frame count. The only build path is raw `.tfrecord` → `.npz`; an
  interrupted build resumes (scans whose output directory already exists are skipped). Processed-only roots (no `raw/`)
  load fine, and are meant to live on local disk — reading many small `.npz` files over CIFS is slow. Build and verify with
  `python -m clouds.datasets.waymo process|verify --root <root> --split <training|validation|testing>`. Builds take hours; never run them
  unattended without a log.

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
