# Repository Organization

A map of `protein_ENM_training/`: what lives where, and how **simulation**,
**training**, **analysis**, and **plotting** code are kept separate.

## Mental model — three code layers

| Layer | Directory | Role | Depends on |
|-------|-----------|------|------------|
| **Base** | `base/` | Shared primitives: structure I/O, ENM construction, physics/simulation, paths, protein metadata | nothing else in-repo |
| **Training** | `training/` | Fit ENM edge stiffnesses to a target (produces `data/trained_ENMs/`) | `base/` |
| **Analysis** | `analysis/` | Load trained ENMs, compute derived quantities, make figures | `base/` |

`base/` is the only shared dependency. Training never imports from `analysis/`
and vice versa — anything both need (FIRE minimiser, Kabsch fit, elastic
energy, edge building) lives in `base/`.

## Top-level layout

```
base/         shared primitives (simulation, structure I/O, config, ENM class)
training/     stiffness-fitting code
  src/        importable library (loss functions, optimizers, checkpoint I/O)
  runners/    entrypoint scripts + SLURM array wrappers
analysis/     analysis library modules
  notebooks/  Jupyter notebooks — one per analysis theme
data/         inputs and generated artifacts (mostly git-ignored)
plots/        generated figures (git-ignored)
docs/         design notes, plans, methodology write-ups
scripts/      operational tooling (cluster <-> local sync)
benchmarks/   solver/performance benchmarking (one-off, self-contained)
Leibler_analysis_plot_notebooks/   external reference notebooks (git-ignored)
```

## `base/` — shared primitives

| File | Contents |
|------|----------|
| `simulate.py` | Physics: FIRE minimiser, Kabsch alignment, elastic energy. Used by both training and analysis. |
| `elastic_network.py` | `ElasticNetwork` class (positions, edges, rest lengths, stiffnesses, NetworkX graph). |
| `structure_io.py` | Load Cα coords from PDB, resolve allosteric pocket indices, build edges. |
| `config.py` | Repo-root-relative path helpers (`plots_path(...)`, `TRAINED_ENMS_DIR`, `OPTIMIZER`). |
| `protein_lookup.py` | Per-protein metadata (PDB ids, chains, conformation pairs). |
| `seq_align_utils.py`, `oligomer_utils.py`, `download_pdbs.py` | Sequence alignment, oligomer handling, PDB fetching. |
| `training_config.json` | Chosen per-`(protein, training_fn, solver)` learning rates and reg strengths. |

## `training/` — stiffness fitting

**`training/src/`** — importable library, no side effects on import:

| File | Contents |
|------|----------|
| `loss_functions.py` | JAX loss functions `(stiffness, network, ...) -> scalar` + JIT'd `grad_*`. |
| `training_utils.py` / `training_utils_backprop.py` | Training loops (IFT/linear solver vs full backprop). |
| `optimizers.py`, `adaptive_lr_utils.py` | Optimizer setup, per-training-fn default learning rates. |
| `checkpoint_io.py`, `training_io.py`, `robust_save_history.py` | Checkpoint resume, history persistence. |
| `protein_enm_trained_losses.py` | Loss recomputation on saved trajectories. |

**`training/runners/`** — executable entrypoints:

- `train_*.py` — one per target type: `train_full_conformation.py`,
  `train_input_output.py`, `train_target_rmsd.py`, `train_target_NN_rmsd.py`,
  `train_protein_cooperativity.py`. Each takes `--protein --seed ...` and
  writes a dated run dir under `data/trained_ENMs/{protein}/`.
- `sweep_learning_rate.py`, `sweep_regularization.py` — hyperparameter sweeps
  (train short runs across a grid, then plot). Do **not** write
  `training_config.json`; LR/reg picks happen in a separate analysis pass.
- `generate_job_params.py`, `generate_hessian_jobs.py` — build CSV parameter
  files for SLURM job arrays by scanning existing seeds.
- `*_array.sh`, `run_*.sh`, `submit_*.sh` — SLURM array / batch wrappers.
- `compute_*.py` (`compute_cost_hessian.py`, `compute_hessian_subprocess.py`,
  `compute_trained_stats.py`, `compute_null_stats.py`) — batch precompute of
  quantities the analysis notebooks consume from cache.
- `check_training_progress.py` — run-status reporting.

## `analysis/` — analysis + plotting

**Library modules** (imported by the notebooks):

| File | Contents |
|------|----------|
| `data_io.py` | Discover/load trained runs by date tag, parse filenames, load stiffness histories. |
| `enm_analysis_shared.py` | Shared constants (`PROTEINS`, training-fn lists), network setup, stats persistence. |
| `susceptibility.py` | Per-edge susceptibility norms from an `ElasticNetwork`. |
| `hessian.py`, `elastic_hessian.py`, `modes.py` | Hessian assembly (cost & elastic), normal modes, Lanczos. |
| `strain_decomposition.py`, `mutation_forcing.py`, `mutation_utils.py`, `mutant_structures.py` | Strain analysis, mutation-as-forcing, AlphaFold mutant handling. |
| `null_models.py` | Uniform / random-uniform stiffness null cases. |
| `plotting.py` | **All matplotlib/seaborn figure functions.** Each takes pre-computed data and writes under `plots/`. |
| `visualization.py` | 3D network rendering, rotating-network GIF helpers. |
| `analysis_utils.py` / `analysis_utils_backprop.py` | Older catch-all utilities (kept for compatibility). |

**`analysis/notebooks/`** — orchestration only. One notebook per theme; each
loads via `data_io` / cache, calls analysis modules for computation and
`plotting.py` for figures:

| Notebook | Theme |
|----------|-------|
| `training_diagnostics.ipynb` | Loss trajectories, LR-sweep aggregation, well-trained cutoffs. |
| `cost_hessian_spectral_analysis.ipynb` | Cost-Hessian (wrt stiffness) spectra, top eigenvectors. |
| `elastic_hessian_spectral_analysis.ipynb` | Elastic-Hessian spectra / normal modes. |
| `susceptibility_costhessian_analysis.ipynb` | `abs(eigenvector)` vs susceptibility, rank-order stability. |
| `displacement_overlap_analysis.ipynb` | Displacement / mode overlap. |
| `trained_vs_baseline_mechanics.ipynb` | Trained vs uniform-stiffness mechanical response. |
| `strain_protein_analysis.ipynb`, `strain_pdzpsd95.ipynb` | Strain-field analysis. |
| `mutation_*.ipynb` | AlphaFold mutant ingest, shear comparison, forcing, GMM covariance. |
| `LacR_actuation.ipynb`, `io_definition.ipynb` | Actuation, input/output node definitions. |
| `directivesForNotebooks/` | Markdown specs describing intended notebook behaviour. |

## Where each concern lives

| Concern | Location |
|---------|----------|
| **Simulation / physics** | `base/simulate.py` (+ `base/elastic_network.py`). Nowhere else. |
| **Training loops & losses** | `training/src/`. Invoked only through `training/runners/`. |
| **Analysis computation** | `analysis/*.py` modules (one concept per file). |
| **Plotting** | `analysis/plotting.py` and `analysis/visualization.py` — separated from computation; functions receive pre-computed data and only render + save. |
| **Notebooks** | Thin orchestration: load → call analysis module → call plotting → save. No heavy logic inline. |
| **Batch precompute for notebooks** | `training/runners/compute_*.py` → results cached in `data/hessians_cache/`. |

## `data/` — inputs and artifacts (git-ignored except `SCA_data/`)

| Path | Contents |
|------|----------|
| `SCA_data/` | Input PDB structures (tracked — needed on the cluster). |
| `trained_ENMs/{protein}/{DDMmmYYYY}_{solver}/` | Training outputs: `*losses*.json` histories, `{protein}_stiffnesses_{seed}_{fn}_reduced.npy`. Also `good_seeds_*.json` cutoff files. |
| `hessians_cache/{cost_spectral,elastic_spectral,displacement_overlap}/` | Precomputed spectra / projections the notebooks read (global recompute-vs-load toggle). |
| `alphafold_mutants/PDZ_PSD95/` | Per-mutant `.npz` (AlphaFold mutant structures / derived data). |
| `ENM_statistics/` | Aggregated trained-ENM statistics. |

## Supporting directories

- **`plots/`** — all generated figures, one subdir per analysis theme
  (`training_diagnostics/`, `elastic_spectral/`, `susceptibility_costhessian/`,
  …). Fully git-ignored and regenerable. `base/plots/` holds a few
  mutation-strain / susceptibility panels written by base-level scripts.
- **`docs/`** — design notes and methodology: plans (`marcelo_BigPlan.md`),
  mapping/indexing notes (`ENM_evCouplings_mapping_plan.md`,
  `evCouplings_ENM_indexing_notes.md`), procedures (`enm_trimming_procedure.md`),
  candidate lists, and `repo_inventory_2026-07-22.md` (detailed file-by-file inventory).
- **`scripts/trained_enms_sync/`** — cluster⇄local operational tooling:
  `sync.py`, `remote_discover.py`, `remote_stat.py`, `sync_lr_sweep.py`,
  `push_lr_recommendations.py`. Not part of the science pipeline.
- **`benchmarks/`** — self-contained solver / performance benchmarking
  (`benchmark_*.py`, `profile_solvers.py`, result JSONs, `optimization_log.md`).
- **`Leibler_analysis_plot_notebooks/`** — external reference notebooks (ATP
  synthase strain analysis, proFSA-based). Kept locally for methodology
  reference; git-ignored, not wired into this repo's pipeline.

## Environment

`environment.yml` — conda env `protein_enm_analysis` (Python 3.11, NumPy/SciPy
stack, NetworkX, BioPython, `jax[cpu]`, plotly, evcouplings, py3Dmol).
