# PIBO-ReaxFF (MoS₂) — reproducibility repository

Physics-Informed Bayesian Optimization (**PIBO**) for ReaxFF parameter
calibration of MoS₂, together with the uncertainty-quantification (UQ)
analysis reported in the manuscript's reviewer response.

This repository is **reproducibility-oriented**: it contains the PIBO
optimization framework and the UQ analysis pipeline so the figures and
tables in the paper can be regenerated from the released artifacts. It is
**not** a turnkey environment for re-running the LAMMPS validation suite
(stress–strain, vacancy formation, S-diffusion, etc.); that part of the
workflow is deliberately excluded (see *Scope* below).

---

## Folder description

```
pibo_reaxff/      PIBO Bayesian-optimization framework (the method)
  optimizers/     PIBO + baselines (PSO, CMA-ES, INDEEDopt, JAX-ReaxFF)
  core/           GP surrogate, blocked / staged PIBO
  data/           VASP/OUTCAR readers + preprocessor
src/              UQ reviewer-response analysis modules
  gp_utils.py     GP surrogate IO + finite-difference grad/Hessian
  hmc.py          NUTS/HMC posterior over the GP-surrogate likelihood
  sloppy.py       Sethna-style sloppy spectrum + profile likelihood
  uncertainty.py  σ_opt / σ_GP / σ_post comparison
  ffield_parse.py pure ReaxFF text parser (no LAMMPS)  ← see Notes
  plotting.py     shared figure styling
scripts/          one-shot drivers
  phase01..08     the UQ pipeline (run in order)
  build_*.py      regenerate data/ inputs (need raw logs, see Notes)
configs/          run + base configs for the BO framework
data/             pre-built UQ inputs (GP surrogate + replicate table + bounds)
results/ffield/   calibrated ReaxFF force fields (*.reax)
results/reviewer_response/recalib_staged_bo/  minimal staged-BO inputs the UQ pipeline reads
vasp_calculations/  DFT reference frames (bond / angle / torsion / nonbond)
lammps_templates/ ReaxFF ffield templates (no LAMMPS data decks)
run.py, main.py   BO framework entry points
```

## Install

```bash
python -m pip install -e .            # core: numpy, scipy, pandas, matplotlib, pyyaml, tqdm
python -m pip install numpyro arviz   # required for phase04 HMC (jax backend)
python -m pip install scikit-learn    # required by the GP surrogate / src.gp_utils
```

Python ≥ 3.11. The BO framework additionally needs a LAMMPS `lmp`
executable on `PATH` (ReaxFF-enabled) — install separately via
`conda install -c conda-forge lammps`. The UQ pipeline does **not** need
LAMMPS.

## Short PIBO run

From the repository root, after installing the package:

```powershell
python run.py --check-only
python run.py --budget 20 --replicates 1 --optimizers pibo --output results/quick_pibo
```

The run writes `results/quick_pibo/benchmark_per_run.csv`, aggregate results,
and posterior parameter reports. `run.py` accepts a ReaxFF-enabled LAMMPS
executable on `PATH`, or an explicit path in `LAMMPS_BIN` (PowerShell:
`$env:LAMMPS_BIN = 'C:\path\to\lmp.exe'`). The separate recalibration
drivers use `PIBO_LMP`.

The released repository does not include `scripts/notes1_param_diff.json`.
When it is absent, the BO framework uses the 46 released bounds in
`configs/ezff_variable_bounds.txt` and their midpoints as prior centers.
MoS2 reference-value uncertainty comparisons are unavailable in that mode.

## Short cobalt run

The copied JAX-ReaxFF cobalt dataset is in `data/cobalt/`. It uses a `geo`
geometry file and `trainset.in` energy equations, so it has a separate PIBO
driver from the MoS2 VASP workflow:

```powershell
python pibo_calculation/cobalt_pibo.py --check-only
python pibo_calculation/cobalt_pibo.py --budget 12 --seed 42
```

For a short run, this driver uses the 18 cluster and dimer energy equations
and their 19 cobalt geometries. It optimizes the 12 parameters listed in
`data/cobalt/params`, starting from `ffield_lit`. The loss is the RMS of each
energy residual divided by its training-set tolerance. It uses PIBO's staged
GP acquisition and the supplied parameter bounds; MoS2-specific physical
rules are not applied. Outputs are `results/cobalt_quick_pibo/summary.json`,
`history.csv`, and `ffield_best`. The full cobalt training set includes
additional structures and targets that this short driver does not evaluate.

## Re-calibrating ReaxFF parameters

Two calibration routes produce a force field. **Both require a
ReaxFF-enabled LAMMPS** — set `PIBO_LMP` to the binary, or put `lmp` on
PATH (`conda install -c conda-forge lammps`):

```bash
# Windows
set PIBO_LMP=C:\path\to\lmp.exe
# Linux / macOS
export PIBO_LMP=/path/to/lmp
```
