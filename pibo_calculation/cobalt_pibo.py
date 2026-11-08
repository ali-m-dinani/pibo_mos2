"""Short PIBO calibration against the cobalt cluster and dimer energy targets."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from run_budget import Budget

from pibo_reaxff.lammps_runner import KCAL_PER_MOL_TO_EV, LAMMPSRunner
from pibo_reaxff.optimizers.pibo import PIBOOptimizer
from pibo_reaxff.parameters import ParameterSpec
from pibo_reaxff.vasp_reader import DFTFrame


REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "data" / "cobalt"
DEFAULT_OUTPUT = REPO / "results" / "cobalt_quick_pibo"
TRAINING_GROUPS = {"Co_clusters", "Co2_cluster"}


@dataclass(frozen=True)
class EnergyTarget:
    tolerance: float
    first: str
    first_divisor: float
    second: str
    second_divisor: float
    target_kcal: float


@dataclass(frozen=True)
class Variable:
    section: int
    column: int
    lower: float
    upper: float
    line_index: int
    token_index: int
    initial: float

    @property
    def name(self) -> str:
        return f"Co_section{self.section}_column{self.column}"


def read_targets(path: Path) -> list[EnergyTarget]:
    targets = []
    group = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# General "):
            group = line.split(maxsplit=2)[-1]
            continue
        if line.startswith("# Volume ") or line.startswith("# General "):
            group = None
            continue
        if group not in TRAINING_GROUPS or not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 6 or fields[1] != "+" or fields[3] != "-":
            raise ValueError(f"Unsupported cobalt energy target: {line}")
        first, first_divisor = fields[2].rsplit("/", 1)
        second, second_divisor = fields[4].rsplit("/", 1)
        targets.append(EnergyTarget(
            float(fields[0]), first, float(first_divisor),
            second, float(second_divisor), float(fields[5])))
    if not targets:
        raise ValueError(f"No cobalt cluster or dimer energy targets in {path}")
    return targets


def read_frames(path: Path, names: set[str]) -> dict[str, DFTFrame]:
    frames = {}
    name = None
    lengths = None
    positions = []
    species = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("DESCRP "):
            name = line.split(maxsplit=1)[1].strip()
            lengths = None
            positions = []
            species = []
        elif name in names and line.startswith("CRYSTX "):
            values = [float(value) for value in line.split()[1:7]]
            if len(values) != 6 or not np.allclose(values[3:], 90):
                raise ValueError(f"Only orthogonal cobalt cells are supported: {name}")
            lengths = values[:3]
        elif name in names and line.startswith("HETATM"):
            fields = line.split()
            species.append(fields[2])
            positions.append([float(value) for value in fields[3:6]])
        elif line.strip() == "END" and name in names:
            if not positions or set(species) != {"Co"}:
                raise ValueError(f"Invalid cobalt geometry: {name}")
            if lengths is None:
                box_length = max(80.0, float(np.max(positions)) + 20.0)
                lengths = [box_length] * 3
            frames[name] = DFTFrame(
                tag=name, energy=0.0, positions=np.asarray(positions),
                cell=np.diag(lengths), species=species)
            name = None
    missing = names - frames.keys()
    if missing:
        raise ValueError(f"Geometries missing from {path}: {sorted(missing)}")
    return frames


def read_variables(path: Path, ffield_lines: list[str]) -> list[Variable]:
    atom_heading = next(index for index, line in enumerate(ffield_lines)
                        if "Nr of atoms" in line)
    bond_heading = next(index for index, line in enumerate(ffield_lines)
                        if "Nr of bonds" in line)
    section_starts = {2: atom_heading + 4, 3: bond_heading + 2}
    variables = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        section, row, column, _, lower, upper = line.split()
        section, row, column = int(section), int(row), int(column)
        if section not in section_starts or row != 1 or column < 1:
            raise ValueError(f"Unsupported cobalt parameter location: {line}")
        row_offset, value_offset = divmod(column - 1, 8)
        line_index = section_starts[section] + row_offset
        token_index = value_offset + ({2: 1, 3: 2}[section] if row_offset == 0 else 0)
        tokens = ffield_lines[line_index].split("!", 1)[0].split()
        initial = float(tokens[token_index])
        variables.append(Variable(
            section, column, float(lower), float(upper),
            line_index, token_index, initial))
    if not variables:
        raise ValueError(f"No tunable cobalt parameters in {path}")
    return variables


def render_ffield(lines: list[str], variables: list[Variable], values: np.ndarray) -> str:
    rendered = lines.copy()
    by_line: dict[int, list[tuple[int, float]]] = {}
    for variable, value in zip(variables, values):
        by_line.setdefault(variable.line_index, []).append((variable.token_index, float(value)))
    for line_index, replacements in by_line.items():
        tokens = rendered[line_index].split("!", 1)[0].split()
        for token_index, value in replacements:
            tokens[token_index] = f"{value:.6f}"
        rendered[line_index] = "  ".join(tokens) + "\n"
    return "".join(rendered)


class CobaltLoss:
    def __init__(self, runner: LAMMPSRunner, ffield_path: Path,
                 lines: list[str], variables: list[Variable],
                 frames: dict[str, DFTFrame], targets: list[EnergyTarget]):
        self.runner = runner
        self.ffield_path = ffield_path
        self.lines = lines
        self.variables = variables
        self.frames = frames
        self.targets = targets
        self.history = []

    def __call__(self, values: np.ndarray) -> float:
        self.ffield_path.write_text(
            render_ffield(self.lines, self.variables, values), encoding="utf-8")
        try:
            energies = {
                name: self.runner.evaluate({}, frame).energy / KCAL_PER_MOL_TO_EV
                for name, frame in self.frames.items()
            }
            residuals = [
                (energies[target.first] / target.first_divisor
                 - energies[target.second] / target.second_divisor
                 - target.target_kcal) / target.tolerance
                for target in self.targets
            ]
            loss = float(np.sqrt(np.mean(np.square(residuals))))
            if not np.isfinite(loss):
                raise ValueError("Non-finite cobalt loss")
            error = ""
        except (RuntimeError, ValueError, OSError) as exc:
            loss = 1e6
            error = str(exc)
            print(f"  [cobalt evaluation failed] {error}", flush=True)
        self.history.append({
            "evaluation": len(self.history) + 1, "loss": loss,
            "error": error,
            **{variable.name: float(value)
               for variable, value in zip(self.variables, values)},
        })
        print(f"  [cobalt {len(self.history)}] loss={loss:.6f}", flush=True)
        return loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--time-limit-seconds", type=float)
    args = parser.parse_args()
    clock = None if args.check_only else Budget(args.output, args.time_limit_seconds)
    if args.budget < 5:
        parser.error("--budget must be at least 5 for the PIBO warm start")

    targets = read_targets(args.data_root / "trainset.in")
    required = {name for target in targets for name in (target.first, target.second)}
    frames = read_frames(args.data_root / "geo", required)
    ffield_lines = (args.data_root / "ffield_lit").read_text(
        encoding="utf-8").splitlines(keepends=True)
    variables = read_variables(args.data_root / "params", ffield_lines)
    specs = [ParameterSpec(variable.name, variable.lower, variable.upper,
                           "cobalt", variable.initial, float("nan"))
             for variable in variables]
    initial = np.asarray([variable.initial for variable in variables])
    print(f"Cobalt: {len(frames)} structures, {len(targets)} energy targets, "
          f"{len(variables)} tunable parameters", flush=True)

    with tempfile.TemporaryDirectory(prefix="cobalt_pibo_") as scratch:
        ffield_path = Path(scratch) / "ffield.reax.Co"
        ffield_path.write_text(
            render_ffield(ffield_lines, variables, initial), encoding="utf-8")
        runner = LAMMPSRunner(elements=("Co",), base_ffield=str(ffield_path))
        try:
            objective = CobaltLoss(
                runner, ffield_path, ffield_lines, variables, frames, targets)
            baseline = objective(initial)
            if baseline >= 1e6:
                raise RuntimeError("The literature cobalt force field failed its baseline evaluation")
            print(f"Literature baseline loss: {baseline:.6f}", flush=True)
            if args.check_only:
                return 0
            objective.history.clear()
            args.output.mkdir(parents=True, exist_ok=True)
            best_seen = [float("inf")]

            def save_best(params, value):
                if not np.isfinite(value) or value >= best_seen[0]:
                    return
                temporary = args.output / "ffield_best.tmp"
                temporary.write_text(
                    render_ffield(ffield_lines, variables, params),
                    encoding="utf-8")
                temporary.replace(args.output / "ffield_best")
                best_seen[0] = value

            objective.checkpoint_callback = save_best
            original_objective = objective

            def timed_objective(params):
                value = original_objective(params)
                temporary = args.output / "history.csv.tmp"
                with temporary.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(original_objective.history[0]))
                    writer.writeheader()
                    writer.writerows(original_objective.history)
                temporary.replace(args.output / "history.csv")
                clock.checkpoint(value)
                return value

            timed_objective.checkpoint_callback = save_best
            clock.begin()
            optimizer = PIBOOptimizer(
                physics_informed=False,
                config={"init_lhs_points": 5, "n_acq_candidates": 256})
            result = optimizer.optimize(
                loss=timed_objective, specs=specs, budget=args.budget,
                rng=np.random.default_rng(args.seed), should_stop=clock.should_stop)
        finally:
            runner.close()

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(objective.history[0]))
        writer.writeheader()
        writer.writerows(objective.history)
    (args.output / "ffield_best").write_text(
        render_ffield(ffield_lines, variables, result.best_x), encoding="utf-8")
    summary = {
        "dataset": str(args.data_root.resolve()),
        "training_groups": sorted(TRAINING_GROUPS),
        "n_structures": len(frames), "n_targets": len(targets),
        "n_parameters": len(variables), "baseline_loss": baseline,
        "best_loss": result.best_loss, "n_evals": result.n_evals,
        "wall_clock_s": result.wall_clock_s, "seed": args.seed,
        "loss_definition": "RMS of (predicted-target)/tolerance across selected energy targets",
        "best_parameters": {variable.name: float(value)
                            for variable, value in zip(variables, result.best_x)},
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Best cobalt loss: {result.best_loss:.6f} "
          f"after {result.n_evals} evaluations; results: {args.output}", flush=True)
    clock.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
