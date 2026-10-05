"""Command line for the same model the web app runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agri_rs.service import calibrate, infer, simulate
from agri_rs.states import CROPS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Infer crop growth stages from spectral indices, without a field visit."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="Simulate a satellite season and infer its stages")
    _add_crop(sim)
    sim.add_argument("--planting", default=None, help="Planting date YYYY-MM-DD")
    sim.add_argument("--interval", type=int, default=5, help="Days between satellite passes")
    sim.add_argument("--cloud", type=float, default=0.18, help="Chance a pass is dropped as cloud")
    sim.add_argument("--seed", type=int, default=7)
    sim.add_argument("--learn", action="store_true", help="Adapt the signature on unlabeled district fields first")
    sim.add_argument("--json", action="store_true", dest="as_json")

    inf = sub.add_parser("infer", help="Infer stages from a CSV (date,ndvi and optional evi,ndre)")
    inf.add_argument("csv", type=Path)
    _add_crop(inf)
    inf.add_argument("--learn", action="store_true", help="Adapt the textbook signature toward this series")
    inf.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)
    if args.command == "simulate":
        payload = simulate(
            args.crop,
            planting_date=args.planting,
            interval_days=args.interval,
            cloud_prob=args.cloud,
            seed=args.seed,
        )
        result = _maybe_learn(args.crop, payload["observations"], "simulator", args.learn, args.seed)
    else:
        observations = _read_csv(args.csv)
        result = _maybe_learn(args.crop, observations, "upload", args.learn, 7)

    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        _print_result(result)
    return 0


def _add_crop(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--crop", default="maize", choices=sorted(CROPS))


def _maybe_learn(crop: str, observations: list[dict], source: str, learn: bool, seed: int) -> dict:
    if not learn:
        return infer(crop, observations)
    trained = calibrate(crop, observations, source=source, seed=seed)
    print(trained["note"])
    if trained["agreement_prior"] is not None:
        print(
            "Simulator agreement "
            f"{trained['agreement_prior']:.0%} textbook -> {trained['agreement_adapted']:.0%} adapted"
        )
        print()
    return trained["adapted_inference"]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} has no header row.")
        rows = []
        for raw in reader:
            rows.append(
                {
                    (key or "").strip().lower(): (value or "").strip()
                    for key, value in raw.items()
                }
            )
    if not rows:
        raise SystemExit(f"{path} has no data rows.")
    return rows


def _print_result(result: dict) -> None:
    names = {stage["id"]: stage["name"] for stage in result["stages"]}
    print(f"{result['model_label']} · log-likelihood {result['log_likelihood']}")
    if result["agreement"] is not None:
        print(f"Agreement with simulator stages: {result['agreement']:.0%}")
    if result["warning"]:
        print(result["warning"])
    print(f"{'Date':<12}{'NDVI':>8}{'EVI':>8}{'NDRE':>8}  {'Stage':<16}Belief")
    for row in result["timeline"]:
        stage = names[row["stage_viterbi"]]
        print(
            f"{row['date']:<12}"
            f"{_num(row['ndvi']):>8}{_num(row['evi']):>8}{_num(row['ndre']):>8}  "
            f"{stage:<16}{row['confidence_viterbi']:>5.0%}"
        )
    latest = result["latest"]
    print()
    print(
        f"Latest clear pass {latest['date']}: {names[latest['stage_filtered']]} "
        f"({latest['confidence_filtered']:.0%} from passes available that day)"
    )
    peak = result["peak"]
    print(f"Peak NDVI {peak['value']:.2f} on {peak['date']}")
    if result["senescence_onset"]:
        print(f"Senescence enters the reconstructed season on {result['senescence_onset']}")


def _num(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
