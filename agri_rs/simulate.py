"""Satellite-style spectral series for one field.

The simulator knows the growth stage, because it built the season. The model
never receives that label. Indices follow the stage signature, blend across
the boundary with the next stage, then pick up sensor noise. Some passes are
dropped as cloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from agri_rs.states import FEATURES, CropModel


@dataclass(frozen=True)
class Pass:
    date: str
    values: np.ndarray | None
    stage: int
    cloudy: bool


@dataclass(frozen=True)
class Season:
    crop_id: str
    planting_date: str
    interval_days: int
    passes: tuple[Pass, ...]

    @property
    def clear_passes(self) -> tuple[Pass, ...]:
        return tuple(item for item in self.passes if not item.cloudy)

    def matrix(self) -> tuple[list[str], np.ndarray, list[int]]:
        clear = self.clear_passes
        dates = [item.date for item in clear]
        values = np.vstack([item.values for item in clear])
        stages = [item.stage for item in clear]
        return dates, values, stages


def simulate_field(
    crop: CropModel,
    planting_date: str | None = None,
    interval_days: int = 5,
    cloud_prob: float = 0.18,
    noise_scale: float = 1.0,
    apply_district: bool = True,
    seed: int | None = None,
) -> Season:
    if interval_days < 1:
        raise ValueError("interval_days must be at least 1")
    if not 0.0 <= cloud_prob < 0.85:
        raise ValueError("cloud_prob must be between 0 and 0.85")
    rng = np.random.default_rng(seed)
    start = datetime.strptime(planting_date or crop.default_planting, "%Y-%m-%d")
    means = crop.means + (crop.district_offset if apply_district else 0.0)
    steps = [max(1, int(round(days / interval_days))) for days in crop.stage_days]
    passes: list[Pass] = []
    day_offset = 0
    for stage, n_steps in enumerate(steps):
        for step in range(n_steps):
            progress = step / max(n_steps - 1, 1)
            signature = means[stage].copy()
            if stage < crop.n_states - 1 and progress > 0.70:
                blend = (progress - 0.70) / 0.30 * 0.55
                signature = (1.0 - blend) * means[stage] + blend * means[stage + 1]
            scale = np.sqrt(crop.variances[stage]) * noise_scale
            reading = np.clip(rng.normal(signature, scale), -0.08, 0.98)
            cloudy = bool(rng.random() < cloud_prob)
            current = start + timedelta(days=day_offset)
            passes.append(
                Pass(
                    date=current.strftime("%Y-%m-%d"),
                    values=None if cloudy else reading,
                    stage=stage,
                    cloudy=cloudy,
                )
            )
            day_offset += interval_days

    clear_count = sum(not item.cloudy for item in passes)
    if clear_count < 8:
        # A monitoring season with almost no clear passes is not a useful demo.
        restored = 0
        for index, item in enumerate(passes):
            if not item.cloudy:
                continue
            signature = means[item.stage]
            scale = np.sqrt(crop.variances[item.stage]) * noise_scale
            reading = np.clip(rng.normal(signature, scale), -0.08, 0.98)
            passes[index] = Pass(item.date, reading, item.stage, cloudy=False)
            restored += 1
            if clear_count + restored >= 8:
                break

    return Season(
        crop_id=crop.id,
        planting_date=start.strftime("%Y-%m-%d"),
        interval_days=interval_days,
        passes=tuple(passes),
    )


def season_to_records(season: Season, include_labels: bool) -> list[dict]:
    records = []
    for item in season.clear_passes:
        assert item.values is not None
        row = {"date": item.date}
        for name, value in zip(FEATURES, item.values):
            row[name] = round(float(value), 4)
        if include_labels:
            row["simulator_stage"] = int(item.stage)
        records.append(row)
    return records
