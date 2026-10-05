"""Growth-stage hidden states and textbook spectral signatures.

Observations are canopy indices a satellite can measure without a field visit:
NDVI (green cover), EVI (dense canopy, less saturation), and NDRE (chlorophyll,
which often declines before NDVI does).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = ("ndvi", "evi", "ndre")

# Shared observation noise. Flowering and grain fill overlap on NDVI on purpose;
# NDRE is what separates them.
_STD = np.array([0.040, 0.034, 0.030], dtype=float)


@dataclass(frozen=True)
class Stage:
    id: str
    name: str
    short: str
    summary: str
    color: str


@dataclass(frozen=True)
class CropModel:
    id: str
    name: str
    stages: tuple[Stage, ...]
    means: np.ndarray
    variances: np.ndarray
    start: np.ndarray
    transition: np.ndarray
    stage_days: tuple[int, ...]
    district_offset: np.ndarray
    default_planting: str
    planting_note: str

    @property
    def n_states(self) -> int:
        return len(self.stages)


def _left_to_right(n: int) -> np.ndarray:
    """Phenology moves forward. A stage usually persists, sometimes advances, rarely skips."""
    trans = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        trans[i, i] = 0.84
        trans[i, i + 1] = 0.14
        if i + 2 < n:
            trans[i, i + 2] = 0.02
        else:
            trans[i, i + 1] += 0.02
    trans[n - 1, n - 1] = 0.90
    trans[n - 1, 0] = 0.10  # harvest returns the pixel to bare soil
    trans /= trans.sum(axis=1, keepdims=True)
    return trans


def transition_mask(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        mask[i, i] = 1.0
        mask[i, i + 1] = 1.0
        if i + 2 < n:
            mask[i, i + 2] = 1.0
    mask[n - 1, n - 1] = 1.0
    mask[n - 1, 0] = 1.0
    return mask


def _variances(n: int) -> np.ndarray:
    return np.tile(_STD**2, (n, 1))


def _start(n: int) -> np.ndarray:
    raw = np.array([0.62, 0.28, 0.08, 0.015, 0.004, 0.001], dtype=float)
    if n != raw.shape[0]:
        raise ValueError(f"Expected {raw.shape[0]} stages, got {n}")
    return raw / raw.sum()


def _crop(
    crop_id: str,
    name: str,
    stages: tuple[Stage, ...],
    means: list[list[float]],
    stage_days: tuple[int, ...],
    district_offset: list[list[float]],
    default_planting: str,
    planting_note: str,
) -> CropModel:
    return CropModel(
        id=crop_id,
        name=name,
        stages=stages,
        means=np.asarray(means, dtype=float),
        variances=_variances(len(stages)),
        start=_start(len(stages)),
        transition=_left_to_right(len(stages)),
        stage_days=stage_days,
        district_offset=np.asarray(district_offset, dtype=float),
        default_planting=default_planting,
        planting_note=planting_note,
    )


_BARE = dict(id="bare", name="Bare soil", short="Bare", color="#b7a48a", summary="Soil and residue dominate the pixel. Canopy signal is near zero.")
_EMERGE = dict(
    id="emergence",
    name="Emergence",
    short="Emerge",
    color="#c6b44a",
    summary="Seedlings are up, but soil still drives most of the reflectance.",
)
_SENESC = dict(
    id="senescence",
    name="Senescence",
    short="Senesc",
    color="#b85c38",
    summary="Leaves lose greenness and the field moves toward harvest.",
)

CROPS: dict[str, CropModel] = {
    "maize": _crop(
        "maize",
        "Maize",
        (
            Stage(**_BARE),
            Stage(**_EMERGE),
            Stage(
                id="vegetative",
                name="Vegetative",
                short="Veg",
                color="#2f8a45",
                summary="Leaf area expands quickly and the canopy starts to close.",
            ),
            Stage(
                id="flowering",
                name="Flowering",
                short="Flower",
                color="#1d7a68",
                summary="Tasseling and silking. Cover and chlorophyll are at their peak.",
            ),
            Stage(
                id="grain_fill",
                name="Grain fill",
                short="Grain",
                color="#d4922a",
                summary="Kernels fill. Cover stays high while chlorophyll begins to slip.",
            ),
            Stage(**_SENESC),
        ),
        [
            [0.16, 0.09, 0.05],
            [0.30, 0.18, 0.16],
            [0.64, 0.45, 0.38],
            [0.84, 0.62, 0.50],
            [0.70, 0.50, 0.30],
            [0.38, 0.22, 0.12],
        ],
        (20, 18, 42, 16, 30, 26),
        # Local canopies peak lower than the textbook signature. Learning has something to find.
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.01, 0.02],
            [-0.05, -0.04, -0.03],
            [-0.08, -0.06, -0.06],
            [-0.05, -0.04, -0.05],
            [0.02, 0.01, 0.01],
        ],
        "2025-11-20",
        "Default planting sits at the start of the rainy season. Change the date to shift the calendar.",
    ),
    "wheat": _crop(
        "wheat",
        "Wheat",
        (
            Stage(**_BARE),
            Stage(**_EMERGE),
            Stage(
                id="tillering",
                name="Tillering",
                short="Tiller",
                color="#2f8a45",
                summary="Tillers build leaf area and the canopy begins to close.",
            ),
            Stage(
                id="heading",
                name="Heading",
                short="Head",
                color="#1d7a68",
                summary="Heads emerge. Green cover is near its maximum.",
            ),
            Stage(
                id="grain_fill",
                name="Grain fill",
                short="Grain",
                color="#d4922a",
                summary="Grain fills. Cover stays high while chlorophyll eases off.",
            ),
            Stage(**_SENESC),
        ),
        [
            [0.15, 0.08, 0.05],
            [0.28, 0.16, 0.15],
            [0.58, 0.40, 0.34],
            [0.78, 0.55, 0.46],
            [0.66, 0.46, 0.28],
            [0.34, 0.20, 0.11],
        ],
        (16, 20, 40, 14, 28, 22),
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.01, 0.015],
            [-0.04, -0.03, -0.03],
            [-0.07, -0.05, -0.05],
            [-0.04, -0.03, -0.04],
            [0.02, 0.01, 0.01],
        ],
        "2025-05-10",
        "Default planting is an irrigated cool-season window. Change the date if the season starts elsewhere.",
    ),
    "soybean": _crop(
        "soybean",
        "Soybean",
        (
            Stage(**_BARE),
            Stage(**_EMERGE),
            Stage(
                id="vegetative",
                name="Vegetative",
                short="Veg",
                color="#2f8a45",
                summary="Trifoliate leaves expand and the rows begin to close.",
            ),
            Stage(
                id="flowering",
                name="Flowering",
                short="Flower",
                color="#1d7a68",
                summary="Flowering through pod set. Canopy cover is at its peak.",
            ),
            Stage(
                id="pod_fill",
                name="Pod fill",
                short="Pods",
                color="#d4922a",
                summary="Pods fill. Cover stays high while chlorophyll starts to fall.",
            ),
            Stage(**_SENESC),
        ),
        [
            [0.15, 0.08, 0.05],
            [0.32, 0.20, 0.18],
            [0.70, 0.50, 0.42],
            [0.88, 0.66, 0.52],
            [0.72, 0.52, 0.31],
            [0.36, 0.21, 0.12],
        ],
        (14, 14, 30, 24, 28, 18),
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.015, 0.02],
            [-0.05, -0.04, -0.03],
            [-0.07, -0.05, -0.05],
            [-0.04, -0.03, -0.05],
            [0.02, 0.015, 0.01],
        ],
        "2025-12-05",
        "Default planting is early in the rainy season. Change the date to match the field.",
    ),
}


def get_crop(crop_id: str) -> CropModel:
    try:
        return CROPS[crop_id]
    except KeyError as exc:
        known = ", ".join(sorted(CROPS))
        raise KeyError(f"Unknown crop '{crop_id}'. Known crops: {known}.") from exc
