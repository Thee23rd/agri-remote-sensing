"""Inference and unsupervised adaptation used by the CLI and the web app."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from agri_rs.hmm import GaussianHMM
from agri_rs.simulate import season_to_records, simulate_field
from agri_rs.states import FEATURES, CROPS, CropModel, get_crop

_INDEX_MIN = -0.2
_INDEX_MAX = 1.0


def list_crops() -> list[dict]:
    return [_crop_public(crop) for crop in CROPS.values()]


def model_from_crop(crop: CropModel, parameters: dict | None = None) -> GaussianHMM:
    if parameters is None:
        return GaussianHMM(crop.means, crop.variances, crop.start, crop.transition)
    means = _as_shaped(parameters.get("means"), (crop.n_states, len(FEATURES)), "means")
    variances = _as_shaped(
        parameters.get("variances"), (crop.n_states, len(FEATURES)), "variances"
    )
    start = _as_shaped(parameters.get("start"), (crop.n_states,), "start")
    transition = _as_shaped(
        parameters.get("transition"), (crop.n_states, crop.n_states), "transition"
    )
    if np.any(variances <= 0):
        raise ValueError("variances must be positive")
    if np.any(start < 0) or np.any(transition < 0):
        raise ValueError("probabilities cannot be negative")
    return GaussianHMM(means, variances, start, transition)


def simulate(
    crop_id: str,
    planting_date: str | None = None,
    interval_days: int = 5,
    cloud_prob: float = 0.18,
    seed: int | None = None,
) -> dict:
    crop = get_crop(crop_id)
    season = simulate_field(
        crop,
        planting_date=planting_date,
        interval_days=interval_days,
        cloud_prob=cloud_prob,
        apply_district=True,
        seed=seed,
    )
    cloudy = sum(item.cloudy for item in season.passes)
    return {
        "crop": crop.id,
        "source": "simulator",
        "planting_date": season.planting_date,
        "interval_days": season.interval_days,
        "cloudy_passes": cloudy,
        "clear_passes": len(season.clear_passes),
        "observations": season_to_records(season, include_labels=True),
        "note": (
            "Clear passes only. The growth stage stored on each row is the simulator's "
            "label for checking the model. Inference does not read it."
        ),
    }


def infer(
    crop_id: str,
    observations: list[dict],
    parameters: dict | None = None,
    model_label: str | None = None,
) -> dict:
    crop = get_crop(crop_id)
    dates, matrix, labels = parse_observations(observations)
    hmm = model_from_crop(crop, parameters)
    decoded = hmm.viterbi(matrix)
    post = hmm.posteriors(matrix)
    smoothed = post["smoothed"]
    filtered = post["filtered"]
    timeline = []
    for index, date in enumerate(dates):
        viterbi_state = int(decoded[index])
        filtered_state = int(np.argmax(filtered[index]))
        row = {
            "date": date,
            "stage_viterbi": crop.stages[viterbi_state].id,
            "stage_filtered": crop.stages[filtered_state].id,
            "confidence_viterbi": round(float(smoothed[index, viterbi_state]), 4),
            "confidence_filtered": round(float(filtered[index, filtered_state]), 4),
            "posterior_smoothed": [round(float(value), 4) for value in smoothed[index]],
            "posterior_filtered": [round(float(value), 4) for value in filtered[index]],
        }
        for feature_index, feature in enumerate(FEATURES):
            value = matrix[index, feature_index]
            row[feature] = None if not np.isfinite(value) else round(float(value), 4)
        timeline.append(row)

    ndvi = matrix[:, 0]
    finite_ndvi = np.isfinite(ndvi)
    peak_index = int(np.argmax(np.where(finite_ndvi, ndvi, -np.inf)))
    senescence_id = crop.stages[-1].id
    senescence_onset = next(
        (row["date"] for row in timeline if row["stage_viterbi"] == senescence_id),
        None,
    )
    warning = None
    if len(dates) < 6:
        warning = "Short series: stage changes will stay uncertain until more clear passes arrive."

    agreement = None
    if labels is not None:
        agreement = round(float(np.mean(decoded == np.asarray(labels))), 4)

    return {
        "crop": crop.id,
        "model_label": model_label or ("Adapted signature" if parameters else "Textbook prior"),
        "features": list(FEATURES),
        "stages": [_stage_public(stage) for stage in crop.stages],
        "clear_passes": len(dates),
        "warning": warning,
        "log_likelihood": round(float(post["log_likelihood"]), 3),
        "agreement": agreement,
        "peak": {
            "feature": "ndvi",
            "value": round(float(ndvi[peak_index]), 4),
            "date": dates[peak_index],
        },
        "senescence_onset": senescence_onset,
        "latest": timeline[-1],
        "timeline": timeline,
        "means": [[round(float(value), 4) for value in row] for row in hmm.means],
        "transition": [[round(float(value), 4) for value in row] for row in hmm.transition],
    }


def calibrate(
    crop_id: str,
    observations: list[dict],
    source: str = "simulator",
    n_fields: int = 28,
    seed: int = 7,
) -> dict:
    """Update spectral signatures from unlabeled passes, then re-score the field.

    Simulator fields belong to a district whose canopy sits off the textbook
    signature. Learning uses other unlabeled fields from that district, and the
    field being viewed is held out. An uploaded series is adapted on its own,
    with a strong pull back toward the textbook signature.
    """
    crop = get_crop(crop_id)
    if source not in {"simulator", "upload"}:
        raise ValueError("source must be 'simulator' or 'upload'")
    if not 4 <= n_fields <= 80:
        raise ValueError("n_fields must be between 4 and 80")

    prior = model_from_crop(crop)
    dates, matrix, labels = parse_observations(observations)
    rng = np.random.default_rng(seed)
    if source == "simulator":
        sequences = []
        for _ in range(n_fields):
            season = simulate_field(
                crop,
                planting_date=crop.default_planting,
                interval_days=5,
                cloud_prob=0.12,
                noise_scale=0.9,
                apply_district=True,
                seed=int(rng.integers(0, 1_000_000_000)),
            )
            _, archive_matrix, _ = season.matrix()
            sequences.append(archive_matrix)
        note = (
            f"Learned from {n_fields} unlabeled district fields. This field was held out. "
            "No growth-stage labels were used."
        )
        mean_strength = 8.0
        transition_strength = 30.0
        iterations = 12
    else:
        sequences = [matrix]
        note = (
            "Adapted the textbook signature toward this uploaded series only. "
            "The prior stays strong so one field cannot rewrite the season."
        )
        mean_strength = 16.0
        transition_strength = 40.0
        iterations = 8

    adapted, trace = prior.fit(
        sequences,
        n_iter=iterations,
        mean_prior_strength=mean_strength,
        transition_prior_strength=transition_strength,
    )
    parameters = _parameters_payload(adapted)
    prior_inference = infer(crop_id, observations, parameters=None, model_label="Textbook prior")
    adapted_inference = infer(
        crop_id, observations, parameters=parameters, model_label="Adapted signature"
    )
    return {
        "crop": crop.id,
        "source": source,
        "note": note,
        "fields_used": n_fields if source == "simulator" else 1,
        "passes_held_out": len(dates),
        "log_likelihood_trace": [round(value, 3) for value in trace],
        "prior_means": [[round(float(value), 4) for value in row] for row in prior.means],
        "adapted_means": parameters["means"],
        "parameters": parameters,
        "prior_inference": prior_inference,
        "adapted_inference": adapted_inference,
        "agreement_prior": prior_inference["agreement"] if labels is not None else None,
        "agreement_adapted": adapted_inference["agreement"] if labels is not None else None,
    }


def parse_observations(observations: list[dict]) -> tuple[list[str], np.ndarray, list[int] | None]:
    if not observations:
        raise ValueError("Add at least one spectral pass.")
    grouped: dict[str, list[np.ndarray]] = {}
    labels: dict[str, int] = {}
    label_count = 0
    for row in observations:
        if "date" not in row:
            raise ValueError("Each pass needs a date.")
        try:
            parsed = datetime.strptime(str(row["date"])[:10], "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Could not read date '{row.get('date')}'. Use YYYY-MM-DD.") from exc
        date = parsed.strftime("%Y-%m-%d")
        values = []
        for feature in FEATURES:
            raw = row.get(feature)
            if raw is None or raw == "":
                values.append(np.nan)
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{feature} on {date} is not a number.") from exc
            if not _INDEX_MIN <= number <= _INDEX_MAX:
                raise ValueError(
                    f"{feature} on {date} is {number:.3f}. Indices need to sit between {_INDEX_MIN} and {_INDEX_MAX}."
                )
            values.append(number)
        vector = np.asarray(values, dtype=float)
        if not np.isfinite(vector).any():
            continue
        grouped.setdefault(date, []).append(vector)
        if row.get("simulator_stage") is not None:
            labels[date] = int(row["simulator_stage"])
            label_count += 1

    if not grouped:
        raise ValueError("Every pass is empty. Include NDVI, EVI, or NDRE.")

    dates = sorted(grouped)
    matrix = []
    for date in dates:
        stacked = np.vstack(grouped[date])
        averaged = np.full(stacked.shape[1], np.nan)
        for feature_index in range(stacked.shape[1]):
            finite = stacked[np.isfinite(stacked[:, feature_index]), feature_index]
            if finite.size:
                averaged[feature_index] = finite.mean()
        matrix.append(averaged)
    array = np.vstack(matrix)
    stage_labels = None
    if label_count == len(observations) and all(date in labels for date in dates):
        stage_labels = [labels[date] for date in dates]
    return dates, array, stage_labels


def _parameters_payload(model: GaussianHMM) -> dict:
    return {
        "means": [[round(float(value), 4) for value in row] for row in model.means],
        "variances": [[round(float(value), 6) for value in row] for row in model.variances],
        "start": [round(float(value), 4) for value in model.start],
        "transition": [[round(float(value), 4) for value in row] for row in model.transition],
    }


def _as_shaped(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Adapted model is missing {name}.")
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-numeric value.")
    return array


def _crop_public(crop: CropModel) -> dict:
    return {
        "id": crop.id,
        "name": crop.name,
        "default_planting": crop.default_planting,
        "planting_note": crop.planting_note,
        "features": [
            {"id": "ndvi", "name": "NDVI", "detail": "Green cover. Rises as the canopy closes and saturates when cover is dense."},
            {"id": "evi", "name": "EVI", "detail": "Dense canopy, with less saturation than NDVI at peak cover."},
            {"id": "ndre", "name": "NDRE", "detail": "Chlorophyll. Often turns down while NDVI is still high, which separates flowering from grain fill."},
        ],
        "stages": [_stage_public(stage) for stage in crop.stages],
        "means": [[round(float(value), 4) for value in row] for row in crop.means],
    }


def _stage_public(stage) -> dict:
    return {
        "id": stage.id,
        "name": stage.name,
        "short": stage.short,
        "summary": stage.summary,
        "color": stage.color,
    }
