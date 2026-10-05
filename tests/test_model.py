import numpy as np

from agri_rs.hmm import GaussianHMM, logsumexp
from agri_rs.service import calibrate, infer, model_from_crop, parse_observations
from agri_rs.simulate import simulate_field
from agri_rs.states import get_crop


def test_logsumexp_matches_a_direct_sum():
    values = np.array([0.2, -1.0, 3.0, -4.0])
    direct = np.log(np.sum(np.exp(values)))
    assert np.isclose(float(logsumexp(values)), direct)


def test_posteriors_are_distributions_and_viterbi_recovers_separated_states():
    means = np.array([[0.0], [4.0], [8.0]])
    variances = np.full((3, 1), 0.04)
    start = np.array([1.0, 0.0, 0.0])
    transition = np.array(
        [
            [0.55, 0.45, 0.0],
            [0.0, 0.55, 0.45],
            [0.0, 0.0, 1.0],
        ]
    )
    hmm = GaussianHMM(means, variances, start, transition)
    rng = np.random.default_rng(2)
    scores = []
    for _ in range(15):
        observations, states = hmm.sample(28, rng)
        observations[4, 0] = np.nan
        # One missing band in a 1-feature model is an empty timestep. Restore it.
        observations[4, 0] = states[4] * 4.0
        post = hmm.posteriors(observations)
        assert np.allclose(post["smoothed"].sum(axis=1), 1.0)
        assert np.allclose(post["filtered"].sum(axis=1), 1.0)
        scores.append(np.mean(hmm.viterbi(observations) == states))
    assert np.mean(scores) > 0.9


def test_baum_welch_raises_likelihood_on_data_from_the_model():
    means = np.array([[0.1, 0.05], [0.7, 0.4], [0.3, 0.15]])
    variances = np.full((3, 2), 0.004)
    start = np.array([0.8, 0.2, 0.0])
    transition = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.0, 0.75, 0.25],
            [0.15, 0.0, 0.85],
        ]
    )
    hmm = GaussianHMM(means, variances, start, transition)
    rng = np.random.default_rng(5)
    sequences = [hmm.sample(36, rng)[0] for _ in range(12)]
    # Start away from the true means so learning has work to do.
    shifted = hmm.copy()
    shifted.means = means + np.array([[0.05, 0.04], [-0.08, -0.06], [0.06, 0.05]])
    _, trace = shifted.fit(
        sequences,
        n_iter=8,
        mean_prior_strength=0.0,
        transition_prior_strength=0.0,
    )
    assert trace[-1] > trace[0]


def test_textbook_prior_reads_a_season_it_matches():
    crop = get_crop("maize")
    season = simulate_field(crop, apply_district=False, cloud_prob=0.0, noise_scale=0.7, seed=3)
    _, matrix, labels = season.matrix()
    path = model_from_crop(crop).viterbi(matrix)
    assert np.mean(path == np.asarray(labels)) >= 0.75


def test_unlabeled_district_learning_moves_the_signature():
    crop = get_crop("maize")
    from agri_rs.simulate import season_to_records

    prior_scores = []
    adapted_scores = []
    flowering = 3
    for seed in (4, 11, 19, 27):
        season = simulate_field(crop, apply_district=True, cloud_prob=0.1, seed=seed)
        records = season_to_records(season, include_labels=True)
        trained = calibrate("maize", records, source="simulator", n_fields=24, seed=seed)
        prior_scores.append(trained["agreement_prior"])
        adapted_scores.append(trained["agreement_adapted"])
        textbook = crop.means[flowering, 0]
        target = textbook + crop.district_offset[flowering, 0]
        adapted = trained["adapted_means"][flowering][0]
        assert abs(adapted - target) < abs(textbook - target)
    assert np.mean(adapted_scores) + 0.02 >= np.mean(prior_scores)


def test_a_missing_index_keeps_the_pass():
    crop = get_crop("maize")
    season = simulate_field(crop, apply_district=False, cloud_prob=0.0, seed=1)
    _, matrix, _ = season.matrix()
    matrix = matrix.copy()
    matrix[3, 1] = np.nan
    matrix[3, 2] = np.nan
    smoothed = model_from_crop(crop).posteriors(matrix)["smoothed"]
    assert smoothed.shape == (matrix.shape[0], crop.n_states)
    assert np.allclose(smoothed.sum(axis=1), 1.0)


def test_parse_sorts_dates_and_ignores_stage_labels_in_the_matrix():
    dates, matrix, labels = parse_observations(
        [
            {"date": "2025-12-02", "ndvi": 0.4, "evi": "", "ndre": 0.2, "simulator_stage": 2},
            {"date": "2025-11-20", "ndvi": 0.16, "evi": 0.09, "ndre": 0.05, "simulator_stage": 0},
        ]
    )
    assert dates == ["2025-11-20", "2025-12-02"]
    assert np.isnan(matrix[1, 1])
    assert labels == [0, 2]
    result = infer("maize", [
        {"date": "2025-11-20", "ndvi": 0.16, "evi": 0.09, "ndre": 0.05, "simulator_stage": 0},
        {"date": "2025-12-02", "ndvi": 0.62, "evi": 0.44, "ndre": 0.36, "simulator_stage": 2},
    ])
    assert result["timeline"][0]["date"] == "2025-11-20"
    assert result["agreement"] is not None
