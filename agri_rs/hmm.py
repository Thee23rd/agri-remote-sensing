"""Diagonal-Gaussian hidden Markov model for a phenology sequence.

Hidden state z_t is the growth stage. Observation x_t is the spectral vector.
Missing indices (cloud-damaged bands) are ignored for that timestep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agri_rs.states import transition_mask


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maxes = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maxes)
    safe = np.where(finite, maxes, 0.0)
    summed = np.sum(np.exp(np.where(finite, values - safe, -np.inf)), axis=axis, keepdims=True)
    out = np.where(finite, safe + np.log(np.maximum(summed, 1e-300)), -np.inf)
    if axis is None:
        return out.reshape(())
    return np.squeeze(out, axis=axis)


def _log_prob(probs: np.ndarray) -> np.ndarray:
    values = np.asarray(probs, dtype=float)
    out = np.full(values.shape, -np.inf, dtype=float)
    positive = values > 0
    out[positive] = np.log(values[positive])
    return out


@dataclass
class GaussianHMM:
    """Left-to-right phenology model with independent Gaussian indices."""

    means: np.ndarray
    variances: np.ndarray
    start: np.ndarray
    transition: np.ndarray

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=float).copy()
        self.variances = np.asarray(self.variances, dtype=float).copy()
        self.start = np.asarray(self.start, dtype=float).copy()
        self.transition = np.asarray(self.transition, dtype=float).copy()
        self.start = self.start / self.start.sum()
        self.transition = self.transition / self.transition.sum(axis=1, keepdims=True)
        n_states, n_features = self.means.shape
        if self.variances.shape != (n_states, n_features):
            raise ValueError("variances must match means")
        if self.start.shape != (n_states,):
            raise ValueError("start length must match the number of stages")
        if self.transition.shape != (n_states, n_states):
            raise ValueError("transition must be square in the number of stages")
        if np.any(self.variances <= 0):
            raise ValueError("variances must be positive")

    @property
    def n_states(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])

    def copy(self) -> "GaussianHMM":
        return GaussianHMM(self.means, self.variances, self.start, self.transition)

    def log_emissions(self, observations: np.ndarray) -> np.ndarray:
        """Log p(x_t | z_t) for each time and state. NaN features are skipped."""
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"observations must have shape (T, {self.n_features})")
        observed = np.isfinite(x)
        filled = np.where(observed, x, 0.0)
        diff = filled[:, None, :] - self.means[None, :, :]
        var = self.variances[None, :, :]
        log_prob = -0.5 * (np.log(2.0 * np.pi * var) + (diff**2) / var)
        log_prob = np.where(observed[:, None, :], log_prob, 0.0)
        empty = ~observed.any(axis=1)
        if np.any(empty):
            raise ValueError("every timestep needs at least one spectral index")
        return log_prob.sum(axis=-1)

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """Most likely stage index at each time, given the whole series."""
        log_emit = self.log_emissions(observations)
        log_trans = _log_prob(self.transition)
        length, n_states = log_emit.shape
        delta = np.empty((length, n_states))
        psi = np.zeros((length, n_states), dtype=int)
        delta[0] = _log_prob(self.start) + log_emit[0]
        for t in range(1, length):
            scores = delta[t - 1][:, None] + log_trans
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = log_emit[t] + np.max(scores, axis=0)
        path = np.empty(length, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(length - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def posteriors(self, observations: np.ndarray) -> dict[str, np.ndarray]:
        """Forward filter, smoothed posterior, and sequence log-likelihood.

        Filtered posterior at t uses only observations up to t (live monitoring).
        Smoothed posterior uses the whole series (season review).
        """
        log_emit = self.log_emissions(observations)
        log_trans = _log_prob(self.transition)
        length, n_states = log_emit.shape
        log_alpha = np.empty((length, n_states))
        log_alpha[0] = _log_prob(self.start) + log_emit[0]
        for t in range(1, length):
            scores = log_alpha[t - 1][:, None] + log_trans
            log_alpha[t] = log_emit[t] + logsumexp(scores, axis=0)

        log_beta = np.zeros((length, n_states))
        for t in range(length - 2, -1, -1):
            scores = log_trans + log_emit[t + 1][None, :] + log_beta[t + 1][None, :]
            log_beta[t] = logsumexp(scores, axis=1)

        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - logsumexp(log_gamma, axis=1)[:, None]
        log_filtered = log_alpha - logsumexp(log_alpha, axis=1)[:, None]
        loglik = logsumexp(log_alpha[-1])
        return {
            "smoothed": np.exp(log_gamma),
            "filtered": np.exp(log_filtered),
            "log_likelihood": loglik,
            "log_alpha": log_alpha,
            "log_beta": log_beta,
            "log_emit": log_emit,
            "log_trans": log_trans,
        }

    def sample(self, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        states = np.empty(length, dtype=int)
        states[0] = int(rng.choice(self.n_states, p=self.start))
        for t in range(1, length):
            states[t] = int(rng.choice(self.n_states, p=self.transition[states[t - 1]]))
        obs = np.empty((length, self.n_features))
        for t in range(length):
            scale = np.sqrt(self.variances[states[t]])
            obs[t] = rng.normal(self.means[states[t]], scale)
        return obs, states

    def fit(
        self,
        sequences: list[np.ndarray],
        n_iter: int = 15,
        mean_prior_strength: float = 0.0,
        transition_prior_strength: float = 0.0,
        var_floor: float = 4e-4,
        tol: float = 1e-4,
    ) -> tuple["GaussianHMM", list[float]]:
        """Baum-Welch. Priors keep a short unlabeled archive from leaving phenology behind.

        mean_prior_strength is a pseudo-count toward this model's emission means.
        transition_prior_strength weights the current transition matrix, and
        backward or out-of-order stage changes stay at probability zero.
        """
        if not sequences:
            raise ValueError("fit needs at least one spectral series")
        model = self.copy()
        prior_means = self.means.copy()
        prior_variances = self.variances.copy()
        prior_trans = self.transition.copy()
        mask = transition_mask(model.n_states)
        trace: list[float] = []
        previous: float | None = None
        for _ in range(n_iter):
            updated, loglik = model._em_step(
                sequences,
                prior_means,
                prior_variances,
                prior_trans,
                mask,
                mean_prior_strength,
                transition_prior_strength,
                var_floor,
            )
            trace.append(float(loglik))
            model = updated
            if previous is not None and abs(trace[-1] - previous) < tol:
                break
            previous = trace[-1]
        return model, trace

    def _em_step(
        self,
        sequences: list[np.ndarray],
        prior_means: np.ndarray,
        prior_variances: np.ndarray,
        prior_trans: np.ndarray,
        mask: np.ndarray,
        mean_prior_strength: float,
        transition_prior_strength: float,
        var_floor: float,
    ) -> tuple["GaussianHMM", float]:
        n_states, n_features = self.means.shape
        mean_numer = np.zeros((n_states, n_features))
        mean_denom = np.zeros((n_states, n_features))
        start_counts = np.zeros(n_states)
        xi_counts = np.zeros((n_states, n_states))
        occupancies: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        total_ll = 0.0

        for observations in sequences:
            post = self.posteriors(observations)
            gamma = post["smoothed"]
            total_ll += float(post["log_likelihood"])
            start_counts += gamma[0]
            observed = np.isfinite(observations)
            filled = np.where(observed, observations, 0.0)
            for k in range(n_states):
                weight = gamma[:, k][:, None] * observed
                mean_numer[k] += (weight * filled).sum(axis=0)
                mean_denom[k] += weight.sum(axis=0)
            occupancies.append((gamma, filled, observed))
            length = observations.shape[0]
            if length < 2:
                continue
            log_alpha = post["log_alpha"]
            log_beta = post["log_beta"]
            log_emit = post["log_emit"]
            log_trans = post["log_trans"]
            for t in range(length - 1):
                log_xi = (
                    log_alpha[t][:, None]
                    + log_trans
                    + log_emit[t + 1][None, :]
                    + log_beta[t + 1][None, :]
                )
                log_xi = log_xi - logsumexp(log_xi)
                xi_counts += np.exp(log_xi)

        means = prior_means.copy()
        for k in range(n_states):
            for d in range(n_features):
                denom = mean_denom[k, d] + mean_prior_strength
                if denom <= 1e-8:
                    continue
                means[k, d] = (
                    mean_numer[k, d] + mean_prior_strength * prior_means[k, d]
                ) / denom

        var_numer = np.zeros((n_states, n_features))
        var_denom = np.zeros((n_states, n_features))
        for gamma, filled, observed in occupancies:
            for k in range(n_states):
                weight = gamma[:, k][:, None] * observed
                var_numer[k] += (weight * (filled - means[k]) ** 2).sum(axis=0)
                var_denom[k] += weight.sum(axis=0)

        variances = prior_variances.copy()
        for k in range(n_states):
            for d in range(n_features):
                denom = var_denom[k, d] + mean_prior_strength
                if denom <= 1e-8:
                    continue
                sample = var_numer[k, d] + mean_prior_strength * prior_variances[k, d]
                variances[k, d] = max(sample / denom, var_floor)

        if start_counts.sum() <= 1e-8:
            start = self.start.copy()
        else:
            start = start_counts / start_counts.sum()

        trans_counts = xi_counts + transition_prior_strength * prior_trans
        trans_counts *= mask
        transition = self.transition.copy()
        for i in range(n_states):
            row = trans_counts[i].sum()
            if row <= 1e-8:
                continue
            transition[i] = trans_counts[i] / row

        return GaussianHMM(means, variances, start, transition), total_ll
