"""Univariate (marginal) distributions used by the Gaussian copula."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

EPS = 1e-6
_MAX_EMPIRICAL_POINTS = 50_000

DISTRIBUTIONS = ("empirical", "norm", "uniform", "beta", "truncnorm", "gamma", "auto")


class Marginal:
    name = "base"

    def fit(self, x: np.ndarray) -> Marginal:  # pragma: no cover - interface
        raise NotImplementedError

    def cdf(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def ppf(self, u: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class Constant(Marginal):
    name = "constant"

    def fit(self, x):
        self.value = float(x[0]) if len(x) else 0.0
        return self

    def cdf(self, x, rng=None):
        return np.full(len(x), 0.5)

    def ppf(self, u):
        return np.full(len(u), self.value)


class Empirical(Marginal):
    """Exact empirical distribution with linear interpolation between observed quantiles.

    Ties (discrete values) are spread uniformly over their probability mass so the
    probability-integral transform stays uniform.
    """

    name = "empirical"

    def fit(self, x):
        x = np.sort(np.asarray(x, dtype=float))
        if len(x) > _MAX_EMPIRICAL_POINTS:
            # x is sorted, so linear interpolation over ranks equals np.quantile (and is O(n)).
            positions = np.linspace(0, len(x) - 1, _MAX_EMPIRICAL_POINTS)
            x = np.interp(positions, np.arange(len(x)), x)
        self.sorted = x
        n = len(x)
        self.probs = (np.arange(n) + 0.5) / n
        return self

    def cdf(self, x, rng=None):
        x = np.asarray(x, dtype=float)
        s, n = self.sorted, len(self.sorted)
        left = np.searchsorted(s, x, side="left")
        right = np.searchsorted(s, x, side="right")
        tied = right > left
        u = np.interp(x, s, self.probs)
        if tied.any():
            frac = rng.random(tied.sum()) if rng is not None else 0.5
            u[tied] = (left[tied] + frac * (right[tied] - left[tied])) / n
        return np.clip(u, EPS, 1 - EPS)

    def ppf(self, u):
        return np.interp(np.asarray(u, dtype=float), self.probs, self.sorted)


class Parametric(Marginal):
    """A scipy.stats distribution fitted by maximum likelihood."""

    def __init__(self, name: str) -> None:
        self.name = name

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        lo, hi = float(x.min()), float(x.max())
        span = hi - lo
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.name == "norm":
                self.dist = stats.norm(*stats.norm.fit(x))
            elif self.name == "uniform":
                self.dist = stats.uniform(lo, span)
            elif self.name == "beta":
                loc, scale = lo - 1e-3 * span, span * 1.002
                a, b, _, _ = stats.beta.fit(x, floc=loc, fscale=scale)
                self.dist = stats.beta(a, b, loc=loc, scale=scale)
            elif self.name == "truncnorm":
                mu, sd = float(x.mean()), float(x.std()) or 1.0
                self.dist = stats.truncnorm((lo - mu) / sd, (hi - mu) / sd, loc=mu, scale=sd)
            elif self.name == "gamma":
                a, loc, scale = stats.gamma.fit(x, floc=lo - 1e-3 * span)
                self.dist = stats.gamma(a, loc=loc, scale=scale)
            else:
                raise ValueError(f"Unknown distribution {self.name!r}")
        return self

    def cdf(self, x, rng=None):
        return np.clip(self.dist.cdf(np.asarray(x, dtype=float)), EPS, 1 - EPS)

    def ppf(self, u):
        return self.dist.ppf(np.asarray(u, dtype=float))


def fit_marginal(x: np.ndarray, distribution: str = "empirical") -> Marginal:
    """Fit a marginal. ``"auto"`` picks the parametric family with the best KS statistic."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.all(x == x[0]):
        return Constant().fit(x)
    if distribution == "empirical":
        return Empirical().fit(x)
    if distribution == "auto":
        sample = x if len(x) <= 5000 else np.random.default_rng(0).choice(x, 5000, replace=False)
        best, best_stat = None, np.inf
        for name in ("norm", "uniform", "beta", "truncnorm", "gamma"):
            try:
                cand = Parametric(name).fit(sample)
                stat = stats.kstest(sample, cand.dist.cdf).statistic
            except Exception:
                continue
            if np.isfinite(stat) and stat < best_stat:
                best, best_stat = cand, stat
        if best is None:
            return Empirical().fit(x)
        return best.fit(x)
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution {distribution!r}; choose from {DISTRIBUTIONS}")
    try:
        return Parametric(distribution).fit(x)
    except Exception:
        return Empirical().fit(x)
