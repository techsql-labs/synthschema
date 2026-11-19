"""Multivariate Gaussian copula with exact conditional sampling."""

from __future__ import annotations

import numpy as np


def _clip_psd(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    sym = (matrix + matrix.T) / 2
    w, v = np.linalg.eigh(sym)
    return (v * np.clip(w, floor, None)) @ v.T


def nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    fixed = _clip_psd(matrix)
    d = np.sqrt(np.diag(fixed))
    out = fixed / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return out


def _cholesky(matrix: np.ndarray) -> np.ndarray:
    jitter = 1e-10
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + jitter * np.eye(len(matrix)))
        except np.linalg.LinAlgError:
            jitter *= 100
    return np.linalg.cholesky(_clip_psd(matrix, 1e-6))


class GaussianCopula:
    """Learns the correlation of normal scores and samples from it."""

    def fit(self, Z: np.ndarray) -> GaussianCopula:
        n, d = Z.shape
        self.dim = d
        if d == 0:
            self.corr = np.zeros((0, 0))
        elif n < 2 or d == 1:
            self.corr = np.eye(d)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(Z, rowvar=False)
            corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0)
            np.fill_diagonal(corr, 1.0)
            self.corr = nearest_correlation(corr)
        self._chol = _cholesky(self.corr) if d else np.zeros((0, 0))
        self._cond_cache: dict = {}
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.dim == 0:
            return np.zeros((n, 0))
        return rng.standard_normal((n, self.dim)) @ self._chol.T

    def sample_conditional(
        self, given_idx: list[int], given: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample the free dimensions given the values of ``given_idx`` (one row per sample)."""
        n = len(given)
        if not given_idx:
            return self.sample(n, rng)
        key = tuple(given_idx)
        if key not in self._cond_cache:
            free = [i for i in range(self.dim) if i not in set(given_idx)]
            s11 = self.corr[np.ix_(free, free)]
            s12 = self.corr[np.ix_(free, given_idx)]
            s22 = self.corr[np.ix_(given_idx, given_idx)]
            a = np.linalg.solve(s22 + 1e-9 * np.eye(len(given_idx)), s12.T).T
            cov = _clip_psd(s11 - a @ s12.T, 1e-10) if free else np.zeros((0, 0))
            chol = _cholesky(cov) if free else cov
            self._cond_cache[key] = (free, a, chol)
        free, a, chol = self._cond_cache[key]
        out = np.empty((n, self.dim))
        out[:, given_idx] = given
        if free:
            out[:, free] = given @ a.T + rng.standard_normal((n, len(free))) @ chol.T
        return out
