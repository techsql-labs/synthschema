"""Reversible column transformers between data values and the copula's uniform space.

Each handler maps one column to one or more columns of values in (0, 1) and back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .marginals import EPS, fit_marginal
from .metadata import Column

_NULL = "__synthschema_null__"


def learn_decimals(values: np.ndarray, max_decimals: int = 6) -> int | None:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    sample = values if len(values) <= 10_000 else values[:: max(1, len(values) // 10_000)]
    for d in range(max_decimals + 1):
        scaled = sample * 10**d
        if np.allclose(scaled, np.round(scaled), rtol=0, atol=1e-6 * max(1.0, 10**d)):
            if d == 0 and len(np.unique(sample)) > 10:
                # Whole numbers: also learn rounding to tens, hundreds, ... (e.g. salaries).
                ints = np.round(sample).astype(np.int64)
                for k in range(6, 0, -1):
                    if (ints % 10**k == 0).all():
                        return -k
            return d
    return None


class CategoricalHandler:
    """Frequency-interval encoding: each category owns a slice of (0, 1) sized by its share."""

    n_latent = 1

    def fit(self, series: pd.Series) -> CategoricalHandler:
        values = series.astype(object).where(series.notna(), _NULL)
        counts = values.value_counts(sort=True)
        self.categories = list(counts.index)
        p = counts.to_numpy(dtype=float)
        p = p / p.sum()
        self.upper = np.cumsum(p)
        self.upper[-1] = 1.0
        self.lower = np.concatenate([[0.0], self.upper[:-1]])
        self.index = {c: i for i, c in enumerate(self.categories)}
        self.probabilities = p
        return self

    def to_uniform(self, series: pd.Series, rng: np.random.Generator | None = None) -> np.ndarray:
        values = series.astype(object).where(series.notna(), _NULL)
        idx = values.map(self.index)
        unknown = idx.isna().to_numpy()
        idx = idx.fillna(0).astype(int).to_numpy()
        lo, hi = self.lower[idx], self.upper[idx]
        frac = rng.random(len(idx)) if rng is not None else 0.5
        u = lo + frac * (hi - lo)
        if unknown.any():
            u[unknown] = rng.random(unknown.sum()) if rng is not None else 0.5
        return np.clip(u, EPS, 1 - EPS)[:, None]

    def from_uniform(self, U: np.ndarray) -> pd.Series:
        idx = np.searchsorted(self.upper, U[:, 0], side="right")
        idx = np.clip(idx, 0, len(self.categories) - 1)
        out = np.array(self.categories, dtype=object)[idx]
        out[out == _NULL] = None
        return pd.Series(out, dtype=object)


class NumericalHandler:
    """A fitted marginal plus an optional null indicator, min/max clipping and rounding."""

    def __init__(
        self,
        distribution: str = "empirical",
        enforce_min_max: bool = True,
        enforce_rounding: bool = True,
    ) -> None:
        self.distribution = distribution
        self.enforce_min_max = enforce_min_max
        self.enforce_rounding = enforce_rounding

    def _values(self, series: pd.Series) -> np.ndarray:
        return pd.to_numeric(series, errors="coerce").astype(float).to_numpy()

    def fit(self, series: pd.Series) -> NumericalHandler:
        x = self._values(series)
        nulls = np.isnan(x)
        self.null_rate = float(nulls.mean()) if len(x) else 0.0
        self.has_nulls = 0 < self.null_rate < 1
        self.all_null = len(x) == 0 or self.null_rate == 1
        vals = x[~nulls]
        self.min = float(vals.min()) if len(vals) else 0.0
        self.max = float(vals.max()) if len(vals) else 0.0
        self.decimals = learn_decimals(vals) if self.enforce_rounding else None
        self.marginal = fit_marginal(vals, self.distribution)
        self.null_handler = CategoricalHandler().fit(pd.Series(nulls)) if self.has_nulls else None
        return self

    @property
    def n_latent(self) -> int:
        return 2 if self.has_nulls else 1

    def to_uniform(self, series: pd.Series, rng: np.random.Generator | None = None) -> np.ndarray:
        x = self._values(series)
        nulls = np.isnan(x)
        u = np.full(len(x), 0.5)
        if (~nulls).any() and not self.all_null:
            u[~nulls] = self.marginal.cdf(x[~nulls], rng)
        if self.has_nulls:
            return np.column_stack([self.null_handler.to_uniform(pd.Series(nulls), rng)[:, 0], u])
        return u[:, None]

    def from_uniform(self, U: np.ndarray) -> pd.Series:
        if self.all_null:
            return pd.Series(np.full(len(U), np.nan))
        values = self.marginal.ppf(U[:, -1])
        if self.enforce_min_max:
            values = np.clip(values, self.min, self.max)
        if self.decimals is not None:
            values = np.round(values, self.decimals)
        if self.has_nulls:
            isnull = self.null_handler.from_uniform(U[:, :1]).astype(bool).to_numpy()
            values = values.astype(float)
            values[isnull] = np.nan
        return pd.Series(values)


_GRANULARITY_NS = {"day": 86_400 * 10**9, "second": 10**9}


def to_datetime(series: pd.Series, fmt: str | None) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        out = series
    elif fmt:
        out = pd.to_datetime(series, format=fmt, errors="coerce")
    else:
        out = pd.to_datetime(series, format="ISO8601", errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def datetime_to_seconds(series: pd.Series, fmt: str | None) -> np.ndarray:
    dt = to_datetime(series, fmt)
    ns = dt.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
    ns[dt.isna().to_numpy()] = np.nan
    return ns / 1e9


def learn_granularity(seconds: np.ndarray) -> str | None:
    vals = seconds[np.isfinite(seconds)]
    if len(vals) == 0:
        return None
    ns = np.round(vals * 1e9).astype(np.int64)
    if (ns % _GRANULARITY_NS["day"] == 0).all():
        return "day"
    if (ns % _GRANULARITY_NS["second"] == 0).all():
        return "second"
    return None


def seconds_to_datetime(seconds: np.ndarray, granularity: str | None) -> pd.Series:
    seconds = np.asarray(seconds, dtype=float)
    out = pd.to_datetime(pd.Series(seconds), unit="s", errors="coerce")
    if granularity == "day":
        out = out.dt.floor("D")
    elif granularity == "second":
        out = out.dt.round("s")
    return out


def format_datetime(dt: pd.Series, fmt: str | None, as_string: bool) -> pd.Series:
    if not as_string:
        return dt
    if fmt:
        out = dt.dt.strftime(fmt)
    else:
        out = dt.map(lambda v: v.isoformat() if pd.notna(v) else None)
    return out.astype(object).where(dt.notna(), None)


class DatetimeHandler:
    """Datetimes are modelled as seconds since the epoch and restored in their own format."""

    def __init__(self, column: Column | None = None, distribution: str = "empirical",
                 enforce_min_max: bool = True) -> None:
        self.fmt = column.datetime_format if column else None
        self.numeric = NumericalHandler(distribution, enforce_min_max, enforce_rounding=False)

    def fit(self, series: pd.Series) -> DatetimeHandler:
        self.as_string = not pd.api.types.is_datetime64_any_dtype(series)
        seconds = datetime_to_seconds(series, self.fmt)
        self.granularity = learn_granularity(seconds)
        self.numeric.fit(pd.Series(seconds))
        return self

    @property
    def n_latent(self) -> int:
        return self.numeric.n_latent

    def to_uniform(self, series: pd.Series, rng: np.random.Generator | None = None) -> np.ndarray:
        return self.numeric.to_uniform(pd.Series(datetime_to_seconds(series, self.fmt)), rng)

    def from_uniform(self, U: np.ndarray) -> pd.Series:
        seconds = self.numeric.from_uniform(U).to_numpy()
        dt = seconds_to_datetime(seconds, self.granularity)
        return format_datetime(dt, self.fmt, self.as_string)


def make_handler(column: Column, distribution: str = "empirical", enforce_min_max: bool = True,
                 enforce_rounding: bool = True):
    if column.sdtype == "numerical":
        return NumericalHandler(distribution, enforce_min_max, enforce_rounding)
    if column.sdtype == "datetime":
        return DatetimeHandler(column, distribution, enforce_min_max)
    if column.sdtype in {"categorical", "boolean"}:
        return CategoricalHandler()
    raise ValueError(f"sdtype {column.sdtype!r} is not modelled")


def restore_dtype(series: pd.Series, dtype) -> pd.Series:
    """Cast a generated column back to the dtype of the training column, where possible."""
    try:
        if isinstance(dtype, pd.CategoricalDtype):
            return series.astype(dtype)
        if pd.api.types.is_bool_dtype(dtype):
            if series.notna().all():
                return series.astype(bool)
            return series.astype("boolean")
        if pd.api.types.is_integer_dtype(dtype):
            num = pd.to_numeric(series, errors="coerce").round()
            if num.notna().all() and isinstance(dtype, np.dtype):
                return num.astype(dtype)
            return num.astype("Int64")
        if pd.api.types.is_float_dtype(dtype):
            return pd.to_numeric(series, errors="coerce").astype(dtype)
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return pd.to_datetime(series, errors="coerce")
        if isinstance(dtype, pd.StringDtype):
            return series.astype(dtype)
    except (TypeError, ValueError):
        return series
    return series
