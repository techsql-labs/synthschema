"""Business rules that must hold on 100% of generated rows.

Rules are enforced outside the model. Where possible a rule becomes a reversible transform
during training (``ship_date >= order_date`` is learned as a non-negative difference), so
the model never has to discover it. Every rule also has a row-level validity check;
synthesizers resample any row that fails it.

Rules can be written as objects, as dicts (for JSON/YAML blueprints) or as SQL-like strings::

    Inequality(low="order_date", high="ship_date")
    {"type": "range", "column": "discount", "min": 0, "max": 0.4}
    "ship_date >= order_date"
    "age BETWEEN 18 AND 90"
    "status IN ('open', 'closed')"
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metadata import Column, Table
from .transformers import (
    datetime_to_seconds,
    format_datetime,
    learn_decimals,
    learn_granularity,
    seconds_to_datetime,
)


def _is_datetime(table: Table | None, column: str, series: pd.Series | None = None) -> bool:
    if table is not None and column in table.columns:
        return table.columns[column].sdtype == "datetime"
    return series is not None and pd.api.types.is_datetime64_any_dtype(series)


def _fmt(table: Table | None, column: str) -> str | None:
    if table is not None and column in table.columns:
        return table.columns[column].datetime_format
    return None


def _as_number(value: Any, is_dt: bool) -> float:
    if is_dt:
        return pd.Timestamp(value).value / 1e9
    return float(value)


class Constraint:
    """Base class. Subclasses override the hooks they need."""

    type = "constraint"

    #: Columns that the transform removes from the training frame.
    consumed_columns: tuple[str, ...] = ()

    def fit(self, data: pd.DataFrame, table: Table | None = None) -> Constraint:
        self.table = table
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def is_valid(self, data: pd.DataFrame) -> np.ndarray:
        return np.ones(len(data), dtype=bool)

    def repair(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make invalid rows valid with the smallest change (used by schema-only generation)."""
        return data

    def derived_columns(self) -> dict[str, Column]:
        """Columns the transform adds, with their sdtypes."""
        return {}

    @property
    def columns(self) -> tuple[str, ...]:  # pragma: no cover - overridden
        return ()

    def to_dict(self) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError

    def __repr__(self) -> str:
        try:
            return f"{type(self).__name__}({self.to_dict()})"
        except NotImplementedError:
            return f"{type(self).__name__}()"


class Inequality(Constraint):
    """``low <= high`` (or ``low < high`` when ``strict``) for numbers or datetimes."""

    type = "inequality"

    def __init__(self, low: str, high: str, strict: bool = False) -> None:
        self.low, self.high, self.strict = low, high, strict
        self.consumed_columns = (high,)
        self._diff = f"__{low}__{high}__diff"

    @property
    def columns(self):
        return (self.low, self.high)

    def _num(self, series: pd.Series, column: str) -> np.ndarray:
        if self._dt:
            return datetime_to_seconds(series, _fmt(self.table, column))
        return pd.to_numeric(series, errors="coerce").astype(float).to_numpy()

    def fit(self, data, table=None):
        self.table = table
        self._dt = _is_datetime(table, self.low, data[self.low]) or _is_datetime(
            table, self.high, data[self.high]
        )
        high = self._num(data[self.high], self.high)
        diff = high - self._num(data[self.low], self.low)
        positive = diff[np.isfinite(diff) & (diff > 0)]
        self._min_gap = float(positive.min()) if len(positive) else 1.0
        if self._dt:
            self._granularity = learn_granularity(high)
            self._as_string = not pd.api.types.is_datetime64_any_dtype(data[self.high])
        else:
            self._decimals = learn_decimals(high[np.isfinite(high)])
        return self

    def transform(self, data):
        out = data.copy()
        out[self._diff] = self._num(data[self.high], self.high) - self._num(data[self.low], self.low)
        return out.drop(columns=[self.high])

    def reverse(self, data):
        out = data.copy()
        diff = pd.to_numeric(out.pop(self._diff), errors="coerce").to_numpy(dtype=float)
        diff = np.where(diff < 0, 0.0, diff)
        if self.strict:
            diff = np.where(diff <= 0, self._min_gap, diff)
        high = self._num(out[self.low], self.low) + diff
        if self._dt:
            dt = seconds_to_datetime(high, self._granularity)
            out[self.high] = format_datetime(dt, _fmt(self.table, self.high), self._as_string).to_numpy()
        else:
            if self._decimals is not None:
                high = np.round(high, self._decimals)
            out[self.high] = high
        return out

    def is_valid(self, data):
        low = self._num(data[self.low], self.low)
        high = self._num(data[self.high], self.high)
        missing = ~np.isfinite(low) | ~np.isfinite(high)
        ok = high > low if self.strict else high >= low
        return ok | missing

    def repair(self, data):
        out = data.copy()
        bad = ~self.is_valid(out)
        if bad.any():
            lo, hi = out.loc[bad, self.low].copy(), out.loc[bad, self.high].copy()
            out.loc[bad, self.low], out.loc[bad, self.high] = hi.to_numpy(), lo.to_numpy()
        return out

    def derived_columns(self):
        return {self._diff: Column("numerical")}

    def to_dict(self):
        out = {"type": self.type, "low": self.low, "high": self.high}
        if self.strict:
            out["strict"] = True
        return out


class Range(Constraint):
    """``min <= column <= max``; either bound may be omitted."""

    type = "range"

    def __init__(self, column: str, min: Any = None, max: Any = None, strict: bool = False) -> None:
        self.column, self.min, self.max, self.strict = column, min, max, strict

    @property
    def columns(self):
        return (self.column,)

    def fit(self, data, table=None):
        self.table = table
        self._dt = _is_datetime(table, self.column, data.get(self.column))
        return self

    def _values(self, data):
        s = data[self.column]
        if getattr(self, "_dt", False):
            return datetime_to_seconds(s, _fmt(getattr(self, "table", None), self.column))
        return pd.to_numeric(s, errors="coerce").astype(float).to_numpy()

    def is_valid(self, data):
        x = self._values(data)
        ok = np.ones(len(x), dtype=bool)
        dt = getattr(self, "_dt", False)
        if self.min is not None:
            lo = _as_number(self.min, dt)
            ok &= (x > lo) if self.strict else (x >= lo)
        if self.max is not None:
            hi = _as_number(self.max, dt)
            ok &= (x < hi) if self.strict else (x <= hi)
        return ok | ~np.isfinite(x)

    def repair(self, data):
        if getattr(self, "_dt", False):
            return data
        out = data.copy()
        x = pd.to_numeric(out[self.column], errors="coerce")
        out[self.column] = x.clip(
            lower=None if self.min is None else float(self.min),
            upper=None if self.max is None else float(self.max),
        )
        return out

    def to_dict(self):
        out = {"type": self.type, "column": self.column}
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if self.strict:
            out["strict"] = True
        return out


class OneOf(Constraint):
    """``column IN (values)``."""

    type = "one_of"

    def __init__(self, column: str, values: Sequence) -> None:
        self.column, self.values = column, list(values)

    @property
    def columns(self):
        return (self.column,)

    def is_valid(self, data):
        s = data[self.column]
        return (s.isin(self.values) | s.isna()).to_numpy()

    def to_dict(self):
        return {"type": self.type, "column": self.column, "values": self.values}


class FixedCombinations(Constraint):
    """Only combinations of these columns seen in the real data may appear
    (e.g. country/currency, category/brand/SKU)."""

    type = "fixed_combinations"

    def __init__(self, columns: Sequence[str]) -> None:
        if len(columns) < 2:
            raise ValueError("FixedCombinations needs at least two columns")
        self._columns = tuple(columns)
        self.consumed_columns = self._columns
        self._combo = "__" + "__".join(self._columns) + "__combo"

    @property
    def columns(self):
        return self._columns

    @staticmethod
    def _key(values) -> str:
        return json.dumps([None if pd.isna(v) else v for v in values], default=str)

    def fit(self, data, table=None):
        self.table = table
        keys = [self._key(r) for r in data[list(self._columns)].itertuples(index=False)]
        self._lookup: dict[str, tuple] = {}
        for key, row in zip(keys, data[list(self._columns)].itertuples(index=False), strict=False):
            self._lookup.setdefault(key, tuple(row))
        return self

    def transform(self, data):
        out = data.copy()
        out[self._combo] = [self._key(r) for r in data[list(self._columns)].itertuples(index=False)]
        return out.drop(columns=list(self._columns))

    def reverse(self, data):
        out = data.copy()
        keys = out.pop(self._combo)
        rows = [self._lookup.get(k, (None,) * len(self._columns)) for k in keys]
        for i, col in enumerate(self._columns):
            out[col] = pd.Series([r[i] for r in rows], index=out.index, dtype=object)
        return out

    def is_valid(self, data):
        keys = [self._key(r) for r in data[list(self._columns)].itertuples(index=False)]
        return np.array([k in self._lookup for k in keys], dtype=bool)

    def derived_columns(self):
        return {self._combo: Column("categorical")}

    def to_dict(self):
        return {"type": self.type, "columns": list(self._columns)}


class Custom(Constraint):
    """Any row-level rule given as a function ``DataFrame -> boolean array``."""

    type = "custom"

    def __init__(self, is_valid: Callable[[pd.DataFrame], Any], columns: Sequence[str] = ()) -> None:
        self._fn = is_valid
        self._columns = tuple(columns)

    @property
    def columns(self):
        return self._columns

    def is_valid(self, data):
        return np.asarray(self._fn(data), dtype=bool)

    def to_dict(self):
        raise NotImplementedError("Custom constraints cannot be serialised")


# ---------------------------------------------------------------------- parsing

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_BETWEEN_RE = re.compile(rf"({_IDENT})\s+BETWEEN\s+(\S+)\s+AND\s+(\S+)", re.IGNORECASE)
_IN_RE = re.compile(rf"^({_IDENT})\s+IN\s*\((.*)\)$", re.IGNORECASE | re.DOTALL)
_CMP_RE = re.compile(rf"^({_IDENT})\s*(>=|<=|>|<)\s*(.+)$")
_AND_RE = re.compile(r"\s+AND\s+", re.IGNORECASE)


def _literal(text: str) -> Any:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return None


def _strip_outer_parens(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        for i, ch in enumerate(text):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and i < len(text) - 1:
                return text  # the first "(" closes before the end: not a wrapper
        text = text[1:-1].strip()
    return text


def parse_rule(rule: str) -> list[Constraint]:
    """Parse a SQL-like rule string into constraints. Raises ``ValueError`` if unsupported."""
    text = _strip_outer_parens(rule)
    out: list[Constraint] = []
    for m in _BETWEEN_RE.finditer(text):
        lo, hi = _literal(m.group(2)), _literal(m.group(3))
        if lo is None or hi is None:
            raise ValueError(f"Unsupported BETWEEN bounds in rule {rule!r}")
        out.append(Range(m.group(1), min=lo, max=hi))
    text = _BETWEEN_RE.sub("", text).strip()
    parts = [_strip_outer_parens(p) for p in _AND_RE.split(text) if p.strip()] if text else []
    for part in parts:
        m = _IN_RE.match(part)
        if m:
            values = [_literal(v) for v in re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", m.group(2))]
            out.append(OneOf(m.group(1), [v for v in values if v is not None]))
            continue
        m = _CMP_RE.match(part)
        if not m:
            raise ValueError(f"Unsupported rule {rule!r}")
        left, op, right = m.group(1), m.group(2), m.group(3).strip()
        strict = op in {">", "<"}
        if re.fullmatch(_IDENT, right):
            low, high = (right, left) if op in {">=", ">"} else (left, right)
            out.append(Inequality(low, high, strict=strict))
            continue
        value = _literal(right)
        if value is None:
            raise ValueError(f"Unsupported rule {rule!r}")
        if op in {">=", ">"}:
            out.append(Range(left, min=value, strict=strict))
        else:
            out.append(Range(left, max=value, strict=strict))
    if not out:
        raise ValueError(f"Unsupported rule {rule!r}")
    return out


def from_spec(spec: Any) -> list[Constraint]:
    """Turn a constraint object, dict or rule string into constraint objects."""
    if isinstance(spec, Constraint):
        return [spec]
    if isinstance(spec, str):
        return parse_rule(spec)
    if isinstance(spec, Mapping):
        if "rule" in spec:
            return parse_rule(spec["rule"])
        kind = str(spec.get("type", "")).lower()
        if kind == "inequality":
            return [Inequality(spec["low"], spec["high"], bool(spec.get("strict", False)))]
        if kind in {"range", "scalar_range"}:
            return [
                Range(
                    spec["column"],
                    spec.get("min", spec.get("min_value")),
                    spec.get("max", spec.get("max_value")),
                    bool(spec.get("strict", False)),
                )
            ]
        if kind == "positive":
            return [Range(spec["column"], min=0, strict=bool(spec.get("strict", True)))]
        if kind == "one_of":
            return [OneOf(spec["column"], spec["values"])]
        if kind == "fixed_combinations":
            return [FixedCombinations(spec["columns"])]
        raise ValueError(f"Unknown constraint type {spec.get('type')!r}")
    raise TypeError(f"Cannot build a constraint from {type(spec).__name__}")


def build_constraints(specs: Sequence[Any] | None) -> list[Constraint]:
    out: list[Constraint] = []
    for spec in specs or []:
        out.extend(from_spec(spec))
    return out
