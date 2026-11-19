"""Single-table synthesis with a Gaussian copula."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ._version import __version__
from .constraints import Constraint, build_constraints
from .copula import GaussianCopula
from .generators import IdGenerator, PiiGenerator, TextGenerator
from .marginals import EPS
from .metadata import Column, Metadata, Table, detect_column
from .transformers import make_handler, restore_dtype


class GaussianCopulaSynthesizer:
    """Learns each column's distribution and the correlations between columns, then samples
    new rows that keep that statistical balance.

    * Numerical and datetime columns: exact empirical marginals by default (or a fitted
      parametric family: ``norm``, ``beta``, ``truncnorm``, ``gamma``, ``uniform`` or ``auto``).
    * Categorical and boolean columns: frequency-interval encoding, so category shares are kept.
    * Correlations: a Gaussian copula over normal scores of every modelled column.
    * Keys, PII and free text are generated, never learned, so real values are not copied.
    * Constraints hold on 100% of rows.

    Example::

        synth = GaussianCopulaSynthesizer()
        synth.fit(df)
        fake = synth.sample(1_000, seed=42)
    """

    def __init__(
        self,
        metadata: Metadata | None = None,
        table_name: str | None = None,
        *,
        constraints: Sequence[Constraint | Mapping | str] | None = None,
        default_distribution: str = "empirical",
        numerical_distributions: Mapping[str, str] | None = None,
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        locale: str | list[str] | None = None,
    ) -> None:
        self.metadata = metadata
        self.table_name = table_name
        self._user_constraints = list(constraints or [])
        self.default_distribution = default_distribution
        self.numerical_distributions = dict(numerical_distributions or {})
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.locale = locale
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, data: pd.DataFrame) -> GaussianCopulaSynthesizer:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("fit() expects a pandas DataFrame")
        if len(data) == 0:
            raise ValueError("Cannot fit on an empty DataFrame")
        data = data.reset_index(drop=True)
        if self.metadata is None:
            self.metadata = Metadata.detect_from_dataframe(data, self.table_name or "table")
        table = self.metadata.get_table(self.table_name)
        self.table_name = table.name
        missing = [c for c in table.columns if c not in data.columns]
        if missing:
            raise ValueError(f"Columns in metadata but not in data: {missing}")

        columns = {str(c): table.columns.get(str(c)) for c in data.columns}
        for name, meta in columns.items():
            if meta is None:
                columns[name] = detect_column(data[name], name)
        self._table = Table(name=table.name, columns=columns, primary_key=table.primary_key)
        self._columns = list(data.columns)
        self._dtypes = dict(data.dtypes)
        self._n_rows = len(data)

        self._constraints = build_constraints(table.constraints) + build_constraints(self._user_constraints)
        work = data
        derived: dict[str, Column] = {}
        for constraint in self._constraints:
            constraint.fit(work, self._table)
            work = constraint.transform(work)
            derived.update(constraint.derived_columns())

        rng = np.random.default_rng(0)
        self._handlers: dict[str, Any] = {}
        self._slices: dict[str, tuple[int, int]] = {}
        self._generators: dict[str, Any] = {}
        blocks = []
        offset = 0
        for col in work.columns:
            meta = columns.get(col) or derived.get(col) or detect_column(work[col], col)
            if meta.is_modeled:
                dist = self.numerical_distributions.get(col, self.default_distribution)
                handler = make_handler(meta, dist, self.enforce_min_max_values, self.enforce_rounding)
                handler.fit(work[col])
                block = handler.to_uniform(work[col], rng)
                self._handlers[col] = handler
                self._slices[col] = (offset, offset + block.shape[1])
                offset += block.shape[1]
                blocks.append(block)
            elif meta.sdtype == "id":
                self._generators[col] = IdGenerator(meta).fit(work[col])
            elif meta.sdtype == "text":
                self._generators[col] = TextGenerator(self.locale).fit(work[col])
            else:
                self._generators[col] = PiiGenerator(meta.sdtype, self.locale).fit(work[col])

        U = np.hstack(blocks) if blocks else np.zeros((len(work), 0))
        Z = stats.norm.ppf(np.clip(U, EPS, 1 - EPS))
        self._copula = GaussianCopula().fit(Z)
        self._fitted = True
        return self

    # ------------------------------------------------------------------ sampling internals
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before sampling")

    def _latent_given(self, cond: pd.DataFrame) -> tuple[list[int], np.ndarray]:
        idx: list[int] = []
        values = []
        for col in cond.columns:
            start, stop = self._slices[col]
            U = self._handlers[col].to_uniform(cond[col].reset_index(drop=True), None)
            idx.extend(range(start, stop))
            values.append(stats.norm.ppf(U))
        return idx, np.hstack(values)

    def _decode(self, Z: np.ndarray, cond: pd.DataFrame | None = None) -> pd.DataFrame:
        U = np.clip(stats.norm.cdf(Z), EPS, 1 - EPS)
        out = pd.DataFrame(index=pd.RangeIndex(len(Z)))
        for col, handler in self._handlers.items():
            start, stop = self._slices[col]
            out[col] = handler.from_uniform(U[:, start:stop]).to_numpy()
        if cond is not None:
            for col in cond.columns:  # conditioned values are kept exactly
                out[col] = cond[col].to_numpy()
        for constraint in reversed(self._constraints):
            out = constraint.reverse(out)
        return out

    def _sample_modeled(self, n: int, rng: np.random.Generator, cond: pd.DataFrame | None) -> pd.DataFrame:
        if cond is not None and len(cond.columns):
            idx, given = self._latent_given(cond)
            Z = self._copula.sample_conditional(idx, given, rng)
        else:
            Z = self._copula.sample(n, rng)
            cond = None
        return self._decode(Z, cond)

    def _valid(self, df: pd.DataFrame) -> np.ndarray:
        mask = np.ones(len(df), dtype=bool)
        for constraint in self._constraints:
            mask &= constraint.is_valid(df)
        return mask

    def _generate(
        self,
        n: int,
        rng: np.random.Generator,
        cond: pd.DataFrame | None = None,
        on_fail: str = "raise",
        max_tries: int = 100,
    ) -> pd.DataFrame:
        """Sample ``n`` complete rows. With ``cond``, row ``i`` is conditioned on ``cond.iloc[i]``.
        The returned index is the position of each row's condition (rows may be dropped only
        when ``on_fail="drop"``)."""
        self._check_fitted()
        if cond is not None:
            cond = cond.reset_index(drop=True)
        df = self._sample_modeled(n, rng, cond)
        valid = self._valid(df)
        tries = 0
        while not valid.all() and tries < max_tries:
            bad = np.flatnonzero(~valid)
            redo = self._sample_modeled(len(bad), rng, None if cond is None else cond.iloc[bad])
            for col in redo.columns:
                if df[col].dtype != redo[col].dtype:
                    df[col] = df[col].astype(object)
                df.loc[bad, col] = redo[col].to_numpy()
            valid[bad] = self._valid(redo)
            tries += 1
        if not valid.all():
            if on_fail == "raise":
                raise ValueError(
                    f"Could not satisfy the constraints for {(~valid).sum()} of {n} rows after "
                    f"{max_tries} attempts; check that the rules agree with each other and the data"
                )
            df = df[valid]
        m = len(df)
        pk = self._table.primary_key
        for col, gen in self._generators.items():
            if col == pk:
                df[col] = gen.generate(m, rng, unique=True, allow_nulls=False)
            else:
                df[col] = gen.generate(m, rng)
        df = df[self._columns]
        for col in self._columns:
            df[col] = restore_dtype(df[col], self._dtypes[col])
        return df

    # ------------------------------------------------------------------ public sampling
    def sample(
        self,
        num_rows: int,
        conditions: Mapping[str, Any] | None = None,
        seed: int | None = None,
        max_tries: int = 100,
    ) -> pd.DataFrame:
        """Generate ``num_rows`` synthetic rows.

        ``conditions`` fixes column values for every row, e.g. ``{"region": "North"}``. The
        other columns are drawn from the copula *conditioned on* those values, so their
        correlations with the fixed columns are kept. ``seed`` makes the output reproducible.
        """
        self._check_fitted()
        if num_rows < 0:
            raise ValueError("num_rows must be >= 0")
        rng = np.random.default_rng(seed)
        if not conditions:
            return self._generate(num_rows, rng, max_tries=max_tries).reset_index(drop=True)
        unknown = [c for c in conditions if c not in self._columns]
        if unknown:
            raise KeyError(f"Unknown condition column(s): {unknown}")
        direct = {k: v for k, v in conditions.items() if k in self._handlers}
        indirect = {k: v for k, v in conditions.items() if k not in self._handlers}
        cond = pd.DataFrame({k: [v] * num_rows for k, v in direct.items()}) if direct else None
        if not indirect:
            return self._generate(num_rows, rng, cond, max_tries=max_tries).reset_index(drop=True)
        return self._reject_sample(num_rows, rng, direct, indirect, max_tries)

    def _reject_sample(self, num_rows, rng, direct, indirect, max_tries) -> pd.DataFrame:
        batch = max(num_rows * 4, 200)
        cond = pd.DataFrame({k: [v] * batch for k, v in direct.items()}) if direct else None
        kept: list[pd.DataFrame] = []
        total = 0
        for _ in range(max_tries):
            df = self._generate(batch, rng, cond, on_fail="drop", max_tries=max_tries)
            mask = np.ones(len(df), dtype=bool)
            for col, value in indirect.items():
                mask &= (df[col] == value).to_numpy()
            if mask.any():
                kept.append(df[mask])
                total += int(mask.sum())
            if total >= num_rows:
                break
        if total < num_rows:
            raise ValueError(
                f"Only {total} of {num_rows} rows matched the conditions {indirect}; these "
                "columns are shaped by constraints, so they are matched by rejection sampling"
            )
        return pd.concat(kept).head(num_rows).reset_index(drop=True)

    def sample_from_conditions(self, conditions: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
        """One synthetic row per row of ``conditions``; non-null cells are fixed values."""
        self._check_fitted()
        rng = np.random.default_rng(seed)
        unknown = [c for c in conditions.columns if c not in self._handlers]
        if unknown:
            raise KeyError(f"sample_from_conditions supports modelled columns only; got {unknown}")
        conditions = conditions.reset_index(drop=True)
        mask = conditions.notna()
        pieces = []
        for pattern, group in conditions.groupby(mask.apply(tuple, axis=1), sort=False):
            cols = [c for c, present in zip(conditions.columns, pattern, strict=False) if present]
            cond = group[cols] if cols else None
            out = self._generate(len(group), rng, cond)
            out.index = group.index
            pieces.append(out)
        return pd.concat(pieces).sort_index()

    # ------------------------------------------------------------------ inspection and io
    def get_metadata(self) -> Metadata:
        return self.metadata

    def get_learned_correlations(self) -> pd.DataFrame:
        """Correlation of normal scores between modelled (latent) columns."""
        self._check_fitted()
        names = []
        for col, (start, stop) in self._slices.items():
            names.extend([col] if stop - start == 1 else [f"{col}.is_null", col])
        return pd.DataFrame(self._copula.corr, index=names, columns=names)

    def save(self, path: str | Path) -> None:
        """Save the fitted synthesizer (pickle). Only load files you trust."""
        with open(path, "wb") as fh:
            pickle.dump({"synthschema_version": __version__, "synthesizer": self}, fh)

    @classmethod
    def load(cls, path: str | Path) -> GaussianCopulaSynthesizer:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 - documented: trusted files only
        obj = payload["synthesizer"] if isinstance(payload, dict) else payload
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj

    def __repr__(self) -> str:
        state = f"fitted on {self._n_rows} rows" if self._fitted else "not fitted"
        return f"GaussianCopulaSynthesizer(table={self.table_name!r}, {state})"
