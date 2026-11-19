"""Multi-table synthesis: parent-first, conditional generation that keeps keys intact.

Tables are generated in topological order. For each relationship the parent's model also
learns *how many children* each parent row has (correlated with the parent's attributes),
and the child's model learns its columns *conditioned on the parent row*. This:

* keeps referential integrity by construction (no orphan foreign keys),
* keeps the child-count distribution per parent,
* keeps cross-table patterns (for example, high-income customers placing larger orders),
* scales linearly with the number of tables.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ._version import __version__
from .generators import IdGenerator
from .metadata import Column, Metadata, Relationship, detect_column
from .single_table import GaussianCopulaSynthesizer
from .transformers import restore_dtype


def _count_col(rel: Relationship) -> str:
    return f"__count__{rel.child_table}__{rel.child_foreign_key}"


def _parent_col(col: str) -> str:
    return f"__parent__{col}"


@dataclass
class _TableInfo:
    n_rows: int
    columns: list[str]
    dtypes: dict
    primary_key: str | None
    pk_generator: IdGenerator | None
    primary: Relationship | None
    secondary: list[Relationship] = field(default_factory=list)
    self_refs: list[Relationship] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    null_rates: dict[str, float] = field(default_factory=dict)


class MultiTableSynthesizer:
    """Synthesizes a set of related tables.

    Example::

        metadata = Metadata.detect_from_dataframes({"customers": customers, "orders": orders})
        synth = MultiTableSynthesizer(metadata)
        synth.fit({"customers": customers, "orders": orders})
        fake = synth.sample(scale=1.0, seed=7)   # dict of DataFrames
    """

    def __init__(
        self,
        metadata: Metadata,
        *,
        default_distribution: str = "empirical",
        numerical_distributions: Mapping[str, Mapping[str, str]] | None = None,
        locale: str | list[str] | None = None,
        max_context_columns: int = 20,
    ) -> None:
        self.metadata = metadata
        self.default_distribution = default_distribution
        self.numerical_distributions = dict(numerical_distributions or {})
        self.locale = locale
        self.max_context_columns = max_context_columns
        self._fitted = False

    def _keys(self, table: str) -> set[str]:
        t = self.metadata.tables[table]
        keys = {r.child_foreign_key for r in self.metadata.relationships if r.child_table == table}
        if t.primary_key:
            keys.add(t.primary_key)
        return keys

    def _context_columns(self, table: str) -> list[str]:
        t = self.metadata.tables[table]
        keys = self._keys(table)
        cols = [c for c, meta in t.columns.items() if meta.is_modeled and c not in keys]
        return cols[: self.max_context_columns]

    # ------------------------------------------------------------------ fit
    def fit(self, data: Mapping[str, pd.DataFrame]) -> MultiTableSynthesizer:
        md = self.metadata
        md.validate()
        md.validate_data(data)
        self._order = md.topological_order()
        self._synths: dict[str, GaussianCopulaSynthesizer] = {}
        self._info: dict[str, _TableInfo] = {}

        for name in self._order:
            table = md.tables[name]
            df = data[name].reset_index(drop=True)
            pk = table.primary_key
            if pk is not None and (df[pk].isna().any() or not df[pk].is_unique):
                raise ValueError(f"Primary key {name}.{pk} must be unique and non-null")
            parents = md.parent_relationships(name)
            children = md.child_relationships(name)
            self_refs = [r for r in md.relationships if r.parent_table == r.child_table == name]
            keys = self._keys(name)

            own = [c for c in df.columns if c not in keys]
            aug = df[own].copy()
            aug_meta: dict[str, Column] = {
                c: table.columns[c] if c in table.columns else detect_column(df[c], c) for c in own
            }
            for rel in children:
                if pk is None:
                    raise ValueError(f"Table {name!r} has children but no primary key")
                counts = data[rel.child_table][rel.child_foreign_key].value_counts()
                aug[_count_col(rel)] = df[pk].map(counts).fillna(0).astype(int).to_numpy()
                aug_meta[_count_col(rel)] = Column("numerical", representation="int")

            primary = parents[0] if parents else None
            null_rates: dict[str, float] = {}
            context: list[str] = []
            if primary is not None:
                ptable = md.tables[primary.parent_table]
                pdf = data[primary.parent_table]
                context = self._context_columns(primary.parent_table)
                lookup = pdf.set_index(ptable.primary_key)[context]
                fk = df[primary.child_foreign_key]
                joined = lookup.reindex(fk.to_numpy())
                for c in context:
                    aug[_parent_col(c)] = joined[c].to_numpy()
                    aug_meta[_parent_col(c)] = ptable.columns[c]
                null_rates[primary.child_foreign_key] = float(1 - fk.isin(pdf[ptable.primary_key]).mean())
            for rel in parents[1:] + self_refs:
                null_rates[rel.child_foreign_key] = float(df[rel.child_foreign_key].isna().mean())

            sub = Metadata()
            sub_table = sub.add_table(name)
            sub_table.columns = aug_meta
            sub_table.constraints = list(table.constraints)
            synth = GaussianCopulaSynthesizer(
                sub,
                name,
                default_distribution=self.default_distribution,
                numerical_distributions=self.numerical_distributions.get(name),
                locale=self.locale,
            )
            synth.fit(aug)
            self._synths[name] = synth
            pk_gen = IdGenerator(table.columns[pk]).fit(df[pk]) if pk else None
            self._info[name] = _TableInfo(
                n_rows=len(df),
                columns=list(df.columns),
                dtypes=dict(df.dtypes),
                primary_key=pk,
                pk_generator=pk_gen,
                primary=primary,
                secondary=parents[1:],
                self_refs=self_refs,
                context=context,
                null_rates=null_rates,
            )
        self._fitted = True
        return self

    # ------------------------------------------------------------------ sample
    def sample(
        self,
        scale: float = 1.0,
        seed: int | None = None,
        num_rows: Mapping[str, int] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Generate every table.

        ``scale`` sizes root tables relative to the training data (child tables follow from
        the learned child counts). ``num_rows`` sets exact row counts for root tables.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before sampling")
        rng = np.random.default_rng(seed)
        num_rows = dict(num_rows or {})
        synthetic: dict[str, pd.DataFrame] = {}

        for name in self._order:
            info, synth = self._info[name], self._synths[name]
            if info.primary is None:
                n = num_rows.get(name, int(round(info.n_rows * scale)) if info.n_rows else 0)
                df = synth._generate(n, rng).reset_index(drop=True)
            else:
                df = self._sample_children(name, info, synth, synthetic, rng)
            if info.primary_key:
                df[info.primary_key] = info.pk_generator.generate(len(df), rng, unique=True, allow_nulls=False)
            for rel in info.secondary:
                df[rel.child_foreign_key] = self._pick_parents(rel, synthetic, len(df), info, rng)
            for rel in info.self_refs:
                keys = df[info.primary_key].to_numpy()
                picked = keys[rng.integers(0, len(keys), size=len(df))].astype(object) if len(keys) else np.array([], dtype=object)
                rate = info.null_rates.get(rel.child_foreign_key, 0.0)
                picked[rng.random(len(df)) < rate] = None
                df[rel.child_foreign_key] = picked
            synthetic[name] = df

        out = {}
        for name in self._order:
            info = self._info[name]
            df = synthetic[name][info.columns].copy()
            for col in self._keys(name):
                if col in df.columns:
                    df[col] = restore_dtype(df[col], info.dtypes[col])
            out[name] = df.reset_index(drop=True)
        return {name: out[name] for name in self.metadata.tables}

    def _sample_children(self, name, info, synth, synthetic, rng) -> pd.DataFrame:
        rel = info.primary
        parent = synthetic[rel.parent_table]
        counts = (
            pd.to_numeric(parent[_count_col(rel)], errors="coerce").fillna(0).clip(lower=0).round().astype(int).to_numpy()
        )
        rep = np.repeat(np.arange(len(parent)), counts)
        fk_values = parent[rel.parent_primary_key].to_numpy()[rep].astype(object)
        ctx = parent.iloc[rep][info.context].reset_index(drop=True)
        ctx.columns = [_parent_col(c) for c in info.context]
        df = synth._generate(len(rep), rng, ctx if len(ctx.columns) else None, on_fail="drop")
        fk_values = fk_values[df.index.to_numpy()]
        df = df.reset_index(drop=True)
        df[rel.child_foreign_key] = fk_values

        null_rate = info.null_rates.get(rel.child_foreign_key, 0.0)
        if 0 < null_rate < 1 and len(df):
            extra_n = int(round(len(df) * null_rate / (1 - null_rate)))
            if extra_n:
                extra = synth._generate(extra_n, rng, on_fail="drop").reset_index(drop=True)
                extra[rel.child_foreign_key] = None
                df = pd.concat([df, extra], ignore_index=True)
        return df

    def _pick_parents(self, rel, synthetic, n, info, rng) -> np.ndarray:
        parent = synthetic[rel.parent_table]
        if len(parent) == 0 or n == 0:
            return np.full(n, None, dtype=object)
        weights = pd.to_numeric(parent[_count_col(rel)], errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float)
        p = weights / weights.sum() if weights.sum() > 0 else None
        keys = parent[rel.parent_primary_key].to_numpy()
        picked = keys[rng.choice(len(parent), size=n, p=p)].astype(object)
        picked[rng.random(n) < info.null_rates.get(rel.child_foreign_key, 0.0)] = None
        return picked

    # ------------------------------------------------------------------ io
    def get_table_synthesizer(self, table: str) -> GaussianCopulaSynthesizer:
        return self._synths[table]

    def save(self, path: str | Path) -> None:
        """Save the fitted synthesizer (pickle). Only load files you trust."""
        with open(path, "wb") as fh:
            pickle.dump({"synthschema_version": __version__, "synthesizer": self}, fh)

    @classmethod
    def load(cls, path: str | Path) -> MultiTableSynthesizer:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 - documented: trusted files only
        obj = payload["synthesizer"] if isinstance(payload, dict) else payload
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "not fitted"
        return f"MultiTableSynthesizer(tables={list(self.metadata.tables)}, {state})"
