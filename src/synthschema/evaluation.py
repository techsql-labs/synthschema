"""Evaluation: is the synthetic data valid, faithful and private?

* :func:`evaluate_quality` - statistical fidelity (column shapes, column-pair trends,
  and child-count shapes for multi-table data). Scores are in [0, 1]; higher is better.
* :func:`run_diagnostic` - structural validity: keys, foreign keys, ranges, categories, rules.
* :func:`evaluate_privacy` - copies of real rows and distance to closest record (DCR).
* :func:`certify` - combines the three into a pass/fail :class:`Certificate` with hard stops.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ._version import __version__
from .constraints import build_constraints
from .metadata import Metadata, Table
from .transformers import datetime_to_seconds

Data = pd.DataFrame | Mapping[str, pd.DataFrame]


def _as_tables(real: Data, synthetic: Data, metadata: Metadata | None) -> tuple[dict, dict, Metadata]:
    if isinstance(real, pd.DataFrame):
        if metadata is None:
            metadata = Metadata.detect_from_dataframe(real)
        name = metadata.get_table().name
        return {name: real}, {name: synthetic}, metadata
    if metadata is None:
        metadata = Metadata.detect_from_dataframes(real)
    return dict(real), dict(synthetic), metadata


def _numeric(series: pd.Series, table: Table, col: str) -> np.ndarray:
    meta = table.columns.get(col)
    if meta is not None and meta.sdtype == "datetime":
        return datetime_to_seconds(series, meta.datetime_format)
    return pd.to_numeric(series, errors="coerce").astype(float).to_numpy()


def _is_numeric(table: Table, col: str) -> bool:
    meta = table.columns.get(col)
    return meta is not None and meta.sdtype in {"numerical", "datetime"}


def _tvd_score(a: pd.Series, b: pd.Series) -> float:
    pa = a.astype(object).where(a.notna(), "__null__").value_counts(normalize=True)
    pb = b.astype(object).where(b.notna(), "__null__").value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    return float(1 - 0.5 * np.abs(pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).sum())


def _ks_score(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float(len(a) == len(b))
    return float(1 - stats.ks_2samp(a, b, method="asymp").statistic)


def _binned(x: np.ndarray, edges: np.ndarray) -> pd.Series:
    out = np.digitize(x, edges[1:-1]).astype(object)
    out[~np.isfinite(x)] = "__null__"
    return pd.Series(out)


@dataclass
class QualityReport:
    """Statistical fidelity. ``score`` is the mean of the property scores."""

    score: float
    properties: dict[str, float]
    details: dict[str, pd.DataFrame] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"score": self.score, "properties": self.properties}

    def __repr__(self) -> str:
        props = ", ".join(f"{k}: {v:.1%}" for k, v in self.properties.items())
        return f"QualityReport(score={self.score:.1%}; {props})"


def _table_quality(real: pd.DataFrame, synth: pd.DataFrame, table: Table, keys: set[str], max_pair_columns: int):
    cols = [c for c in table.modeled_columns if c in real.columns and c in synth.columns and c not in keys]
    shape_rows = []
    for col in cols:
        if _is_numeric(table, col):
            score = _ks_score(_numeric(real[col], table, col), _numeric(synth[col], table, col))
            metric = "KSComplement"
        else:
            score = _tvd_score(real[col], synth[col])
            metric = "TVComplement"
        shape_rows.append({"table": table.name, "column": col, "metric": metric, "score": score})

    pair_rows = []
    pair_cols = cols[:max_pair_columns]
    prepared = {}
    for col in pair_cols:
        if _is_numeric(table, col):
            r, s = _numeric(real[col], table, col), _numeric(synth[col], table, col)
            finite = r[np.isfinite(r)]
            edges = np.unique(np.quantile(finite, np.linspace(0, 1, 11))) if len(finite) else np.array([0, 1])
            prepared[col] = ("num", r, s, _binned(r, edges), _binned(s, edges))
        else:
            r = real[col].astype(object).where(real[col].notna(), "__null__").reset_index(drop=True)
            s = synth[col].astype(object).where(synth[col].notna(), "__null__").reset_index(drop=True)
            prepared[col] = ("cat", None, None, r, s)
    for a, b in combinations(pair_cols, 2):
        ka, ra, sa, rba, sba = prepared[a]
        kb, rb, sb, rbb, sbb = prepared[b]
        if ka == kb == "num":
            rc = pd.Series(ra).corr(pd.Series(rb))
            sc = pd.Series(sa).corr(pd.Series(sb))
            if np.isnan(rc) or np.isnan(sc):
                continue
            score, metric = float(1 - abs(rc - sc) / 2), "CorrelationSimilarity"
        else:
            jr = (rba.astype(str) + "\x1f" + rbb.astype(str)).value_counts(normalize=True)
            js = (sba.astype(str) + "\x1f" + sbb.astype(str)).value_counts(normalize=True)
            idx = jr.index.union(js.index)
            score = float(1 - 0.5 * np.abs(jr.reindex(idx, fill_value=0) - js.reindex(idx, fill_value=0)).sum())
            metric = "ContingencySimilarity"
        pair_rows.append({"table": table.name, "column_1": a, "column_2": b, "metric": metric, "score": score})
    return shape_rows, pair_rows


def evaluate_quality(
    real: Data, synthetic: Data, metadata: Metadata | None = None, max_pair_columns: int = 40
) -> QualityReport:
    """Score how closely the synthetic data reproduces the real data's statistics."""
    real_t, synth_t, md = _as_tables(real, synthetic, metadata)
    shapes, pairs, cards = [], [], []
    for name, table in md.tables.items():
        keys = set(md.foreign_keys(name)) | ({table.primary_key} if table.primary_key else set())
        s, p = _table_quality(real_t[name], synth_t[name], table, keys, max_pair_columns)
        shapes.extend(s)
        pairs.extend(p)
    for rel in md.relationships:
        rc = real_t[rel.child_table][rel.child_foreign_key].value_counts()
        sc = synth_t[rel.child_table][rel.child_foreign_key].value_counts()
        r_counts = real_t[rel.parent_table][rel.parent_primary_key].map(rc).fillna(0).to_numpy(dtype=float)
        s_counts = synth_t[rel.parent_table][rel.parent_primary_key].map(sc).fillna(0).to_numpy(dtype=float)
        cards.append({
            "parent_table": rel.parent_table, "child_table": rel.child_table,
            "metric": "CardinalityShape", "score": _ks_score(r_counts, s_counts),
        })
    details = {
        "Column Shapes": pd.DataFrame(shapes),
        "Column Pair Trends": pd.DataFrame(pairs),
    }
    properties = {
        "Column Shapes": float(np.mean([r["score"] for r in shapes])) if shapes else 1.0,
        "Column Pair Trends": float(np.mean([r["score"] for r in pairs])) if pairs else 1.0,
    }
    if cards:
        details["Cardinality"] = pd.DataFrame(cards)
        properties["Cardinality"] = float(np.mean([r["score"] for r in cards]))
    return QualityReport(float(np.mean(list(properties.values()))), properties, details)


# ---------------------------------------------------------------------- diagnostic


@dataclass
class DiagnosticReport:
    """Structural checks. ``passed`` is False if any hard check fails."""

    checks: pd.DataFrame

    @property
    def passed(self) -> bool:
        hard = self.checks[self.checks["hard"]]
        return bool((hard["score"] >= 1.0).all())

    @property
    def score(self) -> float:
        return float(self.checks["score"].mean()) if len(self.checks) else 1.0

    def failures(self) -> pd.DataFrame:
        return self.checks[self.checks["score"] < 1.0]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "score": self.score, "checks": self.checks.to_dict("records")}

    def __repr__(self) -> str:
        return f"DiagnosticReport(passed={self.passed}, score={self.score:.1%}, checks={len(self.checks)})"


def run_diagnostic(real: Data, synthetic: Data, metadata: Metadata | None = None) -> DiagnosticReport:
    """Check keys, foreign keys, column presence, ranges, categories and rules."""
    real_t, synth_t, md = _as_tables(real, synthetic, metadata)
    rows: list[dict[str, Any]] = []

    def add(table, check, column, score, hard):
        rows.append({"table": table, "check": check, "column": column, "score": float(score), "hard": hard})

    for name, table in md.tables.items():
        r, s = real_t[name], synth_t[name]
        missing = [c for c in r.columns if c not in s.columns]
        add(name, "columns present", None, 1.0 - len(missing) / max(1, len(r.columns)), True)
        if table.primary_key and table.primary_key in s.columns:
            pk = s[table.primary_key]
            add(name, "primary key unique and non-null", table.primary_key,
                float(pk.notna().all() and pk.is_unique), True)
        keys = set(md.foreign_keys(name)) | ({table.primary_key} if table.primary_key else set())
        for col in table.modeled_columns:
            if col in keys or col not in s.columns:
                continue
            if _is_numeric(table, col):
                rv, sv = _numeric(r[col], table, col), _numeric(s[col], table, col)
                rv, sv = rv[np.isfinite(rv)], sv[np.isfinite(sv)]
                if len(rv) and len(sv):
                    within = ((sv >= rv.min()) & (sv <= rv.max())).mean()
                    add(name, "within real min/max", col, within, False)
            else:
                allowed = set(r[col].dropna().unique())
                sv = s[col].dropna()
                if len(sv):
                    add(name, "categories seen in real data", col, sv.isin(allowed).mean(), False)
        for constraint in build_constraints(table.constraints):
            if all(c in s.columns for c in constraint.columns):
                constraint.fit(r, table)
                add(name, f"rule {constraint.to_dict()}", None, constraint.is_valid(s).mean(), True)
    for rel in md.relationships:
        fk = synth_t[rel.child_table][rel.child_foreign_key].dropna()
        parent_keys = set(synth_t[rel.parent_table][rel.parent_primary_key].dropna())
        score = fk.isin(parent_keys).mean() if len(fk) else 1.0
        add(rel.child_table, f"foreign key -> {rel.parent_table}", rel.child_foreign_key, score, True)
    return DiagnosticReport(pd.DataFrame(rows, columns=["table", "check", "column", "score", "hard"]))


# ---------------------------------------------------------------------- privacy


@dataclass
class PrivacyReport:
    exact_match_rate: float
    unique_match_rate: float
    dcr_median: float | None
    baseline_dcr_median: float | None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def dcr_ratio(self) -> float | None:
        if self.dcr_median is None or not self.baseline_dcr_median:
            return None
        return self.dcr_median / self.baseline_dcr_median

    def to_dict(self) -> dict:
        return {
            "exact_match_rate": self.exact_match_rate,
            "unique_match_rate": self.unique_match_rate,
            "dcr_median": self.dcr_median,
            "baseline_dcr_median": self.baseline_dcr_median,
            "dcr_ratio": self.dcr_ratio,
            **self.details,
        }

    def __repr__(self) -> str:
        ratio = "n/a" if self.dcr_ratio is None else f"{self.dcr_ratio:.2f}"
        return (
            f"PrivacyReport(exact_match_rate={self.exact_match_rate:.2%}, "
            f"unique_match_rate={self.unique_match_rate:.2%}, dcr_ratio={ratio})"
        )


def _encode(df: pd.DataFrame, table: Table, cols: list[str], ref: pd.DataFrame) -> np.ndarray:
    parts = []
    for col in cols:
        if _is_numeric(table, col):
            r = _numeric(ref[col], table, col)
            x = _numeric(df[col], table, col)
            mu, sd = np.nanmean(r), np.nanstd(r) or 1.0
            parts.append(np.nan_to_num((x - mu) / sd)[:, None])
        else:
            top = ref[col].astype(str).value_counts().index[:20]
            vals = df[col].astype(str).to_numpy()
            parts.append(np.stack([(vals == t).astype(float) for t in top], axis=1) if len(top) else np.zeros((len(df), 0)))
    return np.hstack(parts) if parts else np.zeros((len(df), 0))


def _min_dist(a: np.ndarray, b: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    out = np.empty(len(a))
    b2 = (b**2).sum(axis=1)
    for start in range(0, len(a), 512):
        chunk = a[start:start + 512]
        d = (chunk**2).sum(axis=1)[:, None] + b2[None, :] - 2 * chunk @ b.T
        d = np.sqrt(np.clip(d, 0, None))
        if exclude_self:
            idx = np.arange(start, start + len(chunk))
            d[np.arange(len(chunk)), idx] = np.inf
        out[start:start + len(chunk)] = d.min(axis=1)
    return out


def evaluate_privacy(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: Metadata | None = None,
    table_name: str | None = None,
    max_rows: int = 2000,
    seed: int = 0,
) -> PrivacyReport:
    """Measure copying of real rows (single table).

    * ``exact_match_rate`` - share of synthetic rows identical to some real row (modelled columns).
    * ``unique_match_rate`` - share of synthetic rows identical to a real row that is *unique*
      in the real data. This is the re-identification risk; it should be 0.
    * DCR - median distance from synthetic rows to their closest real row, compared with the
      same distance between real rows (``dcr_ratio`` near or above 1 means no memorisation).
    """
    if metadata is None:
        metadata = Metadata.detect_from_dataframe(real)
    table = metadata.get_table(table_name)
    keys = {table.primary_key} | set(metadata.foreign_keys(table.name))
    cols = [c for c in table.modeled_columns if c in synthetic.columns and c not in keys]
    if not cols:
        return PrivacyReport(0.0, 0.0, None, None, {"columns": []})
    r_hash = pd.util.hash_pandas_object(real[cols].astype(str), index=False)
    s_hash = pd.util.hash_pandas_object(synthetic[cols].astype(str), index=False)
    counts = r_hash.value_counts()
    matched = s_hash.isin(counts.index)
    unique_hashes = set(counts[counts == 1].index)
    exact = float(matched.mean()) if len(s_hash) else 0.0
    unique = float(s_hash.isin(unique_hashes).mean()) if len(s_hash) else 0.0

    rng = np.random.default_rng(seed)
    r_idx = rng.choice(len(real), min(len(real), max_rows * 2), replace=False)
    s_idx = rng.choice(len(synthetic), min(len(synthetic), max_rows), replace=False)
    ref = real.iloc[r_idx].reset_index(drop=True)
    R = _encode(ref, table, cols, real)
    S = _encode(synthetic.iloc[s_idx].reset_index(drop=True), table, cols, real)
    dcr = baseline = None
    if R.shape[1] and len(R) > 1 and len(S):
        dcr = float(np.median(_min_dist(S, R)))
        half = min(len(R), max_rows)
        baseline = float(np.median(_min_dist(R[:half], R, exclude_self=True)))
    return PrivacyReport(exact, unique, dcr, baseline, {"columns": cols})


# ---------------------------------------------------------------------- certificate


@dataclass
class Certificate:
    """Pass/fail release decision with the evidence behind it."""

    passed: bool
    reasons: list[str]
    quality: QualityReport
    diagnostic: DiagnosticReport
    privacy: dict[str, PrivacyReport]
    thresholds: dict[str, float]
    fingerprint: str
    created_at: str
    version: str = __version__

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "quality": self.quality.to_dict(),
            "diagnostic": {"passed": self.diagnostic.passed, "score": self.diagnostic.score,
                           "failures": self.diagnostic.failures().to_dict("records")},
            "privacy": {k: v.to_dict() for k, v in self.privacy.items()},
            "thresholds": self.thresholds,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "synthschema_version": self.version,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Certificate({status}; quality={self.quality.score:.1%}, "
            f"diagnostic={'ok' if self.diagnostic.passed else 'failed'}, reasons={self.reasons})"
        )


def _fingerprint(synth_t: Mapping[str, pd.DataFrame], md: Metadata) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(md.to_dict(), sort_keys=True, default=str).encode())
    for name in sorted(synth_t):
        h.update(name.encode())
        h.update(pd.util.hash_pandas_object(synth_t[name].astype(str), index=False).to_numpy().tobytes())
    return h.hexdigest()


def certify(
    real: Data,
    synthetic: Data,
    metadata: Metadata | None = None,
    *,
    min_quality: float = 0.80,
    max_unique_match_rate: float = 0.0,
    min_dcr_ratio: float | None = None,
) -> Certificate:
    """Decide whether synthetic data may be released.

    Hard stops: any failed structural check (keys, foreign keys, rules), any copy of a
    real row that is unique in the real data above ``max_unique_match_rate``, or quality
    below ``min_quality``.
    """
    real_t, synth_t, md = _as_tables(real, synthetic, metadata)
    quality = evaluate_quality(real_t, synth_t, md)
    diagnostic = run_diagnostic(real_t, synth_t, md)
    privacy = {name: evaluate_privacy(real_t[name], synth_t[name], md, name) for name in md.tables}
    reasons = []
    if not diagnostic.passed:
        for row in diagnostic.failures().query("hard").to_dict("records"):
            reasons.append(f"structure: {row['table']}: {row['check']} ({row['score']:.1%})")
    if quality.score < min_quality:
        reasons.append(f"quality {quality.score:.1%} is below {min_quality:.0%}")
    for name, rep in privacy.items():
        if rep.unique_match_rate > max_unique_match_rate:
            reasons.append(f"privacy: {name}: {rep.unique_match_rate:.2%} of rows copy a unique real row")
        if min_dcr_ratio is not None and rep.dcr_ratio is not None and rep.dcr_ratio < min_dcr_ratio:
            reasons.append(f"privacy: {name}: DCR ratio {rep.dcr_ratio:.2f} is below {min_dcr_ratio}")
    thresholds = {"min_quality": min_quality, "max_unique_match_rate": max_unique_match_rate}
    if min_dcr_ratio is not None:
        thresholds["min_dcr_ratio"] = min_dcr_ratio
    return Certificate(
        passed=not reasons,
        reasons=reasons,
        quality=quality,
        diagnostic=diagnostic,
        privacy=privacy,
        thresholds=thresholds,
        fingerprint=_fingerprint(synth_t, md),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
