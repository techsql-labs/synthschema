"""Schema-only generation: realistic, key-consistent data from DDL and rules, with no seed data.

Example::

    from synthschema import SchemaSynthesizer

    ddl = '''
    CREATE TABLE customers (
        customer_id INT PRIMARY KEY,
        email VARCHAR(120) NOT NULL,
        segment VARCHAR(20) CHECK (segment IN ('Consumer', 'Corporate')),
        age INT CHECK (age BETWEEN 18 AND 90)
    );
    CREATE TABLE orders (
        order_id INT PRIMARY KEY,
        customer_id INT REFERENCES customers(customer_id),
        order_date DATE,
        ship_date DATE,
        amount DECIMAL(10, 2) CHECK (amount >= 0),
        CHECK (ship_date >= order_date)
    );
    '''
    data = SchemaSynthesizer.from_ddl(ddl).sample(num_rows=100, seed=1)
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constraints import Constraint, Inequality, OneOf, Range, build_constraints, parse_rule
from .generators import TextGenerator, generate_from_regex, generate_pii
from .metadata import _PII_NAME_RULES, Column, Metadata, Table

# ---------------------------------------------------------------------- DDL parsing

_CREATE_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:(?:GLOBAL\s+|LOCAL\s+)?TEMP(?:ORARY)?\s+)?TABLE\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?([\w\.\"`\[\]]+)\s*\(",
    re.IGNORECASE,
)
_TYPE_RE = re.compile(
    r"^([A-Za-z_][\w]*(?:\s+(?:varying|precision|without\s+time\s+zone|with\s+time\s+zone))?)"
    r"\s*(?:\(([^)]*)\))?",
    re.IGNORECASE,
)
_REF_RE = re.compile(r"REFERENCES\s+([\w\.\"`\[\]]+)\s*(?:\(\s*([\w\"`\[\]]+)\s*\))?", re.IGNORECASE)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", sql)


def _ident(name: str) -> str:
    name = name.strip().split(".")[-1]
    return name.strip('"`[]')


def _balanced(text: str, start: int) -> tuple[str, int]:
    """Return the text inside the parenthesis that opens at ``start`` and the end index."""
    depth, i, quote = 0, start, None
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i
        i += 1
    raise ValueError("Unbalanced parentheses in DDL")


def _split_top(text: str, sep: str = ",") -> list[str]:
    parts, depth, quote, cur = [], 0, None, []
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def _checks(text: str) -> list[str]:
    out = []
    for m in re.finditer(r"CHECK\s*\(", text, re.IGNORECASE):
        inner, _ = _balanced(text, m.end() - 1)
        out.append(inner)
    return out


def _quoted_values(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"'((?:[^']|'')*)'", text)]


def _column_from_type(name: str, sql_type: str, args: str | None) -> Column:
    t = sql_type.lower()
    lname = name.lower()
    if t in {"enum", "set"}:
        return Column("categorical", values=_quoted_values(args or ""))
    if t.startswith(("bool", "bit")):
        return Column("boolean")
    if t in {"date"}:
        return Column("datetime", representation="date")
    if t.startswith(("timestamp", "datetime", "smalldatetime", "datetime2", "datetimeoffset")):
        return Column("datetime", representation="datetime")
    if t.startswith(("time",)):
        return Column("datetime", representation="datetime")
    if t in {"int", "integer", "bigint", "smallint", "tinyint", "mediumint", "serial", "bigserial",
             "smallserial", "int2", "int4", "int8"}:
        return Column("numerical", representation="int", decimals=0)
    if t in {"decimal", "numeric", "number", "money", "smallmoney"}:
        scale = 2 if t in {"money", "smallmoney"} else 0
        if args:
            bits = [b.strip() for b in args.split(",")]
            if len(bits) > 1 and bits[1].isdigit():
                scale = int(bits[1])
        if t == "number" and not args:
            scale = 2
        return Column("numerical", representation="int" if scale == 0 else "float", decimals=scale)
    if t in {"float", "real", "double", "double precision", "float4", "float8"}:
        return Column("numerical", representation="float", decimals=2)
    if t in {"uuid", "uniqueidentifier"}:
        return Column("id")
    # character types
    for pattern, sdtype in _PII_NAME_RULES:
        if pattern.search(lname):
            return Column(sdtype)
    if lname in {"description", "comment", "comments", "notes", "note", "remarks", "review", "body", "message", "text"}:
        return Column("text")
    if t in {"text", "clob", "ntext", "longtext", "mediumtext"}:
        return Column("text")
    if lname.endswith(("_id", "_code", "_key", "_uuid")) or lname in {"code", "sku"}:
        return Column("id")
    if lname in _GEO_VALUES:
        return Column("categorical", values=list(_GEO_VALUES[lname]))
    if lname in {"company", "company_name", "employer"}:
        return Column("company")
    return Column("categorical")


def parse_ddl(ddl: str | Path) -> Metadata:
    """Parse ``CREATE TABLE`` statements into :class:`Metadata` (with column hints and rules).

    Understands column types, ``PRIMARY KEY``, ``REFERENCES`` / ``FOREIGN KEY``, ``NOT NULL``,
    ``ENUM(...)``, and ``CHECK`` rules (``IN``, ``BETWEEN``, comparisons with numbers, and
    column-to-column comparisons).
    """
    if isinstance(ddl, Path) or (isinstance(ddl, str) and ddl.strip().lower().endswith(".sql") and "\n" not in ddl):
        ddl = Path(ddl).read_text()
    sql = _strip_comments(str(ddl))
    md = Metadata()
    pending_fks: list[tuple[str, str, str, str | None]] = []
    for m in _CREATE_RE.finditer(sql):
        table_name = _ident(m.group(1))
        body, _ = _balanced(sql, m.end() - 1)
        table = md.add_table(table_name)
        nullable: dict[str, bool] = {}
        for item in _split_top(body):
            if not item:
                continue
            upper = item.upper()
            item_wo_constraint = re.sub(r"^CONSTRAINT\s+[\w\"`\[\]]+\s+", "", item, flags=re.IGNORECASE)
            upper_wo = item_wo_constraint.upper()
            if upper_wo.startswith("PRIMARY KEY"):
                cols = re.search(r"\(([^)]*)\)", item_wo_constraint)
                keys = [_ident(c) for c in cols.group(1).split(",")] if cols else []
                if len(keys) == 1:
                    table.primary_key = keys[0]
                continue
            if upper_wo.startswith("FOREIGN KEY"):
                cols = re.search(r"\(([^)]*)\)", item_wo_constraint)
                ref = _REF_RE.search(item_wo_constraint)
                if cols and ref:
                    fk_cols = [_ident(c) for c in cols.group(1).split(",")]
                    if len(fk_cols) == 1:
                        pending_fks.append((table_name, fk_cols[0], _ident(ref.group(1)),
                                            _ident(ref.group(2)) if ref.group(2) else None))
                continue
            if upper_wo.startswith("CHECK"):
                for expr in _checks(item_wo_constraint):
                    table.constraints.append(expr)
                continue
            if upper_wo.startswith(("UNIQUE", "INDEX", "KEY ", "FULLTEXT", "SPATIAL", "EXCLUDE")):
                continue
            # column definition
            parts = item.split(None, 1)
            col_name = _ident(parts[0])
            rest = parts[1] if len(parts) > 1 else "varchar"
            tm = _TYPE_RE.match(rest.strip())
            sql_type, args = (tm.group(1), tm.group(2)) if tm else ("varchar", None)
            column = _column_from_type(col_name, sql_type, args)
            if "PRIMARY KEY" in upper:
                table.primary_key = col_name
            ref = _REF_RE.search(rest)
            if ref:
                pending_fks.append((table_name, col_name, _ident(ref.group(1)),
                                    _ident(ref.group(2)) if ref.group(2) else None))
            if re.search(r"\bUNIQUE\b", upper):
                column.unique = True
            nullable[col_name] = "NOT NULL" not in upper and "PRIMARY KEY" not in upper
            table.columns[col_name] = column
            for expr in _checks(rest):
                table.constraints.append(expr)
        # keys are ids
        if table.primary_key and table.primary_key in table.columns:
            _make_id(table.columns, table.primary_key)
    for child, fk, parent, parent_col in pending_fks:
        if parent not in md.tables:
            raise ValueError(f"{child}.{fk} references unknown table {parent!r}")
        _make_id(md.tables[child].columns, fk)
        md.add_relationship(parent, child, fk, parent_col)
    for table in md.tables.values():
        _apply_rule_hints(table)
    md.validate()
    return md


def _make_id(columns: dict[str, Column], name: str) -> None:
    old = columns[name]
    rep = "int" if old.representation == "int" or old.sdtype == "numerical" else None
    columns[name] = Column("id", representation=rep, regex_format=old.regex_format)


def _apply_rule_hints(table: Table) -> None:
    """Fold single-column CHECK rules into column hints so generation respects them."""
    for spec in table.constraints:
        try:
            constraints = parse_rule(spec) if isinstance(spec, str) else build_constraints([spec])
        except ValueError:
            continue
        for c in constraints:
            if isinstance(c, OneOf) and c.column in table.columns:
                col = table.columns[c.column]
                if col.sdtype in {"categorical", "id", "text"}:
                    table.columns[c.column] = Column("categorical", values=c.values)
                else:
                    col.values = c.values
            elif isinstance(c, Range) and c.column in table.columns:
                col = table.columns[c.column]
                if c.min is not None:
                    col.min_value = c.min if col.min_value is None else max(col.min_value, c.min)
                if c.max is not None:
                    col.max_value = c.max if col.max_value is None else min(col.max_value, c.max)


# ---------------------------------------------------------------------- generation

_GEO_VALUES = {
    "country": ["United States", "United Kingdom", "Canada", "Germany", "India", "Australia", "France", "Brazil"],
    "region": ["North", "South", "East", "West"],
    "state": ["CA", "TX", "NY", "FL", "IL", "WA", "GA", "OH", "NC", "MA"],
    "city": ["Springfield", "Riverton", "Fairview", "Franklin", "Greenville", "Madison", "Salem", "Arlington"],
    "currency": ["USD", "EUR", "GBP", "INR", "CAD", "AUD"],
}

_NUMERIC_HINTS: list[tuple[re.Pattern, tuple[float, float]]] = [
    (re.compile(r"(^|_)age($|_)"), (18, 90)),
    (re.compile(r"(qty|quantity|units)"), (1, 20)),
    (re.compile(r"(rating|stars|score)"), (1, 5)),
    (re.compile(r"(percent|pct|rate|ratio|discount)"), (0, 0.5)),
    (re.compile(r"(price|amount|cost|total|revenue|balance|salary|income|fee|value)"), (1, 1000)),
    (re.compile(r"(^|_)year($|_)"), (2000, 2025)),
]

_DEFAULT_START = pd.Timestamp("2020-01-01")
_DEFAULT_END = pd.Timestamp("2025-12-31")


def _numeric_bounds(name: str, col: Column) -> tuple[float, float]:
    lo, hi = col.min_value, col.max_value
    if lo is None or hi is None:
        for pattern, (dlo, dhi) in _NUMERIC_HINTS:
            if pattern.search(name.lower()):
                lo = dlo if lo is None else lo
                hi = dhi if hi is None else hi
                break
    lo = 0 if lo is None else lo
    hi = (lo + 1000 if lo >= 0 else 1000) if hi is None else hi
    if hi < lo:
        lo, hi = hi, lo
    return float(lo), float(hi)


def _with_nulls(values: np.ndarray, col: Column, rng: np.random.Generator) -> np.ndarray:
    if col.null_rate:
        values = values.astype(object)
        values[rng.random(len(values)) < col.null_rate] = None
    return values


def _weights(col: Column, k: int) -> np.ndarray | None:
    if col.weights and len(col.weights) == k:
        w = np.asarray(col.weights, dtype=float)
        return w / w.sum()
    return None


def generate_column(name: str, col: Column, n: int, rng: np.random.Generator, locale=None,
                    table_name: str = "t") -> np.ndarray | pd.Series:
    """Generate ``n`` values for one column from its hints alone."""
    if col.values and col.sdtype in {"categorical", "numerical", "boolean"}:
        vals = np.array(col.values, dtype=object)
        return _with_nulls(vals[rng.choice(len(vals), size=n, p=_weights(col, len(vals)))], col, rng)
    if col.sdtype == "numerical":
        lo, hi = _numeric_bounds(name, col)
        x = lo + (hi - lo) * rng.beta(2.0, 2.0, size=n)
        if col.representation == "int" or col.decimals == 0:
            x = np.clip(np.round(x), np.ceil(lo), np.floor(hi)).astype(np.int64)
            return _with_nulls(x, col, rng) if col.null_rate else x
        x = np.round(x, col.decimals if col.decimals is not None else 2)
        return _with_nulls(x, col, rng)
    if col.sdtype == "boolean":
        p = _weights(col, 2)
        x = rng.random(n) < (p[0] if p is not None else 0.5)
        return _with_nulls(x, col, rng) if col.null_rate else x
    if col.sdtype == "datetime":
        start = pd.Timestamp(col.min_value) if col.min_value is not None else _DEFAULT_START
        end = pd.Timestamp(col.max_value) if col.max_value is not None else _DEFAULT_END
        secs = rng.uniform(start.value / 1e9, end.value / 1e9, size=n)
        dt = pd.to_datetime(pd.Series(secs), unit="s")
        dt = dt.dt.floor("D") if col.representation == "date" else dt.dt.round("s")
        if col.datetime_format:
            dt = dt.dt.strftime(col.datetime_format)
        out = dt.to_numpy()
        return _with_nulls(out, col, rng) if col.null_rate else out
    if col.sdtype == "categorical":
        k = 5
        vals = np.array([f"{name}_{i + 1}" for i in range(k)], dtype=object)
        return _with_nulls(vals[rng.integers(0, k, size=n)], col, rng)
    if col.sdtype == "text":
        gen = TextGenerator(locale)
        return _with_nulls(gen.generate(n, rng), col, rng)
    if col.sdtype == "id":
        return generate_keys(name, col, n, rng, table_name)
    return _with_nulls(generate_pii(col.sdtype, n, rng, locale), col, rng)


def generate_keys(name: str, col: Column, n: int, rng: np.random.Generator, table_name: str = "t") -> np.ndarray:
    if col.representation == "int":
        return np.arange(1, n + 1, dtype=np.int64)
    if col.regex_format:
        return generate_from_regex(col.regex_format, n, rng, unique=True)
    prefix = re.sub(r"[^A-Za-z]", "", table_name)[:3].upper() or "ID"
    width = max(6, len(str(n)))
    return np.array([f"{prefix}-{i:0{width}d}" for i in range(1, n + 1)], dtype=object)


class SchemaSynthesizer:
    """Generates data from metadata alone: no seed data, no learning step.

    Column hints (``values``, ``weights``, ``min_value``, ``max_value``, ``decimals``,
    ``null_rate``, ``regex_format``, ``datetime_format``) shape each column. Foreign keys
    always point at generated parent rows, and every rule in the metadata holds on every row.
    """

    def __init__(
        self,
        metadata: Metadata,
        *,
        constraints: Mapping[str, list[Constraint | Mapping | str]] | None = None,
        children_per_parent: float | Mapping[str, float] = 3.0,
        locale: str | list[str] | None = None,
    ) -> None:
        metadata.validate()
        self.metadata = metadata
        self.extra_constraints = dict(constraints or {})
        self.children_per_parent = children_per_parent
        self.locale = locale

    @classmethod
    def from_ddl(cls, ddl: str | Path, **kwargs: Any) -> SchemaSynthesizer:
        return cls(parse_ddl(ddl), **kwargs)

    def _children(self, table: str) -> float:
        if isinstance(self.children_per_parent, Mapping):
            return float(self.children_per_parent.get(table, 3.0))
        return float(self.children_per_parent)

    def sample(self, num_rows: int | Mapping[str, int] = 100, seed: int | None = None) -> dict[str, pd.DataFrame]:
        """Generate every table. ``num_rows`` is the size of root tables (or a dict per table);
        child tables default to ``children_per_parent`` rows per parent row."""
        rng = np.random.default_rng(seed)
        md = self.metadata
        sizes = dict(num_rows) if isinstance(num_rows, Mapping) else {}
        default_root = num_rows if isinstance(num_rows, int) else 100
        out: dict[str, pd.DataFrame] = {}
        for name in md.topological_order():
            table = md.tables[name]
            parents = md.parent_relationships(name)
            if name in sizes:
                n = int(sizes[name])
            elif parents:
                n = int(round(len(out[parents[0].parent_table]) * self._children(name)))
            else:
                n = int(default_root)
            fks = {r.child_foreign_key: r for r in md.relationships if r.child_table == name}
            data: dict[str, Any] = {}
            for col_name, col in table.columns.items():
                if col_name in fks:
                    continue
                if col_name == table.primary_key:
                    data[col_name] = generate_keys(col_name, col, n, rng, name)
                else:
                    data[col_name] = generate_column(col_name, col, n, rng, self.locale, name)
            df = pd.DataFrame(data, index=pd.RangeIndex(n))
            for fk, rel in fks.items():
                if rel.parent_table == name:
                    keys = df[table.primary_key].to_numpy()
                else:
                    keys = out[rel.parent_table][rel.parent_primary_key].to_numpy()
                if len(keys) == 0:
                    df[fk] = None
                    continue
                picked = keys[rng.integers(0, len(keys), size=n)]
                col = table.columns[fk]
                df[fk] = _with_nulls(picked, col, rng) if col.null_rate else picked
            df = df[list(table.columns)]
            df = self._enforce(name, table, df, rng)
            out[name] = df.reset_index(drop=True)
        return out

    def _enforce(self, name: str, table: Table, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        constraints = build_constraints(table.constraints) + build_constraints(self.extra_constraints.get(name))
        for c in constraints:
            if any(col not in df.columns for col in c.columns):
                continue
            c.fit(df, table)
            if isinstance(c, Inequality):
                df = _repair_inequality(c, df, table, rng)
            else:
                df = c.repair(df)
            if not c.is_valid(df).all():
                df = df[c.is_valid(df)]
        return df


def _repair_inequality(c: Inequality, df: pd.DataFrame, table: Table, rng: np.random.Generator) -> pd.DataFrame:
    """Rebuild ``high`` as ``low`` plus a positive gap wherever the rule fails."""
    bad = ~c.is_valid(df)
    if not bad.any():
        return df
    df = df.copy()
    low_col, high_col = table.columns.get(c.low), table.columns.get(c.high)
    if low_col is not None and low_col.sdtype == "datetime":
        low = pd.to_datetime(df.loc[bad, c.low])
        days = rng.integers(0 if not c.strict else 1, 15, size=int(bad.sum()))
        high = low + pd.to_timedelta(days, unit="D")
        if high_col is not None and high_col.representation == "date":
            high = high.dt.floor("D")
        fmt = high_col.datetime_format if high_col is not None else None
        df.loc[bad, c.high] = high.dt.strftime(fmt).to_numpy() if fmt else high.to_numpy()
    else:
        low = pd.to_numeric(df.loc[bad, c.low], errors="coerce").to_numpy(dtype=float)
        span = max(float(np.nanstd(pd.to_numeric(df[c.low], errors="coerce"))), 1.0)
        gap = rng.uniform(0 if not c.strict else 1e-6, span, size=len(low))
        high = low + gap
        if high_col is not None and (high_col.representation == "int" or high_col.decimals == 0):
            high = np.ceil(high)
        elif high_col is not None and high_col.decimals is not None:
            high = np.round(high, high_col.decimals)
        df[c.high] = df[c.high].astype(float) if not pd.api.types.is_float_dtype(df[c.high]) else df[c.high]
        df.loc[bad, c.high] = high
        if high_col is not None and (high_col.representation == "int" or high_col.decimals == 0):
            if df[c.high].notna().all():
                df[c.high] = df[c.high].astype(np.int64)
    return df
