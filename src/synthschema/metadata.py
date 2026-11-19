"""Metadata: the blueprint that describes tables, columns, keys, relationships and rules.

One metadata object drives everything in synthschema: profiling, generation, constraints and
evaluation all read the same description, so a rule is declared once.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import pandas as pd

#: Column types that are learned by the statistical model.
MODELED_SDTYPES = frozenset({"numerical", "categorical", "boolean", "datetime"})

#: Personally identifiable column types. These are never learned; values are always
#: generated from scratch (with Faker when it is installed).
PII_SDTYPES = frozenset(
    {
        "email",
        "phone_number",
        "name",
        "first_name",
        "last_name",
        "address",
        "street_address",
        "ssn",
        "credit_card_number",
        "ip_address",
        "url",
        "company",
        "postcode",
    }
)

#: Column types that are generated rather than learned.
GENERATED_SDTYPES = frozenset({"id", "text"}) | PII_SDTYPES

ALL_SDTYPES = MODELED_SDTYPES | GENERATED_SDTYPES


@dataclass
class Column:
    """Description of one column.

    Only ``sdtype`` is required. The other fields are optional hints. Learned synthesizers
    use the data itself; the schema-only synthesizer relies on these hints.
    """

    sdtype: str
    datetime_format: str | None = None
    regex_format: str | None = None
    representation: str | None = None  # "int", "float", "date" or "datetime"
    values: list | None = None
    weights: list | None = None
    min_value: Any = None
    max_value: Any = None
    decimals: int | None = None
    null_rate: float | None = None
    unique: bool | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if self.sdtype not in ALL_SDTYPES:
            raise ValueError(
                f"Unknown sdtype {self.sdtype!r}. Expected one of: {', '.join(sorted(ALL_SDTYPES))}"
            )

    @property
    def is_pii(self) -> bool:
        return self.sdtype in PII_SDTYPES

    @property
    def is_modeled(self) -> bool:
        return self.sdtype in MODELED_SDTYPES

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping | str) -> Column:
        if isinstance(data, str):
            return cls(sdtype=data)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown column field(s): {', '.join(sorted(unknown))}")
        return cls(**dict(data))


@dataclass
class Relationship:
    """A one-to-many link: each child row points at one parent row (or is null)."""

    parent_table: str
    parent_primary_key: str
    child_table: str
    child_foreign_key: str

    def to_dict(self) -> dict:
        return {
            "parent_table": self.parent_table,
            "parent_primary_key": self.parent_primary_key,
            "child_table": self.child_table,
            "child_foreign_key": self.child_foreign_key,
        }


@dataclass
class Table:
    name: str
    columns: dict[str, Column] = field(default_factory=dict)
    primary_key: str | None = None
    constraints: list = field(default_factory=list)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"columns": {n: c.to_dict() for n, c in self.columns.items()}}
        if self.primary_key:
            out["primary_key"] = self.primary_key
        if self.constraints:
            out["constraints"] = [
                c.to_dict() if hasattr(c, "to_dict") else c for c in self.constraints
            ]
        return out

    @property
    def modeled_columns(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.is_modeled]


class Metadata:
    """Tables, columns, keys, relationships and rules for one dataset.

    Build it by hand, detect it from data with :meth:`detect_from_dataframe` /
    :meth:`detect_from_dataframes`, parse it from SQL DDL with :meth:`from_ddl`, or load a
    JSON/YAML blueprint with :meth:`load`.
    """

    def __init__(self) -> None:
        self.tables: dict[str, Table] = {}
        self.relationships: list[Relationship] = []

    # ------------------------------------------------------------------ building
    def add_table(self, name: str, primary_key: str | None = None) -> Table:
        if name in self.tables:
            raise ValueError(f"Table {name!r} already exists")
        table = Table(name=name, primary_key=primary_key)
        self.tables[name] = table
        return table

    def add_column(self, table: str, name: str, sdtype: str, **hints: Any) -> Column:
        column = Column(sdtype=sdtype, **hints)
        self.get_table(table).columns[name] = column
        return column

    def update_column(self, table: str, name: str, **changes: Any) -> Column:
        t = self.get_table(table)
        if name not in t.columns:
            raise KeyError(f"Column {name!r} not found in table {table!r}")
        data = t.columns[name].to_dict()
        data.update(changes)
        t.columns[name] = Column.from_dict(data)
        return t.columns[name]

    def set_primary_key(self, table: str, column: str) -> None:
        t = self.get_table(table)
        if column not in t.columns:
            raise KeyError(f"Column {column!r} not found in table {table!r}")
        if t.columns[column].sdtype != "id":
            self.update_column(table, column, sdtype="id")
        t.primary_key = column

    def add_relationship(
        self,
        parent_table: str,
        child_table: str,
        child_foreign_key: str,
        parent_primary_key: str | None = None,
    ) -> Relationship:
        parent = self.get_table(parent_table)
        pk = parent_primary_key or parent.primary_key
        if pk is None:
            raise ValueError(f"Parent table {parent_table!r} has no primary key")
        rel = Relationship(parent_table, pk, child_table, child_foreign_key)
        self.relationships.append(rel)
        child = self.get_table(child_table)
        if child_foreign_key in child.columns and child.columns[child_foreign_key].sdtype != "id":
            self.update_column(child_table, child_foreign_key, sdtype="id")
        return rel

    def add_constraint(self, table: str, constraint: Any) -> None:
        """Attach a rule to a table: a dict, a rule string such as ``"ship_date >= order_date"``,
        or a :class:`synthschema.constraints.Constraint` object."""
        self.get_table(table).constraints.append(constraint)

    # ------------------------------------------------------------------ access
    def get_table(self, name: str | None = None) -> Table:
        if name is None:
            if len(self.tables) != 1:
                raise ValueError("Metadata has several tables; pass a table name")
            return next(iter(self.tables.values()))
        if name not in self.tables:
            raise KeyError(f"Table {name!r} not found. Tables: {list(self.tables)}")
        return self.tables[name]

    @property
    def table_names(self) -> list[str]:
        return list(self.tables)

    def foreign_keys(self, table: str) -> list[str]:
        return [r.child_foreign_key for r in self.relationships if r.child_table == table]

    def parent_relationships(self, table: str) -> list[Relationship]:
        return [
            r for r in self.relationships if r.child_table == table and r.parent_table != table
        ]

    def child_relationships(self, table: str) -> list[Relationship]:
        return [
            r for r in self.relationships if r.parent_table == table and r.child_table != table
        ]

    def topological_order(self) -> list[str]:
        """Tables ordered so that every parent comes before its children."""
        indegree = {t: 0 for t in self.tables}
        children: dict[str, list[str]] = {t: [] for t in self.tables}
        for r in self.relationships:
            if r.parent_table == r.child_table:
                continue
            indegree[r.child_table] += 1
            children[r.parent_table].append(r.child_table)
        queue = deque(t for t in self.tables if indegree[t] == 0)
        order: list[str] = []
        while queue:
            t = queue.popleft()
            order.append(t)
            for c in children[t]:
                indegree[c] -= 1
                if indegree[c] == 0:
                    queue.append(c)
        if len(order) != len(self.tables):
            raise ValueError("Relationships contain a cycle; synthschema needs a parent-first order")
        return order

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        """Raise ``ValueError`` if keys or relationships are inconsistent."""
        errors = []
        for t in self.tables.values():
            if t.primary_key is not None:
                if t.primary_key not in t.columns:
                    errors.append(f"{t.name}: primary key {t.primary_key!r} is not a column")
                elif t.columns[t.primary_key].sdtype != "id":
                    errors.append(f"{t.name}: primary key {t.primary_key!r} must have sdtype 'id'")
        for r in self.relationships:
            if r.parent_table not in self.tables:
                errors.append(f"relationship: unknown parent table {r.parent_table!r}")
                continue
            if r.child_table not in self.tables:
                errors.append(f"relationship: unknown child table {r.child_table!r}")
                continue
            parent = self.tables[r.parent_table]
            if parent.primary_key != r.parent_primary_key:
                errors.append(
                    f"relationship {r.parent_table}->{r.child_table}: "
                    f"{r.parent_primary_key!r} is not the primary key of {r.parent_table!r}"
                )
            if r.child_foreign_key not in self.tables[r.child_table].columns:
                errors.append(
                    f"relationship {r.parent_table}->{r.child_table}: "
                    f"foreign key {r.child_foreign_key!r} is not a column"
                )
        if errors:
            raise ValueError("Invalid metadata:\n  " + "\n  ".join(errors))
        self.topological_order()

    def validate_data(self, data: pd.DataFrame | Mapping[str, pd.DataFrame]) -> None:
        if isinstance(data, pd.DataFrame):
            data = {self.get_table().name: data}
        errors = []
        for name, table in self.tables.items():
            if name not in data:
                errors.append(f"missing data for table {name!r}")
                continue
            missing = [c for c in table.columns if c not in data[name].columns]
            if missing:
                errors.append(f"{name}: columns missing from data: {missing}")
        if errors:
            raise ValueError("Data does not match metadata:\n  " + "\n  ".join(errors))

    # ------------------------------------------------------------------ detection
    @classmethod
    def detect_from_dataframe(cls, data: pd.DataFrame, table_name: str = "table") -> Metadata:
        """Infer column types and the primary key of one table."""
        md = cls()
        md._add_detected_table(table_name, data)
        return md

    @classmethod
    def detect_from_dataframes(cls, data: Mapping[str, pd.DataFrame]) -> Metadata:
        """Infer tables, keys and foreign-key relationships from several tables."""
        md = cls()
        for name, df in data.items():
            md._add_detected_table(name, df)
        for child_name, child_df in data.items():
            child = md.tables[child_name]
            for col in child_df.columns:
                if col == child.primary_key:
                    continue
                for parent_name, parent in md.tables.items():
                    if parent_name == child_name or parent.primary_key is None:
                        continue
                    if not _fk_name_matches(col, parent_name, parent.primary_key):
                        continue
                    child_vals = child_df[col].dropna()
                    if child_vals.empty:
                        continue
                    parent_keys = set(data[parent_name][parent.primary_key].dropna())
                    if child_vals.isin(parent_keys).mean() >= 0.9:
                        md.add_relationship(parent_name, child_name, col)
                        break
        return md

    def _add_detected_table(self, name: str, df: pd.DataFrame) -> None:
        table = self.add_table(name)
        for col in df.columns:
            table.columns[str(col)] = detect_column(df[col], str(col))
        table.primary_key = _detect_primary_key(name, df, table)

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        return {
            "tables": {n: t.to_dict() for n, t in self.tables.items()},
            "relationships": [r.to_dict() for r in self.relationships],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> Metadata:
        md = cls()
        for name, tdata in data.get("tables", {}).items():
            table = md.add_table(name, primary_key=tdata.get("primary_key"))
            for cname, cdata in tdata.get("columns", {}).items():
                table.columns[cname] = Column.from_dict(cdata)
            table.constraints = list(tdata.get("constraints", []))
        for rdata in data.get("relationships", []):
            md.relationships.append(Relationship(**rdata))
        md.validate()
        return md

    def save(self, path: str | Path) -> None:
        """Write the blueprint as JSON, or YAML if the path ends in .yaml/.yml."""
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            yaml = _import_yaml()
            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False))
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> Metadata:
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in {".yaml", ".yml"}:
            return cls.from_dict(_import_yaml().safe_load(text))
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_ddl(cls, ddl: str | Path) -> Metadata:
        """Build metadata from SQL ``CREATE TABLE`` statements (a string or a .sql path)."""
        from .schema import parse_ddl

        return parse_ddl(ddl)

    def __repr__(self) -> str:
        parts = []
        for t in self.tables.values():
            parts.append(f"{t.name}({len(t.columns)} columns, pk={t.primary_key})")
        return f"Metadata(tables=[{', '.join(parts)}], relationships={len(self.relationships)})"


# ---------------------------------------------------------------------- detection helpers

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
]

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")

_PII_NAME_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"e[-_ ]?mail"), "email"),
    (re.compile(r"(phone|mobile|cell[-_ ]?no|fax|telephone|^tel$)"), "phone_number"),
    (re.compile(r"^(first|given)[-_ ]?name$|^fname$"), "first_name"),
    (re.compile(r"^(last|family)[-_ ]?name$|^surname$|^lname$"), "last_name"),
    (re.compile(r"^(full[-_ ]?)?name$|^(customer|person|employee|contact|patient|client)[-_ ]?name$"), "name"),
    (re.compile(r"(street|address|addr)"), "address"),
    (re.compile(r"(^ssn$|social[-_ ]?security|national[-_ ]?id|^nin$)"), "ssn"),
    (re.compile(r"(credit[-_ ]?card|card[-_ ]?number|^ccn$|^pan$)"), "credit_card_number"),
    (re.compile(r"(ip[-_ ]?address|^ip$|^ipv4$)"), "ip_address"),
    (re.compile(r"(^url$|website|homepage)"), "url"),
]


def _is_id_name(name: str) -> bool:
    lname = name.lower()
    return (
        lname in {"id", "uuid", "guid", "key", "pk"}
        or lname.endswith(("_id", "_key", "_uuid", "_guid", "_pk"))
        or (name.endswith("Id") or name.endswith("ID"))
    )


def _fk_name_matches(col: str, parent_table: str, parent_pk: str) -> bool:
    lcol, lparent, lpk = col.lower(), parent_table.lower(), parent_pk.lower()
    if lcol == lpk and lpk != "id":
        return True
    stems = {lparent, lparent.rstrip("s"), lparent[:-3] + "y" if lparent.endswith("ies") else lparent}
    return any(lcol in (f"{s}_id", f"{s}id", f"{s}_{lpk}") for s in stems)


def _is_integer_valued(values: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(values):
        return True
    try:
        arr = values.astype(float).to_numpy()
    except (TypeError, ValueError):
        return False
    return bool(len(arr) > 0 and (arr == arr.round()).all())


def detect_datetime_format(values: pd.Series) -> str | None | bool:
    """Return a strptime format that parses every value, ``None`` for ISO-8601 strings that
    need no explicit format, or ``False`` if the values are not dates."""
    sample = values.astype(str).head(200)
    if not sample.str.contains(r"\d", regex=True).all():
        return False
    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        if parsed.notna().all():
            return fmt
    if sample.str.match(r"^\d{4}-\d{2}-\d{2}[T ]").all():
        parsed = pd.to_datetime(sample, format="ISO8601", errors="coerce")
        if parsed.notna().all():
            return None
    return False


def detect_column(series: pd.Series, name: str) -> Column:
    """Infer the :class:`Column` description of one pandas Series."""
    from .generators import infer_regex

    nonnull = series.dropna()
    n, nn = len(series), len(nonnull)
    if nn == 0:
        return Column("categorical")
    nunique = nonnull.nunique()
    is_unique = nunique == nn == n

    if pd.api.types.is_bool_dtype(series):
        return Column("boolean")
    if pd.api.types.is_datetime64_any_dtype(series):
        return Column("datetime")
    if pd.api.types.is_numeric_dtype(series):
        integer = _is_integer_valued(nonnull)
        lname = name.lower()
        if _is_id_name(name) and integer:
            if is_unique or nunique > 0.5 * nn:
                return Column("id", representation="int")
            return Column("categorical")
        for pattern, sdtype in _PII_NAME_RULES:
            if sdtype in {"phone_number", "ssn", "credit_card_number"} and pattern.search(lname):
                return Column(sdtype)
        return Column("numerical", representation="int" if integer else "float")

    strings = nonnull.astype(str)
    sample = strings.sample(min(nn, 500), random_state=0) if nn > 500 else strings
    if sample.str.match(_EMAIL_RE).mean() >= 0.9:
        return Column("email")
    lname = name.lower()
    for pattern, sdtype in _PII_NAME_RULES:
        if pattern.search(lname):
            return Column(sdtype)

    fmt = detect_datetime_format(sample)
    if fmt is not False:
        return Column("datetime", datetime_format=fmt)

    ratio = nunique / nn
    if _is_id_name(name) and (is_unique or ratio > 0.5):
        return Column("id", regex_format=infer_regex(strings))
    avg_len = sample.str.len().mean()
    space_share = sample.str.contains(" ").mean()
    if ratio > 0.5 and nunique > 20 and avg_len >= 20 and space_share > 0.5:
        return Column("text")
    if ratio > 0.9 and nunique > 50:
        if space_share > 0.3:
            return Column("text")
        return Column("id", regex_format=infer_regex(strings))
    return Column("categorical")


def _detect_primary_key(table_name: str, df: pd.DataFrame, table: Table) -> str | None:
    candidates = []
    for col, meta in table.columns.items():
        if meta.sdtype != "id":
            continue
        s = df[col]
        if s.isna().any() or not s.is_unique:
            continue
        lcol = col.lower()
        stem = table_name.lower().rstrip("s")
        rank = 0 if lcol in {"id", f"{stem}_id", f"{table_name.lower()}_id", f"{stem}id"} else 1
        candidates.append((rank, list(df.columns).index(col), col))
    if not candidates:
        return None
    return sorted(candidates)[0][2]


def _import_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("YAML blueprints need PyYAML: pip install 'synthschema[yaml]'") from exc
    return yaml


def as_metadata_dict(tables: Iterable[Table]) -> dict:
    return {t.name: t.to_dict() for t in tables}
