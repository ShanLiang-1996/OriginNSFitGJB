from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".xls", ".xlsx"}
LIFE_COLUMN_KEYWORDS = ("寿命", "life", "cycles", "cycle", "fatigue life", "n")
RESPONSE_COLUMN_KEYWORDS = (
    "塑性应变幅",
    "应变最大值",
    "最大应变",
    "最大应力",
    "应变幅",
    "应力幅",
    "应变",
    "应力",
    "strain maximum",
    "maximum strain",
    "max strain",
    "strain max",
    "strain amplitude",
    "stress maximum",
    "maximum stress",
    "max stress",
    "stress max",
    "stress amplitude",
    "epsilon max",
    "eps max",
    "emax",
    "strain",
    "stress",
    "amplitude",
    "s",
)


@dataclass(frozen=True)
class DataTable:
    source: Path
    frame: pd.DataFrame
    sheet: str | None = None
    group: str | None = None

    @property
    def label(self) -> str:
        parts = [self.source.stem]
        if self.sheet:
            parts.append(str(self.sheet))
        if self.group:
            parts.append(str(self.group))
        return "_".join(parts)


def discover_files(input_dir: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(input_dir.rglob(pattern))
    return sorted(
        {
            path
            for path in files
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        }
    )


def read_table(path: Path) -> list[DataTable]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        grouped_tables = read_grouped_delimited(path, delimiter=",")
        if grouped_tables:
            return grouped_tables
        return [DataTable(path, pd.read_csv(path))]
    if suffix == ".tsv":
        grouped_tables = read_grouped_delimited(path, delimiter="\t")
        if grouped_tables:
            return grouped_tables
        return [DataTable(path, pd.read_csv(path, sep="\t"))]
    if suffix == ".txt":
        return [DataTable(path, pd.read_csv(path, sep=None, engine="python"))]
    if suffix in {".xls", ".xlsx"}:
        sheets = pd.read_excel(path, sheet_name=None)
        return [DataTable(path, frame, sheet=name) for name, frame in sheets.items()]
    raise ValueError(f"Unsupported file type: {path}")


def read_grouped_delimited(path: Path, delimiter: str) -> list[DataTable]:
    rows = _read_delimited_rows(path, delimiter)
    tables: list[DataTable] = []
    group: str | None = None
    index = 0

    while index < len(rows):
        row = rows[index]
        if _is_blank_row(row):
            index += 1
            continue

        if _is_group_row(row):
            group = row[0].strip()
            index += 1
            continue

        if _looks_like_sn_header(row):
            headers = _unique_headers(row)
            index += 1
            data_rows: list[list[str]] = []
            while index < len(rows):
                current = rows[index]
                if _is_blank_row(current):
                    index += 1
                    continue
                if _is_group_row(current) or _looks_like_sn_header(current):
                    break
                data_rows.append(current)
                index += 1

            if data_rows:
                frame = _frame_from_rows(headers, data_rows)
                tables.append(DataTable(path, frame, group=group))
            continue

        index += 1

    return tables


def _read_delimited_rows(path: Path, delimiter: str) -> list[list[str]]:
    last_error: UnicodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return [
                    [cell.strip() for cell in row]
                    for row in csv.reader(handle, delimiter=delimiter)
                ]
        except UnicodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return []


def _is_blank_row(row: list[str]) -> bool:
    return not any(cell.strip() for cell in row)


def _is_group_row(row: list[str]) -> bool:
    non_empty = [cell for cell in row if cell.strip()]
    if len(non_empty) != 1:
        return False
    lowered = non_empty[0].lower()
    return not any(keyword in lowered for keyword in LIFE_COLUMN_KEYWORDS + RESPONSE_COLUMN_KEYWORDS)


def _looks_like_sn_header(row: list[str]) -> bool:
    lowered = [cell.lower().strip() for cell in row if cell.strip()]
    if len(lowered) < 2:
        return False
    has_life = any(any(keyword in cell for keyword in LIFE_COLUMN_KEYWORDS) for cell in lowered)
    has_response = any(
        any(keyword in cell for keyword in RESPONSE_COLUMN_KEYWORDS) for cell in lowered
    )
    return has_life and has_response


def _unique_headers(row: list[str]) -> list[str]:
    headers: list[str] = []
    seen: dict[str, int] = {}
    for index, cell in enumerate(row):
        header = cell.strip() or f"Column{index + 1}"
        count = seen.get(header, 0)
        seen[header] = count + 1
        headers.append(header if count == 0 else f"{header}_{count + 1}")
    return headers


def _frame_from_rows(headers: list[str], rows: list[list[str]]) -> pd.DataFrame:
    width = len(headers)
    normalized = [(row + [""] * width)[:width] for row in rows]
    frame = pd.DataFrame(normalized, columns=headers)
    for column in frame.columns:
        non_empty = frame[column].astype(str).str.strip().ne("")
        converted = pd.to_numeric(frame[column], errors="coerce")
        if non_empty.any() and converted[non_empty].notna().all():
            frame[column] = converted
    return frame


def numeric_xy_columns(frame: pd.DataFrame, x: str | None, y: str | None) -> tuple[str, str]:
    if x and y:
        return x, y

    numeric_columns = list(frame.select_dtypes(include="number").columns)
    if x:
        candidates = [column for column in numeric_columns if column != x]
        if not candidates:
            raise ValueError("Need at least one numeric Y column.")
        return x, y or candidates[0]
    if y:
        candidates = [column for column in numeric_columns if column != y]
        if not candidates:
            raise ValueError("Need at least one numeric X column.")
        return candidates[0], y
    if len(numeric_columns) < 2:
        raise ValueError("Need at least two numeric columns for X/Y fitting.")
    return numeric_columns[0], numeric_columns[1]


def sn_xy_columns(frame: pd.DataFrame, life: str | None, response: str | None) -> tuple[str, str]:
    life_column = life or _find_column_by_keywords(frame, LIFE_COLUMN_KEYWORDS)
    response_column = response or _find_column_by_keywords(frame, RESPONSE_COLUMN_KEYWORDS)

    numeric_columns = list(frame.select_dtypes(include="number").columns)
    if life_column is None or response_column is None:
        if len(numeric_columns) < 2:
            raise ValueError("Need at least two numeric columns for S-N curve fitting.")
        if life_column is None:
            medians = {
                column: float(np.nanmedian(np.abs(frame[column].to_numpy(dtype=float))))
                for column in numeric_columns
            }
            life_column = max(medians, key=medians.get)
        if response_column is None:
            response_column = next(column for column in numeric_columns if column != life_column)

    if life_column == response_column:
        raise ValueError("Life and response columns must be different.")
    if life_column not in frame.columns:
        raise ValueError(f"Life column not found: {life_column}")
    if response_column not in frame.columns:
        raise ValueError(f"Response column not found: {response_column}")
    return life_column, response_column


def _find_column_by_keywords(frame: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    normalized = {str(column).lower().strip(): column for column in frame.columns}
    for keyword in keywords:
        keyword = keyword.lower()
        for lowered, column in normalized.items():
            if lowered == keyword or (len(keyword) > 1 and keyword in lowered):
                return str(column)
    return None
