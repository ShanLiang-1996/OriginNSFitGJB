from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


DEFAULT_PATTERNS = ("*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls")
ProgressPhase = Literal[
    "discover",
    "fit",
    "write_outputs",
    "write_audit",
    "origin",
    "complete",
    "failed",
]


@dataclass(frozen=True)
class AnalysisConfig:
    input_dir: Path = Path("data")
    output_dir: Path = Path("output")
    patterns: tuple[str, ...] = DEFAULT_PATTERNS
    life_column: str | None = None
    response_column: str | None = None
    status_column: str | None = None
    level_column: str | None = None
    replicate_decimals: int = 8
    confidence: float = 0.95
    fit_points: int = 300
    symbol_kind: int = 2
    dry_run: bool = False
    audit: bool = False
    audit_workbook: bool = False
    audit_json: bool = False
    outlier_mode: str = "auto"
    audit_dir: Path | None = None
    hidden_origin: bool = False
    project_path: Path | None = None
    graph_template_path: Path | None = None
    no_graph_template: bool = False
    linearized_graph: bool = False
    no_runout_arrows: bool = False

    def resolved_audit_dir(self) -> Path:
        return self.audit_dir or (self.output_dir / "audit")

    def resolved_project_path(self) -> Path:
        return self.project_path or (self.output_dir / "gjb_analysis.opj")


@dataclass(frozen=True)
class AnalysisProgress:
    phase: ProgressPhase
    message: str
    current: int = 0
    total: int = 0


@dataclass(frozen=True)
class AnalysisTableFailure:
    label: str
    message: str


@dataclass(frozen=True)
class AnalysisRunResult:
    completed: bool
    output_paths: tuple[Path, ...] = ()
    table_failures: tuple[AnalysisTableFailure, ...] = ()
    messages: tuple[str, ...] = ()
    origin_error: str | None = None


ProgressCallback = Callable[[AnalysisProgress], None]
LogCallback = Callable[[str], None]


def emit_progress(
    callback: ProgressCallback | None,
    phase: ProgressPhase,
    message: str,
    *,
    current: int = 0,
    total: int = 0,
) -> None:
    if callback is not None:
        callback(AnalysisProgress(phase=phase, message=message, current=current, total=total))


def emit_log(callback: LogCallback | None, message: str, sink: list[str]) -> None:
    sink.append(message)
    if callback is not None:
        callback(message)
