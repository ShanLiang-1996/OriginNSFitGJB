from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import traceback
from typing import Callable, Literal

import pandas as pd

from .audit import AuditRecord, write_audit_outputs
from .data_loader import discover_files, read_table, strain_life_columns
from .gjb import GJBFit, fit_gjb18a
from .origin_client import OriginAutomationError, OriginClient, OriginGJBJob


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

    def resolved_patterns(self) -> tuple[str, ...]:
        return self.patterns or DEFAULT_PATTERNS

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


def run_analysis(
    config: AnalysisConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
) -> AnalysisRunResult:
    messages: list[str] = []
    failures: list[AnalysisTableFailure] = []
    output_dir = config.output_dir
    figures_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(progress_callback, "discover", "Discovering input files.")
    files = discover_files(config.input_dir, list(config.resolved_patterns()))
    if not files:
        emit_log(
            log_callback,
            f"No supported data files found in {config.input_dir}.",
            messages,
        )
        emit_progress(progress_callback, "failed", "No supported data files found.")
        return AnalysisRunResult(completed=False, messages=tuple(messages))

    summaries: list[dict[str, object]] = []
    fit_frames: list[pd.DataFrame] = []
    runout_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    level_frames: list[pd.DataFrame] = []
    extra_table_frames: dict[str, list[pd.DataFrame]] = {}
    origin_jobs: list[OriginGJBJob] = []
    audit_records: list[AuditRecord] = []

    emit_progress(progress_callback, "fit", "Running GJB analyses.", total=len(files))
    for path in files:
        for table in read_table(path):
            label = _safe_name(table.label)
            try:
                life_column, response_column = strain_life_columns(
                    table.frame,
                    config.life_column,
                    config.response_column,
                )
                fit = fit_gjb18a(
                    table.frame,
                    life_column,
                    response_column,
                    confidence=config.confidence,
                    fit_points=config.fit_points,
                    status_column=config.status_column,
                    level_column=config.level_column,
                    replicate_decimals=config.replicate_decimals,
                    outlier_mode=config.outlier_mode,
                    source_file=str(path),
                    source_sheet=table.sheet or "",
                    source_group=table.group or "",
                    source_label=label,
                )
            except Exception as exc:
                failures.append(AnalysisTableFailure(label=label, message=str(exc)))
                emit_log(
                    log_callback,
                    f"GJB analysis failed for {table.label}: {exc}",
                    messages,
                )
                continue

            title = str(table.group or table.sheet or path.stem)
            summaries.append(
                _gjb_summary_record(
                    fit,
                    path,
                    table.sheet,
                    table.group,
                    label,
                    life_column,
                    response_column,
                )
            )
            fit_frames.append(_with_metadata(fit.data, path, table.sheet, table.group, label))
            if fit.runout_data is not None and not fit.runout_data.empty:
                runout_frames.append(
                    _with_metadata(fit.runout_data, path, table.sheet, table.group, label)
                )
            curve_frames.append(_with_metadata(fit.curve, path, table.sheet, table.group, label))
            level_frames.append(_with_metadata(fit.level_stats, path, table.sheet, table.group, label))
            if fit.extra_tables:
                for name, frame in fit.extra_tables.items():
                    extra_table_frames.setdefault(name, []).append(
                        _with_metadata(frame, path, table.sheet, table.group, label)
                    )
            origin_jobs.append(OriginGJBJob(fit=fit, label=label, title=title))
            audit_records.append(
                AuditRecord(
                    label=label,
                    file=str(path),
                    sheet=table.sheet or "",
                    group=table.group or "",
                    fit=fit,
                )
            )

    if not summaries:
        emit_log(log_callback, "No GJB analyses were completed.", messages)
        emit_progress(progress_callback, "failed", "No GJB analyses were completed.")
        return AnalysisRunResult(
            completed=False,
            table_failures=tuple(failures),
            messages=tuple(messages),
        )

    emit_progress(progress_callback, "write_outputs", "Writing GJB CSV outputs.")
    summary_frame = pd.DataFrame(summaries)
    output_paths = _write_gjb_outputs(
        summary_frame,
        fit_frames,
        runout_frames,
        curve_frames,
        level_frames,
        extra_table_frames,
        output_dir,
    )

    audit_enabled = bool(config.audit or config.audit_workbook or config.audit_json)
    if audit_enabled:
        emit_progress(progress_callback, "write_audit", "Writing audit outputs.")
        audit_paths = write_audit_outputs(
            audit_records,
            summary_frame,
            config.resolved_audit_dir(),
            write_json=bool(config.audit or config.audit_json),
            write_workbook=bool(config.audit or config.audit_workbook),
        )
        output_paths.extend(audit_paths)

    origin_error: str | None = None
    if not config.dry_run:
        emit_progress(progress_callback, "origin", "Running Origin automation.")
        origin = None
        try:
            origin = OriginClient(visible=not config.hidden_origin).__enter__()
            saved_project, figure_records = origin.create_gjb_project(
                origin_jobs,
                summary_frame,
                config.resolved_project_path(),
                figures_dir=figures_dir,
                symbol_kind=config.symbol_kind,
                graph_template_path=config.graph_template_path,
                use_default_graph_template=not config.no_graph_template,
                include_linearized_graph=config.linearized_graph,
                show_runout_arrows=not config.no_runout_arrows,
            )
            _merge_origin_outputs(summaries, saved_project, figure_records)
            summary_frame = pd.DataFrame(summaries)
            summary_frame.to_csv(output_dir / "gjb_summary.csv", index=False, encoding="utf-8-sig")
            emit_log(log_callback, f"Wrote Origin project {saved_project}", messages)
        except OriginAutomationError as exc:
            origin_error = str(exc)
            emit_log(log_callback, f"Origin automation disabled: {exc}", messages)
            log_path = _write_origin_automation_log(output_dir, str(exc))
            output_paths.append(log_path)
            emit_log(log_callback, f"Wrote Origin automation log {log_path}", messages)
        except Exception:
            origin_error = traceback.format_exc()
            emit_log(
                log_callback,
                "Origin automation failed; see origin_automation.log for details.",
                messages,
            )
            log_path = _write_origin_automation_log(output_dir, origin_error)
            output_paths.append(log_path)
            emit_log(log_callback, f"Wrote Origin automation log {log_path}", messages)
        finally:
            if origin is not None:
                origin.__exit__(None, None, None)

    existing_paths = tuple(output_path for output_path in output_paths if output_path.exists())
    for output_path in existing_paths:
        emit_log(log_callback, f"Wrote {output_path}", messages)

    emit_progress(progress_callback, "complete", "Analysis complete.")
    return AnalysisRunResult(
        completed=True,
        output_paths=existing_paths,
        table_failures=tuple(failures),
        messages=tuple(messages),
        origin_error=origin_error,
    )


def _gjb_summary_record(
    fit: GJBFit,
    path: Path,
    sheet: str | None,
    group: str | None,
    label: str,
    life_column: str,
    response_column: str,
) -> dict[str, object]:
    result = fit.result
    return {
        "label": label,
        "file": str(path),
        "sheet": sheet or "",
        "group": group or "",
        "analysis": "GJB/Z 18A 9.3.2 simplified strain-life workflow",
        "model_name": result.model_name,
        "life_column": life_column,
        "response_column": response_column,
        "confidence": result.confidence,
        "parameter_count": result.parameter_count,
        "points": result.points,
        "degrees_of_freedom": result.degrees_of_freedom,
        "coefficient_a1": result.coefficient_a,
        "coefficient_a2": result.coefficient_b,
        "coefficient_a4": result.coefficient_c,
        "coefficient_a1_lower": result.coefficient_a_lower,
        "coefficient_a1_upper": result.coefficient_a_upper,
        "coefficient_a2_lower": result.coefficient_b_lower,
        "coefficient_a2_upper": result.coefficient_b_upper,
        "coefficient_a4_lower": result.coefficient_c_lower,
        "coefficient_a4_upper": result.coefficient_c_upper,
        "standard_error_a1": result.standard_error_a,
        "standard_error_a2": result.standard_error_b,
        "standard_error_a4": result.standard_error_c,
        "t_critical": result.t_critical,
        "f_band_critical": result.f_band_critical,
        "simultaneous_band_factor": result.simultaneous_band_factor,
        "sigma": result.sigma,
        "sigma_squared": result.sigma_squared,
        "residual_sum_squares": result.residual_sum_squares,
        "r2_log_life": result.r2,
        "rmse_log_life": result.rmse_log_life,
        "x_mean": result.x_mean,
        "y_mean": result.y_mean,
        "sxx": result.sxx,
        "sxy": result.sxy,
        "x_min": result.x_min,
        "x_max": result.x_max,
        "life_min": result.life_min,
        "life_max": result.life_max,
        "response_min": result.response_min,
        "response_max": result.response_max,
        "life_response_coefficient_a": result.life_response_coefficient_a,
        "life_response_coefficient_b": result.life_response_coefficient_b,
        "log_likelihood": result.log_likelihood,
        "negative_log_likelihood": result.negative_log_likelihood,
        "n_failure": result.n_failure,
        "n_runout": result.n_runout,
        "success": result.success,
        "optimizer_message": result.optimizer_message,
        "log_life_formula": result.log_life_formula,
        "life_formula": result.life_formula,
        "life_response_formula": result.life_response_formula,
        "response_life_formula": result.response_life_formula,
        "warnings": "; ".join(result.warnings),
        "origin_project": "",
        "engineering_figure": "",
        "linearized_figure": "",
    }


def _with_metadata(
    frame: pd.DataFrame,
    path: Path,
    sheet: str | None,
    group: str | None,
    label: str,
) -> pd.DataFrame:
    result = frame.copy()
    metadata = [
        ("label", label),
        ("file", str(path)),
        ("sheet", sheet or ""),
        ("group", group or ""),
    ]
    for column, value in reversed(metadata):
        result.insert(0, column, value)
    return result


def _write_gjb_outputs(
    summary_frame: pd.DataFrame,
    fit_frames: list[pd.DataFrame],
    runout_frames: list[pd.DataFrame],
    curve_frames: list[pd.DataFrame],
    level_frames: list[pd.DataFrame],
    extra_table_frames: dict[str, list[pd.DataFrame]],
    output_dir: Path,
) -> list[Path]:
    output_paths = [
        output_dir / "gjb_summary.csv",
        output_dir / "gjb_fit_data.csv",
        output_dir / "gjb_runout_data.csv",
        output_dir / "gjb_curve.csv",
        output_dir / "gjb_level_stats.csv",
    ]
    summary_frame.to_csv(output_paths[0], index=False, encoding="utf-8-sig")
    pd.concat(fit_frames, ignore_index=True).to_csv(
        output_paths[1],
        index=False,
        encoding="utf-8-sig",
    )
    if runout_frames:
        pd.concat(runout_frames, ignore_index=True).to_csv(
            output_paths[2],
            index=False,
            encoding="utf-8-sig",
        )
    elif output_paths[2].exists():
        output_paths[2].unlink()
    pd.concat(curve_frames, ignore_index=True).to_csv(
        output_paths[3],
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(level_frames, ignore_index=True).to_csv(
        output_paths[4],
        index=False,
        encoding="utf-8-sig",
    )

    extra_paths: list[Path] = []
    for name, frames in sorted(extra_table_frames.items()):
        if not frames:
            continue
        path = output_dir / f"gjb_{_safe_name(name).lower()}.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
        extra_paths.append(path)
    return output_paths + extra_paths


def _merge_origin_outputs(
    summaries: list[dict[str, object]],
    project_path: Path,
    figure_records: list[dict[str, str]],
) -> None:
    figures_by_label = {record["label"]: record for record in figure_records}
    for summary in summaries:
        summary["origin_project"] = str(project_path)
        figures = figures_by_label.get(str(summary["label"]), {})
        summary["engineering_figure"] = figures.get("engineering_figure", "")
        summary["linearized_figure"] = figures.get("linearized_figure", "")


def _write_origin_automation_log(output_dir: Path, message: str) -> Path:
    log_path = output_dir / "origin_automation.log"
    log_path.write_text(message.rstrip() + "\n", encoding="utf-8-sig")
    return log_path


def _safe_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\s]+', "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "gjb_fit"
