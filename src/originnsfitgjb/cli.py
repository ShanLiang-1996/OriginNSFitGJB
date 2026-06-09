from __future__ import annotations

import argparse
from pathlib import Path
import re
import traceback

import pandas as pd

from .data_loader import discover_files, read_table, strain_life_columns
from .gjb import GJBFit, fit_gjb18a
from .origin_client import OriginAutomationError, OriginClient, OriginGJBJob


DEFAULT_PATTERNS = ["*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="origin-ns-fit-gjb",
        description="Batch GJB/Z 18A strain-life fitting and Origin project generation.",
    )
    parser.add_argument("--input", type=Path, default=Path("data"), help="Input data directory.")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory.")
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="File glob pattern. Can be passed multiple times.",
    )
    parser.add_argument("--life", "--x", dest="life", help="Fatigue life/N column.")
    parser.add_argument(
        "--response",
        "--y",
        dest="response",
        help="Strain or stress response column used as the GJB equivalent strain input.",
    )
    parser.add_argument("--status", help="Optional status column for failure/run-out rows.")
    parser.add_argument("--level", help="Optional nominal level column for grouped diagnostics.")
    parser.add_argument(
        "--replicate-decimals",
        type=int,
        default=8,
        help="Decimals used to group repeated GJB X levels when --level is omitted.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for reported intervals, for example 0.95 or 95.",
    )
    parser.add_argument("--fit-points", type=int, default=300, help="Number of curve points.")
    parser.add_argument("--symbol-kind", type=int, default=2, help="Origin symbol kind for data points.")
    parser.add_argument("--dry-run", action="store_true", help="Skip Origin automation.")
    parser.add_argument("--hidden-origin", action="store_true", help="Do not show Origin UI.")
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Output Origin project path. Defaults to output/gjb_analysis.opj.",
    )
    parser.add_argument(
        "--graph-template",
        type=Path,
        default=None,
        help="Optional Origin graph template (.otp/.otpu) for GJB output graphs.",
    )
    parser.add_argument(
        "--no-graph-template",
        action="store_true",
        help="Skip any bundled graph template and create graphs from Origin defaults.",
    )
    parser.add_argument(
        "--linearized-graph",
        action="store_true",
        help="Also create a GJB linearized diagnostic graph.",
    )
    parser.add_argument(
        "--no-runout-arrows",
        action="store_true",
        help="Hide run-out arrow annotations while keeping run-out scatter points visible.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_gjb_analysis(args)


def run_gjb_analysis(args: argparse.Namespace) -> int:
    input_dir: Path = args.input
    output_dir: Path = args.output
    figures_dir = output_dir / "figures"

    # Step 1 - Create output folders and discover all user-requested input files.
    output_dir.mkdir(parents=True, exist_ok=True)
    files = discover_files(input_dir, args.pattern or DEFAULT_PATTERNS)
    if not files:
        print(f"No supported data files found in {input_dir}.")
        return 1

    summaries: list[dict[str, object]] = []
    fit_frames: list[pd.DataFrame] = []
    runout_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    level_frames: list[pd.DataFrame] = []
    extra_table_frames: dict[str, list[pd.DataFrame]] = {}
    origin_jobs: list[OriginGJBJob] = []

    # Step 2 - Read every table and run the GJB/Z 18A fitting workflow.
    for path in files:
        for table in read_table(path):
            try:
                life_column, response_column = strain_life_columns(
                    table.frame,
                    args.life,
                    args.response,
                )
                fit = fit_gjb18a(
                    table.frame,
                    life_column,
                    response_column,
                    confidence=args.confidence,
                    fit_points=args.fit_points,
                    status_column=args.status,
                    level_column=args.level,
                    replicate_decimals=args.replicate_decimals,
                )
            except Exception as exc:
                print(f"GJB analysis failed for {table.label}: {exc}")
                continue

            label = _safe_name(table.label)
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

    if not summaries:
        print("No GJB analyses were completed.")
        return 1

    # Step 3 - Write CSV outputs before attempting Origin automation.
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

    # Step 4 - Optionally create an Origin project and exported figures.
    if not args.dry_run:
        project_path = args.project or (output_dir / "gjb_analysis.opj")
        origin = None
        try:
            origin = OriginClient(visible=not args.hidden_origin).__enter__()
            saved_project, figure_records = origin.create_gjb_project(
                origin_jobs,
                summary_frame,
                project_path,
                figures_dir=figures_dir,
                symbol_kind=args.symbol_kind,
                graph_template_path=args.graph_template,
                use_default_graph_template=not args.no_graph_template,
                include_linearized_graph=args.linearized_graph,
                show_runout_arrows=not args.no_runout_arrows,
            )
            _merge_origin_outputs(summaries, saved_project, figure_records)
            summary_frame = pd.DataFrame(summaries)
            summary_frame.to_csv(output_dir / "gjb_summary.csv", index=False, encoding="utf-8-sig")
            print(f"Wrote Origin project {saved_project}")
        except OriginAutomationError as exc:
            print(f"Origin automation disabled: {exc}")
            _write_origin_automation_log(output_dir, str(exc))
        except Exception:
            print("Origin automation failed; see origin_automation.log for details.")
            _write_origin_automation_log(output_dir, traceback.format_exc())
        finally:
            if origin is not None:
                origin.__exit__(None, None, None)

    # Step 5 - Report every file produced by the CSV stage.
    for output_path in output_paths:
        if output_path.exists():
            print(f"Wrote {output_path}")
    return 0


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


def _write_origin_automation_log(output_dir: Path, message: str) -> None:
    log_path = output_dir / "origin_automation.log"
    log_path.write_text(message.rstrip() + "\n", encoding="utf-8-sig")
    print(f"Wrote Origin automation log {log_path}")


def _safe_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\s]+', "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "gjb_fit"
