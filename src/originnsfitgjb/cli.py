from __future__ import annotations

import argparse
from pathlib import Path
import re
import traceback

import pandas as pd

from .data_loader import discover_files, read_table, sn_xy_columns
from .e739 import E739Fit, fit_e739
from .fitting import fit_sn_power_law
from .origin_client import OriginAutomationError, OriginClient, OriginE739Job


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="origin-ns-fit-gjb",
        description=(
            "Batch read strain-life data, run the simplified GJB/Z 18A "
            "9.3.2 workflow, and build Origin projects."
        ),
    )
    parser.add_argument(
        "--analysis",
        choices=("e739", "power"),
        default="e739",
        help="Analysis workflow. Defaults to ASTM E739.",
    )
    parser.add_argument("--input", type=Path, default=Path("data"), help="Input data directory.")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory.")
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="File glob pattern. Can be passed multiple times.",
    )
    parser.add_argument("--life", "--x", dest="life", help="Fatigue life/N column. Defaults to 寿命.")
    parser.add_argument(
        "--response",
        "--y",
        dest="response",
        help="Stress/strain response column. Defaults to common strain/stress names.",
    )
    parser.add_argument("--fit-points", type=int, default=300, help="Number of curve points.")
    parser.add_argument("--symbol-kind", type=int, default=2, help="Origin symbol kind for data points.")
    parser.add_argument("--dry-run", action="store_true", help="Skip Origin automation.")
    parser.add_argument("--hidden-origin", action="store_true", help="Do not show Origin UI.")

    parser.add_argument(
        "--e739-x-transform",
        choices=("log", "linear"),
        default="log",
        help="E739 independent variable transform: log means X=log10(response).",
    )
    parser.add_argument(
        "--e739-model",
        choices=(
            "standard",
            "shifted-log",
            "threshold_log_mle",
            "log_threshold_censored_mle",
            "gjb932-strain",
        ),
        default="gjb932-strain",
        help=(
            "Model. gjb932-strain is the project default; standard: log10(N)=A+B*X; shifted-log: "
            "log10(N)=A+B*log10(response-C); threshold_log_mle handles run-out "
            "rows as right-censored observations; gjb932-strain follows the "
            "simplified GJB/Z 18A 9.3.2 Formula 136 workflow."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for E739 intervals/bands, for example 0.95 or 95.",
    )
    parser.add_argument(
        "--status",
        help=(
            "Optional status column. Standard and shifted-log models exclude "
            "run-out/suspended rows from fitting but keep them for export/plots; "
            "threshold_log_mle treats them as right-censored observations."
        ),
    )
    parser.add_argument(
        "--level",
        help="Optional nominal level column for E739 replicate/linearity testing.",
    )
    parser.add_argument(
        "--replicate-decimals",
        type=int,
        default=8,
        help="Decimals used to group replicated E739 X levels when --level is omitted.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Output Origin project path. Defaults to output/e739_analysis.opj.",
    )
    parser.add_argument(
        "--graph-template",
        type=Path,
        default=None,
        help="Optional Origin graph template (.otp/.otpu) for E739 output graphs.",
    )
    parser.add_argument(
        "--no-graph-template",
        action="store_true",
        help="Skip the bundled E739 graph template for older Origin versions.",
    )
    parser.add_argument(
        "--linearized-graph",
        action="store_true",
        help="Also create E739 linearized graphs. By default only engineering graphs are created.",
    )
    parser.add_argument(
        "--no-runout-arrows",
        action="store_true",
        help="Hide run-out arrow annotations while keeping run-out scatter points visible.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.analysis == "power":
        return _run_power_analysis(args)
    return _run_e739_analysis(args)


def _run_e739_analysis(args: argparse.Namespace) -> int:
    input_dir: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    files = _discover_input_files(input_dir, args.pattern)
    if not files:
        print(f"No supported data files found in {input_dir}.")
        return 1

    summaries: list[dict[str, object]] = []
    transformed_frames: list[pd.DataFrame] = []
    runout_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    level_frames: list[pd.DataFrame] = []
    extra_table_frames: dict[str, list[pd.DataFrame]] = {}
    origin_jobs: list[OriginE739Job] = []

    for path in files:
        for table in read_table(path):
            try:
                life_column, response_column = sn_xy_columns(table.frame, args.life, args.response)
                fit = fit_e739(
                    table.frame,
                    life_column,
                    response_column,
                    model=args.e739_model,
                    x_transform=args.e739_x_transform,
                    confidence=args.confidence,
                    fit_points=args.fit_points,
                    status_column=args.status,
                    level_column=args.level,
                    replicate_decimals=args.replicate_decimals,
                )
            except Exception as exc:
                print(f"E739 analysis failed for {table.label}: {exc}")
                continue

            label = _safe_name(table.label)
            title = str(table.group or table.sheet or path.stem)
            summaries.append(
                _e739_summary_record(
                    fit,
                    path,
                    table.sheet,
                    table.group,
                    label,
                    life_column,
                    response_column,
                )
            )
            transformed_frames.append(_with_metadata(fit.data, path, table.sheet, table.group, label))
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
            origin_jobs.append(OriginE739Job(fit=fit, label=label, title=title))

    if not summaries:
        print("No E739 analyses were completed.")
        return 1

    summary_path = output_dir / "e739_summary.csv"
    transformed_path = output_dir / "e739_transformed_data.csv"
    runout_path = output_dir / "e739_runout_data.csv"
    curves_path = output_dir / "e739_curve_bands.csv"
    levels_path = output_dir / "e739_level_stats.csv"

    summary_frame = pd.DataFrame(summaries)
    extra_output_paths = _write_e739_outputs(
        summary_frame,
        transformed_frames,
        runout_frames,
        curve_frames,
        level_frames,
        extra_table_frames,
        output_dir,
    )

    if not args.dry_run:
        project_path = args.project or (output_dir / "e739_analysis.opj")
        origin = None
        try:
            origin = OriginClient(visible=not args.hidden_origin).__enter__()
            saved_project, figure_records = origin.create_e739_project(
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
            summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"Wrote Origin project {saved_project}")
        except OriginAutomationError as exc:
            print(f"Origin automation disabled: {exc}")
            _write_origin_automation_log(output_dir, str(exc))
        except Exception as exc:
            print(f"Origin automation failed: {exc}")
            _write_origin_automation_log(output_dir, traceback.format_exc())
        finally:
            if origin is not None:
                origin.__exit__(None, None, None)

    print(f"Wrote {summary_path}")
    print(f"Wrote {transformed_path}")
    if runout_path.exists():
        print(f"Wrote {runout_path}")
    print(f"Wrote {curves_path}")
    print(f"Wrote {levels_path}")
    for extra_path in extra_output_paths:
        print(f"Wrote {extra_path}")
    return 0


def _run_power_analysis(args: argparse.Namespace) -> int:
    input_dir: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    files = _discover_input_files(input_dir, args.pattern)
    if not files:
        print(f"No supported data files found in {input_dir}.")
        return 1

    summaries: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    lines: list[dict[str, object]] = []
    plot_jobs: list[dict[str, object]] = []

    for path in files:
        for table in read_table(path):
            life_column, response_column = sn_xy_columns(table.frame, args.life, args.response)
            fit = fit_sn_power_law(
                table.frame,
                life_column,
                response_column,
                fit_points=args.fit_points,
            )
            label = _safe_name(table.label)

            curve = fit.curve.copy()
            curve.insert(0, "file", str(path))
            curve.insert(1, "sheet", table.sheet or "")
            curve.insert(2, "group", table.group or "")
            curves.append(curve)
            lines.append(
                {
                    "file": str(path),
                    "sheet": table.sheet or "",
                    "group": table.group or "",
                    "life_column": life_column,
                    "response_column": response_column,
                    "log10_intercept": fit.result.log10_intercept,
                    "log10_slope": fit.result.log10_slope,
                    "log10_formula": fit.result.log10_formula,
                    "life_start": fit.result.life_min,
                    "response_start": fit.result.fit_response_at_life_min,
                    "life_end": fit.result.life_max,
                    "response_end": fit.result.fit_response_at_life_max,
                }
            )

            summary_index = len(summaries)
            summaries.append(
                {
                    "file": str(path),
                    "sheet": table.sheet or "",
                    "group": table.group or "",
                    "life_column": life_column,
                    "response_column": response_column,
                    "points": fit.result.points,
                    "model": "response = a * life^b",
                    "coefficient_a": fit.result.coefficient_a,
                    "coefficient_b": fit.result.coefficient_b,
                    "log10_intercept": fit.result.log10_intercept,
                    "log10_slope": fit.result.log10_slope,
                    "r2": fit.result.r2,
                    "rmse": fit.result.rmse,
                    "life_min": fit.result.life_min,
                    "life_max": fit.result.life_max,
                    "fit_response_at_life_min": fit.result.fit_response_at_life_min,
                    "fit_response_at_life_max": fit.result.fit_response_at_life_max,
                    "response_min": fit.result.response_min,
                    "response_max": fit.result.response_max,
                    "formula": fit.result.formula,
                    "log10_formula": fit.result.log10_formula,
                    "figure": "",
                }
            )
            plot_jobs.append(
                {
                    "summary_index": summary_index,
                    "fit": fit,
                    "life_column": life_column,
                    "response_column": response_column,
                    "output_path": figures_dir / f"{label}.png",
                    "title": table.group or table.sheet or path.stem,
                    "label": table.label,
                }
            )

    origin = None
    if not args.dry_run:
        try:
            origin = OriginClient(visible=not args.hidden_origin).__enter__()
            for job in plot_jobs:
                try:
                    figure_path = origin.plot_sn_curve(
                        job["fit"],
                        str(job["life_column"]),
                        str(job["response_column"]),
                        job["output_path"],
                        title=str(job["title"]),
                        symbol_kind=args.symbol_kind,
                    )
                    summaries[int(job["summary_index"])]["figure"] = str(figure_path)
                except Exception as exc:
                    print(f"Origin plotting failed for {job['label']}: {exc}")
        except OriginAutomationError as exc:
            print(f"Origin automation disabled: {exc}")
        finally:
            if origin is not None:
                origin.__exit__(None, None, None)

    summary_path = output_dir / "fit_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
    curves_path = output_dir / "fit_curves.csv"
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(curves_path, index=False, encoding="utf-8-sig")
    lines_path = output_dir / "fit_lines.csv"
    pd.DataFrame(lines).to_csv(lines_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {summary_path}")
    if curves:
        print(f"Wrote {curves_path}")
    print(f"Wrote {lines_path}")
    return 0


def _discover_input_files(input_dir: Path, patterns: list[str] | None) -> list[Path]:
    return discover_files(input_dir, patterns or ["*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls"])


def _e739_summary_record(
    fit: E739Fit,
    path: Path,
    sheet: str | None,
    group: str | None,
    label: str,
    life_column: str,
    response_column: str,
) -> dict[str, object]:
    result = fit.result
    linearity = result.linearity_test
    return {
        "label": label,
        "file": str(path),
        "sheet": sheet or "",
        "group": group or "",
        "analysis": (
            "ASTM E739 linearized OLS"
            if result.model == "standard"
            else "Threshold-log censored maximum likelihood"
            if result.model == "threshold_log_mle"
            else "GJB/Z 18A 9.3.2 simplified strain-life workflow"
            if result.model == "gjb932-strain"
            else "Shifted-log nonlinear least squares"
        ),
        "model_name": result.model_name,
        "model": result.model,
        "life_column": life_column,
        "response_column": response_column,
        "x_transform": result.x_transform,
        "confidence": result.confidence,
        "parameter_count": result.parameter_count,
        "points": result.points,
        "degrees_of_freedom": result.degrees_of_freedom,
        "coefficient_a": result.coefficient_a,
        "coefficient_b": result.coefficient_b,
        "coefficient_c": result.coefficient_c,
        "threshold": result.threshold,
        "coefficient_a_lower": result.coefficient_a_lower,
        "coefficient_a_upper": result.coefficient_a_upper,
        "coefficient_b_lower": result.coefficient_b_lower,
        "coefficient_b_upper": result.coefficient_b_upper,
        "coefficient_c_lower": result.coefficient_c_lower,
        "coefficient_c_upper": result.coefficient_c_upper,
        "sigma_lower": result.sigma_lower,
        "sigma_upper": result.sigma_upper,
        "standard_error_a": result.standard_error_a,
        "standard_error_b": result.standard_error_b,
        "standard_error_c": result.standard_error_c,
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
        "replication_percent": result.replication_percent,
        "life_response_coefficient_a": result.life_response_coefficient_a,
        "life_response_coefficient_b": result.life_response_coefficient_b,
        "log_likelihood": result.log_likelihood,
        "negative_log_likelihood": result.negative_log_likelihood,
        "n_failure": result.n_failure,
        "n_runout": result.n_runout,
        "success": result.success,
        "optimizer_message": result.optimizer_message,
        "linearity_available": linearity.available,
        "linearity_reason": linearity.reason,
        "linearity_levels": linearity.levels,
        "linearity_lack_of_fit_df": linearity.lack_of_fit_df,
        "linearity_pure_error_df": linearity.pure_error_df,
        "linearity_lack_of_fit_ss": linearity.lack_of_fit_ss,
        "linearity_pure_error_ss": linearity.pure_error_ss,
        "linearity_lack_of_fit_ms": linearity.lack_of_fit_ms,
        "linearity_pure_error_ms": linearity.pure_error_ms,
        "linearity_f_statistic": linearity.f_statistic,
        "linearity_f_critical": linearity.f_critical,
        "linearity_p_value": linearity.p_value,
        "linearity_reject_linear_model": linearity.reject_linear_model,
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


def _write_e739_outputs(
    summary_frame: pd.DataFrame,
    transformed_frames: list[pd.DataFrame],
    runout_frames: list[pd.DataFrame],
    curve_frames: list[pd.DataFrame],
    level_frames: list[pd.DataFrame],
    extra_table_frames: dict[str, list[pd.DataFrame]],
    output_dir: Path,
) -> list[Path]:
    summary_frame.to_csv(output_dir / "e739_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(transformed_frames, ignore_index=True).to_csv(
        output_dir / "e739_transformed_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if runout_frames:
        pd.concat(runout_frames, ignore_index=True).to_csv(
            output_dir / "e739_runout_data.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        stale_runout_path = output_dir / "e739_runout_data.csv"
        if stale_runout_path.exists():
            stale_runout_path.unlink()
    pd.concat(curve_frames, ignore_index=True).to_csv(
        output_dir / "e739_curve_bands.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(level_frames, ignore_index=True).to_csv(
        output_dir / "e739_level_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    extra_paths: list[Path] = []
    for name, frames in sorted(extra_table_frames.items()):
        if not frames:
            continue
        path = output_dir / f"e739_{_safe_name(name).lower()}.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
        extra_paths.append(path)
    return extra_paths


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
    return value or "sn_curve"
