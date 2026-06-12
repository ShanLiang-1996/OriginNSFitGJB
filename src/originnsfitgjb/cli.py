from __future__ import annotations

import argparse
from pathlib import Path

from .analysis_service import AnalysisConfig, run_analysis


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
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Write complete CSV, JSON and Excel audit outputs under --audit-dir.",
    )
    parser.add_argument(
        "--audit-workbook",
        action="store_true",
        help="Write output/audit/gjb_audit_workbook.xlsx for manual review.",
    )
    parser.add_argument(
        "--audit-json",
        action="store_true",
        help="Write per-step JSON metadata and decision records.",
    )
    parser.add_argument(
        "--outlier-mode",
        choices=("auto", "report-only"),
        default="auto",
        help="auto removes documented outliers; report-only records candidates without deleting rows.",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="Audit output directory. Defaults to OUTPUT/audit.",
    )
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
    config = AnalysisConfig(
        input_dir=args.input,
        output_dir=args.output,
        patterns=tuple(args.pattern or ()),
        life_column=args.life,
        response_column=args.response,
        status_column=args.status,
        level_column=args.level,
        replicate_decimals=args.replicate_decimals,
        confidence=args.confidence,
        fit_points=args.fit_points,
        symbol_kind=args.symbol_kind,
        dry_run=args.dry_run,
        audit=args.audit,
        audit_workbook=args.audit_workbook,
        audit_json=args.audit_json,
        outlier_mode=args.outlier_mode,
        audit_dir=args.audit_dir,
        hidden_origin=args.hidden_origin,
        project_path=args.project,
        graph_template_path=args.graph_template,
        no_graph_template=args.no_graph_template,
        linearized_graph=args.linearized_graph,
        no_runout_arrows=args.no_runout_arrows,
    )
    result = run_analysis(config, log_callback=print)
    return 0 if result.completed else 1
