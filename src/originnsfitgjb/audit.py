from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .gjb import GJBAuditStep, GJBFit


@dataclass(frozen=True)
class AuditRecord:
    label: str
    file: str
    sheet: str
    group: str
    fit: GJBFit


AUDIT_STEP_FILES = {
    "Step00_InputChecked": "step00_input_checked",
    "Step01_InitialOLS": "step01_initial_ols",
    "Step02_InitialNLS": "step02_initial_nls",
    "Step03_VarianceAnalysis": "step03_variance_analysis",
    "Step04_RefitData": "step04_refit_data",
    "Step05_RefitResult": "step05_refit_result",
    "Step06_ParameterSignificance": "step06_parameter_significance",
    "Step07_FixedA4LinearFit": "step07_fixed_a4_linear_fit",
    "Step08_ResidualsOutliers": "step08_residuals_outliers",
    "Step09_FinalMLE": "step09_final_mle",
    "Step10_FinalResidualStatistics": "step10_final_residual_statistics",
    "Step11_ModelAssessment": "step11_model_assessment",
    "Step12_R2_DocumentStyle": "step12_r2_document_style",
}

AUDIT_SHEET_NAMES = {
    "Step00_InputChecked": "Step00_InputChecked",
    "Step01_InitialOLS": "Step01_InitialOLS",
    "Step02_InitialNLS": "Step02_InitialNLS",
    "Step03_VarianceAnalysis": "Step03_Variance",
    "Step04_RefitData": "Step04_RefitData",
    "Step05_RefitResult": "Step05_RefitResult",
    "Step06_ParameterSignificance": "Step06_ParamSignif",
    "Step07_FixedA4LinearFit": "Step07_FixedA4Linear",
    "Step08_ResidualsOutliers": "Step08_Outliers",
    "Step09_FinalMLE": "Step09_FinalMLE",
    "Step10_FinalResidualStatistics": "Step10_FinalResiduals",
    "Step11_ModelAssessment": "Step11_ModelAssessment",
    "Step12_R2_DocumentStyle": "Step12_R2",
}

STEP_PURPOSES = {
    "Step00_InputChecked": "Check source rows, numeric conversion, status and positive-domain eligibility.",
    "Step01_InitialOLS": "Compute A4_initial and the initial OLS line on failure rows.",
    "Step02_InitialNLS": "Fit A1, A2 and A4 by unweighted nonlinear least squares.",
    "Step03_VarianceAnalysis": "Estimate h = sigma0 + sigma1/response and decide whether to weight refit.",
    "Step04_RefitData": "Show which runout rows are temporarily treated as failures for refit.",
    "Step05_RefitResult": "Show refit predictions, residuals and objective components.",
    "Step06_ParameterSignificance": "Check A2 negative significance and A4 non-zero significance.",
    "Step07_FixedA4LinearFit": "When weighted, fix A4 and re-estimate A1/A2 linearly.",
    "Step08_ResidualsOutliers": "Report studentized residual outlier decisions and iteration behavior.",
    "Step09_FinalMLE": "Apply final likelihood correction with failure logpdf and runout logsf terms.",
    "Step10_FinalResidualStatistics": "Compute document-style residual statistics without replacing legacy fields.",
    "Step11_ModelAssessment": "Compute the Durbin-Watson style standardized-residual assessment.",
    "Step12_R2_DocumentStyle": "Compute document-style R2 while preserving r2_log_life.",
}

STEP_FORMULAS = {
    "Step00_InputChecked": "gjb_y_log10_life = log10(gjb_life)",
    "Step01_InitialOLS": "A4_initial = 0.5*min(response_failure); X = log10(response-A4_initial); y=A1+A2*X",
    "Step02_InitialNLS": "y_pred = A1 + A2*log10(response-A4); residual = y - y_pred",
    "Step03_VarianceAnalysis": "scaled_abs_residual=abs(R)/sqrt(2/pi); h=sigma0+sigma1/response; weight=1/h^2",
    "Step04_RefitData": "include failures plus runout rows with response > e_min_failure",
    "Step05_RefitResult": "weighted objective=sum((R/h)^2); unweighted objective=sum(R^2)",
    "Step06_ParameterSignificance": "A2_upper_90 < 0; A4_lower_90 > 0 or A4_upper_90 < 0",
    "Step07_FixedA4LinearFit": "Y*=y/h; U=1/h; V=X/h; Y*=A1_corrected*U + A2_corrected*V",
    "Step08_ResidualsOutliers": "G=max(abs(studentized residual)); critical=t(1-alpha/(2n), n-k-1)",
    "Step09_FinalMLE": "failure rows use logpdf; runout rows use logsf; A4 and SD_i fixed",
    "Step10_FinalResidualStatistics": "weighted: WR=R/h, RMSE=sqrt(sum(WR^2)/(n-k)), SD_i=RMSE*h_i",
    "Step11_ModelAssessment": "D=sum((SR_i-SR_{i-1})^2)/sum(SR_i^2); Dcrit=2-4.73/n^0.555",
    "Step12_R2_DocumentStyle": "R2=1-RMSE^2/RTE^2",
}

STEP_INPUTS = {
    "Step00_InputChecked": "source life, response and optional status columns",
    "Step01_InitialOLS": "failure gjb_response, gjb_y_log10_life",
    "Step02_InitialNLS": "failure gjb_response, gjb_y_log10_life, Step01 initial parameters",
    "Step03_VarianceAnalysis": "Step02 residuals and gjb_response",
    "Step04_RefitData": "active rows, gjb_is_failure, gjb_response, e_min_failure",
    "Step05_RefitResult": "Step04 included rows and Step03 h/weight decision",
    "Step06_ParameterSignificance": "Step05 parameters and standard errors",
    "Step07_FixedA4LinearFit": "Step05 refit rows, fixed A4, h and weight",
    "Step08_ResidualsOutliers": "post-significance fit rows and residuals",
    "Step09_FinalMLE": "active rows, final A4 and SD_i",
    "Step10_FinalResidualStatistics": "final fit rows and final coefficients",
    "Step11_ModelAssessment": "Step10 standardized residuals sorted by response",
    "Step12_R2_DocumentStyle": "Step10 RMSE, h and y values",
}

STEP_RULES = {
    "Step03_VarianceAnalysis": "use weighted refit when sigma1_lower_90 > 0 and sigma1 > 0",
    "Step06_ParameterSignificance": "continue only when A2 and A4 significance checks pass",
    "Step07_FixedA4LinearFit": "perform only when weighted refit was selected and significance passed",
    "Step08_ResidualsOutliers": "auto removes candidate only when G > critical; report-only never removes",
    "Step09_FinalMLE": "run only after parameter significance passes",
    "Step11_ModelAssessment": "D < Dcrit indicates possible misfit",
}


def write_audit_outputs(
    records: list[AuditRecord],
    summary_frame: pd.DataFrame,
    audit_dir: Path,
    *,
    write_json: bool,
    write_workbook: bool,
) -> list[Path]:
    """Write CSV, JSON, Markdown and optional Excel audit artifacts."""
    _ensure_audit_dirs(audit_dir)
    output_paths: list[Path] = []
    all_step_tables: dict[str, list[pd.DataFrame]] = {step_id: [] for step_id in AUDIT_STEP_FILES}
    decision_frames: list[pd.DataFrame] = []

    for record in records:
        tables_dir = audit_dir / "tables" / record.label
        json_dir = audit_dir / "json" / record.label
        tables_dir.mkdir(parents=True, exist_ok=True)
        if write_json:
            json_dir.mkdir(parents=True, exist_ok=True)

        steps = record.fit.audit_steps or {}
        for step_id in AUDIT_STEP_FILES:
            step = steps.get(step_id)
            if step is None:
                step = _missing_step(step_id)
            table = _step_table_with_metadata(step, record)
            all_step_tables[step_id].append(table)
            csv_path = tables_dir / f"{AUDIT_STEP_FILES[step_id]}.csv"
            table.to_csv(csv_path, index=False, encoding="utf-8-sig")
            output_paths.append(csv_path)
            if write_json:
                json_path = json_dir / f"{AUDIT_STEP_FILES[step_id]}.json"
                _write_json(json_path, _step_json_payload(step, record, csv_path))
                output_paths.append(json_path)

        if record.fit.decision_log is not None and not record.fit.decision_log.empty:
            decision_frames.append(_with_record_metadata(record.fit.decision_log, record))

    decision_log = (
        pd.concat(decision_frames, ignore_index=True)
        if decision_frames
        else pd.DataFrame(
            columns=[
                "label",
                "file",
                "sheet",
                "group",
                "iteration",
                "step_id",
                "question",
                "value",
                "rule",
                "decision",
                "reason",
                "source_table",
            ]
        )
    )
    decision_csv = audit_dir / "gjb_decision_log.csv"
    decision_log.to_csv(decision_csv, index=False, encoding="utf-8-sig")
    output_paths.append(decision_csv)
    decision_json = audit_dir / "gjb_decision_log.json"
    _write_json(decision_json, _json_ready(decision_log.to_dict(orient="records")))
    output_paths.append(decision_json)

    checklist_md = audit_dir / "gjb_manual_checklist.md"
    checklist_md.write_text(_manual_checklist_markdown(), encoding="utf-8-sig")
    output_paths.append(checklist_md)

    summary_md = audit_dir / "gjb_audit_summary.md"
    summary_md.write_text(
        _audit_summary_markdown(records, audit_dir, write_json, write_workbook),
        encoding="utf-8-sig",
    )
    output_paths.append(summary_md)

    if write_workbook:
        workbook_path = audit_dir / "gjb_audit_workbook.xlsx"
        _write_workbook(
            workbook_path,
            all_step_tables,
            decision_log,
            summary_frame,
        )
        output_paths.append(workbook_path)

    return output_paths


def _ensure_audit_dirs(audit_dir: Path) -> None:
    for child in ("tables", "json", "workbooks", "reports", "plots"):
        (audit_dir / child).mkdir(parents=True, exist_ok=True)


def _missing_step(step_id: str) -> GJBAuditStep:
    return GJBAuditStep(
        step_id=step_id,
        step_name=step_id,
        status="missing",
        input_columns=(),
        output_columns=(),
        formulas=(),
        parameters_in={},
        parameters_out={},
        decision={"missing": True},
        warnings=("Step was not produced because fitting stopped before this point.",),
        table=pd.DataFrame([{"step_id": step_id, "status": "missing"}]),
    )


def _step_table_with_metadata(step: GJBAuditStep, record: AuditRecord) -> pd.DataFrame:
    table = step.table.copy() if step.table is not None else pd.DataFrame()
    if table.empty:
        table = pd.DataFrame([{"step_id": step.step_id, "status": step.status}])
    return _with_record_metadata(table, record)


def _with_record_metadata(frame: pd.DataFrame, record: AuditRecord) -> pd.DataFrame:
    result = frame.copy()
    metadata = [
        ("label", record.label),
        ("file", record.file),
        ("sheet", record.sheet),
        ("group", record.group),
    ]
    for column, value in reversed(metadata):
        if column in result.columns:
            result[column] = value
        else:
            result.insert(0, column, value)
    return result


def _step_json_payload(step: GJBAuditStep, record: AuditRecord, csv_path: Path) -> dict[str, object]:
    return {
        "label": record.label,
        "file": record.file,
        "sheet": record.sheet,
        "group": record.group,
        "step_id": step.step_id,
        "step_name": step.step_name,
        "status": step.status,
        "input_columns": list(step.input_columns),
        "output_columns": list(step.output_columns),
        "formulas": list(step.formulas),
        "parameters_in": _json_ready(step.parameters_in),
        "parameters_out": _json_ready(step.parameters_out),
        "decision": _json_ready(step.decision),
        "warnings": list(step.warnings),
        "row_count": 0 if step.table is None else int(len(step.table)),
        "table_csv": str(csv_path),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def _manual_check_items() -> list[dict[str, str]]:
    return [
        _check("1", "Step00_InputChecked", "original row count matches source", "gjb_row_id", ""),
        _check("2", "Step00_InputChecked", "gjb_life > 0", "gjb_life", ""),
        _check("3", "Step00_InputChecked", "gjb_response > 0", "gjb_response", ""),
        _check("4", "Step00_InputChecked", "gjb_y_log10_life = log10(gjb_life)", "gjb_y_log10_life", ""),
        _check("5", "Step01_InitialOLS", "A4_initial = min(response_failure)/2", "A4_initial", ""),
        _check("6", "Step01_InitialOLS", "X_initial = log10(response - A4_initial)", "X_initial_log10", ""),
        _check("7", "Step02_InitialNLS", "y_pred = A1 + A2*log10(response-A4)", "y_pred_initial_nls", ""),
        _check("8", "Step02_InitialNLS", "residual = y - y_pred", "residual_initial_nls", ""),
        _check("9", "Step03_Variance", "scaled_abs_residual = abs(R)/sqrt(2/pi)", "scaled_abs_residual", ""),
        _check("10", "Step03_Variance", "h = sigma0 + sigma1/response", "h", ""),
        _check("11", "Step03_Variance", "weight = 1/h^2", "weight", ""),
        _check("12", "Step05_RefitResult", "weighted objective = sum((R/h)^2)", "objective_component", ""),
        _check("13", "Step07_FixedA4Linear", "only A1/A2 corrected; A4 fixed", "A4_fixed", ""),
        _check("14", "Step07_FixedA4Linear", "A1^2 and A2^2 are corrected params, not squares", "A1_corrected,A2_corrected", ""),
        _check("15", "Step10_FinalResiduals", "three-parameter model k=3", "k", ""),
        _check("16", "Step09_FinalMLE", "runout likelihood uses logsf", "likelihood_type", ""),
        _check("17", "Origin workbook", "Origin Direct Weighting receives 1/h^2, not h", "weight", ""),
    ]


def _check(
    item: str,
    location: str,
    formula: str,
    output_columns: str,
    note: str,
) -> dict[str, str]:
    return {
        "item": item,
        "check_position": location,
        "formula": formula,
        "program_output_columns": output_columns,
        "manual_recalc": "",
        "pass": "",
        "note": note,
    }


def _manual_checklist_markdown() -> str:
    lines = [
        "# GJB/Z 18A Audit Manual Checklist",
        "",
        "| # | Check position | Formula / rule | Program output columns | Manual recalculation | Pass |",
        "|---|---|---|---|---|---|",
    ]
    for item in _manual_check_items():
        lines.append(
            "| {item} | {check_position} | {formula} | {program_output_columns} |  |  |".format(
                **item
            )
        )
    lines.append("")
    return "\n".join(lines)


def _audit_summary_markdown(
    records: list[AuditRecord],
    audit_dir: Path,
    write_json: bool,
    write_workbook: bool,
) -> str:
    labels = ", ".join(record.label for record in records)
    return "\n".join(
        [
            "# GJB/Z 18A Audit Summary",
            "",
            f"Audit directory: `{audit_dir}`",
            f"Labels: {labels}",
            f"Per-step JSON written: {'yes' if write_json else 'no'}",
            f"Excel workbook written: {'yes' if write_workbook else 'no'}",
            "",
            "The model remains `log10(Nf) = A1 + A2 * log10(response - A4)`.",
            "The response column is used directly as equivalent strain; A3 is not fitted.",
            "Origin automation is downstream of these Python audit outputs.",
            "",
        ]
    )


def _write_workbook(
    workbook_path: Path,
    all_step_tables: dict[str, list[pd.DataFrame]],
    decision_log: pd.DataFrame,
    summary_frame: pd.DataFrame,
) -> None:
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for step_id, sheet_name in AUDIT_SHEET_NAMES.items():
            table = (
                pd.concat(all_step_tables.get(step_id, []), ignore_index=True)
                if all_step_tables.get(step_id)
                else pd.DataFrame()
            )
            meta = _sheet_metadata_frame(step_id, table)
            safe_sheet = _excel_sheet_name(sheet_name)
            meta.to_excel(writer, sheet_name=safe_sheet, index=False, header=False)
            startrow = len(meta) + 2
            table.to_excel(writer, sheet_name=safe_sheet, index=False, startrow=startrow)

        decision_log.to_excel(writer, sheet_name="DecisionLog", index=False)
        pd.DataFrame(_manual_check_items()).to_excel(writer, sheet_name="ManualCheck", index=False)
        summary_frame.to_excel(writer, sheet_name="FinalSummary", index=False)


def _sheet_metadata_frame(step_id: str, table: pd.DataFrame) -> pd.DataFrame:
    decision_result = _sheet_decision_result(table)
    return pd.DataFrame(
        [
            ["Step", step_id],
            ["Purpose", STEP_PURPOSES.get(step_id, step_id)],
            ["Formula", STEP_FORMULAS.get(step_id, "")],
            ["Input columns", STEP_INPUTS.get(step_id, "")],
            ["Output columns", ", ".join(map(str, table.columns))],
            ["Decision rule", STEP_RULES.get(step_id, "")],
            ["Decision result", decision_result],
            ["Warnings", _sheet_warnings(table)],
        ]
    )


def _sheet_decision_result(table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    for column in ("use_weighted", "overall_passed", "performed", "remove_outlier", "likelihood_type", "possible_misfit"):
        if column in table.columns:
            values = sorted({str(value) for value in table[column].dropna().unique()})
            if values:
                return f"{column}: {', '.join(values[:8])}"
    return ""


def _sheet_warnings(table: pd.DataFrame) -> str:
    warning_columns = [column for column in table.columns if "warning" in str(column).lower()]
    values: list[str] = []
    for column in warning_columns:
        values.extend(str(value) for value in table[column].dropna().unique() if str(value).strip())
    return "; ".join(values[:8])


def _excel_sheet_name(name: str) -> str:
    safe = re.sub(r"[\[\]:*?/\\]", "_", name)
    return safe[:31] or "Sheet"
