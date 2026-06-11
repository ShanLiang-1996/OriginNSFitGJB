from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats


GJB_METHOD = "gjb18a-9.3.2"

FAILURE_STATUS_MARKERS = (
    "failure",
    "failed",
    "fail",
    "fracture",
    "broken",
    "yes",
    "true",
    "1",
    "\u5931\u6548",
    "\u65ad\u88c2",
)
NON_FAILURE_STATUS_MARKERS = (
    "runout",
    "run-out",
    "run out",
    "suspended",
    "suspension",
    "censored",
    "right-censored",
    "no",
    "false",
    "0",
    "0.0",
    "\u672a\u5931\u6548",
    "\u505c\u8bd5",
    "\u5220\u5931",
)


@dataclass(frozen=True)
class GJBFitResult:
    life_column: str
    response_column: str
    model: str
    model_name: str
    confidence: float
    x_transform: str
    parameter_count: int
    points: int
    degrees_of_freedom: int
    coefficient_a: float
    coefficient_b: float
    coefficient_c: float | None
    threshold: float | None
    x_mean: float
    y_mean: float
    sxx: float
    sxy: float
    residual_sum_squares: float
    sigma_squared: float
    sigma: float
    r2: float
    rmse_log_life: float
    standard_error_a: float
    standard_error_b: float
    standard_error_c: float | None
    t_critical: float
    f_band_critical: float
    simultaneous_band_factor: float
    coefficient_a_lower: float
    coefficient_a_upper: float
    coefficient_b_lower: float
    coefficient_b_upper: float
    coefficient_c_lower: float | None
    coefficient_c_upper: float | None
    sigma_lower: float | None
    sigma_upper: float | None
    x_min: float
    x_max: float
    life_min: float
    life_max: float
    response_min: float
    response_max: float
    replication_percent: float
    life_response_coefficient_a: float
    life_response_coefficient_b: float
    log_likelihood: float | None
    negative_log_likelihood: float | None
    n_failure: int | None
    n_runout: int | None
    success: bool | None
    optimizer_message: str
    log_life_formula: str
    life_formula: str
    life_response_formula: str
    response_life_formula: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class GJBAuditStep:
    """Structured audit record for one GJB/Z 18A 9.3.2 workflow step."""

    step_id: str
    step_name: str
    status: str
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    formulas: tuple[str, ...]
    parameters_in: dict[str, object]
    parameters_out: dict[str, object]
    decision: dict[str, object]
    warnings: tuple[str, ...]
    table: pd.DataFrame | None = None


@dataclass(frozen=True)
class GJBFit:
    result: GJBFitResult
    data: pd.DataFrame
    curve: pd.DataFrame
    level_stats: pd.DataFrame
    runout_data: pd.DataFrame | None = None
    extra_tables: dict[str, pd.DataFrame] | None = None
    audit_steps: dict[str, GJBAuditStep] | None = None
    decision_log: pd.DataFrame | None = None


def fit_gjb18a(
    frame: pd.DataFrame,
    life_column: str,
    response_column: str,
    *,
    confidence: float = 0.95,
    fit_points: int = 300,
    status_column: str | None = None,
    level_column: str | None = None,
    replicate_decimals: int = 8,
    outlier_mode: str = "auto",
    source_file: str | None = None,
    source_sheet: str | None = None,
    source_group: str | None = None,
    source_label: str | None = None,
) -> GJBFit:
    """Fit the simplified GJB/Z 18A 9.3.2 strain-life workflow.

    The simplified Formula 136 model is
    ``log10(Nf) = A1 + A2 * log10(response - A4)``.  The input response column
    is used directly as equivalent strain; A3 is not fitted.  ``outlier_mode``
    controls whether the documented outlier candidate is removed automatically
    (``auto``) or only reported for manual review (``report-only``).
    """
    confidence = _normalize_confidence(confidence)
    if fit_points < 2:
        raise ValueError("fit_points must be at least 2.")
    outlier_mode = str(outlier_mode).strip().lower()
    if outlier_mode not in {"auto", "report-only"}:
        raise ValueError("outlier_mode must be 'auto' or 'report-only'.")

    # Step 1 - Validate the user-selected source columns before any numeric conversion.
    _require_column(frame, life_column, "life")
    _require_column(frame, response_column, "response")
    if status_column:
        _require_column(frame, status_column, "status")
    if level_column:
        _require_column(frame, level_column, "level")

    # Step 2 - Copy only the columns needed by this workflow and normalize numeric values.
    warnings: list[str] = []
    source_columns = list(dict.fromkeys([response_column, life_column, status_column, level_column]))
    source_columns = [column for column in source_columns if column]
    data = frame[source_columns].copy()
    data["gjb_life"] = pd.to_numeric(data[life_column], errors="coerce")
    data["gjb_response"] = pd.to_numeric(data[response_column], errors="coerce")

    # Step 3 - Drop rows that cannot participate in the strain-life calculation.
    initial_rows = len(data)
    data = data.dropna(subset=["gjb_life", "gjb_response"]).copy()
    dropped_missing = initial_rows - len(data)
    if dropped_missing:
        warnings.append(f"Dropped {dropped_missing} row(s) with missing numeric life/response.")

    # Step 4 - Run the GJB/Z 18A fitting sequence on prepared rows.
    return _fit_prepared_gjb18a(
        data,
        life_column,
        response_column,
        confidence=confidence,
        fit_points=fit_points,
        status_column=status_column,
        level_column=level_column,
        replicate_decimals=replicate_decimals,
        warnings=warnings,
        source_file=source_file,
        source_sheet=source_sheet,
        source_group=source_group,
        source_label=source_label,
        original_row_count=initial_rows,
        dropped_missing_rows=dropped_missing,
        outlier_mode=outlier_mode,
    )


def _normalize_confidence(confidence: float) -> float:
    confidence = float(confidence)
    if confidence > 1.0 and confidence <= 100.0:
        confidence /= 100.0
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence must be between 0 and 1, or between 0 and 100 percent.")
    return confidence


def _require_column(frame: pd.DataFrame, column: str, role: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"GJB {role} column not found: {column}")


def _audit_step(
    step_id: str,
    step_name: str,
    status: str,
    table: pd.DataFrame | None,
    *,
    input_columns: tuple[str, ...],
    formulas: tuple[str, ...],
    parameters_in: dict[str, object] | None = None,
    parameters_out: dict[str, object] | None = None,
    decision: dict[str, object] | None = None,
    warnings: tuple[str, ...] = (),
) -> GJBAuditStep:
    """Create a structured audit record with table column metadata."""
    return GJBAuditStep(
        step_id=step_id,
        step_name=step_name,
        status=status,
        input_columns=input_columns,
        output_columns=tuple(table.columns) if table is not None else (),
        formulas=formulas,
        parameters_in=parameters_in or {},
        parameters_out=parameters_out or {},
        decision=decision or {},
        warnings=warnings,
        table=table,
    )


def _combine_audit_steps(step_id: str, steps: list[GJBAuditStep]) -> GJBAuditStep:
    """Combine multiple iteration audit records into one step table."""
    if not steps:
        raise ValueError(f"No audit steps to combine for {step_id}.")
    first = steps[0]
    frames = [step.table for step in steps if step.table is not None]
    table = pd.concat(frames, ignore_index=True) if frames else None
    statuses = {step.status for step in steps}
    if "failed" in statuses:
        status = "failed"
    elif "warning" in statuses:
        status = "warning"
    elif "candidate" in statuses:
        status = "candidate"
    elif all(step.status == "skipped" for step in steps):
        status = "skipped"
    else:
        status = "completed"
    return GJBAuditStep(
        step_id=step_id,
        step_name=first.step_name,
        status=status,
        input_columns=first.input_columns,
        output_columns=tuple(table.columns) if table is not None else first.output_columns,
        formulas=first.formulas,
        parameters_in={"iterations": [step.parameters_in for step in steps]},
        parameters_out={"iterations": [step.parameters_out for step in steps]},
        decision={"iterations": [step.decision for step in steps]},
        warnings=tuple(warning for step in steps for warning in step.warnings),
        table=table,
    )


def _decision_row(
    iteration: int,
    step_id: str,
    question: str,
    value: object,
    rule: str,
    decision: str,
    reason: str,
    source_table: str,
) -> dict[str, object]:
    """Create one process decision log row."""
    return {
        "iteration": iteration,
        "step_id": step_id,
        "question": question,
        "value": value,
        "rule": rule,
        "decision": decision,
        "reason": reason,
        "source_table": source_table,
    }


def _gjb18a_input_checked_table(
    prepared: pd.DataFrame,
    life_column: str,
    response_column: str,
    status_column: str | None,
) -> pd.DataFrame:
    """Return Step00 input validation rows for Excel/CSV review."""
    columns = ["gjb_row_id"]
    for column in (life_column, response_column, status_column):
        if column and column not in columns:
            columns.append(column)
    columns.extend(
        [
            "gjb_life",
            "gjb_response",
            "gjb_y_log10_life",
            "gjb_is_failure",
            "gjb_original_status",
            "included_in_failure_fit",
            "included_in_runout_mle",
            "positive_domain_check",
            "validation_message",
        ]
    )
    return prepared[columns].copy()

def _fit_prepared_gjb18a(
    data: pd.DataFrame,
    life_column: str,
    response_column: str,
    *,
    confidence: float,
    fit_points: int,
    status_column: str | None,
    level_column: str | None,
    replicate_decimals: int,
    warnings: list[str],
    source_file: str | None,
    source_sheet: str | None,
    source_group: str | None,
    source_label: str | None,
    original_row_count: int,
    dropped_missing_rows: int,
    outlier_mode: str,
) -> GJBFit:
    """Fit prepared rows with the simplified GJB/Z 18A 9.3.2 Formula 136 model.

    Formula 136 uses an equivalent strain.  This implementation follows the
    requested simplification: the input response column is used directly as that strain,
    and A3 is not fitted.  The fitted relation is:

        log10(Nf) = A1 + A2 * log10(strain - A4)
    """
    audit_steps: dict[str, GJBAuditStep] = {}
    iteration_audit_steps: dict[str, list[GJBAuditStep]] = {}
    decision_rows: list[dict[str, object]] = []

    # Step 1 - Keep all numeric rows long enough to audit domain checks.
    prepared_all = data.copy().reset_index(drop=True)
    prepared_all["gjb_row_id"] = np.arange(len(prepared_all), dtype=int)
    before_domain_rows = len(prepared_all)
    positive_mask = (prepared_all["gjb_life"] > 0) & (prepared_all["gjb_response"] > 0)
    dropped_nonpositive = before_domain_rows - int(np.sum(positive_mask))
    if dropped_nonpositive:
        warnings.append(
            f"Dropped {dropped_nonpositive} row(s) outside the positive domain required by gjb18a-strain."
        )
    # Step 2 - Normalize failure/run-out status and compute the log-life response.
    if status_column:
        prepared_all["gjb_is_failure"] = prepared_all[status_column].map(_status_is_failure).astype(bool)
    else:
        prepared_all["gjb_is_failure"] = True
    prepared_all["gjb_y_log10_life"] = np.where(
        prepared_all["gjb_life"].to_numpy(dtype=float) > 0.0,
        np.log10(np.maximum(prepared_all["gjb_life"].to_numpy(dtype=float), 1e-300)),
        np.nan,
    )
    prepared_all["gjb_original_status"] = np.where(
        prepared_all["gjb_is_failure"],
        "failure",
        "runout",
    )
    prepared_all["positive_domain_check"] = positive_mask
    prepared_all["included_in_failure_fit"] = positive_mask & prepared_all["gjb_is_failure"].astype(bool)
    prepared_all["included_in_runout_mle"] = positive_mask & ~prepared_all["gjb_is_failure"].astype(bool)
    prepared_all["validation_message"] = np.where(
        positive_mask,
        "ok",
        "excluded: gjb_life and gjb_response must both be positive",
    )
    step00_table = _gjb18a_input_checked_table(
        prepared_all,
        life_column,
        response_column,
        status_column,
    )
    prepared = prepared_all[positive_mask].copy().reset_index(drop=True)
    step00_warnings = []
    if dropped_missing_rows:
        step00_warnings.append(f"{dropped_missing_rows} row(s) were removed before numeric fitting because life/response was missing.")
    if dropped_nonpositive:
        step00_warnings.append(f"{dropped_nonpositive} row(s) failed the positive domain check.")
    audit_steps["Step00_InputChecked"] = _audit_step(
        "Step00_InputChecked",
        "Input checked",
        "completed",
        step00_table,
        input_columns=tuple(column for column in (life_column, response_column, status_column) if column),
        formulas=("gjb_y_log10_life = log10(gjb_life)",),
        parameters_in={
            "input_file": source_file or "",
            "sheet": source_sheet or "",
            "group": source_group or "",
            "label": source_label or "",
            "life_column": life_column,
            "response_column": response_column,
            "status_column": status_column or "",
        },
        parameters_out={
            "original_rows": int(original_row_count),
            "dropped_missing_rows": int(dropped_missing_rows),
            "dropped_nonpositive_rows": int(dropped_nonpositive),
            "failure_count": int((prepared_all["positive_domain_check"] & prepared_all["gjb_is_failure"]).sum()),
            "runout_count": int((prepared_all["positive_domain_check"] & ~prepared_all["gjb_is_failure"]).sum()),
        },
        decision={
            "positive_domain_rule": "gjb_life > 0 and gjb_response > 0",
            "status_rule": "known runout/suspended/censored markers are treated as runout; otherwise failure",
        },
        warnings=tuple(step00_warnings),
    )
    if prepared.empty:
        raise ValueError("No positive life/strain rows are available for gjb18a-strain.")

    original_failure = prepared[prepared["gjb_is_failure"]].copy()
    if len(original_failure) < 4:
        raise ValueError("gjb18a-strain requires at least four failure points.")
    e_min_failure = float(original_failure["gjb_response"].min())
    if e_min_failure <= 0.0:
        raise ValueError("gjb18a-strain requires positive failure strain values.")

    # Step 3 - Prepare iterative outlier removal state.
    removed_ids: set[int] = set()
    iteration_records: list[pd.DataFrame] = []
    outlier_records: list[pd.DataFrame] = []
    final_state: dict[str, object] | None = None
    max_iterations = min(20, max(1, len(prepared) - 4))

    # Step 4 - Repeat the GJB fit cycle until no outlier is removed or checks fail.
    for iteration in range(1, max_iterations + 1):
        active = prepared[~prepared["gjb_row_id"].isin(removed_ids)].copy().reset_index(drop=True)
        state = _gjb18a_iteration(
            active,
            e_min_failure=e_min_failure,
            confidence=confidence,
            iteration=iteration,
            outlier_mode=outlier_mode,
        )
        iteration_records.extend(state["tables"])
        for step_id, step in state["audit_steps"].items():
            iteration_audit_steps.setdefault(step_id, []).append(step)
        decision_rows.extend(state["decision_rows"])
        outlier_table = state["outlier_table"]
        outlier_records.append(outlier_table)
        final_state = state

        if not state["significance_passed"]:
            warnings.append(str(state["stop_reason"]))
            break
        if not bool(state["remove_outlier"]):
            break

        removed_id = int(state["removed_row_id"])
        removed_ids.add(removed_id)
        warnings.append(
            f"Removed outlier row id {removed_id} during gjb18a-strain 9.3.2.4.4 iteration."
        )
    else:
        warnings.append("Stopped gjb18a-strain outlier iteration after the maximum iteration limit.")

    if final_state is None:
        raise ValueError("gjb18a-strain did not complete an analysis iteration.")

    # Step 5 - Rebuild the active dataset after any removed outliers.
    active = prepared[~prepared["gjb_row_id"].isin(removed_ids)].copy().reset_index(drop=True)
    refit = final_state["post_significance_fit"]
    variance = final_state["variance_model"]
    residuals = final_state["residual_table"].copy()
    significance_passed = bool(final_state["significance_passed"])
    mle_state: dict[str, object] | None = None
    # Step 6 - Apply the final right-censored likelihood correction when parameter checks pass.
    if significance_passed:
        mle_state = _gjb18a_final_mle(
            active,
            refit,
            variance,
        )
        if not bool(mle_state["success"]):
            warnings.append(f"gjb18a-strain final MLE warning: {mle_state['optimizer_message']}")
    active_runout_count = int((~active["gjb_is_failure"].astype(bool)).sum())
    if mle_state is not None:
        decision_rows.append(
            _decision_row(
                int(final_state["iteration"]),
                "Step09_FinalMLE",
                "all domain-valid non-outlier rows included in final MLE ?",
                {
                    "active_rows_after_outlier_removal": len(active),
                    "mle_input_rows": mle_state["mle_input_rows"],
                    "ignored_response_le_A4_rows": mle_state["mle_domain_ignored_rows"],
                    "runout_count": active_runout_count,
                },
                "final MLE uses active rows with response > A4; active rows with response <= A4 are ignored",
                (
                    "domain-valid non-outlier rows included; runout contributes via logsf"
                    if active_runout_count
                    else "domain-valid non-outlier rows included; failure-only likelihood correction"
                ),
                (
                    "runout rows with response > A4 are right-censored; rows with response <= A4 are ignored"
                    if active_runout_count
                    else "no runout rows were available; MLE is based on domain-valid failure likelihood terms"
                ),
                "FinalMLE",
            )
        )
    else:
        decision_rows.append(
            _decision_row(
                int(final_state["iteration"]),
                "Step09_FinalMLE",
                "parameter significance passed ?",
                significance_passed,
                "MLE runs only after parameter significance passes",
                "final MLE skipped",
                str(final_state["stop_reason"]),
                "FinalMLE",
            )
        )

    coefficient_a = float(mle_state["coefficient_a"] if mle_state else refit["coefficient_a"])
    coefficient_b = float(mle_state["coefficient_b"] if mle_state else refit["coefficient_b"])
    coefficient_c = float(refit["coefficient_c"])
    sigma = float(refit["rmse"])
    x_all = _gjb18a_x(active["gjb_response"].to_numpy(dtype=float), coefficient_c)
    y_all = active["gjb_y_log10_life"].to_numpy(dtype=float)
    y_hat_all = coefficient_a + coefficient_b * x_all
    residual_all = y_all - y_hat_all

    # Step 7 - Attach fitted coordinates and residual diagnostics to active rows.
    active["gjb_x"] = x_all
    active["gjb_yhat_log10_life"] = y_hat_all
    active["gjb_residual_log10_life"] = residual_all
    active["gjb_life_fit"] = np.power(10.0, y_hat_all)
    active["gjb_abs_residual_log10_life"] = np.abs(residual_all)
    active["gjb_removed_outlier"] = False
    active["gjb_level"] = _level_values(active, level_column, replicate_decimals)

    removed = prepared[prepared["gjb_row_id"].isin(removed_ids)].copy()
    # Step 8 - Preserve removed rows with diagnostics so reviewers can audit outlier decisions.
    if not removed.empty:
        removed["gjb_removed_outlier"] = True
        removed["gjb_x"] = _gjb18a_x(removed["gjb_response"].to_numpy(dtype=float), coefficient_c)
        removed["gjb_yhat_log10_life"] = coefficient_a + coefficient_b * removed["gjb_x"]
        removed["gjb_residual_log10_life"] = (
            removed["gjb_y_log10_life"] - removed["gjb_yhat_log10_life"]
        )
        removed["gjb_life_fit"] = np.power(10.0, removed["gjb_yhat_log10_life"])
        removed["gjb_abs_residual_log10_life"] = np.abs(removed["gjb_residual_log10_life"])
        removed["gjb_level"] = _level_values(removed, level_column, replicate_decimals)

    data_out = active.copy()
    runout_data = data_out[~data_out["gjb_is_failure"].astype(bool)].copy().reset_index(drop=True)
    level_data = data_out[data_out["gjb_is_failure"].astype(bool)].copy()
    level_stats = _level_statistics(level_data, coefficient_a, coefficient_b)

    # Step 9 - Sample the engineering curve used by CSV output and Origin plots.
    curve = _gjb18a_curve(
        data_out,
        runout_data,
        coefficient_a,
        coefficient_b,
        coefficient_c,
        fit_points,
    )

    fit_sample = final_state["refit_data"].copy()
    x_fit = _gjb18a_x(fit_sample["gjb_response"].to_numpy(dtype=float), coefficient_c)
    y_fit = fit_sample["gjb_y_log10_life"].to_numpy(dtype=float)
    y_fit_hat = coefficient_a + coefficient_b * x_fit
    fit_residual = y_fit - y_fit_hat
    # Step 10 - Compute summary statistics and interval quantities.
    residual_sum_squares = float(np.sum(fit_residual**2))
    total_sum_squares = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = 1.0 if total_sum_squares == 0.0 else 1.0 - residual_sum_squares / total_sum_squares
    degrees_of_freedom = max(1, len(fit_sample) - 3)
    t_critical = float(stats.t.ppf((1.0 + confidence) / 2.0, degrees_of_freedom))
    standard_error_a = float(mle_state["standard_error_a"] if mle_state else refit["standard_error_a"])
    standard_error_b = float(mle_state["standard_error_b"] if mle_state else refit["standard_error_b"])
    standard_error_c = float(refit["standard_error_c"])
    f_band_critical = float(stats.f.ppf(confidence, 3, degrees_of_freedom))
    simultaneous_band_factor = float(np.sqrt(3 * f_band_critical))
    life_response_coefficient_a = float(np.power(10.0, coefficient_a))
    life_response_coefficient_b = float(coefficient_b)

    x_mean = float(np.nanmean(x_fit))
    y_mean = float(np.nanmean(y_fit))
    x_delta = x_fit - x_mean
    y_delta = y_fit - y_mean
    sxx = float(np.nansum(x_delta**2))
    sxy = float(np.nansum(x_delta * y_delta))

    # Step 11 - Add final document-style audit steps that do not replace legacy summary fields.
    if mle_state is not None:
        step09_table = _gjb18a_final_mle_audit_table(mle_state["likelihood"])
        step09_status = "completed" if bool(mle_state["success"]) else "warning"
        step09_parameters_out = {
            "A1_mle": mle_state["coefficient_a"],
            "A2_mle": mle_state["coefficient_b"],
            "A4_fixed": coefficient_c,
            "log_likelihood": mle_state["log_likelihood"],
            "negative_log_likelihood": mle_state["negative_log_likelihood"],
            "success": bool(mle_state["success"]),
            "optimizer_message": mle_state["optimizer_message"],
            "runout_count": active_runout_count,
            "active_rows_after_outlier_removal": int(len(active)),
            "mle_active_rows": mle_state["mle_active_rows"],
            "mle_input_rows": mle_state["mle_input_rows"],
            "mle_failure_rows": mle_state["mle_failure_rows"],
            "mle_runout_rows": mle_state["mle_runout_rows"],
            "mle_domain_ignored_rows": mle_state["mle_domain_ignored_rows"],
            "mle_domain_ignored_row_ids": mle_state["mle_domain_ignored_row_ids"],
            "mle_domain_invalid_rows": mle_state["mle_domain_invalid_rows"],
        }
    else:
        step09_table = _gjb18a_final_mle_skipped_table(active, coefficient_c)
        step09_status = "skipped"
        step09_parameters_out = {
            "A4_fixed": coefficient_c,
            "runout_count": active_runout_count,
            "active_rows_after_outlier_removal": int(len(active)),
            "mle_input_rows": 0,
            "skipped_reason": str(final_state["stop_reason"]),
        }
    audit_steps["Step09_FinalMLE"] = _audit_step(
        "Step09_FinalMLE",
        "Final MLE",
        step09_status,
        step09_table,
        input_columns=("gjb_response", "gjb_life", "gjb_is_failure"),
        formulas=(
            "failure rows use logpdf",
            "runout rows use logsf",
            "all non-outlier rows with response > A4 participate in final MLE",
            "non-outlier rows with response <= A4 are outside the log-domain and are ignored",
            "A4 and SD_i are fixed; only A1/A2 are corrected",
        ),
        parameters_in={"weighted": bool(variance.get("use_weighted", False))},
        parameters_out=step09_parameters_out,
        decision={
            "runout_uses_logsf": active_runout_count > 0,
            "runout_not_plain_failure": True,
            "all_domain_valid_non_outlier_rows_included": (
                bool(
                    mle_state["mle_input_rows"]
                    == len(active) - mle_state["mle_domain_ignored_rows"]
                )
                if mle_state is not None
                else False
            ),
            "ignored_when_response_le_A4": (
                bool(mle_state["mle_domain_ignored_rows"] > 0) if mle_state is not None else False
            ),
        },
        warnings=(() if mle_state is not None else (str(final_state["stop_reason"]),)),
    )

    final_residual_statistics = _gjb18a_final_residual_statistics(
        fit_sample,
        coefficient_a,
        coefficient_b,
        coefficient_c,
        variance,
    )
    audit_steps["Step10_FinalResidualStatistics"] = _audit_step(
        "Step10_FinalResidualStatistics",
        "Final residual statistics",
        "completed",
        final_residual_statistics,
        input_columns=("gjb_y_log10_life", "gjb_response"),
        formulas=(
            "unweighted RMSE = sqrt(sum(R_i^2)/(n-k))",
            "weighted WR_i = R_i/h_i; RMSE = sqrt(sum(WR_i^2)/(n-k)); SD_i = RMSE*h_i",
        ),
        parameters_in={"k": 3},
        parameters_out={
            "RMSE": float(final_residual_statistics["RMSE"].iloc[0]),
            "weighted": bool(final_residual_statistics["weighted"].iloc[0]),
            "df": int(final_residual_statistics["df"].iloc[0]),
        },
        decision={"legacy_r2_unchanged": True},
        warnings=(),
    )
    model_assessment = _gjb18a_model_assessment(final_residual_statistics)
    audit_steps["Step11_ModelAssessment"] = _audit_step(
        "Step11_ModelAssessment",
        "Model assessment",
        "completed",
        model_assessment,
        input_columns=("standardized_residual", "gjb_response"),
        formulas=(
            "D = sum((SR_i - SR_{i-1})^2) / sum(SR_i^2)",
            "Dcrit = 2 - 4.73 / n^0.555",
        ),
        parameters_in={"sort_by": "gjb_response"},
        parameters_out={
            "D": float(model_assessment["D"].iloc[0]) if len(model_assessment) else np.nan,
            "Dcrit": float(model_assessment["Dcrit"].iloc[0]) if len(model_assessment) else np.nan,
            "D_lt_Dcrit": bool(model_assessment["D_lt_Dcrit"].iloc[0]) if len(model_assessment) else False,
            "multi_strain_ratio_residual_mean_check": "not applicable",
        },
        decision={
            "possible_misfit": bool(model_assessment["possible_misfit"].iloc[0]) if len(model_assessment) else False,
            "reason": "single response is used directly as equivalent strain; multi-ratio residual mean check is not applicable",
        },
        warnings=(),
    )
    r2_document_style = _gjb18a_document_style_r2(final_residual_statistics, float(r2))
    audit_steps["Step12_R2_DocumentStyle"] = _audit_step(
        "Step12_R2_DocumentStyle",
        "Document-style R2",
        "completed",
        r2_document_style,
        input_columns=("y", "h", "RMSE"),
        formulas=("R2 = 1 - RMSE^2 / RTE^2",),
        parameters_in={"old_r2_log_life": float(r2)},
        parameters_out={
            "r2_document_style": float(r2_document_style["r2_document_style"].iloc[0]),
            "weighted": bool(r2_document_style["weighted"].iloc[0]),
            "RMSE": float(r2_document_style["RMSE"].iloc[0]),
            "RTE": float(r2_document_style["RTE"].iloc[0]),
        },
        decision={"old_r2_log_life_preserved": True},
        warnings=(),
    )

    for step_id, steps in iteration_audit_steps.items():
        audit_steps[step_id] = _combine_audit_steps(step_id, steps)
    decision_log = pd.DataFrame(decision_rows)

    # Step 12 - Collect step-by-step review tables for CSV and Origin workbooks.
    extra_tables = _gjb18a_extra_tables(
        iteration_records,
        outlier_records,
        residuals,
        removed,
        mle_state,
        final_state,
        decision_log,
        final_residual_statistics,
        model_assessment,
        r2_document_style,
        step09_table,
    )

    if not significance_passed:
        optimizer_success = False
        optimizer_message = str(final_state["stop_reason"])
    elif mle_state is not None:
        optimizer_success = bool(mle_state["success"])
        optimizer_message = str(mle_state["optimizer_message"])
    else:
        optimizer_success = None
        optimizer_message = ""

    # Step 12 - Package scalar results and data tables for callers.
    result = GJBFitResult(
        life_column=life_column,
        response_column=response_column,
        model=GJB_METHOD,
        model_name="GJB/Z 18A 9.3.2 Formula 136",
        confidence=confidence,
        x_transform="log",
        parameter_count=3,
        points=len(data_out),
        degrees_of_freedom=degrees_of_freedom,
        coefficient_a=coefficient_a,
        coefficient_b=coefficient_b,
        coefficient_c=coefficient_c,
        threshold=coefficient_c,
        x_mean=x_mean,
        y_mean=y_mean,
        sxx=sxx,
        sxy=sxy,
        residual_sum_squares=residual_sum_squares,
        sigma_squared=float(sigma**2),
        sigma=sigma,
        r2=float(r2),
        rmse_log_life=float(np.sqrt(np.mean(fit_residual**2))),
        standard_error_a=standard_error_a,
        standard_error_b=standard_error_b,
        standard_error_c=standard_error_c,
        t_critical=t_critical,
        f_band_critical=f_band_critical,
        simultaneous_band_factor=simultaneous_band_factor,
        coefficient_a_lower=float(coefficient_a - t_critical * standard_error_a),
        coefficient_a_upper=float(coefficient_a + t_critical * standard_error_a),
        coefficient_b_lower=float(coefficient_b - t_critical * standard_error_b),
        coefficient_b_upper=float(coefficient_b + t_critical * standard_error_b),
        coefficient_c_lower=float(coefficient_c - t_critical * standard_error_c),
        coefficient_c_upper=float(coefficient_c + t_critical * standard_error_c),
        sigma_lower=None,
        sigma_upper=None,
        x_min=float(np.nanmin(data_out["gjb_x"])),
        x_max=float(np.nanmax(data_out["gjb_x"])),
        life_min=float(data_out["gjb_life"].min()),
        life_max=float(data_out["gjb_life"].max()),
        response_min=float(data_out["gjb_response"].min()),
        response_max=float(data_out["gjb_response"].max()),
        replication_percent=0.0,
        life_response_coefficient_a=life_response_coefficient_a,
        life_response_coefficient_b=life_response_coefficient_b,
        log_likelihood=None if mle_state is None else float(mle_state["log_likelihood"]),
        negative_log_likelihood=(
            None if mle_state is None else float(mle_state["negative_log_likelihood"])
        ),
        n_failure=int(data_out["gjb_is_failure"].astype(bool).sum()),
        n_runout=int((~data_out["gjb_is_failure"].astype(bool)).sum()),
        success=optimizer_success,
        optimizer_message=optimizer_message,
        log_life_formula=_log_life_formula(
            life_column,
            response_column,
            coefficient_a,
            coefficient_b,
            coefficient_c,
        ),
        life_formula=_life_formula(
            life_column,
            response_column,
            coefficient_a,
            coefficient_b,
            coefficient_c,
        ),
        life_response_formula=_life_response_formula(
            life_column,
            response_column,
            life_response_coefficient_a,
            life_response_coefficient_b,
            coefficient_c,
        ),
        response_life_formula=_response_life_formula(
            life_column,
            response_column,
            coefficient_a,
            coefficient_b,
            coefficient_c,
        ),
        warnings=tuple(warnings),
    )
    return GJBFit(
        result=result,
        data=data_out.reset_index(drop=True),
        curve=curve,
        level_stats=level_stats,
        runout_data=runout_data,
        extra_tables=extra_tables,
        audit_steps=audit_steps,
        decision_log=decision_log,
    )


def _gjb18a_iteration(
    active: pd.DataFrame,
    *,
    e_min_failure: float,
    confidence: float,
    iteration: int,
    outlier_mode: str,
) -> dict[str, object]:
    """Run one auditable GJB/Z 18A 9.3.2 refit/outlier iteration."""
    audit_steps: dict[str, GJBAuditStep] = {}
    decision_rows: list[dict[str, object]] = []

    # Step 1 - Work with failure rows for initialization and parameter significance checks.
    failures = active[active["gjb_is_failure"].astype(bool)].copy().reset_index(drop=True)
    if len(failures) < 4:
        raise ValueError("gjb18a-strain outlier iteration left fewer than four failure points.")

    # Step 2 - Formula 136 initialization: A4 starts at half of the minimum failure strain.
    a4_initial = 0.5 * e_min_failure
    x_initial = _gjb18a_x(failures["gjb_response"].to_numpy(dtype=float), a4_initial)
    y_failure = failures["gjb_y_log10_life"].to_numpy(dtype=float)
    raw_a1_initial, raw_a2_initial, initial_cov = _weighted_linear_fit(x_initial, y_failure)
    a1_initial = raw_a1_initial
    a2_initial = raw_a2_initial
    forced_negative_slope = a2_initial >= 0.0
    if a2_initial >= 0.0:
        a2_initial = -1.0 if a2_initial == 0.0 else -abs(a2_initial)
    y_pred_initial = a1_initial + a2_initial * x_initial
    residual_initial = y_failure - y_pred_initial
    initial_ols = failures[["gjb_row_id", "gjb_response", "gjb_y_log10_life"]].copy()
    initial_ols.insert(0, "iteration", iteration)
    initial_ols = initial_ols.rename(
        columns={
            "gjb_response": "x_response",
            "gjb_y_log10_life": "y_log10_life",
        }
    )
    initial_ols["min_failure_response"] = e_min_failure
    initial_ols["A1_initial"] = a1_initial
    initial_ols["A2_initial"] = a2_initial
    initial_ols["A4_initial"] = a4_initial
    initial_ols["x_minus_A4_initial"] = initial_ols["x_response"].astype(float) - a4_initial
    initial_ols["X_initial_log10"] = x_initial
    initial_ols["y_pred_initial_ols"] = y_pred_initial
    initial_ols["residual_initial_ols"] = residual_initial
    initial_ols["residual_squared_initial_ols"] = residual_initial**2
    audit_steps["Step01_InitialOLS"] = _audit_step(
        "Step01_InitialOLS",
        "Initial OLS",
        "completed",
        initial_ols,
        input_columns=("gjb_response", "gjb_y_log10_life"),
        formulas=(
            "A4_initial = 0.5 * min(failure response)",
            "X_initial = log10(response - A4_initial)",
            "y = A1 + A2 * X_initial",
        ),
        parameters_in={"iteration": iteration, "min_failure_response": e_min_failure},
        parameters_out={
            "A1_initial": a1_initial,
            "A2_initial": a2_initial,
            "A4_initial": a4_initial,
            "OLS_SSE": float(np.sum(residual_initial**2)),
            "OLS_covariance": initial_cov,
            "forced_negative_slope": forced_negative_slope,
            "raw_A2_initial": raw_a2_initial,
        },
        decision={
            "forced_negative_slope": forced_negative_slope,
            "reason": (
                "raw OLS slope was non-negative and was forced negative"
                if forced_negative_slope
                else "raw OLS slope was already negative"
            ),
        },
        warnings=(
            ("A2_initial was forced negative because the fatigue-life slope must be negative.",)
            if forced_negative_slope
            else ()
        ),
    )

    # Step 3 - Run the initial nonlinear fit on failure data.
    initial_nls = _gjb18a_nls_fit(
        failures,
        np.array([a1_initial, a2_initial, a4_initial], dtype=float),
        weighted=False,
        variance_model=None,
        e_upper_source=e_min_failure,
    )
    initial_nls_table = _gjb18a_nls_audit_table(
        failures,
        initial_nls,
        iteration,
        "initial_nls",
    )
    audit_steps["Step02_InitialNLS"] = _audit_step(
        "Step02_InitialNLS",
        "Initial nonlinear least squares",
        "completed" if bool(initial_nls["success"]) else "warning",
        initial_nls_table,
        input_columns=("gjb_response", "gjb_y_log10_life"),
        formulas=("y_pred = A1 + A2 * log10(response - A4)", "SSE = sum(residual^2)"),
        parameters_in={
            "initial_parameters": initial_nls["initial_parameters"],
            "lower_bounds": initial_nls["lower_bounds"],
            "upper_bounds": initial_nls["upper_bounds"],
            "A4_lower_bound": initial_nls["lower_bounds"][2],
            "A4_upper_bound": initial_nls["upper_bounds"][2],
        },
        parameters_out={
            "A1_initial_nls": initial_nls["coefficient_a"],
            "A2_initial_nls": initial_nls["coefficient_b"],
            "A4_initial_nls": initial_nls["coefficient_c"],
            "converged": bool(initial_nls["success"]),
            "optimizer_message": initial_nls["optimizer_message"],
            "NLS_SSE": initial_nls["objective"],
            "covariance": initial_nls["covariance"],
            "standard_errors": [
                initial_nls["standard_error_a"],
                initial_nls["standard_error_b"],
                initial_nls["standard_error_c"],
            ],
        },
        decision={"converged": bool(initial_nls["success"])},
        warnings=(() if bool(initial_nls["success"]) else (str(initial_nls["optimizer_message"]),)),
    )

    # Step 4 - Estimate the residual variance model and decide whether weighting is needed.
    variance_model, variance_table = _gjb18a_variance_model(
        failures,
        initial_nls,
        confidence,
        iteration,
    )
    use_weighted = bool(variance_model["use_weighted"])
    audit_steps["Step03_VarianceAnalysis"] = _audit_step(
        "Step03_VarianceAnalysis",
        "Variance analysis",
        "completed",
        variance_table,
        input_columns=("gjb_response", "initial_residual"),
        formulas=(
            "scaled_abs_residual = abs(initial_residual) / sqrt(2/pi)",
            "h = sigma0 + sigma1 / response",
            "weight = 1 / h^2",
        ),
        parameters_in={"confidence": confidence, "iteration": iteration},
        parameters_out=variance_model,
        decision={
            "rule": "use weighted refit when sigma1_lower_90 > 0 and sigma1 > 0",
            "use_weighted": use_weighted,
        },
        warnings=tuple(variance_model.get("warnings", ())),
    )
    decision_rows.append(
        _decision_row(
            iteration,
            "Step03_VarianceAnalysis",
            "sigma1 90% CI lower > 0 ?",
            variance_model["sigma1_lower_90"],
            "sigma1_lower_90 > 0 and sigma1 > 0",
            "use weighted refit" if use_weighted else "use unweighted refit",
            (
                "heteroscedastic variance model selected"
                if use_weighted
                else "sigma1 interval did not support weighting"
            ),
            "VarianceAnalysis",
        )
    )

    # Step 5 - Refit with eligible run-out rows treated as temporary failures.
    refit_data, refit_audit_table = _gjb18a_refit_data(active, e_min_failure)
    audit_steps["Step04_RefitData"] = _audit_step(
        "Step04_RefitData",
        "Refit data selection",
        "completed",
        refit_audit_table,
        input_columns=("gjb_response", "gjb_life", "gjb_is_failure"),
        formulas=("include failure rows and runout rows with response > e_min_failure",),
        parameters_in={"e_min_failure": e_min_failure},
        parameters_out={
            "included_rows": int(refit_audit_table["included_in_refit"].sum()),
            "runout_treated_as_failure": int(refit_audit_table["gjb18a_runout_treated_as_failure"].sum()),
        },
        decision={
            "temporary_failure_rule": "runout response > e_min_failure is included as temporary failure during refit"
        },
        warnings=(),
    )
    refit = _gjb18a_nls_fit(
        refit_data,
        np.array(
            [
                initial_nls["coefficient_a"],
                initial_nls["coefficient_b"],
                initial_nls["coefficient_c"],
            ],
            dtype=float,
        ),
        weighted=bool(variance_model["use_weighted"]),
        variance_model=variance_model,
        e_upper_source=float(refit_data["gjb_response"].min()),
    )
    refit_table = _gjb18a_refit_result_table(refit_data, refit, variance_model, iteration)
    audit_steps["Step05_RefitResult"] = _audit_step(
        "Step05_RefitResult",
        "Refit result",
        "completed" if bool(refit["success"]) else "warning",
        refit_table,
        input_columns=("gjb_response", "gjb_y_log10_life"),
        formulas=(
            "weighted objective = sum((R/h)^2)" if use_weighted else "unweighted objective = sum(R^2)",
            "h = sigma0 + sigma1 / response when weighted",
        ),
        parameters_in={
            "initial_parameters": refit["initial_parameters"],
            "weighted": use_weighted,
        },
        parameters_out={
            "A1_refit": refit["coefficient_a"],
            "A2_refit": refit["coefficient_b"],
            "A4_refit": refit["coefficient_c"],
            "standard_errors": [
                refit["standard_error_a"],
                refit["standard_error_b"],
                refit["standard_error_c"],
            ],
            "covariance": refit["covariance"],
            "converged": bool(refit["success"]),
            "objective": refit["objective"],
        },
        decision={"weighted": use_weighted},
        warnings=(() if bool(refit["success"]) else (str(refit["optimizer_message"]),)),
    )
    # Step 6 - Check A2 and A4 significance before any outlier decision is honored.
    significance = _gjb18a_parameter_significance(refit, confidence, iteration)
    significance_passed = bool(significance["overall_passed"].iloc[0])
    audit_steps["Step06_ParameterSignificance"] = _audit_step(
        "Step06_ParameterSignificance",
        "Parameter significance",
        "passed" if significance_passed else "failed",
        significance,
        input_columns=("coefficient_b", "coefficient_c", "standard_error_b", "standard_error_c"),
        formulas=("A2 passes when A2_upper_90 < 0", "A4 passes when its 90% CI excludes 0"),
        parameters_in={"confidence": confidence, "test_confidence": 0.90},
        parameters_out={"overall_passed": significance_passed},
        decision={
            "A2_upper_90_lt_0": bool(significance.loc[significance["parameter"] == "A2", "passed"].iloc[0]),
            "A4_ci_excludes_0": bool(significance.loc[significance["parameter"] == "A4", "passed"].iloc[0]),
        },
        warnings=tuple(significance.loc[~significance["passed"].astype(bool), "warning"].dropna().astype(str)),
    )
    for _, row in significance.iterrows():
        if row["parameter"] in {"A2", "A4"}:
            decision_rows.append(
                _decision_row(
                    iteration,
                    "Step06_ParameterSignificance",
                    str(row["rule"]),
                    bool(row["passed"]),
                    str(row["rule"]),
                    "passed" if bool(row["passed"]) else "failed",
                    str(row["warning"]),
                    "ParameterSignificance",
                )
            )
    stop_reason = ""
    if not significance_passed:
        stop_reason = (
            "gjb18a-strain stopped because A2 or A4 failed the 90% parameter "
            "significance test after refitting."
        )
    post_significance_fit = refit
    fixed_a4_table = _gjb18a_fixed_a4_skipped_table(
        iteration,
        refit,
        "Skipped because weighted refit was not selected."
        if significance_passed
        else "Skipped because parameter significance did not pass.",
    )
    # Step 7 - If weighted refit was selected, fix A4 and correct A1/A2 linearly.
    if significance_passed and bool(variance_model["use_weighted"]):
        post_significance_fit, fixed_a4_table = _gjb18a_fixed_a4_linear_fit(
            refit_data,
            refit,
            variance_model,
            iteration,
        )
    fixed_a4_performed = bool(fixed_a4_table["performed"].iloc[0]) if len(fixed_a4_table) else False
    audit_steps["Step07_FixedA4LinearFit"] = _audit_step(
        "Step07_FixedA4LinearFit",
        "Fixed A4 linear correction",
        "completed" if fixed_a4_performed else "skipped",
        fixed_a4_table,
        input_columns=("gjb_response", "gjb_y_log10_life", "h"),
        formulas=(
            "A4 is fixed; A1 and A2 are re-estimated only",
            "Y_star = y / h, U = 1 / h, V = X / h",
            "Y_star = A1_corrected * U + A2_corrected * V with no extra intercept",
        ),
        parameters_in={"weighted": use_weighted, "significance_passed": significance_passed},
        parameters_out={
            "performed": fixed_a4_performed,
            "A1": post_significance_fit["coefficient_a"],
            "A2": post_significance_fit["coefficient_b"],
            "A4": post_significance_fit["coefficient_c"],
            "note": "A1^2 and A2^2 in the document are corrected parameters, not squared values.",
            "A3": "not fitted; response is used directly as equivalent strain",
            "equivalence": "_weighted_linear_fit(x, y, weights=1/h^2) is equivalent to no-intercept regression Y*=A1*U+A2*V.",
        },
        decision={
            "rule": "perform only when weighted refit is selected and parameter significance passes",
            "performed": fixed_a4_performed,
        },
        warnings=(),
    )
    decision_rows.append(
        _decision_row(
            iteration,
            "Step07_FixedA4LinearFit",
            "weighted selected and significance passed ?",
            {"weighted": use_weighted, "significance_passed": significance_passed},
            "use_weighted and significance_passed",
            "fixed A4 correction performed" if fixed_a4_performed else "fixed A4 correction skipped",
            str(fixed_a4_table["reason"].iloc[0]) if "reason" in fixed_a4_table else "",
            "FixedA4LinearFit",
        )
    )

    # Step 8 - Compute studentized residuals and decide whether one row should be removed.
    residual_table, residual_summary = _gjb18a_residuals_and_outlier_statistics(
        refit_data,
        post_significance_fit,
        variance_model,
        iteration,
        outlier_mode,
    )
    if not significance_passed:
        residual_summary["remove_outlier"] = False
        residual_summary["removed_row_id"] = None
        residual_summary["outlier_decision_reason"] = (
            "Parameter significance failed; outlier candidate is reported but not removed."
        )
        residual_table["remove_outlier"] = False
        residual_table["removed_row_id"] = None
        residual_table["outlier_decision_reason"] = residual_summary["outlier_decision_reason"]
    remove_outlier = bool(residual_summary["remove_outlier"])
    removed_row_id = residual_summary.get("removed_row_id", None)
    outlier_table = residual_table.copy()
    audit_steps["Step08_ResidualsOutliers"] = _audit_step(
        "Step08_ResidualsOutliers",
        "Residuals and outliers",
        "candidate" if bool(residual_summary["candidate_found"]) else "completed",
        residual_table,
        input_columns=("gjb_response", "gjb_y_log10_life"),
        formulas=("G = max(abs(studentized residual))", "critical = t(1-alpha/(2n), n-k-1)"),
        parameters_in={"outlier_mode": outlier_mode, "alpha": residual_summary["alpha"]},
        parameters_out=residual_summary,
        decision={
            "candidate_found": bool(residual_summary["candidate_found"]),
            "remove_outlier": remove_outlier,
            "reason": residual_summary["outlier_decision_reason"],
        },
        warnings=(),
    )
    decision_rows.append(
        _decision_row(
            iteration,
            "Step08_ResidualsOutliers",
            "G > critical ?",
            {"G": residual_summary["G"], "critical": residual_summary["critical"]},
            "G > critical",
            (
                "outlier removed"
                if remove_outlier
                else (
                    "outlier candidate reported only"
                    if bool(residual_summary["candidate_found"]) and outlier_mode == "report-only"
                    else "no outlier removed"
                )
            ),
            str(residual_summary["outlier_decision_reason"]),
            "Residuals",
        )
    )

    # Step 9 - Return both scalar decisions and review tables for this iteration.
    return {
        "iteration": iteration,
        "initial_nls": initial_nls,
        "variance_model": variance_model,
        "refit": refit,
        "refit_data": refit_data,
        "post_significance_fit": post_significance_fit,
        "significance_passed": significance_passed,
        "stop_reason": stop_reason,
        "residual_table": residual_table,
        "outlier_table": outlier_table,
        "remove_outlier": remove_outlier if significance_passed else False,
        "removed_row_id": removed_row_id,
        "outlier_mode": outlier_mode,
        "audit_steps": audit_steps,
        "decision_rows": decision_rows,
        "tables": [
            ("InitialOLS", initial_ols),
            ("InitialNLS", initial_nls_table),
            ("VarianceAnalysis", variance_table),
            ("RefitData", refit_audit_table),
            ("RefitResult", refit_table),
            ("ParameterSignificance", significance),
            ("FixedA4LinearFit", fixed_a4_table),
        ],
    }


def _gjb18a_x(response: np.ndarray, coefficient_c: float) -> np.ndarray:
    shifted = np.asarray(response, dtype=float) - float(coefficient_c)
    x = np.full_like(shifted, np.nan, dtype=float)
    mask = shifted > 0.0
    x[mask] = np.log10(shifted[mask])
    return x


def _weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = np.asarray(weights, dtype=float)[mask]
    if len(x) < 2:
        raise ValueError("At least two points are required for linear regression.")
    design = np.column_stack((np.ones_like(x), x))
    root_w = np.sqrt(np.maximum(weights, 1e-300))
    weighted_design = design * root_w[:, None]
    weighted_y = y * root_w
    beta = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
    residual = y - design @ beta
    df = max(1, len(x) - 2)
    sigma_squared = float(np.sum(weights * residual**2) / df)
    covariance = sigma_squared * np.linalg.pinv(weighted_design.T @ weighted_design)
    return float(beta[0]), float(beta[1]), covariance


def _gjb18a_nls_fit(
    frame: pd.DataFrame,
    initial: np.ndarray,
    *,
    weighted: bool,
    variance_model: dict[str, float] | None,
    e_upper_source: float,
) -> dict[str, object]:
    response = frame["gjb_response"].to_numpy(dtype=float)
    y = frame["gjb_y_log10_life"].to_numpy(dtype=float)
    epsilon = max(abs(float(e_upper_source)) * 1e-10, 1e-15)
    lower = np.array([-np.inf, -np.inf, 0.0], dtype=float)
    upper = np.array([np.inf, -1e-12, float(e_upper_source) - epsilon], dtype=float)
    start = np.asarray(initial, dtype=float).copy()
    start[1] = min(start[1], -1e-8)
    start[2] = min(max(start[2], lower[2] + epsilon), upper[2] - epsilon)
    optimizer_start = start.copy()

    result = optimize.least_squares(
        _gjb18a_nls_residual,
        x0=start,
        args=(response, y, weighted, variance_model),
        bounds=(lower, upper),
        x_scale=np.array([1.0, 1.0, max(abs(float(e_upper_source)), 1.0)], dtype=float),
        max_nfev=4000,
    )
    if not np.all(np.isfinite(result.fun)):
        raise ValueError("gjb18a-strain nonlinear fit produced non-finite residuals.")

    a, b, c = [float(value) for value in result.x]
    x = _gjb18a_x(response, c)
    y_hat = a + b * x
    residual = y - y_hat
    h_values = _gjb18a_h_values(response, variance_model) if weighted else np.ones_like(response)
    weighted_residual = residual / h_values
    objective_component = weighted_residual**2
    df = max(1, len(frame) - 3)
    rmse = float(np.sqrt(np.sum(weighted_residual**2) / df))
    covariance = rmse**2 * np.linalg.pinv(result.jac.T @ result.jac)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "coefficient_a": a,
        "coefficient_b": b,
        "coefficient_c": c,
        "x": x,
        "y_hat": y_hat,
        "residual": residual,
        "weighted_residual": weighted_residual,
        "h_values": h_values,
        "weights": 1.0 / np.power(h_values, 2),
        "objective_component": objective_component,
        "objective": float(np.sum(objective_component)),
        "rmse": rmse,
        "covariance": covariance,
        "standard_error_a": float(standard_errors[0]),
        "standard_error_b": float(standard_errors[1]),
        "standard_error_c": float(standard_errors[2]),
        "success": bool(result.success),
        "optimizer_message": str(result.message),
        "weighted": weighted,
        "initial_parameters": optimizer_start,
        "lower_bounds": lower,
        "upper_bounds": upper,
    }


def _gjb18a_nls_residual(
    parameters: np.ndarray,
    response: np.ndarray,
    y: np.ndarray,
    weighted: bool,
    variance_model: dict[str, float] | None,
) -> np.ndarray:
    a, b, c = [float(value) for value in parameters]
    x = _gjb18a_x(response, c)
    if np.any(~np.isfinite(x)):
        return np.full_like(y, 1e150, dtype=float)
    residual = a + b * x - y
    if weighted:
        h_values = _gjb18a_h_values(response, variance_model)
        residual = residual / h_values
    return residual


def _gjb18a_variance_model(
    failures: pd.DataFrame,
    initial_nls: dict[str, object],
    confidence: float,
    iteration: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    # GJB/Z 18A 9.3.2.3.3: regress absolute residual scale against inverse strain.
    response = failures["gjb_response"].to_numpy(dtype=float)
    residual = np.asarray(initial_nls["residual"], dtype=float)
    inverse_response = 1.0 / response
    scaled_abs_residual = np.abs(residual) / np.sqrt(2.0 / np.pi)
    design = np.column_stack((np.ones_like(inverse_response), inverse_response))
    beta = np.linalg.lstsq(design, scaled_abs_residual, rcond=None)[0]
    sigma0 = float(beta[0])
    sigma1 = float(beta[1])
    force_zero_intercept = sigma0 < 0.0
    if force_zero_intercept:
        sigma0 = 0.0
        denom = float(np.sum(inverse_response**2))
        sigma1 = 0.0 if denom <= 0.0 else float(np.sum(inverse_response * scaled_abs_residual) / denom)
        fitted = sigma1 * inverse_response
        df = max(1, len(response) - 1)
        x_var = denom
    else:
        fitted = sigma0 + sigma1 * inverse_response
        df = max(1, len(response) - 2)
        x_var = float(np.sum((inverse_response - np.mean(inverse_response)) ** 2))
    residual_scale = scaled_abs_residual - fitted
    sigma_scale = float(np.sqrt(np.sum(residual_scale**2) / df)) if df > 0 else 0.0
    standard_error_sigma1 = (
        float(sigma_scale / np.sqrt(x_var)) if x_var > 0.0 and np.isfinite(sigma_scale) else np.inf
    )
    t90 = float(stats.t.ppf(0.95, df))
    sigma1_lower = float(sigma1 - t90 * standard_error_sigma1)
    sigma1_upper = float(sigma1 + t90 * standard_error_sigma1)
    use_weighted = bool(sigma1_lower > 0.0 and sigma1 > 0.0)
    h_raw = sigma0 + sigma1 / response
    h_nonpositive_mask = h_raw <= 0.0
    h_values = np.maximum(h_raw, 1e-12)
    variance_warnings: list[str] = []
    if force_zero_intercept:
        variance_warnings.append("sigma0 was negative and the variance model was refit through zero.")
    if np.any(h_nonpositive_mask):
        variance_warnings.append(
            "Raw h = sigma0 + sigma1/response had non-positive value(s); numerical clipping to 1e-12 was applied."
        )
    variance_model = {
        "sigma0": sigma0,
        "sigma1": sigma1,
        "sigma1_lower_90": sigma1_lower,
        "sigma1_upper_90": sigma1_upper,
        "standard_error_sigma1": standard_error_sigma1,
        "use_weighted": use_weighted,
        "force_zero_intercept": force_zero_intercept,
        "h_nonpositive_count": int(np.sum(h_nonpositive_mask)),
        "h_values_clipped": bool(np.any(h_nonpositive_mask)),
        "warnings": tuple(variance_warnings),
    }
    table = failures[
        ["gjb_row_id", "gjb_response", "gjb_life", "gjb_y_log10_life"]
    ].copy()
    table["iteration"] = iteration
    table["inverse_response"] = inverse_response
    table["initial_residual"] = residual
    table["scaled_abs_residual"] = scaled_abs_residual
    table["sigma0"] = sigma0
    table["sigma1"] = sigma1
    table["h_raw"] = h_raw
    table["h"] = h_values
    table["weight"] = 1.0 / np.power(h_values, 2)
    table["variance_fit"] = fitted
    table["variance_residual"] = residual_scale
    table["sigma1_lower_90"] = sigma1_lower
    table["sigma1_upper_90"] = sigma1_upper
    table["use_weighted"] = use_weighted
    table["force_zero_intercept"] = force_zero_intercept
    table["h_raw_nonpositive"] = h_nonpositive_mask
    return variance_model, table


def _gjb18a_h_values(
    response: np.ndarray,
    variance_model: dict[str, float] | None,
) -> np.ndarray:
    response = np.asarray(response, dtype=float)
    if not variance_model or not bool(variance_model.get("use_weighted", False)):
        return np.ones_like(response, dtype=float)
    values = float(variance_model["sigma0"]) + float(variance_model["sigma1"]) / response
    return np.maximum(values, 1e-12)


def _gjb18a_refit_data(active: pd.DataFrame, e_min_failure: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select refit rows and audit temporary-failure treatment.

    GJB/Z 18A 9.3.2 refit uses original failures plus run-out rows whose
    response exceeds the minimum failure response.  Those run-out rows are
    temporary failures only for this refit stage; they remain censored in final
    likelihood output.
    """
    failure = active["gjb_is_failure"].astype(bool)
    high_runout = (~failure) & (active["gjb_response"].astype(float) > e_min_failure)
    audit = active.copy().reset_index(drop=True)
    audit["gjb18a_original_is_failure"] = audit["gjb_is_failure"].astype(bool)
    audit["gjb18a_runout_treated_as_failure"] = high_runout.to_numpy(dtype=bool)
    audit["gjb18a_temporary_failure"] = (failure | high_runout).to_numpy(dtype=bool)
    audit["e_min_failure"] = e_min_failure
    audit["included_in_refit"] = audit["gjb18a_temporary_failure"]
    audit["inclusion_reason"] = np.select(
        [
            audit["gjb18a_original_is_failure"].to_numpy(dtype=bool),
            audit["gjb18a_runout_treated_as_failure"].to_numpy(dtype=bool),
        ],
        [
            "original failure included",
            "runout response > e_min_failure; temporarily included as failure",
        ],
        default="runout response <= e_min_failure; excluded from refit",
    )
    refit = audit[audit["included_in_refit"].astype(bool)].copy().reset_index(drop=True)
    refit["gjb18a_temporary_failure"] = True
    if len(refit) < 4:
        raise ValueError("gjb18a-strain refit requires at least four temporary failure points.")
    return refit, audit


def _gjb18a_fit_table(
    fit: dict[str, object],
    iteration: int,
    stage: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "iteration": iteration,
                "stage": stage,
                "coefficient_a": fit["coefficient_a"],
                "coefficient_b": fit["coefficient_b"],
                "coefficient_c": fit["coefficient_c"],
                "standard_error_a": fit["standard_error_a"],
                "standard_error_b": fit["standard_error_b"],
                "standard_error_c": fit["standard_error_c"],
                "rmse": fit["rmse"],
                "weighted": fit["weighted"],
                "success": fit["success"],
                "optimizer_message": fit["optimizer_message"],
            }
        ]
    )


def _gjb18a_nls_audit_table(
    frame: pd.DataFrame,
    fit: dict[str, object],
    iteration: int,
    stage: str,
) -> pd.DataFrame:
    """Return the row-level NLS audit table for Formula 136."""
    table = frame[["gjb_row_id", "gjb_response", "gjb_y_log10_life"]].copy()
    table["iteration"] = iteration
    table["stage"] = stage
    table["A1_initial_nls"] = fit["coefficient_a"]
    table["A2_initial_nls"] = fit["coefficient_b"]
    table["A4_initial_nls"] = fit["coefficient_c"]
    table["X_nls"] = np.asarray(fit["x"], dtype=float)
    table["y_pred_initial_nls"] = np.asarray(fit["y_hat"], dtype=float)
    table["residual_initial_nls"] = np.asarray(fit["residual"], dtype=float)
    table["residual_squared_initial_nls"] = np.asarray(fit["residual"], dtype=float) ** 2
    table["weighted"] = bool(fit["weighted"])
    table["optimizer_success"] = bool(fit["success"])
    table["optimizer_message"] = str(fit["optimizer_message"])
    return table


def _gjb18a_refit_result_table(
    frame: pd.DataFrame,
    fit: dict[str, object],
    variance_model: dict[str, float],
    iteration: int,
) -> pd.DataFrame:
    """Return the row-level refit audit table and objective components."""
    table = frame[["gjb_row_id", "gjb_response", "gjb_y_log10_life"]].copy()
    h_values = np.asarray(fit["h_values"], dtype=float)
    residual = np.asarray(fit["residual"], dtype=float)
    weighted = bool(fit["weighted"])
    table["iteration"] = iteration
    table["h"] = h_values
    table["weight"] = 1.0 / np.power(h_values, 2)
    table["weighted"] = weighted
    table["A1_refit"] = fit["coefficient_a"]
    table["A2_refit"] = fit["coefficient_b"]
    table["A4_refit"] = fit["coefficient_c"]
    table["X_refit"] = np.asarray(fit["x"], dtype=float)
    table["y_pred_refit"] = np.asarray(fit["y_hat"], dtype=float)
    table["residual_refit"] = residual
    table["weighted_residual_refit"] = np.asarray(fit["weighted_residual"], dtype=float)
    table["objective_component"] = (
        np.power(residual / h_values, 2)
        if weighted and bool(variance_model.get("use_weighted", False))
        else np.power(residual, 2)
    )
    table["optimizer_success"] = bool(fit["success"])
    table["optimizer_message"] = str(fit["optimizer_message"])
    return table


def _gjb18a_parameter_significance(
    fit: dict[str, object],
    confidence: float,
    iteration: int,
) -> pd.DataFrame:
    """Audit the 90% significance checks for A2 and A4.

    A2 must have an upper 90% confidence bound below zero.  A4 must have a
    90% confidence interval that excludes zero.  Failure of either check stops
    the later outlier-removal decision from changing the dataset.
    """
    df = max(1, int(round(len(np.asarray(fit["residual"])) - 3)))
    t90 = float(stats.t.ppf(0.95, df))
    b = float(fit["coefficient_b"])
    c = float(fit["coefficient_c"])
    se_b = float(fit["standard_error_b"])
    se_c = float(fit["standard_error_c"])
    b_lower = b - t90 * se_b
    b_upper = b + t90 * se_b
    c_lower = c - t90 * se_c
    c_upper = c + t90 * se_c
    a2_pass = bool(b_upper < 0.0)
    a4_pass = bool(c_lower > 0.0 or c_upper < 0.0)
    overall_passed = bool(a2_pass and a4_pass)
    return pd.DataFrame(
        [
            {
                "iteration": iteration,
                "confidence": confidence,
                "test_confidence": 0.90,
                "degrees_of_freedom": df,
                "t_critical_90": t90,
                "parameter": "A2",
                "estimate": b,
                "standard_error": se_b,
                "lower_90": b_lower,
                "upper_90": b_upper,
                "rule": "A2_upper_90 < 0",
                "passed": a2_pass,
                "warning": "" if a2_pass else "A2 is not significantly negative; stop after refit.",
                "overall_passed": overall_passed,
            },
            {
                "iteration": iteration,
                "confidence": confidence,
                "test_confidence": 0.90,
                "degrees_of_freedom": df,
                "t_critical_90": t90,
                "parameter": "A4",
                "estimate": c,
                "standard_error": se_c,
                "lower_90": c_lower,
                "upper_90": c_upper,
                "rule": "A4_lower_90 > 0 or A4_upper_90 < 0",
                "passed": a4_pass,
                "warning": "" if a4_pass else "A4 confidence interval includes zero; stop after refit.",
                "overall_passed": overall_passed,
            },
        ]
    )


def _gjb18a_fixed_a4_linear_fit(
    refit_data: pd.DataFrame,
    refit: dict[str, object],
    variance_model: dict[str, float],
    iteration: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Correct A1/A2 with A4 fixed after weighted fitting.

    The implementation uses ``_weighted_linear_fit(x, y, weights=1/h^2)``.
    This is mathematically equivalent to the no-intercept regression
    ``Y* = A1 * U + A2 * V`` where ``Y*=y/h``, ``U=1/h`` and ``V=X/h``.
    No A3/equivalent-strain recalculation is performed in this project.
    """
    c = float(refit["coefficient_c"])
    response = refit_data["gjb_response"].to_numpy(dtype=float)
    y = refit_data["gjb_y_log10_life"].to_numpy(dtype=float)
    x = _gjb18a_x(response, c)
    h_values = _gjb18a_h_values(response, variance_model)
    weights = 1.0 / np.power(h_values, 2)
    a, b, covariance_ab = _weighted_linear_fit(x, y, weights=weights)
    residual = y - (a + b * x)
    weighted_residual = residual / h_values
    df = max(1, len(refit_data) - 3)
    rmse = float(np.sqrt(np.sum(weighted_residual**2) / df))
    covariance_ab = covariance_ab * (df / max(1, len(refit_data) - 2))
    se_ab = np.sqrt(np.maximum(np.diag(covariance_ab), 0.0))
    corrected = dict(refit)
    corrected.update(
        {
            "coefficient_a": float(a),
            "coefficient_b": float(b),
            "coefficient_c": c,
            "x": x,
            "y_hat": a + b * x,
            "residual": residual,
            "weighted_residual": weighted_residual,
            "rmse": rmse,
            "standard_error_a": float(se_ab[0]),
            "standard_error_b": float(se_ab[1]),
            "standard_error_c": float(refit["standard_error_c"]),
            "weighted": True,
            "success": True,
            "optimizer_message": "A1/A2 corrected by weighted linear regression with fixed A4.",
        }
    )
    table = refit_data.copy()
    table["iteration"] = iteration
    table["stage"] = "FixedA4LinearFit"
    table["performed"] = True
    table["reason"] = "Weighted refit selected; A4 fixed and A1/A2 corrected."
    table["A4_fixed"] = c
    table["A1_corrected"] = corrected["coefficient_a"]
    table["A2_corrected"] = corrected["coefficient_b"]
    table["h"] = h_values
    table["weight"] = weights
    table["X"] = x
    table["Y_star"] = y / h_values
    table["U"] = 1.0 / h_values
    table["V"] = x / h_values
    table["Y_star_pred"] = corrected["coefficient_a"] * table["U"] + corrected["coefficient_b"] * table["V"]
    table["residual_star"] = table["Y_star"] - table["Y_star_pred"]
    table["residual_original_scale"] = residual
    table["weighted_residual"] = weighted_residual
    table["rmse"] = rmse
    table["standard_error_a"] = corrected["standard_error_a"]
    table["standard_error_b"] = corrected["standard_error_b"]
    table["standard_error_c"] = corrected["standard_error_c"]
    return corrected, table


def _gjb18a_fixed_a4_skipped_table(
    iteration: int,
    refit: dict[str, object],
    reason: str,
) -> pd.DataFrame:
    """Return an auditable placeholder when fixed-A4 correction is skipped."""
    return pd.DataFrame(
        [
            {
                "iteration": iteration,
                "stage": "FixedA4LinearFit",
                "performed": False,
                "reason": reason,
                "gjb_row_id": np.nan,
                "gjb_response": np.nan,
                "gjb_y_log10_life": np.nan,
                "A4_fixed": refit["coefficient_c"],
                "A1_corrected": refit["coefficient_a"],
                "A2_corrected": refit["coefficient_b"],
                "h": np.nan,
                "weight": np.nan,
                "X": np.nan,
                "Y_star": np.nan,
                "U": np.nan,
                "V": np.nan,
                "Y_star_pred": np.nan,
                "residual_star": np.nan,
                "residual_original_scale": np.nan,
                "weighted_residual": np.nan,
            }
        ]
    )


def _gjb18a_residuals_and_outlier_statistics(
    refit_data: pd.DataFrame,
    fit: dict[str, object],
    variance_model: dict[str, float],
    iteration: int,
    outlier_mode: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute residual diagnostics and the auditable outlier decision.

    ``auto`` keeps the historical behavior and removes the candidate when
    ``G > critical``.  ``report-only`` records the same candidate but keeps all
    rows active for manual review.
    """
    a = float(fit["coefficient_a"])
    b = float(fit["coefficient_b"])
    c = float(fit["coefficient_c"])
    response = refit_data["gjb_response"].to_numpy(dtype=float)
    y = refit_data["gjb_y_log10_life"].to_numpy(dtype=float)
    x = _gjb18a_x(response, c)
    h_function = _gjb18a_h_values(response, variance_model)
    residual = y - (a + b * x)
    weighted = bool(variance_model.get("use_weighted", False))
    k = 3
    df = max(1, len(refit_data) - k)
    if weighted:
        weighted_residual = residual / h_function
        rmse = float(np.sqrt(np.sum(weighted_residual**2) / df))
        sd_i = rmse * h_function
    else:
        weighted_residual = residual
        rmse = float(np.sqrt(np.sum(residual**2) / df))
        sd_i = np.full_like(residual, rmse)
    standardized = residual / np.maximum(sd_i, 1e-300)
    design = np.column_stack((1.0 / sd_i, x / sd_i))
    hat = design @ np.linalg.pinv(design.T @ design) @ design.T
    leverage = np.clip(np.diag(hat), 0.0, 0.999999)
    deleted_residual_scale = weighted_residual
    rmse_i_squared = (
        df * rmse**2
        - np.power(deleted_residual_scale, 2) / np.maximum(1.0 - leverage, 1e-12)
    ) / max(1, len(refit_data) - k - 1)
    rmse_i = np.sqrt(np.maximum(rmse_i_squared, 1e-300))
    studentized = (
        standardized
        / np.sqrt(np.maximum(1.0 - leverage, 1e-12))
        * (rmse / np.maximum(rmse_i, 1e-300))
    )
    n = len(refit_data)
    outlier_df = max(1, n - k - 1)
    alpha = 0.05
    critical = float(stats.t.ppf(1.0 - alpha / (2.0 * n), outlier_df))
    abs_t = np.abs(studentized)
    max_index = int(np.nanargmax(abs_t)) if len(abs_t) else 0
    G = float(abs_t[max_index]) if len(abs_t) else 0.0
    candidate_found = bool(np.isfinite(G) and G > critical)
    remove_outlier = bool(candidate_found and outlier_mode == "auto")
    candidate_row_id = int(refit_data.iloc[max_index]["gjb_row_id"]) if len(refit_data) else None
    removed_row_id = candidate_row_id if remove_outlier else None
    if candidate_found and outlier_mode == "auto":
        decision_reason = f"G={G:.6g} > critical={critical:.6g}; auto mode removes row {candidate_row_id}."
    elif candidate_found:
        decision_reason = (
            f"G={G:.6g} > critical={critical:.6g}; report-only mode keeps row {candidate_row_id}."
        )
    else:
        decision_reason = f"G={G:.6g} <= critical={critical:.6g}; no outlier candidate."
    table = refit_data.copy()
    table["iteration"] = iteration
    table["gjb18a_x"] = x
    table["gjb18a_yhat"] = a + b * x
    table["gjb18a_residual"] = residual
    table["gjb18a_h_function"] = h_function
    table["gjb18a_weighted_residual"] = weighted_residual
    table["gjb18a_rmse"] = rmse
    table["gjb18a_sd_i"] = sd_i
    table["gjb18a_standardized_residual"] = standardized
    table["gjb18a_leverage"] = leverage
    table["gjb18a_rmse_i"] = rmse_i
    table["gjb18a_studentized_residual"] = studentized
    table["gjb18a_abs_studentized_residual"] = abs_t
    table["gjb18a_outlier_critical"] = critical
    table["gjb18a_is_max_abs_t"] = False
    table["k"] = k
    table["df"] = df
    table["alpha"] = alpha
    table["G"] = G
    table["critical"] = critical
    table["candidate_row_id"] = candidate_row_id
    table["candidate_outlier"] = False
    table["remove_outlier"] = False
    table["removed_row_id"] = removed_row_id
    table["outlier_mode"] = outlier_mode
    table["outlier_decision_reason"] = decision_reason
    if len(table):
        table.loc[table.index[max_index], "gjb18a_is_max_abs_t"] = True
        table.loc[table.index[max_index], "candidate_outlier"] = candidate_found
        table.loc[table.index[max_index], "remove_outlier"] = remove_outlier
    summary = {
        "G": G,
        "critical": critical,
        "alpha": alpha,
        "k": k,
        "df": df,
        "candidate_found": candidate_found,
        "candidate_row_id": candidate_row_id,
        "remove_outlier": remove_outlier,
        "removed_row_id": removed_row_id,
        "outlier_mode": outlier_mode,
        "outlier_decision_reason": decision_reason,
    }
    return table, summary


def _gjb18a_final_mle(
    active: pd.DataFrame,
    fit: dict[str, object],
    variance_model: dict[str, float],
) -> dict[str, object]:
    """Apply the final right-censored likelihood correction.

    GJB/Z 18A 9.3.2.7.2 fixes A4 and SD_i and corrects only A1/A2.  Every
    non-outlier active row with ``response > A4`` participates in the likelihood.
    Rows at or below A4 are outside the fitted log-domain and are explicitly
    ignored in the audit table.  Failure rows contribute normal ``logpdf`` terms;
    run-out rows contribute survival ``logsf`` terms and are not converted to
    ordinary failure points.
    """
    c = float(fit["coefficient_c"])
    response = active["gjb_response"].to_numpy(dtype=float)
    y = active["gjb_y_log10_life"].to_numpy(dtype=float)
    failures = active["gjb_is_failure"].astype(bool).to_numpy()
    domain = response > c
    response_fit = response[domain]
    y_fit = y[domain]
    failures_fit = failures[domain]
    ignored = active[~domain].copy()
    ignored_ids = ignored["gjb_row_id"].astype(int).tolist()
    if len(y_fit) < 3:
        raise ValueError("gjb18a-strain final MLE has fewer than three domain-valid non-outlier rows.")
    x = _gjb18a_x(response_fit, c)
    h_values = _gjb18a_h_values(response_fit, variance_model)
    sd_i = float(fit["rmse"]) * h_values
    initial_b = min(float(fit["coefficient_b"]), -1e-8)
    initial = np.array([float(fit["coefficient_a"]), np.log(max(-initial_b, 1e-8))])
    result = optimize.minimize(
        _gjb18a_final_mle_negative_log_likelihood,
        initial,
        args=(x, y_fit, failures_fit, sd_i),
        method="L-BFGS-B",
        bounds=[(None, None), (-50.0, 50.0)],
    )
    a, log_minus_b = [float(value) for value in result.x]
    b = -float(np.exp(log_minus_b))
    nll = float(result.fun)
    hessian = _numerical_hessian(
        lambda params: _gjb18a_final_mle_negative_log_likelihood(
            params,
            x,
            y_fit,
            failures_fit,
            sd_i,
        ),
        result.x,
    )
    covariance_transformed = np.linalg.pinv(hessian)
    jacobian_original = np.diag([1.0, b])
    covariance_original = jacobian_original @ covariance_transformed @ jacobian_original.T
    standard_errors = np.sqrt(np.maximum(np.diag(covariance_original), 0.0))
    mu = a + b * x
    likelihood = active[domain].copy().reset_index(drop=True)
    likelihood["domain_valid"] = True
    likelihood["included_in_final_mle"] = True
    likelihood["A4_final_mle"] = c
    likelihood["x"] = x
    likelihood["sd_i"] = sd_i
    likelihood["mu"] = mu
    likelihood["w"] = (y_fit - mu) / sd_i
    likelihood["logpdf"] = stats.norm.logpdf(likelihood["w"]) - np.log(sd_i)
    likelihood["logsf"] = stats.norm.logsf(likelihood["w"])
    likelihood["likelihood_type"] = np.where(failures_fit, "logpdf", "logsf")
    likelihood["log_likelihood_i"] = np.where(
        failures_fit,
        likelihood["logpdf"],
        likelihood["logsf"],
    )
    likelihood["gjb18a_x"] = likelihood["x"]
    likelihood["gjb18a_sd_i"] = likelihood["sd_i"]
    likelihood["gjb18a_mu"] = likelihood["mu"]
    likelihood["gjb18a_w"] = likelihood["w"]
    likelihood["gjb18a_logpdf"] = likelihood["logpdf"]
    likelihood["gjb18a_logsf"] = likelihood["logsf"]
    likelihood["gjb18a_log_likelihood_i"] = likelihood["log_likelihood_i"]
    if not ignored.empty:
        ignored["domain_valid"] = False
        ignored["included_in_final_mle"] = False
        ignored["A4_final_mle"] = c
        ignored["x"] = np.nan
        ignored["sd_i"] = np.nan
        ignored["mu"] = np.nan
        ignored["w"] = np.nan
        ignored["logpdf"] = np.nan
        ignored["logsf"] = np.nan
        ignored["likelihood_type"] = "ignored_response_le_A4"
        ignored["log_likelihood_i"] = np.nan
        ignored["gjb18a_x"] = np.nan
        ignored["gjb18a_sd_i"] = np.nan
        ignored["gjb18a_mu"] = np.nan
        ignored["gjb18a_w"] = np.nan
        ignored["gjb18a_logpdf"] = np.nan
        ignored["gjb18a_logsf"] = np.nan
        ignored["gjb18a_log_likelihood_i"] = np.nan
        likelihood = pd.concat([likelihood, ignored], ignore_index=True)
    return {
        "coefficient_a": a,
        "coefficient_b": b,
        "coefficient_c": c,
        "standard_error_a": float(standard_errors[0]),
        "standard_error_b": float(standard_errors[1]),
        "log_likelihood": -nll,
        "negative_log_likelihood": nll,
        "success": bool(result.success and np.isfinite(nll)),
        "optimizer_message": str(result.message),
        "likelihood": likelihood,
        "mle_active_rows": int(len(active)),
        "mle_input_rows": int(np.sum(domain)),
        "mle_failure_rows": int(np.sum(failures_fit)),
        "mle_runout_rows": int(np.sum(~failures_fit)),
        "mle_domain_ignored_rows": int(np.sum(~domain)),
        "mle_domain_ignored_row_ids": ignored_ids,
        "mle_domain_invalid_rows": int(np.sum(~domain)),
    }


def _gjb18a_final_mle_audit_table(likelihood: pd.DataFrame) -> pd.DataFrame:
    """Normalize likelihood rows to the Step09 audit columns."""
    required = [
        "gjb_row_id",
        "gjb_is_failure",
        "gjb_original_status",
        "gjb_response",
        "gjb_life",
        "gjb_y_log10_life",
        "domain_valid",
        "included_in_final_mle",
        "A4_final_mle",
        "x",
        "sd_i",
        "mu",
        "w",
        "logpdf",
        "logsf",
        "likelihood_type",
        "log_likelihood_i",
    ]
    table = likelihood.copy()
    for column in required:
        if column not in table.columns:
            table[column] = np.nan
    return table[required + [column for column in table.columns if column not in required]]


def _gjb18a_final_mle_skipped_table(active: pd.DataFrame, coefficient_c: float) -> pd.DataFrame:
    """Return Step09 rows when MLE is skipped after failed significance checks."""
    table = active[
        [
            "gjb_row_id",
            "gjb_is_failure",
            "gjb_original_status",
            "gjb_response",
            "gjb_life",
            "gjb_y_log10_life",
        ]
    ].copy()
    table["domain_valid"] = table["gjb_response"].astype(float) > coefficient_c
    table["included_in_final_mle"] = False
    table["A4_final_mle"] = coefficient_c
    table["x"] = np.nan
    table["sd_i"] = np.nan
    table["mu"] = np.nan
    table["w"] = np.nan
    table["logpdf"] = np.nan
    table["logsf"] = np.nan
    table["likelihood_type"] = "skipped"
    table["log_likelihood_i"] = np.nan
    return table


def _gjb18a_final_mle_negative_log_likelihood(
    parameters: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    failures: np.ndarray,
    sd_i: np.ndarray,
) -> float:
    a, log_minus_b = [float(value) for value in parameters]
    b = -float(np.exp(log_minus_b))
    mu = a + b * x
    w = (y - mu) / sd_i
    log_likelihood = float(np.sum(stats.norm.logpdf(w[failures]) - np.log(sd_i[failures])))
    if np.any(~failures):
        log_likelihood += float(np.sum(stats.norm.logsf(w[~failures])))
    if not np.isfinite(log_likelihood):
        return 1e300
    return -log_likelihood


def _gjb18a_curve(
    data: pd.DataFrame,
    runout_data: pd.DataFrame,
    coefficient_a: float,
    coefficient_b: float,
    coefficient_c: float,
    fit_points: int,
) -> pd.DataFrame:
    x_values = _curve_domain_values(
        data,
        runout_data,
        data["gjb_x"].to_numpy(dtype=float),
        coefficient_a,
        coefficient_b,
    )
    x_grid = np.linspace(float(np.nanmin(x_values)), float(np.nanmax(x_values)), fit_points)
    response_grid = coefficient_c + np.power(10.0, x_grid)
    y_grid = coefficient_a + coefficient_b * x_grid
    return pd.DataFrame(
        {
            "gjb_x": x_grid,
            "response": response_grid,
            "log10_life_fit": y_grid,
            "log10_life_lower_band": y_grid,
            "log10_life_upper_band": y_grid,
            "life_fit": np.power(10.0, y_grid),
            "life_lower_band": np.power(10.0, y_grid),
            "life_upper_band": np.power(10.0, y_grid),
        }
    ).sort_values("life_fit").reset_index(drop=True)


def _gjb18a_final_residual_statistics(
    fit_sample: pd.DataFrame,
    coefficient_a: float,
    coefficient_b: float,
    coefficient_c: float,
    variance_model: dict[str, float],
) -> pd.DataFrame:
    """Compute document-style final residual statistics.

    Unweighted: ``RMSE = SD = sqrt(sum(R_i^2)/(n-k))``.
    Weighted: ``WR_i = R_i/h_i``, ``RMSE = sqrt(sum(WR_i^2)/(n-k))``,
    ``SD_i = RMSE * h_i`` and ``SR_i = R_i/SD_i``.
    """
    response = fit_sample["gjb_response"].to_numpy(dtype=float)
    y = fit_sample["gjb_y_log10_life"].to_numpy(dtype=float)
    x = _gjb18a_x(response, coefficient_c)
    y_pred = coefficient_a + coefficient_b * x
    residual = y - y_pred
    weighted = bool(variance_model.get("use_weighted", False))
    h_values = _gjb18a_h_values(response, variance_model) if weighted else np.ones_like(response)
    k = 3
    n = len(fit_sample)
    df = max(1, n - k)
    weighted_residual = residual / h_values if weighted else residual
    rmse = float(np.sqrt(np.sum(weighted_residual**2) / df))
    sd_i = rmse * h_values if weighted else np.full_like(residual, rmse)
    standardized = residual / np.maximum(sd_i, 1e-300)
    table = fit_sample[["gjb_row_id", "gjb_response", "gjb_life", "gjb_y_log10_life"]].copy()
    table["y"] = y
    table["y_pred_final"] = y_pred
    table["residual_final"] = residual
    table["h"] = h_values
    table["weighted_residual"] = weighted_residual
    table["RMSE"] = rmse
    table["SD_i"] = sd_i
    table["standardized_residual"] = standardized
    table["k"] = k
    table["n"] = n
    table["df"] = df
    table["weighted"] = weighted
    return table


def _gjb18a_model_assessment(final_residuals: pd.DataFrame) -> pd.DataFrame:
    """Compute the document Durbin-Watson style model assessment statistic."""
    ordered = final_residuals.sort_values("gjb_response").reset_index(drop=True).copy()
    sr = ordered["standardized_residual"].to_numpy(dtype=float)
    denominator = float(np.sum(sr**2))
    D = float(np.sum(np.diff(sr) ** 2) / denominator) if denominator > 0.0 else np.nan
    n = len(ordered)
    Dcrit = float(2.0 - 4.73 / np.power(n, 0.555)) if n > 0 else np.nan
    possible_misfit = bool(np.isfinite(D) and np.isfinite(Dcrit) and D < Dcrit)
    previous = np.concatenate(([np.nan], sr[:-1])) if len(sr) else np.asarray([], dtype=float)
    ordered["sort_order"] = np.arange(1, len(ordered) + 1, dtype=int)
    ordered["previous_standardized_residual"] = previous
    ordered["diff_from_previous"] = ordered["standardized_residual"] - ordered["previous_standardized_residual"]
    ordered["diff_squared"] = ordered["diff_from_previous"] ** 2
    ordered["D"] = D
    ordered["Dcrit"] = Dcrit
    ordered["D_lt_Dcrit"] = possible_misfit
    ordered["possible_misfit"] = possible_misfit
    ordered["multi_strain_ratio_residual_mean_check"] = "not applicable: response is used directly as equivalent strain"
    return ordered


def _gjb18a_document_style_r2(
    final_residuals: pd.DataFrame,
    old_r2: float,
) -> pd.DataFrame:
    """Compute document-style R2 without replacing the legacy SSE/TSS value."""
    y = final_residuals["y"].to_numpy(dtype=float)
    h = final_residuals["h"].to_numpy(dtype=float)
    weighted = bool(final_residuals["weighted"].iloc[0]) if len(final_residuals) else False
    rmse = float(final_residuals["RMSE"].iloc[0]) if len(final_residuals) else np.nan
    y_bar = float(np.mean(y)) if len(y) else np.nan
    if len(y) <= 1:
        rte = np.nan
    elif weighted:
        rte = float(np.sqrt(np.sum(((y - y_bar) / h) ** 2) / (len(y) - 1)))
    else:
        rte = float(np.sqrt(np.sum((y - y_bar) ** 2) / (len(y) - 1)))
    r2_document = float(1.0 - (rmse**2) / (rte**2)) if rte and np.isfinite(rte) else np.nan
    return pd.DataFrame(
        [
            {
                "old_r2_log_life": old_r2,
                "r2_document_style": r2_document,
                "weighted": weighted,
                "RMSE": rmse,
                "RTE": rte,
                "formula": "R2 = 1 - RMSE^2 / RTE^2",
                "RTE_formula": (
                    "sqrt(sum(((y_i - y_bar)/h_i)^2)/(n-1))"
                    if weighted
                    else "sqrt(sum((y_i - y_bar)^2)/(n-1))"
                ),
            }
        ]
    )


def _gjb18a_extra_tables(
    iteration_records: list[tuple[str, pd.DataFrame]],
    outlier_records: list[pd.DataFrame],
    residuals: pd.DataFrame,
    removed: pd.DataFrame,
    mle_state: dict[str, object] | None,
    final_state: dict[str, object],
    decision_log: pd.DataFrame,
    final_residual_statistics: pd.DataFrame,
    model_assessment: pd.DataFrame,
    r2_document_style: pd.DataFrame,
    final_mle_audit_table: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, list[pd.DataFrame]] = {}
    for name, frame in iteration_records:
        tables.setdefault(name, []).append(frame)
    tables.setdefault("Residuals", []).append(residuals)
    if outlier_records:
        tables.setdefault("OutlierIterations", []).append(pd.concat(outlier_records, ignore_index=True))
    if not removed.empty:
        tables.setdefault("RemovedOutliers", []).append(removed)
    if not decision_log.empty:
        tables.setdefault("DecisionLog", []).append(decision_log)
    tables.setdefault("FinalResidualStatistics", []).append(final_residual_statistics)
    tables.setdefault("ModelAssessment", []).append(model_assessment)
    tables.setdefault("R2DocumentStyle", []).append(r2_document_style)
    if mle_state is not None:
        tables.setdefault("FinalMLE", []).append(
            pd.DataFrame(
                [
                    {
                        "coefficient_a": mle_state["coefficient_a"],
                        "coefficient_b": mle_state["coefficient_b"],
                        "coefficient_c": mle_state["coefficient_c"],
                        "standard_error_a": mle_state["standard_error_a"],
                        "standard_error_b": mle_state["standard_error_b"],
                        "log_likelihood": mle_state["log_likelihood"],
                        "negative_log_likelihood": mle_state["negative_log_likelihood"],
                        "success": mle_state["success"],
                        "optimizer_message": mle_state["optimizer_message"],
                    }
                ]
            )
        )
        tables.setdefault("Likelihood", []).append(mle_state["likelihood"])
    else:
        tables.setdefault("FinalMLE", []).append(
            pd.DataFrame(
                [
                    {
                        "coefficient_a": final_state["post_significance_fit"]["coefficient_a"],
                        "coefficient_b": final_state["post_significance_fit"]["coefficient_b"],
                        "coefficient_c": final_state["post_significance_fit"]["coefficient_c"],
                        "standard_error_a": final_state["post_significance_fit"]["standard_error_a"],
                        "standard_error_b": final_state["post_significance_fit"]["standard_error_b"],
                        "log_likelihood": np.nan,
                        "negative_log_likelihood": np.nan,
                        "success": False,
                        "optimizer_message": final_state["stop_reason"] or "Final MLE was skipped.",
                    }
                ]
            )
        )
        tables.setdefault("Likelihood", []).append(final_mle_audit_table)
    model_checks = pd.DataFrame(
        [
            {
                "final_iteration": final_state["iteration"],
                "significance_passed": final_state["significance_passed"],
                "stop_reason": final_state["stop_reason"],
                "use_weighted": final_state["variance_model"]["use_weighted"],
                "sigma0": final_state["variance_model"]["sigma0"],
                "sigma1": final_state["variance_model"]["sigma1"],
                "outlier_mode": final_state.get("outlier_mode", ""),
            }
        ]
    )
    tables.setdefault("ModelChecks", []).append(model_checks)
    return {name: pd.concat(frames, ignore_index=True) for name, frames in tables.items()}


def _curve_domain_values(
    data: pd.DataFrame,
    runout_data: pd.DataFrame,
    fitted_x: np.ndarray,
    coefficient_a: float,
    coefficient_b: float,
) -> np.ndarray:
    """Return curve X-domain values spanning fit responses and observed lives."""
    values: list[np.ndarray] = [np.asarray(fitted_x, dtype=float)]
    if not runout_data.empty and "gjb_x" in runout_data:
        values.append(runout_data["gjb_x"].to_numpy(dtype=float))

    y_frames = [data["gjb_y_log10_life"]]
    if not runout_data.empty and "gjb_y_log10_life" in runout_data:
        y_frames.append(runout_data["gjb_y_log10_life"])
    if abs(coefficient_b) > 1e-15:
        y_values = pd.concat(y_frames, ignore_index=True).to_numpy(dtype=float)
        values.append((y_values - coefficient_a) / coefficient_b)

    combined = np.concatenate(values)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return np.asarray(fitted_x, dtype=float)
    return combined

def _numerical_hessian(func, parameters: np.ndarray) -> np.ndarray:
    params = np.asarray(parameters, dtype=float)
    n = len(params)
    hessian = np.zeros((n, n), dtype=float)
    steps = np.maximum(np.abs(params) * 1e-4, 1e-5)
    f0 = float(func(params))
    for i in range(n):
        step_i = steps[i]
        plus_i = params.copy()
        minus_i = params.copy()
        plus_i[i] += step_i
        minus_i[i] -= step_i
        f_plus = float(func(plus_i))
        f_minus = float(func(minus_i))
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (step_i**2)
        for j in range(i + 1, n):
            step_j = steps[j]
            pp = params.copy()
            pm = params.copy()
            mp = params.copy()
            mm = params.copy()
            pp[i] += step_i
            pp[j] += step_j
            pm[i] += step_i
            pm[j] -= step_j
            mp[i] -= step_i
            mp[j] += step_j
            mm[i] -= step_i
            mm[j] -= step_j
            hessian_ij = (
                float(func(pp))
                - float(func(pm))
                - float(func(mp))
                + float(func(mm))
            ) / (4.0 * step_i * step_j)
            hessian[i, j] = hessian_ij
            hessian[j, i] = hessian_ij
    return hessian

def _status_is_failure(value: object) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    if not text:
        return True
    if any(marker in text for marker in NON_FAILURE_STATUS_MARKERS):
        return False
    if any(marker in text for marker in FAILURE_STATUS_MARKERS):
        return True
    return True


def _level_values(
    data: pd.DataFrame,
    level_column: str | None,
    replicate_decimals: int,
) -> pd.Series:
    if level_column:
        return data[level_column].astype(str)
    return data["gjb_x"].round(replicate_decimals).astype(str)


def _level_statistics(
    data: pd.DataFrame,
    coefficient_a: float,
    coefficient_b: float,
) -> pd.DataFrame:
    grouped = data.groupby("gjb_level", sort=True, dropna=False)
    level_stats = grouped.agg(
        gjb_level_x=("gjb_x", "mean"),
        gjb_level_y_mean=("gjb_y_log10_life", "mean"),
        gjb_level_count=("gjb_y_log10_life", "size"),
        gjb_level_y_std=("gjb_y_log10_life", "std"),
    ).reset_index()
    level_stats["gjb_level_yhat"] = (
        coefficient_a + coefficient_b * level_stats["gjb_level_x"]
    )
    level_stats["gjb_level_residual"] = (
        level_stats["gjb_level_y_mean"] - level_stats["gjb_level_yhat"]
    )
    return level_stats

def _signed(value: float) -> str:
    if value < 0:
        return f"- {abs(value):.6g}"
    return f"+ {value:.6g}"


def _shifted_response_expression(response_column: str, coefficient_c: float | None) -> str:
    if coefficient_c is None:
        return response_column
    if coefficient_c < 0:
        return f"{response_column} + {abs(coefficient_c):.6g}"
    return f"{response_column} - {coefficient_c:.6g}"


def _log_life_formula(
    life_column: str,
    response_column: str,
    coefficient_a: float,
    coefficient_b: float,
    coefficient_c: float | None,
) -> str:
    # Step formula - Express the fitted GJB relation in log-life form.
    x_expr = f"log10({_shifted_response_expression(response_column, coefficient_c)})"
    return f"log10({life_column}) = {coefficient_a:.6g} {_signed(coefficient_b)} * {x_expr}"


def _life_formula(
    life_column: str,
    response_column: str,
    coefficient_a: float,
    coefficient_b: float,
    coefficient_c: float | None,
) -> str:
    # Step formula - Express life as a direct function of the shifted strain term.
    x_expr = f"log10({_shifted_response_expression(response_column, coefficient_c)})"
    return f"{life_column} = 10^({coefficient_a:.6g} {_signed(coefficient_b)} * {x_expr})"


def _life_response_formula(
    life_column: str,
    response_column: str,
    coefficient_a: float,
    coefficient_b: float,
    coefficient_c: float | None,
) -> str:
    # Step formula - Convert the log-life relation to the engineering plot equation.
    shifted = _shifted_response_expression(response_column, coefficient_c)
    return f"{life_column} = {coefficient_a:.6g} * ({shifted})^{coefficient_b:.6g}"


def _response_life_formula(
    life_column: str,
    response_column: str,
    coefficient_a: float,
    coefficient_b: float,
    coefficient_c: float | None,
) -> str:
    # Step formula - Invert the fitted relation for a target fatigue life.
    if abs(coefficient_b) < 1e-15:
        return "Response-life form is undefined because A2 is approximately zero."
    response_scale = float(np.power(10.0, -coefficient_a / coefficient_b))
    response_exponent = 1.0 / coefficient_b
    return (
        f"{response_column} = {coefficient_c:.6g} + "
        f"{response_scale:.6g} * {life_column}^{response_exponent:.6g}"
    )
