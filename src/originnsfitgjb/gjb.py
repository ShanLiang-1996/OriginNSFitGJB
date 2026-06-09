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
class GJBFit:
    result: GJBFitResult
    data: pd.DataFrame
    curve: pd.DataFrame
    level_stats: pd.DataFrame
    runout_data: pd.DataFrame | None = None
    extra_tables: dict[str, pd.DataFrame] | None = None


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
) -> GJBFit:
    """Fit the simplified GJB/Z 18A 9.3.2 strain-life workflow."""
    confidence = _normalize_confidence(confidence)
    if fit_points < 2:
        raise ValueError("fit_points must be at least 2.")

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
) -> GJBFit:
    """Fit prepared rows with the simplified GJB/Z 18A 9.3.2 Formula 136 model.

    Formula 136 uses an equivalent strain.  This implementation follows the
    requested simplification: the input response column is used directly as that strain,
    and A3 is not fitted.  The fitted relation is:

        log10(Nf) = A1 + A2 * log10(strain - A4)
    """
    # Step 1 - Keep only rows inside the positive life/response domain required by Formula 136.
    prepared = data.copy().reset_index(drop=True)
    prepared["gjb_row_id"] = np.arange(len(prepared), dtype=int)
    before_domain_rows = len(prepared)
    positive_mask = (prepared["gjb_life"] > 0) & (prepared["gjb_response"] > 0)
    prepared = prepared[positive_mask].copy().reset_index(drop=True)
    dropped_nonpositive = before_domain_rows - len(prepared)
    if dropped_nonpositive:
        warnings.append(
            f"Dropped {dropped_nonpositive} row(s) outside the positive domain required by gjb18a-strain."
        )
    if prepared.empty:
        raise ValueError("No positive life/strain rows are available for gjb18a-strain.")

    # Step 2 - Normalize failure/run-out status and compute the log-life response.
    if status_column:
        prepared["gjb_is_failure"] = prepared[status_column].map(_status_is_failure).astype(bool)
    else:
        prepared["gjb_is_failure"] = True
    prepared["gjb_y_log10_life"] = np.log10(prepared["gjb_life"].to_numpy(dtype=float))
    prepared["gjb_original_status"] = np.where(prepared["gjb_is_failure"], "failure", "runout")

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
        )
        iteration_records.extend(state["tables"])
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

    # Step 11 - Collect step-by-step review tables for CSV and Origin workbooks.
    extra_tables = _gjb18a_extra_tables(
        iteration_records,
        outlier_records,
        residuals,
        removed,
        mle_state,
        final_state,
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
    )


def _gjb18a_iteration(
    active: pd.DataFrame,
    *,
    e_min_failure: float,
    confidence: float,
    iteration: int,
) -> dict[str, object]:
    # Step 1 - Work with failure rows for initialization and parameter significance checks.
    failures = active[active["gjb_is_failure"].astype(bool)].copy().reset_index(drop=True)
    if len(failures) < 4:
        raise ValueError("gjb18a-strain outlier iteration left fewer than four failure points.")

    # Step 2 - Formula 136 initialization: A4 starts at half of the minimum failure strain.
    a4_initial = 0.5 * e_min_failure
    x_initial = _gjb18a_x(failures["gjb_response"].to_numpy(dtype=float), a4_initial)
    y_failure = failures["gjb_y_log10_life"].to_numpy(dtype=float)
    a1_initial, a2_initial, initial_cov = _weighted_linear_fit(x_initial, y_failure)
    if a2_initial >= 0.0:
        a2_initial = -1.0 if a2_initial == 0.0 else -abs(a2_initial)
    initial_ols = pd.DataFrame(
        [
            {
                "iteration": iteration,
                "A1_initial": a1_initial,
                "A2_initial": a2_initial,
                "A4_initial": a4_initial,
                "failure_points": len(failures),
            }
        ]
    )

    # Step 3 - Run the initial nonlinear fit on failure data.
    initial_nls = _gjb18a_nls_fit(
        failures,
        np.array([a1_initial, a2_initial, a4_initial], dtype=float),
        weighted=False,
        variance_model=None,
        e_upper_source=e_min_failure,
    )
    initial_nls_table = _gjb18a_fit_table(initial_nls, iteration, "InitialNLS")

    # Step 4 - Estimate the residual variance model and decide whether weighting is needed.
    variance_model, variance_table = _gjb18a_variance_model(
        failures,
        initial_nls,
        confidence,
        iteration,
    )
    # Step 5 - Refit with eligible run-out rows treated as temporary failures.
    refit_data = _gjb18a_refit_data(active, e_min_failure)
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
    refit_table = _gjb18a_fit_table(refit, iteration, "Refit")
    # Step 6 - Check A2 and A4 significance before any outlier decision is honored.
    significance = _gjb18a_parameter_significance(refit, confidence, iteration)
    significance_passed = bool(significance["passed"].iloc[0])
    stop_reason = ""
    if not significance_passed:
        stop_reason = (
            "gjb18a-strain stopped because A2 or A4 failed the 90% parameter "
            "significance test after refitting."
        )
    post_significance_fit = refit
    fixed_a4_table = pd.DataFrame(
        [
            {
                "iteration": iteration,
                "stage": "FixedA4LinearFit",
                "performed": False,
                "reason": "Skipped because weighted refit was not selected.",
                "coefficient_a": refit["coefficient_a"],
                "coefficient_b": refit["coefficient_b"],
                "coefficient_c": refit["coefficient_c"],
            }
        ]
    )
    # Step 7 - If weighted refit was selected, fix A4 and correct A1/A2 linearly.
    if significance_passed and bool(variance_model["use_weighted"]):
        post_significance_fit, fixed_a4_table = _gjb18a_fixed_a4_linear_fit(
            refit_data,
            refit,
            variance_model,
            iteration,
        )

    # Step 8 - Compute studentized residuals and decide whether one row should be removed.
    residual_table, residual_summary = _gjb18a_residuals_and_outlier_statistics(
        refit_data,
        post_significance_fit,
        variance_model,
        iteration,
    )
    remove_outlier = bool(residual_summary["remove_outlier"])
    removed_row_id = residual_summary.get("removed_row_id", None)
    outlier_table = residual_table.copy()
    outlier_table["outlier_G"] = residual_summary["G"]
    outlier_table["outlier_critical"] = residual_summary["critical"]
    outlier_table["outlier_remove"] = False
    if remove_outlier:
        outlier_table.loc[
            outlier_table["gjb_row_id"].astype(int) == int(removed_row_id),
            "outlier_remove",
        ] = True

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
        "tables": [
            ("InitialOLS", initial_ols),
            ("InitialNLS", initial_nls_table),
            ("VarianceAnalysis", variance_table),
            ("RefitData", refit_data.copy()),
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
        "rmse": rmse,
        "covariance": covariance,
        "standard_error_a": float(standard_errors[0]),
        "standard_error_b": float(standard_errors[1]),
        "standard_error_c": float(standard_errors[2]),
        "success": bool(result.success),
        "optimizer_message": str(result.message),
        "weighted": weighted,
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
    variance_model = {
        "sigma0": sigma0,
        "sigma1": sigma1,
        "sigma1_lower_90": sigma1_lower,
        "sigma1_upper_90": sigma1_upper,
        "standard_error_sigma1": standard_error_sigma1,
        "use_weighted": use_weighted,
        "force_zero_intercept": force_zero_intercept,
    }
    table = failures[
        ["gjb_row_id", "gjb_response", "gjb_life", "gjb_y_log10_life"]
    ].copy()
    table["iteration"] = iteration
    table["inverse_response"] = inverse_response
    table["initial_residual"] = residual
    table["scaled_abs_residual"] = scaled_abs_residual
    table["variance_fit"] = fitted
    table["variance_residual"] = residual_scale
    table["sigma0"] = sigma0
    table["sigma1"] = sigma1
    table["sigma1_lower_90"] = sigma1_lower
    table["sigma1_upper_90"] = sigma1_upper
    table["use_weighted"] = use_weighted
    table["force_zero_intercept"] = force_zero_intercept
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


def _gjb18a_refit_data(active: pd.DataFrame, e_min_failure: float) -> pd.DataFrame:
    failure = active["gjb_is_failure"].astype(bool)
    high_runout = (~failure) & (active["gjb_response"].astype(float) > e_min_failure)
    refit = active[failure | high_runout].copy().reset_index(drop=True)
    refit["gjb18a_temporary_failure"] = True
    refit["gjb18a_original_is_failure"] = refit["gjb_is_failure"].astype(bool)
    refit["gjb18a_runout_treated_as_failure"] = ~refit["gjb18a_original_is_failure"]
    if len(refit) < 4:
        raise ValueError("gjb18a-strain refit requires at least four temporary failure points.")
    return refit


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


def _gjb18a_parameter_significance(
    fit: dict[str, object],
    confidence: float,
    iteration: int,
) -> pd.DataFrame:
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
    return pd.DataFrame(
        [
            {
                "iteration": iteration,
                "confidence": confidence,
                "test_confidence": 0.90,
                "degrees_of_freedom": df,
                "t_critical_90": t90,
                "A2": b,
                "A2_standard_error": se_b,
                "A2_lower_90": b_lower,
                "A2_upper_90": b_upper,
                "A2_significant_negative": a2_pass,
                "A4": c,
                "A4_standard_error": se_c,
                "A4_lower_90": c_lower,
                "A4_upper_90": c_upper,
                "A4_significant_nonzero": a4_pass,
                "passed": bool(a2_pass and a4_pass),
            }
        ]
    )


def _gjb18a_fixed_a4_linear_fit(
    refit_data: pd.DataFrame,
    refit: dict[str, object],
    variance_model: dict[str, float],
    iteration: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    # Formula 136 correction after weighted fitting: fix A4 and re-estimate A1/A2 linearly.
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
    table = pd.DataFrame(
        [
            {
                "iteration": iteration,
                "stage": "FixedA4LinearFit",
                "performed": True,
                "reason": "Weighted refit selected; A4 fixed and A1/A2 corrected.",
                "coefficient_a": corrected["coefficient_a"],
                "coefficient_b": corrected["coefficient_b"],
                "coefficient_c": corrected["coefficient_c"],
                "standard_error_a": corrected["standard_error_a"],
                "standard_error_b": corrected["standard_error_b"],
                "standard_error_c": corrected["standard_error_c"],
                "rmse": corrected["rmse"],
            }
        ]
    )
    return corrected, table


def _gjb18a_residuals_and_outlier_statistics(
    refit_data: pd.DataFrame,
    fit: dict[str, object],
    variance_model: dict[str, float],
    iteration: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    # GJB/Z 18A 9.3.2.4.4: externally studentized residual maximum test.
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
    remove_outlier = bool(np.isfinite(G) and G > critical)
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
    if len(table):
        table.loc[table.index[max_index], "gjb18a_is_max_abs_t"] = True
    summary = {
        "G": G,
        "critical": critical,
        "remove_outlier": remove_outlier,
        "removed_row_id": int(table.loc[table.index[max_index], "gjb_row_id"]) if remove_outlier else None,
    }
    return table, summary


def _gjb18a_final_mle(
    active: pd.DataFrame,
    fit: dict[str, object],
    variance_model: dict[str, float],
) -> dict[str, object]:
    # GJB/Z 18A 9.3.2.7.2: A4 and SD_i are fixed; only A1/A2 are corrected by MLE.
    c = float(fit["coefficient_c"])
    response = active["gjb_response"].to_numpy(dtype=float)
    y = active["gjb_y_log10_life"].to_numpy(dtype=float)
    failures = active["gjb_is_failure"].astype(bool).to_numpy()
    domain = response > c
    response_fit = response[domain]
    y_fit = y[domain]
    failures_fit = failures[domain]
    if len(y_fit) < 3:
        raise ValueError("gjb18a-strain final MLE has fewer than three domain-valid rows.")
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
    likelihood["gjb18a_x"] = x
    likelihood["gjb18a_sd_i"] = sd_i
    likelihood["gjb18a_mu"] = mu
    likelihood["gjb18a_w"] = (y_fit - mu) / sd_i
    likelihood["gjb18a_logpdf"] = stats.norm.logpdf(likelihood["gjb18a_w"]) - np.log(sd_i)
    likelihood["gjb18a_logsf"] = stats.norm.logsf(likelihood["gjb18a_w"])
    likelihood["gjb18a_log_likelihood_i"] = np.where(
        failures_fit,
        likelihood["gjb18a_logpdf"],
        likelihood["gjb18a_logsf"],
    )
    if np.any(~domain):
        excluded = active[~domain].copy()
        excluded["gjb18a_x"] = np.nan
        excluded["gjb18a_sd_i"] = np.nan
        excluded["gjb18a_mu"] = np.nan
        excluded["gjb18a_w"] = np.nan
        excluded["gjb18a_logpdf"] = np.nan
        excluded["gjb18a_logsf"] = np.nan
        excluded["gjb18a_log_likelihood_i"] = np.nan
        likelihood = pd.concat([likelihood, excluded], ignore_index=True)
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
    }


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


def _gjb18a_extra_tables(
    iteration_records: list[tuple[str, pd.DataFrame]],
    outlier_records: list[pd.DataFrame],
    residuals: pd.DataFrame,
    removed: pd.DataFrame,
    mle_state: dict[str, object] | None,
    final_state: dict[str, object],
) -> dict[str, pd.DataFrame]:
    tables: dict[str, list[pd.DataFrame]] = {}
    for name, frame in iteration_records:
        tables.setdefault(name, []).append(frame)
    tables.setdefault("Residuals", []).append(residuals)
    if outlier_records:
        tables.setdefault("OutlierIterations", []).append(pd.concat(outlier_records, ignore_index=True))
    if not removed.empty:
        tables.setdefault("RemovedOutliers", []).append(removed)
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
    model_checks = pd.DataFrame(
        [
            {
                "final_iteration": final_state["iteration"],
                "significance_passed": final_state["significance_passed"],
                "stop_reason": final_state["stop_reason"],
                "use_weighted": final_state["variance_model"]["use_weighted"],
                "sigma0": final_state["variance_model"]["sigma0"],
                "sigma1": final_state["variance_model"]["sigma1"],
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
