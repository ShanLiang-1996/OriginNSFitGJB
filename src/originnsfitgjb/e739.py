from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import optimize, stats


E739Transform = Literal["log", "linear"]
E739Model = Literal[
    "standard",
    "shifted-log",
    "threshold_log_mle",
    "log_threshold_censored_mle",
    "gjb932-strain",
]
THRESHOLD_LOG_MLE_MODELS = {"threshold_log_mle", "log_threshold_censored_mle"}
GJB932_STRAIN_MODEL = "gjb932-strain"

FAILURE_STATUS_MARKERS = (
    "failure",
    "failed",
    "fail",
    "fracture",
    "broken",
    "yes",
    "true",
    "1",
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
)


@dataclass(frozen=True)
class E739LinearityTest:
    available: bool
    reason: str
    levels: int
    specimens: int
    lack_of_fit_df: int | None
    pure_error_df: int | None
    lack_of_fit_ss: float | None
    pure_error_ss: float | None
    lack_of_fit_ms: float | None
    pure_error_ms: float | None
    f_statistic: float | None
    f_critical: float | None
    p_value: float | None
    reject_linear_model: bool | None


@dataclass(frozen=True)
class E739FitResult:
    life_column: str
    response_column: str
    model: E739Model
    model_name: str
    confidence: float
    x_transform: E739Transform
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
    linearity_test: E739LinearityTest


@dataclass(frozen=True)
class E739Fit:
    result: E739FitResult
    data: pd.DataFrame
    curve: pd.DataFrame
    level_stats: pd.DataFrame
    runout_data: pd.DataFrame | None = None
    extra_tables: dict[str, pd.DataFrame] | None = None


def fit_e739(
    frame: pd.DataFrame,
    life_column: str,
    response_column: str,
    *,
    model: E739Model = GJB932_STRAIN_MODEL,
    x_transform: E739Transform = "log",
    confidence: float = 0.95,
    fit_points: int = 300,
    status_column: str | None = None,
    level_column: str | None = None,
    replicate_decimals: int = 8,
) -> E739Fit:
    """Fit an ASTM E739-style linearized S-N/epsilon-N relation."""
    confidence = _normalize_confidence(confidence)
    if model not in ("standard", "shifted-log", *THRESHOLD_LOG_MLE_MODELS, GJB932_STRAIN_MODEL):
        raise ValueError(
            "model must be 'standard', 'shifted-log', 'threshold_log_mle', "
            "'log_threshold_censored_mle', or 'gjb932-strain'."
        )
    if model == "log_threshold_censored_mle":
        model = "threshold_log_mle"
    if x_transform not in ("log", "linear"):
        raise ValueError("x_transform must be 'log' or 'linear'.")
    if model in ("shifted-log", "threshold_log_mle", GJB932_STRAIN_MODEL) and x_transform != "log":
        raise ValueError(f"The {model} model requires --e739-x-transform log.")
    if fit_points < 2:
        raise ValueError("fit_points must be at least 2.")
    _require_column(frame, life_column, "life")
    _require_column(frame, response_column, "response")
    if status_column:
        _require_column(frame, status_column, "status")
    if level_column:
        _require_column(frame, level_column, "level")

    warnings: list[str] = []
    source_columns = list(dict.fromkeys([response_column, life_column, status_column, level_column]))
    source_columns = [column for column in source_columns if column]
    data = frame[source_columns].copy()
    data["e739_life"] = pd.to_numeric(data[life_column], errors="coerce")
    data["e739_response"] = pd.to_numeric(data[response_column], errors="coerce")

    initial_rows = len(data)
    data = data.dropna(subset=["e739_life", "e739_response"]).copy()
    dropped_missing = initial_rows - len(data)
    if dropped_missing:
        warnings.append(f"Dropped {dropped_missing} row(s) with missing numeric life/response.")

    if model == GJB932_STRAIN_MODEL:
        return _fit_gjb932_strain(
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

    runout_data = pd.DataFrame(columns=data.columns)
    n_runout: int | None = None
    if status_column:
        status = data[status_column].map(_status_is_failure)
        if model == "threshold_log_mle":
            data["e739_is_failure"] = status.astype(bool)
            n_runout = int((~status).sum())
        else:
            excluded = int((~status).sum())
            runout_data = data[~status].copy()
            data = data[status].copy()
            if excluded:
                warnings.append(
                    "Excluded run-out/suspended row(s) from the OLS fit; retained them "
                    "for run-out exports and plot markers."
                )
    elif model == "threshold_log_mle":
        data["e739_is_failure"] = True
        n_runout = 0

    before_domain_rows = len(data)
    positive_mask = data["e739_life"] > 0
    if x_transform == "log":
        positive_mask &= data["e739_response"] > 0
    data = data[positive_mask].copy()
    dropped_nonpositive = before_domain_rows - len(data)
    if dropped_nonpositive:
        warnings.append(
            f"Dropped {dropped_nonpositive} row(s) outside the positive domain required by the model."
        )

    if model == "threshold_log_mle":
        failure_mask = data["e739_is_failure"].astype(bool).to_numpy()
        n_failure = int(np.sum(failure_mask))
        n_runout = int(len(failure_mask) - n_failure)
        if n_runout:
            warnings.append(
                "Included run-out/suspended row(s) as right-censored observations in MLE."
            )
    else:
        failure_mask = None
        n_failure = len(data)
        n_runout = len(runout_data)

    parameter_count = 4 if model == "threshold_log_mle" else 3 if model == "shifted-log" else 2
    if len(data) <= parameter_count:
        raise ValueError(
            f"At least {parameter_count + 1} valid failure points are required "
            f"for the {model} E739 analysis."
        )
    if model == "threshold_log_mle" and (n_failure is None or n_failure < 3):
        raise ValueError("At least three failure points are required to initialize threshold MLE.")

    data["e739_y_log10_life"] = np.log10(data["e739_life"].to_numpy(dtype=float))

    response = data["e739_response"].to_numpy(dtype=float)
    y = data["e739_y_log10_life"].to_numpy(dtype=float)
    k = len(data)
    y_mean = float(np.mean(y))
    y_delta = y - y_mean
    total_sum_squares = float(np.sum(y_delta**2))
    log_likelihood = None
    negative_log_likelihood = None
    optimizer_success = None
    optimizer_message = ""
    standard_error_sigma = None
    sigma_confidence_interval: tuple[float, float] | None = None

    if model == "threshold_log_mle":
        fit_state = _fit_threshold_log_mle(response, y, failure_mask)
        coefficient_a = fit_state["coefficient_a"]
        coefficient_b = fit_state["coefficient_b"]
        coefficient_c = fit_state["coefficient_c"]
        x = fit_state["x"]
        y_hat = fit_state["y_hat"]
        residual = fit_state["residual"]
        covariance = fit_state["covariance"]
        covariance_mean = fit_state["covariance_mean"]
        sxx = fit_state["sxx"]
        sxy = fit_state["sxy"]
        x_mean = fit_state["x_mean"]
        standard_error_a = fit_state["standard_error_a"]
        standard_error_b = fit_state["standard_error_b"]
        standard_error_c = fit_state["standard_error_c"]
        standard_error_sigma = fit_state["standard_error_sigma"]
        sigma = fit_state["sigma"]
        sigma_squared = sigma**2
        log_likelihood = fit_state["log_likelihood"]
        negative_log_likelihood = fit_state["negative_log_likelihood"]
        optimizer_success = fit_state["success"]
        optimizer_message = fit_state["optimizer_message"]
    elif model == "shifted-log":
        fit_state = _fit_shifted_log_response(response, y)
        coefficient_a = fit_state["coefficient_a"]
        coefficient_b = fit_state["coefficient_b"]
        coefficient_c = fit_state["coefficient_c"]
        x = fit_state["x"]
        y_hat = fit_state["y_hat"]
        residual = fit_state["residual"]
        covariance = fit_state["covariance"]
        covariance_mean = covariance
        sxx = fit_state["sxx"]
        sxy = fit_state["sxy"]
        x_mean = fit_state["x_mean"]
        standard_error_a = fit_state["standard_error_a"]
        standard_error_b = fit_state["standard_error_b"]
        standard_error_c = fit_state["standard_error_c"]
    else:
        coefficient_c = None
        standard_error_c = None
        covariance = None
        covariance_mean = None
        if x_transform == "log":
            x = np.log10(response)
        else:
            x = response
        x_mean = float(np.mean(x))
        x_delta = x - x_mean
        sxx = float(np.sum(x_delta**2))
        if sxx <= 0:
            raise ValueError("The E739 independent variable has no variation.")
        sxy = float(np.sum(x_delta * y_delta))
        coefficient_b = sxy / sxx
        coefficient_a = y_mean - coefficient_b * x_mean
        y_hat = coefficient_a + coefficient_b * x
        residual = y - y_hat
        standard_error_a = None
        standard_error_b = None

    data["e739_x"] = x
    residual = y - y_hat
    residual_sum_squares = float(np.sum(residual**2))
    degrees_of_freedom = k - parameter_count
    if model != "threshold_log_mle":
        sigma_squared = residual_sum_squares / degrees_of_freedom
        sigma = float(np.sqrt(sigma_squared))
    r2 = 1.0 if total_sum_squares == 0 else 1.0 - residual_sum_squares / total_sum_squares
    rmse_log_life = float(np.sqrt(np.mean(residual**2)))
    if model == "standard":
        standard_error_a = float(sigma * np.sqrt(1.0 / k + x_mean**2 / sxx))
        standard_error_b = float(sigma / np.sqrt(sxx))

    t_critical = float(stats.t.ppf((1.0 + confidence) / 2.0, degrees_of_freedom))
    if standard_error_sigma is not None:
        sigma_confidence_interval = (
            max(0.0, float(sigma - t_critical * standard_error_sigma)),
            float(sigma + t_critical * standard_error_sigma),
        )
    band_parameter_count = 3 if model == "threshold_log_mle" else parameter_count
    f_band_critical = float(stats.f.ppf(confidence, band_parameter_count, degrees_of_freedom))
    simultaneous_band_factor = float(np.sqrt(band_parameter_count * f_band_critical))

    data["e739_yhat_log10_life"] = y_hat
    data["e739_residual_log10_life"] = residual
    data["e739_life_fit"] = np.power(10.0, y_hat)
    data["e739_abs_residual_log10_life"] = np.abs(residual)
    if "e739_is_failure" not in data.columns:
        data["e739_is_failure"] = True
    data["e739_level"] = _level_values(data, level_column, replicate_decimals)

    if model == "threshold_log_mle":
        runout_data = data[~data["e739_is_failure"].astype(bool)].copy()
    else:
        runout_data = _prepare_runout_data(
            runout_data,
            level_column,
            replicate_decimals,
            coefficient_a,
            coefficient_b,
            coefficient_c,
            model,
            x_transform,
            warnings,
        )
        n_runout = len(runout_data)
        n_failure = len(data)

    level_data = data[data["e739_is_failure"]].copy() if model == "threshold_log_mle" else data
    level_stats = _level_statistics(level_data, coefficient_a, coefficient_b)
    if model == "threshold_log_mle":
        linearity_test = _unavailable_linearity_test(
            data,
            level_stats,
            "Linearity F test is not defined for censored threshold-log MLE.",
        )
    else:
        linearity_test = _linearity_test(
            data,
            level_stats,
            confidence,
            coefficient_a,
            coefficient_b,
            parameter_count,
        )
    levels = max(1, len(level_stats))
    replication_percent = float(100.0 * (1.0 - levels / k))

    curve_x_values = _curve_domain_values(
        data,
        runout_data,
        x,
        coefficient_a,
        coefficient_b,
    )
    x_grid = np.linspace(float(np.min(curve_x_values)), float(np.max(curve_x_values)), fit_points)
    y_grid = coefficient_a + coefficient_b * x_grid
    if model == "threshold_log_mle":
        band_delta = _threshold_log_mle_band_delta(
            x_grid,
            coefficient_b,
            covariance_mean,
            simultaneous_band_factor,
        )
    elif model == "shifted-log":
        band_delta = _shifted_log_band_delta(
            x_grid,
            coefficient_b,
            covariance,
            simultaneous_band_factor,
        )
    else:
        band_delta = (
            simultaneous_band_factor
            * sigma
            * np.sqrt(1.0 / k + np.power(x_grid - x_mean, 2) / sxx)
        )
    if model in ("shifted-log", "threshold_log_mle"):
        response_grid = coefficient_c + np.power(10.0, x_grid)
    elif x_transform == "log":
        response_grid = np.power(10.0, x_grid)
    else:
        response_grid = x_grid
    curve = pd.DataFrame(
        {
            "e739_x": x_grid,
            "response": response_grid,
            "log10_life_fit": y_grid,
            "log10_life_lower_band": y_grid - band_delta,
            "log10_life_upper_band": y_grid + band_delta,
            "life_fit": np.power(10.0, y_grid),
            "life_lower_band": np.power(10.0, y_grid - band_delta),
            "life_upper_band": np.power(10.0, y_grid + band_delta),
        }
    )
    curve = curve.sort_values("life_fit").reset_index(drop=True)

    log_life_formula = _log_life_formula(
        life_column,
        response_column,
        coefficient_a,
        coefficient_b,
        x_transform,
        model,
        coefficient_c,
    )
    life_response_coefficient_a = float(np.power(10.0, coefficient_a))
    life_response_coefficient_b = float(coefficient_b)
    life_formula = _life_formula(
        life_column,
        response_column,
        coefficient_a,
        coefficient_b,
        x_transform,
        model,
        coefficient_c,
    )
    life_response_formula = _life_response_formula(
        life_column,
        response_column,
        life_response_coefficient_a,
        life_response_coefficient_b,
        x_transform,
        model,
        coefficient_c,
    )
    response_life_formula = _response_life_formula(
        life_column,
        response_column,
        coefficient_a,
        coefficient_b,
        x_transform,
        model,
        coefficient_c,
    )

    result = E739FitResult(
        life_column=life_column,
        response_column=response_column,
        model=model,
        model_name=model,
        confidence=confidence,
        x_transform=x_transform,
        parameter_count=parameter_count,
        points=k,
        degrees_of_freedom=degrees_of_freedom,
        coefficient_a=float(coefficient_a),
        coefficient_b=float(coefficient_b),
        coefficient_c=None if coefficient_c is None else float(coefficient_c),
        threshold=None if coefficient_c is None else float(coefficient_c),
        x_mean=x_mean,
        y_mean=y_mean,
        sxx=sxx,
        sxy=sxy,
        residual_sum_squares=residual_sum_squares,
        sigma_squared=float(sigma_squared),
        sigma=sigma,
        r2=float(r2),
        rmse_log_life=rmse_log_life,
        standard_error_a=float(standard_error_a),
        standard_error_b=float(standard_error_b),
        standard_error_c=None if standard_error_c is None else float(standard_error_c),
        t_critical=t_critical,
        f_band_critical=f_band_critical,
        simultaneous_band_factor=simultaneous_band_factor,
        coefficient_a_lower=float(coefficient_a - t_critical * standard_error_a),
        coefficient_a_upper=float(coefficient_a + t_critical * standard_error_a),
        coefficient_b_lower=float(coefficient_b - t_critical * standard_error_b),
        coefficient_b_upper=float(coefficient_b + t_critical * standard_error_b),
        coefficient_c_lower=(
            None
            if coefficient_c is None or standard_error_c is None
            else float(coefficient_c - t_critical * standard_error_c)
        ),
        coefficient_c_upper=(
            None
            if coefficient_c is None or standard_error_c is None
            else float(coefficient_c + t_critical * standard_error_c)
        ),
        sigma_lower=None if sigma_confidence_interval is None else sigma_confidence_interval[0],
        sigma_upper=None if sigma_confidence_interval is None else sigma_confidence_interval[1],
        x_min=float(np.min(x)),
        x_max=float(np.max(x)),
        life_min=float(data["e739_life"].min()),
        life_max=float(data["e739_life"].max()),
        response_min=float(data["e739_response"].min()),
        response_max=float(data["e739_response"].max()),
        replication_percent=replication_percent,
        life_response_coefficient_a=life_response_coefficient_a,
        life_response_coefficient_b=life_response_coefficient_b,
        log_likelihood=None if log_likelihood is None else float(log_likelihood),
        negative_log_likelihood=(
            None if negative_log_likelihood is None else float(negative_log_likelihood)
        ),
        n_failure=n_failure,
        n_runout=n_runout,
        success=optimizer_success,
        optimizer_message=optimizer_message,
        log_life_formula=log_life_formula,
        life_formula=life_formula,
        life_response_formula=life_response_formula,
        response_life_formula=response_life_formula,
        warnings=tuple(warnings),
        linearity_test=linearity_test,
    )
    return E739Fit(
        result=result,
        data=data.reset_index(drop=True),
        curve=curve,
        level_stats=level_stats,
        runout_data=runout_data.reset_index(drop=True),
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
        raise ValueError(f"E739 {role} column not found: {column}")


def _prepare_runout_data(
    runout_data: pd.DataFrame,
    level_column: str | None,
    replicate_decimals: int,
    coefficient_a: float,
    coefficient_b: float,
    coefficient_c: float | None,
    model: E739Model,
    x_transform: E739Transform,
    warnings: list[str],
) -> pd.DataFrame:
    """Transform run-out rows for export and plotting without using them in OLS fitting."""
    if runout_data.empty:
        return runout_data.copy()

    prepared = runout_data.copy()
    before_domain_rows = len(prepared)
    plot_mask = prepared["e739_life"] > 0
    if x_transform == "log" or model in ("shifted-log", "threshold_log_mle"):
        plot_mask &= prepared["e739_response"] > 0

    prepared = prepared[plot_mask].copy()
    dropped = before_domain_rows - len(prepared)
    if dropped:
        warnings.append(
            f"Dropped {dropped} run-out row(s) outside the positive plotting domain."
        )
    if prepared.empty:
        return prepared

    response = prepared["e739_response"].to_numpy(dtype=float)
    y = np.log10(prepared["e739_life"].to_numpy(dtype=float))
    x = np.full(len(prepared), np.nan, dtype=float)
    if model in ("shifted-log", "threshold_log_mle"):
        if coefficient_c is not None:
            transform_mask = response > float(coefficient_c)
            x[transform_mask] = np.log10(response[transform_mask] - float(coefficient_c))
        else:
            transform_mask = np.zeros(len(prepared), dtype=bool)
    elif x_transform == "log":
        transform_mask = response > 0
        x[transform_mask] = np.log10(response[transform_mask])
    else:
        transform_mask = np.ones(len(prepared), dtype=bool)
        x = response

    invalid_transform = int(len(prepared) - np.sum(transform_mask))
    if invalid_transform:
        warnings.append(
            f"Retained {invalid_transform} run-out row(s) for engineering plots, but left "
            "linearized coordinates blank because they are outside the model transform domain."
        )
    y_hat = coefficient_a + coefficient_b * x
    residual = y - y_hat

    prepared["e739_is_failure"] = False
    prepared["e739_y_log10_life"] = y
    prepared["e739_x"] = x
    prepared["e739_yhat_log10_life"] = y_hat
    prepared["e739_residual_log10_life"] = residual
    prepared["e739_life_fit"] = np.power(10.0, y_hat)
    prepared["e739_abs_residual_log10_life"] = np.abs(residual)
    prepared["e739_level"] = _level_values(prepared, level_column, replicate_decimals)
    return prepared


def _fit_gjb932_strain(
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
) -> E739Fit:
    """Fit the simplified GJB/Z 18A 9.3.2 Formula 136 strain-life model.

    The standard Formula 136 uses an equivalent strain.  This implementation follows the
    requested simplification: the input response column is used directly as that strain,
    and A3 is not fitted.  The fitted relation is:

        log10(Nf) = A1 + A2 * log10(strain - A4)
    """
    prepared = data.copy().reset_index(drop=True)
    prepared["e739_row_id"] = np.arange(len(prepared), dtype=int)
    before_domain_rows = len(prepared)
    positive_mask = (prepared["e739_life"] > 0) & (prepared["e739_response"] > 0)
    prepared = prepared[positive_mask].copy().reset_index(drop=True)
    dropped_nonpositive = before_domain_rows - len(prepared)
    if dropped_nonpositive:
        warnings.append(
            f"Dropped {dropped_nonpositive} row(s) outside the positive domain required by gjb932-strain."
        )
    if prepared.empty:
        raise ValueError("No positive life/strain rows are available for gjb932-strain.")

    if status_column:
        prepared["e739_is_failure"] = prepared[status_column].map(_status_is_failure).astype(bool)
    else:
        prepared["e739_is_failure"] = True
    prepared["e739_y_log10_life"] = np.log10(prepared["e739_life"].to_numpy(dtype=float))
    prepared["e739_original_status"] = np.where(prepared["e739_is_failure"], "failure", "runout")

    original_failure = prepared[prepared["e739_is_failure"]].copy()
    if len(original_failure) < 4:
        raise ValueError("gjb932-strain requires at least four failure points.")
    e_min_failure = float(original_failure["e739_response"].min())
    if e_min_failure <= 0.0:
        raise ValueError("gjb932-strain requires positive failure strain values.")

    removed_ids: set[int] = set()
    iteration_records: list[pd.DataFrame] = []
    outlier_records: list[pd.DataFrame] = []
    final_state: dict[str, object] | None = None
    max_iterations = min(20, max(1, len(prepared) - 4))

    for iteration in range(1, max_iterations + 1):
        active = prepared[~prepared["e739_row_id"].isin(removed_ids)].copy().reset_index(drop=True)
        state = _gjb932_iteration(
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
            f"Removed outlier row id {removed_id} during gjb932-strain 9.3.2.4.4 iteration."
        )
    else:
        warnings.append("Stopped gjb932-strain outlier iteration after the maximum iteration limit.")

    if final_state is None:
        raise ValueError("gjb932-strain did not complete an analysis iteration.")

    active = prepared[~prepared["e739_row_id"].isin(removed_ids)].copy().reset_index(drop=True)
    refit = final_state["post_significance_fit"]
    variance = final_state["variance_model"]
    residuals = final_state["residual_table"].copy()
    significance_passed = bool(final_state["significance_passed"])
    mle_state: dict[str, object] | None = None
    if significance_passed:
        mle_state = _gjb932_final_mle(
            active,
            refit,
            variance,
        )
        if not bool(mle_state["success"]):
            warnings.append(f"gjb932-strain final MLE warning: {mle_state['optimizer_message']}")

    coefficient_a = float(mle_state["coefficient_a"] if mle_state else refit["coefficient_a"])
    coefficient_b = float(mle_state["coefficient_b"] if mle_state else refit["coefficient_b"])
    coefficient_c = float(refit["coefficient_c"])
    sigma = float(refit["rmse"])
    x_all = _gjb932_x(active["e739_response"].to_numpy(dtype=float), coefficient_c)
    y_all = active["e739_y_log10_life"].to_numpy(dtype=float)
    y_hat_all = coefficient_a + coefficient_b * x_all
    residual_all = y_all - y_hat_all

    active["e739_x"] = x_all
    active["e739_yhat_log10_life"] = y_hat_all
    active["e739_residual_log10_life"] = residual_all
    active["e739_life_fit"] = np.power(10.0, y_hat_all)
    active["e739_abs_residual_log10_life"] = np.abs(residual_all)
    active["e739_removed_outlier"] = False
    active["e739_level"] = _level_values(active, level_column, replicate_decimals)

    removed = prepared[prepared["e739_row_id"].isin(removed_ids)].copy()
    if not removed.empty:
        removed["e739_removed_outlier"] = True
        removed["e739_x"] = _gjb932_x(removed["e739_response"].to_numpy(dtype=float), coefficient_c)
        removed["e739_yhat_log10_life"] = coefficient_a + coefficient_b * removed["e739_x"]
        removed["e739_residual_log10_life"] = (
            removed["e739_y_log10_life"] - removed["e739_yhat_log10_life"]
        )
        removed["e739_life_fit"] = np.power(10.0, removed["e739_yhat_log10_life"])
        removed["e739_abs_residual_log10_life"] = np.abs(removed["e739_residual_log10_life"])
        removed["e739_level"] = _level_values(removed, level_column, replicate_decimals)

    data_out = active.copy()
    runout_data = data_out[~data_out["e739_is_failure"].astype(bool)].copy().reset_index(drop=True)
    level_data = data_out[data_out["e739_is_failure"].astype(bool)].copy()
    level_stats = _level_statistics(level_data, coefficient_a, coefficient_b)
    linearity_test = _unavailable_linearity_test(
        level_data,
        level_stats,
        "Linearity F test is not defined for gjb932-strain 9.3.2 workflow.",
    )

    curve = _gjb932_curve(
        data_out,
        runout_data,
        coefficient_a,
        coefficient_b,
        coefficient_c,
        fit_points,
    )

    fit_sample = final_state["refit_data"].copy()
    x_fit = _gjb932_x(fit_sample["e739_response"].to_numpy(dtype=float), coefficient_c)
    y_fit = fit_sample["e739_y_log10_life"].to_numpy(dtype=float)
    y_fit_hat = coefficient_a + coefficient_b * x_fit
    fit_residual = y_fit - y_fit_hat
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

    extra_tables = _gjb932_extra_tables(
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

    result = E739FitResult(
        life_column=life_column,
        response_column=response_column,
        model=GJB932_STRAIN_MODEL,
        model_name=GJB932_STRAIN_MODEL,
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
        x_min=float(np.nanmin(data_out["e739_x"])),
        x_max=float(np.nanmax(data_out["e739_x"])),
        life_min=float(data_out["e739_life"].min()),
        life_max=float(data_out["e739_life"].max()),
        response_min=float(data_out["e739_response"].min()),
        response_max=float(data_out["e739_response"].max()),
        replication_percent=0.0,
        life_response_coefficient_a=life_response_coefficient_a,
        life_response_coefficient_b=life_response_coefficient_b,
        log_likelihood=None if mle_state is None else float(mle_state["log_likelihood"]),
        negative_log_likelihood=(
            None if mle_state is None else float(mle_state["negative_log_likelihood"])
        ),
        n_failure=int(data_out["e739_is_failure"].astype(bool).sum()),
        n_runout=int((~data_out["e739_is_failure"].astype(bool)).sum()),
        success=optimizer_success,
        optimizer_message=optimizer_message,
        log_life_formula=_log_life_formula(
            life_column,
            response_column,
            coefficient_a,
            coefficient_b,
            "log",
            GJB932_STRAIN_MODEL,
            coefficient_c,
        ),
        life_formula=_life_formula(
            life_column,
            response_column,
            coefficient_a,
            coefficient_b,
            "log",
            GJB932_STRAIN_MODEL,
            coefficient_c,
        ),
        life_response_formula=_life_response_formula(
            life_column,
            response_column,
            life_response_coefficient_a,
            life_response_coefficient_b,
            "log",
            GJB932_STRAIN_MODEL,
            coefficient_c,
        ),
        response_life_formula=_response_life_formula(
            life_column,
            response_column,
            coefficient_a,
            coefficient_b,
            "log",
            GJB932_STRAIN_MODEL,
            coefficient_c,
        ),
        warnings=tuple(warnings),
        linearity_test=linearity_test,
    )
    return E739Fit(
        result=result,
        data=data_out.reset_index(drop=True),
        curve=curve,
        level_stats=level_stats,
        runout_data=runout_data,
        extra_tables=extra_tables,
    )


def _gjb932_iteration(
    active: pd.DataFrame,
    *,
    e_min_failure: float,
    confidence: float,
    iteration: int,
) -> dict[str, object]:
    failures = active[active["e739_is_failure"].astype(bool)].copy().reset_index(drop=True)
    if len(failures) < 4:
        raise ValueError("gjb932-strain outlier iteration left fewer than four failure points.")

    # Formula 136 initialisation: A4 starts at half of the original failure minimum strain.
    a4_initial = 0.5 * e_min_failure
    x_initial = _gjb932_x(failures["e739_response"].to_numpy(dtype=float), a4_initial)
    y_failure = failures["e739_y_log10_life"].to_numpy(dtype=float)
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

    initial_nls = _gjb932_nls_fit(
        failures,
        np.array([a1_initial, a2_initial, a4_initial], dtype=float),
        weighted=False,
        variance_model=None,
        e_upper_source=e_min_failure,
    )
    initial_nls_table = _gjb932_fit_table(initial_nls, iteration, "InitialNLS")

    variance_model, variance_table = _gjb932_variance_model(
        failures,
        initial_nls,
        confidence,
        iteration,
    )
    refit_data = _gjb932_refit_data(active, e_min_failure)
    refit = _gjb932_nls_fit(
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
        e_upper_source=float(refit_data["e739_response"].min()),
    )
    refit_table = _gjb932_fit_table(refit, iteration, "Refit")
    significance = _gjb932_parameter_significance(refit, confidence, iteration)
    significance_passed = bool(significance["passed"].iloc[0])
    stop_reason = ""
    if not significance_passed:
        stop_reason = (
            "gjb932-strain stopped because A2 or A4 failed the 90% parameter "
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
    if significance_passed and bool(variance_model["use_weighted"]):
        post_significance_fit, fixed_a4_table = _gjb932_fixed_a4_linear_fit(
            refit_data,
            refit,
            variance_model,
            iteration,
        )

    residual_table, residual_summary = _gjb932_residuals_and_outlier_statistics(
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
            outlier_table["e739_row_id"].astype(int) == int(removed_row_id),
            "outlier_remove",
        ] = True

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


def _gjb932_x(response: np.ndarray, coefficient_c: float) -> np.ndarray:
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


def _gjb932_nls_fit(
    frame: pd.DataFrame,
    initial: np.ndarray,
    *,
    weighted: bool,
    variance_model: dict[str, float] | None,
    e_upper_source: float,
) -> dict[str, object]:
    response = frame["e739_response"].to_numpy(dtype=float)
    y = frame["e739_y_log10_life"].to_numpy(dtype=float)
    epsilon = max(abs(float(e_upper_source)) * 1e-10, 1e-15)
    lower = np.array([-np.inf, -np.inf, 0.0], dtype=float)
    upper = np.array([np.inf, -1e-12, float(e_upper_source) - epsilon], dtype=float)
    start = np.asarray(initial, dtype=float).copy()
    start[1] = min(start[1], -1e-8)
    start[2] = min(max(start[2], lower[2] + epsilon), upper[2] - epsilon)

    result = optimize.least_squares(
        _gjb932_nls_residual,
        x0=start,
        args=(response, y, weighted, variance_model),
        bounds=(lower, upper),
        x_scale=np.array([1.0, 1.0, max(abs(float(e_upper_source)), 1.0)], dtype=float),
        max_nfev=4000,
    )
    if not np.all(np.isfinite(result.fun)):
        raise ValueError("gjb932-strain nonlinear fit produced non-finite residuals.")

    a, b, c = [float(value) for value in result.x]
    x = _gjb932_x(response, c)
    y_hat = a + b * x
    residual = y - y_hat
    h_values = _gjb932_h_values(response, variance_model) if weighted else np.ones_like(response)
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


def _gjb932_nls_residual(
    parameters: np.ndarray,
    response: np.ndarray,
    y: np.ndarray,
    weighted: bool,
    variance_model: dict[str, float] | None,
) -> np.ndarray:
    a, b, c = [float(value) for value in parameters]
    x = _gjb932_x(response, c)
    if np.any(~np.isfinite(x)):
        return np.full_like(y, 1e150, dtype=float)
    residual = a + b * x - y
    if weighted:
        h_values = _gjb932_h_values(response, variance_model)
        residual = residual / h_values
    return residual


def _gjb932_variance_model(
    failures: pd.DataFrame,
    initial_nls: dict[str, object],
    confidence: float,
    iteration: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    # GJB/Z 18A 9.3.2.3.3: regress absolute residual scale against inverse strain.
    response = failures["e739_response"].to_numpy(dtype=float)
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
        ["e739_row_id", "e739_response", "e739_life", "e739_y_log10_life"]
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


def _gjb932_h_values(
    response: np.ndarray,
    variance_model: dict[str, float] | None,
) -> np.ndarray:
    response = np.asarray(response, dtype=float)
    if not variance_model or not bool(variance_model.get("use_weighted", False)):
        return np.ones_like(response, dtype=float)
    values = float(variance_model["sigma0"]) + float(variance_model["sigma1"]) / response
    return np.maximum(values, 1e-12)


def _gjb932_refit_data(active: pd.DataFrame, e_min_failure: float) -> pd.DataFrame:
    failure = active["e739_is_failure"].astype(bool)
    high_runout = (~failure) & (active["e739_response"].astype(float) > e_min_failure)
    refit = active[failure | high_runout].copy().reset_index(drop=True)
    refit["gjb932_temporary_failure"] = True
    refit["gjb932_original_is_failure"] = refit["e739_is_failure"].astype(bool)
    refit["gjb932_runout_treated_as_failure"] = ~refit["gjb932_original_is_failure"]
    if len(refit) < 4:
        raise ValueError("gjb932-strain refit requires at least four temporary failure points.")
    return refit


def _gjb932_fit_table(
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


def _gjb932_parameter_significance(
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


def _gjb932_fixed_a4_linear_fit(
    refit_data: pd.DataFrame,
    refit: dict[str, object],
    variance_model: dict[str, float],
    iteration: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    # Formula 136 correction after weighted fitting: fix A4 and re-estimate A1/A2 linearly.
    c = float(refit["coefficient_c"])
    response = refit_data["e739_response"].to_numpy(dtype=float)
    y = refit_data["e739_y_log10_life"].to_numpy(dtype=float)
    x = _gjb932_x(response, c)
    h_values = _gjb932_h_values(response, variance_model)
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


def _gjb932_residuals_and_outlier_statistics(
    refit_data: pd.DataFrame,
    fit: dict[str, object],
    variance_model: dict[str, float],
    iteration: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    # GJB/Z 18A 9.3.2.4.4: externally studentized residual maximum test.
    a = float(fit["coefficient_a"])
    b = float(fit["coefficient_b"])
    c = float(fit["coefficient_c"])
    response = refit_data["e739_response"].to_numpy(dtype=float)
    y = refit_data["e739_y_log10_life"].to_numpy(dtype=float)
    x = _gjb932_x(response, c)
    h_function = _gjb932_h_values(response, variance_model)
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
    table["gjb932_x"] = x
    table["gjb932_yhat"] = a + b * x
    table["gjb932_residual"] = residual
    table["gjb932_h_function"] = h_function
    table["gjb932_weighted_residual"] = weighted_residual
    table["gjb932_rmse"] = rmse
    table["gjb932_sd_i"] = sd_i
    table["gjb932_standardized_residual"] = standardized
    table["gjb932_leverage"] = leverage
    table["gjb932_rmse_i"] = rmse_i
    table["gjb932_studentized_residual"] = studentized
    table["gjb932_abs_studentized_residual"] = abs_t
    table["gjb932_outlier_critical"] = critical
    table["gjb932_is_max_abs_t"] = False
    if len(table):
        table.loc[table.index[max_index], "gjb932_is_max_abs_t"] = True
    summary = {
        "G": G,
        "critical": critical,
        "remove_outlier": remove_outlier,
        "removed_row_id": int(table.loc[table.index[max_index], "e739_row_id"]) if remove_outlier else None,
    }
    return table, summary


def _gjb932_final_mle(
    active: pd.DataFrame,
    fit: dict[str, object],
    variance_model: dict[str, float],
) -> dict[str, object]:
    # GJB/Z 18A 9.3.2.7.2: A4 and SD_i are fixed; only A1/A2 are corrected by MLE.
    c = float(fit["coefficient_c"])
    response = active["e739_response"].to_numpy(dtype=float)
    y = active["e739_y_log10_life"].to_numpy(dtype=float)
    failures = active["e739_is_failure"].astype(bool).to_numpy()
    domain = response > c
    response_fit = response[domain]
    y_fit = y[domain]
    failures_fit = failures[domain]
    if len(y_fit) < 3:
        raise ValueError("gjb932-strain final MLE has fewer than three domain-valid rows.")
    x = _gjb932_x(response_fit, c)
    h_values = _gjb932_h_values(response_fit, variance_model)
    sd_i = float(fit["rmse"]) * h_values
    initial_b = min(float(fit["coefficient_b"]), -1e-8)
    initial = np.array([float(fit["coefficient_a"]), np.log(max(-initial_b, 1e-8))])
    result = optimize.minimize(
        _gjb932_final_mle_negative_log_likelihood,
        initial,
        args=(x, y_fit, failures_fit, sd_i),
        method="L-BFGS-B",
        bounds=[(None, None), (-50.0, 50.0)],
    )
    a, log_minus_b = [float(value) for value in result.x]
    b = -float(np.exp(log_minus_b))
    nll = float(result.fun)
    hessian = _numerical_hessian(
        lambda params: _gjb932_final_mle_negative_log_likelihood(
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
    likelihood["gjb932_x"] = x
    likelihood["gjb932_sd_i"] = sd_i
    likelihood["gjb932_mu"] = mu
    likelihood["gjb932_w"] = (y_fit - mu) / sd_i
    likelihood["gjb932_logpdf"] = stats.norm.logpdf(likelihood["gjb932_w"]) - np.log(sd_i)
    likelihood["gjb932_logsf"] = stats.norm.logsf(likelihood["gjb932_w"])
    likelihood["gjb932_log_likelihood_i"] = np.where(
        failures_fit,
        likelihood["gjb932_logpdf"],
        likelihood["gjb932_logsf"],
    )
    if np.any(~domain):
        excluded = active[~domain].copy()
        excluded["gjb932_x"] = np.nan
        excluded["gjb932_sd_i"] = np.nan
        excluded["gjb932_mu"] = np.nan
        excluded["gjb932_w"] = np.nan
        excluded["gjb932_logpdf"] = np.nan
        excluded["gjb932_logsf"] = np.nan
        excluded["gjb932_log_likelihood_i"] = np.nan
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


def _gjb932_final_mle_negative_log_likelihood(
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


def _gjb932_curve(
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
        data["e739_x"].to_numpy(dtype=float),
        coefficient_a,
        coefficient_b,
    )
    x_grid = np.linspace(float(np.nanmin(x_values)), float(np.nanmax(x_values)), fit_points)
    response_grid = coefficient_c + np.power(10.0, x_grid)
    y_grid = coefficient_a + coefficient_b * x_grid
    return pd.DataFrame(
        {
            "e739_x": x_grid,
            "response": response_grid,
            "log10_life_fit": y_grid,
            "log10_life_lower_band": y_grid,
            "log10_life_upper_band": y_grid,
            "life_fit": np.power(10.0, y_grid),
            "life_lower_band": np.power(10.0, y_grid),
            "life_upper_band": np.power(10.0, y_grid),
        }
    ).sort_values("life_fit").reset_index(drop=True)


def _gjb932_extra_tables(
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
    if not runout_data.empty and "e739_x" in runout_data:
        values.append(runout_data["e739_x"].to_numpy(dtype=float))

    y_frames = [data["e739_y_log10_life"]]
    if not runout_data.empty and "e739_y_log10_life" in runout_data:
        y_frames.append(runout_data["e739_y_log10_life"])
    if abs(coefficient_b) > 1e-15:
        y_values = pd.concat(y_frames, ignore_index=True).to_numpy(dtype=float)
        values.append((y_values - coefficient_a) / coefficient_b)

    combined = np.concatenate(values)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return np.asarray(fitted_x, dtype=float)
    return combined


def _fit_shifted_log_response(response: np.ndarray, y: np.ndarray) -> dict[str, object]:
    response_min = float(np.min(response))
    response_max = float(np.max(response))
    response_span = response_max - response_min
    if response_span <= 0:
        raise ValueError("The E739 response column has no variation.")

    margin = max(abs(response_min), abs(response_span), 1.0) * 1e-10
    c_upper = response_min - margin
    c_lower = response_min - max(1000.0 * response_span, 1000.0 * abs(response_min), 1.0)
    if not c_lower < c_upper:
        raise ValueError("Could not create a valid search domain for C.")

    seeds: list[float] = [
        response_min - 0.05 * response_span,
        response_min - 0.25 * response_span,
        response_min - response_span,
        response_min - 5.0 * response_span,
        0.0,
    ]
    seeds = [min(max(seed, c_lower + margin), c_upper - margin) for seed in seeds]
    seeds = list(dict.fromkeys(round(seed, 15) for seed in seeds))

    best_result: optimize.OptimizeResult | None = None
    for c_initial in seeds:
        a_initial, b_initial = _linear_parameters_for_shift(response, y, c_initial)
        result = optimize.least_squares(
            _shifted_log_residual,
            x0=np.array([a_initial, b_initial, c_initial], dtype=float),
            args=(response, y),
            bounds=(
                np.array([-np.inf, -np.inf, c_lower], dtype=float),
                np.array([np.inf, np.inf, c_upper], dtype=float),
            ),
            x_scale=np.array([1.0, 1.0, max(abs(response_span), abs(response_min), 1.0)]),
            max_nfev=2000,
        )
        if not result.success:
            continue
        if best_result is None or result.cost < best_result.cost:
            best_result = result

    if best_result is None:
        raise ValueError("The shifted-log model did not converge.")

    coefficient_a, coefficient_b, coefficient_c = [float(value) for value in best_result.x]
    x = np.log10(response - coefficient_c)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_delta = x - x_mean
    y_delta = y - y_mean
    sxx = float(np.sum(x_delta**2))
    if sxx <= 0:
        raise ValueError("The shifted-log independent variable has no variation.")
    sxy = float(np.sum(x_delta * y_delta))
    y_hat = coefficient_a + coefficient_b * x
    residual = y - y_hat
    residual_sum_squares = float(np.sum(residual**2))
    degrees_of_freedom = len(response) - 3
    sigma_squared = residual_sum_squares / degrees_of_freedom
    covariance = sigma_squared * np.linalg.pinv(best_result.jac.T @ best_result.jac)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    return {
        "coefficient_a": coefficient_a,
        "coefficient_b": coefficient_b,
        "coefficient_c": coefficient_c,
        "x": x,
        "x_mean": x_mean,
        "sxx": sxx,
        "sxy": sxy,
        "y_hat": y_hat,
        "residual": residual,
        "covariance": covariance,
        "standard_error_a": float(standard_errors[0]),
        "standard_error_b": float(standard_errors[1]),
        "standard_error_c": float(standard_errors[2]),
    }


def _linear_parameters_for_shift(
    response: np.ndarray,
    y: np.ndarray,
    coefficient_c: float,
) -> tuple[float, float]:
    x = np.log10(response - coefficient_c)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    sxx = float(np.sum(np.power(x - x_mean, 2)))
    if sxx <= 0:
        return y_mean, 0.0
    sxy = float(np.sum((x - x_mean) * (y - y_mean)))
    coefficient_b = sxy / sxx
    coefficient_a = y_mean - coefficient_b * x_mean
    return float(coefficient_a), float(coefficient_b)


def _shifted_log_residual(
    parameters: np.ndarray,
    response: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    coefficient_a, coefficient_b, coefficient_c = parameters
    return coefficient_a + coefficient_b * np.log10(response - coefficient_c) - y


def _shifted_log_band_delta(
    x_grid: np.ndarray,
    coefficient_b: float,
    covariance: np.ndarray,
    simultaneous_band_factor: float,
) -> np.ndarray:
    shifted_response = np.power(10.0, x_grid)
    gradients = np.column_stack(
        (
            np.ones_like(x_grid),
            x_grid,
            -coefficient_b / (np.log(10.0) * shifted_response),
        )
    )
    variances = np.einsum("ij,jk,ik->i", gradients, covariance, gradients)
    return simultaneous_band_factor * np.sqrt(np.maximum(variances, 0.0))


def _fit_threshold_log_mle(
    response: np.ndarray,
    y: np.ndarray,
    failure_mask: np.ndarray,
) -> dict[str, object]:
    response_min = float(np.min(response))
    if response_min <= 0.0:
        raise ValueError("threshold_log_mle requires all response values to be positive.")
    epsilon = max(response_min * 1e-10, 1e-15)
    c_lower = 0.0
    c_upper = response_min - epsilon
    if c_upper <= c_lower:
        raise ValueError("threshold_log_mle requires min(response) to be greater than zero.")

    failures = np.asarray(failure_mask, dtype=bool)
    response_fail = response[failures]
    y_fail = y[failures]
    if len(y_fail) < 3:
        raise ValueError("At least three failure points are required for threshold_log_mle.")

    c_seeds = [0.0, 0.02 * response_min, 0.1 * response_min, 0.3 * response_min]
    c_seeds = [min(max(seed, c_lower), c_upper) for seed in c_seeds]
    c_seeds = list(dict.fromkeys(round(seed, 15) for seed in c_seeds))

    best_result: optimize.OptimizeResult | None = None
    for c_initial in c_seeds:
        a_initial, b_initial = _linear_parameters_for_shift(response_fail, y_fail, c_initial)
        if b_initial >= 0.0:
            b_initial = -1.0 if b_initial == 0.0 else -abs(b_initial)
        residual = y_fail - (a_initial + b_initial * np.log10(response_fail - c_initial))
        sigma_initial = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.1
        if not np.isfinite(sigma_initial) or sigma_initial <= 1e-8:
            sigma_initial = 0.1
        initial = np.array(
            [
                a_initial,
                np.log(max(-b_initial, 1e-8)),
                c_initial,
                np.log(sigma_initial),
            ],
            dtype=float,
        )
        result = optimize.minimize(
            _threshold_log_mle_negative_log_likelihood,
            initial,
            args=(response, y, failures),
            method="L-BFGS-B",
            bounds=[
                (None, None),
                (-50.0, 50.0),
                (c_lower, c_upper),
                (-50.0, 50.0),
            ],
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    if best_result is None:
        raise ValueError("threshold_log_mle did not run.")
    if not np.isfinite(best_result.fun):
        raise ValueError("threshold_log_mle did not converge to a finite likelihood.")

    a, log_minus_b, c, log_sigma = [float(value) for value in best_result.x]
    b = -float(np.exp(log_minus_b))
    sigma = float(np.exp(log_sigma))
    x = np.log10(response - c)
    y_hat = a + b * x
    residual = y - y_hat
    negative_log_likelihood = float(best_result.fun)
    log_likelihood = -negative_log_likelihood

    hessian = _numerical_hessian(
        lambda params: _threshold_log_mle_negative_log_likelihood(params, response, y, failures),
        best_result.x,
    )
    covariance_transformed = np.linalg.pinv(hessian)
    jacobian_original = np.diag([1.0, b, 1.0, sigma])
    covariance_original = jacobian_original @ covariance_transformed @ jacobian_original.T
    covariance_mean = covariance_original[:3, :3]
    standard_errors = np.sqrt(np.maximum(np.diag(covariance_original), 0.0))

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_delta = x - x_mean
    y_delta = y - y_mean
    sxx = float(np.sum(x_delta**2))
    if sxx <= 0:
        raise ValueError("The threshold-log independent variable has no variation.")
    sxy = float(np.sum(x_delta * y_delta))

    return {
        "coefficient_a": a,
        "coefficient_b": b,
        "coefficient_c": c,
        "sigma": sigma,
        "x": x,
        "x_mean": x_mean,
        "sxx": sxx,
        "sxy": sxy,
        "y_hat": y_hat,
        "residual": residual,
        "covariance": covariance_original,
        "covariance_mean": covariance_mean,
        "standard_error_a": float(standard_errors[0]),
        "standard_error_b": float(standard_errors[1]),
        "standard_error_c": float(standard_errors[2]),
        "standard_error_sigma": float(standard_errors[3]),
        "log_likelihood": log_likelihood,
        "negative_log_likelihood": negative_log_likelihood,
        "success": bool(best_result.success),
        "optimizer_message": str(best_result.message),
    }


def _threshold_log_mle_negative_log_likelihood(
    parameters: np.ndarray,
    response: np.ndarray,
    y: np.ndarray,
    failure_mask: np.ndarray,
) -> float:
    a, log_minus_b, c, log_sigma = [float(value) for value in parameters]
    if c < 0.0 or c >= float(np.min(response)):
        return 1e300
    shifted = response - c
    if np.any(shifted <= 0.0):
        return 1e300
    b = -float(np.exp(log_minus_b))
    sigma = float(np.exp(log_sigma))
    if not np.isfinite(sigma) or sigma <= 0.0:
        return 1e300
    mu = a + b * np.log10(shifted)
    failures = np.asarray(failure_mask, dtype=bool)
    log_likelihood = float(np.sum(stats.norm.logpdf(y[failures], loc=mu[failures], scale=sigma)))
    if np.any(~failures):
        z_runout = (y[~failures] - mu[~failures]) / sigma
        log_likelihood += float(np.sum(stats.norm.logsf(z_runout)))
    if not np.isfinite(log_likelihood):
        return 1e300
    return -log_likelihood


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


def _threshold_log_mle_band_delta(
    x_grid: np.ndarray,
    coefficient_b: float,
    covariance_mean: np.ndarray,
    simultaneous_band_factor: float,
) -> np.ndarray:
    shifted_response = np.power(10.0, x_grid)
    gradients = np.column_stack(
        (
            np.ones_like(x_grid),
            x_grid,
            -coefficient_b / (np.log(10.0) * shifted_response),
        )
    )
    variances = np.einsum("ij,jk,ik->i", gradients, covariance_mean, gradients)
    return simultaneous_band_factor * np.sqrt(np.maximum(variances, 0.0))


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
    return data["e739_x"].round(replicate_decimals).astype(str)


def _level_statistics(
    data: pd.DataFrame,
    coefficient_a: float,
    coefficient_b: float,
) -> pd.DataFrame:
    grouped = data.groupby("e739_level", sort=True, dropna=False)
    level_stats = grouped.agg(
        e739_level_x=("e739_x", "mean"),
        e739_level_y_mean=("e739_y_log10_life", "mean"),
        e739_level_count=("e739_y_log10_life", "size"),
        e739_level_y_std=("e739_y_log10_life", "std"),
    ).reset_index()
    level_stats["e739_level_yhat"] = (
        coefficient_a + coefficient_b * level_stats["e739_level_x"]
    )
    level_stats["e739_level_residual"] = (
        level_stats["e739_level_y_mean"] - level_stats["e739_level_yhat"]
    )
    return level_stats


def _linearity_test(
    data: pd.DataFrame,
    level_stats: pd.DataFrame,
    confidence: float,
    coefficient_a: float,
    coefficient_b: float,
    parameter_count: int,
) -> E739LinearityTest:
    k = len(data)
    levels = len(level_stats)
    lack_of_fit_df = levels - parameter_count
    pure_error_df = k - levels
    if lack_of_fit_df <= 0:
        return E739LinearityTest(
            available=False,
            reason=f"Need at least {parameter_count + 1} stress/strain levels.",
            levels=levels,
            specimens=k,
            lack_of_fit_df=None,
            pure_error_df=None,
            lack_of_fit_ss=None,
            pure_error_ss=None,
            lack_of_fit_ms=None,
            pure_error_ms=None,
            f_statistic=None,
            f_critical=None,
            p_value=None,
            reject_linear_model=None,
        )
    if pure_error_df <= 0:
        return E739LinearityTest(
            available=False,
            reason="Need replicated tests to estimate pure error.",
            levels=levels,
            specimens=k,
            lack_of_fit_df=lack_of_fit_df,
            pure_error_df=pure_error_df,
            lack_of_fit_ss=None,
            pure_error_ss=None,
            lack_of_fit_ms=None,
            pure_error_ms=None,
            f_statistic=None,
            f_critical=None,
            p_value=None,
            reject_linear_model=None,
        )

    y_mean_by_level = level_stats.set_index("e739_level")["e739_level_y_mean"]
    joined = data.join(y_mean_by_level, on="e739_level", rsuffix="_level")
    pure_error_ss = float(
        np.sum(
            np.power(
                joined["e739_y_log10_life"].to_numpy(dtype=float)
                - joined["e739_level_y_mean"].to_numpy(dtype=float),
                2,
            )
        )
    )
    yhat_level = coefficient_a + coefficient_b * level_stats["e739_level_x"].to_numpy(dtype=float)
    lack_of_fit_ss = float(
        np.sum(
            level_stats["e739_level_count"].to_numpy(dtype=float)
            * np.power(yhat_level - level_stats["e739_level_y_mean"].to_numpy(dtype=float), 2)
        )
    )
    lack_of_fit_ms = lack_of_fit_ss / lack_of_fit_df
    pure_error_ms = pure_error_ss / pure_error_df
    if pure_error_ms == 0.0:
        f_statistic = float("inf") if lack_of_fit_ms > 0.0 else 0.0
    else:
        f_statistic = float(lack_of_fit_ms / pure_error_ms)
    f_critical = float(stats.f.ppf(confidence, lack_of_fit_df, pure_error_df))
    p_value = float(stats.f.sf(f_statistic, lack_of_fit_df, pure_error_df))
    return E739LinearityTest(
        available=True,
        reason="",
        levels=levels,
        specimens=k,
        lack_of_fit_df=lack_of_fit_df,
        pure_error_df=pure_error_df,
        lack_of_fit_ss=lack_of_fit_ss,
        pure_error_ss=pure_error_ss,
        lack_of_fit_ms=float(lack_of_fit_ms),
        pure_error_ms=float(pure_error_ms),
        f_statistic=f_statistic,
        f_critical=f_critical,
        p_value=p_value,
        reject_linear_model=bool(f_statistic > f_critical),
    )


def _unavailable_linearity_test(
    data: pd.DataFrame,
    level_stats: pd.DataFrame,
    reason: str,
) -> E739LinearityTest:
    return E739LinearityTest(
        available=False,
        reason=reason,
        levels=len(level_stats),
        specimens=len(data),
        lack_of_fit_df=None,
        pure_error_df=None,
        lack_of_fit_ss=None,
        pure_error_ss=None,
        lack_of_fit_ms=None,
        pure_error_ms=None,
        f_statistic=None,
        f_critical=None,
        p_value=None,
        reject_linear_model=None,
    )


def _signed(value: float) -> str:
    if value < 0:
        return f"- {abs(value):.6g}"
    return f"+ {value:.6g}"


def _x_expression(
    response_column: str,
    x_transform: E739Transform,
    model: E739Model,
    coefficient_c: float | None,
) -> str:
    if model in ("shifted-log", "threshold_log_mle", GJB932_STRAIN_MODEL):
        return f"log10({_shifted_response_expression(response_column, coefficient_c)})"
    if x_transform == "log":
        return f"log10({response_column})"
    return response_column


def _log_life_formula(
    life_column: str,
    response_column: str,
    coefficient_a: float,
    coefficient_b: float,
    x_transform: E739Transform,
    model: E739Model,
    coefficient_c: float | None,
) -> str:
    x_expr = _x_expression(response_column, x_transform, model, coefficient_c)
    return f"log10({life_column}) = {coefficient_a:.6g} {_signed(coefficient_b)} * {x_expr}"


def _life_formula(
    life_column: str,
    response_column: str,
    coefficient_a: float,
    coefficient_b: float,
    x_transform: E739Transform,
    model: E739Model,
    coefficient_c: float | None,
) -> str:
    x_expr = _x_expression(response_column, x_transform, model, coefficient_c)
    return (
        f"{life_column} = 10^({coefficient_a:.6g} "
        f"{_signed(coefficient_b)} * {x_expr})"
    )


def _life_response_formula(
    life_column: str,
    response_column: str,
    coefficient_a: float,
    coefficient_b: float,
    x_transform: E739Transform,
    model: E739Model,
    coefficient_c: float | None,
) -> str:
    if model in ("shifted-log", "threshold_log_mle", GJB932_STRAIN_MODEL):
        return (
            f"{life_column} = {coefficient_a:.6g} * "
            f"({_shifted_response_expression(response_column, coefficient_c)})"
            f"^{coefficient_b:.6g}"
        )
    if x_transform == "log":
        return f"{life_column} = {coefficient_a:.6g} * {response_column}^{coefficient_b:.6g}"
    return (
        f"{life_column} = 10^({np.log10(coefficient_a):.6g} "
        f"{_signed(coefficient_b)} * {response_column})"
    )


def _response_life_formula(
    life_column: str,
    response_column: str,
    coefficient_a: float,
    coefficient_b: float,
    x_transform: E739Transform,
    model: E739Model,
    coefficient_c: float | None,
) -> str:
    if abs(coefficient_b) < 1e-15:
        return "Response-life form is undefined because B is approximately zero."
    if model in ("shifted-log", "threshold_log_mle", GJB932_STRAIN_MODEL):
        response_scale = float(np.power(10.0, -coefficient_a / coefficient_b))
        response_exponent = 1.0 / coefficient_b
        return (
            f"{response_column} = {coefficient_c:.6g} + "
            f"{response_scale:.6g} * {life_column}^{response_exponent:.6g}"
        )
    if x_transform == "log":
        response_scale = float(np.power(10.0, -coefficient_a / coefficient_b))
        response_exponent = 1.0 / coefficient_b
        return (
            f"{response_column} = {response_scale:.6g} * "
            f"{life_column}^{response_exponent:.6g}"
        )
    return (
        f"{response_column} = (log10({life_column}) "
        f"{_signed(-coefficient_a)}) / {coefficient_b:.6g}"
    )


def _shifted_response_expression(response_column: str, coefficient_c: float | None) -> str:
    if coefficient_c is None:
        return response_column
    if coefficient_c < 0:
        return f"{response_column} + {abs(coefficient_c):.6g}"
    return f"{response_column} - {coefficient_c:.6g}"


def predict_threshold_log_mle_life(
    result: E739FitResult,
    response: float | np.ndarray,
    failure_probability: float = 0.5,
) -> float | np.ndarray:
    """Predict life quantile N_p for the threshold_log_mle model."""
    _require_threshold_log_mle_result(result)
    response_array = np.asarray(response, dtype=float)
    _require_response_above_threshold(response_array, result.coefficient_c)
    z_value = stats.norm.ppf(float(failure_probability))
    log_life = (
        result.coefficient_a
        + result.coefficient_b * np.log10(response_array - result.coefficient_c)
        + z_value * result.sigma
    )
    predicted = np.power(10.0, log_life)
    return float(predicted) if np.ndim(response) == 0 else predicted


def inverse_threshold_log_mle_response(
    result: E739FitResult,
    life: float | np.ndarray,
    failure_probability: float = 0.5,
) -> float | np.ndarray:
    """Back-calculate response S_p(N0) for the threshold_log_mle model."""
    _require_threshold_log_mle_result(result)
    life_array = np.asarray(life, dtype=float)
    if np.any(life_array <= 0.0):
        raise ValueError("life must be positive.")
    z_value = stats.norm.ppf(float(failure_probability))
    response = result.coefficient_c + np.power(
        10.0,
        (np.log10(life_array) - result.coefficient_a - z_value * result.sigma)
        / result.coefficient_b,
    )
    return float(response) if np.ndim(life) == 0 else response


def _require_threshold_log_mle_result(result: E739FitResult) -> None:
    if result.model != "threshold_log_mle":
        raise ValueError("Prediction requires a threshold_log_mle fit result.")
    if result.coefficient_c is None:
        raise ValueError("threshold_log_mle result is missing coefficient C.")


def _require_response_above_threshold(response: np.ndarray, threshold: float) -> None:
    if np.any(response <= threshold):
        raise ValueError("response must be greater than threshold C for prediction.")
