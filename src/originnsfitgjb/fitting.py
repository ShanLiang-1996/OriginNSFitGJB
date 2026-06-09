from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class SNCurveFitResult:
    coefficient_a: float
    coefficient_b: float
    log10_intercept: float
    log10_slope: float
    r2: float
    rmse: float
    points: int
    life_min: float
    life_max: float
    fit_response_at_life_min: float
    fit_response_at_life_max: float
    response_min: float
    response_max: float
    formula: str
    log10_formula: str


@dataclass(frozen=True)
class SNCurveFit:
    result: SNCurveFitResult
    data: pd.DataFrame
    curve: pd.DataFrame


def sn_power_law_model(life: np.ndarray, coefficient_a: float, coefficient_b: float) -> np.ndarray:
    return coefficient_a * np.power(life, coefficient_b)


def fit_sn_power_law(
    frame: pd.DataFrame,
    life_column: str,
    response_column: str,
    fit_points: int = 300,
) -> SNCurveFit:
    data = frame[[life_column, response_column]].copy()
    data[life_column] = pd.to_numeric(data[life_column], errors="coerce")
    data[response_column] = pd.to_numeric(data[response_column], errors="coerce")
    data = data.dropna()
    data = data[(data[life_column] > 0) & (data[response_column] > 0)]
    data = data.sort_values(life_column).reset_index(drop=True)

    if len(data) < 2:
        raise ValueError("At least two positive S-N points are required for power-law fitting.")

    life = data[life_column].to_numpy(dtype=float)
    response = data[response_column].to_numpy(dtype=float)
    log_life = np.log(life)

    slope, intercept = np.polyfit(log_life, np.log(response), deg=1)
    initial_a = float(np.exp(intercept))
    initial_b = float(slope)

    coefficients, _ = curve_fit(
        sn_power_law_model,
        life,
        response,
        p0=(initial_a, initial_b),
        maxfev=10000,
    )
    coefficient_a = float(coefficients[0])
    coefficient_b = float(coefficients[1])
    log10_intercept = float(np.log10(coefficient_a))
    log10_slope = coefficient_b

    predicted = sn_power_law_model(life, coefficient_a, coefficient_b)
    residual_sum = float(np.sum((response - predicted) ** 2))
    total_sum = float(np.sum((response - np.mean(response)) ** 2))
    r2 = 1.0 if total_sum == 0 else 1.0 - residual_sum / total_sum
    rmse = float(np.sqrt(np.mean((response - predicted) ** 2)))

    life_fit = np.logspace(np.log10(life.min()), np.log10(life.max()), fit_points)
    response_fit = sn_power_law_model(life_fit, coefficient_a, coefficient_b)
    formula = f"\\Delta \\epsilon = {coefficient_a:.6g} (N_f)^{{{coefficient_b:.6g}}}"
    log10_formula = (
        f"log10({response_column}) = {log10_intercept:.6g} "
        f"+ {log10_slope:.6g} * log10({life_column})"
    )

    curve = pd.DataFrame(
        {
            life_column: life_fit,
            response_column: response_fit,
            f"log10_{life_column}": np.log10(life_fit),
            f"log10_{response_column}": np.log10(response_fit),
        }
    )
    result = SNCurveFitResult(
        coefficient_a=coefficient_a,
        coefficient_b=coefficient_b,
        log10_intercept=log10_intercept,
        log10_slope=log10_slope,
        r2=float(r2),
        rmse=rmse,
        points=int(len(data)),
        life_min=float(life.min()),
        life_max=float(life.max()),
        fit_response_at_life_min=float(response_fit[0]),
        fit_response_at_life_max=float(response_fit[-1]),
        response_min=float(response.min()),
        response_max=float(response.max()),
        formula=formula,
        log10_formula=log10_formula,
    )
    return SNCurveFit(result=result, data=data, curve=curve)
