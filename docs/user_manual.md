# OriginNSFitGJB User Manual

## Workflow

OriginNSFitGJB runs one workflow only: the simplified GJB/Z 18A 9.3.2 Formula 136 strain-life fit.

The implemented equation is:

```text
log10(Nf) = A1 + A2 * log10(response - A4)
```

The response column is used directly as the simplified equivalent strain input.

## Processing Steps

1. Read supported input tables from CSV, TSV, TXT, XLS, or XLSX files.
2. Resolve life and response columns from explicit arguments or known keywords.
3. Convert life and response values to numeric data and drop invalid rows.
4. Mark failure and run-out rows from the optional status column.
5. Initialize A4 as half of the minimum failure response.
6. Run the initial nonlinear fit.
7. Build the residual variance model and decide whether weighted refit is needed.
8. Refit with eligible run-out handling required by the simplified workflow.
9. Check A2 and A4 parameter significance.
10. Check and optionally remove an outlier, then repeat the fit cycle.
11. Run final maximum-likelihood correction when significance checks pass.
12. Export review tables, fitted curves, and optional Origin project files.

## Common Commands

Dry-run:

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb --input examples --output output --pattern gjb18a_strain_example.csv --status status --dry-run
```

Origin project:

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb --input data --output output --pattern "*.csv" --status status
```

Build executable:

```powershell
.\.venv\Scripts\pyinstaller.exe OriginNSFitGJB.spec
```

## 图形界面

运行：

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb.gui
```

界面字段覆盖常用输入、输出、拟合、审计和主要 Origin 选项；高级命令行参数仍可通过 CLI 使用。首版主按钮会直接执行完整流程，包括 Origin 自动化。若 Python 侧 CSV 和审计输出已完成但 Origin 失败，请打开输出目录中的 `origin_automation.log` 查看原因。

## Output Tables

`gjb_summary.csv` contains one row per fitted table, including A1/A2/A4, confidence intervals, likelihood, residual statistics, formulas, and Origin output paths.

`gjb_fit_data.csv` contains the active fitted rows with normalized GJB columns such as `gjb_life`, `gjb_response`, `gjb_x`, `gjb_y_log10_life`, and residuals.

`gjb_runout_data.csv` contains run-out rows that remain visible in output review tables and Origin graphs.

`gjb_curve.csv` contains the fitted engineering curve sampled for plotting.

`gjb_initialols.csv`, `gjb_initialnls.csv`, `gjb_varianceanalysis.csv`, `gjb_refitdata.csv`, `gjb_refitresult.csv`, `gjb_parametersignificance.csv`, `gjb_residuals.csv`, `gjb_outlieriterations.csv`, `gjb_finalmle.csv`, `gjb_likelihood.csv`, and `gjb_modelchecks.csv` are step-by-step review tables intended for checking the calculation path.

## Origin Notes

Origin automation uses `originpro`, creates workbooks for each input group, then adds engineering and optional linearized graphs. If automation fails, check:

- Origin can be opened manually.
- No license or first-run dialog is blocking automation.
- The command succeeds with `--dry-run`.
- `output\origin_automation.log` contains the automation exception.
