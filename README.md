# OriginNSFitGJB

OriginNSFitGJB is a focused split from `OriginNSFit` for batch strain-life fitting and Origin plotting automation. The command line defaults to the simplified GJB/Z 18A 9.3.2 strain-life workflow (`gjb932-strain`) and can export CSV review tables, Origin projects, and PNG figures.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
```

`originpro`/`originpy` require Origin or OriginPro on the target Windows machine for real automation. Use `--dry-run` to verify fitting and CSV output without starting Origin.

## Run

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb --input examples --output output --pattern gjb932_strain_example.csv --status status --dry-run
```

Remove `--dry-run` on a machine with Origin installed:

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb --input data --output output --pattern "*.csv" --status status
```

Common output files:

```text
output/e739_summary.csv
output/e739_transformed_data.csv
output/e739_curve_bands.csv
output/e739_level_stats.csv
output/e739_initialols.csv
output/e739_initialnls.csv
output/e739_varianceanalysis.csv
output/e739_refitdata.csv
output/e739_refitresult.csv
output/e739_parametersignificance.csv
output/e739_residuals.csv
output/e739_finalmle.csv
output/e739_likelihood.csv
output/e739_analysis.opj
output/figures/
```

## Build EXE

```powershell
.\.venv\Scripts\pyinstaller.exe OriginNSFitGJB.spec
```

The executable is written to:

```text
dist\OriginNSFitGJB.exe
```

## Layout

```text
src/originnsfitgjb/      Python package and Origin automation code
examples/                Small CSV examples for verification
data/                    Local input data, ignored by Git except .gitkeep
output/                  Generated CSV/Origin/figure output, ignored by Git except .gitkeep
OriginNSFitGJB.spec      PyInstaller configuration
```
