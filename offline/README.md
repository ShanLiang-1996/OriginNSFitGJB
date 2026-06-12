# OriginNSFitGJB Offline Deployment

This folder supports installation on Windows machines that cannot access the internet.

Supported target prepared by the included wheelhouse:

- Windows 10/11 x64
- CPython 3.12 x64
- Origin or OriginPro already installed and activated

## One-Step Install

Run from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\offline\install_offline.ps1
```

The script creates `.venv`, installs dependencies from `offline/wheelhouse`, then installs the local `originnsfitgjb` package in editable mode.

## Validate

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb --input examples --output output --pattern gjb18a_strain_example.csv --status status --dry-run
```

Expected CSV outputs include:

```text
output\gjb_summary.csv
output\gjb_fit_data.csv
output\gjb_runout_data.csv
output\gjb_curve.csv
output\gjb_level_stats.csv
output\gjb_initialols.csv
output\gjb_initialnls.csv
output\gjb_varianceanalysis.csv
output\gjb_refitdata.csv
output\gjb_refitresult.csv
output\gjb_parametersignificance.csv
output\gjb_residuals.csv
```

## Run With Origin

After dry-run succeeds, put real data in `data/` and remove `--dry-run`:

```powershell
.\.venv\Scripts\python.exe -m originnsfitgjb --input data --output output --pattern "*.csv" --status status
```

The default Origin project path is:

```text
output\gjb_analysis.opj
```

## Build EXE Offline

```powershell
.\.venv\Scripts\pyinstaller.exe OriginNSFitGJB.spec
```

The executable is written to:

```text
dist\OriginNSFitGJB.exe
```

如需构建或运行 GUI 版本，离线 wheelhouse 还需要包含 `requirements-gui.txt` 中的 PySide6 相关 wheel。更新 wheelhouse 后再运行离线安装脚本。

## Refresh Wheelhouse

On an online Windows machine with the intended Python version:

```powershell
powershell -ExecutionPolicy Bypass -File .\offline\update_wheelhouse.ps1
```

Then copy the whole project folder to the offline machine.
