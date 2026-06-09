param(
    [string]$Python = "python",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

$OfflineDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $OfflineDir
$Wheelhouse = Join-Path $OfflineDir "wheelhouse"
$Requirements = Join-Path $OfflineDir "requirements-offline.txt"
$ResolvedVenv = Join-Path $ProjectRoot $VenvPath
$VenvPython = Join-Path $ResolvedVenv "Scripts\python.exe"

Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $Wheelhouse)) {
    throw "Missing offline wheelhouse: $Wheelhouse"
}

if (-not (Test-Path -LiteralPath $Requirements)) {
    throw "Missing offline requirements file: $Requirements"
}

if (-not (Test-Path -LiteralPath $ResolvedVenv)) {
    Write-Host "Creating virtual environment at $ResolvedVenv"
    & $Python -m venv $ResolvedVenv
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Virtual environment Python was not found: $VenvPython"
}

Write-Host "Installing packaging tools from offline wheelhouse"
& $VenvPython -m pip install --no-index --find-links $Wheelhouse pip setuptools wheel

Write-Host "Installing project dependencies from offline wheelhouse"
& $VenvPython -m pip install --no-index --find-links $Wheelhouse -r $Requirements

Write-Host "Installing OriginNSFitGJB package"
& $VenvPython -m pip install --no-index --find-links $Wheelhouse --no-build-isolation --no-deps -e $ProjectRoot

Write-Host ""
Write-Host "Offline installation finished."
Write-Host "Validate with:"
Write-Host ".\.venv\Scripts\python.exe -m originnsfitgjb --input examples --output output --pattern gjb18a_strain_example.csv --status status --dry-run"
