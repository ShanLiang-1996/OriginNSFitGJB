param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$OfflineDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $OfflineDir
$Wheelhouse = Join-Path $OfflineDir "wheelhouse"
$Requirements = Join-Path $OfflineDir "requirements-offline.txt"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $Wheelhouse | Out-Null

& $Python -m pip download --only-binary=:all: --dest $Wheelhouse -r $Requirements
