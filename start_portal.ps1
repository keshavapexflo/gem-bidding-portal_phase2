<# Starts the local Streamlit search portal. #>
param()

$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSCommandPath
$Python = Join-Path $ProjectDir '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Setup has not been run. Run .\setup_new_laptop.ps1 first.'
}

Push-Location $ProjectDir
try {
    & $Python -m streamlit run .\app.py
} finally {
    Pop-Location
}
