<#
Creates the isolated Python environment and installs this portal's
dependencies. Run once from PowerShell on each new laptop.
#>
param()

$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSCommandPath
$VenvPython = Join-Path $ProjectDir '.venv\Scripts\python.exe'

function Test-Python311 {
    param([string[]]$Command)
    $pythonArgs = @()
    if ($Command.Length -gt 1) {
        $pythonArgs = $Command[1..($Command.Length - 1)]
    }
    $version = & $Command[0] @pythonArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    return $version -eq '3.11'
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCommand = @('py', '-3.11')
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCommand = @('python')
} else {
    throw 'Python 3.11 was not found. Install Python 3.11 from python.org, then run this script again.'
}

if (-not (Test-Python311 $PythonCommand)) {
    throw 'Python 3.11 is required. Install it from python.org (including the Python Launcher), then run this script again.'
}

Push-Location $ProjectDir
try {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-Host 'Creating Python virtual environment...'
        $pythonArgs = @()
        if ($PythonCommand.Length -gt 1) {
            $pythonArgs = $PythonCommand[1..($PythonCommand.Length - 1)]
        }
        & $PythonCommand[0] @pythonArgs -m venv .venv
    }
    Write-Host 'Installing dependencies (the first install can take several minutes)...'
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r requirements.txt
    foreach ($folder in @('downloads\bids', 'downloads\expired', 'downloads\logs', 'chroma_db', 'static')) {
        New-Item -ItemType Directory -Force -Path $folder | Out-Null
    }
    Write-Host 'Setup complete. Copy the Phase 1 data, then run .\activate_phase_2.ps1.'
} finally {
    Pop-Location
}
