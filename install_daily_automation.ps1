param(
    [ValidatePattern('^([01]\d|2[0-3]):[0-5]\d$')]
    [string]$Time = '11:00'
)

$ProjectDir = Split-Path -Parent $PSCommandPath
$Python = Join-Path $ProjectDir '.venv\Scripts\python.exe'
$Pipeline = Join-Path $ProjectDir 'daily_maintenance.py'
$TaskName = 'LetsBidDailyMaintenance'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python environment not found: $Python"
}

# The task runs only under the current signed-in account; this avoids storing
# a Windows password in the task. Its output is captured in downloads\logs.
$LogDir = Join-Path $ProjectDir 'downloads\logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Command = "cmd.exe /c `"`"$Python`" `"$Pipeline`" >> `"$LogDir\daily_maintenance.log`" 2>&1`""
schtasks.exe /Create /TN $TaskName /TR $Command /SC DAILY /ST $Time /F | Out-Host
Write-Host "Scheduled $TaskName for every day at $Time."
