<#
Registers the daily maintenance task after Phase 1 data has been copied into
this project folder. The task downloads new bids, creates new chunks, embeds
only those changes locally, and performs weekly expiry cleanup.
#>
param(
    [ValidatePattern('^([01]\d|2[0-3]):[0-5]\d$')]
    [string]$Time = '11:00'
)

$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSCommandPath
$RequiredPaths = @(
    'bid_chunks.json',
    'chroma_db',
    'downloads\downloaded_bid_manifest.json',
    'downloads\bids'
)

$MissingPaths = @(
    foreach ($RelativePath in $RequiredPaths) {
        $FullPath = Join-Path $ProjectDir $RelativePath
        if (-not (Test-Path -LiteralPath $FullPath)) {
            $RelativePath
        }
    }
)

if ($MissingPaths.Count -gt 0) {
    throw "Phase 1 data is incomplete. Copy these into this folder first: $($MissingPaths -join ', ')"
}

& (Join-Path $ProjectDir 'install_daily_automation.ps1') -Time $Time
Write-Host "Phase 2 is active. New bids will be processed daily at $Time, and expiry cleanup runs weekly."
