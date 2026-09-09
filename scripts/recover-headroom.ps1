[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$NoLaunch,
    [int]$Port = 8787,
    [string]$ContainerName = 'headroom-default'
)

$ErrorActionPreference = 'Stop'

function Write-Step {
    param([string]$Message)
    Write-Host "[headroom-recover] $Message" -ForegroundColor Cyan
}

function Test-HeadroomReady {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/readyz" -TimeoutSec 5
        return $response.ready -eq $true
    }
    catch {
        return $false
    }
}

function Get-PortOwner {
    $listener = Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $listener) { return $null }

    $process = Get-Process -Id $listener.OwningProcess -ErrorAction Stop
    $commandLine = $null
    try {
        $commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)" -ErrorAction Stop).CommandLine
    }
    catch {
        # Command lines may require elevation. Treat an unknown owner as unsafe to stop.
    }

    [pscustomobject]@{ Id = $listener.OwningProcess; Name = $process.ProcessName; CommandLine = $commandLine }
}

function Test-HeadroomProcess {
    param($ProcessInfo)
    return $ProcessInfo.CommandLine -match '(?i)headroom'
}

function Invoke-Docker {
    param([string[]]$Arguments)
    if ($DryRun) {
        Write-Step "DRY RUN: docker $($Arguments -join ' ')"
        return
    }
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) { throw "docker $($Arguments -join ' ') failed with exit code $LASTEXITCODE." }
}

Write-Step "Checking Headroom on http://127.0.0.1:$Port/readyz"
if (-not (Test-HeadroomReady)) {
    $owner = Get-PortOwner
    if ($owner) {
        if (-not (Test-HeadroomProcess $owner)) {
            throw "Port $Port is owned by PID $($owner.Id) ($($owner.Name)), which is not confirmed as Headroom. It was not stopped. Close that application or free the port, then run this script again."
        }
        Write-Step "Stopping stale Headroom process PID $($owner.Id) ($($owner.Name))."
        if ($DryRun) { Write-Step "DRY RUN: Stop-Process -Id $($owner.Id) -Force" }
        else { Stop-Process -Id $owner.Id -Force; Start-Sleep -Seconds 1 }
    }

    $containerId = & docker container inspect --format '{{.Id}}' $ContainerName 2>$null
    if ($LASTEXITCODE -eq 0 -and $containerId) {
        Write-Step "Starting persistent Docker container '$ContainerName'."
        Invoke-Docker @('start', $ContainerName)
    }
    else {
        Write-Step "No persistent Docker container exists yet. 'headroom wrap codex' will create it."
    }

    if (-not $DryRun) {
        $deadline = (Get-Date).AddSeconds(50)
        do { if (Test-HeadroomReady) { break }; Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline)
        if (-not (Test-HeadroomReady)) {
            throw "Headroom did not become ready on port $Port within 50 seconds. Run: docker logs $ContainerName"
        }
    }
}

if ($DryRun) {
    Write-Step 'DRY RUN complete. No processes, containers, or Codex session were started.'
    exit 0
}

Write-Step 'Headroom is ready.'
if ($NoLaunch) { Write-Step 'Skipping Codex launch because -NoLaunch was supplied.'; exit 0 }

Write-Step 'Launching: headroom wrap codex'
& headroom wrap codex
exit $LASTEXITCODE
