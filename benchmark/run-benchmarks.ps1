<#
.SYNOPSIS
    Runs NumSharp and NumPy benchmarks and generates a consolidated Markdown report.

.DESCRIPTION
    This script executes both C# (BenchmarkDotNet) and Python (NumPy) benchmarks,
    collects the results, calculates performance ratios, and generates a comprehensive
    Markdown report for comparison.

.PARAMETER Quick
    Run quick benchmarks (fewer iterations, faster but less accurate)

.PARAMETER Suite
    Specific suite to run: 'all', 'dispatch', 'fusion', 'arithmetic', 'unary',
    'reduction', 'broadcast', 'creation', 'manipulation', 'slicing'

.PARAMETER OutputPath
    Path for the output report (default: benchmark-report.md)

.PARAMETER SkipCSharp
    Skip C# benchmarks

.PARAMETER SkipPython
    Skip Python benchmarks

.PARAMETER Type
    Specific dtype to benchmark (e.g., int32, float64)

.PARAMETER Size
    Array size: small (1K), medium (100K), large (10M)

.EXAMPLE
    .\run-benchmarks.ps1
    .\run-benchmarks.ps1 -Quick
    .\run-benchmarks.ps1 -Suite arithmetic -Type int32
    .\run-benchmarks.ps1 -Suite all -OutputPath results.md
#>

param(
    [switch]$Quick,
    [ValidateSet('all', 'dispatch', 'fusion', 'arithmetic', 'unary', 'reduction',
                 'broadcast', 'creation', 'manipulation', 'slicing')]
    [string]$Suite = 'all',
    [string]$OutputPath = 'benchmark-report.md',
    [switch]$SkipCSharp,
    [switch]$SkipPython,
    [string]$Type = '',
    [ValidateSet('', 'small', 'medium', 'large')]
    [string]$Size = ''
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ReportPath = Join-Path $ScriptDir $OutputPath
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Colors for console output
function Write-Status { param($msg) Write-Host "`u{25B6} $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "`u{2713} $msg" -ForegroundColor Green }
function Write-Warning { param($msg) Write-Host "`u{26A0} $msg" -ForegroundColor Yellow }

# Performance status icons
function Get-StatusIcon {
    param([double]$Ratio)
    if ($Ratio -le 1.0) { return "`u{2705}" }      # Green check - faster
    if ($Ratio -le 2.0) { return "`u{1F7E1}" }     # Yellow circle - close
    if ($Ratio -le 5.0) { return "`u{1F7E0}" }     # Orange circle - slower
    return "`u{1F534}"                               # Red circle - much slower
}

function Get-StatusText {
    param([double]$Ratio)
    if ($Ratio -le 1.0) { return "faster" }
    if ($Ratio -le 2.0) { return "~2x" }
    if ($Ratio -le 5.0) { return "$([math]::Round($Ratio, 1))x slower" }
    return "$([math]::Round($Ratio, 1))x slower"
}

Write-Host ""
Write-Host "`u{2554}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2557}" -ForegroundColor Blue
Write-Host "`u{2551}       NumSharp vs NumPy Comprehensive Benchmark Suite       `u{2551}" -ForegroundColor Blue
Write-Host "`u{255A}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{255D}" -ForegroundColor Blue
Write-Host ""

# Initialize report
$report = @"
# NumSharp Performance Benchmark Report

**Generated:** $Timestamp
**Mode:** $(if ($Quick) { 'Quick (reduced iterations)' } else { 'Full' })
**Suite:** $Suite

---

## Environment

| Component | Version |
|-----------|---------|
"@

# Get environment info
$dotnetVersion = & dotnet --version 2>$null
$pythonVersion = & python --version 2>$null
$numpyVersion = & python -c "import numpy; print(numpy.__version__)" 2>$null

$report += "`n| .NET SDK | $dotnetVersion |"
$report += "`n| Python | $($pythonVersion -replace 'Python ', '') |"
$report += "`n| NumPy | $numpyVersion |"
$report += "`n| OS | $([System.Environment]::OSVersion.VersionString) |"

try {
    $cpuName = (Get-CimInstance Win32_Processor).Name
    $report += "`n| CPU | $cpuName |"
} catch {
    $report += "`n| CPU | Unknown |"
}

$report += "`n"

# =============================================================================
# Run Python Benchmarks
# =============================================================================

$pythonResults = @()
$pythonJsonPath = Join-Path $ScriptDir "benchmark-report.json"

if (-not $SkipPython) {
    Write-Status "Running Python/NumPy benchmarks..."

    $pythonScript = Join-Path $ScriptDir "NumSharp.Benchmark.Python\numpy_benchmark.py"
    $pythonArgs = @($pythonScript, "--output", $pythonJsonPath)
    if ($Quick) { $pythonArgs += "--quick" }
    if ($Suite -ne 'all') { $pythonArgs += "--suite"; $pythonArgs += $Suite }
    if ($Type) { $pythonArgs += "--type"; $pythonArgs += $Type }
    if ($Size) { $pythonArgs += "--size"; $pythonArgs += $Size }

    try {
        & python @pythonArgs 2>&1 | ForEach-Object { Write-Host "  $_" }

        if (Test-Path $pythonJsonPath) {
            $pythonResults = Get-Content $pythonJsonPath | ConvertFrom-Json
            Write-Success "Python benchmarks complete ($($pythonResults.Count) results)"
        }
    } catch {
        Write-Warning "Python benchmarks failed: $_"
    }
}

# =============================================================================
# Run C# Benchmarks
# =============================================================================

$csharpResults = @{}
$csharpJsonDir = $null

if (-not $SkipCSharp) {
    Write-Status "Building C# benchmarks..."

    $csharpDir = Join-Path $ScriptDir "NumSharp.Benchmark.Core"
    Push-Location $csharpDir

    try {
        & dotnet build -c Release -v q --nologo 2>$null | Out-Null
        Write-Success "Build complete"

        Write-Status "Running C# benchmarks (this may take a few minutes)..."

        $jobType = if ($Quick) { "Short" } else { "Medium" }

        # Build filter based on suite
        $filter = switch ($Suite) {
            'dispatch' { "*Dispatch*" }
            'fusion' { "*Fusion*" }
            'arithmetic' { "*Arithmetic*" }
            'unary' { "*Unary*,*Math*,*ExpLog*,*Trig*,*Power*" }
            'reduction' { "*Reduction*,*Sum*,*Mean*,*VarStd*,*MinMax*,*Prod*" }
            'broadcast' { "*Broadcast*" }
            'creation' { "*Creation*" }
            'manipulation' { "*Manipulation*,*Reshape*,*Stack*,*Dims*" }
            'slicing' { "*Slice*" }
            default { "*" }
        }

        # Run and capture output
        $output = & dotnet run -c Release --no-build -f net10.0 -- --job $jobType --filter $filter --exporters json 2>&1

        # Parse the summary table from output
        $inTable = $false
        $tableLines = @()
        foreach ($line in $output) {
            if ($line -match '^\| Method') { $inTable = $true }
            if ($inTable -and $line -match '^\|') { $tableLines += $line }
            if ($inTable -and $line -notmatch '^\|' -and $line -match '\S') { $inTable = $false }
        }

        if ($tableLines.Count -gt 0) {
            $csharpResults['table'] = $tableLines -join "`n"
            Write-Success "C# benchmarks complete"
        }

        # Find JSON results directory
        $artifactsDir = Join-Path $csharpDir "BenchmarkDotNet.Artifacts\results"
        if (Test-Path $artifactsDir) {
            $csharpJsonDir = $artifactsDir
        }

    } catch {
        Write-Warning "C# benchmarks failed: $_"
    } finally {
        Pop-Location
    }
}

# =============================================================================
# Generate Executive Summary
# =============================================================================

Write-Status "Generating report..."

$report += @"

---

## Executive Summary

"@

if ($pythonResults.Count -gt 0) {
    # Calculate statistics
    $totalOps = $pythonResults.Count
    $faster = 0
    $within2x = 0
    $slower = 0
    $muchSlower = 0

    # Group by suite for summary
    $suiteStats = @{}
    $pythonResults | Group-Object -Property suite | ForEach-Object {
        $suiteStats[$_.Name] = @{
            Count = $_.Count
            AvgMs = ($_.Group | Measure-Object -Property mean_ms -Average).Average
        }
    }

    $report += @"
| Metric | Value |
|--------|-------|
| Operations Tested | $totalOps |
| Suites | $($suiteStats.Keys -join ', ') |
| Array Size (N) | $(if ($pythonResults[0].n) { "{0:N0}" -f $pythonResults[0].n } else { "Varies" }) |

"@
}

# =============================================================================
# NumPy Baseline Performance
# =============================================================================

if ($pythonResults.Count -gt 0) {
    $report += @"

---

## NumPy Baseline Performance

> These results represent NumPy's performance on the same operations.
> NumPy is the reference implementation NumSharp aims to match.

"@

    # Group results by suite
    $suiteGroups = $pythonResults | Group-Object -Property suite

    foreach ($suiteGroup in $suiteGroups) {
        $suiteName = if ($suiteGroup.Name) { $suiteGroup.Name } else { "General" }
        $report += "`n### $suiteName`n`n"
        $report += "| Operation | Type | Mean (ms) | StdDev |`n"
        $report += "|-----------|------|----------:|-------:|`n"

        foreach ($r in $suiteGroup.Group) {
            $report += "| $($r.name) | $($r.dtype) | $([math]::Round($r.mean_ms, 3)) | $([math]::Round($r.stddev_ms, 3)) |`n"
        }
    }
}

# =============================================================================
# C# Results Section
# =============================================================================

if ($csharpResults['table']) {
    $report += @"

---

## NumSharp (C#) Benchmark Results

> BenchmarkDotNet results for NumSharp operations.

### Summary Table

$($csharpResults['table'])

"@
}

$report += @"

---

## Quick Reference

Ratio = NumSharp / NumPy | ✅ ≤1x | 🟡 ≤2x | 🟠 ≤5x | 🔴 >5x

**See benchmark-report.md for the full comparison matrix.**

---

*Generated by run-benchmarks.ps1*
"@

# Write report
$report | Out-File -FilePath $ReportPath -Encoding UTF8

Write-Host ""
Write-Success "Report generated: $ReportPath"

# =============================================================================
# Generate Unified Comparison (if both results exist)
# =============================================================================

$mergeScript = Join-Path $ScriptDir "scripts\merge-results.py"

if ((Test-Path $pythonJsonPath) -and (Test-Path $mergeScript)) {
    Write-Status "Generating unified comparison..."
    try {
        $outputBase = Join-Path $ScriptDir "benchmark-report"
        & python $mergeScript --numpy $pythonJsonPath --output $outputBase 2>&1 | ForEach-Object { Write-Host "  $_" }
        Write-Success "Unified results generated"
    } catch {
        Write-Warning "Failed to generate unified results: $_"
    }
}

# =============================================================================
# Copy report to README.md (only if README.md already exists)
# =============================================================================

$readmePath = Join-Path $ScriptDir "README.md"
if (Test-Path $readmePath) {
    Copy-Item -Path $ReportPath -Destination $readmePath -Force
    Write-Success "README.md updated with benchmark results"
} else {
    Write-Warning "README.md not found - skipping auto-update (create it manually to enable)"
}

Write-Host ""

# Display summary
Write-Host "`u{2554}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2557}" -ForegroundColor Green
Write-Host "`u{2551}  Benchmark Complete!                                         `u{2551}" -ForegroundColor Green
Write-Host "`u{255A}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{255D}" -ForegroundColor Green
Write-Host ""
Write-Host "  Report: " -NoNewline; Write-Host $ReportPath -ForegroundColor Yellow
Write-Host "  README: " -NoNewline; Write-Host $readmePath -ForegroundColor Yellow
Write-Host "  JSON:   " -NoNewline; Write-Host $pythonJsonPath -ForegroundColor Yellow
Write-Host "  View:   " -NoNewline; Write-Host "code $ReportPath" -ForegroundColor Cyan
Write-Host ""
