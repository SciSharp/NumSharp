<#
.SYNOPSIS
    Runs NumSharp and NumPy benchmarks and generates a consolidated Markdown report.

.DESCRIPTION
    This script executes both C# (BenchmarkDotNet) and Python (NumPy) benchmarks,
    collects the results, calculates performance ratios, and generates a comprehensive
    Markdown report for comparison.

    Results are saved to benchmark/results/yyyyMMdd-HHmmss/ and copied to benchmark/.

.PARAMETER Quick
    Run quick benchmarks (fewer iterations, faster but less accurate)

.PARAMETER Suite
    Specific suite to run: 'all', 'dispatch', 'fusion', 'arithmetic', 'unary',
    'reduction', 'broadcast', 'creation', 'manipulation', 'slicing'

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
    .\run-benchmarks.ps1 -Suite all
#>

param(
    [switch]$Quick,
    [ValidateSet('all', 'arithmetic', 'unary', 'reduction', 'broadcast', 'creation',
                 'manipulation', 'slicing', 'experimental', 'dispatch', 'fusion')]
    [string]$Suite = 'all',
    [switch]$SkipCSharp,
    [switch]$SkipPython,
    [string]$Type = '',
    [ValidateSet('', 'small', 'medium', 'large')]
    [string]$Size = ''
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TimestampFolder = Get-Date -Format "yyyyMMdd-HHmmss"
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# =============================================================================
# Setup Results Directory
# =============================================================================

$ResultsDir = Join-Path $ScriptDir "results\$TimestampFolder"
New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

$LogFile = Join-Path $ResultsDir "benchmark.log"

# Logging function - writes to both console and log file
function Write-Log {
    param($msg, $Color = "White", $NoNewline = $false)
    $timestamp = Get-Date -Format "HH:mm:ss"
    $logMsg = "[$timestamp] $msg"
    Add-Content -Path $LogFile -Value $logMsg
    if ($NoNewline) {
        Write-Host $msg -ForegroundColor $Color -NoNewline
    } else {
        Write-Host $msg -ForegroundColor $Color
    }
}

function Write-Status { param($msg) Write-Log "`u{25B6} $msg" -Color Cyan }
function Write-Success { param($msg) Write-Log "`u{2713} $msg" -Color Green }
function Write-Warn { param($msg) Write-Log "`u{26A0} $msg" -Color Yellow }

# Performance status icons
function Get-StatusIcon {
    param([double]$Ratio)
    if ($Ratio -le 1.0) { return "`u{2705}" }      # Green check - faster
    if ($Ratio -le 2.0) { return "`u{1F7E1}" }     # Yellow circle - close
    if ($Ratio -le 5.0) { return "`u{1F7E0}" }     # Orange circle - slower
    return "`u{1F534}"                               # Red circle - much slower
}

# =============================================================================
# Banner
# =============================================================================

Write-Host ""
Write-Host "`u{2554}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2557}" -ForegroundColor Blue
Write-Host "`u{2551}       NumSharp vs NumPy Comprehensive Benchmark Suite       `u{2551}" -ForegroundColor Blue
Write-Host "`u{255A}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{255D}" -ForegroundColor Blue
Write-Host ""

Write-Log "Results directory: $ResultsDir"
Write-Log "Parameters: Suite=$Suite, Quick=$Quick, SkipCSharp=$SkipCSharp, SkipPython=$SkipPython, Type=$Type, Size=$Size"

# =============================================================================
# Initialize Report
# =============================================================================

$report = @"
# NumSharp Performance Benchmark Report

**Generated:** $Timestamp
**Mode:** $(if ($Quick) { 'Quick (reduced iterations)' } else { 'Full' })
**Suite:** $Suite
**Results:** results/$TimestampFolder/

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

Write-Log "Environment: .NET $dotnetVersion, Python $($pythonVersion -replace 'Python ', ''), NumPy $numpyVersion"

# =============================================================================
# Run Python Benchmarks
# =============================================================================

$pythonResults = @()
$numpyJsonPath = Join-Path $ResultsDir "numpy-results.json"

if (-not $SkipPython) {
    Write-Status "Running Python/NumPy benchmarks..."

    $pythonScript = Join-Path $ScriptDir "NumSharp.Benchmark.Python\numpy_benchmark.py"
    $pythonArgs = @($pythonScript, "--output", $numpyJsonPath)
    if ($Quick) { $pythonArgs += "--quick" }
    if ($Suite -ne 'all') { $pythonArgs += "--suite"; $pythonArgs += $Suite }
    if ($Type) { $pythonArgs += "--type"; $pythonArgs += $Type }
    if ($Size) { $pythonArgs += "--size"; $pythonArgs += $Size }

    Write-Log "Python command: python $($pythonArgs -join ' ')"

    try {
        & python @pythonArgs 2>&1 | ForEach-Object {
            Write-Host "  $_"
            Add-Content -Path $LogFile -Value "  [Python] $_"
        }

        if (Test-Path $numpyJsonPath) {
            $pythonResults = Get-Content $numpyJsonPath | ConvertFrom-Json
            Write-Success "Python benchmarks complete ($($pythonResults.Count) results)"
        }
    } catch {
        Write-Warn "Python benchmarks failed: $_"
        Add-Content -Path $LogFile -Value "ERROR: Python benchmarks failed: $_"
    }
} else {
    Write-Log "Skipping Python benchmarks (-SkipPython)"
}

# =============================================================================
# Run C# Benchmarks
# =============================================================================

$csharpResults = @{}
$numsharpJsonPath = Join-Path $ResultsDir "numsharp-results.json"

if (-not $SkipCSharp) {
    Write-Status "Building C# benchmarks..."

    $csharpDir = Join-Path $ScriptDir "NumSharp.Benchmark.Core"
    Push-Location $csharpDir

    try {
        $buildOutput = & dotnet build -c Release -v q --nologo 2>&1
        Add-Content -Path $LogFile -Value "Build output: $buildOutput"
        Write-Success "Build complete"

        Write-Status "Running C# benchmarks (this may take a few minutes)..."

        $jobType = if ($Quick) { "Short" } else { "Medium" }

        # Build filter based on suite
        # Note: 'dispatch' and 'fusion' are Experimental (C# internals, not for NumPy comparison)
        $filter = switch ($Suite) {
            'arithmetic' { "*Arithmetic*" }
            'unary' { "*Unary*,*Math*,*ExpLog*,*Trig*,*Power*" }
            'reduction' { "*Reduction*,*Sum*,*Mean*,*VarStd*,*MinMax*,*Prod*" }
            'broadcast' { "*Broadcast*" }
            'creation' { "*Creation*" }
            'manipulation' { "*Manipulation*,*Reshape*,*Stack*,*Dims*" }
            'slicing' { "*Slice*" }
            'experimental' { "*Experimental*" }
            'dispatch' { "*Experimental*Dispatch*" }
            'fusion' { "*Experimental*Fusion*" }
            default { "*" }  # 'all' runs everything (merge script filters by operation name match)
        }

        Write-Log "C# command: dotnet run -c Release --no-build -f net10.0 -- --job $jobType --filter $filter --exporters json"

        # Run and capture output
        $output = & dotnet run -c Release --no-build -f net10.0 -- --job $jobType --filter $filter --exporters json 2>&1

        # Log all output
        $output | ForEach-Object { Add-Content -Path $LogFile -Value "  [C#] $_" }

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

        # Find and consolidate BenchmarkDotNet JSON results
        $artifactsDir = Join-Path $csharpDir "BenchmarkDotNet.Artifacts\results"
        if (Test-Path $artifactsDir) {
            # Consolidate all BDN results into numsharp-results.json
            $allBenchmarks = @()
            Get-ChildItem -Path $artifactsDir -Filter "*-report-full-compressed.json" | ForEach-Object {
                $bdnContent = Get-Content $_.FullName | ConvertFrom-Json
                if ($bdnContent.Benchmarks) {
                    $allBenchmarks += $bdnContent.Benchmarks
                }
                # Copy original BDN files to results dir
                Copy-Item $_.FullName -Destination $ResultsDir -Force
            }

            # Also copy markdown reports
            Get-ChildItem -Path $artifactsDir -Filter "*-report-github.md" | ForEach-Object {
                Copy-Item $_.FullName -Destination $ResultsDir -Force
            }

            # Create consolidated numsharp-results.json
            if ($allBenchmarks.Count -gt 0) {
                $consolidatedResults = @{
                    Title = "NumSharp Benchmark Results"
                    Timestamp = $Timestamp
                    Benchmarks = $allBenchmarks
                }
                $consolidatedResults | ConvertTo-Json -Depth 10 | Out-File -FilePath $numsharpJsonPath -Encoding UTF8
                Write-Log "Consolidated $($allBenchmarks.Count) C# benchmark results to numsharp-results.json"
            }
        }

    } catch {
        Write-Warn "C# benchmarks failed: $_"
        Add-Content -Path $LogFile -Value "ERROR: C# benchmarks failed: $_"
    } finally {
        Pop-Location
    }
} else {
    Write-Log "Skipping C# benchmarks (-SkipCSharp)"
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

Ratio = NumSharp / NumPy | `u{2705} `u{2264}1x | `u{1F7E1} `u{2264}2x | `u{1F7E0} `u{2264}5x | `u{1F534} >5x

**See benchmark-report.md for the full comparison matrix.**

---

*Generated by run-benchmarks.ps1*
*Results: results/$TimestampFolder/*
"@

# Write report to results dir
$reportPath = Join-Path $ResultsDir "benchmark-report.md"
$report | Out-File -FilePath $reportPath -Encoding UTF8

Write-Success "Report generated: $reportPath"

# =============================================================================
# Generate Unified Comparison (if both results exist)
# =============================================================================

$mergeScript = Join-Path $ScriptDir "scripts\merge-results.py"

if ((Test-Path $numpyJsonPath) -and (Test-Path $mergeScript)) {
    Write-Status "Generating unified comparison..."

    $mergeOutputBase = Join-Path $ResultsDir "benchmark-report"
    $csharpArtifacts = Join-Path $ScriptDir "NumSharp.Benchmark.Core\BenchmarkDotNet.Artifacts\results"

    Write-Log "Merge command: python $mergeScript --numpy $numpyJsonPath --csharp $csharpArtifacts --output $mergeOutputBase"

    try {
        & python $mergeScript --numpy $numpyJsonPath --csharp $csharpArtifacts --output $mergeOutputBase 2>&1 | ForEach-Object {
            Write-Host "  $_"
            Add-Content -Path $LogFile -Value "  [Merge] $_"
        }
        Write-Success "Unified results generated"
    } catch {
        Write-Warn "Failed to generate unified results: $_"
        Add-Content -Path $LogFile -Value "ERROR: Merge failed: $_"
    }
}

# =============================================================================
# Copy Results to benchmark/ (override existing)
# =============================================================================

Write-Status "Copying results to benchmark/..."

# Copy main output files to benchmark/ root
$filesToCopy = @(
    "benchmark-report.md",
    "benchmark-report.json",
    "benchmark-report.csv",
    "numpy-results.json",
    "numsharp-results.json"
)

foreach ($file in $filesToCopy) {
    $sourcePath = Join-Path $ResultsDir $file
    $destPath = Join-Path $ScriptDir $file
    if (Test-Path $sourcePath) {
        Copy-Item -Path $sourcePath -Destination $destPath -Force
        Write-Log "Copied $file to benchmark/"
    }
}

# Update README.md if it exists
$readmePath = Join-Path $ScriptDir "README.md"
$mergedReportPath = Join-Path $ResultsDir "benchmark-report.md"
if ((Test-Path $readmePath) -and (Test-Path $mergedReportPath)) {
    Copy-Item -Path $mergedReportPath -Destination $readmePath -Force
    Write-Success "README.md updated with benchmark results"
}

# =============================================================================
# Final Summary
# =============================================================================

Write-Host ""
Write-Host "`u{2554}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2557}" -ForegroundColor Green
Write-Host "`u{2551}  Benchmark Complete!                                         `u{2551}" -ForegroundColor Green
Write-Host "`u{255A}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{255D}" -ForegroundColor Green
Write-Host ""
Write-Host "  Results:  " -NoNewline; Write-Host $ResultsDir -ForegroundColor Yellow
Write-Host "  Report:   " -NoNewline; Write-Host (Join-Path $ScriptDir "benchmark-report.md") -ForegroundColor Yellow
Write-Host "  README:   " -NoNewline; Write-Host $readmePath -ForegroundColor Yellow
Write-Host "  Log:      " -NoNewline; Write-Host $LogFile -ForegroundColor Yellow
Write-Host ""
Write-Host "  Files in results/$TimestampFolder/:" -ForegroundColor Cyan
Get-ChildItem -Path $ResultsDir | ForEach-Object { Write-Host "    - $($_.Name)" }
Write-Host ""
