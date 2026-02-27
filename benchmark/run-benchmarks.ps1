<#
.SYNOPSIS
    Runs NumSharp and NumPy benchmarks and generates a consolidated Markdown report.

.DESCRIPTION
    This script executes both C# (BenchmarkDotNet) and Python (NumPy) benchmarks,
    collects the results, calculates performance ratios, and generates a comprehensive
    Markdown report for comparison.

    Results are archived in timestamped folders under results/ for historical tracking.

.PARAMETER Quick
    Run quick benchmarks (fewer iterations, faster but less accurate)

.PARAMETER Suite
    Specific suite to run. Standard suites: 'all', 'arithmetic', 'unary',
    'reduction', 'broadcast', 'creation', 'manipulation', 'slicing'
    Experimental suites (use -Experimental): 'dispatch', 'fusion'

.PARAMETER Experimental
    Run experimental/research benchmarks (dispatch, fusion) instead of NumPy comparison.
    These are internal research benchmarks that don't merge with NumPy results.

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
    .\run-benchmarks.ps1 -Experimental -Suite dispatch -SkipPython
#>

param(
    [switch]$Quick,
    [string]$Suite = 'all',
    [switch]$Experimental,
    [string]$OutputPath = 'benchmark-report.md',
    [switch]$SkipCSharp,
    [switch]$SkipPython,
    [string]$Type = '',
    [ValidateSet('', 'small', 'medium', 'large')]
    [string]$Size = ''
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ResultsDir = Join-Path $ScriptDir "results\$Timestamp"

# Standard suites (for NumPy comparison)
$StandardSuites = @('all', 'arithmetic', 'unary', 'reduction', 'broadcast', 'creation', 'manipulation', 'slicing')
# Experimental suites (research benchmarks)
$ExperimentalSuites = @('dispatch', 'fusion')

# Validate suite selection
if ($Experimental) {
    if ($Suite -notin @('all') + $ExperimentalSuites) {
        Write-Host "`n[ERROR] Invalid experimental suite: '$Suite'" -ForegroundColor Red
        Write-Host "Valid experimental suites: $($ExperimentalSuites -join ', ')" -ForegroundColor Yellow
        Write-Host "Use -Experimental -Suite dispatch|fusion" -ForegroundColor Yellow
        exit 1
    }
    if (-not $SkipPython) {
        Write-Host "`n[WARNING] Experimental benchmarks don't merge with NumPy results. Consider using -SkipPython." -ForegroundColor Yellow
    }
} else {
    if ($Suite -in $ExperimentalSuites) {
        Write-Host "`n[ERROR] Suite '$Suite' is an experimental suite." -ForegroundColor Red
        Write-Host "Valid standard suites: $($StandardSuites -join ', ')" -ForegroundColor Yellow
        Write-Host "Use -Experimental flag for experimental benchmarks: .\run-benchmarks.ps1 -Experimental -Suite $Suite" -ForegroundColor Cyan
        exit 1
    }
    if ($Suite -notin $StandardSuites) {
        Write-Host "`n[ERROR] Invalid suite: '$Suite'" -ForegroundColor Red
        Write-Host "Valid standard suites: $($StandardSuites -join ', ')" -ForegroundColor Yellow
        exit 1
    }
}

# Create results directory
New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null
$LogPath = Join-Path $ResultsDir "benchmark.log"

# Logging function
function Write-Log {
    param(
        [string]$Message,
        [string]$Prefix = "",
        [string]$ForegroundColor = "White"
    )
    $timestamp = Get-Date -Format "HH:mm:ss"
    $logLine = "[$timestamp]"
    if ($Prefix) { $logLine += " [$Prefix]" }
    $logLine += " $Message"

    # Write to console
    Write-Host $logLine -ForegroundColor $ForegroundColor

    # Append to log file
    $logLine | Out-File -FilePath $LogPath -Append -Encoding UTF8
}

# Colors for console output
function Write-Status { param($msg) Write-Log $msg -Prefix "INFO" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Log $msg -Prefix "OK" -ForegroundColor Green }
function Write-Warning { param($msg) Write-Log $msg -Prefix "WARN" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Log $msg -Prefix "ERROR" -ForegroundColor Red }

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
if ($Experimental) {
    Write-Host "`u{2551}       NumSharp Experimental Benchmark Suite                  `u{2551}" -ForegroundColor Magenta
} else {
    Write-Host "`u{2551}       NumSharp vs NumPy Comprehensive Benchmark Suite       `u{2551}" -ForegroundColor Blue
}
Write-Host "`u{255A}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{255D}" -ForegroundColor Blue
Write-Host ""

Write-Status "Results will be saved to: $ResultsDir"
Write-Log "Benchmark started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "Mode: $(if ($Experimental) { 'Experimental' } else { 'Standard' }), Suite: $Suite, Quick: $Quick"

# Initialize report
$reportTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$report = @"
# NumSharp Performance Benchmark Report

**Generated:** $reportTimestamp
**Mode:** $(if ($Quick) { 'Quick (reduced iterations)' } else { 'Full' })
**Suite:** $Suite $(if ($Experimental) { '(Experimental)' } else { '' })

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
$numpyJsonPath = Join-Path $ResultsDir "numpy-results.json"

if (-not $SkipPython) {
    Write-Status "Running Python/NumPy benchmarks..."

    $pythonScript = Join-Path $ScriptDir "NumSharp.Benchmark.Python\numpy_benchmark.py"
    $pythonArgs = @($pythonScript, "--output", $numpyJsonPath)
    if ($Quick) { $pythonArgs += "--quick" }
    if ($Suite -ne 'all') { $pythonArgs += "--suite"; $pythonArgs += $Suite }
    if ($Type) { $pythonArgs += "--type"; $pythonArgs += $Type }
    if ($Size) { $pythonArgs += "--size"; $pythonArgs += $Size }

    $pythonCmd = "python " + ($pythonArgs -join " ")
    Write-Log "Command: $pythonCmd"

    try {
        & python @pythonArgs 2>&1 | ForEach-Object {
            Write-Log $_ -Prefix "Python"
        }

        if (Test-Path $numpyJsonPath) {
            $pythonResults = Get-Content $numpyJsonPath | ConvertFrom-Json
            Write-Success "Python benchmarks complete ($($pythonResults.Count) results)"
        }
    } catch {
        Write-Warning "Python benchmarks failed: $_"
    }
} else {
    Write-Status "Skipping Python benchmarks (-SkipPython)"
    Write-Log "Skipping Python benchmarks"
}

# =============================================================================
# Run C# Benchmarks
# =============================================================================

$csharpResults = @{}
$csharpJsonDir = $null
$numsharpJsonPath = Join-Path $ResultsDir "numsharp-results.json"

if (-not $SkipCSharp) {
    Write-Status "Building C# benchmarks..."

    $csharpDir = Join-Path $ScriptDir "NumSharp.Benchmark.GraphEngine"
    Push-Location $csharpDir

    try {
        & dotnet build -c Release -v q --nologo 2>$null | Out-Null
        Write-Success "Build complete"

        Write-Status "Running C# benchmarks (this may take a few minutes)..."

        $jobType = if ($Quick) { "Short" } else { "Medium" }

        # Build filter based on suite and experimental flag
        if ($Experimental) {
            # Experimental benchmarks (dispatch, fusion) are research benchmarks
            # that don't correspond to NumPy operations
            $filter = switch ($Suite) {
                'dispatch' { "*DispatchBenchmarks*" }
                'fusion' { "*FusionBenchmarks*" }
                default { "*DispatchBenchmarks*,*FusionBenchmarks*" }
            }
        } else {
            $filter = switch ($Suite) {
                'arithmetic' { "*Arithmetic*" }
                'unary' { "*Unary*,*Math*,*ExpLog*,*Trig*,*Power*" }
                'reduction' { "*Reduction*,*Sum*,*Mean*,*VarStd*,*MinMax*,*Prod*" }
                'broadcast' { "*Broadcast*" }
                'creation' { "*Creation*" }
                'manipulation' { "*Manipulation*,*Reshape*,*Stack*,*Dims*" }
                'slicing' { "*Slice*" }
                default { "*" }
            }
        }

        # Map -Type parameter to C# DType filter
        # NumPy dtype names -> C# NPTypeCode names
        # BDN filter format: namespace.class.method(N: 10000000, DType: Int32)
        $dtypeFilter = ""
        if ($Type) {
            $dtypeMap = @{
                'int32' = 'Int32'
                'int64' = 'Int64'
                'float32' = 'Single'
                'float64' = 'Double'
                'uint8' = 'Byte'
                'int16' = 'Int16'
                'uint16' = 'UInt16'
                'uint32' = 'UInt32'
                'uint64' = 'UInt64'
                'bool' = 'Boolean'
            }
            $csharpType = $dtypeMap[$Type.ToLower()]
            if ($csharpType) {
                $dtypeFilter = "*DType: $csharpType*"
                Write-Log "Filtering C# benchmarks to DType=$csharpType"
            } else {
                Write-Warning "Unknown type '$Type' - running all types"
            }
        }

        # In Quick mode, only run N=10M (matches NumPy comparison)
        # This dramatically reduces benchmark count
        $sizeFilter = ""
        if ($Quick) {
            $sizeFilter = "*N: 10000000*"
            Write-Log "Quick mode: filtering to N=10000000 only"
        }

        # Combine filters: suite AND size AND dtype
        # BDN parameter order is (N: value, DType: type) so size filter must come before dtype
        # BDN uses glob patterns on full name: namespace.class.method(params)
        # Multiple wildcards in same pattern act as AND
        $filterParts = @()
        foreach ($suitePart in $filter.Split(',')) {
            $combined = $suitePart.Trim()
            if ($sizeFilter) { $combined += $sizeFilter }   # N: comes first in BDN output
            if ($dtypeFilter) { $combined += $dtypeFilter } # DType: comes second
            $filterParts += $combined
        }
        $filter = $filterParts -join ","

        $csharpCmd = "dotnet run -c Release --no-build -f net10.0 -- --job $jobType --filter $filter --exporters json"
        Write-Log "Command: $csharpCmd"

        # Run and capture output
        $output = & dotnet run -c Release --no-build -f net10.0 -- --job $jobType --filter $filter --exporters json 2>&1

        # Log all output
        foreach ($line in $output) {
            Write-Log $line -Prefix "C#"
        }

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

        # Find JSON results directory and copy artifacts
        $artifactsDir = Join-Path $csharpDir "BenchmarkDotNet.Artifacts\results"
        if (Test-Path $artifactsDir) {
            $csharpJsonDir = $artifactsDir

            # Copy BDN artifacts to results folder
            $bdnFiles = Get-ChildItem $artifactsDir -Filter "*.json" -ErrorAction SilentlyContinue
            foreach ($file in $bdnFiles) {
                Copy-Item $file.FullName -Destination $ResultsDir -Force
                Write-Log "Copied BDN artifact: $($file.Name)"
            }
            $mdFiles = Get-ChildItem $artifactsDir -Filter "*.md" -ErrorAction SilentlyContinue
            foreach ($file in $mdFiles) {
                Copy-Item $file.FullName -Destination $ResultsDir -Force
                Write-Log "Copied BDN artifact: $($file.Name)"
            }

            # Create consolidated numsharp-results.json from BDN results
            # (merge-results.py will read from BDN artifacts dir)
        }

    } catch {
        Write-Warning "C# benchmarks failed: $_"
    } finally {
        Pop-Location
    }
} else {
    Write-Status "Skipping C# benchmarks (-SkipCSharp)"
    Write-Log "Skipping C# benchmarks"
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

Ratio = NumSharp / NumPy | `u{2705} ≤1x | `u{1F7E1} ≤2x | `u{1F7E0} ≤5x | `u{1F534} >5x

**See benchmark-report.md for the full comparison matrix.**

---

*Generated by run-benchmarks.ps1*
"@

# Write report to results folder
$reportPath = Join-Path $ResultsDir "benchmark-report.md"
$report | Out-File -FilePath $reportPath -Encoding UTF8
Write-Log "Report written to: $reportPath"

# =============================================================================
# Generate Unified Comparison (if both results exist and not Experimental)
# =============================================================================

$mergeScript = Join-Path $ScriptDir "scripts\merge-results.py"
$mergedJsonPath = Join-Path $ResultsDir "benchmark-report.json"
$mergedCsvPath = Join-Path $ResultsDir "benchmark-report.csv"

if (-not $Experimental -and (Test-Path $numpyJsonPath) -and (Test-Path $mergeScript) -and $csharpJsonDir) {
    Write-Status "Generating unified comparison..."
    Write-Log "Running merge script..."

    $outputBase = Join-Path $ResultsDir "benchmark-report"
    $mergeCmd = "python $mergeScript --numpy $numpyJsonPath --csharp $csharpJsonDir --output $outputBase"
    Write-Log "Command: $mergeCmd"

    try {
        & python $mergeScript --numpy $numpyJsonPath --csharp $csharpJsonDir --output $outputBase 2>&1 | ForEach-Object {
            Write-Log $_ -Prefix "Merge"
        }
        Write-Success "Unified results generated"
    } catch {
        Write-Warning "Failed to generate unified results: $_"
    }
} elseif ($Experimental) {
    Write-Log "Skipping merge (Experimental mode - no NumPy comparison)"
} elseif (-not (Test-Path $numpyJsonPath)) {
    Write-Log "Skipping merge (no numpy-results.json)"
}

# =============================================================================
# Copy results to benchmark/ root
# =============================================================================

Write-Status "Copying results to benchmark root..."

$filesToCopy = @(
    "benchmark.log",
    "benchmark-report.md",
    "benchmark-report.json",
    "benchmark-report.csv",
    "numpy-results.json",
    "numsharp-results.json"
)

foreach ($fileName in $filesToCopy) {
    $sourcePath = Join-Path $ResultsDir $fileName
    if (Test-Path $sourcePath) {
        $destPath = Join-Path $ScriptDir $fileName
        Copy-Item $sourcePath -Destination $destPath -Force
        Write-Log "Copied to root: $fileName"
    }
}

# =============================================================================
# Copy report to README.md (only if README.md already exists)
# =============================================================================

$readmePath = Join-Path $ScriptDir "README.md"
$mergedReportPath = Join-Path $ResultsDir "benchmark-report.md"

# Prefer merged report if available
if (Test-Path (Join-Path $ResultsDir "benchmark-report.md")) {
    $reportToCopy = Join-Path $ResultsDir "benchmark-report.md"
} else {
    $reportToCopy = $reportPath
}

if (Test-Path $readmePath) {
    Copy-Item -Path $reportToCopy -Destination $readmePath -Force
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
Write-Host "  Results:  " -NoNewline; Write-Host $ResultsDir -ForegroundColor Yellow
Write-Host "  Report:   " -NoNewline; Write-Host (Join-Path $ScriptDir "benchmark-report.md") -ForegroundColor Yellow
Write-Host "  README:   " -NoNewline; Write-Host $readmePath -ForegroundColor Yellow
Write-Host "  Log:      " -NoNewline; Write-Host $LogPath -ForegroundColor Yellow
if (Test-Path $numpyJsonPath) {
    Write-Host "  NumPy:    " -NoNewline; Write-Host $numpyJsonPath -ForegroundColor Yellow
}
if (Test-Path $mergedJsonPath) {
    Write-Host "  Merged:   " -NoNewline; Write-Host $mergedJsonPath -ForegroundColor Yellow
}
Write-Host "  View:     " -NoNewline; Write-Host "code $ResultsDir" -ForegroundColor Cyan
Write-Host ""
