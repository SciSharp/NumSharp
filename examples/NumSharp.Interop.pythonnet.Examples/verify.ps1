<#
.SYNOPSIS
    Runs every example script and fails on the first non-zero exit code.

.DESCRIPTION
    Each *.cs file here is a self-contained .NET 10 file-based app that prints one OK/FAIL line per
    claim and exits non-zero when any claim fails, so this is a gate as well as a tour. Needs the
    .NET 10 SDK and a `python` on PATH with numpy (see requirements.txt); scripts that need
    torch/pandas/pyarrow/pillow self-skip (exit 0) when the package is missing.

.PARAMETER Examples
    Base names to run (e.g. 02-four-verbs). Default: all, in numeric order.

.PARAMETER Configuration
    Build configuration passed to `dotnet run` (default Release — NumSharp's kernels are ~2x slower
    in Debug and the in-repo build usually has Release outputs already).

.EXAMPLE
    ./verify.ps1
    ./verify.ps1 -Examples 06-lifetime, 09-pytorch
    PYTHONNET_PYDLL=/usr/lib/libpython3.12.so ./verify.ps1
#>
param(
    [string[]] $Examples = @(),
    [string] $Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'
$sources = @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.cs' | Sort-Object Name)
if ($Examples.Count -gt 0) {
    $sources = @($sources | Where-Object { $_.BaseName -in $Examples })
}
if ($sources.Count -eq 0) { throw 'No example selected.' }

$failed = @()
$stopwatch = [Diagnostics.Stopwatch]::StartNew()
foreach ($source in $sources) {
    Write-Host ""
    Write-Host ("=" * 100)
    Write-Host " $($source.Name)"
    Write-Host ("=" * 100)
    Push-Location -LiteralPath $PSScriptRoot
    try {
        & dotnet run -c $Configuration -v quiet $source.Name
        if ($LASTEXITCODE -ne 0) { $failed += $source.Name; Write-Host "  -> exit code $LASTEXITCODE" -ForegroundColor Red }
    }
    finally { Pop-Location }
}

Write-Host ""
Write-Host ("-" * 100)
if ($failed.Count -eq 0) {
    Write-Host " $($sources.Count) example(s) passed in $([int]$stopwatch.Elapsed.TotalSeconds) s" -ForegroundColor Green
    exit 0
}
Write-Host " FAILED: $($failed -join ', ')" -ForegroundColor Red
exit 1
