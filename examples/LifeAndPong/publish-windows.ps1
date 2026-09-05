# Run from any working directory. Creates an unpack-and-play Windows x64 release.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$project = Join-Path $PSScriptRoot 'NumSharp.LifeAndPong.Desktop/NumSharp.LifeAndPong.Desktop.csproj'
$tests = Join-Path $PSScriptRoot 'NumSharp.LifeAndPong.Tests/NumSharp.LifeAndPong.Tests.csproj'
$releaseRoot = Join-Path $PSScriptRoot ('artifacts/release-' + [Guid]::NewGuid().ToString('N'))
$bundle = Join-Path $releaseRoot 'NumSharp-LifeAndPong-win-x64'
New-Item -ItemType Directory -Path $bundle | Out-Null
Push-Location -LiteralPath $PSScriptRoot
try {
    dotnet test $tests -c Release --nologo
    if ($LASTEXITCODE -ne 0) { throw 'Game tests failed; release stopped.' }
    dotnet publish $project -c Release -r win-x64 --self-contained true -p:PublishTrimmed=false --nologo -o $bundle
    if ($LASTEXITCODE -ne 0) { throw 'Windows publish failed; release stopped.' }
    # Ship the root guides together so links between player/contributor documents survive packaging.
    Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.md' -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $bundle
    }
    foreach ($name in @('preview.png', 'preview.ready.png')) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination $bundle
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'LICENSE') -Destination (Join-Path $bundle 'LICENSE')
    $archive = Join-Path $releaseRoot 'NumSharp-LifeAndPong-win-x64.zip'
    Compress-Archive -LiteralPath $bundle -DestinationPath $archive
    $checksum = Get-FileHash -LiteralPath $archive -Algorithm SHA256
    $checksumLine = $checksum.Hash.ToLowerInvariant() + '  ' + (Split-Path -Leaf $archive)
    Set-Content -LiteralPath (Join-Path $releaseRoot 'SHA256SUMS') -Value $checksumLine -Encoding ascii
    Write-Output $checksumLine
    Write-Output ('Play: ' + (Join-Path $bundle 'NumSharp.LifeAndPong.Desktop.exe'))
    Write-Output ('Share: ' + $archive)
}
finally {
    Pop-Location
}
