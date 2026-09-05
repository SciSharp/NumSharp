# Run from any working directory. Creates an unpack-and-play Windows x64 release.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$project = Join-Path $PSScriptRoot 'NumSharp.LifeAndPong.Desktop/NumSharp.LifeAndPong.Desktop.csproj'
$tests = Join-Path $PSScriptRoot 'NumSharp.LifeAndPong.Tests/NumSharp.LifeAndPong.Tests.csproj'
$releaseRoot = Join-Path $PSScriptRoot ('artifacts/release-' + [Guid]::NewGuid().ToString('N'))
$bundle = Join-Path $releaseRoot 'NumSharp-LifeAndPong-win-x64'
New-Item -ItemType Directory -Path $bundle | Out-Null
dotnet test $tests -c Release --nologo
if ($LASTEXITCODE -ne 0) { throw 'Game tests failed; release stopped.' }
dotnet publish $project -c Release -r win-x64 --self-contained true -p:PublishTrimmed=false --nologo -o $bundle
if ($LASTEXITCODE -ne 0) { throw 'Windows publish failed; release stopped.' }
foreach ($name in @('README.md', 'SPECIFICATION.md', 'ARCADE_DESIGN.md', 'PHYSICS.md', 'preview.png', 'preview.ready.png')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination $bundle
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../../LICENSE') -Destination (Join-Path $bundle 'NumSharp-LICENSE.txt')
$archive = Join-Path $releaseRoot 'NumSharp-LifeAndPong-win-x64.zip'
Compress-Archive -LiteralPath $bundle -DestinationPath $archive
Get-FileHash -LiteralPath $archive -Algorithm SHA256
Write-Output ('Play: ' + (Join-Path $bundle 'NumSharp.LifeAndPong.Desktop.exe'))
Write-Output ('Share: ' + $archive)
