# Build and test the focused solution from any working directory.
[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    dotnet restore NumSharp.LifeAndPong.sln --nologo
    if ($LASTEXITCODE -ne 0) { throw 'Solution restore failed.' }

    dotnet build NumSharp.LifeAndPong.sln -c $Configuration --no-restore --nologo
    if ($LASTEXITCODE -ne 0) { throw 'Solution build failed.' }

    dotnet test NumSharp.LifeAndPong.Tests/NumSharp.LifeAndPong.Tests.csproj -c $Configuration --no-build --no-restore --nologo
    if ($LASTEXITCODE -ne 0) { throw 'Game tests failed.' }
}
finally {
    Pop-Location
}
