Write-Output "--------------------"
Write-Output "Start with building"
Write-Output "--------------------"

$projectFolders = @{};

$projectFolders['projectRoot'] = $PSScriptRoot;
$projectFolders['Projects.Core'] = Join-Path $projectFolders['projectRoot'] src/NumSharp.Core/NumSharp.Core.csproj

dotnet build $projectFolders['Projects.Core']