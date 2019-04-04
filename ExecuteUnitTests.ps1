Write-Output "Starting unit Tests "
Write-Output "--------------------"
Write-Output "--------------------"
Write-Output "Start with building"
Write-Output "--------------------"

$projectFolders = @{};

$projectFolders['projectRoot'] = $PSScriptRoot;
$projectFolders['UnitTest'] = Join-Path $projectFolders['projectRoot'] test/NumSharp.UnitTest;
$projectFolders['UnitTest.Bin'] = Join-Path $projectFolders['UnitTest'] bin
$projectFolders['UnitTest.CSPROJ'] = Join-Path $projectFolders['UnitTest'] NumSharp.UnitTest.csproj

# clear all items before testing
if (Test-Path -Path ($projectFolders['UnitTest.Bin'] ) )
{
    Write-Output "bin folder found - remove items";
    $puffer = Get-ChildItem -Path $projectFolders['UnitTest.Bin'] -Recurse;

    for($idx = 0;$idx -lt $puffer.Length;$idx++)
    {
        Write-Output ("    - " + $puffer[$idx] );
    }

    Remove-Item -Path $projectFolders['UnitTest.Bin'] -Recurse
}

dotnet build ./src/NumSharp.Core/NumSharp.Core.csproj
dotnet build ./test/NumSharp.UnitTest/NumSharp.UnitTest.csproj

$generatedDlls = Get-ChildItem -Recurse -File -Path $projectFolders['UnitTest.Bin']

Write-Output "following items exist:"

for($idx =0; $idx -lt $generatedDlls.Length;$idx++)
{
    Write-Output ("   - " + $generatedDlls[$idx].Name)
} 

dotnet test $projectFolders['UnitTest.CSPROJ']