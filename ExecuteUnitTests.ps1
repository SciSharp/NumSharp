Write-Output "Starting unit Tests "
Write-Output "--------------------"
Write-Output "--------------------"
Write-Output "Start with building"
Write-Output "--------------------"

Remove-Item -Path .\test\NumSharp.UnitTest\bin -Recurse

dotnet build ./src/NumSharp.Core/NumSharp.Core.csproj
dotnet build ./test/NumSharp.UnitTest/NumSharp.UnitTest.csproj

$generatedDlls = Get-ChildItem -Recurse -File -Path .\test\NumSharp.UnitTest\bin

Write-Output "following items exist:"

for($idx =0; $idx -lt $generatedDlls.Length;$idx++)
{
    Write-Output ("   - " + $generatedDlls[$idx].Name)
} 

dotnet test ./test/NumSharp.UnitTest/NumSharp.UnitTest.csproj