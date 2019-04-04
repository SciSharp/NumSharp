# first check if T4 engine is installed
if ( -not ( Get-Command t4 -errorAction SilentlyContinue))
{
    Write-Host "T4 tool was not found - will be installed."
    dotnet tool install -g dotnet-t4
    Write-Host "T4 has been installed."
}
else
{
    Write-Host "T4 tool is already installed."
}

Write-Host ("location of T4 tool is : " + (Get-Command t4).Path);
Write-Host "----------------------------------------------------"
Write-Host "----------------------------------------------------"
Write-Host ("Test T4 before using it.");

# set variables
$projectFolders = @{};
$projectFolders['projectRoot'] = $PSScriptRoot;
$projectFolders['tmp'] = $projectFolders['projectRoot'] + '/tmp_' + [System.Guid]::NewGuid();
$projectFolders['tt.elementwise'] = $projectFolders['projectRoot'] + "/src/NumSharp.Core/Operations/NdArray.ElementsWise.tt";

Write-Host ("new tmp folder at " + $projectFolders["tmp"])

$projectFolders['tmp.tt'] = $projectFolders['tmp'] + '/test.tt';
$projectFolders['tmp.html'] = $projectFolders['tmp'] + '/date.html';

New-Item -ItemType Directory -Path $projectFolders['tmp'];

'<html><body>' > $projectFolders['tmp.tt'];
'The date and time now is: <#= DateTime.Now #>' >> $projectFolders['tmp.tt'];
'</body></html>' >> $projectFolders['tmp.tt'];

Write-Host "";
Write-Host ("folder " + $projectFolders["tmp"]  + " was created.");
Write-Host "";

t4 -o $projectFolders['tmp.html'] $projectFolders['tmp.tt']

if (Test-Path -Path ($projectFolders['tmp.html']))
{
    Write-Host "html doc exist - was generated from tt."
    Write-Host "Everything should be fine..."
}

Write-Host ""
Write-Host "Tidy up now."
Write-Host ""

Remove-Item -Recurse -Path $projectFolders["tmp"]

Write-Host "Start true Code-Generation.";
Write-Host "";
Write-Host "Generate element wise operations + , - , * , /";
Write-Host "";

$supportDataType = New-Object 'System.Collections.Generic.List[System.String]';

$supportDataType.Add('System.Int32');
$supportDataType.Add('System.Int64');
$supportDataType.Add('System.Single');
$supportDataType.Add('System.Double');
$supportDataType.Add('System.Numerics.Complex');
$supportDataType.Add('System.Numerics.Quaternion');


$operationTypeString = [System.String]::Join(';',$supportDataType);

$command = "t4 -o - -p:operationName='+' -p:operationTypesString='" + $operationTypeString + "' " + $projectFolders['tt.elementwise'] + " > " + $projectFolders['projectRoot'] + "\src\NumSharp.Core\Operations\Elementwise\NdArray.Addition.cs";
Write-Host ("execute - " + $command);
Invoke-Expression $command;

$command = "t4 -o - -p:operationName='*' -p:operationTypesString='" + $operationTypeString + "' " + $projectFolders['tt.elementwise'] + " > " + $projectFolders['projectRoot'] + "\src\NumSharp.Core\Operations\Elementwise\NdArray.Multiplication.cs";
Write-Host ("execute - " + $command);
Invoke-Expression $command;

$command = "t4 -o - -p:operationName='/' -p:operationTypesString='" + $operationTypeString + "' " + $projectFolders['tt.elementwise'] + " > " + $projectFolders['projectRoot'] + "\src\NumSharp.Core\Operations\Elementwise\NdArray.Division.cs";
Write-Host ("execute - " + $command);
Invoke-Expression $command;

$command = "t4 -o - -p:operationName='-' -p:operationTypesString='" + $operationTypeString + "' " + $projectFolders['tt.elementwise'] + " > " + $projectFolders['projectRoot'] + "\src\NumSharp.Core\Operations\Elementwise\NdArray.Substraction.cs";
Write-Host ("execute - " + $command);
Invoke-Expression $command;
