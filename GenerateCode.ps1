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

Write-Host ("location of T4 tool is : " + (Get-Command t4).Path)