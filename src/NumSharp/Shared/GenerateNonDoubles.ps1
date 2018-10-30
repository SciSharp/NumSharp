
$searchPattern = ($PSScriptRoot + '/Operation*Double.cs');
$currentScript = Get-ChildItem ($PSScriptRoot + '/GenerateNonDoubles.ps1');

[System.IO.FileInfo[]]$operationFiles = Get-ChildItem $searchPattern ;

#[System.IO.FileInfo[]]$allFiles = Get-ChildItem ($PSScriptRoot);

[System.IO.FileInfo[]]$filesToRemove = Get-ChildItem ($PSScriptRoot) | ? { !$_.Name.Contains('Double')  } | ? {!$_.Name.Equals($currentScript.Name)}

$filesToRemove | % {rm $_.FullName}


$dataTypes =   @('float','Complex','Int32','Int64','decimal','Quaternion');

for($idx = 0; $idx -lt $operationFiles.Length;$idx++)
{
    $contentInFile = Get-Content $operationFiles[$idx];

    # find indexes of start and end 
    $noOfStarts = $contentInFile -cmatch '//start' | % { $contentInFile.IndexOf($_) };
    $noOfEnds = $contentInFile -cmatch '//end' | % { $contentInFile.IndexOf($_) };

    for($jdx = 0;$jdx -lt $dataTypes.Length;$jdx++)
    {
        $newFile = New-Item -ItemType File ($operationFiles[$idx].Directory.FullName + '/' + $operationFiles[$idx].Name.Replace('Double',$dataTypes[$jdx]));

        'using System;' >> $newFile.FullName;
        'using System.Collections.Generic;' >> $newFile.FullName;
        'using System.Numerics;'  >> $newFile.FullName;
        'using System.Linq;'  >> $newFile.FullName;
        'using System.Text;'  >> $newFile.FullName;
        '' >> $newFile.FullName;
        'namespace NumSharp.Shared' >> $newFile.FullName;
        '{' >> $newFile.FullName;
        ('   internal static partial class ' + [string]($newFile.Name.Split('.')[1])) >> $newFile.FullName;
        '   {' >> $newFile.FullName;

        for($kdx = 0; $kdx -lt $noOfEnds.Length;$kdx++)
        {
            $method = $contentInFile[$noOfStarts[$kdx]..$noOfEnds[$kdx]];

            $method = $method.Replace('double',$dataTypes[$jdx]).Replace('Double',$dataTypes[$jdx]);
            
            for($ldx = 0; $ldx -lt $method.Length;$ldx++)
            {
                $method[$ldx] >> $newFile.FullName;
            } 
        }
        '   }' >> $newFile.FullName;
        '}' >> $newFile.FullName;
    }
}