
function Start-Benchmark {
    
    param (
        [Parameter(Position=0)]
        [ValidateSet('amin','Linq','NDArray')]
        [System.String]$Test
    )
    
    begin {
        $currentPath = $pwd;
        Set-Location -Path ./test/NumSharp.Benchmark/
    }
    
    process {
        dotnet run -c release $Test
    }
    
    end {
        Set-Location $currentPath;
    }
}
