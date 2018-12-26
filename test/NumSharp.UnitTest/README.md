We use OpenCover to improve our code review workflow and quality. https://github.com/codecov/example-csharp

Install `OpenCover` from https://www.nuget.org/packages/opencover.
Add `.nuget\packages\opencover\4.6.519\tools` to system bin Path

Install `Codecov` from https://www.nuget.org/packages/Codecov.
Add `.nuget\packages\codecov\1.1.0\tools` to system bin Path

`PS D:\Projects\NumSharp> OpenCover.Console -register:user -target:"dotnet.exe" -targetargs:test -filter:"+[NumSharp.UnitTest]*" -output:"coverage.xml" -oldstyle`
`PS D:\Projects\NumSharp> codecov -f coverage.xml -t <codecov-numsharp-token>`