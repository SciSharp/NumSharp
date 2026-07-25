using System.Runtime.CompilerServices;

// Same friend pattern as NumSharp.Core/Assembly/Properties.cs: the fuzz gate replays the
// host-pinned matmul_parity corpus through this engine, and NumSharp.DotNetRunScript lets ad-hoc
// `dotnet run` probes reach internals by overriding their AssemblyName.
[assembly: InternalsVisibleTo("NumSharp.UnitTest")]
[assembly: InternalsVisibleTo("NumSharp.DotNetRunScript")]
