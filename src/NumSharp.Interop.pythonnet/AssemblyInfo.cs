using System.Runtime.CompilerServices;

// Same friend pattern as NumSharp.Core/Assembly/Properties.cs: the interop unit tests assert on
// internal seams (e.g. the AcquireGil policy factory), and NumSharp.DotNetRunScript lets ad-hoc
// `dotnet run` probes reach internals by overriding their AssemblyName.
[assembly: InternalsVisibleTo("NumSharp.Interop.UnitTests")]
[assembly: InternalsVisibleTo("NumSharp.DotNetRunScript")]
