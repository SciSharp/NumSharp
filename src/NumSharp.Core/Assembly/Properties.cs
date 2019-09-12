using System.Runtime.CompilerServices;
#if !SIGNING
[assembly: InternalsVisibleTo("NumSharp.UnitTest")]
[assembly: InternalsVisibleTo("NumSharp.Benchmark")]
[assembly: InternalsVisibleTo("TensorFlowNET.UnitTest")]
#endif
