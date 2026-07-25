using System.Runtime.CompilerServices;
#if !SIGNING
[assembly: InternalsVisibleTo("NumSharp.UnitTest")]
// Optional out-of-box backends: they subclass DefaultEngine and so need the same internal view of
// Shape (strides/offset) and the promotion helpers that the built-in kernels have.
[assembly: InternalsVisibleTo("NumSharp.Interop.BLAS")]
[assembly: InternalsVisibleTo("NumSharp.Benchmark")]
[assembly: InternalsVisibleTo("TensorFlowNET.UnitTest")]
[assembly: InternalsVisibleTo("NumSharp.DotNetRunScript")]
#endif
