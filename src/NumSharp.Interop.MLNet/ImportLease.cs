using System;
using System.Threading;

namespace NumSharp.Interop.MLNet
{
    /// <summary>
    ///     One NumSharp-side lease on a pinned ML.NET buffer (a dense <see cref="Microsoft.ML.Data.VBuffer{T}"/>'s
    ///     backing <c>T[]</c>), wired as the <see cref="NumSharp.Backends.Unmanaged.UnmanagedMemoryBlock{T}"/> dispose
    ///     hook so it fires exactly once when the LAST NumSharp view over the memory (derived slices included) drops
    ///     its ARC reference — via <see cref="NDArray.Dispose"/>, the NDArray finalizer, or the memory-block Disposer
    ///     finalizer. Lock-free and safe from finalizer threads: it only frees a <see cref="System.Runtime.InteropServices.GCHandle"/>.
    ///     The same primitive the ONNX bridge leases ORT output buffers with.
    /// </summary>
    internal sealed class ImportLease
    {
        private Action _release;
        private readonly long _bytes;
        private int _released;

        internal ImportLease(Action release, long bytes)
        {
            _release = release;
            _bytes = bytes;
            if (_bytes > 0)
                GC.AddMemoryPressure(_bytes);
            NDArrayMLNetInterop.ImportOpened();
        }

        internal bool IsReleased => Volatile.Read(ref _released) != 0;

        internal void Release()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0)
                return;
            try
            {
                Action release = Interlocked.Exchange(ref _release, null);
                release?.Invoke();
            }
            finally
            {
                if (_bytes > 0)
                    GC.RemoveMemoryPressure(_bytes);
                NDArrayMLNetInterop.ImportClosed();
            }
        }
    }
}
