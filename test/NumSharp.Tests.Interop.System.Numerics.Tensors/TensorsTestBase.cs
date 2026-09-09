using System;
using System.Threading;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     Shared plumbing for the System.Numerics.Tensors interop tests, mirroring the ONNX suite's
    ///     <c>OnnxTestBase</c>.
    ///
    ///     <para><b>Leak gate:</b> every test captures the <see cref="NDArrayTensorsInterop.LiveExports"/> /
    ///     <see cref="NDArrayTensorsInterop.LiveImports"/> baseline on entry, and the cleanup FAILS the test
    ///     unless the counters return to that baseline — so every single test doubles as a no-leak /
    ///     no-premature-free assertion. Dispose every handle and view you make (<c>using</c>).</para>
    ///
    ///     <para><b>GC realism:</b> collection-dependent lifecycles must run inside
    ///     <c>[MethodImpl(MethodImplOptions.NoInlining)]</c> helpers — a debug-build JIT keeps untracked temps
    ///     of a frame's references alive until the method returns.</para>
    /// </summary>
    public abstract class TensorsTestBase
    {
        private int _baseExports, _baseImports;

        [TestInitialize]
        public void InteropInit()
        {
            Settle();
            _baseExports = NDArrayTensorsInterop.LiveExports;
            _baseImports = NDArrayTensorsInterop.LiveImports;
        }

        [TestCleanup]
        public void InteropCleanup()
        {
            bool settled = WaitFor(() => NDArrayTensorsInterop.LiveExports <= _baseExports &&
                                         NDArrayTensorsInterop.LiveImports <= _baseImports, 10_000);
            Assert.IsTrue(settled,
                $"interop leaked conversions: LiveExports {_baseExports} -> {NDArrayTensorsInterop.LiveExports}, " +
                $"LiveImports {_baseImports} -> {NDArrayTensorsInterop.LiveImports}");
        }

        // ---- dtype axes ------------------------------------------------------------------------------

        /// <summary>All 15 NumSharp dtypes — every one crosses zero-copy as its own CLR type (nothing refused).</summary>
        protected static readonly NPTypeCode[] AllDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
        };

        // ---- data ------------------------------------------------------------------------------------

        /// <summary>
        ///     A C-contiguous, OWNING <c>arange</c>-valued array of the dtype and shape (values small enough for
        ///     every dtype). Reshape first (a view of the ramp), THEN astype: the copy is what makes the result own
        ///     its buffer (owndata == true).
        /// </summary>
        protected static NDArray Arange(NPTypeCode code, params int[] shape)
        {
            long size = 1;
            foreach (int d in shape)
                size *= d;
            using NDArray ramp = np.arange((int)size);
            using NDArray shaped = shape.Length == 0 ? null : ramp.reshape(shape);
            return (shaped ?? ramp).astype(code, copy: true);
        }

        /// <summary>Raw bytes of an array's logical C-order contents (independent of the interop: NumSharp's own copy + a raw read).</summary>
        protected static unsafe byte[] BytesOf(NDArray nd)
        {
            using NDArray dense = nd.Shape.IsContiguous ? null : nd.copy();
            NDArray src = dense ?? nd;
            long nbytes = src.size * src.dtypesize;
            var bytes = new byte[nbytes];
            if (nbytes > 0)
                new ReadOnlySpan<byte>((byte*)src.Storage.InternalArray.Address + src.Shape.Offset * src.dtypesize, (int)nbytes).CopyTo(bytes);
            return bytes;
        }

        // ---- GC helpers ------------------------------------------------------------------------------

        protected static void Settle()
        {
            for (int i = 0; i < 3; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        protected static bool WaitFor(Func<bool> condition, int timeoutMs)
        {
            var deadline = DateTime.UtcNow.AddMilliseconds(timeoutMs);
            while (!condition())
            {
                if (DateTime.UtcNow > deadline)
                    return false;
                Settle();
                Thread.Sleep(20);
            }
            return true;
        }
    }
}
