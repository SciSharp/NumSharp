using System;
using System.Threading;
using Microsoft.ML;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     Shared plumbing for the ML.NET interop tests.
    ///
    ///     <para><b>Leak gate:</b> every test captures the <see cref="NDArrayMLNetInterop.LiveExports"/> baseline on
    ///     entry, and the cleanup FAILS the test unless the counter returns to that baseline — so every single test
    ///     doubles as a no-leak / no-premature-free assertion. Dispose every <see cref="NDArrayDataView"/> (and every
    ///     ML.NET cursor) you make (<c>using</c>).</para>
    ///
    ///     <para><b>GC realism:</b> collection-dependent lifecycles must run inside
    ///     <c>[MethodImpl(MethodImplOptions.NoInlining)]</c> helpers — a debug-build JIT keeps untracked temps of a
    ///     frame's references alive until the method returns.</para>
    /// </summary>
    public abstract class MLNetTestBase
    {
        private int _baseExports;
        private int _baseImports;

        /// <summary>One shared, deterministic MLContext for the suite.</summary>
        protected static readonly MLContext Ml = new(seed: 0);

        [TestInitialize]
        public void InteropInit()
        {
            Settle();
            _baseExports = NDArrayMLNetInterop.LiveExports;
            _baseImports = NDArrayMLNetInterop.LiveImports;
        }

        [TestCleanup]
        public void InteropCleanup()
        {
            bool settled = WaitFor(() => NDArrayMLNetInterop.LiveExports <= _baseExports &&
                                         NDArrayMLNetInterop.LiveImports <= _baseImports, 10_000);
            Assert.IsTrue(settled,
                $"interop leaked: LiveExports {_baseExports} -> {NDArrayMLNetInterop.LiveExports}, " +
                $"LiveImports {_baseImports} -> {NDArrayMLNetInterop.LiveImports}");
        }

        // ---- data ------------------------------------------------------------------------------------

        /// <summary>The 11 dtypes that map to an ML.NET column type (Char rides UInt16; Half / Decimal are conversions; Complex is refused).</summary>
        protected static readonly NPTypeCode[] DirectDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
            NPTypeCode.Single, NPTypeCode.Double,
        };

        /// <summary>
        ///     A C-contiguous, OWNING <c>arange</c>-valued array of the dtype and shape (values small enough for every
        ///     dtype, incl. bool). Reshape first (a view of the ramp), THEN astype: the copy is what makes the result
        ///     own its buffer.
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

        /// <summary>Assert an array's shape equals <paramref name="dims"/>.</summary>
        protected static void AssertShape(NDArray nd, params long[] dims)
        {
            Assert.AreEqual(dims.Length, nd.ndim, $"ndim mismatch: {nd.Shape}");
            for (int i = 0; i < dims.Length; i++)
                Assert.AreEqual(dims[i], (long)nd.shape[i], $"dim {i} mismatch: {nd.Shape}");
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
