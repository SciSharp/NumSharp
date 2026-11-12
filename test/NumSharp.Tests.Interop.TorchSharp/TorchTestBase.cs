using System;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     Shared plumbing for the TorchSharp interop tests.
    /// </summary>
    /// <remarks>
    ///     <para><b>Leak gate.</b> Every test captures <see cref="NDArrayTorchInterop.LiveExports"/> /
    ///     <see cref="NDArrayTorchInterop.LiveImports"/> on entry, and the cleanup FAILS the test unless both return to that
    ///     baseline: each test is also a no-leak / no-premature-free assertion. Dispose every tensor and view you make, torch
    ///     views of exported tensors included (an indexing temporary such as <c>t[0, 0]</c> keeps the exported buffer until it is
    ///     disposed or finalized) - <see cref="Scope"/> disposes them for you.</para>
    ///     <para><b>GC realism.</b> Collection-dependent lifecycles run inside <c>[MethodImpl(MethodImplOptions.NoInlining)]</c>
    ///     helpers: a frame keeps untracked temporaries of its references alive until it returns.</para>
    /// </remarks>
    public abstract class TorchTestBase
    {
        /// <summary>The export baseline captured before the test.</summary>
        private int _baseExports;

        /// <summary>The import baseline captured before the test.</summary>
        private int _baseImports;

        /// <summary>Settles the GC and captures the counters' baseline.</summary>
        [TestInitialize]
        public void InteropInit()
        {
            Settle();
            _baseExports = NDArrayTorchInterop.LiveExports;
            _baseImports = NDArrayTorchInterop.LiveImports;
        }

        /// <summary>Fails the test unless both counters return to the baseline within 10 s of collections.</summary>
        [TestCleanup]
        public void InteropCleanup()
        {
            bool settled = WaitFor(() => NDArrayTorchInterop.LiveExports <= _baseExports &&
                                         NDArrayTorchInterop.LiveImports <= _baseImports, 10_000);
            Assert.IsTrue(settled,
                $"interop leaked: LiveExports {_baseExports} -> {NDArrayTorchInterop.LiveExports}, " +
                $"LiveImports {_baseImports} -> {NDArrayTorchInterop.LiveImports}");
        }

        // ---- dtype axes ------------------------------------------------------------------------------

        /// <summary>The ten NumSharp dtypes torch shares bit for bit.</summary>
        protected static readonly NPTypeCode[] SharedDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.Int32,
            NPTypeCode.Int64, NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex,
        };

        /// <summary>The five NumSharp dtypes torch lacks (they cross by NEP 50, decimal as float64).</summary>
        protected static readonly NPTypeCode[] ConvertedDtypes =
        {
            NPTypeCode.UInt16, NPTypeCode.UInt32, NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Decimal,
        };

        // ---- data ------------------------------------------------------------------------------------

        /// <summary>
        ///     A C-contiguous, owning array of the dtype and shape holding 0, 1, 2, ... (small enough for every dtype; a bool array
        ///     alternates). Reshape first, then astype: the copy makes the result own its buffer.
        /// </summary>
        /// <param name="code">The dtype.</param>
        /// <param name="shape">The dimensions.</param>
        /// <returns>The array.</returns>
        protected static NDArray Ramp(NPTypeCode code, params int[] shape)
        {
            long size = 1;
            foreach (int d in shape)
                size *= d;
            using NDArray ramp = code == NPTypeCode.Boolean ? np.arange((int)size) % 2 : np.arange((int)size);
            using NDArray shaped = shape.Length == 0 ? ramp.reshape(new Shape()) : ramp.reshape(shape);
            return shaped.astype(code, copy: true);
        }

        /// <summary>The raw bytes of an array's logical C-order contents, read through NumSharp's own copy (no interop).</summary>
        /// <param name="nd">The array.</param>
        /// <returns>The bytes.</returns>
        protected static unsafe byte[] BytesOf(NDArray nd)
        {
            using NDArray dense = nd.copy();
            long nbytes = dense.size * dense.dtypesize;
            var bytes = new byte[nbytes];
            if (nbytes > 0)
                new ReadOnlySpan<byte>((byte*)dense.Storage.InternalArray.Address + dense.Shape.offset * dense.dtypesize, (int)nbytes).CopyTo(bytes);
            return bytes;
        }

        /// <summary>The raw bytes of a tensor's logical C-order contents, read through torch's own copy (no interop).</summary>
        /// <param name="t">The tensor (CPU).</param>
        /// <returns>The bytes.</returns>
        protected static byte[] BytesOf(Tensor t)
        {
            using Tensor dense = t.contiguous().clone();   // clone: a contiguous tensor's contiguous() is itself
            return dense.numel() == 0 ? Array.Empty<byte>() : dense.bytes.ToArray();
        }

        /// <summary>Whether a NumSharp array and a tensor hold the same logical values, bit for bit.</summary>
        /// <param name="nd">The array.</param>
        /// <param name="t">The tensor.</param>
        /// <returns><c>true</c> when shapes and bytes match.</returns>
        protected static bool SameBits(NDArray nd, Tensor t)
            => nd.shape.SequenceEqual(t.shape) && BytesOf(nd).AsSpan().SequenceEqual(BytesOf(t));

        // ---- GC helpers ------------------------------------------------------------------------------

        /// <summary>Three full collections with their finalizers.</summary>
        protected static void Settle()
        {
            for (int i = 0; i < 3; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        /// <summary>Polls a condition, collecting between polls, until it holds or the timeout passes.</summary>
        /// <param name="condition">The condition.</param>
        /// <param name="timeoutMs">The timeout in milliseconds.</param>
        /// <returns>Whether the condition held in time.</returns>
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

        /// <summary>
        ///     A TorchSharp dispose scope: every tensor TorchSharp makes inside it (indexing temporaries included) is disposed when
        ///     it ends, so the leak gate sees only what the test meant to keep.
        /// </summary>
        /// <returns>The scope; dispose it (<c>using</c>).</returns>
        protected static DisposeScope Scope() => torch.NewDisposeScope();

        /// <summary>Reads one element of a tensor as a double inside its own scope (no leaked indexing temporary).</summary>
        /// <param name="t">The tensor.</param>
        /// <param name="index">The coordinates.</param>
        /// <returns>The element.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        protected static double At(Tensor t, params long[] index)
        {
            using var scope = Scope();
            Tensor e = t;
            foreach (long i in index)
                e = e[i];
            return e.to_type(ScalarType.Float64).item<double>();
        }
    }
}
