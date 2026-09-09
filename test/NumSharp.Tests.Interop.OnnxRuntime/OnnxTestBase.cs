using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Shared plumbing for the ONNX Runtime interop tests.
    ///
    ///     <para><b>Leak gate:</b> every test captures the <see cref="NDArrayOnnxInterop.LiveExports"/> /
    ///     <see cref="NDArrayOnnxInterop.LiveImports"/> baseline on entry, and the cleanup FAILS the test
    ///     unless the counters return to that baseline — so every single test doubles as a no-leak /
    ///     no-premature-free assertion. Dispose every handle and view you make (<c>using</c>).</para>
    ///
    ///     <para><b>Models:</b> the committed <c>Models/*.onnx</c> files (see <c>test/oracle/gen_onnx_models.py</c>)
    ///     are copied beside the test binaries; <see cref="Model"/> resolves one, <see cref="OpenSession"/>
    ///     opens it on the CPU execution provider. One shared <see cref="SessionOptions"/> keeps ORT's own
    ///     thread pools small so the suite stays deterministic.</para>
    ///
    ///     <para><b>GC realism:</b> collection-dependent lifecycles must run inside
    ///     <c>[MethodImpl(MethodImplOptions.NoInlining)]</c> helpers — a debug-build JIT keeps untracked temps
    ///     of a frame's references alive until the method returns.</para>
    /// </summary>
    public abstract class OnnxTestBase
    {
        private int _baseExports, _baseImports;

        [TestInitialize]
        public void InteropInit()
        {
            Settle();
            _baseExports = NDArrayOnnxInterop.LiveExports;
            _baseImports = NDArrayOnnxInterop.LiveImports;
        }

        [TestCleanup]
        public void InteropCleanup()
        {
            bool settled = WaitFor(() => NDArrayOnnxInterop.LiveExports <= _baseExports &&
                                         NDArrayOnnxInterop.LiveImports <= _baseImports, 10_000);
            Assert.IsTrue(settled,
                $"interop leaked conversions: LiveExports {_baseExports} -> {NDArrayOnnxInterop.LiveExports}, " +
                $"LiveImports {_baseImports} -> {NDArrayOnnxInterop.LiveImports}");
        }

        // ---- models & sessions ----------------------------------------------------------------------

        private static readonly Lazy<SessionOptions> SharedOptions = new(() =>
        {
            var options = new SessionOptions();
            options.IntraOpNumThreads = 1;
            options.InterOpNumThreads = 1;
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
            return options;
        });

        /// <summary>Absolute path of a committed model, e.g. <c>Model("identity_float32")</c>.</summary>
        protected static string Model(string name)
        {
            string path = Path.Combine(AppContext.BaseDirectory, "Models", name + ".onnx");
            if (!File.Exists(path))
                throw new FileNotFoundException($"test model not found: {path} (run test/oracle/gen_onnx_models.py and rebuild)", path);
            return path;
        }

        /// <summary>An <see cref="InferenceSession"/> over a committed model, CPU execution provider, single-threaded.</summary>
        protected static InferenceSession OpenSession(string model) => new(Model(model), SharedOptions.Value);

        /// <summary>The identity model for a NumSharp dtype — its ORT element type's <c>identity_&lt;dtype&gt;.onnx</c>.</summary>
        protected static string IdentityModel(NPTypeCode code) => code switch
        {
            NPTypeCode.Boolean => "identity_bool",
            NPTypeCode.Byte => "identity_uint8",
            NPTypeCode.SByte => "identity_int8",
            NPTypeCode.Int16 => "identity_int16",
            NPTypeCode.UInt16 => "identity_uint16",
            NPTypeCode.Char => "identity_uint16",
            NPTypeCode.Int32 => "identity_int32",
            NPTypeCode.UInt32 => "identity_uint32",
            NPTypeCode.Int64 => "identity_int64",
            NPTypeCode.UInt64 => "identity_uint64",
            NPTypeCode.Half => "identity_float16",
            NPTypeCode.Single => "identity_float32",
            NPTypeCode.Double => "identity_float64",
            NPTypeCode.Decimal => "identity_float64",
            _ => throw new ArgumentOutOfRangeException(nameof(code), code, "no identity model for this dtype"),
        };

        /// <summary>The 12 dtypes that cross zero-copy (Char and Decimal are the two conversions; Complex is refused).</summary>
        protected static readonly NPTypeCode[] ZeroCopyDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double,
        };

        // ---- data ------------------------------------------------------------------------------------

        /// <summary>
        ///     A C-contiguous, OWNING <c>arange</c>-valued array of the dtype and shape (values small enough for
        ///     every dtype). Reshape first (a view of the ramp), THEN astype: the copy is what makes the result own
        ///     its buffer (owndata == true), which the resize/refcheck tests rely on.
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

        /// <summary>An <c>int[]</c> shape as the <c>long[]</c> NumSharp's <c>nd.shape</c> and ORT's <c>Shape</c> report.</summary>
        protected static long[] Dims(params int[] shape) => Array.ConvertAll(shape, d => (long)d);

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
