using System;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Numpy;                 // Numpy.NET (the Bare flavor): NDarray, Dtype, Numpy.np
using Python.Runtime;
using np2 = Numpy.np;        // NumSharp's np wins bare-name lookup (parent namespace); alias theirs

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Executable proof for <c>docs/website-src/docs/interop/numpy-net.md</c> — coexistence with
    ///     SciSharp's Numpy.NET over one shared engine and zero-copy handoffs.
    ///
    ///     <para>Tests read no markdown and assert no prose (the law of <c>98e6045a</c>): each test
    ///     reproduces one code block or transcript of the page. The page's GIL rule — every
    ///     Numpy.NET call, including <c>NDarray.Dispose()</c>, wrapped in <c>Py.GIL()</c> — is the
    ///     discipline every test here runs under; the deeper behavioural suite is
    ///     <see cref="NumpyNetInteropTests"/>.</para>
    /// </summary>
    [TestClass]
    public class DocExamples_NumpyNetPage : InteropTestBase
    {
        // ============================  ## One process, one engine  ===============================

        /// <summary>
        ///     "Numpy.NET's lazy initialization then sees the running engine and simply imports numpy
        ///     into it — no second interpreter, no installer."
        /// </summary>
        [TestMethod]
        public void SharedEngine_NumpyNetBootsOnOurs()
        {
            using (Gil())
            {
                using NDarray their = np2.arange(6);
                their.ToString().Should().Be("[0 1 2 3 4 5]");
                their.dtype.ToString().Should().Be("int64");
            }

            PyLong("1 + 1").Should().Be(2, "our scopes keep working — one shared engine, not a hijack");
            NDArrayPythonInterop.LiveExports.Should().Be(0, "Numpy.NET's own arrays involve no NumSharp pins");
        }

        /// <summary>
        ///     "the pythonnet versions unify by themselves" — Numpy.Bare declares 3.0.1, NumSharp's
        ///     floor is 3.0.5, and one loaded pythonnet ≥ 3.0.5 serves both.
        /// </summary>
        [TestMethod]
        public void Packaging_BareBindsAndPythonnetUnifies()
        {
            typeof(NDarray).Assembly.GetName().Name.Should().Be("Numpy.Bare",
                "the page tells readers to reference the Bare flavor — these types must come from it");

            var loaded = typeof(PythonEngine).Assembly.GetName().Version;
            (loaded >= new Version(3, 0, 5)).Should().BeTrue(
                $"NuGet must unify Numpy.Bare's pythonnet 3.0.1 up to NumSharp's floor (loaded: {loaded})");

            using (Gil())
            {
                using NDarray probe = np2.arange(3);
                probe.ToString().Should().Be("[0 1 2]", "Numpy.Bare works against the unified pythonnet");
            }
        }

        // ============================  ## The handoff  ===========================================

        /// <summary>
        ///     "Wrap the export: `new NDarray(nd.ToNumpy())`. Numpy.NET's whole C# surface then
        ///     operates over NumSharp's buffer — reads, reductions, in-place fills."
        /// </summary>
        [TestMethod]
        public void Wrap_TheirApiDrivesOurBuffer()
        {
            var ours = np.arange(6).astype(NPTypeCode.Double);

            using (Py.GIL())
            {
                using var wrapped = new NDarray(ours.ToNumpy());

                wrapped.sum().item<double>().Should().BeApproximately(15.0, 1e-9,
                    "Numpy.NET computes over NumSharp memory");
                wrapped.fill(7.0);
            }

            for (int i = 0; i < 6; i++)
                ReadAt<double>(ours, i).Should().BeApproximately(7.0, 1e-12, "the fill wrote into NumSharp's buffer");
        }

        /// <summary>
        ///     "Unwrap its `PyObject`: `their.self.AsNDArray()` for a view" — writes cross both ways
        ///     and NumSharp kernels see it all.
        /// </summary>
        [TestMethod]
        public void Unwrap_OurKernelsRunOverTheirArray()
        {
            NDarray their;
            using (Gil())
                their = np2.arange(8).astype(np2.float64);

            NDArray view;
            using (Gil())
                view = their.self.AsNDArray();

            WriteAt(view, -7.5, 1);
            using (Gil())
                their.item<double>(1).Should().BeApproximately(-7.5, 1e-12, "NumSharp write -> Numpy.NET read");

            using (Gil())
                their.fill(3.25);
            for (int k = 0; k < 8; k++)
                ReadAt<double>(view, k).Should().BeApproximately(3.25, 1e-12, "Numpy.NET write -> NumSharp read");

            ReadAt<double>(np.sum(view)).Should().BeApproximately(26.0, 1e-9, "NumSharp kernels over their buffer");

            view.Dispose();
            using (Gil())
                their.Dispose();
        }

        // ============================  ## Slices, dtypes and compute  ============================

        /// <summary>
        ///     "A Numpy.NET slice is a numpy view, and NumSharp leases it — writes through the
        ///     NumSharp side land in their *base* array"; a strided NumSharp view reads in logical order.
        /// </summary>
        [TestMethod]
        public void Slices_CrossInBothDirections()
        {
            NDarray their, theirSlice;
            using (Gil())
            {
                their = np2.arange(10).astype(np2.float64);
                theirSlice = their["2:8"];
            }

            NDArray sliceView;
            using (Gil())
                sliceView = theirSlice.self.AsNDArray();
            sliceView.size.Should().Be(6);
            ReadAt<double>(sliceView, 0).Should().BeApproximately(2.0, 1e-12);

            WriteAt(sliceView, -1.0, 3);   // slice element 3 == base element 5
            using (Gil())
                their.item<double>(5).Should().BeApproximately(-1.0, 1e-12,
                    "writing our view of their slice hits their base");

            var b = np.arange(24).astype(NPTypeCode.Double).reshape(4, 6);
            using (Gil())
            {
                using var wrappedStrided = new NDarray(b["1:3, ::2"].ToNumpy());
                wrappedStrided.shape.Dimensions.Should().Equal(2, 3);
                wrappedStrided.sum().item<double>().Should().BeApproximately(66.0, 1e-9,
                    "Numpy.NET reads the strided NumSharp view in logical order (6+8+10+12+14+16)");
            }

            sliceView.Dispose();
            using (Gil()) { theirSlice.Dispose(); their.Dispose(); }
        }

        /// <summary>"the numpy dtype is the NumSharp dtype's exact mapping, both ways."</summary>
        [TestMethod]
        public void Dtypes_ArriveExact_BothWays()
        {
            (NPTypeCode tc, string numpyName)[] outbound =
            {
                (NPTypeCode.Double, "float64"), (NPTypeCode.Single, "float32"),
                (NPTypeCode.Int32, "int32"), (NPTypeCode.Int64, "int64"),
                (NPTypeCode.Byte, "uint8"), (NPTypeCode.Boolean, "bool"),
            };
            foreach (var (tc, numpyName) in outbound)
            {
                var ours = np.arange(4).astype(tc);
                using (Gil())
                {
                    using var wrapped = new NDarray(ours.ToNumpy());
                    wrapped.dtype.ToString().Should().Be(numpyName, tc.ToString());
                }
            }

            using (Gil())
            {
                using var theirI4 = np2.arange(4).astype(np2.int32);
                using NDArray nd = theirI4.self.ToNDArray();
                nd.typecode.Should().Be(NPTypeCode.Int32);
                ReadAt<int>(nd, 3).Should().Be(3);

                using var theirF4 = np2.arange(4).astype(np2.float32);
                using NDArray nf = theirF4.self.ToNDArray();
                nf.typecode.Should().Be(NPTypeCode.Single);
            }
        }

        /// <summary>
        ///     "Numpy.NET's `matmul` over wrapped NumSharp inputs equals NumSharp's own" — the page's
        ///     cross-check block, with the mean it prints.
        /// </summary>
        [TestMethod]
        public void Compute_TheirMatmulEqualsOurs()
        {
            var a = (np.arange(4).astype(NPTypeCode.Double) / 2.0).reshape(2, 2);
            var b = (np.arange(4).astype(NPTypeCode.Double) + 1.0).reshape(2, 2);

            NDArray result;
            double theirMean;
            using (Py.GIL())
            {
                using var wa = new NDarray(a.ToNumpy());
                using var wb = new NDarray(b.ToNumpy());
                using var product = np2.matmul(wa, wb);      // real numpy computes...
                theirMean = product.mean();
                result = product.self.ToNDArray();           // ...NumSharp takes the result
            }

            var expected = np.matmul(a, b);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    ReadAt<double>(result, i, j).Should().BeApproximately(ReadAt<double>(expected, i, j), 1e-9,
                        "real numpy's product over wrapped inputs must equal NumSharp's own matmul");

            theirMean.Should().BeApproximately(4.25, 1e-9, "the transcript's mean");
            result.Dispose();
        }

        // ============================  ## Lifetime across three façades  =========================

        /// <summary>
        ///     "Their wrapper pins our buffer … Dispose the wrapper (under the GIL) and the pin
        ///     drains to 0."
        /// </summary>
        [TestMethod]
        public void Lifetime_TheirWrapperPinsOurBuffer_AndDrainsOnDispose()
        {
            var wrapped = MakeWrappedAndDropTheSource();
            Pump(); Pump(); Pump();   // the source NDArray is long collectable

            NDArrayPythonInterop.LiveExports.Should().Be(1,
                "the Numpy.NET wrapper's python reference is the only thing pinning the buffer");
            using (Gil())
                wrapped.sum().item<double>().Should().BeApproximately(20.0, 1e-9,
                    "Numpy.NET still reads valid NumSharp memory after every NumSharp-side reference died");

            using (Gil())
                wrapped.Dispose();    // the GIL rule applies to Dispose too

            WaitFor(() => NDArrayPythonInterop.LiveExports == 0).Should().BeTrue(
                "disposing the wrapper was the last reference — the pin must drain");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private NDarray MakeWrappedAndDropTheSource()
        {
            var src = np.arange(5).astype(NPTypeCode.Double) * 2;   // 0 2 4 6 8 — dropped immediately
            using (Gil())
                return new NDarray(src.ToNumpy());
        }

        /// <summary>
        ///     "Our lease outlives their wrapper … reads and writes through the NumSharp view stay
        ///     valid."
        /// </summary>
        [TestMethod]
        public void Lifetime_OurLeaseOutlivesTheirWrapper()
        {
            NDArray view;
            using (Gil())
            {
                var their = np2.arange(6).astype(np2.float64);
                view = their.self.AsNDArray();
                their.Dispose();   // their wrapper gone; our lease is now the only holder
            }

            Pump(); Pump();
            ReadAt<double>(view, 5).Should().BeApproximately(5.0, 1e-12,
                "the lease keeps the numpy array alive after Numpy.NET's wrapper died");
            WriteAt(view, 11.5, 0);
            ReadAt<double>(view, 0).Should().BeApproximately(11.5, 1e-12);
            NDArrayPythonInterop.LiveImports.Should().Be(1);

            view.Dispose();
        }

        /// <summary>
        ///     "The registered codec sees through the wrapper. `their.self.As&lt;NDArray&gt;()` decodes
        ///     like any numpy array — under the default `Auto` mode, as a zero-copy view."
        /// </summary>
        [TestMethod]
        public void Codec_TheirArraysDecode_BecauseTheyAreNdarrays()
        {
            CodecTests.EnsureCodec();

            NDarray their;
            using (Gil())
                their = np2.arange(6).astype(np2.float64);

            NDArray decoded;
            using (Gil())
                decoded = their.self.As<NDArray>();
            decoded.typecode.Should().Be(NPTypeCode.Double);

            WriteAt(decoded, -100.0, 0);
            using (Gil())
                their.item<double>(0).Should().BeApproximately(-100.0, 1e-12,
                    "the Auto decode of a contiguous numpy source is a zero-copy view");

            decoded.Dispose();
            using (Gil())
                their.Dispose();
        }
    }
}
