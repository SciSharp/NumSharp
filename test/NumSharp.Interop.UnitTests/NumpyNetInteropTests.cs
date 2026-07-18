using System;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using Numpy;
using NumSharp.Interop.PythonNet; // Numpy.NET (the Bare flavor): NDarray, Dtype, Numpy.np
using Python.Runtime;
using np2 = Numpy.np;        // NumSharp's np wins bare-name lookup (parent namespace); alias theirs

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Unit-test-proven interoperability with <b>Numpy.NET</b> (SciSharp's <c>Numpy</c> /
    ///     <c>Numpy.Bare</c> packages) THROUGH the pythonnet bridge. Every section of the docs page
    ///     "Interoperability → Numpy.NET" maps 1:1 to a test here.
    ///
    ///     <para>The two idioms under proof:</para>
    ///     <list type="bullet">
    ///       <item><b>Wrap:</b> <c>new NDarray(nd.ToNumpy())</c> — Numpy.NET's whole C# API operates
    ///         zero-copy over NumSharp's buffer.</item>
    ///       <item><b>Unwrap:</b> <c>ndarray.self</c> (its <see cref="PyObject"/>) into
    ///         <c>ToNDArray</c>/<c>ToNDArrayView</c> — NumSharp kernels run over Numpy.NET's arrays.</item>
    ///     </list>
    ///
    ///     <para><b>The GIL rule:</b> Numpy.NET contains no GIL management of its own — it assumes the
    ///     GIL stays held after <c>PythonEngine.Initialize()</c>. This suite (like any app that calls
    ///     <c>BeginAllowThreads()</c>) therefore wraps EVERY Numpy.NET call, including
    ///     <c>NDarray.Dispose()</c>, in <c>Py.GIL()</c>. That rule is itself part of what these tests
    ///     prove and document.</para>
    /// </summary>
    [TestClass]
    public class NumpyNetInteropTests : InteropTestBase
    {
        [TestMethod]
        public void NumpyNet_BootsOnTheSharedEngine()
        {
            // Numpy.NET's lazy init sees the already-initialized engine (ours) and simply imports
            // numpy into it — no second engine, no installer, same interpreter state.
            using (Gil())
            {
                using var their = np2.arange(6);
                their.ToString().Should().Be("[0 1 2 3 4 5]");
                their.dtype.ToString().Should().Be("int64");
            }

            PyLong("1 + 1").Should().Be(2, "our scopes keep working — one shared engine, not a hijack");
            NDArrayInterop.LiveExports.Should().Be(0, "Numpy.NET's own arrays involve no NumSharp pins");
        }

        [TestMethod]
        public void Wrap_NumSharpBuffer_DrivenByNumpyNetApi_ZeroCopy()
        {
            var ours = np.arange(6).astype(NPTypeCode.Double);

            using (Gil())
            {
                using var wrapped = new NDarray(NDArrayInterop.ToNumpy(ours));

                wrapped.sum().item<double>().Should().BeApproximately(15.0, 1e-9, "Numpy.NET computes over NumSharp memory");

                // pointer-level aliasing, both directions
                WriteAt(ours, 99.5, 2);
                wrapped.item<double>(2).Should().BeApproximately(99.5, 1e-12, "NumSharp write -> Numpy.NET read");

                wrapped.fill(7.0);
                for (int i = 0; i < 6; i++)
                    ReadAt<double>(ours, i).Should().BeApproximately(7.0, 1e-12, "Numpy.NET write -> NumSharp read");

                wrapped.GetData<double>().Should().Equal(new[] { 7.0, 7, 7, 7, 7, 7 });

                // numpy's own verdict on aliasing
                Scope.Set("w", wrapped.self);
            }

            ExportTo("o2", ours);
            PyBool("np.shares_memory(w, o2)").Should().BeTrue("one buffer, three façades (NumSharp, Numpy.NET, scope)");
        }

        [TestMethod]
        public void Unwrap_NumpyNetArray_AsNumSharpView_SharedMutation()
        {
            NDarray their;
            using (Gil())
                their = np2.arange(8).astype(np2.float64);

            NDArray view;
            using (Gil())
                view = NDArrayInterop.ToNDArrayView(their.self);

            NDArrayInterop.LiveImports.Should().Be(1);

            WriteAt(view, -7.5, 1);
            using (Gil())
                their.item<double>(1).Should().BeApproximately(-7.5, 1e-12, "NumSharp write -> Numpy.NET read");

            using (Gil())
                their.fill(3.25);
            for (int k = 0; k < 8; k++)
                ReadAt<double>(view, k).Should().BeApproximately(3.25, 1e-12, "Numpy.NET write -> NumSharp read");

            // NumSharp kernels straight over Numpy.NET's buffer
            ReadAt<double>(np.sum(view)).Should().BeApproximately(26.0, 1e-9);

            using (Gil())
                their.Dispose();
        }

        [TestMethod]
        public void DtypeFidelity_AcrossBothWrapDirections()
        {
            // NumSharp -> Numpy.NET: the wrapped dtype must be the exact numpy name
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
                    using var wrapped = new NDarray(NDArrayInterop.ToNumpy(ours));
                    wrapped.dtype.ToString().Should().Be(numpyName, tc.ToString());
                    switch (tc)
                    {
                        case NPTypeCode.Double: wrapped.GetData<double>().Should().Equal(new[] { 0.0, 1, 2, 3 }); break;
                        case NPTypeCode.Single: wrapped.GetData<float>().Should().Equal(new[] { 0f, 1, 2, 3 }); break;
                        case NPTypeCode.Int32: wrapped.GetData<int>().Should().Equal(new[] { 0, 1, 2, 3 }); break;
                        case NPTypeCode.Int64: wrapped.GetData<long>().Should().Equal(new[] { 0L, 1, 2, 3 }); break;
                        case NPTypeCode.Byte: wrapped.GetData<byte>().Should().Equal(new byte[] { 0, 1, 2, 3 }); break;
                        case NPTypeCode.Boolean: wrapped.GetData<bool>().Should().Equal(new[] { false, true, true, true }); break;
                    }
                }
            }

            // Numpy.NET -> NumSharp: their dtype arrives as the exact NPTypeCode
            using (Gil())
            {
                using var theirI4 = np2.arange(4).astype(np2.int32);
                var nd = NDArrayInterop.ToNDArray(theirI4.self);
                nd.typecode.Should().Be(NPTypeCode.Int32);
                ReadAt<int>(nd, 3).Should().Be(3);

                using var theirF4 = np2.arange(4).astype(np2.float32);
                theirF4.self.ToNDArray().typecode.Should().Be(NPTypeCode.Single);
            }
        }

        [TestMethod]
        public void Slices_CrossTheBoundary_InBothDirections()
        {
            // Numpy.NET slice -> NumSharp view (their slicing produces a numpy view; we lease it)
            NDarray their, theirSlice;
            using (Gil())
            {
                their = np2.arange(10).astype(np2.float64);
                theirSlice = their["2:8"];
            }

            NDArray sliceView;
            using (Gil())
                sliceView = NDArrayInterop.ToNDArrayView(theirSlice.self);
            sliceView.size.Should().Be(6);
            ReadAt<double>(sliceView, 0).Should().BeApproximately(2.0, 1e-12);

            WriteAt(sliceView, -1.0, 3);   // slice element 3 == base element 5
            using (Gil())
                their.item<double>(5).Should().BeApproximately(-1.0, 1e-12, "writing our view of their slice hits their base");

            // NumSharp strided view -> Numpy.NET (their GetData linearizes non-contiguous correctly)
            var b = np.arange(24).astype(NPTypeCode.Double).reshape(4, 6);
            var v = b["1:3, ::2"];
            using (Gil())
            {
                using var wrappedStrided = new NDarray(NDArrayInterop.ToNumpy(v));
                wrappedStrided.shape.Dimensions.Should().Equal(2, 3);
                wrappedStrided.GetData<double>().Should().Equal(new[] { 6.0, 8, 10, 12, 14, 16 },
                    "Numpy.NET must see the LOGICAL order of the strided NumSharp view");
                wrappedStrided.sum().item<double>().Should().BeApproximately(66.0, 1e-9);
            }

            using (Gil()) { theirSlice.Dispose(); their.Dispose(); }
        }

        [TestMethod]
        public void Lifetime_TheirWrapperIsTheOnlyHolder_OfNumSharpMemory()
        {
            [MethodImpl(MethodImplOptions.NoInlining)]
            NDarray MakeWrapped()
            {
                var src = np.arange(5).astype(NPTypeCode.Double) * 2;   // 0 2 4 6 8 — dropped immediately
                using (Gil())
                    return new NDarray(NDArrayInterop.ToNumpy(src));
            }

            var wrapped = MakeWrapped();
            Pump(); Pump(); Pump();   // the source NDArray is long collectable

            NDArrayInterop.LiveExports.Should().Be(1, "the Numpy.NET wrapper's python reference pins the buffer");
            using (Gil())
                wrapped.sum().item<double>().Should().BeApproximately(20.0, 1e-9,
                    "Numpy.NET still reads valid NumSharp memory after every NumSharp-side reference died");

            // NDarray.Dispose decrefs WITHOUT taking the GIL (Numpy.NET assumes it is held) —
            // in a BeginAllowThreads world the caller must hold it. That's the documented rule.
            using (Gil())
                wrapped.Dispose();
            wrapped = null;

            WaitFor(() => NDArrayInterop.LiveExports == 0).Should().BeTrue(
                "disposing the Numpy.NET wrapper was the last reference — the pin must drain");
        }

        [TestMethod]
        public void Lifetime_OurViewOutlivesTheirDisposedWrapper()
        {
            NDArray view;
            using (Gil())
            {
                var their = np2.arange(6).astype(np2.float64);
                view = NDArrayInterop.ToNDArrayView(their.self);
                their.Dispose();   // their wrapper gone; our lease is now the only holder
            }

            Pump(); Pump();
            ReadAt<double>(view, 5).Should().BeApproximately(5.0, 1e-12,
                "the lease must keep the numpy array alive after Numpy.NET's wrapper was disposed");
            WriteAt(view, 11.5, 0);
            ReadAt<double>(view, 0).Should().BeApproximately(11.5, 1e-12);
            NDArrayInterop.LiveImports.Should().Be(1);
        }

        [TestMethod]
        public void Compute_NumpyNetPipeline_CrossCheckedAgainstNumSharp()
        {
            var a = (np.arange(4).astype(NPTypeCode.Double) / 2.0).reshape(2, 2);
            var b = (np.arange(4).astype(NPTypeCode.Double) + 1.0).reshape(2, 2);

            NDArray product;
            double theirMean;
            using (Gil())
            {
                using var wa = new NDarray(NDArrayInterop.ToNumpy(a));
                using var wb = new NDarray(NDArrayInterop.ToNumpy(b));
                using var theirProduct = np2.matmul(wa, wb);
                theirMean = theirProduct.mean();   // Numpy.NET's mean() returns double directly
                product = NDArrayInterop.ToNDArray(theirProduct.self);
            }

            var expected = np.matmul(a, b);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    ReadAt<double>(product, i, j).Should().BeApproximately(ReadAt<double>(expected, i, j), 1e-9,
                        "Numpy.NET's matmul over wrapped NumSharp inputs must equal NumSharp's own matmul");

            theirMean.Should().BeApproximately(ReadAt<double>(np.mean(expected)), 1e-9);
        }

        [TestMethod]
        public void Codec_DecodesNumpyNetArrays_AndScopesInterleave()
        {
            CodecTests.EnsureCodec();

            NDarray their;
            using (Gil())
                their = np2.arange(6).astype(np2.float64);

            // their NDarray's PyObject decodes through our codec (it IS a numpy.ndarray)
            NDArray decoded;
            using (Gil())
                decoded = their.self.As<NDArray>();
            decoded.typecode.Should().Be(NPTypeCode.Double);
            ReadAt<double>(decoded, 5).Should().BeApproximately(5.0, 1e-12);

            WriteAt(decoded, -100.0, 0);
            using (Gil())
                their.item<double>(0).Should().BeApproximately(0.0, 1e-12, "decode defaults to copy — no accidental aliasing");

            // and the same object can simultaneously live in a plain scope
            using (Gil())
                Scope.Set("nn", their.self);
            PyExec("nn *= 2");
            using (Gil())
            {
                their.item<double>(3).Should().BeApproximately(6.0, 1e-12,
                    "the scope name, the Numpy.NET wrapper and numpy itself are one object");
                their.Dispose();
            }
        }
    }
}
