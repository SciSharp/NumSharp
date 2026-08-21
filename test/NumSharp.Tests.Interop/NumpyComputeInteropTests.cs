using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     The other half of the byte-contract: not just that a crossing carries the right BYTES, but
    ///     that real numpy KERNELS run correctly over NumSharp's exported memory and agree with
    ///     NumSharp's own kernels over the same buffer. Every case computes an op twice — once in numpy
    ///     over the zero-copy export, once in NumSharp over the source — and asserts byte-identical
    ///     results, so a stride numpy read wrong (or an element NumSharp laid out wrong) shows up as a
    ///     value divergence, not just a metadata one.
    ///
    ///     <para>Exactness is kept honest by choosing ops that cannot diverge in the last ULP: integer
    ///     elementwise / reductions / matmul / broadcasting are associative-exact, and float
    ///     <c>negative</c>/<c>abs</c> are pure sign-bit ops. The handful of float REDUCTIONS use an
    ///     approximate compare. Broadcast reductions are deliberately excluded (a known NumSharp
    ///     OpenBug, unrelated to interop).</para>
    ///
    ///     <para>numpy is driven through the established <c>Python.np.with(expr, (name, operand))</c> helper —
    ///     it hands numpy the very PyObject the interop exported. <c>np</c> stays NumSharp.</para>
    /// </summary>
    [TestClass]
    public class NumpyComputeInteropTests : InteropTestBase
    {
        // The six memory layouts every compute case walks, at a caller-chosen dtype. Each is built from
        // its OWN fresh arange so the entries never alias one buffer.
        //
        // Dtype choice is deliberate and platform-driven: elementwise / matmul use dtype-PRESERVING ops
        // (int32 add/negate/multiply → int32 on every platform), but a REDUCTION's accumulator is the C
        // `long` — 32-bit on a standard-CPython Windows, 64-bit on unix — so numpy's sum(int32) is int32
        // on GH's windows-latest yet NumSharp's is always int64. Reductions therefore run at int64 (numpy
        // never promotes past it), keeping the byte compare width-identical on all three CI OSes.
        private static (NDArray nd, string label)[] Layouts(NPTypeCode tc) => new[]
        {
            (np.arange(24).astype(tc).reshape(4, 6),                   "contiguous(4,6)"),
            (np.arange(24).astype(tc).reshape(4, 6)["::2"],            "strided-rows(2,6)"),
            (np.arange(24).astype(tc).reshape(4, 6).T,                 "transposed(6,4)"),
            (np.arange(24).astype(tc).reshape(4, 6)["::-1"],           "reversed-rows(4,6)"),
            (np.asfortranarray(np.arange(24).astype(tc).reshape(4, 6)), "F-order(4,6)"),
            (np.arange(24).astype(tc).reshape(2, 3, 4).T,             "transposed-3d(4,3,2)"),
        };

        // ==============================  elementwise: numpy-over-export == NumSharp  ============

        [TestMethod]
        public void Elementwise_NumpyOverEveryLayout_EqualsNumSharpKernel()
        {
            using (Py.GIL())
            {
                foreach (var (nd, label) in Layouts(NPTypeCode.Int32))
                {
                    using (nd)
                    using (PyObject v = nd.ToNumpy())
                    {
                        using (NDArray add = nd + 7)
                        using (PyObject npAdd = Python.np.with("np.add(a, 7)", ("a", v)))
                            ByteContract.AssertSameBytes(add, npAdd, $"add: {label}");

                        using (NDArray neg = -nd)
                        using (PyObject npNeg = Python.np.with("np.negative(a)", ("a", v)))
                            ByteContract.AssertSameBytes(neg, npNeg, $"negative: {label}");

                        using (NDArray sq = nd * nd)
                        using (PyObject npSq = Python.np.with("np.multiply(a, a)", ("a", v)))
                            ByteContract.AssertSameBytes(sq, npSq, $"multiply: {label}");
                    }
                }
            }
        }

        // Broadcast (read-only) export: elementwise still round-trips through numpy correctly.
        [TestMethod]
        public void Elementwise_NumpyOverBroadcastExport_EqualsNumSharpKernel()
        {
            using (Py.GIL())
            {
                using NDArray nd = np.broadcast_to(np.arange(6).astype(NPTypeCode.Int32), new Shape(4, 6));
                using PyObject v = nd.ToNumpy();
                v.writeable().Should().BeFalse("a broadcast export is read-only on both sides");

                using (NDArray add = nd + 1000)
                using (PyObject npAdd = Python.np.with("np.add(a, 1000)", ("a", v)))
                    ByteContract.AssertSameBytes(add, npAdd, "add over broadcast export");
            }
        }

        // Dtype sweep on a contiguous export: every integer width, with dtype-PRESERVING ops (add of a
        // weak scalar, and multiply — whose overflow wraps modularly and identically on both sides), so
        // the result dtype is the source dtype on every platform.
        [TestMethod]
        public void Elementwise_NumpyOverEveryIntegerDtype_EqualsNumSharpKernel()
        {
            NPTypeCode[] dtypes =
            {
                NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.Int32, NPTypeCode.Int64,
                NPTypeCode.Byte, NPTypeCode.UInt16, NPTypeCode.UInt32, NPTypeCode.UInt64,
            };

            using (Py.GIL())
            {
                foreach (NPTypeCode tc in dtypes)
                {
                    using NDArray nd = np.arange(24).astype(tc).reshape(4, 6);
                    using PyObject v = nd.ToNumpy();

                    using (NDArray add = nd + 3)
                    using (PyObject npAdd = Python.np.with("np.add(a, 3)", ("a", v)))
                        ByteContract.AssertSameBytes(add, npAdd, $"add: {tc}");

                    using (NDArray mul = nd * nd)                                     // wraps modularly on both sides
                    using (PyObject npMul = Python.np.with("np.multiply(a, a)", ("a", v)))
                        ByteContract.AssertSameBytes(mul, npMul, $"multiply (modular wrap): {tc}");
                }
            }
        }

        // ==============================  reductions: numpy-over-export == NumSharp  =============

        [TestMethod]
        public void Reductions_NumpyOverStridedExports_EqualNumSharpKernels()
        {
            // Non-broadcast layouts only (broadcast reduce is a separate NumSharp OpenBug); int64 source so
            // the sum accumulator is int64 on every OS (see Layouts()).
            using (Py.GIL())
            {
                foreach (var (nd, label) in Layouts(NPTypeCode.Int64))
                {
                    using (nd)
                    using (PyObject v = nd.ToNumpy())
                    {
                        using (NDArray s0 = np.sum(nd, 0))
                        using (PyObject np0 = Python.np.with("np.sum(a, axis=0)", ("a", v)))
                            ByteContract.AssertSameBytes(s0, np0, $"sum axis0: {label}");   // int64 -> int64 both

                        using (NDArray full = np.sum(nd))
                        using (PyObject npFull = Python.np.with("np.sum(a)", ("a", v)))
                            ByteContract.AssertSameBytes(full, npFull, $"sum full: {label}");

                        using (NDArray mn = np.min(nd, 1))
                        using (PyObject npMn = Python.np.with("np.min(a, axis=1)", ("a", v)))
                            ByteContract.AssertSameBytes(mn, npMn, $"min axis1: {label}");

                        using (NDArray mx = np.max(nd, 0))
                        using (PyObject npMx = Python.np.with("np.max(a, axis=0)", ("a", v)))
                            ByteContract.AssertSameBytes(mx, npMx, $"max axis0: {label}");
                    }
                }
            }
        }

        // ==============================  matmul: numpy-over-exports == NumSharp  ================

        [TestMethod]
        public void Matmul_NumpyOverExports_EqualsNumSharpKernel()
        {
            using (Py.GIL())
            {
                // int64 so the product dtype is int64 on every OS (int32 matmul stays int32 here but the C
                // long accumulator makes that platform-fragile — int64 sidesteps it entirely).
                using NDArray a = np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4);
                using NDArray b = np.arange(8).astype(NPTypeCode.Int64).reshape(4, 2);
                using NDArray bt = np.arange(8).astype(NPTypeCode.Int64).reshape(2, 4).T;   // (4,2) transposed view

                using (PyObject va = a.ToNumpy())
                using (PyObject vb = b.ToNumpy())
                using (PyObject vbt = bt.ToNumpy())
                {
                    using (NDArray nsAb = np.matmul(a, b))
                    using (PyObject npAb = Python.np.with("np.matmul(x, y)", ("x", va), ("y", vb)))
                        ByteContract.AssertSameBytes(nsAb, npAb, "matmul contiguous");

                    using (NDArray nsAbt = np.matmul(a, bt))
                    using (PyObject npAbt = Python.np.with("np.matmul(x, y)", ("x", va), ("y", vbt)))
                        ByteContract.AssertSameBytes(nsAbt, npAbt, "matmul with a transposed export operand");
                }
            }
        }

        // ==============================  in-place ufunc writes through the shared export  =======

        [TestMethod]
        public void InPlaceUfunc_NumpyWritesThroughContiguousExport()
        {
            using (Py.GIL())
            {
                using NDArray nd = np.arange(6).astype(NPTypeCode.Int32);
                using (PyObject v = nd.ToNumpy())
                using (PyObject _ = Python.np.with("np.multiply(a, 3, out=a)", ("a", v))) { }   // in place

                using PyObject expected = Python.np.eval("np.arange(6, dtype='<i4') * 3");
                ByteContract.AssertSameBytes(nd, expected, "numpy's in-place *3 reached the shared NumSharp buffer");
            }
        }

        [TestMethod]
        public void InPlaceUfunc_NumpyWritesThroughStridedExport_RespectsStrides()
        {
            using (Py.GIL())
            {
                // Only rows 0 and 2 are exported (step-2). numpy's in-place += must touch exactly those,
                // leaving row 1 untouched — proving the strided export addresses the right elements.
                using NDArray nd = np.arange(12).astype(NPTypeCode.Int32).reshape(3, 4);
                using (NDArray strided = nd["::2"])
                using (PyObject v = strided.ToNumpy())
                using (PyObject _ = Python.np.with("np.add(a, 100, out=a)", ("a", v))) { }

                using PyObject expected = Python.np.evalStmts(
                    "m = np.arange(12, dtype='<i4').reshape(3, 4)\nm[::2] += 100", "m");
                ByteContract.AssertSameBytes(nd, expected, "in-place add reached rows 0 & 2 only, through the parent buffer");
            }
        }

        // ==============================  broadcasting across two exports  =======================

        [TestMethod]
        public void Broadcasting_NumpyOverTwoExports_EqualsNumSharpKernel()
        {
            using (Py.GIL())
            {
                using NDArray col = np.arange(3).astype(NPTypeCode.Int32).reshape(3, 1);
                using NDArray row = np.arange(4).astype(NPTypeCode.Int32).reshape(1, 4);

                using NDArray nsSum = col + row;                                  // (3,4)
                using PyObject vc = col.ToNumpy();
                using PyObject vr = row.ToNumpy();
                using PyObject npSum = Python.np.with("np.add(c, r)", ("c", vc), ("r", vr));

                ByteContract.AssertSameBytes(nsSum, npSum, "numpy broadcasts two exports the same as NumSharp");

                // and the numpy result crosses BACK to a byte-identical NumSharp array.
                using NDArray imported = npSum.ToNDArray();
                ByteContract.NsBytes(imported).Should().Equal(ByteContract.NsBytes(nsSum), "round-trip of the broadcast result");
            }
        }

        // ==============================  zero-copy proven by numpy's own shares_memory  =========

        [TestMethod]
        public void ZeroCopy_Export_IsProvenBy_SharesMemory()
        {
            using (Py.GIL())
            {
                using NDArray nd = np.arange(10).astype(NPTypeCode.Int32);
                using PyObject v1 = nd.ToNumpy();
                using PyObject v2 = nd.ToNumpy();
                using PyObject copy = nd.ToNumpyCopy();

                Python.np.shares_memory(v1, v2).Should().BeTrue("two views of the same NumSharp buffer share memory");
                Python.np.truthy("np.shares_memory(a, a[2:])", ("a", v1))
                     .Should().BeTrue("a numpy slice of the export still shares the NumSharp buffer");
                Python.np.shares_memory(v1, copy).Should().BeFalse("a copy export is detached");
            }
        }

        // ==============================  float: sign-bit ops exact, reductions approximate  =====

        [TestMethod]
        public void Float_SignBitOps_NumpyOverExport_AreByteExact()
        {
            using (Py.GIL())
            {
                using NDArray nd = np.array(new[] { 1.5, -2.25, 0.0, -0.0, 100.0, -0.5 });
                using PyObject v = nd.ToNumpy();

                using (NDArray neg = -nd)
                using (PyObject npNeg = Python.np.with("np.negative(a)", ("a", v)))
                    ByteContract.AssertSameBytes(neg, npNeg, "float negate is a pure sign-bit flip — bit-exact");

                using (NDArray ab = np.abs(nd))
                using (PyObject npAb = Python.np.with("np.abs(a)", ("a", v)))
                    ByteContract.AssertSameBytes(ab, npAb, "float abs clears the sign bit — bit-exact");
            }
        }

        [TestMethod]
        public void Float_Reductions_NumpyOverExport_ApproximatelyEqualNumSharp()
        {
            using (Py.GIL())
            {
                using NDArray nd = np.arange(1000).astype(NPTypeCode.Double).reshape(25, 40);
                using PyObject v = nd.ToNumpy();

                double nsSum = np.sum(nd).GetAtIndex<double>(0);
                double npSum = Python.np.with("np.sum(a)", ("a", v)).As<double>();
                npSum.Should().BeApproximately(nsSum, Math.Abs(nsSum) * 1e-12 + 1e-9, "numpy sum over the export ≈ NumSharp sum");

                double nsMean = np.mean(nd).GetAtIndex<double>(0);
                double npMean = Python.np.with("np.mean(a)", ("a", v)).As<double>();
                npMean.Should().BeApproximately(nsMean, Math.Abs(nsMean) * 1e-12 + 1e-9, "numpy mean over the export ≈ NumSharp mean");
            }
        }

        // ==============================  codec in a real numpy compute pipeline  ================

        [TestMethod]
        public void Codec_EncodeDecode_ThroughNumpyCompute_MatchesNumSharp()
        {
            var codec = new NumpyCodec();   // a LOCAL codec — never touches the process-global registration
            using (Py.GIL())
            {
                using NDArray nd = np.arange(8).astype(NPTypeCode.Int64).reshape(2, 4);   // int64: cumsum stays int64 on every OS

                using PyObject encoded = codec.TryEncode(nd);            // NDArray -> numpy view
                encoded.Should().NotBeNull("the codec encodes the NDArray as a numpy array");
                using PyObject result = Python.np.with("np.cumsum(x, axis=1)", ("x", encoded));   // numpy computes

                codec.TryDecode(result, out NDArray back).Should().BeTrue("the codec decodes the numpy result");
                using (back)
                using (NDArray nsExpected = np.cumsum(nd, 1))            // int64 -> int64 on both sides
                    ByteContract.NsBytes(back).Should().Equal(ByteContract.NsBytes(nsExpected),
                        "codec-encode → numpy cumsum → codec-decode equals NumSharp cumsum");
            }
        }
    }
}
