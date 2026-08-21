using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OpenBLAS;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     <b>Live byte-for-byte parity of <c>np.linalg.cholesky</c> / <c>np.linalg.qr</c> against real
    ///     NumPy.</b> Every case computes the factorisation twice over the SAME memory — once by
    ///     NumSharp's OpenBLAS backend, once by the embedded CPython's <c>numpy.linalg</c> reading
    ///     NumSharp's ZERO-COPY export — and asserts the two results are byte-identical
    ///     (<see cref="ByteContract"/> compares canonical C-order bytes, so it catches a wrong dtype
    ///     width, endianness, or a strided read the value-only <c>array_equal</c> would accept).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why this is byte-exact and not merely close.</b> NumSharp's <c>cholesky</c>/<c>qr</c> and
    ///     NumPy's both delegate to LAPACK (<c>potrf</c>/<c>geqrf</c>/<c>orgqr</c>). NumSharp bundles the
    ///     scipy-openblas build NumPy 2.4.2 itself pins (verified byte-identical), so on a NumPy-2.4.2
    ///     host the two stacks call the SAME binary. The result bits of a BLAS additionally depend on
    ///     the worker-thread count and the dispatched DYNAMIC_ARCH micro-kernel; <see cref="PythonSession"/>
    ///     enables the backend at <c>threads: 1</c> and these matrices are small enough that OpenBLAS
    ///     stays single-threaded on both sides, so the answer is deterministic and identical. On a host
    ///     whose NumPy uses a materially different BLAS the byte assertion is the point — it would flag
    ///     a real divergence — so there is no tolerance fallback here (unlike a pure value gate).
    ///     </para>
    ///     <para>
    ///     The input is exported <b>zero-copy</b>, so NumPy factorises NumSharp's actual buffer through
    ///     the operand's own strides — the strongest possible cross-check of every layout (C / F /
    ///     transposed / reversed / strided / broadcast). Neither <c>cholesky</c> nor <c>qr</c> mutates
    ///     its input (NumPy's <c>qr</c> copies internally; NumSharp copies into a column-major scratch),
    ///     so the shared buffer is safe to read from both sides.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class CholeskyQrLiveParityTests : InteropTestBase
    {
        private static void RequireLapack()
        {
            if (!OpenBlasEngine.LapackAvailable)
                Assert.Inconclusive("OpenBLAS LAPACK backend not available on this host.");
        }

        // ---- assertion helpers: NumSharp result vs live numpy result, byte-for-byte ---------------

        /// <summary>Factorise with both stacks over the zero-copy export and assert byte-identical.</summary>
        private void AssertCholeskyByteExact(NDArray a, bool upper, string because)
        {
            using var nsL = np.linalg.cholesky(a, upper);
            using (Gil())
            {
                using PyObject va = a.ToNumpy();
                using PyObject npL = Python.np.with(
                    $"np.linalg.cholesky(a, upper={(upper ? "True" : "False")})", ("a", va));
                ByteContract.AssertSameBytes(nsL, npL, because);
            }
        }

        private void AssertQrByteExact(NDArray a, string mode, string because)
        {
            var (q, r) = np.linalg.qr(a, mode);
            try
            {
                using (Gil())
                {
                    using PyObject va = a.ToNumpy();
                    if (mode == "r")
                    {
                        using PyObject npR = Python.np.with("np.linalg.qr(a, mode='r')", ("a", va));
                        ByteContract.AssertSameBytes(r, npR, because + " [R]");
                    }
                    else
                    {
                        // reduced/complete return QRResult(Q,R); raw returns (h, tau) — both index [0]/[1].
                        using PyObject npQ = Python.np.with($"np.linalg.qr(a, mode='{mode}')[0]", ("a", va));
                        using PyObject npR = Python.np.with($"np.linalg.qr(a, mode='{mode}')[1]", ("a", va));
                        ByteContract.AssertSameBytes(q, npQ, because + " [Q]");
                        ByteContract.AssertSameBytes(r, npR, because + " [R]");
                    }
                }
            }
            finally
            {
                q?.Dispose();
                r?.Dispose();
            }
        }

        // ---- shared inputs ------------------------------------------------------------------------

        /// <summary>A non-diagonal SPD matrix (exercises off-diagonal factorisation).</summary>
        private static NDArray RichSpd() => np.array(new double[,] { { 4, 2, 1 }, { 2, 5, 3 }, { 1, 3, 6 } });

        /// <summary>A diagonal SPD whose values survive a cast to every integer dtype (and bool → identity).</summary>
        private static NDArray DiagSpd() => np.array(new double[,] { { 4, 0, 0 }, { 0, 9, 0 }, { 0, 0, 16 } });

        /// <summary>A Hermitian positive-definite matrix.</summary>
        private static NDArray HermPd() => np.array(new Complex[,]
        {
            { new Complex(2, 0), new Complex(1, -1) },
            { new Complex(1, 1), new Complex(3, 0) }
        });

        private static NDArray TallM() => np.array(new double[,]
            { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 }, { 1, 0, 2 } });

        private static NDArray WideM() => np.array(new double[,]
            { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 12, 11 } });

        private static NDArray SquareM() => np.array(new double[,] { { 2, 1, 0 }, { 1, 3, 1 }, { 0, 1, 4 } });

        // =====================================  cholesky  =========================================

        [TestMethod]
        public void Backend_IsOpenBlasWithLapack()
        {
            Assert.IsTrue(OpenBlasEngine.Enabled, "the interop suite runs with OpenBLAS as the default backend");
            RequireLapack();
        }

        [TestMethod]
        public void Cholesky_LowerAndUpper_ValueRich_ByteExact()
        {
            RequireLapack();
            AssertCholeskyByteExact(RichSpd(), upper: false, "cholesky lower, non-diagonal SPD");
            AssertCholeskyByteExact(RichSpd(), upper: true, "cholesky upper, non-diagonal SPD");
        }

        [TestMethod]
        public void Cholesky_Float32_ByteExact()
        {
            RequireLapack();
            AssertCholeskyByteExact(RichSpd().astype(np.float32), upper: false, "cholesky lower float32");
            AssertCholeskyByteExact(RichSpd().astype(np.float32), upper: true, "cholesky upper float32");
        }

        [TestMethod]
        public void Cholesky_Complex_Hermitian_ByteExact()
        {
            RequireLapack();
            AssertCholeskyByteExact(HermPd(), upper: false, "cholesky lower, Hermitian PD");
            AssertCholeskyByteExact(HermPd(), upper: true, "cholesky upper, Hermitian PD");
        }

        [TestMethod]
        public void Cholesky_EveryWideningDtype_ByteExact()
        {
            RequireLapack();
            foreach (var tc in new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64
            })
            {
                // DiagSpd's diagonal is {4,9,16}: fits every width; bool collapses it to the identity.
                AssertCholeskyByteExact(DiagSpd().astype(tc), upper: false, $"cholesky {tc} → float64");
            }
        }

        [TestMethod]
        public void Cholesky_Batched_ByteExact()
        {
            RequireLapack();
            var stack = np.stack(new[]
            {
                np.array(new double[,] { { 4.0, 2.0 }, { 2.0, 5.0 } }),
                np.array(new double[,] { { 9.0, 3.0 }, { 3.0, 10.0 } }),
                np.array(new double[,] { { 25.0, 5.0 }, { 5.0, 3.0 } })
            });
            AssertCholeskyByteExact(stack, upper: false, "cholesky batched (3,2,2)");

            // A 4-D stack (2,2,3,3) of the same SPD tiled through broadcast then materialised.
            var big = np.stack(new[] { np.stack(new[] { RichSpd(), RichSpd() }), np.stack(new[] { RichSpd(), RichSpd() }) });
            AssertCholeskyByteExact(big, upper: false, "cholesky batched (2,2,3,3)");
        }

        [TestMethod]
        public void Cholesky_EveryLayout_ByteExact()
        {
            RequireLapack();
            var s = RichSpd();
            AssertCholeskyByteExact(np.ascontiguousarray(s), false, "C-contiguous");
            AssertCholeskyByteExact(np.asfortranarray(s), false, "F-contiguous");
            AssertCholeskyByteExact(s.T, false, "transposed view (SPD.T == SPD)");
            AssertCholeskyByteExact(s["::-1, ::-1"], false, "reversed both axes (still SPD)");

            // Strided: embed the SPD in a 2x-larger zero array and slice it back out at stride 2.
            var big = np.zeros(new Shape(6, 6), np.float64);
            big["::2, ::2"] = s;
            AssertCholeskyByteExact(big["::2, ::2"], false, "strided view");

            // Broadcast: 3 identical SPD matrices via a stride-0 batch axis (read-only on both sides).
            AssertCholeskyByteExact(np.broadcast_to(s, new Shape(3, 3, 3)), false, "broadcast (3,3,3)");
        }

        [TestMethod]
        public void Cholesky_Edge_1x1_And_Empty_ByteExact()
        {
            RequireLapack();
            AssertCholeskyByteExact(np.array(new double[,] { { 4.0 } }), false, "1x1");
            AssertCholeskyByteExact(np.zeros(new Shape(0, 0), np.float64), false, "empty (0,0)");
        }

        [TestMethod]
        public void Cholesky_NotPositiveDefinite_BothRaise()
        {
            RequireLapack();
            using var bad = np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
            new Action(() => np.linalg.cholesky(bad)).Should().Throw<LinAlgError>()
                .WithMessage("Matrix is not positive definite");
            using (Gil())
            {
                using PyObject va = bad.ToNumpy();
                // numpy raises numpy.linalg.LinAlgError with the identical text — surfaced as a PythonException.
                new Action(() => { using var _ = Python.np.with("np.linalg.cholesky(a)", ("a", va)); })
                    .Should().Throw<PythonException>().WithMessage("*Matrix is not positive definite*");
            }
        }

        // =======================================  qr  =============================================

        [TestMethod]
        public void Qr_Reduced_TallWideSquare_ByteExact()
        {
            RequireLapack();
            AssertQrByteExact(TallM(), "reduced", "qr reduced tall (4,3)");
            AssertQrByteExact(WideM(), "reduced", "qr reduced wide (3,4)");
            AssertQrByteExact(SquareM(), "reduced", "qr reduced square (3,3)");
        }

        [TestMethod]
        public void Qr_Complete_TallWideSquare_ByteExact()
        {
            RequireLapack();
            AssertQrByteExact(TallM(), "complete", "qr complete tall (4,3)");
            AssertQrByteExact(WideM(), "complete", "qr complete wide (3,4)");
            AssertQrByteExact(SquareM(), "complete", "qr complete square (3,3)");
        }

        [TestMethod]
        public void Qr_RMode_ByteExact()
        {
            RequireLapack();
            AssertQrByteExact(TallM(), "r", "qr r-mode tall");
            AssertQrByteExact(WideM(), "r", "qr r-mode wide");
        }

        [TestMethod]
        public void Qr_RawMode_ByteExact()
        {
            RequireLapack();
            // raw returns (h, tau); NumSharp's h is C-contiguous vs numpy's F-view, but ByteContract
            // compares CANONICAL C-order bytes, so the values (which are identical) are what is asserted.
            AssertQrByteExact(TallM(), "raw", "qr raw tall");
            AssertQrByteExact(WideM(), "raw", "qr raw wide");
        }

        [TestMethod]
        public void Qr_Float32_AllModes_ByteExact()
        {
            RequireLapack();
            foreach (var mode in new[] { "reduced", "complete", "r", "raw" })
                AssertQrByteExact(TallM().astype(np.float32), mode, $"qr float32 {mode}");
        }

        [TestMethod]
        public void Qr_Complex_AllModes_ByteExact()
        {
            RequireLapack();
            var m = np.array(new Complex[,]
            {
                { new Complex(1, 2), new Complex(3, -1) },
                { new Complex(4, 0), new Complex(1, 1) },
                { new Complex(-2, 1), new Complex(0, 3) }
            });
            foreach (var mode in new[] { "reduced", "complete", "r", "raw" })
                AssertQrByteExact(m, mode, $"qr complex {mode}");
        }

        [TestMethod]
        public void Qr_EveryWideningDtype_Reduced_ByteExact()
        {
            RequireLapack();
            foreach (var tc in new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64
            })
            {
                AssertQrByteExact(TallM().astype(tc), "reduced", $"qr {tc} → float64");
            }
        }

        [TestMethod]
        public void Qr_Batched_ByteExact()
        {
            RequireLapack();
            var stack3d = np.arange(3 * 5 * 3.0).reshape(3, 5, 3) + np.eye(5, 3);
            AssertQrByteExact(stack3d, "reduced", "qr batched (3,5,3) reduced");
            AssertQrByteExact(stack3d, "complete", "qr batched (3,5,3) complete");

            var stack4d = np.arange(2 * 2 * 4 * 3.0).reshape(2, 2, 4, 3) + np.eye(4, 3);
            AssertQrByteExact(stack4d, "reduced", "qr batched (2,2,4,3) reduced");
            AssertQrByteExact(stack4d, "raw", "qr batched (2,2,4,3) raw");
        }

        [TestMethod]
        public void Qr_EveryLayout_Reduced_ByteExact()
        {
            RequireLapack();
            var m = TallM();
            AssertQrByteExact(np.ascontiguousarray(m), "reduced", "C-contiguous");
            AssertQrByteExact(np.asfortranarray(m), "reduced", "F-contiguous");
            AssertQrByteExact(m.T, "reduced", "transposed view (3,4)");
            AssertQrByteExact(m["::-1"], "reduced", "reversed rows");

            var big = np.zeros(new Shape(8, 6), np.float64);
            big["::2, ::2"] = m;
            AssertQrByteExact(big["::2, ::2"], "reduced", "strided view");

            AssertQrByteExact(np.broadcast_to(m, new Shape(3, 4, 3)), "reduced", "broadcast batch (3,4,3)");
        }

        [TestMethod]
        public void Qr_Edge_DegenerateShapes_ByteExact()
        {
            RequireLapack();
            AssertQrByteExact(np.zeros(new Shape(3, 0), np.float64), "reduced", "(3,0) reduced");
            AssertQrByteExact(np.zeros(new Shape(3, 0), np.float64), "complete", "(3,0) complete → Q=I");
            AssertQrByteExact(np.zeros(new Shape(0, 3), np.float64), "reduced", "(0,3) reduced");
            AssertQrByteExact(np.array(new double[,] { { 5.0 } }), "reduced", "1x1 reduced");
            AssertQrByteExact(np.array(new double[,] { { 5.0 } }), "complete", "1x1 complete");
        }
    }
}
