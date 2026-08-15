using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OpenBLAS;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     <b>Live byte-for-byte parity of the CBLAS product family against real NumPy.</b> Every
    ///     product NumPy 2.4.2 routes through cblas — <c>np.dot</c>/<c>np.matmul</c> and the five
    ///     product gufuncs <c>inner</c>/<c>vdot</c>/<c>vecdot</c>/<c>matvec</c>/<c>vecmat</c> — is
    ///     computed twice over the SAME zero-copy exported memory (once by NumSharp's OpenBLAS backend,
    ///     once by the embedded CPython's numpy) and asserted byte-identical
    ///     (<see cref="ByteContract.AssertSameBytes(NDArray, PyObject, string)"/> compares canonical
    ///     C-order bytes).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The headline is <b>complex128</b>: complex float accumulation is NOT associative, so — unlike
    ///     integer/bool products (bit-exact by construction) and the already-gated float32/float64 dot —
    ///     a portable managed GEMM cannot reproduce NumPy's bits. Only the shared <c>zgemm</c>/
    ///     <c>zgemv</c>/<c>zsyrk</c>/<c>zdotu_sub</c>/<c>zdotc_sub</c> route does, and only when both
    ///     stacks call the SAME bundled scipy-openblas at <c>threads: 1</c> (the levers of
    ///     <see cref="InteropTestBase"/>'s CLAUDE.md §4). The offline host-pinned <c>matmul_parity</c>
    ///     corpus tier gates the same claim from a committed snapshot; this is the live half — numpy
    ///     computing over NumSharp's actual exported bytes, no serialization in between.
    ///     </para>
    ///     <para>
    ///     The two dot flavours are the whole story, and the coverage below is built around telling
    ///     them apart because a complex-specific bug hides in exactly that seam:
    ///     <list type="bullet">
    ///       <item><b>UNCONJUGATED</b> (<c>zdotu</c>) — <c>dot</c>, <c>matmul</c> row·column,
    ///         <c>inner</c>, <c>matvec</c>. <c>Σ x·y</c>.</item>
    ///       <item><b>CONJUGATING</b> (<c>zdotc</c>) — <c>vdot</c>, <c>vecdot</c>, <c>vecmat</c>.
    ///         <c>Σ conj(x)·y</c>, conjugating the FIRST operand.</item>
    ///     </list>
    ///     The builders below give every operand a genuine non-zero, ASYMMETRIC imaginary part, so a
    ///     dropped conjugation changes the answer (a real-only operand would make <c>vdot ≡ dot</c> and
    ///     the test vacuous — the same trap the corpus generator's <c>_mp_values</c> fix guards).
    ///     </para>
    ///     <para>
    ///     The <c>a @ a.T</c> route (shared data pointer, <c>m == p</c>, transposes opposite) takes the
    ///     <b>zsyrk</b> shortcut on BOTH stacks — for complex that is the SYMMETRIC product <c>A·Aᵀ</c>
    ///     (<c>zsyrk</c>, NOT <c>zherk</c>: matmul does not conjugate), so it is byte-exact.
    ///     <c>a @ conj(a).T</c> materialises a fresh operand (different pointer), takes no shortcut, and
    ///     goes through <c>zgemm</c> — also byte-exact, and it exercises the Hermitian product without a
    ///     shortcut.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class ProductsLiveParityTests : InteropTestBase
    {
        /// <summary>
        ///     Inconclusive when the loaded library lacks the complex128 products (a bare real-only
        ///     CBLAS) — there complex <c>dot</c>/<c>matmul</c> fall through to the managed complex
        ///     kernel, a correct but NOT byte-identical answer, so a byte compare would (correctly) go
        ///     red for the wrong reason. Mirrors <c>EigLiveParityTests.RequireLapack</c>.
        /// </summary>
        private static void RequireProducts()
        {
            if (!OpenBlasEngine.ComplexProductsAvailable)
                Assert.Inconclusive("OpenBLAS complex128 products not available on this host.");
        }

        /// <summary>
        ///     Computes the product in NumSharp (already done by the caller, passed as
        ///     <paramref name="ns"/>) and again in the embedded numpy over the SAME zero-copy exported
        ///     operands, asserting byte-identical. Both call the same bundled scipy-openblas z-routines
        ///     at threads:1, so the complex accumulation is bit-reproducible. <paramref name="npExpr"/>
        ///     is a numpy expression over the bound names <c>a</c> and <c>b</c>.
        /// </summary>
        private void AssertBin(NDArray ns, string npExpr, NDArray a, NDArray b, string because)
        {
            using (ns)
            using (Gil())
            {
                using PyObject va = a.ToNumpy();
                using PyObject vb = b.ToNumpy();
                using PyObject np_ = Python.np.with(npExpr, ("a", va), ("b", vb));
                ByteContract.AssertSameBytes(ns, np_, because);
            }
        }

        // ---- operand builders (genuine non-zero, asymmetric imaginary parts) ---------------------

        /// <summary>A length-<paramref name="n"/> complex128 vector; re and im distinct and asymmetric
        /// so conjugation is observable.</summary>
        private static NDArray CVec(int n)
        {
            var d = new Complex[n];
            for (int i = 0; i < n; i++)
                d[i] = new Complex(0.5 * i - 1.25, 2.0 - 0.75 * i);
            return np.array(d);
        }

        /// <summary>An (<paramref name="rows"/>,<paramref name="cols"/>) complex128 matrix, likewise
        /// asymmetric between re and im.</summary>
        private static NDArray CMat(int rows, int cols)
        {
            var d = new Complex[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    d[i, j] = new Complex(1.0 + 0.5 * i - 0.25 * j, 2.0 - 0.75 * i + 0.5 * j);
            return np.array(d);
        }

        // =====================================  backend  =========================================

        [TestMethod]
        public void Backend_IsOpenBlasWithComplexProducts()
        {
            Assert.IsTrue(OpenBlasEngine.Enabled, "the interop suite runs with OpenBLAS as the default backend");
            RequireProducts();
            OpenBlasEngine.ComplexProductsAvailable.Should().BeTrue();
        }

        // =====================================  np.dot  ==========================================

        [TestMethod]
        public void Dot_Complex_MatrixMatrix_ByteExact()
        {
            RequireProducts();
            // 2-D @ 2-D → zgemm.
            AssertBin(np.dot(CMat(3, 4), CMat(4, 5)), "np.dot(a,b)", CMat(3, 4), CMat(4, 5), "dot complex (3,4)@(4,5) zgemm");
        }

        [TestMethod]
        public void Dot_Complex_VectorVector_Unconjugated_ByteExact()
        {
            RequireProducts();
            // 1-D · 1-D → zdotu (UNCONJUGATED — np.dot does not conjugate, unlike np.vdot on the same
            // operands). This is the discriminating case for the dotu/dotc split.
            AssertBin(np.dot(CVec(6), CVec(6)), "np.dot(a,b)", CVec(6), CVec(6), "dot complex vector·vector zdotu");
        }

        [TestMethod]
        public void Dot_Complex_MatrixVector_ByteExact()
        {
            RequireProducts();
            // matrix · vector → zgemv (PyArray_MatrixProduct2's cblas dispatch).
            AssertBin(np.dot(CMat(3, 4), CVec(4)), "np.dot(a,b)", CMat(3, 4), CVec(4), "dot complex matrix·vector zgemv");
        }

        [TestMethod]
        public void Dot_Complex_ND_ByteExact()
        {
            RequireProducts();
            // >2-D operand → NumPy's dotfunc (zdotu per (row, column)) tail of PyArray_MatrixProduct2,
            // NOT gemm — the only route that matches an N-D complex np.dot.
            var a = np.stack(new[] { CMat(2, 3), CMat(2, 3) * new Complex(0, 1) }); // (2,2,3)
            var b = CMat(3, 4);
            AssertBin(np.dot(a, b), "np.dot(a,b)", a, b, "dot complex (2,2,3)·(3,4) zdotu tail");
        }

        [TestMethod]
        public void Dot_Complex_SelfTranspose_Syrk_ByteExact()
        {
            RequireProducts();
            // a @ a.T shares the data pointer → the zsyrk shortcut (symmetric A·Aᵀ). np.dot of a 2-D
            // pair goes through the same @TYPE@_matmul matrixmatrix as matmul.
            var a = CMat(4, 3);
            AssertBin(np.dot(a, a.T), "np.dot(a, a.T)", a, a, "dot complex a@a.T zsyrk shortcut");
        }

        // =====================================  np.matmul  =======================================

        [TestMethod]
        public void Matmul_Complex_2D_ByteExact()
        {
            RequireProducts();
            AssertBin(np.matmul(CMat(3, 4), CMat(4, 2)), "np.matmul(a,b)", CMat(3, 4), CMat(4, 2), "matmul complex (3,4)@(4,2) zgemm");
        }

        [TestMethod]
        public void Matmul_Complex_Batched_ByteExact()
        {
            RequireProducts();
            // Stacked matmul: one zgemm per batch element, plan hoisted once.
            var a = np.stack(new[] { CMat(2, 3), CMat(2, 3) + new Complex(1, -1), CMat(2, 3) * new Complex(0, 2) }); // (3,2,3)
            var b = np.stack(new[] { CMat(3, 4), CMat(3, 4) - new Complex(0, 1), CMat(3, 4) * new Complex(2, 0) }); // (3,3,4)
            AssertBin(np.matmul(a, b), "np.matmul(a,b)", a, b, "matmul complex batched (3,2,3)@(3,3,4)");
        }

        [TestMethod]
        public void Matmul_Complex_SelfTranspose_Syrk_ByteExact()
        {
            RequireProducts();
            var a = CMat(4, 3);
            AssertBin(np.matmul(a, a.T), "np.matmul(a, a.T)", a, a, "matmul complex a@a.T zsyrk shortcut");
        }

        [TestMethod]
        public void Matmul_Complex_Hermitian_NoShortcut_Gemm_ByteExact()
        {
            RequireProducts();
            // a @ conj(a).T: conj materialises a fresh operand (different pointer), so NO syrk shortcut
            // fires on either stack — it is a plain zgemm of the Hermitian product. Exercises zgemm with
            // a conjugated operand without the shortcut.
            var a = CMat(4, 3);
            var ah = np.conjugate(a).T;
            AssertBin(np.matmul(a, ah), "np.matmul(a, np.conjugate(a).T)", a, a, "matmul complex a@a.H zgemm (no shortcut)");
        }

        [TestMethod]
        public void Matmul_Complex_NonBlasable_Strided_ByteExact()
        {
            RequireProducts();
            // A column-strided left operand is not blasable (its inner axis stride is not one element),
            // so both stacks copy it into a temp before gemm (gh-23588). Byte-identical through the copy.
            var left = CMat(4, 6)[":, ::2"]; // (4,3), last-axis stride 2 elements → non-blasable
            var b = CMat(3, 5);
            AssertBin(np.matmul(left, b), "np.matmul(a,b)", left, b, "matmul complex non-blasable strided-left (copy path)");
        }

        [TestMethod]
        public void Matmul_Complex_VectorMatrix_And_MatrixVector_ByteExact()
        {
            RequireProducts();
            // vector @ matrix and matrix @ vector are the two gemv special-cases of @TYPE@_matmul.
            AssertBin(np.matmul(CVec(3), CMat(3, 4)), "np.matmul(a,b)", CVec(3), CMat(3, 4), "matmul complex vector@matrix zgemv");
            AssertBin(np.matmul(CMat(3, 4), CVec(4)), "np.matmul(a,b)", CMat(3, 4), CVec(4), "matmul complex matrix@vector zgemv");
        }

        [TestMethod]
        public void Matmul_Complex_EveryLayout_ByteExact()
        {
            RequireProducts();
            var b = CMat(3, 4);
            var baseA = CMat(2, 3);
            AssertBin(np.matmul(np.ascontiguousarray(baseA), b), "np.matmul(a,b)", np.ascontiguousarray(baseA), b, "matmul complex C-contiguous");
            AssertBin(np.matmul(np.asfortranarray(baseA), b), "np.matmul(a,b)", np.asfortranarray(baseA), b, "matmul complex F-contiguous");
            AssertBin(np.matmul(baseA["::-1, ::-1"], b), "np.matmul(a,b)", baseA["::-1, ::-1"], b, "matmul complex reversed both axes");

            // transposed-left VIEW (F-blasable): (3,2).T == (2,3).
            var t = CMat(3, 2);
            AssertBin(np.matmul(t.T, b), "np.matmul(a,b)", t.T, b, "matmul complex transposed-left view");
        }

        // =====================================  np.inner  ========================================

        [TestMethod]
        public void Inner_Complex_Unconjugated_ByteExact()
        {
            RequireProducts();
            // np.inner is matrixproduct(a, swapaxes(b,-1,-2)) — UNCONJUGATED. Vector and matrix forms.
            AssertBin(np.inner(CVec(5), CVec(5)), "np.inner(a,b)", CVec(5), CVec(5), "inner complex vector·vector");
            AssertBin(np.inner(CMat(2, 3), CMat(4, 3)), "np.inner(a,b)", CMat(2, 3), CMat(4, 3), "inner complex (2,3)·(4,3)");
        }

        // =====================================  np.vdot  =========================================

        [TestMethod]
        public void Vdot_Complex_Conjugated_ByteExact()
        {
            RequireProducts();
            // np.vdot flattens BOTH operands to 1-D and calls the CONJUGATING zdotc — Σ conj(a)·b.
            AssertBin(np.vdot(CVec(6), CVec(6)), "np.vdot(a,b)", CVec(6), CVec(6), "vdot complex vector zdotc");
            AssertBin(np.vdot(CMat(2, 3), CMat(2, 3)), "np.vdot(a,b)", CMat(2, 3), CMat(2, 3), "vdot complex 2-D (flattened) zdotc");
        }

        [TestMethod]
        public void Vdot_Complex_ActuallyConjugates_NonVacuity()
        {
            RequireProducts();
            // Guard the gate against vacuity: with a non-zero imaginary part, the CONJUGATING vdot must
            // DIFFER from the UNCONJUGATED dot on the same operands. If they matched, every conjugation
            // assertion above would be silently meaningless.
            var a = CVec(6);
            var b = CVec(6);
            var conj = Convert.ToDouble(np.abs(np.subtract(np.vdot(a, b), np.dot(a, b))).GetValue(0));
            conj.Should().BeGreaterThan(1e-6, "vdot conjugates the first operand, so it must differ from the unconjugated dot");
        }

        // =====================================  np.vecdot  =======================================

        [TestMethod]
        public void Vecdot_Complex_Conjugated_ByteExact()
        {
            RequireProducts();
            // (n),(n)->() reducing the last axis with the CONJUGATING zdotc per element.
            AssertBin(np.vecdot(CVec(5), CVec(5)), "np.vecdot(a,b)", CVec(5), CVec(5), "vecdot complex vector zdotc");
            AssertBin(np.vecdot(CMat(3, 4), CMat(3, 4)), "np.vecdot(a,b)", CMat(3, 4), CMat(3, 4), "vecdot complex batched (3,4) zdotc");
        }

        [TestMethod]
        public void Vecdot_Complex_Broadcast_ByteExact()
        {
            RequireProducts();
            // Leading-axis broadcast of the operands (b stretches over a's leading axis).
            AssertBin(np.vecdot(CMat(3, 4), CVec(4)), "np.vecdot(a,b)", CMat(3, 4), CVec(4), "vecdot complex (3,4) vs (4,) broadcast");
        }

        // =====================================  np.matvec  =======================================

        [TestMethod]
        public void Matvec_Complex_Unconjugated_ByteExact()
        {
            RequireProducts();
            // (m,n),(n)->(m) — the linear transform, UNCONJUGATED (zgemv / per-row zdotu).
            AssertBin(np.matvec(CMat(3, 4), CVec(4)), "np.matvec(a,b)", CMat(3, 4), CVec(4), "matvec complex (3,4)·(4,)");
            var a = np.stack(new[] { CMat(3, 4), CMat(3, 4) * new Complex(0, 1) }); // (2,3,4)
            AssertBin(np.matvec(a, CVec(4)), "np.matvec(a,b)", a, CVec(4), "matvec complex batched (2,3,4)·(4,)");
        }

        // =====================================  np.vecmat  =======================================

        [TestMethod]
        public void Vecmat_Complex_Conjugated_ByteExact()
        {
            RequireProducts();
            // (n),(n,m)->(m) conjugating the vector. Complex CANNOT use gemv (gemv does not conjugate),
            // so it is @name@_vecmat_via_gemm: a 1×M×N zgemm with CblasConjTrans on the vector.
            AssertBin(np.vecmat(CVec(3), CMat(3, 4)), "np.vecmat(a,b)", CVec(3), CMat(3, 4), "vecmat complex (3,)·(3,4) gemm-ConjTrans");
            var b = np.stack(new[] { CMat(3, 4), CMat(3, 4) + new Complex(1, 1) }); // (2,3,4)
            AssertBin(np.vecmat(CVec(3), b), "np.vecmat(a,b)", CVec(3), b, "vecmat complex batched (3,)·(2,3,4)");
        }

        // ==================  real (s/d) product-gufunc seams — newly routed  =====================
        // The five gufunc seams route through the OpenBLAS backend for float32/float64 too (previously
        // they composed over dot/matmul, or — vecdot — over Multiply+ReduceAdd). Prove the seam is
        // byte-identical to numpy at both real widths; the double-accumulated chunked ?dot the seam
        // uses is exactly NumPy's @name@_dot / @name@_dotc for real dtypes.

        private static NDArray RVec(int n, double scale = 1.0)
        {
            var d = new double[n];
            for (int i = 0; i < n; i++)
                d[i] = (0.5 * i - 1.25) * scale;
            return np.array(d);
        }

        private static NDArray RMat(int rows, int cols)
        {
            var d = new double[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    d[i, j] = 1.0 + 0.5 * i - 0.25 * j;
            return np.array(d);
        }

        [TestMethod]
        public void Products_Real_Float64_Gufuncs_ByteExact()
        {
            RequireProducts();
            AssertBin(np.inner(RMat(2, 3), RMat(4, 3)), "np.inner(a,b)", RMat(2, 3), RMat(4, 3), "inner float64 seam");
            AssertBin(np.vdot(RMat(2, 3), RMat(2, 3)), "np.vdot(a,b)", RMat(2, 3), RMat(2, 3), "vdot float64 seam");
            AssertBin(np.vecdot(RMat(3, 4), RMat(3, 4)), "np.vecdot(a,b)", RMat(3, 4), RMat(3, 4), "vecdot float64 seam");
            AssertBin(np.matvec(RMat(3, 4), RVec(4)), "np.matvec(a,b)", RMat(3, 4), RVec(4), "matvec float64 seam");
            AssertBin(np.vecmat(RVec(3), RMat(3, 4)), "np.vecmat(a,b)", RVec(3), RMat(3, 4), "vecmat float64 seam");
        }

        [TestMethod]
        public void Products_Real_Float32_Gufuncs_ByteExact()
        {
            RequireProducts();
            var m = RMat(3, 4).astype(np.float32);
            var m2 = RMat(3, 4).astype(np.float32);
            var v = RVec(4).astype(np.float32);
            var v3 = RVec(3).astype(np.float32);
            AssertBin(np.vecdot(m, m2), "np.vecdot(a,b)", m, m2, "vecdot float32 seam (double-accumulated sdot)");
            AssertBin(np.matvec(m, v), "np.matvec(a,b)", m, v, "matvec float32 seam sgemv");
            AssertBin(np.vecmat(v3, m), "np.vecmat(a,b)", v3, m, "vecmat float32 seam sgemv");
        }
    }
}
