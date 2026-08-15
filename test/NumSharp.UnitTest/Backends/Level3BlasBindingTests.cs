using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.OpenBLAS;
using Blas = NumSharp.Interop.OpenBLAS.OpenBlasEngine;

namespace NumSharp.UnitTest.Backends
{
    /// <summary>
    ///     The optional Level-3 BLAS routines the OpenBLAS backend binds beyond the parity core
    ///     (gemm/gemv/syrk/dot): <c>trmm</c>, <c>trsm</c>, <c>symm</c>, <c>syr2k</c>, single and double.
    ///     NumPy exposes no <c>np.*</c> that reaches them (its np.* matrix product uses only
    ///     gemm/gemv/syrk/dot; trmm/trsm live in NumPy solely inside its private reference LAPACK), so
    ///     — unlike <c>np.dot</c>/<c>np.matmul</c>, which are bit-pinned by the host <c>matmul_parity</c>
    ///     corpus — there is nothing to compare bits against. They are infrastructure reached through
    ///     the internal <see cref="Blas.IBlasType{T}"/> seam, verified here NUMERICALLY against a
    ///     managed reference across the whole CBLAS parameter space: Side × Uplo × TransA × Diag ×
    ///     sizes × both precisions × <b>both storage orders (Row/Col-major)</b> × <b>non-unit leading
    ///     dimensions</b> (sub-matrix views, with a sentinel proving BLAS never writes outside the
    ///     logical window) × <b>degenerate/empty sizes</b>. Goes Inconclusive on a host with no loadable
    ///     BLAS, exactly like <see cref="MatmulParityBackendTests"/>.
    /// </summary>
    [TestClass]
    public unsafe class Level3BlasBindingTests
    {
        private const double Alpha = 0.75, Beta = -0.5, Sentinel = -98765.0;

        /// <summary>(row-major?, leading-dimension padding): the layout axis every routine is swept over
        /// — contiguous and sub-matrix (padded ld), in both storage orders.</summary>
        private static readonly (bool RowMajor, int Ext)[] Layouts =
            { (true, 0), (true, 2), (false, 0), (false, 3) };

        [TestCleanup]
        public void Cleanup() => OpenBlasEngine.Disable();

        private static void RequireBlas()
        {
            try
            {
                OpenBlasEngine.Enable();
            }
            catch (Exception e)
            {
                Assert.Inconclusive("no CBLAS library on this host: " + e.Message.Split('\n')[0]);
            }
        }

        [TestMethod]
        public void Trmm_MatchesManagedReference_AcrossParameterSpace() => RunTriangular(solve: false);

        [TestMethod]
        public void Trsm_MatchesManagedReference_AcrossParameterSpace() => RunTriangular(solve: true);

        [TestMethod]
        public void Symm_MatchesManagedReference_AcrossParameterSpace()
        {
            RequireBlas();
            var rng = new Random(707);
            var fails = new List<string>();

            foreach (bool dbl in new[] { true, false })
                foreach (var (rm, ext) in Layouts)
                    foreach (var (m, n) in new[] { (3, 4), (5, 2), (2, 2), (1, 3) })
                        foreach (CBlasSide side in new[] { CBlasSide.Left, CBlasSide.Right })
                        {
                            int adim = side == CBlasSide.Left ? m : n;
                            double[] a = Rand(rng, adim, adim), b = Rand(rng, m, n), cIn = Rand(rng, m, n);
                            int lda = Ld(adim, adim, rm, ext), ldb = Ld(m, n, rm, ext), ldc = Ld(m, n, rm, ext);
                            foreach (CBlasUpLo up in new[] { CBlasUpLo.Upper, CBlasUpLo.Lower })
                            {
                                double[] sym = BuildSym(a, adim, up == CBlasUpLo.Upper);
                                double[] prod = side == CBlasSide.Left ? MatMul(sym, adim, adim, b, m, n) : MatMul(b, m, n, sym, adim, adim);
                                var want = new double[m * n];
                                for (int i = 0; i < want.Length; i++) want[i] = Alpha * prod[i] + Beta * cIn[i];

                                double[] phys = NativeSymm(dbl, side, up, m, n, Pack(a, adim, adim, rm, lda), lda,
                                    Pack(b, m, n, rm, ldb), ldb, Pack(cIn, m, n, rm, ldc), ldc, rm);
                                string tag = $"{P(dbl)}symm rm={rm} ext={ext} side={side} up={up} ({m}x{n})";
                                Compare(fails, tag, Unpack(phys, m, n, rm, ldc), want, dbl, null);
                                CheckPad(fails, tag, phys, m, n, rm, ldc, ext);
                            }
                        }

            Assert.AreEqual(0, fails.Count, string.Join("\n", fails));
        }

        [TestMethod]
        public void Syr2k_MatchesManagedReference_AcrossParameterSpace()
        {
            RequireBlas();
            var rng = new Random(909);
            var fails = new List<string>();

            foreach (bool dbl in new[] { true, false })
                foreach (var (rm, ext) in Layouts)
                    foreach (var (n, k) in new[] { (4, 3), (2, 5), (3, 1), (2, 2) })
                        foreach (CBlasTranspose tr in new[] { CBlasTranspose.NoTrans, CBlasTranspose.Trans })
                        {
                            int ar = tr == CBlasTranspose.NoTrans ? n : k, ac = tr == CBlasTranspose.NoTrans ? k : n;
                            int ldab = Ld(ar, ac, rm, ext), ldc = n + ext;
                            double[] a = Rand(rng, ar, ac), b = Rand(rng, ar, ac), cIn = Rand(rng, n, n);
                            foreach (CBlasUpLo up in new[] { CBlasUpLo.Upper, CBlasUpLo.Lower })
                            {
                                double[] t1, t2;
                                if (tr == CBlasTranspose.NoTrans)
                                {
                                    t1 = MatMul(a, n, k, Transpose(b, n, k), k, n);
                                    t2 = MatMul(b, n, k, Transpose(a, n, k), k, n);
                                }
                                else
                                {
                                    t1 = MatMul(Transpose(a, k, n), n, k, b, k, n);
                                    t2 = MatMul(Transpose(b, k, n), n, k, a, k, n);
                                }
                                var want = new double[n * n];
                                var mask = new bool[n * n];
                                for (int i = 0; i < n; i++)
                                    for (int j = 0; j < n; j++)
                                    {
                                        want[i * n + j] = Alpha * (t1[i * n + j] + t2[i * n + j]) + Beta * cIn[i * n + j];
                                        // syr2k writes only the requested triangle; the other keeps its input, so mask it out.
                                        mask[i * n + j] = up == CBlasUpLo.Upper ? j >= i : j <= i;
                                    }
                                double[] phys = NativeSyr2k(dbl, up, tr, n, k, Pack(a, ar, ac, rm, ldab),
                                    Pack(b, ar, ac, rm, ldab), ldab, Pack(cIn, n, n, rm, ldc), ldc, rm);
                                Compare(fails, $"{P(dbl)}syr2k rm={rm} ext={ext} up={up} tr={tr} (n={n},k={k})",
                                    Unpack(phys, n, n, rm, ldc), want, dbl, mask);
                            }
                        }

            Assert.AreEqual(0, fails.Count, string.Join("\n", fails));
        }

        /// <summary>Degenerate / empty dimensions — the BLAS quick-return paths (m=0, n=0, 1×1, and
        /// syr2k k=0 which reduces to <c>C := βC</c>). Must not crash and must respect the definition.</summary>
        [TestMethod]
        public void DegenerateAndEmptySizes_QuickReturnCorrectly()
        {
            RequireBlas();
            var rng = new Random(1313);
            var fails = new List<string>();

            foreach (bool dbl in new[] { true, false })
            {
                foreach (var (m, n) in new[] { (0, 3), (3, 0), (1, 1) })
                {
                    int adim = Math.Max(m, 1);
                    double[] a = RandTri(rng, adim), b = Rand(rng, m, n);
                    double[] got = NativeTriangular(dbl, false, CBlasSide.Left, CBlasUpLo.Upper,
                        CBlasTranspose.NoTrans, CBlasDiag.NonUnit, m, n, a, adim, adim, b, Math.Max(n, 1), true);
                    var want = (m == 1 && n == 1) ? new[] { Alpha * a[0] * b[0] } : new double[m * n];
                    Compare(fails, $"{P(dbl)}trmm empty ({m}x{n})", got, want, dbl, null);
                }

                {
                    int n = 4, k = 0;
                    double[] a = Rand(rng, n, 1), b = Rand(rng, n, 1), cIn = Rand(rng, n, n);
                    var want = new double[n * n];
                    var mask = new bool[n * n];
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++) { want[i * n + j] = Beta * cIn[i * n + j]; mask[i * n + j] = j >= i; }
                    double[] phys = NativeSyr2k(dbl, CBlasUpLo.Upper, CBlasTranspose.NoTrans, n, k,
                        Pack(a, n, 1, true, 1), Pack(b, n, 1, true, 1), 1, Pack(cIn, n, n, true, n), n, true);
                    Compare(fails, $"{P(dbl)}syr2k k=0 (C:=beta*C)", Unpack(phys, n, n, true, n), want, dbl, mask);
                }
            }

            Assert.AreEqual(0, fails.Count, string.Join("\n", fails));
        }

        private static void RunTriangular(bool solve)
        {
            RequireBlas();
            var rng = new Random(solve ? 202 : 101);
            var fails = new List<string>();

            foreach (bool dbl in new[] { true, false })
                foreach (var (rm, ext) in Layouts)
                    foreach (var (m, n) in new[] { (3, 4), (5, 2), (1, 3), (2, 2) })
                        foreach (CBlasSide side in new[] { CBlasSide.Left, CBlasSide.Right })
                        {
                            int adim = side == CBlasSide.Left ? m : n;
                            double[] a = RandTri(rng, adim), b = Rand(rng, m, n);
                            int lda = Ld(adim, adim, rm, ext), ldb = Ld(m, n, rm, ext);
                            foreach (CBlasUpLo up in new[] { CBlasUpLo.Upper, CBlasUpLo.Lower })
                                foreach (CBlasTranspose tA in new[] { CBlasTranspose.NoTrans, CBlasTranspose.Trans })
                                    foreach (CBlasDiag dg in new[] { CBlasDiag.NonUnit, CBlasDiag.Unit })
                                    {
                                        double[] tri = BuildTri(a, adim, up == CBlasUpLo.Upper, dg == CBlasDiag.Unit);
                                        double[] opA = tA == CBlasTranspose.Trans ? Transpose(tri, adim, adim) : tri;
                                        double[] eff = solve ? Inverse(opA, adim) : opA; // trsm ≡ multiply by op(A)⁻¹
                                        double[] want = side == CBlasSide.Left ? MatMul(eff, adim, adim, b, m, n) : MatMul(b, m, n, eff, adim, adim);
                                        for (int i = 0; i < want.Length; i++) want[i] *= Alpha;

                                        double[] phys = NativeTriangular(dbl, solve, side, up, tA, dg, m, n,
                                            a, adim, lda, b, ldb, rm);
                                        string tag = $"{P(dbl)}{(solve ? "trsm" : "trmm")} rm={rm} ext={ext} side={side} up={up} tA={tA} dg={dg} ({m}x{n})";
                                        Compare(fails, tag, Unpack(phys, m, n, rm, ldb), want, dbl, null);
                                        CheckPad(fails, tag, phys, m, n, rm, ldb, ext);
                                    }
                        }

            Assert.AreEqual(0, fails.Count, string.Join("\n", fails));
        }

        // ---- native calls through the internal IBlasType<T> seam --------------------------------
        // Inputs are supplied logical (row-major, ld==width); packing to the physical (order, ld) layout
        // is done here so the caller only decides the layout. Returns the physical output buffer.

        private static double[] NativeTriangular(bool dbl, bool solve, CBlasSide side, CBlasUpLo uplo,
            CBlasTranspose tA, CBlasDiag diag, int m, int n, double[] a, int adim, int lda, double[] b, int ldb, bool rm)
        {
            double[] ap = Pack(a, adim, adim, rm, lda), bp = Pack(b, m, n, rm, ldb);
            CBlasOrder order = rm ? CBlasOrder.RowMajor : CBlasOrder.ColMajor;
            if (dbl)
            {
                fixed (double* pa = ap, pb = bp)
                {
                    Blas.IBlasType<double> o = default(Blas.DoubleBlas);
                    if (solve) o.Trsm(order, side, uplo, tA, diag, m, n, Alpha, pa, lda, pb, ldb);
                    else o.Trmm(order, side, uplo, tA, diag, m, n, Alpha, pa, lda, pb, ldb);
                }
                return bp;
            }
            float[] af = ToF(ap), bf = ToF(bp);
            fixed (float* pa = af, pb = bf)
            {
                Blas.IBlasType<float> o = default(Blas.SingleBlas);
                if (solve) o.Trsm(order, side, uplo, tA, diag, m, n, (float)Alpha, pa, lda, pb, ldb);
                else o.Trmm(order, side, uplo, tA, diag, m, n, (float)Alpha, pa, lda, pb, ldb);
            }
            return ToD(bf);
        }

        private static double[] NativeSymm(bool dbl, CBlasSide side, CBlasUpLo uplo, int m, int n,
            double[] ap, int lda, double[] bp, int ldb, double[] cp, int ldc, bool rm)
        {
            CBlasOrder order = rm ? CBlasOrder.RowMajor : CBlasOrder.ColMajor;
            if (dbl)
            {
                double[] a = (double[])ap.Clone(), b = (double[])bp.Clone(), c = (double[])cp.Clone();
                fixed (double* pa = a, pb = b, pc = c)
                {
                    Blas.IBlasType<double> o = default(Blas.DoubleBlas);
                    o.Symm(order, side, uplo, m, n, Alpha, pa, lda, pb, ldb, Beta, pc, ldc);
                }
                return c;
            }
            float[] af = ToF(ap), bf = ToF(bp), cf = ToF(cp);
            fixed (float* pa = af, pb = bf, pc = cf)
            {
                Blas.IBlasType<float> o = default(Blas.SingleBlas);
                o.Symm(order, side, uplo, m, n, (float)Alpha, pa, lda, pb, ldb, (float)Beta, pc, ldc);
            }
            return ToD(cf);
        }

        private static double[] NativeSyr2k(bool dbl, CBlasUpLo uplo, CBlasTranspose tr, int n, int k,
            double[] ap, double[] bp, int ldab, double[] cp, int ldc, bool rm)
        {
            CBlasOrder order = rm ? CBlasOrder.RowMajor : CBlasOrder.ColMajor;
            if (dbl)
            {
                double[] a = (double[])ap.Clone(), b = (double[])bp.Clone(), c = (double[])cp.Clone();
                fixed (double* pa = a, pb = b, pc = c)
                {
                    Blas.IBlasType<double> o = default(Blas.DoubleBlas);
                    o.Syr2k(order, uplo, tr, n, k, Alpha, pa, ldab, pb, ldab, Beta, pc, ldc);
                }
                return c;
            }
            float[] af = ToF(ap), bf = ToF(bp), cf = ToF(cp);
            fixed (float* pa = af, pb = bf, pc = cf)
            {
                Blas.IBlasType<float> o = default(Blas.SingleBlas);
                o.Syr2k(order, uplo, tr, n, k, (float)Alpha, pa, ldab, pb, ldab, (float)Beta, pc, ldc);
            }
            return ToD(cf);
        }

        // ---- physical <-> logical packing (honours order + leading dimension) --------------------

        private static int Ld(int rows, int cols, bool rowMajor, int ext) => (rowMajor ? cols : rows) + ext;

        /// <summary>Pack a logical r×c row-major matrix into physical storage of the given order + ld,
        /// filling the ld padding with a sentinel so a later check proves BLAS left it untouched.</summary>
        private static double[] Pack(double[] l, int r, int c, bool rowMajor, int ld)
        {
            int outer = rowMajor ? r : c, inner = rowMajor ? c : r;
            var p = new double[outer * ld];
            for (int i = 0; i < p.Length; i++) p[i] = Sentinel;
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    p[rowMajor ? i * ld + j : j * ld + i] = l[i * c + j];
            return p;
        }

        private static double[] Unpack(double[] p, int r, int c, bool rowMajor, int ld)
        {
            var l = new double[r * c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    l[i * c + j] = p[rowMajor ? i * ld + j : j * ld + i];
            return l;
        }

        private static void CheckPad(List<string> fails, string tag, double[] p, int r, int c, bool rowMajor, int ld, int ext)
        {
            if (ext == 0) return;
            int outer = rowMajor ? r : c, inner = rowMajor ? c : r;
            for (int o = 0; o < outer; o++)
                for (int k = inner; k < ld; k++)
                    if (p[o * ld + k] != Sentinel) { fails.Add($"{tag}: BLAS wrote into ld padding"); return; }
        }

        // ---- managed reference math (row-major) --------------------------------------------------

        private static string P(bool dbl) => dbl ? "d" : "s";
        private static float[] ToF(double[] a) => Array.ConvertAll(a, x => (float)x);
        private static double[] ToD(float[] a) => Array.ConvertAll(a, x => (double)x);

        private static double[] Rand(Random rng, int r, int c)
        {
            var a = new double[r * c];
            for (int i = 0; i < a.Length; i++) a[i] = Math.Round(rng.NextDouble() * 2 - 1, 3);
            return a;
        }

        private static double[] RandTri(Random rng, int n)
        {
            var a = Rand(rng, n, n);
            for (int i = 0; i < n; i++) a[i * n + i] = 2.0 + rng.NextDouble(); // nonzero diagonal (invertible)
            return a;
        }

        private static double[] MatMul(double[] a, int ar, int ac, double[] b, int br, int bc)
        {
            var c = new double[ar * bc];
            for (int i = 0; i < ar; i++)
                for (int j = 0; j < bc; j++)
                {
                    double s = 0;
                    for (int p = 0; p < ac; p++) s += a[i * ac + p] * b[p * bc + j];
                    c[i * bc + j] = s;
                }
            return c;
        }

        private static double[] Transpose(double[] a, int r, int c)
        {
            var t = new double[r * c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    t[j * r + i] = a[i * c + j];
            return t;
        }

        private static double[] BuildTri(double[] a, int n, bool upper, bool unit)
        {
            var e = new double[n * n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double v;
                    if (i == j) v = unit ? 1.0 : a[i * n + j];
                    else if (upper) v = j > i ? a[i * n + j] : 0.0;
                    else v = j < i ? a[i * n + j] : 0.0;
                    e[i * n + j] = v;
                }
            return e;
        }

        private static double[] BuildSym(double[] a, int n, bool upper)
        {
            var e = new double[n * n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    int ii = i, jj = j;
                    if (upper) { if (j < i) { ii = j; jj = i; } }
                    else { if (j > i) { ii = j; jj = i; } }
                    e[i * n + j] = a[ii * n + jj];
                }
            return e;
        }

        private static double[] Inverse(double[] m, int n)
        {
            var a = (double[])m.Clone();
            var inv = new double[n * n];
            for (int i = 0; i < n; i++) inv[i * n + i] = 1;
            for (int c = 0; c < n; c++)
            {
                double piv = a[c * n + c];
                for (int j = 0; j < n; j++) { a[c * n + j] /= piv; inv[c * n + j] /= piv; }
                for (int r = 0; r < n; r++)
                    if (r != c)
                    {
                        double f = a[r * n + c];
                        for (int j = 0; j < n; j++) { a[r * n + j] -= f * a[c * n + j]; inv[r * n + j] -= f * inv[c * n + j]; }
                    }
            }
            return inv;
        }

        private static void Compare(List<string> fails, string tag, double[] got, double[] want, bool dbl, bool[] mask)
        {
            double tol = dbl ? 1e-9 : 2e-3, maxd = 0;
            for (int i = 0; i < want.Length; i++)
            {
                if (mask != null && !mask[i]) continue;
                double d = Math.Abs(got[i] - want[i]) / (1 + Math.Abs(want[i]));
                if (d > maxd) maxd = d;
            }
            if (maxd > tol) fails.Add($"{tag}: maxRelDiff={maxd:E3} > {tol:E1}");
        }
    }
}
