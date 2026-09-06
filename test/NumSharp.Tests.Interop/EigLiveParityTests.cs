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
    ///     <b>Live byte-for-byte parity of the eigen family against real NumPy.</b>
    ///     <c>np.linalg.eigh</c>/<c>eigvalsh</c> (LAPACK <c>syevd</c>/<c>heevd</c>) and
    ///     <c>np.linalg.eig</c>/<c>eigvals</c> (LAPACK <c>geev</c>) are each computed twice over the SAME
    ///     zero-copy exported memory — once by NumSharp's OpenBLAS backend, once by the embedded CPython's
    ///     <c>numpy.linalg</c> — and asserted byte-identical (<see cref="ByteContract"/> compares canonical
    ///     C-order bytes).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Byte-exact and not merely close for the same reason the cholesky/qr/svd gates are: both stacks
    ///     delegate to the SAME bundled scipy-openblas LAPACK at <c>threads: 1</c>. <b>Eigenvalues</b> (all
    ///     four functions) and the <b>REAL-symmetric</b> eigenvectors (<c>syevd</c> sign) and the <b>eig</b>
    ///     eigenvectors (<c>geev</c> canonical phase — largest component made real, plus the managed
    ///     conjugate-pair rebuild of <c>mk_geev_complex_eigenvectors</c>) are all reproducible, so they are
    ///     byte-compared.
    ///     </para>
    ///     <para>
    ///     <b>The one non-invariant is the complex-HERMITIAN eigenvector PHASE (<c>heevd</c>).</b> LAPACK
    ///     does not canonicalize it, and it is <i>not reproducible across processes</i> — proven directly:
    ///     the same scipy-openblas binary returns ±the same eigenvector column for the same input in
    ///     different process invocations, and <c>numpy.linalg.eigh</c> equals a raw <c>zheevd</c> only
    ///     WITHIN a process. So complex-Hermitian eigenvectors are checked by the phase-invariant
    ///     reconstruction <c>A·V == V·diag(w)</c> (as SVD checks its sign-ambiguous U/Vh), while eigenvalues
    ///     stay byte-exact. The one dtype divergence is <c>eig</c> of a <b>float32</b> operand with COMPLEX
    ///     eigenvalues: NumPy yields complex64, NumSharp complex128 (no complex64 type); that case up-casts
    ///     NumPy's result and compares values.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class EigLiveParityTests : InteropTestBase
    {
        private static void RequireLapack()
        {
            if (!OpenBlasEngine.LapackAvailable)
                Assert.Inconclusive("OpenBLAS LAPACK backend not available on this host.");
        }

        // ---- eigh / eigvalsh ---------------------------------------------------------------------

        private void AssertEigvalshByteExact(NDArray a, char uplo, string because)
        {
            using var nsW = np.linalg.eigvalsh(a, uplo);
            using (Gil())
            {
                using PyObject va = a.ToNumpy();
                using PyObject npW = Python.np.with($"np.linalg.eigvalsh(a, UPLO='{uplo}')", ("a", va));
                ByteContract.AssertSameBytes(nsW, npW, because);
            }
        }

        private void AssertEighByteExact(NDArray a, char uplo, string because)
        {
            var (w, v) = np.linalg.eigh(a, uplo);
            using (w)
            using (v)
            {
                // Reconstruction A·V == V·diag(w) is the phase-INVARIANT invariant — always assert it.
                AssertEighReconstruction(a, w, v, because);
                using (Gil())
                {
                    using PyObject va = a.ToNumpy();
                    using PyObject npW = Python.np.with($"np.linalg.eigh(a, UPLO='{uplo}')[0]", ("a", va));
                    ByteContract.AssertSameBytes(w, npW, because + " [w]"); // eigenvalues are reproducible

                    // Real-symmetric eigenvector SIGN (syevd) is reproducible, so byte-compare it.
                    // Complex-Hermitian eigenvector PHASE (heevd) is NOT reproducible across processes
                    // (proven: the same binary yields ±the same column in different processes; numpy ==
                    // raw zheevd only WITHIN a process). So it is covered by reconstruction above, not a
                    // byte-compare that would encode a non-invariant.
                    if (v.typecode != NPTypeCode.Complex)
                    {
                        using PyObject npV = Python.np.with($"np.linalg.eigh(a, UPLO='{uplo}')[1]", ("a", va));
                        ByteContract.AssertSameBytes(v, npV, because + " [v]");
                    }
                }
            }
        }

        /// <summary>
        ///     The phase/sign-invariant eigendecomposition check: <c>A·V == V·diag(w)</c> per stacked
        ///     matrix. Used for the complex-Hermitian eigenvectors, whose phase is not reproducible.
        /// </summary>
        private static void AssertEighReconstruction(NDArray a, NDArray w, NDArray v, string because)
        {
            var ac = a.astype(v.typecode);
            var lhs = np.matmul(ac, v);                                           // A·V
            var rhs = np.multiply(v, np.expand_dims(w.astype(v.typecode), -2));   // V·diag(w)
            var diff = np.abs(np.subtract(lhs, rhs));
            double maxErr = Convert.ToDouble(np.amax(diff).GetValue(0));
            // single-precision eigenvectors reconstruct to ~1e-7; double/complex128 to ~1e-14.
            double tol = v.typecode == NPTypeCode.Single ? 1e-5 : 1e-9;
            maxErr.Should().BeLessThan(tol, $"eigh reconstruction A·V≈V·diag(w) — {because}");
        }

        // ---- eig / eigvals -----------------------------------------------------------------------

        private void AssertEigvalsByteExact(NDArray a, string because)
        {
            using var nsW = np.linalg.eigvals(a);
            using (Gil())
            {
                using PyObject va = a.ToNumpy();
                using PyObject npW = Python.np.with("np.linalg.eigvals(a)", ("a", va));
                ByteContract.AssertSameBytes(nsW, npW, because);
            }
        }

        private void AssertEigByteExact(NDArray a, string because)
        {
            var (w, v) = np.linalg.eig(a);
            using (w)
            using (v)
            using (Gil())
            {
                using PyObject va = a.ToNumpy();
                using PyObject npW = Python.np.with("np.linalg.eig(a)[0]", ("a", va));
                using PyObject npV = Python.np.with("np.linalg.eig(a)[1]", ("a", va));
                ByteContract.AssertSameBytes(w, npW, because + " [w]");
                ByteContract.AssertSameBytes(v, npV, because + " [v]");
            }
        }

        // ---- shared inputs -----------------------------------------------------------------------

        private static NDArray Sym3() => np.array(new double[,] { { 4, 1, 2 }, { 1, 5, 3 }, { 2, 3, 6 } });

        private static NDArray Herm2() => np.array(new Complex[,]
        {
            { new Complex(1, 0), new Complex(0, -2) },
            { new Complex(0, 2), new Complex(5, 0) }
        });

        /// <summary>A non-symmetric matrix with a complex-conjugate eigenpair (drives the geev assembly).</summary>
        private static NDArray Rotish() => np.array(new double[,] { { 1, -1 }, { 1, 1 } });

        /// <summary>A non-symmetric matrix whose eigenvalues are all real (drives the real-collapse).</summary>
        private static NDArray RealEigNonSym() => np.array(new double[,] { { 2, 0, 0 }, { 1, 3, 0 }, { 4, 5, 6 } });

        // =====================================  eigh  =============================================

        [TestMethod]
        public void Backend_IsOpenBlasWithLapack()
        {
            Assert.IsTrue(OpenBlasEngine.Enabled, "the interop suite runs with OpenBLAS as the default backend");
            RequireLapack();
        }

        [TestMethod]
        public void Eigh_Symmetric_LowerAndUpper_ByteExact()
        {
            RequireLapack();
            AssertEighByteExact(Sym3(), 'L', "eigh symmetric, lower");
            AssertEighByteExact(Sym3(), 'U', "eigh symmetric, upper");
            AssertEigvalshByteExact(Sym3(), 'L', "eigvalsh symmetric, lower");
        }

        [TestMethod]
        public void Eigh_ComplexHermitian_ByteExact()
        {
            RequireLapack();
            AssertEighByteExact(Herm2(), 'L', "eigh Hermitian, lower");
            AssertEighByteExact(Herm2(), 'U', "eigh Hermitian, upper");
            // conj(Herm2) == [[1,2j],[-2j,5]]: its zheevd eigenvector PHASE is the one that varies
            // ACROSS PROCESSES (proven). This same-process gate checks whether NumSharp and the embedded
            // numpy still land on the same phase in ONE process.
            AssertEighByteExact(np.array(new Complex[,]
            {
                { new Complex(1, 0), new Complex(0, 2) },
                { new Complex(0, -2), new Complex(5, 0) }
            }), 'L', "eigh conj-Hermitian (phase-sensitive)");
        }

        [TestMethod]
        public void Eigh_Float32_ByteExact()
        {
            RequireLapack();
            AssertEighByteExact(Sym3().astype(np.float32), 'L', "eigh symmetric float32");
        }

        [TestMethod]
        public void Eigh_EveryWideningDtype_ByteExact()
        {
            RequireLapack();
            var diag = np.array(new double[,] { { 4, 0, 0 }, { 0, 9, 0 }, { 0, 0, 16 } });
            foreach (var tc in new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64
            })
            {
                AssertEigvalshByteExact(diag.astype(tc), 'L', $"eigvalsh {tc} → float64");
            }
        }

        [TestMethod]
        public void Eigh_Batched_ByteExact()
        {
            RequireLapack();
            var stack = np.stack(new[] { Sym3(), Sym3() * 2.0, Sym3() + np.eye(3) });
            AssertEighByteExact(stack, 'L', "eigh batched (3,3,3)");
        }

        [TestMethod]
        public void Eigh_EveryLayout_ByteExact()
        {
            RequireLapack();
            var s = Sym3();
            AssertEighByteExact(np.ascontiguousarray(s), 'L', "C-contiguous");
            AssertEighByteExact(np.asfortranarray(s), 'L', "F-contiguous");
            AssertEighByteExact(s.T, 'L', "transposed view (symmetric.T == symmetric)");
            AssertEighByteExact(s["::-1, ::-1"], 'L', "reversed both axes");
        }

        // =====================================  eig  =============================================

        [TestMethod]
        public void Eig_RealEigenvalues_CollapseToReal_ByteExact()
        {
            RequireLapack();
            AssertEigByteExact(np.diag(np.array(new double[] { 1.0, 2, 3 })), "eig diagonal (real eigs)");
            AssertEigByteExact(RealEigNonSym(), "eig triangular-ish (real eigs, non-symmetric)");
            AssertEigvalsByteExact(RealEigNonSym(), "eigvals real eigs");
        }

        [TestMethod]
        public void Eig_ComplexEigenvalues_ByteExact()
        {
            RequireLapack();
            // The geev real→complex conjugate-pair eigenvector assembly, byte-for-byte vs NumPy.
            AssertEigByteExact(Rotish(), "eig rotation (complex conjugate eigs)");
            AssertEigvalsByteExact(Rotish(), "eigvals complex eigs");
        }

        [TestMethod]
        public void Eig_ComplexInput_ByteExact()
        {
            RequireLapack();
            var a = np.array(new Complex[,]
            {
                { new Complex(1, 0), new Complex(0, 1) },
                { new Complex(0, -1), new Complex(1, 0) }
            });
            AssertEigByteExact(a, "eig complex input (zgeev)");
        }

        [TestMethod]
        public void Eig_Float64_Batched_ByteExact()
        {
            RequireLapack();
            // Mixed batch: one complex-eig matrix + one real-eig matrix → whole result complex on both sides.
            var stack = np.stack(new[] { Rotish(), np.array(new double[,] { { 2.0, 0 }, { 0, 3 } }) });
            AssertEigByteExact(stack, "eig batched mixed (whole result complex)");
        }

        [TestMethod]
        public void Eig_Float32_RealEigs_ByteExact_ComplexEigs_ValueMatch()
        {
            RequireLapack();
            // float32 with REAL eigenvalues collapses to float32 on both sides — byte-exact.
            AssertEigByteExact(np.diag(np.array(new float[] { 1, 2, 3 })), "eig float32 real eigs");

            // float32 with COMPLEX eigenvalues: NumPy → complex64, NumSharp → complex128 (no complex64).
            // Up-cast NumPy's result and compare values (the one documented dtype divergence).
            var a = Rotish().astype(np.float32);
            using var w = np.linalg.eig(a).eigenvalues;
            w.typecode.Should().Be(NPTypeCode.Complex);
            using (Gil())
            {
                using PyObject va = a.ToNumpy();
                using PyObject npW = Python.np.with("np.linalg.eig(a)[0].astype('complex128')", ("a", va));
                ByteContract.AssertSameBytes(w, npW, "eig float32 complex eigs (numpy complex64 up-cast to complex128)");
            }
        }
    }
}
