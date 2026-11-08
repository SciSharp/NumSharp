using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The matrix tier (<c>test/oracle/gen_matrix_oracle.py</c>): NumPy 2.4.2's answers for <c>numpy.matrix</c>,
    ///     replayed against <see cref="NDMatrix"/> — the constructor (<c>np.matrix</c>, <c>np.asmatrix</c> from
    ///     arrays of every rank, matrices, literals and lists), the matrix-only members (<c>T H A A1 I</c>), the row /
    ///     column / scalar rule of indexing and the assignment rule that follows from it, <c>*</c> as the matrix
    ///     product with scaling in <c>np.dot</c>'s dtype, the matrix power, the elementwise operators, the reductions
    ///     that keep two axes, the methods <c>ndarray</c> hands down, <c>repr</c> / <c>str</c> and iteration.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A matrix result is more than its values, so every case also holds NumSharp to the CLASS NumPy returned
    ///     (<see cref="FuzzCorpus.Expected.Cls"/>: a matrix, a plain array or a scalar) and to the memory facet
    ///     (<see cref="FuzzCorpus.MemFacet"/>: view, copy or identity against each operand, the five flags, the byte
    ///     strides). The calls NumPy refuses are held to its exception type and verbatim text.
    ///     </para>
    ///     <para>
    ///     The tier takes no registry excuse (valueclass <c>matrix</c>, see <see cref="MisalignedRegistry"/>). Known
    ///     divergences are carved by the generator into <c>gaps/matrix_gaps.jsonl</c> under named gap ids and replayed
    ///     by <see cref="Matrix_KnownGaps"/>, which fails while any of them still differs from NumPy.
    ///     </para>
    /// </remarks>
    public partial class FuzzCorpusTests
    {
        /// <summary>
        ///     The matrix tier: every case equal to NumPy in values, class, memory facet, or error type and text.
        ///     A float scaling (<c>m * 2.0</c>) takes its production route here: the direct pass for a
        ///     C-contiguous float operand (the counter proves the corpus reaches it) and, the corpus's matrices
        ///     being small, the two ufunc passes for every other operand.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Matrix()
        {
            int before = NDMatrix.DirectScaleRuns;
            RunCorpus("matrix.jsonl");
            Assert.IsTrue(NDMatrix.DirectScaleRuns - before > 20,
                "the direct scaling pass ran only " + (NDMatrix.DirectScaleRuns - before) + " times");
        }

        /// <summary>
        ///     The matrix tier again with the fused scaling pass FORCED for every float or complex matrix of two
        ///     or more elements: the direct pass is switched off and
        ///     <c>NDMatrix.FusedScaleMinElementsTestOverride</c> is 2. In production a C-contiguous float operand
        ///     never takes that pass, a float view takes it from 4,096 elements on and a complex matrix from 512,
        ///     which the corpus's small matrices never reach. The answers must still be NumPy's to the bit, the
        ///     sign of every zero included, and the counter proves the pass ran.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Matrix_FusedScaling()
        {
            int before = NDMatrix.FusedScaleRuns;
            NDMatrix.DisableDirectScale = true;
            NDMatrix.FusedScaleMinElementsTestOverride = 2;
            try
            {
                RunCorpus("matrix.jsonl");
            }
            finally
            {
                NDMatrix.DisableDirectScale = false;
                NDMatrix.FusedScaleMinElementsTestOverride = 0;
            }

            Assert.IsTrue(NDMatrix.FusedScaleRuns - before > 50,
                "the forced fused scaling pass ran only " + (NDMatrix.FusedScaleRuns - before) + " times");
        }

        /// <summary>
        ///     The matrix tier a third time with BOTH single-pass routes switched off, so every float and complex
        ///     scaling runs the multiply ufunc and then the in-place addition of zero. In production that route
        ///     serves operands of another dtype, small float views and hosts without dynamic code; this replay
        ///     holds it to NumPy for the C-contiguous and complex operands the other routes normally take.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Matrix_TwoPassScaling()
        {
            int direct = NDMatrix.DirectScaleRuns, fused = NDMatrix.FusedScaleRuns;
            NDMatrix.DisableDirectScale = true;
            NDMatrix.FusedScaleMinElementsTestOverride = long.MaxValue;
            try
            {
                RunCorpus("matrix.jsonl");
            }
            finally
            {
                NDMatrix.DisableDirectScale = false;
                NDMatrix.FusedScaleMinElementsTestOverride = 0;
            }

            Assert.AreEqual(0, (NDMatrix.DirectScaleRuns - direct) + (NDMatrix.FusedScaleRuns - fused),
                "a single-pass scaling route ran although both were switched off");
        }

        /// <summary>The carved gaps of <see cref="Matrix"/>: fails while any of them still diverges from NumPy.</summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Matrix_KnownGaps() => RunGapCorpus("matrix_gaps.jsonl");
    }
}
