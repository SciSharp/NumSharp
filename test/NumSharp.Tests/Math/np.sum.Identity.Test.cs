using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends.Iteration;
using static NumSharp.Tests.Utilities.RawBitCases;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     How NumPy STARTS an add reduction, pinned to the bit: <c>np.add.reduce</c> passes the dtype's identity
    ///     (<c>+0.0</c>) as the initial value even for a non-empty input, so <c>np.sum</c> is <c>+0.0 + x0 + x1 …</c>.
    ///     A sum of only <c>-0.0</c> values is therefore <c>+0.0</c>, and a signalling NaN comes back quiet — also for a
    ///     single element and over a length-1 axis, where the result is one addition, never a copy. Every expected
    ///     string is NumPy 2.4.2's output for the same call, probed on 2026-10-03, in
    ///     <see cref="NumSharp.Tests.Utilities.RawBitCases.Describe"/>'s format (<c>dtype (shape) bits…</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The rule reaches every function built on the add reduction: <c>np.mean</c>/<c>np.average</c>
    ///     (<c>umr_sum / n</c>), <c>np.nansum</c>/<c>np.nanmean</c> (the sum after the NaN replacement),
    ///     <c>np.einsum</c>'s single-term sum, <c>np.trapezoid</c>'s one-interval area and <c>np.evaluate</c>'s
    ///     <c>Sum</c>/<c>Mean</c>. Nine of these tests were <c>[OpenBugs]</c> reproductions in
    ///     <c>OpenBugs.Reductions.cs</c> (commit 00f29299) until the identity add landed; the 2026-10-04
    ///     reduction-start probe (c45fb4ed, the <c>reduce_start</c> oracle tier) extended several of them and pinned
    ///     the same fixes from the other side (<see cref="NanSum_LengthOneAxis_IsPositiveZero"/> and the cases each
    ///     test's remarks name moved here from it). The rest pin the parameters and paths the same change touched:
    ///     <c>dtype=</c> loops, the 0-d axis rules, broadcast operands, NumPy's complex mean division, the strided
    ///     float16 NaN-axis walk, and the reduction kernel cache.
    ///     </para>
    ///     <para>
    ///     The rules that are NOT this one stay in <c>OpenBugs.Reductions.cs</c>: <c>add.accumulate</c>
    ///     (<c>np.cumsum</c>) copies its first element instead of adding it to the identity, <c>np.dot</c> of a size-1
    ///     operand is NumPy's scalar branch, a lone complex <c>np.prod</c> element is multiplied by <c>1+0j</c>,
    ///     <c>np.median</c> is <c>np.mean</c> of the middle element(s), and einsum's bare elementwise products start
    ///     from a zero-filled output.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class np_sum_identity_tests
    {
        // ======================================================================================================
        //  The identity add itself: one element, a length-1 axis, the float16 flat route.
        // ======================================================================================================

        /// <summary>
        ///     <c>np.sum</c> of one element (a one-element 1-D array or a 0-d array, any float dtype, with or without a
        ///     widening <c>dtype=</c>) is <c>+0.0 + x</c>, so a <c>-0.0</c> element gives <c>+0.0</c>.
        /// </summary>
        /// <remarks>
        ///     The size-1 route (<c>Default.Reduction.Add.Identity.cs</c> <c>SumOfOneElement</c>) adds the element to a
        ///     0-d identity through the elementwise add kernel. It used to share <c>HandleScalarReduction</c> with
        ///     <c>amin</c>/<c>amax</c>, which must copy (a min/max has no identity), and so returned <c>-0.0</c>.
        ///     <c>ndarray.sum()</c> takes the same route.
        /// </remarks>
        [TestMethod]
        public void Sum_OneElement_IsPositiveZero_LikeNumPysAddReduceIdentity()
        {
            var fails = new List<string>();
            Check(fails, "sum f64 [-0]", () => np.sum(F64(1)), "float64 () 0000000000000000");
            Check(fails, "sum f32 [-0]", () => np.sum(F32(1)), "float32 () 00000000");
            Check(fails, "sum f16 [-0]", () => np.sum(F16(1)), "float16 () 0000");
            Check(fails, "sum c128 [-0-0j]", () => np.sum(C128(1)), "complex128 () 0000000000000000 0000000000000000");
            Check(fails, "sum f64 0-d -0", () => np.sum(NDArray.Scalar(-0.0)), "float64 () 0000000000000000");
            Check(fails, "ndarray.sum f64 [-0]", () => F64(1).sum(), "float64 () 0000000000000000");
            Check(fails, "sum f32 [-0] dtype=f64", () => np.sum(F32(1), np.float64), "float64 () 0000000000000000");
            AssertNoMismatch(fails, "NumPy's add.reduce adds the lone element to the +0.0 identity");
        }

        /// <summary>
        ///     <c>np.sum</c> over an axis of length 1 returns <c>+0.0 + x</c> for every output, so a slab of
        ///     <c>-0.0</c> values gives <c>+0.0</c>, for every float dtype, both axis positions and <c>keepdims</c>.
        /// </summary>
        /// <remarks>
        ///     The length-1 axis route (<c>SumOverLengthOneAxis</c>) writes the slab into a KEEPORDER result and adds the
        ///     identity in place with one SIMD pass. It used to copy the slab through <c>HandleTrivialAxisReduction</c>,
        ///     which <c>amin</c>/<c>amax</c>/<c>nanmin</c>/<c>nanmax</c> still use (they must copy).
        /// </remarks>
        [TestMethod]
        public void Sum_LengthOneAxis_IsPositiveZero()
        {
            var fails = new List<string>();
            const string z64x3 = "0000000000000000 0000000000000000 0000000000000000";
            Check(fails, "sum f64 (1,3) axis0", () => np.sum(F64(3).reshape(1, 3), axis: 0), "float64 (3,) " + z64x3);
            Check(fails, "sum f64 (3,1) axis1", () => np.sum(F64(3).reshape(3, 1), axis: 1), "float64 (3,) " + z64x3);
            Check(fails, "sum f64 (1,3) axis0 keepdims", () => np.sum(F64(3).reshape(1, 3), axis: 0, keepdims: true),
                  "float64 (1,3) " + z64x3);
            Check(fails, "sum f32 (1,3) axis0", () => np.sum(F32(3).reshape(1, 3), axis: 0), "float32 (3,) 00000000 00000000 00000000");
            Check(fails, "sum f16 (1,3) axis0", () => np.sum(F16(3).reshape(1, 3), axis: 0), "float16 (3,) 0000 0000 0000");
            Check(fails, "sum c128 (1,3) axis0", () => np.sum(C128(3).reshape(1, 3), axis: 0),
                  "complex128 (3,) " + z64x3 + " " + z64x3);
            AssertNoMismatch(fails, "NumPy reduces a length-1 axis as +0.0 + x, it does not copy the slab");
        }

        /// <summary>
        ///     The flat <c>np.sum</c> of several float16 <c>-0.0</c> values is <c>+0.0</c>, as it is for float32 and
        ///     float64.
        /// </summary>
        /// <remarks>
        ///     The float16 flat sum runs its own float32-shadow route (<c>SumElementwiseHalfFallback</c>), which now
        ///     starts from the identity (<c>0f + pairwise</c>) the way NumPy's HALF_add reduce does; it used to return
        ///     the pairwise sum alone, whose leaf keeps a <c>-0.0</c>.
        /// </remarks>
        [TestMethod]
        public void Sum_Float16Flat_OfNegativeZeros_IsPositiveZero()
        {
            var fails = new List<string>();
            Check(fails, "sum f16 [-0]*3", () => np.sum(F16(3)), "float16 () 0000");
            Check(fails, "sum f16 [-0]*100", () => np.sum(F16(100)), "float16 () 0000");
            AssertNoMismatch(fails, "NumPy's HALF_add reduce starts from the +0.0 identity");
        }

        /// <summary>
        ///     <c>np.mean</c> (and the unweighted <c>np.average</c>, which is <c>np.mean</c>) of one element, or over a
        ///     length-1 axis, is <c>(+0.0 + x) / n</c>, so <c>-0.0</c> gives <c>+0.0</c>.
        /// </summary>
        /// <remarks>
        ///     NumPy's mean is <c>umr_sum(...) / n</c>, so it inherits the sum's <c>+0.0</c> start. NumSharp's mean
        ///     reuses the sum's one-element and length-1 axis routes with a count division (a no-op for every real
        ///     dtype, NumPy's complex division for complex — see <see cref="Mean_Complex_DividesLikeNumPysComplexLoop"/>).
        ///     The <c>average</c> axis and complex cases came from the 2026-10-04 reduction-start probe (c45fb4ed),
        ///     which pinned them while the copy was still there.
        /// </remarks>
        [TestMethod]
        public void Mean_OneElementOrLengthOneAxis_IsPositiveZero()
        {
            var fails = new List<string>();
            Check(fails, "mean f64 [-0]", () => np.mean(F64(1)), "float64 () 0000000000000000");
            Check(fails, "mean f32 [-0]", () => np.mean(F32(1)), "float32 () 00000000");
            Check(fails, "mean f16 [-0]", () => np.mean(F16(1)), "float16 () 0000");
            Check(fails, "mean f64 (1,3) axis0", () => np.mean(F64(3).reshape(1, 3), axis: 0),
                  "float64 (3,) 0000000000000000 0000000000000000 0000000000000000");
            Check(fails, "average f64 [-0]", () => np.average(F64(1)), "float64 () 0000000000000000");
            Check(fails, "average f64 (1,3) axis0", () => np.average(F64(3).reshape(1, 3), 0),
                  "float64 (3,) 0000000000000000 0000000000000000 0000000000000000");
            Check(fails, "mean c128 [-0-0j]", () => np.mean(C128(1)), "complex128 () 0000000000000000 0000000000000000");
            AssertNoMismatch(fails, "NumPy's mean is the +0.0-started sum divided by the count");
        }

        /// <summary>
        ///     The fused <c>np.evaluate(NDExpr.Sum(x, axis))</c> and <c>NDExpr.Mean(x, axis)</c> over a length-1 axis
        ///     give <c>+0.0</c> for a <c>-0.0</c> slab, as <c>np.sum</c>/<c>np.mean</c> do in NumPy; so do the
        ///     sum-based <c>NanSum</c> and <c>NanMean</c> kinds, for float32 and complex128 too.
        /// </summary>
        /// <remarks>
        ///     <c>DefaultEngine.Evaluate.Stream.cs</c> produces an extent-1 axis straight into the result and then adds
        ///     the identity in place (<c>WriteIdentityPlus</c>); it used to stop after the produce, which mirrored the
        ///     engine's old copy. The two routes must stay together — the engine's is pinned by
        ///     <see cref="Sum_LengthOneAxis_IsPositiveZero"/>. The NanSum / NanMean / float32 / complex128 cases came
        ///     from the 2026-10-04 reduction-start probe (c45fb4ed); the complex <c>Mean</c> case also shows the count
        ///     division, which keeps the imaginary <c>+0.0</c>.
        /// </remarks>
        [TestMethod]
        public void Evaluate_SumOverLengthOneAxis_IsPositiveZero()
        {
            var fails = new List<string>();
            const string z64x3 = "0000000000000000 0000000000000000 0000000000000000";
            Check(fails, "evaluate Sum f64 (1,3) axis0", () => np.evaluate(NDExpr.Sum((NDExpr)F64(3).reshape(1, 3), 0)),
                  "float64 (3,) " + z64x3);
            Check(fails, "evaluate Mean f64 (1,3) axis0", () => np.evaluate(NDExpr.Mean((NDExpr)F64(3).reshape(1, 3), 0)),
                  "float64 (3,) " + z64x3);
            Check(fails, "evaluate NanSum f64 (3,1) axis1", () => np.evaluate(NDExpr.NanSum((NDExpr)F64(3).reshape(3, 1), 1)),
                  "float64 (3,) " + z64x3);
            Check(fails, "evaluate NanMean f32 (1,1) axis0", () => np.evaluate(NDExpr.NanMean((NDExpr)F32(1).reshape(1, 1), 0)),
                  "float32 (1,) 00000000");
            Check(fails, "evaluate Sum c128 (1,3) axis0 keepdims",
                  () => np.evaluate(NDExpr.Sum((NDExpr)C128(3).reshape(1, 3), 0, true)),
                  "complex128 (1,3) " + z64x3 + " " + z64x3);
            Check(fails, "evaluate Mean c128 (1,1) axis1", () => np.evaluate(NDExpr.Mean((NDExpr)C128(1).reshape(1, 1), 1)),
                  "complex128 (1,) 0000000000000000 0000000000000000");
            AssertNoMismatch(fails, "np.evaluate's sum-based kinds must match NumPy's, which start from +0.0");
        }

        /// <summary>
        ///     <c>np.trapezoid</c> over two samples sums ONE product along the integration axis, so a <c>-0.0</c> area
        ///     comes back as <c>+0.0</c> (the 18 cases of the 7,430-case trapezoid differential that differed after
        ///     94c9eb6b).
        /// </summary>
        /// <remarks>
        ///     trapezoid ends in <c>np.sum(…, axis)</c> over a length-1 axis, so it inherits the identity add; the
        ///     <c>-0.0</c> product comes from a negative spacing times a <c>+0.0</c> sum (<c>-1 * (0 + 0) / 2</c>), or
        ///     from a positive spacing times a <c>-0.0</c> sum. The last five cases (a <c>-0.0</c> y, the last axis of
        ///     a 2-D y, float32 / float16 / complex128) came from the 2026-10-04 reduction-start probe (c45fb4ed).
        /// </remarks>
        [TestMethod]
        public void Trapezoid_OneInterval_NegativeZeroArea_IsPositiveZero()
        {
            var fails = new List<string>();
            Check(fails, "trapezoid [0,0] x=[1,0]", () => np.trapezoid(np.array(new[] { 0.0, 0.0 }), np.array(new[] { 1.0, 0.0 })),
                  "float64 () 0000000000000000");
            Check(fails, "trapezoid [0,0] dx=-1", () => np.trapezoid(np.array(new[] { 0.0, 0.0 }), null, -1.0),
                  "float64 () 0000000000000000");
            Check(fails, "trapezoid (2,3) x=[1,0] axis0", () => np.trapezoid(np.zeros(new Shape(2, 3)), np.array(new[] { 1.0, 0.0 }), 1.0, 0),
                  "float64 (3,) 0000000000000000 0000000000000000 0000000000000000");
            Check(fails, "trapezoid [-0,-0] dx=1", () => np.trapezoid(F64(2), null, 1.0), "float64 () 0000000000000000");
            Check(fails, "trapezoid (3,2) dx=-1 axis1", () => np.trapezoid(np.zeros(new Shape(3, 2)), null, -1.0, 1),
                  "float64 (3,) 0000000000000000 0000000000000000 0000000000000000");
            Check(fails, "trapezoid f32 [0,0] dx=-1", () => np.trapezoid(np.zeros(new Shape(2), np.float32), null, -1.0),
                  "float32 () 00000000");
            Check(fails, "trapezoid f16 (2,3) dx=-1 axis0", () => np.trapezoid(np.zeros(new Shape(2, 3), np.float16), null, -1.0, 0),
                  "float16 (3,) 0000 0000 0000");
            Check(fails, "trapezoid c128 [-0-0j]*2 dx=1", () => np.trapezoid(C128(2), null, 1.0),
                  "complex128 () 0000000000000000 0000000000000000");
            AssertNoMismatch(fails, "NumPy's trapezoid sums its one product from the +0.0 identity");
        }

        /// <summary>
        ///     <c>np.einsum</c> summing a single term per output gives <c>+0.0</c> for <c>-0.0</c>: one-operand sums
        ///     (<c>'i->'</c>, <c>'ij->'</c>, <c>'ii->'</c> over one element, <c>'ij->j'</c> / <c>'ij->i'</c> over a
        ///     length-1 axis) and two-operand contractions over a length-1 index (<c>'i,i->'</c>, <c>'ij,j->i'</c>,
        ///     <c>'ij,jk->ik'</c>).
        /// </summary>
        /// <remarks>
        ///     NumPy's default einsum adds every term into a zero-filled output (<c>sum_of_products</c>), so each output
        ///     is <c>+0.0 + term</c>. NumSharp's composition reaches these through <c>np.sum</c> (or a product that sums
        ///     from zero), so they inherit the identity add. The two-operand cases are the passing half of the
        ///     2026-10-04 reduction-start probe's <c>Einsum_ZeroFilledAccumulator_GivesPositiveZero</c> pin
        ///     (c45fb4ed); the other half, the bare elementwise products <c>'i,i->i'</c> / <c>'i,j->ij'</c>, stays in
        ///     <c>OpenBugs.Reductions.cs</c>.
        /// </remarks>
        [TestMethod]
        public void Einsum_SumOfOneTerm_IsPositiveZero()
        {
            var fails = new List<string>();
            const string z64 = "0000000000000000";
            Check(fails, "einsum i-> [-0]", () => np.einsum("i->", F64(1)), "float64 () 0000000000000000");
            Check(fails, "einsum ij->j (1,3)", () => np.einsum("ij->j", F64(3).reshape(1, 3)),
                  "float64 (3,) 0000000000000000 0000000000000000 0000000000000000");
            Check(fails, "einsum ij-> (1,1)", () => np.einsum("ij->", F64(1).reshape(1, 1)), "float64 () " + z64);
            Check(fails, "einsum ij->i (3,1)", () => np.einsum("ij->i", F64(3).reshape(3, 1)),
                  $"float64 (3,) {z64} {z64} {z64}");
            Check(fails, "einsum ii-> (1,1)", () => np.einsum("ii->", F64(1).reshape(1, 1)), "float64 () " + z64);
            Check(fails, "einsum i,i-> [-0].[1]", () => np.einsum("i,i->", F64(1), np.ones(1)), "float64 () " + z64);
            Check(fails, "einsum ij,j->i (2,1).(1)", () => np.einsum("ij,j->i", F64(2).reshape(2, 1), np.ones(1)),
                  $"float64 (2,) {z64} {z64}");
            Check(fails, "einsum ij,jk->ik (2,1).(1,2)", () => np.einsum("ij,jk->ik", F64(2).reshape(2, 1), np.ones(new Shape(1, 2))),
                  $"float64 (2,2) {z64} {z64} {z64} {z64}");
            Check(fails, "einsum ij,jk->ik (1,1).(1,1)", () => np.einsum("ij,jk->ik", F64(1).reshape(1, 1), np.ones(new Shape(1, 1))),
                  "float64 (1,1) " + z64);
            AssertNoMismatch(fails, "NumPy's einsum starts every output from +0.0");
        }

        // ======================================================================================================
        //  The nan-reductions: the sum (or product) after the NaN replacement, reduced to a 0-d result.
        // ======================================================================================================

        /// <summary>
        ///     <c>np.nansum</c> and <c>np.nanprod</c> of a one-element 1-D float array return a 0-d result, and
        ///     <c>np.nansum</c> of a <c>-0.0</c> (one-element 1-D or 0-d) is <c>+0.0</c> — it is <c>np.sum</c> after
        ///     the NaN replacement.
        /// </summary>
        /// <remarks>
        ///     A one-element input now runs NumPy's composition literally (<c>NanReplacedReduction</c>: replace NaN by
        ///     the identity, then <c>np.sum</c>/<c>np.prod</c>). The old shortcut returned the input's clone — shape
        ///     <c>(1,)</c> for a 1-D input and the sign of a <c>-0.0</c>. The <c>axis=0</c>, float16 and float32
        ///     cases came from the 2026-10-04 reduction-start probe (c45fb4ed).
        /// </remarks>
        [TestMethod]
        public void NanSumNanProd_OneElement_ReturnZeroDimensionalResult()
        {
            var fails = new List<string>();
            Check(fails, "nansum f64 [1]", () => np.nansum(np.array(new[] { 1.0 })), "float64 () 3ff0000000000000");
            Check(fails, "nansum f32 [2]", () => np.nansum(np.array(new[] { 2.0f })), "float32 () 40000000");
            Check(fails, "nansum f64 [-0]", () => np.nansum(F64(1)), "float64 () 0000000000000000");
            Check(fails, "nansum f64 0-d -0", () => np.nansum(NDArray.Scalar(-0.0)), "float64 () 0000000000000000");
            Check(fails, "nanprod f64 [3]", () => np.nanprod(np.array(new[] { 3.0 })), "float64 () 4008000000000000");
            Check(fails, "nansum f64 [1] axis0", () => np.nansum(np.array(new[] { 1.0 }), 0), "float64 () 3ff0000000000000");
            Check(fails, "nanprod f64 [3] axis0", () => np.nanprod(np.array(new[] { 3.0 }), 0), "float64 () 4008000000000000");
            Check(fails, "nansum f16 [-0]", () => np.nansum(F16(1)), "float16 () 0000");
            Check(fails, "nansum f32 0-d -0", () => np.nansum(NDArray.Scalar(-0.0f)), "float32 () 00000000");
            Check(fails, "nanprod f32 [2]", () => np.nanprod(np.array(new[] { 2.0f })), "float32 () 40000000");
            AssertNoMismatch(fails, "NumPy's nansum / nanprod reduce to a 0-d result, starting from the identity");
        }

        /// <summary>
        ///     <c>np.nansum</c> over an axis of length 1 is <c>+0.0 + x</c> per output, so a slab of <c>-0.0</c> gives
        ///     <c>+0.0</c> — C and transposed operands, with and without <c>keepdims</c>.
        /// </summary>
        /// <remarks>
        ///     <c>np.nansum</c> is <c>np.sum</c> after the NaN replacement. The float32/float64 contiguous axis kernel
        ///     (<c>NanReduceContiguousAxis</c>) used to return the lone element of a length-1 axis as it was; it now
        ///     starts from the identity like its strided route. Pinned by the 2026-10-04 reduction-start probe
        ///     (c45fb4ed) as <c>NanSum_LengthOneAxis_IsPositiveZero</c> in <c>OpenBugs.Reductions.cs</c>.
        /// </remarks>
        [TestMethod]
        public void NanSum_LengthOneAxis_IsPositiveZero()
        {
            var fails = new List<string>();
            const string z64x3 = "0000000000000000 0000000000000000 0000000000000000";
            Check(fails, "nansum f64 (3,1) axis1", () => np.nansum(F64(3).reshape(3, 1), 1), "float64 (3,) " + z64x3);
            Check(fails, "nansum f64 (3,1) axis1 keepdims", () => np.nansum(F64(3).reshape(3, 1), 1, true), "float64 (3,1) " + z64x3);
            Check(fails, "nansum f64 (1,1) axis0", () => np.nansum(F64(1).reshape(1, 1), 0), "float64 (1,) 0000000000000000");
            Check(fails, "nansum f32 (3,1).T axis0", () => np.nansum(F32(3).reshape(3, 1).T, 0), "float32 (3,) 00000000 00000000 00000000");
            AssertNoMismatch(fails, "NumPy's nansum is np.sum after the NaN replacement, which starts from +0.0");
        }

        /// <summary>
        ///     <c>np.nanmean</c> of a one-element 1-D array or a 0-d array returns a 0-d result, float64 for an integer
        ///     input, and <c>+0.0</c> for a <c>-0.0</c>.
        /// </summary>
        /// <remarks>
        ///     <c>Statistics/np.nanmean.cs</c> sends a one-element input to <c>np.mean</c> (NumPy's nanmean of a
        ///     non-NaN element), which applies the float64 result dtype for integers and starts from <c>+0.0</c>. It
        ///     used to return the input's clone. The NaN case is compared as a token because NumPy's NaN comes from
        ///     <c>0/0</c> and its sign bit is not part of the contract.
        /// </remarks>
        [TestMethod]
        public void NanMean_OneElement_ReturnsZeroDimensionalFloat64()
        {
            var fails = new List<string>();
            Check(fails, "nanmean f64 [1]", () => np.nanmean(np.array(new[] { 1.0 })), "float64 () 3ff0000000000000");
            Check(fails, "nanmean f64 [-0]", () => np.nanmean(F64(1)), "float64 () 0000000000000000");
            Check(fails, "nanmean f64 [nan]", () => np.nanmean(np.array(new[] { double.NaN })), "float64 () nan");
            Check(fails, "nanmean i32 [5]", () => np.nanmean(np.array(new[] { 5 })), "float64 () 4014000000000000");
            Check(fails, "nanmean f64 0-d -0", () => np.nanmean(NDArray.Scalar(-0.0)), "float64 () 0000000000000000");
            Check(fails, "nanmean i32 0-d 5", () => np.nanmean(NDArray.Scalar(5)), "float64 () 4014000000000000");
            AssertNoMismatch(fails, "NumPy's nanmean reduces to a 0-d float result");
        }

        /// <summary>
        ///     <c>np.nanmean</c> over the only axis of a 1-D array returns a 0-d result (a <c>(1,)</c> array with
        ///     <c>keepdims</c>), as NumPy's <c>np.mean</c> over that axis does.
        /// </summary>
        /// <remarks>
        ///     The float32/float64 axis route builds its result through <c>ApplyKeepdims</c>, which now reshapes a
        ///     reduced 1-D input to <c>()</c> (or <c>(1,)</c>); it used to leave the <c>(1,)</c> shape of the internal
        ///     axis kernel. The complex case came from the 2026-10-04 reduction-start probe's
        ///     <c>NanMeanStdVar_1DAxis_ReturnNumPysShape</c> pin (c45fb4ed), whose nanstd / nanvar cases stay open in
        ///     <c>OpenBugs.Reductions.cs</c>.
        /// </remarks>
        [TestMethod]
        public void NanMean_OneDimensionalAxis_ReturnsZeroDimensional()
        {
            var fails = new List<string>();
            Check(fails, "nanmean f64 [1,2,3] axis0", () => np.nanmean(np.array(new[] { 1.0, 2.0, 3.0 }), axis: 0),
                  "float64 () 4000000000000000");
            Check(fails, "nanmean f64 [1,2,3] axis-1", () => np.nanmean(np.array(new[] { 1.0, 2.0, 3.0 }), axis: -1),
                  "float64 () 4000000000000000");
            Check(fails, "nanmean f64 [1,2,3] axis0 keepdims", () => np.nanmean(np.array(new[] { 1.0, 2.0, 3.0 }), axis: 0, keepdims: true),
                  "float64 (1,) 4000000000000000");
            Check(fails, "nanmean f32 [1,nan,3] axis0", () => np.nanmean(np.array(new[] { 1.0f, float.NaN, 3.0f }), axis: 0),
                  "float32 () 40000000");
            Check(fails, "nanmean c128 [1+2j,3+4j] axis0", () => np.nanmean(np.array(new[] { new Complex(1, 2), new Complex(3, 4) }), 0),
                  "complex128 () 4000000000000000 4008000000000000");
            AssertNoMismatch(fails, "NumPy's nanmean over the only axis of a 1-D array is a 0-d result");
        }

        /// <summary>
        ///     <c>np.nansum</c> of a complex array over an axis — including the only axis of a 1-D array and axis 0 of a
        ///     0-d array — finishes and returns the sum of the non-NaN elements (a complex element with a NaN in either
        ///     part is replaced by <c>0</c>).
        /// </summary>
        /// <remarks>
        ///     The complex axis route used to walk the axis with an incrementor that never advanced over a 0-d input
        ///     (the call hung) and refused a 1-D one (<c>InvalidOperationException</c>). It now runs NumPy's composition
        ///     (<c>NanReplacedReduction</c>). The timeout turns a regression into a failure instead of a hung test run.
        ///     The <c>pair</c> cases came from the 2026-10-04 reduction-start probe's
        ///     <c>NanSum_Complex1D_AlongAxis0_ReturnsTheSum</c> pin (c45fb4ed).
        /// </remarks>
        [TestMethod]
        [Timeout(60_000)]
        public void NanSum_Complex_OverAnAxis_Finishes()
        {
            var fails = new List<string>();
            var withNaN = np.array(new[] { new Complex(1, 2), new Complex(double.NaN, 1), new Complex(3, 4) });
            var grid = np.array(new[,] { { new Complex(1, 2), new Complex(1, double.NaN) }, { new Complex(3, 4), new Complex(5, 6) } });
            var pair = np.array(new[] { new Complex(1, 2), new Complex(3, 4) });
            Check(fails, "nansum c128 [1+2j,3+4j] axis0", () => np.nansum(pair, 0), "complex128 () 4010000000000000 4018000000000000");
            Check(fails, "nansum c128 [1+2j] axis0", () => np.nansum(np.array(new[] { new Complex(1, 2) }), 0),
                  "complex128 () 3ff0000000000000 4000000000000000");
            Check(fails, "nansum c128 [1+2j,3+4j][::-1] axis0", () => np.nansum(pair["::-1"], 0),
                  "complex128 () 4010000000000000 4018000000000000");
            Check(fails, "nansum c128 [-0-0j]*2 axis0 keepdims", () => np.nansum(C128(2), 0, true),
                  "complex128 (1,) 0000000000000000 0000000000000000");
            Check(fails, "nansum c128 1-D axis0", () => np.nansum(withNaN, axis: 0),
                  "complex128 () 4010000000000000 4018000000000000");
            Check(fails, "nansum c128 0-d axis0", () => np.nansum(NDArray.Scalar(new Complex(-0.0, -0.0)), axis: 0),
                  "complex128 () 0000000000000000 0000000000000000");
            Check(fails, "nansum c128 1-D -0 axis0", () => np.nansum(C128(1), axis: 0),
                  "complex128 () 0000000000000000 0000000000000000");
            Check(fails, "nansum c128 1-D axis0 keepdims", () => np.nansum(np.array(new[] { new Complex(1, 2), new Complex(3, 4) }), axis: 0, keepdims: true),
                  "complex128 (1,) 4010000000000000 4018000000000000");
            Check(fails, "nansum c128 (2,2) axis1", () => np.nansum(grid, axis: 1),
                  "complex128 (2,) 3ff0000000000000 4000000000000000 4020000000000000 4024000000000000");
            Check(fails, "nansum c128 (2,2) axis0", () => np.nansum(grid, axis: 0),
                  "complex128 (2,) 4010000000000000 4018000000000000 4014000000000000 4018000000000000");
            AssertNoMismatch(fails, "NumPy's nansum replaces NaN by 0 and sums along the axis");
        }

        /// <summary>
        ///     The NaN-aware axis reductions of a float16 VIEW (strided, reversed, transposed) read the view's own
        ///     elements: <c>np.nansum(a[::2], axis=0)</c> of <c>arange(12).reshape(4,3)</c> is <c>[6, 8, 10]</c>.
        /// </summary>
        /// <remarks>
        ///     float16 has no NaN axis IL kernel, so it takes the scalar walk in <c>Default.Reduction.Nan.cs</c>, which
        ///     computes MEMORY offsets from the strides. It used to read them through <c>GetAtIndex</c>, which takes a
        ///     LOGICAL index and maps it through the strides a second time — other elements (or past the view) for
        ///     every non-contiguous input; a strided slab of <c>-0.0</c> summed to values like 0.0289. A C-contiguous
        ///     input was unaffected (its logical and memory offsets agree). The <c>h23</c> and 1-D reversed / stepped
        ///     cases came from the 2026-10-04 reduction-start probe's
        ///     <c>NanReduction_Float16_NonCLayoutAxis_ReadsTheRightElements</c> pin (c45fb4ed), which saw
        ///     <c>np.nanprod(float16([2, 3])[::-1], axis=0)</c> return garbage that changed from run to run.
        /// </remarks>
        [TestMethod]
        public void NanReductions_Float16_StridedAxis_ReadTheViewsElements()
        {
            var fails = new List<string>();
            var a = np.arange(12).astype(np.float16).reshape(4, 3);
            var b = a.copy();
            b[1, 1] = NDArray.Scalar(Half.NaN);
            b[2, 0] = NDArray.Scalar(Half.NaN);
            var negZeros = np.full(new Shape(4, 3), Half.NegativeZero, np.float16);
            var h23 = np.array(new Half[,] { { (Half)1, (Half)2, (Half)3 }, { (Half)4, (Half)5, (Half)6 } });
            Check(fails, "nanprod f16 [2,3][::-1] axis0", () => np.nanprod(np.array(new[] { (Half)2, (Half)3 })["::-1"], 0), "float16 () 4600");
            Check(fails, "nanprod f16 [2,3,5][::-1] axis0", () => np.nanprod(np.array(new[] { (Half)2, (Half)3, (Half)5 })["::-1"], 0),
                  "float16 () 4f80");
            Check(fails, "nansum f16 [1,2,3][::-1] axis0", () => np.nansum(np.array(new[] { (Half)1, (Half)2, (Half)3 })["::-1"], 0),
                  "float16 () 4600");
            Check(fails, "nansum f16 [1..9][::2] axis0", () => np.nansum(np.arange(1, 10).astype(np.float16)["::2"], 0), "float16 () 4e40");
            Check(fails, "nansum f16 (2,3).T axis0", () => np.nansum(h23.T, 0), "float16 (2,) 4600 4b80");
            Check(fails, "nanprod f16 (2,3).T axis1", () => np.nanprod(h23.T, 1), "float16 (3,) 4400 4900 4c80");
            Check(fails, "nansum f16 (2,3)[:, ::-1] axis1", () => np.nansum(h23[":, ::-1"], 1), "float16 (2,) 4600 4b80");
            Check(fails, "nansum f16 [-0]*9[::-1] axis0", () => np.nansum(F16(9)["::-1"], 0), "float16 () 0000");
            Check(fails, "nansum f16 a[::2] axis0", () => np.nansum(a["::2"], axis: 0), "float16 (3,) 4600 4800 4900");
            Check(fails, "nansum f16 a.T axis1", () => np.nansum(a.T, axis: 1), "float16 (3,) 4c80 4d80 4e80");
            Check(fails, "nansum f16 a[::-1] axis0", () => np.nansum(a["::-1"], axis: 0), "float16 (3,) 4c80 4d80 4e80");
            Check(fails, "nansum f16 a[:, ::2] axis1", () => np.nansum(a[":, ::2"], axis: 1), "float16 (4,) 4000 4800 4b00 4d00");
            Check(fails, "nansum f16 nan b[::-1].T axis1", () => np.nansum(b["::-1"].T, axis: 1), "float16 (3,) 4a00 4c80 4e80");
            Check(fails, "nansum f16 -0 slab [::2] axis0", () => np.nansum(negZeros["::2"], axis: 0), "float16 (3,) 0000 0000 0000");
            Check(fails, "nanprod f16 a[1::2] axis0", () => np.nanprod(a["1::2"], axis: 0), "float16 (3,) 4ec0 5100 52e0");
            Check(fails, "nanmax f16 a.T axis1", () => np.nanmax(a.T, axis: 1), "float16 (3,) 4880 4900 4980");
            Check(fails, "nanmin f16 a[::-1] axis0", () => np.nanmin(a["::-1"], axis: 0), "float16 (3,) 0000 3c00 4000");
            AssertNoMismatch(fails, "a NaN-aware axis reduction reads the view's own elements");
        }

        // ======================================================================================================
        //  Parameters and operands the identity add meets: dtype= loops, sNaN, 0-d axes, broadcast operands.
        // ======================================================================================================

        /// <summary>
        ///     <c>np.sum(a, dtype=d)</c> runs NumPy's <c>d</c> loop: every element is cast to <c>d</c> first (unsafe
        ///     casting — a float truncates into an integer loop, a complex keeps only its real part in a real loop, an
        ///     unsigned value wraps into a signed loop), then the add reduction starts from <c>d</c>'s identity. A bool
        ///     loop is <c>np.logical_or.reduce</c>.
        /// </summary>
        /// <remarks>
        ///     A loop dtype that is neither the input nor its default accumulator takes <c>ReduceAddInCastLoop</c>
        ///     (cast, then the same-dtype sum); a bool loop takes <c>ReduceAddBooleanLoop</c>. Before them a
        ///     complex input with a real dtype crashed the process, and a float input with an integer dtype summed the
        ///     floats and cast the total (<c>[1.9, 2.9, 3.9]</c> gave 8, not NumPy's 6).
        /// </remarks>
        [TestMethod]
        public void Sum_DtypeLoop_CastsEveryElementBeforeTheIdentityAdd()
        {
            var fails = new List<string>();
            Check(fails, "sum f32[-0]*3 dtype=f64", () => np.sum(F32(3), np.float64), "float64 () 0000000000000000");
            Check(fails, "sum f64[-0]*3 dtype=f32", () => np.sum(F64(3), np.float32), "float32 () 00000000");
            Check(fails, "sum f64[-0]*3 dtype=f16", () => np.sum(F64(3), np.float16), "float16 () 0000");
            Check(fails, "sum f16[-0]*3 dtype=c128", () => np.sum(F16(3), np.complex128), "complex128 () 0000000000000000 0000000000000000");
            Check(fails, "sum f64[-0] dtype=c128", () => np.sum(F64(1), np.complex128), "complex128 () 0000000000000000 0000000000000000");
            Check(fails, "sum c128[-0-0j]*3 dtype=f64", () => np.sum(C128(3), np.float64), "float64 () 0000000000000000");
            Check(fails, "sum f64[1.9,2.9,3.9] dtype=i32", () => np.sum(np.array(new[] { 1.9, 2.9, 3.9 }), np.int32), "int32 () 6");
            Check(fails, "sum f64[-0,0.5] dtype=i64", () => np.sum(np.array(new[] { -0.0, 0.5 }), np.int64), "int64 () 0");
            Check(fails, "sum i32[0,2,0] dtype=bool", () => np.sum(np.array(new[] { 0, 2, 0 }), np.@bool), "bool () True");
            Check(fails, "sum i32[0,0] dtype=bool", () => np.sum(np.array(new[] { 0, 0 }), np.@bool), "bool () False");
            Check(fails, "sum c128[1+2j,3+4j] dtype=i64", () => np.sum(np.array(new[] { new Complex(1, 2), new Complex(3, 4) }), np.int64), "int64 () 4");
            Check(fails, "sum bool[T,T,F] dtype=f16", () => np.sum(np.array(new[] { true, true, false }), np.float16), "float16 () 4000");
            Check(fails, "sum f32 (1,3) axis0 dtype=f64", () => np.sum(F32(3).reshape(1, 3), 0, np.float64),
                  "float64 (3,) 0000000000000000 0000000000000000 0000000000000000");
            Check(fails, "sum f64 (2,3) -0 axis1 dtype=f32", () => np.sum(F64(6).reshape(2, 3), 1, np.float32), "float32 (2,) 00000000 00000000");
            Check(fails, "sum i8[100,100] dtype=i8", () => np.sum(np.array(new sbyte[] { 100, 100 }), np.int8), "int8 () -56");
            Check(fails, "sum u8[200,100] dtype=i8", () => np.sum(np.array(new byte[] { 200, 100 }), np.int8), "int8 () 44");
            AssertNoMismatch(fails, "NumPy's dtype= selects the loop: each element is cast to it, then summed from its identity");
        }

        /// <summary>
        ///     The identity add QUIETS a signalling NaN, keeping its sign and payload, as the hardware add NumPy runs
        ///     does: <c>np.sum([sNaN 0x7ff0000000000001])</c> is <c>0x7ff8000000000001</c>, also for float32, float16, a
        ///     complex part, a 0-d input, a length-1 axis and <c>np.mean</c>. <c>np.nansum</c> replaces it by 0.
        /// </summary>
        /// <remarks>
        ///     A copy would return the signalling bits unchanged; that is the observable difference between "the
        ///     element" and "<c>+0.0 + element</c>" for a NaN, so it pins that the identity add really runs.
        /// </remarks>
        [TestMethod]
        public void Sum_SignallingNaN_IsQuietedByTheIdentityAdd()
        {
            var fails = new List<string>();
            double sNaN64 = BitConverter.UInt64BitsToDouble(0x7ff0000000000001);
            double negSNaN64 = BitConverter.UInt64BitsToDouble(0xfff0000000000001);
            float sNaN32 = BitConverter.UInt32BitsToSingle(0x7f800001);
            Half sNaN16 = BitConverter.UInt16BitsToHalf(0x7c01);
            Check(fails, "sum f64 [sNaN]", () => np.sum(np.array(new[] { sNaN64 })), "float64 () 7ff8000000000001", rawNaN: true);
            Check(fails, "sum f64 [-sNaN]", () => np.sum(np.array(new[] { negSNaN64 })), "float64 () fff8000000000001", rawNaN: true);
            Check(fails, "sum f32 [sNaN]", () => np.sum(np.array(new[] { sNaN32 })), "float32 () 7fc00001", rawNaN: true);
            Check(fails, "sum f16 [sNaN]", () => np.sum(np.array(new[] { sNaN16 })), "float16 () 7e01", rawNaN: true);
            Check(fails, "sum c128 [sNaN+1j]", () => np.sum(np.array(new[] { new Complex(sNaN64, 1.0) })),
                  "complex128 () 7ff8000000000001 3ff0000000000000", rawNaN: true);
            Check(fails, "sum f64 (1,1) sNaN axis0", () => np.sum(np.array(new[] { sNaN64 }).reshape(1, 1), axis: 0),
                  "float64 (1,) 7ff8000000000001", rawNaN: true);
            Check(fails, "sum f64 0-d sNaN", () => np.sum(NDArray.Scalar(sNaN64)), "float64 () 7ff8000000000001", rawNaN: true);
            Check(fails, "mean f64 [sNaN]", () => np.mean(np.array(new[] { sNaN64 })), "float64 () 7ff8000000000001", rawNaN: true);
            Check(fails, "nansum f64 [sNaN]", () => np.nansum(np.array(new[] { sNaN64 })), "float64 () 0000000000000000", rawNaN: true);
            AssertNoMismatch(fails, "NumPy's +0.0 + sNaN is the quiet NaN with the same sign and payload");
        }

        /// <summary>
        ///     The axis rules of a one-element reduction, which differ by function in NumPy: <c>np.sum</c>,
        ///     <c>ndarray.sum</c>, <c>np.nansum</c> and <c>np.nanmean</c> accept axis 0 / -1 on a 0-d array (they read it
        ///     as every axis), while <c>np.mean</c> and <c>np.average</c> normalize the axis against rank 0 and raise;
        ///     every function raises <see cref="AxisError"/> for an axis past the rank.
        /// </summary>
        /// <remarks>
        ///     A 0-d result stays 0-d even with <c>keepdims</c> (there is no axis to keep), and a one-element 1-D input
        ///     takes the same one-element route as the 0-d one, whose identity add gives <c>+0.0</c>.
        /// </remarks>
        [TestMethod]
        public void ZeroDimensionalAndOneElement_AxisRules()
        {
            var fails = new List<string>();
            const string pz = "float64 () 0000000000000000";
            Check(fails, "sum 0-d axis0", () => np.sum(NDArray.Scalar(-0.0), axis: 0), pz);
            Check(fails, "sum 0-d axis-1", () => np.sum(NDArray.Scalar(-0.0), axis: -1), pz);
            Check(fails, "sum 0-d axis0 keepdims", () => np.sum(NDArray.Scalar(-0.0), axis: 0, keepdims: true), pz);
            Check(fails, "sum 0-d keepdims", () => np.sum(NDArray.Scalar(-0.0), keepdims: true), pz);
            Check(fails, "ndarray.sum 0-d axis0", () => NDArray.Scalar(-0.0).sum(0), pz);
            Check(fails, "mean 0-d", () => np.mean(NDArray.Scalar(-0.0)), pz);
            Check(fails, "nansum 0-d axis0", () => np.nansum(NDArray.Scalar(-0.0), axis: 0), pz);
            Check(fails, "nansum 0-d axis-1", () => np.nansum(NDArray.Scalar(-0.0), axis: -1), pz);
            Check(fails, "nanmean 0-d axis0", () => np.nanmean(NDArray.Scalar(-0.0), axis: 0), pz);
            Check(fails, "sum (1,) axis0", () => np.sum(F64(1), axis: 0), pz);
            Check(fails, "sum (1,) axis-1", () => np.sum(F64(1), axis: -1), pz);
            Check(fails, "sum (1,) axis0 keepdims", () => np.sum(F64(1), axis: 0, keepdims: true), "float64 (1,) 0000000000000000");
            Check(fails, "mean (1,) axis0", () => np.mean(F64(1), axis: 0), pz);
            Check(fails, "sum i32 0-d axis0", () => np.sum(NDArray.Scalar(7), axis: 0), "int64 () 7");
            Check(fails, "sum bool 0-d axis0 dtype=bool", () => np.sum(NDArray.Scalar(true), 0, np.@bool), "bool () True");
            CheckThrows<AxisError>(fails, "sum 0-d axis1", () => np.sum(NDArray.Scalar(-0.0), axis: 1),
                                   "axis 1 is out of bounds for array of dimension 0");
            CheckThrows<AxisError>(fails, "mean 0-d axis0", () => np.mean(NDArray.Scalar(-0.0), axis: 0),
                                   "axis 0 is out of bounds for array of dimension 0");
            CheckThrows<AxisError>(fails, "nansum 0-d axis1", () => np.nansum(NDArray.Scalar(-0.0), axis: 1),
                                   "axis 1 is out of bounds for array of dimension 0");
            CheckThrows<AxisError>(fails, "nanmean 0-d axis1", () => np.nanmean(NDArray.Scalar(-0.0), axis: 1),
                                   "axis 1 is out of bounds for array of dimension 0");
            CheckThrows<AxisError>(fails, "sum (1,) axis1", () => np.sum(F64(1), axis: 1),
                                   "axis 1 is out of bounds for array of dimension 1");
            CheckThrows<AxisError>(fails, "mean (1,) axis1", () => np.mean(F64(1), axis: 1),
                                   "axis 1 is out of bounds for array of dimension 1");
            CheckThrows<AxisError>(fails, "sum i32 0-d axis2", () => np.sum(NDArray.Scalar(7), axis: 2),
                                   "axis 2 is out of bounds for array of dimension 0");
            CheckThrows<AxisError>(fails, "sum bool 0-d axis1 dtype=bool", () => np.sum(NDArray.Scalar(true), 1, np.@bool),
                                   "axis 1 is out of bounds for array of dimension 0");
            CheckThrows<AxisError>(fails, "sum f64 0-d axis1 dtype=f32", () => np.sum(NDArray.Scalar(-0.0), 1, np.float32),
                                   "axis 1 is out of bounds for array of dimension 0");
            AssertNoMismatch(fails, "NumPy's sum family reads axis 0/-1 of a 0-d array as every axis; mean raises");
        }

        /// <summary>
        ///     A broadcast operand (stride 0) sums from the identity too: <c>np.sum(broadcast_to(-0.0, (1,)))</c> and a
        ///     length-1 axis of a broadcast row give <c>+0.0</c>.
        /// </summary>
        /// <remarks>
        ///     The one-element and length-1 axis routes read the operand through the elementwise add kernel, which
        ///     takes any strides, so a read-only broadcast view needs no materialization; the result is a fresh array.
        /// </remarks>
        [TestMethod]
        public void Sum_BroadcastOperands_StartFromTheIdentity()
        {
            var fails = new List<string>();
            const string z64x3 = "0000000000000000 0000000000000000 0000000000000000";
            Check(fails, "sum bcast -0 (1,)", () => np.sum(np.broadcast_to(NDArray.Scalar(-0.0), new Shape(1))), "float64 () 0000000000000000");
            Check(fails, "sum bcast -0 (3,)", () => np.sum(np.broadcast_to(NDArray.Scalar(-0.0), new Shape(3))), "float64 () 0000000000000000");
            Check(fails, "sum bcast [-0] (1,3) axis0", () => np.sum(np.broadcast_to(F64(1), new Shape(1, 3)), axis: 0), "float64 (3,) " + z64x3);
            Check(fails, "sum bcast [[-0],[-0]] (2,3) axis1", () => np.sum(np.broadcast_to(F64(2).reshape(2, 1), new Shape(2, 3)), axis: 1),
                  "float64 (2,) 0000000000000000 0000000000000000");
            Check(fails, "mean bcast [-0] (1,3) axis0", () => np.mean(np.broadcast_to(F64(1), new Shape(1, 3)), axis: 0), "float64 (3,) " + z64x3);
            Check(fails, "sum bcast f32 -0 (1,4) axis0", () => np.sum(np.broadcast_to(NDArray.Scalar(-0.0f), new Shape(1, 4)), axis: 0),
                  "float32 (4,) 00000000 00000000 00000000 00000000");
            AssertNoMismatch(fails, "NumPy's add.reduce of a broadcast operand starts from +0.0");
        }

        /// <summary>
        ///     A complex mean divides the sum by the count with NumPy's complex loop — Smith's algorithm with the
        ///     divisor <c>n+0j</c>: <c>((re + im*0) * (1/n), (im - re*0) * (1/n))</c>. An infinite or NaN part therefore
        ///     poisons the other part, even for one element: <c>np.mean([5+inf j])</c> is <c>(nan, inf)</c>, with the
        ///     NaN that x86 produces for <c>inf*0</c> (the negative quiet NaN <c>0xfff8000000000000</c>).
        /// </summary>
        /// <remarks>
        ///     <c>DefaultEngine.ComplexDivideByCountLikeNumPy</c> is the one division every complex mean route calls
        ///     (the flat mean, the axis post-pass <c>MeanDivideByCount</c>, the one-element and length-1 axis routes,
        ///     <c>np.evaluate</c>'s Mean). It used to divide each part by <c>n</c> separately, which keeps the finite
        ///     part finite. The <c>nan+1j</c> case uses NumPy's <c>np.nan</c> (the positive quiet NaN), whose bits
        ///     propagate through <c>nan*0</c>.
        /// </remarks>
        [TestMethod]
        public void Mean_Complex_DividesLikeNumPysComplexLoop()
        {
            var fails = new List<string>();
            double inf = double.PositiveInfinity;
            double numpyNaN = BitConverter.UInt64BitsToDouble(0x7ff8000000000000);
            var grid = np.array(new[,] { { new Complex(5, inf), new Complex(1, 1) }, { new Complex(1, 1), new Complex(2, 2) } });
            Check(fails, "mean [5+infj]", () => np.mean(np.array(new[] { new Complex(5, inf) })),
                  "complex128 () fff8000000000000 7ff0000000000000");
            Check(fails, "mean [inf-2j]", () => np.mean(np.array(new[] { new Complex(inf, -2) })),
                  "complex128 () 7ff0000000000000 fff8000000000000");
            Check(fails, "mean [5+infj, 1+1j]", () => np.mean(np.array(new[] { new Complex(5, inf), new Complex(1, 1) })),
                  "complex128 () fff8000000000000 7ff0000000000000");
            Check(fails, "mean (1,3) axis0", () => np.mean(np.array(new[,] { { new Complex(5, inf), new Complex(1, 1), new Complex(inf, 2) } }), axis: 0),
                  "complex128 (3,) fff8000000000000 7ff0000000000000 3ff0000000000000 3ff0000000000000 7ff0000000000000 fff8000000000000");
            Check(fails, "mean (2,2) axis0", () => np.mean(grid, axis: 0),
                  "complex128 (2,) fff8000000000000 7ff0000000000000 3ff8000000000000 3ff8000000000000");
            Check(fails, "mean (2,2) axis1", () => np.mean(grid, axis: 1),
                  "complex128 (2,) fff8000000000000 7ff0000000000000 3ff8000000000000 3ff8000000000000");
            Check(fails, "average [5+infj]", () => np.average(np.array(new[] { new Complex(5, inf) })),
                  "complex128 () fff8000000000000 7ff0000000000000");
            Check(fails, "nanmean [5+infj]", () => np.nanmean(np.array(new[] { new Complex(5, inf) })),
                  "complex128 () fff8000000000000 7ff0000000000000");
            Check(fails, "mean [-0-0j]", () => np.mean(C128(1)), "complex128 () 0000000000000000 0000000000000000");
            Check(fails, "mean [nan+1j]", () => np.mean(np.array(new[] { new Complex(numpyNaN, 1) })),
                  "complex128 () 7ff8000000000000 7ff8000000000000");
            Check(fails, "mean 0-d 5+infj", () => np.mean(NDArray.Scalar(new Complex(5, inf))),
                  "complex128 () fff8000000000000 7ff0000000000000");
            AssertNoMismatch(fails, "NumPy divides a complex mean with its complex loop, by n+0j");
        }

        /// <summary>
        ///     The mean of an EMPTY array (or along an empty axis) is NaN in the mean's own dtype: float32 stays
        ///     float32, float16 float16, complex128 is <c>(nan, nan)</c>, and an integer input is float64.
        /// </summary>
        /// <remarks>
        ///     NumPy divides the identity-seeded sum (0) by a count of 0. This route used to return a float64 NaN for
        ///     every dtype. The float NaNs are compared as tokens (a 0/0 sign is not contractual); the complex one is
        ///     raw bits, both parts the x86 default NaN NumPy's zero-divisor branch produces. The <c>(0,3)</c> cases at
        ///     the end came from the <c>Mean_EmptyFloat16Float32Complex_KeepTheirDtype</c> pin (24a31f75, group 5 of
        ///     <c>OpenBugs.Reductions.cs</c>), written while this route was still float64.
        /// </remarks>
        [TestMethod]
        public void Mean_Empty_IsNaNInTheMeansDtype()
        {
            var fails = new List<string>();
            Check(fails, "mean f32 empty", () => np.mean(np.zeros(new Shape(0), np.float32)), "float32 () nan");
            Check(fails, "mean f16 empty", () => np.mean(np.zeros(new Shape(0), np.float16)), "float16 () nan");
            Check(fails, "mean c128 empty", () => np.mean(np.zeros(new Shape(0), np.complex128)),
                  "complex128 () fff8000000000000 fff8000000000000");
            Check(fails, "mean i32 empty", () => np.mean(np.zeros(new Shape(0), np.int32)), "float64 () nan");
            Check(fails, "mean f32 (0,3) axis0", () => np.mean(np.zeros(new Shape(0, 3), np.float32), axis: 0), "float32 (3,) nan nan nan");
            Check(fails, "mean f32 (0,3) axis1", () => np.mean(np.zeros(new Shape(0, 3), np.float32), axis: 1), "float32 (0,) ");
            Check(fails, "mean f32 empty keepdims", () => np.mean(np.zeros(new Shape(0), np.float32), keepdims: true), "float32 (1,) nan");
            Check(fails, "mean c128 (0,2) axis0", () => np.mean(np.zeros(new Shape(0, 2), np.complex128), axis: 0),
                  "complex128 (2,) fff8000000000000 fff8000000000000 fff8000000000000 fff8000000000000");
            Check(fails, "mean f16 (0,2) axis0", () => np.mean(np.zeros(new Shape(0, 2), np.float16), axis: 0), "float16 (2,) nan nan");
            Check(fails, "mean i32 (0,2) axis0", () => np.mean(np.zeros(new Shape(0, 2), np.int32), axis: 0), "float64 (2,) nan nan");
            Check(fails, "mean f64 empty axis0 dtype=f32", () => np.mean(np.zeros(new Shape(0)), 0, np.float32), "float32 () nan");
            Check(fails, "mean f16 (0,3)", () => np.mean(np.zeros(new Shape(0, 3), np.float16)), "float16 () nan");
            Check(fails, "mean f16 (0,3) axis0 keepdims", () => np.mean(np.zeros(new Shape(0, 3), np.float16), 0, keepdims: true),
                  "float16 (1,3) nan nan nan");
            Check(fails, "mean c128 (0,3) axis1", () => np.mean(np.zeros(new Shape(0, 3), np.complex128), 1), "complex128 (0,) ");
            Check(fails, "mean f64 (0,3) axis0", () => np.mean(np.zeros(new Shape(0, 3), np.float64), 0), "float64 (3,) nan nan nan");
            Check(fails, "mean i32 (0,3)", () => np.mean(np.zeros(new Shape(0, 3), np.int32)), "float64 () nan");
            AssertNoMismatch(fails, "NumPy's empty mean is 0/0 in the mean's dtype");
        }

        /// <summary>
        ///     A typed sum and a default sum of the same input dtype and layout do not share a reduction kernel: the
        ///     kernel cache key carries the accumulator, so <c>np.sum(F-order int8, dtype=int8)</c> (int8 loop, 45) and
        ///     the default <c>np.sum</c> (int64 accumulator, 45) both run their own kernel in either order.
        /// </summary>
        /// <remarks>
        ///     <c>NDIterRef.ExecuteReduction</c> used to derive the accumulator from the input dtype even when the caller
        ///     asked for another, so the first call cached an int8-accumulator kernel under a key the default int8 sum
        ///     later read ("IL kernel not available for Sum(SByte)->Int64"). The cache is process-wide, so the order
        ///     inside one test only proves the pair once; both orders run here, on two distinct layouts.
        /// </remarks>
        [TestMethod]
        public void Sum_TypedLoop_KernelCacheKeepsTheAccumulator()
        {
            var fails = new List<string>();
            var fi8 = np.asfortranarray(np.arange(1, 10).astype(np.int8).reshape(3, 3));
            var fbool = np.asfortranarray(np.array(new[,] { { true, false }, { true, true } }));
            Check(fails, "sum F i8 dtype=i8", () => np.sum(fi8, np.int8), "int8 () 45");
            Check(fails, "sum F i8", () => np.sum(fi8), "int64 () 45");
            Check(fails, "sum F i8 dtype=i16", () => np.sum(fi8, np.int16), "int16 () 45");
            Check(fails, "sum F bool", () => np.sum(fbool), "int64 () 3");
            Check(fails, "sum F bool dtype=i8", () => np.sum(fbool, np.int8), "int8 () 3");
            Check(fails, "sum F bool again", () => np.sum(fbool), "int64 () 3");
            AssertNoMismatch(fails, "a reduction kernel is cached per accumulator, not per input dtype alone");
        }
    }
}
