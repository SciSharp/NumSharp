using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs in how NumSharp STARTS a reduction, found while making <c>np.trapezoid</c> bit-exact
    ///     (commit 94c9eb6b, 2026-10-03) and confirmed by the follow-up probes. Each test asserts NumPy 2.4.2's
    ///     answer — dtype, shape and the raw bits of every element — so it FAILS while the bug lives and passes
    ///     once it is fixed. Remove <c>[OpenBugs]</c> from a test when its path is fixed, and move the test next
    ///     to the function's own tests.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The sign of a zero is the only visible difference in most of these cases, so a value comparison
    ///     (<c>0.0 == -0.0</c>) passes them all; that is why nothing caught them. The oracle compares raw bytes, but
    ///     its generators feed nonzero operands to the tiers that would meet these paths (the einsum tier says so
    ///     explicitly). NumPy's three rules, each ported wrong somewhere below:
    ///     </para>
    ///     <list type="number">
    ///         <item><b><c>add.reduce</c> starts from the <c>+0.0</c> identity.</b> For a numeric dtype NumPy's
    ///         reduction passes the identity as the initial value even for a non-empty input, so the result is
    ///         <c>+0.0 + x0 + x1 …</c>: a sum of only <c>-0.0</c> values is <c>+0.0</c>
    ///         (<c>np.sum([-0.0])</c> is <c>0.0</c>). NumSharp's size-1 shortcuts COPY the element instead.</item>
    ///         <item><b><c>add.accumulate</c> copies the first element.</b> <c>np.cumsum([-0.0, -0.0])</c> is
    ///         <c>[-0.0, -0.0]</c>. NumSharp's flat scan starts from <c>default(T)</c>, which is <c>+0.0</c>.</item>
    ///         <item><b><c>np.dot</c> of two length-1 vectors is a plain product.</b> NumPy's
    ///         <c>cblas_matrixproduct</c> classifies a length-1 1-D operand as a scalar and computes
    ///         <c>l == 1</c> as <c>a * b</c>, so <c>np.dot([-0.0], [1.0])</c> is <c>-0.0</c>. NumSharp's managed
    ///         1-D dot starts its sum from <c>0</c>.</item>
    ///     </list>
    ///     <para>
    ///     A fourth group is a shape and dtype bug in the same size-1 shortcuts: <c>np.nansum</c>,
    ///     <c>np.nanprod</c> and <c>np.nanmean</c> of a one-element 1-D array return the input's <c>(1,)</c> clone
    ///     instead of NumPy's 0-d result, and <c>np.nanmean</c> of a one-element or 0-d array keeps an integer
    ///     dtype where NumPy returns float64 (the clone also keeps a <c>-0.0</c>, rule 1).
    ///     </para>
    ///     <para>
    ///     A fifth group (added 2026-10-04) is <c>np.var</c> / <c>np.std</c> at a degenerate size: one element per
    ///     output, a <c>ddof</c> at or past the element count, or an empty float16/float32 input. NumPy's
    ///     <c>_var</c> always runs its arithmetic — <c>(x - mean)²</c> summed, then divided by
    ///     <c>max(n - ddof, 0)</c>, in the input's float dtype — while NumSharp's shortcuts answer without it:
    ///     <c>0</c> instead of NaN for a NaN or ±inf element, <c>0</c> instead of NaN for <c>ddof &gt;= 1</c> over a
    ///     length-1 axis, NaN or a NEGATIVE variance instead of <c>+inf</c> for <c>ddof &gt;= n</c>, and float64
    ///     instead of float16/float32. The oracle had these cases all along but excused them under labels that
    ///     describe other bugs ("summation precision", the min/max "axis SIMD skips NaN" bug, the generic
    ///     reduction-dtype bug); its registry now names this bug and points here. <c>np.mean</c>'s empty path has
    ///     the same float64 hard-code and is pinned with the group.
    ///     </para>
    ///     <para>
    ///     The 2026-10-04 reduction-start probe extended groups 1-4. It ran every reduction, scan and product
    ///     over all-<c>-0.0</c> operands (size 1, length-1 axes, longer runs; C, transposed, reversed and stepped
    ///     layouts; float64, float32, float16 and complex128) against NumPy 2.4.2 bit for bit. The 7,942 cells
    ///     NumSharp gets right are the <c>reduce_start.jsonl</c> oracle tier
    ///     (<c>FuzzCorpusTests.ReductionStart</c>). The cells it gets wrong are that tier's carve
    ///     (<c>gen_oracle._rs_carved</c>, one rule per root cause), and every rule names the test below that pins
    ///     it (the one-element float32/float16 <c>std</c>/<c>var</c> dtype rule names group 5's test). Two more of
    ///     NumPy's rules showed up there:
    ///     </para>
    ///     <list type="bullet">
    ///         <item><b><c>multiply.reduce</c> starts from the identity <c>1</c>, which is not a no-op for a
    ///         complex element.</b> NumPy's complex product of one element is <c>(1+0j) * x</c>, so
    ///         <c>np.prod([-0-0j])</c> is <c>(+0, -0)</c>. NumSharp copies the element.</item>
    ///         <item><b><c>np.median</c> is <c>np.mean</c> of the middle element(s).</b> It starts from
    ///         <c>+0.0</c> too, so a <c>-0.0</c> middle gives <c>+0.0</c>. NumSharp returns the middle element
    ///         itself.</item>
    ///     </list>
    ///     <para>
    ///     The probe also found reductions that return the wrong shape or VALUES (group 6), and
    ///     <c>np.trapezoid</c>'s error for an axis outside <c>y</c> (group 7). Those tests use ordinary values,
    ///     where a value comparison alone already fails.
    ///     </para>
    ///     <para>
    ///     Every expected string below is NumPy 2.4.2's output for the same call (same operands, same arguments),
    ///     probed on 2026-10-03 (groups 1-4) and 2026-10-04 (groups 5-7 and the probe's additions to groups 1-4),
    ///     in <see cref="Describe"/>'s format: <c>dtype (shape) bits…</c>, bits as hex in logical C order (a
    ///     complex element is its real word, then its imaginary word). An error is
    ///     <c>threw &lt;type&gt;: &lt;message&gt;</c>, NumPy's Python exception type and message.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class OpenBugsReductionTests : TestClass
    {
        // ======================================================================================================
        //  GROUP 1 — add.reduce starts from the +0.0 identity; NumSharp's size-1 paths copy the element.
        // ======================================================================================================

        /// <summary>
        ///     <c>np.sum</c> of one element (a one-element 1-D array or a 0-d array, any float dtype, with or
        ///     without a widening <c>dtype=</c>) must be <c>+0.0 + x</c>, so a <c>-0.0</c> element gives
        ///     <c>+0.0</c>. NumSharp returns the element itself.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Default.Reduction.Add.cs</c> <c>ReduceAdd</c> sends a size-1 input to
        ///     <c>HandleScalarReduction</c>, which clones or casts the lone element. That helper is shared with
        ///     <c>amin</c>/<c>amax</c>, which MUST copy (a min/max has no identity), so the fix belongs at the
        ///     sum call site: add the element to <c>+0.0</c> in the result dtype. <c>ndarray.sum()</c> takes the
        ///     same path.
        /// </remarks>
        [TestMethod, OpenBugs]
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
        ///     <c>np.sum</c> over an axis of length 1 must return <c>+0.0 + x</c> for every output, so a slab of
        ///     <c>-0.0</c> values gives <c>+0.0</c>; it holds for every float dtype, both axis positions and
        ///     <c>keepdims</c>. NumSharp copies the slab.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Default.Reduction.Add.cs</c> sends an extent-1 reduced axis to
        ///     <c>HandleTrivialAxisReduction</c>, which writes each element straight into the result. The helper is
        ///     shared with <c>amin</c>/<c>amax</c>/<c>nanmin</c>/<c>nanmax</c> (which must copy) and with
        ///     <c>mean</c>, so the fix belongs at the sum (and mean) call sites. A length-2 axis is already
        ///     correct: <c>np.sum(full((2,3), -0.0), axis=0)</c> gives <c>+0.0</c> on both sides.
        /// </remarks>
        [TestMethod, OpenBugs]
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
        ///     The flat <c>np.sum</c> of several float16 <c>-0.0</c> values must be <c>+0.0</c>. NumSharp keeps the
        ///     sign; float32 and float64 already give <c>+0.0</c> for the same input.
        /// </summary>
        /// <remarks>
        ///     Root cause: the float16 flat sum runs its own route (<c>Default.Reduction.Add.cs</c>: the
        ///     float32-shadow <c>HalfSumKernel</c>, then a narrow to float16), and that route does not start from
        ///     NumPy's <c>+0.0</c> identity the way the float32/float64 <c>PairwiseFold</c> route does. An axis sum
        ///     over a length-2 or longer axis is already correct (<c>(2,3) axis=0</c> and <c>(3,2) axis=1</c> both
        ///     give <c>+0.0</c>).
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Sum_Float16Flat_OfNegativeZeros_IsPositiveZero()
        {
            var fails = new List<string>();
            Check(fails, "sum f16 [-0]*3", () => np.sum(F16(3)), "float16 () 0000");
            Check(fails, "sum f16 [-0]*100", () => np.sum(F16(100)), "float16 () 0000");
            AssertNoMismatch(fails, "NumPy's HALF_add reduce starts from the +0.0 identity");
        }

        /// <summary>
        ///     <c>np.mean</c> (and the unweighted <c>np.average</c>, which is <c>np.mean</c>) of one element, or over
        ///     a length-1 axis, must be <c>(+0.0 + x) / n</c>, so <c>-0.0</c> gives <c>+0.0</c>. NumSharp returns the
        ///     element itself.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Default.Reduction.Mean.cs</c> has its own size-1 shortcut (it converts the lone element
        ///     to the result dtype and wraps it as a scalar) and sends an extent-1 axis to
        ///     <c>HandleTrivialAxisReduction</c>. NumPy's mean
        ///     is <c>umr_sum(...) / n</c>, so it inherits the sum's <c>+0.0</c> start. <c>np.mean</c> of three
        ///     float16 <c>-0.0</c> values is already correct.
        /// </remarks>
        [TestMethod, OpenBugs]
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
        ///     The fused <c>np.evaluate(NDExpr.Sum(x, axis))</c> over a length-1 axis must give <c>+0.0</c> for a
        ///     <c>-0.0</c> slab, as <c>np.sum</c> does in NumPy. NumSharp copies the slab. The same holds for the
        ///     other sum-based kinds over a length-1 axis of a 2-D operand (<c>Mean</c>, <c>NanSum</c>,
        ///     <c>NanMean</c>) and for float32 and complex128. The flat <c>NDExpr.Sum(x)</c> of one element is
        ///     already correct.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>DefaultEngine.Evaluate.Stream.cs</c> produces an extent-1 axis straight into the result
        ///     ON PURPOSE, to mirror the engine's copy in <c>HandleTrivialAxisReduction</c> (its comment says so).
        ///     Fix it together with <see cref="Sum_LengthOneAxis_IsPositiveZero"/>, or the two drift apart. The
        ///     complex128 <c>Mean</c> keeps the real part's <c>-0.0</c> and gives an imaginary <c>+0.0</c>, so the
        ///     copy and the divide by the count both show there. A float16 operand takes another route and is
        ///     already correct.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Evaluate_SumOverLengthOneAxis_IsPositiveZero()
        {
            var fails = new List<string>();
            Check(fails, "evaluate Sum f64 (1,3) axis0", () => np.evaluate(NDExpr.Sum((NDExpr)F64(3).reshape(1, 3), 0)),
                  "float64 (3,) 0000000000000000 0000000000000000 0000000000000000");
            Check(fails, "evaluate Mean f64 (1,3) axis0", () => np.evaluate(NDExpr.Mean((NDExpr)F64(3).reshape(1, 3), 0)),
                  "float64 (3,) 0000000000000000 0000000000000000 0000000000000000");
            Check(fails, "evaluate NanSum f64 (3,1) axis1", () => np.evaluate(NDExpr.NanSum((NDExpr)F64(3).reshape(3, 1), 1)),
                  "float64 (3,) 0000000000000000 0000000000000000 0000000000000000");
            Check(fails, "evaluate NanMean f32 (1,1) axis0", () => np.evaluate(NDExpr.NanMean((NDExpr)F32(1).reshape(1, 1), 0)),
                  "float32 (1,) 00000000");
            Check(fails, "evaluate Sum c128 (1,3) axis0 keepdims",
                  () => np.evaluate(NDExpr.Sum((NDExpr)C128(3).reshape(1, 3), 0, true)),
                  "complex128 (1,3) 0000000000000000 0000000000000000 0000000000000000 0000000000000000 " +
                  "0000000000000000 0000000000000000");
            Check(fails, "evaluate Mean c128 (1,1) axis1", () => np.evaluate(NDExpr.Mean((NDExpr)C128(1).reshape(1, 1), 1)),
                  "complex128 (1,) 0000000000000000 0000000000000000");
            AssertNoMismatch(fails, "np.evaluate's sum-based kinds must match NumPy's, which start from +0.0");
        }

        /// <summary>
        ///     The complex <c>np.prod</c> and <c>np.nanprod</c> of one element (0-d, one-element 1-D, or a
        ///     length-1 axis) must be <c>(1+0j) * x</c>, NumPy's identity times the element, so <c>-0-0j</c> gives
        ///     <c>(+0, -0)</c> and <c>2-0j</c> gives <c>(2, +0)</c>. NumSharp returns the element itself. The fused
        ///     <c>np.evaluate(NDExpr.Prod(x))</c> and <c>NDExpr.NanProd</c> give <c>(+0, +0)</c> instead.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     For a real dtype the identity is exact (<c>1 * x</c> is <c>x</c>, sign included), so copying the
        ///     element is right and float64 / float32 / float16 already match. For complex it is not:
        ///     <c>(1+0j) * (a+bj)</c> is <c>(1*a - 0*b, 1*b + 0*a)</c>, and <c>-0.0 - (-0.0)</c> is <c>+0.0</c>.
        ///     A longer product multiplies anyway and is already correct.
        ///     </para>
        ///     <para>
        ///     Root cause: <c>Default.Reduction.Product.cs</c> sends a size-1 input to
        ///     <c>HandleScalarReduction</c> and an extent-1 axis to <c>HandleTrivialAxisReduction</c>, both of which
        ///     copy. <c>NanProd</c> sends a complex input to <c>ReduceProduct</c>, the same path. The fused
        ///     reduction's fold multiplies the element by the identity and then combines it with its other
        ///     identity accumulators, which turns the imaginary <c>-0.0</c> into <c>+0.0</c>
        ///     (<c>-0.0 * 1 + 0 * 0</c>).
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Prod_OneComplexElement_IsTheIdentityTimesTheElement()
        {
            var fails = new List<string>();
            const string plusMinus = "0000000000000000 8000000000000000";
            Check(fails, "prod c128 [-0-0j]", () => np.prod(C128(1)), "complex128 () " + plusMinus);
            Check(fails, "prod c128 0-d -0-0j", () => np.prod(NDArray.Scalar(new Complex(-0.0, -0.0))), "complex128 () " + plusMinus);
            Check(fails, "prod c128 [-0-0j] keepdims", () => np.prod(C128(1), keepdims: true), "complex128 (1,) " + plusMinus);
            Check(fails, "prod c128 (1,3) axis0", () => np.prod(C128(3).reshape(1, 3), 0),
                  "complex128 (3,) " + plusMinus + " " + plusMinus + " " + plusMinus);
            Check(fails, "prod c128 [2-0j]", () => np.prod(np.array(new[] { new Complex(2.0, -0.0) })),
                  "complex128 () 4000000000000000 0000000000000000");
            Check(fails, "nanprod c128 [-0-0j]", () => np.nanprod(C128(1)), "complex128 () " + plusMinus);
            Check(fails, "nanprod c128 (3,1) axis1", () => np.nanprod(C128(3).reshape(3, 1), 1),
                  "complex128 (3,) " + plusMinus + " " + plusMinus + " " + plusMinus);
            Check(fails, "evaluate Prod c128 [-0-0j]", () => np.evaluate(NDExpr.Prod((NDExpr)C128(1))), "complex128 () " + plusMinus);
            Check(fails, "evaluate NanProd c128 (1,1) axis0", () => np.evaluate(NDExpr.NanProd((NDExpr)C128(1).reshape(1, 1), 0)),
                  "complex128 (1,) " + plusMinus);
            AssertNoMismatch(fails, "NumPy's complex multiply.reduce multiplies the lone element by the identity 1+0j");
        }

        /// <summary>
        ///     <c>np.median</c> and <c>np.nanmedian</c> must be <c>np.mean</c> of the middle element(s), as in
        ///     NumPy, so a <c>-0.0</c> middle gives <c>+0.0</c> (<c>np.median([-0.0, 5.0, -0.0])</c> is
        ///     <c>0.0</c>). NumSharp returns the middle element itself, or <c>(a + b) / 2</c> for an even count,
        ///     which keeps the <c>-0.0</c>. Every cell of the probe's all-<c>-0.0</c> matrix differs: every float
        ///     dtype, complex128, flat and per axis, with and without keepdims.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>DirectILKernelGenerator.Quantile.cs</c> <c>WriteCell</c> with <c>medianMean</c> writes
        ///     <c>prev</c> for an odd count ("γ = 0 … already yields that sample, so it needs no special case") and
        ///     <c>(prev + next) / 2</c> for an even count. NumPy's <c>median</c> calls <c>np.mean(part[indexer])</c>,
        ///     which is <c>(+0.0 + prev [+ next]) / n</c>. The values are equal except for the sign of a zero.
        ///     <c>np.percentile</c> and <c>np.quantile</c> use NumPy's <c>_lerp</c> and already match.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Median_OfNegativeZeros_IsPositiveZero()
        {
            var fails = new List<string>();
            Check(fails, "median f64 [-0,-0,-0]", () => np.median(F64(3)), "float64 () 0000000000000000");
            Check(fails, "median f64 [-0,5,-0]", () => np.median(np.array(new[] { -0.0, 5.0, -0.0 })), "float64 () 0000000000000000");
            Check(fails, "median f32 [-0,-0]", () => np.median(F32(2)), "float32 () 00000000");
            Check(fails, "median f64 (2,3) axis0", () => np.median(F64(6).reshape(2, 3), axis: 0),
                  "float64 (3,) 0000000000000000 0000000000000000 0000000000000000");
            Check(fails, "median c128 [-0-0j]*3", () => np.median(C128(3)), "complex128 () 0000000000000000 0000000000000000");
            Check(fails, "nanmedian f64 [-0]", () => np.nanmedian(F64(1)), "float64 () 0000000000000000");
            Check(fails, "nanmedian f16 [-0]*9", () => np.nanmedian(F16(9)), "float16 () 0000");
            AssertNoMismatch(fails, "NumPy's median is np.mean of the middle element(s), which starts from +0.0");
        }

        /// <summary>
        ///     <c>np.trapezoid</c> over two samples sums ONE product along the integration axis, so a
        ///     <c>-0.0</c> area must come back as <c>+0.0</c>. NumSharp returns <c>-0.0</c>; these are the 18
        ///     cases of the 7,430-case trapezoid differential that still differed after 94c9eb6b.
        /// </summary>
        /// <remarks>
        ///     Root cause: trapezoid ends in <c>np.sum(…, axis)</c> over a length-1 axis, so it inherits
        ///     <see cref="Sum_LengthOneAxis_IsPositiveZero"/> and <see cref="Sum_OneElement_IsPositiveZero_LikeNumPysAddReduceIdentity"/>.
        ///     The <c>-0.0</c> product comes from a negative spacing times a <c>+0.0</c> sum
        ///     (<c>-1 * (0 + 0) / 2</c>).
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Trapezoid_OneInterval_NegativeZeroArea_IsPositiveZero()
        {
            var fails = new List<string>();
            Check(fails, "trapezoid [0,0] x=[1,0]", () => np.trapezoid(np.array(new[] { 0.0, 0.0 }), np.array(new[] { 1.0, 0.0 })),
                  "float64 () 0000000000000000");
            Check(fails, "trapezoid [0,0] dx=-1", () => np.trapezoid(np.array(new[] { 0.0, 0.0 }), null, -1.0),
                  "float64 () 0000000000000000");
            Check(fails, "trapezoid (2,3) x=[1,0] axis0", () => np.trapezoid(np.zeros(new Shape(2, 3)), np.array(new[] { 1.0, 0.0 }), 1.0, 0),
                  "float64 (3,) 0000000000000000 0000000000000000 0000000000000000");
            // 2026-10-04 probe: a -0.0 y with a positive spacing, the last axis of a 2-D y, and the narrower dtypes.
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
        ///     <c>np.einsum</c> summing a single term (<c>'i->'</c> over one element, <c>'ij->j'</c> over a length-1
        ///     axis) must give <c>+0.0</c> for <c>-0.0</c>. NumSharp returns <c>-0.0</c>.
        /// </summary>
        /// <remarks>
        ///     Root cause: the single-operand contraction composes <c>np.sum</c>, so it inherits the two sum bugs
        ///     above. This is one part of the einsum signed-zero divergence the einsum oracle tier avoids by feeding
        ///     nonzero operands (NumPy's <c>sum_of_products</c> accumulator also starts from <c>+0.0</c>).
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Einsum_SumOfOneTerm_IsPositiveZero()
        {
            var fails = new List<string>();
            Check(fails, "einsum i-> [-0]", () => np.einsum("i->", F64(1)), "float64 () 0000000000000000");
            Check(fails, "einsum ij->j (1,3)", () => np.einsum("ij->j", F64(3).reshape(1, 3)),
                  "float64 (3,) 0000000000000000 0000000000000000 0000000000000000");
            AssertNoMismatch(fails, "NumPy's einsum sum starts from +0.0");
        }

        /// <summary>
        ///     NumPy's default <c>np.einsum</c> (<c>optimize=False</c>) adds every term into a ZERO-FILLED output,
        ///     so each output element is <c>+0.0 + terms</c>: a <c>-0.0</c> sum, a lone <c>-0.0</c> term and even
        ///     a bare elementwise product (<c>'i,i->i'</c>, <c>'i,j->ij'</c>) come out <c>+0.0</c>.
        ///     NumSharp's contraction returns <c>-0.0</c> for all of them; for complex128 it returns the bare
        ///     product's <c>(+0, -0)</c> where NumPy returns <c>(+0, +0)</c>. An ordinary operand shows it too:
        ///     <c>np.einsum('i,i->i', [-0.0, 1.0], [1.0, 1.0])</c> is <c>[0.0, 1.0]</c>.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Root cause: <c>np.EinsumContract</c> (<c>LinearAlgebra/np.einsum.Contract.cs</c>) ports NumPy's
        ///     <c>optimize=</c> structure: it takes diagonals and sums with <c>np.sum</c> (which copies a lone
        ///     element, <see cref="Sum_OneElement_IsPositiveZero_LikeNumPysAddReduceIdentity"/>), and it returns a
        ///     contraction with nothing summed as a broadcast <c>np.multiply</c>. NumPy's DEFAULT einsum is
        ///     <c>PyArray_EinsteinSum</c>: the output operand is zeroed and every <c>sum_of_products</c> loop
        ///     computes <c>out += a * b</c>. NumPy's own <c>optimize=True</c> path gives <c>-0.0</c> for the bare
        ///     products here (probed: <c>[-0., 1.]</c> for the ordinary case above), so the fix is to start each
        ///     output from <c>+0.0</c> on the default path, not to change which kernels compose.
        ///     </para>
        ///     <para>
        ///     The einsum oracle tier feeds nonzero operands and so never meets this; the reduce_start tier gates the
        ///     einsum cells that already match (sums over two or more terms, views, the 2-D contractions over
        ///     <c>k = 2</c>).
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Einsum_ZeroFilledAccumulator_GivesPositiveZero()
        {
            var fails = new List<string>();
            const string z64 = "0000000000000000";
            Check(fails, "einsum ij-> (1,1)", () => np.einsum("ij->", F64(1).reshape(1, 1)), "float64 () " + z64);
            Check(fails, "einsum ij->i (3,1)", () => np.einsum("ij->i", F64(3).reshape(3, 1)),
                  $"float64 (3,) {z64} {z64} {z64}");
            Check(fails, "einsum ii-> (1,1)", () => np.einsum("ii->", F64(1).reshape(1, 1)), "float64 () " + z64);
            Check(fails, "einsum i,i-> [-0].[1]", () => np.einsum("i,i->", F64(1), np.ones(1)), "float64 () " + z64);
            Check(fails, "einsum i,i->i (3).(3)", () => np.einsum("i,i->i", F64(3), np.ones(3)),
                  $"float64 (3,) {z64} {z64} {z64}");
            Check(fails, "einsum i,j->ij (2).(3)", () => np.einsum("i,j->ij", F64(2), np.ones(3)),
                  $"float64 (2,3) {z64} {z64} {z64} {z64} {z64} {z64}");
            Check(fails, "einsum ij,j->i (2,1).(1)", () => np.einsum("ij,j->i", F64(2).reshape(2, 1), np.ones(1)),
                  $"float64 (2,) {z64} {z64}");
            Check(fails, "einsum ij,jk->ik (2,1).(1,2)", () => np.einsum("ij,jk->ik", F64(2).reshape(2, 1), np.ones(new Shape(1, 2))),
                  $"float64 (2,2) {z64} {z64} {z64} {z64}");
            Check(fails, "einsum ij,jk->ik (1,1).(1,1)", () => np.einsum("ij,jk->ik", F64(1).reshape(1, 1), np.ones(new Shape(1, 1))),
                  "float64 (1,1) " + z64);
            Check(fails, "einsum i,i->i c128 (3).(3)", () => np.einsum("i,i->i", C128(3), np.ones(new Shape(3), np.complex128)),
                  $"complex128 (3,) {z64} {z64} {z64} {z64} {z64} {z64}");
            Check(fails, "einsum i,i->i [-0,1].[1,1]", () => np.einsum("i,i->i", np.array(new[] { -0.0, 1.0 }), np.ones(2)),
                  $"float64 (2,) {z64} 3ff0000000000000");
            AssertNoMismatch(fails, "NumPy's default einsum adds every term into a zero-filled output");
        }

        // ======================================================================================================
        //  GROUP 2 — add.accumulate copies the first element; NumSharp's flat scan starts from +0.0.
        // ======================================================================================================

        /// <summary>
        ///     The flat <c>np.cumsum</c> (no <c>axis</c>) must COPY the first element, as NumPy's
        ///     <c>add.accumulate</c> does, so a leading <c>-0.0</c> stays <c>-0.0</c> (and <c>-0.0 + -0.0</c> stays
        ///     <c>-0.0</c>). NumSharp returns <c>+0.0</c> at every such position, for every float dtype.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>DirectILKernelGenerator.Scan.cs</c> <c>CumSumHelperSameType</c> starts with
        ///     <c>T sum = default;</c> (<c>+0.0</c>) and adds every element to it; the IL scan paths that seed with
        ///     <c>EmitLoadZero</c> / <c>Complex.Zero</c> look the same. Fix: start from the first element and add
        ///     from the second. The axis forms (<c>(2,1) axis=0</c>, <c>(1,2) axis=1</c>, <c>(2,3)</c> both axes) and a
        ///     one-element input are already correct.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void CumSum_Flat_KeepsFirstNegativeZero_LikeNumPysAccumulate()
        {
            var fails = new List<string>();
            Check(fails, "cumsum f64 [-0,-0]", () => np.cumsum(F64(2)), "float64 (2,) 8000000000000000 8000000000000000");
            Check(fails, "cumsum f64 [-0,1]", () => np.cumsum(np.array(new[] { -0.0, 1.0 })), "float64 (2,) 8000000000000000 3ff0000000000000");
            Check(fails, "cumsum f32 [-0,-0]", () => np.cumsum(F32(2)), "float32 (2,) 80000000 80000000");
            Check(fails, "cumsum f16 [-0,-0]", () => np.cumsum(F16(2)), "float16 (2,) 8000 8000");
            Check(fails, "cumsum c128 [-0-0j]*2", () => np.cumsum(C128(2)),
                  "complex128 (2,) 8000000000000000 8000000000000000 8000000000000000 8000000000000000");
            AssertNoMismatch(fails, "NumPy's add.accumulate copies the first element instead of adding it to +0.0");
        }

        /// <summary>
        ///     The flat <c>np.nancumsum</c> must keep a leading <c>-0.0</c>, as NumPy's
        ///     <c>cumsum(where(isnan(a), 0, a))</c> does. NumSharp returns <c>+0.0</c>.
        /// </summary>
        /// <remarks>
        ///     Root cause: the same flat scan as <see cref="CumSum_Flat_KeepsFirstNegativeZero_LikeNumPysAccumulate"/>.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void NanCumSum_Flat_KeepsFirstNegativeZero()
        {
            var fails = new List<string>();
            Check(fails, "nancumsum f64 [-0,-0]", () => np.nancumsum(F64(2)), "float64 (2,) 8000000000000000 8000000000000000");
            AssertNoMismatch(fails, "NumPy's nancumsum is cumsum after the NaN replacement, which copies the first element");
        }

        /// <summary>
        ///     The complex128 <c>np.cumprod</c> and <c>np.nancumprod</c> (flat and per axis) must COPY the first
        ///     element, as NumPy's <c>multiply.accumulate</c> does, so <c>[-0-0j, -0-0j]</c> gives
        ///     <c>[(-0, -0), (+0, +0)]</c> and <c>[2-0j, 1+0j]</c> gives <c>[(2, -0), (2, +0)]</c>. NumSharp
        ///     multiplies the first element by the identity <c>1+0j</c>, so the first output is <c>(+0, -0)</c> /
        ///     <c>(2, +0)</c> and every later output inherits the change. Real dtypes are already correct (their
        ///     identity is exact), and so is a one-element input (the size-1 shortcut copies).
        /// </summary>
        /// <remarks>
        ///     Root cause: the complex scan in <c>DirectILKernelGenerator.Scan.cs</c> seeds its running product with
        ///     <c>Complex.One</c> and multiplies every element into it, including the first. The same seed shows in
        ///     the NaN-payload pin <c>OpenBugsNaNOperandOrderTests.CumsumCumprod_KeepNumPysNaNAtEveryStep</c>
        ///     (output 0 loses its imaginary payload); fix both together: copy element 0, multiply from element 1.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void CumProd_Complex_CopiesTheFirstElement()
        {
            var fails = new List<string>();
            // The complex words of one element: real then imaginary, m = -0.0, p = +0.0.
            const string mm = "8000000000000000 8000000000000000", pp = "0000000000000000 0000000000000000";
            const string pm = "0000000000000000 8000000000000000";
            Check(fails, "cumprod c128 [-0-0j]*2", () => np.cumprod(C128(2)), $"complex128 (2,) {mm} {pp}");
            Check(fails, "cumprod c128 (2,3) axis1", () => np.cumprod(C128(6).reshape(2, 3), 1),
                  $"complex128 (2,3) {mm} {pp} {pm} {mm} {pp} {pm}");
            Check(fails, "cumprod c128 [2-0j, 1+0j]",
                  () => np.cumprod(np.array(new[] { new Complex(2.0, -0.0), new Complex(1.0, 0.0) })),
                  "complex128 (2,) 4000000000000000 8000000000000000 4000000000000000 0000000000000000");
            Check(fails, "nancumprod c128 [-0-0j]*2", () => np.nancumprod(C128(2)), $"complex128 (2,) {mm} {pp}");
            AssertNoMismatch(fails, "NumPy's multiply.accumulate copies the first element instead of multiplying it by 1+0j");
        }

        // ======================================================================================================
        //  GROUP 3 — np.dot of two length-1 vectors is NumPy's plain product; the managed 1-D dot starts from 0.
        // ======================================================================================================

        /// <summary>
        ///     On the managed path (no BLAS backend), <c>np.dot</c> and <c>np.inner</c> of two length-1 vectors must
        ///     be the plain product <c>a * b</c>, as NumPy computes it, so <c>[-0.0]·[1.0]</c> is <c>-0.0</c> (and a
        ///     complex <c>(-0-0j)·(1+0j)</c> is <c>(+0, -0)</c>). NumSharp's managed dot returns <c>+0.0</c>.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Root cause: <c>Default.Dot.cs</c> sends 1-D·1-D to <c>DotInner1D</c>, a fused kernel whose sum starts
        ///     from 0. NumPy's <c>cblas_matrixproduct</c> (cblasfuncs.c) classifies a length-1 1-D operand as
        ///     <c>_scalar</c> and, for <c>l == 1</c>, writes <c>*ap2 * *ap1</c> (the naive complex product for
        ///     complex). <c>np.matmul</c> and <c>np.vdot</c> of the same vectors are already correct (NumPy's own
        ///     dot loop starts from <c>+0.0</c> there), and so is <c>(1,)·(1,3)</c> (NumPy's daxpy into a zeroed
        ///     output).
        ///     </para>
        ///     <para>
        ///     The test removes the BLAS backend for its duration because <c>NumSharp.Interop.OpenBLAS</c> already
        ///     ports the <c>l == 1</c> branch (<c>OpenBlasEngine.Dot.cs</c>): with the bundled library loaded the
        ///     product is right, and the bug would hide. MSTest runs this project serially, so swapping the
        ///     engine-wide property cannot race another test.
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Dot_LengthOneVectors_IsThePlainProduct_OnTheManagedPath()
        {
            var engine = BackendFactory.GetEngine();
            var saved = engine.Blas;
            engine.Blas = null;
            try
            {
                var fails = new List<string>();
                Check(fails, "dot f64 [-0].[1]", () => np.dot(F64(1), np.array(new[] { 1.0 })), "float64 () 8000000000000000");
                Check(fails, "dot f32 [-0].[1]", () => np.dot(F32(1), np.array(new[] { 1.0f })), "float32 () 80000000");
                Check(fails, "dot c128 [-0-0j].[1+0j]", () => np.dot(C128(1), np.array(new[] { new Complex(1, 0) })),
                      "complex128 () 0000000000000000 8000000000000000");
                Check(fails, "inner f64 [-0].[1]", () => np.inner(F64(1), np.array(new[] { 1.0 })), "float64 () 8000000000000000");
                AssertNoMismatch(fails, "NumPy computes a length-1 dot as the plain product, not as 0 + product");
            }
            finally
            {
                // Restore the engine-wide backend even when an assertion throws, or every later product test
                // in this process would silently run managed.
                engine.Blas = saved;
            }
        }

        /// <summary>
        ///     On the managed path, <c>np.dot</c> and <c>np.inner</c> with any size-1 operand must follow NumPy's
        ///     scalar branch: a 0-d operand times a longer array is a <c>daxpy</c> INTO A ZERO-FILLED output
        ///     (<c>+0.0 + s * b</c>, so <c>-0.0 · [1, 1]</c> is <c>[+0.0, +0.0]</c>), and two size-1 operands are
        ///     the plain product (<c>[[-0.0]] · [[1.0]]</c> is <c>[[-0.0]]</c>). NumSharp returns the plain
        ///     product (<c>-0.0</c>) for the first and a zero-started sum (<c>+0.0</c>) for the second, the
        ///     opposite of NumPy in both.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumPy's <c>cblas_matrixproduct</c> (cblasfuncs.c) runs for every float32 / float64 / complex
        ///     <c>np.dot</c> with operands of at most 2 dimensions, and <c>np.inner</c> reaches it through
        ///     <c>PyArray_MatrixProduct2</c>. It classifies any size-1 operand (0-d, <c>(1,)</c>, <c>(1,1)</c>) as
        ///     <c>_scalar</c>, zeroes the output, and then writes <c>a * b</c> when the result has one element
        ///     (<c>l == 1</c>) or calls <c>?axpy</c> into the zeroed output otherwise. Root cause:
        ///     <c>Default.Dot.cs</c> sends a 0-d operand to <c>Multiply</c> (a plain product) and a
        ///     <c>(1,1)·(1,1)</c> pair to <c>MultiplyMatrix</c> (a sum started from 0). The 0-d · 0-d product and
        ///     <c>(1,1)·(1,3)</c> (an <c>axpy</c> into zeros) already match.
        ///     </para>
        ///     <para>
        ///     The BLAS backend is removed for the test's duration, as in
        ///     <see cref="Dot_LengthOneVectors_IsThePlainProduct_OnTheManagedPath"/>: the OpenBLAS package ports
        ///     <c>cblas_matrixproduct</c> route for route and would hide the managed bug.
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Dot_SizeOneOperand_FollowsNumPysScalarBranch_OnTheManagedPath()
        {
            var engine = BackendFactory.GetEngine();
            var saved = engine.Blas;
            engine.Blas = null;
            try
            {
                var fails = new List<string>();
                const string z64 = "0000000000000000";
                Check(fails, "dot f64 0-d -0 . [1,1]", () => np.dot(NDArray.Scalar(-0.0), np.ones(2)), $"float64 (2,) {z64} {z64}");
                Check(fails, "dot f64 [[-0]] . [[1]]", () => np.dot(F64(1).reshape(1, 1), np.ones(new Shape(1, 1))),
                      "float64 (1,1) 8000000000000000");
                Check(fails, "dot f32 [[-0]] . [[1]]", () => np.dot(F32(1).reshape(1, 1), np.ones(new Shape(1, 1), np.float32)),
                      "float32 (1,1) 80000000");
                Check(fails, "dot c128 0-d (-0-0j) . [1,1]",
                      () => np.dot(NDArray.Scalar(new Complex(-0.0, -0.0)), np.ones(new Shape(2), np.complex128)),
                      $"complex128 (2,) {z64} {z64} {z64} {z64}");
                Check(fails, "dot c128 [[-0-0j]] . [[1+0j]]",
                      () => np.dot(C128(1).reshape(1, 1), np.ones(new Shape(1, 1), np.complex128)),
                      $"complex128 (1,1) {z64} 8000000000000000");
                Check(fails, "inner f64 0-d -0 . [1,1]", () => np.inner(NDArray.Scalar(-0.0), np.ones(2)), $"float64 (2,) {z64} {z64}");
                Check(fails, "inner f64 [[-0]] . [[1]]", () => np.inner(F64(1).reshape(1, 1), np.ones(new Shape(1, 1))),
                      "float64 (1,1) 8000000000000000");
                AssertNoMismatch(fails, "NumPy's cblas_matrixproduct treats a size-1 operand as a scalar");
            }
            finally
            {
                // Restore the engine-wide backend even when an assertion throws (see the test above).
                engine.Blas = saved;
            }
        }

        // ======================================================================================================
        //  GROUP 4 — the nan-reduction size-1 shortcuts return the input's clone: (1,) for a one-element 1-D
        //  array, the input dtype for nanmean (one-element or 0-d), and the sign of a -0.0.
        // ======================================================================================================

        /// <summary>
        ///     <c>np.nansum</c> and <c>np.nanprod</c> of a one-element 1-D float array must return a 0-d result, and
        ///     <c>np.nansum</c> of a <c>-0.0</c> (one-element 1-D or 0-d) must be <c>+0.0</c> (it is <c>np.sum</c>
        ///     after the NaN replacement). NumSharp returns the input's clone: shape <c>(1,)</c> for the 1-D input,
        ///     sign included for both. A NaN element already gives the right 0-d <c>0</c> / <c>1</c>,
        ///     <c>keepdims=True</c> is right by accident (its shape IS <c>(1,)</c>), and <c>np.nanprod</c> of a 0-d
        ///     <c>-0.0</c> is <c>-0.0</c> on both sides (its identity is 1).
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Default.Reduction.Nan.cs</c> <c>NanSum</c> / <c>NanProd</c> take
        ///     <c>shape.IsScalar || (shape.size == 1 &amp;&amp; shape.NDim == 1)</c> into a shortcut that ends with
        ///     <c>return a.Clone();</c> — no reshape to 0-d (the sum's <c>HandleScalarReduction</c> does reshape), no
        ///     <c>keepdims</c> handling and no <c>+0.0</c> start. <c>nanmax</c>/<c>nanmin</c> use
        ///     <c>HandleScalarReduction</c> and are correct. Empty inputs already match NumPy (probed: the 0-d
        ///     <c>0.0</c> / <c>1.0</c>, and the <c>(3,)</c> / <c>(0,)</c> axis results of a <c>(0, 3)</c> array);
        ///     the <c>if (shape.IsEmpty) return arr;</c> line above the shortcut tests an UNINITIALIZED shape, not
        ///     size 0.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void NanSumNanProd_OneElement_ReturnZeroDimensionalResult()
        {
            var fails = new List<string>();
            Check(fails, "nansum f64 [1]", () => np.nansum(np.array(new[] { 1.0 })), "float64 () 3ff0000000000000");
            Check(fails, "nansum f32 [2]", () => np.nansum(np.array(new[] { 2.0f })), "float32 () 40000000");
            Check(fails, "nansum f64 [-0]", () => np.nansum(F64(1)), "float64 () 0000000000000000");
            Check(fails, "nansum f64 0-d -0", () => np.nansum(NDArray.Scalar(-0.0)), "float64 () 0000000000000000");
            Check(fails, "nanprod f64 [3]", () => np.nanprod(np.array(new[] { 3.0 })), "float64 () 4008000000000000");
            // 2026-10-04 probe: the same shortcut with axis=0, and the float16 / float32 lanes.
            Check(fails, "nansum f64 [1] axis0", () => np.nansum(np.array(new[] { 1.0 }), 0), "float64 () 3ff0000000000000");
            Check(fails, "nanprod f64 [3] axis0", () => np.nanprod(np.array(new[] { 3.0 }), 0), "float64 () 4008000000000000");
            Check(fails, "nansum f16 [-0]", () => np.nansum(F16(1)), "float16 () 0000");
            Check(fails, "nansum f32 0-d -0", () => np.nansum(NDArray.Scalar(-0.0f)), "float32 () 00000000");
            Check(fails, "nanprod f32 [2]", () => np.nanprod(np.array(new[] { 2.0f })), "float32 () 40000000");
            AssertNoMismatch(fails, "NumPy's nansum / nanprod reduce to a 0-d result, starting from the identity");
        }

        /// <summary>
        ///     <c>np.nansum</c> over a length-1 axis must give <c>+0.0</c> for a <c>-0.0</c> slab (it is
        ///     <c>np.sum</c> after the NaN replacement). NumSharp copies the slab, for float64 and float32, plain
        ///     and with <c>keepdims</c>, and for a transposed operand too.
        /// </summary>
        /// <remarks>
        ///     Root cause: the float32 / float64 axis kernel (<c>DirectILKernelGenerator.Reduction.Axis.NaN.cs</c>)
        ///     sends a reduced axis of stride 1 to <c>NanReduceContiguousAxis</c>, whose <c>if (size == 1)</c>
        ///     branch returns the lone element itself: the same copy as
        ///     <see cref="Sum_LengthOneAxis_IsPositiveZero"/>. A length-1 axis with any other stride takes
        ///     <c>NanReduceStridedAxis</c>, which starts from <c>0</c> and is already correct; that is why the
        ///     probe's <c>(1,3)</c> axis 0 in C order (stride 3) matched. The oracle carve keeps the whole
        ///     length-1-axis class out until the path is fixed.
        /// </remarks>
        [TestMethod, OpenBugs]
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
        ///     <c>np.nansum</c> of a 1-D complex128 array along <c>axis=0</c> must return the 0-d sum
        ///     (<c>np.nansum([1+2j, 3+4j], axis=0)</c> is <c>4+6j</c>). NumSharp throws
        ///     <c>InvalidOperationException</c> ("Can't construct NDCoordinatesAxisIncrementor with a vector
        ///     shape."), for every length and layout. The flat form and 2-D axis reductions already work.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Default.Reduction.Nan.cs</c> <c>NanSumComplex</c> walks the output with an
        ///     <c>NDCoordinatesAxisIncrementor</c>, which refuses a 1-D shape (the reduced output of a 1-D input has
        ///     no coordinates left). A 1-D axis reduction is the flat reduction plus <c>keepdims</c>; route it
        ///     there.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void NanSum_Complex1D_AlongAxis0_ReturnsTheSum()
        {
            var fails = new List<string>();
            var pair = np.array(new[] { new Complex(1, 2), new Complex(3, 4) });
            Check(fails, "nansum c128 [1+2j,3+4j] axis0", () => np.nansum(pair, 0), "complex128 () 4010000000000000 4018000000000000");
            Check(fails, "nansum c128 [1+2j] axis0", () => np.nansum(np.array(new[] { new Complex(1, 2) }), 0),
                  "complex128 () 3ff0000000000000 4000000000000000");
            Check(fails, "nansum c128 [1+2j,3+4j][::-1] axis0", () => np.nansum(pair["::-1"], 0),
                  "complex128 () 4010000000000000 4018000000000000");
            Check(fails, "nansum c128 [-0-0j]*2 axis0 keepdims", () => np.nansum(C128(2), 0, true),
                  "complex128 (1,) 0000000000000000 0000000000000000");
            AssertNoMismatch(fails, "NumPy's 1-D axis=0 nansum is the flat nansum");
        }

        /// <summary>
        ///     <c>np.nanmean</c>, <c>np.nanstd</c> and <c>np.nanvar</c> of a 1-D array along <c>axis=0</c> must
        ///     return a 0-d result (shape <c>(1,)</c> with <c>keepdims</c>). NumSharp returns <c>(1,)</c>, and
        ///     <c>(1,1)</c> with <c>keepdims</c>, for float64, float32, float16 and complex128. A one-element
        ///     input with <c>keepdims</c> has the opposite bug in <c>nanstd</c> / <c>nanvar</c>: NumSharp drops
        ///     <c>keepdims</c> and returns 0-d where NumPy returns <c>(1,)</c>.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Statistics/np.nanmean.cs</c> <c>nanmean_axis</c> builds the output shape from the
        ///     non-reduced axes and adds a <c>1</c> when none is left (<c>if (outputShapeList.Count == 0)
        ///     outputShapeList.Add(1);</c>), and <c>keepdims</c> then expands that <c>(1,)</c> again; <c>nanstd</c> /
        ///     <c>nanvar</c> share the pattern. Their one-element shortcut returns a scalar without applying
        ///     <c>keepdims</c>. Values are right in every case below.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void NanMeanStdVar_1DAxis_ReturnNumPysShape()
        {
            var fails = new List<string>();
            var v = np.array(new[] { 1.0, 2.0, 3.0 });
            Check(fails, "nanmean f64 [1,2,3] axis0", () => np.nanmean(v, 0), "float64 () 4000000000000000");
            Check(fails, "nanmean f64 [1,2,3] axis0 keepdims", () => np.nanmean(v, 0, true), "float64 (1,) 4000000000000000");
            Check(fails, "nanmean c128 [1+2j,3+4j] axis0", () => np.nanmean(np.array(new[] { new Complex(1, 2), new Complex(3, 4) }), 0),
                  "complex128 () 4000000000000000 4008000000000000");
            Check(fails, "nanstd f64 [1,2,3] axis0", () => np.nanstd(v, 0), "float64 () 3fea20bd700c2c3e");
            Check(fails, "nanvar f64 [1,2,3] axis0 keepdims", () => np.nanvar(v, 0, true), "float64 (1,) 3fe5555555555555");
            Check(fails, "nanstd f64 [5] keepdims", () => np.nanstd(np.array(new[] { 5.0 }), keepdims: true), "float64 (1,) 0000000000000000");
            Check(fails, "nanvar f64 [5] axis0 keepdims", () => np.nanvar(np.array(new[] { 5.0 }), 0, true), "float64 (1,) 0000000000000000");
            AssertNoMismatch(fails, "NumPy reduces the only axis of a 1-D array to a 0-d result");
        }

        /// <summary>
        ///     <c>np.nanmean</c> of a one-element 1-D array or a 0-d array must return a 0-d result, float64 for an
        ///     integer input, and <c>+0.0</c> for a <c>-0.0</c>. NumSharp returns the input's clone: shape
        ///     <c>(1,)</c> for the 1-D input, the integer dtype kept (<c>np.nanmean([5])</c> is an int32 <c>[5]</c>
        ///     and <c>np.nanmean(np.int32(5))</c> an int32 <c>5</c>, instead of the float64 <c>5.0</c>), and the sign
        ///     of a <c>-0.0</c> kept. A 0-d float32 input already gives NumPy's float32 0-d result.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Statistics/np.nanmean.cs</c> returns <c>a.Clone().MarkReductionScalar()</c> when
        ///     <c>shape.IsScalar || (shape.size == 1 &amp;&amp; shape.NDim == 1)</c>, which neither reshapes to 0-d nor
        ///     applies <c>nanmean</c>'s float64 result dtype for integers nor starts the sum from <c>+0.0</c>. Empty
        ///     inputs already match NumPy (probed: 0-d <c>nan</c> in the input's float dtype, float64 for int32,
        ///     and the <c>(3,)</c> / <c>(0,)</c> axis results). The NaN case is compared as a token because NumPy's
        ///     NaN comes from <c>0/0</c> and its sign bit is not part of the contract.
        /// </remarks>
        [TestMethod, OpenBugs]
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

        // ======================================================================================================
        //  GROUP 5 — np.var / np.std at a degenerate size. NumPy's _var always computes
        //  sum((x - mean)²) / max(n - ddof, 0) in the input's float dtype; NumSharp's shortcuts skip that.
        //  np.mean's empty path shares the float64 hard-code of the var/std one.
        // ======================================================================================================

        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c> of ONE element (a one-element 1-D array or a 0-d array, flat, over its
        ///     one axis, or with <c>keepdims</c>) must be NaN when the element is NaN or ±inf, because NumPy computes
        ///     <c>x - mean</c> (<c>nan - nan</c>, <c>inf - inf</c>). NumSharp returns <c>0</c> for every element; a
        ///     finite element already gives NumPy's <c>0</c>. A NaN or inf in one-element data therefore vanishes
        ///     from a variance, which turns a missing-value marker into "no spread".
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Default.Reduction.Var.cs</c> <c>ReduceVar</c> and <c>Default.Reduction.Std.cs</c>
        ///     <c>ReduceStd</c> send <c>shape.IsScalar || (shape.size == 1 &amp;&amp; shape.NDim == 1)</c> to a
        ///     shortcut that returns <c>(arr.size - ddof) &lt;= 0 ? NaN : 0.0</c> WITHOUT reading the element, and
        ///     <c>var_elementwise</c> / <c>std_elementwise</c> repeat it. NumPy's <c>_var</c>
        ///     (<c>numpy/_core/_methods.py</c>) has no size-1 path; for complex it sums
        ///     <c>x.real² + x.imag²</c>, NaN for a NaN or infinite part. The oracle excused these corpus cases
        ///     (<c>scalar_0d</c> / <c>one_element_1d</c> × float64 / complex128 in <c>reduce.jsonl</c> and
        ///     <c>instance.jsonl</c>) as "summation precision" (flat) and as the min/max "axis SIMD skips NaN"
        ///     bug (<c>axis=0</c>) before its registry named this bug.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void VarStd_OneElement_NonFiniteValue_IsNaN()
        {
            var fails = new List<string>();
            Check(fails, "var f64 [nan]", () => np.var(Of(double.NaN)), "float64 () nan");
            Check(fails, "var f64 [inf]", () => np.var(Of(double.PositiveInfinity)), "float64 () nan");
            Check(fails, "var f64 [-inf]", () => np.var(Of(double.NegativeInfinity)), "float64 () nan");
            Check(fails, "std f64 [inf]", () => np.std(Of(double.PositiveInfinity)), "float64 () nan");
            Check(fails, "var f64 0-d nan", () => np.var(NDArray.Scalar(double.NaN)), "float64 () nan");
            Check(fails, "std f64 0-d -inf", () => np.std(NDArray.Scalar(double.NegativeInfinity)), "float64 () nan");
            Check(fails, "var f64 [nan] axis0", () => np.var(Of(double.NaN), 0), "float64 () nan");
            Check(fails, "var f64 [inf] keepdims", () => np.var(Of(double.PositiveInfinity), keepdims: true), "float64 (1,) nan");
            Check(fails, "var c128 [nan+0j]", () => np.var(OfComplex(new Complex(double.NaN, 0))), "float64 () nan");
            Check(fails, "std c128 [0+infj]", () => np.std(OfComplex(new Complex(0, double.PositiveInfinity))), "float64 () nan");
            Check(fails, "var f64 [5] (finite: already right)", () => np.var(Of(5.0)), "float64 () 0000000000000000");
            AssertNoMismatch(fails, "NumPy's _var computes x - mean even for one element, so NaN and inf propagate");
        }

        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c> over an axis of length 1 must be NaN at every output whose element is
        ///     NaN or ±inf (and <c>0</c> elsewhere), for every float dtype, complex, both axis positions and
        ///     <c>keepdims</c>. NumSharp returns zeros without reading the slab, so the non-finite outputs read
        ///     <c>0</c>.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>ReduceVar</c> / <c>ReduceStd</c> send an extent-1 reduced axis
        ///     (<c>shape[axis] == 1</c>) to <c>AllocateReductionZeros</c>. NumPy runs <c>x - mean</c> per output
        ///     like any other axis. Axes of length 2 or more take the IL axis kernel
        ///     (<c>ExecuteAxisVarReductionIL</c>) and propagate NaN and inf already.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void VarStd_LengthOneAxis_NonFiniteValue_IsNaN()
        {
            var fails = new List<string>();
            const string nanThenZeros64 = "nan 0000000000000000 0000000000000000 0000000000000000";
            Check(fails, "var f64 (1,4) axis0 nan", () => np.var(Of(double.NaN, 2, 3, -4).reshape(1, 4), 0),
                  "float64 (4,) " + nanThenZeros64);
            Check(fails, "std f64 (4,1) axis1 inf", () => np.std(Of(double.PositiveInfinity, 2, 3, -4).reshape(4, 1), 1),
                  "float64 (4,) " + nanThenZeros64);
            Check(fails, "var f64 (2,1,2) axis1 -inf", () => np.var(Of(double.NegativeInfinity, 2, 3, -4).reshape(2, 1, 2), 1),
                  "float64 (2,2) " + nanThenZeros64);
            Check(fails, "var f64 (1,4) axis0 nan keepdims", () => np.var(Of(double.NaN, 2, 3, -4).reshape(1, 4), 0, keepdims: true),
                  "float64 (1,4) " + nanThenZeros64);
            Check(fails, "var f32 (1,4) axis0 nan", () => np.var(OfSingle(float.NaN, 2, 3, -4).reshape(1, 4), 0),
                  "float32 (4,) nan 00000000 00000000 00000000");
            Check(fails, "std f16 (1,4) axis0 inf", () => np.std(OfHalf(double.PositiveInfinity, 2, 3, -4).reshape(1, 4), 0),
                  "float16 (4,) nan 0000 0000 0000");
            Check(fails, "var c128 (1,4) axis0 nanj",
                  () => np.var(OfComplex(new Complex(0, double.NaN), new Complex(2, -1), new Complex(3, 0.5), new Complex(-4, 4)).reshape(1, 4), 0),
                  "float64 (4,) " + nanThenZeros64);
            AssertNoMismatch(fails, "NumPy reduces a length-1 axis with the same arithmetic as any other axis");
        }

        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c> over an axis of length 1 with <c>ddof &gt;= 1</c> must be NaN at EVERY
        ///     output: NumPy divides the zero sum of squares by <c>max(1 - ddof, 0) = 0</c>. NumSharp returns zeros,
        ///     so a sample variance (<c>ddof=1</c>) of single observations reads as "no spread" instead of
        ///     "undefined".
        /// </summary>
        /// <remarks>
        ///     Root cause: the same <c>AllocateReductionZeros</c> shortcut as
        ///     <see cref="VarStd_LengthOneAxis_NonFiniteValue_IsNaN"/>; it never looks at <c>ddof</c>. The one-element
        ///     FLAT shortcut does check <c>size - ddof &lt;= 0</c>, and a <c>(1, 1)</c> float64 array reduced flat is
        ///     2-D, so it skips that shortcut and its contiguous route (<c>VarSimdHelper</c>) returns NaN for
        ///     <c>ddof &gt;= n</c>; both already give NumPy's NaN there. Every dtype is affected over a length-1 axis,
        ///     the integer and bool inputs (float64 results) included.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void VarStd_LengthOneAxis_DdofOfOneOrMore_IsNaN()
        {
            var fails = new List<string>();
            Check(fails, "var f64 (1,4) axis0 ddof1", () => np.var(Of(5, 2, 3, -4).reshape(1, 4), 0, ddof: 1), "float64 (4,) nan nan nan nan");
            Check(fails, "std f64 (1,4) axis0 ddof2", () => np.std(Of(5, 2, 3, -4).reshape(1, 4), 0, ddof: 2), "float64 (4,) nan nan nan nan");
            Check(fails, "std f32 (4,1) axis1 ddof1", () => np.std(OfSingle(5, 2, 3, -4).reshape(4, 1), 1, ddof: 1), "float32 (4,) nan nan nan nan");
            Check(fails, "var f16 (2,1,2) axis1 ddof1", () => np.var(OfHalf(5, 2, 3, -4).reshape(2, 1, 2), 1, ddof: 1), "float16 (2,2) nan nan nan nan");
            Check(fails, "var i32 (1,4) axis0 ddof1", () => np.var(OfInt32(5, 2, 3, -4).reshape(1, 4), 0, ddof: 1), "float64 (4,) nan nan nan nan");
            Check(fails, "var bool (1,3) axis0 ddof1", () => np.var(OfBool(true, false, true).reshape(1, 3), 0, ddof: 1), "float64 (3,) nan nan nan");
            Check(fails, "var c128 (1,2) axis0 ddof1",
                  () => np.var(OfComplex(new Complex(1, 2), new Complex(2, -1)).reshape(1, 2), 0, ddof: 1), "float64 (2,) nan nan");
            Check(fails, "var f64 (1,1) axis1 ddof1 keepdims", () => np.var(Of(5.0).reshape(1, 1), 1, keepdims: true, ddof: 1), "float64 (1,1) nan");
            AssertNoMismatch(fails, "NumPy divides by max(n - ddof, 0), which is 0 for one element and ddof >= 1");
        }

        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c> of one element must keep a float16/float32 input's dtype and compute in
        ///     the <c>dtype=</c> argument when one is given. NumSharp returns float64 in every such case, so the
        ///     result type of a float32 pipeline changes when a slice happens to hold one element.
        /// </summary>
        /// <remarks>
        ///     Root cause: the one-element shortcut of <see cref="VarStd_OneElement_NonFiniteValue_IsNaN"/> wraps a C#
        ///     <c>double</c> (<c>NDArray.Scalar(value)</c>), so the result is float64 whatever the input dtype and
        ///     <c>dtype=</c> is ignored. NumPy's <c>_var</c> keeps a float16/float32 input in its own dtype (the
        ///     mean, the squares and the division) and computes in <c>dtype=</c> when it is given. Flat reductions of
        ///     two or more elements and the axis reductions keep the dtype already.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void VarStd_OneElement_KeepsFloat16Float32_AndHonoursTheDtypeArgument()
        {
            var fails = new List<string>();
            Check(fails, "var f32 [5]", () => np.var(OfSingle(5)), "float32 () 00000000");
            Check(fails, "var f16 [5]", () => np.var(OfHalf(5)), "float16 () 0000");
            Check(fails, "std f32 0-d 5", () => np.std(NDArray.Scalar(5f)), "float32 () 00000000");
            Check(fails, "var f32 [5] axis0", () => np.var(OfSingle(5), 0), "float32 () 00000000");
            Check(fails, "var f32 [5] ddof1", () => np.var(OfSingle(5), ddof: 1), "float32 () nan");
            Check(fails, "std f16 [nan]", () => np.std(OfHalf(double.NaN)), "float16 () nan");
            Check(fails, "var f64 [5] dtype=f32", () => np.var(Of(5.0), dtype: np.float32), "float32 () 00000000");
            Check(fails, "std f64 [5] axis0 dtype=f32", () => np.std(Of(5.0), 0, dtype: np.float32), "float32 () 00000000");
            Check(fails, "var i32 [5] dtype=f16", () => np.var(OfInt32(5), dtype: np.float16), "float16 () 0000");
            AssertNoMismatch(fails, "NumPy's _var computes in the input's float dtype, or in dtype= when given");
        }

        /// <summary>
        ///     A flat <c>np.var</c> / <c>np.std</c> of a contiguous float32, float64 or integer array with
        ///     <c>ddof &gt;= n</c> must be <c>+inf</c>: NumPy divides the positive sum of squares by
        ///     <c>max(n - ddof, 0) = 0</c> (only a slice whose squares sum to zero gives <c>0/0 = NaN</c>). NumSharp
        ///     returns NaN for every such case.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>DirectILKernelGenerator.Masking.VarStd.cs</c> <c>VarSimdHelper</c> returns NaN as soon
        ///     as <c>size &lt;= ddof</c>, before it sums anything. The helper serves the contiguous
        ///     float32/float64/integer flat route; float16, complex, bool and strided inputs take
        ///     <c>var_elementwise_fallback</c> (see
        ///     <see cref="VarStd_DdofPastTheCount_GeneralRoute_IsInfinityNotNegative"/>), and the axis kernels clamp
        ///     the divisor with <c>Math.Max(axisSize - ddof, 0)</c>, so both are right at <c>ddof == n</c>. The two
        ///     NaN cases at the end are already right and stay as guards for the fix.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void VarStd_DdofAtLeastTheCount_ContiguousRoute_IsInfinity()
        {
            var fails = new List<string>();
            const string inf64 = "float64 () 7ff0000000000000";
            Check(fails, "var f64 [1,2] ddof2", () => np.var(Of(1, 2), ddof: 2), inf64);
            Check(fails, "std f64 [1,2] ddof2", () => np.std(Of(1, 2), ddof: 2), inf64);
            Check(fails, "var f64 [1,2,4] ddof3", () => np.var(Of(1, 2, 4), ddof: 3), inf64);
            Check(fails, "var f64 [1,2] ddof3", () => np.var(Of(1, 2), ddof: 3), inf64);
            Check(fails, "std f64 [1,2] ddof3", () => np.std(Of(1, 2), ddof: 3), inf64);
            Check(fails, "var f64 (2,3) ddof7", () => np.var(Of(1, 2, 4, 8, 16, 32).reshape(2, 3), ddof: 7), inf64);
            Check(fails, "var f32 [1,2,4] ddof4", () => np.var(OfSingle(1, 2, 4), ddof: 4), "float32 () 7f800000");
            Check(fails, "std f32 [1,2] ddof2", () => np.std(OfSingle(1, 2), ddof: 2), "float32 () 7f800000");
            Check(fails, "var i32 [1,2] ddof2", () => np.var(OfInt32(1, 2), ddof: 2), inf64);
            Check(fails, "var i32 [1,2] ddof3", () => np.var(OfInt32(1, 2), ddof: 3), inf64);
            Check(fails, "var f64 [3,3] ddof2 (constant slice: 0/0, already right)", () => np.var(Of(3, 3), ddof: 2), "float64 () nan");
            Check(fails, "var f64 (1,1) ddof2 (one element: 0/0, already right)", () => np.var(Of(5.0).reshape(1, 1), ddof: 2), "float64 () nan");
            AssertNoMismatch(fails, "NumPy divides a positive sum of squares by max(n - ddof, 0) = 0, which is +inf");
        }

        /// <summary>
        ///     A flat <c>np.var</c> / <c>np.std</c> with <c>ddof &gt; n</c> on the general route (float16, complex,
        ///     bool, a strided view) must be <c>+inf</c> (NaN when the squares sum to zero). NumSharp divides by the
        ///     NEGATIVE <c>n - ddof</c>: <c>var</c> comes out negative (<c>var([1, 2], ddof=3)</c> is <c>-0.5</c>),
        ///     <c>std</c> is the NaN square root of it, and a zero sum gives <c>-0.0</c>. A negative variance is
        ///     impossible, so anything downstream that trusts its sign is wrong.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>var_elementwise_fallback</c>'s <c>VarMomentsReal</c> / <c>VarMomentsBool</c> /
        ///     <c>VarMomentsComplex</c> return <c>sq / (n - ddof)</c> without NumPy's clamp
        ///     (<c>um.maximum(rcount - ddof, 0)</c> in <c>_var</c>). <c>ddof == n</c> divides by an exact 0 and is
        ///     already <c>+inf</c> (the last case stays as a guard for the fix).
        /// </remarks>
        [TestMethod, OpenBugs]
        public void VarStd_DdofPastTheCount_GeneralRoute_IsInfinityNotNegative()
        {
            var fails = new List<string>();
            const string inf64 = "float64 () 7ff0000000000000";
            Check(fails, "var f64 [1,2,4] strided ddof4", () => np.var(Of(1, 0, 2, 0, 4)["::2"], ddof: 4), inf64);
            Check(fails, "std f64 [1,2,4] strided ddof4", () => np.std(Of(1, 0, 2, 0, 4)["::2"], ddof: 4), inf64);
            Check(fails, "var f16 [1,2] ddof3", () => np.var(OfHalf(1, 2), ddof: 3), "float16 () 7c00");
            Check(fails, "std f16 [1,2,4] ddof4", () => np.std(OfHalf(1, 2, 4), ddof: 4), "float16 () 7c00");
            Check(fails, "var c128 [1,2] ddof3", () => np.var(OfComplex(1, 2), ddof: 3), inf64);
            Check(fails, "std bool [T,F] ddof3", () => np.std(OfBool(true, false), ddof: 3), inf64);
            Check(fails, "var f16 (1,1) ddof2", () => np.var(OfHalf(5).reshape(1, 1), ddof: 2), "float16 () nan");
            Check(fails, "var f64 [1,2,4] strided ddof3 (ddof == n: already right)", () => np.var(Of(1, 0, 2, 0, 4)["::2"], ddof: 3), inf64);
            AssertNoMismatch(fails, "NumPy clamps the divisor at 0, so a variance is never negative");
        }

        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c> of an EMPTY float16/float32 array must return NaN (or an empty result) in
        ///     the input's dtype, flat and per axis, and in the <c>dtype=</c> argument when one is given. NumSharp
        ///     returns float64. float64, complex and integer inputs already match (their result is float64).
        /// </summary>
        /// <remarks>
        ///     Root cause: the empty-input path of <c>ReduceVar</c> / <c>ReduceStd</c> returns
        ///     <c>NDArray.Scalar(double.NaN)</c> for a flat reduction and allocates
        ///     <c>typeCode ?? NPTypeCode.Double</c> per axis, so it never consults the input's float dtype, and the
        ///     flat branch also drops <c>dtype=</c>. NumPy's <c>_var</c> runs its arithmetic on the empty slice in
        ///     the input's float dtype: the empty float16/float32 sum divided by the zero count is a NaN of that
        ///     dtype.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void VarStd_EmptyFloat16Float32_KeepTheirDtype()
        {
            var fails = new List<string>();
            Check(fails, "var f16 (0,3)", () => np.var(np.zeros(new Shape(0, 3), np.float16)), "float16 () nan");
            Check(fails, "std f32 (0,3)", () => np.std(np.zeros(new Shape(0, 3), np.float32)), "float32 () nan");
            Check(fails, "var f16 (0,3) axis0", () => np.var(np.zeros(new Shape(0, 3), np.float16), 0), "float16 (3,) nan nan nan");
            Check(fails, "std f32 (0,3) axis0", () => np.std(np.zeros(new Shape(0, 3), np.float32), 0), "float32 (3,) nan nan nan");
            Check(fails, "var f32 (0,3) axis1", () => np.var(np.zeros(new Shape(0, 3), np.float32), 1), "float32 (0,) ");
            Check(fails, "var f16 (0,3) axis0 keepdims", () => np.var(np.zeros(new Shape(0, 3), np.float16), 0, keepdims: true),
                  "float16 (1,3) nan nan nan");
            Check(fails, "var f64 (0,) dtype=f32", () => np.var(np.zeros(new Shape(0), np.float64), dtype: np.float32), "float32 () nan");
            Check(fails, "var f64 (0,3) axis0 (already right)", () => np.var(np.zeros(new Shape(0, 3), np.float64), 0), "float64 (3,) nan nan nan");
            Check(fails, "var c128 (0,3) axis0 (already right)", () => np.var(np.zeros(new Shape(0, 3), np.complex128), 0), "float64 (3,) nan nan nan");
            Check(fails, "var i32 (0,3) (already right)", () => np.var(np.zeros(new Shape(0, 3), np.int32)), "float64 () nan");
            AssertNoMismatch(fails, "NumPy's empty variance is NaN in the input's float dtype");
        }

        /// <summary>
        ///     <c>np.mean</c> of an EMPTY float16, float32 or complex128 array must keep the input's dtype (a NaN, or
        ///     an empty result), flat and per axis, as <c>np.var</c> must in
        ///     <see cref="VarStd_EmptyFloat16Float32_KeepTheirDtype"/>. NumSharp returns float64. float64 and integer
        ///     inputs already match (their mean is float64).
        /// </summary>
        /// <remarks>
        ///     Root cause: the same empty-input pattern in <c>Default.Reduction.Mean.cs</c> <c>ReduceMean</c>: the
        ///     flat branch returns <c>NDArray.Scalar(double.NaN)</c> and the axis branch allocates
        ///     <c>typeCode ?? NPTypeCode.Double</c>. NumPy's <c>_mean</c> divides the empty sum (in the input's
        ///     dtype) by the zero count. A one-element float32 mean already keeps float32. Found on 2026-10-04: these
        ///     are the 18 <c>reduce.jsonl</c> cases the oracle still excuses under the generic "reduction result dtype
        ///     differs" branch now that the var/std cases have their own.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Mean_EmptyFloat16Float32Complex_KeepTheirDtype()
        {
            var fails = new List<string>();
            Check(fails, "mean f16 (0,3)", () => np.mean(np.zeros(new Shape(0, 3), np.float16)), "float16 () nan");
            Check(fails, "mean f32 (0,3) axis0", () => np.mean(np.zeros(new Shape(0, 3), np.float32), 0), "float32 (3,) nan nan nan");
            Check(fails, "mean f32 (0,3) axis1", () => np.mean(np.zeros(new Shape(0, 3), np.float32), 1), "float32 (0,) ");
            Check(fails, "mean f16 (0,3) axis0 keepdims", () => np.mean(np.zeros(new Shape(0, 3), np.float16), 0, keepdims: true),
                  "float16 (1,3) nan nan nan");
            Check(fails, "mean c128 (0,3) axis1", () => np.mean(np.zeros(new Shape(0, 3), np.complex128), 1), "complex128 (0,) ");
            Check(fails, "mean f64 (0,3) axis0 (already right)", () => np.mean(np.zeros(new Shape(0, 3), np.float64), 0), "float64 (3,) nan nan nan");
            Check(fails, "mean i32 (0,3) (already right)", () => np.mean(np.zeros(new Shape(0, 3), np.int32)), "float64 () nan");
            AssertNoMismatch(fails, "NumPy's empty mean is NaN in the input's float or complex dtype");
        }

        // ======================================================================================================
        //  GROUP 6 — reductions that return the wrong shape or values (found by the 2026-10-04 reduction-start
        //  probe; its one-element float32/float16 std/var dtype finding is group 5's
        //  VarStd_OneElement_KeepsFloat16Float32_AndHonoursTheDtypeArgument).
        // ======================================================================================================

        /// <summary>
        ///     The flat <c>np.cumprod</c> / <c>np.nancumprod</c> of a size-1 array with two or more dimensions
        ///     (<c>[[2.0]]</c>, a <c>(1,1,1)</c> array) must return a 1-D <c>(1,)</c> array: a scan never returns
        ///     0-d. NumSharp returns 0-d. <c>np.cumsum</c> of the same input is already correct.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Default.Reduction.CumMul.cs</c> <c>ReduceCumMul</c>'s flat branch reshapes a
        ///     one-element 1-D result to <c>Shape.Scalar</c> (<c>if (... r.Shape.size == 1 &amp;&amp; r.ndim == 1)
        ///     r.Storage.Reshape(Shape.Scalar);</c>), a line that belongs to a reduction, not a scan. A size-1 input
        ///     with one dimension or none never reaches it (the shortcut above returns <c>(1,)</c>), so only an
        ///     N-D size-1 input shows it. The complex <c>[[-0-0j]]</c> case would also show
        ///     <see cref="CumProd_Complex_CopiesTheFirstElement"/>, so the cases here are real and integer.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void CumProd_FlatOfOneByOne_ReturnsOneDimensional()
        {
            var fails = new List<string>();
            Check(fails, "cumprod f64 [[2]] flat", () => np.cumprod(np.array(new double[,] { { 2.0 } })), "float64 (1,) 4000000000000000");
            Check(fails, "cumprod f32 [[-0]] flat", () => np.cumprod(F32(1).reshape(1, 1)), "float32 (1,) 80000000");
            Check(fails, "cumprod f64 (1,1,1) flat", () => np.cumprod(np.full(new Shape(1, 1, 1), 2.0)), "float64 (1,) 4000000000000000");
            Check(fails, "cumprod i32 [[3]] flat", () => np.cumprod(np.array(new int[,] { { 3 } })), "int64 (1,) 3");
            Check(fails, "nancumprod f64 [[2]] flat", () => np.nancumprod(np.array(new double[,] { { 2.0 } })), "float64 (1,) 4000000000000000");
            AssertNoMismatch(fails, "NumPy's flat cumprod returns a 1-D array of the input's size");
        }

        /// <summary>
        ///     The float16 <c>np.nansum</c> and <c>np.nanprod</c> along an axis must read the view's elements:
        ///     <c>np.nanprod([2, 3][::-1], axis=0)</c> is <c>6</c>. On a reversed, stepped or transposed view
        ///     NumSharp reads OTHER memory and returns garbage that changes from run to run (the probes saw
        ///     <c>-243.2</c> and <c>1.9e-4</c> for that call). C-contiguous inputs, the flat (no axis) forms, and float16
        ///     <c>np.nanmean</c> / <c>np.nanstd</c> / <c>np.nanvar</c> (their own axis paths, probed on the same
        ///     views) are already correct.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Root cause: float16 has no IL nan-axis kernel, so <c>Default.Reduction.Nan.cs</c>
        ///     <c>ExecuteNanAxisReduction</c> falls back to <c>ExecuteNanAxisReductionScalar</c>. It builds each
        ///     element's MEMORY offset from the view's strides (<c>inputBaseOffset + i * axisStride</c>) and passes
        ///     it to <c>arr.GetAtIndex</c>, which takes a LOGICAL C-order index and maps it through the strides a
        ///     second time. On a C-contiguous array the two agree, so the bug only shows on other layouts; a
        ///     negative stride gives a negative index. float32 / float64 have IL kernels and do not reach this path.
        ///     </para>
        ///     <para>
        ///     The oracle corpus never caught it. The nanreduce tier excuses EVERY value divergence of these ops as a
        ///     known bug (<c>MisalignedRegistry</c>'s <c>NanReduceOps</c> blanket, "nan-reduction value: NaN masking /
        ///     count / summation-order divergence"), garbage float16 values included, and the reduce_start tier's
        ///     all-<c>-0.0</c> operands hide a wrong element when every element is the same.
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void NanReduction_Float16_NonCLayoutAxis_ReadsTheRightElements()
        {
            var fails = new List<string>();
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
            AssertNoMismatch(fails, "NumPy's float16 nan-reductions read the view's elements on every layout");
        }

        // ======================================================================================================
        //  GROUP 7 — np.trapezoid's error for an axis outside y.
        // ======================================================================================================

        /// <summary>
        ///     <c>np.trapezoid</c> with an axis outside <c>y</c>'s dimensions must raise NumPy's
        ///     <c>IndexError: list assignment index out of range</c>: for a 0-d <c>y</c>, an out-of-range positive
        ///     or negative axis, a 1-D <c>x</c>, and an N-D <c>x</c> whose own rank admits the axis. NumSharp raises
        ///     <c>ArgumentOutOfRangeException("axis N is out of bounds for array of dimension M")</c>. An axis
        ///     outside an N-D <c>x</c> is already right: NumPy's <c>diff(x, axis)</c> raises the <c>AxisError</c>
        ///     first, and so does NumSharp.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumPy's <c>trapezoid</c> never checks the axis. It writes into Python lists of length <c>y.ndim</c>
        ///     (<c>slice1[axis] = slice(1, None)</c>, and <c>shape[axis] = d.shape[0]</c> for a 1-D <c>x</c>), so
        ///     the list assignment raises. NumSharp reproduces NumPy's leaked Python <c>IndexError</c>s verbatim
        ///     elsewhere (<c>np.tensordot</c>, <c>np.fft</c>, <c>numpy.polynomial</c>: "tuple index out of
        ///     range") and has the <c>IndexError</c> type for it.
        ///     </para>
        ///     <para>
        ///     Root cause: <c>Math/np.trapezoid.cs</c> <c>TrapezoidAxis</c> raises the axis error on purpose (its
        ///     documentation calls it "the clearer axis-out-of-bounds error"). The unit tests
        ///     <c>Trapezoid_0D_Throws</c> and <c>Trapezoid_XErrorsComeBeforeYAxisErrors</c> now assert only that
        ///     these calls throw, so they hold before and after the fix.
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Trapezoid_AxisOutsideY_RaisesNumPysIndexError()
        {
            var fails = new List<string>();
            const string indexError = "threw IndexError: list assignment index out of range";
            Check(fails, "trapezoid 0-d y", () => np.trapezoid(np.array(5.0)), indexError);
            Check(fails, "trapezoid 0-d y, 1-D x", () => np.trapezoid(np.array(5.0), np.array(new[] { 1.0, 2.0 })), indexError);
            Check(fails, "trapezoid y(2,3) axis=5", () => np.trapezoid(np.ones(new Shape(2, 3)), axis: 5), indexError);
            Check(fails, "trapezoid y(2,3) axis=-3", () => np.trapezoid(np.ones(new Shape(2, 3)), axis: -3), indexError);
            Check(fails, "trapezoid y(2,3) 1-D x axis=5",
                  () => np.trapezoid(np.ones(new Shape(2, 3)), np.array(new[] { 1.0, 2.0, 3.0 }), axis: 5), indexError);
            Check(fails, "trapezoid y(2,3) x(2,2,3) axis=2",
                  () => np.trapezoid(np.ones(new Shape(2, 3)), np.ones(new Shape(2, 2, 3)), axis: 2), indexError);
            AssertNoMismatch(fails, "NumPy's trapezoid leaks the IndexError of its slice-list assignment");
        }

        // ======================================================================================================
        //  Helpers
        // ======================================================================================================

        /// <summary>Builds a 1-D float64 array of <paramref name="n"/> negative zeros.</summary>
        /// <param name="n">The element count.</param>
        /// <returns>A fresh C-contiguous float64 array holding <c>-0.0</c> in every slot.</returns>
        private static NDArray F64(int n)
        {
            var v = new double[n];
            Array.Fill(v, -0.0);
            return np.array(v);
        }

        /// <summary>Builds a 1-D float32 array of <paramref name="n"/> negative zeros.</summary>
        /// <param name="n">The element count.</param>
        /// <returns>A fresh C-contiguous float32 array holding <c>-0.0f</c> in every slot.</returns>
        private static NDArray F32(int n)
        {
            var v = new float[n];
            Array.Fill(v, -0.0f);
            return np.array(v);
        }

        /// <summary>Builds a 1-D float16 array of <paramref name="n"/> negative zeros.</summary>
        /// <param name="n">The element count.</param>
        /// <returns>A fresh C-contiguous float16 array holding <see cref="Half.NegativeZero"/> in every slot.</returns>
        private static NDArray F16(int n)
        {
            var v = new Half[n];
            Array.Fill(v, Half.NegativeZero);
            return np.array(v);
        }

        /// <summary>Builds a 1-D complex128 array of <paramref name="n"/> values <c>-0.0 - 0.0j</c>.</summary>
        /// <param name="n">The element count.</param>
        /// <returns>A fresh C-contiguous complex128 array whose real and imaginary parts are both <c>-0.0</c>.</returns>
        private static NDArray C128(int n)
        {
            var v = new Complex[n];
            Array.Fill(v, new Complex(-0.0, -0.0));
            return np.array(v);
        }

        /// <summary>
        ///     Builds a 1-D float64 array from literal values, the spelling of NumPy's <c>np.array([...])</c> that the
        ///     group 5 cases mirror (an integer literal widens to its exact double).
        /// </summary>
        /// <param name="values">The elements, in order.</param>
        /// <returns>A fresh C-contiguous float64 array.</returns>
        private static NDArray Of(params double[] values) => np.array(values);

        /// <summary>Builds a 1-D float32 array from literal values (NumPy's <c>np.array([...], np.float32)</c>).</summary>
        /// <param name="values">The elements, in order.</param>
        /// <returns>A fresh C-contiguous float32 array.</returns>
        private static NDArray OfSingle(params float[] values) => np.array(values);

        /// <summary>
        ///     Builds a 1-D float16 array from literal values (NumPy's <c>np.array([...], np.float16)</c>). Each value
        ///     is rounded to float16; every value the cases use (small integers, NaN, ±inf) is exact there, so the
        ///     array holds exactly what NumPy's does.
        /// </summary>
        /// <param name="values">The elements, in order, as doubles.</param>
        /// <returns>A fresh C-contiguous float16 array.</returns>
        private static NDArray OfHalf(params double[] values) => np.array(Array.ConvertAll(values, v => (Half)v));

        /// <summary>
        ///     Builds a 1-D complex128 array from literal values (NumPy's <c>np.array([...], np.complex128)</c>); an
        ///     integer literal converts to <c>value + 0j</c>.
        /// </summary>
        /// <param name="values">The elements, in order.</param>
        /// <returns>A fresh C-contiguous complex128 array.</returns>
        private static NDArray OfComplex(params Complex[] values) => np.array(values);

        /// <summary>Builds a 1-D int32 array from literal values (NumPy's <c>np.array([...], np.int32)</c>).</summary>
        /// <param name="values">The elements, in order.</param>
        /// <returns>A fresh C-contiguous int32 array.</returns>
        private static NDArray OfInt32(params int[] values) => np.array(values);

        /// <summary>Builds a 1-D bool array from literal values (NumPy's <c>np.array([...], np.bool_)</c>).</summary>
        /// <param name="values">The elements, in order.</param>
        /// <returns>A fresh C-contiguous bool array.</returns>
        private static NDArray OfBool(params bool[] values) => np.array(values);

        /// <summary>
        ///     Runs one case and records a line when its <see cref="Describe"/> string differs from NumPy's. An
        ///     exception is recorded as a mismatch too, so one throwing case cannot hide the state of the others
        ///     in the same test — a partial fix then shows exactly which cases still differ.
        /// </summary>
        /// <param name="mismatches">Collects one line per failing case; the test asserts it is empty.</param>
        /// <param name="name">The case label used in the failure message.</param>
        /// <param name="call">The NumSharp call under test.</param>
        /// <param name="expected">NumPy 2.4.2's result in <see cref="Describe"/>'s format.</param>
        private static void Check(List<string> mismatches, string name, Func<NDArray> call, string expected)
        {
            string actual;
            try
            {
                actual = Describe(call());
            }
            catch (Exception e)
            {
                actual = $"threw {e.GetType().Name}: {e.Message}";
            }

            if (actual != expected)
                mismatches.Add($"{name}: expected [{expected}] but NumSharp returned [{actual}]");
        }

        /// <summary>
        ///     Fails the test when <see cref="Check"/> recorded any mismatch, listing EVERY recorded line. A plain
        ///     <c>BeEmpty()</c> on the list prints only its first item, which hides how much of a path a partial fix
        ///     has already repaired.
        /// </summary>
        /// <param name="mismatches">The lines <see cref="Check"/> collected; empty means every case matched NumPy.</param>
        /// <param name="because">The NumPy rule the cases check, shown in the failure message.</param>
        /// <exception cref="Exception">The assertion framework's failure exception, when <paramref name="mismatches"/>
        /// is not empty.</exception>
        private static void AssertNoMismatch(List<string> mismatches, string because)
        {
            string.Join(Environment.NewLine, mismatches).Should().BeEmpty(because);
        }

        /// <summary>
        ///     Renders a result as <c>dtype (shape) bits…</c>: the NumPy dtype name, the shape in tuple form
        ///     (<c>()</c>, <c>(3,)</c>, <c>(1,3)</c>), then every element in logical C order — float64 as 16 hex
        ///     digits, float32 as 8, float16 as 4, complex128 as its real then imaginary word, any other dtype as
        ///     its value. Raw bits are the point: a value comparison treats <c>-0.0</c> and <c>+0.0</c> as equal,
        ///     which is how these bugs went unnoticed.
        /// </summary>
        /// <param name="r">The result to render.</param>
        /// <returns>The rendering; a float NaN element is written as the token <c>nan</c>, because no case here is
        /// about NaN bits and NumPy's 0/0 NaN sign is not part of the contract.</returns>
        private static string Describe(NDArray r)
        {
            string dims = string.Join(",", r.shape);
            string shape = r.ndim == 1 ? $"({dims},)" : $"({dims})";
            // flatten() copies in logical C order, the order NumPy's reshape(-1) printed the expected bits in; a 0-d
            // result has one element, so a reshape view of it is enough.
            var flat = r.ndim == 0 ? r.reshape(1) : r.flatten();
            var parts = new List<string>();
            for (long i = 0; i < flat.size; i++)
            {
                switch (flat.typecode)
                {
                    case NPTypeCode.Double:
                    {
                        double v = flat.GetAtIndex<double>(i);
                        parts.Add(double.IsNaN(v) ? "nan" : BitConverter.DoubleToUInt64Bits(v).ToString("x16"));
                        break;
                    }
                    case NPTypeCode.Single:
                    {
                        float v = flat.GetAtIndex<float>(i);
                        parts.Add(float.IsNaN(v) ? "nan" : BitConverter.SingleToUInt32Bits(v).ToString("x8"));
                        break;
                    }
                    case NPTypeCode.Half:
                    {
                        Half v = flat.GetAtIndex<Half>(i);
                        parts.Add(Half.IsNaN(v) ? "nan" : BitConverter.HalfToUInt16Bits(v).ToString("x4"));
                        break;
                    }
                    case NPTypeCode.Complex:
                    {
                        Complex c = flat.GetAtIndex<Complex>(i);
                        parts.Add(BitConverter.DoubleToUInt64Bits(c.Real).ToString("x16"));
                        parts.Add(BitConverter.DoubleToUInt64Bits(c.Imaginary).ToString("x16"));
                        break;
                    }
                    default:
                        parts.Add(flat.GetAtIndex(i).ToString());
                        break;
                }
            }

            return $"{r.dtype.name} {shape} {string.Join(" ", parts)}";
        }
    }
}
