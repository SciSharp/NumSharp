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
    ///     Every expected string below is NumPy 2.4.2's output for the same call (same operands, same arguments),
    ///     probed on 2026-10-03, in <see cref="Describe"/>'s format: <c>dtype (shape) bits…</c>, bits as hex in
    ///     logical C order (a complex element is its real word, then its imaginary word).
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
            AssertNoMismatch(fails, "NumPy's mean is the +0.0-started sum divided by the count");
        }

        /// <summary>
        ///     The fused <c>np.evaluate(NDExpr.Sum(x, axis))</c> over a length-1 axis must give <c>+0.0</c> for a
        ///     <c>-0.0</c> slab, as <c>np.sum</c> does in NumPy. NumSharp copies the slab. The flat
        ///     <c>NDExpr.Sum(x)</c> of one element is already correct.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>DefaultEngine.Evaluate.Stream.cs</c> produces an extent-1 axis straight into the result
        ///     ON PURPOSE, to mirror the engine's copy in <c>HandleTrivialAxisReduction</c> (its comment says so).
        ///     Fix it together with <see cref="Sum_LengthOneAxis_IsPositiveZero"/>, or the two drift apart.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Evaluate_SumOverLengthOneAxis_IsPositiveZero()
        {
            var fails = new List<string>();
            Check(fails, "evaluate Sum f64 (1,3) axis0", () => np.evaluate(NDExpr.Sum((NDExpr)F64(3).reshape(1, 3), 0)),
                  "float64 (3,) 0000000000000000 0000000000000000 0000000000000000");
            AssertNoMismatch(fails, "np.evaluate's Sum must match np.sum, which is +0.0 + x in NumPy");
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
            AssertNoMismatch(fails, "NumPy's nansum / nanprod reduce to a 0-d result, starting from the identity");
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
