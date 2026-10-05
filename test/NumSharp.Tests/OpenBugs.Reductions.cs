using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using static NumSharp.Tests.Utilities.RawBitCases;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs in how NumSharp STARTS a reduction, a scan or a size-1 product — found while making
    ///     <c>np.trapezoid</c> bit-exact (commit 94c9eb6b, 2026-10-03) and by the 2026-10-04 reduction-start probe —
    ///     plus the probe's wrong shapes and errors (groups 6, 7), which NaN a flat sum of several NaNs returns
    ///     (group 8), and an axis sum in a cast loop whose rows are longer than the buffer (group 9). Each test asserts NumPy
    ///     2.4.2's answer — dtype, shape and the raw bits of every element — so it FAILS while the bug lives and passes
    ///     once it is fixed. Remove <c>[OpenBugs]</c> from a test when its path is fixed, and move the test next to the
    ///     function's own tests.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The sign of a zero is the only visible difference in most of these cases, so a value comparison
    ///     (<c>0.0 == -0.0</c>) passes them; that is why nothing caught them. NumPy's rules, each ported wrong below:
    ///     </para>
    ///     <list type="bullet">
    ///         <item><b><c>multiply.reduce</c> starts from the identity <c>1</c>, which is not a no-op for a
    ///         complex element</b> (group 1). NumPy's complex product of one element is <c>(1+0j) * x</c>, so
    ///         <c>np.prod([-0-0j])</c> is <c>(+0, -0)</c>. NumSharp copies the element.</item>
    ///         <item><b><c>np.median</c> is <c>np.mean</c> of the middle element(s)</b> (group 1). It starts from
    ///         <c>+0.0</c>, so a <c>-0.0</c> middle gives <c>+0.0</c>. NumSharp returns the middle element itself.</item>
    ///         <item><b>The default <c>np.einsum</c> adds every term into a zero-filled output</b> (group 1), so even
    ///         a bare elementwise product comes out <c>+0.0 + a*b</c>.</item>
    ///         <item><b><c>add.accumulate</c> copies the first element</b> (group 2). <c>np.cumsum([-0.0, -0.0])</c>
    ///         is <c>[-0.0, -0.0]</c>. NumSharp's flat scan starts from <c>default(T)</c>, which is <c>+0.0</c>.</item>
    ///         <item><b><c>np.dot</c> of a size-1 operand is NumPy's scalar branch</b> (group 3).
    ///         <c>cblas_matrixproduct</c> computes a one-element result as the plain product
    ///         (<c>np.dot([-0.0], [1.0])</c> is <c>-0.0</c>) and a longer one as <c>?axpy</c> into a zero-filled
    ///         output. NumSharp's managed dot does the opposite in both.</item>
    ///     </list>
    ///     <para>
    ///     The add.reduce rule — <c>add.reduce</c> starts from the <c>+0.0</c> identity, so a sum of only
    ///     <c>-0.0</c> values is <c>+0.0</c>, one element and a length-1 axis included — is fixed. Its tests (np.sum /
    ///     np.mean / np.average / np.nansum / np.nanprod / np.nanmean / np.einsum's sums / np.evaluate's sum kinds /
    ///     np.trapezoid, and the probe's nansum length-1-axis, complex 1-D nansum, nanmean 1-D axis, empty-mean dtype
    ///     and float16 nan-axis read pins) moved to <c>Math/np.sum.Identity.Test.cs</c> without <c>[OpenBugs]</c>.
    ///     </para>
    ///     <para>
    ///     Group 5 (2026-10-04) was <c>np.var</c> / <c>np.std</c> at a degenerate size: one element per output, a
    ///     <c>ddof</c> at or past the element count, or an empty float16/float32 input, where NumSharp's shortcuts
    ///     answered without NumPy's arithmetic. It is fixed — np.var / np.std are NumPy's <c>_var</c> / <c>_std</c>
    ///     statement for statement since 2026-10-05 (plan <c>docs/plans/api-parameter-parity.md</c> A2) — and its tests
    ///     moved to <c>Math/np.var.NumPyPort.Test.cs</c> without <c>[OpenBugs]</c>; the oracle's registry no longer
    ///     excuses any var/std case (the new <c>varstd.jsonl</c> tier holds them to zero excuses).
    ///     </para>
    ///     <para>
    ///     Group 4 (2026-10-04) was <c>np.nanstd</c> / <c>np.nanvar</c> returning <c>(1,)</c> for a 1-D axis reduction
    ///     and dropping <c>keepdims</c> for one element. It is fixed — they are NumPy's <c>nanvar</c> / <c>nanstd</c>
    ///     statement for statement since 2026-10-05 (plan <c>docs/plans/api-parameter-parity.md</c> A3) — and its test
    ///     moved to <c>Math/np.nanvar.NumPyPort.Test.cs</c>; the <c>nanvarstd.jsonl</c> tier holds them to zero excuses.
    ///     </para>
    ///     <para>
    ///     The 2026-10-04 reduction-start probe ran every reduction, scan and product over all-<c>-0.0</c> operands
    ///     (size 1, length-1 axes, longer runs; C, transposed, reversed and stepped layouts; float64, float32,
    ///     float16 and complex128) against NumPy 2.4.2 bit for bit. The cells NumSharp gets right are the
    ///     <c>reduce_start.jsonl</c> oracle tier (<c>FuzzCorpusTests.ReductionStart</c>; 7,942 at delivery, 8,586
    ///     once the identity add landed, 8,610 once np.var / np.std became NumPy's <c>_var</c>, 8,666 once np.nanvar /
    ///     np.nanstd became NumPy's <c>nanvar</c>). The cells it gets wrong are that tier's carve (<c>gen_oracle._rs_carved</c>,
    ///     one rule per root cause, exactly the 782 differing cells), and every rule names the test below that pins it.
    ///     </para>
    ///     <para>
    ///     The probe also found reductions that return the wrong shape (group 6; group 4 is fixed) and <c>np.trapezoid</c>'s
    ///     error for an axis outside <c>y</c> (group 7). Those tests use ordinary values, where a value comparison
    ///     alone already fails.
    ///     </para>
    ///     <para>
    ///     Every expected string below is NumPy 2.4.2's output for the same call (same operands, same arguments),
    ///     probed on 2026-10-03 (groups 1-3) and 2026-10-04 (groups 4-8 and the probe's additions to groups 1-3), in
    ///     <see cref="NumSharp.Tests.Utilities.RawBitCases.Describe"/>'s format: <c>dtype (shape) bits…</c>, bits as hex
    ///     in logical C order (a complex element is its real word, then its imaginary word). An error is
    ///     <c>threw &lt;type&gt;: &lt;message&gt;</c>, NumPy's Python exception type and message.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class OpenBugsReductionTests : TestClass
    {
        // ======================================================================================================
        //  GROUP 1 — identity starts NumSharp still misses: complex multiply.reduce of one element, median as
        //  np.mean of the middle element(s), einsum's zero-filled output for bare products.
        // ======================================================================================================

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
        ///     NumPy's default <c>np.einsum</c> (<c>optimize=False</c>) adds every term into a ZERO-FILLED output,
        ///     so each output element is <c>+0.0 + terms</c>: even a bare elementwise product (<c>'i,i->i'</c>,
        ///     <c>'i,j->ij'</c>) of a <c>-0.0</c> comes out <c>+0.0</c>. NumSharp's contraction returns the bare
        ///     product: <c>-0.0</c>, and for complex128 <c>(+0, -0)</c> where NumPy returns <c>(+0, +0)</c>. An ordinary
        ///     operand shows it too: <c>np.einsum('i,i->i', [-0.0, 1.0], [1.0, 1.0])</c> is <c>[0.0, 1.0]</c>.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Root cause: <c>np.EinsumContract</c> (<c>LinearAlgebra/np.einsum.Contract.cs</c>) ports NumPy's
        ///     <c>optimize=</c> structure and returns a contraction with nothing summed as a broadcast
        ///     <c>np.multiply</c>. NumPy's DEFAULT einsum is <c>PyArray_EinsteinSum</c>: the output operand is zeroed
        ///     and every <c>sum_of_products</c> loop computes <c>out += a * b</c>. NumPy's own <c>optimize=True</c>
        ///     path gives <c>-0.0</c> for the bare products here (probed: <c>[-0., 1.]</c> for the ordinary case
        ///     above), so the fix is to start each output from <c>+0.0</c> on the default path, not to change which
        ///     kernels compose.
        ///     </para>
        ///     <para>
        ///     The probe's seven einsum cases that sum (<c>'ij->'</c>, <c>'ij->i'</c>, <c>'ii->'</c>, <c>'i,i->'</c>,
        ///     <c>'ij,j->i'</c>, <c>'ij,jk->ik'</c> over a length-1 index) pass since the identity-seeded add reduction
        ///     and moved to <c>np_sum_identity_tests.Einsum_SumOfOneTerm_IsPositiveZero</c>. The einsum oracle tier
        ///     feeds nonzero operands and so never meets this; the reduce_start tier gates every other einsum cell.
        ///     </para>
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Einsum_ZeroFilledAccumulator_GivesPositiveZero()
        {
            var fails = new List<string>();
            const string z64 = "0000000000000000";
            Check(fails, "einsum i,i->i (3).(3)", () => np.einsum("i,i->i", F64(3), np.ones(3)),
                  $"float64 (3,) {z64} {z64} {z64}");
            Check(fails, "einsum i,j->ij (2).(3)", () => np.einsum("i,j->ij", F64(2), np.ones(3)),
                  $"float64 (2,3) {z64} {z64} {z64} {z64} {z64} {z64}");
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
        //  GROUP 6 — reductions that return the wrong shape or values (found by the 2026-10-04 reduction-start
        //  probe; its one-element float32/float16 std/var dtype finding was group 5, now fixed and moved to
        //  Math/np.var.NumPyPort.Test.cs as VarStd_OneElement_KeepsFloat16Float32_AndHonoursTheDtypeArgument).
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
        //  GROUP 8 — NumPy's compiled pairwise_sum picks a NaN by operand order; the emitted folds pick another.
        // ======================================================================================================

        /// <summary>
        ///     A flat <c>np.sum</c> of SEVERAL NaNs must return the NaN NumPy's compiled <c>pairwise_sum</c> keeps: the
        ///     win-amd64 wheel (MSVC) keeps the ACCUMULATOR's NaN in the leaf loop (<c>r[k] += a[i+k]</c>), the SECOND
        ///     operand's in the tree combine <c>((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))</c>, the accumulator's in the tail
        ///     and the right half's in <c>left + right</c>. So eight NaNs with payloads 1..8 sum to payload 8 (lane 7
        ///     wins the combine), and 130 to payload 72 (the right leaf's lane-7 seed, element 71). NumSharp returns
        ///     payload 1 for both: the emitted combine keeps the first operand's NaN.
        /// </summary>
        /// <remarks>
        ///     The flat sum's ORDER is NumPy's since the exact add.reduce engine (<c>NumPyAddReduce</c>), so only the
        ///     NaN choice inside each add differs. The value is NaN either way, and the fuzz oracle tokenizes float NaN,
        ///     so nothing else sees it. Fix: give the folds' tree combine and the recursion's <c>left + right</c> the
        ///     explicit NaN priority the add / multiply kernels already carry
        ///     (<c>DirectILKernelGenerator.NaNOperandOrder.cs</c>), and check that RyuJIT keeps the leaf loop's
        ///     <c>Add(acc, x)</c> operand order. Linux NumPy (GCC) orders the operands differently — this pin is
        ///     NumPy's win-amd64 answer, like the add / multiply NaN-order pins.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Sum_SeveralNaNs_KeepsTheNaNNumPysCompiledPairwiseSumKeeps()
        {
            static NDArray NaNs64(int n)
            {
                var v = new double[n];
                for (int i = 0; i < n; i++) v[i] = System.BitConverter.Int64BitsToDouble(0x7ff8000000000000L | (i + 1));
                return np.array(v);
            }

            static NDArray NaNs32(int n)
            {
                var v = new float[n];
                for (int i = 0; i < n; i++) v[i] = System.BitConverter.Int32BitsToSingle(0x7fc00000 | (i + 1));
                return np.array(v);
            }

            var fails = new List<string>();
            Check(fails, "sum f64 NaN payloads 1..8", () => np.sum(NaNs64(8)), "float64 () 7ff8000000000008", rawNaN: true);
            Check(fails, "sum f64 NaN payloads 1..130", () => np.sum(NaNs64(130)), "float64 () 7ff8000000000048", rawNaN: true);
            Check(fails, "sum f32 NaN payloads 1..8", () => np.sum(NaNs32(8)), "float32 () 7fc00008", rawNaN: true);
            Check(fails, "sum f32 NaN payloads 1..130", () => np.sum(NaNs32(130)), "float32 () 7fc00048", rawNaN: true);
            AssertNoMismatch(fails, "NumPy's compiled pairwise_sum keeps the accumulator's NaN in the loop and the second operand's in the combine");
        }

        // ======================================================================================================
        //  GROUP 9 — an axis sum in a CAST loop whose rows are longer than np.getbufsize(). NumPy buffers the cast
        //  operand in fills of the buffer size, and a fill splits a pinned row (the reduced axis innermost): the
        //  row's sum is out += pairwise(part) per fill. NumSharp's cast-loop axis route sums each row in one call.
        // ======================================================================================================

        /// <summary>
        ///     <c>np.sum(x, axis=1, dtype=np.float32)</c> of a float64 <c>(8, 25)</c> array under <c>np.setbufsize(16)</c>
        ///     splits each 25-element row across two buffer fills, as NumPy does (<c>out += pairwise(16 elements)</c>, then
        ///     <c>out += pairwise(9)</c>). NumSharp sums each row in one pairwise call, which is NumPy's answer only while a
        ///     row fits one buffer (the default 8192), so three of the eight sums are 1 ULP off. <c>np.var</c> /
        ///     <c>np.std</c> with the same <c>dtype=</c> inherit it: both sums of NumPy's <c>_var</c> are such cast loops.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Default.Reduction.Add.cs</c> <c>ReduceAddInCastLoop</c>'s axis route casts blocks along the
        ///     outermost kept axis and sums each reduced row in one pairwise call; NumPy's fills come from
        ///     <c>npyiter_find_buffering_setup</c> (<c>.claude/CLAUDE.md</c> names it among np.sum's residuals: "an axis
        ///     reduction in a cast loop whose reduced runs exceed the buffer is not modeled"). Found by the varstd oracle
        ///     tier's section B (2026-10-05, plan <c>docs/plans/api-parameter-parity.md</c> A2), which carves these cells
        ///     (<c>gen_oracle.gen_varstd</c>, "pinned row longer than the buffer") and names this test. NumPy 2.4.2's
        ///     default-buffer answers, which NumSharp gives today, differ from these at elements 1, 6 and 7 (sum) and
        ///     element 5 (var).
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Sum_CastLoopAxis_RowLongerThanTheBuffer_SplitsAcrossFills()
        {
            var fails = new List<string>();
            var x = ((np.arange(200) % 37) * 1.0001 + 1000.0).reshape(8, 25);
            long previous = np.setbufsize(16);
            try
            {
                Check(fails, "sum f64 (8,25) axis1 dtype=f32 bufsize16", () => np.sum(x, 1, dtype: np.float32),
                      "float32 (8,) 46c5a80f 46c6c816 46c7e81e 46c5da11 46c6b016 46c7d01e 46c60c12 46c69816");
                Check(fails, "var f64 (8,25) axis1 dtype=f32 bufsize16", () => np.var(x, 1, dtype: np.float32),
                      "float32 (8,) 42500aa6 4322dadf 428a2078 42500aa6 43216f13 42a96403 42500aa6 431e977a");
            }
            finally
            {
                np.setbufsize(previous);
            }
            AssertNoMismatch(fails, "NumPy's buffered fills split a pinned row longer than the buffer");
        }

        // ======================================================================================================
        //  GROUP 10 — an axis sum over a view with a REVERSED axis (several outputs). NumPy's reduction iterator keeps
        //  the logical order (DONT_NEGATE_STRIDES); NumSharp's axis kernel walks memory order, so cancelling values
        //  give other bits. A one-output axis already takes the exact flat engine, which walks the logical order.
        // ======================================================================================================

        /// <summary>
        ///     <c>np.sum(b[::-1], axis=0)</c> of a float32 <c>(4, 3)</c> array whose first two columns cancel
        ///     (<c>±1e8</c> with <c>1.0</c>s) is <c>[0, 1, 8]</c> in NumPy: it adds the rows in the view's logical order
        ///     (row 3 first). NumSharp adds them in memory order (row 0 first) and gives <c>[1, 0, 8]</c>. <c>np.mean</c>
        ///     inherits it through its sum (<c>[0, 0.25, 2]</c> vs <c>[0.25, 0, 2]</c>).
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Default.Reduction.Add.cs</c> routes an axis sum with several outputs to the axis kernels,
        ///     which walk a negative-stride reduced axis from its lowest address; NumPy's reduce iterator does not negate
        ///     strides. Plan <c>docs/plans/api-parameter-parity.md</c> finding 5 (found by A2's building-block check,
        ///     2026-10-05; also along the non-reversed axis in some layouts). The mean_loops oracle tier carves these cells
        ///     (<c>gen_oracle.gen_mean_loops</c>, "finding 5") and names this test, and the registry keeps a narrow branch
        ///     for np.mean's other tiers (<c>MisalignedRegistry.IsReversedAxisReduction</c>).
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Sum_ReversedViewAxis_WalksTheLogicalOrder()
        {
            var fails = new List<string>();
            var b = np.array(new float[,] { { 1e8f, 1f, 2f }, { 1f, 1e8f, 2f }, { -1e8f, 3f, 2f }, { 1f, -1e8f, 2f } });
            var v = b["::-1"];
            Check(fails, "sum f32 b[::-1] axis0", () => np.sum(v, 0), "float32 (3,) 00000000 3f800000 41000000");
            Check(fails, "mean f32 b[::-1] axis0", () => np.mean(v, 0), "float32 (3,) 00000000 3e800000 40000000");
            AssertNoMismatch(fails, "NumPy adds the rows of a reversed view in its logical order");
        }

        // ======================================================================================================
        //  GROUP 11 — the FLAT sum of a broadcast operand (a stride-0 axis). NumPy's iterator cannot coalesce a
        //  stride-0 axis with its neighbour, so it calls the add loop once per broadcast row: out += pairwise(row).
        //  NumSharp's exact flat engine declines a broadcast operand, and the elementwise fold it falls back to sums
        //  all the elements in its own order.
        // ======================================================================================================

        /// <summary>
        ///     <c>np.sum(np.broadcast_to(r, (4, 5)))</c> of five float32 values is <c>0x40d3f48a</c> in NumPy (four calls
        ///     of the add loop, one per broadcast row, each adding the row's pairwise sum to the result); NumSharp's
        ///     fold gives <c>0x40d3f488</c>. A float64 scalar broadcast to <c>(3, 4)</c> is one ULP off the same way.
        ///     <c>np.mean</c> inherits it through its sum.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>NumPyAddReduce</c> (<c>Default.Reduction.Add.Exact.cs</c>) declines a broadcast operand —
        ///     NpyIter's axis sort skips the comparisons a stride-0 operand takes part in, a schedule the engine does not
        ///     model — and <c>sum_elementwise_il</c> / <c>mean_elementwise_il</c> fold instead (<c>.claude/CLAUDE.md</c>,
        ///     "Math — Reductions", residuals: "a broadcast operand declines to the old fold (value-exact, not
        ///     order-exact)"). Found by the mean_loops oracle tier (2026-10-05, plan
        ///     <c>docs/plans/api-parameter-parity.md</c> X9), which carves these cells (<c>gen_oracle.gen_mean_loops</c>,
        ///     "broadcast") and names this test.
        /// </remarks>
        [TestMethod, OpenBugs]
        public void Sum_BroadcastOperand_Flat_IsNumPysCallSchedule()
        {
            var fails = new List<string>();
            var b = np.broadcast_to(np.array(new[] { -0.8627f, 1.9222f, 0.373f, -1.1762f, 1.3996f }), new Shape(4, 5));
            Check(fails, "sum f32 broadcast (4,5)", () => np.sum(b), "float32 () 40d3f48a");
            Check(fails, "mean f32 broadcast (4,5)", () => np.mean(b), "float32 () 3ea9906e");
            var s = np.broadcast_to(np.array(new[] { -0.8627 }), new Shape(3, 4));
            Check(fails, "sum f64 scalar broadcast (3,4)", () => np.sum(s), "float64 () c024b46dc5d63887");
            Check(fails, "mean f64 scalar broadcast (3,4)", () => np.mean(s), "float64 () bfeb9b3d07c84b5f");
            AssertNoMismatch(fails, "NumPy adds a broadcast operand one row (one add-loop call) at a time");
        }
    }
}
