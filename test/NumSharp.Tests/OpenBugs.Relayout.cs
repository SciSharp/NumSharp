using System;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — operands of an unusual KIND: offset views, empty operands
    // =====================================================================
    //
    // Found 2026-10-07 by the relayout tiers (test/oracle/gen_relayout_oracle.py, replayed by
    // NumSharp.Tests.Oracle/Fuzz/FuzzCorpusTests.Relayout.cs): existing oracle cases re-emitted with one operand moved
    // into an ndarray kind its (op, operand position, dtype class) cell had never seen, NumPy 2.4.2's answer recomputed.
    // Every divergence is carved into NumSharp.Tests.Oracle/Fuzz/corpus/gaps/relayout_gaps.jsonl under a gap id (the
    // GAPS table of the generator) and replayed by Relayout_KnownGaps. This file reproduces, through PUBLIC calls, the
    // members the array-object oracle's repros (OpenBugs.ViewFlags.cs) do not already cover:
    //
    //   * Offset views (VALUES, RL01-RL04). A C-contiguous view whose Shape carries a non-zero offset (an np.split,
    //     np.unstack or np.array_split child) is read from its OWNER'S START by np.digitize (its x is np.searchsorted's
    //     v), np.isin of a complex128 / float16 element (its sort + searchsorted route), np.nancumsum / np.nancumprod
    //     of float16 / complex128 (they run np.where(np.isnan(a), fill, a)), np.ldexp's mantissa, np.unique_values
    //     and np.intersect1d's first operand. The results are silently wrong.
    //   * Empty operands (RL06, RL07). The NDArray comparison operators return np.empty(lhs.Shape, bool) for an empty
    //     left operand, and np.logaddexp / np.logaddexp2 return a clone of an empty operand: neither broadcasts the
    //     operands (no error where NumPy raises, the wrong shape where NumPy broadcasts) nor, for logaddexp, promotes
    //     the dtype. np.equal & co. (the functions) are right.
    //
    // Every expected value below is NumPy 2.4.2's answer for the same call. EXPECTED (when fixed): each test passes;
    // remove its [OpenBugs], move it next to its API's tests, and drop the gap's rule from the generator's _gap_of so
    // the cases return to the gate.
    // =====================================================================
    public partial class OpenBugs
    {
        // ----- Offset views: silent wrong values -----

        /// <summary>
        ///     <c>np.digitize(x, bins)</c> with <c>x</c> an <c>np.split</c> child (offset 3, <c>[0.5, 2.5, 4, 9]</c>) and
        ///     <c>bins = [1, 2, 3, 4]</c>: NumPy returns <c>[0, 2, 4, 4]</c>; NumSharp digitizes the owner's first four
        ///     elements <c>[7, 7, 7, 0.5]</c> (gap RL01: digitize passes x to np.searchsorted as its v).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "relayout-RL01")]
        public void Relayout_Digitize_XAtAnOffset_ShouldReadTheView()
        {
            using var xbase = np.array(new double[] { 7, 7, 7, 0.5, 2.5, 4, 9 });
            var pieces = np.split(xbase, new[] { 3 });
            using var bins = np.array(new double[] { 1, 2, 3, 4 });
            using var r = np.digitize(pieces[1], bins);

            r.ToArray<long>().Should().Equal(new long[] { 0, 2, 4, 4 });
        }

        /// <summary>
        ///     <c>np.isin(element, test)</c> with a complex128 <c>element</c> that is an <c>np.split</c> child (offset 3,
        ///     <c>[1+1j, 2, 3j]</c>) and <c>test = [2, 3j, 5]</c>: NumPy returns <c>[False, True, True]</c>; NumSharp
        ///     returns all False (gap RL01: the complex route searches the owner's start).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "relayout-RL01")]
        public void Relayout_Isin_ComplexElementAtAnOffset_ShouldReadTheView()
        {
            using var ebase = np.array(new Complex[] { 7, 7, 7, new(1, 1), 2, new(0, 3) });
            var pieces = np.split(ebase, new[] { 3 });
            using var test = np.array(new Complex[] { 2, new(0, 3), 5 });
            using var r = np.isin(pieces[1], test);

            r.ToArray<bool>().Should().Equal(new[] { false, true, true });
        }

        /// <summary>
        ///     <c>np.nancumsum</c> / <c>np.nancumprod</c> of a float16 <c>np.split</c> child (offset 3,
        ///     <c>[1.5, nan, 2.25]</c>): NumPy returns <c>[1.5, 1.5, 3.75]</c> and <c>[1.5, 1.5, 3.375]</c>; NumSharp scans
        ///     the owner's sentinels <c>[7, 7, 7]</c> (gap RL02: the float16 / complex128 route runs np.where on the view).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "relayout-RL02")]
        public void Relayout_NanCumsumNanCumprod_Float16AtAnOffset_ShouldReadTheView()
        {
            using var hbase = np.array(new[] { (Half)7, (Half)7, (Half)7, (Half)1.5, Half.NaN, (Half)2.25 });
            var pieces = np.split(hbase, new[] { 3 });
            using var sum = np.nancumsum(pieces[1]);
            using var prod = np.nancumprod(pieces[1]);

            sum.ToArray<Half>().Should().Equal(new[] { (Half)1.5, (Half)1.5, (Half)3.75 });
            prod.ToArray<Half>().Should().Equal(new[] { (Half)1.5, (Half)1.5, (Half)3.375 });
        }

        /// <summary>
        ///     <c>np.nancumsum</c> of a complex128 <c>np.split</c> child (offset 3, <c>[1+2j, nan+1j, 2]</c>): NumPy returns
        ///     <c>[1+2j, 1+2j, 3+2j]</c> (a NaN in either part counts as NaN); NumSharp scans the owner's sentinels
        ///     (gap RL02).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "relayout-RL02")]
        public void Relayout_NanCumsum_ComplexAtAnOffset_ShouldReadTheView()
        {
            using var cbase = np.array(new Complex[] { 7, 7, 7, new(1, 2), new(double.NaN, 1), 2 });
            var pieces = np.split(cbase, new[] { 3 });
            using var r = np.nancumsum(pieces[1]);

            r.ToArray<Complex>().Should().Equal(new Complex[] { new(1, 2), new(1, 2), new(3, 2) });
        }

        /// <summary>
        ///     <c>np.ldexp(x1, x2)</c> with the MANTISSA an <c>np.split</c> child (offset 3, <c>[0.5, -1.5, 3]</c>) and
        ///     <c>x2 = [1, 2, 3]</c>: NumPy returns <c>[1, -6, 24]</c>; NumSharp scales the owner's sentinels to
        ///     <c>[14, 28, 56]</c> (gap RL03; the exponent side is ViewFlags_Ldexp_ExponentAtAnOffset_ShouldReadTheView).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "relayout-RL03")]
        public void Relayout_Ldexp_MantissaAtAnOffset_ShouldReadTheView()
        {
            using var mbase = np.array(new double[] { 7, 7, 7, 0.5, -1.5, 3 });
            var pieces = np.split(mbase, new[] { 3 });
            using var exponents = np.array(new long[] { 1, 2, 3 });
            using var r = np.ldexp(pieces[1], exponents);

            r.ToArray<double>().Should().Equal(new double[] { 1, -6, 24 });
        }

        /// <summary>
        ///     <c>np.unique_values</c> of a float64 <c>np.split</c> child (offset 3, <c>[3, 1, 3, 2]</c>): NumPy returns
        ///     <c>[1, 2, 3]</c>; NumSharp returns <c>[3, 7]</c>. And <c>np.intersect1d(ar1, [1, 2, 7])</c> with <c>ar1</c>
        ///     the int32 child: NumPy returns <c>[1, 2]</c>; NumSharp returns <c>[7]</c> (gap RL04: the set routines read
        ///     either operand from the owner's start, not only ar2).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "relayout-RL04")]
        public void Relayout_UniqueValues_AndIntersect1dFirstOperand_AtAnOffset_ShouldReadTheView()
        {
            using var fbase = np.array(new double[] { 7, 7, 7, 3, 1, 3, 2 });
            var fpieces = np.split(fbase, new[] { 3 });
            using var u = np.unique_values(fpieces[1]);
            u.ToArray<double>().Should().Equal(new double[] { 1, 2, 3 });

            using var ibase = np.array(new[] { 7, 7, 7, 3, 1, 3, 2 });
            var ipieces = np.split(ibase, new[] { 3 });
            using var other = np.array(new[] { 1, 2, 7 });
            using var common = np.intersect1d(ipieces[1], other);
            common.ToArray<int>().Should().Equal(new[] { 1, 2 });
        }

        // ----- Empty operands: no broadcast check -----

        /// <summary>
        ///     The NDArray comparison operators with an EMPTY left operand: NumPy raises <c>ValueError: operands could not
        ///     be broadcast together with shapes (0,5) (4,5)</c> for <c>zeros((0,5)) == zeros((4,5))</c> and broadcasts
        ///     <c>zeros((0,1)) &lt; zeros((0,5))</c> to shape <c>(0, 5)</c>; NumSharp returns an empty result shaped like
        ///     the left operand for both (gap RL06; <c>np.equal</c> itself raises the right error).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "relayout-RL06")]
        public void Relayout_ComparisonOperators_EmptyLeftOperand_ShouldBroadcast()
        {
            using var a = np.zeros(new Shape(0, 5));
            using var b = np.zeros(new Shape(4, 5));
            Action eq = () => { using var r = a == b; };
            eq.Should().Throw<IncorrectShapeException>()
              .WithMessage("operands could not be broadcast together with shapes (0,5) (4,5)*");

            using var c = np.zeros(new Shape(0, 1));
            using var lt = c < a;
            lt.shape.Should().Equal(new long[] { 0, 5 });
        }

        /// <summary>
        ///     <c>np.logaddexp</c> / <c>np.logaddexp2</c> with an EMPTY operand: NumPy's float loop gives an int32 input a
        ///     float64 result (<c>logaddexp(zeros((0,5), int32), array(0))</c> is float64 <c>(0, 5)</c>), raises
        ///     <c>ValueError: operands could not be broadcast together with shapes (0,5) (4,5)</c> for float32
        ///     <c>(0,5)</c> with <c>(4,5)</c>, and broadcasts <c>(0,1)</c> with <c>(0,5)</c> to <c>(0, 5)</c>; NumSharp
        ///     returns a clone of the empty operand in all three (gap RL07).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "relayout-RL07")]
        public void Relayout_Logaddexp_EmptyOperand_ShouldBroadcastAndPromote()
        {
            using var empty32 = np.zeros(new Shape(0, 5), np.int32);
            using var zero = np.array(0);
            using var promoted = np.logaddexp(empty32, zero);
            promoted.dtype.Should().Be(np.float64);
            using var promoted2 = np.logaddexp2(empty32, zero);
            promoted2.dtype.Should().Be(np.float64);

            using var e = np.zeros(new Shape(0, 5), np.float32);
            using var f = np.zeros(new Shape(4, 5), np.float32);
            Action mismatch = () => { using var r = np.logaddexp(e, f); };
            mismatch.Should().Throw<IncorrectShapeException>()
                    .WithMessage("operands could not be broadcast together with shapes (0,5) (4,5)*");

            using var col = np.zeros(new Shape(0, 1));
            using var wide = np.zeros(new Shape(0, 5));
            using var r2 = np.logaddexp(col, wide);
            r2.shape.Should().Equal(new long[] { 0, 5 });
        }
    }
}
