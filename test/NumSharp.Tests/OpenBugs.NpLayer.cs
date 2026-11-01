using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — divergences the np-layer dtype plan found on the way
    // =====================================================================
    //
    // docs/plans/np-layer-dtype-logic.md moved each np.* function's dtype logic below its facade and fixed what it found
    // in the code it moved (D4: one small commit per divergence). Six divergences sat outside the moved code; on
    // 2026-10-31 the owner asked for them to be pinned and deferred ("Create an open bug reproducing the problem/s and
    // defer to tackle them later"). Each group below is one of them, reproduced through public calls with the smallest
    // input that shows it:
    //
    //   NL1  np.copyto into a StringDType array under where= converts the WHOLE source, masked-off elements included;
    //        a ufunc with a StringDType out= and where= (np.logical_and / or / xor / not) reaches it with a temporary
    //        whose masked-off slots are uninitialized, so some calls raise "invalid value encountered in bool to string
    //        cast" at random.
    //   NL2  np.median of complex data holding a NaN returns the mean of the middle values, not NumPy's NaN element.
    //   NL3  the flat np.prod of float32 / float64 multiplies in SIMD lane order, not NumPy's sequential order: values
    //        differ by an ULP, and a product that overflows before it meets a zero is NaN on one side and 0 on the other.
    //   NL4  1-D np.polyval with complex coefficients and an infinite real x: NaN+NaNj where NumPy keeps the
    //        coefficient's imaginary part.
    //   NL5  argument checks in another order than NumPy's: np.nested_iters' converters (and the type of its
    //        no-operand error), np.einsum's out= extent check after its cast check, np.einsum's explicit path not
    //        checked for completeness.
    //   NL6  np.array of a ragged jagged C# array (int[][], double[][], ...) truncates or throws IndexOutOfRangeException
    //        where NumPy raises its inhomogeneous-shape ValueError.
    //   NL7  the plan's B3: user code that np.apply_along_axis / np.apply_over_axes call back runs with the subclass
    //        boundary's depth raised, so a numpy.matrix or chararray the callback passes to an np.* function comes back
    //        plain. P7 would have removed the depth; the owner dropped P7 the same day ("Drop because np.* uses the lhs
    //        to pick which"), so B3 stays open and is pinned here with the six.
    //   NL8  np.sum's add.reduce step `*acc += pairwise_sum(call)` leaves the NaN of a NaN + NaN to the JIT: a Debug
    //        build of NumSharp.Core keeps the call's NaN where NumPy (and a Release build, by the JIT's operand order)
    //        keeps the accumulator's. Found by the journey4 merge's Debug gate, where it fails
    //        ShortRows_SeveralNaNs_KeepTheRunningSumsNaN and Trapezoid_NaNSpacing_IsTheProductsFirstOperand.
    //   NL9  np.polyder differentiates p.size - 1 terms per order where NumPy's `n = len(p) - 1` takes the first axis, and
    //        computes m products where NumPy's recursion computes m + 1 (the last one, at m == 0, discarded): a 2-D p is
    //        differentiated along its rows in NumPy and refused here, a 0-d p raises len()'s TypeError in NumPy at every
    //        order (0 included), and an N-D p whose extra product does not broadcast raises in NumPy, also at m == 0.
    //        Found by P8's move of polyder onto TensorEngine (434cc8d4), which kept the behaviour.
    //   NL10 np.einsum_path with an ellipsis and too few operands fails indexing the operand list with .NET's
    //        IndexOutOfRangeException where NumPy's _parse_einsum_input raises IndexError("list index out of range").
    //        Found by P8's move of einsum_path onto TensorEngine (d0c7a582), which kept the behaviour.
    //   NL11 np.isscalar answers False for a Half (NumPy's float16 scalar) and a BigInteger (a Python int past int64),
    //        where NumPy answers True, and True for a Type object of a primitive (typeof(int)), where NumPy's isscalar(int)
    //        is False. Found by P8's snapshot tool, whose object operand (18835e6b) first ran np.isscalar over them; the
    //        move onto the array object (4cea01d6) kept the behaviour.
    //
    // Every expected value below is NumPy 2.4.2's answer for the same call (probed on win-amd64). EXPECTED (when fixed):
    // each test passes; remove its [OpenBugs], move it next to its API's tests, and drop the oracle excuse named in its
    // summary, if any.
    // =====================================================================
    public partial class OpenBugs
    {
        /// <summary>
        ///     Python's <c>float('nan')</c>: the POSITIVE quiet NaN (0x7FF8000000000000). .NET's <see cref="double.NaN"/> is
        ///     the negative one (0xFFF8000000000000), so a test that hands NumPy's input to NumSharp builds its NaN from these
        ///     bits; otherwise an element-returning function (np.median's NaN check) would answer with the other sign.
        /// </summary>
        private static readonly double NpLayerPositiveNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000);

        /// <summary>
        ///     Reads every cell of a StringDType array in C order, so a test can compare all of them at once.
        /// </summary>
        /// <param name="a">The StringDType array.</param>
        /// <returns>The cells' strings in C order.</returns>
        private static string[] NpLayerCells(NDArray a)
            => Enumerable.Range(0, (int)a.size).Select(i => (string)a.GetValue(i)).ToArray();

        // ------------------------------------------------------------------
        // NL1 — np.copyto into StringDType under where= converts masked-off elements
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.copyto(out, src, where=all_false)</c> with a StringDType <c>out</c> and a bool <c>src</c> holding a byte
        ///     that is no bool value (2, made through a uint8 view): NumPy writes nothing and reads nothing it does not
        ///     write; its masked transfer (<c>PyArray_GetMaskedDTypeTransferFunction</c>) casts only the selected runs.
        ///     NumSharp raises <c>RuntimeError: invalid value encountered in bool to string cast</c>:
        ///     <c>UnmanagedStorage.CopytoAssignVStrings</c> (Backends/Unmanaged/UnmanagedStorage.Assign.VString.cs) converts
        ///     the whole source with <c>src.astype(to, ...)</c> before it applies the mask.
        /// </summary>
        /// <remarks>
        ///     The ufunc route is the visible symptom: <c>DefaultEngine.IntoTextOut</c> computes a ufunc's loop result under
        ///     the mask into a temporary whose masked-off slots are left uninitialized (by design: "the masked copy never
        ///     reads them"), then <c>NDStringArithmetic.CastInto</c> copies it here. So
        ///     <c>np.logical_xor(x, y, out=text, where=all_false)</c> raises at random, depending on what the recycled
        ///     memory holds (1 to 5 of 40 calls per case in a sweep of dtypes and (3, 4) / (2, 3, 4) / 0-d operands, none
        ///     in 2,000 calls of one case alone). This pin uses an invalid byte the caller controls, so it fails on every
        ///     run.
        /// </remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL1")]
        public void NpLayer_Copyto_StringDTypeOut_AllFalseMask_ShouldNotConvertTheSource()
        {
            using var bytes = np.array(new byte[] { 1, 2, 0 });
            using var bad = bytes.view(np.@bool);
            using var dst = np.array(new[] { "x", "y", "z" }, np.dtypes.StringDType());
            using var mask = np.zeros(new Shape(3), np.@bool);

            np.copyto(dst, bad, where: mask);

            NpLayerCells(dst).Should().Equal("x", "y", "z");
        }

        /// <summary>
        ///     <c>np.copyto(out, src, where=[True, False, True])</c> with the invalid bool byte in the MASKED-OFF element:
        ///     NumPy converts the two selected elements only (<c>['True', 'y', 'False']</c>). NumSharp raises
        ///     <c>RuntimeError: invalid value encountered in bool to string cast</c> from the same whole-source conversion as
        ///     <see cref="NpLayer_Copyto_StringDTypeOut_AllFalseMask_ShouldNotConvertTheSource"/>. (With the invalid byte
        ///     SELECTED, NumPy raises that error too; that case is correct today.)
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL1")]
        public void NpLayer_Copyto_StringDTypeOut_Mask_ShouldConvertOnlyTheSelectedElements()
        {
            using var bytes = np.array(new byte[] { 1, 2, 0 });
            using var bad = bytes.view(np.@bool);
            using var dst = np.array(new[] { "x", "y", "z" }, np.dtypes.StringDType());
            using var mask = np.array(new[] { true, false, true });

            np.copyto(dst, bad, where: mask);

            NpLayerCells(dst).Should().Equal("True", "y", "False");
        }

        // ------------------------------------------------------------------
        // NL2 — complex np.median skips NumPy's _median_nancheck
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.median([1+2j, nan+1j, 3, 4+1j])</c>: NumPy 2.4.2 returns <c>nan+1j</c>, the NaN element itself
        ///     (<c>_median_nancheck</c>: after the partition the complex sort order puts a NaN-carrying element last, and
        ///     a NaN there replaces the result). NumSharp returns <c>2+1j</c>, the mean of the two middle values, because
        ///     its complex median path never runs the NaN check. The oracle excuses it today (MisalignedRegistry, "complex
        ///     median: NumPy _median_nancheck NaN-propagation vs NumSharp mean-of-middle"); drop that excuse with the fix.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL2")]
        public void NpLayer_Median_ComplexWithANaN_ShouldReturnTheNaNElement()
        {
            using var a = np.array(new[] { new Complex(1, 2), new Complex(NpLayerPositiveNaN, 1), new Complex(3, 0), new Complex(4, 1) });
            using var m = np.median(a);

            var z = (Complex)m.GetValue();
            BitConverter.DoubleToInt64Bits(z.Real).Should().Be(0x7FF8000000000000);
            z.Imaginary.Should().Be(1.0);
        }

        /// <summary>
        ///     <c>np.median([1+nanj, 2, 3])</c>: the NaN is in the IMAGINARY part. NumPy returns <c>1+nanj</c> (the element);
        ///     NumSharp returns <c>2+0j</c>, the middle value of the sort with the NaN ignored.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL2")]
        public void NpLayer_Median_ComplexWithANaNImaginaryPart_ShouldReturnTheNaNElement()
        {
            using var a = np.array(new[] { new Complex(1, NpLayerPositiveNaN), new Complex(2, 0), new Complex(3, 0) });
            using var m = np.median(a);

            var z = (Complex)m.GetValue();
            z.Real.Should().Be(1.0);
            BitConverter.DoubleToInt64Bits(z.Imaginary).Should().Be(0x7FF8000000000000);
        }

        /// <summary>
        ///     <c>np.median([[1+2j, nan+1j, 3], [4+1j, 5, 6]], axis=1)</c>: NumPy checks every row, giving
        ///     <c>[nan+1j, 5+0j]</c>; the row without a NaN keeps its median. NumSharp's first row is the middle value of
        ///     the sort with the NaN ignored.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL2")]
        public void NpLayer_Median_ComplexWithANaN_AlongAnAxis_ShouldReturnTheNaNElementPerRow()
        {
            using var a = np.array(new[,]
            {
                { new Complex(1, 2), new Complex(NpLayerPositiveNaN, 1), new Complex(3, 0) },
                { new Complex(4, 1), new Complex(5, 0), new Complex(6, 0) },
            });
            using var m = np.median(a, axis: 1);

            var first = (Complex)m.GetValue(0);
            BitConverter.DoubleToInt64Bits(first.Real).Should().Be(0x7FF8000000000000);
            first.Imaginary.Should().Be(1.0);
            ((Complex)m.GetValue(1)).Should().Be(new Complex(5, 0));
        }

        // ------------------------------------------------------------------
        // NL3 — the flat float32 / float64 np.prod multiplies in SIMD lane order
        // ------------------------------------------------------------------

        /// <summary>
        ///     The flat <c>np.prod</c> of a C-contiguous float32 (3, 4) array <c>[[2**64, 2**64, 1, 1], [0, 1, 1, 1],
        ///     [1, 1, 1, 1]]</c>: NumPy's <c>FLOAT_multiply</c> reduce is a sequential loop (<c>acc *= *src</c>,
        ///     loops_arithm_fp.dispatch.c.src: multiply is not pairwise), so <c>2**64 * 2**64</c> overflows to inf before
        ///     the zero and the product is NaN (0xFFC00000). NumSharp's flat route (<c>prod_elementwise_il</c>, the IL SIMD
        ///     reduction) multiplies lane by lane and combines the lanes at the end, so the zero enters a lane product before
        ///     the two 2**64 meet and the product is 0. Random float32 data differs in the last bits the same way: 18 of 35
        ///     C-contiguous 1-D products of 4 to 1,000 values. The oracle's unbounded "reduction summation/two-pass precision (algorithm
        ///     order)" excuse covers prod today; narrow it to sum with the fix.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL3")]
        public void NpLayer_Prod_Float32Flat_ShouldMultiplyInNumPysSequentialOrder()
        {
            const float big = 18446744073709551616f;   // 2**64: its square overflows float32
            using var a = np.array(new float[,] { { big, big, 1, 1 }, { 0, 1, 1, 1 }, { 1, 1, 1, 1 } });
            using var r = np.prod(a);

            BitConverter.SingleToInt32Bits((float)r.GetValue()).Should().Be(unchecked((int)0xFFC00000));
        }

        /// <summary>
        ///     The flat <c>np.prod</c> of an F-ordered float32 (3, 4) array: NumPy's reduction iterator walks the operand in
        ///     MEMORY order (K order: the first axis is innermost here), so the product of these twelve values is 0x409D7A6B,
        ///     one ULP from the C-ordered array's 0x409D7A6C (which NumSharp matches). NumSharp gives 0x409D7A6D. The values
        ///     are the float32 casts of the fourth <c>uniform(0.5, 2.0, size=(3, 4))</c> draw of NumPy's
        ///     <c>default_rng(7)</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL3")]
        public void NpLayer_Prod_Float32FOrdered_ShouldMultiplyInMemoryOrder()
        {
            using var c = np.array(new float[,]
            {
                { 1.0543044805526733f, 0.5056013464927673f, 1.7450716495513916f, 0.7316915988922119f },
                { 0.9013989567756653f, 1.8204982280731201f, 1.2646862268447876f, 1.7707253694534302f },
                { 1.4595757722854614f, 1.6126564741134644f, 0.637243390083313f, 1.3117157220840454f },
            });
            using var f = np.asfortranarray(c);
            using var r = np.prod(f);

            BitConverter.SingleToInt32Bits((float)r.GetValue()).Should().Be(0x409D7A6B);
        }

        /// <summary>
        ///     The flat <c>np.prod</c> of a float32 view reversed on both axes: NumPy's reductions pass
        ///     <c>NPY_ITER_DONT_NEGATE_STRIDES</c>, so a reversed view is multiplied in its LOGICAL order, where the two
        ///     2**64 come first and overflow before the zero: NaN (0xFFC00000). NumSharp gives 0.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL3")]
        public void NpLayer_Prod_Float32Reversed_ShouldMultiplyInLogicalOrder()
        {
            const float big = 18446744073709551616f;   // 2**64
            using var owner = np.array(new float[,] { { 1, 1, 1, 1 }, { 1, 1, 1, 0 }, { 1, 1, big, big } });
            using var reversed = owner["::-1, ::-1"];
            using var r = np.prod(reversed);

            BitConverter.SingleToInt32Bits((float)r.GetValue()).Should().Be(unchecked((int)0xFFC00000));
        }

        /// <summary>
        ///     The flat <c>np.prod</c> of four float64 values: NumPy's sequential <c>DOUBLE_multiply</c> reduce gives NaN for
        ///     <c>[1e200, 1e200, 0, 1]</c> (inf, then inf * 0) and 0 for <c>[1e200, 0, 1e200, 1]</c>. NumSharp answers the
        ///     other way round (both answers fit lanes that pair the first value with the third and the second with the
        ///     fourth), so four values are enough to show the order; random float64 data differs in the last bits in 20 of
        ///     35 C-contiguous 1-D products of 4 to 1,000 values.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL3")]
        public void NpLayer_Prod_Float64Flat_ShouldMultiplyInNumPysSequentialOrder()
        {
            using var overflowFirst = np.array(new[] { 1e200, 1e200, 0.0, 1.0 });
            using var zeroFirst = np.array(new[] { 1e200, 0.0, 1e200, 1.0 });
            using var r1 = np.prod(overflowFirst);
            using var r2 = np.prod(zeroFirst);

            BitConverter.DoubleToInt64Bits((double)r1.GetValue()).Should().Be(unchecked((long)0xFFF8000000000000));
            BitConverter.DoubleToInt64Bits((double)r2.GetValue()).Should().Be(0L);
        }

        // ------------------------------------------------------------------
        // NL4 — 1-D np.polyval computes its first 0 * x in result_type(p, x)
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.polyval([1+0j], [inf])</c>: NumPy's Horner loop starts from <c>y = zeros_like(x)</c>, a REAL zero, so its
        ///     first <c>y * x</c> is the real NaN of <c>0 * inf</c> and <c>+ (1+0j)</c> gives <c>nan+0j</c>. NumSharp's 1-D path
        ///     (<c>DefaultEngine.Polyval</c>, Backends/Default/Polynomial/Default.Polynomial.cs) allocates <c>y</c> in
        ///     <c>result_type(p, x)</c>, complex, so the first product is the complex <c>(0+0j) * (inf+0j)</c> = NaN+NaNj.
        ///     The N-D path already starts in x's dtype.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL4")]
        public void NpLayer_Polyval_ComplexCoefficient_InfiniteX_ShouldKeepARealZeroImaginaryPart()
        {
            using var p = np.array(new[] { new Complex(1, 0) });
            using var x = np.array(new[] { double.PositiveInfinity });
            using var r = np.polyval(p, x);

            var z = (Complex)r.GetValue(0);
            BitConverter.DoubleToInt64Bits(z.Real).Should().Be(unchecked((long)0xFFF8000000000000));
            BitConverter.DoubleToInt64Bits(z.Imaginary).Should().Be(0L);
        }

        /// <summary>
        ///     <c>np.polyval([1+1j], [inf])</c>: NumPy gives <c>nan+1j</c> (the real NaN of <c>0 * inf</c> plus the
        ///     coefficient); NumSharp gives NaN+NaNj, for the reason of
        ///     <see cref="NpLayer_Polyval_ComplexCoefficient_InfiniteX_ShouldKeepARealZeroImaginaryPart"/>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL4")]
        public void NpLayer_Polyval_ComplexCoefficient_InfiniteX_ShouldKeepTheCoefficientsImaginaryPart()
        {
            using var p = np.array(new[] { new Complex(1, 1) });
            using var x = np.array(new[] { double.PositiveInfinity });
            using var r = np.polyval(p, x);

            var z = (Complex)r.GetValue(0);
            BitConverter.DoubleToInt64Bits(z.Real).Should().Be(unchecked((long)0xFFF8000000000000));
            z.Imaginary.Should().Be(1.0);
        }

        // ------------------------------------------------------------------
        // NL5 — argument checks in another order than NumPy's
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.nested_iters([], [[0], [1]])</c>: NumPy raises <c>ValueError: Must provide at least one operand</c>.
        ///     NumSharp raises the same text as an <see cref="ArgumentException"/> (APIs/np.nested_iters.cs), the base class
        ///     of NumSharp's <see cref="ValueError"/>, so a caller catching ValueError misses it.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL5")]
        public void NpLayer_NestedIters_NoOperand_ShouldRaiseValueError()
        {
            Action act = () => np.nested_iters(new NDArray[0], new[] { new[] { 0 }, new[] { 1 } });

            act.Should().Throw<ValueError>().WithMessage("Must provide at least one operand");
        }

        /// <summary>
        ///     NumPy's <c>NpyIter_NestedIters</c> converts <c>flags</c>, <c>order</c> and <c>casting</c> while it parses its
        ///     arguments (<c>PyArg_ParseTupleAndKeywords</c> with <c>NpyIter_GlobalFlagsConverter</c>,
        ///     <c>PyArray_OrderConverter</c>, <c>PyArray_CastingConverter</c>), BEFORE it reads <c>axes</c> or the operands. So
        ///     a bad converter value wins over a bad <c>axes</c> and over an empty operand list. NumSharp checks <c>axes</c>
        ///     first, then the operands, and only then the converters: each call below raises the axes error or the
        ///     no-operand error instead.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL5")]
        public void NpLayer_NestedIters_ConvertersShouldRunBeforeTheAxesAndOperandChecks()
        {
            using var a = np.arange(6).reshape(2, 3);
            int[][] oneLevel = { new[] { 0 } };   // NumPy: "axes must have at least 2 entries for nested iteration"

            Action badFlag = () => np.nested_iters(a, oneLevel, flags: new[] { "bogus" });
            Action badOrder = () => np.nested_iters(a, oneLevel, order: 'Q');
            Action badCasting = () => np.nested_iters(a, oneLevel, casting: "bogus");
            Action badFlagNoOperand = () => np.nested_iters(new NDArray[0], new[] { new[] { 0 }, new[] { 1 } }, flags: new[] { "bogus" });

            badFlag.Should().Throw<ValueError>().WithMessage("Unexpected iterator global flag \"bogus\"");
            badOrder.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Q')");
            badCasting.Should().Throw<ValueError>().WithMessage("casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got 'bogus')");
            badFlagNoOperand.Should().Throw<ValueError>().WithMessage("Unexpected iterator global flag \"bogus\"");
        }

        /// <summary>
        ///     <c>np.einsum('ij->i', ones((3, 4)), out=empty(5, int32))</c>: the out has both a wrong extent and a dtype the
        ///     result may not be cast to under 'safe'. NumPy's iterator reports the broadcast first (<c>ValueError: operands
        ///     could not be broadcast together with remapped shapes [original->remapped]: (3,4)->(3,4) (5,)->(5,newaxis)</c>).
        ///     NumSharp raises the cast's <see cref="TypeError"/>: its extent check runs after the contraction and after the
        ///     cast check (<c>DefaultEngine.Einsum</c>, Backends/Default/LinearAlgebra/Default.Einsum.cs). The test asserts
        ///     the exception type only: for an extent alone NumSharp words the error about the contraction on purpose
        ///     ("einsum() output parameter has shape (5) but the contraction produces (3)").
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL5")]
        public void NpLayer_Einsum_OutExtentError_ShouldPrecedeTheCastError()
        {
            using var a = np.ones(new Shape(3, 4));
            using var o = np.empty(new Shape(5), np.int32);

            Action act = () => np.einsum("ij->i", new[] { a }, @out: o);

            act.Should().Throw<ValueError>();
        }

        /// <summary>
        ///     An explicit <c>optimize=['einsum_path', (0, 1)]</c> for three operands contracts only two of them. NumPy's
        ///     <c>einsum_path</c> checks that one operand is left after the path and raises <c>RuntimeError: Invalid
        ///     einsum_path is specified: 1 more operands has to be contracted.</c>; so does a path of no steps for two
        ///     operands. NumSharp's np.einsum returns the full contraction for both (np.einsum_path itself raises NumPy's
        ///     error; np.einsum does not ask it).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL5")]
        public void NpLayer_Einsum_IncompleteExplicitPath_ShouldRaiseRuntimeError()
        {
            using var a = np.ones(new Shape(2, 3));
            using var b = np.ones(new Shape(3, 4));
            using var c = np.ones(new Shape(4, 5));

            Action threeOperands = () => np.einsum("ij,jk,kl->il", new[] { a, b, c }, optimize: new object[] { "einsum_path", new[] { 0, 1 } });
            Action noStep = () => np.einsum("ij,jk->ik", new[] { a, b }, optimize: new object[] { "einsum_path" });

            threeOperands.Should().Throw<RuntimeError>().WithMessage("Invalid einsum_path is specified: 1 more operands has to be contracted.");
            noStep.Should().Throw<RuntimeError>().WithMessage("Invalid einsum_path is specified: 1 more operands has to be contracted.");
        }

        // ------------------------------------------------------------------
        // NL6 — np.array of a ragged jagged C# array
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.array([[97], [99, 100]])</c> and its siblings: NumPy raises <c>ValueError: setting an array element with
        ///     a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) +
        ///     inhomogeneous part.</c> NumSharp's typed path takes the shape from the first element of each level
        ///     (<c>Shape.ExtractShape</c>), so a longer later row is truncated silently (<c>[[97], [99]]</c>, also for
        ///     double[][] and int[][][]) and a shorter one throws <see cref="IndexOutOfRangeException"/>. (An object[] list
        ///     takes the sequence walk, which raises NumPy's error already.)
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL6")]
        public void NpLayer_Array_RaggedJaggedArray_ShouldRaiseTheInhomogeneousShapeError()
        {
            const string afterOne = "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.";
            const string afterTwo = "setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (2, 2) + inhomogeneous part.";

            Action longerSecondRow = () => np.array(new int[][] { new[] { 97 }, new[] { 99, 100 } });
            Action shorterSecondRow = () => np.array(new int[][] { new[] { 97, 98 }, new[] { 99 } });
            Action doubles = () => np.array(new double[][] { new[] { 1.5 }, new[] { 2.5, 3.5 } });
            Action threeLevels = () => np.array(new int[][][] { new[] { new[] { 1 }, new[] { 2 } }, new[] { new[] { 3 }, new[] { 4, 5 } } });

            longerSecondRow.Should().Throw<ValueError>().WithMessage(afterOne);
            shorterSecondRow.Should().Throw<ValueError>().WithMessage(afterOne);
            doubles.Should().Throw<ValueError>().WithMessage(afterOne);
            threeLevels.Should().Throw<ValueError>().WithMessage(afterTwo);
        }

        // ------------------------------------------------------------------
        // NL7 — the boundary depth reaches user callbacks (the plan's B3)
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.apply_along_axis(f, 1, matrix)</c> where <c>f</c> calls <c>np.sum(other_matrix, 0)</c> and
        ///     <c>np.ravel(chararray)</c>: in NumPy the callback is ordinary user code, so it gets a 1 x 2 matrix
        ///     <c>[[4, 6]]</c> and a chararray, exactly as outside (and as NumSharp gives with a PLAIN outer operand).
        ///     NumSharp's callback gets a plain (2,) array and a plain array: <c>NDMatrixBoundary.Wrapped</c> raises the
        ///     per-thread depth around the whole computation, the user's callback included, and every boundary rule reads
        ///     a raised depth as "nested library call, stay plain". The plan's Appendix B3; its fix was P7 (rules on
        ///     results, the depth deleted), which the owner dropped on 2026-10-31.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL7")]
        public void NpLayer_ApplyAlongAxis_MatrixOperand_CallbackShouldSeeNumPysClasses()
        {
            using var m = np.asmatrix(np.array(new double[,] { { 1, 2 }, { 3, 4 } }));
            using var other = np.asmatrix(np.array(new double[,] { { 1, 2 }, { 3, 4 } }));
            using var text = np.@char.array(new[] { "ab", "cd" });
            // What the callback saw on its LAST call (both rows see the same): the classes and the sum's shape and values.
            Type sumClass = null, ravelClass = null;
            long[] sumShape = null;
            double[] sumValues = null;

            NDArray RowSum(NDArray row)
            {
                using var s = np.sum(other, 0);
                using var r = np.ravel(text);
                sumClass = s.GetType();
                sumShape = s.shape;
                sumValues = s.ToArray<double>();
                ravelClass = r.GetType();
                return np.sum(row);
            }

            using var result = np.apply_along_axis(RowSum, 1, m);

            sumClass.Should().Be(typeof(NDMatrix));
            sumShape.Should().Equal(1L, 2L);
            sumValues.Should().Equal(4.0, 6.0);
            ravelClass.Should().Be(typeof(NDCharArray));
        }

        /// <summary>
        ///     <c>np.apply_over_axes(h, matrix, [0])</c> where <c>h</c> calls <c>np.sum(other_matrix, 0)</c>: NumPy's callback
        ///     gets a 1 x 2 matrix; NumSharp's gets a plain (2,) array, because the call runs inside the boundary's plain
        ///     region for the matrix operand (the depth raised, as in
        ///     <see cref="NpLayer_ApplyAlongAxis_MatrixOperand_CallbackShouldSeeNumPysClasses"/>). The result itself agrees
        ///     (a plain (1, 3) array <c>[[5, 7, 9]]</c> in both).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL7")]
        public void NpLayer_ApplyOverAxes_MatrixOperand_CallbackShouldSeeNumPysClasses()
        {
            using var m = np.asmatrix(np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }));
            using var other = np.asmatrix(np.array(new double[,] { { 1, 2 }, { 3, 4 } }));
            Type sumClass = null;
            long[] sumShape = null;

            NDArray AxisSum(NDArray arr, int axis)
            {
                using var s = np.sum(other, 0);
                sumClass = s.GetType();
                sumShape = s.shape;
                return np.sum(arr, axis);
            }

            using var result = np.apply_over_axes(AxisSum, m, new[] { 0 });

            result.ToArray<double>().Should().Equal(5.0, 7.0, 9.0);
            sumClass.Should().Be(typeof(NDMatrix));
            sumShape.Should().Equal(1L, 2L);
        }

        // ------------------------------------------------------------------
        // NL8 — np.sum's reduce step leaves the NaN of NaN + NaN to the JIT
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.sum</c> of a run whose first element is a NaN and whose rest holds another NaN keeps the FIRST
        ///     element's NaN in NumPy: add.reduce seeds the accumulator with the first element, then runs
        ///     <c>io1 += pairwise_sum(rest)</c>, and x86's <c>addsd</c> keeps its first operand's NaN when both are NaN.
        ///     NumPy 2.4.2 (probed, win-amd64): <c>[nan(1), 2, nan(2)]</c> sums to payload 1 in float64 and float32, and
        ///     rows <c>[[nan(1), 2, nan(2)], [1, nan(3), nan(4)]]</c> along axis 1 to payloads 1 and 3.
        /// </summary>
        /// <remarks>
        ///     Root cause: <c>Default.Reduction.Add.Exact.cs</c>'s <c>AccumulateDouble</c> / <c>AccumulateSingle</c> write
        ///     the step as C#'s <c>*acc = *acc + s</c>, and C# leaves the operand order of a commutative <c>+</c> to the JIT
        ///     (<c>kernels.md</c>: encode NaN priority explicitly). A Debug build of NumSharp.Core compiles the step without
        ///     optimization, loads <c>s</c> first and keeps the pairwise sum's NaN (payloads 2 and 4); a Release build
        ///     happens to keep the accumulator in its register, NumPy's order, so this test FAILS in a Debug build and
        ///     passes in Release, and so do <c>NDEvaluateStreamingTests.ShortRows_SeveralNaNs_KeepTheRunningSumsNaN</c> and
        ///     <c>NpTrapezoidTest.Trapezoid_NaNSpacing_IsTheProductsFirstOperand</c> (np.trapezoid's final sum), which
        ///     stay unmarked so CI's Release run keeps guarding NumPy's answer. The fix chooses the accumulator's NaN
        ///     explicitly in both steps (and in any other <c>acc + s</c> combine the reductions write in C#).
        /// </remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL8")]
        public void NpLayer_Sum_RunOfNaNs_ShouldKeepTheAccumulatorsNaN_InEveryBuild()
        {
            static double N64(long payload) => BitConverter.Int64BitsToDouble(0x7FF8000000000000 | payload);
            static float N32(int payload) => BitConverter.Int32BitsToSingle(0x7FC00000 | payload);
            static long Bits64(NDArray r, int i) => BitConverter.DoubleToInt64Bits(r.GetDouble(i));

            using var a = np.array(new[] { N64(1), 2.0, N64(2) });
            using var s = np.sum(a);
            BitConverter.DoubleToInt64Bits(s.GetDouble()).Should().Be(0x7FF8000000000001, "float64 run: the first element's NaN");

            using var x = np.array(new[,] { { N64(1), 2.0, N64(2) }, { 1.0, N64(3), N64(4) } });
            using var rows = np.sum(x, 1);
            Bits64(rows, 0).Should().Be(0x7FF8000000000001, "row 0 keeps the accumulator's NaN");
            Bits64(rows, 1).Should().Be(0x7FF8000000000003, "row 1 keeps the accumulator's NaN");

            using var f = np.array(new[] { N32(1), 2f, N32(2) });
            using var sf = np.sum(f);
            BitConverter.SingleToInt32Bits(sf.GetSingle()).Should().Be(0x7FC00001, "float32 run: the first element's NaN");
        }

        // ------------------------------------------------------------------
        // NL9 — np.polyder counts p.size terms and computes one product fewer than NumPy
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's polyder is <c>n = len(p) - 1; y = p[:-1] * arange(n, 0, -1); val = p if m == 0 else
        ///     polyder(y, m - 1)</c>: the ramp has <c>len(p) - 1</c> terms (the FIRST axis), so a 2-D p is differentiated
        ///     along its rows, the rows broadcasting against the ramp. Probed: <c>np.polyder(np.ones((3, 2)))</c> is float64
        ///     <c>[[2., 1.], [2., 1.]]</c> and <c>np.polyder(np.arange(6.).reshape(3, 2))</c> is <c>[[0., 1.], [4., 3.]]</c>.
        ///     NumSharp's ramp has <c>p.size - 1</c> terms (5), which do not broadcast against the (2, 2) rows, so both calls
        ///     raise the multiply's IncorrectShapeException.
        /// </summary>
        /// <remarks>Root cause: <c>DefaultEngine.Polyder</c> (Backends/Default/Polynomial/Default.Polynomial.cs) takes
        ///     <c>n = p.size - 1</c> and loops m times instead of NumPy's len() and its recursion. A 1-D p is the same either
        ///     way, which is why the oracle's <c>poly</c> tier (1-D series) passes.</remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL9")]
        public void NpLayer_Polyder_TwoDimensionalP_ShouldDifferentiateAlongTheFirstAxis()
        {
            using var ones = np.ones(new Shape(3, 2));
            using var ramp = np.arange(6.0).reshape(3, 2);

            using var d1 = np.polyder(ones);
            d1.shape.Should().Equal(2L, 2L);
            d1.ToArray<double>().Should().Equal(2.0, 1.0, 2.0, 1.0);

            using var d2 = np.polyder(ramp);
            d2.ToArray<double>().Should().Equal(0.0, 1.0, 4.0, 3.0);
        }

        /// <summary>
        ///     NumPy's <c>n = len(p) - 1</c> runs before the <c>m == 0</c> test, so a 0-d p raises <c>TypeError: len() of
        ///     unsized object</c> at EVERY order, 0 included (probed for m = 0, 1 and 2). NumSharp returns the 0-d array for
        ///     m = 0 and raises the slice's IndexError (<c>too many indices for array</c>) for m &gt;= 1.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL9")]
        public void NpLayer_Polyder_ZeroDimensionalP_ShouldRaiseLensTypeErrorAtEveryOrder()
        {
            using var p = np.array(5.0);

            foreach (int m in new[] { 0, 1, 2 })
            {
                Action act = () => np.polyder(p, m);
                act.Should().Throw<TypeError>().WithMessage("len() of unsized object", $"m = {m}");
            }
        }

        /// <summary>
        ///     NumPy's recursion computes <c>y = p[:-1] * arange(n, 0, -1)</c> at every level, the last one (m == 0) included,
        ///     and discards that last product - but it still raises when the product does not broadcast. Probed:
        ///     <c>np.polyder(np.ones((3, 4)), 0)</c> raises <c>ValueError: operands could not be broadcast together with
        ///     shapes (2,4) (2,) </c>, and <c>np.polyder(np.ones((3, 2)), 2)</c> raises the same for <c>(0,2) (0,) </c> (the
        ///     third product, after two that succeed). NumSharp computes only m products: it returns p for m = 0, and its
        ///     m = 2 call meets its own p.size ramp error at the first product instead. NumSharp raises a broadcast mismatch
        ///     as IncorrectShapeException with NumPy's text (the house type for NumPy's ValueError there).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL9")]
        public void NpLayer_Polyder_ExtraProduct_ShouldRaiseWhereNumPysRecursionDoes()
        {
            using var wide = np.ones(new Shape(3, 4));
            using var tall = np.ones(new Shape(3, 2));

            Action orderZero = () => np.polyder(wide, 0);
            Action orderTwo = () => np.polyder(tall, 2);

            orderZero.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (2,4) (2,) ");
            orderTwo.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (0,2) (0,) ");
        }

        // ------------------------------------------------------------------
        // NL10 — np.einsum_path with an ellipsis and too few operands
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>_parse_einsum_input</c> expands an ellipsis by reading <c>operands[num].shape</c> for each term that
        ///     holds one, before einsum_path compares the number of terms with the number of operands; a term with no operand
        ///     behind it raises <c>IndexError: list index out of range</c>. Probed: <c>np.einsum_path("...-&gt;...")</c> and
        ///     <c>np.einsum_path("...a,...-&gt;...", np.ones(3))</c>. NumSharp's planner indexes its shape array the same way
        ///     and lets .NET's <see cref="IndexOutOfRangeException"/> ("Index was outside the bounds of the array.") escape.
        /// </summary>
        /// <remarks>Root cause: <c>EinsumPathPlanner</c>'s port of <c>_parse_einsum_input</c>
        ///     (src/NumSharp.Core/LinearAlgebra/EinsumPathPlanner*.cs), reached through <c>DefaultEngine.EinsumPath</c>.</remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL10")]
        public void NpLayer_EinsumPath_EllipsisWithoutItsOperand_ShouldRaiseIndexError()
        {
            using var v = np.ones(new Shape(3));

            Action none = () => np.einsum_path("...->...", new NDArray[0]);
            Action tooFew = () => np.einsum_path("...a,...->...", new[] { v });

            none.Should().Throw<IndexError>().WithMessage("list index out of range");
            tooFew.Should().Throw<IndexError>().WithMessage("list index out of range");
        }

        // ------------------------------------------------------------------
        // NL11 — np.isscalar: Half, BigInteger and Type objects
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's isscalar is <c>isinstance(element, generic) or type(element) in ScalarType or isinstance(element,
        ///     numbers.Number)</c>: a float16 scalar is a <c>generic</c>, so <c>np.isscalar(np.float16(1.5))</c> is True
        ///     (probed). <see cref="Half"/> is NumSharp's float16 scalar, but it is no CLR primitive, so NumSharp's
        ///     <c>type.IsPrimitive || obj is decimal</c> answers False.
        /// </summary>
        /// <remarks>Root cause: <c>NDArray.isscalar</c> (src/NumSharp.Core/Backends/NDArray.Numeric.cs) has no Half
        ///     case.</remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL11")]
        public void NpLayer_Isscalar_Half_ShouldBeAScalar()
        {
            np.isscalar((Half)1.5).Should().BeTrue("np.isscalar(np.float16(1.5)) is True");
        }

        /// <summary>
        ///     A Python int of any size is a <c>numbers.Number</c>, so <c>np.isscalar(2**70)</c> is True (probed).
        ///     <see cref="BigInteger"/> is how a C# caller spells a Python int past int64 (the house map), but it is no CLR
        ///     primitive, so NumSharp answers False.
        /// </summary>
        /// <remarks>Root cause: <c>NDArray.isscalar</c> has no BigInteger case.</remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL11")]
        public void NpLayer_Isscalar_BigInteger_ShouldBeAScalar()
        {
            np.isscalar(BigInteger.Pow(2, 70)).Should().BeTrue("np.isscalar(2**70) is True");
        }

        /// <summary>
        ///     A type is not a scalar: <c>np.isscalar(int)</c> and <c>np.isscalar(np.float64)</c> are False (probed; a class
        ///     object is neither a <c>generic</c> nor a <c>numbers.Number</c>). NumSharp reads a <see cref="Type"/> argument
        ///     as the type it names (<c>obj as Type ?? obj.GetType()</c>), so <c>np.isscalar(typeof(int))</c> answers for an
        ///     int: True.
        /// </summary>
        /// <remarks>Root cause: <c>NDArray.isscalar</c>'s <c>case Type _: break;</c>, which sends a Type object to the
        ///     primitive test of the type it names.</remarks>
        [TestMethod, OpenBugs(IssueUrl = "np-layer-NL11")]
        public void NpLayer_Isscalar_TypeObject_ShouldNotBeAScalar()
        {
            np.isscalar(typeof(int)).Should().BeFalse("np.isscalar(int) is False");
            np.isscalar(typeof(double)).Should().BeFalse("np.isscalar(float) is False");
        }
    }
}
