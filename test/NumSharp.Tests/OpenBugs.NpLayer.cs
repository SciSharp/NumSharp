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
    }
}
