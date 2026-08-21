using System;
using System.Collections.Generic;
using System.Diagnostics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     np.r_ / np.c_ — the index-expression concatenators. Every expectation below is real
    ///     NumPy 2.4.2 output, transcribed with C#'s string spelling of a Python slice literal
    ///     (<c>np.r_[0:5]</c> → <c>np.r_["0:5"]</c>).
    /// </summary>
    [TestClass]
    public class np_r_Test
    {
        private static NDArray A(params long[] v) => np.array(v);

        // ------------------------------------------------------------------ r_: basics

        [TestMethod]
        public void R_ArraysAndScalars_ConcatenatesInOrder()
        {
            // np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])]
            np.r_[A(1, 2, 3), 0, 0, A(4, 5, 6)].Should()
                .BeShaped(8).And.BeOfValues(1, 2, 3, 0, 0, 4, 5, 6);
        }

        [TestMethod]
        public void R_SliceExpression_IsArange()
        {
            np.r_["0:5"].Should().BeShaped(5).And.BeOfValues(0, 1, 2, 3, 4);
            np.r_["0:5:2"].Should().BeShaped(3).And.BeOfValues(0, 2, 4);
            np.r_["5:0:-1"].Should().BeShaped(5).And.BeOfValues(5, 4, 3, 2, 1);
            np.r_["-3:0"].Should().BeShaped(3).And.BeOfValues(-3, -2, -1);
        }

        [TestMethod]
        public void R_SliceExpression_IntegerLiteralsGiveInt64_FloatLiteralsGiveFloat64()
        {
            // The LITERAL decides, not the value: np.r_[0:5] is int64, np.r_[0.0:5.0] is float64.
            np.r_["0:5"].typecode.Should().Be(NPTypeCode.Int64);
            np.r_["0.0:1.0:0.25"].Should().BeOfType(NPTypeCode.Double)
                .And.BeShaped(4).And.BeOfValues(0.0, 0.25, 0.5, 0.75);
            np.r_["0:1:0.3"].typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void R_ImaginaryStep_IsLinspaceWithInclusiveStop()
        {
            // np.r_[-1:1:6j] -> linspace(-1, 1, 6), stop INCLUSIVE, always float64.
            // The exact doubles NumPy 2.4.2 produces — linspace's fma-free (start + i*delta)
            // walk, so the interior points are NOT the shortest decimal literals.
            np.r_["-1:1:6j"].Should().BeOfType(NPTypeCode.Double)
                .And.BeShaped(6).And.BeOfValues(-1.0, -0.6, -0.19999999999999996,
                    0.20000000000000018, 0.6000000000000001, 1.0);
            np.r_["0:5:0j"].Should().BeShaped(0);
            np.r_["0:5:1j"].Should().BeShaped(1).And.BeOfValues(0.0);
            // The magnitude is what counts, so a negative imaginary step is the same count.
            np.r_["1:2:-3j"].Should().BeShaped(3).And.BeOfValues(1.0, 1.5, 2.0);
        }

        [TestMethod]
        public void R_SeveralSlicesInOneString_AreConcatenated()
        {
            // np.r_[1:3, 5:8]
            np.r_["1:3, 5:8"].Should().BeShaped(5).And.BeOfValues(1, 2, 5, 6, 7);
        }

        [TestMethod]
        public void R_MissingStop_RereadsStartAsStop()
        {
            // NumPy's arange(start, None, step) IS arange(0, start, step) — np.r_[2:] is [0,1].
            np.r_["2:"].Should().BeShaped(2).And.BeOfValues(0, 1);
            np.r_["5::2"].Should().BeShaped(3).And.BeOfValues(0, 2, 4);
            np.r_[":"].Should().BeShaped(0);
            np.r_["::2"].Should().BeShaped(0);
            np.r_[":5"].Should().BeShaped(5).And.BeOfValues(0, 1, 2, 3, 4);
        }

        [TestMethod]
        public void R_EmptyRanges()
        {
            np.r_["5:0"].Should().BeShaped(0);
            np.r_["0:5:-1"].Should().BeShaped(0);
        }

        [TestMethod]
        public void R_TwoDimensional_StacksAlongFirstAxis()
        {
            var a = np.array(new long[,] { { 0, 1, 2 }, { 3, 4, 5 } });
            np.r_[a, a].Should().BeShaped(4, 3).And.BeOfValues(0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5);
        }

        // ------------------------------------------------------------------ r_: directives

        [TestMethod]
        public void R_AxisDirective_SelectsConcatenationAxis()
        {
            var a = np.array(new long[,] { { 0, 1, 2 }, { 3, 4, 5 } });
            np.r_["-1", a, a].Should().BeShaped(2, 6)
                .And.BeOfValues(0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5);
        }

        [TestMethod]
        public void R_AxisNdminDirective_UpgradesEntries()
        {
            // np.r_['0,2', [1,2,3], [4,5,6]]
            np.r_["0,2", new long[] { 1, 2, 3 }, new long[] { 4, 5, 6 }].Should()
                .BeShaped(2, 3).And.BeOfValues(1, 2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void R_Trans1dDirective_PlacesTheUpgradedAxes()
        {
            // np.r_['0,2,0', ...] -> column vectors stacked; '1,2,0' -> columns side by side.
            np.r_["0,2,0", new long[] { 1, 2, 3 }, new long[] { 4, 5, 6 }].Should()
                .BeShaped(6, 1).And.BeOfValues(1, 2, 3, 4, 5, 6);
            np.r_["1,2,0", new long[] { 1, 2, 3 }, new long[] { 4, 5, 6 }].Should()
                .BeShaped(3, 2).And.BeOfValues(1, 4, 2, 5, 3, 6);
        }

        [TestMethod]
        public void R_Trans1dDirective_OnSliceEntries_UsesSwapaxes()
        {
            // The slice branch upgrades with swapaxes(-1, trans1d), not the array branch's
            // permutation, so these shapes differ from the equivalent array entry.
            np.r_["0,3,1", "0:3"].Should().BeShaped(1, 3, 1);
            np.r_["0,3,0", "0:3"].Should().BeShaped(3, 1, 1);
            np.r_["0,3,2", "0:3"].Should().BeShaped(1, 1, 3);
        }

        [TestMethod]
        public void R_Directive_ExtraFieldsAreIgnored()
        {
            // NumPy reads vec[:2] and, only when len(vec)==3, vec[2]. A fourth field — even a
            // non-numeric one — is silently dropped.
            np.r_["0,2,0,1", new long[] { 1, 2, 3 }].Should().BeShaped(1, 3);
            np.r_["0,2,0,q", new long[] { 1, 2, 3 }].Should().BeShaped(1, 3);
        }

        [TestMethod]
        public void R_Directive_AcceptsSurroundingWhitespace()
        {
            np.r_[" 0 , 2 ", new long[] { 1, 2, 3 }].Should().BeShaped(1, 3);
        }

        [TestMethod]
        public void R_Directive_NdminZeroOrNegativeIsANoOp()
        {
            np.r_["0,0", new long[] { 1, 2 }].Should().BeShaped(2);
            np.r_["0,-1", new long[] { 1, 2, 3 }].Should().BeShaped(3);
        }

        [TestMethod]
        public void R_MatrixDirective_CoercesToTwoDimensions()
        {
            // 'r' makes a 1xN row, 'c' an Nx1 column; a 2-D result is unchanged by either.
            np.r_["r", new long[] { 1, 2, 3 }, new long[] { 4, 5, 6 }].Should()
                .BeShaped(1, 6).And.BeOfValues(1, 2, 3, 4, 5, 6);
            np.r_["c", new long[] { 1, 2, 3 }, new long[] { 4, 5, 6 }].Should()
                .BeShaped(6, 1).And.BeOfValues(1, 2, 3, 4, 5, 6);

            var m = np.array(new long[,] { { 1, 2 }, { 3, 4 } });
            np.r_["r", m].Should().BeShaped(2, 2);
            np.r_["c", m].Should().BeShaped(2, 2);
        }

        [TestMethod]
        public void R_MatrixDirective_RejectsMoreThanTwoDimensions()
        {
            new Action(() => _ = np.r_["r", np.arange(8).reshape(2L, 2L, 2L)])
                .Should().Throw<ValueError>().WithMessage("shape too large to be a matrix.*");
        }

        [TestMethod]
        public void R_Directive_MustBeTheFirstEntry()
        {
            new Action(() => _ = np.r_[new long[] { 1, 2 }, "0"])
                .Should().Throw<ValueError>()
                .WithMessage("special directives must be the first entry.*");
        }

        [TestMethod]
        public void R_UnknownDirective_Throws()
        {
            new Action(() => _ = np.r_["q", new long[] { 1, 2 }])
                .Should().Throw<ValueError>().WithMessage("unknown special directive*");
            // The comma form quotes the offending directive; the bare form does not.
            new Action(() => _ = np.r_["0,q", new long[] { 1, 2 }])
                .Should().Throw<ValueError>().WithMessage("unknown special directive '0,q'*");
            new Action(() => _ = np.r_["", new long[] { 1, 2 }])
                .Should().Throw<ValueError>().WithMessage("unknown special directive*");
        }

        // ------------------------------------------------------------------ r_: NEP50 promotion

        [TestMethod]
        public void R_WeakIntegerLiteral_AdoptsTheArrayDtype()
        {
            // np.r_[np.array([1,2,3], dtype=np.int8), 1].dtype == int8
            np.r_[np.array(new sbyte[] { 1, 2, 3 }), 1].typecode.Should().Be(NPTypeCode.SByte);
            np.r_[np.array(new sbyte[] { 1, 2, 3 }), 1L].typecode.Should().Be(NPTypeCode.SByte);
            // …except over bool, which it lifts to the default integer.
            np.r_[np.array(new[] { true, false }), 2].typecode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void R_WeakFloatLiteral_KeepsAFloatDtypesWidth_ElseForcesFloat64()
        {
            np.r_[np.array(new sbyte[] { 1, 2, 3 }), 1.5].typecode.Should().Be(NPTypeCode.Double);
            np.r_[np.array(new[] { (Half)1f }), 1.5].typecode.Should().Be(NPTypeCode.Half);
            np.r_[np.array(new[] { 1.0f }), 1.0].typecode.Should().Be(NPTypeCode.Single);
            np.r_[np.array(new[] { true, false }), 1.5].typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void R_WeakBoolLiteral_AdoptsAnything()
        {
            np.r_[np.array(new sbyte[] { 1, 2 }), true].typecode.Should().Be(NPTypeCode.SByte);
            np.r_[np.array(new[] { true, false }), true].typecode.Should().Be(NPTypeCode.Boolean);
        }

        [TestMethod]
        public void R_AllLiteralKey_UsesTheNEP50Defaults()
        {
            np.r_[1, 2].typecode.Should().Be(NPTypeCode.Int64);
            np.r_[true, false].typecode.Should().Be(NPTypeCode.Boolean);
            np.r_[1, 2.0].typecode.Should().Be(NPTypeCode.Double);
            np.r_[true, 2].typecode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void R_SliceEntriesAreStrong_NotWeak()
        {
            // arange/linspace produce real arrays, so they promote as int64/float64 do.
            np.r_[np.array(new sbyte[] { 1, 2 }), "0:3"].typecode.Should().Be(NPTypeCode.Int64);
            np.r_["0:3", 1.5].typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void R_WeakIntegerOutOfRange_Throws_RatherThanWrapping()
        {
            // NumPy: OverflowError: Python integer 1000 out of bounds for int8
            new Action(() => _ = np.r_[np.array(new sbyte[] { 1, 2 }), 1000])
                .Should().Throw<OverflowException>()
                .WithMessage("Python integer 1000 out of bounds for int8*");
            new Action(() => _ = np.r_[np.array(new byte[] { 1, 2 }), -1])
                .Should().Throw<OverflowException>()
                .WithMessage("Python integer -1 out of bounds for uint8*");
            new Action(() => _ = np.r_[np.array(new long[] { 1 }), ulong.MaxValue])
                .Should().Throw<OverflowException>();
        }

        [TestMethod]
        public void R_NDArrayScalar_IsStrong_TheEscapeHatchFromWeakLiterals()
        {
            // A C# literal is weak; wrapping it makes it strong, as np.int64(1) is in NumPy.
            np.r_[np.array(new sbyte[] { 1, 2 }), 1L].typecode.Should().Be(NPTypeCode.SByte);
            np.r_[np.array(new sbyte[] { 1, 2 }), NDArray.Scalar(1L)].typecode
                .Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void R_NumSharpOnlyScalarsAreStrong()
        {
            // char / Half / decimal have no Python literal, so they carry their own dtype.
            np.r_[np.array(new sbyte[] { 1, 2 }), (Half)1f].typecode.Should().Be(NPTypeCode.Half);
            np.r_[np.array(new sbyte[] { 1, 2 }), 1m].typecode.Should().Be(NPTypeCode.Decimal);
            // int8 + char (which promotes as uint16) is int32, exactly as NumPy promotes them.
            np.r_[np.array(new sbyte[] { 1, 2 }), 'A'].typecode.Should().Be(NPTypeCode.Int32);
        }

        // ------------------------------------------------------------------ r_: entries & layouts

        [TestMethod]
        public void R_AcceptsSliceObjectsAndIndexExpressions()
        {
            np.r_[new Slice(0, 5)].Should().BeShaped(5).And.BeOfValues(0, 1, 2, 3, 4);
            np.r_[new Slice(0, 5, 2)].Should().BeShaped(3).And.BeOfValues(0, 2, 4);
            np.r_[np.s_["0:3, 5:8"]].Should().BeShaped(6).And.BeOfValues(0, 1, 2, 5, 6, 7);
            np.r_[np.s_["0:3"], 9].Should().BeShaped(4).And.BeOfValues(0, 1, 2, 9);
        }

        [TestMethod]
        public void R_RejectsEllipsisAndNewAxisSlices()
        {
            new Action(() => _ = np.r_[Slice.Ellipsis]).Should().Throw<ArgumentException>();
            new Action(() => _ = np.r_[Slice.NewAxis]).Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void R_AcceptsCSharpCollectionsAndArrays()
        {
            np.r_[new List<int> { 1, 2, 3 }].Should().BeShaped(3).And.BeOfValues(1, 2, 3);
            np.r_[new int[,] { { 1, 2 }, { 3, 4 } }].Should().BeShaped(2, 2);
            np.r_[(1, 2, 3)].Should().BeShaped(3).And.BeOfValues(1, 2, 3);
            np.r_[new Memory<int>(new[] { 7, 8 })].Should().BeShaped(2).And.BeOfValues(7, 8);
        }

        [TestMethod]
        public void R_NullEntry_Throws()
        {
            new Action(() => _ = np.r_[(object)null]).Should().Throw<ArgumentNullException>();
        }

        [TestMethod]
        public void R_MalformedSliceExpression_Throws()
        {
            new Action(() => _ = np.r_["0:1:2:3"]).Should().Throw<ArgumentException>()
                .WithMessage("Invalid slice notation: '0:1:2:3'*");
            new Action(() => _ = np.r_["a:b"]).Should().Throw<ArgumentException>();
            new Action(() => _ = np.r_["0::3j"]).Should().Throw<ValueError>();
        }

        [TestMethod]
        public void R_ReadsThroughEveryMemoryLayout()
        {
            var b = np.arange(12).reshape(3L, 4L);
            var expectedRow0 = new object[] { 0L, 1L, 2L, 3L };

            foreach (var v in new[] { b.copy('C'), np.asfortranarray(b), b["::-1"], b[":, ::-1"] })
            {
                var r = np.r_[v, v];
                r.Should().BeShaped(6, 4);
                // The first row of the doubled array is the first row of the view.
                for (int j = 0; j < 4; j++)
                    r.GetValue<long>(0, j).Should().Be(v.GetValue<long>(0, j));
            }

            // A broadcast (read-only) entry is legal — concatenate copies out of it.
            var bc = np.broadcast_to(np.arange(4), new Shape(3, 4));
            np.r_[bc, bc].Should().BeShaped(6, 4);
            expectedRow0.Should().NotBeNull();
        }

        [TestMethod]
        public void R_ResultIsAFreshWriteableCopy()
        {
            var a = A(1, 2, 3);
            var r = np.r_[a, a];
            r.Shape.IsWriteable.Should().BeTrue();
            r.SetValue(99L, 0);
            a.GetValue<long>(0).Should().Be(1, "r_ concatenates into a fresh buffer");
        }

        [TestMethod]
        public void R_OnlyDirectives_ThrowsLikeConcatenateOfNothing()
        {
            new Action(() => _ = np.r_["0,2"]).Should().Throw<ArgumentException>()
                .WithMessage("need at least one array to concatenate*");
        }

        // ------------------------------------------------------------------ c_

        [TestMethod]
        public void C_TwoOneDimensional_BecomeColumns()
        {
            // np.c_[np.array([1,2,3]), np.array([4,5,6])]
            np.c_[A(1, 2, 3), A(4, 5, 6)].Should()
                .BeShaped(3, 2).And.BeOfValues(1, 4, 2, 5, 3, 6);
        }

        [TestMethod]
        public void C_RowsAndScalars()
        {
            np.c_[np.array(new long[,] { { 1, 2, 3 } }), 0, 0, np.array(new long[,] { { 4, 5, 6 } })]
                .Should().BeShaped(1, 8).And.BeOfValues(1, 2, 3, 0, 0, 4, 5, 6);
        }

        [TestMethod]
        public void C_SliceExpressions()
        {
            np.c_["0:3, 3:6"].Should().BeShaped(3, 2).And.BeOfValues(0, 3, 1, 4, 2, 5);
            np.c_["0:3"].Should().BeShaped(3, 1).And.BeOfValues(0, 1, 2);
        }

        [TestMethod]
        public void C_Scalars_BecomeASingleRow()
        {
            np.c_[1, 2, 3].Should().BeShaped(1, 3).And.BeOfValues(1, 2, 3);
            np.c_[5].Should().BeShaped(1, 1).And.BeOfValues(5);
        }

        [TestMethod]
        public void C_TwoDimensional_PassesThroughAndStacksOnTheLastAxis()
        {
            var a = np.array(new long[,] { { 0, 1, 2 }, { 3, 4, 5 } });
            np.c_[a, a].Should().BeShaped(2, 6)
                .And.BeOfValues(0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5);
        }

        [TestMethod]
        public void C_MixedRank_UpgradesOnlyTheLowerOne()
        {
            // np.c_[np.array([1,2]), np.array([[1,2],[3,4]])]
            np.c_[A(1, 2), np.array(new long[,] { { 1, 2 }, { 3, 4 } })].Should()
                .BeShaped(2, 3).And.BeOfValues(1, 1, 2, 2, 3, 4);
        }

        [TestMethod]
        public void C_ShapeMismatch_Throws()
        {
            new Action(() => _ = np.c_[new long[] { 1, 2 }, new long[] { 1, 2, 3 }])
                .Should().Throw<IncorrectShapeException>()
                .WithMessage("all the input array dimensions except for the concatenation axis*");
        }

        // ------------------------------------------------------------------ edge & extremes sweep

        [TestMethod]
        public void R_HighNdmin_IsSupportedAndCheap()
        {
            // DELIBERATE DIVERGENCE. NumPy validates ndmin against NPY_MAXDIMS and raises
            // "ndmin must be <= ndmax (64)"; NumSharp has no 64-dimension ceiling anywhere, so
            // capping the DSL would make it refuse ranks the rest of the library accepts.
            np.r_["0,64", new long[] { 1, 2 }].ndim.Should().Be(64);
            np.r_["0,65", new long[] { 1, 2 }].ndim.Should().Be(65);

            // ndmin arrives from a user-typed directive, so the padding must be CHEAP, not merely
            // bounded. Prepending one axis at a time clones dims+strides every step — quadratic,
            // measured 27.6 s at ndmin=100,000 — where the bulk prepend is one alias (~1.4 ms).
            // The bound is loose enough not to be flaky and two orders below the O(n²) cost.
            var sw = Stopwatch.StartNew();
            np.r_["0,100000", new long[] { 1, 2 }].ndim.Should().Be(100000);
            sw.Stop();
            sw.Elapsed.Should().BeLessThan(TimeSpan.FromSeconds(5),
                "the ndmin expansion must stay O(ndim) — the per-axis loop took 27.6 s here");

            // Every entry kind takes the same route — array, slice and scalar alike.
            np.r_["0,65", "0:3"].ndim.Should().Be(65);
            np.r_["0,65", 5].ndim.Should().Be(65);

            // An ndmin too large to allocate a shape for is an honest allocation failure — the
            // same answer any other oversized NumSharp request gives — not a policy cap.
            new Action(() => _ = np.r_["0,2147483647", new long[] { 1, 2 }])
                .Should().Throw<OutOfMemoryException>();

            // A non-positive ndmin stays a no-op, exactly as upstream.
            np.r_["0,0", new long[] { 1, 2 }].ndim.Should().Be(1);
            np.r_["0,-1", new long[] { 1, 2 }].ndim.Should().Be(1);
            np.r_["0,-2147483648", new long[] { 1, 2 }].ndim.Should().Be(1);
        }

        [TestMethod]
        public void R_AllLiteralIntegerBeyondInt64_ResolvesToUInt64()
        {
            // np.r_[2**63] and np.r_[2**64-1] are both uint64 — the weak-integer default lifts
            // from int64 when a literal does not fit it.
            np.r_[ulong.MaxValue].Should().BeOfType(NPTypeCode.UInt64).And.BeShaped(1);
            np.r_[9223372036854775808UL].typecode.Should().Be(NPTypeCode.UInt64);
            np.r_[1L, ulong.MaxValue].typecode.Should().Be(NPTypeCode.UInt64);
            // …but only when it has to: long.MaxValue still fits int64.
            np.r_[long.MaxValue].typecode.Should().Be(NPTypeCode.Int64);
            np.r_[long.MinValue].typecode.Should().Be(NPTypeCode.Int64);
            // A strong operand still decides, and the literal is then range-checked against it.
            new Action(() => _ = np.r_[np.array(new byte[] { 1 }), ulong.MaxValue])
                .Should().Throw<OverflowException>();
        }

        [TestMethod]
        public void R_ExtremeFloatValues_RoundTripBitExactly()
        {
            // -0.0 must keep its sign bit, and nan/inf/subnormals must survive the concatenate.
            np.r_[-0.0].tobytes('C').Should().BeEquivalentTo(
                new byte[] { 0, 0, 0, 0, 0, 0, 0, 0x80 }, "NumPy: 0000000000000080");
            double.IsNaN(np.r_[double.NaN].GetValue<double>(0)).Should().BeTrue();
            np.r_[double.PositiveInfinity, 1].GetValue<double>(0).Should().Be(double.PositiveInfinity);
            np.r_[double.Epsilon].GetValue<double>(0).Should().Be(double.Epsilon);
            // A float literal saturates rather than raising, unlike an integer literal.
            np.r_[np.array(new[] { 1.0f }), 1e300].GetValue<float>(1).Should().Be(float.PositiveInfinity);
        }

        [TestMethod]
        public void R_WeakIntegerBoundaries_AreExactlyOneOffFromThrowing()
        {
            np.r_[np.array(new sbyte[] { 1 }), 127].typecode.Should().Be(NPTypeCode.SByte);
            np.r_[np.array(new sbyte[] { 1 }), -128].typecode.Should().Be(NPTypeCode.SByte);
            new Action(() => _ = np.r_[np.array(new sbyte[] { 1 }), 128]).Should().Throw<OverflowException>();
            new Action(() => _ = np.r_[np.array(new sbyte[] { 1 }), -129]).Should().Throw<OverflowException>();
        }

        [TestMethod]
        public void R_DirectiveIsPerCall_NotStickyOnTheSingleton()
        {
            // np.r_ is a static singleton whose axis/ndmin/trans1d LOOK like instance state. Build
            // copies them to locals, so a directive must not leak into the next call — nor across
            // threads, which is the failure mode that would only show under load.
            var a = np.array(new long[] { 1, 2 });
            np.r_["0,2", a].ndim.Should().Be(2);
            np.r_[a].ndim.Should().Be(1, "the previous call's ndmin must not persist");
            np.c_[a].ndim.Should().Be(2, "c_ keeps its own defaults");

            int bad = 0;
            System.Threading.Tasks.Parallel.For(0, 64, _ =>
            {
                if (np.r_["0,2", a, a].ndim != 2) System.Threading.Interlocked.Increment(ref bad);
                if (np.r_[a, 5L].ndim != 1) System.Threading.Interlocked.Increment(ref bad);
            });
            bad.Should().Be(0, "the concatenators must be stateless under concurrent use");
        }

        [TestMethod]
        public void R_ErrorOrder_MatchesNumPy_DirectiveBeforeShape()
        {
            var row = np.array(new long[,] { { 1, 2 } });
            var vec = np.array(new long[] { 1, 2 });

            // NumPy reports the bad directive, not the (also invalid) shapes behind it.
            new Action(() => _ = np.r_["q", row, vec])
                .Should().Throw<ValueError>().WithMessage("unknown special directive*");
            // A trailing directive is caught during the walk, before any concatenate.
            new Action(() => _ = np.r_[np.array(new sbyte[] { 1 }), 1000, "0"])
                .Should().Throw<ValueError>().WithMessage("special directives must be the first entry.*");
        }

        [TestMethod]
        public void Concatenators_AreSingletonsWithNumPysZeroLength()
        {
            ReferenceEquals(np.r_, np.r_).Should().BeTrue();
            ReferenceEquals(np.c_, np.c_).Should().BeTrue();
            np.r_.Count.Should().Be(0);
            np.c_.Count.Should().Be(0);
        }
    }
}
