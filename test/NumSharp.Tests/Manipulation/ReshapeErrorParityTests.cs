using System;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     Error parity for <c>reshape</c> against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Every expectation here was probed against NumPy 2.4.2 and is reproduced verbatim.
    ///     NumSharp keeps its long-standing <see cref="IncorrectShapeException"/> type where NumPy
    ///     raises <c>ValueError</c> (the house convention across the shape APIs) — the MESSAGE is
    ///     what must match character for character.
    ///
    ///     The two bugs these tests pin:
    ///     <list type="bullet">
    ///     <item>A degenerate known dimension made the <c>-1</c> inference divide by zero, so
    ///     <c>np.zeros((0,3)).reshape(-1,0)</c> escaped a raw <see cref="DivideByZeroException"/>
    ///     from inside <c>Shape</c>.</item>
    ///     <item>Only <c>-1</c> counted as "unknown", so a SECOND negative rode through as a
    ///     literal dimension and <c>reshape(-1,-2)</c> silently produced the shape
    ///     <c>(-3,-2)</c> — a negative-extent array rather than an error.</item>
    ///     </list>
    /// </remarks>
    [TestClass]
    public class ReshapeErrorParityTests
    {
        // ---------------------------------------------------------------------------------
        // The DivideByZeroException class: a zero among the KNOWN dims makes the divisor 0.
        // ---------------------------------------------------------------------------------

        [TestMethod]
        public void Reshape_UnknownDim_WithZeroKnownDim_EmptyArray_ReportsNumPyText()
        {
            // np.zeros((0,3)).reshape(-1,0) -> ValueError: cannot reshape array of size 0 into shape (0)
            // Note the leading -1 is DROPPED from NumPy's rendering, so it reads "(0)" not "(0,)".
            new Action(() => np.zeros(new Shape(0, 3)).reshape(-1L, 0L))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 0 into shape (0)");
        }

        [TestMethod]
        public void Reshape_UnknownDim_WithZeroKnownDim_TrailingUnknown_RendersNewaxis()
        {
            // np.zeros((0,3)).reshape(0,-1) -> ValueError: cannot reshape array of size 0 into shape (0,newaxis)
            new Action(() => np.zeros(new Shape(0, 3)).reshape(0L, -1L))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 0 into shape (0,newaxis)");
        }

        [TestMethod]
        public void Reshape_UnknownDim_WithZeroKnownDim_NonEmptyArray_ReportsNumPyText()
        {
            // np.zeros((2,3)).reshape(-1,0) -> ValueError: cannot reshape array of size 6 into shape (0)
            new Action(() => np.zeros(new Shape(2, 3)).reshape(-1L, 0L))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 6 into shape (0)");
        }

        [TestMethod]
        public void Reshape_UnknownDim_WithZeroKnownDim_ArangeSource_ReportsNumPyText()
        {
            // np.arange(0).reshape(-1,0) -> ValueError: cannot reshape array of size 0 into shape (0)
            new Action(() => np.arange(0).reshape(-1L, 0L))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 0 into shape (0)");
        }

        [TestMethod]
        public void Reshape_UnknownDim_InteriorZero_DropsLeadingUnknownFromText()
        {
            // np.zeros(12).reshape(-1,0,3) -> ValueError: cannot reshape array of size 12 into shape (0,3)
            new Action(() => np.zeros(new Shape(12)).reshape(-1L, 0L, 3L))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 12 into shape (0,3)");
        }

        [TestMethod]
        public void Reshape_UnknownDim_ZeroKnownDim_DoesNotThrowDivideByZero()
        {
            // The mechanical triage rule: a raw framework exception at the API boundary is a finding.
            foreach (var act in new Action[]
            {
                () => np.zeros(new Shape(0, 3)).reshape(-1L, 0L),
                () => np.zeros(new Shape(0, 3)).reshape(0L, -1L),
                () => np.arange(0).reshape(-1L, 0L),
                () => np.arange(0).reshape(0L, -1L),
                () => np.zeros(new Shape(0)).reshape(-1L, 0L, 2L),
                () => np.zeros(new Shape(6)).reshape(-1L, 0L),
            })
            {
                new Action(act).Should().NotThrow<DivideByZeroException>();
            }
        }

        // ---------------------------------------------------------------------------------
        // ANY negative dim is "unknown" — not just -1. This is what stops the silent
        // negative-extent shape.
        // ---------------------------------------------------------------------------------

        [TestMethod]
        public void Reshape_TwoNegativeDims_Throws_InsteadOfProducingNegativeShape()
        {
            // np.zeros(6).reshape(-1,-2) -> ValueError: can only specify one unknown dimension
            // NumSharp used to return the shape (-3,-2) here.
            new Action(() => np.zeros(new Shape(6)).reshape(-1L, -2L))
                .Should().ThrowExactly<ValueError>()
                .WithMessage("can only specify one unknown dimension");
        }

        [TestMethod]
        public void Reshape_TwoUnknownDims_ReportsNumPyText()
        {
            new Action(() => np.zeros(new Shape(6)).reshape(-1L, -1L))
                .Should().ThrowExactly<ValueError>()
                .WithMessage("can only specify one unknown dimension");

            // Fires during the scan, so it beats the size check even on an empty array.
            new Action(() => np.zeros(new Shape(0)).reshape(-1L, -1L))
                .Should().ThrowExactly<ValueError>()
                .WithMessage("can only specify one unknown dimension");

            new Action(() => np.zeros(new Shape(6)).reshape(-1L, -3L))
                .Should().ThrowExactly<ValueError>()
                .WithMessage("can only specify one unknown dimension");
        }

        [TestMethod]
        public void Reshape_SingleNegativeDim_IsInferred_WhateverItsValue()
        {
            // NumPy tests `dimensions[i] < 0`, not `== -1`:
            //   np.zeros(6).reshape(-3)     -> (6,)
            //   np.zeros(6).reshape(3,-5)   -> (3, 2)
            //   np.zeros(6).reshape(-2**63) -> (6,)
            np.zeros(new Shape(6)).reshape(-3L).Shape.Dimensions.Should().Equal(6L);
            np.zeros(new Shape(6)).reshape(3L, -5L).Shape.Dimensions.Should().Equal(3L, 2L);
            np.zeros(new Shape(6)).reshape(long.MinValue).Shape.Dimensions.Should().Equal(6L);
            np.zeros(new Shape(6)).reshape(2L, 3L, -1L).Shape.Dimensions.Should().Equal(2L, 3L, 1L);
        }

        [TestMethod]
        public void Reshape_NeverProducesANegativeDimension()
        {
            // Whatever the spelling, no reshape may hand back an array with a negative extent.
            foreach (var act in new Func<NDArray>[]
            {
                () => np.zeros(new Shape(6)).reshape(-1L, -2L),
                () => np.zeros(new Shape(6)).reshape(-2L),
                () => np.zeros(new Shape(6)).reshape(3L, -5L),
                () => np.zeros(new Shape(6)).reshape(long.MinValue),
                () => np.zeros(new Shape(0)).reshape(-1L, -1L),
            })
            {
                try
                {
                    var dims = act().Shape.Dimensions;
                    dims.Should().OnlyContain(d => d >= 0, "a reshape result must not have a negative extent");
                }
                catch (Exception e) when (e is IncorrectShapeException || e is ValueError)
                {
                    // Refusing is the other acceptable outcome.
                }
            }
        }

        // ---------------------------------------------------------------------------------
        // The plain size-mismatch family now carries NumPy's single verbatim text.
        // ---------------------------------------------------------------------------------

        [TestMethod]
        public void Reshape_SizeMismatch_ReportsNumPyText()
        {
            new Action(() => np.zeros(new Shape(6)).reshape(4L, 2L))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 6 into shape (4,2)");

            new Action(() => np.zeros(new Shape(6)).reshape(2L, 2L, 2L))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 6 into shape (2,2,2)");

            new Action(() => np.zeros(new Shape(6)).reshape(4L, -1L))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 6 into shape (4,newaxis)");
        }

        [TestMethod]
        public void Reshape_ToZeroSize_FromNonEmpty_ReportsNumPyText()
        {
            // A one-element newshape closes with ",)" so it reads as a Python 1-tuple.
            new Action(() => np.zeros(new Shape(6)).reshape(0L))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 6 into shape (0,)");

            new Action(() => np.zeros(new Shape(6)).reshape(0L, 0L))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 6 into shape (0,0)");

            new Action(() => np.zeros(new Shape(0)).reshape(1L))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 0 into shape (1,)");
        }

        [TestMethod]
        public void Reshape_ToScalarShape_SizeMismatch_ReportsNumPyText()
        {
            // np.zeros(3).reshape(()) -> ValueError: cannot reshape array of size 3 into shape ()
            new Action(() => np.zeros(new Shape(3)).reshape(new Shape()))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 3 into shape ()");
        }

        [TestMethod]
        public void Reshape_KnownDimsOverflowElementCounter_ReportsSizeMismatch()
        {
            // np.zeros(6).reshape(2**62, 2**62) -> ValueError: cannot reshape array of size 6
            //   into shape (4611686018427387904,4611686018427387904)
            // The product wraps in 64 bits; NumPy reports a size mismatch rather than letting the
            // wrapped (and much smaller) product look like a plausible size.
            new Action(() => np.zeros(new Shape(6)).reshape(1L << 62, 1L << 62))
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 6 into shape (4611686018427387904,4611686018427387904)");
        }

        // ---------------------------------------------------------------------------------
        // Shapes that must still SUCCEED — the fix must not turn legal reshapes into errors.
        // ---------------------------------------------------------------------------------

        [TestMethod]
        public void Reshape_LegalZeroSizeReshapes_StillSucceed()
        {
            np.zeros(new Shape(0, 3)).reshape(3L, 0L).Shape.Dimensions.Should().Equal(3L, 0L);
            np.zeros(new Shape(0, 3)).reshape(-1L).Shape.Dimensions.Should().Equal(0L);
            np.zeros(new Shape(0, 3)).reshape(-1L, 3L).Shape.Dimensions.Should().Equal(0L, 3L);
            np.zeros(new Shape(0, 3)).reshape(-1L, 1L).Shape.Dimensions.Should().Equal(0L, 1L);
            np.zeros(new Shape(0, 3)).reshape(0L, 0L).Shape.Dimensions.Should().Equal(0L, 0L);
            np.zeros(new Shape(0)).reshape(0L).Shape.Dimensions.Should().Equal(0L);
        }

        [TestMethod]
        public void ConvertShapeToString_MatchesNumPyRendering()
        {
            // Direct pins on the renderer, including the two spellings that are easy to conflate.
            Shape.ConvertShapeToString(Array.Empty<long>()).Should().Be("()");
            Shape.ConvertShapeToString(new[] { -1L }).Should().Be("()");
            Shape.ConvertShapeToString(new[] { -1L, -1L }).Should().Be("()");
            Shape.ConvertShapeToString(new[] { 0L }).Should().Be("(0,)");
            Shape.ConvertShapeToString(new[] { -1L, 0L }).Should().Be("(0)");
            Shape.ConvertShapeToString(new[] { 0L, -1L }).Should().Be("(0,newaxis)");
            Shape.ConvertShapeToString(new[] { 4L, 2L }).Should().Be("(4,2)");
            Shape.ConvertShapeToString(new[] { -1L, 2L, 0L }).Should().Be("(2,0)");
            Shape.ConvertShapeToString(new[] { 2L, 0L, -1L }).Should().Be("(2,0,newaxis)");
            Shape.ConvertShapeToString(new[] { -1L, 0L, 3L }).Should().Be("(0,3)");
        }
    }
}
