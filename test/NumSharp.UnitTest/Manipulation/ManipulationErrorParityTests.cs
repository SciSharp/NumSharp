using System;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    ///     Error parity for the shape-manipulation family, probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class ManipulationErrorParityTests
    {
        // =============================================================================
        // np.expand_dims — the axis bound was only half-guarded.
        // =============================================================================

        /// <remarks>
        ///     Three distinct defects, all closed by validating at the np.* boundary:
        ///     a positive overshoot leaked <see cref="ArgumentOutOfRangeException"/> naming a
        ///     parameter ('index') that expand_dims does not have — it came from Arrays.Insert
        ///     three frames down; a negative overshoot reported the NORMALIZED axis instead of the
        ///     one the caller passed; and a 0-d input skipped the check completely, so
        ///     expand_dims(scalar, 5) silently returned shape (1).
        /// </remarks>
        [TestMethod]
        public void ExpandDims_PositiveAxisOutOfBounds_ReportsNumPyText()
        {
            var a6 = np.arange(6);

            // np.expand_dims(np.arange(6), 2) -> AxisError: axis 2 is out of bounds for array of dimension 2
            // The bound is the OUTPUT ndim (input ndim + 1), which is why a 1-D input reports "2".
            new Action(() => np.expand_dims(a6, 2))
                .Should().ThrowExactly<ArgumentException>()
                .WithMessage("axis 2 is out of bounds for array of dimension 2");

            new Action(() => np.expand_dims(a6, 5))
                .Should().ThrowExactly<ArgumentException>()
                .WithMessage("axis 5 is out of bounds for array of dimension 2");

            new Action(() => np.expand_dims(a6, int.MaxValue))
                .Should().ThrowExactly<ArgumentException>()
                .WithMessage("axis 2147483647 is out of bounds for array of dimension 2");
        }

        [TestMethod]
        public void ExpandDims_PositiveAxisOutOfBounds_DoesNotLeakArraysInsert()
        {
            // The specific leak: "Specified argument was out of the range of valid values.
            // (Parameter 'index')" — a parameter expand_dims does not have.
            var ex = new Action(() => np.expand_dims(np.arange(6), 5))
                .Should().Throw<Exception>().Which;

            ex.Message.Should().NotContain("index");
            ex.Message.Should().NotContain("Specified argument was out of the range");
        }

        [TestMethod]
        public void ExpandDims_NegativeAxisOutOfBounds_ReportsTheAxisAsGiven()
        {
            // Was: "Effective axis -1 is less than 0" — the normalized value, not the caller's.
            new Action(() => np.expand_dims(np.arange(6), -3))
                .Should().ThrowExactly<ArgumentException>()
                .WithMessage("axis -3 is out of bounds for array of dimension 2");

            new Action(() => np.expand_dims(np.arange(6), -5))
                .Should().ThrowExactly<ArgumentException>()
                .WithMessage("axis -5 is out of bounds for array of dimension 2");

            new Action(() => np.expand_dims(np.arange(6), int.MinValue))
                .Should().ThrowExactly<ArgumentException>()
                .WithMessage("axis -2147483648 is out of bounds for array of dimension 2");
        }

        [TestMethod]
        public void ExpandDims_ZeroDimensional_IsBoundChecked_NotSilentlyAccepted()
        {
            var s0 = NDArray.Scalar(5.0d);

            // Legal: outNdim is 1, so only 0 and -1 exist.
            np.expand_dims(s0, 0).Shape.Dimensions.Should().Equal(1L);
            np.expand_dims(s0, -1).Shape.Dimensions.Should().Equal(1L);

            // Previously these silently returned shape (1).
            new Action(() => np.expand_dims(s0, 1))
                .Should().ThrowExactly<ArgumentException>()
                .WithMessage("axis 1 is out of bounds for array of dimension 1");

            new Action(() => np.expand_dims(s0, -2))
                .Should().ThrowExactly<ArgumentException>()
                .WithMessage("axis -2 is out of bounds for array of dimension 1");
        }

        [TestMethod]
        public void ExpandDims_LegalAxes_StillWork()
        {
            var a6 = np.arange(6);
            np.expand_dims(a6, 0).Shape.Dimensions.Should().Equal(1L, 6L);
            np.expand_dims(a6, 1).Shape.Dimensions.Should().Equal(6L, 1L);
            np.expand_dims(a6, -1).Shape.Dimensions.Should().Equal(6L, 1L);
            np.expand_dims(a6, -2).Shape.Dimensions.Should().Equal(1L, 6L);

            var a23 = np.arange(6).reshape(2L, 3L);
            np.expand_dims(a23, 0).Shape.Dimensions.Should().Equal(1L, 2L, 3L);
            np.expand_dims(a23, 2).Shape.Dimensions.Should().Equal(2L, 3L, 1L);
            np.expand_dims(a23, -1).Shape.Dimensions.Should().Equal(2L, 3L, 1L);

            // An empty-but-real array must still gain the axis.
            np.expand_dims(np.zeros(new Shape(0, 3)), 0).Shape.Dimensions.Should().Equal(1L, 0L, 3L);
        }

        [TestMethod]
        public void ExpandDims_BothOverloads_AgreeOnTheText()
        {
            // The int[] overload validated correctly all along; the scalar one now matches it.
            new Action(() => np.expand_dims(np.arange(6), new[] { 0, 5 }))
                .Should().ThrowExactly<ArgumentException>()
                .WithMessage("axis 5 is out of bounds for array of dimension 3");

            new Action(() => np.expand_dims(np.arange(6), new[] { 0, 0 }))
                .Should().ThrowExactly<ArgumentException>()
                .WithMessage("repeated axis");
        }

        // =============================================================================
        // ndarray.resize — the SECOND size-overflow site, with NumPy's other loop.
        // =============================================================================

        /// <remarks>
        ///     <c>PyArray_Resize_int</c> is deliberately NOT the same loop as the creation-time
        ///     check in <c>PyArray_NewFromDescr_int</c>, and all three differences are observable:
        ///     it BREAKS at the first zero dimension (so a negative one after a zero is never
        ///     reached), its text omits the "are" ("negative dimensions not allowed"), and an
        ///     overflowing product is a MemoryError rather than a ValueError.
        /// </remarks>
        [TestMethod]
        public void Resize_ZeroDimShortCircuits_SoANegativeAfterItIsAccepted()
        {
            // np.arange(6).resize((0,-1)) -> shape (0, -1), no error.
            var b = np.arange(6);
            b.resize(new Shape(new long[] { 0, -1 }));
            b.shape[0].Should().Be(0);
            b.shape[1].Should().Be(-1);

            // ...while the CREATION guard rejects the very same dimensions.
            new Action(() => np.zeros(new Shape(0, -1L)))
                .Should().ThrowExactly<ValueError>()
                .WithMessage("negative dimensions are not allowed");
        }

        [TestMethod]
        public void Resize_NegativeDimension_UsesResizesOwnWording()
        {
            // "not allowed", NOT the creation path's "are not allowed".
            foreach (var dims in new[] { new long[] { -1, 0 }, new long[] { -1 }, new long[] { 2, -1 } })
            {
                var b = np.arange(6);
                new Action(() => b.resize(new Shape(dims)))
                    .Should().ThrowExactly<IncorrectShapeException>()
                    .WithMessage("negative dimensions not allowed");
            }
        }

        [TestMethod]
        public void Resize_OverflowingProduct_IsMemoryError_NotASilentRelabel()
        {
            // np.arange(6).resize((2**62, 2**62)) -> MemoryError.
            // The product wraps to 0, so before the guard the array was relabelled (2**62, 2**62)
            // over a buffer that was never allocated for it — the same class of defect as
            // np.zeros((2**31, 2**31)), reached by a different route.
            var b = np.arange(6);
            new Action(() => b.resize(new Shape(new[] { 1L << 62, 1L << 62 })))
                .Should().ThrowExactly<OutOfMemoryException>();
        }

        [TestMethod]
        public void Resize_ZeroDimBeforeAHugeOne_IsAccepted()
        {
            // The break-on-zero means no overflow check ever runs here.
            var b = np.arange(6);
            b.resize(new Shape(new[] { 0L, 1L << 62 }));
            b.shape[0].Should().Be(0);
            b.shape[1].Should().Be(1L << 62);
            b.size.Should().Be(0);
        }

        [TestMethod]
        public void Resize_OrdinaryGrowAndShrink_StillCorrect()
        {
            // The allocate-by-count-then-relabel change must not disturb the data path.
            var grow = np.arange(3);
            grow.resize(new Shape(new[] { 5L }));
            grow.Shape.Dimensions.Should().Equal(5L);
            for (int i = 0; i < 3; i++) grow.GetInt64(i).Should().Be(i);
            grow.GetInt64(3).Should().Be(0, "the grown tail is zero-filled");
            grow.GetInt64(4).Should().Be(0);

            var shrink = np.arange(6);
            shrink.resize(new Shape(new[] { 3L }));
            shrink.Shape.Dimensions.Should().Equal(3L);
            for (int i = 0; i < 3; i++) shrink.GetInt64(i).Should().Be(i);

            var grow2d = np.arange(6);
            grow2d.resize(new Shape(new[] { 4L, 3L }));
            grow2d.Shape.Dimensions.Should().Equal(4L, 3L);
            grow2d.GetInt64(0, 0).Should().Be(0);
            grow2d.GetInt64(1, 2).Should().Be(5);
            grow2d.GetInt64(3, 2).Should().Be(0);

            // same-size resize is a pure in-place reshape
            var same = np.arange(6);
            same.resize(new Shape(new[] { 2L, 3L }));
            same.Shape.Dimensions.Should().Equal(2L, 3L);
            same.GetInt64(1, 2).Should().Be(5);
        }
    }
}
