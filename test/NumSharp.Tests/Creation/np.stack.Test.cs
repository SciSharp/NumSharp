using System;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     <c>np.stack</c> as NumPy 2.4.2's statements (<c>numpy/_core/shape_base.py</c>): every shape is compared first,
    ///     then each array gets the new axis and the expanded arrays are concatenated. A 0-d operand expands to <c>(1,)</c>,
    ///     so <c>n</c> scalars stack to <c>(n,)</c>; the old implementation ran the operands through <c>atleast_1d</c> first
    ///     and returned <c>(n, 1)</c>. The expanded operands go into an array of their own, so the caller's array of
    ///     operands is never written. Expected values probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class NpStackTest
    {
        [TestMethod]
        public void Stack_Of0dArrays_IsOneDimensional()
        {
            // np.stack([np.array(5), np.array(6), np.array(7)]) -> array([5, 6, 7]), shape (3,), OWNDATA.
            var r = np.stack(new[] { np.array(5L), np.array(6L), np.array(7L) });
            r.shape.Should().Equal(new long[] { 3 });
            r.dtype.Should().Be(np.int64);
            r.ToArray<long>().Should().Equal(5L, 6L, 7L);
            r.flags.owndata.Should().BeTrue();
            // axis=-1 is the same single axis; axis=1 is outside the 1-D result. NumPy raises AxisError; np.expand_dims's
            // house type is ArgumentException with NumPy's text.
            np.stack(new[] { np.array(5L), np.array(6L) }, -1).ToArray<long>().Should().Equal(5L, 6L);
            Action outside = () => np.stack(new[] { np.array(5L), np.array(6L) }, 1);
            outside.Should().Throw<ArgumentException>().WithMessage("axis 1 is out of bounds for array of dimension 1*");
        }

        [TestMethod]
        public void Stack_Of1dArrays_AlongTheNewLastAxis()
        {
            // np.stack([[1, 2], [3, 4]], axis=1) -> [[1, 3], [2, 4]].
            var r = np.stack(new[] { np.array(new long[] { 1, 2 }), np.array(new long[] { 3, 4 }) }, 1);
            r.shape.Should().Equal(new long[] { 2, 2 });
            r.ToArray<long>().Should().Equal(1L, 3L, 2L, 4L);
        }

        [TestMethod]
        public void Stack_ComparesEveryShapeFirst()
        {
            // np.stack([np.zeros(2), np.zeros(3)]) -> ValueError: all input arrays must have the same shape
            // (house type InvalidOperationException), before the axis is looked at.
            Action act = () => np.stack(new[] { np.zeros(new Shape(2L)), np.zeros(new Shape(3L)) }, 9);
            act.Should().Throw<InvalidOperationException>().WithMessage("all input arrays must have the same shape");
        }

        /// <summary>The expanded operands are built in a separate array: writing them back into the caller's
        /// <c>NDArray[]</c> (the old loop's pattern, safe only behind <c>atleast_1d</c>'s copy) would leave the caller
        /// holding expanded views.</summary>
        [TestMethod]
        public void Stack_LeavesTheCallersArrayOfOperandsAlone()
        {
            var first = np.array(5L);
            var operands = new[] { first, np.array(6L) };
            np.stack(operands);
            ReferenceEquals(operands[0], first).Should().BeTrue();
            operands[0].ndim.Should().Be(0);
            operands[1].ndim.Should().Be(0);
        }
    }
}
