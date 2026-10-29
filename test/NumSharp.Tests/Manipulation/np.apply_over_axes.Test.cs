using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     Tests for <see cref="np.apply_over_axes"/>, verified against NumPy 2.4.2 output.
    ///     apply_over_axes threads func(current, axis) over each axis, re-inserting a dropped
    ///     dimension so the rank is preserved — equivalent to a keepdims reduction over a tuple of
    ///     axes.
    ///
    ///     NumPy reference: https://numpy.org/doc/stable/reference/generated/numpy.apply_over_axes.html
    ///     NumPy source: numpy/lib/_shape_base_impl.py
    /// </summary>
    [TestClass]
    public class np_apply_over_axes_Test
    {
        // arange(24).reshape(2,3,4)
        private static NDArray A() => np.arange(24).reshape(2, 3, 4);

        [TestMethod]
        public void SumOverTwoAxes_PreservesRank()
        {
            // np.apply_over_axes(np.sum, a, [0,2]) -> shape (1,3,1), values [60,92,124]
            var r = np.apply_over_axes((x, ax) => np.sum(x, ax), A(), new[] { 0, 2 });
            r.Should().BeShaped(1, 3, 1);
            r.Should().BeOfValues(60L, 92L, 124L);
        }

        [TestMethod]
        public void ScalarAxis_Overload()
        {
            // single axis -> shape (2,1,4)
            var r = np.apply_over_axes((x, ax) => np.sum(x, ax), A(), 1);
            r.Should().BeShaped(2, 1, 4);
            r.Should().BeOfValues(12L, 15L, 18L, 21L, 48L, 51L, 54L, 57L);
        }

        [TestMethod]
        public void NegativeAxes_NormalizeAgainstOriginalRank()
        {
            // [-3,-1] == [0,2] for a 3-D array -> (1,3,1) [60,92,124]
            var r = np.apply_over_axes((x, ax) => np.sum(x, ax), A(), new[] { -3, -1 });
            r.Should().BeShaped(1, 3, 1);
            r.Should().BeOfValues(60L, 92L, 124L);
        }

        [TestMethod]
        public void KeepdimsFunc_SameRankTakenAsIs()
        {
            // A func that already keeps the rank (keepdims=true) needs no re-expansion.
            var r = np.apply_over_axes((x, ax) => np.sum(x, ax, keepdims: true), A(), new[] { 0, 2 });
            r.Should().BeShaped(1, 3, 1);
            r.Should().BeOfValues(60L, 92L, 124L);
        }

        [TestMethod]
        public void WrongShapeFunc_Throws()
        {
            // A func that drops TWO dimensions cannot be fixed by a single re-insertion.
            ((Action)(() => np.apply_over_axes((x, ax) => np.sum(np.sum(x, 2), 1), A(), new[] { 1 })))
                .Should().ThrowExactly<ValueError>()
                .WithMessage("function is not returning an array of the correct shape");
        }

        [TestMethod]
        public void NullFunc_Throws()
        {
            ((Action)(() => np.apply_over_axes(null, A(), new[] { 0 })))
                .Should().Throw<ArgumentNullException>();
        }

        [TestMethod]
        public void NullArray_Throws()
        {
            ((Action)(() => np.apply_over_axes((x, ax) => x, (NDArray)null, new[] { 0 })))
                .Should().Throw<ArgumentNullException>();
        }
    }
}
