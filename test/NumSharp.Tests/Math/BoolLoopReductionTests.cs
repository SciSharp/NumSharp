using System;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     The bool LOOP of the add / multiply reductions and accumulations: a <c>dtype=bool</c> request runs NumPy's
    ///     <c>BOOL_add</c> (logical or) and <c>BOOL_multiply</c> (logical and) over the input's truth values, its cast to bool
    ///     (a NaN is true, a complex value with a nonzero imaginary part is true, a text is parsed by the str -> bool cast).
    ///     Every expected value is NumPy 2.4.2's output for the same call.
    /// </summary>
    /// <remarks>
    ///     Before the fix (strings phase 2, P2.7, found by the string oracle's section strout): <c>np.prod(x, dtype=bool)</c>
    ///     over every axis threw "Prod not supported for type Boolean", <c>np.cumprod(x, axis, dtype=bool)</c> threw a generic
    ///     constraint violation, <c>np.cumsum(x, axis, dtype=bool)</c> threw "CumSum conversion to Boolean is not supported",
    ///     and the flat <c>np.cumprod(x, dtype=bool)</c> computed in the input's dtype first, so a NaN (true in NumPy) came out
    ///     false and <c>1j</c> came out false.
    /// </remarks>
    [TestClass]
    public class BoolLoopReductionTests
    {
        [TestMethod]
        public void Cumprod_BoolLoop_IsLogicalAnd()
        {
            // np.cumprod([1, 2, 0, 3], dtype=bool) -> [True, True, False, False]
            var r = np.cumprod(np.array(new long[] { 1, 2, 0, 3 }), null, np.@bool);
            r.dtype.Should().Be(np.@bool);
            r.ToArray<bool>().Should().Equal(true, true, false, false);
        }

        [TestMethod]
        public void Cumsum_BoolLoop_IsLogicalOr()
        {
            // np.cumsum([0, 0, 1, 0], dtype=bool) -> [False, False, True, True]
            var r = np.cumsum(np.array(new long[] { 0, 0, 1, 0 }), null, np.@bool);
            r.dtype.Should().Be(np.@bool);
            r.ToArray<bool>().Should().Equal(false, false, true, true);
            // Along an axis: np.cumsum([[0, 1], [0, 0]], axis=0, dtype=bool) -> [[False, True], [False, True]]
            var m = np.cumsum(np.array(new long[,] { { 0, 1 }, { 0, 0 } }), 0, np.@bool);
            m.ToArray<bool>().Should().Equal(false, true, false, true);
        }

        [TestMethod]
        public void Cumprod_BoolLoop_AlongAnAxis()
        {
            // np.cumprod([[1, 0, 2], [3, 4, 0]], 0, dtype=bool) -> [[True, False, True], [True, False, False]]
            var m = np.array(new long[,] { { 1, 0, 2 }, { 3, 4, 0 } });
            var r = np.cumprod(m, 0, np.@bool);
            r.shape.Should().Equal(2L, 3L);
            r.ToArray<bool>().Should().Equal(true, false, true, true, false, false);
            // np.cumprod(m.T, 1, dtype=bool) -> [[True, True], [False, False], [True, False]]
            var t = np.cumprod(m.T, 1, np.@bool);
            t.shape.Should().Equal(3L, 2L);
            t.ToArray<bool>().Should().Equal(true, true, false, false, true, false);
        }

        [TestMethod]
        public void Prod_BoolLoop_IsAllTrue()
        {
            // np.prod([1, 2, 0, 3], dtype=bool) -> False (NumPy's scalar: NumSharp's 0-d array)
            var flat = np.prod(np.array(new long[] { 1, 2, 0, 3 }), null, np.@bool);
            flat.dtype.Should().Be(np.@bool);
            flat.ndim.Should().Be(0);
            ((bool)flat.GetAtIndex(0)).Should().BeFalse();
            var m = np.array(new long[,] { { 1, 0, 2 }, { 3, 4, 0 } });
            np.prod(m, 1, np.@bool).ToArray<bool>().Should().Equal(false, false);
            var kept = np.prod(m, 0, np.@bool, keepdims: true);
            kept.shape.Should().Equal(1L, 3L);
            kept.ToArray<bool>().Should().Equal(true, false, false);
        }

        [TestMethod]
        public void BoolLoop_EmptyInput_IsTheIdentity()
        {
            // np.prod(np.zeros(0, 'i8'), dtype=bool) -> True; np.cumprod(np.zeros(0, 'i8'), dtype=bool) -> []
            ((bool)np.prod(np.zeros(new Shape(0), np.int64), null, np.@bool).GetAtIndex(0)).Should().BeTrue();
            var c = np.cumprod(np.zeros(new Shape(0), np.int64), null, np.@bool);
            c.dtype.Should().Be(np.@bool);
            c.size.Should().Be(0);
        }

        [TestMethod]
        public void BoolLoop_ReadsNumPysTruthValues()
        {
            // A NaN is true: np.cumprod([1.0, nan, 0.0], dtype=bool) -> [True, True, False] (NumSharp gave False for the NaN).
            np.cumprod(np.array(new[] { 1.0, double.NaN, 0.0 }), null, np.@bool).ToArray<bool>().Should().Equal(true, true, false);
            // A nonzero imaginary part is true: np.cumprod([1j, 0j], dtype=bool) -> [True, False].
            np.cumprod(np.array(new[] { new Complex(0, 1), Complex.Zero }), null, np.@bool).ToArray<bool>()
                .Should().Equal(true, false);
        }

        [TestMethod]
        public void BoolLoop_OverText_ParsesThenReduces()
        {
            // np.prod(['3', '1', '2'], dtype=bool) -> True; np.prod(['3', '', '2'], dtype=bool) -> False (the empty text is
            // false).
            ((bool)np.prod(np.array(new[] { "3", "1", "2" }), null, np.@bool).GetAtIndex(0)).Should().BeTrue();
            ((bool)np.prod(np.array(new[] { "3", "", "2" }), null, np.@bool).GetAtIndex(0)).Should().BeFalse();
            // np.cumprod of a transposed text array along axis 1, dtype=bool: every element true.
            var t = np.array(new[] { "3", "1", "2", "4", "2", "5", "1", "3", "2", "2", "1", "4" }).reshape(4, 3).T;
            var r = np.cumprod(t, 1, np.@bool);
            r.shape.Should().Equal(3L, 4L);
            r.ToArray<bool>().Should().OnlyContain(v => v);
        }
    }
}
