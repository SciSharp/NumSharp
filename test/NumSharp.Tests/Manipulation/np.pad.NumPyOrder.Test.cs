using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     np.pad as NumPy 2.4.2 runs it, found when the np-layer dtype plan's P3 batch 7 moved its statements into
    ///     <c>NDArray.ArrayPad.cs</c> and replayed the snapshot (93,720 cases) against NumPy: the order of its checks and their
    ///     texts, <c>_as_pairs</c>' rules for the widths and the values, the result object of a zero-width pad and of a 0-d
    ///     input, the statistics of an integer or bool input and linspace's arithmetic in <c>'linear_ramp'</c>. Every
    ///     expectation is NumPy 2.4.2's output.
    /// </summary>
    [TestClass]
    public class np_pad_numpy_order_tests
    {
        private static NDArray A3 => np.array(new long[] { 1, 2, 4 });

        private static NDArray M => np.arange(6L).reshape(2, 3);

        /// <summary>The widths are read before the mode: a negative width wins over an unknown mode.</summary>
        [TestMethod]
        public void Widths_AreCheckedBeforeTheMode()
        {
            // np.pad([1, 2, 4], -1, None) -> ValueError: index can't contain negative values
            Action negative = () => np.pad(A3, -1, (string)null);
            negative.Should().ThrowExactly<ValueError>().WithMessage("index can't contain negative values");

            // np.pad([1, 2, 4], 1, None) / 'Constant' -> ValueError: mode '...' is not supported
            Action none = () => np.pad(A3, 1, (string)null);
            none.Should().ThrowExactly<ValueError>().WithMessage("mode 'None' is not supported");
            Action caps = () => np.pad(A3, 1, "Constant");
            caps.Should().ThrowExactly<ValueError>().WithMessage("mode 'Constant' is not supported");

            // The callable overload: NumPy's callable(None) is false, so None meets the same lookup.
            Action noFunc = () => np.pad(A3, 1, (np.PadFunc)null);
            noFunc.Should().ThrowExactly<ValueError>().WithMessage("mode 'None' is not supported");

            // np.pad([1, 2, 4], None) -> TypeError: `pad_width` must be of integral type.
            Action noWidth = () => np.pad(A3, (int[])null);
            noWidth.Should().ThrowExactly<TypeError>().WithMessage("`pad_width` must be of integral type.");
        }

        /// <summary>NumPy's keyword check names the keywords as a Python set.</summary>
        [TestMethod]
        public void UnsupportedKeywords_AreAPythonSet()
        {
            // np.pad([1, 2, 4], 1, 'edge', constant_values=1, stat_length=2)
            Action two = () => np.pad(A3, 1, "edge", constant_values: 1, stat_length: 2);
            two.Should().ThrowExactly<ValueError>()
                .WithMessage("unsupported keyword arguments for mode 'edge': {'constant_values', 'stat_length'}");

            // reflect_type is a keyword too: np.pad([1, 2, 4], 1, reflect_type='odd')
            Action reflect = () => np.pad(A3, 1, reflect_type: "odd");
            reflect.Should().ThrowExactly<ValueError>()
                .WithMessage("unsupported keyword arguments for mode 'constant': {'reflect_type'}");
        }

        /// <summary>NumPy reflects every reflect_type but 'odd' as 'even', None included.</summary>
        [TestMethod]
        public void ReflectType_AnyValueButOdd_IsEven()
        {
            // np.pad([1, 2, 4], 2, 'reflect', reflect_type='bogus') -> [4, 2, 1, 2, 4, 2, 1]; reflect_type=None too
            np.pad(A3, 2, "reflect", reflect_type: "bogus").ToArray<long>().Should().Equal(4L, 2L, 1L, 2L, 4L, 2L, 1L);
            np.pad(A3, 2, "reflect", reflect_type: null).ToArray<long>().Should().Equal(4L, 2L, 1L, 2L, 4L, 2L, 1L);
            // np.pad([1, 2, 4], 2, 'symmetric', reflect_type='odd') -> [0, 1, 1, 2, 4, 4, 6]
            np.pad(A3, 2, "symmetric", reflect_type: "odd").ToArray<long>().Should().Equal(0L, 1L, 1L, 2L, 4L, 4L, 6L);
        }

        /// <summary><c>_as_pairs</c> over the TRUE rank, with <c>np.broadcast_to</c>'s texts and <c>x.min()</c>'s.</summary>
        [TestMethod]
        public void Widths_FollowAsPairs()
        {
            // np.pad(np.zeros((2, 3, 4)), np.array([1, 2, 3], np.int32)): three values are no pair
            Action three = () => np.pad(np.zeros(new Shape(2, 3, 4)), new[] { 1, 2, 3 });
            three.Should().ThrowExactly<ValueError>().WithMessage(
                "operands could not be broadcast together with remapped shapes [original->remapped]: (3,)  and requested shape (3,2)");

            // np.pad(np.array(5), np.array([1, 2, 3], np.int32)): a 0-d array's (0, 2) table refuses them too
            Action zeroD = () => np.pad(np.array(5L), new[] { 1, 2, 3 });
            zeroD.Should().ThrowExactly<ValueError>().WithMessage(
                "operands could not be broadcast together with remapped shapes [original->remapped]: (3,)  and requested shape (0,2)");

            // np.pad([1, 2, 4], np.array([], np.int32)): x.min() of nothing
            Action empty = () => np.pad(A3, new int[0]);
            empty.Should().ThrowExactly<ValueError>().WithMessage("zero-size array to reduction operation minimum which has no identity");

            // np.pad(np.arange(6).reshape(2, 3), [[1], [2]]): one width per axis, for both sides
            var perAxis = np.pad(M, new int[,] { { 1 }, { 2 } });
            perAxis.shape.Should().BeEquivalentTo(new long[] { 4, 7 });
            perAxis["1:3, 2:5"].ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);

            // np.pad([1, 2, 4], [[1], [2]]): (2, 1) does not broadcast to (1, 2)
            Action column = () => np.pad(A3, new int[,] { { 1 }, { 2 } });
            column.Should().ThrowExactly<ValueError>().WithMessage(
                "operands could not be broadcast together with remapped shapes [original->remapped]: (2,1)  and requested shape (1,2)");

            // np.pad(np.arange(6).reshape(2, 3), [[1], [-2]]): the negative test runs before the broadcast
            Action negative = () => np.pad(M, new int[,] { { 1 }, { -2 } });
            negative.Should().ThrowExactly<ValueError>().WithMessage("index can't contain negative values");
        }

        /// <summary>A zero-width pad is <c>_pad_simple</c>'s array: an F-ordered input stays F-ordered.</summary>
        [TestMethod]
        public void ZeroWidths_KeepPadSimplesOrder()
        {
            // np.pad(np.asfortranarray(m), 0).flags.f_contiguous -> True; 'edge' too
            var f = np.asfortranarray(M);
            np.pad(f, 0).Shape.IsFContiguous.Should().BeTrue();
            np.pad(f, 0, "edge").Shape.IsFContiguous.Should().BeTrue();
            np.pad(f, 0).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
        }

        /// <summary>The callable runs for every vector, zero widths included, over <c>_pad_simple</c>'s array.</summary>
        [TestMethod]
        public void Callable_RunsForEveryVector_AndKeepsTheOrder()
        {
            // NumPy calls the function with axes [0, 0, 0, 1, 1] for a (2, 3) array padded by 0, and keeps F order.
            var axes = new List<int>();
            var r = np.pad(np.asfortranarray(M), 0, (v, w, axis, k) => axes.Add(axis));
            axes.Should().Equal(0, 0, 0, 1, 1);
            r.Shape.IsFContiguous.Should().BeTrue();
            r.Shape.IsContiguous.Should().BeFalse();
            r.ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
        }

        /// <summary>A 0-d input's value survives <c>_pad_simple</c>'s fill.</summary>
        [TestMethod]
        public void Callable_ZeroD_KeepsTheValue()
        {
            // np.pad(np.array(5), 2, lambda v, w, i, k: None) -> 5
            var r = np.pad(np.array(5L), 2, (v, w, i, k) => { });
            r.ndim.Should().Be(0);
            r.ToArray<long>().Should().Equal(5L);
        }

        /// <summary>
        ///     <c>constant_values</c> through <c>_as_pairs</c>: one or two values are NumPy SCALARS (cast into an unsigned
        ///     array), a larger table goes through <c>tolist()</c> - PYTHON scalars, range-checked into every integer array.
        /// </summary>
        [TestMethod]
        public void ConstantValues_FollowAsPairs()
        {
            // np.pad(np.zeros(2, np.uint8), 1, constant_values=np.array([[1, -1]], np.int32)) -> [1, 0, 0, 255]
            np.pad(np.zeros(new Shape(2), np.uint8), 1, constant_values: new int[,] { { 1, -1 } })
                .ToArray<byte>().Should().Equal((byte)1, (byte)0, (byte)0, (byte)255);

            // np.pad(np.zeros((2, 2), np.uint8), 1, constant_values=np.array([[1, -1], [2, -2]], np.int32))
            Action table = () => np.pad(np.zeros(new Shape(2, 2), np.uint8), 1, constant_values: new int[,] { { 1, -1 }, { 2, -2 } });
            table.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");

            // np.pad(np.arange(6).reshape(2, 3), 1, constant_values=np.array([[10, 20], [30, 40]]))
            np.pad(M, 1, constant_values: np.array(new long[,] { { 10, 20 }, { 30, 40 } })).ToArray<long>().Should().Equal(
                30L, 10L, 10L, 10L, 40L, 30L, 0L, 1L, 2L, 40L, 30L, 3L, 4L, 5L, 40L, 30L, 20L, 20L, 20L, 40L);

            // np.array([[10], [30]]): one value per axis
            np.pad(M, 1, constant_values: np.array(new long[,] { { 10 }, { 30 } })).ToArray<long>().Should().Equal(
                30L, 10L, 10L, 10L, 30L, 30L, 0L, 1L, 2L, 30L, 30L, 3L, 4L, 5L, 30L, 30L, 10L, 10L, 10L, 30L);

            // np.ones((1, 1, 1)): three dimensions skip the shortcuts and fail the broadcast
            Action threeD = () => np.pad(M, 1, constant_values: np.ones(new Shape(1, 1, 1)));
            threeD.Should().ThrowExactly<ValueError>().WithMessage("input operand has more dimensions than allowed by the axis remapping");

            // A Python str of tolist() quotes itself: invalid literal for int() with base 10: 'x'
            Action text = () => np.pad(M, 1, constant_values: new object[,] { { "1", "2" }, { "x", "4" } });
            text.Should().ThrowExactly<ValueError>().WithMessage("invalid literal for int() with base 10: 'x'");

            // A Python complex into a float array: float()'s TypeError
            Action complex = () => np.pad(M.astype(np.float64), 1, constant_values: new object[,] { { 1, 2 }, { 3, new Complex(0, 4) } });
            complex.Should().ThrowExactly<TypeError>().WithMessage("float() argument must be a string or a real number, not 'complex'");
        }

        /// <summary><c>stat_length</c> is <c>np.round(x).astype(np.intp)</c>: a real rounds half to even.</summary>
        [TestMethod]
        public void StatLength_RoundsToIntp()
        {
            // np.pad([1, 2, 4], 2, 'maximum', stat_length=2.5) -> [2, 2, 1, 2, 4, 4, 4] (2.5 rounds to 2)
            np.pad(A3, 2, "maximum", stat_length: 2.5).ToArray<long>().Should().Equal(2L, 2L, 1L, 2L, 4L, 4L, 4L);
            // np.pad(np.arange(8), 1, 'mean', stat_length=3.5) -> [2, 0, ..., 7, 6] (3.5 rounds to 4)
            np.pad(np.arange(8L), 1, "mean", stat_length: 3.5).ToArray<long>()
                .Should().Equal(2L, 0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 6L);
            // NaN becomes intp's minimum: index can't contain negative values
            Action nan = () => np.pad(A3, 2, "maximum", stat_length: double.NaN);
            nan.Should().ThrowExactly<ValueError>().WithMessage("index can't contain negative values");
        }

    }
}
