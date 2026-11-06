using System;
using System.Linq;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     Parity QA-30: the texts np.broadcast_to, np.broadcast_arrays and np.broadcast refuse with are NumPy 2.4.2's,
    ///     character for character (scratchpad probes p167_numpy.py / p170.cs, 30 cases, all equal). np.broadcast_to refuses
    ///     through <c>_broadcast_to</c>'s Python tests and then NpyIter (<c>np.nditer((array,), itershape=shape)</c>):
    ///     <c>convert_shape_to_string</c> shapes (a 1-tuple with its comma, no spaces) and TWO spaces before "and requested
    ///     shape"; np.broadcast_arrays (<c>_broadcast_shape</c>) and np.broadcast (<c>PyArray_Broadcast</c>) name the first
    ///     pair of operands that disagree, as Python tuple reprs. The exception type is the house
    ///     <see cref="IncorrectShapeException"/> of a broadcast refusal (NumPy's ValueError), the negative-dimension refusal
    ///     NumPy's <see cref="ValueError"/>.
    /// </summary>
    [TestClass]
    public class np_broadcast_RefusalTexts_Test
    {
        /// <summary>The head of NpyIter's broadcast_error text when op_axes are set (np.broadcast_to's iterator).</summary>
        private const string Remapped = "operands could not be broadcast together with remapped shapes [original->remapped]: ";

        /// <summary>The head of PyArray_Broadcast's mismatch text (two spaces after the first sentence).</summary>
        private const string Pair = "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between ";

        /// <summary>A fresh float64 zeros array of the given dimensions (the operands' values never matter here).</summary>
        /// <param name="dims">The dimensions; none gives a 0-d array.</param>
        /// <returns>The new array (the caller owns it).</returns>
        private static NDArray Z(params long[] dims) => np.zeros(new Shape(dims));

        /// <summary>Every np.broadcast_to refusal: the 0-d target test first, then a negative target dimension, then
        ///     NpyIter's more-axes text, then its remapped-shapes text for an axis that neither matches nor is 1.</summary>
        [TestMethod]
        public void BroadcastTo_RefusesWithNumPysTexts()
        {
            (long[] Source, long[] Target, string Message)[] rows =
            {
                (new long[] { 3 }, new long[] { 4 }, Remapped + "(3,)  and requested shape (4,)"),
                (new long[] { 2, 3 }, new long[] { 2, 4 }, Remapped + "(2,3)  and requested shape (2,4)"),
                (new long[] { 3, 1 }, new long[] { 2, 4 }, Remapped + "(3,1)  and requested shape (2,4)"),
                // NumPy lets the operand expand a target axis of 1, then refuses the expansion: the same text.
                (new long[] { 3 }, new long[] { 1 }, Remapped + "(3,)  and requested shape (1,)"),
                (new long[] { 3 }, new long[] { 2, 1 }, Remapped + "(3,)  and requested shape (2,1)"),
                (new long[] { 1, 2 }, new long[] { 2, 1 }, Remapped + "(1,2)  and requested shape (2,1)"),
                (new long[] { 0 }, new long[] { 3 }, Remapped + "(0,)  and requested shape (3,)"),
                (new long[] { 3 }, new long[] { 0 }, Remapped + "(3,)  and requested shape (0,)"),
                (new long[] { 2, 1, 3 }, new long[] { 2, 4, 4 }, Remapped + "(2,1,3)  and requested shape (2,4,4)"),
                // More axes than the target, even when the extra leading axes are 1 long.
                (new long[] { 2, 3 }, new long[] { 3 }, "input operand has more dimensions than allowed by the axis remapping"),
                (new long[] { 1, 3 }, new long[] { 3 }, "input operand has more dimensions than allowed by the axis remapping"),
                // _broadcast_to's own first test.
                (new long[] { 3 }, new long[0], "cannot broadcast a non-scalar to a scalar array"),
            };
            foreach (var (source, target, message) in rows)
            {
                using var src = Z(source);
                new Action(() => np.broadcast_to(src, Shape.Adopt(target)).Dispose())
                    .Should().Throw<IncorrectShapeException>().WithMessage(message, $"broadcast_to({string.Join(",", source)} -> {string.Join(",", target)})");
            }

            // A negative target dimension is _broadcast_to's ValueError, checked before NpyIter (QA-13).
            using var two3 = Z(2, 3);
            new Action(() => np.broadcast_to(two3, new Shape(-1L))).Should().Throw<ValueError>()
                .WithMessage("all elements of broadcast shape must be non-negative");

            // Shapes that do broadcast still give the stretched view.
            using var scalar = Z();
            using var stretched = np.broadcast_to(scalar, new Shape(3));
            stretched.shape.Should().Equal(3L);
            using var ones = Z(5, 1, 1, 1);
            using var wide = np.broadcast_to(ones, new Shape(5, 2, 1, 2));
            wide.shape.Should().Equal(5L, 2L, 1L, 2L);
        }

        /// <summary>np.broadcast_arrays and np.broadcast name the first clashing pair (PyArray_Broadcast's scan: axis by axis
        ///     from the left, operands in order within an axis), as Python tuple reprs.</summary>
        [TestMethod]
        public void BroadcastArraysAndBroadcast_NameThePairThatDisagrees()
        {
            using var three = Z(3);
            using var four = Z(4);
            using var two3 = Z(2, 3);
            using var row4 = Z(1, 4);
            using var row5 = Z(1, 5);
            using var col2 = Z(2, 1);
            using var col3 = Z(3, 1);
            using var scalar = Z();
            using var empty = Z(0);
            using var twoEmpty = Z(2, 0);

            (Func<object> Call, string Message)[] rows =
            {
                (() => np.broadcast_arrays(new[] { three, four }), Pair + "arg 0 with shape (3,) and arg 1 with shape (4,)."),
                (() => np.broadcast_arrays(new[] { two3, four, three }), Pair + "arg 0 with shape (2, 3) and arg 1 with shape (4,)."),
                // The leftmost axis that conflicts decides: axis 0 (args 2 and 3) before axis 1 (args 0 and 1).
                (() => np.broadcast_arrays(new[] { row4, row5, col2, col3 }), Pair + "arg 2 with shape (2, 1) and arg 3 with shape (3, 1)."),
                (() => np.broadcast_arrays(new[] { scalar, empty, three }), Pair + "arg 1 with shape (0,) and arg 2 with shape (3,)."),
                (() => np.broadcast_arrays(new[] { twoEmpty, three }), Pair + "arg 0 with shape (2, 0) and arg 1 with shape (3,)."),
                (() => np.broadcast_arrays(three, four), Pair + "arg 0 with shape (3,) and arg 1 with shape (4,)."),
                (() => np.broadcast(three, four), Pair + "arg 0 with shape (3,) and arg 1 with shape (4,)."),
                (() => np.broadcast(two3, four), Pair + "arg 0 with shape (2, 3) and arg 1 with shape (4,)."),
                (() => np.broadcast(row4, row5, col2, col3), Pair + "arg 2 with shape (2, 1) and arg 3 with shape (3, 1)."),
                (() => np.broadcast(twoEmpty, three), Pair + "arg 0 with shape (2, 0) and arg 1 with shape (3,)."),
            };
            foreach (var (call, message) in rows)
                new Action(() => call()).Should().Throw<IncorrectShapeException>().WithMessage(message);

            // One operand cannot disagree with itself.
            np.broadcast_arrays(new[] { three }).Single().shape.Should().Equal(3L);
        }

        /// <summary>
        ///     Past 64 operands np.broadcast_arrays follows <c>_broadcast_shape</c>'s groups: the first 64 operands, then each
        ///     further 63 with the shape found so far as argument 0, so a mismatch is numbered within its group. np.broadcast
        ///     (which NumPy caps at 64 operands and NumSharp does not) numbers every operand in one scan.
        /// </summary>
        [TestMethod]
        public void BroadcastArrays_PastSixtyFourOperands_NumbersWithinTheGroup()
        {
            // total operands of shape (4,), the one at index `at` of shape (5,).
            NDArray[] Ops(int at, int total)
                => Enumerable.Range(0, total).Select(i => i == at ? Z(5) : Z(4)).ToArray();

            new Action(() => np.broadcast_arrays(Ops(65, 66))).Should().Throw<IncorrectShapeException>()
                .WithMessage(Pair + "arg 0 with shape (4,) and arg 2 with shape (5,).");
            new Action(() => np.broadcast_arrays(Ops(63, 64))).Should().Throw<IncorrectShapeException>()
                .WithMessage(Pair + "arg 0 with shape (4,) and arg 63 with shape (5,).");
            new Action(() => np.broadcast_arrays(Ops(67, 70))).Should().Throw<IncorrectShapeException>()
                .WithMessage(Pair + "arg 0 with shape (4,) and arg 4 with shape (5,).");
            new Action(() => np.broadcast(Ops(63, 64))).Should().Throw<IncorrectShapeException>()
                .WithMessage(Pair + "arg 0 with shape (4,) and arg 63 with shape (5,).");
            new Action(() => np.broadcast(Ops(67, 70))).Should().Throw<IncorrectShapeException>()
                .WithMessage(Pair + "arg 0 with shape (4,) and arg 67 with shape (5,).");
        }
    }
}
