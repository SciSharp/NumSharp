using System;
using System.Linq;

namespace NumSharp.Tests.View
{
    /// <summary>
    ///     NumSharp has no <c>NPY_MAXDIMS</c>: an array may have more than NumPy's 64 dimensions (parity QA-29, kept by the
    ///     owner as a documented divergence). NumPy 2.4.2 refuses the 65th dimension wherever an array is made:
    ///     <c>np.zeros((1,) * 65)</c>, <c>np.ones</c>, <c>reshape</c>, <c>np.expand_dims(a64, 0)</c> and
    ///     <c>np.broadcast_to</c> raise <c>ValueError: maximum supported dimension for an ndarray is currently 64, found
    ///     65</c>; <c>a64[np.newaxis]</c> and a broadcast of a 64-d and a 65-d operand raise <c>IndexError: number of
    ///     dimensions must be within [0, 64], indexing result would have 65</c>; <c>np.array(1, ndmin=65)</c> raises
    ///     <c>ValueError: ndmin must be &lt;= ndmax (64)</c>. NumSharp builds every one of those arrays.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Such an array cannot cross to NumPy (the pythonnet bridge hands NumPy an array it cannot represent), and
    ///         printing one is the exception to the reach measured below: <c>ToString()</c> / <c>np.array2string</c>
    ///         recurse once per axis and overflow the stack past about 1,000 to 2,000 dimensions, which ends the process
    ///         (parity QA-50, open). No test here formats a high-rank array, a failing FluentAssertions message included:
    ///         every assertion reads a number.
    ///     </para>
    ///     <para>
    ///         The 200,000-dimension test pins the owner's stated reach: a <c>stackalloc long[ndim]</c> (1.6 MB) on any of
    ///         these paths would overflow a thread's stack there, so a regression ends the test host instead of passing.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class ShapeHighRankTests
    {
        /// <summary>
        ///     The 65-dimension calls NumPy refuses all succeed: <c>np.zeros</c> / <c>np.ones</c> of 65 dimensions,
        ///     <c>np.expand_dims</c> and <c>[np.newaxis]</c> of a 64-d array, <c>np.broadcast_to</c> to 65 dimensions, an
        ///     elementwise add of a 64-d and a 65-d operand, and a reshape to 65 dimensions.
        /// </summary>
        [TestMethod, Misaligned]
        public void SixtyFiveDimensions_AreBuilt_WhereNumPyRefuses()
        {
            var rank65 = Enumerable.Repeat(1L, 65).ToArray();
            rank65[0] = 3;
            using var a64 = np.ones(new Shape(Enumerable.Repeat(1L, 64).ToArray()));

            using var zeros = np.zeros(new Shape(rank65));
            using var expanded = np.expand_dims(a64, 0);
            using var newAxis = a64[np.newaxis];
            using var broadcast = np.broadcast_to(np.array(2.0), new Shape(rank65));
            using var sum = a64 + expanded;
            using var reshaped = np.arange(3.0).reshape(new Shape(rank65));

            zeros.ndim.Should().Be(65);
            zeros.size.Should().Be(3);
            expanded.ndim.Should().Be(65);
            newAxis.ndim.Should().Be(65);
            broadcast.ndim.Should().Be(65);
            broadcast.GetDouble(new long[65]).Should().Be(2.0);
            sum.ndim.Should().Be(65);
            sum.GetDouble(new long[65]).Should().Be(2.0);
            reshaped.ndim.Should().Be(65);
            reshaped.GetDouble(rank65.Select((d, i) => i == 0 ? 2L : 0L).ToArray()).Should().Be(2.0);
        }

        /// <summary>
        ///     The owner's stated reach (about 200,000 dimensions, measured to 300,000): a <c>(3, 1, 1, …)</c> array of
        ///     200,000 dimensions is made, expanded, given a new axis, summed, added to itself, indexed, reshaped back to 1-D
        ///     and broadcast to.
        /// </summary>
        [TestMethod, Misaligned]
        public void TwoHundredThousandDimensions_ComputeLikeAnyArray()
        {
            const int rank = 200_000;
            var dims = Enumerable.Repeat(1L, rank).ToArray();
            dims[0] = 3;
            var origin = new long[rank];

            using var ones = np.ones(new Shape(dims));
            using var expanded = np.expand_dims(ones, 0);
            using var newAxis = ones[np.newaxis];
            using var total = np.sum(ones);
            using var doubled = ones + ones;
            using var flat = ones.reshape(3);
            using var broadcast = np.broadcast_to(np.array(4.0), new Shape(dims));

            ones.ndim.Should().Be(rank);
            expanded.ndim.Should().Be(rank + 1);
            newAxis.ndim.Should().Be(rank + 1);
            total.GetDouble().Should().Be(3.0);
            doubled.ndim.Should().Be(rank);
            doubled.GetDouble(origin).Should().Be(2.0);
            ones.GetDouble(origin).Should().Be(1.0);
            flat.ndim.Should().Be(1);
            flat.size.Should().Be(3);
            broadcast.GetDouble(origin).Should().Be(4.0);
        }
    }
}
