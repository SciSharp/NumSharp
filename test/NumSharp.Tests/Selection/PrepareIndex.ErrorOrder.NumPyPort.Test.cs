using System;
using AwesomeAssertions;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     The order of an advanced index's errors, NumPy's passes: <c>prepare_index</c>'s boolean-shape check first, then the
    ///     integers' bounds (<c>get_view_from_index</c>), then the index arrays' broadcast (<c>mapiter_fill_info</c>, a text
    ///     whose every shape is followed by a space), then the index arrays' values (<c>PyArray_MapIterCheckIndices</c>).
    ///     Expected outcomes probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Found by a 6,000-key differential of the getter and setter against NumPy (P3 batch 8, np-layer dtype plan): the
    ///     checks ran in one pass in the tuple's order, so whichever entry came first decided the error, and the broadcast
    ///     text had no trailing space.
    /// </remarks>
    [TestClass]
    public class prepare_index_error_order_numpy_port_tests
    {
        private static readonly NDArray A2 = np.zeros(new Shape(3, 4));
        private static readonly NDArray A3 = np.zeros(new Shape(2, 3, 4));

        /// <summary>The boolean-shape check comes before an index array's out-of-bounds value.</summary>
        [TestMethod]
        public void MaskShape_BeforeIndexArrayValues()
        {
            Action a = () => _ = A2[new long[] { 0, 9 }, np.array(new[] { true, false })];
            a.Should().Throw<IndexError>().WithMessage(
                "boolean index did not match indexed array along axis 1; size of axis is 4 but size of corresponding boolean axis is 2");
            Action b = () => _ = A2[new long[] { 9 }, np.array(new[] { true, false, true, true })];
            b.Should().Throw<IndexError>().WithMessage("index 9 is out of bounds for axis 0 with size 3");
        }

        /// <summary>An integer's bounds come before the index arrays' broadcast and their values.</summary>
        [TestMethod]
        public void IntegerBounds_BeforeTheBroadcastAndTheValues()
        {
            Action broadcast = () => _ = A3[new long[] { 0, 1 }, new long[] { 0, 0, 0 }, 99];
            broadcast.Should().Throw<IndexError>().WithMessage("index 99 is out of bounds for axis 2 with size 4");
            Action values = () => _ = A3[new long[] { 0, 9 }, 0, 99];
            values.Should().Throw<IndexError>().WithMessage("index 99 is out of bounds for axis 2 with size 4");
            Action first = () => _ = A3[99, new long[] { 0, 0, 0 }, new long[] { 0, 1 }];
            first.Should().Throw<IndexError>().WithMessage("index 99 is out of bounds for axis 0 with size 2");
        }

        /// <summary>The broadcast comes before the values, and its text ends with a space after the last shape.</summary>
        [TestMethod]
        public void Broadcast_BeforeTheValues_WithNumPysText()
        {
            Action a = () => _ = A3[new long[] { 0, 9 }, new long[] { 0, 0, 0 }];
            a.Should().Throw<IndexError>().Where(e => e.Message ==
                "shape mismatch: indexing arrays could not be broadcast together with shapes (2,) (3,) ");
            Action withSlice = () => _ = A3[new long[] { 0, 1 }, new long[] { 0, 0, 0 }, Slice.All];
            withSlice.Should().Throw<IndexError>().Where(e => e.Message ==
                "shape mismatch: indexing arrays could not be broadcast together with shapes (2,) (3,) ");
        }
    }
}
