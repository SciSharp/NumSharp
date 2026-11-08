using System;
using AwesomeAssertions;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     The order of a store's checks, NumPy's <c>array_assign_subscript</c>: <c>prepare_index</c> first (the index's
    ///     entries and structure, the integers' bounds), then the value's conversion into the array's dtype, then MapIter's
    ///     check of the index ARRAYS' values. Which error a store raises when several apply follows from it. Expected
    ///     outcomes probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Found by the replay of P3 batch 8's snapshot through NumPy (np-layer dtype plan): a lone invalid entry (a float, a
    ///     float16, a complex) was never validated before the value's conversion, so <c>np.zeros(3, uint8)[1.5] = -1</c>
    ///     raised the OverflowError; and an index of several entries scanned its index arrays' values before the conversion,
    ///     so <c>a[[0, 99], 0] = 1000</c> on int8 raised the IndexError.
    /// </remarks>
    [TestClass]
    public class setitem_order_numpy_port_tests
    {
        private const string InvalidIndex =
            "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices";

        /// <summary>A (3, 4) int8 array of zeros.</summary>
        private static NDArray I8() => np.zeros(new Shape(3, 4), np.int8);

        /// <summary>An invalid lone entry is the index's IndexError, before the value is converted.</summary>
        [TestMethod]
        public void InvalidLoneEntry_IsRaisedBeforeTheValueIsConverted()
        {
            Action dbl = () => np.zeros(new Shape(3), np.uint8)[1.5] = -1;
            dbl.Should().Throw<IndexError>().Where(e => e.Message == InvalidIndex);
            Action half = () => np.zeros(new Shape(3), np.uint8)[(Half)1] = -1;
            half.Should().Throw<IndexError>().Where(e => e.Message == InvalidIndex);
        }

        /// <summary>
        ///     The index arrays' values are checked after the value's conversion: an out-of-range literal wins over an
        ///     out-of-bounds index-array value, while an out-of-bounds integer (checked by <c>prepare_index</c>) wins over the
        ///     literal.
        /// </summary>
        [TestMethod]
        public void IndexArrayValues_AreCheckedAfterTheValuesConversion()
        {
            Action arrayAndInteger = () => I8()[new long[] { 0, 99 }, 0] = 1000;
            arrayAndInteger.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
            Action twoArrays = () => I8()[np.array(new long[] { 0, 99 }), np.array(new long[] { 0, 0 })] = 1000;
            twoArrays.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
            Action withSlice = () => I8()[new long[] { 0, 99 }, "1:3"] = 1000;
            withSlice.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");

            Action fits = () => I8()[new long[] { 0, 99 }, 0] = 7;
            fits.Should().Throw<IndexError>().WithMessage("index 99 is out of bounds for axis 0 with size 3");
            Action integer = () => I8()[0, 99] = 1000;
            integer.Should().Throw<IndexError>().WithMessage("index 99 is out of bounds for axis 1 with size 4");
        }
    }
}
