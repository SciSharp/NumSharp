using System;
using AwesomeAssertions;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     A Python bool in an index, as NumPy's <c>prepare_index</c> reads it: the 0-d bool array <c>PyArray_FROM_O(True)</c>
    ///     makes. Alone on a 0-d array it is the full mask (<c>HAS_BOOL</c>: the boolean assignment and its texts); alone on any
    ///     other array, or beside other entries, it is <c>HAS_0D_BOOL</c> — an ADVANCED index adding one axis of length 1
    ///     (True) or 0 (False), consuming none, broadcasting with the other advanced entries. A C# <see cref="bool"/> and a 0-d
    ///     bool <see cref="NDArray"/> are both that index. Expected outcomes probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Found by the replay of P3 batch 8's snapshot through NumPy (np-layer dtype plan): a lone C# bool stored into the
    ///     whole array for True (checked against <c>a.shape</c> with the view's text) and did nothing for False (no conversion,
    ///     no value check); beside other entries it was a basic index (a newaxis or an early return), so <c>a[True, True]</c>
    ///     raised a mask IndexError, <c>a[False, 0]</c> a NullReferenceException and <c>a[True, [0, 2]]</c> returned the wrong
    ///     elements; a 0-d bool ARRAY's getter and a string array's C# bool getter gave a C-ordered copy of an F-ordered
    ///     source.
    /// </remarks>
    [TestClass]
    public class bool_index_numpy_port_tests
    {
        /// <summary>A (3, 4) float64 array holding 0 to 11 in C order.</summary>
        private static NDArray A2() => np.arange(12.0).reshape(3, 4);

        /// <summary>The array's values in C order, for value checks.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values as doubles.</returns>
        private static double[] Values(NDArray a) => a.flatten().ToArray<double>();

        /// <summary>
        ///     A lone bool selects a COPY of the array under one new leading axis, laid out in the source's memory order (MapIter's
        ///     subspace): an F-ordered source gives an F-ordered result, for a C# bool, a 0-d bool array and a text array alike.
        /// </summary>
        [TestMethod]
        public void LoneBool_Get_IsACopyInTheSourcesMemoryOrder()
        {
            var c = A2()[true];
            c.shape.Should().Equal(1L, 3L, 4L);
            c.flags.owndata.Should().BeTrue("a bool is an advanced index: NumPy copies");
            A2()[false].shape.Should().Equal(0L, 3L, 4L);

            var f = np.asfortranarray(A2());
            var viaArray = f[np.array(true)];
            viaArray.flags.f_contiguous.Should().BeTrue("NumPy's a[np.array(True)] keeps an F source's order");
            viaArray.flags.c_contiguous.Should().BeFalse();
            Values(viaArray).Should().Equal(Values(A2()));

            var text = np.asfortranarray(np.array(new[] { "a", "bb", "c", "dd", "e", "ff" }).reshape(2, 3));
            var picked = text[true];
            picked.shape.Should().Equal(1L, 2L, 3L);
            picked.flags.f_contiguous.Should().BeTrue("a text array's a[True] keeps the source's order too");
        }

        /// <summary>
        ///     Stored into an array with dimensions, a lone bool is a fancy index of one new axis: the value is converted first
        ///     (an empty selection still raises the conversion's OverflowError) and checked against <c>(1|0,) + a.shape</c>
        ///     with MapIter's text.
        /// </summary>
        [TestMethod]
        public void LoneBool_Set_OnAnArrayWithDimensions_IsAFancyIndexOfOneNewAxis()
        {
            Action trueMismatch = () => A2()[true] = np.arange(5.0);
            trueMismatch.Should().Throw<ValueError>()
                .WithMessage("shape mismatch: value array of shape (5,) could not be broadcast to indexing result of shape (1,3,4)");
            Action falseMismatch = () => A2()[false] = np.arange(5.0);
            falseMismatch.Should().Throw<ValueError>()
                .WithMessage("shape mismatch: value array of shape (5,) could not be broadcast to indexing result of shape (0,3,4)");

            var stored = A2();
            stored[true] = np.arange(4.0);
            Values(stored).Should().Equal(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);
            var untouched = A2();
            untouched[false] = np.arange(4.0);
            Values(untouched).Should().Equal(Values(A2()));

            Action emptyOverflow = () => np.zeros(new Shape(3), np.uint8)[false] = -1;
            emptyOverflow.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");
        }

        /// <summary>A lone bool on a 0-d array is the full mask: NumPy's boolean assignment and its texts.</summary>
        [TestMethod]
        public void LoneBool_Set_OnA0dArray_IsTheBooleanAssignment()
        {
            Action five = () => np.array(5.0)[true] = np.arange(5.0);
            five.Should().Throw<ValueError>().WithMessage(
                "NumPy boolean array indexing assignment cannot assign 5 input values to the 1 output values where the mask is true");
            Action falseFive = () => np.array(5.0)[false] = np.arange(5.0);
            falseFive.Should().Throw<ValueError>().WithMessage(
                "NumPy boolean array indexing assignment cannot assign 5 input values to the 0 output values where the mask is true");
            Action twoDims = () => np.array(5.0)[true] = np.zeros(new Shape(2, 2));
            twoDims.Should().Throw<TypeError>().WithMessage(
                "NumPy boolean array indexing assignment requires a 0 or 1-dimensional input, input has 2 dimensions");

            var x = np.array(5.0);
            x[true] = np.zeros(new Shape(1));
            x.GetDouble().Should().Be(0.0);
        }

        /// <summary>Beside other entries a bool joins the advanced block: a copy, its axis of length 1 or 0 in place.</summary>
        [TestMethod]
        public void BoolBesideOtherEntries_Get_IsAZeroDBoolArray()
        {
            A2()[true, true].shape.Should().Equal(1L, 3L, 4L);
            A2()[true, false].shape.Should().Equal(0L, 3L, 4L);
            A2()[false, 0].shape.Should().Equal(0L, 4L);
            A2()["1:3", false].shape.Should().Equal(2L, 0L, 4L);

            var rows = A2()[true, new long[] { 0, 2 }];
            rows.shape.Should().Equal(2L, 4L);
            Values(rows).Should().Equal(0, 1, 2, 3, 8, 9, 10, 11);

            var first = A2()[true, 0];
            first.shape.Should().Equal(1L, 4L);
            first.flags.owndata.Should().BeTrue("NumPy's a[True, 0] is advanced indexing: a copy, not a view");
        }

        /// <summary>Stored beside other entries, a bool checks the value against the indexing result with MapIter's text.</summary>
        [TestMethod]
        public void BoolBesideOtherEntries_Set_ChecksTheValueAgainstTheIndexingResult()
        {
            Action trueZero = () => A2()[true, 0] = np.arange(5.0);
            trueZero.Should().Throw<ValueError>()
                .WithMessage("shape mismatch: value array of shape (5,) could not be broadcast to indexing result of shape (1,4)");
            Action falseZero = () => A2()[false, 0] = np.arange(5.0);
            falseZero.Should().Throw<ValueError>()
                .WithMessage("shape mismatch: value array of shape (5,) could not be broadcast to indexing result of shape (0,4)");
            Action sliceFalse = () => A2()["1:3", false] = np.arange(2.0);
            sliceFalse.Should().Throw<ValueError>()
                .WithMessage("shape mismatch: value array of shape (2,) could not be broadcast to indexing result of shape (2,0,4)");

            var both = A2();
            both[true, true] = np.arange(4.0);
            Values(both).Should().Equal(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);
            var row = A2();
            row[0, true] = np.zeros(new Shape(1, 1));
            Values(row).Should().Equal(0, 0, 0, 0, 4, 5, 6, 7, 8, 9, 10, 11);
        }
    }
}
