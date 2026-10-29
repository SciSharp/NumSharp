using System;
using AwesomeAssertions;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     <see cref="NDArray.GetIndices"/> and <see cref="NDArray.SetIndices"/> as NumPy's <c>a[tuple(indices)]</c> and
    ///     <c>a[tuple(indices)] = values</c>: every index kind the indexer takes — a mask, an empty tuple (the whole array),
    ///     NumPy's too-many-indices error on a 0-d array. Expected outcomes probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Found by the replay of P3 batch 8's snapshot through NumPy (np-layer dtype plan): both members ran the
    ///     integer-array gather / scatter alone, so a mask raised "arrays used as indices must be of integer (or boolean)
    ///     type", an empty tuple raised <c>ArgumentException</c> ("Value cannot be an empty collection."), and a 0-d array was
    ///     read and written as a one-element 1-D array (NumPy: "too many indices").
    /// </remarks>
    [TestClass]
    public class get_set_indices_numpy_port_tests
    {
        /// <summary>A (3, 4) float64 array holding 0 to 11 in C order.</summary>
        private static NDArray A2() => np.arange(12.0).reshape(3, 4);

        /// <summary>The array's values in C order.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values as doubles.</returns>
        private static double[] Values(NDArray a) => a.flatten().ToArray<double>();

        /// <summary>A mask entry selects the rows it marks; storing through it writes those rows.</summary>
        [TestMethod]
        public void Mask_IsABooleanIndex()
        {
            var mask = np.array(new[] { true, false, true });
            var rows = A2().GetIndices(null, new[] { mask });
            rows.shape.Should().Equal(2L, 4L);
            Values(rows).Should().Equal(0, 1, 2, 3, 8, 9, 10, 11);

            var a = A2();
            a.SetIndices(np.array(0.0), new[] { mask });
            Values(a).Should().Equal(0, 0, 0, 0, 4, 5, 6, 7, 0, 0, 0, 0);
        }

        /// <summary>An empty tuple is <c>a[()]</c>: the whole array (a view for the getter, every element for the setter).</summary>
        [TestMethod]
        public void EmptyTuple_IsTheWholeArray()
        {
            var a = A2();
            var whole = a.GetIndices(null, new NDArray[0]);
            whole.shape.Should().Equal(3L, 4L);
            whole.flags.owndata.Should().BeFalse("a[()] of an array with dimensions is a view");
            Values(whole).Should().Equal(Values(A2()));

            a.SetIndices(np.array(7.0), new NDArray[0]);
            Values(a).Should().Equal(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7);
        }

        /// <summary>An index array into a 0-d array is NumPy's too-many-indices error, for both members.</summary>
        [TestMethod]
        public void IndexArrayIntoA0dArray_IsTooManyIndices()
        {
            const string tooMany = "too many indices for array: array is 0-dimensional, but 1 were indexed";
            Action get = () => np.array(5.0).GetIndices(null, new[] { np.array(new long[] { 0 }) });
            get.Should().Throw<IndexError>().WithMessage(tooMany);
            Action set = () => np.array(5.0).SetIndices(np.array(1.0), new[] { np.array(new long[] { 0 }) });
            set.Should().Throw<IndexError>().WithMessage(tooMany);
            Action two = () => np.array(5.0).GetIndices(null, new[] { np.array(new long[] { 0 }), np.array(new long[] { 0 }) });
            two.Should().Throw<IndexError>().WithMessage("too many indices for array: array is 0-dimensional, but 2 were indexed");
        }

        /// <summary>A null index list is the C# caller's error.</summary>
        [TestMethod]
        public void NullIndexList_IsArgumentNullException()
        {
            Action get = () => A2().GetIndices(null, null);
            get.Should().Throw<ArgumentNullException>();
            Action set = () => A2().SetIndices(np.array(1.0), null);
            set.Should().Throw<ArgumentNullException>();
        }
    }
}
