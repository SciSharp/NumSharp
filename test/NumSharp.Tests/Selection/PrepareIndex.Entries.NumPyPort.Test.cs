using System;
using AwesomeAssertions;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    ///     How NumPy's <c>prepare_index</c> (<c>mapping.c</c>) treats each index ENTRY as it walks the index, on the getter and
    ///     the setter: an integer scalar is an integer only on an array with axes (on a 0-d array it is the 0-d integer array
    ///     <c>PyArray_FROM_O</c> makes, which counts as an indexed axis), a scalar that is no integer is refused where it
    ///     stands with NumPy's general text, and an index array of a non-index dtype with its own text, both before the
    ///     too-many-indices count. Expected outcomes probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Found by the replay of P3 batch 8's snapshot through NumPy (np-layer dtype plan): NumSharp quoted the forms with
    ///     apostrophes and appended <c>(got 'Double')</c>; rounded a double index to an integer in a key of several entries
    ///     (<c>a[0.6, ":"]</c> returned row 1, <c>a[1.5, ":"]</c> reported index 2 out of bounds); refused an integer beside
    ///     another entry on a 0-d array as an invalid index; and on a 0-d string array counted the rank before it looked at
    ///     an entry's type.
    /// </remarks>
    [TestClass]
    public class prepare_index_entries_numpy_port_tests
    {
        private const string InvalidIndex =
            "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices";

        private const string NonIndexArray = "arrays used as indices must be of integer (or boolean) type";

        /// <summary>The getter and the setter (a 0-d int64 value) both raise <paramref name="text"/>, and the setter writes
        ///     nothing.</summary>
        /// <param name="make">Builds the array (called once per side, so the setter's copy is fresh).</param>
        /// <param name="key">The index tuple.</param>
        /// <param name="text">NumPy's IndexError text, verbatim.</param>
        private static void GetAndSetRaise(Func<NDArray> make, object[] key, string text)
        {
            var a = make();
            Action get = () => { var _ = a[key]; };
            get.Should().Throw<IndexError>().Where(e => e.Message == text);
            var b = make();
            var before = b.copy();
            Action set = () => b[key] = NDArray.Scalar(7L);
            set.Should().Throw<IndexError>().Where(e => e.Message == text);
            np.array_equal(b, before).Should().BeTrue("a refused index writes nothing");
        }

        /// <summary>
        ///     A double is no integer, whatever its value: alone and beside other entries, NumPy's general text (it was
        ///     rounded to an integer in a key of several entries: <c>a[0.6, ":"]</c> returned row 1).
        /// </summary>
        [TestMethod]
        public void DoubleEntry_IsRefusedWithNumPysText()
        {
            GetAndSetRaise(() => np.arange(6), new object[] { 1.5 }, InvalidIndex);
            GetAndSetRaise(() => np.arange(6).reshape(2, 3), new object[] { 0, 1.5 }, InvalidIndex);
            GetAndSetRaise(() => np.arange(6).reshape(2, 3), new object[] { 1.5, ":" }, InvalidIndex);
            GetAndSetRaise(() => np.arange(6).reshape(2, 3), new object[] { 0.6, ":" }, InvalidIndex);
            GetAndSetRaise(() => np.arange(6).reshape(2, 3), new object[] { 0, 1.0 }, InvalidIndex);
        }

        /// <summary>A float16 and a decimal entry are refused the same way (NumPy's <c>np.float16(1)</c>, a
        ///     <c>Decimal</c>).</summary>
        [TestMethod]
        public void Float16AndDecimalEntries_AreRefusedWithNumPysText()
        {
            GetAndSetRaise(() => np.arange(6), new object[] { (Half)2 }, InvalidIndex);
            GetAndSetRaise(() => np.arange(6).reshape(2, 3), new object[] { 0, (Half)1 }, InvalidIndex);
            GetAndSetRaise(() => np.arange(6).reshape(2, 3), new object[] { 0, 2m }, InvalidIndex);
        }

        /// <summary>
        ///     On a 0-d array an integer beside another entry is NumPy's 0-d integer array: it counts as an indexed axis, so the
        ///     index is "too many indices" (it was refused as an invalid index).
        /// </summary>
        [TestMethod]
        public void IntegerBesideAnotherEntry_OnA0dArray_IsTooManyIndices()
        {
            const string text = "too many indices for array: array is 0-dimensional, but 2 were indexed";
            GetAndSetRaise(() => np.array(5L), new object[] { (0, ":") }, text);
            GetAndSetRaise(() => np.array(5L), new object[] { 0, ":" }, text);
            GetAndSetRaise(() => np.array(5L), new object[] { 0, 0 }, text);
        }

        /// <summary>
        ///     A 0-d string array names a refused entry before it counts the rank, as NumPy's walk does (it reported "too many
        ///     indices" for all three), and counts an integer beside another entry.
        /// </summary>
        [TestMethod]
        public void StringRoute_ChecksEachEntryBeforeTheRank()
        {
            var t0 = new Func<NDArray>(() => np.array("ab"));
            Action floatArray = () => { var _ = t0()[np.array(new[] { 0.0 })]; };
            floatArray.Should().Throw<IndexError>().Where(e => e.Message == NonIndexArray);
            Action floatScalar = () => { var _ = t0()[1.5]; };
            floatScalar.Should().Throw<IndexError>().Where(e => e.Message == InvalidIndex);
            Action floatThenInt = () => { var _ = t0()[1.5, 0]; };
            floatThenInt.Should().Throw<IndexError>().Where(e => e.Message == InvalidIndex);
            Action tuple = () => { var _ = t0()[(0, ":")]; };
            tuple.Should().Throw<IndexError>().Where(e => e.Message == "too many indices for array: array is 0-dimensional, but 2 were indexed");

            // The setter validates the index before it converts the text value.
            var s = np.array("ab");
            Action set = () => s[np.array(new[] { 0.0 })] = "x";
            set.Should().Throw<IndexError>().Where(e => e.Message == NonIndexArray);
            s.ToString(true).Should().Be("array('ab', dtype='<U2')", "a refused index writes nothing");
        }
    }
}
