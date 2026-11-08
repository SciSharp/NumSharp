using System;
using System.Linq;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     np.isin as NumPy 2.4.2's statements (<c>numpy/lib/_arraysetops_impl.py</c>, <c>isin</c> / <c>_isin</c>): the table
    ///     method, the loop path and the sort path, each with NumPy's answer; every expected value and text was probed
    ///     against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class np_isin_NumPyPort_Test
    {
        /// <summary>
        ///     NumSharp's char stands for NumPy's uint16 (the proxy np.searchsorted's <c>StringDType</c> route reads too):
        ///     <c>kind='table'</c> takes a char array as the integer array it is in NumPy, and a char set read by the
        ///     <c>StringDType</c> route yields numbers, which never equal a text. NumSharp refused the table method for char,
        ///     and read a char set's elements as one-character strs, so <c>'a'</c> was found in <c>['a', 'b']</c>.
        /// </summary>
        [TestMethod]
        public void Char_IsNumPysUint16()
        {
            var chars = np.array(new[] { 'a', 'b' });
            // >>> np.isin(np.array([97, 98], np.uint16), np.array([97], np.uint16), kind='table')  array([ True, False])
            np.isin(chars, np.array(new[] { 'a' }), kind: "table").array_equal(np.array(new[] { true, false }))
                .Should().BeTrue();

            var text = np.array(new[] { "a", "zz" }).astype(np.dtype("T"));
            // >>> T = np.dtypes.StringDType()
            // >>> np.isin(np.array(['a', 'zz'], T), np.array([97, 98], np.uint16))  array([False, False])
            np.isin(text, chars).array_equal(np.array(new[] { false, false })).Should().BeTrue();
            // >>> np.isin(np.array(['a', 'zz'], T), np.array([97, 98], np.uint16), invert=True)  array([ True,  True])
            np.isin(text, chars, invert: true).array_equal(np.array(new[] { true, true })).Should().BeTrue();
            // >>> np.isin(np.array([97, 98], np.uint16), np.array(['a'], T))  array([False, False]); invert: [True, True]
            var a = np.array(new[] { "a" }).astype(np.dtype("T"));
            np.isin(chars, a).array_equal(np.array(new[] { false, false })).Should().BeTrue();
            np.isin(chars, a, invert: true).array_equal(np.array(new[] { true, true })).Should().BeTrue();

            // A NaN-like NA is False under == and under !=, so against a set without strings it is True under invert.
            // >>> x = np.array(['a', np.nan], np.dtypes.StringDType(na_object=np.nan))
            // >>> np.isin(x, np.array([97], np.uint16)), np.isin(x, np.array([97], np.uint16), invert=True)
            // (array([False, False]), array([ True,  True]))
            var x = np.array(new object[] { "a", double.NaN }, np.dtypes.StringDType(na_object: double.NaN));
            np.isin(x, np.array(new[] { 'a' })).array_equal(np.array(new[] { false, false })).Should().BeTrue();
            np.isin(x, np.array(new[] { 'a' }), invert: true).array_equal(np.array(new[] { true, true })).Should().BeTrue();
        }

        /// <summary>
        ///     <c>kind='table'</c> allocates <c>np.zeros(ar2_range + 1, dtype=bool)</c> (<c>np.ones</c> under invert) once the
        ///     range is safe for the set's dtype, so a range of <c>2**63 - 1</c> or more meets that call's shape conversion:
        ///     <c>ValueError: Maximum allowed dimension exceeded</c>, for an empty element too. NumSharp's membership needs
        ///     no table and answered. Without <c>kind='table'</c> NumPy's loop or sort path answers.
        /// </summary>
        [TestMethod]
        public void TableOverARangePastIntp_RaisesNumPysDimensionError()
        {
            var one = np.array(new long[] { 1 });
            // >>> np.isin(np.array([1], np.int64), np.array([0, 2**63 - 1], np.int64), kind='table')
            // ValueError: Maximum allowed dimension exceeded
            Action intMax = () => np.isin(one, np.array(new[] { 0L, long.MaxValue }), kind: "table");
            intMax.Should().Throw<ValueError>().WithMessage("Maximum allowed dimension exceeded");
            // >>> np.isin(np.array([1], np.int64), np.array([-2**63, -1], np.int64), kind='table')  (the same)
            Action intMin = () => np.isin(one, np.array(new[] { long.MinValue, -1L }), kind: "table");
            intMin.Should().Throw<ValueError>().WithMessage("Maximum allowed dimension exceeded");
            // >>> np.isin(np.array([1], np.int64), np.array([0, 2**63 - 1], np.int64), kind='table', invert=True)  (np.ones)
            Action invert = () => np.isin(one, np.array(new[] { 0L, long.MaxValue }), invert: true, kind: "table");
            invert.Should().Throw<ValueError>().WithMessage("Maximum allowed dimension exceeded");
            // >>> np.isin(np.array([1], np.int64), np.array([0, 2**63 - 1], np.uint64), kind='table')  (the same)
            Action unsigned = () => np.isin(one, np.array(new[] { 0UL, (ulong)long.MaxValue }), kind: "table");
            unsigned.Should().Throw<ValueError>().WithMessage("Maximum allowed dimension exceeded");
            // >>> np.isin(np.array([], np.int64), np.array([0, 2**64 - 1], np.uint64), kind='table')  (an empty element too)
            Action empty = () => np.isin(np.zeros(new Shape(0), np.int64), np.array(new[] { 0UL, ulong.MaxValue }), kind: "table");
            empty.Should().Throw<ValueError>().WithMessage("Maximum allowed dimension exceeded");

            // A range past the dtype's maximum is the earlier RuntimeError (the house InvalidOperationException).
            // >>> np.isin(np.array([1], np.int64), np.array([-2**63, 2**63 - 1], np.int64), kind='table')
            // RuntimeError: You have specified kind='table', but the range of values in `ar2` or `ar1` exceed the ...
            Action unsafeRange = () => np.isin(one, np.array(new[] { long.MinValue, long.MaxValue }), kind: "table");
            unsafeRange.Should().Throw<InvalidOperationException>().WithMessage("You have specified kind='table'*");

            // Without kind='table' the range only declines the table method.
            // >>> np.isin(np.array([1], np.int64), np.array([0, 2**63 - 1], np.int64))  array([False])
            np.isin(one, np.array(new[] { 0L, long.MaxValue })).array_equal(np.array(new[] { false })).Should().BeTrue();
            // >>> np.isin(np.array([1], np.int64), np.array([0, 2**63 - 1], np.int64), kind='sort')  array([False])
            np.isin(one, np.array(new[] { 0L, long.MaxValue }), kind: "sort").array_equal(np.array(new[] { false }))
                .Should().BeTrue();
        }

        /// <summary>
        ///     NumPy's sort path (a set too long for the loop path, and no table method) concatenates the unique values in
        ///     their common dtype and flags each value whose successor in the stable sort equals it. Two 64-bit integers past
        ///     2**53 can become one float64 there, and under <c>assume_unique</c> repeated values meet as neighbours: the first
        ///     of two equal neighbours is then a "member". NumSharp answered the exact membership. The table method and the
        ///     loop path stay exact.
        /// </summary>
        [TestMethod]
        public void SortPath_LossyConcatenationAndAssumeUnique_AreNumPysAnswers()
        {
            long big = 1L << 60;
            var near = np.array(new[] { (ulong)big + 1 }.Concat(Enumerable.Range(0, 20).Select(i => (ulong)i)).ToArray());
            // >>> np.isin(np.array([2**60], np.int64), np.array([2**60 + 1] + list(range(20)), np.uint64))  array([ True])
            np.isin(np.array(new[] { big }), near).array_equal(np.array(new[] { true })).Should().BeTrue();
            // >>> ... invert=True  array([False])
            np.isin(np.array(new[] { big }), near, invert: true).array_equal(np.array(new[] { false })).Should().BeTrue();
            // The loop path compares exactly (NumPy's int64 / uint64 loops):
            // >>> np.isin(np.array([2**60] * 200, np.int64), np.array([2**60 + 1, 0], np.uint64))[:3]  [False, False, False]
            var loop = np.isin(np.full(new Shape(200), big), np.array(new[] { (ulong)big + 1, 0UL }));
            loop.array_equal(np.zeros(new Shape(200), np.@bool)).Should().BeTrue();
            // The table method is exact: >>> np.isin(np.array([2**60], np.int64), np.array([2**60+1, 2**60+2], np.uint64))
            // array([False])
            np.isin(np.array(new[] { big }), np.array(new[] { (ulong)big + 1, (ulong)big + 2 }))
                .array_equal(np.array(new[] { false })).Should().BeTrue();

            // Distinct values of the element that become one double: the first of them is flagged.
            // >>> np.isin(np.array([2**64 - 1, 2**64 - 2, 3], np.uint64), np.arange(24.))  array([False,  True,  True])
            var huge = np.array(new[] { ulong.MaxValue, ulong.MaxValue - 1, 3UL });
            var floats = np.arange(24).astype(np.float64);
            np.isin(huge, floats).array_equal(np.array(new[] { false, true, true })).Should().BeTrue();
            // >>> ... invert=True  array([ True, False, False])
            np.isin(huge, floats, invert: true).array_equal(np.array(new[] { true, false, false })).Should().BeTrue();
            // >>> np.isin(np.array([[2**64-1, 3], [2**64-2, 0]], np.uint64), np.arange(24.))
            // array([[False,  True], [ True,  True]])
            var huge2d = np.array(new[] { ulong.MaxValue, 3UL, ulong.MaxValue - 1, 0UL }).reshape(2, 2);
            np.isin(huge2d, floats).array_equal(np.array(new[] { false, true, true, true }).reshape(2, 2)).Should().BeTrue();
            // A 64-bit integer within 2**53 converts exactly: the membership. >>> np.isin(np.array([5, 2**53], np.int64),
            // np.arange(24.) + 0.5)  array([False, False])
            np.isin(np.array(new[] { 5L, 1L << 53 }), floats + 0.5).array_equal(np.array(new[] { false, false }))
                .Should().BeTrue();

            // assume_unique skips np.unique, so repeated values are neighbours after the stable sort.
            var far = np.arange(20).astype(np.float64) + 100.0;
            // >>> np.isin(np.array([5.0, 5.0]), np.arange(20.) + 100, assume_unique=True, invert=True)  array([False,  True])
            np.isin(np.array(new[] { 5.0, 5.0 }), far, assume_unique: true, invert: true)
                .array_equal(np.array(new[] { false, true })).Should().BeTrue();
            // >>> np.isin(np.array([100.0, 100.0, 3.0]), np.arange(20.) + 100, assume_unique=True)  [True, True, False]
            np.isin(np.array(new[] { 100.0, 100.0, 3.0 }), far, assume_unique: true)
                .array_equal(np.array(new[] { true, true, false })).Should().BeTrue();
            // A repeated set value: >>> np.isin(np.array([100.0, 101.0, 3.0]),
            // np.array([100.0, 100.0] + list(np.arange(20.) + 200)), assume_unique=True)  array([ True, False, False])
            var repeated = np.array(new[] { 100.0, 100.0 }.Concat(Enumerable.Range(0, 20).Select(i => 200.0 + i)).ToArray());
            np.isin(np.array(new[] { 100.0, 101.0, 3.0 }), repeated, assume_unique: true)
                .array_equal(np.array(new[] { true, false, false })).Should().BeTrue();
            // Integers too, once the range declines the table method:
            // >>> np.isin(np.array([1, 1, 2], np.int64), np.arange(40, dtype=np.int64) * 1000, assume_unique=True)
            // array([ True, False, False])
            var spread = np.arange(40).astype(np.int64) * 1000L;
            np.isin(np.array(new[] { 1L, 1L, 2L }), spread, assume_unique: true)
                .array_equal(np.array(new[] { true, false, false })).Should().BeTrue();
            // The table method ignores assume_unique: >>> np.isin(np.array([5, 5]), np.arange(20) + 100, assume_unique=True)
            // array([False, False])
            np.isin(np.array(new[] { 5L, 5L }), np.arange(20).astype(np.int64) + 100L, assume_unique: true)
                .array_equal(np.array(new[] { false, false })).Should().BeTrue();
        }

        /// <summary>
        ///     A table NumPy can address but not allocate (<c>2**63 - 2</c> rises to 8 EiB) is NumPy's <c>MemoryError</c>,
        ///     whose threshold is the host's memory; NumSharp's membership needs no table and answers.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void TableNumPyCannotAllocate_IsAnswered()
        {
            // >>> np.isin(np.array([1], np.int64), np.array([0, 2**63 - 2], np.int64), kind='table')
            // MemoryError: Unable to allocate 8.00 EiB for an array with shape (9223372036854775807,) and data type bool
            np.isin(np.array(new long[] { 1 }), np.array(new[] { 0L, long.MaxValue - 1 }), kind: "table")
                .array_equal(np.array(new[] { false })).Should().BeTrue();
        }
    }
}
