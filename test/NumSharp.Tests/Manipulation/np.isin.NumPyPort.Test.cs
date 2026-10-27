using System;

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
    }
}
