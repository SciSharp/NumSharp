using System;
using System.Linq;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The <c>numpy.strings</c> answers the string oracle cannot hold (strings phase 2, P2.5): inputs where NumPy 2.4.2
    ///     reads uninitialized memory, writes past a buffer, loops forever or prints a platform-dependent text, and the one
    ///     layout NumSharp cannot reproduce. Each test pins NumSharp's answer and states NumPy's (probed on win-amd64 2.4.2,
    ///     each statement in its own process).
    /// </summary>
    [TestClass]
    public class StringFunctionsPinsTests
    {
        /// <summary>The values of a 1-D <c>StringDType</c> array.</summary>
        /// <param name="t">The array.</param>
        /// <returns>The strings.</returns>
        private static string[] Texts(NDArray t) => Enumerable.Range(0, (int)t.size).Select(i => (string)t.GetValue(i)).ToArray();

        /// <summary>
        ///     zfill of an empty string is all zeros in every class. NumPy's <c>string_zfill</c> reads the unit at
        ///     <c>out + final_width</c> — past its own output element — to test for a sign, so its answer for <c>''</c> depends
        ///     on the next element's memory (probed: <c>'000'</c> this time); NumSharp reads no such unit.
        /// </summary>
        [TestMethod]
        public void ZFill_EmptyString_IsAllZeros_InEveryClass()
        {
            using NDArray three = 3;
            using var u = np.array(new[] { "", "a", "-1" });
            using var ru = np.strings.zfill(u, three);
            NDArray.AsStringArray(ru).Should().Equal("000", "00a", "-01");
            using var s = np.array(new object[] { Array.Empty<byte>(), new byte[] { 97 } });
            using var rs = np.strings.zfill(s, three);
            rs.dtype.str.Should().Be("|S3");
            using var t = np.array(new[] { "", "a" }, np.dtypes.StringDType());
            using var rt = np.strings.zfill(t, three);
            Texts(rt).Should().Equal("000", "00a");
        }

        /// <summary>
        ///     slice of a <c>StringDType</c> text with a step past int's range selects one code point. NumPy's loop counts the
        ///     selected code points in an <c>int</c> that overflows and never returns (probed: no answer within 20 s); its
        ///     <c>U</c> loop answers <c>'a'</c>, which NumSharp gives for both.
        /// </summary>
        [TestMethod]
        public void Slice_StringDType_HugeStep_SelectsOneCodePoint()
        {
            using NDArray zero = 0;
            using NDArray step = 1L << 40;
            using var t = np.array(new[] { "abcdef" }, np.dtypes.StringDType());
            using var r = np.strings.slice(t, zero, null, step);
            Texts(r).Should().Equal("a");
            using var u = np.array(new[] { "abcdef" });
            using var ru = np.strings.slice(u, zero, null, step);
            NDArray.AsStringArray(ru).Should().Equal("a");
        }

        /// <summary>
        ///     replace over <c>StringDType</c> with a negative count other than -1 replaces nothing, whatever the replacement's
        ///     length. NumPy's loop takes <c>min(count, found)</c> and sizes the output with it: an equal or shorter
        ///     replacement copies the text (probed), a longer one wraps a <c>size_t</c> and writes past the output (probed:
        ///     the process ends with an access violation). Its <c>U</c> wrapper reads any negative count as "every
        ///     occurrence" (<c>'XYZbXYZb'</c>), and so does NumSharp's <c>U</c> path.
        /// </summary>
        [TestMethod]
        public void Replace_StringDType_NegativeCount_ReplacesNothing()
        {
            using NDArray a = "a", xyz = "XYZ", x = "X", minus5 = -5;
            using var t = np.array(new[] { "abab" }, np.dtypes.StringDType());
            using (var r = np.strings.replace(t, a, x, minus5))
                Texts(r).Should().Equal("abab");
            using (var r = np.strings.replace(t, a, xyz, minus5))
                Texts(r).Should().Equal("abab");
            using var u = np.array(new[] { "abab" });
            using var ru = np.strings.replace(u, a, xyz, minus5);
            NDArray.AsStringArray(ru).Should().Equal("XYZbXYZb");
        }

        /// <summary>
        ///     replace over <c>StringDType</c> with an old text of only NULs (no code points: an empty old) inserts the
        ///     replacement around every code point and keeps the trailing NULs — NumPy's printed answer, which it reaches
        ///     through a buffer it sized for fewer bytes than it writes.
        /// </summary>
        [TestMethod]
        public void Replace_StringDType_AllNulOld_InterleavesLikeAnEmptyOld()
        {
            using NDArray nuls = "\0\0", x = "X";
            using var t = np.array(new[] { "ab\0\0" }, np.dtypes.StringDType());
            using var r = np.strings.replace(t, nuls, x);
            Texts(r).Should().Equal("XaXbX\0\0");
        }

        /// <summary>
        ///     A Python int past int64 in an int64 slot raises the LP64 (Linux / macOS) text. NumPy's text follows the platform's
        ///     C <c>long</c>: <c>OverflowError: int too big to convert</c> on Windows (probed), <c>Python int too large to
        ///     convert to C long</c> on LP64; NumSharp pins the LP64 text (the string oracle drops such cases).
        /// </summary>
        [TestMethod]
        public void Find_StartPastInt64_RaisesTheLP64Text()
        {
            using NDArray a = "a";
            using NDArray big = 9223372036854775808UL;
            using var u = np.array(new[] { "ab" });
            Action act = () => np.strings.find(u, a, big);
            act.Should().Throw<OverflowException>().WithMessage("Python int too large to convert to C long");
        }

        /// <summary>
        ///     partition / rpartition over <c>S</c> / <c>U</c> return three separate arrays where NumPy returns three field views
        ///     of one structured array (NumSharp has no structured dtype): the values and widths are NumPy's, except that a part
        ///     NumPy sizes at width 0 (<c>'U0'</c>: no separator found anywhere) is width 1, and each part owns its data
        ///     (NumPy's: <c>OWNDATA</c> False, strides the record's 12 bytes).
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Partition_FixedWidth_ReturnsThreeArrays()
        {
            using NDArray sep = "x";
            using var u = np.array(new[] { "ab", "c" });
            var (before, middle, after) = np.strings.partition(u, sep);
            using (before)
            using (middle)
            using (after)
            {
                NDArray.AsStringArray(before).Should().Equal("ab", "c");
                before.dtype.str.Should().Be("<U2");
                middle.dtype.str.Should().Be("<U1");
                after.dtype.str.Should().Be("<U1"); // NumPy: '<U0'
                NDArray.AsStringArray(after).Should().Equal("", "");
                before.Shape.IsContiguous.Should().BeTrue();
            }
        }
    }
}
