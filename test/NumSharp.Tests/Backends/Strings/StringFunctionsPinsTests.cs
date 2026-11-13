using System;
using System.Linq;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The <c>numpy.strings</c> answers the string oracle did not hold (strings phase 2, P2.5): inputs where NumPy 2.4.2
    ///     read uninitialized memory, wrote past a buffer or looped forever (each fixed in NumPy 2.5, whose answer is
    ///     NumSharp's), a text NumPy prints per platform, the size checks NumPy 2.5 added, and the one layout NumSharp cannot
    ///     reproduce. Each test pins NumSharp's answer and states NumPy's (probed on win-amd64 2.4.2 and 2.5.4, each
    ///     statement in its own process).
    /// </summary>
    [TestClass]
    public class StringFunctionsPinsTests
    {
        /// <summary>The values of a 1-D <c>StringDType</c> array.</summary>
        /// <param name="t">The array.</param>
        /// <returns>The strings.</returns>
        private static string[] Texts(NDArray t) => Enumerable.Range(0, (int)t.size).Select(i => (string)t.GetValue(i)).ToArray();

        /// <summary>
        ///     zfill of an empty string is all zeros in every class. NumPy 2.4.2's <c>string_zfill</c> read the unit at
        ///     <c>out + final_width</c> — past its own output element — to test for a sign, so its answer for <c>''</c>
        ///     depended on the next element's memory (probed: <c>'000'</c> that time); NumPy 2.5 tests for a sign only when
        ///     the text is not empty and padding was added (<c>len &gt; 0 &amp;&amp; final_width &gt; len</c>), and NumSharp
        ///     reads no such unit either.
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
        ///     zfill over <c>StringDType</c> refuses a padded size past int64 with NumPy 2.5's <c>OverflowError: Overflow
        ///     encountered in zfill</c>: its <c>safe_add</c> of the padding (width minus code points) and the text's UTF-8
        ///     size overflows for a width of <c>2**63 - 1</c> over a two-byte character (NumPy 2.4.2 handed the wrapped size to
        ///     its allocator: <c>MemoryError</c>). A width whose size fits int64 but no allocation stays NumPy's
        ///     <c>MemoryError: Failed to allocate string in zfill</c> (one-byte <c>'a'</c>). Both probed on 2.5.4.
        /// </summary>
        [TestMethod]
        public void ZFill_StringDType_SizePastInt64_IsTheOverflowError()
        {
            using NDArray huge = long.MaxValue;
            using var twoByte = np.array(new[] { ((char)0xE9).ToString() }, np.dtypes.StringDType());
            Action overflow = () => np.strings.zfill(twoByte, huge);
            overflow.Should().Throw<OverflowError>().WithMessage("Overflow encountered in zfill");
            using var oneByte = np.array(new[] { "a" }, np.dtypes.StringDType());
            Action memory = () => np.strings.zfill(oneByte, huge);
            memory.Should().Throw<MemoryError>().WithMessage("Failed to allocate string in zfill");
        }

        /// <summary>
        ///     slice of a <c>StringDType</c> text with a step past int's range selects one code point. NumPy 2.4.2's loop
        ///     counted the selected code points in an <c>int</c> that overflowed and never returned (probed: no answer within
        ///     20 s); NumPy 2.5 stops before the step's last advance and answers <c>'a'</c>, as its <c>U</c> loop always did and
        ///     NumSharp does for both.
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
        ///     replace over <c>StringDType</c> with any negative count replaces every occurrence, as its <c>U</c> wrapper does
        ///     (<c>'XYZbXYZb'</c>): NumPy 2.5's loop reads <c>count &lt; 0</c> as "all". (NumPy 2.4's took
        ///     <c>min(count, found)</c>, so a count other than -1 replaced nothing for an equal or shorter replacement and
        ///     wrote past its output for a longer one.)
        /// </summary>
        [TestMethod]
        public void Replace_StringDType_NegativeCount_ReplacesEveryOccurrence()
        {
            using NDArray a = "a", xyz = "XYZ", x = "X", minus5 = -5;
            using var t = np.array(new[] { "abab" }, np.dtypes.StringDType());
            using (var r = np.strings.replace(t, a, x, minus5))
                Texts(r).Should().Equal("XbXb");
            using (var r = np.strings.replace(t, a, xyz, minus5))
                Texts(r).Should().Equal("XYZbXYZb");
            using var u = np.array(new[] { "abab" });
            using var ru = np.strings.replace(u, a, xyz, minus5);
            NDArray.AsStringArray(ru).Should().Equal("XYZbXYZb");
        }

        /// <summary>
        ///     replace over <c>StringDType</c> with an old text of only NULs: the Python str becomes a <c>U</c> array first,
        ///     which drops its trailing NULs, so the old is EMPTY and the replacement goes around every code point — the
        ///     text's trailing NULs included, which are code points since NumPy 2.5.3 (probed against 2.5.4; NumPy 2.4 kept
        ///     them outside: <c>'XaXbX\0\0'</c>).
        /// </summary>
        [TestMethod]
        public void Replace_StringDType_AllNulOld_InterleavesLikeAnEmptyOld()
        {
            using NDArray nuls = "\0\0", x = "X";
            using var t = np.array(new[] { "ab\0\0" }, np.dtypes.StringDType());
            using var r = np.strings.replace(t, nuls, x);
            Texts(r).Should().Equal("XaXbX\0X\0X");
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
            act.Should().Throw<OverflowError>().WithMessage("Python int too large to convert to C long");
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
