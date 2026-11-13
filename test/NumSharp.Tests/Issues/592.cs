using System;
using System.IO;
using System.Linq;
using System.Reflection;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #592, "[Core] NumPy String support in NumSharp"
    ///     (https://github.com/SciSharp/NumSharp/issues/592): the <c>S</c> / <c>U</c> / <c>StringDType</c> dtypes, all 46
    ///     <c>numpy.strings</c> functions, <c>numpy.char</c> with <c>chararray</c>, the array functions on text and the text
    ///     round trips, each against NumPy 2.4.2's output. The answer's "not covered" list is pinned as
    ///     <c>[OpenBugs]</c> so it turns green when those land.
    /// </summary>
    [TestClass]
    public class Issue592
    {
        /// <summary>The answer's sample text array: duplicates, an empty string, and widths up to 5.</summary>
        /// <returns>A fresh <c>&lt;U5</c> array <c>['pear', 'apple', 'fig', '', 'apple']</c>.</returns>
        private static NDArray Fruits() => np.array(new[] { "pear", "apple", "fig", "", "apple" });

        /// <summary>
        ///     <c>S</c>, <c>U</c> and <c>StringDType</c> are real dtypes with NumPy's descriptor strings and item sizes
        ///     (<c>|S3</c> is 3 bytes, <c>&lt;U5</c> 20 bytes of UCS-4, <c>StringDType()</c> a 16-byte cell).
        /// </summary>
        [TestMethod]
        public void TextDtypes_HaveNumPysDescriptors()
        {
            np.dtype("S3").str.Should().Be("|S3");
            np.dtype("S3").itemsize.Should().Be(3);
            np.dtype("U5").str.Should().Be("<U5");
            np.dtype("U5").itemsize.Should().Be(20);
            np.dtype("T").ToString().Should().Be("StringDType()");
            np.dtype("T").itemsize.Should().Be(16);
        }

        /// <summary>
        ///     <c>np.strings</c> has every one of the 46 public functions of NumPy 2.4.2's <c>numpy.strings</c>
        ///     (<c>dir(np.strings)</c> without underscores).
        /// </summary>
        [TestMethod]
        public void NpStrings_HasAllFortySixFunctions()
        {
            var numpyNames = new[]
            {
                "add", "capitalize", "center", "count", "decode", "encode", "endswith", "equal", "expandtabs", "find",
                "greater", "greater_equal", "index", "isalnum", "isalpha", "isdecimal", "isdigit", "islower", "isnumeric",
                "isspace", "istitle", "isupper", "less", "less_equal", "ljust", "lower", "lstrip", "mod", "multiply",
                "not_equal", "partition", "replace", "rfind", "rindex", "rjust", "rpartition", "rstrip", "slice",
                "startswith", "str_len", "strip", "swapcase", "title", "translate", "upper", "zfill",
            };
            numpyNames.Should().HaveCount(46);

            var ours = typeof(np.strings).GetMethods(BindingFlags.Public | BindingFlags.Static).Select(m => m.Name).ToHashSet();
            numpyNames.Where(n => !ours.Contains(n)).Should().BeEmpty("the answer says all 46 numpy.strings functions exist");
        }

        /// <summary>
        ///     <c>numpy.char</c>'s comparisons strip trailing whitespace where the plain operator does not, and
        ///     <c>np.@char.array</c> builds a <c>chararray</c> whose <c>==</c> strips the same way (NumPy 2.4.2's values).
        /// </summary>
        [TestMethod]
        public void NumpyChar_AndChararray_StripLikeNumPy()
        {
            var left = np.array(new[] { "a ", "b" });
            var right = np.array(new[] { "a", "b " });
            np.array_repr(np.@char.equal(left, right)).Should().Be("array([ True,  True])");
            np.array_repr(left == right).Should().Be("array([False, False])");

            var ca = np.@char.array(new[] { "a ", "bc  " });
            ca.Should().BeOfType<NDCharArray>();
            np.array_repr(ca == "a").Should().Be("array([ True, False])");
        }

        /// <summary>
        ///     The array functions the answer lists, over a <c>U</c> array: sort, stable argsort, unique, searchsorted, where,
        ///     take, nonzero (non-empty strings), any / all, concatenate and <c>np.strings.str_len</c>, each as NumPy 2.4.2's
        ///     repr.
        /// </summary>
        [TestMethod]
        public void ArrayFunctions_OnText_MatchNumPy()
        {
            var u = Fruits();
            np.array_repr(np.sort(u)).Should().Be("array(['', 'apple', 'apple', 'fig', 'pear'], dtype='<U5')");
            np.array_repr(np.argsort(u, kind: "stable")).Should().Be("array([3, 1, 4, 2, 0])");
            np.array_repr(np.unique(u)).Should().Be("array(['', 'apple', 'fig', 'pear'], dtype='<U5')");
            np.array_repr(np.searchsorted(np.sort(u), np.array(new[] { "banana", "fig" }))).Should().Be("array([3, 3])");
            np.array_repr(np.where(np.array(new[] { true, false, true, false, true }), u, "-"))
                .Should().Be("array(['pear', '-', 'fig', '-', 'apple'], dtype='<U5')");
            np.array_repr(np.take(u, np.array(new long[] { 2, 0 }))).Should().Be("array(['fig', 'pear'], dtype='<U5')");
            np.array_repr(np.nonzero(u)[0]).Should().Be("array([0, 1, 2, 4])");
            np.any(u).Should().BeTrue();
            np.all(u).Should().BeFalse("the empty string is falsy");
            np.array_repr(np.concatenate(new[] { u["0:2"], np.array(new[] { "kiwi" }) }))
                .Should().Be("array(['pear', 'apple', 'kiwi'], dtype='<U5')");
            np.array_repr(np.strings.str_len(u)).Should().Be("array([4, 5, 3, 0, 5])");
        }

        /// <summary>
        ///     <c>np.save</c> / <c>np.load</c> round-trip a <c>U</c> array and a <c>StringDType</c> array (the latter as NumPy's
        ///     pickle), and <c>np.savetxt</c> / <c>np.loadtxt</c> round-trip a <c>U</c> array.
        /// </summary>
        [TestMethod]
        public void SaveLoad_AndTextIO_RoundTripText()
        {
            var dir = Path.Combine(Path.GetTempPath(), "numsharp-issue592-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                var u = Fruits();
                np.save(Path.Combine(dir, "u.npy"), u);
                np.array_equal((NDArray)np.load(Path.Combine(dir, "u.npy")), u).Should().BeTrue();

                var t = np.array(new[] { "x", "yy" }, dtype: np.dtype("T"));
                np.save(Path.Combine(dir, "t.npy"), t);
                np.array_repr((NDArray)np.load(Path.Combine(dir, "t.npy"), allow_pickle: true))
                    .Should().Be("array(['x', 'yy'], dtype=StringDType())");

                var words = np.array(new[] { "pear", "apple", "fig" });
                np.savetxt(Path.Combine(dir, "w.txt"), words, fmt: "%s");
                np.array_repr(np.loadtxt(Path.Combine(dir, "w.txt"), dtype: np.dtype("U5")))
                    .Should().Be("array(['pear', 'apple', 'fig'], dtype='<U5')");
            }
            finally
            {
                Directory.Delete(dir, recursive: true);
            }
        }

        /// <summary>
        ///     The answer's "not covered" list: <c>np.genfromtxt</c>, <c>np.memmap</c> and <c>np.rec</c> exist in NumPy 2.4.2
        ///     and not in NumSharp yet. Fails until all three are added; then the tag comes off.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "https://github.com/SciSharp/NumSharp/issues/592")]
        public void NotCovered_GenfromtxtMemmapRec_AreMissing()
        {
            var missing = new[] { "genfromtxt", "memmap", "rec" }
                .Where(n => typeof(np).GetMember(n, BindingFlags.Public | BindingFlags.Static).Length == 0)
                .ToArray();
            missing.Should().BeEmpty($"NumPy 2.4.2 has np.genfromtxt, np.memmap and np.rec; missing: {string.Join(", ", missing)}");
        }
    }
}
