using System;
using System.Collections.Generic;
using NumSharp;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for the text examples of three DocFX guides that have no doc-test class of their own:
    ///     <c>docs/website-src/docs/quickstart.md</c> (strings in the quickstart), <c>getting-and-setting-values.md</c>
    ///     ("Reading and writing text") and <c>iterating-and-enumerating.md</c> ("Iterating text"). Each test mirrors one
    ///     documented snippet and asserts what the page claims; every value was probed against NumPy 2.4.2, so a change
    ///     that breaks the guide's promise about text fails here instead of on the site.
    /// </summary>
    [TestClass]
    public class TextInGuidesDocTests
    {
        /// <summary>
        ///     The quickstart's string snippet: a <c>U4</c> array compares with a C# string, sorts, repeats through
        ///     <c>np.strings.multiply</c> (sized to the longest repetition), and as <c>StringDType</c> holds a text longer
        ///     than the width it came from.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void Quickstart_StringsUseTheSameModel()
        {
            var names = np.array(new[] { "cat", "dog", "bird" });
            names.dtype.str.Should().Be("<U4", "inferred fixed width");
            (names == "cat").ToArray<bool>().Should().Equal(true, false, false);
            np.sort(names).ToString(true).Should().Be("array(['bird', 'cat', 'dog'], dtype='<U4')");
            np.strings.multiply(names, 2).ToString(true).Should().Be("array(['catcat', 'dogdog', 'birdbird'], dtype='<U8')");
            var growing = names.astype(np.dtypes.StringDType());
            growing[0] = "a much longer label";
            growing.ToString(true).Should().Be("array(['a much longer label', 'dog', 'bird'], dtype=StringDType())");
        }

        /// <summary>
        ///     The "Reading and writing text" snippet of getting-and-setting-values.md: <c>item()</c> returns a C# string, a
        ///     <c>U4</c> store cuts a longer text, and in a <c>StringDType(na_object: null)</c> array a C# <c>null</c> is the
        ///     missing value (read back as <c>null</c>, printed as <c>None</c>), not the text <c>"None"</c>.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void GettingAndSetting_ReadingAndWritingText()
        {
            var names = np.array(new[] { "cat", "dog" }, "U4");
            ((string)names.item(0)).Should().Be("cat");
            names[1] = "zebras";
            NDArray.AsStringArray(names).Should().Equal("cat", "zebr");
            var text = names.astype(np.dtypes.StringDType(na_object: null));
            text[1] = null;
            text.item(1).Should().BeNull("the NA object, not \"None\"");
            text.ToString(true).Should().Be("array(['cat', None], dtype=StringDType(na_object=None))");
        }

        /// <summary>
        ///     The "Getting and setting values" chararray rule: an element read through <c>c[0]</c> strips trailing
        ///     whitespace, while <c>c.item(0)</c> and <c>tolist()</c> return the stored text; iteration strips like element
        ///     access (the iterating guide's chararray sentence).
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void GettingAndSetting_CharArrayElementsStrip()
        {
            var c = np.@char.array(new[] { "cat  ", "dog" });
            c[0].item().Should().Be("cat");
            c.item(0).Should().Be("cat  ");
            ((List<object>)c.tolist()).Should().Equal("cat  ", "dog");
            var iterated = new List<object>();
            foreach (var element in c)
                iterated.Add(element);
            iterated.Should().Equal("cat", "dog");
        }

        /// <summary>
        ///     The "Iterating text" snippet of iterating-and-enumerating.md: an <c>np.nditer</c> over a <c>StringDType</c>
        ///     array needs <c>refs_ok</c>, and each step's 0-d operand yields the text or the NA object itself
        ///     (<c>null</c>), never its <c>"None"</c> rendering.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
        [TestMethod]
        public void Iterating_StringDTypeNeedsRefsOk()
        {
            var text = np.array(new object[] { "cat", null }, np.dtypes.StringDType(na_object: null));
            var values = new List<object>();
            using (var it = np.nditer(text, flags: new[] { "refs_ok" }))
                foreach (var operands in it)
                    values.Add(operands[0].item());
            // Element by element: the collection Equal cannot take a null expected element.
            values.Should().HaveCount(2);
            values[0].Should().Be("cat");
            values[1].Should().BeNull("the NA object is preserved");

            // Without refs_ok NumPy refuses a StringDType operand (a TypeError), and so does NumSharp.
            Action withoutFlag = () => np.nditer(text).Dispose();
            withoutFlag.Should().Throw<TypeError>();
        }
    }
}
