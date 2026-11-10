using System;
using System.Text;

namespace NumSharp.Tests.Creation;

/// <summary>
///     <c>np.array(values, np.dtype("c"))</c>: NumPy's legacy character dtype sets <c>DISCOVER_STRINGS_AS_SEQUENCES</c>,
///     so every text whose length is not 1 becomes one more axis of its characters (an <c>|S1</c> array that keeps the
///     <c>'c'</c> type character). Every expectation is NumPy 2.4.2's output for the same call (probed).
/// </summary>
/// <remarks>
///     Gymnasium builds its toy_text maps this way (<c>np.asarray(desc, dtype="c")</c>): FrozenLake's
///     <c>["SFFF", "FHFH", "FFFH", "HFFG"]</c> is a (4, 4) byte map.
/// </remarks>
[TestClass]
public class np_array_CharDType_Test
{
    /// <summary>The character dtype under test, <c>np.dtype('c')</c>.</summary>
    private static readonly DType C = np.dtype("c");

    /// <summary>Asserts NumPy's shape, the <c>|S1</c> descriptor with its <c>'c'</c> type character, and the bytes.</summary>
    /// <param name="a">The array to check.</param>
    /// <param name="shape">NumPy's shape.</param>
    /// <param name="bytes">NumPy's <c>tobytes()</c>, as Latin-1 text.</param>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    private static void AssertCharArray(NDArray a, long[] shape, string bytes)
    {
        a.shape.Should().Equal(shape);
        a.dtype.str.Should().Be("|S1");
        a.dtype.@char.Should().Be('c');
        Encoding.Latin1.GetString(a.tobytes()).Should().Be(bytes);
    }

    /// <summary>FrozenLake's map: <c>(4, 4) |S1</c>, one byte per character, row by row.</summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void Rows_SplitIntoCharacters()
    {
        using var map = np.array(new[] { "SFFF", "FHFH", "FFFH", "HFFG" }, C);
        AssertCharArray(map, new long[] { 4, 4 }, "SFFFFHFHFFFHHFFG");
        map.ToString().Should().Be("[[b'S' b'F' b'F' b'F']\n [b'F' b'H' b'F' b'H']\n [b'F' b'F' b'F' b'H']\n [b'H' b'F' b'F' b'G']]");
    }

    /// <summary>
    ///     The lone-text forms: <c>np.array('abc', dtype='c')</c> is (3,), a one-character text stays 0-d, an empty text is
    ///     (0,); <c>np.asanyarray</c> of a lone text splits the same way.
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void LoneText_IsASequenceUnlessOneCharacter()
    {
        using (var three = np.array("abc", C)) AssertCharArray(three, new long[] { 3 }, "abc");
        using (var one = np.array("a", C)) AssertCharArray(one, Array.Empty<long>(), "a");
        using (var none = np.array("", C)) AssertCharArray(none, new long[] { 0 }, "");
        using (var any = np.asanyarray((object)"abc", C)) AssertCharArray(any, new long[] { 3 }, "abc");
    }

    /// <summary>
    ///     Empty texts are empty sequences (<c>['', '']</c> is (2, 0), <c>['']</c> (1, 0)), one-character texts are
    ///     elements (<c>['a', 'b']</c> is (2,)), and an empty list is (0,).
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void EmptyAndOneCharacterTexts()
    {
        using (var twoEmpty = np.array(new[] { "", "" }, C)) AssertCharArray(twoEmpty, new long[] { 2, 0 }, "");
        using (var oneEmpty = np.array(new[] { "" }, C)) AssertCharArray(oneEmpty, new long[] { 1, 0 }, "");
        using (var singles = np.array(new[] { "a", "b" }, C)) AssertCharArray(singles, new long[] { 2 }, "ab");
        using (var noValues = np.array(Array.Empty<string>(), C)) AssertCharArray(noValues, new long[] { 0 }, "");
    }

    /// <summary>
    ///     Nesting adds the character axis last, for a jagged array and a rectangular grid alike, with <c>ndmin</c>
    ///     prepending ones and <c>order='F'</c> keeping the values.
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void NestedTexts_GainACharacterAxis()
    {
        using (var jagged = np.array(new[] { new[] { "ab", "cd" }, new[] { "ef", "gh" } }, C))
            AssertCharArray(jagged, new long[] { 2, 2, 2 }, "abcdefgh");
        using (var grid = np.array(new string[,] { { "ab", "cd" }, { "ef", "gh" } }, C))
            AssertCharArray(grid, new long[] { 2, 2, 2 }, "abcdefgh");
        using (var raised = np.array(new[] { "ab" }, C, ndmin: 3))
            AssertCharArray(raised, new long[] { 1, 1, 2 }, "ab");
        using (var fortran = np.array(new[] { new[] { "ab", "cd" }, new[] { "ef", "gh" } }, C, order: 'F'))
            AssertCharArray(fortran, new long[] { 2, 2, 2 }, "abcdefgh");
    }

    /// <summary>
    ///     Texts of different lengths are a ragged sequence: NumPy's inhomogeneous-shape ValueError, whichever side the
    ///     longer or the one-character text is on, and a number beside a text alike.
    /// </summary>
    /// <param name="first">The first value.</param>
    /// <param name="second">The second value.</param>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    [DataRow("ab", "abc")]
    [DataRow("a", "bc")]
    [DataRow("bc", "a")]
    [DataRow("ab", 12)]
    public void UnequalLengths_AreRagged(object first, object second)
    {
        Action build = () => np.array(new[] { first, second }, C);
        build.Should().ThrowExactly<ValueError>().WithMessage(
            "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. " +
            "The detected shape was (2,) + inhomogeneous part.");
    }

    /// <summary>
    ///     Each character is encoded alone, so a non-ASCII one fails at position 0, quoted as CPython quotes it (a code
    ///     point above U+FFFF is ONE character, so <c>'😀x'</c> is two characters, not three).
    /// </summary>
    /// <param name="text">The text.</param>
    /// <param name="quoted">CPython's quoting of the offending character.</param>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    [DataRow("é", "'\\xe9'")]
    [DataRow("aé", "'\\xe9'")]
    [DataRow("\U0001F600x", "'\\U0001f600'")]
    public void NonAscii_FailsAtPositionZero(string text, string quoted)
    {
        Action build = () => np.array(new[] { text }, C);
        build.Should().ThrowExactly<UnicodeEncodeError>()
            .WithMessage($"'ascii' codec can't encode character {quoted} in position 0: ordinal not in range(128)");
    }

    /// <summary>
    ///     Values that are no text stay elements and are stored as their <c>str()</c> cut to one byte (<c>[1, 2]</c> is
    ///     <c>b'12'</c>, <c>[12]</c> is <c>b'1'</c>), a NUL is a character like any other, and a bytes value (a C#
    ///     <c>byte[]</c>) is a sequence of ints, each stored as its <c>str()</c>: NumPy's <c>[b'ab', b'cd']</c> under
    ///     <c>'c'</c> is <c>b'9991'</c> (97, 98, 99, 100).
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void NumbersNulsAndBytes()
    {
        using (var ints = np.array(new object[] { 1, 2 }, C)) AssertCharArray(ints, new long[] { 2 }, "12");
        using (var twelve = np.array(new object[] { 12 }, C)) AssertCharArray(twelve, new long[] { 1 }, "1");
        using (var nul = np.array(new[] { "a\0b" }, C)) AssertCharArray(nul, new long[] { 1, 3 }, "a\0b");
        using (var bytes = np.array(new object[] { new byte[] { 97, 98 }, new byte[] { 99, 100 } }, C))
            AssertCharArray(bytes, new long[] { 2, 2 }, "9991");
    }

    /// <summary>
    ///     Only array coercion splits. An existing array is cast (<c>&lt;U2</c> to <c>'c'</c> keeps the first character),
    ///     <c>astype('c')</c> casts, <c>np.dtype('S1')</c> (equal to <c>'c'</c>, other type character) truncates, and a fill
    ///     stores the whole text cut to one byte.
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void ArraysCastsAndStores_DoNotSplit()
    {
        using var text = np.array(new[] { "ab", "cd" });
        using (var cast = np.asarray(text, C)) AssertCharArray(cast, new long[] { 2 }, "ac");
        using (var astype = text.astype(C)) AssertCharArray(astype, new long[] { 2 }, "ac");
        using (var s1 = np.array(new[] { "abc", "de" }, np.dtype("S1")))
        {
            s1.shape.Should().Equal(2L);
            s1.dtype.@char.Should().Be('S');
            Encoding.Latin1.GetString(s1.tobytes()).Should().Be("ad");
        }
        using var filled = np.zeros(2, C);
        filled.fill("xyz");
        AssertCharArray(filled, new long[] { 2 }, "xx");
    }
}
