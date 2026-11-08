using System;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     WHICH element a refused text cast names. NumPy's cast loop walks its input in MEMORY order (an F-ordered array
    ///     column by column; a transposed C array in the order its memory lies), so the <c>UnicodeEncodeError</c> of an
    ///     ASCII cast names the first unencodable element of that walk — and so does NumSharp's. Every expected message was
    ///     probed against <c>numpy==2.4.2</c>.
    /// </summary>
    /// <remarks>
    ///     <para>The input is the 2 × 2 text <c>[['a', 'Σ'], ['ǅ', 'b']]</c>: in C order its memory is <c>a, Σ, ǅ, b</c>,
    ///     in F order <c>a, ǅ, Σ, b</c>, so the two unencodable elements trade places between the walks. numpy's results:
    ///     <c>a.astype('S3')</c> names <c>Σ</c>, <c>np.asfortranarray(a).astype('S3')</c> names <c>ǅ</c>,
    ///     <c>a.T.astype('S3')</c> (F-contiguous over C memory) names <c>Σ</c>, and <c>np.asfortranarray(a).T</c>
    ///     (C-contiguous over F memory) names <c>ǅ</c> — the same for a <c>StringDType</c> source.</para>
    ///     <para>Why it is pinned: the pythonnet end-to-end text tests first saw the two sides name different elements, and
    ///     the cause was the INPUT, not the walk — the bridge's import copy is C-contiguous by contract, so a NumSharp copy
    ///     of an F-ordered numpy array is walked as C memory. A walk in logical C order would pass the C-ordered cells here
    ///     and fail the F-ordered and transposed ones, so this class is what keeps the walk honest.</para>
    /// </remarks>
    [TestClass]
    public class TextCastErrorOrderTests
    {
        /// <summary>The capital sigma, unencodable in ASCII (C memory's first unencodable element).</summary>
        private static readonly string Sigma = ((char)0x03A3).ToString();

        /// <summary>The titlecase digraph U+01C5, unencodable in ASCII (F memory's first unencodable element).</summary>
        private static readonly string Dz = ((char)0x01C5).ToString();

        /// <summary>The 2 × 2 <c>U3</c> source in C order.</summary>
        /// <returns>A fresh C-contiguous array.</returns>
        private static NDArray Source() => np.array(new[] { "a", Sigma, Dz, "b" }).astype(np.dtype("U3")).reshape(2, 2);

        /// <summary>
        ///     numpy's <c>UnicodeEncodeError</c> text for an ASCII cast whose first unencodable character is
        ///     <paramref name="c"/> at position 0 (Python's repr escapes a non-ASCII BMP character as <c>\uXXXX</c>).
        /// </summary>
        /// <param name="c">The character the message names.</param>
        /// <returns>The message.</returns>
        private static string EncodeErrorText(string c)
            => "'ascii' codec can't encode character '" + "\\" + "u" + ((int)c[0]).ToString("x4") +
               "' in position 0: ordinal not in range(128)";

        /// <summary>
        ///     The walks where logical C order IS memory order — a C-contiguous input and the transposed view of an
        ///     F-ordered one (C-contiguous over F memory) — name numpy's element, for <c>U</c> (into a sized and an unsized
        ///     <c>S</c>) and for <c>StringDType</c> (into a sized one: both sides refuse an unsized <c>S</c> with a
        ///     <c>TypeError</c>).
        /// </summary>
        [TestMethod]
        public void Astype_WhereCOrderIsMemoryOrder_NamesNumpysElement()
        {
            using var c = Source();
            using var f = np.asfortranarray(c);
            using var fT = f.T;
            using var cText = c.astype(np.dtypes.StringDType());
            foreach (string target in new[] { "S3", "S" })
            {
                new Action(() => c.astype(np.dtype(target))).Should().Throw<UnicodeEncodeError>().WithMessage(EncodeErrorText(Sigma), target);
                new Action(() => fT.astype(np.dtype(target))).Should().Throw<UnicodeEncodeError>().WithMessage(EncodeErrorText(Dz), target);
            }

            new Action(() => cText.astype(np.dtype("S3"))).Should().Throw<UnicodeEncodeError>().WithMessage(EncodeErrorText(Sigma));
        }

        /// <summary>
        ///     The walks where memory order is NOT logical C order — an F-ordered input and the transposed view of a C one
        ///     (F-contiguous over C memory) — name the first unencodable element in MEMORY: <c>ǅ</c> for the F array,
        ///     <c>Σ</c> for the transposed one, as numpy's cast does (<c>PyArray_PrepareTwoRawArrayIter</c> sorts the axes
        ///     by stride before the strided loop runs).
        /// </summary>
        [TestMethod]
        public void Astype_FOrderedOrTransposedInput_NamesTheFirstInMemoryOrder()
        {
            using var c = Source();
            using var f = np.asfortranarray(c);
            using var cT = c.T;
            using var fText = np.asfortranarray(c.astype(np.dtypes.StringDType()));
            foreach (string target in new[] { "S3", "S" })
            {
                new Action(() => f.astype(np.dtype(target))).Should().Throw<UnicodeEncodeError>().WithMessage(EncodeErrorText(Dz), target);
                new Action(() => cT.astype(np.dtype(target))).Should().Throw<UnicodeEncodeError>().WithMessage(EncodeErrorText(Sigma), target);
            }

            new Action(() => fText.astype(np.dtype("S3"))).Should().Throw<UnicodeEncodeError>().WithMessage(EncodeErrorText(Dz));
        }
    }
}
