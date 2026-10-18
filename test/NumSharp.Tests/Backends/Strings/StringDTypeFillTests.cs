using System;
using System.Linq;
using System.Text;
using NumSharp.Backends;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The bulk fills of a fresh NEP 55 <c>StringDType</c> buffer: <c>np.array(values, StringDType())</c> (its two passes:
    ///     the texts that need no arena first, then one arena reservation and the medium / long texts encoded in place) and
    ///     the <c>S</c> → <c>StringDType</c> cast into a destination whose cells are all zero (one reservation, appends at the
    ///     bump pointer). Both must leave the cells and the arena exactly as NumPy's element-by-element stores do; the
    ///     reference here is that store itself, one setitem per element into a fresh buffer.
    /// </summary>
    [TestClass]
    public class StringDTypeFillTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor.</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>
        ///     The reference: a fresh buffer of <paramref name="descr"/> filled by setitem one element at a time — NumPy's
        ///     <c>stringdtype_setitem</c> per element, in element order, which is how NumPy's <c>np.array</c> stores Python
        ///     objects.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <param name="descr">The descriptor.</param>
        /// <returns>The filled array (the caller disposes it).</returns>
        private static NDArray ElementByElement(object[] values, DType descr)
        {
            var a = np.empty(new Shape(values.Length), descr);
            for (int i = 0; i < values.Length; i++)
                a.Storage.VStringSlice.SetIndex(i, values[i]);
            return a;
        }

        /// <summary>
        ///     The texts of a size class drawn from an alphabet of whole code points (a surrogate pair is one draw, so every
        ///     text is valid UTF-16), with a fixed seed.
        /// </summary>
        /// <param name="count">The number of texts.</param>
        /// <param name="lo">The shortest length in code points.</param>
        /// <param name="hi">The longest length in code points.</param>
        /// <param name="alphabet">The code points to draw from.</param>
        /// <param name="seed">The seed.</param>
        /// <returns>The texts.</returns>
        private static string[] Texts(int count, int lo, int hi, string[] alphabet, int seed)
        {
            var rng = new System.Random(seed);
            return Enumerable.Range(0, count).Select(_ =>
            {
                var sb = new StringBuilder();
                int len = rng.Next(lo, hi + 1);
                for (int k = 0; k < len; k++)
                    sb.Append(alphabet[rng.Next(alphabet.Length)]);
                return sb.ToString();
            }).ToArray();
        }

        /// <summary>ASCII code points (letters, digits, a space and a NUL: a NUL inside a text is an ordinary character).</summary>
        private static readonly string[] Ascii = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ0123456789 \0".Select(c => c.ToString()).ToArray();

        /// <summary>Code points of every UTF-8 length: 1, 2 (é, Ω, ж), 3 (中, €) and 4 bytes (😀, 𝄞).</summary>
        private static readonly string[] Mixed = { "a", "b", "é", "Ω", "ж", "中", "€", "😀", "𝄞", "z" };

        /// <summary>
        ///     <c>np.array</c> of strings gives the cells and the arena cursor of the element-by-element store, for texts of
        ///     every size class and both alphabets, alone and mixed — including the 15/16-byte line, texts of few units but more
        ///     than 15 UTF-8 bytes (a medium text found in pass 1) and the empty text (an all-zero cell).
        /// </summary>
        [TestMethod]
        public void Array_EqualsTheElementByElementStore()
        {
            foreach (var (lo, hi) in new[] { (0, 0), (0, 4), (4, 12), (13, 17), (15, 16), (20, 60), (250, 260), (300, 500), (0, 700) })
                foreach (var alphabet in new[] { Ascii, Mixed })
                {
                    string[] texts = Texts(700, lo, hi, alphabet, seed: lo * 1000 + hi);
                    using var bulk = np.array(texts, T);
                    using var reference = ElementByElement(texts.Cast<object>().ToArray(), T);
                    bulk.tobytes().Should().Equal(reference.tobytes(), $"cells of {lo}-{hi} {(alphabet == Ascii ? "ASCII" : "mixed")} texts");
                    bulk.Storage.VStringSlice.Allocator.ArenaUsed.Should().Be(reference.Storage.VStringSlice.Allocator.ArenaUsed);
                    NDArray.AsStringArray(bulk).Should().Equal(texts);
                }
        }

        /// <summary>
        ///     Values of other kinds after a run of strings (a number's <c>str()</c>, <c>bytes</c>, <c>None</c>, a char, the NA
        ///     object of each kind): from the first such value the general store continues from the same arena cursor, so the
        ///     bytes still equal the element-by-element store; a string equal to a string NA object is an NA cell in pass 1.
        /// </summary>
        [TestMethod]
        public void Array_MixedValues_ContinueFromTheSameCursor()
        {
            object[] values =
            {
                "short", new string('m', 40), "", "é中", "😀😀😀😀", new string('L', 300), 5L, 2.5, true,
                new byte[] { 104, 105 }, null, 'c', new string('q', 33), "tail", double.NaN, new string('t', 260),
            };
            foreach (var descr in new[] { T, np.dtypes.StringDType(na_object: null), np.dtypes.StringDType(na_object: double.NaN) })
            {
                using var bulk = np.array(values, descr);
                using var reference = ElementByElement(values, descr);
                bulk.tobytes().Should().Equal(reference.tobytes(), $"cells under {descr.ToString(true)}");
                bulk.Storage.VStringSlice.Allocator.ArenaUsed.Should().Be(reference.Storage.VStringSlice.Allocator.ArenaUsed);
            }
            var na = np.dtypes.StringDType(na_object: new string('N', 20));
            object[] withNA = { new string('N', 20), new string('z', 33), "x", new string('N', 20), new string('y', 300) };
            using (var bulk = np.array(withNA, na))
            using (var reference = ElementByElement(withNA, na))
            {
                bulk.tobytes().Should().Equal(reference.tobytes(), "a string equal to the NA object is an NA cell");
                bulk.GetValue(0).Should().Be(new string('N', 20));
            }
        }

        /// <summary>
        ///     The arena is reserved once and exactly for ASCII text (capacity equals the bytes used: no 25 % growth step
        ///     happened), and a non-ASCII text, counted at one byte per unit by pass 1, still lands right (the arena grows for
        ///     the difference, NumPy's way).
        /// </summary>
        [TestMethod]
        public void Array_ReservesTheArenaOnce()
        {
            foreach (var (lo, hi) in new[] { (20, 60), (300, 500) })
            {
                using var ascii = np.array(Texts(2000, lo, hi, Ascii, seed: 11), T);
                var allocator = ascii.Storage.VStringSlice.Allocator;
                allocator.ArenaCapacity.Should().Be(allocator.ArenaUsed, "an ASCII fill reserves exactly once");
                string[] mixed = Texts(2000, lo, hi, Mixed, seed: 12);
                using var nonAscii = np.array(mixed, T);
                NDArray.AsStringArray(nonAscii).Should().Equal(mixed);
                nonAscii.Storage.VStringSlice.Allocator.ArenaCapacity.Should().BeGreaterThanOrEqualTo(nonAscii.Storage.VStringSlice.Allocator.ArenaUsed);
            }
        }

        /// <summary>
        ///     Errors in NumPy's element order: a lone surrogate in a LONG text before one in a short text is the one reported
        ///     (pass 2's exact size finds it after every earlier text is stored); one in a short text ends pass 1 and the general
        ///     store reports it; a non-text value under <c>coerce=False</c> after long texts raises NumPy's ValueError.
        /// </summary>
        [TestMethod]
        public void Array_RaisesTheFirstFailingElement()
        {
            string[] texts = Texts(40, 300, 300, Ascii, seed: 3);
            texts[5] = texts[5].Substring(0, 100) + "\ud800" + texts[5].Substring(100);
            texts[9] = "a\udc00";
            Action longFirst = () => np.array(texts, T);
            longFirst.Should().Throw<UnicodeEncodeError>()
                .WithMessage("'utf-8' codec can't encode character '\\ud800' in position 100: surrogates not allowed");
            texts[5] = new string('x', 300);
            Action shortOne = () => np.array(texts, T);
            shortOne.Should().Throw<UnicodeEncodeError>()
                .WithMessage("'utf-8' codec can't encode character '\\udc00' in position 1: surrogates not allowed");
            Action coerce = () => np.array(new object[] { new string('a', 500), new string('b', 50), 3.5 }, np.dtypes.StringDType(coerce: false));
            coerce.Should().Throw<ValueError>().WithMessage("StringDType only allows string data when string coercion is disabled.");
        }

        /// <summary>
        ///     <c>astype(StringDType)</c> of an <c>S</c> array into its fresh result reserves the arena once (capacity equals
        ///     the bytes used) and gives the cells of <c>np.array</c> over the decoded texts (the same sequential appends);
        ///     a strided, a reversed and a broadcast source give the right texts.
        /// </summary>
        [TestMethod]
        public void BytesToStringDType_FreshDestination()
        {
            foreach (int width in new[] { 16, 60, 256, 500 })
            {
                string[] texts = Texts(1500, 0, width, Ascii.Where(c => c != "\0").ToArray(), seed: width);
                using var s = np.array(texts.Select(t => (object)Encoding.ASCII.GetBytes(t)).ToArray(), np.dtype($"S{width}"));
                using var t = s.astype(T);
                var allocator = t.Storage.VStringSlice.Allocator;
                allocator.ArenaCapacity.Should().Be(allocator.ArenaUsed, $"S{width} into a fresh array reserves once");
                using (var reference = np.array(texts, T))
                    t.tobytes().Should().Equal(reference.tobytes(), $"S{width} cells");
                using (var strided = s["::3"].astype(T))
                    NDArray.AsStringArray(strided).Should().Equal(texts.Where((_, i) => i % 3 == 0));
                using (var reversed = s["::-1"].astype(T))
                    NDArray.AsStringArray(reversed).Should().Equal(Enumerable.Reverse(texts));
                using (var broadcast = np.broadcast_to(s["0:7"], new Shape(5, 7)).astype(T))
                    NDArray.AsStringArray(broadcast).Should().Equal(Enumerable.Repeat(texts.Take(7), 5).SelectMany(x => x));
            }
        }

        /// <summary>
        ///     An <c>S</c> → <c>StringDType</c> cast into a destination that already holds text is not a fresh fill: nothing is
        ///     reserved, and a short-text cell sends each longer text to the HEAP (NumPy's <c>heap_or_arena_allocate</c>), where
        ///     an append at the bump pointer would overwrite the cell (and leak a heap string it held). Into an empty array made
        ///     by <c>np.empty</c> (all-zero cells) the cast is a fresh fill and reserves once.
        /// </summary>
        /// <remarks>Every public assignment route (<c>np.copyto</c>, slice, fancy and mask assignment) casts an <c>S</c> value
        ///     into a fresh temporary first and copies that temporary in, so the held destination reaches the cast only through
        ///     the copy core itself, <see cref="NumSharp.Backends.Iteration.NDIter.Copy(UnmanagedStorage, UnmanagedStorage)"/>:
        ///     this test drives it there, and through <c>np.copyto</c> for the public answer.</remarks>
        [TestMethod]
        public void BytesToStringDType_OnlyAFreshDestinationReserves()
        {
            string[] texts = Texts(1000, 20, 60, Ascii.Where(c => c != "\0").ToArray(), seed: 5);
            using var s = np.array(texts.Select(t => (object)Encoding.ASCII.GetBytes(t)).ToArray(), np.dtype("S60"));
            foreach (bool direct in new[] { true, false })
            {
                using var held = np.array(Enumerable.Repeat("ab", 1000).ToArray(), T);
                if (direct)
                    NumSharp.Backends.Iteration.NDIter.Copy(held.Storage, s.Storage);
                else
                    np.copyto(held, s);
                NDArray.AsStringArray(held).Should().Equal(texts);
                held.Storage.VStringSlice.Allocator.ArenaCapacity.Should().Be(0, "short cells send the texts to the heap");
                held.Storage.VStringSlice.Allocator.HeapStrings.Should().Be(1000);
            }
            using (var empty = np.empty(new Shape(1000), T))
            {
                NumSharp.Backends.Iteration.NDIter.Copy(empty.Storage, s.Storage);
                NDArray.AsStringArray(empty).Should().Equal(texts);
                empty.Storage.VStringSlice.Allocator.ArenaCapacity.Should().Be(empty.Storage.VStringSlice.Allocator.ArenaUsed);
            }
        }
    }
}
