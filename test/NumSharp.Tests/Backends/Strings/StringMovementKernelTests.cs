using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The fast routes of fixed-width string MOVEMENT, at sizes and widths the string oracle's small arrays do not reach:
    ///     the dense block copy, the 2-D element-grid copy (<c>GetElementCopyKernel</c>: strided, reversed, transposed,
    ///     broadcast, truncating and NUL-padding), the whole-element <c>np.where</c> (<c>GetWhereBytesKernel</c>), the
    ///     doubling fill (past its 64 KB cache-resident source cap, with widths that do not divide it), and the boolean-mask
    ///     and integer-index gathers that run on the fancy-index kernel route. Every expectation is NumPy's element-by-element
    ///     semantics, computed here from C# strings: a selection picks whole elements, a narrower width cuts, a wider one pads
    ///     with NULs (which reading strips).
    /// </summary>
    [TestClass]
    public class StringMovementKernelTests
    {
        private static readonly System.Random Rng = new System.Random(20261007);

        private static string[] Words(int n, int maxLen)
            => Enumerable.Range(0, n).Select(_ => new string(Enumerable.Range(0, Rng.Next(maxLen + 1))
                                                                       .Select(__ => (char)('a' + Rng.Next(26))).ToArray()))
                         .ToArray();

        /// <summary>Reads every element of a 1-D or N-D string array in C order (U as strings, S decoded as ASCII).</summary>
        private static string[] Read(NDArray a)
        {
            if (a.dtype.kind == 'U')
                return NDArray.AsStringArray(a);
            var flat = a.flatten();
            return Enumerable.Range(0, (int)flat.size)
                             .Select(i => System.Text.Encoding.ASCII.GetString((byte[])flat.item(i)))
                             .ToArray();
        }

        private static string Cut(string s, int chars) => s.Length <= chars ? s : s.Substring(0, chars);

        [TestMethod]
        public void DenseCopy_SameLayout_IsOneBlockMove_IncludingOverlap()
        {
            var words = Words(70_000, 8);
            var a = np.array(words, np.dtype("U8"));
            Read(a.copy()).Should().Equal(words);
            Read(np.asfortranarray(a.reshape(350, 200)).copy('F')).Should().Equal(words);

            // a[1:] = a[:-1]: NumPy's raw 1-D walk reads each element before it is overwritten (a memmove).
            var b = np.array(words.Take(1000).ToArray(), np.dtype("U8"));
            b["1:"] = b[":-1"];
            var expected = new[] { words[0] }.Concat(words.Take(999)).ToArray();
            Read(b).Should().Equal(expected);
        }

        [TestMethod]
        public void ElementGridCopy_StridedReversedTransposed()
        {
            var words = Words(60_000, 8);
            var a = np.array(words, np.dtype("U8"));
            Read(a["::3"].copy()).Should().Equal(words.Where((_, i) => i % 3 == 0));
            Read(a["::-1"].copy()).Should().Equal(Enumerable.Reverse(words));

            var m = a.reshape(300, 200);
            var expectedT = Enumerable.Range(0, 200).SelectMany(c => Enumerable.Range(0, 300).Select(r => words[r * 200 + c]));
            Read(np.ascontiguousarray(m.T)).Should().Equal(expectedT);
        }

        [TestMethod]
        public void ElementGridCopy_WidthChange_CutsAndPads()
        {
            var words = Words(50_000, 8);
            var a = np.array(words, np.dtype("U8"));
            Read(a.astype("U3")).Should().Equal(words.Select(w => Cut(w, 3)));
            var wide = a.astype("U13");
            wide.dtype.str.Should().Be("<U13");
            Read(wide).Should().Equal(words);
            // The padding is NUL units (read strips them): the raw tail of the first element is zero.
            wide.tobytes().Skip(4 * words[0].Length).Take(4 * (13 - words[0].Length)).Should().OnlyContain(b => b == 0);

            var s = a.astype("S8");
            Read(s.astype("S2")).Should().Equal(words.Select(w => Cut(w, 2)));
            Read(s["::-2"].astype("S11")).Should().Equal(words.Where((_, i) => i % 2 == 1).Reverse());
        }

        [TestMethod]
        public void ElementGridCopy_BroadcastSource()
        {
            var row = np.array(new[] { "ab", "cde", "" }, np.dtype("U5"));
            var dst = np.empty(new Shape(4000, 3), np.dtype("U7"));
            np.copyto(dst, row);
            Read(dst).Should().Equal(Enumerable.Repeat(new[] { "ab", "cde", "" }, 4000).SelectMany(x => x));
        }

        [TestMethod]
        public void Where_WholeElements_EveryWidth()
        {
            foreach (var spec in new[] { "S1", "S3", "S16", "S17", "U1", "U5", "U9" })
            {
                var dt = np.dtype(spec);
                int chars = dt.kind == 'U' ? dt.itemsize / 4 : dt.itemsize;
                var xs = Words(5000, chars);
                var ys = Words(5000, chars);
                var cond = Enumerable.Range(0, 5000).Select(_ => Rng.Next(2) == 0).ToArray();
                var r = np.where(np.array(cond), np.array(xs, dt), np.array(ys, dt));
                r.dtype.str.Should().Be(dt.str);
                Read(r).Should().Equal(cond.Select((c, i) => c ? xs[i] : ys[i]), spec);
            }
        }

        [TestMethod]
        public void Fill_PastTheCacheCap_WithAWidthThatDoesNotDivideIt()
        {
            // <U3 is 12 bytes: the doubling fill's 64 KB source cap rounds down to whole elements.
            var a = np.empty(new Shape(100_000), np.dtype("U3"));
            a.fill("xyz12");
            Read(a).Should().OnlyContain(s => s == "xyz");

            var f = np.asfortranarray(np.empty(new Shape(300, 301), np.dtype("S7")));
            f.fill("abc");
            Read(f).Should().OnlyContain(s => s == "abc");

            // A strided view takes the copy route and leaves the other elements alone.
            var b = np.array(Enumerable.Repeat("q", 9000).ToArray(), np.dtype("U2"));
            b["::3"].fill("zz");
            Read(b).Select((s, i) => (s, i)).Should().OnlyContain(t => t.s == (t.i % 3 == 0 ? "zz" : "q"));
        }

        [TestMethod]
        public void MaskAndIndexGathers_RunWholeRows()
        {
            var words = Words(6000, 6);
            var a = np.array(words, np.dtype("U6"));
            var mask = Enumerable.Range(0, 6000).Select(_ => Rng.Next(3) == 0).ToArray();
            Read(a[np.array(mask).MakeGeneric<bool>()]).Should().Equal(words.Where((_, i) => mask[i]));
            Read(a[(object)np.array(mask)]).Should().Equal(words.Where((_, i) => mask[i]));

            // A mask over the leading axis of a 2-D string array keeps whole rows.
            var m = a.reshape(60, 100);
            var rowMask = Enumerable.Range(0, 60).Select(r => r % 7 == 0).ToArray();
            var picked = m[np.array(rowMask).MakeGeneric<bool>()];
            picked.Should().BeShaped(rowMask.Count(x => x), 100);
            Read(picked).Should().Equal(Enumerable.Range(0, 60).Where(r => rowMask[r]).SelectMany(r => words.Skip(r * 100).Take(100)));

            // Integer indices with negatives and repeats.
            var idx = new long[] { 5, -1, 5, 0, 5999, -6000 };
            Read(a[np.array(idx)]).Should().Equal(idx.Select(i => words[i < 0 ? i + 6000 : i]));
        }
    }
}
