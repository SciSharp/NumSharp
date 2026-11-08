using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Text;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     <c>ndarray.tolist()</c> as NumPy's <c>recursive_tolist</c>: one list per axis position, walked by the strides from
    ///     the array's first element, each innermost row read by one <see cref="IArraySlice.GetItems"/> call. The reference
    ///     is the walk it replaced — a sub-array view per outer index and <c>item(i)</c> per element of a row — over every
    ///     storage (the 15 numeric dtypes, <c>S</c>, <c>U</c>, <c>StringDType</c> with and without an NA object) and every
    ///     layout kind (strided, reversed, transposed, broadcast, offset views, 0-d views, empty axes, five dimensions).
    /// </summary>
    [TestClass]
    public class TolistWalkTests
    {
        /// <summary>The walk tolist replaced: a sub-array view per outer index, <c>item(i)</c> per element of a 1-D row.</summary>
        /// <param name="a">The array.</param>
        /// <returns>Its nested lists (the scalar for a 0-d array).</returns>
        private static object Reference(NDArray a)
        {
            if (a.ndim == 0)
                return a.Storage.GetAtIndex(0);
            var list = new List<object>();
            if (a.ndim == 1)
            {
                for (long i = 0; i < a.size; i++)
                    list.Add(a.item(i));
                return list;
            }
            for (long i = 0; i < a.shape[0]; i++)
                list.Add(Reference(a[i]));
            return list;
        }

        /// <summary>
        ///     A canonical rendering of a tolist result: every leaf with its CLR type and raw bits (a NaN payload or a -0.0 is
        ///     visible), every list with its type and its items, a text with each non-printable char escaped.
        /// </summary>
        /// <param name="o">A leaf or a list.</param>
        /// <returns>The rendering.</returns>
        private static string Render(object o)
        {
            switch (o)
            {
                case null: return "None";
                case string s: return "s'" + string.Concat(s.Select(ch => ch >= 0x20 && ch < 0x7F && ch != '\'' && ch != '\\' ? ch.ToString() : "\\u" + ((int)ch).ToString("x4"))) + "'";
                case byte[] b: return "b'" + Convert.ToHexString(b) + "'";
                case double d: return "f8:" + BitConverter.DoubleToInt64Bits(d).ToString("x16");
                case float f: return "f4:" + BitConverter.SingleToInt32Bits(f).ToString("x8");
                case Half h: return "f2:" + BitConverter.HalfToUInt16Bits(h).ToString("x4");
                case Complex c: return "c16:" + BitConverter.DoubleToInt64Bits(c.Real).ToString("x16") + "," + BitConverter.DoubleToInt64Bits(c.Imaginary).ToString("x16");
                case decimal m: return "m:" + m.ToString(CultureInfo.InvariantCulture);
                case char ch: return "c:" + ((int)ch).ToString("x4");
                case IList l:
                {
                    var parts = new List<string>();
                    foreach (var x in l)
                        parts.Add(Render(x));
                    return o.GetType().Name + "[" + string.Join(",", parts) + "]";
                }
                default: return o.GetType().Name + ":" + Convert.ToString(o, CultureInfo.InvariantCulture);
            }
        }

        /// <summary>Layout recipes over a C-ordered owner of 24 elements.</summary>
        private static readonly (string Name, Func<NDArray, NDArray> Make)[] Layouts =
        {
            ("c1", a => a),
            ("c2", a => a.reshape(4, 6)),
            ("c3", a => a.reshape(2, 3, 4)),
            ("f2", a => a.reshape(6, 4).T),
            ("t3", a => np.transpose(a.reshape(2, 3, 4), new[] { 2, 0, 1 })),
            ("s1", a => a["1::3"]),
            ("r1", a => a["::-1"]),
            ("r2", a => a.reshape(4, 6)["::-1, ::-2"]),
            ("o1", a => a["5:17"]),
            ("o2", a => a.reshape(4, 6)["1:3, 2:5"]),
            ("col", a => a.reshape(4, 6)[":, 3"]),
            ("bc", a => np.broadcast_to(a.reshape(1, 24)[":, :5"], new Shape(3, 5))),
            ("bc0", a => np.broadcast_to(a["7:8"], new Shape(4))),
            ("z0", a => a.reshape(4, 6)[2, 3]),
            ("e1", a => a["3:3"]),
            ("e2", a => a.reshape(4, 6)[":, 2:2"]),
            ("e3", a => a.reshape(4, 6)["2:2, :"]),
            ("n4", a => a.reshape(2, 1, 3, 4)),
            ("n5", a => a.reshape(1, 2, 1, 3, 4)[":, :, :, ::-1, ::2"]),
        };

        /// <summary>The 15 numeric dtypes as 24-element owners holding edge values (NaN payloads, -0.0, extremes).</summary>
        private static IEnumerable<(string Name, Func<NDArray> Make)> Numeric()
        {
            yield return ("b1", () => np.array(Enumerable.Range(0, 24).Select(i => i % 3 == 0).ToArray()));
            yield return ("i1", () => np.array(Enumerable.Range(0, 24).Select(i => (sbyte)(i * 11 - 128)).ToArray()));
            yield return ("u1", () => np.array(Enumerable.Range(0, 24).Select(i => (byte)(i * 11)).ToArray()));
            yield return ("i2", () => np.array(Enumerable.Range(0, 24).Select(i => (short)(i * 2731 - 32768)).ToArray()));
            yield return ("u2", () => np.array(Enumerable.Range(0, 24).Select(i => (ushort)(i * 2731)).ToArray()));
            yield return ("i4", () => np.array(Enumerable.Range(0, 24).Select(i => i * 178956970 - int.MaxValue).ToArray()));
            yield return ("u4", () => np.array(Enumerable.Range(0, 24).Select(i => (uint)i * 178956970u).ToArray()));
            yield return ("i8", () => np.array(Enumerable.Range(0, 24).Select(i => (long)i * 768614336404564650L - long.MaxValue).ToArray()));
            yield return ("u8", () => np.array(Enumerable.Range(0, 24).Select(i => (ulong)i * 768614336404564650UL).ToArray()));
            yield return ("ch", () => np.array(Enumerable.Range(0, 24).Select(i => (char)(0x41 + i * 1000)).ToArray()));
            yield return ("f2", () => np.array(Enumerable.Range(0, 24).Select(i => i == 3 ? Half.NaN : i == 4 ? Half.NegativeZero : (Half)(i * 1.37 - 9)).ToArray()));
            yield return ("f4", () => np.array(Enumerable.Range(0, 24).Select(i => i == 3 ? float.NaN : i == 4 ? -0.0f : i == 5 ? float.PositiveInfinity : (float)(i * 1.37 - 9)).ToArray()));
            yield return ("f8", () => np.array(Enumerable.Range(0, 24).Select(i => i == 3 ? BitConverter.Int64BitsToDouble(0x7ff8000000000123) : i == 4 ? -0.0 : i * 1.37 - 9).ToArray()));
            yield return ("m16", () => np.array(Enumerable.Range(0, 24).Select(i => i * 1.10m - 3.300m).ToArray()));
            yield return ("c16", () => np.array(Enumerable.Range(0, 24).Select(i => new Complex(i * 1.5, i == 6 ? double.NaN : -i)).ToArray()));
        }

        /// <summary>Texts mixing ASCII, BMP non-ASCII, an astral code point, NULs inside and lone surrogates (for <c>U</c>).</summary>
        /// <param name="seed">The seed.</param>
        /// <param name="lone">Whether lone surrogates may occur (a <c>U</c> element holds one; UTF-8 cannot).</param>
        /// <param name="maxLen">The longest text in code points.</param>
        /// <returns>24 texts, two of them empty.</returns>
        private static string[] Texts(int seed, bool lone, int maxLen)
        {
            var rng = new System.Random(seed);
            const string pool = "abcXYZ019 éñ中文ΑαЖ";
            var texts = new string[24];
            for (int i = 0; i < 24; i++)
            {
                int len = rng.Next(0, maxLen + 1);
                var sb = new StringBuilder();
                for (int k = 0; k < len; k++)
                {
                    int r = rng.Next(100);
                    if (r < 5) sb.Append("😀");
                    else if (r < 7 && lone) sb.Append('\uD800');
                    else if (r < 8) sb.Append('\0');
                    else sb.Append(pool[rng.Next(pool.Length)]);
                }
                texts[i] = sb.ToString();
            }
            texts[5] = "";
            texts[6] = lone ? "\0\0" : "";
            return texts;
        }

        /// <summary>Asserts tolist of <paramref name="a"/> renders as the reference walk does.</summary>
        /// <param name="a">The array.</param>
        /// <param name="because">The case's name.</param>
        private static void AssertWalk(NDArray a, string because)
            => Render(a.tolist()).Should().Be(Render(Reference(a)), because);

        /// <summary>Every numeric dtype, every layout kind, plus 0-d owners and empty shapes.</summary>
        [TestMethod]
        public void Tolist_EqualsTheElementWalk_NumericDtypes()
        {
            foreach (var (dn, make) in Numeric())
            {
                foreach (var (ln, lay) in Layouts)
                    AssertWalk(lay(make()), $"{dn}/{ln}");
                AssertWalk(make()[5].copy(), $"{dn}/0-d owner");
                AssertWalk(make()["0:0"].reshape(0, 3), $"{dn}/(0, 3)");
                AssertWalk(make()["0:0"].reshape(2, 0), $"{dn}/(2, 0)");
                AssertWalk(make()["0:0"].reshape(2, 0, 3), $"{dn}/(2, 0, 3)");
            }
        }

        /// <summary>
        ///     <c>S</c> and <c>U</c> elements of widths around the readers' vector steps (NULs inside, trailing NULs, all-NUL
        ///     and empty values, high bytes, astral code points and lone surrogates), every layout kind.
        /// </summary>
        [TestMethod]
        public void Tolist_EqualsTheElementWalk_FixedWidthText()
        {
            var rng = new System.Random(77);
            var bytes = Enumerable.Range(0, 24).Select(i =>
            {
                var v = new byte[rng.Next(0, 40)];
                for (int k = 0; k < v.Length; k++)
                    v[k] = (byte)rng.Next(1, 256);
                if (v.Length > 3 && i % 4 == 0)
                    v[v.Length / 2] = 0;
                return (object)v;
            }).ToArray();
            bytes[5] = new byte[0];
            bytes[6] = new byte[] { 0, 0, 0 };
            foreach (int w in new[] { 1, 7, 8, 15, 16, 17, 33, 64, 65, 300 })
                foreach (var (ln, lay) in Layouts)
                    AssertWalk(lay(np.array(bytes, np.dtype($"S{w}"))), $"S{w}/{ln}");
            string[] texts = Texts(78, lone: true, maxLen: 45);
            foreach (int w in new[] { 1, 7, 8, 9, 15, 16, 17, 24, 25, 33, 48, 300 })
                foreach (var (ln, lay) in Layouts)
                    AssertWalk(lay(np.array(texts, np.dtype($"U{w}"))), $"U{w}/{ln}");
        }

        /// <summary>
        ///     <c>StringDType</c> elements of every size class (in the cell, medium and long arena texts, empty) with no NA
        ///     object and with the None, NaN and string NA objects, every layout kind: an NA cell reads as the descriptor's NA
        ///     object, the same object for every NA cell.
        /// </summary>
        [TestMethod]
        public void Tolist_EqualsTheElementWalk_StringDType()
        {
            var rng = new System.Random(79);
            string[] texts = Texts(80, lone: false, maxLen: 12);
            for (int i = 0; i < 24; i += 3)
                texts[i] = string.Concat(Enumerable.Repeat("medium text é ", rng.Next(2, 5)));
            for (int i = 1; i < 24; i += 5)
                texts[i] = string.Concat(Enumerable.Repeat("long text 中 😀 ", rng.Next(20, 40)));
            foreach (var (tn, dt, na) in new (string, DType, object)[]
            {
                ("T", np.dtypes.StringDType(), null),
                ("Tnone", np.dtypes.StringDType(null), null),
                ("Tnan", np.dtypes.StringDType(double.NaN), double.NaN),
                ("Tstr", np.dtypes.StringDType("NA"), "NA"),
            })
            {
                object[] values = texts.Select((s, i) => tn != "T" && i % 5 == 2 ? na : (object)s).ToArray();
                foreach (var (ln, lay) in Layouts)
                    AssertWalk(lay(np.array(values, dt)), $"{tn}/{ln}");
            }
        }

        /// <summary>
        ///     An empty axis gives empty lists at its level and the levels above keep their lengths, for every storage — the
        ///     innermost rows of an empty array are never read.
        /// </summary>
        [TestMethod]
        public void Tolist_EmptyAxes_AllStorages()
        {
            foreach (var dt in new DType[] { np.float64, np.dtype("S3"), np.dtype("U3"), np.dtypes.StringDType() })
            {
                using var e0 = np.empty(new Shape(0), dt);
                ((List<object>)e0.tolist()).Should().BeEmpty();
                using var e20 = np.empty(new Shape(2, 0), dt);
                Render(e20.tolist()).Should().Be("List`1[List`1[],List`1[]]", dt.ToString());
                using var e03 = np.empty(new Shape(0, 3), dt);
                Render(e03.tolist()).Should().Be("List`1[]", dt.ToString());
                using var e203 = np.empty(new Shape(2, 0, 3), dt);
                Render(e203.tolist()).Should().Be("List`1[List`1[],List`1[]]", dt.ToString());
            }
        }

        /// <summary>
        ///     A <c>U</c> element with a unit past U+10FFFF raises CPython's SystemError from tolist wherever it is (first row,
        ///     a later row of a transposed view), and item() raises for that element alone.
        /// </summary>
        [TestMethod]
        public unsafe void Tolist_RaisesAtABadUnicodeElement()
        {
            foreach (int bad in new[] { 0, 9, 23 })
            {
                using var a = np.array(Texts(81, lone: false, maxLen: 6).Select(s => s.Length == 0 ? "x" : s).ToArray(), np.dtype("U8"));
                uint* p = (uint*)a.Storage.Address;
                p[bad * 8L] = 0x110000;
                Action c1 = () => a.tolist();
                c1.Should().Throw<SystemError>().WithMessage("invalid maximum character passed to PyUnicode_New");
                Action t2 = () => a.reshape(4, 6).T.tolist();
                t2.Should().Throw<SystemError>().WithMessage("invalid maximum character passed to PyUnicode_New");
                Action item = () => a.item(bad);
                item.Should().Throw<SystemError>();
                a.item((bad + 1) % 24).Should().BeOfType<string>();
            }
        }

        /// <summary>
        ///     A corrupt <c>StringDType</c> cell (an arena text in a buffer that has no arena) raises NumPy's <c>Failed to load
        ///     string in StringDType getitem</c> (MemoryError) from tolist, as item() does for that element.
        /// </summary>
        [TestMethod]
        public unsafe void Tolist_RaisesAtACorruptStringDTypeCell()
        {
            using var a = np.array(new[] { "a", "b", "c", "d" }, np.dtypes.StringDType());
            byte* cell = a.Storage.Address + 2 * 16;
            var saved = new byte[16];
            new ReadOnlySpan<byte>(cell, 16).CopyTo(saved);
            try
            {
                // An initialized arena text of 5 bytes (no short / missing / outside-arena flag) while the buffer has no arena.
                new Span<byte>(cell, 16).Clear();
                cell[8] = 5;
                cell[15] = 0x40;
                Action list = () => a.tolist();
                list.Should().Throw<OutOfMemoryException>().WithMessage("Failed to load string in StringDType getitem");
                Action item = () => a.item(2);
                item.Should().Throw<OutOfMemoryException>().WithMessage("Failed to load string in StringDType getitem");
                a.item(1).Should().Be("b");
            }
            finally
            {
                // The buffer's clear loop must not meet the corrupt cell.
                saved.AsSpan().CopyTo(new Span<byte>(cell, 16));
            }
        }

        /// <summary>
        ///     <see cref="IArraySlice.GetItems"/> gives <see cref="IArraySlice.GetIndex(long)"/>'s value at every slot of
        ///     forward, stepped, backward, repeated (stride 0) and empty runs, for every storage.
        /// </summary>
        [TestMethod]
        public void GetItems_EqualsGetIndex_EveryStorageAndRun()
        {
            var owners = Numeric().Select(n => (n.Name, n.Make())).ToList();
            owners.Add(("S9", np.array(Texts(82, lone: false, maxLen: 9).Select(s => (object)Encoding.UTF8.GetBytes(s)).ToArray(), np.dtype("S9"))));
            owners.Add(("U17", np.array(Texts(83, lone: true, maxLen: 17), np.dtype("U17"))));
            owners.Add(("T", np.array(Texts(84, lone: false, maxLen: 30).Select((s, i) => i % 7 == 3 ? null : (object)s).ToArray(), np.dtypes.StringDType(null))));
            foreach (var (name, a) in owners)
            {
                IArraySlice slice = a.Storage.InternalArray;
                foreach (var (start, stride, count) in new (long, long, int)[] { (0, 1, 24), (3, 3, 7), (23, -1, 24), (22, -2, 11), (5, 0, 4), (0, 1, 0), (11, 1, 1) })
                {
                    var got = new object[count];
                    slice.GetItems(start, stride, got);
                    for (int i = 0; i < count; i++)
                        Render(got[i]).Should().Be(Render(slice.GetIndex(start + i * stride)), $"{name} run ({start}, {stride}, {count}) slot {i}");
                }
                a.Dispose();
            }
        }
    }
}
