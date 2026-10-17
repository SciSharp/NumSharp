using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     NumPy's <c>np.strings.translate</c> (strings phase 2, P2.6 milestone 3): every element form against the managed
    ///     reference (<see cref="StrTranslateTable.TranslateInto"/> / <see cref="BytesTranslateTable.TranslateInto"/>, the
    ///     computation of the C-order walk), the forms against inaccessible pages, the IL kernels against the managed ones,
    ///     the kernel route against the walk route, and the public function against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The string oracle (sections <c>strtrans</c> / <c>ttrans</c>) checks the public function against NumPy over its
    ///     operands; these tests check what the operands do not reach: every <c>S</c> / <c>U</c> width with each route forced
    ///     (the gather path, the whole-width scalar loop, the exact path), a dense range cut short so the managed table
    ///     answers, and each element form at the edge of mapped memory.
    ///     </para>
    ///     <para>Every test is deterministic (fixed seeds) and touches no process-wide switch, so they run in parallel.</para>
    /// </remarks>
    [TestClass]
    public unsafe class StringTranslateTests
    {
        /// <summary>
        ///     Code points the texts are drawn from: ASCII letters the tables map, space and NUL, Latin-1, two- and three-byte
        ///     BMP characters, a lone surrogate and astral characters.
        /// </summary>
        private static readonly uint[] Pool =
        {
            'a', 'b', 'c', 'd', 'e', 'f', 'x', 'y', 'z', 'A', 'Q', ' ', 0, 0xE9, 0xDF, 0xF1, 0x4E2D, 0x6587, 0xD800, 0x10400,
            0x10401, 0x1F600,
        };

        /// <summary>Units a <c>U</c> element can hold that are not text (past U+10FFFF), keyed or not by the tables.</summary>
        private static readonly uint[] UOnly = { 0x110000, 0x7FFFFFF0, 0xFFFF_FFF0 };

        /// <summary>
        ///     The str.translate tables the element tests run (each compiled with a full and a cut dense range): one-to-one,
        ///     deletions, replacements that grow, non-ASCII keys and values, NUL mapped and produced, keys past U+10FFFF, a
        ///     long str table, a bytes table, a numeric table, an unsubscriptable one, and values that raise.
        /// </summary>
        private static IEnumerable<(string Name, Func<int, StrTranslateTable> Make)> StrTables()
        {
            yield return ("mk", d => StrTranslateTable.FromDict(Dict(97, 117, 98, 118, 99, 119, 100, 120, 101, 121, 102, 122), d));
            yield return ("mkdel", d => StrTranslateTable.FromDict(Dict(97, 120, 98, 121, 99, 122, 100, null, 101, null, 102, null), d));
            yield return ("exp", d => StrTranslateTable.FromDict(Dict(97, "XYZ", 98, "Q", 99, ""), d));
            yield return ("uni", d => StrTranslateTable.FromDict(Dict(0xE9, "e", 0x4E2D, "Z", 97, "á", 0x10400, "\U00010428"), d));
            yield return ("nul", d => StrTranslateTable.FromDict(Dict(0, "Z"), d));
            yield return ("toNul", d => StrTranslateTable.FromDict(Dict(97, 0L), d));
            yield return ("past", d => StrTranslateTable.FromDict(Dict(0x110000, "p", 0x7FFFFFF0, null), d));
            yield return ("surr", d => StrTranslateTable.FromDict(Dict(97, 0xDC00, 0xD800, "q"), d));
            yield return ("seq", d => StrTranslateTable.FromCodePoints(Enumerable.Range(0, 300).Select(i => (uint)('k' + i % 7)).ToArray(), d));
            yield return ("bytes", d => StrTranslateTable.FromBytes(Enumerable.Range(0, 150).Select(i => (byte)(i % 3 == 0 ? 0 : 'm' + i % 5)).ToArray(), d));
            yield return ("keep", d => StrTranslateTable.KeepAll(d));
            yield return ("none", _ => StrTranslateTable.Unsubscriptable("NoneType"));
            yield return ("err", d => StrTranslateTable.FromDict(Dict(98, 3.5, 0x4E2D, -1), d));
        }

        /// <summary>A dictionary from alternating keys and values (the C# map of a Python dict literal).</summary>
        /// <param name="kv">Key, value, key, value, ...</param>
        /// <returns>The dictionary.</returns>
        private static Dictionary<object, object> Dict(params object[] kv)
        {
            var d = new Dictionary<object, object>();
            for (int i = 0; i < kv.Length; i += 2)
                d[kv[i]] = kv[i + 1];
            return d;
        }

        /// <summary>A random text of <paramref name="n"/> code points, sometimes ending in NULs, for <c>U</c> sometimes holding
        ///     a unit past U+10FFFF.</summary>
        /// <param name="rng">The generator.</param>
        /// <param name="n">The code points.</param>
        /// <param name="forU">Whether units past U+10FFFF may appear.</param>
        /// <returns>The code points.</returns>
        private static uint[] RandomText(System.Random rng, int n, bool forU)
        {
            var t = new uint[n];
            for (int i = 0; i < n; i++)
                t[i] = Pool[rng.Next(Pool.Length)];
            if (forU && n > 0 && rng.Next(12) == 0)
                t[rng.Next(n)] = UOnly[rng.Next(UOnly.Length)];
            if (n > 0 && rng.Next(4) == 0)
                for (int k = 0, z = rng.Next(1, 4); k < z && k < n; k++)
                    t[n - 1 - k] = 0;
            return t;
        }

        /// <summary>Writes code points as an <c>S</c> element (a code point past 0xFF becomes a byte at or past 0x80) or a
        ///     <c>U</c> element.</summary>
        /// <param name="text">The code points.</param>
        /// <param name="p">The element.</param>
        /// <param name="unit">1 or 4.</param>
        private static void Write(uint[] text, byte* p, int unit)
        {
            for (int i = 0; i < text.Length; i++)
            {
                if (unit == 4)
                    ((uint*)p)[i] = text[i];
                else
                    p[i] = text[i] < 0x100 ? (byte)text[i] : (byte)(0x80 | (text[i] & 0x7F));
            }
        }

        /// <summary>
        ///     The reference of one <c>U</c> element (the walk's computation): the <c>np.str_</c> without trailing NULs (one
        ///     unit past U+10FFFF raises at the scalar), translated by the managed table, cut to the width and zero-padded.
        /// </summary>
        /// <param name="units">The element's units.</param>
        /// <param name="table">The table.</param>
        /// <param name="raises">Receives whether the element raises.</param>
        /// <returns>The output units (null when it raises).</returns>
        private static uint[] ReferenceU(uint[] units, StrTranslateTable table, out bool raises)
        {
            int n = units.Length;
            while (n > 0 && units[n - 1] == 0)
                n--;
            if (n == 1 && units[0] > 0x10FFFF)
            {
                raises = true;
                return null;
            }
            var dst = new uint[units.Length];
            raises = table.TranslateInto(units.AsSpan(0, n), dst, out _) is not null;
            return raises ? null : dst;
        }

        /// <summary>The reference of one <c>S</c> element: the bytes without trailing NULs, each mapped or deleted, the rest
        ///     zero.</summary>
        /// <param name="bytes">The element's bytes.</param>
        /// <param name="table">The table.</param>
        /// <returns>The output bytes.</returns>
        private static byte[] ReferenceS(byte[] bytes, BytesTranslateTable table)
        {
            int n = bytes.Length;
            while (n > 0 && bytes[n - 1] == 0)
                n--;
            var dst = new byte[bytes.Length];
            table.TranslateInto(bytes.AsSpan(0, n), dst);
            return dst;
        }

        /// <summary>Runs the <c>U</c> element body with a prepared state whose flags are cleared by <paramref name="clear"/>
        ///     (forcing the scalar loop or the exact path).</summary>
        /// <param name="table">The table.</param>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="clear">The flags to clear.</param>
        /// <returns>The error word.</returns>
        private static long RunU(StrTranslateTable table, byte* src, int width, byte* dst, TranslateFlags clear)
        {
            var st = default(TranslateState);
            table.Prepare(&st);
            st.Flags &= ~(int)clear;
            var handle = GCHandle.Alloc(table);
            try
            {
                st.Table = GCHandle.ToIntPtr(handle);
                NDTextElements.Translate<uint>(src, width, dst, &st);
            }
            finally
            {
                handle.Free();
            }
            return st.Status;
        }

        /// <summary>Runs the <c>S</c> element body with a prepared state whose flags are cleared by <paramref name="clear"/>.</summary>
        /// <param name="table">The table.</param>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="clear">The flags to clear.</param>
        private static void RunS(BytesTranslateTable table, byte* src, int width, byte* dst, TranslateFlags clear)
        {
            var st = default(TranslateState);
            table.Prepare(&st);
            st.Flags &= ~(int)clear;
            NDTextElements.Translate<byte>(src, width, dst, &st);
        }

        /// <summary>
        ///     The bytes.translate tables the element tests run: a maketrans map, deletions, a table mapping NUL to 'Z' (its
        ///     trailing NULs must not be translated: no NulPreserved), NUL deleted, the identity.
        /// </summary>
        private static IEnumerable<(string Name, BytesTranslateTable Table)> BytesTables()
        {
            var mk = Enumerable.Range(0, 256).Select(i => (byte)i).ToArray();
            mk['a'] = (byte)'u';
            mk['b'] = (byte)'v';
            mk[0xE9] = (byte)'E';
            var nul = (byte[])mk.Clone();
            nul[0] = (byte)'Z';
            yield return ("mk", BytesTranslateTable.Create(mk, true, ReadOnlySpan<byte>.Empty));
            yield return ("mkdel", BytesTranslateTable.Create(mk, true, new byte[] { (byte)'c', (byte)' ', 0xF1 }));
            yield return ("nulmap", BytesTranslateTable.Create(nul, true, ReadOnlySpan<byte>.Empty));
            yield return ("nuldel", BytesTranslateTable.Create(ReadOnlySpan<byte>.Empty, false, new byte[] { 0, (byte)'a' }));
            yield return ("none", BytesTranslateTable.Create(ReadOnlySpan<byte>.Empty, false, ReadOnlySpan<byte>.Empty));
        }

        // =============================================================================================================
        // The element forms against the managed reference
        // =============================================================================================================

        /// <summary>
        ///     Every <c>U</c> width from 1 to 72 units and every <c>S</c> width from 1 to 72 bytes, each table with a full and a
        ///     cut dense range (the managed table answers past it), each route forced in turn — the prepared flags (the gather
        ///     path where it applies), the scalar loop (VectorOk cleared), the exact path (NulPreserved cleared too): every
        ///     output equals the reference, and the error word is set exactly when the reference raises.
        /// </summary>
        [TestMethod]
        public void Elements_EqualTheReference_EveryWidthAndRoute()
        {
            var rng = new System.Random(2630);
            var clears = new[] { TranslateFlags.None, TranslateFlags.VectorOk, TranslateFlags.VectorOk | TranslateFlags.NulPreserved };
            var strTables = StrTables().SelectMany(t => new[]
            {
                (t.Name + "/full", t.Make(StrTranslateTable.MaxDenseLength)),
                (t.Name + "/256", t.Make(256)),
            }).ToArray();
            for (int units = 1; units <= 72; units++)
            {
                int width = units * 4;
                var src = new byte[width];
                var dst = new byte[width + 16];
                for (int rep = 0; rep < 6; rep++)
                {
                    var text = RandomText(rng, units, forU: true);
                    fixed (byte* s = src)
                        Write(text, s, 4);
                    var element = new uint[units];
                    Buffer.BlockCopy(src, 0, element, 0, width);
                    foreach (var (name, table) in strTables)
                    {
                        var want = ReferenceU(element, table, out bool raises);
                        foreach (var clear in clears)
                        {
                            Array.Fill(dst, (byte)0xAB);
                            long status;
                            fixed (byte* s = src, d = dst)
                                status = RunU(table, s, width, d, clear);
                            Assert.AreEqual(raises ? 1L : 0L, status, $"U{units} {name} clear={clear}: error word");
                            if (!raises)
                            {
                                var got = new uint[units];
                                Buffer.BlockCopy(dst, 0, got, 0, width);
                                got.Should().Equal(want, $"U{units} {name} clear={clear}");
                            }
                            // Nothing past the output element is touched (raising elements included).
                            dst.Skip(width).Should().OnlyContain(b => b == 0xAB, $"U{units} {name} clear={clear}: tail");
                        }
                    }
                }
            }
            for (int width = 1; width <= 72; width++)
            {
                var src = new byte[width];
                var dst = new byte[width + 16];
                for (int rep = 0; rep < 6; rep++)
                {
                    fixed (byte* s = src)
                        Write(RandomText(rng, width, forU: false), s, 1);
                    foreach (var (name, table) in BytesTables())
                    {
                        var want = ReferenceS(src, table);
                        foreach (var clear in new[] { TranslateFlags.None, TranslateFlags.NulPreserved })
                        {
                            Array.Fill(dst, (byte)0xAB);
                            fixed (byte* s = src, d = dst)
                                RunS(table, s, width, d, clear);
                            dst.Take(width).Should().Equal(want, $"S{width} {name} clear={clear}");
                            dst.Skip(width).Should().OnlyContain(b => b == 0xAB, $"S{width} {name} clear={clear}: tail");
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     Every element form over elements that end at an inaccessible page or start right after one: an out-of-bounds
        ///     read or write ends the test host. The values are checked on the way.
        /// </summary>
        [TestMethod]
        public void Forms_StayInsideTheirElements()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            var rng = new System.Random(2631);
            using var srcFence = new FencedMemory(pages: 1);
            using var dstFence = new FencedMemory(pages: 1);
            var mk = StrTranslateTable.FromDict(Dict(97, 117, 98, 118, 0xE9, 0x4E2D), StrTranslateTable.MaxDenseLength);
            var del = StrTranslateTable.FromDict(Dict(97, null, 98, "xyz"), StrTranslateTable.MaxDenseLength);
            var bytesTables = BytesTables().Select(t => t.Table).ToArray();
            foreach (bool atEnd in new[] { true, false })
            {
                for (int units = 1; units <= 72; units++)
                {
                    int width = units * 4;
                    byte* s = srcFence.Place(width, atEnd);
                    byte* d = dstFence.Place(width, atEnd);
                    for (int rep = 0; rep < 4; rep++)
                    {
                        var text = RandomText(rng, units, forU: true);
                        Write(text, s, 4);
                        foreach (var table in new[] { mk, del })
                        {
                            var want = ReferenceU(text, table, out bool raises);
                            foreach (var clear in new[] { TranslateFlags.None, TranslateFlags.VectorOk, TranslateFlags.VectorOk | TranslateFlags.NulPreserved })
                            {
                                long status = RunU(table, s, width, d, clear);
                                Assert.AreEqual(raises ? 1L : 0L, status);
                                if (!raises)
                                    new ReadOnlySpan<uint>(d, units).ToArray().Should().Equal(want);
                            }
                        }
                    }
                }
                for (int width = 1; width <= 72; width++)
                {
                    byte* s = srcFence.Place(width, atEnd);
                    byte* d = dstFence.Place(width, atEnd);
                    var bytes = new byte[width];
                    for (int i = 0; i < width; i++)
                        bytes[i] = (byte)rng.Next(256);
                    bytes.AsSpan().CopyTo(new Span<byte>(s, width));
                    foreach (var table in bytesTables)
                        foreach (var clear in new[] { TranslateFlags.None, TranslateFlags.NulPreserved })
                        {
                            RunS(table, s, width, d, clear);
                            new ReadOnlySpan<byte>(d, width).ToArray().Should().Equal(ReferenceS(bytes, table));
                        }
                }
            }
        }

        // =============================================================================================================
        // The IL kernels against the managed kernels
        // =============================================================================================================

        /// <summary>
        ///     The IL shell's translate kernels (constant width, the body inlined) against the managed kernels of hosts without
        ///     dynamic code, over 300 elements: the outputs and the error words are equal.
        /// </summary>
        [TestMethod]
        public void ILKernels_EqualManagedKernels()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("No dynamic code on this host: the managed kernels are the only ones.");
            const int count = 300;
            foreach (int unit in new[] { 1, 4 })
            {
                foreach (int units in new[] { 1, 3, 8, 9, 17, 33, 64, 70 })
                {
                    int width = units * unit;
                    var rng = new System.Random(2632 + width);
                    var text = new byte[count * width];
                    fixed (byte* t = text)
                        for (int i = 0; i < count; i++)
                            Write(RandomText(rng, units, forU: unit == 4), t + (long)i * width, unit);
                    var key = new TextMethodKey(TextMethodOp.Translate, unit, width, width);
                    if (unit == 4)
                    {
                        foreach (var (name, make) in StrTables())
                        {
                            var table = make(StrTranslateTable.MaxDenseLength);
                            var il = RunKernel(ILKernelGenerator.GetTextMethodKernel(key), text, width, count, table, null, out long ilStatus);
                            var managed = RunKernel(NDTextLoop.ManagedKernel(key, masked: false), text, width, count, table, null, out long managedStatus);
                            ilStatus.Should().Be(managedStatus, $"status of {key} {name}");
                            if (ilStatus == 0)
                                il.Should().Equal(managed, $"IL vs managed: {key} {name}");
                        }
                    }
                    else
                    {
                        foreach (var (name, table) in BytesTables())
                        {
                            var il = RunKernel(ILKernelGenerator.GetTextMethodKernel(key), text, width, count, null, table, out _);
                            var managed = RunKernel(NDTextLoop.ManagedKernel(key, masked: false), text, width, count, null, table, out _);
                            il.Should().Equal(managed, $"IL vs managed: {key} {name}");
                        }
                    }
                }
            }
        }

        /// <summary>Runs a translate kernel over contiguous elements into a contiguous output with a prepared state.</summary>
        /// <param name="kernel">The kernel.</param>
        /// <param name="text">The elements.</param>
        /// <param name="width">Their width.</param>
        /// <param name="count">Their count.</param>
        /// <param name="strTable">The str table (a <c>U</c> kernel), or null.</param>
        /// <param name="bytesTable">The bytes table (an <c>S</c> kernel), or null.</param>
        /// <param name="status">Receives the error word.</param>
        /// <returns>The output bytes.</returns>
        private static byte[] RunKernel(NDInnerLoopFunc kernel, byte[] text, int width, int count, StrTranslateTable strTable,
                                        BytesTranslateTable bytesTable, out long status)
        {
            Assert.IsNotNull(kernel);
            var output = new byte[text.Length];
            var st = default(TranslateState);
            GCHandle handle = default;
            if (strTable is not null)
            {
                strTable.Prepare(&st);
                handle = GCHandle.Alloc(strTable);
                st.Table = GCHandle.ToIntPtr(handle);
            }
            else
            {
                bytesTable.Prepare(&st);
            }
            try
            {
                fixed (byte* t = text, o = output)
                {
                    void** ptrs = stackalloc void*[2];
                    long* strides = stackalloc long[2];
                    ptrs[0] = t;
                    ptrs[1] = o;
                    strides[0] = width;
                    strides[1] = width;
                    kernel(ptrs, strides, count, &st);
                }
            }
            finally
            {
                if (handle.IsAllocated)
                    handle.Free();
            }
            status = st.Status;
            return output;
        }

        // =============================================================================================================
        // The kernel route against the walk route
        // =============================================================================================================

        /// <summary>
        ///     The one-table kernel route (a dict table) against the C-order walk (the same dict object once per element, an
        ///     object array of tables): equal results for <c>U</c>, <c>S</c> and <c>StringDType</c> arrays of random texts, over
        ///     tables that keep, change and delete characters and grow the text.
        /// </summary>
        [TestMethod]
        public void KernelRoute_EqualsTheWalk()
        {
            var rng = new System.Random(2633);
            var tables = new[]
            {
                Dict(97, 117L, 98, 118L, 99, 119L),
                Dict(97, null, 0x4E2D, "ZZ", 0xE9, "e"),
                Dict(98, "XYZ", 99, "", 0x10400, null),
                Dict(0xE9, 0x4E2D, 97, "á"),
            };
            const string chars = "abcxyé中文ß \0";
            for (int rep = 0; rep < 24; rep++)
            {
                int n = rng.Next(1, 40);
                int w = rng.Next(1, 20);
                var texts = Enumerable.Range(0, n)
                    .Select(_ => new string(Enumerable.Range(0, rng.Next(0, w + 1)).Select(_ => chars[rng.Next(chars.Length)]).ToArray()))
                    .ToArray();
                foreach (var table in tables)
                {
                    object[] perElement = Enumerable.Repeat((object)table, n).ToArray();
                    using var u = np.array(texts, np.dtype("U" + w));
                    using (var a = np.strings.translate(u, table))
                    using (var b = np.strings.translate(u, perElement))
                        np.array_equal(a, b).Should().BeTrue($"U{w} rep {rep}");
                    using var t = np.array(texts, np.dtypes.StringDType());
                    using (var a = np.strings.translate(t, table))
                    using (var b = np.strings.translate(t, perElement))
                        Strings(a).Should().Equal(Strings(b), $"T rep {rep}");
                }
                var mk = Enumerable.Range(0, 256).Select(i => (byte)((i * 7 + 3) & 0xFF)).ToArray();
                using var s = S(w, texts.Select(x => x.Replace('中', 'q').Replace('文', 'r')).ToArray());
                using (var a = np.strings.translate(s, mk, new byte[] { (byte)'a', 0 }))
                using (var b = np.strings.translate(s, Enumerable.Repeat((object)mk, n).ToArray(), new byte[] { (byte)'a', 0 }))
                    np.array_equal(a, b).Should().BeTrue($"S rep {rep}");
            }
        }

        // =============================================================================================================
        // The public function against NumPy 2.4.2
        // =============================================================================================================

        /// <summary>The values of a 1-D result (strings; a <c>StringDType</c> NA as its NA object).</summary>
        /// <param name="r">The result.</param>
        /// <returns>Its values.</returns>
        private static object[] Strings(NDArray r) => Enumerable.Range(0, (int)r.size).Select(i => r.item(i)).ToArray();

        /// <summary>
        ///     <c>U</c> with dict tables (probed: NumPy 2.4.2): maketrans maps, deletions, replacements cut at the width, keys
        ///     that equal a code point (97.0, True) or never do ('a'), a bool value (code point 1), the largest code point,
        ///     a NUL mapped inside a text but not at its end.
        /// </summary>
        [TestMethod]
        public void U_DictTables_MatchNumPy()
        {
            using var u = np.array(new[] { "abc", "cab", "" });
            Strings(np.strings.translate(u, Dict(97, 120L, 98, 121L, 99, 122L))).Should().Equal("xyz", "zxy", "");
            using var r1 = np.strings.translate(np.array(new[] { "abc", "cab" }), Dict(97, "XYZ", 98, null));
            r1.dtype.Should().Be(np.dtype("U3"));
            Strings(r1).Should().Equal("XYZ", "cXY");
            Strings(np.strings.translate(np.array(new[] { "abc" }), Dict(97.0, "z"))).Should().Equal("zbc");
            Strings(np.strings.translate(np.array(new[] { "abc" }), Dict(true, "z", 97, "q"))).Should().Equal("qbc");
            Strings(np.strings.translate(np.array(new[] { "\u0001bc" }), Dict(true, "z"))).Should().Equal("zbc");
            Strings(np.strings.translate(np.array(new[] { "abc" }), Dict("a", "z"))).Should().Equal("abc");
            Strings(np.strings.translate(np.array(new[] { "abc" }), Dict(97, true))).Should().Equal("\u0001bc");
            Strings(np.strings.translate(np.array(new[] { "abc" }), Dict(97, 0x10FFFF))).Should().Equal("\U0010FFFFbc");
            Strings(np.strings.translate(np.array(new[] { "abc" }), Dict(BigInteger.Pow(2, 70), "z", 97, "q"))).Should().Equal("qbc");
            Strings(np.strings.translate(np.array(new[] { "a\0b\0" }), Dict(0, "Z"))).Should().Equal("aZb");
            Strings(np.strings.translate(np.array(new[] { "ab", "é" }), Dict(97, "é", 233, "a"))).Should().Equal("éb", "a");
            Strings(np.strings.translate(np.array(new[] { "a\U00010400c" }), Dict(0x10400, "x", 99, "\U00010401"))).Should().Equal("ax\U00010401");
            // deletechars is ignored for U, even one that would not broadcast.
            Strings(np.strings.translate(np.array(new[] { "abc" }), Dict(97, "z"), new long[] { 1, 2 })).Should().Equal("zbc");
        }

        /// <summary>
        ///     <c>U</c> with other tables (probed: NumPy 2.4.2): a str or bytes table indexed by code point (trailing NULs of an
        ///     <c>np.str_</c> table dropped), a number keeping everything, None raising only when a character is looked up.
        /// </summary>
        [TestMethod]
        public void U_OtherTables_MatchNumPy()
        {
            using var abc = np.array(new[] { "abc" });
            Strings(np.strings.translate(abc, "xyz")).Should().Equal("abc");
            Strings(np.strings.translate(abc, new string('x', 100))).Should().Equal("xxx");
            Strings(np.strings.translate(abc, Enumerable.Repeat((byte)'x', 100).ToArray())).Should().Equal("xxx");
            Strings(np.strings.translate(np.array(new[] { "\0\u0001\u0002" }), "q\0\0")).Should().Equal("q\u0001\u0002");
            Strings(np.strings.translate(abc, 5L)).Should().Equal("abc");
            Strings(np.strings.translate(abc, 1.5)).Should().Equal("abc");
            Action none = () => np.strings.translate(abc, null);
            none.Should().Throw<TypeError>().WithMessage("'NoneType' object is not subscriptable");
            Strings(np.strings.translate(np.array(new[] { "" }), null)).Should().Equal("");
        }

        /// <summary>
        ///     The errors of str.translate (probed: NumPy 2.4.2): a mapping outside <c>range(0x110000)</c> (a huge int too) is
        ///     a ValueError, a value of another type a TypeError — raised only for a text holding the key, at the first element
        ///     in C order that raises.
        /// </summary>
        [TestMethod]
        public void U_Errors_MatchNumPy()
        {
            using var abc = np.array(new[] { "abc" });
            Action range = () => np.strings.translate(abc, Dict(97, 0x110000));
            range.Should().Throw<ValueError>().WithMessage("character mapping must be in range(0x110000)");
            Action neg = () => np.strings.translate(abc, Dict(97, -1));
            neg.Should().Throw<ValueError>().WithMessage("character mapping must be in range(0x110000)");
            Action big = () => np.strings.translate(abc, Dict(97, BigInteger.Pow(2, 70)));
            big.Should().Throw<ValueError>().WithMessage("character mapping must be in range(0x110000)");
            Action type = () => np.strings.translate(abc, Dict(97, 3.5));
            type.Should().Throw<TypeError>().WithMessage("character mapping must return integer, None or str");
            Action bytes = () => np.strings.translate(abc, Dict(97, new byte[] { (byte)'z' }));
            bytes.Should().Throw<TypeError>().WithMessage("character mapping must return integer, None or str");
            Strings(np.strings.translate(np.array(new[] { "bcd" }), Dict(97, 3.5))).Should().Equal("bcd");
            // C order: 'bc' (element 0) meets 99 -> -1 before 'ab' meets 97 -> 3.5, in a transposed view too.
            Action order = () => np.strings.translate(np.array(new[] { "bc", "ab" }), Dict(97, 3.5, 99, -1));
            order.Should().Throw<ValueError>().WithMessage("character mapping must be in range(0x110000)");
            using var col = np.array(new[] { "bc", "ab" }).reshape(2, 1);
            using var tr = col.T;
            Action orderT = () => np.strings.translate(tr, Dict(97, 3.5, 99, -1));
            orderT.Should().Throw<ValueError>().WithMessage("character mapping must be in range(0x110000)");
        }

        /// <summary>
        ///     <c>U</c> units past U+10FFFF (probed: NumPy 2.4.2): a one-unit text raises at its scalar, a kept unit raises
        ///     when written, a mapped or deleted one does not; a lone surrogate is a character like any other.
        /// </summary>
        [TestMethod]
        public void U_UnitsPastU10FFFF_MatchNumPy()
        {
            static NDArray U(params uint[] units)
            {
                var a = np.zeros(new Shape(units.Length / 2), np.dtype("U2"));
                using var v = a.view(np.uint32);
                for (int i = 0; i < units.Length; i++)
                    v.SetUInt32(units[i], i);
                return a;
            }
            Action kept = () => np.strings.translate(U(0x61, 0x110000), Dict(97, "z"));
            kept.Should().Throw<SystemError>().WithMessage("invalid maximum character passed to PyUnicode_New");
            Action alone = () => np.strings.translate(U(0x110000, 0), Dict(0x110000, "x"));
            alone.Should().Throw<SystemError>().WithMessage("invalid maximum character passed to PyUnicode_New");
            Strings(np.strings.translate(U(0x61, 0x110000), Dict(0x110000, "x"))).Should().Equal("ax");
            Strings(np.strings.translate(U(0x110000, 0x61), Dict(0x110000, null))).Should().Equal("a");
            Strings(np.strings.translate(U(0xD800, 0x61), Dict(97, "z", 0xD800, "q"))).Should().Equal("qz");
            Strings(np.strings.translate(U(0x61, 0x62), Dict(97, 0xDC00))).Should().Equal("\uDC00b");
        }

        /// <summary>
        ///     <c>S</c> (probed: NumPy 2.4.2): bytes.translate's 256-byte table and delete set; a table of another length is a
        ///     ValueError for any non-empty array (even of empty texts) but not an empty one; a table or delete set without a
        ///     buffer is a TypeError, the table checked first; a str argument is its UCS-4 bytes (a 64-character str is a
        ///     256-byte table, a deletechars 'a' deletes NULs too); a number is its raw bytes; NULs inside a text are
        ///     translated, the trailing ones are not; and a bytes delete set loses its trailing NULs like any np.bytes_
        ///     (b'\x00' deletes nothing, b'\x00a' deletes NUL and 'a').
        /// </summary>
        [TestMethod]
        public void S_MatchesNumPy()
        {
            var mk = Enumerable.Range(0, 256).Select(i => (byte)i).ToArray();
            mk['a'] = (byte)'x';
            mk['b'] = (byte)'y';
            mk['c'] = (byte)'z';
            using var s = S(3, "abc", "cab");
            SStrings(np.strings.translate(s, mk)).Should().Equal("xyz", "zxy");
            SStrings(np.strings.translate(s, null, new byte[] { (byte)'a' })).Should().Equal("bc", "cb");
            SStrings(np.strings.translate(s, mk, new byte[] { (byte)'b' })).Should().Equal("xz", "zx");
            Action shortTable = () => np.strings.translate(s, Enumerable.Repeat((byte)'x', 255).ToArray());
            shortTable.Should().Throw<ValueError>().WithMessage("translation table must be 256 characters long");
            using var emptyText = S(1, "");
            Action emptyTextShort = () => np.strings.translate(emptyText, Enumerable.Repeat((byte)'x', 255).ToArray());
            emptyTextShort.Should().Throw<ValueError>().WithMessage("translation table must be 256 characters long");
            using var empty = np.zeros(new Shape(0), np.dtype("S3"));
            np.strings.translate(empty, Enumerable.Repeat((byte)'x', 255).ToArray()).size.Should().Be(0);
            Action nulEnd = () => np.strings.translate(s, Enumerable.Range(0, 255).Select(i => (byte)i).Append((byte)0).ToArray());
            nulEnd.Should().Throw<ValueError>().WithMessage("translation table must be 256 characters long");
            Action dict = () => np.strings.translate(s, Dict(97, 98));
            dict.Should().Throw<TypeError>().WithMessage("a bytes-like object is required, not 'dict'");
            Action order = () => np.strings.translate(s, Enumerable.Repeat((byte)'x', 10).ToArray(), Dict(1, 2));
            order.Should().Throw<ValueError>().WithMessage("translation table must be 256 characters long");
            Action deleteNone = () => np.strings.translate(s, null, new object[] { null });
            deleteNone.Should().Throw<TypeError>().WithMessage("a bytes-like object is required, not 'NoneType'");
            Action strTable = () => np.strings.translate(s, new string('x', 256));
            strTable.Should().Throw<ValueError>().WithMessage("translation table must be 256 characters long");
            using var withNul = S(5, "aéb\0c");
            SStrings(np.strings.translate(withNul, null, "é")).Should().Equal("abc");
            SStrings(np.strings.translate(S(3, "abc"), null, 97L)).Should().Equal("bc");
            var nulMap = Enumerable.Range(0, 256).Select(i => (byte)i).ToArray();
            nulMap[0] = (byte)'Z';
            using var inner = S(4, "a\0b\0");
            SStrings(np.strings.translate(inner, nulMap)).Should().Equal("aZb");
            SStrings(np.strings.translate(inner, null, new byte[] { 0 })).Should().Equal("a\0b");
            SStrings(np.strings.translate(inner, null, new byte[] { 0, (byte)'a' })).Should().Equal("b");
        }

        /// <summary>An <c>S</c> array of Latin-1 texts (each byte its character), of the given width.</summary>
        /// <param name="width">The width.</param>
        /// <param name="texts">The texts.</param>
        /// <returns>The array.</returns>
        private static NDArray S(int width, params string[] texts)
            => np.array(texts.Select(t => (object)System.Text.Encoding.Latin1.GetBytes(t)).ToArray(), np.dtype("S" + width));

        /// <summary>The Latin-1 texts of a 1-D <c>S</c> result (each element as <c>item()</c> gives its bytes).</summary>
        /// <param name="r">The result.</param>
        /// <returns>The texts.</returns>
        private static string[] SStrings(NDArray r)
            => Enumerable.Range(0, (int)r.size).Select(i => System.Text.Encoding.Latin1.GetString((byte[])r.item(i))).ToArray();

        /// <summary>
        ///     <c>StringDType</c> (probed: NumPy 2.4.2): NULs are characters; an NA whose object is not a str is the descriptor's
        ///     TypeError, raised in C order with the other errors; a string NA is translated as its text and a result equal
        ///     to it is the NA; deletechars is one argument too many for str.translate; a surrogate result cannot be stored.
        /// </summary>
        [TestMethod]
        public void StringDType_MatchesNumPy()
        {
            using var t = np.array(new[] { "abc", "cab" }, np.dtypes.StringDType());
            Strings(np.strings.translate(t, Dict(97, "XYZ"))).Should().Equal("XYZbc", "cXYZb");
            Strings(np.strings.translate(np.array(new[] { "a\0b\0" }, np.dtypes.StringDType()), Dict(0, "Z"))).Should().Equal("aZbZ");
            using var none = np.array(new object[] { "abc", null }, np.dtypes.StringDType(null));
            Action noneNA = () => np.strings.translate(none, Dict(97, "XYZ"));
            noneNA.Should().Throw<TypeError>().WithMessage("descriptor 'translate' for 'str' objects doesn't apply to a 'NoneType' object");
            using var nan = np.array(new object[] { "abc", double.NaN }, np.dtypes.StringDType(double.NaN));
            Action nanNA = () => np.strings.translate(nan, Dict(97, "XYZ"));
            nanNA.Should().Throw<TypeError>().WithMessage("descriptor 'translate' for 'str' objects doesn't apply to a 'float' object");
            // C order: element 0 ('abc') raises the table's error before the NA is reached; '' looks nothing up.
            Action first = () => np.strings.translate(none, null);
            first.Should().Throw<TypeError>().WithMessage("'NoneType' object is not subscriptable");
            using var emptyFirst = np.array(new object[] { "", null }, np.dtypes.StringDType(null));
            Action naSecond = () => np.strings.translate(emptyFirst, null);
            naSecond.Should().Throw<TypeError>().WithMessage("descriptor 'translate' for 'str' objects doesn't apply to a 'NoneType' object");
            using var q = np.array(new object[] { "abc", "Q" }, np.dtypes.StringDType("Q"));
            using var rq = np.strings.translate(q, Dict(97, "Q", 98, null, 99, null));
            rq.dtype.Should().Be(np.dtypes.StringDType("Q"));
            Strings(rq).Should().Equal("Q", "Q");
            Action deletechars = () => np.strings.translate(t, Dict(97, "z"), "b");
            deletechars.Should().Throw<TypeError>().WithMessage("str.translate() takes exactly one argument (2 given)");
            using var emptyT = np.array(new string[0], np.dtypes.StringDType());
            np.strings.translate(emptyT, Dict(97, "z"), "b").size.Should().Be(0);
            Action surrogate = () => np.strings.translate(np.array(new[] { "abc" }, np.dtypes.StringDType()), Dict(97, 0xD800));
            surrogate.Should().Throw<TypeError>().WithMessage("result array type does not match underlying function");
            // Element 0's own translate error comes before any setitem failure.
            Action surrogateOrder = () => np.strings.translate(np.array(new[] { "abc", "b" }, np.dtypes.StringDType()), Dict(97, 0xD800, 98, 3.5));
            surrogateOrder.Should().Throw<TypeError>().WithMessage("character mapping must return integer, None or str");
        }

        /// <summary>
        ///     Arrays of tables (probed: NumPy 2.4.2): a list of str tables and a <c>U</c> array broadcast per element; an
        ///     object list holding dicts and None, a 2-D one widening the result; NumPy's multi-iterator text for shapes that do
        ///     not broadcast.
        /// </summary>
        [TestMethod]
        public void ArraysOfTables_MatchNumPy()
        {
            using var two = np.array(new[] { "abc", "de" });
            Strings(np.strings.translate(two, new[] { new string('x', 100), new string('y', 120) })).Should().Equal("xxx", "yy");
            Strings(np.strings.translate(two, np.array(new[] { new string('x', 100) }))).Should().Equal("xxx", "de");
            Action mismatch = () => np.strings.translate(np.array(new[] { "abc", "b", "c" }), np.array(new[] { "zz", "yy" }));
            mismatch.Should().Throw<IncorrectShapeException>()
                .WithMessage("shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (3,) and arg 1 with shape (2,).");
            Action objNone = () => np.strings.translate(np.array(new[] { "abc", "b" }), new object[] { Dict(97, "z"), null });
            objNone.Should().Throw<TypeError>().WithMessage("'NoneType' object is not subscriptable");
            Strings(np.strings.translate(np.array(new[] { "", "b" }), new object[] { null, Dict(98, "q") })).Should().Equal("", "q");
            Action objInt = () => np.strings.translate(np.array(new[] { "abc" }), new object[] { 5L, BigInteger.Pow(2, 70) });
            objInt.Should().Throw<TypeError>().WithMessage("'int' object is not subscriptable");
            using var r = np.strings.translate(np.array(new[] { "abc" }), new object[,] { { Dict(97, "z") }, { Dict(98, "y") } });
            r.shape.Should().Equal(2L, 1L);
            r.item(0, 0).Should().Be("zbc");
            r.item(1, 0).Should().Be("ayc");
        }

        /// <summary>
        ///     The result objects (probed: NumPy 2.4.2): a fresh C-ordered result whatever the operand's layout, a 0-d result
        ///     for a 0-d operand, an empty result for an empty one, a plain array for a chararray operand; non-text operands
        ///     are NumPy's TypeError.
        /// </summary>
        [TestMethod]
        public void ResultObjects_MatchNumPy()
        {
            using var m = np.array(new[] { "ab", "cd", "ef", "gh", "ij", "kl" }).reshape(2, 3);
            using var mt = m.T;
            using var r = np.strings.translate(mt, Dict(97, "A", 103, "G"));
            r.Shape.IsContiguous.Should().BeTrue();
            using var flat = r.flatten();
            Strings(flat).Should().Equal("Ab", "Gh", "cd", "ij", "ef", "kl");
            using var zero = np.strings.translate(NDArray.Scalar("abc"), Dict(97, "z"));
            zero.ndim.Should().Be(0);
            zero.item().Should().Be("zbc");
            using var empty = np.strings.translate(np.zeros(new Shape(2, 0), np.dtype("U3")), Dict(97, "z"));
            empty.shape.Should().Equal(2L, 0L);
            using var c = np.@char.array(new[] { "ab", "c" });
            using var rc = np.strings.translate(c, Dict(97, "z"));
            rc.GetType().Should().Be(typeof(NDArray));
            Strings(rc).Should().Equal("zb", "c");
            Action ints = () => np.strings.translate(np.array(new[] { 1, 2 }), Dict(97, "z"));
            ints.Should().Throw<TypeError>().WithMessage("string operation on non-string array");
        }
    }
}
