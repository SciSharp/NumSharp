using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The fast paths behind the chararray benchmark cells (numpy.char phase 2), each held to the general route it
    ///     replaces: the element stand-in cut from a per-thread arena, the lean stand-in storage, the direct string gathers
    ///     (index array, boolean mask) and their constant-width copy kernels, the parse cache of slice notation, the view and
    ///     copy shapes shared with the engine, and the float text of <c>np.char.array</c>. A difference from the general
    ///     route is a wrong answer the oracle tiers may not reach (a width, a layout, a lifetime).
    /// </summary>
    [TestClass]
    public class CharArrayFastPathTests
    {
        /// <summary>The deterministic strings of the benchmark: a pool word plus trailing spaces / tabs.</summary>
        /// <param name="n">The count.</param>
        /// <param name="width">The longest kept text.</param>
        /// <returns>The values.</returns>
        private static string[] Values(int n, int width)
        {
            string[] pool = { "abc", "abd", "ab", "xyz", "hello", "a b", "q", "abcdefg", "", "zz top" };
            string[] trail = { "", " ", "\t", "  " };
            return Enumerable.Range(0, n).Select(i =>
            {
                string s = pool[(i * 7 + 3) % pool.Length] + trail[(i * 5) % 4];
                return s.Length > width ? s.Substring(0, width) : s;
            }).ToArray();
        }

        // ------------------------------------------------------------------ the element stand-in

        /// <summary>
        ///     Stand-ins are cut from a shared block, yet each is an independent snapshot: thousands of them (several blocks
        ///     for <c>U8</c> and <c>S8</c>, a block of its own past 256 bytes for <c>U100</c>), some kept and the rest disposed,
        ///     keep their stripped text after the source is overwritten and after the arena has moved on.
        /// </summary>
        [TestMethod]
        public void StandIns_AreIndependentSnapshots_AcrossBlocks()
        {
            foreach (var (descr, width) in new[] { ("U8", 8), ("S8", 8), ("U100", 100) })
            {
                var values = Values(3000, width).Select(v => width > 8 ? v + new string('x', (int)(v.Length % 3) * 30) : v).ToArray();
                var c = (NDCharArray)np.array(values, dtype: descr).view(typeof(NDCharArray));
                var kept = new List<(int, NDCharArray)>();
                for (int i = 0; i < values.Length; i++)
                {
                    var s = c[i];
                    if (i % 7 == 0)
                        kept.Add((i, s));
                    else
                        s.Dispose();
                }

                c[":"] = "ZZZZZZZZ";   // overwrite every element of the source
                foreach (var (i, s) in kept)
                {
                    object expected = values[i].TrimEnd(' ', '\t');
                    object actual = s.item();
                    if (actual is byte[] bytes)
                        actual = System.Text.Encoding.ASCII.GetString(bytes);
                    actual.Should().Be(expected, $"{descr} element {i}");
                    s.ndim.Should().Be(0);
                    s.flags.writeable.Should().BeFalse();
                    s.flags.owndata.Should().BeTrue();
                    s.IsScalarStandIn.Should().BeTrue();
                    s.Dispose();
                }

                c.Dispose();
            }
        }

        /// <summary>
        ///     A stand-in disposed on another thread than the one that cut it releases its reference on the shared block
        ///     (atomically): the other stand-ins of the block stay readable, and the cutting thread keeps cutting.
        /// </summary>
        [TestMethod]
        public void StandIns_DisposedOnAnotherThread_LeaveTheRestReadable()
        {
            var values = Values(2000, 8);
            using var c = np.@char.array(values);
            var all = Enumerable.Range(0, values.Length).Select(i => c[i]).ToArray();
            Parallel.For(0, all.Length, i =>
            {
                if (i % 2 == 1)
                    all[i].Dispose();
            });
            for (int i = 0; i < all.Length; i += 2)
            {
                all[i].item().Should().Be(values[i].TrimEnd(' ', '\t'));
                all[i].Dispose();
            }

            c[1].item().Should().Be(values[1].TrimEnd(' ', '\t'));
        }

        /// <summary>
        ///     The lean stand-in storage is field for field the generic construction (<c>new UnmanagedStorage(slice, scalar)</c>
        ///     with WRITEABLE cleared): the same shape and flags, lane, CLR type, descriptor, slice, address and count, and the
        ///     same views and copies.
        /// </summary>
        [TestMethod]
        public unsafe void StandInStorage_IsTheGenericConstruction()
        {
            foreach (var descr in new[] { DTypeRegistry.Str.Descr(5), DTypeRegistry.Bytes.Descr(3) })
            {
                var slice = StringArraySlice.Allocate(descr, 1, fillZeros: true);
                var lean = UnmanagedStorage.ReadOnlyStringScalar(slice);
                var generic = new UnmanagedStorage(slice, Shape.Scalar);
                generic.SetShapeUnsafe(generic.Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));

                lean.Shape._flags.Should().Be(generic.Shape._flags);
                lean.Shape.dimensions.Should().Equal(generic.Shape.dimensions);
                lean.Shape.strides.Should().Equal(generic.Shape.strides);
                lean.Shape.offset.Should().Be(generic.Shape.offset);
                lean.TypeCode.Should().Be(generic.TypeCode);
                lean.DType.Should().Be(generic.DType);
                ReferenceEquals(lean.Descr, generic.Descr).Should().BeTrue();
                ReferenceEquals(lean.InternalArray, generic.InternalArray).Should().BeTrue();
                ((IntPtr)lean.Address).Should().Be((IntPtr)generic.Address);
                lean.Count.Should().Be(generic.Count);
                lean.OwnsData.Should().Be(generic.OwnsData);
                lean.IsView.Should().Be(generic.IsView);
                ((IntPtr)lean.Alias().Address).Should().Be((IntPtr)lean.Address);

                using var a = new NDArray(lean);
                using var b = new NDArray(generic);
                a.dtype.Should().Be(b.dtype);
                a.item().Should().BeEquivalentTo(b.item());
                using var copy = a.copy();
                copy.flags.writeable.Should().BeTrue();
            }
        }

        /// <summary>
        ///     An out-of-range integer raises the same NumPy text on the direct element route as on the view route (the
        ///     integer as given, its axis, the axis length).
        /// </summary>
        [TestMethod]
        public void ElementAccess_OutOfBounds_NumPysText()
        {
            using var c = np.@char.array(new[] { "a", "b", "c" });
            Action high = () => c[3].Dispose();
            high.Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 0 with size 3");
            Action low = () => c[-4].Dispose();
            low.Should().Throw<IndexError>().WithMessage("index -4 is out of bounds for axis 0 with size 3");
            c[-3].item().Should().Be("a");

            using var m = np.@char.array(new[,] { { "a", "b" }, { "c", "d" } });
            Action col = () => m[0, 5].Dispose();
            col.Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 1 with size 2");
            m[1, -1].item().Should().Be("d");
            m[-1].tolist().Should().BeAssignableTo<IList>();
        }

        /// <summary>
        ///     The direct element route reads the element through the view's offset and strides: a stepped view (offset 1),
        ///     a reversed one (negative stride, offset at the end), a transposed matrix and a stepped column, through the
        ///     <c>int</c> indexer, <c>__getitem__</c> and the two-integer index — each against the general route
        ///     (<c>item</c>, which reads through the strides and does not strip).
        /// </summary>
        [TestMethod]
        public void ElementAccess_OfViews_ReadsThroughOffsetAndStrides()
        {
            var values = Values(23, 8);
            using var c = np.@char.array(values);
            foreach (var notation in new[] { "1::2", "::-1", "-2:3:-3", "5:" })
            {
                using NDCharArray view = c[notation];
                for (int i = 0; i < view.size; i++)
                {
                    object expected = ((string)view.item(i)).TrimEnd(' ', '\t');
                    using (var s = view[i])
                        s.item().Should().Be(expected, $"{notation}[{i}]");
                    using (var s = view.__getitem__(i - (int)view.size))
                        s.item().Should().Be(expected, $"{notation}[{i - view.size}]");
                }
            }

            using var m = np.@char.array(new[,] { { "a ", "bb", "c\t" }, { "dd", "e", "f " } });
            using var t = m.T;
            using var col = m[":, ::2"];
            foreach (var view in new[] { t, col })
            {
                for (int i = 0; i < view.shape[0]; i++)
                    for (int j = 0; j < view.shape[1]; j++)
                        using (var s = view[i, j])
                            s.item().Should().Be(((string)view.item(i * view.shape[1] + j)).TrimEnd(' ', '\t'));
            }
        }

        // ------------------------------------------------------------------ the string gathers

        /// <summary>
        ///     <c>a[idx]</c>, <c>np.take(a, idx)</c> and <c>a[mask]</c> of fixed-width text, element by element against the
        ///     source, for widths that are typed MOVs (1/2/4/8/16 bytes), constant-size block copies of any other width (3, 5,
        ///     20, 100, 255, 256 bytes), and the runtime-sized copy past 256 bytes; 1-D elements and 2-D rows; int32 and int64
        ///     indices with negative values; empty and all-false selections.
        /// </summary>
        [TestMethod]
        public void Gathers_EveryWidth_ElementForElement()
        {
            var rng = new System.Random(7);
            foreach (int bytes in new[] { 1, 2, 3, 4, 5, 8, 16, 20, 64, 100, 255, 256, 257, 300, 1024 })
            {
                foreach (bool unicode in new[] { false, true })
                {
                    if (unicode && bytes % 4 != 0)
                        continue;
                    int chars = unicode ? bytes / 4 : bytes;
                    var values = Enumerable.Range(0, 37).Select(i => new string((char)('a' + i % 26), 1 + (i * 13) % chars)).ToArray();
                    using var a = np.array(values, dtype: (unicode ? "U" : "S") + chars);
                    long[] index = Enumerable.Range(0, 50).Select(_ => (long)rng.Next(-37, 37)).ToArray();
                    using var idx64 = np.array(index);
                    using var idx32 = np.array(index.Select(v => (int)v).ToArray());
                    var mask = Enumerable.Range(0, 37).Select(i => i % 3 != 1).ToArray();
                    using var m = np.array(mask).MakeGeneric<bool>();

                    object[] expectedIdx = index.Select(i => a.item(i < 0 ? i + 37 : i)).ToArray();
                    object[] expectedMask = Enumerable.Range(0, 37).Where(i => mask[i]).Select(i => a.item(i)).ToArray();
                    using (var g = a[new object[] { idx64 }])
                        ((IList)g.tolist()).Cast<object>().Should().BeEquivalentTo(expectedIdx, o => o.WithStrictOrdering(), $"a[idx64] {bytes}B");
                    using (var g = a[new object[] { idx32 }])
                        ((IList)g.tolist()).Cast<object>().Should().BeEquivalentTo(expectedIdx, o => o.WithStrictOrdering(), $"a[idx32] {bytes}B");
                    using (var g = np.take(a, idx64))
                        ((IList)g.tolist()).Cast<object>().Should().BeEquivalentTo(expectedIdx, o => o.WithStrictOrdering(), $"take {bytes}B");
                    using (var g = a[m])
                    {
                        ((IList)g.tolist()).Cast<object>().Should().BeEquivalentTo(expectedMask, o => o.WithStrictOrdering(), $"a[mask] {bytes}B");
                        g.flags.owndata.Should().BeTrue();
                        g.flags.c_contiguous.Should().BeTrue();
                    }

                    // Rows of a 2-D array: the slab is a whole row.
                    using var rows = a["0:36"].reshape(12, 3);
                    using var rowIdx = np.array(new long[] { 11, -12, 5 });
                    using (var g = rows[new object[] { rowIdx }])
                    {
                        g.shape.Should().Equal(3L, 3L);
                        g[1, 2].item().Should().BeEquivalentTo(rows[0, 2].item());
                        g[0, 0].item().Should().BeEquivalentTo(rows[11, 0].item());
                    }
                    using var rowMask = np.array(Enumerable.Range(0, 12).Select(i => i % 4 == 0).ToArray()).MakeGeneric<bool>();
                    using (var g = rows[rowMask])
                    {
                        g.shape.Should().Equal(3L, 3L);
                        g[2, 1].item().Should().BeEquivalentTo(rows[8, 1].item());
                    }

                    using (var none = a[np.zeros(new Shape(37), np.@bool).MakeGeneric<bool>()])
                        none.shape.Should().Equal(0L);
                    using (var empty = a[new object[] { np.array(Array.Empty<long>()) }])
                        empty.shape.Should().Equal(0L);
                }
            }
        }

        /// <summary>
        ///     The mask gather's vector stage (32 mask bytes a step) and its scalar tail meet at every length up to 70, and any
        ///     non-zero mask byte selects (a bool array viewed from uint8 data).
        /// </summary>
        [TestMethod]
        public void MaskGather_VectorStageAndTail_AnyNonZeroSelects()
        {
            for (int n = 0; n <= 70; n++)
            {
                var values = Enumerable.Range(0, n).Select(i => "w" + i).ToArray();
                using var a = np.array(values, dtype: "U4");
                var raw = Enumerable.Range(0, n).Select(i => (byte)(i % 5 == 0 ? 0 : i % 3 == 0 ? 7 : 1)).ToArray();
                using var bytes = np.array(raw);
                using var mask = bytes.view(np.@bool).MakeGeneric<bool>();
                using var picked = a[mask];
                var expected = Enumerable.Range(0, n).Where(i => raw[i] != 0).Select(i => (object)values[i]).ToArray();
                ((IList)picked.tolist()).Cast<object>().Should().Equal(expected, $"length {n}");
            }
        }

        /// <summary>A bad index raises NumPy's text on the direct gather and leaves no partial result behind.</summary>
        [TestMethod]
        public void Gather_OutOfBounds_NumPysText()
        {
            using var a = np.array(new[] { "a", "b", "c" });
            using var idx = np.array(new long[] { 0, 3 });
            Action act = () => a[new object[] { idx }].Dispose();
            act.Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 0 with size 3");
            using var neg = np.array(new[] { -4 });
            Action negative = () => a[new object[] { neg }].Dispose();
            negative.Should().Throw<IndexError>().WithMessage("index -4 is out of bounds for axis 0 with size 3");
        }

        /// <summary>The flat gather kernel of every constant width copies exactly that many bytes per index (no more).</summary>
        [TestMethod]
        public unsafe void FlatGather_ConstantWidths_CopyExactlyTheSlab()
        {
            foreach (int width in new[] { 3, 7, 20, 31, 32, 33, 100, 128, 200, 256 })
            {
                int n = 10;
                var src = new byte[n * width];
                for (int i = 0; i < src.Length; i++)
                    src[i] = (byte)(i * 31 + 7);
                var idx = new long[] { 9, 0, -1, 4, 4 };
                var dst = new byte[(idx.Length + 1) * width];
                Array.Fill(dst, (byte)0xEE);
                fixed (byte* s = src)
                fixed (long* ip = idx)
                fixed (byte* d = dst)
                {
                    long status = FancyIndexKernelRoute.Gather(s, ip, false, idx.Length, n, width, d);
                    status.Should().Be(-1);
                }

                for (int j = 0; j < idx.Length; j++)
                {
                    long at = idx[j] < 0 ? idx[j] + n : idx[j];
                    dst.AsSpan(j * width, width).ToArray().Should().Equal(src.AsSpan((int)at * width, width).ToArray(), $"width {width} slab {j}");
                }

                dst.AsSpan(idx.Length * width).ToArray().Should().OnlyContain(b => b == 0xEE, $"width {width}: nothing past the last slab");
            }
        }

        // ------------------------------------------------------------------ views and copies

        /// <summary>
        ///     The parse cache answers fresh slice objects: changing what one call returned never changes a later parse, a bad
        ///     notation raises on every call, and more notations than the cache holds still parse.
        /// </summary>
        [TestMethod]
        public void ParseSlices_Cache_ReturnsFreshSlices()
        {
            var first = Slice.ParseSlices("1:7:2, ::-1, 3");
            first[0].Start = 99;
            first[2].IsIndex = false;
            // A cache HIT's slices are fresh too: modifying them leaves the next hit untouched.
            var hit = Slice.ParseSlices("1:7:2, ::-1, 3");
            hit[0].Start = 42;
            hit[1].Step = 5;
            var second = Slice.ParseSlices("1:7:2, ::-1, 3");
            ReferenceEquals(hit[0], second[0]).Should().BeFalse();
            second[0].Start.Should().Be(1);
            second[0].Stop.Should().Be(7);
            second[0].Step.Should().Be(2);
            second[1].Step.Should().Be(-1);
            second[2].IsIndex.Should().BeTrue();
            ReferenceEquals(first[0], second[0]).Should().BeFalse();

            Action bad = () => Slice.ParseSlices("1:x");
            bad.Should().Throw<ArgumentException>();
            bad.Should().Throw<ArgumentException>();

            for (int i = 0; i < 1200; i++)
            {
                var s = Slice.ParseSlices($"{i}:{i + 2}");
                s[0].Start.Should().Be(i);
                s[0].Stop.Should().Be(i + 2);
            }
        }

        /// <summary>
        ///     The view route's trims keep its answers: a trailing <c>':'</c> after integers is a direct sub-array, an all-<c>':'</c>
        ///     index an alias, a dropped axis and a kept axis through the shape slicing (strided and contiguous sources).
        /// </summary>
        [TestMethod]
        public void GetView_Trims_KeepTheViews()
        {
            using var a = np.arange(60).reshape(3, 4, 5);
            using (var v = a[1, 2, Slice.All])
                ((long[])v.Shape).Should().Equal(5L);
            using (var v = a["1, 2, 1:3"])
            {
                ((long[])v.Shape).Should().Equal(2L);
                v.GetInt64(1).Should().Be(a.GetInt64(1, 2, 2));
            }
            using (var v = a[Slice.All, Slice.All])
                v.Storage.IsView.Should().BeTrue();
            using (var v = a["1:3, 0, ::2"])
            {
                ((long[])v.Shape).Should().Equal(2L, 3L);
                v.GetInt64(1, 2).Should().Be(a.GetInt64(2, 0, 4));
            }
            using var stepped = a[":, ::2"];
            using (var v = stepped["0, 1"])
            {
                ((long[])v.Shape).Should().Equal(5L);
                v.GetInt64(4).Should().Be(a.GetInt64(0, 2, 4));
            }
        }

        /// <summary>
        ///     <c>c.T</c> built on the storage is the engine's transpose: the same shape, strides, offset and flags (a read-only
        ///     source stays read-only), a chararray view that writes through.
        /// </summary>
        [TestMethod]
        public void Transpose_IsTheEnginesView()
        {
            using var m = np.@char.array(new[,] { { "a", "bb", "c" }, { "d", "e", "ff" } });
            using var t = m.T;
            using var plain = ((NDArray)m).transpose();
            t.shape.Should().Equal(plain.shape);
            t.strides.Should().Equal(plain.strides);
            t.Shape.offset.Should().Be(plain.Shape.offset);
            t.Shape._flags.Should().Be(plain.Shape._flags);
            t.flags.owndata.Should().BeFalse();
            t[2, 1] = "zz";
            m[1, 2].item().Should().Be("zz");

            using var ro = (NDCharArray)np.broadcast_to(np.array(new[] { "x", "y" }), new Shape(3, 2)).view(typeof(NDCharArray));
            using var rot = ro.T;
            rot.flags.writeable.Should().BeFalse();
            rot.shape.Should().Equal(2L, 3L);
        }

        /// <summary>
        ///     <c>c.copy(order)</c> allocated as a chararray matches the general copy for every order and layout: values,
        ///     shape, strides, an owned writeable buffer (also from a read-only source), and the class kept.
        /// </summary>
        [TestMethod]
        public void Copy_EveryOrderAndLayout_IsTheGeneralCopy()
        {
            using var c = np.@char.array(new[,] { { "a ", "bb", "c" }, { "d", "e", "ff " } });
            using var t = c.T;
            using var stepped = c[":, ::2"];
            using var ro = (NDCharArray)np.broadcast_to(np.array(new[] { "x", "yy" }), new Shape(3, 2)).view(typeof(NDCharArray));
            using var int8 = (NDCharArray)np.array(new sbyte[] { 1, 2, 3 }).view(typeof(NDCharArray));
            foreach (var src in new[] { c, t, stepped, ro, int8 })
            {
                foreach (char order in "CFAK")
                {
                    using var copy = src.copy(order);
                    using var general = ((NDArray)src).copy(order);
                    copy.Should().BeOfType<NDCharArray>();
                    copy.shape.Should().Equal(general.shape);
                    copy.strides.Should().Equal(general.strides, $"order {order}");
                    copy.flags.owndata.Should().BeTrue();
                    copy.flags.writeable.Should().BeTrue();
                    ((IList)copy.tolist()).Should().BeEquivalentTo((IList)general.tolist(), o => o.WithStrictOrdering());
                }
            }
        }

        /// <summary>
        ///     The fresh-copy shape of a source already laid out as a fresh C buffer is that layout, with the source's flags
        ///     re-set by the allocator: a copy of a read-only array or of a view is writeable and owns its data, and a stray
        ///     stride on an extent-1 axis is not inherited.
        /// </summary>
        [TestMethod]
        public void CopyAsShape_FreshLayout_FlagsReset()
        {
            using var a = np.arange(6).reshape(2, 3);
            a.Storage.SetShapeUnsafe(a.Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));
            using (var copy = a.copy())
            {
                copy.flags.writeable.Should().BeTrue();
                copy.flags.owndata.Should().BeTrue();
                copy.strides.Should().Equal(24L, 8L);
            }

            using var view = np.arange(12)["2:8"];
            using (var copy = view.copy())
            {
                copy.flags.owndata.Should().BeTrue();
                ((long[])copy.Shape).Should().Equal(6L);
                copy.GetInt64(0).Should().Be(2);
            }

            NDIter.CopyAsShape(Shape.Adopt(new long[] { 1, 5 }, new long[] { 0, 1 }), 'C').strides.Should().Equal(5L, 1L);
        }

        // ------------------------------------------------------------------ np.char.array of floats

        /// <summary>
        ///     <c>np.char.array</c> of a float array is the text of <c>np.array(obj.tolist(), dtype=cls)</c>: each value's
        ///     repr at the longest one's width (a float32 / float16 by its exact double's repr), probed against NumPy 2.4.2,
        ///     in C order whatever the source layout (F with <c>order='F'</c>).
        /// </summary>
        [TestMethod]
        public void ArrayOfFloats_IsTheReprOfTolist()
        {
            using (var r = np.@char.array(np.array(new[] { -3.0, -2.63, -2.26, -1.8900000000000001, -1.52 })))
            {
                r.dtype.ToString(true).Should().Be("dtype('S19')");
                ((IList)r.tolist()).Cast<byte[]>().Select(b => System.Text.Encoding.ASCII.GetString(b))
                    .Should().Equal("-3.0", "-2.63", "-2.26", "-1.8900000000000001", "-1.52");
            }

            using (var r = np.@char.array(np.array(new[] { 1e20, 1e-5, 0.1, -0.0, double.PositiveInfinity, double.NaN }), unicode: true))
            {
                r.dtype.ToString(true).Should().Be("dtype('<U5')");
                ((IList)r.tolist()).Cast<string>().Should().Equal("1e+20", "1e-05", "0.1", "-0.0", "inf", "nan");
            }

            using (var r = np.@char.array(np.array(new[] { 0.1f })))
            {
                r.dtype.ToString(true).Should().Be("dtype('S19')");
                System.Text.Encoding.ASCII.GetString((byte[])r.item(0)).Should().Be("0.10000000149011612");
            }

            using (var r = np.@char.array(np.array(new[] { (Half)0.1 }), unicode: true))
                r.item(0).Should().Be("0.0999755859375");

            using var f = np.array(new[,] { { 1.5, 2.25 }, { -3.0, 4.0 } });
            using (var r = np.@char.array(f.T, unicode: true))
            {
                r.flags.c_contiguous.Should().BeTrue();
                ((IList)r.tolist()).Cast<IList>().Select(row => string.Join(",", row.Cast<string>())).Should().Equal("1.5,-3.0", "2.25,4.0");
            }

            using (var r = np.@char.array(f, unicode: true, order: 'F'))
            {
                r.flags.f_contiguous.Should().BeTrue();
                r.item(1).Should().Be("2.25");
            }
        }
    }
}
