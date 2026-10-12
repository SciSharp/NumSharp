using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Moving NEP 55 <c>StringDType</c> elements (strings phase 1, P1.4): the reference-aware copy core
    ///     (<see cref="NDVStringCopy"/>: overlap, broadcast, NA conversion), the gathers and scatters
    ///     (<see cref="NDVStringMove"/>: cells re-homed into the destination's allocator, marks for scatters), the values of a
    ///     store (<see cref="NDVStringMove.StoreValue"/>: Python objects by setitem, ndarrays cast), the indexers, np.nditer's
    ///     views, <c>tobytes()</c> and the NumPy 2.4.2 bugs whose correct answer NumSharp gives. Expected values were probed
    ///     against <c>numpy==2.4.2</c>.
    /// </summary>
    [TestClass]
    public class StringDTypeMovementTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor.</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A fresh <c>StringDType(na_object=None)</c> descriptor.</summary>
        private static DType TN => np.dtypes.StringDType(na_object: null);

        /// <summary>A fresh <c>StringDType(coerce=False)</c> descriptor.</summary>
        private static DType TC => np.dtypes.StringDType(coerce: false);

        /// <summary>A <c>StringDType()</c> array of the given texts.</summary>
        /// <param name="texts">The elements.</param>
        /// <returns>The new 1-D array.</returns>
        private static NDArray A(params string[] texts) => np.array(texts, T);

        /// <summary>The elements of an array in C order (texts and NA objects).</summary>
        /// <param name="a">A <c>StringDType</c> array.</param>
        /// <returns>The values.</returns>
        private static object[] V(NDArray a) => Enumerable.Range(0, (int)a.size).Select(i => a.Storage.GetAtIndex(i)).ToArray();

        /// <summary>A long text (an arena string: more than 15 UTF-8 bytes).</summary>
        /// <param name="c">The repeated character.</param>
        /// <param name="n">The length.</param>
        /// <returns>The text.</returns>
        private static string L(char c, int n = 20) => new string(c, n);

        // ------------------------------------------------------------------------------------------------------------
        // the copy core
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Overlapping 1-D copies take NumPy's raw walk (backward when the source starts below the destination inside
        ///     it), so <c>a[1:] = a[:-1]</c> shifts right and <c>a[:-1] = a[1:]</c> shifts left; a destination running against
        ///     its source, and any multi-dimensional overlap, copies the source first (with its own allocator).
        /// </summary>
        [TestMethod]
        public void Copy_Overlap_IsNumPys()
        {
            using (var a = A("a", "b", "c", "d"))
            {
                a["1:"] = a[":-1"];
                V(a).Should().Equal(new object[] { "a", "a", "b", "c" });
            }
            using (var a = A(L('a'), L('b'), L('c'), L('d')))
            {
                a["1:"] = a[":-1"];
                V(a).Should().Equal(new object[] { L('a'), L('a'), L('b'), L('c') });
            }
            using (var a = A("a", "b", "c", "d"))
            {
                a[":-1"] = a["1:"];
                V(a).Should().Equal(new object[] { "b", "c", "d", "d" });
            }
            using (var a = A(L('a'), "b", L('c'), "d"))
            {
                a["::-1"] = a;
                V(a).Should().Equal(new object[] { "d", L('c'), "b", L('a') });
            }
            using (var m = np.array(new[,] { { "a", L('b') }, { "c", "d" }, { "e", "f" } }, T))
            {
                m["1:, :"] = m[":-1, :"];
                V(m).Should().Equal(new object[] { "a", L('b'), "a", L('b'), "c", "d" });
            }
        }

        /// <summary>
        ///     An element copied onto itself (<c>a[...] = a</c>) is skipped (NumPy's share-memory check), so no arena string is
        ///     duplicated; a broadcast source fills every row with its own duplicates.
        /// </summary>
        [TestMethod]
        public void Copy_SelfAndBroadcast()
        {
            using (var a = A(L('a'), "b", L('c')))
            {
                long used = a.Storage.VStringSlice.Allocator.ArenaUsed;
                a["..."] = a;
                a.Storage.VStringSlice.Allocator.ArenaUsed.Should().Be(used);
                V(a).Should().Equal(new object[] { L('a'), "b", L('c') });
            }
            using (var a = A("b", L('x'), "c"))
            {
                // NumPy 2.4.2 frees the short source cell before duplicating it onto itself: ['', '', ''].
                a["..."] = a[":1"];
                V(a).Should().Equal(new object[] { "b", "b", "b" });
            }
            using (var m = np.empty(new Shape(2, 3), T))
            using (var row = A(L('x'), "y", L('z')))
            {
                m["..."] = row;
                V(m).Should().Equal(new object[] { L('x'), "y", L('z'), L('x'), "y", L('z') });
                m.Storage.VStringSlice.Allocator.ArenaUsed.Should().Be(4 * 21, "each of the 4 arena strings is its own duplicate");
            }
        }

        /// <summary>The 16-byte cells of a <c>StringDType</c> array in C order, as upper-case hex.</summary>
        /// <param name="a">The array.</param>
        /// <returns>One hex string per element.</returns>
        private static string[] Cells(NDArray a)
        {
            byte[] bytes = a.tobytes();
            return Enumerable.Range(0, bytes.Length / 16).Select(i => Convert.ToHexString(bytes, 16 * i, 16)).ToArray();
        }

        /// <summary>
        ///     A copy frees what each destination element held before it copies (NumPy's <c>free_and_copy</c>): short texts
        ///     copied over heap strings release every heap string, and each copied cell is the source's own short cell. The
        ///     copy core's fast path for a short source must therefore stay off a destination that holds heap text. Through
        ///     <c>np.copyto</c> and through a slice assignment.
        /// </summary>
        [TestMethod]
        public void Copy_ShortOverHeapStrings_FreesEachHeapString()
        {
            foreach (bool slice in new[] { false, true })
            {
                using var dst = A(L('q'), L('q'), L('q'), L('q'));
                var allocator = dst.Storage.VStringSlice.Allocator;
                for (int i = 0; i < 4; i++)
                    dst.SetValue(L('h', 600), i); // a longer rewrite moves each element out of its arena slot to the heap
                allocator.HeapStrings.Should().Be(4);
                using var src = A("a", "bb", "ccc", "dddd");
                if (slice)
                    dst[":"] = src;
                else
                    np.copyto(dst, src);
                V(dst).Should().Equal(new object[] { "a", "bb", "ccc", "dddd" });
                allocator.HeapStrings.Should().Be(0, "free_and_copy released each heap string before it copied the short cell");
                Cells(dst).Should().Equal(Cells(src), "the duplicate of a short cell is the cell itself");
            }
        }

        /// <summary>
        ///     An EMPTY source element is <c>NpyString_pack_empty</c> on the destination, which keeps the destination's own
        ///     state: a fresh (all-zero) cell stays all zero, a short or heap cell becomes the short empty string
        ///     (<c>…60</c>), an arena cell keeps its slot offset with size 0 (<c>…40</c>). So an empty source is never a
        ///     16-byte copy of its cell, even when that cell is the short empty string. Cells probed against
        ///     <c>numpy==2.4.2</c>.
        /// </summary>
        [TestMethod]
        public void Copy_EmptySource_IsPackEmptyOnTheDestination()
        {
            using var a = A("ab", L('x'), "cd");
            a.SetValue("", 0); // over a short string: the short empty string
            a.SetValue("", 1); // over an arena string: the slot is kept, size 0
            Cells(a).Should().Equal("00000000000000000000000000000060", "01000000000000000000000000000040",
                                    "63640000000000000000000000000062");
            using var copy = a.copy();
            Cells(copy).Should().Equal("00000000000000000000000000000000", "00000000000000000000000000000000",
                                       "63640000000000000000000000000062");
            using var fresh = np.empty(new Shape(3), T);
            fresh["..."] = a;
            Cells(fresh).Should().Equal(Cells(copy));
            using var shortCells = A("pq", "rs", "tu");
            shortCells["..."] = a;
            Cells(shortCells).Should().Equal("00000000000000000000000000000060", "00000000000000000000000000000060",
                                             "63640000000000000000000000000062");
            using var arena = A(L('y'), L('z'), L('w'));
            arena["..."] = a;
            Cells(arena).Should().Equal("01000000000000000000000000000040", "16000000000000000000000000000040",
                                        "63640000000000000000000000000062");
            using var heap = A(L('y'), L('z'), L('w'));
            for (int i = 0; i < 3; i++)
                heap.SetValue(L('h', 600), i);
            heap["..."] = a;
            Cells(heap).Should().Equal(Cells(shortCells));
            heap.Storage.VStringSlice.Allocator.HeapStrings.Should().Be(0);
            foreach (var x in new[] { copy, fresh, shortCells, arena, heap })
                V(x).Should().Equal(new object[] { "", "", "cd" });
        }

        /// <summary>
        ///     A copy reserves the destination's arena once, for exactly the texts that land in it. NumPy's
        ///     <c>heap_or_arena_allocate</c> puts a new medium / long text in the arena only for a cell whose flags are 0 after
        ///     <c>free_and_copy</c>'s free (a fresh cell, or an NA set on one); a cell that held a short string takes it on the
        ///     heap. So the reserve counts only the former. It changes the arena's capacity, never a cell: the cells are NumPy's
        ///     (probed against <c>numpy==2.4.2</c>).
        /// </summary>
        [TestMethod]
        public void Copy_ReservesArenaSpaceOnlyForTheCellsThatTakeIt()
        {
            using (var src = np.array(new[] { L('x', 30), L('y', 30), "z" }, TN))
            using (var fresh = np.empty(new Shape(3), TN))
            {
                fresh["..."] = src;
                var allocator = fresh.Storage.VStringSlice.Allocator;
                allocator.ArenaCapacity.Should().Be(62, "two 30-byte texts with 1-byte prefixes, reserved at once (the 125 % steps end at 77)");
                allocator.ArenaUsed.Should().Be(62);
                Cells(fresh).Should().Equal("01000000000000001E00000000000040", "20000000000000001E00000000000040",
                                            "7A000000000000000000000000000061");
            }
            using (var src = np.array(new[] { L('x', 30), L('y', 30), L('w', 30) }, TN))
            using (var mixed = np.empty(new Shape(3), TN))
            {
                mixed.SetValue(null, 0); // an NA set on a fresh cell (flags 0x80): its new text takes the arena
                mixed.SetValue("s", 1);  // a short string: its new text takes the heap
                mixed.SetValue("t", 2);
                mixed.SetValue(null, 2); // an NA set on a short string (flags 0xE0): its new text takes the heap
                mixed["..."] = src;
                var allocator = mixed.Storage.VStringSlice.Allocator;
                allocator.ArenaCapacity.Should().Be(31, "only element 0's text lands in the arena");
                allocator.ArenaUsed.Should().Be(31);
                allocator.HeapStrings.Should().Be(2);
                string[] cells = Cells(mixed);
                cells[0].Should().Be("01000000000000001E00000000000040");
                foreach (string heapCell in cells.Skip(1))
                    heapCell.Substring(16).Should().Be("1E000000000000" + "70", "a heap string: size 30, flags INITIALIZED | OUTSIDE_ARENA | LONG");
                V(mixed).Should().Equal(new object[] { L('x', 30), L('y', 30), L('w', 30) });
            }
        }

        /// <summary>
        ///     Copying from a descriptor with an NA object into one without (an unsafe cast) turns NA cells into the NA object's
        ///     <c>str()</c>: <c>'None'</c>, <c>'nan'</c>.
        /// </summary>
        [TestMethod]
        public void Copy_NAIntoNoNA_IsTheNAObjectsStr()
        {
            using (var dst = A("a", "b"))
            using (var src = np.array(new object[] { "x", null }, TN))
            {
                np.copyto(dst, src, "unsafe");
                V(dst).Should().Equal(new object[] { "x", "None" });
            }
            using (var dst = A("a", "b"))
            using (var src = np.array(new object[] { "x", double.NaN }, np.dtypes.StringDType(na_object: double.NaN)))
            {
                np.copyto(dst, src, "unsafe");
                V(dst).Should().Equal(new object[] { "x", "nan" });
            }
            using (var dst = A("a", "b"))
            using (var src = np.array(new object[] { "x", null }, TN))
            {
                Action sameKind = () => np.copyto(dst, src);
                sameKind.Should().Throw<Exception>().WithMessage(
                    "Cannot cast array data from StringDType(na_object=None) to StringDType() according to the rule 'same_kind'");
            }
        }

        /// <summary>The copy core's own checks: null storages, a storage of another layout, a read-only destination, a
        ///     source that does not broadcast, and an empty destination (nothing to copy).</summary>
        [TestMethod]
        public void Copy_Checks()
        {
            using var a = A("a", "b");
            using var u = np.array(new[] { "a", "b" });
            Action nullDst = () => NDVStringCopy.Copy(null, a.Storage);
            nullDst.Should().Throw<ArgumentNullException>();
            Action nullSrc = () => NDVStringCopy.Copy(a.Storage, null);
            nullSrc.Should().Throw<ArgumentNullException>();
            Action notT = () => NDVStringCopy.Copy(a.Storage, u.Storage);
            notT.Should().Throw<InvalidOperationException>();
            using var ro = np.broadcast_to(a, new Shape(3, 2));
            Action readOnly = () => NDVStringCopy.Copy(ro.Storage, a.Storage);
            readOnly.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
            using var three = A("a", "b", "c");
            Action shape = () => NDVStringCopy.Copy(a.Storage, three.Storage);
            shape.Should().Throw<Exception>();
            using var empty = np.empty(new Shape(0), T);
            NDVStringCopy.Copy(empty.Storage, np.empty(new Shape(0), T).Storage);
        }

        // ------------------------------------------------------------------------------------------------------------
        // gathers
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.resize</c> returns NumPy's <c>concatenate(...)[:n].reshape(shape)</c>: a VIEW of a fresh buffer (OWNDATA
        ///     false), its strings duplicated into that buffer's allocator — the gather keeps the numeric route's view structure.
        /// </summary>
        [TestMethod]
        public void Gather_Resize_IsAViewOfAFreshBuffer()
        {
            using var a = A(L('a'), "b", L('c'));
            using var r = np.resize(a, new Shape(5));
            r.flags.owndata.Should().BeFalse();
            r.flags.c_contiguous.Should().BeTrue();
            V(r).Should().Equal(new object[] { L('a'), "b", L('c'), L('a'), "b" });
            np.shares_memory(r, a).Should().BeFalse();
            r.Storage.VStringSlice.Allocator.Should().NotBeSameAs(a.Storage.VStringSlice.Allocator);
        }

        /// <summary>
        ///     The re-home helpers directly: a broadcast raw result is laid out in C order, an empty result needs no strings,
        ///     a multi-source re-home refuses a non-int64 source map, and the flat copy of an empty array is empty.
        /// </summary>
        [TestMethod]
        public void Gather_RehomeHelpers()
        {
            using var a = A(L('a'), "b");
            using (var cells = NDVStringMove.CellArray(a))
            using (var broadcast = np.broadcast_to(cells, new Shape(3, 2)))
            using (var r = NDVStringMove.Rehome(broadcast, a.Storage.VStringSlice, a.dtype))
            {
                r.flags.owndata.Should().BeTrue();
                r.Shape.strides.Should().Equal(2L, 1L);
                V(r).Should().Equal(new object[] { L('a'), "b", L('a'), "b", L('a'), "b" });
            }
            using (var cells = NDVStringMove.CellArray(a))
            using (var which = np.zeros(new Shape(2), NPTypeCode.Int32))
            {
                Action wrongMap = () => NDVStringMove.RehomeMulti(cells, which, new[] { a.Storage.VStringSlice }, a.dtype);
                wrongMap.Should().Throw<ArgumentException>().WithMessage("The source index array of a StringDType gather must be int64.*");
            }
            using (var e = np.empty(new Shape(0), T))
            using (var flat = NDVStringMove.Flatten(e, 'C'))
                flat.size.Should().Be(0);
            using (var none = np.choose(np.zeros(new Shape(0), NPTypeCode.Int64), new[] { A("a")["0:0"], A("b")["0:0"] }))
            {
                none.size.Should().Be(0);
                none.dtype.Equals(T).Should().BeTrue();
            }
        }

        /// <summary>
        ///     <c>np.where</c> with <c>StringDType</c> operands: a <c>U</c> operand joins the common <c>StringDType</c>, a text
        ///     condition is read by its truth (<c>''</c> is False), a numeric condition by non-zero, an empty selection is an
        ///     empty array of the result descriptor.
        /// </summary>
        [TestMethod]
        public void Gather_Where()
        {
            using var a = A(L('a'), "b", "c");
            using (var r = np.where(np.array(new[] { true, false, true }), np.array(new[] { "u", "v", "w" }), a))
                V(r).Should().Equal(new object[] { "u", "b", "w" });
            using (var r = np.where(np.array(new[] { "", "x", "y" }), a, A("z", "z", "z")))
                V(r).Should().Equal(new object[] { "z", "b", "c" });
            using (var r = np.where(np.array(new[] { 0, 1, 2 }), a, A("z", "z", "z")))
                V(r).Should().Equal(new object[] { "z", "b", "c" });
            using (var r = np.where(np.zeros(new Shape(0), NPTypeCode.Boolean), a["0:0"], a["0:0"]))
            {
                r.size.Should().Be(0);
                r.dtype.Equals(T).Should().BeTrue();
            }
        }

        /// <summary>
        ///     A boolean scalar index is NumPy's copy (a new leading axis of length 1, not a view): <c>a[np.True_]</c> is
        ///     <c>(1, n)</c> and shares no memory; <c>a[np.False_]</c> is <c>(0, n)</c>.
        /// </summary>
        [TestMethod]
        public void Gather_BoolScalarIndex()
        {
            using var a = A(L('a'), "b", "c");
            using var t = a[NDArray.Scalar(true)];
            t.Shape.dimensions.Should().Equal(1L, 3L);
            V(t).Should().Equal(new object[] { L('a'), "b", "c" });
            using var f = a[NDArray.Scalar(false)];
            f.Shape.dimensions.Should().Equal(0L, 3L);
        }

        /// <summary>
        ///     NumPy 2.4.2 copies the cells of <c>np.choose</c>, <c>np.put</c> and <c>np.putmask</c> RAW, so arena strings come
        ///     back as garbage or crash the process; NumSharp duplicates every string into the destination's allocator. The
        ///     values asserted are the ones NumPy gives for short strings, which its raw copy gets right.
        /// </summary>
        [TestMethod]
        public void NumPyBug_RawCellCopies_AreCorrectHere()
        {
            using (var r = np.choose(np.array(new[] { 1, 1, 0, 0 }), new[] { A("a", "bc", "def", "g"), A(L('x'), L('y', 30), L('z', 25), L('w', 40)) }))
                V(r).Should().Equal(new object[] { L('x'), L('y', 30), "def", "g" });
            using (var a = A("a", "bc", "def", "g"))
            {
                np.put(a, np.array(new[] { 0, 2 }), np.array(new[] { L('q'), L('r', 18) }));
                V(a).Should().Equal(new object[] { L('q'), "bc", L('r', 18), "g" });
            }
            using (var a = A("a", "bc", "def", "g"))
            {
                np.putmask(a, np.array(new[] { true, false, true, false }), new[] { L('q'), L('r', 18) });
                V(a).Should().Equal(new object[] { L('q'), "bc", L('q'), "g" });
            }
            using (var a = A("a", "bc", "def", "g"))
            {
                np.place(a, np.array(new[] { true, false, true, false }), new[] { L('q'), L('r', 18) });
                V(a).Should().Equal(new object[] { L('q'), "bc", L('r', 18), "g" });
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // scatters and the values of a store
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Fancy and mask assignment: a <c>U</c> array value is cast, a number array becomes each number's <c>str()</c>, a 0-d
        ///     text array converts before the index, an empty selection converts nothing (an invalid-UTF-8 <c>S</c> value is
        ///     never read), a Python int literal under <c>coerce=False</c> is refused while an int ndarray is cast.
        /// </summary>
        [TestMethod]
        public void Scatter_FancyAndMask()
        {
            using var d = A(L('a'), "b", "c");
            d[np.array(new[] { 0, 2 })] = np.array(new[] { "x", "y" });
            V(d).Should().Equal(new object[] { "x", "b", "y" });
            d[np.array(new[] { 0, 2 })] = np.array(new[] { 1, 2 });
            V(d).Should().Equal(new object[] { "1", "b", "2" });
            d[np.array(new[] { 0, 2 })] = np.array("zz");
            V(d).Should().Equal(new object[] { "zz", "b", "zz" });
            d[np.array(new int[0])] = np.array(new[] { new byte[] { 0xFF } });
            V(d).Should().Equal(new object[] { "zz", "b", "zz" });
            d[np.array(new[] { true, false, true })] = np.array(new[] { "m", L('n') });
            V(d).Should().Equal(new object[] { "m", "b", L('n') });

            using var tc = np.array(new[] { "a", "b" }, TC);
            Action literal = () => tc[np.array(new[] { 0 })] = 5;
            literal.Should().Throw<ValueError>().WithMessage("StringDType only allows string data when string coercion is disabled.");
            tc[np.array(new[] { 0 })] = np.array(new[] { 5 });
            V(tc).Should().Equal(new object[] { "5", "b" });
        }

        /// <summary>
        ///     The values of the scatter functions follow NumPy's <c>PyArray_FromAny(values, dtype=StringDType)</c>: Python objects
        ///     (literals, a C# array's elements, None, a boxed number, a list) are stored by setitem — a text keeps its trailing
        ///     NUL, <c>coerce=False</c> refuses a number —, an ndarray is cast.
        /// </summary>
        [TestMethod]
        public void StoreValue_ScatterFunctions()
        {
            using (var a = A("x", "y", "z"))
            {
                np.put(a, np.array(new[] { 0 }), "a\0");
                a.GetValue(0).Should().Be("a\0");
            }
            using (var a = A("x", "y", "z"))
            {
                np.putmask(a, np.array(new[] { true, false, false }), new[] { "a\0" });
                a.GetValue(0).Should().Be("a\0");
            }
            using (var a = A("x", "y", "z"))
            {
                np.place(a, np.array(new[] { false, true, false }), new[] { "b\0" });
                a.GetValue(1).Should().Be("b\0");
            }
            using (var a = A("x", "y", "z"))
            using (var r = np.insert(a, 0, "a\0"))
                V(r).Should().Equal(new object[] { "a\0", "x", "y", "z" });
            using (var a = A("x", "y", "z"))
            {
                np.put_along_axis(a, np.array(new long[] { 0 }), "a\0", 0);
                a.GetValue(0).Should().Be("a\0");
            }
            using (var m = np.array(new[,] { { "x", "y" }, { "z", "w" } }, TN))
            {
                np.fill_diagonal(m, null);
                V(m).Should().Equal(new object[] { null, "y", "z", null });
                np.fill_diagonal(m, new object[] { "p", null });
                V(m).Should().Equal(new object[] { "p", "y", "z", null });
                np.fill_diagonal(m, new List<string> { "l1", "l2" });
                V(m).Should().Equal(new object[] { "l1", "y", "z", "l2" });
                np.fill_diagonal(m, 7);
                V(m).Should().Equal(new object[] { "7", "y", "z", "7" });
                np.fill_diagonal(m, new[,] { { 1.5 } });
                V(m).Should().Equal(new object[] { "1.5", "y", "z", "1.5" });
            }
            using (var tc = np.array(new[] { "x", "y", "z" }, TC))
            {
                Action putInt = () => np.put(tc, np.array(new[] { 0 }), 5);
                putInt.Should().Throw<ValueError>().WithMessage("StringDType only allows string data when string coercion is disabled.");
                Action putmaskList = () => np.putmask(tc, np.array(new[] { true, false, false }), new[] { 5 });
                putmaskList.Should().Throw<ValueError>();
                Action fillDiag = () => np.fill_diagonal(np.array(new[,] { { "x", "y" }, { "z", "w" } }, TC), 5);
                fillDiag.Should().Throw<ValueError>();
                Action insertInt = () => np.insert(tc, 0, 5);
                insertInt.Should().Throw<ValueError>();
                np.put(tc, np.array(new[] { 0 }), np.array(new[] { 5 }));
                tc.GetValue(0).Should().Be("5", "an ndarray is cast, which coerce=False allows");
            }
        }

        /// <summary>The scatter core's own checks: int64 marks, a writeable destination, nothing to do for an empty
        ///     destination, and a mark outside the values (the caller's bug, never a wild read).</summary>
        [TestMethod]
        public void Scatter_MarksChecks()
        {
            using var dst = A("a", "b");
            using var values = A("v");
            using (var marks32 = np.zeros(new Shape(2), NPTypeCode.Int32))
            {
                Action wrongMarks = () => NDVStringMove.ScatterMarks(dst, marks32, values);
                wrongMarks.Should().Throw<ArgumentException>().WithMessage("The marks of a StringDType scatter must be int64.*");
            }
            using (var ro = np.broadcast_to(dst, new Shape(2, 2)))
            using (var marks = np.zeros(new Shape(2, 2), NPTypeCode.Int64))
            {
                Action readOnly = () => NDVStringMove.ScatterMarks(ro, marks, values);
                readOnly.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
            }
            using (var empty = np.empty(new Shape(0), T))
            using (var marks0 = np.zeros(new Shape(0), NPTypeCode.Int64))
                NDVStringMove.ScatterMarks(empty, marks0, values);
            using (var marks = np.array(new long[] { 3, -1 }))
            {
                Action outside = () => NDVStringMove.ScatterMarks(dst, marks, values);
                outside.Should().Throw<IndexOutOfRangeException>().WithMessage("A scatter mark 3 is outside the 1 value elements.");
            }
        }

        /// <summary>
        ///     <c>np.copyto</c> into a <c>StringDType</c> array: a copy onto itself returns at once, a read-only destination is
        ///     NumPy's ValueError, a 0-d <c>where</c> copies everything or nothing.
        /// </summary>
        [TestMethod]
        public void Copyto_SelfReadOnlyAndScalarMask()
        {
            using var a = A(L('a'), "b", "c");
            np.copyto(a, a);
            V(a).Should().Equal(new object[] { L('a'), "b", "c" });
            using (var ro = np.broadcast_to(a, new Shape(2, 3)))
            {
                Action readOnly = () => np.copyto(ro, A("x", "y", "z"));
                readOnly.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
            }
            np.copyto(a, "z", where: np.array(true));
            V(a).Should().Equal(new object[] { "z", "z", "z" });
            np.copyto(a, "q", where: np.array(false));
            V(a).Should().Equal(new object[] { "z", "z", "z" });
        }

        /// <summary>
        ///     <c>NDArray.AsStringArray</c> of a <c>StringDType</c> array: each text, null for a <c>None</c> NA; an NA object that
        ///     is not a string or None (a float NaN) cannot be a C# string and is refused.
        /// </summary>
        [TestMethod]
        public void AsStringArray_NAObjects()
        {
            using (var a = np.array(new object[] { "a", null }, TN))
                NDArray.AsStringArray(a).Should().Equal(new string[] { "a", null });
            using (var a = np.array(new object[] { "a", double.NaN }, np.dtypes.StringDType(na_object: double.NaN)))
            {
                Action nan = () => NDArray.AsStringArray(a);
                nan.Should().Throw<InvalidCastException>().WithMessage("*NA object nan*");
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // np.nditer, tobytes, view, buffers
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     np.nditer needs <c>refs_ok</c> for a <c>StringDType</c> operand; each step's view is a <c>StringDType</c> view of
        ///     the operand's cells sharing its allocator (reads the texts; writes through under <c>readwrite</c>), never a raw
        ///     byte view of the cells; the external-loop chunk is a 1-D view of the run.
        /// </summary>
        [TestMethod]
        public void Nditer_ViewsShareTheAllocator()
        {
            using var a = A(L('a'), "b", L('c'));
            Action noRefs = () => np.nditer(a).Dispose();
            noRefs.Should().Throw<TypeError>().WithMessage("Iterator operand or requested dtype holds references, but the NPY_ITER_REFS_OK flag was not enabled");
            using (var it = np.nditer(a, new[] { "refs_ok" }))
            {
                var seen = new List<object>();
                foreach (var step in it)
                {
                    step[0].dtype.Equals(a.dtype).Should().BeTrue();
                    seen.Add(step[0].Storage.GetAtIndex(0));
                }
                seen.Should().Equal(new object[] { L('a'), "b", L('c') });
            }
            using (var it = np.nditer(a, new[] { "refs_ok" }, new[] { "readwrite" }))
            {
                foreach (var step in it)
                    step[0].SetValue(L('w', 30));
            }
            V(a).Should().Equal(new object[] { L('w', 30), L('w', 30), L('w', 30) });
            using (var it = np.nditer(a, new[] { "refs_ok", "external_loop" }))
            {
                foreach (var step in it)
                    V(step[0]).Should().HaveCount(3);
            }
        }

        /// <summary>
        ///     <c>tobytes()</c>: a buffer already in the requested order gives its own cells; any other layout is ravelled first
        ///     (NumPy 2.4.2's path for a dtype with references), so the copy's arena offsets start at 1 and follow the C walk.
        /// </summary>
        [TestMethod]
        public void Tobytes_RavelsANonContiguousArray()
        {
            using var a = A("a", L('m', 16), "", "bc", L('e', 16), "def");
            var own = a.tobytes();
            BitConverter.ToUInt64(own, 16).Should().Be(1UL, "the base's first arena string");
            BitConverter.ToUInt64(own, 64).Should().Be(18UL, "the base's second arena string: 1 + 16 + 1");
            var view = a["1::3"]; // elements 1 and 4: both arena strings, non-contiguous
            var cells = view.tobytes();
            cells.Length.Should().Be(32);
            BitConverter.ToUInt64(cells, 0).Should().Be(1UL, "the ravelled copy's first arena string");
            BitConverter.ToUInt64(cells, 16).Should().Be(18UL);
            using var f = np.array(new[,] { { L('a', 16), L('b', 16) }, { L('c', 16), L('d', 16) } }, T).T;
            var fc = f.tobytes('F');
            BitConverter.ToUInt64(fc, 0).Should().Be(1UL, "an F-contiguous array's own cells for order F");
            BitConverter.ToUInt64(fc, 16).Should().Be(18UL, "its second cell in memory order: the base's b");
            BitConverter.ToUInt64(fc, 32).Should().Be(35UL, "its third cell: the base's c (1 + 17 + 17)");
        }

        /// <summary>
        ///     <c>view</c>: the array's own descriptor, or an EQUAL one, is a view that keeps reading through the array's
        ///     allocator (NumPy 2.4.2 reads an equal instance's cells through that instance's empty allocator, so its arena
        ///     strings read ''); any other dtype, in either direction, is NumPy's reference refusal.
        /// </summary>
        [TestMethod]
        public void View_OwnDescriptorOnly()
        {
            using var a = A(L('a'), "b");
            using (var same = a.view(a.dtype))
            {
                np.shares_memory(same, a).Should().BeTrue();
                V(same).Should().Equal(new object[] { L('a'), "b" });
            }
            using (var equal = a.view(np.dtypes.StringDType()))
                V(equal).Should().Equal(new object[] { L('a'), "b" });
            Action otherNA = () => a.view(TN);
            otherNA.Should().Throw<TypeError>().WithMessage("Cannot change data-type for array of references.");
            Action toU = () => a.view(np.dtype("U4"));
            toU.Should().Throw<TypeError>().WithMessage("Cannot change data-type for array of references.");
            using var u = np.array(new[] { "ab" });
            Action fromU = () => u.view(T);
            fromU.Should().Throw<TypeError>().WithMessage("Cannot change data-type for array of references.");
        }

        /// <summary>The raw-buffer routes NumPy refuses for a reference dtype: the buffer export and np.frombuffer.</summary>
        [TestMethod]
        public void Buffers_AreRefused()
        {
            using var a = A("a");
            Action mv = () => _ = a.data;
            mv.Should().Throw<ValueError>().WithMessage("cannot include dtype 'numpy.dtypes.StringDType' in a buffer");
            Action fb = () => np.frombuffer(new byte[32], T);
            fb.Should().Throw<ValueError>().WithMessage("cannot create an OBJECT array from memory buffer");
            Action fbString = () => np.frombuffer(new byte[32], "T");
            fbString.Should().Throw<ValueError>().WithMessage("cannot create an OBJECT array from memory buffer");
        }

        // ------------------------------------------------------------------------------------------------------------
        // the remaining movement routes
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Overlapping 1-D copies with a REVERSED destination take NumPy's raw walk after it flips both strides
        ///     (<c>a[2::-1] = a[3:0:-1]</c> is <c>['b','c','d','d']</c>, <c>a[3:0:-1] = a[2::-1]</c> <c>['a','a','b','c']</c>, as
        ///     for <c>U</c>); a source with the same first element and dimensions but another stride is not the same elements,
        ///     so it is copied (NumPy 2.4.2 then empties the self-copied short element, <c>['','c','c','d']</c> — its
        ///     <c>free_and_copy</c> bug; NumSharp keeps it).
        /// </summary>
        [TestMethod]
        public void Copy_ReversedOverlap_And_SameStartOtherStride()
        {
            using (var a = A("a", "b", "c", "d"))
            {
                a["2::-1"] = a["3:0:-1"];
                V(a).Should().Equal(new object[] { "b", "c", "d", "d" });
            }
            using (var a = A("a", "b", "c", "d"))
            {
                a["3:0:-1"] = a["2::-1"];
                V(a).Should().Equal(new object[] { "a", "a", "b", "c" });
            }
            using (var a = A("a", "b", "c", "d"))
            {
                a[":2"] = a["::2"];
                V(a).Should().Equal(new object[] { "a", "c", "c", "d" });
            }
        }

        /// <summary>
        ///     The gather contract for a selection that returns a VIEW of the cell array (a basic slice): the result is the
        ///     matching view of the source — same buffer and allocator, the source's read-only flag kept — not a copy.
        /// </summary>
        [TestMethod]
        public void Gather_ViewOfTheCells_IsAViewOfTheSource()
        {
            using var a = A("a", L('b'), "c", "d");
            using var view = NDVStringMove.Gather(a, cells => cells["1::2"]);
            view.flags.owndata.Should().BeFalse();
            view.Storage.VStringSlice.Allocator.Should().BeSameAs(a.Storage.VStringSlice.Allocator);
            V(view).Should().Equal(new object[] { L('b'), "d" });
            view[0] = "x";
            a.GetValue(1).Should().Be("x", "the view writes through to the source");
            using var ro = np.broadcast_to(a, new Shape(2, 4));
            using var roView = NDVStringMove.Gather(ro, cells => cells["0"]);
            roView.Shape.IsWriteable.Should().BeFalse("a view of a read-only source stays read-only");

            // Raw cells that are a READ-ONLY view of a temporary keep that flag on the re-homed view (RehomeView).
            using var cellsOfA = NDVStringMove.CellArray(a);
            using var temp = cellsOfA.copy();
            using var tempView = temp["1::2"];
            tempView.setflags(write: false);
            using var rehomed = NDVStringMove.Rehome(tempView, a.Storage.VStringSlice, a.dtype);
            rehomed.Shape.IsWriteable.Should().BeFalse();
            rehomed.flags.owndata.Should().BeFalse("a view of a fresh StringDType buffer, as the numeric route's view");
            V(rehomed).Should().Equal(new object[] { "x", "d" });
        }

        /// <summary>
        ///     <c>np.take</c> into an <c>out=</c> array, the boolean mask of the LEADING axes of a 2-D array (rows), and an
        ///     empty sequence given to <c>np.fill_diagonal</c> (the diagonal is left as it is) — each as NumPy.
        /// </summary>
        [TestMethod]
        public void Take_Out_LeadingMask_EmptyDiagonal()
        {
            using var a = A("a", "b", "c");
            using var o = np.empty(new Shape(2), T);
            var r = np.take(a, np.array(new[] { 1, 0 }), @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            V(o).Should().Equal(new object[] { "b", "a" });

            using var m = np.array(new[,] { { "a", "b" }, { "c", "d" } }, T);
            using var rows = m[np.array(new[] { true, false })];
            rows.shape.Should().Equal(1L, 2L);
            rows.flags.owndata.Should().BeTrue();
            V(rows).Should().Equal(new object[] { "a", "b" });
            using var typedMask = np.arange(2) == 0; // a comparison is an NDArray<bool>: the typed mask indexer
            using var typedRows = m[typedMask];
            V(typedRows).Should().Equal(new object[] { "a", "b" });

            np.fill_diagonal(m, new string[0]);
            V(m).Should().Equal(new object[] { "a", "b", "c", "d" });
        }

        /// <summary>
        ///     A fancy or masked store of an ARRAY that must be converted into <c>StringDType</c> converts it only when the
        ///     selection holds an element: an all-False mask with a <c>U</c> value whose code unit has no UTF-8 form succeeds,
        ///     a selecting mask raises the cast's TypeError (NumPy converts while it writes).
        /// </summary>
        [TestMethod]
        public void MaskedStore_ConvertsOnlyWhenSomethingIsSelected()
        {
            using var t = np.array(new[] { "a", "b", "c" }, TC);
            using var bad = np.array(new uint[] { 0xD800 }).view(np.dtype("<U1"));
            t[np.array(new[] { false, false, false })] = bad;
            V(t).Should().Equal(new object[] { "a", "b", "c" });
            t[np.array(new long[0])] = bad; // an empty fancy selection converts nothing either
            V(t).Should().Equal(new object[] { "a", "b", "c" });
            using var none = np.empty(new Shape(0), TC);
            none[np.array(new long[0])] = bad; // nor a store into an empty array
            Action store = () => t[np.array(new[] { true, false, false })] = bad;
            store.Should().Throw<TypeError>().WithMessage("Invalid unicode code point found");
        }

        /// <summary>
        ///     A <c>StringDType</c> MASK (np.putmask / np.place) is read through StringDType's bool cast with its NA rules: an
        ///     NA of a <c>None</c> NA object is False (<c>['', 'x', None]</c> selects only element 1, probed).
        /// </summary>
        [TestMethod]
        public void TextMasks_FollowTheBoolCast()
        {
            using var mask = np.array(new object[] { "", "x", null }, TN);
            using (var a = np.zeros(new Shape(3)))
            {
                np.putmask(a, mask, 7.0);
                a.ToArray<double>().Should().Equal(0.0, 7.0, 0.0);
            }
            using (var a = np.zeros(new Shape(3)))
            {
                np.place(a, mask, new[] { 7.0 });
                a.ToArray<double>().Should().Equal(0.0, 7.0, 0.0);
            }
        }

        /// <summary>
        ///     <c>np.place</c> INTO a <c>StringDType</c> array: a C# literal is a Python object stored by setitem, an array of
        ///     numbers is cast to text, both cycled over the selected elements. (NumPy 2.4.2 crashes the interpreter on
        ///     <c>np.place</c> over a StringDType array — a segmentation fault; NumSharp gives the documented answer of
        ///     <c>np.place</c>.)
        /// </summary>
        [TestMethod]
        public void Place_IntoStringDType()
        {
            using (var t = A("a", "b", "c"))
            {
                np.place(t, np.array(new[] { true, false, true }), "z");
                V(t).Should().Equal(new object[] { "z", "b", "z" });
            }
            using (var t = A("a", "b", "c"))
            {
                np.place(t, np.array(new[] { true, false, true }), np.array(new[] { 1, 2 }));
                V(t).Should().Equal(new object[] { "1", "b", "2" });
            }
        }

        /// <summary>
        ///     <c>astype(same descriptor instance, copy: false)</c> returns the array itself (NumPy:
        ///     <c>t.astype(t.dtype, copy=False) is t</c>), while an equal but fresh descriptor still copies (views are legal only
        ///     between descriptors that share an allocator).
        /// </summary>
        [TestMethod]
        public void Astype_NoCopy_OnlyForTheSameDescriptor()
        {
            using var t = A("a", L('b'));
            ReferenceEquals(t.astype(t.dtype, false), t).Should().BeTrue();
            using var copy = t.astype(T, false);
            ReferenceEquals(copy, t).Should().BeFalse();
            copy.Storage.VStringSlice.Allocator.Should().NotBeSameAs(t.Storage.VStringSlice.Allocator);
        }

        /// <summary>
        ///     Computing on <c>StringDType</c> is phase 2: a comparison of two StringDType arrays (or StringDType with
        ///     <c>U</c>) refuses with NumSharp's phase-2 text — except an EMPTY broadcast, which the shape decides — and a
        ///     reduction refuses before reading the cells.
        /// </summary>
        [TestMethod]
        public void Computation_IsPhaseTwo()
        {
            using var a = A("a");
            using var b = A("a");
            Action eq = () => _ = a == b;
            eq.Should().Throw<StringComputationNotSupportedException>().WithMessage("equal is not supported for StringDType() arrays yet*");
            Action ne = () => _ = a != np.array(new[] { "a" });
            ne.Should().Throw<StringComputationNotSupportedException>().WithMessage("not_equal is not supported*");
            using var empty = np.array(new string[0], T);
            using (var r = empty == b)
            {
                r.size.Should().Be(0);
                r.dtype.Should().Be(np.@bool);
            }
            Action sum = () => np.sum(a);
            sum.Should().Throw<NotSupportedException>().WithMessage("This operation does not support StringDType() arrays yet*");
        }
    }
}
