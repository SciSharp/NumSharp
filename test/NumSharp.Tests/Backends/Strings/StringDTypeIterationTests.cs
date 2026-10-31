using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     np.nditer over NEP 55 <c>StringDType</c> operands with buffering and casting (strings phase 1, P1.7): each buffer
    ///     owns a string allocator, a transfer duplicates every string into the destination's allocator or runs the
    ///     <c>StringDType</c> cast of the pair, the views read and write the strings of the memory they are on, and the
    ///     iterator's setter stores like <c>a[...] = value</c> (<c>StringDType</c>'s setitem for a Python object). Expected
    ///     values were probed against <c>numpy==2.4.2</c>.
    /// </summary>
    [TestClass]
    public class StringDTypeIterationTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor.</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A fresh <c>StringDType(na_object=None)</c> descriptor.</summary>
        private static DType TN => np.dtypes.StringDType(na_object: null);

        /// <summary>A fresh <c>StringDType(coerce=False)</c> descriptor.</summary>
        private static DType TC => np.dtypes.StringDType(coerce: false);

        /// <summary>A text longer than 15 UTF-8 bytes: an arena string, not stored in its cell.</summary>
        private const string Long = "a long string over fifteen";

        /// <summary>The probe array of every case: a short, a two-character, an arena and an empty string.</summary>
        /// <returns>A new <c>StringDType()</c> array.</returns>
        private static NDArray Sample() => np.array(new[] { "a", "bb", Long, "" }, T);

        /// <summary>
        ///     A 2-D array whose transpose, walked in C order, has no constant stride, so the iterator BUFFERS it even without
        ///     a cast (a contiguous operand without a cast is read in place: its "buffered" walk never fills a buffer).
        ///     <c>m.T</c> in C order reads <c>a, '', bb, eee, Long, Long2</c>.
        /// </summary>
        /// <returns>A new (2, 3) <c>StringDType()</c> array.</returns>
        private static NDArray Grid() => np.array(new[,] { { "a", "bb", Long }, { "", "eee", Long + "2" } }, T);

        /// <summary>A text of 30 characters: a MEDIUM string (an arena allocation).</summary>
        /// <param name="c">The repeated character.</param>
        /// <returns>The text.</returns>
        private static string Medium(char c) => new string(c, 30);

        /// <summary>The elements of a fresh C-contiguous array (texts, NA objects or numbers).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values in C order.</returns>
        private static object[] V(NDArray a) => Enumerable.Range(0, (int)a.size).Select(i => a.Storage.GetAtIndex(i)).ToArray();

        /// <summary>Walks a single-operand iterator and returns each step's element (NumPy's <c>[x.item() for x in it]</c>).</summary>
        /// <param name="it">The iterator (disposed here).</param>
        /// <returns>The elements in iteration order.</returns>
        private static List<object> Items(np.NDIterator it)
        {
            var r = new List<object>();
            using (it)
                foreach (var x in it)
                    r.Add(x[0].item());
            return r;
        }

        /// <summary>Walks an external-loop iterator and returns each chunk's elements.</summary>
        /// <param name="it">The iterator (disposed here).</param>
        /// <returns>One list per chunk.</returns>
        private static List<List<object>> Chunks(np.NDIterator it)
        {
            var r = new List<List<object>>();
            using (it)
                foreach (var x in it)
                {
                    var chunk = x[0];
                    r.Add(Enumerable.Range(0, (int)chunk.size).Select(i => chunk[i].item()).ToList());
                }
            return r;
        }

        /// <summary>Cancels a scheduled allocation failure that did not fire.</summary>
        [TestCleanup]
        public void Disarm() => VStringAllocator.DisarmFaultInjection();

        // ------------------------------------------------------------------------------------------------------------
        // Buffered reads and writes without a cast
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     A buffered iterator reads every string through its buffer, whatever the buffer size, and each view is a
        ///     <c>StringDType</c> view with the array's descriptor (NumPy's <c>[StringDType()]</c>), never the raw 16-byte cells.
        /// </summary>
        [TestMethod]
        public void Buffered_ReadsEveryString_AsStringDTypeViews()
        {
            using var t = Sample();
            Items(np.nditer(t, new[] { "refs_ok", "buffered" })).Should().Equal("a", "bb", Long, "");
            Items(np.nditer(t, new[] { "refs_ok", "buffered" }, buffersize: 2)).Should().Equal("a", "bb", Long, "");
            using (var it = np.nditer(t, new[] { "refs_ok", "buffered" }))
                foreach (var x in it)
                    x[0].dtype.Equals(t.dtype).Should().BeTrue("a buffered StringDType view keeps the operand's descriptor");

            // A layout that really fills buffers (NumPy: ['a', '', 'bb', 'eee', Long, Long + '2']).
            using var m = Grid();
            Items(np.nditer(m.T, new[] { "refs_ok", "buffered" }, order: 'C', buffersize: 4)).Should().Equal("a", "", "bb", "eee", Long, Long + "2");
            using (var it = np.nditer(m.T, new[] { "refs_ok", "buffered" }, order: 'C', buffersize: 4))
                foreach (var x in it)
                    x[0].dtype.Equals(m.dtype).Should().BeTrue("a view of a StringDType buffer carries the iteration descriptor");
        }

        /// <summary>
        ///     Under <c>external_loop</c> a buffered iterator hands out one chunk per buffer fill (NumPy, buffersize 3:
        ///     <c>[['a', 'bb', Long], ['']]</c>); an unbuffered one hands out the whole contiguous run.
        /// </summary>
        [TestMethod]
        public void Buffered_ExternalLoop_ChunksFollowTheBuffer()
        {
            using var t = Sample();
            var buffered = Chunks(np.nditer(t, new[] { "refs_ok", "buffered", "external_loop" }, buffersize: 3));
            buffered.Should().HaveCount(2);
            buffered[0].Should().Equal("a", "bb", Long);
            buffered[1].Should().Equal("");
            var plain = Chunks(np.nditer(t, new[] { "refs_ok", "external_loop" }));
            plain.Should().ContainSingle().Which.Should().Equal("a", "bb", Long, "");

            // Chunks of a filled buffer (NumPy, m.T in C order, buffersize 4: [['a', '', 'bb', 'eee'], [Long, Long + '2']]).
            using var m = Grid();
            var filled = Chunks(np.nditer(m.T, new[] { "refs_ok", "buffered", "external_loop" }, order: 'C', buffersize: 4));
            filled.Should().HaveCount(2);
            filled[0].Should().Equal("a", "", "bb", "eee");
            filled[1].Should().Equal(Long, Long + "2");
        }

        /// <summary>
        ///     A buffered <c>readwrite</c> operand writes every string back, through 0-d views and through external-loop
        ///     chunks (NumPy: <c>['a!', 'bb!', Long + '!', '!']</c>).
        /// </summary>
        [TestMethod]
        public void Buffered_ReadWrite_WritesEveryStringBack()
        {
            using var a = Sample();
            using (var it = np.nditer(a, new[] { "refs_ok", "buffered" }, new[] { "readwrite" }))
                foreach (var x in it)
                    x[0][Slice.Ellipsis] = (string)x[0].item() + "!";
            V(a).Should().Equal("a!", "bb!", Long + "!", "!");

            using var b = Sample();
            using (var it = np.nditer(b, new[] { "refs_ok", "buffered", "external_loop" }, new[] { "readwrite" }, buffersize: 3))
                foreach (var x in it)
                {
                    var chunk = x[0];
                    for (int i = 0; i < chunk.size; i++)
                        chunk[i] = (string)chunk[i].item() + "?";
                }
            V(b).Should().Equal("a?", "bb?", Long + "?", "?");

            // Through filled buffers: every window is written back into the operand's strings when the iterator moves on.
            using var m = Grid();
            using (var it = np.nditer(m.T, new[] { "refs_ok", "buffered" }, new[] { "readwrite" }, order: 'C', buffersize: 4))
                foreach (var x in it)
                    x[0][Slice.Ellipsis] = (string)x[0].item() + "!";
            V(m).Should().Equal("a!", "bb!", Long + "!", "!", "eee!", Long + "2!");
        }

        /// <summary>
        ///     Buffered layouts read in NumPy's iteration order: a reversed view in memory order under <c>'K'</c>, a 2-D array
        ///     column by column under <c>'F'</c>, a transposed view in memory order.
        /// </summary>
        [TestMethod]
        public void Buffered_Layouts_ReadInIterationOrder()
        {
            using var t = Sample();
            Items(np.nditer(t["::-1"], new[] { "refs_ok", "buffered" })).Should().Equal("a", "bb", Long, "");
            using var m = np.array(new[,] { { "a", "b", "c" }, { "d", "e", "f" } }, T);
            Items(np.nditer(m, new[] { "refs_ok", "buffered" }, order: 'F')).Should().Equal("a", "d", "b", "e", "c", "f");
            Items(np.nditer(m.T, new[] { "refs_ok", "buffered" })).Should().Equal("a", "b", "c", "d", "e", "f");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Buffered casts
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Casts out of <c>StringDType</c> through the buffer: to <c>U5</c> (truncated, <c>unsafe</c>), to a wide <c>U</c>
        ///     (<c>same_kind</c>), to numbers (parsed) and to bool (an empty string is False); the TN cast keeps the texts and
        ///     gives TN views. <c>safe</c> refuses <c>T → U5</c> and <c>T → int64</c> with NumPy's TypeError texts.
        /// </summary>
        [TestMethod]
        public void BufferedCast_FromStringDType()
        {
            using var t = Sample();
            Items(np.nditer(t, new[] { "refs_ok", "buffered" }, op_dtypes: new DType[] { "U5" }, casting: "unsafe"))
                .Should().Equal("a", "bb", "a lon", "");
            Items(np.nditer(t, new[] { "refs_ok", "buffered" }, op_dtypes: new DType[] { "U30" }, casting: "same_kind"))
                .Should().Equal("a", "bb", Long, "");
            Items(np.nditer(t, new[] { "refs_ok", "buffered" }, op_dtypes: new DType[] { "?" }, casting: "unsafe"))
                .Should().Equal(true, true, true, false);

            using var numbers = np.array(new[] { "12", "-7" }, T);
            Items(np.nditer(numbers, new[] { "refs_ok", "buffered" }, op_dtypes: new DType[] { "i8" }, casting: "unsafe"))
                .Should().Equal(12L, -7L);
            Items(np.nditer(numbers, new[] { "refs_ok", "buffered" }, op_dtypes: new DType[] { "f8" }, casting: "unsafe"))
                .Should().Equal(12.0, -7.0);

            var tn = TN;
            using (var it = np.nditer(t, new[] { "refs_ok", "buffered" }, op_dtypes: new[] { tn }, casting: "unsafe"))
            {
                var seen = new List<object>();
                foreach (var x in it)
                {
                    x[0].dtype.Equals(tn).Should().BeTrue();
                    seen.Add(x[0].item());
                }
                seen.Should().Equal("a", "bb", Long, "");
            }

            Action toU5 = () => np.nditer(t, new[] { "refs_ok", "buffered" }, op_dtypes: new DType[] { "U5" }).Dispose();
            toU5.Should().Throw<TypeError>()
                .WithMessage("Iterator operand 0 dtype could not be cast from StringDType() to dtype('<U5') according to the rule 'safe'");
            Action toI8 = () => np.nditer(numbers, new[] { "refs_ok", "buffered" }, op_dtypes: new DType[] { "i8" }).Dispose();
            toI8.Should().Throw<TypeError>()
                .WithMessage("Iterator operand 0 dtype could not be cast from StringDType() to dtype('int64') according to the rule 'safe'");
        }

        /// <summary>
        ///     Casts into <c>StringDType</c> through the buffer: numbers and bools become their <c>str()</c>
        ///     (<c>'1e+20'</c>, <c>'True'</c>), <c>U</c> text keeps its characters and the views carry the requested descriptor;
        ///     <c>safe</c> refuses <c>U → StringDType</c> (NumPy's text names <c>dtype('&lt;U3')</c>).
        /// </summary>
        [TestMethod]
        public void BufferedCast_ToStringDType()
        {
            var t = T;
            using var ints = np.array(new long[] { 1, 22, -3 });
            Items(np.nditer(ints, new[] { "refs_ok", "buffered" }, op_dtypes: new[] { t })).Should().Equal("1", "22", "-3");
            using var floats = np.array(new[] { 1.5, 1e20 });
            Items(np.nditer(floats, new[] { "refs_ok", "buffered" }, op_dtypes: new[] { t })).Should().Equal("1.5", "1e+20");
            using var bools = np.array(new[] { true, false });
            Items(np.nditer(bools, new[] { "refs_ok", "buffered" }, op_dtypes: new[] { t })).Should().Equal("True", "False");

            using var u = np.array(new[] { "x", "yy", "zzz" });
            using (var it = np.nditer(u, new[] { "refs_ok", "buffered" }, op_dtypes: new[] { t }, casting: "unsafe"))
            {
                var seen = new List<object>();
                foreach (var x in it)
                {
                    x[0].dtype.Equals(t).Should().BeTrue();
                    seen.Add(x[0].item());
                }
                seen.Should().Equal("x", "yy", "zzz");
            }
            Action safe = () => np.nditer(u, new[] { "refs_ok", "buffered" }, op_dtypes: new[] { t }).Dispose();
            safe.Should().Throw<TypeError>()
                .WithMessage("Iterator operand 0 dtype could not be cast from dtype('<U3') to StringDType() according to the rule 'safe'");
        }

        /// <summary>
        ///     A text that does not parse as the requested number raises NumPy's <c>int()</c> text at the buffer fill that
        ///     reads it, and the iterator's buffers are released with it.
        /// </summary>
        [TestMethod]
        public void BufferedCast_UnparsableText_RaisesIntText()
        {
            using var t = Sample();
            Action act = () =>
            {
                using var it = np.nditer(t, new[] { "refs_ok", "buffered" }, op_dtypes: new DType[] { "i8" }, casting: "unsafe");
                foreach (var _ in it) { }
            };
            act.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: 'a'");
        }

        /// <summary>
        ///     NA values follow the casts' NA rules through the buffer: a TN array reads <c>None</c>, its cast to T or to
        ///     <c>U5</c> writes the NA object's name (<c>'None'</c>), and a readwrite pass through a T buffer writes texts back
        ///     (NumPy: <c>['a?', 'None?']</c>).
        /// </summary>
        [TestMethod]
        public void BufferedCast_NAValues()
        {
            using var na = np.array(new object[] { "a", null, "c" }, TN);
            Items(np.nditer(na, new[] { "refs_ok", "buffered" })).Should().Equal(new object[] { "a", null, "c" });
            Items(np.nditer(na, new[] { "refs_ok", "buffered" }, op_dtypes: new[] { T }, casting: "unsafe")).Should().Equal("a", "None", "c");
            Items(np.nditer(na, new[] { "refs_ok", "buffered" }, op_dtypes: new DType[] { "U5" }, casting: "unsafe")).Should().Equal("a", "None", "c");

            using var rw = np.array(new object[] { "a", null }, TN);
            using (var it = np.nditer(rw, new[] { "refs_ok", "buffered" }, new[] { "readwrite" }, op_dtypes: new[] { T }, casting: "unsafe"))
                foreach (var x in it)
                    x[0][Slice.Ellipsis] = (string)x[0].item() + "?";
            V(rw).Should().Equal("a?", "None?");
        }

        /// <summary>
        ///     A readwrite operand of another dtype through a cast buffer: int64 through a T buffer is parsed back
        ///     (<c>[10, 20, 30]</c>), U2 through a T buffer is truncated back (<c>'a '</c>), T through a U3 buffer is written
        ///     as the U3 texts.
        /// </summary>
        [TestMethod]
        public void BufferedCast_ReadWrite_ThroughACastBuffer()
        {
            using var ints = np.array(new long[] { 1, 2, 3 });
            using (var it = np.nditer(ints, new[] { "refs_ok", "buffered" }, new[] { "readwrite" }, op_dtypes: new[] { T }, casting: "unsafe"))
                foreach (var x in it)
                    x[0][Slice.Ellipsis] = (string)x[0].item() + "0";
            V(ints).Should().Equal(10L, 20L, 30L);

            using var u = np.array(new[] { "ab", "cd" });
            using (var it = np.nditer(u, new[] { "refs_ok", "buffered" }, new[] { "readwrite" }, op_dtypes: new[] { T }, casting: "unsafe"))
                foreach (var x in it)
                    x[0][Slice.Ellipsis] = "a much longer string than fits";
            V(u).Should().Equal("a ", "a ");

            using var t = np.array(new[] { "ab", "cd", Long }, T);
            using (var it = np.nditer(t, new[] { "refs_ok", "buffered" }, new[] { "readwrite" }, op_dtypes: new DType[] { "U3" }, casting: "unsafe"))
                foreach (var x in it)
                    x[0][Slice.Ellipsis] = "xyz";
            V(t).Should().Equal("xyz", "xyz", "xyz");
        }

        /// <summary>
        ///     <c>common_dtype</c> over a T and a U operand casts the U operand to T, which <c>safe</c> refuses (NumPy's text
        ///     names operand 1); over T and int64 no common DType exists (NumPy's DTypePromotionError).
        /// </summary>
        [TestMethod]
        public void CommonDType_WithStringDType()
        {
            using var t = np.array(new[] { "a", "bb", Long }, T);
            using var u = np.array(new[] { "x", "yy", "zzz" });
            using var n = np.array(new long[] { 1, 22, -3 });
            Action withU = () => np.nditer(new[] { t, u }, new[] { "refs_ok", "buffered", "common_dtype" }).Dispose();
            withU.Should().Throw<TypeError>()
                .WithMessage("Iterator operand 1 dtype could not be cast from dtype('<U3') to StringDType() according to the rule 'safe'");
            Action withInt = () => np.nditer(new[] { t, n }, new[] { "refs_ok", "buffered", "common_dtype" }).Dispose();
            withInt.Should().Throw<DTypePromotionError>()
                .WithMessage("The DType <class 'numpy.dtypes.StringDType'> could not be promoted by <class 'numpy.dtypes.Int64DType'>.*");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Multi-operand, copy routes, masks, reductions
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     A buffered iterator allocates a T output (NumPy: <c>['xx', 'yyyy', 'zzzzzz']</c>, <c>StringDType()</c>), and the
        ///     writes into its buffer reach the allocated array.
        /// </summary>
        [TestMethod]
        public void Buffered_AllocatedStringDTypeOutput()
        {
            using var u = np.array(new[] { "x", "yy", "zzz" });
            using var it = np.nditer(new NDArray[] { u, null }, new[] { "refs_ok", "buffered" },
                new[] { new[] { "readonly" }, new[] { "writeonly", "allocate" } }, op_dtypes: new DType[] { null, T });
            foreach (var x in it)
                x[1][Slice.Ellipsis] = string.Concat(Enumerable.Repeat((string)x[0].item(), 2));
            var output = it.operands[1];
            output.dtype.Equals(T).Should().BeTrue();
            V(output).Should().Equal("xx", "yyyy", "zzzzzz");
        }

        /// <summary>
        ///     A T input with an allocated T output, buffered: NumPy 2.4.2 raises <c>MemoryError: Failed to load string in
        ///     StringDType getitem</c> reading its own output buffer (and <c>String deallocation failed in clear loop</c> when
        ///     the iterator closes); NumSharp returns the values the loop wrote.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Buffered_StringDTypeInAndOut_GivesTheValuesNumPyFailsToRead()
        {
            using var t = Sample();
            using (var it = np.nditer(new NDArray[] { t, null }, new[] { "refs_ok", "buffered", "external_loop" },
                       new[] { new[] { "readonly" }, new[] { "writeonly", "allocate" } }, op_dtypes: new DType[] { null, T }, buffersize: 3))
            {
                foreach (var x in it)
                    for (int i = 0; i < x[0].size; i++)
                        x[1][i] = (string)x[0][i].item() + "-" + (string)x[0][i].item();
                V(it.operands[1]).Should().Equal("a-a", "bb-bb", Long + "-" + Long, "-");
            }

            using var counts = np.array(new long[] { 1, 2, 3, 4 });
            using (var it = np.nditer(new NDArray[] { t, counts, null }, new[] { "refs_ok", "buffered" },
                       new[] { new[] { "readonly" }, new[] { "readonly" }, new[] { "writeonly", "allocate" } }, op_dtypes: new DType[] { null, null, T }))
            {
                foreach (var x in it)
                    x[2][Slice.Ellipsis] = string.Concat(Enumerable.Repeat((string)x[0].item(), (int)(long)x[1].item()));
                V(it.operands[2]).Should().Equal("a", "bbbb", Long + Long + Long, "");
            }
        }

        /// <summary>
        ///     The copy routes cast without buffers: a <c>copy</c> operand reads a temporary of the requested dtype (T → U5
        ///     truncated, U → T with T views); without <c>copy</c> or buffering the cast is NumPy's TypeError; an
        ///     <c>updateifcopy</c> operand writes its temporary back when the iterator closes, in both directions.
        /// </summary>
        [TestMethod]
        public void CopyRoutes_CastWithoutBuffers()
        {
            using var t = Sample();
            Items(np.nditer(new[] { t }, new[] { "refs_ok" }, new[] { new[] { "readonly", "copy" } }, op_dtypes: new DType[] { "U5" }, casting: "unsafe"))
                .Should().Equal("a", "bb", "a lon", "");
            using var u = np.array(new[] { "x", "yy", "zzz", "w" });
            Items(np.nditer(new[] { u }, new[] { "refs_ok" }, new[] { new[] { "readonly", "copy" } }, op_dtypes: new[] { T }, casting: "unsafe"))
                .Should().Equal("x", "yy", "zzz", "w");
            Action uncopied = () => np.nditer(t, new[] { "refs_ok" }, op_dtypes: new DType[] { "U5" }, casting: "unsafe").Dispose();
            uncopied.Should().Throw<TypeError>()
                .WithMessage("Iterator operand required copying or buffering, but neither copying nor buffering was enabled");
            Action uncopiedNA = () => np.nditer(new[] { t }, new[] { "refs_ok" }, op_dtypes: new[] { TN }).Dispose();
            uncopiedNA.Should().Throw<TypeError>()
                .WithMessage("Iterator operand required copying or buffering, but neither copying nor buffering was enabled");

            using var a = np.array(new[] { "ab", "cd" });
            using (var it = np.nditer(new[] { a }, new[] { "refs_ok" }, new[] { new[] { "readwrite", "updateifcopy" } }, op_dtypes: new[] { T }, casting: "unsafe"))
                foreach (var x in it)
                    x[0][Slice.Ellipsis] = "zz" + (string)x[0].item();
            V(a).Should().Equal("zz", "zz");

            using var b = np.array(new[] { "ab", "cd", Long }, T);
            using (var it = np.nditer(new[] { b }, new[] { "refs_ok" }, new[] { new[] { "readwrite", "updateifcopy" } }, op_dtypes: new DType[] { "U40" }, casting: "unsafe"))
                foreach (var x in it)
                    x[0][Slice.Ellipsis] = (string)x[0].item() + "!";
            V(b).Should().Equal("ab!", "cd!", Long + "!");
        }

        /// <summary>
        ///     A writemasked T operand read through a U9 buffer writes back only where the mask is true (NumPy:
        ///     <c>['newa', 'b', 'newc', 'd']</c>).
        /// </summary>
        [TestMethod]
        public void WriteMasked_ThroughACastBuffer()
        {
            using var a = np.array(new[] { "a", "b", "c", "d" }, T);
            using var mask = np.array(new[] { true, false, true, false });
            using (var it = np.nditer(new[] { a, mask }, new[] { "refs_ok", "buffered" },
                       new[] { new[] { "readwrite", "writemasked" }, new[] { "readonly", "arraymask" } }, op_dtypes: new DType[] { "U9", null }, casting: "unsafe"))
                foreach (var x in it)
                    x[0][Slice.Ellipsis] = "new" + (string)x[0].item();
            V(a).Should().Equal("newa", "b", "newc", "d");
        }

        /// <summary>
        ///     A T reduction operand (the output's axis 0 mapped to the iterator's axis 1) accumulates concatenations, buffered
        ///     and unbuffered (NumPy: <c>['ac', 'bd']</c>), also when the reduction operand is cast (a TN, U8 or T buffer over a
        ///     T or U8 output: the buffer is written back through the cast of the pair); two operands over one T array alias
        ///     it. The buffer sizes here hold the whole reduction; a buffer shorter than the reduced axis is gated by
        ///     <c>OpenBugs.NditerBufferedReduce.cs</c> (fixed by the port of NumPy's buffered iteration, 2026-10-14) and the
        ///     oracle's <c>titer</c> section.
        /// </summary>
        [TestMethod]
        public void Reductions_AndAliasedOperands()
        {
            foreach (var buffered in new[] { false, true })
            {
                using var a = np.array(new[,] { { "a", "b" }, { "c", "d" } }, T);
                using var outp = np.array(new[] { "", "" }, T);
                var flags = buffered ? new[] { "refs_ok", "buffered", "reduce_ok" } : new[] { "refs_ok", "reduce_ok" };
                using (var it = np.nditer(new[] { a, outp }, flags, new[] { new[] { "readonly" }, new[] { "readwrite" } },
                           op_axes: new[] { new[] { 0, 1 }, new[] { -1, 0 } }))
                    foreach (var x in it)
                        x[1][Slice.Ellipsis] = (string)x[1].item() + (string)x[0].item();
                V(outp).Should().Equal(new object[] { "ac", "bd" }, buffered ? "buffered" : "unbuffered");
            }

            // The reduction operand itself cast, so its buffer is really filled and written back through the cast of the pair
            // (NumPy: ['ac', 'bd'] for the TN, U8 and T buffers; ['ab', 'cd'] when the output follows the iterator's axis 0).
            foreach (var (output, buffer, axes, expected) in new (DType, DType, int[], string[])[]
                     {
                         (T, TN, new[] { -1, 0 }, new[] { "ac", "bd" }),
                         (T, TN, new[] { 0, -1 }, new[] { "ab", "cd" }),
                         (T, "U8", new[] { -1, 0 }, new[] { "ac", "bd" }),
                         ("U8", T, new[] { -1, 0 }, new[] { "ac", "bd" }),
                     })
            {
                using var a = np.array(new[,] { { "a", "b" }, { "c", "d" } }, T);
                using var outp = np.array(new[] { "", "" }, output);
                using (var it = np.nditer(new[] { a, outp }, new[] { "refs_ok", "buffered", "reduce_ok" }, new[] { new[] { "readonly" }, new[] { "readwrite" } },
                           op_axes: new[] { new[] { 0, 1 }, axes }, op_dtypes: new DType[] { null, buffer }, casting: "unsafe"))
                    foreach (var x in it)
                        x[1][Slice.Ellipsis] = (string)x[1].item() + (string)x[0].item();
                V(outp).Should().Equal(expected, $"{output} output through a {buffer} buffer, op_axes [{string.Join(", ", axes)}]");
            }

            using var shared = np.array(new[] { "a", "b", "c" }, T);
            using (var it = np.nditer(new[] { shared, shared }, new[] { "refs_ok", "buffered" }, new[] { new[] { "readonly" }, new[] { "writeonly" } }))
                foreach (var x in it)
                    x[1][Slice.Ellipsis] = string.Concat(Enumerable.Repeat((string)x[0].item(), 3));
            V(shared).Should().Equal("aaa", "bbb", "ccc");
        }

        /// <summary>
        ///     np.nested_iters over a T array, plain and buffered: the inner level reads each row (NumPy:
        ///     <c>[['a', 'b', 'c'], ['d', 'e', 'f']]</c>).
        /// </summary>
        [TestMethod]
        public void NestedIters_StringDType()
        {
            foreach (var buffered in new[] { false, true })
            {
                using var m = np.array(new[,] { { "a", "b", "c" }, { "d", "e", "f" } }, T);
                var flags = buffered ? new[] { "refs_ok", "buffered" } : new[] { "refs_ok" };
                var its = np.nested_iters(m, new[] { new[] { 0 }, new[] { 1 } }, flags);
                var rows = new List<List<object>>();
                try
                {
                    foreach (var _ in its[0])
                    {
                        var row = new List<object>();
                        foreach (var y in its[1])
                            row.Add(y[0].item());
                        rows.Add(row);
                    }
                }
                finally
                {
                    foreach (var it in its)
                        it.Dispose();
                }
                rows.Should().HaveCount(2);
                rows[0].Should().Equal("a", "b", "c");
                rows[1].Should().Equal("d", "e", "f");
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // The iterator protocol with StringDType buffers
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The protocol members over a buffered T operand: <c>copy()</c> (its own buffers and strings), <c>reset()</c>,
        ///     <c>iterrange</c> (<c>(0, 4)</c>), <c>value</c> under <c>external_loop</c>, <c>delay_bufalloc</c> + <c>reset()</c>,
        ///     an empty array (<c>zerosize_ok</c>), a 0-d array, and <c>itviews</c> refused with NumPy's text.
        /// </summary>
        [TestMethod]
        public void Protocol_WithStringDTypeBuffers()
        {
            using var t = Sample();
            using (var it = np.nditer(t, new[] { "refs_ok", "buffered" }))
                Items(it.copy()).Should().Equal("a", "bb", Long, "");
            using (var it = np.nditer(t, new[] { "refs_ok", "buffered" }, op_dtypes: new[] { TN }))
                Items(it.copy()).Should().Equal("a", "bb", Long, "");

            using (var it = np.nditer(t, new[] { "refs_ok", "buffered" }))
            {
                var e = it.GetEnumerator();
                e.MoveNext();
                it.reset();
                Items(it).Should().Equal("a", "bb", Long, "");
            }

            using (var it = np.nditer(t, new[] { "refs_ok", "buffered", "ranged" }, buffersize: 2))
            {
                it.iterrange.Should().Be((0L, 4L));
                Items(it).Should().Equal("a", "bb", Long, "");
            }

            using (var it = np.nditer(t, new[] { "refs_ok", "buffered", "external_loop" }))
            {
                var chunk = it.value[0];
                Enumerable.Range(0, (int)chunk.size).Select(i => chunk[i].item()).Should().Equal("a", "bb", Long, "");
            }

            using (var it = np.nditer(t, new[] { "refs_ok", "buffered", "delay_bufalloc" }))
            {
                it.reset();
                Items(it).Should().Equal("a", "bb", Long, "");
            }

            using var empty = np.array(new string[0], T);
            Items(np.nditer(empty, new[] { "refs_ok", "buffered", "zerosize_ok" })).Should().BeEmpty();
            using var scalarBase = np.array(new[] { "zz" }, T);
            using var scalar = scalarBase.reshape(new Shape());
            Items(np.nditer(scalar, new[] { "refs_ok", "buffered" })).Should().Equal("zz");

            using (var it = np.nditer(t, new[] { "refs_ok", "buffered" }))
            {
                Action views = () => _ = it.itviews;
                views.Should().Throw<ValueError>().WithMessage("cannot provide an iterator view when buffering is enabled");
            }
        }

        /// <summary>
        ///     <c>remove_multi_index</c> on a buffered reduction whose merged axes no longer fit the buffering setup makes the
        ///     iterator run the setup again (<c>RebuildBufferingSetup</c>: the buffers are freed and allocated anew, NOT NumPy,
        ///     which keeps the old setup and walks wrong elements or crashes): a <c>StringDType</c> buffer then has a NEW string
        ///     allocator, and the refill must write its strings through that one. The cached transfer of the operand still held
        ///     the freed allocator of the first buffer, so the refilled cells (texts over 15 bytes: arena strings, not stored in
        ///     the cell) could not be read: "Failed to load string in StringDType getitem". Every source string is read once,
        ///     in C order (the source is C-contiguous and the reduced axis stays an axis of its own).
        /// </summary>
        [TestMethod]
        public void RemoveMultiIndex_RebuiltBuffers_FillThroughTheNewAllocator()
        {
            var texts = Enumerable.Range(0, 120).Select(i => $"element-number-{i:D3}-long-text").ToArray();
            using var flat = np.array(texts);
            using var src = flat.reshape(2, 3, 4, 5);
            // op_axes of the output, the buffer size: each pair rebuilds the setup (a smaller buffer fills core by core and
            // keeps it, so it never had the stale transfer).
            foreach (var (axes, bufsize) in new[] { (new[] { 0, -1, 1, 2 }, 13L), (new[] { 0, -1, 1, 2 }, 40L), (new[] { -1, 0, 1, 2 }, 13L), (new[] { -1, 0, 1, 2 }, 40L) })
            {
                var outShape = Enumerable.Range(0, 4).Where(d => axes[d] >= 0).Select(d => (long)src.shape[d]).ToArray();
                using var output = np.zeros(new Shape(outShape));
                var read = new List<object>();
                using (var it = np.nditer(new[] { src, output }, new[] { "buffered", "multi_index", "reduce_ok", "refs_ok" },
                           new[] { new[] { "readonly" }, new[] { "readwrite" } }, op_dtypes: new DType[] { T, np.float64 },
                           casting: "unsafe", op_axes: new[] { null, axes }, buffersize: bufsize))
                {
                    it.remove_multi_index();
                    do
                    {
                        using var x = it[0];
                        read.Add(x.item());
                    } while (it.iternext());
                }
                read.Should().Equal(texts, $"op_axes [{string.Join(", ", axes)}], buffersize {bufsize}");
            }
        }

        /// <summary>
        ///     <c>copy()</c> taken in the middle of a buffered walk owns its buffered strings: it walks on from the same
        ///     position after the original has moved past (and freed) the strings of its own buffer. The operand is a
        ///     transpose walked in C order, so its buffers are really filled (a K-order walk of it is contiguous and in place).
        /// </summary>
        [TestMethod]
        public void Copy_MidIteration_OwnsItsBufferedStrings()
        {
            using var flat = np.array(Enumerable.Range(0, 12).Select(i => Medium((char)('a' + i))).ToArray(), T);
            using var m = flat.reshape(3, 4);
            using var it = np.nditer(m.T, new[] { "refs_ok", "buffered" }, order: 'C', buffersize: 4);
            var e = it.GetEnumerator();
            e.MoveNext();
            e.MoveNext();
            e.MoveNext();
            using var c = it.copy();
            var rest = new List<object>();
            while (e.MoveNext())
                rest.Add(it[0].item());
            var fromCopy = new List<object>();
            foreach (var x in c)
                fromCopy.Add(x[0].item());
            fromCopy.Should().HaveCountGreaterThanOrEqualTo(rest.Count);
            fromCopy.Skip(fromCopy.Count - rest.Count).Should().Equal(rest);
            // The C walk of m.T reads m[0, 0], m[1, 0], m[2, 0], m[0, 1], ...: every element the copy yields is that walk's,
            // read from the copy's OWN buffer for the window it was taken in.
            var walk = Enumerable.Range(0, 12).Select(k => Medium((char)('a' + (k % 3) * 4 + k / 3))).ToList();
            fromCopy.Should().Equal(walk.Skip(walk.Count - fromCopy.Count));
        }

        // ------------------------------------------------------------------------------------------------------------
        // The setter: it[i] = value
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>it[0] = value</c> on a T operand stores like <c>a[...] = value</c> (StringDType's setitem for a Python object,
        ///     a cast for an array), buffered or not: null is the NA object (TN) or <c>'None'</c> (T); a C# number, bool or
        ///     NaN stores its <c>str()</c>; a U array is cast; an int64 array stores <c>'7'</c>. Under <c>coerce=False</c> every
        ///     non-text object raises NumPy's ValueError, while an array is still cast.
        /// </summary>
        [TestMethod]
        public void Setter_StoresLikeSetitem()
        {
            var cases = new (string Name, Func<NDArray> Value, object T, object TN, string TCError)[]
            {
                ("None", () => null, "None", null, "StringDType only allows string data when string coercion is disabled."),
                ("'xy'", () => "xy", "xy", "xy", null),
                ("5", () => 5, "5", "5", "StringDType only allows string data when string coercion is disabled."),
                ("1.5", () => 1.5, "1.5", "1.5", "StringDType only allows string data when string coercion is disabled."),
                ("True", () => true, "True", "True", "StringDType only allows string data when string coercion is disabled."),
                ("nan", () => double.NaN, "nan", "nan", "StringDType only allows string data when string coercion is disabled."),
                ("U array", () => np.array(new[] { "q" }), "q", "q", null),
                ("0-d U array", () => np.array(new[] { "z" }).reshape(new Shape()), "z", "z", null),
                ("int64 array", () => np.array(new long[] { 7 }), "7", "7", null),
            };
            foreach (var buffered in new[] { false, true })
            {
                var flags = buffered ? new[] { "refs_ok", "buffered" } : new[] { "refs_ok" };
                foreach (var c in cases)
                {
                    foreach (var (descr, expected, error) in new (DType, object, string)[] { (T, c.T, null), (TN, c.TN, null), (TC, c.T, c.TCError) })
                    {
                        using var a = np.array(new[] { "a", "b" }, descr);
                        Action store = () =>
                        {
                            using var it = np.nditer(new[] { a }, flags, new[] { new[] { "readwrite" } });
                            foreach (var _ in it)
                                it[0] = c.Value();
                        };
                        if (error != null)
                        {
                            store.Should().Throw<ValueError>().WithMessage(error);
                            continue;
                        }
                        store();
                        V(a).Should().Equal(new[] { expected, expected }, $"{descr} {(buffered ? "buffered" : "unbuffered")} it[0] = {c.Name}");
                    }
                }
            }
        }

        /// <summary>
        ///     The setter over an external-loop chunk stores into every element of it: null is NA for TN and <c>'None'</c>
        ///     for T, buffered or not; an array of the chunk's length stores element by element.
        /// </summary>
        [TestMethod]
        public void Setter_ExternalLoopChunks()
        {
            foreach (var flags in new[] { new[] { "refs_ok", "external_loop" }, new[] { "refs_ok", "buffered", "external_loop" } })
            {
                using var tn = np.array(new[] { "a", "b", "c" }, TN);
                using (var it = np.nditer(new[] { tn }, flags, new[] { new[] { "readwrite" } }))
                    foreach (var _ in it)
                        it[0] = null;
                V(tn).Should().Equal(new object[] { null, null, null });

                using var t = np.array(new[] { "a", "b", "c" }, T);
                using (var it = np.nditer(new[] { t }, flags, new[] { new[] { "readwrite" } }))
                    foreach (var _ in it)
                        it[0] = null;
                V(t).Should().Equal("None", "None", "None");
            }

            using var p = np.array(new[] { "a", "b", "c" }, T);
            using (var it = np.nditer(new[] { p }, new[] { "refs_ok", "external_loop" }, new[] { new[] { "readwrite" } }))
                foreach (var _ in it)
                    it[0] = np.array(new[] { "p", "q", "r" });
            V(p).Should().Equal("p", "q", "r");
        }

        /// <summary>
        ///     Null through a cast buffer: into a TN (or T) buffer over a U2 operand it is NA, written back as the NA object's
        ///     name and truncated (<c>'No'</c>); into a U5 buffer over a T or TN operand it is <c>str(None)</c>, written back as
        ///     the text <c>'None'</c>.
        /// </summary>
        [TestMethod]
        public void Setter_NullThroughACastBuffer()
        {
            foreach (var buffer in new[] { TN, T })
            {
                using var u = np.array(new[] { "ab", "cd" });
                using (var it = np.nditer(new[] { u }, new[] { "refs_ok", "buffered" }, new[] { new[] { "readwrite" } }, op_dtypes: new[] { buffer }, casting: "unsafe"))
                    foreach (var _ in it)
                        it[0] = null;
                V(u).Should().Equal("No", "No");
            }
            foreach (var descr in new[] { T, TN })
            {
                using var t = np.array(new[] { "ab", "cd" }, descr);
                using (var it = np.nditer(new[] { t }, new[] { "refs_ok", "buffered" }, new[] { new[] { "readwrite" } }, op_dtypes: new DType[] { "U5" }, casting: "unsafe"))
                    foreach (var _ in it)
                        it[0] = null;
                V(t).Should().Equal("None", "None");
            }
        }

        /// <summary>
        ///     Null is Python's <c>None</c> in a numeric operand too, stored by setitem's rules: NaN (NumPy's positive
        ///     <c>NPY_NAN</c>, <c>0x7ff8000000000000</c>) into a float operand, <c>nan+nanj</c> into a complex one, <c>False</c>
        ///     into a bool one, and Python's <c>int()</c> TypeError into an integer one, which stores nothing. The setter
        ///     refused the null for every numeric dtype before. Probed against NumPy 2.4.2.
        /// </summary>
        [TestMethod]
        public void Setter_NullIntoANumericOperand_IsNumPysNone()
        {
            using var f = np.array(new[] { 1.0, 2.0 });
            using (var it = np.nditer(new[] { f }, null, new[] { new[] { "readwrite" } }))
                it[0] = null;
            BitConverter.DoubleToInt64Bits(f.GetDouble(0)).Should().Be(0x7ff8000000000000L);
            f.GetDouble(1).Should().Be(2.0);

            using var c = np.array(new[] { new System.Numerics.Complex(1, 0), new System.Numerics.Complex(2, 0) });
            using (var it = np.nditer(new[] { c }, null, new[] { new[] { "readwrite" } }))
                it[0] = null;
            var first = (System.Numerics.Complex)c.Storage.GetAtIndex(0);
            double.IsNaN(first.Real).Should().BeTrue();
            double.IsNaN(first.Imaginary).Should().BeTrue();

            using var b = np.array(new[] { true, true });
            using (var it = np.nditer(new[] { b }, null, new[] { new[] { "readwrite" } }))
                it[0] = null;
            V(b).Should().Equal(false, true);

            using var i = np.array(new long[] { 1, 2 });
            Action store = () =>
            {
                using var it = np.nditer(new[] { i }, null, new[] { new[] { "readwrite" } });
                it[0] = null;
            };
            store.Should().Throw<TypeError>()
                .WithMessage("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
            V(i).Should().Equal(1L, 2L);
        }

        /// <summary>
        ///     A reduction operand's external-loop chunk along its reduced axis has stride 0, which NumPy's views and its
        ///     setter's temporary keep writeable: <c>it[1] = v</c> and <c>y[...] = v</c> store into the one slot (an array
        ///     value leaves its LAST element there), for float64 and for T, while an element store <c>y[1] = v</c> worked
        ///     already. A value that does not broadcast to the chunk is NumPy's ValueError.
        /// </summary>
        [TestMethod]
        public void StrideZeroReductionChunks_TakeStores()
        {
            NDArray Run(bool text, Action<np.NDIterator, NDArray, NDArray> store)
            {
                var input = text ? np.array(new[,] { { "a", "b", "c" }, { "d", "e", "f" } }, T) : np.arange(6.0).reshape(2, 3);
                var outp = text ? np.array(new[] { "", "" }, T) : np.zeros(new Shape(2));
                using (var it = np.nditer(new[] { input, outp }, new[] { "refs_ok", "reduce_ok", "external_loop" },
                           new[] { new[] { "readonly" }, new[] { "readwrite" } }, op_axes: new[] { new[] { 0, 1 }, new[] { 0, -1 } }))
                    foreach (var x in it)
                    {
                        x[1].Shape.strides.Should().Equal(new long[] { 0 }, "the reduced axis is the inner one");
                        store(it, x[0], x[1]);
                    }
                input.Dispose();
                return outp;
            }

            using (var r = Run(false, (it, x, y) => it[1] = 7.0)) V(r).Should().Equal(7.0, 7.0);
            using (var r = Run(false, (it, x, y) => it[1] = x)) V(r).Should().Equal(2.0, 5.0);
            using (var r = Run(false, (it, x, y) => y[Slice.Ellipsis] = 7.0)) V(r).Should().Equal(7.0, 7.0);
            using (var r = Run(false, (it, x, y) => y[Slice.Ellipsis] = x)) V(r).Should().Equal(2.0, 5.0);
            using (var r = Run(false, (it, x, y) => y[1] = 9.0)) V(r).Should().Equal(9.0, 9.0);
            using (var r = Run(true, (it, x, y) => it[1] = "z")) V(r).Should().Equal("z", "z");
            using (var r = Run(true, (it, x, y) => it[1] = x)) V(r).Should().Equal("c", "f");
            using (var r = Run(true, (it, x, y) => y[Slice.Ellipsis] = "z")) V(r).Should().Equal("z", "z");
            using (var r = Run(true, (it, x, y) => y[Slice.Ellipsis] = x)) V(r).Should().Equal("c", "f");
            using (var r = Run(true, (it, x, y) => it[1] = null)) V(r).Should().Equal("None", "None");

            Action mismatch = () => Run(false, (it, x, y) => it[1] = np.array(new[] { 1.0, 2.0 })).Dispose();
            mismatch.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (2,) into shape (3,)");
        }

        // ------------------------------------------------------------------------------------------------------------
        // String lifetime: every buffer's allocator is released
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Buffered walks over every route (a plain buffer, casts both ways, a readwrite cast, <c>copy()</c>, an allocated
        ///     output, nested iterators) leave no string allocator alive once their iterators are disposed: each buffer's
        ///     allocator is freed with its buffer, deterministically (no collection needed).
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // counts the process-wide live StringDType allocators
        public void Buffers_ReleaseTheirAllocators()
        {
            using var t = np.array(new[] { "a", Medium('m'), Long, "" }, T);
            using var u = np.array(new[] { "x", "yy", "zzz", "w" });
            using var ints = np.array(new long[] { 1, 2, 3, 4 });
            using var grid = Grid();
            for (int attempt = 0; ; attempt++)
            {
                GcQuiescence.CollectYoung();
                var window = GcQuiescence.OpenWindow();
                long baseline = VStringAllocator.LiveAllocators;

                Items(np.nditer(t, new[] { "refs_ok", "buffered" }, buffersize: 2));
                Items(np.nditer(grid.T, new[] { "refs_ok", "buffered" }, order: 'C', buffersize: 4));
                using (var it = np.nditer(grid.T, new[] { "refs_ok", "buffered" }, new[] { "readwrite" }, order: 'C', buffersize: 4))
                {
                    var e = it.GetEnumerator();
                    e.MoveNext();
                    Items(it.copy());
                }
                Items(np.nditer(t, new[] { "refs_ok", "buffered" }, op_dtypes: new DType[] { "U30" }, casting: "same_kind"));
                Items(np.nditer(u, new[] { "refs_ok", "buffered" }, op_dtypes: new[] { T }, casting: "unsafe"));
                Items(np.nditer(ints, new[] { "refs_ok", "buffered" }, op_dtypes: new[] { T }));
                using (var it = np.nditer(t, new[] { "refs_ok", "buffered" }, buffersize: 2))
                {
                    var e = it.GetEnumerator();
                    e.MoveNext();
                    Items(it.copy());
                }
                using (var it = np.nditer(new NDArray[] { u, null }, new[] { "refs_ok", "buffered" },
                           new[] { new[] { "readonly" }, new[] { "writeonly", "allocate" } }, op_dtypes: new DType[] { null, T }))
                {
                    foreach (var x in it)
                        x[1][Slice.Ellipsis] = (string)x[0].item() + "!";
                    it.operands[1].Dispose();
                }

                long after = VStringAllocator.LiveAllocators;
                if (GcQuiescence.Undisturbed(window))
                {
                    after.Should().Be(baseline, "every buffer's allocator is freed with the iterator");
                    return;
                }
                if (attempt == 4)
                    Assert.Inconclusive("Every measurement window was disturbed by a collection.");
            }
        }

        /// <summary>
        ///     An allocation failure while a buffer is filled — at construction (the first fill) or at a later step — raises
        ///     NumPy's <c>MemoryError</c> text of the string copy, and the iterator's buffers and their allocators are released
        ///     with it (the live allocator count returns to its baseline). Two buffered routes: a cast to another
        ///     <c>StringDType</c> descriptor (fills of 2 strings), and a 3-D operand whose two inner axes are transposed, walked in
        ///     C order with a buffer of 4 (no cast, but NumPy's buffering setup buffers it: one stride cannot walk it across
        ///     the core of 2 x 2 it chose, and every fill covers one block of the outer axis — 2 fills of 4 strings). A
        ///     single-stride operand without a cast is read in place and allocates nothing: the 2-D transpose with a buffer of 2
        ///     this test used before is one such case (NumPy marks it BUFNEVER). The texts grow from fill to fill, so the second
        ///     fill cannot reuse the arena space the first one freed: every string of the walk is a new allocation, and failing
        ///     one past the first fill's count fails the second fill.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // arms the process-wide StringDType allocator fault injection (VStringAllocator.FailAllocationAfter) and counts live allocators
        public void BufferFill_AllocationFailure_ReleasesTheBuffers()
        {
            var texts = new[] { new string('a', 16), new string('b', 17), new string('c', 60), new string('d', 61) };
            using var a = np.array(texts, T);
            using var flat = np.array(texts, T);
            var texts3 = new[]
            {
                new string('e', 16), new string('f', 17), new string('g', 18), new string('h', 19),
                new string('i', 60), new string('j', 61), new string('k', 62), new string('l', 63),
            };
            using var flat3 = np.array(texts3, T);
            using var m3 = flat3.reshape(2, 2, 2);
            var routes = new (string Name, Func<np.NDIterator> Open, int[] Successes)[]
            {
                ("cast to TN", () => np.nditer(a, new[] { "refs_ok", "buffered" }, op_dtypes: new[] { TN }, buffersize: 2),
                    new[] { 0, 1, 2, 3 }),
                ("inner axes transposed in C order",
                    () => np.nditer(m3.transpose(new[] { 0, 2, 1 }), new[] { "refs_ok", "buffered" }, order: 'C', buffersize: 4),
                    new[] { 0, 1, 3, 4, 5, 7 }),
            };
            foreach (var route in routes)
            foreach (var successes in route.Successes)
            {
                for (int attempt = 0; ; attempt++)
                {
                    GcQuiescence.CollectYoung();
                    var window = GcQuiescence.OpenWindow();
                    long baseline = VStringAllocator.LiveAllocators;
                    VStringAllocator.FailAllocationAfter(successes);
                    try
                    {
                        Action walk = () =>
                        {
                            using var it = route.Open();
                            foreach (var _ in it) { }
                        };
                        walk.Should().Throw<OutOfMemoryException>($"{route.Name}, after {successes} allocations")
                            .WithMessage("Failed to allocate string in string to string cast");
                    }
                    finally
                    {
                        VStringAllocator.DisarmFaultInjection();
                    }
                    long after = VStringAllocator.LiveAllocators;
                    if (GcQuiescence.Undisturbed(window))
                    {
                        after.Should().Be(baseline, $"{route.Name}: the buffers are released after a failure after {successes} allocations");
                        break;
                    }
                    if (attempt == 4)
                        Assert.Inconclusive("Every measurement window was disturbed by a collection.");
                }
            }
            V(a).Should().Equal(texts);
            V(flat).Should().Equal(texts);
            V(flat3).Should().Equal(texts3);
        }
    }
}
