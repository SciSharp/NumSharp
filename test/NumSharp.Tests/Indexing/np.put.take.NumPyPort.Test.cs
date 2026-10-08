using System;
using AwesomeAssertions;

namespace NumSharp.Tests.Indexing
{
    /// <summary>
    ///     <c>np.put</c> / <c>np.take</c> as NumPy's <c>PyArray_PutTo</c> / <c>PyArray_TakeFrom</c>, and the
    ///     <c>clipmode</c> converter every mode-taking selection function shares. Every expected value, type and text is
    ///     NumPy 2.4.2's for the same call.
    /// </summary>
    /// <remarks>
    ///     Found as neighbours of numpy.ma's put (2026-10-08): <c>np.put</c> ignored an overlap between the target and its
    ///     values or indices (silent wrong values), <c>put</c> / <c>take</c> / <c>place</c> / <c>putmask</c> /
    ///     <c>unravel_index</c> read an F-order operand of another dtype in memory order (<c>astype</c>'s 'K' layout under a
    ///     linear kernel), the index-dtype TypeError was skipped for an empty index array, and the errors were
    ///     <see cref="IndexOutOfRangeException"/> / <see cref="ArgumentOutOfRangeException"/> / ArgumentException with
    ///     NumSharp's own clip-mode text.
    /// </remarks>
    [TestClass]
    public class np_put_take_numpy_port_tests
    {
        private static long[] Longs(NDArray a) => a.flatten().ToArray<long>();

        /// <summary>
        ///     NumPy copies the target first when it may share memory with the values or the indices, so every value and
        ///     index is read before any write. NumSharp wrote in place: <c>np.put(a, [1, 2, 3], a[0:3])</c> gave
        ///     <c>[0, 0, 0, 0, 4]</c>, and <c>np.put(a, a, v)</c> raised an IndexError on an index it had just written.
        /// </summary>
        [TestMethod]
        public void Put_OverlappingValuesOrIndices_ReadTheOriginal()
        {
            var a = np.arange(5);
            np.put(a, np.array(new[] { 1, 2, 3 }), a["0:3"]);
            Longs(a).Should().Equal(0, 0, 1, 2, 4);

            a = np.arange(5);
            np.put(a, np.array(new[] { 4, 3, 2, 1, 0 }), a);
            Longs(a).Should().Equal(4, 3, 2, 1, 0);

            a = np.array(new long[] { 1, 2, 3, 0 });
            np.put(a, a, np.array(new long[] { 9, 8, 7, 6 }));
            Longs(a).Should().Equal(6, 9, 8, 7);

            // A reversed view of the same buffer.
            var d = np.arange(6.0);
            np.put(d, np.array(new[] { 5, 4, 3 }), d["::-1"]);
            d.ToArray<double>().Should().Equal(0, 1, 2, 3, 4, 5);
        }

        /// <summary>
        ///     NumPy's raise loop writes as it goes into a contiguous target (the indices before the bad one are written),
        ///     while a non-contiguous target is written through a copy that NumPy discards on the error.
        /// </summary>
        [TestMethod]
        public void Put_RaiseMode_PartialWrites_AreNumPys()
        {
            var a = np.arange(5);
            var act = () => np.put(a, np.array(new[] { 0, 1, 9, 2 }), np.array(new long[] { 7, 8, 9, 10 }));
            act.Should().Throw<IndexError>().WithMessage("index 9 is out of bounds for axis 0 with size 5");
            Longs(a).Should().Equal(7, 8, 2, 3, 4);

            var b = np.arange(10);
            var strided = b["::2"];
            act = () => np.put(strided, np.array(new[] { 0, 1, 9, 2 }), np.array(new long[] { 7, 8, 9, 10 }));
            act.Should().Throw<IndexError>();
            Longs(b).Should().Equal(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        }

        /// <summary>
        ///     F-order operands of another dtype are read in C order (NumPy converts them with CARRAY). The casts used
        ///     <c>astype</c>'s default 'K' and kept them F-ordered under kernels that read linearly.
        /// </summary>
        [TestMethod]
        public void SelectionKernels_ReadFOrderOperandsInCOrder()
        {
            var fValues = np.asfortranarray(np.array(new[,] { { 10, 11 }, { 12, 13 } }));      // int32, F
            var fIdx16 = np.asfortranarray(np.array(new short[,] { { 3, 2 }, { 1, 0 } }));
            var fIdx32 = np.asfortranarray(np.array(new[,] { { 3, 2 }, { 1, 0 } }));
            var src = np.array(new long[] { 100, 101, 102, 103 });

            var a = np.zeros(4, np.int64);
            np.put(a, np.array(new[] { 0, 1, 2, 3 }), fValues);
            Longs(a).Should().Equal(10, 11, 12, 13);

            a = np.zeros(4, np.int64);
            np.put(a, fIdx16, src);
            Longs(a).Should().Equal(103, 102, 101, 100);
            a = np.zeros(4, np.int64);
            np.put(a, fIdx32, src);
            Longs(a).Should().Equal(103, 102, 101, 100);

            Longs(np.take(src, fIdx16)).Should().Equal(103, 102, 101, 100);
            Longs(np.take(src, fIdx32)).Should().Equal(103, 102, 101, 100);
            Longs(np.take(np.arange(8).reshape(2, 4), fIdx16, axis: 1)).Should().Equal(3, 2, 1, 0, 7, 6, 5, 4);

            var fMask = np.asfortranarray(np.array(new sbyte[,] { { 1, 1 }, { 0, 0 } }));
            a = np.zeros(4, np.int64);
            np.putmask(a, fMask, np.array(new long[] { 7, 8, 9, 10 }));
            Longs(a).Should().Equal(7, 8, 0, 0);
            a = np.zeros(4, np.int64);
            np.place(a, fMask, np.array(new long[] { 7, 8 }));
            Longs(a).Should().Equal(7, 8, 0, 0);
            a = np.zeros(4, np.int64);
            np.place(a, np.array(new[] { true, true, true, true }), fValues);
            Longs(a).Should().Equal(10, 11, 12, 13);

            var coords = np.unravel_index(fIdx16, new[] { 2, 2 });
            Longs(coords[0]).Should().Equal(1, 1, 0, 0);
            Longs(coords[1]).Should().Equal(1, 0, 1, 0);
        }

        /// <summary>
        ///     <c>PyArray_PutTo</c>'s order: the mode (parsed with the arguments), the writeability, the index cast — an
        ///     empty float index array too —, the empty target, the value packing, then the no-op of empty values.
        /// </summary>
        [TestMethod]
        public void Put_ChecksInNumPysOrder()
        {
            var ro = np.broadcast_to(np.array(new[] { 1, 2, 3 }), new Shape(3));
            var act = () => np.put(ro, np.array(new[] { 0 }), np.array(new[] { 7 }), "bogus");
            act.Should().Throw<ValueError>().WithMessage("clipmode must be one of 'clip', 'raise', or 'wrap' (got 'bogus')");
            act = () => np.put(ro, np.array(new[] { 0.0 }), np.array(new[] { 7 }));
            act.Should().Throw<ValueError>().WithMessage("put: output array is read-only");

            act = () => np.put(np.arange(4), np.array(new double[0]), np.array(new[] { 7 }));
            act.Should().Throw<TypeError>().WithMessage("Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'");
            act = () => np.put(np.arange(4), np.array(new[] { 0.0 }), np.array(new int[0]));
            act.Should().Throw<TypeError>();
            act = () => np.put(np.zeros(0), np.array(new[] { 0.0 }), np.array(new[] { 7 }));
            act.Should().Throw<TypeError>();
            act = () => np.put(np.arange(4).astype(np.int8), np.array(new[] { 0.0 }), 300);
            act.Should().Throw<TypeError>();

            act = () => np.put(np.zeros(0, np.int8), np.array(new[] { 0 }), 300);
            act.Should().Throw<IndexError>().WithMessage("cannot replace elements of an empty array");
            act = () => np.put(np.arange(4).astype(np.int8), np.array(new long[0]), 300);
            act.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");

            var a = np.arange(4);
            np.put(a, np.array(new[] { 0 }), np.array(new long[] { 7 }), null);   // None is raise
            Longs(a).Should().Equal(7, 1, 2, 3);
        }

        /// <summary>
        ///     <c>PyArray_TakeFrom</c>'s order: the axis (a 0-d source is raveled, so only 0 / -1), the index cast, the out
        ///     checks, then the empty axis, then the gather — whose raise loop checks every index even when the slabs are
        ///     empty, unless the outer block is empty.
        /// </summary>
        [TestMethod]
        public void Take_ChecksInNumPysOrder()
        {
            var zeroD = NDArray.Scalar(5L);
            var act = () => np.take(zeroD, np.array(new[] { 0 }), axis: 1);
            act.Should().Throw<AxisError>().WithMessage("axis 1 is out of bounds for array of dimension 1*");
            Longs(np.take(zeroD, np.array(new[] { 0 }), axis: -1)).Should().Equal(5);
            np.take(zeroD, np.array(new[] { 0 }), axis: 0).shape.Should().Equal(1L);

            act = () => np.take(np.arange(4), np.array(new[] { 0.5 }), axis: 3);
            act.Should().Throw<AxisError>();
            act = () => np.take(np.arange(4), np.array(new[] { 0.5 }), @out: np.zeros(3));
            act.Should().Throw<TypeError>().WithMessage("Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'same_kind'");
            act = () => np.take(np.arange(4), np.array(new double[0]));
            act.Should().Throw<TypeError>();

            act = () => np.take(np.zeros(0), np.array(new[] { 0 }), @out: np.zeros(3));
            act.Should().Throw<ValueError>().WithMessage("output array does not match result of ndarray.take");
            act = () => np.take(np.arange(4), np.array(new[] { 9 }), @out: np.zeros(3, np.int64));
            act.Should().Throw<ValueError>().WithMessage("output array does not match result of ndarray.take");
            var ro = np.broadcast_to(np.zeros(1, np.int64), new Shape(1));
            act = () => np.take(np.arange(4), np.array(new[] { 9 }), @out: ro);
            act.Should().Throw<ValueError>().WithMessage("WRITEBACKIFCOPY base is read-only");

            act = () => np.take(np.zeros(0), np.array(new[] { 0 }));
            act.Should().Throw<IndexError>().WithMessage("cannot do a non-empty take from an empty axes.");
            act = () => np.take(np.zeros(0), np.array(new[] { 0 }), mode: "clip");
            act.Should().Throw<IndexError>().WithMessage("cannot do a non-empty take from an empty axes.");
            np.take(np.zeros(0), np.array(new long[0])).size.Should().Be(0);

            act = () => np.take(np.zeros(new Shape(3, 0)), np.array(new[] { 9 }), axis: 0);
            act.Should().Throw<IndexError>().WithMessage("index 9 is out of bounds for axis 0 with size 3");
            act = () => np.take(np.zeros(new Shape(3, 0, 0)), np.array(new[] { 0 }), axis: 1);
            act.Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 1 with size 0");
            np.take(np.zeros(new Shape(3, 0, 0)), np.array(new[] { 0 }), axis: 1, mode: "clip").shape.Should().Equal(3L, 1L, 0L);
            np.take(np.zeros(new Shape(0, 3)), np.array(new[] { 9 }), axis: 1).shape.Should().Equal(0L, 1L);

            act = () => np.take(np.arange(4), np.array(new[] { 9 }));
            act.Should().Throw<IndexError>().WithMessage("index 9 is out of bounds for axis 0 with size 4");
            act = () => np.take(np.arange(6).reshape(2, 3), np.array(new[] { 5 }), axis: 1);
            act.Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 1 with size 3");
        }

        /// <summary>
        ///     NumPy's <c>PyArray_ClipmodeConverter</c> for every mode-taking selection function: the first character
        ///     selects, the exact lowercase spelling is required, an empty string is refused, None is raise.
        /// </summary>
        [TestMethod]
        public void ClipMode_IsNumPysConverter_Everywhere()
        {
            void Refuses(Action call, string text) => call.Should().Throw<ValueError>().Where(e => e.Message == text);
            const string generic = "clipmode must be one of 'clip', 'raise', or 'wrap' (got 'bogus')";
            const string empty = "clipmode must be one of 'clip', 'raise', or 'wrap' (got '')";
            const string misspelled = "Use one of 'clip', 'raise', or 'wrap' for clip mode";

            Refuses(() => np.put(np.arange(4), np.array(new[] { 0 }), np.array(new[] { 7 }), "bogus"), generic);
            Refuses(() => np.put(np.arange(4), np.array(new[] { 0 }), np.array(new[] { 7 }), "Raise"), misspelled);
            Refuses(() => np.put(np.arange(4), np.array(new[] { 0 }), np.array(new[] { 7 }), ""), empty);
            Refuses(() => np.take(np.arange(4), np.array(new[] { 0 }), mode: "bogus"), generic);
            Refuses(() => np.take(np.arange(4), np.array(new[] { 0 }), mode: "clipped"), misspelled);
            Refuses(() => np.take(np.arange(4), np.array(new[] { 0 }), mode: ""), empty);
            // The mode wins over an out-of-range axis (parsed with the arguments).
            Refuses(() => np.take(np.arange(4), np.array(new[] { 0 }), axis: 1, mode: "bogus"), generic);
            Refuses(() => np.choose(np.array(new[] { 0 }), new[] { np.array(new[] { 1 }) }, mode: "bogus"), generic);

            var mi = new[] { np.array(new[] { 0 }), np.array(new[] { 1 }) };
            Refuses(() => np.ravel_multi_index(mi, new[] { 2, 2 }, mode: "bogus"), generic);
            Refuses(() => np.ravel_multi_index(mi, new[] { 2, 2 }, mode: "Wrap"), misspelled);
            Refuses(() => np.ravel_multi_index(mi, new[] { 2, 2 }, mode: ""), empty);
            Refuses(() => np.ravel_multi_index(mi, new[] { 2, 2 }, modes: new[] { "raise" }),
                "list of clipmodes has wrong length (1 instead of 2)");
            Refuses(() => np.ravel_multi_index(mi, new[] { 2, 2 }, modes: new[] { "raise", "bogus" }), generic);
            // The modes are converted before the coordinates are counted (arr_ravel_multi_index's order).
            Refuses(() => np.ravel_multi_index(new[] { np.array(new[] { 0 }) }, new[] { 2, 2 }, mode: "bogus"), generic);
            np.ravel_multi_index(mi, new[] { 2, 2 }, modes: null).ToArray<long>().Should().Equal(1L);
        }
    }
}
