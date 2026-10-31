using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Sorting;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Strings phase 2, P2.2 (docs/plans/strings-phase2.md §3.2 D19): <c>np.sort</c>, <c>np.argsort</c>,
    ///     <c>ndarray.sort</c>, <c>np.partition</c> and <c>np.argpartition</c> over <c>S</c> / <c>U</c> text, and the
    ///     <c>kind=</c> / <c>order=</c> / <c>stable=</c> arguments of the sorts for every dtype. NumPy's default argsort of
    ///     text is its introsort (<c>string_aquicksort_</c>), which is not stable: equal strings come out in the order its
    ///     comparisons leave them, so every tie-order expectation below is a NumPy 2.4.2 literal (the 40-element tie case,
    ///     an input McIlroy's adversary built to drive the introsort into its heapsort fallback, every key packing).
    ///     The routes (the direct key-record port, the stable radix order with its distinct-strings shortcut, the rank
    ///     port, the count-rank sort of short lines, the managed kernels) are held to one another with route counters
    ///     proving which ran; the kernels to their managed twins; the reads to the operands' bytes with fenced pages.
    /// </summary>
    [TestClass]
    public unsafe class StringSortTests
    {
        // ------------------------------------------------------------------ helpers

        /// <summary>A <c>U</c> array of the given strings (as wide as the longest).</summary>
        /// <param name="values">The strings.</param>
        /// <returns>The array.</returns>
        private static NDArray U(params string[] values) => np.array(values);

        /// <summary>
        ///     A <c>U</c> array built from raw code units (any 32-bit value, the units above U+10FFFF included), each row
        ///     zero-padded to <paramref name="width"/> units.
        /// </summary>
        /// <param name="rows">The units of each element.</param>
        /// <param name="width">The element width in units.</param>
        /// <returns>The array (a copy: writeable, owning).</returns>
        private static NDArray Units(long[][] rows, int width)
        {
            var raw = new byte[rows.Length * width * 4];
            for (int i = 0; i < rows.Length; i++)
                for (int t = 0; t < rows[i].Length; t++)
                    BitConverter.TryWriteBytes(raw.AsSpan((i * width + t) * 4, 4), (uint)rows[i][t]);
            return np.frombuffer(raw, np.dtype($"<U{width}")).copy();
        }

        /// <summary>The int64 elements of an index array, in C order.</summary>
        /// <param name="a">The array.</param>
        /// <returns>The values.</returns>
        private static long[] Ints(NDArray a) => a.ToArray<long>();

        /// <summary>The 40 one-character strings of NumPy's probe: <c>np.random.default_rng(1).choice(['x','y','z'], 40)</c>.</summary>
        private static NDArray Ties40() => U("yyzzxxzzxxzyxzxyyyxxzzzyzxyzxxxyzxyyzxyx".Select(c => c.ToString()).ToArray());

        /// <summary>NumPy 2.4.2's default argsort of <see cref="Ties40"/> (its introsort's tie order).</summary>
        private static readonly long[] Ties40Default =
            { 19, 37, 33, 30, 29, 28, 25, 18, 14, 12, 39, 9, 8, 4, 5, 11, 1, 35, 34, 31, 26, 23, 0, 17, 15, 16, 38, 13, 36, 2, 32, 21, 3, 7, 27, 20, 10, 24, 22, 6 };

        /// <summary>NumPy 2.4.2's stable argsort of <see cref="Ties40"/> (ties in input order).</summary>
        private static readonly long[] Ties40Stable =
            { 4, 5, 8, 9, 12, 14, 18, 19, 25, 28, 29, 30, 33, 37, 39, 0, 1, 11, 15, 16, 17, 23, 26, 31, 34, 35, 38, 2, 3, 6, 7, 10, 13, 20, 21, 22, 24, 27, 32, 36 };

        /// <summary>
        ///     McIlroy's "killer adversary" against NumPy's string introsort on 64 elements, its values halved (so the
        ///     fallback orders ties): NumPy's depth budget runs out and <c>string_aheapsort_</c> sorts one range.
        /// </summary>
        private static readonly int[] Adversary =
        {
            0, 28, 1, 28, 2, 27, 3, 27, 4, 26, 5, 26, 6, 25, 7, 25, 8, 24, 9, 24, 10, 23, 11, 23, 12, 13, 31, 30, 31, 30, 29, 0,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16, 15, 15, 14, 14, 29, 13,
        };

        /// <summary>
        ///     The adversary before halving (McIlroy's values, all distinct; halving each gives <see cref="Adversary"/>):
        ///     NumPy's introsort reaches its heapsort fallback once on it too.
        /// </summary>
        private static readonly int[] AdversaryDistinct =
        {
            0, 57, 2, 56, 4, 55, 6, 54, 8, 53, 10, 52, 12, 51, 14, 50, 16, 49, 18, 48, 20, 47, 22, 46, 24, 27, 63, 61, 62, 60, 59, 1,
            3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 58, 26,
        };

        /// <summary>NumPy 2.4.2's default argsort of <see cref="Adversary"/> written as 3-digit strings (U3 and S3 alike).</summary>
        private static readonly long[] AdversaryDefault =
        {
            0, 31, 2, 32, 4, 33, 6, 34, 8, 35, 10, 36, 12, 37, 14, 38, 16, 39, 18, 40, 20, 41, 22, 42, 24, 43, 25, 63, 60, 61, 59,
            58, 56, 57, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 23, 21, 17, 19, 15, 13, 11, 9, 7, 5, 1, 3, 62, 30, 29, 27,
            28, 26,
        };

        /// <summary>NumPy 2.4.2's stable argsort of <see cref="Adversary"/>.</summary>
        private static readonly long[] AdversaryStable =
        {
            0, 31, 2, 32, 4, 33, 6, 34, 8, 35, 10, 36, 12, 37, 14, 38, 16, 39, 18, 40, 20, 41, 22, 42, 24, 43, 25, 63, 60, 61, 58,
            59, 56, 57, 54, 55, 52, 53, 50, 51, 48, 49, 46, 47, 44, 45, 21, 23, 17, 19, 13, 15, 9, 11, 5, 7, 1, 3, 30, 62, 27, 29,
            26, 28,
        };

        /// <summary>
        ///     Runs <paramref name="body"/> with the test hooks set, restoring them after (they are process-wide switches).
        /// </summary>
        /// <param name="shortcutOff">Sets <see cref="TextSort.ShortcutDisabled"/>.</param>
        /// <param name="directOff">Sets <see cref="TextSort.DirectPortDisabled"/>.</param>
        /// <param name="managed">Turns the IL generator off, so every kernel is its managed twin.</param>
        /// <param name="body">The calls to make.</param>
        /// <param name="packedOff">Sets <see cref="TextSort.PackedWindowDisabled"/>.</param>
        /// <param name="tailOff">Sets <see cref="TextSort.TailPortDisabled"/>.</param>
        private static void WithRoutes(bool shortcutOff, bool directOff, bool managed, Action body, bool packedOff = false,
                                       bool tailOff = false)
        {
            bool shortcut = TextSort.ShortcutDisabled, direct = TextSort.DirectPortDisabled, il = DirectILKernelGenerator.Enabled;
            bool packed = TextSort.PackedWindowDisabled, tail = TextSort.TailPortDisabled;
            try
            {
                TextSort.ShortcutDisabled = shortcutOff;
                TextSort.DirectPortDisabled = directOff;
                TextSort.PackedWindowDisabled = packedOff;
                TextSort.TailPortDisabled = tailOff;
                DirectILKernelGenerator.Enabled = !managed;
                body();
            }
            finally
            {
                TextSort.ShortcutDisabled = shortcut;
                TextSort.DirectPortDisabled = direct;
                TextSort.PackedWindowDisabled = packed;
                TextSort.TailPortDisabled = tail;
                DirectILKernelGenerator.Enabled = il;
            }
        }

        // ------------------------------------------------------------------ NumPy's tie order

        /// <summary>
        ///     The default, quicksort and heapsort kinds (and <c>stable=False</c>, and argpartition's generic introsort) give
        ///     NumPy's introsort permutation; the stable kinds the stable one; the values are the sorted strings.
        /// </summary>
        [TestMethod]
        public void Argsort_Kinds_GiveNumPysTieOrders()
        {
            var b = Ties40();
            Ints(np.argsort(b)).Should().Equal(Ties40Default);
            Ints(np.argsort(b, kind: "quicksort")).Should().Equal(Ties40Default);
            Ints(np.argsort(b, kind: "heapsort")).Should().Equal(Ties40Default);
            Ints(np.argsort(b, stable: false)).Should().Equal(Ties40Default);
            Ints(np.argpartition(b, 5)).Should().Equal(Ties40Default);
            Ints(b.argsort(kind: "h")).Should().Equal(Ties40Default);
            Ints(np.argsort(b, kind: "stable")).Should().Equal(Ties40Stable);
            Ints(np.argsort(b, kind: "mergesort")).Should().Equal(Ties40Stable);
            Ints(np.argsort(b, stable: true)).Should().Equal(Ties40Stable);
            var sorted = NDArray.AsStringArray(np.sort(b));
            sorted.Should().Equal(Enumerable.Repeat("x", 15).Concat(Enumerable.Repeat("y", 12)).Concat(Enumerable.Repeat("z", 13)));
            NDArray.AsStringArray(np.partition(b, 5)).Should().Equal(sorted);
        }

        /// <summary>
        ///     An input built to exhaust NumPy's depth budget: the introsort hands a range to its heapsort fallback, whose tie
        ///     order is NumPy's too — through the direct key-record port and through the rank port (direct port off), for
        ///     <c>U</c> and <c>S</c> alike. The fallback counter proves the heapsort ran.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // flips the process-wide text-sort route switches and IL kernel generation (TextSort, DirectILKernelGenerator.Enabled)
        public void Argsort_HeapsortFallback_IsNumPys()
        {
            var u = U(Adversary.Select(v => v.ToString("D3")).ToArray());
            var s = u.astype(np.dtype("S3"));
            foreach (bool directOff in new[] { false, true })
            {
                WithRoutes(false, directOff, false, () =>
                {
                    long before = TextSort.HeapFallbackRuns;
                    Ints(np.argsort(u)).Should().Equal(AdversaryDefault);
                    Ints(np.argsort(s)).Should().Equal(AdversaryDefault);
                    (TextSort.HeapFallbackRuns - before).Should().Be(2, "both sorts take NumPy's heapsort fallback once");
                    Ints(np.argsort(u, kind: "stable")).Should().Equal(AdversaryStable);
                });
            }
        }

        /// <summary>
        ///     Lines of at most 16 elements: NumPy's introsort is a plain insertion sort there, so every kind gives the stable
        ///     order — which the count-rank route computes.
        /// </summary>
        [TestMethod]
        public void ShortLines_AreStableForEveryKind()
        {
            var b = Ties40()["0:16"];
            var stable = Ints(np.argsort(b, kind: "stable"));
            Ints(np.argsort(b)).Should().Equal(stable);
            // np.argsort(b[:16]) of 'yyzzxxzzxxzyxzxy': the x's, then the y's, then the z's, each in input order.
            stable.Should().Equal(4, 5, 8, 9, 12, 14, 0, 1, 11, 15, 2, 3, 6, 7, 10, 13);
            // 17 elements partition once: no longer the stable order (np.argsort(b[:17]) of 'yyzzxxzzxxzyxzxyy').
            Ints(np.argsort(Ties40()["0:17"])).Should().Equal(8, 14, 12, 4, 5, 9, 11, 15, 0, 1, 16, 6, 10, 3, 2, 13, 7);
        }

        // ------------------------------------------------------------------ the keys

        /// <summary>
        ///     Every key packing of <c>U</c> (8 bits a unit for Latin-1, 16 for the BMP, 21 for any code point, 32 for the
        ///     units above U+10FFFF a view can hold), and a non-ASCII unit after eight ASCII ones (the tie pass re-chooses the
        ///     packing for the next units), against NumPy's permutations.
        /// </summary>
        [TestMethod]
        public void Argsort_EveryKeyPacking_IsNumPys()
        {
            var latin1 = Units(new[]
            {
                new long[] { 255, 233, 255 }, new long[] { 255, 233, 255 }, new long[] { 255, 65, 65 }, new long[] { 65, 65, 255 },
                new long[] { 255, 65, 233 }, new long[] { 255, 65, 255 }, new long[] { 65, 233, 255 }, new long[] { 65, 233, 65 },
                new long[] { 255, 65, 255 }, new long[] { 233, 233, 233 }, new long[] { 233, 233, 233 }, new long[] { 255, 255, 255 },
                new long[] { 255, 233, 233 }, new long[] { 255, 233, 65 }, new long[] { 255, 65, 255 }, new long[] { 233, 65, 65 },
                new long[] { 233, 65, 65 }, new long[] { 233, 255, 233 }, new long[] { 255, 255, 255 }, new long[] { 233, 233, 233 },
            }, 3);
            Ints(np.argsort(latin1)).Should().Equal(3, 7, 6, 16, 15, 9, 10, 19, 17, 2, 4, 8, 14, 5, 13, 12, 1, 0, 18, 11);
            Ints(np.argsort(latin1, kind: "stable")).Should().Equal(3, 7, 6, 15, 16, 9, 10, 19, 17, 2, 4, 5, 8, 14, 13, 12, 0, 1, 11, 18);

            var bmp = Units(new[]
            {
                new long[] { 65, 256, 256 }, new long[] { 65, 20013, 65 }, new long[] { 65, 65, 20013 }, new long[] { 20013, 20013, 65 },
                new long[] { 20013, 256, 256 }, new long[] { 65, 256, 20013 }, new long[] { 256, 65, 256 }, new long[] { 65, 20013, 20013 },
                new long[] { 65, 256, 20013 }, new long[] { 20013, 20013, 256 }, new long[] { 65, 20013, 256 }, new long[] { 65, 65, 256 },
                new long[] { 20013, 256, 256 }, new long[] { 20013, 20013, 256 }, new long[] { 256, 256, 65 }, new long[] { 65, 256, 256 },
                new long[] { 256, 65, 65 }, new long[] { 65, 256, 20013 }, new long[] { 256, 256, 256 }, new long[] { 20013, 256, 256 },
            }, 3);
            Ints(np.argsort(bmp)).Should().Equal(11, 2, 0, 15, 17, 5, 8, 1, 10, 7, 16, 6, 14, 18, 19, 12, 4, 3, 13, 9);
            Ints(np.argsort(bmp, kind: "stable")).Should().Equal(11, 2, 0, 15, 5, 8, 17, 1, 10, 7, 16, 6, 14, 18, 4, 12, 19, 3, 9, 13);

            var astral = Units(new[]
            {
                new long[] { 128512, 128512, 128512 }, new long[] { 128512, 128512, 1114111 }, new long[] { 1114111, 65, 128512 },
                new long[] { 128512, 128512, 65 }, new long[] { 65, 128512, 1114111 }, new long[] { 65, 128512, 1114111 },
                new long[] { 1114111, 65, 65 }, new long[] { 1114111, 65, 65 }, new long[] { 128512, 1114111, 65 },
                new long[] { 128512, 128512, 65 }, new long[] { 1114111, 1114111, 128512 }, new long[] { 1114111, 128512, 1114111 },
                new long[] { 1114111, 128512, 65 }, new long[] { 65, 128512, 65 }, new long[] { 1114111, 1114111, 65 },
                new long[] { 128512, 65, 65 }, new long[] { 128512, 1114111, 1114111 }, new long[] { 128512, 128512, 128512 },
                new long[] { 65, 128512, 1114111 }, new long[] { 128512, 65, 65 },
            }, 3);
            Ints(np.argsort(astral)).Should().Equal(13, 18, 4, 5, 19, 15, 9, 3, 17, 0, 1, 8, 16, 7, 6, 2, 12, 11, 14, 10);
            Ints(np.argsort(astral, kind: "stable")).Should().Equal(13, 4, 5, 18, 15, 19, 3, 9, 0, 17, 1, 8, 16, 6, 7, 2, 12, 11, 14, 10);

            var over = Units(new[]
            {
                new long[] { 4294967295, 65, 4294967295 }, new long[] { 4294967295, 1114112, 1114112 }, new long[] { 4294967295, 1114112, 1114112 },
                new long[] { 65, 1114112, 4294967295 }, new long[] { 65, 65, 4294967295 }, new long[] { 1114112, 4294967295, 65 },
                new long[] { 4294967295, 65, 1114112 }, new long[] { 4294967295, 65, 1114112 }, new long[] { 4294967295, 1114112, 65 },
                new long[] { 1114112, 1114112, 4294967295 }, new long[] { 4294967295, 1114112, 65 }, new long[] { 1114112, 65, 4294967295 },
                new long[] { 4294967295, 1114112, 4294967295 }, new long[] { 1114112, 1114112, 4294967295 }, new long[] { 4294967295, 4294967295, 4294967295 },
                new long[] { 65, 1114112, 4294967295 }, new long[] { 65, 65, 1114112 }, new long[] { 4294967295, 4294967295, 4294967295 },
                new long[] { 1114112, 65, 4294967295 }, new long[] { 1114112, 4294967295, 4294967295 },
            }, 3);
            Ints(np.argsort(over)).Should().Equal(16, 4, 15, 3, 11, 18, 9, 13, 5, 19, 7, 6, 0, 10, 8, 2, 1, 12, 14, 17);
            Ints(np.argsort(over, kind: "stable")).Should().Equal(16, 4, 3, 15, 11, 18, 9, 13, 5, 19, 6, 7, 0, 8, 10, 1, 2, 12, 14, 17);

            long[] ninth = { 20013, 98, 20013, 20013, 98, 20013, 20013, 20013, 98, 20013, 98, 98, 98, 98, 20013, 98, 98, 98, 98, 98 };
            var late = Units(ninth.Select(u => Enumerable.Repeat(97L, 8).Append(u).ToArray()).ToArray(), 9);
            Ints(np.argsort(late)).Should().Equal(19, 1, 17, 16, 4, 15, 13, 8, 18, 10, 11, 12, 14, 0, 6, 5, 3, 2, 7, 9);
        }

        /// <summary>
        ///     NULs compare as units: an embedded NUL orders below every other unit and a trailing one pads like the width's
        ///     padding (<c>'a'</c> and <c>'a\0'</c> are one string); <c>S</c> bytes compare unsigned (0x80 above 0x7f).
        /// </summary>
        [TestMethod]
        public void Argsort_NulsAndHighBytes_AreNumPys()
        {
            var nul = Units(new[]
            {
                new long[] { 0, 97, 97, 97 }, new long[] { 97, 0, 0, 0 }, new long[] { 0, 97, 0, 0 }, new long[] { 0, 97, 0, 97 },
                new long[] { 97, 97, 97, 97 }, new long[] { 97, 97, 97, 97 }, new long[] { 97, 0, 97, 97 }, new long[] { 0, 0, 97, 97 },
                new long[] { 0, 0, 0, 0 }, new long[] { 0, 97, 97, 97 }, new long[] { 97, 0, 97, 97 }, new long[] { 97, 97, 0, 97 },
                new long[] { 0, 97, 97, 97 }, new long[] { 0, 0, 97, 0 }, new long[] { 97, 0, 0, 0 }, new long[] { 0, 0, 0, 97 },
                new long[] { 97, 97, 97, 0 }, new long[] { 97, 97, 97, 0 }, new long[] { 0, 97, 0, 97 }, new long[] { 0, 0, 0, 97 },
            }, 4);
            var order = Ints(np.argsort(nul));
            order.Should().Equal(8, 19, 15, 13, 7, 2, 3, 18, 12, 0, 9, 14, 1, 10, 6, 11, 16, 17, 5, 4);
            // The values sort is the gather of any correct order (equal strings are equal bytes).
            np.sort(nul).tobytes().Should().Equal(np.take(nul, np.array(order)).tobytes());

            var s = np.frombuffer(new byte[] { 0x80, 0, 0x7F, 0, 0x61, 0, 0, 0, 0x61, 0x80 }, np.dtype("S2")).copy();
            Ints(np.argsort(s)).Should().Equal(3, 2, 4, 1, 0);
            np.sort(s).tobytes().Should().Equal(new byte[] { 0, 0, 0x61, 0, 0x61, 0x80, 0x7F, 0, 0x80, 0 });
        }

        // ------------------------------------------------------------------ arguments

        /// <summary>
        ///     NumPy's argument rules, in its order (<c>array_sort</c> / <c>array_argsort</c>): <c>kind</c> parsed first (the
        ///     first letter, any case; the refusal quotes the argument as Python's repr), then <c>kind</c> with
        ///     <c>stable</c>, then <c>order</c> — all before the axis — for text and numbers alike.
        /// </summary>
        [TestMethod]
        public void Arguments_FollowNumPysRulesAndOrder()
        {
            var row = U("b", "a", "c");
            foreach (var kind in new[] { "q", "Quick", "h", "HEAP", "m", "Merge", "s", "STABLE" })
                Ints(np.argsort(row, kind: kind)).Should().Equal(1, 0, 2);

            Action timsort = () => np.argsort(row, kind: "timsort");
            timsort.Should().Throw<ValueError>().WithMessage("sort kind must be one of 'quick', 'heap', or 'stable' (got 'timsort')");
            Action empty = () => np.sort(row, kind: "");
            empty.Should().Throw<ValueError>().WithMessage("sort kind must be one of 'quick', 'heap', or 'stable' (got '')");
            Action quoted = () => np.sort(row, kind: "it's");
            quoted.Should().Throw<ValueError>().WithMessage("sort kind must be one of 'quick', 'heap', or 'stable' (got \"it's\")");
            Action numbers = () => np.sort(np.arange(3), kind: "x");
            numbers.Should().Throw<ValueError>().WithMessage("sort kind must be one of 'quick', 'heap', or 'stable' (got 'x')");

            Action both = () => np.argsort(row, kind: "stable", stable: false);
            both.Should().Throw<ValueError>().WithMessage("`kind` and keyword parameters can't be provided at the same time. Use only one of them.");
            Action order = () => np.sort(row, order: "x");
            order.Should().Throw<ValueError>().WithMessage("Cannot specify order when the array has no fields.");

            // kind and order come before the axis: np.argsort(row, axis=9, kind='x') / (axis=9, order='x').
            Action kindFirst = () => np.argsort(row, 9, kind: "x");
            kindFirst.Should().Throw<ValueError>().WithMessage("sort kind must be one of*");
            Action orderFirst = () => np.argsort(row, 9, order: "x");
            orderFirst.Should().Throw<ValueError>().WithMessage("Cannot specify order*");
            Action axis = () => np.argsort(row, 9);
            axis.Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 1*");
        }

        /// <summary>
        ///     <c>StringDType</c>'s sort loop accepts the default and stable kinds only: the heapsort kind is NumPy's
        ///     <c>RuntimeError</c> — after the axis (and writeability) checks for a sort, before the axis check for an argsort,
        ///     and even for one element.
        /// </summary>
        [TestMethod]
        public void StringDType_RefusesTheHeapsortKind()
        {
            var t = np.array(new[] { "b", "a" }).astype(np.dtype("T"));
            Action sort = () => np.sort(t, kind: "heapsort");
            sort.Should().Throw<RuntimeError>().WithMessage("unsupported sort kind");
            Action one = () => np.sort(t["0:1"], kind: "heapsort");
            one.Should().Throw<RuntimeError>().WithMessage("unsupported sort kind");
            Action sortAxis = () => np.sort(t, 5, kind: "heapsort");
            sortAxis.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 1*");
            Action argsortAxis = () => np.argsort(t, 5, kind: "heapsort");
            argsortAxis.Should().Throw<RuntimeError>().WithMessage("unsupported sort kind");
            var readOnly = np.broadcast_to(t, new Shape(2, 2));
            Action ro = () => readOnly.sort(kind: "heapsort");
            ro.Should().Throw<ValueError>().WithMessage("sort array is read-only");
        }

        // ------------------------------------------------------------------ layouts

        /// <summary>
        ///     Every axis and layout against NumPy: axis 0 / 1 / None, a transpose, reversed columns, a broadcast operand; the
        ///     sorted copy keeps the input's memory order (an F input stays F) and the indices are C-contiguous.
        /// </summary>
        [TestMethod]
        public void Layouts_AreNumPys()
        {
            var m = np.array(new[,] { { "pear", "fig", "apple", "fig" }, { "kiwi", "date", "date", "apple" }, { "fig", "pear", "kiwi", "apple" } });
            Ints(np.argsort(m, 0)).Should().Equal(2, 1, 0, 1, 1, 0, 1, 2, 0, 2, 2, 0);
            Ints(np.argsort(m, 1)).Should().Equal(2, 1, 3, 0, 3, 1, 2, 0, 3, 0, 2, 1);
            Ints(np.argsort(m, null)).Should().Equal(2, 7, 11, 5, 6, 1, 3, 8, 4, 10, 0, 9);
            NDArray.AsStringArray(np.sort(m, 0)).Should().Equal("fig", "date", "apple", "apple", "kiwi", "fig", "date", "apple", "pear", "pear", "kiwi", "fig");
            NDArray.AsStringArray(np.sort(m, null)).Should().Equal("apple", "apple", "apple", "date", "date", "fig", "fig", "fig", "kiwi", "kiwi", "pear", "pear");
            Ints(np.argsort(m.T, 1)).Should().Equal(2, 1, 0, 1, 0, 2, 0, 1, 2, 1, 2, 0);
            Ints(np.argsort(m[":, ::-1"], 1)).Should().Equal(1, 0, 2, 3, 0, 1, 2, 3, 0, 3, 1, 2);

            var f = np.asfortranarray(m);
            var sortedF = np.sort(f, 0);
            sortedF.Shape.IsFContiguous.Should().BeTrue();
            sortedF.Shape.IsContiguous.Should().BeFalse();
            NDArray.AsStringArray(sortedF).Should().Equal(NDArray.AsStringArray(np.sort(m, 0)));
            np.argsort(f, 0).Shape.IsContiguous.Should().BeTrue();

            var broadcast = np.broadcast_to(U("b", "a", "c"), new Shape(2, 3));
            Ints(np.argsort(broadcast, 0)).Should().Equal(0, 0, 0, 1, 1, 1);
            Ints(np.argsort(broadcast, 1)).Should().Equal(1, 0, 2, 1, 0, 2);
        }

        /// <summary>
        ///     <c>ndarray.sort</c> sorts in place, through a view into its base; a read-only operand refuses with NumPy's text;
        ///     <c>ndarray.partition</c> sorts too (a string has no partition function).
        /// </summary>
        [TestMethod]
        public void InPlace_SortsThroughViews()
        {
            var m = np.array(new[,] { { "pear", "fig", "apple", "fig" }, { "kiwi", "date", "date", "apple" }, { "fig", "pear", "kiwi", "apple" } });
            m[":, 1:3"].sort(1);
            NDArray.AsStringArray(m).Should().Equal("pear", "apple", "fig", "fig", "kiwi", "date", "date", "apple", "fig", "kiwi", "pear", "apple");
            Action ro = () => np.broadcast_to(U("b", "a", "c"), new Shape(2, 3)).sort();
            ro.Should().Throw<ValueError>().WithMessage("sort array is read-only");
            var p = U("c", "a", "b");
            p.partition(1);
            NDArray.AsStringArray(p).Should().Equal("a", "b", "c");
            Ints(np.argpartition(np.array(new[,] { { "pear", "fig", "apple", "fig" } }), 2, 1)).Should().Equal(2, 1, 3, 0);
        }

        // ------------------------------------------------------------------ routes

        /// <summary>
        ///     The routes give one permutation: the default one, the stable radix order without the shortcut, without the
        ///     direct ports (the rank port), without the window-packed records, without the tail-compare port, and with every
        ///     kernel managed — over exact keys with duplicates (the direct port over window-packed records), distinct
        ///     strings (shortcut), long strings with a common prefix and duplicates (tie pass + rank port), mostly distinct
        ///     strings longer than their keys with a few duplicates (the tail-compare port), short rows (count-rank, one block
        ///     call) and strided short rows (keyed one row at a time). The counters prove each route ran.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // flips the process-wide text-sort route switches and IL kernel generation (TextSort, DirectILKernelGenerator.Enabled)
        public void Routes_GiveOnePermutation()
        {
            var rng = new System.Random(11);
            string Word(int len) => new string(Enumerable.Range(0, len).Select(_ => (char)('a' + rng.Next(3))).ToArray());
            var categorical = U(Enumerable.Range(0, 600).Select(_ => Word(4)).ToArray());
            var distinct = U(Enumerable.Range(0, 5000).Select(i => "w" + i.ToString("D6")).OrderBy(_ => rng.Next()).ToArray());
            var prefixed = U(Enumerable.Range(0, 900).Select(_ => "https://example/" + Word(3)).ToArray());
            var rows = U(Enumerable.Range(0, 400).Select(_ => Word(2)).ToArray()).reshape(40, 10);
            var cube = U(Enumerable.Range(0, 240).Select(_ => Word(2)).ToArray()).reshape(4, 6, 10);
            // Twelve-letter strings over 26 letters with 6 duplicated: their 8-unit keys rarely repeat, and do not cover them.
            var longWords = Enumerable.Range(0, 900)
                .Select(_ => new string(Enumerable.Range(0, 12).Select(_ => (char)('a' + rng.Next(26))).ToArray())).ToList();
            for (int d = 0; d < 6; d++)
                longWords[rng.Next(900)] = longWords[rng.Next(900)];
            // Two duplicates at positions the key sample reads (every 7th of 900): the repeats that send the line to the
            // tail-compare port (without one, distinct-looking strings take the radix and the shortcut).
            longWords[14] = longWords[7];
            longWords[28] = longWords[21];
            var tail = U(longWords.ToArray());
            var stridedRows = rows.T;
            var inputs = new[] { categorical, distinct, prefixed, rows, cube, tail, stridedRows };

            long[][] Run() => inputs.SelectMany(a => new[] { Ints(np.argsort(a)), Ints(np.argsort(a, kind: "stable")) }).ToArray();

            long direct = TextSort.DirectPortRuns, shortcut = TextSort.ShortcutRuns, port = TextSort.PortRuns, blocks = TextSort.BlockCalls;
            long packed = TextSort.PackedDirectRuns, tailRuns = TextSort.TailPortRuns;
            var reference = Run();
            (TextSort.PackedDirectRuns - packed).Should().BeGreaterThan(0, "the categorical line takes the window-packed direct port");
            (TextSort.TailPortRuns - tailRuns).Should().BeGreaterThan(0, "the long-word line takes the tail-compare port");
            (TextSort.DirectPortRuns - direct).Should().BeGreaterThan(TextSort.TailPortRuns - tailRuns, "the categorical line takes the direct port");
            (TextSort.ShortcutRuns - shortcut).Should().BeGreaterThan(0, "the distinct line takes the shortcut");
            (TextSort.PortRuns - port).Should().BeGreaterThan(TextSort.DirectPortRuns - direct, "the prefixed line takes the rank port");
            (TextSort.BlockCalls - blocks).Should().Be(2 * inputs.Length, "every call hands its lines to the kernel in one block");

            foreach (var (shortcutOff, directOff, managed, packedOff, tailOff) in new[]
                     {
                         (true, false, false, false, false), (false, true, false, false, false), (true, true, false, false, false),
                         (false, false, true, false, false), (false, false, false, true, false), (false, false, false, false, true),
                     })
                WithRoutes(shortcutOff, directOff, managed, () => Run().Should().BeEquivalentTo(reference, o => o.WithStrictOrdering()),
                           packedOff, tailOff);
        }

        /// <summary>
        ///     Short rows (at most 16 elements, counted ranks) whose 8-unit keys tie are ordered by the units after the keys,
        ///     not by position: rows laid end to end, an F-ordered copy (strided rows, keyed one row at a time), the columns
        ///     (axis 0) and the same rows as bytes. NumPy 2.4.2: <c>np.argsort(rows, axis=1)</c> is
        ///     <c>[[1, 2, 0], [2, 1, 0], [1, 0, 2]]</c>.
        /// </summary>
        [TestMethod]
        public void ShortRows_TiedKeys_AreOrderedByTheirTails()
        {
            var rows = U("abcdefghZ", "abcdefghA", "abcdefghM", "zzzzzzzzb", "zzzzzzzza", "a", "qqqqqqqqq", "qqqqqqqqa", "qqqqqqqqq")
                .reshape(3, 3);
            long[] expectedRows = { 1, 2, 0, 2, 1, 0, 1, 0, 2 };
            Ints(np.argsort(rows, 1)).Should().Equal(expectedRows);
            Ints(np.argsort(np.asfortranarray(rows), 1)).Should().Equal(expectedRows, "strided rows are keyed one row at a time");
            Ints(np.argsort(rows.astype(np.dtype("S9")), 1)).Should().Equal(expectedRows, "bytes key 8 of their 9 bytes");
            Ints(np.argsort(rows, 0)).Should().Equal(0L, 0L, 1L, 2L, 2L, 0L, 1L, 1L, 2L);
        }

        /// <summary>
        ///     A line with more elements than the window of its varying key bits holds values has duplicates for certain, so
        ///     it takes the direct port without a sample repeat: 40,000 two-unit strings whose units span 128 Latin-1 values
        ///     (a 15-bit window), arranged so that the evenly spaced key sample holds no repeat. The rank route (direct port
        ///     off) gives the same permutation.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // flips the process-wide text-sort route switches and IL kernel generation (TextSort, DirectILKernelGenerator.Enabled)
        public void CertainDuplicates_TakeTheDirectPort()
        {
            var values = Enumerable.Range(0, 40000)
                .Select(i => i % 16384)
                .Select(v => new string(new[] { (char)(0x80 + (v >> 7)), (char)(0x80 + (v & 127)) }))
                .ToArray();
            var a = U(values);
            long direct = TextSort.DirectPortRuns, packed = TextSort.PackedDirectRuns;
            var viaDirect = Ints(np.argsort(a));
            (TextSort.DirectPortRuns - direct).Should().Be(1, "duplicates are certain, so the line takes the direct port");
            (TextSort.PackedDirectRuns - packed).Should().Be(1, "the 15-bit window packs");
            WithRoutes(false, true, false, () => Ints(np.argsort(a)).Should().Equal(viaDirect));
        }

        /// <summary>
        ///     The MSD key sort (<see cref="RadixSort.ArgSortU64Msd"/>) leaves the one stable order the LSD passes leave —
        ///     indices and sorted keys alike — over the key shapes that stress an MSD radix: random 64-bit keys, eight
        ///     lowercase letters, eight digits, a shared four-byte prefix, short NUL-padded strings, 50 distinct keys repeated
        ///     (buckets of equal keys), every key equal, keys equal but for one, sorted and reversed input, and one-bit
        ///     differences at every byte; for lines of 2 to 5,000 keys, around the insertion cutoff and the 4,096-key limit.
        /// </summary>
        [TestMethod]
        public void MsdKeySort_IsTheStableLsdOrder()
        {
            var rng = new System.Random(20261019);
            string[] shapes = { "rand64", "lower8", "digits", "prefix", "short", "few", "equal", "allButOne", "sorted", "reversed", "onebit" };
            int[] sizes = { 2, 3, 17, 31, 32, 33, 64, 120, 121, 500, 1000, 4096, 5000 };
            long* hist = stackalloc long[8 * 256];
            foreach (string shape in shapes)
            {
                foreach (int n in sizes)
                {
                    var src = new ulong[n];
                    ulong[] few = Enumerable.Range(0, 50).Select(_ => (ulong)rng.NextInt64()).ToArray();
                    for (int i = 0; i < n; i++)
                    {
                        ulong k = 0;
                        int len = rng.Next(1, 9);
                        for (int j = 0; j < 8; j++)
                        {
                            int c = shape switch
                            {
                                "lower8" => 'a' + rng.Next(26),
                                "digits" => '0' + rng.Next(10),
                                "prefix" => j < 4 ? "http"[j] : 'a' + rng.Next(26),
                                "short" => j < len ? 'a' + rng.Next(26) : 0,
                                _ => rng.Next(256),
                            };
                            k = (k << 8) | (ulong)c;
                        }
                        src[i] = shape switch
                        {
                            "few" => few[rng.Next(50)],
                            "equal" => 0x6162636465666768UL,
                            "allButOne" => i == n / 2 ? 0x6162636465666769UL : 0x6162636465666768UL,
                            "onebit" => 1UL << rng.Next(64),
                            _ => k,
                        };
                    }
                    if (shape == "sorted")
                        Array.Sort(src);
                    if (shape == "reversed")
                    {
                        Array.Sort(src);
                        Array.Reverse(src);
                    }
                    ulong or = 0, and = ulong.MaxValue;
                    foreach (ulong v in src)
                    {
                        or |= v;
                        and &= v;
                    }
                    ulong varying = or ^ and;

                    var k1 = (ulong[])src.Clone();
                    var k2 = (ulong[])src.Clone();
                    var kt = new ulong[n];
                    var x1 = Enumerable.Range(0, n).Select(i => (long)i).ToArray();
                    var x2 = (long[])x1.Clone();
                    var xt = new long[n];
                    long[] lsdIdx, msdIdx;
                    ulong[] lsdKeys, msdKeys;
                    fixed (ulong* pk1 = k1, pk2 = k2, pkt = kt)
                    fixed (long* px1 = x1, px2 = x2, pxt = xt)
                    {
                        long* r1 = RadixSort.ArgSortU64(pk1, pkt, px1, pxt, n, hist, varying, out ulong* s1);
                        lsdIdx = new ReadOnlySpan<long>(r1, n).ToArray();
                        lsdKeys = new ReadOnlySpan<ulong>(s1, n).ToArray();
                        long* r2 = RadixSort.ArgSortU64Msd(pk2, pkt, px2, pxt, n, hist, varying, out ulong* s2);
                        msdIdx = new ReadOnlySpan<long>(r2, n).ToArray();
                        msdKeys = new ReadOnlySpan<ulong>(s2, n).ToArray();
                    }
                    msdIdx.Should().Equal(lsdIdx, $"the stable order of {n} '{shape}' keys");
                    msdKeys.Should().Equal(lsdKeys, $"the sorted keys of {n} '{shape}' keys");
                }
            }
        }

        /// <summary>
        ///     The text sorts take the MSD key sort on short lines of byte-packed keys (<c>S</c>, and <c>U</c> text whose units
        ///     fit a byte), for a line and for the tie runs of its pass, and give the permutations the LSD passes give
        ///     (<see cref="TextSort.MsdDisabled"/>): distinct strings (the shortcut), prefixed strings (tie runs), categorical
        ///     strings, bytes, and non-Latin text (a 16-bit packing, which keeps the LSD passes).
        /// </summary>
        [TestMethod]
        [DoNotParallelize]
        public void MsdRoute_GivesTheLsdPermutation()
        {
            var rng = new System.Random(5);
            string Word(int len, string alphabet) => new string(Enumerable.Range(0, len).Select(_ => alphabet[rng.Next(alphabet.Length)]).ToArray());
            const string lower = "abcdefghijklmnopqrstuvwxyz";
            var distinct = U(Enumerable.Range(0, 1000).Select(_ => Word(30, lower)).ToArray());
            var prefixed = U(Enumerable.Range(0, 3000).Select(_ => "https://example/" + Word(4, lower)).ToArray());
            var categorical = U(Enumerable.Range(0, 2000).Select(_ => Word(3, "xyz")).ToArray());
            var bytes = distinct.astype(np.dtype("S30"));
            var cyrillic = U(Enumerable.Range(0, 1000).Select(_ => Word(12, "абвгдежзий")).ToArray());
            var inputs = new[] { distinct, prefixed, categorical, bytes, cyrillic };

            long[][] Permutations() => inputs.SelectMany(a => new[] { Ints(np.argsort(a)), Ints(np.argsort(a, kind: "stable")) }).ToArray();
            NDArray[] Sorted() => inputs.Select(a => np.sort(a)).ToArray();

            long msd = TextSort.MsdRuns;
            var viaMsd = Permutations();
            var sortedViaMsd = Sorted();
            (TextSort.MsdRuns - msd).Should().BeGreaterThan(0, "the byte-packed lines take the MSD key sort");
            bool saved = TextSort.MsdDisabled;
            try
            {
                TextSort.MsdDisabled = true;
                long before = TextSort.MsdRuns;
                Permutations().Should().BeEquivalentTo(viaMsd, o => o.WithStrictOrdering());
                var sortedViaLsd = Sorted();
                for (int i = 0; i < inputs.Length; i++)
                    np.array_equal(sortedViaLsd[i], sortedViaMsd[i]).Should().BeTrue($"the sorted values of input {i}");
                (TextSort.MsdRuns - before).Should().Be(0, "the hook keeps every key sort on the LSD passes");
            }
            finally
            {
                TextSort.MsdDisabled = saved;
            }
        }

        /// <summary>
        ///     The three heapsort fallbacks (two-ulong records, packed records, prefix-key records with the tail compare)
        ///     leave one permutation on tie-heavy data, where NumPy's sift picks the left child of two equal siblings: every
        ///     element ties on its 8-unit key and the ninth unit takes one of three values.
        /// </summary>
        [TestMethod]
        public void Heapsorts_GiveOnePermutation()
        {
            var rng = new System.Random(31);
            foreach (int n in new[] { 2, 3, 7, 16, 17, 40, 200, 1000 })
            {
                for (int trial = 0; trial < 4; trial++)
                {
                    var ninth = Enumerable.Range(0, n).Select(_ => rng.Next(trial == 0 ? 1 : 3)).ToArray();
                    var a = U(ninth.Select(v => "aaaaaaaa" + (char)('a' + v)).ToArray()).astype(np.dtype("U9"));
                    byte* text = (byte*)a.Storage.Address;
                    var records = new ulong[2 * n];
                    var packed = new ulong[n];
                    var tail = new ulong[2 * n];
                    for (int i = 0; i < n; i++)
                    {
                        records[2 * i] = (ulong)ninth[i];
                        records[2 * i + 1] = (ulong)i;
                        packed[i] = ((ulong)ninth[i] << 32) | (ulong)i;
                        tail[2 * i] = 0x6161616161616161UL;   // "aaaaaaaa": every key ties
                        tail[2 * i + 1] = (ulong)i;
                    }
                    fixed (ulong* r = records, pk = packed, t = tail)
                    {
                        TextSort.HeapArgSortRecords(r, n);
                        TextSort.HeapArgSortPacked(pk, n);
                        TextSort.HeapArgSortTextRecords(t, n, text, 36, 32, 36, 4);
                    }
                    var expected = Enumerable.Range(0, n).Select(i => records[2 * i + 1]).ToArray();
                    packed.Select(v => v & 0xFFFFFFFF).Should().Equal(expected, $"packed heapsort of {n}");
                    Enumerable.Range(0, n).Select(i => tail[2 * i + 1]).Should().Equal(expected, $"tail heapsort of {n}");
                    // A heapsort leaves the keys in order.
                    Enumerable.Range(1, n - 1).All(i => records[2 * (i - 1)] <= records[2 * i]).Should().BeTrue();
                }
            }
        }

        // ------------------------------------------------------------------ primitives

        /// <summary>
        ///     The count-rank sort of short lines is the stable order of its keys, for every length up to 16 and keys with
        ///     many ties (the sign bit set in some, which the vector form's signed compare must not misread).
        /// </summary>
        [TestMethod]
        public void CountRank_IsTheStableOrder()
        {
            var rng = new System.Random(5);
            var keys = new ulong[32];
            var sortedKeys = new ulong[16];
            var sortedIdx = new long[16];
            for (int trial = 0; trial < 4000; trial++)
            {
                int n = 1 + trial % 16;
                for (int i = 0; i < 32; i++)
                    keys[i] = ((ulong)rng.Next(4) << 62) | (ulong)rng.Next(3);
                fixed (ulong* k = keys, sk = sortedKeys)
                fixed (long* sx = sortedIdx)
                    TextSort.CountRankOrder(k, n, sk, sx);
                var expected = Enumerable.Range(0, n).OrderBy(i => keys[i]).ToArray();
                sortedIdx.Take(n).Should().Equal(expected.Select(i => (long)i));
                sortedKeys.Take(n).Should().Equal(expected.Select(i => keys[i]));
            }
        }

        /// <summary>
        ///     The packed introsort port (rank and index in one ulong) asks NumPy's questions with the same answers as the
        ///     two-ulong records port: the same permutation over random ranks with heavy ties and over the adversary's ranks.
        /// </summary>
        [TestMethod]
        public void PackedPort_EqualsTheRecordsPort()
        {
            var rng = new System.Random(9);
            foreach (int n in new[] { 17, 40, 64, 300, 3000 })
            {
                for (int trial = 0; trial < 20; trial++)
                {
                    var ranks = Enumerable.Range(0, n).Select(_ => (ulong)rng.Next(trial % 5 == 0 ? n : 1 + trial)).ToArray();
                    if (n == 64 && trial == 0)
                        ranks = Adversary.Select(v => (ulong)v).ToArray();
                    var packed = new ulong[n];
                    var records = new ulong[2 * n];
                    for (int i = 0; i < n; i++)
                    {
                        packed[i] = (ranks[i] << 32) | (ulong)i;
                        records[2 * i] = ranks[i];
                        records[2 * i + 1] = (ulong)i;
                    }
                    fixed (ulong* p = packed, r = records)
                    {
                        TextSort.IntroArgSortPacked(p, n);
                        TextSort.IntroArgSortRecords(r, n);
                    }
                    packed.Select(v => v & 0xFFFFFFFF).Should().Equal(Enumerable.Range(0, n).Select(i => records[2 * i + 1]));
                }
            }
        }

        /// <summary>
        ///     The tail-compare port (NumPy's introsort over prefix keys that do not cover the strings, comparing the units
        ///     after the keys when two keys tie) gives the permutation the records port gives over the strings' dense ranks:
        ///     random twelve-letter strings with heavy prefix ties and duplicates, and the adversary's values as eight-digit
        ///     keys whose ninth unit is constant (pairs of equal strings) or alternating (pairs equal in the key only) — both
        ///     drive the port into its heapsort fallback, the second through tail compares.
        /// </summary>
        [TestMethod]
        public void TailPort_EqualsTheRecordsPort()
        {
            var rng = new System.Random(21);
            var lines = new List<string[]>();
            foreach (int n in new[] { 17, 40, 300, 2000 })
                for (int trial = 0; trial < 6; trial++)
                {
                    int head = 1 + trial, rest = 3;
                    lines.Add(Enumerable.Range(0, n)
                        .Select(_ => new string(Enumerable.Range(0, 12).Select(c => (char)('a' + rng.Next(c < 8 ? head : rest))).ToArray()))
                        .ToArray());
                }
            lines.Add(Adversary.Select(v => v.ToString("D8") + "z").ToArray());
            lines.Add(AdversaryDistinct.Select(u => (u / 2).ToString("D8") + (u % 2 == 0 ? "a" : "b")).ToArray());

            foreach (var strings in lines)
            {
                int n = strings.Length, width = strings.Max(x => x.Length);
                var a = U(strings).astype(np.dtype("U" + width));
                byte* text = (byte*)a.Storage.Address;
                // The 8-bit packing of the first 8 units (every unit here is ASCII): the key kernel's packing.
                var tailRecords = new ulong[2 * n];
                for (int i = 0; i < n; i++)
                {
                    ulong key = 0;
                    for (int u = 0; u < 8; u++)
                        key |= (ulong)(u < strings[i].Length ? strings[i][u] : 0) << (56 - 8 * u);
                    tailRecords[2 * i] = key;
                    tailRecords[2 * i + 1] = (ulong)i;
                }
                // Dense ranks of the whole strings (ordinal order is code-point order for ASCII).
                var distinctSorted = strings.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToList();
                var rankOf = distinctSorted.Select((x, r) => (x, r)).ToDictionary(t => t.x, t => (ulong)t.r);
                var rankRecords = new ulong[2 * n];
                for (int i = 0; i < n; i++)
                {
                    rankRecords[2 * i] = rankOf[strings[i]];
                    rankRecords[2 * i + 1] = (ulong)i;
                }
                long heap = TextSort.HeapFallbackRuns;
                fixed (ulong* t = tailRecords, r = rankRecords)
                {
                    TextSort.IntroArgSortTextRecords(t, n, text, 4L * width, 32, 4 * width, 4);
                    TextSort.IntroArgSortRecords(r, n);
                }
                Enumerable.Range(0, n).Select(i => tailRecords[2 * i + 1])
                    .Should().Equal(Enumerable.Range(0, n).Select(i => rankRecords[2 * i + 1]), n + " strings like " + strings[0]);
                if (strings[0].Length == 9)
                    (TextSort.HeapFallbackRuns - heap).Should().Be(2, "the adversary drives both ports into the heapsort fallback");
            }
        }

        /// <summary>
        ///     Every emitted key kernel equals its managed twin (keys and unit ORs), over every packing and unit count, the
        ///     identity and an index array, positive / negative / zero strides.
        /// </summary>
        [TestMethod]
        public void KeyKernels_EqualTheManagedTwins()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("Dynamic code is unavailable: the managed twins are the kernels.");
            var rng = new System.Random(3);
            var packings = new List<(int Unit, int Units, int Bits, bool Or)>();
            for (int u = 1; u <= 8; u++)
            {
                packings.Add((1, u, 8, false));
                packings.Add((4, u, 8, true));
                packings.Add((4, u, 8, false));
            }
            for (int u = 1; u <= 4; u++)
                packings.Add((4, u, 16, false));
            for (int u = 1; u <= 3; u++)
                packings.Add((4, u, 21, false));
            for (int u = 1; u <= 2; u++)
                packings.Add((4, u, 32, false));

            const int count = 37;
            foreach (var (unit, units, bits, or) in packings)
            {
                int width = System.Math.Max(units * unit, 8 * unit);   // a wider element: the kernel reads only its first units
                var data = new byte[count * width * 3];
                rng.NextBytes(data);
                if (unit == 4 && bits < 32)
                {
                    // Units that fit the packing (the caller re-keys otherwise), with some ASCII and some zeros.
                    uint limit = bits == 8 ? 0x100u : bits == 16 ? 0x10000u : 0x200000u;
                    for (int i = 0; i < data.Length; i += 4)
                        BitConverter.TryWriteBytes(data.AsSpan(i, 4), (uint)rng.Next(4) == 0 ? 0u : (uint)rng.NextInt64(limit));
                }
                var kernel = ILKernelGenerator.GetTextSortKeyKernel(unit, units, bits, or);
                kernel.Should().NotBeNull();
                var index = Enumerable.Range(0, count).OrderBy(_ => rng.Next()).Select(i => (long)i).ToArray();
                foreach (long stride in new long[] { width, 2L * width, -width, 0 })
                {
                    foreach (bool indexed in new[] { false, true })
                    {
                        var got = new ulong[count];
                        var want = new ulong[count];
                        var gotOr = new uint[4];
                        var wantOr = new uint[4];
                        fixed (byte* d = data)
                        fixed (long* ix = index)
                        fixed (ulong* g = got, w = want)
                        fixed (uint* go = gotOr, wo = wantOr)
                        {
                            byte* text = stride < 0 ? d + (count - 1) * width : d;
                            kernel(text, stride, indexed ? ix : null, count, g, go);
                            ILKernelGenerator.TextSortKeysManaged(unit, units, bits, or, text, stride, indexed ? ix : null, count, w, wo);
                        }
                        got.Should().Equal(want, $"keys of unit {unit}, {units} units, {bits} bits, stride {stride}, indexed {indexed}");
                        if (or)
                            gotOr.Should().Equal(wantOr);
                    }
                }
            }
        }

        /// <summary>The gather kernel equals its managed twin for every width (unrolled and loop forms), with and without an index array.</summary>
        [TestMethod]
        public void GatherKernel_EqualsTheManagedTwin()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("Dynamic code is unavailable: the managed twin is the kernel.");
            var rng = new System.Random(4);
            const int count = 23;
            foreach (int width in Enumerable.Range(0, 41).Concat(new[] { 64, 100, 300, 301 }))
            {
                var kernel = ILKernelGenerator.GetTextGatherKernel(width);
                kernel.Should().NotBeNull();
                var src = new byte[count * width * 2 + 1];
                rng.NextBytes(src);
                var index = Enumerable.Range(0, count).OrderBy(_ => rng.Next()).Select(i => (long)i).ToArray();
                foreach (bool indexed in new[] { false, true })
                {
                    foreach (long srcStride in new long[] { width, -2L * width })
                    {
                        var got = new byte[count * width * 2 + 8];
                        var want = new byte[got.Length];
                        fixed (byte* s = src, g = got, w = want)
                        fixed (long* ix = index)
                        {
                            byte* from = srcStride < 0 ? s + (count - 1) * 2 * width : s;
                            kernel(from, srcStride, indexed ? ix : null, count, g + 3, 2L * width);
                            ILKernelGenerator.TextGatherManaged(width, from, srcStride, indexed ? ix : null, count, w + 3, 2L * width);
                        }
                        got.Should().Equal(want, $"width {width}, indexed {indexed}, stride {srcStride}");
                    }
                }
            }
        }

        // ------------------------------------------------------------------ memory safety

        /// <summary>
        ///     Every route reads only its operand's bytes: arrays laid against an inaccessible page (at the start and at the
        ///     end of the readable span) — <c>U8</c> elements, whose key is one 32-byte load, <c>U9</c>, <c>U3</c>, <c>S8</c>,
        ///     <c>S5</c> — sorted, argsorted with both kinds, sorted in place, as long lines, a 17-element line and short rows.
        /// </summary>
        [TestMethod]
        public void Sorts_ReadNothingOutsideTheirOperands()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            var rng = new System.Random(8);
            foreach (var dtype in new[] { "U8", "U9", "U3", "S8", "S5" })
            {
                var d = np.dtype(dtype);
                int chars = dtype[0] == 'U' ? d.itemsize / 4 : d.itemsize;
                foreach (var shape in new[] { new long[] { 600 }, new long[] { 17 }, new long[] { 30, 10 } })
                {
                    long count = shape.Aggregate(1L, (x, y) => x * y);
                    var values = U(Enumerable.Range(0, (int)count).Select(_ => new string(Enumerable.Range(0, rng.Next(chars + 1)).Select(_ => (char)('a' + rng.Next(3))).ToArray())).ToArray());
                    var data = values.astype(d).reshape(new Shape(shape));
                    long bytes = count * d.itemsize;
                    foreach (bool atEnd in new[] { false, true })
                    {
                        using var fence = new FencedMemory(pages: (int)((bytes + FencedMemory.PageBytes - 1) / FencedMemory.PageBytes) + 1);
                        byte* p = fence.Place(bytes, atEnd);
                        using var fenced = np.frombuffer((IntPtr)p, bytes, d).reshape(new Shape(shape));
                        np.copyto(fenced, data);
                        Ints(np.argsort(fenced)).Should().Equal(Ints(np.argsort(data)));
                        Ints(np.argsort(fenced, kind: "stable")).Should().Equal(Ints(np.argsort(data, kind: "stable")));
                        np.sort(fenced).tobytes().Should().Equal(np.sort(data).tobytes());
                        fenced.sort();
                        fenced.tobytes().Should().Equal(np.sort(data).tobytes());
                    }
                }
            }
        }
    }
}
