using System;
using System.Numerics;
using AwesomeAssertions;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     A copy or a cast of a stepped, reversed or transposed view reads NO byte outside the view's buffer: every
    ///     source array of these tests is a view over <see cref="FencedMemory"/>, with its first byte directly above an
    ///     inaccessible page or its last byte directly below one.
    /// </summary>
    /// <remarks>
    ///     <para><b>The defect this pins.</b> The strided SIMD kernels read whole vectors. The two-load deinterleave
    ///     of the one- and two-byte kernels (<c>a[::2]</c>: <c>DirectILKernelGenerator.Cast.SubwordCopy.cs</c>,
    ///     <c>.Cast.SubwordNarrow.cs</c>, <c>.Cast.SubwordWiden.cs</c>, the float16-to-bool kernel) read ONE element
    ///     past the last element of the view. Values were right, so no value test saw it; the process ended with an
    ///     access violation when the view's last element was the last one of a buffer that ends where mapped memory
    ///     ends:</para>
    ///     <code>
    ///     var z = np.zeros(new Shape(40 * 4096), NPTypeCode.Byte);   // 163,840 bytes: its own region of pages
    ///     var c = z["1::2"].copy();                                    // AccessViolationException (NumPy: 81,920 zeros)
    ///     </code>
    ///     <para><b>What a failure looks like.</b> Not a failed assertion: the test host ends ("Test host process
    ///     crashed") with the stack of the kernel that left its operand.</para>
    ///     <para>The values are checked too, against the same cast of a contiguous copy whose elements were read one
    ///     at a time. Windows only (the fences are reserved pages); Inconclusive elsewhere.</para>
    /// </remarks>
    [TestClass]
    public unsafe class StridedCastFenceTests : TestClass
    {
        /// <summary>Every NumSharp dtype.</summary>
        private static readonly NPTypeCode[] AllTypes =
        {
            NPTypeCode.Boolean, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32,
            NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
            NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
        };

        /// <summary>
        ///     The value of base element <paramref name="j"/>: a small non-negative integer every dtype holds exactly
        ///     (0 … 100; bool reads it by truth value), different from its neighbours so that a misplaced element shows.
        /// </summary>
        /// <param name="j">Index in the base buffer.</param>
        /// <returns>The value.</returns>
        private static int ValueAt(long j) => (int)((j * 7 + 3) % 101);

        /// <summary>Writes <see cref="ValueAt"/> of every element into a base buffer of dtype <paramref name="t"/>.</summary>
        /// <param name="p">First element.</param>
        /// <param name="t">The dtype.</param>
        /// <param name="n">Element count.</param>
        private static void Fill(byte* p, NPTypeCode t, long n)
        {
            for (long j = 0; j < n; j++)
            {
                int v = ValueAt(j);
                switch (t)
                {
                    case NPTypeCode.Boolean: p[j] = v != 0 ? (byte)1 : (byte)0; break;
                    case NPTypeCode.SByte: ((sbyte*)p)[j] = (sbyte)v; break;
                    case NPTypeCode.Byte: p[j] = (byte)v; break;
                    case NPTypeCode.Int16: ((short*)p)[j] = (short)v; break;
                    case NPTypeCode.UInt16: ((ushort*)p)[j] = (ushort)v; break;
                    case NPTypeCode.Char: ((char*)p)[j] = (char)v; break;
                    case NPTypeCode.Int32: ((int*)p)[j] = v; break;
                    case NPTypeCode.UInt32: ((uint*)p)[j] = (uint)v; break;
                    case NPTypeCode.Int64: ((long*)p)[j] = v; break;
                    case NPTypeCode.UInt64: ((ulong*)p)[j] = (ulong)v; break;
                    case NPTypeCode.Half: ((Half*)p)[j] = (Half)v; break;
                    case NPTypeCode.Single: ((float*)p)[j] = v; break;
                    case NPTypeCode.Double: ((double*)p)[j] = v; break;
                    case NPTypeCode.Decimal: ((decimal*)p)[j] = v; break;
                    default: ((Complex*)p)[j] = new Complex(v, 0); break;
                }
            }
        }

        /// <summary>A boxed element as a number (bool as 0 / 1, complex by its real part).</summary>
        /// <param name="v">The element.</param>
        /// <returns>Its value.</returns>
        private static double Number(object v) => v switch
        {
            bool b => b ? 1 : 0,
            char c => c,
            Half h => (double)h,
            Complex z => z.Real,
            _ => Convert.ToDouble(v),
        };

        /// <summary>A base array of <paramref name="n"/> elements of <paramref name="t"/> over fenced memory.</summary>
        /// <param name="fence">The fenced memory.</param>
        /// <param name="t">The dtype.</param>
        /// <param name="n">Element count.</param>
        /// <param name="atEnd">The buffer's last byte is the last accessible one (else its first byte is the first
        ///     accessible one).</param>
        /// <returns>The array: a view of the fence, filled with <see cref="ValueAt"/>. It owns no memory.</returns>
        private static NDArray FencedBase(FencedMemory fence, NPTypeCode t, long n, bool atEnd)
        {
            long bytes = n * t.SizeOf();
            byte* low = fence.Place(bytes, atEnd);
            Fill(low, t, n);
            return np.frombuffer((IntPtr)low, bytes, t);
        }

        /// <summary>
        ///     Checks one view of a fenced base: its contiguous copy holds the base's elements in the view's order, and
        ///     its cast to every dtype equals the cast of that copy.
        /// </summary>
        /// <param name="what">Label for the failure message.</param>
        /// <param name="view">The view under test (stepped, reversed, transposed…).</param>
        /// <param name="expectedIndices">For each element of the view in logical order, its index in the base.</param>
        /// <param name="src">The view's dtype.</param>
        private static void AssertViewCopiesAndCasts(string what, NDArray view, long[] expectedIndices, NPTypeCode src)
        {
            // The same-type copy: the strided copy kernel of the dtype's width.
            using var copy = view.copy();
            copy.size.Should().Be(expectedIndices.Length, what);
            using var flat = copy.flatten();
            for (long k = 0; k < expectedIndices.Length; k++)
            {
                double want = src == NPTypeCode.Boolean ? (ValueAt(expectedIndices[k]) != 0 ? 1 : 0) : ValueAt(expectedIndices[k]);
                double have = Number(flat.GetAtIndex(k));
                if (have != want)
                    Assert.Fail($"{what}: copy[{k}] is {have}, the base holds {want} at {expectedIndices[k]}");
            }

            // Every cast: the strided cast kernel of the pair against the contiguous one of the same pair.
            foreach (var dst in AllTypes)
            {
                using var strided = view.astype(dst);
                using var contiguous = copy.astype(dst);
                strided.Shape.dimensions.Should().Equal(contiguous.Shape.dimensions, $"{what} -> {dst}");
                using var a = np.ascontiguousarray(strided);
                using var b = np.ascontiguousarray(contiguous);
                if (!a.Unsafe.ReadOnlyBytes().SequenceEqual(b.Unsafe.ReadOnlyBytes()))
                    Assert.Fail($"{what} -> {dst}: the cast of the view differs from the cast of its contiguous copy");
            }
        }

        /// <summary>
        ///     One-dimensional views of a buffer against a fence, for every dtype: contiguous, every step of 2 and 3 at
        ///     every start, reversed, and negative steps. Lengths cover each kernel's vector group (32 results for
        ///     one-byte elements, 16 for two-byte ones, 8 and 4 above) with one group, one more element, and one less.
        /// </summary>
        [TestMethod]
        public void OneDimensionalViews_OfAFencedBuffer_CopyAndCastWithoutLeavingIt()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("Fenced memory needs VirtualAlloc (Windows).");

            using var fence = new FencedMemory(pages: 2);
            // (slice text, start index, step) of every view; the count follows from the base length.
            var layouts = new (string Slice, Func<long, long> Start, long Step)[]
            {
                (":", n => 0, 1), ("::2", n => 0, 2), ("1::2", n => 1, 2), ("::3", n => 0, 3), ("1::3", n => 1, 3), ("2::3", n => 2, 3),
                ("::-1", n => n - 1, -1), ("::-2", n => n - 1, -2), ("-2::-2", n => n - 2, -2), ("::-3", n => n - 1, -3),
            };
            foreach (var src in AllTypes)
            foreach (long n in new long[] { 1, 2, 3, 15, 16, 17, 31, 32, 33, 34, 63, 64, 65, 66, 67, 129, 130, 131 })
            foreach (bool atEnd in new[] { false, true })
            {
                var baseArray = FencedBase(fence, src, n, atEnd);
                foreach (var (slice, startOf, step) in layouts)
                {
                    long start = startOf(n);
                    if (start < 0 || start >= n)
                        continue;
                    long count = step > 0 ? (n - start + step - 1) / step : (start + (-step)) / (-step);
                    var indices = new long[count];
                    for (long k = 0; k < count; k++)
                        indices[k] = start + k * step;
                    AssertViewCopiesAndCasts($"{src} n={n} [{slice}] atEnd={atEnd}", baseArray[slice], indices, src);
                }
            }
        }

        /// <summary>
        ///     Two-dimensional views of a buffer against a fence, for every dtype: stepped and reversed columns (the
        ///     inner loop of the strided kernels, once per row), stepped and reversed rows, and the transpose. The last
        ///     row of the buffer ends at the high fence and its first row starts at the low one, so a kernel that
        ///     reads past a row's last element, or before its first, leaves the buffer on one of them.
        /// </summary>
        [TestMethod]
        public void TwoDimensionalViews_OfAFencedBuffer_CopyAndCastWithoutLeavingIt()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("Fenced memory needs VirtualAlloc (Windows).");

            using var fence = new FencedMemory(pages: 4);
            const long rows = 3;
            // (slice text, row start, row step, column start, column step, transposed afterwards)
            var layouts = new (string Slice, long R0, long RStep, long C0, long CStep, bool T)[]
            {
                (":, ::2", 0, 1, 0, 2, false), (":, 1::2", 0, 1, 1, 2, false), (":, ::-1", 0, 1, -1, -1, false), (":, ::-2", 0, 1, -1, -2, false),
                ("::-1, ::2", -1, -1, 0, 2, false), ("::-1, 1::2", -1, -1, 1, 2, false), ("::-1, ::-1", -1, -1, -1, -1, false),
                ("::2, :", 0, 2, 0, 1, false), (":, 2::3", 0, 1, 2, 3, false), (":, :", 0, 1, 0, 1, true), (":, 1::2", 0, 1, 1, 2, true),
            };
            foreach (var src in AllTypes)
            foreach (long cols in new long[] { 2, 3, 16, 17, 32, 33, 34, 64, 65, 66, 130 })
            foreach (bool atEnd in new[] { false, true })
            {
                var baseArray = FencedBase(fence, src, rows * cols, atEnd).reshape(rows, cols);
                foreach (var (slice, r0Raw, rStep, c0Raw, cStep, transposed) in layouts)
                {
                    long r0 = r0Raw < 0 ? rows - 1 : r0Raw, c0 = c0Raw < 0 ? cols - 1 : c0Raw;
                    if (c0 >= cols)
                        continue;
                    long rCount = rStep > 0 ? (rows - r0 + rStep - 1) / rStep : (r0 + (-rStep)) / (-rStep);
                    long cCount = cStep > 0 ? (cols - c0 + cStep - 1) / cStep : (c0 + (-cStep)) / (-cStep);
                    var view = baseArray[slice];
                    var indices = new long[rCount * cCount];
                    if (!transposed)
                    {
                        for (long i = 0; i < rCount; i++)
                        for (long j = 0; j < cCount; j++)
                            indices[i * cCount + j] = (r0 + i * rStep) * cols + (c0 + j * cStep);
                    }
                    else
                    {
                        view = view.T;
                        for (long j = 0; j < cCount; j++)
                        for (long i = 0; i < rCount; i++)
                            indices[j * rCount + i] = (r0 + i * rStep) * cols + (c0 + j * cStep);
                    }
                    AssertViewCopiesAndCasts($"{src} ({rows},{cols}) [{slice}]{(transposed ? ".T" : "")} atEnd={atEnd}", view, indices, src);
                }
            }
        }

        /// <summary>
        ///     The call of the class remarks, on the allocator's own memory: a zeroed array of 40 pages is its own
        ///     region, with nothing mapped behind it (the region's allocation granule is 16 pages; the last eight are
        ///     never usable). A stepped view that ends at its last element is copied and cast.
        /// </summary>
        [TestMethod]
        public void SteppedViewEndingAtTheLastElementOfAZeroedRegion_IsCopied()
        {
            foreach (var t in new[] { NPTypeCode.Byte, NPTypeCode.Boolean, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Char, NPTypeCode.Half })
            {
                long n = 40L * 4096 / t.SizeOf();
                using var z = np.zeros(new Shape(n), t);
                using var copy = z["1::2"].copy();
                copy.size.Should().Be(n / 2);
                np.any(copy).Should().BeFalse($"{t}: a copy of zeros");
                using var narrow = z["1::2"].astype(NPTypeCode.Boolean);
                np.any(narrow).Should().BeFalse($"{t} -> bool: a cast of zeros");
                using var wide = z["1::2"].astype(NPTypeCode.Int16);
                np.any(wide).Should().BeFalse($"{t} -> int16: a cast of zeros");
                using var same = z["1::2"].astype(t.SizeOf() == 1 ? NPTypeCode.SByte : NPTypeCode.UInt16);
                np.any(same).Should().BeFalse($"{t}: a same-width cast of zeros");
            }
        }
    }
}
