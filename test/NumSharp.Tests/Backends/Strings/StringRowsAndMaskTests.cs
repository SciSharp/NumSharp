using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Strings phase 2, P2.A (docs/plans/strings-phase2.md §3.2 D16 / D17): the rows forms of the text kernels, driven by
    ///     <see cref="NDIterRef.ForEachRows"/>, and the masked comparison kernels that blend <c>out = mask ? result : out</c>.
    ///     Every form must compute exactly what the 1-D kernels compute, so the tests compare them three ways: the rows
    ///     kernels against the 1-D kernels called row by row on raw memory (negative and zero strides included), the masked
    ///     kernels against the managed reference blended under the mask (mask bytes other than 0 and 1 included), and the
    ///     public calls through the new routes against the same calls through the old ones (the
    ///     <c>RowsDisabled</c> / <c>MaskedBlendDisabled</c> switches), with route counters proving which route ran. The old
    ///     routes are the ones the string oracle validated against NumPy 2.4.2; the oracle replays every section through the
    ///     new routes as well.
    /// </summary>
    [TestClass]
    public unsafe class StringRowsAndMaskTests
    {
        // ------------------------------------------------------------------ raw-memory helpers

        /// <summary>
        ///     A raw operand laid out as <paramref name="rows"/> × <paramref name="inner"/> elements of
        ///     <paramref name="width"/> bytes at the given byte strides (zero and negative strides allowed): the buffer and
        ///     the offset of element (0, 0) inside it.
        /// </summary>
        private sealed class RawOperand
        {
            /// <summary>The backing bytes.</summary>
            internal readonly byte[] Buffer;

            /// <summary>The byte offset of element (0, 0).</summary>
            internal readonly long Origin;

            /// <summary>The byte distance between two elements of a row.</summary>
            internal readonly long InnerStride;

            /// <summary>The byte distance between two rows.</summary>
            internal readonly long OuterStride;

            /// <summary>Lays the operand out (see the type summary).</summary>
            /// <param name="rows">The row count.</param>
            /// <param name="inner">The elements per row.</param>
            /// <param name="width">The element width in bytes.</param>
            /// <param name="innerStride">The element stride (bytes; may be 0 or negative).</param>
            /// <param name="outerStride">The row stride (bytes; may be 0 or negative).</param>
            internal RawOperand(int rows, int inner, int width, long innerStride, long outerStride)
            {
                long lo = 0, hi = 0;
                foreach (long r in new long[] { 0, rows - 1 })
                    foreach (long i in new long[] { 0, inner - 1 })
                    {
                        long o = r * outerStride + i * innerStride;
                        lo = System.Math.Min(lo, o);
                        hi = System.Math.Max(hi, o);
                    }
                Origin = -lo;
                Buffer = new byte[hi - lo + width];
                InnerStride = innerStride;
                OuterStride = outerStride;
            }

            /// <summary>The byte offset of element (r, i).</summary>
            /// <param name="r">The row.</param>
            /// <param name="i">The element in the row.</param>
            /// <returns>The offset into <see cref="Buffer"/>.</returns>
            internal long At(long r, long i) => Origin + r * OuterStride + i * InnerStride;
        }

        /// <summary>
        ///     Writes random text into every element of a raw operand: units from a tiny alphabet (NUL, 'a', 'b'), a random
        ///     length (so trailing NULs and strict prefixes are common), and for <c>U</c> now and then a unit above U+10FFFF
        ///     or 0xFFFFFFFF (compared unsigned).
        /// </summary>
        /// <param name="rng">The random source.</param>
        /// <param name="x">The operand.</param>
        /// <param name="rows">Its rows.</param>
        /// <param name="inner">Its elements per row.</param>
        /// <param name="width">Its element width in bytes.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        private static void FillText(System.Random rng, RawOperand x, int rows, int inner, int width, int unit)
        {
            for (int r = 0; r < rows; r++)
                for (int i = 0; i < inner; i++)
                {
                    long at = x.At(r, i);
                    int len = rng.Next(width / unit + 1);
                    for (int u = 0; u < width / unit; u++)
                    {
                        uint v = u >= len ? 0u : (uint)"\0ab"[rng.Next(3)];
                        if (unit == 4 && u < len && rng.Next(16) == 0)
                            v = rng.Next(2) == 0 ? 0x110000u : 0xFFFFFFFFu;
                        if (unit == 1)
                            x.Buffer[at + u] = (byte)v;
                        else
                            BitConverter.GetBytes(v).CopyTo(x.Buffer, at + u * 4);
                    }
                }
        }

        /// <summary>The strides (inner, outer) a raw operand is tested with: dense, broadcast in either axis, both, reversed rows.</summary>
        /// <param name="pattern">0 dense, 1 inner broadcast (a column value), 2 outer broadcast (a row vector), 3 scalar, 4 reversed rows.</param>
        /// <param name="width">The element width.</param>
        /// <param name="inner">The elements per row.</param>
        /// <returns>The (inner, outer) byte strides.</returns>
        private static (long Inner, long Outer) Strides(int pattern, int width, int inner) => pattern switch
        {
            0 => (width + 3, (width + 3) * (inner + 2)),
            1 => (0, width + 5),
            2 => (width, 0),
            3 => (0, 0),
            _ => (width, -(width * (inner + 1))),
        };

        // ------------------------------------------------------------------ the kernels on raw memory

        /// <summary>
        ///     The rows comparison kernels (plain and masked) against the managed 1-D reference run row by row: both
        ///     classes, widths across the hoistable limit, both operand orders, every op, and the operands in five stride
        ///     patterns each (dense, inner broadcast, row broadcast, scalar, reversed rows). The masked form starts from
        ///     random output bytes and reads mask bytes of 0, 1, 2 and 255 (any nonzero selects).
        /// </summary>
        [TestMethod]
        public void CompareRowsKernels_MatchTheReferenceRowByRow()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("Dynamic code is unavailable: the managed routes are the kernels.");
            var rng = new System.Random(20261015);
            const int rows = 5, inner = 4;
            long mismatches = 0, maskedOffKept = 0;
            // Every stackalloc sits at method level: one inside a loop would grow the frame on each iteration.
            void** p = stackalloc void*[4];
            long* si = stackalloc long[4];
            long* so = stackalloc long[4];
            void** rp = stackalloc void*[3];
            long* rs = stackalloc long[3];
            foreach (int unit in new[] { 1, 4 })
            foreach (int wa in unit == 1 ? new[] { 1, 8, 33, 300 } : new[] { 4, 36, 260 })
            foreach (int wb in unit == 1 ? new[] { 3, 8, 300 } : new[] { 12, 36 })
            foreach (TextCompareOp op in Enum.GetValues<TextCompareOp>())
            foreach (int aIndex in new[] { 0, 1 })
            foreach (bool masked in new[] { false, true })
            {
                var kernel = ILKernelGenerator.GetTextCompareRowsKernel(op, unit, wa, wb, aIndex, masked);
                var reference = NDStringComparison.ManagedKernel(op, unit, wa, wb, aIndex);
                int w0 = aIndex == 0 ? wa : wb, w1 = aIndex == 0 ? wb : wa;
                for (int pa = 0; pa < 5; pa++)
                for (int pb = 0; pb < 5; pb++)
                {
                    var (i0, o0) = Strides(pa, w0, inner);
                    var (i1, o1) = Strides(pb, w1, inner);
                    var x0 = new RawOperand(rows, inner, w0, i0, o0);
                    var x1 = new RawOperand(rows, inner, w1, i1, o1);
                    FillText(rng, x0, rows, inner, w0, unit);
                    FillText(rng, x1, rows, inner, w1, unit);
                    var outRows = new byte[rows * inner];
                    rng.NextBytes(outRows);
                    for (int k = 0; k < outRows.Length; k++)
                        outRows[k] &= 1;
                    var start = (byte[])outRows.Clone();
                    var mask = new byte[rows * inner];
                    for (int k = 0; k < mask.Length; k++)
                        mask[k] = new byte[] { 0, 1, 2, 255 }[rng.Next(4)];
                    var expected = (byte[])outRows.Clone();
                    var row = new byte[inner];

                    fixed (byte* b0 = x0.Buffer, b1 = x1.Buffer, bo = outRows, bm = mask, br = row)
                    {
                        p[0] = b0 + x0.Origin; p[1] = b1 + x1.Origin; p[2] = bo; p[3] = bm;
                        si[0] = i0; si[1] = i1; si[2] = 1; si[3] = 1;
                        so[0] = o0; so[1] = o1; so[2] = inner; so[3] = inner;
                        kernel(p, si, inner, so, rows, null);

                        // The reference: the managed 1-D kernel per row, then the blend for the masked form.
                        for (int r = 0; r < rows; r++)
                        {
                            rp[0] = b0 + x0.At(r, 0); rp[1] = b1 + x1.At(r, 0); rp[2] = br;
                            rs[0] = i0; rs[1] = i1; rs[2] = 1;
                            reference(rp, rs, inner, null);
                            for (int i = 0; i < inner; i++)
                            {
                                int k = r * inner + i;
                                expected[k] = !masked || mask[k] != 0 ? row[i] : start[k];
                            }
                        }
                    }
                    for (int k = 0; k < expected.Length; k++)
                    {
                        if (outRows[k] != expected[k])
                            mismatches++;
                        if (masked && mask[k] == 0)
                            maskedOffKept++;
                    }
                }
            }
            mismatches.Should().Be(0);
            maskedOffKept.Should().BeGreaterThan(0, "the masked forms must have met masked-off elements");
        }

        /// <summary>
        ///     The 1-D masked comparison kernels against the managed reference blended under the mask: every broadcast
        ///     pattern of A and B, an output walked forward, with a gap and backward, and a mask that is dense, strided or a
        ///     single byte for the whole call.
        /// </summary>
        [TestMethod]
        public void CompareMaskedKernels_BlendTheReferenceUnderTheMask()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("Dynamic code is unavailable: the managed routes are the kernels.");
            var rng = new System.Random(7);
            const int n = 37;
            long mismatches = 0;
            void** p = stackalloc void*[4];
            long* st = stackalloc long[4];
            void** rp = stackalloc void*[3];
            long* rs = stackalloc long[3];
            foreach (int unit in new[] { 1, 4 })
            foreach (int wa in unit == 1 ? new[] { 2, 16, 40 } : new[] { 8, 32, 44 })
            foreach (int wb in unit == 1 ? new[] { 5, 16 } : new[] { 4, 32 })
            foreach (TextCompareOp op in Enum.GetValues<TextCompareOp>())
            foreach (int aIndex in new[] { 0, 1 })
            {
                var kernel = ILKernelGenerator.GetTextCompareMaskedKernel(op, unit, wa, wb, aIndex);
                var reference = NDStringComparison.ManagedKernel(op, unit, wa, wb, aIndex);
                int w0 = aIndex == 0 ? wa : wb, w1 = aIndex == 0 ? wb : wa;
                for (int pattern = 0; pattern < 4; pattern++)
                foreach (long outStride in new long[] { 1, 2, -1 })
                foreach (long maskStride in new long[] { 1, 3, 0 })
                {
                    long s0 = (pattern & 1) != 0 ? 0 : w0 + 1, s1 = (pattern & 2) != 0 ? 0 : w1;
                    var x0 = new RawOperand(1, n, w0, s0, 0);
                    var x1 = new RawOperand(1, n, w1, s1, 0);
                    FillText(rng, x0, 1, n, w0, unit);
                    FillText(rng, x1, 1, n, w1, unit);
                    var outBuf = new byte[n * 2 + 1];
                    for (int k = 0; k < outBuf.Length; k++)
                        outBuf[k] = (byte)rng.Next(2);
                    var start = (byte[])outBuf.Clone();
                    var maskBuf = new byte[n * 3 + 1];
                    for (int k = 0; k < maskBuf.Length; k++)
                        maskBuf[k] = new byte[] { 0, 1, 2, 255 }[rng.Next(4)];
                    var refOut = new byte[n];
                    long outOrigin = outStride < 0 ? n - 1 : 0;
                    fixed (byte* b0 = x0.Buffer, b1 = x1.Buffer, bo = outBuf, bm = maskBuf, br = refOut)
                    {
                        p[0] = b0 + x0.Origin; p[1] = b1 + x1.Origin; p[2] = bo + outOrigin; p[3] = bm;
                        st[0] = s0; st[1] = s1; st[2] = outStride; st[3] = maskStride;
                        kernel(p, st, n, null);
                        rp[0] = p[0]; rp[1] = p[1]; rp[2] = br;
                        rs[0] = s0; rs[1] = s1; rs[2] = 1;
                        reference(rp, rs, n, null);
                    }
                    for (int i = 0; i < n; i++)
                    {
                        long o = outOrigin + i * outStride;
                        byte want = maskBuf[i * maskStride] != 0 ? refOut[i] : start[o];
                        if (outBuf[o] != want)
                            mismatches++;
                    }
                    // Bytes the output does not cover are never written.
                    for (long k = 0; k < outBuf.Length; k++)
                    {
                        bool covered = outStride == 2 ? k % 2 == 0 && k / 2 < n : k < n;
                        if (!covered && outBuf[k] != start[k])
                            mismatches++;
                    }
                }
            }
            mismatches.Should().Be(0);
        }

        /// <summary>
        ///     The rows truth kernels (both flavors) against the managed 1-D kernel row by row, the cast flavor's flag
        ///     included: set exactly when some row holds a <c>U</c> element that is one unit above U+10FFFF, wherever that
        ///     row sits in the block.
        /// </summary>
        [TestMethod]
        public void TruthRowsKernels_MatchTheReferenceRowByRow_FlagIncluded()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("Dynamic code is unavailable: the managed routes are the kernels.");
            var rng = new System.Random(15);
            const int rows = 6, inner = 3;
            void** p = stackalloc void*[2];
            long* si = stackalloc long[2];
            long* so = stackalloc long[2];
            void** rp = stackalloc void*[2];
            long* rs = stackalloc long[2];
            int flaggedBlocks = 0;
            foreach (int unit in new[] { 1, 4 })
            foreach (int w in unit == 1 ? new[] { 1, 8, 33, 300 } : new[] { 4, 12, 36, 300 })
            foreach (bool check in new[] { false, true })
            for (int pattern = 0; pattern < 5; pattern++)
            for (int trial = 0; trial < 3; trial++)
            {
                var (ii, oo) = Strides(pattern, w, inner);
                var x = new RawOperand(rows, inner, w, ii, oo);
                FillText(rng, x, rows, inner, w, unit);
                // A lone unit above U+10FFFF in a random later row, in two trials of three.
                if (unit == 4 && trial != 0)
                {
                    long at = x.At(1 + rng.Next(rows - 1), rng.Next(inner));
                    Array.Clear(x.Buffer, (int)at, w);
                    BitConverter.GetBytes(0x110000u).CopyTo(x.Buffer, at);
                }
                var got = new byte[rows * inner];
                var want = new byte[rows * inner];
                int f1 = 0, f2 = 0;
                var reference = NDStringTruth.ManagedKernel(unit, w, check);
                fixed (byte* bx = x.Buffer, bg = got, bw = want)
                {
                    p[0] = bx + x.Origin; p[1] = bg;
                    si[0] = ii; si[1] = 1;
                    so[0] = oo; so[1] = inner;
                    ILKernelGenerator.GetTextTruthRowsKernel(unit, w, check)(p, si, inner, so, rows, &f1);
                    for (int r = 0; r < rows; r++)
                    {
                        rp[0] = bx + x.At(r, 0); rp[1] = bw + r * inner;
                        rs[0] = ii; rs[1] = 1;
                        reference(rp, rs, inner, &f2);
                    }
                }
                got.Should().Equal(want);
                f1.Should().Be(f2);
                flaggedBlocks += f1;
            }
            flaggedBlocks.Should().BeGreaterThan(0, "the cast flavor must have met a lone unit above U+10FFFF");
        }

        // ------------------------------------------------------------------ the public routes

        /// <summary>Runs <paramref name="call"/> through the new routes and through the old ones and returns both results.</summary>
        /// <param name="call">The call (it builds its own outputs, so both runs start from the same state).</param>
        /// <param name="rowsRuns">The comparison rows route's counter delta of the new run.</param>
        /// <param name="blendRuns">The blend route's counter delta of the new run.</param>
        /// <returns>The new route's and the old route's results as bool arrays.</returns>
        private static (bool[] New, bool[] Old) BothRoutes(Func<NDArray> call, out long rowsRuns, out long blendRuns)
        {
            long r0 = NDStringComparison.RowsRuns, b0 = NDStringComparison.MaskedBlendRuns;
            var fresh = call().ToArray<bool>();
            rowsRuns = NDStringComparison.RowsRuns - r0;
            blendRuns = NDStringComparison.MaskedBlendRuns - b0;
            NDStringComparison.RowsDisabled = true;
            NDStringComparison.MaskedBlendDisabled = true;
            try
            {
                return (fresh, call().ToArray<bool>());
            }
            finally
            {
                NDStringComparison.RowsDisabled = false;
                NDStringComparison.MaskedBlendDisabled = false;
            }
        }

        /// <summary>Random words of 0..<paramref name="width"/> letters from a three-letter alphabet, as a <c>U</c> array.</summary>
        /// <param name="rng">The random source.</param>
        /// <param name="width">The width (characters).</param>
        /// <param name="shape">The shape.</param>
        /// <returns>The array.</returns>
        private static NDArray Words(System.Random rng, int width, params long[] shape)
        {
            long n = shape.Aggregate(1L, (a, b) => a * b);
            var s = Enumerable.Range(0, (int)n).Select(_ => new string(Enumerable.Range(0, rng.Next(width + 1)).Select(__ => "abc"[rng.Next(3)]).ToArray())).ToArray();
            return np.array(s, dtype: $"U{width}").reshape(shape);
        }

        /// <summary>A random bool mask (two thirds true).</summary>
        /// <param name="rng">The random source.</param>
        /// <param name="shape">The shape.</param>
        /// <returns>The mask.</returns>
        private static NDArray Mask(System.Random rng, params long[] shape)
        {
            long n = shape.Aggregate(1L, (a, b) => a * b);
            return np.array(Enumerable.Range(0, (int)n).Select(_ => rng.Next(3) != 0).ToArray()).reshape(shape);
        }

        /// <summary>
        ///     The rows route of the comparisons: narrow slices against a scalar and against each other, a column-broadcast
        ///     operand, a row-broadcast operand against a strided view, a 3-D walk whose leading axis does not fold (the
        ///     per-block odometer), reversed rows and an F-ordered pair; each the same bytes as the per-row route, and the
        ///     rows route really ran.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // flips the process-wide string-comparison route switches (NDStringComparison)
        public void Comparisons_RowsRoute_MatchesThePerRowRoute()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("Dynamic code is unavailable: the rows route needs the IL kernels.");
            var rng = new System.Random(3);
            var m = Words(rng, 4, 120, 10);
            var col = Words(rng, 4, 120, 1);
            var row = Words(rng, 4, 3);
            var t3 = Words(rng, 4, 6, 20, 9);
            var t4 = Words(rng, 3, 6, 6, 5, 7);
            var needle = np.array("ab");
            var cases = new (string, Func<NDArray>)[]
            {
                // Every axis stepped: no two adjacent axes coalesce, so two leading axes are walked block by block.
                ("4-D, two leading axes walked per block", () => np.equal(t4["::2, ::2, ::2, 1:4"], needle)),
                ("slice vs scalar", () => np.equal(m[":, :3"], needle)),
                ("slice vs slice", () => np.less(m[":, :3"], m[":, 5:8"])),
                ("column broadcast", () => np.greater_equal(col, m[":, :3"])),
                ("row broadcast vs strided", () => np.not_equal(row, m[":, ::4"])),
                ("3-D leading axis", () => np.less_equal(t3["::2, :, :4"], t3["1::2, :, 4:8"])),
                ("reversed rows", () => np.greater(m["::-1, :3"], m[":, -3:"])),
                // F slices stepped on both axes: element strides (2, 240), which do not coalesce (240 != 2 * 60), so the
                // iterator walks an F-ordered inner axis of 60 and 5 outer rows. (An unstepped F column slice is
                // F-contiguous and takes the trivial loop; a row-stepped one, strides (2, 120), coalesces to one axis.)
                ("F pair of stepped slices", () => np.equal(np.asfortranarray(m)["::2, ::2"], np.asfortranarray(m)["1::2, 1::2"])),
            };
            foreach (var (name, call) in cases)
            {
                var (fresh, old) = BothRoutes(call, out long rows, out _);
                fresh.Should().Equal(old, name);
                rows.Should().Be(1, $"{name} must run through the rows kernel");
            }
        }

        /// <summary>
        ///     The blend route of a <c>where=</c> mask into a bool output: a dense mask (1-D through the trivial loop, and
        ///     over narrow rows), a row mask, a column mask, a scalar mask, a strided output, and an output that IS the mask
        ///     (<c>out=m, where=m</c>); each the same bytes as the per-run route, masked-off elements untouched.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // flips the process-wide string-comparison route switches (NDStringComparison)
        public void Comparisons_BlendRoute_MatchesThePerRunRoute()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("Dynamic code is unavailable: the blend route needs the IL kernels.");
            var rng = new System.Random(5);
            var m = Words(rng, 4, 120, 10);
            var u = Words(rng, 6, 700);
            var v = Words(rng, 6, 700);
            var needle = np.array("ab");
            var mk = Mask(rng, 120, 3);
            var mk1 = Mask(rng, 700);
            var mkRow = Mask(rng, 3);
            var mkCol = Mask(rng, 120, 1);
            var cases = new (string, Func<NDArray>, bool)[]
            {
                ("dense 1-D (trivial loop)", () => np.equal(u, v, np.ones(new Shape(700L), NPTypeCode.Boolean), mk1), false),
                // The trivial loop with a stride-0 operand: a 0-d mask read once per element at stride 0, both values.
                ("0-d mask true (trivial loop)", () => np.less(u, v, np.zeros(new Shape(700L), NPTypeCode.Boolean), np.array(true)), false),
                ("0-d mask false (trivial loop)", () => np.less(u, v, np.ones(new Shape(700L), NPTypeCode.Boolean), np.array(false)), false),
                // The trivial loop over 1-D views at their own strides: reversed inputs, a stepped output.
                ("reversed 1-D, stepped output (trivial loop)", () =>
                {
                    var big = np.ones(new Shape(1400L), NPTypeCode.Boolean);
                    np.greater(u["::-1"], v, big["::2"], mk1);
                    return big;
                }, false),
                ("dense over narrow rows", () => np.less(m[":, :3"], needle, np.ones(new Shape(120, 3), NPTypeCode.Boolean), mk), true),
                ("row mask", () => np.greater(m[":, :3"], needle, np.ones(new Shape(120, 3), NPTypeCode.Boolean), mkRow), true),
                ("column mask", () => np.not_equal(m[":, :3"], needle, np.ones(new Shape(120, 3), NPTypeCode.Boolean), mkCol), true),
                ("scalar mask", () => np.less_equal(m[":, :3"], needle, np.ones(new Shape(120, 3), NPTypeCode.Boolean), np.array(false)), true),
                ("strided output", () =>
                {
                    var big = np.ones(new Shape(120, 6), NPTypeCode.Boolean);
                    np.greater_equal(m[":, :3"], m[":, 4:7"], big[":, ::2"], mk);
                    return big;
                }, true),
                ("out is the mask", () =>
                {
                    var alias = mk.copy();
                    np.equal(m[":, :3"], needle, alias, alias);
                    return alias;
                }, true),
            };
            foreach (var (name, call, rowsExpected) in cases)
            {
                var (fresh, old) = BothRoutes(call, out long rows, out long blend);
                fresh.Should().Equal(old, name);
                blend.Should().Be(1, $"{name} must run through the blend");
                rows.Should().Be(rowsExpected ? 1 : 0, $"{name}: rows route");
            }
        }

        /// <summary>
        ///     A <c>where=</c> mask into a non-bool <c>out=</c> keeps the per-run route (its write-masked copy-back): the
        ///     masked-off elements keep their value 7, the selected ones get 0 / 1 (probed against NumPy 2.4.2:
        ///     <c>np.less(['b','a'], 'b', out=np.full(2, 7, 'i4'), where=[True, False])</c> is <c>[0, 7]</c>).
        /// </summary>
        [TestMethod]
        public void Comparisons_CastOut_KeepsThePerRunRoute()
        {
            long b0 = NDStringComparison.MaskedBlendRuns;
            var o = np.full(new Shape(2L), 7, NPTypeCode.Int32);
            np.less(np.array(new[] { "b", "a" }), np.array("b"), o, np.array(new[] { true, false }));
            o.ToArray<int>().Should().Equal(0, 7);
            (NDStringComparison.MaskedBlendRuns - b0).Should().Be(0);
            var o2 = np.full(new Shape(2L), 7, NPTypeCode.Int32);
            np.less(np.array(new[] { "a", "c" }), np.array("b"), o2, np.array(new[] { false, true }));
            o2.ToArray<int>().Should().Equal(7, 0);
        }

        /// <summary>
        ///     The rows route of the truth readers: <c>astype(bool)</c>, <c>count_nonzero</c> and <c>nonzero</c> of a narrow
        ///     slice are the per-row route's values, and the cast flavor raises NumPy's SystemError for a lone unit above
        ///     U+10FFFF in a later row (the rows kernel's call-wide flag), while the nonzero flavor does not.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // flips the process-wide string-truth route switch (NDStringTruth.RowsDisabled)
        public void Truth_RowsRoute_MatchesThePerRowRoute_AndRaisesFromAnyRow()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("Dynamic code is unavailable: the rows route needs the IL kernels.");
            var rng = new System.Random(9);
            var m = Words(rng, 3, 150, 7);
            var slice = m[":, 2:5"];
            long t0 = NDStringTruth.RowsRuns;
            var fresh = slice.astype(np.@bool).ToArray<bool>();
            long count = np.count_nonzero(slice);
            var nz = np.nonzero(slice)[0].ToArray<long>();
            (NDStringTruth.RowsRuns - t0).Should().BeGreaterThan(0);
            NDStringTruth.RowsDisabled = true;
            try
            {
                slice.astype(np.@bool).ToArray<bool>().Should().Equal(fresh);
                np.count_nonzero(slice).Should().Be(count);
                np.nonzero(slice)[0].ToArray<long>().Should().Equal(nz);
            }
            finally
            {
                NDStringTruth.RowsDisabled = false;
            }

            // A (4, 4) U1 matrix over raw units: row 2, column 1 is the lone unit 0x110000 (a one-unit str CPython refuses).
            var units = new uint[16];
            for (int k = 0; k < units.Length; k++)
                units[k] = 'a';
            units[2 * 4 + 1] = 0x110000;
            var raw = np.ndarray(new Shape(4, 4), "U1", buffer: units);
            var narrow = raw[":, :3"];
            Action cast = () => narrow.astype(np.@bool);
            cast.Should().Throw<SystemError>().WithMessage(NDStringTruth.InvalidCodePointMessage);
            np.count_nonzero(narrow).Should().Be(12);   // the nonzero route never raises: 0x110000 is nonzero
        }

        /// <summary>
        ///     <see cref="NDIterRef.ForEachRows"/> folds the trailing axes every operand walks contiguously and walks the
        ///     others per block: a 4-D copy through a managed rows kernel visits every element once, in place of the 1-D
        ///     route; <see cref="NDIterRef.CanRunRows"/> refuses a one-axis, a buffered and a masked iteration, and
        ///     <see cref="NDIterRef.ForEachRows"/> throws when asked anyway.
        /// </summary>
        [TestMethod]
        public void ForEachRows_FoldsAndWalksTheBlocks_AndRefusesWhatItCannotDrive()
        {
            // A (6, 6, 5, 7) source viewed as (3, 3, 3, 3), every axis stepped: element strides (420, 70, 14, 1), so no two
            // adjacent axes coalesce (420 != 70 * 3, 70 != 14 * 3, 14 != 1 * 3). The block is the last two axes (3 rows of
            // 3), and the leading two are walked by the odometer — the inner one wraps twice, so a wrong wrap step reads
            // the wrong elements (a view whose leading axes coalesce would only wrap after the last block).
            var src = np.arange(6 * 6 * 5 * 7).reshape(6, 6, 5, 7).astype(NPTypeCode.Int64);
            var view = src["::2, ::2, ::2, 1:4"];
            var dst = np.zeros(new Shape(view.shape), NPTypeCode.Int64);
            long blocks = 0;
            NDInnerLoop2DFunc copy = (ptrs, inner, innerCount, outer, outerCount, aux) =>
            {
                blocks++;
                for (long r = 0; r < outerCount; r++)
                    for (long i = 0; i < innerCount; i++)
                        *(long*)((byte*)ptrs[1] + r * outer[1] + i * inner[1]) = *(long*)((byte*)ptrs[0] + r * outer[0] + i * inner[0]);
            };
            using (var iter = NDIterRef.AdvancedNew(2, new[] { view, dst }, NDIterGlobalFlags.EXTERNAL_LOOP,
                       NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                       new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }))
            {
                iter.CanRunRows.Should().BeTrue();
                iter.ForEachRows(copy);
            }
            dst.ToArray<long>().Should().Equal(view.ToArray<long>());
            blocks.Should().Be(3 * 3, "one call per block: the two leading axes walked here, the 3 rows of each block inside the call");

            // A contiguous pair coalesces to one axis: ForEach's single call is already optimal.
            var a = np.arange(10).astype(NPTypeCode.Int64);
            var b = np.zeros(new Shape(10L), NPTypeCode.Int64);
            using (var iter = NDIterRef.AdvancedNew(2, new[] { a, b }, NDIterGlobalFlags.EXTERNAL_LOOP,
                       NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                       new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }))
            {
                iter.CanRunRows.Should().BeFalse();
                bool threw = false;
                try { iter.ForEachRows(copy); }
                catch (InvalidOperationException) { threw = true; }
                threw.Should().BeTrue();
            }

            // A buffered (cast) iteration hands whole buffer fills to ForEach.
            var strided = np.arange(60).reshape(6, 10)[":, :3"];
            var f = np.zeros(new Shape(6, 3), NPTypeCode.Double);
            using (var iter = NDIterRef.AdvancedNew(2, new[] { strided, f },
                       NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER,
                       NPY_CASTING.NPY_SAFE_CASTING, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY },
                       opDescrs: new[] { DType.Double, DType.Double }))
                iter.CanRunRows.Should().BeFalse();
        }

        /// <summary>
        ///     The rows, masked and rows-masked kernels read and write nothing outside their operands: the text operand, the
        ///     mask and the output each sit against an inaccessible page (first element at the low fence, last element at the
        ///     high fence), and every route runs over narrow rows and the dense 1-D layout. A violation ends the test host.
        /// </summary>
        [TestMethod]
        public void RowsAndMaskedKernels_ReadNothingOutsideTheirOperands()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("Dynamic code is unavailable: the managed routes are the kernels.");
            var rng = new System.Random(21);
            const int rows = 9, cols = 8, width = 5;   // U5: 20-byte elements, so chunk reads overlap inside an element
            var words = Words(rng, width, rows, cols);
            var maskData = Mask(rng, rows, 3);
            long textBytes = rows * cols * width * 4L;
            foreach (bool atEnd in new[] { false, true })
            {
                using var fence = new FencedMemory(pages: (int)((textBytes + 64 + FencedMemory.PageBytes - 1) / FencedMemory.PageBytes) + 1);
                byte* p = fence.Place(textBytes, atEnd);
                using var fenced = np.frombuffer((IntPtr)p, textBytes, np.dtype($"U{width}")).reshape(rows, cols);
                np.copyto(fenced, words);
                // The slice's last column is column 7: its last element ends at the buffer's last byte (atEnd).
                var slice = fenced[":, 5:8"];
                var expected = np.equal(words[":, 5:8"], np.array("ab")).ToArray<bool>();
                np.equal(slice, np.array("ab")).ToArray<bool>().Should().Equal(expected);
                slice.astype(np.@bool).ToArray<bool>().Should().Equal(words[":, 5:8"].astype(np.@bool).ToArray<bool>());

                using var maskFence = new FencedMemory(pages: 2);
                byte* pm = maskFence.Place(rows * 3, atEnd);
                using var fencedMask = np.frombuffer((IntPtr)pm, rows * 3L, np.@bool).reshape(rows, 3);
                np.copyto(fencedMask, maskData);
                using var outFence = new FencedMemory(pages: 2);
                byte* po = outFence.Place(rows * 3, atEnd);
                using var fencedOut = np.frombuffer((IntPtr)po, rows * 3L, np.@bool).reshape(rows, 3);
                fencedOut.fill(true);
                np.less(slice, np.array("ab"), fencedOut, fencedMask);
                var keep = np.ones(new Shape(rows, 3), NPTypeCode.Boolean);
                np.less(words[":, 5:8"], np.array("ab"), keep, maskData);
                fencedOut.ToArray<bool>().Should().Equal(keep.ToArray<bool>());

                // The dense 1-D masked layout (the trivial loop): the flat text, a flat mask and a flat output.
                var flatText = fenced.reshape(rows * cols);
                using var flatMaskFence = new FencedMemory(pages: 2);
                byte* pfm = flatMaskFence.Place(rows * cols, atEnd);
                using var flatMask = np.frombuffer((IntPtr)pfm, (long)(rows * cols), np.@bool);
                np.copyto(flatMask, Mask(rng, rows * cols));
                using var flatOutFence = new FencedMemory(pages: 2);
                byte* pfo = flatOutFence.Place(rows * cols, atEnd);
                using var flatOut = np.frombuffer((IntPtr)pfo, (long)(rows * cols), np.@bool);
                flatOut.fill(false);
                np.equal(flatText, flatText, flatOut, flatMask);
                flatOut.ToArray<bool>().Should().Equal(flatMask.ToArray<bool>());
            }
        }
    }
}
