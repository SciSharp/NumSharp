using System;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     The 2-D block kernel's stationary-out mode (DirectILKernelGenerator.InnerLoop2D.Stationary.cs): a reduction's
    ///     rows <c>{o, x, o}</c> — <c>o[i] = o[i] op x[r][i]</c> for every row r, the out at outer stride 0 — with the out
    ///     row held in registers across the rows. Each call is pinned against the PLAIN 2-D kernel driving the same call
    ///     (the out loaded and stored on every row), byte for byte, so a strip that dropped a row, read a column of the
    ///     wrong row, mis-reversed a reversed row, used the wrong tail mask or mixed the vector and scalar bodies fails.
    /// </summary>
    /// <remarks>
    ///     The reduction's rows form reaches the variant only for float64 / float32 add (<c>NumPyAddLoops</c>), so the
    ///     tests call the kernels directly: every SIMD dtype, the binary ops whose bodies the kernels compose (add with
    ///     NumPy's NaN operand order, subtract, multiply, xor), row widths around every vector group (the 8-, 4- and
    ///     1-vector strips, the masked and the scalar tails, the 4- and 1-column scalar strips), contiguous, reversed and
    ///     broadcast rows, rows running downward in memory, reductions over several row blocks (the partial last block, the
    ///     next-block prefetch), NaN payloads, buffers against inaccessible pages, and the calls the mode does not take.
    /// </remarks>
    [TestClass]
    public unsafe class StationaryOutKernelTests
    {
        /// <summary>The ten numeric SIMD dtypes (every lane width, signed and unsigned, both floats).</summary>
        private static readonly NPTypeCode[] SimdTypes =
        {
            NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32,
            NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Single, NPTypeCode.Double,
        };

        /// <summary>Row widths around every strip and tail of every lane width (4 to 32 lanes a vector).</summary>
        private static readonly long[] Widths =
            { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 127, 128, 129, 255, 256, 257, 263, 300 };

        /// <summary>The element size of a dtype in bytes.</summary>
        /// <param name="t">The dtype.</param>
        /// <returns>Its size.</returns>
        private static int Size(NPTypeCode t) => t switch
        {
            NPTypeCode.SByte or NPTypeCode.Byte => 1,
            NPTypeCode.Int16 or NPTypeCode.UInt16 => 2,
            NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Single => 4,
            _ => 8,
        };

        /// <summary>Writes <paramref name="v"/> as element <paramref name="i"/> of dtype <paramref name="t"/>.</summary>
        /// <param name="p">The buffer.</param>
        /// <param name="t">The dtype.</param>
        /// <param name="i">The element index.</param>
        /// <param name="v">The value (a whole number for the integer dtypes; it wraps).</param>
        private static void Write(byte* p, NPTypeCode t, long i, double v)
        {
            switch (t)
            {
                case NPTypeCode.SByte: ((sbyte*)p)[i] = unchecked((sbyte)(long)v); break;
                case NPTypeCode.Byte: p[i] = unchecked((byte)(long)v); break;
                case NPTypeCode.Int16: ((short*)p)[i] = unchecked((short)(long)v); break;
                case NPTypeCode.UInt16: ((ushort*)p)[i] = unchecked((ushort)(long)v); break;
                case NPTypeCode.Int32: ((int*)p)[i] = unchecked((int)(long)v); break;
                case NPTypeCode.UInt32: ((uint*)p)[i] = unchecked((uint)(long)v); break;
                case NPTypeCode.Int64: ((long*)p)[i] = (long)v; break;
                case NPTypeCode.UInt64: ((ulong*)p)[i] = unchecked((ulong)(long)v); break;
                case NPTypeCode.Single: ((float*)p)[i] = (float)v; break;
                default: ((double*)p)[i] = v; break;
            }
        }

        /// <summary>
        ///     Fills <paramref name="count"/> elements with random values: whole numbers in [-60, 60) (a float gets a
        ///     quarter on top), and for a float dtype a quiet NaN whose payload names the position at every
        ///     <paramref name="nanEvery"/>-th element (0: none).
        /// </summary>
        /// <param name="rnd">The calling test's seeded generator.</param>
        /// <param name="p">The buffer.</param>
        /// <param name="t">The dtype.</param>
        /// <param name="count">The element count.</param>
        /// <param name="nanEvery">The NaN period (0: no NaN).</param>
        /// <param name="salt">Mixed into the NaN payloads, so two operands' NaNs differ.</param>
        private static void Fill(System.Random rnd, byte* p, NPTypeCode t, long count, int nanEvery, int salt)
        {
            bool isFloat = t == NPTypeCode.Single || t == NPTypeCode.Double;
            for (long i = 0; i < count; i++)
            {
                if (isFloat && nanEvery > 0 && i % nanEvery == 0)
                {
                    if (t == NPTypeCode.Double)
                        ((long*)p)[i] = unchecked((long)0x7FF8000000000000UL) | ((long)salt << 24) | (i & 0xFFFFFF);
                    else
                        ((int*)p)[i] = 0x7FC00000 | (salt << 16) | (int)(i & 0xFFFF);
                    continue;
                }
                Write(p, t, i, rnd.Next(-60, 60) + (isFloat ? 0.25 : 0));
            }
        }

        /// <summary>The plain kernel, the stationary-out variant and the reversed-run variant of one binary op over one dtype.</summary>
        /// <param name="op">The binary op.</param>
        /// <param name="t">The dtype (both inputs and the output).</param>
        /// <returns>(plain, reversed-run variant, stationary-out variant).</returns>
        private static (ND2DElementwiseKernel Plain, ND2DElementwiseKernel Reversed, ND2DElementwiseKernel Stationary) Kernels(BinaryOp op, NPTypeCode t)
        {
            var types = new[] { t, t, t };
            var key = InnerLoopKernelKey.Binary(op, t, t, t);
            Action<ILGenerator> scalar = il => DirectILKernelGenerator.EmitScalarOperation(il, op, t, NaNOperandOrder.NumPyUfunc);
            Action<ILGenerator> vector = il => DirectILKernelGenerator.EmitVectorOperation(il, op, t, NaNOperandOrder.NumPyUfunc);
            return (DirectILKernelGenerator.Compile2DElementwiseKernel(types, scalar, vector, key),
                    DirectILKernelGenerator.Compile2DReversedRunKernel(types, scalar, vector, key),
                    DirectILKernelGenerator.Compile2DStationaryOutKernel(types, scalar, vector, key));
        }

        /// <summary>
        ///     One reduction call <c>{o, x, o}</c> over <paramref name="rows"/> rows of <paramref name="n"/>: operand 1 a
        ///     contiguous row of a matrix ("C"), a reversed one ("R": each row read from its last element down), one row
        ///     broadcast down the rows ("BR"), or the matrix's rows taken from the last one up — <c>x[::-1, :]</c> ("RO": a
        ///     negative row stride, as <c>np.sum(m[::-1], axis=0)</c> hands the reduction) and <c>x[::-1, ::-1]</c> ("RR").
        /// </summary>
        /// <param name="k">The kernel.</param>
        /// <param name="o">The out row (also operand 0).</param>
        /// <param name="x">The matrix's first element in memory (or the broadcast row).</param>
        /// <param name="e">The element size.</param>
        /// <param name="rows">The row count.</param>
        /// <param name="n">The row width.</param>
        /// <param name="mode">"C", "R", "BR", "RO" or "RR".</param>
        /// <exception cref="ArgumentException">An unknown <paramref name="mode"/>.</exception>
        private static void Reduce(ND2DElementwiseKernel k, byte* o, byte* x, int e, long rows, long n, string mode)
        {
            void** ptrs = stackalloc void*[3];
            long* inner = stackalloc long[3];
            long* outer = stackalloc long[3];
            ptrs[0] = o; ptrs[2] = o;
            inner[0] = e; inner[2] = e;
            outer[0] = 0; outer[2] = 0;
            switch (mode)
            {
                case "C":
                case "BR":
                    ptrs[1] = x;
                    inner[1] = e;
                    outer[1] = mode == "BR" ? 0 : n * e;
                    break;
                case "R":
                    ptrs[1] = x + (n - 1) * e;
                    inner[1] = -e;
                    outer[1] = n * e;
                    break;
                case "RO":
                    // The last row first: its first element, then each earlier row (the row stride negative).
                    ptrs[1] = x + (rows - 1) * n * e;
                    inner[1] = e;
                    outer[1] = -n * e;
                    break;
                case "RR":
                    // The matrix's last element first: every row read from its end down, the rows from the last up.
                    ptrs[1] = x + (rows * n - 1) * e;
                    inner[1] = -e;
                    outer[1] = -n * e;
                    break;
                default:
                    throw new ArgumentException($"Unknown row mode '{mode}'.", nameof(mode));
            }
            k(ptrs, inner, n, outer, rows);
        }

        /// <summary>
        ///     The rows one block of the stationary-out loops holds for rows of <paramref name="n"/> elements of
        ///     <paramref name="e"/> bytes (the generator's own rule), so a test can run two full blocks and a partial one
        ///     whatever the tuning.
        /// </summary>
        /// <param name="n">The row width.</param>
        /// <param name="e">The element size.</param>
        /// <returns>The block's row count.</returns>
        private static long BlockRows(long n, int e)
            => System.Math.Max(DirectILKernelGenerator.StationaryBlockMinRows,
                        DirectILKernelGenerator.StationaryBlockBytes / System.Math.Max(1, n * e));

        /// <summary>
        ///     Runs one reduction through the plain route (the plain kernel, or the reversed-run variant for reversed rows)
        ///     and the stationary-out variant from the same out, and asserts the same bytes.
        /// </summary>
        /// <param name="op">The op.</param>
        /// <param name="t">The dtype.</param>
        /// <param name="o0">The starting out row.</param>
        /// <param name="x">Operand 1's data.</param>
        /// <param name="rows">The row count.</param>
        /// <param name="n">The row width.</param>
        /// <param name="mode">"C", "R", "BR", "RO" or "RR" (see <see cref="Reduce"/>).</param>
        /// <param name="o1">Scratch for the plain route's out (n elements).</param>
        /// <param name="o2">Scratch for the stationary route's out (n elements).</param>
        /// <param name="what">The case a failure names.</param>
        private static void AssertSameAsPlain(BinaryOp op, NPTypeCode t, byte* o0, byte* x, long rows, long n, string mode,
            byte* o1, byte* o2, string what)
        {
            int e = Size(t);
            var (plain, reversed, stationary) = Kernels(op, t);
            Buffer.MemoryCopy(o0, o1, n * e, n * e);
            Buffer.MemoryCopy(o0, o2, n * e, n * e);
            Reduce(mode is "R" or "RR" ? reversed : plain, o1, x, e, rows, n, mode);
            Reduce(stationary, o2, x, e, rows, n, mode);
            var a = new ReadOnlySpan<byte>(o1, (int)(n * e));
            var b = new ReadOnlySpan<byte>(o2, (int)(n * e));
            if (!a.SequenceEqual(b))
            {
                int k = 0;
                while (a[k] == b[k]) k++;
                Assert.Fail($"{what}: element {k / e} differs from the plain kernel's");
            }
        }

        /// <summary>
        ///     Every SIMD dtype x add / subtract / multiply / xor x contiguous, reversed and broadcast rows x widths around
        ///     every strip and tail x 1 to 33 rows, against the plain kernel.
        /// </summary>
        [TestMethod]
        public void EveryDtypeOpAndWidth_MatchesThePlainKernel()
        {
            var rnd = new System.Random(29);
            foreach (var t in SimdTypes)
            {
                int e = Size(t);
                bool isFloat = t == NPTypeCode.Single || t == NPTypeCode.Double;
                var ops = isFloat
                    ? new[] { BinaryOp.Add, BinaryOp.Subtract, BinaryOp.Multiply }
                    : new[] { BinaryOp.Add, BinaryOp.Subtract, BinaryOp.Multiply, BinaryOp.BitwiseXor };
                foreach (long n in Widths)
                foreach (long rows in new long[] { 1, 2, 7, 33 })
                {
                    byte* x = (byte*)NativeMemory.Alloc((nuint)(rows * n * e));
                    byte* o0 = (byte*)NativeMemory.Alloc((nuint)(n * e));
                    byte* o1 = (byte*)NativeMemory.Alloc((nuint)(n * e));
                    byte* o2 = (byte*)NativeMemory.Alloc((nuint)(n * e));
                    try
                    {
                        Fill(rnd, x, t, rows * n, 0, 1);
                        Fill(rnd, o0, t, n, 0, 2);
                        foreach (var op in ops)
                        foreach (var mode in new[] { "C", "R", "BR" })
                            AssertSameAsPlain(op, t, o0, x, rows, n, mode, o1, o2, $"{op} {t} {mode} ({rows}, {n})");
                    }
                    finally
                    {
                        NativeMemory.Free(x); NativeMemory.Free(o0); NativeMemory.Free(o1); NativeMemory.Free(o2);
                    }
                }
            }
        }

        /// <summary>
        ///     Float add and multiply over NaNs with distinct payloads in both the out and the rows: each lane keeps the
        ///     NaN the plain kernel keeps (NumPy's operand order, in the op's bodies).
        /// </summary>
        [TestMethod]
        public void NaNPayloads_KeepThePlainKernelsBytes()
        {
            var rnd = new System.Random(31);
            foreach (var t in new[] { NPTypeCode.Single, NPTypeCode.Double })
            foreach (long n in new long[] { 3, 8, 13, 33, 70, 129 })
            foreach (long rows in new long[] { 1, 5, 16 })
            {
                int e = Size(t);
                byte* x = (byte*)NativeMemory.Alloc((nuint)(rows * n * e));
                byte* o0 = (byte*)NativeMemory.Alloc((nuint)(n * e));
                byte* o1 = (byte*)NativeMemory.Alloc((nuint)(n * e));
                byte* o2 = (byte*)NativeMemory.Alloc((nuint)(n * e));
                try
                {
                    Fill(rnd, x, t, rows * n, 3, 1);
                    Fill(rnd, o0, t, n, 2, 2);
                    foreach (var op in new[] { BinaryOp.Add, BinaryOp.Multiply })
                    foreach (var mode in new[] { "C", "R", "BR" })
                        AssertSameAsPlain(op, t, o0, x, rows, n, mode, o1, o2, $"{op} {t} {mode} ({rows}, {n}) NaN payloads");
                }
                finally
                {
                    NativeMemory.Free(x); NativeMemory.Free(o0); NativeMemory.Free(o1); NativeMemory.Free(o2);
                }
            }
        }

        /// <summary>
        ///     The out row and the matrix against inaccessible pages (the out at the low fence and at the high one, the
        ///     matrix's first row starting at the low fence): no strip, masked tail (forward or reversed mask) or scalar
        ///     strip reads or writes outside them. A violation ends the test host.
        /// </summary>
        [TestMethod]
        public void FencedBuffers_ReadAndWriteNothingOutsideTheOperands()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("Fenced memory needs VirtualAlloc (Windows).");

            using var fenceX = new FencedMemory(pages: 8);
            using var fenceO = new FencedMemory(pages: 2);
            var rnd = new System.Random(37);
            foreach (var t in SimdTypes)
            foreach (long n in new long[] { 1, 3, 5, 9, 17, 33, 65, 129, 263 })
            foreach (bool atEnd in new[] { false, true })
            {
                int e = Size(t);
                const long rows = 6;
                byte* x = fenceX.Place(rows * n * e, atEnd: false);
                byte* o2 = fenceO.Place(n * e, atEnd);
                byte* o0 = (byte*)NativeMemory.Alloc((nuint)(n * e));
                byte* o1 = (byte*)NativeMemory.Alloc((nuint)(n * e));
                try
                {
                    Fill(rnd, x, t, rows * n, 0, 1);
                    Fill(rnd, o0, t, n, 0, 2);
                    foreach (var mode in new[] { "C", "R", "BR" })
                        AssertSameAsPlain(BinaryOp.Add, t, o0, x, rows, n, mode, o1, o2, $"Add {t} {mode} ({rows}, {n}) fenced atEnd={atEnd}");
                }
                finally
                {
                    NativeMemory.Free(o0); NativeMemory.Free(o1);
                }
            }
        }

        /// <summary>
        ///     Reductions over two full row blocks and a partial one, at each dtype's block size for the width: the blocks
        ///     in row order (each lane still its rows in order), the out strip carried from block to block, the partial
        ///     last block, the next-block prefetch of compact blocks (x86; it changes no byte, so this proves the loops
        ///     around it), and rows running downward in memory (<c>x[::-1, :]</c>, <c>x[::-1, ::-1]</c>), with NaN payloads
        ///     in the float rows and the out, against the plain kernel.
        /// </summary>
        [TestMethod]
        public void SeveralRowBlocks_MatchThePlainKernel()
        {
            var rnd = new System.Random(43);
            foreach (var t in SimdTypes)
            foreach (long n in new long[] { 1, 8, 33, 100, 129 })
            {
                int e = Size(t);
                bool isFloat = t == NPTypeCode.Single || t == NPTypeCode.Double;
                long rows = 2 * BlockRows(n, e) + 7;
                byte* x = (byte*)NativeMemory.Alloc((nuint)(rows * n * e));
                byte* o0 = (byte*)NativeMemory.Alloc((nuint)(n * e));
                byte* o1 = (byte*)NativeMemory.Alloc((nuint)(n * e));
                byte* o2 = (byte*)NativeMemory.Alloc((nuint)(n * e));
                try
                {
                    Fill(rnd, x, t, rows * n, isFloat ? 7 : 0, 1);
                    Fill(rnd, o0, t, n, isFloat ? 3 : 0, 2);
                    foreach (var op in new[] { BinaryOp.Add, BinaryOp.Multiply })
                    foreach (var mode in new[] { "C", "R", "BR", "RO", "RR" })
                        AssertSameAsPlain(op, t, o0, x, rows, n, mode, o1, o2, $"{op} {t} {mode} ({rows}, {n}) several blocks");
                }
                finally
                {
                    NativeMemory.Free(x); NativeMemory.Free(o0); NativeMemory.Free(o1); NativeMemory.Free(o2);
                }
            }
        }

        /// <summary>
        ///     <see cref="SeveralRowBlocks_MatchThePlainKernel"/>'s reductions with the matrix against inaccessible pages
        ///     — starting at the low fence and ending at the high one — and the out at either fence: no block, partial last
        ///     block, strip or tail reads or writes outside the operands, whichever way the rows run. A violation ends the
        ///     test host.
        /// </summary>
        [TestMethod]
        public void SeveralRowBlocks_FencedBuffers_ReadAndWriteNothingOutsideTheOperands()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("Fenced memory needs VirtualAlloc (Windows).");

            // The largest matrix below: 2 · 32 + 7 rows of 129 eight-byte elements (73,272 bytes), or 2 · 16384 + 7 rows
            // of one byte; 24 pages hold either.
            using var fenceX = new FencedMemory(pages: 24);
            using var fenceO = new FencedMemory(pages: 2);
            var rnd = new System.Random(47);
            foreach (var t in SimdTypes)
            foreach (long n in new long[] { 1, 9, 100, 129 })
            foreach (bool atEnd in new[] { false, true })
            {
                int e = Size(t);
                long rows = 2 * BlockRows(n, e) + 7;
                byte* x = fenceX.Place(rows * n * e, atEnd);
                byte* o2 = fenceO.Place(n * e, !atEnd);
                byte* o0 = (byte*)NativeMemory.Alloc((nuint)(n * e));
                byte* o1 = (byte*)NativeMemory.Alloc((nuint)(n * e));
                try
                {
                    Fill(rnd, x, t, rows * n, 0, 1);
                    Fill(rnd, o0, t, n, 0, 2);
                    foreach (var mode in new[] { "C", "R", "BR", "RO", "RR" })
                        AssertSameAsPlain(BinaryOp.Add, t, o0, x, rows, n, mode, o1, o2,
                            $"Add {t} {mode} ({rows}, {n}) several blocks fenced atEnd={atEnd}");
                }
                finally
                {
                    NativeMemory.Free(o0); NativeMemory.Free(o1);
                }
            }
        }

        /// <summary>
        ///     Calls the mode does not take — a separate output, an out that moves along the rows, a stepped row — run the
        ///     variant's ordinary dispatch and give the plain kernel's bytes.
        /// </summary>
        [TestMethod]
        public void CallsOutsideTheMode_RunTheOrdinaryDispatch()
        {
            var rnd = new System.Random(41);
            void** ptrs = stackalloc void*[3];
            long* inner = stackalloc long[3];
            long* outer = stackalloc long[3];
            foreach (var t in new[] { NPTypeCode.Int16, NPTypeCode.Int32, NPTypeCode.Double })
            foreach (long n in new long[] { 3, 8, 17, 40 })
            {
                int e = Size(t);
                const long rows = 4;
                var (plain, _, stationary) = Kernels(BinaryOp.Add, t);
                byte* a = (byte*)NativeMemory.Alloc((nuint)(rows * n * e * 2));
                byte* b = (byte*)NativeMemory.Alloc((nuint)(rows * n * e * 2));
                byte* c1 = (byte*)NativeMemory.Alloc((nuint)(rows * n * e));
                byte* c2 = (byte*)NativeMemory.Alloc((nuint)(rows * n * e));
                try
                {
                    Fill(rnd, a, t, rows * n * 2, 0, 1);
                    Fill(rnd, b, t, rows * n * 2, 0, 2);
                    // (1) a + b -> c, every operand a full (rows, n) block; (2) a stepped second operand (inner stride 2e).
                    foreach (int step in new[] { 1, 2 })
                    {
                        foreach (var (k, c) in new[] { (plain, (IntPtr)c1), (stationary, (IntPtr)c2) })
                        {
                            ptrs[0] = a; ptrs[1] = b; ptrs[2] = (void*)c;
                            inner[0] = e; inner[1] = e * step; inner[2] = e;
                            outer[0] = n * e; outer[1] = n * e * step; outer[2] = n * e;
                            k(ptrs, inner, n, outer, rows);
                        }
                        new ReadOnlySpan<byte>(c1, (int)(rows * n * e)).SequenceEqual(new ReadOnlySpan<byte>(c2, (int)(rows * n * e)))
                            .Should().BeTrue($"{t} n={n} step={step}: the variant's ordinary dispatch gives the plain kernel's bytes");
                    }
                }
                finally
                {
                    NativeMemory.Free(a); NativeMemory.Free(b); NativeMemory.Free(c1); NativeMemory.Free(c2);
                }
            }
        }
    }
}
