using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     The reversed-run stages of the element-wise kernels (DirectILKernelGenerator.InnerLoop.Reversed.cs): an operand
    ///     walked at inner byte stride −elemSize (<c>a[::-1]</c>, <c>m[:, ::-1]</c>) loads each vector with one contiguous
    ///     load and a lane reversal, and a reversed output stores that way, instead of the hardware gather or the scalar
    ///     strided loop. Each case is pinned against the SAME call with every reversed operand materialized contiguous,
    ///     compared byte for byte, so a stage that reversed the wrong lanes, read the wrong element, mis-stepped a row or
    ///     reversed a tail mask fails.
    /// </summary>
    /// <remarks>
    ///     The dtypes cover every SIMD lane width (1, 2, 4 and 8 bytes; signed, unsigned, float and bool) and the lengths
    ///     each vector group's edges (32 lanes of a byte, 16 of an int16, 8 and 4 above): one group, one element more and
    ///     one less, four groups (the unrolled loop) and the scalar tail after them. The fenced test puts every reversed
    ///     operand against an inaccessible page on both sides, so a load or store that leaves the view ends the test host.
    /// </remarks>
    [TestClass]
    public unsafe class ReversedRunStageTests
    {
        /// <summary>Every dtype whose same-dtype element-wise kernels have a vector body (Boolean rides byte lanes).</summary>
        private static readonly DType[] SimdDtypes =
        {
            np.@bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64,
        };

        /// <summary>
        ///     Lengths around every vector group: 0 and 1, both sides of one, two and four groups of each lane width, and
        ///     odd tails behind several unrolled iterations.
        /// </summary>
        private static readonly long[] Lengths = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 131, 263 };

        /// <summary>
        ///     Random values of <paramref name="dt"/>: whole numbers in [0, 100) (a float gets a quarter on top, bool takes
        ///     the parity), so every dtype holds them exactly and no integer op overflows.
        /// </summary>
        /// <param name="rnd">The calling test's seeded generator.</param>
        /// <param name="dt">The dtype.</param>
        /// <param name="shape">The shape (an empty array gives a 0-d array).</param>
        /// <returns>A fresh C-contiguous array.</returns>
        private static NDArray Data(System.Random rnd, DType dt, params long[] shape)
        {
            long n = shape.Aggregate(1L, (a, b) => a * b);
            var v = new double[n];
            bool isFloat = dt == np.float32 || dt == np.float64;
            bool isBool = dt == np.@bool;
            for (long i = 0; i < n; i++)
            {
                int w = rnd.Next(100);
                v[i] = isBool ? w & 1 : w + (isFloat ? 0.25 : 0);
            }
            return np.array(v).reshape(shape).astype(dt);
        }

        /// <summary>
        ///     The binary ufuncs a dtype sweeps: the arithmetic ones with an asymmetric one (subtract / divide) so a swapped
        ///     operand shows, and the bitwise ones for the integer and bool dtypes.
        /// </summary>
        /// <param name="dt">The dtype.</param>
        /// <returns>(name, ufunc) pairs.</returns>
        private static (string Name, Func<NDArray, NDArray, NDArray, NDArray> Op)[] BinaryOps(DType dt)
        {
            if (dt == np.@bool)
                return new (string, Func<NDArray, NDArray, NDArray, NDArray>)[]
                {
                    ("add", (a, b, o) => np.add(a, b, o)), ("multiply", (a, b, o) => np.multiply(a, b, o)),
                    ("bitwise_xor", (a, b, o) => np.bitwise_xor(a, b, o)), ("maximum", (a, b, o) => np.maximum(a, b, o)),
                };
            if (dt == np.float32 || dt == np.float64)
                return new (string, Func<NDArray, NDArray, NDArray, NDArray>)[]
                {
                    ("add", (a, b, o) => np.add(a, b, o)), ("subtract", (a, b, o) => np.subtract(a, b, o)),
                    ("multiply", (a, b, o) => np.multiply(a, b, o)), ("divide", (a, b, o) => np.divide(a, b, o)),
                    ("maximum", (a, b, o) => np.maximum(a, b, o)),
                };
            return new (string, Func<NDArray, NDArray, NDArray, NDArray>)[]
            {
                ("add", (a, b, o) => np.add(a, b, o)), ("subtract", (a, b, o) => np.subtract(a, b, o)),
                ("multiply", (a, b, o) => np.multiply(a, b, o)), ("bitwise_xor", (a, b, o) => np.bitwise_xor(a, b, o)),
                ("minimum", (a, b, o) => np.minimum(a, b, o)),
            };
        }

        /// <summary>
        ///     Asserts that two results have the same dtype, shape and BYTES (NaN payloads and the sign of zero included).
        /// </summary>
        /// <param name="got">The result of the route under test.</param>
        /// <param name="want">The result of the materialized reference.</param>
        /// <param name="what">The case a failure names.</param>
        /// <exception cref="AssertFailedException">The dtype, the shape or a byte differs.</exception>
        private static void SameBytes(NDArray got, NDArray want, string what)
        {
            got.dtype.Should().Be(want.dtype, what);
            got.shape.Should().Equal(want.shape, what);
            // np.ascontiguousarray returns a contiguous argument ITSELF: dispose only a copy it made, or a caller that
            // compares against the same `want` twice reads a buffer the pool already handed to someone else.
            var a = np.ascontiguousarray(got);
            var b = np.ascontiguousarray(want);
            try
            {
                var x = a.Unsafe.ReadOnlyBytes();
                var y = b.Unsafe.ReadOnlyBytes();
                if (!x.SequenceEqual(y))
                {
                    int k = 0;
                    while (k < x.Length && x[k] == y[k]) k++;
                    Assert.Fail($"{what}: byte {k} (element {k / System.Math.Max(1, (int)got.dtype.itemsize)}) differs from the "
                                + $"materialized reference: got bytes [{string.Join(",", x.ToArray().Take(16))}], want "
                                + $"[{string.Join(",", y.ToArray().Take(16))}]");
                }
            }
            finally
            {
                if (!ReferenceEquals(a, got)) a.Dispose();
                if (!ReferenceEquals(b, want)) b.Dispose();
            }
        }

        /// <summary>
        ///     One-dimensional calls of every pattern with a reversed operand — each input contiguous (C), reversed (R) or
        ///     a 0-d scalar (S), at least one R — through the 1-D shell's stages, for every SIMD dtype and length; then the
        ///     same calls into a REVERSED out= (the reversed-output stages), and with a contiguous out= around a reversed
        ///     input.
        /// </summary>
        [TestMethod]
        public void OneDimensional_EveryReversedPattern_MatchesMaterializedOperands()
        {
            var rnd = new System.Random(11);
            string[] forms = { "C", "R", "S" };
            foreach (var dt in SimdDtypes)
            foreach (long n in Lengths)
            {
                NDArray Operand(string form) => form switch
                {
                    "C" => Data(rnd, dt, n),
                    "R" => Data(rnd, dt, n)["::-1"],
                    _ => Data(rnd, dt),
                };
                foreach (var (name, op) in BinaryOps(dt))
                foreach (var fl in forms)
                foreach (var fr in forms)
                {
                    if (fl == "S" && fr == "S")
                        continue;
                    var lhs = Operand(fl);
                    var rhs = Operand(fr);
                    var want = op(np.ascontiguousarray(lhs), np.ascontiguousarray(rhs), null);
                    string what = $"{name}({fl}, {fr}) {dt} n={n}";

                    if (fl == "R" || fr == "R")
                        SameBytes(op(lhs, rhs, null), want, what);

                    // A reversed out= (and a contiguous one) around the same operands.
                    var outBase = np.zeros(new Shape(n), want.dtype);
                    op(lhs, rhs, outBase["::-1"]);
                    SameBytes(outBase["::-1"], want, what + " out=reversed");
                }
            }
        }

        /// <summary>
        ///     Unary ops over a reversed 1-D input (the reversed-run unary kernel, <c>DefaultEngine.TryReversedSimdUnaryOp</c>)
        ///     for every SIMD dtype and length, against the same op over the contiguous copy; float inputs also hold NaNs
        ///     with payloads, −0.0 and infinities, so a sign or payload the vector body and the scalar tail treat
        ///     differently would show if the kernel split its elements differently.
        /// </summary>
        [TestMethod]
        public void OneDimensional_UnaryOverReversedInput_MatchesMaterializedOperand()
        {
            var rnd = new System.Random(23);
            foreach (var dt in SimdDtypes)
            foreach (long n in Lengths)
            {
                var a = Data(rnd, dt, n);
                bool isFloat = dt == np.float32 || dt == np.float64;
                if (isFloat)
                {
                    double[] specials = { double.NaN, -0.0, double.PositiveInfinity, double.NegativeInfinity, -2.5 };
                    for (long i = 0; i < n; i += 5)
                    {
                        double v = specials[(int)(i / 5 % specials.Length)];
                        if (dt == np.float64)
                            a.SetDouble(double.IsNaN(v) ? BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8000000000000UL | i)) : v, i);
                        else
                            a.SetSingle(double.IsNaN(v) ? BitConverter.Int32BitsToSingle(unchecked((int)0xFFC00000) | (int)(i & 0xFFFF)) : (float)v, i);
                    }
                }
                var rev = a["::-1"];
                var copy = np.ascontiguousarray(rev);
                if (dt != np.@bool)
                {
                    SameBytes(np.negative(rev), np.negative(copy), $"negative {dt} n={n}");
                    SameBytes(np.square(rev), np.square(copy), $"square {dt} n={n}");
                }
                SameBytes(np.absolute(rev), np.absolute(copy), $"absolute {dt} n={n}");
                if (isFloat)
                {
                    SameBytes(np.sqrt(rev), np.sqrt(copy), $"sqrt {dt} n={n}");
                    SameBytes(np.floor(rev), np.floor(copy), $"floor {dt} n={n}");
                    SameBytes(np.reciprocal(rev), np.reciprocal(copy), $"reciprocal {dt} n={n}");
                }
            }
        }

        /// <summary>
        ///     Float add / subtract / multiply / divide whose operands hold NaNs with distinct payloads at the same
        ///     positions: a reversed operand keeps the NaN the contiguous route keeps (the kernels' NumPy operand order
        ///     runs per lane in the unchanged vector body), at every length.
        /// </summary>
        [TestMethod]
        public void OneDimensional_NaNPairs_KeepTheContiguousRoutesBytes()
        {
            foreach (var dt in new[] { np.float32, np.float64 })
            foreach (long n in Lengths)
            {
                NDArray Planted(int salt)
                {
                    var a = Data(new System.Random(salt), dt, n);
                    for (long i = 0; i < n; i += 3)
                    {
                        // Quiet NaNs whose payload names the operand and the position; every third element.
                        if (dt == np.float64)
                            a.SetDouble(BitConverter.Int64BitsToDouble(unchecked((long)0x7FF8000000000000UL | ((long)salt << 32) | i)), i);
                        else
                            a.SetSingle(BitConverter.Int32BitsToSingle(0x7FC00000 | (salt << 16) | (int)(i & 0xFFFF)), i);
                    }
                    return a;
                }

                foreach (var (name, op) in new (string, Func<NDArray, NDArray, NDArray>)[]
                         {
                             ("add", (a, b) => np.add(a, b)), ("subtract", (a, b) => np.subtract(a, b)),
                             ("multiply", (a, b) => np.multiply(a, b)), ("divide", (a, b) => np.divide(a, b)),
                         })
                foreach (var (fl, fr) in new[] { ("R", "C"), ("C", "R"), ("R", "R") })
                {
                    var lhs = fl == "R" ? Planted(1)["::-1"] : Planted(1);
                    var rhs = fr == "R" ? Planted(2)["::-1"] : Planted(2);
                    SameBytes(op(lhs, rhs), op(np.ascontiguousarray(lhs), np.ascontiguousarray(rhs)),
                        $"{name}({fl}, {fr}) {dt} n={n} NaN pairs");
                }
            }
        }

        /// <summary>
        ///     Two-dimensional calls through the 2-D block kernel's reversed-run stages: each input a full array (C), its
        ///     reversed columns (R), reversed rows and columns (RR), a reversed row broadcast down the rows (RW), a row
        ///     (CW), a column (SC: one value per row) or a 0-d scalar (S0), at least one of them reversed, over every row
        ///     width around the vector groups (the one-vector, masked and generic row shapes).
        /// </summary>
        [TestMethod]
        public void TwoDimensional_EveryReversedRowPattern_MatchesMaterializedOperands()
        {
            var rnd = new System.Random(13);
            var dtypes = new[] { np.@bool, np.uint8, np.int16, np.int32, np.float32, np.int64, np.float64 };
            string[] forms = { "C", "R", "RR", "RW", "CW", "SC", "S0" };
            const long rows = 3;
            foreach (var dt in dtypes)
            foreach (long w in new long[] { 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 32, 33, 37 })
            {
                NDArray Operand(string form) => form switch
                {
                    "C" => Data(rnd, dt, rows, w),
                    "R" => Data(rnd, dt, rows, w)[":, ::-1"],
                    "RR" => Data(rnd, dt, rows, w)["::-1, ::-1"],
                    "RW" => Data(rnd, dt, w)["::-1"],
                    "CW" => Data(rnd, dt, w),
                    "SC" => Data(rnd, dt, rows, 1),
                    _ => Data(rnd, dt),
                };
                // The asymmetric op of the dtype (bool has none: xor).
                Func<NDArray, NDArray, NDArray> op = dt == np.@bool
                    ? (a, b) => np.bitwise_xor(a, b)
                    : (a, b) => np.subtract(a, b);
                foreach (var fl in forms)
                foreach (var fr in forms)
                {
                    bool reversed = fl is "R" or "RR" or "RW" || fr is "R" or "RR" or "RW";
                    if (!reversed)
                        continue;
                    var lhs = Operand(fl);
                    var rhs = Operand(fr);
                    SameBytes(op(lhs, rhs), op(np.ascontiguousarray(lhs), np.ascontiguousarray(rhs)),
                        $"op({fl}, {fr}) {dt} ({rows}, {w})");
                }
            }
        }

        /// <summary>
        ///     Unary ops over reversed rows (the 2-D block kernel's unary R stage), a 3-D view whose leading axis does not
        ///     fold into the block (walked per block), and a reversed 2-D out= (the 1-D shell's reversed-output stage, row
        ///     by row).
        /// </summary>
        [TestMethod]
        public void TwoDimensional_UnaryLeadingAxesAndReversedOut_MatchMaterializedOperands()
        {
            var rnd = new System.Random(17);
            foreach (var dt in SimdDtypes)
            foreach (long w in new long[] { 1, 3, 4, 8, 9, 17, 33, 65 })
            {
                var m = Data(rnd, dt, 5, w);
                foreach (var view in new[] { ":, ::-1", "::-1, ::-1", "1:4, ::-1" })
                {
                    var v = m[view];
                    if (dt != np.@bool)
                        SameBytes(np.negative(v), np.negative(np.ascontiguousarray(v)), $"negative {dt} [{view}] w={w}");
                    if (dt == np.float32 || dt == np.float64)
                        SameBytes(np.sqrt(v), np.sqrt(np.ascontiguousarray(v)), $"sqrt {dt} [{view}] w={w}");
                    SameBytes(np.absolute(v), np.absolute(np.ascontiguousarray(v)), $"absolute {dt} [{view}] w={w}");
                }

                // A leading axis that does not fold into the block: x[::2, :, ::-1].
                var x = Data(rnd, dt, 6, 3, w)["::2, :, ::-1"];
                var y = Data(rnd, dt, 3, 3, w);
                SameBytes(np.add(x, y), np.add(np.ascontiguousarray(x), y), $"add 3-D {dt} w={w}");

                // A reversed 2-D out=: the 2-D gate refuses it (its output row must be contiguous), so each row is one
                // 1-D call with the output reversed.
                var a = Data(rnd, dt, 4, w);
                var b = Data(rnd, dt, 4, w);
                var outBase = np.zeros(new Shape(4, w), dt);
                np.add(a, b, outBase[":, ::-1"]);
                SameBytes(outBase[":, ::-1"], np.add(a, b), $"add out=[:, ::-1] {dt} w={w}");
            }
        }

        /// <summary>
        ///     Every reversed operand and reversed output against an inaccessible page: a 1-D view over a fenced buffer that
        ///     starts at its last element and ends at its first, in both directions of the fence, and 2-D reversed rows whose
        ///     first row starts at the low fence (a masked tail with the wrong lane mask would touch the page below it).
        ///     A load or a store outside the view ends the test host; the values are checked against the materialized call.
        /// </summary>
        [TestMethod]
        public void ReversedOperands_OfFencedBuffers_ReadAndWriteNothingOutsideTheView()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("Fenced memory needs VirtualAlloc (Windows).");

            using var fence = new FencedMemory(pages: 4);
            var rnd = new System.Random(19);
            foreach (var dt in SimdDtypes)
            foreach (long n in new long[] { 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 263 })
            foreach (bool atEnd in new[] { false, true })
            {
                long bytes = n * dt.itemsize;
                var other = Data(rnd, dt, n);

                // A reversed input over the fence: R + C and C + R.
                var fenced = np.frombuffer((IntPtr)fence.Place(bytes, atEnd), bytes, dt);
                np.copyto(fenced, Data(rnd, dt, n));
                var rev = fenced["::-1"];
                SameBytes(np.add(rev, other), np.add(np.ascontiguousarray(rev), other), $"add(R, C) {dt} n={n} atEnd={atEnd}");
                SameBytes(np.add(other, rev), np.add(other, np.ascontiguousarray(rev)), $"add(C, R) {dt} n={n} atEnd={atEnd}");
                SameBytes(np.absolute(rev), np.absolute(np.ascontiguousarray(rev)), $"absolute(R) {dt} n={n} atEnd={atEnd}");
                var want = np.add(other, np.ascontiguousarray(rev));

                // A reversed output over the fence.
                np.add(other, np.ascontiguousarray(rev), fenced["::-1"]);
                SameBytes(fenced["::-1"], want, $"add out=reversed {dt} n={n} atEnd={atEnd}");

                // Reversed rows: (rows, cols) over the fence, its first row at the low fence when !atEnd.
                foreach (long cols in new long[] { 1, 3, 5, 7, 9, 17 })
                {
                    long rows = n / cols;
                    if (rows < 1)
                        continue;
                    long rb = rows * cols * dt.itemsize;
                    var block = np.frombuffer((IntPtr)fence.Place(rb, atEnd), rb, dt).reshape(rows, cols);
                    np.copyto(block, Data(rnd, dt, rows, cols));
                    var rrow = block[":, ::-1"];
                    var full = Data(rnd, dt, rows, cols);
                    SameBytes(np.add(rrow, full), np.add(np.ascontiguousarray(rrow), full),
                        $"add([:, ::-1], C) {dt} ({rows}, {cols}) atEnd={atEnd}");
                }
            }
        }

        /// <summary>
        ///     The reversed-run variants are the kernels that run a reversed iteration (the speedup engaged): after a 1-D and
        ///     a 2-D reversed add, both variant caches hold the add's key — and the variants are separate kernels from the
        ///     plain ones every other iteration runs.
        /// </summary>
        [TestMethod]
        [DoNotParallelize] // reads the process-wide kernel caches another test may clear (GeneratedDelegates.ClearInnerLoop)
        public void ReversedIterations_RunTheReversedRunVariants()
        {
            if (!DirectILKernelGenerator.CanReverseLanes(NPTypeCode.Int32))
                Assert.Inconclusive("The host has no 256-bit AVX2 lane reversal: reversed operands keep the gather route.");

            var key = InnerLoopKernelKey.Binary(BinaryOp.Add, NPTypeCode.Int32, NPTypeCode.Int32, NPTypeCode.Int32);
            var a = np.arange(100).astype(np.int32);
            var m = np.arange(60).astype(np.int32).reshape(6, 10);

            np.add(a["::-1"], a).Should().NotBeNull();
            DirectILKernelGenerator.TryGetReversedRunInnerLoop(key, out var variant).Should().BeTrue("a 1-D reversed add runs the 1-D variant");

            np.add(m[":, ::-1"], m).Should().NotBeNull();
            DirectILKernelGenerator.TryGet2DReversedRunKernel(key, out _).Should().BeTrue("a 2-D reversed-row add runs the 2-D variant");

            np.add(a["::2"], a["::2"]).Should().NotBeNull();
            DirectILKernelGenerator.TryGetInnerLoop(key, out var plain).Should().BeTrue("a strided add runs the plain kernel");
            ReferenceEquals(plain, variant).Should().BeFalse("the plain kernel is not the variant");
        }
    }
}
