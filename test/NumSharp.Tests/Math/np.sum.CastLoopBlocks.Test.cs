using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     An AXIS <c>np.sum(a, axis, dtype=X)</c> in a cast loop reduces a cast copy of the operand (NumPy casts every
    ///     element before its add loop sees it). Large operands are cast and reduced in blocks along a non-reduced axis
    ///     (<c>DefaultEngine.TryReduceAddInCastLoopBlocked</c>) instead of one full copy; these tests pin that the blocked
    ///     route returns the whole-copy route's result BIT FOR BIT — every output's elements, order and kernel are the
    ///     same — across dtype pairs, layouts, axes and keepdims.
    /// </summary>
    /// <remarks>
    ///     The block size is lowered to 64 elements so small operands split into many blocks (some of a single index),
    ///     and the values span ~60 binades, so a changed summation order changes the bits.
    /// </remarks>
    [TestClass]
    public class np_sum_cast_loop_block_tests
    {
        /// <summary>
        ///     Random operand values in <paramref name="dtype"/>: float16 within ±100, int32 / int64 within ±2^30, every
        ///     other dtype (complex: real and a permuted imaginary part) <c>(u − 0.5)·2^e</c> with <c>e</c> in
        ///     [-30, 30) — about 60 binades, so a changed float summation order changes the bits.
        /// </summary>
        /// <param name="n">Element count.</param>
        /// <param name="dtype">The operand dtype.</param>
        /// <param name="seed">The random seed (each case its own).</param>
        /// <returns>A fresh 1-D array.</returns>
        private static NDArray Values(int n, DType dtype, int seed)
        {
            var rng = new System.Random(seed);
            var v = new double[n];
            for (int i = 0; i < n; i++)
                v[i] = dtype.GetTypeCode() switch
                {
                    NPTypeCode.Half => (rng.NextDouble() - 0.5) * 200,
                    NPTypeCode.Int32 or NPTypeCode.Int64 => rng.Next(-1 << 30, 1 << 30),
                    _ => System.Math.ScaleB(rng.NextDouble() - 0.5, rng.Next(-30, 30)),
                };
            if (dtype.GetTypeCode() == NPTypeCode.Complex)
            {
                var c = new System.Numerics.Complex[n];
                for (int i = 0; i < n; i++) c[i] = new System.Numerics.Complex(v[i], v[(i * 7 + 3) % n]);
                return np.array(c);
            }
            return np.array(v).astype(dtype);
        }

        /// <summary>
        ///     The blocked axis cast loop returns the whole-copy route's bits for every dtype pair, layout, axis and
        ///     keepdims, at block sizes of one or two indices, a few indices and about a third of the operand.
        /// </summary>
        [TestMethod]
        public void AxisSum_CastLoop_BlockedEqualsTheWholeCastCopy()
        {
            var pairs = new (DType Src, DType Loop)[]
            {
                (np.float32, np.float64), (np.float16, np.float32), (np.float64, np.float32), (np.float16, np.float64),
                (np.int32, np.float64), (np.int64, np.float32), (np.complex128, np.float64), (np.float64, np.complex128),
            };
            var layouts = new (string Name, int N, Func<NDArray, NDArray> View)[]
            {
                ("C(30,40)", 1200, b => b.reshape(30, 40)),
                ("F(30,40)", 1200, b => b.reshape(40, 30).T),
                ("strided cols", 2400, b => b.reshape(30, 80)[":, ::2"]),
                ("reversed", 1200, b => b.reshape(30, 40)["::-1, ::-1"]),
                ("C(6,10,20)", 1200, b => b.reshape(6, 10, 20)),
                ("transposed 3-D", 1200, b => b.reshape(6, 10, 20).transpose(new[] { 2, 0, 1 })),
                ("F(6,10,20)", 1200, b => b.reshape(20, 10, 6).T),
                ("odd C(31,17)", 527, b => b.reshape(31, 17)),
                ("odd F(31,17)", 527, b => b.reshape(17, 31).T),
                ("strided 3-D", 2400, b => b.reshape(6, 10, 40)[":, :, ::2"]),
                ("offset rows", 1271, b => b.reshape(41, 31)["1:, 1:"]),
            };

            var fails = new List<string>();
            long saved = DefaultEngine.CastLoopBlockElements;
            try
            {
                int seed = 1;
                foreach (var (src, loop) in pairs)
                    foreach (var (name, n, view) in layouts)
                    {
                        var a = view(Values(n, src, seed++));
                        for (int axis = 0; axis < a.ndim; axis++)
                            foreach (bool keepdims in new[] { false, true })
                            {
                                DefaultEngine.CastLoopBlockElements = long.MaxValue;
                                var whole = np.sum(a, axis, keepdims, loop);
                                string w = $"{whole.dtype.name} ({string.Join(",", whole.shape)}) {Convert.ToHexString(whole.tobytes())}";
                                // Several block sizes: one-or-two-index blocks, a few indices, about a third of the operand.
                                foreach (long block in new long[] { 1, 64, 200 })
                                {
                                    DefaultEngine.CastLoopBlockElements = block;
                                    var blocked = np.sum(a, axis, keepdims, loop);
                                    string b = $"{blocked.dtype.name} ({string.Join(",", blocked.shape)}) {Convert.ToHexString(blocked.tobytes())}";
                                    if (w != b)
                                        fails.Add($"{src.name}->{loop.name} {name} axis={axis} keepdims={keepdims} block={block}");
                                }
                            }
                    }
            }
            finally
            {
                DefaultEngine.CastLoopBlockElements = saved;
            }

            string.Join(System.Environment.NewLine, fails).Should().BeEmpty();
        }

        /// <summary>
        ///     The FLAT <c>np.sum(a, dtype=X)</c> in an integer or bool cast loop, cut into blocks along the operand's
        ///     outermost axis in memory (<c>DefaultEngine.TryReduceAddInCastLoopFlatBlocked</c>), returns the whole-copy
        ///     route's result for every dtype pair (truncating float casts, wrapping integer casts, complex → integer
        ///     keeping the real part, X → bool), layout, keepdims and block size — the loop is order-free (modular
        ///     addition, logical OR), so any partition is NumPy's answer — and the blocked route really runs
        ///     (<c>DefaultEngine.CastLoopFlatBlockedRuns</c>), or the equality would hold vacuously.
        /// </summary>
        [TestMethod]
        public void FlatSum_OrderFreeCastLoop_BlockedEqualsTheWholeCastCopy()
        {
            var pairs = new (DType Src, DType Loop)[]
            {
                (np.float64, np.int32), (np.float64, np.uint8), (np.int64, np.int8), (np.float32, np.int16),
                (np.complex128, np.int64), (np.float64, np.@bool), (np.int32, np.uint16), (np.uint64, np.int32),
            };
            var layouts = new (string Name, int N, Func<NDArray, NDArray> View)[]
            {
                ("1-D", 1200, b => b),
                ("1-D reversed", 1200, b => b["::-1"]),
                ("C(30,40)", 1200, b => b.reshape(30, 40)),
                ("F(30,40)", 1200, b => b.reshape(40, 30).T),
                ("strided cols", 2400, b => b.reshape(30, 80)[":, ::2"]),
                ("transposed 3-D", 1200, b => b.reshape(6, 10, 20).transpose(new[] { 2, 0, 1 })),
                ("offset rows", 1271, b => b.reshape(41, 31)["1:, 1:"]),
            };

            var fails = new List<string>();
            long saved = DefaultEngine.CastLoopBlockElements;
            try
            {
                int seed = 100;
                foreach (var (src, loop) in pairs)
                    foreach (var (name, n, view) in layouts)
                    {
                        // Integer-valued with fractions (a float cast truncates), negatives (an unsigned cast wraps) and
                        // zeros (the bool cast's False).
                        var rng = new System.Random(seed++);
                        var v = new double[n];
                        for (int i = 0; i < n; i++)
                            v[i] = rng.Next(5) == 0 ? 0 : rng.Next(-200000, 200000) * 0.75;
                        var basis = src.GetTypeCode() == NPTypeCode.UInt64
                            ? np.abs(np.array(v)).astype(src)
                            : np.array(v).astype(src);
                        var a = view(basis);
                        foreach (bool keepdims in new[] { false, true })
                        {
                            DefaultEngine.CastLoopBlockElements = long.MaxValue;
                            var whole = np.sum(a, null, keepdims, loop);
                            string w = $"{whole.dtype.name} ({string.Join(",", whole.shape)}) {Convert.ToHexString(whole.tobytes())}";
                            foreach (long block in new long[] { 1, 64, 200 })
                            {
                                DefaultEngine.CastLoopBlockElements = block;
                                int runs = DefaultEngine.CastLoopFlatBlockedRuns;
                                var blocked = np.sum(a, null, keepdims, loop);
                                string b = $"{blocked.dtype.name} ({string.Join(",", blocked.shape)}) {Convert.ToHexString(blocked.tobytes())}";
                                if (w != b)
                                    fails.Add($"{src.name}->{loop.name} {name} keepdims={keepdims} block={block}: {b} vs {w}");
                                if (DefaultEngine.CastLoopFlatBlockedRuns != runs + 1)
                                    fails.Add($"{src.name}->{loop.name} {name} keepdims={keepdims} block={block}: the blocked route did not run");
                            }
                        }
                    }
            }
            finally
            {
                DefaultEngine.CastLoopBlockElements = saved;
            }

            string.Join(System.Environment.NewLine, fails).Should().BeEmpty();
        }

        /// <summary>
        ///     The order-sensitive operand of <see cref="AxisSum_CastLoop_IsNumPys"/>: <c>((i·7919) mod 1000 − 500)·2^((i·13) mod
        ///     61 − 30)</c> as float32 (exact; ~60 binades, so no float64 sum of it is order-free), or for float16
        ///     <c>((i·7919) mod 201 − 100)·2^((i·13) mod 7 − 3)</c>.
        /// </summary>
        /// <param name="n">Element count.</param>
        /// <param name="dtype">float32 or float16.</param>
        /// <returns>A fresh 1-D array.</returns>
        private static NDArray Ramp(int n, DType dtype)
        {
            var v = new double[n];
            bool half = dtype.GetTypeCode() == NPTypeCode.Half;
            for (long i = 0; i < n; i++)
                v[i] = half
                    ? System.Math.ScaleB((i * 7919) % 201 - 100, (int)((i * 13) % 7 - 3))
                    : System.Math.ScaleB((i * 7919) % 1000 - 500, (int)((i * 13) % 61 - 30));
            return np.array(v).astype(dtype);
        }

        /// <summary>
        ///     <c>np.sum(a, axis, dtype=X)</c> in a widening cast loop is NumPy's, bit for bit, on every layout and axis — the
        ///     cast-copy route (blocked for the large operand) where the reduced axis is the innermost one, the axis kernel's
        ///     own float32 → float64 path where it is not. Expectations: NumPy 2.4.2's result, as
        ///     <c>dtype (shape) sha256[:16]</c> of its C-order bytes.
        /// </summary>
        [TestMethod]
        public void AxisSum_CastLoop_IsNumPys()
        {
            var cases = new (string Name, Func<NDArray> Call, string Expected)[]
            {
            ("float32->float64 C(60,70) axis 0", () => np.sum(Ramp(4200, np.float32).reshape(60, 70), 0, np.float64), "float64 (70) 212f276597a9e005"),
            ("float32->float64 C(60,70) axis 1", () => np.sum(Ramp(4200, np.float32).reshape(60, 70), 1, np.float64), "float64 (60) c16afd5d024d98c8"),
            ("float32->float64 F(60,70) axis 0", () => np.sum(Ramp(4200, np.float32).reshape(70, 60).T, 0, np.float64), "float64 (70) 3386a883cd9de781"),
            ("float32->float64 F(60,70) axis 1", () => np.sum(Ramp(4200, np.float32).reshape(70, 60).T, 1, np.float64), "float64 (60) d06ed48ab300d4d7"),
            ("float32->float64 strided cols axis 0", () => np.sum(Ramp(8400, np.float32).reshape(60, 140)[":, ::2"], 0, np.float64), "float64 (70) e233e97b56611eeb"),
            ("float32->float64 strided cols axis 1", () => np.sum(Ramp(8400, np.float32).reshape(60, 140)[":, ::2"], 1, np.float64), "float64 (60) 41e667251a20f9c4"),
            ("float32->float64 reversed axis 0", () => np.sum(Ramp(4200, np.float32).reshape(60, 70)["::-1, ::-1"], 0, np.float64), "float64 (70) 0788b9c4308e008e"),
            ("float32->float64 reversed axis 1", () => np.sum(Ramp(4200, np.float32).reshape(60, 70)["::-1, ::-1"], 1, np.float64), "float64 (60) 46a24003b336d0a9"),
            ("float32->float64 C(5,30,40) axis 0", () => np.sum(Ramp(6000, np.float32).reshape(5, 30, 40), 0, np.float64), "float64 (30,40) f12e9818bc84ed7f"),
            ("float32->float64 C(5,30,40) axis 1", () => np.sum(Ramp(6000, np.float32).reshape(5, 30, 40), 1, np.float64), "float64 (5,40) c1f3400a85be94a1"),
            ("float32->float64 C(5,30,40) axis 2", () => np.sum(Ramp(6000, np.float32).reshape(5, 30, 40), 2, np.float64), "float64 (5,30) 280857f38b33c93c"),
            ("float32->float64 transposed 3-D axis 0", () => np.sum(Ramp(6000, np.float32).reshape(5, 30, 40).transpose(new[] { 2, 0, 1 }), 0, np.float64), "float64 (5,30) 280857f38b33c93c"),
            ("float32->float64 transposed 3-D axis 1", () => np.sum(Ramp(6000, np.float32).reshape(5, 30, 40).transpose(new[] { 2, 0, 1 }), 1, np.float64), "float64 (40,30) b98d7c72dd4930ed"),
            ("float32->float64 transposed 3-D axis 2", () => np.sum(Ramp(6000, np.float32).reshape(5, 30, 40).transpose(new[] { 2, 0, 1 }), 2, np.float64), "float64 (40,5) 65c86faf270b11b5"),
            ("float32->float64 big C(600,500) axis 0", () => np.sum(Ramp(300000, np.float32).reshape(600, 500), 0, np.float64), "float64 (500) 99866704f1395e4e"),
            ("float32->float64 big C(600,500) axis 1", () => np.sum(Ramp(300000, np.float32).reshape(600, 500), 1, np.float64), "float64 (600) 343d9475e2f13ac1"),
            ("float16->float32 C(60,70) axis 0", () => np.sum(Ramp(4200, np.float16).reshape(60, 70), 0, np.float32), "float32 (70) 59df204f08feeff5"),
            ("float16->float32 C(60,70) axis 1", () => np.sum(Ramp(4200, np.float16).reshape(60, 70), 1, np.float32), "float32 (60) 4c5e7103f9d856af"),
            ("float16->float32 F(60,70) axis 0", () => np.sum(Ramp(4200, np.float16).reshape(70, 60).T, 0, np.float32), "float32 (70) 0b3188f483735f55"),
            ("float16->float32 F(60,70) axis 1", () => np.sum(Ramp(4200, np.float16).reshape(70, 60).T, 1, np.float32), "float32 (60) 92c304b55c35635f"),
            ("float16->float32 strided cols axis 0", () => np.sum(Ramp(8400, np.float16).reshape(60, 140)[":, ::2"], 0, np.float32), "float32 (70) b8325488d14daf02"),
            ("float16->float32 strided cols axis 1", () => np.sum(Ramp(8400, np.float16).reshape(60, 140)[":, ::2"], 1, np.float32), "float32 (60) fb59055623ec01d0"),
            ("float16->float32 reversed axis 0", () => np.sum(Ramp(4200, np.float16).reshape(60, 70)["::-1, ::-1"], 0, np.float32), "float32 (70) 04b4f22de8b8d3f7"),
            ("float16->float32 reversed axis 1", () => np.sum(Ramp(4200, np.float16).reshape(60, 70)["::-1, ::-1"], 1, np.float32), "float32 (60) 6bdbea8d57e97276"),
            ("float16->float32 C(5,30,40) axis 0", () => np.sum(Ramp(6000, np.float16).reshape(5, 30, 40), 0, np.float32), "float32 (30,40) 75476d84c82c2dcc"),
            ("float16->float32 C(5,30,40) axis 1", () => np.sum(Ramp(6000, np.float16).reshape(5, 30, 40), 1, np.float32), "float32 (5,40) ec42ac2601883626"),
            ("float16->float32 C(5,30,40) axis 2", () => np.sum(Ramp(6000, np.float16).reshape(5, 30, 40), 2, np.float32), "float32 (5,30) 0050b32225cd1073"),
            ("float16->float32 transposed 3-D axis 0", () => np.sum(Ramp(6000, np.float16).reshape(5, 30, 40).transpose(new[] { 2, 0, 1 }), 0, np.float32), "float32 (5,30) 0050b32225cd1073"),
            ("float16->float32 transposed 3-D axis 1", () => np.sum(Ramp(6000, np.float16).reshape(5, 30, 40).transpose(new[] { 2, 0, 1 }), 1, np.float32), "float32 (40,30) 6f5ad371c73a94da"),
            ("float16->float32 transposed 3-D axis 2", () => np.sum(Ramp(6000, np.float16).reshape(5, 30, 40).transpose(new[] { 2, 0, 1 }), 2, np.float32), "float32 (40,5) 7665851cb5ad71de"),
            ("float16->float32 big C(600,500) axis 0", () => np.sum(Ramp(300000, np.float16).reshape(600, 500), 0, np.float32), "float32 (500) 4c4841ee2fed60e2"),
            ("float16->float32 big C(600,500) axis 1", () => np.sum(Ramp(300000, np.float16).reshape(600, 500), 1, np.float32), "float32 (600) 7875cf07ed06c708"),
            };

            var fails = new List<string>();
            foreach (var (name, call, expected) in cases)
            {
                var r = call();
                string hash = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(r.tobytes())).Substring(0, 16).ToLowerInvariant();
                string actual = $"{r.dtype.name} ({string.Join(",", r.shape)}) {hash}";
                if (actual != expected)
                    fails.Add($"{name}: expected [{expected}] but NumSharp returned [{actual}]");
            }

            string.Join(System.Environment.NewLine, fails).Should().BeEmpty();
        }
    }
}
