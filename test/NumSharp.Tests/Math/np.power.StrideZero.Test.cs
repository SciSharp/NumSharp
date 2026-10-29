using System;
using System.Collections.Generic;
using static NumSharp.Tests.Utilities.RawBitCases;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.power's float loops as NumPy 2.4.2's DOUBLE_power / FLOAT_power (plan
    ///     <c>docs/plans/api-parameter-parity.md</c> X13): when the exponent's STEP in NumPy's inner-loop call is 0, an
    ///     exponent of -1, 0, 0.5, 1 or 2 gives <c>1/x</c>, <c>1</c>, <c>sqrt(x)</c>, <c>x</c> or <c>x*x</c>; otherwise, and
    ///     always for float16 and np.float_power, the C library <c>pow</c>. Which calls see step 0 is NumPy's trivial-loop
    ///     and iterator decision, so the tests walk it: plain / out= / where= routes, one-element and broadcast exponents,
    ///     columns on both sides of the buffering threshold, the trivial loop's cast copy, dtype=.
    /// </summary>
    /// <remarks>
    ///     <para>Every expected string is NumPy 2.4.2's output for the same call (win-amd64, probed 2026-10-06), in
    ///     <see cref="NumSharp.Tests.Utilities.RawBitCases.Describe"/>'s format with raw NaN bits. The bases hold values on
    ///     which the fast path and pow DIFFER — <c>757.7617475757002 ** 2</c> (x*x <c>...03c1</c>, pow <c>...03c2</c>),
    ///     <c>-0.0 ** 0.5</c> (sqrt -0.0, pow +0.0), <c>-inf ** 0.5</c> (sqrt NaN, pow +inf), <c>72.21291942905373 ** -1</c>
    ///     (1/x <c>...483</c>, pow <c>...482</c>), <c>973.9663027747129 ** 0.5</c>, and a signalling NaN (x keeps it, pow
    ///     quiets it; 1 vs pow's NaN for exponent 0) — so each case shows which loop ran.</para>
    /// </remarks>
    [TestClass]
    public class PowerStrideZeroTests : TestClass
    {
        /// <summary>The float64 bases: one discriminating value per fast exponent, the last a signalling NaN.</summary>
        /// <returns>A fresh 6-element C-contiguous float64 array.</returns>
        /// <remarks>
        /// Built from raw bits through an integer array: a NaN written as a C# float constant
        /// (<c>BitConverter.UInt32BitsToSingle(0x7F800001)</c>) can be folded by the JIT, which keeps float constants as
        /// doubles, into a QUIET NaN — the signalling input this test needs would silently be gone.
        /// </remarks>
        private static NDArray F64Bases()
            => np.array(new ulong[]
            {
                0x4087AE180F1CEB84, 0x8000000000000000, 0xFFF0000000000000, 0x40520DA078D01E00, 0x408E6FBAFCF2FB68,
                0x7FF0000000000001,   // 757.7617475757002, -0.0, -inf, 72.21291942905373, 973.9663027747129, sNaN
            }).view(np.float64);

        /// <summary>The float32 bases, chosen the same way for FLOAT_power (built from raw bits, see <see cref="F64Bases"/>).</summary>
        /// <returns>A fresh 6-element C-contiguous float32 array.</returns>
        private static NDArray F32Bases()
            => np.array(new uint[]
            {
                0xC1163EC5, 0x80000000, 0xFF800000, 0x431BEAAF, 0x445F0313,
                0x7F800001,           // -9.390324592590332, -0.0, -inf, 155.91673278808594, 892.0480346679688, sNaN
            }).view(np.float32);

        /// <summary>
        ///     <c>x ** scalar</c>: the plain route takes NumPy's fast path for -1, 0, 0.5, 1 and 2 (a 0-d exponent has
        ///     step 0 on every route) and pow for 3; a full-shape exponent is pow for every value.
        /// </summary>
        [TestMethod]
        public void Plain_ZeroDExponent_TakesTheFastPath_AFullExponentDoesNot()
        {
            var fails = new List<string>();
            var x = F64Bases();
            Check(fails, "f64 ** 2", () => np.power(x, NDArray.Scalar(2.0)),
                  "float64 (6,) 412185f5bb7003c1 0000000000000000 7ff0000000000000 40b45eb4aae20983 412cf304b7c7116b 7ff8000000000001", true);
            Check(fails, "f64 ** 0.5", () => np.power(x, NDArray.Scalar(0.5)),
                  "float64 (6,) 403b870871b16d12 8000000000000000 fff8000000000000 4020fee21157aa51 403f355be0cfbd27 7ff8000000000001", true);
            Check(fails, "f64 ** -1", () => np.power(x, NDArray.Scalar(-1.0)),
                  "float64 (6,) 3f559f1f53a0e60b fff0000000000000 8000000000000000 3f8c5c4eb7d75483 3f50d26a7a48b676 7ff8000000000001", true);
            Check(fails, "f64 ** 1", () => np.power(x, NDArray.Scalar(1.0)),
                  "float64 (6,) 4087ae180f1ceb84 8000000000000000 fff0000000000000 40520da078d01e00 408e6fbafcf2fb68 7ff0000000000001", true);
            Check(fails, "f64 ** 0", () => np.power(x, NDArray.Scalar(0.0)),
                  "float64 (6,) 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000", true);
            Check(fails, "f64 ** 3 (pow)", () => np.power(x, NDArray.Scalar(3.0)),
                  "float64 (6,) 41b9ef3c67454375 8000000000000000 fff0000000000000 4116fbe47fe70f90 41cb88e88e091d79 7ff8000000000001", true);
            Check(fails, "f64 ** full 2 (pow)", () => np.power(x, np.full(new Shape(6), 2.0)),
                  "float64 (6,) 412185f5bb7003c2 0000000000000000 7ff0000000000000 40b45eb4aae20983 412cf304b7c7116b 7ff8000000000001", true);
            Check(fails, "f64 ** full 0.5 (pow)", () => np.power(x, np.full(new Shape(6), 0.5)),
                  "float64 (6,) 403b870871b16d12 0000000000000000 7ff0000000000000 4020fee21157aa51 403f355be0cfbd28 7ff8000000000001", true);
            Check(fails, "f64 ** full -1 (pow)", () => np.power(x, np.full(new Shape(6), -1.0)),
                  "float64 (6,) 3f559f1f53a0e60b fff0000000000000 8000000000000000 3f8c5c4eb7d75482 3f50d26a7a48b676 7ff8000000000001", true);
            Check(fails, "f64 ** full 1 (pow)", () => np.power(x, np.full(new Shape(6), 1.0)),
                  "float64 (6,) 4087ae180f1ceb84 8000000000000000 fff0000000000000 40520da078d01e00 408e6fbafcf2fb68 7ff8000000000001", true);
            Check(fails, "f64 ** full 0 (pow)", () => np.power(x, np.full(new Shape(6), 0.0)),
                  "float64 (6,) 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 7ff8000000000001", true);
            AssertNoMismatch(fails, "DOUBLE_power takes its fast path exactly when the exponent's step is 0");
        }

        /// <summary>The float32 twin: FLOAT_power's fast path (1/x divides in double and rounds, the same float).</summary>
        [TestMethod]
        public void Plain_Float32_ZeroDExponent_TakesTheFastPath_AFullExponentDoesNot()
        {
            var fails = new List<string>();
            var x = F32Bases();
            Check(fails, "f32 ** 2", () => np.power(x, NDArray.Scalar(2.0f)), "float32 (6,) 42b05b3d 00000000 7f800000 46bdec0e 4942465b 7fc00001", true);
            Check(fails, "f32 ** 0.5", () => np.power(x, NDArray.Scalar(0.5f)), "float32 (6,) ffc00000 80000000 ffc00000 4147c95e 41eeeff9 7fc00001", true);
            Check(fails, "f32 ** -1", () => np.power(x, NDArray.Scalar(-1.0f)), "float32 (6,) bdda18ca ff800000 80000000 3bd229d9 3a92ef0d 7fc00001", true);
            Check(fails, "f32 ** 1", () => np.power(x, NDArray.Scalar(1.0f)), "float32 (6,) c1163ec5 80000000 ff800000 431beaaf 445f0313 7f800001", true);
            Check(fails, "f32 ** 0", () => np.power(x, NDArray.Scalar(0.0f)), "float32 (6,) 3f800000 3f800000 3f800000 3f800000 3f800000 3f800000", true);
            Check(fails, "f32 ** full 2 (pow)", () => np.power(x, np.full(new Shape(6), 2.0f)), "float32 (6,) 42b05b3c 00000000 7f800000 46bdec0e 4942465b 7fc00001", true);
            Check(fails, "f32 ** full 0.5 (pow)", () => np.power(x, np.full(new Shape(6), 0.5f)), "float32 (6,) ffc00000 00000000 7f800000 4147c95e 41eeeff8 7fc00001", true);
            Check(fails, "f32 ** full -1 (pow)", () => np.power(x, np.full(new Shape(6), -1.0f)), "float32 (6,) bdda18ca ff800000 80000000 3bd229d8 3a92ef0d 7fc00001", true);
            Check(fails, "f32 ** full 1 (pow)", () => np.power(x, np.full(new Shape(6), 1.0f)), "float32 (6,) c1163ec5 80000000 ff800000 431beaaf 445f0313 7fc00001", true);
            Check(fails, "f32 ** full 0 (pow)", () => np.power(x, np.full(new Shape(6), 0.0f)), "float32 (6,) 3f800000 3f800000 3f800000 3f800000 3f800000 7fc00001", true);
            AssertNoMismatch(fails, "FLOAT_power takes its fast path exactly when the exponent's step is 0");
        }

        /// <summary>
        ///     <c>np.power(x, 2.0, out=o)</c> must be <c>x * x</c>, as the plain call is: NumPy runs the same inner loop
        ///     whether the result is allocated or given, so the out= route takes the fast path too. It used to call the
        ///     C runtime's pow there (757.7617475757002 ** 2 was <c>...03c2</c>); moved here from the OpenBugs pins.
        /// </summary>
        [TestMethod]
        public void Out_ScalarExponent_TakesNumPysFastPath()
        {
            var fails = new List<string>();
            var x = np.array(new[] { 757.7617475757002, -205.48057283278354 });
            var o = np.empty_like(x);
            Check(fails, "power f64 out= 2.0", () => { np.power(x, NDArray.Scalar(2.0), @out: o); return o; },
                  "float64 (2,) 412185f5bb7003c1 40e49dc8818783d1");
            var xf = np.array(new[] { -901.92664f });
            var of = np.empty_like(xf);
            Check(fails, "power f32 out= 2.0", () => { np.power(xf, NDArray.Scalar(2.0f), @out: of); return of; },
                  "float32 (1,) 494699fb");
            // A float64 loop into a float32 out: x*x in float64, then the out's cast (a signalling NaN is quieted on the way).
            var x6 = F64Bases();
            var o32 = np.zeros(new Shape(6), np.float32);
            Check(fails, "power f64 into f32 out= 2.0", () => { np.power(x6, NDArray.Scalar(2.0), @out: o32); return o32; },
                  "float32 (6,) 490c2fae 00000000 7f800000 45a2f5a5 49679826 7fc00000", true);
            AssertNoMismatch(fails, "NumPy's power takes its scalar-exponent fast path on the out= route too");
        }

        /// <summary>
        ///     <c>where=</c> runs NumPy's masked loop, one call per mask-true run, each with the exponent's step 0: the fast
        ///     path for the selected positions, the out's prior contents elsewhere.
        /// </summary>
        [TestMethod]
        public void Where_ScalarExponent_TakesTheFastPath_AndKeepsMaskedSlots()
        {
            var fails = new List<string>();
            var x = F64Bases();
            var o = np.full(new Shape(6), -1.0);
            var m = np.array(new[] { true, true, false, true, true, true });
            Check(fails, "power f64 ** 0.5 where=", () => { np.power(x, NDArray.Scalar(0.5), @out: o, where: m); return o; },
                  "float64 (6,) 403b870871b16d12 8000000000000000 bff0000000000000 4020fee21157aa51 403f355be0cfbd27 7ff8000000000001", true);
            AssertNoMismatch(fails, "the masked loop sees the 0-d exponent with step 0 in every run");
        }

        /// <summary>
        ///     The trivial loop gives a 1-D operand its own stride, so a one-element exponent of the base's own shape is
        ///     stepped through (pow): <c>(1,) ** (1,)</c> and <c>0-d ** (1,)</c> (a (1,) result, not 0-d) are pow, while
        ///     NpyIter broadcasts a one-element exponent against a longer base with step 0 (fast) — and a (1,1) exponent
        ///     makes the result (1, n). A same-shape stride-0 broadcast view is step 0 in the trivial loop (fast) unless it
        ///     needs a cast, which the trivial loop makes into a contiguous copy first (pow).
        /// </summary>
        [TestMethod]
        public void OneElementExponents_FollowNumPysTrivialLoopAndIterator()
        {
            var fails = new List<string>();
            var x = F64Bases();
            Check(fails, "(1,) ** (1,)", () => np.power(np.array(new[] { 757.7617475757002 }), np.array(new[] { 2.0 })),
                  "float64 (1,) 412185f5bb7003c2");
            Check(fails, "0-d ** (1,)", () => np.power(NDArray.Scalar(757.7617475757002), np.array(new[] { 2.0 })),
                  "float64 (1,) 412185f5bb7003c2");
            Check(fails, "(1,) -0.0 ** (1,) 0.5", () => np.power(np.array(new[] { -0.0 }), np.array(new[] { 0.5 })),
                  "float64 (1,) 0000000000000000");
            Check(fails, "(6,) ** (1,)", () => np.power(x, np.array(new[] { 2.0 })),
                  "float64 (6,) 412185f5bb7003c1 0000000000000000 7ff0000000000000 40b45eb4aae20983 412cf304b7c7116b 7ff8000000000001", true);
            Check(fails, "(6,) ** (1,1)", () => np.power(x, np.array(new[,] { { 0.5 } })),
                  "float64 (1,6) 403b870871b16d12 8000000000000000 fff8000000000000 4020fee21157aa51 403f355be0cfbd27 7ff8000000000001", true);
            Check(fails, "(1,) ** (1,1)", () => np.power(np.array(new[] { 757.7617475757002 }), np.array(new[,] { { 2.0 } })),
                  "float64 (1,1) 412185f5bb7003c1");
            Check(fails, "(6,) ** broadcast_to (6,)", () => np.power(x, np.broadcast_to(np.array(new[] { 2.0 }), new Shape(6))),
                  "float64 (6,) 412185f5bb7003c1 0000000000000000 7ff0000000000000 40b45eb4aae20983 412cf304b7c7116b 7ff8000000000001", true);
            Check(fails, "(6,) ** broadcast_to int64 (6,)", () => np.power(x, np.broadcast_to(np.array(new[] { 2L }), new Shape(6))),
                  "float64 (6,) 412185f5bb7003c2 0000000000000000 7ff0000000000000 40b45eb4aae20983 412cf304b7c7116b 7ff8000000000001", true);
            AssertNoMismatch(fails, "the exponent's step is the trivial loop's or NpyIter's");
        }

        /// <summary>
        ///     A column exponent reaches NumPy's loop with step 0 only when NpyIter does not buffer it, which its cost model
        ///     decides from the row length and the buffer size: rows of 3 are buffered (pow), rows of 4097 are not (each row
        ///     its own call: x*x for the first, sqrt for the second). At <c>np.setbufsize(256)</c> the threshold is 128;
        ///     the trivial loop's cast copy of a 1-D stride-0 exponent stops at 256 elements.
        /// </summary>
        [TestMethod]
        public void ColumnExponents_AndCasts_FollowNumPysBufferingDecision()
        {
            var fails = new List<string>();
            var xx = np.array(new[,] { { 757.7617475757002, -0.0, 3.0 }, { double.NegativeInfinity, 757.7617475757002, 5.0 } });
            var col = np.array(new[,] { { 2.0 }, { 0.5 } });
            Check(fails, "(2,3) ** (2,1) buffered", () => np.power(xx, col),
                  "float64 (2,3) 412185f5bb7003c2 0000000000000000 4022000000000000 7ff0000000000000 403b870871b16d12 4001e3779b97f4a8");

            var wide = Cycled(new[] { 757.7617475757002, -0.0, double.NegativeInfinity, 2.0 }, 2, 4097);
            var r = np.power(wide, col);
            Check(fails, "(2,4097) ** (2,1) head", () => r[":, :4"],
                  "float64 (2,4) 412185f5bb7003c1 0000000000000000 7ff0000000000000 4010000000000000 8000000000000000 fff8000000000000 3ff6a09e667f3bcd 403b870871b16d12", true);
            Check(fails, "(2,4097) ** (2,1) tail", () => r[":, -1:"], "float64 (2,1) 412185f5bb7003c1 8000000000000000");

            long previous = np.setbufsize(256);
            try
            {
                Check(fails, "bufsize 256: (2,128) ** (2,1) buffered",
                      () => np.power(Cycled(new[] { 757.7617475757002, -0.0, double.NegativeInfinity, 2.0 }, 2, 128), col)[":, :3"],
                      "float64 (2,3) 412185f5bb7003c2 0000000000000000 7ff0000000000000 403b870871b16d12 0000000000000000 7ff0000000000000");
                Check(fails, "bufsize 256: (2,129) ** (2,1) per row",
                      () => np.power(Cycled(new[] { 757.7617475757002, -0.0, double.NegativeInfinity, 2.0 }, 2, 129), col)[":, :3"],
                      "float64 (2,3) 412185f5bb7003c1 0000000000000000 7ff0000000000000 8000000000000000 fff8000000000000 3ff6a09e667f3bcd", true);
                Check(fails, "bufsize 256: (256,) ** broadcast int64 (copied)",
                      () => np.power(Cycled(new[] { 757.7617475757002, 1.5 }, 256), np.broadcast_to(np.array(new[] { 2L }), new Shape(256)))[":1"],
                      "float64 (1,) 412185f5bb7003c2");
                Check(fails, "bufsize 256: (257,) ** broadcast int64 (buffered, step 0)",
                      () => np.power(Cycled(new[] { 757.7617475757002, 1.5 }, 257), np.broadcast_to(np.array(new[] { 2L }), new Shape(257)))[":1"],
                      "float64 (1,) 412185f5bb7003c1");
            }
            finally
            {
                np.setbufsize(previous);
            }
            AssertNoMismatch(fails, "NpyIter's buffering decides whether a varying exponent reaches the loop with step 0");
        }

        /// <summary>
        ///     Two boundaries of NumPy's step decision that a mutation pass found unguarded (2026-10-06): one-element
        ///     operands of ONE shape take the trivial loop (pow) unless the out needs a cast — check_for_trivial_loop
        ///     refuses it then, and NpyIter hands the extent-1 exponent step 0 (the fast path, visible through a complex128
        ///     out, which keeps the one-ULP difference of x*x and pow); and with every input cast,
        ///     npyiter_find_buffering_setup stops growing the core at EXACTLY bufsize elements, so at np.setbufsize(256)
        ///     rows of 256 keep a column exponent at step 0 (sqrt: -0.0, NaN) where rows of 255 do not (pow: +0.0, +inf).
        /// </summary>
        [TestMethod]
        public void OneElementCastOut_AndTheBufferStop_FollowNumPysDecision()
        {
            var fails = new List<string>();
            foreach (var shape in new[] { new Shape(1), new Shape(1, 1) })
            {
                string s = shape.NDim == 1 ? "(1,)" : "(1,1)";
                var x = np.full(shape, 757.7617475757002);
                var e = np.full(shape, 2.0);
                Check(fails, $"{s} ** {s} out=complex128 (NpyIter, fast)", () => np.power(x, e, np.empty(shape, np.complex128)),
                      $"complex128 {s} 412185f5bb7003c1 0000000000000000");
                Check(fails, $"{s} ** {s} out=float64 (trivial loop, pow)", () => np.power(x, e, np.empty(shape, np.float64)),
                      $"float64 {s} 412185f5bb7003c2");
            }

            long previous = np.setbufsize(256);
            try
            {
                foreach (var (n, want) in new[] { (255, "0000000000000000 7ff0000000000000"), (256, "8000000000000000 fff8000000000000") })
                {
                    var v = new float[2 * n];
                    Array.Fill(v, 3f);
                    v[n] = -0.0f;
                    v[n + 1] = float.NegativeInfinity;
                    var x = np.array(v).reshape(2, n);
                    var col = np.array(new float[,] { { 2f }, { 0.5f } });
                    Check(fails, $"bufsize 256: f32 (2,{n}) ** f32 (2,1) dtype=f64",
                          () => np.power(x, col, dtype: np.float64)["1, :2"], $"float64 (2,) {want}", true);
                }
            }
            finally
            {
                np.setbufsize(previous);
            }
            AssertNoMismatch(fails, "the trivial loop's cast-out refusal and the buffering stop decide the exponent's step");
        }

        /// <summary>
        ///     <c>dtype=</c> picks the loop, and the exponent is compared in it: 0.50000001 rounds to 0.5f in a float32
        ///     loop, so the float64 bases take FLOAT_power's sqrt.
        /// </summary>
        [TestMethod]
        public void Dtype_ComparesTheExponentInTheLoopDtype()
        {
            var fails = new List<string>();
            Check(fails, "f64 ** 0.50000001 dtype=f32",
                  () => np.power(np.array(new[] { 892.0480346679688, -0.0, double.NegativeInfinity }), NDArray.Scalar(0.50000001), dtype: np.float32),
                  "float32 (3,) 41eeeff9 80000000 ffc00000", true);
            Check(fails, "int8 ** 0.5 (float64 loop)", () => np.power(np.array(new sbyte[] { 4, 9, 2 }), NDArray.Scalar(0.5)),
                  "float64 (3,) 4000000000000000 4008000000000000 3ff6a09e667f3bcd");
            Check(fails, "int32 (3,) ** (1,1)", () => np.power(np.array(new[] { 1, 2, 3 }), np.array(new[,] { { 2 } })),
                  "int32 (1,3) 1 4 9");
            AssertNoMismatch(fails, "the loop dtype decides which fast path, if any, applies");
        }

        /// <summary>
        ///     float16 is NumPy's HALF_power: powf on the halves widened bit for bit, no fast path — <c>-0.0 ** 0.5</c> is
        ///     +0.0, <c>-inf ** 0.5</c> is +inf, a signalling NaN is quieted (and <c>sNaN ** 0</c> is that NaN, not 1), a
        ///     negative NaN keeps its sign, and <c>51648 ** 0.333</c> rounds from powf's float (<c>0x50a4</c>).
        /// </summary>
        [TestMethod]
        public void Float16_IsPowfWithoutAFastPath()
        {
            var fails = new List<string>();
            var h = np.array(new ushort[] { 0x8000, 0xFC00, 0x7C01, 0xFC01, 0x7A4E }).view(np.float16);
            Check(fails, "f16 ** 0.5", () => np.power(h, NDArray.Scalar((Half)0.5)), "float16 (5,) 0000 7c00 7e01 fe01 5b1a", true);
            Check(fails, "f16 ** 1", () => np.power(h, NDArray.Scalar((Half)1.0)), "float16 (5,) 8000 fc00 7e01 fe01 7a4e", true);
            Check(fails, "f16 ** 0", () => np.power(h, NDArray.Scalar((Half)0.0)), "float16 (5,) 3c00 3c00 7e01 7e01 3c00", true);
            Check(fails, "f16 ** 3", () => np.power(h, NDArray.Scalar((Half)3.0)), "float16 (5,) 8000 fc00 7e01 fe01 7c00", true);
            Check(fails, "f16 ** 0.333", () => np.power(h, NDArray.Scalar((Half)0.333)), "float16 (5,) 0000 7c00 7e01 fe01 50a4", true);
            AssertNoMismatch(fails, "HALF_power is powf on every element");
        }

        /// <summary>
        ///     np.float_power's loop is a plain pow loop (PyUFunc_dd_d): no fast path even for a 0-d exponent, so
        ///     <c>float_power(757.7617475757002, 2.0)</c> is pow's <c>...03c2</c>, not np.power's <c>x*x</c>.
        /// </summary>
        [TestMethod]
        public void FloatPower_NeverTakesTheFastPath()
        {
            var fails = new List<string>();
            var x = F64Bases();
            Check(fails, "float_power 2", () => np.float_power(x, NDArray.Scalar(2.0)),
                  "float64 (6,) 412185f5bb7003c2 0000000000000000 7ff0000000000000 40b45eb4aae20983 412cf304b7c7116b 7ff8000000000001", true);
            Check(fails, "float_power 0.5", () => np.float_power(x, NDArray.Scalar(0.5)),
                  "float64 (6,) 403b870871b16d12 0000000000000000 7ff0000000000000 4020fee21157aa51 403f355be0cfbd28 7ff8000000000001", true);
            Check(fails, "float_power -1", () => np.float_power(x, NDArray.Scalar(-1.0)),
                  "float64 (6,) 3f559f1f53a0e60b fff0000000000000 8000000000000000 3f8c5c4eb7d75482 3f50d26a7a48b676 7ff8000000000001", true);
            Check(fails, "float_power 1", () => np.float_power(x, NDArray.Scalar(1.0)),
                  "float64 (6,) 4087ae180f1ceb84 8000000000000000 fff0000000000000 40520da078d01e00 408e6fbafcf2fb68 7ff8000000000001", true);
            Check(fails, "float_power 0", () => np.float_power(x, NDArray.Scalar(0.0)),
                  "float64 (6,) 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 7ff8000000000001", true);
            AssertNoMismatch(fails, "float_power's PyUFunc_dd_d loop calls pow for every element");
        }

        /// <summary>
        ///     The stride-0 kernel's vector bodies (<c>PowerStrideZeroVector128/256/512</c>, one per width, generic over the
        ///     two float loops) give every lane the scalar body's bits: a uniform exponent vector (what the broadcast loops
        ///     hand them) through the one-compare fast forms, and a vector whose lanes differ, or whose exponent is off the
        ///     fast set (3, NaN), through the element-by-element form. The scalar body is NumPy's (the tests above).
        /// </summary>
        [TestMethod]
        public void VectorBodies_GiveEveryLaneTheScalarBodysBits()
        {
            var fails = new List<string>();
            // Bases read from integer arrays at run time, so no NaN constant can be folded to a quiet NaN.
            ulong[] b64 = { 0x4087AE180F1CEB84, 0x8000000000000000, 0xFFF0000000000000, 0x40520DA078D01E00,
                            0x408E6FBAFCF2FB68, 0x7FF0000000000001, 0xFFF8000000000123, 0x0000000000000003 };
            uint[] b32 = { 0xC1163EC5, 0x80000000, 0xFF800000, 0x431BEAAF, 0x445F0313, 0x7F800001, 0xFFC00123, 0x00000003 };
            var x64 = new double[32];
            var x32 = new float[32];
            for (int i = 0; i < 32; i++)
            {
                x64[i] = BitConverter.UInt64BitsToDouble(b64[i % b64.Length]);
                x32[i] = BitConverter.UInt32BitsToSingle(b32[i % b32.Length]);
            }

            double[] exps = { -1.0, 0.0, -0.0, 0.5, 1.0, 2.0, 3.0, double.NaN, 0.25 };
            foreach (var e in exps)
            {
                CheckVectorLanes(fails, $"uniform {e}", x64, Filled(32, e));
                CheckVectorLanes(fails, $"uniform {e}", x32, Array.ConvertAll(Filled(32, e), v => (float)v));
            }
            var mixed = new double[32];
            for (int i = 0; i < 32; i++)
                mixed[i] = exps[(i * 5) % exps.Length];
            CheckVectorLanes(fails, "mixed", x64, mixed);
            CheckVectorLanes(fails, "mixed", x32, Array.ConvertAll(mixed, v => (float)v));
            AssertNoMismatch(fails, "a vector body must give each lane the scalar body's bits");
        }

        /// <summary>
        ///     Runs every vector width over every window of <paramref name="xs"/> / <paramref name="es"/> and records each
        ///     lane whose bits differ from <c>DirectILKernelGenerator.PowerStrideZero</c> of that lane.
        /// </summary>
        /// <typeparam name="T">The loop dtype, double or float.</typeparam>
        /// <param name="fails">Where differences are recorded.</param>
        /// <param name="label">The case name.</param>
        /// <param name="xs">The bases (at least one 512-bit vector long).</param>
        /// <param name="es">The exponents, as long as <paramref name="xs"/>.</param>
        private static void CheckVectorLanes<T>(List<string> fails, string label, T[] xs, T[] es)
            where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
        {
            void Compare(string width, int start, ReadOnlySpan<T> got)
            {
                for (int k = 0; k < got.Length; k++)
                {
                    string want = BitsOf(NumSharp.Backends.Kernels.DirectILKernelGenerator.PowerStrideZero(xs[start + k], es[start + k]));
                    string have = BitsOf(got[k]);
                    if (want != have)
                        fails.Add($"{typeof(T).Name} {width} {label} lane {start + k}: scalar {want}, vector {have}");
                }
            }

            for (int s = 0; s + System.Runtime.Intrinsics.Vector128<T>.Count <= xs.Length; s++)
            {
                var r = NumSharp.Backends.Kernels.DirectILKernelGenerator.PowerStrideZeroVector128(
                    System.Runtime.Intrinsics.Vector128.Create<T>(xs.AsSpan(s)), System.Runtime.Intrinsics.Vector128.Create<T>(es.AsSpan(s)));
                var lanes = new T[System.Runtime.Intrinsics.Vector128<T>.Count];
                System.Runtime.Intrinsics.Vector128.CopyTo(r, lanes);
                Compare("v128", s, lanes);
            }
            for (int s = 0; s + System.Runtime.Intrinsics.Vector256<T>.Count <= xs.Length; s++)
            {
                var r = NumSharp.Backends.Kernels.DirectILKernelGenerator.PowerStrideZeroVector256(
                    System.Runtime.Intrinsics.Vector256.Create<T>(xs.AsSpan(s)), System.Runtime.Intrinsics.Vector256.Create<T>(es.AsSpan(s)));
                var lanes = new T[System.Runtime.Intrinsics.Vector256<T>.Count];
                System.Runtime.Intrinsics.Vector256.CopyTo(r, lanes);
                Compare("v256", s, lanes);
            }
            for (int s = 0; s + System.Runtime.Intrinsics.Vector512<T>.Count <= xs.Length; s++)
            {
                var r = NumSharp.Backends.Kernels.DirectILKernelGenerator.PowerStrideZeroVector512(
                    System.Runtime.Intrinsics.Vector512.Create<T>(xs.AsSpan(s)), System.Runtime.Intrinsics.Vector512.Create<T>(es.AsSpan(s)));
                var lanes = new T[System.Runtime.Intrinsics.Vector512<T>.Count];
                System.Runtime.Intrinsics.Vector512.CopyTo(r, lanes);
                Compare("v512", s, lanes);
            }
        }

        /// <summary>The raw bits of a float32 / float64 value, in hex.</summary>
        /// <typeparam name="T">float or double.</typeparam>
        /// <param name="v">The value.</param>
        /// <returns>8 or 16 hex digits.</returns>
        private static string BitsOf<T>(T v) where T : unmanaged
            => System.Runtime.CompilerServices.Unsafe.SizeOf<T>() == 8
                ? System.Runtime.CompilerServices.Unsafe.As<T, ulong>(ref v).ToString("x16")
                : System.Runtime.CompilerServices.Unsafe.As<T, uint>(ref v).ToString("x8");

        /// <summary>An array of <paramref name="n"/> copies of <paramref name="v"/>.</summary>
        /// <param name="n">The length.</param>
        /// <param name="v">The value.</param>
        /// <returns>The array.</returns>
        private static double[] Filled(int n, double v)
        {
            var a = new double[n];
            Array.Fill(a, v);
            return a;
        }

        /// <summary>The values of <paramref name="pattern"/> cycled into a fresh C-contiguous float64 array (NumPy's np.resize).</summary>
        /// <param name="pattern">The values to repeat.</param>
        /// <param name="dims">The array's dimensions.</param>
        /// <returns>The array.</returns>
        private static NDArray Cycled(double[] pattern, params long[] dims)
        {
            long n = 1;
            foreach (var d in dims)
                n *= d;
            var v = new double[n];
            for (long i = 0; i < n; i++)
                v[i] = pattern[i % pattern.Length];
            return np.array(v).reshape(new Shape(dims));
        }
    }
}
