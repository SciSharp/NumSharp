using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     <c>np.prod</c> / <c>np.cumprod</c> over complex128 and float16 run NumPy's multiply loops in the call form NumPy's
    ///     iterator chooses. complex128: a reduce-form call (a flat product, a row whose reduced axis is innermost) and every
    ///     accumulate step are <c>CDOUBLE_multiply</c>'s MSVC-contracted <c>loop_scalar</c>
    ///     (<c>im = fma(a.im, b.re, a.re*b.im)</c>); a slab (an outer reduced axis) is the elementwise <c>simd_cmul</c>.
    ///     float16: <c>HALF_multiply</c>'s reduce form keeps a float32 accumulator for one call and narrows once (so a partial
    ///     that overflows float32 is inf even if later factors would shrink it, and one that overflows only float16 is not);
    ///     the elementwise form narrows every product. Every literal was probed against NumPy 2.4.2 (win-amd64).
    /// </summary>
    /// <remarks>
    ///     Parity QA-10 and QA-16 (docs/bugs/parity-qa-2026-11-02.md). The plain np.prod of a complex128 / float16 loop now
    ///     runs the keyword path's reduction iterator (<c>DefaultEngine.AddMulNumPy</c>), and the complex cumprod scans
    ///     multiply with <c>ILKernelGenerator.PolyLoopScalarComplexMultiply</c>.
    /// </remarks>
    [TestClass]
    public class np_prod_complex_and_half_loop_tests
    {
        /// <summary>Asserts the raw bits of both parts of <paramref name="z"/>.</summary>
        /// <param name="z">The value read back.</param>
        /// <param name="re">The expected real part's bits, as NumPy prints them.</param>
        /// <param name="im">The expected imaginary part's bits.</param>
        /// <param name="because">What the value is, for the failure text.</param>
        private static void ShouldHaveBits(Complex z, ulong re, ulong im, string because)
        {
            BitConverter.DoubleToInt64Bits(z.Real).Should().Be(unchecked((long)re), because);
            BitConverter.DoubleToInt64Bits(z.Imaginary).Should().Be(unchecked((long)im), because);
        }

        /// <summary>Asserts the raw bits of a float16 value.</summary>
        /// <param name="h">The value read back.</param>
        /// <param name="bits">The expected bits, as NumPy prints them.</param>
        /// <param name="because">What the value is, for the failure text.</param>
        private static void ShouldHaveBits(Half h, ushort bits, string because)
            => BitConverter.HalfToUInt16Bits(h).Should().Be(bits, because);

        /// <summary>The deterministic complex128 sequence both sides of the probe built (3,000 values near 1).</summary>
        /// <returns>A fresh 1-D complex128 array.</returns>
        private static NDArray ComplexSequence()
            => np.array(Enumerable.Range(0, 3000)
                .Select(i => new Complex(1.0 + (((i * 7919L) % 201) - 100) * 1e-3, (((i * 104729L) % 199) - 99) * 1e-3))
                .ToArray());

        /// <summary>The deterministic float16 sequence of the probe (20,000 values near 1), cast from float64.</summary>
        /// <param name="multiplier">The sequence's step multiplier (7919 for the main sequence).</param>
        /// <returns>A fresh 1-D float64 array (cast it to the dtype the case needs).</returns>
        private static NDArray NearOneSequence(long multiplier)
            => np.array(Enumerable.Range(0, 20000).Select(i => 1.0 + (((i * multiplier) % 201) - 100) * 1e-4).ToArray());

        /// <summary>
        ///     The finding's case: <c>x = -3+1e10j</c>, <c>x**4</c> by np.prod and by np.cumprod. NumPy's imaginary part is
        ///     <c>1.2e+31</c> (<c>0x4662eec2eb3869af</c>); the elementwise form gave one ULP more.
        /// </summary>
        [TestMethod]
        public void Complex_ProdAndCumprod_UseTheContractedLoopScalar()
        {
            using var x = np.array(Enumerable.Repeat(new Complex(-3, 1e10), 4).ToArray());

            using var prod = np.prod(x);
            ShouldHaveBits((Complex)prod.GetValue(), 0x483d6329f1c35ca5UL, 0x4662eec2eb3869afUL, "prod(x) = x**4");

            using var cumprod = np.cumprod(x);
            ShouldHaveBits(cumprod.GetAtIndex<Complex>(0), 0xc008000000000000UL, 0x4202a05f20000000UL, "cumprod[0]");
            ShouldHaveBits(cumprod.GetAtIndex<Complex>(1), 0xc415af1d78b58c40UL, 0xc22bf08eb0000000UL, "cumprod[1]");
            ShouldHaveBits(cumprod.GetAtIndex<Complex>(2), 0x4448650127cc3dc8UL, 0xc6293e5939a08ceaUL, "cumprod[2]");
            ShouldHaveBits(cumprod.GetAtIndex<Complex>(3), 0x483d6329f1c35ca5UL, 0x4662eec2eb3869afUL, "cumprod[3] = prod");
        }

        /// <summary>
        ///     The call form follows NumPy's iterator: a flat product, a strided view and an innermost-axis row are reduce-form
        ///     calls (loop_scalar), an outer reduced axis is slab mode (simd_cmul: NumPy's <c>prod(axis=0)</c> of a (1000, 3)
        ///     array differs from the loop_scalar products of the same columns in the last bits), and nanprod is prod.
        /// </summary>
        [TestMethod]
        public void Complex_Prod_FollowsNumPysCallForms()
        {
            using var c = ComplexSequence();

            using var flat = np.prod(c);
            ShouldHaveBits((Complex)flat.GetValue(), 0x3ff2ca9645c355d2UL, 0xbfc29dc31c8d8c50UL, "flat");

            using var rows = np.prod(c.reshape(3, 1000), axis: 1);
            ShouldHaveBits(rows.GetAtIndex<Complex>(0), 0x3ff0e89e52f43ce5UL, 0xbfc6e6cfab24a04cUL, "axis 1 row 0 (reduce form)");
            ShouldHaveBits(rows.GetAtIndex<Complex>(1), 0x3ff0e2576a7d8f06UL, 0x3f9e601e159e1855UL, "axis 1 row 1");
            ShouldHaveBits(rows.GetAtIndex<Complex>(2), 0x3ff0bb63e50bacadUL, 0x3f9190f6d9042fd0UL, "axis 1 row 2");

            using var slabs = np.prod(c.reshape(1000, 3), axis: 0);
            ShouldHaveBits(slabs.GetAtIndex<Complex>(0), 0x3fd8a96e4c7d71a0UL, 0xbf96187b7a99efb9UL, "axis 0 column 0 (slab: simd_cmul)");
            ShouldHaveBits(slabs.GetAtIndex<Complex>(1), 0x40071f0d1b19fa68UL, 0x3fb00e513bee8f8eUL, "axis 0 column 1");
            ShouldHaveBits(slabs.GetAtIndex<Complex>(2), 0x3ff0e7733a8e6e81UL, 0xbfb820520381b28dUL, "axis 0 column 2");

            using var strided = np.prod(c["::3"]);
            ShouldHaveBits((Complex)strided.GetValue(), 0x3fd8a96e4c7d71aaUL, 0xbf96187b7a99efcaUL, "column 0 as a strided flat product (reduce form)");

            using var nan = np.nanprod(c);
            ShouldHaveBits((Complex)nan.GetValue(), 0x3ff2ca9645c355d2UL, 0xbfc29dc31c8d8c50UL, "nanprod without NaN is prod");
        }

        /// <summary>
        ///     Every accumulate step is loop_scalar, on every axis: cumprod's last element equals the flat product, its last row
        ///     along axis 0 is the loop_scalar product of each column (NOT np.prod(axis=0)'s slab bits), and its last column
        ///     along axis 1 is np.prod(axis=1).
        /// </summary>
        [TestMethod]
        public void Complex_Cumprod_EveryStepIsLoopScalar_OnEveryAxis()
        {
            using var c = ComplexSequence();

            using var flat = np.cumprod(c);
            ShouldHaveBits(flat.GetAtIndex<Complex>(2999), 0x3ff2ca9645c355d2UL, 0xbfc29dc31c8d8c50UL, "flat cumprod's last element");

            using var down = np.cumprod(c.reshape(1000, 3), axis: 0);
            using var lastRow = down["-1"];
            ShouldHaveBits(lastRow.GetAtIndex<Complex>(0), 0x3fd8a96e4c7d71aaUL, 0xbf96187b7a99efcaUL, "axis 0 last row [0]");
            ShouldHaveBits(lastRow.GetAtIndex<Complex>(1), 0x40071f0d1b19fa40UL, 0x3fb00e513bee8efbUL, "axis 0 last row [1]");
            ShouldHaveBits(lastRow.GetAtIndex<Complex>(2), 0x3ff0e7733a8e6e91UL, 0xbfb820520381b2aaUL, "axis 0 last row [2]");

            using var across = np.cumprod(c.reshape(3, 1000), axis: 1);
            using var lastColumn = across[":, -1"];
            ShouldHaveBits(lastColumn.GetAtIndex<Complex>(0), 0x3ff0e89e52f43ce5UL, 0xbfc6e6cfab24a04cUL, "axis 1 last column [0]");
            ShouldHaveBits(lastColumn.GetAtIndex<Complex>(1), 0x3ff0e2576a7d8f06UL, 0x3f9e601e159e1855UL, "axis 1 last column [1]");
            ShouldHaveBits(lastColumn.GetAtIndex<Complex>(2), 0x3ff0bb63e50bacadUL, 0x3f9190f6d9042fd0UL, "axis 1 last column [2]");

            // A non-contiguous input takes the cumprod fallback's copy: the same steps.
            using var stridedCum = np.cumprod(c["::3"]);
            ShouldHaveBits(stridedCum.GetAtIndex<Complex>(999), 0x3fd8a96e4c7d71aaUL, 0xbf96187b7a99efcaUL, "strided cumprod's last element");
        }

        /// <summary>
        ///     The finding's float16 cases: a float32 accumulator overflows at the 17th factor of 200 and the zero makes it NaN
        ///     (plain float16 and <c>dtype=float16</c> over uint8 alike); <c>60000**9</c> is already inf in float32, so later
        ///     factors of 1e-4 cannot bring it back; a row whose float16 partial would overflow (90000) stays finite in float32
        ///     (90.0625), along axis 1 and along an axis 0 that leaves one output; with two outputs axis 0 is slab mode, which
        ///     narrows every product (inf).
        /// </summary>
        [TestMethod]
        public void Half_Prod_IsHalfMultiplysFloat32Call()
        {
            var values = Enumerable.Repeat(200.0, 20).Concat(new[] { 0.0 }).ToArray();
            using var f64 = np.array(values);
            using var f16 = f64.astype(np.float16);
            using var u8 = f64.astype(np.uint8);

            using var plain = np.prod(f16);
            Half.IsNaN((Half)plain.GetValue()).Should().BeTrue("inf * 0 in the float32 accumulator");
            using var cast = np.prod(u8, dtype: np.float16);
            Half.IsNaN((Half)cast.GetValue()).Should().BeTrue("the uint8 cast loop reduces in the same HALF loop");
            using var keep = np.prod(f16.reshape(3, 7), keepdims: true);
            keep.shape.Should().Equal(1L, 1L);
            Half.IsNaN(keep.GetAtIndex<Half>(0)).Should().BeTrue("keepdims: the same flat product");

            using var big = np.array(Enumerable.Repeat(60000.0, 9).Concat(Enumerable.Repeat(1e-4, 10)).ToArray()).astype(np.float16);
            using var bigProd = np.prod(big);
            ShouldHaveBits((Half)bigProd.GetValue(), 0x7c00, "60000**9 overflows float32 before the small factors");

            using var row = np.array(new[,] { { 300.0, 300.0, 0.001 } }).astype(np.float16);
            using var rowProd = np.prod(row, axis: 1);
            ShouldHaveBits(rowProd.GetAtIndex<Half>(0), 0x55a1, "a row is one float32 call: 90.0625");

            using var column = np.array(new[,] { { 300.0 }, { 300.0 }, { 0.001 } }).astype(np.float16);
            using var columnProd = np.prod(column, axis: 0);
            ShouldHaveBits(columnProd.GetAtIndex<Half>(0), 0x55a1, "one output: NumPy's iterator drops the extent-1 axis, a reduce-form call");

            using var slab = np.array(new[,] { { 300.0, 300.0 }, { 300.0, 300.0 }, { 0.001, 0.001 } }).astype(np.float16);
            using var slabProd = np.prod(slab, axis: 0);
            ShouldHaveBits(slabProd.GetAtIndex<Half>(0), 0x7c00, "slab mode narrows 90000 to inf");
            ShouldHaveBits(slabProd.GetAtIndex<Half>(1), 0x7c00, "slab mode, second output");
        }

        /// <summary>
        ///     A long float16 product in every layout: contiguous, a strided view, a transposed input (K order), rows, slabs, and
        ///     a float32 input under <c>dtype=float16</c>, whose buffered cast narrows the running value at every 8,192-element
        ///     fill exactly where NumPy's does (<c>39db</c>, where one call over the cast copy gives <c>39da</c>).
        /// </summary>
        [TestMethod]
        public void Half_Prod_LongInputs_MatchNumPyInEveryLayout()
        {
            using var det64 = NearOneSequence(7919);
            using var det = det64.astype(np.float16);

            using var flat = np.prod(det);
            ShouldHaveBits((Half)flat.GetValue(), 0x39f7, "contiguous flat");
            using var strided = np.prod(det["::3"]);
            ShouldHaveBits((Half)strided.GetValue(), 0x377b, "strided flat");
            using var transposed = np.prod(det.reshape(4, 5000).T);
            ShouldHaveBits((Half)transposed.GetValue(), 0x39f7, "transposed flat (memory order)");

            using var rows = np.prod(det.reshape(4, 5000), axis: 1);
            ushort[] rowBits = { 0x3bb6, 0x3b86, 0x3b58, 0x3b2b };
            for (int i = 0; i < 4; i++)
                ShouldHaveBits(rows.GetAtIndex<Half>(i), rowBits[i], $"row {i}");

            using var slabs = np.prod(det.reshape(5000, 4), axis: 0);
            ushort[] slabBits = { 0x3be3, 0x3bdb, 0x3bd4, 0x3bd3 };
            for (int i = 0; i < 4; i++)
                ShouldHaveBits(slabs.GetAtIndex<Half>(i), slabBits[i], $"slab output {i}");

            using var fills64 = NearOneSequence(7927);
            using var fills32 = fills64.astype(np.float32);
            using var buffered = np.prod(fills32, dtype: np.float16);
            ShouldHaveBits((Half)buffered.GetValue(), 0x39db, "the cast loop narrows at each buffer fill");
            using var castCopy = fills32.astype(np.float16);
            using var oneCall = np.prod(castCopy);
            ShouldHaveBits((Half)oneCall.GetValue(), 0x39da, "one call over the float16 copy narrows once");
            using var castRows = np.prod(fills32.reshape(4, 5000), axis: 1, dtype: np.float16);
            ShouldHaveBits(castRows.GetAtIndex<Half>(0), 0x3b68, "a cast loop along an axis");
        }
    }
}
