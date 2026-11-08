using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     <c>np.nansum</c> / <c>np.nanprod</c> as NumPy's statements (<c>numpy/lib/_nanfunctions_impl.py</c>):
    ///     <c>a, mask = _replace_nan(a, 0)</c> (<c>1</c> for nanprod) — a K-order copy with every NaN overwritten
    ///     (<c>np.copyto(a, val, where=isnan(a))</c>; a complex element is NaN when either part is) — then
    ///     <c>np.sum</c> / <c>np.prod</c>, so the bits are those engines'. Every expected value is NumPy 2.4.2's raw bits
    ///     for the same call over the same <c>RandomState</c> draws.
    /// </summary>
    /// <remarks>
    ///     Parity QA-15 (docs/bugs/parity-qa-2026-11-02.md, the owner's "NumPy's statements", 2026-11-05): NumSharp ran fused
    ///     NaN-skipping kernels in another summation order — float16 per axis several ULP off (<c>168.25</c> for NumPy's
    ///     <c>168.375</c>), float32 / float64 one to three ULP, nanprod one ULP — which the oracle excused as the W4-B
    ///     summation-order divergence; nansum / nanprod have left that excuse.
    /// </remarks>
    [TestClass]
    public class np_nansum_nanprod_numpy_port_tests
    {
        /// <summary>NumPy's <c>np.nan</c>: the positive quiet NaN (C#'s <c>double.NaN</c> is the negative one).</summary>
        private static readonly double NaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000);

        /// <summary>The raw bits of a float64 (or complex128, two words per element) result, C order.</summary>
        /// <param name="r">The result.</param>
        /// <returns>One word per float64.</returns>
        private static ulong[] Bits64(NDArray r) => np.ascontiguousarray(r).view(np.uint64).ToArray<ulong>();

        /// <summary>The raw bits of a float32 result, C order.</summary>
        /// <param name="r">The result.</param>
        /// <returns>One word per element.</returns>
        private static uint[] Bits32(NDArray r) => np.ascontiguousarray(r).view(np.uint32).ToArray<uint>();

        /// <summary>The raw bits of a float16 result, C order.</summary>
        /// <param name="r">The result.</param>
        /// <returns>One word per element.</returns>
        private static ushort[] Bits16(NDArray r) => np.ascontiguousarray(r).view(np.uint16).ToArray<ushort>();

        /// <summary>
        ///     <c>RandomState(seed).random_sample(n) * scale + shift</c> with every <paramref name="every"/>-th element NaN —
        ///     the probe's inputs, which NumSharp's RandomState reproduces bit for bit.
        /// </summary>
        /// <param name="seed">The RandomState seed.</param>
        /// <param name="n">The element count.</param>
        /// <param name="every">The NaN step.</param>
        /// <param name="scale">The multiplier.</param>
        /// <param name="shift">The offset.</param>
        /// <returns>A fresh 1-D float64 array.</returns>
        private static NDArray Data(int seed, long n, int every = 17, double scale = 200.0, double shift = -100.0)
        {
            var rs = np.random.RandomState(seed);
            var v = rs.random_sample(n) * scale + shift;
            v[$"::{every}"] = NaN;
            return v;
        }

        /// <summary>
        ///     The parity-QA reproduction: float16 nansum along axis 0 is np.sum's HALF loop over the zero-filled copy (the
        ///     old kernel gave <c>168.25</c> for column 0), and the float64 flat nansum is np.sum's pairwise schedule.
        /// </summary>
        [TestMethod]
        public void Nansum_IsTheSumOfTheNaNReplacedArray()
        {
            using var x100 = Data(100, 100);
            using var f16 = x100.astype(np.float16).reshape(10, 10);
            using var x1000 = Data(1000, 1000);

            Bits16(np.nansum(f16, axis: 0)).Should().Equal((ushort)0x5943, (ushort)0xD7AC, (ushort)0xDAC6, (ushort)0xD367,
                (ushort)0xD8F4, (ushort)0x59F2, (ushort)0x5910, (ushort)0xD8AF, (ushort)0xD9C7, (ushort)0xD207);
            Bits64(np.nansum(x1000)).Should().Equal(0xC0A1E6DE64F42DA0UL);   // -2291.4343639665312
        }

        /// <summary>
        ///     float32 over a (37, 41) matrix: flat, along each axis (keepdims kept), and the transposed view along axis 1 —
        ///     the same sums as axis 0 of the original, because the copy is K-ordered and np.sum reads it as NumPy does.
        /// </summary>
        [TestMethod]
        public void Nansum_Float32_FlatAxesAndTransposedView()
        {
            using var m = Data(7, 37 * 41).astype(np.float32).reshape(37, 41);

            Bits32(np.nansum(m)).Should().Equal(0xC4BB3D8Cu);
            var axis0 = Bits32(np.nansum(m, axis: 0));
            axis0.Take(4).Should().Equal(0x43F7B09Eu, 0x435592D9u, 0x4217EA32u, 0x42F263C7u);
            axis0.Last().Should().Be(0xC3E1CC66u);
            using var keep = np.nansum(m, axis: 1, keepdims: true);
            keep.shape.Should().Equal(37L, 1L);
            Bits32(keep).Take(3).Should().Equal(0x43015634u, 0xC322BB64u, 0x4368D246u);
            Bits32(keep).Last().Should().Be(0xC05EE8FCu);
            Bits32(np.nansum(m.T, axis: 1)).Should().Equal(axis0);
        }

        /// <summary>A float64 reversed, stepped view (<c>m[::-2, 1::3]</c>), flat and along axis 0.</summary>
        [TestMethod]
        public void Nansum_Float64_ReversedSteppedView()
        {
            using var m = Data(9, 60 * 50).reshape(60, 50);
            var v = m["::-2, 1::3"];

            Bits64(np.nansum(v)).Should().Equal(0xC04B617BE84343A0UL);
            var axis0 = Bits64(np.nansum(v, axis: 0));
            axis0.Length.Should().Be(17);
            axis0.Take(3).Should().Equal(0x405DA008600958B9UL, 0x405A000F34300B56UL, 0xC064D2D5101021FEUL);
            axis0.Last().Should().Be(0x4070CA893F0F06E7UL);
        }

        /// <summary>
        ///     nanprod is np.prod of the copy with NaN set to 1: float64 flat, float32 along axis 1 and float16 along axis 0
        ///     (values near 1, so the products stay finite).
        /// </summary>
        [TestMethod]
        public void Nanprod_IsTheProductOfTheNaNReplacedArray()
        {
            using var p = Data(11, 200, every: 7, scale: 0.5, shift: 0.75);

            Bits64(np.nanprod(p)).Should().Equal(0x4002801C42339F7CUL);
            Bits32(np.nanprod(p.astype(np.float32).reshape(10, 20), axis: 1)).Should().Equal(0x3F092139u, 0x3F8795B5u,
                0x3F12FB03u, 0x403824FBu, 0x3FBCC066u, 0x3F99211Bu, 0x403F91FFu, 0x3F94D1AEu, 0x3F1A8A7Cu, 0x3F2A6A8Bu);
            var h = Bits16(np.nanprod(p.astype(np.float16).reshape(10, 20), axis: 0));
            h.Take(4).Should().Equal((ushort)0x3858, (ushort)0x3DAC, (ushort)0x3E62, (ushort)0x39C8);
            h.Last().Should().Be((ushort)0x3F79);
        }

        /// <summary>
        ///     complex128: a NaN in EITHER part makes the element NaN (<c>isnan</c> of a complex), so the whole element is
        ///     replaced — by <c>0</c> for nansum (flat and along axis 1), by <c>1</c> for nanprod.
        /// </summary>
        [TestMethod]
        public void Complex128_ANaNInEitherPartReplacesTheElement()
        {
            var rs = np.random.RandomState(13);
            var re = (rs.random_sample(64L) * 2.0 - 1.0).ToArray<double>();
            var im = (rs.random_sample(64L) * 2.0 - 1.0).ToArray<double>();
            var z = Enumerable.Range(0, 64).Select(k => new Complex(re[k], im[k])).ToArray();
            for (int k = 0; k < 64; k += 9) z[k] = new Complex(NaN, 0.5);
            for (int k = 4; k < 64; k += 9) z[k] = new Complex(0.25, NaN);
            // 1 + 0.01 * z as NumPy computes it (the NaN elements stay NaN in some part, and are replaced).
            var zp = z.Select(c => double.IsNaN(c.Real) || double.IsNaN(c.Imaginary)
                ? new Complex(NaN, NaN) : new Complex(1.0 + 0.01 * c.Real, 0.01 * c.Imaginary)).ToArray();
            using var za = np.array(z);
            using var zpa = np.array(zp);

            Bits64(np.nansum(za)).Should().Equal(0x3FA2F81BD7291880UL, 0xC00F13B7808DABEFUL);
            var rows = Bits64(np.nansum(za.reshape(8, 8), axis: 1));
            rows.Take(4).Should().Equal(0x3FFBB2E3E056A06CUL, 0xBFD01A73B50287E4UL, 0xBFFD3CEAD18657FEUL, 0x3FE3B4D7B06BE766UL);
            rows.Skip(14).Should().Equal(0x3FF49C2900B46C79UL, 0x3FF1D403933DD378UL);
            Bits64(np.nanprod(zpa)).Should().Equal(0x3FEFFDE75B1BD897UL, 0xBFA3EEDB3BA1AB72UL);
        }

        /// <summary>
        ///     The axis result is the array np.sum / np.prod allocate: it owns its data, takes the reduced array's axis order
        ///     (a permuted input's K-order copy is permuted, an F input gives F), and keepdims re-inserts the axis in place.
        ///     Found by the array-object oracle while porting: np.evaluate's own result was C-ordered, and a keepdims result
        ///     a view of it (NumPy's strides below; a length-1 axis's stride is not compared).
        /// </summary>
        [TestMethod]
        public void AxisResult_OwnsItsDataInTheInputsOrder()
        {
            using var c = (np.arange(24.0) + 1.0).reshape(2, 3, 4);
            var permuted = c.transpose(new[] { 2, 0, 1 });             // (4, 2, 3), strides (8, 96, 32)
            using var f3 = np.asfortranarray(c);

            foreach (var reduce in new Func<NDArray, int, bool, NDArray>[]
                     {
                         (x, ax, kd) => np.nansum(x, axis: ax, keepdims: kd),
                         (x, ax, kd) => np.nanprod(x, axis: ax, keepdims: kd),
                     })
            {
                using var p = reduce(permuted, 1, false);
                p.strides.Should().Equal(8L, 32L);
                p.flags.owndata.Should().BeTrue();
                // The stride of a re-inserted length-1 axis is not part of the layout (NumPy's flags ignore it, and so
                // does the oracle's strides facet): only the extent > 1 axes are compared.
                using var pk = reduce(permuted, 1, true);
                pk.shape.Should().Equal(4L, 1L, 3L);
                new[] { pk.strides[0], pk.strides[2] }.Should().Equal(8L, 32L);
                pk.flags.owndata.Should().BeTrue();
                using var fk = reduce(f3, 1, true);
                fk.shape.Should().Equal(2L, 1L, 4L);
                new[] { fk.strides[0], fk.strides[2] }.Should().Equal(8L, 16L);
                fk.flags.f_contiguous.Should().BeTrue();
                fk.flags.owndata.Should().BeTrue();
                using var ck = reduce(c, 0, true);
                ck.strides.Should().Equal(96L, 32L, 8L);
                ck.flags.owndata.Should().BeTrue();
            }

            using var prod = np.nanprod(permuted, axis: 1);
            prod.ravel().ToArray<double>().Take(5).Should().Equal(13.0, 85.0, 189.0, 28.0, 108.0);
        }

        /// <summary>
        ///     The identities: an all-NaN slice sums to <c>+0.0</c> and multiplies to <c>1.0</c>; <c>-0.0</c> elements sum to
        ///     <c>+0.0</c> (np.sum's +0 identity, one element included); a NaN 0-d array gives the identity; an integer input
        ///     is np.sum itself (int32 → int64).
        /// </summary>
        [TestMethod]
        public void Identities_AllNaNNegativeZeroZeroDAndIntegers()
        {
            using var allnan = np.array(new[,] { { NaN, NaN }, { 1.5, NaN } });

            Bits64(np.nansum(allnan, axis: 1)).Should().Equal(0x0000000000000000UL, 0x3FF8000000000000UL);
            Bits64(np.nanprod(allnan, axis: 1)).Should().Equal(0x3FF0000000000000UL, 0x3FF8000000000000UL);
            Bits64(np.nansum(np.array(new[] { -0.0, NaN }))).Should().Equal(0UL);
            Bits64(np.nansum(np.array(new[] { -0.0 }))).Should().Equal(0UL);
            Bits64(np.nansum(np.array(new[] { -0.0, -0.0 }))).Should().Equal(0UL);
            Bits64(np.nansum(NDArray.Scalar(NaN))).Should().Equal(0UL);
            Bits64(np.nanprod(NDArray.Scalar(NaN))).Should().Equal(0x3FF0000000000000UL);
            using var ints = np.nansum(np.arange(10, NPTypeCode.Int32));
            ints.typecode.Should().Be(NPTypeCode.Int64);
            ints.GetInt64(0).Should().Be(45L);
        }
    }
}
