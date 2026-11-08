using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     The float32 / float64 <c>np.prod</c> multiplies in SIMD lane order, where NumPy's <c>FLOAT_multiply</c> /
    ///     <c>DOUBLE_multiply</c> reduce is a sequential loop (<c>acc *= *src</c>; multiply is not pairwise): a documented
    ///     divergence the owner chose to keep (the np-layer plan's D24, 2026-11-06, NL3: "Keep SIMD lanes"). Measured on
    ///     float64 (NumPy / NumSharp's lanes / a sequential loop in NumSharp): 1K 4.11 / 1.12 / 1.27 us, 100K 141.65 / 8.79 /
    ///     139.22 us, 1M 1405.55 / 229.80 / 1395.75 us - the lanes are 3.7-16x NumPy, NumPy's order would be NumPy's speed -
    ///     so the flat and the axis products, and the <c>out=</c> route, keep them.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     What differs: the last bits of a product of many values (random float32 data: 18 of 35 C-contiguous 1-D
    ///     products of 4 to 1,000 values differ by an ULP; float64: 20 of 35), the order an F-ordered or reversed operand is
    ///     walked in, and a product that overflows before it meets a zero (NumPy's <c>inf * 0</c> is NaN, NumSharp's lanes
    ///     can meet the zero first). Which bits come back depends on the host's vector width (V128 / V256 / V512), so the
    ///     cases below use values whose answer is the same for every width: two huge values in neighbouring lanes and a
    ///     zero in the first lane of every width.
    ///     </para>
    ///     <para>
    ///     NumPy's exact order is one keyword away: with <c>initial=</c> or <c>where=</c> the product runs NumPy's reduction
    ///     iterator and its sequential loop (<c>NumPyMultiplyLoops</c>), bit for bit, so <c>np.prod(a, initial: 1.0)</c> is
    ///     NumPy's <c>np.prod(a)</c> for a float array.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class np_prod_lane_order_tests
    {
        /// <summary>
        ///     64 float64 values: 1e200 twice at the front, a zero at index 8 (the first lane of a V128, V256 and V512 vector
        ///     alike), ones elsewhere. NumPy's sequential product overflows to inf at the second value and <c>inf * 0</c> is
        ///     NaN; NumSharp's lanes multiply the first 1e200 by the zero and the second by ones, so the product is +0.0.
        /// </summary>
        /// <returns>A fresh array.</returns>
        private static NDArray OverflowBeforeZero64()
        {
            var v = new double[64];
            Array.Fill(v, 1.0);
            v[0] = 1e200;
            v[1] = 1e200;
            v[8] = 0.0;
            return np.array(v);
        }

        /// <summary>The float32 form of <see cref="OverflowBeforeZero64"/>: 2**64 twice, a zero at index 16 (the first lane
        ///     of every float32 vector width up to 16 lanes).</summary>
        /// <returns>A fresh array of 64 values.</returns>
        private static NDArray OverflowBeforeZero32()
        {
            var v = new float[64];
            Array.Fill(v, 1f);
            v[0] = 18446744073709551616f;   // 2**64: its square overflows float32
            v[1] = 18446744073709551616f;
            v[16] = 0f;
            return np.array(v);
        }

        /// <summary>
        ///     The flat float64 product keeps its SIMD lanes: +0.0 where NumPy's sequential loop gives NaN. With
        ///     <c>initial=1.0</c> the product is NumPy's loop and NaN.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void FlatFloat64_MultipliesInSimdLanes()
        {
            using var a = OverflowBeforeZero64();
            using var lanes = np.prod(a);
            BitConverter.DoubleToInt64Bits(lanes.GetAtIndex<double>(0)).Should().Be(0L, "NumSharp's lanes meet the zero before the overflow (NumPy: nan)");

            using var sequential = np.prod(a, initial: 1.0);
            double.IsNaN(sequential.GetAtIndex<double>(0)).Should().BeTrue("initial= runs NumPy's sequential loop, whose inf * 0 is NaN");
        }

        /// <summary>The flat float32 product, the same way: +0.0, and NaN with <c>initial=1.0</c>.</summary>
        [TestMethod]
        [Misaligned]
        public void FlatFloat32_MultipliesInSimdLanes()
        {
            using var a = OverflowBeforeZero32();
            using var lanes = np.prod(a);
            BitConverter.SingleToInt32Bits(lanes.GetAtIndex<float>(0)).Should().Be(0, "NumSharp's lanes meet the zero before the overflow (NumPy: nan)");

            using var sequential = np.prod(a, initial: 1.0);
            float.IsNaN(sequential.GetAtIndex<float>(0)).Should().BeTrue("initial= runs NumPy's sequential loop");
        }

        /// <summary>
        ///     The axis product multiplies each row in lanes too, and so does the <c>out=</c> route (it computes the same plain
        ///     product): <c>[0, 1]</c> where NumPy gives <c>[nan, 1]</c>. With <c>initial=1.0</c> each row is NumPy's loop.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void AxisFloat64_MultipliesEachRowInSimdLanes()
        {
            using var row = OverflowBeforeZero64();
            using var ones = np.ones(new Shape(64));
            using var x = np.stack(new[] { row, ones });

            using var lanes = np.prod(x, 1);
            BitConverter.DoubleToInt64Bits(lanes.GetAtIndex<double>(0)).Should().Be(0L, "NumPy: nan");
            lanes.GetAtIndex<double>(1).Should().Be(1.0);

            using var o = np.zeros(new Shape(2));
            np.prod(x, 1, @out: o);
            BitConverter.DoubleToInt64Bits(o.GetAtIndex<double>(0)).Should().Be(0L, "the out= route computes the same lanes");

            using var sequential = np.prod(x, 1, initial: 1.0);
            double.IsNaN(sequential.GetAtIndex<double>(0)).Should().BeTrue("initial= runs NumPy's sequential loop");
            sequential.GetAtIndex<double>(1).Should().Be(1.0);
        }
    }
}
