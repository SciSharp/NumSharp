using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     The float32 / float64 <c>np.prod</c> is NumPy's <c>FLOAT_multiply</c> / <c>DOUBLE_multiply</c> reduce: every
    ///     output starts at the identity 1 and multiplies its elements one after the other (<c>acc *= *src</c>; multiply is
    ///     not pairwise), flat, along an axis and through <c>out=</c>. Regression pins of the owner's decision of
    ///     2026-11-12, which reversed D24 / NL3 ("Keep SIMD lanes", 2026-11-06): the SIMD lanes it replaced differed in the
    ///     last bit of most long products and met a zero before an overflow (0 where NumPy gives NaN).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The order is NumPy's iterator's: a contiguous array over every axis in memory order (an F-ordered array
    ///     column by column), a reversed view in its index order (NumPy does not flip it), each output of an axis product
    ///     in increasing order of the reduced index. Expected bits were produced by NumPy 2.4.2 for the formula inputs
    ///     below (<c>1 + i * 1e-3</c>).
    ///     </para>
    ///     <para>
    ///     Speed: the rows of a C-contiguous axis product run eight independent chains at once, an outer axis multiplies
    ///     whole rows with vectors, and a flat product is NumPy's dependent chain at NumPy's own speed
    ///     (<c>Default.Reduction.Product.Sequential.cs</c>).
    ///     </para>
    /// </remarks>
    [TestClass]
    public class np_prod_sequential_order_tests
    {
        /// <summary>
        ///     64 float64 values: 1e200 twice at the front, a zero at index 8 (the first lane of a V128, V256 and V512 vector
        ///     alike), ones elsewhere. NumPy's sequential product overflows to inf at the second value and <c>inf * 0</c> is
        ///     NaN; SIMD lanes would multiply the first 1e200 by the zero and the second by ones, giving +0.0.
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

        /// <summary>The formula line <c>1 + i * 1e-3</c>, i = 0 .. n - 1, as float64.</summary>
        /// <param name="n">The length.</param>
        /// <returns>A fresh array.</returns>
        private static NDArray Ramp(int n)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++) v[i] = 1.0 + i * 1e-3;
            return np.array(v);
        }

        /// <summary>The flat float64 product is NumPy's sequential one: the overflow meets the zero as NaN.</summary>
        [TestMethod]
        public void FlatFloat64_OverflowBeforeZero_IsNaN()
        {
            using var a = OverflowBeforeZero64();
            using var p = np.prod(a);
            double.IsNaN(p.GetAtIndex<double>(0)).Should().BeTrue("NumPy's sequential loop reaches inf * 0");
            using var sequential = np.prod(a, initial: 1.0);
            double.IsNaN(sequential.GetAtIndex<double>(0)).Should().BeTrue("initial= runs the same loop");
        }

        /// <summary>The flat float32 product, the same way.</summary>
        [TestMethod]
        public void FlatFloat32_OverflowBeforeZero_IsNaN()
        {
            using var a = OverflowBeforeZero32();
            using var p = np.prod(a);
            float.IsNaN(p.GetAtIndex<float>(0)).Should().BeTrue("NumPy's sequential loop reaches inf * 0");
        }

        /// <summary>
        ///     The axis product and its <c>out=</c> route multiply each row sequentially: <c>[nan, 1]</c>, NumPy's answer.
        /// </summary>
        [TestMethod]
        public void AxisFloat64_EachRowIsSequential()
        {
            using var row = OverflowBeforeZero64();
            using var ones = np.ones(new Shape(64));
            using var x = np.stack(new[] { row, ones });

            using var p = np.prod(x, 1);
            double.IsNaN(p.GetAtIndex<double>(0)).Should().BeTrue("NumPy: nan");
            p.GetAtIndex<double>(1).Should().Be(1.0);

            using var o = np.zeros(new Shape(2));
            np.prod(x, 1, @out: o);
            double.IsNaN(o.GetAtIndex<double>(0)).Should().BeTrue("the out= route computes the same product");
        }

        /// <summary>
        ///     NumPy 2.4.2's bits of flat products of <c>1 + i * 1e-3</c>: float64 and float32, forward and reversed (a
        ///     reversed view multiplies in its index order), where a different order differs in the last bits.
        /// </summary>
        [TestMethod]
        public void FlatProducts_AreNumPysBits()
        {
            (int N, ulong F64, uint F32, ulong Reversed)[] cases =
            {
                (4, 0x3FF0189EFF22FA19, 0x3F80C4F9, 0x3FF0189EFF22FA19),
                (7, 0x3FF056BC5E846C01, 0x3F82B5E4, 0x3FF056BC5E846C00),
                (100, 0x405E2E8DBC99AAB0, 0x42F1746E, 0x405E2E8DBC99AAB4),
                (1000, 0x62BBF3CEE4ABC5BB, 0x7F800000, 0x62BBF3CEE4ABC5AF),
            };
            foreach (var (n, f64, f32, reversed) in cases)
            {
                using var a = Ramp(n);
                using var p = np.prod(a);
                ((ulong)BitConverter.DoubleToInt64Bits(p.GetAtIndex<double>(0))).Should().Be(f64, $"float64 n = {n}");
                using var a32 = a.astype(NPTypeCode.Single);
                using var p32 = np.prod(a32);
                ((uint)BitConverter.SingleToInt32Bits(p32.GetAtIndex<float>(0))).Should().Be(f32, $"float32 n = {n}");
                using var rev = np.prod(a["::-1"]);
                ((ulong)BitConverter.DoubleToInt64Bits(rev.GetAtIndex<double>(0))).Should().Be(reversed, $"reversed n = {n}");
            }
        }

        /// <summary>
        ///     NumPy 2.4.2's bits of a (3, 100) product of the same formula along each axis: rows (eight chains at once) and
        ///     the outer axis (the elementwise form).
        /// </summary>
        [TestMethod]
        public void AxisProducts_AreNumPysBits()
        {
            using var m = Ramp(300).reshape(3, 100);
            using var rows = np.prod(m, 1);
            ulong[] rowBits = { 0x405E2E8DBC99AAB0, 0x41309F8E934396AF, 0x41F11B91629FAFE2 };
            for (int i = 0; i < 3; i++)
                ((ulong)BitConverter.DoubleToInt64Bits(rows.GetAtIndex<double>(i))).Should().Be(rowBits[i], $"row {i}");

            using var cols = np.prod(m, 0);
            ulong[] colBits = { 0x3FF51EB851EB851F, 0x3FF52D8FA0604C09, 0x3FF53C6DDC1DEFF8 };
            for (int j = 0; j < 3; j++)
                ((ulong)BitConverter.DoubleToInt64Bits(cols.GetAtIndex<double>(j))).Should().Be(colBits[j], $"column {j}");
        }
    }
}
