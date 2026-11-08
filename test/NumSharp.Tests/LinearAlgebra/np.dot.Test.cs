using System;
using System.Linq;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.LinearAlgebra
{
    [TestClass]
    public class np_dot_test : TestClass
    {
        [TestMethod]
        public void Vector_Vector()
        {
            var a = arange(5);
            var b = arange(5);
            np.dot(a, b).Should().BeScalar(30);
        }

        [TestMethod]
        public void Dot0X0()
        {
            int x = 2;
            int y = 3;
            int z = (int)np.dot((NDArray)x, (NDArray)y);

            Assert.AreEqual(z, 6);
        }

        [TestMethod]
        public void Dot1x1()
        {
            var x = np.arange(3);
            var y = np.arange(3, 6);

            int nd3 = (int)np.dot(x, y);
            Assert.IsTrue(nd3 == 14);
        }

        [TestMethod]
        public void Dot2x1()
        {
            var x = np.array(1, 1, 1, 2, 2, 2, 2, 3).reshape((4, 2));
            var y = np.array(2, 3);

            np.dot(x, y).Should().BeShaped(4).And.BeOfValues(5, 8, 10, 13);
        }

        [TestMethod]
        public void Dot2x2()
        {
            var x = array((2, 2), 3, 1, 1, 2);
            var y = array((2, 2), 2, 3, 1, 2);
            np.dot(x, y).Should().BeShaped(2, 2).And.BeOfValues(7, 11, 4, 7);
        }

        [TestMethod]
        public void Dot2222x2222()
        {
            var x = arange((2, 2, 2, 2));
            var y = arange((2, 2, 2, 2));
            np.dot(x, y).Should().BeShaped(2, 2, 2, 2, 2, 2)
                .And.BeOfValues(2, 3, 6, 7, 10, 11, 14, 15, 6, 11, 26, 31, 46, 51,
                    66, 71, 10, 19, 46, 55, 82, 91, 118, 127, 14, 27, 66, 79, 118,
                    131, 170, 183, 18, 35, 86, 103, 154, 171, 222, 239, 22, 43, 106,
                    127, 190, 211, 274, 295, 26, 51, 126, 151, 226, 251, 326, 351,
                    30, 59, 146, 175, 262, 291, 378, 407);
        }

        [TestMethod]
        public void Dot3412x5621()
        {
            var x = arange((3, 4, 1, 2));
            var y = arange((5, 6, 2, 1));
            np.dot(x, y).Should().BeShaped(3, 4, 1, 5, 6, 1)
                .And.BeOfValues(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173, 183, 193, 203, 213, 223, 233, 243, 253, 263, 273, 283, 293, 5, 23, 41, 59, 77, 95, 113, 131, 149, 167, 185, 203, 221, 239, 257, 275, 293, 311, 329, 347, 365, 383, 401, 419, 437, 455, 473, 491, 509, 527, 7, 33, 59, 85, 111, 137, 163, 189, 215, 241, 267, 293, 319, 345, 371, 397, 423, 449, 475, 501, 527, 553, 579, 605, 631, 657, 683, 709, 735, 761, 9, 43, 77, 111, 145, 179, 213, 247, 281, 315, 349, 383, 417, 451, 485, 519, 553, 587, 621, 655, 689, 723, 757, 791, 825, 859, 893, 927, 961, 995, 11, 53, 95, 137, 179, 221, 263, 305, 347, 389, 431, 473, 515, 557, 599, 641, 683, 725, 767, 809, 851, 893, 935, 977, 1019, 1061, 1103, 1145, 1187, 1229, 13, 63, 113, 163, 213, 263, 313, 363, 413, 463, 513, 563, 613, 663, 713, 763, 813, 863, 913, 963, 1013, 1063, 1113, 1163, 1213, 1263, 1313, 1363, 1413, 1463, 15, 73, 131, 189, 247, 305, 363, 421, 479, 537, 595, 653, 711, 769, 827, 885, 943, 1001, 1059, 1117, 1175, 1233, 1291, 1349, 1407, 1465, 1523, 1581, 1639, 1697, 17, 83, 149, 215, 281, 347, 413, 479, 545, 611, 677, 743, 809, 875, 941, 1007, 1073, 1139, 1205, 1271, 1337, 1403, 1469, 1535, 1601, 1667, 1733, 1799, 1865, 1931, 19, 93, 167, 241, 315, 389, 463, 537, 611, 685, 759, 833, 907, 981, 1055, 1129, 1203, 1277, 1351, 1425, 1499, 1573, 1647, 1721, 1795, 1869, 1943, 2017, 2091, 2165, 21, 103, 185, 267, 349, 431, 513, 595, 677, 759, 841, 923, 1005, 1087, 1169, 1251, 1333, 1415, 1497, 1579, 1661, 1743, 1825, 1907, 1989, 2071, 2153, 2235, 2317, 2399, 23, 113, 203, 293, 383, 473, 563, 653, 743, 833, 923, 1013, 1103, 1193, 1283, 1373, 1463, 1553, 1643, 1733, 1823, 1913, 2003, 2093, 2183, 2273, 2363, 2453, 2543, 2633);
        }

        [TestMethod]
        public void Dot311x511()
        {
            var x = arange((3, 1, 1));
            var y = arange((5, 1, 1));
            np.dot(x, y).Should().BeShaped(3, 1, 5, 1)
                .And.BeOfValues(0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 2, 4, 6, 8);
        }

        [TestMethod]
        public void Dot2x3And3x2()
        {
            var x = np.array(new float[,] {{0, 1, 2}, {3, 4, 5}});

            var y = np.array(new float[,] {{0, 3}, {1, 4}, {2, 5}});

            var z = np.dot(x, y);

            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<float>(), new float[] {5, 14, 14, 50}));
        }

        [TestMethod]
        public void Dot30_300x30_300()
        {
            var rng = np.random.RandomState(42);
            var a = rng.randn(30, 300);
            var b = rng.randn(300, 30);
            var c = np.dot(a, b);
        }

        /// <summary>
        ///     NumPy's N-D <c>np.dot</c> (the last route of <c>PyArray_MatrixProduct2</c>) calls the dtype's dot function once
        ///     per output element: an integer dot sums in its own width and wraps (the sum modulo 2^bits), a complex dot keeps
        ///     both parts, and float64 runs scipy-openblas' ddot, whose strided loop adds four products a step as
        ///     <c>t1 += m1 + m3; t2 += m2 + m4</c>. NumSharp's former route summed every dtype's products in double and
        ///     converted the sum. Each expected value is NumPy 2.4.2's answer, and each case gave other bits on the former
        ///     route.
        /// </summary>
        [TestMethod]
        public void Dot_ND_IsTheDtypesDotFunction()
        {
            // int64: [2**62, 3] . [4, 5] = 2**64 + 15, which wraps to 15. A double sum rounds it to 2**64, which no int64 holds.
            var i8 = np.dot(np.array(new long[] {1L << 62, 3}).reshape(1, 1, 2), np.array(new long[] {4, 5}).reshape(2, 1));
            i8.typecode.Should().Be(NPTypeCode.Int64);
            i8.GetInt64(0, 0, 0).Should().Be(15L);

            // uint16: every sum below wraps at 2**16 (65535 * 65535 alone is far past it).
            var u = np.array(new ushort[]
            {
                3, 65535, 2, 0, 5, 65532, 1, 7, 65534, 6, 4, 65533, 8, 65531, 9, 65530, 10, 65529
            }).reshape(2, 3, 3);
            var ru = np.ravel(np.dot(u, u.T));
            ru.typecode.Should().Be(NPTypeCode.UInt16);
            Enumerable.Range(0, 8).Select(i => (long)ru.GetUInt16(i)).ToArray().Should()
                .Equal(11L, 65534L, 6L, 37L, 6L, 65504L, 65532L, 64L);

            // complex128: (1+2j)(2-1j) + (3-1j)(1+1j) = 8+5j. The former route multiplied the real parts: 5.
            var c = np.array(new[] {new Complex(1, 2), new Complex(3, -1)}).reshape(1, 1, 2);
            var d = np.array(new[] {new Complex(2, -1), new Complex(1, 1)}).reshape(2, 1);
            np.dot(c, d).GetAtIndex<Complex>(0).Should().Be(new Complex(8, 5));

            // float64: a (1, 1, 6) row against a (1, 6, 2) C-ordered matrix, whose contracted axis has a stride of 2: ddot's
            // strided loop gives 0xc026b59bc33f8552, where a sequential sum gives ...8551.
            var a = np.array(new[] {1.83, 1.848, 0.092, -1.285, -2.676, -0.7}).reshape(1, 1, 6);
            var b = np.array(new[]
            {
                -0.549, -2.728, -2.707, 2.995, 0.914, -1.593, -0.39, 2.845, 2.386, 2.065, -0.646, -0.042
            }).reshape(1, 6, 2);
            var f8 = np.ravel(np.dot(a, b));
            BitConverter.DoubleToInt64Bits(f8.GetDouble(0)).Should().Be(unchecked((long)0xC026B59BC33F8552UL));
            BitConverter.DoubleToInt64Bits(f8.GetDouble(1)).Should().Be(unchecked((long)0xC0218346FDEB52CAUL));
        }

        /// <summary>
        ///     FIXED 2026-10-26 (the np-layer dtype plan's P2; the matrix tier's gap MX10, pinned in OpenBugs.Matrix.cs until
        ///     then): <c>np.dot(a, b)</c> with a (1, 4, 3) complex128 <c>a</c> and a (3, 4) complex128 <c>b</c> returns
        ///     NumPy's <c>[32-8j, 38-8j, 44-8j, 50-8j]</c> in its first row. The N-D route converted every element to float64,
        ///     which kept only the real parts. The contracted axis of <c>b</c> has a stride of 4 elements, so this case reaches
        ///     the strided form of NumPy's complex dot, where <see cref="Dot_ND_IsTheDtypesDotFunction" /> reaches the
        ///     contiguous one.
        /// </summary>
        [TestMethod]
        public void Dot_NDWithMD_Complex_KeepsTheImaginaryParts()
        {
            using var real = np.arange(12).reshape(1, 4, 3).astype(np.complex128);
            using var a = real + new Complex(0, 1);
            using var b = np.arange(12).reshape(3, 4).astype(np.complex128) * new Complex(1, -1);
            using var r = np.dot(a, b);

            r.shape.Should().Equal(1L, 4L, 4L);
            r.GetAtIndex<Complex>(0).Should().Be(new Complex(32, -8));
            r.GetAtIndex<Complex>(3).Should().Be(new Complex(50, -8));
        }
    }
}
