using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Diagnostics;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra
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
            Console.WriteLine(np.dot(x, y).ToString(false));
            np.dot(x, y).Should().BeShaped(2, 2).And.BeOfValues(7, 11, 4, 7);
        }

        [TestMethod]
        public void Dot2222x2222()
        {
            var x = arange((2, 2, 2, 2));
            var y = arange((2, 2, 2, 2));
            Console.WriteLine(np.dot(x, y).ToString(false));
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
            Console.WriteLine(np.dot(x, y).ToString(false));
            np.dot(x, y).Should().BeShaped(3, 4, 1, 5, 6, 1)
                .And.BeOfValues(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173, 183, 193, 203, 213, 223, 233, 243, 253, 263, 273, 283, 293, 5, 23, 41, 59, 77, 95, 113, 131, 149, 167, 185, 203, 221, 239, 257, 275, 293, 311, 329, 347, 365, 383, 401, 419, 437, 455, 473, 491, 509, 527, 7, 33, 59, 85, 111, 137, 163, 189, 215, 241, 267, 293, 319, 345, 371, 397, 423, 449, 475, 501, 527, 553, 579, 605, 631, 657, 683, 709, 735, 761, 9, 43, 77, 111, 145, 179, 213, 247, 281, 315, 349, 383, 417, 451, 485, 519, 553, 587, 621, 655, 689, 723, 757, 791, 825, 859, 893, 927, 961, 995, 11, 53, 95, 137, 179, 221, 263, 305, 347, 389, 431, 473, 515, 557, 599, 641, 683, 725, 767, 809, 851, 893, 935, 977, 1019, 1061, 1103, 1145, 1187, 1229, 13, 63, 113, 163, 213, 263, 313, 363, 413, 463, 513, 563, 613, 663, 713, 763, 813, 863, 913, 963, 1013, 1063, 1113, 1163, 1213, 1263, 1313, 1363, 1413, 1463, 15, 73, 131, 189, 247, 305, 363, 421, 479, 537, 595, 653, 711, 769, 827, 885, 943, 1001, 1059, 1117, 1175, 1233, 1291, 1349, 1407, 1465, 1523, 1581, 1639, 1697, 17, 83, 149, 215, 281, 347, 413, 479, 545, 611, 677, 743, 809, 875, 941, 1007, 1073, 1139, 1205, 1271, 1337, 1403, 1469, 1535, 1601, 1667, 1733, 1799, 1865, 1931, 19, 93, 167, 241, 315, 389, 463, 537, 611, 685, 759, 833, 907, 981, 1055, 1129, 1203, 1277, 1351, 1425, 1499, 1573, 1647, 1721, 1795, 1869, 1943, 2017, 2091, 2165, 21, 103, 185, 267, 349, 431, 513, 595, 677, 759, 841, 923, 1005, 1087, 1169, 1251, 1333, 1415, 1497, 1579, 1661, 1743, 1825, 1907, 1989, 2071, 2153, 2235, 2317, 2399, 23, 113, 203, 293, 383, 473, 563, 653, 743, 833, 923, 1013, 1103, 1193, 1283, 1373, 1463, 1553, 1643, 1733, 1823, 1913, 2003, 2093, 2183, 2273, 2363, 2453, 2543, 2633);
        }

        [TestMethod]
        public void Dot311x511()
        {
            var x = arange((3, 1, 1));
            var y = arange((5, 1, 1));
            Console.WriteLine(np.dot(x, y).ToString(false));
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
            var sw = new Stopwatch();
            sw.Start();
            var a = np.random.randn(30, 300);
            var b = np.random.randn(300, 30);
            var c = np.dot(a, b);
            sw.Stop();
            Console.WriteLine(sw.ElapsedMilliseconds);
        }
    }
}
