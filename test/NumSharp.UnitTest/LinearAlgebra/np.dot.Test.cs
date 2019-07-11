using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Diagnostics;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    /// Test concolve with standard example from 
    /// https://www.numpy.org/devdocs/reference/generated/numpy.convolve.html
    /// </summary>
    [TestClass]
    public class NdArrayDotTest
    {
        [TestMethod]
        public void Dot0X0()
        {
            int x = 2;
            int y = 3;
            int z = np.dot(x, y);

            Assert.AreEqual(z, 6);
        }

        [TestMethod]
        public void Dot1x1()
        {
            var x = np.arange(3);
            var y = np.arange(3, 6);

            int nd3 = np.dot(x, y);
            Assert.IsTrue(nd3 == 14);
        }

        [TestMethod]
        public void Dot2x1()
        {
            var x = np.array(new int[,] {{1, 1}, {1, 2}, {2, 2}, {2, 3}});

            var y = np.array(new int[] {2, 3});

            var z = np.dot(x, y);

            Assert.AreEqual(z.Data<int>(0), 5);
            Assert.AreEqual(z.Data<int>(1), 8);
            Assert.AreEqual(z.Data<int>(2), 10);
            Assert.AreEqual(z.Data<int>(3), 13);
        }

        [TestMethod]
        public void Dot2x2()
        {
            var x = np.array(new float[,] {{3, 1}, {1, 2}});

            var y = np.array(new float[,] {{2, 3}, {1, 2}});

            var z = np.dot(x, y);

            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<float>(), new float[] {7, 11, 4, 7}));
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
        public void DotRandn()
        {
            var sw = new Stopwatch();
            sw.Start();
            var a = np.random.randn(300, 300);
            var b = np.random.randn(300, 300);
            var c = np.dot(a, b);
            sw.Stop();
            Console.WriteLine(sw.ElapsedMilliseconds);
        }
    }
}
