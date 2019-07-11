using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.UnitTest.LinearAlgebra
{
    [TestClass]
    public class NpMatMulTest
    {
        [TestMethod]
        public void TwoAndTwoInt()
        {
            var a = np.array(new int[][] {new int[] {1, 0}, new int[] {0, 1}});

            var b = np.array(new int[][] {new int[] {4, 1}, new int[] {2, 2}});

            var c = np.matmul(a, b);

            var p = new int[] {4, 1, 2, 2};
            Assert.IsTrue(Enumerable.SequenceEqual(p, c.Data<int>()));

            a = np.array(new int[][] {new int[] {1, 2}, new int[] {3, 4}});

            b = np.array(new int[][] {new int[] {5, 6}, new int[] {7, 8}});

            c = np.matmul(a, b);

            p = new int[] {19, 22, 43, 50};
            Assert.IsTrue(Enumerable.SequenceEqual(p, c.Data<int>()));
        }
    }
}
