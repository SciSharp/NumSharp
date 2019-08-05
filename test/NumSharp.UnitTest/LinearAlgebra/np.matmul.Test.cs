using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra
{
    [TestClass]
    public class np_matmul_test
    {
        [TestMethod]
        public void Case1_2_2()
        {
            var a = np.arange(9).reshape(3, 3);
            var b = np.arange(9).reshape(3, 3);

            var ret = np.matmul(a, b);
            Console.WriteLine(ret.typecode);
            ret.flat.AsIterator<int>().ToArray().Should().BeEquivalentTo(15, 18, 21, 42, 54, 66, 69, 90, 111);
        }

        [TestMethod]
        public void Case2_2_2()
        {
            var a = np.full(2, (3, 3));
            var b = np.full(2, (3, 3));
            var ret = np.matmul(a, b);
            Console.WriteLine(ret.typecode);
            ret.flat.AsIterator().Cast<object>().Distinct().ToArray().Should().Contain(12).And.HaveCount(1);
        }

        [TestMethod]
        public void Case1_2_1()
        {
            var a = np.arange(9).reshape(3, 3);
            var b = np.arange(3).reshape(3);

            var ret = np.matmul(a, b);
            Console.WriteLine(ret.typecode);
            ret.flat.AsIterator<int>().ToArray().Should().BeEquivalentTo(5, 14, 23);
        }

        [TestMethod]
        public void Case2_2_1()
        {
            var a = np.full(2, (3, 3));
            var b = np.full(3, (3));
            var ret = np.matmul(a, b);
            Console.WriteLine(ret.typecode);
            ret.flat.AsIterator().Cast<object>().Distinct().ToArray().Should().Contain(18).And.HaveCount(1);
        }

        [TestMethod]
        public void Case_3_2_2__3_2_2()
        {
            var a = np.full(2, (3, 2, 2));
            var b = np.full(3, (3, 2, 2));
            var ret = np.matmul(a, b);
            ret.Should().AllValuesBe(12).And.BeShaped(3, 2, 2);
        }

        [TestMethod]
        public void Case_3_1_2_2__3_2_2()
        {
            var a = np.full(2, (3, 1, 2, 2));
            var b = np.full(3, (3, 2, 2));
            var ret = np.matmul(a, b);
            ret.Should().AllValuesBe(12).And.BeShaped(3, 3, 2, 2);
        }

        [TestMethod]
        public void Case_3_1_2_2__3_2_2_Arange()
        {
            var a = np.arange(2 * 1 * 2 * 2).reshape((2, 1, 2, 2));
            var b = np.arange(2 * 2 * 2).reshape((2, 2, 2));
            var ret = np.matmul(a, b);
            ret.Should().BeOfValues(2, 3, 6, 11, 6, 7, 26, 31, 10, 19, 14, 27, 46, 55, 66, 79).And.BeShaped(2, 2, 2, 2);
        }

        [TestMethod]
        public void Case1_3_1_vs_1_3()
        {
            var a = np.arange(3).reshape(3, 1);
            var b = np.arange(3).reshape(1, 3);

            var ret = np.matmul(a, b);
            Console.WriteLine(ret.typecode);
            ret.flat.AsIterator<int>().ToArray().Should().BeEquivalentTo(0, 0, 0, 0, 1, 2, 0, 2, 4);
        }

        [TestMethod]
        public void Case2_3_1_vs_1_3()
        {
            var a = np.full(2, (3, 1));
            var b = np.full(2, (1, 3));
            var ret = np.matmul(a, b);
            Console.WriteLine(ret.typecode);
            ret.flat.AsIterator().Cast<object>().Distinct().ToArray().Should().Contain(4).And.HaveCount(1);
        }

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
