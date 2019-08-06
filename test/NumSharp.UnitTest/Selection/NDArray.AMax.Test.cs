using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using FluentAssertions;

namespace NumSharp.UnitTest.Selection
{
    [TestClass]
    public class NDArrayAMaxTest
    {
        [TestMethod]
        public void argmax12()
        {
            NDArray x = DataSample.Int32D12;
            Console.WriteLine(x.ToString(false));
            int y0 = np.argmax(x);
            Assert.AreEqual(y0, 3);
        }

        [TestMethod]
        public void argmax_case1()
        {
            var a = np.arange(27).reshape(3, 3, 3);
            np.argmax(a).Should().Be(26);
        }

        [TestMethod]
        public void argmax_case2()
        {
            var a = np.arange(27).reshape(3, 3, 3);
            np.argmax(a, axis: 1).Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [TestMethod]
        public void argmax_case3()
        {
            var a = np.arange(27).reshape(3, 3, 3);
            np.argmax(a, axis: 0).Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [TestMethod]
        public void argmax_case4()
        {
            var a = np.arange(27).reshape(3, 3, 3);
            np.argmax(a, axis: 2).Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [TestMethod]
        public void argmax_case5()
        {
            var a = np.arange(6).reshape(2, 3) + 10;
            np.argmax(a).Should().Be(5);
            var ret = np.argmax(a, axis: 0);
            ret.Cast<int>().Should().AllBeEquivalentTo(1);
            ret.size.Should().Be(3);
            ret.shape.Should().HaveCount(1).And.ContainInOrder(3);
        }

        [TestMethod]
        public void argmax_case6()
        {
            var a = np.arange(6).reshape(2, 3) + 10;
            np.argmax(a).Should().Be(5);
            var ret = np.argmax(a, axis: 1);
            ret.Cast<int>().Should().AllBeEquivalentTo(2);
            ret.size.Should().Be(2);
            ret.shape.Should().HaveCount(1).And.ContainInOrder(2);
        }

        [TestMethod]
        public void argmin_case5()
        {
            var a = np.arange(6).reshape(2, 3) + 10;
            np.argmin(a).Should().Be(0);
            var ret = np.argmin(a, axis: 0);
            ret.Cast<int>().Should().AllBeEquivalentTo(0);
            ret.size.Should().Be(3);
            ret.shape.Should().HaveCount(1).And.ContainInOrder(3);
        }

        [TestMethod]
        public void argmin_case6()
        {
            var a = np.arange(6).reshape(2, 3) + 10;
            np.argmin(a).Should().Be(0);
            var ret = np.argmin(a, axis: 1);
            ret.Cast<int>().Should().AllBeEquivalentTo(0);
            ret.size.Should().Be(2);
            ret.shape.Should().HaveCount(1).And.ContainInOrder(2);
        }

        [TestMethod]
        public void argmin_case2()
        {
            var a = np.arange(27).reshape(3, 3, 3);
            np.argmin(a, axis: 1).Cast<int>().Should().AllBeEquivalentTo(0);
        }

        [TestMethod]
        public void argmin_case3()
        {
            var a = np.arange(27).reshape(3, 3, 3);
            np.argmin(a, axis: 0).Cast<int>().Should().AllBeEquivalentTo(0);
        }

        [TestMethod]
        public void argmin_case4()
        {
            var a = np.arange(27).reshape(3, 3, 3);
            np.argmin(a, axis: 2).Cast<int>().Should().AllBeEquivalentTo(0);
        }

        [TestMethod]
        public void argmin_case1()
        {
            var a = np.arange(27).reshape(3, 3, 3);
            np.argmin(a).Should().Be(0);
        }

        [TestMethod]
        public void argmax4x3()
        {
            NDArray x = DataSample.Int32D4x3;

            var y0 = np.argmax(x, 0);
            Assert.IsTrue(Enumerable.SequenceEqual(y0.Data<int>(), new int[] {0, 3, 2}));

            var y1 = np.argmax(x, 1);
            Assert.IsTrue(Enumerable.SequenceEqual(y1.Data<int>(), new int[] {0, 1, 2, 1}));
        }

        [TestMethod]
        public void amax()
        {
            //default type
            var n = np.arange(0, 12, 0.1);
            double d1 = np.amax<double>(n);
            Assert.IsTrue(d1.Equals(11.9));

            //no axis
            n = np.arange(4).reshape(2, 2);
            var max = np.amax<int>(n);

            Assert.IsTrue(max == 3);

            //2D with axis
            var n1 = np.amax(n, 0).MakeGeneric<int>();
            Assert.IsTrue(n1.GetAtIndex(0) == 2);
            Assert.IsTrue(n1.GetAtIndex(1) == 3);

            n1 = np.amax(n, 1).MakeGeneric<int>();
            Assert.IsTrue(n1[0] == 1);
            Assert.IsTrue(n1[1] == 3);

            //3D
            /*n = np.arange(24).reshape(4, 3, 2);
            n1 = np.amax(n, 0).MakeGeneric<int>();
            Assert.IsTrue(n1[0, 1] == 19);
            Assert.IsTrue(n1[2, 1] == 23);
            Assert.IsTrue(n1[1, 1] == 21);
            n1 = np.amax(n, 1).MakeGeneric<int>();
            Assert.IsTrue(n1[1, 1] == 11);
            Assert.IsTrue(n1[2, 1] == 17);
            Assert.IsTrue(n1[3, 0] == 22);

            //4D
            n = np.arange(24).reshape(2, 3, 2, 2);
            n1 = np.amax(n, 1).MakeGeneric<int>();
            Assert.IsTrue(n1[0, 0, 1] == 9);
            Assert.IsTrue(n1[1, 0, 1] == 21);
            Assert.IsTrue(n1[1, 1, 1] == 23);
            n1 = np.amax(n, 3).MakeGeneric<int>();
            Assert.IsTrue(n1[0, 1, 1] == 7);
            Assert.IsTrue(n1[1, 1, 1] == 19);
            Assert.IsTrue(n1[1, 2, 1] == 23);*/
        }
    }
}
