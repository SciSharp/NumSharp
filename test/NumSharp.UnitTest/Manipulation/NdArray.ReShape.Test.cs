using System;
using System.Runtime.CompilerServices;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;
using static NumSharp.Slice;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class NdArrayReShapeTest
    {
        [TestMethod]
        public void ReShape()
        {
            var nd = np.arange(6);
            var n1 = np.reshape(nd, 3, 2);
            var n = n1.MakeGeneric<int>();

            Assert.IsTrue(n[0, 0] == 0);
            Assert.IsTrue(n[1, 1] == 3);
            Assert.IsTrue(n[2, 1] == 5);

            n = np.reshape(np.arange(6), 2, 3, 1).MakeGeneric<int>();
            Assert.IsTrue(n[1, 1, 0] == 4);
            Assert.IsTrue(n[1, 2, 0] == 5);

            n = np.reshape(np.arange(12), 2, 3, 2).MakeGeneric<int>();
            Assert.IsTrue(n[0, 0, 1] == 1);
            Assert.IsTrue(n[1, 0, 1] == 7);
            Assert.IsTrue(n[1, 1, 0] == 8);

            n = np.reshape(np.arange(12), 3, 4).MakeGeneric<int>();
            Assert.IsTrue(n[1, 1] == 5);
            Assert.IsTrue(n[2, 0] == 8);

            n = np.reshape(n, 2, 6).MakeGeneric<int>();

            Assert.IsTrue(n[1, 0] == 6);
        }

        /// <summary>
        /// numpy allow us to give one of new shape parameter as -1 (eg: (2,-1) or (-1,3) but not (-1, -1)). 
        /// It simply means that it is an unknown dimension and we want numpy to figure it out. 
        /// And numpy will figure this by looking at the 'length of the array and remaining dimensions' and making sure it satisfies the above mentioned criteria
        /// </summary>
        [TestMethod]
        public void ReshapeNegative()
        {
            NDArray nd;
            nd = np.arange(12).reshape(-1, 2);
            Assert.IsTrue(nd.shape[0] == 6);
            Assert.IsTrue(nd.shape[1] == 2);

            nd = np.arange(12).reshape(new Shape(-1, 2));
            Assert.IsTrue(nd.shape[0] == 6);
            Assert.IsTrue(nd.shape[1] == 2);

            nd = np.arange(12).reshape(2, -1);
            Assert.IsTrue(nd.shape[0] == 2);
            Assert.IsTrue(nd.shape[1] == 6);

            nd = np.arange(12).reshape(1, 3, 4);
            nd = nd.reshape(-1, 3);
            Assert.IsTrue(nd.shape[0] == 4);
            Assert.IsTrue(nd.shape[1] == 3);

            nd = np.arange(12).MakeGeneric<int>();
            nd = nd.reshape(1, 3, 4);
            nd = nd.reshape(3, -1);
            Assert.IsTrue(nd.shape[0] == 3);
            Assert.IsTrue(nd.shape[1] == 4);

            nd = np.arange(100 * 100 * 3).MakeGeneric<int>();
            nd = nd.reshape(100, 100, 3);
            nd = nd.reshape(-1, 3);
            Assert.IsTrue(nd.shape[0] == 10000);
            Assert.IsTrue(nd.shape[1] == 3);

            /*np.arange(15801033);
            np.reshape(2531, 2081, 3);
            np.reshape(-1, 3);
            Assert.IsTrue(np.shape[0] == 5267011);
            Assert.IsTrue(np.shape[1] == 3);*/
        }

        [TestMethod]
        public void ValueTest()
        {
            var x = np.arange(4).MakeGeneric<int>();
            var y = x.reshape(2, 2).MakeGeneric<int>();
            y[0, 1] = 8;
            Assert.AreEqual(x[1], y[0, 1]);
        }

        [TestMethod]
        public void TwoNegativeMinusOne()
        {
            var x = np.arange(9).reshape(3, 1, 1, 3);
            new Action(() => x.reshape(-1, 3, -1)).Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Case1_negativeone()
        {
            var x = np.full(2, (3, 3, 1, 1, 3));
            x.reshape((-1, 3)).shape.Should().BeEquivalentTo(9, 3);
        }

        [TestMethod]
        public void Case2_Slice()
        {
            var a = arange((3, 2, 2));
            a = a[0, All];
            a.Should().BeShaped(2, 2).And.BeOfValues(0, 1, 2, 3);

            a = a.reshape(1, 4);
            a.Should().BeShaped(1, 4).And.BeOfValues(0, 1, 2, 3);
            a[0, 2].Should().BeScalar(2);
        }

        [TestMethod, Ignore("Broadcasting and then using -1 during resahping is not supported (has TODO).")]
        public void Case2_Slice_Broadcast()
        {
            //alloc
            var a = arange((3, 2, 2)); //ends up 2, 2
            var b = arange((2, 2, 2));
            //slice
            a = a[0, All];
            a.Should().BeShaped(2, 2).And.BeOfValues(0, 1, 2, 3);
            //broadcast
            (a, b) = np.broadcast_arrays(a, b);
            a.Should().BeShaped(2, 2, 2).And.BeOfValues(0, 1, 2, 3, 0, 1, 2, 3);
            b.Should().BeShaped(2, 2, 2);
            //reshape
            a = a.reshape(1, -1);
            var t = a.ToString(false);
            a.Should().BeShaped(1, 8);
            a = a.reshape(-1, 2);
            a.Should().BeShaped(1, 8);
        }

        [TestMethod]
        public void Case3_Slice_Broadcast()
        {
            //alloc
            var a = arange((2, 2, 2)); //ends up 2, 2
            var b = arange((1, 2, 2));
            Console.WriteLine(a.ToString(true));
            Console.WriteLine(b.ToString(true));
            //broadcast
            (a, b) = np.broadcast_arrays(a, b);
            a.Should().BeShaped(2, 2, 2).And.BeOfValues(0, 1, 2, 3, 4, 5, 6, 7);
            b.Should().BeShaped(2, 2, 2).And.BeOfValues(0, 1, 2, 3, 0, 1, 2, 3);
            //slice
            a = a[0, All];
            b = b[0, All];
            a.Should().BeShaped(2, 2).And.BeOfValues(0, 1, 2, 3);
            b.Should().BeShaped(2, 2).And.BeOfValues(0, 1, 2, 3);

            //reshape
            var resh = a.reshape_unsafe(1, 4);
            resh.Should().BeShaped(1, 4).And.BeOfValues(0, 1, 2, 3);
        }

        [TestMethod]
        public void Case4_Slice_Broadcast()
        {
            //alloc
            var a = arange((2, 2, 2)); //ends up 2, 2
            var b = arange((1, 2, 2));
            Console.WriteLine(a.ToString(true));
            Console.WriteLine(b.ToString(true));
            //broadcast
            (a, b) = np.broadcast_arrays(a, b);
            a.Should().BeShaped(2, 2, 2).And.BeOfValues(0, 1, 2, 3, 4, 5, 6, 7);
            b.Should().BeShaped(2, 2, 2).And.BeOfValues(0, 1, 2, 3, 0, 1, 2, 3);
            //slice
            a = a[1, All];
            b = b[0, All];
            a.Should().BeShaped(2, 2).And.BeOfValues(4, 5, 6, 7);
            b.Should().BeShaped(2, 2).And.BeOfValues(0, 1, 2, 3);

            var resh = a.reshape_unsafe(1, 4);
            resh.Should().BeShaped(1, 4).And.BeOfValues(4, 5, 6, 7);
        }

        [TestMethod]
        public void Case5_Slice_Broadcast()
        {
            //alloc
            var a = arange((2, 2, 2)); //ends up 2, 2
            var b = arange((1, 2, 2));
            Console.WriteLine(a.ToString(true));
            Console.WriteLine(b.ToString(true));
            //broadcast
            (a, b) = np.broadcast_arrays(a, b);
            a.Should().BeShaped(2, 2, 2).And.BeOfValues(0, 1, 2, 3, 4, 5, 6, 7);
            b.Should().BeShaped(2, 2, 2).And.BeOfValues(0, 1, 2, 3, 0, 1, 2, 3);
            //slice
            a = a[0, All];
            b = b[1, All];
            a.Should().BeShaped(2, 2).And.BeOfValues(0, 1, 2, 3);
            b.Should().BeShaped(2, 2).And.BeOfValues(0, 1, 2, 3);

            var resh = b.reshape_unsafe(1, 4);
            resh.Should().BeShaped(1, 4).And.BeOfValues(0, 1, 2, 3);
        }

        private NDArray arange(ITuple tuple)
        {
            var dims = new int[tuple.Length];
            var size = 1;
            for (int i = 0; i < tuple.Length; i++)
            {
                dims[i] = Convert.ToInt32(tuple[i]);
                size *= dims[i];
            }

            return np.arange(size).reshape(dims);
        }
    }
}
