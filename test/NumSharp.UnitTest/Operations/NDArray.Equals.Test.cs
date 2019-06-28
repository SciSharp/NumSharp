using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Operations
{
    [TestClass]
    public class NDArrayEqualsTest
    {

        [TestMethod]
        public void IntTwo1D_NDArrayEquals()
        {
            var np0 = new NDArray(new[] { 0, 0, 0, 0 }, new Shape(4));
            var np1 = new NDArray(new[] { 1, 2, 3, 4 }, new Shape(4));
            var np2 = new NDArray(new[] { 1, 2, 3, 4 }, new Shape(4));

            var np3 = np1 == np2;
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { true, true, true, true }, np3.Data<bool>()));
            var np3S = np.array_equal(np1, np2);
            Assert.IsTrue(np3S);
            
            var np4 = np0 == np2;
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { false, false, false, false }, np4.Data<bool>()));
            var np4S = np.array_equal(np0, np2);
            Assert.IsFalse(np4S);


        }

        [TestMethod]
        public void IntAnd1D_NDArrayEquals()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4 }, new Shape(4));

            var np2 = np1 == 2;
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { false, true, false, false }, np2.Data<bool>()));
        }

        [TestMethod]
        public void IntTwo2D_NDArrayEquals()
        {
            var np1 = new NDArray(typeof(int), new Shape(2, 3));
            np1.ReplaceData(new[] { 1, 2, 3, 4, 5, 6 });

            var np2 = new NDArray(typeof(int), new Shape(2, 3));
            np2.ReplaceData(new[] { 1, 2, 3, 4, 5, 6 });

            var np3 = np1 == np2;

            // expected
            var np3S = np.array_equal(np1, np2);
            Assert.IsTrue(np3S);
            var np4 = new bool[] { true, true, true, true, true, true };
            Assert.IsTrue(Enumerable.SequenceEqual(np3.Data<bool>(), np4));



            var np5 = new NDArray(typeof(int), new Shape(2, 3));
            np5.ReplaceData(new[] { 0, 0, 0, 0, 0, 0 });

            var np6 = np1 == np5;
            // expected
            var np6S = np.array_equal(np1, np5);
            Assert.IsFalse(np6S);
            var np7 = new bool[] { false, false, false, false, false, false, };
            Assert.IsTrue(Enumerable.SequenceEqual(np6.Data<bool>(), np7));

        }


        [TestMethod]
        public void IntAnd2D_NDArrayEquals()
        {
            var np1 = new NDArray(typeof(int), new Shape(2, 3));
            np1.ReplaceData(new[] { 1, 2, 3, 4, 5, 6 });

            var np2 = np1 == 2;

            // expected
            var np3 = new bool[] { false, true, false, false, false, false };
            Assert.IsTrue(Enumerable.SequenceEqual(np2.Data<bool>(), np3));
        }
    }
}
