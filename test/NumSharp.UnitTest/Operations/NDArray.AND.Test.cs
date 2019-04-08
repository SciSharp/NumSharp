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
    public class NDArrayAndTest
    {

        [TestMethod]
        public void BoolTwo1D_NDArrayAND()
        {
            var np1 = new NDArray(new [] { true, true, false, false }, new Shape(4));
            var np2 = new NDArray(new[] { true, false, true, false }, new Shape(4));

            var np3 = np1 & np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new [] { true, false, false, false }, np3.Data<bool>()));
        }

        [TestMethod]
        public void BoolTwo2D_NDArrayAND()
        {
            var np1 = new NDArray(typeof(bool), new Shape(2, 3));
            np1.SetData(new bool[] { true, true, false, false, true, false });

            var np2 = new NDArray(typeof(bool), new Shape(2, 3));
            np2.SetData(new bool[] { true, false, true, false, true, true });
            
            var np3 = np1 & np2;

            // expected
            var np4 = new bool[] { true, false, false, false, true, false };

            Assert.IsTrue(Enumerable.SequenceEqual(np3.Data<bool>(), np4));
        }
    }
}
