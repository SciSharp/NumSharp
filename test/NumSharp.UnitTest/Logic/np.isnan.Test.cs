using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Logic
{
    [TestClass]
    public class np_isnan_Test
    {
        [Ignore("TODO: fix this test")]
        [TestMethod]
        public void np_isnan_1D()
        {
            var np1 = new NDArray(new[] {1.0, Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN});
            var np2 = new NDArray(new[] {double.NegativeInfinity, double.PositiveInfinity, Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN});
            var np3 = new NDArray(new int[] {333, 444});

            Assert.IsTrue(Enumerable.SequenceEqual(np.isnan(np1).Data<bool>(), new[] {false, false, false, false, false, false, true}));
            Assert.AreEqual(7, np.isnan(np1).size);
            Assert.AreEqual(1, np.isnan(np1).ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(np.isnan(np2).Data<bool>(), new[] {false, false, false, false, false, false, false, true}));
            Assert.AreEqual(8, np.isnan(np2).size);
            Assert.AreEqual(1, np.isnan(np2).ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(np.isnan(np3).Data<bool>(), new[] {false, false}));
            Assert.AreEqual(2, np.isnan(np3).size);
            Assert.AreEqual(1, np.isnan(np3).ndim);
        }

        [Ignore("TODO: fix this test")]
        [TestMethod]
        public void np_isnan_2D()
        {
            var np1 = new NDArray(new[] {Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN}, new Shape(2, 3));
            var np2 = new NDArray(typeof(double), new Shape(39, 17));
            var np3 = new NDArray(new[] {double.NegativeInfinity, double.PositiveInfinity, Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN}, new Shape(2, 4));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isnan(np1).Data<bool>(), new[] {false, false, false, false, false, true}));
            Assert.AreEqual(6, np.isnan(np1).size);
            Assert.AreEqual(2, np.isnan(np1).ndim);
            Assert.IsFalse(np.all(np.isnan(np2)));
            Assert.AreEqual(39 * 17, np.isnan(np2).size);
            Assert.AreEqual(2, np.isnan(np2).ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(np.isnan(np3).Data<bool>(), new[] {false, false, false, false, false, false, false, true}));
            Assert.AreEqual(8, np.isnan(np3).size);
            Assert.AreEqual(2, np.isnan(np3).ndim);
        }
    }
}
