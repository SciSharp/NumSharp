using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;


namespace NumSharp.UnitTest.Logic
{
    [TestClass]
    public class np_isfinite_Test
    {
        [TestMethod]
        public void np_isfinite_1D()
        {
            var np1 = new NDArray(new[] {1.0, Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN});
            var np2 = new NDArray(new[] {double.NegativeInfinity, double.PositiveInfinity, Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN});
            var np3 = new NDArray(new int[] {333, 444});

            Assert.IsTrue(Enumerable.SequenceEqual(np.isfinite(np1).Data<bool>(), new[] {true, true, true, true, true, true, false}));
            Assert.AreEqual(7, np.isfinite(np1).size);
            Assert.AreEqual(1, np.isfinite(np1).ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(np.isfinite(np2).Data<bool>(), new[] {false, false, true, true, true, true, true, false}));
            Assert.AreEqual(8, np.isfinite(np2).size);
            Assert.AreEqual(1, np.isfinite(np2).ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(np.isfinite(np3).Data<bool>(), new[] {true, true}));
            Assert.AreEqual(2, np.isfinite(np3).size);
            Assert.AreEqual(1, np.isfinite(np3).ndim);
        }

        [TestMethod]
        public void np_isfinite_2D()
        {
            var np1 = new NDArray(new[] {Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN}, new Shape(2, 3));
            var np2 = new NDArray(typeof(double), new Shape(39, 17));
            var np3 = new NDArray(new[] {double.NegativeInfinity, double.PositiveInfinity, Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN}, new Shape(2, 4));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isfinite(np1).Data<bool>(), new[] {true, true, true, true, true, false}));
            Assert.AreEqual(6, np.isfinite(np1).size);
            Assert.AreEqual(2, np.isfinite(np1).ndim);
            Assert.IsTrue(np.all(np.isfinite(np2)));
            Assert.AreEqual(39 * 17, np.isfinite(np2).size);
            Assert.AreEqual(2, np.isfinite(np2).ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(np.isfinite(np3).Data<bool>(), new[] {false, false, true, true, true, true, true, false}));
            Assert.AreEqual(8, np.isfinite(np3).size);
            Assert.AreEqual(2, np.isfinite(np3).ndim);
        }
    }
}
