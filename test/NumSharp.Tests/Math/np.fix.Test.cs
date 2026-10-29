using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// np.fix — round toward zero, dtype-preserving. In NumPy 2.4.2 fix(x, out) is a verbatim
    /// delegation to trunc(x, out=out) (numpy/lib/_ufunclike_impl.py), so these expectations mirror
    /// np.trunc; the tests additionally pin the fix-specific API (positional out=, no where/dtype) and
    /// that it stays an exact alias of trunc. All values verified against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class np_fix_Test
    {
        [TestMethod]
        public void Fix_Float64()
        {
            var r = np.fix(np.array(new[] { 2.1, 2.9, -2.1, -2.9, 0.0 }));
            Assert.AreEqual(typeof(double), r.dtype);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<double>(), new[] { 2.0, 2.0, -2.0, -2.0, 0.0 }));
        }

        [TestMethod]
        public void Fix_PreservesDtype()
        {
            // int / bool are identity (kept unchanged), NOT promoted to float — matching trunc.
            np.fix(np.array(new[] { 3, -3 })).dtype.Should().Be(typeof(int));
            np.fix(np.array(new sbyte[] { 3, -3 })).dtype.Should().Be(typeof(sbyte));
            np.fix(np.array(new byte[] { 3 })).dtype.Should().Be(typeof(byte));
            np.fix(np.array(new[] { true, false })).dtype.Should().Be(typeof(bool));
            np.fix(np.array(new[] { 2.7f, -2.7f })).dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        public void Fix_Half()
        {
            var r = np.fix(np.array(new[] { (Half)2.7, (Half)(-2.7) }));
            Assert.AreEqual(typeof(Half), r.dtype);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<Half>(), new[] { (Half)2.0, (Half)(-2.0) }));
        }

        [TestMethod]
        public void Fix_NanInf_PassThrough()
        {
            var r = np.fix(np.array(new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity }));
            Assert.IsTrue(double.IsNaN(r.GetDouble(0)));
            Assert.IsTrue(double.IsPositiveInfinity(r.GetDouble(1)));
            Assert.IsTrue(double.IsNegativeInfinity(r.GetDouble(2)));
        }

        [TestMethod]
        public void Fix_PreservesSignOfZero()
        {
            // trunc(-0.0) keeps the sign of zero; fix inherits it. Probe via the sign bit.
            var r = np.fix(np.array(new[] { -0.0, 0.0 }));
            Assert.IsTrue(np.signbit(r).Data<bool>().SequenceEqual(new[] { true, false }));
        }

        [TestMethod]
        public void Fix_Out_ReturnsSameInstance()
        {
            var o = np.zeros(new Shape(4), np.float64);
            var r = np.fix(np.array(new[] { 2.1, 2.9, -2.1, -2.9 }), o);
            Assert.IsTrue(ReferenceEquals(r, o));
            Assert.IsTrue(Enumerable.SequenceEqual(o.Data<double>(), new[] { 2.0, 2.0, -2.0, -2.0 }));
        }

        [TestMethod]
        public void Fix_2D()
        {
            var r = np.fix(np.array(new[] { 1.9, -1.9, 2.5, -2.5 }).reshape(2, 2));
            Assert.AreEqual(2, r.ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<double>(), new[] { 1.0, -1.0, 2.0, -2.0 }));
        }

        [TestMethod]
        public void Fix_IsExactAliasOfTrunc()
        {
            var x = np.array(new[] { 5.6, -5.6, 0.4, -0.4, 100.9, -100.9 });
            Assert.IsTrue(Enumerable.SequenceEqual(np.fix(x).Data<double>(), np.trunc(x).Data<double>()));
        }

        [TestMethod]
        public void Fix_Scalar()
        {
            Assert.AreEqual(3.0, np.fix(np.array(3.7)).GetDouble());
            Assert.AreEqual(-3.0, np.fix(np.array(-3.7)).GetDouble());
        }
    }
}
