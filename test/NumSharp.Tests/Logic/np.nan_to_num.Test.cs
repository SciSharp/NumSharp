using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.Logic
{
    /// <summary>
    /// Tests for np.nan_to_num — replace NaN with 0 and +/-Inf with large finite values (or user fills).
    /// All expected values probed against NumPy 2.4.2.
    /// NumPy reference: numpy/lib/_type_check_impl.py.
    /// </summary>
    [TestClass]
    public class np_nan_to_num_Test
    {
        [TestMethod]
        public void nan_to_num_Scalars()
        {
            Assert.AreEqual(double.MaxValue, np.nan_to_num(np.array(double.PositiveInfinity)).GetDouble());
            Assert.AreEqual(double.MinValue, np.nan_to_num(np.array(double.NegativeInfinity)).GetDouble());
            Assert.AreEqual(0.0, np.nan_to_num(np.array(double.NaN)).GetDouble());
            Assert.AreEqual(5.0, np.nan_to_num(np.array(5.0)).GetDouble());
        }

        [TestMethod]
        public void nan_to_num_Float64_Defaults()
        {
            var x = np.array(new[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN, -128.0, 128.0 });
            Assert.IsTrue(Enumerable.SequenceEqual(np.nan_to_num(x).Data<double>(),
                new[] { double.MaxValue, double.MinValue, 0.0, -128.0, 128.0 }));
        }

        [TestMethod]
        public void nan_to_num_Float32_Defaults()
        {
            var x = np.array(new[] { float.PositiveInfinity, float.NegativeInfinity, float.NaN, -128f, 128f });
            Assert.IsTrue(Enumerable.SequenceEqual(np.nan_to_num(x).Data<float>(),
                new[] { float.MaxValue, float.MinValue, 0f, -128f, 128f }));
        }

        [TestMethod]
        public void nan_to_num_Float16_Defaults()
        {
            var x = np.array(new[] { Half.PositiveInfinity, Half.NegativeInfinity, Half.NaN, (Half)(-128), (Half)128 });
            Assert.IsTrue(Enumerable.SequenceEqual(np.nan_to_num(x).Data<Half>(),
                new[] { Half.MaxValue, Half.MinValue, (Half)0, (Half)(-128), (Half)128 }));
        }

        [TestMethod]
        public void nan_to_num_IntegerReturnedUnchanged()
        {
            var xi = np.array(new[] { 1, 2, 3 });
            var ri = np.nan_to_num(xi);
            Assert.AreEqual(typeof(int), ri.dtype);
            Assert.IsTrue(Enumerable.SequenceEqual(ri.Data<int>(), new[] { 1, 2, 3 }));
            // copy=True -> a distinct array; copy=False -> the same instance.
            Assert.IsFalse(ReferenceEquals(ri, xi));
            Assert.IsTrue(ReferenceEquals(np.nan_to_num(xi, copy: false), xi));
        }

        [TestMethod]
        public void nan_to_num_BoolReturnedUnchanged()
        {
            var xb = np.array(new[] { true, false, true });
            Assert.IsTrue(Enumerable.SequenceEqual(np.nan_to_num(xb).Data<bool>(), new[] { true, false, true }));
        }

        [TestMethod]
        public void nan_to_num_CopySemantics()
        {
            var xf = np.array(new[] { 1.0, double.NaN, 2.0 });

            // copy=True (default): original untouched.
            var rc = np.nan_to_num(xf);
            Assert.IsTrue(double.IsNaN(xf.GetDouble(1)));
            Assert.IsFalse(ReferenceEquals(rc, xf));

            // copy=False: mutates in place, returns the same instance.
            var rc2 = np.nan_to_num(xf, copy: false);
            Assert.IsTrue(ReferenceEquals(rc2, xf));
            Assert.AreEqual(0.0, xf.GetDouble(1));
        }

        [TestMethod]
        public void nan_to_num_CustomScalarFills()
        {
            var x = np.array(new[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN, -128.0, 128.0 });
            Assert.IsTrue(Enumerable.SequenceEqual(
                np.nan_to_num(x, nan: -9999, posinf: 33333333, neginf: 33333333).Data<double>(),
                new[] { 33333333.0, 33333333.0, -9999.0, -128.0, 128.0 }));
        }

        [TestMethod]
        public void nan_to_num_IndividualFills()
        {
            var x = np.array(new[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN, -128.0, 128.0 });

            Assert.IsTrue(Enumerable.SequenceEqual(np.nan_to_num(x, posinf: 1.0).Data<double>(),
                new[] { 1.0, double.MinValue, 0.0, -128.0, 128.0 }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.nan_to_num(x, neginf: -1.0).Data<double>(),
                new[] { double.MaxValue, -1.0, 0.0, -128.0, 128.0 }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.nan_to_num(x, nan: 7.0).Data<double>(),
                new[] { double.MaxValue, double.MinValue, 7.0, -128.0, 128.0 }));
        }

        [TestMethod]
        public void nan_to_num_ArrayFills()
        {
            var x = np.array(new[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN, -128.0, 128.0 });
            var nan = np.array(new[] { 11.0, 12.0, -9999.0, 13.0, 14.0 });
            var posinf = np.array(new[] { 33333333.0, 11.0, 12.0, 13.0, 14.0 });
            var neginf = np.array(new[] { 11.0, 33333333.0, 12.0, 13.0, 14.0 });
            Assert.IsTrue(Enumerable.SequenceEqual(
                np.nan_to_num(x, nan: nan, posinf: posinf, neginf: neginf).Data<double>(),
                new[] { 33333333.0, 33333333.0, -9999.0, -128.0, 128.0 }));
        }

        [TestMethod]
        public void nan_to_num_Complex_Default()
        {
            var y = np.array(new[]
            {
                new Complex(double.PositiveInfinity, double.NaN),
                new Complex(double.NaN, 0),
                new Complex(double.NaN, double.PositiveInfinity)
            });
            var r = np.nan_to_num(y).Data<Complex>().ToArray();
            Assert.AreEqual(new Complex(double.MaxValue, 0), r[0]);
            Assert.AreEqual(new Complex(0, 0), r[1]);
            Assert.AreEqual(new Complex(0, double.MaxValue), r[2]);
        }

        [TestMethod]
        public void nan_to_num_Complex_ScalarFills()
        {
            var y = np.array(new[]
            {
                new Complex(double.PositiveInfinity, double.NaN),
                new Complex(double.NaN, 0),
                new Complex(double.NaN, double.PositiveInfinity)
            });
            var r = np.nan_to_num(y, nan: 111111, posinf: 222222).Data<Complex>().ToArray();
            Assert.AreEqual(new Complex(222222, 111111), r[0]);
            Assert.AreEqual(new Complex(111111, 0), r[1]);
            Assert.AreEqual(new Complex(111111, 222222), r[2]);
        }

        [TestMethod]
        public void nan_to_num_Complex_ArrayFills()
        {
            var y = np.array(new[]
            {
                new Complex(double.PositiveInfinity, double.NaN),
                new Complex(double.NaN, 0),
                new Complex(double.NaN, double.PositiveInfinity)
            });
            var nan = np.array(new[] { 11.0, 12.0, 13.0 });
            var posinf = np.array(new[] { 21.0, 22.0, 23.0 });
            var neginf = np.array(new[] { 31.0, 32.0, 33.0 });
            var r = np.nan_to_num(y, nan: nan, posinf: posinf, neginf: neginf).Data<Complex>().ToArray();
            Assert.AreEqual(new Complex(21, 11), r[0]);
            Assert.AreEqual(new Complex(12, 0), r[1]);
            Assert.AreEqual(new Complex(13, 23), r[2]);
        }

        [TestMethod]
        public void nan_to_num_2D()
        {
            var m = np.array(new[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN, 1.0, 2.0, 3.0 }).reshape(2, 3);
            var r = np.nan_to_num(m);
            Assert.AreEqual(2, r.ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<double>(),
                new[] { double.MaxValue, double.MinValue, 0.0, 1.0, 2.0, 3.0 }));
        }

        [TestMethod]
        public void nan_to_num_NonContiguous_InPlace()
        {
            // copy=False on a reversed (negative-stride) view: composition path writes through.
            var v = np.array(new[] { double.PositiveInfinity, 1.0, double.NegativeInfinity, 2.0, double.NaN, 3.0 });
            var rev = v["::-1"];
            var r = np.nan_to_num(rev, copy: false);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<double>(),
                new[] { 3.0, 0.0, 2.0, double.MinValue, 1.0, double.MaxValue }));
            // base v mutated through the view.
            Assert.IsTrue(Enumerable.SequenceEqual(v.Data<double>(),
                new[] { double.MaxValue, 1.0, double.MinValue, 2.0, 0.0, 3.0 }));
        }

        [TestMethod]
        public void nan_to_num_Empty()
        {
            var r = np.nan_to_num(np.array(new double[0]));
            Assert.AreEqual(0, r.size);
            Assert.AreEqual(typeof(double), r.dtype);
        }
    }
}
