using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.Logic
{
    /// <summary>
    /// Tests for np.isposinf / np.isneginf — element-wise test for +Inf / -Inf.
    /// All expected values probed against NumPy 2.4.2.
    /// NumPy reference: numpy/lib/_ufunclike_impl.py (isinf(x) &amp; ~signbit(x) / isinf(x) &amp; signbit(x)).
    /// </summary>
    [TestClass]
    public class np_isposinf_isneginf_Test
    {
        [TestMethod]
        public void isposinf_1D()
        {
            var arr = np.array(new[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN, 0.0, -1.0 });
            Assert.IsTrue(Enumerable.SequenceEqual(np.isposinf(arr).Data<bool>(),
                new[] { true, false, false, false, false }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isneginf(arr).Data<bool>(),
                new[] { false, true, false, false, false }));
        }

        [TestMethod]
        public void isposinf_isneginf_ReturnsBoolDtype()
        {
            var arr = np.array(new[] { double.PositiveInfinity, 1.0 });
            Assert.AreEqual(typeof(bool), np.isposinf(arr).dtype);
            Assert.AreEqual(typeof(bool), np.isneginf(arr).dtype);
        }

        [TestMethod]
        public void isposinf_Float32()
        {
            var arr = np.array(new[] { float.PositiveInfinity, float.NegativeInfinity, float.NaN, 1.0f });
            Assert.IsTrue(Enumerable.SequenceEqual(np.isposinf(arr).Data<bool>(), new[] { true, false, false, false }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isneginf(arr).Data<bool>(), new[] { false, true, false, false }));
        }

        [TestMethod]
        public void isposinf_Half()
        {
            var arr = np.array(new[] { Half.PositiveInfinity, Half.NegativeInfinity, Half.NaN, (Half)1 });
            Assert.IsTrue(Enumerable.SequenceEqual(np.isposinf(arr).Data<bool>(), new[] { true, false, false, false }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isneginf(arr).Data<bool>(), new[] { false, true, false, false }));
        }

        [TestMethod]
        public void isposinf_2D()
        {
            var arr = np.array(new[] {
                double.PositiveInfinity, 1.0, double.NaN,
                double.NegativeInfinity, 2.0, double.MaxValue
            }).reshape(2, 3);
            var pos = np.isposinf(arr);
            Assert.AreEqual(2, pos.ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(pos.Data<bool>(), new[] { true, false, false, false, false, false }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isneginf(arr).Data<bool>(), new[] { false, false, false, true, false, false }));
        }

        [TestMethod]
        public void isposinf_IntegerAndBool_AlwaysFalse()
        {
            Assert.IsFalse(np.any(np.isposinf(np.array(new[] { 1, -2, 3, int.MaxValue, int.MinValue }))));
            Assert.IsFalse(np.any(np.isneginf(np.array(new[] { 1, -2, 3, int.MaxValue, int.MinValue }))));
            Assert.IsFalse(np.any(np.isposinf(np.array(new byte[] { 0, 1, 255 }))));
            Assert.IsFalse(np.any(np.isposinf(np.array(new[] { true, false }))));
            Assert.IsFalse(np.any(np.isneginf(np.array(new[] { true, false }))));
        }

        [TestMethod]
        public void isposinf_Decimal_AlwaysFalse()
        {
            // Decimal has no Inf domain (always finite).
            Assert.IsFalse(np.any(np.isposinf(np.array(new[] { 1m, -2m, decimal.MaxValue }))));
            Assert.IsFalse(np.any(np.isneginf(np.array(new[] { 1m, -2m, decimal.MinValue }))));
        }

        [TestMethod]
        public void isposinf_Scalar()
        {
            Assert.IsTrue(np.isposinf(np.array(double.PositiveInfinity)).GetBoolean());
            Assert.IsFalse(np.isposinf(np.array(double.NegativeInfinity)).GetBoolean());
            Assert.IsFalse(np.isposinf(np.array(double.NaN)).GetBoolean());
            Assert.IsTrue(np.isneginf(np.array(double.NegativeInfinity)).GetBoolean());
            Assert.IsFalse(np.isneginf(np.array(double.PositiveInfinity)).GetBoolean());
        }

        [TestMethod]
        public void isposinf_Empty()
        {
            var pos = np.isposinf(np.array(new double[0]));
            Assert.AreEqual(0, pos.size);
            Assert.AreEqual(typeof(bool), pos.dtype);
        }

        [TestMethod]
        public void isposinf_ReversedView()
        {
            // Non-contiguous (negative-stride) source read in logical order.
            var arr = np.array(new[] { double.PositiveInfinity, 1.0, double.NegativeInfinity, 2.0, double.NaN });
            var rev = arr["::-1"]; // [nan, 2, -inf, 1, inf]
            Assert.IsTrue(Enumerable.SequenceEqual(np.isposinf(rev).Data<bool>(), new[] { false, false, false, false, true }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isneginf(rev).Data<bool>(), new[] { false, false, true, false, false }));
        }

        [TestMethod]
        public void isposinf_SteppedView()
        {
            var arr = np.array(new[] { double.PositiveInfinity, 1.0, double.NegativeInfinity, 2.0, double.PositiveInfinity, 3.0 });
            var step = arr["::2"]; // [inf, -inf, inf]
            Assert.IsTrue(Enumerable.SequenceEqual(np.isposinf(step).Data<bool>(), new[] { true, false, true }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isneginf(step).Data<bool>(), new[] { false, true, false }));
        }

        [TestMethod]
        public void isposinf_MaxValueIsNotInfinity()
        {
            var arr = np.array(new[] { double.MaxValue, double.MinValue });
            Assert.IsFalse(np.any(np.isposinf(arr)));
            Assert.IsFalse(np.any(np.isneginf(arr)));
        }

        [TestMethod]
        public void isposinf_Complex_Throws()
        {
            // NumPy: signbit is ambiguous on complex -> TypeError, verbatim message.
            var c = np.array(new[] { new Complex(1, 2), new Complex(double.PositiveInfinity, 0) });
            var ex = Assert.ThrowsException<TypeError>(() => np.isposinf(c));
            Assert.AreEqual("This operation is not supported for complex128 values because it would be ambiguous.", ex.Message);
            Assert.ThrowsException<TypeError>(() => np.isneginf(c));
        }

        [TestMethod]
        public void isposinf_Out_NumericAndBool()
        {
            var x = np.array(new[] { double.NegativeInfinity, 0.0, double.PositiveInfinity });

            // numeric out: stores 0/1, returns the same instance (NumPy positional out).
            var yInt = np.array(new[] { 2, 2, 2 });
            var rInt = np.isposinf(x, yInt);
            Assert.IsTrue(ReferenceEquals(rInt, yInt));
            Assert.IsTrue(Enumerable.SequenceEqual(yInt.Data<int>(), new[] { 0, 0, 1 }));

            // bool out
            var yBool = np.array(new[] { false, false, false });
            np.isneginf(x, yBool);
            Assert.IsTrue(Enumerable.SequenceEqual(yBool.Data<bool>(), new[] { true, false, false }));
        }
    }
}
