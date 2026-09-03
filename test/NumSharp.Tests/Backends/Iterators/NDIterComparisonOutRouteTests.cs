using System;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    ///     The comparison ufunc <c>out=</c> route runs the whole-array SIMD comparison kernel
    ///     through the iterator (<c>NDIterRef.TryExecuteComparison</c>) when the inputs share a
    ///     dtype and the bool out is contiguous in iteration order, and keeps the per-chunk
    ///     scalar inner loop for everything else. Both must agree with the no-out route on
    ///     every dtype and layout — measured 43.4 µs → 9.3 µs on np.less(f64, f64, out=) at 100K.
    /// </summary>
    [TestClass]
    public class NDIterComparisonOutRouteTests
    {
        private static void AssertSameAsNoOut(NDArray lhs, NDArray rhs, NDArray @out, string what)
        {
            var expected = np.less(lhs, rhs);
            var got = np.less(lhs, rhs, @out);
            Assert.AreSame(@out, got, $"{what}: out must be returned");
            Assert.IsTrue(np.array_equal(expected, @out), $"{what}: out= must equal the no-out result");

            expected = np.equal(lhs, rhs);
            np.equal(lhs, rhs, @out);
            Assert.IsTrue(np.array_equal(expected, @out), $"{what}: equal");

            expected = np.greater_equal(lhs, rhs);
            np.greater_equal(lhs, rhs, @out);
            Assert.IsTrue(np.array_equal(expected, @out), $"{what}: greater_equal");
        }

        [TestMethod]
        public void SameDtype_Contiguous_Every_Dtype()
        {
            var codes = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
            };
            foreach (var code in codes)
            {
                var a = (np.arange(1000) % 7).astype(code);
                var b = (np.arange(1000) % 5).astype(code);
                var o = np.empty(new Shape(1000), np.bool_);
                AssertSameAsNoOut(a, b, o, code.ToString());
            }
        }

        [TestMethod]
        public void Scalar_Broadcast_Rhs_Into_Contiguous_Out()
        {
            var a = (np.arange(1003) % 11).astype(np.float64);
            var s = NDArray.Scalar(4.0, NPTypeCode.Double);
            var o = np.empty(new Shape(1003), np.bool_);
            AssertSameAsNoOut(a, s, o, "scalar rhs");
            AssertSameAsNoOut(s, a, o, "scalar lhs");
        }

        [TestMethod]
        public void Row_Broadcast_Into_Contiguous_Out()
        {
            var a = (np.arange(60) % 9).astype(np.int32).reshape(3, 4, 5);
            var row = (np.arange(5) % 3).astype(np.int32);
            var o = np.empty(new Shape(3, 4, 5), np.bool_);
            AssertSameAsNoOut(a, row, o, "row broadcast");
        }

        [TestMethod]
        public void Strided_Inputs_Into_Contiguous_Out()
        {
            var a = np.arange(200).astype(np.float64)["::2"];
            var b = np.arange(100).astype(np.float64)["::-1"];
            var o = np.empty(new Shape(100), np.bool_);
            AssertSameAsNoOut(a, b, o, "strided/reversed inputs");
        }

        [TestMethod]
        public void Strided_Out_Falls_Back_And_Is_Correct()
        {
            var a = (np.arange(50) % 7).astype(np.float64);
            var b = (np.arange(50) % 5).astype(np.float64);
            var oBase = np.zeros(new Shape(100), np.bool_);
            var o = oBase["::2"];
            AssertSameAsNoOut(a, b, o, "strided out");
            // The untouched interleaved slots keep their prior contents.
            for (int i = 1; i < 100; i += 2)
                Assert.IsFalse(oBase.GetBoolean(i));
        }

        [TestMethod]
        public void F_Order_Out_Falls_Back_And_Is_Correct()
        {
            var a = (np.arange(12) % 7).astype(np.float64).reshape(3, 4);
            var b = (np.arange(12) % 5).astype(np.float64).reshape(3, 4);
            var o = np.empty(new Shape(4, 3), np.bool_).T;   // F-contiguous (3,4)
            AssertSameAsNoOut(a, b, o, "F-order out");
        }

        [TestMethod]
        public void Mixed_Dtypes_Compare_At_Common_Type()
        {
            // probed A3: greater(i8 2^53+1, f8 2^53) → False, equal → True (both cast to f64 first)
            var i = np.array(new long[] { (1L << 53) + 1, 5 });
            var f = np.array(new double[] { 1L << 53, 4.5 });
            var o = np.empty(new Shape(2), np.bool_);
            np.greater(i, f, o);
            Assert.IsFalse(o.GetBoolean(0));
            Assert.IsTrue(o.GetBoolean(1));
            np.equal(i, f, o);
            Assert.IsTrue(o.GetBoolean(0));
            Assert.IsFalse(o.GetBoolean(1));
        }

        [TestMethod]
        public void NaN_Semantics_Match_IEEE()
        {
            var a = np.array(new[] { double.NaN, 1.0, double.NaN });
            var b = np.array(new[] { 1.0, double.NaN, double.NaN });
            var o = np.empty(new Shape(3), np.bool_);
            np.less(a, b, o);
            Assert.IsFalse(o.GetBoolean(0)); Assert.IsFalse(o.GetBoolean(1)); Assert.IsFalse(o.GetBoolean(2));
            np.not_equal(a, b, o);
            Assert.IsTrue(o.GetBoolean(0)); Assert.IsTrue(o.GetBoolean(1)); Assert.IsTrue(o.GetBoolean(2));
            np.equal(a, b, o);
            Assert.IsFalse(o.GetBoolean(2));
        }

        [TestMethod]
        public void Bool_Out_Aliasing_An_Input_Is_Overlap_Safe()
        {
            var m1 = (np.arange(64) % 2 == 0);
            var m2 = (np.arange(64) % 3 == 0);
            var expected = np.less(m1, m2);
            var got = np.less(m1, m2, m1);   // out aliases lhs → COPY_IF_OVERLAP temp + write-back
            Assert.AreSame(m1, got);
            Assert.IsTrue(np.array_equal(expected, m1));
        }

        [TestMethod]
        public void Cast_Out_Keeps_The_Buffered_Route()
        {
            // A non-bool out is a CAST operand (buffered flush): must still be correct.
            var a = (np.arange(30) % 7).astype(np.float64);
            var b = (np.arange(30) % 5).astype(np.float64);
            var o = np.empty(new Shape(30), np.int32);
            np.less(a, b, o);
            var expected = np.less(a, b);
            for (int i = 0; i < 30; i++)
                Assert.AreEqual(expected.GetBoolean(i) ? 1 : 0, o.GetInt32(i), $"[{i}]");
        }
    }
}
