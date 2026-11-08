using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.Logic
{
    /// <summary>
    /// Tests for np.signbit — element-wise test of the IEEE sign bit. All expected values probed
    /// against NumPy 2.4.2. NumPy reference: the <c>signbit</c> ufunc (numpy/_core). signbit is the
    /// primitive isposinf/isneginf are defined on (isinf(x) &amp; ~signbit(x) / isinf(x) &amp; signbit(x)).
    /// It is NOT <c>x &lt; 0</c>: it reads the raw sign bit, so -0.0 and a negative NaN are True.
    /// </summary>
    [TestClass]
    public class np_signbit_Test
    {
        // Controlled NaN bit patterns — C#'s double.NaN is the NEGATIVE quiet NaN (0xfff8…), and
        // `-double.NaN` is not a reliable way to get a positive one, so build both by bits.
        private static readonly double PosQNaN = BitConverter.Int64BitsToDouble(0x7ff8000000000000L);
        private static readonly double NegQNaN = BitConverter.Int64BitsToDouble(unchecked((long)0xfff8000000000000UL));

        [TestMethod]
        public void signbit_1D_Float64()
        {
            var arr = np.array(new[] { -0.0, 0.0, double.NegativeInfinity, double.PositiveInfinity, -3.0, 3.0 });
            Assert.IsTrue(Enumerable.SequenceEqual(np.signbit(arr).Data<bool>(),
                new[] { true, false, true, false, true, false }));
        }

        [TestMethod]
        public void signbit_NaN_Sign()
        {
            // Positive NaN → False, negative NaN → True (the raw sign bit, not "is it NaN").
            var arr = np.array(new[] { PosQNaN, NegQNaN });
            Assert.IsTrue(Enumerable.SequenceEqual(np.signbit(arr).Data<bool>(), new[] { false, true }));
        }

        [TestMethod]
        public void signbit_ReturnsBoolDtype()
        {
            Assert.AreEqual(typeof(bool), np.signbit(np.array(new[] { -1.0, 1.0 })).dtype);
            Assert.AreEqual(typeof(bool), np.signbit(np.array(new[] { -1, 1 })).dtype);
        }

        [TestMethod]
        public void signbit_Float32()
        {
            var arr = np.array(new[] { -0.0f, 0.0f, -2.5f, 2.5f, float.NegativeInfinity, float.PositiveInfinity });
            Assert.IsTrue(Enumerable.SequenceEqual(np.signbit(arr).Data<bool>(),
                new[] { true, false, true, false, true, false }));
        }

        [TestMethod]
        public void signbit_Half()
        {
            // -0.0 → True via the raw f16 sign bit (Half.IsNegative), not a value comparison.
            var arr = np.array(new[] { (Half)(-2.5), (Half)2.5, (Half)(-0.0), Half.NegativeInfinity, Half.PositiveInfinity });
            Assert.IsTrue(Enumerable.SequenceEqual(np.signbit(arr).Data<bool>(),
                new[] { true, false, true, true, false }));
        }

        [TestMethod]
        public void signbit_SignedIntegers()
        {
            // Signed: x < 0 (the two's-complement MSB), including the type minimum.
            Assert.IsTrue(Enumerable.SequenceEqual(np.signbit(np.array(new sbyte[] { -1, 0, 1, sbyte.MinValue, sbyte.MaxValue })).Data<bool>(),
                new[] { true, false, false, true, false }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.signbit(np.array(new short[] { -1, 0, 1, short.MinValue })).Data<bool>(),
                new[] { true, false, false, true }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.signbit(np.array(new[] { -1, 0, 1, int.MinValue })).Data<bool>(),
                new[] { true, false, false, true }));
            Assert.IsTrue(Enumerable.SequenceEqual(np.signbit(np.array(new[] { -1L, 0L, 5L, long.MinValue })).Data<bool>(),
                new[] { true, false, false, true }));
        }

        [TestMethod]
        public void signbit_UnsignedAndBool_AlwaysFalse()
        {
            // No sign bit is ever set for unsigned / bool / char.
            Assert.IsFalse(np.any(np.signbit(np.array(new byte[] { 0, 1, 255 }))));
            Assert.IsFalse(np.any(np.signbit(np.array(new ushort[] { 0, 1, 65535 }))));
            Assert.IsFalse(np.any(np.signbit(np.array(new uint[] { 0, 1, uint.MaxValue }))));
            Assert.IsFalse(np.any(np.signbit(np.array(new ulong[] { 0, 1, ulong.MaxValue }))));
            Assert.IsFalse(np.any(np.signbit(np.array(new[] { true, false }))));
            Assert.IsFalse(np.any(np.signbit(np.array(new[] { 'a', 'z', '\0' }))));
        }

        [TestMethod]
        public void signbit_Decimal()
        {
            // No NumPy analog. Strictly-negative test (Math.Sign < 0); -0.0m → False (documented).
            Assert.IsTrue(Enumerable.SequenceEqual(np.signbit(np.array(new[] { -5m, 0m, 5m, decimal.MinValue })).Data<bool>(),
                new[] { true, false, false, true }));
        }

        [TestMethod]
        public void signbit_2D()
        {
            var arr = np.arange(-4, 4).reshape(2, 4); // [[-4,-3,-2,-1],[0,1,2,3]]
            var r = np.signbit(arr);
            Assert.AreEqual(2, r.ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<bool>(),
                new[] { true, true, true, true, false, false, false, false }));
        }

        [TestMethod]
        public void signbit_Scalar()
        {
            Assert.IsTrue(np.signbit(np.array(-0.0)).GetBoolean());
            Assert.IsFalse(np.signbit(np.array(0.0)).GetBoolean());
            Assert.IsTrue(np.signbit(np.array(-3.5)).GetBoolean());
            Assert.IsFalse(np.signbit(np.array(3.5)).GetBoolean());
        }

        [TestMethod]
        public void signbit_Empty()
        {
            var r = np.signbit(np.array(new double[0]));
            Assert.AreEqual(0, r.size);
            Assert.AreEqual(typeof(bool), r.dtype);
        }

        [TestMethod]
        public void signbit_ReversedView()
        {
            var arr = np.array(new[] { -3.0, -2.0, -1.0, 0.0, 1.0, 2.0 });
            var rev = arr["::-1"]; // [2,1,0,-1,-2,-3]
            Assert.IsTrue(Enumerable.SequenceEqual(np.signbit(rev).Data<bool>(),
                new[] { false, false, false, true, true, true }));
        }

        [TestMethod]
        public void signbit_SteppedView()
        {
            var arr = np.array(new[] { -3.0, -2.0, -1.0, 0.0, 1.0, 2.0 });
            var step = arr["::2"]; // [-3,-1,1]
            Assert.IsTrue(Enumerable.SequenceEqual(np.signbit(step).Data<bool>(), new[] { true, true, false }));
        }

        [TestMethod]
        public void signbit_LargeArray_SimdPathConsistent()
        {
            // Exercise the SIMD unroll + remainder + tail across the four vectorized dtypes; each
            // signbit(v) must equal the raw sign test element-for-element.
            var rng = new System.Random(7);
            var d = Enumerable.Range(0, 5001).Select(_ => rng.NextDouble() * 2000 - 1000).ToArray();
            var got = np.signbit(np.array(d)).Data<bool>().ToArray();
            for (int i = 0; i < d.Length; i++)
                Assert.AreEqual(double.IsNegative(d[i]), got[i], $"f64 mismatch at {i}");

            var ints = Enumerable.Range(0, 5001).Select(_ => rng.Next(-1000, 1000)).ToArray();
            var gotI = np.signbit(np.array(ints)).Data<bool>().ToArray();
            for (int i = 0; i < ints.Length; i++)
                Assert.AreEqual(ints[i] < 0, gotI[i], $"i32 mismatch at {i}");
        }

        [TestMethod]
        public void signbit_Complex_Throws()
        {
            // signbit has no complex loop — NumPy's verbatim ufunc-no-loop TypeError.
            var c = np.array(new[] { new Complex(1, 2), new Complex(-3, 4) });
            var ex = Assert.ThrowsException<TypeError>(() => np.signbit(c));
            Assert.AreEqual(
                "ufunc 'signbit' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''",
                ex.Message);
        }

        [TestMethod]
        public void signbit_Out_NumericAndBool()
        {
            var x = np.array(new[] { -2.5, 2.5, -1.0 });

            // numeric out: stores 0/1, returns the same instance (NumPy positional out).
            var yInt = np.array(new[] { 9, 9, 9 });
            var rInt = np.signbit(x, yInt);
            Assert.IsTrue(ReferenceEquals(rInt, yInt));
            Assert.IsTrue(Enumerable.SequenceEqual(yInt.Data<int>(), new[] { 1, 0, 1 }));

            // bool out
            var yBool = np.array(new[] { false, false, false });
            np.signbit(x, yBool);
            Assert.IsTrue(Enumerable.SequenceEqual(yBool.Data<bool>(), new[] { true, false, true }));
        }

        [TestMethod]
        public void signbit_DtypeParam()
        {
            // dtype=bool is a legal no-op; any non-bool dtype has no loop (NumPy TypeError).
            var x = np.array(new[] { -2.5, 2.5 });
            Assert.IsTrue(Enumerable.SequenceEqual(np.signbit(x, dtype: np.@bool).Data<bool>(), new[] { true, false }));
            Assert.ThrowsException<IncorrectTypeException>(() => np.signbit(x, dtype: np.int32));
        }
    }
}
