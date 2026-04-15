using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Logic
{
    /// <summary>
    /// Tests for NumPy 2.x type promotion rules (NEP50).
    /// Verifies that mixed-type operations produce correct result dtypes.
    /// </summary>
    [TestClass]
    public class TypePromotionTests
    {
        #region Integer Promotions

        [TestMethod]
        public void IntegerPromotion_Uint8_Int16()
        {
            // uint8 + int16 → int16 (int16 can hold both ranges)
            var a = np.array(new byte[] { 1, 2, 3 });
            var b = np.array(new short[] { 1, 2, 3 });
            var result = a + b;
            Assert.AreEqual(NPTypeCode.Int16, result.typecode);
        }

        [TestMethod]
        public void IntegerPromotion_Uint8_Int32()
        {
            var a = np.array(new byte[] { 1, 2, 3 });
            var b = np.array(new int[] { 1, 2, 3 });
            var result = a + b;
            Assert.AreEqual(NPTypeCode.Int32, result.typecode);
        }

        [TestMethod]
        public void IntegerPromotion_Uint16_Int16()
        {
            // uint16 + int16 → int32 (need larger to hold both ranges)
            var a = np.array(new ushort[] { 1, 2, 3 });
            var b = np.array(new short[] { 1, 2, 3 });
            var result = a + b;
            Assert.AreEqual(NPTypeCode.Int32, result.typecode);
        }

        [TestMethod]
        public void IntegerPromotion_Uint32_Int32()
        {
            // uint32 + int32 → int64 (need larger to hold both ranges)
            var a = np.array(new uint[] { 1, 2, 3 });
            var b = np.array(new int[] { 1, 2, 3 });
            var result = a + b;
            Assert.AreEqual(NPTypeCode.Int64, result.typecode);
        }

        [TestMethod]
        public void IntegerPromotion_Uint32_Int64()
        {
            var a = np.array(new uint[] { 1, 2, 3 });
            var b = np.array(new long[] { 1, 2, 3 });
            var result = a + b;
            Assert.AreEqual(NPTypeCode.Int64, result.typecode);
        }

        [TestMethod]
        public void IntegerPromotion_Uint64_Int64()
        {
            // uint64 + int64 → float64 (no larger integer type available)
            var a = np.array(new ulong[] { 1, 2, 3 });
            var b = np.array(new long[] { 1, 2, 3 });
            var result = a + b;
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        #endregion

        #region Float Promotions (NEP50)

        [TestMethod]
        public void FloatPromotion_Int32_Float32_ReturnsFloat64()
        {
            // NEP50: int32 + float32 → float64 (NOT float32!)
            // This is a key NumPy 2.x change
            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.array(new float[] { 1.0f, 2.0f, 3.0f });
            var result = a + b;
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        [TestMethod]
        public void FloatPromotion_Int32_Float64()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.array(new double[] { 1.0, 2.0, 3.0 });
            var result = a + b;
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        [TestMethod]
        public void FloatPromotion_Float32_Float64()
        {
            var a = np.array(new float[] { 1.0f, 2.0f, 3.0f });
            var b = np.array(new double[] { 1.0, 2.0, 3.0 });
            var result = a + b;
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        #endregion

        #region Boolean Promotions

        [TestMethod]
        public void BoolPromotion_Bool_Int32()
        {
            var a = np.array(new bool[] { true, false, true });
            var b = np.array(new int[] { 1, 2, 3 });
            var result = a + b;
            Assert.AreEqual(NPTypeCode.Int32, result.typecode);
        }

        [TestMethod]
        public void BoolPromotion_Bool_Float32()
        {
            var a = np.array(new bool[] { true, false, true });
            var b = np.array(new float[] { 1.0f, 2.0f, 3.0f });
            var result = a + b;
            Assert.AreEqual(NPTypeCode.Single, result.typecode);
        }

        [TestMethod]
        public void BoolPromotion_Bool_Float64()
        {
            var a = np.array(new bool[] { true, false, true });
            var b = np.array(new double[] { 1.0, 2.0, 3.0 });
            var result = a + b;
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        #endregion

        #region Scalar + Array (NEP50 - scalar doesn't promote)

        [TestMethod]
        public void ScalarPromotion_Int32Array_IntScalar()
        {
            // NEP50: Scalar doesn't promote array dtype
            var a = np.array(new int[] { 1, 2, 3 });
            var result = a + 1;
            Assert.AreEqual(NPTypeCode.Int32, result.typecode);
        }

        [TestMethod]
        public void ScalarPromotion_Float32Array_IntScalar()
        {
            // NEP50: Int scalar doesn't promote float32 to float64
            var a = np.array(new float[] { 1.0f, 2.0f, 3.0f });
            var result = a + 1;
            Assert.AreEqual(NPTypeCode.Single, result.typecode);
        }

        [TestMethod]
        public void ScalarPromotion_Float32Array_DoubleScalar()
        {
            // NEP50: Double scalar doesn't promote float32 array
            var a = np.array(new float[] { 1.0f, 2.0f, 3.0f });
            var result = a + 1.0;
            Assert.AreEqual(NPTypeCode.Single, result.typecode);
        }

        [TestMethod]
        public void ScalarPromotion_Int32Array_DoubleScalar()
        {
            // Float scalar DOES promote int array to float
            var a = np.array(new int[] { 1, 2, 3 });
            var result = a + 1.0;
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        #endregion

        #region All Operations Preserve Type Rules

        [TestMethod]
        public void Subtraction_SameRulesAsAddition()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.array(new float[] { 1.0f, 2.0f, 3.0f });
            var result = a - b;
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        [TestMethod]
        public void Multiplication_SameRulesAsAddition()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.array(new float[] { 1.0f, 2.0f, 3.0f });
            var result = a * b;
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        [TestMethod]
        public void Division_SameRulesAsAddition()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.array(new float[] { 1.0f, 2.0f, 3.0f });
            var result = a / b;
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        #endregion
    }
}
