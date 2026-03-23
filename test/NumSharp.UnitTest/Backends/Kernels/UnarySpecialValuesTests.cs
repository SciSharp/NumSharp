using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels
{
    /// <summary>
    /// Tests for unary operations with special IEEE 754 float values.
    /// Verifies correct handling of: 0.0, -0.0, +inf, -inf, NaN
    /// </summary>
    public class UnarySpecialValuesTests
    {
        // Helper to check if a value is negative zero
        private static bool IsNegativeZero(double value) => value == 0.0 && double.IsNegative(value);

        #region np.sqrt

        [Test]
        public void Sqrt_SpecialValues()
        {
            // NumPy: np.sqrt([0, -0, inf, -inf, nan]) = [0, -0, inf, nan, nan]
            var input = np.array(new double[] { 0.0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
            var result = np.sqrt(input);

            Assert.AreEqual(0.0, result.GetDouble(0));           // sqrt(0) = 0
            Assert.IsTrue(IsNegativeZero(result.GetDouble(1)));  // sqrt(-0) = -0 (IEEE 754)
            Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(2)));  // sqrt(inf) = inf
            Assert.IsTrue(double.IsNaN(result.GetDouble(3)));    // sqrt(-inf) = nan
            Assert.IsTrue(double.IsNaN(result.GetDouble(4)));    // sqrt(nan) = nan
        }

        #endregion

        #region np.log

        [Test]
        public void Log_SpecialValues()
        {
            // NumPy: np.log([0, -0, inf, -inf, nan]) = [-inf, -inf, inf, nan, nan]
            var input = np.array(new double[] { 0.0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
            var result = np.log(input);

            Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(0)));  // log(0) = -inf
            Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(1)));  // log(-0) = -inf
            Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(2)));  // log(inf) = inf
            Assert.IsTrue(double.IsNaN(result.GetDouble(3)));    // log(-inf) = nan
            Assert.IsTrue(double.IsNaN(result.GetDouble(4)));    // log(nan) = nan
        }

        #endregion

        #region np.exp

        [Test]
        public void Exp_SpecialValues()
        {
            // NumPy: np.exp([0, -0, inf, -inf, nan]) = [1, 1, inf, 0, nan]
            var input = np.array(new double[] { 0.0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
            var result = np.exp(input);

            Assert.AreEqual(1.0, result.GetDouble(0));           // exp(0) = 1
            Assert.AreEqual(1.0, result.GetDouble(1));           // exp(-0) = 1
            Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(2)));  // exp(inf) = inf
            Assert.AreEqual(0.0, result.GetDouble(3));           // exp(-inf) = 0
            Assert.IsTrue(double.IsNaN(result.GetDouble(4)));    // exp(nan) = nan
        }

        #endregion

        #region np.sin

        [Test]
        public void Sin_SpecialValues()
        {
            // NumPy: np.sin([0, -0, inf, -inf, nan]) = [0, -0, nan, nan, nan]
            var input = np.array(new double[] { 0.0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
            var result = np.sin(input);

            Assert.AreEqual(0.0, result.GetDouble(0));           // sin(0) = 0
            Assert.IsTrue(IsNegativeZero(result.GetDouble(1)));  // sin(-0) = -0
            Assert.IsTrue(double.IsNaN(result.GetDouble(2)));    // sin(inf) = nan
            Assert.IsTrue(double.IsNaN(result.GetDouble(3)));    // sin(-inf) = nan
            Assert.IsTrue(double.IsNaN(result.GetDouble(4)));    // sin(nan) = nan
        }

        #endregion

        #region np.cos

        [Test]
        public void Cos_SpecialValues()
        {
            // NumPy: np.cos([0, -0, inf, -inf, nan]) = [1, 1, nan, nan, nan]
            var input = np.array(new double[] { 0.0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
            var result = np.cos(input);

            Assert.AreEqual(1.0, result.GetDouble(0));           // cos(0) = 1
            Assert.AreEqual(1.0, result.GetDouble(1));           // cos(-0) = 1
            Assert.IsTrue(double.IsNaN(result.GetDouble(2)));    // cos(inf) = nan
            Assert.IsTrue(double.IsNaN(result.GetDouble(3)));    // cos(-inf) = nan
            Assert.IsTrue(double.IsNaN(result.GetDouble(4)));    // cos(nan) = nan
        }

        #endregion

        #region np.tan

        [Test]
        public void Tan_SpecialValues()
        {
            // NumPy: np.tan([0, -0, inf, -inf, nan]) = [0, -0, nan, nan, nan]
            var input = np.array(new double[] { 0.0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
            var result = np.tan(input);

            Assert.AreEqual(0.0, result.GetDouble(0));           // tan(0) = 0
            Assert.IsTrue(IsNegativeZero(result.GetDouble(1)));  // tan(-0) = -0
            Assert.IsTrue(double.IsNaN(result.GetDouble(2)));    // tan(inf) = nan
            Assert.IsTrue(double.IsNaN(result.GetDouble(3)));    // tan(-inf) = nan
            Assert.IsTrue(double.IsNaN(result.GetDouble(4)));    // tan(nan) = nan
        }

        #endregion

        #region np.sign

        [Test]
        public void Sign_SpecialValues()
        {
            // NumPy: np.sign([0, -0, inf, -inf, nan]) = [0, 0, 1, -1, nan]
            var input = np.array(new double[] { 0.0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
            var result = np.sign(input);

            Assert.AreEqual(0.0, result.GetDouble(0));           // sign(0) = 0
            Assert.AreEqual(0.0, result.GetDouble(1));           // sign(-0) = 0 (not -0!)
            Assert.AreEqual(1.0, result.GetDouble(2));           // sign(inf) = 1
            Assert.AreEqual(-1.0, result.GetDouble(3));          // sign(-inf) = -1
            Assert.IsTrue(double.IsNaN(result.GetDouble(4)));    // sign(nan) = nan
        }

        #endregion

        #region np.negative

        [Test]
        public void Negative_SpecialValues()
        {
            // NumPy: np.negative([0, -0, inf, -inf, nan]) = [-0, 0, -inf, inf, nan]
            var input = np.array(new double[] { 0.0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
            var result = np.negative(input);

            Assert.IsTrue(IsNegativeZero(result.GetDouble(0)));  // -(0) = -0
            Assert.AreEqual(0.0, result.GetDouble(1));           // -(-0) = 0
            Assert.IsFalse(IsNegativeZero(result.GetDouble(1))); // Ensure it's not -0
            Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(2)));  // -(inf) = -inf
            Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(3)));  // -(-inf) = inf
            Assert.IsTrue(double.IsNaN(result.GetDouble(4)));    // -(nan) = nan
        }

        #endregion

        #region np.reciprocal

        [Test]
        public void Reciprocal_SpecialValues()
        {
            // NumPy: np.reciprocal([0, -0, inf, -inf, nan]) = [inf, -inf, 0, -0, nan]
            var input = np.array(new double[] { 0.0, -0.0, double.PositiveInfinity, double.NegativeInfinity, double.NaN });
            var result = np.reciprocal(input);

            Assert.IsTrue(double.IsPositiveInfinity(result.GetDouble(0)));  // 1/0 = inf
            Assert.IsTrue(double.IsNegativeInfinity(result.GetDouble(1)));  // 1/(-0) = -inf
            Assert.AreEqual(0.0, result.GetDouble(2));           // 1/inf = 0
            Assert.IsTrue(IsNegativeZero(result.GetDouble(3)));  // 1/(-inf) = -0
            Assert.IsTrue(double.IsNaN(result.GetDouble(4)));    // 1/nan = nan
        }

        #endregion

        #region Float32 tests

        [Test]
        public void Sqrt_Float32_SpecialValues()
        {
            // Same behavior for float32
            var input = np.array(new float[] { 0.0f, -0.0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN });
            var result = np.sqrt(input);

            Assert.AreEqual(0.0f, result.GetSingle(0));
            Assert.IsTrue(float.IsNegative(result.GetSingle(1)) && result.GetSingle(1) == 0.0f);
            Assert.IsTrue(float.IsPositiveInfinity(result.GetSingle(2)));
            Assert.IsTrue(float.IsNaN(result.GetSingle(3)));
            Assert.IsTrue(float.IsNaN(result.GetSingle(4)));
        }

        [Test]
        public void Exp_Float32_SpecialValues()
        {
            var input = np.array(new float[] { 0.0f, -0.0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN });
            var result = np.exp(input);

            Assert.AreEqual(1.0f, result.GetSingle(0));
            Assert.AreEqual(1.0f, result.GetSingle(1));
            Assert.IsTrue(float.IsPositiveInfinity(result.GetSingle(2)));
            Assert.AreEqual(0.0f, result.GetSingle(3));
            Assert.IsTrue(float.IsNaN(result.GetSingle(4)));
        }

        #endregion

        #region NaN propagation

        [Test]
        public void NaN_Propagates_Through_All_Operations()
        {
            // NaN should propagate through all unary operations
            var nan = np.array(new double[] { double.NaN });

            Assert.IsTrue(double.IsNaN(np.sqrt(nan).GetDouble()));
            Assert.IsTrue(double.IsNaN(np.log(nan).GetDouble()));
            Assert.IsTrue(double.IsNaN(np.exp(nan).GetDouble()));
            Assert.IsTrue(double.IsNaN(np.sin(nan).GetDouble()));
            Assert.IsTrue(double.IsNaN(np.cos(nan).GetDouble()));
            Assert.IsTrue(double.IsNaN(np.tan(nan).GetDouble()));
            Assert.IsTrue(double.IsNaN(np.sign(nan).GetDouble()));
            Assert.IsTrue(double.IsNaN(np.negative(nan).GetDouble()));
            Assert.IsTrue(double.IsNaN(np.reciprocal(nan).GetDouble()));
            Assert.IsTrue(double.IsNaN(np.abs(nan).GetDouble()));
        }

        #endregion
    }
}
