using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels
{
    /// <summary>
    /// Dtype coverage tests verifying all 12 NumSharp types work with major operations.
    ///
    /// NumSharp's 12 dtypes:
    /// - Boolean, Byte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char, Single, Double, Decimal
    ///
    /// Operations tested:
    /// - Unary: sqrt, abs, negate
    /// - Binary: add, multiply
    /// - Reduction: sum, max
    /// - Comparison: equal, less
    ///
    /// Known limitations documented inline.
    /// </summary>
    public class DtypeCoverageTests
    {
        #region Dtype Lists

        // All 12 NumSharp dtypes
        private static readonly NPTypeCode[] AllDtypes = new[]
        {
            NPTypeCode.Boolean,
            NPTypeCode.Byte,
            NPTypeCode.Int16,
            NPTypeCode.UInt16,
            NPTypeCode.Int32,
            NPTypeCode.UInt32,
            NPTypeCode.Int64,
            NPTypeCode.UInt64,
            NPTypeCode.Char,
            NPTypeCode.Single,
            NPTypeCode.Double,
            NPTypeCode.Decimal
        };

        // Numeric dtypes (excludes Boolean, Char)
        private static readonly NPTypeCode[] NumericDtypes = new[]
        {
            NPTypeCode.Byte,
            NPTypeCode.Int16,
            NPTypeCode.UInt16,
            NPTypeCode.Int32,
            NPTypeCode.UInt32,
            NPTypeCode.Int64,
            NPTypeCode.UInt64,
            NPTypeCode.Single,
            NPTypeCode.Double,
            NPTypeCode.Decimal
        };

        // Float dtypes (sqrt returns meaningful results)
        private static readonly NPTypeCode[] FloatDtypes = new[]
        {
            NPTypeCode.Single,
            NPTypeCode.Double,
            NPTypeCode.Decimal
        };

        // Signed numeric dtypes (negate makes sense)
        private static readonly NPTypeCode[] SignedDtypes = new[]
        {
            NPTypeCode.Int16,
            NPTypeCode.Int32,
            NPTypeCode.Int64,
            NPTypeCode.Single,
            NPTypeCode.Double,
            NPTypeCode.Decimal
        };

        #endregion

        #region Unary Operations

        [Test]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Sqrt_FloatDtypes(NPTypeCode dtype)
        {
            // sqrt is meaningful for float types
            // NumPy: sqrt returns float64 for most inputs
            var arr = np.array(new[] { 1.0, 4.0, 9.0 }).astype(dtype);
            var result = np.sqrt(arr);

            // Should complete without exception
            Assert.AreEqual(3, result.size);
            // Result should be approximately correct, use Convert.ToDouble for type-agnostic access
            var v0 = Convert.ToDouble(result.GetAtIndex(0));
            var v1 = Convert.ToDouble(result.GetAtIndex(1));
            var v2 = Convert.ToDouble(result.GetAtIndex(2));
            Assert.IsTrue(v0 >= 0.9 && v0 <= 1.1, $"sqrt(1) should be 1, got {v0}");
            Assert.IsTrue(v1 >= 1.9 && v1 <= 2.1, $"sqrt(4) should be 2, got {v1}");
            Assert.IsTrue(v2 >= 2.9 && v2 <= 3.1, $"sqrt(9) should be 3, got {v2}");
        }

        [Test]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        public void Sqrt_IntegerDtypes(NPTypeCode dtype)
        {
            // sqrt on integers works (result is float64)
            var arr = np.array(new[] { 1, 4, 9 }).astype(dtype);
            var result = np.sqrt(arr);

            Assert.AreEqual(3, result.size);
            // Result dtype should be Double (float64)
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        [Test]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Abs_NumericDtypes(NPTypeCode dtype)
        {
            // abs works for all numeric types
            var arr = np.array(new[] { 1, 2, 3 }).astype(dtype);
            var result = np.abs(arr);

            Assert.AreEqual(3, result.size);
            // Should preserve non-negative values, use Convert.ToDouble for type-agnostic access
            var value = Convert.ToDouble(result.GetAtIndex(0));
            Assert.IsTrue(value >= 0, $"abs result should be non-negative, got {value}");
        }

        [Test]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Negate_SignedDtypes(NPTypeCode dtype)
        {
            // negate works for signed types
            var arr = np.array(new[] { 1, 2, 3 }).astype(dtype);
            var result = np.negative(arr);

            Assert.AreEqual(3, result.size);
            // Result should be negative, use Convert.ToDouble for type-agnostic access
            var value = Convert.ToDouble(result.GetAtIndex(0));
            Assert.IsTrue(value < 0);
        }

        #endregion

        #region Binary Operations

        [Test]
        [Arguments(NPTypeCode.Boolean)]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Char)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Add_AllDtypes(NPTypeCode dtype)
        {
            // Addition works for all 12 dtypes
            var a = np.array(new[] { 1, 2, 3 }).astype(dtype);
            var b = np.array(new[] { 1, 1, 1 }).astype(dtype);
            var result = np.add(a, b);

            Assert.AreEqual(3, result.size);
            // Get value using type-agnostic access
            var rawValue = result.GetAtIndex(0);
            // Char doesn't support Convert.ToDouble, need special handling
            double value = dtype == NPTypeCode.Char
                ? (double)(char)rawValue
                : Convert.ToDouble(rawValue);

            // Boolean: true + true = true (boolean OR) or 2 depending on implementation
            // Char: '\x01' + '\x01' = '\x02' = 2
            // All others: 1+1=2
            if (dtype == NPTypeCode.Boolean)
            {
                // Boolean addition can be 1 (logical OR) or 2 (arithmetic)
                Assert.IsTrue(value >= 0.9 && value <= 2.1, $"Boolean add result should be 1 or 2, got {value}");
            }
            else
            {
                Assert.IsTrue(value >= 1.9 && value <= 2.1, $"Add result should be 2, got {value}");
            }
        }

        [Test]
        [Arguments(NPTypeCode.Boolean)]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Char)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Multiply_AllDtypes(NPTypeCode dtype)
        {
            // Multiplication works for all 12 dtypes
            var a = np.array(new[] { 1, 2, 3 }).astype(dtype);
            var b = np.array(new[] { 2, 2, 2 }).astype(dtype);
            var result = np.multiply(a, b);

            Assert.AreEqual(3, result.size);
            // Get values using type-agnostic access
            // Char doesn't support Convert.ToDouble, need special handling
            var raw0 = result.GetAtIndex(0);
            var raw1 = result.GetAtIndex(1);
            double value0 = dtype == NPTypeCode.Char ? (double)(char)raw0 : Convert.ToDouble(raw0);
            double value1 = dtype == NPTypeCode.Char ? (double)(char)raw1 : Convert.ToDouble(raw1);

            // Boolean: true * true = true (boolean AND) = 1
            // All others: 1*2=2, 2*2=4
            if (dtype == NPTypeCode.Boolean)
            {
                Assert.IsTrue(value0 >= 0.9 && value0 <= 1.1, $"Boolean multiply should be 1, got {value0}");
            }
            else
            {
                Assert.IsTrue(value0 >= 1.9 && value0 <= 2.1, $"Multiply result[0] should be 2, got {value0}");
                Assert.IsTrue(value1 >= 3.9 && value1 <= 4.1, $"Multiply result[1] should be 4, got {value1}");
            }
        }

        [Test]
        [Arguments(NPTypeCode.Int32, NPTypeCode.Double)]
        [Arguments(NPTypeCode.Int64, NPTypeCode.Single)]
        [Arguments(NPTypeCode.Byte, NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt16, NPTypeCode.Int64)]
        public void Add_MixedDtypes(NPTypeCode dtype1, NPTypeCode dtype2)
        {
            // Mixed type addition with type promotion
            var a = np.array(new[] { 1, 2, 3 }).astype(dtype1);
            var b = np.array(new[] { 1, 1, 1 }).astype(dtype2);
            var result = np.add(a, b);

            Assert.AreEqual(3, result.size);
            // Should succeed with type promotion
        }

        #endregion

        #region Reduction Operations

        [Test]
        [Arguments(NPTypeCode.Boolean)]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Char)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Sum_AllDtypes(NPTypeCode dtype)
        {
            // Sum works for all 12 dtypes
            var arr = np.array(new[] { 1, 2, 3 }).astype(dtype);
            var result = np.sum(arr);

            // Should produce scalar result
            Assert.IsTrue(result.Shape.IsScalar || result.size == 1);

            // Use Convert.ToDouble for type-agnostic access
            // (np.sum promotes int types to int64, so GetDouble() won't work)
            var value = Convert.ToDouble(result.GetAtIndex(0));

            // Boolean: [1,2,3] -> [true,true,true] -> sum = 3
            // All others: 1+2+3 = 6
            if (dtype == NPTypeCode.Boolean)
                Assert.IsTrue(value >= 2.9 && value <= 3.1, $"Boolean sum should be 3, got {value}");
            else
                Assert.IsTrue(value >= 5.9 && value <= 6.1, $"Sum should be 6, got {value}");
        }

        [Test]
        // Note: Boolean and Char are excluded - np.amax doesn't support them
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Max_AllDtypes(NPTypeCode dtype)
        {
            // Max works for numeric dtypes (excluding Boolean and Char)
            var arr = np.array(new[] { 1, 3, 2 }).astype(dtype);
            var result = np.amax(arr);

            // Should produce scalar result
            Assert.IsTrue(result.Shape.IsScalar || result.size == 1);
            // Use Convert.ToDouble for type-agnostic access
            var value = Convert.ToDouble(result.GetAtIndex(0));

            // max([1,3,2]) = 3
            Assert.IsTrue(value >= 2.9 && value <= 3.1, $"Max should be 3, got {value}");
        }

        [Test]
        // Note: Boolean and Char are excluded - np.amin doesn't support them
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Min_AllDtypes(NPTypeCode dtype)
        {
            // Min works for numeric dtypes (excluding Boolean and Char)
            var arr = np.array(new[] { 2, 1, 3 }).astype(dtype);
            var result = np.amin(arr);

            // Should produce scalar result
            Assert.IsTrue(result.Shape.IsScalar || result.size == 1);
            // Use Convert.ToDouble for type-agnostic access
            var value = Convert.ToDouble(result.GetAtIndex(0));

            // Boolean: [2,1,3] -> [true,true,true] -> min = 1 (true)
            // All others: min([2,1,3]) = 1
            // Both cases should be 1
            Assert.IsTrue(value >= 0.9 && value <= 1.1, $"Min should be 1, got {value}");
        }

        [Test]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Mean_NumericDtypes(NPTypeCode dtype)
        {
            // Mean works for numeric dtypes
            var arr = np.array(new[] { 1, 2, 3 }).astype(dtype);
            var result = np.mean(arr);

            // Should produce scalar result
            Assert.IsTrue(result.Shape.IsScalar || result.size == 1);
            // Mean of [1,2,3] = 2, use Convert.ToDouble for type-agnostic access
            // Mean always returns Double for all input types
            var value = Convert.ToDouble(result.GetAtIndex(0));
            Assert.IsTrue(value >= 1.9 && value <= 2.1);
        }

        #endregion

        #region Comparison Operations

        [Test]
        [Arguments(NPTypeCode.Boolean)]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Char)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Equal_AllDtypes(NPTypeCode dtype)
        {
            // Equality comparison works for all 12 dtypes
            var a = np.array(new[] { 1, 2, 3 }).astype(dtype);
            var b = np.array(new[] { 1, 0, 3 }).astype(dtype);
            var result = np.equal(a, b);

            Assert.AreEqual(3, result.size);
            Assert.AreEqual(NPTypeCode.Boolean, result.typecode);
            Assert.IsTrue(result.GetBoolean(0));   // 1 == 1
            Assert.IsFalse(result.GetBoolean(1));  // 2 != 0
            Assert.IsTrue(result.GetBoolean(2));   // 3 == 3
        }

        [Test]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Char)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Less_NumericDtypes(NPTypeCode dtype)
        {
            // Less-than comparison for numeric dtypes
            var a = np.array(new[] { 1, 2, 3 }).astype(dtype);
            var b = np.array(new[] { 2, 2, 2 }).astype(dtype);
            var result = np.less(a, b);

            Assert.AreEqual(3, result.size);
            Assert.AreEqual(NPTypeCode.Boolean, result.typecode);
            Assert.IsTrue(result.GetBoolean(0));   // 1 < 2
            Assert.IsFalse(result.GetBoolean(1));  // 2 < 2 is false
            Assert.IsFalse(result.GetBoolean(2));  // 3 < 2 is false
        }

        [Test]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Char)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Greater_NumericDtypes(NPTypeCode dtype)
        {
            // Greater-than comparison for numeric dtypes
            var a = np.array(new[] { 1, 2, 3 }).astype(dtype);
            var b = np.array(new[] { 2, 2, 2 }).astype(dtype);
            var result = np.greater(a, b);

            Assert.AreEqual(3, result.size);
            Assert.AreEqual(NPTypeCode.Boolean, result.typecode);
            Assert.IsFalse(result.GetBoolean(0));  // 1 > 2 is false
            Assert.IsFalse(result.GetBoolean(1));  // 2 > 2 is false
            Assert.IsTrue(result.GetBoolean(2));   // 3 > 2
        }

        #endregion

        #region Edge Cases

        [Test]
        [Arguments(NPTypeCode.Boolean)]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Char)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void EmptyArray_AllDtypes(NPTypeCode dtype)
        {
            // Empty arrays should be handled for all dtypes
            var arr = np.array(Array.Empty<int>()).astype(dtype);
            Assert.AreEqual(0, arr.size);
        }

        [Test]
        [Arguments(NPTypeCode.Boolean)]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Char)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void ScalarArray_AllDtypes(NPTypeCode dtype)
        {
            // Scalar arrays should work for all dtypes
            var arr = np.array(42).astype(dtype);
            Assert.IsTrue(arr.Shape.IsScalar || arr.size == 1);
        }

        [Test]
        [Arguments(NPTypeCode.Boolean)]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Char)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Reshape_AllDtypes(NPTypeCode dtype)
        {
            // Reshape should work for all dtypes
            var arr = np.array(new[] { 1, 2, 3, 4, 5, 6 }).astype(dtype);
            var reshaped = arr.reshape(2, 3);

            CollectionAssert.AreEqual(new long[] { 2, 3 }, reshaped.shape);
            Assert.AreEqual(6L, reshaped.size);
        }

        [Test]
        [Arguments(NPTypeCode.Boolean)]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Char)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Slice_AllDtypes(NPTypeCode dtype)
        {
            // Slicing should work for all dtypes
            var arr = np.array(new[] { 1, 2, 3, 4, 5 }).astype(dtype);
            var sliced = arr["1:4"];

            Assert.AreEqual(3, sliced.size);
        }

        [Test]
        [Arguments(NPTypeCode.Boolean)]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Char)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Decimal)]
        public void Copy_AllDtypes(NPTypeCode dtype)
        {
            // Copy should work for all dtypes
            var arr = np.array(new[] { 1, 2, 3 }).astype(dtype);
            var copied = arr.copy();

            Assert.AreEqual(3, copied.size);
            Assert.AreEqual(dtype, copied.typecode);
            // Verify it's a true copy (different address)
            unsafe
            {
                Assert.AreNotEqual((IntPtr)arr.Address, (IntPtr)copied.Address);
            }
        }

        #endregion

        #region Known Limitations Documentation

        /*
         * KNOWN DTYPE LIMITATIONS:
         *
         * 1. Boolean:
         *    - sqrt(bool): Returns Double, treating true=1, false=0
         *    - negate(bool): Uses bitwise NOT, not logical NOT (BUG - Misaligned with NumPy)
         *    - abs(bool): Returns Double (NumSharp behavior)
         *
         * 2. Char:
         *    - Treated as UInt16 internally for arithmetic
         *    - sqrt(char): Returns Double
         *    - Comparisons use char's numeric value
         *
         * 3. Decimal:
         *    - Not a NumPy type (NumSharp extension)
         *    - sqrt(decimal): Uses DecimalEx library
         *    - Some precision differences vs float64
         *
         * 4. Unsigned integers (Byte, UInt16, UInt32, UInt64):
         *    - negate(): Results wrap around (no exception)
         *    - Subtraction can underflow without warning
         *
         * 5. Type Promotion (NumPy 2.x / NEP50):
         *    - int32 + int32 -> int32 (not int64 as in some older NumPy)
         *    - sum(int32) -> int64 (accumulator promotion)
         *    - int / int -> float64 (true division)
         */

        #endregion
    }
}
