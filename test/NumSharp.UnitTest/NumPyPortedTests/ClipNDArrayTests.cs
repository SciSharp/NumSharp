using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.NumPyPortedTests
{
    /// <summary>
    /// Tests for np.clip with array-valued min/max bounds.
    /// Validates the IL kernel migration of Default.ClipNDArray.cs.
    /// </summary>
    [TestClass]
    public class ClipNDArrayTests
    {
        #region Basic Array Bounds Tests

        [TestMethod]
        public void ClipNDArray_BasicArrayBounds_MatchesNumPy()
        {
            // NumPy: np.clip([1,2,3,4,5,6,7,8,9], [2,2,2,3,3,3,4,4,4], [5,5,5,6,6,6,7,7,7])
            // Expected: [2, 2, 3, 4, 5, 6, 7, 7, 7]
            var a = np.array(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var min_arr = np.array(new int[] { 2, 2, 2, 3, 3, 3, 4, 4, 4 });
            var max_arr = np.array(new int[] { 5, 5, 5, 6, 6, 6, 7, 7, 7 });

            var result = np.clip(a, min_arr, max_arr);

            result.Should().BeOfValues(2, 2, 3, 4, 5, 6, 7, 7, 7);
        }

        [TestMethod]
        public void ClipNDArray_MinArrayOnly_MatchesNumPy()
        {
            // NumPy: np.clip([1,2,3,4,5], [2,2,2,2,2], None) = [2,2,3,4,5]
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var min_arr = np.array(new int[] { 2, 2, 2, 2, 2 });

            var result = np.clip(a, min_arr, null);

            result.Should().BeOfValues(2, 2, 3, 4, 5);
        }

        [TestMethod]
        public void ClipNDArray_MaxArrayOnly_MatchesNumPy()
        {
            // NumPy: np.clip([1,2,3,4,5], None, [3,3,3,3,3]) = [1,2,3,3,3]
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var max_arr = np.array(new int[] { 3, 3, 3, 3, 3 });

            var result = np.clip(a, null, max_arr);

            result.Should().BeOfValues(1, 2, 3, 3, 3);
        }

        #endregion

        #region Broadcasting Tests

        [TestMethod]
        public void ClipNDArray_BroadcastMinAlongAxis0_MatchesNumPy()
        {
            // NumPy:
            // a = np.arange(12).reshape(3, 4) = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
            // min_arr = [2, 3, 4, 5] (broadcasts along axis 0)
            // np.clip(a, min_arr, None) = [[2,3,4,5],[4,5,6,7],[8,9,10,11]]
            var a = np.arange(12).reshape(3, 4);
            var min_arr = np.array(new int[] { 2, 3, 4, 5 });

            var result = np.clip(a, min_arr, null);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void ClipNDArray_BroadcastMaxAlongAxis0_MatchesNumPy()
        {
            // NumPy:
            // a = np.arange(12).reshape(3, 4)
            // max_arr = [7, 8, 9, 10]
            // np.clip(a, None, max_arr) = [[0,1,2,3],[4,5,6,7],[7,8,9,10]]
            var a = np.arange(12).reshape(3, 4);
            var max_arr = np.array(new int[] { 7, 8, 9, 10 });

            var result = np.clip(a, null, max_arr);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10);
        }

        [TestMethod]
        public void ClipNDArray_ScalarMinArrayMax_MatchesNumPy()
        {
            // Mixed: scalar min, array max
            var a = np.arange(12).reshape(3, 4);
            var max_arr = np.repeat(8, 12).reshape(3, 4);

            var result = np.clip(a, 3, max_arr);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8);
        }

        [TestMethod]
        public void ClipNDArray_ArrayMinNullMax_MatchesNumPy()
        {
            // From np.clip.Test.cs Case2
            var a = np.arange(12).reshape(3, 4);
            var minmax = np.repeat(8, 12).reshape(3, 4);

            var result = np.clip(a, minmax, null);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 11);
        }

        [TestMethod]
        public void ClipNDArray_NullMinArrayMax_MatchesNumPy()
        {
            // From np.clip.Test.cs Case3
            var a = np.arange(12).reshape(3, 4);
            var max = np.repeat(8, 12).reshape(3, 4);

            var result = np.clip(a, null, max);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8);
        }

        #endregion

        #region Edge Cases

        [TestMethod]
        public void ClipNDArray_MinGreaterThanMax_UsesMaxValue()
        {
            // NumPy behavior: when min[i] > max[i], result is max[i]
            // np.clip([1,2,3,4,5], [6,6,6,6,6], [3,3,3,3,3]) = [3,3,3,3,3]
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var min_arr = np.array(new int[] { 6, 6, 6, 6, 6 });
            var max_arr = np.array(new int[] { 3, 3, 3, 3, 3 });

            var result = np.clip(a, min_arr, max_arr);

            result.Should().BeOfValues(3, 3, 3, 3, 3);
        }

        [TestMethod]
        [Misaligned]
        public void ClipNDArray_NaNInBoundsArray_PropagatesNaN()
        {
            // NumPy: NaN in bounds propagates to result
            // NumSharp: IComparable.CompareTo doesn't handle NaN propagation
            // This is a known behavioral difference - NaN comparison returns false,
            // so the value is not clipped and remains unchanged.
            var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var min_arr = np.array(new double[] { double.NaN, 2.0, 3.0, 3.0, 4.0 });

            var result = np.clip(a, min_arr, null);
            var data = result.GetData<double>();

            // In NumPy: data[0] would be NaN
            // In NumSharp: data[0] stays as 1.0 because NaN comparison returns false
            Assert.IsTrue(double.IsNaN(data[0]), "Expected NaN at index 0");
            Assert.AreEqual(2.0, data[1]);
            Assert.AreEqual(3.0, data[2]);
            Assert.AreEqual(4.0, data[3]);
            Assert.AreEqual(5.0, data[4]);
        }

        [TestMethod]
        public void ClipNDArray_EmptyArray_ReturnsEmpty()
        {
            var a = np.array(new double[0]);
            var min_arr = np.array(new double[0]);
            var max_arr = np.array(new double[0]);

            var result = np.clip(a, min_arr, max_arr);

            Assert.AreEqual(0, result.size);
        }

        [TestMethod]
        public void ClipNDArray_BothNone_ReturnsCopy()
        {
            var a = np.arange(10);
            var result = np.clip(a, null, null);

            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        }

        #endregion

        #region Dtype Tests

        [TestMethod]
        public void ClipNDArray_Float64Array_PreservesDtype()
        {
            var a = np.arange(10.0);
            var min_arr = np.array(new double[] { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 });
            var max_arr = np.array(new double[] { 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 });

            var result = np.clip(a, min_arr, max_arr);

            Assert.AreEqual(np.float64, result.dtype);
            result.Should().BeOfValues(2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0);
        }

        [TestMethod]
        public void ClipNDArray_Int32Array_PreservesDtype()
        {
            // Explicit int32 array for testing dtype preservation
            var a = np.array(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var min_arr = np.array(new int[] { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 });
            var max_arr = np.array(new int[] { 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 });

            var result = np.clip(a, min_arr, max_arr);

            Assert.AreEqual(np.int32, result.dtype);
            result.Should().BeOfValues(2, 2, 2, 3, 4, 5, 6, 7, 7, 7);
        }

        #endregion

        #region Contiguous vs Non-Contiguous Tests

        [TestMethod]
        public void ClipNDArray_TransposedArray_MatchesNumPy()
        {
            // Non-contiguous input (transposed)
            // NumPy: np.clip(arange(12).reshape(3,4).T, 2, 8)
            var a = np.arange(12).reshape(3, 4).T;
            var min_arr = np.array(new int[] { 2, 2, 2 });  // broadcasts to (4,3)
            var max_arr = np.array(new int[] { 8, 8, 8 });

            var result = np.clip(a, min_arr, max_arr);

            result.Should().BeShaped(4, 3);
            // Expected: [[2,4,8],[2,5,8],[2,6,8],[3,7,8]]
            result.Should().BeOfValues(2, 4, 8, 2, 5, 8, 2, 6, 8, 3, 7, 8);
        }

        [TestMethod]
        public void ClipNDArray_SlicedArray_MatchesNumPy()
        {
            // Sliced input (every other element)
            var a = np.arange(20);
            var sliced = a["::" + 2];  // [0,2,4,6,8,10,12,14,16,18]
            var min_arr = np.array(new int[] { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 });
            var max_arr = np.array(new int[] { 15, 15, 15, 15, 15, 15, 15, 15, 15, 15 });

            var result = np.clip(sliced, min_arr, max_arr);

            result.Should().BeOfValues(3, 3, 4, 6, 8, 10, 12, 14, 15, 15);
        }

        #endregion

        #region NumPy 2.x min=/max= Keyword Aliases & Default-None Bounds

        [TestMethod]
        public void Clip_MinKeywordAlias_OnlyLowerBound_MatchesNumPy()
        {
            // NumPy 2.x: np.clip(np.arange(10), min=3) == [3,3,3,3,4,5,6,7,8,9]
            var a = np.arange(10);

            var result = np.clip(a, min: 3);

            result.Should().BeOfValues(3, 3, 3, 3, 4, 5, 6, 7, 8, 9);
        }

        [TestMethod]
        public void Clip_MaxKeywordAlias_OnlyUpperBound_MatchesNumPy()
        {
            // NumPy 2.x: np.clip(np.arange(10), max=5) == [0,1,2,3,4,5,5,5,5,5]
            var a = np.arange(10);

            var result = np.clip(a, max: 5);

            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 5, 5, 5, 5);
        }

        [TestMethod]
        public void Clip_MinAndMaxKeywordAliases_BothBounds_MatchesNumPy()
        {
            // NumPy 2.x: np.clip(np.arange(10), min=3, max=7) == [3,3,3,3,4,5,6,7,7,7]
            var a = np.arange(10);

            var result = np.clip(a, min: 3, max: 7);

            result.Should().BeOfValues(3, 3, 3, 3, 4, 5, 6, 7, 7, 7);
        }

        [TestMethod]
        public void Clip_AMinNullAMaxScalar_MatchesNumPy()
        {
            // NumPy: np.clip(np.arange(10), a_min=None, a_max=5) == [0,1,2,3,4,5,5,5,5,5]
            var a = np.arange(10);

            var result = np.clip(a, a_min: null, a_max: 5);

            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 5, 5, 5, 5);
        }

        [TestMethod]
        public void Clip_AMinScalarAMaxNull_MatchesNumPy()
        {
            // NumPy: np.clip(np.arange(10), a_min=3, a_max=None) == [3,3,3,3,4,5,6,7,8,9]
            var a = np.arange(10);

            var result = np.clip(a, a_min: 3, a_max: null);

            result.Should().BeOfValues(3, 3, 3, 3, 4, 5, 6, 7, 8, 9);
        }

        [TestMethod]
        public void Clip_NoBounds_ReturnsCopy_MatchesNumPy()
        {
            // NumPy: np.clip(np.arange(10)) returns copy unchanged
            var a = np.arange(10);

            var result = np.clip(a);

            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
            // Verify it's a copy (not the same instance / aliased buffer)
            unsafe
            {
                Assert.AreNotEqual((IntPtr)a.Address, (IntPtr)result.Address);
            }
        }

        [TestMethod]
        public void Clip_MinKeywordWithArrayBound_BroadcastsCorrectly()
        {
            // NumPy: np.clip(np.arange(10), min=[5,4,3,2,1,1,2,3,4,5]) == [5,4,3,3,4,5,6,7,8,9]
            var a = np.arange(10);
            var min_arr = np.array(new int[] { 5, 4, 3, 2, 1, 1, 2, 3, 4, 5 });

            var result = np.clip(a, min: min_arr);

            result.Should().BeOfValues(5, 4, 3, 3, 4, 5, 6, 7, 8, 9);
        }

        [TestMethod]
        public void Clip_ConflictingAMinAndMin_Throws()
        {
            var a = np.arange(10);

            Assert.ThrowsExactly<ArgumentException>(() => np.clip(a, a_min: 2, min: 3));
        }

        [TestMethod]
        public void Clip_ConflictingAMaxAndMax_Throws()
        {
            var a = np.arange(10);

            Assert.ThrowsExactly<ArgumentException>(() => np.clip(a, a_max: 2, max: 3));
        }

        [TestMethod]
        public void Clip_MinKeywordWithDtype_PromotesResult()
        {
            // NumPy: np.clip(np.arange(10), min=3.5, dtype=np.float64) == [3.5,3.5,3.5,3.5,4.,5.,6.,7.,8.,9.]
            var a = np.arange(10);

            var result = np.clip(a, min: 3.5, dtype: NPTypeCode.Double);

            Assert.AreEqual(np.float64, result.dtype);
            result.Should().BeOfValues(3.5, 3.5, 3.5, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        }

        #endregion

        #region NumPy 2.x Parity — Dtype Promotion (NEP-50 Weak Scalar Rule)

        [TestMethod]
        public void Clip_Uint8WithIntScalars_PreservesUint8()
        {
            // NumPy NEP 50: np.clip(uint8_arr, 50, 75) preserves uint8
            // (Python int literals are weak — they don't promote).
            // NumSharp mirrors via "0-d same-kind bound preserves outType".
            var arr = np.full(new Shape(5), (byte)100, np.uint8);

            var result = np.clip(arr, 50, 75);

            Assert.AreEqual(np.uint8, result.dtype);
            Assert.AreEqual((byte)75, result.GetByte(0));
        }

        [TestMethod]
        public void Clip_Int32WithFloatScalar_PromotesToFloat64()
        {
            // NumPy: np.clip(int32_arr, min=3.5) → float64 (cross-kind promotion)
            var arr = np.arange(10).astype(NPTypeCode.Int32);

            var result = np.clip(arr, min: 3.5);

            Assert.AreEqual(np.float64, result.dtype);
            Assert.AreEqual(3.5, result.GetDouble(0));
            Assert.AreEqual(3.5, result.GetDouble(3));
            Assert.AreEqual(4.0, result.GetDouble(4));
        }

        [TestMethod]
        public void Clip_Int32WithFloat32Scalar_PromotesToFloat64()
        {
            // NumPy: int32 + float32 promotes to float64 in result_type (NEP 50).
            var arr = np.arange(10).astype(NPTypeCode.Int32);

            var result = np.clip(arr, min: 3.0f);

            Assert.AreEqual(np.float64, result.dtype);
        }

        [TestMethod]
        public void Clip_FloatArrayWithIntScalars_PreservesFloat()
        {
            // NumPy: np.clip(float64_arr, 3, 7) preserves float64.
            var arr = np.arange(10.0);

            var result = np.clip(arr, 3, 7);

            Assert.AreEqual(np.float64, result.dtype);
            Assert.AreEqual(3.0, result.GetDouble(0));
        }

        [TestMethod]
        public void Clip_Int32WithInt64ArrayBound_PromotesToInt64()
        {
            // NumPy: 1-d int64 bound promotes int32 → int64 via result_type.
            var arr = np.arange(10).astype(NPTypeCode.Int32);
            var lo = np.array(new long[] { 3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L, 3L });

            var result = np.clip(arr, min: lo);

            Assert.AreEqual(np.int64, result.dtype);
        }

        [TestMethod]
        public void Clip_DtypeNoBoundsCastsInput()
        {
            // NumPy: np.clip(int32_arr, dtype=np.float32) acts like astype + copy.
            var arr = np.arange(10).astype(NPTypeCode.Int32);

            var result = np.clip(arr, dtype: NPTypeCode.Single);

            Assert.AreEqual(np.float32, result.dtype);
            Assert.AreEqual(0f, result.GetSingle(0));
            Assert.AreEqual(9f, result.GetSingle(9));
        }

        [TestMethod]
        public void Clip_DtypeOverrideForcesNarrowerType()
        {
            // NumPy: np.clip(float64_arr, min=3.0, dtype=np.int32) → int32 (truncates).
            var arr = np.arange(10.0) + 0.5; // 0.5, 1.5, ..., 9.5

            var result = np.clip(arr, min: 3.0, dtype: NPTypeCode.Int32);

            Assert.AreEqual(np.int32, result.dtype);
            // 0.5/1.5/2.5 clipped up to 3 → int 3. 3.5+ → truncated to 3.
            Assert.AreEqual(3, result.GetInt32(0));
            Assert.AreEqual(3, result.GetInt32(3));
            Assert.AreEqual(9, result.GetInt32(9));
        }

        [TestMethod]
        public void Clip_NaNBoundOnIntArray_UpcastsToFloat()
        {
            // NumPy: np.clip(int32_arr, min=NaN) upcasts to float64, result all NaN.
            var arr = np.arange(10).astype(NPTypeCode.Int32);

            var result = np.clip(arr, min: double.NaN);

            Assert.AreEqual(np.float64, result.dtype);
            var data = result.GetData<double>();
            foreach (var v in data)
                Assert.IsTrue(double.IsNaN(v), $"Expected NaN, got {v}");
        }

        #endregion

        #region NumPy 2.x Parity — `out=` Parameter Edge Cases

        [TestMethod]
        public void Clip_OutInPlace_ReturnsAndMutatesInput()
        {
            // NumPy: np.clip(src, 3, 7, out=src) mutates src in place; result is src.
            var src = np.arange(10.0);

            var result = np.clip(src, 3.0, 7.0, @out: src);

            unsafe { Assert.AreEqual((IntPtr)src.Address, (IntPtr)result.Address); }
            Assert.AreEqual(3.0, src.GetDouble(0));
            Assert.AreEqual(3.0, src.GetDouble(3));
            Assert.AreEqual(7.0, src.GetDouble(9));
        }

        [TestMethod]
        public void Clip_OutSeparateBuffer_LeavesInputUnchanged()
        {
            var src = np.arange(10.0);
            var dst = np.empty(new Shape(10), NPTypeCode.Double);

            var result = np.clip(src, 3.0, 7.0, @out: dst);

            unsafe { Assert.AreEqual((IntPtr)dst.Address, (IntPtr)result.Address); }
            // src unchanged
            Assert.AreEqual(0.0, src.GetDouble(0));
            Assert.AreEqual(9.0, src.GetDouble(9));
            // dst populated
            Assert.AreEqual(3.0, dst.GetDouble(0));
            Assert.AreEqual(7.0, dst.GetDouble(9));
        }

        [TestMethod]
        public void Clip_OutShapeMismatch_Throws()
        {
            var src = np.arange(10.0);
            var bad = np.empty(new Shape(5), NPTypeCode.Double);

            Assert.ThrowsExactly<ArgumentException>(() => np.clip(src, 3.0, 7.0, @out: bad));
        }

        [TestMethod]
        public void Clip_OutDtypeMismatch_Throws()
        {
            // NumPy raises _UFuncOutputCastingError when @out dtype is narrower
            // than the result dtype. NumSharp surfaces this as ArgumentException.
            var src = np.arange(10.0) + 0.5;
            var out_int = np.empty(new Shape(10), NPTypeCode.Int32);

            Assert.ThrowsExactly<ArgumentException>(() => np.clip(src, 1.5, 7.5, @out: out_int));
        }

        [TestMethod]
        public void Clip_OutWithNoBounds_CopiesIntoOut()
        {
            // NumPy: np.clip(src, out=dst) copies src into dst when no bounds given.
            var src = np.arange(10.0);
            var dst = np.empty(new Shape(10), NPTypeCode.Double);

            var result = np.clip(src, @out: dst);

            unsafe { Assert.AreEqual((IntPtr)dst.Address, (IntPtr)result.Address); }
            Assert.AreEqual(0.0, dst.GetDouble(0));
            Assert.AreEqual(9.0, dst.GetDouble(9));
        }

        #endregion

        #region NumPy 2.x Parity — Special Float Values via Kwarg Form

        [TestMethod]
        public void Clip_MinNegInfKwarg_NoOp()
        {
            // NumPy: np.clip(arr, min=-inf) is a no-op for finite inputs.
            var arr = np.arange(10.0);

            var result = np.clip(arr, min: double.NegativeInfinity);

            for (int i = 0; i < 10; i++)
                Assert.AreEqual((double)i, result.GetDouble(i));
        }

        [TestMethod]
        public void Clip_MaxPosInfKwarg_NoOp()
        {
            var arr = np.arange(10.0);

            var result = np.clip(arr, max: double.PositiveInfinity);

            for (int i = 0; i < 10; i++)
                Assert.AreEqual((double)i, result.GetDouble(i));
        }

        [TestMethod]
        public void Clip_NaNMinKwarg_PropagatesNaN()
        {
            // NumPy: np.clip(float_arr, min=NaN) → all NaN
            var arr = np.arange(7.0);

            var result = np.clip(arr, min: double.NaN);

            Assert.AreEqual(np.float64, result.dtype);
            var data = result.GetData<double>();
            foreach (var v in data)
                Assert.IsTrue(double.IsNaN(v));
        }

        [TestMethod]
        public void Clip_NaNMaxKwarg_PropagatesNaN()
        {
            var arr = np.arange(7.0);

            var result = np.clip(arr, max: double.NaN);

            var data = result.GetData<double>();
            foreach (var v in data)
                Assert.IsTrue(double.IsNaN(v));
        }

        #endregion

        #region NumPy 2.x Parity — Zero-Dimensional (Scalar) Input

        [TestMethod]
        public void Clip_ScalarInput_BothBounds_PreservesNdim0()
        {
            // NumPy: np.clip(np.array(5), 3, 7) returns 0-d array with value 5.
            var s = NDArray.Scalar<int>(5);

            var result = np.clip(s, 3, 7);

            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(5, result.GetInt32());
        }

        [TestMethod]
        public void Clip_ScalarInput_AboveMax_ClampsAndPreservesNdim0()
        {
            // NumPy: np.clip(np.array(10), max=3) → 0-d with value 3.
            var s = NDArray.Scalar<int>(10);

            var result = np.clip(s, max: 3);

            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(3, result.GetInt32());
        }

        [TestMethod]
        public void Clip_ScalarInput_NoBounds_PreservesNdim0()
        {
            var s = NDArray.Scalar<int>(42);

            var result = np.clip(s);

            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(42, result.GetInt32());
        }

        #endregion

        #region NumPy 2.x Parity — Half / Complex / Decimal Dtypes via Kwarg

        [TestMethod]
        public void Clip_HalfArray_MinMaxKwargs_PreservesHalf()
        {
            // NumPy: np.clip(np.array([1,5,10,15], dtype=np.float16), 3, 10)
            //   == [3, 5, 10, 10], dtype=float16
            var h = np.array(new Half[] { (Half)1, (Half)5, (Half)10, (Half)15 });

            var result = np.clip(h, min: (Half)3, max: (Half)10);

            Assert.AreEqual(typeof(Half), result.dtype);
            Assert.AreEqual((Half)3, result.GetHalf(0));
            Assert.AreEqual((Half)5, result.GetHalf(1));
            Assert.AreEqual((Half)10, result.GetHalf(2));
            Assert.AreEqual((Half)10, result.GetHalf(3));
        }

        [TestMethod]
        public void Clip_ComplexArray_MinKwargOnly_LexOrdering()
        {
            // NumPy: np.clip(complex_arr, min=2+0j) uses lex ordering on (real,imag).
            // For inputs [1+1j, 5+5j, 10+10j], result is [2+0j, 5+5j, 10+10j].
            var c = np.array(new Complex[] { new(1, 1), new(5, 5), new(10, 10) });
            var lo = NDArray.Scalar<Complex>(new Complex(2, 0));

            var result = np.clip(c, min: lo);

            Assert.AreEqual(typeof(Complex), result.dtype);
            Assert.AreEqual(new Complex(2, 0), result.GetComplex(0));
            Assert.AreEqual(new Complex(5, 5), result.GetComplex(1));
            Assert.AreEqual(new Complex(10, 10), result.GetComplex(2));
        }

        [TestMethod]
        public void Clip_ComplexArray_BothArrayBoundsViaKwargs_LexOrdering()
        {
            // NumPy: np.clip([1+5j, 3+0j, 5+10j], 2+1j, 4+2j) → [2+1j, 3+0j, 4+2j]
            var a = np.array(new Complex[] { new(1, 5), new(3, 0), new(5, 10) });
            var lo = np.array(new Complex[] { new(2, 1) });
            var hi = np.array(new Complex[] { new(4, 2) });

            var result = np.clip(a, min: lo, max: hi);

            Assert.AreEqual(new Complex(2, 1), result.GetComplex(0));
            Assert.AreEqual(new Complex(3, 0), result.GetComplex(1));
            Assert.AreEqual(new Complex(4, 2), result.GetComplex(2));
        }

        #endregion

        #region NumPy 2.x Parity — Broadcasting via Kwarg Form

        [TestMethod]
        public void Clip_2D_RowVectorMinKwarg_BroadcastsAlongAxis0()
        {
            // NumPy: np.clip(arange(12).reshape(3,4), min=[1,2,3,4]) broadcasts
            // [1,2,3,4] along axis 0:
            //   [[1,2,3,4], [4,5,6,7], [8,9,10,11]]
            var a = np.arange(12).reshape(3, 4);
            var mn = np.array(new int[] { 1, 2, 3, 4 });

            var result = np.clip(a, min: mn);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(1L, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Clip_2D_ColumnVectorMaxKwarg_BroadcastsAlongAxis1()
        {
            // NumPy: np.clip(arange(12).reshape(3,4), max=[[10],[5],[8]]) broadcasts
            // along axis 1:
            //   row0: 0,1,2,3 clipped by 10 → 0,1,2,3
            //   row1: 4,5,6,7 clipped by 5  → 4,5,5,5
            //   row2: 8,9,10,11 clipped by 8 → 8,8,8,8
            var a = np.arange(12).reshape(3, 4);
            var mx = np.array(new int[] { 10, 5, 8 }).reshape(3, 1);

            var result = np.clip(a, max: mx);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(0L, 1, 2, 3, 4, 5, 5, 5, 8, 8, 8, 8);
        }

        [TestMethod]
        public void Clip_2D_RowMinAndColumnMaxViaKwargs_Broadcasts()
        {
            // Mixed broadcast: min=row(4), max=col(3,1).
            // Both broadcast independently against the 3x4 input.
            //   row0 (0..3):  bounded below by [1,2,3,4], above by 10 → 1,2,3,4
            //   row1 (4..7):  bounded below by [1,2,3,4], above by 5  → 4,5,5,5
            //   row2 (8..11): bounded below by [1,2,3,4], above by 8  → 8,8,8,8
            var a = np.arange(12).reshape(3, 4);
            var mn = np.array(new int[] { 1, 2, 3, 4 });
            var mx = np.array(new int[] { 10, 5, 8 }).reshape(3, 1);

            var result = np.clip(a, min: mn, max: mx);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(1L, 2, 3, 4, 4, 5, 5, 5, 8, 8, 8, 8);
        }

        #endregion

        #region NumPy 2.x Parity — Reversed / Strided Inputs via Kwarg

        [TestMethod]
        public void Clip_ReversedSliceInput_MinMaxKwargs()
        {
            // NumPy: np.clip(np.arange(20)[::-1], 3, 15) ==
            //   [15,15,15,15,15,14,13,12,11,10,9,8,7,6,5,4,3,3,3,3]
            var a = np.arange(20);
            var reversed = a["::-1"];

            var result = np.clip(reversed, min: 3, max: 15);

            result.Should().BeOfValues(15L, 15, 15, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 3, 3, 3);
        }

        #endregion

        #region NumPy 2.x Parity — Empty Arrays via Kwarg

        [TestMethod]
        public void Clip_EmptyArray_MinOnly_ReturnsEmpty()
        {
            var e = np.array(new double[0]);

            var result = np.clip(e, min: 3.0);

            Assert.AreEqual(0, result.size);
            Assert.AreEqual(np.float64, result.dtype);
        }

        [TestMethod]
        public void Clip_EmptyArray_MaxOnly_ReturnsEmpty()
        {
            var e = np.array(new double[0]);

            var result = np.clip(e, max: 5.0);

            Assert.AreEqual(0, result.size);
        }

        [TestMethod]
        public void Clip_EmptyArray_NoBounds_DtypeOverride()
        {
            // NumPy: np.clip(np.array([],dtype=float64), dtype=float32) → empty float32
            var e = np.array(new double[0]);

            var result = np.clip(e, dtype: NPTypeCode.Single);

            Assert.AreEqual(0, result.size);
            Assert.AreEqual(np.float32, result.dtype);
        }

        #endregion
    }
}
