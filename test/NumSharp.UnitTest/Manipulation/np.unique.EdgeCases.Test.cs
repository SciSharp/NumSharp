using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using TUnit.Core;

namespace NumSharp.UnitTest.Manipulation;

/// <summary>
/// Comprehensive edge case tests for np.unique sorting fix (commit c0eed5d1).
/// Verifies that np.unique returns SORTED unique values, matching NumPy 2.x behavior.
/// </summary>
public class np_unique_EdgeCases_Test
{
        #region Empty Arrays

        [Test]
        public void Unique_EmptyArray_Int32()
        {
            // >>> np.unique(np.array([], dtype=np.int32))
            // array([], dtype=int32)
            var arr = np.array(Array.Empty<int>());
            var result = np.unique(arr);

            result.size.Should().Be(0);
            result.shape.Should().BeEquivalentTo(new long[] { 0 });
            result.dtype.Should().Be(typeof(int));
        }

        [Test]
        public void Unique_EmptyArray_Double()
        {
            var arr = np.array(Array.Empty<double>());
            var result = np.unique(arr);

            result.size.Should().Be(0);
            result.dtype.Should().Be(typeof(double));
        }

        [Test]
        public void Unique_EmptyArray_Boolean()
        {
            var arr = np.array(Array.Empty<bool>());
            var result = np.unique(arr);

            result.size.Should().Be(0);
            result.dtype.Should().Be(typeof(bool));
        }

        #endregion

        #region Single Element

        [Test]
        public void Unique_SingleElement_Int32()
        {
            // >>> np.unique(np.array([42]))
            // array([42])
            var arr = np.array(new int[] { 42 });
            var result = np.unique(arr);

            result.Should().BeShaped(1).And.BeOfValues(42);
        }

        [Test]
        public void Unique_SingleElement_Double()
        {
            var arr = np.array(new double[] { 3.14 });
            var result = np.unique(arr);

            result.Should().BeShaped(1).And.BeOfValues(3.14);
        }

        [Test]
        public void Unique_SingleElement_Boolean_True()
        {
            var arr = np.array(new bool[] { true });
            var result = np.unique(arr);

            result.Should().BeShaped(1).And.BeOfValues(true);
        }

        [Test]
        public void Unique_SingleElement_Boolean_False()
        {
            var arr = np.array(new bool[] { false });
            var result = np.unique(arr);

            result.Should().BeShaped(1).And.BeOfValues(false);
        }

        #endregion

        #region All Duplicates

        [Test]
        public void Unique_AllDuplicates_Int32()
        {
            // >>> np.unique(np.array([5, 5, 5, 5]))
            // array([5])
            var arr = np.array(new int[] { 5, 5, 5, 5 });
            var result = np.unique(arr);

            result.Should().BeShaped(1).And.BeOfValues(5);
        }

        [Test]
        public void Unique_AllDuplicates_Double()
        {
            var arr = np.array(new double[] { 2.71, 2.71, 2.71 });
            var result = np.unique(arr);

            result.Should().BeShaped(1).And.BeOfValues(2.71);
        }

        [Test]
        public void Unique_AllDuplicates_Boolean_AllTrue()
        {
            // >>> np.unique(np.array([True, True, True]))
            // array([ True])
            var arr = np.array(new bool[] { true, true, true });
            var result = np.unique(arr);

            result.Should().BeShaped(1).And.BeOfValues(true);
        }

        [Test]
        public void Unique_AllDuplicates_Boolean_AllFalse()
        {
            // >>> np.unique(np.array([False, False, False]))
            // array([False])
            var arr = np.array(new bool[] { false, false, false });
            var result = np.unique(arr);

            result.Should().BeShaped(1).And.BeOfValues(false);
        }

        #endregion

        #region Boolean Arrays

        [Test]
        public void Unique_Boolean_MixedValues()
        {
            // >>> np.unique(np.array([True, False, True, False, True]))
            // array([False,  True])
            // Note: False sorts before True (False=0, True=1)
            var arr = np.array(new bool[] { true, false, true, false, true });
            var result = np.unique(arr);

            result.Should().BeShaped(2);
            // Sorted: False comes before True
            result.GetBoolean(0).Should().BeFalse();
            result.GetBoolean(1).Should().BeTrue();
        }

        #endregion

        #region All 12 Supported Dtypes

        [Test]
        public void Unique_AllDtypes_Byte()
        {
            var arr = np.array(new byte[] { 3, 1, 2, 1 });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues((byte)1, (byte)2, (byte)3);
            result.dtype.Should().Be(typeof(byte));
        }

        [Test]
        public void Unique_AllDtypes_Int16()
        {
            var arr = np.array(new short[] { 3, 1, 2, 1 });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues((short)1, (short)2, (short)3);
            result.dtype.Should().Be(typeof(short));
        }

        [Test]
        public void Unique_AllDtypes_UInt16()
        {
            var arr = np.array(new ushort[] { 3, 1, 2, 1 });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues((ushort)1, (ushort)2, (ushort)3);
            result.dtype.Should().Be(typeof(ushort));
        }

        [Test]
        public void Unique_AllDtypes_Int32()
        {
            var arr = np.array(new int[] { 3, 1, 2, 1 });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues(1, 2, 3);
            result.dtype.Should().Be(typeof(int));
        }

        [Test]
        public void Unique_AllDtypes_UInt32()
        {
            var arr = np.array(new uint[] { 3, 1, 2, 1 });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues(1u, 2u, 3u);
            result.dtype.Should().Be(typeof(uint));
        }

        [Test]
        public void Unique_AllDtypes_Int64()
        {
            var arr = np.array(new long[] { 3, 1, 2, 1 });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues(1L, 2L, 3L);
            result.dtype.Should().Be(typeof(long));
        }

        [Test]
        public void Unique_AllDtypes_UInt64()
        {
            var arr = np.array(new ulong[] { 3, 1, 2, 1 });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues(1UL, 2UL, 3UL);
            result.dtype.Should().Be(typeof(ulong));
        }

        [Test]
        public void Unique_AllDtypes_Char()
        {
            var arr = np.array(new char[] { 'c', 'a', 'b', 'a' });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues('a', 'b', 'c');
            result.dtype.Should().Be(typeof(char));
        }

        [Test]
        public void Unique_AllDtypes_Single()
        {
            var arr = np.array(new float[] { 3.0f, 1.0f, 2.0f, 1.0f });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues(1.0f, 2.0f, 3.0f);
            result.dtype.Should().Be(typeof(float));
        }

        [Test]
        public void Unique_AllDtypes_Double()
        {
            var arr = np.array(new double[] { 3.0, 1.0, 2.0, 1.0 });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues(1.0, 2.0, 3.0);
            result.dtype.Should().Be(typeof(double));
        }

        [Test]
        public void Unique_AllDtypes_Decimal()
        {
            var arr = np.array(new decimal[] { 3.0m, 1.0m, 2.0m, 1.0m });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues(1.0m, 2.0m, 3.0m);
            result.dtype.Should().Be(typeof(decimal));
        }

        #endregion

        #region NaN and Infinity (Float Types)

        [Test]
        public void Unique_NaN_Double()
        {
            // >>> np.unique(np.array([1.0, np.nan, 2.0, np.nan, 1.0, np.inf, -np.inf]))
            // array([-inf,   1.,   2.,  inf,  nan])
            // Note: NaN appears once at the end (after sorting), Inf values are sorted normally
            var arr = np.array(new double[] { 1.0, double.NaN, 2.0, double.NaN, 1.0, double.PositiveInfinity, double.NegativeInfinity });
            var result = np.unique(arr);

            // NumPy deduplicates NaN to single instance
            // Sorted order: -inf, 1.0, 2.0, inf, nan
            result.Should().BeShaped(5);

            result.GetDouble(0).Should().Be(double.NegativeInfinity);
            result.GetDouble(1).Should().Be(1.0);
            result.GetDouble(2).Should().Be(2.0);
            result.GetDouble(3).Should().Be(double.PositiveInfinity);
            double.IsNaN(result.GetDouble(4)).Should().BeTrue();
        }

        [Test]
        public void Unique_NaN_Single()
        {
            // >>> np.unique(np.array([1.0, np.nan, 2.0, np.nan], dtype=np.float32))
            // array([ 1.,  2., nan], dtype=float32)
            var arr = np.array(new float[] { 1.0f, float.NaN, 2.0f, float.NaN });
            var result = np.unique(arr);

            result.Should().BeShaped(3);
            result.GetSingle(0).Should().Be(1.0f);
            result.GetSingle(1).Should().Be(2.0f);
            float.IsNaN(result.GetSingle(2)).Should().BeTrue();
        }

        [Test]
        public void Unique_Infinity_Only()
        {
            var arr = np.array(new double[] { double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity });
            var result = np.unique(arr);

            result.Should().BeShaped(2);
            result.GetDouble(0).Should().Be(double.NegativeInfinity);
            result.GetDouble(1).Should().Be(double.PositiveInfinity);
        }

        [Test]
        public void Unique_AllNaN()
        {
            // All NaN values should deduplicate to one
            var arr = np.array(new double[] { double.NaN, double.NaN, double.NaN });
            var result = np.unique(arr);

            result.Should().BeShaped(1);
            double.IsNaN(result.GetDouble(0)).Should().BeTrue();
        }

        #endregion

        #region Sliced/Non-Contiguous Arrays

        [Test]
        public void Unique_SlicedArray_Column()
        {
            // Slicing a column creates a non-contiguous view
            var arr = np.arange(20).reshape(4, 5);
            var sliced = arr[":, 0"]; // First column: [0, 5, 10, 15]

            var result = np.unique(sliced);

            result.Should().BeShaped(4).And.BeOfValues(0L, 5L, 10L, 15L);
        }

        [Test]
        public void Unique_SlicedArray_StridedSlice()
        {
            // >>> arr = np.arange(20).reshape(4, 5)
            // >>> np.unique(arr[::2, 1::2])  # [[1, 3], [11, 13]]
            // array([ 1,  3, 11, 13])
            var arr = np.arange(20).reshape(4, 5);
            var sliced = arr["::2, 1::2"];

            var result = np.unique(sliced);

            result.Should().BeShaped(4).And.BeOfValues(1L, 3L, 11L, 13L);
        }

        [Test]
        public void Unique_SlicedArray_WithDuplicates()
        {
            // Create array with duplicates, then slice
            var arr = np.array(new int[] { 5, 3, 5, 1, 3, 1, 5, 3 }).reshape(2, 4);
            var sliced = arr[":, ::2"]; // [[5, 5], [3, 5]]

            var result = np.unique(sliced);

            result.Should().BeShaped(2).And.BeOfValues(3, 5);
        }

        [Test]
        public void Unique_ReversedArray()
        {
            // >>> np.unique(np.array([9, 7, 5, 3, 1])[::-1])
            // array([1, 3, 5, 7, 9])
            var arr = np.array(new int[] { 9, 7, 5, 3, 1 });
            var reversed = arr["::-1"];

            var result = np.unique(reversed);

            result.Should().BeShaped(5).And.BeOfValues(1, 3, 5, 7, 9);
        }

        #endregion

        #region Negative Values

        [Test]
        public void Unique_NegativeValues_Int32()
        {
            // >>> np.unique(np.array([-3, -1, -2, 0, 1, -1]))
            // array([-3, -2, -1,  0,  1])
            var arr = np.array(new int[] { -3, -1, -2, 0, 1, -1 });
            var result = np.unique(arr);

            result.Should().BeShaped(5).And.BeOfValues(-3, -2, -1, 0, 1);
        }

        [Test]
        public void Unique_NegativeValues_Double()
        {
            var arr = np.array(new double[] { -3.5, -1.5, -2.5, 0.0, 1.5, -1.5 });
            var result = np.unique(arr);

            result.Should().BeShaped(5).And.BeOfValues(-3.5, -2.5, -1.5, 0.0, 1.5);
        }

        [Test]
        public void Unique_AllNegative()
        {
            var arr = np.array(new int[] { -5, -3, -1, -3, -5 });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues(-5, -3, -1);
        }

        #endregion

        #region Large Values (Boundary Conditions)

        [Test]
        public void Unique_LargeValues_Int64()
        {
            // >>> np.unique(np.array([2**62, -2**62, 0, 2**62], dtype=np.int64))
            // array([-4611686018427387904,                    0,  4611686018427387904])
            long large = 1L << 62;
            var arr = np.array(new long[] { large, -large, 0, large });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues(-large, 0L, large);
        }

        [Test]
        public void Unique_LargeValues_UInt64()
        {
            ulong large = 1UL << 62;
            var arr = np.array(new ulong[] { large, 0, large, ulong.MaxValue });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues(0UL, large, ulong.MaxValue);
        }

        [Test]
        public void Unique_MinMaxValues_Int32()
        {
            var arr = np.array(new int[] { int.MaxValue, int.MinValue, 0, int.MaxValue });
            var result = np.unique(arr);

            result.Should().BeShaped(3).And.BeOfValues(int.MinValue, 0, int.MaxValue);
        }

        #endregion

        #region Sorting Verification

        [Test]
        public void Unique_VerifySorting_UnsortedInput()
        {
            // This is the core test for the sorting fix
            // >>> np.unique(np.array([5, 2, 9, 2, 5, 1, 8]))
            // array([1, 2, 5, 8, 9])
            var arr = np.array(new int[] { 5, 2, 9, 2, 5, 1, 8 });
            var result = np.unique(arr);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 5, 8, 9);
        }

        [Test]
        public void Unique_VerifySorting_ReverseInput()
        {
            // Input is in reverse order - verify output is sorted ascending
            var arr = np.array(new int[] { 10, 8, 6, 4, 2 });
            var result = np.unique(arr);

            result.Should().BeShaped(5).And.BeOfValues(2, 4, 6, 8, 10);
        }

        [Test]
        public void Unique_VerifySorting_AlreadySorted()
        {
            // Input already sorted - verify it stays sorted
            var arr = np.array(new int[] { 1, 1, 2, 3, 3, 3, 4, 5 });
            var result = np.unique(arr);

            result.Should().BeShaped(5).And.BeOfValues(1, 2, 3, 4, 5);
        }

        #endregion

        #region Multidimensional Arrays

        [Test]
        public void Unique_2DArray_Flattens()
        {
            // np.unique flattens multidimensional arrays
            // >>> np.unique(np.array([[3, 1], [2, 1]]))
            // array([1, 2, 3])
            var arr = np.array(new int[,] { { 3, 1 }, { 2, 1 } });
            var result = np.unique(arr);

            result.ndim.Should().Be(1);
            result.Should().BeShaped(3).And.BeOfValues(1, 2, 3);
        }

        [Test]
        public void Unique_3DArray_Flattens()
        {
            var arr = np.array(new int[,,] { { { 3, 1 }, { 2, 1 } }, { { 3, 4 }, { 1, 2 } } });
            var result = np.unique(arr);

            result.ndim.Should().Be(1);
            result.Should().BeShaped(4).And.BeOfValues(1, 2, 3, 4);
        }

        #endregion
}
