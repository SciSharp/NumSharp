using System;
using System.Linq;
using AwesomeAssertions;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Tests for np.repeat matching NumPy 2.x behavior.
    /// Based on NumPy tests from numpy/_core/tests/test_multiarray.py and test_numeric.py.
    /// </summary>
    public class np_repeat_tests
    {
        #region Basic Scalar Repeat Tests

        [Test]
        public void Scalar_Int()
        {
            // np.repeat(3, 4) -> [3, 3, 3, 3]
            var nd = np.repeat(3, 4);
            nd.size.Should().Be(4);
            nd.ToArray<int>().Should().ContainInOrder(3, 3, 3, 3);
        }

        [Test]
        public void Scalar_Double()
        {
            var nd = np.repeat(2.5, 3);
            nd.size.Should().Be(3);
            nd.ToArray<double>().Should().ContainInOrder(2.5, 2.5, 2.5);
        }

        [Test]
        public void Scalar_ZeroRepeats()
        {
            var nd = np.repeat(5, 0);
            nd.size.Should().Be(0);
        }

        #endregion

        #region Basic Array Repeat Tests (NumPy test_numeric.py)

        [Test]
        public void Array_UniformRepeat()
        {
            // From NumPy: np.repeat([1, 2, 3], 2) -> [1, 1, 2, 2, 3, 3]
            var a = np.array(new int[] { 1, 2, 3 });
            var nd = np.repeat(a, 2);
            nd.ToArray<int>().Should().ContainInOrder(1, 1, 2, 2, 3, 3);
        }

        [Test]
        public void Array_ZeroRepeats()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var nd = np.repeat(a, 0);
            nd.size.Should().Be(0);
        }

        [Test]
        public void Array_EmptyInput()
        {
            var a = np.array(new int[0]);
            var nd = np.repeat(a, 2);
            nd.size.Should().Be(0);
        }

        #endregion

        #region 2D Array Tests (NumPy test_multiarray.py)

        [Test]
        public void Array2D_Flattens()
        {
            // From NumPy: np.repeat([[1, 2, 3], [4, 5, 6]], 2) flattens then repeats
            var m = np.array(new int[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3);
            var nd = np.repeat(m, 2);
            nd.ToArray<int>().Should().ContainInOrder(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6);
        }

        [Test]
        public void Simple2DArray()
        {
            var x = np.array(new int[][] { new int[] { 1, 2 }, new int[] { 3, 4 } });
            var nd = np.repeat(x, 2);
            nd.ToArray<int>().Should().ContainInOrder(1, 1, 2, 2, 3, 3, 4, 4);
        }

        #endregion

        #region Per-Element Repeats Tests (NumPy test_multiarray.py)

        [Test]
        public void ArrayRepeats_Basic()
        {
            // From NumPy: m.repeat([1, 3, 2, 1, 1, 2]) where m = [1, 2, 3, 4, 5, 6]
            var m = np.array(new int[] { 1, 2, 3, 4, 5, 6 });
            var repeats = np.array(new int[] { 1, 3, 2, 1, 1, 2 });
            var nd = np.repeat(m, repeats);
            nd.ToArray<int>().Should().ContainInOrder(1, 2, 2, 2, 3, 3, 4, 5, 6, 6);
        }

        [Test]
        public void ArrayRepeats_AllZeros()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var repeats = np.array(new int[] { 0, 0, 0 });
            var nd = np.repeat(a, repeats);
            nd.size.Should().Be(0);
        }

        [Test]
        public void ArrayRepeats_MixedZeroAndNonZero()
        {
            // np.repeat([1, 2, 3, 4], [0, 2, 0, 3]) -> [2, 2, 4, 4, 4]
            var a = np.array(new int[] { 1, 2, 3, 4 });
            var repeats = np.array(new int[] { 0, 2, 0, 3 });
            var nd = np.repeat(a, repeats);
            nd.ToArray<int>().Should().ContainInOrder(2, 2, 4, 4, 4);
        }

        #endregion

        #region Non-Contiguous Array Tests

        [Test]
        public void SlicedArray_StridedView()
        {
            // Sliced arrays should work correctly
            // np.arange returns int64 by default (NumPy 2.x)
            var a = np.arange(10);  // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            var sliced = a["::2"];  // [0, 2, 4, 6, 8]
            var nd = np.repeat(sliced, 2);
            nd.ToArray<long>().Should().ContainInOrder(0L, 0L, 2L, 2L, 4L, 4L, 6L, 6L, 8L, 8L);
        }

        [Test]
        public void TransposedArray()
        {
            // Transposed arrays (non-contiguous) should work correctly
            var a = np.array(new int[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3);
            var transposed = a.T;  // 3x2, ravel order: [1, 4, 2, 5, 3, 6]
            var nd = np.repeat(transposed, 2);
            nd.ToArray<int>().Should().ContainInOrder(1, 1, 4, 4, 2, 2, 5, 5, 3, 3, 6, 6);
        }

        #endregion

        #region All Dtype Tests

        [Test]
        public void Dtype_Boolean()
        {
            var a = np.array(new bool[] { true, false, true });
            var nd = np.repeat(a, 2);
            nd.ToArray<bool>().Should().ContainInOrder(true, true, false, false, true, true);
        }

        [Test]
        public void Dtype_Byte()
        {
            var a = np.array(new byte[] { 1, 2, 3 });
            var nd = np.repeat(a, 2);
            nd.ToArray<byte>().Should().ContainInOrder((byte)1, (byte)1, (byte)2, (byte)2, (byte)3, (byte)3);
        }

        [Test]
        public void Dtype_Int16()
        {
            var a = np.array(new short[] { 1, 2, 3 });
            var nd = np.repeat(a, 2);
            nd.ToArray<short>().Should().ContainInOrder((short)1, (short)1, (short)2, (short)2, (short)3, (short)3);
        }

        [Test]
        public void Dtype_UInt16()
        {
            var a = np.array(new ushort[] { 1, 2, 3 });
            var nd = np.repeat(a, 2);
            nd.ToArray<ushort>().Should().ContainInOrder((ushort)1, (ushort)1, (ushort)2, (ushort)2, (ushort)3, (ushort)3);
        }

        [Test]
        public void Dtype_Int32()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var nd = np.repeat(a, 2);
            nd.ToArray<int>().Should().ContainInOrder(1, 1, 2, 2, 3, 3);
        }

        [Test]
        public void Dtype_UInt32()
        {
            var a = np.array(new uint[] { 1, 2, 3 });
            var nd = np.repeat(a, 2);
            nd.ToArray<uint>().Should().ContainInOrder(1u, 1u, 2u, 2u, 3u, 3u);
        }

        [Test]
        public void Dtype_Int64()
        {
            var a = np.array(new long[] { 1, 2, 3 });
            var nd = np.repeat(a, 2);
            nd.ToArray<long>().Should().ContainInOrder(1L, 1L, 2L, 2L, 3L, 3L);
        }

        [Test]
        public void Dtype_UInt64()
        {
            var a = np.array(new ulong[] { 1, 2, 3 });
            var nd = np.repeat(a, 2);
            nd.ToArray<ulong>().Should().ContainInOrder(1UL, 1UL, 2UL, 2UL, 3UL, 3UL);
        }

        [Test]
        public void Dtype_Char()
        {
            var a = np.array(new char[] { 'a', 'b', 'c' });
            var nd = np.repeat(a, 2);
            nd.ToArray<char>().Should().ContainInOrder('a', 'a', 'b', 'b', 'c', 'c');
        }

        [Test]
        public void Dtype_Single()
        {
            var a = np.array(new float[] { 1.5f, 2.5f, 3.5f });
            var nd = np.repeat(a, 2);
            nd.ToArray<float>().Should().ContainInOrder(1.5f, 1.5f, 2.5f, 2.5f, 3.5f, 3.5f);
        }

        [Test]
        public void Dtype_Double()
        {
            var a = np.array(new double[] { 1.5, 2.5, 3.5 });
            var nd = np.repeat(a, 2);
            nd.ToArray<double>().Should().ContainInOrder(1.5, 1.5, 2.5, 2.5, 3.5, 3.5);
        }

        [Test]
        public void Dtype_Decimal()
        {
            var a = np.array(new decimal[] { 1.5m, 2.5m, 3.5m });
            var nd = np.repeat(a, 2);
            nd.ToArray<decimal>().Should().ContainInOrder(1.5m, 1.5m, 2.5m, 2.5m, 3.5m, 3.5m);
        }

        #endregion

        #region Array Repeats with All Dtypes

        [Test]
        public void ArrayRepeats_Double()
        {
            var a = np.array(new double[] { 1.1, 2.2, 3.3 });
            var repeats = np.array(new int[] { 1, 2, 3 });
            var nd = np.repeat(a, repeats);
            nd.ToArray<double>().Should().ContainInOrder(1.1, 2.2, 2.2, 3.3, 3.3, 3.3);
        }

        #endregion

        #region Large Repeat Counts

        [Test]
        public void LargeRepeatCount()
        {
            var a = np.array(new int[] { 1, 2 });
            var nd = np.repeat(a, 1000);
            nd.size.Should().Be(2000);
            nd.ToArray<int>().Take(5).Should().ContainInOrder(1, 1, 1, 1, 1);
            nd.ToArray<int>().TakeLast(5).Should().ContainInOrder(2, 2, 2, 2, 2);
        }

        #endregion

        #region Error Cases

        [Test]
        public void Error_NegativeScalarRepeat()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            Action act = () => np.repeat(a, -1);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Error_NegativeInArrayRepeat()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var repeats = np.array(new int[] { 1, -1, 2 });
            Action act = () => np.repeat(a, repeats);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Error_MismatchedArraySizes()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var repeats = np.array(new int[] { 1, 2 });  // Size mismatch
            Action act = () => np.repeat(a, repeats);
            act.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Error_NegativeScalarRepeatValue()
        {
            Action act = () => np.repeat(5, -1);
            act.Should().Throw<ArgumentException>();
        }

        #endregion

        #region Dtype Preservation Tests

        [Test]
        public void DtypePreservation_Int32()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var nd = np.repeat(a, 2);
            nd.typecode.Should().Be(NPTypeCode.Int32);
        }

        [Test]
        public void DtypePreservation_Double()
        {
            var a = np.array(new double[] { 1.0, 2.0, 3.0 });
            var nd = np.repeat(a, 2);
            nd.typecode.Should().Be(NPTypeCode.Double);
        }

        [Test]
        public void DtypePreservation_Boolean()
        {
            var a = np.array(new bool[] { true, false });
            var nd = np.repeat(a, 2);
            nd.typecode.Should().Be(NPTypeCode.Boolean);
        }

        #endregion
    }
}
