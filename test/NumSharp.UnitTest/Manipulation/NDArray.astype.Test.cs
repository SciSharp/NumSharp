using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// astype(copy:) semantics pinned against NumPy 2.4.2:
    ///   - copy:false with the dtype already satisfied returns THE INPUT ITSELF
    ///     (`a.astype(a.dtype, copy=False) is a` → True in NumPy);
    ///   - any dtype conversion ALWAYS materializes a new array and NEVER modifies
    ///     the input — with copy:true or copy:false alike.
    /// The former NumSharp behavior swapped the input's storage in-place on a
    /// copy:false conversion, which leaked out of np.allclose / np.isclose /
    /// np.where as silent operand dtype corruption.
    /// </summary>
    [TestClass]
    public class astypeTests
    {
        [TestMethod]
        public void Upcasting()
        {
            var nd = np.ones(new Shape(3, 3), np.int32);
            var int64_copied = nd.astype(np.int64, true);
            var int64 = nd.astype(np.int64, false);

            // A conversion always yields a fresh array; the input keeps its dtype.
            Assert.IsTrue(int64_copied.Array != nd.Array);
            Assert.IsTrue(int64.Array != nd.Array);
            nd.GetTypeCode.Should().Be(NPTypeCode.Int32);
            int64_copied.GetTypeCode.Should().Be(NPTypeCode.Int64);
            int64.GetTypeCode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void UpcastingByteToLong()
        {
            var nd = np.ones(new Shape(3, 3), np.uint8);
            var int64_copied = nd.astype(np.int64, true);
            var int64 = nd.astype(np.int64, false);

            Assert.IsTrue(int64_copied.Array != nd.Array);
            Assert.IsTrue(int64.Array != nd.Array);
            nd.GetTypeCode.Should().Be(NPTypeCode.Byte);
            int64_copied.GetTypeCode.Should().Be(NPTypeCode.Int64);
            int64.GetTypeCode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void UpcastingCharsToLong()
        {
            var nd = np.ones(new Shape(3, 3), np.@char);
            var int64_copied = nd.astype(np.int64, true);
            var int64 = nd.astype(np.int64, false);

            Assert.IsTrue(int64_copied.Array != nd.Array);
            Assert.IsTrue(int64.Array != nd.Array);
            nd.GetTypeCode.Should().Be(NPTypeCode.Char);
            int64_copied.GetTypeCode.Should().Be(NPTypeCode.Int64);
            int64.GetTypeCode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void DowncastingIntToShort()
        {
            var nd = np.ones(new Shape(3, 3), np.int32);
            var int16_copied = nd.astype(np.int16, true);
            var int16 = nd.astype(np.int16, false);

            Assert.IsTrue(int16_copied.Array != nd.Array);
            Assert.IsTrue(int16.Array != nd.Array);
            nd.GetTypeCode.Should().Be(NPTypeCode.Int32);
            int16_copied.GetTypeCode.Should().Be(NPTypeCode.Int16);
            int16.GetTypeCode.Should().Be(NPTypeCode.Int16);
        }

        [TestMethod]
        public void DowncastingDoubleToInt()
        {
            var nd = np.ones(new Shape(3, 3), np.float64);
            for (int i = 0; i < nd.size; i++)
            {
                nd.SetAtIndex(1.3d, i);
            }

            var int32_copied = nd.astype(np.int32, true);
            var int32 = nd.astype(np.int32, false);

            Assert.IsTrue(int32_copied.Array != nd.Array);
            Assert.IsTrue(int32.Array != nd.Array);
            int32_copied.GetTypeCode.Should().Be(NPTypeCode.Int32);
            int32.GetTypeCode.Should().Be(NPTypeCode.Int32);

            // The converted arrays hold the truncated values; the input still holds 1.3d.
            for (int i = 0; i < nd.size; i++)
            {
                int32.GetAtIndex<int>(i).Should().Be(1);
                int32_copied.GetAtIndex<int>(i).Should().Be(1);
                nd.GetAtIndex<double>(i).Should().Be(1.3d);
            }
        }

        [TestMethod]
        public void DowncastingIntToUShort()
        {
            var nd = np.ones(new Shape(3, 3), np.int32);
            nd[2, 2].Data<int>()[0].Should().Be(1);
            var int16_copied = nd.astype(np.uint16, true);
            var int16 = nd.astype(np.uint16, false);

            Assert.IsTrue(int16_copied.Array != nd.Array);
            Assert.IsTrue(int16.Array != nd.Array);
            nd.GetTypeCode.Should().Be(NPTypeCode.Int32);
            int16_copied.GetTypeCode.Should().Be(NPTypeCode.UInt16);
            int16.GetTypeCode.Should().Be(NPTypeCode.UInt16);
            int16.GetAtIndex<ushort>(0).Should().Be((ushort)1);
        }

        [TestMethod]
        public void SameDtype_NoCopy_ReturnsSelf()
        {
            // NumPy: a.astype(a.dtype, copy=False) is a → True.
            var nd = np.ones(new Shape(3, 3), np.int32);
            var same = nd.astype(np.int32, false);
            Assert.IsTrue(ReferenceEquals(same, nd));

            // copy:true with the same dtype still copies.
            var copied = nd.astype(np.int32, true);
            Assert.IsFalse(ReferenceEquals(copied, nd));
            Assert.IsTrue(copied.Array != nd.Array);
        }

        [TestMethod]
        public void ConversionDoesNotMutate_OperandsOfCallers()
        {
            // The historical bug this file pins: np.allclose / np.isclose / np.where
            // internally astype(copy:false) their operands — that must never flip the
            // caller's arrays to float64/bool in place.
            var x = np.array(new float[] { 1f, 2f, 3f });
            var y = np.array(new float[] { 1f, 2f, 3f });
            np.allclose(x, y).Should().BeTrue();
            x.GetTypeCode.Should().Be(NPTypeCode.Single);
            y.GetTypeCode.Should().Be(NPTypeCode.Single);

            var cond = np.array(new int[] { 1, 0, 2 });
            using var _ = np.where(cond, np.array(new double[] { 1, 1, 1 }), np.array(new double[] { 2, 2, 2 }));
            cond.GetTypeCode.Should().Be(NPTypeCode.Int32);
        }

        [TestMethod]
        public void CastEmptyNDArray()
        {
            var nd = new NDArray(NPTypeCode.Int32);
            var int16_copied = nd.astype(np.uint16, true);
            int16_copied.Should().BeEquivalentTo(nd);
            int16_copied.Shape.IsEmpty.Should().BeTrue();

            // A copy:false conversion on the empty sentinel also leaves the input alone.
            var int16 = nd.astype(np.uint16, false);
            int16.GetTypeCode.Should().Be(NPTypeCode.UInt16);
            nd.GetTypeCode.Should().Be(NPTypeCode.Int32);
        }

        [TestMethod, Ignore("String dtype is not supported")]
        public void CastingStringToByte()
        {
            throw new NotSupportedException();
            //var nd = np.ones(np.chars, 3, 3);
            //nd[2, 2].Data<string>()[0].Should().Be("1");
            //var output_copied = nd.astype(np.uint8, true);
            //var output = nd.astype(np.uint8, false);

            ////test copying
            //output_copied.Array.Should().Equal(nd.Array);
            //output.Array.Should().Equal(nd.Array);
            //output_copied.Array.GetType().GetElementType().Should().Be<byte>();
            //output.Array.GetType().GetElementType().Should().Be<byte>();
        }

        [TestMethod, Ignore("String dtype is not supported")]
        public void CastingByteToString()
        {
            throw new NotSupportedException();
            //var nd = np.ones(new Shape(3, 3), np.uint8);
            //nd[2, 2].Data<byte>()[0].Should().Be(1);
            //var output_copied = nd.astype(np.chars, true);
            //var output = nd.astype(np.chars, false);

            //output_copied[2, 2].Data<string>()[0].Should().Be("1");

            ////test copying
            //output_copied.Array.Should().Equal(nd.Array);
            //output.Array.Should().Equal(nd.Array);
            //output_copied.Array.GetType().GetElementType().Should().Be<string>();
            //output.Array.GetType().GetElementType().Should().Be<string>();
        }


        [TestMethod, Ignore("Complex dtype is not supported yet")] //TODO!
        public void CastingIntToComplex()
        {
            //var nd = np.ones(new Shape(3, 3), np.int32);
            //nd[2, 2].Data<int>()[0].Should().Be(1);
            //var output_copied = nd.astype(np.complex128, true);
            //var output = nd.astype(np.complex128, false);
            //
            ////test copying
            //output_copied.Array.Should().Equal(nd.Array);
            //output.Array.Should().Equal(nd.Array);
            //output_copied.Array.GetType().GetElementType().Should().Be<Complex>();
            //output.Array.GetType().GetElementType().Should().Be<Complex>();
            //
            //output_copied[2, 2].GetComplex(0).Should().Be(new Complex(1, 0));
        }
    }
}
