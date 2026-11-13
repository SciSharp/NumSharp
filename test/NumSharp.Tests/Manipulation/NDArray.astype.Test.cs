using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Manipulation
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
        /// <summary>int32 to int64: both the copying and the copy=False call return a new array (the dtype changes); the input keeps int32.</summary>
        [TestMethod]
        public void Upcasting()
        {
            var nd = np.ones(new Shape(3, 3), np.int32);
            var int64_copied = nd.astype(np.int64);
            var int64 = nd.astype(np.int64, copy: false);

            // A conversion always yields a fresh array; the input keeps its dtype.
            Assert.IsTrue(int64_copied.Array != nd.Array);
            Assert.IsTrue(int64.Array != nd.Array);
            nd.GetTypeCode.Should().Be(NPTypeCode.Int32);
            int64_copied.GetTypeCode.Should().Be(NPTypeCode.Int64);
            int64.GetTypeCode.Should().Be(NPTypeCode.Int64);
        }

        /// <summary>uint8 to int64: a new array either way; the input keeps uint8.</summary>
        [TestMethod]
        public void UpcastingByteToLong()
        {
            var nd = np.ones(new Shape(3, 3), np.uint8);
            var int64_copied = nd.astype(np.int64);
            var int64 = nd.astype(np.int64, copy: false);

            Assert.IsTrue(int64_copied.Array != nd.Array);
            Assert.IsTrue(int64.Array != nd.Array);
            nd.GetTypeCode.Should().Be(NPTypeCode.Byte);
            int64_copied.GetTypeCode.Should().Be(NPTypeCode.Int64);
            int64.GetTypeCode.Should().Be(NPTypeCode.Int64);
        }

        /// <summary>char to int64: a new array either way; the input keeps char.</summary>
        [TestMethod]
        public void UpcastingCharsToLong()
        {
            var nd = np.ones(new Shape(3, 3), np.@char);
            var int64_copied = nd.astype(np.int64);
            var int64 = nd.astype(np.int64, copy: false);

            Assert.IsTrue(int64_copied.Array != nd.Array);
            Assert.IsTrue(int64.Array != nd.Array);
            nd.GetTypeCode.Should().Be(NPTypeCode.Char);
            int64_copied.GetTypeCode.Should().Be(NPTypeCode.Int64);
            int64.GetTypeCode.Should().Be(NPTypeCode.Int64);
        }

        /// <summary>int32 to int16: a new array either way; the input keeps int32.</summary>
        [TestMethod]
        public void DowncastingIntToShort()
        {
            var nd = np.ones(new Shape(3, 3), np.int32);
            var int16_copied = nd.astype(np.int16);
            var int16 = nd.astype(np.int16, copy: false);

            Assert.IsTrue(int16_copied.Array != nd.Array);
            Assert.IsTrue(int16.Array != nd.Array);
            nd.GetTypeCode.Should().Be(NPTypeCode.Int32);
            int16_copied.GetTypeCode.Should().Be(NPTypeCode.Int16);
            int16.GetTypeCode.Should().Be(NPTypeCode.Int16);
        }

        /// <summary>float64 1.3 to int32 truncates to 1 in both results and leaves the input's 1.3.</summary>
        [TestMethod]
        public void DowncastingDoubleToInt()
        {
            var nd = np.ones(new Shape(3, 3), np.float64);
            for (int i = 0; i < nd.size; i++)
            {
                nd.SetAtIndex(1.3d, i);
            }

            var int32_copied = nd.astype(np.int32);
            var int32 = nd.astype(np.int32, copy: false);

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

        /// <summary>int32 to uint16: a new array either way, the values kept.</summary>
        [TestMethod]
        public void DowncastingIntToUShort()
        {
            var nd = np.ones(new Shape(3, 3), np.int32);
            nd[2, 2].Data<int>()[0].Should().Be(1);
            var int16_copied = nd.astype(np.uint16);
            var int16 = nd.astype(np.uint16, copy: false);

            Assert.IsTrue(int16_copied.Array != nd.Array);
            Assert.IsTrue(int16.Array != nd.Array);
            nd.GetTypeCode.Should().Be(NPTypeCode.Int32);
            int16_copied.GetTypeCode.Should().Be(NPTypeCode.UInt16);
            int16.GetTypeCode.Should().Be(NPTypeCode.UInt16);
            int16.GetAtIndex<ushort>(0).Should().Be((ushort)1);
        }

        /// <summary>NumPy: a.astype(a.dtype, copy=False) is a; with copy the same dtype still copies.</summary>
        [TestMethod]
        public void SameDtype_NoCopy_ReturnsSelf()
        {
            // NumPy: a.astype(a.dtype, copy=False) is a → True.
            var nd = np.ones(new Shape(3, 3), np.int32);
            var same = nd.astype(np.int32, copy: false);
            Assert.IsTrue(ReferenceEquals(same, nd));

            // copy:true with the same dtype still copies.
            var copied = nd.astype(np.int32);
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

        /// <summary>The empty sentinel casts to an empty uint16 array (copy or not) and stays int32 itself.</summary>
        [TestMethod]
        public void CastEmptyNDArray()
        {
            var nd = new NDArray(NPTypeCode.Int32);
            var int16_copied = nd.astype(np.uint16);
            int16_copied.Should().BeEquivalentTo(nd);
            int16_copied.Shape.IsEmpty.Should().BeTrue();

            // A copy:false conversion on the empty sentinel also leaves the input alone.
            var int16 = nd.astype(np.uint16, copy: false);
            int16.GetTypeCode.Should().Be(NPTypeCode.UInt16);
            nd.GetTypeCode.Should().Be(NPTypeCode.Int32);
        }

        /// <summary>
        ///     Text parses into numbers as NumPy's cast does: <c>np.ones((3, 3), dtype='U1')</c> holds the text <c>'1'</c> in
        ///     every element, and <c>astype(np.uint8)</c> parses each one with Python's <c>int()</c> — a uint8 array of ones
        ///     (NumPy 2.4.2). The dtype changes, so <c>copy=False</c> cannot hand the input back: both calls return a new
        ///     array and leave the text array as it was.
        /// </summary>
        [TestMethod]
        public void CastingStringToByte()
        {
            var nd = np.ones(new Shape(3, 3), np.dtype("U1"));
            nd.item(2, 2).Should().Be("1");
            var output_copied = nd.astype(np.uint8);
            var output = nd.astype(np.uint8, copy: false);

            output_copied.Should().BeOfType<byte>().And.BeShaped(3, 3).And.AllValuesBe((byte)1);
            output.Should().BeOfType<byte>().And.BeShaped(3, 3).And.AllValuesBe((byte)1);
            output.Should().NotBeSameAs(nd.Storage);
            nd.dtype.str.Should().Be("<U1");
            nd.item(2, 2).Should().Be("1");
        }

        /// <summary>
        ///     Numbers become text as NumPy's cast writes them, each value's <c>str()</c>: <c>np.ones((3, 3),
        ///     np.uint8).astype('U')</c> is a <c>&lt;U3</c> array of <c>'1'</c> (NumPy 2.4.2; an unsized <c>'U'</c> takes
        ///     the widest text a uint8 can need, three characters). Both <c>copy</c> forms return a new array; the input keeps
        ///     its uint8 values.
        /// </summary>
        [TestMethod]
        public void CastingByteToString()
        {
            var nd = np.ones(new Shape(3, 3), np.uint8);
            nd.item(2, 2).Should().Be((byte)1);
            var output_copied = nd.astype(np.dtype("U"));
            var output = nd.astype(np.dtype("U"), copy: false);

            output_copied.dtype.str.Should().Be("<U3");
            output.dtype.str.Should().Be("<U3");
            output_copied.Should().BeShaped(3, 3);
            NDArray.AsStringArray(output_copied).Should().HaveCount(9).And.OnlyContain(s => s == "1");
            NDArray.AsStringArray(output).Should().HaveCount(9).And.OnlyContain(s => s == "1");
            output.Should().NotBeSameAs(nd.Storage);
            nd.Should().BeOfType<byte>().And.AllValuesBe((byte)1);
        }


        /// <summary>Placeholder for int32 to complex128 (its body is commented out; the complex casts are covered by the casting suites).</summary>
        [TestMethod, Ignore("Complex dtype is not supported yet")] //TODO!
        public void CastingIntToComplex()
        {
            //var nd = np.ones(new Shape(3, 3), np.int32);
            //nd[2, 2].Data<int>()[0].Should().Be(1);
            //var output_copied = nd.astype(np.complex128);
            //var output = nd.astype(np.complex128, copy: false);
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
