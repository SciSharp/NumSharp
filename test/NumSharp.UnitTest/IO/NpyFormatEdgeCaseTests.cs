using System;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.IO;

namespace NumSharp.UnitTest.IO
{
    /// <summary>
    /// Battle-tested edge cases for .npy/.npz format handling.
    /// </summary>
    [TestClass]
    public class NpyFormatEdgeCaseTests
    {
        #region Sliced and Non-Contiguous Arrays

        [TestMethod]
        public void Save_Load_SlicedArray_WithOffset()
        {
            // Regression test for Shape.offset bug
            var arr = np.arange(20).reshape(4, 5);
            var sliced = arr["1:3", "1:4"]; // Shape: [2, 3], offset: 6

            Assert.IsFalse(sliced.Shape.IsContiguous);
            Assert.AreEqual(6, sliced.Shape.offset);

            using var ms = new MemoryStream();
            np.save(ms, sliced);
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.IsTrue(new long[] { 2, 3 }.SequenceEqual(loaded.shape));
            // np.arange yields Int64, so read the element back at that dtype.
            var slicedFlat = sliced.flatten();
            var loadedFlat = loaded.flatten();
            Assert.AreEqual(6L, slicedFlat.GetAtIndex<long>(0));
            Assert.AreEqual(6L, loadedFlat.GetAtIndex<long>(0));
            Assert.IsTrue(np.array_equal(sliced, loaded));
        }

        [TestMethod]
        public void Save_Load_ColumnSlice()
        {
            var arr = np.arange(20).reshape(4, 5);
            var colSlice = arr[":, 1:3"]; // All rows, columns 1-2

            Assert.IsFalse(colSlice.Shape.IsContiguous);

            using var ms = new MemoryStream();
            np.save(ms, colSlice);
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.IsTrue(np.array_equal(colSlice, loaded));
        }

        [TestMethod]
        public void Save_Load_StepSlice()
        {
            var arr = np.arange(10);
            var stepped = arr["::2"]; // Every other element

            Assert.AreEqual(5, stepped.size);

            using var ms = new MemoryStream();
            np.save(ms, stepped);
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.IsTrue(np.array_equal(stepped, loaded));
        }

        [TestMethod]
        public void Save_Load_TransposedArray()
        {
            var arr = np.arange(12).reshape(3, 4);
            var transposed = arr.T;

            Assert.IsFalse(transposed.Shape.IsContiguous);

            using var ms = new MemoryStream();
            np.save(ms, transposed);
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.IsTrue(new long[] { 4, 3 }.SequenceEqual(loaded.shape));
            Assert.IsTrue(np.array_equal(transposed, loaded));
        }

        [TestMethod]
        public void Save_Load_BroadcastView()
        {
            var arr = np.arange(3);
            var broadcasted = np.broadcast_to(arr, (3, 3));

            Assert.IsTrue(broadcasted.Shape.IsBroadcasted);

            using var ms = new MemoryStream();
            np.save(ms, broadcasted);
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.IsTrue(np.array_equal(broadcasted, loaded));
        }

        #endregion

        #region Special Float Values

        [TestMethod]
        public void Save_Load_SpecialFloatValues()
        {
            var arr = np.array(new double[] {
                double.NaN,
                double.PositiveInfinity,
                double.NegativeInfinity,
                0.0,
                double.MaxValue,
                double.MinValue,
                double.Epsilon
            });

            using var ms = new MemoryStream();
            np.save(ms, arr);
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.IsTrue(double.IsNaN(loaded.GetDouble(0)));
            Assert.IsTrue(double.IsPositiveInfinity(loaded.GetDouble(1)));
            Assert.IsTrue(double.IsNegativeInfinity(loaded.GetDouble(2)));
            Assert.AreEqual(0.0, loaded.GetDouble(3));
            Assert.AreEqual(double.MaxValue, loaded.GetDouble(4));
            Assert.AreEqual(double.MinValue, loaded.GetDouble(5));
            Assert.AreEqual(double.Epsilon, loaded.GetDouble(6));
        }

        #endregion

        #region Header Format Compliance

        [TestMethod]
        public void Header_KeysSortedAlphabetically()
        {
            var arr = np.arange(6).reshape(2, 3);
            using var ms = new MemoryStream();
            np.save(ms, arr);
            var bytes = ms.ToArray();

            int headerEnd = Array.IndexOf(bytes, (byte)'\n', 10);
            string header = System.Text.Encoding.ASCII.GetString(bytes, 10, headerEnd - 10);

            int descrPos = header.IndexOf("'descr'");
            int fortranPos = header.IndexOf("'fortran_order'");
            int shapePos = header.IndexOf("'shape'");

            Assert.IsTrue(descrPos < fortranPos);
            Assert.IsTrue(fortranPos < shapePos);
        }

        [TestMethod]
        public void Header_HasTrailingComma()
        {
            var arr = np.arange(6).reshape(2, 3);
            using var ms = new MemoryStream();
            np.save(ms, arr);
            var bytes = ms.ToArray();

            int headerEnd = Array.IndexOf(bytes, (byte)'\n', 10);
            string header = System.Text.Encoding.ASCII.GetString(bytes, 10, headerEnd - 10).Trim();

            Assert.IsTrue(header.EndsWith(", }"));
        }

        [TestMethod]
        public void Header_OneElementTupleHasTrailingComma()
        {
            var arr = np.arange(5); // 1D array
            using var ms = new MemoryStream();
            np.save(ms, arr);
            var bytes = ms.ToArray();

            int headerEnd = Array.IndexOf(bytes, (byte)'\n', 10);
            string header = System.Text.Encoding.ASCII.GetString(bytes, 10, headerEnd - 10);

            Assert.IsTrue(header.Contains("(5,)"), $"1D shape should be (5,) not (5): {header}");
        }

        [TestMethod]
        public void Header_DataStartsAt64ByteBoundary()
        {
            var arr = np.arange(10);
            using var ms = new MemoryStream();
            np.save(ms, arr);
            var bytes = ms.ToArray();

            int headerEnd = Array.LastIndexOf(bytes, (byte)'\n', bytes.Length - (int)(arr.size * arr.dtypesize) - 1);
            int dataStart = headerEnd + 1;

            Assert.AreEqual(0, dataStart % 64, $"Data starts at byte {dataStart}, not 64-byte aligned");
        }

        #endregion

        #region True Scalars

        [TestMethod]
        public void Save_Load_TrueScalar()
        {
            var scalar = new NDArray(NPTypeCode.Int32, Shape.NewScalar());
            scalar.SetAtIndex(42, 0);

            Assert.AreEqual(0, scalar.ndim);
            Assert.AreEqual(0, scalar.shape.Length);
            Assert.IsTrue(scalar.Shape.IsScalar);

            using var ms = new MemoryStream();
            np.save(ms, scalar);

            // Verify header has empty shape ()
            var bytes = ms.ToArray();
            int headerEnd = Array.IndexOf(bytes, (byte)'\n', 10);
            string header = System.Text.Encoding.ASCII.GetString(bytes, 10, headerEnd - 10);
            Assert.IsTrue(header.Contains("'shape': ()"), $"Scalar shape should be (): {header}");

            // Verify round-trip
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.AreEqual(0, loaded.ndim);
            Assert.IsTrue(loaded.Shape.IsScalar);
            Assert.AreEqual(42, loaded.GetInt32(0));
        }

        #endregion

        #region Multi-dimensional Shapes

        [DataTestMethod]
        [DataRow(new long[] { 1 })]
        [DataRow(new long[] { 1, 1, 1 })]
        [DataRow(new long[] { 10 })]
        [DataRow(new long[] { 2, 3, 4 })]
        [DataRow(new long[] { 2, 3, 4, 5 })]
        public void Save_Load_VariousShapes(long[] shape)
        {
            long size = shape.Aggregate(1L, (a, b) => a * b);
            var arr = np.arange(size).reshape(shape);

            using var ms = new MemoryStream();
            np.save(ms, arr);
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.IsTrue(shape.SequenceEqual(loaded.shape));
            Assert.IsTrue(np.array_equal(arr, loaded));
        }

        #endregion

        #region Empty Arrays

        [TestMethod]
        public void Save_Load_Empty1D()
        {
            var empty = np.array(Array.Empty<double>());

            using var ms = new MemoryStream();
            np.save(ms, empty);
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.AreEqual(0, loaded.size);
        }

        [TestMethod]
        public void Save_Load_EmptyMultiDimensional()
        {
            var empty = np.zeros(new long[] { 0, 3, 4 }, NPTypeCode.Double);

            using var ms = new MemoryStream();
            np.save(ms, empty);
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.IsTrue(new long[] { 0, 3, 4 }.SequenceEqual(loaded.shape));
            Assert.AreEqual(0, loaded.size);
        }

        #endregion

        #region Multiple Arrays in Stream

        [TestMethod]
        public void Save_Load_MultipleArraysInOneStream()
        {
            var arr1 = np.array(new long[] { 1, 2, 3 });
            var arr2 = np.array(new long[] { 4, 5, 6, 7 });
            var arr3 = np.array(new long[] { 8, 9 });

            using var ms = new MemoryStream();
            np.save(ms, arr1);
            np.save(ms, arr2);
            np.save(ms, arr3);

            ms.Position = 0;
            var loaded1 = np.load_npy(ms);
            var loaded2 = np.load_npy(ms);
            var loaded3 = np.load_npy(ms);

            Assert.IsTrue(np.array_equal(arr1, loaded1));
            Assert.IsTrue(np.array_equal(arr2, loaded2));
            Assert.IsTrue(np.array_equal(arr3, loaded3));
        }

        #endregion

        #region Error Handling

        [TestMethod]
        public void Load_InvalidMagic_ThrowsFormatException()
        {
            using var ms = new MemoryStream(new byte[] { 0, 1, 2, 3, 4, 5, 6, 7 });
            Assert.ThrowsException<FormatException>(() => np.load(ms));
        }

        /// <summary>
        ///     A file that starts correctly but stops short is a FORMAT error, not an EOF.
        /// </summary>
        /// <remarks>
        ///     NumPy draws this line deliberately and we follow it: a truncated-but-present file raises
        ///     ValueError ("EOF: reading magic string, expected 8 bytes got 6" for these exact bytes),
        ///     while EOFError is reserved for "No data left in file" — a wholly empty read at the
        ///     dispatch level. Mapping both onto EndOfStreamException, as this test used to expect,
        ///     throws that distinction away: "this file is corrupt" and "there is nothing here" are
        ///     different answers, and only the second is a normal end-of-input.
        /// </remarks>
        [TestMethod]
        public void Load_TruncatedFile_ThrowsFormatException()
        {
            using var ms = new MemoryStream(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' });

            var e = Assert.ThrowsException<FormatException>(() => np.load(ms));
            StringAssert.Contains(e.Message, "EOF: reading magic string, expected 8 bytes got 6");
        }

        /// <summary>An EMPTY file is the one case that really is an end-of-stream.</summary>
        [TestMethod]
        public void Load_EmptyFile_ThrowsEndOfStreamException()
        {
            using var ms = new MemoryStream(Array.Empty<byte>());

            var e = Assert.ThrowsException<EndOfStreamException>(() => np.load(ms));
            StringAssert.Contains(e.Message, "No data left in file");
        }

        [TestMethod]
        public void Save_DecimalType_ThrowsNotSupportedException()
        {
            var arr = np.array(new decimal[] { 1.5m, 2.5m });
            using var ms = new MemoryStream();
            Assert.ThrowsException<NotSupportedException>(() => np.save(ms, arr));
        }

        #endregion

        #region All Supported dtypes

        [DataTestMethod]
        [DataRow(NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.Byte)]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double)]
        public void Save_Load_AllSupportedDtypes(NPTypeCode typeCode)
        {
            var arr = np.zeros(new long[] { 3 }, typeCode);

            using var ms = new MemoryStream();
            np.save(ms, arr);
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.AreEqual(typeCode, loaded.typecode);
            Assert.AreEqual(arr.size, loaded.size);
        }

        #endregion
    }
}
