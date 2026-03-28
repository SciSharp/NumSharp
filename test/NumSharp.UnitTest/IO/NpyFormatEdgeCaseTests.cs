using System;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.IO;
using TUnit.Core;

namespace NumSharp.UnitTest.IO
{
    /// <summary>
    /// Battle-tested edge cases for .npy/.npz format handling.
    /// </summary>
    public class NpyFormatEdgeCaseTests
    {
        #region Sliced and Non-Contiguous Arrays

        [Test]
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

            Assert.IsTrue(new int[] { 2, 3 }.SequenceEqual(loaded.shape));
            Assert.AreEqual(6, sliced.GetInt32(0, 0));
            Assert.AreEqual(6, loaded.GetInt32(0, 0));
            Assert.IsTrue(np.array_equal(sliced, loaded));
        }

        [Test]
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

        [Test]
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

        [Test]
        public void Save_Load_TransposedArray()
        {
            var arr = np.arange(12).reshape(3, 4);
            var transposed = arr.T;

            Assert.IsFalse(transposed.Shape.IsContiguous);

            using var ms = new MemoryStream();
            np.save(ms, transposed);
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.IsTrue(new int[] { 4, 3 }.SequenceEqual(loaded.shape));
            Assert.IsTrue(np.array_equal(transposed, loaded));
        }

        [Test]
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

        [Test]
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

        [Test]
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

        [Test]
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

        [Test]
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

        [Test]
        public void Header_DataStartsAt64ByteBoundary()
        {
            var arr = np.arange(10);
            using var ms = new MemoryStream();
            np.save(ms, arr);
            var bytes = ms.ToArray();

            int headerEnd = Array.LastIndexOf(bytes, (byte)'\n', bytes.Length - arr.size * arr.dtypesize - 1);
            int dataStart = headerEnd + 1;

            Assert.AreEqual(0, dataStart % 64, $"Data starts at byte {dataStart}, not 64-byte aligned");
        }

        #endregion

        #region True Scalars

        [Test]
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

        [Test]
        [Arguments(new int[] { 1 })]
        [Arguments(new int[] { 1, 1, 1 })]
        [Arguments(new int[] { 10 })]
        [Arguments(new int[] { 2, 3, 4 })]
        [Arguments(new int[] { 2, 3, 4, 5 })]
        public void Save_Load_VariousShapes(int[] shape)
        {
            int size = shape.Aggregate(1, (a, b) => a * b);
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

        [Test]
        public void Save_Load_Empty1D()
        {
            var empty = np.array(Array.Empty<double>());

            using var ms = new MemoryStream();
            np.save(ms, empty);
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.AreEqual(0, loaded.size);
        }

        [Test]
        public void Save_Load_EmptyMultiDimensional()
        {
            var empty = np.zeros(new int[] { 0, 3, 4 }, NPTypeCode.Double);

            using var ms = new MemoryStream();
            np.save(ms, empty);
            ms.Position = 0;
            var loaded = np.load_npy(ms);

            Assert.IsTrue(new int[] { 0, 3, 4 }.SequenceEqual(loaded.shape));
            Assert.AreEqual(0, loaded.size);
        }

        #endregion

        #region Multiple Arrays in Stream

        [Test]
        public void Save_Load_MultipleArraysInOneStream()
        {
            var arr1 = np.array(new int[] { 1, 2, 3 });
            var arr2 = np.array(new int[] { 4, 5, 6, 7 });
            var arr3 = np.array(new int[] { 8, 9 });

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

        [Test]
        public void Load_InvalidMagic_ThrowsFormatException()
        {
            using var ms = new MemoryStream(new byte[] { 0, 1, 2, 3, 4, 5, 6, 7 });
            Assert.ThrowsException<FormatException>(() => np.load(ms));
        }

        [Test]
        public void Load_TruncatedFile_ThrowsEndOfStreamException()
        {
            using var ms = new MemoryStream(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' });
            Assert.ThrowsException<EndOfStreamException>(() => np.load(ms));
        }

        [Test]
        public void Save_DecimalType_ThrowsNotSupportedException()
        {
            var arr = np.array(new decimal[] { 1.5m, 2.5m });
            using var ms = new MemoryStream();
            Assert.ThrowsException<NotSupportedException>(() => np.save(ms, arr));
        }

        #endregion

        #region All Supported dtypes

        [Test]
        [Arguments(NPTypeCode.Boolean)]
        [Arguments(NPTypeCode.Byte)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.UInt16)]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.UInt32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.UInt64)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        public void Save_Load_AllSupportedDtypes(NPTypeCode typeCode)
        {
            var arr = np.zeros(new int[] { 3 }, typeCode);

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
