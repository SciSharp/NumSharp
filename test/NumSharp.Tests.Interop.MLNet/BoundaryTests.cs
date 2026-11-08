using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     Boundary values: the sizes where an <c>int</c> byte count or a managed array would overflow, the row counts that
    ///     cross the import's chunk boundaries, and the degenerate shapes (no rows, no columns, zero-width vectors). Sizes
    ///     past 2 GB are reached through broadcast views (no memory) on export; an import of that size really allocates its
    ///     result, so those tests run only where the machine has the memory and report Inconclusive elsewhere.
    /// </summary>
    [TestClass]
    public class BoundaryTests : MLNetTestBase
    {
        /// <summary>The physical memory a test that allocates multi-gigabyte results needs before it runs.</summary>
        private const long BigTestMemory = 12L << 30;

        /// <summary>
        ///     A sparse <see cref="VBuffer{T}"/> whose dense form is exactly 2 GB (2^28 doubles — the shape of a hashed sparse
        ///     feature vector) densifies correctly through both import verbs. The densify used to clear
        ///     <c>length * sizeof(T)</c> bytes computed as an <c>int</c>, which overflows at 2 GB, and threw
        ///     ArgumentOutOfRangeException instead.
        /// </summary>
        [TestMethod]
        public void SparseVBuffer_OfTwoGigabytesDense_Densifies()
        {
            RequireMemory(BigTestMemory);
            int length = 1 << 28;
            var sparse = new VBuffer<double>(length, 3, new[] { 1.5, 2.5, 3.5 }, new[] { 0, 1000, length - 1 });
            foreach (string verb in new[] { "ToNDArray", "AsNDArray" })
            {
                using NDArray nd = verb == "ToNDArray" ? sparse.ToNDArray() : sparse.AsNDArray();
                Assert.AreEqual((long)length, nd.size, verb);
                Assert.AreEqual(1.5, nd.GetDouble(0), verb);
                Assert.AreEqual(2.5, nd.GetDouble(1000), verb);
                Assert.AreEqual(3.5, nd.GetDouble(length - 1), verb);
                Assert.AreEqual(3L, (long)np.count_nonzero(nd), $"{verb}: every other element must be zero");
            }
        }

        /// <summary>
        ///     A DENSE <see cref="VBuffer{T}"/> past 2 GB of bytes (2^28 + 2 doubles) crosses both import verbs byte for
        ///     byte: the copy counts in elements (a byte count would overflow an <c>int</c> here, the class of bug the
        ///     sparse densify had and ONNX Runtime's <c>DenseTensor</c> copy had), and the zero-copy view addresses all of it.
        ///     The buffer is never written but at a few sentinels around both ends and the 2 GB byte boundary.
        /// </summary>
        [TestMethod]
        public unsafe void DenseVBuffer_PastTwoGigabytes_ImportsThroughBothVerbs()
        {
            RequireMemory(BigTestMemory);
            int length = (1 << 28) + 2;
            double[] values = GC.AllocateUninitializedArray<double>(length);
            foreach (int i in new[] { 0, 1, (1 << 28) - 1, 1 << 28, length / 2, length - 1 })
                values[i] = i * 0.5 + 0.25;
            var buffer = new VBuffer<double>(length, values);
            foreach (string verb in new[] { "ToNDArray", "AsNDArray" })
            {
                using NDArray nd = verb == "ToNDArray" ? buffer.ToNDArray() : buffer.AsNDArray();
                Assert.AreEqual((long)length, nd.size, verb);
                byte* a = (byte*)nd.Storage.Address + nd.Shape.Offset * sizeof(double);
                fixed (double* w = values)
                {
                    long bytes = (long)length * sizeof(double);
                    // Compared in 1 GB chunks: no span covers the whole buffer, which is the point of the test.
                    for (long off = 0; off < bytes; off += 1 << 30)
                    {
                        int chunk = (int)Math.Min(1 << 30, bytes - off);
                        Assert.IsTrue(new ReadOnlySpan<byte>(a + off, chunk).SequenceEqual(new ReadOnlySpan<byte>((byte*)w + off, chunk)),
                            $"{verb}: bytes differ in the chunk at {off}");
                    }
                }
            }
            GC.KeepAlive(values);
        }

        /// <summary>
        ///     A vector column of more than <see cref="Array.MaxLength"/> elements in total (2^20 rows × 2,048) imports into
        ///     one NDArray. The import used to stage the rows into a managed <c>T[rowCount * width]</c>, which failed with
        ///     OutOfMemoryException ("Array dimensions exceeded supported range") past <see cref="Array.MaxLength"/> although
        ///     the NDArray holds it. The source is a broadcast view, so only the result costs memory.
        /// </summary>
        [TestMethod]
        public void VectorColumn_PastTheManagedArrayLimit_Imports()
        {
            RequireMemory(BigTestMemory);
            using NDArray seven = NDArray.Scalar((byte)7);
            using NDArray wide = np.broadcast_to(seven, new Shape(1 << 20, 2048));
            Assert.IsTrue(wide.size > Array.MaxLength);
            using NDArrayDataView view = wide.AsDataView("F");
            using NDArray back = view.ToNDArray("F");
            AssertShape(back, 1 << 20, 2048);
            Assert.AreEqual((byte)7, back.GetByte(0, 0));
            Assert.AreEqual((byte)7, back.GetByte((1 << 20) - 1, 2047));
            Assert.AreEqual(back.size, (long)np.count_nonzero(back), "every element must have been written");
        }

        /// <summary>
        ///     A view that does not report its row count (an ML.NET row filter) is read into growing chunks and copied
        ///     together once: every row count around the chunk boundaries (the first chunk holds 4,096 rows, then 8,192,
        ///     then 16,384) comes back complete and in order, for a scalar column and for a vector column.
        /// </summary>
        [TestMethod]
        public void UnknownRowCount_EveryChunkBoundary_ImportsEveryRow()
        {
            foreach (long rows in new long[] { 0, 1, 4095, 4096, 4097, 12287, 12288, 12289, 40000 })
            {
                using NDArray values = np.arange((double)rows);
                using NDArray columns = np.stack(new[] { values, values * 2.0, values * 3.0 }, axis: 1);
                using NDArrayDataView source = columns.AsDataView(new[] { "a", "b", "c" });
                IDataView vectors = Ml.Transforms.Concatenate("V", "a", "b", "c").Fit(source).Transform(source);
                IDataView filtered = Ml.Data.FilterRowsByColumn(vectors, "a", lowerBound: double.MinValue, upperBound: double.MaxValue);
                Assert.IsNull(filtered.GetRowCount(), "the filter must not report a row count, or this does not test the chunked path");

                using NDArray scalar = filtered.ToNDArray("b");
                AssertShape(scalar, rows);
                using NDArray wantScalar = values * 2.0;
                Assert.IsTrue(np.array_equal(scalar, wantScalar), $"{rows} rows: scalar column");

                using NDArray vector = filtered.ToNDArray("V");
                AssertShape(vector, rows, 3);
                Assert.IsTrue(np.array_equal(vector, columns), $"{rows} rows: vector column");
            }
        }

        /// <summary>
        ///     The degenerate shapes import with the right shape: no rows (scalar and vector, the vector keeping its declared
        ///     width), zero-width vectors on every row, and a single row.
        /// </summary>
        [TestMethod]
        public void DegenerateShapes_ImportWithTheirShape()
        {
            using (NDArray none = np.zeros(new Shape(0, 3), NPTypeCode.Single))
            using (NDArrayDataView view = none.AsDataView("F"))
            using (NDArray back = view.ToNDArray("F"))
                AssertShape(back, 0, 3);
            using (NDArray none = np.zeros(new Shape(0), NPTypeCode.Int32))
            using (NDArrayDataView view = none.AsDataView(new[] { "x" }))
            using (NDArray back = view.ToNDArray("x"))
                AssertShape(back, 0);
            using (NDArray narrow = np.zeros(new Shape(4, 0), NPTypeCode.Double))
            using (NDArrayDataView view = narrow.AsDataView("F"))
            using (NDArray back = view.ToNDArray("F"))
                AssertShape(back, 4, 0);
            using (NDArray one = Arange(NPTypeCode.Int64, 1, 5))
            using (NDArrayDataView view = one.AsDataView("F"))
            using (NDArray back = view.ToNDArray("F"))
            {
                AssertShape(back, 1, 5);
                Assert.IsTrue(np.array_equal(back, one));
            }
        }

        /// <summary>
        ///     The export side's size limits sit exactly at ML.NET's buffer cap: one element past
        ///     <see cref="NDArrayMLNetInterop.MaxBufferLength"/> is refused for <c>ToVBuffer</c> and for a vector column,
        ///     while a scalar column of that many ROWS is fine (rows are read one at a time).
        /// </summary>
        [TestMethod]
        public void ExportLimits_SitAtMLNetsBufferCap()
        {
            using NDArray one = NDArray.Scalar(1f);
            long past = (long)NDArrayMLNetInterop.MaxBufferLength + 1;
            using NDArray longVector = np.broadcast_to(one, new Shape(past));
            Assert.ThrowsException<NotSupportedException>(() => longVector.ToVBuffer<float>());
            Assert.ThrowsException<NotSupportedException>(() => longVector.AsDataView("F"));
            using NDArrayDataView rows = longVector.AsDataView(new[] { "x" });
            Assert.AreEqual(past, rows.GetRowCount());
        }

        /// <summary>Skips the calling test (Inconclusive) unless the machine has at least <paramref name="bytes"/> of physical memory.</summary>
        /// <param name="bytes">The memory the test needs.</param>
        /// <exception cref="AssertInconclusiveException">The machine has less.</exception>
        private static void RequireMemory(long bytes)
        {
            long total = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            if (total < bytes)
                Assert.Inconclusive($"needs {bytes >> 30} GB of physical memory, the machine has {total >> 30} GB");
        }
    }
}
