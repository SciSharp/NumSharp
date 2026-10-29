using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     Metamorphic relations: two routes to the same answer must agree, with no oracle for the answer itself. Slicing rows
    ///     commutes with viewing; the vector-column view of a matrix and the scalar-column view of its transpose read the same
    ///     numbers transposed; a view, its snapshot and the view of its copy read the same; <c>ToVBuffer</c> equals the vector
    ///     rows concatenated; round trips are idempotent; and a zero-copy chain (VBuffer → NDArray → data view) reads, and
    ///     follows writes to, one buffer.
    /// </summary>
    [TestClass]
    public class MetamorphicTests : MLNetTestBase
    {
        /// <summary>The (rows, features) float32 layouts the relations run over.</summary>
        /// <returns>The layouts, by name; each a fresh view.</returns>
        private static IEnumerable<(string Name, NDArray View)> Matrices()
        {
            yield return ("contiguous", Arange(NPTypeCode.Single, 7, 5));
            yield return ("transposed", Arange(NPTypeCode.Single, 5, 7).T);
            yield return ("reversed both", Arange(NPTypeCode.Single, 7, 5)["::-1, ::-1"]);
            yield return ("stepped", Arange(NPTypeCode.Single, 14, 10)["::2, 1::2"]);
            yield return ("broadcast row", np.broadcast_to(Arange(NPTypeCode.Single, 5), new Shape(7, 5)));
            yield return ("F-contiguous", np.asfortranarray(Arange(NPTypeCode.Single, 7, 5)));
        }

        /// <summary>Rows <c>a..b</c> read through the view of <c>m</c> equal the rows read through the view of <c>m[a:b]</c>.</summary>
        [TestMethod]
        public void SlicingRowsCommutesWithViewing()
        {
            foreach (var (name, m) in Matrices())
            {
                using NDArrayDataView whole = m.AsDataView("F");
                List<float[]> all = Oracle.VectorRows<float>(whole, "F");
                foreach ((int a, int b) in new[] { (0, 7), (2, 5), (3, 3), (6, 7) })
                {
                    using NDArray part = m[$"{a}:{b}"];
                    using NDArrayDataView partView = part.AsDataView("F");
                    List<float[]> got = Oracle.VectorRows<float>(partView, "F");
                    Assert.AreEqual(b - a, got.Count, $"{name}[{a}:{b}]: row count");
                    for (int r = 0; r < got.Count; r++)
                        CollectionAssert.AreEqual(all[a + r], got[r], $"{name}[{a}:{b}]: row {r}");
                }
            }
        }

        /// <summary>
        ///     The vector-column view of <c>m</c> and the scalar-column view of <c>m.T</c> read the same numbers transposed:
        ///     row <c>r</c> element <c>j</c> of the first is column <c>r</c> of row <c>j</c> of the second.
        /// </summary>
        [TestMethod]
        public void VectorViewOfAMatrix_IsTheScalarViewOfItsTranspose_Transposed()
        {
            foreach (var (name, m) in Matrices())
            {
                using NDArrayDataView vectors = m.AsDataView("F");
                using NDArray mT = m.T;
                string[] names = Enumerable.Range(0, (int)m.shape[0]).Select(i => "r" + i).ToArray();
                using NDArrayDataView scalars = mT.AsDataView(names);
                List<float[]> rows = Oracle.VectorRows<float>(vectors, "F");
                List<float[]> columns = Oracle.ScalarRows<float>(scalars, names);
                for (int r = 0; r < rows.Count; r++)
                for (int j = 0; j < rows[r].Length; j++)
                    Assert.AreEqual(rows[r][j], columns[j][r], $"{name}: [{r},{j}]");
            }
        }

        /// <summary>A view, its snapshot (<c>ToDataView</c>) and the view of its contiguous copy read the same rows.</summary>
        [TestMethod]
        public void ViewSnapshotAndCopy_ReadTheSame()
        {
            foreach (var (name, m) in Matrices())
            {
                using NDArrayDataView view = m.AsDataView("F");
                using NDArrayDataView snapshot = m.ToDataView("F");
                using NDArray copy = np.ascontiguousarray(m);
                using NDArrayDataView copyView = copy.AsDataView("F");
                byte[] want = Oracle.Bytes(Oracle.VectorRows<float>(copyView, "F").SelectMany(r => r).ToArray());
                Assert.IsTrue(Oracle.SameBytes(Oracle.Bytes(Oracle.VectorRows<float>(view, "F").SelectMany(r => r).ToArray()), want), $"{name}: view");
                Assert.IsTrue(Oracle.SameBytes(Oracle.Bytes(Oracle.VectorRows<float>(snapshot, "F").SelectMany(r => r).ToArray()), want), $"{name}: snapshot");
            }
        }

        /// <summary><c>ToVBuffer(m)</c> equals the vector rows of <c>m</c>'s view concatenated (a C-order flatten).</summary>
        [TestMethod]
        public void ToVBuffer_IsTheRowsConcatenated()
        {
            foreach (var (name, m) in Matrices())
            {
                using NDArrayDataView view = m.AsDataView("F");
                float[] rows = Oracle.VectorRows<float>(view, "F").SelectMany(r => r).ToArray();
                VBuffer<float> flat = m.ToVBuffer<float>();
                CollectionAssert.AreEqual(rows, flat.DenseValues().ToArray(), name);
            }
        }

        /// <summary>Round trips through a data view are idempotent: crossing the result again changes nothing, three times over.</summary>
        [TestMethod]
        public void RoundTripsAreIdempotent()
        {
            foreach (var (name, m) in Matrices())
            {
                NDArray current;
                using (NDArrayDataView first = m.AsDataView("F"))
                    current = first.ToNDArray("F");
                byte[] want = Oracle.LogicalBytes(current);
                for (int trip = 0; trip < 3; trip++)
                {
                    NDArray next;
                    using (NDArrayDataView again = current.ToDataView("F"))
                        next = again.ToNDArray("F");
                    current.Dispose();
                    current = next;
                    Assert.IsTrue(Oracle.SameBytes(Oracle.LogicalBytes(current), want), $"{name}: round trip {trip + 2} changed the elements");
                }
                current.Dispose();
            }
        }

        /// <summary>
        ///     A zero-copy chain reads one buffer: a dense <see cref="VBuffer{T}"/> → <c>AsNDArray</c> (zero-copy) → a 2-D
        ///     reshape → <c>AsDataView</c> (zero-copy) → a cursor reads the buffer's values; a write into the VBuffer's own array
        ///     is then seen by a new cursor.
        /// </summary>
        [TestMethod]
        public void ZeroCopyChain_FromVBufferToCursor_ReadsOneBuffer()
        {
            float[] values = Enumerable.Range(0, 12).Select(i => (float)i).ToArray();
            var buffer = new VBuffer<float>(12, values);
            using NDArray flat = buffer.AsNDArray();
            using NDArray grid = flat.reshape(3, 4);
            using NDArrayDataView view = grid.AsDataView("F");
            CollectionAssert.AreEqual(values, Oracle.VectorRows<float>(view, "F").SelectMany(r => r).ToArray(), "the chain reads the buffer");
            values[6] = -1f;   // the VBuffer's own array, written directly
            Assert.AreEqual(-1f, Oracle.VectorRows<float>(view, "F")[1][2], "the chain does not follow the buffer");
        }
    }
}
