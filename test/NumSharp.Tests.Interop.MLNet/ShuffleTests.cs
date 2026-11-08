using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     The view shuffles (<c>CanShuffle</c>): a cursor opened with a <see cref="Random"/> walks every row once in a
    ///     random permutation drawn from it, ids stay the source rows, a seed reproduces its order, and the cursor contract
    ///     holds unchanged. The reason it exists: SDCA uses its source's shuffle, and without one it fell back to ML.NET's
    ///     pool shuffle, which cannot mix rows stored in class order (0.50 accuracy on class-sorted rows).
    /// </summary>
    [TestClass]
    public class ShuffleTests : MLNetTestBase
    {
        /// <summary>
        ///     A shuffled cursor visits each row exactly once, not in order, each row's values read with its id; an
        ///     unshuffled cursor (no <see cref="Random"/>) stays sequential.
        /// </summary>
        [TestMethod]
        public void ShuffledCursor_VisitsEveryRowOnce_WithItsOwnValues_AndIdsAreSourceRows()
        {
            using NDArray x = Arange(NPTypeCode.Single, 50, 3);           // row r holds 3r, 3r+1, 3r+2
            using NDArrayDataView view = x.AsDataView("F");
            Assert.IsTrue(view.CanShuffle);

            var ids = new List<long>();
            using (DataViewRowCursor cursor = view.GetRowCursor(view.Schema, new Random(7)))
            {
                ValueGetter<VBuffer<float>> get = cursor.GetGetter<VBuffer<float>>(view.Schema["F"]);
                ValueGetter<DataViewRowId> id = cursor.GetIdGetter();
                VBuffer<float> row = default;
                DataViewRowId rowId = default;
                long expectedPosition = 0;
                while (cursor.MoveNext())
                {
                    Assert.AreEqual(expectedPosition++, cursor.Position, "Position still counts 0, 1, 2, ...");
                    get(ref row);
                    id(ref rowId);
                    long r = (long)rowId.Low;
                    CollectionAssert.AreEqual(new float[] { 3 * r, 3 * r + 1, 3 * r + 2 }, row.DenseValues().ToArray(), $"row {r}");
                    ids.Add(r);
                }
                Assert.AreEqual(-1, cursor.Position, "off the rows after the last, as unshuffled");
            }
            CollectionAssert.AreEquivalent(Enumerable.Range(0, 50).Select(i => (long)i).ToArray(), ids, "every row exactly once");
            CollectionAssert.AreNotEqual(Enumerable.Range(0, 50).Select(i => (long)i).ToArray(), ids, "a shuffled cursor is not in order");

            using (DataViewRowCursor plain = view.GetRowCursor(view.Schema))
            {
                ValueGetter<DataViewRowId> id = plain.GetIdGetter();
                DataViewRowId rowId = default;
                for (long expected = 0; plain.MoveNext(); expected++)
                {
                    id(ref rowId);
                    Assert.AreEqual((ulong)expected, rowId.Low, "no Random: the rows in order");
                }
            }
        }

        /// <summary>The order is a function of the <see cref="Random"/>: a seed gives the same order again, another seed another.</summary>
        [TestMethod]
        public void TheOrder_IsReproducibleFromTheSeed()
        {
            using NDArray x = Arange(NPTypeCode.Int32, 200);
            using NDArrayDataView view = x.AsDataView(new[] { "v" });
            int[] a = Order(view, 11), b = Order(view, 11), c = Order(view, 12);
            CollectionAssert.AreEqual(a, b, "the same seed, the same order");
            CollectionAssert.AreNotEqual(a, c, "another seed, another order");
        }

        /// <summary>No row and one row shuffle to themselves; a getter off a row still throws ML.NET's error.</summary>
        [TestMethod]
        public void EmptyAndSingleRowViews_Shuffle_AndOffRowGettersStillThrow()
        {
            using NDArray none = np.zeros(new Shape(0, 2), np.float32);
            using (NDArrayDataView view = none.AsDataView("F"))
            using (DataViewRowCursor cursor = view.GetRowCursor(view.Schema, new Random(1)))
                Assert.IsFalse(cursor.MoveNext());

            using NDArray one = Arange(NPTypeCode.Single, 1, 2);
            using (NDArrayDataView view = one.AsDataView("F"))
            using (DataViewRowCursor cursor = view.GetRowCursor(view.Schema, new Random(1)))
            {
                ValueGetter<VBuffer<float>> get = cursor.GetGetter<VBuffer<float>>(view.Schema["F"]);
                VBuffer<float> row = default;
                Assert.ThrowsException<InvalidOperationException>(() => get(ref row), "before the first row");
                Assert.IsTrue(cursor.MoveNext());
                get(ref row);
                CollectionAssert.AreEqual(new float[] { 0, 1 }, row.DenseValues().ToArray());
                Assert.IsFalse(cursor.MoveNext());
                Assert.ThrowsException<InvalidOperationException>(() => get(ref row), "after the last row");
            }
        }

        /// <summary>
        ///     The case it exists for: SDCA on 20,000 rows stored in CLASS ORDER (all of one class, then all of the other)
        ///     learns from the view's own shuffle. Single-threaded SDCA with a fixed seed, so the run is deterministic; a
        ///     non-shuffling source left it at chance (0.50).
        /// </summary>
        [TestMethod]
        public void Sdca_OnClassOrderedRows_LearnsFromTheViewsShuffle()
        {
            var rng = np.random.RandomState(5);
            const int n = 20000, d = 10;
            using NDArray c0 = (rng.randn(n / 2, d) * 1.5).astype(np.float32);
            using NDArray c1 = (rng.randn(n / 2, d) * 1.5 + 0.6).astype(np.float32);
            using NDArray x = np.vstack(new[] { c0, c1 });
            using NDArray y = np.concatenate(new[] { np.zeros(new Shape(n / 2), np.float32), np.ones(new Shape(n / 2), np.float32) });
            using NDArray table = np.column_stack(new[] { x, y });
            string[] names = Enumerable.Range(0, d).Select(i => $"f{i}").Append("Label").ToArray();

            var ml = new MLContext(seed: 1);
            using NDArrayDataView view = table.AsDataView(names);
            IDataView labelled = ml.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Boolean).Fit(view).Transform(view);
            IDataView data = ml.Transforms.Concatenate("Features", names.Take(d).ToArray()).Fit(labelled).Transform(labelled);
            Assert.IsTrue(data.CanShuffle, "the transforms pass the view's shuffle through");
            var sdca = ml.BinaryClassification.Trainers.SdcaLogisticRegression(new SdcaLogisticRegressionBinaryTrainer.Options
            {
                MaximumNumberOfIterations = 10, NumberOfThreads = 1, Shuffle = true,
            });
            ITransformer model = sdca.Fit(data);
            var metrics = ml.BinaryClassification.Evaluate(model.Transform(data));
            Assert.IsTrue(metrics.Accuracy > 0.65, $"SDCA learned the classes from the shuffled rows (accuracy {metrics.Accuracy:0.000})");
        }

        /// <summary>The rows a shuffled cursor visits, in its order.</summary>
        /// <param name="view">A view with an int32 scalar column <c>v</c> whose value is its row.</param>
        /// <param name="seed">The seed of the cursor's <see cref="Random"/>.</param>
        /// <returns>The values read, in the cursor's order.</returns>
        private static int[] Order(NDArrayDataView view, int seed)
        {
            var seen = new List<int>();
            using DataViewRowCursor cursor = view.GetRowCursor(view.Schema, new Random(seed));
            ValueGetter<int> get = cursor.GetGetter<int>(view.Schema["v"]);
            int v = 0;
            while (cursor.MoveNext())
            {
                get(ref v);
                seen.Add(v);
            }
            return seen.ToArray();
        }
    }
}
