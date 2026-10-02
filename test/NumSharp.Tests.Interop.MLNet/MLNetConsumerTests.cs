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
    ///     Consumer-driven coverage: ML.NET itself — its own readers, data operations, transforms and a trainer — reads the
    ///     NumSharp views, through cursor patterns this bridge's own tests never use (POCO materialization, caching, row
    ///     skipping and taking, shuffling, train/test splitting, schema projection). Each result is compared either with
    ///     NumSharp's own elements, or — where ML.NET computes something — with ML.NET's own result over a C-contiguous copy
    ///     of the same data (a strided view must be indistinguishable from its contiguous copy to every ML.NET consumer).
    /// </summary>
    [TestClass]
    public class MLNetConsumerTests : MLNetTestBase
    {
        /// <summary>A POCO row with one float vector column, for ML.NET's own <c>CreateEnumerable</c> reader.</summary>
        private sealed class VectorRow
        {
            /// <summary>The vector column (named to match the views below).</summary>
            [VectorType]
            public float[] F { get; set; }
        }

        /// <summary>A POCO row with two scalar columns.</summary>
        private sealed class ScalarRow
        {
            /// <summary>The first column.</summary>
            public double a { get; set; }

            /// <summary>The second column.</summary>
            public double b { get; set; }
        }

        /// <summary>The strided float32 layouts the consumers read: every way a (rows, features) view can be laid out.</summary>
        /// <returns>The layouts, by name; each a fresh view.</returns>
        private static IEnumerable<(string Name, NDArray View)> FloatViews()
        {
            yield return ("contiguous", Arange(NPTypeCode.Single, 6, 4));
            yield return ("transposed", Arange(NPTypeCode.Single, 4, 6).T);
            yield return ("reversed rows", Arange(NPTypeCode.Single, 6, 4)["::-1"]);
            yield return ("reversed columns", Arange(NPTypeCode.Single, 6, 4)[":, ::-1"]);
            yield return ("stepped", Arange(NPTypeCode.Single, 12, 8)["::2, 1::2"]);
            yield return ("broadcast row", np.broadcast_to(Arange(NPTypeCode.Single, 4), new Shape(6, 4)));
            yield return ("F-contiguous", np.asfortranarray(Arange(NPTypeCode.Single, 6, 4)));
            yield return ("Half converted", Arange(NPTypeCode.Half, 6, 4).T["::-1"].T);
        }

        /// <summary>
        ///     ML.NET's own POCO reader (<c>CreateEnumerable</c>) reads every row of every layout as NumSharp's rows, for a
        ///     vector column and for scalar columns, with a reused row object and without.
        /// </summary>
        [TestMethod]
        public void CreateEnumerable_EveryLayout_ReadsNumSharpsRows()
        {
            foreach (var (name, view) in FloatViews())
            {
                using NDArray single = view.typecode == NPTypeCode.Single ? null : view.astype(NPTypeCode.Single);
                float[][] want = Rows(single ?? view);
                using NDArrayDataView dv = view.AsDataView("F");
                foreach (bool reuse in new[] { false, true })
                {
                    List<float[]> got = Ml.Data.CreateEnumerable<VectorRow>(dv, reuseRowObject: reuse).Select(r => r.F.ToArray()).ToList();
                    AssertRows(want, got, $"{name}, reuse {reuse}");
                }
            }

            using NDArray m = Arange(NPTypeCode.Double, 2, 5).T;   // (5, 2): two scalar columns
            using NDArrayDataView scalars = m.AsDataView(new[] { "a", "b" });
            ScalarRow[] rows = Ml.Data.CreateEnumerable<ScalarRow>(scalars, reuseRowObject: false).ToArray();
            Assert.AreEqual(5, rows.Length);
            for (int r = 0; r < 5; r++)
            {
                Assert.AreEqual(m.GetDouble(r, 0), rows[r].a, $"row {r} column a");
                Assert.AreEqual(m.GetDouble(r, 1), rows[r].b, $"row {r} column b");
            }
        }

        /// <summary>
        ///     ML.NET's data operations over every layout: caching (ML.NET reads the whole view, possibly with a cursor set),
        ///     taking and skipping rows, shuffling (a permutation of the rows) and train/test splitting (a partition of the
        ///     rows) all see exactly NumSharp's rows.
        /// </summary>
        [TestMethod]
        public void DataOperations_EveryLayout_SeeNumSharpsRows()
        {
            foreach (var (name, view) in FloatViews())
            {
                using NDArray single = view.typecode == NPTypeCode.Single ? null : view.astype(NPTypeCode.Single);
                float[][] want = Rows(single ?? view);
                using NDArrayDataView dv = view.AsDataView("F");

                AssertRows(want, Oracle.VectorRows<float>(Ml.Data.Cache(dv), "F"), $"{name}: Cache");
                AssertRows(want.Take(2).ToArray(), Oracle.VectorRows<float>(Ml.Data.TakeRows(dv, 2), "F"), $"{name}: TakeRows(2)");
                AssertRows(want.Skip(2).ToArray(), Oracle.VectorRows<float>(Ml.Data.SkipRows(dv, 2), "F"), $"{name}: SkipRows(2)");

                List<float[]> shuffled = Oracle.VectorRows<float>(Ml.Data.ShuffleRows(dv, seed: 7), "F");
                AssertSameRowMultiset(want, shuffled, $"{name}: ShuffleRows");

                DataOperationsCatalog.TrainTestData split = Ml.Data.TrainTestSplit(dv, testFraction: 0.5, seed: 3);
                List<float[]> union = Oracle.VectorRows<float>(split.TrainSet, "F").Concat(Oracle.VectorRows<float>(split.TestSet, "F")).ToList();
                AssertSameRowMultiset(want, union, $"{name}: TrainTestSplit train + test");
            }
        }

        /// <summary>
        ///     ML.NET's transforms compute the same result over a strided view as over its C-contiguous copy: min-max
        ///     normalization, a concatenation of scalar columns read from a transposed view, and a copy, compared slot for
        ///     slot (bit-identical: the same arithmetic over the same values).
        /// </summary>
        [TestMethod]
        public void Transforms_OverStridedViews_EqualTheirContiguousCopies()
        {
            foreach (var (name, view) in FloatViews())
            {
                using NDArray copy = np.ascontiguousarray(view);
                using NDArrayDataView strided = view.AsDataView("F");
                using NDArrayDataView dense = copy.AsDataView("F");
                var normalize = Ml.Transforms.NormalizeMinMax("N", "F");
                AssertRows(
                    Oracle.VectorRows<float>(normalize.Fit(dense).Transform(dense), "N").ToArray(),
                    Oracle.VectorRows<float>(normalize.Fit(strided).Transform(strided), "N"),
                    $"{name}: NormalizeMinMax");
                var copyColumn = Ml.Transforms.CopyColumns("G", "F");
                AssertRows(Rows(view.typecode == NPTypeCode.Single ? view : copy.astype(NPTypeCode.Single)),
                    Oracle.VectorRows<float>(copyColumn.Fit(strided).Transform(strided), "G"), $"{name}: CopyColumns");
            }

            using NDArray m = Arange(NPTypeCode.Single, 3, 5).T;   // (5, 3)
            using NDArrayDataView columns = m.AsDataView(new[] { "x", "y", "z" });
            IDataView concatenated = Ml.Transforms.Concatenate("V", "x", "y", "z").Fit(columns).Transform(columns);
            AssertRows(Rows(m), Oracle.VectorRows<float>(concatenated, "V"), "Concatenate over a transposed view");
        }

        /// <summary>
        ///     A trainer over a strided view learns exactly what it learns over the contiguous copy: SDCA regression, single
        ///     threaded with a fixed seed, gives bit-identical scores. A cursor that skipped, repeated or misread a row would
        ///     train a different model.
        /// </summary>
        [TestMethod]
        public void Trainer_OverAStridedView_LearnsWhatItLearnsFromTheCopy()
        {
            using NDArray features = Arange(NPTypeCode.Single, 8, 40)[":, ::2"].T;   // (20, 8) strided, transposed
            using NDArray labels = np.sum(features, axis: 1).astype(NPTypeCode.Single);
            using NDArray copy = np.ascontiguousarray(features);

            float[] Train(NDArray x)
            {
                using NDArrayDataView fv = x.AsDataView("Features");
                using NDArrayDataView lv = labels.AsDataView(new[] { "Label" });
                IDataView data = Ml.Data.LoadFromEnumerable(
                    Ml.Data.CreateEnumerable<FeatureRow>(fv, reuseRowObject: false)
                        .Zip(Ml.Data.CreateEnumerable<LabelRow>(lv, reuseRowObject: false), (f, l) => new TrainRow { Features = f.Features, Label = l.Label })
                        .ToList());
                var ml = new MLContext(seed: 11);
                var trainer = ml.Regression.Trainers.Sdca(new SdcaRegressionTrainer.Options { NumberOfThreads = 1, MaximumNumberOfIterations = 5 });
                ITransformer model = trainer.Fit(data);
                return Oracle.ScalarRows<float>(model.Transform(data), new[] { "Score" }).Select(r => r[0]).ToArray();
            }

            CollectionAssert.AreEqual(Train(copy), Train(features), "scores from the strided view differ from the copy's");
        }

        /// <summary>A POCO row with a fixed-size feature vector.</summary>
        private sealed class FeatureRow
        {
            /// <summary>The features.</summary>
            [VectorType(8)]
            public float[] Features { get; set; }
        }

        /// <summary>A POCO row with a label.</summary>
        private sealed class LabelRow
        {
            /// <summary>The label.</summary>
            public float Label { get; set; }
        }

        /// <summary>A POCO training row.</summary>
        private sealed class TrainRow
        {
            /// <summary>The features.</summary>
            [VectorType(8)]
            public float[] Features { get; set; }

            /// <summary>The label.</summary>
            public float Label { get; set; }
        }

        /// <summary>A 1-D or 2-D float array's rows (a 1-D array is one row), read through NumSharp's own indexer.</summary>
        /// <param name="nd">The array.</param>
        /// <returns>One array per row.</returns>
        private static float[][] Rows(NDArray nd)
        {
            float[] flat = Oracle.Elements<float>(nd);
            int width = nd.ndim == 2 ? (int)nd.shape[1] : (int)nd.size;
            int rows = nd.ndim == 2 ? (int)nd.shape[0] : 1;
            return Enumerable.Range(0, rows).Select(r => flat.Skip(r * width).Take(width).ToArray()).ToArray();
        }

        /// <summary>Asserts two row lists are equal row by row, element by element.</summary>
        /// <param name="want">The expected rows.</param>
        /// <param name="got">The rows read.</param>
        /// <param name="what">What was read, for the message.</param>
        private static void AssertRows(IReadOnlyList<float[]> want, IReadOnlyList<float[]> got, string what)
        {
            Assert.AreEqual(want.Count, got.Count, $"{what}: row count");
            for (int r = 0; r < want.Count; r++)
                CollectionAssert.AreEqual(want[r], got[r], $"{what}: row {r}");
        }

        /// <summary>Asserts two row lists hold the same rows, in any order (a shuffle or a split).</summary>
        /// <param name="want">The expected rows.</param>
        /// <param name="got">The rows read.</param>
        /// <param name="what">What was read, for the message.</param>
        private static void AssertSameRowMultiset(IReadOnlyList<float[]> want, IReadOnlyList<float[]> got, string what)
        {
            static string Key(float[] row) => string.Join(",", row.Select(v => BitConverter.SingleToInt32Bits(v)));
            CollectionAssert.AreEquivalent(want.Select(Key).ToList(), got.Select(Key).ToList(), what);
        }
    }
}
