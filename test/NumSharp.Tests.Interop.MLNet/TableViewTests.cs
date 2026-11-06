using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     The dictionary form, one column per array (<c>AsDataView(IReadOnlyDictionary&lt;string, NDArray&gt;)</c>):
    ///     what ML.NET training needs - a <c>Features</c> vector and a <c>Label</c> of its own dtype in one view. The
    ///     single-array view could not express it (one array, one dtype): training needed <c>np.hstack(X, y)</c> as
    ///     scalar columns plus a conversion step, and a model of ML.NET's multiclass recipe could not even score a
    ///     features-only view.
    /// </summary>
    [TestClass]
    public class TableViewTests : MLNetTestBase
    {
        /// <summary>
        ///     A bool label next to float32 features trains a binary classifier directly (no hstack, no conversion step),
        ///     the trained model scores label-less rows (a plain <c>AsDataView("Features")</c>) exactly as it scores the
        ///     training view, the multiclass recipe (<c>MapValueToKey("Label")</c> in the chain) trains AND scores a view
        ///     that carries a label, and each column reads back with its own dtype and values.
        /// </summary>
        [TestMethod]
        public void FeaturesAndLabel_OfTheirOwnDtypes_TrainAndScore()
        {
            var rng = np.random.RandomState(3);
            using NDArray x = rng.rand(200, 4).astype(np.float32);
            using NDArray sums = np.sum(x, axis: 1);
            using NDArray y = sums > 2.0f;                                   // (200,) bool

            using (NDArrayDataView view = new Dictionary<string, NDArray> { ["Features"] = x, ["Label"] = y }.AsDataView())
            {
                Assert.AreEqual("Vector<Single, 4>", view.Schema["Features"].Type.ToString());
                Assert.AreEqual(BooleanDataViewType.Instance, view.Schema["Label"].Type);
                ITransformer model = Ml.BinaryClassification.Trainers.LbfgsLogisticRegression().Fit(view);
                using NDArray predicted = model.Transform(view).ToNDArray("PredictedLabel");
                using NDArray hits = predicted == y;
                Assert.IsTrue((double)np.mean(hits) > 0.9, "a separable label is learned");

                // The trainer reads the label at fit time only: the docs' "score new rows without a label column".
                using NDArrayDataView unlabelled = x.AsDataView("Features");
                using NDArray scoredWithoutLabel = model.Transform(unlabelled).ToNDArray("PredictedLabel");
                Assert.IsTrue(np.array_equal(scoredWithoutLabel, predicted));

                using NDArray features = view.ToNDArray("Features");
                using NDArray label = view.ToNDArray("Label");
                Assert.IsTrue(np.array_equal(features, x));
                Assert.IsTrue(np.array_equal(label, y));
            }

            using NDArray classes = np.floor_divide(sums * 1.5f, 2.0f).astype(np.float32);   // 0, 1, 2
            using (NDArrayDataView view = new Dictionary<string, NDArray> { ["Features"] = x, ["Label"] = classes }.AsDataView())
            {
                ITransformer recipe = Ml.Transforms.Conversion.MapValueToKey("Label")
                    .Append(Ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(maximumNumberOfIterations: 20))
                    .Fit(view);
                using NDArray score = recipe.Transform(view).ToNDArray("Score");
                AssertShape(score, 200, 3);
            }
        }

        /// <summary>
        ///     Every array is read through its own strides (a transposed, a stepped and a reversed array, each column a
        ///     different layout and dtype), Half is converted to Single in a temporary the view owns, and the snapshot form
        ///     has no coupling to its sources (they are disposed before it is read).
        /// </summary>
        [TestMethod]
        public void EveryColumn_ReadsItsOwnLayoutAndDtype_AndTheSnapshotIsIndependent()
        {
            using NDArray wide = Arange(NPTypeCode.Double, 3, 6);
            using NDArray transposed = wide.T;                              // (6, 3) float64, F-ordered
            using NDArray ints = Arange(NPTypeCode.Int32, 12);
            using NDArray stepped = ints["::2"];                            // (6,) int32, stride 2
            using NDArray halves = Arange(NPTypeCode.Half, 6, 2);
            using NDArray reversed = halves["::-1"];                        // (6, 2) float16, negative stride

            var columns = new Dictionary<string, NDArray> { ["T"] = transposed, ["S"] = stepped, ["H"] = reversed };
            using (NDArrayDataView view = columns.AsDataView())
            {
                Assert.AreEqual(NumberDataViewType.Int32, view.Schema["S"].Type);
                Assert.AreEqual("Vector<Single, 2>", view.Schema["H"].Type.ToString(), "Half is fed as Single");
                using NDArray t = view.ToNDArray("T");
                using NDArray s = view.ToNDArray("S");
                using NDArray h = view.ToNDArray("H");
                Assert.IsTrue(np.array_equal(t, transposed));
                Assert.IsTrue(np.array_equal(s, stepped));
                using NDArray hf = reversed.astype(np.float32);
                Assert.IsTrue(np.array_equal(h, hf));
            }

            NDArrayDataView snapshot;
            using (NDArray a = Arange(NPTypeCode.Single, 5, 2))
            using (NDArray b = Arange(NPTypeCode.Int64, 5))
            {
                snapshot = new Dictionary<string, NDArray> { ["A"] = a, ["B"] = b }.ToDataView();
            }
            using (snapshot)
            using (NDArray b = snapshot.ToNDArray("B"))
                CollectionAssert.AreEqual(new long[] { 0, 1, 2, 3, 4 }, b.ToArray<long>(), "the snapshot outlives its disposed sources");
        }

        /// <summary>
        ///     Everything the names, shapes and dtypes decide is refused before anything is pinned or converted: no column,
        ///     an empty name, a null array, a 0-d or 3-D array, rows that disagree, a complex or a text array.
        /// </summary>
        [TestMethod]
        public void Refusals_ComeBeforeAnythingIsPinned()
        {
            using NDArray x = Arange(NPTypeCode.Single, 4, 2);
            using NDArray y3 = Arange(NPTypeCode.Single, 3);
            using NDArray cube = Arange(NPTypeCode.Single, 2, 2, 2);
            using NDArray complex = np.ones(new Shape(4), NPTypeCode.Complex);
            using NDArray text = np.array(new[] { "a", "b", "c", "d" });

            Assert.ThrowsException<ArgumentNullException>(() => ((IReadOnlyDictionary<string, NDArray>)null).AsDataView());
            Assert.ThrowsException<ArgumentException>(() => new Dictionary<string, NDArray>().AsDataView());
            Assert.ThrowsException<ArgumentException>(() => new Dictionary<string, NDArray> { [""] = x }.AsDataView());
            Assert.ThrowsException<ArgumentNullException>(() => new Dictionary<string, NDArray> { ["X"] = null }.AsDataView());
            Assert.ThrowsException<ArgumentException>(() => new Dictionary<string, NDArray> { ["X"] = cube }.AsDataView());
            ArgumentException rows = Assert.ThrowsException<ArgumentException>(() => new Dictionary<string, NDArray> { ["X"] = x, ["Y"] = y3 }.AsDataView());
            StringAssert.Contains(rows.Message, "column 'Y' has 3 rows but column 'X' has 4");
            Assert.ThrowsException<NotSupportedException>(() => new Dictionary<string, NDArray> { ["X"] = x, ["C"] = complex }.AsDataView());
            NotSupportedException t = Assert.ThrowsException<NotSupportedException>(() => new Dictionary<string, NDArray> { ["X"] = x, ["T"] = text }.ToDataView());
            StringAssert.Contains(t.Message, "text arrays");
            Assert.IsTrue(x.Storage.InternalArray.IsUniquelyReferenced, "a refused view leaves no pin behind");
        }

        /// <summary>
        ///     A released buffer in the SECOND array is found when the pins are taken: the first array's pin and its
        ///     temporary are given back, so the failure leaves nothing referenced.
        /// </summary>
        [TestMethod]
        public void AReleasedSecondArray_GivesBackTheFirstArraysPinAndTemporary()
        {
            using NDArray halves = Arange(NPTypeCode.Half, 4, 2);          // converted to a Single temporary the view would own
            NDArray released = Arange(NPTypeCode.Single, 4);
            released.Dispose();
            Assert.ThrowsException<ObjectDisposedException>(() => new Dictionary<string, NDArray> { ["H"] = halves, ["R"] = released }.AsDataView());
            Assert.IsTrue(halves.Storage.InternalArray.IsUniquelyReferenced, "the first array's pin is given back");
        }

        /// <summary>
        ///     The view holds a pin on EVERY array and each cursor its own set: the sources may be disposed while the view
        ///     and a cursor still read them, and disposing the view and the cursor gives every pin back.
        /// </summary>
        [TestMethod]
        public void PinsOnEveryArray_OutliveTheSources_AndAreAllGivenBack()
        {
            NDArray x = Arange(NPTypeCode.Single, 3, 2);
            NDArray y = Arange(NPTypeCode.Int32, 3);
            NDArrayDataView view = new Dictionary<string, NDArray> { ["X"] = x, ["Y"] = y }.AsDataView();
            Assert.IsFalse(x.Storage.InternalArray.IsUniquelyReferenced);
            Assert.IsFalse(y.Storage.InternalArray.IsUniquelyReferenced);
            x.Dispose();
            y.Dispose();

            using (DataViewRowCursor cursor = view.GetRowCursor(view.Schema))
            {
                ValueGetter<VBuffer<float>> gx = cursor.GetGetter<VBuffer<float>>(view.Schema["X"]);
                ValueGetter<int> gy = cursor.GetGetter<int>(view.Schema["Y"]);
                view.Dispose();                                             // the cursor's own pins keep both buffers
                VBuffer<float> row = default;
                int label = 0;
                var seen = new List<float>();
                var labels = new List<int>();
                while (cursor.MoveNext())
                {
                    gx(ref row);
                    gy(ref label);
                    seen.AddRange(row.DenseValues());
                    labels.Add(label);
                }
                CollectionAssert.AreEqual(new float[] { 0, 1, 2, 3, 4, 5 }, seen);
                CollectionAssert.AreEqual(new[] { 0, 1, 2 }, labels);
            }
        }
    }
}
