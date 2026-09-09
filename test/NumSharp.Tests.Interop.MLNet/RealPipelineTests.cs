using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     End-to-end scenarios that produce the harder output-column shapes — a multiclass Score VECTOR, a one-hot
    ///     vector, a key column — and read them back into NumSharp.
    /// </summary>
    [TestClass]
    public class RealPipelineTests : MLNetTestBase
    {
        [TestMethod]
        public void Multiclass_ScoreVectorColumn_MaterializesAsMatrix()
        {
            // Three linearly-separable clusters -> 3 classes.
            var rows = new System.Collections.Generic.List<MRow>();
            for (int i = 0; i < 30; i++)
            {
                int cls = i % 3;
                rows.Add(new MRow { F0 = cls * 10 + (i % 5) * 0.1f, F1 = cls * 10 + (i % 7) * 0.1f, Label = "c" + cls });
            }
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);

            var pipeline = Ml.Transforms.Conversion.MapValueToKey("Label")
                .Append(Ml.Transforms.Concatenate("Features", "F0", "F1"))
                .Append(Ml.MulticlassClassification.Trainers.SdcaMaximumEntropy());
            ITransformer model = pipeline.Fit(dv);
            IDataView output = model.Transform(dv);

            // Score is a length-3 probability vector per row.
            Assert.IsInstanceOfType(output.Schema["Score"].Type, typeof(VectorDataViewType));
            using NDArray score = output.ToNDArray("Score");
            AssertShape(score, 30, 3);
            Assert.AreEqual(NPTypeCode.Single, score.typecode);
            Assert.IsTrue((bool)np.all(np.isfinite(score)), "all scores finite");
            using NDArray rowSums = np.sum(score, axis: -1);
            using NDArray ones = np.ones_like(rowSums);
            Assert.IsTrue((bool)np.allclose(rowSums, ones, atol: 1e-4), "max-entropy scores are probabilities (rows sum to 1)");

            // The predicted class == argmax of the score vector.
            using NDArray predicted = np.argmax(score, axis: -1);
            using NDArray key = output.ToNDArray("PredictedLabel");   // 1-based key
            Assert.AreEqual(NPTypeCode.UInt32, key.typecode);
            using NDArray keyZeroBased = key.astype(NPTypeCode.Int64) - 1;
            Assert.IsTrue(np.array_equal(predicted.astype(NPTypeCode.Int64), keyZeroBased), "argmax(Score) == PredictedLabel-1");
        }

        [TestMethod]
        public void OneHotEncoding_VectorColumn_IsOneHotPerRow()
        {
            var rows = new[]
            {
                new CatRow { Cat = "red" }, new CatRow { Cat = "green" },
                new CatRow { Cat = "blue" }, new CatRow { Cat = "red" },
            };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            ITransformer model = Ml.Transforms.Categorical.OneHotEncoding("OneHot", "Cat").Fit(dv);
            IDataView output = model.Transform(dv);

            using NDArray oneHot = output.ToNDArray("OneHot");
            Assert.AreEqual(4L, oneHot.shape[0]);
            Assert.AreEqual(3L, oneHot.shape[1], "three distinct categories");
            using NDArray rowSums = np.sum(oneHot, axis: -1);
            using NDArray ones = np.ones_like(rowSums);
            Assert.IsTrue((bool)np.allclose(rowSums, ones), "each row is one-hot (sums to 1)");
            // rows 0 and 3 are both "red" -> identical one-hot rows
            using NDArray r0 = oneHot[0];
            using NDArray r3 = oneHot[3];
            Assert.IsTrue(np.array_equal(r0, r3));
        }

        [TestMethod]
        public void KeyRoundTrip_ValueToKey_ReadsAsUInt32()
        {
            var rows = new[] { new CatRow { Cat = "a" }, new CatRow { Cat = "b" }, new CatRow { Cat = "c" }, new CatRow { Cat = "b" } };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            ITransformer model = Ml.Transforms.Conversion.MapValueToKey("Key", "Cat").Fit(dv);
            IDataView output = model.Transform(dv);
            using NDArray key = output.ToNDArray("Key");
            Assert.AreEqual(NPTypeCode.UInt32, key.typecode);
            AssertShape(key, 4);
            // a->1, b->2, c->3, b->2
            CollectionAssert.AreEqual(new uint[] { 1, 2, 3, 2 }, key.ToArray<uint>());
        }

        [TestMethod]
        public void NumSharp_AsDataView_ReadByMLNet_MatchesPocoLoad()
        {
            // The same numbers via a NumSharp-backed view and via ML.NET's own POCO loader must read identically.
            using NDArray data = np.array(new float[] { 1, 2, 3, 4, 5, 6 }).reshape(3, 2);
            using NDArrayDataView nsView = data.AsDataView(new[] { "A", "B" });

            var poco = new[]
            {
                new AB { A = 1, B = 2 }, new AB { A = 3, B = 4 }, new AB { A = 5, B = 6 },
            };
            IDataView pocoView = Ml.Data.LoadFromEnumerable(poco);

            using NDArray nsA = nsView.ToNDArray("A");
            using NDArray pocoA = pocoView.ToNDArray("A");
            Assert.IsTrue(np.array_equal(nsA, pocoA), "NumSharp-backed and POCO-backed views read the same column");
        }

        private class MRow
        {
            public float F0 { get; set; }
            public float F1 { get; set; }
            public string Label { get; set; }
        }

        private class CatRow
        {
            public string Cat { get; set; }
        }

        private class AB
        {
            public float A { get; set; }
            public float B { get; set; }
        }
    }
}
