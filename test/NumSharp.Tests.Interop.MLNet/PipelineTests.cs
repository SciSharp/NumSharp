using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     The whole point: an NDArray feeds a REAL <c>Microsoft.ML</c> pipeline as an <see cref="IDataView"/> and the
    ///     output column comes straight back as an NDArray — <c>transformer.Transform(nd.AsDataView(...)).ToNDArray(...)</c>,
    ///     no per-row POCO plumbing.
    /// </summary>
    [TestClass]
    public class PipelineTests : MLNetTestBase
    {
        [TestMethod]
        public void NormalizeMinMax_FeatureMatrix_RoundTrips()
        {
            using NDArray x = Arange(NPTypeCode.Single, 5, 3);
            using NDArrayDataView dv = x.AsDataView("Features");

            var pipeline = Ml.Transforms.NormalizeMinMax("Norm", "Features");
            ITransformer model = pipeline.Fit(dv);
            IDataView outView = model.Transform(dv);

            using NDArray norm = outView.ToNDArray("Norm");
            AssertShape(norm, 5, 3);
            using NDArray mn = np.min(norm);
            using NDArray mx = np.max(norm);
            Assert.IsTrue((float)mn >= 0f, "min >= 0");
            Assert.IsTrue((float)mx <= 1f + 1e-6f, "max <= 1");
            // per-column min value (row 0) maps to 0, max (row 4) to 1
            Assert.AreEqual(0f, (float)norm[0, 0], 1e-6f);
            Assert.AreEqual(1f, (float)norm[4, 0], 1e-6f);
        }

        [TestMethod]
        public void Concatenate_ScalarColumns_IntoFeatureVector()
        {
            using NDArray data = Arange(NPTypeCode.Single, 4, 2);   // columns f0, f1
            using NDArrayDataView dv = data.AsDataView(new[] { "f0", "f1" });

            var pipeline = Ml.Transforms.Concatenate("Features", "f0", "f1");
            ITransformer model = pipeline.Fit(dv);
            IDataView outView = model.Transform(dv);

            using NDArray features = outView.ToNDArray("Features");
            AssertShape(features, 4, 2);
            Assert.IsTrue(np.array_equal(data, features), "concatenated features == the source columns");
        }

        [TestMethod]
        public void TrainRegression_ThenPredict_RunsThroughTheBridge()
        {
            // A clean linear target y = 2*f0 + 3*f1 over 20 samples. The trainer's fit quality is not the
            // subject here (Sdca on 20 scaled points underfits) — what is asserted is that the whole bridge
            // works: an NDArray-backed IDataView trains a real model and its Score column reads back as an NDArray.
            using NDArray f0 = Arange(NPTypeCode.Single, 20);
            using NDArray f1 = (Arange(NPTypeCode.Single, 20) * 0.5f);
            using NDArray label = (2f * f0 + 3f * f1);
            using NDArray data = np.stack(new[] { f0, f1, label }, axis: 1);   // (20,3)

            using NDArrayDataView dv = data.AsDataView(new[] { "f0", "f1", "Label" });

            var pipeline = Ml.Transforms.Concatenate("Features", "f0", "f1")
                .Append(Ml.Transforms.NormalizeMinMax("Features", "Features"))
                .Append(Ml.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features", maximumNumberOfIterations: 100));
            ITransformer model = pipeline.Fit(dv);
            IDataView outView = model.Transform(dv);

            using NDArray score = outView.ToNDArray("Score");
            AssertShape(score, 20);
            Assert.AreEqual(NPTypeCode.Single, score.typecode);
            Assert.IsTrue((bool)np.all(np.isfinite(score)), "all predictions finite");
            using NDArray std = np.std(score);
            Assert.IsTrue((float)std > 1e-3f, "the model consumed the feature variation the IDataView fed it (scores vary)");
        }

        [TestMethod]
        public void PredictSingleSample_1D_Features()
        {
            using NDArray train = Arange(NPTypeCode.Single, 10, 2);
            using NDArray labels = np.arange(10).astype(NPTypeCode.Single);
            using NDArray data = np.concatenate(new[] { train, labels.reshape(10, 1) }, axis: 1);   // (10,3): f0,f1,Label

            using NDArrayDataView trainView = data.AsDataView(new[] { "f0", "f1", "Label" });
            var pipeline = Ml.Transforms.Concatenate("Features", "f0", "f1")
                .Append(Ml.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features", maximumNumberOfIterations: 100));
            ITransformer model = pipeline.Fit(trainView);

            // One sample as a 1-D feature vector -> a single-row IDataView with the two scalar columns.
            using NDArray sample = np.array(new float[] { 4f, 5f }).reshape(1, 2);
            using NDArrayDataView sampleView = sample.AsDataView(new[] { "f0", "f1" });
            IDataView outView = model.Transform(sampleView);

            using NDArray score = outView.ToNDArray("Score");
            AssertShape(score, 1);
            Assert.IsTrue((bool)np.all(np.isfinite(score)), "single prediction is finite");
        }
    }
}
