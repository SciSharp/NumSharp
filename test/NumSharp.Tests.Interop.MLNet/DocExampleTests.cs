using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     The code the docs (docs/website-src/docs/interop/mlnet.md) and README teach — kept runnable so the
    ///     examples cannot rot (the analog of the ONNX bridge's DocExampleTests).
    /// </summary>
    [TestClass]
    public class DocExampleTests : MLNetTestBase
    {
        [TestMethod]
        public void TwoVerbBridge_FeedATrainedModel_ReadTheScore()
        {
            // "Feeding a model is a two-verb bridge — no per-transformer runner:
            //   transformer.Transform(nd.AsDataView("Features")).ToNDArray("Score")"
            using NDArray train = BuildRegressionData(out ITransformer trainedModel);

            // A (rows, features) matrix in NumSharp.
            using NDArray features = np.array(new float[,] { { 5.1f, 3.5f }, { 4.9f, 3.0f }, { 6.3f, 3.3f } });
            using NDArrayDataView input = features.AsDataView(new[] { "f0", "f1" });
            IDataView output = trainedModel.Transform(input);
            using NDArray scores = output.ToNDArray("Score");

            AssertShape(scores, 3);
            Assert.AreEqual(NPTypeCode.Single, scores.typecode);
            Assert.IsTrue((bool)np.all(np.isfinite(scores)));
        }

        [TestMethod]
        public void FeatureMatrix_NormalizeMinMax_TheZeroToPredictionSnippet()
        {
            // The "From zero to a prediction" snippet: a (rows, features) matrix -> one Features vector column.
            using NDArray features = np.array(new float[,] { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } });
            using NDArrayDataView input = features.AsDataView("Features");

            var pipeline = Ml.Transforms.NormalizeMinMax("Norm", "Features");
            ITransformer model = pipeline.Fit(input);
            IDataView output = model.Transform(input);
            using NDArray norm = output.ToNDArray("Norm");

            AssertShape(norm, 3, 2);
            using NDArray mn = np.min(norm);
            using NDArray mx = np.max(norm);
            Assert.IsTrue((float)mn >= 0f && (float)mx <= 1f + 1e-6f);
        }

        [TestMethod]
        public void PostProcess_TopKOverSoftmax_OfAScoreColumn()
        {
            // The classifier post-processing example: a score matrix -> stable softmax -> argmax / top-k, all np.
            using NDArray scores = np.array(new float[] { 0.1f, 3.0f, 0.2f, 2.5f, 0.05f,
                                                          2.0f, 0.1f, 4.0f, 0.3f, 0.6f }).reshape(2, 5);
            using NDArray probs = Postprocess.Softmax(scores);           // rows sum to 1
            using NDArray top1 = Postprocess.Argmax(probs, axis: -1);    // predicted class id
            (NDArray values, NDArray indices) = Postprocess.TopK(scores, k: 3);

            using (values)
            using (indices)
            {
                using NDArray rowSums = np.sum(probs, axis: -1);
                using NDArray ones = np.ones_like(rowSums);
                Assert.IsTrue((bool)np.allclose(rowSums, ones));
                Assert.AreEqual(1L, (long)top1[0]);   // row 0 argmax at index 1 (3.0)
                Assert.AreEqual(2L, (long)top1[1]);   // row 1 argmax at index 2 (4.0)
                AssertShape(values, 2, 3);
                Assert.AreEqual(3.0f, (float)values[0, 0]);   // row 0 top value
                Assert.AreEqual(4.0f, (float)values[1, 0]);   // row 1 top value
            }
        }

        [TestMethod]
        public void ScalarColumns_TrainingData_Concatenate_TheTwoShapesSnippet()
        {
            // The "one scalar column per feature" snippet: build training data, Concatenate into Features, train.
            using NDArray f0 = np.arange(20).astype(NPTypeCode.Single);
            using NDArray f1 = (np.arange(20).astype(NPTypeCode.Single) * 0.5f);
            using NDArray label = (2f * f0 + 3f * f1);
            using NDArray data = np.stack(new[] { f0, f1, label }, axis: 1);   // (20, 3)

            using NDArrayDataView t = data.AsDataView(new[] { "SepalLength", "SepalWidth", "Label" });
            var pipeline = Ml.Transforms.Concatenate("Features", "SepalLength", "SepalWidth")
                .Append(Ml.Regression.Trainers.Sdca(labelColumnName: "Label", maximumNumberOfIterations: 50));
            ITransformer model = pipeline.Fit(t);

            IDataView output = model.Transform(t);
            using NDArray scores = output.ToNDArray("Score");
            AssertShape(scores, 20);
            Assert.IsTrue((bool)np.all(np.isfinite(scores)));
        }

        private NDArray BuildRegressionData(out ITransformer model)
        {
            NDArray f0 = np.arange(10).astype(NPTypeCode.Single);
            NDArray f1 = (np.arange(10).astype(NPTypeCode.Single) * 0.3f);
            NDArray label = (f0 + 2f * f1);
            NDArray data = np.stack(new[] { f0, f1, label }, axis: 1);   // (10,3)
            f0.Dispose(); f1.Dispose(); label.Dispose();

            using NDArrayDataView t = data.AsDataView(new[] { "f0", "f1", "Label" });
            var pipeline = Ml.Transforms.Concatenate("Features", "f0", "f1")
                .Append(Ml.Regression.Trainers.Sdca(labelColumnName: "Label", maximumNumberOfIterations: 50));
            model = pipeline.Fit(t);
            return data;
        }
    }
}
