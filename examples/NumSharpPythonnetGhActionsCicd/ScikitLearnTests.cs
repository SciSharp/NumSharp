using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetGhActionsCicd;

/// <summary>
///     scikit-learn: estimators fit on NumSharp arrays, and what they learn (labels, centers, coefficients,
///     predictions) returns as NumSharp arrays that C# checks against the data it generated.
/// </summary>
[TestClass]
public class ScikitLearnTests : PythonTest
{
    /// <summary>
    ///     <c>KMeans</c> separates two groups of points that NumSharp generated, and its centers are the
    ///     means that NumSharp computes for the same groups.
    /// </summary>
    [TestMethod]
    public void KMeansFindsTheTwoGroups()
    {
        using var scope = NDScope.Open();
        np.random.seed(7);
        NDArray cold = np.random.normal(-2.0, 0.5, new Shape(50, 2));        // 50 points around (-2, -2)
        NDArray warm = np.random.normal(3.0, 0.5, new Shape(50, 2));         // 50 points around (3, 3)
        NDArray points = np.concatenate((cold, warm));                       // (100, 2)
        using (Py.GIL())
        {
            py.Exec("from sklearn.cluster import KMeans");
            dynamic model = py.KMeans(n_clusters: 2, n_init: 10, random_state: 0).fit(points);
            NDArray labels = model.labels_;                                  // one cluster number for each point
            NDArray centers = model.cluster_centers_;                        // (2, 2)

            Assert.AreEqual(100L, labels.size);
            Assert.AreEqual(NPTypeCode.Int32, labels.typecode, "scikit-learn numbers clusters with 32-bit integers");
            int coldLabel = labels.GetInt32(0), warmLabel = labels.GetInt32(99);
            Assert.AreNotEqual(coldLabel, warmLabel, "the two groups get different cluster numbers");
            for (int i = 0; i < 50; i++)
            {
                Assert.AreEqual(coldLabel, labels.GetInt32(i), $"point {i} belongs to the first group");
                Assert.AreEqual(warmLabel, labels.GetInt32(50 + i), $"point {50 + i} belongs to the second group");
            }

            NDArray coldMean = np.mean(cold, 0), warmMean = np.mean(warm, 0);
            for (int axis = 0; axis < 2; axis++)
            {
                Assert.AreEqual(coldMean.GetDouble(axis), centers.GetDouble(coldLabel, axis), 1e-9, "a center is the mean of its group");
                Assert.AreEqual(warmMean.GetDouble(axis), centers.GetDouble(warmLabel, axis), 1e-9);
            }
        }
    }

    /// <summary>
    ///     <c>LinearRegression</c> recovers the weights and the offset that NumSharp used to build the
    ///     target, and its predictions are the target again.
    /// </summary>
    [TestMethod]
    public void LinearRegressionRecoversTheWeights()
    {
        using var scope = NDScope.Open();
        np.random.seed(11);
        NDArray features = np.random.uniform(-1.0, 1.0, new Shape(40, 3));
        NDArray weights = np.array(new[] { 1.5, -2.0, 0.5 });
        NDArray target = np.dot(features, weights) + 4.0;                    // NumSharp builds the exact answer
        using (Py.GIL())
        {
            py.Exec("from sklearn.linear_model import LinearRegression");
            dynamic model = py.LinearRegression().fit(features, target);
            NDArray learned = model.coef_;
            double offset = model.intercept_;
            double score = model.score(features, target);
            NDArray predicted = model.predict(features);

            for (int i = 0; i < 3; i++)
                Assert.AreEqual(weights.GetDouble(i), learned.GetDouble(i), 1e-9);
            Assert.AreEqual(4.0, offset, 1e-9);
            Assert.AreEqual(1.0, score, 1e-12, "R squared of an exact fit");
            Assert.IsTrue(np.allclose(predicted, target));
        }
    }

    /// <summary>
    ///     A scikit-learn pipeline (scale, then project on the main axis) transforms NumSharp data. NumSharp
    ///     verifies the result: the projection keeps almost all of the variance of two correlated columns.
    /// </summary>
    [TestMethod]
    public void PipelineTransformsNumSharpData()
    {
        using var scope = NDScope.Open();
        np.random.seed(3);
        NDArray length = np.random.normal(10.0, 2.0, new Shape(200));
        NDArray width = 0.5 * length + np.random.normal(0.0, 0.1, new Shape(200));   // nearly proportional to the length
        NDArray table = np.stack(new[] { length, width }, 1);                         // (200, 2)
        using (Py.GIL())
        {
            py.Exec("from sklearn.pipeline import make_pipeline\n" +
                    "from sklearn.preprocessing import StandardScaler\n" +
                    "from sklearn.decomposition import PCA");
            dynamic pipeline = py.make_pipeline(py.StandardScaler(), py.PCA(n_components: 1));
            NDArray projected = pipeline.fit_transform(table);
            NDArray kept = pipeline[1].explained_variance_ratio_;

            CollectionAssert.AreEqual(new long[] { 200, 1 }, projected.shape);
            Assert.AreEqual(0.0, np.mean(projected).GetDouble(), 1e-9, "a projection of centered data is centered");
            Assert.AreEqual(0.997, kept.GetDouble(0), 0.002, "one axis explains the two correlated columns");
            Assert.AreEqual(10.0, np.mean(length).GetDouble(), 0.5, "the pipeline did not change the NumSharp input");
        }
    }
}
