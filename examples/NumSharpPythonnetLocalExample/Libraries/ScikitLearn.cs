using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetLocalExample;

/// <summary>
///     scikit-learn: estimators fit on NumSharp arrays, and what they learn (labels, centers, coefficients,
///     predictions) returns as NumSharp arrays that C# compares with the data it generated.
/// </summary>
/// <remarks>
///     Read it top to bottom: the comment beside a line states what that line leaves behind. The same
///     steps, with every statement asserted, are <c>ScikitLearnTests.cs</c> in NumSharpPythonnetGhActionsCicd.
/// </remarks>
static class ScikitLearnExample
{
    /// <summary>Runs the example.</summary>
    /// <param name="py">A Python namespace in which <c>import numpy as np</c> already ran. It is used only inside <c>using (Py.GIL())</c> blocks.</param>
    /// <exception cref="PythonException">
    ///     A Python statement raised an error, for example because the library is not installed in the
    ///     interpreter that the process embeds.
    /// </exception>
    public static void Run(dynamic py)
    {
        using var scope = NDScope.Open();                                    // every NDArray built below is released when the method returns
        using (Py.GIL())
            py.Exec("from sklearn.cluster import KMeans\n" +
                    "from sklearn.linear_model import LinearRegression\n" +
                    "from sklearn.pipeline import make_pipeline\n" +
                    "from sklearn.preprocessing import StandardScaler\n" +
                    "from sklearn.decomposition import PCA");

        // ---- 1. KMeans separates two groups of points that NumSharp generated -------------------------
        np.random.seed(7);                                                   // NumSharp's generator gives the numbers NumPy gives for the same seed
        NDArray cold = np.random.normal(-2.0, 0.5, new Shape(50, 2));        // 50 points around (-2, -2)
        NDArray warm = np.random.normal(3.0, 0.5, new Shape(50, 2));         // 50 points around (3, 3)
        NDArray points = np.concatenate((cold, warm));                       // (100, 2)
        using (Py.GIL())
        {
            dynamic model = py.KMeans(n_clusters: 2, n_init: 10, random_state: 0).fit(points);   // constructor keywords are C# named arguments
            NDArray labels = model.labels_;                                  // (100,) int32: one cluster number for the first 50 points, the other for the last 50
            NDArray centers = model.cluster_centers_;                        // (2, 2): the rows are np.mean(cold, 0) and np.mean(warm, 0), in the order of the labels
        }

        // ---- 2. LinearRegression recovers the weights NumSharp built the target with ------------------
        np.random.seed(11);
        NDArray features = np.random.uniform(-1.0, 1.0, new Shape(40, 3));
        NDArray weights = np.array(new[] { 1.5, -2.0, 0.5 });
        NDArray target = np.dot(features, weights) + 4.0;                    // NumSharp builds the exact answer
        using (Py.GIL())
        {
            dynamic model = py.LinearRegression().fit(features, target);
            NDArray learned = model.coef_;                                   // [1.5, -2.0, 0.5]
            double offset = model.intercept_;                                // 4.0
            double score = model.score(features, target);                    // 1.0: R squared of an exact fit
            NDArray predicted = model.predict(features);                     // the target again (np.allclose)
        }

        // ---- 3. a pipeline: scale, then project on the main axis --------------------------------------
        np.random.seed(3);
        NDArray length = np.random.normal(10.0, 2.0, new Shape(200));
        NDArray width = 0.5 * length + np.random.normal(0.0, 0.1, new Shape(200));   // nearly proportional to the length
        NDArray table = np.stack(new[] { length, width }, 1);                         // (200, 2)
        using (Py.GIL())
        {
            dynamic pipeline = py.make_pipeline(py.StandardScaler(), py.PCA(n_components: 1));   // Python objects are arguments like any other
            NDArray projected = pipeline.fit_transform(table);               // (200, 1), centered on 0
            NDArray kept = pipeline[1].explained_variance_ratio_;            // [0.997]: one axis explains the two correlated columns
        }
    }
}
