using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Pythonic;

/// <summary>
///     The C# port of <c>ScikitLearn/classification.py</c>: train a classifier on the breast cancer table,
///     then measure it on patients that it did not see. It is the feature that scikit-learn is used for
///     most: split, scale, fit, predict, score.
/// </summary>
/// <remarks>
///     <para>Read it beside the Python program. The comment above a group of lines is the Python that the
///     group ports, and the two programs print the same lines.</para>
///     <list type="bullet">
///       <item>A scikit-learn call is a member of <see cref="Python.sklearn"/> with the same name.</item>
///       <item>A NumPy line is a NumSharp line with the same operators: <c>y_test == 0</c>, <c>a &amp; b</c>,
///         <c>~mask</c>, <c>p[mask]</c>. Only a slice is spelled as a string: <c>X[:, :10]</c> is <c>X[":, :10"]</c>.</item>
///       <item>An array changes sides in one visible call. A name that ends in <c>_np</c> is NumPy's side of
///         the array that NumSharp holds under the name without the ending.</item>
///     </list>
/// </remarks>
static class Classification
{
    /// <summary>Runs the program and prints its seven lines to the console.</summary>
    /// <exception cref="PythonInteropException">No usable Python was found, or scikit-learn is not installed in it. The message says what to do.</exception>
    /// <exception cref="PythonException">A scikit-learn call raised a Python error.</exception>
    public static void Run()
    {
        using var scope = NDScope.Open();                    // every NDArray below is released when the method returns

        // X, y = load_breast_cancer(return_X_y=True)
        (NDArray X, NDArray y) = PythonRuntime.Run(() =>
        {
            using PyObject table = Python.sklearn.datasets.load_breast_cancer(return_X_y: true);
            using PyObject X_np = table[0], y_np = table[1];
            return (X_np.ToNDArray(), y_np.ToNDArray());     // one copy each: from here on NumSharp owns the table
        });

        // print(f"tumors     {X.shape[0]} with {X.shape[1]} numbers each: {np.count_nonzero(y == 0)} malignant, {np.count_nonzero(y == 1)} benign")
        Console.WriteLine($"tumors     {X.shape[0]} with {X.shape[1]} numbers each: {np.count_nonzero(y == 0)} malignant, {np.count_nonzero(y == 1)} benign");

        // means = X[:, :10]
        NDArray means = X[":, :10"];                         // a view of X: ten of every thirty numbers, nothing copied

        var (y_test, y_pred, p_benign, accuracy, roc_auc, confusion) = PythonRuntime.Run(() =>
        {
            using PyObject means_np = means.ToNumpy(), y_np = y.ToNumpy();   // NumPy views NumSharp's memory, the strided columns included

            // X_train, X_test, y_train, y_test = train_test_split(means, y, test_size=0.25, stratify=y, random_state=0)
            using PyObject parts = Python.sklearn.model_selection.train_test_split(means_np, y_np, test_size: 0.25, stratify: y_np, random_state: 0);
            using PyObject X_train = parts[0], X_test = parts[1], y_train = parts[2], y_test_np = parts[3];

            // scaler = StandardScaler()
            // X_train_scaled = scaler.fit_transform(X_train)
            // X_test_scaled = scaler.transform(X_test)
            using PyObject scaler = Python.sklearn.preprocessing.StandardScaler.New();
            using PyObject X_train_scaled = Python.sklearn.preprocessing.StandardScaler.fit_transform(scaler, X_train);
            using PyObject X_test_scaled = Python.sklearn.preprocessing.StandardScaler.transform(scaler, X_test);

            // model = LogisticRegression(max_iter=1000)
            // model.fit(X_train_scaled, y_train)
            using PyObject model = Python.sklearn.linear_model.LogisticRegression.New(max_iter: 1000);
            Python.sklearn.linear_model.LogisticRegression.fit(model, X_train_scaled, y_train);

            // y_pred = model.predict(X_test_scaled)
            // proba = model.predict_proba(X_test_scaled)
            // p_benign = proba[:, 1]
            using PyObject y_pred_np = Python.sklearn.linear_model.LogisticRegression.predict(model, X_test_scaled);
            using PyObject proba_np = Python.sklearn.linear_model.LogisticRegression.predict_proba(model, X_test_scaled);
            NDArray proba = proba_np.ToNDArray();            // (143, 2): the probabilities are NumSharp's now
            NDArray p_benign = proba[":, 1"];                // NumSharp slices as NumPy does: a view, every second number

            // accuracy_score(y_test, y_pred), roc_auc_score(y_test, p_benign), confusion_matrix(y_test, y_pred)
            using PyObject p_benign_np = p_benign.ToNumpy(); // that view returns to scikit-learn without a copy
            double accuracy = Python.sklearn.metrics.accuracy_score_t(y_test_np, y_pred_np);
            double roc_auc = Python.sklearn.metrics.roc_auc_score_t(y_test_np, p_benign_np);
            using PyObject confusion_np = Python.sklearn.metrics.confusion_matrix(y_test_np, y_pred_np);

            return (y_test_np.ToNDArray(), y_pred_np.ToNDArray(), p_benign, accuracy, roc_auc, confusion_np.ToNDArray());
        });

        // print(f"accuracy   {accuracy_score(y_test, y_pred):.4f}")
        // print(f"roc auc    {roc_auc_score(y_test, p_benign):.4f}")
        // print(confusion_matrix(y_test, y_pred))
        Console.WriteLine($"accuracy   {accuracy:F4}");
        Console.WriteLine($"roc auc    {roc_auc:F4}");
        Console.WriteLine(confusion);                        // NumSharp prints an array as NumPy does

        // malignant = y_test == 0
        // missed = np.count_nonzero(malignant & (y_pred == 1))
        // limit = p_benign[malignant].max()
        // referred = p_benign <= limit
        NDArray malignant = y_test == 0;
        long missed = np.count_nonzero(malignant & (y_pred == 1));
        double limit = (double)p_benign[malignant].max();
        NDArray referred = p_benign <= limit;

        // print(f"missed     {missed} of {np.count_nonzero(malignant)} malignant tumors at the limit 0.5")
        // print(f"limit      {limit:.4f} finds all of them and refers {np.count_nonzero(referred & ~malignant)} of {np.count_nonzero(~malignant)} benign tumors")
        Console.WriteLine($"missed     {missed} of {np.count_nonzero(malignant)} malignant tumors at the limit 0.5");
        Console.WriteLine($"limit      {limit:F4} finds all of them and refers {np.count_nonzero(referred & ~malignant)} of {np.count_nonzero(~malignant)} benign tumors");
    }
}
