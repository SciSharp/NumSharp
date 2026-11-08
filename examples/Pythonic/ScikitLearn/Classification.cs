namespace Pythonic;

using NumSharp;                                    // import numpy as np
using static Python.builtins;                      // print needs no import in Python
using static Python.sklearn.datasets;              // from sklearn.datasets import load_breast_cancer
using static Python.sklearn.linear_model;          // from sklearn.linear_model import LogisticRegression
using static Python.sklearn.metrics;               // from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
using static Python.sklearn.model_selection;       // from sklearn.model_selection import train_test_split
using static Python.sklearn.preprocessing;         // from sklearn.preprocessing import StandardScaler

/// <summary>
///     Breast cancer screening: train a classifier, then measure it on patients that it did not see. It is
///     the feature that scikit-learn is used for most: split, scale, fit, predict, score.
/// </summary>
/// <remarks>
///     <para>This is the C# port of <c>ScikitLearn/classification.py</c>. Put the two files side by side:
///     they have the same statements in the same order, and they print the same lines.</para>
///     <para>What C# writes in another way: <c>var</c> in front of a new name, <c>new</c> in front of a
///     class, <c>:</c> for the <c>=</c> of a keyword argument, <c>$</c> for the <c>f</c> of an f-string, and a
///     slice as a string (<c>X[":, :10"]</c>). The one line that Python does not have is the scope.</para>
/// </remarks>
static class Classification
{
    /// <summary>Runs the program and prints its seven lines to the console.</summary>
    /// <exception cref="PythonInteropException">No usable Python was found, or scikit-learn is not installed in it. The message says what to do.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">A scikit-learn call raised a Python error.</exception>
    public static void Run()
    {
        using var scope = new PythonScope(NDScope.Open());     // releases every array and every Python object of this method when it returns

        // 569 tumors with 30 numbers each. The label is 0 for a malignant tumor and 1 for a benign one.
        var (X, y) = load_breast_cancer(return_X_y: true);
        print($"tumors     {X.shape[0]} with {X.shape[1]} numbers each: {np.count_nonzero(y == 0)} malignant, {np.count_nonzero(y == 1)} benign");

        // The table holds ten measurements of the cell nuclei three times: the mean, the standard error and the
        // worst value. This model uses the ten means, which are the first ten columns.
        var means = X[":, :10"];

        // Keep one quarter of the patients apart. The model does not see them before the test.
        var (X_train, X_test, y_train, y_test) = train_test_split(means, y, test_size: 0.25, stratify: y, random_state: 0);

        // An area is near 650 and a smoothness near 0.1. Bring each column to mean 0 and deviation 1.
        // The scaler learns the two numbers of a column from the training patients only.
        var scaler = new StandardScaler();
        var X_train_scaled = scaler.fit_transform(X_train);
        var X_test_scaled = scaler.transform(X_test);

        var model = new LogisticRegression(max_iter: 1000);
        model.fit(X_train_scaled, y_train);

        var y_pred = model.predict(X_test_scaled);
        var proba = model.predict_proba(X_test_scaled);
        var p_benign = proba[":, 1"];

        print($"accuracy   {accuracy_score(y_test, y_pred):.4f}");
        print($"roc auc    {roc_auc_score(y_test, p_benign):.4f}");
        print(confusion_matrix(y_test, y_pred));

        // A malignant tumor that the model calls benign is the error that costs most. The model calls a tumor
        // benign when p_benign is above 0.5. Which limit finds every malignant tumor of the test, and how many
        // benign tumors does that limit send to a biopsy?
        var malignant = y_test == 0;
        var missed = np.count_nonzero(malignant & (y_pred == 1));
        var limit = p_benign[malignant].max();
        var referred = p_benign <= limit;
        print($"missed     {missed} of {np.count_nonzero(malignant)} malignant tumors at the limit 0.5");
        print($"limit      {limit:.4f} finds all of them and refers {np.count_nonzero(referred & ~malignant)} of {np.count_nonzero(~malignant)} benign tumors");
    }
}
