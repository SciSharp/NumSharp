using Python.Runtime;

// The classes of the tree carry Python's module names, and those are lower case. The compiler warns that
// a lower-case type name can become a keyword of a later C# version (CS8981); the names stay as Python has them.
#pragma warning disable CS8981

namespace Pythonic;

static partial class Python
{
    /// <summary>
    ///     <c>import sklearn</c>: the scikit-learn calls of this project, reached as <c>Python.sklearn.*</c>.
    ///     Each nested class is one scikit-learn module, and each member is one call with Python's names.
    /// </summary>
    /// <remarks>
    ///     Call every member inside a <see cref="PythonRuntime.Run(Action)"/> body. An array argument is a
    ///     NumPy array (or anything scikit-learn accepts as one); a NumSharp array becomes one with
    ///     <c>ToNumpy()</c>, without a copy. A returned <see cref="PyObject"/> belongs to the caller.
    /// </remarks>
    internal static class sklearn
    {
        /// <summary><c>sklearn.datasets</c>: the tables that are installed together with scikit-learn.</summary>
        internal static class datasets
        {
            /// <summary>The module. It belongs to the process: it is not disposed.</summary>
            private static PyObject Module => PythonRuntime.Import("sklearn.datasets");

            /// <summary>
            ///     <c>load_breast_cancer(return_X_y=...)</c>: the Wisconsin breast cancer table, 569 tumors
            ///     with 30 numbers each and the diagnosis of each tumor (0 malignant, 1 benign).
            /// </summary>
            /// <param name="return_X_y">
            ///     <c>true</c> gives the tuple <c>(X, y)</c>: a <c>(569, 30)</c> float64 array and a
            ///     <c>(569,)</c> int64 array. <c>false</c>, Python's default, gives a <c>Bunch</c> object
            ///     that also holds the column names.
            /// </param>
            /// <returns>A new reference to the tuple or to the <c>Bunch</c>. Dispose it.</returns>
            /// <exception cref="PythonInteropException">scikit-learn is not installed in the embedded Python.</exception>
            /// <exception cref="PythonException">The call raised a Python error.</exception>
            internal static PyObject load_breast_cancer(bool return_X_y = false)
            {
                using PyObject function = Module.GetAttr("load_breast_cancer");
                using var kwargs = new PyDict();
                using (PyObject value = return_X_y.ToPython())
                    kwargs["return_X_y"] = value;                    // the dict takes its own reference
                return function.Invoke(Array.Empty<PyObject>(), kwargs);
            }
        }

        /// <summary><c>sklearn.model_selection</c>: splits of a table into the part to learn from and the part to test on.</summary>
        internal static class model_selection
        {
            /// <summary>The module. It belongs to the process: it is not disposed.</summary>
            private static PyObject Module => PythonRuntime.Import("sklearn.model_selection");

            /// <summary>
            ///     <c>train_test_split(X, y, test_size=..., random_state=..., stratify=...)</c>: shuffles the
            ///     rows of two arrays in the same order and cuts both into a training part and a test part.
            /// </summary>
            /// <param name="X">The table, one row for each sample.</param>
            /// <param name="y">The label of each row. It has as many entries as <paramref name="X"/> has rows.</param>
            /// <param name="test_size">The part of the rows that goes to the test, between 0 and 1. <c>null</c> leaves Python's default, 0.25.</param>
            /// <param name="random_state">The seed of the shuffle. The same seed gives the same split on every run. <c>null</c> gives another split each time.</param>
            /// <param name="stratify">Labels whose proportions both parts must keep, in most cases <paramref name="y"/> itself. <c>null</c> splits without regard to the labels.</param>
            /// <returns>
            ///     A new reference to a Python list of four new arrays: <c>[X_train, X_test, y_train, y_test]</c>.
            ///     The four arrays are copies. Dispose the list, and each entry that you take out of it.
            /// </returns>
            /// <exception cref="PythonInteropException">scikit-learn is not installed in the embedded Python.</exception>
            /// <exception cref="PythonException">scikit-learn refused the arguments, for example because the two arrays have different lengths.</exception>
            internal static PyObject train_test_split(PyObject X, PyObject y, double? test_size = null, int? random_state = null, PyObject? stratify = null)
            {
                using PyObject function = Module.GetAttr("train_test_split");
                using var kwargs = new PyDict();
                if (test_size is { } size)
                    using (PyObject value = new PyFloat(size))
                        kwargs["test_size"] = value;
                if (random_state is { } seed)
                    using (PyObject value = new PyInt(seed))
                        kwargs["random_state"] = value;
                if (stratify is not null)
                    kwargs["stratify"] = stratify;                   // the caller keeps its reference
                return function.Invoke(new[] { X, y }, kwargs);
            }
        }

        /// <summary><c>sklearn.preprocessing</c>: transformers that prepare a table for a model.</summary>
        internal static class preprocessing
        {
            /// <summary>The module. It belongs to the process: it is not disposed.</summary>
            private static PyObject Module => PythonRuntime.Import("sklearn.preprocessing");

            /// <summary>
            ///     <c>sklearn.preprocessing.StandardScaler</c>: brings each column to mean 0 and standard
            ///     deviation 1. <see cref="New"/> is the constructor. The other members are the methods of
            ///     the object and take it as their first argument.
            /// </summary>
            internal static class StandardScaler
            {
                /// <summary><c>StandardScaler()</c>: a scaler that has not seen data yet.</summary>
                /// <returns>A new reference to the scaler. Dispose it.</returns>
                /// <exception cref="PythonInteropException">scikit-learn is not installed in the embedded Python.</exception>
                /// <exception cref="PythonException">The constructor raised a Python error.</exception>
                internal static PyObject New()
                {
                    using PyObject type = Module.GetAttr("StandardScaler");
                    return type.Invoke();
                }

                /// <summary>
                ///     <c>scaler.fit_transform(X)</c>: learns the mean and the deviation of each column of
                ///     <paramref name="X"/>, keeps them in the scaler, and returns <paramref name="X"/> scaled with them.
                /// </summary>
                /// <param name="scaler">The scaler from <see cref="New"/>.</param>
                /// <param name="X">The table to learn from, in most cases the training part only.</param>
                /// <returns>A new reference to a new float64 array with the shape of <paramref name="X"/>. Dispose it.</returns>
                /// <exception cref="PythonException">scikit-learn refused the table, for example because it holds a NaN that the scaler cannot use.</exception>
                internal static PyObject fit_transform(PyObject scaler, PyObject X) => scaler.InvokeMethod("fit_transform", X);

                /// <summary>
                ///     <c>scaler.transform(X)</c>: scales <paramref name="X"/> with the numbers that
                ///     <see cref="fit_transform"/> learned. It learns nothing, so a test part does not leak into the model.
                /// </summary>
                /// <param name="scaler">A scaler that <see cref="fit_transform"/> already ran on.</param>
                /// <param name="X">A table with the same columns as the one the scaler learned from.</param>
                /// <returns>A new reference to a new float64 array with the shape of <paramref name="X"/>. Dispose it.</returns>
                /// <exception cref="PythonException">The scaler has not learned yet, or <paramref name="X"/> has another number of columns.</exception>
                internal static PyObject transform(PyObject scaler, PyObject X) => scaler.InvokeMethod("transform", X);
            }
        }

        /// <summary><c>sklearn.linear_model</c>: models that are a weighted sum of the columns.</summary>
        internal static class linear_model
        {
            /// <summary>The module. It belongs to the process: it is not disposed.</summary>
            private static PyObject Module => PythonRuntime.Import("sklearn.linear_model");

            /// <summary>
            ///     <c>sklearn.linear_model.LogisticRegression</c>: a classifier that gives each class a
            ///     probability. <see cref="New"/> is the constructor. The other members are the methods of
            ///     the object and take it as their first argument.
            /// </summary>
            internal static class LogisticRegression
            {
                /// <summary><c>LogisticRegression(max_iter=...)</c>: a model that has not learned yet.</summary>
                /// <param name="max_iter">The most steps the solver may take. <c>null</c> leaves Python's default, 100, which a table that is not scaled can exceed.</param>
                /// <returns>A new reference to the model. Dispose it.</returns>
                /// <exception cref="PythonInteropException">scikit-learn is not installed in the embedded Python.</exception>
                /// <exception cref="PythonException">The constructor raised a Python error.</exception>
                internal static PyObject New(int? max_iter = null)
                {
                    using PyObject type = Module.GetAttr("LogisticRegression");
                    using var kwargs = new PyDict();
                    if (max_iter is { } steps)
                        using (PyObject value = new PyInt(steps))
                            kwargs["max_iter"] = value;
                    return type.Invoke(Array.Empty<PyObject>(), kwargs);
                }

                /// <summary><c>model.fit(X, y)</c>: learns the weights from a table and its labels. The model keeps them.</summary>
                /// <param name="model">The model from <see cref="New"/>.</param>
                /// <param name="X">The training table, one row for each sample.</param>
                /// <param name="y">The class of each row.</param>
                /// <remarks>Python returns the model itself from <c>fit</c>, so that calls can be chained. This member drops that second reference.</remarks>
                /// <exception cref="PythonException">scikit-learn refused the data, for example because <paramref name="y"/> holds one class only.</exception>
                internal static void fit(PyObject model, PyObject X, PyObject y)
                {
                    using PyObject self = model.InvokeMethod("fit", X, y);
                }

                /// <summary><c>model.predict(X)</c>: the class that the model gives each row.</summary>
                /// <param name="model">A model that <see cref="fit"/> already ran on.</param>
                /// <param name="X">A table with the columns the model learned from.</param>
                /// <returns>A new reference to a new array with one class for each row, in the dtype of the training labels. Dispose it.</returns>
                /// <exception cref="PythonException">The model has not learned yet, or <paramref name="X"/> has another number of columns.</exception>
                internal static PyObject predict(PyObject model, PyObject X) => model.InvokeMethod("predict", X);

                /// <summary><c>model.predict_proba(X)</c>: the probability that the model gives each class, for each row.</summary>
                /// <param name="model">A model that <see cref="fit"/> already ran on.</param>
                /// <param name="X">A table with the columns the model learned from.</param>
                /// <returns>
                ///     A new reference to a new <c>(rows, classes)</c> float64 array. The columns follow the
                ///     sorted class labels, and each row sums to 1. Dispose it.
                /// </returns>
                /// <exception cref="PythonException">The model has not learned yet, or <paramref name="X"/> has another number of columns.</exception>
                internal static PyObject predict_proba(PyObject model, PyObject X) => model.InvokeMethod("predict_proba", X);
            }
        }

        /// <summary><c>sklearn.metrics</c>: numbers that say how good a prediction is.</summary>
        internal static class metrics
        {
            /// <summary>The module. It belongs to the process: it is not disposed.</summary>
            private static PyObject Module => PythonRuntime.Import("sklearn.metrics");

            /// <summary><c>accuracy_score(y_true, y_pred)</c> as a .NET number: the part of the rows whose predicted class is the true class.</summary>
            /// <param name="y_true">The true class of each row.</param>
            /// <param name="y_pred">The predicted class of each row.</param>
            /// <returns>A number between 0 and 1.</returns>
            /// <exception cref="PythonInteropException">scikit-learn is not installed in the embedded Python.</exception>
            /// <exception cref="PythonException">The two arrays have different lengths.</exception>
            internal static double accuracy_score_t(PyObject y_true, PyObject y_pred)
            {
                using PyObject function = Module.GetAttr("accuracy_score");
                using PyObject score = function.Invoke(y_true, y_pred);
                return score.As<double>();
            }

            /// <summary>
            ///     <c>roc_auc_score(y_true, y_score)</c> as a .NET number: the probability that the model
            ///     gives a row of class 1 a higher score than a row of class 0. It does not depend on a
            ///     decision limit.
            /// </summary>
            /// <param name="y_true">The true class of each row, 0 or 1.</param>
            /// <param name="y_score">The score of each row for class 1, for example its predicted probability.</param>
            /// <returns>A number between 0 and 1. A model that guesses gets 0.5.</returns>
            /// <exception cref="PythonInteropException">scikit-learn is not installed in the embedded Python.</exception>
            /// <exception cref="PythonException"><paramref name="y_true"/> holds one class only, or the two arrays have different lengths.</exception>
            internal static double roc_auc_score_t(PyObject y_true, PyObject y_score)
            {
                using PyObject function = Module.GetAttr("roc_auc_score");
                using PyObject score = function.Invoke(y_true, y_score);
                return score.As<double>();
            }

            /// <summary><c>confusion_matrix(y_true, y_pred)</c>: how many rows of each true class got each predicted class.</summary>
            /// <param name="y_true">The true class of each row.</param>
            /// <param name="y_pred">The predicted class of each row.</param>
            /// <returns>
            ///     A new reference to a new <c>(classes, classes)</c> int64 array. Row <c>i</c>, column
            ///     <c>j</c> counts the rows of true class <c>i</c> that the model called class <c>j</c>. Dispose it.
            /// </returns>
            /// <exception cref="PythonInteropException">scikit-learn is not installed in the embedded Python.</exception>
            /// <exception cref="PythonException">The two arrays have different lengths.</exception>
            internal static PyObject confusion_matrix(PyObject y_true, PyObject y_pred)
            {
                using PyObject function = Module.GetAttr("confusion_matrix");
                return function.Invoke(y_true, y_pred);
            }
        }
    }
}
