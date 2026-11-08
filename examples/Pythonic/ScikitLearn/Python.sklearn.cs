using NumSharp;
using Python.Runtime;

// The classes of the tree carry Python's module names, and those are lower case. The compiler warns that
// a lower-case type name can become a keyword of a later C# version (CS8981); the names stay as Python has them.
#pragma warning disable CS8981

namespace Pythonic;

static partial class Python
{
    /// <summary>
    ///     <c>import sklearn</c>: the scikit-learn calls of this project. Each nested class is one
    ///     scikit-learn module. Import a module with <c>using static Python.sklearn.metrics;</c>, as Python
    ///     does with <c>from sklearn.metrics import ...</c>.
    /// </summary>
    /// <remarks>
    ///     An array argument is a NumSharp array, and scikit-learn reads its memory without a copy. An array
    ///     result is a NumSharp array over the memory that scikit-learn allocated. Each member takes the GIL
    ///     itself.
    /// </remarks>
    internal static class sklearn
    {
        /// <summary><c>sklearn.datasets</c>: the tables that are installed together with scikit-learn.</summary>
        internal static class datasets
        {
            /// <summary>The module. It belongs to the process: it is not disposed.</summary>
            private static PyObject Module => PythonRuntime.Import("sklearn.datasets");

            /// <summary>
            ///     <c>X, y = load_breast_cancer(return_X_y=True)</c>: the Wisconsin breast cancer table, 569
            ///     tumors with 30 numbers each and the diagnosis of each tumor (0 malignant, 1 benign).
            /// </summary>
            /// <param name="return_X_y">
            ///     Must be <c>true</c>. Python returns another type for <c>False</c> (a <c>Bunch</c> object
            ///     that also holds the column names), and a C# method has one return type. This tree has the
            ///     pair form only.
            /// </param>
            /// <returns><c>X</c>, a <c>(569, 30)</c> float64 array, and <c>y</c>, a <c>(569,)</c> int64 array.</returns>
            /// <exception cref="NotSupportedException"><paramref name="return_X_y"/> is <c>false</c>.</exception>
            /// <exception cref="PythonInteropException">No usable Python was found, or scikit-learn is not installed in it.</exception>
            /// <exception cref="PythonException">The call raised a Python error.</exception>
            public static (NDArray X, NDArray y) load_breast_cancer(bool return_X_y)
            {
                if (!return_X_y)
                    throw new NotSupportedException("load_breast_cancer(return_X_y: false) returns a Bunch object in Python. This tree has the (X, y) form only.");

                using Py.GILState gil = PythonRuntime.GIL();
                using PyObject pair = Call(Module, "load_breast_cancer", [], ("return_X_y", true));
                using PyObject X = pair[0], y = pair[1];
                return (AsArray(X), AsArray(y));
            }
        }

        /// <summary><c>sklearn.model_selection</c>: splits of a table into the part to learn from and the part to test on.</summary>
        internal static class model_selection
        {
            /// <summary>The module. It belongs to the process: it is not disposed.</summary>
            private static PyObject Module => PythonRuntime.Import("sklearn.model_selection");

            /// <summary>
            ///     <c>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=..., random_state=..., stratify=...)</c>:
            ///     shuffles the rows of two arrays in the same order and cuts both into a training part and a
            ///     test part.
            /// </summary>
            /// <param name="X">The table, one row for each sample.</param>
            /// <param name="y">The label of each row. It has as many entries as <paramref name="X"/> has rows.</param>
            /// <param name="test_size">The part of the rows that goes to the test, between 0 and 1. <c>null</c> leaves Python's default, 0.25.</param>
            /// <param name="random_state">The seed of the shuffle. The same seed gives the same split on every run. <c>null</c> gives another split each time.</param>
            /// <param name="stratify">Labels whose proportions both parts must keep, in most cases <paramref name="y"/> itself. <c>null</c> splits without regard to the labels.</param>
            /// <returns>Four new arrays. They are copies of the rows: a write to one of them does not reach <paramref name="X"/> or <paramref name="y"/>.</returns>
            /// <exception cref="PythonInteropException">No usable Python was found, or scikit-learn is not installed in it.</exception>
            /// <exception cref="PythonException">scikit-learn refused the arguments, for example because the two arrays have different lengths.</exception>
            public static (NDArray X_train, NDArray X_test, NDArray y_train, NDArray y_test) train_test_split(
                NDArray X, NDArray y, double? test_size = null, int? random_state = null, NDArray? stratify = null)
            {
                using Py.GILState gil = PythonRuntime.GIL();
                using PyObject parts = Call(Module, "train_test_split", [X, y],
                    ("test_size", test_size), ("random_state", random_state), ("stratify", stratify));
                using PyObject X_train = parts[0], X_test = parts[1], y_train = parts[2], y_test = parts[3];
                return (AsArray(X_train), AsArray(X_test), AsArray(y_train), AsArray(y_test));
            }
        }

        /// <summary><c>sklearn.preprocessing</c>: transformers that prepare a table for a model.</summary>
        internal static class preprocessing
        {
            /// <summary>The import name of the module, for the constructors of its classes.</summary>
            private const string Name = "sklearn.preprocessing";

            /// <summary><c>sklearn.preprocessing.StandardScaler</c>: brings each column to mean 0 and standard deviation 1.</summary>
            internal sealed class StandardScaler : PythonObject
            {
                /// <summary><c>StandardScaler()</c>: a scaler that has not seen data yet.</summary>
                /// <exception cref="PythonInteropException">No usable Python was found, or scikit-learn is not installed in it.</exception>
                /// <exception cref="PythonException">The constructor raised a Python error.</exception>
                public StandardScaler() : base(Create(Name, "StandardScaler")) { }

                /// <summary>
                ///     <c>scaler.fit_transform(X)</c>: learns the mean and the deviation of each column of
                ///     <paramref name="X"/>, keeps them in the scaler, and returns <paramref name="X"/> scaled with them.
                /// </summary>
                /// <param name="X">The table to learn from, in most cases the training part only.</param>
                /// <returns>A new float64 array with the shape of <paramref name="X"/>.</returns>
                /// <exception cref="PythonException">scikit-learn refused the table, for example because it has no rows.</exception>
                /// <exception cref="ObjectDisposedException">The scaler was released.</exception>
                public NDArray fit_transform(NDArray X)
                {
                    using Py.GILState gil = PythonRuntime.GIL();
                    using PyObject result = Call(Self, "fit_transform", [X]);
                    return AsArray(result);
                }

                /// <summary>
                ///     <c>scaler.transform(X)</c>: scales <paramref name="X"/> with the numbers that
                ///     <see cref="fit_transform"/> learned. It learns nothing, so a test part does not leak into the model.
                /// </summary>
                /// <param name="X">A table with the same columns as the one the scaler learned from.</param>
                /// <returns>A new float64 array with the shape of <paramref name="X"/>.</returns>
                /// <exception cref="PythonException">The scaler has not learned yet, or <paramref name="X"/> has another number of columns.</exception>
                /// <exception cref="ObjectDisposedException">The scaler was released.</exception>
                public NDArray transform(NDArray X)
                {
                    using Py.GILState gil = PythonRuntime.GIL();
                    using PyObject result = Call(Self, "transform", [X]);
                    return AsArray(result);
                }
            }
        }

        /// <summary><c>sklearn.linear_model</c>: models that are a weighted sum of the columns.</summary>
        internal static class linear_model
        {
            /// <summary>The import name of the module, for the constructors of its classes.</summary>
            private const string Name = "sklearn.linear_model";

            /// <summary><c>sklearn.linear_model.LogisticRegression</c>: a classifier that gives each class a probability.</summary>
            internal sealed class LogisticRegression : PythonObject
            {
                /// <summary><c>LogisticRegression(max_iter=...)</c>: a model that has not learned yet.</summary>
                /// <param name="max_iter">The most steps the solver may take. <c>null</c> leaves Python's default, 100, which a table that is not scaled can exceed.</param>
                /// <exception cref="PythonInteropException">No usable Python was found, or scikit-learn is not installed in it.</exception>
                /// <exception cref="PythonException">The constructor raised a Python error.</exception>
                public LogisticRegression(int? max_iter = null) : base(Create(Name, "LogisticRegression", ("max_iter", max_iter))) { }

                /// <summary><c>model.fit(X, y)</c>: learns the weights from a table and its labels. The model keeps them.</summary>
                /// <param name="X">The training table, one row for each sample.</param>
                /// <param name="y">The class of each row.</param>
                /// <returns>This model, as Python returns it, so that a call can follow: <c>new LogisticRegression().fit(X, y).predict(X)</c>.</returns>
                /// <exception cref="PythonException">scikit-learn refused the data, for example because <paramref name="y"/> holds one class only.</exception>
                /// <exception cref="ObjectDisposedException">The model was released.</exception>
                public LogisticRegression fit(NDArray X, NDArray y)
                {
                    using Py.GILState gil = PythonRuntime.GIL();
                    using PyObject self = Call(Self, "fit", [X, y]);     // Python returns the model itself: a second reference, released here
                    return this;
                }

                /// <summary><c>model.predict(X)</c>: the class that the model gives each row.</summary>
                /// <param name="X">A table with the columns the model learned from.</param>
                /// <returns>A new array with one class for each row, in the dtype of the training labels.</returns>
                /// <exception cref="PythonException">The model has not learned yet, or <paramref name="X"/> has another number of columns.</exception>
                /// <exception cref="ObjectDisposedException">The model was released.</exception>
                public NDArray predict(NDArray X)
                {
                    using Py.GILState gil = PythonRuntime.GIL();
                    using PyObject result = Call(Self, "predict", [X]);
                    return AsArray(result);
                }

                /// <summary><c>model.predict_proba(X)</c>: the probability that the model gives each class, for each row.</summary>
                /// <param name="X">A table with the columns the model learned from.</param>
                /// <returns>A new <c>(rows, classes)</c> float64 array. The columns follow the sorted class labels, and each row sums to 1.</returns>
                /// <exception cref="PythonException">The model has not learned yet, or <paramref name="X"/> has another number of columns.</exception>
                /// <exception cref="ObjectDisposedException">The model was released.</exception>
                public NDArray predict_proba(NDArray X)
                {
                    using Py.GILState gil = PythonRuntime.GIL();
                    using PyObject result = Call(Self, "predict_proba", [X]);
                    return AsArray(result);
                }
            }
        }

        /// <summary><c>sklearn.metrics</c>: numbers that say how good a prediction is.</summary>
        internal static class metrics
        {
            /// <summary>The module. It belongs to the process: it is not disposed.</summary>
            private static PyObject Module => PythonRuntime.Import("sklearn.metrics");

            /// <summary><c>accuracy_score(y_true, y_pred)</c>: the part of the rows whose predicted class is the true class.</summary>
            /// <param name="y_true">The true class of each row.</param>
            /// <param name="y_pred">The predicted class of each row.</param>
            /// <returns>A number between 0 and 1.</returns>
            /// <exception cref="PythonInteropException">No usable Python was found, or scikit-learn is not installed in it.</exception>
            /// <exception cref="PythonException">The two arrays have different lengths.</exception>
            public static double accuracy_score(NDArray y_true, NDArray y_pred)
            {
                using Py.GILState gil = PythonRuntime.GIL();
                using PyObject score = Call(Module, "accuracy_score", [y_true, y_pred]);
                return score.As<double>();
            }

            /// <summary>
            ///     <c>roc_auc_score(y_true, y_score)</c>: the probability that the model gives a row of class
            ///     1 a higher score than a row of class 0. It does not depend on a decision limit.
            /// </summary>
            /// <param name="y_true">The true class of each row, 0 or 1.</param>
            /// <param name="y_score">The score of each row for class 1, for example its predicted probability.</param>
            /// <returns>A number between 0 and 1. A model that guesses gets 0.5.</returns>
            /// <exception cref="PythonInteropException">No usable Python was found, or scikit-learn is not installed in it.</exception>
            /// <exception cref="PythonException"><paramref name="y_true"/> holds one class only, or the two arrays have different lengths.</exception>
            public static double roc_auc_score(NDArray y_true, NDArray y_score)
            {
                using Py.GILState gil = PythonRuntime.GIL();
                using PyObject score = Call(Module, "roc_auc_score", [y_true, y_score]);
                return score.As<double>();
            }

            /// <summary><c>confusion_matrix(y_true, y_pred)</c>: how many rows of each true class got each predicted class.</summary>
            /// <param name="y_true">The true class of each row.</param>
            /// <param name="y_pred">The predicted class of each row.</param>
            /// <returns>
            ///     A new <c>(classes, classes)</c> int64 array. Row <c>i</c>, column <c>j</c> counts the rows
            ///     of true class <c>i</c> that the model called class <c>j</c>.
            /// </returns>
            /// <exception cref="PythonInteropException">No usable Python was found, or scikit-learn is not installed in it.</exception>
            /// <exception cref="PythonException">The two arrays have different lengths.</exception>
            public static NDArray confusion_matrix(NDArray y_true, NDArray y_pred)
            {
                using Py.GILState gil = PythonRuntime.GIL();
                using PyObject result = Call(Module, "confusion_matrix", [y_true, y_pred]);
                return AsArray(result);
            }
        }
    }
}
