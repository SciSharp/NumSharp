using Python.Runtime;

namespace Pythonic;

/// <summary>
///     The pythonic surface of this project: one static class whose nested classes follow the Python
///     module paths, so that a call reads in C# as it reads in Python.
///     <code>
///     # Python                                              // C#
///     from sklearn.model_selection import train_test_split
///     parts = train_test_split(X, y, test_size=0.25)        using PyObject parts = Python.sklearn.model_selection.train_test_split(X, y, test_size: 0.25);
///     model = LogisticRegression(max_iter=1000)             using PyObject model = Python.sklearn.linear_model.LogisticRegression.New(max_iter: 1000);
///     model.fit(X, y)                                       Python.sklearn.linear_model.LogisticRegression.fit(model, X, y);
///     </code>
///     Each library has its branch in a file of its own: <c>ScikitLearn/Python.sklearn.cs</c>.
/// </summary>
/// <remarks>
///     <para>The rules of the tree:</para>
///     <list type="bullet">
///       <item><b>Always write <c>Python.</c> in front.</b> The class hides pythonnet's <c>Python.Runtime</c>
///         namespace in this project, so a file reaches pythonnet through <c>using Python.Runtime;</c> and
///         the short names (<see cref="PyObject"/>, <c>Py</c>).</item>
///       <item><b>One member for each Python call, with Python's names.</b> A module is a nested class, a
///         function is a method, a keyword argument is a parameter of the same name. A keyword parameter
///         that is <c>null</c> is not sent, so Python's own default applies.</item>
///       <item><b>The members of a Python object are a nested class that takes the object</b>, as Python
///         itself can write them: <c>LogisticRegression.fit(model, X, y)</c>. The constructor is
///         <c>New</c>.</item>
///       <item><b>A member returns a <see cref="PyObject"/> that the caller owns</b> and disposes with
///         <c>using</c>. A member whose name ends in <c>_t</c> returns a .NET value in its place
///         (<c>accuracy_score_t</c> returns a <see cref="double"/>).</item>
///       <item><b>Only calls with a name that is known at compile time are members.</b> A class that a
///         user selects at run time stays a plain <c>GetAttr</c> at the place that needs it.</item>
///       <item><b>Every member runs under the GIL</b>: call it inside a <see cref="PythonRuntime.Run(Action)"/>
///         body.</item>
///       <item><b>Arrays do not cross through the tree.</b> A NumSharp array becomes a NumPy array with
///         <c>ToNumpy()</c> (a view, no copy) and comes back with <c>ToNDArray()</c> (one copy) or
///         <c>AsNDArray()</c> (a view). No member copies elements in a loop.</item>
///     </list>
/// </remarks>
static partial class Python
{
}
