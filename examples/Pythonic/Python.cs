using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Pythonic;

/// <summary>
///     The pythonic surface of this project: one static class whose nested classes follow the Python
///     module paths, so that a Python program reads the same in C#.
///     <code>
///     # Python                                                // C#
///     from sklearn.preprocessing import StandardScaler        using static Python.sklearn.preprocessing;
///
///     scaler = StandardScaler()                               var scaler = new StandardScaler();
///     X_train_scaled = scaler.fit_transform(X_train)          var X_train_scaled = scaler.fit_transform(X_train);
///     X_test_scaled = scaler.transform(X_test)                var X_test_scaled = scaler.transform(X_test);
///     </code>
///     Each library has its branch in a file of its own: <c>ScikitLearn/Python.sklearn.cs</c>.
/// </summary>
/// <remarks>
///     <para>The rules of the tree:</para>
///     <list type="bullet">
///       <item><b>A Python module is a static class, and <c>using static</c> is its import.</b> The
///         directive brings the functions and the classes of the module into the file, as
///         <c>from module import ...</c> does. Without the directive, the full path works:
///         <c>Python.sklearn.metrics.accuracy_score(y_true, y_pred)</c>.</item>
///       <item><b>A Python function is a static method with Python's names.</b> A keyword argument is a
///         parameter of the same name, so <c>test_size=0.25</c> is <c>test_size: 0.25</c>. A keyword
///         parameter that is <c>null</c> is not sent, so Python's own default applies.</item>
///       <item><b>A Python class is a C# class</b> that derives from <see cref="PythonObject"/>. Its
///         constructor is the Python constructor and its methods are the Python methods.</item>
///       <item><b>A member takes and returns .NET values</b>: NumSharp arrays, numbers, tuples, objects of
///         the tree. No <see cref="PyObject"/> leaves the tree.</item>
///       <item><b>Arrays cross inside the member, without a copy in both directions.</b> An argument is a
///         NumPy view of NumSharp's memory for the length of the call. A result is a NumSharp view of the
///         memory that NumPy allocated.</item>
///       <item><b>Every member takes the GIL itself</b>, and releases each Python object it made before it
///         returns. What the caller keeps (arrays, objects of the tree) belongs to its
///         <see cref="PythonScope"/>.</item>
///       <item><b>Only a call whose name is known at compile time is a member.</b> A tree holds the calls
///         that its programs use, and no more.</item>
///     </list>
///     <para>Inside this project the class hides pythonnet's <c>Python.Runtime</c> namespace. A file of the
///     tree reaches pythonnet through <c>using Python.Runtime;</c> above its namespace line and the short
///     names (<see cref="PyObject"/>, <c>Py</c>). A port needs neither.</para>
/// </remarks>
static partial class Python
{
    /// <summary>
    ///     Calls <c>target.name(*args, **kwargs)</c>: a function of a module, a method of an object, or a
    ///     class (which constructs an instance). This is the one place where .NET arguments become Python
    ///     objects, and where those objects are released again.
    /// </summary>
    /// <param name="target">The module or the object that has the attribute. The caller keeps its reference.</param>
    /// <param name="name">The Python name of the function, the method or the class.</param>
    /// <param name="args">The positional arguments, as .NET values (see <see cref="Argument"/>).</param>
    /// <param name="kwargs">The keyword arguments. A keyword whose value is <c>null</c> is not sent.</param>
    /// <returns>A new reference to what Python returned. The caller disposes it.</returns>
    /// <remarks>The caller holds the GIL. An array argument is exported as a view only for the length of this call, unless the Python side keeps the array: then NumSharp's memory stays alive for as long as Python holds it.</remarks>
    /// <exception cref="PythonException">The call raised a Python error. The frames of its traceback are emptied first (see <see cref="ReleaseFrames"/>), so the error does not keep the arguments alive.</exception>
    /// <exception cref="ArgumentNullException">A positional argument is <c>null</c>.</exception>
    /// <exception cref="NotSupportedException">An argument has a .NET type with no Python form in this tree.</exception>
    private static PyObject Call(PyObject target, string name, object[] args, params (string name, object? value)[] kwargs)
    {
        // What this call creates for Python (views of arrays, boxed numbers) and must release afterwards.
        var made = new List<PyObject>(args.Length + kwargs.Length);
        try
        {
            var positional = new PyObject[args.Length];
            for (int i = 0; i < args.Length; i++)
                positional[i] = Argument(args[i], made);

            using var keywords = new PyDict();
            foreach ((string keyword, object? value) in kwargs)
                if (value is not null)
                    keywords[keyword] = Argument(value, made);      // the dict takes its own reference

            return target.InvokeMethod(name, positional, keywords);
        }
        catch (PythonException error)
        {
            ReleaseFrames(error);
            throw;
        }
        finally
        {
            foreach (PyObject temporary in made)
                temporary.Dispose();
        }
    }

    /// <summary>
    ///     Empties the stack frames that the traceback of a Python error holds, with Python's own
    ///     <c>traceback.clear_frames</c>. The error keeps its type, its message and its traceback text.
    /// </summary>
    /// <param name="error">The error of a call that just failed.</param>
    /// <remarks>
    ///     The local variables of those frames are the arguments of the failed call: NumPy views of NumSharp
    ///     memory. Without this step they stay alive as long as the exception does on the Python side,
    ///     which was measured to be longer than a .NET garbage collection and a full Python collection:
    ///     pythonnet releases the objects of a collected exception in batches. The caller holds the GIL.
    /// </remarks>
    private static void ReleaseFrames(PythonException error)
    {
        if (error.Traceback is not { } traceback)
            return;

        try
        {
            using PyObject none = PythonRuntime.Import("traceback").InvokeMethod("clear_frames", traceback);
        }
        catch (PythonException)
        {
            // The release is an extra. The error of the call is what the caller must see, not a second one.
        }
    }

    /// <summary>The Python form of one .NET argument.</summary>
    /// <param name="value">
    ///     A NumSharp array (NumPy gets a view of its memory, no copy), an object of the tree (the Python
    ///     object it holds), a <see cref="bool"/>, an integer, a <see cref="double"/> or <see cref="float"/>
    ///     (a Python <c>float</c>, which is a double), or a <see cref="string"/>.
    /// </param>
    /// <param name="made">Receives the reference when this method created one. The caller disposes every entry after the call.</param>
    /// <returns>A reference for the call. It is in <paramref name="made"/>, or it belongs to <paramref name="value"/>.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="value"/> is <c>null</c>.</exception>
    /// <exception cref="NotSupportedException">The type of <paramref name="value"/> is not in the list above.</exception>
    private static PyObject Argument(object value, List<PyObject> made)
    {
        ArgumentNullException.ThrowIfNull(value);

        // An object of the tree already is a Python object. It keeps its reference: nothing to release.
        if (value is PythonObject held)
            return held.Self;

        PyObject created = value switch
        {
            NDArray array => array.ToNumpy(),                        // a view: NumPy reads NumSharp's memory
            bool flag => flag.ToPython(),
            sbyte or byte or short or ushort or int or uint or long => new PyInt(Convert.ToInt64(value)),
            ulong large => new PyInt(large),
            double number => new PyFloat(number),
            float number => new PyFloat(number),
            string text => new PyString(text),
            _ => throw new NotSupportedException($"A value of type {value.GetType()} has no Python form in this tree."),
        };
        made.Add(created);
        return created;
    }

    /// <summary>Calls the constructor of a Python class: <c>module.type(**kwargs)</c>. For the base call of a tree class.</summary>
    /// <param name="module">The import name of the module that has the class, for example <c>sklearn.preprocessing</c>.</param>
    /// <param name="type">The Python name of the class.</param>
    /// <param name="kwargs">The keyword arguments of the constructor. A keyword whose value is <c>null</c> is not sent.</param>
    /// <returns>A new reference to the instance. The <see cref="PythonObject"/> that receives it owns it.</returns>
    /// <remarks>It takes the GIL itself, because a constructor initializer cannot hold a <c>using</c>.</remarks>
    /// <exception cref="PythonInteropException">No usable Python was found, or the library is not installed in it.</exception>
    /// <exception cref="PythonException">The constructor raised a Python error.</exception>
    private static PyObject Create(string module, string type, params (string name, object? value)[] kwargs)
    {
        using Py.GILState gil = PythonRuntime.GIL();
        return Call(PythonRuntime.Import(module), type, [], kwargs);
    }

    /// <summary>
    ///     A NumPy array that Python returned, as a NumSharp array over the SAME memory. Nothing is copied:
    ///     a write on one side shows on the other, as it does between two names of one array in Python.
    /// </summary>
    /// <param name="value">The NumPy array. The caller keeps and disposes its reference.</param>
    /// <returns>
    ///     A view that keeps NumPy's memory alive until it is disposed, which the open
    ///     <see cref="PythonScope"/> does. A read-only NumPy array gives a read-only view.
    /// </returns>
    /// <remarks>
    ///     The caller holds the GIL. A member whose result NumSharp cannot view copies it with
    ///     <c>ToNDArray()</c> in place of this call: a complex64 array, for example, which NumSharp holds as
    ///     complex128.
    /// </remarks>
    /// <exception cref="NotSupportedException">The object is not an array (it has no buffer), or its dtype has no exact NumSharp dtype.</exception>
    private static NDArray AsArray(PyObject value) => value.AsNDArray(allowReadonly: true);
}
