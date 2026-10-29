using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Pythonic;

/// <summary>
///     The lifetime of one Python program inside C#. While a scope is open on a thread, it collects every
///     NumSharp array and every <see cref="PythonObject"/> that the thread creates. When it closes, it
///     releases all of them. It is the one line of a port that Python does not have:
///     <code>
///     using var scope = new PythonScope(NDScope.Open());
///     </code>
/// </summary>
/// <remarks>
///     <para><b>Why it exists.</b> Python frees an object when its last name goes away. C# has no such
///     moment for a local variable, and the garbage collector does not know how large a Python object is.
///     The scope gives the port that moment: the end of the method. After it, no NumPy array views NumSharp
///     memory, and no NumSharp array holds memory that Python allocated.</para>
///     <para><b>Two scopes in one.</b> NumSharp's own <see cref="NDScope"/> collects the arrays, the views
///     of Python memory included. This class adds the Python objects and closes both in the right order.
///     The <see cref="NDScope"/> is opened in the port's own line, so that NumSharp's analyzer sees it: a
///     method that creates arrays without a scope gets the warning NDW012.</para>
///     <para><b>What must not leave it.</b> An array or an object that the scope collected is released with
///     it. Return plain numbers from a port, or copy an array and take the copy out of the scope
///     (<see cref="NDScope.Detach(NDArray)"/>).</para>
///     <para><b>Python garbage.</b> Python frees an object that is part of a reference cycle only when its
///     collector runs, and such garbage can hold an array that the program passed in. SciPy's optimizer
///     leaves one: after <c>LogisticRegression.fit</c> it holds the training table. So when Python still
///     holds NumSharp memory after the release, the scope runs Python's collector, the youngest generation
///     first (less than a millisecond).</para>
///     <para><b>One thread.</b> A scope belongs to the thread that opened it, and scopes close innermost
///     first. A <c>using</c> declaration in a method without <c>await</c> gives both.</para>
/// </remarks>
sealed class PythonScope : IDisposable
{
    /// <summary>The innermost open scope of the current thread, or <c>null</c>.</summary>
    [ThreadStatic]
    private static PythonScope? t_current;

    /// <summary>The scope that was open on the thread when this one opened. It becomes the current one again when this one closes.</summary>
    private readonly PythonScope? _outer;

    /// <summary>NumSharp's scope, which this scope owns and closes. It collects every NDArray that the thread creates.</summary>
    private readonly NDScope _arrays;

    /// <summary>The Python objects that were created while this scope was the current one, oldest first.</summary>
    private readonly List<PythonObject> _objects = [];

    /// <summary>
    ///     How many NumSharp buffers Python held when the scope opened. More than that at the end means
    ///     that Python still holds a buffer of this scope. The count is for the whole process, so with
    ///     Python work on other threads it is a hint, which is all that <see cref="Dispose"/> needs.
    /// </summary>
    private readonly int _exportsAtOpen = NDArrayPythonInterop.LiveExports;

    /// <summary>True after <see cref="Dispose"/>, so that a second call does nothing.</summary>
    private bool _closed;

    /// <summary>Opens a scope on the calling thread. It is the current scope until it closes.</summary>
    /// <param name="arrays">
    ///     A NumSharp scope that the caller opens in the same expression: <c>NDScope.Open()</c>. This scope
    ///     owns it from here on and closes it. Do not dispose it yourself.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="arrays"/> is <c>null</c>.</exception>
    public PythonScope(NDScope arrays)
    {
        _arrays = arrays ?? throw new ArgumentNullException(nameof(arrays));
        _outer = t_current;
        t_current = this;
    }

    /// <summary>Adds a new Python object to the current scope of the calling thread. Without an open scope, the object stays on its own.</summary>
    /// <param name="created">The object that was just created.</param>
    internal static void Track(PythonObject created) => t_current?._objects.Add(created);

    /// <summary>
    ///     Closes the scope: releases its Python objects, newest first, and then its NumSharp arrays. An
    ///     array that viewed Python memory hands that memory back here; Python frees it a moment later, on a
    ///     thread-pool thread (see <see cref="NDArrayPythonInterop.LiveImports"/>). When Python garbage
    ///     still holds NumSharp memory, Python's collector runs.
    /// </summary>
    /// <exception cref="InvalidOperationException">The scope is closed on another thread, or before a scope that was opened after it.</exception>
    /// <exception cref="PythonException">A collection of Python garbage raised a Python error.</exception>
    public void Dispose()
    {
        if (_closed)
            return;

        // The current scope is per thread. To close this one from another thread, or out of order, would
        // leave that thread with a scope that is already closed.
        if (!ReferenceEquals(t_current, this))
            throw new InvalidOperationException("A PythonScope closes on the thread that opened it, innermost first.");

        _closed = true;
        t_current = _outer;

        try
        {
            if (_objects.Count > 0 && PythonRuntime.IsRunning)
            {
                using Py.GILState gil = PythonRuntime.GIL();      // one GIL for all of them: each Dispose would take it again
                for (int i = _objects.Count - 1; i >= 0; i--)
                    _objects[i].Dispose();
            }

            _objects.Clear();
        }
        finally
        {
            _arrays.Dispose();                                    // needs no GIL: NumSharp only hands the Python memory back
        }

        // Everything with a name is released. What Python holds now, it holds through garbage in a
        // reference cycle. Collect the youngest generation first: it is where such garbage is in most
        // cases, and it costs less than a millisecond. A full collection costs about 25.
        if (PythonRuntime.IsRunning)
            for (int generation = 0; generation <= 2 && NDArrayPythonInterop.LiveExports > _exportsAtOpen; generation++)
                Python.gc.collect(generation);
    }
}
