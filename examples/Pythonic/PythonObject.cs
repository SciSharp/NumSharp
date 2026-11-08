using Python.Runtime;

namespace Pythonic;

/// <summary>
///     A Python object that C# holds. Each Python class of the <see cref="Python"/> tree is a C# class
///     that derives from this one, so that the object is used in C# as it is used in Python:
///     <c>var scaler = new StandardScaler(); scaler.fit_transform(X);</c>
/// </summary>
/// <remarks>
///     <para><b>Lifetime.</b> The object owns one Python reference. A <see cref="PythonScope"/> that is open
///     on the thread releases it when the scope closes, which is why a port needs no <c>using</c> for each
///     object. Without a scope, call <see cref="Dispose"/>; an object that is never disposed is released by
///     pythonnet's finalizer at some later garbage collection.</para>
///     <para><b>The GIL.</b> A derived class takes the GIL at the start of each member
///     (<see cref="PythonRuntime.GIL"/>) and uses <see cref="Self"/> only while it holds it.</para>
/// </remarks>
abstract class PythonObject : IDisposable
{
    /// <summary>The Python reference that this object owns, or <c>null</c> after <see cref="Dispose"/>.</summary>
    private PyObject? _self;

    /// <summary>
    ///     Holds a Python object and takes ownership of the reference. The new object joins the
    ///     <see cref="PythonScope"/> that is open on the calling thread, when there is one.
    /// </summary>
    /// <param name="self">A reference that the caller owns and gives away: after this call only <see cref="Dispose"/> releases it.</param>
    /// <exception cref="ArgumentNullException"><paramref name="self"/> is <c>null</c>.</exception>
    protected PythonObject(PyObject self)
    {
        _self = self ?? throw new ArgumentNullException(nameof(self));
        PythonScope.Track(this);
    }

    /// <summary>
    ///     The Python object, for the members of the tree: the target of a method call, or an argument of
    ///     another member. Use it only while the thread holds the GIL, and do not dispose it.
    /// </summary>
    /// <exception cref="ObjectDisposedException">The object was released, by <see cref="Dispose"/> or by its scope.</exception>
    internal PyObject Self =>
        _self ?? throw new ObjectDisposedException(GetType().Name, "The Python object was released. Its PythonScope closed, or Dispose was called.");

    /// <summary>Python's <c>str(obj)</c>, for example <c>LogisticRegression(max_iter=1000)</c>.</summary>
    /// <returns>The text that Python prints for the object. A released object gives its C# type name between angle brackets and does not throw.</returns>
    /// <exception cref="PythonException">The <c>__str__</c> of the object raised a Python error.</exception>
    public override string ToString()
    {
        // ToString must stay usable in a debugger and in a log line after the scope closed.
        if (_self is null || !PythonRuntime.IsRunning)
            return $"<released {GetType().Name}>";

        using Py.GILState gil = PythonRuntime.GIL();
        return Self.ToString() ?? "";
    }

    /// <summary>
    ///     Releases the Python reference. Python frees the object when this was its last reference. The call
    ///     does nothing the second time, and nothing after the engine shut down (the object died with it).
    /// </summary>
    public void Dispose()
    {
        // Take the reference out first: a second call, or a call from another thread, finds nothing to release.
        PyObject? self = Interlocked.Exchange(ref _self, null);
        if (self is null || !PythonRuntime.IsRunning)
            return;

        using Py.GILState gil = PythonRuntime.GIL();          // a decref is a use of the object
        self.Dispose();
    }
}
