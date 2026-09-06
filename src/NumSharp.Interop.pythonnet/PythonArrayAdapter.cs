using System;
using System.Threading;
using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    /// <summary>
    ///     Adapts a Python-library array object that does not itself expose a PEP 3118 buffer or NumPy
    ///     array interface into an object the existing <see cref="NDArrayPythonInterop"/> memory bridge
    ///     can consume. Adapters choose only the library-specific route to a canonical Python array;
    ///     pointer, shape, stride, dtype, writeability and lifetime decisions remain in the bridge.
    /// </summary>
    public interface IPythonArrayAdapter
    {
        /// <summary>A stable diagnostic/registration name. Names are unique in the process-wide registry.</summary>
        string Name { get; }

        /// <summary>
        ///     Type-level capability check used by pythonnet's decoder pipeline. Called under the GIL.
        ///     It must not inspect instance state and should return <c>false</c>, not throw, for an
        ///     unrelated Python type.
        /// </summary>
        bool CanAdapt(PyType objectType);

        /// <summary>
        ///     Return a new owned <see cref="PyObject"/> wrapper around a canonical array/buffer object,
        ///     or <c>null</c> to decline. Called under the GIL; the caller disposes the returned wrapper
        ///     after the memory bridge has taken its own lifetime reference.
        /// </summary>
        /// <param name="source">The library-specific Python array object.</param>
        /// <param name="allowCopy">
        ///     <c>false</c>: adaptation must preserve shared memory and may not detach, transfer devices,
        ///     resolve lazy value bits or otherwise materialize. <c>true</c>: those transformations are
        ///     permitted because the downstream operation is already an explicit copy.
        /// </param>
        PyObject Adapt(PyObject source, bool allowCopy);
    }

    /// <summary>
    ///     Thread-safe, process-wide registry for library-specific Python array adapters. Applications
    ///     can implement <see cref="IPythonArrayAdapter"/> and call <see cref="Register"/> once; the
    ///     ordinary <see cref="NDArrayPythonInterop"/> verbs and registered <see cref="NumpyCodec"/>
    ///     then discover the adapter automatically. Adapter objects must be managed and Python-session
    ///     neutral because registrations survive interpreter restarts.
    /// </summary>
    public static class PythonArrayAdapterRegistry
    {
        private static readonly object Gate = new object();
        private static IPythonArrayAdapter[] _adapters =
        {
            TorchPythonArrayAdapter.Instance,
            PandasPythonArrayAdapter.Instance,
        };

        /// <summary>
        ///     Register an adapter process-wide. Registration is thread-safe and idempotent by
        ///     <see cref="IPythonArrayAdapter.Name"/>; the built-in Torch and Pandas adapters are
        ///     already present.
        /// </summary>
        /// <param name="adapter">The session-neutral adapter to add.</param>
        /// <returns><c>true</c> when added; <c>false</c> when the same name was already registered.</returns>
        public static bool Register(IPythonArrayAdapter adapter)
        {
            if (adapter is null) throw new ArgumentNullException(nameof(adapter));
            if (string.IsNullOrWhiteSpace(adapter.Name))
                throw new ArgumentException("a Python array adapter must have a non-empty Name.", nameof(adapter));

            lock (Gate)
            {
                IPythonArrayAdapter[] current = Volatile.Read(ref _adapters);
                for (int i = 0; i < current.Length; i++)
                    if (string.Equals(current[i].Name, adapter.Name, StringComparison.Ordinal))
                        return false;

                var next = new IPythonArrayAdapter[current.Length + 1];
                Array.Copy(current, next, current.Length);
                next[current.Length] = adapter;
                Volatile.Write(ref _adapters, next);
                return true;
            }
        }

        internal static bool CanAdapt(PyType objectType)
        {
            IPythonArrayAdapter[] adapters = Volatile.Read(ref _adapters);
            for (int i = 0; i < adapters.Length; i++)
            {
                try
                {
                    if (adapters[i].CanAdapt(objectType))
                        return true;
                }
                catch
                {
                    // A third-party capability probe must not break pythonnet's decoder search.
                }
            }

            return false;
        }

        internal static PyObject TryAdapt(PyObject source, bool allowCopy)
        {
            using PyObject typeObject = source.GetPythonType();
            using var objectType = new PyType(typeObject);
            IPythonArrayAdapter[] adapters = Volatile.Read(ref _adapters);
            for (int i = 0; i < adapters.Length; i++)
            {
                bool canAdapt;
                try { canAdapt = adapters[i].CanAdapt(objectType); }
                catch { continue; }
                if (canAdapt)
                {
                    PyObject adapted = adapters[i].Adapt(source, allowCopy);
                    if (adapted is not null)
                        return adapted;
                }
            }

            return null;
        }
    }
}
