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
    ///     Built-in adapter from <c>torch.Tensor</c> to its official NumPy interchange object. It is
    ///     registered automatically and imports no Torch assembly or NuGet dependency.
    /// </summary>
    public sealed class TorchPythonArrayAdapter : IPythonArrayAdapter
    {
        /// <summary>The process-wide stateless adapter instance.</summary>
        public static TorchPythonArrayAdapter Instance { get; } = new TorchPythonArrayAdapter();

        private TorchPythonArrayAdapter() { }

        /// <inheritdoc/>
        public string Name => "torch.Tensor";

        /// <inheritdoc/>
        public bool CanAdapt(PyType objectType)
        {
            if (objectType is null)
                return false;

            try
            {
                // Walk the MRO so torch.nn.Parameter and user Tensor subclasses are accepted too.
                using PyObject mro = objectType.GetAttr("__mro__");
                using var types = PyTuple.AsTuple(mro);
                long count = types.Length();
                for (int i = 0; i < count; i++)
                {
                    using PyObject type = types[i];
                    using PyObject module = type.GetAttr("__module__");
                    using PyObject name = type.GetAttr("__name__");
                    if (module.As<string>() == "torch" && name.As<string>() == "Tensor")
                        return true;
                }
            }
            catch
            {
                // CanAdapt participates in pythonnet's global conversion search and must never throw.
            }

            return false;
        }

        /// <inheritdoc/>
        public PyObject Adapt(PyObject source, bool allowCopy)
        {
            if (source is null) throw new ArgumentNullException(nameof(source));
            if (!allowCopy)
                return source.InvokeMethod("numpy");

            // Tensor.numpy(force=True), spelled as its documented expansion so this compiles against
            // every supported pythonnet v3 without relying on keyword-argument Invoke overload details.
            using PyObject detached = source.InvokeMethod("detach");
            using PyObject cpu = detached.InvokeMethod("cpu");
            using PyObject resolvedConjugate = cpu.InvokeMethod("resolve_conj");
            using PyObject resolvedNegative = resolvedConjugate.InvokeMethod("resolve_neg");
            return resolvedNegative.InvokeMethod("numpy");
        }
    }

    /// <summary>Thread-safe process-wide adapter registry. Adapter objects are managed and session-neutral.</summary>
    internal static class PythonArrayAdapterRegistry
    {
        private static readonly object Gate = new object();
        private static IPythonArrayAdapter[] _adapters = { TorchPythonArrayAdapter.Instance };

        internal static bool Register(IPythonArrayAdapter adapter)
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
