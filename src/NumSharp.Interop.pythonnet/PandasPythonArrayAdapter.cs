using System;
using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    /// <summary>
    ///     Built-in adapter from Pandas <c>DataFrame</c>, <c>Series</c>, <c>Index</c> and
    ///     <c>ExtensionArray</c> objects to their official NumPy interchange object. It is registered
    ///     automatically and imports no Pandas assembly or NuGet dependency.
    /// </summary>
    public sealed class PandasPythonArrayAdapter : IPythonArrayAdapter
    {
        /// <summary>The process-wide stateless adapter instance.</summary>
        public static PandasPythonArrayAdapter Instance { get; } = new PandasPythonArrayAdapter();

        private PandasPythonArrayAdapter() { }

        /// <inheritdoc/>
        public string Name => "pandas";

        /// <inheritdoc/>
        public bool CanAdapt(PyType objectType)
        {
            if (objectType is null)
                return false;

            try
            {
                // Walk the MRO so RangeIndex, Pandas extension arrays, and user/geopandas subclasses
                // are recognized through Pandas' public base types rather than private concrete names.
                using PyObject mro = objectType.GetAttr("__mro__");
                using var types = PyTuple.AsTuple(mro);
                long count = types.Length();
                for (int i = 0; i < count; i++)
                {
                    using PyObject type = types[i];
                    using PyObject module = type.GetAttr("__module__");
                    using PyObject name = type.GetAttr("__name__");
                    string moduleName = module.As<string>();
                    string typeName = name.As<string>();
                    bool isPandasModule = moduleName == "pandas" ||
                                          moduleName.StartsWith("pandas.", StringComparison.Ordinal);
                    if (isPandasModule &&
                        (typeName == "DataFrame" || typeName == "Series" || typeName == "Index" ||
                         typeName == "ExtensionArray"))
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

            // The default is copy=False on every supported Pandas to_numpy API. The downstream
            // ToNDArray copy bridge always takes its own C-contiguous copy, so requesting another
            // eager Pandas copy here would only double-copy.
            PyObject first = source.InvokeMethod("to_numpy");
            if (allowCopy)
                return first;

            try
            {
                // Pandas explicitly documents that copy=False does NOT guarantee a view. A homogeneous
                // NumPy-backed object returns projections over stable storage, whereas mixed blocks and
                // many extension arrays materialize a fresh result on every call. Two independently
                // requested projections must overlap before the shared-memory bridge may call this a
                // view. Empty arrays are safe: there is no addressable element or mutation to share.
                using PyObject size = first.GetAttr("size");
                if (size.As<long>() == 0)
                    return first;

                using PyObject second = source.InvokeMethod("to_numpy");
                if (!np.shares_memory(first, second))
                    throw new NotSupportedException(
                        "Pandas materialized the to_numpy(copy=False) result; a stable shared-memory " +
                        "NumSharp view cannot be proven. Use ToNDArray (copy), or codec Auto/Copy mode.");

                return first;
            }
            catch
            {
                first.Dispose();
                throw;
            }
        }
    }
}
