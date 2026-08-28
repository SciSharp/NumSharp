using System;
using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
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
}
