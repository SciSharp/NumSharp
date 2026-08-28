using System;
using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    /// <summary>
    ///     Optional PyTorch bridge built on PyTorch's official numpy interchange APIs. The assembly has
    ///     no compile-time PyTorch dependency; <c>torch</c> is imported only when one of these methods is
    ///     called, and a missing installation surfaces as pythonnet's normal <see cref="PythonException"/>.
    /// </summary>
    public static class TorchInterop
    {
        /// <summary>
        ///     Convert a NumSharp array to a CPU <c>torch.Tensor</c>. The default is a zero-copy tensor
        ///     created by <c>torch.from_numpy(source.ToNumpy())</c>, so mutations are visible in both
        ///     directions and the NumSharp buffer remains rooted for the tensor's lifetime.
        ///     <see cref="NPTypeCode.Decimal"/> is the documented exception: neither NumPy nor PyTorch
        ///     has that dtype, so it converts to an independent float64 tensor.
        /// </summary>
        /// <param name="source">The NumSharp array to expose to PyTorch.</param>
        /// <param name="copy">
        ///     <c>false</c> (default) requires a safely shareable layout. <c>true</c> first materializes
        ///     an independent C-contiguous numpy array, which also supports negative-stride and
        ///     non-writeable NumSharp sources and detaches subsequent tensor writes from NumSharp.
        /// </param>
        /// <param name="requireGIL">
        ///     GIL policy, exactly as on <see cref="NDArrayPythonInterop.ToNumpy(NDArray, bool?)"/>.
        /// </param>
        /// <exception cref="InvalidOperationException">
        ///     A shared conversion was requested for a non-writeable array. PyTorch documents writes
        ///     through a tensor created from a read-only numpy array as undefined behavior.
        /// </exception>
        /// <exception cref="NotSupportedException">
        ///     A shared conversion was requested for a negative-stride layout, which PyTorch tensors do
        ///     not support. Use <paramref name="copy"/> to materialize it.
        /// </exception>
        public static PyObject ToTorch(this NDArray source, bool copy = false, bool? requireGIL = null)
        {
            if (source is null) throw new ArgumentNullException(nameof(source));

            if (!copy)
            {
                if (!source.Shape.IsWriteable)
                    throw new InvalidOperationException(
                        "PyTorch cannot safely share a non-writeable NumSharp array: torch.from_numpy warns that " +
                        "writing through a tensor over read-only numpy memory is undefined behavior. Pass copy:true.");

                long[] strides = source.Shape.Strides;
                for (int i = 0; i < strides.Length; i++)
                    if (strides[i] < 0)
                        throw new NotSupportedException(
                            "PyTorch tensors do not support negative strides. Pass copy:true to materialize a " +
                            "C-contiguous tensor with the same logical values.");
            }

            PythonRuntimeInterop.EnsureEngine();
            PythonRuntimeInterop.DrainPending();
            using (NDArrayPythonInterop.AcquireGil(requireGIL))
            using (PyObject array = source.ToNumpy(copy, requireGIL: false))
                return PythonRuntimeInterop.TorchFromNumpy.Invoke(array);
        }

        /// <summary>
        ///     Take a zero-copy NumSharp view of a CPU <c>torch.Tensor</c> through
        ///     <c>tensor.numpy()</c>. Mutations are visible in both directions, positive non-contiguous
        ///     strides are preserved, and the tensor storage remains alive until the last NumSharp view
        ///     is released.
        /// </summary>
        /// <remarks>
        ///     PyTorch permits its sharing <c>numpy()</c> conversion only for CPU tensors that do not
        ///     require gradients and have no unresolved conjugate/negative bit. Use
        ///     <see cref="ToTorchNDArray(PyObject, bool, bool?)"/> with <c>force:true</c> when an
        ///     independent CPU copy is acceptable instead.
        /// </remarks>
        public static NDArray AsTorchNDArray(this PyObject tensor, bool? requireGIL = null)
        {
            if (tensor is null) throw new ArgumentNullException(nameof(tensor));
            PythonRuntimeInterop.EnsureEngine();
            PythonRuntimeInterop.DrainPending();

            using (NDArrayPythonInterop.AcquireGil(requireGIL))
            using (PyObject array = tensor.InvokeMethod("numpy"))
                return NDArrayPythonInterop.ToNDArrayView(array, allowReadonly: false, requireGIL: false);
        }

        /// <summary>
        ///     Copy a <c>torch.Tensor</c> into an independent C-contiguous NumSharp array. With
        ///     <paramref name="force"/> set, this performs PyTorch's documented
        ///     <c>detach().cpu().resolve_conj().resolve_neg().numpy()</c> sequence first, so CUDA/MPS,
        ///     gradient-tracking, conjugated and negative-bit tensors can cross through a CPU copy.
        /// </summary>
        /// <param name="tensor">The PyTorch tensor object.</param>
        /// <param name="force">
        ///     Match <c>Tensor.numpy(force=True)</c> before copying to NumSharp. This may transfer device
        ///     data to CPU and always detaches autograd history.
        /// </param>
        /// <param name="requireGIL">
        ///     GIL policy, exactly as on <see cref="NDArrayPythonInterop.ToNDArray(PyObject, bool?)"/>.
        /// </param>
        public static NDArray ToTorchNDArray(this PyObject tensor, bool force = false, bool? requireGIL = null)
        {
            if (tensor is null) throw new ArgumentNullException(nameof(tensor));
            PythonRuntimeInterop.EnsureEngine();
            PythonRuntimeInterop.DrainPending();

            using (NDArrayPythonInterop.AcquireGil(requireGIL))
            using (PyObject array = force ? NumpyForce(tensor) : tensor.InvokeMethod("numpy"))
                return NDArrayPythonInterop.ToNDArray(array, requireGIL: false);
        }

        private static PyObject NumpyForce(PyObject tensor)
        {
            using PyObject detached = tensor.InvokeMethod("detach");
            using PyObject cpu = detached.InvokeMethod("cpu");
            using PyObject resolvedConjugate = cpu.InvokeMethod("resolve_conj");
            using PyObject resolvedNegative = resolvedConjugate.InvokeMethod("resolve_neg");
            return resolvedNegative.InvokeMethod("numpy");
        }
    }
}
