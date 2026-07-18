using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    /// <summary>
    ///     Fluent NumSharp → Python conversions (<c>using NumSharp.Interop;</c> to opt in).
    ///     Convention: <c>To…</c> without "Copy" shares memory, <c>…Copy</c> is independent.
    /// </summary>
    public static class NumSharpPythonExtensions
    {
        /// <inheritdoc cref="NDArrayInterop.ToNumpy(NDArray)"/>
        public static PyObject ToNumpy(this NDArray source) => NDArrayInterop.ToNumpy(source);

        /// <inheritdoc cref="NDArrayInterop.ToNumpy(NDArray, bool)"/>
        public static PyObject ToNumpy(this NDArray source, bool copy) => NDArrayInterop.ToNumpy(source, copy);

        /// <inheritdoc cref="NDArrayInterop.ToNumpyCopy(NDArray)"/>
        public static PyObject ToNumpyCopy(this NDArray source) => NDArrayInterop.ToNumpyCopy(source);

        /// <summary>
        ///     Alias of <see cref="ToNumpy(NDArray)"/> matching pythonnet's <c>ToPython()</c> naming.
        ///     More specific than pythonnet's <c>object.ToPython()</c> extension, so it wins overload
        ///     resolution for <see cref="NDArray"/> — and unlike the untyped one it produces a numpy
        ///     array even without <see cref="NDArrayInterop.RegisterCodec()"/>.
        /// </summary>
        /// <inheritdoc cref="NDArrayInterop.ToNumpy(NDArray)"/>
        public static PyObject ToPython(this NDArray source) => NDArrayInterop.ToNumpy(source);

        /// <inheritdoc cref="NDArrayInterop.ToMemoryView(NDArray)"/>
        public static PyObject ToMemoryView(this NDArray source) => NDArrayInterop.ToMemoryView(source);
    }

    /// <summary>
    ///     Fluent Python → NumSharp conversions (<c>using NumSharp.Interop;</c> to opt in).
    ///     Convention mirrors numpy's <c>array</c>/<c>asarray</c>: <c>To…</c> copies, <c>As…</c> shares.
    /// </summary>
    public static class PyObjectNumSharpExtensions
    {
        /// <inheritdoc cref="NDArrayInterop.ToNDArray(PyObject)"/>
        public static NDArray ToNDArray(this PyObject obj) => NDArrayInterop.ToNDArray(obj);

        /// <inheritdoc cref="NDArrayInterop.ToNDArrayView(PyObject, bool)"/>
        public static NDArray AsNDArray(this PyObject obj, bool allowReadonly = false) => NDArrayInterop.ToNDArrayView(obj, allowReadonly);
    }
}
