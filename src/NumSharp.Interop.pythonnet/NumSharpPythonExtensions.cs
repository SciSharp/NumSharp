using Python.Runtime;

namespace NumSharp.Interop
{
    /// <summary>
    ///     Fluent NumSharp → Python conversions (<c>using NumSharp.Interop;</c> to opt in).
    ///     Convention: <c>To…</c> without "Copy" shares memory, <c>…Copy</c> is independent.
    /// </summary>
    public static class NumSharpPythonExtensions
    {
        /// <inheritdoc cref="PythonConvert.ToNumpy(NDArray)"/>
        public static PyObject ToNumpy(this NDArray source) => PythonConvert.ToNumpy(source);

        /// <inheritdoc cref="PythonConvert.ToNumpy(NDArray, bool)"/>
        public static PyObject ToNumpy(this NDArray source, bool copy) => PythonConvert.ToNumpy(source, copy);

        /// <inheritdoc cref="PythonConvert.ToNumpyCopy(NDArray)"/>
        public static PyObject ToNumpyCopy(this NDArray source) => PythonConvert.ToNumpyCopy(source);

        /// <summary>
        ///     Alias of <see cref="ToNumpy(NDArray)"/> matching pythonnet's <c>ToPython()</c> naming.
        ///     More specific than pythonnet's <c>object.ToPython()</c> extension, so it wins overload
        ///     resolution for <see cref="NDArray"/> — and unlike the untyped one it produces a numpy
        ///     array even without <see cref="PythonConvert.RegisterCodec()"/>.
        /// </summary>
        /// <inheritdoc cref="PythonConvert.ToNumpy(NDArray)"/>
        public static PyObject ToPython(this NDArray source) => PythonConvert.ToNumpy(source);

        /// <inheritdoc cref="PythonConvert.ToMemoryView(NDArray)"/>
        public static PyObject ToMemoryView(this NDArray source) => PythonConvert.ToMemoryView(source);
    }

    /// <summary>
    ///     Fluent Python → NumSharp conversions (<c>using NumSharp.Interop;</c> to opt in).
    ///     Convention mirrors numpy's <c>array</c>/<c>asarray</c>: <c>To…</c> copies, <c>As…</c> shares.
    /// </summary>
    public static class PyObjectNumSharpExtensions
    {
        /// <inheritdoc cref="PythonConvert.ToNDArray(PyObject)"/>
        public static NDArray ToNDArray(this PyObject obj) => PythonConvert.ToNDArray(obj);

        /// <inheritdoc cref="PythonConvert.ToNDArrayView(PyObject, bool)"/>
        public static NDArray AsNDArray(this PyObject obj, bool allowReadonly = false) => PythonConvert.ToNDArrayView(obj, allowReadonly);
    }
}
