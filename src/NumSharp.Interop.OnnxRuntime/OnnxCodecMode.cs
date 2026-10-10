namespace NumSharp.Interop.OnnxRuntime
{
    /// <summary>
    ///     How a conversion between a NumSharp <see cref="NDArray"/> and an ONNX Runtime tensor is materialized: as a
    ///     zero-copy view of the same memory, or as an independent copy. One enum serves both directions (NumSharp →
    ///     ORT and ORT → NumSharp) and every entry point that takes a <c>mode</c> argument. The members and their
    ///     meanings are those of the pythonnet bridge's <c>NumpyCodecMode</c>.
    /// </summary>
    /// <remarks>
    ///     <para><b>When a view is possible.</b> ONNX Runtime tensors are dense, row-major and carry no strides, and an
    ///     ONNX element type is fixed-size. So ORT can read a NumSharp buffer in place only when the array is
    ///     C-contiguous and its dtype crosses bit for bit: bool, the eight integers, float16, float32, float64, and
    ///     char (as UInt16). Everything else needs a copy: a Fortran-ordered, stepped, reversed, transposed or broadcast
    ///     view (a C-order copy), a Decimal array (a float64 conversion) and text (ORT keeps each string as a separate
    ///     native object).</para>
    ///     <para><b>What no mode can repair.</b> A copy does not help a dtype that has no ONNX element type at all
    ///     (Complex), an array whose buffer was already released, or a size the target container cannot hold. Those
    ///     are refused in every mode.</para>
    ///     <para><b>Writes.</b> ORT writes into the memory it was given. With a view that is the caller's array; with a
    ///     copy it is a temporary, and the caller's array does not change. <see cref="OrtTensor.IsView"/> and
    ///     <see cref="OrtTensor{T}.IsView"/> say which one a handle holds.</para>
    /// </remarks>
    public enum OnnxCodecMode
    {
        /// <summary>
        ///     A zero-copy view when the dtype and the layout permit it, an independent copy otherwise. It never fails
        ///     because of a layout or a convertible dtype, so it is the default: the common C-contiguous case costs
        ///     nothing, and an unusual array still crosses with correct values.
        /// </summary>
        /// <remarks>Use <see cref="View"/> when a silent copy would be a defect (a hot loop, an output buffer).</remarks>
        Auto = 0,

        /// <summary>
        ///     Always a zero-copy view. A conversion that needs a copy throws <see cref="System.InvalidOperationException"/>
        ///     and names what prevents the view. For callers that depend on shared memory (ORT writing into the array,
        ///     a loop that must not allocate) and want a loud failure otherwise.
        /// </summary>
        View = 1,

        /// <summary>
        ///     Always an independent copy: no shared memory and no lifetime coupling between the caller's array and the
        ///     tensor. For a caller that goes on writing the array while ORT still reads the tensor.
        /// </summary>
        Copy = 2,
    }
}
