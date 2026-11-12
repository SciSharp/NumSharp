namespace NumSharp.Interop.TorchSharp
{
    /// <summary>
    ///     How a conversion between a NumSharp <see cref="NDArray"/> and a TorchSharp tensor is materialized: as a zero-copy
    ///     view of the same memory, or as an independent copy. One enum serves both directions (NumSharp → torch and torch →
    ///     NumSharp) and every verb that takes a <c>mode</c>. The members and their meanings are those of the pythonnet
    ///     bridge's <c>NumpyCodecMode</c> and the ONNX Runtime bridge's <c>OnnxCodecMode</c>; in NumPy's and PyTorch's own
    ///     spelling they are the <c>copy</c> keyword of <c>np.asarray</c> / <c>torch.asarray</c>: Auto is <c>copy=None</c>,
    ///     View is <c>copy=False</c>, Copy is <c>copy=True</c>.
    /// </summary>
    /// <remarks>
    ///     <para><b>When a view is possible, NumSharp → torch.</b> libtorch takes foreign memory through <c>from_blob</c>, so the
    ///     array's own buffer becomes the tensor's storage when the dtype has a torch counterpart bit for bit (bool, uint8,
    ///     int8, int16, int32, int64, float16, float32, float64, complex128), every stride is non-negative (torch has no
    ///     negative strides), the data is element-aligned, and the array is writeable. A broadcast view over writeable memory
    ///     is shared too: it crosses as torch's own <c>expand</c> of the memory under it. Everything else needs a copy: a
    ///     reversed view, a read-only array (a NumPy scalar, read-only foreign memory, a broadcast of either), and a dtype
    ///     torch lacks (uint16, uint32, uint64, char and decimal cross by NumPy's NEP 50 promotion, see
    ///     <see cref="NDArrayTorchInterop.ToScalarType"/>).</para>
    ///     <para><b>When a view is possible, torch → NumSharp.</b> A CPU tensor with ordinary strided storage, a dtype NumSharp
    ///     has, and no pending conjugate or negative bit is viewed in place. A tensor on another device, a sparse tensor, a
    ///     conjugated or negated view, and a complex64 or bfloat16 tensor (widened to complex128 / float32) need a copy. A view
    ///     of a tensor that requires grad is READ-ONLY: a write through NumSharp would bypass autograd's version counter.</para>
    ///     <para><b>What no mode can repair.</b> A copy does not help text (torch has no strings), a quantized tensor (the
    ///     caller chooses <c>dequantize()</c> or <c>int_repr()</c>), a meta tensor (it holds no values), or an array or tensor
    ///     that was already disposed. Those are refused in every mode.</para>
    ///     <para><b>Writes.</b> libtorch writes into the memory it was given. With a view that is the caller's array; with a
    ///     copy it is a buffer the tensor owns alone. <see cref="NDArrayTorchInterop.MayShareMemory"/> tells which one a result
    ///     holds.</para>
    /// </remarks>
    public enum TorchCodecMode
    {
        /// <summary>
        ///     A zero-copy view when the dtype and the layout permit it, an independent copy otherwise. It never fails because
        ///     of a layout or a dtype that converts, so it is the default: the common case costs nothing, and an unusual array
        ///     or tensor still crosses with correct values.
        /// </summary>
        /// <remarks>Use <see cref="View"/> when a silent copy would be a defect (a hot loop, a buffer torch must write into).</remarks>
        Auto = 0,

        /// <summary>
        ///     Always a zero-copy view. A crossing that needs a copy throws <see cref="System.InvalidOperationException"/> and
        ///     names what prevents the view. For callers that depend on shared memory (torch writing into the array, a loop that
        ///     must not allocate) and want a loud failure otherwise.
        /// </summary>
        View = 1,

        /// <summary>
        ///     Always an independent copy: no shared memory and no lifetime coupling between the source and the result. For a
        ///     caller that goes on writing the source while the other side still reads the result.
        /// </summary>
        Copy = 2,
    }
}
