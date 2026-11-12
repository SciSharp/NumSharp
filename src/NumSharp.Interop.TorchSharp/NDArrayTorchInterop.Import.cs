using System;
using System.Collections.Generic;
using NumSharp.Backends;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Interop.TorchSharp
{
    public static partial class NDArrayTorchInterop
    {
        // ===========================  TorchSharp  ->  NumSharp  ===========================

        /// <summary>
        ///     Reads a TorchSharp tensor as a NumSharp array: by default (<see cref="TorchCodecMode.Auto"/>) a view of the tensor's
        ///     own storage whenever NumSharp can read it in place, and a copy otherwise. The <c>numpy(force=True)</c> of PyTorch,
        ///     with View as its strict <c>numpy()</c>.
        /// </summary>
        /// <remarks>
        ///     <para><b>When it shares.</b> A CPU tensor with strided storage (any strides torch made: transposed, sliced,
        ///     expanded, overlapping), a dtype NumSharp has, and no pending conjugate or negative bit is viewed in place, both ways
        ///     writeable. An expanded (stride-0) tensor becomes a read-only broadcast view, by NumPy's rule. A tensor that
        ///     REQUIRES GRAD (every <c>nn.Parameter</c>) is viewed READ-ONLY: nothing in TorchSharp lets a NumSharp write bump
        ///     autograd's version counter, so a write would corrupt a backward pass silently; NumSharp refuses it instead
        ///     (<c>t.detach().AsNDArray()</c> is a writeable view, the caller having opted out of autograd).</para>
        ///     <para><b>When it copies (Auto, Copy).</b> A tensor on another device (through <c>cpu()</c>), a sparse tensor
        ///     (through <c>to_dense()</c>), a conjugated or negated view (through <c>resolve_conj()</c> / <c>resolve_neg()</c>),
        ///     and a complex64 or bfloat16 tensor, widened exactly by NumSharp (complex64 → complex128 through the float pairs,
        ///     bfloat16 → float32 by shifting the 16 bits into the top half). Under Auto, a private intermediate (the CPU copy of a
        ///     CUDA tensor) is viewed rather than copied again; Copy always returns an owning, C-contiguous array.</para>
        ///     <para><b>Lifetime.</b> A view holds its own <c>detach()</c> of the tensor - a separate <c>TensorImpl</c> over the
        ///     same storage, outside every <c>DisposeScope</c> - released when the LAST NumSharp view over it (derived slices
        ///     included) is disposed or finalized. The caller may dispose the tensor, or <c>set_</c> it to other storage, while
        ///     NumSharp reads. What no reference covers: an in-place RESIZE of the storage (libtorch's <c>resize_</c>) frees the
        ///     memory the view reads; PyTorch's <c>numpy()</c> marks the storage non-resizable, which TorchSharp offers no way to
        ///     do. The view books the storage's bytes with NumSharp's memory accounting, so forgotten views pace a collection.</para>
        /// </remarks>
        /// <param name="tensor">The tensor to read; any device, layout and dtype except quantized and meta.</param>
        /// <param name="mode">Auto (share when possible, else copy), View (share or throw), Copy (always independent).</param>
        /// <returns>An array with the tensor's shape and values: a view of its storage, or an array that owns a copy.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a <see cref="TorchCodecMode"/> member.</exception>
        /// <exception cref="ObjectDisposedException">The tensor was disposed.</exception>
        /// <exception cref="NotSupportedException">The tensor is quantized, a meta tensor (no values), or of a dtype NumSharp
        ///     cannot hold even widened (complex32, the float8 family).</exception>
        /// <exception cref="InvalidOperationException"><paramref name="mode"/> is View and the tensor needs a copy; the message
        ///     names why.</exception>
        /// <exception cref="System.Runtime.InteropServices.ExternalException">libtorch failed (its own message).</exception>
        public static NDArray AsNDArray(this Tensor tensor, TorchCodecMode mode = TorchCodecMode.Auto)
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));
            ValidateMode(mode);
            ThrowIfDisposed(tensor);
            try
            {
                ScalarType scalarType = tensor.dtype;
                NPTypeCode code = ImportCode(scalarType, out bool widened, nameof(AsNDArray)).GetTypeCode();
                DeviceType device = tensor.device_type;
                if (device == DeviceType.META)
                    throw new NotSupportedException(
                        $"{nameof(AsNDArray)}: a meta tensor holds no values (it describes shapes and dtypes only), so nothing can " +
                        "cross in any mode.");

                bool offDevice = device != DeviceType.CPU;
                bool sparse = tensor.is_sparse;
                // A pending conjugation or negation is a flag on the tensor, not on its memory: the memory holds the values as
                // they were (measured: conj() of [1+3j] reads 3 from memory, -3 resolved), so a view would show wrong values.
                bool conj = tensor.is_conj();
                bool neg = tensor.is_neg();
                string blocker = offDevice ? $"the tensor lives on {device}, and NumSharp reads CPU memory only"
                    : sparse ? "the tensor is sparse (indices and values, not a strided buffer)"
                    : conj ? "the tensor carries a pending conjugation (conj() is lazy: its memory holds the unconjugated values)"
                    : neg ? "the tensor carries a pending negation (its memory holds the values before the negation)"
                    : widened ? $"NumSharp has no {Name(scalarType)}; it crosses widened to {code} (an exact copy)"
                    : null;

                if (blocker is null && mode != TorchCodecMode.Copy)
                    return ViewOf(tensor, code, readOnly: tensor.requires_grad);
                if (mode == TorchCodecMode.View)
                    throw new InvalidOperationException($"{nameof(AsNDArray)}(View): {blocker}. Auto and Copy cross it as a copy.");
                return CopyOf(tensor, code, widened, offDevice, mode == TorchCodecMode.Auto);
            }
            finally
            {
                // Every native call above went through the tensor's handle: keep its finalizer from releasing it mid-call.
                GC.KeepAlive(tensor);
            }
        }

        /// <summary>
        ///     Copies a TorchSharp tensor into an owning, C-contiguous NumSharp array: <see cref="AsNDArray"/> with
        ///     <see cref="TorchCodecMode.Copy"/>. Any device, sparse, conjugated or negated tensors included; complex64 and
        ///     bfloat16 widen exactly to complex128 and float32.
        /// </summary>
        /// <param name="tensor">The tensor to copy.</param>
        /// <returns>A new array that shares nothing with the tensor.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">The tensor was disposed.</exception>
        /// <exception cref="NotSupportedException">The tensor is quantized, a meta tensor, or of a dtype NumSharp cannot hold.</exception>
        /// <exception cref="System.Runtime.InteropServices.ExternalException">libtorch failed (its own message).</exception>
        public static NDArray ToNDArray(this Tensor tensor) => AsNDArray(tensor, TorchCodecMode.Copy);

        /// <summary>
        ///     A NumSharp view of a CPU, strided, resolved tensor's own storage, holding a <c>detach()</c> of it through an
        ///     <see cref="ImportLease"/>.
        /// </summary>
        /// <param name="tensor">The caller's tensor.</param>
        /// <param name="code">The NumSharp dtype of its elements (a shared one).</param>
        /// <param name="readOnly">Whether NumSharp must refuse writes (the tensor requires grad).</param>
        /// <returns>The view, or a fresh empty array for a tensor with no element (nothing to share).</returns>
        /// <exception cref="System.Runtime.InteropServices.ExternalException">libtorch failed to detach the tensor.</exception>
        /// <exception cref="InvalidOperationException">libtorch returned no detached tensor and recorded no error.</exception>
        /// <exception cref="NotSupportedException">The tensor has no addressable storage.</exception>
        private static unsafe NDArray ViewOf(Tensor tensor, NPTypeCode code, bool readOnly)
        {
            long[] dims = tensor.shape;
            if (CountOf(dims) == 0)
                return new NDArray(code, new Shape(dims), fillZeros: false);   // an empty tensor's data pointer may be null
            long[] strides = tensor.stride();

            // Our own TensorImpl over the storage: unlike an alias of the caller's TensorImpl, it keeps the memory alive through a
            // set_() that rebinds the caller's tensor, and it belongs to no DisposeScope.
            IntPtr keep = LibTorchSharpNative.THSTensor_detach(tensor.Handle);
            if (keep == IntPtr.Zero)
            {
                torch.CheckForErrors();
                throw new InvalidOperationException($"{nameof(AsNDArray)}: libtorch could not detach the tensor and recorded no error.");
            }
            return AdoptHandle(keep, code, dims, strides, readOnly);
        }

        /// <summary>
        ///     Builds the view over an owned <c>at::Tensor*</c>: the lease takes the handle (and books its storage), the view's
        ///     memory-block Disposer releases the lease after the last NumSharp view.
        /// </summary>
        /// <param name="handle">An owned <c>at::Tensor*</c> with at least one element; disposed here on failure.</param>
        /// <param name="code">The NumSharp dtype of its elements.</param>
        /// <param name="dims">Its dimensions (adopted by the view's Shape).</param>
        /// <param name="strides">Its element strides (adopted by the view's Shape).</param>
        /// <param name="readOnly">Whether NumSharp must refuse writes.</param>
        /// <returns>The view.</returns>
        /// <exception cref="System.Runtime.InteropServices.ExternalException">libtorch reported an error reading the data pointer.</exception>
        /// <exception cref="NotSupportedException">The tensor has no addressable storage.</exception>
        private static unsafe NDArray AdoptHandle(IntPtr handle, NPTypeCode code, long[] dims, long[] strides, bool readOnly)
        {
            ImportLease lease = null;
            try
            {
                IntPtr data = LibTorchSharpNative.THSTensor_data(handle);
                if (data == IntPtr.Zero)
                {
                    torch.CheckForErrors();
                    throw new NotSupportedException($"{nameof(AsNDArray)}: the tensor has no addressable storage (an opaque layout).");
                }
                lease = new ImportLease(handle, (long)LibTorchSharpNative.THSStorage_nbytes(handle));
                return ViewOver(code, (void*)data, dims, strides, readOnly, lease.Release);
            }
            catch
            {
                // Before the lease existed the handle is still this method's; after, the lease owns it (and a memory block that
                // already captured lease.Release may call it again from its finalizer: the lease's once-guard absorbs that).
                if (lease is null)
                    LibTorchSharpNative.THSTensor_dispose(handle);
                else
                    lease.Release();
                throw;
            }
        }

        /// <summary>
        ///     The copy paths: brings the values to a CPU, dense, resolved tensor (an intermediate only where one is needed), then
        ///     views that private intermediate (Auto, a dtype NumSharp has), widens it into an owning array (complex64, bfloat16),
        ///     or copies it into an owning, C-contiguous array (Copy).
        /// </summary>
        /// <param name="tensor">The caller's tensor (never written, never disposed here).</param>
        /// <param name="code">The NumSharp dtype of the result.</param>
        /// <param name="widened">Whether the elements are complex64 / bfloat16 and need NumSharp's exact widening.</param>
        /// <param name="offDevice">Whether the tensor lives off the CPU.</param>
        /// <param name="auto">Whether the caller's mode is Auto (an intermediate may then be viewed instead of copied again).</param>
        /// <returns>The array.</returns>
        private static NDArray CopyOf(Tensor tensor, NPTypeCode code, bool widened, bool offDevice, bool auto)
        {
            // Every intermediate this method makes: each is registered on the caller's DisposeScope when TorchSharp creates it,
            // so each is disposed here (which also takes it off the scope), the adopted one included - its wrapper is empty then.
            var made = new List<Tensor>(4);
            try
            {
                Tensor work = tensor;
                // cpu(), to_dense(), resolve_conj() and resolve_neg() return the SAME tensor when there is nothing to do, so each
                // runs only when its condition holds: every intermediate is then fresh memory nothing else references. cpu() and
                // the three others kept their signatures from TorchSharp 0.100.0 to 0.107.0; the `to` overloads did not.
                if (offDevice)
                    made.Add(work = work.cpu());
                if (work.is_sparse)
                    made.Add(work = work.to_dense());
                if (work.is_conj())
                    made.Add(work = work.resolve_conj());
                if (work.is_neg())
                    made.Add(work = work.resolve_neg());

                if (widened)
                    return code == NPTypeCode.Complex ? WidenComplex64(work) : WidenBFloat16(work);

                if (auto && !ReferenceEquals(work, tensor))
                {
                    // Auto: the intermediate already is a private copy. View it instead of copying it again: its handle moves to
                    // the lease (out of the DisposeScope first, then out of the managed wrapper, whose Dispose is a no-op after).
                    long[] dims = work.shape;
                    if (CountOf(dims) == 0)
                        return new NDArray(code, new Shape(dims), fillZeros: false);
                    long[] strides = work.stride();
                    work.DetachFromDisposeScope();
                    IntPtr handle = work.DecoupleFromNativeHandle();
                    // Writeable: nothing but this view references the intermediate's memory, so no autograd state can depend on it.
                    return AdoptHandle(handle, code, dims, strides, readOnly: false);
                }
                return CopyInto(work, code);
            }
            finally
            {
                foreach (Tensor t in made)
                    t.Dispose();
            }
        }

        /// <summary>An owning, C-contiguous NumSharp copy of a CPU, strided, resolved tensor (NumPy's copy reads any strides).</summary>
        /// <param name="work">The tensor; the caller keeps it alive for the call.</param>
        /// <param name="code">The NumSharp dtype of its elements.</param>
        /// <returns>The copy.</returns>
        private static NDArray CopyInto(Tensor work, NPTypeCode code)
        {
            long[] dims = work.shape;
            var result = new NDArray(code, new Shape((long[])dims.Clone()), fillZeros: false);
            try
            {
                if (CountOf(dims) > 0)
                {
                    using NDArray source = Transient(work, code, dims, work.stride());
                    np.copyto(result, source);
                }
                return result;
            }
            catch
            {
                // The result was never handed out: free its buffer now rather than leave it to the finalizer.
                result.Dispose();
                throw;
            }
            finally
            {
                GC.KeepAlive(work);   // the transient view read the tensor's memory through a raw pointer
            }
        }

        /// <summary>
        ///     complex64 → complex128, exactly: the tensor's memory is (real, imaginary) float32 pairs, viewed as a float32 array with
        ///     a trailing axis of 2, and each half is copied into the result's <c>.real</c> / <c>.imag</c> (float32 → float64 is
        ///     exact; signed zeros and NaN payload bits survive, measured).
        /// </summary>
        /// <param name="work">A CPU, strided, resolved complex64 tensor; the caller keeps it alive.</param>
        /// <returns>An owning complex128 array.</returns>
        private static NDArray WidenComplex64(Tensor work)
        {
            long[] dims = work.shape;
            var result = new NDArray(NPTypeCode.Complex, new Shape((long[])dims.Clone()), fillZeros: false);
            try
            {
                if (CountOf(dims) > 0)
                {
                    long[] strides = work.stride();
                    var pairDims = new long[dims.Length + 1];
                    var pairStrides = new long[dims.Length + 1];
                    for (int i = 0; i < dims.Length; i++)
                    {
                        pairDims[i] = dims[i];
                        pairStrides[i] = strides[i] * 2;   // one complex64 element is two float32 elements
                    }
                    pairDims[dims.Length] = 2;
                    pairStrides[dims.Length] = 1;
                    using NDArray pairs = Transient(work, NPTypeCode.Single, pairDims, pairStrides);
                    using NDArray re = pairs["..., 0"];
                    using NDArray im = pairs["..., 1"];
                    using NDArray resultRe = result.real;
                    using NDArray resultIm = result.imag;
                    np.copyto(resultRe, re);
                    np.copyto(resultIm, im);
                }
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
            finally
            {
                GC.KeepAlive(work);
            }
        }

        /// <summary>
        ///     bfloat16 → float32, exactly: a bfloat16 is the upper 16 bits of the float32 with the same value, so each pattern is
        ///     zero-extended into the result's bits (viewed as uint32) and shifted into the top half - NumSharp's own copy and
        ///     <c>left_shift</c>, no rounding anywhere (signed zeros and NaN payloads survive, measured).
        /// </summary>
        /// <param name="work">A CPU, strided, resolved bfloat16 tensor; the caller keeps it alive.</param>
        /// <returns>An owning float32 array.</returns>
        private static NDArray WidenBFloat16(Tensor work)
        {
            long[] dims = work.shape;
            var result = new NDArray(NPTypeCode.Single, new Shape((long[])dims.Clone()), fillZeros: false);
            try
            {
                if (CountOf(dims) > 0)
                {
                    using NDArray patterns = Transient(work, NPTypeCode.UInt16, dims, work.stride());
                    using NDArray bits = result.view(np.uint32);
                    np.copyto(bits, patterns);   // uint16 → uint32 is a safe cast: the pattern lands in the low half
                    // A uint32 operand keeps the shift in uint32 (a weak int literal would too, but a 0-d array is explicit), and
                    // out= writes the result's own bits in place.
                    using NDArray sixteen = NDArray.Scalar(16u);
                    np.left_shift(bits, sixteen, @out: bits);
                }
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
            finally
            {
                GC.KeepAlive(work);
            }
        }

        /// <summary>
        ///     A read-only NumSharp view of a tensor's memory that holds NOTHING alive: for a copy that runs while the caller keeps
        ///     the tensor alive (and must <c>GC.KeepAlive</c> it after the last read).
        /// </summary>
        /// <param name="work">The CPU, strided tensor.</param>
        /// <param name="code">The dtype to read its elements as.</param>
        /// <param name="dims">The dimensions to read (adopted by the view's Shape).</param>
        /// <param name="strides">The element strides to read (adopted by the view's Shape).</param>
        /// <returns>The transient view; dispose it before the tensor can go away.</returns>
        /// <exception cref="NotSupportedException">The tensor has no addressable storage.</exception>
        private static unsafe NDArray Transient(Tensor work, NPTypeCode code, long[] dims, long[] strides)
        {
            IntPtr data = LibTorchSharpNative.THSTensor_data(work.Handle);
            if (data == IntPtr.Zero)
            {
                torch.CheckForErrors();
                throw new NotSupportedException($"{nameof(AsNDArray)}: the tensor has no addressable storage (an opaque layout).");
            }
            // The no-op dispose: the block wraps memory the tensor owns, and the caller's reference to the tensor is what keeps it.
            return ViewOver(code, (void*)data, dims, strides, readOnly: true, dispose: static () => { });
        }

        /// <summary>The number of elements of a shape (1 for 0-d).</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>Their product.</returns>
        private static long CountOf(long[] dims)
        {
            long count = 1;
            foreach (long d in dims)
                count *= d;
            return count;
        }
    }
}
