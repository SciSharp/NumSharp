using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Interop.TorchSharp
{
    public static partial class NDArrayTorchInterop
    {
        // ===========================  NumSharp  ->  TorchSharp  ===========================

        /// <summary>The casting rules NumPy accepts, in its own order (for the refusal of any other).</summary>
        private static readonly string[] CastingRules = { "no", "equiv", "safe", "same_kind", "unsafe" };

        /// <summary>
        ///     Hands a NumSharp array to libtorch as a CPU <see cref="Tensor"/>: by default (<see cref="TorchCodecMode.Auto"/>) a
        ///     tensor over the array's OWN buffer whenever torch can read it in place, and an independent copy otherwise.
        /// </summary>
        /// <remarks>
        ///     <para><b>When it shares.</b> No conversion (the array's dtype is one of the ten torch shares bit for bit and
        ///     <paramref name="dtype"/> names it or nothing), every stride non-negative, the data element-aligned, and the array
        ///     writeable - or a broadcast view over writeable memory, which crosses as torch's own <c>expand</c> of the memory
        ///     under it (stride 0, libtorch's overlap checks apply: <c>t.add_(1)</c> is refused, <c>t[0].fill_(7)</c> writes the
        ///     one row the broadcast repeats). A C-contiguous array becomes <c>from_blob</c>; any other layout <c>from_blob</c>
        ///     over the elements it reaches plus <c>as_strided</c>. Writes go both ways.</para>
        ///     <para><b>When it copies (Auto, Copy).</b> A reversed view (torch has no negative strides), a read-only array (a
        ///     NumPy scalar, read-only foreign memory, a broadcast of either: torch has no read-only tensors), misaligned data,
        ///     and a conversion. The copy is a fresh NumSharp buffer filled by <c>np.copyto(…, casting)</c> and lent to libtorch
        ///     the same way, so it is freed when the tensor is, and nothing else holds it.</para>
        ///     <para><b>Dtypes.</b> With no <paramref name="dtype"/>, the five NumSharp dtypes torch lacks convert by NEP 50
        ///     (<see cref="ToScalarType"/>: uint16 → int32, uint32 → int64, uint64 → float64, char → int32, decimal → float64).
        ///     With a <paramref name="dtype"/>, any of the ten shared torch dtypes is a target, checked by NumPy's
        ///     <paramref name="casting"/> rule: <c>AsTensor(dtype: ScalarType.Int64)</c> downcasts uint64 (values from 2^63 wrap,
        ///     as <c>astype(np.int64)</c> does). bfloat16 and complex64 are not targets NumSharp can write: convert the result on
        ///     the torch side.</para>
        ///     <para><b>Lifetime.</b> The tensor holds its own reference on the NumSharp buffer, released by libtorch when the LAST
        ///     tensor over that storage dies (this one and every torch view of it: disposed, collected by a <c>DisposeScope</c>,
        ///     or finalized). The array may be disposed meanwhile; <c>nd.resize(...)</c> refuses to move a buffer torch
        ///     still reads. The tensor is registered on the thread's current <c>DisposeScope</c> like any tensor TorchSharp
        ///     makes. A device copy is the caller's next call: <c>nd.AsTensor().cuda()</c>.</para>
        /// </remarks>
        /// <param name="array">The array to cross; any layout and dtype except text.</param>
        /// <param name="mode">Auto (share when possible, else copy), View (share or throw), Copy (always independent).</param>
        /// <param name="dtype">The tensor's dtype: null for the array's own (or its NEP 50 promotion), or one of Bool, Byte, Int8,
        ///     Int16, Int32, Int64, Float16, Float32, Float64, ComplexFloat64. A dtype other than the array's own is a conversion,
        ///     so a copy.</param>
        /// <param name="casting">NumPy's casting rule for a conversion: <c>"no"</c>, <c>"equiv"</c>, <c>"safe"</c>,
        ///     <c>"same_kind"</c> (the default, as for <c>np.copyto</c>) or <c>"unsafe"</c>. Unused when nothing converts.</param>
        /// <returns>A CPU tensor with the array's shape, sharing its memory or holding a copy.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="array"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a <see cref="TorchCodecMode"/> member.</exception>
        /// <exception cref="ArgumentException"><paramref name="dtype"/> is a torch dtype NumSharp cannot write (bfloat16, complex64,
        ///     quantized), or <paramref name="casting"/> is not one of NumPy's five rules.</exception>
        /// <exception cref="NotSupportedException">The array is text (torch has no strings), or the loaded TorchSharp's native
        ///     <c>THSTensor_new</c> has a form this package does not know.</exception>
        /// <exception cref="InvalidOperationException"><paramref name="mode"/> is View and the array needs a copy; the message names
        ///     why.</exception>
        /// <exception cref="InvalidCastException">The conversion is not allowed by <paramref name="casting"/> (NumPy's message).</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        /// <exception cref="System.Runtime.InteropServices.ExternalException">libtorch refused the tensor (its own message).</exception>
        public static Tensor AsTensor(this NDArray array, TorchCodecMode mode = TorchCodecMode.Auto, ScalarType? dtype = null,
            string casting = "same_kind")
        {
            if (array is null)
                throw new ArgumentNullException(nameof(array));
            ValidateMode(mode);
            ValidateCasting(casting);
            // Text first, by the array's dtype: its type code is the String storage lane, which the maps below cannot name.
            RefuseText(array.dtype, nameof(AsTensor));
            NPTypeCode source = array.typecode;
            NPTypeCode target = dtype.HasValue
                ? ExplicitTarget(dtype.Value)
                : TryCounterpart(source, out _) ? source : DefaultTarget(source);
            LibTorchSharpNative.EnsureLoaded();

            // Pin before anything reads the buffer: TryAddRef is the only check that it has not been released (a released
            // buffer may already belong to another array), and the reference keeps it alive while the copy below reads it.
            IArraySlice pin = PinSlice(array, nameof(AsTensor));
            try
            {
                string blocker = target != source
                    ? ConversionBlocker(source, target, dtype.HasValue)
                    : ViewBlocker(array, pin);
                if (blocker is null && mode != TorchCodecMode.Copy)
                {
                    // The view's lease takes this reference over: null it so the finally below does not give it back too.
                    IArraySlice lent = pin;
                    pin = null;
                    return ExportView(array, lent);
                }
                if (mode == TorchCodecMode.View)
                    throw new InvalidOperationException($"{nameof(AsTensor)}(View): {blocker}. Auto and Copy cross it as a copy.");
                return ExportCopy(array, target, casting);
            }
            finally
            {
                pin?.Release();
                GC.KeepAlive(array);
            }
        }

        /// <summary>
        ///     Copies a NumSharp array into an independent CPU <see cref="Tensor"/>: <see cref="AsTensor"/> with
        ///     <see cref="TorchCodecMode.Copy"/>. Any layout (reversed, broadcast, transposed) and any dtype a NumPy cast reaches.
        /// </summary>
        /// <param name="array">The array to copy; any layout and dtype except text.</param>
        /// <param name="dtype">The tensor's dtype, as for <see cref="AsTensor"/> (null: the array's own, or its NEP 50 promotion).</param>
        /// <param name="casting">NumPy's casting rule for a conversion (default <c>"same_kind"</c>).</param>
        /// <returns>A CPU tensor holding a C-contiguous copy that shares nothing with the array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="array"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="dtype"/> is a torch dtype NumSharp cannot write, or
        ///     <paramref name="casting"/> is not one of NumPy's five rules.</exception>
        /// <exception cref="NotSupportedException">The array is text, or the loaded TorchSharp's <c>THSTensor_new</c> is unknown.</exception>
        /// <exception cref="InvalidCastException">The conversion is not allowed by <paramref name="casting"/>.</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        /// <exception cref="System.Runtime.InteropServices.ExternalException">libtorch refused the tensor.</exception>
        public static Tensor ToTensor(this NDArray array, ScalarType? dtype = null, string casting = "same_kind")
            => AsTensor(array, TorchCodecMode.Copy, dtype, casting);

        /// <summary>The NumSharp dtype an explicit torch <c>dtype</c> is written as: one of the ten shared dtypes.</summary>
        /// <param name="dtype">The caller's torch dtype.</param>
        /// <returns>The NumSharp type code with that layout.</returns>
        /// <exception cref="ArgumentException">A torch dtype NumSharp has no layout for.</exception>
        private static NPTypeCode ExplicitTarget(ScalarType dtype)
        {
            switch (dtype)
            {
                case ScalarType.Bool: return NPTypeCode.Boolean;
                case ScalarType.Byte: return NPTypeCode.Byte;
                case ScalarType.Int8: return NPTypeCode.SByte;
                case ScalarType.Int16: return NPTypeCode.Int16;
                case ScalarType.Int32: return NPTypeCode.Int32;
                case ScalarType.Int64: return NPTypeCode.Int64;
                case ScalarType.Float16: return NPTypeCode.Half;
                case ScalarType.Float32: return NPTypeCode.Single;
                case ScalarType.Float64: return NPTypeCode.Double;
                case ScalarType.ComplexFloat64: return NPTypeCode.Complex;
                default:
                    throw new ArgumentException(
                        $"{nameof(AsTensor)}: NumSharp has no {Name(dtype)} to convert into. Cross the array as it is and convert on the " +
                        $"torch side (for example nd.AsTensor().to(ScalarType.{Name(dtype)})).", nameof(dtype));
            }
        }

        /// <summary>Refuses a casting rule NumPy does not know, with NumPy's own message.</summary>
        /// <param name="casting">The caller's rule.</param>
        /// <exception cref="ArgumentException"><paramref name="casting"/> is not one of NumPy's five rules.</exception>
        private static void ValidateCasting(string casting)
        {
            if (Array.IndexOf(CastingRules, casting) < 0)
                throw new ArgumentException("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'", nameof(casting));
        }

        /// <summary>Why a conversion cannot be a view (View mode's message).</summary>
        /// <param name="source">The array's dtype.</param>
        /// <param name="target">The dtype it converts to.</param>
        /// <param name="explicitDtype">Whether the caller named the target.</param>
        /// <returns>The reason.</returns>
        private static string ConversionBlocker(NPTypeCode source, NPTypeCode target, bool explicitDtype)
            => explicitDtype
                ? $"the requested dtype converts the {source} array to {target}, and a converted copy is not a view"
                : $"NumSharp {source} has no TorchSharp ScalarType and crosses as {target} (NEP 50), and a converted copy is not a view";

        /// <summary>
        ///     What prevents torch from reading the array's own buffer in place, or null when nothing does (the dtype is already
        ///     known to be a shared one).
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="pin">Its pinned slice (for the data address).</param>
        /// <returns>The reason, or null.</returns>
        private static unsafe string ViewBlocker(NDArray array, IArraySlice pin)
        {
            Shape shape = array.Shape;
            long[] dims = shape.dimensions;
            long[] strides = shape.strides;
            for (int i = 0; i < dims.Length; i++)
            {
                // A dimension of length 1 (or 0) is never stepped, so its stride does not matter (it is normalized to 0).
                if (dims[i] > 1 && strides[i] < 0)
                    return "the array has a negative stride (a reversed view such as a[::-1]), and torch has no negative strides";
            }

            if (!shape.IsWriteable)
            {
                if (!shape.IsBroadcasted)
                    return "the array is read-only (a NumPy scalar, read-only foreign memory, or a view of either), and torch has no " +
                           "read-only tensors: sharing it would let torch write into it";
                // A broadcast is read-only by NumPy's rule (its lanes overlap), not because its memory is: it crosses as torch's
                // own expand when the memory under it is writeable, which the owner storage (BaseStorage, the root of every view
                // chain) records. Read through the storage, not nd.@base: that property allocates and pins a new NDArray.
                UnmanagedStorage root = array.Storage.BaseStorage;
                if (root is null || !root.Shape.IsWriteable)
                    return "the array is a broadcast of read-only memory (a NumPy scalar or read-only foreign memory), and torch has " +
                           "no read-only tensors: sharing it would let torch write into it";
            }

            // libtorch's kernels assume element alignment (c10::complex<double> is declared alignas(16), which lets the compiler
            // use aligned 16-byte loads). NumSharp's buffers are aligned; only a byte-reinterpreting view at an odd offset is not.
            long align = array.typecode == NPTypeCode.Complex ? 16 : array.dtypesize;
            byte* first = (byte*)pin.Address + shape.offset * array.dtypesize;
            if (shape.size > 0 && (long)first % align != 0)
                return $"the data is not aligned to {align} bytes (a byte-reinterpreting view at an odd offset), and libtorch's " +
                       "kernels assume element alignment";
            return null;
        }

        /// <summary>
        ///     A tensor over the array's own buffer, which the export lease holds through <paramref name="lent"/>: from_blob for a
        ///     C-contiguous array, from_blob + as_strided for any other non-negative-stride layout, and the memory under a
        ///     broadcast + <c>expand</c>.
        /// </summary>
        /// <param name="array">The array (its dtype has a counterpart; <see cref="ViewBlocker"/> found nothing).</param>
        /// <param name="lent">One ARC reference on the array's buffer, owned by this method from the call on.</param>
        /// <returns>The tensor; libtorch releases the reference when the last tensor over its storage dies.</returns>
        /// <exception cref="NotSupportedException">The loaded TorchSharp's <c>THSTensor_new</c> is unknown.</exception>
        /// <exception cref="System.Runtime.InteropServices.ExternalException">libtorch refused the tensor.</exception>
        private static unsafe Tensor ExportView(NDArray array, IArraySlice lent)
        {
            ExportLease lease;
            try
            {
                lease = new ExportLease(lent);
            }
            catch
            {
                // The lease never existed, so the reference is still this method's to give back.
                lent.Release();
                throw;
            }

            Shape shape = array.Shape;
            ScalarType scalarType = Counterpart(array.typecode, nameof(AsTensor));
            byte* first = (byte*)lent.Address + shape.offset * array.dtypesize;
            // Torch copies the sizes and strides it is given; the clones only keep NumSharp's shared arrays untouched.
            long[] dims = (long[])shape.dimensions.Clone();

            if (shape.size == 0 || (shape.IsContiguous && !shape.IsBroadcasted))
                return FromBlob(first, dims, scalarType, lease);

            // A dimension of length <= 1 is never stepped, so its stride is irrelevant to addressing; as_strided refuses a
            // negative one (a length-1 axis of a reversed view), so every such stride becomes 0.
            long[] strides = (long[])shape.strides.Clone();
            for (int i = 0; i < dims.Length; i++)
                if (dims[i] <= 1)
                    strides[i] = 0;

            if (shape.IsBroadcasted)
            {
                // The memory under the broadcast: every stride-0 axis of length > 1 collapsed to length 1. Torch's own expand then
                // repeats it - the representation torch gives its own broadcasts, with its overlap checks on in-place writes.
                long[] under = (long[])dims.Clone();
                for (int i = 0; i < under.Length; i++)
                    if (strides[i] == 0 && under[i] > 1)
                        under[i] = 1;
                Tensor compact = Strided(first, under, strides, scalarType, lease);
                try
                {
                    return compact.expand(dims);
                }
                finally
                {
                    // The expanded tensor holds the storage; this intermediate only drops its own reference.
                    compact.Dispose();
                }
            }
            return Strided(first, dims, strides, scalarType, lease);
        }

        /// <summary>
        ///     A tensor over a strided, non-negative-stride layout: from_blob over the elements it reaches from its first one, as
        ///     a flat tensor, then <c>as_strided</c> - or from_blob alone when the layout is C-contiguous.
        /// </summary>
        /// <param name="first">The first element's address.</param>
        /// <param name="dims">The dimensions.</param>
        /// <param name="strides">The element strides, none negative.</param>
        /// <param name="scalarType">The element type.</param>
        /// <param name="lease">The lease whose deleter the storage gets (used by exactly one from_blob).</param>
        /// <returns>The strided tensor.</returns>
        private static unsafe Tensor Strided(byte* first, long[] dims, long[] strides, ScalarType scalarType, ExportLease lease)
        {
            if (IsCContiguous(dims, strides))
                return FromBlob(first, dims, scalarType, lease);
            Tensor flat = FromBlob(first, new[] { ReachFrom(dims, strides) }, scalarType, lease);
            try
            {
                // as_strided checks the view against the storage (it refused an out-of-bounds layout, measured), and the reach
                // is exactly the elements the layout addresses, so a layout this bridge computed wrongly fails here, loudly.
                return flat.as_strided(dims, strides, 0L);
            }
            finally
            {
                flat.Dispose();
            }
        }

        /// <summary>Whether element strides describe the C-contiguous layout of the dimensions (a length of 0 or 1 matches any stride).</summary>
        /// <param name="dims">The dimensions.</param>
        /// <param name="strides">The element strides.</param>
        /// <returns><c>true</c> when from_blob's implied row-major strides address the same elements.</returns>
        private static bool IsCContiguous(long[] dims, long[] strides)
        {
            long expected = 1;
            for (int i = dims.Length - 1; i >= 0; i--)
            {
                if (dims[i] > 1 && strides[i] != expected)
                    return false;
                expected *= dims[i];
            }
            return true;
        }

        /// <summary>
        ///     libtorch's from_blob over NumSharp memory, with the lease's deleter: a CPU tensor of the given C-contiguous
        ///     dimensions, wrapped for TorchSharp (and so registered on the thread's current <c>DisposeScope</c>).
        /// </summary>
        /// <param name="data">The first element's address.</param>
        /// <param name="dims">The dimensions (empty for a 0-d tensor).</param>
        /// <param name="scalarType">The element type.</param>
        /// <param name="lease">The lease that owns the buffer's reference; released by libtorch's deleter, or here on failure.</param>
        /// <returns>The tensor.</returns>
        /// <exception cref="NotSupportedException">The loaded TorchSharp's <c>THSTensor_new</c> is unknown.</exception>
        /// <exception cref="System.Runtime.InteropServices.ExternalException">libtorch refused the tensor.</exception>
        private static unsafe Tensor FromBlob(void* data, long[] dims, ScalarType scalarType, ExportLease lease)
        {
            IntPtr handle;
            try
            {
                handle = LibTorchSharpNative.NewTensor(data, lease.FunctionPointer, dims, scalarType);
            }
            catch
            {
                // An unknown ABI (or a marshalling failure) never reached libtorch: the reference is still the lease's to give back.
                lease.Release();
                throw;
            }

            if (handle == IntPtr.Zero)
            {
                // libtorch refused. It validates the sizes before it takes ownership of the pointer, so normally the deleter never
                // ran; an allocation failure after that point destroys the storage during unwinding, and then it did. The lease's
                // once-guard releases the reference exactly once either way.
                lease.Release();
                torch.CheckForErrors();
                throw new InvalidOperationException($"{nameof(AsTensor)}: libtorch returned no tensor and recorded no error.");
            }

            try
            {
                return Tensor.UnsafeCreateTensor(handle);
            }
            catch
            {
                // The native tensor exists but no managed wrapper owns it: drop it, and libtorch's deleter releases the lease.
                LibTorchSharpNative.THSTensor_dispose(handle);
                throw;
            }
        }

        /// <summary>
        ///     An independent tensor: a fresh C-contiguous NumSharp buffer of the target dtype, filled by <c>np.copyto</c> (any
        ///     layout, the conversion, the casting check), then lent to libtorch like a view, so libtorch frees it with the tensor
        ///     and nothing else holds it.
        /// </summary>
        /// <param name="array">The source array (its buffer is pinned by the caller for the whole copy).</param>
        /// <param name="target">The dtype to write (one with a torch counterpart).</param>
        /// <param name="casting">NumPy's casting rule.</param>
        /// <returns>The tensor over the copy.</returns>
        /// <exception cref="InvalidCastException">The conversion is not allowed by <paramref name="casting"/>.</exception>
        private static Tensor ExportCopy(NDArray array, NPTypeCode target, string casting)
        {
            // A copy of the dimensions, so the destination's Shape owns its own array.
            var destination = new NDArray(target, new Shape((long[])array.Shape.dimensions.Clone()), fillZeros: false);
            try
            {
                // NumPy's copy: it refuses a cast the rule forbids before it writes, broadcasts, and reads any strides.
                np.copyto(destination, array, casting: casting);
                return ExportView(destination, PinSlice(destination, nameof(AsTensor)));
            }
            finally
            {
                // Drops the destination's own reference. After a successful export the lease's reference keeps the buffer until
                // libtorch frees the storage; after a failure this is the last reference, and the buffer goes back to the pool.
                destination.Dispose();
            }
        }
    }
}
