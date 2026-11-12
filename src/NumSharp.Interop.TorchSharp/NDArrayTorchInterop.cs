using System;
using System.Numerics;
using System.Threading;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Interop.TorchSharp
{
    /// <summary>
    ///     Interop between NumSharp <see cref="NDArray"/> and TorchSharp tensors (.NET bindings for libtorch, the engine under
    ///     PyTorch): <see cref="AsTensor"/> hands a NumSharp buffer to libtorch, <see cref="AsNDArray"/> views a tensor's storage
    ///     as an <see cref="NDArray"/>, and both take a <see cref="TorchCodecMode"/> - Auto (a view when possible, else a copy),
    ///     View (sharing or an exception), Copy (always independent).
    /// </summary>
    /// <remarks>
    ///     <para><b>A conversion library.</b> It moves descriptions of memory across the boundary and computes nothing: no
    ///     <c>TensorEngine</c> seam, no <c>[ModuleInitializer]</c>, and referencing the package changes nothing until a verb
    ///     is called.</para>
    ///     <para><b>The verbs.</b> <c>As…</c> takes a mode and shares when it can; <c>To…</c> is its Copy form.</para>
    ///     <list type="table">
    ///         <item><term><see cref="AsTensor"/></term><description>NumSharp → a <c>torch.Tensor</c> over the array's own buffer (any non-negative-stride writeable layout; a broadcast view of writeable memory as torch's <c>expand</c>), else a copy (Auto) or an exception (View)</description></item>
    ///         <item><term><see cref="ToTensor"/></term><description>NumSharp → an independent tensor (any layout, any dtype a NumPy cast reaches)</description></item>
    ///         <item><term><see cref="AsNDArray"/></term><description>torch → an NDArray view of a CPU tensor's storage (read-only when the tensor requires grad), else a copy (Auto) or an exception (View)</description></item>
    ///         <item><term><see cref="ToNDArray"/></term><description>torch → an owning, C-contiguous NDArray copy (any device, sparse, conjugated, complex64, bfloat16)</description></item>
    ///     </list>
    ///     <para><b>Who owns what.</b> An exported view's tensor keeps an ARC reference on the NumSharp buffer, released by
    ///     libtorch's deleter when the last tensor over that storage dies, so the array may be disposed while torch still reads
    ///     it. An imported view keeps a <c>detach()</c> of the tensor (its own <c>TensorImpl</c> over the storage), disposed when
    ///     the last NumSharp view over it is, so the caller may dispose the tensor - or <c>set_</c> it to other storage - while
    ///     NumSharp still reads it. What no reference can cover: an in-place RESIZE of the tensor's storage (libtorch's
    ///     <c>resize_</c> through any API) frees the memory a view reads. PyTorch's own <c>numpy()</c> marks the storage
    ///     non-resizable for that reason; TorchSharp offers no way to.</para>
    ///     <para><b>Dtypes.</b> Ten cross bit for bit: bool, uint8, int8, int16, int32, int64, float16, float32, float64 and
    ///     complex128. NumSharp's uint16, uint32, uint64 and char have no TorchSharp <c>ScalarType</c> and cross by NumPy's NEP 50
    ///     promotion (<c>np.promote_types(dtype, np.int8)</c>: int32, int64, float64, int32), decimal as float64; an explicit
    ///     <c>dtype</c> converts through <c>np.copyto</c>'s <c>casting</c> check. torch's complex64 and bfloat16 have no NumSharp
    ///     dtype and are widened, exactly, to complex128 and float32. Text and quantized tensors are refused in every mode.</para>
    ///     <para><b>Threads.</b> TorchSharp's <c>Tensor.Dispose</c> takes no lock: a tensor disposed on another thread while a
    ///     verb reads it is the caller's race, as with any TorchSharp call. The bridge's own leases are thread-safe, and
    ///     libtorch may release an exported buffer on any thread.</para>
    /// </remarks>
    public static partial class NDArrayTorchInterop
    {
        /// <summary>The live export leases (<see cref="LiveExports"/>).</summary>
        private static int _liveExports;

        /// <summary>The live import leases (<see cref="LiveImports"/>).</summary>
        private static int _liveImports;

        /// <summary>
        ///     The number of NumSharp buffers libtorch still holds through a tensor from <see cref="AsTensor"/> (views and copies
        ///     alike: a copy is a NumSharp buffer lent the same way). Each drops when libtorch frees the storage, so a count that
        ///     stays above its baseline after every tensor was disposed is a leak; one that drops while a tensor still reads the
        ///     buffer would be a use-after-free. Tests read it as a gate.
        /// </summary>
        public static int LiveExports => Volatile.Read(ref _liveExports);

        /// <summary>
        ///     The number of torch storages NumSharp views from <see cref="AsNDArray"/> still hold. Each drops when the LAST
        ///     NumSharp view over that storage (derived slices included) is disposed or finalized.
        /// </summary>
        public static int LiveImports => Volatile.Read(ref _liveImports);

        /// <summary>Counts a new export lease.</summary>
        internal static void ExportOpened() => Interlocked.Increment(ref _liveExports);

        /// <summary>Uncounts an export lease (once, from its release).</summary>
        internal static void ExportClosed() => Interlocked.Decrement(ref _liveExports);

        /// <summary>Counts a new import lease.</summary>
        internal static void ImportOpened() => Interlocked.Increment(ref _liveImports);

        /// <summary>Uncounts an import lease (once, from its release).</summary>
        internal static void ImportClosed() => Interlocked.Decrement(ref _liveImports);

        // ===================================  the dtype maps  ===================================

        /// <summary>c10's <c>QInt8</c> ordinal; TorchSharp's <c>ScalarType</c> names it only from 0.107.0, the bridge compiles against 0.100.0.</summary>
        private const ScalarType QInt8 = (ScalarType)12;

        /// <summary>c10's <c>QUInt8</c> ordinal (named by TorchSharp from 0.107.0).</summary>
        private const ScalarType QUInt8 = (ScalarType)13;

        /// <summary>c10's <c>QInt32</c> ordinal (named by TorchSharp from 0.107.0).</summary>
        private const ScalarType QInt32 = (ScalarType)14;

        /// <summary>c10's <c>UInt16</c> ordinal (libtorch 2.3 and later); TorchSharp's <c>ScalarType</c> does not name it.</summary>
        private const ScalarType UInt16 = (ScalarType)27;

        /// <summary>c10's <c>UInt32</c> ordinal (libtorch 2.3 and later); TorchSharp's <c>ScalarType</c> does not name it.</summary>
        private const ScalarType UInt32 = (ScalarType)28;

        /// <summary>c10's <c>UInt64</c> ordinal (libtorch 2.3 and later); TorchSharp's <c>ScalarType</c> does not name it.</summary>
        private const ScalarType UInt64 = (ScalarType)29;

        /// <summary>
        ///     The torch dtype a NumSharp dtype crosses as when the caller names none: its own counterpart when torch has one bit
        ///     for bit, otherwise NumPy's NEP 50 promotion computed by NumSharp itself (<c>np.promote_types(dtype, np.int8)</c>,
        ///     the smallest dtype that holds every value of the array's dtype and of the smallest signed integer), and float64 for
        ///     decimal (the ONNX Runtime and pythonnet bridges' rule; decimal promotes to itself).
        /// </summary>
        /// <remarks>
        ///     The five that convert: uint16 → int32, uint32 → int64, char → int32 (a UTF-16 unit is an unsigned 16-bit code),
        ///     uint64 → float64 (exact up to 2^53; NEP 50's answer, since no integer holds both 0..2^64-1 and the negative int8
        ///     range), decimal → float64 (rounded to 53 bits). A converted array is always a copy. For an integer uint64 tensor
        ///     ask for it: <c>AsTensor(dtype: ScalarType.Int64)</c> (values from 2^63 wrap, as <c>astype(np.int64)</c> does) or
        ///     <c>nd.view(np.int64).AsTensor()</c> to share the same bits.
        /// </remarks>
        /// <param name="dtype">The NumSharp dtype.</param>
        /// <returns>The torch dtype <see cref="AsTensor"/> produces for it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="NotSupportedException">A text dtype (<c>U</c>, <c>S</c>, <c>StringDType</c>): torch has no strings.</exception>
        public static ScalarType ToScalarType(DType dtype)
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));
            RefuseText(dtype, nameof(ToScalarType));
            NPTypeCode code = dtype.GetTypeCode();
            if (TryCounterpart(code, out ScalarType direct))
                return direct;
            return Counterpart(DefaultTarget(code), nameof(ToScalarType));
        }

        /// <summary>
        ///     The NumSharp dtype a torch dtype imports as: its own counterpart for the ten shared dtypes and for libtorch's
        ///     uint16 / uint32 / uint64 (c10 codes 27 to 29, which TorchSharp's <c>ScalarType</c> does not name), and the exact
        ///     widening for the two NumSharp lacks: complex64 → complex128, bfloat16 → float32.
        /// </summary>
        /// <param name="scalarType">The torch dtype.</param>
        /// <returns>The dtype of the NDArray <see cref="AsNDArray"/> returns.</returns>
        /// <exception cref="NotSupportedException">A quantized dtype (the caller chooses <c>dequantize()</c> or
        ///     <c>int_repr()</c>), or a dtype NumSharp cannot hold (complex32, the float8 family, the sub-byte types).</exception>
        public static DType ToDType(ScalarType scalarType) => ImportCode(scalarType, out _, nameof(ToDType));

        /// <summary>
        ///     Whether a NumSharp array and a tensor's memory MAY overlap: whether the byte ranges their elements span intersect
        ///     (NumPy's <c>np.may_share_memory</c> test). The way to tell whether <see cref="TorchCodecMode.Auto"/> viewed or
        ///     copied: a view overlaps its source, a copy never does.
        /// </summary>
        /// <remarks>
        ///     A bounds test: two interleaved views (<c>a[::2]</c> and <c>a[1::2]</c>) overlap without sharing an element. A tensor
        ///     off the CPU, or without storage (sparse, meta), shares nothing NumSharp can address and answers <c>false</c>, as
        ///     do empty arrays and tensors.
        /// </remarks>
        /// <param name="array">The NumSharp array.</param>
        /// <param name="tensor">The tensor.</param>
        /// <returns><c>true</c> when the two byte ranges intersect.</returns>
        /// <exception cref="ArgumentNullException">Either argument is null.</exception>
        /// <exception cref="ObjectDisposedException">The tensor or the array was disposed.</exception>
        public static unsafe bool MayShareMemory(this NDArray array, Tensor tensor)
        {
            if (array is null)
                throw new ArgumentNullException(nameof(array));
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));
            ThrowIfDisposed(tensor);
            if (array.size == 0 || tensor.numel() == 0 || tensor.device_type != DeviceType.CPU || tensor.is_sparse)
                return false;

            // The array's span: its first element's address plus the extremes its strides reach (a stride may be negative).
            IArraySlice slice = PinSlice(array, nameof(MayShareMemory));
            try
            {
                long itemsize = array.dtypesize;
                byte* first = (byte*)slice.Address + array.Shape.offset * itemsize;
                ByteSpan(first, array.Shape.dimensions, array.Shape.strides, itemsize, out byte* aLo, out byte* aHi);

                // The tensor's span: data_ptr() is its first element; torch strides are never negative.
                byte* tFirst = (byte*)LibTorchSharpNative.THSTensor_data(tensor.Handle);
                if (tFirst == null)
                    return false;
                ByteSpan(tFirst, tensor.shape, tensor.stride(), tensor.element_size(), out byte* tLo, out byte* tHi);
                return aLo < tHi && tLo < aHi;
            }
            finally
            {
                slice.Release();
                GC.KeepAlive(tensor);   // the native read above used the tensor's handle; keep its finalizer from racing it
                GC.KeepAlive(array);
            }
        }

        // ===================================  shared plumbing  ===================================

        /// <summary>The exact bit-for-bit counterpart of a NumSharp dtype, when torch has one.</summary>
        /// <param name="code">The NumSharp type code.</param>
        /// <param name="scalarType">The counterpart, when the method returns <c>true</c>.</param>
        /// <returns><c>true</c> for bool, uint8, int8, int16, int32, int64, float16, float32, float64 and complex128.</returns>
        internal static bool TryCounterpart(NPTypeCode code, out ScalarType scalarType)
        {
            switch (code)
            {
                case NPTypeCode.Boolean: scalarType = ScalarType.Bool; return true;
                case NPTypeCode.Byte: scalarType = ScalarType.Byte; return true;
                case NPTypeCode.SByte: scalarType = ScalarType.Int8; return true;
                case NPTypeCode.Int16: scalarType = ScalarType.Int16; return true;
                case NPTypeCode.Int32: scalarType = ScalarType.Int32; return true;
                case NPTypeCode.Int64: scalarType = ScalarType.Int64; return true;
                case NPTypeCode.Half: scalarType = ScalarType.Float16; return true;
                case NPTypeCode.Single: scalarType = ScalarType.Float32; return true;
                case NPTypeCode.Double: scalarType = ScalarType.Float64; return true;
                case NPTypeCode.Complex: scalarType = ScalarType.ComplexFloat64; return true;
                default: scalarType = default; return false;
            }
        }

        /// <summary><see cref="TryCounterpart"/> for a code that must have one (the targets the bridge itself produces).</summary>
        /// <param name="code">A NumSharp type code with a torch counterpart.</param>
        /// <param name="verb">The calling verb, for the message.</param>
        /// <returns>The counterpart.</returns>
        /// <exception cref="NotSupportedException">The code has no counterpart (an internal mapping error, not a caller's).</exception>
        internal static ScalarType Counterpart(NPTypeCode code, string verb)
            => TryCounterpart(code, out ScalarType st)
                ? st
                : throw new NotSupportedException($"{verb}: NumSharp {code} has no TorchSharp ScalarType.");

        /// <summary>
        ///     The NumSharp dtype a NumSharp dtype WITHOUT a torch counterpart converts to by default: NumPy's NEP 50 promotion
        ///     with int8 (computed by <c>np.promote_types</c>, so it follows NumSharp's promotion table), and float64 for decimal.
        /// </summary>
        /// <param name="code">A type code with no counterpart (uint16, uint32, uint64, char, decimal).</param>
        /// <returns>The code of the dtype the array converts to: one with a counterpart.</returns>
        /// <exception cref="NotSupportedException">The promotion lands on a dtype torch lacks (no NumSharp dtype does today).</exception>
        internal static NPTypeCode DefaultTarget(NPTypeCode code)
        {
            if (code == NPTypeCode.Decimal)
                return NPTypeCode.Double;   // decimal promotes to itself; the bridges' rule is float64 (rounded to 53 bits)
            NPTypeCode promoted = np.promote_types(code, NPTypeCode.SByte).GetTypeCode();
            if (TryCounterpart(promoted, out _))
                return promoted;
            throw new NotSupportedException(
                $"NumSharp {code} has no TorchSharp ScalarType, and NEP 50 promotes it to {promoted}, which has none either. " +
                "Convert it first: nd.astype(...) to a dtype torch has.");
        }

        /// <summary>
        ///     The NumSharp dtype a torch dtype imports as, and whether it is widened (no NumSharp dtype holds it as it is).
        /// </summary>
        /// <param name="scalarType">The torch dtype.</param>
        /// <param name="widened"><c>true</c> for complex64 and bfloat16, whose values cross by an exact widening copy.</param>
        /// <param name="verb">The calling verb, for the message.</param>
        /// <returns>The NumSharp dtype.</returns>
        /// <exception cref="NotSupportedException">A quantized dtype, or one NumSharp cannot hold.</exception>
        internal static DType ImportCode(ScalarType scalarType, out bool widened, string verb)
        {
            widened = false;
            switch (scalarType)
            {
                case ScalarType.Bool: return np.@bool;
                case ScalarType.Byte: return np.uint8;
                case ScalarType.Int8: return np.int8;
                case ScalarType.Int16: return np.int16;
                case ScalarType.Int32: return np.int32;
                case ScalarType.Int64: return np.int64;
                case ScalarType.Float16: return np.float16;
                case ScalarType.Float32: return np.float32;
                case ScalarType.Float64: return np.float64;
                case ScalarType.ComplexFloat64: return np.complex128;
                case UInt16: return np.uint16;
                case UInt32: return np.uint32;
                case UInt64: return np.uint64;
                case ScalarType.ComplexFloat32: widened = true; return np.complex128;
                case ScalarType.BFloat16: widened = true; return np.float32;
                case QInt8:
                case QUInt8:
                case QInt32:
                    throw new NotSupportedException(
                        $"{verb}: the tensor is quantized ({Name(scalarType)}), and NumSharp has no quantized dtype. Choose what to " +
                        "cross: t.dequantize() for its real values, or t.int_repr() for the stored integers.");
                default:
                    throw new NotSupportedException(
                        $"{verb}: NumSharp has no dtype for torch's {Name(scalarType)}. Convert it on the torch side first " +
                        "(for example t.to(ScalarType.Float32)).");
            }
        }

        /// <summary>A torch dtype's name for a message: TorchSharp's own where it has one, else c10's.</summary>
        /// <param name="scalarType">The dtype.</param>
        /// <returns>The name.</returns>
        internal static string Name(ScalarType scalarType)
        {
            switch ((int)scalarType)
            {
                case 12: return "QInt8";
                case 13: return "QUInt8";
                case 14: return "QInt32";
                case 8: return "ComplexHalf";
                case 27: return "UInt16";
                case 28: return "UInt32";
                case 29: return "UInt64";
                default: return scalarType.ToString();
            }
        }

        /// <summary>
        ///     Refuses a text dtype - fixed-width <c>U</c> / <c>S</c> or <c>StringDType</c>, NumSharp's String storage lane - with
        ///     one message naming the array's class. torch has no string tensors, so no mode can cross text.
        /// </summary>
        /// <param name="dtype">The array's dtype (never its type code: a text array's code is the storage lane).</param>
        /// <param name="verb">The calling verb, for the message.</param>
        /// <exception cref="NotSupportedException">The dtype is text.</exception>
        internal static void RefuseText(DType dtype, string verb)
        {
            if (dtype.GetTypeCode() != NPTypeCode.String && dtype.kind != 'S' && dtype.kind != 'U' && dtype.kind != 'T')
                return;
            string what = dtype.kind == 'T' ? $"a StringDType array ({dtype})" : $"a fixed-width text array ({dtype.str})";
            throw new NotSupportedException(
                $"{verb}: torch has no string tensors, so {what} cannot cross in any mode. Encode it first (for example np.unique " +
                "with return_inverse for category ids), or cross it to numpy through the pythonnet bridge.");
        }

        /// <summary>Refuses a mode outside the three members (a cast integer).</summary>
        /// <param name="mode">The caller's mode.</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not Auto, View or Copy.</exception>
        internal static void ValidateMode(TorchCodecMode mode)
        {
            if ((uint)mode > (uint)TorchCodecMode.Copy)
                throw new ArgumentOutOfRangeException(nameof(mode), mode, "TorchCodecMode is Auto, View or Copy.");
        }

        /// <summary>Refuses a tensor whose native handle was released (TorchSharp's <c>Handle</c> would throw a less specific error).</summary>
        /// <param name="tensor">The tensor.</param>
        /// <exception cref="ObjectDisposedException">The tensor was disposed (or decoupled from its native handle).</exception>
        internal static void ThrowIfDisposed(Tensor tensor)
        {
            if (tensor.IsInvalid)
                throw new ObjectDisposedException(nameof(Tensor), "the TorchSharp tensor has been disposed (its native handle is released).");
        }

        /// <summary>
        ///     Takes an ARC reference on the array's buffer: the only check that the buffer has not been released, and what keeps
        ///     it alive while it is read. The caller owns the reference and gives it back with <see cref="IArraySlice.Release"/>
        ///     (or hands it to an <see cref="ExportLease"/>).
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="verb">The calling verb, for the message.</param>
        /// <returns>The array's slice, holding one new reference.</returns>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        internal static IArraySlice PinSlice(NDArray array, string verb)
        {
            IArraySlice slice = array.Storage.InternalArray;
            if (slice is null || !slice.TryAddRef())
                throw new ObjectDisposedException(nameof(NDArray), $"{verb}: the NumSharp array's buffer has already been released.");
            return slice;
        }

        /// <summary>
        ///     The byte range <c>[lo, hi)</c> the elements of a strided layout span, from its first element (negative strides
        ///     extend it downwards, as a reversed NumSharp view does).
        /// </summary>
        /// <param name="first">The first element's address.</param>
        /// <param name="dims">The dimensions (all non-zero).</param>
        /// <param name="strides">The element strides.</param>
        /// <param name="itemsize">The element size in bytes.</param>
        /// <param name="lo">The lowest byte any element occupies.</param>
        /// <param name="hi">One past the highest byte any element occupies.</param>
        internal static unsafe void ByteSpan(byte* first, long[] dims, long[] strides, long itemsize, out byte* lo, out byte* hi)
        {
            long minOffset = 0, maxOffset = 0;
            for (int i = 0; i < dims.Length; i++)
            {
                long reach = (dims[i] - 1) * strides[i];
                if (reach < 0)
                    minOffset += reach;
                else
                    maxOffset += reach;
            }
            lo = first + minOffset * itemsize;
            hi = first + (maxOffset + 1) * itemsize;
        }

        /// <summary>
        ///     The number of elements a non-negative-stride layout reaches from its first element: <c>1 + Σ (len_i - 1)·stride_i</c>,
        ///     0 for an empty layout - the length of the flat buffer a view over it addresses.
        /// </summary>
        /// <param name="dims">The dimensions.</param>
        /// <param name="strides">The element strides (none negative where its dimension exceeds 1).</param>
        /// <returns>The reach in elements.</returns>
        internal static long ReachFrom(long[] dims, long[] strides)
        {
            long reach = 1;
            for (int i = 0; i < dims.Length; i++)
            {
                if (dims[i] <= 0)
                    return 0;
                reach += (dims[i] - 1) * strides[i];
            }
            return reach;
        }

        /// <summary>
        ///     Wraps foreign memory as a NumSharp <see cref="IArraySlice"/> whose memory-block Disposer invokes
        ///     <paramref name="dispose"/> exactly once when the LAST NumSharp reference (any view sharing the block) is released -
        ///     deterministically through <see cref="NDArray.Dispose"/> or by the finalizer safety net. The primitive every bridge
        ///     leases foreign buffers with.
        /// </summary>
        /// <param name="code">The element type.</param>
        /// <param name="data">The first element's address.</param>
        /// <param name="count">The number of elements the block spans.</param>
        /// <param name="dispose">What the last release runs (a lease's release, or a no-op for a transient view).</param>
        /// <returns>The slice.</returns>
        /// <exception cref="NotSupportedException">A type code with no unmanaged element type.</exception>
        internal static unsafe IArraySlice WrapExternal(NPTypeCode code, void* data, long count, Action dispose)
        {
            switch (code)
            {
                case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>((bool*)data, count, dispose));
                case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>((byte*)data, count, dispose));
                case NPTypeCode.SByte: return new ArraySlice<sbyte>(new UnmanagedMemoryBlock<sbyte>((sbyte*)data, count, dispose));
                case NPTypeCode.Int16: return new ArraySlice<short>(new UnmanagedMemoryBlock<short>((short*)data, count, dispose));
                case NPTypeCode.UInt16: return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>((ushort*)data, count, dispose));
                case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>((int*)data, count, dispose));
                case NPTypeCode.UInt32: return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>((uint*)data, count, dispose));
                case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>((long*)data, count, dispose));
                case NPTypeCode.UInt64: return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>((ulong*)data, count, dispose));
                case NPTypeCode.Half: return new ArraySlice<Half>(new UnmanagedMemoryBlock<Half>((Half*)data, count, dispose));
                case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>((float*)data, count, dispose));
                case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>((double*)data, count, dispose));
                case NPTypeCode.Complex: return new ArraySlice<Complex>(new UnmanagedMemoryBlock<Complex>((Complex*)data, count, dispose));
                default: throw new NotSupportedException($"NumSharp {code} is not a dtype a torch tensor's memory is viewed as.");
            }
        }

        /// <summary>
        ///     An <see cref="NDArray"/> over foreign memory with the given element layout: a flat block of
        ///     <c>ReachFrom(dims, strides)</c> elements and a strided view of it.
        /// </summary>
        /// <param name="code">The element type.</param>
        /// <param name="data">The first element's address.</param>
        /// <param name="dims">The dimensions (adopted: the caller must not write them afterwards).</param>
        /// <param name="strides">The element strides, none negative (adopted).</param>
        /// <param name="readOnly">Whether NumSharp must refuse writes through the view (both the base and the view carry it).</param>
        /// <param name="dispose">What the last NumSharp release runs.</param>
        /// <returns>A view that holds one reference on the new block.</returns>
        internal static unsafe NDArray ViewOver(NPTypeCode code, void* data, long[] dims, long[] strides, bool readOnly, Action dispose)
        {
            long reach = ReachFrom(dims, strides);
            IArraySlice block = WrapExternal(code, data, reach, dispose);
            Shape flat = Shape.Vector(reach);
            // Shape(dims, strides, offset, bufferSize) adopts the arrays and checks nothing: the strides are torch's (never
            // negative) and reach covers every element they address, which is what makes the unchecked form safe here.
            Shape view = new Shape(dims, strides, 0, reach);
            if (readOnly)
            {
                // The base storage carries the flag too, so a view derived from this one (a slice, a reshape) stays read-only.
                flat = flat.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                view = view.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
            }
            return new NDArray(new UnmanagedStorage(block, flat).Alias(view));
        }
    }
}
