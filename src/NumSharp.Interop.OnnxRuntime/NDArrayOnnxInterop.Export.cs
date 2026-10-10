using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.OnnxRuntime
{
    public static partial class NDArrayOnnxInterop
    {
        // ===========================  NumSharp  ->  ONNX Runtime  ===========================

        /// <summary>
        ///     Give a NumSharp array to ONNX Runtime as an <see cref="OrtValue"/> tensor built over a NumSharp buffer
        ///     (<c>OrtValue.CreateTensorValueWithData</c>, so there is no per-element fill loop). <paramref name="mode"/>
        ///     decides which buffer: the array's own (a zero-copy view, ORT reads and writes the array's memory) or a
        ///     temporary copy the returned handle owns.
        ///
        ///     <para><b>Modes.</b> ORT tensors are dense row-major and carry no strides, so ORT can read the array in
        ///     place only when it is <b>C-contiguous</b> and its dtype crosses bit for bit. With
        ///     <see cref="OnnxCodecMode.Auto"/> (the default) such an array is shared, and every other array crosses as
        ///     a copy with the same values: a sliced / transposed / negative-stride / broadcast / Fortran-order view as
        ///     a C-order copy, a <see cref="NPTypeCode.Decimal"/> array as a float64 conversion. With
        ///     <see cref="OnnxCodecMode.View"/> those arrays throw, so a caller that depends on shared memory (ORT
        ///     writing into the array) fails loudly. With <see cref="OnnxCodecMode.Copy"/> every array is copied.
        ///     <see cref="OrtTensor.IsView"/> reports which one the handle holds. Contiguous views at a non-zero
        ///     offset (<c>np.split</c> / <c>np.unstack</c> children) share exactly their window. Scalars become 0-d
        ///     tensors; empty arrays become empty tensors.</para>
        ///
        ///     <para><b>Dtypes:</b> bool, the eight integers, float and double cross as themselves;
        ///     <see cref="NPTypeCode.Half"/> crosses as <see cref="TensorElementType.Float16"/> (bit-identical,
        ///     NaN payloads included); <see cref="NPTypeCode.Char"/> as <see cref="TensorElementType.UInt16"/>.
        ///     <see cref="NPTypeCode.Decimal"/> has no ONNX type and always crosses as a float64 copy.
        ///     <see cref="NPTypeCode.Complex"/> throws in every mode: a copy cannot give it an ONNX type.</para>
        ///
        ///     <para><b>Lifetime:</b> the handle takes its own atomic reference on the buffer ORT reads, so the
        ///     memory survives even if <paramref name="source"/> is disposed or collected while ORT holds the
        ///     tensor; disposing the handle disposes the <see cref="OrtValue"/>, releases the reference and frees
        ///     the temporary copy, if there is one. <b>Keep the handle alive across <c>session.Run</c></b> (a
        ///     <c>using</c> around the run is the idiom). While a view is exported,
        ///     <c>ndarray.resize(refcheck: true)</c> on the source refuses to reallocate, exactly as NumPy refuses
        ///     for a referenced buffer.</para>
        /// </summary>
        /// <remarks>
        ///     <b>Footgun: a copy absorbs ORT's writes.</b> When the handle holds a copy (<see cref="OrtTensor.IsView"/>
        ///     is <c>false</c>), a model that writes into the tensor (a pre-allocated output, an <c>OrtIoBinding</c>
        ///     output) writes the temporary, and <paramref name="source"/> does not change. Pass
        ///     <see cref="OnnxCodecMode.View"/> for a tensor ORT must write, or read the result from
        ///     <see cref="OrtTensor.Source"/>.
        /// </remarks>
        /// <param name="source">The NumSharp array to give to ORT; any layout under <see cref="OnnxCodecMode.Auto"/> and <see cref="OnnxCodecMode.Copy"/>.</param>
        /// <param name="mode">View, copy, or view-when-possible (the default); see <see cref="OnnxCodecMode"/>.</param>
        /// <returns>A disposable handle owning the <see cref="OrtValue"/> (<see cref="OrtTensor.Value"/>), the buffer reference and any temporary copy.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a member of <see cref="OnnxCodecMode"/>.</exception>
        /// <exception cref="InvalidOperationException"><paramref name="mode"/> is <see cref="OnnxCodecMode.View"/> and the array is not C-contiguous or is Decimal.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ONNX tensor element type (Complex).</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        public static OrtTensor AsOrtValue(this NDArray source, OnnxCodecMode mode = OnnxCodecMode.Auto) => AsOrtValueCore(source, ownsSource: false, mode);

        /// <summary>
        ///     <see cref="AsOrtValue"/> whose returned handle also owns <paramref name="source"/> when
        ///     <paramref name="ownsSource"/> is <c>true</c> (a temporary the caller wants disposed with the
        ///     handle). Ownership transfers only on success; on failure the caller still owns its array.
        /// </summary>
        /// <remarks>
        ///     When the array crosses as a copy and <paramref name="ownsSource"/> is <c>true</c>, the copy supersedes
        ///     the caller's temporary: <paramref name="source"/> is disposed at once and the handle owns only the copy.
        /// </remarks>
        /// <param name="source">The array to give to ORT.</param>
        /// <param name="ownsSource"><c>true</c> when <paramref name="source"/> is a temporary the handle should dispose.</param>
        /// <param name="mode">View, copy, or view-when-possible; see <see cref="OnnxCodecMode"/>.</param>
        /// <returns>A handle owning the <see cref="OrtValue"/>, the buffer reference and any temporary.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a member of <see cref="OnnxCodecMode"/>.</exception>
        /// <exception cref="InvalidOperationException"><paramref name="mode"/> is <see cref="OnnxCodecMode.View"/> and the array is not C-contiguous or is Decimal.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ONNX tensor element type (Complex).</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        internal static unsafe OrtTensor AsOrtValueCore(NDArray source, bool ownsSource, OnnxCodecMode mode)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            EnsureDefined(mode);

            // The dtype is decided from the source alone, before anything is read: Complex has no ONNX element type
            // in any mode, so it must not cost a copy first. Decimal is fed as float64 (FeedTypeCode); the low-level
            // dtype maps stay honest and still refuse Decimal, only the array-producing verbs convert.
            TensorElementType elementType = ToTensorElementType(FeedTypeCode(source.typecode));

            // The view / copy decision, and the copy itself. A copy is a TEMPORARY nobody but this export can ever
            // dispose, so the handle takes ownership of it and disposes it when the handle dies.
            NDArray fed = MaterializeForExport(source, mode, nameof(AsOrtValue), out bool isView);
            NDArray temporary = isView ? null : fed;
            try
            {
                long[] dims = LongDims(fed.Shape);
                IArraySlice slice = Pin(fed, out byte* data, out long nbytes);
                OrtValue value = null;
                try
                {
                    // Does NOT own or free the memory; the handle's ARC reference guarantees validity for as long
                    // as the OrtValue exists. Row-major, no strides — which is why the fed array is C-contiguous.
                    value = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, elementType, dims, (IntPtr)data, nbytes);
                    var handle = new OrtTensor(value, fed, slice, (IntPtr)data, ownsSource: temporary is not null || ownsSource, isView, elementType, dims);
                    if (temporary is not null && ownsSource)
                        source.Dispose();   // the caller's temporary is superseded by the copy the handle now owns
                    temporary = null;       // ownership moved to the handle
                    return handle;
                }
                catch
                {
                    value?.Dispose();
                    slice.Release();
                    throw;
                }
            }
            finally
            {
                temporary?.Dispose();   // only reached on failure — the success path nulled it
            }
        }

        /// <summary>
        ///     The view / copy decision every mode-taking export verb shares: returns the array ONNX Runtime reads, which
        ///     is <paramref name="source"/> itself for a view and a new C-contiguous array of the fed dtype for a copy.
        /// </summary>
        /// <remarks>
        ///     <para>A view is possible when the dtype crosses bit for bit (every supported dtype except Decimal) and the
        ///     array is C-contiguous. The caller has already refused a dtype with no ONNX element type, so this method
        ///     never sees Complex.</para>
        ///     <para>Every read of the caller's array (the conversion, the copy) runs under a reference on its buffer
        ///     (<see cref="ConvertPinned"/>, <see cref="CopyPinned"/>), so a released buffer is refused instead of being
        ///     copied from freed memory. A view is not read here: the caller's <see cref="Pin(NDArray)"/> checks it.</para>
        ///     <para>A Decimal view that is not C-contiguous is converted first and laid out in C order second: the second
        ///     pass then moves 8-byte elements, where a copy before the conversion would move 16-byte ones.</para>
        /// </remarks>
        /// <param name="source">The caller's array; any layout.</param>
        /// <param name="mode">The requested mode (already validated by <see cref="EnsureDefined"/>).</param>
        /// <param name="verb">The public verb, named in the refusal.</param>
        /// <param name="isView"><c>true</c> when the result is <paramref name="source"/> itself; <c>false</c> when it is a temporary the caller must dispose (or hand to a handle).</param>
        /// <returns><paramref name="source"/>, or a new C-contiguous array the caller owns.</returns>
        /// <exception cref="InvalidOperationException"><paramref name="mode"/> is <see cref="OnnxCodecMode.View"/> and the array needs a copy.</exception>
        /// <exception cref="ObjectDisposedException">A copy was needed and the array's buffer has already been released.</exception>
        internal static NDArray MaterializeForExport(NDArray source, OnnxCodecMode mode, string verb, out bool isView)
        {
            bool decimalSource = source.typecode == NPTypeCode.Decimal;
            bool viewPossible = !decimalSource && source.Shape.IsContiguous;
            if (mode == OnnxCodecMode.View && !viewPossible)
                throw new InvalidOperationException(ViewImpossibleMessage(verb, decimalSource));
            if (viewPossible && mode != OnnxCodecMode.Copy)
            {
                isView = true;
                return source;
            }

            isView = false;
            if (!decimalSource)
                return CopyPinned(source);   // a C-order copy; 0-d stays 0-d

            // Decimal has no ONNX element type (16-byte, non-IEEE): convert it to float64, the conversion this
            // package's own guidance recommends (astype(Double), lossy beyond ~16 significant digits). astype lays its
            // result out in order 'K', so a transposed source converts to a Fortran-ordered array that still needs the
            // C-order pass.
            NDArray converted = ConvertPinned(source, NPTypeCode.Double);
            if (converted.Shape.IsContiguous)
                return converted;
            try
            {
                return converted.copy();
            }
            finally
            {
                converted.Dispose();   // the 'K'-ordered intermediate, on success and on failure alike
            }
        }

        /// <summary>Refuses a <see cref="OnnxCodecMode"/> value that is not one of the three members (a cast from an integer).</summary>
        /// <param name="mode">The mode a caller passed.</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not <see cref="OnnxCodecMode.Auto"/>, <see cref="OnnxCodecMode.View"/> or <see cref="OnnxCodecMode.Copy"/>.</exception>
        internal static void EnsureDefined(OnnxCodecMode mode)
        {
            if (mode != OnnxCodecMode.Auto && mode != OnnxCodecMode.View && mode != OnnxCodecMode.Copy)
                throw new ArgumentOutOfRangeException(nameof(mode), mode, "mode must be OnnxCodecMode.Auto, OnnxCodecMode.View or OnnxCodecMode.Copy.");
        }

        /// <summary>The refusal of <see cref="OnnxCodecMode.View"/> for an array ONNX Runtime cannot read in place.</summary>
        /// <param name="verb">The verb refusing (named in the message).</param>
        /// <param name="decimalSource"><c>true</c> when the dtype (Decimal) prevents the view; <c>false</c> when the layout does.</param>
        /// <returns>The message, naming the two modes that accept the array.</returns>
        internal static string ViewImpossibleMessage(string verb, bool decimalSource) =>
            $"{verb}(mode: OnnxCodecMode.View): " +
            (decimalSource
                ? "a NumSharp Decimal array has no ONNX tensor element type, so it crosses as a float64 conversion, which is a copy."
                : "the array is not C-contiguous; ONNX Runtime tensors are dense row-major and carry no strides, so a sliced / " +
                  "transposed / negative-stride / broadcast / Fortran-order view cannot be shared.") +
            " Pass OnnxCodecMode.Auto (a copy when a view is impossible) or OnnxCodecMode.Copy.";

        /// <summary>
        ///     Copy a NumSharp array into an independent ORT-allocated tensor (no shared memory, no lifetime
        ///     coupling — dispose the <see cref="OrtValue"/> when done, as with any ORT value). Any layout: a
        ///     sliced / transposed / broadcast view is copied in its logical (C) order. Same dtype rules as
        ///     <see cref="AsOrtValue"/> (Decimal → float64; Complex throws).
        ///
        ///     <para>The difference from <c>AsOrtValue(mode: OnnxCodecMode.Copy)</c> is who owns the copy: here ORT's
        ///     allocator does and the result is a bare <see cref="OrtValue"/> with no handle to keep alive; there a
        ///     NumSharp temporary does, behind an <see cref="OrtTensor"/> handle, with no 2 GB limit.</para>
        /// </summary>
        /// <remarks>
        ///     Refusals come in the order of what they cost to decide: the dtype and the 2 GB limit first (they need
        ///     only the dtype and the shape, so nothing is converted, copied or allocated for an array that cannot
        ///     cross), then the buffer's liveness, then the copy. A tensor past 2 GB cannot go through this verb at
        ///     all — share it with <see cref="AsOrtValue"/>, which hands ORT a native pointer.
        /// </remarks>
        /// <param name="source">The array to copy; any layout.</param>
        /// <returns>A new ORT-allocated tensor holding the array's values in logical (C) order; the caller disposes it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ONNX tensor element type, or the array exceeds the 2 GB span limit of ORT's managed API.</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        public static unsafe OrtValue ToOrtValue(this NDArray source)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));

            // Decide everything the dtype and the shape alone decide BEFORE the source is read. The 2 GB check used
            // to run after the source had been converted and densified and ORT had allocated the tensor, so a large
            // strided array cost two full-size allocations just to be refused (or died with OutOfMemoryException in
            // the densifying copy, which is what a 2^40-element broadcast view did). The byte count is compared
            // without multiplying (size > int.MaxValue / itemsize ⇔ size × itemsize > int.MaxValue for positive
            // integers), because a broadcast view's element count can be large enough to overflow the product.
            NPTypeCode feedCode = FeedTypeCode(source.typecode);
            TensorElementType elementType = ToTensorElementType(feedCode);
            long size = source.Shape.Size;
            int feedItemSize = feedCode.SizeOf();
            if (size > int.MaxValue / feedItemSize)
                throw new NotSupportedException(TooLargeForSpanMessage(size, feedItemSize));

            // Take the reference on the SOURCE's buffer before reading it in any way: Pin's TryAddRef is the only
            // check that the buffer has not been released. The Decimal conversion and the densifying copy below
            // used to run first, and the Pin further down then succeeded on their fresh result, so a released buffer
            // was read silently and the tensor carried another array's values. Holding the reference across every
            // read also keeps the buffer alive while it is read. Even an empty array is checked.
            IArraySlice sourcePin = Pin(source);
            try
            {
                using NDArray converted = source.typecode == NPTypeCode.Decimal ? source.astype(NPTypeCode.Double) : null;
                NDArray logical = converted ?? source;

                // ORT tensors are dense row-major: a non-C-contiguous view is read in LOGICAL order through a C-order
                // copy (0-d stays 0-d; np.ascontiguousarray would lift a scalar to shape (1,), so copy() it is).
                using NDArray dense = logical.Shape.IsContiguous ? null : logical.copy();
                NDArray src = dense ?? logical;
                Shape shape = src.Shape;
                long[] dims = LongDims(shape);

                IArraySlice slice = Pin(src, out byte* data, out long nbytes);
                try
                {
                    OrtValue value = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, elementType, dims);
                    try
                    {
                        if (nbytes > 0)
                        {
                            // Refused up front already (same element count, same feed itemsize); kept beside the
                            // (int) cast below so the cast can never truncate whatever the up-front check becomes.
                            if (nbytes > int.MaxValue)
                                throw new NotSupportedException(TooLargeForSpanMessage(shape.Size, src.dtypesize));
                            Span<byte> dst = value.GetTensorMutableRawData();
                            new ReadOnlySpan<byte>(data, (int)nbytes).CopyTo(dst);
                        }

                        return value;
                    }
                    catch
                    {
                        value.Dispose();
                        throw;
                    }
                }
                finally
                {
                    slice.Release();
                    GC.KeepAlive(src);
                }
            }
            finally
            {
                sourcePin.Release();
                GC.KeepAlive(source);
            }
        }

        /// <summary>
        ///     Give a NumSharp array to ONNX Runtime as a row-major <see cref="DenseTensor{T}"/> over a NumSharp
        ///     buffer — the legacy tensor surface behind <c>NamedOnnxValue.CreateFromTensor</c> and the
        ///     <c>session.Run(IReadOnlyCollection&lt;NamedOnnxValue&gt;)</c> overloads. The tensor's
        ///     <see cref="DenseTensor{T}.Buffer"/> is a <see cref="Memory{T}"/> over unmanaged memory (an internal
        ///     <see cref="System.Buffers.MemoryManager{T}"/>); the returned handle holds the buffer reference and
        ///     must outlive every use of the tensor. <paramref name="mode"/> decides which buffer: the array's own (a
        ///     zero-copy view) or a temporary copy the handle owns.
        ///
        ///     <para><typeparamref name="T"/> must be the C# type ORT uses for the array's dtype
        ///     (<see cref="ToTensorElementClrType"/>): <see cref="Float16"/> for <see cref="NPTypeCode.Half"/>,
        ///     <see cref="ushort"/> for <see cref="NPTypeCode.Char"/>, <see cref="double"/> for
        ///     <see cref="NPTypeCode.Decimal"/>, otherwise the dtype's own type — a mismatch throws rather than
        ///     silently reinterpreting the bytes.</para>
        ///
        ///     <para><b>Modes.</b> As for <see cref="AsOrtValue"/>: under <see cref="OnnxCodecMode.Auto"/> (the default)
        ///     a C-contiguous array of a bit-exact dtype is shared, and every other array (a sliced / transposed /
        ///     negative-stride / broadcast / Fortran-order view, a Decimal array) crosses as a C-order copy;
        ///     <see cref="OnnxCodecMode.View"/> throws for those; <see cref="OnnxCodecMode.Copy"/> always copies.
        ///     <see cref="OrtTensor{T}.IsView"/> reports which one the handle holds.</para>
        /// </summary>
        /// <remarks>
        ///     <para>Refusals come in the order of what they cost to decide: the element type, the element count and the
        ///     <see cref="int"/> dimensions first (they need only the dtype and the shape, so an array that cannot cross
        ///     is neither converted nor copied), then the mode against the layout, then the buffer's liveness.</para>
        ///     <para><b>The tensor is always row-major.</b> A Fortran-contiguous array used to be shared as a column-major
        ///     tensor (<c>reverseStride: true</c>). No ONNX Runtime route reads such a tensor correctly: a run that
        ///     receives it through <c>NamedOnnxValue.CreateFromTensor</c> throws, and
        ///     <c>OrtValue.CreateTensorValueFromMemory</c> over its <see cref="Memory{T}"/> reads the buffer in memory
        ///     order as if it were row-major, so the model gets the elements at the wrong indices (a (3,4) arange was
        ///     read as <c>0, 4, 8, 1, …</c>). A Fortran-ordered array now crosses as a C-order copy (or throws under
        ///     <see cref="OnnxCodecMode.View"/>).</para>
        ///     <para><b>Footgun: a copy absorbs writes.</b> When the handle holds a copy, writes through the tensor land in
        ///     the temporary (readable through <see cref="OrtTensor{T}.Source"/>), not in <paramref name="source"/>.</para>
        /// </remarks>
        /// <typeparam name="T">The C# element type ORT uses for the array's dtype (<see cref="ToTensorElementClrType"/>); anything else is refused.</typeparam>
        /// <param name="source">The array to give to ORT; any layout under <see cref="OnnxCodecMode.Auto"/> and <see cref="OnnxCodecMode.Copy"/>.</param>
        /// <param name="mode">View, copy, or view-when-possible (the default); see <see cref="OnnxCodecMode"/>.</param>
        /// <returns>A disposable handle owning the tensor, the buffer reference and any temporary copy; keep it alive while the tensor is used.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a member of <see cref="OnnxCodecMode"/>.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="T"/> is not the element type ORT uses for the array's dtype.</exception>
        /// <exception cref="InvalidOperationException"><paramref name="mode"/> is <see cref="OnnxCodecMode.View"/> and the array is not C-contiguous or is Decimal.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ONNX element type (Complex), or the array has more elements than a <see cref="Memory{T}"/> can index (in every mode: the copy lives behind a <see cref="Memory{T}"/> too).</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        public static OrtTensor<T> AsDenseTensor<T>(this NDArray source, OnnxCodecMode mode = OnnxCodecMode.Auto) where T : unmanaged
            => AsDenseTensorCore<T>(source, ownsSource: false, mode);

        /// <summary>
        ///     <see cref="AsDenseTensor{T}"/> whose returned handle also owns <paramref name="source"/> when
        ///     <paramref name="ownsSource"/> is <c>true</c>. Ownership transfers only on success; on failure the caller
        ///     still owns its array.
        /// </summary>
        /// <remarks>
        ///     When the array crosses as a copy and <paramref name="ownsSource"/> is <c>true</c>, the copy supersedes
        ///     the caller's temporary: <paramref name="source"/> is disposed at once and the handle owns only the copy.
        /// </remarks>
        /// <typeparam name="T">The C# element type ORT uses for the fed dtype (float64 for a Decimal array).</typeparam>
        /// <param name="source">The array to give to ORT.</param>
        /// <param name="ownsSource"><c>true</c> when <paramref name="source"/> is a temporary the handle should dispose.</param>
        /// <param name="mode">View, copy, or view-when-possible; see <see cref="OnnxCodecMode"/>.</param>
        /// <returns>A handle owning the tensor, the buffer reference and any temporary.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a member of <see cref="OnnxCodecMode"/>.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="T"/> is not the element type ORT uses for the array's dtype.</exception>
        /// <exception cref="InvalidOperationException"><paramref name="mode"/> is <see cref="OnnxCodecMode.View"/> and the array is not C-contiguous or is Decimal.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ONNX element type (Complex), or the array has more elements than a <see cref="Memory{T}"/> can index.</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        internal static unsafe OrtTensor<T> AsDenseTensorCore<T>(NDArray source, bool ownsSource, OnnxCodecMode mode) where T : unmanaged
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            EnsureDefined(mode);

            // Decide everything the dtype and the shape alone decide BEFORE anything is converted or copied: the element
            // type is that of the FED dtype (Decimal is fed as float64), and neither the conversion nor the C-order copy
            // changes the shape, so the source's shape is the tensor's. These ran after the conversion once, so a refused
            // call still converted the whole array first — for a broadcast view, a full-size float64 temporary of every
            // element it addresses. The element limit holds in every mode: a copy is fronted by a Memory<T> as well.
            EnsureElementType<T>(FeedTypeCode(source.typecode), nameof(AsDenseTensor));
            Shape sourceShape = source.Shape;
            if (sourceShape.Size > int.MaxValue)
                throw new NotSupportedException($"AsDenseTensor: Memory<T> holds at most {int.MaxValue} elements, the array has {sourceShape.Size}. Use AsOrtValue (a native pointer, no such limit).");
            int[] dims = IntDims(sourceShape, nameof(AsDenseTensor));

            // The view / copy decision, and the copy itself (see AsOrtValueCore): a copy is a temporary the handle owns.
            NDArray fed = MaterializeForExport(source, mode, nameof(AsDenseTensor), out bool isView);
            NDArray temporary = isView ? null : fed;
            try
            {
                IArraySlice slice = Pin(fed, out byte* data, out _);
                try
                {
                    // The manager gets the buffer too: each pin of the tensor's Memory takes its own reference on it, so a
                    // consumer that pins (ORT's CreateTensorValueFromMemory, tensor.AsNDArray()) outlives this handle safely.
                    // It also records whether the fed array is writeable, so a NumSharp view of the tensor stays read-only
                    // when the array is (the tensor itself cannot be protected: ORT has no read-only tensors).
                    var manager = new UnmanagedMemoryManager<T>((T*)data, (int)sourceShape.Size, slice, fed.Shape.IsWriteable);
                    // Row-major always: the fed array is C-contiguous, and no ORT route reads a column-major tensor right.
                    var tensor = new DenseTensor<T>(manager.Memory, dims);
                    var handle = new OrtTensor<T>(tensor, manager, fed, slice, ownsSource: temporary is not null || ownsSource, isView);
                    if (temporary is not null && ownsSource)
                        source.Dispose();   // the caller's temporary is superseded by the copy the handle now owns
                    temporary = null;       // ownership moved to the handle
                    return handle;
                }
                catch
                {
                    slice.Release();
                    throw;
                }
            }
            finally
            {
                temporary?.Dispose();   // only reached on failure — the success path nulled it
            }
        }

        /// <summary>
        ///     Copy a NumSharp array into an independent, row-major <see cref="DenseTensor{T}"/> over a fresh
        ///     managed <c>T[]</c> (no shared memory, no lifetime coupling). Any layout is read in logical (C)
        ///     order. <typeparamref name="T"/> follows the same rule as <see cref="AsDenseTensor{T}"/>.
        /// </summary>
        /// <remarks>
        ///     The element type, the size and the dimensions are checked before anything is converted, copied or
        ///     allocated; the buffer's liveness is checked next, even for an empty array.
        /// </remarks>
        /// <typeparam name="T">The C# element type ORT uses for the array's dtype (float64 for Decimal); anything else is refused.</typeparam>
        /// <param name="source">The array to copy; any layout.</param>
        /// <returns>A new row-major tensor over a managed array holding the values in logical (C) order.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="T"/> is not the element type ORT uses for the array's dtype.</exception>
        /// <exception cref="NotSupportedException">
        ///     The dtype has no ONNX element type (Complex); the array has more elements than a managed <c>T[]</c> can
        ///     hold (<see cref="Array.MaxLength"/>, 2,147,483,591 — a little below <see cref="int.MaxValue"/>); or an
        ///     axis is longer than <see cref="DenseTensor{T}"/>'s <see cref="int"/> dimensions allow.
        /// </exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        public static unsafe DenseTensor<T> ToDenseTensor<T>(this NDArray source) where T : unmanaged
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));

            // Decide everything the dtype and the shape alone decide BEFORE the source is read: the element type is
            // that of the FED dtype (Decimal is fed as float64), and the conversion and the copy never change the
            // shape, so the source's shape is the tensor's.
            EnsureElementType<T>(FeedTypeCode(source.typecode), nameof(ToDenseTensor));

            // The limit is Array.MaxLength, NOT int.MaxValue: a managed array holds at most 2,147,483,591 elements, so
            // comparing against int.MaxValue let every size in between reach `new T[n]`, which fails with
            // OutOfMemoryException ("Array dimensions exceeded supported range") instead of this documented refusal.
            Shape sourceShape = source.Shape;
            if (sourceShape.Size > Array.MaxLength)
                throw new NotSupportedException(
                    $"ToDenseTensor: a managed T[] holds at most {Array.MaxLength} elements (Array.MaxLength), the array has {sourceShape.Size}. " +
                    "Share the buffer zero-copy instead: AsDenseTensor (up to int.MaxValue elements) or AsOrtValue (a native pointer, no such limit).");
            int[] dims = IntDims(sourceShape, nameof(ToDenseTensor));

            // Take the reference on the SOURCE's buffer before reading it in any way: Pin's TryAddRef is the only
            // check that the buffer has not been released. The Decimal conversion and the densifying copy below used
            // to run first (and an empty array was never pinned at all), so a released buffer was read silently and
            // the tensor carried another array's values. Holding the reference across every read also keeps the
            // buffer alive while it is read; the managed array is allocated only once the source is known to be live.
            IArraySlice sourcePin = Pin(source);
            try
            {
                using NDArray converted = source.typecode == NPTypeCode.Decimal ? source.astype(NPTypeCode.Double) : null;
                NDArray logical = converted ?? source;
                Shape shape = logical.Shape;

                var data = new T[shape.Size];
                if (shape.Size > 0)
                {
                    using NDArray dense = shape.IsContiguous ? null : logical.copy();
                    NDArray src = dense ?? logical;
                    IArraySlice slice = Pin(src, out byte* p, out long nbytes);
                    try
                    {
                        fixed (T* dst = data)
                            Buffer.MemoryCopy(p, dst, nbytes, nbytes);
                    }
                    finally
                    {
                        slice.Release();
                        GC.KeepAlive(src);
                    }
                }

                return new DenseTensor<T>(data, dims);
            }
            finally
            {
                sourcePin.Release();
                GC.KeepAlive(source);
            }
        }

        /// <summary>
        ///     The refusal for a copy through ONNX Runtime's managed data API, whose <see cref="Span{T}"/> of bytes
        ///     cannot exceed 2 GB. The byte count is formatted from the element count and the item size in
        ///     <see cref="decimal"/>, because a broadcast view can hold enough elements to overflow a <see cref="long"/>
        ///     product.
        /// </summary>
        /// <param name="count">The number of elements to copy.</param>
        /// <param name="itemsize">The size in bytes of one element of the dtype that crosses.</param>
        /// <returns>The message, naming the zero-copy export that has no such limit.</returns>
        /// <remarks>
        ///     It used to name <c>AsNDArray</c> as well, but that is an IMPORT, and it cannot take a tensor past 2 GB either
        ///     (<see cref="ImportTooLargeMessage"/>): only the export, <see cref="AsOrtValue"/>, hands ORT a native pointer.
        /// </remarks>
        internal static string TooLargeForSpanMessage(long count, int itemsize) =>
            $"the tensor is {(decimal)count * itemsize} bytes, beyond the 2 GB System.Span limit of ONNX Runtime's managed data API " +
            "(GetTensorMutableRawData / GetTensorDataAsSpan). Share the buffer zero-copy with AsOrtValue instead of copying it: " +
            "ORT then reads NumSharp's memory through a native pointer, with no size limit.";

        /// <summary>
        ///     The refusal for reading a tensor past 2 GB out of an <see cref="OrtValue"/> (<see cref="ToNDArray(OrtValue)"/>,
        ///     <see cref="AsNDArray(OrtValue, bool, OnnxCodecMode)"/>). ORT's managed API exposes a tensor's memory only as a
        ///     <see cref="Span{T}"/> of bytes (<c>GetTensorMutableRawData</c> and every typed or <c>TensorSpan</c> variant go
        ///     through it), whose length it computes with an unchecked <c>int</c> cast; there is no other public route to the
        ///     pointer. So such a tensor can be neither copied nor viewed — but ORT can WRITE one into NumSharp's memory, which
        ///     the message names.
        /// </summary>
        /// <param name="verb">The import verb refusing (named in the message).</param>
        /// <param name="count">The tensor's element count.</param>
        /// <param name="itemsize">The size in bytes of one element.</param>
        /// <returns>The message, naming the pre-allocated-output route that has no size limit.</returns>
        internal static string ImportTooLargeMessage(string verb, long count, int itemsize) =>
            $"{verb}: the tensor is {(decimal)count * itemsize} bytes, beyond the 2 GB System.Span limit of ONNX Runtime's managed data API: " +
            "ORT exposes a tensor's memory only as a Span of bytes (GetTensorMutableRawData), so a tensor this large can be neither copied " +
            "nor viewed out of an OrtValue. Let ORT write it into NumSharp's memory instead: pre-allocate the output (np.empty(shape, dtype)) " +
            "and run session.Run(inputs, outputs), or bind nd.AsOrtValue() as the output OrtValue.";
    }
}
