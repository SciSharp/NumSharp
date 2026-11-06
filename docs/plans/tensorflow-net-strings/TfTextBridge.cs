using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;
using Tensorflow;
using Tensorflow.Eager;
using static Tensorflow.Binding;
using NSArray = NumSharp.NDArray;
using NSShape = NumSharp.Shape;

namespace TfProbe
{
    /// <summary>
    ///     A prototype of the text half of a NumSharp-side TensorFlow bridge (what an in-repo
    ///     <c>NumSharp.Interop.TensorFlow</c> package could ship): NumSharp text arrays (<c>S</c>, <c>U</c>,
    ///     <c>StringDType</c>) to TensorFlow string tensors and back, writing and reading TensorFlow's 24-byte
    ///     <c>TF_TString</c> cells straight from NumSharp's buffers, with no managed <c>string</c> / <c>byte[]</c>
    ///     intermediate per element.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why NumSharp-side.</b> The <c>StringDType</c> routes read the 16-byte NEP 55 cells through NumSharp's internal
    ///     <see cref="VStringAllocator"/> (an arena or heap text cannot be found from the cell alone), so only an assembly
    ///     NumSharp grants its internals to can move <c>StringDType</c> text without decoding it to UTF-16 strings first.
    ///     Code inside TensorFlow.NET could use NumSharp's PUBLIC surface only (<see cref="NSArray.AsStringArray"/>,
    ///     <c>GetAtIndex</c>), which is the "public route" the probe compares against.
    ///     </para>
    ///     <para>
    ///     <b>Ownership of the cells.</b> TensorFlow.NET builds its string tensors with <c>TF_AllocateTensor</c>, whose buffer
    ///     deallocator frees the memory but runs no <c>tstring</c> destructor, and frees the long texts itself
    ///     (<c>SafeStringTensorHandle.ReleaseHandle</c> calls <c>TF_StringDealloc</c> per cell) when ITS handle is released,
    ///     even while another tensor still shares the buffer. Here the cells are a native block given to
    ///     <c>TF_NewTensor</c> with a deallocator that frees the long texts and then the block: TensorFlow calls it when the
    ///     LAST tensor sharing the buffer is gone, so the texts live exactly as long as TensorFlow reads them.
    ///     </para>
    ///     <para>
    ///     <b>Element semantics</b> follow what Python TensorFlow does with NumPy text (probed with TF 2.21 / NumPy 2.4.2):
    ///     a <c>U</c> element crosses as its UTF-8 encoding without trailing NULs (a lone surrogate is refused, Python TF's
    ///     "Unable to convert element to UTF-8"); an <c>S</c> element as its bytes without trailing NULs; a
    ///     <c>StringDType</c> element as its UTF-8 text, NUL bytes included. Python TF refuses <c>StringDType</c> arrays
    ///     outright ("Unsupported numpy type"); accepting them here is the probe's question, not NumPy's answer.
    ///     </para>
    /// </remarks>
    internal static unsafe class TfTextBridge
    {
        /// <summary><c>sizeof(TF_TString)</c> on a 64-bit build (<c>tensorflow/tsl/platform/ctstring_internal.h</c>).</summary>
        internal const int TStringSize = 24;

        /// <summary>The bytes NumSharp's <c>StringDType</c> cell is (<c>SIZEOF_NPY_PACKED_STATIC_STRING</c>).</summary>
        private const int VCellSize = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;

        /// <summary>How many cell blocks a deallocator has freed: the lifetime checks read it.</summary>
        internal static long ReleasedBlocks;

        /// <summary>How many NumSharp buffer pins a view export's deallocator has released.</summary>
        internal static long ReleasedPins;

        [DllImport("tensorflow")] private static extern void TF_StringInit(IntPtr tstr);
        [DllImport("tensorflow")] private static extern void TF_StringCopy(IntPtr dst, byte* src, nuint size);
        [DllImport("tensorflow")] private static extern void TF_StringAssignView(IntPtr dst, byte* src, nuint size);
        [DllImport("tensorflow")] private static extern byte* TF_StringGetDataPointer(IntPtr tstr);
        [DllImport("tensorflow")] private static extern nuint TF_StringGetSize(IntPtr tstr);
        [DllImport("tensorflow")] private static extern void TF_StringDealloc(IntPtr tstr);

        // TF_NewTensor with a typed function-pointer deallocator: TensorFlow.NET's own declaration takes a delegate, which
        // would have to be rooted for the life of every tensor; an [UnmanagedCallersOnly] static needs no root.
        [DllImport("tensorflow")]
        private static extern IntPtr TF_NewTensor(TF_DataType dtype, long* dims, int numDims, void* data, nuint len,
            delegate* unmanaged[Cdecl]<void*, nuint, void*, void> deallocator, void* deallocatorArg);

        /// <summary>How a <see cref="ToTensor"/> call gives TensorFlow each element's bytes.</summary>
        internal enum Payload
        {
            /// <summary>Copy every text into TensorFlow-owned <c>tstring</c>s (texts of 22 bytes or fewer live in the cell).</summary>
            Copy,

            /// <summary>
            ///     Point every <c>tstring</c> at the text inside the NumSharp buffer (<c>TF_TSTR_VIEW</c>), so no text byte is
            ///     copied; the tensor pins the NumSharp buffer until TensorFlow's last reference is gone. <c>S</c> and
            ///     <c>StringDType</c> only (a <c>U</c> element is UCS-4, not UTF-8). The NumSharp array must not be WRITTEN
            ///     while a tensor views it: an <c>S</c> write changes what TensorFlow reads, and a <c>StringDType</c> write can
            ///     move or free the arena / heap text a view points at.
            /// </summary>
            View,
        }

        /// <summary>
        ///     Copy (or view) a NumSharp text array into a new eager TensorFlow string tensor of the same shape, in logical
        ///     (C) order.
        /// </summary>
        /// <param name="source">An <c>S</c>, <c>U</c> or <c>StringDType</c> array of any shape; a non-C-contiguous view is
        ///     first copied into C order by NumSharp (one element move, then the same per-element write).</param>
        /// <param name="payload"><see cref="Payload.Copy"/> (default), or <see cref="Payload.View"/> for an <c>S</c> /
        ///     <c>StringDType</c> array that stays unmodified while the tensor lives.</param>
        /// <returns>A new eager tensor of dtype <c>tf.string</c>; TensorFlow frees its texts with its last reference.</returns>
        /// <exception cref="ArgumentException"><paramref name="source"/> is not a text array.</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> element is NA (TensorFlow strings have no missing
        ///     value); a <c>U</c> element holds a lone surrogate or a unit past U+10FFFF (no UTF-8 form, Python TF's refusal);
        ///     or <see cref="Payload.View"/> was asked for a <c>U</c> array.</exception>
        /// <exception cref="InvalidOperationException">TensorFlow refused the tensor or its eager handle.</exception>
        internal static EagerTensor ToTensor(NSArray source, Payload payload = Payload.Copy)
        {
            if (source is null || !source.Storage.IsString)
                throw new ArgumentException("ToTensor takes an S, U or StringDType array.", nameof(source));
            char kind = source.dtype.kind;
            if (payload == Payload.View && kind == 'U')
                throw new NotSupportedException("A U element is UCS-4 code points, not UTF-8: it can only be copied (Payload.Copy).");

            // The per-element writers read ONE C-ordered run; any other layout is put in C order by NumSharp first (for
            // StringDType that move goes through NumSharp's own dup, so arena / heap texts land in the copy's allocator).
            NSArray run = source.Shape.IsContiguous ? source : np.ascontiguousarray(source);
            bool ownsRun = !ReferenceEquals(run, source);
            try
            {
                long count = run.size;
                long[] dims = run.shape;
                nuint len = (nuint)(count * TStringSize);
                // The block TensorFlow's buffer will own; 64-byte aligned like TensorFlow's own allocations, never
                // zero-length (a zero-element tensor still hands its block to the deallocator).
                byte* cells = (byte*)NativeMemory.AlignedAlloc(Math.Max(len, (nuint)TStringSize), 64);
                // Every cell becomes a valid empty tstring BEFORE any text is written, so a failure half way leaves a
                // block the deallocator path can walk.
                for (long i = 0; i < count; i++)
                    TF_StringInit((IntPtr)(cells + i * TStringSize));

                IArraySlice pin = null;
                try
                {
                    if (payload == Payload.View)
                    {
                        // The pin is the counted reference the deallocator gives back: it keeps the NumSharp buffer alive
                        // even after the caller disposes the array, for as long as TensorFlow holds the cells.
                        pin = run.Storage.InternalArray;
                        if (!pin.TryAddRef())
                            throw new ObjectDisposedException("source", "The array's buffer has already been released.");
                    }

                    switch (kind)
                    {
                        case 'S': WriteBytesElements(run, cells, count, payload); break;
                        case 'U': WriteUnicodeElements(run, cells, count); break;
                        default: WriteVStringElements(run, cells, count, payload); break;
                    }

                    // The pin rides as the deallocator's argument through a GCHandle (a managed reference cannot cross).
                    void* arg = pin is null ? null : (void*)GCHandle.ToIntPtr(GCHandle.Alloc(pin));
                    IntPtr raw;
                    fixed (long* d = dims)
                    {
                        delegate* unmanaged[Cdecl]<void*, nuint, void*, void> release = &ReleaseCells;
                        raw = TF_NewTensor(TF_DataType.TF_STRING, d, dims.Length, cells, len, release, arg);
                    }
                    if (raw == IntPtr.Zero)
                        throw new InvalidOperationException("TF_NewTensor refused the string tensor.");

                    // From here TensorFlow owns the block (its deallocator frees it): the catch below must not free it again.
                    cells = null;
                    pin = null;
                    using var tensorHandle = new SafeTensorHandle(raw);
                    // The eager handle shares the buffer with the TF_Tensor, which can then go (the using above): the
                    // buffer, and with it the texts, lives until TensorFlow drops its last reference.
                    SafeEagerTensorHandle eager = c_api.TFE_NewTensorHandle(tensorHandle, tf.Status);
                    tf.Status.Check(true);
                    return new EagerTensor(eager);
                }
                catch
                {
                    if (cells != null)
                    {
                        // Nothing reached TensorFlow: free the long texts written so far, then the block, and the pin.
                        for (long i = 0; i < count; i++)
                            TF_StringDealloc((IntPtr)(cells + i * TStringSize));
                        NativeMemory.AlignedFree(cells);
                    }
                    pin?.Release();
                    throw;
                }
            }
            finally
            {
                if (ownsRun)
                    run.Dispose();
            }
        }

        /// <summary>
        ///     TensorFlow's deallocator for a block built by <see cref="ToTensor"/>: free each cell's long text (a no-op for
        ///     a short or VIEW cell), the block, and the NumSharp pin of a view export.
        /// </summary>
        /// <param name="data">The block.</param>
        /// <param name="len">Its length in bytes (cells × <see cref="TStringSize"/>).</param>
        /// <param name="arg">A <see cref="GCHandle"/> to the pinned <see cref="IArraySlice"/>, or null for a copy export.</param>
        /// <remarks>Runs on whichever thread drops TensorFlow's last reference; the ARC release is thread-safe.</remarks>
        [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static void ReleaseCells(void* data, nuint len, void* arg)
        {
            long count = (long)(len / TStringSize);
            for (long i = 0; i < count; i++)
                TF_StringDealloc((IntPtr)((byte*)data + i * TStringSize));
            NativeMemory.AlignedFree(data);
            Interlocked.Increment(ref ReleasedBlocks);
            if (arg != null)
            {
                var handle = GCHandle.FromIntPtr((IntPtr)arg);
                ((IArraySlice)handle.Target).Release();
                handle.Free();
                Interlocked.Increment(ref ReleasedPins);
            }
        }

        /// <summary>Write the <c>S</c> elements of a C-ordered run: each element's bytes up to its last non-NUL byte.</summary>
        /// <param name="run">The C-contiguous <c>S</c> array.</param>
        /// <param name="cells">The initialized tstring cells.</param>
        /// <param name="count">The element count.</param>
        /// <param name="payload">Copy or view.</param>
        private static void WriteBytesElements(NSArray run, byte* cells, long count, Payload payload)
        {
            int width = run.dtype.itemsize;
            byte* first = run.Storage.Address + run.Shape.offset * (long)width;
            for (long i = 0; i < count; i++)
            {
                byte* text = first + i * width;
                int size = width;
                // NumPy's getitem (and Python TF) drops trailing NUL padding; a NUL inside the text stays.
                while (size > 0 && text[size - 1] == 0)
                    size--;
                IntPtr cell = (IntPtr)(cells + i * TStringSize);
                if (payload == Payload.View)
                    TF_StringAssignView(cell, text, (nuint)size);
                else if (size != 0)
                    TF_StringCopy(cell, text, (nuint)size);
            }
        }

        /// <summary>
        ///     Write the <c>U</c> elements of a C-ordered run: each element's code points (without trailing NULs) encoded to
        ///     UTF-8 in a reused scratch buffer, then copied into the cell.
        /// </summary>
        /// <param name="run">The C-contiguous <c>U</c> array.</param>
        /// <param name="cells">The initialized tstring cells.</param>
        /// <param name="count">The element count.</param>
        /// <exception cref="NotSupportedException">A unit has no UTF-8 form (a surrogate, or past U+10FFFF).</exception>
        private static void WriteUnicodeElements(NSArray run, byte* cells, long count)
        {
            int chars = run.dtype.itemsize / 4;
            uint* first = (uint*)(run.Storage.Address + run.Shape.offset * (long)run.dtype.itemsize);
            // 4 UTF-8 bytes per code point at most: one buffer for the whole run.
            byte[] scratch = new byte[Math.Max(4, chars * 4)];
            fixed (byte* buf = scratch)
            {
                for (long i = 0; i < count; i++)
                {
                    uint* units = first + i * chars;
                    int n = chars;
                    while (n > 0 && units[n - 1] == 0)
                        n--;
                    int size = 0;
                    for (int k = 0; k < n; k++)
                    {
                        uint cp = units[k];
                        if (cp < 0x80)
                            buf[size++] = (byte)cp;
                        else if (cp < 0x800)
                        {
                            buf[size++] = (byte)(0xC0 | (cp >> 6));
                            buf[size++] = (byte)(0x80 | (cp & 0x3F));
                        }
                        else if (cp < 0x10000)
                        {
                            if (cp >= 0xD800 && cp <= 0xDFFF)
                                throw new NotSupportedException($"Element {i} holds the lone surrogate U+{cp:X4}, which has no UTF-8 form (Python TF: 'Unable to convert element to UTF-8').");
                            buf[size++] = (byte)(0xE0 | (cp >> 12));
                            buf[size++] = (byte)(0x80 | ((cp >> 6) & 0x3F));
                            buf[size++] = (byte)(0x80 | (cp & 0x3F));
                        }
                        else if (cp <= 0x10FFFF)
                        {
                            buf[size++] = (byte)(0xF0 | (cp >> 18));
                            buf[size++] = (byte)(0x80 | ((cp >> 12) & 0x3F));
                            buf[size++] = (byte)(0x80 | ((cp >> 6) & 0x3F));
                            buf[size++] = (byte)(0x80 | (cp & 0x3F));
                        }
                        else
                            throw new NotSupportedException($"Element {i} holds the unit 0x{cp:X}, past U+10FFFF.");
                    }
                    if (size != 0)
                        TF_StringCopy((IntPtr)(cells + i * TStringSize), buf, (nuint)size);
                }
            }
        }

        /// <summary>
        ///     Write the <c>StringDType</c> elements of a C-ordered run: each cell is unpacked by its own allocator
        ///     (<c>NpyString_load</c>) and its UTF-8 bytes copied (or viewed) into the tstring, under the allocator's lock.
        /// </summary>
        /// <param name="run">The C-contiguous <c>StringDType</c> array.</param>
        /// <param name="cells">The initialized tstring cells.</param>
        /// <param name="count">The element count.</param>
        /// <param name="payload">Copy or view.</param>
        /// <exception cref="NotSupportedException">An element is NA: a TensorFlow string has no missing value (Python TF
        ///     refuses StringDType arrays altogether).</exception>
        private static void WriteVStringElements(NSArray run, byte* cells, long count, Payload payload)
        {
            VStringArraySlice slice = run.Storage.VStringSlice;
            VStringAllocator allocator = slice.Allocator;
            byte* first = run.Storage.Address + run.Shape.offset * (long)VCellSize;
            // One lock for the whole walk, as NumPy's loops take it: load's pointers stay valid while it is held.
            allocator.acquire_allocator();
            try
            {
                for (long i = 0; i < count; i++)
                {
                    int status = allocator.load(first + i * VCellSize, out byte* text, out long size);
                    if (status == 1)
                        throw new NotSupportedException($"Element {i} is NA: a TensorFlow string tensor has no missing value. Replace the NAs with a text first, or pass a StringDType array without them.");
                    if (status < 0)
                        throw new InvalidOperationException($"Element {i}'s cell names an arena that does not exist (a corrupt cell).");
                    IntPtr cell = (IntPtr)(cells + i * TStringSize);
                    if (payload == Payload.View)
                        TF_StringAssignView(cell, text, (nuint)size);
                    else if (size != 0)
                        TF_StringCopy(cell, text, (nuint)size);
                }
            }
            finally
            {
                allocator.release_allocator();
            }
        }

        /// <summary>What <see cref="ToNDArray"/> builds from a string tensor.</summary>
        internal enum TextKind
        {
            /// <summary><c>StringDType</c>: every text kept whole (NUL bytes, any length); invalid UTF-8 is refused, as Python's <c>tensor.numpy().astype(StringDType())</c> raises <c>UnicodeDecodeError</c>.</summary>
            StringDType,

            /// <summary><c>S</c> of the longest text's width: the raw bytes, any bytes; trailing NULs read back stripped (NumPy's <c>S</c>), Python's <c>tensor.numpy().astype('S')</c>.</summary>
            Bytes,
        }

        /// <summary>
        ///     Copy an eager TensorFlow string tensor into a new owning NumSharp text array of its shape, reading the
        ///     <c>TF_TString</c> cells directly.
        /// </summary>
        /// <param name="tensor">A <c>tf.string</c> tensor (eager; resolved to host memory by TensorFlow.NET).</param>
        /// <param name="kind">The NumSharp dtype to build.</param>
        /// <returns>A new owning C-contiguous array: <c>StringDType</c> (no NA object) or <c>S{longest}</c>.</returns>
        /// <exception cref="ArgumentException"><paramref name="tensor"/> is not a string tensor.</exception>
        /// <exception cref="UnicodeDecodeError"><see cref="TextKind.StringDType"/> was asked and an element is not valid UTF-8.</exception>
        /// <exception cref="OutOfMemoryException">The <c>StringDType</c> allocator could not take a text.</exception>
        internal static NSArray ToNDArray(Tensor tensor, TextKind kind = TextKind.StringDType)
        {
            if (tensor.dtype != TF_DataType.TF_STRING)
                throw new ArgumentException($"ToNDArray reads a tf.string tensor, not {tensor.dtype}.", nameof(tensor));
            long[] dims = tensor.shape.dims;
            long count = 1;
            foreach (long d in dims)
                count *= d;
            byte* cells = (byte*)tensor.TensorDataPointer;
            var shape = new NSShape(dims);

            if (kind == TextKind.Bytes)
            {
                long width = 1;
                for (long i = 0; i < count; i++)
                    width = Math.Max(width, (long)TF_StringGetSize((IntPtr)(cells + i * TStringSize)));
                NSArray result = np.zeros(shape, np.dtype("S" + width));
                byte* dst = result.Storage.Address;
                for (long i = 0; i < count; i++)
                {
                    IntPtr cell = (IntPtr)(cells + i * TStringSize);
                    long size = (long)TF_StringGetSize(cell);
                    System.Buffer.MemoryCopy(TF_StringGetDataPointer(cell), dst + i * width, size, size);
                }
                GC.KeepAlive(tensor);
                return result;
            }

            // StringDType: validate every element first (the error names the FIRST bad element, before anything is
            // packed, as numpy's decode raises), and sum the arena bytes so the arena is reserved once.
            long arena = 0;
            for (long i = 0; i < count; i++)
            {
                IntPtr cell = (IntPtr)(cells + i * TStringSize);
                long size = (long)TF_StringGetSize(cell);
                var bytes = new ReadOnlySpan<byte>(TF_StringGetDataPointer(cell), checked((int)size));
                if (!System.Text.Unicode.Utf8.IsValid(bytes))
                    RaiseDecodeError(bytes);
                if (size > VStringAllocator.NPY_SHORT_STRING_MAX_SIZE)
                    arena += VStringAllocator.ArenaBytesForText(size);
            }

            NSArray text = np.empty(shape, np.dtypes.StringDType());
            VStringAllocator allocator = text.Storage.VStringSlice.Allocator;
            byte* out0 = text.Storage.Address;
            allocator.acquire_allocator();
            try
            {
                allocator.TryReserve(arena);
                for (long i = 0; i < count; i++)
                {
                    IntPtr cell = (IntPtr)(cells + i * TStringSize);
                    if (allocator.pack(out0 + i * VCellSize, TF_StringGetDataPointer(cell), (long)TF_StringGetSize(cell)) < 0)
                        throw new OutOfMemoryException("Failed to pack string while converting from a TensorFlow string tensor.");
                }
            }
            finally
            {
                allocator.release_allocator();
            }
            GC.KeepAlive(tensor);
            return text;
        }

        /// <summary>
        ///     Raise CPython's exact <c>UnicodeDecodeError</c> for one element by decoding it with NumSharp's own codec
        ///     (<c>np.strings.decode</c> is <c>bytes.decode('utf-8')</c> per element), so the text names the byte, the
        ///     position and the reason exactly as Python's <c>tensor.numpy().astype(StringDType())</c> does.
        /// </summary>
        /// <param name="bytes">The invalid element.</param>
        /// <exception cref="UnicodeDecodeError">Always.</exception>
        /// <exception cref="InvalidOperationException">The codec accepted the bytes (cannot happen for invalid UTF-8).</exception>
        private static void RaiseDecodeError(ReadOnlySpan<byte> bytes)
        {
            using NSArray one = np.array(new object[] { bytes.ToArray() }, np.dtype("S" + Math.Max(1, bytes.Length)));
            using NSArray never = np.strings.decode(one);
            throw new InvalidOperationException("np.strings.decode accepted bytes that System.Text.Unicode.Utf8 calls invalid.");
        }
    }
}
