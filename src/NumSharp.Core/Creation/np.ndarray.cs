using System;
using System.ComponentModel;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Constructs an array — NumPy's low-level <c>numpy.ndarray(shape, dtype=float, buffer=None, offset=0,
        ///     strides=None, order=None)</c>. Without a buffer it allocates an UNINITIALIZED array, like
        ///     <see cref="empty(Shape, DType, string)"/>; with a buffer it lays the array over the buffer's BYTES without
        ///     copying them, so writes through the result change the buffer.
        /// </summary>
        /// <param name="shape">
        ///     The dimensions. Only <see cref="Shape.Dimensions"/> is read: the strides come from <paramref name="strides"/>
        ///     or <paramref name="order"/>. With a buffer, the one-dimensional shape <c>-1</c> takes every whole element
        ///     after <paramref name="offset"/>. <c>default(Shape)</c> is NumPy's <c>None</c> and is refused.
        /// </param>
        /// <param name="dtype">The element type; null is float64 (NumPy's default). With a buffer, the buffer's bytes are
        ///     read as this type, whatever the buffer's own element type.</param>
        /// <param name="buffer">
        ///     The memory to use, or null to allocate. A C- or F-contiguous <see cref="NDArray"/>, its
        ///     <see cref="NDArray.data"/> (<see cref="MemoryView"/>), an <see cref="Backends.Unmanaged.IArraySlice"/>, a C#
        ///     array of any rank whose elements hold no references (pinned while the result lives), an
        ///     <see cref="ArraySegment{T}"/> or <see cref="Memory{T}"/> of bytes, a <see cref="ReadOnlyMemory{T}"/> of bytes
        ///     (read-only, NumPy's <c>bytes</c>), or a <see cref="Half"/>, <see cref="char"/> or <see cref="decimal"/>
        ///     value (a read-only copy of its bytes, as a NumPy scalar exports). A larger buffer than needed is fine: the
        ///     array uses a part of it.
        /// </param>
        /// <param name="offset">The byte offset of the first element into <paramref name="buffer"/>. Ignored without a
        ///     buffer, as in NumPy. It need not be a multiple of the element size (the result is then not
        ///     <c>ALIGNED</c>).</param>
        /// <param name="strides">The BYTE stride of every axis (NumPy's unit, as <see cref="NDArray.strides"/> reports),
        ///     or null for the strides of <paramref name="order"/>. Checked against the buffer, or against the array's own
        ///     size without one. A stride of 0 repeats elements and keeps the array writeable.</param>
        /// <param name="order"><c>'C'</c> (the default, NumPy's <c>None</c>) or <c>'F'</c> for the row-major or
        ///     column-major strides; <c>'A'</c> and <c>'K'</c> mean <c>'C'</c> here, and lowercase is accepted. Ignored
        ///     when <paramref name="strides"/> is given.</param>
        /// <returns>
        ///     Without a buffer: a new array that owns its uninitialized memory (an empty one has all-zero strides, as
        ///     NumPy allocates it). With a buffer: a view of the buffer that does not own its data, whose <c>base</c> is the
        ///     buffer's owner (an array) or a <c>uint8</c> array over the whole buffer, writeable when the buffer is, and
        ///     <c>ALIGNED</c> as NumPy computes it from the data pointer and the strides.
        /// </returns>
        /// <exception cref="TypeError"><paramref name="shape"/> is <c>default(Shape)</c> (<c>Use () not None as shape
        ///     arguments</c>); <paramref name="buffer"/> is not a buffer (<c>a bytes-like object is required, not
        ///     'list'</c>); or the buffer is too small without <paramref name="strides"/> (<c>buffer is too small for
        ///     requested array</c>).</exception>
        /// <exception cref="ValueError"><paramref name="order"/> is invalid; <paramref name="strides"/> has the wrong length
        ///     or does not fit the buffer (<c>strides is incompatible with shape of requested array and size of
        ///     buffer</c>); a dimension is negative or the byte count overflows; an <see cref="NDArray"/> buffer is not
        ///     contiguous (<c>ndarray is not contiguous</c>); or a negative <paramref name="offset"/> puts elements before
        ///     the buffer.</exception>
        /// <exception cref="BufferError">A <see cref="MemoryView"/> buffer over a non-contiguous array.</exception>
        /// <exception cref="NotSupportedException">A stride that addresses elements is not a multiple of the element
        ///     size (NumSharp stores strides in whole elements), <paramref name="dtype"/> has no storage yet
        ///     (datetime64), or <paramref name="dtype"/> is an UNSIZED string (<c>'S'</c>, <c>'U'</c>, <c>np.str_</c>):
        ///     NumPy builds a zero-width <c>S0</c>/<c>U0</c> array here, which NumSharp's string buffers cannot hold. With a
        ///     buffer and <c>StringDType</c> on either side: another dtype over a <c>StringDType</c> array's cells, a
        ///     <c>StringDType</c> dtype over any other buffer, or an offset that splits a cell (NumPy lays the array
        ///     anyway; see the remarks).</exception>
        /// <remarks>
        ///     <para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html. The checks run in NumPy's order
        ///     (<c>array_new</c> in <c>numpy/_core/src/multiarray/arrayobject.c</c>): the buffer, the order, the strides'
        ///     length, the strides against the buffer, the shape <c>-1</c> or the too-small check, then the dimensions.
        ///     The shape and dtype are converted by C# before the call, so a bad dtype string fails first even when the
        ///     shape is also bad (NumPy reports the shape).
        ///     </para>
        ///     <para>
        ///     <b>Safer than NumPy.</b> NumPy's own checks let four inputs through that read outside the buffer: a negative
        ///     offset, the shape <c>-1</c> with a negative stride, an offset near 2^63 (its arithmetic wraps), and strides
        ///     over an empty buffer. After NumPy's checks pass, NumSharp checks the exact byte range and refuses those, with
        ///     the strides error when strides were given, else the offset error of <see cref="frombuffer(byte[], DType, long, long)"/>
        ///     for a negative offset, else the too-small error.
        ///     </para>
        ///     <para>
        ///     <b>Lifetime.</b> The result keeps the buffer's memory alive: a NumSharp array buffer may be disposed while
        ///     the result lives, and a C# array stays pinned until the last array over it is disposed or collected. Pinning
        ///     a large managed array for a long time fragments the managed heap, so dispose the result when done.
        ///     </para>
        ///     <para>
        ///     <b>Divergences</b> (pinned by <c>[Misaligned]</c> tests): a non-native dtype (<c>'&gt;i4'</c>, <c>'&gt;U2'</c>)
        ///     with a buffer DECODES the bytes into a new native array instead of viewing them, and without a buffer
        ///     allocates the native dtype (NumSharp has no big-endian storage); <c>complex64</c> does not exist; a
        ///     multi-character order string cannot be expressed; a view of an unaligned result reports <c>ALIGNED</c>
        ///     again; an unsized string dtype is refused (NumPy builds a zero-width <c>S0</c>/<c>U0</c> array, and for the
        ///     shape <c>(-1,)</c> over a buffer it divides by the zero item size and crashes the process).
        ///     </para>
        ///     <para>
        ///     <b>Strings.</b> A sized <c>S</c>/<c>U</c> dtype reads the bytes as fixed-width elements, exactly as NumPy
        ///     does: <c>np.ndarray((4,), 'S3', buffer=b'abcdefghijkl')</c> is <c>[b'abc', b'def', b'ghi', b'jkl']</c>, and a
        ///     <c>U</c> dtype reads native UCS-4 code points (a code point above <c>U+10FFFF</c> is stored, and NumPy's
        ///     <c>SystemError</c> is raised only when that element is read). A string array buffer viewed with another
        ///     string width or class (<c>'&lt;U1'</c> over a <c>'&lt;U2'</c> array) is an alias of its bytes, never its
        ///     elements.
        ///     </para>
        ///     <para>
        ///     <b>StringDType.</b> Without a buffer the array's 16-byte cells are zeroed (every element <c>''</c>, NumPy's
        ///     <c>NPY_NEEDS_INIT</c>). With a buffer, a <c>StringDType</c> dtype over a <c>StringDType</c> array is NumPy's
        ///     view of its cells (offset and strides in bytes, whole cells), sharing the array's allocator, with the
        ///     requested descriptor's NA object and coercion. NumPy also lays any other dtype over a <c>StringDType</c>
        ///     array's raw cells, and a <c>StringDType</c> array over any bytes; NumSharp refuses both, because a cell of a
        ///     long text is a reference into its allocator (a writable numeric view could corrupt it, and raw bytes have no
        ///     allocator). <c>np.frombuffer(a.tobytes(), dtype)</c> reads the same cells as a copy.
        ///     </para>
        /// </remarks>
        public static NDArray ndarray(Shape shape, DType dtype = null, object buffer = null, long offset = 0, long[] strides = null, char order = 'C')
        {
            // PyArray_IntpConverter refuses None before any other argument is converted.
            if (shape.dimensions is null)
                throw new TypeError("Use () not None as shape arguments");

            return UnmanagedStorage.ArrayNew(shape, dtype, buffer, offset, strides, order);
        }

        /// <summary>
        ///     Catches the call shape of the old signature, <c>ndarray(shape, dtype, buffer, order)</c>: its fourth
        ///     positional argument was the order, and a C# <see cref="char"/> converts silently to the new
        ///     <see cref="long"/> <c>offset</c> (<c>'F'</c> would read as offset 70).
        /// </summary>
        /// <param name="shape">The dimensions.</param>
        /// <param name="dtype">The element type.</param>
        /// <param name="buffer">The buffer.</param>
        /// <param name="offset">The old order argument, passed by position.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="NotSupportedException">Always; the call does not compile anyway.</exception>
        /// <remarks>Name the argument instead: <c>np.ndarray(shape, dtype, buffer, order: 'F')</c>.</remarks>
        [Obsolete("np.ndarray's fourth parameter is `offset` (NumPy's order: shape, dtype, buffer, offset, strides, order), " +
                  "and a char converts to it silently. Name the order argument: np.ndarray(shape, dtype, buffer, order: 'F').", error: true)]
        [EditorBrowsable(EditorBrowsableState.Never)]
        public static NDArray ndarray(Shape shape, DType dtype, object buffer, char offset)
            => throw new NotSupportedException("Name the order argument: np.ndarray(shape, dtype, buffer, order: 'F').");
    }
}
