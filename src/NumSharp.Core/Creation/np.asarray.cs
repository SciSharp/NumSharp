using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.asarray('abc')</c>: a 0-d fixed-width Unicode array holding <paramref name="a"/> (see
        ///     <see cref="array(string)"/>).
        /// </summary>
        /// <param name="a">The text.</param>
        /// <returns>A 0-d <c>U</c> array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>NumPy's parameter name <c>a</c> since API-parity batch 8 (it was <c>data</c>, a name NumPy's <c>asarray</c> refuses as a keyword).</remarks>
        public static NDArray asarray(string a) => array(a);

        /// <summary>
        ///     NumPy's <c>np.asarray(2)</c>: a new 0-d array holding one C# value. Like <c>np.array(2)</c>, the
        ///     result is an explicit 0-d ARRAY (<see cref="NDArray.IsExplicitZeroDimArray"/>): a matrix multiplied by
        ///     it takes NumPy's 1 x 1 matrix route instead of scaling.
        /// </summary>
        /// <typeparam name="T">The value's type; it must map to a NumSharp dtype.</typeparam>
        /// <param name="a">The value.</param>
        /// <returns>A new writeable 0-d array that owns its one element.</returns>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no NumSharp dtype.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html. NumPy's parameter name <c>a</c> since API-parity batch 8 (it was <c>data</c>, a name NumPy's <c>asarray</c> refuses as a keyword).</remarks>
        public static NDArray asarray<T>(T a) where T : struct
        {
            var nd = new NDArray(typeof(T), new Shape(Array.Empty<long>(), copy: false));
            nd.ReplaceData(new T[] {a});
            return nd.MarkExplicitZeroDimArray();
        }

        /// <summary>
        ///     NumPy's <c>np.asarray(['a', 'bb'])</c>: a 1-D fixed-width Unicode array (see <see cref="array(string[])"/>).
        /// </summary>
        /// <param name="a">The values.</param>
        /// <param name="ndim">Kept for source compatibility; a string vector is always 1-D.</param>
        /// <returns>The <c>U</c> array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="NotSupportedException">An entry is null (an <c>object</c> array in NumPy).</exception>
        /// <remarks>NumPy's parameter name <c>a</c> since API-parity batch 8 (it was <c>data</c>, a name NumPy's <c>asarray</c> refuses as a keyword).</remarks>
        public static NDArray asarray(string[] a, int ndim = 1) => array(a);

        /// <summary>
        ///     NumPy's <c>np.asarray([1, 2, 3])</c> of a C# array of values: a new 1-D array of <typeparamref name="T"/>'s
        ///     dtype holding a COPY of <paramref name="a"/> (a C# array is NumPy's Python list, which np.asarray always
        ///     converts into a new array).
        /// </summary>
        /// <typeparam name="T">The element type; it must map to a NumSharp dtype.</typeparam>
        /// <param name="a">The values (only their count and elements are read: a multi-dimensional C# array binds the
        ///     <see cref="NDArray"/> form through its implicit conversion).</param>
        /// <param name="ndim">Kept for source compatibility; the result is always 1-D.</param>
        /// <returns>A new 1-D array that owns a copy of the values.</returns>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null.</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no NumSharp dtype.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html. NumPy's parameter name <c>a</c> since API-parity batch 8 (it was <c>data</c>, a name NumPy's <c>asarray</c> refuses as a keyword).</remarks>
        public static NDArray asarray<T>(T[] a, int ndim = 1) where T : struct
        {
            var nd = new NDArray(typeof(T), new Shape(a.Length));
            nd.ReplaceData(a);
            return nd;
        }

        /// <summary>
        ///     Convert the input to an ndarray, matching NumPy 2.x semantics.
        ///     If <paramref name="a"/> already satisfies the requested dtype/layout, it is returned as-is — no copy.
        /// </summary>
        /// <param name="a">Input ndarray. A C# literal (<c>np.asarray(1000, np.int8)</c>, whose 0-d array is marked
        ///     <see cref="NDArray.WasPythonLiteral"/>) is NumPy's Python scalar: with a <paramref name="dtype"/> it is packed
        ///     with NumPy's range checks into a new array, not cast.</param>
        /// <param name="dtype">Requested dtype. <c>null</c> keeps the input dtype.</param>
        /// <param name="order">'C' (row-major), 'F' (column-major), 'A' (any contiguous), 'K' (keep); <c>null</c> (NumPy's
        ///     <c>None</c>, the default) is 'K'. 'A'/'K' never force a copy on layout grounds.</param>
        /// <param name="copy">Tri-state: <c>null</c> = copy only if needed (default), <c>true</c> = always copy, <c>false</c> = never copy (raises if a copy would be required).</param>
        /// <param name="like">Reference array for array-function dispatch — accepted for NumPy parity but has no observable effect in NumSharp.</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted.</param>
        /// <returns>NDArray with the requested dtype and memory layout. Returns <paramref name="a"/> when no copy is needed.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <exception cref="NotSupportedException"><paramref name="a"/> is null (Python's <c>None</c>) and
        ///     <paramref name="dtype"/> is null: NumPy builds an object array, a dtype NumSharp does not have. With a dtype,
        ///     <c>None</c> is the 0-d array of its setitem value (NaN, <c>nan+nanj</c>, <c>False</c>, <c>'None'</c>, a
        ///     <c>StringDType</c> NA object).</exception>
        /// <exception cref="ValueError">An unsupported <paramref name="device"/> (NumPy's <c>Device not understood. Only "cpu"
        ///     is allowed, but received: ...</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="copy"/> is <c>false</c> and a copy (or a C# literal's or
        ///     <c>None</c>'s conversion) is required.</exception>
        /// <exception cref="OverflowError">A C# literal out of <paramref name="dtype"/>'s integer range (NumPy's
        ///     OverflowError texts).</exception>
        /// <exception cref="ValueError">A float NaN literal into an integer <paramref name="dtype"/>.</exception>
        /// <exception cref="TypeError">A complex literal into a real <paramref name="dtype"/>, or <c>None</c> into an integer
        ///     (or <c>decimal</c>) <paramref name="dtype"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html</remarks>
        public static NDArray asarray(
            NDArray a,
            DType dtype = null,
            char? order = null,
            bool? copy = null,
            NDArray like = null,
            string device = null)
        {


            // Only "cpu" is supported (matches NumPy 2.x). null is treated as "default". The creation functions' one check
            // (PyArray_DeviceConverterOptional), with NumPy's ValueError text.
            DeviceArguments.ValidateDevice(device);

            // `like` exists for NumPy's __array_function__ dispatch protocol. NumSharp has
            // no array-subclass dispatch, so the value is accepted but never read.
            _ = like;

            // order=None is NumPy's 'K' for asarray (PyArray_OrderConverter leaves NPY_KEEPORDER in place).
            return NDArray.asarray(a, dtype, order ?? 'K', copy);
        }

        /// <summary>
        ///     Convert a <see cref="MemoryView"/> (obtained from <see cref="NDArray.data"/>) to an ndarray —
        ///     the consumer round-trip of <c>ndarray.data</c>. Matches NumPy's <c>np.asarray(a.data)</c>,
        ///     which reads the buffer's shape/strides/dtype and returns a zero-copy VIEW sharing its memory
        ///     (copying only if a dtype/layout change forces it), preserving the source's N-D shape and
        ///     layout. Equivalent to <c>np.asarray(a.obj, …)</c>.
        /// </summary>
        /// <param name="a">A <see cref="MemoryView"/> over an array.</param>
        /// <param name="dtype">Requested dtype. <c>null</c> keeps the source dtype.</param>
        /// <param name="order">'C', 'F', 'A', or 'K'; <c>null</c> (NumPy's <c>None</c>, the default) is 'K'.</param>
        /// <param name="copy">Tri-state copy: <c>null</c> = if-needed, <c>true</c> = always, <c>false</c> = never (raises).</param>
        /// <param name="like">Reference for array-function dispatch — accepted for parity, no effect.</param>
        /// <param name="device">Only <c>"cpu"</c> or <c>null</c>.</param>
        /// <returns>The array the memory view exposes (a view sharing its memory), or a copy when the dtype / order /
        ///     copy request forces one.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError">An unsupported <paramref name="device"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="copy"/> is <c>false</c> and a copy is required.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html. NumPy's parameter name <c>a</c> since API-parity batch 8 (it was <c>buffer</c>, a name NumPy's <c>asarray</c> refuses as a keyword).</remarks>
        public static NDArray asarray(
            MemoryView a,
            DType dtype = null,
            char? order = null,
            bool? copy = null,
            NDArray like = null,
            string device = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return asarray(a.obj, dtype, order, copy, like, device);
        }

        /// <summary>
        ///     Convert a <see cref="MemoryView"/> to an ndarray, taking a NumPy-style dtype string
        ///     (e.g. <c>"float32"</c>, <c>"&lt;i4"</c>). See
        ///     <see cref="asarray(MemoryView, DType, char?, bool?, NDArray, string)"/>.
        /// </summary>
        /// <param name="a">A <see cref="MemoryView"/> over an array.</param>
        /// <param name="dtype">The requested dtype as NumPy's dtype string (parsed as <c>np.dtype(string)</c> does).</param>
        /// <param name="order">'C', 'F', 'A', or 'K'; <c>null</c> (NumPy's <c>None</c>, the default) is 'K'.</param>
        /// <param name="copy">Tri-state copy: <c>null</c> = if-needed, <c>true</c> = always, <c>false</c> = never (raises).</param>
        /// <param name="like">Reference for array-function dispatch — accepted for parity, no effect.</param>
        /// <param name="device">Only <c>"cpu"</c> or <c>null</c>.</param>
        /// <returns>The array the memory view exposes (a view sharing its memory), or a copy when the dtype / order /
        ///     copy request forces one.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="dtype"/> is not a dtype string NumPy understands.</exception>
        /// <exception cref="ValueError">An unsupported <paramref name="device"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="copy"/> is <c>false</c> and a copy is required.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html. NumPy's parameter name <c>a</c> since API-parity batch 8 (it was <c>buffer</c>, a name NumPy's <c>asarray</c> refuses as a keyword).</remarks>
        public static NDArray asarray(
            MemoryView a,
            string dtype,
            char? order = null,
            bool? copy = null,
            NDArray like = null,
            string device = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return asarray(a.obj, dtype, order, copy, like, device);
        }
    }
}
