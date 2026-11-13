using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Create an array from an existing <see cref="NDArray"/>. Matches NumPy's default
        ///     <c>np.array(a)</c> which always copies (NumPy 2.x: <c>copy=True</c> by default) into a FRESH, WRITEABLE array
        ///     laid out in the source's K order — <see cref="copy(NDArray, char, bool)"/>'s route with <c>order='K'</c>, so a
        ///     read-only, F-ordered or broadcast source gives a writeable copy that keeps the source's memory order (a
        ///     <c>(0, 8)</c>-strided broadcast copies F-contiguous).
        /// </summary>
        /// <param name="nd">Source array.</param>
        /// <param name="copy">When <c>true</c> (default) the data is copied as described above; when <c>false</c> the
        ///     storage is shared (alias), flags included. For "copy only if needed" semantics use
        ///     <see cref="asarray(NDArray, DType, char?, bool?, NDArray, string)"/>.</param>
        /// <returns>A new array object, a plain array whatever <paramref name="nd"/>'s class (NumPy's <c>subok=False</c>).
        ///     A 0-d source that was built as an explicit 0-d ARRAY (<see cref="NDArray.IsExplicitZeroDimArray"/>) gives an
        ///     explicit 0-d array again, as NumPy's <c>np.array(np.array(2))</c> is a 0-d array; so does the copy of a NumPy
        ///     scalar (a read-only 0-d array that is no explicit array), as <c>np.array(np.float64(1.5))</c> is.</returns>
        /// <exception cref="NullReferenceException"><paramref name="nd"/> is <c>null</c>.</exception>
        /// <remarks>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.array.html</para>
        ///     <para>Until parity QA-35 the copy was a storage clone that kept the source's flags: a read-only C, F or 0-d
        ///     source gave a READ-ONLY copy (a write raised "assignment destination is read-only"), and a broadcast source
        ///     came back C-ordered where NumPy's K-order copy is F-contiguous.</para>
        /// </remarks>
        [MethodImpl(OptimizeAndInline)]
        public static NDArray array(NDArray nd, bool copy = true)
        {
            // PyArray_NewCopy(a, NPY_KEEPORDER): a fresh owner, writeable, in the iterator's K order - exactly np.copy's
            // copy (which also keeps the array's engine). `np.` qualifies the call: the bool parameter shadows the name.
            var result = copy
                ? np.copy(nd, 'K')
                : new NDArray(nd.Storage) { TensorEngine = nd.TensorEngine };
            // The mark is per object, so a new object must receive it explicitly. A copy of a NumPy SCALAR (a read-only 0-d
            // array that is no explicit array: a reduction's result, NDArray.Scalar(...)) is an explicit 0-d ARRAY too:
            // np.array(np.float64(1.5)) is array(1.5), which np.isscalar rejects (probed 2.4.2).
            bool explicitResult = nd.IsExplicitZeroDimArray || (copy && nd.IsNumPyScalar && !nd.Shape.IsWriteable);
            return explicitResult ? result.MarkExplicitZeroDimArray() : result;
        }

        /// <summary>
        ///     Create an array from a <see cref="MemoryView"/> (obtained from <see cref="NDArray.data"/>) —
        ///     the consumer round-trip of <c>ndarray.data</c>. Matches NumPy's <c>np.array(a.data)</c>,
        ///     which reads the buffer's shape/strides/dtype and (by default) COPIES, preserving the
        ///     source's N-D shape and layout. Equivalent to <c>np.array(buffer.obj, copy)</c>; a copy is
        ///     always writeable, even from a read-only (broadcast) source.
        /// </summary>
        /// <param name="buffer">A <see cref="MemoryView"/> over an array.</param>
        /// <param name="copy">When <c>true</c> (default) the source storage is cloned; when <c>false</c> the storage is shared (alias).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        [MethodImpl(OptimizeAndInline)]
        public static NDArray array(MemoryView buffer, bool copy = true)
        {
            if (buffer is null)
                throw new ArgumentNullException(nameof(buffer));
            return array(buffer.obj, copy);
        }

        /// <summary>
        ///     Creates a scalar (0-dimensional) <see cref="NDArray"/> from a single value.
        /// </summary>
        /// <typeparam name="T">The type of the value, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="scalar">The scalar value.</param>
        /// <returns>A 0-dimensional <see cref="NDArray"/> containing the scalar value.</returns>
        /// <remarks>
        ///     <para>
        ///     NumPy: np.array(42) creates a 0-dimensional array with shape ().
        ///     https://numpy.org/doc/stable/reference/generated/numpy.array.html
        ///     </para>
        ///     <para>
        ///     The result is marked as an explicit 0-d ARRAY (<see cref="NDArray.IsExplicitZeroDimArray"/>), which
        ///     one operator reads: <c>matrix * np.array(2)</c> is the product with a 1 x 1 matrix, as in NumPy, and
        ///     raises unless the matrix has one column. Use <see cref="NDArray.Scalar{T}(T)"/> or a C# number for a
        ///     value that must SCALE a matrix; library code that builds a computed scalar must use
        ///     <see cref="NDArray.Scalar{T}(T)"/> too.
        ///     </para>
        /// </remarks>
        [MethodImpl(OptimizeAndInline)]
        public static NDArray array<T>(T scalar) where T : unmanaged => NDArray.WritableScalar(scalar).MarkExplicitZeroDimArray();

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from an array with an unknown size or dtype.
        /// </summary>
        /// <param name="ndmin">Specifies the minimum number of dimensions that the resulting array should have. Ones will be pre-pended to the shape as needed to meet this requirement.</param>
        /// <param name="copy">Always copies if the array is larger than 1-d.</param>
        /// <param name="order">Memory layout: 'C' (row-major, default), 'F' (column-major), 'A'/'K' (resolved from source).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        [MethodImpl(Optimize)]
        [SuppressMessage("ReSharper", "InvalidXmlDocComment")]
        public static NDArray array(Array array, DType dtype = null, int ndmin = 0, bool copy = true, char order = 'C')
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            return NDArray.array(array, dtype, ndmin, copy, order);
        }

        /// <summary>
        ///     Creates a Vector <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html <br></br>Always performs a copy.</remarks>
        public static NDArray array<T>(params T[] data) where T : unmanaged => new NDArray(ArraySlice.FromArray(data, true), Shape.Vector(data.Length));

        /// <summary>
        ///     Creates a Vector <see cref="NDArray"/> from given <paramref name="data"/> with specified dtype.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="dtype">
        ///     The desired dtype for the array — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>, an
        ///     <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert implicitly.
        ///     If different from <typeparamref name="T"/>, the data will be cast.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data cast to the specified dtype.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[] data, DType dtype) where T : unmanaged
        {
            return NDArray.array(data, dtype);
        }

        /// <summary>
        ///     Creates a Vector <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The enumeration of data to create <see cref="NDArray"/> from.</param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html <br></br>Always performs a copy.</remarks>
        public static NDArray array<T>(IEnumerable<T> data) where T : unmanaged
        {
            var slice = ArraySlice.FromArray(data.ToArray(), false);
            return new NDArray(slice, Shape.Vector(slice.Count));
        }

        /// <summary>
        ///     Creates a vector of exactly <paramref name="size"/> elements, the first <paramref name="size"/> items
        ///     of <paramref name="data"/>: NumPy's <c>np.fromiter(data, dtype=T, count=size)</c>. Use it to read a
        ///     known number of items from a long or endless sequence; an enumeration that holds fewer items is an
        ///     error, not a shorter array.
        /// </summary>
        /// <typeparam name="T">The element type, one of NumSharp's dtypes; the result's dtype.</typeparam>
        /// <param name="data">The enumeration to read. It is enumerated once, and at most <paramref name="size"/>
        ///     items are taken from it (<c>MoveNext</c> is called <paramref name="size"/> times when it holds that
        ///     many).</param>
        /// <param name="size">The number of items to read, and the result's length.</param>
        /// <returns>A fresh 1-D array of <paramref name="size"/> elements.</returns>
        /// <exception cref="ValueError"><paramref name="data"/> holds fewer than <paramref name="size"/> items
        ///     (NumPy's text, <c>iterator too short: Expected 5 but iterator had only 2 items.</c>), or
        ///     <paramref name="size"/> is negative (<c>negative dimensions are not allowed</c>).</exception>
        /// <exception cref="NullReferenceException"><paramref name="data"/> is null.</exception>
        /// <remarks>See <see cref="array{T}(IEnumerable{T}, long)"/>, which this overload calls.</remarks>
        public static NDArray array<T>(IEnumerable<T> data, int size) where T : unmanaged
            => array(data, (long)size);

        /// <summary>
        ///     Creates a vector of exactly <paramref name="size"/> elements, the first <paramref name="size"/> items
        ///     of <paramref name="data"/>: NumPy's <c>np.fromiter(data, dtype=T, count=size)</c>. Use it to read a
        ///     known number of items from a long or endless sequence; an enumeration that holds fewer items is an
        ///     error, not a shorter array.
        /// </summary>
        /// <typeparam name="T">The element type, one of NumSharp's dtypes; the result's dtype.</typeparam>
        /// <param name="data">The enumeration to read. It is enumerated once, and at most <paramref name="size"/>
        ///     items are taken from it (<c>MoveNext</c> is called <paramref name="size"/> times when it holds that
        ///     many).</param>
        /// <param name="size">The number of items to read, and the result's length.</param>
        /// <returns>A fresh 1-D array of <paramref name="size"/> elements.</returns>
        /// <exception cref="ValueError"><paramref name="data"/> holds fewer than <paramref name="size"/> items
        ///     (NumPy's text, <c>iterator too short: Expected 5 but iterator had only 2 items.</c>), or
        ///     <paramref name="size"/> is negative (<c>negative dimensions are not allowed</c>).</exception>
        /// <exception cref="NullReferenceException"><paramref name="data"/> is null.</exception>
        /// <remarks>
        ///     <para>
        ///         NumPy's <c>PyArray_FromIter</c> (<c>ctors.c</c>): the array of <c>count</c> elements is allocated
        ///         first, one <c>PyIter_Next</c> fills each element and none runs past <c>count</c> (a longer iterator
        ///         keeps its remaining items), and an iterator that ends early raises the <c>ValueError</c> above. An
        ///         exception the enumeration throws propagates unchanged, as an iterator's error does in NumPy. Either
        ///         way the buffer is released before the exception leaves, so a failed call holds no memory.
        ///     </para>
        ///     <para>
        ///         Until 2026-10-30 a short enumeration returned an array of <paramref name="size"/> elements whose
        ///         tail was never written (heap contents).
        ///     </para>
        ///     <para>
        ///         Not NumPy's: a negative <paramref name="size"/> is refused. NumPy's <c>count=-1</c> reads every
        ///         item, which is <see cref="array{T}(IEnumerable{T})"/> here, and its other negative counts return an
        ///         empty array without reading.
        ///     </para>
        /// </remarks>
        public static NDArray array<T>(IEnumerable<T> data, long size) where T : unmanaged
        {
            var slice = new ArraySlice<T>(new UnmanagedMemoryBlock<T>(size));
            // The items read so far; fewer than `size` once the enumeration ends is NumPy's "iterator too short".
            long i = 0;
            bool filled = false;
            try
            {
                unsafe
                {
                    using (var enumerator = data.GetEnumerator())
                    {
                        Func<bool> next = enumerator.MoveNext;

                        var addr = slice.Address;
                        // `i < size` is tested first, so a longer enumeration is advanced exactly `size` times.
                        for (; i < size && next(); i++)
                        {
                            addr[i] = enumerator.Current;
                        }
                    }
                }

                if (i < size)
                    throw new ValueError(string.Create(CultureInfo.InvariantCulture, $"iterator too short: Expected {size} but iterator had only {i} items."));
                filled = true;
            }
            finally
            {
                // Cleanup on failure: no NDArray holds the slice yet, so its only reference is this one. Freeing it
                // here returns the buffer to the pool now rather than leaving it to the block's finalizer.
                if (!filled)
                    slice.DangerousFree();
            }

            return new NDArray(slice, Shape.Vector(slice.Count));
        }

        /// <summary>
        ///     Creates a 0-d fixed-width Unicode array holding <paramref name="value"/> — NumPy's <c>np.array('abc')</c>,
        ///     <c>array('abc', dtype='&lt;U3')</c> with shape <c>()</c>. The width is the value's length in code points (a surrogate
        ///     pair counts once); an empty value is <c>&lt;U1</c>, NumPy's minimum.
        /// </summary>
        /// <param name="value">The text.</param>
        /// <returns>A 0-d <c>U</c> array, marked an explicit ARRAY (<see cref="NDArray.IsExplicitZeroDimArray"/>): NumPy's
        ///     ndarray, not an <c>np.str_</c> scalar.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        /// <remarks>
        ///     <b>Breaking (NumPy parity):</b> this overload used to build a 1-D <see cref="char"/> vector of the characters.
        ///     That conversion remains available as <see cref="NDArray.FromString(string)"/> and
        ///     <c>np.array(value.ToCharArray())</c>.
        ///     <para>The result is an ndarray as <c>np.array('x')</c> is (API-parity batch 12's finding, fixed in batch 13):
        ///     <c>np.isscalar(np.array("x"))</c> is False, and a reader of NumPy's scalar / array difference (np.interp's
        ///     <c>float()</c> of a fill value, the scalar-type <c>isinstance</c> rule) sees the array NumPy has. A C# string
        ///     LITERAL converted implicitly stays NumPy's Python <c>str</c> (<see cref="NDArray.WasPythonStr"/>).</para>
        /// </remarks>
        [MethodImpl(Inline)]
        public static NDArray array(string value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            // np.array's other scalar forms mark their 0-d result an explicit array (np.array<T>(T)); the text form is the
            // same call in NumPy, so it carries the same mark.
            return UnmanagedStorage.FromAnyScalar(value, null).MarkExplicitZeroDimArray();
        }

        /// <summary>
        ///     Creates a 0-d fixed-width string array holding <paramref name="value"/> with the given string dtype —
        ///     NumPy's <c>np.array('abc', dtype='U5')</c> / <c>dtype='S'</c>. Under the character dtype
        ///     <c>np.dtype("c")</c> a text whose length is not 1 is a sequence, as NumPy reads it: <c>np.array("abc",
        ///     np.dtype("c"))</c> is the (3,) <c>|S1</c> array of its characters.
        /// </summary>
        /// <param name="value">The text.</param>
        /// <param name="dtype">A string dtype (<c>"U"</c>, <c>"U5"</c>, <c>"S"</c>, <c>"c"</c>, <c>np.str_</c>,
        ///     <c>np.bytes_</c>) or null to discover.</param>
        /// <returns>A 0-d <c>S</c>/<c>U</c> array, or the 1-D character array under <c>"c"</c> (0-d for one character); a
        ///     0-d result is marked an explicit ARRAY (<see cref="NDArray.IsExplicitZeroDimArray"/>), NumPy's ndarray.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        /// <exception cref="ValueError">A numeric dtype and text that does not parse as that number (<c>could not convert
        ///     string to float: 'x'</c>, NumPy's <c>int()</c> / <c>float()</c> texts).</exception>
        /// <exception cref="OverflowError">An integer text outside an integer dtype's range.</exception>
        public static NDArray array(string value, DType dtype)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            // An explicit 0-d array as np.array's other forms make (MarkExplicitZeroDimArray leaves a 1-D result unmarked).
            return UnmanagedStorage.FromLoneText(value, dtype).MarkExplicitZeroDimArray();
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/> — NumPy's <c>np.array</c> of nested lists:
        ///     every row must have the same length.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from. Its shape is NumPy's discovery
        ///     (<see cref="NDArray.DiscoverJaggedShape"/>): every row is checked, and no rows at all gives shape <c>(0,)</c>.</param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <exception cref="ValueError">A ragged or null row (NumPy's <c>setting an array element with a sequence. The
        ///     requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous
        ///     part.</c>): a longer row used to be cut silently and a shorter one to throw IndexOutOfRange (parity QA-45).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html <br></br>Always performs a copy.</remarks>
        [SuppressMessage("ReSharper", "SuggestVarOrType_SimpleTypes")]
        [SuppressMessage("ReSharper", "SuggestVarOrType_Elsewhere")]
        [MethodImpl(Optimize)]
        public static NDArray array<T>(T[][] data) where T : unmanaged
        {
            // NumPy's discovery first: a ragged input raises before anything is allocated; an empty level ends the dims.
            long[] dims = NDArray.DiscoverJaggedShape(data);
            if (dims.Length < 2)
                return new NDArray(InfoOf<T>.NPTypeCode, new Shape(dims, copy: false));
            unsafe
            {
                int len1 = data.Length;
                int len2 = data[0].Length;

                NDArray @out = new NDArray(InfoOf<T>.NPTypeCode, new Shape(len1, len2));

                var strides = @out.Shape.strides;
                long stride1 = strides[0];
                Debug.Assert(strides[1] == 1);

                T* addr = (T*)@out.Address;
                for (int i1 = 0; i1 < len1; i1++)
                {
                    T* addr1 = addr + i1 * stride1;
                    T[] src1 = data[i1];
                    for (int i2 = 0; i2 < len2; i2++)
                    {
                        *(addr1 + i2) = src1[i2];
                    }
                }

                return @out;
            }
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/> — NumPy's <c>np.array</c> of nested lists:
        ///     every child at every level must have its level's length.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from. Its shape is NumPy's discovery
        ///     (<see cref="NDArray.DiscoverJaggedShape"/>): every child is checked, and the dims end at the first empty level
        ///     (<c>[[], []]</c> is <c>(2, 0)</c>).</param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <exception cref="ValueError">A ragged or null child (NumPy's inhomogeneous-shape text, naming the depth the
        ///     children agreed to).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html <br></br>Always performs a copy.</remarks>
        [SuppressMessage("ReSharper", "SuggestVarOrType_SimpleTypes")]
        [SuppressMessage("ReSharper", "SuggestVarOrType_Elsewhere")]
        [MethodImpl(Optimize)]
        public static NDArray array<T>(T[][][] data) where T : unmanaged
        {
            // NumPy's discovery first: a ragged input raises before anything is allocated; an empty level ends the dims.
            long[] dims = NDArray.DiscoverJaggedShape(data);
            if (dims.Length < 3)
                return new NDArray(InfoOf<T>.NPTypeCode, new Shape(dims, copy: false));
            unsafe
            {
                int len1 = data.Length;
                int len2 = data[0].Length;
                int len3 = data[0][0].Length;

                NDArray @out = new NDArray(InfoOf<T>.NPTypeCode, new Shape(len1, len2, len3));

                var strides = @out.Shape.strides;
                long stride1 = strides[0];
                long stride2 = strides[1];
                Debug.Assert(strides[2] == 1);

                T* addr = (T*)@out.Address;
                for (int i1 = 0; i1 < len1; i1++)
                {
                    T* addr1 = addr + i1 * stride1;
                    T[][] src1 = data[i1];
                    for (int i2 = 0; i2 < len2; i2++)
                    {
                        T* addr2 = addr1 + i2 * stride2;
                        T[] src2 = src1[i2];
                        for (int i3 = 0; i3 < len3; i3++)
                        {
                            *(addr2 + i3) = src2[i3];
                        }
                    }
                }

                return @out;
            }
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/> — NumPy's <c>np.array</c> of nested lists:
        ///     every child at every level must have its level's length.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from. Its shape is NumPy's discovery
        ///     (<see cref="NDArray.DiscoverJaggedShape"/>): every child is checked, and the dims end at the first empty
        ///     level.</param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <exception cref="ValueError">A ragged or null child (NumPy's inhomogeneous-shape text, naming the depth the
        ///     children agreed to).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html <br></br>Always performs a copy.</remarks>
        [SuppressMessage("ReSharper", "SuggestVarOrType_SimpleTypes")]
        [SuppressMessage("ReSharper", "SuggestVarOrType_Elsewhere")]
        [MethodImpl(Optimize)]
        public static NDArray array<T>(T[][][][] data) where T : unmanaged
        {
            // NumPy's discovery first: a ragged input raises before anything is allocated; an empty level ends the dims.
            long[] dims = NDArray.DiscoverJaggedShape(data);
            if (dims.Length < 4)
                return new NDArray(InfoOf<T>.NPTypeCode, new Shape(dims, copy: false));
            unsafe
            {
                int len1 = data.Length;
                int len2 = data[0].Length;
                int len3 = data[0][0].Length;
                int len4 = data[0][0][0].Length;

                NDArray @out = new NDArray(InfoOf<T>.NPTypeCode, new Shape(len1, len2, len3, len4));

                var strides = @out.Shape.strides;
                long stride1 = strides[0];
                long stride2 = strides[1];
                long stride3 = strides[2];
                Debug.Assert(strides[3] == 1);

                T* addr = (T*)@out.Address;
                for (int i1 = 0; i1 < len1; i1++)
                {
                    T* addr1 = addr + i1 * stride1;
                    T[][][] src1 = data[i1];
                    for (int i2 = 0; i2 < len2; i2++)
                    {
                        T* addr2 = addr1 + i2 * stride2;
                        T[][] src2 = src1[i2];
                        for (int i3 = 0; i3 < len3; i3++)
                        {
                            T* addr3 = addr2 + i3 * stride3;
                            T[] src3 = src2[i3];
                            for (int i4 = 0; i4 < len4; i4++)
                            {
                                *(addr3 + i4) = src3[i4];
                            }
                        }
                    }
                }

                return @out;
            }
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/> — NumPy's <c>np.array</c> of nested lists:
        ///     every child at every level must have its level's length.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from. Its shape is NumPy's discovery
        ///     (<see cref="NDArray.DiscoverJaggedShape"/>): every child is checked, and the dims end at the first empty
        ///     level.</param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <exception cref="ValueError">A ragged or null child (NumPy's inhomogeneous-shape text, naming the depth the
        ///     children agreed to).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html <br></br>Always performs a copy.</remarks>
        [SuppressMessage("ReSharper", "SuggestVarOrType_SimpleTypes")]
        [SuppressMessage("ReSharper", "SuggestVarOrType_Elsewhere")]
        [MethodImpl(Optimize)]
        public static NDArray array<T>(T[][][][][] data) where T : unmanaged
        {
            // NumPy's discovery first: a ragged input raises before anything is allocated; an empty level ends the dims.
            long[] dims = NDArray.DiscoverJaggedShape(data);
            if (dims.Length < 5)
                return new NDArray(InfoOf<T>.NPTypeCode, new Shape(dims, copy: false));
            unsafe
            {
                int len1 = data.Length;
                int len2 = data[0].Length;
                int len3 = data[0][0].Length;
                int len4 = data[0][0][0].Length;
                int len5 = data[0][0][0][0].Length;

                NDArray @out = new NDArray(InfoOf<T>.NPTypeCode, new Shape(len1, len2, len3, len4, len5));

                var strides = @out.Shape.strides;
                long stride1 = strides[0];
                long stride2 = strides[1];
                long stride3 = strides[2];
                long stride4 = strides[3];
                Debug.Assert(strides[4] == 1);

                T* addr = (T*)@out.Address;
                for (int i1 = 0; i1 < len1; i1++)
                {
                    T* addr1 = addr + i1 * stride1;
                    T[][][][] src1 = data[i1];
                    for (int i2 = 0; i2 < len2; i2++)
                    {
                        T* addr2 = addr1 + i2 * stride2;
                        T[][][] src2 = src1[i2];
                        for (int i3 = 0; i3 < len3; i3++)
                        {
                            T* addr3 = addr2 + i3 * stride3;
                            T[][] src3 = src2[i3];
                            for (int i4 = 0; i4 < len4; i4++)
                            {
                                T* addr4 = addr3 + i4 * stride4;
                                T[] src4 = src3[i4];
                                for (int i5 = 0; i5 < len5; i5++)
                                {
                                    *(addr4 + i5) = src4[i5];
                                }
                            }
                        }
                    }
                }

                return @out;
            }
        }

        // %l = "data.GetLength("
        // %r = ")"
        // %foreach range(1,16)%

        // /// <summary>
        // ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        // /// </summary>
        // /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        // /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        // /// <param name="copy">
        // ///     If true then the array will be copied to a newly allocated memory.<br></br>
        // ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        // /// </param>
        // /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        // /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        // public static NDArray array<T>(T[#(repeat(",", i))] data, bool copy = true) where T : unmanaged
        // {
            // if (data == null)
                // throw new ArgumentNullException(nameof(data));

            // return new NDArray(ArraySlice.FromArray(data, copy), new Shape(#(repeat("^l+str(n.ToString())+r", i+1, ", "))));
        // }
        // %

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[] data, bool copy) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.Length));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/> with specified dtype.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="dtype">
        ///     The desired dtype for the array — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>, an
        ///     <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert implicitly.
        ///     If different from <typeparamref name="T"/>, the data will be cast.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data cast to the specified dtype.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,] data, DType dtype) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return NDArray.array(data, dtype);
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9), data.GetLength(10)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9), data.GetLength(10), data.GetLength(11)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9), data.GetLength(10), data.GetLength(11), data.GetLength(12)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9), data.GetLength(10), data.GetLength(11), data.GetLength(12), data.GetLength(13)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9), data.GetLength(10), data.GetLength(11), data.GetLength(12), data.GetLength(13), data.GetLength(14)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9), data.GetLength(10), data.GetLength(11), data.GetLength(12), data.GetLength(13), data.GetLength(14), data.GetLength(15)));
        }
    }
}
