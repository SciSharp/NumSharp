using System;
using System.Collections;
using System.Numerics;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyInterpolate.cs — chebinterpolate: interpolation at the Chebyshev points of the first kind
// =============================================================================
//
// NumPy 2.4.2 (numpy/polynomial/chebyshev.py), statement for statement:
//
//   def chebinterpolate(func, deg, args=()):
//       deg = np.asarray(deg)
//       if deg.ndim > 0 or deg.dtype.kind not in 'iu' or deg.size == 0:
//           raise TypeError("deg must be an int")
//       if deg < 0:
//           raise ValueError("expected deg >= 0")
//       order = deg + 1
//       xcheb = chebpts1(order)
//       yfunc = func(xcheb, *args)
//       m = chebvander(xcheb, deg)
//       c = np.dot(m.T, yfunc)
//       c[0] /= order
//       c[1:] /= 0.5 * order
//       return c
//
// ROUTE. A composition of engines that are already NumPy's: chebpts1 (unit 2's emitted fill), chebvander (U5's kernel),
// np.dot (the OpenBLAS backend's port of cblas_matrixproduct when it is installed — gemv for a vector y, gemm for a matrix
// one, dotfunc for an N-D one — the managed product otherwise) and two in-place np.divide calls. NumPy's time goes into
// chebvander's Python loop over the degree, np.dot's argument conversion and the two divisions; here those are one kernel
// call, the backend's call and two ufunc calls. SPEED: see benchmark/polynomial/polyinterp_results.md.
//
// THE ARGUMENTS, all probed against NumPy 2.4.2:
//   * deg is `np.asarray(deg)`: an integer of any width (a C# integer or BigInteger is a Python int: int64, or uint64 from
//     2**63; past 2**64 it is an OBJECT array), a char (np.uint16) or a 0-d integer NDArray. A bool, a float (even an
//     integral one), a complex, a str, None, any sequence (even [3]) or sized array is "deg must be an int" — after
//     np.asarray itself, which raises its inhomogeneous-shape ValueError for a ragged sequence first.
//   * `order = deg + 1` is a ufunc add on the 0-d array, which WRAPS in deg's dtype: np.uint8(255), np.int8(127),
//     np.uint64(2**64 - 1) and 2**63 - 1 (int64) all give an order chebpts1 rejects ("npts must be >= 1"); 2**63 (uint64)
//     gives 2**63 + 1, a count in np.arange's 2**63 band, so chebpts1 returns an EMPTY array, func is CALLED with it, and
//     chebvander then raises "Maximum allowed dimension exceeded".
//   * `func(xcheb, *args)` unpacks args with Python's `*`: any iterable (a tuple, a list, an NDArray along its first axis,
//     a str as one-character strs, a dict as its keys); a 0-d array is the TypeError "iteration over a 0-d array" and any
//     other object "Value after * must be an iterable, not {type}". The unpacking runs before the call, so it is reported
//     before a null func ("'NoneType' object is not callable").
//   * yfunc is anything np.dot accepts: an NDArray, a Python list / tuple (np.array's coercion: a ragged one is its
//     inhomogeneous-shape ValueError), a typed C# array, a scalar (np.dot of a 0-d operand is a multiply, so c becomes the
//     (deg + 1, deg + 1) matrix m.T * y with row 0 divided by order) or a str (np.dot's own error for text). c's dtype is
//     np.dot's: float64 for every real input, complex128 for a complex one.
//   * `c[0] /= order` divides by the integer order and `c[1:] /= 0.5 * order` by a float64; for a complex c both run
//     NumPy's CDOUBLE_divide (Smith's algorithm, a multiplication by the reciprocal of the real divisor), as the house
//     np.divide does — the scalar path of a 1-D c (`np.complex128 / np.int64`) calls that same loop in NumPy.
//
// NOT REPRODUCED (NotSupportedException): a func result NumPy turns into an OBJECT array (None, a dict, a list holding
// None or a Python int past uint64) — NumPy computes np.dot with Python objects there — and a deg that is an object array
// (a Python int past uint64 or below -2**63) raises NumPy's TypeError, not this. A numpy.matrix result is computed as a
// plain array (NumPy returns a matrix): the numpy.polynomial package computes plain throughout.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     <c>chebinterpolate</c>: NumPy 2.4.2's statements (see the file header), composed from chebpts1, chebvander, np.dot and
    ///     np.divide.
    /// </summary>
    internal static class NDPolyInterpolate
    {
        /// <summary>
        ///     <c>chebinterpolate(func, deg, args=())</c>: the Chebyshev series of degree <c>deg</c> that interpolates
        ///     <paramref name="func"/> at the <c>deg + 1</c> Chebyshev points of the first kind.
        /// </summary>
        /// <param name="func">The function, called once as <c>func(xcheb, args)</c> with the points (a new float64 array of
        ///     <c>deg + 1</c> elements) and the unpacked <paramref name="args"/>; it returns anything <c>np.dot</c> accepts.</param>
        /// <param name="deg">The degree: an integer (see the file header for the accepted kinds).</param>
        /// <param name="args">The extra arguments: any Python iterable under the house mapping, or null for NumPy's <c>()</c>.</param>
        /// <returns><c>c</c>: np.dot's new array (float64 or complex128), of shape <c>(deg + 1,)</c> for a vector result,
        ///     <c>(deg + 1,) + y.shape[:-2] + y.shape[-1:]</c> for an N-D one, <c>(deg + 1, deg + 1)</c> for a scalar one.</returns>
        /// <exception cref="TypeError"><c>deg must be an int</c>; <c>iteration over a 0-d array</c> or <c>Value after * must be
        ///     an iterable, not …</c> (args); <c>'NoneType' object is not callable</c> (a null func).</exception>
        /// <exception cref="ValueError"><c>expected deg &gt;= 0</c>; chebpts1's <c>npts must be &gt;= 1</c> (an order that wrapped
        ///     to zero or below), <c>array is too big; …</c>; chebvander's <c>Maximum allowed dimension exceeded</c>; np.asarray's
        ///     inhomogeneous-shape text (a ragged deg or result); np.dot's <c>shapes … not aligned</c> (a result of the wrong
        ///     length).</exception>
        /// <exception cref="ArgumentException">np.dot's alignment errors, where NumSharp's product raises them as ArgumentException.</exception>
        /// <exception cref="NotSupportedException">A func result NumPy computes with as Python objects, or a deg past the
        ///     integer dtypes (see the file header).</exception>
        /// <exception cref="OutOfMemoryException">A degree whose matrix the machine cannot hold (NumPy's MemoryError).</exception>
        /// <remarks>Every exception <paramref name="func"/> throws propagates unchanged.</remarks>
        public static NDArray ChebInterpolate(Func<NDArray, object[], object> func, object deg, object args)
        {
            // deg = np.asarray(deg), then NumPy's two checks.
            BigInteger value = IntegerDegree(deg, out NPTypeCode degType);
            if (value.Sign < 0)
                throw new ValueError("expected deg >= 0");

            // order = deg + 1 (the 0-d array's add, which wraps in deg's dtype); xcheb = chebpts1(order). chebpts1 reads its
            // argument through int() and `_npts != npts`, which see only the integer, so the Python int stands for the
            // NumPy scalar.
            BigInteger order = AddOneInDtype(value, degType);
            NDArray xcheb = NDPolyChebPoints.Pts1(order);

            // yfunc = func(xcheb, *args): the unpacking runs before the call, so its TypeErrors come before a null func's.
            object[] unpacked = StarArgs(args);
            if (func is null)
                throw new TypeError("'NoneType' object is not callable");
            object yfunc = func(xcheb, unpacked);

            // m = chebvander(xcheb, deg). A caller's 0-d array is passed itself — NumPy reads it again here, after func ran —
            // anything else as its integer (np.asarray made a new array of it). chebvander uses the points func was handed.
            NDArray m = NDPolyVander.Vander(PolyBasis.Chebyshev, xcheb, deg is NDArray ? deg : value);

            // c = np.dot(m.T, yfunc). m is np.moveaxis's view of chebvander's (deg + 1, n) buffer, so m.T is that buffer:
            // C-contiguous, as in NumPy, and np.dot sees the same operand layout. A numpy.matrix result is computed plain
            // (the region keeps np.dot's matrix rule out: the numpy.polynomial package computes plain).
            NDArray y = DotOperand(yfunc);
            NDArray c;
            using (NDMatrixBoundary.PlainFor(y))
                c = np.dot(m.T, y);

            // c[0] /= order; c[1:] /= 0.5 * order — in place through views. The divisors' VALUES are what NumPy divides by:
            // the integer converted to the loop dtype (exact for any order a matrix of order**2 elements can have), and the
            // product 0.5 * order computed in double (a Python float times a NumPy integer is float64).
            double d0 = (double)order;
            NDArray head = c[0];
            np.divide(head, NDArray.Scalar(d0), @out: head);
            NDArray tail = c[new Slice(1, null)];
            np.divide(tail, NDArray.Scalar(0.5 * d0), @out: tail);
            return c;
        }

        /// <summary>
        ///     <c>np.asarray(deg)</c> and NumPy's <c>deg.ndim &gt; 0 or deg.dtype.kind not in 'iu'</c> test: the integer and its
        ///     dtype for an argument chebinterpolate accepts.
        /// </summary>
        /// <param name="deg">The argument.</param>
        /// <param name="dtype">The 0-d array's dtype: int64 (or uint64 from 2**63) for a Python int, uint16 for a char, the
        ///     array's own for a 0-d NDArray.</param>
        /// <returns>The integer.</returns>
        /// <exception cref="TypeError"><c>deg must be an int</c>: a bool, a float, a complex, a str, None, a sequence, a sized
        ///     array, an integer past the integer dtypes (NumPy's object array) or any other object.</exception>
        /// <exception cref="ValueError">A ragged sequence (np.asarray's inhomogeneous-shape text, raised before the test).</exception>
        private static BigInteger IntegerDegree(object deg, out NPTypeCode dtype)
        {
            switch (deg)
            {
                case NDArray nd:
                    // A 0-d INTEGER array (char included: NumSharp's uint16-like dtype). A bool array is kind 'b'.
                    if (nd.ndim > 0 || nd.typecode == NPTypeCode.Boolean || !PolyTyping.IsIntLike(nd.typecode))
                        throw DegMustBeAnInt();
                    dtype = nd.typecode;
                    return PolyIndexArgument.AsInt(nd, "deg");
                case bool:
                    throw DegMustBeAnInt();   // np.asarray(True) is a bool array
                case char ch:
                    dtype = NPTypeCode.UInt16;   // np.uint16, NumSharp's char (the house map)
                    return ch;
                case sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger:
                {
                    // A Python int: np.asarray makes int64, uint64 past int64, and an OBJECT array past uint64 (or below int64).
                    BigInteger v = PolyIndexArgument.AsInt(deg, "deg");
                    if (v >= long.MinValue && v <= long.MaxValue)
                        dtype = NPTypeCode.Int64;
                    else if (v.Sign > 0 && v <= ulong.MaxValue)
                        dtype = NPTypeCode.UInt64;
                    else
                        throw DegMustBeAnInt();
                    return v;
                }
            }
            // np.asarray of a sequence walks it first: a ragged one is its ValueError, any other has at least one dim.
            if (PolySequence.IsSequence(deg))
                PolySequence.NDim(deg);
            // Everything else — a float / complex / str / None (0-d arrays of another kind), a typed C# array (an ndarray of
            // one or more dims), a sequence, any other object (a 0-d object array) — fails the test.
            throw DegMustBeAnInt();
        }

        /// <summary>NumPy's <c>TypeError("deg must be an int")</c>.</summary>
        /// <returns>The exception, for the caller to throw.</returns>
        private static TypeError DegMustBeAnInt() => new TypeError("deg must be an int");

        /// <summary>
        ///     <c>deg + 1</c> on a 0-d integer array of <paramref name="dtype"/>: the ufunc add, whose integer overflow wraps
        ///     into the dtype's range (two's complement).
        /// </summary>
        /// <param name="value">deg's integer (in the dtype's range).</param>
        /// <param name="dtype">deg's integer dtype.</param>
        /// <returns><c>value + 1</c>, wrapped.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="dtype"/> is not an integer dtype.</exception>
        private static BigInteger AddOneInDtype(BigInteger value, NPTypeCode dtype)
        {
            (int bits, bool signed) = dtype switch
            {
                NPTypeCode.SByte => (8, true),
                NPTypeCode.Byte => (8, false),
                NPTypeCode.Int16 => (16, true),
                NPTypeCode.UInt16 or NPTypeCode.Char => (16, false),
                NPTypeCode.Int32 => (32, true),
                NPTypeCode.UInt32 => (32, false),
                NPTypeCode.Int64 => (64, true),
                NPTypeCode.UInt64 => (64, false),
                _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "an integer dtype"),
            };
            BigInteger modulus = BigInteger.One << bits;
            BigInteger r = (value + 1) % modulus;
            if (r.Sign < 0)
                r += modulus;
            // The upper half of the unsigned range is the signed dtype's negative half.
            if (signed && r >= modulus >> 1)
                r -= modulus;
            return r;
        }

        /// <summary>
        ///     Python's <c>*args</c> unpacking (<c>list(args)</c>) under the house mapping.
        /// </summary>
        /// <param name="args">The argument: null (NumPy's default <c>()</c>), an <see cref="NDArray"/> (iterated along its first
        ///     axis, as NumSharp's own enumerator does), a typed C# array (an ndarray: the same), a string (its characters, as
        ///     one-character strings), a dictionary (its keys), a tuple, an array of references, or any other enumerable.</param>
        /// <returns>The items, in order.</returns>
        /// <exception cref="TypeError">A 0-d array (<c>iteration over a 0-d array</c>, its iterator's error), or an object that
        ///     is not iterable (<c>Value after * must be an iterable, not {type}</c>, CPython's text).</exception>
        internal static object[] StarArgs(object args)
        {
            switch (args)
            {
                case null:
                    return Array.Empty<object>();
                case NDArray nd:
                    return Enumerate(nd);
                case string s:
                {
                    var chars = new object[s.Length];
                    for (int i = 0; i < chars.Length; i++)
                        chars[i] = s[i].ToString();
                    return chars;
                }
                case IDictionary dict:
                {
                    var keys = new object[dict.Count];
                    dict.Keys.CopyTo(keys, 0);
                    return keys;
                }
            }
            // A typed C# array (or Memory<T>) of a dtype is an ndarray: iterated as one, so a 2-D array yields row arrays.
            if (PolySequence.IsArrayLike(args))
            {
                // The conversion is this call's intermediate; its rows (views) are handed to func and stay alive through
                // their own references.
                var nd = np.asanyarray(args);
                return Enumerate(nd);
            }
            // A multi-dimensional object[,] is a Python list of its rows: hand func each row as a plain object[] list (the
            // coercion's own row type is internal to it).
            if (args is Array grid && grid.Rank > 1)
                return RowsOf(grid, new int[grid.Rank], 0);
            if (PolySequence.IsSequence(args))
                return PolySequence.Items(args);
            throw new TypeError($"Value after * must be an iterable, not {NDPolySeries.PythonTypeName(args)}");
        }

        /// <summary>
        ///     Python's <c>list(arr)</c> through NumSharp's enumerator: the elements of a 1-D array, the sub-arrays (views) of an
        ///     N-D one.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <returns>The items.</returns>
        /// <exception cref="TypeError">A 0-d array (<c>iteration over a 0-d array</c>).</exception>
        private static object[] Enumerate(NDArray nd)
        {
            var list = new System.Collections.Generic.List<object>();
            foreach (object o in nd)
                list.Add(o);
            return list.ToArray();
        }

        /// <summary>
        ///     The items of a multi-dimensional array along dimension <paramref name="dim"/> (the leading indices fixed by
        ///     <paramref name="index"/>): its elements on the last dimension, nested <c>object[]</c> rows above it — Python's
        ///     list of lists.
        /// </summary>
        /// <param name="a">The array (rank ≥ 2).</param>
        /// <param name="index">The index buffer (one entry per dimension; the entries before <paramref name="dim"/> are fixed).</param>
        /// <param name="dim">The dimension to walk.</param>
        /// <returns>The items.</returns>
        private static object[] RowsOf(Array a, int[] index, int dim)
        {
            int lo = a.GetLowerBound(dim), n = a.GetLength(dim);
            var items = new object[n];
            for (int i = 0; i < n; i++)
            {
                index[dim] = lo + i;
                items[i] = dim == a.Rank - 1 ? a.GetValue(index) : RowsOf(a, index, dim + 1);
            }
            return items;
        }

        /// <summary>
        ///     np.dot's conversion of func's result (<c>PyArray_FromAny</c>): an NDArray as it is; a Python sequence by
        ///     np.array's coercion; a str or a sequence of strs as NumSharp's text array (np.dot then gives NumPy's answer for
        ///     text); a scalar as a 0-d array of its discovered dtype; a typed C# array as an ndarray of its dtype.
        /// </summary>
        /// <param name="y">func's result.</param>
        /// <returns>The operand.</returns>
        /// <exception cref="ValueError">A ragged sequence (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="NotSupportedException">None, or a value NumPy makes an object array of (NumPy's np.dot then
        ///     computes with Python objects).</exception>
        private static NDArray DotOperand(object y)
        {
            switch (y)
            {
                case NDArray nd:
                    return nd;
                case string:
                    return np.asanyarray(y);
            }
            if (PolySequence.IsSequence(y))
            {
                NDArray arr = PolySequence.ToArrayOrNonNumeric(y, out var nonNumeric);
                if (arr is not null)
                    return arr;
                if (nonNumeric.IsObject)
                    throw nonNumeric.Refusal;
                return np.asanyarray(y);   // a str array: np.dot's text errors
            }
            // A scalar (a 0-d array of its discovered dtype) or a typed C# array; None and other objects refuse here.
            return PolyNumber.FromObject(y).ToNDArray();
        }
    }
}
