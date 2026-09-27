using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Threading;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolySeries.cs — numpy.polynomial's additive family and polyutils substrate (plan U1)
// =============================================================================
//
// NumPy 2.4.2's polyutils (as_series, trimseq, trimcoef, getdomain, mapparms, mapdomain, _add, _sub) and the
// basis constructors {p}line, ported line for line. The element loops run in the IL kernels of
// DirectILKernelGenerator.PolySeries.cs; the Python-level scalar arithmetic runs in PolyNumber
// (NDPolyNumber.cs); this file is the orchestration between them, in NumPy's statement order — which is
// also its ERROR order (an empty array is reported before a missing common type, a too-short domain before
// anything is divided).
//
// WHAT A C# ARGUMENT MEANS HERE
// -----------------------------
// NumPy converts coefficient arguments with np.array(a, ndmin=1); the C# spelling of each Python form:
//   * NDArray                         -> itself (0-d reads as one element; never copied until the result)
//   * bool / integer / float / double / Complex / BigInteger
//                                     -> a Python scalar: np.array discovers bool / int64 / float64 / complex128
//   * Half / char / decimal           -> a NumPy scalar of that dtype
//   * a typed C# array (double[], int[,], Complex[], …)
//                                     -> an ndarray of its element dtype (the house mapping)
//   * object[] / IList / other IEnumerable
//                                     -> a Python list: element dtypes DISCOVERED like np.array([...])
//   * string / string[]               -> a str array: sized like NumPy's, then "no common type"
// Domains passed to mapparms/mapdomain are INDEXED, never converted, exactly as NumPy indexes them: a typed
// C# array is an ndarray (its elements are NumPy scalars), an object[]/IList a Python list and a ValueTuple a
// Python tuple (their elements are Python scalars) — which decides scalarmath vs CPython arithmetic and
// whether a zero-length domain raises ZeroDivisionError or yields inf.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     One coefficient argument after NumPy's <c>np.array(a, ndmin=1)</c>, read in place: the address, length
    ///     and byte stride of its axis 0 plus what NumPy's validation reads. An NDArray argument is never copied;
    ///     a C# scalar or Python list is materialized once (it has no memory to point at).
    /// </summary>
    internal readonly unsafe struct PolySeriesView
    {
        /// <summary>The array whose memory <see cref="Ptr"/> points into (null for a non-numeric argument).</summary>
        public readonly NDArray Source;
        /// <summary>Address of element 0 (along axis 0).</summary>
        public readonly byte* Ptr;
        /// <summary>Length of axis 0 — the series length when <see cref="Ndim"/> is 1.</summary>
        public readonly long Len;
        /// <summary>Signed byte stride of axis 0.</summary>
        public readonly long Stride;
        /// <summary>Element dtype (<see cref="NPTypeCode.Empty"/> for a non-numeric argument).</summary>
        public readonly NPTypeCode Dtype;
        /// <summary><c>ndim</c> after <c>ndmin=1</c> (at least 1).</summary>
        public readonly int Ndim;
        /// <summary>Total element count (<c>a.size</c>).</summary>
        public readonly long Size;
        /// <summary>A str array: sized like NumPy's, but np.common_type rejects it.</summary>
        public readonly bool NonNumeric;

        private PolySeriesView(NDArray source, byte* ptr, long len, long stride, NPTypeCode dtype, int ndim, long size, bool nonNumeric)
        {
            Source = source; Ptr = ptr; Len = len; Stride = stride; Dtype = dtype; Ndim = ndim; Size = size; NonNumeric = nonNumeric;
        }

        /// <summary>A view of <paramref name="a"/>'s axis 0 (a 0-d array reads as one element).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The view.</returns>
        public static PolySeriesView Of(NDArray a)
        {
            byte* p = (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;
            if (a.ndim == 0)
                return new PolySeriesView(a, p, 1, 0, a.typecode, 1, 1, false);
            return new PolySeriesView(a, p, a.Shape.dimensions[0], a.Shape.strides[0] * a.dtypesize, a.typecode, a.ndim, a.size, false);
        }

        /// <summary>A view of ONE element of a 1-D array (NumPy iterating a 1-D array yields 0-d elements).</summary>
        /// <param name="a">The 1-D array.</param>
        /// <param name="i">The element index.</param>
        /// <returns>The one-element view.</returns>
        public static PolySeriesView Element(NDArray a, long i)
        {
            byte* p = (byte*)a.Storage.Address + (a.Shape.offset + i * a.Shape.strides[0]) * a.dtypesize;
            return new PolySeriesView(a, p, 1, 0, a.typecode, 1, 1, false);
        }

        /// <summary>
        ///     A C-contiguous array of any rank read as ONE run of its <c>size</c> elements (1-D, unit stride, from
        ///     the first element) — the flat order of a C-contiguous array is its logical order.
        /// </summary>
        /// <param name="a">A C-contiguous array (not checked).</param>
        /// <returns>The flat view.</returns>
        public static PolySeriesView Flat(NDArray a)
        {
            byte* p = (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;
            return new PolySeriesView(a, p, a.size, a.dtypesize, a.typecode, 1, a.size, false);
        }

        /// <summary>A str array of <paramref name="size"/> elements (1-D): passes the size/ndim checks, fails the common type.</summary>
        /// <param name="size">Element count.</param>
        /// <returns>The view.</returns>
        public static PolySeriesView Str(long size) => new PolySeriesView(null, null, size, 0, NPTypeCode.Empty, 1, size, true);

        /// <summary>
        ///     The same series starting at element <paramref name="start"/> (<c>v[start:start+length]</c>): same dtype
        ///     and stride, the address advanced along axis 0. Used to walk a long series in blocks.
        /// </summary>
        /// <param name="start">First element (0 ≤ start ≤ <see cref="Len"/>; not checked).</param>
        /// <param name="length">Element count of the sub-series (not checked against <see cref="Len"/>).</param>
        /// <returns>The sub-series view (1-D, <see cref="Size"/> = <paramref name="length"/>).</returns>
        public PolySeriesView Slice(long start, long length)
            => new PolySeriesView(Source, Ptr + start * Stride, length, Stride, Dtype, 1, length, NonNumeric);

        /// <summary>
        ///     The same elements in the opposite order (<c>v[::-1]</c>): the address of the last element along axis 0
        ///     and the negated stride. Only meaningful for an ORDER-FREE consumer — an integer min/max reads a reversed
        ///     view as the forward run it covers — since every other reader would see the elements reversed.
        /// </summary>
        /// <returns>The reversed view (1-D, same length and dtype; a one-element view is returned equivalent).</returns>
        public PolySeriesView Reversed()
            => new PolySeriesView(Source, Ptr + (Len - 1) * Stride, Len, -Stride, Dtype, 1, Len, NonNumeric);
    }

    /// <summary>
    ///     The <c>numpy.polynomial</c> additive family and <c>polyutils</c> substrate (see the file header):
    ///     one shared implementation behind all six basis facades and the <c>polyutils</c> module.
    /// </summary>
    internal static unsafe class NDPolySeries
    {
        // ---------------------------------------------------------------------------------------------
        //  Argument conversion
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>np.array(a, ndmin=1, copy=None)</c> for one coefficient argument, as a
        ///     <see cref="PolySeriesView"/> (see the file header for the C# → Python mapping).
        /// </summary>
        /// <param name="a">The argument.</param>
        /// <returns>The view.</returns>
        /// <exception cref="NotSupportedException">null (NumPy builds an object array), or a Python int beyond uint64.</exception>
        /// <exception cref="ValueError">A Python list whose items have inhomogeneous shapes.</exception>
        public static PolySeriesView AsCoefficientArray(object a)
        {
            switch (a)
            {
                case null:
                    throw new NotSupportedException("None has no NumSharp equivalent: NumPy builds an object array from it, a dtype NumSharp does not have");
                case NDArray nd:
                    return PolySeriesView.Of(nd);
                case string:
                    return PolySeriesView.Str(1);
                case string[] strs:
                    return PolySeriesView.Str(strs.Length);
                case Array arr when arr.GetType().GetElementType() != typeof(object):
                    // A typed C# array is an ndarray of its element dtype (the house mapping).
                    return PolySeriesView.Of(np.asanyarray(arr));
                case IEnumerable seq:
                    return PolySeriesView.Of(ListToArray(seq));
                default:
                    // A scalar: np.array discovers its dtype (Python) or keeps it (NumPy scalar).
                    return PolySeriesView.Of(PolyNumber.MakeArray(PolyNumber.FromObject(a)));
            }
        }

        /// <summary>
        ///     <c>np.array(list)</c> for a Python list given as an <see cref="IEnumerable"/> of C# values: each item
        ///     is classified by <see cref="PolyNumber.FromObject"/> and the dtype DISCOVERED over all of them
        ///     (<c>[1, 2.5]</c> is float64, <c>[True, 2]</c> int64), items of one shape stacking into a new axis.
        /// </summary>
        /// <param name="seq">The list.</param>
        /// <returns>The new array (an empty list is NumPy's float64 <c>(0,)</c>).</returns>
        /// <exception cref="ValueError">Items of different shapes.</exception>
        /// <exception cref="NotSupportedException">A null or string item (object / str arrays).</exception>
        private static NDArray ListToArray(IEnumerable seq)
        {
            var items = new List<PolyNumber>();
            foreach (var o in seq)
                items.Add(PolyNumber.FromObject(o));
            if (items.Count == 0)
                return new NDArray(NPTypeCode.Double, new Shape(0L), false);
            return PolyNumber.MakeArray(items.ToArray());
        }

        /// <summary>
        ///     NumPy's per-array checks in <c>as_series</c>: <c>Coefficient array is empty</c> before
        ///     <c>Coefficient array is not 1-d</c>, array by array in argument order.
        /// </summary>
        /// <param name="v">The converted argument.</param>
        /// <exception cref="ValueError">Empty, or not 1-d.</exception>
        public static void Validate(in PolySeriesView v)
        {
            if (v.Size == 0)
                throw new ValueError("Coefficient array is empty");
            if (v.Ndim != 1)
                throw new ValueError("Coefficient array is not 1-d");
        }

        /// <summary>
        ///     <c>np.common_type</c> over the converted arguments, with <c>as_series</c>' rejection: a bool or str
        ///     array has no common type, and NumPy (no object dtype to fall back to here) raises.
        /// </summary>
        /// <param name="views">The arguments (at least one).</param>
        /// <returns>The coefficient dtype: float16/float32/float64/complex128, or NumSharp's decimal.</returns>
        /// <exception cref="ValueError"><c>Coefficient arrays have no common type</c>.</exception>
        public static NPTypeCode CommonType(ReadOnlySpan<PolySeriesView> views)
        {
            var codes = new NPTypeCode[views.Length];
            for (int i = 0; i < views.Length; i++)
            {
                if (views[i].NonNumeric)
                    throw new ValueError("Coefficient arrays have no common type");
                codes[i] = views[i].Dtype;
            }
            try
            {
                return np.common_type_code(codes);
            }
            catch (TypeError)
            {
                // np.common_type's "can't get common type for non-numeric array" (a bool array): as_series has no
                // object dtype to retry with, so it raises its own ValueError.
                throw new ValueError("Coefficient arrays have no common type");
            }
        }

        /// <summary>trimseq's kept length of a converted 1-D argument (a str array is never trimmed: 'a' != 0).</summary>
        /// <param name="v">The argument.</param>
        /// <returns>The length NumPy's trimseq keeps.</returns>
        public static long TrimLength(in PolySeriesView v)
        {
            if (v.NonNumeric)
                return v.Len;
            // The kernel reports "the sequence itself" as +n and "a slice seq[:k]" as -k; only the length matters here.
            long r = DirectILKernelGenerator.GetPolyTrimLenKernel(v.Dtype)(v.Ptr, v.Len, v.Stride);
            return r < 0 ? -r : r;
        }

        /// <summary>
        ///     A NEW owning C-contiguous array of <paramref name="t"/> holding the first <paramref name="n"/>
        ///     elements of <paramref name="v"/> — as_series' <c>np.array(a, copy=True, dtype=dtype)</c>.
        /// </summary>
        /// <param name="v">The source.</param>
        /// <param name="n">Element count.</param>
        /// <param name="t">Target dtype.</param>
        /// <returns>The copy.</returns>
        /// <exception cref="NotSupportedException">A dtype pair the house conversion does not support.</exception>
        public static NDArray CopyAs(in PolySeriesView v, long n, NPTypeCode t)
        {
            var r = new NDArray(t, new Shape(n), false);
            CopyInto(v, n, t, (byte*)r.Storage.Address);
            return r;
        }

        /// <summary>
        ///     Writes the first <paramref name="n"/> elements of <paramref name="v"/>, converted to
        ///     <paramref name="t"/>, to the contiguous buffer <paramref name="dst"/> — the element loop of
        ///     <see cref="CopyAs"/>, also used by <see cref="AddSub"/> to materialize its in-place target.
        /// </summary>
        /// <param name="v">The source (any stride).</param>
        /// <param name="n">Element count.</param>
        /// <param name="t">Target dtype.</param>
        /// <param name="dst">Destination: room for <paramref name="n"/> elements of <paramref name="t"/>, not
        ///     overlapping the source.</param>
        /// <exception cref="NotSupportedException">A dtype pair the house conversion does not support.</exception>
        /// <remarks>
        ///     Every route stores NumPy's bits — they differ only in speed. A contiguous same-dtype source is a
        ///     memcpy (at any length: there is nothing to convert). A contiguous source of another dtype takes the
        ///     house SIMD cast kernel from <see cref="DirectILKernelGenerator.PolyHouseKernelThreshold"/> elements on
        ///     (the scalar float16 widen costs ~1.4 ns an element, the house Giesen widen ~0.17; even float32 →
        ///     float64 is 5× faster vectorized, because the scalar cvtss2sd chain carries a false register
        ///     dependency). A strided or reversed source takes the house strided kernel when the copy is a pure bit
        ///     copy of an element of at most 8 bytes (the SIMD reverse / deinterleave / gather copies: 100K
        ///     stride-2 float32 8 µs, float16 5 µs, against 20-25 µs one element at a time), or when float16 is
        ///     involved in a conversion (<see cref="DirectILKernelGenerator.PolyScalarIsEmulated"/>: the
        ///     stage-and-widen casts); every other strided source — a converting one, whose house strided cast is
        ///     generic scalar code — a short series, or a pair the house has no kernel for runs the scalar poly cast
        ///     kernel.
        ///     <para>[SkipLocalsInit]: the blocked strided-cast route's stack scratch is written by the strided copy
        ///     before the cast reads it, and this method runs once per BLOCK inside the combine and mapdomain loops —
        ///     zeroing up to 8 KB of scratch on every call cost more than converting the block.</para>
        /// </remarks>
        [SkipLocalsInit]
        public static void CopyInto(in PolySeriesView v, long n, NPTypeCode t, byte* dst)
        {
            int size = DirectILKernelGenerator.GetTypeSize(v.Dtype);
            // A 0-d source reads as one element with stride 0, so `Stride == size` also rules it out.
            if (v.Stride == size && n > 0)
            {
                if (v.Dtype == t)
                {
                    long bytes = n * size;
                    Buffer.MemoryCopy(v.Ptr, dst, bytes, bytes);
                    return;
                }
                if (n >= DirectILKernelGenerator.PolyHouseKernelThreshold)
                {
                    var cast = DirectILKernelGenerator.GetPolyContiguousCast(v.Dtype, t);
                    if (cast != null)
                    {
                        cast(v.Ptr, dst, n);
                        return;
                    }
                }
            }
            else if (n >= DirectILKernelGenerator.PolyHouseKernelThreshold && v.Stride % size == 0)
            {
                // The house kernels take ELEMENT strides (the view's are bytes; a view's stride is always a whole
                // number of elements, the modulo test above is defensive) and a 1-D shape.
                long srcStride = v.Stride / size, dstStride = 1;
                if ((v.Dtype == t && size <= 8)
                    || DirectILKernelGenerator.PolyScalarIsEmulated(v.Dtype) || DirectILKernelGenerator.PolyScalarIsEmulated(t))
                {
                    var strided = DirectILKernelGenerator.GetPolyStridedCast(v.Dtype, t);
                    if (strided != null)
                    {
                        long extent = n;
                        strided(v.Ptr, dst, &srcStride, &dstStride, &extent, 1);
                        return;
                    }
                }
                else if (size <= 8 && DirectILKernelGenerator.IsVectorizedContiguousCast(v.Dtype, t))
                {
                    // A converting strided source: the house strided CAST is generic scalar code, but its two halves
                    // both vectorize — the SIMD strided copy (same dtype) into an L1-resident block, then the SIMD
                    // contiguous cast out of it — so the conversion runs block by block through a stack scratch.
                    var copy = DirectILKernelGenerator.GetPolyStridedCast(v.Dtype, v.Dtype);
                    var cast = DirectILKernelGenerator.GetPolyContiguousCast(v.Dtype, t);
                    if (copy != null && cast != null)
                    {
                        const int block = 1024;
                        byte* scratch = stackalloc byte[block * size];
                        int tsize = DirectILKernelGenerator.GetTypeSize(t);
                        for (long k = 0; k < n; k += block)
                        {
                            long m = Math.Min(block, n - k);
                            copy(v.Ptr + k * v.Stride, scratch, &srcStride, &dstStride, &m, 1);
                            cast(scratch, dst + k * tsize, m);
                        }
                        return;
                    }
                }
            }
            DirectILKernelGenerator.GetPolyCastKernel(v.Dtype, t)(v.Ptr, n, v.Stride, dst);
        }

        /// <summary>
        ///     A view of <paramref name="a"/>'s first <paramref name="k"/> rows (<c>a[:k]</c>): same strides and
        ///     offset, writeability inherited, never owning its data — NumPy's trimseq result.
        /// </summary>
        /// <param name="a">The array (ndim ≥ 1).</param>
        /// <param name="k">Rows kept.</param>
        /// <returns>The view.</returns>
        public static NDArray Prefix(NDArray a, long k)
        {
            var dims = (long[])a.Shape.dimensions.Clone();
            dims[0] = k;
            var shape = new Shape(dims, (long[])a.Shape.strides.Clone(), a.Shape.offset, a.Shape.bufferSize);
            return new NDArray(a.Storage.Alias(ref shape));
        }

        // ---------------------------------------------------------------------------------------------
        //  polyutils.as_series / trimseq
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>polyutils.as_series(alist, trim)</c>: every item of <paramref name="alist"/> as a 1-D array, all
        ///     converted to their <c>np.common_type</c> and returned as fresh copies — trimmed of trailing zeros
        ///     first when <paramref name="trim"/>.
        /// </summary>
        /// <param name="alist">An iterable of coefficient arguments (an NDArray iterates its first axis).</param>
        /// <param name="trim">Remove trailing zeros (trimseq) before converting.</param>
        /// <returns>The arrays, in order.</returns>
        /// <exception cref="TypeError"><paramref name="alist"/> is not iterable (a scalar, null, a 0-d array).</exception>
        /// <exception cref="ValueError">An item is empty or not 1-d, or the items have no common type.</exception>
        public static NDArray[] AsSeries(object alist, bool trim)
        {
            var views = Items(alist);
            foreach (var v in views)
                Validate(v);
            var lengths = new long[views.Length];
            for (int i = 0; i < views.Length; i++)
                lengths[i] = trim ? TrimLength(views[i]) : views[i].Len;
            if (views.Length == 0)
                return System.Array.Empty<NDArray>();   // NumPy: np.common_type() of nothing, then an empty list
            NPTypeCode t = CommonType(views);
            var ret = new NDArray[views.Length];
            for (int i = 0; i < views.Length; i++)
                ret[i] = CopyAs(views[i], lengths[i], t);
            return ret;
        }

        /// <summary>
        ///     Python's iteration of <c>alist</c>, each item converted by <c>np.array(a, ndmin=1)</c>.
        /// </summary>
        /// <param name="alist">The iterable.</param>
        /// <returns>The converted items.</returns>
        /// <exception cref="TypeError">A non-iterable (CPython's <c>'int' object is not iterable</c>, NumPy's
        ///     <c>iteration over a 0-d array</c>).</exception>
        private static PolySeriesView[] Items(object alist)
        {
            switch (alist)
            {
                case null:
                    throw new TypeError("'NoneType' object is not iterable");
                case NDArray nd:
                {
                    if (nd.ndim == 0)
                        throw new TypeError("iteration over a 0-d array");
                    long n = nd.Shape.dimensions[0];
                    var r = new PolySeriesView[n];
                    for (long i = 0; i < n; i++)
                        r[i] = nd.ndim == 1 ? PolySeriesView.Element(nd, i) : PolySeriesView.Of(nd[(int)i]);
                    return r;
                }
                case string s:
                {
                    // A str iterates its characters; each is a one-character str.
                    var r = new PolySeriesView[s.Length];
                    for (int i = 0; i < s.Length; i++)
                        r[i] = PolySeriesView.Str(1);
                    return r;
                }
                case Array arr when arr.Rank > 1:
                    return Items(np.asanyarray(arr));   // a rectangular array iterates its rows, like the list of lists
                case IEnumerable seq:
                {
                    var r = new List<PolySeriesView>();
                    foreach (var item in seq)
                        r.Add(AsCoefficientArray(item));
                    return r.ToArray();
                }
                default:
                    throw new TypeError($"'{PythonTypeName(alist)}' object is not iterable");
            }
        }

        /// <summary>The Python type name CPython's messages quote for a C# value.</summary>
        /// <param name="o">The value (not null).</param>
        /// <returns><c>int</c>, <c>float</c>, <c>complex</c>, <c>bool</c>, NumPy's scalar names for Half/char/decimal,
        ///     or the CLR type name.</returns>
        private static string PythonTypeName(object o) => o switch
        {
            bool => "bool",
            sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger => "int",
            float or double => "float",
            Complex => "complex",
            Half => "numpy.float16",
            _ => o.GetType().Name,
        };

        /// <summary>
        ///     <c>polyutils.trimseq(seq)</c>: <paramref name="seq"/> itself when it is empty or its last element is
        ///     nonzero, otherwise the view <c>seq[:i+1]</c> ending at the last nonzero element (at least one element
        ///     is kept).
        /// </summary>
        /// <param name="seq">The sequence (any rank ≥ 1; rows of a 2-D+ array must hold exactly one element each,
        ///     since NumPy tests their truth value).</param>
        /// <returns><paramref name="seq"/> (same instance) or a view.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="seq"/> is null.</exception>
        /// <exception cref="TypeError">A 0-d array (<c>len() of unsized object</c>).</exception>
        /// <exception cref="ValueError">Rows with several elements or none (NumPy's truth-value errors).</exception>
        public static NDArray TrimSeq(NDArray seq)
        {
            if (seq is null) throw new ArgumentNullException(nameof(seq));
            if (seq.ndim == 0)
                throw new TypeError("len() of unsized object");
            long len = seq.Shape.dimensions[0];
            if (len == 0)
                return seq;
            long perRow = seq.size / len;
            if (perRow != 1)
            {
                // `seq[-1] != 0` is a row: NumPy's truth value of it raises unless it has exactly one element.
                if (perRow == 0)
                    throw new ValueError("The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.");
                throw new ValueError("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
            }
            var v = PolySeriesView.Of(seq);
            long k = DirectILKernelGenerator.GetPolyTrimLenKernel(v.Dtype)(v.Ptr, v.Len, v.Stride);
            // +len: NumPy's `return seq` (the same object); -k: its slice `seq[:k]`, a view even when k == len.
            return k >= 0 ? seq : Prefix(seq, -k);
        }

        // ---------------------------------------------------------------------------------------------
        //  polyutils._add / _sub  (every basis's {p}add / {p}sub)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>polyutils._add(c1, c2)</c> (<paramref name="subtract"/> false) or <c>_sub(c1, c2)</c> (true):
        ///     <c>as_series</c> both, update the longer in place (<c>c2</c> on a tie; for <c>_sub</c> with a
        ///     subtrahend at least as long, <c>c2 = -c2; c2[:n1] += c1</c>), then trimseq the result.
        /// </summary>
        /// <param name="c1">First series.</param>
        /// <param name="c2">Second series.</param>
        /// <param name="subtract">Subtract <paramref name="c2"/> instead of adding it.</param>
        /// <returns>A new array of the common dtype, or a view of one when trailing zeros were trimmed.</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type.</exception>
        public static NDArray AddSub(object c1, object c2, bool subtract)
        {
            var v1 = AsCoefficientArray(c1);
            var v2 = AsCoefficientArray(c2);
            Validate(v1);
            Validate(v2);
            long n1 = TrimLength(v1), n2 = TrimLength(v2);
            NPTypeCode t = CommonType(new[] { v1, v2 });

            // `if len(c1) > len(c2)` updates c1, else c2 — the in-place target is kernel operand A.
            bool firstIsTarget = n1 > n2;
            var (a, na, b, nb) = firstIsTarget ? (v1, n1, v2, n2) : (v2, n2, v1, n1);
            PolyCombineOp op = !subtract ? PolyCombineOp.Add : firstIsTarget ? PolyCombineOp.Subtract : PolyCombineOp.NegateAdd;

            var r = new NDArray(t, new Shape(na), false);
            byte* rp = (byte*)r.Storage.Address;
            BinaryOp update = op == PolyCombineOp.Subtract ? BinaryOp.Subtract : BinaryOp.Add;
            long k = PreferHouseCombine(update, a, na, b, t)
                ? CombineViaHouseKernels(op, update, a, na, b, nb, t, rp)
                : DirectILKernelGenerator.GetPolyCombineKernel(op, a.Dtype, b.Dtype, t)(a.Ptr, a.Stride, na, b.Ptr, b.Stride, nb, rp);
            // trimseq(ret): the array itself (+na) or its slice ret[:k] (-k) — a view, as NumPy returns it.
            return k >= 0 ? r : Prefix(r, -k);
        }

        /// <summary>
        ///     Whether <see cref="AddSub"/> should run through the house vector kernels
        ///     (<see cref="CombineViaHouseKernels"/>) rather than the fused scalar combine kernel — a pure speed
        ///     choice, both store NumPy's bits.
        /// </summary>
        /// <param name="update">The update's binary op.</param>
        /// <param name="a">The in-place target.</param>
        /// <param name="na">Its trimmed length.</param>
        /// <param name="b">The other operand.</param>
        /// <param name="t">The result dtype.</param>
        /// <returns>True when the house route is the faster one.</returns>
        /// <remarks>
        ///     The house route needs a long series (<see cref="DirectILKernelGenerator.PolyHouseKernelThreshold"/>)
        ///     and a result dtype whose same-dtype kernel is a vector loop, and then pays off when every step it
        ///     adds is itself a vector loop (measured at 100K): each operand is either already contiguous in the
        ///     result dtype or contiguous with a VECTORIZED house cast to it
        ///     (<see cref="DirectILKernelGenerator.IsVectorizedContiguousCast"/> — int8…uint32, float16 and float32
        ///     sources, and int64/uint64 into float64), so the conversions and the update all run SIMD in L1-resident
        ///     blocks (float32 27 → 7 µs, float16 459 → 80 µs); or float16 is anywhere in the operation — the fused
        ///     kernel emulates each float16 widen / add / narrow in software, which dwarfs any extra pass, strided or
        ///     not. A strided operand beyond <see cref="VectorReadable"/>'s rule, or one whose cast has no vector form
        ///     (int64/uint64 → float32, anything → complex), keeps the fused scalar kernel: it reads each element once
        ///     and is at the memory bound, while the
        ///     house route with a scalar cast measured slower (int32 → float64 via the old scalar-speed house cast
        ///     38.8 → 51.6 µs; a stride-2 float64 series 49.2 → 70.7 µs).
        /// </remarks>
        private static bool PreferHouseCombine(BinaryOp update, in PolySeriesView a, long na, in PolySeriesView b, NPTypeCode t)
        {
            if (na < DirectILKernelGenerator.PolyHouseKernelThreshold || !DirectILKernelGenerator.IsVectorizedSameTypeBinary(update, t))
                return false;
            if (DirectILKernelGenerator.PolyScalarIsEmulated(t)
                || DirectILKernelGenerator.PolyScalarIsEmulated(a.Dtype)
                || DirectILKernelGenerator.PolyScalarIsEmulated(b.Dtype))
                return true;
            return VectorReadable(a, t) && VectorReadable(b, t);
        }

        /// <summary>
        ///     Whether the house route reads operand <paramref name="v"/> into dtype <paramref name="t"/> with vector
        ///     loops only (<see cref="CopyInto"/>'s routes) AND that pays off: already of <paramref name="t"/> (a direct
        ///     read or memcpy when contiguous, the SIMD reverse / deinterleave / gather copy when strided), or of a dtype
        ///     the house casts to <paramref name="t"/> with a SIMD kernel (directly when contiguous, after the SIMD
        ///     strided copy into a block when strided).
        /// </summary>
        /// <param name="v">The operand.</param>
        /// <param name="t">The result dtype.</param>
        /// <returns>True when the house route reads the operand faster than the fused scalar kernel would.</returns>
        /// <remarks>A STRIDED operand qualifies for a result of at most 4 bytes (float32), or when its own element is a
        ///     sub-word (1-2 bytes, where the SIMD deinterleave / reverse copies are far faster than a scalar strided
        ///     read): a strided 4/8-byte operand in a float64 result keeps the fused scalar loop, which is already
        ///     bound by the strided reads (a stride-2 float64 series spans twice its size, past L2 at 100K) — the
        ///     house route's extra passes measured slower there (stride-2 int32 → float64 41 → 65 µs, stride-2 float64
        ///     49 → 66 µs), while a strided float32 add (29 → 16-26 µs) and a strided int16 → float64 add (55 → 44 µs)
        ///     gained. (float16 anywhere takes the house route regardless: <see cref="PreferHouseCombine"/>.)</remarks>
        private static bool VectorReadable(in PolySeriesView v, NPTypeCode t)
        {
            int size = DirectILKernelGenerator.GetTypeSize(v.Dtype);
            bool contiguous = v.Stride == size;
            if (!contiguous && (size > 8 || v.Stride % size != 0 || (DirectILKernelGenerator.GetTypeSize(t) > 4 && size > 2)))
                return false;
            return v.Dtype == t || DirectILKernelGenerator.IsVectorizedContiguousCast(v.Dtype, t);
        }

        /// <summary>
        ///     Elements per block of <see cref="CombineViaHouseKernels"/>' blocked route: a block of the result, the
        ///     converted block of the other operand and both source blocks (at most 8 bytes an element each — only
        ///     float16/float32/float64 results take the route) stay inside a 48 KB L1 data cache.
        /// </summary>
        private const int CombineBlock = 1024;

        /// <summary>
        ///     The body of <see cref="AddSub"/> for a long series whose dtype has a vector same-dtype kernel:
        ///     NumPy's own steps — the converted copy of the in-place target, <c>c2 = -c2</c> for a subtraction whose
        ///     subtrahend is at least as long, then <c>c[:nb] OP= other</c> — each run by a house SIMD kernel, then
        ///     trimseq over the result. Stores exactly what <see cref="DirectILKernelGenerator.GetPolyCombineKernel"/>
        ///     stores (same conversions, same operand order), in vector passes instead of one scalar pass.
        /// </summary>
        /// <param name="op">The in-place update NumPy performs.</param>
        /// <param name="update">The binary op of the update (<c>Add</c>, or <c>Subtract</c> for <see cref="PolyCombineOp.Subtract"/>).</param>
        /// <param name="a">The in-place target (the longer operand, c2 on a tie).</param>
        /// <param name="na">Its trimmed length (the result length).</param>
        /// <param name="b">The other operand.</param>
        /// <param name="nb">Its trimmed length (<c>1 &lt;= nb &lt;= na</c>).</param>
        /// <param name="t">The result dtype (np.common_type of both; float16, float32 or float64 here).</param>
        /// <param name="rp">The result buffer: <paramref name="na"/> contiguous elements of <paramref name="t"/>.</param>
        /// <returns>trimseq of the result, encoded as <see cref="PolyTrimLenKernel"/> returns it.</returns>
        /// <exception cref="NotSupportedException">A dtype pair the house conversion does not support.</exception>
        /// <remarks>
        ///     When both operands are already contiguous in <paramref name="t"/> and nothing is negated, the update
        ///     is one pass straight from the two sources (<c>r[:nb] = a[:nb] OP b[:nb]</c>) plus a memcpy of the
        ///     untouched tail. Otherwise the steps run BLOCK by block (<see cref="CombineBlock"/> elements, the shape
        ///     of NumPy's own buffered ufunc loop): the target block is materialized into the result (memcpy, SIMD
        ///     cast, or the strided poly cast), negated in place when NumPy negates it, and updated in place from the
        ///     other operand's block — read directly when it is contiguous in <paramref name="t"/>, else converted into
        ///     an L1-resident scratch block first. Blocking keeps the extra passes in L1, so the route costs the same
        ///     memory traffic as the fused scalar kernel while every element-wise step is a vector loop; whole-series
        ///     passes with a full-size temporary measured SLOWER than the fused kernel for converted or strided
        ///     float64 operands (a 100K int32 + int32 → float64 add went 41 µs → 57 µs). The negation must come first
        ///     and stay a separate step: <c>(-a) + b</c> and <c>b - a</c> keep different NaNs, and NumPy's is the
        ///     negated one.
        ///     <para>[SkipLocalsInit]: the scratch block is always written (<see cref="CopyInto"/>) before the update
        ///     reads it, so the runtime's zeroing of it is pure overhead.</para>
        /// </remarks>
        [SkipLocalsInit]
        private static long CombineViaHouseKernels(PolyCombineOp op, BinaryOp update, in PolySeriesView a, long na,
                                                   in PolySeriesView b, long nb, NPTypeCode t, byte* rp)
        {
            int size = DirectILKernelGenerator.GetTypeSize(t);
            var kernel = DirectILKernelGenerator.GetMixedTypeKernel(new MixedTypeKernelKey(t, t, t, update, ExecutionPath.SimdFull));
            // The SimdFull contract reads only the element count (its last argument); the 1-D strides/shape it is
            // handed are well-formed anyway, and the same holds for the contiguous unary kernel below.
            long unit = 1, extent = nb;
            bool aDirect = a.Dtype == t && a.Stride == size;
            bool bDirect = b.Dtype == t && b.Stride == size;
            bool negate = op == PolyCombineOp.NegateAdd;

            if (!negate && aDirect && bDirect)
            {
                // r[:nb] = a[:nb] OP b[:nb] (NumPy's operand order: the target first), r[nb:] = a[nb:].
                kernel(a.Ptr, b.Ptr, rp, &unit, &unit, &extent, 1, nb);
                long tailBytes = (na - nb) * size;
                if (tailBytes > 0)
                    Buffer.MemoryCopy(a.Ptr + nb * size, rp + nb * size, tailBytes, tailBytes);
                return DirectILKernelGenerator.GetPolyTrimLenKernel(t)(rp, na, size);
            }

            UnaryKernel negateKernel = negate
                ? DirectILKernelGenerator.GetUnaryKernel(new UnaryKernelKey(t, t, UnaryOp.Negate, true))
                : null;
            // Scratch for a converted block of the other operand, sized by the dtype (at most 8 bytes an element on
            // this route, so at most 8 KB of stack).
            byte* scratch = stackalloc byte[CombineBlock * size];
            for (long i = 0; i < na; i += CombineBlock)
            {
                long m = Math.Min(CombineBlock, na - i);
                byte* rb = rp + i * size;
                // as_series' converted copy of the target, block by block.
                CopyInto(a.Slice(i, m), m, t, rb);
                if (negate)
                {
                    // `c2 = -c2` over the whole target — every block, including those past nb — before the update.
                    long len = m;
                    negateKernel(rb, rb, &unit, &len, 1, m);
                }
                if (i >= nb)
                    continue;
                // `c[:nb] OP= other` on this block's overlap with other; the kernel reads r[j] and other[j] before it
                // writes r[j], so the in-place update is safe.
                long mb = Math.Min(m, nb - i);
                byte* bb;
                if (bDirect)
                    bb = b.Ptr + i * size;
                else
                {
                    CopyInto(b.Slice(i, mb), mb, t, scratch);
                    bb = scratch;
                }
                long len2 = mb;
                kernel(rb, bb, rb, &unit, &unit, &len2, 1, mb);
            }
            return DirectILKernelGenerator.GetPolyTrimLenKernel(t)(rp, na, size);
        }

        // ---------------------------------------------------------------------------------------------
        //  polyutils.trimcoef  (every basis's {p}trim)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>polyutils.trimcoef(c, tol)</c>: <c>as_series([c])</c>, then the prefix up to the last coefficient
        ///     whose magnitude exceeds <paramref name="tol"/> — or <c>c[:1]*0</c> when none does.
        /// </summary>
        /// <param name="c">The series.</param>
        /// <param name="tol">The tolerance as NumPy sees it (a Python int/float, a NumPy scalar, …).</param>
        /// <returns>A new array of the coefficient dtype.</returns>
        /// <exception cref="ValueError"><c>tol must be non-negative</c>, an empty / non-1-d series, no common type, or
        ///     a multi-element / empty tolerance array (NumPy's truth-value errors).</exception>
        /// <exception cref="TypeError">A Python complex tolerance (no ordering).</exception>
        /// <exception cref="OverflowException">A Python int tolerance too large for the comparison dtype.</exception>
        public static NDArray TrimCoef(object c, in PolyNumber tol)
        {
            if (tol.LessZero())
                throw new ValueError("tol must be non-negative");
            var v = AsCoefficientArray(c);
            Validate(v);
            long len = TrimLength(v);   // as_series trims first; the kept prefix never loses a coefficient above tol
            NPTypeCode t = CommonType(new[] { v });

            // `np.abs(c) > tol`: abs of a complex series is float64; the comparison dtype is NEP 50's.
            NPTypeCode ta = t == NPTypeCode.Complex ? NPTypeCode.Double : t;
            NPTypeCode tk = tol.IsPython ? PolyNumber.WeakPromote(ta, tol.Py) : PolyTyping.Promote(ta, tol.Dtype);
            if (tol.IsNdArray && tol.Array.ndim > 1)
                throw new ValueError("too many values to unpack (expected 1)");   // np.nonzero of a 2-D comparison

            PolyRaw16 tolRaw;
            if (tol.IsNdArray)
                PolyNumber.FromArray(tol.Array.reshape(Shape.Scalar)).WriteAs(tk, &tolRaw);
            else
                tol.WriteAs(tk, &tolRaw);
            long k = DirectILKernelGenerator.GetPolyLastAboveKernel(v.Dtype, t, tk)(v.Ptr, len, v.Stride, (byte*)&tolRaw);

            if (k == 0)
            {
                // c[:1] * 0 — a ufunc multiply by the weak int 0 in the coefficient dtype, so -0.0 / NaN / the fused
                // complex product's zero signs are NumPy's.
                var r = CopyAs(v, 1, t);
                PolyRaw16 zero = default;
                byte* p = (byte*)r.Storage.Address;
                DirectILKernelGenerator.GetPolyScalarBinaryKernel(BinaryOp.Multiply, t, PolyComplexProduct.Simd)(p, (byte*)&zero, p);
                return r;
            }
            return CopyAs(v, k, t);
        }

        // ---------------------------------------------------------------------------------------------
        //  polyutils.getdomain
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>polyutils.getdomain(x)</c>: <c>[x.min(), x.max()]</c> of <c>as_series([x], trim=False)</c>; for a
        ///     complex x the corners <c>[complex(min re, min im), complex(max re, max im)]</c>.
        /// </summary>
        /// <param name="x">The points.</param>
        /// <returns>A new 2-element array of the coefficient dtype.</returns>
        /// <exception cref="ValueError">Empty / non-1-d x, or no common type (bool).</exception>
        /// <remarks>
        ///     NumPy reduces a fresh C-contiguous COPY, and the ±0 / NaN-payload answer of a float min/max depends
        ///     on the reduction schedule, i.e. on the layout: a strided float x is therefore reduced as that copy would
        ///     be, while a contiguous one is reduced in place (same schedule, same bits). An integer x is reduced in its
        ///     own dtype and the two results converted (int → float64 is monotone, so min/max commute with it) — in
        ///     memory order, since its min/max do not depend on the order (see <see cref="IsOrderFreeForMinMax"/>).
        ///     <para>An x of a dtype NumPy's exact min/max schedule serves (float64/float32 and the eight integer widths)
        ///     takes <see cref="TryDomainBlocked"/> whatever its layout: ONE pass over 8 KB windows of the copy — read in
        ///     place from a unit-stride x (a reversed integer x as the forward run it covers), packed into an L1 scratch
        ///     otherwise — folding both the min and the max while each window is in L1, bit-identical to reducing the
        ///     whole copy. The copy-then-reduce route it replaced allocated the copy and read it twice more (100K
        ///     stride-2 float64 / int64 / uint64: ~37 / ~45 / ~50 µs, about NumPy's time; now ~23 / ~30 / ~31) and paid
        ///     two engine reductions' fixed cost on a short x (16 points: ~0.9 → ~0.3 µs). A float16 / char x keeps
        ///     that route (no exact schedule serves it).</para>
        /// </remarks>
        public static NDArray GetDomain(object x)
        {
            var v = AsCoefficientArray(x);
            Validate(v);
            NPTypeCode t = CommonType(new[] { v });
            bool contiguous = v.Stride == DirectILKernelGenerator.GetTypeSize(v.Dtype) || v.Len == 1;
            var r = new NDArray(t, new Shape(2L), false);
            byte* dst = (byte*)r.Storage.Address;
            int size = r.dtypesize;

            if (t == NPTypeCode.Complex)
            {
                NDArray c = v.Dtype == NPTypeCode.Complex && contiguous ? Row(v) : CopyAs(v, v.Len, NPTypeCode.Complex);
                NDArray re = np.real(c), im = np.imag(c);
                var lo = new Complex(Scalar<double>(np.amin(re)), Scalar<double>(np.amin(im)));
                var hi = new Complex(Scalar<double>(np.amax(re)), Scalar<double>(np.amax(im)));
                *(Complex*)dst = lo;
                *(Complex*)(dst + size) = hi;
                return r;
            }

            // A dtype the exact schedule serves takes ONE blocked pass over every layout (TryDomainBlocked), writing the
            // result itself. An integer / char x has no ±0 and no NaN, so its min and max are the same in ANY order: a
            // reversed one is handed over as the forward run it covers, which the pass then folds in place.
            int elem = DirectILKernelGenerator.GetTypeSize(v.Dtype);
            bool reversedOrderFree = IsOrderFreeForMinMax(v.Dtype) && v.Stride == -elem;
            if (TryDomainBlocked(reversedOrderFree ? v.Reversed() : v, t, dst, size))
                return r;

            // What the two reductions read when the blocked pass declines (float16 / char, or a test hook). A float x
            // reduces NumPy's fresh C-contiguous copy (see remarks: its ±0 / NaN answer depends on the schedule). An
            // integer / char x reduces in any order: a reversed view as the forward contiguous block it covers (a view,
            // no copy), any other stride packed first with the SIMD strided copy — either way the reductions run their
            // contiguous SIMD loops instead of the strided iterator schedule (100K int16 stride-2: 43 → ~10 µs).
            NDArray src, temp = null;
            if (contiguous)
                src = Row(v);
            else if (v.Dtype == t)
                src = temp = CopyAs(v, v.Len, t);
            else if (reversedOrderFree)
                src = temp = Row(v)["::-1"];
            else if (IsOrderFreeForMinMax(v.Dtype) && elem <= 8)
                src = temp = CopyAs(v, v.Len, v.Dtype);
            else
                src = Row(v);
            try
            {
                using (var mn = np.amin(src))
                using (var mx = np.amax(src))
                {
                    PolyConstPool.ConvertBuffer((byte*)mn.Storage.Address + mn.Shape.offset * mn.dtypesize, mn.typecode, dst, t, 1);
                    PolyConstPool.ConvertBuffer((byte*)mx.Storage.Address + mx.Shape.offset * mx.dtypesize, mx.typecode, dst + size, t, 1);
                }
            }
            finally
            {
                temp?.Dispose();
            }
            return r;
        }

        /// <summary>
        ///     Whether min/max over dtype <paramref name="t"/> give the same BITS in any reduction order: true for the
        ///     integers and char (no signed zero, no NaN, ties are identical values); false for the floats (a ±0 tie
        ///     and a NaN payload follow the schedule), complex and decimal (whose zero carries a sign flag).
        /// </summary>
        /// <param name="t">The dtype.</param>
        /// <returns>True for an order-free min/max.</returns>
        private static bool IsOrderFreeForMinMax(NPTypeCode t)
            => t is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char
                 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64;

        /// <summary>
        ///     Bytes per block of <see cref="MinMaxBlocked{T}"/> — 1,024 float64 / 2,048 float32 / 8,192 int8 elements.
        ///     It must be a whole number of NumPy's 8-vector groups for every lane width (8 × 32 bytes = 256 divides
        ///     8,192), because every block but the last is folded group by group and a group straddling two blocks would
        ///     change the fold order; and it must stay in L1 together with the SOURCE lines its strided copy reads (twice
        ///     the block for a stride-2 x) — the limit <see cref="AffineBlockBytes"/> measured.
        /// </summary>
        private const int DomainBlockBytes = 8192;

        /// <summary>
        ///     Test and benchmark switch for <see cref="TryDomainBlocked"/>. While <see langword="true"/> on the calling
        ///     thread, getdomain takes the copy-then-reduce route the blocked pass replaced — which lets a test compare the
        ///     two routes on one input, byte for byte, and a benchmark time them in ONE process (100K-point timings move
        ///     with the allocator's page state between runs).
        /// </summary>
        /// <remarks>Thread-static on purpose: MSTest runs test classes in parallel, and a process-wide switch flipped by
        ///     one test would silently re-route another test's getdomain calls. Production code never sets it.</remarks>
        [ThreadStatic] internal static bool DisableBlockedDomain;

        /// <summary>
        ///     Counts the getdomain calls <see cref="TryDomainBlocked"/> served on the calling thread — the copy route
        ///     computes the same answer, so a test needs the count to prove the blocked pass actually ran.
        /// </summary>
        [ThreadStatic] internal static long BlockedDomainRuns;

        /// <summary>
        ///     getdomain's <c>[x.min(), x.max()]</c> of a 1-D <paramref name="v"/> in ONE blocked pass: the elements of
        ///     NumPy's C-contiguous copy are read <see cref="DomainBlockBytes"/> at a time — in place when
        ///     <paramref name="v"/> is a unit-stride run of the reduced dtype, else produced into an L1 scratch by
        ///     <see cref="CopyInto"/> (the house SIMD strided copy / cast) — and folded by NumPy's exact min/max schedule
        ///     (<see cref="MinMaxBlocked{T}"/>), then both results are converted to the coefficient dtype and written to
        ///     <paramref name="dst"/>.
        /// </summary>
        /// <param name="v">The points: 1-D, at least one element, any stride. For an integer x the caller may pass a
        ///     reversed view as its forward run (<see cref="PolySeriesView.Reversed"/>): integer min/max is order-free.
        ///     A float x must be passed as it is — its order decides which ±0 / NaN survives.</param>
        /// <param name="t">The coefficient dtype of the result.</param>
        /// <param name="dst">The result's two elements (<paramref name="size"/> bytes each).</param>
        /// <param name="size">Byte size of one element of <paramref name="t"/>.</param>
        /// <returns>False — nothing written — when <see cref="DisableBlockedDomain"/> or
        ///     <see cref="NDExpr.DisableExactMinMax"/> is set on this thread, or when the exact schedule does not serve
        ///     the points' dtype (<see cref="NumPyMinMaxReduce.Supports"/>: float16, char, decimal and complex decline).</returns>
        /// <remarks>
        ///     <para>The points are reduced in their OWN dtype and the two results converted: for float64/float32 that
        ///     dtype IS the coefficient dtype (<c>np.common_type</c> of one float array is itself), so this is NumPy's
        ///     reduction of its copy exactly; for an integer the conversion to float64 is monotone, so converting the
        ///     extremes equals taking the extremes of the converted copy (and integers carry no ±0 / NaN whose bits could
        ///     depend on the order).</para>
        ///     <para>uint64 is the exception: its points are converted to float64 as they are packed (the house
        ///     exponent-splice cast) and the float64 copy is reduced — what NumPy does, and exact by the same monotonicity
        ///     (the converted values are non-negative and NaN-free, so equal values have equal bits). AVX2 has no unsigned
        ///     64-bit compare, and the emulated one costs two sign flips per min/max step: measured over an L1 block of
        ///     100K points, the uint64 folds took 20.4 µs where the cast plus the float64 folds take ~14 (the int64 folds,
        ///     13.9 µs, stay in int64 — its cast is the slower splice and would cost more than it saves).</para>
        ///     <para>Honours <see cref="NDExpr.DisableExactMinMax"/> like every other exact min/max route: the fallback
        ///     then reduces the copy with the engine's older kernels (value-identical, not NumPy's ±0 / NaN bits).</para>
        ///     <para>[SkipLocalsInit]: the 8 KB scratch is written by the copy before any fold reads it, and zeroing it
        ///     would be a measurable part of a short call.</para>
        /// </remarks>
        [SkipLocalsInit]
        internal static bool TryDomainBlocked(in PolySeriesView v, NPTypeCode t, byte* dst, int size)
        {
            if (DisableBlockedDomain || NDExpr.DisableExactMinMax || !NumPyMinMaxReduce.Supports(v.Dtype))
                return false;

            // The dtype the packed copy holds and the folds reduce (see remarks: uint64 packs as float64 = t).
            NPTypeCode rt = v.Dtype == NPTypeCode.UInt64 ? t : v.Dtype;
            // One element of the reduced dtype each: Supports admits nothing wider than 8 bytes.
            ulong lo = 0, hi = 0;
            byte* scratch = stackalloc byte[DomainBlockBytes];
            // The generic fold needs a static element type; every case runs the SAME code (MinMaxBlocked<T>).
            switch (rt)
            {
                case NPTypeCode.Double: MinMaxBlocked(v, rt, scratch, (double*)&lo, (double*)&hi); break;
                case NPTypeCode.Single: MinMaxBlocked(v, rt, scratch, (float*)&lo, (float*)&hi); break;
                case NPTypeCode.SByte: MinMaxBlocked(v, rt, scratch, (sbyte*)&lo, (sbyte*)&hi); break;
                case NPTypeCode.Byte: MinMaxBlocked(v, rt, scratch, (byte*)&lo, (byte*)&hi); break;
                case NPTypeCode.Int16: MinMaxBlocked(v, rt, scratch, (short*)&lo, (short*)&hi); break;
                case NPTypeCode.UInt16: MinMaxBlocked(v, rt, scratch, (ushort*)&lo, (ushort*)&hi); break;
                case NPTypeCode.Int32: MinMaxBlocked(v, rt, scratch, (int*)&lo, (int*)&hi); break;
                case NPTypeCode.UInt32: MinMaxBlocked(v, rt, scratch, (uint*)&lo, (uint*)&hi); break;
                case NPTypeCode.Int64: MinMaxBlocked(v, rt, scratch, (long*)&lo, (long*)&hi); break;
                default: return false;   // unreachable: Supports admits no other dtype, and uint64 reduces as float64
            }

            BlockedDomainRuns++;
            PolyConstPool.ConvertBuffer(&lo, rt, dst, t, 1);
            PolyConstPool.ConvertBuffer(&hi, rt, dst + size, t, 1);
            return true;
        }

        /// <summary>
        ///     NumPy's <c>(c.min(), c.max())</c> of <c>c</c> = the C-contiguous copy of <paramref name="v"/>, without
        ///     the copy: element 0 seeds both accumulators (<c>splat(c[0])</c>, the value NumPy's reduce copies into its
        ///     result), then the rest of <c>c</c> is taken block by block — read where it lies when <paramref name="v"/>
        ///     is already a unit-stride run of <paramref name="rt"/> (the copy would equal it), produced into
        ///     <paramref name="scratch"/> otherwise — and every block is folded by BOTH lanes while it is in L1: full
        ///     blocks whole 8-vector groups at a time (<see cref="NumPyMinMaxReduce.FoldGroups{T,TLane}"/>), the last
        ///     block through the single vectors, the horizontal reduce and the scalar tail
        ///     (<see cref="NumPyMinMaxReduce.Finish{T,TLane}"/>).
        /// </summary>
        /// <typeparam name="T">The element type of <paramref name="rt"/>.</typeparam>
        /// <param name="v">The points (1-D, any stride, at least one element).</param>
        /// <param name="rt">The dtype the copy is produced in and reduced at (a dtype
        ///     <see cref="NumPyMinMaxReduce.Supports"/> admits): the points' own, or the coefficient dtype they convert
        ///     to (see <see cref="TryDomainBlocked"/>).</param>
        /// <param name="scratch"><see cref="DomainBlockBytes"/> bytes of scratch, not overlapping the points.</param>
        /// <param name="lo">Receives the minimum (NumPy's bits).</param>
        /// <param name="hi">Receives the maximum (NumPy's bits).</param>
        /// <remarks>
        ///     Bit-identical to one <c>simd_reduce_c</c> call over the whole copy — the argument
        ///     <c>DefaultEngine.StreamMinMax</c> rests on: the blocks are counted from the element AFTER the seed and
        ///     hold a whole number of groups, so NumPy's groups never straddle a block, and within a lane NumPy's group
        ///     tree and single-vector loop are one ordered fold of an associative op, which the carried accumulator
        ///     continues across blocks unchanged. So which ±0 survives a tie and where a NaN turns canonical are
        ///     exactly NumPy's. A one-element x returns its element untouched for both (NumPy never runs the loop).
        ///     The seed is produced by the same <see cref="CopyInto"/> as the blocks, so a converting copy converts it
        ///     exactly as it converts every other element.
        ///     <para>In place, the first fold of a window pulls it into L1 and the second reads it there — one pass over
        ///     memory where the two engine reductions made two (100K contiguous float64 / int32, same process: 11.2 /
        ///     6.2 µs before, 9.4 / 4.8 after).</para>
        /// </remarks>
        private static void MinMaxBlocked<T>(in PolySeriesView v, NPTypeCode rt, byte* scratch, T* lo, T* hi)
            where T : unmanaged, INumber<T>
        {
            long n = v.Len;
            // A unit-stride run already of the reduced dtype IS its copy: its windows are folded where they lie.
            bool inPlace = rt == v.Dtype && v.Stride == sizeof(T);
            // Element 0 of the copy (v.Ptr is the logical first element whatever the stride, a reversed view included).
            T seed;
            if (inPlace)
                seed = *(T*)v.Ptr;
            else
            {
                CopyInto(v.Slice(0, 1), 1, rt, scratch);
                seed = *(T*)scratch;
            }
            if (n == 1)
            {
                *lo = seed;
                *hi = seed;
                return;
            }

            var accMin = Vector256.Create(seed);
            var accMax = accMin;
            long block = DomainBlockBytes / sizeof(T);
            long groupElems = Vector256<T>.Count * 8L;
            for (long k = 1; ; k += block)
            {
                long m = Math.Min(block, n - k);
                T* p;
                if (inPlace)
                    p = (T*)v.Ptr + k;
                else
                {
                    CopyInto(v.Slice(k, m), m, rt, scratch);
                    p = (T*)scratch;
                }
                if (k + m >= n)
                {
                    *lo = NumPyMinMaxReduce.Finish<T, NumPyMinMaxReduce.MinLane<T>>(accMin, p, m);
                    *hi = NumPyMinMaxReduce.Finish<T, NumPyMinMaxReduce.MaxLane<T>>(accMax, p, m);
                    return;
                }
                // A full block: `block` is a whole number of groups (see DomainBlockBytes), so this folds all of it.
                accMin = NumPyMinMaxReduce.FoldGroups<T, NumPyMinMaxReduce.MinLane<T>>(accMin, p, m / groupElems);
                accMax = NumPyMinMaxReduce.FoldGroups<T, NumPyMinMaxReduce.MaxLane<T>>(accMax, p, m / groupElems);
            }
        }

        /// <summary>The 1-D array a converted argument reads (its source when that is already 1-D, else a 1-element view).</summary>
        /// <param name="v">The argument.</param>
        /// <returns>The array.</returns>
        private static NDArray Row(in PolySeriesView v)
            => v.Source.ndim == 1 ? v.Source : v.Source.reshape(new Shape(1L));

        /// <summary>A 0-d result's value.</summary>
        /// <typeparam name="T">Element type.</typeparam>
        /// <param name="a">The 0-d array (disposed here: reductions return fresh arrays).</param>
        /// <returns>The value.</returns>
        private static T Scalar<T>(NDArray a) where T : unmanaged
        {
            using (a)
                return *(T*)((byte*)a.Storage.Address + a.Shape.offset * a.dtypesize);
        }

        // ---------------------------------------------------------------------------------------------
        //  polyutils.mapparms / mapdomain
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>seq[i]</c> as Python evaluates it on a domain argument: an ndarray yields a NumPy scalar (1-D) or
        ///     a sub-array; a Python sequence (object[] / IList / ValueTuple) its item; anything else raises NumPy's
        ///     or CPython's error.
        /// </summary>
        /// <param name="seq">The domain argument.</param>
        /// <param name="i">0 or 1.</param>
        /// <returns>The element.</returns>
        /// <exception cref="IndexError">Too short (NumPy's / CPython's texts), a 0-d array, or a NumPy scalar.</exception>
        /// <exception cref="TypeError">A Python scalar or None (<c>'int' object is not subscriptable</c>), or a str
        ///     (whose one-character items cannot be subtracted).</exception>
        public static PolyNumber Index(object seq, int i)
        {
            switch (seq)
            {
                case null:
                    throw new TypeError("'NoneType' object is not subscriptable");
                case NDArray nd:
                    return IndexArray(nd, i);
                case string s:
                    if (i >= s.Length)
                        throw new IndexError("string index out of range");
                    // Both items exist (index 1 is read first): the subtraction that follows is str - str.
                    throw new TypeError("unsupported operand type(s) for -: 'str' and 'str'");
                case Array arr when arr.GetType().GetElementType() != typeof(object):
                    return IndexArray(np.asanyarray(arr), i);   // a typed C# array is an ndarray
                case ITuple tup:
                    if (i >= tup.Length)
                        throw new IndexError("tuple index out of range");
                    return PolyNumber.FromObject(tup[i]);
                case IList list:
                    if (i >= list.Count)
                        throw new IndexError("list index out of range");
                    return PolyNumber.FromObject(list[i]);
                case Half or char or decimal:
                    throw new IndexError("invalid index to scalar variable.");
                default:
                    throw new TypeError($"'{PythonTypeName(seq)}' object is not subscriptable");
            }
        }

        /// <summary>ndarray <c>a[i]</c>: a NumPy scalar for 1-D, a sub-array view otherwise.</summary>
        /// <param name="nd">The array.</param>
        /// <param name="i">The index.</param>
        /// <returns>The element.</returns>
        /// <exception cref="IndexError">0-d, or axis 0 too short (NumPy's texts).</exception>
        private static PolyNumber IndexArray(NDArray nd, int i)
        {
            if (nd.ndim == 0)
                throw new IndexError("too many indices for array: array is 0-dimensional, but 1 were indexed");
            long n = nd.Shape.dimensions[0];
            if (i >= n)
                throw new IndexError($"index {i} is out of bounds for axis 0 with size {n}");
            if (nd.ndim > 1)
                return PolyNumber.FromArray(nd[i]);
            var v = PolySeriesView.Element(nd, i);
            return PolyNumber.FromScalar(nd.typecode, v.Ptr);
        }

        /// <summary>
        ///     <c>polyutils.mapparms(old, new)</c>: the offset and scale of the linear map sending <c>old[i]</c> to
        ///     <c>new[i]</c>, in Python's evaluation order:
        ///     <c>oldlen = old[1] - old[0]; newlen = new[1] - new[0];
        ///     off = (old[1]*new[0] - old[0]*new[1]) / oldlen; scl = newlen / oldlen</c>.
        /// </summary>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns><c>(off, scl)</c>, each of NumPy's kind (Python / NumPy scalar / array).</returns>
        /// <exception cref="IndexError">A domain shorter than 2.</exception>
        /// <exception cref="TypeError">A domain that is not subscriptable, or a bool subtraction.</exception>
        /// <exception cref="DivideByZeroException">A zero-length domain given as Python numbers (ZeroDivisionError).</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        public static (PolyNumber Off, PolyNumber Scl) MapParms(object old, object @new)
        {
            // Python lists/tuples of Python numbers: the machine-number lane (it only ever answers "applicable",
            // raising nothing but the ZeroDivisionError Python raises at the same point).
            if (TryMapParmsFast(old, @new, out var laneOff, out var laneScl))
                return (laneOff.ToPolyNumber(), laneScl.ToPolyNumber());
            // Two 1-D ndarrays of one dtype: the fused scalarmath kernel.
            if (TryMapParmsScalars(old, @new, out var off, out var scl))
                return (off, scl);
            return MapParmsGeneral(old, @new);
        }

        /// <summary>
        ///     The exact general lane of <see cref="MapParms(object, object)"/>: every element read through
        ///     <see cref="Index"/> into a <see cref="PolyNumber"/> (Python ints as <see cref="BigInteger"/>) and every
        ///     operation dispatched by <see cref="PolyNumber.Binary"/>. It is the REFERENCE the machine-number lane is
        ///     proven against (the corpus replay and the unit tests compare the two); callers want
        ///     <see cref="MapParms(object, object)"/>.
        /// </summary>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns><c>(off, scl)</c>, each of NumPy's kind.</returns>
        /// <exception cref="IndexError">A domain shorter than 2.</exception>
        /// <exception cref="TypeError">A domain that is not subscriptable, or a bool subtraction.</exception>
        /// <exception cref="DivideByZeroException">A zero-length domain given as Python numbers.</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        internal static (PolyNumber Off, PolyNumber Scl) MapParmsGeneral(object old, object @new)
        {
            // Python finishes `oldlen = old[1] - old[0]` before it touches `new`, so an index or arithmetic error in
            // old wins over one in new. Each element is read ONCE: the later `old[1]`/`new[0]` reads in Python's
            // off expression return the same, unchanged values (arithmetic never mutates its operands), and a
            // domain element conversion is the costly part of the whole function.
            var o1 = Index(old, 1);
            var o0 = Index(old, 0);
            var oldlen = PolyNumber.Binary(BinaryOp.Subtract, o1, o0);
            var n1 = Index(@new, 1);
            var n0 = Index(@new, 0);
            var newlen = PolyNumber.Binary(BinaryOp.Subtract, n1, n0);
            return MapParmsFinish(o1, o0, n1, n0, oldlen, newlen);
        }

        /// <summary>
        ///     <see cref="MapParms(object, object)"/> for domains given as C# 2-tuples (Python tuples): the CPython
        ///     machine-number lane (<see cref="TryMapParmsFast{T0,T1,T2,T3}"/>) when every element is a Python int,
        ///     float or complex, otherwise <see cref="MapParmsSlow{T0,T1,T2,T3}"/>.
        /// </summary>
        /// <typeparam name="T0">Type of <c>old[0]</c>.</typeparam>
        /// <typeparam name="T1">Type of <c>old[1]</c>.</typeparam>
        /// <typeparam name="T2">Type of <c>new[0]</c>.</typeparam>
        /// <typeparam name="T3">Type of <c>new[1]</c>.</typeparam>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns><c>(off, scl)</c>, each of NumPy's kind.</returns>
        /// <exception cref="TypeError">NumPy's bool subtraction (NumPy bool elements).</exception>
        /// <exception cref="DivideByZeroException">A zero-length domain of Python numbers (ZeroDivisionError).</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        /// <exception cref="NotSupportedException">A null or string element (see <see cref="PolyNumber.FromObject"/>).</exception>
        public static (PolyNumber Off, PolyNumber Scl) MapParms<T0, T1, T2, T3>(in (T0, T1) old, in (T2, T3) @new)
        {
            if (TryMapParmsFast(old, @new, out var off, out var scl))
                return (off.ToPolyNumber(), scl.ToPolyNumber());
            return MapParmsSlow(old, @new);
        }

        /// <summary>
        ///     The general lane of <see cref="MapParms{T0,T1,T2,T3}"/>: every element classified by the house mapping
        ///     (<see cref="PolyNumber.FromValue{T}"/> — no boxing for C# numbers) and run through the same arithmetic
        ///     as <see cref="MapParms(object, object)"/>, in the same order (a tuple element read cannot fail, so
        ///     only the conversions and operations can raise, exactly where Python's would).
        /// </summary>
        /// <typeparam name="T0">Type of <c>old[0]</c>.</typeparam>
        /// <typeparam name="T1">Type of <c>old[1]</c>.</typeparam>
        /// <typeparam name="T2">Type of <c>new[0]</c>.</typeparam>
        /// <typeparam name="T3">Type of <c>new[1]</c>.</typeparam>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns><c>(off, scl)</c>, each of NumPy's kind.</returns>
        /// <exception cref="TypeError">NumPy's bool subtraction.</exception>
        /// <exception cref="DivideByZeroException">A zero-length domain of Python numbers.</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        /// <exception cref="NotSupportedException">A null or string element.</exception>
        public static (PolyNumber Off, PolyNumber Scl) MapParmsSlow<T0, T1, T2, T3>(in (T0, T1) old, in (T2, T3) @new)
        {
            var o1 = PolyNumber.FromValue(old.Item2);
            var o0 = PolyNumber.FromValue(old.Item1);
            var oldlen = PolyNumber.Binary(BinaryOp.Subtract, o1, o0);
            var n1 = PolyNumber.FromValue(@new.Item2);
            var n0 = PolyNumber.FromValue(@new.Item1);
            var newlen = PolyNumber.Binary(BinaryOp.Subtract, n1, n0);
            return MapParmsFinish(o1, o0, n1, n0, oldlen, newlen);
        }

        /// <summary>The rest of mapparms once both lengths exist: <c>off = (old[1]*new[0] - old[0]*new[1]) / oldlen</c>,
        ///     then <c>scl = newlen / oldlen</c> (Python's order: off's division raises first).</summary>
        /// <param name="o1"><c>old[1]</c>.</param>
        /// <param name="o0"><c>old[0]</c>.</param>
        /// <param name="n1"><c>new[1]</c>.</param>
        /// <param name="n0"><c>new[0]</c>.</param>
        /// <param name="oldlen"><c>old[1] - old[0]</c>.</param>
        /// <param name="newlen"><c>new[1] - new[0]</c>.</param>
        /// <returns><c>(off, scl)</c>.</returns>
        /// <exception cref="DivideByZeroException">A zero Python <paramref name="oldlen"/>.</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        private static (PolyNumber Off, PolyNumber Scl) MapParmsFinish(in PolyNumber o1, in PolyNumber o0, in PolyNumber n1, in PolyNumber n0,
                                                                     in PolyNumber oldlen, in PolyNumber newlen)
        {
            var p = PolyNumber.Binary(BinaryOp.Multiply, o1, n0);
            var q = PolyNumber.Binary(BinaryOp.Multiply, o0, n1);
            var off = PolyNumber.Binary(BinaryOp.Divide, PolyNumber.Binary(BinaryOp.Subtract, p, q), oldlen);
            var scl = PolyNumber.Binary(BinaryOp.Divide, newlen, oldlen);
            return (off, scl);
        }

        // ---------------------------------------------------------------------------------------------
        //  The CPython fast lane of mapparms / mapdomain: Python int / float / complex on machine numbers
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     A Python int that fits <c>long</c>, a Python float or a Python complex — one operand of the domain-map
        ///     fast lane, which runs CPython's arithmetic on machine numbers instead of <see cref="BigInteger"/>-backed
        ///     <see cref="PolyNumber"/>s. The lane has no semantics of its own: every float/complex operation is the
        ///     SAME <see cref="PyScalar"/> helper the general lane calls (NaN operand priority included), an int
        ///     quotient is <c>long_true_divide</c>, and any int result that would leave <c>long</c> makes the lane
        ///     bail to the exact general lane — so it can only ever be faster, never different.
        /// </summary>
        internal readonly struct PyNum
        {
            /// <summary>The Python type.</summary>
            public readonly PyKind Kind;
            /// <summary>The int value (<see cref="PyKind.Int"/>).</summary>
            public readonly long I;
            /// <summary>The complex value (<see cref="PyKind.Complex"/>); a float keeps its value in the real part.</summary>
            public readonly Complex C;

            private PyNum(PyKind kind, long i, Complex c) { Kind = kind; I = i; C = c; }

            /// <summary>A Python int.</summary><param name="v">The value.</param><returns>The operand.</returns>
            public static PyNum Int(long v) => new PyNum(PyKind.Int, v, default);

            /// <summary>A Python float.</summary><param name="v">The value.</param><returns>The operand.</returns>
            public static PyNum Float(double v) => new PyNum(PyKind.Float, 0, new Complex(v, 0));

            /// <summary>A Python complex.</summary><param name="v">The value.</param><returns>The operand.</returns>
            public static PyNum Cplx(Complex v) => new PyNum(PyKind.Complex, 0, v);

            /// <summary>
            ///     CPython's <c>float(x)</c> of an int/float operand in a mixed operation: an int is rounded to the
            ///     nearest double, ties to even (<c>PyLong_AsDouble</c>; the conversion instruction .NET emits rounds
            ///     the same way, and is exact below 2^53). Not meaningful for a complex.
            /// </summary>
            public double AsDouble => Kind == PyKind.Int ? I : C.Real;

            /// <summary>CPython's <c>TO_COMPLEX</c>: an int/float operand of a complex operation becomes <c>(value, +0.0)</c>.</summary>
            public Complex AsComplex => Kind == PyKind.Complex ? C : new Complex(AsDouble, 0.0);

            /// <summary>This value as the <see cref="PolyNumber"/> the general lane would hold.</summary>
            /// <returns>The Python number.</returns>
            public PolyNumber ToPolyNumber() => PolyNumber.FromPython(Kind switch
            {
                PyKind.Int => PyScalar.Int(I),
                PyKind.Float => PyScalar.Float(C.Real),
                _ => PyScalar.Cplx(C),
            });

            /// <summary>This value boxed as <see cref="PolyNumber.ToObject"/> boxes a Python number
            ///     (<c>long</c> / <c>double</c> / <see cref="Complex"/>).</summary>
            /// <returns>The boxed value.</returns>
            public object ToObject() => Kind switch
            {
                // Each arm cast to object: long, double and Complex share the natural type Complex (both convert
                // to it implicitly), so an uncast switch would box EVERY result as a Complex.
                PyKind.Int => (object)I,
                PyKind.Float => (object)C.Real,
                _ => (object)C,
            };
        }

        /// <summary>
        ///     Reads a statically typed domain element into the lane when it is a Python int (a C# integer or bool —
        ///     CPython's bool arithmetic is int arithmetic), a Python float (<c>float</c>/<c>double</c>) or a Python
        ///     complex (<see cref="Complex"/>). Every other type — NumPy scalars (<see cref="Half"/>, <c>char</c>,
        ///     <c>decimal</c>), arrays, <see cref="BigInteger"/>, reference types — returns false.
        /// </summary>
        /// <typeparam name="T">The element's static type; each test folds away for a value-type instantiation.</typeparam>
        /// <param name="v">The element.</param>
        /// <param name="r">The operand.</param>
        /// <returns>Whether the element belongs to the lane (a <c>ulong</c> above <c>long.MaxValue</c> does not).</returns>
        private static bool TryNum<T>(T v, out PyNum r)
        {
            if (typeof(T) == typeof(int)) { r = PyNum.Int(Unsafe.As<T, int>(ref v)); return true; }
            if (typeof(T) == typeof(double)) { r = PyNum.Float(Unsafe.As<T, double>(ref v)); return true; }
            if (typeof(T) == typeof(long)) { r = PyNum.Int(Unsafe.As<T, long>(ref v)); return true; }
            if (typeof(T) == typeof(float)) { r = PyNum.Float(Unsafe.As<T, float>(ref v)); return true; }
            if (typeof(T) == typeof(Complex)) { r = PyNum.Cplx(Unsafe.As<T, Complex>(ref v)); return true; }
            if (typeof(T) == typeof(short)) { r = PyNum.Int(Unsafe.As<T, short>(ref v)); return true; }
            if (typeof(T) == typeof(sbyte)) { r = PyNum.Int(Unsafe.As<T, sbyte>(ref v)); return true; }
            if (typeof(T) == typeof(byte)) { r = PyNum.Int(Unsafe.As<T, byte>(ref v)); return true; }
            if (typeof(T) == typeof(ushort)) { r = PyNum.Int(Unsafe.As<T, ushort>(ref v)); return true; }
            if (typeof(T) == typeof(uint)) { r = PyNum.Int(Unsafe.As<T, uint>(ref v)); return true; }
            if (typeof(T) == typeof(bool)) { r = PyNum.Int(Unsafe.As<T, bool>(ref v) ? 1 : 0); return true; }
            if (typeof(T) == typeof(ulong) && Unsafe.As<T, ulong>(ref v) <= long.MaxValue) { r = PyNum.Int((long)Unsafe.As<T, ulong>(ref v)); return true; }
            if (typeof(T) == typeof(object)) return TryNum((object)v, out r);   // a (object, object) tuple's boxed elements
            r = default;
            return false;
        }

        /// <summary>
        ///     A Python-kind <see cref="PolyNumber"/> (a mapdomain point the facade classified) as a lane operand: a
        ///     float or complex always, an int when it fits <c>long</c>. NumPy scalars and arrays are not Python numbers.
        /// </summary>
        /// <param name="p">The value.</param>
        /// <param name="r">The operand.</param>
        /// <returns>Whether the value belongs to the lane.</returns>
        public static bool TryPyNum(in PolyNumber p, out PyNum r)
        {
            r = default;
            if (!p.IsPython)
                return false;
            switch (p.Py.Kind)
            {
                case PyKind.Float: r = PyNum.Float(p.Py.F); return true;
                case PyKind.Complex: r = PyNum.Cplx(p.Py.C); return true;
                default:
                    if (p.Py.I < s_longMin || p.Py.I > s_longMax)
                        return false;
                    r = PyNum.Int((long)p.Py.I);
                    return true;
            }
        }

        // The long range as BigIntegers, built once: comparing against a long literal converts it to a BigInteger on
        // EVERY comparison, and both bounds lie outside the int range, so each conversion allocates its digits.
        /// <summary><c>long.MinValue</c>.</summary>
        private static readonly BigInteger s_longMin = long.MinValue;
        /// <summary><c>long.MaxValue</c>.</summary>
        private static readonly BigInteger s_longMax = long.MaxValue;

        /// <summary><see cref="TryNum{T}"/> for a boxed element (a Python list/tuple item).</summary>
        /// <param name="o">The element.</param>
        /// <param name="r">The operand.</param>
        /// <returns>Whether the element belongs to the lane.</returns>
        private static bool TryNum(object o, out PyNum r)
        {
            switch (o)
            {
                case int v: r = PyNum.Int(v); return true;
                case double v: r = PyNum.Float(v); return true;
                case long v: r = PyNum.Int(v); return true;
                case float v: r = PyNum.Float(v); return true;
                case Complex v: r = PyNum.Cplx(v); return true;
                case short v: r = PyNum.Int(v); return true;
                case sbyte v: r = PyNum.Int(v); return true;
                case byte v: r = PyNum.Int(v); return true;
                case ushort v: r = PyNum.Int(v); return true;
                case uint v: r = PyNum.Int(v); return true;
                case bool v: r = PyNum.Int(v ? 1 : 0); return true;
                case ulong v when v <= long.MaxValue: r = PyNum.Int((long)v); return true;
                default: r = default; return false;
            }
        }

        /// <summary>
        ///     Items 0 and 1 of a PYTHON sequence domain — an <c>object[]</c> or other non-array <see cref="IList"/>
        ///     (a list) or an <see cref="ITuple"/> (a tuple) — when both are lane numbers. A typed C# array or an
        ///     NDArray is an ndarray (NumPy scalars, not Python numbers), and a short sequence is left to the general
        ///     lane's IndexError: this only ever answers "is the lane applicable", never raises.
        /// </summary>
        /// <param name="seq">The domain argument.</param>
        /// <param name="e0">Item 0.</param>
        /// <param name="e1">Item 1.</param>
        /// <returns>Whether both items were read.</returns>
        private static bool TryLaneDomain(object seq, out PyNum e0, out PyNum e1)
        {
            e0 = e1 = default;
            switch (seq)
            {
                case object[] list:
                    return list.Length >= 2 && TryNum(list[0], out e0) && TryNum(list[1], out e1);
                case ITuple tup:
                    return tup.Length >= 2 && TryNum(tup[0], out e0) && TryNum(tup[1], out e1);
                case IList list when seq is not System.Array:
                    return list.Count >= 2 && TryNum(list[0], out e0) && TryNum(list[1], out e1);
                default:
                    return false;
            }
        }

        /// <summary>CPython's <c>a + b</c> in the lane: exact for two ints, <see cref="PyScalar.ComplexSum"/> with a
        ///     complex, else <see cref="PyScalar.FloatAdd"/> (the general lane's helpers, NaN priority included).</summary>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <param name="r">The sum.</param>
        /// <returns>False when an int sum leaves <c>long</c> (the lane must bail to exact integers).</returns>
        private static bool TryAdd(in PyNum a, in PyNum b, out PyNum r)
        {
            if (a.Kind == PyKind.Int && b.Kind == PyKind.Int)
            {
                long s = unchecked(a.I + b.I);
                // Two's-complement overflow: operands of one sign whose sum took the other sign.
                if (((a.I ^ s) & (b.I ^ s)) < 0) { r = default; return false; }
                r = PyNum.Int(s);
                return true;
            }
            r = a.Kind == PyKind.Complex || b.Kind == PyKind.Complex
                ? PyNum.Cplx(PyScalar.ComplexSum(a.AsComplex, b.AsComplex))
                : PyNum.Float(PyScalar.FloatAdd(a.AsDouble, b.AsDouble));
            return true;
        }

        /// <summary>
        ///     CPython's <c>a - b</c> in the lane: exact for two ints, <see cref="PyScalar.ComplexDiff"/> with a
        ///     complex, else <see cref="PyScalar.FloatSub"/> — the SAME helpers the general lane uses, so the two agree
        ///     even on which of two NaN operands survives (a C# operator leaves that to RyuJIT's register allocation).
        /// </summary>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <param name="r">The difference.</param>
        /// <returns>False when an int difference leaves <c>long</c>.</returns>
        private static bool TrySub(in PyNum a, in PyNum b, out PyNum r)
        {
            if (a.Kind == PyKind.Int && b.Kind == PyKind.Int)
            {
                long d = unchecked(a.I - b.I);
                // Two's-complement overflow: operands of different signs whose difference took the subtrahend's sign.
                if (((a.I ^ b.I) & (a.I ^ d)) < 0) { r = default; return false; }
                r = PyNum.Int(d);
                return true;
            }
            r = a.Kind == PyKind.Complex || b.Kind == PyKind.Complex
                ? PyNum.Cplx(PyScalar.ComplexDiff(a.AsComplex, b.AsComplex))
                : PyNum.Float(PyScalar.FloatSub(a.AsDouble, b.AsDouble));
            return true;
        }

        /// <summary>CPython's <c>a * b</c> in the lane: exact for two ints, <see cref="PyScalar.ComplexProd"/> (the
        ///     naive, unfused product) with a complex, else <see cref="PyScalar.FloatMul"/>.</summary>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <param name="r">The product.</param>
        /// <returns>False when an int product leaves <c>long</c>.</returns>
        private static bool TryMul(in PyNum a, in PyNum b, out PyNum r)
        {
            if (a.Kind == PyKind.Int && b.Kind == PyKind.Int)
            {
                long hi = Math.BigMul(a.I, b.I, out long lo);
                // The 128-bit product fits a long exactly when its high half is the low half's sign extension.
                if (hi != (lo >> 63)) { r = default; return false; }
                r = PyNum.Int(lo);
                return true;
            }
            r = a.Kind == PyKind.Complex || b.Kind == PyKind.Complex
                ? PyNum.Cplx(PyScalar.ComplexProd(a.AsComplex, b.AsComplex))
                : PyNum.Float(PyScalar.FloatMul(a.AsDouble, b.AsDouble));
            return true;
        }

        /// <summary>
        ///     CPython's true division <c>a / b</c> in the lane — never an int. Two ints go through
        ///     <c>long_true_divide</c>'s rules: magnitudes below 2^53 are exact doubles, so one IEEE division is the
        ///     correctly rounded quotient CPython computes (a zero dividend keeps the XOR sign: <c>0 / -5</c> is
        ///     <c>-0.0</c>), anything larger takes the exact algorithm (<see cref="PyScalar.IntTrueDivide"/>). A complex
        ///     operand goes through <see cref="PyScalar.ComplexQuotient"/> (3.12's <c>_Py_c_quot</c>), a float one
        ///     through <see cref="PyScalar.FloatDiv"/> — the general lane's helpers.
        /// </summary>
        /// <param name="a">Dividend.</param>
        /// <param name="b">Divisor.</param>
        /// <returns>The quotient (a float or a complex).</returns>
        /// <exception cref="DivideByZeroException">A zero divisor, with CPython 3.12's text for the operand kinds:
        ///     <c>division by zero</c> (two ints), <c>complex division by zero</c>, <c>float division by zero</c>
        ///     (-0.0 is a zero divisor).</exception>
        private static PyNum Div(in PyNum a, in PyNum b)
        {
            if (a.Kind == PyKind.Int && b.Kind == PyKind.Int)
            {
                const long Small = 1L << 53;
                if (b.I != 0 && a.I > -Small && a.I < Small && b.I > -Small && b.I < Small)
                    return PyNum.Float((double)a.I / b.I);
                return PyNum.Float(PyScalar.IntTrueDivide(a.I, b.I));   // also raises CPython's int ZeroDivisionError
            }
            if (a.Kind == PyKind.Complex || b.Kind == PyKind.Complex)
                return PyNum.Cplx(PyScalar.ComplexQuotient(a.AsComplex, b.AsComplex));
            return PyNum.Float(PyScalar.FloatDiv(a.AsDouble, b.AsDouble));   // raises CPython's float ZeroDivisionError
        }

        /// <summary>
        ///     mapparms in the lane: Python's <c>oldlen = old[1] - old[0]; newlen = new[1] - new[0];
        ///     off = (old[1]*new[0] - old[0]*new[1]) / oldlen; scl = newlen / oldlen</c> on machine numbers — both
        ///     results a Python float or complex (true division never yields an int).
        /// </summary>
        /// <param name="o1"><c>old[1]</c>.</param>
        /// <param name="o0"><c>old[0]</c>.</param>
        /// <param name="n1"><c>new[1]</c>.</param>
        /// <param name="n0"><c>new[0]</c>.</param>
        /// <param name="off">The offset.</param>
        /// <param name="scl">The scale.</param>
        /// <returns>False when an int intermediate leaves <c>long</c> — nothing observable happened, and the caller
        ///     runs the general lane, which recomputes everything exactly.</returns>
        /// <exception cref="DivideByZeroException">A zero-length old domain, raised by off's division exactly where
        ///     Python raises it (the subtractions and products before it cannot raise for Python numbers).</exception>
        private static bool TryMapParmsLane(in PyNum o1, in PyNum o0, in PyNum n1, in PyNum n0, out PyNum off, out PyNum scl)
        {
            off = scl = default;
            if (!TrySub(o1, o0, out var oldlen) || !TrySub(n1, n0, out var newlen)
                || !TryMul(o1, n0, out var p) || !TryMul(o0, n1, out var q) || !TrySub(p, q, out var num))
                return false;
            off = Div(num, oldlen);
            scl = Div(newlen, oldlen);
            return true;
        }

        /// <summary>
        ///     The lane for statically typed 2-tuple domains: applicable when every element is a Python int, float or
        ///     complex (see <see cref="TryNum{T}"/>).
        /// </summary>
        /// <typeparam name="T0">Type of <c>old[0]</c>.</typeparam>
        /// <typeparam name="T1">Type of <c>old[1]</c>.</typeparam>
        /// <typeparam name="T2">Type of <c>new[0]</c>.</typeparam>
        /// <typeparam name="T3">Type of <c>new[1]</c>.</typeparam>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <param name="off">The offset.</param>
        /// <param name="scl">The scale.</param>
        /// <returns>False when an element is not a lane number or an int intermediate leaves <c>long</c> (run the
        ///     general lane).</returns>
        /// <exception cref="DivideByZeroException">A zero-length old domain (Python's ZeroDivisionError).</exception>
        public static bool TryMapParmsFast<T0, T1, T2, T3>(in (T0, T1) old, in (T2, T3) @new, out PyNum off, out PyNum scl)
        {
            off = scl = default;
            return TryNum(old.Item2, out var o1) && TryNum(old.Item1, out var o0)
                && TryNum(@new.Item2, out var n1) && TryNum(@new.Item1, out var n0)
                && TryMapParmsLane(o1, o0, n1, n0, out off, out scl);
        }

        /// <summary>The lane for boxed domains: applicable when both are Python sequences (lists / tuples, see
        ///     <see cref="TryLaneDomain"/>) of lane numbers.</summary>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <param name="off">The offset.</param>
        /// <param name="scl">The scale.</param>
        /// <returns>False when the lane does not apply (run the general lane).</returns>
        /// <exception cref="DivideByZeroException">A zero-length old domain.</exception>
        public static bool TryMapParmsFast(object old, object @new, out PyNum off, out PyNum scl)
        {
            off = scl = default;
            return TryLaneDomain(old, out var o0, out var o1) && TryLaneDomain(@new, out var n0, out var n1)
                && TryMapParmsLane(o1, o0, n1, n0, out off, out scl);
        }

        /// <summary>mapdomain's <c>off + scl*x</c> in the lane (a Python point and lane offset/scale).</summary>
        /// <param name="x">The point.</param>
        /// <param name="off">The offset.</param>
        /// <param name="scl">The scale.</param>
        /// <param name="r">The mapped point.</param>
        /// <returns>False when an int intermediate leaves <c>long</c> (cannot happen for mapparms' float/complex
        ///     offset and scale; kept for the lane's contract).</returns>
        public static bool TryMapDomainLane(in PyNum x, in PyNum off, in PyNum scl, out PyNum r)
        {
            r = default;
            return TryMul(scl, x, out var m) && TryAdd(off, m, out r);
        }

        /// <summary>
        ///     mapparms of two 1-D ndarray domains of ONE non-bool dtype — numpy.polynomial's own case
        ///     (<c>ABCPolyBase.domain</c>/<c>window</c> are float64 arrays): all four elements are NumPy scalars of that
        ///     dtype, so the six scalarmath operations run as one fused kernel
        ///     (<see cref="DirectILKernelGenerator.GetPolyMapParmsKernel"/>), bit-identical to the per-operation chain.
        ///     Reading <c>new</c> before computing <c>oldlen</c> is safe: a same-dtype non-bool subtraction cannot
        ///     raise, and lengths ≥ 2 are checked, so no observable event changes order.
        /// </summary>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <param name="off">The offset (a NumPy scalar of the true-division dtype).</param>
        /// <param name="scl">The scale.</param>
        /// <returns>False when the domains are not two such arrays (run the general lane).</returns>
        /// <exception cref="DivideByZeroException">A decimal domain of zero length (NumSharp's decimal division
        ///     raises; NumPy has no decimal dtype).</exception>
        public static bool TryMapParmsScalars(object old, object @new, out PolyNumber off, out PolyNumber scl)
        {
            off = scl = default;
            if (old is not NDArray oa || @new is not NDArray na)
                return false;
            NPTypeCode t = oa.typecode;
            if (t == NPTypeCode.Boolean || na.typecode != t || oa.ndim != 1 || na.ndim != 1
                || oa.Shape.dimensions[0] < 2 || na.Shape.dimensions[0] < 2)
                return false;
            var o0 = PolySeriesView.Element(oa, 0);
            var o1 = PolySeriesView.Element(oa, 1);
            var n0 = PolySeriesView.Element(na, 0);
            var n1 = PolySeriesView.Element(na, 1);
            PolyRaw16 ro, rs;
            DirectILKernelGenerator.GetPolyMapParmsKernel(t)(o1.Ptr, o0.Ptr, n1.Ptr, n0.Ptr, (byte*)&ro, (byte*)&rs);
            NPTypeCode rt = PolyTyping.IsIntLike(t) ? NPTypeCode.Double : t;
            off = PolyNumber.FromScalar(rt, &ro);
            scl = PolyNumber.FromScalar(rt, &rs);
            return true;
        }

        // ---------------------------------------------------------------------------------------------
        //  polyutils.mapdomain
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>polyutils.mapdomain(x, old, new)</c>: <c>off + scl * x</c> with <c>(off, scl) = mapparms(old, new)</c>
        ///     (see <see cref="MapDomainWith"/>).
        /// </summary>
        /// <param name="x">The points: an ndarray (NumPy converts every non-scalar with np.asanyarray), or a
        ///     Python / NumPy scalar (left unconverted, so its arithmetic is CPython's / scalarmath).</param>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns>The mapped points, of NumPy's kind.</returns>
        /// <exception cref="IndexError">A domain shorter than 2.</exception>
        /// <exception cref="TypeError">A domain that is not subscriptable.</exception>
        /// <exception cref="DivideByZeroException">A zero-length Python domain.</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        public static PolyNumber MapDomain(in PolyNumber x, object old, object @new)
        {
            var (off, scl) = MapParms(old, @new);
            return MapDomainWith(x, off, scl);
        }

        /// <summary>
        ///     <c>off + scl * x</c> as NumPy evaluates it. An ndarray x of rank ≥ 1 runs ONE fused pass
        ///     (<see cref="np.evaluate(NDExpr, NDArray)"/>) whose dtypes are NumPy's two-ufunc sequence's: off and scl
        ///     are pre-converted to their operation's loop dtype (for a Python value, the dtype NEP 50 gives it against
        ///     x; for a NumPy scalar, the promoted dtype the ufunc casts it to), so the fused kernel's strong 0-d
        ///     parameters promote exactly as the originals do — unless NEP 50 makes an operation complex64, which runs
        ///     on the float32 kernels of <see cref="MapDomainComplex64"/>, or both operations share a float64 / float32 /
        ///     complex128 loop over 1-D or C-contiguous points, which <see cref="MapDomainAffine"/> maps block by block
        ///     with the same roundings and no iterator. Any other x (or an array-valued off/scl from a 2-D
        ///     domain) runs the operator dispatch of <see cref="PolyNumber.Binary"/> — a 0-d x included: NumPy's
        ///     <c>scl * x</c> is then a one-element ufunc whose result is a NumPy SCALAR, and <c>off + that</c> is
        ///     scalarmath, a sequence the scalar engine reproduces op for op (a complex64 off/scl stays in float32 —
        ///     the fused pass would compute it in complex128), and faster than an iterator pass for one element.
        /// </summary>
        /// <param name="x">The points.</param>
        /// <param name="off">The offset (from mapparms).</param>
        /// <param name="scl">The scale (from mapparms).</param>
        /// <returns>The mapped points.</returns>
        /// <exception cref="IncorrectShapeException">Array operands that do not broadcast.</exception>
        /// <exception cref="OverflowException">A Python int that does not fit its loop dtype.</exception>
        public static PolyNumber MapDomainWith(in PolyNumber x, in PolyNumber off, in PolyNumber scl)
        {
            if (!x.IsNdArray || off.IsNdArray || scl.IsNdArray)
                return PolyNumber.Binary(BinaryOp.Add, off, PolyNumber.Binary(BinaryOp.Multiply, scl, x));

            NDArray xa = x.Array;
            NPTypeCode t1 = scl.IsPython ? PolyNumber.WeakPromote(xa.typecode, scl.Py) : PolyTyping.Promote(scl.Dtype, xa.typecode);
            NPTypeCode t2 = off.IsPython ? PolyNumber.WeakPromote(t1, off.Py) : PolyTyping.Promote(off.Dtype, t1);
            // Both parameters enter the fused pass already converted to their operation's loop dtype — the cast the
            // ufunc applies to them — so the kernel promotes to the same t1 / t2 and converts nothing per element (a
            // float32 NumPy scale left as float32 against a complex128 x was re-widened to complex at every point).
            NDArray sclP = ParamArray(0, scl, t1);
            NDArray offP = ParamArray(1, off, t2);

            // A complex loop may be NumPy's COMPLEX64 one (NEP 50: a Python complex with a float16/float32 x, a
            // complex64 off/scl with a narrow x): the fused pass would compute it in complex128, so it runs on the
            // float32 kernels instead (see MapDomainComplex64).
            if (t2 == NPTypeCode.Complex)
            {
                bool mulC64 = t1 == NPTypeCode.Complex && PolyNumber.IsComplex64Loop(scl, xa.typecode, false);
                bool addC64 = PolyNumber.IsComplex64Loop(off, mulC64 ? NPTypeCode.Complex : t1, mulC64);
                if (mulC64 || addC64)
                    return PolyNumber.FromArray(MapDomainComplex64(xa, off, scl, t1, sclP, offP, mulC64, addC64));

                // A complex128 x (contiguous, or any 1-D stride) with complex128 off/scl: the fused pass computes
                // simd_cmul one element at a time (~0.8 ns each); the dedicated kernel runs the same arithmetic two
                // values per AVX2 vector.
                if (t1 == NPTypeCode.Complex && xa.typecode == NPTypeCode.Complex && xa.size > 0
                    && (xa.Shape.IsContiguous || xa.ndim == 1))
                {
                    var r = new NDArray(NPTypeCode.Complex, new Shape((long[])xa.Shape.dimensions.Clone()), false);
                    DirectILKernelGenerator.PolyComplex128Affine(
                        (Complex*)((byte*)xa.Storage.Address + xa.Shape.offset * sizeof(Complex)), xa.size,
                        xa.Shape.IsContiguous ? 1 : xa.Shape.strides[0],
                        *(Complex*)((byte*)sclP.Storage.Address + sclP.Shape.offset * sizeof(Complex)),
                        *(Complex*)((byte*)offP.Storage.Address + offP.Shape.offset * sizeof(Complex)),
                        (Complex*)r.Storage.Address);
                    return PolyNumber.FromArray(r);
                }
            }

            // A float64 / float32 / complex128 loop whose offset shares the scale's loop dtype, over points that are
            // 1-D (any stride) or C-contiguous: convert the points to the loop dtype block by block into an L1 scratch
            // and map each block with the dedicated affine kernel — one read of x and one write of the result, with
            // no temporary and no iterator setup (see MapDomainAffine). A complex128 x never gets here in those
            // layouts (the branch above takes it). The shapes the fused pass already runs at vector speed stay on it
            // (see FusedPassIsFaster).
            if (t2 == t1 && t1 is NPTypeCode.Double or NPTypeCode.Single or NPTypeCode.Complex && xa.size > 0
                && (xa.ndim == 1 || xa.Shape.IsContiguous) && !FusedPassIsFaster(xa, t1) && !DisableAffineMapDomain)
                return PolyNumber.FromArray(MapDomainAffine(xa, t1, sclP, offP));

            NDArray staged = StageMapDomainPoints(xa, t1);
            try
            {
                return PolyNumber.FromArray(np.evaluate(NDExpr.Arr(offP) + NDExpr.Arr(sclP) * NDExpr.Arr(staged ?? xa)));
            }
            finally
            {
                staged?.Dispose();
            }
        }

        /// <summary>
        ///     Bytes per block of <see cref="MapDomainAffine"/> — 1,024 float64 / 2,048 float32 / 512 complex128
        ///     points. The converted block must stay in L1 together with the SOURCE lines the conversion reads, which
        ///     for a stride-2 x are twice its bytes: 16 KB blocks (2,048 float64) measured 0.72× the fused pass on a
        ///     100K stride-2 float64 x where 8 KB blocks kept parity, because 32 KB of source plus 16 KB of scratch
        ///     overflows a 48 KB L1.
        /// </summary>
        private const int AffineBlockBytes = 8192;

        /// <summary>
        ///     Point count from which a NON-contiguous x whose conversion the fused pass does in registers (x already of
        ///     the loop dtype, or a vector widen edge) goes back to the fused pass — the point from which that pass
        ///     packs a strided x contiguous (<see cref="StagedStridedMinPoints"/>), walks a reversed view in memory
        ///     order (NDIter flips the negative stride) or gathers a same-width strided one, all at vector speed, so
        ///     the affine route's copy of every block into scratch only adds work. Below it the fused pass's per-call
        ///     setup and scalar strided loop cost more (measured in one process, fused/affine: 2,048 points 1.06-2.29,
        ///     4,096 points 0.92-2.76; at 65,536 points 0.82-0.95 except int32 → float64 1.08-1.16). One threshold for
        ///     every pair: the per-pair crossovers spread from ~4K (float64) to ~64K (float32), and at 8,192 the
        ///     route it gives away is never more than ~20% faster.
        /// </summary>
        private const long AffineStridedFusedMinPoints = StagedStridedMinPoints;

        /// <summary>
        ///     Whether mapdomain's fused np.evaluate pass maps <paramref name="x"/> at least as fast as
        ///     <see cref="MapDomainAffine"/> — measured in one process, same arrays, interleaved. On every other shape
        ///     the affine route is the faster one (fused/affine over 249 shapes: geomean 1.34 at 16 points, 1.53-1.62
        ///     at 1,000, 1.36-1.46 at 100K, up to 2.1 on a 100K contiguous int8/int16/int64/uint64 x; the worst cell
        ///     0.94, a 100K stride-2 float16 x).
        /// </summary>
        /// <param name="x">The points (1-D, or C-contiguous).</param>
        /// <param name="t1">The loop dtype (float64, float32 or complex128).</param>
        /// <returns>True for: a CONTIGUOUS x the fused kernel widens with a vector edge (int32/uint32/float32 →
        ///     float64, int16/uint16 → float32 — widened in registers, no scratch pass: 1,000 points fused 0.60-0.64 µs,
        ///     affine 0.66-0.80) or of bool (0.63-0.73× on the affine route at 100K); a NON-contiguous x already of
        ///     the loop dtype or on a vector edge from <see cref="AffineStridedFusedMinPoints"/> points.</returns>
        private static bool FusedPassIsFaster(NDArray x, NPTypeCode t1)
        {
            NPTypeCode xt = x.typecode;
            if (x.Shape.IsContiguous)
                return NDExprVec.HasWidenEdge(xt, t1) || xt == NPTypeCode.Boolean;
            return (xt == t1 || NDExprVec.HasWidenEdge(xt, t1)) && x.size >= AffineStridedFusedMinPoints;
        }

        /// <summary>
        ///     Test and benchmark switch for <see cref="MapDomainAffine"/>. While <see langword="true"/> on the calling
        ///     thread, mapdomain skips the affine route and runs the fused np.evaluate pass it replaced — which is what
        ///     lets a test compare the two routes on one input, byte for byte, and a benchmark time them in ONE process
        ///     (100K-point timings move 1.5-2× between runs with the allocator's page state, so two separate runs
        ///     cannot be compared).
        /// </summary>
        /// <remarks>Thread-static on purpose: MSTest runs test classes in parallel, and a process-wide switch flipped by
        ///     one test would silently re-route another test's mapdomain calls. Production code never sets it.</remarks>
        [ThreadStatic] internal static bool DisableAffineMapDomain;

        /// <summary>
        ///     Counts the calls <see cref="MapDomainAffine"/> served on the calling thread. A test reads it before and
        ///     after a mapdomain call to prove the affine route actually ran — the fused pass computes the same answer,
        ///     so a silent decline would pass every value check.
        /// </summary>
        [ThreadStatic] internal static long AffineMapDomainRuns;

        /// <summary>
        ///     mapdomain's <c>off + scl*x</c> for a float64, float32 or complex128 loop <paramref name="t"/> shared by
        ///     both operations: a new array of x's shape holding, for each point, the point converted to
        ///     <paramref name="t"/> (NumPy's loop cast), multiplied by the scale and then offset — NumPy's two ufunc
        ///     calls, rounded as they round them.
        /// </summary>
        /// <param name="x">The points: 1-D (any stride) or C-contiguous, at least one element.</param>
        /// <param name="t">The loop dtype of both operations (float64, float32 or complex128).</param>
        /// <param name="sclP">The scale as a 0-d array of <paramref name="t"/>.</param>
        /// <param name="offP">The offset as a 0-d array of <paramref name="t"/>.</param>
        /// <returns>A new C-contiguous array of <paramref name="t"/> with x's shape.</returns>
        /// <exception cref="NotSupportedException">A points dtype the house conversion does not support.</exception>
        /// <remarks>
        ///     <para>The points are read in their memory order and the result written in the same order, so a 1-D
        ///     view of any stride (reversed included) maps element for element; an N-D x must be C-contiguous (its
        ///     flat order IS its logical order) — NumPy's K-order output for any other N-D layout keeps that layout,
        ///     which the fused pass reproduces instead.</para>
        ///     <para>Points already of <paramref name="t"/> and contiguous are mapped in place, with no conversion;
        ///     any other points are converted <see cref="AffineBlockBytes"/> at a time by <see cref="CopyInto"/> — the house
        ///     SIMD casts and strided copies (int64 → float64 is the exponent-splice cast, a reversed or stride-2 view
        ///     the SIMD reverse / deinterleave copy) — into a stack scratch the kernel then reads, so the converted
        ///     points never leave L1. The fused pass it replaces converted a dtype without a vector widen edge one
        ///     point at a time, or staged it through a whole temporary array first.</para>
        ///     <para>A float64/float32 loop multiplies then adds with separate roundings (<see
        ///     cref="DirectILKernelGenerator.PolyAffineDouble"/>); a complex128 loop is <c>CDOUBLE_multiply</c>'s
        ///     scalar-broadcast <c>simd_cmul</c> then the component-wise add (<see
        ///     cref="DirectILKernelGenerator.PolyComplex128Affine"/>), the points converted to complex with a +0.0
        ///     imaginary part as NumPy's loop cast gives them.</para>
        ///     <para>[SkipLocalsInit]: every scratch block is written by the conversion before the kernel reads it; the
        ///     runtime's zeroing of the 8 KB scratch would otherwise be most of the cost of a short call.</para>
        /// </remarks>
        [SkipLocalsInit]
        private static NDArray MapDomainAffine(NDArray x, NPTypeCode t, NDArray sclP, NDArray offP)
        {
            int tsize = DirectILKernelGenerator.GetTypeSize(t);
            byte* scl = (byte*)sclP.Storage.Address + sclP.Shape.offset * tsize;
            byte* off = (byte*)offP.Storage.Address + offP.Shape.offset * tsize;
            var r = new NDArray(t, new Shape((long[])x.Shape.dimensions.Clone()), false);
            byte* dst = (byte*)r.Storage.Address;
            long n = x.size;
            // The points as one run: a 1-D view keeps its own (signed) stride; a C-contiguous N-D array is read flat
            // (its elements are consecutive from the view's first one).
            PolySeriesView v = x.ndim == 1 ? PolySeriesView.Of(x) : PolySeriesView.Flat(x);
            AffineMapDomainRuns++;
            if (v.Dtype == t && v.Stride == tsize)
            {
                // Already the loop dtype, contiguous: no conversion — the kernel reads the points directly.
                Affine(t, v.Ptr, n, scl, off, dst);
                return r;
            }
            int block = AffineBlockBytes / tsize;
            byte* scratch = stackalloc byte[AffineBlockBytes];
            for (long k = 0; k < n; k += block)
            {
                long m = Math.Min(block, n - k);
                CopyInto(v.Slice(k, m), m, t, scratch);
                Affine(t, scratch, m, scl, off, dst + k * tsize);
            }
            return r;
        }

        /// <summary>
        ///     Runs the affine kernel of the loop dtype <paramref name="t"/> over <paramref name="n"/> contiguous points
        ///     already converted to it.
        /// </summary>
        /// <param name="t">float64, float32 or complex128.</param>
        /// <param name="x">The points.</param>
        /// <param name="n">Point count.</param>
        /// <param name="scl">Address of the scale (one element of <paramref name="t"/>).</param>
        /// <param name="off">Address of the offset (one element of <paramref name="t"/>).</param>
        /// <param name="r">Destination (<paramref name="n"/> elements of <paramref name="t"/>).</param>
        private static void Affine(NPTypeCode t, byte* x, long n, byte* scl, byte* off, byte* r)
        {
            // The three loop kinds mapdomain's real/complex128 arithmetic has; float16 and complex64 loops never
            // come here (the float16 fused pass and MapDomainComplex64 serve them).
            if (t == NPTypeCode.Double)
                DirectILKernelGenerator.PolyAffineDouble((double*)x, n, *(double*)scl, *(double*)off, (double*)r);
            else if (t == NPTypeCode.Single)
                DirectILKernelGenerator.PolyAffineSingle((float*)x, n, *(float*)scl, *(float*)off, (float*)r);
            else
                DirectILKernelGenerator.PolyComplex128Affine((Complex*)x, n, 1, *(Complex*)scl, *(Complex*)off, (Complex*)r);
        }

        /// <summary>
        ///     The points array handed to mapdomain's fused pass in place of <paramref name="x"/> when the fused
        ///     kernel would otherwise run an element-at-a-time path the house SIMD kernels can avoid — or null to
        ///     use <paramref name="x"/> as it is. The substitute holds the SAME values, exactly converted to the
        ///     multiply's loop dtype <paramref name="t1"/> (or left in x's dtype when the fused kernel widens it with
        ///     vector code), so the kernel still computes in t1 and the result's bits are unchanged.
        /// </summary>
        /// <param name="x">The points (rank ≥ 1).</param>
        /// <param name="t1">The loop dtype of <c>scl * x</c> (NumPy's; the fused kernel converts x to it anyway, and
        ///     the 0-d scale it multiplies by is already of t1, so a t1 substitute promotes to the same loop).</param>
        /// <returns>A new C-contiguous array (the caller disposes it), or null.</returns>
        /// <remarks>
        ///     <para>Most real and complex128 loops never get here — <see cref="MapDomainAffine"/> maps 1-D and
        ///     C-contiguous points directly. What still runs the fused pass: a float16 loop, an offset whose loop
        ///     dtype differs from the scale's, an N-D non-contiguous x (whose K-order result layout the fused pass
        ///     keeps), and a contiguous x the fused kernel widens with a vector edge. For those:</para>
        ///     <para>The fused kernel vectorizes over x only when x is contiguous and either already of t1 or of a dtype
        ///     it widens to t1 with a vector edge (<see cref="NDExprVec.HasWidenEdge"/>: int16/uint16 → float32,
        ///     int32/uint32/float32 → float64); its strided path for a mixed-width tree, and for float16, is one element
        ///     at a time. So (100K points, NumPy in brackets):</para>
        ///     <para>• a contiguous x of at least <see cref="StagedContiguousMinPoints"/> points (512 for float16, 1,024
        ///     into a float32 loop, 4,096 into a float64 one) whose conversion has no
        ///     vector edge but a vectorized house cast (<see cref="DirectILKernelGenerator.IsVectorizedContiguousCast"/>:
        ///     float16 → float32/float64, int8/uint8 → float32, 8/16-bit and 64-bit integers → float64) is cast to t1
        ///     first with one direct SIMD cast — measured before the affine route took these shapes over: float16 →
        ///     float32 140 → 25 µs [92 µs], int8 → float32 38 → 9.5 µs [44 µs]; shorter ones keep the fused scalar
        ///     conversion, which is cheaper than a temporary;</para>
        ///     <para>• a non-contiguous x in a mixed-width or float16 tree is copied contiguous first with the house SIMD
        ///     strided copy (reverse / deinterleave / gather, elements of at most 8 bytes), then cast to t1 when its edge
        ///     is not a vector one — a stride-2 float16 x in a float16 loop 1,070 → 251 µs [620 µs];</para>
        ///     <para>• anything else is used as is: a same-dtype x (the fused kernel's contiguous and gather paths are
        ///     vector already — except float16's strided one, handled above), or a conversion the house has no vector
        ///     cast for (int64/uint64 → float32, anything → complex).</para>
        /// </remarks>
        private static NDArray StageMapDomainPoints(NDArray x, NPTypeCode t1)
        {
            if (x.size < DirectILKernelGenerator.PolyHouseKernelThreshold)
                return null;
            NPTypeCode xt = x.typecode;
            bool contiguous = x.Shape.IsContiguous;
            if (xt == t1)
                return !contiguous && DirectILKernelGenerator.PolyScalarIsEmulated(xt) ? x.copy() : null;

            bool vectorEdge = NDExprVec.HasWidenEdge(xt, t1);
            bool vectorCast = DirectILKernelGenerator.IsVectorizedContiguousCast(xt, t1);
            if (contiguous)
            {
                // A contiguous x whose conversion the fused kernel runs one element at a time (no vector edge) but the
                // house casts vectorize: ONE SIMD cast straight into the new buffer — not x.astype(t1), whose setup
                // (~0.3 µs) is most of the cost of converting a thousand points — and only for enough points that the
                // temporary pays (see StagedContiguousMinPoints).
                if (vectorEdge || !vectorCast || x.size < StagedContiguousMinPoints(xt, t1))
                    return null;
                var cast = DirectILKernelGenerator.GetPolyContiguousCast(xt, t1);
                if (cast == null)
                    return null;
                var staged = new NDArray(t1, new Shape((long[])x.Shape.dimensions.Clone()), false);
                cast((byte*)x.Storage.Address + x.Shape.offset * DirectILKernelGenerator.GetTypeSize(xt), (byte*)staged.Storage.Address, x.size);
                return staged;
            }

            // Non-contiguous in a mixed-width tree: the fused strided path would be scalar. Pack it contiguous with the
            // SIMD strided copy (bit-exact) — converted to t1 on the way unless the fused kernel's widen edge is a
            // vector one already. A 1-D x (mapdomain's usual points) goes through CopyInto's blocked copy-then-cast:
            // ONE temporary, the converted block never leaving L1. Only worth an allocation for a long x: at 1,000
            // points the scalar strided pass (0.9 µs) beats pack + fused pass (1.6 µs).
            if (x.size < StagedStridedMinPoints || DirectILKernelGenerator.GetTypeSize(xt) > 8 || (!vectorEdge && !vectorCast))
                return null;
            NPTypeCode target = vectorEdge ? xt : t1;
            if (x.ndim == 1)
                return CopyAs(PolySeriesView.Of(x), x.size, target);
            NDArray packed = x.copy();
            if (vectorEdge)
                return packed;
            try
            {
                return packed.astype(t1);
            }
            finally
            {
                packed.Dispose();
            }
        }

        /// <summary>
        ///     Point count from which <see cref="StageMapDomainPoints"/> packs a NON-contiguous x before mapdomain's fused
        ///     pass: the pack allocates a temporary, which pays for itself only once the scalar strided pass it replaces
        ///     costs more than an allocation (measured: loses at 1,000 points, wins from ~8K).
        /// </summary>
        private const long StagedStridedMinPoints = 8192;

        /// <summary>
        ///     Point count from which <see cref="StageMapDomainPoints"/> casts a CONTIGUOUS x of dtype
        ///     <paramref name="xt"/> to the loop dtype <paramref name="t1"/> before mapdomain's fused pass. The staged
        ///     route costs a temporary (~0.3 µs) on top of a SIMD cast and the fused pass's vector body; what it saves
        ///     is the fused kernel's per-point scalar conversion, so the break-even moves with that conversion's price:
        /// </summary>
        /// <param name="xt">x's dtype.</param>
        /// <param name="t1">The multiply's loop dtype.</param>
        /// <returns>The minimum point count for staging.</returns>
        /// <remarks>
        ///     Measured (µs, staged vs fused): a float16 x — the emulated scalar widen, ~1.5 ns a point — wins from
        ///     ~256-512 points (f16 → f32 at 512: 0.91 vs 1.33; f16 → f64 at 512: 1.01 vs 1.33; at 256 a tie); an
        ///     integer x into a float32 loop — whose vector body is 8 lanes — from ~1,000 (u8 → f32 at 1,000: 0.90 vs
        ///     0.99; at 4,000: 1.26 vs 2.16); into a float64 loop, where the fused scalar conversion costs only ~0.45 ns
        ///     a point, from ~4,000 (int8/uint8/int16/uint16/char/int64/uint64 at 1,000: 0.96-1.2 vs 0.87-0.96; at
        ///     16,384: 4.9-6.4 vs 6.6-7.2).
        /// </remarks>
        private static long StagedContiguousMinPoints(NPTypeCode xt, NPTypeCode t1)
            => DirectILKernelGenerator.PolyScalarIsEmulated(xt) ? 512
                : DirectILKernelGenerator.GetTypeSize(t1) <= 4 ? 1024 : 4096;

        /// <summary>
        ///     mapdomain's <c>off + scl*x</c> for an ndarray x when NumPy runs at least one of the two operations in
        ///     its COMPLEX64 loop — NumPy's values, carried in a complex128 array (NumSharp has no complex64 dtype, #569;
        ///     the result dtype is the one documented divergence). Three shapes arise:
        ///     <list type="bullet">
        ///         <item><c>scl*x</c> complex64 and <c>off + …</c> complex64 — both steps on the float32 kernels;</item>
        ///         <item><c>scl*x</c> complex64 and <c>off + …</c> complex128 (a complex128 / float64 NumPy off) — the
        ///         complex64 product, widened exactly, then the fused complex128 add;</item>
        ///         <item><c>scl*x</c> real (float16/float32) and <c>off + …</c> complex64 — the real product through
        ///         the fused pass, then the complex64 add of the real values (imaginary part +0).</item>
        ///     </list>
        /// </summary>
        /// <param name="x">The points (rank ≥ 1).</param>
        /// <param name="off">The offset.</param>
        /// <param name="scl">The scale.</param>
        /// <param name="t1">The product's loop dtype as NumSharp spells it (<see cref="NPTypeCode.Complex"/> for complex64).</param>
        /// <param name="sclP">The 0-d scale parameter.</param>
        /// <param name="offP">The 0-d offset parameter (the complex128-add shape's fused pass).</param>
        /// <param name="mulC64">The product runs in complex64.</param>
        /// <param name="addC64">The sum runs in complex64.</param>
        /// <returns>A new complex128 array of x's shape (C-contiguous).</returns>
        /// <exception cref="OverflowException">A Python value that does not fit its loop dtype.</exception>
        /// <remarks>
        ///     x enters a complex64 product as float32 values: every dtype NEP 50 lets into that loop (bool, the 8/16-bit
        ///     integers, char, float16, float32) converts to float32 exactly, so a float32 copy of x (or x itself when it
        ///     already is a contiguous float32 array) IS NumPy's cast-to-complex64 buffer minus its +0 imaginary parts,
        ///     which the kernels apply implicitly. A Python complex / float operand enters rounded per component to
        ///     float32 (<see cref="Complex64Of"/>), as NumPy converts it into the complex64 loop.
        /// </remarks>
        private static NDArray MapDomainComplex64(NDArray x, in PolyNumber off, in PolyNumber scl, NPTypeCode t1,
                                                  NDArray sclP, NDArray offP, bool mulC64, bool addC64)
        {
            long n = x.size;
            var r = new NDArray(NPTypeCode.Complex, new Shape((long[])x.Shape.dimensions.Clone()), false);
            var rp = (Complex*)r.Storage.Address;
            NDArray temp = null, product = null;
            try
            {
                if (mulC64)
                {
                    // The kernels walk memory linearly and the result is C-ordered, so x must be a C-contiguous
                    // float32 buffer: x itself, or its C-order float32 copy (astype's default 'K' would keep an
                    // F-ordered x F-ordered).
                    bool direct = x.typecode == NPTypeCode.Single && x.Shape.IsContiguous;
                    temp = direct ? null : x.astype(NPTypeCode.Single, true, 'C');
                    float* xp = direct ? (float*)((byte*)x.Storage.Address + x.Shape.offset * 4) : (float*)temp.Storage.Address;
                    Complex s = Complex64Of(scl);
                    if (addC64)
                    {
                        // Both operations complex64: one fused pass (the product stays in a register).
                        Complex o = Complex64Of(off);
                        DirectILKernelGenerator.PolyComplex64AffineReal(xp, n, (float)s.Real, (float)s.Imaginary, (float)o.Real, (float)o.Imaginary, rp);
                        return r;
                    }
                    DirectILKernelGenerator.PolyComplex64ScaleReal(xp, n, (float)s.Real, (float)s.Imaginary, rp);
                    // complex128 sum: a complex128 / float64 NumPy off plus the exactly widened complex64 product.
                    var sum = np.evaluate(NDExpr.Arr(offP) + NDExpr.Arr(r));
                    r.Dispose();
                    r = null;
                    return sum;
                }

                // Real product (float16 / float32, NumPy's real loop — the fused pass is bit-exact for it), then the
                // complex64 sum of its values.
                NDArray staged = StageMapDomainPoints(x, t1);
                try
                {
                    product = np.evaluate(NDExpr.Arr(sclP) * NDExpr.Arr(staged ?? x));
                }
                finally
                {
                    staged?.Dispose();
                }
                // Same C-contiguous float32 requirement for the product (a float16 product widens exactly).
                bool productDirect = product.typecode == NPTypeCode.Single && product.Shape.IsContiguous;
                temp = productDirect ? null : product.astype(NPTypeCode.Single, true, 'C');
                NDArray values = temp ?? product;
                float* vp = (float*)((byte*)values.Storage.Address + values.Shape.offset * 4);
                Complex oo = Complex64Of(off);
                DirectILKernelGenerator.PolyComplex64AddScalarReal(vp, n, (float)oo.Real, (float)oo.Imaginary, rp);
                return r;
            }
            catch
            {
                r?.Dispose();
                throw;
            }
            finally
            {
                temp?.Dispose();
                product?.Dispose();
            }
        }

        /// <summary>
        ///     A scalar-sized operand as NumPy enters it into a complex64 loop: converted to complex, each component
        ///     rounded to float32 (a Python complex or float is rounded; a complex64 carrier or a float16/float32 NumPy
        ///     scalar is already exact).
        /// </summary>
        /// <param name="v">The operand.</param>
        /// <returns>The complex64 value in a <see cref="Complex"/> carrier.</returns>
        /// <exception cref="OverflowException">A Python int too large for the loop.</exception>
        private static Complex Complex64Of(in PolyNumber v)
        {
            PolyRaw16 raw;
            v.WriteAs(NPTypeCode.Complex, &raw);
            return DirectILKernelGenerator.PolyComplex64Round(*(Complex*)&raw);
        }

        /// <summary>Slots per parameter role in <see cref="t_mapParams"/>: one per NPTypeCode value (Complex = 128 is the largest).</summary>
        private const int ParamSlots = 129;

        /// <summary>
        ///     This thread's 0-d arrays that carry mapdomain's scale (<c>[dtype]</c>) and offset
        ///     (<c>[ParamSlots + dtype]</c>) into the fused pass. np.evaluate copies a 0-d input into its kernel's
        ///     parameter block once, before the pass, and keeps no reference afterwards — so the same two arrays
        ///     serve every call on the thread and save two ~200 ns allocations per call. They are detached from
        ///     every <see cref="NDScope"/> and live as long as the thread (at most one 16-byte buffer per role and
        ///     dtype); [ThreadStatic] makes concurrent callers use distinct arrays.
        /// </summary>
        [ThreadStatic] private static NDArray[] t_mapParams;

        /// <summary>
        ///     The reusable 0-d parameter array of role <paramref name="which"/> and dtype <paramref name="t"/>,
        ///     holding <paramref name="v"/> converted to <paramref name="t"/> (<see cref="PolyNumber.WriteAs"/>: NumPy's
        ///     operand conversion). Valid until this thread's next mapdomain call.
        /// </summary>
        /// <param name="which">0 = scale, 1 = offset (both can share a dtype).</param>
        /// <param name="v">The scalar-sized value.</param>
        /// <param name="t">The parameter dtype.</param>
        /// <returns>The shared 0-d array.</returns>
        /// <exception cref="OverflowException">A Python int that does not fit <paramref name="t"/>.</exception>
        private static NDArray ParamArray(int which, in PolyNumber v, NPTypeCode t)
        {
            var cache = t_mapParams ??= new NDArray[2 * ParamSlots];
            ref NDArray slot = ref cache[which * ParamSlots + (int)t];
            if (slot is null)
            {
                var created = new NDArray(t, Shape.Scalar, false);
                NDScope.Detach(created);   // first use may happen inside a caller's scope, which must not dispose it
                slot = created;
            }
            v.WriteAs(t, (byte*)slot.Storage.Address + slot.Shape.offset * slot.dtypesize);
            return slot;
        }

        // ---------------------------------------------------------------------------------------------
        //  {p}line
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>{p}line(off, scl)</c>: the series of <c>off + scl*x</c> in <paramref name="basis"/> —
        ///     <c>np.array([off, scl])</c> (Laguerre <c>[off + scl, -scl]</c>, physicists' Hermite
        ///     <c>[off, scl / 2]</c>) when <c>scl != 0</c>, else <c>np.array([off])</c>.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="off">The offset (any NumPy / Python number).</param>
        /// <param name="scl">The scale.</param>
        /// <returns>A new array of the dtype np.array discovers for the entries (<c>polyline(1, 2)</c> is int64).</returns>
        /// <exception cref="ValueError">An array scale with several elements or none, or entries of inhomogeneous shape.</exception>
        /// <exception cref="TypeError">NumPy's boolean negative (lagline of NumPy bools).</exception>
        /// <exception cref="OverflowException">Python int overflow (hermline's <c>scl / 2</c>, NumPy's weak int ranges).</exception>
        /// <exception cref="NotSupportedException">An entry NumPy would store in an object array (a Python int beyond
        ///     uint64, null, a str).</exception>
        public static NDArray Line(PolyBasis basis, object off, object scl)
        {
            // Scalar operands (C# numbers: Python / NumPy scalars) allocate nothing but the result, so the scope
            // (~40 ns of a ~250 ns call) is only opened when an operand is, or converts to, an ndarray — whose
            // conversion and ufunc arithmetic leave intermediates for the scope to reclaim.
            if (PolyNumber.IsScalarValue(off) && PolyNumber.IsScalarValue(scl))
                return LineCore(basis, off, scl);
            using var scope = NDScope.Open();
            return scope.Returns(LineCore(basis, off, scl));
        }

        /// <summary>The body of <see cref="Line"/>, run inside a scope whenever an operand can create arrays.</summary>
        /// <param name="basis">The basis.</param>
        /// <param name="off">The offset.</param>
        /// <param name="scl">The scale.</param>
        /// <returns>The new series array.</returns>
        /// <exception cref="ValueError">See <see cref="Line"/>.</exception>
        /// <exception cref="TypeError">See <see cref="Line"/>.</exception>
        /// <exception cref="OverflowException">See <see cref="Line"/>.</exception>
        /// <exception cref="NotSupportedException">See <see cref="Line"/>.</exception>
        private static NDArray LineCore(PolyBasis basis, object off, object scl)
        {
            var o = PolyNumber.FromObject(off);
            var s = PolyNumber.FromObject(scl);
            if (!s.NotZero())
                return PolyNumber.MakeArray(o);
            switch (basis)
            {
                case PolyBasis.Laguerre:
                {
                    var first = PolyNumber.Binary(BinaryOp.Add, o, s);   // Python evaluates the list left to right
                    return PolyNumber.MakeArray(first, PolyNumber.Negate(s));
                }
                case PolyBasis.Hermite:
                    return PolyNumber.MakeArray(o, PolyNumber.Binary(BinaryOp.Divide, s, PolyNumber.FromPython(PyScalar.Int(2))));
                default:
                    return PolyNumber.MakeArray(o, s);
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  Module constants ({p}domain / {p}zero / {p}one / {p}x)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The shared module constant in <paramref name="slot"/>, built by <paramref name="build"/> on first use.
        ///     NumPy's constants are ordinary writeable module-level ndarrays — the SAME object on every access,
        ///     so a write persists — and that is reproduced: one process-wide instance, writeable, detached from
        ///     every <see cref="NDScope"/> (it may be first touched inside a caller's scope, which would otherwise
        ///     dispose it), and holding one extra buffer reference so a caller's <c>Dispose()</c> of the shared
        ///     object cannot free memory every later read still uses.
        /// </summary>
        /// <param name="slot">The static field caching the instance.</param>
        /// <param name="build">Builds a fresh instance.</param>
        /// <returns>The shared instance.</returns>
        public static NDArray Constant(ref NDArray slot, Func<NDArray> build)
        {
            var existing = Volatile.Read(ref slot);
            if (existing is not null)
                return existing;
            var created = build();
            NDScope.Detach(created);
            created.Storage.InternalArray.TryAddRef();   // never released: the buffer lives as long as the process
            var winner = Interlocked.CompareExchange(ref slot, created, null);
            if (winner is null)
                return created;
            // Lost a first-use race: undo the extra reference and drop this copy.
            created.Storage.InternalArray.Release();
            created.Dispose();
            return winner;
        }
    }
}
