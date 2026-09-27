using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
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

        /// <summary>A str array of <paramref name="size"/> elements (1-D): passes the size/ndim checks, fails the common type.</summary>
        /// <param name="size">Element count.</param>
        /// <returns>The view.</returns>
        public static PolySeriesView Str(long size) => new PolySeriesView(null, null, size, 0, NPTypeCode.Empty, 1, size, true);
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
        public static NDArray CopyAs(in PolySeriesView v, long n, NPTypeCode t)
        {
            var r = new NDArray(t, new Shape(n), false);
            DirectILKernelGenerator.GetPolyCastKernel(v.Dtype, t)(v.Ptr, n, v.Stride, (byte*)r.Storage.Address);
            return r;
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
            long k = DirectILKernelGenerator.GetPolyCombineKernel(op, a.Dtype, b.Dtype, t)(
                a.Ptr, a.Stride, na, b.Ptr, b.Stride, nb, (byte*)r.Storage.Address);
            // trimseq(ret): the array itself (+na) or its slice ret[:k] (-k) — a view, as NumPy returns it.
            return k >= 0 ? r : Prefix(r, -k);
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
        ///     on the reduction schedule, i.e. on the layout: a strided float x is therefore copied first, while a
        ///     contiguous one is reduced in place (same schedule, same bits) and an integer one is reduced in its own
        ///     dtype and the two results converted (int → float64 is monotone, so min/max commute with it).
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

            NDArray src = v.Dtype == t && !contiguous ? CopyAs(v, v.Len, t) : Row(v);
            using (var mn = np.amin(src))
            using (var mx = np.amax(src))
            {
                PolyConstPool.ConvertBuffer((byte*)mn.Storage.Address + mn.Shape.offset * mn.dtypesize, mn.typecode, dst, t, 1);
                PolyConstPool.ConvertBuffer((byte*)mx.Storage.Address + mx.Shape.offset * mx.dtypesize, mx.typecode, dst + size, t, 1);
            }
            return r;
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
        ///     <c>off + scl * x</c> as NumPy evaluates it. An ndarray x runs ONE fused pass
        ///     (<see cref="np.evaluate(NDExpr, NDArray)"/>) whose dtypes are NumPy's two-ufunc sequence's: a Python
        ///     off/scl is pre-converted to the dtype NEP 50 gives it against x, so the fused kernel's strong 0-d
        ///     parameters promote exactly as the weak values do. Any other x (or an array-valued off/scl from a 2-D
        ///     domain) runs the operator dispatch of <see cref="PolyNumber.Binary"/>.
        /// </summary>
        /// <param name="x">The points.</param>
        /// <param name="off">The offset (from mapparms).</param>
        /// <param name="scl">The scale (from mapparms).</param>
        /// <returns>The mapped points.</returns>
        /// <exception cref="IncorrectShapeException">Array operands that do not broadcast.</exception>
        /// <exception cref="OverflowException">A Python int that does not fit its loop dtype.</exception>
        public static PolyNumber MapDomainWith(in PolyNumber x, in PolyNumber off, in PolyNumber scl)
        {
            if (x.Kind != PolyNumberKind.Array || off.IsNdArray || scl.IsNdArray)
                return PolyNumber.Binary(BinaryOp.Add, off, PolyNumber.Binary(BinaryOp.Multiply, scl, x));

            NDArray xa = x.Array;
            NPTypeCode t1 = scl.IsPython ? PolyNumber.WeakPromote(xa.typecode, scl.Py) : PolyTyping.Promote(scl.Dtype, xa.typecode);
            NPTypeCode t2 = off.IsPython ? PolyNumber.WeakPromote(t1, off.Py) : PolyTyping.Promote(off.Dtype, t1);
            NDArray sclP = ParamArray(0, scl, scl.IsPython ? t1 : scl.Dtype);
            NDArray offP = ParamArray(1, off, off.IsPython ? t2 : off.Dtype);
            return PolyNumber.FromArray(np.evaluate(NDExpr.Arr(offP) + NDExpr.Arr(sclP) * NDExpr.Arr(xa)));
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
