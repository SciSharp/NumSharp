using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Fill the array with a scalar value, IN PLACE (NumPy's <c>ndarray.fill</c>). Every element —
        ///     across whatever memory layout this array has (contiguous, F-order, sliced, transposed,
        ///     negative-stride) — is set to <paramref name="value"/> coerced to this array's dtype.
        ///
        ///     <para>
        ///     Coercion follows NumPy's scalar-assignment (NEP50 weak-scalar) rules, probed against
        ///     NumPy 2.4.2. A C# primitive is NumSharp's analog of a Python scalar (weak): assigned to an
        ///     INTEGER dtype it is range-checked — an out-of-bounds value RAISES
        ///     (<c>OverflowException</c> "Python integer 300 out of bounds for int8") rather than wrapping,
        ///     and a float source is TRUNCATED toward zero before the check (<c>3.9</c> stores 3, <c>300.0</c>
        ///     into int8 raises <c>OverflowException</c>; NaN raises <c>ValueError</c> "cannot convert float NaN
        ///     to integer"; ±inf raises <c>OverflowException</c> "cannot convert float infinity to integer"; a
        ///     complex source raises <c>TypeError</c>, exactly as NumPy's setitem runs <c>int()</c>/<c>float()</c>
        ///     on the value). Assigned to a float/complex dtype it casts, saturating to
        ///     ±inf on overflow (<c>float32.fill(1e300)</c> → inf). A 0-d <see cref="NDArray"/> is a STRONG
        ///     scalar and WRAPS on cast (matching an <c>np.int64</c> scalar); a higher-rank array is a
        ///     sequence and raises <c>ValueError("setting an array element with a sequence.")</c>.
        ///     </para>
        ///
        ///     NumPy checks writeability FIRST, then packs the scalar (which may raise) BEFORE touching any
        ///     element — so a read-only destination raises the read-only error even for a bad value, and an
        ///     out-of-range value raises even on an EMPTY array. Both orderings are reproduced.
        /// </summary>
        /// <param name="value">The scalar to fill with. A C# primitive (weak) or a 0-d <see cref="NDArray"/> (strong).</param>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null (NumSharp house convention,
        ///     as in <see cref="np.fill_diagonal"/>; NumPy instead yields NaN for a float array / TypeError otherwise).</exception>
        /// <exception cref="NumSharpException">This array is read-only (broadcast view / read-only memmap);
        ///     NumPy raises <c>ValueError: assignment destination is read-only</c>.</exception>
        /// <exception cref="OverflowException">A weak integer/float value is out of range for an integer dtype,
        ///     or a ±inf value is assigned to an integer dtype (NumPy's <c>OverflowError</c>).</exception>
        /// <exception cref="ValueError"><paramref name="value"/> is a multi-element array (a sequence), or a NaN
        ///     value is assigned to an integer dtype (NumPy's <c>ValueError</c>).</exception>
        /// <exception cref="TypeError">A complex <paramref name="value"/> is assigned to a real (non-bool integer
        ///     or float) dtype — NumPy funnels it through <c>int()</c>/<c>float()</c>, which reject a complex.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.fill.html</remarks>
        public void fill(object value)
        {
            // NumPy (convert.c :: PyArray_FillWithScalar): writeability is checked FIRST, through the same
            // guard every other write path uses.
            NumSharpException.ThrowIfNotWriteable(Shape);

            // Coerce the scalar to this array's dtype with NumPy scalar-coercion. This may raise
            // (OverflowException / ValueError) and — matching NumPy's PyArray_Pack — runs regardless of
            // size, so a bad value is rejected even on an empty array, BEFORE any element is written.
            object scalar = CoerceFillValue(value);

            if (size == 0)
                return;

            // A C- or F-contiguous (non-broadcast) array occupies exactly [offset, offset+size) contiguous
            // slots, so — order-independent for a constant — the whole window can be splatted with the
            // vectorized fill (reaches NumPy's fill speed at cache-resident sizes). A view whose INNERMOST
            // axis is still a contiguous run (a row slice m[::2], a row-strided/offset row view) is a stack
            // of such runs and takes the per-run splat below (NumPy's per-row memset speed). Every remaining
            // layout (transposed / non-unit or negative inner stride / broadcast) is written through its own
            // strides by broadcasting the 0-d scalar with the same NDIter engine np.copyto uses.
            if (Shape.IsContiguous || Shape.IsFContiguous)
                FillContiguousWindow(scalar);
            else if (!TryFillInnerContiguousRuns(scalar))
                NDIter.Copy(this, NDArray.Scalar(scalar));
        }

        /// <summary>
        ///     Splat <paramref name="scalar"/> (already this array's exact dtype) across the contiguous
        ///     logical window [offset, offset+size) via the vectorized <see cref="UnmanagedSpan{T}"/> fill.
        /// </summary>
        private void FillContiguousWindow(object scalar)
        {
            IArraySlice window = Storage.InternalArray;
            long n = size;
            // A contiguous VIEW may sit at [offset, offset+size) inside a larger buffer (np.split children,
            // a row of a matrix); narrow to exactly the logical window before splatting (zero-copy, writes
            // through) — mirrors UnmanagedStorage.GetData<T>()'s own narrowing.
            if (Shape.offset != 0 || window.Count != n)
                window = window.Slice(Shape.offset, n);
            GetWindowFiller(dtype)(window, scalar);
        }

        // Dtype-agnostic dispatch to the generic SIMD fill: one cached delegate per dtype (the reflection-
        // cache pattern), each closing over FillWindowGeneric<T> so the loop lives in the shared vectorized
        // UnmanagedSpan.Fill — not a hand-written per-dtype switch.
        private static readonly ConcurrentDictionary<Type, Action<IArraySlice, object>> _windowFillers
            = new ConcurrentDictionary<Type, Action<IArraySlice, object>>();

        private static Action<IArraySlice, object> GetWindowFiller(Type dtype)
            => _windowFillers.GetOrAdd(dtype, static t => (Action<IArraySlice, object>)typeof(NDArray)
                .GetMethod(nameof(FillWindowGeneric), BindingFlags.NonPublic | BindingFlags.Static)
                .MakeGenericMethod(t)
                .CreateDelegate(typeof(Action<IArraySlice, object>)));

        private static void FillWindowGeneric<T>(IArraySlice window, object value) where T : unmanaged
            => window.AsSpan<T>().Fill((T)value);

        /// <summary>
        ///     Fast path for a NON-contiguous view whose INNERMOST axis is still a unit-stride contiguous run
        ///     (length &gt; 1) — a row slice <c>m[::2]</c>, a row-strided or offset row view — which is a stack
        ///     of contiguous runs. Each run is splatted with the same vectorized <see cref="UnmanagedSpan{T}"/>
        ///     fill the contiguous window uses (reaching NumPy's per-row memset speed), walking the outer axes
        ///     as an odometer so the inner run never touches the element-scatter iterator. Works at ANY rank
        ///     (NumSharp has no ndim cap; the odometer heap-allocates past the stack budget). Returns
        ///     <see langword="false"/> — so the caller falls back to <see cref="NDIter.Copy(NDArray, NDArray)"/>
        ///     — for any layout without a real contiguous inner run (broadcast, non-unit or negative inner
        ///     stride, or a length-≤1 inner axis).
        /// </summary>
        private bool TryFillInnerContiguousRuns(object scalar)
        {
            Shape shape = Shape;
            int ndim = shape.NDim;
            if (ndim < 1 || shape.IsBroadcasted)
                return false;                                  // 0-d handled above; stride-0 dims -> NDIter
            long[] strides = shape.strides;
            long[] dims = shape.dimensions;
            if (strides[ndim - 1] != 1 || dims[ndim - 1] <= 1)
                return false;                                  // innermost must be a unit-stride run of len > 1
            GetInnerRunFiller(dtype)(Storage.InternalArray, shape, scalar);
            return true;
        }

        // Cached per-dtype delegate for the inner-run fill (reflection-cache pattern, like GetWindowFiller):
        // each closes over FillInnerRunsGeneric<T> so the run splat lives in the shared vectorized span fill.
        private static readonly ConcurrentDictionary<Type, Action<IArraySlice, Shape, object>> _innerRunFillers
            = new ConcurrentDictionary<Type, Action<IArraySlice, Shape, object>>();

        private static Action<IArraySlice, Shape, object> GetInnerRunFiller(Type dtype)
            => _innerRunFillers.GetOrAdd(dtype, static t => (Action<IArraySlice, Shape, object>)typeof(NDArray)
                .GetMethod(nameof(FillInnerRunsGeneric), BindingFlags.NonPublic | BindingFlags.Static)
                .MakeGenericMethod(t)
                .CreateDelegate(typeof(Action<IArraySlice, Shape, object>)));

        // Splat `value` across each contiguous inner run of `shape` (innermost axis unit-stride, length > 1),
        // advancing over the outer axes as an odometer. Runs directly on the whole-buffer span via a struct
        // Slice (no per-run allocation); the run splat is the SIMD UnmanagedSpan<T>.Fill.
        private static void FillInnerRunsGeneric<T>(IArraySlice buffer, Shape shape, object value) where T : unmanaged
        {
            var span = buffer.AsSpan<T>();                     // whole backing buffer; indexed from shape.offset
            long[] dims = shape.dimensions;
            long[] strides = shape.strides;
            int outerNd = shape.NDim - 1;                      // axes above the contiguous inner run
            long innerLen = dims[outerNd];
            T v = (T)value;

            // Odometer over the outer axes — supports ANY rank (NumSharp has no ndim cap): stackalloc for the
            // common shallow ranks, heap only for a pathologically deep one (the FiniteScan.cs pattern).
            Span<long> coord = outerNd <= 64 ? stackalloc long[outerNd] : new long[outerNd];
            coord.Clear();
            long outerCount = 1;
            for (int i = 0; i < outerNd; i++) outerCount *= dims[i];

            for (long o = 0; o < outerCount; o++)
            {
                long off = shape.offset;
                for (int i = 0; i < outerNd; i++) off += coord[i] * strides[i];
                span.Slice(off, innerLen).Fill(v);
                for (int i = outerNd - 1; i >= 0; i--)         // advance odometer, innermost outer axis fastest
                {
                    if (++coord[i] < dims[i]) break;
                    coord[i] = 0;
                }
            }
        }

        /// <summary>
        ///     Coerce <paramref name="value"/> to this array's dtype with NumPy scalar-assignment semantics,
        ///     returning it boxed as the dtype's EXACT C# type (required by the strongly-typed fill).
        /// </summary>
        private object CoerceFillValue(object value)
        {
            if (value is null)
                // NumPy reaches its dtype converter (float → NaN, else TypeError); NumSharp treats a null
                // scalar as a caller bug, matching np.fill_diagonal's deliberate divergence.
                throw new ArgumentNullException(nameof(value), "fill() value must not be null.");

            // A 0-d NDArray is a STRONG scalar (wraps on cast, like an np.int64 scalar); anything of higher
            // rank is a sequence, which NumPy's PyArray_Pack rejects.
            if (value is NDArray nd)
            {
                if (nd.ndim != 0)
                    throw new ValueError("setting an array element with a sequence.");
                return nd.astype(dtype).GetAtIndex(0);
            }

            // Fast path: a value already of the exact dtype IS the stored representation — it always fits
            // and needs no coercion (the hot `int32.fill(42)` / `float64.fill(3.5)` case, no allocation).
            if (value.GetType() == dtype)
                return value;

            NPTypeCode tc = typecode;

            if (IsIntegerLikeTypeCode(tc))
            {
                // A bool target truthiness-tests and can never overflow; every other integer dtype
                // range-checks the weak scalar (float sources truncate toward zero first).
                if (tc != NPTypeCode.Boolean)
                    CheckWeakScalarFitsInteger(value, tc);

                var src = new NDArray(value.GetType(), 1);
                src.SetAtIndex(value, 0);
                return src.astype(dtype).GetAtIndex(0); // astype WRAPS — the bounds check above already gated overflow
            }

            if (tc == NPTypeCode.Complex)
                return value is Complex c ? c : new Complex(Convert.ToDouble(value), 0);

            // Target is now a REAL float family (Half/Single/Double/Decimal). A weak complex source is a
            // TypeError — NumPy's setitem runs float() on the object, which refuses a complex regardless of
            // a zero imaginary part (probed 2.4.2). The strong 0-d complex NDArray took the raw-cast path
            // above, which drops the imaginary part instead — matching NumPy's 0-d-array raw cast.
            if (value is Complex)
                throw new TypeError("float() argument must be a string or a real number, not 'complex'");

            if (tc == NPTypeCode.Half)
                return (Half)Convert.ToDouble(value);

            // float32 / float64 / decimal target: Convert saturates a float overflow to ±inf (matching
            // NumPy) for the float targets. A Half source has no IConvertible, so widen it through double.
            object v = value is Half hv ? (double)hv : value;
            return Convert.ChangeType(v, dtype);
        }

        private static bool IsIntegerLikeTypeCode(NPTypeCode tc)
            => tc is NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte
                or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32
                or NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Char;

        // NumPy's weak-scalar bounds check for a C# primitive assigned to a NON-bool integer dtype: an
        // integer source must fit the target's inclusive range; a float source is truncated toward zero and
        // the TRUNCATED value must fit (NaN/±inf raise). The shared NEP50 primitive
        // (NDExprTypeRules.CheckIntLiteralFits) supplies the message "Python integer {n} out of bounds for {dtype}".
        private static void CheckWeakScalarFitsInteger(object value, NPTypeCode target)
        {
            switch (value)
            {
                case bool:
                    return;
                case sbyte or byte or short or ushort or int or uint or long or char:
                    NDExprTypeRules.CheckIntLiteralFits(Convert.ToInt64(value), target);
                    return;
                case ulong u:
                    if (target == NPTypeCode.UInt64) return;                 // any ulong fits uint64
                    if (u <= long.MaxValue) { NDExprTypeRules.CheckIntLiteralFits((long)u, target); return; }
                    throw ScalarOutOfBounds(u.ToString(), target);           // >= 2^63 fits only uint64
                case Half or float or double:
                {
                    double d = value is Half h ? (double)h : Convert.ToDouble(value);
                    // NumPy's float->int setitem funnels through the same conversion Python's int() does, so
                    // a NaN is a ValueError and a ±inf an OverflowError — both with a dtype-INDEPENDENT
                    // message (probed against NumPy 2.4.2, identical across every int width and both signs).
                    if (double.IsNaN(d))
                        throw new ValueError("cannot convert float NaN to integer");
                    if (double.IsInfinity(d))
                        throw new OverflowException("cannot convert float infinity to integer");
                    double t = Math.Truncate(d);
                    if (t >= -9223372036854775808.0 && t < 9223372036854775808.0)
                    {
                        NDExprTypeRules.CheckIntLiteralFits((long)t, target);
                        return;
                    }
                    if (target == NPTypeCode.UInt64 && t >= 0.0 && t <= 18446744073709551615.0)
                        return;                                              // a large positive still fits uint64
                    throw ScalarOutOfBounds(t.ToString("F0"), target);
                }
                case Complex:
                    // A weak complex assigned to a NON-bool integer dtype is a TypeError — NumPy's setitem
                    // runs int() on the object, which refuses a complex (probed 2.4.2). A bool target never
                    // reaches here (it truthiness-tests earlier); the strong 0-d complex NDArray already took
                    // the raw-cast path that drops the imaginary part.
                    throw new TypeError(
                        "int() argument must be a string, a bytes-like object or a real number, not 'complex'");
                default:
                    return; // Decimal source into an integer target: governed by astype below
            }
        }

        private static OverflowException ScalarOutOfBounds(string shown, NPTypeCode target)
            => new OverflowException($"Python integer {shown} out of bounds for {target.AsNumpyDtypeName()}");
    }
}
