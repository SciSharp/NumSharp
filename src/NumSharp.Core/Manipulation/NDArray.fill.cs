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
        ///     into int8 raises, NaN/±inf raise). Assigned to a float/complex dtype it casts, saturating to
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
        /// <exception cref="OverflowException">A weak integer/float value is out of range for an integer dtype.</exception>
        /// <exception cref="ValueError"><paramref name="value"/> is a multi-element array (a sequence).</exception>
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
            // vectorized fill (reaches NumPy's fill speed at cache-resident sizes). Every other layout
            // (strided / transposed / negative-stride) is written through its own strides by broadcasting
            // the 0-d scalar across the destination with the same NDIter engine np.copyto uses.
            if (Shape.IsContiguous || Shape.IsFContiguous)
                FillContiguousWindow(scalar);
            else
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
                    if (double.IsNaN(d) || double.IsInfinity(d))
                        throw new OverflowException(
                            $"cannot convert non-finite float {d} to integer dtype {target.AsNumpyDtypeName()}");
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
                default:
                    return; // Decimal/Complex source into an integer target: governed by astype below
            }
        }

        private static OverflowException ScalarOutOfBounds(string shown, NPTypeCode target)
            => new OverflowException($"Python integer {shown} out of bounds for {target.AsNumpyDtypeName()}");
    }
}
