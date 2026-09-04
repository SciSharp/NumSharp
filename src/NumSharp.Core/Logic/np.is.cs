using System;
using System.Collections;
using System.Numerics;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Returns a boolean array where two arrays are element-wise equal within a
        /// tolerance.
        /// The tolerance values are positive, typically very small numbers.The    
        /// relative difference (`rtol` * abs(`b`)) and the absolute difference
        /// `atol` are added together to compare against the absolute difference
        /// between `a` and `b`.
        /// Warning: The default `atol` is not appropriate for comparing numbers
        /// that are much smaller than one(see Notes).
        /// 
        /// See also <seealso cref="allclose"/>
        ///
        ///Notes:
        /// For finite values, isclose uses the following equation to test whether
        /// two floating point values are equivalent.
        /// <code>absolute(`a` - `b`) less than or equal to (`atol` + `rtol` * absolute(`b`))</code>
        /// Unlike the built-in `math.isclose`, the above equation is not symmetric
        /// in `a` and `b` -- it assumes `b` is the reference value -- so that
        /// `isclose(a, b)` might be different from `isclose(b, a)`. Furthermore,
        /// the default value of atol is not zero, and is used to determine what
        /// small values should be considered close to zero.The default value is
        /// appropriate for expected values of order unity: if the expected values
        /// are significantly smaller than one, it can result in false positives.
        /// `atol` should be carefully selected for the use case at hand. A zero value
        /// for `atol` will result in `False` if either `a` or `b` is zero.
        /// </summary>
        /// <param name="a">Input array to compare with b</param>
        /// <param name="b">Input array to compare with a.</param>
        /// <param name="rtol">The relative tolerance parameter(see Notes)</param>
        /// <param name="atol">The absolute tolerance parameter(see Notes)</param>
        /// <param name="equal_nan">Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        ///considered equal to NaN's in `b` in the output array.</param>
        ///<returns>
        ///  Returns a boolean array of where `a` and `b` are equal within the
        /// given tolerance.If both `a` and `b` are scalars, returns a single
        /// boolean value.
        ///</returns>
        public static NDArray<bool> isclose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8,
            bool equal_nan = false)
            => a.TensorEngine.IsClose(a, b, rtol, atol, equal_nan);

        /// <summary>
        /// Test element-wise for finiteness (not infinity and not Not a Number).
        /// Mirrors NumPy's ufunc signature: <c>isfinite(x, /, out=None, *, where=True, dtype=None)</c>.
        /// A plain call returns a bool-dtype array (the instance is an <see cref="NDArray{T}"/> of bool).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <param name="dtype">Validate-only (NumPy parity): the predicate has bool loops only — any non-bool request raises the no-loop TypeError.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html</remarks>
        public static NDArray isfinite(NDArray a, NDArray @out = null, NDArray where = null, DType dtype = null)
            => a.TensorEngine.IsFinite(a, dtype, @out, where);

        /// <summary>
        /// Test element-wise for Not a Number.
        /// Mirrors NumPy's ufunc signature: <c>isnan(x, /, out=None, *, where=True, dtype=None)</c>.
        /// A plain call returns a bool-dtype array (the instance is an <see cref="NDArray{T}"/> of bool).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <param name="dtype">Validate-only (NumPy parity): the predicate has bool loops only — any non-bool request raises the no-loop TypeError.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.isnan.html</remarks>
        public static NDArray isnan(NDArray a, NDArray @out = null, NDArray where = null, DType dtype = null)
            => a.TensorEngine.IsNan(a, dtype, @out, where);

        /// <summary>
        /// Test element-wise for positive or negative infinity.
        /// Mirrors NumPy's ufunc signature: <c>isinf(x, /, out=None, *, where=True, dtype=None)</c>.
        /// A plain call returns a bool-dtype array (the instance is an <see cref="NDArray{T}"/> of bool).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="@out">A location into which the result is stored; any numeric dtype (bool casts same_kind to all, True→1); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <param name="dtype">Validate-only (NumPy parity): the predicate has bool loops only — any non-bool request raises the no-loop TypeError.</param>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.isinf.html
        /// - Float/Double: True if value is +Inf or -Inf
        /// - Integer types: Always False (integers cannot be Inf)
        /// - NaN: Returns False (NaN is not infinity)
        /// </remarks>
        public static NDArray isinf(NDArray a, NDArray @out = null, NDArray where = null, DType dtype = null)
            => a.TensorEngine.IsInf(a, dtype, @out, where);

        /// <summary>
        ///     Returns true incase of a number, bool or string. If null, returns false.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html</remarks>
        public static bool isscalar(object obj)
        {
            switch (obj)
            {
                case null:
                    return false;
                case NDArray nd:
                    return nd.ndim == 0 && nd.size == 1;
                case Type _:
                    break;
                case Complex _:
                case string _:
                case bool _:
                    return true;
            }

            var type = obj as Type ?? obj.GetType();
            if (type.IsArray)
            {
                return false;
            }

            //type.IsPrimitive checks for: Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, IntPtr, UIntPtr, Char, Double, and Single.
            return type.IsPrimitive || obj is decimal;
        }

        /// <summary>
        ///     Check whether or not an object can be iterated over. Returns <c>true</c> if
        ///     <paramref name="y"/> has an iterator method or is a sequence, and <c>false</c> otherwise.
        /// </summary>
        /// <param name="y">Input object.</param>
        /// <returns><c>true</c> if the object is iterable, <c>false</c> otherwise.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.iterable.html
        ///     <para>
        ///     Port of NumPy's <c>numpy.iterable</c> (<c>numpy/lib/_function_base_impl.py</c>), whose whole
        ///     body is <c>try: iter(y); return True; except TypeError: return False</c>. It is a pure
        ///     predicate — it does NOT iterate the data, it only tests whether iteration is possible — so it
        ///     needs no kernel/NDIter/loop of any kind (O(1) rank/type check).
        ///     </para>
        ///     <para>
        ///     The one surprise NumPy documents is 0-dimensional arrays: although a 0-d <see cref="NDArray"/>
        ///     is a collection type, <c>iter()</c> on it raises <c>TypeError("iteration over a 0-d array")</c>
        ///     (see <see cref="NDArray.GetEnumerator"/>), so <c>np.iterable(np.array(1.0))</c> is <c>false</c>
        ///     while any array of rank ≥ 1 (empty included) is <c>true</c>.
        ///     </para>
        ///     C# type mapping (each matches NumPy's <c>iter()</c> outcome, probed against NumPy 2.4.2):
        ///     <list type="bullet">
        ///       <item><description><c>null</c> → false (NumPy's <c>iter(None)</c> raises TypeError).</description></item>
        ///       <item><description><see cref="NDArray"/> → <c>ndim != 0</c> (0-d is the only non-iterable array).</description></item>
        ///       <item><description><see cref="string"/> → true (Python strings are iterable).</description></item>
        ///       <item><description>Any <see cref="IEnumerable"/> — C# arrays, lists, dictionaries, sets … → true.</description></item>
        ///       <item><description>Everything else — the scalar value types int/double/bool/Complex/Half/decimal/char … → false.</description></item>
        ///     </list>
        ///     <para>
        ///     Deliberate C# divergences (probed against NumPy 2.4.2): NumSharp maps Python's
        ///     <c>iter()</c>-ability onto C#'s <see cref="IEnumerable"/> — i.e. "is this <c>foreach</c>-able?".
        ///     Four inputs are iterable in Python but their C# analogs cannot be <c>foreach</c>'d, so they
        ///     return <c>false</c> here while NumPy returns <c>true</c>: a bare <see cref="IEnumerator"/> /
        ///     <c>IEnumerator&lt;T&gt;</c> cursor (a Python iterator is self-iterable, but a C# enumerator
        ///     has no <c>GetEnumerator</c>), and <see cref="ValueTuple"/> / <see cref="Tuple"/> / <c>ITuple</c>
        ///     (they implement no <see cref="IEnumerable"/>). Real ported code passes the collection itself —
        ///     an array, <c>List</c>, or <see cref="NDArray"/>, all foreach-able — which matches NumPy exactly.
        ///     </para>
        /// </remarks>
        public static bool iterable(object y)
        {
            switch (y)
            {
                case null:
                    return false;
                case NDArray nd:
                    // A 0-d array is the only non-iterable array: iter() on it raises TypeError, exactly
                    // as NDArray.GetEnumerator() does; every rank>=1 array (incl. empty) iterates. Checked
                    // ahead of the IEnumerable case because NDArray itself implements IEnumerable.
                    return nd.ndim != 0;
                case string _:
                    // Python str is iterable (np.iterable("abc") == True). string is also IEnumerable, so
                    // this branch documents the deliberate parity choice rather than changing the outcome.
                    return true;
                case IEnumerable _:
                    return true;
                default:
                    // Scalar value types (bool/byte/sbyte/short/ushort/int/uint/long/ulong/char/float/
                    // double/Half/decimal/Complex) are not iterable — matching iter() on a NumPy scalar.
                    return false;
            }
        }
    }
}
