using System;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

// =============================================================================
// QuantileIndexes.cs — NumPy 2.4.2's quantile index arithmetic, in the precision of q's dtype
// =============================================================================
//
// WHAT NUMPY DOES (numpy/lib/_function_base_impl.py: _QuantileMethods, _compute_virtual_index, _get_indexes, _get_gamma,
// _discrete_interpolation_to_boundaries, _inverted_cdf, _closest_observation)
//   `virtual_indexes = method["get_virtual_index"](values_count, quantiles)` is ordinary NumPy arithmetic between the
//   Python int `values_count` (or `values_count - 1`), the Python constants of the method, and the q ARRAY. Under NEP 50
//   the Python numbers are weak: each is converted to q's dtype and every operation is rounded to q's dtype. So a
//   float16 q computes `(n - 1) * q` in float16 — `n - 1` itself rounds (2051 is 2052 in float16, 69999 is inf) — and
//   the neighbours, the weight and `1 - t` follow in float16. A float32 q does the same in float32; a Python float q
//   (weak) became a float64 array first, and an integer q computes in int64 (exact here).
//     * The selection methods (lower, higher, nearest, inverted_cdf, closest_observation) cast their index with
//       `.astype(np.intp)`: NaN, ±inf and anything outside int64 become INT64_MIN on x86-64 (cvttsd2si's "integer
//       indefinite"), which the partition then reports as `kth(=INT64_MIN + n) out of bounds (n)`.
//     * The interpolating methods take `_get_indexes`: previous = floor(vi), next = previous + 1 (rounded again in q's
//       dtype), both -1 where `vi >= n - 1` (the Python int converted to q's dtype: inf in float16 from 65520 on), both 0
//       where `vi < 0`, both -1 where vi is NaN — but only for inexact DATA (a float array); integer data keeps the NaN,
//       whose intp cast is INT64_MIN. Then `_get_gamma`: `virtual_indexes - previous_indexes` in float64 (q's dtype
//       against intp), the method's fix_gamma, and the cast back to q's dtype.
//     * `arr.partition(kth)` validates every kth first (partition_prep_kth_array): a negative kth gains n, and one that
//       is still negative or reaches n raises ValueError `kth(=k) out of bounds (n)`, the first in the list's order — q's
//       order for the take path (the inexact take path appends -1), ascending for the interpolating path (its list is
//       `np.unique([0, -1, previous..., next...])`). An array of size 0 skips the check.
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   Every operation is computed in double and rounded to q's dtype right after it (Round<TQ>). For float16 and float32
//   that is exactly NumPy's arithmetic: one IEEE operation of two float16/float32 values computed in double and rounded
//   once more is correctly rounded (double carries 53 bits, at least 2p + 2 for p = 11 and p = 24), and NumPy's float16
//   loops compute in float32 and round, which is correctly rounded too (24 >= 2 * 11 + 2). The type parameter TQ (double,
//   float or Half) is a JIT constant, so each instantiation is straight-line code without a dtype branch.
// =============================================================================

namespace NumSharp.Statistics
{
    /// <summary>
    ///     NumPy's quantile index arithmetic (the virtual index, the selected or neighbouring indexes, the interpolation
    ///     weight and the partition's kth validation) computed in the precision of q's dtype: see the file header.
    /// </summary>
    /// <remarks>
    ///     The type parameter <c>TQ</c> of every generic member is q's precision: <see cref="double"/> (a float64 q, a Python
    ///     float or int q, an integer or bool q array, whose index arithmetic is exact in double), <see cref="float"/> (a
    ///     float32 q array) or <see cref="Half"/> (a float16 q array). Any other type computes in double.
    /// </remarks>
    internal static class QuantileIndexes
    {
        /// <summary>The lower bound of int64 as a double (-2^63, exact): the smallest value NumPy's intp cast keeps.</summary>
        private const double IntpMinAsDouble = -9223372036854775808.0;

        /// <summary>2^63 as a double: the first value above the int64 range.</summary>
        private const double IntpLimitAsDouble = 9223372036854775808.0;

        /// <summary>
        ///     Rounds a double to q's dtype: one NumPy operation in q's dtype is the double result of the same operation on
        ///     the same (q-dtype) operands, rounded by this (see the file header).
        /// </summary>
        /// <typeparam name="TQ">q's precision: <see cref="Half"/>, <see cref="float"/>, or anything else for double (no
        ///     rounding).</typeparam>
        /// <param name="x">The double value.</param>
        /// <returns>The value rounded to nearest-even in q's dtype (float16 overflows to ±inf from 65520 on), as a
        ///     double.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static double Round<TQ>(double x) where TQ : unmanaged
        {
            // JIT-folded per instantiation: only one of these lines survives.
            if (typeof(TQ) == typeof(Half)) return (double)(Half)x;
            if (typeof(TQ) == typeof(float)) return (float)x;
            return x;
        }

        /// <summary>
        ///     <see cref="Round{TQ}"/> for a precision known only at run time: the callers that round one value per quantile
        ///     (percentile's <c>q / 100</c>, the text paths' lerp), where no kernel is specialized.
        /// </summary>
        /// <param name="precision">q's precision: <see cref="NPTypeCode.Half"/>, <see cref="NPTypeCode.Single"/>, or anything
        ///     else for double (no rounding).</param>
        /// <param name="x">The double value.</param>
        /// <returns>The value rounded to nearest-even in that precision, as a double.</returns>
        internal static double Round(NPTypeCode precision, double x)
            => precision switch
            {
                NPTypeCode.Half => (double)(Half)x,
                NPTypeCode.Single => (float)x,
                _ => x,
            };

        /// <summary>
        ///     q's precision for NumPy's index arithmetic: a float16 / float32 q ARRAY computes in its own dtype; a Python
        ///     number (a C# literal, weak: NumPy made it a float64 or int64 array) and every other dtype compute in double.
        /// </summary>
        /// <param name="q">The q array as the caller gave it.</param>
        /// <returns><see cref="NPTypeCode.Half"/>, <see cref="NPTypeCode.Single"/> or <see cref="NPTypeCode.Double"/>.</returns>
        internal static NPTypeCode PrecisionOf(NDArray q)
            => !q.WasPythonLiteral && q.typecode is NPTypeCode.Half or NPTypeCode.Single ? q.typecode : NPTypeCode.Double;

        /// <summary>
        ///     q's dtype for NumPy's index arithmetic of <c>np.quantile</c> / <c>np.nanquantile</c> (which do not divide q): a
        ///     float16 / float32 q ARRAY computes in its own float dtype, an integer q array in its integer dtype (wrapping;
        ///     a bool q in int64, the dtype of <c>(n - 1) * bool_array</c>), and a Python number or any other dtype in
        ///     double. np.percentile divides first, so it reads <see cref="PrecisionOf"/> of the quotient instead.
        /// </summary>
        /// <param name="q">The q array as the caller gave it.</param>
        /// <returns>Half, Single, an integer typecode (SByte to UInt64), or Double.</returns>
        internal static NPTypeCode IndexDtypeOf(NDArray q)
        {
            if (q.WasPythonLiteral)
                return NPTypeCode.Double;
            return q.typecode switch
            {
                NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16
                    or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64
                    => q.typecode,
                NPTypeCode.Boolean => NPTypeCode.Int64,
                _ => NPTypeCode.Double,
            };
        }

        /// <summary>
        ///     NumPy's <c>.astype(np.intp)</c> of a float on x86-64 (cvttsd2si / cvttss2si): truncation toward zero, and the
        ///     "integer indefinite" <see cref="long.MinValue"/> for NaN, ±inf and every value outside the int64 range.
        /// </summary>
        /// <param name="x">The float value (q's dtype, held in a double).</param>
        /// <returns>The intp.</returns>
        /// <remarks>
        ///     Written out instead of a C# cast because .NET 9 made float-to-integer casts saturating (NaN is 0 and +inf is
        ///     <see cref="long.MaxValue"/> there), while .NET 8 on x64 still gives the hardware's value: NumPy's answer must
        ///     not depend on the runtime. An arm64 NumPy saturates instead (fcvtzs); the x86-64 value is the one pinned, the
        ///     same choice as NumSharp's other C-undefined conversions.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long ToIntp(double x)
            => x >= IntpMinAsDouble && x < IntpLimitAsDouble ? (long)x : long.MinValue;

        /// <summary>
        ///     The index a selection method takes for one quantile: NumPy's <c>get_virtual_index</c> of lower, higher,
        ///     nearest, inverted_cdf and closest_observation, as intp. 'linear' with an integer q takes
        ///     <see cref="QuantileMethod.Lower"/>'s index (its <c>(n - 1) * q</c> is already an integer).
        /// </summary>
        /// <typeparam name="TQ">q's precision (see the class remarks).</typeparam>
        /// <param name="method">A selection method.</param>
        /// <param name="n">The line's length (NumPy's <c>values_count</c>; 0 for an empty line).</param>
        /// <param name="q">The quantile (a value of q's dtype) in [0, 1].</param>
        /// <returns>The index: in <c>[0, n - 1]</c> for a float64 q, but possibly <c>n</c> or more (q's rounding) or
        ///     <see cref="long.MinValue"/> (an index that is NaN or ±inf in float16) for a narrower q, and negative for an
        ///     empty line — the partition or the take reports those.</returns>
        /// <exception cref="ArgumentOutOfRangeException">An interpolating method (never passed: those interpolate).</exception>
        internal static long TakeIndex<TQ>(QuantileMethod method, long n, double q) where TQ : unmanaged
        {
            switch (method)
            {
                case QuantileMethod.Lower:
                    // np.floor((n - 1) * quantiles).astype(np.intp): the Python int n - 1 converted to q's dtype, one
                    // multiply in q's dtype, then the floor (exact in any float dtype).
                    return ToIntp(Math.Floor(Round<TQ>(Round<TQ>(n - 1) * q)));
                case QuantileMethod.Higher:
                    return ToIntp(Math.Ceiling(Round<TQ>(Round<TQ>(n - 1) * q)));
                case QuantileMethod.Nearest:
                    // np.around rounds half to even.
                    return ToIntp(Math.Round(Round<TQ>(Round<TQ>(n - 1) * q), MidpointRounding.ToEven));
                case QuantileMethod.InvertedCdf:
                    // _inverted_cdf: _discrete_interpolation_to_boundaries((n * quantiles) - 1, gamma == 0).
                    return DiscreteBoundary<TQ>(Round<TQ>(Round<TQ>(Round<TQ>(n) * q) - 1), closestObservation: false);
                case QuantileMethod.ClosestObservation:
                    // _closest_observation: ... ((n * quantiles) - 1 - 0.5, gamma == 0 & floor(index) % 2 == 1), left to
                    // right: the - 1 rounds before the - 0.5.
                    return DiscreteBoundary<TQ>(Round<TQ>(Round<TQ>(Round<TQ>(Round<TQ>(n) * q) - 1) - 0.5),
                        closestObservation: true);
                default:
                    throw new ArgumentOutOfRangeException(nameof(method), method, "not a selection method");
            }
        }

        /// <summary>
        ///     NumPy's <c>_discrete_interpolation_to_boundaries(index, gamma_condition_fun)</c>: <c>previous = floor(index)</c>,
        ///     <c>next = previous + 1</c> and <c>gamma = index - previous</c> in q's dtype; the previous index where the
        ///     condition holds and the next one otherwise, cast to intp; a negative result clipped to 0
        ///     (<c>res[res &lt; 0] = 0</c>, which also clips the INT64_MIN of a NaN or infinite index).
        /// </summary>
        /// <typeparam name="TQ">q's precision.</typeparam>
        /// <param name="index">The method's index, a value of q's dtype.</param>
        /// <param name="closestObservation">closest_observation's condition (<c>gamma == 0 &amp; floor(index) % 2 == 1</c>)
        ///     instead of inverted_cdf's (<c>gamma == 0</c>).</param>
        /// <returns>The index (0 or more).</returns>
        private static long DiscreteBoundary<TQ>(double index, bool closestObservation) where TQ : unmanaged
        {
            double previous = Math.Floor(index);
            double next = Round<TQ>(previous + 1);
            double gamma = Round<TQ>(index - previous);
            // np.floor(index) % 2: Python's floored modulo (exact for an integral value; NaN for ±inf and NaN, never 1).
            double mod = previous - 2 * Math.Floor(previous / 2);
            bool takePrevious = gamma == 0 && (!closestObservation || mod == 1);
            long result = ToIntp(takePrevious ? previous : next);
            return result < 0 ? 0 : result;
        }

        /// <summary>
        ///     An interpolating method's <c>get_virtual_index(n, quantiles)</c> for one quantile, with NumPy's own expressions
        ///     and operation order, each operation rounded to q's dtype.
        /// </summary>
        /// <typeparam name="TQ">q's precision.</typeparam>
        /// <param name="method">An interpolating method (linear, midpoint, averaged_inverted_cdf, interpolated_inverted_cdf,
        ///     hazen, weibull, median_unbiased, normal_unbiased).</param>
        /// <param name="n">The line's length.</param>
        /// <param name="q">The quantile (a value of q's dtype) in [0, 1].</param>
        /// <returns>The virtual index (fractional, possibly outside <c>[0, n - 1]</c>, and ±inf or NaN in float16 once
        ///     <c>n - 1</c> overflows it).</returns>
        /// <exception cref="ArgumentOutOfRangeException">A selection method (those take an index, see
        ///     <see cref="TakeIndex{TQ}"/>).</exception>
        internal static double VirtualIndex<TQ>(QuantileMethod method, long n, double q) where TQ : unmanaged
        {
            switch (method)
            {
                case QuantileMethod.Linear:
                    // (n - 1) * quantiles
                    return Round<TQ>(Round<TQ>(n - 1) * q);
                case QuantileMethod.Midpoint:
                {
                    // 0.5 * (np.floor((n - 1) * quantiles) + np.ceil((n - 1) * quantiles)): the sum rounds in q's dtype
                    // (2047 + 2048 is 4096 in float16), the halving is exact.
                    double x = Round<TQ>(Round<TQ>(n - 1) * q);
                    return Round<TQ>(0.5 * Round<TQ>(Math.Floor(x) + Math.Ceiling(x)));
                }
                case QuantileMethod.AveragedInvertedCdf:
                    // (n * quantiles) - 1
                    return Round<TQ>(Round<TQ>(Round<TQ>(n) * q) - 1);
                // _compute_virtual_index(n, quantiles, alpha, beta) with each method's Python constants; the coefficient
                // `1 - alpha - beta` is Python arithmetic (exact ints for 0/1, doubles otherwise), converted to q's dtype
                // where it meets q.
                case QuantileMethod.InterpolatedInvertedCdf:
                    return ComputeVirtualIndex<TQ>(n, q, alpha: 0, coefficient: 1 - 0 - 1);
                case QuantileMethod.Hazen:
                    return ComputeVirtualIndex<TQ>(n, q, alpha: 0.5, coefficient: 1 - 0.5 - 0.5);
                case QuantileMethod.Weibull:
                    return ComputeVirtualIndex<TQ>(n, q, alpha: 0, coefficient: 1 - 0 - 0);
                case QuantileMethod.MedianUnbiased:
                    return ComputeVirtualIndex<TQ>(n, q, alpha: 1 / 3.0, coefficient: 1 - 1 / 3.0 - 1 / 3.0);
                case QuantileMethod.NormalUnbiased:
                    return ComputeVirtualIndex<TQ>(n, q, alpha: 3 / 8.0, coefficient: 1 - 3 / 8.0 - 3 / 8.0);
                default:
                    throw new ArgumentOutOfRangeException(nameof(method), method, "not an interpolating method");
            }
        }

        /// <summary>
        ///     NumPy's <c>_compute_virtual_index(n, quantiles, alpha, beta)</c> =
        ///     <c>n * quantiles + (alpha + quantiles * (1 - alpha - beta)) - 1</c>, left to right, each operation in q's dtype
        ///     with the Python numbers converted to it first.
        /// </summary>
        /// <typeparam name="TQ">q's precision.</typeparam>
        /// <param name="n">The line's length.</param>
        /// <param name="q">The quantile.</param>
        /// <param name="alpha">The method's alpha.</param>
        /// <param name="coefficient"><c>1 - alpha - beta</c>, computed as Python computes it.</param>
        /// <returns>The virtual index.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ComputeVirtualIndex<TQ>(long n, double q, double alpha, double coefficient)
            where TQ : unmanaged
            => Round<TQ>(Round<TQ>(Round<TQ>(Round<TQ>(n) * q) + Round<TQ>(Round<TQ>(alpha) + Round<TQ>(q * Round<TQ>(coefficient))))
                         - 1);

        /// <summary>
        ///     An interpolating method's neighbours and weight for one quantile: NumPy's <c>_get_indexes</c> (previous and
        ///     next as intp, before the partition's negative-index adjustment) and <c>_get_gamma</c> (the weight t, with the
        ///     method's fix_gamma, in q's dtype), plus <c>1 - t</c> as <c>_lerp</c> computes it (in q's dtype).
        /// </summary>
        /// <typeparam name="TQ">q's precision.</typeparam>
        /// <param name="method">An interpolating method.</param>
        /// <param name="n">The line's length (at least 0).</param>
        /// <param name="q">The quantile (a value of q's dtype) in [0, 1].</param>
        /// <param name="inexact">Whether the DATA is a float array: <c>_get_indexes</c> then sets a NaN virtual index's
        ///     neighbours to -1; integer data keeps the NaN, which the intp cast makes <see cref="long.MinValue"/>.</param>
        /// <param name="previous">The previous index: in <c>[0, n - 1]</c>, -1 at or past the last index (or for a NaN
        ///     virtual index over inexact data), or <see cref="long.MinValue"/> (a NaN virtual index over integer data).</param>
        /// <param name="next">The next index, on the same terms (<c>previous + 1</c> rounded in q's dtype can be
        ///     <c>n</c>).</param>
        /// <param name="t">The interpolation weight (a value of q's dtype; NaN or ±inf when the virtual index is).</param>
        /// <param name="oneMinusT"><c>1 - t</c> in q's dtype.</param>
        /// <exception cref="ArgumentOutOfRangeException">A selection method.</exception>
        internal static void Neighbours<TQ>(QuantileMethod method, long n, double q, bool inexact,
            out long previous, out long next, out double t, out double oneMinusT) where TQ : unmanaged
            => NeighboursFrom<TQ>(method, n, VirtualIndex<TQ>(method, n, q), inexact, out previous, out next, out t,
                out oneMinusT);

        /// <summary>
        ///     <see cref="Neighbours{TQ}"/> from a virtual index already computed: NumPy's <c>_get_indexes</c> and
        ///     <c>_get_gamma</c> over a virtual index of q's float dtype.
        /// </summary>
        /// <typeparam name="TQ">The virtual index's precision (double for the float64 index of an integer q's methods with
        ///     float constants).</typeparam>
        /// <param name="method">An interpolating method (its fix_gamma).</param>
        /// <param name="n">The line's length.</param>
        /// <param name="virtualIndex">The virtual index, a value of <typeparamref name="TQ"/>.</param>
        /// <param name="inexact">Whether the data is a float array (see <see cref="Neighbours{TQ}"/>).</param>
        /// <param name="previous">The previous index (see <see cref="Neighbours{TQ}"/>).</param>
        /// <param name="next">The next index.</param>
        /// <param name="t">The interpolation weight.</param>
        /// <param name="oneMinusT"><c>1 - t</c>.</param>
        internal static void NeighboursFrom<TQ>(QuantileMethod method, long n, double virtualIndex, bool inexact,
            out long previous, out long next, out double t, out double oneMinusT) where TQ : unmanaged
        {
            // _get_indexes: floor(vi) and floor(vi) + 1 (in q's dtype), then the three overrides in NumPy's order — the
            // below-bounds one after the above-bounds one, so a virtual index in [-1, 0) ends at 0 for an empty line.
            double low = Math.Floor(virtualIndex);
            double high = Round<TQ>(low + 1);
            if (virtualIndex >= Round<TQ>(n - 1))
            {
                low = -1;
                high = -1;
            }
            if (virtualIndex < 0)
            {
                low = 0;
                high = 0;
            }
            if (inexact && double.IsNaN(virtualIndex))
            {
                low = -1;
                high = -1;
            }
            previous = ToIntp(low);
            next = ToIntp(high);

            // _get_gamma: `virtual_indexes - previous_indexes` is float64 (q's dtype against intp), so the raw weight is the
            // exact difference; fix_gamma; then `np.asanyarray(gamma, dtype=virtual_indexes.dtype)` rounds it to q's dtype.
            double gamma = virtualIndex - previous;
            double fixedGamma = method switch
            {
                // `_get_gamma_mask(shape, 1., 0.5, where=gamma == 0)`.
                QuantileMethod.AveragedInvertedCdf => gamma == 0 ? 0.5 : 1.0,
                // `_get_gamma_mask(shape, 0.5, 0., where=index % 1 == 0)`: Python's floored modulo of the virtual index (NaN
                // for ±inf and NaN, so those take 0.5).
                QuantileMethod.Midpoint => virtualIndex - Math.Floor(virtualIndex) == 0 ? 0.0 : 0.5,
                _ => gamma,
            };
            t = Round<TQ>(fixedGamma);
            // _lerp's `1 - t`: the weak int 1 against t's dtype.
            oneMinusT = Round<TQ>(1 - t);
        }

        // ── an integer q array: NumPy's arithmetic in q's integer dtype ───────────────────────────────────────────

        /// <summary>
        ///     Whether TQ is an integer q's dtype (int8 to uint64; a bool q computes in int64, NumPy's default integer, and
        ///     arrives as <see cref="long"/>): the plan then runs <see cref="IntegerPlan{TQ}"/>.
        /// </summary>
        /// <typeparam name="TQ">q's precision type.</typeparam>
        /// <returns>True for an integer type (JIT-folded per instantiation).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsIntegerPrecision<TQ>() where TQ : unmanaged
            => typeof(TQ) == typeof(sbyte) || typeof(TQ) == typeof(byte) || typeof(TQ) == typeof(short)
               || typeof(TQ) == typeof(ushort) || typeof(TQ) == typeof(int) || typeof(TQ) == typeof(uint)
               || typeof(TQ) == typeof(long) || typeof(TQ) == typeof(ulong);

        /// <summary>Whether an integer q's dtype is unsigned.</summary>
        /// <typeparam name="TQ">An integer type.</typeparam>
        /// <returns>True for byte, ushort, uint and ulong.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsUnsigned<TQ>() where TQ : unmanaged
            => typeof(TQ) == typeof(byte) || typeof(TQ) == typeof(ushort) || typeof(TQ) == typeof(uint)
               || typeof(TQ) == typeof(ulong);

        /// <summary>NumPy's name of an integer q's dtype, for its OverflowError texts.</summary>
        /// <typeparam name="TQ">An integer type.</typeparam>
        /// <returns><c>int8</c> to <c>uint64</c>.</returns>
        private static string IntegerName<TQ>() where TQ : unmanaged
            => typeof(TQ) == typeof(sbyte) ? "int8" : typeof(TQ) == typeof(byte) ? "uint8"
               : typeof(TQ) == typeof(short) ? "int16" : typeof(TQ) == typeof(ushort) ? "uint16"
               : typeof(TQ) == typeof(int) ? "int32" : typeof(TQ) == typeof(uint) ? "uint32"
               : typeof(TQ) == typeof(long) ? "int64" : "uint64";

        /// <summary>
        ///     An integer operation's result in q's dtype: the exact value wrapped to its width (C's modular integer
        ///     arithmetic, which NumPy's integer loops are). A uint64 keeps its two's-complement bits in the long (compare it
        ///     with <see cref="AtLeast{TQ}"/>, widen it with <see cref="IntegerToDouble{TQ}"/>).
        /// </summary>
        /// <typeparam name="TQ">An integer type.</typeparam>
        /// <param name="x">The exact result.</param>
        /// <returns>The wrapped value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long Wrap<TQ>(long x) where TQ : unmanaged
        {
            unchecked
            {
                if (typeof(TQ) == typeof(sbyte)) return (sbyte)x;
                if (typeof(TQ) == typeof(byte)) return (byte)x;
                if (typeof(TQ) == typeof(short)) return (short)x;
                if (typeof(TQ) == typeof(ushort)) return (ushort)x;
                if (typeof(TQ) == typeof(int)) return (int)x;
                if (typeof(TQ) == typeof(uint)) return (uint)x;
                return x;   // int64, and uint64's bits
            }
        }

        /// <summary>
        ///     NEP 50's conversion of a weak Python int to q's integer dtype, the first step of an operation with q: a value
        ///     outside the dtype's range raises (it is never wrapped).
        /// </summary>
        /// <typeparam name="TQ">An integer type.</typeparam>
        /// <param name="value">The Python int.</param>
        /// <returns><paramref name="value"/>.</returns>
        /// <exception cref="OverflowException">NumPy's <c>Python integer 200 out of bounds for int8</c>.</exception>
        private static long PyInt<TQ>(long value) where TQ : unmanaged
        {
            bool fits;
            if (typeof(TQ) == typeof(sbyte)) fits = value >= sbyte.MinValue && value <= sbyte.MaxValue;
            else if (typeof(TQ) == typeof(byte)) fits = value >= 0 && value <= byte.MaxValue;
            else if (typeof(TQ) == typeof(short)) fits = value >= short.MinValue && value <= short.MaxValue;
            else if (typeof(TQ) == typeof(ushort)) fits = value >= 0 && value <= ushort.MaxValue;
            else if (typeof(TQ) == typeof(int)) fits = value >= int.MinValue && value <= int.MaxValue;
            else if (typeof(TQ) == typeof(uint)) fits = value >= 0 && value <= uint.MaxValue;
            else if (typeof(TQ) == typeof(ulong)) fits = value >= 0;
            else fits = true;
            if (!fits)
                throw new OverflowException($"Python integer {value} out of bounds for {IntegerName<TQ>()}");
            return value;
        }

        /// <summary>
        ///     A comparison of q's integer array with a Python int (<c>vi &gt;= n - 1</c>): NumPy compares the mathematical
        ///     values, even when the Python int is outside the dtype (a uint8 array is <c>&gt;= -1</c> everywhere).
        /// </summary>
        /// <typeparam name="TQ">An integer type.</typeparam>
        /// <param name="value">The array value (wrapped; a uint64 as its bits).</param>
        /// <param name="pyInt">The Python int.</param>
        /// <returns>Whether <paramref name="value"/> &gt;= <paramref name="pyInt"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool AtLeast<TQ>(long value, long pyInt) where TQ : unmanaged
        {
            if (typeof(TQ) == typeof(ulong))
                return pyInt < 0 || (ulong)value >= (ulong)pyInt;
            return value >= pyInt;
        }

        /// <summary>An integer array value as a double (a uint64 from its unsigned value, rounded once as NumPy's cast to
        ///     float64 rounds it; .NET 8's own conversion rounds twice from 2^63 on).</summary>
        /// <typeparam name="TQ">An integer type.</typeparam>
        /// <param name="value">The value (wrapped).</param>
        /// <returns>The nearest double.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double IntegerToDouble<TQ>(long value) where TQ : unmanaged
            => typeof(TQ) == typeof(ulong) ? Converts.ToDouble((ulong)value) : value;

        /// <summary>
        ///     NumPy's index plan for an integer q ARRAY (validated, so every value is 0 or 1): every operation in q's integer
        ///     dtype, as NumPy's integer loops compute it — the Python ints <c>n</c> / <c>n - 1</c> converted first
        ///     (<c>Python integer 200 out of bounds for int8</c>, even for an empty q), the products and sums wrapped (an
        ///     int8 midpoint's <c>(n - 1) + (n - 1)</c> wraps negative, so q = 1 reads the first element), a float constant
        ///     turning the rest float64 (hazen, median_unbiased, normal_unbiased, closest_observation's <c>- 0.5</c>,
        ///     midpoint's <c>0.5 *</c>), and the integer methods' <c>_get_indexes</c> writing -1 into an index array of q's
        ///     dtype (an unsigned one raises).
        /// </summary>
        /// <typeparam name="TQ">q's integer type (a bool q arrives as <see cref="long"/>).</typeparam>
        /// <param name="method">The method ('linear' over an integer q is taken as 'lower').</param>
        /// <param name="n">The line's length.</param>
        /// <param name="q">The quantiles (0 or 1).</param>
        /// <param name="nQs">How many.</param>
        /// <param name="inexact">Whether the data is a float array (only the float64 virtual indexes read it).</param>
        /// <param name="raw">Receives the raw indexes (2 per quantile).</param>
        /// <param name="weight">Receives the weights (2 per quantile): integers of q's dtype for the integer methods.</param>
        /// <exception cref="OverflowException"><c>n</c> (or <c>n - 1</c>) outside q's dtype; an unsigned q under weibull,
        ///     averaged_inverted_cdf or interpolated_inverted_cdf (NumPy's <c>previous_indexes[above] = -1</c>: <c>Python
        ///     integer -1 out of bounds for uint8</c>).</exception>
        internal static unsafe void IntegerPlan<TQ>(QuantileMethod method, long n, double* q, int nQs, bool inexact,
            long* raw, double* weight) where TQ : unmanaged
        {
            // The method's first operation with q converts its Python int: `(n - 1) * quantiles` or `n * quantiles`. NumPy
            // converts it whatever q holds, so an empty q raises too.
            bool minusOne = method is QuantileMethod.Lower or QuantileMethod.Higher or QuantileMethod.Nearest
                or QuantileMethod.Linear or QuantileMethod.Midpoint;
            long c = PyInt<TQ>(minusOne ? n - 1 : n);
            for (int j = 0; j < nQs; j++)
            {
                long qv = (long)q[j];
                long a = 2L * j, b = 2L * j + 1;
                switch (method)
                {
                    case QuantileMethod.Lower:
                    case QuantileMethod.Higher:
                    case QuantileMethod.Nearest:
                    case QuantileMethod.Linear:
                    {
                        // np.floor / np.ceil / np.around of an integer array are the array; .astype(np.intp) keeps the value
                        // (a uint64's bits).
                        long index = Wrap<TQ>(c * qv);
                        raw[a] = index;
                        raw[b] = index;
                        weight[a] = 0;
                        weight[b] = 1;
                        break;
                    }
                    case QuantileMethod.InvertedCdf:
                    {
                        // _discrete_interpolation_to_boundaries((n * quantiles) - 1, gamma == 0): previous = floor(index) is
                        // the index, gamma = index - previous is 0, so the previous index — a uint8's 0 - 1 = 255 kept,
                        // a signed -1 (or a uint64's bits) clipped to 0.
                        long index = Wrap<TQ>(Wrap<TQ>(c * qv) - 1);
                        raw[a] = raw[b] = index < 0 ? 0 : index;
                        weight[a] = 0;
                        weight[b] = 1;
                        break;
                    }
                    case QuantileMethod.ClosestObservation:
                    {
                        // `(n * quantiles) - 1 - 0.5`: the - 1 wraps in q's dtype, the - 0.5 (a Python float) is float64.
                        long index = DiscreteBoundary<double>(IntegerToDouble<TQ>(Wrap<TQ>(Wrap<TQ>(c * qv) - 1)) - 0.5,
                            closestObservation: true);
                        raw[a] = raw[b] = index;
                        weight[a] = 0;
                        weight[b] = 1;
                        break;
                    }
                    case QuantileMethod.AveragedInvertedCdf:
                        IntegerNeighbours<TQ>(method, n, Wrap<TQ>(Wrap<TQ>(c * qv) - 1), out raw[a], out raw[b], out weight[a],
                            out weight[b]);
                        break;
                    case QuantileMethod.InterpolatedInvertedCdf:
                        // _compute_virtual_index(n, q, 0, 1) with Python ints: n * q + (0 + q * 0) - 1.
                        IntegerNeighbours<TQ>(method, n, Wrap<TQ>(Wrap<TQ>(Wrap<TQ>(c * qv) + Wrap<TQ>(0 + Wrap<TQ>(qv * 0))) - 1),
                            out raw[a], out raw[b], out weight[a], out weight[b]);
                        break;
                    case QuantileMethod.Weibull:
                        // _compute_virtual_index(n, q, 0, 0): n * q + (0 + q * 1) - 1 (n + 1 can wrap; - 1 wraps back).
                        IntegerNeighbours<TQ>(method, n, Wrap<TQ>(Wrap<TQ>(Wrap<TQ>(c * qv) + Wrap<TQ>(0 + Wrap<TQ>(qv * 1))) - 1),
                            out raw[a], out raw[b], out weight[a], out weight[b]);
                        break;
                    case QuantileMethod.Hazen:
                    case QuantileMethod.MedianUnbiased:
                    case QuantileMethod.NormalUnbiased:
                    {
                        // `n * quantiles` stays an integer array; `alpha + quantiles * (1 - alpha - beta)` meets Python floats
                        // and is float64, and so is their sum.
                        double alpha = method == QuantileMethod.Hazen ? 0.5 : method == QuantileMethod.MedianUnbiased ? 1 / 3.0 : 3 / 8.0;
                        double coefficient = 1 - alpha - alpha;
                        double vi = IntegerToDouble<TQ>(Wrap<TQ>(c * qv)) + (alpha + qv * coefficient) - 1;
                        NeighboursFrom<double>(method, n, vi, inexact, out raw[a], out raw[b], out weight[a], out weight[b]);
                        break;
                    }
                    case QuantileMethod.Midpoint:
                    {
                        // 0.5 * (np.floor(x) + np.ceil(x)) with x = (n - 1) * quantiles an integer array: the sum wraps in
                        // q's dtype, the halving is float64.
                        long x = Wrap<TQ>(c * qv);
                        double vi = 0.5 * IntegerToDouble<TQ>(Wrap<TQ>(x + x));
                        NeighboursFrom<double>(method, n, vi, inexact, out raw[a], out raw[b], out weight[a], out weight[b]);
                        break;
                    }
                    default:
                        throw new ArgumentOutOfRangeException(nameof(method), method, "not a quantile method");
                }
            }
        }

        /// <summary>
        ///     <c>_get_indexes</c> and <c>_get_gamma</c> over an integer virtual index (weibull, averaged_inverted_cdf,
        ///     interpolated_inverted_cdf with an integer q): the index arrays have q's dtype, the raw weight
        ///     <c>vi - previous</c> is int64, and the weight cast back to q's dtype is an integer (averaged's 0.5 truncates to
        ///     0, 1.0 is 1).
        /// </summary>
        /// <typeparam name="TQ">q's integer type.</typeparam>
        /// <param name="method">The method (its fix_gamma).</param>
        /// <param name="n">The line's length.</param>
        /// <param name="virtualIndex">The virtual index (wrapped in q's dtype).</param>
        /// <param name="previous">The previous index: 0 before the first index, -1 at or past the last one.</param>
        /// <param name="next">The next index.</param>
        /// <param name="t">The weight, an integer of q's dtype.</param>
        /// <param name="oneMinusT"><c>1 - t</c> in q's dtype.</param>
        /// <exception cref="OverflowException">An unsigned q whose virtual index reaches the last index (every one does,
        ///     q being 0 or 1): NumPy's <c>previous_indexes[indexes_above_bounds] = -1</c>, <c>Python integer -1 out of bounds
        ///     for uint8</c>.</exception>
        private static void IntegerNeighbours<TQ>(QuantileMethod method, long n, long virtualIndex,
            out long previous, out long next, out double t, out double oneMinusT) where TQ : unmanaged
        {
            // floor() of an integer array is the array; `add(previous_indexes, 1)` wraps.
            long low = virtualIndex;
            long high = Wrap<TQ>(virtualIndex + 1);
            if (AtLeast<TQ>(virtualIndex, n - 1))
            {
                // A setitem of the Python int -1 into q's dtype.
                if (IsUnsigned<TQ>())
                    throw new OverflowException($"Python integer -1 out of bounds for {IntegerName<TQ>()}");
                low = -1;
                high = -1;
            }
            if (!IsUnsigned<TQ>() && virtualIndex < 0)
            {
                low = 0;
                high = 0;
            }
            // (np.isnan of an integer array is all False.) .astype(np.intp) keeps the values.
            previous = low;
            next = high;
            // `virtual_indexes - previous_indexes`: q's dtype against intp is int64 (the unsigned dtypes raised above).
            long gamma = virtualIndex - previous;
            long tq = method == QuantileMethod.AveragedInvertedCdf
                ? (gamma == 0 ? 0 : 1)            // _get_gamma_mask's 0.5 / 1.0, cast to an integer dtype
                : Wrap<TQ>(gamma);                 // the identity fix_gamma, cast to q's dtype
            t = tq;
            // _lerp's `1 - t` in q's dtype.
            oneMinusT = Wrap<TQ>(1 - tq);
        }

        /// <summary>
        ///     NumPy's kth validation (<c>partition_prep_kth_array</c>) of the take path's partition:
        ///     <c>arr.partition(virtual_indexes.ravel())</c>, with <c>-1</c> appended for inexact data
        ///     (<c>concatenate((virtual_indexes.ravel(), [-1]))</c>) — checked in that order.
        /// </summary>
        /// <param name="indexes">The selected indexes in q's C order (<see cref="TakeIndex{TQ}"/>).</param>
        /// <param name="count">How many indexes.</param>
        /// <param name="stride">The distance between two indexes in <paramref name="indexes"/> (2 when they are the
        ///     previous half of a previous/next pair).</param>
        /// <param name="n">The line's length (the partitioned axis); at least 1.</param>
        /// <exception cref="ValueError">An index that is still negative after adding <paramref name="n"/>, or at least
        ///     <paramref name="n"/>: NumPy's <c>kth(=k) out of bounds (n)</c>, k after the adjustment. The appended -1 is
        ///     always valid for a line that is not empty.</exception>
        internal static unsafe void CheckTakeKth(long* indexes, int count, int stride, long n)
        {
            for (int i = 0; i < count; i++)
                CheckKth(indexes[(long)i * stride], n);
        }

        /// <summary>
        ///     NumPy's kth validation of the interpolating path's partition:
        ///     <c>arr.partition(np.unique(np.concatenate(([0, -1], previous_indexes.ravel(), next_indexes.ravel()))))</c> —
        ///     the distinct values in ascending order, the first invalid one reported.
        /// </summary>
        /// <param name="neighbours">The previous and next indexes of every quantile (<see cref="Neighbours{TQ}"/>), as
        ///     pairs.</param>
        /// <param name="count">How many values (twice the number of quantiles).</param>
        /// <param name="n">The line's length; at least 1 (so the 0 and -1 NumPy adds are valid).</param>
        /// <exception cref="ValueError">NumPy's <c>kth(=k) out of bounds (n)</c> for the smallest value that is still
        ///     negative after adding <paramref name="n"/> (every negative value sorts first), else for the smallest value
        ///     of at least <paramref name="n"/>.</exception>
        internal static unsafe void CheckInterpolationKth(long* neighbours, int count, long n)
        {
            // Ascending order without the sort: the negative values come first, and among them only the smallest can be
            // invalid (every value from -n on is valid); after them the first invalid value is the smallest one >= n.
            long smallest = -1;
            long firstPastEnd = long.MaxValue;
            for (int i = 0; i < count; i++)
            {
                long k = neighbours[i];
                if (k < smallest)
                    smallest = k;
                if (k >= n && k < firstPastEnd)
                    firstPastEnd = k;
            }
            CheckKth(smallest, n);
            if (firstPastEnd != long.MaxValue)
                CheckKth(firstPastEnd, n);
        }

        /// <summary>One kth of <c>partition_prep_kth_array</c>: a negative kth gains n; the result must lie in [0, n).</summary>
        /// <param name="kth">The kth as given.</param>
        /// <param name="n">The partitioned axis's length.</param>
        /// <exception cref="ValueError">NumPy's <c>kth(=k) out of bounds (n)</c>.</exception>
        private static void CheckKth(long kth, long n)
        {
            if (kth < 0)
                kth += n;
            if (kth < 0 || kth >= n)
                throw new ValueError($"kth(={kth}) out of bounds ({n})");
        }
    }
}
