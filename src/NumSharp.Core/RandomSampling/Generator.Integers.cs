using System;
using System.Numerics;
using NumSharp.Generic;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     Return random integers from <paramref name="low"/> (inclusive) to <paramref name="high"/>
        ///     (exclusive, or inclusive when <paramref name="endpoint"/> is true).
        /// </summary>
        /// <param name="low">Lowest integer drawn (or, when <paramref name="high"/> is null, one above the highest with low = 0).</param>
        /// <param name="high">If provided, one above the largest integer drawn (or the largest when <paramref name="endpoint"/>).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value; <c>()</c> returns a 0-d array.</param>
        /// <param name="dtype">Desired native integer or bool dtype. Default is int64.</param>
        /// <param name="endpoint">If true, sample from the closed interval <c>[low, high]</c>.</param>
        /// <returns>The draws, of <paramref name="dtype"/>.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is a native non-integer dtype (<c>Unsupported dtype dtype('float64') for integers</c>).</exception>
        /// <exception cref="ValueError">
        ///     <paramref name="dtype"/> has a non-native byte order; or the bounds fall outside the dtype
        ///     (<c>low/high is out of bounds for &lt;dtype&gt;</c>); or the interval is empty (<c>low &gt;= high</c>,
        ///     <c>high &lt;= 0</c> and their closed-interval forms).
        /// </exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html
        ///     <br/>
        ///     Uses Lemire's method (NumPy's Generator default, <c>use_masked=False</c>) — NOT the
        ///     legacy masked rejection of <c>RandomState.randint</c> — so the stream is byte-identical
        ///     to <c>default_rng(seed).integers(...)</c>. A zero-size request returns an empty array before
        ///     the bounds are checked and draws nothing, exactly as NumPy's <c>_rand_*</c> does. Validation and
        ///     the per-width fills are shared with the legacy <c>randint</c> (<see cref="BoundedIntegers"/>).
        /// </remarks>
        public NDArray integers(long low, long? high = null, Shape size = default, DType dtype = null, bool endpoint = false)
            => BoundedIntegers.Draw(_bitGenerator, low, high.HasValue ? (Int128)high.Value : (Int128?)null, size,
                                    dtype ?? DType.Int64, endpoint, useMasked: false, "integers", legacyByteOrder: false);

        /// <summary>
        ///     Unsigned overload of <see cref="integers(long, long?, Shape, DType, bool)"/> — the C# spelling of a
        ///     bound above <see cref="long.MaxValue"/>, which NumPy expresses with arbitrary-precision Python ints:
        ///     the upper half of the <c>uint64</c> range (full range: <c>integers(0UL, ulong.MaxValue,
        ///     dtype: np.uint64, endpoint: true)</c>) and the exclusive int64 high <c>2**63</c>
        ///     (<c>integers(0UL, 9223372036854775808UL, dtype: np.int64)</c>).
        /// </summary>
        /// <param name="low">Lowest integer drawn (or, when <paramref name="high"/> is null, one above the highest with low = 0).</param>
        /// <param name="high">If provided, one above the largest integer drawn (or the largest when <paramref name="endpoint"/>).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value; <c>()</c> returns a 0-d array.</param>
        /// <param name="dtype">Desired native integer or bool dtype. Default is int64.</param>
        /// <param name="endpoint">If true, sample from the closed interval <c>[low, high]</c>.</param>
        /// <returns>The draws, of <paramref name="dtype"/>.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is a native non-integer dtype.</exception>
        /// <exception cref="ValueError">Non-native byte order, bounds outside the dtype, or an empty interval — as for the signed overload.</exception>
        /// <remarks>
        ///     Both overloads share one arbitrary-precision (<see cref="Int128"/>) range check, so every value
        ///     expressible in either spelling is validated exactly as NumPy validates the Python int.
        /// </remarks>
        public NDArray integers(ulong low, ulong? high = null, Shape size = default, DType dtype = null, bool endpoint = false)
            => BoundedIntegers.Draw(_bitGenerator, low, high.HasValue ? (Int128)high.Value : (Int128?)null, size,
                                    dtype ?? DType.Int64, endpoint, useMasked: false, "integers", legacyByteOrder: false);

        /// <summary>
        ///     Arbitrary-precision overload of <see cref="integers(long, long?, Shape, DType, bool)"/> — the C# spelling of a
        ///     Python int past the <c>long</c>/<c>ulong</c> range, such as NumPy's full-range idiom
        ///     <c>integers(0, 2**64, dtype=np.uint64)</c> (an EXCLUSIVE <c>2**64</c>) or a bound NumPy rejects as out of range.
        /// </summary>
        /// <param name="low">Lowest integer drawn (or, when <paramref name="high"/> is null, one above the highest with low = 0).</param>
        /// <param name="high">If provided, one above the largest integer drawn (or the largest when <paramref name="endpoint"/>).</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value; <c>()</c> returns a 0-d array.</param>
        /// <param name="dtype">Desired native integer or bool dtype. Default is int64.</param>
        /// <param name="endpoint">If true, sample from the closed interval <c>[low, high]</c>.</param>
        /// <returns>The draws, of <paramref name="dtype"/>.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is a native non-integer dtype.</exception>
        /// <exception cref="ValueError">Non-native byte order, bounds outside the dtype (<c>low/high is out of bounds for
        ///     &lt;dtype&gt;</c>), or an empty interval.</exception>
        /// <remarks>
        ///     Validated exactly as NumPy validates the Python int: a value past <c>±2**100</c> — beyond every dtype's range —
        ///     reaches the checks clamped, which report the same error the exact value would.
        /// </remarks>
        public NDArray integers(BigInteger low, BigInteger? high = null, Shape size = default, DType dtype = null, bool endpoint = false)
            => BoundedIntegers.Draw(_bitGenerator, BoundedIntegers.ClampToInt128(low),
                                    high.HasValue ? BoundedIntegers.ClampToInt128(high.Value) : (Int128?)null, size,
                                    dtype ?? DType.Int64, endpoint, useMasked: false, "integers", legacyByteOrder: false);

        /// <summary>
        ///     Array-bounds overload of <see cref="integers(long, long?, Shape, DType, bool)"/>: NumPy's
        ///     <c>integers(low, high)</c> with array-like bounds, which broadcast against each other (and against
        ///     <paramref name="size"/> when one is given) — one draw per output position, each from its own
        ///     <c>[low, high)</c> (or <c>[low, high]</c> when <paramref name="endpoint"/>).
        /// </summary>
        /// <param name="low">The low bound(s) — any integer, bool or float array (a float truncates toward zero, as Python's
        ///     <c>int()</c> and NumPy's forced cast do). Null is Python's <c>None</c> (a <see cref="TypeError"/>).</param>
        /// <param name="high">The high bound(s); null is NumPy's <c>high=None</c>: the bounds are <c>[0, low)</c>.</param>
        /// <param name="size">Output shape; default is the bounds' broadcast shape (or one value when both are 0-d).</param>
        /// <param name="dtype">Desired native integer or bool dtype. Default is int64.</param>
        /// <param name="endpoint">If true, sample from the closed interval <c>[low, high]</c>.</param>
        /// <returns>The draws, of <paramref name="dtype"/>.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is a native non-integer dtype; a bound is <c>None</c> or a
        ///     0-d complex.</exception>
        /// <exception cref="ValueError">Non-native byte order; a bound outside the dtype; an empty interval anywhere; a NaN
        ///     bound; bounds that do not broadcast, or a size they do not broadcast with; a negative size.</exception>
        /// <exception cref="OverflowException">An infinite bound (NumPy's <c>OverflowError</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html
        ///     <br/>
        ///     NumPy's <c>_rand_&lt;dtype&gt;</c>: two 0-d bounds take the scalar path above (Python's <c>int()</c> of each);
        ///     otherwise <c>_rand_&lt;dtype&gt;_broadcast</c> — the bounds checked over the whole array in NumPy's order, then
        ///     one Lemire draw per position in C order, the 8/16-bit and bool dtypes splitting 32-bit words across positions.
        ///     NumPy's two broadcast quirks are reproduced (a size smaller than the bounds' broadcast, and 64-bit float
        ///     bounds in a non-C layout) — see <see cref="BoundedIntegers.DrawArray"/>. Byte-identical to
        ///     <c>default_rng(seed).integers(low_arr, high_arr, ...)</c>.
        /// </remarks>
        public NDArray integers(NDArray low, NDArray high = null, Shape size = default, DType dtype = null, bool endpoint = false)
            => BoundedIntegers.DrawArray(_bitGenerator, low, high, size, dtype ?? DType.Int64, endpoint, useMasked: false,
                                         "integers", legacyByteOrder: false);

        /// <summary>
        ///     Return random bytes.
        /// </summary>
        /// <param name="length">Number of random bytes.</param>
        /// <returns>
        ///     A 1-D <see cref="NDArray{T}"/> of <see cref="byte"/> (dtype <c>uint8</c>), length
        ///     <paramref name="length"/> — the NumSharp analogue of NumPy's <c>bytes</c> object.
        /// </returns>
        /// <exception cref="ValueError"><paramref name="length"/> is below the smallest length NumPy accepts (<c>negative dimensions are not allowed</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.bytes.html
        ///     <br/>
        ///     Byte-identical to NumPy: draws <c>ceil(length/4)</c> uint32 words from the bit generator (via the
        ///     32-bit buffered path), packs them little-endian, and truncates to <paramref name="length"/>.
        ///     As with <see cref="NumPyRandom.bytes"/>, the result is an unmanaged-backed
        ///     <see cref="NDArray{T}"/>, so — matching NumPy's 64-bit <c>npy_intp</c> length — it is
        ///     NOT capped at <see cref="Array.MaxLength"/> and a request over 2 GiB still succeeds.
        /// </remarks>
        public NDArray<byte> bytes(long length)
        {
            lock (_bitGenerator.@lock)
                return NumPyRandom.BytesCore(length, static bg => bg.NextUInt32(), _bitGenerator);
        }
    }
}
