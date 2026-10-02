using System;
using System.Numerics;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Return random integers from the "discrete uniform" distribution in the half-open interval [low, high).
        /// </summary>
        /// <param name="low">Lowest (signed) integer to be drawn from the distribution (unless high is not provided, in which case this parameter is one above the highest such integer).</param>
        /// <param name="high">If provided, one above the largest (signed) integer to be drawn from the distribution. If null (NumPy's <c>None</c>), results are from [0, low).</param>
        /// <param name="size">Output shape. If None, a single value is returned.</param>
        /// <param name="dtype">Desired dtype of the result. Default is NumPy's <c>dtype=int</c>, which mtrand maps to C <c>long</c>
        ///     (<c>np.dtype("long")</c>): int64 here, the LP64 (Linux/macOS) width NumSharp's legacy integers model — NumPy's
        ///     Windows build, whose <c>long</c> is 32-bit, returns int32 and rejects bounds past <c>2**31</c>.</param>
        /// <returns>Random integers from the appropriate distribution, or a single such random int if size not provided.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is not an integer/bool dtype (<c>Unsupported dtype dtype('float64') for randint</c>).</exception>
        /// <exception cref="ValueError">The bounds fall outside the dtype (<c>low/high is out of bounds for &lt;dtype&gt;</c>) or the interval is empty (<c>low &gt;= high</c>; <c>high &lt;= 0</c> for the one-argument form).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
        ///     <br/>
        ///     NumPy's legacy sampler — masked rejection (<c>use_masked=True</c>), NOT the Generator's Lemire — with
        ///     the per-dtype stream consumption (8/16-bit dtypes and bool split one 32-bit word; a range that fits
        ///     32 bits uses 32-bit draws even when the bounds need 64), so every dtype is byte-identical to
        ///     <c>np.random.RandomState(seed).randint(...)</c>. <c>high=-1</c> is a real bound (NumPy's
        ///     <c>randint(-10, -1)</c>); only null means "high omitted". A non-native byte order is drawn as the
        ///     native dtype (NumPy only warns). A zero-size request returns an empty array before the bounds are
        ///     checked. Holds the bit generator's lock for the fill. The default dtype does not change the draws: a
        ///     range that fits 32 bits takes the same buffered 32-bit masked sampler at either integer width, so the
        ///     values equal the win-amd64 int32 results NumPy's Windows build returns for the same seed.
        /// </remarks>
        public NDArray randint(long low, long? high = null, Shape size = default, DType dtype = null)
            => BoundedIntegers.Draw(randomizer, low, high.HasValue ? (Int128)high.Value : (Int128?)null, size,
                                    dtype ?? LegacyLong, endpoint: false, useMasked: true, "randint", legacyByteOrder: true);

        /// <summary>
        ///     Unsigned overload of <see cref="randint(long, long?, Shape, DType)"/> — the C# spelling of a bound above
        ///     <see cref="long.MaxValue"/> (the upper half of the uint64 range, e.g. <c>randint(2**63, 2**64 - 1,
        ///     dtype=np.uint64)</c>) and of the exclusive int64 high <c>2**63</c>.
        /// </summary>
        /// <param name="low">Lowest integer drawn (or, when <paramref name="high"/> is null, one above the highest with low = 0).</param>
        /// <param name="high">If provided, one above the largest integer drawn.</param>
        /// <param name="size">Output shape. If None, a single value is returned.</param>
        /// <param name="dtype">Desired integer dtype. Default is NumPy's C <c>long</c> — int64 in NumSharp's LP64 model.</param>
        /// <returns>Random integers from the appropriate distribution, or a single such random int if size not provided.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is not an integer/bool dtype.</exception>
        /// <exception cref="ValueError">The bounds fall outside the dtype or the interval is empty.</exception>
        public NDArray randint(ulong low, ulong? high = null, Shape size = default, DType dtype = null)
            => BoundedIntegers.Draw(randomizer, low, high.HasValue ? (Int128)high.Value : (Int128?)null, size,
                                    dtype ?? LegacyLong, endpoint: false, useMasked: true, "randint", legacyByteOrder: true);

        /// <summary>
        ///     Arbitrary-precision overload of <see cref="randint(long, long?, Shape, DType)"/> — the C# spelling of a Python
        ///     int past the <c>long</c>/<c>ulong</c> range (<c>randint(0, 2**64, dtype=np.uint64)</c>, or a bound NumPy rejects).
        /// </summary>
        /// <param name="low">Lowest integer drawn (or, when <paramref name="high"/> is null, one above the highest with low = 0).</param>
        /// <param name="high">If provided, one above the largest integer drawn.</param>
        /// <param name="size">Output shape. If None, a single value is returned.</param>
        /// <param name="dtype">Desired integer dtype. Default is NumPy's C <c>long</c> — int64 in NumSharp's LP64 model.</param>
        /// <returns>Random integers from the appropriate distribution, or a single such random int if size not provided.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is not an integer/bool dtype.</exception>
        /// <exception cref="ValueError">The bounds fall outside the dtype or the interval is empty.</exception>
        /// <remarks>
        ///     Validated exactly as NumPy validates the Python int: a value past <c>±2**100</c> — beyond every dtype's range —
        ///     reaches the checks clamped, which report the same error the exact value would.
        /// </remarks>
        public NDArray randint(BigInteger low, BigInteger? high = null, Shape size = default, DType dtype = null)
            => BoundedIntegers.Draw(randomizer, BoundedIntegers.ClampToInt128(low),
                                    high.HasValue ? BoundedIntegers.ClampToInt128(high.Value) : (Int128?)null, size,
                                    dtype ?? LegacyLong, endpoint: false, useMasked: true, "randint", legacyByteOrder: true);

        /// <summary>
        ///     Array-bounds overload of <see cref="randint(long, long?, Shape, DType)"/>: NumPy's <c>randint(low, high)</c>
        ///     with array-like bounds, which broadcast against each other (and against <paramref name="size"/> when one is
        ///     given) — one draw per output position, each from its own <c>[low, high)</c>.
        /// </summary>
        /// <param name="low">The low bound(s) — any integer, bool or float array (a float truncates toward zero). Null is
        ///     Python's <c>None</c> (a <see cref="TypeError"/>).</param>
        /// <param name="high">The high bound(s); null is NumPy's <c>high=None</c>: the bounds are <c>[0, low)</c>.</param>
        /// <param name="size">Output shape; default is the bounds' broadcast shape (or one value when both are 0-d).</param>
        /// <param name="dtype">Desired integer dtype. Default is NumPy's C <c>long</c> — int64 in NumSharp's LP64 model.</param>
        /// <returns>The draws, of <paramref name="dtype"/>.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is not an integer/bool dtype; a bound is <c>None</c> or a 0-d
        ///     complex.</exception>
        /// <exception cref="ValueError">A bound outside the dtype; an empty interval anywhere; a NaN bound; bounds that do not
        ///     broadcast, or a size they do not broadcast with; a negative size.</exception>
        /// <exception cref="OverflowException">An infinite bound (NumPy's <c>OverflowError</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
        ///     <br/>
        ///     NumPy's <c>_rand_&lt;dtype&gt;</c> with the legacy masked rejection: two 0-d bounds take the scalar path; otherwise
        ///     <c>_rand_&lt;dtype&gt;_broadcast</c>, one draw per position in C order (see <see cref="BoundedIntegers.DrawArray"/>
        ///     for NumPy's check order and its two broadcast quirks). Byte-identical to
        ///     <c>np.random.RandomState(seed).randint(low_arr, high_arr, ...)</c> — the LP64 width, as for every legacy integer.
        /// </remarks>
        public NDArray randint(NDArray low, NDArray high = null, Shape size = default, DType dtype = null)
            => BoundedIntegers.DrawArray(randomizer, low, high, size, dtype ?? LegacyLong, endpoint: false, useMasked: true,
                                         "randint", legacyByteOrder: true);
    }
}
