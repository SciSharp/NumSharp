using System;
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
