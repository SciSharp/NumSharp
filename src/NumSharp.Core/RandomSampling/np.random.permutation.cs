using System;
using System.Numerics;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Randomly permute <c>np.arange(x)</c>.
        /// </summary>
        /// <param name="x">The length of the range.</param>
        /// <returns>
        ///     A shuffled range of dtype <c>np.result_type(x, np.long)</c> — int64 (<see cref="LegacyLong"/>): whether the
        ///     int is weak (a Python int) or a strong int32, the result type against a 64-bit <c>long</c> is int64, LP64
        ///     NumPy's answer (Windows NumPy's 32-bit <c>long</c> gives int32).
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html
        ///     <br/>The shuffle draws the same <c>random_interval</c> indices at any element width, so the permutation equals
        ///     NumPy's on either platform; only the dtype follows the model.
        /// </remarks>
        public NDArray permutation(int x) => permutation((long)x);

        /// <summary>
        ///     Randomly permute <c>np.arange(x)</c> (a C# <c>long</c> is a weak integer, like the Python int it stands for).
        /// </summary>
        /// <param name="x">The length of the range; a negative length gives an empty range, as <c>np.arange</c> does.</param>
        /// <returns>A shuffled int64 range (<c>np.result_type(x, np.long)</c> with the LP64 <c>long</c>, <see cref="LegacyLong"/>).</returns>
        /// <exception cref="ValueError">A length whose int64 byte count cannot be addressed (<c>array is too big; ...</c>).</exception>
        /// <exception cref="OutOfMemoryException">A valid length that cannot be allocated (NumPy's <c>MemoryError</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html
        /// </remarks>
        public NDArray permutation(long x) => LegacyPermutation(x);

        /// <summary>
        ///     NumPy's legacy <c>permutation(n)</c> for an integer (a Python int, so it may exceed int64 — the
        ///     <c>choice(pop, replace=False)</c> path passes a uint64 population through here):
        ///     <c>arr = np.arange(n, dtype=np.long); self.shuffle(arr)</c>.
        /// </summary>
        /// <param name="n">The range length.</param>
        /// <returns>The shuffled int64 range (<see cref="LegacyLong"/>).</returns>
        /// <exception cref="ValueError">The length is past npy_intp (<c>Maximum allowed size exceeded</c>) or its int64 byte
        ///     count cannot be addressed (<c>array is too big; ...</c>).</exception>
        /// <exception cref="OutOfMemoryException">A valid length that cannot be allocated (NumPy's <c>MemoryError</c>).</exception>
        private NDArray LegacyPermutation(BigInteger n)
        {
            var nd = np.arange(ArangeLength(n));
            shuffle(nd);
            return nd;
        }

        /// <summary>
        ///     The length NumPy's <c>np.arange(n)</c> gives an integer stop <paramref name="n"/> (start 0, step 1) —
        ///     <c>PyArray_ArangeObj</c>'s <c>_calc_length</c> + <c>_arange_safe_ceil_to_intp</c>, including what they do at
        ///     the int64 edge on x86.
        /// </summary>
        /// <param name="n">The stop (a Python int).</param>
        /// <returns>The length: <paramref name="n"/> itself below the edge, 0 for a non-positive stop and for the band
        ///     whose double rounds to exactly <c>2**63</c>.</returns>
        /// <exception cref="ValueError">The length's double exceeds npy_intp's (<c>Maximum allowed size exceeded</c> —
        ///     NumPy's OverflowError from the length computation, re-raised as ValueError).</exception>
        /// <remarks>
        ///     NumPy computes the length as <c>ceil(float((stop - start) / step))</c> — a DOUBLE — and range-checks it
        ///     against <c>(double)NPY_MAX_INTP</c>, which rounds to <c>2**63</c>. A stop in <c>[2**63 - 512, 2**63 + 1024]</c>
        ///     rounds to exactly <c>2**63</c>, passes the check, and its <c>(npy_intp)</c> cast (x86's cvttsd2si) is
        ///     <c>INT64_MIN</c> — a negative length, which arange turns into an EMPTY array. So
        ///     <c>np.random.permutation(2**63 - 1)</c> is empty on NumPy's x86 builds (Windows and Linux alike, probed), while
        ///     a stop that stays below the band allocates (and <c>2**62</c> is <c>array is too big</c>); larger stops raise.
        ///     The double conversion is Python's correctly rounded one (<see cref="PythonInt.ToDouble"/>), not .NET's.
        /// </remarks>
        private static long ArangeLength(BigInteger n)
        {
            if (n.Sign <= 0)
                return 0;
            const double MaxIntpAsDouble = 9223372036854775808.0; // (double)NPY_MAX_INTP: 2**63 after rounding
            double ivalue = Math.Ceiling(PythonInt.ToDouble(n));
            if (!(ivalue <= MaxIntpAsDouble))
                throw new ValueError("Maximum allowed size exceeded");
            // (npy_intp)2**63 is INT64_MIN on x86: a negative length, i.e. an empty range.
            return ivalue == MaxIntpAsDouble ? 0 : (long)ivalue;
        }

        /// <summary>
        ///     Randomly permute a sequence (a copy of <paramref name="x"/>; for a multi-dimensional array only the first
        ///     axis is permuted).
        /// </summary>
        /// <param name="x">The array to permute (at least 1-D).</param>
        /// <returns>A permuted copy; <paramref name="x"/> is not modified.</returns>
        /// <exception cref="IndexError"><paramref name="x"/> is 0-d (<c>x must be an integer or at least 1-dimensional</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html
        ///     <br/>
        ///     NumPy's legacy form: a 1-D input is copied and shuffled in place; an N-D input shuffles an intp index
        ///     vector (the same <c>random_interval</c> draws) and returns <c>x[idx]</c>.
        /// </remarks>
        [NDScoped] // reclaims the N-D path's index vector; the gathered copy is yielded
        public NDArray permutation(NDArray x)
        {
            if (x.ndim < 1)
                throw new IndexError("x must be an integer or at least 1-dimensional");

            if (x.ndim == 1)
            {
                var arr = x.copy();
                shuffle(arr);
                return arr;
            }

            var idx = np.arange(x.shape[0]);
            shuffle(idx);
            return np.take(x, idx, axis: 0);
        }
    }
}
