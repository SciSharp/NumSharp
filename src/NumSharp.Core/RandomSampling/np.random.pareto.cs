using System;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a Pareto II or Lomax distribution with specified shape.
        /// </summary>
        /// <param name="a">Shape of the distribution. Must be positive (&gt; 0; NaN is accepted and samples NaN, as in NumPy).</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized Pareto distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is <c>&lt;= 0</c> (<c>a &lt;= 0</c>), or <paramref name="size"/> has a
        ///     negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.pareto.html
        ///     <br/>
        ///     NumPy's pareto returns samples from the Pareto II (Lomax) distribution,
        ///     not the classical Pareto distribution. The relationship is:
        ///     if Y ~ Pareto(a, m=1) then X = Y - 1 ~ Lomax(a).
        ///     <br/>
        ///     The probability density function is:
        ///     f(x; a) = a / (1 + x)^(a+1)  for x >= 0
        ///     <br/>
        ///     The mean is 1/(a-1) for a > 1, undefined otherwise.
        ///     <br/>
        ///     NumPy's <c>legacy_pareto</c>: <c>exp(E / a) - 1</c> with <c>E = -log(1 - U)</c> — the legacy spelling (not
        ///     <c>expm1</c>, and not the former <c>U^(-1/a) - 1</c>, which consumed the same uniform but rounded differently).
        ///     One draw per value, so the uniforms come from the bit generator's bulk fill and are transformed in place;
        ///     byte-identical to <c>np.random.RandomState(seed).pareto</c>. Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray pareto(double a, Shape size)
        {
            RandomConstraints.Check(a, "a", ConstraintType.CONS_POSITIVE);

            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(LegacyPareto(ref one, a));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    randomizer.FillDouble(dst, n);
                // The draws are all taken; legacy_pareto's transform of each, in place.
                for (long i = 0; i < n; i++)
                    dst[i] = Math.Exp(-Math.Log(1.0 - dst[i]) / a) - 1;
            }

            return ret;
        }

        /// <summary>
        ///     Draw samples from a Pareto II or Lomax distribution with specified shape.
        /// </summary>
        /// <param name="a">Shape of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the parameterized Pareto distribution.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is <c>&lt;= 0</c>, or a size dimension is negative.</exception>
        /// <remarks>
        ///     A source-compatibility shim ranked BELOW the <c>Shape</c> overloads
        ///     (<c>OverloadResolutionPriority(-1)</c>): an int, an array or a tuple converts to <c>Shape</c> with the same
        ///     meaning, so a C# 13+ caller always binds the NumPy-shaped overload — and <c>size: default</c> (NumPy's
        ///     explicit <c>size=None</c>) is no longer ambiguous between the shims (or, for a <c>long</c> shim, silently a
        ///     zero-length size). Kept so code compiled against it keeps binding.
        /// </remarks>
        [OverloadResolutionPriority(-1)]
        public NDArray pareto(double a, int[] size)
            => pareto(a, new Shape(size));

        /// <summary>
        ///     Draw samples from a Pareto II or Lomax distribution with specified shape.
        /// </summary>
        /// <param name="a">Shape of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized Pareto distribution.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is <c>&lt;= 0</c>, or a size dimension is negative.</exception>
        /// <remarks>
        ///     A source-compatibility shim ranked BELOW the <c>Shape</c> overloads
        ///     (<c>OverloadResolutionPriority(-1)</c>): an int, an array or a tuple converts to <c>Shape</c> with the same
        ///     meaning, so a C# 13+ caller always binds the NumPy-shaped overload — and <c>size: default</c> (NumPy's
        ///     explicit <c>size=None</c>) is no longer ambiguous between the shims (or, for a <c>long</c> shim, silently a
        ///     zero-length size). Kept so code compiled against it keeps binding.
        /// </remarks>
        [OverloadResolutionPriority(-1)]
        public NDArray pareto(double a, long[] size)
            => pareto(a, new Shape(size));

        /// <summary>
        ///     Draw samples from a Pareto II or Lomax distribution with specified shape.
        /// </summary>
        /// <param name="a">Shape of the distribution. Must be positive (&gt; 0).</param>
        /// <param name="size">Output shape as a single integer — NumPy's integer <c>size</c>: one npy_intp (int64) dimension.</param>
        /// <returns>Drawn samples from the parameterized Pareto distribution.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is <c>&lt;= 0</c>, or <paramref name="size"/> is negative.</exception>
        /// <remarks>
        ///     A source-compatibility shim ranked BELOW the <c>Shape</c> overloads
        ///     (<c>OverloadResolutionPriority(-1)</c>): an int, an array or a tuple converts to <c>Shape</c> with the same
        ///     meaning, so a C# 13+ caller always binds the NumPy-shaped overload — and <c>size: default</c> (NumPy's
        ///     explicit <c>size=None</c>) is no longer ambiguous between the shims (or, for a <c>long</c> shim, silently a
        ///     zero-length size). Kept so code compiled against it keeps binding.
        /// </remarks>
        [OverloadResolutionPriority(-1)]
        public NDArray pareto(double a, long size)
            => pareto(a, new long[] { size });

        /// <summary>
        ///     Draw a single sample from a Pareto II or Lomax distribution.
        /// </summary>
        /// <param name="a">Shape of the distribution. Must be positive (&gt; 0).</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is <c>&lt;= 0</c>.</exception>
        public NDArray pareto(double a) => pareto(a, Shape.Scalar);
    }
}
