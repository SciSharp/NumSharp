using System;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     <c>numpy.polynomial</c>'s least-squares fits (plan U6): <c>polyfit</c>, <c>chebfit</c>, <c>legfit</c>,
    ///     <c>lagfit</c>, <c>hermfit</c>, <c>hermefit</c>. Pairs 1:1 with <c>gen_oracle.py</c>'s <c>gen_polyfit</c>: the named
    ///     arguments <c>x</c>, <c>y</c>, <c>deg</c>, <c>rcond</c> and <c>w</c> decode in that order (the generator's encode
    ///     order), <c>full</c> is a plain JSON bool. A <c>full=True</c> case is a tuple of NumPy's five slots
    ///     <c>[c, resids, rank, s, rcond]</c>; any other case the coefficient array.
    /// </summary>
    public static partial class OpRegistry
    {
        /// <summary>Runs one fit case through the basis' facade.</summary>
        /// <param name="op">The corpus key (<c>"chebyshev.chebfit"</c>, …).</param>
        /// <param name="a">The case's argument decoder.</param>
        /// <returns>The facade's result.</returns>
        /// <exception cref="NotSupportedException">A key that is not one of the six fits.</exception>
        private static NDPolyFitResult PolyFitCall(string op, PolyArgs a)
        {
            object x = a.Get("x"), y = a.Get("y"), deg = a.Get("deg"), rcond = a.Opt("rcond");
            bool full = a.Flag("full", false);
            object w = a.Opt("w");
            return op switch
            {
                "polynomial.polyfit" => np.polynomial.polynomial.polyfit(x, y, deg, rcond, full, w),
                "chebyshev.chebfit" => np.polynomial.chebyshev.chebfit(x, y, deg, rcond, full, w),
                "legendre.legfit" => np.polynomial.legendre.legfit(x, y, deg, rcond, full, w),
                "laguerre.lagfit" => np.polynomial.laguerre.lagfit(x, y, deg, rcond, full, w),
                "hermite.hermfit" => np.polynomial.hermite.hermfit(x, y, deg, rcond, full, w),
                "hermite_e.hermefit" => np.polynomial.hermite_e.hermefit(x, y, deg, rcond, full, w),
                _ => throw new NotSupportedException($"polynomial fit '{op}' is not registered in OpRegistry"),
            };
        }

        /// <summary>Whether a corpus key is one of the six fits.</summary>
        /// <param name="op">The key.</param>
        /// <returns>True for <c>{module}.{p}fit</c>.</returns>
        private static bool IsPolyFitOp(string op)
            => op is "polynomial.polyfit" or "chebyshev.chebfit" or "legendre.legfit" or "laguerre.lagfit"
                or "hermite.hermfit" or "hermite_e.hermefit";

        /// <summary>
        ///     A <c>full=True</c> fit's five slots, NumPy's <c>(c, [resids, rank, s, rcond])</c> flattened in order — the
        ///     coefficients as values or (a <c>"facet": "flags"</c> case) their flags.
        /// </summary>
        /// <param name="op">The corpus key.</param>
        /// <param name="a">The case's argument decoder.</param>
        /// <returns>The slots.</returns>
        /// <exception cref="InvalidOperationException">The facade returned no statistics for a full case.</exception>
        private static NDArray[] PolyFitTuple(string op, PolyArgs a)
        {
            NDPolyFitResult r = PolyFitCall(op, a);
            if (r.residuals is null)
                throw new InvalidOperationException($"{op}: a full=True fit returned no statistics");
            // The statistics are read before the facet, which disposes the coefficient array of a flags case.
            NDArray res = r.residuals, rank = r.rank, s = r.singular_values, rc = r.rcond;
            return new[] { a.CalcFacet(r), res, rank, s, rc };
        }
    }
}
