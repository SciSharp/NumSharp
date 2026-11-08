using System;

namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     NumPy's <c>finfo.__repr__</c> (<c>numpy/_core/getlimits.py</c>), the printing half of <see cref="finfo"/>: the
    ///     dtype decisions it takes (which scalar prints the resolution, which printf format prints the limits) live here,
    ///     below the facade, as the layering plan puts every dtype branch (ratchet R1, <c>docs/plans/np-layer-dtype-logic.md</c>).
    /// </summary>
    internal static class MachineLimitsRepr
    {
        /// <summary>
        ///     <c>repr(np.finfo(dtype))</c>, the same text under every culture:
        ///     <c>finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)</c>.
        /// </summary>
        /// <param name="dtype">The finfo's dtype (already the real part's for a complex input).</param>
        /// <param name="resolution">The decimal resolution, held as a double.</param>
        /// <param name="min">The most negative finite value.</param>
        /// <param name="max">The largest finite value.</param>
        /// <returns>The repr text.</returns>
        /// <remarks>
        ///     NumPy's statements: <c>resolution</c> is the dtype's own scalar printed by <c>str()</c>
        ///     (<c>np.float16(10) ** -3</c> prints <c>0.001</c>, a float32 <c>1e-06</c>), and <c>min</c> / <c>max</c> go
        ///     through the dtype's <c>_MACHAR_PARAMS</c> printf format, stripped: <c>%24.16e</c> for float64,
        ///     <c>%15.7e</c> for float32, <c>%12.5e</c> for float16 (<c>6.55040e+04</c>; printf widens the scalar with
        ///     <c>float()</c>). The long double has no such format, so NumPy prints <c>str()</c> of the value, and so does
        ///     NumSharp's decimal (a dtype NumPy does not have, whose limits are held as doubles). Every number goes
        ///     through NumSharp's ports of NumPy's printers (<see cref="ArrayFormatter.ScalarStr"/>,
        ///     <see cref="PrintfFormatter.Format"/>), never .NET's current culture (parity QA-41).
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        internal static string Finfo(DType dtype, double resolution, double min, double max)
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));
            NPTypeCode code = dtype.GetTypeCode();
            // float16 / float32 print as their own scalars (str() picks their shortest digits, printf widens them with
            // float()); everything else is held and printed as a double.
            NPTypeCode scalarCode = code is NPTypeCode.Half or NPTypeCode.Single ? code : NPTypeCode.Double;
            string format = ReferenceEquals(dtype.Meta, DTypeRegistry.LongDouble) ? null : code switch
            {
                NPTypeCode.Double => "%24.16e",
                NPTypeCode.Single => "%15.7e",
                NPTypeCode.Half => "%12.5e",
                _ => null,
            };

            object Box(double value) => scalarCode switch
            {
                NPTypeCode.Half => (Half)value,
                NPTypeCode.Single => (float)value,
                _ => value,
            };

            string Limit(double value) => format is null
                ? ArrayFormatter.ScalarStr(Box(value), scalarCode)
                : PrintfFormatter.Format(format, Box(value), scalarCode).Trim();

            return "finfo(resolution=" + ArrayFormatter.ScalarStr(Box(resolution), scalarCode)
                   + ", min=" + Limit(min) + ", max=" + Limit(max) + ", dtype=" + dtype.name + ")";
        }
    }
}
