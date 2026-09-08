using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Get information about the step size of a date or time type — NumPy's <c>np.datetime_data(dtype)</c>:
        ///     the unit string and the multiplier of a <c>datetime64</c> / <c>timedelta64</c> descriptor
        ///     (<c>np.datetime_data(np.dtype("M8[10ns]")) == ("ns", 10)</c>; a bare <c>M8</c> is <c>("generic", 1)</c>).
        /// </summary>
        /// <param name="dtype">A datetime64 or timedelta64 descriptor (any spelling that converts to <see cref="DType"/>, e.g. <c>"M8[s]"</c>).</param>
        /// <returns>The <c>(unit, count)</c> tuple NumPy returns.</returns>
        /// <exception cref="TypeError"><c>cannot get datetime metadata from non-datetime type</c> — verbatim NumPy.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.datetime_data.html</remarks>
        public static (string unit, int count) datetime_data(DType dtype)
        {
            if (dtype is null || dtype.DatetimeMetadata is not { } meta)
                throw new TypeError("cannot get datetime metadata from non-datetime type");
            return (meta.UnitString, meta.Num);
        }
    }
}
