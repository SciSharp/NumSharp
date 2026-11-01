using System;

namespace NumSharp.Backends
{
    public partial class DefaultFftEngine
    {
        // The sample-frequency helpers of numpy/fft/_helper.py — NumPy's statements over the array object's own
        // routines (np.arange's PyArray_ArangeObj, np.concatenate's NDArray.concatenate) and the operand engine's
        // arithmetic; they compute FULLY (no kernel) and call no np.* facade, so the engine layer never re-enters the
        // facade's class boundary. fftshift and ifftshift are not members: they roll the array object (np.fft's facades
        // over np.roll).

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements in NumPy's order: <c>val = 1.0 / (n * d)</c> (Python's ZeroDivisionError when the product
        ///     is zero, for either sign of n), <c>results = empty(n, int, device=device)</c> (the device check, then the
        ///     negative length), the two <c>arange</c> halves, then <c>results * val</c>. NumPy's <c>isinstance(n,
        ///     integer_types)</c> guard is the C# parameter type: the np.fft facade's double overload raises it.
        /// </remarks>
        [NDScoped]
        public override NDArray Fftfreq(long n, double d = 1.0, string device = null)
        {
            // NumPy's error ORDER (probed 2.4.2): val = 1.0/(n*d) FIRST — ZeroDivisionError when
            // n*d == 0 (i.e. n == 0 OR d == 0, for ANY sign of n; .NET's float 1/0 is +inf, not a
            // throw, so the divisor is checked explicitly). THEN empty(n, int, device=device), which
            // validates the device before it rejects a negative length.
            double nd = (double)n * d;
            if (nd == 0.0)
                throw new DivideByZeroException("float division by zero");
            DeviceArguments.ValidateDevice(device);
            if (n < 0)
                throw new ValueError("negative dimensions are not allowed");
            double val = 1.0 / nd;
            long N = (n - 1) / 2 + 1;                     // n>=1 here, so int division == floor division.
            // np.arange(start, stop) is UnmanagedStorage.ArangeObj with step 1 and no dtype (int64 for Python ints), and
            // np.concatenate of the two plain halves is the array object's NDArray.concatenate along axis 0 with
            // np.concatenate's default 'same_kind' casting: the facades' own bodies, without their class-boundary test
            // (both halves are plain arrays the engine just made).
            var p1 = UnmanagedStorage.ArangeObj(0L, N, 1L, null);          // [0 .. N-1]
            var p2 = UnmanagedStorage.ArangeObj(-(n / 2), 0L, 1L, null);   // [-(n/2) .. -1]
            var results = NDArray.concatenate(new[] { p1, p2 }, 0, null, null, "same_kind");   // length n
            return results * val;                         // -> float64
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements in NumPy's order: <c>val = 1.0 / (n * d)</c>, <c>N = n // 2 + 1</c>,
        ///     <c>results = arange(0, N, dtype=int, device=device)</c> (the device check; a negative n gives an empty result,
        ///     not an error), then <c>results * val</c>.
        /// </remarks>
        [NDScoped]
        public override NDArray Rfftfreq(long n, double d = 1.0, string device = null)
        {
            // NumPy: val = 1.0/(n*d) FIRST (ZeroDivisionError when n*d == 0), THEN arange(0, n//2+1,
            // device=device) (device validated here). There is NO negative-dimension check — a
            // negative n makes n//2+1 <= 0 so arange yields an EMPTY float64 array
            // (rfftfreq(-5) -> [] rather than an error; unlike fftfreq, which calls empty(n)).
            double nd = (double)n * d;
            if (nd == 0.0)
                throw new DivideByZeroException("float division by zero");
            DeviceArguments.ValidateDevice(device);
            double val = 1.0 / nd;
            // N = n//2 + 1. Arithmetic right-shift is floor-division-by-2 for two's-complement ints
            // (n>>1 == n//2 for BOTH signs), so a negative n gives N<=0 exactly as NumPy's floor does
            // (C#'s '/' truncates toward zero, which would wrongly make rfftfreq(-1) -> [0.0]).
            long N = (n >> 1) + 1;
            // np.arange(0, N)'s body (UnmanagedStorage.ArangeObj, step 1, no dtype); an empty result when N <= 0.
            var results = UnmanagedStorage.ArangeObj(0L, Math.Max(0L, N), 1L, null);
            return results * val;                         // -> float64
        }
    }
}
