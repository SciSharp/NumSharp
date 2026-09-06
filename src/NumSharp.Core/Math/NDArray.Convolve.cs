using System;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Returns the discrete, linear convolution of two one-dimensional sequences.
        ///
        /// The convolution operator is often seen in signal processing, where it models the effect of a linear time-invariant system on a signal[1]. In probability theory, the sum of two independent random variables is distributed according to the convolution of their individual distributions.
        ///
        /// If v is longer than a, the arrays are swapped before computation.
        /// </summary>
        /// <param name="v">The second one-dimensional input array.</param>
        /// <param name="mode">'full', 'same', or 'valid'. Default is 'full'.</param>
        /// <returns>Discrete, linear convolution of a and v.</returns>
        /// <remarks>
        /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
        ///
        /// convolve is <c>correlate(a, v[::-1], mode)</c> (numpy/_core/numeric.py): it reverses the
        /// kernel and runs the shared sliding multiply-accumulate engine (<see cref="SlidingCorrelate"/>).
        /// Unlike correlate it does NOT conjugate a complex kernel and is commutative, so the
        /// "swap if v longer" below needs no output reversal. See <c>np.correlate.cs</c> for the
        /// float bit-parity notes (both share the same kernel).
        /// </remarks>
        [NDScoped]
        public NDArray convolve(NDArray v, string mode = "full")
        {
            NDArray a = this;

            // Validate inputs are 1D
            if (a.ndim != 1)
                throw new IncorrectShapeException("First argument must be a 1-dimensional array.");
            if (v.ndim != 1)
                throw new IncorrectShapeException("Second argument must be a 1-dimensional array.");

            // Validate non-empty (NumPy convolve's own messages, distinct from correlate's).
            if (a.size == 0)
                throw new ArgumentException("a cannot be empty", nameof(a));
            if (v.size == 0)
                throw new ArgumentException("v cannot be empty", nameof(v));

            var m = ParseSlidingMode(mode);

            // NumPy swaps if v is longer than a (convolve is commutative, so no output reversal).
            if (v.size > a.size)
            {
                var temp = a;
                a = v;
                v = temp;
            }

            // Determine output type using NumPy's type promotion rules.
            var retType = np._FindCommonType(a, v);

            // convolve(a, v) == correlate(a, v[::-1]): reverse the kernel, then the shared engine
            // reads it forward. The reversed view is materialized to a contiguous, retType buffer;
            // the view and both materializations are scope-tracked and die with the call.
            NDArray data = MaterializeForSliding(a, retType);
            NDArray kernel = MaterializeForSliding(v["::-1"], retType);

            return SlidingCorrelate(data, kernel, retType, m);
        }
    }
}
