using System;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Cross-correlation of two 1-dimensional sequences:
        /// <c>c_k = sum_n a_{n+k} * conj(v_n)</c> (NumPy signal-processing convention).
        /// </summary>
        /// <param name="v">The second one-dimensional input array.</param>
        /// <param name="mode">'valid', 'same', or 'full'. Default is 'valid' (unlike convolve, which defaults to 'full').</param>
        /// <returns>Discrete cross-correlation of a and v.</returns>
        /// <remarks>
        /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.correlate.html
        ///
        /// Port of NumPy's <c>PyArray_Correlate2</c> (numpy/_core/src/multiarray/multiarraymodule.c):
        /// the second argument is complex-conjugated first (real inputs unchanged); the shared
        /// engine (<see cref="SlidingCorrelate"/>) then swaps the operands when <c>len(a) &lt; len(v)</c>
        /// — correlate is NOT commutative — and the output is reversed in that case
        /// (<c>_pyarray_revert</c>). The engine reads the kernel forward. Float bit-parity notes are
        /// in <c>np.correlate.cs</c>.
        /// </remarks>
        public NDArray correlate(NDArray v, string mode = "valid")
        {
            NDArray a = this;

            // NumPy parses the mode during argument parsing, before touching the arrays.
            var m = ParseSlidingMode(mode);

            // NumPy's PyArray_FromAny(op, typec, 1, 1, ...) requires exactly 1-D operands (a 0-d
            // scalar or a 2-D array is rejected — correlate, unlike convolve, does not promote 0-d).
            if (a.ndim != 1)
                throw new IncorrectShapeException("First argument must be a 1-dimensional array.");
            if (v.ndim != 1)
                throw new IncorrectShapeException("Second argument must be a 1-dimensional array.");

            // Empty check (inside _pyarray_correlate, on the pre-swap operands, so "first"/"second"
            // always refer to a/v regardless of the later swap).
            if (a.size == 0)
                throw new ArgumentException("first array argument cannot be empty", nameof(a));
            if (v.size == 0)
                throw new ArgumentException("second array argument cannot be empty", nameof(v));

            var retType = np._FindCommonType(a, v);

            // Conjugate the (common-typed) second argument for complex inputs, exactly as
            // PyArray_Correlate2 does — BEFORE the swap, so an inverted correlate uses conj(v) as
            // the data and the untouched a as the kernel.
            NDArray kernel = MaterializeForSliding(v, retType);
            if (retType == NPTypeCode.Complex)
                kernel = MaterializeForSliding(np.conjugate(kernel), retType);

            NDArray data = MaterializeForSliding(a, retType);

            // _pyarray_correlate swaps so the data is the longer operand, remembering it inverted.
            bool inverted = false;
            if (data.size < kernel.size)
            {
                var tmp = data;
                data = kernel;
                kernel = tmp;
                inverted = true;
            }

            NDArray result = SlidingCorrelate(data, kernel, retType, m);

            // If we swapped, reverse the output (ret = ret[::-1]).
            if (inverted)
                result = result["::-1"].copy();

            return result;
        }
    }
}
