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
        /// <exception cref="ArgumentException">An unknown <paramref name="mode"/> (NumPy's two mode texts).</exception>
        /// <exception cref="ValueError"><c>object of too small depth for desired array</c> (a 0-d operand) or <c>object too
        ///     deep for desired array</c> (two or more dimensions), <c>first array argument cannot be empty</c> /
        ///     <c>second array argument cannot be empty</c>, or — for a fixed-width string operand, after every check
        ///     above — <c>data type must provide an itemsize</c>.</exception>
        [NDScoped]
        public NDArray correlate(NDArray v, string mode = "valid")
        {
            NDArray a = this;

            // NumPy parses the mode during argument parsing, before touching the arrays.
            var m = ParseSlidingMode(mode);

            // NumPy's PyArray_FromAny(op, typec, 1, 1, ...) requires exactly 1-D operands (a 0-d
            // scalar or a 2-D array is rejected — correlate, unlike convolve, does not promote 0-d),
            // with its own two texts, a then v.
            ThrowIfNotOneDimensional(a);
            ThrowIfNotOneDimensional(v);

            // Empty check (inside _pyarray_correlate, on the pre-swap operands, so "first"/"second"
            // always refer to a/v regardless of the later swap).
            if (a.size == 0)
                throw new ValueError("first array argument cannot be empty");
            if (v.size == 0)
                throw new ValueError("second array argument cannot be empty");

            // _pyarray_correlate then allocates its result for the type number, which a string type cannot size.
            if (Backends.Strings.NDStringUfuncs.IsString(a) || Backends.Strings.NDStringUfuncs.IsString(v))
                throw Backends.Strings.NDStringUfuncs.NoItemsize();

            var retType = np._FindCommonType(a, v);

            // The materializations, the complex conjugate, and the inverted-path forward buffer
            // are all scope-tracked; only the yielded result survives (a/v are never tracked).

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

            // conj(v) is a fresh array, so only a one-element `a` passed through with a non-positive stride can keep a
            // complex product off cblas (see DotOperandBlasable).
            bool complexDotViaBlas = DotOperandBlasable(a, retType, reversed: false);
            NDArray result = SlidingCorrelate(data, kernel, retType, m, complexDotViaBlas);

            // If we swapped, reverse the output (ret = ret[::-1]).
            if (inverted)
                result = result["::-1"].copy();

            return result;
        }

        /// <summary>
        ///     NumPy's <c>PyArray_FromAny(op, typec, 1, 1, ...)</c> depth check: an operand of correlate must be exactly 1-D.
        /// </summary>
        /// <param name="operand">The operand to check.</param>
        /// <exception cref="ValueError"><c>object of too small depth for desired array</c> for a 0-d operand, <c>object too
        ///     deep for desired array</c> for two or more dimensions.</exception>
        private static void ThrowIfNotOneDimensional(NDArray operand)
        {
            if (operand.ndim == 0)
                throw new ValueError("object of too small depth for desired array");
            if (operand.ndim > 1)
                throw new ValueError("object too deep for desired array");
        }
    }
}
