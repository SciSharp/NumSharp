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
        /// <exception cref="ArgumentException">An unknown <paramref name="mode"/> (NumPy's two mode texts).</exception>
        /// <exception cref="ValueError"><c>a cannot be empty</c> / <c>v cannot be empty</c> (on the operands after the
        ///     length swap), <c>object too deep for desired array</c> for an operand of two or more dimensions, or — for a
        ///     fixed-width string operand, after every check above — <c>data type must provide an itemsize</c> (NumPy's
        ///     result allocation for the string type number).</exception>
        [NDScoped]
        public NDArray convolve(NDArray v, string mode = "full")
        {
            // NumPy's np.convolve, statement for statement: `array(a, ndmin=1)` (a 0-d operand becomes one element),
            // the swap by len() (the FIRST dimension), the two emptiness checks on the swapped operands, then
            // multiarray.correlate(a, v[::-1], mode): the mode, the 1-D requirement of PyArray_FromAny(op, typec, 1, 1)
            // and the result allocation, which a string type number cannot size.
            NDArray a = this.ndim == 0 ? this.reshape(1) : this;
            if (v.ndim == 0)
                v = v.reshape(1);

            // NumPy swaps if v is longer than a (convolve is commutative, so no output reversal).
            if (v.Shape.dimensions[0] > a.Shape.dimensions[0])
            {
                var temp = a;
                a = v;
                v = temp;
            }

            // Validate non-empty (NumPy convolve's own messages, distinct from correlate's).
            if (a.Shape.dimensions[0] == 0)
                throw new ValueError("a cannot be empty");
            if (v.Shape.dimensions[0] == 0)
                throw new ValueError("v cannot be empty");

            var m = ParseSlidingMode(mode);

            // PyArray_FromAny(op, typec, 1, 1): an operand of two or more dimensions is too deep.
            if (a.ndim != 1 || v.ndim != 1)
                throw new ValueError("object too deep for desired array");

            // The result allocation needs a fixed item size, which a string type number does not have.
            if (Backends.Strings.NDStringUfuncs.IsString(a) || Backends.Strings.NDStringUfuncs.IsString(v))
                throw Backends.Strings.NDStringUfuncs.NoItemsize();

            // Determine output type using NumPy's type promotion rules.
            var retType = np._FindCommonType(a, v);

            // convolve(a, v) == correlate(a, v[::-1]): reverse the kernel, then the shared engine
            // reads it forward. The reversed view is materialized to a contiguous, retType buffer;
            // the view and both materializations are scope-tracked and die with the call.
            NDArray data = MaterializeForSliding(a, retType);
            NDArray kernel = MaterializeForSliding(v["::-1"], retType);

            // NumPy hands a one-element operand to its dotfunc with the operand's own stride (np.convolve's kernel
            // reversed), which can keep a complex product off cblas and on CDOUBLE_dot's plain loop.
            bool complexDotViaBlas = DotOperandBlasable(a, retType, reversed: false) && DotOperandBlasable(v, retType, reversed: true);
            return SlidingCorrelate(data, kernel, retType, m, complexDotViaBlas);
        }
    }
}
