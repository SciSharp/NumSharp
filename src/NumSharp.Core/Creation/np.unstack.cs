using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Split an array into a sequence of arrays along the given axis.
        ///     The <paramref name="axis"/> parameter specifies the dimension along
        ///     which the array will be split. For example, if <c>axis=0</c> (the
        ///     default) it will be the first dimension and if <c>axis=-1</c> it
        ///     will be the last dimension.
        ///     Added in NumPy 2.1.
        /// </summary>
        /// <param name="x">The array to be unstacked.</param>
        /// <param name="axis">Axis along which the array will be split. Default: 0.</param>
        /// <returns>
        ///     The unstacked arrays — <c>x.shape[axis]</c> VIEWS into
        ///     <paramref name="x"/> (shared memory, matching NumPy), each with
        ///     shape equal to <c>x.shape</c> with the <paramref name="axis"/>
        ///     entry removed.
        ///     An empty <paramref name="x"/> gives empty views too. A <c>numpy.char.chararray</c> operand of two or
        ///     more dimensions gives chararray views (NumPy iterates the chararray); a 1-d one gives its elements as
        ///     NumPy's iteration does, the read-only stand-ins of the Python <c>str</c> / <c>bytes</c> with the trailing
        ///     whitespace stripped.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="x"/> is 0-d ("Input array must be at least 1-d.").</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range: moveaxis' "source: axis … is out of
        ///     bounds for array of dimension …".</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.unstack.html
        ///     <br></br>
        ///     <c>unstack</c> serves as the reverse operation of <see cref="stack"/>:
        ///     <c>stack(unstack(x, axis), axis) == x</c>. Semantically equivalent to
        ///     NumPy's <c>tuple(np.moveaxis(x, axis, 0))</c> — moving <c>axis</c>
        ///     to the front and then indexing it away leaves the remaining axes in
        ///     their original relative order, so each view is constructed directly
        ///     by dropping <c>axis</c> from dims/strides and advancing the offset
        ///     by <c>i * strides[axis]</c>; no iterator or data movement anywhere.
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray[] unstack(NDArray x, int axis = 0)
        {
            if (x is null)
                throw new ArgumentNullException(nameof(x));

            // NumPy's statements (the rank check, moveaxis' axis check, one view per index along the axis) are the
            // array object's.
            return NDArray.unstack(x, axis);
        }
    }
}
