using System;
using System.Numerics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return an array drawn from elements in <paramref name="choicelist"/>, depending on
        ///     <paramref name="condlist"/>. The output at position <c>m</c> is the <c>m</c>-th element
        ///     of the array in <paramref name="choicelist"/> where the <c>m</c>-th element of the
        ///     corresponding array in <paramref name="condlist"/> is <c>true</c>. When multiple
        ///     conditions are satisfied, the FIRST one encountered in <paramref name="condlist"/> wins;
        ///     positions where every condition is <c>false</c> take <paramref name="default"/>.
        /// </summary>
        /// <param name="condlist">
        ///     The conditions that determine which array in <paramref name="choicelist"/> each output
        ///     element is taken from. Must be boolean arrays and the same length as
        ///     <paramref name="choicelist"/>. All conditions are broadcast against each other.
        /// </param>
        /// <param name="choicelist">
        ///     The arrays the output elements are drawn from. Each entry is either an
        ///     <see cref="NDArray"/> (strong dtype) or a boxed C# scalar. As in NumPy (NEP50), an
        ///     <c>int</c>/<c>float</c>/<c>double</c>/<see cref="Complex"/> literal is a <em>weak</em>
        ///     scalar that adopts the other operands' dtype, while <c>bool</c>, <c>char</c>,
        ///     <see cref="Half"/>, <c>decimal</c>, arrays and every <see cref="NDArray"/> are strong.
        ///     An <see cref="NDArray"/>[] binds here directly via array covariance; scalar choices need
        ///     an explicit <c>new object[] { … }</c>. All choices AND <paramref name="default"/> are
        ///     broadcast against each other.
        /// </param>
        /// <param name="default">
        ///     The value inserted where all conditions are <c>false</c>. <c>null</c> (the C# default)
        ///     is NumPy's <c>default=0</c> — a weak python int. Accepts a scalar or an
        ///     <see cref="NDArray"/> (which participates in the choice broadcast shape).
        /// </param>
        /// <returns>
        ///     A fresh C-contiguous array whose dtype is <see cref="result_type"/> of every choice and
        ///     the default (NEP50), and whose shape is the broadcast of the conditions against the
        ///     choices.
        /// </returns>
        /// <exception cref="ValueError">
        ///     <paramref name="condlist"/> and <paramref name="choicelist"/> differ in length, or
        ///     <paramref name="condlist"/> is empty.
        /// </exception>
        /// <exception cref="TypeError">A condition is not a boolean array.</exception>
        /// <exception cref="IncorrectShapeException">
        ///     The conditions, the choices, or the two groups against each other cannot be broadcast
        ///     (NumSharp's house form of NumPy's broadcast <c>ValueError</c>).
        /// </exception>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.select</c> (<c>numpy/lib/_function_base_impl.py</c>): fill the
        ///     result with the default, then <see cref="copyto"/> each choice onto it under its
        ///     condition mask in REVERSE order so the first matching condition takes precedence — the
        ///     same composition NumPy uses, so every masked write rides NumSharp's SIMD masked-cast
        ///     kernel. The contiguous, no-cast, full-size-array-choice case (the one NumPy leaves
        ///     memory-bound, since its (n+1) masked copytos each re-read and re-write the whole result)
        ///     is instead served by a fused single-pass IL kernel — see
        ///     <see cref="Backends.Kernels.DirectILKernelGenerator.GetSelectKernel"/>.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.select.html
        /// </remarks>
        public static NDArray select(NDArray[] condlist, object[] choicelist, object @default = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(condlist);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (condlist is null) throw new ArgumentNullException(nameof(condlist));
            if (choicelist is null) throw new ArgumentNullException(nameof(choicelist));

            // NumPy's numpy.select after the null checks: the statements are the array object's (NDArray.FunctionBase.cs)
            return NDArray.select(condlist, choicelist, @default);
        }
    }
}
