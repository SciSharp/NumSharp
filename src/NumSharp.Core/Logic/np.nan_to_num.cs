using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Replace NaN with zero and infinity with large finite numbers (default behaviour) or with the
        ///     values supplied via <paramref name="nan"/>, <paramref name="posinf"/> and/or
        ///     <paramref name="neginf"/>.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <param name="copy">
        ///     Whether to create a copy of <paramref name="x"/> (<c>true</c>, the default) or replace values
        ///     in place (<c>false</c>). With <c>false</c> the returned array may be <paramref name="x"/>
        ///     itself (writes go through to shared memory).
        /// </param>
        /// <param name="nan">
        ///     Value(s) used to fill NaN. A real scalar (int/float/bool/char/float16) or an array_like
        ///     (<see cref="NDArray"/> or a C# array) broadcast position-wise. <c>null</c> (default) fills NaN with
        ///     <c>0.0</c>. Cast same_kind into the real dtype, as NumPy's <c>copyto</c> casts it: text and complex
        ///     fills are refused.
        /// </param>
        /// <param name="posinf">
        ///     Value(s) used to fill +Inf. <c>null</c> (default) fills with the largest finite value
        ///     representable by <paramref name="x"/>'s (real) dtype.
        /// </param>
        /// <param name="neginf">
        ///     Value(s) used to fill -Inf. <c>null</c> (default) fills with the most negative finite value
        ///     representable by <paramref name="x"/>'s (real) dtype.
        /// </param>
        /// <returns>
        ///     <paramref name="x"/> with the non-finite values replaced. If <c>copy=false</c> this may be
        ///     <paramref name="x"/> itself. Integer/boolean/decimal inputs are returned unchanged (a copy when
        ///     <c>copy=true</c>) — they are not inexact and hold no NaN/Inf.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
        ///     <para>
        ///     Port of NumPy's <c>numpy.nan_to_num</c> (<c>numpy/lib/_type_check_impl.py</c>). For a complex
        ///     input the replacement is applied to the real and imaginary components separately, both using the
        ///     float64 <c>finfo</c> limits (matching NumPy). The common contiguous scalar-fill case runs a
        ///     single fused whole-array kernel (<see cref="DirectILKernelGenerator.NanToNum"/>) — one read + one
        ///     write, no intermediate allocation — instead of NumPy's isnan/isposinf/isneginf + three
        ///     <c>copyto(where=)</c> passes. Array-valued fills and non-contiguous in-place targets take the
        ///     faithful <c>copyto</c> composition.
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="InvalidCastException">On an inexact input, a text or complex fill (or a complex / text array
        ///     fill): NumPy's <c>copyto</c> refusal (<c>Cannot cast scalar from dtype('&lt;U1') to dtype('float64')
        ///     according to the rule 'same_kind'</c>, NumPy's TypeError text), raised at the statement that meets the fill;
        ///     or a scalar fill of a type that does not convert to a number.</exception>
        /// <exception cref="ValueError">An array fill that does not broadcast into <paramref name="x"/>.</exception>
        public static NDArray nan_to_num(NDArray x, bool copy = true, object nan = null, object posinf = null, object neginf = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(x))
                return NDMatrixBoundary.Wrapped((x, copy, nan, posinf, neginf), static s => nan_to_num(s.x, s.copy, s.nan, s.posinf, s.neginf));
            return np.EngineOf(x).NanToNum(x, copy, nan, posinf, neginf);
        }
    }
}
