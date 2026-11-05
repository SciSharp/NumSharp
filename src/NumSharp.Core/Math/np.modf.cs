using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the fractional and integral parts of an array, element-wise — NumPy's two-output
        ///     <c>modf</c> ufunc. Both parts carry the sign of the input value.
        /// </summary>
        /// <param name="x">Input array. Integer/bool/char promote to the narrowest float that fits the
        ///     width (bool/int8/uint8 to float16, int16/uint16/char to float32, int32+ to float64);
        ///     float16/float32/float64 are preserved; Complex has no loop (TypeError); Decimal is a
        ///     NumSharp extension.</param>
        /// <param name="outFrac">Optional provided output for the fractional part (NumPy's <c>out[0]</c>,
        ///     i.e. the first positional out — <c>np.modf(x, out1)</c>). Must be same_kind-castable from
        ///     the loop dtype; returned as-is (reference identity). Null auto-allocates.</param>
        /// <param name="outIntegral">Optional provided output for the integral part (NumPy's <c>out[1]</c>).
        ///     Null auto-allocates.</param>
        /// <param name="where">Optional boolean write mask; masked-off slots of BOTH provided outputs keep
        ///     their prior contents.</param>
        /// <param name="dtype">Optional loop dtype override; only the float loops (Half/Single/Double/Decimal)
        ///     are valid — a non-float request raises the no-loop TypeError.</param>
        /// <returns>The (fractional, integral) pair. This is a scalar-shaped 0-d pair if <paramref name="x"/>
        ///     is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.modf.html
        ///
        ///   modf(1.5) = (0.5, 1.0);  modf(-2.7) = (-0.7, -2.0);  modf(inf) = (+0.0, +inf);
        ///   modf(-inf) = (-0.0, -inf);  modf(nan) = (nan, nan);  modf(-0.0) = (-0.0, -0.0).
        ///
        /// Fresh (auto-allocated) outputs take NumPy's layout: modf has two outputs, so NumPy's iterator always
        /// allocates them, in the input's memory order (an F-contiguous input yields F-contiguous outputs, a
        /// transposed one keeps its axis order; the mask votes too).</remarks>
        [NDScoped]
        public static (NDArray Fractional, NDArray Integral) modf(
            NDArray x, NDArray outFrac = null, NDArray outIntegral = null, NDArray where = null, DType dtype = null)
            // Fresh outputs take NumPy's K order (TwoOutputsInNumPyLayout, shared with np.frexp); a provided out owns
            // its own layout so it is never relabelled (its reference identity is returned untouched). Mirrors frexp's
            // out-vs-fresh split.
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand: both outputs come back as matrices, as from NumPy
                ? NDMatrixBoundary.WrappedPair((x, outFrac, outIntegral, where, dtype),
                    static s => modf(s.x, s.outFrac, s.outIntegral, s.where, s.dtype), outFrac, outIntegral)
                : (outFrac is null && outIntegral is null)
                ? TwoOutputsInNumPyLayout(x, where, dtype,
                    static (xv, wv, d) => xv.TensorEngine.ModF(xv, d, null, null, wv))
                : x.TensorEngine.ModF(x, dtype, outFrac, outIntegral, where);

        /// <summary>
        /// Computes the two fresh outputs of a one-input, two-output ufunc (np.modf, np.frexp) in the layout NumPy
        /// allocates them in. A two-output ufunc never takes NumPy's trivial loop, so its iterator allocates both outputs
        /// in K order — the input's memory order, with a <c>where=</c> mask voting
        /// (<see cref="Shape.UfuncOutputShape"/> with the trivial loop refused): F for an F input, the axis order of a
        /// transposed one, F for a flipped F one. The engines compute C order, so a non-C layout runs them over the input
        /// (and the mask) viewed in that order and relabels both C results.
        /// </summary>
        /// <typeparam name="TState">What the computation needs besides the operands (the <c>dtype=</c> request).</typeparam>
        /// <param name="x">The input.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <param name="state">The computation's state.</param>
        /// <param name="compute">The engine call: given the input and the mask (views, when the layout is not C), the two
        ///     fresh outputs, C-ordered.</param>
        /// <returns>The two outputs in NumPy's layout.</returns>
        /// <exception cref="Exception">Whatever <paramref name="compute"/> raises (the views are disposed first).</exception>
        /// <remarks>A mask that would grow the result's shape leaves the decision to the engine (its C layout and its
        ///     broadcast errors): only a mask that broadcasts into the input's own dims votes here.</remarks>
        private static (NDArray, NDArray) TwoOutputsInNumPyLayout<TState>(NDArray x, NDArray where, TState state,
            Func<NDArray, NDArray, TState, (NDArray, NDArray)> compute)
        {
            if (where is not null && !BroadcastsInto(where.Shape, x.Shape))
                return compute(x, where, state);
            var votes = where is null ? new[] { x.Shape } : new[] { x.Shape, where.Shape };
            if (!DefaultEngine.TryNonCResultLayout(votes, x.Shape.dimensions, trivialLoopEligible: false, out var layout))
                return compute(x, where, state);

            int nd = layout.NDim;
            int[] perm = layout.DenseAxisOrder();
            NDArray xv = null, wv = null;
            try
            {
                xv = DefaultEngine.OperandInLayoutOrder(x, nd, perm);
                wv = where is null ? null : DefaultEngine.OperandInLayoutOrder(where, nd, perm);
                var (first, second) = compute(xv, wv, state);
                return (NDArray.AdoptLayout(first, layout), NDArray.AdoptLayout(second, layout));
            }
            finally
            {
                DefaultEngine.DisposeLayoutView(xv, x);
                if (where is not null)
                    DefaultEngine.DisposeLayoutView(wv, where);
            }
        }

        /// <summary>Whether <paramref name="small"/> broadcasts into <paramref name="big"/>'s dims without growing them.</summary>
        /// <param name="small">The operand that broadcasts (a mask).</param>
        /// <param name="big">The shape it must fit.</param>
        /// <returns>True when every right-aligned extent of <paramref name="small"/> is 1 or equals <paramref name="big"/>'s.</returns>
        private static bool BroadcastsInto(in Shape small, in Shape big)
        {
            var sd = small.dimensions;
            var bd = big.dimensions;
            if (sd.Length > bd.Length)
                return false;
            int shift = bd.Length - sd.Length;
            for (int i = 0; i < sd.Length; i++)
            {
                if (sd[i] != 1 && sd[i] != bd[i + shift])
                    return false;
            }
            return true;
        }
    }
}
