using System;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Test element-wise whether the IEEE sign bit is set (np.signbit). A full ufunc:
        ///     <c>signbit(x, /, out=None, *, where=True, dtype=None)</c>, result always bool.
        /// </summary>
        /// <param name="a">Input array (any dtype except Complex).</param>
        /// <param name="dtype">
        ///     Validate-only (NumPy parity): signbit has bool-output loops ONLY, so a non-bool request
        ///     raises the no-loop <see cref="IncorrectTypeException"/>; <c>bool</c> is a legal no-op.
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored; any numeric dtype (bool casts same_kind to all,
        ///     True→1). The same instance is returned. If <c>null</c> (and <paramref name="where"/> null),
        ///     a freshly allocated boolean array is returned.
        /// </param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written; masked-off out slots keep prior contents.</param>
        /// <returns>A boolean array (or <paramref name="out"/>), True where the element's sign bit is set.</returns>
        /// <exception cref="TypeError">
        ///     The input dtype is Complex — signbit has no complex loop (NumPy raises the same
        ///     "not supported for the input types" TypeError; its signbit is ambiguous on complex).
        /// </exception>
        /// <exception cref="IncorrectTypeException"><paramref name="dtype"/> is a non-bool dtype (no matching loop).</exception>
        /// <remarks>
        ///     NumPy behaviour (numpy/_core, the <c>signbit</c> ufunc), all probed against 2.4.2:
        ///     <list type="bullet">
        ///       <item>Half/Single/Double: the raw IEEE sign bit — <c>-0.0</c> and a NEGATIVE NaN are
        ///         True, <c>+0.0</c>/<c>+inf</c>/positive NaN False (this is NOT <c>x &lt; 0</c>).</item>
        ///       <item>Signed integers: <c>x &lt; 0</c> (the two's-complement MSB IS the sign bit).</item>
        ///       <item>Unsigned integers / bool: always False.</item>
        ///       <item>Decimal (no NumPy analog): strictly-negative test; <c>-0.0m</c> → False.</item>
        ///     </list>
        ///     Single/Double take the dedicated SIMD predicate kernel (per-lane MSB extract + packed
        ///     bool store); signed Int32/Int64 too (same MSB extract). Every other dtype resolves through
        ///     the general unary scalar route — which already beats NumPy's own scalar signbit loops.
        /// </remarks>
        public override NDArray SignBit(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order (probed 2.4.2): where parse → loop resolution
            // (complex-input / bad-dtype) → out cast → shape.
            ValidateWhereMask(where);

            // signbit has no complex loop: NumPy raises the generic "not supported for the
            // input types" TypeError (its signbit ufunc simply has no c16 loop).
            if (a.typecode == NPTypeCode.Complex)
                throw new TypeError(
                    "ufunc 'signbit' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to the casting rule ''safe''");

            // dtype= is validate-only: bool loops only (dtype=bool no-op, else no-loop TypeError).
            ValidateBoolLoopDtype(dtype?.GetTypeCode(), "signbit");

            // Plain call: keep the typed NDArray<bool> instance (TensorEngine contract) via the cheap
            // AsGeneric wrap (NOT MakeGeneric — the engine result is a fresh array we own), exactly
            // like the sibling isnan/isinf predicates.
            if (@out is null && where is null)
            {
                using var result = ExecuteUnaryOp(a, UnaryOp.SignBit, NPTypeCode.Boolean);
                return result.AsGeneric<bool>();
            }

            // ufunc out=/where= path: the predicate body emits bool at the INPUT dtype; a non-bool out
            // engages the windowed bool→X flush. Returns the provided out (may be any numeric dtype).
            return ExecuteUnaryOp(a, UnaryOp.SignBit, NPTypeCode.Boolean, @out, where);
        }
    }
}
