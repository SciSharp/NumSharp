using NumSharp.Backends.Strings;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        public override NDArray Correlate(NDArray a, NDArray v, string mode = "valid")
            => CorrelateNumPy(a, v, mode, correlate2: false);

        /// <inheritdoc />
        public override NDArray Correlate2(NDArray a, NDArray v, string mode = "valid")
            => CorrelateNumPy(a, v, mode, correlate2: true);

        /// <summary>
        ///     The body of NumPy's two correlations, statement for statement: <c>array_correlate</c> /
        ///     <c>array_correlate2</c>'s mode argument, then <c>PyArray_Correlate</c> / <c>PyArray_Correlate2</c>
        ///     (numpy/_core/src/multiarray/multiarraymodule.c) with <c>_pyarray_correlate</c>.
        /// </summary>
        /// <param name="a">The first operand as NumPy's C function receives it (for np.convolve: the longer one, after its
        ///     Python swap).</param>
        /// <param name="v">The second operand as NumPy's C function receives it (for np.convolve: its kernel reversed).</param>
        /// <param name="mode">The mode name.</param>
        /// <param name="correlate2">True for <c>PyArray_Correlate2</c> (np.correlate): the second operand conjugated for a
        ///     complex dtype, and the result reversed when the operands were swapped. False for <c>PyArray_Correlate</c>
        ///     (np.convolve's <c>multiarray.correlate</c>), which does neither.</param>
        /// <returns>A fresh 1-D array of the common dtype.</returns>
        /// <exception cref="System.ArgumentException">An unknown <paramref name="mode"/> (NumPy's two texts).</exception>
        /// <exception cref="System.NullReferenceException">A null <paramref name="mode"/>.</exception>
        /// <exception cref="ValueError">An operand that is not one-dimensional, an empty operand, or a fixed-width string
        ///     operand (NumPy's texts, in its order).</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> operand's refusal or first-product error.</exception>
        /// <exception cref="System.NotSupportedException">A <c>StringDType</c> operand whose OBJECT product NumPy
        ///     computes.</exception>
        /// <remarks>The materialized operands, the conjugate, and the first result of a reversed product are temporaries:
        ///     the [NDScoped] weaver returns their buffers when the call ends, and only the returned array survives (the
        ///     caller's operands are never tracked).</remarks>
        [NDScoped]
        private NDArray CorrelateNumPy(NDArray a, NDArray v, string mode, bool correlate2)
        {
            // array_correlate(2) parses the mode during argument parsing, before it touches the arrays.
            var m = SlidingDot.ParseSlidingMode(mode);

            // PyArray_ObjectType over both operands: a StringDType operand has no legacy type number, so its answers (the
            // legacy-only refusal, or the OBJECT fallback's own checks and first product) come before every check below.
            if (NDStringUfuncs.IsVString(a) || NDStringUfuncs.IsVString(v))
                throw NDStringUfuncs.VStringSlidingAnswer(a, v, (int)m, correlate2 ? "correlate" : "convolve");

            // PyArray_FromAny(op, typec, 1, 1, ...): exactly one dimension each, with its two texts, the first operand first.
            ThrowIfNotOneDimensional(a);
            ThrowIfNotOneDimensional(v);

            // _pyarray_correlate's emptiness checks, on the operands before its swap, so "first" / "second" always name the
            // arguments as given (behind np.convolve they never fire: its Python body checked already).
            if (a.size == 0)
                throw new ValueError("first array argument cannot be empty");
            if (v.size == 0)
                throw new ValueError("second array argument cannot be empty");

            // _pyarray_correlate then allocates its result for the type number, which a string type cannot size.
            if (NDStringUfuncs.IsString(a) || NDStringUfuncs.IsString(v))
                throw NDStringUfuncs.NoItemsize();

            // PyArray_ObjectType(op1) then (op2, typenum): the promotion of two arrays with dimensions, which is NumPy's
            // promote_types for every pair of the 15 storage dtypes (the frozen array-array table np._FindCommonType reads
            // agrees on all 225 pairs; written in place, as the engine calls no np.* function).
            NPTypeCode at = a.GetTypeCode, vt = v.GetTypeCode;
            NPTypeCode retType = at == vt ? at : DTypePromotion.PromoteTypes(DType.From(at), DType.From(vt)).GetTypeCode();

            // PyArray_Correlate2 conjugates the (common-typed) second operand for a complex dtype, BEFORE the swap, so an
            // inverted correlation uses conj(v) as the data and the untouched a as the kernel.
            NDArray kernel = SlidingDot.MaterializeForSliding(v, retType);
            if (correlate2 && retType == NPTypeCode.Complex)
                kernel = SlidingDot.MaterializeForSliding(Conjugate(kernel), retType);

            NDArray data = SlidingDot.MaterializeForSliding(a, retType);

            // _pyarray_correlate makes the data the longer operand, remembering it inverted.
            bool inverted = false;
            if (data.size < kernel.size)
            {
                var tmp = data;
                data = kernel;
                kernel = tmp;
                inverted = true;
            }

            // NumPy's complex dot reaches cblas only when both operands reach it with a positive stride (DotOperandBlasable):
            // only a one-element operand passed through as it is can fail that. Behind PyArray_Correlate2 the second operand
            // of a complex product is always a fresh array (the conjugate, or the cast to complex128), so only `a` counts;
            // behind PyArray_Correlate both do (np.convolve's reversed one-element kernel has a negative stride).
            bool complexDotViaBlas = SlidingDot.DotOperandBlasable(a, retType, reversed: false)
                                     && (correlate2 || SlidingDot.DotOperandBlasable(v, retType, reversed: false));
            NDArray result = SlidingDot.SlidingCorrelate(data, kernel, retType, m, complexDotViaBlas);

            // PyArray_Correlate2 reverses the output of an inverted product (ret = ret[::-1], _pyarray_revert);
            // PyArray_Correlate leaves it as computed.
            if (correlate2 && inverted)
                result = result["::-1"].copy();

            return result;
        }

        /// <summary>
        ///     NumPy's <c>PyArray_FromAny(op, typec, 1, 1, ...)</c> depth check: an operand of a correlation must be exactly
        ///     one-dimensional.
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
