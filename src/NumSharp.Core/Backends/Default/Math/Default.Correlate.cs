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
        /// <exception cref="ValueError">An unknown <paramref name="mode"/> (NumPy's two texts, see
        ///     <see cref="SlidingDot.ParseSlidingMode"/>), then an operand that is not one-dimensional, an empty operand, or a
        ///     text operand (<c>function not available for this data type</c>; NumPy's texts, in its order).</exception>
        /// <exception cref="UnicodeEncodeError">A <paramref name="mode"/> holding a lone surrogate.</exception>
        /// <exception cref="TypeError">A null <paramref name="mode"/> (NumPy's <c>None</c>), a <c>U</c> operand beside
        ///     <c>StringDType</c> (the 'safe' conversion's refusal), or an OBJECT product's first-product error.</exception>
        /// <exception cref="System.NotSupportedException">A <c>StringDType</c> operand whose OBJECT product NumPy
        ///     computes.</exception>
        /// <remarks>The materialized operands and the conjugate are temporaries: the [NDScoped] weaver returns their
        ///     buffers when the call ends, and only the returned array survives (the caller's operands are never tracked).
        ///     A swapped product is reversed in its own buffer, as NumPy's is.</remarks>
        [NDScoped]
        private NDArray CorrelateNumPy(NDArray a, NDArray v, string mode, bool correlate2)
        {
            // array_correlate(2) parses the mode during argument parsing, before it touches the arrays.
            var m = SlidingDot.ParseSlidingMode(mode);

            // PyArray_DTypeFromObject over both operands (NumPy 2.5): the common descriptor of a text pair, or none when a
            // StringDType operand has no common descriptor with the other — then NumPy correlates Python objects, whose own
            // checks and first product answer before every check below.
            DType textDescr = null;
            if (NDStringUfuncs.IsText(a) || NDStringUfuncs.IsText(v))
            {
                textDescr = NDStringUfuncs.ProductDescriptor(a, v);
                if (textDescr is null)
                    throw NDStringUfuncs.VStringSlidingAnswer(a, v, (int)m, correlate2 ? "correlate" : "convolve");
            }

            // PyArray_FromAny(op, typec, 1, 1, ...): exactly one dimension each, with its two texts, then the 'safe' cast
            // into the common descriptor (only a U operand beside StringDType fails it), the first operand first.
            ThrowIfNotOneDimensional(a);
            if (textDescr is not null)
                NDStringUfuncs.ThrowIfUnsafeProductCast(a, textDescr);
            ThrowIfNotOneDimensional(v);
            if (textDescr is not null)
                NDStringUfuncs.ThrowIfUnsafeProductCast(v, textDescr);

            // _pyarray_correlate's emptiness checks, on the operands before its swap, so "first" / "second" always name the
            // arguments as given (behind np.convolve they never fire: its Python body checked already).
            if (a.size == 0)
                throw new ValueError("first array argument cannot be empty");
            if (v.size == 0)
                throw new ValueError("second array argument cannot be empty");

            // _pyarray_correlate then allocates its result in the common descriptor and looks up its dot function, which
            // no string descriptor has (NumPy 2.4 failed the allocation instead: "data type must provide an itemsize").
            if (textDescr is not null)
                throw NDStringUfuncs.FunctionNotAvailable();

            // PyArray_ObjectType(op1) then (op2, typenum): the promotion of two arrays with dimensions, which is NumPy's
            // promote_types for every pair of the 15 storage dtypes (the frozen array-array table np._FindCommonType reads
            // agrees on all 225 pairs): the dtype layer's type-code form, the engine behind np.promote_types.
            NPTypeCode at = a.GetTypeCode, vt = v.GetTypeCode;
            NPTypeCode retType = DTypePromotion.PromoteTypes(at, vt);

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

            // PyArray_Correlate2 reverses the output of an inverted product in place (_pyarray_revert), so the result is
            // still the fresh owning array; PyArray_Correlate leaves it as computed.
            if (correlate2 && inverted)
                SlidingDot.RevertInPlace(result);

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
