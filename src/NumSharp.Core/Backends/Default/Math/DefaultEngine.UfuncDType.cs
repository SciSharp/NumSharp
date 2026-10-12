namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's refusal text for a <c>dtype=</c> / <c>signature=</c> argument that names more than a DType class
        ///     (<c>_get_dtype</c>, <c>numpy/_core/src/umath/ufunc_object.c</c>), verbatim.
        /// </summary>
        internal const string GeneralDTypeOnlyText =
            "The `dtype` and `signature` arguments to ufuncs only select the general DType and not details such as the byte " +
            "order or time unit. You can avoid this error by using the scalar types `np.float64` or the dtype string notation.";

        /// <summary>
        ///     NumPy's refusal text for a <c>dtype=</c> / <c>signature=</c> argument that is an INSTANCE of a non-legacy DType
        ///     class (a <c>StringDType</c> descriptor; <c>_get_dtype</c>), verbatim. NumPy accepts the class object there; NumSharp
        ///     has no way to pass a class to <c>dtype=</c>, so every <c>StringDType</c> request is refused this way.
        /// </summary>
        internal const string NewStyleDTypeInstanceText =
            "Cannot pass a new user DType instance to the `dtype` or `signature` arguments of ufuncs. Pass the DType class instead.";

        /// <summary>
        ///     The loop lane a ufunc's (or a reduce-based reduction's) <c>dtype=</c> argument selects, after NumPy's
        ///     <c>_get_dtype</c> check: the argument chooses only the DType CLASS, so a descriptor that is neither its
        ///     class's singleton nor equivalent to it — a byte-swapped <c>'&gt;f8'</c>, a sized string <c>'U6'</c> — is refused
        ///     (<c>'f8'</c>, <c>'=f8'</c>, <c>np.float64</c>, and the unsized <c>'U'</c> / <c>'S'</c> pass).
        /// </summary>
        /// <param name="dtype">The <c>dtype=</c> argument, or null.</param>
        /// <returns>The storage lane of the class (null when no dtype was given).</returns>
        /// <exception cref="TypeError">A descriptor with details beyond its class (<see cref="GeneralDTypeOnlyText"/>).</exception>
        /// <remarks>
        ///     NumPy runs this for every ufunc call and for the reductions built on <c>ufunc.reduce</c> (sum, prod, mean,
        ///     var, std, max, min and their nan forms), but NOT for the accumulations (cumsum, cumprod and their nan forms),
        ///     which take the descriptor as it is (probed 2.4.2) — so the cumulative engine members keep the plain
        ///     conversion.
        /// </remarks>
        internal static NPTypeCode? UfuncLoopCode(DType dtype)
        {
            if (dtype is null)
                return null;
            ThrowIfNotGeneralDType(dtype);
            return dtype.GetTypeCode();
        }

        /// <summary>
        ///     <see cref="UfuncLoopCode(DType)"/> for a ufunc entry point that knows its operands: after the class check it
        ///     runs NumPy's string-aware argument checks (<see cref="Strings.NDStringUfuncs.ThrowIfStringCall"/>), so a
        ///     string <c>dtype=</c> request never reaches a numeric kernel (the string lane has no element size) and a string
        ///     operand reports a read-only <c>out=</c> or a non-bool <c>where=</c> before its loop answer, as NumPy does.
        /// </summary>
        /// <param name="dtype">The <c>dtype=</c> argument, or null.</param>
        /// <param name="ufunc">NumPy's ufunc name (it is quoted by the refusal texts).</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input, or null for a unary ufunc.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>The storage lane of the requested class (null when no dtype was given, or when NumPy's string add promoter
        ///     ignores a string request for two strings of one class); never the string lane.</returns>
        /// <exception cref="TypeError">A descriptor with details beyond its class, or NumPy's refusal of a string request.</exception>
        /// <exception cref="ValueError">A string operand with a read-only <paramref name="out"/>.</exception>
        /// <exception cref="System.ArgumentException">A string operand with a non-bool <paramref name="where"/>, or a cast the
        ///     string loop refuses.</exception>
        /// <exception cref="StringComputationNotSupportedException">A string request NumPy computes (<c>add</c> formats
        ///     numbers as text and concatenates them).</exception>
        /// <remarks>
        ///     Two reference compares and two typecode reads when no operand holds text and no string class is requested;
        ///     the hot numeric path pays nothing else.
        /// </remarks>
        internal static NPTypeCode? UfuncLoopCode(DType dtype, string ufunc, NDArray x1, NDArray x2 = null,
            NDArray @out = null, NDArray where = null)
        {
            var code = UfuncLoopCode(dtype);
            // NumPy's string add promoter ignores a string request for two strings of one class: no loop request at all,
            // and the engine's string guard concatenates (its out= / where= checks run there, in NumPy's order).
            if (Strings.NDStringUfuncs.IsIgnoredStringRequest(ufunc, dtype, x1, x2))
                return null;
            Strings.NDStringUfuncs.ThrowIfStringCall(ufunc, dtype, x1, x2, @out, where);
            return code;
        }

        /// <summary>
        ///     NumPy's <c>_get_dtype</c> test alone (see <see cref="UfuncLoopCode(DType)"/>): a descriptor passes when it IS its class's
        ///     singleton or is equivalent to it (<c>PyArray_EquivTypes</c>).
        /// </summary>
        /// <param name="dtype">A requested loop descriptor.</param>
        /// <exception cref="TypeError">A byte-swapped, sized-string or otherwise detailed descriptor
        ///     (<see cref="GeneralDTypeOnlyText"/>), or an instance of a non-legacy class such as <c>StringDType</c>
        ///     (<see cref="NewStyleDTypeInstanceText"/>, checked first, as in NumPy).</exception>
        internal static void ThrowIfNotGeneralDType(DType dtype)
        {
            if (!dtype.Meta.IsLegacy)
                throw new TypeError(NewStyleDTypeInstanceText);
            var singleton = dtype.Meta.Singleton;
            if (ReferenceEquals(singleton, dtype))
                return;
            if (singleton is null || !DTypeCasting.EquivTypes(singleton, dtype))
                throw new TypeError(GeneralDTypeOnlyText);
        }
    }
}
