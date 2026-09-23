using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    /// <summary>
    /// <c>np.gcd</c> / <c>np.lcm</c> dispatch — the number-theoretic binary ufuncs.
    ///
    /// NumPy loop resolution (probed 2.4.2): gcd/lcm have loops for the INTEGER dtypes ONLY
    /// (<c>bb->b … QQ->Q</c>) via <c>PyUFunc_SimpleUniformOperationTypeResolver</c> — every operand and
    /// the output share ONE promoted integer dtype. Unlike the bitwise family there is NO bool loop
    /// (<c>gcd(bool, bool)</c> raises), and a <c>uint64</c>+signed pair promotes to <c>float64</c> under
    /// NEP50, which has no loop and raises too. Anything non-integer (bool/half/single/double/decimal/
    /// complex, or a no-loop promotion) raises NumPy's <c>UFuncTypeError</c>: <c>ufunc '{name}' did not
    /// contain a loop with signature matching types (…, …) -&gt; None</c>. Validation order pinned by probes:
    ///   ① where must be bool → ② loop resolution (no-loop TypeError) →
    ///   ③ out same_kind cast → ④ broadcast/shape.
    /// ①② run here; ③④ (and the dtype= input-cast) run inside the shared ufunc Into-path / ExecuteBinaryOp.
    /// The kernels themselves have NO SIMD path (a data-dependent Euclidean loop per element, which NumPy
    /// does not vectorize either); they route through the scalar per-element <c>NDGcdLcm</c> helpers.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Execute <c>np.gcd</c> — element-wise greatest common divisor of the operands' magnitudes.
        /// </summary>
        /// <param name="lhs">First input array.</param>
        /// <param name="rhs">Second input array.</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc <c>dtype=</c>); must name an integer loop, else the no-loop error.</param>
        /// <param name="out">Output location (NumPy ufunc <c>out=</c>); the same instance is returned when supplied.</param>
        /// <param name="where">Boolean mask (NumPy ufunc <c>where=</c>); masked-off elements keep their prior value.</param>
        /// <returns>The element-wise gcd in the promoted integer dtype.</returns>
        /// <exception cref="TypeError">The inputs (or an explicit <paramref name="dtype"/>) name no integer gcd loop — bool/float/complex/decimal, or a uint64+signed promotion to float64.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not boolean, an input cannot be same_kind-cast to an explicit <paramref name="dtype"/>, or <paramref name="out"/> cannot hold the result / does not broadcast.</exception>
        public override NDArray Gcd(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            ValidateWhereMask(where);                                    // ① where must be bool
            ValidateGcdLcmLoop(lhs, rhs, typeCode, "gcd");              // ② integer-only loop resolution
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Gcd, @out, where, typeCode);
        }

        /// <summary>
        /// Execute <c>np.lcm</c> — element-wise lowest common multiple of the operands' magnitudes
        /// (0 when either is 0; the product wraps the dtype on overflow, matching NumPy).
        /// </summary>
        /// <param name="lhs">First input array.</param>
        /// <param name="rhs">Second input array.</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc <c>dtype=</c>); must name an integer loop, else the no-loop error.</param>
        /// <param name="out">Output location (NumPy ufunc <c>out=</c>); the same instance is returned when supplied.</param>
        /// <param name="where">Boolean mask (NumPy ufunc <c>where=</c>); masked-off elements keep their prior value.</param>
        /// <returns>The element-wise lcm in the promoted integer dtype.</returns>
        /// <exception cref="TypeError">The inputs (or an explicit <paramref name="dtype"/>) name no integer lcm loop — bool/float/complex/decimal, or a uint64+signed promotion to float64.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not boolean, an input cannot be same_kind-cast to an explicit <paramref name="dtype"/>, or <paramref name="out"/> cannot hold the result / does not broadcast.</exception>
        public override NDArray Lcm(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            ValidateWhereMask(where);
            ValidateGcdLcmLoop(lhs, rhs, typeCode, "lcm");
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Lcm, @out, where, typeCode);
        }

        /// <summary>
        /// NumPy loop resolution for the gcd/lcm family (probed 2.4.2). When <paramref name="dtype"/> is
        /// supplied it selects the loop directly, so ONLY the dtype must name an integer loop (the input →
        /// dtype same_kind cast is validated later, inside <c>ExecuteBinaryOp</c>). When it is absent the
        /// loop is resolved from the inputs' promoted common type. Either way, a non-integer resolution
        /// (bool, any float/complex/decimal, or a uint64+signed pair promoting to float64) raises NumPy's
        /// no-loop <see cref="TypeError"/> — text verbatim, with the input operand dtype CLASS names and a
        /// <c>-&gt; None</c> tail (inputs path) or a <c>-&gt; {dtype}DType</c> tail (dtype= path). Char rides
        /// along as the NumSharp unsigned-16-bit integer extension (<see cref="NDExprTypeRules.IsIntegerKind"/>
        /// treats it as an integer kind).
        /// </summary>
        /// <param name="lhs">First input array — its dtype class names the first slot of the error signature.</param>
        /// <param name="rhs">Second input array — its dtype class names the second slot of the error signature.</param>
        /// <param name="typeCode">The explicit <c>dtype=</c> loop, or null to resolve from the inputs.</param>
        /// <param name="ufuncName">"gcd" or "lcm" — quoted verbatim in the error text.</param>
        /// <exception cref="TypeError">No integer loop matches.</exception>
        private static void ValidateGcdLcmLoop(NDArray lhs, NDArray rhs, NPTypeCode? typeCode, string ufuncName)
        {
            if (typeCode.HasValue)
            {
                // dtype= names the loop directly. An integer dtype is fine here; the input→dtype cast is
                // validated in ExecuteBinaryOp (same_kind), producing NumPy's "Cannot cast … input N" text.
                if (NDExprTypeRules.IsIntegerKind(typeCode.Value))
                    return;
                // A non-integer dtype= has no loop: "(lhs, rhs) -> {dtype}DType".
                throw new TypeError(GcdLcmNoLoopMessage(ufuncName, lhs.GetTypeCode, rhs.GetTypeCode, typeCode.Value));
            }

            // No dtype=: resolve the loop from the inputs' NEP50 common type (uint64+signed -> float64,
            // bool+bool -> bool, any float/complex/decimal -> itself — all non-integer -> no loop).
            var common = np._FindCommonType(lhs, rhs);
            if (NDExprTypeRules.IsIntegerKind(common))
                return;
            throw new TypeError(GcdLcmNoLoopMessage(ufuncName, lhs.GetTypeCode, rhs.GetTypeCode, null));
        }

        /// <summary>
        /// Build NumPy's verbatim no-loop message for gcd/lcm (the
        /// <c>PyUFunc_SimpleUniformOperationTypeResolver</c> form): the two INPUT operand dtype classes,
        /// then <c>-&gt; None</c> when the loop was resolved from the inputs, or <c>-&gt; {out}DType</c>
        /// when an explicit <c>dtype=</c> named the (missing) output loop. Reuses
        /// <see cref="NumPyDTypeClassName"/> (the shared <c>numpy.dtypes.*DType</c> speller).
        /// </summary>
        /// <param name="ufuncName">"gcd" or "lcm".</param>
        /// <param name="lhsType">First input's dtype.</param>
        /// <param name="rhsType">Second input's dtype.</param>
        /// <param name="outType">The requested output loop dtype, or null for the inputs-resolution form.</param>
        /// <returns>The exception message, byte-identical to NumPy 2.4.2.</returns>
        private static string GcdLcmNoLoopMessage(string ufuncName, NPTypeCode lhsType, NPTypeCode rhsType, NPTypeCode? outType)
        {
            string outRepr = outType.HasValue
                ? $"<class 'numpy.dtypes.{NumPyDTypeClassName(outType.Value)}'>"
                : "None";
            return $"ufunc '{ufuncName}' did not contain a loop with signature matching types " +
                   $"(<class 'numpy.dtypes.{NumPyDTypeClassName(lhsType)}'>, " +
                   $"<class 'numpy.dtypes.{NumPyDTypeClassName(rhsType)}'>) -> {outRepr}";
        }
    }
}
