using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section <c>strsets</c> (<c>gen_strsets</c>, strings phase 2, P2.2 milestone C):
    ///     <c>np.quantile</c> / <c>np.percentile</c> with every q form, axis form, method, keepdims and <c>out=</c>. The
    ///     section's <c>np.unique</c>, Array-API unique, <c>np.isin</c>, set-routine, <c>np.lexsort</c>,
    ///     <c>np.sort_complex</c> and histogram cases replay through the ops of sections strshape and strsurface
    ///     (<c>unique</c>, <c>unique_api</c>, <c>isin</c>, <c>setop</c>, <c>lexsort</c>, <c>sort_complex</c>, <c>hist</c>),
    ///     which take the same parameters.
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>Replays one section-strsets op, or reports that <paramref name="op"/> is not one of them.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters: for <c>quant_kw</c> the function <c>name</c> (<c>quantile</c> /
        ///     <c>percentile</c>), <c>qkind</c> (<c>f</c> a Python float, <c>fl</c> a list of floats, <c>i</c> a Python int,
        ///     <c>arr</c> an ndarray operand named by <c>q_op</c>), <c>q</c>, <c>axis</c> (null, an int or a list of ints),
        ///     <c>method</c>, <c>keepdims</c> and <c>out_op</c> (the out operand, when present).</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result (the out array itself for an <c>out=</c> call, whose view and base the
        ///     replay compares).</param>
        /// <returns>True when <paramref name="op"/> is a section-strsets op (its result is in <paramref name="result"/>).</returns>
        /// <exception cref="InvalidOperationException">An unknown function name or q kind.</exception>
        private static bool TryInvokeSets(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            if (op != "quant_kw")
            {
                result = null;
                return false;
            }
            result = QuantileKeywords(p, ops);
            return true;
        }

        /// <summary>
        ///     One <c>np.quantile</c> / <c>np.percentile</c> call through the overload NumPy's call ports to: a Python float is
        ///     a <see cref="double"/>, a list a <see cref="double"/>[], a Python int a <see cref="long"/> (quantile; percentile
        ///     has no integer overload and NumPy divides the q anyway), an array q an <see cref="NDArray"/>; one axis an
        ///     <see cref="int"/>, a tuple an <see cref="int"/>[].
        /// </summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The operands: the data, then the array q and the out array where the parameters name them.</param>
        /// <returns>The call's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown function name or q kind.</exception>
        private static object QuantileKeywords(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            NDArray a = ops[0];
            string name = Str(p, "name");
            string method = Str(p, "method");
            bool keepdims = p["keepdims"].GetBoolean();
            NDArray @out = NullableInt(p, "out_op") is int oi ? ops[oi] : null;
            JsonElement axisElement = p["axis"];
            int[] axes = axisElement.ValueKind == JsonValueKind.Array ? axisElement.EnumerateArray().Select(e => e.GetInt32()).ToArray() : null;
            int? axis = axisElement.ValueKind == JsonValueKind.Number ? axisElement.GetInt32() : null;
            bool percentile = name switch
            {
                "percentile" => true,
                "quantile" => false,
                _ => throw new InvalidOperationException($"the string oracle has no quantile '{name}'"),
            };
            switch (Str(p, "qkind"))
            {
                case "f":
                {
                    double q = p["q"].GetDouble();
                    if (axes is not null)
                        return percentile ? np.percentile(a, q, axes, @out, method: method, keepdims: keepdims)
                                          : np.quantile(a, q, axes, @out, method: method, keepdims: keepdims);
                    return percentile ? np.percentile(a, q, axis, @out, method: method, keepdims: keepdims)
                                      : np.quantile(a, q, axis, @out, method: method, keepdims: keepdims);
                }
                case "fl":
                {
                    double[] q = p["q"].EnumerateArray().Select(e => e.GetDouble()).ToArray();
                    if (axes is not null)
                        return percentile ? np.percentile(a, q, axes, @out, method: method, keepdims: keepdims)
                                          : np.quantile(a, q, axes, @out, method: method, keepdims: keepdims);
                    return percentile ? np.percentile(a, q, axis, @out, method: method, keepdims: keepdims)
                                      : np.quantile(a, q, axis, @out, method: method, keepdims: keepdims);
                }
                case "i":
                {
                    long q = p["q"].GetInt64();
                    // percentile has no integer overload: C# binds a Python int to its double one, the value NumPy's
                    // true_divide(q, 100) reads.
                    if (percentile)
                        return axes is not null ? np.percentile(a, q, axes, @out, method: method, keepdims: keepdims)
                                                : np.percentile(a, q, axis, @out, method: method, keepdims: keepdims);
                    return axes is not null ? np.quantile(a, q, axes, @out, method: method, keepdims: keepdims)
                                            : np.quantile(a, q, axis, @out, method: method, keepdims: keepdims);
                }
                case "arr":
                {
                    NDArray q = ops[Int(p, "q_op")];
                    if (axes is not null)
                        return percentile ? np.percentile(a, q, axes, @out, method: method, keepdims: keepdims)
                                          : np.quantile(a, q, axes, @out, method: method, keepdims: keepdims);
                    return percentile ? np.percentile(a, q, axis, @out, method: method, keepdims: keepdims)
                                      : np.quantile(a, q, axis, @out, method: method, keepdims: keepdims);
                }
            }
            throw new InvalidOperationException($"the string oracle has no q kind '{Str(p, "qkind")}'");
        }
    }
}
