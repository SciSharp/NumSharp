using System.Collections.Generic;
using System.Text.Json;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The registry half of the matrix-operand tiers (<c>gen_matrix_operand_oracle.py</c>): the op keys whose
    ///     ordinary entry is a C# OPERATOR are replayed through the <c>np.*</c> function when an operand is a
    ///     <c>numpy.matrix</c>.
    /// </summary>
    /// <remarks>
    ///     NumPy's side of every such key is the function (<c>np.add(m, a)</c>), and it returns a matrix. In
    ///     NumSharp the operators of an <see cref="NDArray"/> reference keep the plain-array meaning for a matrix
    ///     object (they are bound at compile time), and only the public <c>np.*</c> functions follow the run-time
    ///     class. The ordinary entries (<c>ops[0] + ops[1]</c>) would therefore test the operator, which is not the
    ///     subject of these tiers.
    /// </remarks>
    public static partial class OpRegistry
    {
        /// <summary>Whether any operand of the case was rebuilt as a matrix (<c>"matrix": true</c> in its descriptor).</summary>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>True when at least one operand is an <see cref="NDMatrix"/>.</returns>
        private static bool HasMatrixOperand(NDArray[] ops)
        {
            foreach (NDArray o in ops)
                if (o is NDMatrix)
                    return true;
            return false;
        }

        /// <summary>
        ///     The <c>np.*</c> function form of an op key whose ordinary entry is an operator.
        /// </summary>
        /// <param name="op">The op key.</param>
        /// <param name="p">The case params (none of these keys reads one here: a case with a <c>dtype</c> param
        ///     already takes the function form in the ordinary entry).</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The function's result when the key has an operator entry.</param>
        /// <returns>True when <paramref name="op"/> is one of the operator keys and <paramref name="result"/> is set.</returns>
        private static bool TryApplyFunctionForm(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops, out NDArray result)
        {
            result = null;
            bool loopGiven = p.TryGetValue("dtype", out JsonElement loop) && loop.ValueKind != JsonValueKind.Null;
            if (loopGiven)
                return false;
            switch (op)
            {
                // mean_loops.jsonl records `dtype: null` for a call without dtype=, and its ordinary entry then
                // calls the ENGINE for the flat form (np.mean has no flat overload with dtype=). The engine is
                // below the public boundary, so the flat form goes through np.mean here.
                case "mean" when p.ContainsKey("dtype"):
                {
                    bool keepdims = p.TryGetValue("keepdims", out JsonElement k) && k.ValueKind == JsonValueKind.True;
                    result = p.TryGetValue("axis", out JsonElement ax) && ax.ValueKind == JsonValueKind.Number
                        ? np.mean(ops[0], ax.GetInt32(), keepdims)
                        : np.mean(ops[0], keepdims);
                    return true;
                }
                case "add": result = np.add(ops[0], ops[1]); return true;
                case "subtract": result = np.subtract(ops[0], ops[1]); return true;
                case "multiply": result = np.multiply(ops[0], ops[1]); return true;
                case "divide": result = np.divide(ops[0], ops[1]); return true;
                case "bitwise_and": result = np.bitwise_and(ops[0], ops[1]); return true;
                case "bitwise_or": result = np.bitwise_or(ops[0], ops[1]); return true;
                case "bitwise_xor": result = np.bitwise_xor(ops[0], ops[1]); return true;
                case "equal": result = np.equal(ops[0], ops[1]); return true;
                case "not_equal": result = np.not_equal(ops[0], ops[1]); return true;
                case "less": result = np.less(ops[0], ops[1]); return true;
                case "greater": result = np.greater(ops[0], ops[1]); return true;
                case "less_equal": result = np.less_equal(ops[0], ops[1]); return true;
                case "greater_equal": result = np.greater_equal(ops[0], ops[1]); return true;
                // The ordinary entry calls the typed np.argsort<T>, a NumSharp-only spelling; NumPy's function is
                // the untyped one.
                case "argsort": result = np.argsort(ops[0], p["axis"].GetInt32()); return true;
                default: return false;
            }
        }
    }
}
