using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's op for sections <c>strfn</c> / <c>tfn</c> (<c>gen_strfn</c> / <c>gen_tfn</c>, strings phase 2,
    ///     P2.5): one <c>numpy.strings</c> function per case — <c>str_len</c>, the find family, <c>startswith</c> /
    ///     <c>endswith</c>, the strips, the pads, <c>zfill</c>, <c>expandtabs</c>, <c>replace</c>, <c>partition</c> /
    ///     <c>rpartition</c>, <c>slice</c> and the comparison names — called with the generator's positional arguments.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Each argument is an operand (<c>{"op": i}</c>), a NumPy <c>np.str_</c> (<c>{"npstr": …}</c>: a 0-d text that HAS a
    ///     dtype, <c>NDArray.Scalar(string)</c>), or a Python value (<c>{"py": …}</c>) passed the way a ported NumPy line
    ///     passes it: a str as a C# <see cref="string"/> (NumPy's dtype-less Python str), an int as a C# <see cref="long"/>
    ///     literal (a weak Python int), a float as a <see cref="double"/>, bytes as <c>NDArray.Scalar(byte[])</c> (the
    ///     generator calls NumPy with <c>np.bytes_</c>, a dtype-carrying scalar), None as null — which the generator passes
    ///     only where NumPy's None means what NumSharp's null default means (an <c>end</c>, a strip set, a slice bound).
    ///     </para>
    ///     <para>
    ///     <c>slice</c>'s arity is NumPy's: one argument after the text is the STOP, so the overload follows the argument
    ///     count. A function of three outputs returns them as an <c>object[]</c>, which the comparator reads as NumPy's
    ///     tuple (arrays) or its items (the Python strs of 0-d <c>StringDType</c> parts).
    ///     </para>
    /// </remarks>
    internal static partial class StringOracleOps
    {
        /// <summary>Replays one section strfn / tfn op, or reports that <paramref name="op"/> is not one.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters: <c>name</c>, <c>args</c>, and an optional <c>kw</c>.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result.</param>
        /// <returns>True when <paramref name="op"/> is <c>str_fn</c>.</returns>
        /// <exception cref="InvalidOperationException">A function name or arity the op does not know (a generator / table
        ///     mismatch).</exception>
        private static bool TryInvokeMethods(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            if (op != "str_fn")
            {
                result = null;
                return false;
            }
            result = StringsFunction(p, ops);
            return true;
        }

        /// <summary>One <c>np.strings.&lt;name&gt;(*args, **kw)</c> call with the generator's arguments.</summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The result (an <c>object[]</c> for the three parts of a partition).</returns>
        /// <exception cref="InvalidOperationException">An unknown function name or arity.</exception>
        private static object StringsFunction(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            string name = Str(p, "name");
            var args = p["args"].EnumerateArray().Select(e => StringsArgument(e, ops)).ToArray();
            NDArray A(int i) => i < args.Length ? args[i] : null;
            switch (name)
            {
                case "str_len":
                {
                    NDArray @out = null, where = null;
                    DType dtype = null;
                    if (p.TryGetValue("kw", out var kw))
                    {
                        if (kw.TryGetProperty("out", out var o))
                            @out = ops[o.GetInt32()];
                        if (kw.TryGetProperty("where", out var w))
                            where = ops[w.GetInt32()];
                        if (kw.TryGetProperty("dtype", out var d))
                            dtype = np.dtype(d.GetString());
                    }
                    return np.strings.str_len(A(0), @out, where, dtype);
                }
                case "find": return np.strings.find(A(0), A(1), A(2), A(3));
                case "rfind": return np.strings.rfind(A(0), A(1), A(2), A(3));
                case "index": return np.strings.index(A(0), A(1), A(2), A(3));
                case "rindex": return np.strings.rindex(A(0), A(1), A(2), A(3));
                case "count": return np.strings.count(A(0), A(1), A(2), A(3));
                case "startswith": return np.strings.startswith(A(0), A(1), A(2), A(3));
                case "endswith": return np.strings.endswith(A(0), A(1), A(2), A(3));
                case "strip": return np.strings.strip(A(0), A(1));
                case "lstrip": return np.strings.lstrip(A(0), A(1));
                case "rstrip": return np.strings.rstrip(A(0), A(1));
                case "center": return np.strings.center(A(0), A(1), A(2));
                case "ljust": return np.strings.ljust(A(0), A(1), A(2));
                case "rjust": return np.strings.rjust(A(0), A(1), A(2));
                case "zfill": return np.strings.zfill(A(0), A(1));
                case "expandtabs": return np.strings.expandtabs(A(0), A(1));
                case "replace": return np.strings.replace(A(0), A(1), A(2), A(3));
                case "partition":
                {
                    var (before, separator, after) = np.strings.partition(A(0), A(1));
                    return new object[] { before, separator, after };
                }
                case "rpartition":
                {
                    var (before, separator, after) = np.strings.rpartition(A(0), A(1));
                    return new object[] { before, separator, after };
                }
                case "slice":
                    // NumPy's slice(a, start=None, stop=np._NoValue, step=None, /): one argument after the text is the stop.
                    return args.Length switch
                    {
                        1 => np.strings.slice(A(0)),
                        2 => np.strings.slice(A(0), A(1)),
                        3 => np.strings.slice(A(0), A(1), A(2)),
                        4 => np.strings.slice(A(0), A(1), A(2), A(3)),
                        _ => throw new InvalidOperationException($"np.strings.slice takes 1 to 4 arguments, the case has {args.Length}"),
                    };
                case "equal": return np.strings.equal(A(0), A(1));
                case "not_equal": return np.strings.not_equal(A(0), A(1));
                case "less": return np.strings.less(A(0), A(1));
                case "less_equal": return np.strings.less_equal(A(0), A(1));
                case "greater": return np.strings.greater(A(0), A(1));
                case "greater_equal": return np.strings.greater_equal(A(0), A(1));
                default:
                    throw new InvalidOperationException($"the string oracle has no numpy.strings function '{name}'");
            }
        }

        /// <summary>
        ///     One argument of a numpy.strings case as a ported line passes it (see the class remarks): an operand, a
        ///     dtype-carrying 0-d text, or a Python value through NumSharp's literal conversions.
        /// </summary>
        /// <param name="e">The encoded argument.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The argument (null for None).</returns>
        /// <exception cref="FormatException">An unknown argument encoding.</exception>
        private static NDArray StringsArgument(JsonElement e, NDArray[] ops)
        {
            if (e.TryGetProperty("op", out var op))
                return ops[op.GetInt32()];
            if (e.TryGetProperty("npstr", out var npstr))
                return NDArray.Scalar((string)StringOracleCorpus.Decode(npstr));
            if (!e.TryGetProperty("py", out var py))
                throw new FormatException($"unknown numpy.strings argument {e}");
            return StringOracleCorpus.Decode(py) switch
            {
                null => null,
                // Python bytes reach NumPy as np.bytes_ (a 0-d S scalar with a dtype): NumSharp's NDArray.Scalar(byte[]).
                byte[] bytes => NDArray.Scalar(bytes),
                // A Python int past int64 and within uint64 is a C# ulong literal (np.asanyarray makes it uint64).
                System.Numerics.BigInteger big when big >= 0 && big <= ulong.MaxValue => NDArray.FromPythonOperand((ulong)big),
                // A str, an int, a float, a bool, a complex: the 0-d array its C# literal's implicit conversion makes.
                var value => NDArray.FromPythonOperand(value),
            };
        }
    }
}
