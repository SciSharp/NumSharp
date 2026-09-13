using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text.Json;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The C# half of the <c>evaluate.jsonl</c> tier (generator: <c>gen_oracle.gen_evaluate</c>).
    ///     Parses the corpus prefix grammar into an <see cref="NDExpr"/> over the reconstructed
    ///     operands and runs <c>np.evaluate</c>; NumPy's unfused node-by-node chain is the oracle.
    ///
    ///     Grammar (must stay 1:1 with the Python evaluator):
    ///     <code>
    ///       in&lt;k&gt;               operand k
    ///       li:&lt;int&gt; lu:&lt;uint64&gt;  weak Python int          lf:&lt;float&gt;  weak Python float
    ///       lb:0|1                 weak Python bool         lc:&lt;re&gt;;&lt;im&gt; weak Python complex
    ///       lh:&lt;float&gt;            STRONG float16 scalar
    ///       fn(arg,...)            node — see BuildNode
    ///     </code>
    ///     params["reduce"] = {kind, axis, keepdims} wraps the tree in a root reduction;
    ///     params["out"] = true names the LAST operand as the out= target (tuple result:
    ///     [returned, whole out base buffer], the out_where convention).
    /// </summary>
    public static partial class OpRegistry
    {
        internal static NDArray EvaluateFromCorpus(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            bool hasOut = p.TryGetValue("out", out var o) && o.ValueKind == JsonValueKind.True;
            var expr = BuildEvaluateTree(p, ops, hasOut ? ops.Length - 1 : ops.Length);
            return hasOut ? np.evaluate(expr, @out: ops[ops.Length - 1]) : np.evaluate(expr);
        }

        internal static NDArray[] EvaluateOutFromCorpus(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var target = ops[ops.Length - 1];
            var expr = BuildEvaluateTree(p, ops, ops.Length - 1);
            var returned = np.evaluate(expr, @out: target);
            return new[] { returned, BaseBuffer(target) };
        }

        private static NDExpr BuildEvaluateTree(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops, int nInputs)
        {
            var parser = new EvaluateExprParser(p["expr"].GetString(), ops, nInputs);
            NDExpr tree = parser.Parse();

            if (p.TryGetValue("reduce", out var red) && red.ValueKind == JsonValueKind.Object)
            {
                string kind = red.GetProperty("kind").GetString();
                int? axis = red.TryGetProperty("axis", out var ax) && ax.ValueKind == JsonValueKind.Number ? ax.GetInt32() : null;
                bool keepdims = red.TryGetProperty("keepdims", out var kd) && kd.ValueKind == JsonValueKind.True;
                tree = (kind, axis) switch
                {
                    ("sum", null) => NDExpr.Sum(tree),
                    ("prod", null) => NDExpr.Prod(tree),
                    ("min", null) => NDExpr.Min(tree),
                    ("max", null) => NDExpr.Max(tree),
                    ("mean", null) => NDExpr.Mean(tree),
                    ("sum", int a) => NDExpr.Sum(tree, a, keepdims),
                    ("prod", int a) => NDExpr.Prod(tree, a, keepdims),
                    ("min", int a) => NDExpr.Min(tree, a, keepdims),
                    ("max", int a) => NDExpr.Max(tree, a, keepdims),
                    ("mean", int a) => NDExpr.Mean(tree, a, keepdims),
                    _ => throw new NotSupportedException($"evaluate reduce kind '{kind}'"),
                };
            }

            return tree;
        }

        /// <summary>Recursive-descent parser over the prefix grammar; one instance per case.</summary>
        private sealed class EvaluateExprParser
        {
            private readonly List<string> _tokens;
            private readonly NDArray[] _ops;
            private readonly int _nInputs;
            private int _pos;

            public EvaluateExprParser(string expr, NDArray[] ops, int nInputs)
            {
                _tokens = Tokenize(expr);
                _ops = ops;
                _nInputs = nInputs;
            }

            private static List<string> Tokenize(string expr)
            {
                var toks = new List<string>();
                int i = 0;
                while (i < expr.Length)
                {
                    char c = expr[i];
                    if (char.IsWhiteSpace(c)) { i++; continue; }
                    if (c == '(' || c == ')' || c == ',') { toks.Add(c.ToString()); i++; continue; }
                    int start = i;
                    while (i < expr.Length && expr[i] != '(' && expr[i] != ')' && expr[i] != ',') i++;
                    toks.Add(expr.Substring(start, i - start));
                }
                return toks;
            }

            public NDExpr Parse()
            {
                var e = ParseNode();
                if (_pos != _tokens.Count)
                    throw new FormatException($"trailing tokens in evaluate expr at {_pos}");
                return e;
            }

            private string Next() => _tokens[_pos++];

            private NDExpr ParseNode()
            {
                string tok = Next();
                if (tok.StartsWith("in", StringComparison.Ordinal) && int.TryParse(tok.AsSpan(2), out int k))
                {
                    if (k >= _nInputs) throw new FormatException($"in{k} out of range ({_nInputs} inputs)");
                    return NDExpr.Arr(_ops[k]);
                }

                int colon = tok.IndexOf(':');
                if (colon > 0)
                    return Literal(tok.Substring(0, colon), tok.Substring(colon + 1));

                if (Next() != "(") throw new FormatException($"expected '(' after {tok}");
                var args = new List<NDExpr>();
                while (true)
                {
                    args.Add(ParseNode());
                    string sep = Next();
                    if (sep == ")") break;
                    if (sep != ",") throw new FormatException($"expected ',' or ')' in evaluate expr, got '{sep}'");
                }
                return BuildNode(tok, args);
            }

            private static NDExpr Literal(string kind, string val)
            {
                switch (kind)
                {
                    case "li":
                    {
                        long v = long.Parse(val, CultureInfo.InvariantCulture);
                        // A C# int literal and a long literal are BOTH weak Python ints; keep the
                        // narrower spelling where it fits so the corpus exercises both ctor paths.
                        return v >= int.MinValue && v <= int.MaxValue ? NDExpr.Const((int)v) : NDExpr.Const(v);
                    }
                    case "lu": return NDExpr.Const(ulong.Parse(val, CultureInfo.InvariantCulture));
                    case "lf": return NDExpr.Const(ParseDouble(val));
                    case "lb": return NDExpr.Const(val.Trim() == "1");
                    case "lc":
                    {
                        var parts = val.Split(';');
                        return NDExpr.Const(new Complex(ParseDouble(parts[0]), ParseDouble(parts[1])));
                    }
                    case "lh": return NDExpr.Const((Half)ParseDouble(val));
                    default: throw new FormatException($"unknown evaluate literal '{kind}:{val}'");
                }
            }

            private static double ParseDouble(string s)
            {
                s = s.Trim();
                return s switch
                {
                    "nan" => double.NaN,
                    "inf" => double.PositiveInfinity,
                    "-inf" => double.NegativeInfinity,
                    _ => double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture),
                };
            }

            private static NDExpr BuildNode(string name, List<NDExpr> a)
            {
                switch (name)
                {
                    // binary
                    case "add": return NDExpr.Add(a[0], a[1]);
                    case "sub": return NDExpr.Subtract(a[0], a[1]);
                    case "mul": return NDExpr.Multiply(a[0], a[1]);
                    case "div": return NDExpr.Divide(a[0], a[1]);
                    case "mod": return NDExpr.Mod(a[0], a[1]);
                    case "pow": return NDExpr.Power(a[0], a[1]);
                    case "floordiv": return NDExpr.FloorDivide(a[0], a[1]);
                    case "atan2": return NDExpr.ATan2(a[0], a[1]);
                    case "and": return NDExpr.BitwiseAnd(a[0], a[1]);
                    case "or": return NDExpr.BitwiseOr(a[0], a[1]);
                    case "xor": return NDExpr.BitwiseXor(a[0], a[1]);
                    case "min": return NDExpr.Min(a[0], a[1]);
                    case "max": return NDExpr.Max(a[0], a[1]);
                    case "eq": return NDExpr.Equal(a[0], a[1]);
                    case "ne": return NDExpr.NotEqual(a[0], a[1]);
                    case "lt": return NDExpr.Less(a[0], a[1]);
                    case "le": return NDExpr.LessEqual(a[0], a[1]);
                    case "gt": return NDExpr.Greater(a[0], a[1]);
                    case "ge": return NDExpr.GreaterEqual(a[0], a[1]);
                    case "where": return NDExpr.Where(a[0], a[1], a[2]);
                    // unary
                    case "neg": return NDExpr.Negate(a[0]);
                    case "abs": return NDExpr.Abs(a[0]);
                    case "sqrt": return NDExpr.Sqrt(a[0]);
                    case "square": return NDExpr.Square(a[0]);
                    case "recip": return NDExpr.Reciprocal(a[0]);
                    case "sign": return NDExpr.Sign(a[0]);
                    case "cbrt": return NDExpr.Cbrt(a[0]);
                    case "exp": return NDExpr.Exp(a[0]);
                    case "exp2": return NDExpr.Exp2(a[0]);
                    case "expm1": return NDExpr.Expm1(a[0]);
                    case "log": return NDExpr.Log(a[0]);
                    case "log2": return NDExpr.Log2(a[0]);
                    case "log10": return NDExpr.Log10(a[0]);
                    case "log1p": return NDExpr.Log1p(a[0]);
                    case "sin": return NDExpr.Sin(a[0]);
                    case "cos": return NDExpr.Cos(a[0]);
                    case "tan": return NDExpr.Tan(a[0]);
                    case "sinh": return NDExpr.Sinh(a[0]);
                    case "cosh": return NDExpr.Cosh(a[0]);
                    case "tanh": return NDExpr.Tanh(a[0]);
                    case "asin": return NDExpr.ASin(a[0]);
                    case "acos": return NDExpr.ACos(a[0]);
                    case "atan": return NDExpr.ATan(a[0]);
                    case "asinh": return NDExpr.Asinh(a[0]);
                    case "acosh": return NDExpr.Acosh(a[0]);
                    case "atanh": return NDExpr.Atanh(a[0]);
                    case "deg2rad": return NDExpr.Deg2Rad(a[0]);
                    case "rad2deg": return NDExpr.Rad2Deg(a[0]);
                    case "floor": return NDExpr.Floor(a[0]);
                    case "ceil": return NDExpr.Ceil(a[0]);
                    case "round": return NDExpr.Round(a[0]);
                    case "trunc": return NDExpr.Truncate(a[0]);
                    case "not": return NDExpr.BitwiseNot(a[0]);
                    case "lnot": return NDExpr.LogicalNot(a[0]);
                    case "isnan": return NDExpr.IsNaN(a[0]);
                    case "isfinite": return NDExpr.IsFinite(a[0]);
                    case "isinf": return NDExpr.IsInf(a[0]);
                    default: throw new NotSupportedException($"evaluate node '{name}' has no NDExpr mapping");
                }
            }
        }
    }
}
