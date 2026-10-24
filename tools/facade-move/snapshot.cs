#:project ../../src/NumSharp.Core
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Nullable=disable

// The snapshot tool S of docs/plans/np-layer-dtype-logic.md (phase P0).
//
// A move of dtype logic out of the np.* facade must change nothing. This tool records NumSharp's OWN outcome for every
// public overload of the named functions over a generated argument matrix, so the outcome before a move (the parent
// commit) and after it (the child) can be compared line by line: the diff of a move commit is empty.
//
// usage (from the repo root, after every Core edit: --no-cache, or dotnet run reuses a stale build):
//   dotnet run --no-cache -c Release tools/facade-move/snapshot.cs -- <out.tsv> np.cumsum np.cumprod NDArray.cumsum ...
//   diff <parent.tsv> <child.tsv>
//
// A name "np.f" is every public static overload of np.f; "NDArray.f" every public instance overload declared on NDArray;
// "NDArray.op_X" every public static operator overload declared on NDArray (op_Equality, op_UnaryPlus, ...), whose first
// NDArray parameter is the operand; "NDMatrix.f" every public instance overload declared on NDMatrix, called on
// np.asmatrix of each operand (a layout asmatrix refuses, the 3-D ones, is SKIP) with an out= variant made a matrix too.
// Each output line is "<case id> TAB <outcome>":
//   case id  = the overload's signature | the operand's label | the varied parameter and its value label
//   outcome  = ARR class dtype (shape) flags hash [is_out] [OUT hash] [MUTATED in=hash]   for an NDArray result
//              VAL type value / TUPLE[...] / null                                            for other results
//              EXC type: message                                                             for an exception
// The hash is the first 16 hex digits of SHA-256 over the result's bytes in C order (repr for StringDType, whose cells
// hold pointers). Nothing in a line depends on the process (no addresses, no timing), so two runs of one build are
// identical; check that once before trusting a diff.
//
// The matrix: phase A calls every overload once per operand (18 dtypes x 13 layouts) with every other parameter at its
// base value; phase B varies one parameter at a time over its variants, on a reduced operand pool (every dtype x the
// layouts c2d, f2d, 0d, empty, 3d); phase C varies every PAIR of parameters together on a small pool (8 dtypes x c2d,
// 3d, 0d), where checks that meet only when two arguments are off their defaults show. Parameters are recognized by type
// and name (axis, out, where, q, weights, method, dtype, bool flags, the int offset / axis1 / axis2 / ddof, a second array
// operand named lhs / rhs / left / right / x / y / other, the histogram family's bins as an int, an estimator name, an
// array or a double[], and its (lower, upper) range); an overload with a required parameter of another kind is reported
// once as SKIP. A result that is a struct of arrays (np.histogram's HistogramResult) renders each array field.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using NumSharp;

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: snapshot.cs <out.tsv> <np.name | NDArray.name | NDArray.op_X | NDMatrix.name>...");
    return 2;
}

var output = new StreamWriter(args[0], false, new UTF8Encoding(false)) { NewLine = "\n" };
int lines = 0;
foreach (var target in args.Skip(1))
    foreach (var line in Snapshot.Run(target))
    {
        output.WriteLine(line);
        lines++;
    }
output.Flush();
output.Dispose();
Console.WriteLine($"{lines} cases written to {args[0]}");
return 0;

/// <summary>
///     The generator and recorder of the snapshot tool: builds operands and argument variants, invokes the overloads by
///     reflection and renders each outcome as one deterministic line.
/// </summary>
static class Snapshot
{
    /// <summary>The dtypes of the operand pool: the 15 storage dtypes and the three text dtypes.</summary>
    private static readonly string[] Dtypes =
    {
        "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "char",
        "float16", "float32", "float64", "decimal", "complex128", "S5", "U5", "T",
    };

    /// <summary>The layouts of the full operand pool (phase A).</summary>
    private static readonly string[] Layouts =
    {
        "c2d", "c2d_special", "f2d", "strided", "reversed", "bcast", "0d", "0d_special", "empty", "empty1d", "1d", "3d",
        "3d_special",
    };

    /// <summary>The layouts of the reduced pool used when another parameter varies (phase B).</summary>
    private static readonly string[] ReducedLayouts = { "c2d", "f2d", "0d", "empty", "3d" };

    /// <summary>The plain numeric value sequence; operands take a prefix of it.</summary>
    private static readonly double[] Plain =
    {
        3, -1, 2, 0, 5, -4, 1, 7, -2, 6, 4, -3, 8, -5, 9, -6, 10, -7, 11, -8, 12, -9, 13, -10,
    };

    /// <summary>The special numeric sequence: NaN (the positive quiet NaN of Python), signed zeros and infinities mixed in.</summary>
    private static readonly double[] Special =
    {
        3.5, double.PositiveInfinity, -0.0, BitConverter.Int64BitsToDouble(0x7FF8000000000000), 2.25, -1.5,
        double.NegativeInfinity, 0.0, 7.75, -2.0, 1e300, -6.5, 4.0, 1e-300, -3.25, 8.0, 9.5, -0.0,
        BitConverter.Int64BitsToDouble(0x7FF8000000000000), 10.0, 11.5, -8.0, 12.0, -9.5,
    };

    /// <summary>The plain text sequence (numbers written as text, so a numeric dtype= can parse them).</summary>
    private static readonly string[] PlainText = Plain.Select(v => v.ToString(CultureInfo.InvariantCulture)).ToArray();

    /// <summary>The special text sequence: text that parses, text that does not, the empty string.</summary>
    private static readonly string[] SpecialText =
    {
        "3.5", "x", "", "nan", "-1", "ab", "1e3", "inf", "z9", "0", "  2", "-0", "7", "q", "4.25", "", "b", "5", "-inf",
        "c", "8", "d", "9", "e",
    };

    /// <summary>
    ///     Snapshots every overload of one target.
    /// </summary>
    /// <param name="target"><c>np.name</c> for the public static overloads of <c>np.name</c>, <c>NDArray.name</c> for the
    ///     public instance overloads declared on NDArray (<c>NDArray.op_X</c> for its static operators),
    ///     <c>NDMatrix.name</c> for the public instance overloads declared on NDMatrix.</param>
    /// <returns>The output lines, overloads ordered by signature.</returns>
    /// <exception cref="ArgumentException">The target names no overload.</exception>
    /// <remarks>
    ///     An NDMatrix target is called on <c>np.asmatrix</c> of each pool operand: the matrix members are where numpy.matrix's
    ///     own reductions live (their <c>out=</c> paths implement NumPy's reducelike rule), and a move of their bodies into the
    ///     engine is held to this snapshot like any other.
    /// </remarks>
    public static IEnumerable<string> Run(string target)
    {
        string name = target.Substring(target.IndexOf('.') + 1);
        bool onNDArray = target.StartsWith("NDArray.", StringComparison.Ordinal);
        bool onMatrix = target.StartsWith("NDMatrix.", StringComparison.Ordinal);
        // An operator is a static method of NDArray whose first NDArray parameter is the operand, as np.f's is.
        bool instance = (onNDArray && !name.StartsWith("op_", StringComparison.Ordinal)) || onMatrix;
        var methods = (onMatrix
                ? typeof(NDMatrix).GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => m.DeclaringType == typeof(NDMatrix))
                : instance
                    ? typeof(NDArray).GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => m.DeclaringType == typeof(NDArray))
                    : onNDArray
                        ? typeof(NDArray).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.DeclaringType == typeof(NDArray))
                        : typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static))
            .Where(m => m.Name == name && !m.IsGenericMethodDefinition)
            .Select(m => (Method: m, Signature: Signature(target, m)))
            .OrderBy(t => t.Signature, StringComparer.Ordinal)
            .ToList();
        if (methods.Count == 0)
            throw new ArgumentException($"no public overload named {target}");
        foreach (var (method, signature) in methods)
            foreach (var line in RunOverload(method, signature, instance, onMatrix))
                yield return line;
    }

    /// <summary>A stable text for an overload: the target and its parameter types and names.</summary>
    /// <param name="target">The target as given on the command line.</param>
    /// <param name="m">The overload.</param>
    /// <returns>For example <c>np.cumsum(NDArray arr, Int32? axis, DType typeCode, NDArray out)</c>.</returns>
    private static string Signature(string target, MethodInfo m) =>
        target + "(" + string.Join(", ", m.GetParameters().Select(p => TypeName(p.ParameterType) + " " + p.Name)) + ")";

    /// <summary>A short type name: <c>Int32?</c> for a nullable, <c>Double[]</c> for an array.</summary>
    /// <param name="t">The type.</param>
    /// <returns>The short name.</returns>
    private static string TypeName(Type t) =>
        Nullable.GetUnderlyingType(t) is { } u ? u.Name + "?" : t.Name;

    /// <summary>The role of a parameter, decided by its type and name.</summary>
    private enum Role
    {
        Operand, Operand2, Out, Weights, Where, QArray, QDouble, QDoubles, QLong, Axis, AxisInt, AxisArray, IntArg, DType,
        Bool, Method, Order,
        // np.histogram / np.histogram_bin_edges: bins as an int count, an estimator name, an array (edges or a 0-d count)
        // or a double[] of edges, and the optional (lower, upper) range.
        Bins, BinsName, BinsArray, BinsDoubles, Range,
        Default, Unsupported,
    }

    /// <summary>
    ///     Classifies a parameter.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <param name="isFirstArray">Whether it is the first NDArray parameter of a static overload (the operand).</param>
    /// <returns>Its role; <see cref="Role.Default"/> for an optional parameter of another kind, which keeps its default.</returns>
    private static Role Classify(ParameterInfo p, bool isFirstArray)
    {
        Type t = p.ParameterType;
        string n = p.Name;
        // An NDMatrix parameter (numpy.matrix's out=, typed as the matrix NumPy requires) takes the array roles too; other
        // NDArray subclasses (NDArray<bool> masks, ...) keep falling through, so the snapshots of the np.* and NDArray targets
        // are unchanged by the NDMatrix support.
        if (t == typeof(NDArray) || t == typeof(NDMatrix))
            return isFirstArray ? Role.Operand
                : n == "out" ? Role.Out
                : n == "weights" ? Role.Weights
                : n == "where" ? Role.Where
                : n is "lhs" or "rhs" or "left" or "right" or "x" or "y" or "other" ? Role.Operand2
                : n == "q" ? Role.QArray
                : n == "bins" ? Role.BinsArray
                : p.HasDefaultValue ? Role.Default : Role.Unsupported;
        // The histogram family's bins in its other C# forms, and its range.
        if (n == "bins" && t == typeof(int)) return Role.Bins;
        if (n == "bins" && t == typeof(string)) return Role.BinsName;
        if (n == "bins" && t == typeof(double[])) return Role.BinsDoubles;
        if (n == "range" && t == typeof((double, double)?)) return Role.Range;
        if (n == "q" && t == typeof(double)) return Role.QDouble;
        if (n == "q" && t == typeof(double[])) return Role.QDoubles;
        if (n == "q" && t == typeof(long)) return Role.QLong;
        if (n == "axis" && t == typeof(int?)) return Role.Axis;
        if (n == "axis" && t == typeof(int)) return Role.AxisInt;
        if (n == "axis" && t == typeof(int[])) return Role.AxisArray;
        // np.trace / np.diagonal's integer arguments, and numpy.matrix's var / std ddof: varied around their defaults (out of
        // range included; a ddof past the count makes the divisor 0).
        if (n is "offset" or "axis1" or "axis2" or "ddof" && t == typeof(int)) return Role.IntArg;
        if (t == typeof(DType)) return Role.DType;
        if (t == typeof(bool)) return Role.Bool;
        if (n == "method" && t == typeof(string)) return Role.Method;
        if (n == "order" && (t == typeof(string) || t == typeof(char))) return Role.Order;
        return p.HasDefaultValue ? Role.Default : Role.Unsupported;
    }

    /// <summary>
    ///     The variants of a parameter: a label and a factory per value. The first variant is the base value.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <param name="role">Its role.</param>
    /// <returns>The variants; for <see cref="Role.Out"/> only the base (null) — out variants depend on the result.</returns>
    private static List<(string Label, Func<object> Make)> Variants(ParameterInfo p, Role role)
    {
        var list = new List<(string, Func<object>)>();
        void Add(string label, Func<object> make) => list.Add((label, make));
        switch (role)
        {
            case Role.Out:
            case Role.Weights:
            case Role.Where:
                Add("null", () => null);
                break;
            case Role.Operand2:
                // The base is the operand's own copy (MakeOperand2 "same"); the variants follow from the operand.
                Add("same", () => null);
                break;
            case Role.QArray:
                Add("0.5", () => np.array(0.5));
                Add("[0.25,0.75]", () => np.array(new[] { 0.25, 0.75 }));
                Add("[[0.1],[0.9]]", () => np.array(new[] { 0.1, 0.9 }).reshape(new Shape(2, 1)));
                Add("[1.5]", () => np.array(new[] { 1.5 }));
                Add("int[0,1]", () => np.array(new long[] { 0, 1 }));
                Add("bool[True]", () => np.array(new[] { true }));
                Add("f32[0.5]", () => np.array(new[] { 0.5f }));
                Add("T[0.5]", () => np.array(new[] { "0.5" }).astype(np.dtype("T")));
                break;
            case Role.QDouble:
                foreach (var q in new[] { 0.5, 0.0, 1.0, 1.5, -0.25 })
                    Add(q.ToString("R", CultureInfo.InvariantCulture), () => q);
                Add("nan", () => BitConverter.Int64BitsToDouble(0x7FF8000000000000));
                break;
            case Role.QDoubles:
                Add("[0.25,0.75]", () => new[] { 0.25, 0.75 });
                Add("[]", () => Array.Empty<double>());
                Add("[0.5,2.0]", () => new[] { 0.5, 2.0 });
                break;
            case Role.QLong:
                foreach (var q in new long[] { 0, 1, 2, -1 })
                    Add(q.ToString(CultureInfo.InvariantCulture), () => q);
                break;
            case Role.Axis:
                Add("null", () => null);
                foreach (var a in new[] { 0, 1, -1, 2, 5, -6 })
                    Add(a.ToString(CultureInfo.InvariantCulture), () => (int?)a);
                break;
            case Role.AxisInt:
                foreach (var a in new[] { 0, 1, -1, 5 })
                    Add(a.ToString(CultureInfo.InvariantCulture), () => a);
                break;
            case Role.IntArg:
            {
                // The default first (the base value), then values on both sides of it and one out of range.
                int d0 = p.HasDefaultValue && p.DefaultValue is int di ? di : 0;
                Add(d0.ToString(CultureInfo.InvariantCulture), () => d0);
                foreach (var v in p.Name == "offset" ? new[] { 1, -1, 3 } : new[] { 0, 1, 2, -1, 5 })
                    if (v != d0)
                        Add(v.ToString(CultureInfo.InvariantCulture), () => v);
                break;
            }
            case Role.AxisArray:
                Add("null", () => null);
                foreach (var a in new[] { new[] { 0 }, new[] { 1 }, new[] { 0, 1 }, new[] { -1 }, new[] { 0, 0 }, new[] { 1, 2 }, new[] { 9 } })
                    Add("[" + string.Join(",", a) + "]", () => a.ToArray());
                break;
            case Role.DType:
                Add("null", () => null);
                foreach (var d in new[] { "float64", "int32", "int8", "bool", "float16", "complex128", "U5", "T" })
                    Add(d, () => np.dtype(d));
                break;
            case Role.Bool:
                bool b0 = p.HasDefaultValue && p.DefaultValue is bool db && db;
                Add(b0 ? "true" : "false", () => b0);
                Add(b0 ? "false" : "true", () => !b0);
                break;
            case Role.Method:
                foreach (var m in new[] { "linear", "lower", "nearest", "inverted_cdf", "median_unbiased", "bogus" })
                    Add(m, () => m);
                break;
            case Role.Order:
                foreach (var o in new[] { "K", "C", "F", "A", "Z" })
                    Add(o, () => p.ParameterType == typeof(char) ? o[0] : o);
                break;
            case Role.Bins:
            {
                // The default count first, then a single bin, a few bins, and the two counts NumPy refuses.
                int count0 = p.HasDefaultValue && p.DefaultValue is int dcount ? dcount : 10;
                Add(count0.ToString(CultureInfo.InvariantCulture), () => count0);
                foreach (var count in new[] { 1, 3, 0, -2 })
                    if (count != count0)
                        Add(count.ToString(CultureInfo.InvariantCulture), () => count);
                break;
            }
            case Role.BinsName:
                // Every estimator NumPy knows (stone runs a histogram per candidate count), then an unknown name.
                foreach (var e in new[] { "auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt", "bogus" })
                    Add(e, () => e);
                break;
            case Role.BinsArray:
                // Explicit edges inside and around the pool's values (-10 .. 13): float64 and int64 edges, edges NumPy refuses
                // (decreasing, two-dimensional), a 0-d integer count and a 0-d float (not an index), the degenerate empty and
                // one-edge arrays, edges with a NaN (no comparison is True, so they pass the monotonic check: an inner NaN
                // makes NumPy's cumulative counts negative, a NaN last edge counts the NaN samples), repeated edges (an empty
                // bin inside, an inclusive last bin of zero width), infinite outer edges, and text edges.
                Add("f64[-2,0,1.5,5]", () => np.array(new[] { -2.0, 0.0, 1.5, 5.0 }));
                Add("i64[-4,0,2,4]", () => np.array(new long[] { -4, 0, 2, 4 }));
                Add("decr[5,1]", () => np.array(new[] { 5.0, 1.0 }));
                Add("2d", () => np.array(new[] { 0.0, 1.0, 2.0, 3.0 }).reshape(new Shape(2, 2)));
                Add("0d_i3", () => np.array(3L));
                Add("0d_f2.5", () => np.array(2.5));
                Add("empty", () => np.empty(new Shape(0), np.dtype("float64")));
                Add("one[1.5]", () => np.array(new[] { 1.5 }));
                Add("nan[0,nan,2]", () => np.array(new[] { 0.0, BitConverter.Int64BitsToDouble(0x7FF8000000000000), 2.0 }));
                Add("nanlast[0,2,nan]", () => np.array(new[] { 0.0, 2.0, BitConverter.Int64BitsToDouble(0x7FF8000000000000) }));
                Add("dup[0,1,1,2]", () => np.array(new[] { 0.0, 1.0, 1.0, 2.0 }));
                Add("dupend[0,1,2,2]", () => np.array(new[] { 0.0, 1.0, 2.0, 2.0 }));
                Add("inf[-inf,0,inf]", () => np.array(new[] { double.NegativeInfinity, 0.0, double.PositiveInfinity }));
                Add("U[1,2,10]", () => np.array(new[] { "1", "2", "10" }).astype(np.dtype("U5")));
                Add("T[a,b]", () => np.array(new[] { "a", "b" }).astype(np.dtype("T")));
                break;
            case Role.BinsDoubles:
                Add("[-2,0,1.5,5]", () => new[] { -2.0, 0.0, 1.5, 5.0 });
                Add("[]", () => Array.Empty<double>());
                Add("[3,1]", () => new[] { 3.0, 1.0 });
                Add("[1]", () => new[] { 1.0 });
                break;
            case Role.Range:
                // No range first, then ranges inside, around and outside the data, and the ones NumPy refuses (a lower
                // edge above the upper one, a NaN or infinite bound); a zero-width range widens by 0.5 on both sides.
                Add("null", () => null);
                Add("(0,1)", () => ((double, double)?)(0.0, 1.0));
                Add("(-1,10)", () => ((double, double)?)(-1.0, 10.0));
                Add("(1,0)", () => ((double, double)?)(1.0, 0.0));
                Add("(nan,1)", () => ((double, double)?)(BitConverter.Int64BitsToDouble(0x7FF8000000000000), 1.0));
                Add("(0,inf)", () => ((double, double)?)(0.0, double.PositiveInfinity));
                Add("(2,2)", () => ((double, double)?)(2.0, 2.0));
                break;
            case Role.Default:
                Add("default", () => p.DefaultValue is DBNull ? null : p.DefaultValue);
                break;
        }
        return list;
    }

    /// <summary>The dtypes of the small operand pool phase C runs every parameter pair on.</summary>
    private static readonly string[] PairDtypes = { "bool", "int32", "uint8", "float16", "float64", "complex128", "U5", "T" };

    /// <summary>The layouts of the small operand pool phase C runs every parameter pair on.</summary>
    private static readonly string[] PairLayouts = { "c2d", "3d", "0d" };

    /// <summary>
    ///     Snapshots one overload: phase A over the full operand pool, phase B varying each parameter on the reduced pool,
    ///     phase C varying every pair of parameters together on the small pool.
    /// </summary>
    /// <param name="m">The overload.</param>
    /// <param name="signature">Its stable signature text.</param>
    /// <param name="instance">Whether the operand is the receiver of an instance method.</param>
    /// <param name="matrix">Whether the receiver is an NDMatrix (np.asmatrix of each pool operand, see
    ///     <see cref="Call"/>).</param>
    /// <returns>The output lines.</returns>
    /// <remarks>
    ///     Phase C exists because a moved prologue can reorder checks that only meet when two arguments are off their
    ///     defaults together (an <c>out=</c> sized for one <c>axis</c>, a <c>method</c> with an array <c>q</c>).
    /// </remarks>
    private static IEnumerable<string> RunOverload(MethodInfo m, string signature, bool instance, bool matrix)
    {
        var ps = m.GetParameters();
        var roles = new Role[ps.Length];
        bool sawArray = instance;
        for (int i = 0; i < ps.Length; i++)
        {
            roles[i] = Classify(ps[i], !sawArray && ps[i].ParameterType == typeof(NDArray));
            if (ps[i].ParameterType == typeof(NDArray)) sawArray = true;
        }
        int unsupported = Array.IndexOf(roles, Role.Unsupported);
        if (unsupported >= 0)
        {
            yield return $"{signature}\tSKIP unsupported required parameter {TypeName(ps[unsupported].ParameterType)} {ps[unsupported].Name}";
            yield break;
        }
        var variants = ps.Select((p, i) => roles[i] == Role.Operand ? null : Variants(p, roles[i])).ToArray();
        int operandIndex = Array.IndexOf(roles, Role.Operand);
        var none = Array.Empty<(int, int)>();

        // The number of non-base variants per parameter (0 for the operand and for parameters kept at their default).
        int Count(int i) => roles[i] switch
        {
            Role.Operand => 0,
            Role.Out => OutVariantLabels.Length,
            Role.Weights => WeightVariantLabels.Length,
            Role.Where => WhereVariantLabels.Length,
            Role.Operand2 => Operand2VariantLabels.Length - 1,
            _ => variants[i].Count - 1,
        };

        // phase A: every operand, every other parameter at its base
        foreach (var dt in Dtypes)
            foreach (var layout in Layouts)
                yield return Call(m, signature, instance, matrix, operandIndex, dt, layout, variants, roles, none);

        // phase B: one parameter varied at a time, on the reduced pool
        for (int p = 0; p < ps.Length; p++)
            for (int v = 1; v <= Count(p); v++)
                foreach (var dt in Dtypes)
                    foreach (var layout in ReducedLayouts)
                        yield return Call(m, signature, instance, matrix, operandIndex, dt, layout, variants, roles, new[] { (p, v) });

        // phase C: every pair of parameters varied together, on the small pool
        for (int p1 = 0; p1 < ps.Length; p1++)
            for (int p2 = p1 + 1; p2 < ps.Length; p2++)
                for (int v1 = 1; v1 <= Count(p1); v1++)
                    for (int v2 = 1; v2 <= Count(p2); v2++)
                        foreach (var dt in PairDtypes)
                            foreach (var layout in PairLayouts)
                                yield return Call(m, signature, instance, matrix, operandIndex, dt, layout, variants, roles,
                                    new[] { (p1, v1), (p2, v2) });
    }

    /// <summary>The out= variants, built from the base call's result.</summary>
    private static readonly string[] OutVariantLabels = { "same", "float64", "int8", "bool", "U5", "T", "wrongshape", "readonly" };

    /// <summary>The weights= variants, built from the operand (see <see cref="MakeWeights"/>).</summary>
    private static readonly string[] WeightVariantLabels =
        { "ones", "negative", "wrongshape", "int", "complex", "U5", "T", "bool", "int8", "float32", "float16" };

    /// <summary>The second-operand variants, built from the operand (see <see cref="MakeOperand2"/>); the first is the
    ///     base.</summary>
    private static readonly string[] Operand2VariantLabels = { "same", "int32", "float64", "U5", "T", "0d", "row", "badshape" };

    /// <summary>The where= variants, built from the operand (see <see cref="MakeWhere"/>).</summary>
    private static readonly string[] WhereVariantLabels = { "mask", "row", "allfalse", "false0d", "int", "wrongshape" };

    /// <summary>
    ///     Makes one call and renders its outcome.
    /// </summary>
    /// <param name="m">The overload.</param>
    /// <param name="signature">Its signature text.</param>
    /// <param name="instance">Whether the operand is the receiver.</param>
    /// <param name="matrix">Whether the receiver is an NDMatrix: the pool operand passes through <c>np.asmatrix</c> (a
    ///     layout it refuses is SKIP), and an <c>out=</c> parameter typed NDMatrix receives <c>np.asmatrix</c> of its
    ///     variant.</param>
    /// <param name="operandIndex">The operand parameter's index (static overloads), or -1.</param>
    /// <param name="dt">The operand dtype.</param>
    /// <param name="layout">The operand layout.</param>
    /// <param name="variants">Per parameter, its variants (null for the operand).</param>
    /// <param name="roles">Per parameter, its role.</param>
    /// <param name="assignments">The varied parameters and their 1-based variant indices; empty for phase A.</param>
    /// <returns>One output line.</returns>
    /// <remarks>
    ///     An <c>out=</c> variant is sized from the call's result with every OTHER assignment applied and <c>out</c>
    ///     null; when that call throws, the operand's shape and dtype stand in. A <c>weights=</c> or <c>where=</c> variant
    ///     is built from the operand.
    /// </remarks>
    private static string Call(MethodInfo m, string signature, bool instance, bool matrix, int operandIndex, string dt,
        string layout, List<(string Label, Func<object> Make)>[] variants, Role[] roles,
        IReadOnlyList<(int Param, int Variant)> assignments)
    {
        var ps = m.GetParameters();
        string varied = assignments.Count == 0
            ? "base"
            : string.Join(",", assignments.Select(a => $"{ps[a.Param].Name}={Label(a.Param, a.Variant)}"));
        string Label(int p, int v) => roles[p] switch
        {
            Role.Out => OutVariantLabels[v - 1],
            Role.Weights => WeightVariantLabels[v - 1],
            Role.Where => WhereVariantLabels[v - 1],
            Role.Operand2 => Operand2VariantLabels[v],
            _ => variants[p][v].Label,
        };

        NDArray operand = MakeOperand(dt, layout);
        // A matrix receiver is np.asmatrix's view of the pool operand (0-d -> (1, 1), 1-D -> (1, N)); a layout it refuses
        // (more than two long axes: "shape too large to be a matrix.") has no matrix to call the member on.
        if (matrix && operand is not null)
            operand = AsMatrixOrNull(operand);
        if (operand is null)
            return $"{signature}|{dt}/{layout}|{varied}\tSKIP operand unavailable";
        string inBefore = HashOf(operand);
        var argv = new object[ps.Length];
        NDArray outArray = null;
        try
        {
            for (int i = 0; i < ps.Length; i++)
                argv[i] = i == operandIndex ? operand
                    : roles[i] == Role.Operand2 ? MakeOperand2(Operand2VariantLabels[0], operand)
                    : variants[i][0].Make();
            int outParam = -1, outVariant = 0;
            foreach (var (p, v) in assignments)
            {
                switch (roles[p])
                {
                    case Role.Out:
                        outParam = p;
                        outVariant = v;
                        break;
                    case Role.Weights:
                        argv[p] = MakeWeights(WeightVariantLabels[v - 1], operand);
                        break;
                    case Role.Where:
                        argv[p] = MakeWhere(WhereVariantLabels[v - 1], operand);
                        break;
                    case Role.Operand2:
                        argv[p] = MakeOperand2(Operand2VariantLabels[v], operand);
                        break;
                    default:
                        argv[p] = variants[p][v].Make();
                        break;
                }
            }
            if (outParam >= 0)
            {
                // out= follows the result of the same call without it; a call that throws leaves the operand's shape
                Shape shape = operand.Shape;
                DType resultType = operand.dtype;
                try
                {
                    if (Invoke(m, instance, operand, (object[])argv.Clone()) is NDArray baseResult)
                    {
                        shape = baseResult.Shape;
                        resultType = baseResult.dtype;
                    }
                }
                catch { /* the operand's shape stands in */ }
                outArray = MakeOut(OutVariantLabels[outVariant - 1], new Shape(shape.Dimensions), resultType);
                // numpy.matrix's out= must be a matrix (NumPy calls out._collapse on it): the matrix view of the variant
                // is the object passed, so "is_out" and the OUT hash read the same object the member received. A
                // variant asmatrix refuses (wrongshape grows a third axis) is a SETUP line.
                if (ps[outParam].ParameterType == typeof(NDMatrix))
                    outArray = np.asmatrix(outArray);
                argv[outParam] = outArray;
            }
        }
        catch (Exception e)
        {
            return $"{signature}|{dt}/{layout}|{varied}\tSETUP {Describe(e)}";
        }

        string outcome;
        try
        {
            object result = Invoke(m, instance, operand, argv);
            outcome = Render(result, outArray);
            if (outArray is not null && !ReferenceEquals(result, outArray))
                outcome += " OUT " + HashOf(outArray);
        }
        catch (Exception e)
        {
            outcome = "EXC " + Describe(e);
        }
        string inAfter = HashOf(operand);
        if (inAfter != inBefore)
            outcome += " MUTATED in=" + inAfter;
        return $"{signature}|{dt}/{layout}|{varied}\t{outcome}";
    }

    /// <summary>Invokes an overload, unwrapping the reflection exception.</summary>
    /// <param name="m">The overload.</param>
    /// <param name="instance">Whether the operand is the receiver.</param>
    /// <param name="operand">The operand.</param>
    /// <param name="argv">The arguments.</param>
    /// <returns>The return value.</returns>
    private static object Invoke(MethodInfo m, bool instance, NDArray operand, object[] argv)
    {
        try
        {
            return m.Invoke(instance ? operand : null, argv);
        }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerException).Throw();
            throw;
        }
    }

    /// <summary>An exception as one line: its type's full name and its message, newlines escaped.</summary>
    /// <param name="e">The exception.</param>
    /// <returns><c>Type: message</c>.</returns>
    private static string Describe(Exception e) =>
        e.GetType().FullName + ": " + (e.Message ?? "").Replace("\r", "").Replace("\n", "\\n");

    /// <summary>
    ///     Builds an operand: the dtype's values in the layout, or null when the dtype cannot be built.
    /// </summary>
    /// <param name="dt">The dtype name.</param>
    /// <param name="layout">The layout name.</param>
    /// <returns>A fresh operand.</returns>
    private static NDArray MakeOperand(string dt, string layout)
    {
        bool special = layout.EndsWith("_special", StringComparison.Ordinal);
        string baseLayout = special ? layout.Substring(0, layout.Length - "_special".Length) : layout;
        NDArray Values(int count) => Build(dt, count, special);
        switch (baseLayout)
        {
            case "c2d": return Values(12).reshape(new Shape(3, 4));
            case "f2d": return np.asfortranarray(Values(12).reshape(new Shape(3, 4)));
            case "strided": return Values(24).reshape(new Shape(3, 8))[":, ::2"];
            case "reversed": return Values(12).reshape(new Shape(3, 4))["::-1, ::-1"];
            case "bcast": return np.broadcast_to(Values(4).reshape(new Shape(1, 4)), new Shape(3, 4));
            case "0d": return Values(1).reshape(Shape.Scalar);
            case "empty": return Values(0).reshape(new Shape(0, 3));
            case "empty1d": return Values(0);
            case "1d": return Values(6);
            case "3d": return Values(24).reshape(new Shape(2, 3, 4));
            default: throw new ArgumentException("unknown layout " + layout);
        }
    }

    /// <summary>
    ///     A fresh 1-D array of <paramref name="count"/> values of a dtype, from the plain or the special sequence.
    /// </summary>
    /// <param name="dt">The dtype name.</param>
    /// <param name="count">The length (at most 24).</param>
    /// <param name="special">Whether to use the special sequence (NaN, signed zeros, infinities; text that does not parse).</param>
    /// <returns>A C-contiguous 1-D array.</returns>
    private static NDArray Build(string dt, int count, bool special)
    {
        switch (dt)
        {
            case "S5":
            case "U5":
            case "T":
                var text = (special ? SpecialText : PlainText).Take(count).ToArray();
                return count == 0 ? np.empty(new Shape(0), np.dtype(dt)) : np.array(text).astype(np.dtype(dt));
            case "decimal":
                return np.array((special ? Special : Plain).Take(count)
                    .Select(d => double.IsFinite(d) && Math.Abs(d) < 1e20 ? (decimal)d : 0m).ToArray());
            case "complex128":
                var src = (special ? Special : Plain).Take(count).ToArray();
                return np.array(src.Select((d, i) => new Complex(d, i % 3 == 0 ? -d : 0.5 * i)).ToArray());
            case "char":
                return np.array(Plain.Take(count).Select(d => (char)('a' + (int)Math.Abs(d))).ToArray());
            default:
                var values = np.array((special ? Special : Plain).Take(count).ToArray());
                return values.astype(np.dtype(dt));
        }
    }

    /// <summary>
    ///     The NDMatrix receiver of an NDMatrix target: <c>np.asmatrix</c> of a pool operand, or null when NumPy's
    ///     matrix refuses that shape.
    /// </summary>
    /// <param name="operand">A pool operand (not null).</param>
    /// <returns>A new matrix view of <paramref name="operand"/> (0-d becomes (1, 1), 1-D (1, N)), or null for an operand
    ///     with more than two axes longer than one ("shape too large to be a matrix.").</returns>
    /// <remarks>Only NumPy's ValueError is read as "no matrix": any other exception is a defect of the tool or of
    ///     np.asmatrix and propagates, so it cannot hide as a SKIP line.</remarks>
    private static NDArray AsMatrixOrNull(NDArray operand)
    {
        try
        {
            return np.asmatrix(operand);
        }
        catch (ValueError)
        {
            return null;
        }
    }

    /// <summary>
    ///     Builds an out= array for a result of the given shape and dtype.
    /// </summary>
    /// <param name="label">The variant: same, float64, int8, bool, U5, T, wrongshape, readonly.</param>
    /// <param name="shape">The result's shape.</param>
    /// <param name="resultType">The result's dtype.</param>
    /// <returns>A fresh array (made read-only for <c>readonly</c>).</returns>
    private static NDArray MakeOut(string label, Shape shape, DType resultType)
    {
        switch (label)
        {
            case "same": return np.zeros(shape, resultType);
            case "float64": return np.zeros(shape, np.dtype("float64"));
            case "int8": return np.zeros(shape, np.dtype("int8"));
            case "bool": return np.zeros(shape, np.dtype("bool"));
            case "U5": return np.zeros(shape, np.dtype("U5"));
            case "T": return np.zeros(shape, np.dtype("U5")).astype(np.dtype("T"));
            case "wrongshape":
                return np.zeros(new Shape(shape.Dimensions.Concat(new long[] { 2 }).ToArray()), resultType);
            case "readonly":
                // a fresh array made read-only (setflags): valid for every shape, the 0-d one included
                var readOnly = np.zeros(shape, resultType);
                readOnly.setflags(write: false);
                return readOnly;
            default: throw new ArgumentException("unknown out variant " + label);
        }
    }

    /// <summary>
    ///     Builds a weights= array for an operand.
    /// </summary>
    /// <param name="label">The variant: <c>ones</c> / <c>negative</c> (float64), <c>wrongshape</c> (one more axis),
    ///     <c>int</c> (int64 ones), <c>complex</c> (complex128 ones with an imaginary part: np.histogram's complex weights
    ///     path), <c>U5</c> / <c>T</c> (the text "1" in a fixed-width or StringDType array: the text routes of the
    ///     functions that take weights), <c>bool</c> / <c>int8</c> / <c>float32</c> / <c>float16</c> (ones in a dtype
    ///     narrower than the accumulator: NumPy's results keep the weights' dtype, and its in-place accumulations cast
    ///     back into it, so a bool total refuses the int64 cumulative sum).</param>
    /// <param name="operand">The operand.</param>
    /// <returns>A fresh array.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray MakeWeights(string label, NDArray operand)
    {
        var shape = new Shape(operand.Shape.Dimensions);
        long size = 1;
        foreach (long d in shape.Dimensions) size *= d;
        NDArray Text(string dt) => size == 0
            ? np.empty(shape, np.dtype(dt))
            : np.array(Enumerable.Repeat("1", (int)size).ToArray()).astype(np.dtype(dt)).reshape(shape);
        switch (label)
        {
            case "ones": return np.ones(shape, np.dtype("float64"));
            case "negative": return np.ones(shape, np.dtype("float64")) * -1.0;
            case "wrongshape": return np.ones(new Shape(shape.Dimensions.Concat(new long[] { 2 }).ToArray()), np.dtype("float64"));
            case "int": return np.ones(shape, np.dtype("int64"));
            case "complex":
                return np.array(Enumerable.Repeat(new Complex(1.0, 0.5), (int)size).ToArray()).reshape(shape);
            case "U5": return Text("U5");
            case "T": return Text("T");
            case "bool": return np.ones(shape, np.dtype("bool"));
            case "int8": return np.ones(shape, np.dtype("int8"));
            case "float32": return np.ones(shape, np.dtype("float32"));
            case "float16": return np.ones(shape, np.dtype("float16"));
            default: throw new ArgumentException("unknown weights variant " + label);
        }
    }

    /// <summary>
    ///     Builds a second array operand for an operand (the right side of an operator, the other array of a binary
    ///     function): a copy of it and arrays that differ in dtype class, rank or shape.
    /// </summary>
    /// <param name="label">The variant: <c>same</c> (a copy of the operand), <c>int32</c> / <c>float64</c> (a ramp of the
    ///     operand's shape), <c>U5</c> / <c>T</c> (text of the operand's shape: a text operand compares as text, a number
    ///     meets NumPy's failed-comparison rule), <c>0d</c> (an int64 0-d 2), <c>row</c> (the last axis only: it
    ///     broadcasts), <c>badshape</c> (seven elements, which no operand of this pool broadcasts against).</param>
    /// <param name="operand">The operand.</param>
    /// <returns>A fresh array.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray MakeOperand2(string label, NDArray operand)
    {
        long[] dims = operand.Shape.Dimensions;
        long size = 1;
        foreach (long d in dims) size *= d;
        NDArray Ramp(string dt, long count) =>
            np.array(Enumerable.Range(0, (int)count).Select(i => (double)(i % 4)).ToArray()).astype(np.dtype(dt));
        NDArray Text(string dt, long count) => count == 0
            ? np.empty(new Shape(0), np.dtype(dt))
            : np.array(Enumerable.Range(0, (int)count).Select(i => (i % 3).ToString(CultureInfo.InvariantCulture)).ToArray()).astype(np.dtype(dt));
        switch (label)
        {
            case "same": return operand.copy();
            case "int32": return Ramp("int32", size).reshape(new Shape(dims));
            case "float64": return Ramp("float64", size).reshape(new Shape(dims));
            case "U5": return Text("U5", size).reshape(new Shape(dims));
            case "T": return Text("T", size).reshape(new Shape(dims));
            case "0d": return np.array(2L);
            case "row": return Ramp("int32", dims.Length == 0 ? 1 : dims[dims.Length - 1]);
            case "badshape": return Ramp("int32", 7);
            default: throw new ArgumentException("unknown second-operand variant " + label);
        }
    }

    /// <summary>
    ///     Builds a where= mask for an operand: the masks NumPy's reduction accepts and the ones its prologue refuses.
    /// </summary>
    /// <param name="label">The variant: <c>mask</c> (bool, the operand's shape, every third element False), <c>row</c>
    ///     (bool, the operand's last axis only: it broadcasts; one element for a 0-d operand), <c>allfalse</c> (bool, the
    ///     operand's shape, all False: only identities reach the result), <c>false0d</c> (a 0-d False), <c>int</c> (int64
    ///     ones of the operand's shape: not a bool mask), <c>wrongshape</c> (bool with one more axis than the
    ///     operand).</param>
    /// <param name="operand">The operand.</param>
    /// <returns>A fresh array.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray MakeWhere(string label, NDArray operand)
    {
        long[] dims = operand.Shape.Dimensions;
        long size = 1;
        foreach (long d in dims) size *= d;
        NDArray Pattern(long count) =>
            np.array(Enumerable.Range(0, (int)count).Select(i => i % 3 != 0).ToArray());
        switch (label)
        {
            case "mask": return Pattern(size).reshape(new Shape(dims));
            case "row": return Pattern(dims.Length == 0 ? 1 : dims[dims.Length - 1]);
            case "allfalse": return np.zeros(new Shape(dims), np.dtype("bool"));
            case "false0d": return np.array(false);
            case "int": return np.ones(new Shape(dims), np.dtype("int64"));
            case "wrongshape": return np.ones(new Shape(dims.Concat(new long[] { 2 }).ToArray()), np.dtype("bool"));
            default: throw new ArgumentException("unknown where variant " + label);
        }
    }

    /// <summary>
    ///     Renders a result as one deterministic text.
    /// </summary>
    /// <param name="result">The return value.</param>
    /// <param name="outArray">The out= array passed, or null.</param>
    /// <returns>The outcome text.</returns>
    private static string Render(object result, NDArray outArray)
    {
        switch (result)
        {
            case null:
                return "null";
            case NDArray a:
                var f = a.flags;
                string flags = $"{(f.c_contiguous ? "C" : "-")}{(f.f_contiguous ? "F" : "-")}{(f.owndata ? "O" : "-")}{(f.writeable ? "W" : "-")}";
                return $"ARR {a.GetType().Name} {a.dtype} ({string.Join(",", a.shape)}) {flags} {HashOf(a)}" +
                       (outArray is not null && ReferenceEquals(a, outArray) ? " is_out" : "");
            case string s:
                return "VAL String " + s.Replace("\n", "\\n");
            case double d:
                return "VAL Double " + BitConverter.DoubleToInt64Bits(d).ToString("x16");
            case float fl:
                return "VAL Single " + BitConverter.SingleToInt32Bits(fl).ToString("x8");
            case ITuple t:
                return "TUPLE(" + string.Join("; ", Enumerable.Range(0, t.Length).Select(i => Render(t[i], outArray))) + ")";
            case IEnumerable e:
                return "LIST[" + string.Join("; ", e.Cast<object>().Select(o => Render(o, outArray))) + "]";
            default:
                // A struct of arrays (np.histogram's HistogramResult, np.unique's result structs): each public array field
                // in declaration order, so the arrays the call returns are in the line. Its ToString would only name the
                // type, which would hide every value from the snapshot.
                var arrayFields = result.GetType().IsValueType
                    ? result.GetType().GetFields(BindingFlags.Public | BindingFlags.Instance)
                        .OrderBy(fi => fi.MetadataToken).ToArray()
                    : Array.Empty<FieldInfo>();
                if (arrayFields.Length > 0
                    && arrayFields.All(fi => typeof(NDArray).IsAssignableFrom(fi.FieldType) || fi.FieldType == typeof(NDArray[])))
                    return "STRUCT " + result.GetType().Name + "(" +
                           string.Join("; ", arrayFields.Select(fi => fi.Name + "=" + Render(fi.GetValue(result), outArray))) + ")";
                return "VAL " + result.GetType().Name + " " + Convert.ToString(result, CultureInfo.InvariantCulture);
        }
    }

    /// <summary>
    ///     The first 16 hex digits of SHA-256 over an array's contents in C order: its bytes, or its repr for StringDType
    ///     (whose cells hold pointers) and for anything <c>tobytes</c> refuses.
    /// </summary>
    /// <param name="a">The array.</param>
    /// <returns>The digest, or <c>unhashable(type)</c> when neither form is available.</returns>
    private static string HashOf(NDArray a)
    {
        byte[] data;
        try
        {
            data = a.dtype.kind == 'T'
                ? Encoding.UTF8.GetBytes(a.ToString(true))
                : np.ascontiguousarray(a).tobytes();
        }
        catch (Exception e)
        {
            try { data = Encoding.UTF8.GetBytes(a.ToString(true)); }
            catch { return "unhashable(" + e.GetType().Name + ")"; }
        }
        return Convert.ToHexString(SHA256.HashData(data)).Substring(0, 16).ToLowerInvariant();
    }
}
