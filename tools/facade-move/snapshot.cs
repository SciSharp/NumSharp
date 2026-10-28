#:project ../../src/NumSharp.Core
#:project ../../src/NumSharp.Interop.OpenBLAS
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
// With the environment variable SNAPSHOT_BYTES=1 every array result's line also carries its C-order bytes (" BYTES
// <hex>"), to measure a value mismatch element by element; compare only snapshots made the same way.
// With SNAPSHOT_OPENBLAS=1 the tool installs the NumSharp.Interop.OpenBLAS backend at ONE thread before the first call, so
// the LAPACK factorisations of np.linalg compute (without it they raise OpenBlasMissingBackendException for a non-empty
// operand; det / slogdet / solve / inv take the managed LU)
// and every product with a BLAS route takes it; NumPy's own scipy-openblas at OPENBLAS_NUM_THREADS=1 gives the same bits.
// Without the variable the backend is never loaded (its module initializer runs only when a method naming its types is
// prepared), so the snapshots of the other functions are what they were. Compare only snapshots made the same way.
//
// A name "np.f" is every public static overload of np.f; "np.linalg.f" every public static overload of np.linalg.f (run on
// pools of square matrices, stacks, rectangles and empties, see LinalgLayouts); "np.emath.f" every public instance overload
// declared on the module class EmathModule, called on np.emath (its first array parameter is the operand, logn's x and
// power's exponent p are the second operand); "NDArray.f" every public instance overload declared on NDArray;
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
// operand named lhs / rhs / left / right / x / y / other / a2 (np.array_equiv's and the legacy poly arithmetic's
// a1 / a2) / ar2 (the set routines' ar1 / ar2), the histogram family's bins as an int, an estimator name, an
// array or a double[], and its (lower, upper) range, np.histogramdd / np.histogram2d's sample as a sequence of columns,
// their per-dimension bins as an int[], an NDArray[] or an object[], and their per-dimension ranges, np.trapezoid's
// optional sample points x and its spacing dx, np.searchsorted's keys v (an array, or a C# int / double), its side
// name and its sorter, np.cov / np.corrcoef's optional second set of variables y, nullable ddof and frequency /
// analytic weights, the gufuncs' second operand x2, axes= lists, nullable keepdims and casting rule, np.gradient's
// spacing (one, one per axis, or the varargs list: scalars, 0-d arrays, coordinates of the operand's axis lengths) and
// edge_order, and np.linalg's second operand b, multi_dot's array list, the orders ord / p, UPLO, qr's mode, matrix_power's
// n, tensorinv's ind, tensordot's axes (a count, or the two axis lists), tensorsolve's axes and the nullable tolerances
// rcond / rtol / tol, and np.insert / np.delete's positions obj (a C# int / long, a slice, an int[] / long[] sequence, an
// index array or a boolean mask) and np.insert's values (an array, or any object: a number, a string, a bool, null, a C#
// array, an NDArray passed as object), and np.isin's test set test_elements (the operand's own values, sets of other
// dtypes, sizes, ranges and layouts that reach NumPy's table, loop and sort paths) and its algorithm name kind, and
// np.average / np.average_returned's weights (of the operand's shape, along one axis or two in either order, a 0-d, zero
// sums, other dtypes, the forms NumPy refuses) and axis tuple (single axes, both orders of a pair, the empty tuple, a
// repeated and an out-of-range axis), and np.convolve / np.correlate's second sequence v (the operand's copy, other dtypes
// and lengths, one-element arrays of either stride sign, special values, text, the shapes NumPy refuses, strided and
// reversed views) and their mode name (the three names, near misses, other strings, null), and np.nditer's operand list
// op (the operand alone, with a copy, with an output to allocate, with a row, another dtype, a 0-d, a shape that does not
// broadcast, none), flags (each tracking and loop flag, the buffered forms, refs_ok, common_dtype, reduce_ok, ranged,
// two refused forms), op_flags (one list, or one per operand), op_dtypes (numbers, the unsized 'U' / 'S' NumPy adapts, a
// sized 'U5', StringDType, two refused forms), op_axes and itershape (built from the operand's rank and shape) and
// buffersize, its order and casting taking the general roles); an overload with a required
// parameter of another kind is reported once as SKIP. An int axis
// starts from its default when it has one. A search function (np.searchsorted) runs on pools of 1-D haystacks: sorted,
// strided, reversed, unsorted and empty ones beside one 2-D operand; np.convolve / np.correlate run on the same 1-D pools
// and a 24-element one (long1d), whose float products reach the BLAS-shaped vector regimes. An np.linalg function runs on pools of matrices
// (LinalgLayouts, 20 layouts in phase A: square ones in C, F, transposed, strided, reversed and broadcast form, a stack, a
// positive definite and a singular matrix, rectangles, empties; 7 in phase B, a square matrix and a rectangle in phase C),
// and multi_dot's array list is built from the operand (MakeLinalgArrays). A result that is a struct of arrays (np.histogram's
// HistogramResult, np.histogramdd's HistogramddResult) renders each array field, and one that keeps its arrays in
// properties (np.unique's UniqueResult, np.unique_all / unique_counts / unique_inverse's result structs) each NDArray
// property, null for an output not requested; np.gradient's GradientResult renders as GRAD1(array) or GRADN(array; ...).
// An np.nditer iterator is pulled to its end and renders as ITER nop=N dtypes=[...] steps=S with, per operand, its views'
// dtype, the run-length list of their shapes, their flag sets and one hash over every step's view (RenderIterator); take
// nditer snapshots under NUMSHARP_DEBUG_GUARD_PAGES=1, where an allocated output's memory reads as zeros.
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

// The OpenBLAS backend only on request: the call sits in its own non-inlined method, so a run without the variable never
// prepares a method that names the package's types, and the package's module initializer (which installs the bundled
// backend) never runs.
if (Environment.GetEnvironmentVariable("SNAPSHOT_OPENBLAS") == "1")
    Console.Error.WriteLine(Snapshot.EnableOpenBlas());

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
    /// <summary>
    ///     Installs the NumSharp.Interop.OpenBLAS backend (the bundled scipy-openblas, NumPy's own build) pinned to ONE
    ///     thread, the parity configuration: the bits of a BLAS / LAPACK result depend on the thread count.
    /// </summary>
    /// <returns>The backend's description, for the console.</returns>
    /// <exception cref="DllNotFoundException">No OpenBLAS library could be loaded (the bundled one is not staged:
    ///     <c>tools/fetch_openblas.py</c>).</exception>
    /// <remarks>
    ///     Not inlined, on purpose: the package's module initializer runs when the runtime first prepares a method that names
    ///     one of its types, so naming <c>OpenBlasEngine</c> in the top-level statements would install the backend for every
    ///     run. Here it runs only when the caller asks; it installs the bundled backend, and <c>Enable(threads: 1)</c> then
    ///     pins the thread count on the library it loaded.
    /// </remarks>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static string EnableOpenBlas()
    {
        NumSharp.Interop.OpenBLAS.OpenBlasEngine.Enable(threads: 1);
        return "snapshot: OpenBLAS backend at one thread: " + NumSharp.Interop.OpenBLAS.OpenBlasEngine.Info;
    }

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
    /// <param name="target"><c>np.name</c> for the public static overloads of <c>np.name</c>, <c>np.linalg.name</c> for
    ///     those of the module class <c>np.linalg</c>, <c>np.emath.name</c> for the public instance overloads declared on the
    ///     module class <see cref="EmathModule"/> (called on <c>np.emath</c>), <c>NDArray.name</c> for the public instance
    ///     overloads declared on NDArray (<c>NDArray.op_X</c> for its static operators), <c>NDMatrix.name</c> for the public
    ///     instance overloads declared on NDMatrix.</param>
    /// <returns>The output lines, overloads ordered by signature.</returns>
    /// <exception cref="ArgumentException">The target names no overload.</exception>
    /// <remarks>
    ///     An NDMatrix target is called on <c>np.asmatrix</c> of each pool operand: the matrix members are where numpy.matrix's
    ///     own reductions live (their <c>out=</c> paths implement NumPy's reducelike rule), and a move of their bodies into the
    ///     engine is held to this snapshot like any other.
    /// </remarks>
    public static IEnumerable<string> Run(string target)
    {
        // np.linalg.f is a member of the nested module class np.linalg, np.emath.f an instance member of the module object
        // np.emath: the name is what follows the LAST dot.
        bool onLinalg = target.StartsWith("np.linalg.", StringComparison.Ordinal);
        bool onEmath = target.StartsWith("np.emath.", StringComparison.Ordinal);
        string name = target.Substring((onLinalg || onEmath ? target.LastIndexOf('.') : target.IndexOf('.')) + 1);
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
                        : onLinalg
                            ? typeof(np.linalg).GetMethods(BindingFlags.Public | BindingFlags.Static)
                            : onEmath
                                ? typeof(EmathModule).GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => m.DeclaringType == typeof(EmathModule))
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
        // np.histogramdd / np.histogram2d: the sample as a sequence of columns, the per-dimension bins as an int[] / NDArray[]
        // / object[], and the per-dimension ranges. Their values depend on the sample's dimension count, so they are built
        // from the operand (see MakeDd) like the weights.
        SampleSeq, BinsInts, BinsArrays, BinsObjects, RangeArray,
        // np.trapezoid: the sample points x (an optional array whose default, null, means "evenly spaced by dx", so its base
        // is null and its variants are built from the operand like a second operand's) and the uniform spacing dx.
        Spacing, Dx,
        // np.searchsorted: the keys v (an array built from the operand, or a C# int / double key), the side name, and the
        // sorter (built from the operand: its argsort in several dtypes and the forms NumPy refuses).
        Keys, KeyInt, KeyDouble, Side, Sorter,
        // np.cov / np.corrcoef: the optional second set of variables y (base null, variants built from the operand like a
        // second operand's), the nullable ddof, and the frequency / analytic weights (built from the operand: one per
        // observation, for either orientation, and the forms NumPy refuses).
        Variables, Ddof, FWeights, AWeights,
        // The gufuncs (np.matmul / np.vecdot / np.matvec / np.vecmat): the per-operand core axes axes= (an int[][]),
        // matmul's nullable keepdims, and its casting rule.
        AxesList, KeepdimsNullable, Casting,
        // np.gradient: the single spacing of the one-spacing overloads (a scalar for every axis or one axis' coordinates;
        // base null, no spacing), a per-axis spacing of the two- and three-spacing overloads (base 1.0), the explicit
        // spacing list varargs (base empty), all built from the operand's axis lengths, and the boundary order edge_order.
        GradSpacing, GradAxisSpacing, GradVarargs, EdgeOrder,
        // np.linalg: multi_dot's array list (built from the operand), the norm / cond order (an object: a number, an
        // infinity, a name), eigh's UPLO, qr's mode, matrix_power's exponent n, tensordot's two axis lists, and the
        // nullable tolerances (pinv's rcond / rtol, matrix_rank's tol / rtol, lstsq's rcond).
        LinalgArrays, Ord, Uplo, QrMode, Power, AxesPairArg, NullableDouble,
        // np.insert / np.delete: the positions obj as a C# int / long, a slice, an int[] / long[] sequence, or an index array
        // or boolean mask; np.insert's values as an array, or as any object (a C# number, string, bool, null, a C# array, an
        // NDArray passed as object).
        InsertObj, InsertObjSlice, InsertObjSeq, InsertObjArray, InsertValues, InsertValue,
        // np.isin: the test set test_elements (built from the operand: its own values, sets of other dtypes, sizes, ranges
        // and layouts, so NumPy's table, loop and sort paths are all reached) and the algorithm name kind.
        IsinTest, IsinKind,
        // np.average / np.average_returned: the weights (built from the operand: of its shape, along one axis or two in
        // either order, a 0-d, sums that are zero, and the forms NumPy refuses) and the axis tuple (NumPy's
        // normalize_axis_tuple forms, an unsorted pair included, so _weights_are_valid's transpose is reached).
        AverageWeights, AverageAxes,
        // np.convolve / np.correlate and their NDArray members: the second sequence v (built from the operand: its copy,
        // sequences of other dtypes and lengths, one-element arrays of either stride sign, special values, text, a 0-d, a
        // 2-D, an empty one, strided and reversed views; see MakeSlidingKernel) and the mode name (the three names, the near
        // misses and other strings NumPy refuses with two different texts, and null).
        SlidingKernel, SlidingMode,
        // np.nditer: the multi-operand overload's operand list (built from the operand: the operand alone, with a copy, with
        // an output to allocate, with a row, an operand of another dtype, a 0-d, one that does not broadcast, and no
        // operand; see MakeNditerOps), the global flags, the per-operand flags (one list, or one per operand), op_dtypes,
        // op_axes and itershape (built from the operand's rank and shape) and buffersize. order and casting take the
        // general Order / Casting roles.
        NditerOps, NditerFlags, NditerOpFlags, NditerOpFlagsList, NditerOpDtypes, NditerOpAxes, NditerItershape,
        NditerBuffersize,
        Default, Unsupported,
    }

    /// <summary>
    ///     Whether a parameter belongs to np.nditer, whose <c>op</c> list, <c>flags</c>, <c>op_flags</c>, <c>op_dtypes</c>,
    ///     <c>op_axes</c>, <c>itershape</c> and <c>buffersize</c> roles are its alone: no other snapshot target declares
    ///     them, and np.nested_iters (whose <c>op</c> list and flags would read the same way) keeps its defaults until its
    ///     own family.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.nditer overload.</returns>
    private static bool IsNditer(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name == "nditer";

    /// <summary>
    ///     Whether a parameter belongs to np.average or np.average_returned, whose <c>weights</c> / <c>axis</c> tuple roles
    ///     are theirs alone: the histogram family names a parameter <c>weights</c> and np.sum / np.mean name an axis list
    ///     with other variants, and their snapshots must not change.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.average or np.average_returned overload.</returns>
    private static bool IsAverage(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name is "average" or "average_returned";

    /// <summary>
    ///     Whether a parameter belongs to np.insert or np.delete, whose <c>obj</c> / <c>values</c> / <c>value</c> roles are
    ///     theirs alone: other functions name parameters <c>value</c> or <c>obj</c> with other meanings (np.can_cast,
    ///     np.min_scalar_type, np.isscalar), and their snapshots must not change.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.insert or np.delete overload.</returns>
    private static bool IsInsertOrDelete(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name is "insert" or "delete";

    /// <summary>
    ///     Whether a parameter belongs to np.isin, whose <c>test_elements</c> / <c>kind</c> roles are its alone: np.sort,
    ///     np.argsort and np.partition name a parameter <c>kind</c> with another meaning, and their snapshots must not change.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.isin overload.</returns>
    private static bool IsIsin(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name == "isin";

    /// <summary>
    ///     Whether a parameter belongs to np.convolve / np.correlate or their NDArray members, whose <c>v</c> / <c>mode</c>
    ///     roles are theirs alone: np.searchsorted names its keys <c>v</c> and np.take / np.linalg.qr name a parameter
    ///     <c>mode</c> with other meanings, and their snapshots must not change.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public np.convolve / np.correlate overload or of NDArray's convolve / correlate.</returns>
    private static bool IsSliding(ParameterInfo p) =>
        (p.Member.DeclaringType == typeof(np) || p.Member.DeclaringType == typeof(NDArray))
        && p.Member.Name is "convolve" or "correlate";

    /// <summary>
    ///     Whether a parameter belongs to an np.emath function, whose exponent <c>p</c> (np.emath.power) is the second operand:
    ///     np.linalg's norm / cond name an order <c>p</c> with another meaning, and their snapshots must not change.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public instance member of <see cref="EmathModule"/>.</returns>
    private static bool IsEmath(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(EmathModule);

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
                // np.average's weights before the general weights role (its variants follow the reduced axes).
                : n == "weights" && IsAverage(p) ? Role.AverageWeights
                : n == "weights" ? Role.Weights
                : n == "where" ? Role.Where
                // np.trapezoid's x is optional (null = evenly spaced), unlike a binary function's second operand x: tested
                // before the operand names so its base stays the default.
                : n == "x" && p.HasDefaultValue ? Role.Spacing
                // np.convolve / np.correlate's second sequence, before np.searchsorted's keys of the same name.
                : n == "v" && IsSliding(p) ? Role.SlidingKernel
                // np.searchsorted's keys and sorter.
                : n == "v" ? Role.Keys
                : n == "sorter" ? Role.Sorter
                // np.cov / np.corrcoef's optional second set of variables (null = none), unlike a binary function's
                // second operand y: tested before the operand names so its base stays the default. Their weights.
                : n == "y" && p.HasDefaultValue ? Role.Variables
                : n == "fweights" ? Role.FWeights
                : n == "aweights" ? Role.AWeights
                // a2: np.array_equiv(a1, a2) and the legacy polyadd / polysub / polymul (a1, the first array, is the operand).
                // ar2: the set routines np.union1d / np.intersect1d / np.setxor1d / np.setdiff1d (ar1 is the operand), which
                // run np.unique's body over their concatenation.
                : n is "lhs" or "rhs" or "left" or "right" or "x" or "y" or "other" or "x2" or "b" or "a2" or "ar2" ? Role.Operand2
                // np.emath.power's exponent (np.emath.logn's x is caught by the names above).
                : n == "p" && IsEmath(p) ? Role.Operand2
                // np.insert / np.delete's index array or mask, and np.insert's values array.
                : n == "obj" && IsInsertOrDelete(p) ? Role.InsertObjArray
                : n == "values" && IsInsertOrDelete(p) ? Role.InsertValues
                // np.isin's test set.
                : n == "test_elements" && IsIsin(p) ? Role.IsinTest
                : n == "q" ? Role.QArray
                : n == "bins" ? Role.BinsArray
                : p.HasDefaultValue ? Role.Default : Role.Unsupported;
        // The histogram family's bins in its other C# forms, and its range.
        if (n == "bins" && t == typeof(int)) return Role.Bins;
        if (n == "bins" && t == typeof(string)) return Role.BinsName;
        if (n == "bins" && t == typeof(double[])) return Role.BinsDoubles;
        if (n == "range" && t == typeof((double, double)?)) return Role.Range;
        if (n == "bins" && t == typeof(int[])) return Role.BinsInts;
        if (n == "bins" && t == typeof(NDArray[])) return Role.BinsArrays;
        if (n == "bins" && t == typeof(object[])) return Role.BinsObjects;
        if (n == "range" && t == typeof((double, double)?[])) return Role.RangeArray;
        if (n == "dx" && t == typeof(double)) return Role.Dx;
        // np.searchsorted's C# scalar keys (a strong int32 / float64 key) and its side name.
        if (n == "v" && t == typeof(int)) return Role.KeyInt;
        if (n == "v" && t == typeof(double)) return Role.KeyDouble;
        if (n == "side" && t == typeof(string)) return Role.Side;
        if (n == "q" && t == typeof(double)) return Role.QDouble;
        if (n == "q" && t == typeof(double[])) return Role.QDoubles;
        if (n == "q" && t == typeof(long)) return Role.QLong;
        if (n == "axis" && t == typeof(int?)) return Role.Axis;
        if (n == "axis" && t == typeof(int)) return Role.AxisInt;
        // np.average's axis tuple (its own variants: an unsorted pair reaches _weights_are_valid's transpose).
        if (n == "axis" && t == typeof(int[]) && IsAverage(p)) return Role.AverageAxes;
        // An axis list, and np.linalg.tensorsolve's axes (the axes of a to move to the end; null = none).
        if ((n == "axis" || n == "axes") && t == typeof(int[])) return Role.AxisArray;
        // np.trace / np.diagonal's integer arguments, numpy.matrix's var / std ddof, np.linalg.tensorinv's ind and
        // tensordot's axis count: varied around their defaults (out of range included; a ddof past the count makes the
        // divisor 0).
        if (n is "offset" or "axis1" or "axis2" or "ddof" or "ind" or "axes" && t == typeof(int)) return Role.IntArg;
        // np.cov's nullable ddof (null = bias's default).
        if (n == "ddof" && t == typeof(int?)) return Role.Ddof;
        // The gufuncs' axes= list, matmul's nullable keepdims and its casting rule.
        if (n == "axes" && t == typeof(int[][])) return Role.AxesList;
        if (n == "keepdims" && t == typeof(bool?)) return Role.KeepdimsNullable;
        if (n == "casting" && t == typeof(string)) return Role.Casting;
        // np.gradient's spacings (typed object / object[]: NumPy's *varargs, each a scalar or a coordinate array) and its
        // boundary order.
        if (n == "spacing" && t == typeof(object)) return Role.GradSpacing;
        if (n is "spacing0" or "spacing1" or "spacing2" && t == typeof(object)) return Role.GradAxisSpacing;
        if (n == "varargs" && t == typeof(object[])) return Role.GradVarargs;
        if (n == "edge_order" && t == typeof(int)) return Role.EdgeOrder;
        // np.linalg: the norm / cond order (typed object: NumPy's ord / p takes numbers and names), eigh's UPLO, qr's mode
        // (only np.linalg's and np.convolve / np.correlate's, below: np.take's mode keeps its default), matrix_power's
        // exponent (required), tensordot's two axis lists (required), and the nullable tolerances.
        if ((n == "ord" || n == "p") && t == typeof(object)) return Role.Ord;
        if (n == "UPLO" && t == typeof(char)) return Role.Uplo;
        // np.convolve / np.correlate's mode name.
        if (n == "mode" && t == typeof(string) && IsSliding(p)) return Role.SlidingMode;
        bool inLinalg = p.Member.DeclaringType == typeof(np.linalg);
        if (n == "mode" && t == typeof(string) && inLinalg) return Role.QrMode;
        if (n == "n" && t == typeof(int) && inLinalg) return Role.Power;
        if (n is "axesA" or "axesB" && t == typeof(int[])) return Role.AxesPairArg;
        if (n is "rcond" or "rtol" or "tol" && t == typeof(double?)) return Role.NullableDouble;
        // np.insert / np.delete's positions in their other C# forms, and np.insert's object value (only theirs: other
        // functions name an object parameter value too, np.can_cast and np.min_scalar_type among them).
        if (IsInsertOrDelete(p))
        {
            if (n == "obj" && (t == typeof(int) || t == typeof(long))) return Role.InsertObj;
            if (n == "obj" && t == typeof(Slice)) return Role.InsertObjSlice;
            if (n == "obj" && (t == typeof(int[]) || t == typeof(long[]))) return Role.InsertObjSeq;
            if (n == "value" && t == typeof(object)) return Role.InsertValue;
        }
        // np.isin's algorithm name (only its: np.sort's, np.argsort's and np.partition's kind keep their defaults).
        if (n == "kind" && t == typeof(string) && IsIsin(p)) return Role.IsinKind;
        // np.nditer's arguments (only its; its order and casting take the general roles below and above).
        if (IsNditer(p))
        {
            if (n == "flags" && t == typeof(string[])) return Role.NditerFlags;
            if (n == "op_flags" && t == typeof(string[])) return Role.NditerOpFlags;
            if (n == "op_flags" && t == typeof(string[][])) return Role.NditerOpFlagsList;
            if (n == "op_dtypes" && t == typeof(DType[])) return Role.NditerOpDtypes;
            if (n == "op_axes" && t == typeof(int[][])) return Role.NditerOpAxes;
            if (n == "itershape" && t == typeof(long[])) return Role.NditerItershape;
            if (n == "buffersize" && t == typeof(long)) return Role.NditerBuffersize;
        }
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
            case Role.SampleSeq:
            case Role.BinsInts:
            case Role.BinsArrays:
            case Role.BinsObjects:
            case Role.RangeArray:
                // Built from the operand (MakeDd); the factory here is never called.
                Add(DdLabels(role)[0], () => null);
                break;
            case Role.Operand2:
                // The base is the operand's own copy (MakeOperand2 "same"); the variants follow from the operand.
                Add("same", () => null);
                break;
            case Role.Spacing:
                // The base is the default (no sample points); the variants follow from the operand (MakeSpacing).
                Add("null", () => null);
                break;
            case Role.Keys:
                // The base is the operand's own copy (MakeKeys "same"); the variants follow from the operand.
                Add("same", () => null);
                break;
            case Role.SlidingKernel:
                // The base is the operand's own copy (MakeSlidingKernel "same"); the variants follow from the operand.
                Add("same", () => null);
                break;
            case Role.SlidingMode:
            {
                // The overload's default first (np.convolve's 'full', np.correlate's 'valid'), the other two names, then the
                // strings NumPy's correlatemode_parser refuses: near misses (the right first letter: a capital, one letter, a
                // trailing space) with its own text, and other strings (one holding a quote, which NumPy's %R prints in
                // double quotes; the empty one) with the string converter's text quoting them; and null, Python's None, which
                // is neither a str nor an int ("convolve/correlate mode not understood").
                string baseMode = (string)p.DefaultValue;
                Add(baseMode, () => baseMode);
                foreach (var s in new[] { "valid", "same", "full" })
                    if (s != baseMode)
                        Add(s, () => s);
                foreach (var (label, s) in new[]
                         {
                             ("Full", "Full"), ("v", "v"), ("same_", "same "), ("x", "x"), ("quote", "it's"), ("empty", ""),
                             ("null", (string)null),
                         })
                    Add(label, () => s);
                break;
            }
            case Role.IsinTest:
                // The base is the operand's own copy (MakeIsinTest "same"); the variants follow from the operand.
                Add("same", () => null);
                break;
            case Role.IsinKind:
                // The default (automatic) first, the two names NumPy knows, then two it refuses: another word and a
                // capitalized name (the check is case-sensitive).
                foreach (var s in new[] { null, "sort", "table", "x", "Sort" })
                    Add(s ?? "null", () => s);
                break;
            case Role.NditerOps:
            case Role.NditerOpAxes:
            case Role.NditerItershape:
                // Built from the operand (MakeNditerOps / MakeNditerOpAxes / MakeNditerItershape); the factory here is never
                // called. The base is the operand alone / no op_axes / no itershape.
                Add(role == Role.NditerOps ? "one" : "null", () => null);
                break;
            case Role.NditerFlags:
                // No flags first, then each tracking and loop flag alone, the buffered forms (with the external loop, with
                // growing inner loops, with a delayed allocation that needs a reset before the first pull), the flags a
                // StringDType or a reduction operand needs, and two forms NumPy refuses (an unknown name, an index tracked
                // together with the external loop).
                Add("null", () => null);
                foreach (var (label, f) in new[]
                         {
                             ("multi_index", new[] { "multi_index" }), ("c_index", new[] { "c_index" }),
                             ("f_index", new[] { "f_index" }), ("external_loop", new[] { "external_loop" }),
                             ("buffered", new[] { "buffered" }), ("buffered_ext", new[] { "buffered", "external_loop" }),
                             ("grow_inner", new[] { "buffered", "external_loop", "grow_inner" }),
                             ("delay", new[] { "buffered", "delay_bufalloc" }), ("refs_ok", new[] { "refs_ok" }),
                             ("refs_buffered", new[] { "refs_ok", "buffered" }), ("zerosize_ok", new[] { "zerosize_ok" }),
                             ("common_dtype", new[] { "common_dtype" }), ("reduce_ok", new[] { "reduce_ok" }),
                             ("ranged", new[] { "ranged" }), ("bogus", new[] { "bogus" }),
                             ("mi_ext", new[] { "multi_index", "external_loop" }),
                         })
                    Add(label, () => (string[])f.Clone());
                break;
            case Role.NditerOpFlags:
                // The single operand's flags: none first (readonly), each access mode, a copy (readonly) and a write-back
                // copy (readwrite), the layout requests, and two forms NumPy refuses (an unknown name; no access mode).
                Add("null", () => null);
                foreach (var (label, f) in new[]
                         {
                             ("readonly", new[] { "readonly" }), ("readwrite", new[] { "readwrite" }),
                             ("writeonly", new[] { "writeonly" }), ("copy", new[] { "readonly", "copy" }),
                             ("updateifcopy", new[] { "readwrite", "updateifcopy" }),
                             ("no_broadcast", new[] { "readonly", "no_broadcast" }), ("contig", new[] { "readonly", "contig" }),
                             ("aligned_nbo", new[] { "readonly", "aligned", "nbo" }), ("bogus", new[] { "bogus" }),
                             ("empty", Array.Empty<string>()),
                         })
                    Add(label, () => (string[])f.Clone());
                break;
            case Role.NditerOpFlagsList:
                // The multi-operand form's flags: none first, one list broadcast to every operand, one list per operand
                // (an output to allocate, a written first operand), and two forms NumPy refuses (an unknown name; more
                // lists than operands).
                Add("null", () => null);
                foreach (var (label, f) in new[]
                         {
                             ("ro", new[] { new[] { "readonly" } }),
                             ("ro_alloc", new[] { new[] { "readonly" }, new[] { "writeonly", "allocate" } }),
                             ("rw_ro", new[] { new[] { "readwrite" }, new[] { "readonly" } }),
                             ("ro_ro", new[] { new[] { "readonly" }, new[] { "readonly" } }),
                             ("bogus", new[] { new[] { "bogus" } }),
                             ("three", new[] { new[] { "readonly" }, new[] { "readonly" }, new[] { "readonly" } }),
                         })
                    Add(label, () => f.Select(x => (string[])x.Clone()).ToArray());
                break;
            case Role.NditerOpDtypes:
                // None first, then one requested dtype (numbers, the unsized text classes 'U' / 'S' NumPy adapts to the
                // operand, a sized 'U5', StringDType), and two forms NumPy refuses for one operand (two entries) or reads as
                // "the operand's own" (a null entry).
                Add("null", () => null);
                foreach (var d in new[] { "float64", "int8", "bool", "complex128", "U", "U5", "S", "T" })
                    Add(d, () => new[] { np.dtype(d) });
                Add("two", () => new[] { np.dtype("float64"), np.dtype("float64") });
                Add("null_entry", () => new DType[] { null });
                break;
            case Role.NditerBuffersize:
                // The default (0, NumPy's buffer size) first, then buffers of one and three elements (several fills over
                // every pool operand) and a large one.
                foreach (var b in new long[] { 0, 1, 3, 8192 })
                    Add(b.ToString(CultureInfo.InvariantCulture), () => b);
                break;
            case Role.AverageWeights:
                // The base is no weights (the mean); the variants follow from the operand (MakeAverageWeights).
                Add("null", () => null);
                break;
            case Role.AverageAxes:
                // No axes (every axis) first, then single axes, both orders of a pair (an unsorted one reaches
                // _weights_are_valid's argsort transpose), the empty tuple (no axis reduced), and the forms
                // normalize_axis_tuple refuses (a repeated axis, an out-of-range one).
                Add("null", () => null);
                foreach (var a in new[] { new[] { 0 }, new[] { 1 }, new[] { -1 }, new[] { 0, 1 }, new[] { 1, 0 }, new[] { 1, 2 },
                             new[] { 2, 0 }, Array.Empty<int>(), new[] { 0, 0 }, new[] { 9 } })
                    Add("[" + string.Join(",", a) + "]", () => a.ToArray());
                break;
            case Role.KeyInt:
                // A key inside the pool's values, the ends of the int32 range, a key below and above every value.
                foreach (var k in new[] { 2, 0, -5, 100, int.MaxValue, int.MinValue })
                    Add(k.ToString(CultureInfo.InvariantCulture), () => k);
                break;
            case Role.KeyDouble:
            {
                // A key between the pool's values, a tie, a negative zero, Python's float('nan') (the positive quiet NaN:
                // .NET's double.NaN is negative, and NumPy sorts and compares NaN by value), both infinities, a huge key.
                double nan = BitConverter.Int64BitsToDouble(0x7FF8000000000000);
                foreach (var (label, k) in new[]
                         {
                             ("2.5", 2.5), ("3", 3.0), ("-0", -0.0), ("nan", nan), ("inf", double.PositiveInfinity),
                             ("-inf", double.NegativeInfinity), ("1e300", 1e300),
                         })
                    Add(label, () => k);
                break;
            }
            case Role.Side:
                // The default first, the other name, then the three forms NumPy refuses: a capitalized and a one-letter name
                // (its finalized deprecation's text) and any other string (its general text), the empty one included.
                foreach (var s in new[] { "left", "right", "Left", "l", "x", "" })
                    Add(s.Length == 0 ? "empty" : s, () => s);
                break;
            case Role.Sorter:
                // The base is the default (no sorter); the variants follow from the operand (MakeSorter).
                Add("null", () => null);
                break;
            case Role.Variables:
            case Role.FWeights:
            case Role.AWeights:
                // The base is the default (no second set of variables / no weights); the variants follow from the operand
                // (MakeOperand2 for y, MakeCovWeights for the weights).
                Add("null", () => null);
                break;
            case Role.Ddof:
                // The default (null: bias decides) first, then the values NumPy's normalization meets: 0 and 1 (bias's two
                // answers), 2, the pool's observation count of a (3, 4) operand (a zero factor), one past it (a negative
                // factor, replaced by 0 and so an infinite scale) and a negative ddof (a factor above the count).
                Add("null", () => null);
                foreach (var d in new[] { 0, 1, 2, 4, 5, -1 })
                    Add(d.ToString(CultureInfo.InvariantCulture), () => (int?)d);
                break;
            case Role.AxesList:
                // No axes first, then the forms each gufunc accepts for some operand pair and the forms they refuse:
                // vecdot's two-entry and three-entry lists, matmul's 2-D cores (and x1's core transposed), matvec's and
                // vecmat's mixed cores, vecdot's keepdims output entry, one entry only, an axis out of range, a repeated
                // axis, and an entry of the wrong length.
                Add("null", () => null);
                foreach (var (label, axes) in new (string, int[][])[]
                         {
                             ("vec", new[] { new[] { -1 }, new[] { -1 } }),
                             ("vec3", new[] { new[] { -1 }, new[] { -1 }, Array.Empty<int>() }),
                             ("mat", new[] { new[] { -2, -1 }, new[] { -2, -1 }, new[] { -2, -1 } }),
                             ("matT", new[] { new[] { -1, -2 }, new[] { -2, -1 }, new[] { -2, -1 } }),
                             ("mv", new[] { new[] { -2, -1 }, new[] { -1 }, new[] { -1 } }),
                             ("vm", new[] { new[] { -1 }, new[] { -2, -1 }, new[] { -1 } }),
                             ("keep", new[] { new[] { 0 }, new[] { 0 }, new[] { 0 } }),
                             ("one", new[] { new[] { 0 } }),
                             ("oob", new[] { new[] { 5 }, new[] { 0 }, Array.Empty<int>() }),
                             ("rep", new[] { new[] { 0, 0 }, new[] { -2, -1 }, new[] { -2, -1 } }),
                             ("wrong", new[] { new[] { 0, 1 }, new[] { 0 }, Array.Empty<int>() }),
                         })
                    Add(label, () => axes.Select(e => e.ToArray()).ToArray());
                break;
            case Role.KeepdimsNullable:
                // matmul's keepdims: absent first, then both values (NumPy refuses either: matmul's output has core
                // dimensions).
                Add("null", () => null);
                Add("true", () => (bool?)true);
                Add("false", () => (bool?)false);
                break;
            case Role.Casting:
            {
                // The default first, then the other rules and a name NumPy refuses.
                string c0 = p.HasDefaultValue && p.DefaultValue is string dc ? dc : "same_kind";
                Add(c0, () => c0);
                foreach (var c in new[] { "no", "equiv", "safe", "same_kind", "unsafe", "bogus" })
                    if (c != c0)
                        Add(c, () => c);
                break;
            }
            case Role.GradSpacing:
                // The base is the default (null: no spacing, unit steps); the variants follow from the operand (MakeGradSpacing).
                Add(GradSpacingVariantLabels[0], () => null);
                break;
            case Role.GradAxisSpacing:
                // The base is a unit step for this axis (a C# double: NumPy's Python float); the variants follow from the
                // operand (MakeGradSpacing along this parameter's axis).
                Add(GradAxisSpacingVariantLabels[0], () => 1.0);
                break;
            case Role.GradVarargs:
                // The base is the empty list (no spacing); the variants follow from the operand (MakeGradVarargs).
                Add(GradVarargsVariantLabels[0], () => Array.Empty<object>());
                break;
            case Role.EdgeOrder:
            {
                // The default first, then the other order NumPy computes, and the values its single check lets through
                // (0 and -1 take the second-order branch with a one-element shape check) or refuses (3).
                int e0 = p.HasDefaultValue && p.DefaultValue is int de ? de : 1;
                Add(e0.ToString(CultureInfo.InvariantCulture), () => e0);
                foreach (var e in new[] { 2, 0, 3, -1 })
                    if (e != e0)
                        Add(e.ToString(CultureInfo.InvariantCulture), () => e);
                break;
            }
            case Role.Dx:
            {
                // The default first, then a fraction, a negative spacing, a negative zero (the sign of a zero area), and the
                // two non-finite values (a NaN spacing, an infinite one: inf * 0 is NaN). The NaN is Python's float('nan'),
                // the positive quiet NaN: .NET's double.NaN is negative, and the sign travels into the result's bits.
                double dx0 = p.HasDefaultValue && p.DefaultValue is double ddx ? ddx : 1.0;
                Add(dx0.ToString("R", CultureInfo.InvariantCulture), () => dx0);
                foreach (var (label, dx) in new[]
                         {
                             ("0.5", 0.5), ("-2", -2.0), ("-0", -0.0), ("nan", BitConverter.Int64BitsToDouble(0x7FF8000000000000)),
                             ("inf", double.PositiveInfinity),
                         })
                    if (BitConverter.DoubleToInt64Bits(dx) != BitConverter.DoubleToInt64Bits(dx0))
                        Add(label, () => dx);
                break;
            }
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
            {
                // The default first when there is one (np.trapezoid's -1), else 0, so phase A calls every operand with the
                // axis a caller omits; then both signs of every axis of the pool's ranks and one out of range.
                int a0 = p.HasDefaultValue && p.DefaultValue is int da ? da : 0;
                Add(a0.ToString(CultureInfo.InvariantCulture), () => a0);
                foreach (var a in new[] { 0, 1, -1, 2, -2, -3, 5 })
                    if (a != a0)
                        Add(a.ToString(CultureInfo.InvariantCulture), () => a);
                break;
            }
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
            case Role.LinalgArrays:
                // Built from the operand (MakeLinalgArrays); the factory here is never called.
                Add(LinalgArraysVariantLabels[0], () => null);
                break;
            case Role.Ord:
                // No order first (NumPy's None: the Frobenius / 2-norm default, cond's 2-norm), then the orders NumPy's norms
                // and cond know (Python ints, the infinities as floats, a fraction, the names) and the ones they refuse.
                Add("null", () => null);
                foreach (var (label, ord) in new (string, object)[]
                         {
                             ("1", 1), ("-1", -1), ("2", 2), ("-2", -2), ("0", 0), ("3", 3),
                             ("inf", double.PositiveInfinity), ("-inf", double.NegativeInfinity), ("0.5", 0.5),
                             ("fro", "fro"), ("nuc", "nuc"), ("f", "f"), ("bogus", "bogus"),
                         })
                    Add(label, () => ord);
                break;
            case Role.Uplo:
                // The default first, the other triangle, a lowercase letter (NumPy upper-cases it) and a letter it refuses.
                foreach (var c in new[] { 'L', 'U', 'l', 'X' })
                    Add(c.ToString(), () => c);
                break;
            case Role.QrMode:
                // The default first, the other modes NumPy computes, the spellings NumPy 1.x deprecated and an unknown one.
                foreach (var m in new[] { "reduced", "complete", "r", "raw", "full", "economic", "bogus" })
                    Add(m, () => m);
                break;
            case Role.Power:
                // A required exponent: the square first, then the identity, the operand itself, a cube, the inverse and its
                // square (both need a backend), and a higher power.
                foreach (var e in new[] { 2, 0, 1, 3, -1, -2, 5 })
                    Add(e.ToString(CultureInfo.InvariantCulture), () => e);
                break;
            case Role.AxesPairArg:
                // tensordot's two required axis lists: the last axis of each operand first (the operand and its copy agree),
                // then the first axis, two axes, none (an outer product) and an axis out of range.
                foreach (var a in new[] { new[] { 1 }, new[] { 0 }, new[] { 0, 1 }, Array.Empty<int>(), new[] { 5 } })
                    Add("[" + string.Join(",", a) + "]", () => a.ToArray());
                break;
            case Role.NullableDouble:
                // Unset first (NumPy's default tolerance), then a large and a zero tolerance, a negative one, NumPy's 1e-15,
                // and the two non-finite values (Python's positive NaN: .NET's double.NaN is negative).
                Add("null", () => null);
                foreach (var (label, d) in new[]
                         {
                             ("0.5", 0.5), ("0", 0.0), ("-1", -1.0), ("1e-15", 1e-15),
                             ("nan", BitConverter.Int64BitsToDouble(0x7FF8000000000000)), ("inf", double.PositiveInfinity),
                         })
                    Add(label, () => (double?)d);
                break;
            case Role.InsertObj:
            {
                // One position: before the second element first, then the first, the last (negative), a position inside
                // every axis of the pool, one inside some axes only and its negative, the append position of a (3, 4)
                // operand's flattening (12), one past it and its negative. A long parameter gets the same values as long.
                bool asLong = p.ParameterType == typeof(long);
                foreach (var o in new[] { 1, 0, -1, 3, 4, -4, 12, 13, -13 })
                    Add(o.ToString(CultureInfo.InvariantCulture), asLong ? (Func<object>)(() => (long)o) : () => o);
                break;
            }
            case Role.InsertObjSlice:
                // Two positions first, then the whole axis, every other position, the reversed axis (NumPy's
                // arange(*obj.indices(N)) runs backwards), an empty slice, the last two, a stepped reversed slice and one
                // past the end (empty).
                foreach (var s in new[] { "1:3", ":", "::2", "::-1", "1:1", "-2:", "10:2:-3", "100:" })
                    Add(s, () => new Slice(s));
                break;
            case Role.InsertObjSeq:
            {
                // Two sorted positions first, then none, one (NumPy's single-index branch, without the scalar moveaxis),
                // two unsorted, a repeated one, the first and the last, three, one past a (3, 4) operand's flattening and
                // its negative. A long[] parameter gets the same values as long[].
                bool asLong = p.ParameterType == typeof(long[]);
                foreach (var seq in new[]
                         {
                             new[] { 1, 3 }, Array.Empty<int>(), new[] { 1 }, new[] { 3, 1 }, new[] { 1, 1 }, new[] { 0, -1 },
                             new[] { 1, 2, 3 }, new[] { 13 }, new[] { -13 },
                         })
                    Add("[" + string.Join(",", seq) + "]",
                        asLong ? (Func<object>)(() => seq.Select(v => (long)v).ToArray()) : () => seq.ToArray());
                break;
            }
            case Role.InsertObjArray:
                // An int64 pair first, then a 0-d index (NumPy's scalar branch), a one-element index (the single-index
                // branch without the moveaxis), unsorted int32 and uint8 indices, a boolean mask of three (positions 0
                // and 2), and the forms NumPy refuses or reads otherwise: two-dimensional masks and indices, one and two
                // float indices, text indices of one and two elements, a complex index, an empty index array and one past
                // the end.
                Add("i64[1,3]", () => np.array(new long[] { 1, 3 }));
                Add("0d_i64_1", () => np.array(1L));
                Add("i64[1]", () => np.array(new long[] { 1 }));
                Add("i32[3,1]", () => np.array(new[] { 3, 1 }));
                Add("u8[0,2]", () => np.array(new byte[] { 0, 2 }));
                Add("bool[T,F,T]", () => np.array(new[] { true, false, true }));
                Add("bool2d", () => np.array(new[] { true, false }).reshape(new Shape(1, 2)));
                Add("i64_2d", () => np.array(new long[] { 1 }).reshape(new Shape(1, 1)));
                Add("f64[1]", () => np.array(new[] { 1.0 }));
                Add("f64[1,2]", () => np.array(new[] { 1.0, 2.0 }));
                Add("U[1]", () => np.array(new[] { "1" }));
                Add("U[1,2]", () => np.array(new[] { "1", "2" }));
                Add("c128[1]", () => np.array(new[] { new Complex(1, 0) }));
                Add("i64[]", () => np.empty(new Shape(0), np.dtype("int64")));
                Add("i64[13]", () => np.array(new long[] { 13 }));
                break;
            case Role.InsertValues:
                // A 0-d int64 first (one value, broadcast), then two int64 values, two float64 values, a 0-d float64, a
                // column of two, text values of both classes (a numeric operand parses them), a 0-d bool, an empty array,
                // and None (null).
                Add("0d_i64_7", () => np.array(7L));
                Add("i64[7,8]", () => np.array(new long[] { 7, 8 }));
                Add("f64[1.5,2.5]", () => np.array(new[] { 1.5, 2.5 }));
                Add("0d_f64", () => np.array(1.5));
                Add("col[[7],[8]]", () => np.array(new long[] { 7, 8 }).reshape(new Shape(2, 1)));
                Add("U[7]", () => np.array(new[] { "7" }));
                Add("T[7]", () => np.array(new[] { "7" }).astype(np.dtype("T")));
                Add("0d_bool", () => np.array(true));
                Add("f64[]", () => np.empty(new Shape(0), np.dtype("float64")));
                Add("null", () => null);
                break;
            case Role.InsertValue:
                // A C# int first (NumPy's Python int), then a double, a string, a bool, null (None), an int out of the
                // small integer ranges, a C# int[] (a Python list), a mixed object[] and an NDArray passed as object.
                foreach (var (label, value) in new (string, Func<object>)[]
                         {
                             ("7", () => 7), ("1.5", () => 1.5), ("'7'", () => "7"), ("true", () => true),
                             ("null", () => null), ("1000", () => 1000), ("int[7,8]", () => new[] { 7, 8 }),
                             ("obj[7,'x']", () => new object[] { 7, "x" }), ("nd[7,8]", () => np.array(new long[] { 7, 8 })),
                         })
                    Add(label, value);
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

    /// <summary>The 1-D layouts a search function's haystack needs: sorted values (np.sort of the pool's prefix, the
    ///     special ones with NaN last), every other element of a sorted run (a strided haystack) and a descending run (a
    ///     reversed view, which NumPy's binary search reads as unsorted).</summary>
    private static readonly string[] SortedLayouts = { "sorted1d", "sorted1d_special", "strided1d", "reversed1d" };

    /// <summary>The reduced pool of a search function (phase B): the 1-D layouts, the unsorted 1-D and empty 1-D operands,
    ///     and one 2-D operand for the haystack's rank error.</summary>
    private static readonly string[] SortedReducedLayouts = { "sorted1d", "1d", "empty1d", "sorted1d_special", "strided1d", "reversed1d", "c2d" };

    /// <summary>The small pool of a search function (phase C).</summary>
    private static readonly string[] SortedPairLayouts = { "sorted1d", "empty1d", "c2d" };

    /// <summary>
    ///     The full operand pool of an np.linalg overload (phase A). Most of np.linalg takes a square matrix or a stack of
    ///     them and refuses the general pool's (3, 4) operands at its first check, so the pool is built around square ones:
    ///     a (3, 3) matrix in C order (<c>sq3</c>, also with the special values), F order (<c>sq3f</c>), transposed
    ///     (<c>sq3t</c>), every other column of a (3, 6) (<c>sq3s</c>), reversed along both axes (<c>sq3r</c>), broadcast
    ///     from one row (<c>bcast3</c>: read-only and singular), the (2, 2) and (1, 1) matrices, a (2, 3, 3) stack, a
    ///     symmetric (Hermitian for complex) positive definite matrix (<c>spd</c>: the input cholesky and eigh need), an
    ///     exactly singular one (<c>singular</c>: LU meets an exact zero pivot, so solve and inv raise), the rectangles
    ///     (3, 4) and (4, 3) (lstsq, qr, svd, pinv, matrix_rank, the norms), the empty square (0, 0), an empty stack
    ///     (0, 3, 3), and the general pool's 1-D, 0-d, (0, 3) and (2, 3, 4) operands for the rank and shape checks.
    /// </summary>
    private static readonly string[] LinalgLayouts =
    {
        "sq3", "sq3_special", "sq3f", "sq3t", "sq3s", "sq3r", "bcast3", "sq2", "sq1", "stack", "spd", "singular", "c2d",
        "rect_t", "empty00", "emptybatch", "1d", "0d", "empty", "3d",
    };

    /// <summary>The reduced pool of an np.linalg overload (phase B): a square matrix in both orders, a stack, the
    ///     positive definite matrix, a rectangle, the empty square and a 0-d operand.</summary>
    private static readonly string[] LinalgReducedLayouts = { "sq3", "sq3f", "stack", "spd", "c2d", "empty00", "0d" };

    /// <summary>The small pool of an np.linalg overload (phase C): a square matrix and a rectangle.</summary>
    private static readonly string[] LinalgPairLayouts = { "sq3", "c2d" };

    /// <summary>The reduced pool of np.convolve / np.correlate (phase B): the 1-D layouts (plain, sorted, special, strided,
    ///     reversed and the 24-element one, whose float products reach the vector regimes of NumPy's dot functions), a 0-d
    ///     operand (np.convolve makes it one element, np.correlate refuses it), an empty 1-D one and a 2-D one (too
    ///     deep).</summary>
    private static readonly string[] SlidingReducedLayouts =
        { "1d", "sorted1d", "sorted1d_special", "strided1d", "reversed1d", "long1d", "0d", "empty1d", "c2d" };

    /// <summary>The small pool of np.convolve / np.correlate (phase C).</summary>
    private static readonly string[] SlidingPairLayouts = { "1d", "long1d", "0d", "c2d" };

    /// <summary>
    ///     The operand layouts of the three phases for an overload: the general pools; for np.searchsorted /
    ///     NDArray.searchsorted, pools of 1-D haystacks (every other operand of the general pools fails NumPy's 1-D check
    ///     before anything else is read, so the varied keys, sides and sorters would never reach the search); for
    ///     np.convolve / np.correlate and their NDArray members, the same 1-D layouts and a 24-element one (a general pool's
    ///     2-D operand fails their depth check first); for np.linalg, pools of square matrices, stacks and rectangles
    ///     (<see cref="LinalgLayouts"/>).
    /// </summary>
    /// <param name="m">The overload.</param>
    /// <returns>The layouts of phase A, phase B and phase C.</returns>
    private static (string[] Full, string[] Reduced, string[] Pair) PoolsFor(MethodInfo m) =>
        m.Name == "searchsorted"
            ? (Layouts.Concat(SortedLayouts).ToArray(), SortedReducedLayouts, SortedPairLayouts)
            : m.Name is "convolve" or "correlate" && (m.DeclaringType == typeof(np) || m.DeclaringType == typeof(NDArray))
                ? (Layouts.Concat(SortedLayouts).Append("long1d").ToArray(), SlidingReducedLayouts, SlidingPairLayouts)
                : m.DeclaringType == typeof(np.linalg)
                    ? (LinalgLayouts, LinalgReducedLayouts, LinalgPairLayouts)
                    : (Layouts, ReducedLayouts, PairLayouts);

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
            // np.histogramdd's sequence form: its NDArray[] sample is built from the pool operand (its columns), and the
            // NDArray parameters after it (weights) keep their own roles.
            if (!sawArray && ps[i].ParameterType == typeof(NDArray[]) && ps[i].Name == "sample")
            {
                roles[i] = Role.SampleSeq;
                sawArray = true;
                continue;
            }
            // np.linalg.multi_dot's array list likewise carries the operand (MakeLinalgArrays), so its out= after it is an
            // out= and not the operand.
            if (!sawArray && ps[i].ParameterType == typeof(NDArray[]) && ps[i].Name == "arrays")
            {
                roles[i] = Role.LinalgArrays;
                sawArray = true;
                continue;
            }
            // np.nditer's multi-operand list carries the operand too (MakeNditerOps).
            if (!sawArray && ps[i].ParameterType == typeof(NDArray[]) && ps[i].Name == "op" && IsNditer(ps[i]))
            {
                roles[i] = Role.NditerOps;
                sawArray = true;
                continue;
            }
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
            Role.Spacing => SpacingVariantLabels.Length - 1,
            Role.Keys => KeysVariantLabels.Length - 1,
            Role.SlidingKernel => SlidingKernelVariantLabels.Length - 1,
            Role.IsinTest => IsinTestVariantLabels.Length - 1,
            Role.AverageWeights => AverageWeightVariantLabels.Length - 1,
            Role.Sorter => SorterVariantLabels.Length - 1,
            Role.Variables => VariablesVariantLabels.Length - 1,
            Role.FWeights => FWeightVariantLabels.Length - 1,
            Role.AWeights => AWeightVariantLabels.Length - 1,
            Role.GradSpacing => GradSpacingVariantLabels.Length - 1,
            Role.GradAxisSpacing => GradAxisSpacingVariantLabels.Length - 1,
            Role.GradVarargs => GradVarargsVariantLabels.Length - 1,
            Role.LinalgArrays => LinalgArraysVariantLabels.Length - 1,
            Role.NditerOps => NditerOpsVariantLabels.Length - 1,
            Role.NditerOpAxes => NditerOpAxesVariantLabels.Length - 1,
            Role.NditerItershape => NditerItershapeVariantLabels.Length - 1,
            Role.SampleSeq or Role.BinsInts or Role.BinsArrays or Role.BinsObjects or Role.RangeArray => DdLabels(roles[i]).Length - 1,
            _ => variants[i].Count - 1,
        };

        // The operand layouts of the three phases: the general pools, or the 1-D pools of a search function (below).
        var (fullLayouts, reducedLayouts, pairLayouts) = PoolsFor(m);

        // phase A: every operand, every other parameter at its base
        foreach (var dt in Dtypes)
            foreach (var layout in fullLayouts)
                yield return Call(m, signature, instance, matrix, operandIndex, dt, layout, variants, roles, none);

        // phase B: one parameter varied at a time, on the reduced pool
        for (int p = 0; p < ps.Length; p++)
            for (int v = 1; v <= Count(p); v++)
                foreach (var dt in Dtypes)
                    foreach (var layout in reducedLayouts)
                        yield return Call(m, signature, instance, matrix, operandIndex, dt, layout, variants, roles, new[] { (p, v) });

        // phase C: every pair of parameters varied together, on the small pool
        for (int p1 = 0; p1 < ps.Length; p1++)
            for (int p2 = p1 + 1; p2 < ps.Length; p2++)
                for (int v1 = 1; v1 <= Count(p1); v1++)
                    for (int v2 = 1; v2 <= Count(p2); v2++)
                        foreach (var dt in PairDtypes)
                            foreach (var layout in pairLayouts)
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
    private static readonly string[] Operand2VariantLabels = { "same", "int32", "float64", "U5", "T", "0d", "row", "badshape", "transposed" };

    /// <summary>np.trapezoid's sample-point variants, built from the operand (see <see cref="MakeSpacing"/>); the first,
    ///     null, is the base.</summary>
    private static readonly string[] SpacingVariantLabels =
    {
        "null", "same", "int32", "float64", "U5", "T", "0d", "row", "col", "decr", "badlen", "bool", "complex128", "float16",
        "special", "highrank", "f", "bcast",
    };

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
            Role.Spacing => SpacingVariantLabels[v],
            Role.Keys => KeysVariantLabels[v],
            Role.SlidingKernel => SlidingKernelVariantLabels[v],
            Role.IsinTest => IsinTestVariantLabels[v],
            Role.AverageWeights => AverageWeightVariantLabels[v],
            Role.Sorter => SorterVariantLabels[v],
            Role.Variables => VariablesVariantLabels[v],
            Role.FWeights => FWeightVariantLabels[v],
            Role.AWeights => AWeightVariantLabels[v],
            Role.GradSpacing => GradSpacingVariantLabels[v],
            Role.GradAxisSpacing => GradAxisSpacingVariantLabels[v],
            Role.GradVarargs => GradVarargsVariantLabels[v],
            Role.LinalgArrays => LinalgArraysVariantLabels[v],
            Role.NditerOps => NditerOpsVariantLabels[v],
            Role.NditerOpAxes => NditerOpAxesVariantLabels[v],
            Role.NditerItershape => NditerItershapeVariantLabels[v],
            Role.SampleSeq or Role.BinsInts or Role.BinsArrays or Role.BinsObjects or Role.RangeArray => DdLabels(roles[p])[v],
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
                    : roles[i] == Role.Keys ? MakeKeys(KeysVariantLabels[0], operand)
                    : roles[i] == Role.SlidingKernel ? MakeSlidingKernel(SlidingKernelVariantLabels[0], operand)
                    : roles[i] == Role.IsinTest ? MakeIsinTest(IsinTestVariantLabels[0], operand)
                    : roles[i] == Role.LinalgArrays ? MakeLinalgArrays(LinalgArraysVariantLabels[0], operand)
                    : roles[i] == Role.NditerOps ? MakeNditerOps(NditerOpsVariantLabels[0], operand)
                    : IsDdRole(roles[i]) ? MakeDd(roles[i], DdLabels(roles[i])[0], operand, m)
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
                    case Role.Spacing:
                        argv[p] = MakeSpacing(SpacingVariantLabels[v], operand);
                        break;
                    case Role.Keys:
                        argv[p] = MakeKeys(KeysVariantLabels[v], operand);
                        break;
                    case Role.SlidingKernel:
                        argv[p] = MakeSlidingKernel(SlidingKernelVariantLabels[v], operand);
                        break;
                    case Role.IsinTest:
                        argv[p] = MakeIsinTest(IsinTestVariantLabels[v], operand);
                        break;
                    case Role.AverageWeights:
                        argv[p] = MakeAverageWeights(AverageWeightVariantLabels[v], operand);
                        break;
                    case Role.Sorter:
                        argv[p] = MakeSorter(SorterVariantLabels[v], operand);
                        break;
                    case Role.Variables:
                        argv[p] = MakeOperand2(VariablesVariantLabels[v], operand);
                        break;
                    case Role.FWeights:
                        argv[p] = MakeCovWeights(frequency: true, FWeightVariantLabels[v], operand);
                        break;
                    case Role.AWeights:
                        argv[p] = MakeCovWeights(frequency: false, AWeightVariantLabels[v], operand);
                        break;
                    case Role.GradSpacing:
                        argv[p] = MakeGradSpacing(GradSpacingVariantLabels[v], operand, axis: 0);
                        break;
                    case Role.GradAxisSpacing:
                        // spacing0 / spacing1 / spacing2: the spacing of axis 0 / 1 / 2, so coordinates take that axis' length.
                        argv[p] = MakeGradSpacing(GradAxisSpacingVariantLabels[v], operand, axis: ps[p].Name[^1] - '0');
                        break;
                    case Role.GradVarargs:
                        argv[p] = MakeGradVarargs(GradVarargsVariantLabels[v], operand);
                        break;
                    case Role.LinalgArrays:
                        argv[p] = MakeLinalgArrays(LinalgArraysVariantLabels[v], operand);
                        break;
                    case Role.NditerOps:
                        argv[p] = MakeNditerOps(NditerOpsVariantLabels[v], operand);
                        break;
                    case Role.NditerOpAxes:
                        argv[p] = MakeNditerOpAxes(NditerOpAxesVariantLabels[v], operand);
                        break;
                    case Role.NditerItershape:
                        argv[p] = MakeNditerItershape(NditerItershapeVariantLabels[v], operand);
                        break;
                    case Role.SampleSeq:
                    case Role.BinsInts:
                    case Role.BinsArrays:
                    case Role.BinsObjects:
                    case Role.RangeArray:
                        argv[p] = MakeDd(roles[p], DdLabels(roles[p])[v], operand, m);
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
            // A static member has no receiver; an instance member of a module class (np.emath's) is called on its module
            // object, with the operand among the arguments.
            return m.Invoke(instance ? operand : m.IsStatic ? null : ModuleReceiver(m), argv);
        }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerException).Throw();
            throw;
        }
    }

    /// <summary>The module object an instance member of a module class is called on.</summary>
    /// <param name="m">An instance member declared on a module class.</param>
    /// <returns><c>np.emath</c> for a member of <see cref="EmathModule"/>.</returns>
    /// <exception cref="ArgumentException">A module class the tool does not know: its module object would have to be named
    ///     here first.</exception>
    private static object ModuleReceiver(MethodInfo m) =>
        m.DeclaringType == typeof(EmathModule)
            ? np.emath
            : throw new ArgumentException($"no module object for {m.DeclaringType}");

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
            // The search functions' haystacks (SortedLayouts). np.sort of a dtype it refuses leaves the operand unavailable.
            case "sorted1d": return SortedOrNull(Values(6));
            case "strided1d": return SortedOrNull(Values(12))?["::2"];
            case "reversed1d": return SortedOrNull(Values(6))?["::-1"];
            // np.convolve / np.correlate's long sequence: the whole 24-value prefix, so a product of it with a long kernel
            // reaches the vector regimes of NumPy's dot functions (ddot from 16 terms, zdotu from 8).
            case "long1d": return Values(24);
            // np.linalg's square matrices, stacks and rectangles (LinalgLayouts). The plain prefixes give nonsingular
            // matrices (sq3's determinant is 48, sq3s's 1, sq2's 2, the stack's 48 and -472).
            case "sq3": return Values(9).reshape(new Shape(3, 3));
            case "sq3f": return np.asfortranarray(Values(9).reshape(new Shape(3, 3)));
            case "sq3t": return Values(9).reshape(new Shape(3, 3)).T;
            case "sq3s": return Values(18).reshape(new Shape(3, 6))[":, ::2"];
            case "sq3r": return Values(9).reshape(new Shape(3, 3))["::-1, ::-1"];
            case "bcast3": return np.broadcast_to(Values(3).reshape(new Shape(1, 3)), new Shape(3, 3));
            case "sq2": return Values(4).reshape(new Shape(2, 2));
            case "sq1": return Values(1).reshape(new Shape(1, 1));
            case "stack": return Values(18).reshape(new Shape(2, 3, 3));
            case "spd": return BuildFrom(dt, SpdValues, SpdComplexValues).reshape(new Shape(3, 3));
            case "singular": return BuildFrom(dt, SingularValues, SingularComplexValues).reshape(new Shape(3, 3));
            case "rect_t": return Values(12).reshape(new Shape(3, 4)).T;
            case "empty00": return Values(0).reshape(new Shape(0, 0));
            case "emptybatch": return Values(0).reshape(new Shape(0, 3, 3));
            default: throw new ArgumentException("unknown layout " + layout);
        }
    }

    /// <summary>The symmetric positive definite (3, 3) matrix of the <c>spd</c> layout, row by row (its leading minors
    ///     are 4, 11 and 18).</summary>
    private static readonly double[] SpdValues = { 4, 1, 0, 1, 3, 1, 0, 1, 2 };

    /// <summary>The Hermitian positive definite (3, 3) matrix of the <c>spd</c> layout's complex128 operand (its leading
    ///     minors are 4, 10 and 16): eigh and cholesky read one triangle, so both triangles hold the conjugate pair.</summary>
    private static readonly Complex[] SpdComplexValues =
    {
        new(4, 0), new(1, -1), new(0, 0),
        new(1, 1), new(3, 0), new(0, 1),
        new(0, 0), new(0, -1), new(2, 0),
    };

    /// <summary>The exactly singular (3, 3) matrix of the <c>singular</c> layout: its second row is twice the first, so
    ///     partial-pivoting LU eliminates the first row to exact zeros (every multiplier is a power of two) and meets an
    ///     exact zero pivot, the case solve and inv refuse.</summary>
    private static readonly double[] SingularValues = { 1, 2, 3, 2, 4, 6, 1, 0, 1 };

    /// <summary>The complex128 form of the <c>singular</c> layout: the first two rows scaled by 1+1j (the second still twice
    ///     the first; the multipliers 0.5 and 0.25-0.25j stay exact), the third row real.</summary>
    private static readonly Complex[] SingularComplexValues =
    {
        new(1, 1), new(2, 2), new(3, 3),
        new(2, 2), new(4, 4), new(6, 6),
        new(1, 0), new(0, 0), new(1, 0),
    };

    /// <summary>
    ///     A fresh 1-D array of a dtype holding given values: the operand of a layout whose values matter (a positive
    ///     definite or a singular matrix), where a prefix of the plain sequence would not do.
    /// </summary>
    /// <param name="dt">The dtype name.</param>
    /// <param name="values">The values (small non-negative integers, so every numeric dtype holds them exactly).</param>
    /// <param name="complexValues">The complex128 operand's values, of the same length.</param>
    /// <returns>A C-contiguous 1-D array: text dtypes hold each value's invariant text, char the letter <c>'a' + |v|</c>
    ///     (<see cref="Build"/>'s char mapping), decimal and the numeric dtypes the values.</returns>
    private static NDArray BuildFrom(string dt, double[] values, Complex[] complexValues)
    {
        switch (dt)
        {
            case "S5":
            case "U5":
            case "T":
                return np.array(values.Select(v => v.ToString(CultureInfo.InvariantCulture)).ToArray()).astype(np.dtype(dt));
            case "decimal":
                return np.array(values.Select(v => (decimal)v).ToArray());
            case "complex128":
                return np.array(complexValues);
            case "char":
                return np.array(values.Select(v => (char)('a' + (int)Math.Abs(v))).ToArray());
            default:
                return np.array(values).astype(np.dtype(dt));
        }
    }

    /// <summary>np.sort of a 1-D array, or null when NumSharp's np.sort refuses its dtype.</summary>
    /// <param name="values">The values.</param>
    /// <returns>A sorted copy (NaN last), or null.</returns>
    private static NDArray SortedOrNull(NDArray values)
    {
        try
        {
            return np.sort(values);
        }
        catch (NotSupportedException)
        {
            return null;
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
    ///     broadcasts), <c>badshape</c> (seven elements, which no operand of this pool broadcasts against), <c>transposed</c>
    ///     (a copy of the operand transposed: a (3, 4) operand's (4, 3) partner, which a matrix product accepts).</param>
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
            case "transposed": return operand.T.copy();
            default: throw new ArgumentException("unknown second-operand variant " + label);
        }
    }

    /// <summary>
    ///     Builds np.trapezoid's sample points for an operand: N-D points of its shape (one per sample, differenced along the
    ///     axis), 1-D points along one axis (broadcast over the others), and the forms NumPy refuses.
    /// </summary>
    /// <param name="label">The variant: <c>same</c> (a copy of the operand: points of its own dtype, text for a text
    ///     operand), <c>int32</c> / <c>float64</c> / <c>float16</c> / <c>complex128</c> (increasing points of the operand's
    ///     shape with uneven steps), <c>U5</c> / <c>T</c> (text of the operand's shape: diff's subtract refuses it),
    ///     <c>0d</c> (a 0-d 2.0: diff's "at least one dimensional"), <c>row</c> / <c>col</c> / <c>decr</c> (1-D points as long
    ///     as the last / the first axis; decreasing ones integrate backwards), <c>badlen</c> (1-D points of seven, which no
    ///     axis of the pool has), <c>bool</c> (bool points of the operand's shape: NumPy's diff is a not_equal),
    ///     <c>special</c> (float64 points of the operand's shape with NaN, signed zeros and infinities), <c>highrank</c>
    ///     (points with one more leading axis: the product broadcasts to it), <c>f</c> (float64 points of the operand's
    ///     shape in F order: they vote on the reduction input's layout), <c>bcast</c> (1-D points broadcast to the operand's
    ///     shape, a stride-0 view, which abstains from the vote on the axes it repeats).</param>
    /// <param name="operand">The operand.</param>
    /// <returns>A fresh array, or null for the base label.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray MakeSpacing(string label, NDArray operand)
    {
        long[] dims = operand.Shape.Dimensions;
        long size = 1;
        foreach (long d in dims) size *= d;
        long last = dims.Length == 0 ? 1 : dims[^1];
        long first = dims.Length == 0 ? 1 : dims[0];
        var shape = new Shape(dims);
        // Increasing points with uneven steps (0, 0.5, 1.5, 2, 3, 3.5, ...: steps 0.5 and 1 in turn), so the products of
        // the spacing and the sums differ from one interval to the next.
        NDArray Points(long count) =>
            np.array(Enumerable.Range(0, (int)count).Select(i => i * 0.75 - (i % 2 == 0 ? 0.0 : 0.25)).ToArray());
        NDArray Text(string dt) => size == 0
            ? np.empty(shape, np.dtype(dt))
            : np.array(Enumerable.Range(0, (int)size).Select(i => (i % 3).ToString(CultureInfo.InvariantCulture)).ToArray())
                .astype(np.dtype(dt)).reshape(shape);
        switch (label)
        {
            case "null": return null;
            case "same": return operand.copy();
            case "int32":
                return np.array(Enumerable.Range(0, (int)size).Select(i => (long)(2 * i - i % 3)).ToArray()).astype(np.dtype("int32")).reshape(shape);
            case "float64": return Points(size).reshape(shape);
            case "U5": return Text("U5");
            case "T": return Text("T");
            case "0d": return np.array(2.0);
            case "row": return Points(last);
            case "col": return Points(first);
            case "decr": return np.array(Enumerable.Range(0, (int)last).Select(i => 4.0 - i * 0.75 - (i % 2 == 0 ? 0.0 : 0.25)).ToArray());
            case "badlen": return Points(7);
            case "bool": return np.array(Enumerable.Range(0, (int)size).Select(i => i % 3 != 1).ToArray()).reshape(shape);
            case "complex128":
                return np.array(Enumerable.Range(0, (int)size).Select(i => new Complex(i * 0.75, (i % 3) * 0.5)).ToArray()).reshape(shape);
            case "float16": return Points(size).astype(np.dtype("float16")).reshape(shape);
            case "special": return np.array(Special.Take((int)size).ToArray()).reshape(shape);
            case "highrank": return Points(2 * size).reshape(new Shape(new long[] { 2 }.Concat(dims).ToArray()));
            case "f": return np.asfortranarray(Points(size).reshape(shape));
            case "bcast": return dims.Length == 0 ? np.array(2.0) : np.broadcast_to(Points(last), shape);
            default: throw new ArgumentException("unknown sample-point variant " + label);
        }
    }

    /// <summary>np.searchsorted's key variants, built from the operand (see <see cref="MakeKeys"/>); the first is the
    ///     base.</summary>
    private static readonly string[] KeysVariantLabels =
    {
        "same", "int32", "float64", "frac", "U5", "T", "S5", "0d", "0d_f", "2d", "empty", "special", "bool", "complex", "row",
    };

    /// <summary>np.searchsorted's sorter variants, built from the operand (see <see cref="MakeSorter"/>); the first, null,
    ///     is the base.</summary>
    private static readonly string[] SorterVariantLabels =
        { "null", "argsort", "int32", "uint64", "float", "bool", "2d", "short", "oob", "neg", "reversed" };

    /// <summary>
    ///     Builds np.searchsorted's keys for an operand: the haystack's own values, keys of other dtypes and shapes around
    ///     the pool's values, and text.
    /// </summary>
    /// <param name="label">The variant: <c>same</c> (a copy of the operand: keys of its dtype, every value a tie),
    ///     <c>int32</c> / <c>float64</c> (seven keys from -6 to 6 / -3 to 3, around the pool's values), <c>frac</c> (seven
    ///     float64 keys between them), <c>U5</c> / <c>T</c> / <c>S5</c> (seven text keys: numbers, a letter and the empty
    ///     string), <c>0d</c> / <c>0d_f</c> (a 0-d int64 2 / float64 2.5: a scalar result), <c>2d</c> (a (2, 3) float64
    ///     block), <c>empty</c> (no key), <c>special</c> (float64 keys with NaN, signed zeros and infinities), <c>bool</c>,
    ///     <c>complex</c>, <c>row</c> (an int32 ramp as long as the operand's last axis).</param>
    /// <param name="operand">The operand (the haystack).</param>
    /// <returns>A fresh array.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray MakeKeys(string label, NDArray operand)
    {
        long[] dims = operand.Shape.Dimensions;
        string[] text = { "-1", "0", "2", "3", "10", "x", "" };
        switch (label)
        {
            case "same": return operand.copy();
            case "int32": return np.array(Enumerable.Range(0, 7).Select(i => 2 * i - 6).ToArray());
            case "float64": return np.array(Enumerable.Range(0, 7).Select(i => i - 3.0).ToArray());
            case "frac": return np.array(new[] { -4.5, -0.5, 0.5, 2.5, 4.5, 5.5, 13.5 });
            case "U5": return np.array(text).astype(np.dtype("U5"));
            case "T": return np.array(text).astype(np.dtype("T"));
            case "S5": return np.array(text).astype(np.dtype("S5"));
            case "0d": return np.array(2L);
            case "0d_f": return np.array(2.5);
            case "2d": return np.array(Enumerable.Range(0, 6).Select(i => i - 3.0).ToArray()).reshape(new Shape(2, 3));
            case "empty": return np.empty(new Shape(0), np.dtype("float64"));
            case "special": return np.array(Special.Take(7).ToArray());
            case "bool": return np.array(new[] { true, false, true });
            case "complex": return np.array(new[] { new Complex(1, 1), new Complex(-1, 0), new Complex(2, -1) });
            case "row":
                return np.array(Enumerable.Range(0, (int)(dims.Length == 0 ? 1 : dims[^1])).Select(i => i % 4).ToArray());
            default: throw new ArgumentException("unknown key variant " + label);
        }
    }

    /// <summary>np.convolve / np.correlate's second-sequence variants, built from the operand (see
    ///     <see cref="MakeSlidingKernel"/>); the first is the base.</summary>
    private static readonly string[] SlidingKernelVariantLabels =
    {
        "same", "int32", "float64", "short", "int8", "uint64", "float16", "complex", "one", "one_c", "one_c_rev", "long",
        "special", "bool", "U5", "T", "S5", "0d", "2d", "empty", "strided", "reversed",
    };

    /// <summary>
    ///     Builds np.convolve / np.correlate's second sequence for an operand: its own values, sequences of other dtypes
    ///     (the result dtype is their promotion) and lengths (longer than the operand: NumPy swaps the operands, and
    ///     np.correlate reverses its result), the one-element arrays whose stride decides NumPy's complex dot route, special
    ///     values, text, and the shapes NumPy refuses.
    /// </summary>
    /// <param name="label">The variant: <c>same</c> (a copy of the operand: equal lengths), <c>int32</c> / <c>float64</c>
    ///     (seven values -6 to 6 / -3 to 3: one longer than the pool's 1-D operands), <c>short</c> (float64 1.5, -2: a
    ///     kernel shorter than the data, so the ramps of every mode), <c>int8</c> / <c>uint64</c> / <c>float16</c> (three
    ///     values: small-integer promotion, int64 with uint64 giving float64, the HALF loop), <c>complex</c> (three complex
    ///     values: np.correlate conjugates them), <c>one</c> (one float64 element), <c>one_c</c> (one fresh complex element:
    ///     a positive stride, which np.convolve's reversal makes negative) and <c>one_c_rev</c> (the reversed view of one
    ///     complex element: a negative stride, which the reversal makes positive) — NumPy's complex dot takes cblas only for
    ///     positive strides, and its result rule differs for non-finite values —, <c>long</c> (twenty float64 values with
    ///     fractional parts: beside the 24-element operand a dot of up to twenty terms, past ddot's and zdotu's scalar
    ///     regimes), <c>special</c> (seven float64 values with NaN, signed zeros and infinities), <c>bool</c>, <c>U5</c> /
    ///     <c>T</c> / <c>S5</c> (seven text values), <c>0d</c> (a 0-d int64 2: np.convolve makes it one element,
    ///     np.correlate refuses it), <c>2d</c> (a (2, 3) float64 block: too deep), <c>empty</c> (no element),
    ///     <c>strided</c> / <c>reversed</c> (every other element of 0..13 / 6 down to 0, as views: NumPy copies a
    ///     non-contiguous operand into a fresh array).</param>
    /// <param name="operand">The operand (the first sequence).</param>
    /// <returns>A fresh array, or a view of one (<c>one_c_rev</c>, <c>strided</c>, <c>reversed</c>).</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray MakeSlidingKernel(string label, NDArray operand)
    {
        string[] text = { "-1", "0", "2", "3", "10", "x", "" };
        switch (label)
        {
            case "same": return operand.copy();
            case "int32": return np.array(Enumerable.Range(0, 7).Select(i => 2 * i - 6).ToArray());
            case "float64": return np.array(Enumerable.Range(0, 7).Select(i => i - 3.0).ToArray());
            case "short": return np.array(new[] { 1.5, -2.0 });
            case "int8": return np.array(new sbyte[] { 1, -2, 3 });
            case "uint64": return np.array(new ulong[] { 1, 2, 3 });
            case "float16": return np.array(new[] { (Half)0.5, (Half)(-1.5), (Half)2.0 });
            case "complex": return np.array(new[] { new Complex(1, 1), new Complex(-1, 0), new Complex(2, -1) });
            case "one": return np.array(new[] { 2.5 });
            case "one_c": return np.array(new[] { new Complex(1, 2) });
            case "one_c_rev": return np.array(new[] { new Complex(1, 2) })["::-1"];
            case "long": return np.array(Enumerable.Range(0, 20).Select(i => i % 7 - 3.0 + 0.25 * (i % 3)).ToArray());
            case "special": return np.array(Special.Take(7).ToArray());
            case "bool": return np.array(new[] { true, false, true });
            case "U5": return np.array(text).astype(np.dtype("U5"));
            case "T": return np.array(text).astype(np.dtype("T"));
            case "S5": return np.array(text).astype(np.dtype("S5"));
            case "0d": return np.array(2L);
            case "2d": return np.array(Enumerable.Range(0, 6).Select(i => i - 3.0).ToArray()).reshape(new Shape(2, 3));
            case "empty": return np.empty(new Shape(0), np.dtype("float64"));
            case "strided": return np.arange(14.0)["::2"];
            case "reversed": return np.arange(7.0)["::-1"];
            default: throw new ArgumentException("unknown sliding kernel variant " + label);
        }
    }

    /// <summary>np.isin's test-set variants, built from the operand (see <see cref="MakeIsinTest"/>); the first is the
    ///     base.</summary>
    private static readonly string[] IsinTestVariantLabels =
    {
        "same", "int32", "float64", "U5", "T", "S5", "0d", "empty", "long", "widelong", "wide", "nan", "nanlong", "u64",
        "u64long", "bool", "complex", "bcast", "strided",
    };

    /// <summary>
    ///     Builds np.isin's test set for an operand: its own values, and sets that send NumPy's <c>_isin</c> down each of its
    ///     paths. For an integer or bool pair NumPy takes the table method while the set's range is at most six times the two
    ///     sizes; otherwise (and for every other dtype) the loop path while <c>len(ar2) &lt; 10 * len(ar1) ** 0.145</c>
    ///     (fewer than 10 to 15 values for this pool), else the sort path.
    /// </summary>
    /// <param name="label">The variant: <c>same</c> (a copy of the operand: every element a member), <c>int32</c> /
    ///     <c>float64</c> (a ramp 0, 1, 2, 3, 0, ... of the operand's shape: repeated values), <c>U5</c> / <c>T</c> / <c>S5</c>
    ///     (text "0", "1", "2", ... of the operand's shape: a text set against a number, str against bytes), <c>0d</c> (a 0-d
    ///     int64 2), <c>empty</c> (no value), <c>long</c> (forty int64 values -3 to 3, repeated: a narrow range, the table
    ///     method for an integer operand and the sort path for another), <c>widelong</c> (forty int64 values -3, 997, 1997,
    ///     ...: a range too wide for the table method, so the sort path), <c>wide</c> (int64 0, 1000000, 3, -1: too wide for
    ///     the table method, short enough for the loop path), <c>nan</c> (float64 NaN, -0.0, 3, inf, -1: the loop path),
    ///     <c>nanlong</c> (forty float64 values with NaN, signed zeros and infinities: the sort path), <c>u64</c> (uint64 3,
    ///     2**63 + 1, 7, 0: a signed operand meets uint64 on the loop path), <c>u64long</c> (forty uint64 values, twenty past
    ///     2**63: the sort path, where NumPy concatenates a signed operand with them in float64), <c>bool</c> ([True]),
    ///     <c>complex</c> ([1+0j, 3+0j, 2+1j]), <c>bcast</c> (int64 [1, 3] broadcast to (3, 2): a stride-0 set),
    ///     <c>strided</c> (int64 0, 3, 6, ..., 18: a stepped view).</param>
    /// <param name="operand">The operand (the tested values).</param>
    /// <returns>A fresh array or view.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray MakeIsinTest(string label, NDArray operand)
    {
        long[] dims = operand.Shape.Dimensions;
        long size = 1;
        foreach (long d in dims) size *= d;
        NDArray Text(string dt) => size == 0
            ? np.empty(new Shape(dims), np.dtype(dt))
            : np.array(Enumerable.Range(0, (int)size).Select(i => (i % 3).ToString(CultureInfo.InvariantCulture)).ToArray())
                .astype(np.dtype(dt)).reshape(new Shape(dims));
        // Python's float('nan'): the positive quiet NaN (.NET's double.NaN is negative).
        double nan = BitConverter.Int64BitsToDouble(0x7FF8000000000000);
        switch (label)
        {
            case "same": return operand.copy();
            case "int32": return np.array(Enumerable.Range(0, (int)size).Select(i => i % 4).ToArray()).reshape(new Shape(dims));
            case "float64":
                return np.array(Enumerable.Range(0, (int)size).Select(i => (double)(i % 4)).ToArray()).reshape(new Shape(dims));
            case "U5": return Text("U5");
            case "T": return Text("T");
            case "S5": return Text("S5");
            case "0d": return np.array(2L);
            case "empty": return np.empty(new Shape(0), np.dtype("int64"));
            case "long": return np.array(Enumerable.Range(0, 40).Select(i => (long)(i % 7) - 3).ToArray());
            case "widelong": return np.array(Enumerable.Range(0, 40).Select(i => i * 1000L - 3).ToArray());
            case "wide": return np.array(new long[] { 0, 1_000_000, 3, -1 });
            case "nan": return np.array(new[] { nan, -0.0, 3.0, double.PositiveInfinity, -1.0 });
            case "nanlong": return np.array(Special.Concat(Plain.Take(16)).ToArray());
            case "u64": return np.array(new ulong[] { 3, (1UL << 63) + 1, 7, 0 });
            case "u64long":
                return np.array(Enumerable.Range(0, 20).Select(i => (1UL << 63) + (ulong)i)
                    .Concat(Enumerable.Range(0, 20).Select(i => (ulong)i)).ToArray());
            case "bool": return np.array(new[] { true });
            case "complex": return np.array(new[] { new Complex(1, 0), new Complex(3, 0), new Complex(2, 1) });
            case "bcast": return np.broadcast_to(np.array(new long[] { 1, 3 }), new Shape(3, 2));
            case "strided": return np.array(Enumerable.Range(0, 20).Select(i => (long)i).ToArray())["::3"];
            default: throw new ArgumentException("unknown test-set variant " + label);
        }
    }

    /// <summary>np.average's weights variants, built from the operand (see <see cref="MakeAverageWeights"/>); the first,
    ///     null (no weights: the mean), is the base.</summary>
    private static readonly string[] AverageWeightVariantLabels =
    {
        "null", "ones", "pattern", "zeros", "zeroslice", "negative", "nan", "firstaxis", "lastaxis", "axes01", "axes10",
        "axes12", "axes20", "0d", "bcast", "int", "int8", "bool", "float32", "float16", "complex", "U5", "T", "wrongshape",
    };

    /// <summary>
    ///     Builds np.average's weights for an operand: arrays of its shape, arrays along the axes NumPy's
    ///     <c>_weights_are_valid</c> reshapes (<c>wgt.shape == tuple(a.shape[ax] for ax in axis)</c>, then
    ///     <c>wgt.transpose(np.argsort(axis))</c>), weight sums that are zero, and the forms NumPy refuses.
    /// </summary>
    /// <param name="label">The variant: <c>ones</c> / <c>pattern</c> (float64 1, or 0.5, 2, 1, 3, ... so the weighted
    ///     average differs from the mean) / <c>zeros</c> (every sum zero: NumPy's ZeroDivisionError) / <c>zeroslice</c> (ones
    ///     with the first slice along axis 0 zero: zero sums for a reduction over the other axes only) / <c>negative</c> /
    ///     <c>nan</c> (the pattern with a NaN first, Python's positive quiet NaN) of the operand's shape; <c>firstaxis</c> /
    ///     <c>lastaxis</c> (the pattern along axis 0 / the last axis: one axis' weights), <c>axes01</c> / <c>axes10</c> /
    ///     <c>axes12</c> / <c>axes20</c> (the pattern of shape <c>(shape[i], shape[j])</c>: two axes' weights in the order of
    ///     the axis tuple, an unsorted one transposed by NumPy; a missing axis counts as length 1), <c>0d</c> (a float64 2),
    ///     <c>bcast</c> (the last axis' pattern broadcast to the operand's shape: stride-0 weights); ones of another dtype
    ///     (<c>int</c> int64, <c>int8</c>, <c>bool</c>, <c>float32</c>, <c>float16</c>, <c>complex</c> 1+0.5j, the text "1"
    ///     in <c>U5</c> / <c>T</c>); <c>wrongshape</c> (one more axis).</param>
    /// <param name="operand">The pool operand.</param>
    /// <returns>A fresh array (or a broadcast view), or null for the base label.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray MakeAverageWeights(string label, NDArray operand)
    {
        if (label == "null")
            return null;
        long[] dims = operand.Shape.Dimensions;
        var shape = new Shape(dims);
        long size = 1;
        foreach (long d in dims) size *= d;
        long Dim(int axis) => axis < dims.Length ? dims[axis] : 1;
        double PatternValue(int i) => new[] { 0.5, 2.0, 1.0, 3.0 }[i % 4];
        NDArray Values(long[] of, Func<int, double> value)
        {
            long count = 1;
            foreach (long d in of) count *= d;
            return np.array(Enumerable.Range(0, (int)count).Select(value).ToArray()).reshape(new Shape(of));
        }
        NDArray Pattern(params long[] of) => Values(of, PatternValue);
        NDArray Text(string dt) => size == 0
            ? np.empty(shape, np.dtype(dt))
            : np.array(Enumerable.Repeat("1", (int)size).ToArray()).astype(np.dtype(dt)).reshape(shape);
        switch (label)
        {
            case "ones": return np.ones(shape, np.dtype("float64"));
            case "pattern": return Pattern(dims);
            case "zeros": return np.zeros(shape, np.dtype("float64"));
            case "zeroslice":
            {
                var w = np.ones(shape, np.dtype("float64"));
                if (dims.Length >= 1 && dims[0] > 0)
                    w["0"] = 0.0;
                return w;
            }
            case "negative": return np.ones(shape, np.dtype("float64")) * -1.0;
            case "nan":
                return Values(dims, i => i == 0 ? BitConverter.Int64BitsToDouble(0x7FF8000000000000) : PatternValue(i));
            case "firstaxis": return Pattern(Dim(0));
            case "lastaxis": return Pattern(dims.Length == 0 ? 1 : dims[^1]);
            case "axes01": return Pattern(Dim(0), Dim(1));
            case "axes10": return Pattern(Dim(1), Dim(0));
            case "axes12": return Pattern(Dim(1), Dim(2));
            case "axes20": return Pattern(Dim(2), Dim(0));
            case "0d": return np.array(2.0);
            // a 0-d operand has no axis to broadcast along: its own 0-d pattern
            case "bcast": return dims.Length == 0 ? Pattern() : np.broadcast_to(Pattern(dims[^1]), shape);
            case "int": return np.ones(shape, np.dtype("int64"));
            case "int8": return np.ones(shape, np.dtype("int8"));
            case "bool": return np.ones(shape, np.dtype("bool"));
            case "float32": return Pattern(dims).astype(np.dtype("float32"));
            case "float16": return Pattern(dims).astype(np.dtype("float16"));
            case "complex":
                return np.array(Enumerable.Repeat(new Complex(1.0, 0.5), (int)size).ToArray()).reshape(shape);
            case "U5": return Text("U5");
            case "T": return Text("T");
            case "wrongshape":
                return np.ones(new Shape(dims.Concat(new long[] { 2 }).ToArray()), np.dtype("float64"));
            default: throw new ArgumentException("unknown average weights variant " + label);
        }
    }

    /// <summary>
    ///     Builds np.searchsorted's sorter for an operand: the argsort of its flattened values (so it sorts a 1-D haystack)
    ///     in several dtypes, and the forms NumPy refuses.
    /// </summary>
    /// <param name="label">The variant: <c>argsort</c> (np.argsort of the flattened operand, int64), <c>int32</c> (the
    ///     same cast to int32), <c>uint64</c> (cast to uint64: no 'safe' cast to intp), <c>float</c> / <c>bool</c> (not
    ///     integers), <c>2d</c> (the argsort as a (1, n) array: not 1-D), <c>short</c> (one entry fewer than the haystack),
    ///     <c>oob</c> / <c>neg</c> (the last entry replaced by n / by -1: an index outside the haystack, refused when the
    ///     search reaches it), <c>reversed</c> (n-1 .. 0 as a reversed view: valid indices that do not sort it).</param>
    /// <param name="operand">The operand (the haystack).</param>
    /// <returns>A fresh array, or null for the base label.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray MakeSorter(string label, NDArray operand)
    {
        if (label == "null")
            return null;
        NDArray order = np.argsort(operand.ravel());
        long n = order.size;
        NDArray Replaced(long value)
        {
            var copy = order.copy();
            if (n > 0)
                copy[n - 1] = NDArray.Scalar(value);
            return copy;
        }
        switch (label)
        {
            case "argsort": return order;
            case "int32": return order.astype(np.dtype("int32"));
            case "uint64": return order.astype(np.dtype("uint64"));
            case "float": return order.astype(np.dtype("float64"));
            case "bool": return np.ones(new Shape(n), np.dtype("bool"));
            case "2d": return order.reshape(new Shape(1, n));
            case "short": return n > 0 ? order[$":{n - 1}"] : order;
            case "oob": return Replaced(n);
            case "neg": return Replaced(-1);
            case "reversed": return np.arange(n)["::-1"];
            default: throw new ArgumentException("unknown sorter variant " + label);
        }
    }

    /// <summary>np.cov / np.corrcoef's second-set-of-variables variants: none (the base), then the second operands of
    ///     <see cref="MakeOperand2"/>.</summary>
    private static readonly string[] VariablesVariantLabels =
        { "null", "same", "int32", "float64", "U5", "T", "0d", "row", "badshape" };

    /// <summary>np.cov's frequency-weight variants, built from the operand (see <see cref="MakeCovWeights"/>); the first,
    ///     null, is the base.</summary>
    private static readonly string[] FWeightVariantLabels =
        { "null", "ones", "counts", "zeros", "float", "frac", "negative", "2d", "wronglen", "rows", "bool", "U5", "T", "complex" };

    /// <summary>np.cov's analytic-weight variants, built from the operand (see <see cref="MakeCovWeights"/>); the first,
    ///     null, is the base.</summary>
    private static readonly string[] AWeightVariantLabels =
        { "null", "ones", "pattern", "zeros", "negative", "2d", "wronglen", "rows", "int", "nan", "U5", "T", "complex" };

    /// <summary>
    ///     The number of observations np.cov sees in an operand: <c>X = array(m, ndmin=2)</c> holds the variables in its
    ///     rows (<paramref name="rowvar"/>) or, for a 2-D operand, in its columns; a 0-d operand is one observation, a 1-D
    ///     one is one variable. A 3-D operand is refused before any weight is read; its last axis stands in.
    /// </summary>
    /// <param name="operand">The pool operand.</param>
    /// <param name="rowvar">Whether rows are variables (the default).</param>
    /// <returns>The observation count.</returns>
    private static long CovObservations(NDArray operand, bool rowvar)
    {
        long[] dims = operand.Shape.Dimensions;
        return dims.Length switch
        {
            0 => 1,
            1 => dims[0],
            2 => rowvar ? dims[1] : dims[0],
            _ => dims[^1],
        };
    }

    /// <summary>
    ///     Builds np.cov's frequency or analytic weights for an operand: one weight per observation (the default
    ///     orientation; <c>rows</c> for <c>rowvar=False</c>, which a phase-C pair reaches), and the forms NumPy refuses.
    /// </summary>
    /// <param name="frequency">True for <c>fweights</c>, false for <c>aweights</c>.</param>
    /// <param name="label">The variant. Both: <c>ones</c>, <c>zeros</c> (a zero weight sum: np.average's
    ///     ZeroDivisionError), <c>negative</c> (one -1: NumPy's "cannot be negative"), <c>2d</c> (the weights as a (1, n)
    ///     array: "cannot handle multidimensional"), <c>wronglen</c> (one weight too many: "incompatible numbers of
    ///     samples"), <c>rows</c> (ones for the other orientation), <c>U5</c> / <c>T</c> (the text "1": NumPy's
    ///     <c>asarray(w, dtype=float)</c> parses it), <c>complex</c> (1+0.5j). Frequency weights only: <c>counts</c> (int64
    ///     2, 1, 0, 3, ...), <c>float</c> (integral float64 values, accepted), <c>frac</c> (a fraction: "fweights must be
    ///     integer"), <c>bool</c>. Analytic weights only: <c>pattern</c> (float64 0.5, 2, 1, 3, ...), <c>int</c> (int64 1,
    ///     2, 3, ...), <c>nan</c> (a NaN weight, Python's positive quiet NaN: no comparison refuses it).</param>
    /// <param name="operand">The pool operand.</param>
    /// <returns>A fresh array, or null for the base label.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray MakeCovWeights(bool frequency, string label, NDArray operand)
    {
        if (label == "null")
            return null;
        long n = CovObservations(operand, rowvar: true);
        long rows = CovObservations(operand, rowvar: false);
        NDArray Doubles(long count, Func<int, double> value) =>
            np.array(Enumerable.Range(0, (int)count).Select(value).ToArray());
        NDArray Longs(long count, Func<int, long> value) =>
            np.array(Enumerable.Range(0, (int)count).Select(value).ToArray());
        NDArray Text(string dt) => n == 0
            ? np.empty(new Shape(0), np.dtype(dt))
            : np.array(Enumerable.Repeat("1", (int)n).ToArray()).astype(np.dtype(dt));
        switch (label)
        {
            case "ones": return frequency ? Longs(n, _ => 1) : Doubles(n, _ => 1.0);
            case "zeros": return frequency ? Longs(n, _ => 0) : Doubles(n, _ => 0.0);
            case "negative": return frequency ? Longs(n, i => i == 0 ? -1 : 1) : Doubles(n, i => i == 0 ? -1.0 : 1.0);
            case "2d": return (frequency ? Longs(n, _ => 1) : Doubles(n, _ => 1.0)).reshape(new Shape(1, n));
            case "wronglen": return frequency ? Longs(n + 1, _ => 1) : Doubles(n + 1, _ => 1.0);
            case "rows": return frequency ? Longs(rows, _ => 1) : Doubles(rows, _ => 1.0);
            case "U5": return Text("U5");
            case "T": return Text("T");
            case "complex": return np.array(Enumerable.Repeat(new Complex(1.0, 0.5), (int)n).ToArray());
            // frequency weights only
            case "counts": return Longs(n, i => new long[] { 2, 1, 0, 3 }[i % 4]);
            case "float": return Doubles(n, i => i % 3 + 1.0);
            case "frac": return Doubles(n, i => i == 0 ? 1.5 : 1.0);
            case "bool": return np.ones(new Shape(n), np.dtype("bool"));
            // analytic weights only
            case "pattern": return Doubles(n, i => new[] { 0.5, 2.0, 1.0, 3.0 }[i % 4]);
            case "int": return Longs(n, i => i + 1);
            case "nan": return Doubles(n, i => i == 0 ? BitConverter.Int64BitsToDouble(0x7FF8000000000000) : 1.0);
            default: throw new ArgumentException("unknown cov weights variant " + label);
        }
    }

    /// <summary>np.gradient's single-spacing variants (see <see cref="MakeGradSpacing"/>); the first, null (no spacing), is
    ///     the base.</summary>
    private static readonly string[] GradSpacingVariantLabels =
    {
        "null", "2", "0.5", "-1.5", "0", "nan", "inf", "true", "f16", "f32", "dec", "char", "str", "0d_f64", "0d_i64",
        "0d_f32", "coords", "coords_last", "int_coords", "const", "const_int", "f32_coords", "f16_coords", "c128_coords",
        "bool_coords", "U5_coords", "doubles", "longs", "2d", "badlen", "special", "dup", "decr", "strided", "list",
        "objlist", "list_const", "ragged",
    };

    /// <summary>np.gradient's per-axis spacing variants of the two- and three-spacing overloads (see
    ///     <see cref="MakeGradSpacing"/>); the first, a unit step, is the base.</summary>
    private static readonly string[] GradAxisSpacingVariantLabels =
    {
        "1", "2", "0.5", "0d_f64", "f16", "null", "str", "coords", "int_coords", "const", "f32_coords", "2d", "badlen",
        "doubles", "list",
    };

    /// <summary>np.gradient's spacing-list variants (see <see cref="MakeGradVarargs"/>); the first, the empty list, is the
    ///     base.</summary>
    private static readonly string[] GradVarargsVariantLabels =
    {
        "empty", "one", "one_0d", "one_coords", "scalars", "coords", "mixed", "int_coords", "const", "f32", "toomany", "2d",
        "badlen", "null", "str", "doubles", "lists", "ragged",
    };

    /// <summary>
    ///     One coordinate of np.gradient's uneven sample points: 0, 0.5, 1.5, 2, 3, 3.5, ... (steps of 0.5 and 1 in turn), so
    ///     the stencil's coefficients differ from one point to the next.
    /// </summary>
    /// <param name="i">The coordinate's index.</param>
    /// <returns>The coordinate.</returns>
    private static double GradPoint(int i) => i * 0.75 - (i % 2 == 0 ? 0.0 : 0.25);

    /// <summary>The first <paramref name="count"/> uneven coordinates (<see cref="GradPoint"/>) as a float64 array.</summary>
    /// <param name="count">The number of coordinates (an axis length of the pool, so small).</param>
    /// <returns>A fresh 1-D float64 array (empty for a zero count).</returns>
    private static NDArray GradPoints(long count) =>
        np.array(Enumerable.Range(0, (int)count).Select(GradPoint).ToArray());

    /// <summary>The first <paramref name="count"/> uneven integer coordinates (0, 1, 4, 5, 8, ...: steps of 1 and 3 in turn) as
    ///     C# longs.</summary>
    /// <param name="count">The number of coordinates.</param>
    /// <returns>A fresh array.</returns>
    private static long[] GradIntPoints(long count) =>
        Enumerable.Range(0, (int)count).Select(i => (long)(2 * i - i % 2)).ToArray();

    /// <summary>
    ///     Builds one np.gradient spacing for an operand: the C# scalars NumSharp reads as Python numbers, NumPy scalars and
    ///     0-d arrays (strong under NEP 50), coordinate arrays as long as an axis of the operand, and the forms NumPy refuses.
    /// </summary>
    /// <param name="label">The variant (<see cref="GradSpacingVariantLabels"/> / <see cref="GradAxisSpacingVariantLabels"/>):
    ///     <c>null</c> (no spacing for a single spacing, a None entry for a per-axis one), <c>1</c> / <c>2</c> (a C# double /
    ///     int), <c>0.5</c> / <c>-1.5</c> / <c>0</c> (a C# double: a fraction, a negative step, a zero step), <c>nan</c>
    ///     (Python's positive quiet NaN) / <c>inf</c>, <c>true</c> (a C# bool: a Python bool), <c>f16</c> (the <c>Half</c> 0.5:
    ///     NumPy's np.float16, strong), <c>f32</c> (the C# float 0.1f: a Python float of its value), <c>dec</c> (a decimal: no
    ///     NumPy dtype), <c>char</c> (the C# char 'a': NumPy's np.uint16 97), <c>str</c> (the string "2": a Python str),
    ///     <c>0d_f64</c> / <c>0d_i64</c> / <c>0d_f32</c> (0-d arrays: strong), coordinates as long as the axis (<c>coords</c>
    ///     uneven float64, <c>coords_last</c> the same for the operand's LAST axis, <c>int_coords</c> uneven int64,
    ///     <c>const</c> / <c>const_int</c> evenly spaced float64 / int64, which NumPy reduces to their step, a NumPy scalar,
    ///     <c>f32_coords</c> / <c>f16_coords</c> / <c>c128_coords</c> / <c>bool_coords</c> / <c>U5_coords</c> in other dtypes,
    ///     <c>doubles</c> / <c>longs</c> as C# arrays (Python lists), <c>special</c> with NaN, infinities and signed zeros,
    ///     <c>dup</c> with a zero step, <c>decr</c> decreasing, <c>strided</c> a stepped view, <c>list</c> / <c>objlist</c> /
    ///     <c>list_const</c> as Python lists: a <c>List&lt;double&gt;</c> of the uneven coordinates, an <c>object[]</c> of C#
    ///     ints (Python ints: int64 coordinates), a <c>List&lt;double&gt;</c> of the even ones), and the coordinate forms NumPy
    ///     refuses (<c>2d</c> a column, <c>badlen</c> one coordinate too many, <c>ragged</c> a list holding a list, which
    ///     np.ndim refuses).</param>
    /// <param name="operand">The operand.</param>
    /// <param name="axis">The axis whose length the coordinates take (the first for the single spacing); an axis the operand
    ///     does not have takes length 3, a call NumPy refuses for its spacing count or its axis anyway.</param>
    /// <returns>The spacing: a boxed C# scalar, a string, an array, or null.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static object MakeGradSpacing(string label, NDArray operand, int axis)
    {
        long[] dims = operand.Shape.Dimensions;
        long n = axis < dims.Length ? dims[axis] : 3;
        long last = dims.Length == 0 ? 3 : dims[^1];
        // Every arm is cast to object: uncast, the switch's natural type would be NDArray through NumSharp's implicit
        // conversions, which would turn each C# scalar into a 0-d array, not the Python scalar the variant stands for.
        return label switch
        {
            "null" => null,
            "1" => (object)1.0,
            "2" => (object)2,
            "0.5" => (object)0.5,
            "-1.5" => (object)-1.5,
            "0" => (object)0.0,
            "nan" => (object)BitConverter.Int64BitsToDouble(0x7FF8000000000000),
            "inf" => (object)double.PositiveInfinity,
            "true" => (object)true,
            "f16" => (object)(Half)0.5,
            "f32" => (object)0.1f,
            "dec" => (object)0.5m,
            "char" => (object)'a',
            "str" => (object)"2",
            "0d_f64" => (object)np.array(0.5),
            "0d_i64" => (object)np.array(2L),
            "0d_f32" => (object)np.array(0.1f),
            "coords" => (object)GradPoints(n),
            "coords_last" => (object)GradPoints(last),
            "int_coords" => (object)np.array(GradIntPoints(n)),
            "const" => (object)np.array(Enumerable.Range(0, (int)n).Select(i => 0.5 * i).ToArray()),
            "const_int" => (object)np.array(Enumerable.Range(0, (int)n).Select(i => 3L * i).ToArray()),
            "f32_coords" => (object)GradPoints(n).astype(np.dtype("float32")),
            "f16_coords" => (object)GradPoints(n).astype(np.dtype("float16")),
            "c128_coords" => (object)np.array(Enumerable.Range(0, (int)n).Select(i => new Complex(GradPoint(i), (i % 3) * 0.5)).ToArray()),
            "bool_coords" => (object)np.array(Enumerable.Range(0, (int)n).Select(i => i % 3 != 1).ToArray()),
            "U5_coords" => (object)(n == 0
                ? np.empty(new Shape(0), np.dtype("U5"))
                : np.array(Enumerable.Range(0, (int)n).Select(i => (i % 3).ToString(CultureInfo.InvariantCulture)).ToArray())
                    .astype(np.dtype("U5"))),
            "doubles" => (object)Enumerable.Range(0, (int)n).Select(GradPoint).ToArray(),
            "longs" => (object)GradIntPoints(n),
            "2d" => (object)GradPoints(n).reshape(new Shape(n, 1)),
            "badlen" => (object)GradPoints(n + 1),
            "special" => (object)np.array(Special.Take((int)n).ToArray()),
            "dup" => (object)np.array(Enumerable.Range(0, (int)n).Select(i => GradPoint(i == 1 ? 0 : i)).ToArray()),
            "decr" => (object)np.array(Enumerable.Range(0, (int)n).Select(i => 4.0 - GradPoint(i)).ToArray()),
            "strided" => (object)GradPoints(2 * n)["::2"],
            // Python lists (the house map: a List<T>, an object[]): np.ndim reads them as coordinates, np.array's rules
            // convert them (a C# int item is a Python int, so int64 coordinates).
            "list" => (object)Enumerable.Range(0, (int)n).Select(GradPoint).ToList(),
            "objlist" => (object)GradIntPoints(n).Select(v => (object)(int)v).ToArray(),
            "list_const" => (object)Enumerable.Range(0, (int)n).Select(i => 0.5 * i).ToList(),
            // A ragged list: np.ndim's np.asarray raises the inhomogeneous-shape ValueError, whatever the axis' length.
            "ragged" => (object)new object[] { 0.0, new object[] { 0.5 } },
            _ => throw new ArgumentException("unknown gradient spacing variant " + label),
        };
    }

    /// <summary>
    ///     Builds np.gradient's spacing list for an operand: one entry per axis of the operand (each a scalar or that axis'
    ///     coordinates), a single entry, and the counts and entries NumPy refuses.
    /// </summary>
    /// <param name="label">The variant (<see cref="GradVarargsVariantLabels"/>): <c>empty</c> (no spacing), <c>one</c> (a
    ///     single C# double 2.0: one step for every axis), <c>one_0d</c> (a single 0-d float64 array), <c>one_coords</c> (the
    ///     first axis' coordinates alone: valid for one requested axis), <c>scalars</c> (a C# double per axis: 1, 2, 0.5, 1,
    ///     ...), <c>coords</c> / <c>int_coords</c> / <c>const</c> / <c>f32</c> / <c>doubles</c> / <c>lists</c> (each axis'
    ///     uneven float64, uneven int64, evenly spaced float64 or uneven float32 coordinates, or a C# double[] or a Python list
    ///     (<c>List&lt;double&gt;</c>) of the uneven ones), <c>mixed</c> (the first axis' coordinates, 2.0 for the other
    ///     axes), <c>toomany</c> (one scalar more than the operand has axes), and a first entry NumPy refuses beside unit
    ///     steps for the other axes (<c>2d</c> a column of coordinates, <c>badlen</c> one coordinate too many, <c>null</c> a
    ///     None, <c>str</c> the string "2", <c>ragged</c> a list holding a list; one entry even for a 0-d operand).</param>
    /// <param name="operand">The operand.</param>
    /// <returns>A fresh object array.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static object[] MakeGradVarargs(string label, NDArray operand)
    {
        long[] dims = operand.Shape.Dimensions;
        int nd = dims.Length;
        long first = nd == 0 ? 3 : dims[0];
        object[] PerAxis(Func<long, object> make) => dims.Select(make).ToArray();
        object[] FirstThenUnit(object entry) =>
            new[] { entry }.Concat(Enumerable.Repeat((object)1.0, Math.Max(nd, 1) - 1)).ToArray();
        return label switch
        {
            "empty" => Array.Empty<object>(),
            "one" => new object[] { 2.0 },
            "one_0d" => new object[] { np.array(0.5) },
            "one_coords" => new object[] { GradPoints(first) },
            "scalars" => Enumerable.Range(0, nd).Select(k => (object)new[] { 1.0, 2.0, 0.5 }[k % 3]).ToArray(),
            "coords" => PerAxis(len => GradPoints(len)),
            "int_coords" => PerAxis(len => np.array(GradIntPoints(len))),
            "const" => PerAxis(len => np.array(Enumerable.Range(0, (int)len).Select(i => 0.5 * i).ToArray())),
            "f32" => PerAxis(len => GradPoints(len).astype(np.dtype("float32"))),
            "doubles" => PerAxis(len => Enumerable.Range(0, (int)len).Select(GradPoint).ToArray()),
            "mixed" => dims.Select((len, k) => k == 0 ? (object)GradPoints(len) : 2.0).ToArray(),
            "toomany" => Enumerable.Repeat((object)1.0, nd + 1).ToArray(),
            "2d" => FirstThenUnit(GradPoints(first).reshape(new Shape(first, 1))),
            "badlen" => FirstThenUnit(GradPoints(first + 1)),
            "null" => FirstThenUnit(null),
            "str" => FirstThenUnit("2"),
            "lists" => PerAxis(len => Enumerable.Range(0, (int)len).Select(GradPoint).ToList()),
            "ragged" => FirstThenUnit(new object[] { 0.0, new object[] { 0.5 } }),
            _ => throw new ArgumentException("unknown gradient spacing-list variant " + label),
        };
    }

    /// <summary>np.linalg.multi_dot's array-list variants (see <see cref="MakeLinalgArrays"/>); the first, the operand and
    ///     its transpose, is the base.</summary>
    private static readonly string[] LinalgArraysVariantLabels = { "pair", "triple", "four", "vecs", "mixed", "mid3d", "one", "none" };

    /// <summary>
    ///     Builds np.linalg.multi_dot's array list for an operand: lists of two, three and four arrays (NumPy's three code
    ///     paths: a plain np.dot, the closed-form order of three, the dynamic-programming order), the vector forms, a mixed
    ///     dtype, and the lists NumPy refuses.
    /// </summary>
    /// <param name="label">The variant (<see cref="LinalgArraysVariantLabels"/>): <c>pair</c> (the operand and its
    ///     transpose: NumPy's two-array list is np.dot itself, which takes any rank), <c>triple</c> / <c>four</c> (the operand,
    ///     its transpose, the operand[, its transpose]: the 2-D checks apply), <c>vecs</c> (the operand's first column, the
    ///     operand, its first row: a 1-D first array is a row and a 1-D last one a column; an operand that is not a non-empty
    ///     matrix keeps the pair), <c>mixed</c> (the operand and float64 ones of its transpose's shape: the product
    ///     promotes), <c>mid3d</c> (a (2, 2, 2) array between two operands: NumPy's 2-D check names its rank), <c>one</c> /
    ///     <c>none</c> (one array, no array: "Expecting at least two arrays.").</param>
    /// <param name="operand">The operand.</param>
    /// <returns>A fresh array of arrays (its entries may be views of the operand).</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray[] MakeLinalgArrays(string label, NDArray operand)
    {
        switch (label)
        {
            case "pair": return new[] { operand, operand.T };
            case "triple": return new[] { operand, operand.T, operand };
            case "four": return new[] { operand, operand.T, operand, operand.T };
            case "vecs":
                // An empty matrix has no first row or column to take: it keeps the pair, as other ranks do.
                return operand.ndim == 2 && operand.shape[0] > 0 && operand.shape[1] > 0
                    ? new[] { operand[":, 0"], operand, operand["0"] }
                    : new[] { operand, operand.T };
            case "mixed": return new[] { operand, np.ones(new Shape(operand.T.shape), np.dtype("float64")) };
            case "mid3d": return new[] { operand, np.ones(new Shape(2, 2, 2), np.dtype("float64")), operand };
            case "one": return new[] { operand };
            case "none": return Array.Empty<NDArray>();
            default: throw new ArgumentException("unknown multi_dot arrays variant " + label);
        }
    }

    /// <summary>np.nditer's operand-list variants (see <see cref="MakeNditerOps"/>); the first, the operand alone, is the
    ///     base.</summary>
    private static readonly string[] NditerOpsVariantLabels = { "one", "pair", "alloc", "row", "int32", "zero_d", "badshape", "none" };

    /// <summary>np.nditer's op_axes variants (see <see cref="MakeNditerOpAxes"/>); the first, no op_axes, is the base.</summary>
    private static readonly string[] NditerOpAxesVariantLabels = { "null", "same", "rev", "newaxis", "bad", "dup", "empty" };

    /// <summary>np.nditer's itershape variants (see <see cref="MakeNditerItershape"/>); the first, no itershape, is the
    ///     base.</summary>
    private static readonly string[] NditerItershapeVariantLabels = { "null", "same", "deduce", "lead", "wrong" };

    /// <summary>
    ///     Builds np.nditer's multi-operand list for an operand: the operand alone, with a copy of itself, with an output the
    ///     iterator allocates (NumPy's <c>None</c>: its dtype is the operands' result type), with a float64 row of its last
    ///     axis (broadcast), with an int32 ramp of its shape (another dtype: the views keep each operand's own), with a 0-d,
    ///     with an array that does not broadcast, and no operand at all.
    /// </summary>
    /// <param name="label">The variant (<see cref="NditerOpsVariantLabels"/>).</param>
    /// <param name="operand">The pool operand.</param>
    /// <returns>A fresh operand list (its first entry is the operand itself).</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray[] MakeNditerOps(string label, NDArray operand)
    {
        long[] dims = operand.Shape.Dimensions;
        switch (label)
        {
            case "one": return new[] { operand };
            case "pair": return new[] { operand, operand.copy() };
            case "alloc": return new[] { operand, null };
            case "row":
                // The last axis' length (a 0-d operand's row is one element), so the row broadcasts against the operand.
                return new[] { operand, np.ones(new Shape(dims.Length == 0 ? 1 : dims[^1]), np.dtype("float64")) };
            case "int32":
            {
                long size = 1;
                foreach (long d in dims) size *= d;
                return new[] { operand, np.arange(size).astype(np.dtype("int32")).reshape(new Shape(dims)) };
            }
            case "zero_d": return new[] { operand, np.array(1.0) };
            case "badshape": return new[] { operand, np.ones(new Shape(7), np.dtype("float64")) };
            case "none": return Array.Empty<NDArray>();
            default: throw new ArgumentException("unknown nditer operand-list variant " + label);
        }
    }

    /// <summary>
    ///     Builds np.nditer's op_axes for an operand (one entry: the single-operand form's count; the multi-operand form
    ///     with more operands refuses it): the identity, the axes reversed, a new leading axis (-1), an out-of-range axis, a
    ///     repeated axis, and no axes.
    /// </summary>
    /// <param name="label">The variant (<see cref="NditerOpAxesVariantLabels"/>); <c>null</c> is no op_axes.</param>
    /// <param name="operand">The pool operand (its rank).</param>
    /// <returns>The op_axes, or null.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static int[][] MakeNditerOpAxes(string label, NDArray operand)
    {
        int ndim = operand.ndim;
        int[] identity = Enumerable.Range(0, ndim).ToArray();
        switch (label)
        {
            case "null": return null;
            case "same": return new[] { identity };
            // Enumerable.Reverse by name: under C# 14 an array's .Reverse() binds the in-place span overload.
            case "rev": return new[] { Enumerable.Reverse(identity).ToArray() };
            case "newaxis": return new[] { new[] { -1 }.Concat(identity).ToArray() };
            case "bad": return new[] { new[] { ndim } };
            case "dup": return new[] { new[] { 0, 0 }.Concat(identity.Skip(2)).ToArray() };
            case "empty": return new[] { Array.Empty<int>() };
            default: throw new ArgumentException("unknown nditer op_axes variant " + label);
        }
    }

    /// <summary>
    ///     Builds np.nditer's itershape for an operand: its own shape, every entry deduced (-1), an extra leading axis of 2
    ///     (the operand broadcasts into it), and a shape whose first entry does not match the operand's.
    /// </summary>
    /// <param name="label">The variant (<see cref="NditerItershapeVariantLabels"/>); <c>null</c> is no itershape.</param>
    /// <param name="operand">The pool operand (its shape).</param>
    /// <returns>The itershape, or null.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static long[] MakeNditerItershape(string label, NDArray operand)
    {
        long[] dims = operand.Shape.Dimensions;
        switch (label)
        {
            case "null": return null;
            case "same": return (long[])dims.Clone();
            case "deduce": return dims.Select(_ => -1L).ToArray();
            case "lead": return new[] { 2L }.Concat(dims).ToArray();
            case "wrong": return dims.Length == 0 ? new[] { 3L } : new[] { dims[0] + 1 }.Concat(dims.Skip(1)).ToArray();
            default: throw new ArgumentException("unknown nditer itershape variant " + label);
        }
    }

    /// <summary>The labels of a histogramdd / histogram2d role; the first is the base value.</summary>
    /// <param name="role">One of <see cref="Role.SampleSeq"/>, <see cref="Role.BinsInts"/>, <see cref="Role.BinsArrays"/>,
    ///     <see cref="Role.BinsObjects"/>, <see cref="Role.RangeArray"/>.</param>
    /// <returns>The labels <see cref="MakeDd"/> understands.</returns>
    /// <exception cref="ArgumentException">Another role (a bug in the tool).</exception>
    private static string[] DdLabels(Role role) => role switch
    {
        Role.SampleSeq => new[] { "cols", "mixed", "ragged", "single" },
        Role.BinsInts => new[] { "2s", "mixed", "wronglen", "zero", "neg" },
        Role.BinsArrays => new[] { "f64", "i64", "decr", "2d", "0d_i3", "U", "nan", "empty", "wronglen" },
        Role.BinsObjects => new[] { "mixed", "float", "text", "bool", "0d", "doubles", "wronglen", "null" },
        Role.RangeArray => new[] { "null", "all01", "first01", "bad", "nan", "wronglen", "zero" },
        _ => throw new ArgumentException("not a histogramdd role: " + role),
    };

    /// <summary>Whether a role's values are built from the operand by <see cref="MakeDd"/>.</summary>
    /// <param name="role">The role.</param>
    /// <returns>True for the histogramdd / histogram2d roles.</returns>
    private static bool IsDdRole(Role role) =>
        role is Role.SampleSeq or Role.BinsInts or Role.BinsArrays or Role.BinsObjects or Role.RangeArray;

    /// <summary>
    ///     The dimension count a histogramdd / histogram2d call sees for a pool operand: 2 for histogram2d; for histogramdd,
    ///     the columns of a 2-D sample (the last axis of a 3-D one, which NumSharp refuses anyway), one for a 1-D or 0-d
    ///     sample, and the length of the sequence <see cref="MakeSampleSeq"/> builds for the sequence form.
    /// </summary>
    /// <param name="m">The overload.</param>
    /// <param name="operand">The pool operand.</param>
    /// <returns>The dimension count the per-dimension variants are sized for (the "wronglen" variants take one more).</returns>
    private static int DdDims(MethodInfo m, NDArray operand)
    {
        if (m.Name == "histogram2d")
            return 2;
        bool sequence = m.GetParameters()[0].ParameterType == typeof(NDArray[]);
        int nd = operand.ndim;
        if (sequence)
            return nd == 0 ? 1 : nd == 2 ? (int)operand.shape[1] : 2;
        return nd == 2 ? (int)operand.shape[1] : nd == 3 ? (int)operand.shape[2] : 1;
    }

    /// <summary>
    ///     Builds a histogramdd / histogram2d argument from the operand: the sample sequence, or per-dimension bins or ranges
    ///     sized for the call's dimension count (<see cref="DdDims"/>).
    /// </summary>
    /// <param name="role">The parameter's role.</param>
    /// <param name="label">The variant (see <see cref="DdLabels"/>): bins as counts (<c>2s</c>, <c>mixed</c> 2, 3, 4, ...,
    ///     <c>wronglen</c> one count too many, <c>zero</c> / <c>neg</c> a first count NumPy refuses), as edge arrays (the
    ///     BinsArray edges of np.histogram per dimension, the first one replaced by a decreasing, 2-D, 0-d, text, NaN or empty
    ///     one, or one array too many), as a mixed object[] (counts beside edges, a float, a str, a bool, a 0-d array, a
    ///     double[], one entry too many, a null), ranges (none, (0, 1) everywhere or first only, a first range NumPy refuses
    ///     (decreasing, NaN), one range too many, a zero-width first range), and the sample sequence (the operand's columns,
    ///     the last one replaced by a float64 ramp, shortened by one, or alone).</param>
    /// <param name="operand">The pool operand.</param>
    /// <param name="m">The overload (its name and sample form decide the dimension count).</param>
    /// <returns>A fresh argument.</returns>
    /// <exception cref="ArgumentException">An unknown role or label (a bug in the tool).</exception>
    private static object MakeDd(Role role, string label, NDArray operand, MethodInfo m)
    {
        int d = DdDims(m, operand);
        NDArray F64() => np.array(new[] { -2.0, 0.0, 1.5, 5.0 });
        switch (role)
        {
            case Role.SampleSeq:
                return MakeSampleSeq(label, operand);
            case Role.BinsInts:
            {
                int[] counts = label switch
                {
                    "2s" => Enumerable.Repeat(2, d).ToArray(),
                    "mixed" => Enumerable.Range(0, d).Select(i => 2 + i).ToArray(),
                    "wronglen" => Enumerable.Repeat(2, d + 1).ToArray(),
                    "zero" => Enumerable.Range(0, d).Select(i => i == 0 ? 0 : 2).ToArray(),
                    "neg" => Enumerable.Range(0, d).Select(i => i == 0 ? -1 : 2).ToArray(),
                    _ => throw new ArgumentException("unknown int[] bins variant " + label),
                };
                return counts;
            }
            case Role.BinsArrays:
            {
                NDArray First() => label switch
                {
                    "f64" => F64(),
                    "i64" => np.array(new long[] { -4, 0, 2, 4 }),
                    "decr" => np.array(new[] { 5.0, 1.0 }),
                    "2d" => np.array(new[] { 0.0, 1.0, 2.0, 3.0 }).reshape(new Shape(2, 2)),
                    "0d_i3" => np.array(3L),
                    "U" => np.array(new[] { "1", "2", "10" }).astype(np.dtype("U5")),
                    "nan" => np.array(new[] { 0.0, BitConverter.Int64BitsToDouble(0x7FF8000000000000), 2.0 }),
                    "empty" => np.empty(new Shape(0), np.dtype("float64")),
                    "wronglen" => F64(),
                    _ => throw new ArgumentException("unknown NDArray[] bins variant " + label),
                };
                int count = label == "wronglen" ? d + 1 : d;
                var edges = new NDArray[count];
                for (int i = 0; i < count; i++)
                    edges[i] = i == 0 ? First() : label == "i64" ? np.array(new long[] { -4, 0, 2, 4 }) : F64();
                return edges;
            }
            case Role.BinsObjects:
            {
                // Every arm is cast to object: uncast, the switch's natural type is NDArray (the best common type of int,
                // double, string, bool, double[] and NDArray through NumSharp's implicit conversions), which turned each C#
                // scalar into a 0-d array and the double[] into a 1-D one — not the Python scalars these variants stand for.
                object First() => label switch
                {
                    "mixed" => (object)3,
                    "float" => (object)2.0,
                    "text" => (object)"a",
                    "bool" => (object)true,
                    "0d" => (object)np.array(3L),
                    "doubles" => (object)new[] { -2.0, 0.0, 1.5, 5.0 },
                    "wronglen" => (object)3,
                    "null" => (object)null,
                    _ => throw new ArgumentException("unknown object[] bins variant " + label),
                };
                int count = label == "wronglen" ? d + 1 : d;
                var specs = new object[count];
                // After the first entry, edge arrays alternate with counts (NumPy's mixed per-dimension specs).
                for (int i = 0; i < count; i++)
                    specs[i] = i == 0 ? First() : i % 2 == 1 ? F64() : (object)3;
                return specs;
            }
            case Role.RangeArray:
            {
                if (label == "null")
                    return null;
                (double, double)? First() => label switch
                {
                    "all01" or "first01" or "wronglen" => (0.0, 1.0),
                    "bad" => (1.0, 0.0),
                    "nan" => (BitConverter.Int64BitsToDouble(0x7FF8000000000000), 1.0),
                    "zero" => (2.0, 2.0),
                    _ => throw new ArgumentException("unknown range[] variant " + label),
                };
                int count = label == "wronglen" ? d + 1 : d;
                var ranges = new (double, double)?[count];
                for (int i = 0; i < count; i++)
                    ranges[i] = i == 0 ? First() : label is "all01" or "wronglen" ? (0.0, 1.0) : null;
                return ranges;
            }
            default:
                throw new ArgumentException("not a histogramdd role: " + role);
        }
    }

    /// <summary>
    ///     Builds np.histogramdd's sequence-of-columns sample from a pool operand.
    /// </summary>
    /// <param name="label"><c>cols</c>: the columns of a 2-D operand, a 1-D operand and its copy, a 0-d operand alone, the
    ///     two rows of a 3-D operand (2-D entries); <c>mixed</c>: the same with the last entry replaced by a float64 ramp of
    ///     its shape (columns of two dtypes); <c>ragged</c>: the last entry one element shorter (or longer when empty);
    ///     <c>single</c>: the first entry alone.</param>
    /// <param name="operand">The pool operand.</param>
    /// <returns>A fresh array of column arrays.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray[] MakeSampleSeq(string label, NDArray operand)
    {
        NDArray[] cols = operand.ndim switch
        {
            0 => new[] { operand },
            1 => new[] { operand, operand.copy() },
            2 => Enumerable.Range(0, (int)operand.shape[1]).Select(i => operand[$":, {i}"]).ToArray(),
            _ => new[] { operand[0], operand[1] },
        };
        NDArray Ramp(Shape shape)
        {
            long size = 1;
            foreach (long x in shape.Dimensions) size *= x;
            return np.array(Enumerable.Range(0, (int)size).Select(i => (double)(i % 4)).ToArray()).reshape(shape);
        }
        switch (label)
        {
            case "cols":
                return cols;
            case "mixed":
            {
                var mixed = (NDArray[])cols.Clone();
                mixed[^1] = Ramp(new Shape(cols[^1].Shape.Dimensions));
                return mixed;
            }
            case "ragged":
            {
                var ragged = (NDArray[])cols.Clone();
                NDArray last = cols[^1].ravel();
                ragged[^1] = last.size > 0 ? last[$":{last.size - 1}"] : np.zeros(new Shape(1), last.dtype);
                return ragged;
            }
            case "single":
                return new[] { cols[0] };
            default:
                throw new ArgumentException("unknown sample sequence variant " + label);
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
                       (outArray is not null && ReferenceEquals(a, outArray) ? " is_out" : "") + BytesOf(a);
            case string s:
                return "VAL String " + s.Replace("\n", "\\n");
            case double d:
                return "VAL Double " + BitConverter.DoubleToInt64Bits(d).ToString("x16");
            case float fl:
                return "VAL Single " + BitConverter.SingleToInt32Bits(fl).ToString("x8");
            case ITuple t:
                return "TUPLE(" + string.Join("; ", Enumerable.Range(0, t.Length).Select(i => Render(t[i], outArray))) + ")";
            case np.GradientResult g:
                // np.gradient's result keeps its arrays in private fields (the struct-of-arrays rule below sees none): GRAD1
                // for NumPy's bare ndarray (one requested axis), GRADN for its tuple, each component in axis order.
                return (g.IsSingle ? "GRAD1(" : "GRADN(") +
                       string.Join("; ", Enumerable.Range(0, g.Length).Select(k => Render(g[k], outArray))) + ")";
            case np.NDIterator iterator:
                // Before the IEnumerable case: an iterator is enumerable, and its per-step views rendered one by one would
                // make lines of kilobytes.
                return RenderIterator(iterator);
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
                // A struct that keeps its arrays in properties (np.unique's UniqueResult, the Array-API UniqueAllResult /
                // UniqueCountsResult / UniqueInverseResult): each public NDArray property in declaration order, an output
                // that was not requested rendering as null. Only the NDArray properties count: UniqueResult also forwards
                // values' size / shape / dtype, which repeat what the values array's own rendering holds, and its indexer
                // (a property with parameters) is a view of the same outputs.
                var arrayProperties = arrayFields.Length == 0 && result.GetType().IsValueType
                    ? result.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance)
                        .Where(pi => pi.GetIndexParameters().Length == 0 && typeof(NDArray).IsAssignableFrom(pi.PropertyType))
                        .OrderBy(pi => pi.MetadataToken).ToArray()
                    : Array.Empty<PropertyInfo>();
                if (arrayProperties.Length > 0)
                    return "STRUCT " + result.GetType().Name + "(" +
                           string.Join("; ", arrayProperties.Select(pi => pi.Name + "=" + Render(pi.GetValue(result), outArray))) + ")";
                return "VAL " + result.GetType().Name + " " + Convert.ToString(result, CultureInfo.InvariantCulture);
        }
    }

    /// <summary>
    ///     Renders an np.nditer iterator by pulling it to its end, as a Python <c>for x in it</c> does: <c>ITER nop=N
    ///     dtypes=[...] steps=S</c>, then per operand the dtype of its views, the run-length list of their shapes
    ///     (<c>()x24</c>, <c>(4,)x6</c>), the distinct flag sets in the order they appear and ONE hash over the bytes of every
    ///     step's view (repr for StringDType, as <see cref="HashOf"/>), and <c>EXC@k</c> with the exception when the
    ///     construction's properties, a pull or a view raises at step k. The iterator is disposed (its write-backs
    ///     resolved) before the line is returned.
    /// </summary>
    /// <param name="it">The iterator np.nditer returned.</param>
    /// <returns>The outcome text.</returns>
    /// <remarks>
    ///     The views alias the iterator's memory (an operand, a buffer, an allocated output), so each is read at its own
    ///     step; an output the iterator allocates holds what the allocator left there, which the snapshot reads as zeros
    ///     only under <c>NUMSHARP_DEBUG_GUARD_PAGES=1</c> (every buffer fresh): take nditer snapshots with it. More than
    ///     100,000 steps ends the iteration (no pool operand has that many elements: a step that never ends is a bug).
    /// </remarks>
    private static string RenderIterator(np.NDIterator it)
    {
        var sb = new StringBuilder("ITER");
        int steps = 0;
        IncrementalHash[] hashes = null;
        string[] dtypes = null;
        List<(string Shape, int Count)>[] shapes = null;
        List<string>[] flags = null;
        string error = null;
        try
        {
            sb.Append($" nop={it.nop} dtypes=[{string.Join(", ", it.dtypes.Select(d => d?.ToString() ?? "null"))}]");
            foreach (NDArray[] views in it)
            {
                if (hashes is null)
                {
                    hashes = views.Select(_ => IncrementalHash.CreateHash(HashAlgorithmName.SHA256)).ToArray();
                    dtypes = new string[views.Length];
                    shapes = views.Select(_ => new List<(string, int)>()).ToArray();
                    flags = views.Select(_ => new List<string>()).ToArray();
                }
                for (int k = 0; k < views.Length; k++)
                {
                    NDArray v = views[k];
                    dtypes[k] ??= v.dtype.ToString();
                    string shape = "(" + string.Join(",", v.shape) + ")";
                    var runs = shapes[k];
                    if (runs.Count > 0 && runs[^1].Shape == shape)
                        runs[^1] = (shape, runs[^1].Count + 1);
                    else
                        runs.Add((shape, 1));
                    var f = v.flags;
                    string fl = $"{(f.c_contiguous ? "C" : "-")}{(f.f_contiguous ? "F" : "-")}{(f.owndata ? "O" : "-")}{(f.writeable ? "W" : "-")}";
                    if (!flags[k].Contains(fl))
                        flags[k].Add(fl);
                    hashes[k].AppendData(ViewBytes(v));
                }
                if (++steps > 100_000)
                {
                    error = "ABORT more than 100000 steps";
                    break;
                }
            }
        }
        catch (Exception e)
        {
            error = $"EXC@{steps} {Describe(e)}";
        }
        finally
        {
            it.Dispose();
        }
        sb.Append($" steps={steps}");
        if (hashes is not null)
            for (int k = 0; k < hashes.Length; k++)
                sb.Append($" op{k}={dtypes[k]} {string.Join(",", shapes[k].Select(r => r.Shape + "x" + r.Count))} " +
                          $"{string.Join("/", flags[k])} " +
                          Convert.ToHexString(hashes[k].GetHashAndReset()).Substring(0, 16).ToLowerInvariant());
        if (error is not null)
            sb.Append(' ').Append(error);
        return sb.ToString();
    }

    /// <summary>The bytes <see cref="HashOf"/> hashes for an array (its C-order bytes, or its repr for StringDType and
    ///     for anything <c>tobytes</c> refuses).</summary>
    /// <param name="a">The array.</param>
    /// <returns>The bytes (the type name of the failure when neither form is available).</returns>
    private static byte[] ViewBytes(NDArray a)
    {
        try
        {
            return a.dtype.kind == 'T'
                ? Encoding.UTF8.GetBytes(a.ToString(true))
                : np.ascontiguousarray(a).tobytes();
        }
        catch (Exception e)
        {
            try { return Encoding.UTF8.GetBytes(a.ToString(true)); }
            catch { return Encoding.UTF8.GetBytes("unhashable(" + e.GetType().Name + ")"); }
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

    /// <summary>Whether ARR lines carry the result's bytes (environment variable <c>SNAPSHOT_BYTES=1</c>).</summary>
    private static readonly bool WithBytes = Environment.GetEnvironmentVariable("SNAPSHOT_BYTES") == "1";

    /// <summary>
    ///     With <c>SNAPSHOT_BYTES=1</c>, the result's bytes in C order as <c> BYTES &lt;hex&gt;</c>, so a value mismatch the
    ///     hash shows can be measured element by element (ULP distances, NaN payloads) against NumPy's bytes; otherwise
    ///     nothing. A diagnostic mode: a line with bytes never equals a line without, so compare snapshots made the same way.
    /// </summary>
    /// <param name="a">The result.</param>
    /// <returns>The suffix, or the empty string (the mode off, a StringDType result, or bytes NumPy's tobytes refuses).</returns>
    private static string BytesOf(NDArray a)
    {
        if (!WithBytes || a.dtype.kind == 'T')
            return "";
        try
        {
            return " BYTES " + Convert.ToHexString(np.ascontiguousarray(a).tobytes()).ToLowerInvariant();
        }
        catch
        {
            return "";
        }
    }
}
