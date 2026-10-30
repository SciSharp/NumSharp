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
// power's exponent p are the second operand); "np.fft.f" every public instance overload declared on the module class
// FourierModule, called on np.fft; "NDArray.f" every public instance overload declared on NDArray (its indexer's
// accessors are "NDArray.get_Item" and "NDArray.set_Item");
// "NDArray.op_X" every public static operator overload declared on NDArray (op_Equality, op_UnaryPlus, ...), whose first
// NDArray parameter is the operand; "NDMatrix.f" every public instance overload declared on NDMatrix, called on
// np.asmatrix of each operand (a layout asmatrix refuses, the 3-D ones, is SKIP) with an out= variant made a matrix too;
// "np.r_" and "np.c_" the indexer of the object (np.AxisConcatenator's this[params object[] key]), called on it;
// "np.mgrid" and "np.ogrid" the indexer of the nd_grid object (np.MGridClass / np.OGridClass), whose key is its operand
// (GridKeyLabels: the operand label "key/<slices>"). np.ravel_multi_index's array overloads build their coordinate list
// from the pool operand (MakeRavelCoords); its scalar overload's coordinate TUPLE is its operand
// (RavelScalarCoordsLabels: the operand label "coords/[i,j]").
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
// the max / min family's initial= (np.max / np.amax / np.min / np.amin / np.nanmax / np.nanmin and NDArray's max / min: C#
// numbers as Python's int and float, NumPy scalars, a 0-d array, NaN and infinities, values past a narrow loop's range,
// text and a sequence), so a where= mask meets its masked reduction,
// the sort family's kind (np.sort / np.argsort's sort kinds, np.partition / np.argpartition's select kind), stable,
// order, kth (an int, an int[] or an index array) and np.lexsort's key list built from the operand, and
// np.average / np.average_returned's weights (of the operand's shape, along one axis or two in either order, a 0-d, zero
// sums, other dtypes, the forms NumPy refuses) and axis tuple (single axes, both orders of a pair, the empty tuple, a
// repeated and an out-of-range axis), and np.convolve / np.correlate's second sequence v (the operand's copy, other dtypes
// and lengths, one-element arrays of either stride sign, special values, text, the shapes NumPy refuses, strided and
// reversed views) and their mode name (the three names, near misses, other strings, null), and np.nditer's operand list
// op (the operand alone, with a copy, with an output to allocate, with a row, another dtype, a 0-d, a shape that does not
// broadcast, none), flags (each tracking and loop flag, the buffered forms, refs_ok, common_dtype, reduce_ok, ranged,
// two refused forms), op_flags (one list, or one per operand), op_dtypes (numbers, the unsized 'U' / 'S' NumPy adapts, a
// sized 'U5', StringDType, two refused forms), op_axes and itershape (built from the operand's rank and shape) and
// buffersize, its order and casting taking the general roles), and np.einsum's subscripts (built from the operand's rank:
// views, sums, a diagonal, ellipses, two- and three-operand contractions, an outer product, refused forms), its operand
// list (built from the operand, see MakeEinsumOps) and optimize, its out / dtype / order / casting taking the general
// roles, and np.interp's sample points xp and values fp (fixed arrays, independent of the operand x: a ramp in several
// dtypes and layouts, repeated, decreasing and NaN points, complex and text values, the shapes NumPy refuses), its fills
// left / right (a double, or a Complex for the complex overload) and its period, and np.fft's transform length n (None,
// crops, zero pads, the odd and even lengths the real transforms tell apart, the lengths NumPy refuses), norm (the three
// names, None and a near miss), the N-D lengths s (one or two axes, -1, an empty and a refused length, more lengths than
// axes) and fftfreq / rfftfreq's window length n (an int, a long, or a double NumPy refuses), spacing d and device, the
// axis / axes taking the general roles, and np.nan_to_num's fill values and np.real_if_close's tol, and
// np.unravel_index's dimensions (empty, zero, negative and overflowing ones), order (C / F in either case, A / K, a
// refused letter) and the scalar overload's flat index, np.packbits / np.unpackbits' bitorder and count and np.bincount's
// minlength, and the item-selection family's arguments (see ItemArgLabels: index arrays and scalar indices, the clip
// mode, values, masks, repeat counts, choices, np.compress's condition, whose operand is its second parameter a, and
// np.resize / ndarray.resize's new shapes; np.choose runs on the non-negative 1-D pools too), and the assignment family's
// (see AssignArgLabels: np.copyto's sources - arrays of the operand's shape in other dtypes, 0-d and broadcast ones, views
// of the destination itself, None, Python ints past every C# integer, strs, and the C# numbers of the generic overload,
// which is instantiated for every unmanaged type NumSharp stores - and ndarray.fill's value), and the coercion family's
// HOST operand (see HostForms: np.array / np.asarray / np.asanyarray, NDArray.Scalar and NDArray's conversion operators,
// whose first parameter is a C# value rather than an NDArray - a typed, jagged or object array, a nested object list or
// grid, a List / an iterator / a Memory / a tuple, a scalar, a str, a str array, a MemoryView, null - built from each pool
// operand in every form its parameter type admits, a combination without such a value producing no line; their generic
// overloads are instantiated for every element type NumSharp stores; ndmin / ndim, copy, device, a dtype name and a read
// size take their own variants), and the spacing generators' (np.linspace / np.logspace / np.geomspace, which take no
// array: their operand is a (start, stop) PAIR from SpacePairLabels - increasing, decreasing, equal, mixed-sign, zero,
// NaN, infinite, signed-zero and extreme pairs, complex ones for geomspace's Complex overload - with the sample count,
// the log base, the moveaxis destination and the device varied), np.trim_zeros' trim spec and np.vander's column count
// N, and the constructor family's (np.empty / np.zeros / np.ones / np.full / np.eye / np.identity / np.ndarray, which take
// no array: their operand is a SHAPE from CtorShapeLabels - 0-d, empty, 1-D to 3-D, unit, negative and overflowing
// dims - in the C# form its parameter takes (an int, an int[] / long[], a Shape; a form the shape has no value in
// produces no line), with np.full's fill value (CtorFillLabels: C# numbers of every kind, text, bytes, None, a Python int
// past every C# integer, NaN and -0.0, 0-d and broadcast arrays, a C# array), np.eye's M and k, the creation order (C
// first: NumPy's creation functions refuse A and K) and np.ndarray's buffer, offset and strides varied; the _like forms'
// fill value and shape= take the same variants), np.indices / np.indices_sparse's (their operand is the grid's SHAPE as a
// constructor's, from IndicesShapeLabels - the constructor shapes and the lengths a dtype's arange decides on - with every
// dtype over every shape; a sparse grid's overflowing shape produces no line), np.diag / np.diagflat's diagonal k (np.eye's
// variants), np.frombuffer's (which takes no array either: its operand is a
// BUFFER holding the pool operand's C-order bytes in the C# form its first parameter takes - a byte[], NumSharp's MemoryView
// of the operand itself, an ArraySegment / a Memory of bytes inside a larger array, a Memory over unmanaged memory, an
// address with its byte length, the generic overload's typed C# array of the operand's values (instantiated for every
// element type NumSharp stores) - see BufferForms, with the count, the byte offset, the address overloads' byte length and
// the dtype as a descriptor, an NPTypeCode or a name varied; its ReadOnlySpan overloads cannot be called by reflection),
// np.fromstring's (its TEXT is the operand: the pool's own value sequence - integers, Python float reprs with inf / nan /
// -0.0, or the text pool's strings - joined by ", " or " ", or with unmatched data after it, as a str or as UTF-8 bytes,
// one with an embedded NUL; see TextSourceForms), np.fromfile's (its FILE: a stream or a temporary file holding the
// operand's C-order bytes or that text, a stream that cannot seek, a missing file; the stream's final position joins the
// outcome) and NDArray.tofile's (its sink, a MemoryStream or a temporary file, whose bytes join the outcome), with the
// separator, the count, the offset and tofile's format varied,
// np.loadtxt's (its SOURCE: the pool's value sequence written as rows of text in the C# form its first parameter takes - a
// temporary file, also gzip-compressed, a missing one, a byte stream, a TextReader, a list of lines - plain, comma-
// separated, with comments, with blank lines, with a ragged last row, quoted; see LoadTxtSourceForms) with its dtype,
// comments, delimiter, converters (from a menu NumPy names alike), skiprows, usecols, unpack, ndmin, encoding, max_rows and
// quotechar varied, and np.savetxt's (X is the pool operand; its sink - a temporary file, a MemoryStream, a StringWriter -
// joins the outcome) with fmt (a string, or a list built from X's columns), delimiter, newline, header, footer, comments
// and encoding varied,
// np.where's x / y given as C# values (WhereScalar) and np.extract's arr
// and ndarray.real / imag's setters' value (the second operand's role), and - through the item-selection wiring, built
// from the pool operand - np.take_along_axis / np.put_along_axis's indices and values (AlongIndicesLabels,
// AlongValuesLabels), np.fill_diagonal's val (FillDiagValueLabels, Python lists included), np.select's condition list,
// choice list and default
// (np.select has no array parameter: all three come from the pool operand) and np.piecewise's condition list in its
// NDArray[] and bare forms, its function list from a fixed menu NumPy names alike (negate, square, a callable of the
// extra args, constants) and its extra args, np.delete's boolean-mask obj (a C# bool[]), and np.pad's pad_width in each
// C# form (an int, an int[], an int[,] built from the operand's rank, a (before, after) tuple, a dict), its mode (every
// name NumPy has, one it refuses, two callables from a menu NumPy names alike, null), its constant_values / end_values /
// stat_length / reflect_type and a callable's kwargs (PadArgLabels), np.ix_'s sequences (IxArgsLabels: the operand, its
// flattened values, masks, C# arrays and lists, empty ones of both kinds, refused ranks, None) and np.r_ / np.c_'s key
// (ConcatKeyLabels: the operand beside C# scalars of every kind, arrays, slices and every directive form), and
// np.apply_along_axis's func1d from a fixed menu NumPy names alike and its extra args (ApplyFuncLabels,
// ApplyArgsLabels), and NDArray's indexers' (get_Item / set_Item, __setitem__ and the fancy pair GetIndices / SetIndices:
// the key, built from the operand's rank - integers of every C# kind, slice notation, Slice objects, the ellipsis and new
// axes, C# bools and tuples, index arrays of several dtypes, ranks and layouts, masks, mixed advanced indices and the
// entries NumPy or C# refuses; IndexKeyLabels -, the fancy pair's index list, and the stored value - the selection's own
// values reversed, C# literals, text, 0-d and one-element arrays, None, a Python int past every C# integer, a list, a shape
// that does not broadcast; IndexValueLabels / IndexObjectValueLabels), and NDArray.astype's (the cast family: its dtype
// over every class a cast reaches - the numeric dtypes, S / U at three widths and unsized, StringDType with and without an
// NA object - and its casting rule with astype's own 'same_value'; CastDTypeLabels / CastCastingLabels, on a reduced
// pool that keeps the special layouts and the strided, reversed and broadcast views, and pairs over every pool dtype);
// an overload
// with a required parameter of another
// kind is reported once as SKIP. An int
// axis
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
// nditer snapshots under NUMSHARP_DEBUG_GUARD_PAGES=1, where an allocated output's memory reads as zeros. Take the coercion
// family's under it too: np.array<T>(IEnumerable<T>, size) allocates size elements and leaves those past the sequence's
// end as allocated, which differs from run to run in the pooled allocator. A rectangular C# array result (the explicit
// conversion to Array) renders as MDARRAY elementType[dims] LIST[...] (its dimensions, then its elements in C order).
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using NumSharp;
using NDExpr = NumSharp.Backends.Iteration.NDExpr;

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
    ///     module class <see cref="EmathModule"/> (called on <c>np.emath</c>), <c>np.fft.name</c> for those declared on the
    ///     module class <see cref="FourierModule"/> (called on <c>np.fft</c>), <c>NDArray.name</c> for the public instance
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
        // np.fft.f is an instance member of the module object np.fft, as np.emath's are.
        bool onFft = target.StartsWith("np.fft.", StringComparison.Ordinal);
        string name = target.Substring((onLinalg || onEmath || onFft ? target.LastIndexOf('.') : target.IndexOf('.')) + 1);
        bool onNDArray = target.StartsWith("NDArray.", StringComparison.Ordinal);
        bool onMatrix = target.StartsWith("NDMatrix.", StringComparison.Ordinal);
        // np.r_ / np.c_ name the indexer of the object (np.AxisConcatenator's get_Item), called on that object.
        bool onConcatenator = target is "np.r_" or "np.c_";
        if (onConcatenator)
            IndexerReceiver = target == "np.r_" ? np.r_ : np.c_;
        // np.mgrid / np.ogrid name the indexer of the nd_grid object (np.MGridClass / np.OGridClass's get_Item), called on
        // that object; the key is their only argument, so it is the operand (GridKeyLabels).
        bool onGrid = target is "np.mgrid" or "np.ogrid";
        if (onGrid)
            IndexerReceiver = target == "np.mgrid" ? np.mgrid : np.ogrid;
        // An operator is a static method of NDArray whose first NDArray parameter is the operand, as np.f's is; so is a
        // static factory of NDArray (NDArray.Scalar, NDArrayStaticTargets), whose first parameter is the coercion family's
        // host operand.
        bool instance = (onNDArray && !name.StartsWith("op_", StringComparison.Ordinal) && !NDArrayStaticTargets.Contains(name))
                        || onMatrix;
        var methods = (onConcatenator
                ? typeof(np.AxisConcatenator).GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => m.Name == "get_Item")
                : onGrid
                ? (target == "np.mgrid" ? typeof(np.MGridClass) : typeof(np.OGridClass))
                    .GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => m.Name == "get_Item")
                : onMatrix
                ? typeof(NDMatrix).GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => m.DeclaringType == typeof(NDMatrix))
                : instance
                    ? typeof(NDArray).GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => m.DeclaringType == typeof(NDArray))
                    : onNDArray
                        ? typeof(NDArray).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.DeclaringType == typeof(NDArray))
                        : onLinalg
                            ? typeof(np.linalg).GetMethods(BindingFlags.Public | BindingFlags.Static)
                            : onEmath
                                ? typeof(EmathModule).GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => m.DeclaringType == typeof(EmathModule))
                                : onFft
                                    ? typeof(FourierModule).GetMethods(BindingFlags.Public | BindingFlags.Instance).Where(m => m.DeclaringType == typeof(FourierModule))
                                    : typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static))
            .Where(m => onConcatenator || onGrid || m.Name == name)
            // a generic overload runs closed over the types the snapshot gives it (np.copyto<T>), else not at all
            .SelectMany(m => m.IsGenericMethodDefinition ? SnapshotInstantiations(m) : new[] { m })
            .Select(m => (Method: m, Signature: Signature(target, m)))
            .OrderBy(t => t.Signature, StringComparer.Ordinal)
            .ToList();
        if (methods.Count == 0)
            throw new ArgumentException($"no public overload named {target}");
        foreach (var (method, signature) in methods)
            foreach (var line in RunOverload(method, signature, instance, onMatrix))
                yield return line;
    }

    /// <summary>
    ///     The static members of NDArray a target <c>NDArray.name</c> names (instead of its instance members, which every
    ///     other NDArray target names): the factories of the coercion family, whose first parameter is the host operand.
    ///     The operators are static too, but they are named <c>op_X</c>, which already selects them.
    /// </summary>
    private static readonly HashSet<string> NDArrayStaticTargets = new(StringComparer.Ordinal) { "Scalar" };

    /// <summary>The object a target <c>np.r_</c> / <c>np.c_</c> calls its indexer on (set by <see cref="Run"/>).</summary>
    private static object IndexerReceiver;

    /// <summary>A stable text for an overload: the target, a closed generic overload's type arguments, and its parameter
    ///     types and names.</summary>
    /// <param name="target">The target as given on the command line.</param>
    /// <param name="m">The overload.</param>
    /// <returns>For example <c>np.cumsum(NDArray arr, Int32? axis, DType typeCode, NDArray out)</c>, or
    ///     <c>NDArray.Scalar&lt;Double&gt;(Object value)</c> for a closed generic one (whose parameters alone can read like
    ///     another overload's: <c>Scalar&lt;T&gt;(object)</c> closed over every element type beside
    ///     <c>Scalar(object)</c>).</returns>
    private static string Signature(string target, MethodInfo m) =>
        target + (m.IsGenericMethod ? "<" + string.Join(",", m.GetGenericArguments().Select(a => a.Name)) + ">" : "") +
        "(" + string.Join(", ", m.GetParameters().Select(p => TypeName(p.ParameterType) + " " + p.Name)) + ")";

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
        // np.delete's boolean mask as a C# bool[] (its own overload; NumPy reads it as an index array of bools).
        InsertObjMask,
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
        // np.einsum: the subscripts (built from the operand's rank: the identity view, sums, a transpose, the implicit
        // output, a diagonal, ellipses, two- and three-operand contractions, an outer product, and the forms NumPy
        // refuses; see MakeEinsumSubscripts), the operand list (built from the operand: it alone, with a copy, with an
        // int32 ramp, with text, with a 0-d, three of it, none; see MakeEinsumOps) and optimize. out, dtype, order and
        // casting take the general roles.
        EinsumSubscripts, EinsumOps, EinsumOptimize,
        // np.polyint's integration constants k: an NDArray (null, a 0-d, one value, five, two, integers, text) or the C#
        // double of its convenience overload. Its order m takes the IntArg role.
        PolyintK, PolyintKDouble,
        // The sort family (np.sort / np.argsort / np.partition / np.argpartition / np.lexsort and the ndarray methods sort /
        // argsort / partition / argpartition): the sort kind of sort / argsort and the select kind of partition /
        // argpartition (the names NumPy parses and the ones it refuses), the keyword-only stable, the order (a field name,
        // which every NumSharp dtype refuses), kth as a C# int, an int[] and an index array (positions inside the axis,
        // negative ones, out of range, unsorted, repeated, empty, and the index arrays NumPy refuses), and np.lexsort's key
        // list (built from the operand: it alone, two keys, keys of other dtypes, a mismatched shape, none; see
        // MakeLexKeys).
        SortKind, PartitionKind, Stable, SortOrder, Kth, KthSeq, KthArray, LexKeys,
        // The max / min family's initial= (np.max / np.amax / np.min / np.amin / np.nanmax / np.nanmin and NDArray's max /
        // min): the value every output starts from, which a where= mask needs (see the Initial variants).
        Initial,
        // np.interp: the sample points xp and values fp (fixed arrays independent of the operand x: an increasing ramp
        // in several dtypes and layouts, repeated, decreasing and NaN points, complex and text values, and the forms NumPy
        // refuses), the fills left / right (a C# double, or a Complex for the complex overload) and the period.
        InterpXp, InterpFp, InterpFill, InterpFillComplex, InterpPeriod,
        // np.fft: the 1-D transforms' length n (nullable: None = the input's length along the axis, 2 * (m - 1) for irfft /
        // hfft), the norm name, the N-D transforms' lengths s, and fftfreq / rfftfreq's window length n (an int or a long,
        // or a double that NumPy refuses), sample spacing d and device.
        FftN, FftNorm, FftS, FftFreqN, FftD, FftDevice,
        // np.nan_to_num's fill values nan / posinf / neginf (objects: NumPy's array_like scalars or arrays, so the fused
        // scalar path and the copyto composition are both reached) and np.real_if_close's tol (a multiple of the
        // dtype's eps above 1, an absolute bound otherwise).
        NanToNumFill, RealIfCloseTol,
        // The C functions of numpy._core.multiarray whose arguments the general roles do not read: np.unravel_index's
        // dimensions (an int[]: empty, zero, negative and overflowing ones included), its order (a char: C / F in either
        // case, the A / K NumPy refuses after its other checks, a letter it refuses first) and the scalar overload's flat
        // index; np.packbits / np.unpackbits' bitorder (NumPy's two prefix rules) and unpackbits' count (none, truncating,
        // padding, negative trims and an over-long trim); np.bincount's minlength.
        UnravelShape, UnravelOrder, UnravelFlat, PackBitOrder, UnpackCount, BincountMinLength,
        // The item-selection family (np.take / np.put / np.putmask / np.place / np.repeat / np.choose / np.compress /
        // np.resize and NDArray's take / put / repeat / choose / compress / resize): every parameter its general roles do
        // not read - index arrays and scalar indices, the clip mode, values, masks, counts, choices, conditions and new
        // shapes - takes this one role, its variants chosen by the parameter's name and type (ItemArgLabels) and built
        // from the operand where they must fit it (MakeItemArg).
        ItemArg,
        // The assignment family (np.copyto and ndarray.fill): np.copyto's source in each of its C# forms (an array, a Python
        // int past every C# integer, a str, a C# number of the generic overload) and ndarray.fill's value take this role,
        // their variants chosen by the parameter's name and type (AssignArgLabels) and built from the destination where
        // they must fit it (MakeAssignArg); dst, casting and where take the general roles.
        AssignArg,
        // NDArray.Contains (C#'s `value in arr`, NumPy's ndarray.__contains__: (self == value).any()): the value searched
        // for, an object - a C# int / double / bool, a string (a char array searches its characters), null, NaN, a C#
        // int[] and an NDArray, so the comparison's broadcast and its refusals are reached.
        ContainsValue,
        // np.evaluate: the expression, built over the operand (an embedded leaf, NDExpr.Arr; the positional overload's
        // NDExpr.Input(0) over an operand list holding the operand alone): the identity and a computed child, and every
        // root reduction the engine runs itself or delegates (sum, ptp, nanmin / nanmax, argmax / argmin, any / all), flat,
        // along axis 0 and with keepdims (see EvalExprVariantLabels and MakeEvalExpr).
        EvalExpr, EvalExprPositional, EvalOperands,
        // The coercion family (np.array / np.asarray / np.asanyarray, NDArray.Scalar and NDArray's implicit / explicit
        // conversion operators): the HOST operand - the first parameter when it is not an NDArray, a C# value built from
        // the pool operand in one of the forms its type admits (HostForms, MakeHostOperand) - and the parameters only
        // this family declares: copy as NumPy's tri-state (bool?), the device name, the dtype as a name (np.asarray's
        // np.MemoryView overload), and the item count np.array reads from an IEnumerable. Its ndmin / ndim take IntArg, its
        // dtype, order and bool copy the general roles.
        HostOperand, CoerceCopy, CoerceDevice, CoerceDTypeName, CoerceSize,
        // The spacing generators (np.linspace / np.logspace / np.geomspace) take no array: their operand is a (start, stop)
        // PAIR (SpacePairLabels; the stop parameter is the pair's second value, built in Call), with the sample count, the
        // log base and the moveaxis destination of their own; endpoint and dtype take the general roles, the device the
        // coercion family's. np.trim_zeros' trim spec (required on its axis-list overload) and np.vander's column count N.
        SpacePair, SpaceStop, SpaceNum, SpaceBase, SpaceAxis, TrimSpec, VanderN,
        // The constructor family (np.empty / np.zeros / np.ones / np.full / np.eye / np.identity / np.ndarray) takes no
        // array: its operand is a SHAPE (CtorShapeLabels; np.eye's N and np.identity's n are its first dim), built in Call
        // in the C# form of the parameter (MakeCtorShape). The fill value of np.full / np.full_like, the shape= of the _like
        // forms (base: the prototype's own), np.eye's M and k (whose variants np.diag / np.diagflat's k take too), the
        // creation order (C first, unlike the general Order role: NumPy's creation functions refuse A and K), and
        // np.ndarray's buffer, byte offset and byte strides.
        CtorShape, CtorFill, CtorShapeArg, EyeM, EyeK, CtorOrder, NdBuffer, NdOffset, NdStrides,
        // np.frombuffer takes no array: its operand is a BUFFER, its first parameter (a byte[], a MemoryView, an
        // ArraySegment / a Memory of bytes, an address, a typed C# array), built in Call from the pool operand's C-order
        // bytes in the parameter's C# form (BufferForms, TryMakeBufferOperand). The address overloads' byte length is built
        // from that buffer (BufferLengthLabels: never past its end, the memory being the tool's); the count, the byte offset
        // and the dtype as a descriptor, an NPTypeCode or a name take their own variants.
        BufferOperand, BufferLength, BufferCount, BufferOffset, BufferDType, BufferTypeCode, BufferDTypeName,
        // np.fromstring / np.fromfile take no array either: the first is read from a TEXT (a str or its bytes), the second
        // from a FILE (a stream or a path), each built in Call from the pool operand (TextSourceForms, FileSourceForms).
        // NDArray.tofile writes into a SINK (a MemoryStream or a temporary file) whose bytes join the outcome. Their
        // separator, np.fromfile's offset, tofile's format take their own variants; the count takes np.frombuffer's.
        TextSource, FileSource, IoSep, ToFileFormat, SinkStream, SinkFile,
        // np.loadtxt takes no array: its operand is a SOURCE of text rows (a file name, a byte stream, a TextReader, a list
        // of lines), built in Call from the pool operand (LoadTxtSourceForms, TryMakeLoadTxtSource). np.savetxt writes X (the
        // pool operand) into a SINK - a temporary file or a MemoryStream (tofile's roles) or a StringWriter (SinkWriter) -
        // whose bytes join the outcome. Their other parameters take TextIOArg's variants (TextIOArgVariants, by function and
        // name), except savetxt's fmt list, built from X's column count (SaveTxtFmtsLabels, MakeSaveTxtFmts).
        LoadTxtSource, TextIOArg, SaveTxtFmts, SinkWriter,
        // np.where's x / y given as a C# value (its object overloads): NumPy's array_like - a Python int / float / complex /
        // bool / str, a NumPy scalar, a 0-d array, a C# array - which np.where converts as NDArray.FromPythonOperand does.
        WhereScalar,
        // np.mgrid / np.ogrid take no array: the KEY of their indexer is all they take, so it is the operand
        // (GridKeyLabels: slices as strings, Slice objects, the empty key, entries that are no slice), built in Call
        // (MakeGridKey).
        GridKey,
        // np.ravel_multi_index: the coordinate list of its array overloads, built from the pool operand (the operand
        // twice, with a second coordinate array of another dtype, layout or shape, a 0-d one, a bool and a float one,
        // too few and too many, a null entry, none; RavelCoordsVariantLabels, MakeRavelCoords); its dims (an int[]:
        // covering every unsigned pool value, small ones, a zero, empty, negative and overflowing ones), its mode (a
        // string: NumPy's three, the texts its parser refuses, None) and modes (a string[] of one mode per dimension:
        // mixed, the wrong length, None, a None entry, a bad entry); and its scalar overload's coordinate TUPLE, which
        // is that overload's operand (RavelScalarCoordsLabels, built in Call by MakeRavelScalarCoords). Its order takes
        // UnravelOrder.
        RavelCoords, RavelDims, RavelMode, RavelModes, RavelScalarCoords,
        // NDArray.astype (the cast family): its dtype over every class a cast reaches - the numeric storage dtypes, the
        // fixed-width text classes at three widths and unsized (np.dtype('U') / np.dtype('S'), which astype adapts to the
        // array), StringDType with and without an NA object - and its casting rule with astype's own 'same_value'
        // (CastDTypeLabels, CastCastingLabels). copy and order take the general Bool / Order roles.
        CastDType, CastCasting,
        Default, Unsupported,
    }

    /// <summary>
    ///     Whether a parameter belongs to NDArray.astype, the cast family: its dtype takes <see cref="CastDTypeLabels"/> and its
    ///     casting rule <see cref="CastCastingLabels"/> (the other targets keep the general DType / Casting variants, and so
    ///     their snapshots).
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public instance NDArray.astype overload.</returns>
    private static bool IsCast(ParameterInfo p) =>
        p.Member is MethodInfo m && IsCastMember(m);

    /// <summary>Whether an overload is NDArray.astype, the cast family's member.</summary>
    /// <param name="m">The overload.</param>
    /// <returns>True for a public instance NDArray.astype overload.</returns>
    private static bool IsCastMember(MethodInfo m) =>
        m.DeclaringType == typeof(NDArray) && m.Name == "astype" && !m.IsStatic;

    /// <summary>
    ///     The dtypes NDArray.astype's dtype takes (the base, float64, first): every numeric storage dtype (NumSharp's
    ///     decimal and char included: the replay skips them), the fixed-width text classes at the pool's width (5), a narrower
    ///     one (truncation) and a wider one (NUL padding), the unsized 'S' / 'U' NumPy adapts to the array
    ///     (<c>PyArray_AdaptDescriptorToArray</c>), StringDType, a StringDType whose NA object is NaN, and null (the C#
    ///     refusal). Together with the pool's text operands (S5, U5, T), their special layouts (text that does not parse,
    ///     the empty string) and the special numbers (NaN, infinities, signed zeros), every text cast pair runs: numbers
    ///     formatted as text, text parsed into numbers, text into bool, S / U widths and classes, StringDType both ways.
    /// </summary>
    private static readonly string[] CastDTypeLabels =
    {
        "float64", "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32",
        "complex128", "decimal", "char", "S5", "S3", "S8", "U5", "U3", "U8", "S", "U", "T", "T_nan", "null",
    };

    /// <summary>
    ///     The descriptor of a <see cref="CastDTypeLabels"/> label: <c>np.dtype(label)</c>, StringDType with a NaN NA object
    ///     for "T_nan", null for "null".
    /// </summary>
    /// <param name="label">The label.</param>
    /// <returns>The descriptor, or null.</returns>
    private static DType MakeCastDType(string label) => label switch
    {
        "null" => null,
        "T_nan" => np.dtypes.StringDType(double.NaN),
        _ => np.dtype(label),
    };

    /// <summary>
    ///     NDArray.astype's casting rules: its default ('unsafe') first, the four stricter rules, NumPy 2.4's 'same_value'
    ///     (which astype accepts and no text cast satisfies) and a name NumPy refuses.
    /// </summary>
    private static readonly string[] CastCastingLabels = { "unsafe", "no", "equiv", "safe", "same_kind", "same_value", "bogus" };

    /// <summary>
    ///     The reduced pool of NDArray.astype (phase B, where the dtype varies): the general reduced pool with the special
    ///     layouts (text that does not parse, the empty string, NaN and infinities: the casts' errors and their
    ///     formatting) and the strided, reversed and broadcast views (the cast routes of non-contiguous sources).
    /// </summary>
    private static readonly string[] CastReducedLayouts =
        { "c2d", "c2d_special", "f2d", "strided", "reversed", "bcast", "0d", "0d_special", "empty", "3d", "3d_special" };

    /// <summary>
    ///     The small pool of NDArray.astype (phase C, where the dtype meets the casting rule, the order or copy): a matrix, its
    ///     special values, a 3-D operand and a 0-d one, over EVERY pool dtype (a cast rule depends on both classes).
    /// </summary>
    private static readonly string[] CastPairLayouts = { "c2d", "c2d_special", "3d", "0d" };

    /// <summary>
    ///     Whether a parameter belongs to a public static np.evaluate overload, whose <c>expr</c> (an
    ///     <see cref="NDExpr"/> built over the operand) and <c>operands</c> (the positional overload's operand list) carry
    ///     the operand: the first <see cref="NDArray"/> parameter after them is <c>out</c>, never the operand.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of np.evaluate.</returns>
    private static bool IsEvaluate(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name == "evaluate";

    /// <summary>
    ///     Whether an overload is elementwise with a where= mask: it takes an NDArray <c>where</c> and an <c>out</c> and no
    ///     <c>axis</c> (a ufunc, np.clip, np.evaluate). Such a call's fresh result has uninitialized masked-off elements;
    ///     a reduction's where= (np.sum, np.max, ...) excludes elements instead and defines every output.
    /// </summary>
    /// <param name="ps">The overload's parameters.</param>
    /// <returns>True for an elementwise overload with a where= mask.</returns>
    private static bool IsElementwiseWithWhere(ParameterInfo[] ps) =>
        ps.Any(q => q.Name == "where" && q.ParameterType == typeof(NDArray))
        && ps.Any(q => q.Name == "out")
        && !ps.Any(q => q.Name == "axis");

    /// <summary>Whether a parameter belongs to np.where, whose object <c>x</c> / <c>y</c> take <see cref="Role.WhereScalar"/>.</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.where overload.</returns>
    private static bool IsWhere(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name == "where";

    /// <summary>Whether a parameter belongs to np.extract, whose <c>arr</c> is a second operand (only its).</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.extract overload.</returns>
    private static bool IsExtract(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name == "extract";

    /// <summary>
    ///     Whether a parameter belongs to NDArray.real's or NDArray.imag's setter, whose <c>value</c> (the array copied into
    ///     the real or imaginary part) is a second operand.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for the value parameter of <c>set_real</c> / <c>set_imag</c>.</returns>
    private static bool IsComplexPartSetter(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(NDArray) && p.Member.Name is "set_real" or "set_imag";

    /// <summary>
    ///     Whether a parameter is NDArray.Contains's <c>value</c>: other members name an object parameter <c>value</c> with
    ///     other meanings (np.insert, np.can_cast, np.min_scalar_type), and their snapshots must not change.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for the parameter of the public instance method NDArray.Contains.</returns>
    private static bool IsContains(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(NDArray) && p.Member.Name == "Contains";

    /// <summary>
    ///     Whether a parameter belongs to the item-selection family - np.take, np.put, np.putmask, np.place, np.repeat,
    ///     np.choose, np.compress, np.resize and NDArray's take, put, repeat, choose, compress and resize (NumPy's
    ///     <c>item_selection.c</c>, <c>compiled_base.c</c>'s <c>arr_place</c> / <c>arr_putmask</c> and np.resize) - whose
    ///     <c>indices</c> / <c>index</c> / <c>mode</c> / <c>values</c> / <c>value</c> / <c>mask</c> / <c>vals</c> /
    ///     <c>repeats</c> / <c>choices</c> / <c>condition</c> / <c>new_shape</c> roles are theirs alone (see
    ///     <see cref="ItemArgLabels"/>): np.insert names a parameter <c>values</c> and np.linalg a <c>mode</c> with other
    ///     meanings, and their snapshots must not change.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.* or public instance NDArray overload of the family.</returns>
    private static bool IsItemSelection(ParameterInfo p) =>
        (p.Member.DeclaringType == typeof(np)
         && p.Member.Name is "take" or "put" or "putmask" or "place" or "repeat" or "choose" or "compress" or "resize")
        || (p.Member.DeclaringType == typeof(NDArray)
            && p.Member.Name is "take" or "put" or "repeat" or "choose" or "compress" or "resize")
        || IsAlongAxis(p) || IsFillDiagonal(p) || IsSelect(p) || IsPiecewise(p) || IsPad(p) || IsIndexTricks(p)
        || IsApplyAlongAxis(p) || IsIndexer(p);

    /// <summary>Whether a parameter belongs to np.select, whose condition list, choice list and default are all built from
    ///     the pool operand (<see cref="SelectCondLabels"/>, <see cref="SelectChoiceLabels"/>, <see cref="SelectDefaultLabels"/>):
    ///     it has no array parameter of its own.</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.select overload.</returns>
    private static bool IsSelect(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name == "select";

    /// <summary>Whether a parameter belongs to np.piecewise, whose x is the operand and whose condition list (in its
    ///     <c>NDArray[]</c> and bare <c>NDArray</c> forms), function list and extra args take
    ///     <see cref="PiecewiseCondListLabels"/>, <see cref="PiecewiseCondArrayLabels"/>, <see cref="PiecewiseFuncLabels"/> and
    ///     <see cref="PiecewiseArgsLabels"/> (the <c>bool</c> form takes the general bool role).</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.piecewise overload.</returns>
    private static bool IsPiecewise(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name == "piecewise";

    /// <summary>
    ///     Whether a parameter belongs to np.take_along_axis or np.put_along_axis, whose <c>indices</c> (and
    ///     np.put_along_axis's <c>values</c>) are built from the operand (<see cref="AlongIndicesLabels"/>,
    ///     <see cref="AlongValuesLabels"/>) through the item-selection wiring.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.take_along_axis / np.put_along_axis overload.</returns>
    private static bool IsAlongAxis(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name is "take_along_axis" or "put_along_axis";

    /// <summary>
    ///     Whether a parameter belongs to np.apply_along_axis, whose <c>func1d</c> comes from a fixed menu NumPy names alike
    ///     (<see cref="ApplyFuncLabels"/>, <see cref="ApplyArgsFuncLabels"/>) and whose extra <c>args</c> take
    ///     <see cref="ApplyArgsLabels"/>; its <c>arr</c> is the operand and its <c>axis</c> takes the general int axis role.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.apply_along_axis overload.</returns>
    private static bool IsApplyAlongAxis(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name == "apply_along_axis";

    /// <summary>Whether a parameter belongs to np.fill_diagonal, whose <c>val</c> takes
    ///     <see cref="FillDiagValueLabels"/> through the item-selection wiring.</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.fill_diagonal overload.</returns>
    private static bool IsFillDiagonal(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name == "fill_diagonal";

    /// <summary>Whether a parameter belongs to np.pad, whose <c>pad_width</c> (each C# form), <c>mode</c> (a name or a
    ///     callable), <c>constant_values</c> / <c>end_values</c> / <c>stat_length</c>, <c>reflect_type</c> and the callable's
    ///     <c>kwargs</c> take <see cref="PadArgLabels"/> through the item-selection wiring (<see cref="MakePadArg"/>).</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.pad overload.</returns>
    private static bool IsPad(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name == "pad";

    // ---- NDArray's indexers (P3 batch 8): get_Item / set_Item, __setitem__, GetIndices / SetIndices ------------------

    /// <summary>
    ///     Whether a parameter belongs to an indexer member of NDArray: the indexer's accessors <c>get_Item</c> /
    ///     <c>set_Item</c> (NumPy's <c>ndarray.__getitem__</c> / <c>__setitem__</c> with an NDArray value), the protocol
    ///     member <c>__setitem__</c> (any C# value, Python's None included) and the public fancy pair <c>GetIndices</c> /
    ///     <c>SetIndices</c>. Their key, index list and stored value take <see cref="IndexerArgLabels"/> through the
    ///     item-selection wiring (<see cref="MakeIndexerArg"/>); GetIndices' <c>out</c> takes the general out role, whose
    ///     variants follow the result of the same call without it.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of one of those public instance members declared on NDArray. <c>__getitem__</c> is left
    ///     out on purpose: it is <c>this[ItemKey(key)]</c>, the getter's own lines again.</returns>
    private static bool IsIndexer(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(NDArray)
        && p.Member.Name is "get_Item" or "set_Item" or "__setitem__" or "GetIndices" or "SetIndices";

    /// <summary>
    ///     The variant labels of an indexer parameter, the base first.
    /// </summary>
    /// <param name="p">The parameter (of an <see cref="IsIndexer"/> member).</param>
    /// <returns><see cref="IndexKeyLabels"/> for a key, <see cref="IndexListLabels"/> for the fancy pair's index list,
    ///     <see cref="IndexValueLabels"/> for an NDArray value, <see cref="IndexObjectValueLabels"/> for
    ///     <c>__setitem__</c>'s object value; null for GetIndices' <c>out</c>, which takes the general out role.</returns>
    private static string[] IndexerArgLabels(ParameterInfo p)
    {
        Type t = p.ParameterType;
        if (p.Name == "key")
            return IndexKeyLabels;
        if (p.Name == "indices" && t == typeof(NDArray[]))
            return IndexListLabels;
        if (p.Name is "value" or "values" && t == typeof(NDArray))
            return IndexValueLabels;
        if (p.Name == "value" && t == typeof(object))
            return IndexObjectValueLabels;
        return null;
    }

    /// <summary>
    ///     The indexer's keys, the base first, each an index tuple built from the operand's rank and dimensions
    ///     (<see cref="MakeIndexKey"/>).
    ///     Basic: integers of each C# kind (an int, a negative one, one past every axis, a long, a short, a ulong, a char),
    ///     slice notation (<c>":"</c>, <c>"1:"</c>, <c>"::-1"</c>, <c>"::2"</c>, <c>"1:3"</c>, and the lone <c>"..., 0"</c>
    ///     that holds a whole index), the ellipsis, a new axis, one zero per axis and one more than the rank, a C# tuple entry,
    ///     C# true / false (NumPy's 0-d bools), a <see cref="Slice"/> object, and the entries NumPy refuses or C# alone has
    ///     (a double, null, text that is not slice notation).
    ///     Advanced: integer index arrays (int64, int32, uint8; an int[], a long[], a List of int; empty, out of bounds, 2-D,
    ///     0-d, F-ordered 2-D, a strided view and a C-contiguous view at an offset), two arrays, an array beside a slice on
    ///     either side, split by a slice, after an ellipsis, a new axis or an integer, an F-ordered 2-D array after a slice,
    ///     masks (of the operand's shape, of the first axis as an NDArray and as a bool[], a wrong length, 0-d, beside a slice
    ///     on either side), and the index arrays NumPy refuses (float64, text).
    /// </summary>
    private static readonly string[] IndexKeyLabels =
    {
        "int0", "intneg1", "int99", "long1", "short1", "ulong1", "char1",
        "colon", "from1", "rev", "step2", "s1to3", "lone", "ell", "new", "allzero", "toomany", "tuple", "true", "false",
        "sliceobj", "double", "null", "badtext",
        "ia", "ia32", "iau8", "intarr", "longarr", "list", "iaempty", "iaoob", "ia2d", "ia0d", "iaf", "iastrided",
        "iaoffset", "twoia", "colonia", "iacolon", "split", "ellia", "newia", "intia", "coloniaf",
        "mask", "mask0", "boolarr", "maskbad", "mask0d", "maskcolon", "colonmask", "fia", "tia",
    };

    /// <summary>
    ///     The fancy pair's index lists (one array per leading axis), the base first: one int64 array, two, an empty one, an
    ///     out-of-bounds index, an int32 array, the float64 array NumPy refuses, a bool array (a mask of the first axis), one
    ///     more array than the operand has axes, no array, and null.
    /// </summary>
    private static readonly string[] IndexListLabels =
        { "one", "two", "empty", "oob", "i32", "f64", "bool", "toomany", "none", "null" };

    /// <summary>
    ///     The NDArray values the indexer setters store, the base first: the selection's own values reversed along its first
    ///     axis (<c>"selrev"</c>, built from the key of the same call by <see cref="ResolveIndexerSelection"/>), C# literals
    ///     through NDArray's implicit conversions (NumPy's Python int 7, Python float 2.5 and the str <c>"3"</c>), a 0-d
    ///     array of the operand's dtype (a strong value), a one-element array of it, an int64 array of five elements (which
    ///     broadcasts only to a last axis of 5 or 1), a one-element float64 array, and null.
    /// </summary>
    private static readonly string[] IndexValueLabels =
        { "selrev", "lit7", "lit2_5", "littext", "zero_d", "one", "five", "f64", "null" };

    /// <summary>
    ///     <c>__setitem__</c>'s values, the base first: the selection's values reversed (an NDArray, see
    ///     <see cref="IndexValueLabels"/>), C# numbers as Python's int and float, a bool, a negative int (out of range for
    ///     the unsigned dtypes), a str, a Python int past every C# integer (a <see cref="BigInteger"/>), a Python list, and
    ///     None.
    /// </summary>
    private static readonly string[] IndexObjectValueLabels =
        { "selrev", "int7", "dbl", "bool", "neg", "str", "big", "list", "none" };

    /// <summary>
    ///     The reduced pool of the indexer members (phase B): every layout of the full pool but two of the special-value ones,
    ///     because indexing moves elements whatever their values, while strides, offsets, broadcasting and the rank decide
    ///     which elements move.
    /// </summary>
    private static readonly string[] IndexerReducedLayouts =
        { "c2d", "f2d", "strided", "reversed", "bcast", "0d", "empty", "empty1d", "1d", "3d", "3d_special" };

    /// <summary>The pair pool of the indexer members (phase C: key x value, index list x out or values).</summary>
    private static readonly string[] IndexerPairLayouts = { "c2d", "3d", "0d", "1d" };

    /// <summary>
    ///     The stand-in <see cref="MakeIndexerArg"/> returns for a <c>"selrev"</c> value: the value depends on the key of the
    ///     same call, which the builder does not see, so <see cref="ResolveIndexerSelection"/> replaces it before the call.
    /// </summary>
    private static readonly object IndexSelectionPending = new();

    /// <summary>
    ///     Builds an indexer argument (<see cref="Role.ItemArg"/>) for an operand.
    /// </summary>
    /// <param name="p">The parameter, whose name and type select the label set (<see cref="IndexerArgLabels"/>).</param>
    /// <param name="label">The variant.</param>
    /// <param name="operand">The pool operand (the keys, masks and index lists are built from its rank and dimensions, the
    ///     zero and one-element values from its dtype).</param>
    /// <returns>The key as the <c>object[]</c> tuple (<c>__setitem__</c>'s object key receives the same tuple, which
    ///     <c>NDArray.ItemKey</c> keeps whole), the index list as an <c>NDArray[]</c>, or the value (for <c>"selrev"</c>,
    ///     <see cref="IndexSelectionPending"/>).</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static object MakeIndexerArg(ParameterInfo p, string label, NDArray operand)
    {
        if (p.Name == "key")
            return MakeIndexKey(label, operand);
        if (p.Name == "indices")
            return MakeIndexList(label, operand);
        if (p.ParameterType == typeof(object))
            return label switch
            {
                "selrev" => IndexSelectionPending,
                "int7" => 7,
                "dbl" => 2.5,
                "bool" => true,
                "neg" => -1,
                "str" => "3",
                "big" => BigInteger.Pow(2, 70),
                "list" => new object[] { 1, 2 },
                "none" => null,
                _ => throw new ArgumentException("unknown __setitem__ value variant " + label),
            };
        return label switch
        {
            "selrev" => IndexSelectionPending,
            // NDArray's implicit conversions, as C# source passes a literal to the setter: marked as the Python int, float
            // and str they stand for (NDArray.PythonLiteral, PythonStrLiteral).
            "lit7" => (NDArray)7,
            "lit2_5" => (NDArray)2.5,
            "littext" => (NDArray)"3",
            "zero_d" => np.zeros(Shape.Scalar, operand.dtype),
            "one" => np.zeros(new Shape(1), operand.dtype),
            "five" => np.arange(5L),
            "f64" => np.array(new[] { 0.5 }),
            "null" => null,
            _ => throw new ArgumentException("unknown indexer value variant " + label),
        };
    }

    /// <summary>
    ///     Builds an index tuple of <see cref="IndexKeyLabels"/> for an operand: the keys that need the rank (one zero per
    ///     axis, one more than the rank) or a length (a mask of the operand's shape or of its first or second axis) read them
    ///     from the operand; a 0-d operand's first-axis length is taken as 1.
    /// </summary>
    /// <param name="label">The variant.</param>
    /// <param name="operand">The pool operand.</param>
    /// <returns>The index tuple, one entry per C# argument of the indexer.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static object[] MakeIndexKey(string label, NDArray operand)
    {
        int nd = operand.ndim;
        long[] dims = operand.Shape.Dimensions;
        long d0 = nd >= 1 ? dims[0] : 1, d1 = nd >= 2 ? dims[1] : 1;
        NDArray Longs(params long[] v) => np.array(v);
        NDArray Alternating(long count) => np.array(Enumerable.Range(0, (int)count).Select(i => i % 2 == 0).ToArray());
        object[] Zeros(int count) => Enumerable.Repeat((object)0, count).ToArray();
        // [[0, 1], [1, 0]] laid out in F order: its memory order differs from its C order, which a gather that reads an
        // index array by memory instead of by position gets wrong.
        NDArray FOrderedIndex() => np.asfortranarray(Longs(0, 1, 1, 0).reshape(new Shape(2, 2)));
        return label switch
        {
            "int0" => new object[] { 0 },
            "intneg1" => new object[] { -1 },
            "int99" => new object[] { 99 },
            "long1" => new object[] { 1L },
            "short1" => new object[] { (short)1 },
            "ulong1" => new object[] { 1UL },
            "char1" => new object[] { (char)1 },
            "colon" => new object[] { ":" },
            "from1" => new object[] { "1:" },
            "rev" => new object[] { "::-1" },
            "step2" => new object[] { "::2" },
            "s1to3" => new object[] { "1:3" },
            "lone" => new object[] { "..., 0" },
            "ell" => new object[] { Slice.Ellipsis },
            "new" => new object[] { Slice.NewAxis },
            "allzero" => Zeros(nd),
            "toomany" => Zeros(nd + 1),
            "tuple" => new object[] { (0, ":") },
            "true" => new object[] { true },
            "false" => new object[] { false },
            "sliceobj" => new object[] { new Slice(1, null, 2) },
            "double" => new object[] { 1.5 },
            "null" => new object[] { null },
            "badtext" => new object[] { "x" },
            "ia" => new object[] { Longs(0, -1) },
            "ia32" => new object[] { np.array(new[] { 1, 0 }) },
            "iau8" => new object[] { np.array(new byte[] { 1 }) },
            "intarr" => new object[] { new[] { 0, 0 } },
            "longarr" => new object[] { new[] { -1L } },
            "list" => new object[] { new List<int> { 0, 1 } },
            "iaempty" => new object[] { np.array(Array.Empty<long>()) },
            "iaoob" => new object[] { Longs(0, 99) },
            "ia2d" => new object[] { Longs(0, -1).reshape(new Shape(2, 1)) },
            "ia0d" => new object[] { np.array(1L) },
            "iaf" => new object[] { FOrderedIndex() },
            "iastrided" => new object[] { Longs(0, 9, 1, 9)["::2"] },
            "iaoffset" => new object[] { Longs(9, 0, 1)["1:"] },
            "twoia" => new object[] { Longs(0, 1), Longs(1, 0) },
            "colonia" => new object[] { ":", Longs(0, -1) },
            "iacolon" => new object[] { Longs(0, -1), ":" },
            "split" => new object[] { Longs(0), ":", Longs(0) },
            "ellia" => new object[] { Slice.Ellipsis, Longs(0, -1) },
            "newia" => new object[] { Slice.NewAxis, Longs(0, -1) },
            "intia" => new object[] { 0, Longs(0, -1) },
            "coloniaf" => new object[] { ":", FOrderedIndex() },
            "mask" => new object[] { nd == 0 ? np.array(true) : Alternating(operand.size).reshape(new Shape(dims)) },
            "mask0" => new object[] { Alternating(d0) },
            "boolarr" => new object[] { Enumerable.Range(0, (int)d0).Select(i => i % 2 == 0).ToArray() },
            "maskbad" => new object[] { Alternating(d0 + 1) },
            "mask0d" => new object[] { np.array(true) },
            "maskcolon" => new object[] { Alternating(d0), ":" },
            "colonmask" => new object[] { ":", Alternating(d1) },
            "fia" => new object[] { np.array(new[] { 0.0 }) },
            "tia" => new object[] { np.array(new[] { "0" }) },
            _ => throw new ArgumentException("unknown index key variant " + label),
        };
    }

    /// <summary>
    ///     Builds an index list of <see cref="IndexListLabels"/> for GetIndices / SetIndices.
    /// </summary>
    /// <param name="label">The variant.</param>
    /// <param name="operand">The pool operand (its rank sizes the over-long list, its first axis the bool array).</param>
    /// <returns>The index arrays, one per leading axis, or null.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray[] MakeIndexList(string label, NDArray operand)
    {
        int nd = operand.ndim;
        long d0 = nd >= 1 ? operand.Shape.Dimensions[0] : 1;
        NDArray Longs(params long[] v) => np.array(v);
        return label switch
        {
            "one" => new[] { Longs(0, -1) },
            "two" => new[] { Longs(0, 1), Longs(1, 0) },
            "empty" => new[] { np.array(Array.Empty<long>()) },
            "oob" => new[] { Longs(0, 99) },
            "i32" => new[] { np.array(new[] { 1, 0 }) },
            "f64" => new[] { np.array(new[] { 0.0 }) },
            "bool" => new[] { np.array(Enumerable.Range(0, (int)d0).Select(i => i % 2 == 0).ToArray()) },
            "toomany" => Enumerable.Range(0, nd + 1).Select(_ => Longs(0)).ToArray(),
            "none" => Array.Empty<NDArray>(),
            "null" => null,
            _ => throw new ArgumentException("unknown index list variant " + label),
        };
    }

    /// <summary>
    ///     Replaces a pending <c>"selrev"</c> value (<see cref="IndexSelectionPending"/>) with the selection's own values,
    ///     reversed along their first axis, for the key or index list of the same call: the store then writes the selection's
    ///     values back in another order, so a store that drops, repeats or misplaces an element changes the operand's hash.
    /// </summary>
    /// <param name="ps">The member's parameters (the key is named <c>key</c>, the index list <c>indices</c>).</param>
    /// <param name="argv">The call's arguments, the key already in place; a pending value is replaced in place.</param>
    /// <param name="operand">The pool operand the selection is read from (a copy: the call's store has not run).</param>
    /// <remarks>
    ///     A key the getter refuses (null, an out-of-bounds index, a refused entry) has no selection: its value is then a 0-d
    ///     zero of the operand's dtype, so the setter still meets the key and refuses it as it would any value.
    /// </remarks>
    private static void ResolveIndexerSelection(ParameterInfo[] ps, object[] argv, NDArray operand)
    {
        for (int i = 0; i < argv.Length; i++)
        {
            if (!ReferenceEquals(argv[i], IndexSelectionPending))
                continue;
            int keyAt = Array.FindIndex(ps, q => q.Name is "key" or "indices");
            // An NDArray[] index list is the indexer's tuple of index arrays (array covariance), as SetIndices' own
            // StringDType route reads it.
            object[] key = keyAt < 0 ? null : argv[keyAt] as object[];
            NDArray value;
            try
            {
                NDArray selection = operand[key].copy();
                value = selection.ndim >= 1 ? np.flip(selection, 0).copy() : selection;
            }
            catch (Exception)
            {
                value = np.zeros(Shape.Scalar, operand.dtype);
            }
            argv[i] = value;
        }
    }

    /// <summary>
    ///     The variant labels of an item-selection parameter (<see cref="Role.ItemArg"/>), the base first, or null for a
    ///     parameter the family's roles do not read (its <c>a</c> / <c>arr</c> operand, <c>axis</c>, <c>out</c>,
    ///     <c>refcheck</c> take the general roles).
    /// </summary>
    /// <param name="p">The parameter (of an <see cref="IsItemSelection"/> overload).</param>
    /// <returns>The labels <see cref="MakeItemArg"/> builds, or null.</returns>
    private static string[] ItemArgLabels(ParameterInfo p)
    {
        Type t = p.ParameterType;
        string n = p.Name;
        // NDArray's indexers first: their key, indices and value names are their own (IndexerArgLabels).
        if (IsIndexer(p)) return IndexerArgLabels(p);
        // The along-axis pair and np.fill_diagonal first: their parameter names are the item-selection family's own.
        if (IsAlongAxis(p) && n == "indices" && t == typeof(NDArray)) return AlongIndicesLabels;
        if (IsAlongAxis(p) && n == "values" && t == typeof(NDArray)) return AlongValuesLabels;
        if (IsFillDiagonal(p) && n == "val" && t == typeof(object)) return FillDiagValueLabels;
        if (IsSelect(p))
            return n switch { "condlist" => SelectCondLabels, "choicelist" => SelectChoiceLabels, "default" => SelectDefaultLabels, _ => null };
        if (IsPiecewise(p))
            return n == "condlist" && t == typeof(NDArray[]) ? PiecewiseCondListLabels
                : n == "condlist" && t == typeof(NDArray) ? PiecewiseCondArrayLabels
                : n == "funclist" ? PiecewiseFuncLabels
                : n == "args" ? PiecewiseArgsLabels
                : null;
        if (IsPad(p))
            return PadArgLabels(p);
        if (IsIndexTricks(p))
            return IndexTricksArgLabels(p);
        if (IsApplyAlongAxis(p))
            return n == "func1d" && t == typeof(Func<NDArray, NDArray>) ? ApplyFuncLabels
                : n == "func1d" && t == typeof(Func<NDArray, object[], NDArray>) ? ApplyArgsFuncLabels
                : n == "args" && t == typeof(object[]) ? ApplyArgsLabels
                : null;
        if (IsAlongAxis(p) || IsFillDiagonal(p)) return null;
        if (n == "indices" && t == typeof(NDArray)) return ItemIndicesLabels;
        if (n == "index" && t == typeof(long)) return ItemIndexLabels;
        if (n == "mode" && t == typeof(string)) return ItemModeLabels;
        if (n == "value" && t == typeof(object)) return ItemPutValueLabels;
        if (n == "mask" && t == typeof(NDArray)) return ItemMaskLabels;
        if (n is "values" or "vals" && t == typeof(NDArray)) return ItemValuesLabels;
        if (n is "values" or "vals" && t == typeof(Array)) return ItemArrayValuesLabels;
        if (n == "repeats" && (t == typeof(int) || t == typeof(long))) return ItemRepeatCountLabels;
        if (n == "repeats" && t == typeof(NDArray)) return ItemRepeatArrayLabels;
        if (n == "choices" && t == typeof(NDArray[])) return ItemChoiceListLabels;
        if (n == "choices" && t == typeof(object[])) return ItemChoiceObjectLabels;
        if (n == "choices" && t == typeof(NDArray)) return ItemChoiceStackLabels;
        if (n == "condition" && t == typeof(NDArray)) return ItemConditionLabels;
        if (n == "new_shape" && (t == typeof(Shape) || t == typeof(long[]))) return ItemNewShapeLabels;
        return null;
    }

    /// <summary>np.take / np.put's index arrays: positions inside every pool axis, negative ones, one past every axis
    ///     either way (the clip modes' cases), a 0-d index, none, a 2-D index, the integer dtypes NumPy casts (uint64 under
    ///     take's same_kind, refused by put's safe rule; uint8), the dtypes it refuses (float, bool, text), a strided view,
    ///     and null.</summary>
    private static readonly string[] ItemIndicesLabels =
    {
        "i64[1]", "i64[0,2]", "i32[2,0]", "i64[-1]", "i64[99]", "i64[-99]", "0d_i64_1", "i64[]", "2d_i64", "u64[1]",
        "u8[1]", "f64[1.0]", "bool[T,F]", "i64_strided", "U1[1]", "null",
    };

    /// <summary>
    ///     np.take_along_axis / np.put_along_axis's index arrays, built from the operand along its last axis (the axis
    ///     default): a permutation of every row ("ramp", reversed positions), one index per row (0, -1, and 99 past every
    ///     axis), the ramp as a negative-stride view, in int32 and in the unsigned dtypes NumPy accepts, the float and bool
    ///     dtypes it refuses ("`indices` must be an integer array"), a 1-D index (axis=None's form, a rank mismatch
    ///     otherwise), one index broadcast over every other axis, and null.
    /// </summary>
    private static readonly string[] AlongIndicesLabels =
        { "ramp", "zero", "neg1", "oob", "ramp_flip", "ramp_i32", "zero_u8", "zero_u64", "zero_f64", "zero_bool", "flat", "bcast", "null" };

    /// <summary>np.put_along_axis's values: the operand's own copy (the ramp's shape), a 0-d int64, a one-element float, a
    ///     0-d text value, an empty array (no broadcast), and null.</summary>
    private static readonly string[] AlongValuesLabels = { "same", "0d_7", "f64[2.5]", "0d_text", "i64[]", "null" };

    /// <summary>
    ///     np.apply_along_axis's one-argument functions, a menu NumPy names alike (the Python twin in each delegate's
    ///     comment): a 0-d result (the axis is dropped: np.sum, the first element, a constant int64, a constant str), a 1-D
    ///     result as long as the slice (negate, np.cumsum, a cast to float32, the slice itself) or shorter (the first two
    ///     elements), and a 2-D result (np.outer of the slice with itself: two dims replace the axis).
    /// </summary>
    private static readonly string[] ApplyFuncLabels =
        { "sum", "neg", "cumsum", "head2", "outer", "same", "first", "const", "text", "f32" };

    /// <summary>np.apply_along_axis's function with extra args: the slice plus the Python sum of the args
    ///     (<c>lambda v, *a: v + sum(a)</c>).</summary>
    private static readonly string[] ApplyArgsFuncLabels = { "addargs" };

    /// <summary>np.apply_along_axis's extra args: none (the base; NumPy's sum of none is the Python int 0), one Python
    ///     int, two Python ints.</summary>
    private static readonly string[] ApplyArgsLabels = { "none", "one", "two" };

    /// <summary>
    ///     Builds an np.apply_along_axis argument: a function of the menu (static delegates, so no call captures the
    ///     operand) or the extra args.
    /// </summary>
    /// <param name="p">The parameter (<c>func1d</c> or <c>args</c>).</param>
    /// <param name="label">The variant.</param>
    /// <returns>The delegate, or the <c>object[]</c> the params parameter receives.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static object MakeApplyArg(ParameterInfo p, string label)
    {
        if (p.Name == "args")
            return label switch
            {
                "none" => Array.Empty<object>(),
                "one" => new object[] { 1 },
                "two" => new object[] { 1, 2 },
                _ => throw new ArgumentException("unknown apply_along_axis args variant " + label),
            };
        if (p.ParameterType == typeof(Func<NDArray, object[], NDArray>))
            return label == "addargs" ? ApplyAddArgs : throw new ArgumentException("unknown apply_along_axis func1d variant " + label);
        return label switch
        {
            "sum" => ApplySum,
            "neg" => ApplyNeg,
            "cumsum" => ApplyCumsum,
            "head2" => ApplyHead2,
            "outer" => ApplyOuter,
            "same" => ApplySame,
            "first" => ApplyFirst,
            "const" => ApplyConst,
            "text" => ApplyText,
            "f32" => ApplyF32,
            _ => throw new ArgumentException("unknown apply_along_axis func1d variant " + label),
        };
    }

    /// <summary><c>lambda v: np.sum(v)</c></summary>
    private static readonly Func<NDArray, NDArray> ApplySum = v => np.sum(v);

    /// <summary><c>lambda v: -v</c></summary>
    private static readonly Func<NDArray, NDArray> ApplyNeg = v => -v;

    /// <summary><c>lambda v: np.cumsum(v)</c></summary>
    private static readonly Func<NDArray, NDArray> ApplyCumsum = v => np.cumsum(v);

    /// <summary><c>lambda v: v[:2]</c></summary>
    private static readonly Func<NDArray, NDArray> ApplyHead2 = v => v[":2"];

    /// <summary><c>lambda v: np.outer(v, v)</c></summary>
    private static readonly Func<NDArray, NDArray> ApplyOuter = v => np.outer(v, v);

    /// <summary><c>lambda v: v</c></summary>
    private static readonly Func<NDArray, NDArray> ApplySame = v => v;

    /// <summary><c>lambda v: v[0]</c></summary>
    private static readonly Func<NDArray, NDArray> ApplyFirst = v => v[0];

    /// <summary><c>lambda v: np.array(7)</c> (int64)</summary>
    private static readonly Func<NDArray, NDArray> ApplyConst = v => np.array(7L);

    /// <summary><c>lambda v: np.array('ab')</c></summary>
    private static readonly Func<NDArray, NDArray> ApplyText = v => np.array("ab");

    /// <summary><c>lambda v: v.astype(np.float32)</c></summary>
    private static readonly Func<NDArray, NDArray> ApplyF32 = v => v.astype(np.float32);

    /// <summary><c>lambda v, *a: v + sum(a)</c> (the args are Python ints)</summary>
    private static readonly Func<NDArray, object[], NDArray> ApplyAddArgs = (v, a) => v + a.Sum(x => (int)x);

    /// <summary>np.fill_diagonal's val (NumPy's <c>a.flat[...] = val</c>): a Python int first, then a float, a bool, a
    ///     complex, a str, None, an int past int8 and one past every C# integer, a NumPy float16, a C# array (tiled
    ///     cyclically), an empty one (a no-op), the operand's own copy (raveled and tiled), a 0-d int64, and Python lists
    ///     (object[]: NumPy stores every item by the dtype's setitem) - of ints with one past int8, of a str, of a float
    ///     and a NaN, of None, and an empty one.</summary>
    private static readonly string[] FillDiagValueLabels =
    {
        "7", "2.5", "true", "1+2j", "x", "null", "300", "2**70", "half1.5", "f64{1,2}", "i32{}", "same", "0d_5",
        "obj{300,2}", "obj{x}", "obj{1.5,nan}", "obj{null}", "obj{}",
    };

    /// <summary>
    ///     Builds an np.take_along_axis / np.put_along_axis argument for an operand: its index array (see
    ///     <see cref="AlongIndicesLabels"/>) or np.put_along_axis's values (<see cref="AlongValuesLabels"/>).
    /// </summary>
    /// <param name="p">The parameter (<c>indices</c> or <c>values</c>).</param>
    /// <param name="label">The variant.</param>
    /// <param name="operand">The pool operand (the indices index its last axis; "same" copies it).</param>
    /// <returns>The array, or null.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static object MakeAlongArg(ParameterInfo p, string label, NDArray operand)
    {
        long[] dims = operand.Shape.Dimensions;
        long last = dims.Length == 0 ? 1 : dims[^1];
        long size = 1;
        foreach (long d in dims) size *= d;
        // One index per row: the operand's dims with the last one 1 (a 0-d operand's index is 0-d).
        long[] oneDims = dims.Length == 0 ? Array.Empty<long>() : dims[..^1].Append(1L).ToArray();
        long oneSize = 1;
        foreach (long d in oneDims) oneSize *= d;
        NDArray Filled(long value) => np.array(Enumerable.Repeat(value, (int)oneSize).ToArray()).reshape(new Shape(oneDims));
        NDArray Ramp()
        {
            // Every row reversed: position i of a row reads element last - 1 - i (a permutation along the last axis).
            var v = new long[size];
            for (long i = 0; i < size; i++)
                v[i] = last - 1 - i % last;
            return np.array(v).reshape(new Shape(dims));
        }
        if (p.Name == "indices")
            return label switch
            {
                "ramp" => Ramp(),
                "zero" => Filled(0),
                "neg1" => Filled(-1),
                "oob" => Filled(99),
                "ramp_flip" => dims.Length == 0 ? Ramp() : Ramp()["..., ::-1"],
                "ramp_i32" => Ramp().astype(np.int32),
                "zero_u8" => Filled(0).astype(np.uint8),
                "zero_u64" => Filled(0).astype(np.uint64),
                "zero_f64" => Filled(0).astype(np.float64),
                "zero_bool" => Filled(0).astype(np.@bool),
                "flat" => np.array(new long[] { 0 }),
                "bcast" => np.zeros(new Shape(Enumerable.Repeat(1L, dims.Length).ToArray()), np.int64),
                "null" => null,
                _ => throw new ArgumentException("unknown along-axis indices variant " + label),
            };
        return label switch
        {
            "same" => operand.copy(),
            "0d_7" => np.array(7L),
            "f64[2.5]" => np.array(new[] { 2.5 }),
            "0d_text" => np.array("x"),
            "i64[]" => np.array(Array.Empty<long>()),
            "null" => null,
            _ => throw new ArgumentException("unknown along-axis values variant " + label),
        };
    }

    /// <summary>
    ///     np.select's condition lists, built from the pool operand's shape: alternating True / False and its complement
    ///     (the base: both choices win somewhere), all True twice (the first condition wins), none true (the default
    ///     everywhere), an int64 first condition (NumPy's "invalid entry 0 in condlist" TypeError), rows of the last axis
    ///     (broadcast), two 0-d conditions, conditions of a shape that does not broadcast, one condition and none (the
    ///     length errors), and null.
    /// </summary>
    private static readonly string[] SelectCondLabels =
        { "alt2", "first", "none", "int", "row", "0d", "badshape", "one", "empty", "null" };

    /// <summary>
    ///     np.select's choice lists: the operand's copy and its zeros (the base), two weak Python scalars (an int and a
    ///     float), the operand beside a Python int past int8, the operand beside a strong 0-d float64, two strs, a NumPy
    ///     float16 beside a Python int, float64 rows of the last axis, choices of a shape that does not broadcast, a null
    ///     entry (NumPy's object array), one choice (the length error), and null.
    /// </summary>
    private static readonly string[] SelectChoiceLabels =
        { "pair", "scalars", "big", "strong0d", "text", "half", "row", "badshape", "nullentry", "one", "null" };

    /// <summary>np.select's default: null (NumPy's default=0) first, then a Python int, one past int8, a float, a complex,
    ///     a bool, a str, a strong 0-d int64, the operand's copy, a float64 row of the last axis and an array of a shape
    ///     that does not broadcast.</summary>
    private static readonly string[] SelectDefaultLabels =
        { "null", "0", "1000", "2.5", "c128", "true", "x", "0d_int64", "same", "row", "badshape" };

    /// <summary>
    ///     np.piecewise's condition lists (the NDArray[] form), built from the operand's shape: alternating True / False and
    ///     its complement (the base), all True twice (the last condition wins), none true, int64 conditions (their truth),
    ///     two 0-d conditions, one condition and three (the function-count errors against two functions), none (NumPy's
    ///     condlist[0] IndexError), and null.
    /// </summary>
    private static readonly string[] PiecewiseCondListLabels = { "alt2", "first2", "none2", "int2", "0d2", "one", "three", "empty", "null" };

    /// <summary>
    ///     np.piecewise's bare condition array (NumPy's promotion of a non-list condlist), built from the operand's shape:
    ///     the two conditions stacked on a new first axis (the base: split back into two), the operand-shaped mask (one
    ///     condition for a 1-D x, its rows for a 2-D+ x), a 0-d condition (NumPy's "too many indices" IndexError), an empty
    ///     first axis (NumPy's "index 0 is out of bounds" IndexError), and null.
    /// </summary>
    private static readonly string[] PiecewiseCondArrayLabels = { "stack2", "alt", "0d", "empty0", "null" };

    /// <summary>
    ///     np.piecewise's function lists, from a fixed menu NumPy can name the same way: negate and square (callables, the
    ///     base), two weak constants (an int and a float, stored into x's dtype), three entries (the third is the default),
    ///     a Python int past int8 (NumPy's OverflowError for a narrow integer x), a complex constant (its TypeError for a
    ///     real x), two strong 0-d arrays, a callable that adds the first extra arg, one entry (the function-count error),
    ///     a null entry, and null.
    /// </summary>
    private static readonly string[] PiecewiseFuncLabels =
        { "neg_sq", "consts", "default3", "big", "c128", "arrays", "args", "one", "nullentry", "null" };

    /// <summary>np.piecewise's extra args: none (the base), one Python int, two Python ints.</summary>
    private static readonly string[] PiecewiseArgsLabels = { "none", "one", "two" };

    /// <summary>np.piecewise's callable that negates its slice (NumPy's <c>lambda v: -v</c>).</summary>
    private static readonly Func<NDArray, NDArray> PiecewiseNegate = v => -v;

    /// <summary>np.piecewise's callable that squares its slice (NumPy's <c>lambda v: v * v</c>).</summary>
    private static readonly Func<NDArray, NDArray> PiecewiseSquare = v => v * v;

    /// <summary>np.piecewise's callable that adds the first extra arg (NumPy's <c>lambda v, *a: v + a[0] if a else v</c>,
    ///     a Python int: the C# int converts to its weak literal array).</summary>
    private static readonly Func<NDArray, object[], NDArray> PiecewiseAddArg =
        (v, a) => a.Length == 0 ? v : v + (int)a[0];

    /// <summary>np.pad's integer pad_width (both sides of every axis): one first, none, three, and a negative width
    ///     (NumPy's ValueError).</summary>
    private static readonly string[] PadWidthIntLabels = { "1", "0", "3", "-1" };

    /// <summary>np.pad's <c>int[]</c> pad_width (an array of shape (n,), which <c>_as_pairs</c> broadcasts to (ndim, 2)): a
    ///     before / after pair first, one width, two zeros, a negative width, three widths (NumPy's broadcast error), none,
    ///     and null (NumPy's None: an object array, "`pad_width` must be of integral type.").</summary>
    private static readonly string[] PadWidthArrayLabels = { "{1,2}", "{2}", "{0,0}", "{1,-1}", "{1,2,3}", "{}", "null" };

    /// <summary>np.pad's <c>int[,]</c> pad_width, built from the operand's rank: one (before, after) row per axis first,
    ///     one row for every axis, zeros, a negative width, one row too many (NumPy's broadcast error), and null.</summary>
    private static readonly string[] PadWidthTableLabels = { "rows", "one_row", "zeros", "neg", "extra_row", "null" };

    /// <summary>np.pad's <c>(before, after)</c> pad_width: (1, 2) first, none, before only, a negative before.</summary>
    private static readonly string[] PadWidthTupleLabels = { "(1,2)", "(0,0)", "(3,0)", "(-1,1)" };

    /// <summary>np.pad's dict pad_width (NumPy 2.3+): axis 0 by one first, the last axis by (1, 2), two axes, an axis past
    ///     every rank (NumPy's IndexError), a negative width, a bool width (an int to Python's match), a float width
    ///     (NumPy's assert_never), none, and null.</summary>
    private static readonly string[] PadWidthDictLabels =
        { "{0:1}", "{-1:(1,2)}", "{0:(2,0),-1:1}", "{5:1}", "{0:-1}", "{0:true}", "{0:2.5}", "{}", "null" };

    /// <summary>np.pad's mode names: 'constant' (the default) first, then every other mode NumPy has, a name it refuses,
    ///     and null (NumPy's None, which it refuses the same way).</summary>
    private static readonly string[] PadModeLabels =
    {
        "constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect", "symmetric", "wrap", "empty",
        "bogus", "null",
    };

    /// <summary>np.pad's callable modes: one that writes 7 into each vector's pad area (the base), one that writes its
    ///     kwargs' "v" (7 without it), and null (NumPy's None).</summary>
    private static readonly string[] PadFuncLabels = { "fill7", "fill_kw", "null" };

    /// <summary>np.pad's constant_values: not given (null) first, a Python int, a float, an int past int8, a complex, a
    ///     str, a bool, a (before, after) pair, one pair per axis (built from the operand's rank) and a 0-d array.</summary>
    private static readonly string[] PadConstantLabels = { "null", "5", "2.5", "1000", "c128", "x", "true", "(1,2)", "rows", "0d_5" };

    /// <summary>np.pad's end_values (linear_ramp): not given first, a Python int, a float, a (before, after) pair, a str.</summary>
    private static readonly string[] PadEndLabels = { "null", "5", "2.5", "(1,2)", "x" };

    /// <summary>np.pad's stat_length (the statistic modes): not given (the whole axis) first, one, a (before, after) pair,
    ///     zero (NumPy's "stat_length of 0 yields no value for padding" for maximum / minimum), a negative length (its
    ///     ValueError), and one longer than every axis.</summary>
    private static readonly string[] PadStatLabels = { "null", "1", "(1,2)", "0", "-1", "100" };

    /// <summary>np.pad's reflect_type: 'even' (the default) first, 'odd', a value NumPy reads as even, and null.</summary>
    private static readonly string[] PadReflectLabels = { "even", "odd", "bogus", "null" };

    /// <summary>The kwargs np.pad hands a callable mode: none first, then {"v": 9}.</summary>
    private static readonly string[] PadKwargsLabels = { "null", "v9" };

    /// <summary>The labels of an np.pad parameter (<see cref="IsPad"/>), the base first, or null for its operand.</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>The labels <see cref="MakePadArg"/> builds, or null.</returns>
    private static string[] PadArgLabels(ParameterInfo p)
    {
        Type t = p.ParameterType;
        return p.Name switch
        {
            "pad_width" when t == typeof(int) => PadWidthIntLabels,
            "pad_width" when t == typeof(int[]) => PadWidthArrayLabels,
            "pad_width" when t == typeof(int[,]) => PadWidthTableLabels,
            "pad_width" when t == typeof((int, int)) => PadWidthTupleLabels,
            "pad_width" when t == typeof(IDictionary<int, object>) => PadWidthDictLabels,
            "mode" when t == typeof(string) => PadModeLabels,
            "mode" when t == typeof(np.PadFunc) => PadFuncLabels,
            "constant_values" => PadConstantLabels,
            "end_values" => PadEndLabels,
            "stat_length" => PadStatLabels,
            "reflect_type" => PadReflectLabels,
            "kwargs" => PadKwargsLabels,
            _ => null,
        };
    }

    /// <summary>np.pad's callable that writes 7 into each vector's pad area (NumPy's
    ///     <c>def f(v, w, iaxis, kw): v[:w[0]] = 7; v[v.shape[0] - w[1]:] = 7</c>, each side only when it is wider than 0).</summary>
    private static readonly np.PadFunc PadFill7 = (v, w, axis, kw) => PadFillSides(v, w, 7);

    /// <summary>np.pad's callable that writes its kwargs' "v" (7 without one) into each vector's pad area (NumPy's
    ///     <c>kw.get('v', 7)</c>).</summary>
    private static readonly np.PadFunc PadFillKw = (v, w, axis, kw) =>
        PadFillSides(v, w, kw is IDictionary<string, object> d && d.TryGetValue("v", out var value) && value is int i ? i : 7);

    /// <summary>Writes a Python int (the C# int's weak literal array, packed by the store as NumPy packs a Python int) into
    ///     the first <c>before</c> and the last <c>after</c> elements of a 1-D vector.</summary>
    /// <param name="v">The vector (a view into the padded array).</param>
    /// <param name="w">The (before, after) widths.</param>
    /// <param name="value">The value.</param>
    private static void PadFillSides(NDArray v, (int before, int after) w, int value)
    {
        long len = v.Shape.Dimensions[^1];
        if (w.before > 0)
            v[$":{w.before}"] = value;
        if (w.after > 0)
            v[$"{len - w.after}:"] = value;
    }

    /// <summary>
    ///     Builds an np.pad argument for an operand (see the label sets above).
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <param name="label">The variant.</param>
    /// <param name="operand">The pool operand: the per-axis tables take its rank.</param>
    /// <returns>The argument: an int, a C# array, a tuple, a dictionary, a string, a callable, a boxed value, an array, or
    ///     null.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static object MakePadArg(ParameterInfo p, string label, NDArray operand)
    {
        Type t = p.ParameterType;
        int ndim = Math.Max(operand.ndim, 1);
        int[,] Table(Func<int, (int before, int after)> row, int rows)
        {
            var table = new int[rows, 2];
            for (int i = 0; i < rows; i++)
                (table[i, 0], table[i, 1]) = row(i);
            return table;
        }
        switch (p.Name)
        {
            case "pad_width" when t == typeof(int):
                return int.Parse(label, CultureInfo.InvariantCulture);
            case "pad_width" when t == typeof(int[]):
                return label switch
                {
                    "{1,2}" => new[] { 1, 2 },
                    "{2}" => new[] { 2 },
                    "{0,0}" => new[] { 0, 0 },
                    "{1,-1}" => new[] { 1, -1 },
                    "{1,2,3}" => new[] { 1, 2, 3 },
                    "{}" => Array.Empty<int>(),
                    "null" => null,
                    _ => throw new ArgumentException("unknown pad_width array variant " + label),
                };
            case "pad_width" when t == typeof(int[,]):
                return label switch
                {
                    "rows" => Table(i => (i + 1, i % 2), ndim),
                    "one_row" => Table(i => (1, 2), 1),
                    "zeros" => Table(i => (0, 0), ndim),
                    "neg" => Table(i => (i == 0 ? -1 : 1, 1), ndim),
                    "extra_row" => Table(i => (1, 1), ndim + 1),
                    "null" => null,
                    _ => throw new ArgumentException("unknown pad_width table variant " + label),
                };
            case "pad_width" when t == typeof((int, int)):
                return label switch
                {
                    "(1,2)" => (1, 2),
                    "(0,0)" => (0, 0),
                    "(3,0)" => (3, 0),
                    "(-1,1)" => (-1, 1),
                    _ => throw new ArgumentException("unknown pad_width tuple variant " + label),
                };
            case "pad_width" when t == typeof(IDictionary<int, object>):
                return label switch
                {
                    "{0:1}" => new Dictionary<int, object> { [0] = 1 },
                    "{-1:(1,2)}" => new Dictionary<int, object> { [-1] = (1, 2) },
                    "{0:(2,0),-1:1}" => new Dictionary<int, object> { [0] = (2, 0), [-1] = 1 },
                    "{5:1}" => new Dictionary<int, object> { [5] = 1 },
                    "{0:-1}" => new Dictionary<int, object> { [0] = -1 },
                    "{0:true}" => new Dictionary<int, object> { [0] = true },
                    "{0:2.5}" => new Dictionary<int, object> { [0] = 2.5 },
                    "{}" => new Dictionary<int, object>(),
                    "null" => null,
                    _ => throw new ArgumentException("unknown pad_width dict variant " + label),
                };
            case "mode" when t == typeof(string):
                return label == "null" ? null : label;
            case "mode":
                return label switch
                {
                    "fill7" => PadFill7,
                    "fill_kw" => PadFillKw,
                    "null" => null,
                    _ => throw new ArgumentException("unknown pad callable variant " + label),
                };
            case "constant_values":
                return label switch
                {
                    "null" => null,
                    "5" => 5,
                    "2.5" => 2.5,
                    "1000" => 1000,
                    "c128" => new Complex(1, 2),
                    "x" => "x",
                    "true" => true,
                    "(1,2)" => (1, 2),
                    "rows" => Table(i => (i + 1, -(i + 1)), ndim),
                    "0d_5" => np.array(5L),
                    _ => throw new ArgumentException("unknown constant_values variant " + label),
                };
            case "end_values":
                return label switch
                {
                    "null" => null,
                    "5" => 5,
                    "2.5" => 2.5,
                    "(1,2)" => (1, 2),
                    "x" => "x",
                    _ => throw new ArgumentException("unknown end_values variant " + label),
                };
            case "stat_length":
                return label switch
                {
                    "null" => null,
                    "1" => 1,
                    "(1,2)" => (1, 2),
                    "0" => 0,
                    "-1" => -1,
                    "100" => 100,
                    _ => throw new ArgumentException("unknown stat_length variant " + label),
                };
            case "reflect_type":
                return label == "null" ? null : label;
            case "kwargs":
                return label switch
                {
                    "null" => null,
                    "v9" => new Dictionary<string, object> { ["v"] = 9 },
                    _ => throw new ArgumentException("unknown kwargs variant " + label),
                };
            default:
                throw new ArgumentException($"no np.pad variants for {p.Name}");
        }
    }

    /// <summary>
    ///     Whether a parameter belongs to NumPy's index tricks with no array parameter of their own: np.ix_'s sequences and the
    ///     key of np.r_ / np.c_'s indexer (<see cref="np.AxisConcatenator"/>'s <c>this[params object[] key]</c>, a target
    ///     <c>np.r_</c> or <c>np.c_</c>), both built from the pool operand (<see cref="IxArgsLabels"/>,
    ///     <see cref="ConcatKeyLabels"/>).
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for np.ix_'s <c>args</c> and the concatenator's <c>key</c>.</returns>
    private static bool IsIndexTricks(ParameterInfo p) =>
        (p.Member.DeclaringType == typeof(np) && p.Member.Name == "ix_")
        || (p.Member.DeclaringType == typeof(np.AxisConcatenator) && p.Member.Name == "get_Item");

    /// <summary>
    ///     np.ix_'s sequences, each built from the pool operand: the operand itself (1-D works, any other rank is NumPy's
    ///     "Cross index must be 1 dimensional"), its flattened values twice, beside a bool mask of its size, a bool mask
    ///     alone, an int32 C# array (an ndarray of the house map), an EMPTY int32 C# array (an ndarray: no intp cast), an
    ///     EMPTY object[] (a Python list: float64 cast to intp), a float C# array, an object[] of C# ints (a list: int64), a
    ///     2-D C# array, a 0-d array, null (NumPy's None), and no sequence at all.
    /// </summary>
    private static readonly string[] IxArgsLabels =
    {
        "op", "flat,flat", "flat,mask", "mask", "i32[0,2]", "i32[]", "obj[]", "f64[0.5,1.5]", "obj[1,2]", "i32[,]", "0d",
        "null", "none",
    };

    /// <summary>
    ///     The key of np.r_ / np.c_, built from the pool operand (a slice is a string with a colon, a special directive a
    ///     string without one, as the indexer reads them): the operand alone (the base), twice, beside each kind of C#
    ///     scalar (a Python int, a negative one, one past int8, a float, a bool, a complex, an int past int64, the NumPy
    ///     scalar np.float16 a Half is, a strong 0-d array), beside an int8 array and a text array, after an int slice,
    ///     beside a float slice, an imaginary-step slice and an open slice, each directive form (an axis, a negative axis,
    ///     axis and ndmin, axis / ndmin / trans1d, the matrix letters, two refused ones), a directive alone, no entry at all,
    ///     and a non-colon string after the first entry (NumPy's "special directives must be the first entry.").
    /// </summary>
    private static readonly string[] ConcatKeyLabels =
    {
        "op", "op,op", "op,5", "op,-1", "op,1000", "op,2.5", "op,true", "op,c128", "op,u64max", "op,f16", "op,0d_i64",
        "op,i8[1,2]", "op,U[ab]", "0:3,op", "op,0.5:2:0.5", "op,0:1:3j", "op,2:", "d0,op,op", "d-1,op,op", "d0,2,op,op",
        "d1,2,0,op", "dr,op", "dc,op", "dq,op", "d0,q,op", "d0", "none", "op,x",
    };

    /// <summary>The labels of an index-tricks parameter (<see cref="IsIndexTricks"/>).</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>The labels <see cref="MakeIndexTricksArg"/> builds.</returns>
    private static string[] IndexTricksArgLabels(ParameterInfo p) =>
        p.Member.Name == "ix_" ? IxArgsLabels : ConcatKeyLabels;

    /// <summary>
    ///     Builds np.ix_'s sequences or the concatenator's key for an operand (see <see cref="IxArgsLabels"/>,
    ///     <see cref="ConcatKeyLabels"/>).
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <param name="label">The variant label.</param>
    /// <param name="operand">The pool operand.</param>
    /// <returns>The <c>object[]</c> the params parameter receives.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static object MakeIndexTricksArg(ParameterInfo p, string label, NDArray operand)
    {
        long size = operand.size;
        NDArray Mask() => np.array(Enumerable.Range(0, (int)size).Select(i => i % 2 == 0).ToArray());
        if (p.Member.Name == "ix_")
            return label switch
            {
                "op" => new object[] { operand },
                "flat,flat" => new object[] { operand.flatten(), operand.flatten() },
                "flat,mask" => new object[] { operand.flatten(), Mask() },
                "mask" => new object[] { Mask() },
                "i32[0,2]" => new object[] { new[] { 0, 2 } },
                "i32[]" => new object[] { Array.Empty<int>() },
                "obj[]" => new object[] { Array.Empty<object>() },
                "f64[0.5,1.5]" => new object[] { new[] { 0.5, 1.5 } },
                "obj[1,2]" => new object[] { new object[] { 1, 2 } },
                "i32[,]" => new object[] { new[,] { { 0, 1 }, { 1, 0 } } },
                "0d" => new object[] { np.array(5L) },
                "null" => new object[] { null },
                "none" => Array.Empty<object>(),
                _ => throw new ArgumentException("unknown ix_ variant " + label),
            };
        return label switch
        {
            "op" => new object[] { operand },
            "op,op" => new object[] { operand, operand },
            "op,5" => new object[] { operand, 5 },
            "op,-1" => new object[] { operand, -1 },
            "op,1000" => new object[] { operand, 1000 },
            "op,2.5" => new object[] { operand, 2.5 },
            "op,true" => new object[] { operand, true },
            "op,c128" => new object[] { operand, new Complex(1, 2) },
            "op,u64max" => new object[] { operand, ulong.MaxValue },
            "op,f16" => new object[] { operand, (Half)1.5 },
            "op,0d_i64" => new object[] { operand, NDArray.Scalar(1L) },
            "op,i8[1,2]" => new object[] { operand, np.array(new sbyte[] { 1, 2 }) },
            "op,U[ab]" => new object[] { operand, np.array(new[] { "ab" }) },
            "0:3,op" => new object[] { "0:3", operand },
            "op,0.5:2:0.5" => new object[] { operand, "0.5:2:0.5" },
            "op,0:1:3j" => new object[] { operand, "0:1:3j" },
            "op,2:" => new object[] { operand, "2:" },
            "d0,op,op" => new object[] { "0", operand, operand },
            "d-1,op,op" => new object[] { "-1", operand, operand },
            "d0,2,op,op" => new object[] { "0,2", operand, operand },
            "d1,2,0,op" => new object[] { "1,2,0", operand },
            "dr,op" => new object[] { "r", operand },
            "dc,op" => new object[] { "c", operand },
            "dq,op" => new object[] { "q", operand },
            "d0,q,op" => new object[] { "0,q", operand },
            "d0" => new object[] { "0" },
            "none" => Array.Empty<object>(),
            "op,x" => new object[] { operand, "x" },
            _ => throw new ArgumentException("unknown concatenator key variant " + label),
        };
    }

    /// <summary>
    ///     Whether an overload is the indexer of np.mgrid or np.ogrid (<see cref="np.MGridClass"/> /
    ///     <see cref="np.OGridClass"/>'s <c>this[params object[] key]</c>, NumPy's <c>nd_grid.__getitem__</c>): its key is
    ///     all it takes, so the key is the operand (<see cref="GridKeyLabels"/>).
    /// </summary>
    /// <param name="m">The overload.</param>
    /// <returns>True for the two indexers.</returns>
    private static bool IsGridIndexer(MethodInfo m) =>
        m.Name == "get_Item" && (m.DeclaringType == typeof(np.MGridClass) || m.DeclaringType == typeof(np.OGridClass));

    /// <summary>
    ///     The keys np.mgrid / np.ogrid run on, spelled as the indexer reads them (a slice is a string with a colon; several
    ///     entries are a Python tuple of slices). Single slices (NumPy's except branch: arange, or for an imaginary step
    ///     <c>arange(0, n) * step + start</c>): ints, a missing stop (the start is the stop), a step, a reversed range, a
    ///     negative step, float bounds, a float step whose arange delta is not the step, imaginary steps over an exact and an
    ///     inexact range, descending, one point and none, a zero step. Meshes (NumPy's <c>indices</c> / <c>arange</c> scaled
    ///     by <c>nn * step + start</c>): two int slices, one string holding both, a float bound, an imaginary axis, the
    ///     float-step and inexact imaginary axes, a float step that is exact, a reversed axis (the dense grid's negative
    ///     size, the open grid's empty axis), an empty axis, three axes, a slice without its stop, Slice objects (several and
    ///     one), the empty key, and entries that are no slice (a string without a colon, an int). Then the edges NumPy's
    ///     Python arithmetic decides: zero steps over Python ints and floats (<c>division by zero</c> / <c>float division by
    ///     zero</c>), an imaginary step without its stop (one point works, more need the stop), the error order of a zero
    ///     step, a missing stop and a negative size, a <c>-0.0</c> start, fractional and negative imaginary steps, float
    ///     steps whose <c>i * step + start</c> differs from arange, NaN / infinite / huge sizes, underflowed quotients, and
    ///     Python ints past 2**53 and past int64.
    /// </summary>
    private static readonly string[] GridKeyLabels =
    {
        "0:5", "5:", "2:8:2", "5:0", "0:5:-1", "0.5:3", "0.5:1.5:0.1", "0:1:5j", "0.1:0.7:7j", "-1:1:5j", "1:0:5j",
        "1:4:1j", "0:5:0j", "0:5:0",
        "0:3,0:4", "one:0:3,0:4", "0:2,0.0:2", "0:3,0:1:3j", "0.5:1.5:0.1,0:2", "0.1:0.7:7j,0:2", "0:2:0.5,0:3", "5:0,0:3",
        "0:3,0:0", "0:2,0:3,0:4", "0:3,2:", "slice(0,3),slice(1,5,2)", "slice(1,4)", "none", "x", "5",
        "0:3:0,0:2", "0.0:3:0,0:2", "0:3:0.0,0:2", "0:5:0.0", "3:3:0", "3.0:3:0", "::0",
        "5::1j", "5::3j", "5.5::3j", "0:2,5::1j", "0:2,5::3j", "0:3:0,0:", "0:,0:3:0", "0:-1,5::3j",
        "-0.0:5:1j", "-0.0:5", "-0.0:2,0:1.5", "0:1:2.5j", "0:1:-3j", "0:1:-3j,0:2", "0.3:0.9:7j", "0.3:0.9:7j,0:2",
        "1:2:0.1,0:1", "0.1:0.2:0.01,0:1", "0:10:3,0:2", "1:0.5:-0.2,0:2",
        "0:nan,0:2", "0:Infinity,0:2", "0:nan", "0:Infinity", "0:1:nanj,0:2", "0:1:Infinityj", "0:1e300", "0:1e300,0:2",
        "0:-1e300,0:2", "0:4e18,0:2", "0:1e-300:1e300", "0:1e-300:1e300,0:2", "0:1:Infinity,0:2", "0:1:Infinity",
        "9007199254740993:9007199254740995", "9007199254740993:9007199254740995,0:2",
        "9007199254740993:9007199254740995:1.0,0:2", "0:9223372036854775808:4611686018427387904,0:2",
        "9223372036854775808:9223372036854775810,0:2", "-9223372036854775808:-9223372036854775806,0:1",
    };

    /// <summary>The key of a <see cref="GridKeyLabels"/> label: the <c>object[]</c> the indexer's params parameter receives.</summary>
    /// <param name="label">The label: comma-separated slices (each its own string entry), <c>one:</c> for one string holding
    ///     them all, <c>slice(...)</c> entries for <see cref="Slice"/> objects, <c>none</c> for the empty key.</param>
    /// <returns>The key entries.</returns>
    private static object[] MakeGridKey(string label) => label switch
    {
        "none" => Array.Empty<object>(),
        "5" => new object[] { 5 },
        "slice(0,3),slice(1,5,2)" => new object[] { new Slice(0, 3), new Slice(1, 5, 2) },
        "slice(1,4)" => new object[] { new Slice(1, 4) },
        _ when label.StartsWith("one:", StringComparison.Ordinal) => new object[] { label.Substring("one:".Length) },
        _ => label.Split(',').Cast<object>().ToArray(),
    };

    /// <summary>
    ///     The coordinate tuples np.ravel_multi_index's scalar overload runs on (its operand, the stand-in dtype label
    ///     "coords"): inside the base dims (70000, 70000) - an inner point, the origin, the last element -, each bound
    ///     on each side (a coordinate at the dimension, a negative one, one past the negative dimension), too few and
    ///     too many coordinates, none, and the int64 extremes (wrap reduces them, clip saturates them). The first five
    ///     are the reduced pool (phase B), the first three the small one (phase C).
    /// </summary>
    private static readonly string[] RavelScalarCoordsLabels =
    {
        "[3,4]", "[0,0]", "[69999,69999]", "[70000,0]", "[-1,0]", "[0,-70001]", "[1]", "[1,2,3]", "[]",
        "[9223372036854775807,0]", "[-9223372036854775808,0]",
    };

    /// <summary>The coordinate tuple of a <see cref="RavelScalarCoordsLabels"/> label: the <c>long[]</c> the scalar
    ///     overload takes.</summary>
    /// <param name="label">The label: the coordinates in brackets, comma-separated, invariant integers.</param>
    /// <returns>A fresh array of the coordinates.</returns>
    private static long[] MakeRavelScalarCoords(string label)
    {
        string inner = label.Substring(1, label.Length - 2);
        return inner.Length == 0
            ? Array.Empty<long>()
            : inner.Split(',')
                .Select(v => long.Parse(v, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture))
                .ToArray();
    }

    /// <summary>The variants of np.ravel_multi_index's coordinate list (<see cref="MakeRavelCoords"/>); the first is
    ///     the base.</summary>
    private static readonly string[] RavelCoordsVariantLabels =
    {
        "pair", "int64", "fortran", "zero_d", "row", "mismatch", "bool", "float", "one", "triple", "none", "nullentry",
    };

    /// <summary>
    ///     Builds np.ravel_multi_index's coordinate list from a pool operand: the operand twice (the base: two
    ///     coordinates of its dtype and layout), the operand with a C-ordered and with an F-ordered int64 ramp of its
    ///     shape (the cast route and the result layout NumPy's iterator picks from mixed layouts), with a 0-d int64 and
    ///     with an int64 row of its last dimension (broadcasts), with an int64 vector one element longer than the
    ///     operand (no broadcast), with a bool and with a float64 array of its shape (the bool one is read as 0 / 1,
    ///     the float one is refused after the operand's check), the operand alone and three times (the count check), no
    ///     coordinate, and a null entry.
    /// </summary>
    /// <param name="label">The variant (<see cref="RavelCoordsVariantLabels"/>).</param>
    /// <param name="operand">The pool operand.</param>
    /// <returns>A fresh coordinate list (its first entry is the operand itself, but for "none").</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray[] MakeRavelCoords(string label, NDArray operand)
    {
        long[] dims = operand.Shape.Dimensions;
        long size = 1;
        foreach (long d in dims) size *= d;
        long last = dims.Length == 0 ? 1 : dims[^1];
        // The ramp values 0, 1, 2, 3, 0, ... are inside every dims variant but the zero and negative ones.
        NDArray Ramp(long n) => (np.arange(n) % 4L).astype(np.dtype("int64"));
        switch (label)
        {
            case "pair": return new[] { operand, operand };
            case "int64": return new[] { operand, Ramp(size).reshape(new Shape(dims)) };
            case "fortran": return new[] { operand, np.asfortranarray(Ramp(size).reshape(new Shape(dims))) };
            case "zero_d": return new[] { operand, np.array(2L) };
            case "row": return new[] { operand, Ramp(last) };
            case "mismatch": return new[] { operand, Ramp(size + 1) };
            case "bool": return new[] { operand, (Ramp(size) % 2L == 0L).reshape(new Shape(dims)) };
            case "float": return new[] { operand, Ramp(size).astype(np.dtype("float64")).reshape(new Shape(dims)) };
            case "one": return new[] { operand };
            case "triple": return new[] { operand, operand, operand };
            case "none": return Array.Empty<NDArray>();
            case "nullentry": return new[] { operand, null };
            default: throw new ArgumentException("unknown ravel_multi_index coordinate-list variant " + label);
        }
    }

    /// <summary>
    ///     Builds an np.select / np.piecewise argument for an operand (see the label sets above).
    /// </summary>
    /// <param name="p">The parameter (condlist, choicelist, default, funclist or args).</param>
    /// <param name="label">The variant.</param>
    /// <param name="operand">The pool operand: the conditions take its shape, the choices and the defaults its values.</param>
    /// <returns>The argument: an array list, an object list, a boxed value, an array, or null.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static object MakeSelectArg(ParameterInfo p, string label, NDArray operand)
    {
        long[] dims = operand.Shape.Dimensions;
        long size = 1;
        foreach (long d in dims) size *= d;
        long last = dims.Length == 0 ? 1 : dims[^1];
        NDArray Mask(bool even, long count, long[] shape) =>
            np.array(Enumerable.Range(0, (int)count).Select(i => (i % 2 == 0) == even).ToArray()).reshape(new Shape(shape));
        NDArray Alt() => Mask(true, size, dims);
        NDArray NotAlt() => Mask(false, size, dims);
        NDArray Row(double start) => np.array(Enumerable.Range(0, (int)last).Select(i => start + i).ToArray());
        string n = p.Name;
        if (n == "condlist" && p.ParameterType == typeof(NDArray[]))
            return label switch
            {
                "alt2" => new[] { Alt(), NotAlt() },
                "first" or "first2" => new[] { np.ones(new Shape(dims), np.@bool), np.ones(new Shape(dims), np.@bool) },
                "none" or "none2" => new[] { np.zeros(new Shape(dims), np.@bool), np.zeros(new Shape(dims), np.@bool) },
                "int" => new[] { Alt().astype(np.int64), NotAlt() },
                "int2" => new[] { Alt().astype(np.int64), NotAlt().astype(np.int64) },
                "row" => new[] { Mask(true, last, new[] { last }), Mask(false, last, new[] { last }) },
                "0d" or "0d2" => new[] { np.array(true), np.array(false) },
                "badshape" => new[] { Mask(true, 7, new long[] { 7 }), Mask(false, 7, new long[] { 7 }) },
                "one" => new[] { Alt() },
                "three" => new[] { Alt(), NotAlt(), Alt() },
                "empty" => Array.Empty<NDArray>(),
                "null" => null,
                _ => throw new ArgumentException("unknown condlist variant " + label),
            };
        if (n == "condlist")
            return label switch
            {
                "stack2" => np.stack(new[] { Alt(), NotAlt() }),
                "alt" => Alt(),
                "0d" => np.array(true),
                "empty0" => np.zeros(new Shape(new long[] { 0 }.Concat(dims).ToArray()), np.@bool),
                "null" => null,
                _ => throw new ArgumentException("unknown bare condlist variant " + label),
            };
        if (n == "choicelist")
            return label switch
            {
                "pair" => new object[] { operand.copy(), np.zeros_like(operand) },
                "scalars" => new object[] { 7, 2.5 },
                "big" => new object[] { operand.copy(), 1000 },
                "strong0d" => new object[] { operand.copy(), np.array(2.5) },
                "text" => new object[] { "a", "bb" },
                "half" => new object[] { (Half)1.5, 7 },
                "row" => new object[] { Row(1.0), Row(-1.0) },
                "badshape" => new object[] { np.zeros(new Shape(7)), np.zeros(new Shape(7)) },
                "nullentry" => new object[] { operand.copy(), null },
                "one" => new object[] { operand.copy() },
                "null" => null,
                _ => throw new ArgumentException("unknown choicelist variant " + label),
            };
        if (n == "default")
            return label switch
            {
                "null" => null,
                "0" => 0,
                "1000" => 1000,
                "2.5" => 2.5,
                "c128" => new Complex(1, 2),
                "true" => true,
                "x" => "x",
                "0d_int64" => np.array(5L),
                "same" => operand.copy(),
                "row" => Row(0.5),
                "badshape" => np.zeros(new Shape(7)),
                _ => throw new ArgumentException("unknown default variant " + label),
            };
        if (n == "funclist")
            return label switch
            {
                "neg_sq" => new object[] { PiecewiseNegate, PiecewiseSquare },
                "consts" => new object[] { 1, 2.5 },
                "default3" => new object[] { 1, 2, 3 },
                "big" => new object[] { 1000, 0 },
                "c128" => new object[] { new Complex(1, 2), 0 },
                "arrays" => new object[] { np.array(5L), np.array(2.5) },
                "args" => new object[] { PiecewiseAddArg, 0 },
                "one" => new object[] { 1 },
                "nullentry" => new object[] { null, 0 },
                "null" => null,
                _ => throw new ArgumentException("unknown funclist variant " + label),
            };
        if (n == "args")
            return label switch
            {
                "none" => Array.Empty<object>(),
                "one" => new object[] { 1 },
                "two" => new object[] { 1, 2 },
                _ => throw new ArgumentException("unknown args variant " + label),
            };
        throw new ArgumentException($"no select / piecewise variants for {n}");
    }

    /// <summary>Builds np.fill_diagonal's val (see <see cref="FillDiagValueLabels"/>).</summary>
    /// <param name="label">The variant.</param>
    /// <param name="operand">The pool operand ("same" copies it).</param>
    /// <returns>The boxed C# value, an array, or null (Python's None).</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static object MakeFillDiagValue(string label, NDArray operand) => label switch
    {
        "7" => 7,
        "2.5" => 2.5,
        "true" => true,
        "1+2j" => new Complex(1, 2),
        "x" => "x",
        "null" => null,
        "300" => 300,
        "2**70" => BigInteger.Pow(2, 70),
        "half1.5" => (Half)1.5,
        "f64{1,2}" => new[] { 1.0, 2.0 },
        "i32{}" => Array.Empty<int>(),
        "same" => operand.copy(),
        "0d_5" => np.array(5L),
        "obj{300,2}" => new object[] { 300, 2 },
        "obj{x}" => new object[] { "x" },
        "obj{1.5,nan}" => new object[] { 1.5, double.NaN },
        "obj{null}" => new object[] { null },
        "obj{}" => Array.Empty<object>(),
        _ => throw new ArgumentException("unknown fill_diagonal value variant " + label),
    };

    /// <summary>np.take / np.put's scalar index: inside every pool axis first, the first, the last, and one past either
    ///     end.</summary>
    private static readonly string[] ItemIndexLabels = { "1", "0", "-1", "99", "-99" };

    /// <summary>The clip mode of np.take / np.put / np.choose: the default first, the other two names, what NumPy's
    ///     converter reads from the first letter (a lower and a capital letter, a prefix), what it refuses, and null.</summary>
    private static readonly string[] ItemModeLabels = { "raise", "wrap", "clip", "c", "W", "rais", "x", "empty", "null" };

    /// <summary>np.put's scalar value (the object overload): a Python int, a float, a bool, None, a str, an int past a
    ///     narrow dtype, a C# array (a sequence) and a 0-d array.</summary>
    private static readonly string[] ItemPutValueLabels = { "7", "2.5", "true", "null", "x", "300", "f64{1,2}", "0d_5" };

    /// <summary>np.putmask / np.place's masks, built from the operand: alternating True / False of its shape, none, all,
    ///     an integer and a float mask (their truth), one element too few and too many, a (1, size) reshape, a text mask, a
    ///     strided view, a broadcast True, and null.</summary>
    private static readonly string[] ItemMaskLabels =
        { "alt", "none", "all", "int", "float", "short", "long", "2d", "text", "strided", "bcast", "null" };

    /// <summary>np.put / np.putmask / np.place's value arrays: three int64 values (cycled), one float, none (np.place's
    ///     refusal, np.putmask's no-op), text, a 2-D array, the operand's own copy, and null.</summary>
    private static readonly string[] ItemValuesLabels = { "i64[7,8,9]", "f64[2.5]", "i64[]", "U1[x]", "2d_i64", "same", "null" };

    /// <summary>np.putmask / np.place's C# array values: doubles, one int, strings, none, mixed objects, and null.</summary>
    private static readonly string[] ItemArrayValuesLabels = { "f64{1.5,2.5}", "i32{7}", "str{x}", "i32{}", "obj{1,a}", "null" };

    /// <summary>np.repeat's scalar count: two first, none, one, three, and the negative count NumPy refuses.</summary>
    private static readonly string[] ItemRepeatCountLabels = { "2", "0", "1", "3", "-1" };

    /// <summary>np.repeat's count arrays, built from the operand: one count (broadcast), one per element (0, 1, 2, ...;
    ///     the flat call's length), a negative count, two counts (a length NumPy refuses for most operands), the dtypes it
    ///     casts or refuses (float, uint8, text), a 0-d count, a 2-D count, and null.</summary>
    private static readonly string[] ItemRepeatArrayLabels =
        { "i64[2]", "each", "i64[-1]", "i64[1,2]", "f64[2.0]", "u8[2]", "0d_2", "2d", "U1[2]", "null" };

    /// <summary>np.choose's choice lists, built from the operand: three 0-d int64 choices, two copies of the operand,
    ///     a float and an int choice, two text choices, one choice, none, two that do not broadcast, and null.</summary>
    private static readonly string[] ItemChoiceListLabels = { "0d3", "same2", "mixed", "text", "one", "empty", "bad", "null" };

    /// <summary>np.choose's object choices: three Python ints, a float beside a str, the operand's copy beside a Python
    ///     int, none, and null.</summary>
    private static readonly string[] ItemChoiceObjectLabels = { "ints", "mixed", "arrays", "empty", "null" };

    /// <summary>np.choose's choices as one array whose first axis enumerates them: three values, two stacked copies of
    ///     the operand, a 0-d array, and null.</summary>
    private static readonly string[] ItemChoiceStackLabels = { "stack3", "stack_op", "0d", "null" };

    /// <summary>np.compress's conditions, built from the operand: alternating True / False over its size, a short one
    ///     (NumPy reads the first elements), one element longer with a True past the end, none true, an integer and a
    ///     float condition, an empty one, a 2-D one, text, and null.</summary>
    private static readonly string[] ItemConditionLabels =
        { "alt", "short", "long", "none", "int", "float", "empty", "2d", "text", "null" };

    /// <summary>np.resize / ndarray.resize's new shapes: the operand's own first, then smaller, reshaped, empty, 0-d and
    ///     larger shapes, and a negative dimension.</summary>
    private static readonly string[] ItemNewShapeLabels = { "same", "6", "2x3", "0", "scalar", "3x4", "10", "2x-1" };

    /// <summary>
    ///     Builds an item-selection argument (<see cref="Role.ItemArg"/>) for an operand.
    /// </summary>
    /// <param name="p">The parameter, whose name and type select the label set (<see cref="ItemArgLabels"/>).</param>
    /// <param name="label">The variant.</param>
    /// <param name="operand">The pool operand (the masks, conditions, per-element counts, the operand-copy choices and
    ///     values, and the "same" shape are built from it).</param>
    /// <returns>The argument: an array, a C# number, string, array or shape, or null.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static object MakeItemArg(ParameterInfo p, string label, NDArray operand)
    {
        Type t = p.ParameterType;
        string n = p.Name;
        long[] dims = operand.Shape.Dimensions;
        long size = 1;
        foreach (long d in dims) size *= d;
        NDArray Alternating(long count) => np.array(Enumerable.Range(0, (int)count).Select(i => i % 2 == 0).ToArray());
        NDArray Longs(params long[] v) => np.array(v);
        // The along-axis pair and np.fill_diagonal build their own (their names are the family's own).
        if (IsAlongAxis(p))
            return MakeAlongArg(p, label, operand);
        if (IsFillDiagonal(p))
            return MakeFillDiagValue(label, operand);
        if (IsSelect(p) || IsPiecewise(p))
            return MakeSelectArg(p, label, operand);
        if (IsPad(p))
            return MakePadArg(p, label, operand);
        if (IsIndexTricks(p))
            return MakeIndexTricksArg(p, label, operand);
        if (IsApplyAlongAxis(p))
            return MakeApplyArg(p, label);
        if (IsIndexer(p))
            return MakeIndexerArg(p, label, operand);
        if (n == "indices")
            return label switch
            {
                "i64[1]" => Longs(1),
                "i64[0,2]" => Longs(0, 2),
                "i32[2,0]" => np.array(new[] { 2, 0 }),
                "i64[-1]" => Longs(-1),
                "i64[99]" => Longs(99),
                "i64[-99]" => Longs(-99),
                "0d_i64_1" => np.array(1L),
                "i64[]" => np.array(Array.Empty<long>()),
                "2d_i64" => Longs(0, 1, 1, 0).reshape(new Shape(2, 2)),
                "u64[1]" => np.array(new ulong[] { 1 }),
                "u8[1]" => np.array(new byte[] { 1 }),
                "f64[1.0]" => np.array(new[] { 1.0 }),
                "bool[T,F]" => np.array(new[] { true, false }),
                "i64_strided" => Longs(0, 9, 1, 9, 2)["::2"],
                "U1[1]" => np.array(new[] { "1" }),
                "null" => null,
                _ => throw new ArgumentException("unknown indices variant " + label),
            };
        if (n == "index")
            return long.Parse(label, CultureInfo.InvariantCulture);
        if (n == "mode")
            return label switch { "empty" => "", "null" => null, _ => label };
        if (n == "value")
            return label switch
            {
                "7" => 7,
                "2.5" => 2.5,
                "true" => true,
                "null" => null,
                "x" => "x",
                "300" => 300,
                "f64{1,2}" => new[] { 1.0, 2.0 },
                "0d_5" => np.array(5L),
                _ => throw new ArgumentException("unknown value variant " + label),
            };
        if (n == "mask")
            return label switch
            {
                "alt" => Alternating(size).reshape(new Shape(dims)),
                "none" => np.zeros(new Shape(dims), np.dtype("bool")),
                "all" => np.ones(new Shape(dims), np.dtype("bool")),
                "int" => np.array(Enumerable.Range(0, (int)size).Select(i => (long)(i % 3)).ToArray()).reshape(new Shape(dims)),
                "float" => np.array(Enumerable.Range(0, (int)size).Select(i => (i % 2) * 0.5).ToArray()).reshape(new Shape(dims)),
                "short" => Alternating(Math.Max(size - 1, 0)),
                "long" => Alternating(size + 1),
                "2d" => Alternating(size).reshape(new Shape(1, size)),
                "text" => np.array(Enumerable.Range(0, (int)size).Select(i => i % 2 == 0 ? "a" : "").ToArray()).reshape(new Shape(dims)),
                "strided" => Alternating(2 * size)["::2"],
                "bcast" => np.broadcast_to(np.array(true), new Shape(dims)),
                "null" => null,
                _ => throw new ArgumentException("unknown mask variant " + label),
            };
        if (n is "values" or "vals" && t == typeof(NDArray))
            return label switch
            {
                "i64[7,8,9]" => Longs(7, 8, 9),
                "f64[2.5]" => np.array(new[] { 2.5 }),
                "i64[]" => np.array(Array.Empty<long>()),
                "U1[x]" => np.array(new[] { "x" }),
                "2d_i64" => Longs(7, 8, 9, 10).reshape(new Shape(2, 2)),
                "same" => operand.copy(),
                "null" => null,
                _ => throw new ArgumentException("unknown values variant " + label),
            };
        if (n is "values" or "vals")
            return label switch
            {
                "f64{1.5,2.5}" => new[] { 1.5, 2.5 },
                "i32{7}" => new[] { 7 },
                "str{x}" => new[] { "x" },
                "i32{}" => Array.Empty<int>(),
                "obj{1,a}" => new object[] { 1, "a" },
                "null" => null,
                _ => throw new ArgumentException("unknown array values variant " + label),
            };
        if (n == "repeats" && t == typeof(int))
            return int.Parse(label, CultureInfo.InvariantCulture);
        if (n == "repeats" && t == typeof(long))
            return long.Parse(label, CultureInfo.InvariantCulture);
        if (n == "repeats")
            return label switch
            {
                "i64[2]" => Longs(2),
                "each" => np.array(Enumerable.Range(0, (int)size).Select(i => (long)(i % 3)).ToArray()),
                "i64[-1]" => Longs(-1),
                "i64[1,2]" => Longs(1, 2),
                "f64[2.0]" => np.array(new[] { 2.0 }),
                "u8[2]" => np.array(new byte[] { 2 }),
                "0d_2" => np.array(2L),
                "2d" => Longs(2).reshape(new Shape(1, 1)),
                "U1[2]" => np.array(new[] { "2" }),
                "null" => null,
                _ => throw new ArgumentException("unknown repeats variant " + label),
            };
        if (n == "choices" && t == typeof(NDArray[]))
            return label switch
            {
                "0d3" => new[] { np.array(10L), np.array(20L), np.array(30L) },
                "same2" => new[] { operand.copy(), operand.copy() },
                "mixed" => new[] { np.array(1.5), np.array(2) },
                "text" => new[] { np.array("a"), np.array("bb") },
                "one" => new[] { np.array(7L) },
                "empty" => Array.Empty<NDArray>(),
                "bad" => new[] { np.zeros(new Shape(7)), np.zeros(new Shape(7)) },
                "null" => null,
                _ => throw new ArgumentException("unknown choice list variant " + label),
            };
        if (n == "choices" && t == typeof(object[]))
            return label switch
            {
                "ints" => new object[] { 10, 20, 30 },
                "mixed" => new object[] { 1.5, "x" },
                "arrays" => new object[] { operand.copy(), 5 },
                "empty" => Array.Empty<object>(),
                "null" => null,
                _ => throw new ArgumentException("unknown choice object variant " + label),
            };
        if (n == "choices")
            return label switch
            {
                "stack3" => Longs(10, 20, 30),
                "stack_op" => np.stack(new[] { operand.copy(), operand.copy() }),
                "0d" => np.array(5L),
                "null" => null,
                _ => throw new ArgumentException("unknown choice stack variant " + label),
            };
        if (n == "condition")
            return label switch
            {
                "alt" => Alternating(size),
                "short" => np.array(new[] { true, false }),
                "long" => np.array(Enumerable.Range(0, (int)size + 1).Select(i => i == size || i % 2 == 0).ToArray()),
                "none" => np.zeros(new Shape(size), np.dtype("bool")),
                "int" => np.array(Enumerable.Range(0, (int)size).Select(i => (long)(i % 3)).ToArray()),
                "float" => np.array(Enumerable.Range(0, (int)size).Select(i => (i % 2) * 0.5).ToArray()),
                "empty" => np.array(Array.Empty<bool>()),
                "2d" => Alternating(size).reshape(new Shape(1, size)),
                "text" => np.array(Enumerable.Range(0, (int)size).Select(i => i % 2 == 0 ? "a" : "").ToArray()),
                "null" => null,
                _ => throw new ArgumentException("unknown condition variant " + label),
            };
        if (n == "new_shape")
        {
            long[] shape = label switch
            {
                "same" => dims,
                "6" => new long[] { 6 },
                "2x3" => new long[] { 2, 3 },
                "0" => new long[] { 0 },
                "scalar" => Array.Empty<long>(),
                "3x4" => new long[] { 3, 4 },
                "10" => new long[] { 10 },
                "2x-1" => new long[] { 2, -1 },
                _ => throw new ArgumentException("unknown new_shape variant " + label),
            };
            // A Shape cannot hold a negative dimension; the C# array form passes it as given (ndarray.resize's refusal).
            return t == typeof(long[]) ? shape : new Shape(shape);
        }
        throw new ArgumentException($"no item-selection variants for {n} ({t.Name})");
    }

    /// <summary>
    ///     Whether a parameter belongs to the assignment family - np.copyto (NumPy's <c>array_copyto</c> over
    ///     <c>array_assign_array.c</c> / <c>array_assign_scalar.c</c>) and ndarray.fill (<c>PyArray_FillWithScalar</c>) -
    ///     whose <c>src</c> / <c>value</c> roles are theirs alone (see <see cref="AssignArgLabels"/>).
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.copyto overload (closed generic ones included) or of the public
    ///     instance NDArray.fill.</returns>
    private static bool IsAssignment(ParameterInfo p) =>
        (p.Member.DeclaringType == typeof(np) && p.Member.Name == "copyto")
        || (p.Member.DeclaringType == typeof(NDArray) && p.Member.Name == "fill");

    /// <summary>
    ///     The variant labels of an assignment parameter (<see cref="Role.AssignArg"/>), the base first, or null for a
    ///     parameter the family's role does not read (np.copyto's <c>dst</c> operand, <c>casting</c> and <c>where</c> take
    ///     the general roles).
    /// </summary>
    /// <param name="p">The parameter (of an <see cref="IsAssignment"/> overload).</param>
    /// <returns>The labels <see cref="MakeAssignArg"/> builds, or null.</returns>
    private static string[] AssignArgLabels(ParameterInfo p)
    {
        Type t = p.ParameterType;
        string n = p.Name;
        if (n == "value" && t == typeof(object)) return FillValueLabels;
        if (n != "src") return null;
        if (t == typeof(NDArray)) return CopytoSrcLabels;
        if (t == typeof(BigInteger)) return CopytoBigLabels;
        if (t == typeof(string)) return CopytoStringLabels;
        return CopytoScalarLabels.TryGetValue(t, out var labels) ? labels : null;
    }

    /// <summary>np.copyto's array sources, built from the destination: its own copy, arrays of its shape in other dtypes
    ///     (int32, float64, complex128, bool, U5, S5, StringDType), 0-d sources (NumPy's <c>PyArray_AssignRawScalar</c>:
    ///     int64, float64, U1, StringDType), a broadcast row, a copy with a leading length-1 axis (stripped by NumPy's
    ///     backward-compatibility rule), a shape that broadcasts against no pool operand, the transposed copy, a reversed
    ///     view of the destination itself (overlap), the destination itself (the redundant self-copy), and None.</summary>
    private static readonly string[] CopytoSrcLabels =
    {
        "same", "int32", "float64", "complex128", "bool", "U5", "S5", "T", "0d", "0d_f64", "0d_U", "0d_T", "row", "lead1",
        "badshape", "transposed", "overlap_rev", "self", "null",
    };

    /// <summary>np.copyto's Python ints of any size: small ones, the edges of int32 / int64 / uint64 and past them.</summary>
    private static readonly string[] CopytoBigLabels =
        { "5", "-1", "300", "2^31", "2^63-1", "2^63", "2^64-1", "2^64", "-2^63", "-2^63-1", "2^100", "-2^100" };

    /// <summary>np.copyto's str sources: a number's text, a word, the empty string, a bool's text, a negative int's, NaN's,
    ///     a non-ASCII character, and null.</summary>
    private static readonly string[] CopytoStringLabels =
        { "1.5", "abc", "", "True", "-7", "nan", "ü", "null" };

    /// <summary>The C# numbers of np.copyto's generic overload, per type: the base first, then the edges NumPy's weak-scalar
    ///     rule range-checks or rounds (an int past a narrow destination, the 64-bit limits, NaN, infinities, -0.0).</summary>
    private static readonly Dictionary<Type, string[]> CopytoScalarLabels = new()
    {
        [typeof(sbyte)] = new[] { "5", "-128" },
        [typeof(byte)] = new[] { "5", "255" },
        [typeof(short)] = new[] { "5", "-300" },
        [typeof(ushort)] = new[] { "5", "65535" },
        [typeof(int)] = new[] { "5", "-1", "300", "2147483647" },
        [typeof(uint)] = new[] { "5", "4294967295" },
        [typeof(long)] = new[] { "5", "-1", "300", "9223372036854775807", "-9223372036854775808" },
        [typeof(ulong)] = new[] { "5", "9223372036854775808", "18446744073709551615" },
        [typeof(float)] = new[] { "1.5", "NaN", "-0.0", "3.4e38" },
        [typeof(double)] = new[] { "1.5", "-2.5", "NaN", "inf", "-0.0", "1e300", "0.1" },
        [typeof(Complex)] = new[] { "1+2j", "NaN+0j", "0+0j" },
        [typeof(bool)] = new[] { "true", "false" },
        [typeof(Half)] = new[] { "1.5", "NaN", "65504" },
        [typeof(char)] = new[] { "a", "7", "ü" },
        [typeof(decimal)] = new[] { "1.5", "-7", "79228162514264337593543950335" },
    };

    /// <summary>ndarray.fill's values: Python ints (small, negative, past a narrow dtype), floats (a fraction, NaN), a bool,
    ///     a complex, NumPy scalars (float16, a decimal, a char), strs (a word, the empty string), None, a Python int past
    ///     uint64, 0-d arrays (int64, U1), a one-element and a two-element array (refused), a C# long, float and ulong.</summary>
    private static readonly string[] FillValueLabels =
    {
        "7", "-1", "300", "1.5", "NaN", "true", "1+2j", "h1.5", "m1.5", "c'a'", "abc", "empty", "null", "2^70", "0d_i64",
        "0d_U", "1elem", "2elem", "l5", "f1.5", "u2^64-1",
    };

    /// <summary>
    ///     Builds an assignment argument (<see cref="Role.AssignArg"/>) for a destination.
    /// </summary>
    /// <param name="p">The parameter, whose name and type select the label set (<see cref="AssignArgLabels"/>).</param>
    /// <param name="label">The variant.</param>
    /// <param name="operand">The pool operand, np.copyto's destination or the array filled (the array sources of its
    ///     shape and the views of it are built from it).</param>
    /// <returns>The argument: an array, a C# number of the parameter's type, a <see cref="BigInteger"/>, a string, or
    ///     null.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static object MakeAssignArg(ParameterInfo p, string label, NDArray operand)
    {
        Type t = p.ParameterType;
        long[] dims = operand.Shape.Dimensions;
        long size = 1;
        foreach (long d in dims) size *= d;
        var shape = new Shape(dims);
        NDArray Ramp(string dt) =>
            np.array(Enumerable.Range(0, (int)size).Select(i => (double)(i % 4)).ToArray()).astype(np.dtype(dt)).reshape(shape);
        NDArray Text(string dt) => size == 0
            ? np.empty(shape, np.dtype(dt))
            : np.array(Enumerable.Range(0, (int)size).Select(i => (i % 3).ToString(CultureInfo.InvariantCulture)).ToArray())
                .astype(np.dtype(dt)).reshape(shape);
        if (t == typeof(NDArray))
            return label switch
            {
                "same" => operand.copy(),
                "int32" or "float64" or "complex128" or "bool" => Ramp(label),
                "U5" or "S5" or "T" => Text(label),
                "0d" => np.array(2L),
                "0d_f64" => np.array(2.5),
                "0d_U" => np.array("7"),
                "0d_T" => np.array("7").astype(np.dtype("T")),
                "row" => np.array(Enumerable.Range(0, (int)(dims.Length == 0 ? 1 : dims[^1])).Select(i => (long)(i % 4)).ToArray()),
                "lead1" => operand.copy().reshape(new Shape(new long[] { 1 }.Concat(dims).ToArray())),
                "badshape" => np.array(Enumerable.Range(0, 7).Select(i => (long)i).ToArray()),
                "transposed" => operand.T.copy(),
                // a view of the destination itself, its first axis reversed: the overlap rule (a 0-d destination: itself)
                "overlap_rev" => dims.Length == 0 ? operand : operand["::-1"],
                "self" => operand,
                "null" => null,
                _ => throw new ArgumentException("unknown copyto source variant " + label),
            };
        if (t == typeof(BigInteger))
            return label switch
            {
                "2^31" => BigInteger.Pow(2, 31),
                "2^63-1" => BigInteger.Pow(2, 63) - 1,
                "2^63" => BigInteger.Pow(2, 63),
                "2^64-1" => BigInteger.Pow(2, 64) - 1,
                "2^64" => BigInteger.Pow(2, 64),
                "-2^63" => -BigInteger.Pow(2, 63),
                "-2^63-1" => -BigInteger.Pow(2, 63) - 1,
                "2^100" => BigInteger.Pow(2, 100),
                "-2^100" => -BigInteger.Pow(2, 100),
                _ => BigInteger.Parse(label, CultureInfo.InvariantCulture),
            };
        if (t == typeof(string))
            return label == "null" ? null : label;
        if (p.Name == "value")
            return label switch
            {
                "7" => 7,
                "-1" => -1,
                "300" => 300,
                "1.5" => 1.5,
                "NaN" => double.NaN,
                "true" => true,
                "1+2j" => new Complex(1, 2),
                "h1.5" => (Half)1.5,
                "m1.5" => 1.5m,
                "c'a'" => 'a',
                "abc" => "abc",
                "empty" => "",
                "null" => null,
                "2^70" => BigInteger.Pow(2, 70),
                "0d_i64" => np.array(7L),
                "0d_U" => np.array("7"),
                "1elem" => np.array(new[] { 5L }),
                "2elem" => np.array(new[] { 1L, 2L }),
                "l5" => 5L,
                "f1.5" => 1.5f,
                "u2^64-1" => ulong.MaxValue,
                _ => throw new ArgumentException("unknown fill value variant " + label),
            };
        // the generic copyto's C# number of the instantiated type
        var inv = CultureInfo.InvariantCulture;
        if (t == typeof(sbyte)) return sbyte.Parse(label, inv);
        if (t == typeof(byte)) return byte.Parse(label, inv);
        if (t == typeof(short)) return short.Parse(label, inv);
        if (t == typeof(ushort)) return ushort.Parse(label, inv);
        if (t == typeof(int)) return int.Parse(label, inv);
        if (t == typeof(uint)) return uint.Parse(label, inv);
        if (t == typeof(long)) return long.Parse(label, inv);
        if (t == typeof(ulong)) return ulong.Parse(label, inv);
        if (t == typeof(float)) return label == "NaN" ? float.NaN : float.Parse(label, inv);
        if (t == typeof(double))
            return label switch { "NaN" => double.NaN, "inf" => double.PositiveInfinity, _ => double.Parse(label, inv) };
        if (t == typeof(Complex))
            return label switch { "1+2j" => new Complex(1, 2), "NaN+0j" => new Complex(double.NaN, 0), _ => Complex.Zero };
        if (t == typeof(bool)) return label == "true";
        if (t == typeof(Half)) return label == "NaN" ? Half.NaN : (Half)double.Parse(label, inv);
        if (t == typeof(char)) return label[0];
        if (t == typeof(decimal)) return decimal.Parse(label, inv);
        throw new ArgumentException($"unknown assignment parameter {t.Name} {p.Name}");
    }

    /// <summary>
    ///     The closed instances of a generic overload the snapshot calls: np.copyto&lt;T&gt; for every type of
    ///     <see cref="CopytoScalarLabels"/> (NumPy's Python scalars and the NumPy scalars NumSharp's C# types stand for);
    ///     the coercion family's generic overloads (np.array&lt;T&gt;, np.asarray&lt;T&gt;, NDArray.Scalar&lt;T&gt;) for
    ///     every element type NumSharp stores (<see cref="HostElementTypes"/>), each fed only the pool operands of its own
    ///     dtype; the constructor family's (np.empty&lt;T&gt;, np.zeros&lt;T&gt;, np.ones&lt;T&gt;, np.full&lt;T&gt;) for the
    ///     same element types, each over every shape; np.frombuffer&lt;TSource&gt;'s for the same element types, each fed
    ///     only the pool operands of its own dtype (its buffer is their values as a <c>TSource[]</c>); none for any other
    ///     generic method, whose targets keep their snapshots.
    /// </summary>
    /// <param name="m">A generic method definition.</param>
    /// <returns>Its closed instances, or none.</returns>
    private static IEnumerable<MethodInfo> SnapshotInstantiations(MethodInfo m) =>
        m.DeclaringType == typeof(np) && m.Name == "copyto"
            ? CopytoScalarLabels.Keys.Select(t => m.MakeGenericMethod(t))
            : IsCoercion(m) || IsCtor(m) || IsFromBuffer(m)
                ? HostElementTypes.Select(t => m.MakeGenericMethod(t))
                : Enumerable.Empty<MethodInfo>();

    /// <summary>
    ///     Whether an overload is a spacing generator - np.linspace, np.logspace or np.geomspace (NumPy's
    ///     <c>_core/function_base.py</c>) - whose operand is a (start, stop) pair rather than an array.
    /// </summary>
    /// <param name="m">The overload.</param>
    /// <returns>True for a public static overload of one of those names on np.</returns>
    private static bool IsSpacing(MethodInfo m) =>
        m.DeclaringType == typeof(np) && m.Name is "linspace" or "logspace" or "geomspace";

    /// <summary>Whether a parameter belongs to a spacing generator (<see cref="IsSpacing(MethodInfo)"/>).</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True when its method is one of the spacing generators.</returns>
    private static bool IsSpacing(ParameterInfo p) => p.Member is MethodInfo m && IsSpacing(m);

    /// <summary>
    ///     The (start, stop) pairs of the real spacing overloads: increasing, decreasing and equal pairs, zero at either end
    ///     (geomspace's "Geometric sequence cannot include zero"), all-negative and mixed-sign pairs (geomspace's rotation
    ///     and its NaN interior), a NaN and an infinite end, signed zeros, the extremes of float64, fractions.
    /// </summary>
    private static readonly string[] RealSpacePairs =
    {
        "0,1", "1,10", "-5,5", "2,2", "10,0", "1,1000", "-1,-1000", "-1,1", "0.5,-0.5", "nan,1", "1,inf", "-0,0",
        "1e-300,1e300", "3.5,-2.25", "100,1",
    };

    /// <summary>
    ///     The (start, stop) pairs of geomspace's Complex overload: real, purely imaginary and general complex ends, an
    ///     all-negative and a mixed-sign real pair, a zero end, a NaN end, an equal pair.
    /// </summary>
    private static readonly string[] ComplexSpacePairs =
        { "1,8", "1j,8j", "1+1j,1000+1000j", "-1,-8", "1,-1", "0,1", "nan,1", "1,1", "2-3j,-4+5j" };

    /// <summary>The pair labels for a spacing overload's start parameter type.</summary>
    /// <param name="t">The type of the start parameter (double, float or Complex).</param>
    /// <returns>The pair labels, phase A's pool; phase B takes the first five, phase C the first three.</returns>
    private static string[] SpacePairLabels(Type t) => t == typeof(Complex) ? ComplexSpacePairs : RealSpacePairs;

    /// <summary>
    ///     One end of a spacing pair as the parameter's type: the label's text before the comma (start) or after it
    ///     (stop), parsed as a real (<c>nan</c>, <c>inf</c>, <c>-0</c>, a number) or, for a Complex parameter, as
    ///     <c>a</c>, <c>bj</c> or <c>a+bj</c> / <c>a-bj</c>.
    /// </summary>
    /// <param name="t">The parameter type: double, float or Complex.</param>
    /// <param name="label">The pair label (<c>"start,stop"</c>).</param>
    /// <param name="start">True for the start value, false for the stop value.</param>
    /// <returns>The boxed value of type <paramref name="t"/>.</returns>
    /// <exception cref="ArgumentException">A parameter type the generators do not take (a bug in the tool).</exception>
    private static object MakeSpaceValue(Type t, string label, bool start)
    {
        string text = label.Split(',')[start ? 0 : 1];
        static double Real(string s) => s switch
        {
            "nan" => double.NaN,
            "inf" => double.PositiveInfinity,
            "-inf" => double.NegativeInfinity,
            "-0" => -0.0,
            _ => double.Parse(s, CultureInfo.InvariantCulture),
        };
        if (t == typeof(double))
            return Real(text);
        if (t == typeof(float))
            return (float)Real(text);
        if (t != typeof(Complex))
            throw new ArgumentException("no spacing value of type " + t.Name);
        if (!text.EndsWith("j", StringComparison.Ordinal))
            return new Complex(Real(text), 0.0);
        // a+bj / a-bj: the sign that starts the imaginary part is the last + or - after the first character
        string body = text.Substring(0, text.Length - 1);
        int split = Math.Max(body.LastIndexOf('+'), body.LastIndexOf('-'));
        if (split <= 0)
            return new Complex(0.0, Real(body));
        return new Complex(Real(body.Substring(0, split)), Real(body.Substring(split)));
    }

    /// <summary>
    ///     Whether an overload belongs to the constructor family: np.empty / np.zeros / np.ones / np.full (NumPy's creation
    ///     functions over <c>ctors.c</c> and <c>_core/numeric.py</c>), their <c>_like</c> forms, np.eye / np.identity and
    ///     the np.ndarray constructor. The family's overloads without an array parameter take a shape as their operand.
    /// </summary>
    /// <param name="m">The overload.</param>
    /// <returns>True for a public static overload of one of those names on np.</returns>
    private static bool IsCtor(MethodInfo m) =>
        m.DeclaringType == typeof(np) && m.Name is "empty" or "zeros" or "ones" or "full" or "eye" or "identity" or "ndarray"
            or "empty_like" or "zeros_like" or "ones_like" or "full_like";

    /// <summary>Whether a parameter belongs to the constructor family (<see cref="IsCtor(MethodInfo)"/>).</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True when its method is one of the constructors.</returns>
    private static bool IsCtor(ParameterInfo p) => p.Member is MethodInfo m && IsCtor(m);

    /// <summary>Whether a parameter belongs to a <c>_like</c> form of the constructor family (whose order defaults to K).</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of np.empty_like / np.zeros_like / np.ones_like / np.full_like.</returns>
    private static bool IsLike(ParameterInfo p) => p.Member.Name.EndsWith("_like", StringComparison.Ordinal);

    /// <summary>
    ///     Whether an overload is np.indices or np.indices_sparse (NumPy's <c>indices</c> in <c>_core/numeric.py</c>, its
    ///     <c>sparse=False</c> and <c>sparse=True</c> forms), which take no array: their operand is the grid's SHAPE, built
    ///     as a constructor's (<see cref="TryMakeCtorShape"/>) from their own pool (<see cref="IndicesShapeLabels"/>).
    /// </summary>
    /// <param name="m">The overload.</param>
    /// <returns>True for a public static overload of one of the two names on np.</returns>
    /// <remarks>
    ///     Kept apart from <see cref="IsCtor(MethodInfo)"/>: the constructor family's own parameter roles (the fill value,
    ///     the creation order, np.eye's k) must not reach np.indices, whose only other parameter is the dtype.
    /// </remarks>
    private static bool IsIndices(MethodInfo m) =>
        m.DeclaringType == typeof(np) && m.Name is "indices" or "indices_sparse";

    /// <summary>
    ///     Whether an overload is np.frombuffer (NumPy's <c>PyArray_FromBuffer</c>, <c>ctors.c</c>), which takes no array: its
    ///     operand is the buffer its first parameter names (<see cref="BufferForms"/>).
    /// </summary>
    /// <param name="m">The overload (the generic one closed or not).</param>
    /// <returns>True for a public static overload of np.frombuffer.</returns>
    private static bool IsFromBuffer(MethodInfo m) => m.DeclaringType == typeof(np) && m.Name == "frombuffer";

    /// <summary>Whether a parameter belongs to np.frombuffer (<see cref="IsFromBuffer(MethodInfo)"/>).</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True when its method is an np.frombuffer overload.</returns>
    private static bool IsFromBuffer(ParameterInfo p) => p.Member is MethodInfo m && IsFromBuffer(m);

    /// <summary>
    ///     Whether an overload reads or writes NumPy's text and binary files through a C# value the tool builds: np.fromstring
    ///     (NumPy's <c>PyArray_FromString</c>), np.fromfile (<c>PyArray_FromFile</c>) or NDArray.tofile (<c>PyArray_ToFile</c>).
    /// </summary>
    /// <param name="m">The overload.</param>
    /// <returns>True for one of the three.</returns>
    private static bool IsFileIO(MethodInfo m) =>
        (m.DeclaringType == typeof(np) && m.Name is "fromstring" or "fromfile")
        || (m.DeclaringType == typeof(NDArray) && m.Name == "tofile");

    /// <summary>Whether a parameter belongs to one of the file functions (<see cref="IsFileIO(MethodInfo)"/>).</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True when its method is np.fromstring, np.fromfile or NDArray.tofile.</returns>
    private static bool IsFileIO(ParameterInfo p) => p.Member is MethodInfo m && IsFileIO(m);

    /// <summary>
    ///     Whether an overload reads or writes NumPy's text files: np.loadtxt (NumPy's <c>_read</c> over
    ///     <c>_load_from_filelike</c>) or np.savetxt.
    /// </summary>
    /// <param name="m">The overload.</param>
    /// <returns>True for one of the two.</returns>
    private static bool IsTextIO(MethodInfo m) => m.DeclaringType == typeof(np) && m.Name is "loadtxt" or "savetxt";

    /// <summary>Whether a parameter belongs to np.loadtxt or np.savetxt (<see cref="IsTextIO(MethodInfo)"/>).</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True when its method is np.loadtxt or np.savetxt.</returns>
    private static bool IsTextIO(ParameterInfo p) => p.Member is MethodInfo m && IsTextIO(m);

    /// <summary>
    ///     The shapes the constructor family runs on (its operand pool): 0-d, a vector, a matrix, an empty vector, an empty
    ///     axis between others, a unit matrix, a 3-D shape, a negative dim alone and after a positive one (NumPy's
    ///     "negative dimensions are not allowed"), and two dims whose float64 byte count overflows (NumPy's "array is too
    ///     big"). Phase B takes the first five, phase C the first three.
    /// </summary>
    private static readonly string[] CtorShapeLabels =
        { "()", "(3,)", "(2,3)", "(0,)", "(2,0,3)", "(1,1)", "(4,1,2)", "(-1,)", "(2,-3)", "(2147483648,2147483648)" };

    /// <summary>
    ///     The grids np.indices / np.indices_sparse run on: the constructor shapes (<see cref="CtorShapeLabels"/>: the empty
    ///     tuple, empty and unit axes, negative dims - which NumPy's dense form refuses in <c>empty</c> and its sparse form
    ///     reshapes into an empty axis - and two dims whose byte count overflows), then the lengths the dtype's
    ///     <c>arange</c> decides on: a (2, 2) grid, whose aranges stay within the two elements a bool arange allows, and a
    ///     (200,) one, whose arange wraps in int8. Every dtype runs over every shape (phase B takes them all).
    /// </summary>
    private static readonly string[] IndicesShapeLabels =
        { "()", "(3,)", "(2,3)", "(0,)", "(2,0,3)", "(1,1)", "(4,1,2)", "(-1,)", "(2,-3)", "(2147483648,2147483648)", "(2,2)", "(200,)" };

    /// <summary>
    ///     A constructor's shape operand in its parameter's C# form: an int (a 1-D shape's length; np.eye's N and
    ///     np.identity's n take the first dim of any shape), an int[] / long[] of the dims, or a <see cref="Shape"/>.
    /// </summary>
    /// <param name="p">The shape parameter.</param>
    /// <param name="label">The shape label (<see cref="CtorShapeLabels"/>).</param>
    /// <param name="value">Receives the boxed value.</param>
    /// <returns>False when the shape has no value in this form (a 0-d or 2-D shape as an int, a dim past int as an int /
    ///     int[]), or when np.indices_sparse would allocate it (a dim past 2**24); the call is then not made.</returns>
    private static bool TryMakeCtorShape(ParameterInfo p, string label, out object value)
    {
        string inner = label.Trim('(', ')');
        long[] dims = inner.Length == 0
            ? Array.Empty<long>()
            : inner.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(s => long.Parse(s, CultureInfo.InvariantCulture)).ToArray();
        Type t = p.ParameterType;
        bool fitsInt = dims.All(d => d >= int.MinValue && d <= int.MaxValue);
        value = null;
        // np.indices_sparse builds one arange per dim before anything could refuse the grid (NumPy's sparse=True has no
        // empty((N,) + dimensions) to raise "array is too big"), so the overflowing shape would allocate a 2**31-element
        // arange on both sides: 16 GiB. The dense form raises in its allocation first and keeps the shape.
        if (p.Member.Name == "indices_sparse" && dims.Any(d => d > 1 << 24))
            return false;
        if (t == typeof(int))
        {
            // np.eye's N / np.identity's n: the first dim of any shape; a creation function's int shape: a 1-D shape only.
            bool sizeArg = p.Name is "N" or "n";
            if (dims.Length == 0 || (!sizeArg && dims.Length != 1) || !fitsInt)
                return false;
            value = (int)dims[0];
            return true;
        }
        if (t == typeof(int[]))
        {
            if (!fitsInt)
                return false;
            value = dims.Select(d => (int)d).ToArray();
            return true;
        }
        if (t == typeof(long[]))
        {
            value = dims;
            return true;
        }
        if (t == typeof(Shape))
        {
            value = new Shape(dims);
            return true;
        }
        return false;
    }

    /// <summary>
    ///     The fill values of np.full / np.full_like (NumPy's <c>copyto(a, fill_value, casting='unsafe')</c>): a C# int
    ///     first, then each C# number kind (a Python float, a bool, a NumPy float16, NumSharp's decimal, a complex, a char), a
    ///     Python int past int8 and one past every C# integer, NaN and -0.0, text and bytes, None, a 0-d int64 array, a row
    ///     that broadcasts into a (2, 3) or (3,) shape and one that broadcasts into none of them, and a C# int[].
    /// </summary>
    private static readonly string[] CtorFillLabels =
    {
        "7", "2.5", "true", "half1.5", "dec1.5", "1+2j", "char_x", "300", "-1", "2**70", "nan", "-0.0", "str_ab", "bytes_ab",
        "none", "0d_int64", "row3", "row4", "int[3]",
    };

    /// <summary>The fill value of a <see cref="CtorFillLabels"/> label.</summary>
    /// <param name="label">The label.</param>
    /// <returns>The boxed C# value (null for None).</returns>
    private static object MakeCtorFill(string label) => label switch
    {
        "7" => 7,
        "2.5" => 2.5,
        "true" => true,
        "half1.5" => (Half)1.5,
        "dec1.5" => 1.5m,
        "1+2j" => new Complex(1, 2),
        "char_x" => 'x',
        "300" => 300,
        "-1" => -1,
        "2**70" => BigInteger.Pow(2, 70),
        "nan" => double.NaN,
        "-0.0" => -0.0,
        "str_ab" => "ab",
        "bytes_ab" => new byte[] { 97, 98 },
        "none" => null,
        "0d_int64" => np.array(5L),
        "row3" => np.array(new[] { 1.5, 2.5, 3.5 }),
        "row4" => np.array(new[] { 1, 2, 3, 4 }),
        "int[3]" => new[] { 4, 5, 6 },
        _ => throw new ArgumentException("no fill value " + label),
    };

    /// <summary>
    ///     The C# element types of the 15 dtypes NumSharp stores, in the pool's dtype order: the types the coercion
    ///     family's generic overloads are closed over (every one satisfies their <c>unmanaged</c> / <c>struct</c>
    ///     constraints).
    /// </summary>
    private static readonly Type[] HostElementTypes =
    {
        typeof(bool), typeof(sbyte), typeof(byte), typeof(short), typeof(ushort), typeof(int), typeof(uint), typeof(long),
        typeof(ulong), typeof(char), typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex),
    };

    /// <summary>
    ///     Whether an overload belongs to the coercion family: np.array, np.asarray and np.asanyarray (NumPy's
    ///     <c>_array_fromobject_generic</c> over <c>PyArray_FromAny</c>: <c>array_coercion.c</c>'s discovery and fill),
    ///     NDArray.Scalar (NumSharp's <c>np.array(scalar, dtype)</c>) and NDArray's implicit / explicit conversion operators
    ///     (a C# array, a str or a C# scalar into an array, an array into a C# array or scalar).
    /// </summary>
    /// <param name="m">The overload.</param>
    /// <returns>True for a public static overload of one of those names on np or NDArray.</returns>
    private static bool IsCoercion(MethodInfo m) =>
        (m.DeclaringType == typeof(np) && m.Name is "array" or "asarray" or "asanyarray")
        || (m.DeclaringType == typeof(NDArray) && m.Name is "Scalar" or "op_Implicit" or "op_Explicit");

    /// <summary>Whether a parameter belongs to a coercion overload (<see cref="IsCoercion(MethodInfo)"/>).</summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True when its method is one of the coercion family's.</returns>
    private static bool IsCoercion(ParameterInfo p) => p.Member is MethodInfo m && IsCoercion(m);

    /// <summary>
    ///     The element type a pool dtype's operand holds as a C# value: the storage type of the 15 numeric dtypes (the
    ///     order of <see cref="HostElementTypes"/>), <see cref="string"/> for the Unicode and <c>StringDType</c> pools,
    ///     <c>byte[]</c> for the bytes pool (one Python <c>bytes</c> value).
    /// </summary>
    /// <param name="dt">A pool dtype name (<see cref="Dtypes"/>).</param>
    /// <returns>The element type.</returns>
    /// <exception cref="ArgumentException">An unknown dtype name (a bug in the tool).</exception>
    private static Type HostElementType(string dt) => dt switch
    {
        "bool" => typeof(bool),
        "int8" => typeof(sbyte),
        "uint8" => typeof(byte),
        "int16" => typeof(short),
        "uint16" => typeof(ushort),
        "int32" => typeof(int),
        "uint32" => typeof(uint),
        "int64" => typeof(long),
        "uint64" => typeof(ulong),
        "char" => typeof(char),
        "float16" => typeof(Half),
        "float32" => typeof(float),
        "float64" => typeof(double),
        "decimal" => typeof(decimal),
        "complex128" => typeof(Complex),
        "U5" or "T" => typeof(string),
        "S5" => typeof(byte[]),
        _ => throw new ArgumentException("unknown pool dtype " + dt),
    };

    /// <summary>
    ///     The forms a coercion overload's host operand takes, by the parameter's type (and, for an <c>object</c>
    ///     parameter, by the method): each a C# value built from the pool operand by <see cref="TryMakeHostOperand"/>.
    ///     <list type="bullet">
    ///         <item><c>nd</c>: the NDArray itself (np.asanyarray's array case; a 0-d one passed to NDArray.Scalar).</item>
    ///         <item><c>typed</c>: a rectangular C# array of the operand's shape and element type (rank 1 or more; a text
    ///             operand gives a <c>string</c> or <c>byte[]</c> array) - NumSharp's ndarray-like C# array.</item>
    ///         <item><c>jagged</c>: the same values as nested typed arrays (rank 2 or more).</item>
    ///         <item><c>list</c>: nested <c>object[]</c> of boxed values - NumPy's list of lists (rank 1 or more).</item>
    ///         <item><c>grid</c>: a rectangular <c>object</c> array (rank 2 or more), NumSharp's nested list of rows.</item>
    ///         <item><c>rows</c>: an <c>object[]</c> holding the operand's rows as NDArrays (rank 2 or more).</item>
    ///         <item><c>glist</c> / <c>linq</c> / <c>memory</c>: a <c>List&lt;T&gt;</c>, an iterator that is no collection,
    ///             a <c>Memory&lt;T&gt;</c> of a 1-D operand's values.</item>
    ///         <item><c>tuple</c>: a ValueTuple of a 1-D operand's boxed values (seven at most).</item>
    ///         <item><c>scalar</c>: a 0-d operand's value as its C# scalar (<c>string</c> / <c>byte[]</c> for text).</item>
    ///         <item><c>flat</c>: the values in C order as a <c>T[]</c> (a generic <c>T[]</c> parameter; any rank).</item>
    ///         <item><c>str</c> / <c>strs</c> / <c>withnull</c>: a 0-d operand's value as a str, every value's str as a
    ///             <c>string[]</c>, that array with its first entry null.</item>
    ///         <item><c>empty</c> / <c>null</c>: the empty str and null, built once (the float64 0-d slot).</item>
    ///         <item><c>data</c>: the operand's <see cref="np.MemoryView"/> (<c>ndarray.data</c>).</item>
    ///     </list>
    /// </summary>
    /// <param name="p">The host operand parameter.</param>
    /// <returns>The form names, in the order the pools try them.</returns>
    private static string[] HostForms(ParameterInfo p)
    {
        Type t = p.ParameterType.IsByRef ? p.ParameterType.GetElementType() : p.ParameterType;
        if (t == typeof(object))
            // NDArray.Scalar builds ONE element: its object value is a scalar, None, or a 0-d array passed as an object.
            return p.Member.Name == "Scalar"
                ? new[] { "scalar", "nd", "null" }
                : new[] { "nd", "typed", "jagged", "list", "grid", "rows", "glist", "linq", "memory", "tuple", "scalar", "null" };
        if (t == typeof(Array)) return new[] { "typed", "jagged", "list", "grid" };
        if (t == typeof(string)) return new[] { "str", "empty", "null" };
        if (t == typeof(string[])) return new[] { "strs", "withnull", "null" };
        if (t == typeof(np.MemoryView)) return new[] { "data", "null" };
        if (t.IsArray && t.GetArrayRank() > 1) return new[] { "typed" };
        if (t.IsArray && t.GetElementType().IsArray) return new[] { "jagged" };
        if (t.IsArray) return new[] { "flat" };
        if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(IEnumerable<>)) return new[] { "glist", "linq" };
        return new[] { "scalar" };
    }

    /// <summary>
    ///     Builds the coercion family's host operand: <paramref name="form"/> (<see cref="HostForms"/>) of the pool operand
    ///     for a parameter of type <paramref name="t"/>, or false when the operand has no such form (a rank the form
    ///     needs, an element type the parameter does not take, a value built only once).
    /// </summary>
    /// <param name="t">The parameter type (a by-ref <c>in object</c> already unwrapped).</param>
    /// <param name="form">The form.</param>
    /// <param name="dt">The operand's pool dtype.</param>
    /// <param name="baseLayout">The operand's pool layout.</param>
    /// <param name="operand">The pool operand.</param>
    /// <param name="oneElement">Whether the parameter is ONE element's value (NDArray.Scalar's): its <c>nd</c> form is a
    ///     0-d operand only.</param>
    /// <param name="value">Receives the host value.</param>
    /// <returns>Whether the form exists for this operand and parameter.</returns>
    /// <exception cref="ArgumentException">An unknown form (a bug in the tool).</exception>
    private static bool TryMakeHostOperand(Type t, string form, string dt, string baseLayout, NDArray operand, bool oneElement,
        out object value)
    {
        value = null;
        Type e = HostElementType(dt);
        int rank = operand.ndim;
        long size = operand.size;
        bool scalarLayout = baseLayout is "0d" or "0d_special";
        // The once-built forms (null, the empty str) take one slot: the float64 0-d operand.
        bool onceSlot = dt == "float64" && baseLayout == "0d";
        // A typed parameter (T, T[], T[][], T[,], IEnumerable<T>) takes only the operands of its own element type.
        if (PinnedHostElement(t) is { } pinned && pinned != e)
            return false;
        object[] items = Enumerable.Range(0, (int)size).Select(i => operand.GetAtIndex(i)).ToArray();
        int[] dims = operand.shape.Select(d => (int)d).ToArray();
        switch (form)
        {
            case "nd":
                // NDArray.Scalar's value is ONE element, so only a 0-d array reaches it as an object; np.asanyarray takes
                // every operand.
                if (oneElement && !scalarLayout)
                    return false;
                value = operand;
                return true;
            case "typed":
                if (rank == 0 || (t.IsArray && t.GetArrayRank() != rank))
                    return false;
                value = Rectangular(e, dims, items);
                return true;
            case "jagged":
                if (rank < 2 || (t.IsArray && JaggedDepth(t) != rank))
                    return false;
                value = Jagged(e, dims, items, 0, 0);
                return true;
            case "list":
                if (rank == 0)
                    return false;
                value = Jagged(typeof(object), dims, items, 0, 0);
                return true;
            case "grid":
                if (rank < 2)
                    return false;
                value = Rectangular(typeof(object), dims, items);
                return true;
            case "rows":
                if (rank < 2)
                    return false;
                value = Enumerable.Range(0, dims[0]).Select(i => (object)operand[i]).ToArray();
                return true;
            case "glist":
            case "linq":
            case "memory":
                if (rank != 1)
                    return false;
                Array typedItems = Rectangular(e, dims, items);
                value = form switch
                {
                    "glist" => Activator.CreateInstance(typeof(List<>).MakeGenericType(e), typedItems),
                    "linq" => typeof(Snapshot).GetMethod(nameof(Iterate), BindingFlags.NonPublic | BindingFlags.Static)
                        .MakeGenericMethod(e).Invoke(null, new object[] { typedItems }),
                    _ => Activator.CreateInstance(typeof(Memory<>).MakeGenericType(e), typedItems),
                };
                return true;
            case "tuple":
                if (rank != 1 || size > 7)
                    return false;
                value = TupleOf(items);
                return true;
            case "scalar":
                if (!scalarLayout)
                    return false;
                value = items[0];
                return true;
            case "flat":
                value = Rectangular(e, new[] { (int)size }, items);
                return true;
            case "str":
                if (!scalarLayout)
                    return false;
                value = TextOf(items[0]);
                return true;
            case "strs":
                if (rank == 0)
                    return false;
                value = items.Select(TextOf).ToArray();
                return true;
            case "withnull":
                if (rank != 1 || size == 0 || e != typeof(string))
                    return false;
                var withNull = items.Select(TextOf).ToArray();
                withNull[0] = null;
                value = withNull;
                return true;
            case "empty":
                if (!onceSlot)
                    return false;
                value = "";
                return true;
            case "null":
                if (!onceSlot)
                    return false;
                value = null;
                return true;
            case "data":
                // ndarray.data refuses a dtype that holds references (StringDType, as NumPy refuses its buffer): that operand
                // has no MemoryView, so no line.
                try
                {
                    value = operand.data;
                }
                catch (ValueError)
                {
                    return false;
                }
                return true;
            default:
                throw new ArgumentException("unknown host operand form " + form);
        }
    }

    /// <summary>The innermost element type of an array type (<c>double</c> for <c>double[][]</c> and <c>double[,]</c>).</summary>
    /// <param name="t">An array type.</param>
    /// <returns>Its innermost element type.</returns>
    private static Type InnermostElement(Type t)
    {
        while (t.IsArray)
            t = t.GetElementType();
        return t;
    }

    /// <summary>How many single-rank array levels a jagged array type nests (<c>double[][]</c>: 2).</summary>
    /// <param name="t">An array type.</param>
    /// <returns>The nesting depth.</returns>
    private static int JaggedDepth(Type t)
    {
        int depth = 0;
        while (t.IsArray && t.GetArrayRank() == 1)
        {
            depth++;
            t = t.GetElementType();
        }
        return depth;
    }

    /// <summary>A rectangular C# array of <paramref name="e"/> elements and <paramref name="dims"/>, filled in C order.</summary>
    /// <param name="e">The element type.</param>
    /// <param name="dims">The dimensions (rank 1 or more).</param>
    /// <param name="items">The values in C order (each assignable to <paramref name="e"/>).</param>
    /// <returns>The array.</returns>
    private static Array Rectangular(Type e, int[] dims, object[] items)
    {
        var result = Array.CreateInstance(e, dims);
        var coords = new int[dims.Length];
        for (int flat = 0; flat < items.Length; flat++)
        {
            result.SetValue(items[flat], coords);
            for (int d = coords.Length - 1; d >= 0; d--)
            {
                if (++coords[d] < dims[d])
                    break;
                coords[d] = 0;
            }
        }
        return result;
    }

    /// <summary>
    ///     A jagged C# array (nested single-rank arrays) of <paramref name="e"/> leaves for the axes from
    ///     <paramref name="axis"/> on, starting at flat position <paramref name="start"/>: <c>double[][]</c> for a 2-D
    ///     operand, <c>object[]</c> holding <c>object[]</c> rows when <paramref name="e"/> is <c>object</c> (NumPy's nested
    ///     list).
    /// </summary>
    /// <param name="e">The leaf element type.</param>
    /// <param name="dims">The operand's dimensions.</param>
    /// <param name="items">The values in C order.</param>
    /// <param name="axis">The axis this level spans.</param>
    /// <param name="start">The flat position of this level's first value.</param>
    /// <returns>The nested array.</returns>
    private static Array Jagged(Type e, int[] dims, object[] items, int axis, int start)
    {
        int stride = 1;
        for (int d = axis + 1; d < dims.Length; d++)
            stride *= dims[d];
        Type levelElement = e;
        // The element type of this level: e for the last axis, an array of the next level's type above it (object stays
        // object at every level, NumPy's list of lists).
        if (e != typeof(object))
            for (int d = axis + 1; d < dims.Length; d++)
                levelElement = levelElement.MakeArrayType();
        var level = Array.CreateInstance(levelElement, dims[axis]);
        for (int i = 0; i < dims[axis]; i++)
            level.SetValue(axis == dims.Length - 1 ? items[start + i] : Jagged(e, dims, items, axis + 1, start + i * stride), i);
        return level;
    }

    /// <summary>A ValueTuple of up to seven boxed values (<see cref="System.Runtime.CompilerServices.ITuple"/>), the
    ///     empty <see cref="ValueTuple"/> for none.</summary>
    /// <param name="items">The values.</param>
    /// <returns>The tuple.</returns>
    private static object TupleOf(object[] items)
    {
        if (items.Length == 0)
            return ValueTuple.Create();
        var create = typeof(ValueTuple).GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Single(m => m.Name == "Create" && m.GetGenericArguments().Length == items.Length);
        return create.MakeGenericMethod(Enumerable.Repeat(typeof(object), items.Length).ToArray()).Invoke(null, items);
    }

    /// <summary>An iterator over <paramref name="items"/> that is no collection (no <c>Count</c>, no <c>CopyTo</c>): the
    ///     <c>linq</c> host form, NumPy's generator argument.</summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="items">The values.</param>
    /// <returns>The lazy sequence.</returns>
    private static IEnumerable<T> Iterate<T>(T[] items)
    {
        foreach (var item in items)
            yield return item;
    }

    /// <summary>A value's text as a C# str: a <c>string</c> itself, a <c>byte[]</c> decoded as Latin-1, a number in the
    ///     invariant culture, null as null.</summary>
    /// <param name="v">The value.</param>
    /// <returns>Its text.</returns>
    private static string TextOf(object v) => v switch
    {
        null => null,
        string s => s,
        byte[] b => Encoding.Latin1.GetString(b),
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        _ => v.ToString(),
    };

    /// <summary>
    ///     Whether a parameter belongs to np.einsum, whose <c>subscripts</c> / <c>operands</c> / <c>optimize</c> roles are its
    ///     alone: no other snapshot target declares them.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.einsum overload.</returns>
    private static bool IsEinsum(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name == "einsum";

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
    ///     Whether a parameter belongs to the sort family - np.sort, np.argsort, np.partition, np.argpartition, np.lexsort
    ///     and NDArray's sort, argsort, partition and argpartition - whose <c>kind</c> / <c>stable</c> / <c>order</c> /
    ///     <c>kth</c> / <c>keys</c> roles are theirs alone: np.isin names a parameter <c>kind</c>, and np.copy /
    ///     np.ravel / np.reshape name an <c>order</c> with other meanings, so their snapshots must not change.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of one of the nine public members.</returns>
    /// <summary>
    ///     Whether a parameter belongs to the max / min family with NumPy's keywords - np.max, np.amax, np.min, np.amin,
    ///     np.nanmax, np.nanmin and NDArray's max and min - whose <c>initial</c> role is theirs alone: np.sum / np.prod
    ///     take an <c>initial</c> too, and their snapshots must not change.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of one of the eight public members.</returns>
    private static bool IsMaxMinFamily(ParameterInfo p) =>
        (p.Member.DeclaringType == typeof(np) && p.Member.Name is "max" or "amax" or "min" or "amin" or "nanmax" or "nanmin")
        || (p.Member.DeclaringType == typeof(NDArray) && p.Member.Name is "max" or "min");

    private static bool IsSortFamily(ParameterInfo p) =>
        (p.Member.DeclaringType == typeof(np) && p.Member.Name is "sort" or "argsort" or "partition" or "argpartition" or "lexsort")
        || (p.Member.DeclaringType == typeof(NDArray) && p.Member.Name is "sort" or "argsort" or "partition" or "argpartition");

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
    ///     Whether a parameter belongs to np.interp, whose <c>xp</c> / <c>fp</c> / <c>left</c> / <c>right</c> /
    ///     <c>period</c> roles are its alone: other functions name a second operand <c>left</c> / <c>right</c> (an NDArray,
    ///     the Operand2 role), and their snapshots must not change.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.interp overload.</returns>
    /// <summary>
    ///     Whether a parameter belongs to an np.fft function (a public instance member of the module class
    ///     <see cref="FourierModule"/>), whose <c>n</c> / <c>norm</c> / <c>s</c> / <c>d</c> / <c>device</c> roles are its alone:
    ///     np.linalg.matrix_power's <c>n</c> is an exponent, and the snapshots of the other targets must not change.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public instance member of <see cref="FourierModule"/>.</returns>
    private static bool IsFft(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(FourierModule);

    private static bool IsInterp(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name == "interp";

    /// <summary>
    ///     Whether a parameter belongs to np.nan_to_num or np.real_if_close, whose <c>nan</c> / <c>posinf</c> /
    ///     <c>neginf</c> and <c>tol</c> roles are theirs alone: no other target names an object <c>nan</c> or a double
    ///     <c>tol</c> it should vary this way.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.nan_to_num or np.real_if_close overload.</returns>
    private static bool IsTypeCheckFill(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np) && p.Member.Name is "nan_to_num" or "real_if_close";

    /// <summary>
    ///     Whether a parameter belongs to np.unravel_index, np.ravel_multi_index, np.packbits / np.unpackbits or
    ///     np.bincount, whose <c>shape</c> / <c>order</c> / <c>index</c>, <c>multi_index</c> / <c>coords</c> /
    ///     <c>dims</c> / <c>mode</c> / <c>modes</c>, <c>bitorder</c> / <c>count</c> and <c>minlength</c> roles are
    ///     theirs alone: the order of np.unravel_index and np.ravel_multi_index defaults to C (the general <see
    ///     cref="Role.Order"/> starts at K, which NumPy refuses there), and no other target names these parameters so.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for a parameter of a public static np.unravel_index, np.ravel_multi_index, np.packbits,
    ///     np.unpackbits or np.bincount overload.</returns>
    private static bool IsMultiarrayIndexing(ParameterInfo p) =>
        p.Member.DeclaringType == typeof(np)
        && p.Member.Name is "unravel_index" or "ravel_multi_index" or "packbits" or "unpackbits" or "bincount";

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
        // np.polyint's order and constants (only its): m around its default 1 (0, 2, a negative order, 5), k its own
        // variants, before the general array roles would read an NDArray k as a second operand.
        if (p.Member.DeclaringType == typeof(np) && p.Member.Name == "polyint")
        {
            if (n == "m" && t == typeof(int)) return Role.IntArg;
            if (n == "k" && t == typeof(NDArray)) return Role.PolyintK;
            if (n == "k" && t == typeof(double)) return Role.PolyintKDouble;
        }
        // The item-selection family's own parameters (only theirs; its operand, axis, out and refcheck take the general
        // roles below), before the general array roles would read an index array or a mask as a second operand.
        if (IsItemSelection(p) && ItemArgLabels(p) is not null)
            return Role.ItemArg;
        // The assignment family's sources and fill value (only theirs; np.copyto's dst, casting and where take the general
        // roles below), before the general array roles would read an array source as a second operand.
        if (IsAssignment(p) && AssignArgLabels(p) is not null)
            return Role.AssignArg;
        // The constructor family's own parameters (only theirs; the dtype and a _like form's prototype take the general
        // roles below): the fill value, a _like form's shape=, np.eye's M and k, a creation function's order (C first; a
        // _like form's order defaults to K and takes the general Order role), np.ndarray's buffer / offset / strides, and
        // the device.
        if (IsCtor(p))
        {
            if (n == "fill_value" && t == typeof(object)) return Role.CtorFill;
            if (n == "shape" && t == typeof(Shape) && IsLike(p)) return Role.CtorShapeArg;
            if (n == "M" && t == typeof(int?)) return Role.EyeM;
            if (n == "k" && t == typeof(int)) return Role.EyeK;
            if (n == "order" && t == typeof(char) && !IsLike(p)) return Role.CtorOrder;
            if (n == "buffer" && t == typeof(object)) return Role.NdBuffer;
            if (n == "offset" && t == typeof(long)) return Role.NdOffset;
            if (n == "strides" && t == typeof(long[])) return Role.NdStrides;
            if (n == "device" && t == typeof(string)) return Role.CoerceDevice;
        }
        // np.frombuffer's own parameters (only its; its buffer is the operand, set in RunOverload, and its dispose keeps its
        // default null: the memory stays the tool's): the address overloads' byte length (built from the buffer), the count,
        // the byte offset, and the dtype as a descriptor, an NPTypeCode or a name.
        if (IsFromBuffer(p))
        {
            if (n == "byteLength" && t == typeof(long)) return Role.BufferLength;
            if (n == "count" && t == typeof(long)) return Role.BufferCount;
            if (n == "offset" && t == typeof(long)) return Role.BufferOffset;
            if (n == "dtype" && t == typeof(DType)) return Role.BufferDType;
            if (n == "dtype" && t == typeof(NPTypeCode)) return Role.BufferTypeCode;
            if (n == "dtype" && t == typeof(string)) return Role.BufferDTypeName;
        }
        // The file functions' own parameters (only theirs; np.fromstring's / np.fromfile's first parameter is the operand,
        // set in RunOverload, and their like= keeps its default): the separator, the count, np.fromfile's offset, the dtype
        // (np.frombuffer's variants: numeric, non-native, fixed-width and unsized strings, StringDType), NDArray.tofile's
        // format and its sink (a stream or a file name).
        if (IsFileIO(p))
        {
            if (n == "dtype" && t == typeof(DType)) return Role.BufferDType;
            if (n == "sep" && t == typeof(string)) return Role.IoSep;
            if (n == "count" && t == typeof(long)) return Role.BufferCount;
            if (n == "offset" && t == typeof(long)) return Role.BufferOffset;
            if (n == "format" && t == typeof(string)) return Role.ToFileFormat;
            if (n == "stream" && t == typeof(Stream) && p.Member.Name == "tofile") return Role.SinkStream;
            if (n == "fid" && t == typeof(string)) return Role.SinkFile;
        }
        // np.loadtxt's and np.savetxt's own parameters (np.loadtxt's first parameter is the operand, set in RunOverload, and
        // savetxt's X the general operand): savetxt's sink (a file name, a stream, a TextWriter), its fmt as a list (built
        // from X's columns), the dtype (np.frombuffer's variants), unpack (a bool), every other one TextIOArg's variants.
        if (IsTextIO(p) && n != "X")
        {
            if (p.Member.Name == "savetxt" && p.Position == 0)
                return t == typeof(Stream) ? Role.SinkStream : t == typeof(TextWriter) ? Role.SinkWriter : Role.SinkFile;
            if (n == "fmt" && t == typeof(string[])) return Role.SaveTxtFmts;
            if (n == "dtype" && t == typeof(DType)) return Role.BufferDType;
            if (n == "unpack" && t == typeof(bool)) return Role.Bool;
            return Role.TextIOArg;
        }
        // np.where's x / y as a C# value (only its object overloads: its NDArray x / y take the second-operand role below).
        if (IsWhere(p) && n is "x" or "y" && t == typeof(object))
            return Role.WhereScalar;
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
                // np.partition / np.argpartition and the ndarray methods' index array of positions.
                : n == "kth" && IsSortFamily(p) ? Role.KthArray
                // np.interp's sample points and values (required, fixed arrays: see the InterpXp / InterpFp variants).
                : n == "xp" && IsInterp(p) ? Role.InterpXp
                : n == "fp" && IsInterp(p) ? Role.InterpFp
                // np.extract's arr (condition, the first array, is the operand) and the value NDArray.real / NDArray.imag's
                // setters copy into the array (the array is the operand).
                : n == "arr" && IsExtract(p) ? Role.Operand2
                : n == "value" && IsComplexPartSetter(p) ? Role.Operand2
                : n == "q" ? Role.QArray
                : n == "bins" ? Role.BinsArray
                : p.HasDefaultValue ? Role.Default : Role.Unsupported;
        // The spacing generators' own parameters (only theirs, and before the general axis rule: their axis is the moveaxis
        // destination of a 1-D result, not an axis of an operand): the sample count, the log base, the destination and the
        // device. np.trim_zeros' trim spec and np.vander's column count (only theirs: no other target names them).
        if (IsSpacing(p))
        {
            if (n == "num" && (t == typeof(int) || t == typeof(long))) return Role.SpaceNum;
            if (n == "base" && t == typeof(double)) return Role.SpaceBase;
            if (n == "axis" && t == typeof(int)) return Role.SpaceAxis;
            if (n == "device" && t == typeof(string)) return Role.CoerceDevice;
        }
        if (p.Member.DeclaringType == typeof(np) && p.Member.Name == "trim_zeros" && n == "trim" && t == typeof(string))
            return Role.TrimSpec;
        if (p.Member.DeclaringType == typeof(np) && p.Member.Name == "vander" && n == "N" && t == typeof(int?))
            return Role.VanderN;
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
        // np.diag / np.diagflat's diagonal k, the same diagonal as np.eye's (0 the main one, a positive value above, a
        // negative one below): np.eye's variants, so the 1-D branch builds matrices padded on either side and far past the
        // vector, and the 2-D branch reads diagonals beyond either corner.
        if (n == "k" && t == typeof(int) && p.Member is MethodInfo km && km.DeclaringType == typeof(np) && km.Name is "diag" or "diagflat")
            return Role.EyeK;
        // np.cov's nullable ddof (null = bias's default).
        if (n == "ddof" && t == typeof(int?)) return Role.Ddof;
        // The gufuncs' axes= list, matmul's nullable keepdims and its casting rule.
        if (n == "axes" && t == typeof(int[][])) return Role.AxesList;
        if (n == "keepdims" && t == typeof(bool?)) return Role.KeepdimsNullable;
        // NDArray.astype's dtype and casting rule take the cast family's variants (only its: every other target keeps the
        // general DType and Casting variants, and so its snapshot).
        if (IsCast(p) && t == typeof(DType)) return Role.CastDType;
        if (IsCast(p) && n == "casting" && t == typeof(string)) return Role.CastCasting;
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
            if (n == "obj" && t == typeof(bool[])) return Role.InsertObjMask;
            if (n == "value" && t == typeof(object)) return Role.InsertValue;
        }
        // NDArray.Contains's value (only its: see IsContains).
        if (n == "value" && t == typeof(object) && IsContains(p)) return Role.ContainsValue;
        // np.isin's algorithm name (only its: the sort family's kind takes the roles below).
        if (n == "kind" && t == typeof(string) && IsIsin(p)) return Role.IsinKind;
        // The max / min family's initial= (an object: NumPy's array_like scalar).
        if (n == "initial" && t == typeof(object) && IsMaxMinFamily(p)) return Role.Initial;
        // The sort family: the sort kind of sort / argsort, the select kind of partition / argpartition, stable, order
        // (an object for sort / argsort, a string for the partitions: one role, field names), and kth as an int / int[].
        if (IsSortFamily(p))
        {
            if (n == "kind" && t == typeof(string))
                return p.Member.Name is "partition" or "argpartition" ? Role.PartitionKind : Role.SortKind;
            if (n == "stable" && t == typeof(bool?)) return Role.Stable;
            if (n == "order" && (t == typeof(object) || t == typeof(string))) return Role.SortOrder;
            if (n == "kth" && t == typeof(int)) return Role.Kth;
            if (n == "kth" && t == typeof(int[])) return Role.KthSeq;
        }
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
        // np.nan_to_num's fill values and np.real_if_close's tolerance (only theirs).
        if (IsTypeCheckFill(p))
        {
            if (n is "nan" or "posinf" or "neginf" && t == typeof(object)) return Role.NanToNumFill;
            if (n == "tol" && t == typeof(double)) return Role.RealIfCloseTol;
        }
        // np.unravel_index's dimensions, order and flat index, np.ravel_multi_index's dims, mode and modes (its order
        // is np.unravel_index's), np.packbits / np.unpackbits' bitorder and count, and np.bincount's minlength (only
        // theirs; before the general Order rule below, whose base K NumPy refuses here).
        if (IsMultiarrayIndexing(p))
        {
            if (n == "shape" && t == typeof(int[])) return Role.UnravelShape;
            if (n == "order" && t == typeof(char)) return Role.UnravelOrder;
            if (n == "index" && t == typeof(long)) return Role.UnravelFlat;
            if (n == "dims" && t == typeof(int[])) return Role.RavelDims;
            if (n == "mode" && t == typeof(string)) return Role.RavelMode;
            if (n == "modes" && t == typeof(string[])) return Role.RavelModes;
            if (n == "bitorder" && t == typeof(string)) return Role.PackBitOrder;
            if (n == "count" && t == typeof(long?)) return Role.UnpackCount;
            if (n == "minlength" && t == typeof(int)) return Role.BincountMinLength;
        }
        // np.interp's fills and period (only its).
        if (IsInterp(p))
        {
            if (n is "left" or "right" && t == typeof(double?)) return Role.InterpFill;
            if (n is "left" or "right" && t == typeof(Complex?)) return Role.InterpFillComplex;
            if (n == "period" && t == typeof(double?)) return Role.InterpPeriod;
        }
        // np.einsum's subscripts and optimize (only its; its out, dtype, order and casting take the general roles).
        if (IsEinsum(p))
        {
            if (n == "subscripts" && t == typeof(string)) return Role.EinsumSubscripts;
            if (n == "optimize" && t == typeof(object)) return Role.EinsumOptimize;
        }
        // np.fft's arguments (only its: its a / x take the operand role above, its axis / axes the general roles).
        if (IsFft(p))
        {
            if (n == "n" && t == typeof(int?)) return Role.FftN;
            if (n == "norm" && t == typeof(string)) return Role.FftNorm;
            if (n == "s" && t == typeof(int[])) return Role.FftS;
            if (n == "n" && (t == typeof(int) || t == typeof(long) || t == typeof(double))) return Role.FftFreqN;
            if (n == "d" && t == typeof(double)) return Role.FftD;
            if (n == "device" && t == typeof(string)) return Role.FftDevice;
        }
        // The coercion family's own parameters (only its: the other targets keep their snapshots, and np.fft's device has a
        // role of its own): ndmin / ndim around their defaults, copy as NumPy's tri-state, the device, the dtype given by
        // name (np.asarray's MemoryView overload) and the count np.array reads from an IEnumerable.
        if (IsCoercion(p))
        {
            if (n is "ndmin" or "ndim" && t == typeof(int)) return Role.IntArg;
            if (n == "copy" && t == typeof(bool?)) return Role.CoerceCopy;
            if (n == "device" && t == typeof(string)) return Role.CoerceDevice;
            if (n == "dtype" && t == typeof(string)) return Role.CoerceDTypeName;
            if (n == "size" && (t == typeof(int) || t == typeof(long))) return Role.CoerceSize;
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
            case Role.SortKind:
                // The default (None) first, the four names NumPy documents, then what its sortkind_parser reads from the
                // first character alone (a capital, one letter, a name it only starts) and what it refuses (another first
                // letter, the empty string).
                foreach (var s in new[] { null, "quicksort", "heapsort", "stable", "mergesort", "Q", "s", "spam", "timsort", "x", "" })
                    Add(s is null ? "null" : s.Length == 0 ? "empty" : s, () => s);
                break;
            case Role.PartitionKind:
                // The default first, then null (NumPy's None, which the select-kind converter refuses only as text) and the
                // strings it refuses: a sort kind, a capital, another word.
                foreach (var s in new[] { "introselect", null, "quicksort", "Introselect", "x" })
                    Add(s ?? "null", () => s);
                break;
            case Role.Stable:
                // The default (None) first, then both values: with a kind given either one is refused.
                Add("null", () => (bool?)null);
                Add("true", () => (bool?)true);
                Add("false", () => (bool?)false);
                break;
            case Role.SortOrder:
            {
                // The default (None) first, then a field name and, for sort / argsort's object order, a list of names:
                // every NumSharp dtype has no fields, so each is NumPy's refusal.
                Add("null", () => null);
                Add("f0", () => "f0");
                if (p.ParameterType == typeof(object))
                    Add("[f0,f1]", () => new[] { "f0", "f1" });
                break;
            }
            case Role.Kth:
                // A position inside every axis of the pool first, then the first, the last (negative), the second axis
                // position of a (3, 4) operand, one past every pool axis and its negative.
                foreach (var k in new[] { 1, 0, -1, 2, 99, -99 })
                    Add(k.ToString(CultureInfo.InvariantCulture), () => k);
                break;
            case Role.KthSeq:
                // One position first, then two sorted, two unsorted, a repeated one, none (NumPy's empty intp array: no
                // partition at all), a negative one with the first, one out of range, and null.
                foreach (var seq in new[]
                         {
                             new[] { 1 }, new[] { 0, 2 }, new[] { 2, 0 }, new[] { 1, 1 }, Array.Empty<int>(), new[] { -1, 0 },
                             new[] { 99 },
                         })
                    Add("[" + string.Join(",", seq) + "]", () => seq.ToArray());
                Add("null", () => null);
                break;
            case Role.KthArray:
                // An int64 index first, then two, unsorted int32 ones, a 0-d index, a uint8 one, a uint64 index past int64
                // (NumPy's modular cast to intp makes it negative), an empty index array, and the forms NumPy refuses: a
                // boolean index, a float index, a two-dimensional index ("object too deep"), and null.
                Add("i64[1]", () => np.array(new long[] { 1 }));
                Add("i64[0,2]", () => np.array(new long[] { 0, 2 }));
                Add("i32[2,0]", () => np.array(new[] { 2, 0 }));
                Add("0d_i64_1", () => np.array(1L));
                Add("u8[1]", () => np.array(new byte[] { 1 }));
                Add("u64[2^63]", () => np.array(new ulong[] { 1UL << 63 }));
                Add("i64[]", () => np.array(Array.Empty<long>()));
                Add("bool[T]", () => np.array(new[] { true }));
                Add("f64[1.0]", () => np.array(new[] { 1.0 }));
                Add("2d_i64", () => np.array(new long[] { 1 }).reshape(new Shape(1, 1)));
                Add("null", () => null);
                break;
            case Role.LexKeys:
                // Built from the operand (MakeLexKeys); the factory here is never called. The base is the operand alone.
                Add(LexKeysVariantLabels[0], () => null);
                break;
            case Role.EvalExpr:
            case Role.EvalExprPositional:
            case Role.EvalOperands:
                // Built from the operand (MakeEvalExpr; the operand list is the operand alone); the factory here is never
                // called. The base expression is the identity of the operand.
                Add(role == Role.EvalOperands ? "one" : EvalExprVariantLabels[0], () => null);
                break;
            case Role.ItemArg:
                // Built by MakeItemArg (from the operand where a value must fit it); the factory here is never called.
                Add(ItemArgLabels(p)[0], () => null);
                break;
            case Role.AssignArg:
                // Built by MakeAssignArg (from the destination where a source must fit it); the factory is never called.
                Add(AssignArgLabels(p)[0], () => null);
                break;
            case Role.Initial:
                // None first (no initial: a where= mask then meets NumPy's no-identity error), then C# numbers as Python's int
                // and float (zero, a negative, a fraction, a value past int8 / uint8 and one past int64's range as a
                // ulong), NaN and an infinity (refused by an integer loop), NumPy scalars (a float16, a char), a bool, a 0-d
                // array, and what NumPy refuses (text, a sequence).
                foreach (var (label, value) in new (string, object)[]
                         {
                             ("null", null), ("0", 0), ("-1", -1), ("2.5", 2.5), ("300", 300), ("u64max", ulong.MaxValue),
                             ("nan", double.NaN), ("-inf", double.NegativeInfinity), ("f16_1.5", (Half)1.5), ("char_A", 'A'),
                             ("true", true), ("0d_i64_7", null), ("text", "7"), ("seq", new[] { 1, 2 }),
                         })
                    Add(label, label == "0d_i64_7" ? () => np.array(7L) : () => value);
                break;
            case Role.NditerOps:
            case Role.NditerOpAxes:
            case Role.NditerItershape:
                // Built from the operand (MakeNditerOps / MakeNditerOpAxes / MakeNditerItershape); the factory here is never
                // called. The base is the operand alone / no op_axes / no itershape.
                Add(role == Role.NditerOps ? "one" : "null", () => null);
                break;
            case Role.EinsumSubscripts:
            case Role.EinsumOps:
                // Built from the operand (MakeEinsumSubscripts / MakeEinsumOps); the factory here is never called. The base is
                // the identity of the operand's labels / the operand alone.
                Add(role == Role.EinsumSubscripts ? EinsumSubscriptsVariantLabels[0] : EinsumOpsVariantLabels[0], () => null);
                break;
            case Role.EinsumOptimize:
                // The default (NumPy's False) first, then the forms NumPy accepts (True, a path name, a name it tolerates and
                // treats as greedy, a (name, memory) pair, a pair whose memory is a bool, which NumPy's isinstance(x, (int,
                // float)) takes for an int) and the ones it refuses (a number, a pair whose memory is text: "Did not
                // understand the path").
                Add("null", () => null);
                Add("true", () => true);
                Add("greedy", () => "greedy");
                Add("optimal", () => "optimal");
                Add("x", () => "x");
                Add("pair", () => ("greedy", 100));
                Add("pairbool", () => ("greedy", true));
                Add("pairtext", () => ("greedy", "100"));
                Add("int", () => 5);
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
            case Role.CoerceCopy:
                // NumPy's copy=None first (a copy only when the dtype or the layout needs one), then always and never (which
                // raises "Unable to avoid copy" when one is needed).
                Add("null", () => null);
                Add("true", () => (bool?)true);
                Add("false", () => (bool?)false);
                break;
            case Role.CoerceDevice:
                // None first, NumPy's one device, and one it refuses.
                Add("null", () => null);
                Add("cpu", () => "cpu");
                Add("cuda", () => "cuda");
                break;
            case Role.CoerceDTypeName:
                // No dtype first, then names of a number, an integer and a sized text dtype, and a name np.dtype refuses.
                Add("null", () => null);
                foreach (var name in new[] { "float64", "int32", "<U5", "bogus" })
                    Add(name, () => name);
                break;
            case Role.SpacePair:
            case Role.SpaceStop:
                // Built from the pair label in Call (MakeSpaceValue); the factory here is never called.
                Add("pair", () => null);
                break;
            case Role.SpaceNum:
            {
                // NumPy's default count first, then none, one, two (the endpoints only), an odd count and a negative one
                // (NumPy's "Number of samples, -1, must be non-negative.").
                bool isLong = p.ParameterType == typeof(long);
                foreach (var count in new[] { 50, 0, 1, 2, 7, -1 })
                    Add(count.ToString(CultureInfo.InvariantCulture), () => isLong ? (object)(long)count : count);
                break;
            }
            case Role.SpaceBase:
                // The default base first, then two, a base below one, a negative one (NaN for a fractional exponent), one,
                // zero and e.
                foreach (var b in new[] { 10.0, 2.0, 0.5, -2.0, 1.0, 0.0, Math.E })
                    Add(b.ToString("R", CultureInfo.InvariantCulture), () => b);
                break;
            case Role.SpaceAxis:
                // The scalar generators' result is 1-D: NumPy's moveaxis destination accepts 0 and -1 and refuses the rest.
                foreach (var a in new[] { 0, -1, 1, -2 })
                    Add(a.ToString(CultureInfo.InvariantCulture), () => a);
                break;
            case Role.TrimSpec:
                // The default first, then each spelling NumPy accepts (either case, either order) and three it refuses: an
                // unknown letter, the empty spec, and null (read as the empty spec).
                foreach (var s in new[] { "fb", "f", "b", "BF", "bf", "fx", "", "null" })
                    Add(s == "" ? "empty" : s, () => s == "null" ? null : s);
                break;
            case Role.WhereScalar:
                // A weak Python int first, then the other Python scalars (float, complex, bool, str), NumPy scalars (float16,
                // char), a strong 0-d int64 array, a decimal and a C# array (both through np.asanyarray), and null (refused).
                Add("7", () => 7);
                Add("2.5", () => 2.5);
                Add("c128", () => new Complex(1, -2));
                Add("true", () => true);
                Add("ab", () => "ab");
                Add("f16", () => (Half)1.5);
                Add("char", () => 'z');
                Add("0d", () => NDArray.Scalar(3L));
                Add("dec", () => 2.5m);
                Add("arr", () => new[] { 1.0, -2.0 });
                Add("null", () => null);
                break;
            case Role.VanderN:
                // None (a square matrix) first, then no column, one, three, more columns than any pool operand has rows, and
                // a negative count (the allocation's "negative dimensions are not allowed").
                Add("null", () => null);
                foreach (var nCols in new[] { 0, 1, 3, 9, -1 })
                    Add(nCols.ToString(CultureInfo.InvariantCulture), () => (int?)nCols);
                break;
            case Role.CtorShape:
                // Built from the shape label in Call (TryMakeCtorShape); the factory here is never called.
                Add("shape", () => null);
                break;
            case Role.CtorFill:
                foreach (var label in CtorFillLabels)
                    Add(label, () => MakeCtorFill(label));
                break;
            case Role.CtorShapeArg:
                // A _like form's shape=: its default first when it has one (the prototype's own shape, NumPy's shape=None),
                // then a matrix, a vector of the same size, 0-d, an empty vector, a matrix of other dims and a negative dim
                // (NumPy's "negative dimensions are not allowed"). A required one starts at the matrix.
                if (p.HasDefaultValue)
                    Add("default", () => p.DefaultValue is Shape given ? given : default(Shape));
                foreach (var dims in new[] { new long[] { 2, 3 }, new long[] { 6 }, new long[0], new long[] { 0 }, new long[] { 3, 2 }, new long[] { -1 } })
                    Add("(" + string.Join(",", dims) + (dims.Length == 1 ? ",)" : ")"), () => new Shape(dims));
                break;
            case Role.EyeM:
                // None (a square matrix) first, then no column, fewer and more columns than N's shapes have, and a negative
                // count (the allocation's "negative dimensions are not allowed").
                Add("null", () => null);
                foreach (var cols in new[] { 0, 2, 5, -1 })
                    Add(cols.ToString(CultureInfo.InvariantCulture), () => (int?)cols);
                break;
            case Role.EyeK:
                // The main diagonal first, then one above and below, beyond either corner of every shape, and far past it.
                foreach (var k in new[] { 0, 1, -1, 3, -3, 100 })
                    Add(k.ToString(CultureInfo.InvariantCulture), () => k);
                break;
            case Role.CtorOrder:
                // A creation function's default C first, then F, the A and K NumPy's creation functions refuse ("only 'C' or
                // 'F' order is permitted"), a lower-case spelling and a letter NumPy refuses first.
                foreach (var o in new[] { 'C', 'F', 'A', 'K', 'c', 'Z' })
                    Add(o.ToString(), () => o);
                break;
            case Role.NdBuffer:
                // No buffer first (np.ndarray allocates), then buffers it lays the array over: the bytes of 0..47 (48 bytes),
                // too few bytes, a C-contiguous float64 array, a transposed (F-contiguous) one, a strided view (NumPy's
                // "ndarray is not contiguous"), read-only bytes, and a str (no buffer: NumPy's TypeError).
                Add("null", () => null);
                Add("bytes48", () => Enumerable.Range(0, 48).Select(b => (byte)b).ToArray());
                Add("bytes4", () => new byte[] { 1, 2, 3, 4 });
                Add("f8_c", () => np.arange(6.0));
                Add("f8_f", () => np.arange(6.0).reshape(2, 3).T);
                Add("f8_strided", () => np.arange(12.0)["::2"]);
                Add("rom48", () => (ReadOnlyMemory<byte>)Enumerable.Range(0, 48).Select(b => (byte)b).ToArray());
                Add("str", () => "abc");
                break;
            case Role.NdOffset:
                // No offset first, then a whole float64 element, a misaligned byte offset, a negative offset and one past
                // any buffer above.
                foreach (var off in new[] { 0L, 8L, 3L, -8L, 1000L })
                    Add(off.ToString(CultureInfo.InvariantCulture), () => off);
                break;
            case Role.NdStrides:
                // None (C strides) first, then explicit C and F byte strides of a (2, 3) float64 layout, a one-axis stride, a
                // negative stride, a stride that splits an element, and a stride list of the wrong length.
                Add("null", () => null);
                foreach (var s in new[] { new long[] { 24, 8 }, new long[] { 8, 16 }, new long[] { 8 }, new long[] { -8 }, new long[] { 4 }, new long[] { 8, 8, 8 } })
                    Add("[" + string.Join(",", s) + "]", () => s);
                break;
            case Role.BufferLength:
                // Built from the buffer in Call (MakeBufferLength); the factory here is never called.
                Add(BufferLengthLabels[0], () => null);
                break;
            case Role.IoSep:
            {
                // The separator, by a label without a comma (the case id's variant list is comma-separated): np.fromstring's
                // base is ',' (its default, null, is the removed binary mode, every line of which would be the same
                // refusal), np.fromfile's and NDArray.tofile's their default ''. Then the others NumPy reads: the empty
                // separator (binary), a space (any whitespace run), ', ', a newline, null (NumPy's None: a TypeError for
                // np.fromfile and tofile; np.fromstring's omitted sep), one that never matches, and a NUL ("embedded null
                // character").
                var seps = new (string Label, string Value)[]
                {
                    ("comma", ","), ("empty", ""), ("space", " "), ("comma_space", ", "), ("newline", "\n"),
                    ("null", null), ("x", "x"), ("nul", "\0"),
                };
                string baseLabel = p.Member.Name == "fromstring" ? "comma" : "empty";
                foreach (var (label, value) in seps.Where(sp => sp.Label == baseLabel).Concat(seps.Where(sp => sp.Label != baseLabel)))
                    Add(label, () => value);
                break;
            }
            case Role.ToFileFormat:
                // NDArray.tofile's format: its default '%s' first, then the conversions NumPy's % operator applies to each
                // item, a literal with no conversion, two conversions, a lone '%' and null.
                foreach (var fmt in new[] { "%s", "%d", "%.3f", "%e", "%x", "%5.1f", "abc", "%s%s", "%", "null" })
                {
                    string value = fmt == "null" ? null : fmt;
                    Add(fmt, () => value);
                }
                break;
            case Role.SinkStream:
            case Role.SinkFile:
            case Role.SinkWriter:
                // Built per call in Call (a fresh MemoryStream, a temporary file name, a fresh StringWriter); the factory here
                // is never called.
                Add("sink", () => null);
                break;
            case Role.SaveTxtFmts:
                // Built from the operand in Call (MakeSaveTxtFmts); the factory here is never called.
                Add(SaveTxtFmtsLabels[0], () => null);
                break;
            case Role.TextIOArg:
                foreach (var (label, value) in TextIOArgVariants(p))
                    Add(label, () => value);
                break;
            case Role.BufferCount:
                // Every element of the buffer first, then none, one, a few, more than any pool operand's bytes hold, and
                // another negative count (NumPy reads every negative count as "all").
                foreach (var c in new[] { -1L, 0L, 1L, 3L, 100L, -2L })
                    Add(c.ToString(CultureInfo.InvariantCulture), () => c);
                break;
            case Role.BufferOffset:
                // No offset first, then byte offsets that keep and break each element size's alignment, a negative offset and
                // one past every pool operand's bytes.
                foreach (var off in new[] { 0L, 1L, 2L, 4L, 8L, 16L, -1L, 1000L })
                    Add(off.ToString(CultureInfo.InvariantCulture), () => off);
                break;
            case Role.BufferDType:
                // The default (null: float64) first, then numeric descriptors of every item size, two non-native ones (the
                // bytes read big-endian), the fixed-width strings (one-byte S3, UCS-4 U2 and a non-native U2, NumPy's 'c', which
                // is S1), the unsized S NumPy refuses, and StringDType, whose cells hold references.
                Add("null", () => null);
                foreach (var d in new[] { "float64", "int32", "int8", "uint16", "bool", "float16", "complex128", ">i4", ">f8", "S3", "U2", ">U2", "c", "S", "T" })
                    Add(d, () => np.dtype(d));
                break;
            case Role.BufferTypeCode:
                // The required type code: float64 first, then the other storage lanes (decimal and char are NumSharp's), the
                // unsized String lane (NumPy's np.str_, refused) and Empty (read as float64).
                foreach (var c in new[]
                         {
                             NPTypeCode.Double, NPTypeCode.Int32, NPTypeCode.SByte, NPTypeCode.UInt16, NPTypeCode.Boolean,
                             NPTypeCode.Half, NPTypeCode.Complex, NPTypeCode.Int64, NPTypeCode.Decimal, NPTypeCode.Char,
                             NPTypeCode.String, NPTypeCode.Empty,
                         })
                    Add(c.ToString(), () => c);
                break;
            case Role.BufferDTypeName:
                // The required dtype name: float64 first, then sized codes under each byte-order prefix, one-letter codes, the
                // complex64 code NumSharp refuses, a NumPy name, the fixed-width strings and NumPy's 'c', StringDType, a name
                // nothing parses, the empty name and null.
                foreach (var d in new[] { "float64", "<f8", ">f8", "=i4", ">i4", "|u1", "b1", "?", ">u2", "f2", ">f2", "c16", ">c16", "c8", "int32", "S3", "<U2", ">U2", "c", "T", "bogus", "" })
                    Add(d, () => d);
                Add("null", () => null);
                break;
            case Role.CoerceSize:
            {
                // The number of items np.array reads from an IEnumerable: a prefix first, then none, more than a pool
                // operand holds (the rest of the array stays as allocated) and a negative count.
                bool isLong = p.ParameterType == typeof(long);
                foreach (var count in new[] { 3, 0, 100, -1 })
                    Add(count.ToString(CultureInfo.InvariantCulture), () => isLong ? (object)(long)count : count);
                break;
            }
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
            case Role.CastDType:
                // NDArray.astype's dtype: float64 first (the base), then every class a cast reaches (CastDTypeLabels).
                foreach (var d in CastDTypeLabels)
                    Add(d, () => MakeCastDType(d));
                break;
            case Role.CastCasting:
                // NDArray.astype's casting rule: its default first (CastCastingLabels' 'unsafe'), then the rest.
                foreach (var c in CastCastingLabels)
                    Add(c, () => c);
                break;
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
            case Role.PolyintK:
                // None first (zeros), then a 0-d constant, one value (broadcast to every order), five (enough for every
                // order varied), two (too few past order 2: "k must be a scalar or a rank-1 array of length 1 or >m."),
                // integers, and text.
                Add("null", () => null);
                Add("zero_d", () => np.array(2.5));
                Add("one", () => np.array(new[] { 2.5 }));
                Add("five", () => np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }));
                Add("two", () => np.array(new[] { 1.0, 2.0 }));
                Add("int", () => np.array(new[] { 1, 2, 3, 4, 5 }));
                Add("text", () => np.array(new[] { "a", "b", "c", "d", "e" }));
                break;
            case Role.PolyintKDouble:
                Add("2.5", () => 2.5);
                Add("-1.5", () => -1.5);
                Add("nan", () => double.NaN);
                break;
            case Role.InterpXp:
                // An increasing float64 ramp 0..5 first (the pool's values fall inside and outside it), the ramp in other
                // dtypes and layouts (strided, reversed), a repeated point, decreasing and NaN points (NumPy does not check
                // monotonicity: its loop answers them deterministically), complex and text points (the safe-cast refusals,
                // or text parsed under a period), and the forms NumPy refuses (2-D, 0-d, empty, a length other than fp's).
                Add("ramp", () => np.array(new[] { 0.0, 1, 2, 3, 4, 5 }));
                Add("int64", () => np.array(new long[] { 0, 1, 2, 3, 4, 5 }));
                Add("float32", () => np.array(new float[] { 0, 1, 2, 3, 4, 5 }));
                Add("strided", () => np.array(new[] { 0.0, 9, 1, 9, 2, 9, 3, 9, 4, 9, 5, 9 })["::2"]);
                Add("reversed", () => np.array(new[] { 5.0, 4, 3, 2, 1, 0 })["::-1"]);
                Add("dup", () => np.array(new[] { 0.0, 1, 1, 2, 4, 5 }));
                Add("desc", () => np.array(new[] { 5.0, 4, 3, 2, 1, 0 }));
                Add("nan", () => np.array(new[] { 0.0, 1, double.NaN, 3, 4, 5 }));
                Add("complex", () => np.array(new[] { new Complex(0, 0), 1, 2, 3, 4, 5 }));
                Add("text", () => np.array(new[] { "0", "1", "2", "3", "4", "5" }));
                Add("2d", () => np.array(new[] { 0.0, 1, 2, 3, 4, 5 }).reshape(new Shape(2, 3)));
                Add("0d", () => np.array(2.0));
                Add("empty", () => np.empty(new Shape(0), np.float64));
                Add("len5", () => np.array(new[] { 0.0, 1, 2, 3, 4 }));
                break;
            case Role.InterpFp:
                // Real float64 values first, then other real dtypes, complex values (NumPy's complex loop), NaN values,
                // text, strided and reversed views, and the forms NumPy refuses (2-D, 0-d, empty, a length other than
                // xp's).
                Add("ramp", () => np.array(new[] { 10.0, 20, 15, 5, 0, -5 }));
                Add("int64", () => np.array(new long[] { 10, 20, 15, 5, 0, -5 }));
                Add("float32", () => np.array(new float[] { 10, 20, 15, 5, 0, -5 }));
                Add("float16", () => np.array(new[] { 10.0, 20, 15, 5, 0, -5 }).astype(np.float16));
                Add("bool", () => np.array(new[] { true, false, true, true, false, true }));
                Add("complex", () => np.array(new[]
                    { new Complex(1, 1), new Complex(2, -1), new Complex(0, 3), 4, new Complex(-1, 0.5), new Complex(5, 5) }));
                Add("nan", () => np.array(new[] { 10.0, double.NaN, 15, 5, double.PositiveInfinity, -5 }));
                Add("text", () => np.array(new[] { "10", "20", "15", "5", "0", "-5" }));
                Add("strided", () => np.array(new[] { 10.0, 0, 20, 0, 15, 0, 5, 0, 0, 0, -5, 0 })["::2"]);
                Add("reversed", () => np.array(new[] { -5.0, 0, 5, 15, 20, 10 })["::-1"]);
                Add("2d", () => np.array(new[] { 10.0, 20, 15, 5, 0, -5 }).reshape(new Shape(2, 3)));
                Add("0d", () => np.array(2.0));
                Add("empty", () => np.empty(new Shape(0), np.float64));
                Add("len5", () => np.array(new[] { 10.0, 20, 15, 5, 0 }));
                break;
            case Role.InterpFill:
                // None first (fp[0] / fp[-1]), then a number, NaN and an infinity.
                Add("null", () => (double?)null);
                Add("-1.5", () => (double?)-1.5);
                Add("nan", () => (double?)double.NaN);
                Add("inf", () => (double?)double.PositiveInfinity);
                break;
            case Role.InterpFillComplex:
                // None first, then a complex value and a complex with a zero imaginary part (NumPy's real loop refuses
                // both: PyFloat_AsDouble takes no Python complex).
                Add("null", () => (Complex?)null);
                Add("(1+2j)", () => (Complex?)new Complex(1, 2));
                Add("(3+0j)", () => (Complex?)new Complex(3, 0));
                break;
            case Role.FftN:
                // None first (the input's length; irfft / hfft's 2 * (m - 1)), then lengths that crop and zero pad, the odd
                // and even lengths the real transforms tell apart (rfft_n_odd / rfft_n_even, a Nyquist term), a prime past
                // the pool's lengths (pocketfft's generic radix), and the lengths NumPy refuses.
                Add("null", () => (int?)null);
                foreach (var v in new[] { 1, 2, 3, 4, 5, 8, 11, 0, -1 })
                    Add(v.ToString(CultureInfo.InvariantCulture), () => (int?)v);
                break;
            case Role.FftNorm:
                // None first, then NumPy's three names (forward / inverse swap them) and a near miss NumPy refuses (its text
                // differs between the forward and the inverse transforms).
                Add("null", () => null);
                foreach (var v in new[] { "backward", "ortho", "forward", "Ortho" })
                    Add(v, () => v);
                break;
            case Role.FftS:
                // None first (the input's shape along the axes), then one and two lengths, -1 (the input's length), an empty
                // and a zero length, more lengths than the operand has axes, and lengths that crop and pad.
                Add("null", () => null);
                foreach (var a in new[]
                         {
                             new[] { 4 }, new[] { -1 }, new[] { 2, 3 }, new[] { 3, -1 }, Array.Empty<int>(), new[] { 0 },
                             new[] { 3, 3, 3 }, new[] { 5, 1 },
                         })
                    Add("[" + string.Join(",", a) + "]", () => a.ToArray());
                break;
            case Role.FftFreqN:
                // fftfreq / rfftfreq's required window length: its C# type decides (a double is NumPy's "n should be an
                // integer"): an even and an odd length, one, zero (float division by zero) and a negative one (fftfreq's
                // "negative dimensions", rfftfreq's empty result).
                if (p.ParameterType == typeof(double))
                {
                    Add("8.0", () => 8.0);
                    Add("2.5", () => 2.5);
                }
                else
                    foreach (var v in new long[] { 8, 9, 1, 0, -3 })
                        Add(v.ToString(CultureInfo.InvariantCulture),
                            p.ParameterType == typeof(int) ? () => (object)(int)v : () => (object)v);
                break;
            case Role.FftD:
                // The default first, then a fraction, a negative spacing and zero (float division by zero).
                foreach (var v in new[] { 1.0, 0.5, -2.0, 0.0 })
                    Add(v.ToString("R", CultureInfo.InvariantCulture), () => v);
                break;
            case Role.FftDevice:
                // None first, NumPy's one device, and one it refuses.
                Add("null", () => null);
                Add("cpu", () => "cpu");
                Add("cuda", () => "cuda");
                break;
            case Role.NanToNumFill:
                // None first (NumPy's 0.0 / the dtype's largest and smallest finite values), then the scalar kinds the
                // fused path converts (a double, an int, a bool, a float16, NaN as a fill, a value past float16's range),
                // the array fills that take the copyto composition (a 0-d array, a row, a C# double[] of a row's length,
                // an int32 array), and text NumPy refuses.
                foreach (var (label, value) in new (string, Func<object>)[]
                         {
                             ("null", () => null), ("1.5", () => 1.5), ("-7", () => -7), ("true", () => true),
                             ("half2.5", () => (Half)2.5), ("nan", () => double.NaN), ("1e6", () => 1e6),
                             ("nd0d", () => np.array(9.5)), ("ndrow", () => np.array(new[] { 1.0, 2, 3, 4 })),
                             ("dbl[4]", () => new[] { 1.0, 2, 3, 4 }), ("i32row", () => np.array(new[] { 1, 2, 3, 4 })),
                             ("'x'", () => "x"),
                         })
                    Add(label, value);
                break;
            case Role.RealIfCloseTol:
                // The default 100 (a multiple of eps) first, then a larger multiple, the 1.0 boundary (absolute from
                // there), a tiny absolute bound, zero and a negative bound.
                foreach (var v in new[] { 100.0, 1e6, 1.0, 1e-20, 0.0, -1.0 })
                    Add(v.ToString("R", CultureInfo.InvariantCulture), () => v);
                break;
            case Role.UnravelShape:
                // A product of 90,000 first (every non-negative index of the pool's operands is in range, the uint8 /
                // uint16 wraps of -1 included), then small ones (most of the pool's indices out of range: the error that
                // names the first one in NumPy's iteration order), the empty shape (only 0-d indices), a zero dimension,
                // NumPy's -1 overflow sentinel, two negative dimensions (a positive product: the divmod kernel), one negative
                // dimension (no valid index), an overflowing product, seventeen dimensions (the reciprocal kernel's heap
                // table) and a single one.
                foreach (var dims in new[]
                         {
                             new[] { 300, 300 }, new[] { 3, 4 }, new[] { 2, 3, 4 }, Array.Empty<int>(), new[] { 0, 3 },
                             new[] { -1 }, new[] { -30, -30 }, new[] { -2, 3 }, new[] { int.MaxValue, int.MaxValue, 3 },
                             Enumerable.Repeat(2, 17).ToArray(), new[] { 1 },
                         })
                    Add("[" + string.Join(",", dims) + "]", () => (int[])dims.Clone());
                break;
            case Role.UnravelOrder:
                // C (the default) first, then F, both in lower case, the A and K NumPy refuses after its other checks, and a
                // letter its argument parse refuses first.
                foreach (var o in new[] { 'C', 'F', 'c', 'f', 'A', 'K', 'Z' })
                    Add(o.ToString(), () => o);
                break;
            case Role.UnravelFlat:
                // An index inside (90,000 =) 300 x 300 first, then the bounds of that shape on both sides, a negative index
                // and long.MaxValue.
                foreach (var v in new[] { 5L, 0L, 299L, 89999L, 90000L, -1L, long.MaxValue })
                    Add(v.ToString(CultureInfo.InvariantCulture), () => v);
                break;
            case Role.RavelDims:
                // (70000, 70000) first: every value of the unsigned pools fits (a uint8 / uint16 / char wrap of -1
                // included), so the raise mode refuses only the negative ones. Then small dimensions (both bounds
                // reached in wrap and clip), a zero dimension (refused once there is a coordinate to fold), the empty
                // shape (the count check; no coordinate gives the flat index 0), a product that overflows int64
                // (refused before the count is checked), negative dimensions (no overflow: their ravel strides are
                // negative, the raise mode refuses every coordinate, wrap and clip fold them), ones, three dimensions
                // and a single one.
                foreach (var dims in new[]
                         {
                             new[] { 70000, 70000 }, new[] { 3, 4 }, new[] { 0, 3 }, Array.Empty<int>(),
                             new[] { int.MaxValue, int.MaxValue, 3 }, new[] { 3, -2 }, new[] { -2, 3 },
                             new[] { -30, -30 }, new[] { 1, 1 }, new[] { 2, 3, 4 }, new[] { 70000 },
                         })
                    Add("[" + string.Join(",", dims) + "]", () => (int[])dims.Clone());
                break;
            case Role.RavelMode:
                // raise (the default) first, then wrap and clip, a misspelling of a mode (its first letter selects one:
                // NumPy's "Use one of ..." text), a longer word that starts like one, an unknown word and the empty
                // string (the generic text with the argument's repr), and None (raise).
                foreach (var mode in new[] { "raise", "wrap", "clip", "Raise", "clipped", "bogus", "", null })
                    Add(mode is null ? "null" : "'" + mode + "'", () => mode);
                break;
            case Role.RavelModes:
                // One mode per dimension of the base dims, raise first; then mixed modes both ways, None (raise
                // everywhere), the wrong lengths (one, none, three), a bad entry after a good one, a None entry and a
                // misspelled entry.
                foreach (var modes in new[]
                         {
                             new[] { "raise", "raise" }, new[] { "wrap", "clip" }, new[] { "clip", "wrap" }, null,
                             new[] { "raise" }, Array.Empty<string>(), new[] { "raise", "raise", "raise" },
                             new[] { "wrap", "bogus" }, new[] { null, "clip" }, new[] { "Wrap", "clip" },
                         })
                    Add(modes is null ? "null" : "[" + string.Join(",", modes.Select(s => s ?? "null")) + "]",
                        () => (string[])modes?.Clone());
                break;
            case Role.RavelCoords:
                // Built from the operand (MakeRavelCoords); the factory here is never called. The base is the operand
                // twice.
                Add(RavelCoordsVariantLabels[0], () => null);
                break;
            case Role.PackBitOrder:
                // big (the default) first, then little, NumPy's prefix forms (packbits reads the "little" / "big" prefix,
                // unpackbits the first character only, so "l" and "b" split the two), the empty string, None and a
                // capitalized name both refuse.
                foreach (var o in new[] { "big", "little", "littlefoo", "bigx", "l", "b", "", null, "Big" })
                    Add(o is null ? "null" : "'" + o + "'", () => o);
                break;
            case Role.UnpackCount:
                // None (every bit) first, then zero, a count that truncates, one byte's worth, counts that pad with zeros,
                // negative trims and a trim longer than the bits.
                Add("null", () => (long?)null);
                foreach (var v in new[] { 0L, 3L, 8L, 13L, 100L, -1L, -8L, -100L })
                    Add(v.ToString(CultureInfo.InvariantCulture), () => (long?)v);
                break;
            case Role.BincountMinLength:
                // 0 (the default) first, then lengths below and above the pool's largest value, and a negative one.
                foreach (var v in new[] { 0, 1, 5, 20, -1 })
                    Add(v.ToString(CultureInfo.InvariantCulture), () => v);
                break;
            case Role.InterpPeriod:
                // None first, then periods smaller and larger than the points' span, a fraction, a negative one (NumPy takes
                // its absolute value), zero (refused) and NaN.
                Add("null", () => (double?)null);
                Add("4", () => (double?)4.0);
                Add("6", () => (double?)6.0);
                Add("2.5", () => (double?)2.5);
                Add("-6", () => (double?)-6.0);
                Add("0", () => (double?)0.0);
                Add("nan", () => (double?)double.NaN);
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
            case Role.InsertObjMask:
                // A mask of three first (the first axis of the (3, 4) pools; NumPy's length check refuses it against the
                // flattened axis=None operand), then four (their last axis), six, twelve and twenty-four (the 1-D, the
                // flattened (3, 4) and the flattened 3-D operands), all False, and an empty mask.
                foreach (var (name, mask) in new[]
                         {
                             ("[T,F,T]", new[] { true, false, true }),
                             ("[F,T,F,T]", new[] { false, true, false, true }),
                             ("alt6", Enumerable.Range(0, 6).Select(i => i % 2 == 0).ToArray()),
                             ("alt12", Enumerable.Range(0, 12).Select(i => i % 2 == 0).ToArray()),
                             ("alt24", Enumerable.Range(0, 24).Select(i => i % 3 == 0).ToArray()),
                             ("none12", new bool[12]),
                             ("[]", Array.Empty<bool>()),
                         })
                    Add(name, () => mask.ToArray());
                break;
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
            case Role.ContainsValue:
                // 0 first (most pools hold a zero, so the search finds it), then a value most do not hold, a double, a bool,
                // a string (a char array's character search; a text array's element), null (None), NaN (never equal), a
                // C# int[] (a Python list, compared element-wise and broadcast) and an NDArray of two values.
                foreach (var (label, value) in new (string, Func<object>)[]
                         {
                             ("0", () => 0), ("7", () => 7), ("1.5", () => 1.5), ("true", () => true), ("'a'", () => "a"),
                             ("null", () => null), ("nan", () => double.NaN), ("int[0,1]", () => new[] { 0, 1 }),
                             ("nd[0,1]", () => np.array(new long[] { 0, 1 })),
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
    ///     The layouts np.trim_zeros adds to the general pool (phase A): operands with zero borders - a 1-D one with two
    ///     zeros at each end, a 2-D one inside a zero frame, a 3-D one inside a zero box - and an all-zero 2-D one (every
    ///     axis trims to nothing). The general pool's operands have no zero edge, so on them nothing is trimmed.
    /// </summary>
    private static readonly string[] TrimLayouts = { "zpad1d", "zpad2d", "zpad3d", "allzero2d" };

    /// <summary>The reduced pool of np.trim_zeros (phase B, where trim and axis vary): the zero-border operands, a plain
    ///     2-D one, a 0-d one (returned as it is) and an empty one.</summary>
    private static readonly string[] TrimReducedLayouts = { "zpad1d", "zpad2d", "zpad3d", "allzero2d", "c2d", "0d", "empty" };

    /// <summary>The small pool of np.trim_zeros (phase C).</summary>
    private static readonly string[] TrimPairLayouts = { "zpad1d", "zpad2d", "0d" };

    /// <summary>The reduced pool of np.ix_ and np.r_ / np.c_ (phase B, where their sequences and keys vary): 1-D operands
    ///     first (np.ix_ takes only 1-D sequences, and the concatenators join a 1-D operand with the 1-D arrays their scalars
    ///     and slices become), then the general reduced pool.</summary>
    private static readonly string[] IndexTricksReducedLayouts = { "1d", "empty1d", "c2d", "f2d", "0d", "empty", "3d" };

    /// <summary>The small pool of np.ix_ and np.r_ / np.c_ (phase C).</summary>
    private static readonly string[] IndexTricksPairLayouts = { "1d", "c2d", "0d" };

    /// <summary>
    ///     The reduced pool of np.savetxt (phase B): the layouts it writes - a matrix, an F-ordered one, a vector and an
    ///     empty matrix. A 0-d or 3-D X is refused before any other parameter is read (<c>Expected 1D or 2D array</c>), so
    ///     phase A's line for those layouts is all they can tell.
    /// </summary>
    private static readonly string[] SaveTxtReducedLayouts = { "c2d", "f2d", "1d", "empty" };

    /// <summary>The small pool of np.savetxt (phase C): a matrix and a vector, the two shapes it writes.</summary>
    private static readonly string[] SaveTxtPairLayouts = { "c2d", "1d" };

    /// <summary>
    ///     The 1-D layouts np.diag / np.diagflat add to the general pool (phase A): a vector read every other element and one
    ///     read backwards, whose stride the diagonal scatter follows (the general pool's 1-D operands are contiguous).
    /// </summary>
    private static readonly string[] DiagExtraLayouts = { "strided1d", "reversed1d" };

    /// <summary>The reduced pool of np.diag / np.diagflat (phase B, where k varies): the 1-D operands np.diag builds a matrix
    ///     from first - contiguous, empty, strided and reversed - then the general reduced pool, whose 2-D operands np.diag
    ///     reads a diagonal of and np.diagflat ravels.</summary>
    private static readonly string[] DiagReducedLayouts = { "1d", "empty1d", "strided1d", "reversed1d", "c2d", "f2d", "0d", "empty", "3d" };

    /// <summary>The small pool of np.diag / np.diagflat (phase C).</summary>
    private static readonly string[] DiagPairLayouts = { "1d", "c2d", "0d" };

    /// <summary>
    ///     The 1-D layouts np.bincount adds to the general pool (phase A): the plain prefix's magnitudes, contiguous, every
    ///     other element and reversed. The general pool's 1-D operands hold negative values, which a signed dtype refuses
    ///     at once and an unsigned one wraps into a bin count of up to 2^32 (an array of 32 GB), so only these reach the
    ///     counting and weighted kernels for every integer dtype.
    /// </summary>
    private static readonly string[] BincountLayouts = { "nonneg1d", "nonneg1d_strided", "nonneg1d_reversed" };

    /// <summary>The reduced pool of np.bincount (phase B, where weights and minlength vary): the magnitude layouts, an
    ///     empty 1-D operand (the early zeros), a 0-d and a 2-D one (NumPy's two depth errors). The general pool's 1-D
    ///     operand is left out: its uint32 form would size a 32 GB result for every variant.</summary>
    private static readonly string[] BincountReducedLayouts = { "nonneg1d", "nonneg1d_strided", "nonneg1d_reversed", "empty1d", "0d", "c2d" };

    /// <summary>The small pool of np.bincount (phase C).</summary>
    private static readonly string[] BincountPairLayouts = { "nonneg1d", "empty1d", "0d" };

    /// <summary>The reduced pool of np.choose / NDArray.choose (phase B): a non-negative index array (most of its
    ///     indices name a choice, so the selection itself runs) beside the general 2-D, 0-d, empty and 3-D operands.</summary>
    private static readonly string[] ChooseReducedLayouts = { "nonneg1d", "c2d", "0d", "empty", "3d" };

    /// <summary>The small pool of np.choose / NDArray.choose (phase C).</summary>
    private static readonly string[] ChoosePairLayouts = { "nonneg1d", "c2d", "0d" };

    /// <summary>
    ///     The operand layouts of the three phases for an overload: the general pools; for np.searchsorted /
    ///     NDArray.searchsorted, pools of 1-D haystacks (every other operand of the general pools fails NumPy's 1-D check
    ///     before anything else is read, so the varied keys, sides and sorters would never reach the search); for
    ///     np.convolve / np.correlate and their NDArray members, the same 1-D layouts and a 24-element one (a general pool's
    ///     2-D operand fails their depth check first), and for np.vander too (its 1-D check comes first); for np.trim_zeros,
    ///     the general pool and operands with zero borders (<see cref="TrimLayouts"/>); for np.bincount, the general pool
    ///     and 1-D operands of non-negative
    ///     values (<see cref="BincountLayouts"/>); for np.choose and NDArray.choose, the general pool and the same
    ///     non-negative 1-D operands (index arrays whose indices mostly name a choice); for np.linalg, pools of square
    ///     matrices, stacks and rectangles (<see cref="LinalgLayouts"/>); for np.savetxt, the general pool in phase A and
    ///     the 1-D and 2-D layouts it writes when another parameter varies (<see cref="SaveTxtReducedLayouts"/>); for np.ix_
    ///     and np.r_ / np.c_, the general pool with 1-D operands first in phase B; for np.diag and np.diagflat, the general
    ///     pool and a strided and a reversed vector, 1-D operands first in phase B (<see cref="DiagReducedLayouts"/>).
    /// </summary>
    /// <param name="m">The overload.</param>
    /// <returns>The layouts of phase A, phase B and phase C.</returns>
    private static (string[] Full, string[] Reduced, string[] Pair) PoolsFor(MethodInfo m) =>
        m.Name == "searchsorted"
            ? (Layouts.Concat(SortedLayouts).ToArray(), SortedReducedLayouts, SortedPairLayouts)
            : (m.Name is "convolve" or "correlate" && (m.DeclaringType == typeof(np) || m.DeclaringType == typeof(NDArray)))
              || (m.Name == "vander" && m.DeclaringType == typeof(np))
                ? (Layouts.Concat(SortedLayouts).Append("long1d").ToArray(), SlidingReducedLayouts, SlidingPairLayouts)
                : m.Name == "trim_zeros" && m.DeclaringType == typeof(np)
                    ? (Layouts.Concat(TrimLayouts).ToArray(), TrimReducedLayouts, TrimPairLayouts)
                : m.Name == "choose" && (m.DeclaringType == typeof(np) || m.DeclaringType == typeof(NDArray))
                    ? (Layouts.Concat(BincountLayouts).ToArray(), ChooseReducedLayouts, ChoosePairLayouts)
                    : m.Name == "bincount" && m.DeclaringType == typeof(np)
                        ? (Layouts.Concat(BincountLayouts).ToArray(), BincountReducedLayouts, BincountPairLayouts)
                        : m.DeclaringType == typeof(np.linalg)
                            ? (LinalgLayouts, LinalgReducedLayouts, LinalgPairLayouts)
                            : (m.Name == "ix_" && m.DeclaringType == typeof(np))
                              || (m.Name == "get_Item" && m.DeclaringType == typeof(np.AxisConcatenator))
                                ? (Layouts, IndexTricksReducedLayouts, IndexTricksPairLayouts)
                                : m.Name == "savetxt" && m.DeclaringType == typeof(np)
                                    ? (Layouts, SaveTxtReducedLayouts, SaveTxtPairLayouts)
                                : m.DeclaringType == typeof(np) && m.Name is "diag" or "diagflat"
                                    ? (Layouts.Concat(DiagExtraLayouts).ToArray(), DiagReducedLayouts, DiagPairLayouts)
                                : m.DeclaringType == typeof(NDArray)
                                  && m.Name is "get_Item" or "set_Item" or "__setitem__" or "GetIndices" or "SetIndices"
                                    ? (Layouts, IndexerReducedLayouts, IndexerPairLayouts)
                                : IsCastMember(m)
                                    ? (Layouts, CastReducedLayouts, CastPairLayouts)
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
            // A spacing generator's operand: its (start, stop) pair, built per call from the pair label (SpacePairLabels);
            // the stop parameter is the pair's second value, so it is not varied on its own.
            if (IsSpacing(m) && ps[i].Name == "start")
            {
                roles[i] = Role.SpacePair;
                sawArray = true;
                continue;
            }
            if (IsSpacing(m) && ps[i].Name == "stop")
            {
                roles[i] = Role.SpaceStop;
                continue;
            }
            // A constructor's operand when it takes no array: its shape (np.eye's N, np.identity's n), built per call from
            // the shape label in the parameter's C# form (TryMakeCtorShape). A _like form's prototype is the general operand.
            // np.indices / np.indices_sparse's dimensions are such a shape too.
            if (i == 0 && !instance && (IsCtor(m) || IsIndices(m)) && ps[0].ParameterType != typeof(NDArray))
            {
                roles[i] = Role.CtorShape;
                sawArray = true;
                continue;
            }
            // np.mgrid / np.ogrid's key (the params object[] of their indexer): the operand, built per call from its label
            // (MakeGridKey).
            if (i == 0 && !instance && IsGridIndexer(m))
            {
                roles[i] = Role.GridKey;
                sawArray = true;
                continue;
            }
            // np.ravel_multi_index's coordinate list carries the pool operand (MakeRavelCoords); its scalar overload's
            // coordinate tuple is that overload's operand, built per call from its label (MakeRavelScalarCoords).
            if (i == 0 && !instance && IsMultiarrayIndexing(ps[0]) && ps[0].ParameterType == typeof(NDArray[])
                && ps[0].Name == "multi_index")
            {
                roles[i] = Role.RavelCoords;
                sawArray = true;
                continue;
            }
            if (i == 0 && !instance && IsMultiarrayIndexing(ps[0]) && ps[0].ParameterType == typeof(long[])
                && ps[0].Name == "coords")
            {
                roles[i] = Role.RavelScalarCoords;
                sawArray = true;
                continue;
            }
            // The coercion family's host operand: its first parameter when that is not an NDArray (np.array's C# arrays,
            // strs and scalars, np.asanyarray's object, NDArray.Scalar's value, a conversion operator's C# source). It is
            // built from the pool operand (TryMakeHostOperand), and an NDArray parameter after it (like=) is no operand.
            if (i == 0 && !instance && IsCoercion(m) && ps[0].ParameterType != typeof(NDArray))
            {
                roles[i] = Role.HostOperand;
                sawArray = true;
                continue;
            }
            // np.frombuffer's operand: its buffer, every overload's first parameter, built per call from the pool operand's
            // bytes in the parameter's C# form (TryMakeBufferOperand). A ReadOnlySpan<byte> cannot be passed by reflection,
            // so that overload is reported as SKIP.
            if (i == 0 && !instance && IsFromBuffer(m))
            {
                roles[i] = ps[0].ParameterType.IsByRefLike ? Role.Unsupported : Role.BufferOperand;
                sawArray = true;
                continue;
            }
            // np.fromstring's text and np.fromfile's file, built per call from the pool operand (TryMakeTextSource,
            // TryMakeFileSource).
            if (i == 0 && !instance && IsFileIO(m) && m.Name is "fromstring" or "fromfile")
            {
                roles[i] = m.Name == "fromstring" ? Role.TextSource : Role.FileSource;
                sawArray = true;
                continue;
            }
            // np.loadtxt's source (a file name, a byte stream, a TextReader, a list of lines), built per call from the pool
            // operand (TryMakeLoadTxtSource).
            if (i == 0 && !instance && IsTextIO(m) && m.Name == "loadtxt")
            {
                roles[i] = Role.LoadTxtSource;
                sawArray = true;
                continue;
            }
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
            // np.evaluate's expression carries the operand (MakeEvalExpr): an embedded NDExpr.Arr leaf, or NDExpr.Input(0)
            // when the overload takes a positional operand list, which then holds the operand alone.
            if (ps[i].ParameterType == typeof(NDExpr) && ps[i].Name == "expr" && IsEvaluate(ps[i]))
            {
                roles[i] = ps.Any(q => q.Name == "operands" && q.ParameterType == typeof(NDArray[]))
                    ? Role.EvalExprPositional
                    : Role.EvalExpr;
                sawArray = true;
                continue;
            }
            if (ps[i].ParameterType == typeof(NDArray[]) && ps[i].Name == "operands" && IsEvaluate(ps[i]))
            {
                roles[i] = Role.EvalOperands;
                sawArray = true;
                continue;
            }
            // np.einsum's operand list carries the operand too (MakeEinsumOps).
            if (!sawArray && ps[i].ParameterType == typeof(NDArray[]) && ps[i].Name == "operands" && IsEinsum(ps[i]))
            {
                roles[i] = Role.EinsumOps;
                sawArray = true;
                continue;
            }
            // np.compress's condition comes before the array it selects from: a is the operand, the condition is built
            // from it (MakeItemArg), so the condition must not take the operand's place.
            if (!sawArray && ps[i].Name == "condition" && ps[i].ParameterType == typeof(NDArray) && IsItemSelection(ps[i]))
            {
                roles[i] = Role.ItemArg;
                continue;
            }
            // np.lexsort's key list carries the operand too (MakeLexKeys).
            if (!sawArray && ps[i].ParameterType == typeof(NDArray[]) && ps[i].Name == "keys" && IsSortFamily(ps[i]))
            {
                roles[i] = Role.LexKeys;
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
        var variants = ps.Select((p, i) => roles[i] is Role.Operand or Role.HostOperand or Role.SpacePair or Role.CtorShape or Role.BufferOperand or Role.TextSource or Role.FileSource or Role.GridKey or Role.RavelScalarCoords ? null : Variants(p, roles[i])).ToArray();
        int operandIndex = Array.IndexOf(roles, Role.Operand);
        // A coercion overload's operand is its host operand, built per call from the pool operand.
        int hostIndex = Array.IndexOf(roles, Role.HostOperand);
        bool host = operandIndex < 0 && hostIndex >= 0;
        if (host)
            operandIndex = hostIndex;
        // A spacing generator's operand is its (start, stop) pair.
        int pairIndex = Array.IndexOf(roles, Role.SpacePair);
        bool spacing = operandIndex < 0 && pairIndex >= 0;
        if (spacing)
            operandIndex = pairIndex;
        // A constructor's operand is its shape.
        int shapeIndex = Array.IndexOf(roles, Role.CtorShape);
        bool ctor = operandIndex < 0 && shapeIndex >= 0;
        if (ctor)
            operandIndex = shapeIndex;
        // np.mgrid / np.ogrid's operand is their key.
        int keyIndex = Array.IndexOf(roles, Role.GridKey);
        bool gridKey = operandIndex < 0 && keyIndex >= 0;
        if (gridKey)
            operandIndex = keyIndex;
        // np.ravel_multi_index's scalar overload's operand is its coordinate tuple.
        int coordsIndex = Array.IndexOf(roles, Role.RavelScalarCoords);
        bool ravelScalar = operandIndex < 0 && coordsIndex >= 0;
        if (ravelScalar)
            operandIndex = coordsIndex;
        // np.frombuffer's operand is its buffer.
        int bufferIndex = Array.IndexOf(roles, Role.BufferOperand);
        bool buffer = operandIndex < 0 && bufferIndex >= 0;
        if (buffer)
            operandIndex = bufferIndex;
        // np.fromstring's is its text, np.fromfile's its file.
        int sourceIndex = Array.FindIndex(roles, r => r is Role.TextSource or Role.FileSource);
        bool source = operandIndex < 0 && sourceIndex >= 0;
        if (source)
            operandIndex = sourceIndex;
        // np.loadtxt's is its source.
        int loadIndex = Array.IndexOf(roles, Role.LoadTxtSource);
        bool loadSource = operandIndex < 0 && loadIndex >= 0;
        if (loadSource)
            operandIndex = loadIndex;
        var none = Array.Empty<(int, int)>();

        // The number of non-base variants per parameter (0 for the operand and for parameters kept at their default).
        int Count(int i) => roles[i] switch
        {
            Role.Operand => 0,
            Role.HostOperand => 0,
            Role.SpacePair => 0,
            Role.CtorShape => 0,
            Role.GridKey => 0,
            Role.RavelScalarCoords => 0,
            Role.RavelCoords => RavelCoordsVariantLabels.Length - 1,
            Role.BufferOperand => 0,
            Role.BufferLength => BufferLengthLabels.Length - 1,
            Role.TextSource or Role.FileSource => 0,
            Role.SinkStream or Role.SinkFile or Role.SinkWriter => 0,
            Role.LoadTxtSource => 0,
            Role.SaveTxtFmts => SaveTxtFmtsLabels.Length - 1,
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
            Role.EinsumSubscripts => EinsumSubscriptsVariantLabels.Length - 1,
            Role.EinsumOps => EinsumOpsVariantLabels.Length - 1,
            Role.LexKeys => LexKeysVariantLabels.Length - 1,
            Role.EvalExpr or Role.EvalExprPositional => EvalExprVariantLabels.Length - 1,
            Role.EvalOperands => 0,
            Role.ItemArg => ItemArgLabels(ps[i]).Length - 1,
            Role.AssignArg => AssignArgLabels(ps[i]).Length - 1,
            Role.SampleSeq or Role.BinsInts or Role.BinsArrays or Role.BinsObjects or Role.RangeArray => DdLabels(roles[i]).Length - 1,
            _ => variants[i].Count - 1,
        };

        // The operand layouts of the three phases: the general pools, or the 1-D pools of a search function (below).
        var (fullLayouts, reducedLayouts, pairLayouts) = PoolsFor(m);
        string[] dtypes = Dtypes, pairDtypes = PairDtypes;
        // NDArray.astype's rule checks depend on BOTH classes: its pairs (the dtype with the casting rule, the order, copy)
        // run over every pool dtype, not the general eight.
        if (IsCastMember(m))
            pairDtypes = dtypes;
        if (host)
        {
            // A host operand takes every form its parameter type admits ("layout:form", see HostForms), and a typed
            // parameter (T, T[], T[,], IEnumerable<T>) only the pool operands of its element type: the other dtypes have
            // no such C# value, so they would produce no line anyway.
            string[] forms = HostForms(ps[operandIndex]);
            string[] WithForms(string[] layouts) => layouts.SelectMany(l => forms.Select(f => l + ":" + f)).ToArray();
            fullLayouts = WithForms(fullLayouts);
            reducedLayouts = WithForms(reducedLayouts);
            pairLayouts = WithForms(pairLayouts);
            Type hostType = ps[operandIndex].ParameterType.IsByRef ? ps[operandIndex].ParameterType.GetElementType() : ps[operandIndex].ParameterType;
            if (PinnedHostElement(hostType) is { } pinned)
            {
                dtypes = Dtypes.Where(d => HostElementType(d) == pinned).ToArray();
                pairDtypes = dtypes;
            }
        }
        if (spacing)
        {
            // A generator takes no pool operand: its pools are (start, stop) pairs of the start parameter's type, one
            // stand-in dtype label ("pair"), the first five pairs for phase B and the first three for phase C.
            fullLayouts = SpacePairLabels(ps[operandIndex].ParameterType);
            reducedLayouts = fullLayouts.Take(5).ToArray();
            pairLayouts = fullLayouts.Take(3).ToArray();
            dtypes = new[] { "pair" };
            pairDtypes = dtypes;
        }
        if (ctor)
        {
            // A constructor takes no pool operand: its pools are shapes, one stand-in dtype label ("shape"), the first five
            // shapes for phase B and the first three for phase C. np.indices' grids are tiny and a dtype's arange decides on
            // the later ones (a bool arange of more than two elements, an int8 one that wraps), so its phase B takes them
            // all.
            bool grid = IsIndices(m);
            fullLayouts = grid ? IndicesShapeLabels : CtorShapeLabels;
            reducedLayouts = grid ? fullLayouts : fullLayouts.Take(5).ToArray();
            pairLayouts = fullLayouts.Take(3).ToArray();
            dtypes = new[] { "shape" };
            pairDtypes = dtypes;
        }
        if (gridKey)
        {
            // np.mgrid / np.ogrid take no pool operand: their pool is the keys, one stand-in dtype label ("key"). The key is
            // their only argument, so only phase A runs; the reduced and small pools are kept for symmetry.
            fullLayouts = GridKeyLabels;
            reducedLayouts = fullLayouts;
            pairLayouts = fullLayouts.Take(3).ToArray();
            dtypes = new[] { "key" };
            pairDtypes = dtypes;
        }
        if (ravelScalar)
        {
            // np.ravel_multi_index's scalar overload takes no pool operand: its pool is the coordinate tuples, one
            // stand-in dtype label ("coords"), the first five for phase B and the first three for phase C.
            fullLayouts = RavelScalarCoordsLabels;
            reducedLayouts = fullLayouts.Take(5).ToArray();
            pairLayouts = fullLayouts.Take(3).ToArray();
            dtypes = new[] { "coords" };
            pairDtypes = dtypes;
        }
        if (buffer)
        {
            // A buffer takes every form its parameter type admits ("layout:form", see BufferForms), and the generic
            // overload's typed array only the pool operands of its element type: the other dtypes have no such C# array.
            string[] forms = BufferForms(ps[operandIndex]);
            string[] WithBufferForms(string[] layouts) => layouts.SelectMany(l => forms.Select(f => l + ":" + f)).ToArray();
            fullLayouts = WithBufferForms(fullLayouts);
            reducedLayouts = WithBufferForms(reducedLayouts);
            pairLayouts = WithBufferForms(pairLayouts);
            if (m.IsGenericMethod)
            {
                Type element = ps[operandIndex].ParameterType.GetElementType();
                dtypes = Dtypes.Where(d => HostElementType(d) == element).ToArray();
                pairDtypes = dtypes;
            }
        }
        if (source)
        {
            // A text or a file takes every form its parameter type admits ("layout:form"). The text depends on the pool's
            // value SEQUENCE only (numbers, or the text pool's strings), so np.fromstring runs on one numeric and one text
            // pool; np.fromfile's binary file holds the operand's own bytes, so it runs on every dtype.
            string[] forms = roles[operandIndex] == Role.TextSource ? TextSourceForms(ps[operandIndex]) : FileSourceForms(ps[operandIndex]);
            string[] WithSourceForms(string[] layouts) => layouts.SelectMany(l => forms.Select(f => l + ":" + f)).ToArray();
            fullLayouts = WithSourceForms(fullLayouts);
            reducedLayouts = WithSourceForms(reducedLayouts);
            pairLayouts = WithSourceForms(pairLayouts);
            if (roles[operandIndex] == Role.TextSource)
            {
                dtypes = new[] { "float64", "U5" };
                pairDtypes = dtypes;
            }
        }
        if (loadSource)
        {
            // np.loadtxt's source takes every form its parameter type admits in phase A, and only data forms when another
            // parameter varies - the plain rows and the comma-separated ones in phase B, the plain rows in phase C (a form
            // built once, null or a missing file, would repeat one refusal for every variant). The text depends on the pool's
            // value SEQUENCE only, so it runs on one numeric and one text pool.
            string[] forms = LoadTxtSourceForms(ps[operandIndex]);
            fullLayouts = fullLayouts.SelectMany(l => forms.Select(f => l + ":" + f)).ToArray();
            reducedLayouts = reducedLayouts.SelectMany(l => new[] { l + ":rows", l + ":csv" }).ToArray();
            pairLayouts = pairLayouts.Select(l => l + ":rows").ToArray();
            dtypes = new[] { "float64", "U5" };
            pairDtypes = dtypes;
        }
        // A host operand form the pool operand does not have (Call returns null) produces no line; an overload whose host
        // parameter no pool operand fits (np.array<T>(T[,,,,]): no pool operand has rank 5) reports that once.
        int produced = 0;

        // phase A: every operand, every other parameter at its base
        foreach (var dt in dtypes)
            foreach (var layout in fullLayouts)
                if (Call(m, signature, instance, matrix, operandIndex, dt, layout, variants, roles, none) is { } lineA)
                {
                    produced++;
                    yield return lineA;
                }

        // phase B: one parameter varied at a time, on the reduced pool
        for (int p = 0; p < ps.Length; p++)
            for (int v = 1; v <= Count(p); v++)
                foreach (var dt in dtypes)
                    foreach (var layout in reducedLayouts)
                        if (Call(m, signature, instance, matrix, operandIndex, dt, layout, variants, roles, new[] { (p, v) }) is { } lineB)
                        {
                            produced++;
                            yield return lineB;
                        }

        // phase C: every pair of parameters varied together, on the small pool
        for (int p1 = 0; p1 < ps.Length; p1++)
            for (int p2 = p1 + 1; p2 < ps.Length; p2++)
                for (int v1 = 1; v1 <= Count(p1); v1++)
                    for (int v2 = 1; v2 <= Count(p2); v2++)
                        foreach (var dt in pairDtypes)
                            foreach (var layout in pairLayouts)
                                if (Call(m, signature, instance, matrix, operandIndex, dt, layout, variants, roles,
                                        new[] { (p1, v1), (p2, v2) }) is { } lineC)
                                {
                                    produced++;
                                    yield return lineC;
                                }

        if (produced == 0)
            yield return $"{signature}\tSKIP no pool operand has a host form this parameter takes";
    }

    /// <summary>
    ///     The text of the special numeric pool (<see cref="Special"/>) as Python writes each float (<c>repr</c>): the text
    ///     np.fromstring and np.fromfile read for a special layout of a numeric pool. The replay builds the same strings from
    ///     the same floats.
    /// </summary>
    private static readonly string[] SpecialNumberText =
    {
        "3.5", "inf", "-0.0", "nan", "2.25", "-1.5", "-inf", "0.0", "7.75", "-2.0", "1e+300", "-6.5", "4.0", "1e-300",
        "-3.25", "8.0", "9.5", "-0.0", "nan", "10.0", "11.5", "-8.0", "12.0", "-9.5",
    };

    /// <summary>
    ///     The items of the pool operand as TEXT: its value sequence, not its dtype's rendering - the integers of
    ///     <see cref="Plain"/>, the Python float reprs of <see cref="Special"/> (<see cref="SpecialNumberText"/>), or a text
    ///     pool's strings - as many as the operand holds. The replay rebuilds the same items from its own copy of the
    ///     sequences.
    /// </summary>
    /// <param name="dt">The pool dtype (a text pool reads its strings; any other its numbers).</param>
    /// <param name="baseLayout">The pool layout (a <c>_special</c> layout reads the special sequence).</param>
    /// <param name="size">The operand's element count.</param>
    /// <returns>The items.</returns>
    private static string[] PoolTextItems(string dt, string baseLayout, long size)
    {
        bool special = baseLayout.EndsWith("_special", StringComparison.Ordinal);
        bool text = dt is "S5" or "U5" or "T";
        IEnumerable<string> items = text
            ? (special ? SpecialText : PlainText)
            : special ? SpecialNumberText : Plain.Select(v => ((long)v).ToString(CultureInfo.InvariantCulture));
        return items.Take((int)size).ToArray();
    }

    /// <summary>
    ///     The forms np.fromstring's text takes, by its parameter's type: a <c>string</c> or its UTF-8 <c>byte[]</c>, each
    ///     <c>comma</c> (the items joined by ", "), <c>space</c> (by " "), <c>tail</c> (", " with unmatched data after the
    ///     last item), and for the bytes <c>nul</c> (an embedded NUL after the first item, where NumPy's C string ends);
    ///     <c>null</c> once.
    /// </summary>
    /// <param name="p">The text parameter.</param>
    /// <returns>The form names.</returns>
    private static string[] TextSourceForms(ParameterInfo p) =>
        p.ParameterType == typeof(byte[])
            ? new[] { "comma", "space", "tail", "nul", "null" }
            : new[] { "comma", "space", "tail", "null" };

    /// <summary>
    ///     Builds np.fromstring's text: <paramref name="form"/> (<see cref="TextSourceForms"/>) of the pool operand's items
    ///     (<see cref="PoolTextItems"/>), or false for a form built only once.
    /// </summary>
    /// <param name="t">The text parameter's type (<c>string</c> or <c>byte[]</c>).</param>
    /// <param name="form">The form.</param>
    /// <param name="dt">The pool dtype.</param>
    /// <param name="baseLayout">The pool layout.</param>
    /// <param name="operand">The pool operand (its size).</param>
    /// <param name="value">Receives the text.</param>
    /// <returns>Whether the form exists for this operand.</returns>
    /// <exception cref="ArgumentException">An unknown form (a bug in the tool).</exception>
    private static bool TryMakeTextSource(Type t, string form, string dt, string baseLayout, NDArray operand, out object value)
    {
        value = null;
        if (form == "null")
            return dt == "float64" && baseLayout == "0d";
        string text = SourceText(form, PoolTextItems(dt, baseLayout, operand.size));
        value = t == typeof(byte[]) ? Encoding.UTF8.GetBytes(text) : text;
        return true;
    }

    /// <summary>The text of a text form (<see cref="TextSourceForms"/>, <see cref="FileSourceForms"/>) over the items.</summary>
    /// <param name="form">The form: comma, space, tail or nul (a text file's <c>text</c> is comma).</param>
    /// <param name="items">The items.</param>
    /// <returns>The text.</returns>
    /// <exception cref="ArgumentException">An unknown form (a bug in the tool).</exception>
    private static string SourceText(string form, string[] items) => form switch
    {
        "comma" or "text" or "textfile" => string.Join(", ", items),
        "space" => string.Join(" ", items),
        "tail" => string.Join(", ", items) + ", junk",
        "nul" => items.Length == 0 ? "\0" : items[0] + "\0, " + string.Join(", ", items.Skip(1)),
        _ => throw new ArgumentException("unknown text form " + form),
    };

    /// <summary>
    ///     The forms np.fromfile's file takes, by its parameter's type: for a <see cref="Stream"/>, <c>bin</c> (a
    ///     MemoryStream of the operand's C-order bytes), <c>text</c> (one of its items as text, joined by ", "),
    ///     <c>unseekable</c> (the bytes behind a stream that cannot seek: NumSharp reads it, NumPy cannot) and <c>null</c>;
    ///     for a file name, <c>binfile</c> and <c>textfile</c> (a temporary file of the same bytes or text), <c>missing</c> (a
    ///     name no file has) and <c>null</c>.
    /// </summary>
    /// <param name="p">The file parameter.</param>
    /// <returns>The form names.</returns>
    private static string[] FileSourceForms(ParameterInfo p) =>
        p.ParameterType == typeof(Stream)
            ? new[] { "bin", "text", "unseekable", "null" }
            : new[] { "binfile", "textfile", "missing", "null" };

    /// <summary>
    ///     The temporary directory of the file forms and sinks, made on first use (a run over other names makes none) and
    ///     deleted when the process exits; its files are rewritten per call. Its random name never reaches an outcome: the
    ///     only file name an outcome can carry is the missing file's (<see cref="MissingFileName"/>).
    /// </summary>
    private static readonly Lazy<string> FileIODirLazy = new(() =>
    {
        string dir = Directory.CreateTempSubdirectory("numsharp-snapshot-").FullName;
        // Best effort: a file still open at exit (none is: every stream the tool opens is closed by its call or here)
        // would leave the directory behind, which costs nothing but disk.
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            try { Directory.Delete(dir, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        };
        return dir;
    });

    /// <summary>The temporary directory of the file forms and sinks (<see cref="FileIODirLazy"/>).</summary>
    private static string FileIODir => FileIODirLazy.Value;

    /// <summary>
    ///     The name of np.fromfile's <c>missing</c> form: relative (to the working directory, the repo root) and fixed,
    ///     because FileNotFoundException's text carries it and two runs must give the same line; the replay asks NumPy
    ///     for the same name.
    /// </summary>
    private const string MissingFileName = "numsharp-snapshot-no-such-file.bin";

    /// <summary>
    ///     Builds np.fromfile's file: <paramref name="form"/> (<see cref="FileSourceForms"/>) of the pool operand, or false
    ///     when the operand has no such form (a StringDType operand has no bytes; a text form exists for the numeric float64
    ///     and the U5 pool only, as np.fromstring's text: its items depend on the pool's sequence, not its dtype; a form is
    ///     built once).
    /// </summary>
    /// <param name="t">The file parameter's type (<see cref="Stream"/> or <c>string</c>).</param>
    /// <param name="form">The form.</param>
    /// <param name="dt">The pool dtype.</param>
    /// <param name="baseLayout">The pool layout.</param>
    /// <param name="operand">The pool operand.</param>
    /// <param name="value">Receives the stream or the file name.</param>
    /// <returns>Whether the form exists for this operand.</returns>
    /// <exception cref="ArgumentException">An unknown form (a bug in the tool).</exception>
    private static bool TryMakeFileSource(Type t, string form, string dt, string baseLayout, NDArray operand, out object value)
    {
        value = null;
        bool onceSlot = dt == "float64" && baseLayout == "0d";
        switch (form)
        {
            case "null":
                return onceSlot;
            case "missing":
                if (!onceSlot)
                    return false;
                value = MissingFileName;
                return true;
        }
        byte[] bytes;
        if (form is "text" or "textfile")
        {
            if (dt is not ("float64" or "U5"))
                return false;
            bytes = Encoding.UTF8.GetBytes(SourceText(form, PoolTextItems(dt, baseLayout, operand.size)));
        }
        else
        {
            bytes = BufferBytesOf(operand);
            if (bytes is null)
                return false;
        }
        switch (form)
        {
            case "bin":
            case "text":
                value = new MemoryStream(bytes, writable: false);
                return true;
            case "unseekable":
                value = new ForwardOnlyStream(bytes);
                return true;
            case "binfile":
            case "textfile":
            {
                string name = Path.Combine(FileIODir, "source.bin");
                File.WriteAllBytes(name, bytes);
                value = name;
                return true;
            }
            default:
                throw new ArgumentException("unknown file form " + form);
        }
    }

    /// <summary>
    ///     The forms np.loadtxt's source takes, by its parameter's type: <c>rows</c> (the pool's items as rows, fields joined
    ///     by a space), <c>csv</c> (joined by a comma), <c>comments</c> (a comment line first and a comment after each row),
    ///     <c>blank</c> (a blank line before each row and at the end), <c>ragged</c> (the last row one field short),
    ///     <c>quoted</c> (every field in double quotes); for a file name also <c>gz</c> (the rows gzip-compressed under a
    ///     <c>.gz</c> name, which NumPy's DataSource decompresses) and <c>missing</c> (a name no file has); <c>null</c> once.
    /// </summary>
    /// <param name="p">The source parameter.</param>
    /// <returns>The form names.</returns>
    private static string[] LoadTxtSourceForms(ParameterInfo p) =>
        p.ParameterType == typeof(string)
            ? new[] { "rows", "csv", "comments", "blank", "ragged", "quoted", "gz", "missing", "null" }
            : new[] { "rows", "csv", "comments", "blank", "ragged", "quoted", "null" };

    /// <summary>
    ///     The name of np.loadtxt's <c>missing</c> form: relative and fixed, as np.fromfile's (<see cref="MissingFileName"/>),
    ///     because NumPy's <c>not found</c> text carries it.
    /// </summary>
    private const string MissingTextName = "numsharp-snapshot-no-such-file.txt";

    /// <summary>
    ///     The lines of a np.loadtxt source form over the pool operand's items (<see cref="PoolTextItems"/>, in C order): a
    ///     0-d operand is one row of one field, a 1-D one a column, an N-D one its first axis' rows of the remaining
    ///     elements. The replay builds the same lines from the same items.
    /// </summary>
    /// <param name="form">The form (<see cref="LoadTxtSourceForms"/>; <c>gz</c> holds the plain rows).</param>
    /// <param name="items">The items.</param>
    /// <param name="operand">The pool operand (its shape).</param>
    /// <returns>The lines without their newlines, or null when the operand has no such form (a ragged row needs two rows of
    ///     two fields).</returns>
    /// <exception cref="ArgumentException">An unknown form (a bug in the tool).</exception>
    private static string[] LoadTxtLines(string form, string[] items, NDArray operand)
    {
        int rows = operand.ndim == 0 ? 1 : (int)operand.shape[0];
        int cols = rows == 0 ? 0 : items.Length / rows;
        IEnumerable<string> Row(int r) => items.Skip(r * cols).Take(cols);
        var lines = new List<string>();
        switch (form)
        {
            case "rows":
            case "gz":
                for (int r = 0; r < rows; r++)
                    lines.Add(string.Join(" ", Row(r)));
                break;
            case "csv":
                for (int r = 0; r < rows; r++)
                    lines.Add(string.Join(",", Row(r)));
                break;
            case "comments":
                lines.Add("# header line");
                for (int r = 0; r < rows; r++)
                    lines.Add(string.Join(" ", Row(r)) + " # note");
                break;
            case "blank":
                for (int r = 0; r < rows; r++)
                {
                    lines.Add("");
                    lines.Add(string.Join(" ", Row(r)));
                }
                lines.Add("");
                break;
            case "ragged":
                if (rows < 2 || cols < 2)
                    return null;
                for (int r = 0; r < rows; r++)
                    lines.Add(string.Join(" ", r == rows - 1 ? Row(r).Take(cols - 1) : Row(r)));
                break;
            case "quoted":
                for (int r = 0; r < rows; r++)
                    lines.Add(string.Join(" ", Row(r).Select(s => "\"" + s + "\"")));
                break;
            default:
                throw new ArgumentException("unknown loadtxt form " + form);
        }
        return lines.ToArray();
    }

    /// <summary>
    ///     Builds np.loadtxt's source: <paramref name="form"/> (<see cref="LoadTxtSourceForms"/>) of the pool operand's items
    ///     in the parameter's C# form - a temporary file of the UTF-8 text (gzip-compressed under a <c>.gz</c> name for
    ///     <c>gz</c>; rewritten per call), a MemoryStream of the UTF-8 text, a StringReader, an array of the lines - or
    ///     false when the operand has no such form (a ragged row, or a form built once).
    /// </summary>
    /// <param name="t">The source parameter's type.</param>
    /// <param name="form">The form.</param>
    /// <param name="dt">The pool dtype.</param>
    /// <param name="baseLayout">The pool layout.</param>
    /// <param name="operand">The pool operand.</param>
    /// <param name="value">Receives the source.</param>
    /// <returns>Whether the form exists for this operand.</returns>
    /// <exception cref="ArgumentException">An unknown form or parameter type (a bug in the tool).</exception>
    private static bool TryMakeLoadTxtSource(Type t, string form, string dt, string baseLayout, NDArray operand, out object value)
    {
        value = null;
        bool onceSlot = dt == "float64" && baseLayout == "0d";
        switch (form)
        {
            case "null":
                return onceSlot;
            case "missing":
                if (!onceSlot)
                    return false;
                value = MissingTextName;
                return true;
        }
        string[] lines = LoadTxtLines(form, PoolTextItems(dt, baseLayout, operand.size), operand);
        if (lines is null)
            return false;
        byte[] utf8 = Encoding.UTF8.GetBytes(lines.Length == 0 ? "" : string.Join("\n", lines) + "\n");
        if (t == typeof(string))
        {
            string name = Path.Combine(FileIODir, form == "gz" ? "source.txt.gz" : "source.txt");
            if (form == "gz")
            {
                using var file = File.Create(name);
                using var zip = new GZipStream(file, CompressionLevel.Optimal);
                zip.Write(utf8);
            }
            else
            {
                File.WriteAllBytes(name, utf8);
            }
            value = name;
            return true;
        }
        if (t == typeof(Stream))
            value = new MemoryStream(utf8, writable: false);
        else if (t == typeof(TextReader))
            value = new StringReader(Encoding.UTF8.GetString(utf8));
        else if (t == typeof(IEnumerable<string>))
            value = lines;
        else
            throw new ArgumentException("unknown loadtxt source type " + t);
        return true;
    }

    /// <summary>
    ///     np.savetxt's fmt as a list (its <c>string[]</c> overloads), built from X's column count (NumPy's <c>ncol</c>: the
    ///     second axis of a 2-D X, 1 otherwise): <c>each</c> (one <c>%.3f</c> per column, the base), <c>long</c> (one more:
    ///     NumPy's <c>fmt has wrong shape.</c>), <c>mixed</c> (<c>%d</c> and <c>%.2f</c> by turns), <c>empty</c> and
    ///     <c>null</c>.
    /// </summary>
    private static readonly string[] SaveTxtFmtsLabels = { "each", "long", "mixed", "empty", "null" };

    /// <summary>Builds np.savetxt's fmt list <paramref name="label"/> (<see cref="SaveTxtFmtsLabels"/>) for X.</summary>
    /// <param name="label">The variant.</param>
    /// <param name="operand">The pool operand (X).</param>
    /// <returns>The list, or null.</returns>
    /// <exception cref="ArgumentException">An unknown variant (a bug in the tool).</exception>
    private static string[] MakeSaveTxtFmts(string label, NDArray operand)
    {
        int cols = operand.ndim == 2 ? (int)operand.shape[1] : 1;
        return label switch
        {
            "each" => Enumerable.Repeat("%.3f", cols).ToArray(),
            "long" => Enumerable.Repeat("%.3f", cols + 1).ToArray(),
            "mixed" => Enumerable.Range(0, cols).Select(i => i % 2 == 0 ? "%d" : "%.2f").ToArray(),
            "empty" => Array.Empty<string>(),
            "null" => null,
            _ => throw new ArgumentException("unknown savetxt fmt list " + label),
        };
    }

    /// <summary>np.loadtxt's <c>len</c> converter: the field's length (Python's <c>len(s)</c>, a Python int).</summary>
    private static readonly Func<string, object> LoadTxtConvLen = s => (long)s.Length;

    /// <summary>np.loadtxt's <c>upper</c> converter: the field upper-cased (Python's <c>s.upper()</c> over ASCII).</summary>
    private static readonly Func<string, object> LoadTxtConvUpper = s => s.ToUpperInvariant();

    /// <summary>np.loadtxt's <c>zero</c> converter: the Python int 0, whatever the field.</summary>
    private static readonly Func<string, object> LoadTxtConvZero = _ => 0L;

    /// <summary>np.loadtxt's <c>raise</c> converter: a ValueError for every field (NumPy chains it as the conversion
    ///     error's cause).</summary>
    private static readonly Func<string, object> LoadTxtConvRaise = _ => throw new ValueError("converter refused");

    /// <summary>
    ///     The variants of np.loadtxt's and np.savetxt's parameters (<see cref="Role.TextIOArg"/>), by function and name: each
    ///     list starts with the parameter's default (so the base call is the default call), then values NumPy reads and values
    ///     it refuses. The converters come from a menu the replay names alike (<see cref="LoadTxtConvLen"/> and its
    ///     siblings). Labels hold no comma: the case id's variant list is comma-separated.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>The (label, value) pairs.</returns>
    /// <exception cref="ArgumentException">A parameter the tool has no variants for (a new parameter: give it its
    ///     variants).</exception>
    private static (string Label, object Value)[] TextIOArgVariants(ParameterInfo p) => (p.Member.Name, p.Name) switch
    {
        ("loadtxt", "comments") => new (string, object)[]
        {
            ("hash", "#"), ("percent", "%"), ("slashes", "//"), ("null", null), ("list", new[] { "#", "%" }), ("empty", ""),
            ("int", 5),
        },
        ("loadtxt", "delimiter") => new (string, object)[]
        {
            ("null", null), ("comma", ","), ("semicolon", ";"), ("tab", "\t"), ("space", " "), ("two", ",,"),
            ("newline", "\n"), ("hash", "#"),
        },
        ("loadtxt", "converters") => new (string, object)[]
        {
            ("null", null), ("len", LoadTxtConvLen), ("upper", LoadTxtConvUpper), ("raise", LoadTxtConvRaise),
            ("d0zero", new Dictionary<int, Func<string, object>> { [0] = LoadTxtConvZero }),
            ("dneglen", new Dictionary<int, Func<string, object>> { [-1] = LoadTxtConvLen }),
            ("d9len", new Dictionary<int, Func<string, object>> { [9] = LoadTxtConvLen }),
            ("dnull", new Dictionary<int, Func<string, object>> { [0] = null }),
            ("int", 5),
        },
        ("loadtxt", "skiprows") => new (string, object)[] { ("0", 0), ("1", 1), ("2", 2), ("100", 100), ("-1", -1) },
        ("loadtxt", "usecols") => new (string, object)[]
        {
            ("null", null), ("0", new[] { 0 }), ("-1", new[] { -1 }), ("0;0", new[] { 0, 0 }), ("1;0", new[] { 1, 0 }),
            ("9", new[] { 9 }), ("empty", Array.Empty<int>()),
        },
        ("loadtxt", "ndmin") => new (string, object)[] { ("0", 0), ("1", 1), ("2", 2), ("3", 3) },
        ("loadtxt", "encoding") => new (string, object)[]
        {
            ("null", null), ("utf-8", "utf-8"), ("latin1", "latin1"), ("bytes", "bytes"), ("ascii", "ascii"),
            ("utf-16", "utf-16"), ("bogus", "bogus"),
        },
        ("loadtxt", "max_rows") => new (string, object)[] { ("null", null), ("0", 0), ("1", 1), ("2", 2), ("-1", -1) },
        ("loadtxt", "quotechar") => new (string, object)[]
        {
            ("null", null), ("dq", "\""), ("sq", "'"), ("two", "qq"), ("hash", "#"),
        },
        ("savetxt", "fmt") => new (string, object)[]
        {
            ("%.18e", "%.18e"), ("%d", "%d"), ("%s", "%s"), ("%.3f", "%.3f"), ("%x", "%x"), ("%5.2f", "%5.2f"),
            ("abc", "abc"), ("%", "%"), ("pair", "%d;%d"), ("null", null),
        },
        ("savetxt", "delimiter") => new (string, object)[]
        {
            ("space", " "), ("comma", ","), ("tab", "\t"), ("empty", ""), ("two", ";;"), ("null", null),
        },
        ("savetxt", "newline") => new (string, object)[]
        {
            ("lf", "\n"), ("crlf", "\r\n"), ("semicolon", ";"), ("empty", ""), ("null", null),
        },
        ("savetxt", "header") => new (string, object)[] { ("empty", ""), ("h", "h"), ("twolines", "two\nlines"), ("null", null) },
        ("savetxt", "footer") => new (string, object)[] { ("empty", ""), ("f", "f"), ("twolines", "two\nlines"), ("null", null) },
        ("savetxt", "comments") => new (string, object)[] { ("hash", "# "), ("slashes", "//"), ("empty", ""), ("null", null) },
        ("savetxt", "encoding") => new (string, object)[]
        {
            ("null", null), ("utf-8", "utf-8"), ("latin1", "latin1"), ("ascii", "ascii"), ("utf-16", "utf-16"),
            ("bogus", "bogus"),
        },
        _ => throw new ArgumentException($"no text IO variants for {p.Member.Name}'s {p.Name}"),
    };

    /// <summary>
    ///     The bytes a sink received, for the outcome: <c> WROTE &lt;hash&gt;/&lt;length&gt;</c> (the first 16 hex digits of
    ///     their SHA-256), read from the MemoryStream, the StringWriter (its text as UTF-8) or the temporary file (then
    ///     deleted), or <c> WROTE none</c> when the call created no file.
    /// </summary>
    /// <param name="sink">The MemoryStream, the StringWriter or the file name the call received.</param>
    /// <returns>The suffix.</returns>
    private static string WrittenBytes(object sink)
    {
        byte[] bytes;
        if (sink is StringWriter text)
        {
            bytes = Encoding.UTF8.GetBytes(text.ToString());
        }
        else if (sink is MemoryStream ms)
        {
            bytes = ms.ToArray();
        }
        else if (sink is string name && File.Exists(name))
        {
            bytes = File.ReadAllBytes(name);
            File.Delete(name);
        }
        else
        {
            return " WROTE none";
        }
        return $" WROTE {Convert.ToHexString(SHA256.HashData(bytes)).Substring(0, 16).ToLowerInvariant()}/{bytes.Length}";
    }

    /// <summary>
    ///     A read-only stream over bytes that cannot seek (NumPy's np.fromfile needs a seekable file descriptor; NumSharp
    ///     reads such a stream), reporting how far it was read as its position.
    /// </summary>
    private sealed class ForwardOnlyStream : Stream
    {
        /// <summary>The bytes.</summary>
        private readonly byte[] _bytes;

        /// <summary>The next byte to read.</summary>
        private int _next;

        /// <summary>A stream over <paramref name="bytes"/>, from their first byte.</summary>
        /// <param name="bytes">The bytes (not copied).</param>
        public ForwardOnlyStream(byte[] bytes) => _bytes = bytes;

        /// <inheritdoc/>
        public override bool CanRead => true;

        /// <inheritdoc/>
        public override bool CanSeek => false;

        /// <inheritdoc/>
        public override bool CanWrite => false;

        /// <inheritdoc/>
        /// <exception cref="NotSupportedException">Always: the stream cannot seek.</exception>
        public override long Length => throw new NotSupportedException();

        /// <summary>How many bytes were read (setting it is refused: the stream cannot seek).</summary>
        /// <exception cref="NotSupportedException">On a set.</exception>
        public override long Position
        {
            get => _next;
            set => throw new NotSupportedException();
        }

        /// <inheritdoc/>
        public override int Read(byte[] buffer, int offset, int count)
        {
            int n = Math.Min(count, _bytes.Length - _next);
            Array.Copy(_bytes, _next, buffer, offset, n);
            _next += n;
            return n;
        }

        /// <inheritdoc/>
        public override void Flush() { }

        /// <inheritdoc/>
        /// <exception cref="NotSupportedException">Always.</exception>
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        /// <inheritdoc/>
        /// <exception cref="NotSupportedException">Always.</exception>
        public override void SetLength(long value) => throw new NotSupportedException();

        /// <inheritdoc/>
        /// <exception cref="NotSupportedException">Always.</exception>
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    /// <summary>
    ///     The memory the buffer forms <c>native</c> and <c>ptr</c> lay their bytes in (NumSharp uint8 arrays, so a read past
    ///     a buffer's end meets a guard page under <c>NUMSHARP_DEBUG_GUARD_PAGES=1</c>), kept alive for the whole run: a result
    ///     of np.frombuffer over an address views that memory without owning it, as it would the caller's memory.
    /// </summary>
    private static readonly List<NDArray> BufferMemory = new();

    /// <summary>
    ///     The forms np.frombuffer's buffer operand takes, by the type of its first parameter: each a C# value built from the
    ///     pool operand by <see cref="TryMakeBufferOperand"/>.
    ///     <list type="bullet">
    ///         <item><c>bytes</c>: a <c>byte[]</c> of the operand's C-order bytes (NumPy's <c>bytearray</c>: writable).</item>
    ///         <item><c>data</c>: the operand's own <see cref="np.MemoryView"/> (<c>ndarray.data</c>: np.frombuffer refuses a
    ///             non-contiguous operand's, and a read-only operand's makes a read-only result).</item>
    ///         <item><c>segment</c>: an <c>ArraySegment&lt;byte&gt;</c> of the bytes at offset 3 of a larger array, junk bytes
    ///             after them; <c>whole</c>: one over exactly the bytes.</item>
    ///         <item><c>memory</c>: a <c>Memory&lt;byte&gt;</c> of the bytes at offset 3 of a larger array; <c>native</c>:
    ///             one over unmanaged memory (no array behind it: np.frombuffer copies it through a span).</item>
    ///         <item><c>ptr</c>: the address of unmanaged memory holding the bytes (an <c>IntPtr</c>, or a <c>void*</c> boxed
    ///             for reflection).</item>
    ///         <item><c>flat</c>: the generic overload's <c>TSource[]</c> of the operand's values in C order.</item>
    ///         <item><c>null</c> / <c>default</c> / <c>nullptr</c>: null, a default <c>ArraySegment</c> / <c>Memory</c>, a null
    ///             address, built once (the float64 0-d slot).</item>
    ///     </list>
    /// </summary>
    /// <param name="p">The buffer parameter.</param>
    /// <returns>The form names, in the order the pools try them.</returns>
    /// <exception cref="ArgumentException">A buffer type the tool does not know (a new overload: give it its forms).</exception>
    private static string[] BufferForms(ParameterInfo p)
    {
        Type t = p.ParameterType;
        // The generic overload first: its TSource[] closed over byte is a byte[] too, and takes the operand's VALUES.
        if (p.Member is MethodInfo { IsGenericMethod: true } && t.IsArray) return new[] { "flat", "null" };
        if (t == typeof(byte[])) return new[] { "bytes", "null" };
        if (t == typeof(np.MemoryView)) return new[] { "data", "null" };
        if (t == typeof(ArraySegment<byte>)) return new[] { "segment", "whole", "default" };
        if (t == typeof(Memory<byte>)) return new[] { "memory", "native", "default" };
        if (t == typeof(IntPtr) || t.IsPointer) return new[] { "ptr", "nullptr" };
        throw new ArgumentException("unknown np.frombuffer buffer type " + TypeName(t));
    }

    /// <summary>
    ///     Builds np.frombuffer's buffer operand: <paramref name="form"/> (<see cref="BufferForms"/>) of the pool operand for a
    ///     buffer parameter of type <paramref name="t"/>, or false when the operand has no such form (a StringDType operand has
    ///     no bytes and no memoryview, a typed array takes only the operands of its element type, a value is built once).
    /// </summary>
    /// <param name="t">The buffer parameter's type.</param>
    /// <param name="form">The form.</param>
    /// <param name="dt">The operand's pool dtype.</param>
    /// <param name="baseLayout">The operand's pool layout.</param>
    /// <param name="operand">The pool operand.</param>
    /// <param name="value">Receives the buffer (a <c>void*</c> boxed by <see cref="Pointer.Box"/>).</param>
    /// <param name="length">Receives the buffer's length in bytes, which the address overloads' byte length is built from
    ///     (<see cref="MakeBufferLength"/>); 0 for the once-built forms and the typed array, whose overloads take none.</param>
    /// <returns>Whether the form exists for this operand.</returns>
    /// <exception cref="ArgumentException">An unknown form (a bug in the tool).</exception>
    private static unsafe bool TryMakeBufferOperand(Type t, string form, string dt, string baseLayout, NDArray operand,
        out object value, out long length)
    {
        value = null;
        length = 0;
        // The once-built forms take one slot: the float64 0-d operand.
        bool onceSlot = dt == "float64" && baseLayout == "0d";
        switch (form)
        {
            case "null":
            case "default":
            case "nullptr":
                if (!onceSlot)
                    return false;
                value = form == "null" ? null
                    : form == "default" ? (t == typeof(ArraySegment<byte>) ? default(ArraySegment<byte>) : (object)default(Memory<byte>))
                    : t.IsPointer ? Pointer.Box(null, t) : IntPtr.Zero;
                return true;
            case "data":
                // ndarray.data refuses a dtype that holds references (StringDType, as NumPy refuses its buffer): that operand
                // has no MemoryView, so no line.
                try
                {
                    var view = operand.data;
                    value = view;
                    length = view.nbytes;
                }
                catch (ValueError)
                {
                    return false;
                }
                return true;
            case "flat":
            {
                // The generic overload's TSource[]: only the operands whose values are TSource (RunOverload filters the
                // dtypes, so this holds for every operand it hands over).
                Type e = t.GetElementType();
                if (HostElementType(dt) != e)
                    return false;
                object[] items = Enumerable.Range(0, (int)operand.size).Select(i => operand.GetAtIndex(i)).ToArray();
                value = Rectangular(e, new[] { items.Length }, items);
                return true;
            }
        }
        // The remaining forms carry the operand's C-order bytes; a StringDType operand has none.
        byte[] bytes = BufferBytesOf(operand);
        if (bytes is null)
            return false;
        length = bytes.LongLength;
        switch (form)
        {
            case "bytes":
                value = bytes;
                return true;
            case "whole":
                value = new ArraySegment<byte>(bytes);
                return true;
            case "segment":
            case "memory":
            {
                // The bytes at offset 3 of a larger array with junk after them: the segment or memory, not the array behind it,
                // bounds the buffer.
                var larger = Enumerable.Repeat((byte)0xA5, bytes.Length + 8).ToArray();
                Buffer.BlockCopy(bytes, 0, larger, 3, bytes.Length);
                value = form == "segment"
                    ? new ArraySegment<byte>(larger, 3, bytes.Length)
                    : (object)new Memory<byte>(larger, 3, bytes.Length);
                return true;
            }
            case "native":
            case "ptr":
            {
                // Unmanaged memory holding the bytes: a NumSharp uint8 array one byte longer (an empty buffer still has an
                // address), kept alive for the run (BufferMemory).
                var memory = np.empty(new Shape(bytes.Length + 1), np.uint8);
                BufferMemory.Add(memory);
                byte* address = memory.Unsafe.Pointer<byte>();
                bytes.AsSpan().CopyTo(new Span<byte>(address, bytes.Length));
                if (form == "native")
                {
                    // A Memory<byte> over exactly the bytes, backed by a memory manager (no array behind it).
                    var window = memory[$":{bytes.Length}"];
                    BufferMemory.Add(window);
                    value = window.Unsafe.Memory<byte>();
                }
                else
                {
                    value = t.IsPointer ? Pointer.Box(address, t) : (object)(IntPtr)address;
                }
                return true;
            }
            default:
                throw new ArgumentException("unknown buffer form " + form);
        }
    }

    /// <summary>
    ///     The pool operand's bytes in C order, the content of every buffer form but the memoryview and the typed array: the
    ///     raw <c>size * itemsize</c> bytes of a C-contiguous copy (NumPy's <c>arr.tobytes()</c>, read without NumSharp's
    ///     tobytes, a batch 6 function), or null for a StringDType operand, whose cells hold pointers.
    /// </summary>
    /// <param name="operand">The pool operand.</param>
    /// <returns>A new array of the bytes, or null.</returns>
    private static byte[] BufferBytesOf(NDArray operand) =>
        operand.dtype.hasobject ? null : np.ascontiguousarray(operand).Unsafe.ReadOnlyBytes().ToArray();

    /// <summary>
    ///     The address overloads' <c>byteLength</c> variants, built from the buffer's length (<see cref="MakeBufferLength"/>):
    ///     the exact length first, then one byte less (a partial element for every multi-byte dtype) and none. No variant
    ///     reaches past the buffer: the memory is the tool's, and a view past its end would read whatever follows it.
    /// </summary>
    private static readonly string[] BufferLengthLabels = { "exact", "less1", "zero" };

    /// <summary>A byte length of <see cref="BufferLengthLabels"/> for a buffer of <paramref name="length"/> bytes.</summary>
    /// <param name="label">The variant.</param>
    /// <param name="length">The buffer's length in bytes.</param>
    /// <returns>The byte length, never negative.</returns>
    /// <exception cref="ArgumentException">An unknown variant (a bug in the tool).</exception>
    private static long MakeBufferLength(string label, long length) => label switch
    {
        "exact" => length,
        "less1" => Math.Max(0, length - 1),
        "zero" => 0L,
        _ => throw new ArgumentException("unknown byte length " + label),
    };

    /// <summary>
    ///     The element type a typed host parameter pins (<c>T</c> of <c>T</c>, <c>T[]</c>, <c>T[][]</c>, <c>T[,]</c> or
    ///     <c>IEnumerable&lt;T&gt;</c>), or null for a parameter that takes any element type (<c>object</c>, <c>Array</c>,
    ///     a str or str array, a <see cref="np.MemoryView"/>).
    /// </summary>
    /// <param name="t">The host parameter's type (by-ref unwrapped).</param>
    /// <returns>The pinned element type, or null.</returns>
    private static Type PinnedHostElement(Type t) =>
        t == typeof(object) || t == typeof(Array) || t == typeof(string) || t == typeof(string[]) || t == typeof(np.MemoryView)
            ? null
            : t.IsArray ? InnermostElement(t)
            : t.IsGenericType && t.GetGenericTypeDefinition() == typeof(IEnumerable<>) ? t.GetGenericArguments()[0]
            : t;

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
    /// <param name="layout">The operand layout; for a host operand <c>layout:form</c> (the pool layout, then the C# form of
    ///     <see cref="HostForms"/>), for np.frombuffer's buffer likewise (<see cref="BufferForms"/>).</param>
    /// <param name="variants">Per parameter, its variants (null for the operand).</param>
    /// <param name="roles">Per parameter, its role.</param>
    /// <param name="assignments">The varied parameters and their 1-based variant indices; empty for phase A.</param>
    /// <returns>One output line; null when the overload's host operand (the coercion family's) or buffer (np.frombuffer's)
    ///     has no value in the form the layout names for this operand (<see cref="TryMakeHostOperand"/>,
    ///     <see cref="TryMakeBufferOperand"/>), which RunOverload skips.</returns>
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
            Role.EinsumSubscripts => EinsumSubscriptsVariantLabels[v],
            Role.EinsumOps => EinsumOpsVariantLabels[v],
            Role.LexKeys => LexKeysVariantLabels[v],
            Role.RavelCoords => RavelCoordsVariantLabels[v],
            Role.EvalExpr or Role.EvalExprPositional => EvalExprVariantLabels[v],
            Role.ItemArg => ItemArgLabels(ps[p])[v],
            Role.AssignArg => AssignArgLabels(ps[p])[v],
            Role.BufferLength => BufferLengthLabels[v],
            Role.SaveTxtFmts => SaveTxtFmtsLabels[v],
            Role.SampleSeq or Role.BinsInts or Role.BinsArrays or Role.BinsObjects or Role.RangeArray => DdLabels(roles[p])[v],
            _ => variants[p][v].Label,
        };

        // A spacing generator's operand is its (start, stop) pair, not a pool operand: a fresh 0-d array stands in for the
        // MUTATED check, which then never fires.
        bool spacingCall = operandIndex >= 0 && roles[operandIndex] == Role.SpacePair;
        // A constructor's operand is its shape: a fresh 0-d array stands in for the same MUTATED check.
        bool ctorCall = operandIndex >= 0 && roles[operandIndex] == Role.CtorShape;
        // np.mgrid / np.ogrid's operand is their key: a fresh 0-d array stands in for the same MUTATED check. Its label holds
        // colons, so it is never split into "layout:form".
        bool gridKeyCall = operandIndex >= 0 && roles[operandIndex] == Role.GridKey;
        // np.ravel_multi_index's scalar overload's operand is its coordinate tuple: a fresh 0-d array stands in too.
        bool ravelScalarCall = operandIndex >= 0 && roles[operandIndex] == Role.RavelScalarCoords;
        // A host operand's layout is "layout:form" (RunOverload): the pool operand of the layout, then its C# form.
        int formAt = gridKeyCall ? -1 : layout.IndexOf(':');
        string baseLayout = formAt < 0 ? layout : layout.Substring(0, formAt);
        NDArray operand = spacingCall || ctorCall || gridKeyCall || ravelScalarCall
            ? np.array(0.0)
            : MakeOperand(dt, baseLayout);
        // A matrix receiver is np.asmatrix's view of the pool operand (0-d -> (1, 1), 1-D -> (1, N)); a layout it refuses
        // (more than two long axes: "shape too large to be a matrix.") has no matrix to call the member on.
        if (matrix && operand is not null)
            operand = AsMatrixOrNull(operand);
        if (operand is null)
            return $"{signature}|{dt}/{layout}|{varied}\tSKIP operand unavailable";
        object operandArg = operand;
        if (spacingCall)
            operandArg = MakeSpaceValue(ps[operandIndex].ParameterType, layout, start: true);
        // A shape with no value in the parameter's form (a 2-D shape as an int) makes no line at all.
        if (ctorCall && !TryMakeCtorShape(ps[operandIndex], layout, out operandArg))
            return null;
        if (gridKeyCall)
            operandArg = MakeGridKey(layout);
        if (ravelScalarCall)
            operandArg = MakeRavelScalarCoords(layout);
        if (operandIndex >= 0 && roles[operandIndex] == Role.HostOperand)
        {
            // The coercion family's C# value of the operand; a form this operand does not have makes no line at all.
            Type hostType = ps[operandIndex].ParameterType.IsByRef ? ps[operandIndex].ParameterType.GetElementType() : ps[operandIndex].ParameterType;
            if (!TryMakeHostOperand(hostType, layout.Substring(formAt + 1), dt, baseLayout, operand,
                    oneElement: m.Name == "Scalar", out operandArg))
                return null;
        }
        // np.frombuffer's buffer, built from the operand's bytes ("layout:form" as a host operand's); a form this operand
        // does not have makes no line at all. Its length is the address overloads' exact byte length.
        long bufferLength = 0;
        if (operandIndex >= 0 && roles[operandIndex] == Role.BufferOperand
            && !TryMakeBufferOperand(ps[operandIndex].ParameterType, layout.Substring(formAt + 1), dt, baseLayout, operand,
                out operandArg, out bufferLength))
            return null;
        // np.fromstring's text and np.fromfile's file ("layout:form" as a host operand's); a form this operand does not have
        // makes no line at all.
        if (operandIndex >= 0 && roles[operandIndex] == Role.TextSource
            && !TryMakeTextSource(ps[operandIndex].ParameterType, layout.Substring(formAt + 1), dt, baseLayout, operand, out operandArg))
            return null;
        if (operandIndex >= 0 && roles[operandIndex] == Role.FileSource
            && !TryMakeFileSource(ps[operandIndex].ParameterType, layout.Substring(formAt + 1), dt, baseLayout, operand, out operandArg))
            return null;
        // np.loadtxt's source ("layout:form" as a host operand's); a form this operand does not have makes no line at all.
        if (operandIndex >= 0 && roles[operandIndex] == Role.LoadTxtSource
            && !TryMakeLoadTxtSource(ps[operandIndex].ParameterType, layout.Substring(formAt + 1), dt, baseLayout, operand, out operandArg))
            return null;
        string inBefore = HashOf(operand);
        var argv = new object[ps.Length];
        NDArray outArray = null;
        try
        {
            for (int i = 0; i < ps.Length; i++)
                argv[i] = i == operandIndex ? operandArg
                    : roles[i] == Role.Operand2 ? MakeOperand2(Operand2VariantLabels[0], operand)
                    : roles[i] == Role.Keys ? MakeKeys(KeysVariantLabels[0], operand)
                    : roles[i] == Role.SlidingKernel ? MakeSlidingKernel(SlidingKernelVariantLabels[0], operand)
                    : roles[i] == Role.IsinTest ? MakeIsinTest(IsinTestVariantLabels[0], operand)
                    : roles[i] == Role.LinalgArrays ? MakeLinalgArrays(LinalgArraysVariantLabels[0], operand)
                    : roles[i] == Role.NditerOps ? MakeNditerOps(NditerOpsVariantLabels[0], operand)
                    : roles[i] == Role.EinsumSubscripts ? MakeEinsumSubscripts(EinsumSubscriptsVariantLabels[0], operand)
                    : roles[i] == Role.EinsumOps ? MakeEinsumOps(EinsumOpsVariantLabels[0], operand)
                    : roles[i] == Role.LexKeys ? MakeLexKeys(LexKeysVariantLabels[0], operand)
                    : roles[i] == Role.RavelCoords ? MakeRavelCoords(RavelCoordsVariantLabels[0], operand)
                    : roles[i] == Role.EvalExpr ? MakeEvalExpr(EvalExprVariantLabels[0], operand, positional: false)
                    : roles[i] == Role.EvalExprPositional ? MakeEvalExpr(EvalExprVariantLabels[0], operand, positional: true)
                    : roles[i] == Role.EvalOperands ? new[] { operand }
                    : roles[i] == Role.ItemArg ? MakeItemArg(ps[i], ItemArgLabels(ps[i])[0], operand)
                    : roles[i] == Role.AssignArg ? MakeAssignArg(ps[i], AssignArgLabels(ps[i])[0], operand)
                    : roles[i] == Role.SpaceStop ? MakeSpaceValue(ps[i].ParameterType, layout, start: false)
                    : roles[i] == Role.BufferLength ? MakeBufferLength(BufferLengthLabels[0], bufferLength)
                    : roles[i] == Role.SinkStream ? new MemoryStream()
                    : roles[i] == Role.SinkFile ? Path.Combine(FileIODir, "sink.bin")
                    : roles[i] == Role.SinkWriter ? new StringWriter(CultureInfo.InvariantCulture)
                    : roles[i] == Role.SaveTxtFmts ? MakeSaveTxtFmts(SaveTxtFmtsLabels[0], operand)
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
                    case Role.EinsumSubscripts:
                        argv[p] = MakeEinsumSubscripts(EinsumSubscriptsVariantLabels[v], operand);
                        break;
                    case Role.EinsumOps:
                        argv[p] = MakeEinsumOps(EinsumOpsVariantLabels[v], operand);
                        break;
                    case Role.LexKeys:
                        argv[p] = MakeLexKeys(LexKeysVariantLabels[v], operand);
                        break;
                    case Role.RavelCoords:
                        argv[p] = MakeRavelCoords(RavelCoordsVariantLabels[v], operand);
                        break;
                    case Role.ItemArg:
                        argv[p] = MakeItemArg(ps[p], ItemArgLabels(ps[p])[v], operand);
                        break;
                    case Role.AssignArg:
                        argv[p] = MakeAssignArg(ps[p], AssignArgLabels(ps[p])[v], operand);
                        break;
                    case Role.BufferLength:
                        argv[p] = MakeBufferLength(BufferLengthLabels[v], bufferLength);
                        break;
                    case Role.SaveTxtFmts:
                        argv[p] = MakeSaveTxtFmts(SaveTxtFmtsLabels[v], operand);
                        break;
                    case Role.EvalExpr:
                    case Role.EvalExprPositional:
                        argv[p] = MakeEvalExpr(EvalExprVariantLabels[v], operand, positional: roles[p] == Role.EvalExprPositional);
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
            // The indexer setters' "selrev" value: the selection's own values, which depend on the key of the same call.
            ResolveIndexerSelection(ps, argv, operand);
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
            // An elementwise call with a where= mask and no out= (a ufunc such as np.power, np.evaluate) leaves the
            // masked-off elements of its fresh result uninitialized (NumPy's ufunc rule), so their bytes differ from run to
            // run: zero them first, and the line depends only on the elements the call defines. A text result starts zeroed
            // (its dtype needs init) and is left alone.
            if (result is NDArray fresh && outArray is null && fresh.typecode != NPTypeCode.String && IsElementwiseWithWhere(ps))
            {
                int whereIndex = Array.FindIndex(ps, q => q.Name == "where");
                if (argv[whereIndex] is NDArray mask)
                    np.copyto(fresh, 0, casting: "unsafe", where: np.logical_not(mask));
            }
            outcome = Render(result, outArray);
            if (outArray is not null && !ReferenceEquals(result, outArray))
                outcome += " OUT " + HashOf(outArray);
        }
        catch (Exception e)
        {
            outcome = "EXC " + Describe(e);
        }
        // A sink's bytes (NDArray.tofile), written before a refusal too, and a read stream's final position (np.fromfile
        // leaves a file where its read stopped), join the outcome.
        int sinkIndex = Array.FindIndex(roles, r => r is Role.SinkStream or Role.SinkFile or Role.SinkWriter);
        if (sinkIndex >= 0)
            outcome += WrittenBytes(argv[sinkIndex]);
        if (operandIndex >= 0 && roles[operandIndex] == Role.FileSource && operandArg is Stream read)
            outcome += " POS " + read.Position.ToString(CultureInfo.InvariantCulture);
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
    /// <returns><c>np.emath</c> for a member of <see cref="EmathModule"/>, <c>np.fft</c> for one of
    ///     <see cref="FourierModule"/>.</returns>
    /// <exception cref="ArgumentException">A module class the tool does not know: its module object would have to be named
    ///     here first.</exception>
    private static object ModuleReceiver(MethodInfo m) =>
        m.DeclaringType == typeof(np.AxisConcatenator) || m.DeclaringType == typeof(np.MGridClass)
        || m.DeclaringType == typeof(np.OGridClass)
            ? IndexerReceiver
            : m.DeclaringType == typeof(EmathModule)
            ? np.emath
            : m.DeclaringType == typeof(FourierModule)
                ? np.fft
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
            // np.trim_zeros' operands with zero borders (TrimLayouts): the values written inside a zero frame of the dtype
            // (an empty string for the text dtypes, False for bool, '\0' for char), so every trimmed axis has zeros to cut.
            case "zpad1d": return ZeroFramed(dt, Values(5), new Shape(9), "2:7");
            case "zpad2d": return ZeroFramed(dt, Values(12).reshape(new Shape(3, 4)), new Shape(5, 6), "1:4, 1:5");
            case "zpad3d": return ZeroFramed(dt, Values(12).reshape(new Shape(1, 3, 4)), new Shape(3, 5, 6), "1:2, 1:4, 1:5");
            case "allzero2d": return np.zeros(new Shape(3, 4), np.dtype(dt));
            // np.convolve / np.correlate's long sequence: the whole 24-value prefix, so a product of it with a long kernel
            // reaches the vector regimes of NumPy's dot functions (ddot from 16 terms, zdotu from 8).
            case "long1d": return Values(24);
            // np.bincount's 1-D operands of valid bins (BincountLayouts): the plain prefix's magnitudes, contiguous, every
            // other element of twelve, and reversed.
            case "nonneg1d": return Magnitudes(dt, 6);
            case "nonneg1d_strided": return Magnitudes(dt, 12)["::2"];
            case "nonneg1d_reversed": return Magnitudes(dt, 6)["::-1"];
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

    /// <summary>
    ///     A zero array of the pool dtype with <paramref name="inner"/> written into the region <paramref name="region"/>
    ///     selects: np.trim_zeros' zero-border operands.
    /// </summary>
    /// <param name="dt">The pool dtype name.</param>
    /// <param name="inner">The values (of <paramref name="dt"/>) written inside the frame.</param>
    /// <param name="shape">The framed array's shape.</param>
    /// <param name="region">The slice string of the region the values fill.</param>
    /// <returns>A fresh C-contiguous array whose border is the dtype's zero.</returns>
    private static NDArray ZeroFramed(string dt, NDArray inner, Shape shape, string region)
    {
        var framed = np.zeros(shape, np.dtype(dt));
        framed[region] = inner;
        return framed;
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

    /// <summary>
    ///     The plain sequence's magnitudes (3, 1, 2, 0, 5, 4, 7, ...) in a dtype: np.bincount's valid bins for every integer
    ///     dtype. A dtype without a numeric value of its own (text, char, decimal, complex) takes the plain values
    ///     (<see cref="Build"/>): np.bincount refuses it before it reads an element.
    /// </summary>
    /// <param name="dt">The dtype name.</param>
    /// <param name="count">How many values.</param>
    /// <returns>A fresh 1-D array.</returns>
    private static NDArray Magnitudes(string dt, int count) =>
        dt is "S5" or "U5" or "T" or "decimal" or "complex128" or "char"
            ? Build(dt, count, special: false)
            : np.array(Plain.Take(count).Select(Math.Abs).ToArray()).astype(np.dtype(dt));

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

    /// <summary>np.einsum's subscripts variants (see <see cref="MakeEinsumSubscripts"/>); the first, the identity of the
    ///     operand's labels (NumPy's view path), is the base.</summary>
    private static readonly string[] EinsumSubscriptsVariantLabels =
        { "id", "sum", "first", "rev", "implicit", "diag", "ellipsis", "ellsum", "pair", "pairsum", "outer", "triple", "badlabel", "badchar", "noarrow" };

    /// <summary>np.einsum's operand-list variants (see <see cref="MakeEinsumOps"/>); the first, the operand alone, is the
    ///     base.</summary>
    private static readonly string[] EinsumOpsVariantLabels = { "one", "pair", "int32", "text", "zero_d", "triple", "none" };

    /// <summary>
    ///     Builds np.einsum's subscripts for an operand from its rank: the identity of its labels (<c>ab-&gt;ab</c>, the view
    ///     path), a sum over every axis, a sum over all but the first, the labels reversed (a transposed view), the implicit
    ///     output, a diagonal (<c>aa-&gt;a</c> on a matrix: its first two labels the same), an ellipsis identity, a sum over
    ///     the last axis behind an ellipsis, a two-operand product of equal labels, its full contraction, an outer product
    ///     (the second operand's labels in upper case), a three-operand product, and three forms NumPy refuses: an output
    ///     label no input has, a digit among the labels, and an arrow without its <c>&gt;</c>.
    /// </summary>
    /// <param name="label">The variant (<see cref="EinsumSubscriptsVariantLabels"/>).</param>
    /// <param name="operand">The pool operand (its rank).</param>
    /// <returns>The subscripts.</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static string MakeEinsumSubscripts(string label, NDArray operand)
    {
        int rank = operand.ndim;
        string l = "abcdefgh".Substring(0, rank);
        string upper = l.ToUpperInvariant();
        string rev = new string(l.Reverse().ToArray());
        return label switch
        {
            "id" => l + "->" + l,
            "sum" => l + "->",
            "first" => l + "->" + (rank == 0 ? "" : l.Substring(0, 1)),
            "rev" => l + "->" + rev,
            "implicit" => l,
            // A matrix's diagonal: its first two labels the same (aab->ab for a 3-D operand); a 1-D or 0-d operand keeps
            // its identity, which has no repeated label.
            "diag" => rank >= 2 ? "a" + l.Substring(1).Replace('b', 'a') + "->" + ("a" + l.Substring(2)) : l + "->" + l,
            "ellipsis" => "...->...",
            "ellsum" => rank == 0 ? "...->..." : "..." + l.Substring(rank - 1) + "->...",
            "pair" => l + "," + l + "->" + l,
            "pairsum" => l + "," + l + "->",
            "outer" => l + "," + upper + "->" + l + upper,
            "triple" => l + "," + l + "," + l + "->" + l,
            "badlabel" => l + "->z",
            "badchar" => l + "1",
            "noarrow" => l + "-" + l,
            _ => throw new ArgumentException("unknown einsum subscripts variant " + label),
        };
    }

    /// <summary>
    ///     Builds np.einsum's operand list for an operand: the operand alone, with a copy of itself, with an int32 ramp of its
    ///     shape (another dtype: the loop dtype is the operands' result type), with U5 text of its shape, with a 0-d float64,
    ///     three of it, and no operand at all.
    /// </summary>
    /// <param name="label">The variant (<see cref="EinsumOpsVariantLabels"/>).</param>
    /// <param name="operand">The pool operand.</param>
    /// <returns>A fresh operand list (its first entry is the operand itself).</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    private static NDArray[] MakeEinsumOps(string label, NDArray operand)
    {
        long[] dims = operand.Shape.Dimensions;
        long size = 1;
        foreach (long d in dims) size *= d;
        switch (label)
        {
            case "one": return new[] { operand };
            case "pair": return new[] { operand, operand.copy() };
            case "int32": return new[] { operand, np.arange(size).astype(np.dtype("int32")).reshape(new Shape(dims)) };
            case "text":
            {
                var words = new string[size];
                for (long i = 0; i < size; i++)
                    words[i] = (i % 3).ToString(CultureInfo.InvariantCulture);
                return new[] { operand, np.array(words).astype(np.dtype("U5")).reshape(new Shape(dims)) };
            }
            case "zero_d": return new[] { operand, np.array(2.0) };
            case "triple": return new[] { operand, operand.copy(), operand.copy() };
            case "none": return Array.Empty<NDArray>();
            default: throw new ArgumentException("unknown einsum operand-list variant " + label);
        }
    }

    /// <summary>The variants of np.lexsort's key list (MakeLexKeys); the first is the base.</summary>
    private static readonly string[] LexKeysVariantLabels = { "one", "pair", "neg", "int32", "text", "mismatch", "zero_d", "nullkey", "none" };

    /// <summary>
    ///     Builds np.lexsort's key list from a pool operand: the operand alone, the operand twice, the operand after its
    ///     negation (a second key that reorders every tie of the first), an int32 ramp of the operand's shape and a U5
    ///     text key of it (keys of other dtypes, the text one sorted by its strings), a key of another shape ("all keys need
    ///     to be the same shape"), a 0-d key, a null entry, and no key ("need sequence of keys with len > 0").
    /// </summary>
    /// <param name="label">The variant (<see cref="LexKeysVariantLabels"/>).</param>
    /// <param name="operand">The pool operand.</param>
    /// <returns>A fresh key list (its first entry is the operand itself, but for "none").</returns>
    /// <exception cref="ArgumentException">An unknown label (a bug in the tool).</exception>
    /// <summary>The variants of np.evaluate's expression, the base first (see <see cref="MakeEvalExpr"/>).</summary>
    private static readonly string[] EvalExprVariantLabels =
    {
        "arr", "arr*2", "sum", "sum_ax0", "ptp", "ptp_ax0", "ptp_kd", "ptp(arr*2)", "nanmin", "nanmin_ax0", "nanmax",
        "nanmax_ax0_kd", "nanmin(arr*2)", "argmax", "argmax_ax0", "argmax_ax0_kd", "argmin", "argmin_ax0",
        "argmax(arr*2)", "argmax(arr*2)_ax0", "argmin(arr*2)_ax0_kd", "any", "all_ax0",
    };

    /// <summary>
    ///     Builds np.evaluate's expression over the operand: the identity ("arr"), a computed child ("arr*2", the operand
    ///     times a weak 2), and the root reductions over the leaf or the computed child - sum, ptp, nanmin / nanmax,
    ///     argmax / argmin, any / all - flat, along axis 0 ("_ax0") and with keepdims ("_kd"). The delegating kinds (ptp,
    ///     nanmin / nanmax, argmax / argmin) reach the engine's own reductions over the materialized child, the leaf forms of
    ///     argmax / argmin its streamed leaf route.
    /// </summary>
    /// <param name="label">One of <see cref="EvalExprVariantLabels"/>.</param>
    /// <param name="operand">The pool operand.</param>
    /// <param name="positional">Build over <c>NDExpr.Input(0)</c> (the positional overload, whose operand list holds the
    ///     operand) instead of an embedded <c>NDExpr.Arr(operand)</c>.</param>
    /// <returns>The expression.</returns>
    /// <exception cref="ArgumentException">An unknown label.</exception>
    private static NDExpr MakeEvalExpr(string label, NDArray operand, bool positional)
    {
        NDExpr leaf = positional ? NDExpr.Input(0) : NDExpr.Arr(operand);
        NDExpr twice = leaf * 2;
        return label switch
        {
            "arr" => leaf,
            "arr*2" => twice,
            "sum" => NDExpr.Sum(leaf),
            "sum_ax0" => NDExpr.Sum(leaf, 0),
            "ptp" => NDExpr.Ptp(leaf),
            "ptp_ax0" => NDExpr.Ptp(leaf, 0),
            "ptp_kd" => NDExpr.Ptp(leaf, true),
            "ptp(arr*2)" => NDExpr.Ptp(twice),
            "nanmin" => NDExpr.NanMin(leaf),
            "nanmin_ax0" => NDExpr.NanMin(leaf, 0),
            "nanmax" => NDExpr.NanMax(leaf),
            "nanmax_ax0_kd" => NDExpr.NanMax(leaf, 0, true),
            "nanmin(arr*2)" => NDExpr.NanMin(twice),
            "argmax" => NDExpr.ArgMax(leaf),
            "argmax_ax0" => NDExpr.ArgMax(leaf, 0),
            "argmax_ax0_kd" => NDExpr.ArgMax(leaf, 0, true),
            "argmin" => NDExpr.ArgMin(leaf),
            "argmin_ax0" => NDExpr.ArgMin(leaf, 0),
            "argmax(arr*2)" => NDExpr.ArgMax(twice),
            "argmax(arr*2)_ax0" => NDExpr.ArgMax(twice, 0),
            "argmin(arr*2)_ax0_kd" => NDExpr.ArgMin(twice, 0, true),
            "any" => NDExpr.Any(leaf),
            "all_ax0" => NDExpr.All(leaf, 0),
            _ => throw new ArgumentException($"unknown np.evaluate expression variant {label}"),
        };
    }

    private static NDArray[] MakeLexKeys(string label, NDArray operand)
    {
        long[] dims = operand.Shape.Dimensions;
        long size = 1;
        foreach (long d in dims) size *= d;
        switch (label)
        {
            case "one": return new[] { operand };
            case "pair": return new[] { operand, operand.copy() };
            case "neg":
            {
                // The negation computes in the operand's dtype; a dtype negation refuses (bool, text) leaves a copy, which
                // still exercises the two-key passes.
                NDArray second;
                try { second = np.negative(operand); }
                catch { second = operand.copy(); }
                return new[] { operand, second };
            }
            case "int32": return new[] { operand, np.arange(size).astype(np.dtype("int32")).reshape(new Shape(dims)) };
            case "text":
            {
                var words = new string[size];
                for (long i = 0; i < size; i++)
                    words[i] = ((size - i) % 3).ToString(CultureInfo.InvariantCulture);
                return new[] { operand, np.array(words).astype(np.dtype("U5")).reshape(new Shape(dims)) };
            }
            case "mismatch": return new[] { operand, np.arange(size + 1).astype(np.dtype("int64")) };
            case "zero_d": return new[] { operand, np.array(2.0) };
            case "nullkey": return new[] { operand, null };
            case "none": return Array.Empty<NDArray>();
            default: throw new ArgumentException("unknown lexsort key-list variant " + label);
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
            case np.MGridResult mg:
                // np.mgrid's result is NumPy's ndarray (the dense grid, or a single slice's 1-D array): rendered as that
                // array, so the line reads like NumPy's.
                return Render(mg.Grid, outArray);
            case np.OGridResult og:
                // np.ogrid's result is NumPy's bare ndarray for one slice (its except branch) and its tuple otherwise (the
                // empty key's empty tuple included).
                return og.Length == 1 ? Render(og[0], outArray) : Render(og.ToArray(), outArray);
            case np.NDIterator iterator:
                // Before the IEnumerable case: an iterator is enumerable, and its per-step views rendered one by one would
                // make lines of kilobytes.
                return RenderIterator(iterator);
            case Array md when md.Rank > 1:
                // A rectangular C# array (an NDArray's explicit conversion to Array): its element type and dimensions, which
                // the flat enumeration below would lose, then its elements in C order.
                return "MDARRAY " + md.GetType().GetElementType().Name + "[" +
                       string.Join(",", Enumerable.Range(0, md.Rank).Select(md.GetLength)) + "] LIST[" +
                       string.Join("; ", md.Cast<object>().Select(o => Render(o, outArray))) + "]";
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
