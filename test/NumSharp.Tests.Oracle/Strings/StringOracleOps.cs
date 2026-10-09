using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's op table: each corpus op key is replayed as the C# call a ported NumPy line makes
    ///     (<c>np.copy(a, order='F')</c> → <c>np.copy(a, 'F')</c>), with the same arguments the generator passed NumPy.
    /// </summary>
    /// <remarks>
    ///     The table reads its arguments from <see cref="StringOracleCorpus.Case.Params"/>: plain JSON numbers / bools /
    ///     lists for shapes, axes, indices and masks, and the generator's text encoding (<c>{"u"}</c> / <c>{"b"}</c>) for
    ///     values (<see cref="StringOracleCorpus.Decode"/>). A C# <see cref="string"/> is NumPy's <c>str</c> and a
    ///     <c>byte[]</c> NumPy's <c>bytes</c> (<c>NDArray.Scalar(byte[])</c> is the 0-d <c>S</c> array).
    /// </remarks>
    internal static partial class StringOracleOps
    {
        /// <summary>Replays one case.</summary>
        /// <param name="c">The case.</param>
        /// <param name="ops">Its reconstructed operands.</param>
        /// <returns>The call's result (an array, a tuple, a value, a text, a bool, a dtype) or the mutated operand.</returns>
        /// <exception cref="InvalidOperationException">An op key the table does not know (a generator / table mismatch).</exception>
        internal static object Invoke(StringOracleCorpus.Case c, NDArray[] ops)
        {
            var p = c.Params ?? new Dictionary<string, JsonElement>();
            NDArray a = ops.Length > 0 ? ops[0] : null;
            NDArray b = ops.Length > 1 ? ops[1] : null;
            switch (c.Op)
            {
                // ---- descriptors ----
                case "dtype_props": return np.dtype(Str(p, "spec"));
                case "promote_types": return np.promote_types(np.dtype(Str(p, "a")), np.dtype(Str(p, "b")));
                case "result_type": return np.result_type(p["specs"].EnumerateArray().Select(e => e.GetString()).ToArray());
                case "can_cast": return np.can_cast(np.dtype(Str(p, "from")), np.dtype(Str(p, "to")), Str(p, "casting"));
                case "issubdtype": return np.issubdtype(Str(p, "a"), Str(p, "b"));

                // ---- creation ----
                case "array":
                {
                    var values = StringOracleCorpus.Decode(p["values"]);
                    var managed = ManagedArray(values);
                    var dt = Dt(p, "dtype");
                    int ndmin = p.TryGetValue("ndmin", out var nm) ? nm.GetInt32() : 0;
                    char order = p.TryGetValue("order", out var o) && o.ValueKind == JsonValueKind.String ? o.GetString()[0] : 'C';
                    return np.array(managed, dt, ndmin, true, order);
                }
                case "array_scalar":
                {
                    var v = StringOracleCorpus.Decode(p["value"]);
                    var dt = Dt(p, "dtype");
                    if (v is string s)
                        return dt is null ? np.array(s) : np.array(s, dt);
                    return dt is null ? NDArray.Scalar(v) : NDArray.Scalar(v, dt);
                }
                case "zeros": return np.zeros(ShapeOf(p, "shape"), np.dtype(Str(p, "dtype")));
                case "empty": return np.empty(ShapeOf(p, "shape"), np.dtype(Str(p, "dtype")));
                case "ones": return np.ones(ShapeOf(p, "shape"), np.dtype(Str(p, "dtype")));
                case "full": return np.full(ShapeOf(p, "shape"), StringOracleCorpus.Decode(p["fill"]), Dt(p, "dtype"));
                case "zeros_like": return np.zeros_like(a, Dt(p, "dtype"));
                case "empty_like": return np.empty_like(a, Dt(p, "dtype"));
                case "ones_like": return np.ones_like(a);
                case "full_like": return np.full_like(a, StringOracleCorpus.Decode(p["fill"]), Dt(p, "dtype"));
                case "asarray": return np.asarray(a, Dt(p, "dtype"));
                case "ascontiguousarray": return np.ascontiguousarray(a);
                case "asfortranarray": return np.asfortranarray(a);
                case "frombuffer":
                    return np.frombuffer(Convert.FromHexString(Str(p, "buffer")), np.dtype(Str(p, "dtype")),
                                         p["count"].GetInt64(), p["offset"].GetInt64());
                case "frombuffer_str":
                    // The dtype-STRING overload: its own parser must route every string spelling ('c', 'a3', '|S2') to the
                    // descriptor route, exactly as np.dtype reads it.
                    return np.frombuffer(Convert.FromHexString(Str(p, "buffer")), Str(p, "dtype"),
                                         p["count"].GetInt64(), p["offset"].GetInt64());
                case "ndarray":
                {
                    // np.ndarray(shape, dtype, buffer=..., offset, strides, order): the buffer is the hex bytes (NumPy got a
                    // bytearray, a C# byte[] is the same mutable buffer) or the case's string operand.
                    object buffer = a is not null ? a : Convert.FromHexString(Str(p, "buffer"));
                    long[] strides = p.TryGetValue("strides", out var st) && st.ValueKind == JsonValueKind.Array ? Longs(st) : null;
                    char order = p.TryGetValue("order", out var od) && od.ValueKind == JsonValueKind.String ? od.GetString()[0] : 'C';
                    return np.ndarray(ShapeOf(p, "shape"), np.dtype(Str(p, "dtype")), buffer, p["offset"].GetInt64(), strides, order);
                }

                // ---- casts ----
                case "astype":
                    return a.astype(np.dtype(Str(p, "dtype")), p["copy"].GetBoolean(), Str(p, "order")[0], Str(p, "casting"));

                // ---- views ----
                case "transpose": return np.transpose(a);
                case "swapaxes": return np.swapaxes(a, Int(p, "a1"), Int(p, "a2"));
                case "moveaxis": return np.moveaxis(a, Int(p, "src"), Int(p, "dst"));
                case "rot90": return np.rot90(a, Int(p, "k"));
                case "diagonal": return np.diagonal(a, Int(p, "offset"));
                case "ravel": return np.ravel(a, Str(p, "order")[0]);
                case "reshape": return a.reshape(ShapeOf(p, "shape"), Str(p, "order")[0]);
                case "squeeze": return np.squeeze(a);
                case "expand_dims": return np.expand_dims(a, Int(p, "axis"));
                case "flip": return NullableInt(p, "axis") is int ax ? np.flip(a, ax) : np.flip(a);
                case "atleast_2d": return np.atleast_2d(a);
                case "atleast_3d": return np.atleast_3d(a);
                case "broadcast_to": return np.broadcast_to(a, ShapeOf(p, "shape"));
                case "view": return a.view(np.dtype(Str(p, "dtype")));
                case "getitem": return a[Index(p["index"])];

                // ---- copies and movement ----
                case "copy": return np.copy(a, Str(p, "order")[0]);
                case "flatten": return a.flatten(Str(p, "order")[0]);
                case "repeat":
                {
                    var axis = NullableInt(p, "axis");
                    return p["repeats"].ValueKind == JsonValueKind.Array
                        ? np.repeat(a, np.array(Longs(p["repeats"])), axis)
                        : np.repeat(a, p["repeats"].GetInt32(), axis);
                }
                case "tile": return np.tile(a, Longs(p["reps"]).Select(x => (int)x).ToArray());
                case "roll": return np.roll(a, Int(p, "shift"), NullableInt(p, "axis"));
                case "resize": return np.resize(a, ShapeOf(p, "shape"));
                case "take":
                    return np.take(a, np.array(Longs(p["indices"])), NullableInt(p, "axis"), null, Str(p, "mode"));
                case "take_along_axis": return np.take_along_axis(a, RawArray(p["indices"]), Int(p, "axis"));
                case "compress": return np.compress(RawArray(p["condition"]), a, NullableInt(p, "axis"));
                case "extract": return np.extract(RawArray(p["condition"]), a);
                case "delete": return np.delete(a, Longs(p["obj"]).Select(x => (int)x).ToArray(), NullableInt(p, "axis"));
                case "insert": return np.insert(a, Int(p, "obj"), ToOperand(StringOracleCorpus.Decode(p["values"])), NullableInt(p, "axis"));
                case "append": return np.append(a, ManagedArray(StringOracleCorpus.Decode(p["values"])), NullableInt(p, "axis"));
                case "tril": return np.tril(a, Int(p, "k"));
                case "triu": return np.triu(a, Int(p, "k"));
                case "diag": return np.diag(a, Int(p, "k"));
                case "diagflat": return np.diagflat(a, Int(p, "k"));
                case "split": return np.array_split(a, Int(p, "sections"), Int(p, "axis"));
                case "pad": return Pad(a, p);
                case "concatenate": return np.concatenate(ops, NullableInt(p, "axis"));
                case "stack": return np.stack(ops, Int(p, "axis"));
                case "vstack": return np.vstack(ops);
                case "hstack": return np.hstack(ops);
                case "dstack": return np.dstack(ops);
                case "column_stack": return np.column_stack(ops);
                case "where": return np.where(RawArray(p["cond"]), a, b);
                case "where_scalar":
                {
                    var cond = RawArray(p["cond"]);
                    object x = p["x"].ValueKind == JsonValueKind.Null ? a : StringOracleCorpus.Decode(p["x"]);
                    object y = p["y"].ValueKind == JsonValueKind.Null ? a : StringOracleCorpus.Decode(p["y"]);
                    return np.where(cond, ToOperand(x), ToOperand(y));
                }
                case "choose": return np.choose(RawArray(p["index"]), ops);
                case "meshgrid": return (NDArray[])np.meshgrid(a, b);
                case "broadcast_arrays": return np.broadcast_arrays(ops);
                case "unstack": return np.unstack(a, Int(p, "axis"));

                // ---- in-place writes (the mutated operand is the result) ----
                case "setitem":
                    a[Index(p["index"])] = ToOperand(StringOracleCorpus.Decode(p["value"]));
                    return a;
                case "setitem_array":
                    a[Index(p["index"])] = b;
                    return a;
                case "fill":
                    a.fill(StringOracleCorpus.Decode(p["value"]));
                    return a;
                case "put":
                    np.put(a, np.array(Longs(p["indices"])), ToOperand(ManagedArray(StringOracleCorpus.Decode(p["values"]))), Str(p, "mode"));
                    return a;
                // NumPy got a Python LIST of values: a C# array binds the list overloads (converted straight into a's dtype).
                case "place":
                    np.place(a, RawArray(p["mask"]), ManagedArray(StringOracleCorpus.Decode(p["values"])));
                    return a;
                case "putmask":
                    np.putmask(a, RawArray(p["mask"]), ManagedArray(StringOracleCorpus.Decode(p["values"])));
                    return a;
                case "copyto":
                    np.copyto(a, b, Str(p, "casting"), p["where"].ValueKind == JsonValueKind.Null ? null : RawArray(p["where"]));
                    return a;
                case "put_along_axis":
                    np.put_along_axis(a, RawArray(p["indices"]), b, Int(p, "axis"));
                    return a;

                // ---- element access ----
                case "tolist": return a.tolist();
                case "item": return NullableInt(p, "index") is int i ? a.item(i) : a.item();
                case "iter":
                {
                    var list = new List<object>();
                    foreach (var x in a)
                        list.Add(x);
                    return list;
                }

                // ---- printing ----
                case "repr": return a.ToString(true);
                case "str": return a.ToString(false);
                case "repr_opts":
                    using (np.printoptions(threshold: Int(p, "threshold"), edgeitems: Int(p, "edgeitems"), linewidth: Int(p, "linewidth")))
                        return a.ToString(true);
                case "str_opts":
                    using (np.printoptions(threshold: Int(p, "threshold"), edgeitems: Int(p, "edgeitems"), linewidth: Int(p, "linewidth")))
                        return a.ToString(false);
                case "array2string":
                    return np.array2string(a, max_line_width: Int(p, "max_line_width"), separator: Str(p, "separator"),
                                           prefix: Str(p, "prefix"));
                case "array_repr": return np.array_repr(a, max_line_width: Int(p, "max_line_width"));
                case "array_str": return np.array_str(a, max_line_width: Int(p, "max_line_width"));

                // ---- ufuncs and the calls NumPy refuses or computes on text ----
                case "ufunc": return Ufunc(Str(p, "name"), ops, null);
                case "ufunc_dtype": return Ufunc(Str(p, "name"), ops, np.dtype(Str(p, "dtype")));
                case "operator":
                    switch (Str(p, "op"))
                    {
                        case "==": return a == b;
                        case "!=": return a != b;
                        case "<": return a < b;
                    }
                    break;
                case "call": return Call(Str(p, "name"), a);
            }
            // Section J (string dtype= requests and the order of a text call's problems) lives in its own partial.
            if (TryInvokePrecedence(c.Op, p, ops, out var precedence))
                return precedence;
            // Sections K and L (answers the shape decides, the checks before a conversion) live in their own partial.
            if (TryInvokeShape(c.Op, p, ops, out var shaped))
                return shaped;
            // Section M (the rest of the surface: linalg, products, random, histograms, ...) lives in its own partial.
            if (TryInvokeSurface(c.Op, p, ops, out var surface))
                return surface;
            // Section N (numpy.ma over text) builds masked arguments from data / mask operand pairs in its own partial.
            if (TryInvokeMa(c.Op, p, ops, out var masked))
                return masked;
            // Section striter (np.nditer / np.nested_iters over text, recorded as traces) lives in its own partial.
            if (TryInvokeNditer(c.Op, p, ops, out var iterated))
                return iterated;
            // Section strcomp's own ops (the C# string operators, the truth reductions along an axis) live in their partial.
            if (TryInvokeCompute(c.Op, p, ops, out var computed))
                return computed;
            throw new InvalidOperationException($"the string oracle has no op '{c.Op}'");
        }

        /// <summary>np.pad with the generator's arguments: an int or per-axis pairs, a mode, an optional text constant.</summary>
        /// <param name="a">The array.</param>
        /// <param name="p">The case parameters.</param>
        /// <returns>The padded array.</returns>
        private static NDArray Pad(NDArray a, Dictionary<string, JsonElement> p)
        {
            string mode = Str(p, "mode");
            object constant = p["constant"].ValueKind == JsonValueKind.Null ? null : StringOracleCorpus.Decode(p["constant"]);
            var pw = p["pad_width"];
            if (pw.ValueKind == JsonValueKind.Number)
                return constant is null ? np.pad(a, pw.GetInt32(), mode) : np.pad(a, pw.GetInt32(), mode, constant_values: constant);
            var pairs = pw.EnumerateArray().Select(e => e.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray();
            var grid = new int[pairs.Length, 2];
            for (int i = 0; i < pairs.Length; i++)
            {
                grid[i, 0] = pairs[i][0];
                grid[i, 1] = pairs[i][1];
            }
            return constant is null ? np.pad(a, grid, mode) : np.pad(a, grid, mode, constant_values: constant);
        }

        /// <summary>A ufunc by NumPy name, with an optional loop dtype (<c>dtype=</c>).</summary>
        /// <param name="name">The ufunc name.</param>
        /// <param name="ops">One or two operands.</param>
        /// <param name="dt">The <c>dtype=</c> argument, or null.</param>
        /// <returns>The result (an array, or a tuple for the two-output ufuncs).</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static object Ufunc(string name, NDArray[] ops, DType dt)
        {
            var x = ops[0];
            var y = ops.Length > 1 ? ops[1] : null;
            switch (name)
            {
                case "add": return np.add(x, y, dtype: dt);
                case "subtract": return np.subtract(x, y, dtype: dt);
                case "multiply": return np.multiply(x, y, dtype: dt);
                case "divide": return np.divide(x, y, dtype: dt);
                case "true_divide": return np.true_divide(x, y, dtype: dt);
                case "floor_divide": return np.floor_divide(x, y, dtype: dt);
                case "power": return np.power(x, y, dtype: dt);
                case "float_power": return np.float_power(x, y, dtype: dt);
                case "remainder": return np.remainder(x, y, dtype: dt);
                case "mod": return np.mod(x, y, dtype: dt);
                case "fmod": return np.fmod(x, y, dtype: dt);
                case "divmod": return np.divmod(x, y, dtype: dt);
                case "maximum": return np.maximum(x, y, dtype: dt);
                case "minimum": return np.minimum(x, y, dtype: dt);
                case "fmax": return np.fmax(x, y, dtype: dt);
                case "fmin": return np.fmin(x, y, dtype: dt);
                case "arctan2": return np.arctan2(x, y, dtype: dt);
                case "hypot": return np.hypot(x, y, dtype: dt);
                case "copysign": return np.copysign(x, y, dtype: dt);
                case "nextafter": return np.nextafter(x, y, dtype: dt);
                case "logaddexp": return np.logaddexp(x, y, dtype: dt);
                case "logaddexp2": return np.logaddexp2(x, y, dtype: dt);
                case "heaviside": return np.heaviside(x, y, dtype: dt);
                case "ldexp": return np.ldexp(x, y);
                case "gcd": return np.gcd(x, y, dtype: dt);
                case "lcm": return np.lcm(x, y, dtype: dt);
                case "bitwise_and": return np.bitwise_and(x, y, dtype: dt);
                case "bitwise_or": return np.bitwise_or(x, y, dtype: dt);
                case "bitwise_xor": return np.bitwise_xor(x, y, dtype: dt);
                case "left_shift": return np.left_shift(x, y, dtype: dt);
                case "right_shift": return np.right_shift(x, y, dtype: dt);
                case "matmul": return np.matmul(x, y, dtype: dt);
                case "equal": return np.equal(x, y, dtype: dt);
                case "not_equal": return np.not_equal(x, y, dtype: dt);
                case "less": return np.less(x, y, dtype: dt);
                case "less_equal": return np.less_equal(x, y, dtype: dt);
                case "greater": return np.greater(x, y, dtype: dt);
                case "greater_equal": return np.greater_equal(x, y, dtype: dt);
                case "logical_and": return np.logical_and(x, y, dtype: dt);
                case "logical_or": return np.logical_or(x, y, dtype: dt);
                case "logical_xor": return np.logical_xor(x, y, dtype: dt);
                case "negative": return np.negative(x, dtype: dt);
                case "positive": return np.positive(x, dtype: dt);
                case "absolute": return np.absolute(x, dtype: dt);
                case "fabs": return np.fabs(x, dtype: dt);
                case "sign": return np.sign(x, dtype: dt);
                case "sqrt": return np.sqrt(x, dtype: dt);
                case "cbrt": return np.cbrt(x, dtype: dt);
                case "square": return np.square(x, dtype: dt);
                case "reciprocal": return np.reciprocal(x, dtype: dt);
                case "exp": return np.exp(x, dtype: dt);
                case "exp2": return np.exp2(x, dtype: dt);
                case "expm1": return np.expm1(x, dtype: dt);
                case "log": return np.log(x, dtype: dt);
                case "log2": return np.log2(x, dtype: dt);
                case "log10": return np.log10(x, dtype: dt);
                case "log1p": return np.log1p(x, dtype: dt);
                case "sin": return np.sin(x, dtype: dt);
                case "cos": return np.cos(x, dtype: dt);
                case "tan": return np.tan(x, dtype: dt);
                case "arcsin": return np.arcsin(x, dtype: dt);
                case "arccos": return np.arccos(x, dtype: dt);
                case "arctan": return np.arctan(x, dtype: dt);
                case "sinh": return np.sinh(x, dtype: dt);
                case "cosh": return np.cosh(x, dtype: dt);
                case "tanh": return np.tanh(x, dtype: dt);
                case "arcsinh": return np.arcsinh(x, dtype: dt);
                case "arccosh": return np.arccosh(x, dtype: dt);
                case "arctanh": return np.arctanh(x, dtype: dt);
                case "deg2rad": return np.deg2rad(x, dtype: dt);
                case "rad2deg": return np.rad2deg(x, dtype: dt);
                case "degrees": return np.degrees(x, dtype: dt);
                case "radians": return np.radians(x, dtype: dt);
                case "floor": return np.floor(x, dtype: dt);
                case "ceil": return np.ceil(x, dtype: dt);
                case "trunc": return np.trunc(x, dtype: dt);
                case "rint": return np.rint(x, dtype: dt);
                case "isnan": return np.isnan(x, dtype: dt);
                case "isinf": return np.isinf(x, dtype: dt);
                case "isfinite": return np.isfinite(x, dtype: dt);
                case "signbit": return np.signbit(x, dtype: dt);
                case "logical_not": return np.logical_not(x, dtype: dt);
                case "invert": return np.invert(x, dtype: dt);
                case "bitwise_not": return np.bitwise_not(x, dtype: dt);
                case "spacing": return np.spacing(x, dtype: dt);
                case "bitwise_count": return np.bitwise_count(x, dtype: dt);
                case "modf": return np.modf(x, dtype: dt);
                case "frexp": return np.frexp(x);
                case "conjugate": return np.conjugate(x, dtype: dt);
            }
            throw new InvalidOperationException($"the string oracle has no ufunc '{name}'");
        }

        /// <summary>The single-array calls of the refusal table, spelled as the generator's <c>_call_named</c> spells them.</summary>
        /// <param name="name">The NumPy function name.</param>
        /// <param name="x">The string array.</param>
        /// <returns>The call's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static object Call(string name, NDArray x)
        {
            var other = x.copy();
            switch (name)
            {
                case "sum": return np.sum(x);
                case "prod": return np.prod(x);
                case "mean": return np.mean(x);
                case "std": return np.std(x);
                case "var": return np.var(x);
                case "max": return np.max(x);
                case "min": return np.min(x);
                case "ptp": return np.ptp(x);
                case "cumsum": return np.cumsum(x);
                case "cumprod": return np.cumprod(x);
                case "median": return np.median(x);
                case "average": return np.average(x);
                case "nansum": return np.nansum(x);
                case "nanprod": return np.nanprod(x);
                case "nanmean": return np.nanmean(x);
                case "nanstd": return np.nanstd(x);
                case "nanvar": return np.nanvar(x);
                case "nanmax": return np.nanmax(x);
                case "nanmin": return np.nanmin(x);
                case "nanmedian": return np.nanmedian(x);
                case "nancumsum": return np.nancumsum(x);
                case "nancumprod": return np.nancumprod(x);
                case "percentile": return np.percentile(x, 50);
                case "quantile": return np.quantile(x, 0.5);
                case "nanpercentile": return np.nanpercentile(x, 50);
                case "nanquantile": return np.nanquantile(x, 0.5);
                case "argmax": return np.argmax(x);
                case "argmin": return np.argmin(x);
                case "all": return np.all(x);
                case "any": return np.any(x);
                case "count_nonzero": return np.count_nonzero(x);
                case "nonzero": return np.nonzero(x);
                case "flatnonzero": return np.flatnonzero(x);
                case "argwhere": return np.argwhere(x);
                case "sort": return np.sort(x);
                case "argsort": return np.argsort(x);
                case "unique": return (NDArray)np.unique(x);
                case "trim_zeros": return np.trim_zeros(x);
                case "diff": return np.diff(x);
                case "ediff1d": return np.ediff1d(x);
                case "gradient": return np.gradient(x);
                case "trace": return np.trace(x.reshape(2, 3));
                case "cov": return np.cov(x);
                case "corrcoef": return np.corrcoef(x);
                case "sinc": return np.sinc(x);
                case "angle": return np.angle(x);
                case "i0": return np.i0(x);
                case "unwrap": return np.unwrap(x);
                case "vander": return np.vander(x, 2);
                case "round": return np.round_(x);
                case "around": return np.around(x);
                // NumPy's x[0] is a SCALAR (np.str_), whose descriptor is its own text length: NDArray.Scalar of the element.
                case "clip": return np.clip(x, NDArray.Scalar(x.item(0)), NDArray.Scalar(x.item(1)));
                case "isreal": return np.isreal(x);
                case "iscomplex": return np.iscomplex(x);
                case "real": return np.real(x);
                case "imag": return np.imag(x);
                case "nan_to_num": return np.nan_to_num(x);
                case "real_if_close": return np.real_if_close(x);
                case "fix": return np.fix(x);
                case "histogram": return np.histogram(x);
                case "lexsort": return np.lexsort(new[] { x });
                case "searchsorted": return np.searchsorted(np.sort(x), x[0]);
                case "partition": return np.partition(x, 1);
                case "argpartition": return np.argpartition(x, 1);
                case "isin": return np.isin(x, other[":2"]);
                case "union1d": return np.union1d(x, other);
                case "intersect1d": return np.intersect1d(x, other);
                case "setdiff1d": return np.setdiff1d(x, other[":2"]);
                case "setxor1d": return np.setxor1d(x, other[":2"]);
                case "array_equal": return np.array_equal(x, other);
                case "array_equiv": return np.array_equiv(x, other);
                case "isclose": return np.isclose(x, other);
                case "allclose": return np.allclose(x, other);
                case "linalg.norm": return np.linalg.norm(x);
                case "fft.fft": return np.fft.fft(x);
                case "nanargmax": return np.nanargmax(x);
                case "nanargmin": return np.nanargmin(x);
                case "logical_not": return np.logical_not(x);
                case "bincount": return np.bincount(x);
                case "digitize": return np.digitize(np.arange(3.0), x);
                case "interp": return np.interp(np.arange(3.0), x, x);
                case "polyval": return np.polyval(x, 2.0);
                case "trapezoid": return np.trapezoid(x);
                case "convolve": return np.convolve(x, other);
                case "correlate": return np.correlate(x, other);
                case "dot": return np.dot(x, other);
                case "inner": return np.inner(x, other);
                case "vdot": return np.vdot(x, other);
                case "outer": return np.outer(x, other);
                case "kron": return np.kron(x, other);
                case "tensordot": return np.tensordot(x, other, 1);
                case "cross": return np.cross(x[":3"], other[":3"]);
                case "einsum": return np.einsum("i,i->i", x, other);
                case "byteswap": return x.byteswap();
                case "conj": return np.conj(x);
            }
            throw new InvalidOperationException($"the string oracle has no call '{name}'");
        }

        // ---- argument helpers ----

        /// <summary>A string parameter.</summary>
        private static string Str(Dictionary<string, JsonElement> p, string key) => p[key].GetString();

        /// <summary>An int parameter.</summary>
        private static int Int(Dictionary<string, JsonElement> p, string key) => p[key].GetInt32();

        /// <summary>An int-or-None parameter (absent counts as None).</summary>
        private static int? NullableInt(Dictionary<string, JsonElement> p, string key)
            => p.TryGetValue(key, out var e) && e.ValueKind == JsonValueKind.Number ? e.GetInt32() : null;

        /// <summary>A dtype-or-None parameter.</summary>
        private static DType Dt(Dictionary<string, JsonElement> p, string key)
            => p.TryGetValue(key, out var e) && e.ValueKind == JsonValueKind.String ? np.dtype(e.GetString()) : null;

        /// <summary>A shape parameter (a JSON list of ints).</summary>
        private static Shape ShapeOf(Dictionary<string, JsonElement> p, string key) => new Shape(Longs(p[key]));

        /// <summary>A flat JSON list of ints, or one int, as longs.</summary>
        private static long[] Longs(JsonElement e)
            => e.ValueKind == JsonValueKind.Array ? e.EnumerateArray().Select(x => x.GetInt64()).ToArray() : new[] { e.GetInt64() };

        /// <summary>
        ///     A nested JSON list of plain bools or ints (masks, conditions, indices) as an NDArray of bool or int64 with
        ///     the list's shape.
        /// </summary>
        /// <param name="e">The nested list.</param>
        /// <returns>The array.</returns>
        private static NDArray RawArray(JsonElement e)
        {
            var dims = new List<long>();
            var leaves = new List<JsonElement>();
            Flatten(e, 0, dims, leaves);
            bool isBool = leaves.Count > 0 && (leaves[0].ValueKind == JsonValueKind.True || leaves[0].ValueKind == JsonValueKind.False);
            var shape = new Shape(dims.ToArray());
            if (isBool)
            {
                var data = leaves.Select(x => x.GetBoolean()).ToArray();
                return np.array(data).reshape(shape);
            }
            return np.array(leaves.Select(x => x.GetInt64()).ToArray()).reshape(shape);
        }

        /// <summary>Walks a rectangular nested JSON list, recording its dimensions and its leaves in C order.</summary>
        private static void Flatten(JsonElement e, int depth, List<long> dims, List<JsonElement> leaves)
        {
            if (e.ValueKind != JsonValueKind.Array)
            {
                leaves.Add(e);
                return;
            }
            int n = e.GetArrayLength();
            if (dims.Count == depth)
                dims.Add(n);
            foreach (var x in e.EnumerateArray())
                Flatten(x, depth + 1, dims, leaves);
        }

        /// <summary>
        ///     A decoded value as the C# argument a ported NumPy line passes: an array stays an array, a string a string
        ///     (the implicit conversion makes it the 0-d <c>U</c> array), a <c>byte[]</c> the 0-d <c>S</c> array, a number a
        ///     0-d numeric array.
        /// </summary>
        /// <param name="v">The value.</param>
        /// <returns>The NDArray operand.</returns>
        private static NDArray ToOperand(object v) => v switch
        {
            NDArray nd => nd,
            string s => s,
            byte[] bytes => NDArray.Scalar(bytes),
            Array arr => np.array(arr),
            long l => NDArray.Scalar(l),
            double d => NDArray.Scalar(d),
            bool bo => NDArray.Scalar(bo),
            _ => throw new InvalidOperationException($"no operand for {v?.GetType().Name ?? "null"}"),
        };

        /// <summary>
        ///     A decoded value list as the C# array a ported NumPy line passes: a rectangular nest of strings becomes
        ///     <c>string[]</c> / <c>string[,]</c> / <c>string[,,]</c>; any other rectangular nest (bytes, mixed, numbers,
        ///     None) an <c>object</c> array of the same rank; a ragged nest a jagged <c>object[]</c> of those. A
        ///     non-list value is returned as is.
        /// </summary>
        /// <param name="v">The decoded value.</param>
        /// <returns>The managed array (or the value).</returns>
        internal static object ManagedArrayOrValue(object v) => v is List<object> ? ManagedArray(v) : v;

        /// <summary>See <see cref="ManagedArrayOrValue"/>; <paramref name="v"/> must be a decoded list.</summary>
        /// <param name="v">The decoded list.</param>
        /// <returns>The managed array.</returns>
        internal static Array ManagedArray(object v)
        {
            if (v is not List<object> top)
                throw new InvalidOperationException("a managed array needs a list");
            var dims = new List<int>();
            if (!TryRectangular(top, 0, dims))
                return top.Select(x => x is List<object> l ? (object)ManagedArray(l) : x).ToArray();
            var leaves = new List<object>();
            CollectLeaves(top, leaves);
            var elementType = leaves.Count > 0 && leaves.All(x => x is string) ? typeof(string) : typeof(object);
            var result = Array.CreateInstance(elementType, dims.ToArray());
            var coords = new int[dims.Count];
            foreach (var leaf in leaves)
            {
                result.SetValue(leaf, coords);
                for (int d = coords.Length - 1; d >= 0; d--)
                {
                    if (++coords[d] < dims[d])
                        break;
                    coords[d] = 0;
                }
            }
            return result;
        }

        /// <summary>Whether a decoded nest is rectangular; records its dimensions.</summary>
        private static bool TryRectangular(List<object> list, int depth, List<int> dims)
        {
            if (dims.Count == depth)
                dims.Add(list.Count);
            else if (dims[depth] != list.Count)
                return false;
            bool anyList = list.Any(x => x is List<object>);
            bool allList = list.All(x => x is List<object>);
            if (anyList && !allList)
                return false;
            if (!allList || list.Count == 0)
                return dims.Count == depth + 1 || list.Count == 0;
            foreach (List<object> child in list)
                if (!TryRectangular(child, depth + 1, dims))
                    return false;
            return true;
        }

        /// <summary>The leaves of a decoded nest in C order.</summary>
        private static void CollectLeaves(List<object> list, List<object> leaves)
        {
            foreach (var x in list)
                if (x is List<object> l)
                    CollectLeaves(l, leaves);
                else
                    leaves.Add(x);
        }

        /// <summary>
        ///     A NumPy index expression as the C# indexer argument: <c>{"int"}</c> → int, <c>{"slice"}</c> →
        ///     <see cref="Slice"/>, <c>{"none"}</c> → <see cref="Slice.NewAxis"/>, <c>{"ellipsis"}</c> →
        ///     <see cref="Slice.Ellipsis"/>, <c>{"arr"}</c> → an int64 index array, <c>{"mask"}</c> / <c>{"mask2"}</c> → a
        ///     bool mask.
        /// </summary>
        /// <param name="index">The encoded index tuple.</param>
        /// <returns>The indexer arguments.</returns>
        private static object[] Index(JsonElement index)
        {
            var parts = new List<object>();
            foreach (var e in index.EnumerateArray())
            {
                if (e.TryGetProperty("int", out var i))
                    parts.Add(i.GetInt32());
                else if (e.TryGetProperty("slice", out var s))
                {
                    var f = s.EnumerateArray().Select(x => x.ValueKind == JsonValueKind.Null ? (long?)null : x.GetInt64()).ToArray();
                    parts.Add(new Slice(f[0], f[1], f[2] ?? 1));
                }
                else if (e.TryGetProperty("none", out _))
                    parts.Add(Slice.NewAxis);
                else if (e.TryGetProperty("ellipsis", out _))
                    parts.Add(Slice.Ellipsis);
                else if (e.TryGetProperty("arr", out var arr))
                    parts.Add(RawArray(arr));
                else if (e.TryGetProperty("mask", out var m))
                    parts.Add(RawArray(m));
                else if (e.TryGetProperty("mask2", out var m2))
                    parts.Add(RawArray(m2));
                else
                    throw new InvalidOperationException($"unknown index entry {e}");
            }
            return parts.ToArray();
        }
    }
}
