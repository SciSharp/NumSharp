using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section N (<c>gen_strma</c>): <c>numpy.ma</c> over text — every module function
    ///     applied to masked text arrays (1-D, 2-D, empty, 0-d, bytes, all masked, with a fill value), the binary functions
    ///     over text / number / bytes pairs, and the constructors' text masks and fill values. Each op replays the C# call a
    ///     ported NumPy line makes, over masked arguments rebuilt the way the generator built them.
    /// </summary>
    /// <remarks>
    ///     The generator sends each masked argument as a data operand plus a boolean mask operand (or a <c>mask=True</c> /
    ///     <c>mask=False</c> flag, or nomask), a fill value and the hard-mask flag (<c>params["mops"]</c>); a plain array
    ///     argument as an operand; a Python value as an encoded value. <see cref="MaArgs"/> rebuilds the same objects with
    ///     <c>np.ma.array(data, mask:, fill_value:, hard_mask:)</c>.
    /// </remarks>
    internal static partial class StringOracleOps
    {
        /// <summary>Replays one section-N (or section-tma) op, or reports that <paramref name="op"/> is not one of them.</summary>
        /// <param name="op">The corpus op key: <c>ma0</c> (no argument), <c>ma1</c> (one masked argument), <c>ma2</c> (two
        ///     arguments) — shared by strma and tma, whose arguments arrive as operands — or <c>tma0</c> (section tma's
        ///     constructors, which build their <c>StringDType</c> arrays themselves; <see cref="TMa0"/>).</param>
        /// <param name="p">The case parameters (<c>name</c> selects the function, <c>mops</c> the arguments).</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <param name="result">The call's result.</param>
        /// <returns>True when <paramref name="op"/> is a section-N op.</returns>
        /// <exception cref="InvalidOperationException">A name the op does not know (a generator / table mismatch).</exception>
        private static bool TryInvokeMa(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            switch (op)
            {
                case "ma0":
                    result = Ma0(Str(p, "name"));
                    return true;
                case "tma0":
                    result = TMa0(Str(p, "name"));
                    return true;
                case "ma1":
                {
                    var args = MaArgs(p, ops);
                    result = Ma1(Str(p, "name"), (NDMaskedArray)args[0]);
                    return true;
                }
                case "ma2":
                {
                    var args = MaArgs(p, ops);
                    result = Ma2(Str(p, "name"), args[0], args[1]);
                    return true;
                }
                default:
                    result = null;
                    return false;
            }
        }

        /// <summary>
        ///     Rebuilds a case's arguments from <c>params["mops"]</c>: a masked array over its data operand (with its mask
        ///     operand, its <c>mask=True</c> / <c>mask=False</c> flag or no mask, its fill value and hardness), a plain
        ///     operand, or a decoded Python value.
        /// </summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The arguments, in call order.</returns>
        private static object[] MaArgs(Dictionary<string, JsonElement> p, NDArray[] ops)
        {
            var args = new List<object>();
            foreach (var e in p["mops"].EnumerateArray())
            {
                if (e.TryGetProperty("d", out var d))
                {
                    int m = e.GetProperty("m").GetInt32();
                    // Typed `object` on every arm: a `bool` arm next to an NDArray arm would convert through NumSharp's
                    // implicit scalar → NDArray conversion, turning NumPy's `mask=True` literal into a 0-d mask ARRAY
                    // (resized over the data — a view — where the literal is a fresh all-True array).
                    object mask = m >= 0 ? (object)ops[m] : e.TryGetProperty("mbool", out var mb) ? (object)mb.GetBoolean() : null;
                    object fill = e.TryGetProperty("fill", out var f) ? DecodeMaValue(f) : null;
                    bool hard = e.TryGetProperty("hard", out var h) && h.GetBoolean();
                    // np.ma.array(view, mask=mask, fill_value=fill, hard_mask=hard): the data aliases the operand.
                    args.Add(np.ma.array(ops[d.GetInt32()], mask: mask, fill_value: fill, hard_mask: hard));
                }
                else if (e.TryGetProperty("a", out var a))
                    args.Add(ops[a.GetInt32()]);
                else
                    args.Add(DecodeMaValue(e.GetProperty("v")));
            }
            return args.ToArray();
        }

        /// <summary>A Python value of section N: the corpus encodings plus a complex (<c>{"complex": [re, im]}</c>).</summary>
        /// <param name="e">The encoded value.</param>
        /// <returns>The C# value a ported line passes (a Python int as a <see cref="long"/>, a str as a <see cref="string"/>,
        ///     bytes as a <see cref="byte"/>[]).</returns>
        private static object DecodeMaValue(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.Object && e.TryGetProperty("complex", out var c))
                return new Complex(c[0].GetDouble(), c[1].GetDouble());
            return StringOracleCorpus.Decode(e);
        }

        /// <summary>An all-<c>true</c> / all-<c>false</c> boolean array of a masked argument's shape.</summary>
        private static NDArray Bools(NDMaskedArray x, bool value)
            => value ? np.ones(new Shape(x.shape), np.@bool) : np.zeros(new Shape(x.shape), np.@bool);

        /// <summary>The C# bytes of an ASCII literal (NumPy's <c>b'...'</c>).</summary>
        private static byte[] B(string ascii) => Encoding.ASCII.GetBytes(ascii);

        /// <summary>One masked text argument through the named <c>np.ma</c> call, as the generator's table makes it.</summary>
        /// <param name="name">The table key.</param>
        /// <param name="x">The masked argument.</param>
        /// <returns>The call's result (a mutating call returns its target).</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static object Ma1(string name, NDMaskedArray x)
        {
            var ma = np.ma;
            int first = x.ndim > 0 ? (int)x.shape[0] : 1;
            switch (name)
            {
                // ---- masked ufuncs ----
                case "abs": return ma.abs(x);
                case "absolute": return ma.absolute(x);
                case "negative": return ma.negative(x);
                case "fabs": return ma.fabs(x);
                case "conjugate": return ma.conjugate(x);
                case "angle": return ma.angle(x);
                case "around": return ma.around(x);
                case "floor": return ma.floor(x);
                case "ceil": return ma.ceil(x);
                case "exp": return ma.exp(x);
                case "sin": return ma.sin(x);
                case "cos": return ma.cos(x);
                case "sinh": return ma.sinh(x);
                case "cosh": return ma.cosh(x);
                case "tanh": return ma.tanh(x);
                case "arctan": return ma.arctan(x);
                case "arcsinh": return ma.arcsinh(x);
                case "logical_not": return ma.logical_not(x);
                case "sqrt": return ma.sqrt(x);
                case "log": return ma.log(x);
                case "log2": return ma.log2(x);
                case "log10": return ma.log10(x);
                case "tan": return ma.tan(x);
                case "arcsin": return ma.arcsin(x);
                case "arccos": return ma.arccos(x);
                case "arccosh": return ma.arccosh(x);
                case "arctanh": return ma.arctanh(x);

                // ---- reductions, scans, statistics ----
                case "sum": return ma.sum(x);
                case "prod": return ma.prod(x);
                case "product": return ma.product(x);
                case "min": return ma.min(x);
                case "max": return ma.max(x);
                case "amin": return ma.amin(x);
                case "amax": return ma.amax(x);
                case "ptp": return ma.ptp(x);
                case "count": return ma.count(x);
                case "mean": return ma.mean(x);
                case "var": return ma.var(x);
                case "std": return ma.std(x);
                case "anom": return ma.anom(x);
                case "anomalies": return ma.anomalies(x);
                case "cumsum": return ma.cumsum(x);
                case "cumprod": return ma.cumprod(x);
                case "argmin": return ma.argmin(x);
                case "argmax": return ma.argmax(x);
                case "all": return ma.all(x);
                case "any": return ma.any(x);
                case "alltrue": return ma.alltrue(x);
                case "sometrue": return ma.sometrue(x);
                case "median": return ma.median(x);

                // ---- masks, data, conversion, movement ----
                case "getdata": return ma.getdata(x);
                case "getmask": return ma.getmask(x);
                case "getmaskarray": return ma.getmaskarray(x);
                case "is_mask": return ma.is_mask(x);
                case "is_masked": return ma.is_masked(x);
                case "copy": return ma.copy(x);
                case "ravel": return ma.ravel(x);
                case "flatten": return x.flatten();
                case "transpose": return ma.transpose(x);
                case "squeeze": return ma.squeeze(x);
                case "atleast_1d": return ma.atleast_1d(x);
                case "atleast_2d": return ma.atleast_2d(x);
                case "atleast_3d": return ma.atleast_3d(x);
                case "compressed": return ma.compressed(x);
                case "count_masked": return ma.count_masked(x);
                case "diff": return ma.diff(x);
                case "ediff1d": return ma.ediff1d(x);
                case "sort": return ma.sort(x);
                case "argsort": return ma.argsort(x);
                case "unique": return ma.unique(x);
                case "harden_mask": return ma.harden_mask(x);
                case "soften_mask": return ma.soften_mask(x);
                case "shrink_mask": return x.shrink_mask();
                case "asarray": return ma.asarray(x);
                case "asanyarray": return ma.asanyarray(x);
                case "array": return ma.array(x);
                case "masked_array": return ma.masked_array(x);
                case "masked_all_like": return ma.masked_all_like(x);
                case "zeros_like": return ma.zeros_like(x);
                case "ones_like": return ma.ones_like(x);
                case "empty_like": return ma.empty_like(x);
                case "fix_invalid": return ma.fix_invalid(x);
                case "masked_invalid": return ma.masked_invalid(x);
                case "minimum_fill_value": return ma.minimum_fill_value(x);
                case "maximum_fill_value": return ma.maximum_fill_value(x);
                case "make_mask": return ma.make_mask(x);
                case "flatten_mask": return ma.flatten_mask(x);
                case "filled": return ma.filled(x);
                case "ndim": return ma.ndim(x);
                case "size": return ma.size(x);
                case "diag": return ma.diag(x);
                case "diagflat": return ma.diagflat(x);
                case "trace": return ma.trace(x);
                case "nonzero": return ma.nonzero(x);
                case "round": return ma.round(x);
                case "round_": return ma.round_(x);
                case "compress_nd": return ma.compress_nd(x);
                case "compress_rowcols": return ma.compress_rowcols(x);
                case "compress_rows": return ma.compress_rows(x);
                case "compress_cols": return ma.compress_cols(x);
                case "mask_rowcols": return ma.mask_rowcols(x);
                case "mask_rows": return ma.mask_rows(x);
                case "mask_cols": return ma.mask_cols(x);
                case "flatnotmasked_edges": return ma.flatnotmasked_edges(x);
                case "vander": return ma.vander(x);
                case "shape": return ma.shape(x);
                case "default_fill_value": return ma.default_fill_value(x.dtype);
                case "fill_value": return x.fill_value;
                case "repr": return x.ToString(true);
                case "str": return x.ToString();
                case "make_mask_noshrink": return ma.make_mask(x, shrink: false);
                case "astype_U1": return x.astype("U1");
                case "astype_S2": return x.astype("S2");
                case "astype_f8": return x.astype("f8");
                case "set_fill_q": x.fill_value = "qqq"; return x;
                case "set_fill_none": x.fill_value = null; return x;
                case "set_fill_5": x.fill_value = 5; return x;
                case "set_fill_value_q": ma.set_fill_value(x, "qqq"); return x;
                case "filled_q": return x.filled("q");
                case "filled_5": return x.filled(5);
                case "filled_bytes": return x.filled(B("Q"));
                case "print_disabled":
                    ma.masked_print_option.enable(false);
                    try
                    {
                        return x.ToString();
                    }
                    finally
                    {
                        ma.masked_print_option.enable(true);
                    }

                // ---- the masked_* family and the text-aware fills ----
                case "masked_object_ab": return ma.masked_object(x, "ab");
                case "masked_values_ab": return ma.masked_values(x, "ab");
                case "masked_values_15": return ma.masked_values(x, 1.5);
                case "masked_values_1": return ma.masked_values(x, 1);
                case "masked_object_15": return ma.masked_object(x, 1.5);
                case "masked_object_true": return ma.masked_object(x, true);
                case "masked_equal_15": return ma.masked_equal(x, 1.5);
                case "masked_greater_1": return ma.masked_greater(x, 1);
                case "masked_less_15": return ma.masked_less(x, 1.5);
                case "masked_inside_12": return ma.masked_inside(x, 1.0, 2.0);
                case "masked_inside_21": return ma.masked_inside(x, 2, 1);
                case "masked_outside_12": return ma.masked_outside(x, 1.0, 2.0);
                case "masked_inside_ab": return ma.masked_inside(x, "a", "b");
                case "masked_outside_ab": return ma.masked_outside(x, "b", "a");
                case "masked_inside_mix": return ma.masked_inside(x, 1.0, "b");
                case "masked_inside_bytes": return ma.masked_inside(x, B("a"), B("b"));
                case "masked_inside_complex": return ma.masked_inside(x, new Complex(0, 1), new Complex(0, 2));
                case "putmask_text": ma.putmask(x, Bools(x, true), "xyz"); return x;
                case "putmask_num": ma.putmask(x, Bools(x, true), 5); return x;
                case "putmask_none": ma.putmask(x, Bools(x, false), 1.5); return x;
                case "putmask_textmask": ma.putmask(x, x.data, "q"); return x;
                case "putmask_intmask": ma.putmask(x, np.zeros(new Shape(x.shape), np.int64), "q"); return x;
                case "putmask_masked":
                    ma.putmask(x, Bools(x, true), ma.array(np.full(new Shape(x.shape), "y"), mask: true));
                    return x;
                case "average_textw": return ma.average(np.ones(new Shape(x.shape)), weights: x);
                case "average_numw": return ma.average(x, weights: np.ones(new Shape(x.shape)));
                case "average_short": return ma.average(x, weights: np.ones(new Shape(1L)));
                case "average_ax0": return ma.average(x, axis: 0, weights: np.ones(new Shape((long)first)));
                case "average_ax5": return ma.average(x, axis: 5);
                case "masked_where_self": return ma.masked_where(x, np.zeros(new Shape(x.shape)));
                case "masked_where_short": return ma.masked_where(np.array(new[] { true, false }), x);
                case "masked_where_0d": return ma.masked_where(np.array(true), x);
                case "masked_where_bool": return ma.masked_where(Bools(x, false), x);
                case "mask_or_bool": return ma.mask_or(x, Bools(x, false));
                case "mask_or_self": return ma.mask_or(x, x);
                case "mask_or_false": return ma.mask_or(false, x);
                case "mask_or_nomask": return ma.mask_or(ma.nomask, x);
                case "compress_self": return ma.compress(x, np.arange(first));
                case "fix_invalid_textmask": return ma.fix_invalid(np.zeros(new Shape(x.shape)), mask: x.data);
                case "fix_invalid_mask2": return ma.fix_invalid(x, mask: new[] { true, false });
                case "common_fill_self": return ma.common_fill_value(x, x);
                case "common_fill_float": return ma.common_fill_value(x, ma.array(np.array(new[] { 1.0 })));
                case "take_0": return ma.take(x, np.array(new long[] { 0 }));
                case "repeat_2": return ma.repeat(x, 2);
                case "reshape_flat": return ma.reshape(x, new Shape(x.size));
                case "resize_4": return ma.resize(x, new Shape(4L));
                case "expand_dims_0": return ma.expand_dims(x, 0);
                case "concatenate_self": return ma.concatenate(new object[] { x, x });
                case "stack_self": return ma.stack(new object[] { x, x });
                case "hstack_self": return ma.hstack(x, x);
                case "vstack_self": return ma.vstack(x, x);
                case "clip_ab": return ma.clip(x, "a", "b");
                case "clip_num": return ma.clip(x, 0, 1);
                case "where_self": return ma.where(Bools(x, true), x, x);
                case "where_textcond": return ma.where(x, 1, 2);
                case "sort_end_false": return ma.sort(x, endwith: false);
                case "count_ax0": return ma.count(x, axis: 0);

                // ---- section tma's additions (gen_strings_oracle._tma_unary_extra): element access, stores, tolist, the
                // data / mask attributes, casts, filled with an explicit value, where / concatenate beside U text ----
                case "getitem_0": return x[0];
                case "getitem_1": return x[1];
                case "getitem_last": return x[-1];
                case "getitem_slice": return x["1:"];
                case "getitem_rev": return x["::-1"];
                case "getitem_mask": return x[Bools(x, true)];
                case "getitem_fancy": return x[np.array(new long[] { 0, 0 })];
                case "setitem_text": x[1] = "zz"; return x;
                case "setitem_long": x[0] = new string('w', 40); return x;
                case "setitem_masked": x[1] = ma.masked; return x;
                case "setitem_none": x[1] = null; return x;
                case "setitem_slice": x[":"] = "y"; return x;
                case "setitem_num": x[1] = 5; return x;
                case "tolist": return x.tolist();
                case "data": return x.data;
                case "mask_attr": return x.mask;
                case "astype_T": return x.astype(np.dtypes.StringDType());
                case "astype_TN": return x.astype(np.dtypes.StringDType(na_object: null));
                case "astype_U3": return x.astype("U3");
                case "astype_S3": return x.astype("S3");
                case "astype_i8": return x.astype("i8");
                case "astype_b": return x.astype(np.@bool);
                case "filled_long": return x.filled(new string('v', 30));
                case "filled_none": return x.filled(null);
                case "concat_u":
                    return x.ndim == 1
                        ? ma.concatenate(new object[] { x, ma.array(np.array(new[] { "q" }), mask: new[] { true }) })
                        : ma.concatenate(new object[] { x, x });
                case "where_text": return ma.where(Bools(x, true), x, "zz");
                case "copy_method": return x.copy();
                case "view_method": return x.view();
                case "reshape_method": return x.reshape(-1);
                case "swapaxes": return x.ndim > 0 ? ma.swapaxes(x, 0, -1) : x;
                default:
                    throw new InvalidOperationException($"strma ma1 has no function '{name}'");
            }
        }

        /// <summary>Two arguments through the named <c>np.ma</c> call.</summary>
        /// <param name="name">The table key.</param>
        /// <param name="x">The first argument (masked, plain or a value).</param>
        /// <param name="y">The second argument.</param>
        /// <returns>The call's result.</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static object Ma2(string name, object x, object y)
        {
            var ma = np.ma;
            return name switch
            {
                "add" => ma.add(x, y),
                "subtract" => ma.subtract(x, y),
                "multiply" => ma.multiply(x, y),
                "divide" => ma.divide(x, y),
                "true_divide" => ma.true_divide(x, y),
                "floor_divide" => ma.floor_divide(x, y),
                "mod" => ma.mod(x, y),
                "remainder" => ma.remainder(x, y),
                "fmod" => ma.fmod(x, y),
                "power" => ma.power(x, y),
                "hypot" => ma.hypot(x, y),
                "arctan2" => ma.arctan2(x, y),
                "maximum" => ma.maximum(x, y),
                "minimum" => ma.minimum(x, y),
                "equal" => ma.equal(x, y),
                "not_equal" => ma.not_equal(x, y),
                "less" => ma.less(x, y),
                "less_equal" => ma.less_equal(x, y),
                "greater" => ma.greater(x, y),
                "greater_equal" => ma.greater_equal(x, y),
                "logical_and" => ma.logical_and(x, y),
                "logical_or" => ma.logical_or(x, y),
                "logical_xor" => ma.logical_xor(x, y),
                "bitwise_and" => ma.bitwise_and(x, y),
                "bitwise_or" => ma.bitwise_or(x, y),
                "bitwise_xor" => ma.bitwise_xor(x, y),
                "left_shift" => ma.left_shift(x, y),
                "right_shift" => ma.right_shift(x, y),
                "allequal" => ma.allequal(x, y),
                "allclose" => ma.allclose(x, y),
                "inner" => ma.inner(x, y),
                "innerproduct" => ma.innerproduct(x, y),
                "outer" => ma.outer(x, y),
                "outerproduct" => ma.outerproduct(x, y),
                "dot" => ma.dot(x, y),
                "convolve" => ma.convolve(x, y),
                "correlate" => ma.correlate(x, y),
                "append" => ma.append(x, y),
                "isin" => ma.isin(x, y),
                "in1d" => ma.in1d(x, y),
                "intersect1d" => ma.intersect1d(x, y),
                "union1d" => ma.union1d(x, y),
                "setdiff1d" => ma.setdiff1d(x, y),
                "setxor1d" => ma.setxor1d(x, y),
                "mask_or" => ma.mask_or(x, y),
                "common_fill_value" => ma.common_fill_value(x, y),
                "masked_where" => ma.masked_where(x, y),
                "masked_equal" => ma.masked_equal(x, y),
                "masked_not_equal" => ma.masked_not_equal(x, y),
                "masked_greater" => ma.masked_greater(x, y),
                "masked_greater_equal" => ma.masked_greater_equal(x, y),
                "masked_less" => ma.masked_less(x, y),
                "masked_less_equal" => ma.masked_less_equal(x, y),
                "masked_values" => ma.masked_values(x, y),
                "masked_object" => ma.masked_object(x, y),
                "compress" => ma.compress(x, y),
                "filled" => ma.filled(x, y),
                "average" => ma.average(x, weights: y),
                _ => throw new InvalidOperationException($"strma ma2 has no function '{name}'"),
            };
        }

        /// <summary>A constructor / mask-helper call of section N (no operand; the arrays are spelled here as the generator
        ///     spells them).</summary>
        /// <param name="name">The table key.</param>
        /// <returns>The call's result (null for <c>putmask</c>, which returns None).</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static object Ma0(string name)
        {
            var ma = np.ma;
            NDArray U3() => np.array(new[] { "ab", "c", "de" });
            NDArray S3() => U3().astype("S2");
            NDArray F() => np.array(new[] { 1.5, 2.0, 3.0 });
            bool[] M(params int[] bits) => bits.Select(v => v != 0).ToArray();
            switch (name)
            {
                case "text_mask": return ma.array(np.arange(3), mask: np.array(new[] { "a", "", "b" }));
                case "text_mask_empty": return ma.array(np.zeros(new Shape(0L)), mask: np.zeros(new Shape(0L), "U1"));
                case "text_mask_size": return ma.array(np.arange(3), mask: np.array(new[] { "a", "" }));
                case "text_mask_one": return ma.array(np.arange(3), mask: np.array(new[] { "a" }));
                case "mask_size": return ma.array(U3(), mask: M(1, 0));
                case "mask_one": return ma.array(U3(), mask: M(1));
                case "mask_reshape":
                    return ma.array(np.array(new[,] { { "ab", "c", "x" }, { "de", "f", "y" } }), mask: np.zeros(new Shape(3L, 2L), np.@bool));
                case "mask_true": return ma.array(U3(), mask: true);
                case "mask_false": return ma.array(U3(), mask: false);
                case "keep_mask": return ma.array(ma.array(U3(), mask: M(1, 0, 0)), mask: M(0, 1, 0));
                case "keep_mask_false": return ma.array(ma.array(U3(), mask: M(1, 0, 0)), mask: M(0, 1, 0), keep_mask: false);
                case "fill_hello": return ma.array(U3(), fill_value: "hello");
                case "fill_5": return ma.array(U3(), fill_value: 5);
                case "fill_bytes_u": return ma.array(U3(), fill_value: B("xyz"));
                case "fill_str_s": return ma.array(S3(), fill_value: "xyz");
                case "fill_nonascii_s": return ma.array(S3(), fill_value: "é");
                case "fill_nonascii_bytes_u": return ma.array(U3(), fill_value: new byte[] { 0xe9 });
                case "fill_str_float": return ma.array(F(), fill_value: "abc");
                case "fill_str_int": return ma.array(np.arange(3), fill_value: "abc");
                case "fill_str_bool": return ma.array(np.array(new[] { true }), fill_value: "abc");
                case "fill_bytes_float": return ma.array(F(), fill_value: B("1.5"));
                case "fill_array_u": return ma.array(U3(), fill_value: np.array(new[] { "xyz" }));
                case "fill_inherit": return ma.array(ma.array(U3(), fill_value: "zz"));
                case "fill_inherit_dtype": return ma.array(ma.array(U3(), fill_value: "zz"), dtype: "U1");
                case "hard_inherit": return ma.array(ma.array(U3(), mask: M(1, 0, 0), hard_mask: true));
                case "masked_all_u2": return ma.masked_all(new Shape(3L), "U2");
                case "masked_all_s2": return ma.masked_all(new Shape(2L, 2L), "S2");
                case "default_fill_u": return ma.default_fill_value(np.dtype("U2"));
                case "default_fill_s": return ma.default_fill_value(np.dtype("S2"));
                case "make_mask_text": return ma.make_mask(np.array(new[] { "a", "" }));
                case "make_mask_text_empty": return ma.make_mask(np.zeros(new Shape(0L), "U1"), shrink: false);
                case "mask_or_text_bool": return ma.mask_or(U3(), np.zeros(new Shape(3L), np.@bool));
                case "mask_or_bool_text": return ma.mask_or(np.zeros(new Shape(3L), np.@bool), U3());
                case "mask_or_u_s": return ma.mask_or(U3(), S3());
                case "mask_or_widths": return ma.mask_or(U3(), np.array(new[] { "a", "b", "c" }));
                case "mask_or_empty": return ma.mask_or(np.zeros(new Shape(0L), "U1"), np.zeros(new Shape(0L), "U1"));
                case "mask_or_int_bool": return ma.mask_or(np.array(new long[] { 1, 0, 1 }), np.array(new[] { true, false, true }));
                case "mask_or_0d_text": return ma.mask_or(np.array(false), U3());
                case "masked_where_short": return ma.masked_where(M(1, 0), U3());
                case "masked_where_col": return ma.masked_where(np.array(new[,] { { true }, { false } }), np.zeros(new Shape(2L, 3L)));
                case "masked_where_text": return ma.masked_where(np.array(new[] { "a", "", "b" }), np.arange(3));
                case "masked_where_text_short": return ma.masked_where(np.array(new[] { "a", "" }), np.arange(3));
                case "masked_where_empty": return ma.masked_where(np.zeros(new Shape(0L), "U1"), np.zeros(new Shape(0L)));
                case "masked_where_allfalse": return ma.masked_where(M(0, 0, 0), U3());
                case "putmask_plain":
                    ma.putmask(U3(), M(1, 0, 1), "zz");
                    return null;
                case "compress_2d_text": return ma.compress(np.array(new[,] { { "a" } }), U3());
                case "compress_rows_1d": return ma.compress_rows(U3());
                case "compress_cols_1d": return ma.compress_cols(U3());
                case "mask_rows_1d": return ma.mask_rows(U3());
                case "mask_cols_1d": return ma.mask_cols(U3());
                case "common_fill_u_s": return ma.common_fill_value(ma.array(U3()), ma.array(S3()));
                case "common_fill_u_u1": return ma.common_fill_value(ma.array(U3()), ma.array(np.array(new[] { "a" })));
                default:
                    throw new InvalidOperationException($"strma ma0 has no function '{name}'");
            }
        }

        /// <summary>
        ///     A constructor / mask-helper call of section tma (<c>gen_tma</c>'s <c>ctor</c> list): numpy.ma building masked
        ///     arrays over NEP 55 <c>StringDType</c> data — <c>StringDType</c> masks (cast to bool), fill values of every
        ///     kind into a <c>T</c> array (text, numbers, bytes, <c>T</c> / <c>U</c> arrays; <c>coerce=False</c> refusals),
        ///     inherited fills and hard masks, <c>masked_all</c>, the default fill value of every <c>T</c> descriptor, mask
        ///     reconciliation, <c>common_fill_value</c>, and joins beside <c>U</c> / <c>S</c> / numbers. The arrays are spelled
        ///     here as the generator spells them.
        /// </summary>
        /// <param name="name">The table key.</param>
        /// <returns>The call's result (null for <c>putmask</c>, which returns None).</returns>
        /// <exception cref="InvalidOperationException">An unknown name.</exception>
        private static object TMa0(string name)
        {
            var ma = np.ma;
            // Fresh descriptors per call, as the generator's TSpec.build() makes them.
            DType T() => np.dtypes.StringDType();
            DType TN() => np.dtypes.StringDType(na_object: null);
            DType TNAN() => np.dtypes.StringDType(na_object: double.NaN);
            DType TC() => np.dtypes.StringDType(coerce: false);
            NDArray T3(DType dt = null) => np.array(new[] { "ab", "c", "de" }, dt ?? T());
            bool[] M(params int[] bits) => bits.Select(v => v != 0).ToArray();
            switch (name)
            {
                case "t_mask": return ma.array(np.arange(3), mask: np.array(new[] { "a", "", "b" }, T()));
                case "t_mask_na": return ma.array(np.arange(3), mask: np.array(new object[] { "a", null, "b" }, TN()));
                case "t_mask_empty": return ma.array(np.zeros(new Shape(0L)), mask: np.zeros(new Shape(0L), T()));
                case "t_mask_size": return ma.array(np.arange(3), mask: np.array(new[] { "a", "" }, T()));
                case "mask_size": return ma.array(T3(), mask: M(1, 0));
                case "mask_one": return ma.array(T3(), mask: M(1));
                case "mask_reshape":
                    return ma.array(np.array(new[,] { { "ab", "c", "x" }, { "de", "f", "y" } }, T()),
                        mask: np.zeros(new Shape(3L, 2L), np.@bool));
                case "mask_true": return ma.array(T3(), mask: true);
                case "mask_false": return ma.array(T3(), mask: false);
                case "keep_mask": return ma.array(ma.array(T3(), mask: M(1, 0, 0)), mask: M(0, 1, 0));
                case "keep_mask_false": return ma.array(ma.array(T3(), mask: M(1, 0, 0)), mask: M(0, 1, 0), keep_mask: false);
                case "fill_hello": return ma.array(T3(), fill_value: "hello");
                case "fill_long": return ma.array(T3(), fill_value: new string('f', 40));
                case "fill_5": return ma.array(T3(), fill_value: 5);
                case "fill_15": return ma.array(T3(), fill_value: 1.5);
                case "fill_true": return ma.array(T3(), fill_value: true);
                case "fill_bytes": return ma.array(T3(), fill_value: B("xyz"));
                case "fill_5_coerce_false": return ma.array(T3(TC()), fill_value: 5);
                case "fill_text_coerce_false": return ma.array(T3(TC()), fill_value: "q");
                case "fill_none_na": return ma.array(T3(TN()), fill_value: "zz");
                case "fill_array_t": return ma.array(T3(), fill_value: np.array(new[] { "xyz" }, T()));
                case "fill_array_u": return ma.array(T3(), fill_value: np.array(new[] { "xyz" }));
                case "fill_0d_t": return ma.array(T3(), fill_value: np.array("xyz", T()));
                case "fill_t_into_u": return ma.array(np.array(new[] { "ab", "c" }), fill_value: np.array("xyz", T()));
                case "fill_t_into_f": return ma.array(np.array(new[] { 1.5 }), fill_value: np.array("2.5", T()));
                case "fill_inherit": return ma.array(ma.array(T3(), fill_value: "zz"));
                case "fill_inherit_dtype_u": return ma.array(ma.array(T3(), fill_value: "zz"), dtype: "U1");
                case "fill_inherit_dtype_t": return ma.array(ma.array(np.array(new[] { "ab" }), fill_value: "zz"), dtype: T());
                case "fill_inherit_dtype_tn": return ma.array(ma.array(T3(), fill_value: "zz"), dtype: TN());
                case "hard_inherit": return ma.array(ma.array(T3(), mask: M(1, 0, 0), hard_mask: true));
                case "masked_all_t": return ma.masked_all(new Shape(3L), T());
                case "masked_all_tn": return ma.masked_all(new Shape(2L, 2L), TN());
                case "masked_all_tc": return ma.masked_all(new Shape(2L), TC());
                case "default_fill_t": return ma.default_fill_value(T());
                case "default_fill_tn": return ma.default_fill_value(TN());
                case "default_fill_tnan": return ma.default_fill_value(TNAN());
                case "default_fill_tc": return ma.default_fill_value(TC());
                case "default_fill_t_arr": return ma.default_fill_value(T3());
                case "minimum_fill_t": return ma.minimum_fill_value(T3());
                case "maximum_fill_t": return ma.maximum_fill_value(T3(TN()));
                case "make_mask_t": return ma.make_mask(np.array(new[] { "a", "" }, T()));
                case "make_mask_t_empty": return ma.make_mask(np.zeros(new Shape(0L), T()), shrink: false);
                case "make_mask_tn": return ma.make_mask(np.array(new object[] { "a", null }, TN()));
                case "mask_or_t_bool": return ma.mask_or(T3(), np.zeros(new Shape(3L), np.@bool));
                case "mask_or_bool_t": return ma.mask_or(np.zeros(new Shape(3L), np.@bool), T3());
                case "mask_or_t_u": return ma.mask_or(T3(), np.array(new[] { "a", "b", "c" }));
                case "mask_or_t_t": return ma.mask_or(T3(), T3());
                case "mask_or_t_tn": return ma.mask_or(T3(), T3(TN()));
                case "masked_where_t": return ma.masked_where(np.array(new[] { "a", "", "b" }, T()), np.arange(3));
                case "masked_where_t_short": return ma.masked_where(np.array(new[] { "a", "" }, T()), np.arange(3));
                case "masked_where_short": return ma.masked_where(M(1, 0), T3());
                case "masked_where_allfalse": return ma.masked_where(M(0, 0, 0), T3());
                case "masked_where_copy_false": return ma.masked_where(M(1, 0, 0), T3(), copy: false);
                case "putmask_plain":
                    ma.putmask(T3(), M(1, 0, 1), "zz");
                    return null;
                case "compress_rows_1d": return ma.compress_rows(T3());
                case "compress_cols_1d": return ma.compress_cols(T3());
                case "mask_rows_1d": return ma.mask_rows(T3());
                case "mask_cols_1d": return ma.mask_cols(T3());
                case "common_fill_t_u": return ma.common_fill_value(ma.array(T3()), ma.array(np.array(new[] { "a" })));
                case "common_fill_t_t": return ma.common_fill_value(ma.array(T3()), ma.array(T3()));
                case "common_fill_t_tn": return ma.common_fill_value(ma.array(T3()), ma.array(T3(TN())));
                case "common_fill_t_f": return ma.common_fill_value(ma.array(T3()), ma.array(np.array(new[] { 1.5 })));
                case "array_list": return ma.array(new object[] { "ab", "c" }, dtype: T(), mask: M(0, 1));
                case "array_list_na": return ma.array(new object[] { "ab", null }, dtype: TN(), mask: M(0, 0));
                case "array_list_num": return ma.array(new object[] { 1, 2.5 }, dtype: T());
                case "array_list_num_coerce_false": return ma.array(new object[] { 1, 2.5 }, dtype: TC());
                case "masked_array_t": return ma.masked_array(T3(), mask: M(0, 0, 1), fill_value: "w");
                case "asarray_t": return ma.asarray(T3());
                case "asanyarray_t": return ma.asanyarray(T3());
                case "filled_plain_t": return ma.filled(T3());
                case "getdata_t": return ma.getdata(T3());
                case "getmaskarray_t": return ma.getmaskarray(T3());
                case "zeros_t": return ma.zeros(new Shape(3L), T());
                case "ones_t": return ma.ones(new Shape(2L), T());
                case "empty_t": return ma.empty(new Shape(2L), T());
                case "masked_values_t": return ma.masked_values(T3(), "c");
                case "masked_object_t": return ma.masked_object(T3(), "c");
                case "masked_equal_t": return ma.masked_equal(T3(), "c");
                case "fix_invalid_t": return ma.fix_invalid(T3());
                case "masked_invalid_t": return ma.masked_invalid(T3());
                case "concatenate_t_u":
                    return ma.concatenate(new object[] { ma.array(T3(), mask: M(1, 0, 0)), ma.array(np.array(new[] { "q" })) });
                case "concatenate_u_t":
                    return ma.concatenate(new object[] { ma.array(np.array(new[] { "q" })), ma.array(T3(), mask: M(1, 0, 0)) });
                case "concatenate_t_tn": return ma.concatenate(new object[] { ma.array(T3()), ma.array(T3(TN())) });
                case "concatenate_t_s":
                    return ma.concatenate(new object[] { ma.array(T3()), ma.array(np.array(new[] { "q" }).astype("S1")) });
                case "stack_t": return ma.stack(new object[] { ma.array(T3(), mask: M(1, 0, 0)), ma.array(T3()) });
                case "where_t_u":
                    return ma.where(M(1, 0, 1), ma.array(T3(), mask: M(1, 0, 0)), np.array(new[] { "q", "r", "s" }));
                case "where_t_num": return ma.where(M(1, 0, 1), ma.array(T3()), 1.5);
                default:
                    throw new InvalidOperationException($"tma tma0 has no function '{name}'");
            }
        }
    }
}
