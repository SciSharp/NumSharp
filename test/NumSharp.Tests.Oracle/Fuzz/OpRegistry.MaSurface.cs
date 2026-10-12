using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The numpy.ma SURFACE half of the registry: the adapters behind <c>ma_surface.jsonl</c>
    ///     (test/oracle/gen_ma_surface_oracle.py), the numpy.ma functions the coverage inventory listed as available but
    ///     unproven — the creation wrappers, the joins, the fill-value helpers, the predicates and queries, ids, the
    ///     hard-mask switches, masked_object, putmask, convolve / correlate / cov / corrcoef, the edge finders, the
    ///     compressors, ndenumerate and apply_along_axis / apply_over_axes.
    ///
    ///     <para>
    ///     <b>What the adapters may do.</b> Map a case's params to the C# call NumPy's call stands for, and format or wrap
    ///     NumSharp's answer for the comparison (a <see cref="Slice"/> as Python's <c>slice(...)</c>, a returned
    ///     <c>long[]</c> shape as a tuple, a boxed fill value as its Python repr) — never repair it. A NumPy call the C#
    ///     signature cannot express (a parameter NumSharp's overload lacks, an argument kind it does not take) throws
    ///     <see cref="NotSupportedException"/> naming the missing piece, so the case fails and the generator carves it
    ///     as an API-shape gap rather than the registry quietly calling something else.
    ///     </para>
    ///     <para>
    ///     <b>Params.</b> A Python scalar rides JSON as its C# literal (<see cref="MaPyValue"/>): int → <see cref="long"/>
    ///     (<see cref="BigInteger"/> past it), float → <see cref="double"/> ("nan" / "inf" / "-inf" strings for the
    ///     non-finite ones), bool, complex → <c>{"re", "im"}</c>, None → null, str → string.
    ///     </para>
    /// </summary>
    public static partial class OpRegistry
    {
        /// <summary>
        ///     Dispatches a surface op whose result kind is <c>masked</c>, <c>array</c> or <c>scalar</c> (reached from
        ///     <see cref="ApplyMasked"/>'s default branch, and from its guarded cases for the op names the older ma tiers
        ///     already registered).
        /// </summary>
        /// <param name="name">The bare numpy.ma function name (the <c>ma.</c> prefix stripped).</param>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed masked operands.</param>
        /// <returns>The result: an <see cref="NDMaskedArray"/>, an <see cref="NDArray"/>, or a boxed bool / int / long.</returns>
        /// <exception cref="NotSupportedException">The op is not a surface op, or the case's NumPy call has no C# form.</exception>
        internal static object ApplyMaskedSurface(string name, IReadOnlyDictionary<string, JsonElement> p, NDMaskedArray[] ops)
        {
            switch (name)
            {
                // ---- creation (no operand: NumPy's _convert2ma wrappers) -------------------------------------------
                case "arange": return MaArange(p);
                case "zeros":
                case "ones":
                case "empty":
                {
                    if (p.ContainsKey("order"))
                        throw new NotSupportedException($"np.ma.{name} has no order parameter (NumPy's ma.{name}(shape, dtype, order))");
                    var shape = new Shape(ParseLongArray(p["shape"]));
                    var dt = SurfDType(p, "dtype");
                    var r = name == "zeros" ? np.ma.zeros(shape, dt) : name == "ones" ? np.ma.ones(shape, dt) : np.ma.empty(shape, dt);
                    return MaFacet(p, r);
                }
                case "zeros_like":
                case "ones_like":
                case "empty_like":
                {
                    // "plain": NumPy's prototype is the operand's DATA (an ndarray), not a MaskedArray.
                    object proto = p.ContainsKey("plain") ? ops[0].data : ops[0];
                    var dt = SurfDType(p, "dtype");
                    var r = name == "zeros_like" ? np.ma.zeros_like(proto, dt)
                        : name == "ones_like" ? np.ma.ones_like(proto, dt) : np.ma.empty_like(proto, dt);
                    return MaFacet(p, r);
                }
                case "identity": return np.ma.identity(SurfInt(p, "n"), SurfDType(p, "dtype"));
                case "indices":
                    if (p.ContainsKey("sparse"))
                        throw new NotSupportedException("np.ma.indices has no sparse parameter");
                    return np.ma.indices(ParseLongArray(p["dimensions"]), SurfDType(p, "dtype"));
                case "frombuffer":
                    return np.ma.frombuffer(FuzzCorpus.FromHex(SurfStr(p, "buffer")), SurfDType(p, "dtype"),
                                            p["count"].GetInt64(), p["offset"].GetInt64());
                case "fromfunction":
                    // No scope: the menu callbacks dispose their own temporaries, so a leak inside np.ma.fromfunction
                    // itself stays visible to the leak sweep.
                    return np.ma.fromfunction(FromFunctionFunc(SurfStr(p, "func")), new Shape(ParseLongArray(p["shape"])),
                                              SurfDType(p, "dtype"));

                // ---- predicates and queries -----------------------------------------------------------------------
                case "getmask": return np.ma.getmask(p.ContainsKey("plain") ? ops[0].data : ops[0]);
                case "is_mask": return MaQuery(ops, p, x => np.ma.is_mask(x));
                case "isarray": return MaQuery(ops, p, x => np.ma.isarray(x));
                case "isMA": return MaQuery(ops, p, x => np.ma.isMA(x));
                case "isMaskedArray": return MaQuery(ops, p, x => np.ma.isMaskedArray(x));
                case "ndim": return MaQuery(ops, p, x => np.ma.ndim(x));
                case "size":
                    if (p.ContainsKey("axis"))
                        throw new NotSupportedException("np.ma.size has no axis parameter (NumPy's ma.size(obj, axis=None))");
                    return MaQuery(ops, p, x => np.ma.size(x));
                case "ids": return MaIdsDelta(p, ops);
                case "harden_mask":
                case "soften_mask": return MaHardMask(name, p, ops);
                case "masked_object":
                    if (p.ContainsKey("shrink"))
                        throw new NotSupportedException("np.ma.masked_object has no shrink parameter");
                    return np.ma.masked_object(ops[0], MaPyValue(p["value"]));

                // ---- putmask: the mutated array is the result ---------------------------------------------------------
                case "putmask": return MaPutmask(p, ops);

                // ---- products ----------------------------------------------------------------------------------------
                case "convolve":
                case "correlate": return MaSlidingProduct(name, p, ops);
                case "cov":
                    return np.ma.cov(ops[0], p.ContainsKey("y") ? ops[1] : null, MaBool(p, "rowvar", true), MaBool(p, "bias", false),
                                     MaBool(p, "allow_masked", true), SurfNullableInt(p, "ddof"));
                case "corrcoef":
                    return np.ma.corrcoef(ops[0], p.ContainsKey("y") ? ops[1] : null, MaBool(p, "rowvar", true),
                                          MaBool(p, "allow_masked", true));

                // ---- compressors ---------------------------------------------------------------------------------------
                case "compress_nd": return np.ma.compress_nd(ops[0], MaAxisParam(p, "compress_nd"));
                case "compress_rowcols": return np.ma.compress_rowcols(ops[0], MaAxisParam(p, "compress_rowcols"));

                // ---- apply_* -------------------------------------------------------------------------------------------
                case "apply_along_axis":
                {
                    // Every masked array the callback returns is the caller's (np.ma's callback contract): a scope reclaims
                    // them, Yield keeps only the assembled result.
                    using var scope = NDScope.Open();
                    var r = np.ma.apply_along_axis(AlongAxisMaFunc(SurfStr(p, "func")), SurfInt(p, "axis"), ops[0]);
                    return MaskedArrayModule.Yield(scope, r);
                }
                case "apply_over_axes":
                {
                    using var scope = NDScope.Open();
                    var f = OverAxesMaFunc(SurfStr(p, "func"));
                    var axes = p["axes"];
                    var r = axes.ValueKind == JsonValueKind.Array
                        ? np.ma.apply_over_axes(f, ops[0], axes.EnumerateArray().Select(x => x.GetInt32()).ToArray())
                        : np.ma.apply_over_axes(f, ops[0], axes.GetInt32());
                    return MaskedArrayModule.Yield(scope, r);
                }
                default:
                    throw new NotSupportedException($"ma op 'ma.{name}' not registered in OpRegistry.ApplyMasked / ApplyMaskedSurface");
            }
        }

        /// <summary>Dispatches an op whose result kind is <c>text</c>. A surface op returns a Python object NumPy returns,
        /// as the normalized repr <c>gen_ma_surface_oracle.pyrepr</c> writes for NumPy's (see <see cref="MaRepr"/>); any
        /// other op (ma_fill's <c>ma.fill</c> script) runs through <see cref="ApplyMasked"/>, whose result is the text.</summary>
        /// <param name="op">The <c>ma.</c>-prefixed corpus op key.</param>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed masked operands.</param>
        /// <returns>The repr text; for an op outside the surface tier, <see cref="ApplyMasked"/>'s string, or a rendering
        ///     of a non-string result (<c>&lt;TypeName&gt; value</c>), or null for a null result.</returns>
        /// <exception cref="NotSupportedException">Neither this dispatcher nor <see cref="ApplyMasked"/> knows the op, or
        ///     the case's NumPy call has no C# form.</exception>
        internal static string ApplyMaskedText(string op, IReadOnlyDictionary<string, JsonElement> p, NDMaskedArray[] ops)
        {
            string name = op.StartsWith("ma.", StringComparison.Ordinal) ? op.Substring(3) : op;
            switch (name)
            {
                case "default_fill_value":
                    // NumSharp's overload takes a DType only; NumPy also takes an array or a scalar (whose dtype it reads).
                    if (!p.ContainsKey("dtype"))
                        throw new NotSupportedException("np.ma.default_fill_value takes a DType only (NumPy's also takes an array or a scalar)");
                    return MaRepr(np.ma.default_fill_value(np.dtype(SurfStr(p, "dtype"))));
                case "minimum_fill_value":
                case "maximum_fill_value":
                {
                    // The argument: a DType (dtype param), a Python scalar (value param), else the masked operand.
                    object arg = p.ContainsKey("dtype") ? np.dtype(SurfStr(p, "dtype"))
                        : p.ContainsKey("value") ? MaPyValue(p["value"]) : ops[0];
                    return MaRepr(name == "minimum_fill_value" ? np.ma.minimum_fill_value(arg) : np.ma.maximum_fill_value(arg));
                }
                case "set_fill_value":
                    if (p.ContainsKey("plain"))
                    {
                        // NumPy ignores a plain ndarray and returns None; the C# method returns nothing (void ≡ None).
                        np.ma.set_fill_value(ops[0].data, MaPyValue(p["value"]));
                        return "None";
                    }
                    np.ma.set_fill_value(ops[0], MaPyValue(p["value"]));
                    return MaRepr(ops[0].fill_value);
                case "common_fill_value":
                {
                    // The cases set each operand's fill value first (null: keep the dtype default).
                    if (p["fa"].ValueKind != JsonValueKind.Null)
                        np.ma.set_fill_value(ops[0], MaPyValue(p["fa"]));
                    if (p["fb"].ValueKind != JsonValueKind.Null)
                        np.ma.set_fill_value(ops[1], MaPyValue(p["fb"]));
                    return MaRepr(np.ma.common_fill_value(ops[0], ops[1]));
                }
                case "shape":
                    // NumPy's ma.shape is a tuple; NumSharp returns the same dims as a long[].
                    return (string)MaQuery(ops, p, x => PyTuple(np.ma.shape(x).Select(d => (object)d)));
                case "masked_object":
                {
                    // The fill_value facet: NumPy's masked_object sets fill_value=value (converted to the dtype). The masked
                    // result itself is this replay's (a data copy and a mask), released once its fill value is read.
                    var r = np.ma.masked_object(ops[0], MaPyValue(p["value"]));
                    string text = MaRepr(r.fill_value);
                    ReleaseMaView(r, ops[0]);
                    return text;
                }
                case "clump_masked": return MaRepr(np.ma.clump_masked(ops[0]));
                case "clump_unmasked": return MaRepr(np.ma.clump_unmasked(ops[0]));
                case "flatnotmasked_contiguous": return MaRepr(np.ma.flatnotmasked_contiguous(ops[0]));
                case "flatnotmasked_edges": return MaEdgeArray(np.ma.flatnotmasked_edges(ops[0]));
                case "notmasked_contiguous": return MaRepr(np.ma.notmasked_contiguous(ops[0], SurfNullableInt(p, "axis")));
                case "notmasked_edges":
                {
                    // Polymorphic like NumPy's: an int64 NDArray {first, last} (NumPy's array) or null for the flat case, an
                    // NDArray[][] {mins, maxs} (NumPy's list of two tuples of arrays) for an explicit axis of a >1-D array.
                    object r = np.ma.notmasked_edges(ops[0], SurfNullableInt(p, "axis"));
                    return r switch
                    {
                        null => "None",
                        NDArray e => MaEdgeArray(e),
                        NDArray[][] pair => MaEdgePairs(pair),
                        _ => throw new NotSupportedException($"notmasked_edges returned {r.GetType().Name}"),
                    };
                }
                case "ndenumerate":
                {
                    if (p.ContainsKey("compressed"))
                        throw new NotSupportedException("np.ma.ndenumerate has no compressed parameter (NumSharp always skips masked elements)");
                    // NumPy yields (index tuple, value); NumSharp a (long[] index, boxed value) pair: the index array IS the tuple.
                    var parts = np.ma.ndenumerate(ops[0])
                        .Select(e => "(" + PyTuple(e.index.Select(x => (object)x)) + ", " + MaRepr(e.value) + ")");
                    return "[" + string.Join(", ", parts) + "]";
                }
                default:
                {
                    // The other tier with a text result, ma_fill (its "ma.fill" script: a rendered fill-value
                    // observation, or common_fill_value's "None"), runs through ApplyMasked and is compared verbatim. A
                    // non-text result (an array where NumPy answered None) shows as its own rendering, so it fails the
                    // comparison instead of matching by accident. An op neither dispatcher knows throws ApplyMasked's
                    // NotSupportedException.
                    object raw = ApplyMasked(op, p, ops);
                    return raw as string ?? (raw is null ? null : $"<{raw.GetType().Name}> {raw}");
                }
            }
        }

        /// <summary>Dispatches a surface op whose result kind is <c>tuple</c> (a tuple of plain arrays).</summary>
        /// <param name="op">The <c>ma.</c>-prefixed corpus op key.</param>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed masked operands.</param>
        /// <returns>The slots, in NumPy's tuple order.</returns>
        /// <exception cref="NotSupportedException">The op is not a tuple op.</exception>
        internal static NDArray[] ApplyMaskedArrayTuple(string op, IReadOnlyDictionary<string, JsonElement> p, NDMaskedArray[] ops)
        {
            string name = op.StartsWith("ma.", StringComparison.Ordinal) ? op.Substring(3) : op;
            return name switch
            {
                "nonzero" => np.ma.nonzero(ops[0]).Cast<NDArray>().ToArray(),
                _ => throw new NotSupportedException($"ma tuple op '{op}' not registered in OpRegistry.ApplyMaskedArrayTuple"),
            };
        }

        // ---- creation helpers ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.ma.arange</c> with the overload NumPy's positional bounds select: all Python ints that fit an int →
        ///     <c>(int)</c> / <c>(int, int)</c> / <c>(int, int, int)</c>; otherwise <c>(double, double[, double])</c>, which a
        ///     mixed int / float call binds in C# too.
        /// </summary>
        /// <param name="p">The case params: <c>args</c> (1–3 Python numbers) and an optional <c>dtype</c>.</param>
        /// <returns>The masked range.</returns>
        /// <exception cref="NotSupportedException">dtype= (no such parameter), a lone float bound (no <c>(double stop)</c>
        /// overload), or an integer bound past int.</exception>
        private static NDMaskedArray MaArange(IReadOnlyDictionary<string, JsonElement> p)
        {
            if (p.ContainsKey("dtype"))
                throw new NotSupportedException("np.ma.arange has no dtype parameter");
            var args = p["args"].EnumerateArray().ToArray();
            bool allInt = args.All(a => a.ValueKind == JsonValueKind.Number && a.TryGetInt32(out _));
            if (allInt)
            {
                var v = args.Select(a => a.GetInt32()).ToArray();
                return v.Length switch
                {
                    1 => np.ma.arange(v[0]),
                    2 => np.ma.arange(v[0], v[1]),
                    _ => np.ma.arange(v[0], v[1], v[2]),
                };
            }
            if (args.Length == 1)
                throw new NotSupportedException("np.ma.arange has no (double stop) overload (NumPy's ma.arange(5.5))");
            var d = args.Select(PyFloat).ToArray();
            return d.Length == 2 ? np.ma.arange(d[0], d[1]) : np.ma.arange(d[0], d[1], d[2]);
        }

        /// <summary>
        ///     The <c>nodata</c> facet of empty / empty_like: the result with its uninitialized data replaced by zeros of its
        ///     dtype and its mask (or nomask) kept — the generator applies the same to NumPy's result, so shape, dtype, mask
        ///     and form are compared and the arbitrary bytes are not. Without the facet param the result passes through.
        /// </summary>
        /// <param name="p">The case params.</param>
        /// <param name="r">NumSharp's result.</param>
        /// <returns>The facet (a new masked array over fresh zeros) or <paramref name="r"/> itself.</returns>
        private static NDMaskedArray MaFacet(IReadOnlyDictionary<string, JsonElement> p, NDMaskedArray r)
        {
            if (!p.ContainsKey("facet"))
                return r;
            var zeros = np.zeros(new Shape(r.data.Shape.dimensions), r.data.dtype);
            // The uninitialized data is dropped (this replay owns it); the mask moves to the facet unchanged.
            var facet = new NDMaskedArray(zeros, r._mask);
            r.data.Dispose();
            return facet;
        }

        /// <summary>The fromfunction callable menu — the C# twin of <c>gen_ma_surface_oracle.FROMFUNCTION_FUNCS</c>. No int
        /// literals: a C# and a Python literal promote differently against a narrow grid.</summary>
        /// <param name="name">The menu name.</param>
        /// <returns>The vectorized function of the per-axis index grids.</returns>
        /// <exception cref="NotSupportedException">An unknown name (a corpus/registry drift).</exception>
        private static Func<NDArray[], NDArray> FromFunctionFunc(string name) => name switch
        {
            "first" => ix => ix[0],
            "add2" => ix => ix[0] + ix[1],
            "mul2" => ix => ix[0] * ix[1],
            "sub2" => ix => ix[0] - ix[1],
            "gt2" => ix => ix[0] > ix[1],
            "sq2" => ix =>
            {
                // i*i is this callback's temporary (NumPy's too): released once the sum is built.
                using var square = ix[0] * ix[0];
                return square + ix[1];
            },
            "add3" => ix =>
            {
                using var partial = ix[0] + ix[1];
                return partial + ix[2];
            },
            _ => throw new NotSupportedException($"fromfunction menu function '{name}'"),
        };

        // ---- predicate / query helpers --------------------------------------------------------------------------------------

        /// <summary>
        ///     The object a predicate / query case passes (<c>gen_ma_surface_oracle.PRED_ARGS</c>): the masked operand
        ///     (default), its data ndarray, its full mask, <c>getmask(op)</c>, a Python list (a C# <c>bool[]</c>), a Python
        ///     int (a boxed long), the masked singleton or nomask.
        /// </summary>
        /// <param name="ops">The reconstructed masked operands.</param>
        /// <param name="p">The case params (<c>arg</c>, absent → the masked operand).</param>
        /// <returns>The argument object.</returns>
        /// <exception cref="NotSupportedException">An unknown arg form (a corpus/registry drift).</exception>
        private static object MaPredArg(NDMaskedArray[] ops, IReadOnlyDictionary<string, JsonElement> p)
        {
            string arg = p.TryGetValue("arg", out var a) ? a.GetString() : "masked";
            return arg switch
            {
                "masked" => ops[0],
                "data" => ops[0].data,
                "maskarray" => np.ma.getmaskarray(ops[0]),
                "getmask" => np.ma.getmask(ops[0]),
                "list" => new[] { true, false },
                "int" => 5L,
                "singleton" => np.ma.masked,
                "nomask" => np.ma.nomask,
                _ => throw new NotSupportedException($"ma predicate arg '{arg}'"),
            };
        }

        /// <summary>Runs a predicate / query over the case's argument form and then releases the one argument this replay
        /// may have BUILT (getmaskarray of an unmasked operand is a fresh all-False array). No ambient scope: an array the
        /// numpy.ma call itself leaves behind stays visible to the leak sweep.</summary>
        /// <param name="ops">The reconstructed masked operands.</param>
        /// <param name="p">The case params (<c>arg</c>).</param>
        /// <param name="query">The numpy.ma call.</param>
        /// <returns>The boxed answer (a bool, an int, a long or a string), never an array.</returns>
        private static object MaQuery(NDMaskedArray[] ops, IReadOnlyDictionary<string, JsonElement> p, Func<object, object> query)
        {
            object arg = MaPredArg(ops, p);
            try
            {
                return query(arg);
            }
            finally
            {
                // Only a fresh getmaskarray is ours: the operand's own mask, the data, nomask and the singleton are not.
                if (arg is NDArray built && !ReferenceEquals(built, ops[0]._mask) && !ReferenceEquals(built, ops[0].data)
                    && !ReferenceEquals(built, np.ma.nomask))
                    built.Dispose();
            }
        }

        /// <summary>
        ///     ids' relational facet: <c>ids(view) - ids(op)</c> for the data and the mask address (bytes), where the view is
        ///     the case's index recipe applied to the masked operand. NumPy's ids is <c>(ctypes.data, mask.ctypes.data)</c>
        ///     (the address of the first element) or <c>id(nomask)</c>, so the deltas are byte offsets NumPy defines.
        /// </summary>
        /// <param name="p">The case params (<c>view</c>: "T" or a comma-separated slice tuple).</param>
        /// <param name="ops">The reconstructed masked operands.</param>
        /// <returns>An int64 array [data delta, mask delta].</returns>
        private static NDArray MaIdsDelta(IReadOnlyDictionary<string, JsonElement> p, NDMaskedArray[] ops)
        {
            var view = MaView(ops[0], SurfStr(p, "view"));
            var a = np.ma.ids(ops[0]);
            var b = np.ma.ids(view);
            ReleaseMaView(view, ops[0]);
            return np.array(new[] { b.data - a.data, b.mask - a.mask });
        }

        /// <summary>The masked operand indexed by an ids view recipe: "T" is the transpose; anything else a comma-separated
        /// index tuple whose parts are slice strings ("1:", "::-1") or ints.</summary>
        /// <param name="m">The masked operand.</param>
        /// <param name="recipe">The recipe.</param>
        /// <returns>The view.</returns>
        private static NDMaskedArray MaView(NDMaskedArray m, string recipe)
        {
            if (recipe == "T")
                return m.T;
            var parts = recipe.Split(',').Select(s => s.Trim())
                .Select(s => s.Contains(':') ? (object)s : int.Parse(s, CultureInfo.InvariantCulture)).ToArray();
            return (NDMaskedArray)m[parts];
        }

        /// <summary>Disposes the data and mask views a view recipe created and the stored fill the result holds of its own
        /// (never the operand's own parts): a view holds a counted reference to its owner's buffer, and a stored fill is
        /// a pooled 0-d array (numpy.ma's <c>_fill_value</c>), which the leak sweep would otherwise read as escapes.</summary>
        /// <param name="view">The view (or a result derived from the operand, such as masked_object's).</param>
        /// <param name="owner">The operand it was taken from.</param>
        /// <remarks>The fill is released only when it is not the owner's: a view shares its owner's stored fill (NumPy's
        /// <c>_update_from</c>), while a constructor such as <c>np.ma.masked_object</c> stores a converted copy of its
        /// own.</remarks>
        private static void ReleaseMaView(NDMaskedArray view, NDMaskedArray owner)
        {
            if (!ReferenceEquals(view.data, owner.data))
                view.data.Dispose();
            if (view._mask is not null && !ReferenceEquals(view._mask, owner._mask))
                view._mask.Dispose();
            if (view._fill_value is not null && !ReferenceEquals(view._fill_value, owner._fill_value))
                view._fill_value.Dispose();
        }

        /// <summary>
        ///     The harden_mask / soften_mask facets: <c>self</c> (the call returns the array itself), <c>hardmask</c> (the flag
        ///     after the steps), <c>assign</c> (the array after the steps and one indexer assignment: under a hard mask a
        ///     masked slot keeps its data and stays masked).
        /// </summary>
        /// <param name="name">"harden_mask" or "soften_mask".</param>
        /// <param name="p">The case params (<c>facet</c>; <c>steps</c> "hard" or "hard_soft"; <c>index</c>; <c>value</c>).</param>
        /// <param name="ops">The reconstructed masked operands.</param>
        /// <returns>A boxed bool (self / hardmask) or the mutated masked array (assign).</returns>
        /// <exception cref="NotSupportedException">An unknown facet or value form (a corpus/registry drift).</exception>
        private static object MaHardMask(string name, IReadOnlyDictionary<string, JsonElement> p, NDMaskedArray[] ops)
        {
            string facet = SurfStr(p, "facet");
            var a = ops[0];
            if (facet == "self")
                return ReferenceEquals(name == "harden_mask" ? np.ma.harden_mask(a) : np.ma.soften_mask(a), a);
            np.ma.harden_mask(a);
            if (SurfStr(p, "steps") == "hard_soft")
                np.ma.soften_mask(a);
            if (facet == "hardmask")
                return a.hardmask;
            if (facet != "assign")
                throw new NotSupportedException($"hard-mask facet '{facet}'");
            string idx = SurfStr(p, "index");
            object key = idx.Contains(':') ? idx : int.Parse(idx, CultureInfo.InvariantCulture);
            var v = p["value"];
            if (v.ValueKind == JsonValueKind.String && v.GetString() == "marr")
            {
                // ma.array([5, 6], mask=[True, False]).astype(a.dtype): a Python int list is int64. Every part is this
                // replay's and is released after the assignment (which copies what it needs), on a throw too.
                using var d = np.array(new long[] { 5, 6 });
                using var m = np.array(new[] { true, false });
                var cast = np.ma.array(d, m).astype(a.dtype);
                try
                {
                    a[key] = cast;
                }
                finally
                {
                    if (!ReferenceEquals(cast.data, d))
                        cast.data.Dispose();
                    if (cast._mask is not null && !ReferenceEquals(cast._mask, m))
                        cast._mask.Dispose();
                }
                return a;
            }
            object value = v.ValueKind == JsonValueKind.String
                ? v.GetString() switch
                {
                    "masked" => np.ma.masked,
                    var s => throw new NotSupportedException($"hard-mask value '{s}'"),
                }
                : MaPyValue(v);
            a[key] = value;
            return a;
        }

        // ---- putmask / products / axis helpers ------------------------------------------------------------------------------

        /// <summary>
        ///     <c>ma.putmask(a, mask, values)</c> with the case's steps: harden the target first (<c>hard</c>), take the
        ///     values from a param (a Python scalar, or "masked" for the singleton) or from the third operand. The mask is the
        ///     second operand's DATA (the generator passes NumPy the plain bool array). The mutated target is the result.
        /// </summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">[target, mask(, values)].</param>
        /// <returns>The target after the call.</returns>
        private static NDMaskedArray MaPutmask(IReadOnlyDictionary<string, JsonElement> p, NDMaskedArray[] ops)
        {
            var a = ops[0];
            if (p.ContainsKey("hard"))
                a.harden_mask();
            object values = p.TryGetValue("value", out var v)
                ? (v.ValueKind == JsonValueKind.String && v.GetString() == "masked" ? np.ma.masked : MaPyValue(v))
                : ops[2];
            np.ma.putmask(a, ops[1].data, values);
            return a;
        }

        /// <summary>convolve / correlate with the params a case passes (an absent mode / propagate_mask is the C# default,
        /// which is NumPy's: "full" for convolve, "valid" for correlate, propagate_mask=true).</summary>
        /// <param name="name">"convolve" or "correlate".</param>
        /// <param name="p">The case params.</param>
        /// <param name="ops">[a, v].</param>
        /// <returns>The masked product.</returns>
        private static NDMaskedArray MaSlidingProduct(string name, IReadOnlyDictionary<string, JsonElement> p, NDMaskedArray[] ops)
        {
            string mode = SurfNullableStr(p, "mode");
            bool? prop = p.TryGetValue("propagate_mask", out var pm) ? pm.GetBoolean() : null;
            bool conv = name == "convolve";
            if (mode is null)
                return prop is null
                    ? (conv ? np.ma.convolve(ops[0], ops[1]) : np.ma.correlate(ops[0], ops[1]))
                    : (conv ? np.ma.convolve(ops[0], ops[1], propagate_mask: prop.Value)
                            : np.ma.correlate(ops[0], ops[1], propagate_mask: prop.Value));
            return prop is null
                ? (conv ? np.ma.convolve(ops[0], ops[1], mode) : np.ma.correlate(ops[0], ops[1], mode))
                : (conv ? np.ma.convolve(ops[0], ops[1], mode, prop.Value) : np.ma.correlate(ops[0], ops[1], mode, prop.Value));
        }

        /// <summary>An optional int axis param of the compressors; a list (NumPy's axis tuple) has no C# form.</summary>
        /// <param name="p">The case params.</param>
        /// <param name="fn">The function name (for the refusal text).</param>
        /// <returns>The axis or null.</returns>
        /// <exception cref="NotSupportedException">A tuple axis.</exception>
        private static int? MaAxisParam(IReadOnlyDictionary<string, JsonElement> p, string fn)
        {
            if (!p.TryGetValue("axis", out var a) || a.ValueKind == JsonValueKind.Null)
                return null;
            if (a.ValueKind == JsonValueKind.Array)
                throw new NotSupportedException($"np.ma.{fn} takes an int axis only (NumPy also takes a tuple)");
            return a.GetInt32();
        }

        /// <summary>A bool param with a default (NumPy's keyword default when the case omits it).</summary>
        private static bool MaBool(IReadOnlyDictionary<string, JsonElement> p, string key, bool dflt)
            => p.TryGetValue(key, out var v) ? v.GetBoolean() : dflt;

        /// <summary>The apply_along_axis callable menu — the C# twin of <c>gen_ma_surface_oracle.ALONG_FUNCS</c>.</summary>
        /// <param name="name">The menu name.</param>
        /// <returns>The per-slice function.</returns>
        /// <exception cref="NotSupportedException">An unknown name (a corpus/registry drift).</exception>
        private static Func<NDMaskedArray, NDMaskedArray> AlongAxisMaFunc(string name) => name switch
        {
            "sum" => x => np.ma.sum(x),
            "max" => x => np.ma.max(x),
            "min" => x => np.ma.min(x),
            "mean" => x => np.ma.mean(x),
            "mul2" => x => np.ma.multiply(x, 2),
            "cumsum" => x => np.ma.cumsum(x),
            "rev" => x => (NDMaskedArray)x["::-1"],
            "sort" => x => np.ma.sort(x),
            _ => throw new NotSupportedException($"apply_along_axis menu function '{name}'"),
        };

        /// <summary>The apply_over_axes callable menu — the C# twin of <c>gen_ma_surface_oracle.OVER_FUNCS</c>.</summary>
        /// <param name="name">The menu name.</param>
        /// <returns>The (array, axis) function.</returns>
        /// <exception cref="NotSupportedException">An unknown name (a corpus/registry drift).</exception>
        private static Func<NDMaskedArray, int, NDMaskedArray> OverAxesMaFunc(string name) => name switch
        {
            "sum" => (x, a) => np.ma.sum(x, a),
            "max" => (x, a) => np.ma.max(x, a),
            "mean" => (x, a) => np.ma.mean(x, a),
            "sumkd" => (x, a) => np.ma.sum(x, a, keepdims: true),
            "cumsum" => (x, a) => np.ma.cumsum(x, a),
            _ => throw new NotSupportedException($"apply_over_axes menu function '{name}'"),
        };

        // ---- Python values and reprs ----------------------------------------------------------------------------------------

        /// <summary>
        ///     A Python scalar param as the C# literal NumPy's Python object stands for: int → <see cref="long"/> (a
        ///     <see cref="BigInteger"/> past long), float → <see cref="double"/> (the "nan" / "inf" / "-inf" strings),
        ///     bool, complex <c>{"re", "im"}</c> → <see cref="Complex"/>, None → null, any other string → the string.
        /// </summary>
        /// <param name="v">The param element.</param>
        /// <returns>The boxed value.</returns>
        private static object MaPyValue(JsonElement v)
        {
            switch (v.ValueKind)
            {
                case JsonValueKind.Null: return null;
                case JsonValueKind.True: return true;
                case JsonValueKind.False: return false;
                case JsonValueKind.Number:
                {
                    if (v.TryGetInt64(out long l))
                        return l;
                    string raw = v.GetRawText();
                    // An integral literal past long is a Python int (BigInteger); anything with a point / exponent a float.
                    return raw.IndexOfAny(new[] { '.', 'e', 'E' }) < 0
                        ? BigInteger.Parse(raw, CultureInfo.InvariantCulture)
                        : v.GetDouble();
                }
                case JsonValueKind.String:
                    return v.GetString() switch
                    {
                        "nan" => double.NaN,
                        "inf" => double.PositiveInfinity,
                        "-inf" => double.NegativeInfinity,
                        var s => s,
                    };
                case JsonValueKind.Object:
                    return new Complex(PyFloat(v.GetProperty("re")), PyFloat(v.GetProperty("im")));
                default:
                    throw new NotSupportedException($"Python value param of kind {v.ValueKind}");
            }
        }

        /// <summary>
        ///     The normalized Python repr of a value NumSharp returns where NumPy returns a Python object — the C# side of
        ///     <c>gen_ma_surface_oracle.pyrepr</c>: a 0-d array is its element (NumPy's <c>.item()</c>), an N-d array its
        ///     <c>array(...)</c> repr (NumSharp's np.array_repr port), a <see cref="Slice"/> <c>slice(start, stop, step)</c>,
        ///     the masked singleton <c>masked</c>, lists / tuples recursively, every scalar CPython's repr (<see cref="PyRepr"/>).
        /// </summary>
        /// <param name="v">The value.</param>
        /// <returns>The text.</returns>
        private static string MaRepr(object v)
        {
            switch (v)
            {
                case null: return "None";
                case NDMaskedConstant: return "masked";
                case NDArray a when a.ndim == 0: return MaRepr(a.GetAtIndex(0));
                case NDArray a: return a.ToString(true);
                case Slice s:
                    return $"slice({(s.Start is null ? "None" : s.Start.Value.ToString(CultureInfo.InvariantCulture))}, " +
                           $"{(s.Stop is null ? "None" : s.Stop.Value.ToString(CultureInfo.InvariantCulture))}, " +
                           $"{s.Step.ToString(CultureInfo.InvariantCulture)})";
                case string: return PyRepr(v);
                case ITuple t: return PyTupleText(Enumerable.Range(0, t.Length).Select(i => MaRepr(t[i])));
                case IList list:
                {
                    var sb = new StringBuilder("[");
                    for (int i = 0; i < list.Count; i++)
                    {
                        if (i > 0) sb.Append(", ");
                        sb.Append(MaRepr(list[i]));
                    }
                    return sb.Append(']').ToString();
                }
                default: return PyRepr(v);
            }
        }

        /// <summary>Python's tuple syntax over already-formatted items: <c>()</c>, <c>(x,)</c> or <c>(x, y)</c>.</summary>
        private static string PyTupleText(IEnumerable<string> items)
        {
            var parts = items.ToList();
            return parts.Count == 1 ? "(" + parts[0] + ",)" : "(" + string.Join(", ", parts) + ")";
        }

        /// <summary>The repr of notmasked_edges' explicit-axis result (NumPy's list of two tuples of arrays,
        /// <c>[(mins...), (maxs...)]</c>), after which the edge arrays — the caller's by NumSharp's contract — are
        /// released.</summary>
        /// <param name="pair">{mins, maxs}: one compressed edge array per dimension each.</param>
        /// <returns>The text.</returns>
        private static string MaEdgePairs(NDArray[][] pair)
        {
            string text = "[" + string.Join(", ", pair.Select(t => PyTupleText(t.Select(a => a.ToString(true))))) + "]";
            foreach (var side in pair)
                foreach (var a in side)
                    a.Dispose();
            return text;
        }

        /// <summary>The repr of an edge pair, which NumSharp returns as NumPy does: an int64 ndarray (or None for no
        /// unmasked element). The array is the replayed call's own result, so this method disposes it once its
        /// <c>array([...])</c> repr is read.</summary>
        /// <param name="e">The {first, last} pair, or null.</param>
        /// <returns>The text; <c>None</c> for a null pair.</returns>
        private static string MaEdgeArray(NDArray e)
        {
            if (e is null)
                return "None";
            string text = e.ToString(true);
            e.Dispose();
            return text;
        }
    }
}
