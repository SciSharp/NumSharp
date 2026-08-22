using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The non-single-array half of the registry: iterator TRACES, tuple results, dtype
    ///     results and text results. Pairs 1:1 with gen_oracle.py's <c>gen_iter</c> /
    ///     <c>gen_dtype_text</c> generators, exactly as <see cref="Apply"/> pairs with the rest.
    ///
    ///     <para>
    ///     <b>Why iterators can be gated at all.</b> np.ndindex / np.ndenumerate / np.nditer /
    ///     np.broadcast return no array, which is why they were left out of this corpus. But their
    ///     MATERIALIZED iteration is an ordinary array — and it is the artifact that matters, since
    ///     what these objects actually promise is an ORDER. Recording the trace turns "does NumSharp
    ///     traverse like NumPy" into an ordinary bit-comparison.
    ///     </para>
    ///
    ///     <para>
    ///     <b>The iteration protocol used here is NumPy's own</b> — <c>while (!it.finished) { read;
    ///     it.iternext(); }</c> — not <c>foreach</c>. np.NDIterator.MoveNext publishes THEN advances
    ///     (matching Python's <c>__next__</c>), so mixing the two styles would offset the trace by
    ///     one step. Values are read with GetAtIndex, which COPIES: <c>it[0]</c> aliases the
    ///     iterator's live data pointer (and, when buffered, a buffer the next step refills), so
    ///     holding the view instead of its value is the documented foot-gun.
    ///     </para>
    /// </summary>
    public static partial class OpRegistry
    {
        /// <summary>Route a case to the entry point its result kind implies.</summary>
        public static object Invoke(string kind, string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
            => kind switch
            {
                "tuple" => ApplyTuple(op, p, ops),
                "dtype" => ApplyDtype(op, p, ops),
                "text" => ApplyText(op, p, ops),
                _ => Apply(op, p, ops),
            };

        // ---- array / scalar results ---------------------------------------------------------

        /// <summary>
        ///     Ops whose result is one array (or a scalar wrapped as a 0-d array), reached from
        ///     <see cref="Apply"/>'s default branch.
        /// </summary>
        internal static NDArray ApplyExtended(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            switch (op)
            {
                // ---- iterator traces (array-shaped) ----
                //
                // np.ndindex has NO operand: the index space comes entirely from params, like the
                // `tri` generator already in Apply.
                case "ndindex":
                {
                    var shape = ParseLongArray(p["shape"]);
                    var rows = new List<long[]>();
                    foreach (var ix in np.ndindex(shape))
                        rows.Add(ix);
                    return IndexMatrix(rows, shape.Length);
                }

                // The value stream in iteration order — the whole point of the tier. A wrong
                // order/layout resolution shows up here and nowhere else in the corpus.
                case "nditer":
                case "nditer_values":
                {
                    var vals = new List<object>();
                    using (var it = np.nditer(ops[0], order: ParseOrder(p)))
                    {
                        while (!it.finished)
                        {
                            vals.Add(it[0].GetAtIndex(0));
                            it.iternext();
                        }
                    }
                    return ValueVector(vals, ops[0].typecode);
                }

                // nested_iters is object-returning, like nditer. Materializing the recursive
                // value stream makes its axis partition / child rebase contract byte-comparable.
                case "nested_iters":
                {
                    int[][] axes = p["axes"].EnumerateArray()
                        .Select(ParseIntArray)
                        .ToArray();
                    var vals = new List<object>();
                    var iters = np.nested_iters(ops[0], axes, flags: new[] { "multi_index" },
                                                order: ParseOrder(p));
                    try
                    {
                        void Walk(int level)
                        {
                            foreach (var _ in iters[level])
                            {
                                if (level == iters.Length - 1)
                                    vals.Add(iters[level][0].GetAtIndex(0));
                                else
                                    Walk(level + 1);
                            }
                        }

                        Walk(0);
                    }
                    finally
                    {
                        foreach (var it in iters)
                            it.Dispose();
                    }
                    return ValueVector(vals, ops[0].typecode);
                }

                // The tracked flat index stream (NumPy's c_index / f_index).
                case "nditer_index":
                {
                    var flag = p["index"].GetString();          // "c_index" | "f_index"
                    var idx = new List<object>();
                    using (var it = np.nditer(ops[0], flags: new[] { flag }, order: ParseOrder(p)))
                    {
                        while (!it.finished)
                        {
                            idx.Add(it.index);
                            it.iternext();
                        }
                    }
                    return ValueVector(idx, NPTypeCode.Int64);
                }

                // np.broadcast's resolved shape, as an array so it bit-compares like anything else.
                case "broadcast_shape":
                {
                    var b = np.broadcast(ops);
                    var dims = new List<object>();
                    foreach (var d in b.shape.Dimensions)
                        dims.Add(d);
                    return ValueVector(dims, NPTypeCode.Int64);
                }

                // ---- scalar results (the np.allclose pattern: wrap in a 0-d array) ----
                case "can_cast":
                    return NDArray.Scalar(np.can_cast(FuzzCorpus.DtypeToTC(p["from"].GetString()),
                                                      FuzzCorpus.DtypeToTC(p["to"].GetString()),
                                                      p["casting"].GetString()));
                case "isdtype":
                    return NDArray.Scalar(np.isdtype(FuzzCorpus.DtypeToTC(p["dtype"].GetString()),
                                                      p["kind_name"].GetString()));
                case "issubdtype":
                    return NDArray.Scalar(np.issubdtype(FuzzCorpus.DtypeToTC(p["a"].GetString()),
                                                        FuzzCorpus.DtypeToTC(p["b"].GetString())));
                case "isfortran":
                    return NDArray.Scalar(np.isfortran(ops[0]));
                case "iterable":
                    return NDArray.Scalar(np.iterable(ops[0]));
                case "isscalar":
                    return NDArray.Scalar(np.isscalar(ops[0]));
                case "iscomplexobj":
                    return NDArray.Scalar(np.iscomplexobj(ops[0]));
                case "isrealobj":
                    return NDArray.Scalar(np.isrealobj(ops[0]));
                case "size":
                    return NDArray.Scalar(np.size(ops[0],
                        p.TryGetValue("axis", out var ax) && ax.ValueKind != JsonValueKind.Null
                            ? ax.GetInt32() : (int?)null));

                default:
                    throw new NotSupportedException($"op '{op}' is not registered in OpRegistry");
            }
        }

        // ---- tuple results ------------------------------------------------------------------

        /// <summary>
        ///     Ops returning N arrays. The corpus records every slot, so ARITY is asserted too —
        ///     which the older which/piece params (one slot per case) structurally cannot do.
        /// </summary>
        public static NDArray[] ApplyTuple(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            switch (op)
            {
                // (index, value) for every element, in flatiter C-order whatever the layout.
                case "ndenumerate":
                {
                    var rows = new List<long[]>();
                    var vals = new List<object>();
                    foreach (var (index, value) in np.ndenumerate(ops[0]))
                    {
                        rows.Add(index);
                        vals.Add(value);
                    }
                    return new[]
                    {
                        IndexMatrix(rows, ops[0].ndim),
                        ValueVector(vals, ops[0].typecode)
                    };
                }

                // The multi_index stream alongside the values it labels.
                case "nditer_multi_index":
                {
                    var rows = new List<long[]>();
                    var vals = new List<object>();
                    using (var it = np.nditer(ops[0], flags: new[] { "multi_index" }, order: ParseOrder(p)))
                    {
                        while (!it.finished)
                        {
                            rows.Add(it.multi_index);
                            vals.Add(it[0].GetAtIndex(0));
                            it.iternext();
                        }
                    }
                    return new[]
                    {
                        IndexMatrix(rows, ops[0].ndim),
                        ValueVector(vals, ops[0].typecode)
                    };
                }

                // EXTERNAL_LOOP: the concatenated values plus the CHUNK LENGTHS. The chunking is
                // itself observable behavior (how the iterator coalesces dimensions), and nothing
                // else in the corpus can see it.
                case "nditer_extloop":
                {
                    var vals = new List<object>();
                    var lens = new List<object>();
                    using (var it = np.nditer(ops[0], flags: new[] { "external_loop" }, order: ParseOrder(p)))
                    {
                        while (!it.finished)
                        {
                            var chunk = it[0];
                            long n = chunk.size;
                            lens.Add(n);
                            for (long i = 0; i < n; i++)
                                vals.Add(chunk.GetAtIndex(i));
                            it.iternext();
                        }
                    }
                    return new[]
                    {
                        ValueVector(vals, ops[0].typecode),
                        ValueVector(lens, NPTypeCode.Int64)
                    };
                }

                // Two operands walked in lockstep (broadcasting inside the iterator).
                case "nditer_pair":
                {
                    var a = new List<object>();
                    var b = new List<object>();
                    using (var it = np.nditer(new[] { ops[0], ops[1] }, order: ParseOrder(p)))
                    {
                        while (!it.finished)
                        {
                            a.Add(it[0].GetAtIndex(0));
                            b.Add(it[1].GetAtIndex(0));
                            it.iternext();
                        }
                    }
                    return new[] { ValueVector(a, ops[0].typecode), ValueVector(b, ops[1].typecode) };
                }

                // np.broadcast's per-operand value streams.
                case "broadcast_values":
                {
                    var streams = new List<object>[ops.Length];
                    for (int i = 0; i < ops.Length; i++)
                        streams[i] = new List<object>();

                    foreach (var vals in np.broadcast(ops))
                        for (int i = 0; i < ops.Length; i++)
                            streams[i].Add(vals[i]);

                    var ret = new NDArray[ops.Length];
                    for (int i = 0; i < ops.Length; i++)
                        ret[i] = ValueVector(streams[i], ops[i].typecode);
                    return ret;
                }

                // np.nonzero over ANY rank — one index array per dimension. The existing `nonzero`
                // op in Apply is hardwired to slot [0] with 1-D cases only, so the second-and-later
                // arrays of an N-D nonzero were never gated.
                case "nonzero_all":
                {
                    var t = np.nonzero(ops[0]);
                    var ret = new NDArray[t.Length];
                    for (int i = 0; i < t.Length; i++)
                        ret[i] = t[i];
                    return ret;
                }

                case "meshgrid":
                {
                    return np.meshgrid(ops[0], ops[1],
                        indexing: p.TryGetValue("indexing", out var indexing) ? indexing.GetString() : "xy",
                        sparse: p.TryGetValue("sparse", out var sparse) && sparse.GetBoolean(),
                        copy: !p.TryGetValue("copy", out var copy) || copy.GetBoolean()).ToArray();
                }

                case "split":
                    return np.split(ops[0], p["sections"].GetInt32(), p["axis"].GetInt32());
                case "array_split":
                    return np.array_split(ops[0], p["sections"].GetInt32(), p["axis"].GetInt32());
                case "hsplit":
                    return np.hsplit(ops[0], p["sections"].GetInt32());
                case "vsplit":
                    return np.vsplit(ops[0], p["sections"].GetInt32());
                case "dsplit":
                    return np.dsplit(ops[0], p["sections"].GetInt32());
                case "unstack":
                    return np.unstack(ops[0], p["axis"].GetInt32());
                case "modf":
                {
                    var (fractional, integral) = np.modf(ops[0]);
                    return new[] { fractional, integral };
                }
                case "average_returned":
                {
                    int? axis = p["axis"].ValueKind == JsonValueKind.Null
                        ? (int?)null
                        : p["axis"].GetInt32();
                    NDArray weights = p["weighted"].GetBoolean() ? ops[1] : null;
                    var (average, sumOfWeights) = np.average_returned(ops[0], axis, weights,
                        p["keepdims"].GetBoolean());
                    return new[] { average, sumOfWeights };
                }
                case "unique_counts":
                    return np.unique_counts(ops[0]).ToArray();
                case "unique_inverse":
                    return np.unique_inverse(ops[0]).ToArray();
                case "unique_all":
                    return np.unique_all(ops[0]).ToArray();

                // ---- ufunc out= / where= --------------------------------------------------
                //
                // TWO slots on purpose. Slot 0 is what the call returned; slot 1 is the ENTIRE
                // base buffer behind `out`. The second is what gives this tier teeth:
                //
                //   * `where` masking is defined by what does NOT change — masked-off slots keep
                //     their prior contents — and "unchanged" is only observable if the prior
                //     contents are recorded and re-checked. The operand descriptor carries them.
                //   * when `out` is a STRIDED, OFFSET or NEGSTRIDE view, a kernel that walks the
                //     buffer instead of the view corrupts elements OUTSIDE the window. A
                //     view-shaped comparison cannot see that; the base buffer can.
                case "out_binary":
                case "out_unary":
                {
                    bool binary = op == "out_binary";
                    string ufunc = p["ufunc"].GetString();
                    int outIndex = binary ? 2 : 1;

                    var target = ops[outIndex];
                    var mask = p.TryGetValue("where", out var w) && w.GetBoolean()
                        ? ops[outIndex + 1]
                        : null;

                    var returned = binary
                        ? ApplyBinaryOut(ufunc, ops[0], ops[1], target, mask)
                        : ApplyUnaryOut(ufunc, ops[0], target, mask);

                    return new[] { returned, BaseBuffer(target) };
                }

                case "unravel_index_all":
                    return np.unravel_index(ops[0], ParseIntArray(p["shape"]),
                        p.TryGetValue("order", out var uriOrder) ? uriOrder.GetString()[0] : 'C');

                case "broadcast_arrays":
                    return np.broadcast_arrays(ops);
                case "indices_sparse":
                    return np.indices_sparse(ParseIntArray(p["dimensions"]),
                        FuzzCorpus.DtypeToTC(p["dtype"].GetString()));

                // ---- LAPACK factorisation family (linalg_parity tier; TUPLE results) ----
                // svd/eig/eigh/qr return multiple arrays; lstsq returns four. Computable ONLY
                // through NumSharp.Interop.OpenBLAS (the host-pinned gate enables it). ARITY is
                // asserted by CompareTuple. Array-result siblings (cholesky/eigvals/eigvalsh/
                // svdvals/pinv/matrix_rank/cond/norm, svd compute_uv=false, qr mode='r') are in
                // OpRegistry.cs::Apply. Params pair 1:1 with gen_oracle.py::gen_linalg_parity.
                case "svd":
                {
                    var (u, s, vh) = np.linalg.svd(ops[0], full_matrices: p["full_matrices"].GetBoolean());
                    return new[] { u, s, vh };
                }
                case "eig":
                {
                    var (w, v) = np.linalg.eig(ops[0]);
                    return new[] { w, v };
                }
                case "eigh":
                {
                    var (w, v) = np.linalg.eigh(ops[0], ParseUplo(p));
                    return new[] { w, v };
                }
                case "qr":                                                          // reduced/complete/raw
                {
                    var (q, r) = np.linalg.qr(ops[0], p["mode"].GetString());
                    return new[] { q, r };
                }
                case "lstsq":
                {
                    var (x, res, rank, sv) = np.linalg.lstsq(ops[0], ops[1]);
                    return new[] { x, res, rank, sv };
                }
                // slogdet -> (sign, logabsdet). LU (getrf), so byte-reproducible. For a complex
                // operand `sign` is a unit-modulus COMPLEX and `logabsdet` stays real float —
                // CompareTuple asserts both the arity and each slot's dtype. Array siblings
                // (det/inv/solve/tensorinv/tensorsolve) are in OpRegistry.cs::Apply.
                case "slogdet":
                {
                    var (sign, logabsdet) = np.linalg.slogdet(ops[0]);
                    return new[] { sign, logabsdet };
                }

                // polydiv -> (quotient, remainder). Portable (polynomial long division).
                case "polydiv":
                {
                    var (q, r) = np.polydiv(ops[0], ops[1]);
                    return new[] { q, r };
                }

                default:
                    throw new NotSupportedException($"tuple op '{op}' is not registered in OpRegistry");
            }
        }

        // ---- dtype results ------------------------------------------------------------------

        /// <summary>
        ///     Promotion helpers, whose result IS a dtype. Compared by NumPy dtype name, so the
        ///     NEP50 promotion table gets the same differential treatment as array values.
        /// </summary>
        public static NPTypeCode ApplyDtype(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            switch (op)
            {
                case "result_type_arrays":
                    return np.result_type(ops);
                case "result_type_dtypes":
                    return np.result_type(FuzzCorpus.DtypeToTC(p["a"].GetString()),
                                          FuzzCorpus.DtypeToTC(p["b"].GetString()));
                case "promote_types":
                    return np.promote_types(FuzzCorpus.DtypeToTC(p["a"].GetString()),
                                            FuzzCorpus.DtypeToTC(p["b"].GetString()));
                case "min_scalar_type":
                    return np.min_scalar_type(ParseScalar(p["value"]));
                case "dtype":
                    return np.dtype(p["dtype"].GetString()).typecode;
                case "common_type":
                    return np.common_type_code(ops);
                default:
                    throw new NotSupportedException($"dtype op '{op}' is not registered in OpRegistry");
            }
        }

        // ---- text results -------------------------------------------------------------------

        /// <summary>
        ///     Printing. NumSharp claims a byte-exact port of NumPy 2.4.2's arrayprint, and this is
        ///     where that claim becomes a gate rather than a one-off validation.
        /// </summary>
        public static string ApplyText(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            switch (op)
            {
                case "array_str": return np.array_str(ops[0]);
                case "array_repr": return np.array_repr(ops[0]);
                case "mintypecode": return np.mintypecode(p["typechars"].GetString()).ToString();
                // einsum_path returns the contraction planner's info STRING (shape-derived, so
                // the operand values are irrelevant). Byte-identical to NumPy for non-ellipsis
                // subscripts; the tuple's path structure is encoded in this same string.
                case "einsum_path":
                    return np.einsum_path(p["subscripts"].GetString(), ops,
                                          (object)p["optimize"].GetString()).repr;
                case "savetxt":
                {
                    using var writer = new StringWriter(CultureInfo.InvariantCulture);
                    np.savetxt(writer, ops[0], p["fmt"].GetString(), p["delimiter"].GetString(),
                               p["newline"].GetString(), p["header"].GetString(),
                               p["footer"].GetString(), p["comments"].GetString());
                    return writer.ToString();
                }
                case "get_state":
                {
                    np.random.seed(p["seed"].GetUInt32());
                    int draws = p["draws"].GetInt32();
                    if (draws > 0)
                        _ = np.random.random_sample(draws);
                    var state = np.random.get_state();
                    ulong bits = unchecked((ulong)BitConverter.DoubleToInt64Bits(state.CachedGaussian));
                    return state.Algorithm + "|" + state.Pos.ToString(CultureInfo.InvariantCulture) + "|" +
                           state.HasGauss.ToString(CultureInfo.InvariantCulture) + "|" +
                           bits.ToString("x16", CultureInfo.InvariantCulture) + "|" +
                           string.Join(",", state.Key.Select(x => x.ToString(CultureInfo.InvariantCulture)));
                }
                default:
                    throw new NotSupportedException($"text op '{op}' is not registered in OpRegistry");
            }
        }

        // ---- helpers ------------------------------------------------------------------------

        /// <summary>
        ///     The WHOLE buffer behind an operand, as a flat 1-D array. FuzzCorpus.Reconstruct
        ///     builds every operand as a view aliasing a 1-D contiguous storage of exactly
        ///     bufferSize elements, so a length-bufferSize vector over the same storage is that
        ///     buffer — including the elements the view's window does not address.
        /// </summary>
        private static NDArray BaseBuffer(NDArray view)
            => new NDArray(view.Storage, Shape.Vector(view.Shape.bufferSize));

        /// <summary>
        ///     NumPy's <c>f(x1, x2, out=…, where=…)</c>. Passing <c>where: null</c> is NumPy's
        ///     default <c>where=True</c> (compute every element), which is NOT the same as a
        ///     mask of all False — hence the tier generates both.
        /// </summary>
        private static NDArray ApplyBinaryOut(string ufunc, NDArray a, NDArray b, NDArray o, NDArray w)
            => ufunc switch
            {
                "add" => np.add(a, b, o, w),
                "subtract" => np.subtract(a, b, o, w),
                "multiply" => np.multiply(a, b, o, w),
                "divide" => np.divide(a, b, o, w),
                "power" => np.power(a, b, o, w),
                "mod" => np.mod(a, b, o, w),
                "floor_divide" => np.floor_divide(a, b, o, w),
                "arctan2" => np.arctan2(a, b, o, w),
                "bitwise_and" => np.bitwise_and(a, b, o, w),
                "bitwise_or" => np.bitwise_or(a, b, o, w),
                "bitwise_xor" => np.bitwise_xor(a, b, o, w),
                "less" => np.less(a, b, o, w),
                "greater_equal" => np.greater_equal(a, b, o, w),
                "equal" => np.equal(a, b, o, w),
                _ => throw new NotSupportedException($"out_binary ufunc '{ufunc}'")
            };

        private static NDArray ApplyUnaryOut(string ufunc, NDArray x, NDArray o, NDArray w)
            => ufunc switch
            {
                "sqrt" => np.sqrt(x, o, w),
                "negative" => np.negative(x, o, w),
                "abs" => np.abs(x, o, w),
                "square" => np.square(x, o, w),
                "exp" => np.exp(x, o, w),
                "log" => np.log(x, o, w),
                "sin" => np.sin(x, o, w),
                "floor" => np.floor(x, o, w),
                "ceil" => np.ceil(x, o, w),
                "rint" => np.rint(x, o, w),
                "sign" => np.sign(x, o, w),
                "reciprocal" => np.reciprocal(x, o, w),
                "invert" => np.invert(x, o, w),
                "isnan" => np.isnan(x, o, w),
                _ => throw new NotSupportedException($"out_unary ufunc '{ufunc}'")
            };

        /// <summary>An (N, ndim) int64 index matrix — NumPy's <c>np.array(list_of_indices)</c>.</summary>
        private static NDArray IndexMatrix(List<long[]> rows, int ndim)
        {
            var ret = new NDArray(NPTypeCode.Int64, new Shape(rows.Count, ndim));
            for (int r = 0; r < rows.Count; r++)
                for (int k = 0; k < ndim; k++)
                    ret.SetAtIndex(rows[r][k], (long)r * ndim + k);
            return ret;
        }

        /// <summary>A 1-D array of the boxed values, in the order they were produced.</summary>
        private static NDArray ValueVector(IReadOnlyList<object> vals, NPTypeCode tc)
        {
            var ret = new NDArray(tc, Shape.Vector(vals.Count));
            for (int i = 0; i < vals.Count; i++)
                ret.SetAtIndex(vals[i], i);
            return ret;
        }

        /// <summary>Iteration order; absent means NumPy's default 'K'.</summary>
        private static char ParseOrder(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("order", out var o) && o.ValueKind == JsonValueKind.String
                ? o.GetString()[0]
                : 'K';

        /// <summary>
        ///     A JSON scalar as the boxed C# type NumPy's own scalar would be — int64 for an
        ///     integer, double for a float, bool for a bool. min_scalar_type's answer depends on
        ///     it, so the distinction is load-bearing.
        /// </summary>
        private static object ParseScalar(JsonElement v) => v.ValueKind switch
        {
            JsonValueKind.True or JsonValueKind.False => v.GetBoolean(),
            JsonValueKind.Number when v.TryGetInt64(out var l) => l,
            JsonValueKind.Number => v.GetDouble(),
            _ => throw new NotSupportedException($"scalar kind {v.ValueKind}")
        };
    }
}
