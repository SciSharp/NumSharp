using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The <c>ndarray.*</c> INSTANCE half of the registry (coverage plan §D / row G0) plus the
    ///     <c>emath.*</c> scimath module (§A2/E5). Pairs 1:1 with <c>gen_oracle.py</c>'s
    ///     <c>gen_instance</c> / <c>gen_emath</c>, exactly as <see cref="OpRegistry.Apply"/> pairs
    ///     with the value generators.
    ///
    ///     <para>
    ///     <b>Why instance forms get their own corpus keys.</b> The registry dispatched
    ///     <c>np.foo(a)</c> ~300 times against <c>a.foo()</c> ~5, so instance-default and overload
    ///     divergences (<c>a.max(axis)</c> vs <c>np.max</c>, <c>a.reshape(-1)</c>,
    ///     <c>a.round(n)</c>, the in-place <c>a.sort()</c>) had no differential coverage at all —
    ///     the instance methods are frequently DIFFERENT code paths from the static twins. Keys
    ///     carry the <c>ndarray.</c> prefix (the <c>ma.</c> convention) so
    ///     <c>OracleSurfaceCoverageTests</c> discovers instance coverage from the corpus and
    ///     <see cref="MisalignedRegistry"/> strips the prefix before classifying, letting the
    ///     shared excuse branches (float var/std accumulation order, complex ULP envelopes) apply
    ///     to the instance spelling exactly as to the <c>np.*</c> one.
    ///     </para>
    ///
    ///     <para>
    ///     <b>In-place mutators use the out_where two-slot contract</b> (§D3's operand-after
    ///     comparator, expressed with the existing tuple machinery): slot 0 is the post-call VIEW,
    ///     slot 1 the entire post-call BASE buffer — so a mutator that writes outside a strided
    ///     view's window is caught, which a view-shaped comparison cannot see.
    ///     </para>
    /// </summary>
    public static partial class OpRegistry
    {
        /// <summary>
        ///     Dispatch an <c>ndarray.&lt;method&gt;</c> corpus op whose result is a single array or
        ///     a scalar wrapped 0-d (the <c>array</c>/<c>scalar</c> kinds). The tuple-kind ops
        ///     (nonzero + the in-place mutators) live in <see cref="ApplyInstanceTuple"/>.
        /// </summary>
        /// <param name="name">The bare member name, <c>ndarray.</c> prefix already stripped.</param>
        /// <param name="p">The case's params dictionary (axis/decimals/shape/… per generator job).</param>
        /// <param name="ops">Reconstructed operands; <c>ops[0]</c> is always the receiver.</param>
        /// <returns>The instance call's result (a fresh array, a view, or a wrapped scalar).</returns>
        /// <exception cref="NotSupportedException">The name has no registered instance case — a
        ///     corpus/registry drift that must fail the tier loudly (the registry contract).</exception>
        internal static NDArray ApplyInstance(string name, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var a = ops[0];
            switch (name)
            {
                // ---- reductions through the INSTANCE overloads (defaults differ from np.*) ----
                case "all": return a.all(ParseAxis(p));
                case "any": return a.any(ParseAxis(p));
                case "max": return ParseAxis(p) is int mxAx ? a.max(mxAx) : a.max();
                case "min": return ParseAxis(p) is int mnAx ? a.min(mnAx) : a.min();
                case "mean": return ParseAxis(p) is int meAx ? a.mean(meAx) : a.mean();
                case "sum": return ParseAxis(p) is int suAx ? a.sum(suAx) : a.sum();
                case "prod": return a.prod(ParseAxis(p));
                case "std": return ParseAxis(p) is int sdAx ? a.std(sdAx) : a.std();
                case "var": return ParseAxis(p) is int vrAx ? a.var(vrAx) : a.var();
                case "argmax": return a.argmax(p["axis"].GetInt32());
                case "argmin": return a.argmin(p["axis"].GetInt32());
                case "argsort": return a.argsort(p["axis"].GetInt32());
                case "argpartition": return a.argpartition(p["kth"].GetInt32(), p["axis"].GetInt32());

                // ---- conversions / copies / reshapes ----
                case "astype": return a.astype(FuzzCorpus.DtypeToTC(p["dtype"].GetString()));
                case "copy": return a.copy();
                case "ravel": return a.ravel();
                case "flatten": return a.flatten(p["order"].GetString()[0]);
                case "reshape":
                {
                    // The instance reshape takes params long[] and resolves -1 like NumPy; the
                    // corpus spells the request exactly as the Python call did (a.reshape(-1)).
                    int[] shape = ParseIntArray(p["shape"]);
                    var dims = new long[shape.Length];
                    for (int i = 0; i < shape.Length; i++)
                        dims[i] = shape[i];
                    return a.reshape(dims);
                }
                case "squeeze": return a.squeeze();
                case "transpose":
                    return p.ContainsKey("axes") ? a.transpose(ParseIntArray(p["axes"])) : a.transpose();
                case "swapaxes": return a.swapaxes(p["a1"].GetInt32(), p["a2"].GetInt32());
                case "diagonal": return a.diagonal(p["offset"].GetInt32());
                case "trace": return a.trace(p["offset"].GetInt32());

                // ---- elementwise / scan / selection instance forms ----
                case "conj": return a.conj();
                case "cumsum": return a.cumsum(ParseAxis(p));
                case "cumprod": return a.cumprod(ParseAxis(p));
                case "clip": return a.clip(p["lo"].GetInt32(), p["hi"].GetInt32());
                case "round": return a.round(p["decimals"].GetInt32());
                case "repeat": return a.repeat(p["repeats"].GetInt32());
                case "compress": return a.compress(ops[1], ParseAxis(p));
                case "take": return a.take(ops[1], ParseAxis(p));
                case "searchsorted": return a.searchsorted(ops[1], p["side"].GetString());
                case "choose": return a.choose(new[] { ops[1], ops[2] });
                case "dot": return a.dot(ops[1]);

                // ---- reinterpret family ----
                case "view":
                    return p.ContainsKey("dtype")
                        ? a.view(FuzzCorpus.DtypeToTC(p["dtype"].GetString()))
                        : a.view();
                case "byteswap": return a.byteswap();
                case "getfield":
                    return a.getfield(FuzzCorpus.DtypeToTC(p["dtype"].GetString()),
                                      p["offset"].GetInt32());

                // ---- scalar-kind: item / len / the D4 property scalars ----
                //
                // NumPy's a.item() returns the nearest PYTHON scalar (bool / int / float /
                // complex), and the generator records np.asarray(that) — so bool stays bool,
                // every integer lane (char included) lands int64, every float lane float64
                // (an exact widening for float16/float32), complex stays complex128. The
                // uint64 lane is excluded by the generator (a > 2^63-1 value re-narrows to
                // uint64 on the NumPy side, which this mapping cannot express).
                case "item":
                    return ItemScalar(p.TryGetValue("index", out var ix)
                        ? a.item(ix.GetInt64())
                        : a.item());
                case "__len__": return NDArray.Scalar(a.__len__());
                case "nbytes": return NDArray.Scalar(a.nbytes);
                case "itemsize": return NDArray.Scalar((long)a.itemsize);
                case "ndim": return NDArray.Scalar((long)a.ndim);
                case "size": return NDArray.Scalar((long)a.size);
                case "strides":
                    // NumPy strides are BYTES and so are NumSharp's (the strides->bytes parity
                    // change); recorded as an int64 vector.
                    return np.array(a.strides);

                // ---- D4 array-kind property reads ----
                case "T": return a.T;
                case "mT": return a.mT;
                case "real": return a.real;
                case "imag": return a.imag;
                case "flat": return a.flat;

                // The raw C-/F-order element bytes as a uint8 vector — the one instance-only
                // method whose result is bytes, not an array of the receiver's dtype.
                case "tobytes": return np.array(a.tobytes(p["order"].GetString()[0]));

                // In-place with ARRAY kind: after resize the OLD base buffer no longer exists
                // (the storage was reallocated/relabelled), so the mutated array itself is the
                // whole observable state — unlike sort/fill, which keep their base and ride the
                // two-slot tuple contract in ApplyInstanceTuple.
                case "resize":
                {
                    // FuzzCorpus.Reconstruct wraps the corpus bytes WITHOUT ownership, and
                    // ndarray.resize refuses a non-owning array by contract ("cannot resize this
                    // array: it does not own its data") — a harness artifact, not a parity gap.
                    // The VALUE claim (shrink truncates, grow zero-fills, C-order relabel) is
                    // therefore gated on an owning copy of the same bytes; the ownership/refcheck
                    // contract itself stays unit-suite territory (Manipulation/NDArray.resize).
                    var owned = a.copy();
                    int[] shape = ParseIntArray(p["shape"]);
                    var dims = new long[shape.Length];
                    for (int i = 0; i < shape.Length; i++)
                        dims[i] = shape[i];
                    owned.resize(dims);
                    return owned;
                }

                default:
                    throw new NotSupportedException($"instance op 'ndarray.{name}' is not registered in OpRegistry");
            }
        }

        /// <summary>
        ///     Dispatch an <c>ndarray.&lt;method&gt;</c> tuple-kind op: <c>nonzero</c> (arity ==
        ///     ndim, asserted by CompareTuple) and the IN-PLACE mutators, whose two slots are
        ///     [post-call view, post-call whole base buffer] — NumPy's post-call operand is the
        ///     oracle, and the base slot catches writes outside a strided view's window.
        /// </summary>
        /// <param name="name">The bare member name, <c>ndarray.</c> prefix already stripped.</param>
        /// <param name="p">The case's params (axis/kth/value/mode per generator job).</param>
        /// <param name="ops">Reconstructed operands; <c>ops[0]</c> is the receiver and — for the
        ///     mutators — the array MUTATED in place (put also reads indices/values from
        ///     <c>ops[1..2]</c>).</param>
        /// <returns>The tuple slots in the generator's recorded order.</returns>
        /// <exception cref="NotSupportedException">The name has no registered tuple case (registry
        ///     drift — must fail the tier loudly).</exception>
        internal static NDArray[] ApplyInstanceTuple(string name, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var a = ops[0];
            switch (name)
            {
                case "nonzero": return a.nonzero();

                case "sort":
                    a.sort(p["axis"].GetInt32());
                    return MutatedOperand(a);
                case "partition":
                    a.partition(p["kth"].GetInt32(), p["axis"].GetInt32());
                    return MutatedOperand(a);
                case "fill":
                    // The generator emits `true` for bool receivers and an integer otherwise;
                    // fill's own CoerceFillValue performs the NumPy scalar coercion (and raises
                    // on a non-writeable/overflowing target), so the raw boxed value is passed.
                    a.fill(p["value"].ValueKind is JsonValueKind.True or JsonValueKind.False
                        ? p["value"].GetBoolean()
                        : (object)p["value"].GetInt64());
                    return MutatedOperand(a);
                case "put":
                    a.put(ops[1], ops[2], p["mode"].GetString());
                    return MutatedOperand(a);

                default:
                    throw new NotSupportedException($"instance tuple op 'ndarray.{name}' is not registered in OpRegistry");
            }
        }

        /// <summary>
        ///     Dispatch an <c>np.emath.*</c> corpus op (gen_emath). All single-array except
        ///     <c>logn</c> (operands [n, x] — NumPy's argument order) and <c>power</c>
        ///     (operands [x, p]).
        /// </summary>
        /// <param name="name">The bare member name, <c>emath.</c> prefix already stripped.</param>
        /// <param name="p">The case's params (empty for the whole module today).</param>
        /// <param name="ops">Reconstructed operands per the generator's recorded order.</param>
        /// <returns>The scimath result — complex128 whenever the promotion rule triggered.</returns>
        /// <exception cref="NotSupportedException">The name has no registered emath case.</exception>
        internal static NDArray ApplyEmath(string name, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
            => name switch
            {
                "sqrt" => np.emath.sqrt(ops[0]),
                "log" => np.emath.log(ops[0]),
                "log2" => np.emath.log2(ops[0]),
                "log10" => np.emath.log10(ops[0]),
                "arccos" => np.emath.arccos(ops[0]),
                "arcsin" => np.emath.arcsin(ops[0]),
                "arctanh" => np.emath.arctanh(ops[0]),
                "logn" => np.emath.logn(ops[0], ops[1]),
                "power" => np.emath.power(ops[0], ops[1]),
                _ => throw new NotSupportedException($"emath op 'emath.{name}' is not registered in OpRegistry"),
            };

        /// <summary>
        ///     The two-slot result for an in-place mutator: the mutated view itself plus the ENTIRE
        ///     base buffer behind it (via the shared <see cref="BaseBuffer"/> helper) — mirroring
        ///     the out_where contract so out-of-window corruption is visible to the comparator.
        /// </summary>
        /// <param name="a">The operand that was just mutated in place.</param>
        /// <returns>[mutated view, whole base buffer as a flat vector].</returns>
        private static NDArray[] MutatedOperand(NDArray a) => new[] { a, BaseBuffer(a) };

        /// <summary>
        ///     Wrap <c>a.item()</c>'s boxed element as the 0-d array NumPy's
        ///     <c>np.asarray(a.item())</c> would produce: Python-scalar semantics, so bool stays
        ///     bool, all integer lanes (char included) widen to int64, all float lanes widen to
        ///     float64 (exact for float16/float32), complex stays complex128. Decimal — NumSharp's
        ///     NumPy-less dtype — maps to float64 for symmetry, though the generator never emits it.
        /// </summary>
        /// <param name="v">The boxed element <see cref="NDArray.item()"/> returned.</param>
        /// <returns>A 0-d NDArray of the Python-scalar dtype.</returns>
        /// <exception cref="OverflowException">An integer lane held a value outside int64 —
        ///     unreachable because the generator excludes the uint64 lane (the one lane whose
        ///     Python int can exceed int64).</exception>
        private static NDArray ItemScalar(object v) => v switch
        {
            bool b => NDArray.Scalar(b),
            Complex cx => NDArray.Scalar(cx),
            Half h => NDArray.Scalar((double)h),
            float f => NDArray.Scalar((double)f),
            double d => NDArray.Scalar(d),
            decimal m => NDArray.Scalar((double)m),
            char c => NDArray.Scalar((long)c),
            _ => NDArray.Scalar(Convert.ToInt64(v)),
        };
    }
}
