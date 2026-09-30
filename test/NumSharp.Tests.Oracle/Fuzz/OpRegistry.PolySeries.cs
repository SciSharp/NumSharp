using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The <c>numpy.polynomial</c> additive family and <c>polyutils</c> (plan U1): <c>{p}add</c>, <c>{p}sub</c>,
    ///     <c>{p}trim</c>, <c>{p}line</c>, the <c>{p}domain/zero/one/x</c> constants of the six bases, and
    ///     <c>polyutils.as_series/trimseq/trimcoef/getdomain/mapparms/mapdomain</c>. Pairs 1:1 with
    ///     <c>gen_oracle.py</c>'s <c>gen_polyseries</c>.
    ///
    ///     <para>
    ///     <b>Arguments are named</b> (<c>c1</c>, <c>c2</c>, <c>c</c>, <c>tol</c>, <c>off</c>, <c>scl</c>, <c>old</c>,
    ///     <c>new</c>, <c>x</c>, <c>alist</c>, <c>seq</c>). Each is <c>"a"</c> (the next operand), a Python scalar
    ///     (rebuilt as the C# primitive meaning that Python type), a Python list (<c>object[]</c>), a Python tuple
    ///     (a <see cref="ValueTuple"/>) or a str — so the facade sees the same KIND of argument NumPy saw, which
    ///     decides CPython-vs-NumPy arithmetic in mapparms/mapdomain/line. Operands are consumed in decode order,
    ///     the generator's encode order.
    ///     </para>
    /// </summary>
    public static partial class OpRegistry
    {
        /// <summary>Sequential decoder of a case's named arguments (operands consumed in call order).</summary>
        private sealed class PolyArgs
        {
            private readonly IReadOnlyDictionary<string, JsonElement> _p;
            private readonly NDArray[] _ops;
            private int _next;

            /// <summary>Wraps a case.</summary>
            /// <param name="p">The params.</param>
            /// <param name="ops">The reconstructed operands.</param>
            public PolyArgs(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops) { _p = p; _ops = ops; }

            /// <summary>Whether the case carries argument <paramref name="name"/>.</summary>
            /// <param name="name">The argument name.</param>
            /// <returns>True when present.</returns>
            public bool Has(string name) => _p.ContainsKey(name);

            /// <summary>Decodes argument <paramref name="name"/>.</summary>
            /// <param name="name">The argument name.</param>
            /// <returns>The C# value.</returns>
            public object Get(string name) => Decode(_p[name]);

            /// <summary>
            ///     Decodes one spec that is not a named argument — an entry of the evaluation family's <c>xs</c> list —
            ///     consuming operands in the same order as the named decoder (<see cref="Consumed"/> reports how many).
            /// </summary>
            /// <param name="e">The spec: <c>"a"</c>, a Python scalar, a list / tuple (nested, holding operands), or a str.</param>
            /// <returns>The C# value.</returns>
            /// <exception cref="NotSupportedException">An unknown spec kind or a tuple of more than 8 items.</exception>
            public object Spec(JsonElement e) => Decode(e);

            /// <summary>How many operands the decoded specs have consumed (the next operand's index).</summary>
            public int Consumed => _next;

            /// <summary>Decodes an optional argument: null when absent — the calculus facades' spelling of NumPy's default
            ///     (<c>k=[]</c>, <c>lbnd=0</c>, <c>scl=1</c>).</summary>
            /// <param name="name">The argument name.</param>
            /// <returns>The C# value, or null.</returns>
            public object Opt(string name) => _p.ContainsKey(name) ? Decode(_p[name]) : null;

            /// <summary>
            ///     Reads an optional-int argument written as a plain JSON number or JSON null (the series algebra's
            ///     <c>maxpower</c>, where null is NumPy's explicit <c>None</c>).
            /// </summary>
            /// <param name="name">The argument name (must be present).</param>
            /// <returns>The int, or null for JSON null.</returns>
            public int? NullableInt(string name) => _p[name].ValueKind == JsonValueKind.Null ? null : _p[name].GetInt32();

            /// <summary>The raw JSON of argument <paramref name="name"/> (must be present): how the series algebra tells a
            ///     plain <c>maxpower</c> (a JSON int or null) from a spec-encoded one (any other kind).</summary>
            /// <param name="name">The argument name.</param>
            /// <returns>The JSON element.</returns>
            public JsonElement Raw(string name) => _p[name];

            /// <summary>Reads an int argument written as a plain JSON number (the calculus <c>m</c> and <c>axis</c>).</summary>
            /// <param name="name">The argument name.</param>
            /// <param name="fallback">NumPy's default when absent.</param>
            /// <returns>The int.</returns>
            public int Int(string name, int fallback) => _p.TryGetValue(name, out var v) ? v.GetInt32() : fallback;

            /// <summary>
            ///     The calculus result as the corpus records it: the array itself, or — for a <c>"facet": "flags"</c> case —
            ///     the bool array <c>[C_CONTIGUOUS, F_CONTIGUOUS, OWNDATA]</c> of the result, NumPy's layout contract
            ///     (moveaxis views of fresh C-order buffers, K-order copies, NpyIter-allocated <c>c[:1]*0</c>).
            /// </summary>
            /// <param name="r">The facade's result (always a fresh array or a view of a fresh buffer — never an operand).</param>
            /// <returns>The recorded array.</returns>
            public NDArray CalcFacet(NDArray r)
            {
                if (!_p.TryGetValue("facet", out var f) || f.GetString() != "flags")
                    return r;
                var flags = new[] { r.flags.c_contiguous, r.flags.f_contiguous, r.flags.owndata };
                // The facet REPLACES the result, so the harness never sees (or disposes) it: release it here, or the
                // leak gate (UndisposedIntermediateTests) reads its buffer as escaped on every flags case.
                r.Dispose();
                return np.array(flags);
            }

            /// <summary>Decodes a bool parameter (as_series' <c>trim</c>), written as a Python-bool spec or a JSON bool.</summary>
            /// <param name="name">The parameter name.</param>
            /// <param name="fallback">Value when absent.</param>
            /// <returns>The flag.</returns>
            public bool Flag(string name, bool fallback)
            {
                if (!_p.TryGetValue(name, out var v))
                    return fallback;
                return v.ValueKind == JsonValueKind.Object ? v.GetProperty("bool").GetBoolean() : v.GetBoolean();
            }

            /// <summary>One spec → its C# value (recursive for lists/tuples, depth-first operand consumption).</summary>
            /// <param name="e">The spec.</param>
            /// <returns>The value: an operand NDArray, an <c>object[]</c> (list), a <c>ValueTuple</c> of any arity (tuple),
            ///     a string, or a Python scalar's C# type.</returns>
            /// <exception cref="NotSupportedException">An unknown spec kind.</exception>
            private object Decode(JsonElement e)
            {
                if (e.ValueKind == JsonValueKind.String)
                    return _ops[_next++];   // "a"
                string kind = e.GetProperty("kind").GetString();
                switch (kind)
                {
                    case "list":
                    {
                        var items = new List<object>();
                        foreach (var it in e.GetProperty("items").EnumerateArray())
                            items.Add(Decode(it));
                        return items.ToArray();
                    }
                    case "tuple":
                    {
                        var items = new List<object>();
                        foreach (var it in e.GetProperty("items").EnumerateArray())
                            items.Add(Decode(it));
                        // A C# tuple literal of that arity (the empty ValueTuple for Python's (); eight or more items nest
                        // in TRest the way the compiler nests them, which ITuple flattens back into one tuple).
                        return PolySequence.MakeTuple(items.ToArray(), items.Count);
                    }
                    case "str":
                        return e.GetProperty("str").GetString();
                    case "none":
                        return null;   // Python's None: C# null (the house map)
                    case "npscalar":
                        // A NumPy scalar with a C# spelling of its own: np.float16 is Half (the house map); the other
                        // NumPy scalar kinds have none (a C# float / double / long IS the Python float / int).
                        if (e.GetProperty("dtype").GetString() != "float16")
                            throw new NotSupportedException($"NumPy scalar of dtype {e.GetProperty("dtype").GetString()} has no C# spelling");
                        return BitConverter.UInt16BitsToHalf(Convert.ToUInt16(e.GetProperty("bits").GetString().Substring(2), 16));
                    default:
                        return PythonScalar(e);
                }
            }
        }

        /// <summary>
        ///     A facade result as the array the corpus records: an NDArray as is; a boxed C# number (a Python /
        ///     NumPy scalar result) as a 0-d array of the dtype its C# type stands for (<c>double</c> → float64,
        ///     <c>float</c> → float32, <see cref="Half"/> → float16, <see cref="Complex"/> → complex128, …).
        /// </summary>
        /// <param name="o">The result.</param>
        /// <returns>The array.</returns>
        /// <exception cref="NotSupportedException">A result type with no dtype.</exception>
        internal static NDArray PolyResultArray(object o) => o switch
        {
            NDArray a => a,
            double v => NDArray.Scalar(v),
            float v => NDArray.Scalar(v),
            Half v => NDArray.Scalar(v),
            Complex v => NDArray.Scalar(v),
            decimal v => NDArray.Scalar(v),
            long v => NDArray.Scalar(v),
            ulong v => NDArray.Scalar(v),
            int v => NDArray.Scalar(v),
            uint v => NDArray.Scalar(v),
            short v => NDArray.Scalar(v),
            ushort v => NDArray.Scalar(v),
            sbyte v => NDArray.Scalar(v),
            byte v => NDArray.Scalar(v),
            char v => NDArray.Scalar(v),
            bool v => NDArray.Scalar(v),
            _ => throw new NotSupportedException($"result type {o?.GetType().Name ?? "null"} has no dtype"),
        };

        /// <summary>
        ///     Runs a polyutils domain function (<c>mapparms</c> / <c>mapdomain</c>) through every C# route that must
        ///     give NumPy's answer, and requires them to agree exactly — the same result dtype and bytes, or the same
        ///     exception type and message:
        ///     <list type="bullet">
        ///         <item>the boxed <c>object</c> overload (<paramref name="viaObject"/>) — which runs the CPython
        ///             machine-number lane for Python lists/tuples of Python numbers;</item>
        ///         <item>the exact general lane (<c>NDPolySeries.MapParmsGeneral</c>: <see cref="BigInteger"/>-backed
        ///             Python numbers through <c>PolyNumber.Binary</c>) — the REFERENCE the lane is proven against;</item>
        ///         <item>for 2-tuple domains, the generic tuple overload a statically typed caller binds
        ///             (<c>mapparms&lt;T0..T3&gt;</c> / <c>mapdomain&lt;T0..T3&gt;</c>, invoked with each tuple rebuilt as
        ///             the element-typed <see cref="ValueTuple{T1,T2}"/> the C# literal would be).</item>
        ///     </list>
        ///     The generic overload's outcome (else the object overload's) is what the corpus then compares with NumPy.
        /// </summary>
        /// <param name="name"><c>mapparms</c> or <c>mapdomain</c>.</param>
        /// <param name="x">mapdomain's point(s); null for mapparms.</param>
        /// <param name="old">The decoded source domain.</param>
        /// <param name="nw">The decoded target domain.</param>
        /// <param name="viaObject">The object-overload call.</param>
        /// <returns>The chosen route's result (a boxed tuple for mapparms).</returns>
        /// <exception cref="InvalidOperationException">Two routes disagree.</exception>
        private static object ViaDomainRoutes(string name, object x, object old, object nw, Func<object> viaObject)
        {
            // The scope reclaims every route's result but the one handed back, and every comparison copy (the
            // scope-audit sweep replays through this registry and would otherwise count those as undisposed
            // intermediates).
            using var scope = NDScope.Open();
            var legs = new List<(string Label, object Result, Exception Error)>
            {
                Run("object overload", viaObject),
                Run("general lane", () => GeneralLane(name, x, old, nw)),
            };

            // mapdomain's generic overloads exist for the x types C# binds them from: NDArray, object[] (a Python list),
            // double, Complex.
            Type xType = x switch
            {
                null => null, NDArray => typeof(NDArray), object[] => typeof(object[]), double => typeof(double),
                Complex => typeof(Complex), _ => typeof(void),
            };
            if (old is ValueTuple<object, object> to && nw is ValueTuple<object, object> tn && xType != typeof(void))
            {
                Type ElementType(object e) => e?.GetType() ?? typeof(object);
                var types = Array.ConvertAll(new[] { to.Item1, to.Item2, tn.Item1, tn.Item2 }, ElementType);
                object typedOld = Activator.CreateInstance(typeof(ValueTuple<,>).MakeGenericType(types[0], types[1]), to.Item1, to.Item2);
                object typedNew = Activator.CreateInstance(typeof(ValueTuple<,>).MakeGenericType(types[2], types[3]), tn.Item1, tn.Item2);
                var generic = Array.Find(typeof(PolyUtilsModule).GetMethods(), m =>
                    m.Name == name && m.IsGenericMethodDefinition &&
                    (xType == null ? m.GetParameters().Length == 2 : m.GetParameters().Length == 3 && m.GetParameters()[0].ParameterType == xType))
                    ?? throw new InvalidOperationException($"no generic {name} overload for x of type {xType?.Name ?? "(none)"}");
                object[] args = xType == null ? new[] { typedOld, typedNew } : new[] { x, typedOld, typedNew };
                legs.Add(Run("generic overload", () =>
                {
                    try { return generic.MakeGenericMethod(types).Invoke(np.polynomial.polyutils, args); }
                    catch (System.Reflection.TargetInvocationException tie)
                    {
                        // Surface the facade's own exception, exactly as a direct call would.
                        System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(tie.InnerException).Throw();
                        throw;
                    }
                }));
            }

            // Every route against the reference (the general lane).
            var reference = legs[1];
            for (int k = 0; k < legs.Count; k++)
            {
                if (k == 1) continue;
                var leg = legs[k];
                if (leg.Error != null || reference.Error != null)
                {
                    if (leg.Error?.GetType() != reference.Error?.GetType() || leg.Error?.Message != reference.Error?.Message)
                        throw new InvalidOperationException($"{name}: the {leg.Label} {Describe(leg)} but the {reference.Label} {Describe(reference)}");
                    continue;
                }
                NDArray[] s1 = Slots(leg.Result), s2 = Slots(reference.Result);
                for (int i = 0; i < s1.Length; i++)
                    if (s1[i].typecode != s2[i].typecode || !s1[i].Shape.dimensions.AsSpan().SequenceEqual(s2[i].Shape.dimensions)
                        || !FuzzCorpus.ResultBytes(s1[i]).AsSpan().SequenceEqual(FuzzCorpus.ResultBytes(s2[i])))
                        throw new InvalidOperationException($"{name}: the {leg.Label} and the {reference.Label} disagree in slot {i}: " +
                                                            $"{s1[i].typecode}{s1[i].Shape} vs {s2[i].typecode}{s2[i].Shape}");
            }

            var chosen = legs[legs.Count - 1].Label == "generic overload" ? legs[legs.Count - 1] : legs[0];
            if (chosen.Error != null)
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(chosen.Error).Throw();
            if (chosen.Result is NDArray nd)
                return scope.Returns(nd);
            if (chosen.Result is System.Runtime.CompilerServices.ITuple tup)
                scope.Returns(tup);   // array components of a 2-D-domain mapparms
            return chosen.Result;

            static (string, object, Exception) Run(string label, Func<object> f)
            {
                try { return (label, f(), null); }
                catch (Exception e) { return (label, null, e); }
            }

            static string Describe((string Label, object Result, Exception Error) leg)
                => leg.Error == null ? "returned" : $"threw {leg.Error.GetType().Name}({leg.Error.Message})";

            // Every slot as the corpus records it: a boxed tuple is two slots.
            static NDArray[] Slots(object r) => r is ValueTuple<object, object> t
                ? new[] { PolyResultArray(t.Item1), PolyResultArray(t.Item2) }
                : new[] { PolyResultArray(r) };
        }

        /// <summary>
        ///     The facade's computation with the exact general lane in place of the machine-number lane: mapparms'
        ///     <c>(off, scl)</c> from <c>NDPolySeries.MapParmsGeneral</c>, and for mapdomain the facade's own point
        ///     classification (an ndarray x; a bool x becomes a 0-d array; any other sequence is converted; a scalar
        ///     is left alone), <c>MapDomainWith</c>, and the facade's result form (the array, or a 0-d result unwrapped
        ///     to its boxed scalar).
        /// </summary>
        /// <param name="name"><c>mapparms</c> or <c>mapdomain</c>.</param>
        /// <param name="x">mapdomain's point(s); null for mapparms.</param>
        /// <param name="old">The source domain.</param>
        /// <param name="nw">The target domain.</param>
        /// <returns>The reference result.</returns>
        private static object GeneralLane(string name, object x, object old, object nw)
        {
            PolyNumber px = default;
            // The facade's classification: an ndarray as is, a bool as NumPy's 0-d array, anything else through FromObject
            // (a Python number / NumPy scalar stays a scalar; a typed C# array converts whole; a Python tuple / list by
            // np.array's nested coercion).
            if (name == "mapdomain")
                px = x switch
                {
                    NDArray xa => PolyNumber.FromArray(xa),
                    bool b => PolyNumber.FromArray(NDArray.Scalar(b)),
                    _ => PolyNumber.FromObject(x),
                };
            var (off, scl) = NDPolySeries.MapParmsGeneral(old, nw);
            if (name == "mapparms")
                return (off.ToObject(), scl.ToObject());
            var r = NDPolySeries.MapDomainWith(px, off, scl);
            if (x is NDArray)
                return r.ToNDArray();
            return r.IsZeroDimArray ? PolyNumber.ScalarOf(r.Array).ToObject() : r.ToObject();
        }

        /// <summary>
        ///     One facet of a module constant (<c>{p}domain/zero/one/x</c>), the case axis of a member that takes no
        ///     argument: <c>value</c> — the constant itself; <c>identity</c> — whether two reads return the SAME
        ///     instance (NumPy's module attribute); <c>writeable</c> / <c>owndata</c> — its flags.
        /// </summary>
        /// <param name="read">Reads the constant from its module.</param>
        /// <param name="a">The case arguments (the <c>facet</c> string).</param>
        /// <returns>The constant, or the facet as a 0-d bool.</returns>
        /// <exception cref="NotSupportedException">An unknown facet.</exception>
        private static NDArray ConstantFacet(Func<NDArray> read, PolyArgs a)
        {
            string facet = (string)a.Get("facet");
            return facet switch
            {
                "value" => read(),
                "identity" => NDArray.Scalar(ReferenceEquals(read(), read())),
                "writeable" => NDArray.Scalar(read().flags.writeable),
                "owndata" => NDArray.Scalar(read().flags.owndata),
                _ => throw new NotSupportedException($"constant facet '{facet}'"),
            };
        }

        /// <summary>
        ///     Dispatches an additive-family / polyutils key to the facade member NumPy's name refers to (the
        ///     SAME member: the module part picks the submodule, so <c>laguerre.lagline</c> is exercised on
        ///     <c>np.polynomial.laguerre</c>).
        /// </summary>
        /// <param name="module">The corpus module (<c>polynomial</c> … <c>hermite_e</c>, <c>polyutils</c>).</param>
        /// <param name="fn">NumPy's function name.</param>
        /// <param name="p">The case's params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The single-array result.</returns>
        /// <exception cref="NotSupportedException">An unknown key.</exception>
        internal static NDArray ApplyPolySeries(string module, string fn, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var a = new PolyArgs(p, ops);
            if (module == "polyutils")
            {
                var u = np.polynomial.polyutils;
                switch (fn)
                {
                    case "trimseq":
                    {
                        var seq = a.Get("seq");
                        if (seq is NDArray nd)
                            return u.trimseq(nd);
                        // A Python list binds the object[] overload and a tuple the general one, as ported code binds them;
                        // NumPy returns the same KIND (the sequence itself or its slice), and the corpus records np.asarray
                        // of it — so the replay compares np.array's coercion of the returned sequence.
                        object r = seq is object[] list ? u.trimseq(list) : u.trimseq(seq);
                        return r as NDArray ?? PolySequence.ToArray(r);
                    }
                    case "trimcoef": return a.Has("tol") ? u.trimcoef(a.Get("c"), a.Get("tol")) : u.trimcoef(a.Get("c"));
                    case "getdomain": return u.getdomain(a.Get("x"));
                    case "mapdomain":
                    {
                        var x = a.Get("x");
                        object old = a.Get("old"), nw = a.Get("new");
                        // An array x takes the NDArray overload and a Python list (object[]) the list overload — the ones
                        // ported code binds; anything else the general one — and tuple domains additionally the generic
                        // overload of that x kind, which must agree with it.
                        object r = ViaDomainRoutes("mapdomain", x, old, nw, () => x switch
                        {
                            NDArray xa => u.mapdomain(xa, old, nw),
                            object[] xl => u.mapdomain(xl, old, nw),
                            _ => u.mapdomain(x, old, nw),
                        });
                        return PolyResultArray(r);
                    }
                    default:
                        throw new NotSupportedException($"polyutils op '{fn}' is not registered in OpRegistry");
                }
            }

            string prefix = module switch
            {
                "polynomial" => "poly",
                "chebyshev" => "cheb",
                "legendre" => "leg",
                "laguerre" => "lag",
                "hermite" => "herm",
                "hermite_e" => "herme",
                _ => throw new NotSupportedException($"polynomial module '{module}' is not registered in OpRegistry"),
            };
            string name = fn.StartsWith(prefix, StringComparison.Ordinal) ? fn.Substring(prefix.Length) : fn;
            // The series algebra (U2, OpRegistry.PolyAlgebra.cs) first; the additive family / calculus below otherwise.
            if (TryApplyPolyAlgebra(module, name, a, out var algebra))
                return algebra;
            switch (module)
            {
                case "polynomial":
                {
                    var m = np.polynomial.polynomial;
                    return name switch
                    {
                        "add" => m.polyadd(a.Get("c1"), a.Get("c2")),
                        "sub" => m.polysub(a.Get("c1"), a.Get("c2")),
                        "trim" => a.Has("tol") ? m.polytrim(a.Get("c"), a.Get("tol")) : m.polytrim(a.Get("c")),
                        "line" => m.polyline(a.Get("off"), a.Get("scl")),
                        "domain" => ConstantFacet(() => m.polydomain, a),
                        "zero" => ConstantFacet(() => m.polyzero, a),
                        "one" => ConstantFacet(() => m.polyone, a),
                        "x" => ConstantFacet(() => m.polyx, a),
                        "der" => a.CalcFacet(m.polyder(a.Get("c"), a.Int("m", 1), a.Opt("scl"), a.Int("axis", 0))),
                        "int" => a.CalcFacet(m.polyint(a.Get("c"), a.Int("m", 1), a.Opt("k"), a.Opt("lbnd"), a.Opt("scl"), a.Int("axis", 0))),
                        _ => throw new NotSupportedException($"polynomial op '{fn}' is not registered in OpRegistry"),
                    };
                }
                case "chebyshev":
                {
                    var m = np.polynomial.chebyshev;
                    return name switch
                    {
                        "add" => m.chebadd(a.Get("c1"), a.Get("c2")),
                        "sub" => m.chebsub(a.Get("c1"), a.Get("c2")),
                        "trim" => a.Has("tol") ? m.chebtrim(a.Get("c"), a.Get("tol")) : m.chebtrim(a.Get("c")),
                        "line" => m.chebline(a.Get("off"), a.Get("scl")),
                        "domain" => ConstantFacet(() => m.chebdomain, a),
                        "zero" => ConstantFacet(() => m.chebzero, a),
                        "one" => ConstantFacet(() => m.chebone, a),
                        "x" => ConstantFacet(() => m.chebx, a),
                        "der" => a.CalcFacet(m.chebder(a.Get("c"), a.Int("m", 1), a.Opt("scl"), a.Int("axis", 0))),
                        "int" => a.CalcFacet(m.chebint(a.Get("c"), a.Int("m", 1), a.Opt("k"), a.Opt("lbnd"), a.Opt("scl"), a.Int("axis", 0))),
                        _ => throw new NotSupportedException($"chebyshev op '{fn}' is not registered in OpRegistry"),
                    };
                }
                case "legendre":
                {
                    var m = np.polynomial.legendre;
                    return name switch
                    {
                        "add" => m.legadd(a.Get("c1"), a.Get("c2")),
                        "sub" => m.legsub(a.Get("c1"), a.Get("c2")),
                        "trim" => a.Has("tol") ? m.legtrim(a.Get("c"), a.Get("tol")) : m.legtrim(a.Get("c")),
                        "line" => m.legline(a.Get("off"), a.Get("scl")),
                        "domain" => ConstantFacet(() => m.legdomain, a),
                        "zero" => ConstantFacet(() => m.legzero, a),
                        "one" => ConstantFacet(() => m.legone, a),
                        "x" => ConstantFacet(() => m.legx, a),
                        "der" => a.CalcFacet(m.legder(a.Get("c"), a.Int("m", 1), a.Opt("scl"), a.Int("axis", 0))),
                        "int" => a.CalcFacet(m.legint(a.Get("c"), a.Int("m", 1), a.Opt("k"), a.Opt("lbnd"), a.Opt("scl"), a.Int("axis", 0))),
                        _ => throw new NotSupportedException($"legendre op '{fn}' is not registered in OpRegistry"),
                    };
                }
                case "laguerre":
                {
                    var m = np.polynomial.laguerre;
                    return name switch
                    {
                        "add" => m.lagadd(a.Get("c1"), a.Get("c2")),
                        "sub" => m.lagsub(a.Get("c1"), a.Get("c2")),
                        "trim" => a.Has("tol") ? m.lagtrim(a.Get("c"), a.Get("tol")) : m.lagtrim(a.Get("c")),
                        "line" => m.lagline(a.Get("off"), a.Get("scl")),
                        "domain" => ConstantFacet(() => m.lagdomain, a),
                        "zero" => ConstantFacet(() => m.lagzero, a),
                        "one" => ConstantFacet(() => m.lagone, a),
                        "x" => ConstantFacet(() => m.lagx, a),
                        "der" => a.CalcFacet(m.lagder(a.Get("c"), a.Int("m", 1), a.Opt("scl"), a.Int("axis", 0))),
                        "int" => a.CalcFacet(m.lagint(a.Get("c"), a.Int("m", 1), a.Opt("k"), a.Opt("lbnd"), a.Opt("scl"), a.Int("axis", 0))),
                        _ => throw new NotSupportedException($"laguerre op '{fn}' is not registered in OpRegistry"),
                    };
                }
                case "hermite":
                {
                    var m = np.polynomial.hermite;
                    return name switch
                    {
                        "add" => m.hermadd(a.Get("c1"), a.Get("c2")),
                        "sub" => m.hermsub(a.Get("c1"), a.Get("c2")),
                        "trim" => a.Has("tol") ? m.hermtrim(a.Get("c"), a.Get("tol")) : m.hermtrim(a.Get("c")),
                        "line" => m.hermline(a.Get("off"), a.Get("scl")),
                        "domain" => ConstantFacet(() => m.hermdomain, a),
                        "zero" => ConstantFacet(() => m.hermzero, a),
                        "one" => ConstantFacet(() => m.hermone, a),
                        "x" => ConstantFacet(() => m.hermx, a),
                        "der" => a.CalcFacet(m.hermder(a.Get("c"), a.Int("m", 1), a.Opt("scl"), a.Int("axis", 0))),
                        "int" => a.CalcFacet(m.hermint(a.Get("c"), a.Int("m", 1), a.Opt("k"), a.Opt("lbnd"), a.Opt("scl"), a.Int("axis", 0))),
                        _ => throw new NotSupportedException($"hermite op '{fn}' is not registered in OpRegistry"),
                    };
                }
                default:   // hermite_e
                {
                    var m = np.polynomial.hermite_e;
                    return name switch
                    {
                        "add" => m.hermeadd(a.Get("c1"), a.Get("c2")),
                        "sub" => m.hermesub(a.Get("c1"), a.Get("c2")),
                        "trim" => a.Has("tol") ? m.hermetrim(a.Get("c"), a.Get("tol")) : m.hermetrim(a.Get("c")),
                        "line" => m.hermeline(a.Get("off"), a.Get("scl")),
                        "domain" => ConstantFacet(() => m.hermedomain, a),
                        "zero" => ConstantFacet(() => m.hermezero, a),
                        "one" => ConstantFacet(() => m.hermeone, a),
                        "x" => ConstantFacet(() => m.hermex, a),
                        "der" => a.CalcFacet(m.hermeder(a.Get("c"), a.Int("m", 1), a.Opt("scl"), a.Int("axis", 0))),
                        "int" => a.CalcFacet(m.hermeint(a.Get("c"), a.Int("m", 1), a.Opt("k"), a.Opt("lbnd"), a.Opt("scl"), a.Int("axis", 0))),
                        _ => throw new NotSupportedException($"hermite_e op '{fn}' is not registered in OpRegistry"),
                    };
                }
            }
        }

        /// <summary>
        ///     The tuple-valued polynomial functions: <c>as_series</c> (one slot per series — arity asserted),
        ///     <c>mapparms</c> (<c>(off, scl)</c>, each a Python / NumPy scalar recorded as a 0-d array, or an array),
        ///     and the six <c>{p}div</c> (<c>(quo, rem)</c>, dispatched by <see cref="ApplyPolyAlgebraTuple"/>).
        /// </summary>
        /// <param name="op">The key.</param>
        /// <param name="p">The params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The slots.</returns>
        /// <exception cref="NotSupportedException">An unknown key.</exception>
        internal static NDArray[] ApplyPolySeriesTuple(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var a = new PolyArgs(p, ops);
            var u = np.polynomial.polyutils;
            switch (op)
            {
                case "polyutils.as_series":
                {
                    var alist = a.Get("alist");
                    return u.as_series(alist, a.Flag("trim", true));
                }
                case "polyutils.mapparms":
                {
                    object old = a.Get("old"), nw = a.Get("new");
                    var (off, scl) = ((object, object))ViaDomainRoutes("mapparms", null, old, nw, () => u.mapparms(old, nw));
                    return new[] { PolyResultArray(off), PolyResultArray(scl) };
                }
                default:
                    // {p}div's (quo, rem) — the series algebra's one tuple (OpRegistry.PolyAlgebra.cs).
                    if (!op.StartsWith("polyutils.", StringComparison.Ordinal))
                        return ApplyPolyAlgebraTuple(op, a);
                    throw new NotSupportedException($"polynomial tuple op '{op}' is not registered in OpRegistry");
            }
        }
    }
}
