using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     <c>numpy.polynomial</c>'s six series classes (plan U10, U11's class half): <see cref="NDPolynomial"/>,
    ///     <see cref="NDChebyshev"/>, <see cref="NDLegendre"/>, <see cref="NDLaguerre"/>, <see cref="NDHermite"/> and
    ///     <see cref="NDHermiteE"/>. Pairs 1:1 with <c>gen_oracle.py</c>'s <c>gen_polyclass</c> (the T-polyclass header):
    ///     <list type="bullet">
    ///         <item>the key is <c>&lt;module&gt;.&lt;Class&gt;</c> (the constructor) or <c>&lt;module&gt;.&lt;Class&gt;.&lt;member&gt;</c>
    ///             — the member NumPy dispatched to (<c>__radd__</c> for <c>x + p</c>);</item>
    ///         <item><c>params.args</c> lists the call's arguments as <c>[name, spec]</c> pairs in the order their operands
    ///             were described, decoded in that order; a <c>"series"</c> spec is built through the public constructor and
    ///             property setters (<see cref="PolyClassNew"/>);</item>
    ///         <item><c>params.array0d</c> names the 0-d operands that were explicit <c>np.array(x)</c> (marked here as
    ///             <c>np.array</c> marks them); an unmarked 0-d operand is a NumPy scalar;</item>
    ///         <item><c>params.sym</c> is the operator of a binary / comparison case, applied with C#'s own operators through
    ///             <c>dynamic</c>, so the overload the compiler would bind for those runtime types is the one exercised.</item>
    ///     </list>
    ///     A series result is the tuple <c>[coef, domain, window, tag]</c> with <c>tag = [class id, code points of the
    ///     symbol]</c> (int64); every series this registry builds or receives is disposed before the call returns, so the leak
    ///     gate sees the class's own traffic only.
    /// </summary>
    public static partial class OpRegistry
    {
        /// <summary>The six class names in the generator's id order (the tag's class id is the index).</summary>
        private static readonly string[] PolyClassNames = { "Polynomial", "Chebyshev", "Legendre", "Laguerre", "Hermite", "HermiteE" };

        /// <summary>The module each class lives in (the key's first part).</summary>
        private static readonly Dictionary<string, string> PolyClassModules = new(StringComparer.Ordinal)
        {
            ["Polynomial"] = "polynomial", ["Chebyshev"] = "chebyshev", ["Legendre"] = "legendre",
            ["Laguerre"] = "laguerre", ["Hermite"] = "hermite", ["HermiteE"] = "hermite_e",
        };

        /// <summary>The series a polyclass case built while decoding (disposed when the case returns).</summary>
        [ThreadStatic] private static List<object> s_polyClassOwned;

        /// <summary>Whether <paramref name="op"/> is a series-class key (<c>chebyshev.Chebyshev</c>, <c>chebyshev.Chebyshev.copy</c>).</summary>
        /// <param name="op">The corpus key.</param>
        /// <returns>True for a class key of the T-polyclass tier.</returns>
        internal static bool IsPolyClassOp(string op)
        {
            int dot = op.IndexOf('.');
            if (dot < 0)
                return false;
            int dot2 = op.IndexOf('.', dot + 1);
            string cls = dot2 < 0 ? op.Substring(dot + 1) : op.Substring(dot + 1, dot2 - dot - 1);
            return PolyClassModules.TryGetValue(cls, out var module) && string.CompareOrdinal(op, 0, module, 0, Math.Max(dot, module.Length)) == 0
                   && dot == module.Length;
        }

        /// <summary>A polyclass case whose result is one array (or a number / bool as a 0-d array).</summary>
        /// <param name="op">The key.</param><param name="p">The params.</param><param name="ops">The operands.</param>
        /// <returns>The result array (the registry's own: the harness disposes it).</returns>
        internal static NDArray PolyClassArray(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
            => (NDArray)PolyClassRun(op, p, ops, "array");

        /// <summary>A polyclass case whose result is a tuple (a series, divmod, linspace, mapparms, a state, a full fit).</summary>
        /// <param name="op">The key.</param><param name="p">The params.</param><param name="ops">The operands.</param>
        /// <returns>The slots (the registry's own arrays).</returns>
        internal static NDArray[] PolyClassTuple(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
            => (NDArray[])PolyClassRun(op, p, ops, "tuple");

        /// <summary>A polyclass case whose result is text (the printers, the str attributes).</summary>
        /// <param name="op">The key.</param><param name="p">The params.</param><param name="ops">The operands.</param>
        /// <returns>The text.</returns>
        internal static string PolyClassText(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
            => (string)PolyClassRun(op, p, ops, "text");

        /// <summary>
        ///     Runs one case: marks the explicit 0-d operands, decodes the arguments in order, applies the print style and
        ///     options, calls the member, converts the result to the requested kind, and disposes every series involved.
        /// </summary>
        /// <param name="op">The key.</param><param name="p">The params.</param><param name="ops">The operands.</param>
        /// <param name="kind"><c>array</c>, <c>tuple</c> or <c>text</c>.</param>
        /// <returns>The converted result.</returns>
        /// <exception cref="NotSupportedException">An unknown member.</exception>
        private static object PolyClassRun(string op, IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops, string kind)
        {
            if (p.TryGetValue("array0d", out var explicitArrays))
                foreach (JsonElement position in explicitArrays.EnumerateArray())
                    ops[position.GetInt32()].MarkExplicitZeroDimArray();

            var previous = s_polyClassOwned;
            var owned = s_polyClassOwned = new List<object>();
            bool styleBefore = NDPolyPrinting.UseUnicode;
            IDisposable printScope = p.TryGetValue("opts", out var o) ? PolyClassPrintOptions(o) : null;
            try
            {
                if (p.TryGetValue("style", out var st))
                    np.polynomial.set_default_printstyle(st.GetString());
                var a = new PolyArgs(p, ops);
                var args = new Dictionary<string, object>(StringComparer.Ordinal);
                var order = new List<string>();
                foreach (JsonElement pair in p["args"].EnumerateArray())
                {
                    string name = pair[0].GetString();
                    args[name] = a.Spec(pair[1]);
                    order.Add(name);
                }

                int dot = op.IndexOf('.');
                int dot2 = op.IndexOf('.', dot + 1);
                string cls = dot2 < 0 ? op.Substring(dot + 1) : op.Substring(dot + 1, dot2 - dot - 1);
                string member = dot2 < 0 ? null : op.Substring(dot2 + 1);

                object r = PolyClassDispatch(cls, member, args, p);
                // Each arm cast to object: a switch expression takes its arms' best common type, and NDArray[] and string
                // both convert implicitly to NDArray (np.array of them), which would turn the slots into one array.
                return kind switch
                {
                    "tuple" => (object)PolyClassSlots(r, owned),
                    "text" => (object)(r as string ?? (r is null ? "None" : r.ToString())),
                    _ => (object)PolyClassResultArray(r, owned),
                };
            }
            finally
            {
                printScope?.Dispose();
                np.polynomial.set_default_printstyle(styleBefore ? "unicode" : "ascii");
                s_polyClassOwned = previous;
                // Every series and state array the case built or received, each once (a series result can be an operand:
                // `+p` returns p itself). Reference identity, not Equals: a series is unhashable and its == is NumPy's.
                for (int i = 0; i < owned.Count; i++)
                {
                    bool seen = false;
                    for (int j = 0; j < i && !seen; j++)
                        seen = ReferenceEquals(owned[i], owned[j]);
                    if (seen)
                        continue;
                    if (owned[i] is IDisposable d)
                        d.Dispose();
                }
            }
        }

        /// <summary>The case's print options (<c>params.opts</c>), applied as <c>np.printoptions</c> for the call.</summary>
        /// <param name="o">The options object.</param>
        /// <returns>The scope restoring the previous options.</returns>
        private static IDisposable PolyClassPrintOptions(JsonElement o)
            => np.printoptions(
                precision: o.TryGetProperty("precision", out var prec) ? prec.GetInt32() : null,
                threshold: o.TryGetProperty("threshold", out var thr) ? thr.GetInt32() : null,
                linewidth: o.TryGetProperty("linewidth", out var lw) ? lw.GetInt32() : null,
                suppress: o.TryGetProperty("suppress", out var sup) ? sup.GetBoolean() : null,
                nanstr: o.TryGetProperty("nanstr", out var ns) ? ns.GetString() : null,
                infstr: o.TryGetProperty("infstr", out var inf) ? inf.GetString() : null,
                sign: o.TryGetProperty("sign", out var sg) ? sg.GetString()[0] : null,
                floatmode: o.TryGetProperty("floatmode", out var fm) ? fm.GetString() : null);

        /// <summary>
        ///     Builds a series argument through the public surface (the decoder's <c>"series"</c> spec): the class's
        ///     constructor, then each assigned attribute through its setter, in order.
        /// </summary>
        /// <param name="cls">The class name.</param>
        /// <param name="coef">The coefficients.</param><param name="domain">The domain, or null.</param>
        /// <param name="window">The window, or null.</param><param name="symbol">The symbol.</param>
        /// <returns>The series (registered for disposal).</returns>
        /// <exception cref="NotSupportedException">An unknown class.</exception>
        internal static NDPolyBase PolyClassNew(string cls, object coef, object domain, object window, string symbol)
        {
            NDPolyBase s = cls switch
            {
                "Polynomial" => new NDPolynomial(coef, domain, window, symbol),
                "Chebyshev" => new NDChebyshev(coef, domain, window, symbol),
                "Legendre" => new NDLegendre(coef, domain, window, symbol),
                "Laguerre" => new NDLaguerre(coef, domain, window, symbol),
                "Hermite" => new NDHermite(coef, domain, window, symbol),
                "HermiteE" => new NDHermiteE(coef, domain, window, symbol),
                _ => throw new NotSupportedException($"polynomial class '{cls}' is not registered in OpRegistry"),
            };
            s_polyClassOwned?.Add(s);
            return s;
        }

        /// <summary>
        ///     The constructor case (the key <c>&lt;module&gt;.&lt;Class&gt;</c>, NumPy's
        ///     <c>numpy.polynomial.chebyshev.Chebyshev(...)</c>) through the module facade's factory
        ///     (<c>np.polynomial.chebyshev.Chebyshev(...)</c>, what a ported <c>from numpy.polynomial.chebyshev import
        ///     Chebyshev</c> call reaches), so the tier gates the factory itself and not only <c>new NDChebyshev(...)</c>.
        /// </summary>
        /// <param name="cls">The class name.</param>
        /// <param name="coef">The coefficients.</param><param name="domain">The domain, or null.</param>
        /// <param name="window">The window, or null.</param><param name="symbol">The symbol.</param>
        /// <returns>The series (registered for disposal).</returns>
        /// <exception cref="NotSupportedException">An unknown class.</exception>
        private static NDPolyBase PolyClassFactory(string cls, object coef, object domain, object window, string symbol)
        {
            NDPolyBase s = cls switch
            {
                "Polynomial" => np.polynomial.polynomial.Polynomial(coef, domain, window, symbol),
                "Chebyshev" => np.polynomial.chebyshev.Chebyshev(coef, domain, window, symbol),
                "Legendre" => np.polynomial.legendre.Legendre(coef, domain, window, symbol),
                "Laguerre" => np.polynomial.laguerre.Laguerre(coef, domain, window, symbol),
                "Hermite" => np.polynomial.hermite.Hermite(coef, domain, window, symbol),
                "HermiteE" => np.polynomial.hermite_e.HermiteE(coef, domain, window, symbol),
                _ => throw new NotSupportedException($"polynomial class '{cls}' is not registered in OpRegistry"),
            };
            s_polyClassOwned?.Add(s);
            return s;
        }

        /// <summary>Assigns one attribute of a series argument (the <c>"assign"</c> list of its spec), through the setter.</summary>
        /// <param name="s">The series.</param><param name="attr">The attribute.</param><param name="value">The value.</param>
        /// <exception cref="NotSupportedException">An attribute the tier does not assign.</exception>
        internal static void PolyClassAssign(NDPolyBase s, string attr, object value)
        {
            switch (attr)
            {
                case "coef": s.coef = (NDArray)value; break;
                case "domain": s.domain = (NDArray)value; break;
                case "window": s.window = (NDArray)value; break;
                case "maxpower": s.maxpower = value; break;
                default: throw new NotSupportedException($"series attribute '{attr}' is not assigned by the tier");
            }
        }

        /// <summary>The CLR type of a class name (convert's <c>kind</c>).</summary>
        /// <param name="cls">The name.</param>
        /// <returns>The type.</returns>
        /// <exception cref="NotSupportedException">An unknown class.</exception>
        private static Type PolyClassType(string cls) => cls switch
        {
            "Polynomial" => typeof(NDPolynomial),
            "Chebyshev" => typeof(NDChebyshev),
            "Legendre" => typeof(NDLegendre),
            "Laguerre" => typeof(NDLaguerre),
            "Hermite" => typeof(NDHermite),
            "HermiteE" => typeof(NDHermiteE),
            _ => throw new NotSupportedException($"polynomial class '{cls}' is not registered in OpRegistry"),
        };

        /// <summary>Calls the member a case names.</summary>
        /// <param name="cls">The class.</param><param name="member">The member, or null for the constructor.</param>
        /// <param name="args">The decoded arguments by name.</param><param name="p">The params (sym, fmt, kind, func).</param>
        /// <returns>NumSharp's raw result.</returns>
        /// <exception cref="NotSupportedException">An unknown member.</exception>
        /// <exception cref="InvalidOperationException">Two C# spellings of one printer disagree (<c>__str__()</c> against
        ///     <c>ToString()</c>, <c>__format__</c> against <c>string.Format</c>): a NumSharp defect the case reports.</exception>
        private static object PolyClassDispatch(string cls, string member, Dictionary<string, object> args,
            IReadOnlyDictionary<string, JsonElement> p)
        {
            object Arg(string name) => args[name];
            object Opt(string name) => args.TryGetValue(name, out var v) ? v : null;
            bool Has(string name) => args.ContainsKey(name);
            int IntArg(string name, int fallback) => args.TryGetValue(name, out var v) ? Convert.ToInt32(v) : fallback;

            if (member is null)
                return PolyClassFactory(cls, Arg("coef"), Opt("domain"), Opt("window"), Has("symbol") ? (string)Arg("symbol") : "x");

            if (p.TryGetValue("sym", out var symEl))
                return PolyClassOperator(symEl.GetString(), member, Arg("a"), Arg("b"));

            // Object identity (NumPy's `is`): +p is p; a getter returns the stored array; a class default is one shared
            // object; a derived or copied series owns copies.
            if (p.TryGetValue("facet", out var facetEl))
                return PolyClassIdentity(facetEl.GetString(), member, args);

            switch (member)
            {
                case "__neg__": return -(dynamic)Arg("a");
                case "__pos__": return +(dynamic)Arg("a");
                case "identity": return PolyClassStatic(cls, "identity", args);
                case "basis": return PolyClassStatic(cls, "basis", args);
                case "fromroots": return PolyClassStatic(cls, "fromroots", args);
                case "fit": return PolyClassStatic(cls, "fit", args);
                case "cast": return PolyClassStatic(cls, "cast", args);
                case "interpolate":
                    return NDChebyshev.interpolate(ChebInterpolateFunc(p["func"].GetString()), Arg("deg"), Opt("domain"), Opt("args"));
            }

            var self = (NDPolyBase)Arg("self");
            switch (member)
            {
                case "copy": return self.copy();
                case "degree": return self.degree();
                case "mapparms": return self.mapparms();
                case "deriv": return Has("m") ? self.deriv(IntArg("m", 1)) : self.deriv();
                case "integ":
                    return self.integ(IntArg("m", 1), Opt("k"), Opt("lbnd"));
                case "roots": return self.roots();
                case "linspace":
                    return Has("n") ? self.linspace(Convert.ToInt64(Arg("n")), Opt("domain")) : self.linspace();
                case "trim": return Has("tol") ? self.trim(Arg("tol")) : self.trim();
                case "cutdeg": return self.cutdeg(Arg("deg"));
                case "truncate": return self.truncate(Arg("size"));
                case "convert":
                {
                    Type kind = p.TryGetValue("kind", out var k) ? PolyClassType(k.GetString()) : null;
                    return self.convert(Opt("domain"), kind, Opt("window"));
                }
                case "__call__":
                {
                    object x = Arg("x");
                    // A series point binds the generic __call__<TSeries> (a series result), any other point __call__(object), as
                    // the compiler binds them for these runtime types.
                    return x is NDPolyBase ? ((dynamic)self).__call__((dynamic)x) : self.__call__(x);
                }
                case "__len__": return self.__len__();
                case "__iter__": return PolyClassIterate(self);
                case "coef": return self.coef.copy();
                case "domain": return self.domain.copy();
                case "window": return self.window.copy();
                case "symbol": return self.symbol;
                case "basis_name": return self.basis_name;
                case "maxpower": return self.maxpower;
                case "has_samecoef": return self.has_samecoef((NDPolyBase)Opt("other"));
                case "has_samedomain": return self.has_samedomain((NDPolyBase)Opt("other"));
                case "has_samewindow": return self.has_samewindow((NDPolyBase)Opt("other"));
                case "has_sametype": return self.has_sametype(Opt("other"));
                case "__getstate__": return PolyClassState(self.__getstate__());
                case "__setstate__":
                {
                    var target = (NDPolyBase)Arg("target");
                    var state = PolyClassState(self.__getstate__());
                    target.__setstate__(state);
                    return target;
                }
                case "__str__":
                {
                    string viaDunder = self.__str__();
                    string viaToString = self.ToString();
                    if (!string.Equals(viaDunder, viaToString, StringComparison.Ordinal))
                        throw new InvalidOperationException($"{cls}.__str__() and ToString() disagree: '{viaDunder}' vs '{viaToString}'");
                    return viaDunder;
                }
                case "__format__":
                {
                    string fmt = p["fmt"].GetString();
                    string viaDunder = self.__format__(fmt);
                    // C#'s composite formatting reaches the same printer through IFormattable.
                    string viaFormattable = string.Format(System.Globalization.CultureInfo.InvariantCulture,
                        fmt.Length == 0 ? "{0}" : "{0:" + fmt + "}", self);
                    if (!string.Equals(viaDunder, viaFormattable, StringComparison.Ordinal))
                        throw new InvalidOperationException($"{cls}.__format__ and string.Format disagree: '{viaDunder}' vs '{viaFormattable}'");
                    return viaDunder;
                }
                case "__repr__": return self.__repr__();
                case "_repr_latex_": return self._repr_latex_();
                default:
                    throw new NotSupportedException($"polynomial class member '{cls}.{member}' is not registered in OpRegistry");
            }
        }

        /// <summary>
        ///     An object-identity case (NumPy's <c>is</c>, recorded as a bool): <c>identity</c> — <c>+p is p</c>, or a getter
        ///     read twice; <c>shared</c> — the attribute of two series (one class default when neither has its own);
        ///     <c>derived</c> — <c>(p + 1).attr is p.attr</c>; <c>copied</c> — <c>p.copy().attr is p.attr</c>.
        /// </summary>
        /// <param name="facet">The facet.</param><param name="member">The member (<c>__pos__</c> or an attribute).</param>
        /// <param name="args">The decoded arguments.</param>
        /// <returns>The bool.</returns>
        /// <exception cref="NotSupportedException">An unknown facet or attribute.</exception>
        private static object PolyClassIdentity(string facet, string member, Dictionary<string, object> args)
        {
            if (member == "__pos__")
            {
                var a = (NDPolyBase)args["a"];
                return ReferenceEquals(+(dynamic)a, a);
            }
            var self = (NDPolyBase)args["self"];
            NDArray Attr(NDPolyBase s) => member switch
            {
                "coef" => s.coef,
                "domain" => s.domain,
                "window" => s.window,
                _ => throw new NotSupportedException($"identity facet of series attribute '{member}'"),
            };
            switch (facet)
            {
                case "identity":
                    return ReferenceEquals(Attr(self), Attr(self));
                case "shared":
                    return ReferenceEquals(Attr(self), Attr((NDPolyBase)args["other"]));
                case "derived":
                {
                    var derived = (NDPolyBase)((dynamic)self + 1L);
                    s_polyClassOwned?.Add(derived);
                    return ReferenceEquals(Attr(derived), Attr(self));
                }
                case "copied":
                {
                    var copy = self.copy();
                    s_polyClassOwned?.Add(copy);
                    return ReferenceEquals(Attr(copy), Attr(self));
                }
                default:
                    throw new NotSupportedException($"series identity facet '{facet}'");
            }
        }

        /// <summary>
        ///     One binary / comparison case: <c>params.sym</c> applied with C#'s operator (or the protocol method C# spells
        ///     by name: <c>//</c>, <c>divmod</c>, <c>**</c>, <c>__eq__</c>, <c>__ne__</c>), the series on the side the key
        ///     names (<c>__radd__</c>: the right one).
        /// </summary>
        /// <param name="sym">The operator.</param><param name="member">The key's member.</param>
        /// <param name="a">The left operand.</param><param name="b">The right operand.</param>
        /// <returns>The result.</returns>
        /// <exception cref="NotSupportedException">An unknown operator.</exception>
        private static object PolyClassOperator(string sym, string member, object a, object b)
        {
            bool reflected = member.StartsWith("__r", StringComparison.Ordinal);
            // Both operands are dynamic, so the runtime binder resolves the overload from their RUNTIME types — the
            // overload the compiler binds for a caller holding those types (a static object operand would bind as object).
            // A null right operand keeps the static type object: `p + (object)null` is the spelling a C# caller needs (a
            // bare null literal is ambiguous between the reference-typed overloads). The left operand is never null.
            dynamic l = a;
            if (b is null)
            {
                object none = null;
                switch (sym)
                {
                    case "+": return l + none;
                    case "-": return l - none;
                    case "*": return l * none;
                    case "/": return l / none;
                    case "%": return l % none;
                    case "//": return l.__floordiv__(none);
                    case "divmod": return l.__divmod__(none);
                    case "**": return l.__pow__(none);
                    case "==": return l == none;
                    case "!=": return l != none;
                }
            }
            dynamic r = b;
            // `p == arr` is ambiguous in C# between the series' ==(NDPolyBase, object) and NDArray's ==(object, NDArray): a
            // C# caller writes `p == (object)arr` (or p.__eq__(arr)), which is what this replay spells.
            if (b is NDArray && sym is "==" or "!=")
            {
                object boxed = b;
                return sym == "==" ? l == boxed : l != boxed;
            }
            switch (sym)
            {
                case "+": return l + r;
                case "-": return l - r;
                case "*": return l * r;
                case "/": return l / r;
                case "%": return l % r;
                case "//": return reflected ? r.__rfloordiv__(a) : l.__floordiv__(b);
                case "divmod": return reflected ? r.__rdivmod__(a) : l.__divmod__(b);
                case "**": return l.__pow__(b);
                case "==": return l == r;
                case "!=": return l != r;
                case "__eq__": return ((NDPolyBase)a).__eq__(b);
                case "__ne__": return ((NDPolyBase)a).__ne__(b);
                default: throw new NotSupportedException($"series operator '{sym}' is not registered in OpRegistry");
            }
        }

        /// <summary>The class methods, through each concrete class's own static member.</summary>
        /// <param name="cls">The class.</param><param name="name">The method.</param><param name="args">The arguments.</param>
        /// <returns>The result (a series, or the full fit's slots).</returns>
        /// <exception cref="NotSupportedException">An unknown class or method.</exception>
        private static object PolyClassStatic(string cls, string name, Dictionary<string, object> args)
        {
            object Opt(string n) => args.TryGetValue(n, out var v) ? v : null;
            string Sym() => args.TryGetValue("symbol", out var v) ? (string)v : "x";
            // fromroots' domain: omitted is NumPy's default [] (the [Optional] parameter's Missing), an explicit null None.
            object RootsDomain() => args.TryGetValue("domain", out var v) ? v : System.Reflection.Missing.Value;
            bool full = args.TryGetValue("full", out var f) && (bool)f;
            NDPolyBase r;
            switch (cls)
            {
                case "Polynomial":
                    switch (name)
                    {
                        case "identity": return NDPolynomial.identity(Opt("domain"), Opt("window"), Sym());
                        case "basis": return NDPolynomial.basis(args["deg"], Opt("domain"), Opt("window"), Sym());
                        case "fromroots": return NDPolynomial.fromroots(args["roots"], RootsDomain(), Opt("window"), Sym());
                        case "cast": return NDPolynomial.cast((NDPolyBase)args["series"], Opt("domain"), Opt("window"));
                        case "fit":
                        {
                            var s = NDPolynomial.fit(args["x"], args["y"], args["deg"], Opt("domain"), Opt("rcond"), full, Opt("w"), Opt("window"), Sym());
                            if (!full) return s;
                            var (series, st) = s;
                            return PolyClassFitSlots(series, st);
                        }
                    }
                    break;
                case "Chebyshev":
                    switch (name)
                    {
                        case "identity": return NDChebyshev.identity(Opt("domain"), Opt("window"), Sym());
                        case "basis": return NDChebyshev.basis(args["deg"], Opt("domain"), Opt("window"), Sym());
                        case "fromroots": return NDChebyshev.fromroots(args["roots"], RootsDomain(), Opt("window"), Sym());
                        case "cast": return NDChebyshev.cast((NDPolyBase)args["series"], Opt("domain"), Opt("window"));
                        case "fit":
                        {
                            var s = NDChebyshev.fit(args["x"], args["y"], args["deg"], Opt("domain"), Opt("rcond"), full, Opt("w"), Opt("window"), Sym());
                            if (!full) return s;
                            var (series, st) = s;
                            return PolyClassFitSlots(series, st);
                        }
                    }
                    break;
                case "Legendre":
                    switch (name)
                    {
                        case "identity": return NDLegendre.identity(Opt("domain"), Opt("window"), Sym());
                        case "basis": return NDLegendre.basis(args["deg"], Opt("domain"), Opt("window"), Sym());
                        case "fromroots": return NDLegendre.fromroots(args["roots"], RootsDomain(), Opt("window"), Sym());
                        case "cast": return NDLegendre.cast((NDPolyBase)args["series"], Opt("domain"), Opt("window"));
                        case "fit":
                        {
                            var s = NDLegendre.fit(args["x"], args["y"], args["deg"], Opt("domain"), Opt("rcond"), full, Opt("w"), Opt("window"), Sym());
                            if (!full) return s;
                            var (series, st) = s;
                            return PolyClassFitSlots(series, st);
                        }
                    }
                    break;
                case "Laguerre":
                    switch (name)
                    {
                        case "identity": return NDLaguerre.identity(Opt("domain"), Opt("window"), Sym());
                        case "basis": return NDLaguerre.basis(args["deg"], Opt("domain"), Opt("window"), Sym());
                        case "fromroots": return NDLaguerre.fromroots(args["roots"], RootsDomain(), Opt("window"), Sym());
                        case "cast": return NDLaguerre.cast((NDPolyBase)args["series"], Opt("domain"), Opt("window"));
                        case "fit":
                        {
                            var s = NDLaguerre.fit(args["x"], args["y"], args["deg"], Opt("domain"), Opt("rcond"), full, Opt("w"), Opt("window"), Sym());
                            if (!full) return s;
                            var (series, st) = s;
                            return PolyClassFitSlots(series, st);
                        }
                    }
                    break;
                case "Hermite":
                    switch (name)
                    {
                        case "identity": return NDHermite.identity(Opt("domain"), Opt("window"), Sym());
                        case "basis": return NDHermite.basis(args["deg"], Opt("domain"), Opt("window"), Sym());
                        case "fromroots": return NDHermite.fromroots(args["roots"], RootsDomain(), Opt("window"), Sym());
                        case "cast": return NDHermite.cast((NDPolyBase)args["series"], Opt("domain"), Opt("window"));
                        case "fit":
                        {
                            var s = NDHermite.fit(args["x"], args["y"], args["deg"], Opt("domain"), Opt("rcond"), full, Opt("w"), Opt("window"), Sym());
                            if (!full) return s;
                            var (series, st) = s;
                            return PolyClassFitSlots(series, st);
                        }
                    }
                    break;
                case "HermiteE":
                    switch (name)
                    {
                        case "identity": return NDHermiteE.identity(Opt("domain"), Opt("window"), Sym());
                        case "basis": return NDHermiteE.basis(args["deg"], Opt("domain"), Opt("window"), Sym());
                        case "fromroots": return NDHermiteE.fromroots(args["roots"], RootsDomain(), Opt("window"), Sym());
                        case "cast": return NDHermiteE.cast((NDPolyBase)args["series"], Opt("domain"), Opt("window"));
                        case "fit":
                        {
                            var s = NDHermiteE.fit(args["x"], args["y"], args["deg"], Opt("domain"), Opt("rcond"), full, Opt("w"), Opt("window"), Sym());
                            if (!full) return s;
                            var (series, st) = s;
                            return PolyClassFitSlots(series, st);
                        }
                    }
                    break;
            }
            throw new NotSupportedException($"polynomial class method '{cls}.{name}' is not registered in OpRegistry");
        }

        /// <summary>A full fit's eight slots: the series' four, then lstsq's four statistics (the caller's arrays).</summary>
        /// <param name="series">The fitted series.</param>
        /// <param name="st">The statistics.</param>
        /// <returns>The slots.</returns>
        private static NDArray[] PolyClassFitSlots(NDPolyBase series,
            (NDArray residuals, NDArray rank, NDArray singular_values, NDArray rcond) st)
        {
            s_polyClassOwned?.Add(series);
            var slots = new List<NDArray>(PolyClassSeriesSlots(series)) { st.residuals, st.rank, st.singular_values, st.rcond };
            return slots.ToArray();
        }

        /// <summary>
        ///     A state dictionary as the case records it (and keeps for disposal): the dictionary itself, its arrays registered
        ///     with the case's owned objects — they are the caller's copies.
        /// </summary>
        /// <param name="state">The state from <c>__getstate__</c>.</param>
        /// <returns><paramref name="state"/>.</returns>
        private static OrderedDictionary PolyClassState(OrderedDictionary state)
        {
            foreach (DictionaryEntry e in state)
                if (e.Value is NDArray nd)
                    s_polyClassOwned?.Add(nd);
            return state;
        }

        /// <summary>
        ///     <c>np.array(list(p))</c>: the series iterated with <c>foreach</c> (NumPy's <c>iter(self.coef)</c>), rebuilt as an
        ///     array of the coefficients' dtype — boxed scalars for a 1-D coefficient array, row arrays (stacked) for an
        ///     assigned 2-D one.
        /// </summary>
        /// <param name="self">The series.</param>
        /// <returns>The array.</returns>
        private static NDArray PolyClassIterate(NDPolyBase self)
        {
            var items = new List<object>();
            foreach (object v in self)
                items.Add(v);
            if (items.Count > 0 && items[0] is NDArray)
            {
                var rows = new NDArray[items.Count];
                for (int i = 0; i < rows.Length; i++)
                    rows[i] = (NDArray)items[i];
                return np.stack(rows);
            }
            Type t = self.coef.dtype.type;
            var typed = Array.CreateInstance(t, items.Count);
            for (int i = 0; i < items.Count; i++)
                typed.SetValue(items[i], i);
            return np.array(typed);
        }

        /// <summary>The four slots of a series: copies of its arrays (the harness disposes slots) and the tag.</summary>
        /// <param name="s">The series.</param>
        /// <returns>[coef, domain, window, tag].</returns>
        private static NDArray[] PolyClassSeriesSlots(NDPolyBase s)
        {
            string name = s.GetType().Name.Substring(2);   // NDChebyshev -> Chebyshev
            var tag = new List<long> { Array.IndexOf(PolyClassNames, name) };
            // Python's ord() per CODE POINT: a symbol outside the BMP is one rune, two UTF-16 units.
            foreach (Rune r in s.symbol.EnumerateRunes())
                tag.Add(r.Value);
            return new[] { s.coef.copy(), s.domain.copy(), s.window.copy(), np.array(tag.ToArray()) };
        }

        /// <summary>A tuple-kind result as its slots (registering every series for disposal).</summary>
        /// <param name="r">The result.</param>
        /// <param name="owned">The case's owned objects.</param>
        /// <returns>The slots.</returns>
        /// <exception cref="NotSupportedException">A result the tier does not record as a tuple.</exception>
        private static NDArray[] PolyClassSlots(object r, List<object> owned)
        {
            switch (r)
            {
                case NDPolyBase s:
                    owned.Add(s);
                    return PolyClassSeriesSlots(s);
                case NDArray[] slots:
                    return slots;
                case OrderedDictionary state:
                {
                    var slots = new List<NDArray>();
                    foreach (DictionaryEntry e in state)
                    {
                        slots.Add(PolyClassCodePoints((string)e.Key));
                        slots.Add(e.Value switch
                        {
                            string text => PolyClassCodePoints(text),
                            NDArray nd => nd.copy(),
                            var v => PolyResultArray(v),
                        });
                    }
                    return slots.ToArray();
                }
                case ITuple t when t.Length == 2 && t[0] is NDPolyBase q && t[1] is NDPolyBase rem:
                {
                    owned.Add(q);
                    owned.Add(rem);
                    var slots = new List<NDArray>(PolyClassSeriesSlots(q));
                    slots.AddRange(PolyClassSeriesSlots(rem));
                    return slots.ToArray();
                }
                case ITuple t:
                {
                    var slots = new NDArray[t.Length];
                    for (int i = 0; i < slots.Length; i++)
                        slots[i] = PolyResultArray(t[i]);
                    return slots;
                }
                default:
                    throw new NotSupportedException($"polyclass result of type {r?.GetType().Name ?? "null"} is not a tuple");
            }
        }

        /// <summary>A str as the int64 array of its code points (a symbol, a state key).</summary>
        /// <param name="text">The text.</param>
        /// <returns>The array.</returns>
        private static NDArray PolyClassCodePoints(string text)
        {
            var cps = new List<long>();
            foreach (Rune r in text.EnumerateRunes())
                cps.Add(r.Value);
            return np.array(cps.ToArray());
        }

        /// <summary>An array-kind result as an array: a series result registered for disposal is not one of them.</summary>
        /// <param name="r">The result.</param>
        /// <param name="owned">The case's owned objects.</param>
        /// <returns>The array.</returns>
        /// <exception cref="NotSupportedException">A series where the case records an array.</exception>
        private static NDArray PolyClassResultArray(object r, List<object> owned)
        {
            switch (r)
            {
                case NDPolyBase s:
                    owned.Add(s);
                    throw new NotSupportedException($"a {s.GetType().Name} result where the case records an array");
                case bool b: return NDArray.Scalar(b);
                default: return PolyResultArray(r);
            }
        }
    }
}
