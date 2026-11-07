using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Cecil.Rocks;

namespace NumSharp.Build;

/// <summary>
///     Weaves NumPy's array-subclass handling into NumSharp.Core's public <c>np.*</c> functions (the self-weave only).
///     A function's mechanism is DECLARED ON THE COMPUTATION it returns (NumSharp.Core's
///     <c>Backends/Subclass/NDSubclassAttributes.cs</c>); this pass finds that computation in the compiled IL, reads its
///     declaration, and gives the function:
///     <list type="number">
///     <item>a PROLOGUE before its first instruction: one <c>NDSubclass.IsOperand</c> test per <see cref="T:NumSharp.NDArray"/>
///           (or <c>NDArray[]</c>) parameter, branching to the subclass path - plain arrays fall through to the untouched
///           body;</item>
///     <item>a generated SUBCLASS PATH, <c>&lt;name&gt;subclass</c>, with the same parameters: the method dispatch of a
///           <c>[WrapReduction]</c> / <c>[WrapFunc]</c> computation (an <c>is NDMatrix __ndmatrix</c> arm per subclass member
///           marked <c>[NumPyMethod]</c>), the entry checks, then an <c>NDScope</c> around the re-call of the function with
///           PLAIN VIEWS of the subclass operands, the unaliasing of an operand handed back, and the declared mechanism on the
///           result.</item>
///     </list>
///     The re-call takes the function's plain path because the views are plain arrays: no per-thread depth is involved,
///     and every computation and public function below it computes plain.
/// </summary>
/// <remarks>
///     <para>
///     <b>Which functions.</b> The public methods of the API modules that take NumPy's subclass operands (<c>np</c> and the
///     types nested in it, <c>np.char</c>, <c>np.strings</c>, <c>np.linalg</c>, <c>np.fft</c>, <c>np.emath</c>; see
///     <see cref="WovenModules"/>) with at least one <c>NDArray</c> / <c>NDArray[]</c> parameter. A function whose return
///     value comes from another overload of itself with the operands passed straight on is TRANSPARENT: the overload it
///     forwards to is woven, so it is left alone.
///     </para>
///     <para>
///     <b>Which mechanism.</b> The returned value's producers are traced back through the IL (branches, locals, casts). A
///     producer that is a call resolves to: the called computation's declaration; the ufunc mechanism when the computation is
///     shaped like a ufunc call (only NDArray inputs, then an optional <c>dtype</c>, then <c>out</c> and <c>where</c>);
///     another public function's own mechanism (mapped through the call's arguments); a non-public helper's returns, traced
///     the same way; otherwise the plain answer. An operand handed back and <c>null</c> are neutral, and so is a call of
///     the runtime itself (<c>NDSubclass</c>). Producers that disagree are an error, never a guess.
///     </para>
///     <para>
///     <b>Report.</b> When the environment variable <c>NUMSHARP_SUBCLASS_REPORT</c> names a file, every API method's
///     resolution is written there as TSV (the migration's review list).
///     </para>
///     <para>
///     Every problem is a build error (NDW022..NDW029 and NDW031): an unwoven function would return the plain class
///     where NumPy returns the subclass.
///     </para>
/// </remarks>
internal static class SubclassWeaver
{
    /// <summary>The runtime the woven code calls.</summary>
    private const string DispatcherFullName = "NumSharp.NDSubclass";

    /// <summary>The NumPy modules whose public functions are woven (their <c>[ModuleName]</c>).</summary>
    /// <remarks>numpy.ma keeps plain views of its operands as a container, np.random converts its parameters the same way,
    ///     and numpy.polynomial computes plain by design; none of them answers with a subclass.</remarks>
    private static readonly HashSet<string> WovenModules = new(StringComparer.Ordinal)
    {
        "np", "np.char", "np.strings", "np.linalg", "np.fft", "np.emath",
    };

    // ---- the declarations (NumSharp.Core's NDSubclassAttributes.cs) ----
    private const string UfuncAttr = "NumSharp.UfuncAttribute";
    private const string ArrayWrapAttr = "NumSharp.ArrayWrapAttribute";
    private const string NewSubtypeAttr = "NumSharp.NewSubtypeAttribute";
    private const string GetSubTypeAttr = "NumSharp.GetSubTypeAttribute";
    private const string AsArrayAttr = "NumSharp.AsArrayAttribute";
    private const string AsArrayForAttr = "NumSharp.AsArrayForAttribute";
    private const string WrapReductionAttr = "NumSharp.WrapReductionAttribute";
    private const string WrapFuncAttr = "NumSharp.WrapFuncAttribute";
    private const string WrapArgumentAttr = "NumSharp.WrapArgumentAttribute";
    private const string ReducesThroughAttr = "NumSharp.ReducesThroughAttribute";
    private const string SubclassRuleAttr = "NumSharp.SubclassRuleAttribute";
    private const string SubclassAwareAttr = "NumSharp.SubclassAwareAttribute";
    private const string ArraySubclassAttr = "NumSharp.ArraySubclassAttribute";
    private const string NumPyMethodAttr = "NumSharp.NumPyMethodAttribute";
    private const string PlainReceiverAttr = "NumSharp.PlainReceiverAttribute";
    private const string ModuleNameAttr = "NumSharp.ModuleNameAttribute";

    private const string NDArrayFullName = "NumSharp.NDArray";
    private const string NDArrayArrayFullName = "NumSharp.NDArray[]";

    /// <summary>The mechanism families a declaration resolves to.</summary>
    private enum Kind
    {
        /// <summary>No declaration: plain views in, the result as computed (an operand handed back stays the caller's).</summary>
        Undeclared,

        /// <summary>The function forwards to another overload of itself: not woven.</summary>
        Transparent,

        /// <summary>The computation is <c>[SubclassAware]</c> (it reads the operand's class itself): not woven, so it
        ///     receives the caller's objects.</summary>
        Aware,

        /// <summary><c>NDSubclass.ArrayWrap</c> (ufuncs and <c>[ArrayWrap]</c>).</summary>
        ArrayWrap,

        /// <summary><c>NDSubclass.NewSubtype</c>.</summary>
        NewSubtype,

        /// <summary><c>NDSubclass.GetSubType</c>.</summary>
        GetSubType,

        /// <summary><c>NDSubclass.AsArray</c>.</summary>
        AsArray,

        /// <summary><c>NDSubclass.Collapse</c> after the entry check of <c>[ReducesThrough]</c>.</summary>
        ReducesThrough,

        /// <summary>A <c>[SubclassRule]</c> method.</summary>
        Custom,

        /// <summary>
        ///     A <c>[PlainReceiver]</c> member (NumSharp's static-typing rule): the receiver and every NDArray operand are
        ///     handed in as PLAIN views without provenance, and the result is the member's own - no mechanism, no audit.
        /// </summary>
        PlainReceiver,
    }

    /// <summary>
    ///     A resolved mechanism, its operands bound to the WOVEN function's parameters (the caller's objects).
    /// </summary>
    private sealed class Rule
    {
        /// <summary>The mechanism family.</summary>
        public Kind Kind;

        /// <summary>The computation the declaration was read from (for the report).</summary>
        public string Source = "";

        /// <summary>Whether the operands were bound through the call's arguments (true) or by role (false).</summary>
        public bool BoundByCallSite = true;

        /// <summary>ArrayWrap: the wrapping inputs (at most three), in NumPy's order.</summary>
        public List<ParameterDefinition> Inputs = new();

        /// <summary>ArrayWrap of a ufunc: the inputs' views are marked for the trivial-loop copy.</summary>
        public bool UfuncInputs;

        /// <summary>
        ///     ArrayWrap of a ufunc whose out= NumPy creates from ONE operand (<c>[Ufunc(Wrap = "a")]</c>): the wrapping
        ///     inputs, a subset of <see cref="Inputs"/> (which are all marked); null when every input may wrap.
        /// </summary>
        public List<ParameterDefinition> WrapInputs;

        /// <summary>NewSubtype / ReducesThrough / dispatch: the operand NumPy calls the method on.</summary>
        public ParameterDefinition Self;

        /// <summary>GetSubType: the sequence operand.</summary>
        public ParameterDefinition Arrays;

        /// <summary>The <c>out=</c> parameter, or null.</summary>
        public ParameterDefinition Out;

        /// <summary>The selected results of a multi-array result (-1: every one).</summary>
        public int Outputs = -1;

        /// <summary>NewSubtype: skip the matrix finalize rule.</summary>
        public bool KeepRank;

        /// <summary>NewSubtype: the <c>NewSubtypeZeroDim</c> value.</summary>
        public int ZeroDim;

        /// <summary>NewSubtype: run the finalize check on the result dtype at the entry.</summary>
        public bool FinalizeFirst;

        /// <summary>The <c>NDSubclassKind</c> bits NumPy converts with asarray first (<c>[AsArrayFor]</c>).</summary>
        public int AsArrayKinds;

        /// <summary>ReducesThrough: the method the keyword reaches, or null.</summary>
        public string ReducesMethod;

        /// <summary>ReducesThrough: the method reached first for an inexact dtype, or null.</summary>
        public string InexactMethod;

        /// <summary>ReducesThrough: the non-matrix classes take <c>NewSubtype</c> (a <c>[NewSubtype]</c> next to it).</summary>
        public bool SubtypeOthers;

        /// <summary>Custom: the rule method, or null for a check only.</summary>
        public MethodDefinition CustomRule;

        /// <summary>Custom: the entry check, or null.</summary>
        public MethodDefinition CustomCheck;

        /// <summary>Method dispatch: the NumPy method's name, or null when the computation declares none.</summary>
        public string DispatchMethod;

        /// <summary>Method dispatch: <c>_wrapreduction</c> (true) or <c>_wrapfunc</c> (false).</summary>
        public bool DispatchIsReduction;

        /// <summary>Method dispatch (<c>_wrapfunc</c>): the parameters passed only when given.</summary>
        public HashSet<string> WhenGiven = new(StringComparer.Ordinal);

        /// <summary>Method dispatch: the method parameters NumPy's function computes first (<c>[WrapArgument]</c>), each
        ///     with its computing method.</summary>
        public Dictionary<string, MethodDefinition> WrapArguments = new(StringComparer.Ordinal);

        /// <summary>
        ///     A transparent function's forwarded mechanism, bound to ITS OWN parameters (the callee's rule re-bound through
        ///     the forwarding call): a different function that returns this one uses it.
        /// </summary>
        public Rule Forwarded;

        /// <summary>Whether the rule declares something even with no result mechanism: a method dispatch, or an
        ///     <c>[AsArrayFor]</c> exception over the plain answer.</summary>
        public bool Declares => Kind != Kind.Undeclared || DispatchMethod is not null || AsArrayKinds != 0;

        /// <summary>A short text for the report.</summary>
        /// <returns>The kind and its bindings.</returns>
        public string Describe()
        {
            string Names(IEnumerable<ParameterDefinition> ps) => string.Join(",", ps.Select(p => p?.Name ?? "-"));
            string s = Kind switch
            {
                Kind.ArrayWrap => $"{(UfuncInputs ? "Ufunc" : "ArrayWrap")}({Names(Inputs)}{(WrapInputs is not null ? $"; wrap={Names(WrapInputs)}" : "")}" +
                                  $"{(Outputs != -1 ? $"; outputs={Outputs}" : "")})",
                Kind.NewSubtype => $"NewSubtype({Self?.Name}{(KeepRank ? "; keepRank" : "")}{(ZeroDim != 0 ? $"; zeroDim={ZeroDim}" : "")}{(FinalizeFirst ? "; finalizeFirst" : "")})",
                Kind.GetSubType => $"GetSubType({Arrays?.Name})",
                Kind.ReducesThrough => $"ReducesThrough({ReducesMethod ?? "null"}{(InexactMethod != null ? "/" + InexactMethod : "")}; {Self?.Name}{(SubtypeOthers ? "; others NewSubtype" : "")})",
                Kind.Custom => $"Custom({CustomRule?.Name ?? "-"}{(CustomCheck != null ? "; check " + CustomCheck.Name : "")})",
                _ => Kind.ToString(),
            };
            if (Out is not null)
                s += $" out={Out.Name}";
            if (DispatchMethod is not null)
                s = $"{(DispatchIsReduction ? "WrapReduction" : "WrapFunc")}({DispatchMethod}; self={Self?.Name}) + " + s;
            if (AsArrayKinds != 0)
                s += $" asArrayKinds={AsArrayKinds}";
            return s;
        }

        /// <summary>Whether two rules weave the same code (same mechanism, same bindings).</summary>
        /// <param name="other">The other rule.</param>
        /// <returns>True when they are interchangeable.</returns>
        public bool SameAs(Rule other)
            => other is not null && Kind == other.Kind && UfuncInputs == other.UfuncInputs && Self == other.Self &&
               Arrays == other.Arrays && Out == other.Out && Outputs == other.Outputs && KeepRank == other.KeepRank &&
               ZeroDim == other.ZeroDim && FinalizeFirst == other.FinalizeFirst && AsArrayKinds == other.AsArrayKinds &&
               ReducesMethod == other.ReducesMethod && InexactMethod == other.InexactMethod && SubtypeOthers == other.SubtypeOthers &&
               CustomRule == other.CustomRule && CustomCheck == other.CustomCheck &&
               DispatchMethod == other.DispatchMethod && DispatchIsReduction == other.DispatchIsReduction &&
               Inputs.SequenceEqual(other.Inputs) &&
               (WrapInputs is null ? other.WrapInputs is null : other.WrapInputs is not null && WrapInputs.SequenceEqual(other.WrapInputs));
    }

    /// <summary>One array subclass the generated dispatch can route to.</summary>
    /// <param name="Type">The class.</param>
    /// <param name="PyName">NumPy's class name, for NumPy's error texts.</param>
    private readonly record struct Subclass(TypeDefinition Type, string PyName);

    /// <summary>Everything one weave needs, resolved once.</summary>
    private sealed class Context
    {
        /// <summary>The module being woven (NumSharp.Core).</summary>
        public ModuleDefinition Module = null!;

        /// <summary><c>NumSharp.NDSubclass</c>: every injected call targets one of its methods.</summary>
        public TypeDefinition Dispatcher = null!;

        /// <summary>Where the build errors go.</summary>
        public TextWriter Stderr = null!;

        /// <summary>The <c>[ArraySubclass]</c> classes, in declaration order (the dispatch arms' order).</summary>
        public List<Subclass> Subclasses = new();

        /// <summary>The memoized resolutions: a computation shared by many functions is resolved once.</summary>
        public Dictionary<MethodDefinition, Rule> Resolved = new();

        /// <summary>The methods being resolved (a cycle resolves to the plain answer instead of recursing forever).</summary>
        public HashSet<MethodDefinition> Resolving = new();

        /// <summary>The report lines (TSV), written when <c>NUMSHARP_SUBCLASS_REPORT</c> is set.</summary>
        public List<string> Report = new();

        /// <summary><c>NumSharp.NDArray</c>.</summary>
        public TypeReference NDArray = null!;

        /// <summary><c>NumSharp.NDArray[]</c>.</summary>
        public TypeReference NDArrayArray = null!;

        /// <summary><c>NumSharp.NDScope</c> (Open, Returns, Dispose).</summary>
        public TypeDefinition Scope = null!;

        /// <summary><c>NumSharp.INDArrayCarrier</c> (a result struct's egress).</summary>
        public TypeDefinition Carrier = null!;

        /// <summary>The number of build errors reported so far.</summary>
        public int Errors;
    }

    /// <summary>
    ///     Weaves every API function of <paramref name="module"/> that takes a subclass operand.
    /// </summary>
    /// <param name="module">The module being woven (NumSharp.Core; any other module is left alone).</param>
    /// <param name="verbose">Whether to list each woven function.</param>
    /// <param name="stdout">Where the verbose list goes.</param>
    /// <param name="stderr">Where the errors go (MSBuild's canonical format).</param>
    /// <param name="errors">Incremented once per refused function.</param>
    /// <returns>The number of functions woven.</returns>
    internal static int Weave(ModuleDefinition module, bool verbose, TextWriter stdout, TextWriter stderr, ref int errors)
    {
        var dispatcher = module.GetType(DispatcherFullName);
        if (dispatcher is null)
            return 0;   // not NumSharp.Core: nothing in this assembly declares the mechanisms

        // Idempotent: an assembly this pass already wove (its marker deleted, an incremental check that misfired) carries
        // the generated paths. Weaving it again would put a second prologue before the first and add a second method of
        // the same name, so a woven module is left as it is.
        if (IsAlreadyWoven(module))
        {
            if (verbose)
                stdout.WriteLine("NumSharp.Build: the subclass paths are already woven; skipping the subclass weave");
            return 0;
        }

        var ctx = new Context
        {
            Module = module,
            Dispatcher = dispatcher,
            Stderr = stderr,
            NDArray = module.GetType(NDArrayFullName),
            Scope = module.GetType("NumSharp.NDScope"),
            Carrier = module.GetType("NumSharp.INDArrayCarrier"),
        };
        ctx.NDArrayArray = new ArrayType(ctx.NDArray);
        foreach (var t in AllTypes(module))
        {
            var declared = t.CustomAttributes.FirstOrDefault(a => a.AttributeType.FullName == ArraySubclassAttr);
            if (declared is not null)
                ctx.Subclasses.Add(new Subclass(t, (string)declared.ConstructorArguments[0].Value));
        }

        // Collect first: the weave adds methods to the types being enumerated.
        var api = new List<MethodDefinition>();
        foreach (var type in AllTypes(module))
            if (IsWovenOwner(type))
                foreach (var m in type.Methods)
                    if (IsApiMethod(m))
                        api.Add(m);

        // Resolve EVERY function before weaving any: a woven body (its prologue calls the generated path) must never be
        // analyzed as the source of another function's mechanism.
        var plan = new List<(MethodDefinition M, Rule Rule, List<ParameterDefinition> Operands)>();
        foreach (var m in api)
        {
            var operands = OperandParameters(m);
            if (operands.Count == 0)
                continue;
            Rule rule;
            try
            {
                rule = Resolve(ctx, m);
            }
            catch (WeaveException e)
            {
                Fail(ctx, e.Code, m, e.Message);
                ctx.Report.Add($"{m.DeclaringType.FullName}\t{Signature(m)}\tERROR\t{e.Code}: {e.Message}");
                continue;
            }
            ctx.Report.Add($"{m.DeclaringType.FullName}\t{Signature(m)}\t{rule.Kind}\t{rule.Describe()}\t" +
                           $"{rule.Source}\t{(rule.BoundByCallSite ? "call-site" : "role")}");
            if (rule.Kind is not (Kind.Transparent or Kind.Aware))
                plan.Add((m, rule, operands));
        }

        // [PlainReceiver] members (NDArray's members, NumSharp's static-typing rule): every one with a body, in any owner.
        foreach (var type in AllTypes(module))
            foreach (var m in type.Methods)
                if (m.HasBody && m.CustomAttributes.Any(a => a.AttributeType.FullName == PlainReceiverAttr))
                {
                    // The re-call runs the member on an NDArray view: an instance member must be NDArray's own (a derived
                    // type's member would run on an object of the wrong type).
                    if (m.HasThis && m.DeclaringType.FullName != NDArrayFullName)
                    {
                        Fail(ctx, "NDW031", m, "[PlainReceiver] applies to NDArray's own members and to static members");
                        continue;
                    }
                    var rule = new Rule { Kind = Kind.PlainReceiver, Source = "[PlainReceiver]" };
                    ctx.Report.Add($"{m.DeclaringType.FullName}\t{Signature(m)}\tPlainReceiver\t{rule.Describe()}\t{rule.Source}\tcall-site");
                    plan.Add((m, rule, OperandParameters(m)));
                }

        int woven = 0;
        foreach (var (m, rule, operands) in plan)
        {
            if (m.HasGenericParameters)
            {
                // A generic function (a typed iterator, asscalar<T>, copyto<T>) computes plain on its operand: nothing to weave.
                // One whose computation declares a mechanism would need its generic parameters cloned onto the subclass path.
                if (rule.Kind == Kind.Undeclared && rule.DispatchMethod is null)
                    continue;
                Fail(ctx, "NDW031", m, $"a generic function cannot be woven ({rule.Describe()} from {rule.Source}); give it a " +
                                       "non-generic overload that the generic one forwards to");
                continue;
            }
            try
            {
                var sub = GenerateSubclassPath(ctx, m, rule, operands);
                WeavePrologue(m, sub, operands, ctx, rule);
                woven++;
                if (verbose)
                    stdout.WriteLine($"NumSharp.Build: woven (array-subclass {rule.Describe()}): {m.FullName}");
            }
            catch (WeaveException e)
            {
                Fail(ctx, e.Code, m, e.Message);
            }
        }

        // Every analyzed body back to its short instruction forms (a woven one was already optimized after its edit; a
        // second pass is harmless).
        foreach (var analyzed in Flow.Simplified)
            analyzed.OptimizeMacros();
        Flow.Simplified.Clear();

        string reportPath = Environment.GetEnvironmentVariable("NUMSHARP_SUBCLASS_REPORT");
        if (!string.IsNullOrEmpty(reportPath))
            File.WriteAllLines(reportPath, new[] { "owner\tmethod\tkind\trule\tsource\tbinding" }.Concat(ctx.Report));

        errors += ctx.Errors;
        return woven;
    }

    #region Discovery

    /// <summary>Whether <paramref name="type"/>'s public functions are woven: a <c>[ModuleName]</c> in
    ///     <see cref="WovenModules"/>, or a type nested in <c>np</c> without a module name of its own.</summary>
    /// <param name="type">The type.</param>
    /// <returns>True for an API owner.</returns>
    private static bool IsWovenOwner(TypeDefinition type)
    {
        for (var t = type; t is not null; t = t.DeclaringType)
        {
            var module = t.CustomAttributes.FirstOrDefault(a => a.AttributeType.FullName == ModuleNameAttr);
            if (module is not null)
                return WovenModules.Contains((string)module.ConstructorArguments[0].Value);
        }
        return false;
    }

    /// <summary>Whether <paramref name="m"/> is a public function of an API module (no accessor, operator or
    ///     constructor).</summary>
    /// <param name="m">The method.</param>
    /// <returns>True for a public function with a body.</returns>
    private static bool IsApiMethod(MethodDefinition m)
        => m.IsPublic && m.HasBody && !m.IsConstructor && !m.IsSpecialName && !m.IsAbstract &&
           IsPublicType(m.DeclaringType) && IsWovenOwner(m.DeclaringType);

    /// <summary>Whether a type is reachable from outside the assembly (public at every nesting level).</summary>
    /// <param name="t">The type.</param>
    /// <returns>True when public.</returns>
    private static bool IsPublicType(TypeDefinition t)
    {
        for (; t is not null; t = t.DeclaringType)
            if (!(t.IsPublic || t.IsNestedPublic))
                return false;
        return true;
    }

    /// <summary>The parameters the prologue tests: every <c>NDArray</c> and <c>NDArray[]</c> parameter.</summary>
    /// <param name="m">The function.</param>
    /// <returns>The operand parameters in declaration order.</returns>
    private static List<ParameterDefinition> OperandParameters(MethodDefinition m)
        => m.Parameters.Where(p => IsNDArray(p) || p.ParameterType.FullName == NDArrayArrayFullName).ToList();

    /// <summary>
    ///     Whether a parameter carries one NDArray: by value, or by reference (<c>in NDArray x1</c>). A by-reference
    ///     operand is read through its address everywhere the weaver loads it, and its plain alias is passed back by the
    ///     address of a local, so the re-call keeps the overload the caller chose.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>True for <c>NDArray</c> and <c>NDArray&amp;</c>; false for anything else, NDArray[] included.</returns>
    private static bool IsNDArray(ParameterDefinition p)
        => p.ParameterType.FullName == NDArrayFullName ||
           p.ParameterType is ByReferenceType br && br.ElementType.FullName == NDArrayFullName;

    /// <summary>
    ///     The full name of the value a parameter carries: its type's, or the element type's for a by-reference parameter.
    ///     Comparisons between a caller's and a callee's parameters use it, so <c>in NDArray</c> binds to <c>NDArray</c>.
    /// </summary>
    /// <param name="p">The parameter.</param>
    /// <returns>The value type's full name.</returns>
    private static string ValueTypeName(ParameterDefinition p)
        => p.ParameterType is ByReferenceType br ? br.ElementType.FullName : p.ParameterType.FullName;

    /// <summary>
    ///     Emits the VALUE of a parameter: <c>ldarg</c>, dereferenced (<c>ldind.ref</c>) for a by-reference NDArray, so
    ///     every runtime call receives the array itself, never its address.
    /// </summary>
    /// <param name="il">The IL processor of the method being emitted.</param>
    /// <param name="p">The parameter, of the method being emitted.</param>
    private static void LoadValue(ILProcessor il, ParameterDefinition p)
    {
        il.Append(il.Create(OpCodes.Ldarg, p));
        if (p.ParameterType.IsByReference)
            il.Append(il.Create(OpCodes.Ldind_Ref));
    }

    /// <summary>
    ///     The parameter whose VALUE an instruction pushes: <c>ldarg p</c> of a by-value parameter, or the
    ///     <c>ldind.ref</c> that dereferences <c>ldarg p</c> of a by-reference NDArray (how C# reads an <c>in</c> operand).
    ///     A by-reference parameter's bare <c>ldarg</c> pushes its address, not the array, so it is not a load of it.
    /// </summary>
    /// <param name="producer">The instruction, or null.</param>
    /// <returns>The parameter, or null for any other instruction.</returns>
    private static ParameterDefinition LoadedParameter(Instruction producer)
    {
        if (producer?.OpCode.Code == Code.Ldarg && producer.Operand is ParameterDefinition p)
            return p.ParameterType.IsByReference ? null : p;
        if (producer?.OpCode.Code == Code.Ldind_Ref && producer.Previous?.OpCode.Code == Code.Ldarg &&
            producer.Previous.Operand is ParameterDefinition q && q.ParameterType.IsByReference)
            return q;
        return null;
    }

    #endregion

    #region Resolution

    /// <summary>
    ///     The mechanism of <paramref name="m"/>: from the computations its returns come from (memoized; a cycle is an
    ///     error).
    /// </summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="m">The function or helper.</param>
    /// <returns>The rule, its operands bound to <paramref name="m"/>'s parameters.</returns>
    /// <exception cref="WeaveException">Producers that disagree, or a declaration that cannot be bound.</exception>
    private static Rule Resolve(Context ctx, MethodDefinition m)
    {
        if (ctx.Resolved.TryGetValue(m, out var known))
            return known;
        if (!ctx.Resolving.Add(m))
            return new Rule { Kind = Kind.Undeclared, Source = "(recursive)" };
        try
        {
            var rule = ResolveUncached(ctx, m);
            ctx.Resolved[m] = rule;
            return rule;
        }
        finally
        {
            ctx.Resolving.Remove(m);
        }
    }

    /// <summary>The body of <see cref="Resolve"/>.</summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="m">The function or helper.</param>
    /// <returns>The rule.</returns>
    /// <exception cref="WeaveException">Producers that disagree.</exception>
    private static Rule ResolveUncached(Context ctx, MethodDefinition m)
    {
        var flow = new Flow(m.Body);
        var producers = new List<Instruction>();
        foreach (var ret in m.Body.Instructions.Where(i => i.OpCode.Code == Code.Ret))
            producers.AddRange(flow.ValueSources(ret));

        var rules = new List<Rule>();
        bool transparent = false, undeclared = false, aware = false;
        Rule forwarded = null;
        foreach (var p in producers.Distinct())
        {
            switch (p?.OpCode.Code)
            {
                case Code.Call or Code.Callvirt or Code.Newobj:
                    var callee = (MethodReference)p.Operand;
                    if (callee.DeclaringType.FullName == DispatcherFullName)
                        continue;   // the runtime (an alias handed back, a container's plain view): neutral
                    var r = RuleOfCall(ctx, m, flow, p, callee);
                    if (r.Kind == Kind.Transparent && r.Forwarded is not null)
                        forwarded ??= r.Forwarded;
                    if (r.Kind == Kind.Aware)
                        aware = true;
                    else if (r.Kind == Kind.Transparent)
                        transparent = true;
                    else if (r.Kind == Kind.Undeclared && !r.Declares)
                        undeclared = true;
                    else
                        rules.Add(r);
                    break;
                case Code.Ldarg or Code.Ldnull:
                    continue;   // an operand handed back, or null: neutral
                case Code.Ldind_Ref when LoadedParameter(p) is not null:
                    continue;   // an `in` operand handed back: neutral
                default:
                    undeclared = true;
                    break;
            }
        }

        if (aware)
            return new Rule { Kind = Kind.Aware, Source = "subclass-aware computation" };
        if (rules.Count == 0)
            return new Rule
            {
                Kind = transparent && !undeclared ? Kind.Transparent : Kind.Undeclared, Source = transparent ? "forwards" : "",
                Forwarded = forwarded,
            };
        var first = rules[0];
        foreach (var other in rules.Skip(1))
            if (!first.SameAs(other))
                throw new WeaveException("NDW022", $"its returns come from computations with different subclass mechanisms " +
                                                   $"({first.Describe()} from {first.Source} and {other.Describe()} from {other.Source}); " +
                                                   "give each route one computation, or declare a [SubclassRule]");
        if (undeclared)
            first.Source += " (+undeclared routes)";
        return first;
    }

    /// <summary>
    ///     The rule a call contributes as a producer of <paramref name="m"/>'s return value.
    /// </summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="m">The method whose return is being traced.</param>
    /// <param name="flow">Its flow graph.</param>
    /// <param name="call">The call instruction.</param>
    /// <param name="callee">The called method.</param>
    /// <returns>The rule bound to <paramref name="m"/>'s parameters.</returns>
    /// <exception cref="WeaveException">A declaration that cannot be bound.</exception>
    private static Rule RuleOfCall(Context ctx, MethodDefinition m, Flow flow, Instruction call, MethodReference callee)
    {
        MethodDefinition target;
        try
        {
            target = callee.Resolve();
        }
        catch (AssemblyResolutionException)
        {
            return new Rule { Kind = Kind.Undeclared };
        }
        if (target is null || target.Module != ctx.Module)
            return new Rule { Kind = Kind.Undeclared, Source = callee.FullName };

        var args = flow.ArgumentProducers(call, target);

        // A conversion of the computed array to the function's C# return type (np.argmax(a) returns (long)ArgMax(a)): the
        // converted call is the computation.
        if (target.IsStatic && target.Name is "op_Explicit" or "op_Implicit" && args.Length == 1 &&
            args[0]?.OpCode.Code is Code.Call or Code.Callvirt)
            return RuleOfCall(ctx, m, flow, args[0], (MethodReference)args[0].Operand);

        // Another public function: its own mechanism, re-bound through the call's arguments.
        if (target != m && IsApiMethod(target) && OperandParameters(target).Count > 0)
        {
            // A forwarder to the same NumPy function (another overload, np.char.isalpha to np.strings.isalpha) that passes
            // its own operands straight on is left to the callee, which tests them (and any operand it builds) itself.
            if (target.Name == m.Name && OperandsPassedStraight(args, m))
            {
                var forwardedTo = Resolve(ctx, target);
                var effective = forwardedTo.Kind == Kind.Transparent ? forwardedTo.Forwarded : forwardedTo;
                return new Rule
                {
                    Kind = Kind.Transparent, Source = target.FullName,
                    Forwarded = effective is null || !effective.Declares || effective.Kind is Kind.Aware or Kind.Transparent
                        ? effective
                        : Rebind(ctx, effective, target, args, m, target.FullName),
                };
            }
            var inner = Resolve(ctx, target);
            if (inner.Kind == Kind.Transparent && inner.Forwarded is not null)
                inner = inner.Forwarded;   // the overload it forwards to answers
            if (inner.Kind is Kind.Transparent or Kind.Aware || inner.Kind == Kind.Undeclared && !inner.Declares)
                return new Rule { Kind = inner.Kind == Kind.Transparent ? Kind.Undeclared : inner.Kind, Source = target.FullName };
            return Rebind(ctx, inner, target, args, m, target.FullName);
        }

        var declared = Declared(ctx, target, args, m);
        if (declared is not null)
            return declared;

        // A non-public helper of an API owner (or of NDArray's array-object home) that returns the computation: trace it.
        if (target.HasBody && !target.IsPublic && IsWovenOwner(target.DeclaringType) && target != m)
        {
            var inner = Resolve(ctx, target);
            if (inner.Kind is Kind.Transparent || inner.Kind == Kind.Undeclared && !inner.Declares)
                return new Rule { Kind = Kind.Undeclared, Source = target.FullName };
            return Rebind(ctx, inner, target, args, m, target.FullName);
        }
        return new Rule { Kind = Kind.Undeclared, Source = target.FullName };
    }

    /// <summary>
    ///     The rule a computation DECLARES (its attributes, or the ufunc shape), bound to <paramref name="m"/>'s parameters
    ///     through the call's arguments, by role where an argument is not a parameter of <paramref name="m"/>.
    /// </summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="target">The computation.</param>
    /// <param name="args">The call's argument producers (index 0 is the receiver of an instance call).</param>
    /// <param name="m">The calling function.</param>
    /// <returns>The rule, or null when the computation declares nothing.</returns>
    /// <exception cref="WeaveException">A declaration that names no parameter of the computation.</exception>
    private static Rule Declared(Context ctx, MethodDefinition target, Instruction[] args, MethodDefinition m)
    {
        CustomAttribute Attr(string name) => target.CustomAttributes.FirstOrDefault(a => a.AttributeType.FullName == name);
        var rule = new Rule { Source = target.DeclaringType.Name + "." + target.Name, Out = OutParameter(m) };

        var reduction = Attr(WrapReductionAttr);
        var func = Attr(WrapFuncAttr);
        if (reduction is not null || func is not null)
        {
            var d = reduction ?? func;
            rule.DispatchMethod = (string)d.ConstructorArguments[0].Value;
            rule.DispatchIsReduction = reduction is not null;
            foreach (var prop in d.Properties)
                if (prop.Name == "WhenGiven" && prop.Argument.Value is string list)
                    foreach (var n in list.Split(','))
                        rule.WhenGiven.Add(n.Trim());
            foreach (var w in target.CustomAttributes.Where(a => a.AttributeType.FullName == WrapArgumentAttr))
            {
                string parameter = (string)w.ConstructorArguments[0].Value;
                var holder = ((TypeReference)w.ConstructorArguments[1].Value).Resolve();
                string name = (string)w.ConstructorArguments[2].Value;
                rule.WrapArguments[parameter] = holder.Methods.FirstOrDefault(x => x.Name == name && x.IsStatic)
                    ?? throw new WeaveException("NDW023", $"{holder.FullName} has no static method '{name}' for [WrapArgument({parameter})]");
            }
        }

        foreach (var a in target.CustomAttributes.Where(a => a.AttributeType.FullName == AsArrayForAttr))
            rule.AsArrayKinds |= KindBit(ctx, (TypeReference)a.ConstructorArguments[0].Value, target);

        if (Attr(SubclassAwareAttr) is not null)
            return new Rule { Kind = Kind.Aware, Source = target.DeclaringType.Name + "." + target.Name };

        CustomAttribute attr;
        if ((attr = Attr(ReducesThroughAttr)) is not null)
        {
            // A matrix intermediate's _collapse; a [NewSubtype] next to it is how NumPy's body answers for the other
            // classes (np.percentile of a chararray: the subtype copy partitioned in place).
            rule.Kind = Kind.ReducesThrough;
            rule.ReducesMethod = attr.ConstructorArguments[0].Value as string;
            rule.InexactMethod = attr.Properties.FirstOrDefault(p => p.Name == "InexactMethod").Argument.Value as string;
            rule.Self = SelfOf(target, null, args, m, rule);
            var others = Attr(NewSubtypeAttr);
            if (others is not null)
            {
                rule.SubtypeOthers = true;
                foreach (var prop in others.Properties)
                    if (prop.Name == "ZeroDim")
                        rule.ZeroDim = (int)prop.Argument.Value;
            }
        }
        else if ((attr = Attr(SubclassRuleAttr)) is not null)
        {
            rule.Kind = Kind.Custom;
            var holder = ((TypeReference)attr.ConstructorArguments[0].Value).Resolve();
            string ruleName = attr.ConstructorArguments[1].Value as string;
            string checkName = attr.Properties.FirstOrDefault(p => p.Name == "Check").Argument.Value as string;
            if (ruleName is not null)
                rule.CustomRule = holder.Methods.FirstOrDefault(x => x.Name == ruleName && x.IsStatic)
                                  ?? throw new WeaveException("NDW023", $"{holder.FullName} has no static rule method '{ruleName}'");
            if (checkName is not null)
                rule.CustomCheck = holder.Methods.FirstOrDefault(x => x.Name == checkName && x.IsStatic)
                                   ?? throw new WeaveException("NDW023", $"{holder.FullName} has no static check method '{checkName}'");
            // A custom rule binds its arguments by NAME, so it needs no self operand: a function whose operands are only
            // sequences (np.meshgrid(NDArray[] xi), np.atleast_1d(params NDArray[])) has none, and that is no error.
            rule.Self = m.Parameters.Any(IsNDArray) ? SelfOf(target, null, args, m, rule) : null;
        }
        else if ((attr = Attr(UfuncAttr)) is not null)
        {
            rule.Kind = Kind.ArrayWrap;
            rule.UfuncInputs = true;
            BindInputs(rule, target, UfuncInputNames(target), args, m);
            rule.Outputs = OutputsOf(attr);
            // Wrap = "a": every input is marked, only the named ones choose the wrap (NumPy's out= is empty_like(a)).
            if (attr.Properties.FirstOrDefault(p => p.Name == "Wrap").Argument.Value is string wrap)
            {
                var wrapRule = new Rule();
                BindInputs(wrapRule, target, wrap.Split(',').Select(n => n.Trim()).ToList(), args, m);
                rule.WrapInputs = wrapRule.Inputs;
                rule.BoundByCallSite &= wrapRule.BoundByCallSite;
            }
        }
        else if ((attr = Attr(ArrayWrapAttr)) is not null)
        {
            rule.Kind = Kind.ArrayWrap;
            BindInputs(rule, target, ConstructorStrings(attr), args, m);
            rule.Outputs = OutputsOf(attr);
        }
        else if ((attr = Attr(NewSubtypeAttr)) is not null)
        {
            rule.Kind = Kind.NewSubtype;
            rule.Self = SelfOf(target, (string)attr.ConstructorArguments[0].Value, args, m, rule);
            foreach (var prop in attr.Properties)
            {
                if (prop.Name == "KeepRank")
                    rule.KeepRank = (bool)prop.Argument.Value;
                else if (prop.Name == "ZeroDim")
                    rule.ZeroDim = (int)prop.Argument.Value;
                else if (prop.Name == "FinalizeFirst")
                    rule.FinalizeFirst = (bool)prop.Argument.Value;
            }
        }
        else if ((attr = Attr(GetSubTypeAttr)) is not null)
        {
            rule.Kind = Kind.GetSubType;
            string arraysName = (string)attr.ConstructorArguments[0].Value;
            rule.Arrays = MapNamed(target, arraysName, args, m)
                          ?? m.Parameters.FirstOrDefault(p => p.ParameterType.FullName == NDArrayArrayFullName);
            if (rule.Arrays is null)
                throw new WeaveException("NDW024", $"[GetSubType({arraysName})] on {target.FullName}: the function has no NDArray[] operand");
        }
        else if ((attr = Attr(AsArrayAttr)) is not null)
        {
            rule.Kind = Kind.AsArray;
        }
        else if (UfuncShaped(target))
        {
            rule.Kind = Kind.ArrayWrap;
            rule.UfuncInputs = true;
            BindInputs(rule, target, UfuncInputNames(target), args, m);
            rule.Source += " (ufunc shape)";
        }
        else if (rule.DispatchMethod is not null || rule.AsArrayKinds != 0)
        {
            rule.Kind = Kind.Undeclared;   // a dispatch (or an asarray exception) over the plain answer
        }
        else
        {
            return null;
        }

        if (rule.DispatchMethod is not null && rule.Self is null)
            rule.Self = SelfOf(target, null, args, m, rule);
        if (rule.AsArrayKinds != 0 && rule.Self is null)
            rule.Self = SelfOf(target, null, args, m, rule);
        return rule;
    }

    /// <summary>The <c>NDSubclassKind</c> bit of an <c>[AsArrayFor(typeof(...))]</c> subclass.</summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="subclass">The declared subclass.</param>
    /// <param name="target">The declaring computation (for the error).</param>
    /// <returns><c>1 &lt;&lt; kind</c>.</returns>
    /// <exception cref="WeaveException">Not one of the known subclasses.</exception>
    private static int KindBit(Context ctx, TypeReference subclass, MethodDefinition target)
        => subclass.FullName switch
        {
            "NumSharp.NDMatrix" => 1 << 1,
            "NumSharp.NDCharArray" => 1 << 2,
            _ => throw new WeaveException("NDW024", $"[AsArrayFor({subclass.FullName})] on {target.FullName} names no array subclass"),
        };

    /// <summary>Binds a wrap's inputs: each named computation parameter through the call site, else by role.</summary>
    /// <param name="rule">The rule being built.</param>
    /// <param name="target">The computation.</param>
    /// <param name="names">Its input parameters' names (at most three are kept).</param>
    /// <param name="args">The call's argument producers.</param>
    /// <param name="m">The calling function.</param>
    private static void BindInputs(Rule rule, MethodDefinition target, IReadOnlyList<string> names, Instruction[] args, MethodDefinition m)
    {
        var roles = m.Parameters.Where(p => IsNDArray(p) && p.Name is not "out" and not "where").ToList();
        for (int i = 0; i < names.Count && rule.Inputs.Count < 3; i++)
        {
            var mapped = MapNamed(target, names[i], args, m);
            if (mapped is null || !IsNDArray(mapped))
            {
                rule.BoundByCallSite = false;
                mapped = i < roles.Count ? roles[i] : null;
            }
            if (mapped is not null && !rule.Inputs.Contains(mapped))
                rule.Inputs.Add(mapped);
        }
    }

    /// <summary>The operand NumPy calls the method on, bound to the calling function's parameter.</summary>
    /// <param name="target">The computation.</param>
    /// <param name="name">The declared self parameter's name (<c>this</c> for the receiver), or null for the first operand.</param>
    /// <param name="args">The call's argument producers.</param>
    /// <param name="m">The calling function.</param>
    /// <param name="rule">The rule (its binding mode is updated).</param>
    /// <returns>The parameter of <paramref name="m"/>.</returns>
    /// <exception cref="WeaveException">The function has no NDArray operand.</exception>
    private static ParameterDefinition SelfOf(MethodDefinition target, string name, Instruction[] args, MethodDefinition m, Rule rule)
    {
        ParameterDefinition mapped = null;
        if (name is not null)
            mapped = MapNamed(target, name, args, m);
        else
        {
            // The computation's first NDArray operand (its receiver when it is an NDArray member).
            if (target.HasThis && target.DeclaringType.FullName == NDArrayFullName)
                mapped = MapToCaller(args, target, null);
            else
            {
                var first = target.Parameters.FirstOrDefault(p => p.ParameterType.FullName == NDArrayFullName);
                if (first is not null)
                    mapped = MapToCaller(args, target, first);
            }
        }
        if (mapped is null || !IsNDArray(mapped))
        {
            rule.BoundByCallSite = false;
            mapped = m.Parameters.FirstOrDefault(IsNDArray);
        }
        return mapped ?? throw new WeaveException("NDW024", $"{target.FullName}'s mechanism needs an NDArray operand, and the function has none");
    }

    /// <summary>The calling function's parameter passed straight as the computation's parameter <paramref name="name"/>.</summary>
    /// <param name="target">The computation.</param>
    /// <param name="name">The parameter's name (<c>this</c> for the receiver).</param>
    /// <param name="args">The call's argument producers.</param>
    /// <param name="m">The calling function.</param>
    /// <returns>The parameter, or null when the argument is anything else.</returns>
    /// <exception cref="WeaveException">The computation has no parameter of that name.</exception>
    private static ParameterDefinition MapNamed(MethodDefinition target, string name, Instruction[] args, MethodDefinition m)
    {
        if (name == "this")
            return MapToCaller(args, target, null);
        var p = target.Parameters.FirstOrDefault(x => x.Name == name);
        if (p is null)
            throw new WeaveException("NDW024", $"the declaration on {target.FullName} names '{name}', which is not one of its parameters");
        return MapToCaller(args, target, p);
    }

    /// <summary>
    ///     The calling function's parameter that a call passes straight (a load of it, through casts) as the callee's
    ///     parameter <paramref name="calleeParam"/> (null: the receiver).
    /// </summary>
    /// <param name="args">The call's argument producers (index 0 is the receiver of an instance call), or null.</param>
    /// <param name="callee">The called method.</param>
    /// <param name="calleeParam">The callee's parameter, or null for its receiver.</param>
    /// <returns>The caller's parameter, or null.</returns>
    private static ParameterDefinition MapToCaller(Instruction[] args, MethodDefinition callee, ParameterDefinition calleeParam)
    {
        if (args is null)
            return null;
        int slot = calleeParam is null ? 0 : calleeParam.Index + (callee.HasThis ? 1 : 0);
        if (slot >= args.Length)
            return null;
        // A by-value load, or the dereference of an `in` operand's address.
        return LoadedParameter(args[slot]);
    }

    /// <summary>
    ///     A rule resolved for a callee, re-bound to the caller's parameters through the call's arguments (by role where an
    ///     argument is not a caller parameter).
    /// </summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="inner">The callee's rule.</param>
    /// <param name="callee">The callee.</param>
    /// <param name="args">The call's argument producers.</param>
    /// <param name="m">The caller.</param>
    /// <param name="source">The report text.</param>
    /// <returns>The re-bound rule.</returns>
    private static Rule Rebind(Context ctx, Rule inner, MethodDefinition callee, Instruction[] args, MethodDefinition m, string source)
    {
        var r = new Rule
        {
            Kind = inner.Kind, UfuncInputs = inner.UfuncInputs, Outputs = inner.Outputs, KeepRank = inner.KeepRank,
            ZeroDim = inner.ZeroDim, FinalizeFirst = inner.FinalizeFirst, AsArrayKinds = inner.AsArrayKinds,
            ReducesMethod = inner.ReducesMethod, InexactMethod = inner.InexactMethod, SubtypeOthers = inner.SubtypeOthers,
            CustomRule = inner.CustomRule,
            CustomCheck = inner.CustomCheck, DispatchMethod = inner.DispatchMethod, DispatchIsReduction = inner.DispatchIsReduction,
            WhenGiven = inner.WhenGiven, WrapArguments = inner.WrapArguments, BoundByCallSite = inner.BoundByCallSite,
            Source = source + " <- " + inner.Source,
            Out = OutParameter(m),
        };
        var roles = m.Parameters.Where(p => IsNDArray(p) && p.Name is not "out" and not "where").ToList();
        ParameterDefinition Map(ParameterDefinition calleeParam, ParameterDefinition fallback)
        {
            if (calleeParam is null)
                return null;
            var mapped = MapToCaller(args, callee, calleeParam);
            if (mapped is not null && ValueTypeName(mapped) == ValueTypeName(calleeParam))
                return mapped;
            r.BoundByCallSite = false;
            return fallback;
        }
        for (int i = 0; i < inner.Inputs.Count; i++)
        {
            var mapped = Map(inner.Inputs[i], i < roles.Count ? roles[i] : null);
            if (mapped is not null && !r.Inputs.Contains(mapped))
                r.Inputs.Add(mapped);
        }
        if (inner.WrapInputs is not null)
        {
            // The wrapping subset maps like the inputs it is part of (a role fallback takes the input's position).
            r.WrapInputs = new List<ParameterDefinition>();
            foreach (var w in inner.WrapInputs)
            {
                int position = inner.Inputs.IndexOf(w);
                var mapped = Map(w, position >= 0 && position < roles.Count ? roles[position] : null);
                if (mapped is not null && !r.WrapInputs.Contains(mapped))
                    r.WrapInputs.Add(mapped);
            }
        }
        r.Self = Map(inner.Self, m.Parameters.FirstOrDefault(IsNDArray));
        r.Arrays = Map(inner.Arrays, m.Parameters.FirstOrDefault(p => p.ParameterType.FullName == NDArrayArrayFullName));
        return r;
    }

    /// <summary>
    ///     Whether every operand parameter of <paramref name="m"/> reaches the call as a direct argument (a load of the
    ///     parameter itself), so the callee sees exactly the caller's operands.
    /// </summary>
    /// <param name="args">The call's argument producers.</param>
    /// <param name="m">The calling function.</param>
    /// <returns>True when no operand of <paramref name="m"/> is transformed or dropped on the way.</returns>
    private static bool OperandsPassedStraight(Instruction[] args, MethodDefinition m)
    {
        if (args is null)
            return false;
        foreach (var p in OperandParameters(m))
            if (!args.Any(a => LoadedParameter(a) == p))
                return false;
        return true;
    }

    /// <summary>The function's <c>out=</c> parameter (an <c>NDArray</c> named <c>out</c>), or null.</summary>
    /// <param name="m">The function.</param>
    /// <returns>The parameter, or null.</returns>
    private static ParameterDefinition OutParameter(MethodDefinition m)
        => m.Parameters.FirstOrDefault(p => p.Name == "out" && p.ParameterType.FullName == NDArrayFullName);

    /// <summary>
    ///     Whether a computation is shaped like a ufunc call: it returns <c>NDArray</c>, and its parameters are one to three
    ///     <c>NDArray</c> inputs, then an optional <c>DType dtype</c>, then <c>NDArray out</c> and <c>NDArray where</c> -
    ///     nothing else (a reduction's <c>axis</c> / <c>keepdims</c> / <c>initial</c> rule it out).
    /// </summary>
    /// <param name="m">The computation.</param>
    /// <returns>True for a ufunc by shape.</returns>
    private static bool UfuncShaped(MethodDefinition m)
    {
        if (m.ReturnType.FullName != NDArrayFullName)
            return false;
        var ps = m.Parameters;
        int n = ps.Count;
        if (n < 3 || ps[n - 1].Name != "where" || ps[n - 1].ParameterType.FullName != NDArrayFullName ||
            ps[n - 2].Name != "out" || ps[n - 2].ParameterType.FullName != NDArrayFullName)
            return false;
        int head = n - 2;
        if (head > 0 && ps[head - 1].Name == "dtype" && ps[head - 1].ParameterType.FullName == "NumSharp.DType")
            head--;
        if (head < 1 || head > 3)
            return false;
        for (int i = 0; i < head; i++)
            if (ps[i].ParameterType.FullName != NDArrayFullName)
                return false;
        return true;
    }

    /// <summary>A ufunc computation's input names: its NDArray parameters other than <c>out</c> and <c>where</c>.</summary>
    /// <param name="m">The computation.</param>
    /// <returns>The names in order.</returns>
    private static List<string> UfuncInputNames(MethodDefinition m)
        => m.Parameters.Where(p => p.ParameterType.FullName == NDArrayFullName && p.Name is not "out" and not "where")
            .Select(p => p.Name).ToList();

    /// <summary>The <c>Outputs</c> selection of a wrap attribute as a bit mask (-1: every result).</summary>
    /// <param name="attr">The attribute.</param>
    /// <returns>The mask.</returns>
    private static int OutputsOf(CustomAttribute attr)
    {
        var text = attr.Properties.FirstOrDefault(p => p.Name == "Outputs").Argument.Value as string;
        if (string.IsNullOrWhiteSpace(text))
            return -1;
        int mask = 0;
        foreach (var part in text.Split(','))
            mask |= 1 << int.Parse(part.Trim());
        return mask;
    }

    #endregion

    #region The flow graph (value producers, call arguments)

    /// <summary>
    ///     The small part of an IL flow analysis the resolution needs: which instructions produced the value on top of the
    ///     stack at a consumer (through branches, locals, casts and <c>dup</c>), and which produced each argument of a call
    ///     (straight-line argument evaluation only).
    /// </summary>
    private sealed class Flow
    {
        /// <summary>
        ///     Every body an analysis put into long instruction forms: <see cref="Weave"/> optimizes them back at the end, so a
        ///     function that is only analyzed keeps its compact encoding.
        /// </summary>
        public static readonly HashSet<MethodBody> Simplified = new();

        /// <summary>The method body.</summary>
        private readonly MethodBody _body;

        /// <summary>Each instruction's predecessors (fall-through and branches).</summary>
        private readonly Dictionary<Instruction, List<Instruction>> _preds = new();

        /// <summary>Instructions that are the target of a branch.</summary>
        private readonly HashSet<Instruction> _targets = new();

        /// <summary>Builds the graph over the body's LONG instruction forms (<c>ldarg</c>, <c>ldloc</c>, <c>stloc</c> with
        ///     operands), which the analysis compares; the caller restores the short forms afterwards
        ///     (<see cref="Simplified"/>).</summary>
        /// <param name="body">The method body.</param>
        public Flow(MethodBody body)
        {
            _body = body;
            body.SimplifyMacros();
            Simplified.Add(body);
            var list = body.Instructions;
            for (int i = 0; i < list.Count; i++)
            {
                var ins = list[i];
                if (i + 1 < list.Count && FallsThrough(ins))
                    Pred(list[i + 1]).Add(ins);
                switch (ins.Operand)
                {
                    case Instruction t when ins.OpCode.FlowControl is FlowControl.Branch or FlowControl.Cond_Branch:
                        Pred(t).Add(ins);
                        _targets.Add(t);
                        break;
                    case Instruction[] many:
                        foreach (var t in many)
                        {
                            Pred(t).Add(ins);
                            _targets.Add(t);
                        }
                        break;
                }
            }
            foreach (var h in body.ExceptionHandlers)
            {
                _targets.Add(h.HandlerStart);
                if (h.FilterStart is not null)
                    _targets.Add(h.FilterStart);
            }
        }

        /// <summary>The predecessor list of an instruction (created on demand).</summary>
        /// <param name="i">The instruction.</param>
        /// <returns>Its list.</returns>
        private List<Instruction> Pred(Instruction i)
        {
            if (!_preds.TryGetValue(i, out var l))
                _preds[i] = l = new List<Instruction>();
            return l;
        }

        /// <summary>Whether control falls from <paramref name="i"/> to the next instruction.</summary>
        /// <param name="i">The instruction.</param>
        /// <returns>False after an unconditional branch, a return, a throw or a leave.</returns>
        private static bool FallsThrough(Instruction i)
            => i.OpCode.FlowControl is not (FlowControl.Branch or FlowControl.Return or FlowControl.Throw) &&
               i.OpCode.Code is not (Code.Leave or Code.Leave_S or Code.Endfinally);

        /// <summary>
        ///     The instructions that produced the value on top of the stack when <paramref name="consumer"/> runs: a call, a
        ///     load of a parameter, <c>ldnull</c>, or any other producer (classified by the caller). Locals are followed to
        ///     their stores; casts and <c>dup</c> to their input.
        /// </summary>
        /// <param name="consumer">An instruction that consumes the stack top (<c>ret</c>, <c>stloc</c>).</param>
        /// <returns>The producers (null for a route the analysis cannot follow).</returns>
        public List<Instruction> ValueSources(Instruction consumer)
        {
            var result = new List<Instruction>();
            Sources(consumer, result, new HashSet<Instruction>(), 0);
            return result;
        }

        /// <summary>The recursion of <see cref="ValueSources"/>.</summary>
        /// <param name="at">The instruction whose incoming stack top is traced.</param>
        /// <param name="result">Where the producers go.</param>
        /// <param name="seen">The instructions already traced (loops).</param>
        /// <param name="depth">The recursion depth (a guard).</param>
        private void Sources(Instruction at, List<Instruction> result, HashSet<Instruction> seen, int depth)
        {
            if (depth > 64 || !seen.Add(at))
                return;
            if (!_preds.TryGetValue(at, out var preds) || preds.Count == 0)
            {
                result.Add(null);
                return;
            }
            foreach (var p in preds)
            {
                switch (p.OpCode.Code)
                {
                    case Code.Br or Code.Br_S or Code.Nop:
                        Sources(p, result, seen, depth + 1);   // the value is still on the stack
                        break;
                    case Code.Castclass or Code.Isinst or Code.Unbox_Any or Code.Dup:
                        Sources(p, result, seen, depth + 1);   // the same object, re-typed or duplicated
                        break;
                    case Code.Ldloc:
                        var v = (VariableDefinition)p.Operand;
                        if (_body.Instructions.Any(i => i.OpCode.Code == Code.Ldloca && i.Operand == v))
                        {
                            result.Add(null);   // its address escapes: any store is possible
                            break;
                        }
                        foreach (var store in _body.Instructions.Where(i => i.OpCode.Code == Code.Stloc && i.Operand == v))
                            Sources(store, result, seen, depth + 1);
                        break;
                    default:
                        if (p.OpCode.FlowControl is FlowControl.Cond_Branch)
                            result.Add(null);   // a value below a branch condition: not followed
                        else
                            result.Add(p);
                        break;
                }
            }
        }

        /// <summary>
        ///     The producers of each argument of <paramref name="call"/> (index 0 is the receiver of an instance call), found
        ///     by walking the straight-line code before it; null when the walk crosses a branch target.
        /// </summary>
        /// <param name="call">A call or newobj instruction.</param>
        /// <param name="callee">The resolved callee.</param>
        /// <returns>The producers, an element null where unknown; null when nothing could be attributed.</returns>
        public Instruction[] ArgumentProducers(Instruction call, MethodDefinition callee)
        {
            bool isNewObj = call.OpCode.Code == Code.Newobj;
            int count = callee.Parameters.Count + (callee.HasThis && !isNewObj ? 1 : 0);
            var producers = new Instruction[count];
            if (count == 0)
                return producers;
            if (_targets.Contains(call))
                return producers;   // the arguments arrive from several routes: none is attributed
            int height = count;   // stack height above the first argument's slot, just before the call
            for (var i = call.Previous; i is not null; i = i.Previous)
            {
                if (_targets.Contains(i.Next) && i.Next != call || i.OpCode.FlowControl is FlowControl.Branch or FlowControl.Cond_Branch
                        or FlowControl.Return or FlowControl.Throw)
                    return producers;   // a join or a branch: the remaining slots stay unknown
                (int pop, int push) = StackEffect(i);
                // The pushed values occupy [height - push, height - 1].
                for (int k = 0; k < push; k++)
                {
                    int slot = height - push + k;
                    if (slot >= 0 && slot < count && producers[slot] is null)
                        producers[slot] = i;
                }
                height = height - push + pop;
                if (height <= 0)
                    return producers;
            }
            return producers;
        }

        /// <summary>The number of values an instruction pops and pushes.</summary>
        /// <param name="i">The instruction.</param>
        /// <returns>(pop, push).</returns>
        private static (int Pop, int Push) StackEffect(Instruction i)
        {
            if (i.Operand is MethodReference mr && i.OpCode.Code is Code.Call or Code.Callvirt or Code.Newobj)
            {
                bool newobj = i.OpCode.Code == Code.Newobj;
                int pop = mr.Parameters.Count + (mr.HasThis && !newobj ? 1 : 0);
                int push = newobj || mr.ReturnType.MetadataType != MetadataType.Void ? 1 : 0;
                return (pop, push);
            }
            int popCount = i.OpCode.StackBehaviourPop switch
            {
                StackBehaviour.Pop0 => 0,
                StackBehaviour.Pop1 or StackBehaviour.Popi or StackBehaviour.Popref => 1,
                StackBehaviour.Pop1_pop1 or StackBehaviour.Popi_pop1 or StackBehaviour.Popi_popi or StackBehaviour.Popi_popi8 or
                    StackBehaviour.Popi_popr4 or StackBehaviour.Popi_popr8 or StackBehaviour.Popref_pop1 or StackBehaviour.Popref_popi => 2,
                StackBehaviour.Popi_popi_popi or StackBehaviour.Popref_popi_popi or StackBehaviour.Popref_popi_popi8 or
                    StackBehaviour.Popref_popi_popr4 or StackBehaviour.Popref_popi_popr8 or StackBehaviour.Popref_popi_popref => 3,
                _ => 0,
            };
            int pushCount = i.OpCode.StackBehaviourPush switch
            {
                StackBehaviour.Push0 => 0,
                StackBehaviour.Push1_push1 => 2,
                _ => 1,
            };
            return (popCount, pushCount);
        }
    }

    #endregion

    #region Weaving

    /// <summary>
    ///     Inserts the prologue before the function's first instruction: each operand parameter tested with
    ///     <c>NDSubclass.IsOperand</c>, a branch to a call of the subclass path, and a jump over it into the original body.
    /// </summary>
    /// <param name="m">The function.</param>
    /// <param name="sub">Its generated subclass path.</param>
    /// <param name="operands">The operand parameters.</param>
    /// <param name="ctx">The weave.</param>
    /// <param name="rule">The function's mechanism (a <c>[PlainReceiver]</c> member also tests its receiver).</param>
    private static void WeavePrologue(MethodDefinition m, MethodDefinition sub, List<ParameterDefinition> operands, Context ctx, Rule rule)
    {
        var module = m.Module;
        var body = m.Body;
        body.SimplifyMacros();
        var il = body.GetILProcessor();
        var first = body.Instructions[0];
        var isOperand = module.ImportReference(Runtime(ctx, "IsOperand", NDArrayFullName));
        var isOperandMany = module.ImportReference(Runtime(ctx, "IsOperand", NDArrayArrayFullName));

        var subCall = il.Create(OpCodes.Nop);
        // A [PlainReceiver] member's receiver is its first operand (a matrix object seen through an NDArray reference).
        if (rule.Kind == Kind.PlainReceiver && m.HasThis)
        {
            il.InsertBefore(first, il.Create(OpCodes.Ldarg, body.ThisParameter));
            il.InsertBefore(first, il.Create(OpCodes.Call, isOperand));
            il.InsertBefore(first, il.Create(OpCodes.Brtrue, subCall));
        }
        foreach (var p in operands)
        {
            il.InsertBefore(first, il.Create(OpCodes.Ldarg, p));
            if (p.ParameterType.IsByReference)
                il.InsertBefore(first, il.Create(OpCodes.Ldind_Ref));   // an `in NDArray` operand: test the array, not its address
            il.InsertBefore(first, il.Create(OpCodes.Call, IsNDArray(p) ? isOperand : isOperandMany));
            il.InsertBefore(first, il.Create(OpCodes.Brtrue, subCall));
        }
        il.InsertBefore(first, il.Create(OpCodes.Br, first));
        il.InsertBefore(first, subCall);
        if (m.HasThis)
            il.InsertBefore(first, il.Create(OpCodes.Ldarg, body.ThisParameter));
        foreach (var p in m.Parameters)
            il.InsertBefore(first, il.Create(OpCodes.Ldarg, p));
        il.InsertBefore(first, il.Create(OpCodes.Call, sub));
        il.InsertBefore(first, il.Create(OpCodes.Ret));
        body.OptimizeMacros();
    }

    /// <summary>
    ///     Generates <c>&lt;name&gt;subclass</c>: the method dispatch, the entry checks, then the re-call on plain views inside an
    ///     <c>NDScope</c>, the unaliasing and the mechanism (see the class summary).
    /// </summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="m">The function.</param>
    /// <param name="rule">Its resolved mechanism.</param>
    /// <param name="operands">The operand parameters.</param>
    /// <returns>The new method (added to the function's type).</returns>
    /// <exception cref="WeaveException">A result type the mechanism has no overload for, or a dispatch that cannot be bound.</exception>
    private static MethodDefinition GenerateSubclassPath(Context ctx, MethodDefinition m, Rule rule, List<ParameterDefinition> operands)
    {
        var module = m.Module;
        var attrs = MethodAttributes.Private | MethodAttributes.HideBySig | (m.IsStatic ? MethodAttributes.Static : 0);
        var sub = new MethodDefinition("<" + m.Name + (rule.Kind == Kind.PlainReceiver ? ">plain" : ">subclass"), attrs, m.ReturnType);
        var map = new Dictionary<ParameterDefinition, ParameterDefinition>();
        foreach (var p in m.Parameters)
        {
            var q = new ParameterDefinition(p.Name, ParameterAttributes.None, p.ParameterType);
            sub.Parameters.Add(q);
            map[p] = q;
        }
        // [CompilerGenerated], referenced through the module's core library (the reflection importer cannot import a
        // runtime type into a module that targets another framework's reference assemblies).
        var generatedType = new TypeReference("System.Runtime.CompilerServices", "CompilerGeneratedAttribute", module,
                                              module.TypeSystem.CoreLibrary);
        var generatedCtor = new MethodReference(".ctor", module.TypeSystem.Void, generatedType) { HasThis = true };
        sub.CustomAttributes.Add(new CustomAttribute(module.ImportReference(generatedCtor)));

        // Attached to the type first: the body's `this` parameter needs the declaring type. A refused binding below
        // detaches it again, so the type is left untouched.
        m.DeclaringType.Methods.Add(sub);
        try
        {
            BuildSubclassPath(ctx, m, sub, rule, operands, map);
        }
        catch
        {
            m.DeclaringType.Methods.Remove(sub);
            throw;
        }
        return sub;
    }

    /// <summary>The body of <see cref="GenerateSubclassPath"/>.</summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="m">The function.</param>
    /// <param name="sub">The (attached) subclass path, its parameters created.</param>
    /// <param name="rule">The function's mechanism.</param>
    /// <param name="operands">The operand parameters.</param>
    /// <param name="map">The function's parameters to the subclass path's.</param>
    /// <exception cref="WeaveException">A result type the mechanism has no overload for, or a dispatch that cannot be bound.</exception>
    private static void BuildSubclassPath(Context ctx, MethodDefinition m, MethodDefinition sub, Rule rule,
                                          List<ParameterDefinition> operands, Dictionary<ParameterDefinition, ParameterDefinition> map)
    {
        var module = m.Module;
        var body = sub.Body;
        body.InitLocals = true;
        var il = body.GetILProcessor();
        ParameterDefinition P(ParameterDefinition p) => p is null ? null : map[p];

        bool receiver = rule.Kind == Kind.PlainReceiver;

        // The audit hook (the caller of the user's call). A [PlainReceiver] member is NumSharp's static-typing rule, not a
        // subclass answer: library code that calls it on a subclass object is computing plain on purpose.
        if (!receiver)
        {
            il.Append(il.Create(OpCodes.Ldstr, m.Name));
            il.Append(il.Create(OpCodes.Call, module.ImportReference(Runtime(ctx, "AuditEntry", "System.String"))));
        }

        // NumPy's method dispatch first: a subclass's own method gets the call before anything is computed.
        VariableDefinition wrapit = null;
        if (rule.DispatchMethod is not null)
            wrapit = EmitDispatch(ctx, m, sub, rule, map, il);

        // The entry checks (NumPy's errors that come before the computation); a [PlainReceiver] member has none.
        if (!receiver)
            EmitChecks(ctx, m, rule, map, il);

        // NDScope scope = NDScope.Open(); try { aliases; r = m(aliases); unalias; r = mechanism(r); r = scope.Returns(r); }
        // finally { scope.Dispose(); } return r;
        // A void function (np.copyto, np.put) has no result: the subclass path only hands it plain views.
        bool isVoid = m.ReturnType.MetadataType == MetadataType.Void;
        VariableDefinition result = null;
        if (!isVoid)
        {
            result = new VariableDefinition(m.ReturnType);
            body.Variables.Add(result);
        }
        // A result that may hold arrays the scope cannot yield (an iterator over the operand, a broadcast object) gets no
        // scope: the scope would dispose the plain views and intermediates the result still references.
        var egress = isVoid ? Egress.Nothing : EgressOf(ctx, m.ReturnType);
        bool scoped = egress != Egress.Unscoped;
        VariableDefinition scope = null;
        if (scoped)
        {
            scope = new VariableDefinition(module.ImportReference(ctx.Scope));
            body.Variables.Add(scope);
            il.Append(il.Create(OpCodes.Call, module.ImportReference(ctx.Scope.Methods.First(x => x.Name == "Open" && x.IsStatic && !x.HasParameters))));
            il.Append(il.Create(OpCodes.Stloc, scope));
        }

        var aliases = new Dictionary<ParameterDefinition, VariableDefinition>();
        Instruction tryStart = null;
        foreach (var p in operands)
        {
            // An `in NDArray` operand's alias lives in a local of the element type; the re-call passes its address.
            var valueType = p.ParameterType is ByReferenceType byRef ? byRef.ElementType : p.ParameterType;
            var v = new VariableDefinition(valueType);
            body.Variables.Add(v);
            aliases[p] = v;
            var load = il.Create(OpCodes.Ldarg, P(p));
            tryStart ??= load;
            il.Append(load);
            if (p.ParameterType.IsByReference)
                il.Append(il.Create(OpCodes.Ldind_Ref));
            // A [PlainReceiver] member computes as for plain arrays: views WITHOUT the class provenance a computation could
            // react to. Every other path hands the computation views that remember the subclass they stand for.
            string plain = receiver
                ? "PlainView"
                : valueType.FullName == NDArrayArrayFullName
                    ? "Plain"
                    : rule.UfuncInputs && rule.Inputs.Contains(p) ? "PlainUfuncInput" : "Plain";
            il.Append(il.Create(OpCodes.Call, module.ImportReference(Runtime(ctx, plain, valueType.FullName))));
            il.Append(il.Create(OpCodes.Stloc, v));
            if (!scoped)
            {
                // No scope of its own: keep an ambient scope (a scoped caller's) from disposing the view the result holds.
                il.Append(il.Create(OpCodes.Ldloc, v));
                LoadValue(il, P(p));
                il.Append(il.Create(OpCodes.Call, module.ImportReference(Runtime(ctx, "KeepAlias", valueType.FullName, valueType.FullName))));
            }
        }

        // A [PlainReceiver] member's receiver: its plain view (the re-call runs the member on it).
        VariableDefinition receiverAlias = null;
        if (receiver && m.HasThis)
        {
            receiverAlias = new VariableDefinition(module.ImportReference(ctx.NDArray));
            body.Variables.Add(receiverAlias);
            var load = il.Create(OpCodes.Ldarg, body.ThisParameter);
            tryStart ??= load;
            il.Append(load);
            il.Append(il.Create(OpCodes.Call, module.ImportReference(Runtime(ctx, "PlainView", NDArrayFullName))));
            il.Append(il.Create(OpCodes.Stloc, receiverAlias));
            if (!scoped)
            {
                il.Append(il.Create(OpCodes.Ldloc, receiverAlias));
                il.Append(il.Create(OpCodes.Ldarg, body.ThisParameter));
                il.Append(il.Create(OpCodes.Call, module.ImportReference(Runtime(ctx, "KeepAlias", NDArrayFullName, NDArrayFullName))));
            }
        }

        // The re-call: the plain views take the function's plain path.
        if (m.HasThis)
            il.Append(receiverAlias is not null ? il.Create(OpCodes.Ldloc, receiverAlias) : il.Create(OpCodes.Ldarg, body.ThisParameter));
        foreach (var p in m.Parameters)
            il.Append(aliases.TryGetValue(p, out var av)
                ? il.Create(p.ParameterType.IsByReference ? OpCodes.Ldloca : OpCodes.Ldloc, av)
                : il.Create(OpCodes.Ldarg, P(p)));
        il.Append(il.Create(OpCodes.Call, m));
        if (!isVoid)
            il.Append(il.Create(OpCodes.Stloc, result));

        // An operand handed back is the caller's object again: an NDArray operand's view, and an element of a sequence
        // operand's plain copy (np.atleast_1d(m, c) hands each array back when it already has the rank).
        var resultType = m.ReturnType;
        var unalias = isVoid ? null : UnaliasFor(ctx, resultType, NDArrayFullName);
        if (unalias is not null)
            foreach (var p in operands.Where(IsNDArray))
            {
                il.Append(il.Create(OpCodes.Ldloc, result));
                il.Append(il.Create(OpCodes.Ldloc, aliases[p]));
                LoadValue(il, P(p));
                il.Append(il.Create(OpCodes.Call, module.ImportReference(unalias)));
                il.Append(il.Create(OpCodes.Stloc, result));
            }
        if (unalias is not null && receiverAlias is not null)
        {
            il.Append(il.Create(OpCodes.Ldloc, result));
            il.Append(il.Create(OpCodes.Ldloc, receiverAlias));
            il.Append(il.Create(OpCodes.Ldarg, body.ThisParameter));
            il.Append(il.Create(OpCodes.Call, module.ImportReference(unalias)));
            il.Append(il.Create(OpCodes.Stloc, result));
        }
        var unaliasEach = isVoid ? null : UnaliasFor(ctx, resultType, NDArrayArrayFullName);
        if (unaliasEach is not null)
            foreach (var p in operands.Where(x => x.ParameterType.FullName == NDArrayArrayFullName))
            {
                il.Append(il.Create(OpCodes.Ldloc, result));
                il.Append(il.Create(OpCodes.Ldloc, aliases[p]));
                il.Append(il.Create(OpCodes.Ldarg, P(p)));
                il.Append(il.Create(OpCodes.Call, module.ImportReference(unaliasEach)));
                il.Append(il.Create(OpCodes.Stloc, result));
            }

        if (!isVoid)
        {
            // The mechanism (with NumPy's asarray exception per subclass, and _wrapit's wrap after a refused dispatch).
            EmitMechanism(ctx, m, rule, map, result, wrapit, il);

            // scope.Returns(result) for the shapes NDScope yields; a carrier struct yields its arrays itself.
            if (scoped)
                EmitEgress(ctx, resultType, egress, scope, result, il);
        }

        if (!scoped)
        {
            // No scope: the plain result is the answer as it stands.
            il.Append(il.Create(OpCodes.Ldloc, result));
            il.Append(il.Create(OpCodes.Ret));
            body.OptimizeMacros();
            return;
        }

        var leaveTarget = isVoid ? il.Create(OpCodes.Ret) : il.Create(OpCodes.Ldloc, result);
        il.Append(il.Create(OpCodes.Leave, leaveTarget));
        var handlerStart = il.Create(OpCodes.Ldloc, scope);
        il.Append(handlerStart);
        il.Append(il.Create(OpCodes.Callvirt, module.ImportReference(ctx.Scope.Methods.First(x => x.Name == "Dispose" && !x.IsStatic))));
        il.Append(il.Create(OpCodes.Endfinally));
        il.Append(leaveTarget);
        if (!isVoid)
            il.Append(il.Create(OpCodes.Ret));

        tryStart ??= body.Instructions.First(i => i.OpCode.Code == Code.Call && ((MethodReference)i.Operand).Name == m.Name);
        body.ExceptionHandlers.Add(new ExceptionHandler(ExceptionHandlerType.Finally)
        {
            TryStart = tryStart,
            TryEnd = handlerStart,
            HandlerStart = handlerStart,
            HandlerEnd = leaveTarget,
        });
        body.OptimizeMacros();
    }

    /// <summary>
    ///     Emits the mechanism on the result local: the declared rule, the asarray exception of <c>[AsArrayFor]</c>, and
    ///     <c>_wrapit</c>'s wrap when a <c>[WrapFunc]</c> dispatch fell back.
    /// </summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="m">The function.</param>
    /// <param name="rule">The mechanism.</param>
    /// <param name="map">The function's parameters to the subclass path's.</param>
    /// <param name="result">The result local.</param>
    /// <param name="wrapit">The <c>_wrapit</c> flag, or null.</param>
    /// <param name="il">The subclass path's IL.</param>
    /// <exception cref="WeaveException">No runtime overload for the result type.</exception>
    private static void EmitMechanism(Context ctx, MethodDefinition m, Rule rule, Dictionary<ParameterDefinition, ParameterDefinition> map,
                                      VariableDefinition result, VariableDefinition wrapit, ILProcessor il)
    {
        var module = m.Module;
        string rt = m.ReturnType.FullName;
        var done = il.Create(OpCodes.Nop);

        if (wrapit is not null)
        {
            // _wrapit: asarray(obj).method(...) then conv.wrap(result) - the self's __array_wrap__.
            var notWrapit = il.Create(OpCodes.Nop);
            il.Append(il.Create(OpCodes.Ldloc, wrapit));
            il.Append(il.Create(OpCodes.Brfalse, notWrapit));
            if (rt == NDArrayFullName)
            {
                il.Append(il.Create(OpCodes.Ldloc, result));
                il.Append(il.Create(OpCodes.Ldnull));
                LoadValue(il, map[rule.Self]);
                il.Append(il.Create(OpCodes.Ldnull));
                il.Append(il.Create(OpCodes.Ldnull));
                il.Append(il.Create(OpCodes.Call, module.ImportReference(Runtime(ctx, "ArrayWrap", NDArrayFullName, NDArrayFullName,
                    NDArrayFullName, NDArrayFullName, NDArrayFullName))));
                il.Append(il.Create(OpCodes.Stloc, result));
            }
            il.Append(il.Create(OpCodes.Br, done));
            il.Append(notWrapit);
        }

        if (rule.AsArrayKinds != 0)
        {
            // [AsArrayFor]: NumPy's isinstance test converts that class with asarray first.
            var normal = il.Create(OpCodes.Nop);
            LoadValue(il, map[rule.Self]);
            il.Append(il.Create(OpCodes.Ldc_I4, rule.AsArrayKinds));
            il.Append(il.Create(OpCodes.Call, module.ImportReference(Runtime(ctx, "ConvertsWithAsArray", NDArrayFullName, "System.Int32"))));
            il.Append(il.Create(OpCodes.Brfalse, normal));
            EmitCall(ctx, il, module, result, "AsArray", rt, () => { });
            il.Append(il.Create(OpCodes.Br, done));
            il.Append(normal);
        }

        void Load(ParameterDefinition p)
        {
            if (p is null)
                il.Append(il.Create(OpCodes.Ldnull));
            else
                LoadValue(il, map[p]);
        }

        switch (rule.Kind)
        {
            case Kind.ArrayWrap:
                // A ufunc whose out= NumPy makes from one operand wraps by that subset; every other wrap by all inputs.
                var wrapping = rule.WrapInputs ?? rule.Inputs;
                EmitCall(ctx, il, module, result, "ArrayWrap", rt, () =>
                {
                    Load(rule.Out);
                    for (int i = 0; i < 3; i++)
                        Load(i < wrapping.Count ? wrapping[i] : null);
                    if (rt != NDArrayFullName)
                        il.Append(il.Create(OpCodes.Ldc_I4, rule.Outputs));
                });
                break;
            case Kind.NewSubtype:
                if (m.ReturnType.MetadataType == MetadataType.Int64)
                    EmitCall(ctx, il, module, result, "NewSubtype", rt, () => Load(rule.Self));
                else
                    EmitCall(ctx, il, module, result, "NewSubtype", rt, () =>
                    {
                        Load(rule.Out);
                        Load(rule.Self);
                        il.Append(il.Create(rule.KeepRank ? OpCodes.Ldc_I4_1 : OpCodes.Ldc_I4_0));
                        il.Append(il.Create(OpCodes.Ldc_I4, rule.ZeroDim));
                    });
                break;
            case Kind.GetSubType:
                EmitCall(ctx, il, module, result, "GetSubType", rt, () =>
                {
                    Load(rule.Out);
                    Load(rule.Arrays);
                });
                break;
            case Kind.AsArray:
                EmitCall(ctx, il, module, result, "AsArray", rt, () => { });
                break;
            case Kind.ReducesThrough:
                EmitCall(ctx, il, module, result, "Collapse", rt, () =>
                {
                    Load(rule.Out);
                    Load(rule.Self);
                    EmitAxis(ctx, il, m, map);
                    il.Append(il.Create(rule.SubtypeOthers ? OpCodes.Ldc_I4_1 : OpCodes.Ldc_I4_0));
                    il.Append(il.Create(OpCodes.Ldc_I4, rule.ZeroDim));
                });
                break;
            case Kind.Custom when rule.CustomRule is not null:
                if (rule.CustomRule.ReturnType.FullName != m.ReturnType.FullName ||
                    rule.CustomRule.Parameters.Count == 0 || rule.CustomRule.Parameters[0].ParameterType.FullName != m.ReturnType.FullName)
                    throw new WeaveException("NDW025", $"[SubclassRule] {rule.CustomRule.FullName} must take and return '{m.ReturnType.FullName}'");
                il.Append(il.Create(OpCodes.Ldloc, result));
                foreach (var q in rule.CustomRule.Parameters.Skip(1))
                    LoadByName(il, m, map, q, rule.CustomRule);
                il.Append(il.Create(OpCodes.Call, module.ImportReference(rule.CustomRule)));
                il.Append(il.Create(OpCodes.Stloc, result));
                break;
        }
        il.Append(done);
    }

    /// <summary>Emits <c>result = NDSubclass.{name}(result, args...)</c> for the result type's overload.</summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="il">The IL.</param>
    /// <param name="module">The module.</param>
    /// <param name="result">The result local.</param>
    /// <param name="name">The runtime mechanism.</param>
    /// <param name="resultType">The result type's full name.</param>
    /// <param name="loadArgs">Emits the arguments after the result.</param>
    /// <exception cref="WeaveException">No overload takes that result type.</exception>
    private static void EmitCall(Context ctx, ILProcessor il, ModuleDefinition module, VariableDefinition result, string name, string resultType,
                                 Action loadArgs)
    {
        var target = ctx.Dispatcher.Methods.FirstOrDefault(x => x.Name == name && x.IsStatic && x.Parameters.Count > 0 &&
                                                               x.Parameters[0].ParameterType.FullName == resultType &&
                                                               x.ReturnType.FullName == resultType);
        if (target is null)
            throw new WeaveException("NDW025", $"NDSubclass.{name} has no overload for a '{resultType}' result; give the computation a " +
                                               "[SubclassRule] or add the overload");
        il.Append(il.Create(OpCodes.Ldloc, result));
        loadArgs();
        il.Append(il.Create(OpCodes.Call, module.ImportReference(target)));
        il.Append(il.Create(OpCodes.Stloc, result));
    }

    /// <summary>Loads the function's <c>axis</c> as an <c>int?</c> (an <c>int</c> axis converted, none as null).</summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="il">The IL.</param>
    /// <param name="m">The function.</param>
    /// <param name="map">The function's parameters to the subclass path's.</param>
    private static void EmitAxis(Context ctx, ILProcessor il, MethodDefinition m, Dictionary<ParameterDefinition, ParameterDefinition> map)
    {
        var axis = m.Parameters.FirstOrDefault(p => p.Name == "axis");
        var module = m.Module;
        if (axis is not null && axis.ParameterType.FullName == "System.Int32")
        {
            il.Append(il.Create(OpCodes.Ldarg, map[axis]));
            il.Append(il.Create(OpCodes.Call, module.ImportReference(Runtime(ctx, "Axis", "System.Int32"))));
        }
        else if (axis is not null && axis.ParameterType.FullName == "System.Nullable`1<System.Int32>")
            il.Append(il.Create(OpCodes.Ldarg, map[axis]));
        else
            il.Append(il.Create(OpCodes.Call, module.ImportReference(Runtime(ctx, "NoAxis"))));
    }

    /// <summary>
    ///     Loads the function's argument for a rule or check parameter of the same name (converted to <c>int?</c> for an
    ///     <c>int</c> axis, boxed for an <c>object</c> parameter, dereferenced for an <c>in</c> operand); a parameter the
    ///     function lacks gets its declared default (<c>null</c>, a bool or int constant, or <c>default</c> of a value
    ///     type), so one rule serves every overload that reaches its computation.
    /// </summary>
    /// <param name="il">The IL.</param>
    /// <param name="m">The function.</param>
    /// <param name="map">The function's parameters to the subclass path's.</param>
    /// <param name="q">The rule's parameter.</param>
    /// <param name="ruleMethod">The rule (for the error).</param>
    /// <exception cref="WeaveException">A parameter the function lacks and the rule gives no default.</exception>
    private static void LoadByName(ILProcessor il, MethodDefinition m, Dictionary<ParameterDefinition, ParameterDefinition> map,
                                   ParameterDefinition q, MethodDefinition ruleMethod)
    {
        var p = m.Parameters.FirstOrDefault(x => x.Name == q.Name);
        if (p is not null && ValueTypeName(p) == q.ParameterType.FullName)
        {
            LoadValue(il, map[p]);   // an `in` operand is passed as the array it refers to
            return;
        }
        if (p is not null && q.ParameterType.FullName == "System.Object")
        {
            // An `object` rule parameter takes the function's argument of that name whatever its type (np.require's
            // requirements: a string[] in one overload, a string in the other): a reference as it is, a value boxed.
            LoadValue(il, map[p]);
            var pType = p.ParameterType is ByReferenceType byRef ? byRef.ElementType : p.ParameterType;
            if (pType.IsValueType || pType.IsGenericParameter)
                il.Append(il.Create(OpCodes.Box, m.Module.ImportReference(pType)));
            return;
        }
        if (p is not null && p.ParameterType.FullName == "System.Int32" && q.ParameterType.FullName == "System.Nullable`1<System.Int32>")
        {
            // an int argument for an int? parameter (np.std's ddof for DdofWithCorrection's)
            il.Append(il.Create(OpCodes.Ldarg, map[p]));
            il.Append(il.Create(OpCodes.Call, m.Module.ImportReference(
                m.Module.GetType(DispatcherFullName).Methods.First(x => x.Name == "Axis" && x.Parameters.Count == 1))));
            return;
        }
        if (!q.HasDefault && !q.ParameterType.IsValueType || q.HasDefault && q.Constant is null && !q.ParameterType.IsValueType)
        {
            il.Append(il.Create(OpCodes.Ldnull));
            return;
        }
        if (q.ParameterType.FullName == "System.Boolean")
        {
            il.Append(il.Create(q.HasConstant && q.Constant is true ? OpCodes.Ldc_I4_1 : OpCodes.Ldc_I4_0));
            return;
        }
        if (q.ParameterType.FullName == "System.Int32" && q.HasConstant && q.Constant is int constant)
        {
            il.Append(il.Create(OpCodes.Ldc_I4, constant));   // the rule's declared int default
            return;
        }
        if (q.ParameterType.IsValueType && q.HasDefault && q.Constant is null)
        {
            // A value-type parameter declared `= default` (an int? axis, a tuple of outs another overload does not have):
            // default(T) through a zero-initialized temporary, the IL form of `default`.
            var type = m.Module.ImportReference(q.ParameterType);
            var temp = new VariableDefinition(type);
            il.Body.Variables.Add(temp);
            il.Append(il.Create(OpCodes.Ldloca, temp));
            il.Append(il.Create(OpCodes.Initobj, type));
            il.Append(il.Create(OpCodes.Ldloc, temp));
            return;
        }
        throw new WeaveException("NDW026", $"{ruleMethod.FullName}'s parameter '{q.Name}' ({q.ParameterType.Name}) has no parameter of " +
                                           $"that name and type in {m.Name} and no default");
    }

    /// <summary>
    ///     Emits the entry checks: <c>[ReducesThrough]</c>'s keyword error, <c>[NewSubtype(FinalizeFirst = true)]</c>'s
    ///     finalize check, and a <c>[SubclassRule]</c>'s check.
    /// </summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="m">The function.</param>
    /// <param name="rule">The mechanism.</param>
    /// <param name="map">The function's parameters to the subclass path's.</param>
    /// <param name="il">The IL.</param>
    private static void EmitChecks(Context ctx, MethodDefinition m, Rule rule, Dictionary<ParameterDefinition, ParameterDefinition> map, ILProcessor il)
    {
        var module = m.Module;
        if (rule.Kind == Kind.ReducesThrough)
        {
            LoadValue(il, map[rule.Self]);
            var keepdims = m.Parameters.FirstOrDefault(p => p.Name == "keepdims" && p.ParameterType.FullName == "System.Boolean");
            il.Append(keepdims is null ? il.Create(OpCodes.Ldc_I4_0) : il.Create(OpCodes.Ldarg, map[keepdims]));
            il.Append(rule.ReducesMethod is null ? il.Create(OpCodes.Ldnull) : il.Create(OpCodes.Ldstr, rule.ReducesMethod));
            il.Append(rule.InexactMethod is null ? il.Create(OpCodes.Ldnull) : il.Create(OpCodes.Ldstr, rule.InexactMethod));
            il.Append(il.Create(OpCodes.Call, module.ImportReference(Runtime(ctx, "ReducesThroughCheck", NDArrayFullName, "System.Boolean",
                "System.String", "System.String"))));
        }
        if (rule.Kind == Kind.NewSubtype && rule.FinalizeFirst)
        {
            LoadValue(il, map[rule.Self]);
            var dtype = m.Parameters.FirstOrDefault(p => p.Name == "dtype" && p.ParameterType.FullName == "NumSharp.DType");
            il.Append(dtype is null ? il.Create(OpCodes.Ldnull) : il.Create(OpCodes.Ldarg, map[dtype]));
            il.Append(il.Create(OpCodes.Call, module.ImportReference(Runtime(ctx, "FinalizeFirst", NDArrayFullName, "NumSharp.DType"))));
        }
        if (rule.CustomCheck is not null)
        {
            foreach (var q in rule.CustomCheck.Parameters)
                LoadByName(il, m, map, q, rule.CustomCheck);
            il.Append(il.Create(OpCodes.Call, module.ImportReference(rule.CustomCheck)));
            if (rule.CustomCheck.ReturnType.MetadataType != MetadataType.Void)
                il.Append(il.Create(OpCodes.Pop));
        }
    }

    /// <summary>
    ///     GENERATES NumPy's method dispatch at the subclass path's entry, one arm per subclass with a <c>[NumPyMethod]</c>
    ///     member of the method's name:
    ///     <code>
    ///     if (a is NDMatrix __sub)
    ///     {
    ///         if (keepdims) throw new TypeError("matrix.sum() got an unexpected keyword argument 'keepdims'");   // _wrapreduction
    ///         // or, for _wrapfunc:  if (keepdims) { wrapit = true; goto next; }                                 // _wrapit
    ///         return __sub.sum(axis, dtype, @out);              // the arguments mapped by NAME
    ///     }
    ///     next:
    ///     </code>
    /// </summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="m">The function.</param>
    /// <param name="sub">The subclass path.</param>
    /// <param name="rule">The mechanism (its dispatch fields).</param>
    /// <param name="map">The function's parameters to the subclass path's.</param>
    /// <param name="il">The IL.</param>
    /// <returns>The <c>_wrapit</c> flag local for a <c>[WrapFunc]</c>, else null.</returns>
    /// <exception cref="WeaveException">A method parameter without a function parameter of that name and type.</exception>
    private static VariableDefinition EmitDispatch(Context ctx, MethodDefinition m, MethodDefinition sub, Rule rule,
                                                   Dictionary<ParameterDefinition, ParameterDefinition> map, ILProcessor il)
    {
        var module = m.Module;
        var body = sub.Body;
        VariableDefinition wrapit = null;
        if (!rule.DispatchIsReduction)
        {
            wrapit = new VariableDefinition(module.TypeSystem.Boolean);
            body.Variables.Add(wrapit);
        }
        var typeError = module.ImportReference(module.GetType("NumSharp.TypeError").Methods.First(x =>
            x.IsConstructor && x.Parameters.Count == 1 && x.Parameters[0].ParameterType.FullName == "System.String"));

        foreach (var subclass in ctx.Subclasses)
        {
            var target = subclass.Type.Methods.FirstOrDefault(x => !x.IsStatic && x.Name == rule.DispatchMethod &&
                                                                   x.CustomAttributes.Any(a => a.AttributeType.FullName == NumPyMethodAttr));
            if (target is null)
                continue;   // the subclass inherits ndarray's method: the computation and its rule answer

            // A function returning a C# number (np.argmax(a) -> long) has no arm: the subclass method's answer is an array,
            // and NumPy's scalar of it is what the function computes plainly anyway.
            if (target.ReturnType.FullName != m.ReturnType.FullName && (m.ReturnType.IsValueType || m.ReturnType.FullName != NDArrayFullName))
                continue;

            // NumPy passes keywords: each method parameter takes the function's parameter of the same name; one the function
            // does not have is NumPy's default (np.argmax(a, axis) passes out=None).
            var args = new List<(ParameterDefinition Param, bool ToNullable)>();
            var computed = new Dictionary<int, MethodDefinition>();
            var consumed = new HashSet<string>(StringComparer.Ordinal);
            bool bindable = true;
            foreach (var q in target.Parameters)
            {
                if (rule.WrapArguments.TryGetValue(q.Name, out var compute))
                {
                    // NumPy's function computes this argument first (np.std folds correction= into ddof).
                    computed[args.Count] = compute;
                    args.Add((null, false));
                    foreach (var cp in compute.Parameters)
                        consumed.Add(cp.Name);
                    continue;
                }
                var p = m.Parameters.FirstOrDefault(x => x.Name == q.Name);
                if (p is null && !q.ParameterType.IsValueType)
                {
                    args.Add((null, false));
                    continue;
                }
                if (p is null)
                    throw new WeaveException("NDW027", $"{subclass.PyName}.{rule.DispatchMethod}'s parameter '{q.Name}' has no parameter of " +
                                                       $"that name in {m.Name}");
                bool toNullable = p.ParameterType.FullName == "System.Int32" &&
                                  q.ParameterType.FullName == "System.Nullable`1<System.Int32>";
                if (!toNullable && ValueTypeName(p) != q.ParameterType.FullName)
                {
                    // An overload whose argument the method cannot take (a tuple axis for matrix.all's int axis): no arm,
                    // the computation's other mechanism (or the plain answer) shapes this overload's result.
                    bindable = false;
                    break;
                }
                args.Add((p, toNullable));
            }
            if (!bindable)
                continue;
            var missing = m.Parameters.Where(p => p != rule.Self && target.Parameters.All(q => q.Name != p.Name) &&
                                                  !consumed.Contains(p.Name)).ToList();

            var next = il.Create(OpCodes.Nop);
            var subVar = new VariableDefinition(subclass.Type);
            body.Variables.Add(subVar);
            LoadValue(il, map[rule.Self]);
            il.Append(il.Create(OpCodes.Isinst, subclass.Type));   // self is NDMatrix __sub
            il.Append(il.Create(OpCodes.Stloc, subVar));
            il.Append(il.Create(OpCodes.Ldloc, subVar));
            il.Append(il.Create(OpCodes.Brfalse, next));
            foreach (var p in missing)
            {
                // _wrapreduction passes axis / out always and dtype when not None; _wrapfunc every parameter outside WhenGiven.
                bool always = rule.DispatchIsReduction ? p.Name is "axis" or "out" : !rule.WhenGiven.Contains(p.Name);
                var skip = il.Create(OpCodes.Nop);
                if (!always && !EmitIsDefault(il, map[p], skip, module))
                    throw new WeaveException("NDW028", $"parameter '{p.Name}' of {m.Name} has no default the dispatch can compare");
                if (rule.DispatchIsReduction)
                {
                    il.Append(il.Create(OpCodes.Ldstr, $"{subclass.PyName}.{rule.DispatchMethod}() got an unexpected keyword argument '{p.Name}'"));
                    il.Append(il.Create(OpCodes.Newobj, typeError));
                    il.Append(il.Create(OpCodes.Throw));
                }
                else
                {
                    // The bound method raises TypeError: _wrapit computes on the plain array and wraps.
                    il.Append(il.Create(OpCodes.Ldc_I4_1));
                    il.Append(il.Create(OpCodes.Stloc, wrapit));
                    il.Append(il.Create(OpCodes.Br, next));
                }
                il.Append(skip);
            }
            il.Append(il.Create(OpCodes.Ldloc, subVar));
            for (int ai = 0; ai < args.Count; ai++)
            {
                var (p, toNullable) = args[ai];
                if (computed.TryGetValue(ai, out var compute))
                {
                    foreach (var cp in compute.Parameters)
                        LoadByName(il, m, map, cp, compute);
                    il.Append(il.Create(OpCodes.Call, module.ImportReference(compute)));
                    continue;
                }
                if (p is null)
                {
                    il.Append(il.Create(OpCodes.Ldnull));
                    continue;
                }
                LoadValue(il, map[p]);
                if (toNullable)
                    il.Append(il.Create(OpCodes.Call, module.ImportReference(Runtime(ctx, "Axis", "System.Int32"))));
            }
            il.Append(il.Create(OpCodes.Callvirt, module.ImportReference(target)));
            if (target.ReturnType.FullName != m.ReturnType.FullName)
            {
                if (m.ReturnType.FullName == NDArrayFullName && !target.ReturnType.IsValueType)
                    il.Append(il.Create(OpCodes.Castclass, module.ImportReference(m.ReturnType)));
                else
                    throw new WeaveException("NDW027", $"{subclass.PyName}.{rule.DispatchMethod} returns {target.ReturnType.Name}, {m.Name} " +
                                                       $"returns {m.ReturnType.Name}");
            }
            il.Append(il.Create(OpCodes.Ret));   // the subclass method's answer is final
            il.Append(next);
        }
        return wrapit;
    }

    /// <summary>
    ///     Emits a branch to <paramref name="skip"/> taken when <paramref name="p"/> holds its declared default (NumPy's "not
    ///     passed": C# cannot tell an omitted argument from its default value).
    /// </summary>
    /// <param name="il">The IL.</param>
    /// <param name="p">The subclass path's parameter (carries no default: the function's is read).</param>
    /// <param name="skip">Where to go when it holds its default.</param>
    /// <param name="module">The module.</param>
    /// <returns>False when the parameter's type is not one the comparison supports.</returns>
    private static bool EmitIsDefault(ILProcessor il, ParameterDefinition p, Instruction skip, ModuleDefinition module)
    {
        var t = p.ParameterType;
        if (t is ByReferenceType byRef && !byRef.ElementType.IsValueType)
        {
            // an `in` reference: its default is null, read through the address
            il.Append(il.Create(OpCodes.Ldarg, p));
            il.Append(il.Create(OpCodes.Ldind_Ref));
            il.Append(il.Create(OpCodes.Brfalse, skip));
            return true;
        }
        if (t.IsByReference)
            return false;   // a by-reference value type: no comparison is emitted for it
        if (t.FullName == "System.Boolean")
        {
            il.Append(il.Create(OpCodes.Ldarg, p));
            il.Append(il.Create(OpCodes.Brfalse, skip));
            return true;
        }
        if (!t.IsValueType)
        {
            il.Append(il.Create(OpCodes.Ldarg, p));
            il.Append(il.Create(OpCodes.Brfalse, skip));
            return true;
        }
        if (t is GenericInstanceType g && g.ElementType.FullName == "System.Nullable`1")
        {
            var hasValue = new MethodReference("get_HasValue", module.TypeSystem.Boolean, g) { HasThis = true };
            il.Append(il.Create(OpCodes.Ldarga, p));
            il.Append(il.Create(OpCodes.Call, module.ImportReference(hasValue)));
            il.Append(il.Create(OpCodes.Brfalse, skip));
            return true;
        }
        return false;
    }

    /// <summary>How the subclass path hands a result type out of its scope.</summary>
    private enum Egress
    {
        /// <summary>Holds no array (a number, a bool, a string, a DType, a byte[]): nothing to yield.</summary>
        Nothing,

        /// <summary>An NDArray-derived object, a one-dimensional array of them, or a ValueTuple:
        ///     <c>scope.Returns</c>.</summary>
        Yield,

        /// <summary>An <c>INDArrayCarrier</c> struct: its <c>YieldTo(scope)</c>.</summary>
        Carrier,

        /// <summary>
        ///     Anything else (an iterator over the operand, a broadcast object, a struct that is no carrier): it may hold
        ///     arrays the scope cannot yield, so the path runs without a scope (see <see cref="BuildSubclassPath"/>).
        /// </summary>
        Unscoped,
    }

    /// <summary>The egress kind of a woven function's result type.</summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="t">The function's return type (not void).</param>
    /// <returns>How the subclass path hands the result out.</returns>
    private static Egress EgressOf(Context ctx, TypeReference t)
    {
        if (t.IsPrimitive || t.MetadataType == MetadataType.String)
            return Egress.Nothing;
        if (t is ArrayType array)
        {
            if (array.Rank == 1 && DerivesFromNDArray(array.ElementType))
                return Egress.Yield;
            // An array of numbers (np.save's byte[]) holds no NDArray; any other array may.
            return array.ElementType.IsPrimitive ? Egress.Nothing : Egress.Unscoped;
        }
        if (DerivesFromNDArray(t))
            return Egress.Yield;
        if (t is GenericInstanceType tuple && tuple.ElementType.FullName.StartsWith("System.ValueTuple`"))
            return Egress.Yield;   // NDScope.Returns(ITuple) yields whatever NDArray each component carries
        var resolved = t.Resolve();
        if (resolved is null)
            return Egress.Unscoped;
        if (resolved.IsEnum)
            return Egress.Nothing;
        if (resolved.IsValueType && resolved.Interfaces.Any(i => i.InterfaceType.FullName == ctx.Carrier.FullName))
            return Egress.Carrier;
        // The number structs and the dtype descriptors hold no array.
        if (resolved.FullName is "System.Decimal" or "System.Half" or "System.Numerics.Complex" or "System.Numerics.BigInteger" or
            "NumSharp.DType" or "System.Type")
            return Egress.Nothing;
        return Egress.Unscoped;
    }

    /// <summary>Whether a type is <c>NDArray</c> or derives from it (<c>NDArray&lt;T&gt;</c>, <c>NDMatrix</c>,
    ///     ...).</summary>
    /// <param name="t">The type.</param>
    /// <returns>True for an array type of the library.</returns>
    private static bool DerivesFromNDArray(TypeReference t)
    {
        for (var d = t.Resolve(); d is not null; d = d.BaseType?.Resolve())
            if (d.FullName == NDArrayFullName)
                return true;
        return false;
    }

    /// <summary>
    ///     Emits the subclass path's egress: <c>result = scope.Returns&lt;T&gt;(result)</c> for an NDArray-derived result
    ///     (<c>T</c> its own type, so an <c>NDArray&lt;bool&gt;</c> or an <c>NDMatrix</c> is yielded, never disposed), the
    ///     array overload for a one-dimensional array of them, <c>Returns(ITuple)</c> over the boxed tuple for a ValueTuple
    ///     (whatever arrays its components carry), or <c>result.YieldTo(scope)</c> for a carrier struct; a result holding no
    ///     array needs nothing.
    /// </summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="resultType">The function's return type.</param>
    /// <param name="egress">Its egress kind (<see cref="EgressOf"/>; never <see cref="Egress.Unscoped"/> here).</param>
    /// <param name="scope">The scope local.</param>
    /// <param name="result">The result local.</param>
    /// <param name="il">The IL.</param>
    private static void EmitEgress(Context ctx, TypeReference resultType, Egress egress, VariableDefinition scope, VariableDefinition result,
                                   ILProcessor il)
    {
        var module = il.Body.Method.Module;
        if (egress == Egress.Nothing)
            return;
        if (egress == Egress.Carrier)
        {
            il.Append(il.Create(OpCodes.Ldloca, result));
            il.Append(il.Create(OpCodes.Ldloc, scope));
            il.Append(il.Create(OpCodes.Constrained, resultType));
            il.Append(il.Create(OpCodes.Callvirt, module.ImportReference(ctx.Carrier.Methods.First(x => x.Name == "YieldTo"))));
            return;
        }
        if (resultType is GenericInstanceType tuple && tuple.ElementType.FullName.StartsWith("System.ValueTuple`"))
        {
            // scope.Returns((ITuple)(object)result): the general tuple egress (every arity, any component mix).
            var returnsTuple = ctx.Scope.Methods.First(x => x.Name == "Returns" && !x.HasGenericParameters && x.Parameters.Count == 1 &&
                                                            x.Parameters[0].ParameterType.FullName == "System.Runtime.CompilerServices.ITuple");
            il.Append(il.Create(OpCodes.Ldloc, scope));
            il.Append(il.Create(OpCodes.Ldloc, result));
            il.Append(il.Create(OpCodes.Box, module.ImportReference(resultType)));
            il.Append(il.Create(OpCodes.Callvirt, module.ImportReference(returnsTuple)));
            il.Append(il.Create(OpCodes.Pop));   // the tuple's arrays are re-parented in place; the value is unchanged
            return;
        }
        GenericInstanceMethod returns;
        if (resultType is ArrayType array)
        {
            returns = new GenericInstanceMethod(module.ImportReference(ctx.Scope.Methods.First(x =>
                x.Name == "Returns" && x.GenericParameters.Count == 1 && x.Parameters[0].ParameterType is ArrayType)));
            returns.GenericArguments.Add(module.ImportReference(array.ElementType));
        }
        else
        {
            returns = new GenericInstanceMethod(module.ImportReference(ctx.Scope.Methods.First(x =>
                x.Name == "Returns" && x.GenericParameters.Count == 1 && x.Parameters[0].ParameterType is not ArrayType)));
            returns.GenericArguments.Add(module.ImportReference(resultType));
        }
        il.Append(il.Create(OpCodes.Ldloc, scope));
        il.Append(il.Create(OpCodes.Ldloc, result));
        il.Append(il.Create(OpCodes.Callvirt, returns));
        il.Append(il.Create(OpCodes.Stloc, result));
    }

    /// <summary>
    ///     The <c>NDSubclass.Unalias</c> overload for a result type and an operand kind, or null when the result type holds no
    ///     operand of that kind (a number, a struct the mechanism rebuilds).
    /// </summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="resultType">The function's return type.</param>
    /// <param name="aliasType">The operand's type: <c>NumSharp.NDArray</c> (one plain view) or <c>NumSharp.NDArray[]</c> (a
    ///     sequence's plain copy, unaliased element by element).</param>
    /// <returns>The overload, or null.</returns>
    private static MethodDefinition UnaliasFor(Context ctx, TypeReference resultType, string aliasType)
        => ctx.Dispatcher.Methods.FirstOrDefault(x => x.Name == "Unalias" && x.Parameters.Count == 3 &&
                                                     x.Parameters[0].ParameterType.FullName == resultType.FullName &&
                                                     x.Parameters[1].ParameterType.FullName == aliasType);

    /// <summary>A runtime method of <c>NDSubclass</c> by name and parameter types.</summary>
    /// <param name="ctx">The weave.</param>
    /// <param name="name">The method's name.</param>
    /// <param name="parameterTypes">The full names of its parameter types.</param>
    /// <returns>The method.</returns>
    /// <exception cref="WeaveException">The runtime has no such method (weaver and library out of sync).</exception>
    private static MethodDefinition Runtime(Context ctx, string name, params string[] parameterTypes)
        => ctx.Dispatcher.Methods.FirstOrDefault(x => x.Name == name && x.Parameters.Count == parameterTypes.Length &&
                                                     x.Parameters.Select(p => p.ParameterType.FullName).SequenceEqual(parameterTypes))
           ?? throw new WeaveException("NDW029", $"NumSharp.NDSubclass.{name}({string.Join(", ", parameterTypes)}) is missing - weaver " +
                                                 "and library out of sync");

    #endregion

    #region Helpers

    /// <summary>The string constructor arguments of an attribute (a <c>params string[]</c> or single strings).</summary>
    /// <param name="attr">The attribute.</param>
    /// <returns>The strings in order.</returns>
    private static List<string> ConstructorStrings(CustomAttribute attr)
    {
        var list = new List<string>();
        foreach (var arg in attr.ConstructorArguments)
        {
            if (arg.Value is string s)
                list.Add(s);
            else if (arg.Value is CustomAttributeArgument[] many)
                list.AddRange(many.Select(x => (string)x.Value));
        }
        return list;
    }

    /// <summary>
    ///     Whether this pass already wove <paramref name="module"/>: a woven-module owner holds a generated subclass path
    ///     (<c>&lt;name&gt;subclass</c>) or a <c>[PlainReceiver]</c> member's plain path (<c>&lt;name&gt;plain</c>),
    ///     names the
    ///     C# compiler never produces (an angle bracket is not an identifier character).
    /// </summary>
    /// <param name="module">The module about to be woven.</param>
    /// <returns>True when a generated path is present.</returns>
    private static bool IsAlreadyWoven(ModuleDefinition module)
    {
        foreach (var type in AllTypes(module))
            foreach (var m in type.Methods)
                if (m.Name.Length > 2 && m.Name[0] == '<' && (m.Name.EndsWith(">subclass", StringComparison.Ordinal) ||
                                                             m.Name.EndsWith(">plain", StringComparison.Ordinal)))
                    return true;
        return false;
    }

    /// <summary>Every type of the module, nested ones included.</summary>
    /// <param name="module">The module.</param>
    /// <returns>The types.</returns>
    private static IEnumerable<TypeDefinition> AllTypes(ModuleDefinition module)
    {
        var stack = new Stack<TypeDefinition>(module.Types);
        while (stack.Count > 0)
        {
            var t = stack.Pop();
            yield return t;
            foreach (var n in t.NestedTypes)
                stack.Push(n);
        }
    }

    /// <summary>A method's signature for the report: name and parameter types.</summary>
    /// <param name="m">The method.</param>
    /// <returns>The text.</returns>
    private static string Signature(MethodDefinition m)
        => $"{m.Name}({string.Join(", ", m.Parameters.Select(p => p.ParameterType.Name + " " + p.Name))})";

    /// <summary>Writes one build error.</summary>
    /// <param name="ctx">The weave (its error count is raised).</param>
    /// <param name="code">The diagnostic id.</param>
    /// <param name="m">The method.</param>
    /// <param name="text">What is wrong.</param>
    private static void Fail(Context ctx, string code, MethodDefinition m, string text)
    {
        ctx.Stderr.WriteLine($"NumSharp.Build : error {code}: '{m.FullName}': {text}");
        ctx.Errors++;
    }

    /// <summary>A refused declaration or binding, with its diagnostic id.</summary>
    private sealed class WeaveException : Exception
    {
        /// <summary>Creates the exception.</summary>
        /// <param name="code">The diagnostic id (NDW022..NDW029, NDW031).</param>
        /// <param name="message">What is wrong.</param>
        public WeaveException(string code, string message) : base(message) => Code = code;

        /// <summary>The diagnostic id.</summary>
        public string Code { get; }
    }

    #endregion
}
