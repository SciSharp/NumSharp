using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace NumSharp.Build.Analyzer
{
    /// <summary>
    ///     NDW020 — NumSharp's own rule that every exception text is culture-independent, as NumPy's texts are (parity
    ///     QA-41): a number formatted without a format provider takes the CURRENT culture, so
    ///     <c>$"index {i} is out of bounds"</c> reads <c>index ؜-6</c> under ar-SA (U+061C before the hyphen), <c>−6</c>
    ///     under sv-SE (U+2212), <c>1,5</c> under de-DE and <c>∞</c> for an infinity under en-US.
    ///     <para>
    ///     OFF by default (<c>isEnabledByDefault: false</c>): the analyzer ships inside the NumSharp package, and a
    ///     consumer's exception texts are not NumSharp's business. NumSharp.Core turns it on as an ERROR in its own
    ///     <c>src/NumSharp.Core/.editorconfig</c>, so a new culture-dependent text fails NumSharp's build instead of
    ///     regressing silently.
    ///     </para>
    ///     <para>
    ///     What counts as an exception message: every <c>string</c> argument of an exception's constructor, whether
    ///     created (<c>new X(...)</c>, <c>throw new</c>, a factory's <c>return new</c>) or chained (<c>: base(...)</c>);
    ///     the locals such an argument reads, followed to their assignments and <c>StringBuilder</c> appends in the same
    ///     body; and the returned text of every method, local function or property getter OF THIS COMPILATION that a
    ///     message calls (or whose type's <c>ToString()</c> override a message formats), transitively. The last part is
    ///     only known when the whole compilation has been seen, so it is reported from the compilation-end action under
    ///     a second descriptor (<see cref="Nw020Reached"/>) that names the method the text came through.
    ///     </para>
    ///     <para>
    ///     What is flagged inside such a text (<see cref="Scanner.IsCultureSensitive"/>): a number, <c>object</c>,
    ///     <c>dynamic</c>, <c>ValueType</c> or type parameter (any of which may be a number at run time) formatted with no
    ///     provider — an interpolation hole, a <c>+</c> operand, <c>ToString()</c>, <c>Convert.ToString</c>,
    ///     <c>string.Format</c>, <c>string.Join</c> / <c>string.Concat</c>, <c>StringBuilder.Append</c> /
    ///     <c>AppendFormat</c> / <c>AppendJoin</c> / <c>Insert</c>. An interpolated string handed to a handler that takes
    ///     a provider (<c>string.Create(CultureInfo.InvariantCulture, $"...")</c>, <c>sb.Append(provider, $"...")</c>) or
    ///     converted to <c>FormattableString</c> (<c>FormattableString.Invariant</c>) chose its culture, so it is not
    ///     flagged; its holes are still scanned (a nested <c>$"..."</c> does not inherit the outer provider). A number
    ///     passed as <see cref="System.ArgumentOutOfRangeException"/>'s <c>actualValue</c> is flagged too: the BCL appends
    ///     <c>Actual value was ….</c> to the message, formatted with the current culture (NumSharp's <c>AxisError</c>
    ///     passes the axis there).
    ///     </para>
    ///     <para>
    ///     Not followed (documented gaps, not silently passed): a message that reaches the constructor through a
    ///     PARAMETER (a throw helper's <c>string message</c>), and text built in a field or another type's state.
    ///     </para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class InvariantExceptionTextAnalyzer : DiagnosticAnalyzer
    {
        private const string Category = "NumSharp.Build";
        private const string HelpLink = "https://scisharp.github.io/NumSharp/docs/exceptions.html#culture-independent-texts";
        private const string Title = "An exception message formats a number with the current culture";

        /// <summary>
        ///     The fixes both texts end with, best first. The order matters to an agent reading the build line: the
        ///     interpolated-string wrap is the mechanical fix for almost every site, and the last sentence blocks the
        ///     tempting wrong fix (setting the thread culture changes every other text the process prints).
        /// </summary>
        private const string Fixes =
            "Take the first fix that fits: (1) an interpolated string: wrap it as string.Create(CultureInfo.InvariantCulture, $\"...\") " +
            "(a + chain of interpolated strings as one); (2) ToString, Convert.ToString, string.Format or StringBuilder.AppendFormat: " +
            "pass CultureInfo.InvariantCulture; (3) a + operand, string.Join, StringBuilder.Append or an ArgumentOutOfRangeException " +
            "actualValue of a number: format the number " +
            "with ToString(CultureInfo.InvariantCulture). Never change CultureInfo.CurrentCulture to fix it.";

        private const string Why =
            "so the text changes with the culture (de-DE '1,5', sv-SE U+2212 and ar-SA U+061C minus signs, en-US '∞') while NumPy's texts never do. ";

        private const string Description =
            "NumSharp's own build rule, off by default (NumSharp.Core turns it on as an error in its .editorconfig): every exception " +
            "text is culture-independent, as NumPy's are. Checked: the string arguments of an exception's constructor (new, or a " +
            ": base(...) call), the locals they read (their assignments and StringBuilder appends in the same body), and the returned " +
            "text of any method, local function or property getter of the compilation that such an argument calls, or whose type's " +
            "ToString override it formats, transitively. Flagged: a number, object, dynamic, ValueType or type parameter formatted " +
            "without a format provider: an interpolation hole, a + operand, ToString(), Convert.ToString, string.Format, " +
            "string.Join / Concat, StringBuilder.Append / AppendFormat / AppendJoin / Insert, and a number passed as an " +
            "ArgumentOutOfRangeException's actualValue (its Message appends it with the current culture). An interpolated string " +
            "handed to a handler that takes a provider, or converted to FormattableString, chose its culture. Not followed: a " +
            "message passed in through a parameter, and text built in a field.";

        /// <summary>
        ///     Builds an NDW020 descriptor: an Error when enabled, off by default, sharing the title, description and help
        ///     link. The id comes FIRST because the release-tracking analyzer reads the rule id from a descriptor
        ///     factory's first argument (RS2002 otherwise); <c>AnalyzerReleases.Unshipped.md</c> lists NDW020 as
        ///     <c>Disabled</c>, the tracked severity of a rule that is off by default.
        /// </summary>
        /// <param name="id">The rule id (always <c>NDW020</c>).</param>
        /// <param name="message">The message format; <c>{0}</c> is the construct in words.</param>
        /// <param name="customTags">Extra tags (<see cref="WellKnownDiagnosticTags.CompilationEnd"/> for the end-of-compilation text).</param>
        /// <returns>The descriptor.</returns>
        private static DiagnosticDescriptor Rule(string id, string message, params string[] customTags) =>
            new DiagnosticDescriptor(id, Title, message, Category, DiagnosticSeverity.Error,
                isEnabledByDefault: false, description: Description, helpLinkUri: HelpLink, customTags: customTags);

        /// <summary>
        ///     NDW020 at a construct that formats a number inside an exception constructor's argument (or a local it
        ///     reads): reported as soon as the constructor call is analyzed, so it shows live in an IDE.
        /// </summary>
        internal static readonly DiagnosticDescriptor Nw020 = Rule("NDW020",
            "{0} formats a number with the current culture inside an exception message, " + Why + Fixes);

        /// <summary>
        ///     NDW020 at a construct in a method whose returned text reaches an exception message. Known only once every
        ///     message and every method of the compilation has been seen, so it is reported from the compilation-end
        ///     action, which is what the <see cref="WellKnownDiagnosticTags.CompilationEnd"/> tag declares (RS1037): an
        ///     IDE shows it after a full-solution analysis or a build, not as the code is typed.
        /// </summary>
        internal static readonly DiagnosticDescriptor Nw020Reached = Rule("NDW020",
            "{0} formats a number with the current culture in '{1}', whose text reaches an exception message, " + Why + Fixes,
            WellKnownDiagnosticTags.CompilationEnd);

        /// <inheritdoc />
        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => ImmutableArray.Create(Nw020, Nw020Reached);

        /// <summary>
        ///     Registers the three operation callbacks that collect messages and message producers, and the
        ///     compilation-end callback that reports what the producers reached.
        /// </summary>
        /// <param name="context">The analysis context Roslyn hands every analyzer once.</param>
        public override void Initialize(AnalysisContext context)
        {
            context.EnableConcurrentExecution();
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.RegisterCompilationStartAction(start =>
            {
                var scanner = Scanner.Create(start.Compilation);
                // A compilation without System.Exception (a broken or core-less reference set) has no exception
                // constructors to check.
                if (scanner is null)
                    return;
                var state = new CompilationState();
                start.RegisterOperationAction(ctx => OnObjectCreation(ctx, scanner, state), OperationKind.ObjectCreation);
                start.RegisterOperationAction(ctx => OnInvocation(ctx, scanner, state), OperationKind.Invocation);
                start.RegisterOperationAction(ctx => OnReturn(ctx, scanner, state), OperationKind.Return);
                start.RegisterCompilationEndAction(ctx => OnCompilationEnd(ctx, state));
            });
        }

        /// <summary>
        ///     <c>new X(...)</c> of an exception type: scans each string argument as an exception message.
        /// </summary>
        /// <param name="context">The callback's context (the creation, its containing symbol, the reporter).</param>
        /// <param name="scanner">The compilation's scanner.</param>
        /// <param name="state">The compilation's shared collections.</param>
        private static void OnObjectCreation(OperationAnalysisContext context, Scanner scanner, CompilationState state)
        {
            var creation = (IObjectCreationOperation)context.Operation;
            if (!scanner.IsException(creation.Type))
                return;
            ScanMessageArguments(context, creation.Arguments, scanner, state);
        }

        /// <summary>
        ///     A constructor CALL that is not a creation: a <c>: base(...)</c> / <c>: this(...)</c> initializer of an
        ///     exception type, whose string arguments become the message of every instance built through it.
        /// </summary>
        /// <param name="context">The callback's context.</param>
        /// <param name="scanner">The compilation's scanner.</param>
        /// <param name="state">The compilation's shared collections.</param>
        private static void OnInvocation(OperationAnalysisContext context, Scanner scanner, CompilationState state)
        {
            var invocation = (IInvocationOperation)context.Operation;
            // Every invocation in the compilation passes through here, so the cheap kind test comes first.
            if (invocation.TargetMethod.MethodKind != MethodKind.Constructor || !scanner.IsException(invocation.TargetMethod.ContainingType))
                return;
            ScanMessageArguments(context, invocation.Arguments, scanner, state);
        }

        /// <summary>
        ///     Scans the <c>string</c>-typed arguments of an exception constructor, reports what they format with the
        ///     current culture, and records the source methods they call as roots for the compilation-end pass.
        /// </summary>
        /// <param name="context">The callback's context (the reporter).</param>
        /// <param name="arguments">The constructor's arguments.</param>
        /// <param name="scanner">The compilation's scanner.</param>
        /// <param name="state">The compilation's shared collections.</param>
        private static void ScanMessageArguments(OperationAnalysisContext context, ImmutableArray<IArgumentOperation> arguments,
            Scanner scanner, CompilationState state)
        {
            foreach (var argument in arguments)
            {
                if (argument.Value is null || argument.Parameter is null)
                    continue;
                // ArgumentOutOfRangeException appends "Actual value was {actualValue}." to its Message, formatting the
                // value with the current culture (AxisError passes the axis there): a number handed over as is
                // makes the text culture-dependent just as an interpolation would.
                if (scanner.IsActualValueParameter(argument.Parameter))
                {
                    var value = Scanner.StripImplicit(argument.Value);
                    if (scanner.IsCultureSensitive(value.Type) && state.Reported.TryAdd(value.Syntax.GetLocation(), 0))
                        context.ReportDiagnostic(Diagnostic.Create(Nw020, value.Syntax.GetLocation(),
                            "the actualValue " + Scanner.Quote(value) + " (" + Scanner.Display(value.Type) + "), which ArgumentOutOfRangeException appends to its text,"));
                    continue;
                }
                // Only text parameters carry a message (message, paramName).
                if (argument.Parameter.Type.SpecialType != SpecialType.System_String)
                    continue;
                var result = new ScanResult();
                scanner.Scan(argument.Value, result);
                foreach (var finding in result.Findings)
                    if (state.Reported.TryAdd(finding.Location, 0))
                        context.ReportDiagnostic(Diagnostic.Create(Nw020, finding.Location, finding.What));
                foreach (var callee in result.Edges)
                    state.Roots.TryAdd(callee, 0);
            }
        }

        /// <summary>
        ///     A <c>return</c> (or an expression body's implicit one) of a method, local function or getter that returns
        ///     <c>string</c>: scans the returned text and keeps what it found under the owner, for the compilation-end
        ///     pass to report if a message turns out to reach the owner.
        /// </summary>
        /// <param name="context">The callback's context (the return, its containing symbol).</param>
        /// <param name="scanner">The compilation's scanner.</param>
        /// <param name="state">The compilation's shared collections.</param>
        private static void OnReturn(OperationAnalysisContext context, Scanner scanner, CompilationState state)
        {
            var ret = (IReturnOperation)context.Operation;
            if (ret.ReturnedValue is null)
                return;
            var owner = OwnerOf(ret, context.ContainingSymbol);
            if (owner is null || owner.ReturnType.SpecialType != SpecialType.System_String)
                return;
            var result = new ScanResult();
            scanner.Scan(ret.ReturnedValue, result);
            // Most returns format nothing and call nothing of this compilation: keep only those that matter, so the
            // table stays the size of the message producers rather than of every string-returning method.
            if (result.Findings.Count == 0 && result.Edges.Count == 0)
                return;
            state.Producers.GetOrAdd(owner.OriginalDefinition, _ => new ConcurrentBag<ScanResult>()).Add(result);
        }

        /// <summary>
        ///     The function a <c>return</c> belongs to: the nearest enclosing local function, else the containing member
        ///     (a property's getter for a property). A return inside a lambda belongs to the lambda, which no message
        ///     calls by name, so it has no owner here.
        /// </summary>
        /// <param name="ret">The return operation.</param>
        /// <param name="containingSymbol">The callback's containing symbol.</param>
        /// <returns>The owning method, or null for a lambda's return or a non-method container.</returns>
        private static IMethodSymbol OwnerOf(IReturnOperation ret, ISymbol containingSymbol)
        {
            for (var parent = ret.Parent; parent != null; parent = parent.Parent)
            {
                if (parent is IAnonymousFunctionOperation)
                    return null;
                if (parent is ILocalFunctionOperation localFunction)
                    return localFunction.Symbol;
            }
            return containingSymbol switch
            {
                IMethodSymbol method => method,
                IPropertySymbol property => property.GetMethod,
                _ => null,
            };
        }

        /// <summary>
        ///     Walks from the methods exception messages call through the producers' own calls, and reports every
        ///     culture-sensitive construct in a producer reached. Runs once, single-threaded, after every operation
        ///     callback of the compilation has finished.
        /// </summary>
        /// <param name="context">The compilation-end context (the reporter).</param>
        /// <param name="state">The compilation's shared collections.</param>
        private static void OnCompilationEnd(CompilationAnalysisContext context, CompilationState state)
        {
            var queue = new Queue<IMethodSymbol>(state.Roots.Keys);
            var seen = new HashSet<IMethodSymbol>(SymbolEqualityComparer.Default);
            while (queue.Count > 0)
            {
                var method = queue.Dequeue();
                if (!seen.Add(method) || !state.Producers.TryGetValue(method, out var results))
                    continue;
                string name = method.ContainingType is null ? method.Name : method.ContainingType.Name + "." + method.Name;
                foreach (var result in results)
                {
                    foreach (var finding in result.Findings)
                        if (state.Reported.TryAdd(finding.Location, 0))
                            context.ReportDiagnostic(Diagnostic.Create(Nw020Reached, finding.Location, finding.What, name));
                    foreach (var callee in result.Edges)
                        queue.Enqueue(callee);
                }
            }
        }

        /// <summary>One culture-sensitive construct: where it is, and the words that name it in the diagnostic.</summary>
        private readonly struct Finding
        {
            /// <summary>The construct's location (the interpolated string, the operand, the call).</summary>
            public readonly Location Location;

            /// <summary>The construct in words, the diagnostic's first argument (<c>the interpolation {i} (long)</c>).</summary>
            public readonly string What;

            /// <summary>Records a construct.</summary>
            /// <param name="location">Where it is.</param>
            /// <param name="what">What it is, in words.</param>
            public Finding(Location location, string what)
            {
                Location = location;
                What = what;
            }
        }

        /// <summary>What one scan of a text expression found: the constructs to report and the source methods it calls.</summary>
        private sealed class ScanResult
        {
            /// <summary>The culture-sensitive constructs, in the order the walk met them.</summary>
            public readonly List<Finding> Findings = new List<Finding>();

            /// <summary>The source methods (original definitions) whose returned text flows into the scanned text.</summary>
            public readonly List<IMethodSymbol> Edges = new List<IMethodSymbol>();
        }

        /// <summary>
        ///     The collections one compilation's callbacks share. Every callback may run concurrently with the others, so
        ///     each is a concurrent collection; the compilation-end pass reads them after the callbacks are done.
        /// </summary>
        private sealed class CompilationState
        {
            /// <summary>Methods an exception message calls directly: where the compilation-end walk starts.</summary>
            public readonly ConcurrentDictionary<IMethodSymbol, byte> Roots =
                new ConcurrentDictionary<IMethodSymbol, byte>(SymbolEqualityComparer.Default);

            /// <summary>Per string-returning method, the scans of its returns that found something or call something.</summary>
            public readonly ConcurrentDictionary<IMethodSymbol, ConcurrentBag<ScanResult>> Producers =
                new ConcurrentDictionary<IMethodSymbol, ConcurrentBag<ScanResult>>(SymbolEqualityComparer.Default);

            /// <summary>
            ///     Locations already reported. A local read by two messages, or a text reached both directly and through a
            ///     producer, would otherwise draw the same diagnostic twice.
            /// </summary>
            public readonly ConcurrentDictionary<Location, byte> Reported = new ConcurrentDictionary<Location, byte>();
        }

        /// <summary>
        ///     The per-compilation walker: the BCL symbols the checks compare against, resolved once, and the scan of one
        ///     text expression.
        /// </summary>
        private sealed class Scanner
        {
            /// <summary>The longest expression text a finding quotes before it is cut with an ellipsis.</summary>
            private const int QuoteCap = 48;

            private readonly INamedTypeSymbol _exception;
            private readonly INamedTypeSymbol _argumentOutOfRange;
            private readonly INamedTypeSymbol _formatProvider;
            private readonly INamedTypeSymbol _formattable;
            private readonly INamedTypeSymbol _convertible;
            private readonly INamedTypeSymbol _formattableString;
            private readonly INamedTypeSymbol _stringBuilder;
            private readonly INamedTypeSymbol _convert;
            private readonly INamedTypeSymbol _guid;
            private readonly INamedTypeSymbol _version;

            /// <summary>Captures the resolved symbols (any but <paramref name="exception"/> may be null on an odd reference set).</summary>
            /// <param name="compilation">The compilation the symbols came from.</param>
            /// <param name="exception"><c>System.Exception</c>.</param>
            private Scanner(Compilation compilation, INamedTypeSymbol exception)
            {
                _exception = exception;
                _argumentOutOfRange = compilation.GetTypeByMetadataName("System.ArgumentOutOfRangeException");
                _formatProvider = compilation.GetTypeByMetadataName("System.IFormatProvider");
                _formattable = compilation.GetTypeByMetadataName("System.IFormattable");
                _convertible = compilation.GetTypeByMetadataName("System.IConvertible");
                _formattableString = compilation.GetTypeByMetadataName("System.FormattableString");
                _stringBuilder = compilation.GetTypeByMetadataName("System.Text.StringBuilder");
                _convert = compilation.GetTypeByMetadataName("System.Convert");
                _guid = compilation.GetTypeByMetadataName("System.Guid");
                _version = compilation.GetTypeByMetadataName("System.Version");
            }

            /// <summary>Resolves the scanner's symbols for <paramref name="compilation"/>.</summary>
            /// <param name="compilation">The compilation to analyze.</param>
            /// <returns>The scanner, or null when the compilation has no <c>System.Exception</c> to anchor the rule on.</returns>
            public static Scanner Create(Compilation compilation)
            {
                var exception = compilation.GetTypeByMetadataName("System.Exception");
                return exception is null ? null : new Scanner(compilation, exception);
            }

            /// <summary>Whether <paramref name="type"/> is <c>System.Exception</c> or derives from it.</summary>
            /// <param name="type">The type to test (null answers false).</param>
            /// <returns>True for an exception type.</returns>
            public bool IsException(ITypeSymbol type)
            {
                for (var t = type as INamedTypeSymbol; t != null; t = t.BaseType)
                    if (SymbolEqualityComparer.Default.Equals(t, _exception))
                        return true;
                return false;
            }

            /// <summary>
            ///     Whether <paramref name="parameter"/> is the <c>actualValue</c> of an
            ///     <see cref="System.ArgumentOutOfRangeException"/> constructor (or of a derived exception's, such as
            ///     NumSharp's <c>AxisError</c> chaining to it): the value its <c>Message</c> appends as
            ///     <c>Actual value was ….</c>, formatted with the current culture.
            /// </summary>
            /// <param name="parameter">The constructor parameter an argument is bound to.</param>
            /// <returns>True for that parameter.</returns>
            public bool IsActualValueParameter(IParameterSymbol parameter)
            {
                if (parameter.Name != "actualValue" || parameter.Type.SpecialType != SpecialType.System_Object
                    || !(parameter.ContainingSymbol is IMethodSymbol constructor) || constructor.MethodKind != MethodKind.Constructor)
                    return false;
                for (var t = constructor.ContainingType; t != null; t = t.BaseType)
                    if (Is(t, _argumentOutOfRange))
                        return true;
                return false;
            }

            /// <summary>
            ///     Whether a value of static type <paramref name="type"/> formats differently under another culture when
            ///     no provider is given: a number (every IFormattable but the culture-free Guid, Version, char and enums),
            ///     or a type that may hold one at run time (object, dynamic, ValueType, IFormattable, IConvertible, a
            ///     type parameter not pinned to a non-formattable class).
            /// </summary>
            /// <param name="type">The static type (null answers false).</param>
            /// <returns>True when the text may depend on the current culture.</returns>
            public bool IsCultureSensitive(ITypeSymbol type)
            {
                switch (type)
                {
                    case null:
                    case IErrorTypeSymbol _:
                        return false;
                    case IDynamicTypeSymbol _:
                        return true;
                    case ITypeParameterSymbol parameter:
                        // A class constraint that is not formattable pins the value to that class (NDArray, Shape):
                        // its ToString is the class's, not a number's. Every other shape may be a number at run time.
                        foreach (var constraint in parameter.ConstraintTypes)
                            if (constraint.TypeKind == TypeKind.Class && !IsCultureSensitive(constraint))
                                return false;
                        return true;
                }
                if (type.TypeKind == TypeKind.Enum)
                    return false;
                switch (type.SpecialType)
                {
                    case SpecialType.System_String:
                    case SpecialType.System_Char:
                    case SpecialType.System_Boolean:
                    case SpecialType.System_Enum:
                        return false;
                    case SpecialType.System_Object:
                    case SpecialType.System_ValueType:
                        return true;
                }
                if (type is INamedTypeSymbol named && named.OriginalDefinition.SpecialType == SpecialType.System_Nullable_T)
                    return IsCultureSensitive(named.TypeArguments[0]);
                if (Is(type, _guid) || Is(type, _version))
                    return false;
                if (Is(type, _formattable) || Is(type, _convertible))
                    return true;
                foreach (var implemented in type.AllInterfaces)
                    if (Is(implemented, _formattable))
                        return true;
                return false;
            }

            /// <summary>
            ///     Walks <paramref name="root"/> (a message argument or a returned text) and everything under it, and
            ///     collects the culture-sensitive constructs and the source methods whose text flows in.
            /// </summary>
            /// <param name="root">The text expression.</param>
            /// <param name="result">Receives the findings and the edges.</param>
            /// <remarks>
            ///     Iterative (an explicit stack): a long <c>+</c> chain nests one binary operation per operand, deep enough
            ///     to matter for a recursive walk on the compiler's thread. Lambdas are walked too: the text a
            ///     <c>Select(d =&gt; d.ToString())</c> inside a message builds is part of the message. A nested exception
            ///     creation is skipped: its own arguments are its own message, analyzed at its own creation.
            /// </remarks>
            public void Scan(IOperation root, ScanResult result)
            {
                var stack = new Stack<IOperation>();
                // Interpolated strings whose culture was chosen by the receiver (a provider-taking handler, a
                // FormattableString conversion): their holes are still walked, the string itself is not flagged.
                var chosenCulture = new HashSet<IOperation>();
                var followedLocals = new HashSet<ILocalSymbol>(SymbolEqualityComparer.Default);
                stack.Push(root);
                while (stack.Count > 0)
                {
                    var operation = stack.Pop();
                    if (operation is null)
                        continue;
                    switch (operation)
                    {
                        case IObjectCreationOperation creation when !ReferenceEquals(operation, root) && IsException(creation.Type):
                            continue;
                        case IInterpolatedStringHandlerCreationOperation handler when HandlerTakesProvider(handler):
                            MarkChosen(handler.Content, chosenCulture);
                            break;
                        case IConversionOperation conversion when Is(conversion.Type, _formattableString) || Is(conversion.Type, _formattable):
                            MarkChosen(conversion.Operand, chosenCulture);
                            break;
                        case IInterpolatedStringOperation interpolated when !chosenCulture.Contains(interpolated):
                            CheckHoles(interpolated, result);
                            break;
                        case IBinaryOperation binary when binary.OperatorKind == BinaryOperatorKind.Add
                                                          && binary.Type?.SpecialType == SpecialType.System_String:
                            CheckFormattedOperand(binary.LeftOperand, "the + operand", result);
                            CheckFormattedOperand(binary.RightOperand, "the + operand", result);
                            break;
                        case ICompoundAssignmentOperation compound when compound.OperatorKind == BinaryOperatorKind.Add
                                                                        && compound.Type?.SpecialType == SpecialType.System_String:
                            CheckFormattedOperand(compound.Value, "the += operand", result);
                            break;
                        case IInvocationOperation invocation:
                            CheckInvocation(invocation, result);
                            break;
                        case IPropertyReferenceOperation property when property.Type?.SpecialType == SpecialType.System_String
                                                                       && property.Property.GetMethod is { } getter
                                                                       && IsInSource(getter):
                            result.Edges.Add(getter.OriginalDefinition);
                            break;
                        case ILocalReferenceOperation local when IsFollowedLocalType(local.Local.Type) && followedLocals.Add(local.Local):
                            PushLocalWrites(local, stack);
                            break;
                    }
                    foreach (var child in operation.ChildOperations)
                        stack.Push(child);
                }
            }

            /// <summary>
            ///     Marks an interpolated string (or each side of an interpolated-string addition) as formatted under a
            ///     culture its receiver chose.
            /// </summary>
            /// <param name="content">The handler's content or the converted operand.</param>
            /// <param name="chosenCulture">The set of strings not to flag.</param>
            private static void MarkChosen(IOperation content, HashSet<IOperation> chosenCulture)
            {
                switch (content)
                {
                    case IInterpolatedStringOperation interpolated:
                        chosenCulture.Add(interpolated);
                        break;
                    case IInterpolatedStringAdditionOperation addition:
                        MarkChosen(addition.Left, chosenCulture);
                        MarkChosen(addition.Right, chosenCulture);
                        break;
                }
            }

            /// <summary>
            ///     Whether an interpolated-string handler is built with a format provider
            ///     (<c>DefaultInterpolatedStringHandler(int, int, IFormatProvider)</c> under <c>string.Create</c>,
            ///     <c>StringBuilder.AppendInterpolatedStringHandler(int, int, StringBuilder, IFormatProvider)</c>).
            /// </summary>
            /// <param name="handler">The handler creation.</param>
            /// <returns>True when its constructor takes a provider; false for a provider-less handler or an invalid one.</returns>
            private bool HandlerTakesProvider(IInterpolatedStringHandlerCreationOperation handler)
                => handler.HandlerCreation is IObjectCreationOperation creation && creation.Constructor != null
                   && TakesProvider(creation.Constructor);

            /// <summary>Whether any parameter of <paramref name="method"/> is an <c>IFormatProvider</c>.</summary>
            /// <param name="method">The method or constructor.</param>
            /// <returns>True when a provider can be passed.</returns>
            private bool TakesProvider(IMethodSymbol method)
            {
                foreach (var parameter in method.Parameters)
                    if (Is(parameter.Type, _formatProvider) || Implements(parameter.Type, _formatProvider))
                        return true;
                return false;
            }

            /// <summary>
            ///     Flags an interpolated string formatted with the current culture when one of its holes is
            ///     culture-sensitive, and records the source <c>ToString()</c> overrides its holes format.
            /// </summary>
            /// <param name="interpolated">The interpolated string (a plain <c>$"..."</c> or a provider-less handler's content).</param>
            /// <param name="result">Receives the finding and the edges.</param>
            private void CheckHoles(IInterpolatedStringOperation interpolated, ScanResult result)
            {
                IOperation first = null;
                int count = 0;
                foreach (var part in interpolated.Parts)
                {
                    var value = HoleValue(part);
                    if (value is null)
                        continue;
                    if (IsCultureSensitive(value.Type))
                    {
                        first ??= value;
                        count++;
                    }
                    else
                    {
                        AddToStringEdge(value.Type, result);
                    }
                }
                if (first is null)
                    return;
                string more = count > 1 ? " and " + (count - 1) + " more" : "";
                result.Findings.Add(new Finding(interpolated.Syntax.GetLocation(),
                    "the interpolation {" + Quote(first) + "} (" + Display(first.Type) + ")" + more));
            }

            /// <summary>
            ///     The formatted value of one interpolated-string part: a classic hole's expression, or a handler's
            ///     <c>AppendFormatted</c> argument; null for literal text.
            /// </summary>
            /// <param name="part">The part.</param>
            /// <returns>The value with implicit conversions removed, or null.</returns>
            private static IOperation HoleValue(IInterpolatedStringContentOperation part)
            {
                switch (part)
                {
                    case IInterpolationOperation hole:
                        return StripImplicit(hole.Expression);
                    case IInterpolatedStringAppendOperation append when append.Kind == OperationKind.InterpolatedStringAppendFormatted
                                                                       && append.AppendCall is IInvocationOperation call
                                                                       && call.Arguments.Length > 0:
                        return StripImplicit(call.Arguments[0].Value);
                    default:
                        return null;
                }
            }

            /// <summary>
            ///     Flags a value that a concatenation turns into text through its <c>ToString()</c> (a <c>+</c> or
            ///     <c>+=</c> operand of type number, object, …), or records the source <c>ToString()</c> it calls.
            /// </summary>
            /// <param name="operand">The operand.</param>
            /// <param name="role">The operand's role in words (<c>the + operand</c>).</param>
            /// <param name="result">Receives the finding or the edge.</param>
            private void CheckFormattedOperand(IOperation operand, string role, ScanResult result)
            {
                var value = StripImplicit(operand);
                if (value?.Type is null || value.Type.SpecialType == SpecialType.System_String)
                    return;
                if (IsCultureSensitive(value.Type))
                    result.Findings.Add(new Finding(value.Syntax.GetLocation(), role + " " + Quote(value) + " (" + Display(value.Type) + ")"));
                else
                    AddToStringEdge(value.Type, result);
            }

            /// <summary>
            ///     The calls that format text: <c>ToString</c>, <c>Convert.ToString</c>, <c>string.Format</c> /
            ///     <c>Join</c> / <c>Concat</c>, the <c>StringBuilder</c> appends; and a call to a source method returning
            ///     <c>string</c>, whose returned text flows in (an edge).
            /// </summary>
            /// <param name="invocation">The call.</param>
            /// <param name="result">Receives the findings and the edges.</param>
            private void CheckInvocation(IInvocationOperation invocation, ScanResult result)
            {
                var method = invocation.TargetMethod;
                var containing = method.ContainingType;
                if (method.Name == "ToString" && !method.IsStatic && invocation.Instance != null && !TakesProvider(method))
                {
                    var receiver = StripImplicit(invocation.Instance);
                    if (IsCultureSensitive(receiver.Type))
                        result.Findings.Add(new Finding(invocation.Syntax.GetLocation(),
                            "'" + Quote(invocation) + "' (" + Display(receiver.Type) + ") without a format provider"));
                }
                else if (containing?.SpecialType == SpecialType.System_String && method.IsStatic)
                {
                    if (method.Name == "Format" && !TakesProvider(method))
                        FlagFormattedArguments(invocation, skip: 1, "string.Format", result);
                    else if (method.Name == "Join" || method.Name == "Concat")
                        FlagJoinedValues(invocation, skip: method.Name == "Join" ? 1 : 0, "string." + method.Name, result);
                }
                else if (Is(containing, _convert) && method.Name == "ToString" && !TakesProvider(method)
                         && invocation.Arguments.Length == 1)
                {
                    // Convert.ToString(value, toBase) prints an integer in base 2 / 8 / 16 with no sign of the culture,
                    // so only the one-argument form is checked.
                    var value = StripImplicit(invocation.Arguments[0].Value);
                    if (IsCultureSensitive(value.Type))
                        result.Findings.Add(new Finding(invocation.Syntax.GetLocation(),
                            "'" + Quote(invocation) + "' (" + Display(value.Type) + ") without a format provider"));
                }
                else if (Is(containing, _stringBuilder))
                {
                    CheckStringBuilderCall(invocation, result);
                }

                if (method.ReturnType.SpecialType == SpecialType.System_String && IsInSource(method))
                    result.Edges.Add(method.OriginalDefinition);
            }

            /// <summary>
            ///     <c>StringBuilder.Append(number)</c> / <c>Insert(index, number)</c> (the overloads taking the value's own
            ///     type format it with the current culture), <c>AppendFormat</c> without a provider, and
            ///     <c>AppendJoin</c> over numbers.
            /// </summary>
            /// <param name="invocation">The call on a <c>StringBuilder</c>.</param>
            /// <param name="result">Receives the findings.</param>
            private void CheckStringBuilderCall(IInvocationOperation invocation, ScanResult result)
            {
                var method = invocation.TargetMethod;
                switch (method.Name)
                {
                    case "Append" when method.Parameters.Length == 1 && invocation.Arguments.Length == 1:
                    case "Insert" when method.Parameters.Length == 2 && invocation.Arguments.Length == 2:
                        var argument = invocation.Arguments[invocation.Arguments.Length - 1];
                        var value = StripImplicit(argument.Value);
                        if (argument.Parameter != null && IsCultureSensitive(argument.Parameter.Type) && IsCultureSensitive(value.Type))
                            result.Findings.Add(new Finding(invocation.Syntax.GetLocation(),
                                "'" + Quote(invocation) + "' (" + Display(value.Type) + ")"));
                        else if (value.Type != null && value.Type.SpecialType != SpecialType.System_String)
                            AddToStringEdge(value.Type, result);
                        break;
                    case "AppendFormat" when !TakesProvider(method):
                        FlagFormattedArguments(invocation, skip: 1, "StringBuilder.AppendFormat", result);
                        break;
                    case "AppendJoin":
                        FlagJoinedValues(invocation, skip: 1, "StringBuilder.AppendJoin", result);
                        break;
                }
            }

            /// <summary>
            ///     A composite-format call without a provider (<c>string.Format(fmt, args)</c>,
            ///     <c>AppendFormat</c>): flags it when one of its formatted arguments is culture-sensitive.
            /// </summary>
            /// <param name="invocation">The call.</param>
            /// <param name="skip">How many leading arguments are not formatted values (the format string).</param>
            /// <param name="name">The call's name in words.</param>
            /// <param name="result">Receives the finding.</param>
            private void FlagFormattedArguments(IInvocationOperation invocation, int skip, string name, ScanResult result)
            {
                foreach (var value in ValuesOf(invocation, skip))
                {
                    if (!IsCultureSensitive(value.Type))
                        continue;
                    result.Findings.Add(new Finding(invocation.Syntax.GetLocation(),
                        name + " over " + Quote(value) + " (" + Display(value.Type) + ") without a format provider"));
                    return;
                }
            }

            /// <summary>
            ///     <c>string.Join</c> / <c>string.Concat</c> / <c>AppendJoin</c>: each element is turned into text by its
            ///     <c>ToString()</c> (no overload takes a provider), so a generic element type or an element value that is
            ///     culture-sensitive is flagged.
            /// </summary>
            /// <param name="invocation">The call.</param>
            /// <param name="skip">How many leading arguments are not joined values (the separator).</param>
            /// <param name="name">The call's name in words.</param>
            /// <param name="result">Receives the finding.</param>
            private void FlagJoinedValues(IInvocationOperation invocation, int skip, string name, ScanResult result)
            {
                var method = invocation.TargetMethod;
                ITypeSymbol element = null;
                if (method.IsGenericMethod && method.TypeArguments.Length == 1)
                {
                    element = method.TypeArguments[0];
                }
                else
                {
                    foreach (var value in ValuesOf(invocation, skip))
                    {
                        // An array passed whole: its element type is what each ToString() formats. A reference
                        // conversion of an array ((object[])types, written to pick the params object[] overload) does
                        // not change what the elements are, so the element type is read through it.
                        var source = value;
                        while (source is IConversionOperation conversion && conversion.Operand?.Type is IArrayTypeSymbol)
                            source = conversion.Operand;
                        var type = source.Type is IArrayTypeSymbol array ? array.ElementType : value.Type;
                        if (IsCultureSensitive(type))
                        {
                            element = type;
                            break;
                        }
                    }
                }
                if (element != null && IsCultureSensitive(element))
                    result.Findings.Add(new Finding(invocation.Syntax.GetLocation(), name + " over " + Display(element)));
            }

            /// <summary>
            ///     The values a call formats, past its leading <paramref name="skip"/> arguments: each argument, with a
            ///     <c>params</c> array expanded into its elements; implicit conversions (the boxing to object) removed.
            /// </summary>
            /// <param name="invocation">The call.</param>
            /// <param name="skip">How many leading arguments to pass over.</param>
            /// <returns>The values, in order.</returns>
            private static IEnumerable<IOperation> ValuesOf(IInvocationOperation invocation, int skip)
            {
                for (int i = skip; i < invocation.Arguments.Length; i++)
                {
                    var argument = invocation.Arguments[i];
                    if (argument.ArgumentKind == ArgumentKind.ParamArray && argument.Value is IArrayCreationOperation array
                        && array.Initializer != null)
                    {
                        foreach (var element in array.Initializer.ElementValues)
                            yield return StripImplicit(element);
                    }
                    else if (argument.Value != null)
                    {
                        yield return StripImplicit(argument.Value);
                    }
                }
            }

            /// <summary>
            ///     Pushes what was written into a followed local: its declaration's initializer, the values assigned to it,
            ///     each <c>+=</c> (whose operand the walk then checks), and the <c>StringBuilder</c> calls made on it
            ///     (directly or down a fluent chain), all within the body the reference sits in.
            /// </summary>
            /// <param name="reference">The reference that made the walk follow the local.</param>
            /// <param name="stack">The walk's stack.</param>
            /// <remarks>
            ///     The writes come from the body's <see cref="LocalWrites"/> index, built by ONE walk of the body and shared
            ///     by every scan of that body: Roslyn hands every operation callback of a method the same tree, and a
            ///     large method with many throws that read string locals would otherwise walk its whole body once per
            ///     followed local per message (most of the rule's cost on NumSharp.Core before the index).
            /// </remarks>
            private void PushLocalWrites(ILocalReferenceOperation reference, Stack<IOperation> stack)
            {
                IOperation body = reference;
                while (body.Parent != null)
                    body = body.Parent;
                var index = _localWrites.GetValue(body, BuildLocalWrites);
                if (index.TryGetValue(reference.Local, out var writes))
                    foreach (var write in writes)
                        stack.Push(write);
            }

            /// <summary>
            ///     Walks a body once and records, per <c>string</c> or <c>StringBuilder</c> local, the operations a scan must
            ///     follow: the declaration's initializer value, each simple assignment's value, each compound assignment
            ///     itself (the scan checks its <c>+=</c> operand), and each <c>StringBuilder</c> call whose fluent chain
            ///     starts at the local.
            /// </summary>
            /// <param name="body">The root of a method's (or field initializer's) operation tree.</param>
            /// <returns>The index; locals without a write are absent.</returns>
            private LocalWrites BuildLocalWrites(IOperation body)
            {
                var index = new LocalWrites();
                foreach (var operation in body.Descendants())
                {
                    switch (operation)
                    {
                        case IVariableDeclaratorOperation declarator when IsFollowedLocalType(declarator.Symbol.Type):
                            var initializer = declarator.Initializer ?? (declarator.Parent as IVariableDeclarationOperation)?.Initializer;
                            if (initializer?.Value != null)
                                index.Add(declarator.Symbol, initializer.Value);
                            break;
                        case ISimpleAssignmentOperation assignment when assignment.Target is ILocalReferenceOperation target
                                                                        && IsFollowedLocalType(target.Local.Type):
                            index.Add(target.Local, assignment.Value);
                            break;
                        case ICompoundAssignmentOperation compound when compound.Target is ILocalReferenceOperation target
                                                                        && IsFollowedLocalType(target.Local.Type):
                            index.Add(target.Local, compound);
                            break;
                        case IInvocationOperation call when Is(call.TargetMethod.ContainingType, _stringBuilder)
                                                            && ChainReceiver(call) is ILocalReferenceOperation receiver:
                            index.Add(receiver.Local, call);
                            break;
                    }
                }
                return index;
            }

            /// <summary>
            ///     Per body root, its <see cref="LocalWrites"/> index. Weak keys: the index lives exactly as long as Roslyn
            ///     keeps the body's operation tree, so the compilation's trees are never pinned by the rule.
            /// </summary>
            private readonly ConditionalWeakTable<IOperation, LocalWrites> _localWrites = new ConditionalWeakTable<IOperation, LocalWrites>();

            /// <summary>
            ///     The writes of one body's text-carrying locals, keyed by the local. Built once, then only read, so the
            ///     concurrent scans of one body share it without a lock.
            /// </summary>
            private sealed class LocalWrites : Dictionary<ILocalSymbol, List<IOperation>>
            {
                /// <summary>Creates an empty index keyed by symbol identity.</summary>
                public LocalWrites() : base(SymbolEqualityComparer.Default)
                {
                }

                /// <summary>Records one operation to follow for <paramref name="local"/>.</summary>
                /// <param name="local">The local written.</param>
                /// <param name="write">The operation the scan pushes when it follows the local.</param>
                public void Add(ILocalSymbol local, IOperation write)
                {
                    if (!TryGetValue(local, out var writes))
                        this[local] = writes = new List<IOperation>();
                    writes.Add(write);
                }
            }

            /// <summary>The innermost receiver of a fluent call chain (<c>sb</c> in <c>sb.Append(a).Append(b)</c>).</summary>
            /// <param name="call">The outermost call.</param>
            /// <returns>The receiver the chain starts from, conversions removed.</returns>
            private static IOperation ChainReceiver(IInvocationOperation call)
            {
                IOperation receiver = StripImplicit(call.Instance);
                while (receiver is IInvocationOperation inner && inner.Instance != null)
                    receiver = StripImplicit(inner.Instance);
                return receiver;
            }

            /// <summary>The local types whose writes the walk follows: a <c>string</c> and a <c>StringBuilder</c>.</summary>
            /// <param name="type">The local's type.</param>
            /// <returns>True for a text-carrying local.</returns>
            private bool IsFollowedLocalType(ITypeSymbol type)
                => type?.SpecialType == SpecialType.System_String || Is(type, _stringBuilder);

            /// <summary>
            ///     Records the <c>ToString()</c> override a value of a SOURCE type runs when a message formats it (a
            ///     <c>Slice</c> or <c>Shape</c> in a hole), so the override's own text is checked as a producer.
            /// </summary>
            /// <param name="type">The formatted value's type.</param>
            /// <param name="result">Receives the edge.</param>
            private static void AddToStringEdge(ITypeSymbol type, ScanResult result)
            {
                for (var t = type as INamedTypeSymbol; t != null; t = t.BaseType)
                {
                    foreach (var member in t.GetMembers("ToString"))
                    {
                        if (member is IMethodSymbol method && method.Parameters.Length == 0 && method.IsOverride)
                        {
                            // The nearest override is the one that runs; one of metadata is outside the compilation.
                            if (IsInSource(method))
                                result.Edges.Add(method.OriginalDefinition);
                            return;
                        }
                    }
                }
            }

            /// <summary>Whether <paramref name="method"/> is declared in the compilation being analyzed.</summary>
            /// <param name="method">The method.</param>
            /// <returns>True when its definition has a source location.</returns>
            private static bool IsInSource(IMethodSymbol method)
            {
                foreach (var location in method.OriginalDefinition.Locations)
                    if (location.IsInSource)
                        return true;
                return false;
            }

            /// <summary>Removes the implicit conversions the compiler wraps around a value (the boxing to object of a hole or an operand).</summary>
            /// <param name="operation">The operation (may be null).</param>
            /// <returns>The value as written.</returns>
            internal static IOperation StripImplicit(IOperation operation)
            {
                while (operation is IConversionOperation conversion && conversion.IsImplicit)
                    operation = conversion.Operand;
                return operation;
            }

            /// <summary>Whether <paramref name="type"/> implements <paramref name="interfaceType"/>.</summary>
            /// <param name="type">The type.</param>
            /// <param name="interfaceType">The interface (null answers false).</param>
            /// <returns>True when it is among the type's interfaces.</returns>
            private static bool Implements(ITypeSymbol type, INamedTypeSymbol interfaceType)
            {
                if (type is null || interfaceType is null)
                    return false;
                foreach (var implemented in type.AllInterfaces)
                    if (SymbolEqualityComparer.Default.Equals(implemented, interfaceType))
                        return true;
                return false;
            }

            /// <summary>Symbol identity that tolerates a symbol the compilation could not resolve.</summary>
            /// <param name="type">The type tested.</param>
            /// <param name="known">The resolved well-known type (null answers false).</param>
            /// <returns>True when both are the same type.</returns>
            private static bool Is(ITypeSymbol type, INamedTypeSymbol known)
                => known != null && SymbolEqualityComparer.Default.Equals(type, known);

            /// <summary>An expression's source text for a finding, cut to <see cref="QuoteCap"/> characters on one line.</summary>
            /// <param name="operation">The expression.</param>
            /// <returns>The quoted text.</returns>
            /// <remarks>
            ///     The call of a conditional access (<c>x?.ToString()</c>) is only the <c>.ToString()</c> after the
            ///     <c>?</c>, which names nothing; the whole access is quoted instead.
            /// </remarks>
            internal static string Quote(IOperation operation)
            {
                var syntax = operation.Syntax;
                string text = syntax.ToString();
                if (text.StartsWith(".", System.StringComparison.Ordinal) && syntax.Parent != null)
                    text = syntax.Parent.ToString();
                text = text.Replace("\r", " ").Replace("\n", " ");
                return text.Length <= QuoteCap ? text : text.Substring(0, QuoteCap - 1) + "…";
            }

            /// <summary>A type's short C# name for a finding (<c>long</c>, <c>double?</c>, <c>T</c>).</summary>
            /// <param name="type">The type (null prints as <c>?</c>).</param>
            /// <returns>The name.</returns>
            internal static string Display(ITypeSymbol type)
                => type?.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat) ?? "?";
        }
    }
}
