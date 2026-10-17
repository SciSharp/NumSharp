using System.Globalization;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using NumSharp;

// Module surfaces are DISCOVERED, not hardcoded: every public type in NumSharp.Core annotated with
// [ModuleName("...")] is a NumPy module host (np itself, NDArray, and each function-namespace facade —
// NumPyRandom/FourierModule/np.linalg today). A new facade joins the coverage artifact by annotation
// alone. The hardcoded typeof(...) list this replaces is how the whole np.fft surface and every linalg
// factorisation went missing from the artifact in the first place.
var assembly = typeof(np).Assembly;
var annotatedTypes = new Dictionary<Type, string>();
foreach (var type in assembly.GetExportedTypes())
{
    var module = type.GetCustomAttribute<ModuleNameAttribute>(inherit: false);
    if (module is not null)
        annotatedTypes.Add(type, module.Name);
}

if (annotatedTypes.Count == 0)
    throw new InvalidOperationException(
        $"No [ModuleName]-annotated types found in {assembly.GetName().Name} — the inventory would be empty.");

// The scan-integrity guards run BEFORE inventorying: a shape that would let members hide from the
// scan is an error at the source, not a silent gap in the artifact.
GuardHierarchy(annotatedTypes);
GuardFacadeShapes(assembly, annotatedTypes);

// Members are reflected with Static AND Instance flags for every module: a static class cannot have
// instance members, and an instance facade that grows a public static helper must still be visible
// (with per-member Static recorded) instead of silently escaping an instance-only scan.
const BindingFlags MemberFlags =
    BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly;

var modules = new SortedDictionary<string, TypeInventory>(StringComparer.Ordinal);
foreach (var (type, moduleName) in annotatedTypes)
{
    if (modules.TryGetValue(moduleName, out var taken))
        throw new InvalidOperationException(
            $"Duplicate [ModuleName(\"{moduleName}\")]: both {taken.Type} and {type.FullName} claim it.");
    modules.Add(moduleName, InspectType(type, MemberFlags));
}

var inventory = new ApiInventory(
    // Schema 6 adds TypeInventory.Operators: the C# operator and conversion methods, which the generator
    // needs to credit NumPy's Python operator protocols (ndarray.__add__ is spelled `a + b` in both
    // languages). Earlier schemas dropped every special-name method, so an operator was invisible.
    // Schema 7 adds ApiMember.Overloads: every method overload with its parameters as DATA (name, type,
    // default, params, by-ref) and the attributes that decide how C# binds it. Before it the inventory
    // carried each overload only as display text, so the generator could credit a NumPy function by NAME
    // and nothing else: np.sum read "available" with out=/initial=/where= missing, seven overloads and an
    // NPTypeCode parameter, because no step ever looked at a parameter.
    SchemaVersion: 7,
    AssemblyVersion: assembly.GetName().Version?.ToString() ?? "unknown",
    Modules: modules,
    // The full public surface OUTSIDE the annotated modules (type -> member names). The generator
    // cross-checks every still-missing in-scope NumPy export against this index and fails when the
    // name exists here — a NumPy function implemented on an unannotated type is a scan miss, not a
    // genuine gap. Extension-method hosts are exported static classes, so they land here too.
    UnannotatedSurface: BuildUnannotatedSurface(assembly, annotatedTypes.Keys),
    // Every exported public type's full name. A NumPy CLASS export (numpy.random.Generator/PCG64/
    // SeedSequence/BitGenerator/MT19937) is credited when NumSharp exports a type of the same name —
    // the member-name index above cannot carry that, since a class like BitGenerator has no public
    // members of its own. Includes types with no public members, which UnannotatedSurface omits.
    ExportedTypes: assembly.GetExportedTypes()
        .Select(type => type.FullName ?? type.Name)
        .OrderBy(name => name, StringComparer.Ordinal)
        .ToArray(),
    // Object APIs are a separate inventory from module facades: Generator, dtype, iterator and
    // bit-generator members are not exports of np.random/np themselves. Include inherited members
    // (PCG64's BitGenerator contract, for example), but never System.Object's infrastructure.
    ObjectTypes: assembly.GetExportedTypes()
        .Where(type => !type.IsEnum)
        .OrderBy(type => type.FullName, StringComparer.Ordinal)
        .ToDictionary(type => type.FullName ?? type.Name,
            type => InspectType(type, BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance
                | BindingFlags.FlattenHierarchy), StringComparer.Ordinal));

Console.WriteLine(JsonSerializer.Serialize(inventory, new JsonSerializerOptions
{
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    WriteIndented = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
}));

/// <summary>
///     Reflects one type's public surface: methods, properties, fields and (schema 6) C# operators and conversions,
///     each grouped by name with every overload's signature, and (schema 7) every method overload's parameters as
///     data. Members declared outside NumSharp (System.Object's ToString/GetHashCode when not overridden) are left
///     out, so a protocol is never credited to .NET's defaults.
/// </summary>
/// <param name="type">The type to reflect.</param>
/// <param name="flags">
///     The binding flags: <c>DeclaredOnly</c> for module facades (their members are their own), or
///     <c>FlattenHierarchy</c> for the object inventory (inherited members and a base type's static operators too).
/// </param>
/// <returns>The type's inventory, every collection ordered by name for deterministic output.</returns>
static TypeInventory InspectType(Type type, BindingFlags flags)
{
    var methods = type.GetMethods(flags)
        .Where(method => !method.IsSpecialName && method.DeclaringType?.Assembly == type.Assembly)
        .GroupBy(method => method.Name, StringComparer.Ordinal)
        .Select(group => new ApiMember(
            group.Key,
            "method",
            group.Select(FormatMethod).OrderBy(value => value, StringComparer.Ordinal).ToArray(),
            group.All(IsObsolete),
            group.All(method => method.IsStatic),
            DeclaringTypes(group))
        {
            // Schema 7: the overloads as data, in the same order as Signatures, for the generator's
            // signature-parity pass (coverage/signature_parity.py).
            Overloads = group.Select(DescribeOverload).OrderBy(overload => overload.Signature, StringComparer.Ordinal).ToArray(),
        })
        .OrderBy(member => member.Name, StringComparer.Ordinal)
        .ToArray();

    var properties = type.GetProperties(flags)
        .Where(property => property.DeclaringType?.Assembly == type.Assembly)
        .GroupBy(property => property.Name, StringComparer.Ordinal)
        .Select(group => new ApiMember(
            group.Key,
            "property",
            group.Select(FormatProperty).OrderBy(value => value, StringComparer.Ordinal).ToArray(),
            group.All(IsObsolete),
            group.All(IsStaticProperty),
            DeclaringTypes(group)))
        .OrderBy(member => member.Name, StringComparer.Ordinal)
        .ToArray();

    var fields = type.GetFields(flags)
        .Where(field => field.DeclaringType?.Assembly == type.Assembly)
        .GroupBy(field => field.Name, StringComparer.Ordinal)
        .Select(group => new ApiMember(
            group.Key,
            "field",
            group.Select(FormatField).OrderBy(value => value, StringComparer.Ordinal).ToArray(),
            group.All(IsObsolete),
            group.All(field => field.IsStatic),
            DeclaringTypes(group)))
        .OrderBy(member => member.Name, StringComparer.Ordinal)
        .ToArray();

    // Operators are the special-name methods the `methods` list above excludes. Only C# operator and
    // conversion methods qualify (their CLR names all start with "op_"); property accessors and event
    // methods are special-name too and stay excluded. With FlattenHierarchy (the object inventory) a
    // base type's static operators appear as well, so every overload records its declaring type and its
    // operand types: the generator credits a protocol only when the OWNER type itself is an operand.
    var operators = type.GetMethods(flags)
        .Where(method => method.IsSpecialName && method.Name.StartsWith("op_", StringComparison.Ordinal)
                         && method.DeclaringType?.Assembly == type.Assembly)
        .GroupBy(method => method.Name, StringComparer.Ordinal)
        .Select(group => new ApiOperator(
            group.Key,
            group.Select(FormatOperatorOverload)
                .OrderBy(overload => overload.Signature, StringComparer.Ordinal)
                .ToArray()))
        .OrderBy(op => op.Name, StringComparer.Ordinal)
        .ToArray();

    return new TypeInventory(type.FullName ?? type.Name, methods, properties, fields, operators);
}

/// <summary>
///     Describes one operator or conversion overload with the structure the generator matches on:
///     the result type, each operand's type and the type that declares the overload.
/// </summary>
/// <param name="method">An <c>op_*</c> special-name method (an operator or a conversion).</param>
/// <returns>
///     The overload's signature text (the same format as <see cref="FormatMethod" />), its result type and
///     its operand types in <see cref="FriendlyType" /> spelling (a nested type keeps the CLR <c>+</c>), and
///     the full name of the declaring type, which differs from the inspected type for an operator
///     inherited through <see cref="BindingFlags.FlattenHierarchy" />.
/// </returns>
static OperatorOverload FormatOperatorOverload(MethodInfo method) => new(
    FormatMethod(method),
    FriendlyType(method.ReturnType),
    method.GetParameters().Select(parameter => FriendlyType(parameter.ParameterType)).ToArray(),
    method.DeclaringType!.FullName ?? method.DeclaringType.Name);

/// <summary>
///     Describes one method overload with what decides how a ported NumPy call binds to it: its parameters in
///     declaration order (with defaults, <c>params</c> and by-ref modifiers), its generic parameters and the
///     attributes that change overload resolution or visibility.
/// </summary>
/// <param name="method">A public, non-special-name method.</param>
/// <returns>
///     The overload. <see cref="MethodOverload.Signature" /> is the same text as the member's
///     <see cref="ApiMember.Signatures" /> entry, so the two lists can be read side by side.
/// </returns>
/// <remarks>
///     <c>[OverloadResolutionPriority]</c> is matched by attribute NAME: NumSharp.Core polyfills it for net8.0, so its
///     CLR type is not the BCL's on every target. The priority matters to the generator because C# 12 (the default
///     language version of a net8.0 project) ignores it, so an overload set that resolves only through it is
///     ambiguous for those callers.
/// </remarks>
static MethodOverload DescribeOverload(MethodInfo method)
{
    int? priority = method.GetCustomAttributesData()
        .Where(attribute => attribute.AttributeType.Name == "OverloadResolutionPriorityAttribute"
                            && attribute.ConstructorArguments.Count == 1)
        .Select(attribute => (int?)Convert.ToInt32(attribute.ConstructorArguments[0].Value, CultureInfo.InvariantCulture))
        .FirstOrDefault();
    var browsable = method.GetCustomAttribute<System.ComponentModel.EditorBrowsableAttribute>();
    return new MethodOverload(
        FormatMethod(method),
        FriendlyType(method.ReturnType),
        method.GetParameters().Select(DescribeParameter).ToArray(),
        method.IsGenericMethodDefinition ? method.GetGenericArguments().Select(argument => argument.Name).ToArray() : [],
        priority,
        IsObsolete(method),
        browsable?.State == System.ComponentModel.EditorBrowsableState.Never,
        method.IsStatic,
        method.DeclaringType!.FullName ?? method.DeclaringType.Name);
}

/// <summary>
///     Describes one parameter for the generator's comparison with NumPy's signature: name, type, whether a ported
///     call may omit it, its default as C# text and as a kind, and its <c>params</c> / by-ref modifiers.
/// </summary>
/// <param name="parameter">A method parameter.</param>
/// <returns>
///     The parameter. <see cref="ApiParameter.DefaultKind" /> is <c>null</c>, <c>"null"</c> (a null default, or a
///     struct's <c>default</c>, which reflection also reports as null), <c>"bool"</c>, <c>"number"</c>,
///     <c>"string"</c>, <c>"char"</c>, <c>"enum"</c> or <c>"other"</c>; <see cref="ApiParameter.Default" /> is the
///     literal (<c>"default"</c> for a struct's default, so it is not read as a null reference).
/// </returns>
static ApiParameter DescribeParameter(ParameterInfo parameter)
{
    var parameterType = parameter.ParameterType.IsByRef ? parameter.ParameterType.GetElementType()! : parameter.ParameterType;
    string? kind = null;
    string? text = null;
    if (parameter.HasDefaultValue)
    {
        var value = parameter.DefaultValue;
        kind = value is null || value == DBNull.Value || value == Missing.Value ? "null"
            : value is bool ? "bool"
            : value is string ? "string"
            : value is char ? "char"
            : value.GetType().IsEnum ? "enum"
            : value.GetType().IsPrimitive || value is decimal ? "number"
            : "other";
        // A non-nullable struct parameter declared `= default` has no constant: reflection reports null. Its C#
        // spelling is `default`, and it means "not given" exactly like a null reference default.
        text = kind == "null" && parameterType.IsValueType && Nullable.GetUnderlyingType(parameterType) is null
            ? "default"
            : FormatDefault(value);
    }

    return new ApiParameter(
        parameter.Name ?? string.Empty,
        FriendlyType(parameterType),
        parameter.HasDefaultValue || parameter.IsOptional,
        text,
        kind,
        parameter.GetCustomAttribute<ParamArrayAttribute>() is not null,
        parameter.IsOut ? "out" : parameter.ParameterType.IsByRef ? (parameter.IsIn ? "in" : "ref") : null);
}

static string[] DeclaringTypes(IEnumerable<MemberInfo> members) => members
    .Select(member => member.DeclaringType!.FullName ?? member.DeclaringType.Name)
    .Distinct(StringComparer.Ordinal).OrderBy(name => name, StringComparer.Ordinal).ToArray();

static bool IsObsolete(MemberInfo member) => member.GetCustomAttribute<ObsoleteAttribute>() is not null;

static bool IsStaticProperty(PropertyInfo property) => (property.GetMethod ?? property.SetMethod)!.IsStatic;

/// <summary>
///     Guard: DeclaredOnly reflection sees nothing inherited, so an annotated module must sit on a
///     hierarchy where that loses nothing — its base is object/ValueType or itself annotated.
///     Without this, giving a facade a base class silently hides every inherited public member.
/// </summary>
static void GuardHierarchy(IReadOnlyDictionary<Type, string> annotatedTypes)
{
    foreach (var (type, moduleName) in annotatedTypes)
    {
        var baseType = type.BaseType;
        if (baseType is null || baseType == typeof(object) || baseType == typeof(ValueType)
            || annotatedTypes.ContainsKey(baseType))
            continue;
        throw new InvalidOperationException(
            $"[ModuleName(\"{moduleName}\")] type {type.FullName} derives from {baseType.FullName}, whose public " +
            "members a DeclaredOnly scan will not see. Annotate the base type or flatten the hierarchy.");
    }
}

/// <summary>
///     Guard: a type SHAPED like a module facade must be annotated. Two shapes are checked, both
///     keyed on NumPy-style lowercase function names so C# infrastructure (UnmanagedStorage,
///     TensorEngine) never false-positives: a property on an annotated host returning a concrete
///     assembly class with many lowercase instance methods (the FourierModule shape — exactly how
///     np.fft went missing), and a public nested static class with lowercase static methods (the
///     np.linalg shape). Result structs, DSL indexer classes (r_/c_/mgrid), iterators, delegates
///     and abstract seams all fall outside both shapes by construction.
/// </summary>
static void GuardFacadeShapes(Assembly assembly, IReadOnlyDictionary<Type, string> annotatedTypes)
{
    // NumPyRandom hosts 48 lowercase functions, FourierModule 18, np.linalg 31; the closest
    // non-modules are FlatIterator (copy) and the DSL classes (none). The threshold only needs to
    // split those regimes — and the generator's stray-host gate backstops anything that slips.
    const int FacadeFunctionThreshold = 8;
    const BindingFlags DeclaredMembers =
        BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly;

    static int LowercaseFunctionCount(Type type, BindingFlags flags)
        => type.GetMethods(flags).Count(m => !m.IsSpecialName && char.IsLower(m.Name[0]));

    foreach (var (host, moduleName) in annotatedTypes)
    {
        foreach (var property in host.GetProperties(DeclaredMembers))
        {
            var returned = property.PropertyType;
            if (returned.Assembly != assembly || !returned.IsClass || returned.IsAbstract)
                continue;
            if (annotatedTypes.ContainsKey(returned) || typeof(Delegate).IsAssignableFrom(returned))
                continue;
            var functions = LowercaseFunctionCount(returned, BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);
            if (functions >= FacadeFunctionThreshold)
                throw new InvalidOperationException(
                    $"{host.Name}.{property.Name} returns {returned.FullName}, which declares {functions} public " +
                    "lowercase instance methods — module-facade shaped. Annotate it with [ModuleName(\"...\")] " +
                    "so the API inventory cannot miss its surface.");
        }

        foreach (var nested in host.GetNestedTypes(BindingFlags.Public))
        {
            if (!(nested.IsAbstract && nested.IsSealed) || nested.Name.StartsWith('_') || annotatedTypes.ContainsKey(nested))
                continue;
            var functions = LowercaseFunctionCount(nested, BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly);
            if (functions > 0)
                throw new InvalidOperationException(
                    $"{nested.FullName} is a public nested static class of [ModuleName(\"{moduleName}\")] host " +
                    $"{host.Name} with {functions} public lowercase static methods — module-facade shaped. " +
                    "Annotate it with [ModuleName(\"...\")] or prefix its name with '_'.");
        }
    }
}

/// <summary>
///     The public surface OUTSIDE the annotated modules: every other exported type's public declared
///     member names. This is the generator's stray-host index — the data behind "this NumPy export is
///     'missing', yet its name exists on an unannotated type".
/// </summary>
static SortedDictionary<string, string[]> BuildUnannotatedSurface(Assembly assembly, IEnumerable<Type> annotatedTypes)
{
    var annotated = new HashSet<Type>(annotatedTypes);
    const BindingFlags DeclaredMembers =
        BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly;

    var surface = new SortedDictionary<string, string[]>(StringComparer.Ordinal);
    foreach (var type in assembly.GetExportedTypes())
    {
        if (annotated.Contains(type) || type.IsEnum)
            continue;
        var members = type.GetMethods(DeclaredMembers).Where(m => !m.IsSpecialName).Select(m => m.Name)
            .Concat(type.GetProperties(DeclaredMembers).Select(p => p.Name))
            .Concat(type.GetFields(DeclaredMembers).Select(f => f.Name))
            .Distinct(StringComparer.Ordinal)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();
        if (members.Length > 0)
            surface.Add(type.FullName ?? type.Name, members);
    }

    return surface;
}

static string FormatMethod(MethodInfo method)
{
    var parameters = string.Join(", ", method.GetParameters().Select(FormatParameter));
    return $"{FriendlyType(method.ReturnType)} {method.Name}({parameters})";
}

static string FormatProperty(PropertyInfo property)
{
    var access = property.CanRead && property.CanWrite ? "get; set;" : property.CanRead ? "get;" : "set;";
    var indexes = property.GetIndexParameters();
    return indexes.Length == 0
        ? $"{FriendlyType(property.PropertyType)} {property.Name} {{ {access} }}"
        : $"{FriendlyType(property.PropertyType)} {property.Name}[{string.Join(", ", indexes.Select(FormatParameter))}] {{ {access} }}";
}

static string FormatField(FieldInfo field)
{
    var modifier = field.IsLiteral ? "const " : field.IsInitOnly ? "readonly " : string.Empty;
    return $"{modifier}{FriendlyType(field.FieldType)} {field.Name}";
}

static string FormatParameter(ParameterInfo parameter)
{
    var prefix = parameter.GetCustomAttribute<ParamArrayAttribute>() is not null
        ? "params "
        : parameter.IsOut
            ? "out "
            : parameter.ParameterType.IsByRef
                ? "ref "
                : string.Empty;
    var type = FriendlyType(parameter.ParameterType.IsByRef
        ? parameter.ParameterType.GetElementType()!
        : parameter.ParameterType);
    var optional = parameter.HasDefaultValue ? $" = {FormatDefault(parameter.DefaultValue)}" : string.Empty;
    return $"{prefix}{type} {parameter.Name}{optional}";
}

static string FormatDefault(object? value)
{
    if (value is null || value == DBNull.Value || value == Missing.Value)
        return "null";
    if (value is string text)
        return JsonSerializer.Serialize(text);
    if (value is char character)
        return $"'{character}'";
    if (value is bool boolean)
        return boolean ? "true" : "false";
    if (value.GetType().IsEnum)
        return $"{FriendlyType(value.GetType())}.{value}";
    return Convert.ToString(value, CultureInfo.InvariantCulture) ?? "null";
}

static string FriendlyType(Type type)
{
    if (type.IsArray)
        return $"{FriendlyType(type.GetElementType()!)}[]";
    if (type.IsGenericParameter)
        return type.Name;

    var nullable = Nullable.GetUnderlyingType(type);
    if (nullable is not null)
        return $"{FriendlyType(nullable)}?";

    var aliases = new Dictionary<Type, string>
    {
        [typeof(void)] = "void",
        [typeof(bool)] = "bool",
        [typeof(byte)] = "byte",
        [typeof(sbyte)] = "sbyte",
        [typeof(short)] = "short",
        [typeof(ushort)] = "ushort",
        [typeof(int)] = "int",
        [typeof(uint)] = "uint",
        [typeof(long)] = "long",
        [typeof(ulong)] = "ulong",
        [typeof(float)] = "float",
        [typeof(double)] = "double",
        [typeof(decimal)] = "decimal",
        [typeof(char)] = "char",
        [typeof(string)] = "string",
        [typeof(object)] = "object"
    };
    if (aliases.TryGetValue(type, out var alias))
        return alias;

    if (!type.IsGenericType)
        return type.FullName ?? type.Name;

    var name = (type.GetGenericTypeDefinition().FullName ?? type.Name).Split('`')[0];
    return $"{name}<{string.Join(", ", type.GetGenericArguments().Select(FriendlyType))}>";
}

/// <summary>
///     The tool's whole output: NumSharp.Core's public surface as the coverage generator reads it. The generator
///     refuses any other <see cref="SchemaVersion" />, because an older schema silently lacks data it relies on
///     (schema 6 the operators that credit Python protocols, schema 7 the overloads behind signature parity).
/// </summary>
/// <param name="SchemaVersion">The output format's version (7 since the overloads became data).</param>
/// <param name="AssemblyVersion">NumSharp.Core's assembly version, printed in the artifact.</param>
/// <param name="Modules">The <c>[ModuleName]</c>-annotated module facades, keyed by module name.</param>
/// <param name="UnannotatedSurface">Every other exported type's public member names (the stray-host index).</param>
/// <param name="ExportedTypes">Every exported public type's full name (the type-match index).</param>
/// <param name="ObjectTypes">Every exported type's members, inherited ones included (the object-API index).</param>
internal sealed record ApiInventory(
    int SchemaVersion,
    string AssemblyVersion,
    // Keyed by [ModuleName] value ("np", "ndarray", "np.random", "np.fft", "np.linalg", ...), ordinal
    // order for deterministic output. Dictionary KEYS bypass the camelCase policy, so the dotted
    // module names survive verbatim.
    IReadOnlyDictionary<string, TypeInventory> Modules,
    // Every OTHER exported type's public member names — the stray-host index the generator checks
    // still-missing NumPy exports against.
    IReadOnlyDictionary<string, string[]> UnannotatedSurface,
    // Every exported public type's full name — the type-match index behind crediting a NumPy CLASS
    // export against a NumSharp type of the same name (numpy.random.Generator -> NumSharp.Generator).
    IReadOnlyList<string> ExportedTypes,
    // Exact CLR full names, with inherited public members and declaration ownership for source links.
    IReadOnlyDictionary<string, TypeInventory> ObjectTypes);

/// <summary>One reflected type: its public members grouped by member kind.</summary>
/// <param name="Type">The type's CLR full name (a nested type keeps <c>+</c>, as in <c>NumSharp.np+FlatIterator</c>).</param>
/// <param name="Methods">Public methods, one entry per name with every overload's signature.</param>
/// <param name="Properties">Public properties, including indexers (the C# name <c>Item</c>).</param>
/// <param name="Fields">Public fields and constants.</param>
/// <param name="Operators">
///     C# operator and conversion methods (<c>op_Addition</c>, <c>op_Explicit</c>, ...), one entry per CLR
///     name; empty for a type without operators. Added in schema 6 so the generator can credit NumPy's
///     Python operator protocols, which a C# operator spells with the same symbol.
/// </param>
internal sealed record TypeInventory(
    string Type,
    ApiMember[] Methods,
    ApiMember[] Properties,
    ApiMember[] Fields,
    ApiOperator[] Operators);

/// <summary>Every overload of one C# operator or conversion on a type.</summary>
/// <param name="Name">The CLR special name, such as <c>op_Addition</c> or <c>op_Explicit</c>.</param>
/// <param name="Overloads">The overloads, ordered by signature text for deterministic output.</param>
internal sealed record ApiOperator(
    string Name,
    OperatorOverload[] Overloads);

/// <summary>One operator or conversion overload, structured for operand matching.</summary>
/// <param name="Signature">The overload as text, in the format of every other method signature.</param>
/// <param name="ReturnType">The result type; for a conversion, the type converted TO.</param>
/// <param name="ParameterTypes">
///     The operand types in order: two for a binary operator (left, right), one for a unary operator or a
///     conversion (the type converted FROM).
/// </param>
/// <param name="DeclaringType">
///     The full name of the type that declares the overload; it differs from the inspected type when the
///     operator is inherited through <see cref="BindingFlags.FlattenHierarchy" />.
/// </param>
internal sealed record OperatorOverload(
    string Signature,
    string ReturnType,
    string[] ParameterTypes,
    string DeclaringType);

/// <summary>
///     One public member NAME of a type with every declaration behind it: what the generator credits a NumPy API to.
///     Crediting is by name; whether the member's parameters are NumPy's is read from <see cref="Overloads" />.
/// </summary>
/// <param name="Name">The member name, case-sensitive like NumPy's.</param>
/// <param name="Kind"><c>method</c>, <c>property</c> or <c>field</c>.</param>
/// <param name="Signatures">Every declaration as display text, ordered by text.</param>
/// <param name="Obsolete">Whether EVERY declaration is <c>[Obsolete]</c>.</param>
/// <param name="Static">Whether EVERY declaration is static.</param>
/// <param name="DeclaringTypes">The full names of the types that declare the member (several for inherited members).</param>
internal sealed record ApiMember(
    string Name,
    string Kind,
    string[] Signatures,
    bool Obsolete,
    // True when EVERY declaration behind this name is static — meaningful on instance facades, where
    // a static helper would otherwise be indistinguishable from the module's functions.
    bool Static,
    string[] DeclaringTypes)
{
    /// <summary>
    ///     Schema 7: every overload of a METHOD as data, ordered like <see cref="Signatures" />; null (and so
    ///     omitted from the JSON) for properties and fields. The generator compares these parameters with NumPy's
    ///     signature, which display text cannot support.
    /// </summary>
    public MethodOverload[]? Overloads { get; init; }
}

/// <summary>One method overload, structured for the generator's signature-parity comparison.</summary>
/// <param name="Signature">The overload as text, identical to its <see cref="ApiMember.Signatures" /> entry.</param>
/// <param name="ReturnType">The result type in <c>FriendlyType</c> spelling.</param>
/// <param name="Parameters">The parameters in declaration order.</param>
/// <param name="GenericParameters">The method's own generic parameter names (<c>T</c> of <c>array&lt;T&gt;</c>); empty when none.</param>
/// <param name="OverloadResolutionPriority">The <c>[OverloadResolutionPriority]</c> value, or null without the attribute.</param>
/// <param name="Obsolete">Whether the overload is <c>[Obsolete]</c>.</param>
/// <param name="Hidden">Whether the overload is <c>[EditorBrowsable(Never)]</c>.</param>
/// <param name="Static">Whether the overload is static (an instance method of an object API binds as <c>a.name(...)</c>).</param>
/// <param name="DeclaringType">The full name of the declaring type (a base type for an inherited member).</param>
internal sealed record MethodOverload(
    string Signature,
    string ReturnType,
    ApiParameter[] Parameters,
    string[] GenericParameters,
    int? OverloadResolutionPriority,
    bool Obsolete,
    bool Hidden,
    bool Static,
    string DeclaringType);

/// <summary>One parameter of a method overload.</summary>
/// <param name="Name">The parameter name (what a C# named argument, NumPy's keyword, must spell).</param>
/// <param name="Type">The parameter type in <c>FriendlyType</c> spelling, by-ref element type for <c>out</c>/<c>ref</c>/<c>in</c>.</param>
/// <param name="Optional">Whether a call may omit it (it has a default).</param>
/// <param name="Default">The default as C# text (<c>null</c>, <c>default</c>, <c>"\n"</c>, <c>'C'</c>, <c>1E-08</c>, ...); null without a default.</param>
/// <param name="DefaultKind">The default's kind (see <c>DescribeParameter</c>); null without a default.</param>
/// <param name="IsParams">Whether it is a <c>params</c> array (NumPy's <c>*args</c> counterpart).</param>
/// <param name="ByRef"><c>"out"</c>, <c>"ref"</c> or <c>"in"</c> for a by-reference parameter; null otherwise.</param>
internal sealed record ApiParameter(
    string Name,
    string Type,
    bool Optional,
    string? Default,
    string? DefaultKind,
    bool IsParams,
    string? ByRef);
