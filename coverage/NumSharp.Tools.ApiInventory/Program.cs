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
var modules = new SortedDictionary<string, TypeInventory>(StringComparer.Ordinal);
foreach (var type in assembly.GetExportedTypes())
{
    var module = type.GetCustomAttribute<ModuleNameAttribute>(inherit: false);
    if (module is null)
        continue;
    if (modules.TryGetValue(module.Name, out var taken))
        throw new InvalidOperationException(
            $"Duplicate [ModuleName(\"{module.Name}\")]: both {taken.Type} and {type.FullName} claim it.");

    // A static class hosts static functions (np, np.linalg); an instance facade hosts instance
    // methods reached through its singleton property (np.random, np.fft) or the array itself.
    var isStaticClass = type.IsAbstract && type.IsSealed;
    var flags = BindingFlags.Public | BindingFlags.DeclaredOnly
                | (isStaticClass ? BindingFlags.Static : BindingFlags.Instance);
    modules.Add(module.Name, InspectType(type, flags));
}

if (modules.Count == 0)
    throw new InvalidOperationException(
        $"No [ModuleName]-annotated types found in {assembly.GetName().Name} — the inventory would be empty.");

var inventory = new ApiInventory(
    SchemaVersion: 2,
    AssemblyVersion: assembly.GetName().Version?.ToString() ?? "unknown",
    Modules: modules);

Console.WriteLine(JsonSerializer.Serialize(inventory, new JsonSerializerOptions
{
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    WriteIndented = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
}));

static TypeInventory InspectType(Type type, BindingFlags flags)
{
    var methods = type.GetMethods(flags)
        .Where(method => !method.IsSpecialName)
        .GroupBy(method => method.Name, StringComparer.Ordinal)
        .Select(group => new ApiMember(
            group.Key,
            "method",
            group.Select(FormatMethod).OrderBy(value => value, StringComparer.Ordinal).ToArray(),
            group.All(IsObsolete)))
        .OrderBy(member => member.Name, StringComparer.Ordinal)
        .ToArray();

    var properties = type.GetProperties(flags)
        .GroupBy(property => property.Name, StringComparer.Ordinal)
        .Select(group => new ApiMember(
            group.Key,
            "property",
            group.Select(FormatProperty).OrderBy(value => value, StringComparer.Ordinal).ToArray(),
            group.All(IsObsolete)))
        .OrderBy(member => member.Name, StringComparer.Ordinal)
        .ToArray();

    var fields = type.GetFields(flags)
        .GroupBy(field => field.Name, StringComparer.Ordinal)
        .Select(group => new ApiMember(
            group.Key,
            "field",
            group.Select(FormatField).OrderBy(value => value, StringComparer.Ordinal).ToArray(),
            group.All(IsObsolete)))
        .OrderBy(member => member.Name, StringComparer.Ordinal)
        .ToArray();

    return new TypeInventory(type.FullName ?? type.Name, methods, properties, fields);
}

static bool IsObsolete(MemberInfo member) => member.GetCustomAttribute<ObsoleteAttribute>() is not null;

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

internal sealed record ApiInventory(
    int SchemaVersion,
    string AssemblyVersion,
    // Keyed by [ModuleName] value ("np", "ndarray", "np.random", "np.fft", "np.linalg", ...), ordinal
    // order for deterministic output. Dictionary KEYS bypass the camelCase policy, so the dotted
    // module names survive verbatim.
    IReadOnlyDictionary<string, TypeInventory> Modules);

internal sealed record TypeInventory(
    string Type,
    ApiMember[] Methods,
    ApiMember[] Properties,
    ApiMember[] Fields);

internal sealed record ApiMember(
    string Name,
    string Kind,
    string[] Signatures,
    bool Obsolete);
