#:project ../../src/NumSharp.Core
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

// Dumps the public surface of EVERY NumPy module NumSharp exposes — each type annotated [ModuleName("...")]: np,
// ndarray, np.linalg, np.fft, np.random, np.ma, np.emath, np.char, np.strings, np.dtypes, np.typing, np.polynomial
// and its bases — as structured JSON, so signature_diff.py can compare every overload with NumPy 2.4.2's
// inspect.signature. The module list is DISCOVERED (the ApiInventory rule), never hardcoded: a new facade joins the
// comparison by its annotation alone. One object API is added by hand: np.random.Generator (see below).
//
// Output (one object per module, keyed by the [ModuleName] value):
//   { "np.linalg": { "type": "NumSharp.np+linalg",
//                    "methods": { "norm": [ { "static": true, "generic": [], "returns": "NDArray",
//                                             "orpa": null, "obsolete": null, "hidden": false,
//                                             "params": [ { "name": "x", "type": "NumSharp.NDArray",
//                                                           "optional": false, "default": null,
//                                                           "defaultKind": null, "params": false,
//                                                           "byRef": null } ... ] } ... ] },
//                    "properties": { "name": "type" }, "fields": { "name": "type" },
//                    "types": [ "nested type name" ] } }
//
// Usage (from this directory): dotnet run --no-cache dump_modules.cs <output.json> [<types.json>]
// The optional second file holds EVERY exported type (DescribeAnyType): the object APIs, NDArray<T>, constructors and
// extension methods the facade dump does not reach.
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using NumSharp;

var assembly = typeof(np).Assembly;
var modules = new SortedDictionary<string, object>(StringComparer.Ordinal);
foreach (var type in assembly.GetExportedTypes())
{
    var module = type.GetCustomAttribute<ModuleNameAttribute>(inherit: false);
    if (module is null)
        continue;
    if (modules.ContainsKey(module.Name))
        throw new InvalidOperationException($"Two types claim [ModuleName(\"{module.Name}\")]");
    modules.Add(module.Name, DescribeType(type));
}

if (modules.Count == 0)
    throw new InvalidOperationException("No [ModuleName]-annotated type found: the dump would be empty.");

// Object APIs that NumPy users call like module functions, keyed by the NumPy path of their class. They carry no
// [ModuleName] (they are classes, not modules), so they are listed here: np.random.default_rng() returns a
// Generator, whose methods are NumPy's modern random API. RandomState needs no entry: NumSharp's np.random facade IS
// RandomState (np.random.RandomState(seed) returns the same class), so its methods are already the np.random module.
foreach (var (numpyPath, type) in new[] { ("np.random.Generator", typeof(Generator)) })
{
    if (!modules.TryAdd(numpyPath, DescribeType(type)))
        throw new InvalidOperationException($"{numpyPath} is both an object API and a [ModuleName] value.");
}

string outPath = args.Length > 0 ? args[0] : "modules_dump.json";
File.WriteAllText(outPath, JsonSerializer.Serialize(modules, new JsonSerializerOptions { WriteIndented = false }));
Console.WriteLine($"{modules.Count} modules -> {outPath}");

// Second output (optional): EVERY exported type, so the comparison can reach what the module facades do not hold —
// the classes behind NumPy's object APIs (NDMaskedArray = numpy.ma.MaskedArray, NDMatrix = numpy.matrix, DType =
// numpy.dtype, the bit generators, ...), the typed NDArray<T>, constructors (NumPy's class call), and extension
// methods, which take part in `a.name(...)` binding exactly like instance methods of the type they extend.
if (args.Length > 1)
{
    var types = new SortedDictionary<string, object>(StringComparer.Ordinal);
    foreach (var type in assembly.GetExportedTypes())
        types[type.FullName ?? type.Name] = DescribeAnyType(type);
    File.WriteAllText(args[1], JsonSerializer.Serialize(types, new JsonSerializerOptions { WriteIndented = false }));
    Console.WriteLine($"{types.Count} exported types -> {args[1]}");
}

/// <summary>
///     Reflects one module facade: its public methods (DeclaredOnly — a module's functions are its own; the ApiInventory
///     guards that annotated hosts never hide members in a base class) grouped by name with every overload, and the
///     names and types of its public properties and fields (NumPy's attributes and constants).
/// </summary>
/// <param name="type">The [ModuleName]-annotated type.</param>
/// <returns>A JSON-serializable dictionary: <c>type</c>, <c>methods</c>, <c>properties</c>, <c>fields</c>,
///     <c>types</c> (the public nested types' names).</returns>
static Dictionary<string, object> DescribeType(Type type)
{
    const BindingFlags flags = BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly;
    var methods = type.GetMethods(flags)
        .Where(m => !m.IsSpecialName)
        .GroupBy(m => m.Name, StringComparer.Ordinal)
        .OrderBy(g => g.Key, StringComparer.Ordinal)
        .ToDictionary(g => g.Key, g => g.Select(DescribeMethod).ToList(), StringComparer.Ordinal);
    var properties = type.GetProperties(flags)
        .Where(p => p.GetIndexParameters().Length == 0)
        .GroupBy(p => p.Name, StringComparer.Ordinal)
        .OrderBy(g => g.Key, StringComparer.Ordinal)
        .ToDictionary(g => g.Key, g => (object)Friendly(g.First().PropertyType), StringComparer.Ordinal);
    var fields = type.GetFields(flags)
        .GroupBy(f => f.Name, StringComparer.Ordinal)
        .OrderBy(g => g.Key, StringComparer.Ordinal)
        .ToDictionary(g => g.Key, g => (object)Friendly(g.First().FieldType), StringComparer.Ordinal);
    // Public nested types count as names of the module (np.typing.NBitBase is a nested class, NumPy's a class
    // attribute); without them such a NumPy name would be reported absent.
    var nested = type.GetNestedTypes(BindingFlags.Public)
        .Select(t => t.Name.Split('`')[0])
        .Distinct(StringComparer.Ordinal)
        .OrderBy(n => n, StringComparer.Ordinal)
        .ToArray();
    return new Dictionary<string, object>
    {
        ["type"] = type.FullName ?? type.Name,
        ["methods"] = methods,
        ["properties"] = properties,
        ["fields"] = fields,
        ["types"] = nested,
    };
}

/// <summary>
///     Any exported type, for the second output: its module-style description (<see cref="DescribeType"/>) plus what
///     an object API needs — the kind, the base type (to walk inherited members inside NumSharp), whether it is a
///     static class (an extension-method host), its generic parameters and its public constructors (NumPy's class
///     call, <c>np.random.PCG64(seed)</c> against <c>new PCG64(seed)</c>).
/// </summary>
/// <param name="type">An exported type.</param>
/// <returns>A JSON-serializable dictionary: the keys of <see cref="DescribeType"/> plus <c>kind</c>, <c>base</c>,
///     <c>static</c>, <c>genericParams</c> and <c>ctors</c>.</returns>
static Dictionary<string, object> DescribeAnyType(Type type)
{
    var d = DescribeType(type);
    d["kind"] = type.IsEnum ? "enum" : type.IsInterface ? "interface" : typeof(Delegate).IsAssignableFrom(type) ? "delegate"
        : type.IsValueType ? "struct" : "class";
    d["base"] = type.BaseType is null || type.BaseType == typeof(object) || type.BaseType == typeof(ValueType)
        ? null! : (type.BaseType.IsGenericType ? type.BaseType.GetGenericTypeDefinition().FullName : type.BaseType.FullName)!;
    d["static"] = type.IsAbstract && type.IsSealed;
    d["genericParams"] = type.IsGenericTypeDefinition ? type.GetGenericArguments().Select(a => a.Name).ToArray() : Array.Empty<string>();
    // An abstract class cannot be constructed, so its constructors are not part of what a caller can write.
    d["ctors"] = type.IsAbstract || type.IsEnum
        ? new List<Dictionary<string, object?>>()
        : type.GetConstructors(BindingFlags.Public | BindingFlags.Instance).Select(DescribeConstructor).ToList();
    return d;
}

/// <summary>One public constructor, described like a method overload (no return type, never generic or static).</summary>
/// <param name="c">The constructor.</param>
/// <returns>A JSON-serializable dictionary with the keys of <see cref="DescribeMethod"/>.</returns>
static Dictionary<string, object?> DescribeConstructor(ConstructorInfo c)
{
    int? orpa = c.GetCustomAttributesData()
        .Where(a => a.AttributeType.Name == "OverloadResolutionPriorityAttribute" && a.ConstructorArguments.Count == 1)
        .Select(a => (int?)Convert.ToInt32(a.ConstructorArguments[0].Value, CultureInfo.InvariantCulture))
        .FirstOrDefault();
    var obsolete = c.GetCustomAttribute<ObsoleteAttribute>();
    var browsable = c.GetCustomAttribute<EditorBrowsableAttribute>();
    return new Dictionary<string, object?>
    {
        ["static"] = false,
        ["generic"] = Array.Empty<string>(),
        ["returns"] = Friendly(c.DeclaringType!),
        ["orpa"] = orpa,
        ["obsolete"] = obsolete is null ? null : new Dictionary<string, object?> { ["error"] = obsolete.IsError, ["message"] = obsolete.Message },
        ["hidden"] = browsable?.State == EditorBrowsableState.Never,
        ["extension"] = false,
        ["extends"] = null,
        ["params"] = c.GetParameters().Select(DescribeParameter).ToList(),
    };
}

/// <summary>
///     One overload: static or instance, generic type parameters, return type, the attributes that decide how C# binds
///     it (<c>[OverloadResolutionPriority]</c>, which C# 12 ignores; <c>[Obsolete]</c>; <c>[EditorBrowsable(Never)]</c>),
///     whether it is an extension method (and the type its first parameter extends) and every parameter.
/// </summary>
/// <param name="m">The method.</param>
/// <returns>A JSON-serializable dictionary.</returns>
static Dictionary<string, object?> DescribeMethod(MethodInfo m)
{
    // The priority attribute is matched by NAME: NumSharp polyfills it for net8.0, so its CLR type differs per TFM.
    int? orpa = m.GetCustomAttributesData()
        .Where(a => a.AttributeType.Name == "OverloadResolutionPriorityAttribute" && a.ConstructorArguments.Count == 1)
        .Select(a => (int?)Convert.ToInt32(a.ConstructorArguments[0].Value, CultureInfo.InvariantCulture))
        .FirstOrDefault();
    var obsolete = m.GetCustomAttribute<ObsoleteAttribute>();
    var browsable = m.GetCustomAttribute<EditorBrowsableAttribute>();
    // An extension method binds as `x.name(...)` on the type of its first parameter, so it is one more overload of
    // that type's member of the same name.
    bool extension = m.IsDefined(typeof(System.Runtime.CompilerServices.ExtensionAttribute), inherit: false);
    var parameters = m.GetParameters();
    return new Dictionary<string, object?>
    {
        ["static"] = m.IsStatic,
        ["generic"] = m.IsGenericMethodDefinition ? m.GetGenericArguments().Select(a => a.Name).ToArray() : Array.Empty<string>(),
        ["returns"] = Friendly(m.ReturnType),
        ["orpa"] = orpa,
        ["obsolete"] = obsolete is null ? null : new Dictionary<string, object?> { ["error"] = obsolete.IsError, ["message"] = obsolete.Message },
        ["hidden"] = browsable?.State == EditorBrowsableState.Never,
        ["extension"] = extension,
        ["extends"] = extension && parameters.Length > 0 ? ExtendedTypeName(parameters[0].ParameterType) : null,
        ["params"] = parameters.Select(DescribeParameter).ToList(),
    };
}

/// <summary>The full name of the type an extension method's first parameter extends (the generic definition for a
///     constructed generic type, the element for a by-reference parameter), so it can be matched to a dumped type.</summary>
/// <param name="t">The first parameter's type.</param>
/// <returns>A full type name such as <c>NumSharp.NDArray</c> or <c>NumSharp.Generic.NDArray`1</c>, or the generic
///     parameter's name for an open <c>this T x</c>.</returns>
static string ExtendedTypeName(Type t)
{
    if (t.IsByRef)
        t = t.GetElementType()!;
    if (t.IsGenericParameter)
        return t.Name;
    if (t.IsGenericType)
        t = t.GetGenericTypeDefinition();
    return t.FullName ?? t.Name;
}

/// <summary>One parameter: its name, its C# type, whether and how it defaults, and whether it is a <c>params</c>
///     array or a by-reference (<c>out</c> / <c>ref</c>) parameter.</summary>
/// <param name="p">The parameter.</param>
/// <returns>A JSON-serializable dictionary; <c>default</c> is the C# literal as text, <c>defaultKind</c> its kind
///     (<c>null</c>, <c>bool</c>, <c>number</c>, <c>string</c>, <c>char</c>, <c>enum</c>, <c>other</c>).</returns>
static Dictionary<string, object?> DescribeParameter(ParameterInfo p)
{
    bool hasDefault = p.HasDefaultValue;
    object? value = hasDefault ? p.DefaultValue : null;
    // A struct parameter's `default` (e.g. `Shape s = default`) has no CLR constant: HasDefaultValue is true and the
    // value is null; Missing.Value / DBNull mark an optional parameter without a usable value.
    string? kind = !hasDefault ? null
        : value is null || value == DBNull.Value || value == Missing.Value ? "null"
        : value is bool ? "bool"
        : value is string ? "string"
        : value is char ? "char"
        : value.GetType().IsEnum ? "enum"
        : value is IConvertible && value.GetType().IsPrimitive || value is decimal ? "number"
        : "other";
    string? text = kind switch
    {
        null => null,
        // A struct's default (`Shape size = default`) reads null through reflection, like a reference type's null;
        // C# source writes it `default`, so the text says so while the kind stays "null" (both mean "not given").
        "null" => p.ParameterType.IsValueType && Nullable.GetUnderlyingType(p.ParameterType) is null ? "default" : "null",
        "bool" => (bool)value! ? "true" : "false",
        "string" => JsonSerializer.Serialize((string)value!),
        "char" => $"'{value}'",
        "enum" => $"{value!.GetType().Name}.{value}",
        "number" => Convert.ToString(value, CultureInfo.InvariantCulture),
        _ => value!.ToString(),
    };
    return new Dictionary<string, object?>
    {
        ["name"] = p.Name,
        ["type"] = Friendly(p.ParameterType.IsByRef ? p.ParameterType.GetElementType()! : p.ParameterType),
        ["optional"] = hasDefault || p.IsOptional,
        ["default"] = text,
        ["defaultKind"] = kind,
        ["params"] = p.GetCustomAttribute<ParamArrayAttribute>() is not null,
        ["byRef"] = p.IsOut ? "out" : p.ParameterType.IsByRef ? (p.IsIn ? "in" : "ref") : null,
    };
}

/// <summary>A readable C# type name: keywords for the primitives, <c>T?</c> for nullables, <c>T[]</c> for arrays,
///     generic arguments spelled out, namespaces dropped.</summary>
/// <param name="t">The type.</param>
/// <returns>The name.</returns>
static string Friendly(Type t)
{
    if (t.IsArray)
        return Friendly(t.GetElementType()!) + "[" + new string(',', t.GetArrayRank() - 1) + "]";
    var underlying = Nullable.GetUnderlyingType(t);
    if (underlying is not null)
        return Friendly(underlying) + "?";
    if (t.IsGenericParameter)
        return t.Name;
    string name = t == typeof(int) ? "int" : t == typeof(long) ? "long" : t == typeof(double) ? "double"
        : t == typeof(float) ? "float" : t == typeof(bool) ? "bool" : t == typeof(string) ? "string"
        : t == typeof(object) ? "object" : t == typeof(void) ? "void" : t == typeof(byte) ? "byte"
        : t == typeof(sbyte) ? "sbyte" : t == typeof(short) ? "short" : t == typeof(ushort) ? "ushort"
        : t == typeof(uint) ? "uint" : t == typeof(ulong) ? "ulong" : t == typeof(char) ? "char"
        : t == typeof(decimal) ? "decimal" : t.Name.Split('`')[0];
    if (t.IsNested && !t.IsGenericParameter && name == t.Name.Split('`')[0])
        name = t.DeclaringType!.Name.Split('`')[0] + "." + name;
    return t.IsGenericType ? name + "<" + string.Join(", ", t.GetGenericArguments().Select(Friendly)) + ">" : name;
}
