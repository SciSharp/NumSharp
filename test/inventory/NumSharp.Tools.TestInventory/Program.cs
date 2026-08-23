using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;

var assemblyNames = args.Length > 0
    ? args
    : new[] { "NumSharp.Tests", "NumSharp.Tests.Oracle", "NumSharp.Tests.Interop" };

var projects = new List<ProjectInventory>();
foreach (string name in assemblyNames.OrderBy(value => value, StringComparer.Ordinal))
{
    var assembly = Assembly.Load(new AssemblyName(name));
    var tests = new List<TestInventory>();
    foreach (var type in SafeTypes(assembly).OrderBy(type => type.FullName, StringComparer.Ordinal))
    {
        var typeAttributes = TypeAttributes(type).ToArray();
        foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance |
                                                BindingFlags.Static | BindingFlags.DeclaredOnly)
                                   .OrderBy(method => method.Name, StringComparer.Ordinal))
        {
            var methodAttributes = method.GetCustomAttributesData().ToArray();
            if (!methodAttributes.Any(IsTestMethod))
                continue;

            var allAttributes = typeAttributes.Concat(methodAttributes).ToArray();
            string[] categories = Categories(allAttributes).Distinct(StringComparer.Ordinal)
                .OrderBy(value => value, StringComparer.Ordinal).ToArray();
            int dataRows = methodAttributes.Count(attribute => attribute.AttributeType.Name == "DataRowAttribute");
            bool dynamicData = methodAttributes.Any(attribute => attribute.AttributeType.Name == "DynamicDataAttribute");
            bool ignored = allAttributes.Any(attribute => attribute.AttributeType.Name == "IgnoreAttribute");
            string? ignoreReason = allAttributes
                .Where(attribute => attribute.AttributeType.Name == "IgnoreAttribute")
                .SelectMany(attribute => attribute.ConstructorArguments)
                .Where(argument => argument.Value is string)
                .Select(argument => (string?)argument.Value)
                .FirstOrDefault();
            string? issueUrl = allAttributes
                .Where(attribute => attribute.AttributeType.Name == "OpenBugsAttribute")
                .SelectMany(attribute => attribute.NamedArguments)
                .Where(argument => argument.MemberName == "IssueUrl")
                .Select(argument => argument.TypedValue.Value as string)
                .FirstOrDefault();

            tests.Add(new TestInventory(
                Id: $"{assembly.GetName().Name}::{type.FullName}.{method.Name}",
                Type: type.FullName ?? type.Name,
                Method: method.Name,
                Signature: FormatMethod(method),
                Categories: categories,
                DataRows: dataRows,
                DynamicData: dynamicData,
                Ignored: ignored,
                IgnoreReason: ignoreReason,
                IssueUrl: issueUrl));
        }
    }

    projects.Add(new ProjectInventory(
        Name: assembly.GetName().Name ?? name,
        AssemblyVersion: assembly.GetName().Version?.ToString() ?? "unknown",
        Tests: tests.OrderBy(test => test.Id, StringComparer.Ordinal).ToArray()));
}

var inventory = new RawTestInventory(SchemaVersion: 1, Projects: projects.ToArray());
Console.WriteLine(JsonSerializer.Serialize(inventory, new JsonSerializerOptions
{
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    WriteIndented = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
}));

static IEnumerable<Type> SafeTypes(Assembly assembly)
{
    try { return assembly.GetTypes(); }
    catch (ReflectionTypeLoadException error) { return error.Types.Where(type => type is not null)!; }
}

static IEnumerable<CustomAttributeData> TypeAttributes(Type type)
{
    for (Type? current = type; current is not null && current != typeof(object); current = current.BaseType)
        foreach (var attribute in current.GetCustomAttributesData())
            yield return attribute;
}

static bool IsTestMethod(CustomAttributeData attribute)
{
    for (Type? current = attribute.AttributeType; current is not null; current = current.BaseType)
        if (current.Name is "TestMethodAttribute" or "DataTestMethodAttribute")
            return true;
    return false;
}

static IEnumerable<string> Categories(IEnumerable<CustomAttributeData> attributes)
{
    var shorthand = new Dictionary<string, string[]>(StringComparer.Ordinal)
    {
        ["OpenBugsAttribute"] = new[] { "OpenBugs" },
        ["MisalignedAttribute"] = new[] { "Misaligned" },
        ["WindowsOnlyAttribute"] = new[] { "WindowsOnly" },
        ["LongIndexingAttribute"] = new[] { "LongIndexing" },
        ["HighMemoryAttribute"] = new[] { "HighMemory" },
        ["LargeMemoryTestAttribute"] = new[] { "OpenBugs", "HighMemory" },
    };

    foreach (var attribute in attributes)
    {
        if (shorthand.TryGetValue(attribute.AttributeType.Name, out var mapped))
        {
            foreach (string category in mapped)
                yield return category;
        }
        if (attribute.AttributeType.Name != "TestCategoryAttribute")
            continue;
        foreach (var argument in attribute.ConstructorArguments)
            foreach (string category in StringArguments(argument))
                yield return category;
    }
}

static IEnumerable<string> StringArguments(CustomAttributeTypedArgument argument)
{
    if (argument.Value is string value)
    {
        yield return value;
        yield break;
    }
    if (argument.Value is IEnumerable<CustomAttributeTypedArgument> values)
        foreach (var child in values)
            foreach (string nestedValue in StringArguments(child))
                yield return nestedValue;
}

static string FormatMethod(MethodInfo method)
    => $"{FriendlyType(method.ReturnType)} {method.Name}(" +
       string.Join(", ", method.GetParameters().Select(parameter =>
           $"{FriendlyType(parameter.ParameterType)} {parameter.Name}")) + ")";

static string FriendlyType(Type type)
{
    if (type.IsByRef)
        return FriendlyType(type.GetElementType()!) + "&";
    if (type.IsArray)
        return FriendlyType(type.GetElementType()!) + "[]";
    if (type.IsGenericParameter)
        return type.Name;
    if (!type.IsGenericType)
        return type.FullName ?? type.Name;
    string name = (type.GetGenericTypeDefinition().FullName ?? type.Name).Split('`')[0];
    return name + "<" + string.Join(", ", type.GetGenericArguments().Select(FriendlyType)) + ">";
}

internal sealed record RawTestInventory(int SchemaVersion, ProjectInventory[] Projects);
internal sealed record ProjectInventory(string Name, string AssemblyVersion, TestInventory[] Tests);
internal sealed record TestInventory(
    string Id,
    string Type,
    string Method,
    string Signature,
    string[] Categories,
    int DataRows,
    bool DynamicData,
    bool Ignored,
    string? IgnoreReason,
    string? IssueUrl);
