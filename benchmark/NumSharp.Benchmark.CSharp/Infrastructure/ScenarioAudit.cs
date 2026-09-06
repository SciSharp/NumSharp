using System.Collections;
using System.Reflection;
using System.Text.Json;
using BenchmarkDotNet.Attributes;
using NumSharp;

namespace NumSharp.Benchmark.CSharp.Infrastructure;

/// <summary>
/// Discovers the exact official BenchmarkDotNet scenario titles and dtype parameters without
/// executing a benchmark. This is deliberately driven by BDN attributes rather than a parallel
/// hand-maintained inventory.
/// </summary>
internal static class ScenarioAudit
{
    private static readonly IReadOnlyDictionary<string, string> OfficialNamespaces =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["NumSharp.Benchmark.CSharp.Benchmarks.Arithmetic"] = "Arithmetic",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Unary"] = "Unary",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Reduction"] = "Reduction",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Broadcasting"] = "Broadcast",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Creation"] = "Creation",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Manipulation"] = "Manipulation",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Slicing"] = "Slicing",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Comparison"] = "Comparison",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Bitwise"] = "Bitwise",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Logic"] = "Logic",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Statistics"] = "Statistics",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Sorting"] = "Sorting",
            ["NumSharp.Benchmark.CSharp.Benchmarks.LinearAlgebra"] = "LinearAlgebra",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Selection"] = "Selection",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Fourier"] = "Fourier",
            ["NumSharp.Benchmark.CSharp.Benchmarks.Random"] = "Random",
            ["NumSharp.Benchmark.CSharp.Benchmarks.NDArrayApi"] = "NDArray",
            ["NumSharp.Benchmark.CSharp.Benchmarks.ApiSurface"] = "ApiSurface",
        };

    private static readonly IReadOnlyDictionary<NPTypeCode, string> DtypeNames =
        new Dictionary<NPTypeCode, string>
        {
            [NPTypeCode.Boolean] = "bool",
            [NPTypeCode.Byte] = "uint8",
            [NPTypeCode.SByte] = "int8",
            [NPTypeCode.Int16] = "int16",
            [NPTypeCode.UInt16] = "uint16",
            [NPTypeCode.Int32] = "int32",
            [NPTypeCode.UInt32] = "uint32",
            [NPTypeCode.Int64] = "int64",
            [NPTypeCode.UInt64] = "uint64",
            [NPTypeCode.Char] = "char",
            [NPTypeCode.Half] = "float16",
            [NPTypeCode.Single] = "float32",
            [NPTypeCode.Double] = "float64",
            [NPTypeCode.Decimal] = "decimal",
            [NPTypeCode.Complex] = "complex128",
        };

    private static readonly string[] DtypeOrder =
    [
        "bool", "uint8", "int8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
        "char", "float16", "float32", "float64", "decimal", "complex128",
    ];

    public static void WriteJson(Assembly assembly)
    {
        var rows = assembly.GetTypes()
            .Where(type => type.Namespace is not null && OfficialNamespaces.ContainsKey(type.Namespace))
            .SelectMany(type => DiscoverType(type, OfficialNamespaces[type.Namespace!]))
            .OrderBy(row => row.Suite, StringComparer.Ordinal)
            .ThenBy(row => row.Title, StringComparer.Ordinal)
            .ThenBy(row => row.Type, StringComparer.Ordinal)
            .ThenBy(row => row.Method, StringComparer.Ordinal)
            .ToArray();

        var payload = new
        {
            schema_version = 1,
            source = "BenchmarkDotNet attribute reflection",
            assembly = assembly.GetName().Name,
            rows = rows.Select(row => new
            {
                title = row.Title,
                suite = row.Suite,
                category = row.Category,
                type = row.Type,
                method = row.Method,
                dtypes = row.Dtypes,
            }),
        };
        Console.WriteLine(JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
    }

    /// <summary>
    /// Executes every official benchmark method once at its smallest declared parameter values.
    /// The dtype filter is the same NUMSHARP_BENCHMARK_DTYPES contract used by BDN. This proves
    /// that an audited check mark represents executable C# rather than reflection metadata alone.
    /// </summary>
    public static bool VerifyExecution(Assembly assembly)
    {
        var requested = BenchmarkRunSelection.RequestedDtypes;
        var failures = new List<string>();
        int executed = 0;
        int expected = 0;

        foreach (var type in assembly.GetTypes()
                     .Where(type => type.Namespace is not null && OfficialNamespaces.ContainsKey(type.Namespace))
                     .OrderBy(type => type.FullName, StringComparer.Ordinal))
        {
            var methods = type.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.DeclaredOnly)
                .Where(method => method.GetCustomAttribute<BenchmarkAttribute>() is not null)
                .OrderBy(method => method.Name, StringComparer.Ordinal)
                .ToArray();
            if (methods.Length == 0)
                continue;

            var dtypeProperty = FindDtypeProperty(type);
            var dtypes = ReadParameterValues(type, dtypeProperty).Cast<object>()
                .Select(value => (Code: (NPTypeCode)value, Name: DtypeName(value)))
                .Where(item => requested.Count == 0 || requested.Contains(item.Name))
                .ToArray();

            foreach (var dtype in dtypes)
            {
                expected += methods.Length;
                object? instance = null;
                try
                {
                    instance = Activator.CreateInstance(type)
                        ?? throw new InvalidOperationException($"Could not construct {type.FullName}.");
                    SetSmallestParameters(type, instance, dtypeProperty, dtype.Code);
                    InvokeAttributed<GlobalSetupAttribute>(type, instance);

                    foreach (var method in methods)
                    {
                        try
                        {
                            DisposeResult(method.Invoke(instance, null));
                            executed++;
                        }
                        catch (Exception error)
                        {
                            failures.Add($"{dtype.Name} | {type.Name}.{method.Name} | {RootCause(error).GetType().Name}: {RootCause(error).Message}");
                        }
                    }
                }
                catch (Exception error)
                {
                    var cause = RootCause(error);
                    foreach (var method in methods)
                        failures.Add($"{dtype.Name} | {type.Name}.{method.Name} | setup {cause.GetType().Name}: {cause.Message}");
                }
                finally
                {
                    if (instance is not null)
                    {
                        try { InvokeAttributed<GlobalCleanupAttribute>(type, instance); }
                        catch (Exception error)
                        {
                            var cause = RootCause(error);
                            failures.Add($"{dtype.Name} | {type.Name} cleanup | {cause.GetType().Name}: {cause.Message}");
                        }
                    }
                }
            }
        }

        Console.WriteLine($"Scenario execution: {executed:N0} / {expected:N0} function × dtype cells succeeded.");
        foreach (var failure in failures)
            Console.WriteLine($"FAIL {failure}");
        return failures.Count == 0 && executed == expected;
    }

    private static IEnumerable<ScenarioRow> DiscoverType(Type type, string suite)
    {
        var methods = type.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.DeclaredOnly)
            .Where(method => method.GetCustomAttribute<BenchmarkAttribute>() is not null)
            .ToArray();
        if (methods.Length == 0)
            yield break;

        var dtypes = DiscoverDtypes(type);
        var category = string.Join(", ", type.GetCustomAttributes<BenchmarkCategoryAttribute>(true)
            .SelectMany(attribute => attribute.Categories)
            .Distinct(StringComparer.Ordinal));

        foreach (var method in methods)
        {
            var benchmark = method.GetCustomAttribute<BenchmarkAttribute>()!;
            var title = string.IsNullOrWhiteSpace(benchmark.Description) ? method.Name : benchmark.Description;
            yield return new ScenarioRow(
                title, suite, category, type.FullName ?? type.Name, method.Name, dtypes);
        }
    }

    private static string[] DiscoverDtypes(Type type)
    {
        var dtypeProperty = FindDtypeProperty(type);
        var names = ReadParameterValues(type, dtypeProperty).Cast<object>()
            .Select(DtypeName)
            .Distinct(StringComparer.Ordinal)
            .OrderBy(name => Array.IndexOf(DtypeOrder, name))
            .ToArray();
        if (names.Length == 0)
            throw new InvalidOperationException($"{type.FullName}.DType produced no BDN parameter values.");
        return names;
    }

    private static PropertyInfo FindDtypeProperty(Type type) =>
        type.GetProperties(BindingFlags.Instance | BindingFlags.Public)
            .FirstOrDefault(property => property.Name == "DType" && property.DeclaringType == type)
        ?? type.GetProperty("DType", BindingFlags.Instance | BindingFlags.Public)
        ?? throw new InvalidOperationException($"{type.FullName} has no DType scenario parameter.");

    private static IEnumerable ReadParameterValues(Type type, PropertyInfo property)
    {
        if (property.GetCustomAttribute<ParamsSourceAttribute>(true) is { } source)
            return ReadSource(type, source.Name);
        if (property.GetCustomAttribute<ParamsAttribute>(true) is { } parameters)
            return parameters.Values;
        throw new InvalidOperationException($"{type.FullName}.{property.Name} has no readable BDN parameter source.");
    }

    private static void SetSmallestParameters(Type type, object instance, PropertyInfo dtypeProperty, NPTypeCode dtype)
    {
        dtypeProperty.SetValue(instance, dtype);
        foreach (var property in type.GetProperties(BindingFlags.Instance | BindingFlags.Public))
        {
            if (!property.CanWrite || property.Name == dtypeProperty.Name)
                continue;

            IEnumerable? values = null;
            if (property.GetCustomAttribute<ParamsSourceAttribute>(true) is { } source)
                values = ReadSource(type, source.Name);
            else if (property.GetCustomAttribute<ParamsAttribute>(true) is { } parameters)
                values = parameters.Values;
            if (values is null)
                continue;

            var candidates = values.Cast<object>().ToArray();
            if (candidates.Length == 0)
                continue;
            var selected = property.Name == "N" && candidates.All(value => value is int)
                ? candidates.OrderBy(value => (int)value).First()
                : candidates[0];
            property.SetValue(instance, selected);
        }
    }

    private static void InvokeAttributed<TAttribute>(Type type, object instance) where TAttribute : Attribute
    {
        foreach (var method in type.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                     .Where(method => method.GetCustomAttribute<TAttribute>(true) is not null))
            method.Invoke(instance, null);
    }

    private static void DisposeResult(object? result)
    {
        if (result is IDisposable disposable)
        {
            disposable.Dispose();
            return;
        }
        if (result is Array array)
            foreach (var item in array)
                if (item is IDisposable child)
                    child.Dispose();
    }

    private static Exception RootCause(Exception error)
    {
        while (error is TargetInvocationException { InnerException: not null })
            error = error.InnerException;
        return error;
    }

    private static IEnumerable ReadSource(Type type, string name)
    {
        const BindingFlags flags = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.FlattenHierarchy;
        object? value = type.GetProperty(name, flags)?.GetValue(null)
            ?? type.GetField(name, flags)?.GetValue(null)
            ?? type.GetMethod(name, flags, binder: null, types: Type.EmptyTypes, modifiers: null)?.Invoke(null, null);
        return value as IEnumerable
            ?? throw new InvalidOperationException($"Could not read {type.FullName}.{name} as an enumerable BDN parameter source.");
    }

    private static string DtypeName(object value) => value switch
    {
        NPTypeCode code when DtypeNames.TryGetValue(code, out var name) => name,
        string name when DtypeOrder.Contains(name, StringComparer.Ordinal) => name,
        _ => throw new InvalidOperationException($"Unsupported dtype parameter {value} ({value.GetType().FullName})."),
    };

    private sealed record ScenarioRow(
        string Title, string Suite, string Category, string Type, string Method, string[] Dtypes);
}
