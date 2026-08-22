#:project ../NumSharp.Benchmark.CSharp/NumSharp.Benchmark.CSharp.csproj
#:property AssemblyName=NumSharp.Benchmark.BodySmoke
#:property PublishAot=false
#:property WarningLevel=0

// Run with: dotnet run -c Release --no-cache benchmark/scripts/smoke_benchmark_bodies.cs
// --no-cache matters: file-app caching otherwise does not notice a changed ProjectReference.

using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

#if DEBUG
Console.WriteLine("FATAL: Debug build — rerun with -c Release");
return;
#pragma warning disable CS0162
#endif

np.multithreading(false);

var officialNamespaces = new HashSet<string>
{
    "Arithmetic", "Unary", "Reduction", "Broadcasting", "Creation", "Manipulation",
    "Slicing", "Comparison", "Bitwise", "Logic", "Statistics", "Sorting",
    "LinearAlgebra", "Selection", "Fourier", "Random", "NDArrayApi", "ApiSurface"
};

var failures = new List<string>();
int calls = 0;
var assembly = typeof(BenchmarkBase).Assembly;
var classes = assembly.GetTypes()
    .Where(type => type is { IsClass: true, IsAbstract: false }
        && type.Namespace?.StartsWith("NumSharp.Benchmark.CSharp.Benchmarks.") == true
        && officialNamespaces.Contains(type.Namespace.Split('.').Last())
        && type.GetMethods().Any(method => method.GetCustomAttribute<BenchmarkAttribute>() != null))
    .OrderBy(type => type.FullName)
    .ToArray();

foreach (var type in classes)
{
    object? instance = null;
    try
    {
        instance = Activator.CreateInstance(type)!;
        SetRepresentativeParameters(type, instance);
        foreach (var setup in type.GetMethods().Where(method => method.GetCustomAttribute<GlobalSetupAttribute>() != null))
            setup.Invoke(instance, null);

        foreach (var benchmark in type.GetMethods().Where(method => method.GetCustomAttribute<BenchmarkAttribute>() != null))
        {
            try
            {
                var result = benchmark.Invoke(instance, null);
                DisposeResult(result);
                calls++;
            }
            catch (Exception error)
            {
                var cause = error is TargetInvocationException { InnerException: not null } tie ? tie.InnerException : error;
                failures.Add($"{type.FullName}.{benchmark.Name}: {cause!.GetType().Name}: {cause.Message.Split('\n')[0]}");
            }
        }
    }
    catch (Exception error)
    {
        var cause = error is TargetInvocationException { InnerException: not null } tie ? tie.InnerException : error;
        failures.Add($"{type.FullName}.Setup: {cause!.GetType().Name}: {cause.Message.Split('\n')[0]}");
        continue;
    }
    finally
    {
        if (instance is not null)
        {
            foreach (var cleanup in type.GetMethods().Where(method => method.GetCustomAttribute<GlobalCleanupAttribute>() != null))
            {
                try { cleanup.Invoke(instance, null); }
                catch { /* the primary body failure is the useful smoke diagnostic */ }
            }
        }
    }
}

if (failures.Count == 0)
{
    Console.WriteLine($"OK: invoked {calls} benchmark bodies across {classes.Length} official classes.");
    return;
}

Console.WriteLine($"FAIL: {failures.Count} benchmark bodies/setups failed ({calls} passed):");
foreach (var failure in failures) Console.WriteLine("  " + failure);
Environment.ExitCode = 1;

static void SetRepresentativeParameters(Type type, object instance)
{
    foreach (var property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance)
                 .Where(property => property.CanWrite && property.Name == "N"))
        property.SetValue(instance, ArraySizeSource.Small);

    var typesProperty = type.GetProperty("Types", BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy);
    NPTypeCode representative = NPTypeCode.Double;
    if (typesProperty?.GetValue(null) is IEnumerable types)
    {
        foreach (var value in types)
        {
            if (value is NPTypeCode code) representative = code;
            break;
        }
    }

    foreach (var property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance)
                 .Where(property => property.CanWrite && property.Name == "DType"))
        property.SetValue(instance, representative);
}

static void DisposeResult(object? result)
{
    if (result is null || result is string) return;
    if (result is IDisposable disposable)
    {
        disposable.Dispose();
        return;
    }
    if (result is ITuple tuple)
    {
        for (int i = 0; i < tuple.Length; i++) DisposeResult(tuple[i]);
        return;
    }
    if (result is IEnumerable sequence)
        foreach (var item in sequence) DisposeResult(item);
}
