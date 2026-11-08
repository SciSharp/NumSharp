#:project ../../src/NumSharp.Core
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

// Dumps every public overload of np.* (static) and NDArray (instance) as JSON:
// { "np": { name: [ [ {name, type, optional, default}... ], ... ] }, "ndarray": { ... } }
// The parameter-parity inventory compares these against NumPy 2.4.2's inspect.signature.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using NumSharp;

static object Describe(MethodInfo m) => m.GetParameters().Select(p => new Dictionary<string, object>
{
    ["name"] = p.Name,
    ["type"] = p.ParameterType.IsGenericType
        ? p.ParameterType.Name.Split('`')[0] + "<" + string.Join(",", p.ParameterType.GetGenericArguments().Select(a => a.Name)) + ">"
        : p.ParameterType.Name,
    ["optional"] = p.HasDefaultValue || p.IsOptional,
    ["params"] = p.GetCustomAttributes(typeof(ParamArrayAttribute), false).Length > 0,
    ["default"] = p.HasDefaultValue ? (p.DefaultValue?.ToString() ?? "null") : null,
}).ToList();

static Dictionary<string, List<object>> Collect(IEnumerable<MethodInfo> methods) => methods
    .Where(m => !m.IsSpecialName)
    .GroupBy(m => m.Name)
    .OrderBy(g => g.Key, StringComparer.Ordinal)
    .ToDictionary(g => g.Key, g => g.Select(Describe).ToList());

var npMethods = typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly);
var ndMethods = typeof(NDArray).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);

var result = new Dictionary<string, object>
{
    ["np"] = Collect(npMethods),
    ["ndarray"] = Collect(ndMethods),
};
string outPath = args.Length > 0 ? args[0] : "api_dump.json";
File.WriteAllText(outPath, JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = false }));
Console.WriteLine($"np: {((Dictionary<string, List<object>>)result["np"]).Count} names, ndarray: {((Dictionary<string, List<object>>)result["ndarray"]).Count} names -> {outPath}");
