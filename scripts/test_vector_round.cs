#:property PublishAot=false

using System.Numerics;
using System.Reflection;

// Check what vector methods are available for rounding
var containerType = typeof(Vector);
var methods = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
    .Where(m => m.Name.Contains("Round") || m.Name.Contains("Floor") || m.Name.Contains("Ceil") || m.Name.Contains("Truncat"))
    .Select(m => $"{m.Name}({string.Join(", ", m.GetParameters().Select(p => p.ParameterType.Name))})")
    .Distinct()
    .OrderBy(x => x);

Console.WriteLine("Vector rounding methods:");
foreach (var method in methods)
    Console.WriteLine($"  {method}");

// Also check Vector128/256/512
Console.WriteLine("\nVector128 rounding methods:");
var v128Type = typeof(System.Runtime.Intrinsics.Vector128);
var v128methods = v128Type.GetMethods(BindingFlags.Public | BindingFlags.Static)
    .Where(m => m.Name.Contains("Round") || m.Name.Contains("Floor") || m.Name.Contains("Ceil") || m.Name.Contains("Truncat"))
    .Select(m => $"{m.Name}({string.Join(", ", m.GetParameters().Select(p => p.ParameterType.Name))})")
    .Distinct()
    .OrderBy(x => x);
foreach (var method in v128methods)
    Console.WriteLine($"  {method}");
