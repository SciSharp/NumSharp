#!/usr/bin/env dotnet run
#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using System;
using NumSharp;

Console.WriteLine("Testing comparison operators (Bug 66, 67, 68)...\n");

// Bug 66: NotEquals NDArray vs NDArray
Console.WriteLine("Bug 66: Testing a != b (NDArray vs NDArray):");
try
{
    var a = np.array(new int[] { 1, 2, 3 });
    var b = np.array(new int[] { 1, 0, 3 });
    var result = a != b;

    bool pass = !result.GetBoolean(0) && result.GetBoolean(1) && !result.GetBoolean(2);
    Console.WriteLine($"  a = {a}, b = {b}");
    Console.WriteLine($"  Result: {result}");
    Console.WriteLine($"  Expected: [False, True, False]");
    Console.WriteLine($"  Actual: [{result.GetBoolean(0)}, {result.GetBoolean(1)}, {result.GetBoolean(2)}]");
    Console.WriteLine($"  PASS: {pass}");
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL - EXCEPTION: {ex.GetType().Name}: {ex.Message}");
}
Console.WriteLine();

// Bug 67: GreaterThan NDArray vs NDArray
Console.WriteLine("Bug 67: Testing a > b (NDArray vs NDArray):");
try
{
    var a = np.array(new int[] { 1, 5, 3 });
    var b = np.array(new int[] { 2, 4, 3 });
    var result = a > b;

    bool pass = !result.GetBoolean(0) && result.GetBoolean(1) && !result.GetBoolean(2);
    Console.WriteLine($"  a = {a}, b = {b}");
    Console.WriteLine($"  Result: {result}");
    Console.WriteLine($"  Expected: [False, True, False]");
    Console.WriteLine($"  Actual: [{result.GetBoolean(0)}, {result.GetBoolean(1)}, {result.GetBoolean(2)}]");
    Console.WriteLine($"  PASS: {pass}");
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL - EXCEPTION: {ex.GetType().Name}: {ex.Message}");
}
Console.WriteLine();

// Bug 68: LessThan NDArray vs NDArray
Console.WriteLine("Bug 68: Testing a < b (NDArray vs NDArray):");
try
{
    var a = np.array(new int[] { 1, 5, 3 });
    var b = np.array(new int[] { 2, 4, 3 });
    var result = a < b;

    bool pass = result.GetBoolean(0) && !result.GetBoolean(1) && !result.GetBoolean(2);
    Console.WriteLine($"  a = {a}, b = {b}");
    Console.WriteLine($"  Result: {result}");
    Console.WriteLine($"  Expected: [True, False, False]");
    Console.WriteLine($"  Actual: [{result.GetBoolean(0)}, {result.GetBoolean(1)}, {result.GetBoolean(2)}]");
    Console.WriteLine($"  PASS: {pass}");
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL - EXCEPTION: {ex.GetType().Name}: {ex.Message}");
}
Console.WriteLine();

Console.WriteLine("All bug tests completed.");
