#!/usr/bin/env dotnet-script
#r "K:/source/NumSharp/src/NumSharp.Core/bin/Debug/net10.0/NumSharp.Core.dll"
using NumSharp;

// Test 1: Empty array axis reduction
Console.WriteLine("=== Empty Array Axis Reduction ===");
try {
    var arr = np.zeros(new int[] { 0, 3 });
    Console.WriteLine($"Input shape: ({string.Join(", ", arr.shape)})");
    var result = np.sum(arr, axis: 0);
    Console.WriteLine($"Result shape: ({string.Join(", ", result.shape)})");
    Console.WriteLine($"Result dtype: {result.dtype}");
    Console.WriteLine($"Result size: {result.size}");
} catch (Exception e) {
    Console.WriteLine($"Error: {e.Message}");
}

// Test 2: Single row matrix axis 0 reduction
Console.WriteLine("\n=== Single Row Matrix Axis 0 ===");
try {
    var arr = np.array(new int[,] { { 1, 2, 3 } });  // shape (1, 3)
    Console.WriteLine($"Input shape: ({string.Join(", ", arr.shape)})");
    var result = np.sum(arr, axis: 0);
    Console.WriteLine($"Result shape: ({string.Join(", ", result.shape)})");
    Console.WriteLine($"Result size: {result.size}");
    for (int i = 0; i < result.size; i++)
        Console.WriteLine($"  result[{i}] = {result.GetInt64(i)}");
} catch (Exception e) {
    Console.WriteLine($"Error: {e.GetType().Name}: {e.Message}");
}

// Test 3: Single column matrix axis 1 reduction
Console.WriteLine("\n=== Single Column Matrix Axis 1 ===");
try {
    var arr = np.array(new int[,] { { 1 }, { 2 }, { 3 } });  // shape (3, 1)
    Console.WriteLine($"Input shape: ({string.Join(", ", arr.shape)})");
    var result = np.sum(arr, axis: 1);
    Console.WriteLine($"Result shape: ({string.Join(", ", result.shape)})");
    Console.WriteLine($"Result size: {result.size}");
    for (int i = 0; i < result.size; i++)
        Console.WriteLine($"  result[{i}] = {result.GetInt64(i)}");
} catch (Exception e) {
    Console.WriteLine($"Error: {e.GetType().Name}: {e.Message}");
}
