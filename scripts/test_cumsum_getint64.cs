#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;

Console.WriteLine("=== Testing GetInt64 on cumsum result ===\n");

var arr = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
Console.WriteLine($"Input dtype: {arr.dtype.Name}");

var result = np.cumsum(arr, axis: 0);
Console.WriteLine($"Result dtype: {result.dtype.Name}");
Console.WriteLine($"Result shape: ({string.Join(",", result.shape)})");
Console.WriteLine($"Result size: {result.size}");

// Test GetInt64
try {
    Console.WriteLine($"\nTrying GetInt64(0, 0)...");
    var val = result.GetInt64(0, 0);
    Console.WriteLine($"  Success: {val}");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
}

// Test GetValue
try {
    Console.WriteLine($"\nTrying GetValue(0, 0)...");
    var val = result.GetValue(0, 0);
    Console.WriteLine($"  Success: {val} (type: {val?.GetType().Name})");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
}

// Test indexer
try {
    Console.WriteLine($"\nTrying result[0, 0]...");
    var val = result[0, 0];
    Console.WriteLine($"  Success: {val} (dtype: {val.dtype.Name})");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
}

Console.WriteLine("\n=== Test Complete ===");
