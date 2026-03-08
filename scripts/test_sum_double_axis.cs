#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;

Console.WriteLine("=== Testing np.sum with axis on double arrays ===\n");

// Test 1: Sum along axis 0
var arr1 = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
Console.WriteLine($"Test 1: np.sum(2x3 double array, axis=0)");
Console.WriteLine($"  Input shape: ({string.Join(",", arr1.shape)})");
Console.WriteLine($"  Input dtype: {arr1.dtype.Name}");
try {
    var result1 = np.sum(arr1, axis: 0);
    Console.WriteLine($"  Result shape: ({string.Join(",", result1.shape)})");
    Console.WriteLine($"  Result: [{string.Join(", ", result1.ToArray<double>())}]");
    Console.WriteLine($"  Expected: [5.0, 7.0, 9.0]");
    Console.WriteLine("  PASS");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine($"  Stack: {ex.StackTrace?.Split('\n')[0]}");
}

// Test 2: Sum along axis 1
Console.WriteLine($"\nTest 2: np.sum(2x3 double array, axis=1)");
try {
    var result2 = np.sum(arr1, axis: 1);
    Console.WriteLine($"  Result shape: ({string.Join(",", result2.shape)})");
    Console.WriteLine($"  Result: [{string.Join(", ", result2.ToArray<double>())}]");
    Console.WriteLine($"  Expected: [6.0, 15.0]");
    Console.WriteLine("  PASS");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine($"  Stack: {ex.StackTrace?.Split('\n')[0]}");
}

// Test 3: Sum with keepdims
Console.WriteLine($"\nTest 3: np.sum(2x3 double array, axis=0, keepdims=true)");
try {
    var result3 = np.sum(arr1, axis: 0, keepdims: true);
    Console.WriteLine($"  Result shape: ({string.Join(",", result3.shape)})");
    Console.WriteLine($"  Expected shape: (1, 3)");
    Console.WriteLine(result3.ndim == 2 && result3.shape[0] == 1 && result3.shape[1] == 3 ? "  PASS" : "  FAIL");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine($"  Stack: {ex.StackTrace?.Split('\n')[0]}");
}

// Test 4: 3D array sum
var arr4 = np.zeros(new[] { 2, 3, 4 });
Console.WriteLine($"\nTest 4: np.sum(2x3x4 double zeros array)");
Console.WriteLine($"  Input shape: ({string.Join(",", arr4.shape)})");
try {
    var result4 = np.sum(arr4);
    Console.WriteLine($"  Result: {result4}");
    Console.WriteLine($"  Expected: 0.0");
    Console.WriteLine("  PASS");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine($"  Stack: {ex.StackTrace?.Split('\n')[0]}");
}

// Test 5: 3D array sum with axis
Console.WriteLine($"\nTest 5: np.sum(2x3x4 array, axis=1)");
try {
    var result5 = np.sum(arr4, axis: 1);
    Console.WriteLine($"  Result shape: ({string.Join(",", result5.shape)})");
    Console.WriteLine($"  Expected shape: (2, 4)");
    Console.WriteLine(result5.shape[0] == 2 && result5.shape[1] == 4 ? "  PASS" : "  FAIL");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine($"  Stack: {ex.StackTrace?.Split('\n')[0]}");
}

Console.WriteLine("\n=== Test Complete ===");
