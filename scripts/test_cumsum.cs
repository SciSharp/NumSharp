#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;

Console.WriteLine("=== Testing np.cumsum ===\n");

// Test 1: 1D cumsum (no axis)
try {
    var arr1 = np.array(new int[] { 1, 2, 3, 4, 5 });
    Console.WriteLine($"Test 1: np.cumsum([1,2,3,4,5])");
    Console.WriteLine($"  Input dtype: {arr1.dtype.Name}");
    var result1 = np.cumsum(arr1);
    Console.WriteLine($"  Result dtype: {result1.dtype.Name}");
    Console.WriteLine($"  Result shape: ({string.Join(",", result1.shape)})");
    Console.WriteLine($"  Result: [{string.Join(", ", result1.ToArray<long>())}]");
    Console.WriteLine($"  Expected: [1, 3, 6, 10, 15]");
    Console.WriteLine("  PASS");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
}

// Test 2: 2D cumsum with axis=0
try {
    var arr2 = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
    Console.WriteLine($"\nTest 2: np.cumsum([[1,2,3],[4,5,6]], axis=0)");
    Console.WriteLine($"  Input shape: ({string.Join(",", arr2.shape)})");
    Console.WriteLine($"  Input dtype: {arr2.dtype.Name}");
    var result2 = np.cumsum(arr2, axis: 0);
    Console.WriteLine($"  Result dtype: {result2.dtype.Name}");
    Console.WriteLine($"  Result shape: ({string.Join(",", result2.shape)})");
    Console.WriteLine($"  Result ndim: {result2.ndim}");
    Console.WriteLine($"  Expected: [[1,2,3],[5,7,9]]");
    // Try to access elements
    Console.WriteLine($"  result[0,0] = {result2.GetValue(0, 0)}");
    Console.WriteLine($"  result[1,0] = {result2.GetValue(1, 0)}");
    Console.WriteLine($"  result[1,2] = {result2.GetValue(1, 2)}");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine($"  Stack: {ex.StackTrace?.Split('\n').FirstOrDefault()}");
}

// Test 3: 2D cumsum with axis=1
try {
    var arr3 = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
    Console.WriteLine($"\nTest 3: np.cumsum([[1,2,3],[4,5,6]], axis=1)");
    var result3 = np.cumsum(arr3, axis: 1);
    Console.WriteLine($"  Result dtype: {result3.dtype.Name}");
    Console.WriteLine($"  Result shape: ({string.Join(",", result3.shape)})");
    Console.WriteLine($"  Expected: [[1,3,6],[4,9,15]]");
    Console.WriteLine($"  result[0,2] = {result3.GetValue(0, 2)}");
    Console.WriteLine($"  result[1,2] = {result3.GetValue(1, 2)}");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine($"  Stack: {ex.StackTrace?.Split('\n').FirstOrDefault()}");
}

Console.WriteLine("\n=== Test Complete ===");
