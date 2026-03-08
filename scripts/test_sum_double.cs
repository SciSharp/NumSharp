#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;

Console.WriteLine("=== Testing np.sum on double arrays ===\n");

// Test 1: Basic double sum
var arr1 = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
Console.WriteLine($"Test 1: np.sum(double array [1,2,3,4,5])");
Console.WriteLine($"  Input dtype: {arr1.dtype.Name}");
try {
    var result1 = np.sum(arr1);
    Console.WriteLine($"  Result: {result1}");
    Console.WriteLine($"  Expected: 15.0");
    Console.WriteLine(result1.GetDouble() == 15.0 ? "  PASS" : "  FAIL");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
}

// Test 2: 2D double array
var arr2 = np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
Console.WriteLine($"\nTest 2: np.sum(2D double array)");
Console.WriteLine($"  Input dtype: {arr2.dtype.Name}");
try {
    var result2 = np.sum(arr2);
    Console.WriteLine($"  Result: {result2}");
    Console.WriteLine($"  Expected: 10.0");
    Console.WriteLine(result2.GetDouble() == 10.0 ? "  PASS" : "  FAIL");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
}

// Test 3: np.zeros (default is double)
var arr3 = np.zeros(5);
Console.WriteLine($"\nTest 3: np.sum(np.zeros(5))");
Console.WriteLine($"  Input dtype: {arr3.dtype.Name}");
try {
    var result3 = np.sum(arr3);
    Console.WriteLine($"  Result: {result3}");
    Console.WriteLine($"  Expected: 0.0");
    Console.WriteLine(result3.GetDouble() == 0.0 ? "  PASS" : "  FAIL");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
}

// Test 4: float (Single) for comparison
var arr4 = np.array(new float[] { 1.0f, 2.0f, 3.0f });
Console.WriteLine($"\nTest 4: np.sum(float array [1,2,3])");
Console.WriteLine($"  Input dtype: {arr4.dtype.Name}");
try {
    var result4 = np.sum(arr4);
    Console.WriteLine($"  Result: {result4}");
    Console.WriteLine($"  Expected: 6.0");
    Console.WriteLine(result4.GetSingle() == 6.0f ? "  PASS" : "  FAIL");
} catch (Exception ex) {
    Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}");
}

Console.WriteLine("\n=== Test Complete ===");
