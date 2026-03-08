#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;

Console.WriteLine("=== Testing np.power type promotion ===\n");

// Test 1: int32 ^ int (should stay int32 in NumPy)
Console.WriteLine("Test 1: np.power(int32[2,3,4], 2)");
var arr1 = np.array(new int[] { 2, 3, 4 });
var result1 = np.power(arr1, 2);
Console.WriteLine($"  Input dtype: {arr1.dtype.Name}");
Console.WriteLine($"  Result dtype: {result1.dtype.Name}");
Console.WriteLine($"  Result: [{string.Join(", ", result1.ToArray<int>())}]");
Console.WriteLine($"  Expected: [4, 9, 16]");

// Test 2: int32 ^ float (should return float64 in NumPy!)
Console.WriteLine("\nTest 2: np.power(int32[2,3,4], 0.5)");
var arr2 = np.array(new int[] { 2, 3, 4 });
var result2 = np.power(arr2, 0.5);
Console.WriteLine($"  Input dtype: {arr2.dtype.Name}");
Console.WriteLine($"  Result dtype: {result2.dtype.Name}");
Console.WriteLine($"  Expected dtype: Double (float64)");
// Try to get actual values
try {
    var vals = result2.ToArray<double>();
    Console.WriteLine($"  Result: [{string.Join(", ", vals.Select(v => v.ToString("F4")))}]");
    Console.WriteLine($"  Expected: [1.4142, 1.7321, 2.0000]");
} catch {
    Console.WriteLine($"  Result (as int): [{string.Join(", ", result2.ToArray<int>())}]");
    Console.WriteLine($"  BUG: Truncated to integers!");
}

// Test 3: int32 ^ float (negative exponent)
Console.WriteLine("\nTest 3: np.power(int32[2,4,8], -1.0)");
var arr3 = np.array(new int[] { 2, 4, 8 });
var result3 = np.power(arr3, -1.0);
Console.WriteLine($"  Result dtype: {result3.dtype.Name}");
Console.WriteLine($"  Expected dtype: Double");
try {
    var vals = result3.ToArray<double>();
    Console.WriteLine($"  Result: [{string.Join(", ", vals.Select(v => v.ToString("F4")))}]");
    Console.WriteLine($"  Expected: [0.5, 0.25, 0.125]");
} catch {
    Console.WriteLine($"  Result (as int): [{string.Join(", ", result3.ToArray<int>())}]");
    Console.WriteLine($"  BUG: Values are zero due to truncation!");
}

Console.WriteLine("\n=== NumPy Reference ===");
Console.WriteLine(">>> np.power(np.array([2,3,4], dtype=np.int32), 0.5).dtype");
Console.WriteLine("float64");
Console.WriteLine(">>> np.power(np.array([2,3,4], dtype=np.int32), 2).dtype");
Console.WriteLine("int32");

Console.WriteLine("\n=== Test Complete ===");
