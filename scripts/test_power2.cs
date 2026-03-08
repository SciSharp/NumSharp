#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// v2 after fix

using NumSharp;

Console.WriteLine("=== Testing np.power type promotion (after fix) ===\n");

// Test 1: int32 ^ int (should stay int32)
Console.WriteLine("Test 1: np.power(int32[2,3,4], 2)");
var arr1 = np.array(new int[] { 2, 3, 4 });
var result1 = np.power(arr1, 2);
Console.WriteLine($"  Input dtype: {arr1.dtype.Name}");
Console.WriteLine($"  Result dtype: {result1.dtype.Name}");
Console.WriteLine($"  Expected dtype: Int32");
Console.WriteLine($"  Result: [{string.Join(", ", result1.ToArray<int>())}]");
Console.WriteLine(result1.dtype == typeof(int) ? "  PASS" : "  FAIL");

// Test 2: int32 ^ float (should return float64!)
Console.WriteLine("\nTest 2: np.power(int32[2,3,4], 0.5)");
var arr2 = np.array(new int[] { 2, 3, 4 });
var result2 = np.power(arr2, 0.5);
Console.WriteLine($"  Input dtype: {arr2.dtype.Name}");
Console.WriteLine($"  Result dtype: {result2.dtype.Name}");
Console.WriteLine($"  Expected dtype: Double");
var vals2 = result2.ToArray<double>();
Console.WriteLine($"  Result: [{string.Join(", ", vals2.Select(v => v.ToString("F4")))}]");
Console.WriteLine($"  Expected: [1.4142, 1.7321, 2.0000]");
Console.WriteLine(result2.dtype == typeof(double) ? "  PASS" : "  FAIL");

// Test 3: int32 ^ float (negative exponent)
Console.WriteLine("\nTest 3: np.power(int32[2,4,8], -1.0)");
var arr3 = np.array(new int[] { 2, 4, 8 });
var result3 = np.power(arr3, -1.0);
Console.WriteLine($"  Result dtype: {result3.dtype.Name}");
Console.WriteLine($"  Expected dtype: Double");
var vals3 = result3.ToArray<double>();
Console.WriteLine($"  Result: [{string.Join(", ", vals3.Select(v => v.ToString("F4")))}]");
Console.WriteLine($"  Expected: [0.5000, 0.2500, 0.1250]");
Console.WriteLine(result3.dtype == typeof(double) ? "  PASS" : "  FAIL");

// Test 4: float ^ float (should stay float)
Console.WriteLine("\nTest 4: np.power(float32[2,3,4], 0.5)");
var arr4 = np.array(new float[] { 2f, 3f, 4f });
var result4 = np.power(arr4, 0.5);
Console.WriteLine($"  Result dtype: {result4.dtype.Name}");
Console.WriteLine($"  Expected dtype: Single (float32)");
Console.WriteLine(result4.dtype == typeof(float) ? "  PASS" : "  FAIL");

// Test 5: double ^ float (should stay double)
Console.WriteLine("\nTest 5: np.power(float64[2,3,4], 0.5)");
var arr5 = np.array(new double[] { 2.0, 3.0, 4.0 });
var result5 = np.power(arr5, 0.5);
Console.WriteLine($"  Result dtype: {result5.dtype.Name}");
Console.WriteLine($"  Expected dtype: Double");
Console.WriteLine(result5.dtype == typeof(double) ? "  PASS" : "  FAIL");

Console.WriteLine("\n=== Test Complete ===");
