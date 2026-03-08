#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// Force rebuild v2

using NumSharp;

Console.WriteLine("=== Testing np.linspace dtype (v2) ===\n");

// Test 1: Default call with integers
Console.WriteLine("Test 1: np.linspace(0, 10, 5)");
var result1 = np.linspace(0, 10, 5);
Console.WriteLine($"  dtype: {result1.dtype.Name}");
Console.WriteLine($"  Expected: Double (float64)");
Console.WriteLine(result1.dtype == typeof(double) ? "  PASS" : "  FAIL");

// Test 2: Call with explicit doubles
Console.WriteLine("\nTest 2: np.linspace(0.0, 10.0, 5)");
var result2 = np.linspace(0.0, 10.0, 5);
Console.WriteLine($"  dtype: {result2.dtype.Name}");
Console.WriteLine(result2.dtype == typeof(double) ? "  PASS" : "  FAIL");

// Test 3: Call with floats - NOW should also return float64
Console.WriteLine("\nTest 3: np.linspace(0f, 10f, 5)");
var result3 = np.linspace(0f, 10f, 5);
Console.WriteLine($"  dtype: {result3.dtype.Name}");
Console.WriteLine($"  Expected: Double (NumPy always returns float64)");
Console.WriteLine(result3.dtype == typeof(double) ? "  PASS" : "  FAIL");

Console.WriteLine("\n=== Test Complete ===");
