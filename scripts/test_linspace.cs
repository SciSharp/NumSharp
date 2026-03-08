#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;

Console.WriteLine("=== Testing np.linspace dtype ===\n");

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
Console.WriteLine($"  Expected: Double (float64)");
Console.WriteLine(result2.dtype == typeof(double) ? "  PASS" : "  FAIL");

// Test 3: Call with floats
Console.WriteLine("\nTest 3: np.linspace(0f, 10f, 5)");
var result3 = np.linspace(0f, 10f, 5);
Console.WriteLine($"  dtype: {result3.dtype.Name}");
Console.WriteLine($"  Expected: Single (float32) - using float overload");
Console.WriteLine(result3.dtype == typeof(float) ? "  PASS (correct for float input)" : "  FAIL");

// Test 4: numpy behavior check
Console.WriteLine("\nNumPy reference:");
Console.WriteLine("  np.linspace(0, 10, 5).dtype → float64");
Console.WriteLine("  np.linspace(0.0, 10.0, 5).dtype → float64");

Console.WriteLine("\n=== Test Complete ===");
