#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;

Console.WriteLine("=== Testing np.moveaxis ===\n");

// Test 1: Move axis 0 to position 2
// NumPy: np.moveaxis(np.zeros((3, 4, 5)), 0, -1).shape -> (4, 5, 3)
var arr1 = np.zeros(new[] { 3, 4, 5 });
Console.WriteLine($"Test 1: np.moveaxis(shape=(3,4,5), source=0, dest=-1)");
Console.WriteLine($"  Input shape: ({string.Join(",", arr1.shape)})");
var result1 = np.moveaxis(arr1, 0, -1);
Console.WriteLine($"  Output shape: ({string.Join(",", result1.shape)})");
Console.WriteLine($"  Expected: (4, 5, 3)");
if (result1.shape[0] == 4 && result1.shape[1] == 5 && result1.shape[2] == 3)
    Console.WriteLine("  PASS");
else
    Console.WriteLine("  FAIL");

// Test 2: Move axis 2 to position 0
// NumPy: np.moveaxis(np.zeros((3, 4, 5)), 2, 0).shape -> (5, 3, 4)
var arr2 = np.zeros(new[] { 3, 4, 5 });
Console.WriteLine($"\nTest 2: np.moveaxis(shape=(3,4,5), source=2, dest=0)");
Console.WriteLine($"  Input shape: ({string.Join(",", arr2.shape)})");
var result2 = np.moveaxis(arr2, 2, 0);
Console.WriteLine($"  Output shape: ({string.Join(",", result2.shape)})");
Console.WriteLine($"  Expected: (5, 3, 4)");
if (result2.shape[0] == 5 && result2.shape[1] == 3 && result2.shape[2] == 4)
    Console.WriteLine("  PASS");
else
    Console.WriteLine("  FAIL");

// Test 3: Move axis -1 to position 0
// NumPy: np.moveaxis(np.zeros((3, 4, 5)), -1, 0).shape -> (5, 3, 4)
var arr3 = np.zeros(new[] { 3, 4, 5 });
Console.WriteLine($"\nTest 3: np.moveaxis(shape=(3,4,5), source=-1, dest=0)");
Console.WriteLine($"  Input shape: ({string.Join(",", arr3.shape)})");
var result3 = np.moveaxis(arr3, -1, 0);
Console.WriteLine($"  Output shape: ({string.Join(",", result3.shape)})");
Console.WriteLine($"  Expected: (5, 3, 4)");
if (result3.shape[0] == 5 && result3.shape[1] == 3 && result3.shape[2] == 4)
    Console.WriteLine("  PASS");
else
    Console.WriteLine("  FAIL");

// Test 4: Multiple axes
// NumPy: np.moveaxis(np.zeros((3, 4, 5)), [0, 1], [-1, -2]).shape -> (5, 4, 3)
var arr4 = np.zeros(new[] { 3, 4, 5 });
Console.WriteLine($"\nTest 4: np.moveaxis(shape=(3,4,5), source=[0,1], dest=[-1,-2])");
Console.WriteLine($"  Input shape: ({string.Join(",", arr4.shape)})");
var result4 = np.moveaxis(arr4, new[] { 0, 1 }, new[] { -1, -2 });
Console.WriteLine($"  Output shape: ({string.Join(",", result4.shape)})");
Console.WriteLine($"  Expected: (5, 4, 3)");
if (result4.shape[0] == 5 && result4.shape[1] == 4 && result4.shape[2] == 3)
    Console.WriteLine("  PASS");
else
    Console.WriteLine("  FAIL");

Console.WriteLine("\n=== Test Complete ===");
