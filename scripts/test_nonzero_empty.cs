#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;

Console.WriteLine("=== Testing np.nonzero on empty arrays ===\n");

int passed = 0;
int failed = 0;

// Test 1: Empty 1D array
try
{
    var empty1d = np.array(new int[0]);
    Console.WriteLine($"Test 1: Empty 1D array, shape={string.Join(",", empty1d.shape)}");
    var result = np.nonzero(empty1d);
    Console.WriteLine($"  Result: {result.Length} arrays");
    if (result.Length == 1 && result[0].size == 0)
    {
        Console.WriteLine("  PASS: Returns 1 empty array for 1D input");
        passed++;
    }
    else
    {
        Console.WriteLine($"  FAIL: Expected 1 empty array, got {result.Length} arrays with sizes [{string.Join(",", result.Select(r => r.size))}]");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: Exception - {ex.Message}");
    failed++;
}

// Test 2: Empty 2D array (0x3)
try
{
    var empty2d = np.zeros(new int[] { 0, 3 });
    Console.WriteLine($"\nTest 2: Empty 2D array (0x3), shape={string.Join(",", empty2d.shape)}");
    var result = np.nonzero(empty2d);
    Console.WriteLine($"  Result: {result.Length} arrays");
    if (result.Length == 2 && result[0].size == 0 && result[1].size == 0)
    {
        Console.WriteLine("  PASS: Returns 2 empty arrays for 2D input");
        passed++;
    }
    else
    {
        Console.WriteLine($"  FAIL: Expected 2 empty arrays, got {result.Length} arrays with sizes [{string.Join(",", result.Select(r => r.size))}]");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: Exception - {ex.Message}");
    failed++;
}

// Test 3: Empty 3D array (2x0x4)
try
{
    var empty3d = np.zeros(new int[] { 2, 0, 4 });
    Console.WriteLine($"\nTest 3: Empty 3D array (2x0x4), shape={string.Join(",", empty3d.shape)}");
    var result = np.nonzero(empty3d);
    Console.WriteLine($"  Result: {result.Length} arrays");
    if (result.Length == 3 && result.All(r => r.size == 0))
    {
        Console.WriteLine("  PASS: Returns 3 empty arrays for 3D input");
        passed++;
    }
    else
    {
        Console.WriteLine($"  FAIL: Expected 3 empty arrays, got {result.Length} arrays with sizes [{string.Join(",", result.Select(r => r.size))}]");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: Exception - {ex.Message}");
    failed++;
}

// Test 4: All zeros (no nonzeros, but not empty array)
try
{
    var allZeros = np.zeros(new int[] { 5 });
    Console.WriteLine($"\nTest 4: All zeros (5 elements), shape={string.Join(",", allZeros.shape)}");
    var result = np.nonzero(allZeros);
    Console.WriteLine($"  Result: {result.Length} arrays with sizes [{string.Join(",", result.Select(r => r.size))}]");
    if (result.Length == 1 && result[0].size == 0)
    {
        Console.WriteLine("  PASS: Returns 1 empty array (no nonzeros found)");
        passed++;
    }
    else
    {
        Console.WriteLine($"  FAIL: Expected 1 empty array");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: Exception - {ex.Message}");
    failed++;
}

// Test 5: Normal case (sanity check)
try
{
    var normal = np.array(new[] { 0, 1, 0, 2, 0 });
    Console.WriteLine($"\nTest 5: Normal array [0,1,0,2,0]");
    var result = np.nonzero(normal);
    Console.WriteLine($"  Result: {result.Length} arrays");
    var indices = result[0].ToArray<int>();
    Console.WriteLine($"  Indices: [{string.Join(",", indices)}]");
    if (result.Length == 1 && indices.Length == 2 && indices[0] == 1 && indices[1] == 3)
    {
        Console.WriteLine("  PASS: Returns correct nonzero indices [1, 3]");
        passed++;
    }
    else
    {
        Console.WriteLine($"  FAIL: Expected indices [1, 3]");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: Exception - {ex.Message}");
    failed++;
}

Console.WriteLine($"\n=== Results: {passed}/{passed + failed} tests passed ===");
Environment.Exit(failed > 0 ? 1 : 0);
