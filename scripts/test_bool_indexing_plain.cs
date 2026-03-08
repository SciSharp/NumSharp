#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;

Console.WriteLine("=== Testing Boolean Indexing with plain NDArray (BUG-1 verification) ===\n");

int passed = 0;
int failed = 0;

// Test 1: Plain NDArray (non-generic) boolean mask
try
{
    Console.WriteLine("Test 1: Plain NDArray (non-generic) boolean mask");
    var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
    // Create mask as plain NDArray, not NDArray<bool>
    NDArray mask = np.array(new[] { true, false, true, false, true, false });
    Console.WriteLine($"  a = [{string.Join(", ", a.ToArray<int>())}]");
    Console.WriteLine($"  mask type: {mask.GetType().Name}, typecode: {mask.typecode}");

    var result = a[mask];  // This was throwing before BUG-1 fix
    Console.WriteLine($"  a[mask] = [{string.Join(", ", result.ToArray<int>())}]");
    Console.WriteLine($"  Expected: [1, 3, 5]");

    var arr = result.ToArray<int>();
    if (arr.Length == 3 && arr[0] == 1 && arr[1] == 3 && arr[2] == 5)
    {
        Console.WriteLine("  PASS");
        passed++;
    }
    else
    {
        Console.WriteLine("  FAIL: Wrong values");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: Exception - {ex.GetType().Name}: {ex.Message}");
    failed++;
}

// Test 2: Comparison result (a % 2 == 1) which returns plain NDArray
try
{
    Console.WriteLine("\nTest 2: Comparison result as mask");
    var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
    var compResult = a % 2 == 1;
    Console.WriteLine($"  comparison result type: {compResult.GetType().Name}, typecode: {compResult.typecode}");

    var result = a[compResult];
    Console.WriteLine($"  a[a % 2 == 1] = [{string.Join(", ", result.ToArray<int>())}]");

    var arr = result.ToArray<int>();
    if (arr.Length == 3 && arr[0] == 1 && arr[1] == 3 && arr[2] == 5)
    {
        Console.WriteLine("  PASS");
        passed++;
    }
    else
    {
        Console.WriteLine("  FAIL: Wrong values");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: Exception - {ex.GetType().Name}: {ex.Message}");
    failed++;
}

// Test 3: NDArray<bool> (generic) - should use fast path
try
{
    Console.WriteLine("\nTest 3: NDArray<bool> (generic) mask");
    var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
    var mask = np.array(new[] { true, false, true, false, true, false }).MakeGeneric<bool>();
    Console.WriteLine($"  mask type: {mask.GetType().Name}");

    var result = a[mask];
    Console.WriteLine($"  a[mask] = [{string.Join(", ", result.ToArray<int>())}]");

    var arr = result.ToArray<int>();
    if (arr.Length == 3 && arr[0] == 1 && arr[1] == 3 && arr[2] == 5)
    {
        Console.WriteLine("  PASS");
        passed++;
    }
    else
    {
        Console.WriteLine("  FAIL: Wrong values");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: Exception - {ex.GetType().Name}: {ex.Message}");
    failed++;
}

Console.WriteLine($"\n=== Results: {passed}/{passed + failed} tests passed ===");
Environment.Exit(failed > 0 ? 1 : 0);
