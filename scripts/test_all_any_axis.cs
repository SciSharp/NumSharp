#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;

Console.WriteLine("=== Testing np.all/any with axis parameter ===\n");

int passed = 0;
int failed = 0;

// Test 1: np.all flat (no axis) - should work
try
{
    var arr = np.array(new[,] { { true, true }, { true, false } });
    var result = np.all(arr);
    Console.WriteLine($"Test 1: np.all(2D array) = {result}");
    if (result == false) { Console.WriteLine("  PASS"); passed++; }
    else { Console.WriteLine("  FAIL: expected false"); failed++; }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: {ex.GetType().Name}: {ex.Message}");
    failed++;
}

// Test 2: np.any flat (no axis) - should work
try
{
    var arr = np.array(new[,] { { false, false }, { false, true } });
    var result = np.any(arr);
    Console.WriteLine($"Test 2: np.any(2D array) = {result}");
    if (result == true) { Console.WriteLine("  PASS"); passed++; }
    else { Console.WriteLine("  FAIL: expected true"); failed++; }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: {ex.GetType().Name}: {ex.Message}");
    failed++;
}

// Test 3: np.all with axis=0 - THIS IS THE BUG
try
{
    var arr = np.array(new[,] { { 1, 2 }, { 3, 0 } });
    Console.WriteLine($"Test 3: np.all(arr, axis=0) where arr=[[1,2],[3,0]]");
    var result = np.all(arr, axis: 0);
    Console.WriteLine($"  Result: [{string.Join(", ", result.ToArray<bool>())}]");
    Console.WriteLine($"  Expected: [True, False] (column 0 all non-zero, column 1 has zero)");
    var resArr = result.ToArray<bool>();
    if (resArr.Length == 2 && resArr[0] == true && resArr[1] == false)
    {
        Console.WriteLine("  PASS");
        passed++;
    }
    else
    {
        Console.WriteLine("  FAIL: wrong values");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: {ex.GetType().Name}: {ex.Message}");
    failed++;
}

// Test 4: np.all with axis=1
try
{
    var arr = np.array(new[,] { { 1, 2 }, { 3, 0 } });
    Console.WriteLine($"Test 4: np.all(arr, axis=1) where arr=[[1,2],[3,0]]");
    var result = np.all(arr, axis: 1);
    Console.WriteLine($"  Result: [{string.Join(", ", result.ToArray<bool>())}]");
    Console.WriteLine($"  Expected: [True, False] (row 0 all non-zero, row 1 has zero)");
    var resArr = result.ToArray<bool>();
    if (resArr.Length == 2 && resArr[0] == true && resArr[1] == false)
    {
        Console.WriteLine("  PASS");
        passed++;
    }
    else
    {
        Console.WriteLine("  FAIL: wrong values");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: {ex.GetType().Name}: {ex.Message}");
    failed++;
}

// Test 5: np.any with axis=0
try
{
    var arr = np.array(new[,] { { 0, 1 }, { 0, 0 } });
    Console.WriteLine($"Test 5: np.any(arr, axis=0) where arr=[[0,1],[0,0]]");
    var result = np.any(arr, axis: 0);
    Console.WriteLine($"  Result: [{string.Join(", ", result.ToArray<bool>())}]");
    Console.WriteLine($"  Expected: [False, True] (column 0 all zero, column 1 has non-zero)");
    var resArr = result.ToArray<bool>();
    if (resArr.Length == 2 && resArr[0] == false && resArr[1] == true)
    {
        Console.WriteLine("  PASS");
        passed++;
    }
    else
    {
        Console.WriteLine("  FAIL: wrong values");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: {ex.GetType().Name}: {ex.Message}");
    failed++;
}

// Test 6: np.any with axis=1
try
{
    var arr = np.array(new[,] { { 0, 1 }, { 0, 0 } });
    Console.WriteLine($"Test 6: np.any(arr, axis=1) where arr=[[0,1],[0,0]]");
    var result = np.any(arr, axis: 1);
    Console.WriteLine($"  Result: [{string.Join(", ", result.ToArray<bool>())}]");
    Console.WriteLine($"  Expected: [True, False] (row 0 has non-zero, row 1 all zero)");
    var resArr = result.ToArray<bool>();
    if (resArr.Length == 2 && resArr[0] == true && resArr[1] == false)
    {
        Console.WriteLine("  PASS");
        passed++;
    }
    else
    {
        Console.WriteLine("  FAIL: wrong values");
        failed++;
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  FAIL: {ex.GetType().Name}: {ex.Message}");
    failed++;
}

Console.WriteLine($"\n=== Results: {passed}/{passed + failed} tests passed ===");
Environment.Exit(failed > 0 ? 1 : 0);
