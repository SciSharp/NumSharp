#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;

Console.WriteLine("=== Testing Boolean Indexing (BUG-1 Investigation) ===\n");

// Test 1: Explicit mask on 1D array
Console.WriteLine("Test 1: Explicit mask on 1D array");
var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
Console.WriteLine($"  a = [{string.Join(", ", a.ToArray<int>())}]");
var mask = np.array(new[] { true, false, true, false, true, false }).MakeGeneric<bool>();
Console.WriteLine($"  mask = [{string.Join(", ", mask.ToArray<bool>())}]");
var result = a[mask];
Console.WriteLine($"  a[mask] = [{string.Join(", ", result.ToArray<int>())}]");
Console.WriteLine($"  Expected: [1, 3, 5]");
if (result.size == 3)
{
    var arr = result.ToArray<int>();
    if (arr[0] == 1 && arr[1] == 3 && arr[2] == 5)
        Console.WriteLine("  PASS");
    else
        Console.WriteLine($"  FAIL: Got wrong values");
}
else
{
    Console.WriteLine($"  FAIL: Expected size 3, got {result.size}");
}

// Test 2: Condition-based mask
Console.WriteLine("\nTest 2: Condition-based mask (a % 2 == 1)");
var a2 = np.array(new[] { 1, 2, 3, 4, 5, 6 });
Console.WriteLine($"  a = [{string.Join(", ", a2.ToArray<int>())}]");
var condMask = a2 % 2 == 1;
Console.WriteLine($"  a % 2 == 1 gives mask of shape {string.Join(",", condMask.shape)}, size {condMask.size}");
var result2 = a2[condMask];
Console.WriteLine($"  a[a % 2 == 1] = [{string.Join(", ", result2.ToArray<int>())}]");
Console.WriteLine($"  Expected: [1, 3, 5]");
if (result2.size == 3)
{
    var arr = result2.ToArray<int>();
    if (arr[0] == 1 && arr[1] == 3 && arr[2] == 5)
        Console.WriteLine("  PASS");
    else
        Console.WriteLine($"  FAIL: Got wrong values");
}
else
{
    Console.WriteLine($"  FAIL: Expected size 3, got {result2.size}");
}

// Test 3: 2D array with 1D mask (row selection)
Console.WriteLine("\nTest 3: 2D array with 1D mask (row selection)");
var a3 = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
Console.WriteLine($"  a = [[1,2,3],[4,5,6],[7,8,9]], shape={string.Join(",", a3.shape)}");
var mask3 = np.array(new[] { true, false, true }).MakeGeneric<bool>();
Console.WriteLine($"  mask = [True, False, True]");
var result3 = a3[mask3];
Console.WriteLine($"  a[mask] shape: {string.Join(",", result3.shape)}");
Console.WriteLine($"  a[mask] = {result3}");
Console.WriteLine($"  Expected: [[1,2,3],[7,8,9]] with shape (2,3)");

// Test 4: 2D array with 2D mask (element selection, flattens)
Console.WriteLine("\nTest 4: 2D array with 2D mask (element selection)");
var a4 = np.array(new[,] { { 1, 2 }, { 3, 4 } });
Console.WriteLine($"  a = [[1,2],[3,4]], shape={string.Join(",", a4.shape)}");
var mask4 = np.array(new[,] { { true, false }, { false, true } }).MakeGeneric<bool>();
Console.WriteLine($"  mask = [[True,False],[False,True]]");
var result4 = a4[mask4];
Console.WriteLine($"  a[mask] shape: {string.Join(",", result4.shape)}");
Console.WriteLine($"  a[mask] = [{string.Join(", ", result4.ToArray<int>())}]");
Console.WriteLine($"  Expected: [1, 4] with shape (2,)");

// Test 5: All false mask
Console.WriteLine("\nTest 5: All false mask");
var a5 = np.array(new[] { 1, 2, 3 });
var mask5 = np.array(new[] { false, false, false }).MakeGeneric<bool>();
var result5 = a5[mask5];
Console.WriteLine($"  a[all false] shape: {string.Join(",", result5.shape)}, size: {result5.size}");
Console.WriteLine($"  Expected: empty array with shape (0,)");

// Test 6: All true mask
Console.WriteLine("\nTest 6: All true mask");
var a6 = np.array(new[] { 1, 2, 3 });
var mask6 = np.array(new[] { true, true, true }).MakeGeneric<bool>();
var result6 = a6[mask6];
Console.WriteLine($"  a[all true] = [{string.Join(", ", result6.ToArray<int>())}]");
Console.WriteLine($"  Expected: [1, 2, 3]");

Console.WriteLine("\n=== Investigation Complete ===");
