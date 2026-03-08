#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

// Force recompile v3
using NumSharp;

Console.WriteLine("=== Testing int32 overflow in arange/sum ===\n");

// Test 1: arange default type
Console.WriteLine("Test 1: np.arange default type");
var arr1 = np.arange(10);
Console.WriteLine($"  np.arange(10).dtype = {arr1.dtype.Name}");
Console.WriteLine($"  NumPy 2.x uses: int64 by default for integer ranges");

// Test 2: Large sum that overflows int32
Console.WriteLine("\nTest 2: Sum that overflows int32");
var arr2 = np.arange(100000);  // 0 to 99999
var sum2 = np.sum(arr2);
long expectedSum = (long)99999 * 100000 / 2;  // Gauss formula: n*(n-1)/2
Console.WriteLine($"  np.arange(100000).sum()");
Console.WriteLine($"  Expected (correct): {expectedSum}");
Console.WriteLine($"  Got: {sum2}");
Console.WriteLine($"  Sum dtype: {sum2.dtype.Name}");
Console.WriteLine($"  int32 max: {int.MaxValue}");

if (expectedSum > int.MaxValue)
{
    Console.WriteLine($"  Expected > int32.MaxValue, so overflow would occur with int32");
}

// Test 3: Even larger - guaranteed overflow
Console.WriteLine("\nTest 3: Guaranteed overflow scenario");
var arr3 = np.arange(70000);
var sum3 = np.sum(arr3);
long expectedSum3 = (long)69999 * 70000 / 2;
Console.WriteLine($"  np.arange(70000).sum()");
Console.WriteLine($"  Expected (correct): {expectedSum3}");
Console.WriteLine($"  Got: {sum3}");

// Check if negative (overflow indicator)
var sumValue = Convert.ToInt64(sum3.GetAtIndex(0));
if (sumValue < 0 || sumValue != expectedSum3)
{
    Console.WriteLine($"  OVERFLOW DETECTED: result is wrong");
}
else
{
    Console.WriteLine($"  CORRECT: no overflow");
}

// Test 4: What type does sum return?
Console.WriteLine("\nTest 4: Sum accumulator type");
var arr4 = np.arange(10);
var sum4 = np.sum(arr4);
Console.WriteLine($"  Input dtype: {arr4.dtype.Name}");
Console.WriteLine($"  Sum result dtype: {sum4.dtype.Name}");
Console.WriteLine($"  NumPy 2.x behavior: int32 input -> int64 sum (safe)");

Console.WriteLine("\n=== Analysis Complete ===");
