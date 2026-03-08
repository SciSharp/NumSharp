#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;
using NumSharp.Backends.Kernels;

Console.WriteLine("=== Testing SIMD Round/Floor/Ceil ===\n");
Console.WriteLine($"SIMD enabled: {ILKernelGenerator.Enabled}, VectorBits: {ILKernelGenerator.VectorBits}");

int passed = 0;
int failed = 0;

void TestOpDouble(string name, NDArray input, double[] expected, Func<NDArray, NDArray> op)
{
    try
    {
        var result = op(input);
        var resultArr = result.ToArray<double>();

        bool match = resultArr.Length == expected.Length;
        if (match)
        {
            for (int i = 0; i < resultArr.Length; i++)
            {
                if (Math.Abs(resultArr[i] - expected[i]) > 1e-10)
                {
                    match = false;
                    break;
                }
            }
        }

        if (match)
        {
            Console.WriteLine($"  {name}: PASS");
            passed++;
        }
        else
        {
            Console.WriteLine($"  {name}: FAIL");
            Console.WriteLine($"    Expected: [{string.Join(", ", expected)}]");
            Console.WriteLine($"    Got:      [{string.Join(", ", resultArr)}]");
            failed++;
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  {name}: FAIL - {ex.GetType().Name}: {ex.Message}");
        failed++;
    }
}

void TestOpFloat(string name, NDArray input, float[] expected, Func<NDArray, NDArray> op)
{
    try
    {
        var result = op(input);
        var resultArr = result.ToArray<float>();

        bool match = resultArr.Length == expected.Length;
        if (match)
        {
            for (int i = 0; i < resultArr.Length; i++)
            {
                if (Math.Abs(resultArr[i] - expected[i]) > 1e-5f)
                {
                    match = false;
                    break;
                }
            }
        }

        if (match)
        {
            Console.WriteLine($"  {name}: PASS");
            passed++;
        }
        else
        {
            Console.WriteLine($"  {name}: FAIL");
            Console.WriteLine($"    Expected: [{string.Join(", ", expected)}]");
            Console.WriteLine($"    Got:      [{string.Join(", ", resultArr)}]");
            failed++;
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  {name}: FAIL - {ex.GetType().Name}: {ex.Message}");
        failed++;
    }
}

// Test data - values that test rounding edge cases
var testDouble = np.array(new double[] { 1.1, 1.5, 1.9, 2.5, -1.1, -1.5, -1.9, -2.5 });
var testFloat = np.array(new float[] { 1.1f, 1.5f, 1.9f, 2.5f, -1.1f, -1.5f, -1.9f, -2.5f });

// Large array to ensure SIMD path is used
var largeDouble = np.arange(0, 1000).astype(NPTypeCode.Double) + 0.5;
var largeFloat = np.arange(0, 1000).astype(NPTypeCode.Single) + 0.5f;

Console.WriteLine("\n--- np.round_ (banker's rounding, ToEven) ---");
// NumPy uses banker's rounding (round half to even)
// 1.5 -> 2, 2.5 -> 2, -1.5 -> -2, -2.5 -> -2
TestOpDouble("double[]", testDouble, new double[] { 1.0, 2.0, 2.0, 2.0, -1.0, -2.0, -2.0, -2.0 }, x => np.round_(x));
TestOpFloat("float[]", testFloat, new float[] { 1.0f, 2.0f, 2.0f, 2.0f, -1.0f, -2.0f, -2.0f, -2.0f }, x => np.round_(x));

// Test large array to ensure SIMD path
var roundedLarge = np.round_(largeDouble);
var allIntegers = roundedLarge.ToArray<double>().All(x => x == Math.Round(x));
Console.WriteLine($"  large double[1000]: {(allIntegers ? "PASS" : "FAIL")} (all values are integers)");
if (allIntegers) passed++; else failed++;

Console.WriteLine("\n--- np.floor ---");
TestOpDouble("double[]", testDouble, new double[] { 1.0, 1.0, 1.0, 2.0, -2.0, -2.0, -2.0, -3.0 }, x => np.floor(x));
TestOpFloat("float[]", testFloat, new float[] { 1.0f, 1.0f, 1.0f, 2.0f, -2.0f, -2.0f, -2.0f, -3.0f }, x => np.floor(x));

Console.WriteLine("\n--- np.ceil ---");
TestOpDouble("double[]", testDouble, new double[] { 2.0, 2.0, 2.0, 3.0, -1.0, -1.0, -1.0, -2.0 }, x => np.ceil(x));
TestOpFloat("float[]", testFloat, new float[] { 2.0f, 2.0f, 2.0f, 3.0f, -1.0f, -1.0f, -1.0f, -2.0f }, x => np.ceil(x));

Console.WriteLine($"\n=== Results: {passed}/{passed + failed} tests passed ===");
Environment.Exit(failed > 0 ? 1 : 0);
