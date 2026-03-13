#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Configuration=Release
using NumSharp;
using System.Diagnostics;

double Benchmark(Action action, int iterations = 100)
{
    var times = new List<double>();
    for (int i = 0; i < iterations; i++)
    {
        var sw = Stopwatch.StartNew();
        action();
        sw.Stop();
        times.Add(sw.Elapsed.TotalMilliseconds);
    }
    return times.Average();
}

Console.WriteLine(new string('=', 70));
Console.WriteLine("NumSharp Reduction Benchmarks");
Console.WriteLine(new string('=', 70));
Console.WriteLine();

int[] sizes = { 100, 10000, 1000000 };
int iterations = 100;

Console.WriteLine($"Iterations per benchmark: {iterations}");
Console.WriteLine();

var random = new Random(42);

foreach (var size in sizes)
{
    Console.WriteLine($"\n--- Array size: {size:N0} elements ---");
    
    // Create array with random values
    var data = new double[size];
    for (int i = 0; i < size; i++)
        data[i] = random.NextDouble();
    var arr = np.array(data);
    
    // Warm up
    _ = np.sum(arr);
    
    var results = new Dictionary<string, double>
    {
        ["sum"] = Benchmark(() => np.sum(arr), iterations),
        ["prod"] = Benchmark(() => np.prod(arr), iterations),
        ["mean"] = Benchmark(() => np.mean(arr), iterations),
        ["min"] = Benchmark(() => np.amin(arr), iterations),
        ["max"] = Benchmark(() => np.amax(arr), iterations)
    };
    
    foreach (var (op, timeMs) in results)
    {
        Console.WriteLine($"  {op,-8}: {timeMs,10:F4} ms");
    }
}

Console.WriteLine("\n" + new string('=', 70));
Console.WriteLine("NumSharp MatMul Benchmarks");
Console.WriteLine(new string('=', 70));

int[] matrixSizes = { 100, 500, 1000 };
int matmulIterations = 10;

Console.WriteLine($"Iterations per benchmark: {matmulIterations}");
Console.WriteLine();

foreach (var size in matrixSizes)
{
    Console.WriteLine($"\n--- Matrix size: {size}x{size} ---");
    
    // Create matrices with random values
    var dataA = new double[size * size];
    var dataB = new double[size * size];
    for (int i = 0; i < size * size; i++)
    {
        dataA[i] = random.NextDouble();
        dataB[i] = random.NextDouble();
    }
    var A = np.array(dataA).reshape(size, size);
    var B = np.array(dataB).reshape(size, size);
    
    // Warm up
    _ = np.matmul(A, B);
    
    var timeMs = Benchmark(() => np.matmul(A, B), matmulIterations);
    
    // Calculate GFLOPS (2*N^3 for matrix multiply)
    double flops = 2.0 * Math.Pow(size, 3);
    double gflops = (flops / (timeMs / 1000)) / 1e9;
    
    Console.WriteLine($"  matmul:   {timeMs,10:F4} ms  ({gflops:F2} GFLOPS)");
}

Console.WriteLine("\n" + new string('=', 70));
Console.WriteLine("Summary: NumSharp benchmark complete");
Console.WriteLine(new string('=', 70));
