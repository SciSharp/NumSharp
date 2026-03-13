#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
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
Console.WriteLine("NumSharp Int32 Reduction Benchmarks");
Console.WriteLine(new string('=', 70));

int[] sizes = { 100, 10000, 1000000 };
int iterations = 100;

var random = new Random(42);

foreach (var size in sizes)
{
    Console.WriteLine($"\n--- Array size: {size:N0} elements (int32) ---");
    
    var data = new int[size];
    for (int i = 0; i < size; i++)
        data[i] = random.Next(0, 1000);
    var arr = np.array(data);
    
    _ = np.sum(arr);  // warm up
    
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

Console.WriteLine();
