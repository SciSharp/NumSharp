#:project ../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using NumSharp;

Console.WriteLine("=== Testing np.convolve ===\n");

try
{
    var a = np.array(new[] { 1.0, 2.0, 3.0 });
    var v = np.array(new[] { 0.5, 1.0 });
    var result = np.convolve(a, v);
    Console.WriteLine($"np.convolve([1,2,3], [0.5,1]) = [{string.Join(", ", result.ToArray<double>())}]");
}
catch (NotImplementedException ex)
{
    Console.WriteLine($"EXPECTED: NotImplementedException - {ex.Message}");
    Console.WriteLine("\nThis is expected. The function now throws a proper exception");
    Console.WriteLine("instead of returning null (which caused NullReferenceException).");
}
catch (NullReferenceException ex)
{
    Console.WriteLine($"BUG STILL EXISTS: NullReferenceException - {ex.Message}");
}
catch (Exception ex)
{
    Console.WriteLine($"OTHER: {ex.GetType().Name} - {ex.Message}");
}
