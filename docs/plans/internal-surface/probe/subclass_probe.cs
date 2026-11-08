#:project ../../../../src/NumSharp.Core
#:property PublishAot=false
#:property ReferenceNDArrayAnalyzer=false

// =============================================================================
// subclass_probe.cs - what a CONSUMER's NDArray subclass gets back from NumSharp
// =============================================================================
//
// The subclass is declared outside NumSharp.dll and uses public API only, as a consumer's would. Every line prints the
// class of one operation's result; NumPy's twin (subclass_probe.py) prints `MyArray` for all of them, because NumPy keeps
// an ndarray subclass through ufuncs, operators, views, methods and joins (__array_wrap__ / __array_finalize__ /
// __array_priority__). Run from the repository root: dotnet run docs/plans/internal-surface/probe/subclass_probe.cs
// =============================================================================

using NumSharp;

var x = new MyArray(np.arange(6.0).reshape(2, 3));
Report("x itself", x);
Report("np.add(x, x)", np.add(x, x));
Report("x + 1", x + 1);
Report("-x", -x);
Report("x[\"0\"] (row view)", x["0"]);
Report("x[\":, 1:\"] (slice view)", x[":, 1:"]);
Report("x.T", x.T);
Report("x.reshape(3, 2)", x.reshape(3, 2));
Report("np.sum(x, 0)", np.sum(x, 0));
Report("np.concatenate([x, x])", np.concatenate(new NDArray[] { x, x }));
Report("np.sqrt(x)", np.sqrt(x));
Report("x.copy()", x.copy());
Report("np.asanyarray(x)", np.asanyarray(x));
Console.WriteLine($"same object from np.asanyarray: {ReferenceEquals(np.asanyarray(x), x)}");

// Prints one result's class beside the operation that produced it.
static void Report(string what, NDArray result) => Console.WriteLine($"{what,-28} -> {result.GetType().Name}");

/// <summary>
///     A consumer's array subclass: a view of an existing array's storage with a NumPy class priority of its own (above
///     numpy.matrix's 10.0, so NumPy would wrap every ufunc result in this class).
/// </summary>
sealed class MyArray : NDArray
{
    /// <summary>Shares <paramref name="source"/>'s storage (no copy), through the public <c>Unsafe.Storage</c> accessor.</summary>
    /// <param name="source">The array whose storage the subclass views.</param>
    public MyArray(NDArray source) : base(source.Unsafe.Storage) { }

    /// <summary>NumPy's <c>__array_priority__</c>: public and virtual in NumSharp, the one protocol member a consumer can override.</summary>
    public override double __array_priority__ => 20.0;
}
