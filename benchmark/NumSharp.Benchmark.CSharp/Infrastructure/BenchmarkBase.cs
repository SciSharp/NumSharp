using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using NumSharp;

namespace NumSharp.Benchmark.CSharp.Infrastructure;

/// <summary>
/// Base class for NumSharp benchmarks with standard setup patterns.
/// Provides array creation and type parameterization.
/// </summary>
[MemoryDiagnoser]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[RankColumn]
[CategoriesColumn]
public abstract class BenchmarkBase
{
    /// <summary>
    /// Array size parameter. Override in derived class with [Params] or [ParamsSource].
    /// </summary>
    public virtual int N { get; set; } = ArraySizeSource.Large;

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    protected const int Seed = 42;

    /// <summary>
    /// Create a random array of the specified type and size.
    /// </summary>
    protected static NDArray CreateRandomArray(int n, NPTypeCode dtype, int seed = Seed)
    {
        np.random.seed(seed);

        return dtype switch
        {
            NPTypeCode.Boolean => np.random.randint(0, 2, new Shape(n)).astype(np.@bool),
            NPTypeCode.Byte => np.random.randint(0, 256, new Shape(n)).astype(np.uint8),
            NPTypeCode.SByte => np.random.randint(-100, 100, new Shape(n)).astype(NPTypeCode.SByte),
            NPTypeCode.Int16 => np.random.randint(-1000, 1000, new Shape(n)).astype(np.int16),
            NPTypeCode.UInt16 => np.random.randint(0, 2000, new Shape(n)).astype(np.uint16),
            NPTypeCode.Int32 => np.random.randint(-1000, 1000, new Shape(n)),
            NPTypeCode.UInt32 => np.random.randint(0, 2000, new Shape(n)).astype(np.uint32),
            NPTypeCode.Int64 => np.random.randint(-1000, 1000, new Shape(n)).astype(np.int64),
            NPTypeCode.UInt64 => np.random.randint(0, 2000, new Shape(n)).astype(np.uint64),
            NPTypeCode.Char => np.random.randint(32, 127, new Shape(n)).astype(NPTypeCode.Char),
            NPTypeCode.Half => (np.random.rand(n) * 100 - 50).astype(NPTypeCode.Half),
            NPTypeCode.Single => (np.random.rand(n) * 100 - 50).astype(np.float32),
            NPTypeCode.Double => np.random.rand(n) * 100 - 50,
            NPTypeCode.Decimal => (np.random.rand(n) * 100 - 50).astype(NPTypeCode.Decimal),
            NPTypeCode.Complex => (np.random.rand(n) * 100 - 50).astype(NPTypeCode.Complex),
            _ => throw new ArgumentException($"Unsupported type: {dtype}")
        };
    }

    /// <summary>
    /// Create a random 2D array of the specified type and shape.
    /// </summary>
    protected static NDArray CreateRandomArray2D(int rows, int cols, NPTypeCode dtype, int seed = Seed)
    {
        np.random.seed(seed);
        var flat = CreateRandomArray(rows * cols, dtype, seed);
        return flat.reshape(rows, cols);
    }

    /// <summary>
    /// Create a random 3D array of the specified type and shape.
    /// </summary>
    protected static NDArray CreateRandomArray3D(int d1, int d2, int d3, NPTypeCode dtype, int seed = Seed)
    {
        np.random.seed(seed);
        var flat = CreateRandomArray(d1 * d2 * d3, dtype, seed);
        return flat.reshape(d1, d2, d3);
    }

    /// <summary>
    /// Create a positive random array (for operations like log, sqrt, division, modulo).
    /// Returns values >= 1 to avoid divide-by-zero errors.
    /// </summary>
    protected static NDArray CreatePositiveArray(int n, NPTypeCode dtype, int seed = Seed)
    {
        np.random.seed(seed);

        return dtype switch
        {
            // Floating point: rand() * 100 + 1 gives [1, 101)
            NPTypeCode.Half => (np.random.rand(n) * 100 + 1).astype(NPTypeCode.Half),
            NPTypeCode.Single => (np.random.rand(n) * 100 + 1).astype(np.float32),
            NPTypeCode.Double => np.random.rand(n) * 100 + 1,
            NPTypeCode.Decimal => (np.random.rand(n) * 100 + 1).astype(NPTypeCode.Decimal),
            NPTypeCode.Complex => (np.random.rand(n) * 100 + 1).astype(NPTypeCode.Complex),
            // Integers: randint(1, N) gives [1, N) - never zero
            NPTypeCode.Boolean => np.random.randint(1, 2, new Shape(n)).astype(np.@bool),
            NPTypeCode.Byte => np.random.randint(1, 256, new Shape(n)).astype(np.uint8),
            NPTypeCode.SByte => np.random.randint(1, 100, new Shape(n)).astype(NPTypeCode.SByte),
            NPTypeCode.Int16 => np.random.randint(1, 1000, new Shape(n)).astype(np.int16),
            NPTypeCode.UInt16 => np.random.randint(1, 2000, new Shape(n)).astype(np.uint16),
            NPTypeCode.Int32 => np.random.randint(1, 1000, new Shape(n)),
            NPTypeCode.UInt32 => np.random.randint(1, 2000, new Shape(n)).astype(np.uint32),
            NPTypeCode.Int64 => np.random.randint(1, 1000, new Shape(n)).astype(np.int64),
            NPTypeCode.UInt64 => np.random.randint(1, 2000, new Shape(n)).astype(np.uint64),
            NPTypeCode.Char => np.random.randint(1, 127, new Shape(n)).astype(NPTypeCode.Char),
            _ => throw new ArgumentException($"Unsupported type: {dtype}")
        };
    }

    /// <summary>
    /// Get a scalar value of the specified type.
    /// </summary>
    protected static object GetScalar(NPTypeCode dtype, double value = 42.0) => dtype switch
    {
        NPTypeCode.Boolean => value != 0,
        NPTypeCode.Byte => (byte)Math.Abs(value),
        NPTypeCode.SByte => (sbyte)value,
        NPTypeCode.Int16 => (short)value,
        NPTypeCode.UInt16 => (ushort)Math.Abs(value),
        NPTypeCode.Int32 => (int)value,
        NPTypeCode.UInt32 => (uint)Math.Abs(value),
        NPTypeCode.Int64 => (long)value,
        NPTypeCode.UInt64 => (ulong)Math.Abs(value),
        NPTypeCode.Char => (char)(int)Math.Abs(value),
        NPTypeCode.Half => (Half)value,
        NPTypeCode.Single => (float)value,
        NPTypeCode.Double => value,
        NPTypeCode.Decimal => (decimal)value,
        NPTypeCode.Complex => new System.Numerics.Complex(value, 0),
        _ => throw new ArgumentException($"Unsupported type: {dtype}")
    };

    /// <summary>
    /// Execute an operation for a dtype whose NumPy-compatible contract is rejection, and turn
    /// that expected rejection into a successful benchmark result. This keeps unsupported cells
    /// executable/auditable without pretending they perform an elementwise kernel. Such C#-only
    /// evidence has no NumPy timing peer and therefore never enters comparison rollups.
    /// </summary>
    protected static object VerifyUnsupportedDtype(Func<object> operation)
    {
        try
        {
            var unexpected = operation();
            (unexpected as IDisposable)?.Dispose();
            throw new InvalidOperationException("The dtype was expected to be rejected, but the operation succeeded.");
        }
        catch (TypeError)
        {
            return UnsupportedDtypeEvidence.Instance;
        }
        catch (NotSupportedException)
        {
            return UnsupportedDtypeEvidence.Instance;
        }
        catch (InvalidCastException)
        {
            return UnsupportedDtypeEvidence.Instance;
        }
        catch (IncorrectTypeException)
        {
            return UnsupportedDtypeEvidence.Instance;
        }
    }

    /// <summary>
    /// Record a known unsupported dtype without invoking a path that cannot safely report a
    /// managed exception (for example, a process-fatal invalid kernel combination).
    /// </summary>
    protected static object RecordUnsafeUnsupportedDtype() => UnsupportedDtypeEvidence.Instance;

    private sealed class UnsupportedDtypeEvidence
    {
        public static readonly UnsupportedDtypeEvidence Instance = new();
        private UnsupportedDtypeEvidence() { }
    }
}

/// <summary>
/// Base for official scenarios whose class does not already provide its own dtype parameter.
/// Dtype-agnostic APIs still receive the parameter so the audit can state explicitly that the
/// function is executable for every requested dtype; array-bearing classes consume it in setup.
/// Experimental microbenchmarks deliberately remain on <see cref="BenchmarkBase"/> so their case
/// counts are not multiplied by the official function × dtype matrix.
/// </summary>
public abstract class OfficialBenchmarkBase : BenchmarkBase
{
    [ParamsSource(nameof(ScenarioTypes))]
    public NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> ScenarioTypes => TypeParameterSource.AllNumericTypes;
}

/// <summary>
/// Base class for benchmarks parameterized by both size and type.
/// Derived classes must define their own Types property with [ParamsSource].
/// </summary>
public abstract class TypedBenchmarkBase : BenchmarkBase
{
    /// <summary>
    /// Type parameter. Derived class must provide [ParamsSource].
    /// </summary>
    public NPTypeCode DType { get; set; }
}
