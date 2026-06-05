using NumSharp;

namespace NumSharp.Benchmark.GraphEngine.Infrastructure;

/// <summary>
/// Provides type parameter sources for parameterized benchmarks.
/// Supports all 12 NumSharp NPTypeCodes.
/// </summary>
public static class TypeParameterSource
{
    /// <summary>
    /// All 15 NumSharp supported types.
    /// Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char, Half, Single, Double, Decimal, Complex
    /// </summary>
    public static IEnumerable<NPTypeCode> AllNumericTypes => new[]
    {
        NPTypeCode.Boolean,
        NPTypeCode.Byte,
        NPTypeCode.SByte,
        NPTypeCode.Int16,
        NPTypeCode.UInt16,
        NPTypeCode.Int32,
        NPTypeCode.UInt32,
        NPTypeCode.Int64,
        NPTypeCode.UInt64,
        NPTypeCode.Char,
        NPTypeCode.Half,
        NPTypeCode.Single,
        NPTypeCode.Double,
        NPTypeCode.Decimal,
        NPTypeCode.Complex
    };

    /// <summary>
    /// Integer types only: bool, byte, sbyte, int16, uint16, int32, uint32, int64, uint64, char
    /// </summary>
    public static IEnumerable<NPTypeCode> IntegerTypes => new[]
    {
        NPTypeCode.Boolean,
        NPTypeCode.Byte,
        NPTypeCode.SByte,
        NPTypeCode.Int16,
        NPTypeCode.UInt16,
        NPTypeCode.Int32,
        NPTypeCode.UInt32,
        NPTypeCode.Int64,
        NPTypeCode.UInt64,
        NPTypeCode.Char
    };

    /// <summary>
    /// Floating-point types only: half, float, double, decimal
    /// </summary>
    public static IEnumerable<NPTypeCode> FloatingTypes => new[]
    {
        NPTypeCode.Half,
        NPTypeCode.Single,
        NPTypeCode.Double,
        NPTypeCode.Decimal
    };

    /// <summary>
    /// Common types for fast benchmarks: int32, int64, float, double
    /// These cover the most common use cases with minimal runtime.
    /// </summary>
    public static IEnumerable<NPTypeCode> CommonTypes => new[]
    {
        NPTypeCode.Int32,
        NPTypeCode.Int64,
        NPTypeCode.Single,
        NPTypeCode.Double
    };

    /// <summary>
    /// Minimal types for smoke tests: int32, double
    /// </summary>
    public static IEnumerable<NPTypeCode> MinimalTypes => new[]
    {
        NPTypeCode.Int32,
        NPTypeCode.Double
    };

    /// <summary>
    /// Types that support standard arithmetic (+, -, *): excludes bool and char.
    /// Complex supports +,-,*,/ and (via magnitude) min/max; it has no modulo, but the
    /// modulo/divide benchmarks restrict themselves to CommonTypes, so it is safe here.
    /// </summary>
    public static IEnumerable<NPTypeCode> ArithmeticTypes => new[]
    {
        NPTypeCode.SByte,
        NPTypeCode.Byte,
        NPTypeCode.Int16,
        NPTypeCode.UInt16,
        NPTypeCode.Int32,
        NPTypeCode.UInt32,
        NPTypeCode.Int64,
        NPTypeCode.UInt64,
        NPTypeCode.Half,
        NPTypeCode.Single,
        NPTypeCode.Double,
        NPTypeCode.Decimal,
        NPTypeCode.Complex
    };

    /// <summary>
    /// Types that support transcendental functions (sqrt, exp, log, trig): half, float, double, decimal
    /// </summary>
    public static IEnumerable<NPTypeCode> TranscendentalTypes => new[]
    {
        NPTypeCode.Half,
        NPTypeCode.Single,
        NPTypeCode.Double,
        NPTypeCode.Decimal
    };

    /// <summary>
    /// Get the NumPy dtype name for a given NPTypeCode.
    /// Used for matching with Python benchmark results.
    /// </summary>
    public static string GetDtypeName(NPTypeCode code) => code switch
    {
        NPTypeCode.Boolean => "bool",
        NPTypeCode.Byte => "uint8",
        NPTypeCode.SByte => "int8",
        NPTypeCode.Int16 => "int16",
        NPTypeCode.UInt16 => "uint16",
        NPTypeCode.Int32 => "int32",
        NPTypeCode.UInt32 => "uint32",
        NPTypeCode.Int64 => "int64",
        NPTypeCode.UInt64 => "uint64",
        NPTypeCode.Char => "uint16",  // char is 16-bit in C#
        NPTypeCode.Half => "float16",
        NPTypeCode.Single => "float32",
        NPTypeCode.Double => "float64",
        NPTypeCode.Decimal => "float128",  // closest approximation (no exact NumPy peer)
        NPTypeCode.Complex => "complex128",
        _ => throw new ArgumentException($"Unknown NPTypeCode: {code}")
    };

    /// <summary>
    /// Get the short display name for a given NPTypeCode.
    /// </summary>
    public static string GetShortName(NPTypeCode code) => code switch
    {
        NPTypeCode.Boolean => "bool",
        NPTypeCode.Byte => "u8",
        NPTypeCode.Int16 => "i16",
        NPTypeCode.UInt16 => "u16",
        NPTypeCode.Int32 => "i32",
        NPTypeCode.UInt32 => "u32",
        NPTypeCode.Int64 => "i64",
        NPTypeCode.UInt64 => "u64",
        NPTypeCode.Char => "char",
        NPTypeCode.Single => "f32",
        NPTypeCode.Double => "f64",
        NPTypeCode.Decimal => "dec",
        _ => code.ToString()
    };
}
