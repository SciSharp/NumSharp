using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    /// <summary>
    /// Machine limits for floating point types.
    /// </summary>
    /// <remarks>
    /// https://numpy.org/doc/stable/reference/generated/numpy.finfo.html
    /// </remarks>
    public sealed class finfo
    {
        /// <summary>
        /// Number of bits occupied by the type.
        /// </summary>
        public int bits { get; }

        /// <summary>
        /// The difference between 1.0 and the next smallest representable float larger than 1.0.
        /// </summary>
        public double eps { get; }

        /// <summary>
        /// The difference between 1.0 and the next smallest representable float smaller than 1.0.
        /// </summary>
        public double epsneg { get; }

        /// <summary>
        /// The largest representable number.
        /// </summary>
        public double max { get; }

        /// <summary>
        /// The smallest representable number (typically -max).
        /// </summary>
        public double min { get; }

        /// <summary>
        /// The smallest positive usable number (alias for smallest_normal).
        /// </summary>
        public double tiny { get; }

        /// <summary>
        /// The smallest positive normal number.
        /// </summary>
        public double smallest_normal { get; }

        /// <summary>
        /// The smallest positive subnormal number.
        /// </summary>
        public double smallest_subnormal { get; }

        /// <summary>
        /// The approximate number of decimal digits to which this kind of float is precise.
        /// </summary>
        public int precision { get; }

        /// <summary>
        /// The approximate decimal resolution of this type (10^-precision).
        /// </summary>
        public double resolution { get; }

        /// <summary>
        /// The smallest positive power of the base (2) that causes overflow.
        /// </summary>
        public int maxexp { get; }

        /// <summary>
        /// The most negative power of the base (2) consistent with there being no leading 0s in the mantissa — NumPy's
        /// <c>minexp</c>, so <c>2**minexp == smallest_normal</c>: -14 (float16), -126 (float32), -1022 (float64 and
        /// complex128). NumSharp reported -125 / -1021 (C's <c>FLT_MIN_EXP</c> convention) before API-parity batch 7.
        /// </summary>
        public int minexp { get; }

        /// <summary>
        /// NumPy's <c>nmant</c>: the number of bits in the stored mantissa (the fraction, without the implicit leading 1) —
        /// 10 (float16), 23 (float32), 52 (float64 and complex128). <c>eps == 2**-nmant</c> for the IEEE formats.
        /// </summary>
        /// <remarks>NumSharp-only for <c>decimal</c>: 96, the bits of its integer coefficient.</remarks>
        public int nmant { get; }

        /// <summary>
        /// NumPy's <c>iexp</c>: the number of bits in the exponent field — 5 (float16), 8 (float32), 11 (float64 and
        /// complex128). The same value as <see cref="nexp"/> (NumPy keeps both names).
        /// </summary>
        /// <remarks>NumSharp-only for <c>decimal</c>: 8, the byte that holds its scale.</remarks>
        public int iexp { get; }

        /// <summary>
        /// NumPy's <c>nexp</c>: the number of bits in the exponent including its sign and bias — the same value as
        /// <see cref="iexp"/> for every format NumPy has (5 / 8 / 11).
        /// </summary>
        /// <remarks>NumSharp-only for <c>decimal</c>: 8, as <see cref="iexp"/>.</remarks>
        public int nexp { get; }

        /// <summary>
        /// NumPy's <c>machep</c>: the exponent that yields <see cref="eps"/>, <c>eps == 2**machep</c> — -10 (float16),
        /// -23 (float32), -52 (float64 and complex128).
        /// </summary>
        /// <remarks>NumSharp-only for <c>decimal</c>: -28, a power of TEN (<c>eps == 10**-28</c>): decimal is a base-10
        ///     format, which its <see cref="eps"/> and <see cref="minexp"/> already follow.</remarks>
        public int machep { get; }

        /// <summary>
        /// NumPy's <c>negep</c>: the exponent that yields <see cref="epsneg"/>, <c>epsneg == 2**negep</c> — -11 (float16),
        /// -24 (float32), -53 (float64 and complex128).
        /// </summary>
        /// <remarks>NumSharp-only for <c>decimal</c>: -28, a power of ten like <see cref="machep"/>.</remarks>
        public int negep { get; }

        /// <summary>
        /// The dtype of this floating point type — NumPy's <c>finfo.dtype</c> (a <see cref="DType"/> descriptor,
        /// like <c>np.finfo(np.float64).dtype</c>). For a complex input this is the underlying REAL float dtype
        /// (<c>float64</c> for complex128), matching NumPy. Being a <see cref="DType"/> it exposes
        /// <see cref="DType.kind"/> / <see cref="DType.name"/> / <see cref="DType.itemsize"/> and compares
        /// structurally (<c>info.dtype == np.float64</c>); an <see cref="NPTypeCode"/> also converts back implicitly.
        /// </summary>
        public DType dtype { get; }

        /// <summary>
        /// Create finfo for the specified dtype.
        /// </summary>
        /// <param name="dtype">A floating point dtype (a <see cref="Type"/>, <see cref="NPTypeCode"/>, dtype string or <see cref="DType"/> — all convert implicitly).</param>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="dtype"/> is not a floating point (or complex) type — NumPy's text,
        ///     which names the dtype's SCALAR CLASS: <c>data type &lt;class 'numpy.int32'&gt; not inexact</c>,
        ///     <c>&lt;class 'numpy.str_'&gt;</c> for text. A <see cref="ValueError"/> is an <see cref="ArgumentException"/>, so
        ///     code that catches the latter still catches it.</exception>
        public finfo(DType dtype)
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));

            // Text is not inexact: NumPy names the scalar type (data type <class 'numpy.str_'> not inexact).
            if (dtype.Meta is StringDTypeMeta text)
                throw new ValueError($"data type <class 'numpy.{text.ScalarName}'> not inexact");

            NPTypeCode typeCode = dtype.GetTypeCode();
            // NumPy: `newdtype = obj2sctype(dtype); if not issubclass(newdtype, inexact): raise ValueError(f"data type
            // {newdtype!r} not inexact")` — the repr of the dtype's scalar CLASS (<class 'numpy.int32'>, <class 'numpy.bool'>),
            // which NDScalarType.Of renders. Before API-parity batch 7 this was an ArgumentException naming the dtype
            // ('int32') with .NET's "(Parameter 'dtype')" suffix.
            if (!IsFloatType(typeCode))
                throw new ValueError($"data type {NDScalarType.Of(dtype.Meta)} not inexact");

            // NumPy parity: np.finfo(np.complex128).dtype == np.float64.
            // The finfo represents the precision of the underlying real component, so
            // we report float64's machine limits with dtype set to the real type.
            // System.Numerics.Complex is 2 × float64 → underlying dtype is Double. NumPy's _convert_to_float maps each
            // complex CLASS to its real class, and a real class to itself, so the long-double classes stay long double:
            // np.finfo(np.longdouble).dtype and np.finfo(np.clongdouble).dtype are both 'g' (the float64 lane).
            if (ReferenceEquals(dtype.Meta, DTypeRegistry.LongDouble) || ReferenceEquals(dtype.Meta, DTypeRegistry.CLongDouble))
                this.dtype = DTypeRegistry.LongDouble.singleton;
            else
                this.dtype = typeCode == NPTypeCode.Complex ? NPTypeCode.Double : typeCode;

            switch (typeCode)
            {
                case NPTypeCode.Half:
                    // IEEE 754 binary16: 1 sign + 5 exponent + 10 mantissa bits.
                    bits = 16;
                    eps = 0.0009765625;              // 2^-10
                    epsneg = 0.00048828125;          // 2^-11
                    max = (double)Half.MaxValue;      // 65504
                    min = (double)Half.MinValue;      // -65504
                    smallest_normal = 6.103515625e-05; // 2^-14
                    smallest_subnormal = 5.960464477539063e-08; // 2^-24 (= (double)Half.Epsilon)
                    tiny = smallest_normal;
                    precision = 3;                    // decimal digits of precision
                    resolution = 1e-3;                // 10^-precision
                    maxexp = 16;                      // bias+1 = 2^15*(2-eps) = MaxValue
                    minexp = -14;                     // 2^-14 = smallest normal
                    nmant = 10;                       // NumPy's _MACHAR_PARAMS for half: 10 fraction bits
                    iexp = nexp = 5;                  // 5 exponent bits
                    machep = -10;                     // eps = 2^-10
                    negep = -11;                      // epsneg = 2^-11
                    break;

                case NPTypeCode.Single:
                    bits = 32;
                    // float.Epsilon is the smallest subnormal
                    // Machine epsilon for float: MathF.BitIncrement(1.0f) - 1.0f ≈ 1.1920929e-07
                    // Note: Must use MathF, not Math (which only works on double)
                    eps = MathF.BitIncrement(1.0f) - 1.0f;
                    epsneg = 1.0f - MathF.BitDecrement(1.0f);
                    max = float.MaxValue;
                    min = -float.MaxValue;
                    // 2^-126 EXACTLY (1.1754943508222875e-38), NumPy's np.float32(1.1754944e-38) widened. The value used to be
                    // the 10-digit decimal 1.175494351e-38, which is not 2^-126 (smallest_normal != 2**minexp by 1.8e-48).
                    smallest_normal = 1.1754943508222875e-38;
                    smallest_subnormal = float.Epsilon;  // ~1.4e-45
                    tiny = smallest_normal;
                    precision = 6;
                    resolution = 1e-6;
                    maxexp = 128;   // 2^127 * (2-eps) = MaxValue
                    // NumPy's minexp is the exponent OF the smallest normal (2^-126), not C's FLT_MIN_EXP (-125, one
                    // higher because C counts the mantissa as 0.1xxx): NumPy 2.4.2 np.finfo(np.float32).minexp == -126.
                    minexp = -126;
                    nmant = 23;     // 23 fraction bits
                    iexp = nexp = 8;
                    machep = -23;   // eps = 2^-23
                    negep = -24;    // epsneg = 2^-24
                    break;

                case NPTypeCode.Double:
                case NPTypeCode.Complex:  // NumPy: finfo(complex128) reports float64 values
                    bits = 64;
                    eps = Math.BitIncrement(1.0) - 1.0;  // ~2.22e-16
                    epsneg = 1.0 - Math.BitDecrement(1.0);
                    max = double.MaxValue;
                    min = -double.MaxValue;
                    smallest_normal = 2.2250738585072014e-308;  // 2^-1022
                    smallest_subnormal = double.Epsilon;  // ~5e-324
                    tiny = smallest_normal;
                    precision = 15;
                    resolution = 1e-15;
                    maxexp = 1024;   // 2^1023 * (2-eps) = MaxValue
                    // As for float32: NumPy's minexp is the smallest normal's exponent (2^-1022), not DBL_MIN_EXP (-1021).
                    minexp = -1022;
                    nmant = 52;      // 52 fraction bits
                    iexp = nexp = 11;
                    machep = -52;    // eps = 2^-52
                    negep = -53;     // epsneg = 2^-53
                    break;

                case NPTypeCode.Decimal:
                    // Decimal is a 128-bit type with different semantics
                    // It doesn't have subnormals like IEEE floats
                    bits = 128;
                    eps = 1e-28;  // Approximately 1 / 10^28
                    epsneg = 1e-28;
                    max = (double)decimal.MaxValue;
                    min = (double)decimal.MinValue;
                    smallest_normal = 1e-28;  // Smallest positive decimal
                    smallest_subnormal = 1e-28;  // No subnormals for decimal
                    tiny = smallest_normal;
                    precision = 28;
                    resolution = 1e-28;
                    maxexp = 96;   // Approximate - decimal has different representation
                    minexp = -28;  // Approximate
                    // NumPy has no decimal: these follow decimal's own base-10 layout, as eps / minexp above already do —
                    // a 96-bit integer coefficient, a scale byte (0..28), and eps = epsneg = 10^-28.
                    nmant = 96;
                    iexp = nexp = 8;
                    machep = -28;
                    negep = -28;
                    break;

                default:
                    throw new ArgumentException($"Unsupported float type: {typeCode}");
            }
        }

        /// <summary>
        ///     NumPy's <c>repr(np.finfo(dtype))</c>, the same text under every culture:
        ///     <c>finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)</c>.
        /// </summary>
        /// <returns>The repr text.</returns>
        /// <remarks>
        ///     NumPy's <c>finfo.__repr__</c>, printed by <see cref="Backends.Printing.MachineLimitsRepr.Finfo"/>: the
        ///     resolution through the dtype's own scalar <c>str()</c>, min / max through the dtype's <c>_MACHAR_PARAMS</c>
        ///     printf format (<c>%24.16e</c> / <c>%15.7e</c> / <c>%12.5e</c>), the long double's and decimal's <c>str()</c>.
        ///     A complex dtype reports its real part (<c>dtype=float64</c>). Never .NET's current culture (parity QA-41:
        ///     this text was <c>finfo(resolution=1E-06, min=-3.402823E+038, ...)</c>, with a decimal comma under de-DE and
        ///     U+2212 minus signs under sv-SE).
        /// </remarks>
        public override string ToString() => Backends.Printing.MachineLimitsRepr.Finfo(dtype, resolution, min, max);

        private static bool IsFloatType(NPTypeCode typeCode)
        {
            return typeCode switch
            {
                NPTypeCode.Half => true,
                NPTypeCode.Single => true,
                NPTypeCode.Double => true,
                NPTypeCode.Complex => true, // reports underlying float precision
                NPTypeCode.Decimal => true,  // Partial support - no subnormals
                _ => false
            };
        }
    }

    public partial class np
    {
        /// <summary>
        /// Machine limits for floating point types.
        /// </summary>
        /// <returns>An finfo object describing the floating point type limits.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="ValueError">Not a floating point or complex dtype: NumPy's <c>data type &lt;class
        ///     'numpy.int32'&gt; not inexact</c>.</exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.finfo.html
        /// <para>An ARRAY is not a dtype: NumPy's <c>np.finfo(arr)</c> raises <c>data type &lt;class 'numpy.object_'&gt; not
        /// inexact</c>; NumSharp has no array overload since API-parity batch 7, so pass <c>arr.dtype</c>.</para>
        /// </remarks>
        /// <example>
        /// <code>
        /// var info = np.finfo(np.float64);
        /// Console.WriteLine(info.bits);       // 64
        /// Console.WriteLine(info.eps);        // ~2.22e-16
        /// Console.WriteLine(info.precision);  // 15
        /// </code>
        /// </example>
        /// <param name="dtype">
        ///     The floating point dtype — one descriptor parameter, like NumPy's: a C# <see cref="Type"/> (<c>typeof(double)</c>),
        ///     an <see cref="NPTypeCode"/> or a <see cref="DType"/> (<c>np.float64</c>, <c>arr.dtype</c>) convert implicitly; a
        ///     string binds <see cref="finfo(string)"/>, which reads it as NumPy does.
        /// </param>
        public static finfo finfo(DType dtype)
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));
            return new finfo(dtype);
        }

        /// <summary>
        ///     Machine limits of a floating-point scalar CLASS — NumPy's <c>np.finfo(cls)</c>: a concrete class gives its
        ///     dtype's limits (<c>np.finfo((NDScalarType)np.float32)</c> is <c>np.finfo(np.float32)</c>), a complex class its
        ///     real part's (<c>np.finfo(np.complex64)</c> is float32's, although NumSharp has no complex64 dtype), and a
        ///     Python built-in class the limits of the NumPy class NumPy reads it as (<c>np.finfo(NDScalarType.PyFloat)</c>
        ///     is float64's).
        /// </summary>
        /// <param name="dtype">The scalar class.</param>
        /// <returns>The limits.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="ValueError">
        ///     The class is not inexact — NumPy's text, which names the class: <c>data type &lt;class 'numpy.int64'&gt; not
        ///     inexact</c> (also for <c>int</c>, whose NumPy class is int64). An abstract class is <c>data type &lt;class
        ///     'numpy.object_'&gt; not inexact</c>: NumPy's <c>finfo</c> retries a value that converts to no dtype with the
        ///     dtype of its TYPE, and the type of a class is Python's <c>type</c>, an object dtype — so even
        ///     <c>np.finfo(np.floating)</c> names <c>object_</c>.
        /// </exception>
        public static finfo finfo(NDScalarType dtype)
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));
            if (dtype.IsAbstract)
                throw new ValueError("data type <class 'numpy.object_'> not inexact");

            // A Python built-in class is read as NumPy's dtype(cls).type (float -> float64, int -> int64), as every NumPy
            // dtype argument is; then NumPy's obj2sctype(dtype) must be inexact, and its error names that class.
            var cls = dtype.NumPyClass;
            if (!inexact.__subclasscheck__(cls))
                throw new ValueError($"data type {cls} not inexact");
            // NumPy's _convert_to_float: a complex class has its real part's limits. complex64 has no NumSharp dtype, so
            // its float32 part is named here instead of reached through the descriptor.
            if (cls.IsDtypeless)
                return new finfo(float32);
            return new finfo(cls.ToDescriptor());
        }

        /// <summary>
        ///     Machine limits of the floating-point dtype a STRING names — NumPy's <c>np.finfo("float64")</c>, read as NumPy
        ///     reads it: <c>numeric.dtype(dtype)</c>, and when that raises TypeError (the string names no dtype) the dtype of
        ///     the value's TYPE, <c>str</c> — so <c>np.finfo("foo")</c> and <c>np.finfo("")</c> are NumPy's <c>data type
        ///     &lt;class 'numpy.str_'&gt; not inexact</c>.
        /// </summary>
        /// <param name="dtype">A dtype string (<c>"float32"</c>, <c>"f2"</c>, <c>"double"</c>); null is NumPy's None, which
        ///     NumPy 2.4.2 still reads as float64 (with a DeprecationWarning since 1.25; NumSharp has no warnings).</param>
        /// <returns>An finfo object describing the floating point type limits.</returns>
        /// <exception cref="ValueError">The string names a non-inexact dtype (<c>"int32"</c>: <c>data type &lt;class
        ///     'numpy.int32'&gt; not inexact</c>) or no dtype at all (<c>&lt;class 'numpy.str_'&gt;</c>) — NumPy's
        ///     texts.</exception>
        /// <exception cref="NotSupportedException">The string names a NumPy dtype NumSharp does not model (<c>"O"</c>,
        ///     <c>"V"</c>): <see cref="dtype(string)"/>'s refusal.</exception>
        /// <example>
        /// <code>
        /// var info = np.finfo("float64");
        /// Console.WriteLine(info.bits);  // 64
        /// </code>
        /// </example>
        /// <remarks>An ARRAY is not a dtype: NumPy's <c>np.finfo(arr)</c> reads its type (an object dtype) and raises
        ///     <c>data type &lt;class 'numpy.object_'&gt; not inexact</c>; NumSharp has no array overload since API-parity
        ///     batch 7, so pass <c>arr.dtype</c>. Before that batch this parameter was named <c>dtypeName</c>, and a null or
        ///     empty string raised NumSharp's own ArgumentException.</remarks>
        public static finfo finfo(string dtype)
        {
            // NumPy: `try: dtype = numeric.dtype(dtype) except TypeError: dtype = numeric.dtype(type(dtype))`, after
            // `dtype is None` only warns. A string naming no dtype falls back to str's dtype, which the constructor refuses
            // with NumPy's text; only a TypeError takes the fallback (an unmodelled NumPy dtype's NotSupportedException
            // propagates). `np.dtype` — not the parameter, which shadows the method's simple name here.
            DType resolved;
            if (dtype is null)
                resolved = float64;
            else
            {
                try
                {
                    resolved = np.dtype(dtype);
                }
                catch (TypeError)
                {
                    resolved = str_;
                }
            }
            return new finfo(resolved);
        }
    }
}
