using System;
using System.Globalization;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    ///     Type conversion operators for NDArray.
    ///
    ///     NumPy alignment (see OPERATOR_ALIGNMENT.md Section 7):
    ///     - scalar → NDArray: IMPLICIT (always safe, no data loss)
    ///     - NDArray → scalar: EXPLICIT (may fail if ndim != 0, matches NumPy's int(arr) / float(arr) / bool(arr))
    ///
    ///     Integer targets are CPython's <c>int()</c> of the element, then a checked fit: a float NaN raises
    ///     <see cref="ValueError"/>, an infinity or a value the C# type cannot hold <see cref="OverflowException"/>
    ///     (NumPy's Python int is unbounded; the fixed-width C# type refuses instead of wrapping).
    ///
    ///     Complex source guard:
    ///     - An explicit NDArray → integer or float cast from a Complex-typed array throws <see cref="TypeError"/> with
    ///       CPython's <c>int()</c> / <c>float()</c> text (Python's <c>int(complex)</c> / <c>float(complex)</c> refuse;
    ///       NumSharp has no ComplexWarning, so the silent imaginary drop stays an error). Use <c>np.real</c> explicitly
    ///       before casting. <c>(bool)</c> is the element's truth and <c>(Complex)</c> the element, as in NumPy.
    /// </summary>
    public partial class NDArray
    {
        // ===== scalar → NDArray: IMPLICIT (safe, creates 0-d array) =====
        //
        // The integer primitives, float, double and Complex are NumPy's Python int / float / complex (the C# boundary
        // map), so their 0-d array is marked WasPythonLiteral (NumPy's NPY_ARRAY_WAS_PYTHON_LITERAL): a store packs it
        // with NumPy's range checks (a[0] = 1000 on an int8 array raises) where an ndarray value is cast. bool is not
        // marked (NumPy marks only an exact int / float / complex, and a bool stores like its cast). Half / char are NumPy
        // scalars in the map: their 0-d array is marked WasNumPyScalar (range-checked into a SIGNED integer, cast into the
        // rest). decimal (no NumPy analog) is a plain 0-d array.

        /// <summary>A 0-d bool array (not a Python-literal mark: NumPy stores a bool like its cast).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(bool d) => NDArray.PythonBoolLiteral(d);
        /// <summary>A 0-d int8 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(sbyte d) => NDArray.PythonLiteral<sbyte>(d);
        /// <summary>A 0-d uint8 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(byte d) => NDArray.PythonLiteral<byte>(d);
        /// <summary>A 0-d int16 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(short d) => NDArray.PythonLiteral<short>(d);
        /// <summary>A 0-d uint16 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(ushort d) => NDArray.PythonLiteral<ushort>(d);
        /// <summary>A 0-d int32 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(int d) => NDArray.PythonLiteral<int>(d);
        /// <summary>A 0-d uint32 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(uint d) => NDArray.PythonLiteral<uint>(d);
        /// <summary>A 0-d int64 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(long d) => NDArray.PythonLiteral<long>(d);
        /// <summary>A 0-d uint64 array marked as a Python int literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(ulong d) => NDArray.PythonLiteral<ulong>(d);
        /// <summary>A 0-d char array marked as a NumPy uint16 scalar (see <see cref="WasNumPyScalar"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(char d) => NDArray.NumPyScalarLiteral<char>(d);
        /// <summary>A 0-d float16 array marked as a NumPy float16 scalar (see <see cref="WasNumPyScalar"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(Half d) => NDArray.NumPyScalarLiteral<Half>(d);
        /// <summary>A 0-d float32 array marked as a Python float literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(float d) => NDArray.PythonLiteral<float>(d);
        /// <summary>A 0-d float64 array marked as a Python float literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(double d) => NDArray.PythonLiteral<double>(d);
        /// <summary>A 0-d decimal array (NumSharp-only, a NumPy-scalar-like value, not a Python-literal mark).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(decimal d) => NDArray.Scalar<decimal>(d);
        /// <summary>A 0-d complex128 array marked as a Python complex literal (see <see cref="WasPythonLiteral"/>).</summary>
        /// <param name="d">The value.</param>
        public static implicit operator NDArray(Complex d) => NDArray.PythonLiteral<Complex>(d);

        // ===== NDArray → scalar: EXPLICIT (NumPy's int(arr) / float(arr) / complex(arr), which require 0-d, and
        //       bool(arr), the truth test of any one-element array) =====

        /// <summary>
        ///     Validates the preconditions of every NDArray → number cast (<c>int()</c> / <c>float()</c> /
        ///     <c>complex()</c>; <c>(bool)</c> runs only the element half, <see cref="EnsureElementConvertible"/>):
        ///     <list type="bullet">
        ///       <item><c>ndim == 0</c> (only 0-d arrays can be converted to Python scalars, per NumPy 2.x), checked
        ///             before the element.</item>
        ///       <item>A complex source converts to <see cref="Complex"/> and to bool (its truth), and is refused by every
        ///             integer target with CPython's <c>int()</c> text and by every float target with its <c>float()</c>
        ///             text (NumPy's <c>array_int</c> / <c>array_float</c> hand the complex scalar to them).</item>
        ///       <item>A string source: complex is NumPy's cast refusal; every other target reads the text through
        ///             <see cref="ScalarValue"/> (parsed, or tested for emptiness for bool).</item>
        ///     </list>
        /// </summary>
        /// <param name="nd">The array being converted.</param>
        /// <param name="target">The conversion's target lane: <see cref="NPTypeCode.Boolean"/>, an integer lane
        ///     (<see cref="NPTypeCode.Char"/> included), a float lane (<see cref="NPTypeCode.Decimal"/> included) or
        ///     <see cref="NPTypeCode.Complex"/>.</param>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (NumPy's <c>only 0-dimensional arrays can be
        ///     converted to Python scalars</c>, which <c>int()</c> / <c>float()</c> / <c>complex()</c> raise even for a
        ///     one-element array; it was an <see cref="IncorrectShapeException"/> with NumSharp's own words until parity
        ///     QA-14); a complex source to an integer target (<c>int() argument must be a string, a bytes-like object or a
        ///     real number, not 'complex'</c>) or a float target (<c>float() argument must be a string or a real number, not
        ///     'complex'</c>); or a string source to complex (NumPy's <c>Unable to convert dtype('&lt;U2') to
        ///     complex</c>).</exception>
        private static void EnsureCastableToScalar(NDArray nd, NPTypeCode target)
        {
            if (nd.ndim != 0)
                throw new TypeError("only 0-dimensional arrays can be converted to Python scalars");
            EnsureElementConvertible(nd, target);
        }

        /// <summary>
        ///     The element checks of <see cref="EnsureCastableToScalar"/> without its rank check — what <c>bool(arr)</c>
        ///     needs, since NumPy's truth test (<c>array_nonzero</c>) reads the only element of a one-element array of ANY
        ///     rank.
        /// </summary>
        /// <param name="nd">The array whose (single) element is converted.</param>
        /// <param name="target">The conversion's target lane (see <see cref="EnsureCastableToScalar"/>).</param>
        /// <exception cref="TypeError">A complex source to an integer or float target, or a string source to complex
        ///     (the texts of <see cref="EnsureCastableToScalar"/>).</exception>
        private static void EnsureElementConvertible(NDArray nd, NPTypeCode target)
        {
            if (nd.typecode == NPTypeCode.Complex)
            {
                // bool(arr) is the element's truth and complex(arr) the element itself; only int() / float() refuse a
                // complex value, each with CPython's own text (the old "can't convert complex to int" was NumSharp's).
                if (target is NPTypeCode.Boolean or NPTypeCode.Complex)
                    return;
                throw new TypeError(target is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal
                    ? "float() argument must be a string or a real number, not 'complex'"
                    : "int() argument must be a string, a bytes-like object or a real number, not 'complex'");
            }
            // complex(arr) never reads the text: NumPy's array_complex first asks whether the descriptor casts to
            // complex128 under same_kind, and a string does not (TypeError "Unable to convert dtype('<U2') to complex"),
            // before any parsing.
            if (nd.typecode == NPTypeCode.String && target == NPTypeCode.Complex)
                throw new TypeError($"Unable to convert {nd.dtype.ToString(true)} to complex");
        }

        /// <summary>
        ///     NumPy's <c>int(arr)</c> of a 0-d array: the EXACT Python int its element converts to, before any C# width
        ///     enters — a bool is 0 / 1, an integer itself, a float or decimal truncated toward zero, a text parsed by
        ///     CPython's <c>int()</c>. The integer conversions check this value against their C# type afterwards
        ///     (<see cref="ToCheckedInteger{T}"/>), so a value no C# type can hold overflows loudly instead of wrapping.
        /// </summary>
        /// <param name="nd">A 0-d, non-complex array (checked by <see cref="EnsureCastableToScalar"/>).</param>
        /// <returns>The exact integer value.</returns>
        /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>, CPython's
        ///     <c>float.__int__</c>), or a text that is not an integer literal (CPython's texts).</exception>
        /// <exception cref="OverflowError">A float infinity (<c>cannot convert float infinity to integer</c>), or an
        ///     integer literal past 64 bits (no C# integer holds it).</exception>
        /// <exception cref="InvalidCastException">An element of no numeric kind (unreachable: every storage lane but
        ///     complex, which is refused first, is handled).</exception>
        private static BigInteger PythonIntOf(NDArray nd)
        {
            // A text element parses with CPython's int() into a long / ulong; a numeric element comes back boxed as itself.
            object v = ScalarValue(nd, NPTypeCode.Int64);
            switch (v)
            {
                case bool b: return b ? BigInteger.One : BigInteger.Zero;
                case sbyte i8: return i8;
                case byte u8: return u8;
                case short i16: return i16;
                case ushort u16: return u16;
                case char c: return c;
                case int i32: return i32;
                case uint u32: return u32;
                case long i64: return i64;
                case ulong u64: return u64;
                case Half h: return FloatToPythonInt((double)h);   // float16 widens to double exactly
                case float f: return FloatToPythonInt(f);          // so does float32
                case double d: return FloatToPythonInt(d);
                case decimal m: return new BigInteger(decimal.Truncate(m));
                default: throw new InvalidCastException($"Cannot convert {v?.GetType().Name ?? "null"} to an integer.");
            }
        }

        /// <summary>
        ///     CPython's <c>int(float)</c>: NaN and the infinities refuse, every finite value truncates toward zero into an
        ///     exact integer (a double past 2^53 IS an integer, so nothing is lost).
        /// </summary>
        /// <param name="value">The float element.</param>
        /// <returns>The truncated value, exact.</returns>
        /// <exception cref="ValueError"><paramref name="value"/> is NaN (<c>cannot convert float NaN to integer</c>).</exception>
        /// <exception cref="OverflowError"><paramref name="value"/> is ±inf (<c>cannot convert float infinity to
        ///     integer</c>).</exception>
        private static BigInteger FloatToPythonInt(double value)
        {
            if (double.IsNaN(value))
                throw new ValueError("cannot convert float NaN to integer");
            if (double.IsInfinity(value))
                throw new OverflowError("cannot convert float infinity to integer");
            return new BigInteger(Math.Truncate(value));
        }

        /// <summary>
        ///     An explicit integer conversion of a 0-d array: NumPy's <c>int(arr)</c> (<see cref="PythonIntOf"/>), then a
        ///     CHECKED fit into <typeparamref name="T"/> — the C# type is the narrow end where Python's int is unbounded, so
        ///     a value it cannot hold raises (with NumPy's LP64 texts for packing a Python int into that dtype) where the
        ///     conversion used to wrap silently (<c>(int)np.array(3e9)</c> was <c>int.MinValue</c>, <c>(long)</c> of
        ///     uint64 max was <c>-1</c>: parity QA-42).
        /// </summary>
        /// <typeparam name="T">The C# integer type (or <see cref="char"/>).</typeparam>
        /// <param name="nd">The array being converted.</param>
        /// <param name="target">The lane of <typeparamref name="T"/>.</param>
        /// <returns>The element as <typeparamref name="T"/>, exact.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>).</exception>
        /// <exception cref="TypeError">A complex element (CPython's <c>int()</c> text).</exception>
        /// <exception cref="ValueError">A float NaN, or a text that is not an integer literal.</exception>
        /// <exception cref="OverflowError">A float infinity; a value outside <typeparamref name="T"/>
        ///     (<c>Python integer 3000000000 out of bounds for int32</c>; past the 64-bit C long, <c>Python int too large to
        ///     convert to C long</c>, except into the unsigned 32 / 64-bit types, which NumPy converts through C unsigned
        ///     long).</exception>
        private static T ToCheckedInteger<T>(NDArray nd, NPTypeCode target) where T : unmanaged
        {
            EnsureCastableToScalar(nd, target);
            BigInteger value = PythonIntOf(nd);
            (long min, ulong max) = NDExprTypeRules.IntBounds(target);
            if (value < min || value > max)
            {
                // NumPy packs a Python int through PyLong_AsLong for every integer dtype but uint32 / uint64, so a value
                // past the 64-bit C long reports CPython's text first; anything else is NumPy's own bounds text.
                if (target != NPTypeCode.UInt32 && target != NPTypeCode.UInt64 && (value > long.MaxValue || value < long.MinValue))
                    throw new OverflowError("Python int too large to convert to C long");
                throw new OverflowError(
                    $"Python integer {value.ToString(CultureInfo.InvariantCulture)} out of bounds for {target.AsNumpyDtypeName()}");
            }

            // In range: the cast below is exact (no wrap can occur), so it may go through the general converter.
            object boxed = value.Sign < 0 || value <= long.MaxValue ? (long)value : (object)(ulong)value;
            return Converts.ChangeType<T>(boxed);
        }

        /// <summary>
        ///     The element of a 0-d array as the value an explicit conversion converts: the element itself for a numeric
        ///     array; for a text array what NumPy's <c>int(arr)</c> / <c>float(arr)</c> / <c>bool(arr)</c> make of it —
        ///     <c>array_int</c> / <c>array_float</c> take the element as a Python <c>str</c> / <c>bytes</c>
        ///     (<c>PyArray_GETITEM</c>) and run CPython's <c>int()</c> / <c>float()</c>, so the errors quote <c>'1.5'</c>,
        ///     not <c>np.str_('1.5')</c>; <c>bool(arr)</c> is the element's truth (<c>UNICODE_nonzero</c>: any non-NUL
        ///     character).
        /// </summary>
        /// <param name="nd">A 0-d array (checked by <see cref="EnsureCastableToScalar"/>).</param>
        /// <param name="target">The conversion's target: bool takes the truth; an integer type (char included) the parsed
        ///     Python int as a long (or a ulong past long.MaxValue); a float type (decimal included) the parsed float.</param>
        /// <returns>The value for <see cref="Converts.ChangeType{T}(object)"/>.</returns>
        /// <exception cref="ValueError">The text is not a literal of the target's kind (CPython's texts), or an integer
        ///     literal has more than 4300 digits.</exception>
        /// <exception cref="OverflowError">An integer literal past 64 bits (no C# integer holds it).</exception>
        private static object ScalarValue(NDArray nd, NPTypeCode target)
        {
            object v = nd.Storage.GetAtIndex(0);
            if (nd.typecode != NPTypeCode.String)
                return v;
            if (target == NPTypeCode.Boolean)
                return v is string s ? s.Length > 0 : ((byte[])v).Length > 0;   // trailing NULs are already stripped
            if (target is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal)
            {
                double d;
                unsafe
                {
                    if (v is string text)
                        Backends.Strings.NDTextParse.ParseObject(text, NPTypeCode.Double, (byte*)&d);
                    else
                        Backends.Strings.NDTextParse.ParseObject((byte[])v, NPTypeCode.Double, (byte*)&d);
                }
                return d;
            }
            var big = v is string str ? PythonInt.Parse(str, 10) : PythonInt.ParseBytes((byte[])v, 10);
            if (big >= long.MinValue && big <= long.MaxValue)
                return (long)big;
            if (big.Sign > 0 && big <= ulong.MaxValue)
                return (ulong)big;
            throw new OverflowError("Python int too large to convert to C long");
        }

        /// <summary>
        ///     NumPy's <c>bool(arr)</c> (<c>array_nonzero</c>): the truth of the ONLY element of a one-element array of any
        ///     rank (<c>(bool)np.array(new[,] { { 0.0 } })</c> is False) — nonzero for a number (a NaN is True), either part
        ///     nonzero for a complex (<c>(bool)np.array(1j)</c> is True), non-empty for a text. A larger or an empty array is
        ///     ambiguous and refused with NumPy's two ValueErrors (parity QA-14: every N-D array was refused with the numeric
        ///     conversions' 0-d rule).
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <returns>The element's truth.</returns>
        /// <exception cref="ValueError">More than one element (<c>The truth value of an array with more than one element is
        ///     ambiguous. Use a.any() or a.all()</c>) or none (<c>The truth value of an empty array is ambiguous. Use
        ///     `array.size &gt; 0` to check that an array is not empty.</c>).</exception>
        public static explicit operator bool(NDArray nd)
        {
            // array_nonzero: n == 1 reads the element; anything else is ambiguous (NumPy 2.x made the empty case an
            // error too).
            if (nd.size != 1)
                throw new ValueError(nd.size == 0
                    ? "The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty."
                    : "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
            EnsureElementConvertible(nd, NPTypeCode.Boolean);
            // CDOUBLE_nonzero: real != 0 || imag != 0, so a NaN part counts as nonzero and -0j is False.
            if (nd.typecode == NPTypeCode.Complex)
                return (Complex)nd.Storage.GetAtIndex(0) != Complex.Zero;
            return Converts.ChangeType<bool>(ScalarValue(nd, NPTypeCode.Boolean));
        }

        /// <summary>NumPy's <c>int(arr)</c> as an int8, checked (see <see cref="ToCheckedInteger{T}"/>).</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as an <see cref="sbyte"/>, exact.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>), or a complex element.</exception>
        /// <exception cref="ValueError">A float NaN, or a text that is not an integer literal.</exception>
        /// <exception cref="OverflowError">A float infinity, or a value outside int8.</exception>
        public static explicit operator sbyte(NDArray nd) => ToCheckedInteger<sbyte>(nd, NPTypeCode.SByte);

        /// <summary>NumPy's <c>int(arr)</c> as a uint8, checked (see <see cref="ToCheckedInteger{T}"/>): a negative value
        ///     raises instead of wrapping.</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as a <see cref="byte"/>, exact.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>), or a complex element.</exception>
        /// <exception cref="ValueError">A float NaN, or a text that is not an integer literal.</exception>
        /// <exception cref="OverflowError">A float infinity, or a value outside uint8.</exception>
        public static explicit operator byte(NDArray nd) => ToCheckedInteger<byte>(nd, NPTypeCode.Byte);

        /// <summary>NumPy's <c>int(arr)</c> as an int16, checked (see <see cref="ToCheckedInteger{T}"/>).</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as a <see cref="short"/>, exact.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>), or a complex element.</exception>
        /// <exception cref="ValueError">A float NaN, or a text that is not an integer literal.</exception>
        /// <exception cref="OverflowError">A float infinity, or a value outside int16.</exception>
        public static explicit operator short(NDArray nd) => ToCheckedInteger<short>(nd, NPTypeCode.Int16);

        /// <summary>NumPy's <c>int(arr)</c> as a uint16, checked (see <see cref="ToCheckedInteger{T}"/>).</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as a <see cref="ushort"/>, exact.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>), or a complex element.</exception>
        /// <exception cref="ValueError">A float NaN, or a text that is not an integer literal.</exception>
        /// <exception cref="OverflowError">A float infinity, or a value outside uint16.</exception>
        public static explicit operator ushort(NDArray nd) => ToCheckedInteger<ushort>(nd, NPTypeCode.UInt16);

        /// <summary>NumPy's <c>int(arr)</c> as an int32, checked (see <see cref="ToCheckedInteger{T}"/>):
        ///     <c>(int)np.array(-3.7)</c> is -3, NaN / inf / <c>3e9</c> raise.</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as an <see cref="int"/>, exact.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>), or a complex element.</exception>
        /// <exception cref="ValueError">A float NaN, or a text that is not an integer literal.</exception>
        /// <exception cref="OverflowError">A float infinity, or a value outside int32.</exception>
        public static explicit operator int(NDArray nd) => ToCheckedInteger<int>(nd, NPTypeCode.Int32);

        /// <summary>NumPy's <c>int(arr)</c> as a uint32, checked (see <see cref="ToCheckedInteger{T}"/>).</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as a <see cref="uint"/>, exact.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>), or a complex element.</exception>
        /// <exception cref="ValueError">A float NaN, or a text that is not an integer literal.</exception>
        /// <exception cref="OverflowError">A float infinity, or a value outside uint32.</exception>
        public static explicit operator uint(NDArray nd) => ToCheckedInteger<uint>(nd, NPTypeCode.UInt32);

        /// <summary>NumPy's <c>int(arr)</c> as an int64, checked (see <see cref="ToCheckedInteger{T}"/>): uint64 max and
        ///     <c>1e300</c> raise instead of returning -1 / <c>long.MinValue</c>.</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as a <see cref="long"/>, exact.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>), or a complex element.</exception>
        /// <exception cref="ValueError">A float NaN, or a text that is not an integer literal.</exception>
        /// <exception cref="OverflowError">A float infinity, or a value outside int64.</exception>
        public static explicit operator long(NDArray nd) => ToCheckedInteger<long>(nd, NPTypeCode.Int64);

        /// <summary>NumPy's <c>int(arr)</c> as a uint64, checked (see <see cref="ToCheckedInteger{T}"/>).</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as a <see cref="ulong"/>, exact.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>), or a complex element.</exception>
        /// <exception cref="ValueError">A float NaN, or a text that is not an integer literal.</exception>
        /// <exception cref="OverflowError">A float infinity, or a value outside uint64.</exception>
        public static explicit operator ulong(NDArray nd) => ToCheckedInteger<ulong>(nd, NPTypeCode.UInt64);

        /// <summary>NumPy's <c>int(arr)</c> as a UTF-16 code unit, checked against uint16's range (NumSharp's char lane;
        ///     see <see cref="ToCheckedInteger{T}"/>).</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as a <see cref="char"/>, exact.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>), or a complex element.</exception>
        /// <exception cref="ValueError">A float NaN, or a text that is not an integer literal.</exception>
        /// <exception cref="OverflowError">A float infinity, or a value outside [0, 65535].</exception>
        public static explicit operator char(NDArray nd) => ToCheckedInteger<char>(nd, NPTypeCode.Char);

        /// <summary>NumPy's <c>float(arr)</c> rounded to float32 (an out-of-range value is ±inf, as NumPy's float32 of a
        ///     Python float).</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as a <see cref="float"/>.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>).</exception>
        /// <exception cref="TypeError">A complex element (CPython's <c>float()</c> text).</exception>
        /// <exception cref="ValueError">A text that is not a float literal.</exception>
        public static explicit operator float(NDArray nd)
        {
            EnsureCastableToScalar(nd, NPTypeCode.Single);
            return Converts.ChangeType<float>(ScalarValue(nd, NPTypeCode.Single));
        }

        /// <summary>NumPy's <c>float(arr)</c>.</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as a <see cref="double"/>.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>).</exception>
        /// <exception cref="TypeError">A complex element (CPython's <c>float()</c> text).</exception>
        /// <exception cref="ValueError">A text that is not a float literal.</exception>
        public static explicit operator double(NDArray nd)
        {
            EnsureCastableToScalar(nd, NPTypeCode.Double);
            return Converts.ChangeType<double>(ScalarValue(nd, NPTypeCode.Double));
        }

        /// <summary>The element as a decimal (NumSharp's lane; NumPy has none), through the cast astype runs
        ///     (<see cref="Converts.ChangeType{TOut}(object)"/>). Footgun: a float NaN, an infinity or a magnitude decimal
        ///     cannot hold converts to 0, NumSharp's decimal convention, not an error.</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as a <see cref="decimal"/>.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>).</exception>
        /// <exception cref="TypeError">A complex element (CPython's <c>float()</c> text).</exception>
        /// <exception cref="ValueError">A text that is not a float literal.</exception>
        public static explicit operator decimal(NDArray nd)
        {
            EnsureCastableToScalar(nd, NPTypeCode.Decimal);
            return Converts.ChangeType<decimal>(ScalarValue(nd, NPTypeCode.Decimal));
        }

        /// <summary>NumPy's <c>float(arr)</c> rounded to float16 (an out-of-range value is ±inf).</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as a <see cref="Half"/>.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>).</exception>
        /// <exception cref="TypeError">A complex element (CPython's <c>float()</c> text).</exception>
        /// <exception cref="ValueError">A text that is not a float literal.</exception>
        public static explicit operator Half(NDArray nd)
        {
            EnsureCastableToScalar(nd, NPTypeCode.Half);
            return Converts.ChangeType<Half>(ScalarValue(nd, NPTypeCode.Half));
        }

        /// <summary>NumPy's <c>complex(arr)</c>: any numeric element (a real one gets a zero imaginary part).</summary>
        /// <param name="nd">A 0-d array.</param>
        /// <returns>The element as a <see cref="Complex"/>.</returns>
        /// <exception cref="TypeError"><paramref name="nd"/> is not 0-d (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>).</exception>
        /// <exception cref="TypeError">A text element (<c>Unable to convert dtype('&lt;U2') to complex</c>).</exception>
        public static explicit operator Complex(NDArray nd)
        {
            // Complex target: no source-type restriction. ndim==0 still required.
            EnsureCastableToScalar(nd, NPTypeCode.Complex);
            return Converts.ChangeType<Complex>(nd.Storage.GetAtIndex(0));
        }

        public static explicit operator string(NDArray d) => d.ToString(false);
    }
}
