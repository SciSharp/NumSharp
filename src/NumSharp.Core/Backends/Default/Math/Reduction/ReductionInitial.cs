using System;
using System.Collections;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    /// <summary>
    ///     NumPy's conversion of a reduction's <c>initial=</c> argument into the loop dtype — <c>PyArray_Pack(descr, buf,
    ///     initial)</c>, called by <c>PyUFunc_ReduceWrapper</c> (<c>numpy/_core/src/umath/reduction.c</c>) after the
    ///     reduction iterator is built and before the result is filled with the value.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The same conversion is NumPy's element setter everywhere a value is STORED, so the store paths share it:
    ///     <c>np.copyto</c> with a Python scalar, <c>ndarray.fill</c>, <c>np.full</c> / <c>np.full_like</c>,
    ///     <c>np.insert</c>, <c>np.fill_diagonal</c>, the flat iterator's setters, and — through
    ///     <see cref="NDArray.PackForStore"/> — the indexer setters, <c>np.put</c>, <c>np.putmask</c>, <c>np.place</c>,
    ///     <c>np.put_along_axis</c> and <c>np.asarray</c> with a dtype, for a C# literal (an
    ///     <see cref="NDArray.WasPythonLiteral"/> array is packed as the literal it was made from).
    ///     </para>
    ///     <para>
    ///     <c>PyArray_Pack</c> is the array's element setter (<c>arr[()] = initial</c>), so the value follows NumPy's
    ///     setitem rules, which differ by the KIND of value. The C# boundary map (the house rule of
    ///     <c>np.clip</c> and <c>numpy.polynomial</c>): <c>bool</c>, the integer primitives and <see cref="BigInteger"/>
    ///     are Python ints, <c>float</c> / <c>double</c> are Python floats, <see cref="Complex"/> is a Python complex,
    ///     <see cref="Half"/> / <c>char</c> / <c>decimal</c> are NumPy scalars (a float16, an unsigned integer, and the
    ///     NumSharp-only decimal), a <c>string</c> is a Python str, an <see cref="NDArray"/> is an ndarray, and any other
    ///     array or collection is a Python list (a <c>System.Runtime.CompilerServices.ITuple</c> a tuple).
    ///     </para>
    ///     <para>
    ///     The consequences, probed against NumPy 2.4.2:
    ///     <list type="bullet">
    ///     <item>A Python int into an integer loop is range-checked, never wrapped: <c>initial=1000</c> into int8 raises
    ///     "Python integer 1000 out of bounds for int8". The text follows the LP64 C <c>long</c> (Linux / macOS), the
    ///     convention NumSharp pins everywhere (<see cref="NDExprTypeRules.CheckPyIntPacksIntoLoop"/>): past int64 every
    ///     integer loop says "Python int too large to convert to C long"; uint32 and uint64 convert through C
    ///     <c>unsigned long</c>, so their range extends to 2**64 - 1 before that text. Windows NumPy (32-bit C long)
    ///     already says "too large" past ±2**31 for the 8/16/32-bit loops.</item>
    ///     <item>A Python float into an integer loop is truncated toward zero first (<c>7.5</c> → 7), then range-checked
    ///     like the Python int it became; NaN raises <c>ValueError</c> "cannot convert float NaN to integer", an infinity
    ///     <c>OverflowError</c> "cannot convert float infinity to integer".</item>
    ///     <item>A Python int or float into a float loop rounds through a double (an int correctly rounded, unlike
    ///     .NET's truncating <c>(double)BigInteger</c>; past the double range "int too large to convert to float"), then to
    ///     float32 / float16 with overflow to infinity, never an error.</item>
    ///     <item>A Python complex into a real loop is a <c>TypeError</c>: float() / int() of a complex.</item>
    ///     <item>A 0-d ndarray is CAST (unsafe, wrapping: a 0-d int64 1000 into int8 is -24), unlike the Python int with
    ///     the same value; an ndarray with dimensions is "setting an array element with a sequence." (complex loops:
    ///     "only 0-dimensional arrays can be converted to Python scalars").</item>
    ///     <item>A NumPy scalar (<see cref="Half"/>, <c>char</c>) depends on the loop's DType class: a SIGNED integer loop
    ///     knows every NumPy scalar and int()-converts it with the range check (<c>np.float16(300)</c> into int8 raises);
    ///     every other loop CASTS it (<c>np.float16(300)</c> or <c>np.uint16(300)</c> into uint8 is 44).</item>
    ///     <item>A bool loop takes Python truthiness of anything (a NaN is true, a non-empty string or list is true).</item>
    ///     <item>A str parses with Python's <c>float()</c> / <c>int()</c> / <c>complex()</c> grammar.</item>
    ///     </list>
    ///     </para>
    ///     <para>
    ///     Exception types follow the house map: NumPy's <c>OverflowError</c> is <see cref="OverflowException"/>, its
    ///     <c>ValueError</c> is <see cref="ValueError"/>, its <c>TypeError</c> is <see cref="TypeError"/>.
    ///     </para>
    /// </remarks>
    internal static class ReductionInitial
    {
        /// <summary>The bytes a packed value can take: the widest NumSharp element (decimal, complex128).</summary>
        internal const int SlotBytes = 16;

        /// <summary>
        ///     Packs <paramref name="initial"/> into one element of <paramref name="loop"/>, written at the start of
        ///     <paramref name="slot"/> — NumPy's <c>PyArray_Pack</c> for a reduction's <c>initial=</c>.
        /// </summary>
        /// <param name="initial">The caller's value; <c>null</c> is NumPy's <c>None</c>, which means "no initial value"
        ///     (<c>initial=None</c> is <c>PyUFunc_ReduceWrapper</c>'s explicit unset).</param>
        /// <param name="loop">The reduction's loop dtype (the result dtype NumPy's <c>op_dtypes[0]</c> names).</param>
        /// <param name="slot">Receives the packed element; at least <see cref="SlotBytes"/> bytes.</param>
        /// <returns><c>false</c> when <paramref name="initial"/> is <c>null</c> (nothing packed), else <c>true</c>.</returns>
        /// <exception cref="ArgumentException"><paramref name="slot"/> is shorter than <see cref="SlotBytes"/>.</exception>
        /// <exception cref="OverflowException">An integer that does not fit an integer <paramref name="loop"/>, a float
        ///     infinity into an integer loop, or a Python int past the double range into a float loop (NumPy's
        ///     <c>OverflowError</c> texts).</exception>
        /// <exception cref="ValueError">A float NaN into an integer loop, an unparsable string, an ndarray with
        ///     dimensions or a sequence (NumPy's <c>ValueError</c> texts).</exception>
        /// <exception cref="TypeError">A complex into a real loop, a sequence into an integer or complex loop, or a value
        ///     of a type NumPy cannot convert (NumPy's <c>TypeError</c> texts).</exception>
        internal static bool TryPack(object initial, NPTypeCode loop, Span<byte> slot)
        {
            if (initial is null)
                return false;
            if (slot.Length < SlotBytes)
                throw new ArgumentException($"The initial slot needs {SlotBytes} bytes.", nameof(slot));

            // A C# literal's 0-d array (NDArray.WasPythonLiteral, NumPy's NPY_ARRAY_WAS_PYTHON_LITERAL temporary) stands
            // for the Python scalar it was made from: pack the literal's own value, not the array's cast. A Half / char
            // value's array (NDArray.WasNumPyScalar) unwraps to the NumPy scalar, whose case below applies NumPy's rule.
            if (initial is NDArray literal && (literal.WasPythonLiteral || literal.WasPythonStr || literal.WasNumPyScalar))
                initial = literal.Storage.GetAtIndex(0);

            slot.Clear();
            // A bool loop takes Python truthiness (BOOL_setitem's PyObject_IsTrue), whatever the value's kind, except a
            // 0-d ndarray, which is CAST like every other array (a cast to bool is the same "!= 0" for a number).
            if (loop == NPTypeCode.Boolean && initial is not NDArray)
            {
                slot[0] = (byte)(IsTrue(initial) ? 1 : 0);
                return true;
            }

            switch (initial)
            {
                case NDArray nd:
                    PackArray(nd, loop, slot);
                    return true;
                case string s:
                    PackText(s, null, loop, slot);
                    return true;
                case bool b:
                    // A Python bool is an int subclass: True packs as 1 everywhere a number goes.
                    PackPyInt(b ? BigInteger.One : BigInteger.Zero, loop, slot);
                    return true;
                case sbyte or byte or short or ushort or int or uint or long:
                    PackPyInt(Convert.ToInt64(initial, CultureInfo.InvariantCulture), loop, slot);
                    return true;
                case ulong u:
                    PackPyInt(u, loop, slot);
                    return true;
                case BigInteger bi:
                    PackPyInt(bi, loop, slot);
                    return true;
                case char c:
                    // NumSharp's char is NumPy's uint16 SCALAR, and a NumPy scalar packs by the loop's DType class: a SIGNED
                    // integer DType knows every NumPy scalar (signed_integers_is_known_scalar_types) and runs its setitem,
                    // which range-checks the int the scalar holds (np.uint16(300) into int8 raises); every other DType casts
                    // it (npy_cast_raw_scalar_item, unsafe: np.uint16(300) into uint8 is 44).
                    if (IsSignedIntegerLoop(loop))
                        PackPyInt(c, loop, slot);
                    else
                        CastScalar(c, NPTypeCode.Char, loop, slot);
                    return true;
                case double d:
                    PackPyFloat(d, loop, slot);
                    return true;
                case float f:
                    // A float32 value into a float32 loop keeps its bits (NumPy's FLOAT_setitem copies a float32 scalar
                    // as is); a widen-narrow round trip would quiet a signalling NaN.
                    if (loop == NPTypeCode.Single)
                    {
                        Unsafe.WriteUnaligned(ref slot[0], f);
                        return true;
                    }

                    PackPyFloat(f, loop, slot);
                    return true;
                case Half h:
                    // np.float16 scalar, packed by the loop's DType class like char above: a signed integer loop runs
                    // int(h) through its setitem (truncated, range-checked; NaN / inf are Python's int() errors); every other
                    // loop CASTS it — exactly into a float or complex loop (a NaN keeps its payload), NumPy's unsafe
                    // float→int conversion into an unsigned loop (np.float16(300) into uint8 is 44, no error).
                    if (IsSignedIntegerLoop(loop))
                        PackPyFloat((double)h, loop, slot);
                    else
                        CastScalar(h, NPTypeCode.Half, loop, slot);
                    return true;
                case decimal m:
                    PackDecimal(m, loop, slot);
                    return true;
                case Complex z:
                    PackComplex(z, loop, slot);
                    return true;
                case byte[] bytes:
                    // A Python bytes (checked before IEnumerable: a byte[] is one value, not a list of ints).
                    PackText(null, bytes, loop, slot);
                    return true;
                case ITuple:
                    PackSequence("tuple", loop);
                    return true;
                case IEnumerable:
                    PackSequence("list", loop);
                    return true;
                default:
                    throw new TypeError(IsIntegerLoop(loop)
                        ? $"int() argument must be a string, a bytes-like object or a real number, not '{initial.GetType().Name}'"
                        : $"float() argument must be a string or a real number, not '{initial.GetType().Name}'");
            }
        }

        /// <summary>
        ///     Reads the packed element of <paramref name="slot"/> as <typeparamref name="T"/>.
        /// </summary>
        /// <typeparam name="T">The loop dtype's element type.</typeparam>
        /// <param name="slot">A slot filled by <see cref="TryPack"/>.</param>
        /// <returns>The packed value.</returns>
        internal static T Read<T>(ReadOnlySpan<byte> slot) where T : unmanaged
            => Unsafe.ReadUnaligned<T>(ref Unsafe.AsRef(in slot[0]));

        /// <summary>
        ///     Reads the packed element of <paramref name="slot"/> boxed as <paramref name="loop"/>'s exact CLR type — the
        ///     form the store paths that take a boxed fill value (<c>ndarray.fill</c>, <c>np.full</c>) consume.
        /// </summary>
        /// <param name="slot">A slot filled by <see cref="TryPack"/> (or another packer) for <paramref name="loop"/>.</param>
        /// <param name="loop">The dtype the slot holds.</param>
        /// <returns>The boxed element (<c>sbyte</c> for int8, <see cref="Half"/> for float16, …).</returns>
        /// <exception cref="NotSupportedException"><paramref name="loop"/> is not a numeric storage dtype (string, empty).</exception>
        /// <remarks>A representation switch like <c>UnmanagedStorage.GetAtIndex</c>'s, not a compute path: every dtype reads
        ///     the same bytes.</remarks>
        internal static object Box(ReadOnlySpan<byte> slot, NPTypeCode loop)
        {
            switch (loop)
            {
                case NPTypeCode.Boolean: return Read<byte>(slot) != 0;
                case NPTypeCode.SByte: return Read<sbyte>(slot);
                case NPTypeCode.Byte: return Read<byte>(slot);
                case NPTypeCode.Int16: return Read<short>(slot);
                case NPTypeCode.UInt16: return Read<ushort>(slot);
                case NPTypeCode.Int32: return Read<int>(slot);
                case NPTypeCode.UInt32: return Read<uint>(slot);
                case NPTypeCode.Int64: return Read<long>(slot);
                case NPTypeCode.UInt64: return Read<ulong>(slot);
                case NPTypeCode.Char: return Read<char>(slot);
                case NPTypeCode.Half: return Read<Half>(slot);
                case NPTypeCode.Single: return Read<float>(slot);
                case NPTypeCode.Double: return Read<double>(slot);
                case NPTypeCode.Decimal: return Read<decimal>(slot);
                case NPTypeCode.Complex: return Read<Complex>(slot);
                default: throw new NotSupportedException($"ReductionInitial.Box: {loop} is not a numeric dtype.");
            }
        }

        /// <summary>
        ///     Packs a C# number standing for a NumPy SCALAR of its own dtype (an <c>np.int64</c>, <c>np.float64</c>, …)
        ///     into one element of <paramref name="loop"/> — <c>PyArray_Pack</c>'s rule for a NumPy scalar, which differs
        ///     from a Python scalar's: only a SIGNED integer DType knows every NumPy scalar as its own
        ///     (<c>signed_integers_is_known_scalar_types</c>) and int()-converts it with the range check; every other
        ///     DType CASTS it (<c>npy_cast_raw_scalar_item</c>, unsafe).
        /// </summary>
        /// <param name="value">A boxed number: an integer primitive, <c>bool</c>, <c>float</c>, <c>double</c>,
        ///     <see cref="Half"/>, <c>char</c>, <c>decimal</c> or <see cref="Complex"/>; anything else (an array, a string)
        ///     goes through <see cref="TryPack"/>.</param>
        /// <param name="loop">The destination dtype.</param>
        /// <param name="slot">Receives the element; at least <see cref="SlotBytes"/> bytes.</param>
        /// <exception cref="OverflowException">A signed integer loop and a value whose int() does not fit it, or an
        ///     infinity ("cannot convert float infinity to integer").</exception>
        /// <exception cref="ValueError">A signed integer loop and a NaN ("cannot convert float NaN to integer").</exception>
        /// <exception cref="ArgumentException"><paramref name="slot"/> is shorter than <see cref="SlotBytes"/>.</exception>
        /// <exception cref="TypeError">A value that is not a number, refused by <see cref="TryPack"/>'s rules.</exception>
        /// <remarks>
        ///     <para>
        ///     This is how <c>np.pad</c> stores its constants: <c>_as_pairs</c> runs <c>np.array(constant_values)</c> and
        ///     indexes it, so each constant reaches <c>padded[...] = value</c> as a NumPy scalar. Probed against NumPy
        ///     2.4.2: <c>np.pad(int8, 1, constant_values=1000)</c> raises "Python integer 1000 out of bounds for int8"
        ///     (int()), while <c>np.pad(uint8, 1, constant_values=1000)</c> pads with 232 (the cast); a complex constant
        ///     into a signed integer keeps its real part (int() of a NumPy complex drops the imaginary part with a
        ///     ComplexWarning), into an unsigned one it is cast.
        ///     </para>
        ///     <para>
        ///     The C# boundary map gives each value its own C# type as the NumPy scalar's dtype (a C# <c>int</c> is an
        ///     <c>np.int32</c>, NumSharp's house rule for a discovered C# int); the int() and the cast agree with NumPy's
        ///     int64 for every value an <c>int</c> holds.
        ///     </para>
        /// </remarks>
        internal static void PackNumPyScalar(object value, NPTypeCode loop, Span<byte> slot)
        {
            if (slot.Length < SlotBytes)
                throw new ArgumentException($"The scalar slot needs {SlotBytes} bytes.", nameof(slot));

            NPTypeCode own = value switch
            {
                bool => NPTypeCode.Boolean, sbyte => NPTypeCode.SByte, byte => NPTypeCode.Byte, short => NPTypeCode.Int16,
                ushort => NPTypeCode.UInt16, int => NPTypeCode.Int32, uint => NPTypeCode.UInt32, long => NPTypeCode.Int64,
                ulong => NPTypeCode.UInt64, char => NPTypeCode.Char, Half => NPTypeCode.Half, float => NPTypeCode.Single,
                double => NPTypeCode.Double, decimal => NPTypeCode.Decimal, Complex => NPTypeCode.Complex,
                _ => NPTypeCode.Empty,
            };
            if (own == NPTypeCode.Empty)
            {
                // Not a number: a Python object of another kind (an array, a list, a string) packs by its own rules.
                TryPack(value, loop, slot);
                return;
            }

            slot.Clear();
            if (IsSignedIntegerLoop(loop))
            {
                // The signed integer setitem: PyNumber_Long(scalar) — truncation for a float (NaN / inf are int()'s
                // errors), the real part for a complex (NumPy's complex __int__) — then @TYPE@_safe_pyint_setitem's range
                // check with the LP64 texts.
                switch (value)
                {
                    case Complex z:
                        PackPyFloat(z.Real, loop, slot);
                        return;
                    case Half h:
                        PackPyFloat((double)h, loop, slot);
                        return;
                    case float f:
                        PackPyFloat(f, loop, slot);
                        return;
                    case double d:
                        PackPyFloat(d, loop, slot);
                        return;
                    case decimal m:
                        PackDecimal(m, loop, slot);
                        return;
                    case bool b:
                        PackPyInt(b ? BigInteger.One : BigInteger.Zero, loop, slot);
                        return;
                    case ulong u:
                        PackPyInt(u, loop, slot);
                        return;
                    case char c:
                        PackPyInt(c, loop, slot);
                        return;
                    default:
                        PackPyInt(Convert.ToInt64(value, CultureInfo.InvariantCulture), loop, slot);
                        return;
                }
            }

            // Every other DType casts the NumPy scalar from its own dtype (the house converters carry NumPy's MSVC-pinned
            // unsafe conversions: 1000 into uint8 is 232, a NaN into an unsigned integer the cast loop's value).
            CastScalarValue(value, loop, slot);
        }

        /// <summary>
        ///     Casts a boxed C# number from its own dtype into one element of <paramref name="loop"/> — NumPy's unsafe cast
        ///     of a NumPy scalar (<c>npy_cast_raw_scalar_item</c>), which <c>PyArray_FromAny</c> applies to a NumPy scalar
        ///     like a 0-d array and <c>PyArray_Pack</c> applies wherever the DType does not know the scalar.
        /// </summary>
        /// <param name="value">A boxed number: an integer primitive, <c>bool</c>, <c>float</c>, <c>double</c>,
        ///     <see cref="Half"/>, <c>char</c>, <c>decimal</c> or <see cref="Complex"/>.</param>
        /// <param name="loop">The destination dtype.</param>
        /// <param name="slot">Receives the element; at least <see cref="SlotBytes"/> bytes.</param>
        /// <exception cref="ArgumentException"><paramref name="value"/> is not one of those numbers, or
        ///     <paramref name="slot"/> is shorter than <see cref="SlotBytes"/>.</exception>
        /// <remarks>The house converters carry NumPy's (MSVC-pinned) unsafe conversions: no error, 300 into uint8 is 44, a
        ///     complex into a real dtype keeps its real part.</remarks>
        internal static void CastScalarValue(object value, NPTypeCode loop, Span<byte> slot)
        {
            if (slot.Length < SlotBytes)
                throw new ArgumentException($"The scalar slot needs {SlotBytes} bytes.", nameof(slot));
            slot.Clear();
            NPTypeCode own = value?.GetType().GetTypeCode() ?? NPTypeCode.Empty;
            switch (value)
            {
                case bool b: CastScalar(b, own, loop, slot); return;
                case sbyte v: CastScalar(v, own, loop, slot); return;
                case byte v: CastScalar(v, own, loop, slot); return;
                case short v: CastScalar(v, own, loop, slot); return;
                case ushort v: CastScalar(v, own, loop, slot); return;
                case int v: CastScalar(v, own, loop, slot); return;
                case uint v: CastScalar(v, own, loop, slot); return;
                case long v: CastScalar(v, own, loop, slot); return;
                case ulong v: CastScalar(v, own, loop, slot); return;
                case char v: CastScalar(v, own, loop, slot); return;
                case Half v: CastScalar(v, own, loop, slot); return;
                case float v: CastScalar(v, own, loop, slot); return;
                case double v: CastScalar(v, own, loop, slot); return;
                case decimal v: CastScalar(v, own, loop, slot); return;
                case Complex v: CastScalar(v, own, loop, slot); return;
                default:
                    throw new ArgumentException($"CastScalarValue: a {value?.GetType().Name ?? "null"} is not a number.", nameof(value));
            }
        }

        /// <summary>
        ///     Python truthiness (<c>PyObject_IsTrue</c>) of a non-array initial value — the whole of NumPy's bool setitem.
        /// </summary>
        /// <param name="v">The value (not <c>null</c>, not an <see cref="NDArray"/>).</param>
        /// <returns>Whether Python would call it true: a nonzero number (a NaN included), a non-empty string or sequence,
        ///     and any other object.</returns>
        private static bool IsTrue(object v)
        {
            switch (v)
            {
                case bool b: return b;
                case double d: return d != 0 || double.IsNaN(d);
                case float f: return f != 0 || float.IsNaN(f);
                case Half h: return h != Half.Zero || Half.IsNaN(h);
                case decimal m: return m != 0m;
                case Complex z: return z.Real != 0 || z.Imaginary != 0 || double.IsNaN(z.Real) || double.IsNaN(z.Imaginary);
                case BigInteger bi: return !bi.IsZero;
                case char c: return c != 0;
                case sbyte or byte or short or ushort or int or uint or long:
                    return Convert.ToInt64(v, CultureInfo.InvariantCulture) != 0;
                case ulong u: return u != 0;
                case string s: return s.Length != 0;
                case ICollection col: return col.Count != 0;
                case IEnumerable seq:
                {
                    // A Python list is true when it has an element; enumerate only as far as the first one.
                    var e = seq.GetEnumerator();
                    try { return e.MoveNext(); }
                    finally { (e as IDisposable)?.Dispose(); }
                }
                default: return true;
            }
        }

        /// <summary>
        ///     Packs an ndarray: a 0-d one is cast into the loop dtype with NumPy's unsafe cast (wrapping, NaN to an integer
        ///     loop by the house float→int rule); one with dimensions is refused the way NumPy's setitem refuses it, except
        ///     that a bool loop takes a one-element array's truth.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="slot">Receives the element.</param>
        /// <exception cref="ValueError"><paramref name="nd"/> has dimensions (and, for a bool loop, a size other than 1) and
        ///     the loop is real or bool: "setting an array element with a sequence."</exception>
        /// <exception cref="TypeError"><paramref name="nd"/> has dimensions and the loop is complex.</exception>
        private static unsafe void PackArray(NDArray nd, NPTypeCode loop, Span<byte> slot)
        {
            // BOOL_setitem's PyObject_IsTrue takes a one-element array's truth, its element's. With any other size the
            // truth-value error is raised, and setitem replaces it (the value is a sequence) with the sequence ValueError,
            // as for every other loop.
            if (nd.ndim != 0 && !(loop == NPTypeCode.Boolean && nd.size == 1))
            {
                if (loop == NPTypeCode.Complex)
                    throw new TypeError("only 0-dimensional arrays can be converted to Python scalars");
                throw new ValueError("setting an array element with a sequence.");
            }

            // The house converters carry NumPy's (MSVC-pinned) unsafe conversions. The one element sits at the shape's
            // offset (every index 0): a contiguous slice re-seats Address with offset 0, a strided view keeps its base and a
            // non-zero offset — Address + offset·itemsize is right for both, and no intermediate array is made.
            byte* p = (byte*)nd.Address + nd.Shape.offset * nd.dtypesize;
            fixed (byte* d = slot)
                NDIterCasting.ConvertValue(p, d, nd.GetTypeCode, loop);
        }

        /// <summary>
        ///     Packs a Python int — NumPy's setitem of a <c>PyLong</c>: an integer loop range-checks it (LP64 texts), a
        ///     float loop rounds it through a double, a complex loop takes it as the real part, decimal keeps it exactly.
        /// </summary>
        /// <param name="v">The integer.</param>
        /// <param name="loop">The loop dtype (not bool — the caller handles truthiness).</param>
        /// <param name="slot">Receives the element.</param>
        /// <exception cref="OverflowException">The value does not fit an integer loop, or exceeds the double range for a
        ///     float or complex loop, or the decimal range.</exception>
        private static void PackPyInt(BigInteger v, NPTypeCode loop, Span<byte> slot)
        {
            if (IsIntegerLoop(loop))
            {
                CheckPyIntFits(v, loop);
                WriteInteger(v, loop, slot);
                return;
            }

            if (loop == NPTypeCode.Decimal)
            {
                // NumSharp-only: an exact decimal, or .NET's overflow past ±7.9e28.
                if (v > (BigInteger)decimal.MaxValue || v < (BigInteger)decimal.MinValue)
                    throw new OverflowException("Python int too large to convert to decimal");
                Unsafe.WriteUnaligned(ref slot[0], (decimal)v);
                return;
            }

            // PyLong_AsDouble: correctly rounded, OverflowError past the double range (Python raises; .NET would give inf).
            double d = PythonInt.ToDouble(v);
            if (double.IsInfinity(d))
                throw new OverflowException("int too large to convert to float");
            WriteFloat(d, loop, slot);
        }

        /// <summary>
        ///     The LP64 range check NumPy's integer setitem makes (<c>@TYPE@_safe_pyint_setitem</c> over
        ///     <c>MyPyLong_AsLong</c> / <c>MyPyLong_AsUnsignedLong</c>), for any Python int.
        /// </summary>
        /// <param name="v">The integer.</param>
        /// <param name="loop">An integer loop dtype.</param>
        /// <exception cref="OverflowException">"Python int too large to convert to C long" when the value cannot even be
        ///     read as the C long the loop converts through (below int64, or above int64 for every loop but uint32 / uint64,
        ///     whose C unsigned long reaches 2**64 - 1); else "Python integer {v} out of bounds for {dtype}".</exception>
        private static void CheckPyIntFits(BigInteger v, NPTypeCode loop)
        {
            (long min, ulong max) = NDExprTypeRules.IntBounds(loop);
            if (v >= min && v <= max)
                return;

            bool unsignedLong = loop is NPTypeCode.UInt32 or NPTypeCode.UInt64;
            if (v < long.MinValue || (unsignedLong ? v > ulong.MaxValue : v > long.MaxValue))
                throw new OverflowException("Python int too large to convert to C long");
            throw new OverflowException($"Python integer {v} out of bounds for {loop.AsNumpyDtypeName()}");
        }

        /// <summary>
        ///     Packs a Python float — NumPy's setitem of a <c>PyFloat</c>: truncated toward zero into an integer loop, rounded
        ///     into a narrower float, the real part of a complex.
        /// </summary>
        /// <param name="d">The value.</param>
        /// <param name="loop">The loop dtype (not bool).</param>
        /// <param name="slot">Receives the element.</param>
        /// <exception cref="ValueError"><paramref name="d"/> is NaN and the loop is an integer.</exception>
        /// <exception cref="OverflowException"><paramref name="d"/> is infinite and the loop is an integer, or the
        ///     truncated value does not fit it, or (decimal) out of range.</exception>
        private static void PackPyFloat(double d, NPTypeCode loop, Span<byte> slot)
        {
            if (IsIntegerLoop(loop))
            {
                // int(float): NaN / infinity are Python's own errors; the truncated integer then meets the int rules.
                PackPyInt(PythonInt.From(d), loop, slot);
                return;
            }

            if (loop == NPTypeCode.Decimal)
            {
                Unsafe.WriteUnaligned(ref slot[0], (decimal)d);
                return;
            }

            WriteFloat(d, loop, slot);
        }

        /// <summary>
        ///     Packs a decimal (NumSharp-only, no NumPy analog): exact into a decimal loop, truncated like a float into an
        ///     integer loop (exactly, through <see cref="BigInteger"/>), rounded through a double into a float loop.
        /// </summary>
        /// <param name="m">The value.</param>
        /// <param name="loop">The loop dtype (not bool).</param>
        /// <param name="slot">Receives the element.</param>
        /// <exception cref="OverflowException">The truncated value does not fit an integer loop.</exception>
        private static void PackDecimal(decimal m, NPTypeCode loop, Span<byte> slot)
        {
            if (loop == NPTypeCode.Decimal)
            {
                Unsafe.WriteUnaligned(ref slot[0], m);
                return;
            }

            if (IsIntegerLoop(loop))
            {
                PackPyInt(new BigInteger(decimal.Truncate(m)), loop, slot);
                return;
            }

            WriteFloat((double)m, loop, slot);
        }

        /// <summary>
        ///     Packs a Python complex: only a complex loop takes it; real loops raise the TypeError Python's float() / int()
        ///     raise.
        /// </summary>
        /// <param name="z">The value.</param>
        /// <param name="loop">The loop dtype (not bool).</param>
        /// <param name="slot">Receives the element.</param>
        /// <exception cref="TypeError">The loop is real.</exception>
        private static void PackComplex(Complex z, NPTypeCode loop, Span<byte> slot)
        {
            if (loop == NPTypeCode.Complex)
            {
                Unsafe.WriteUnaligned(ref slot[0], z);
                return;
            }

            throw new TypeError(IsIntegerLoop(loop)
                ? "int() argument must be a string, a bytes-like object or a real number, not 'complex'"
                : "float() argument must be a string or a real number, not 'complex'");
        }

        /// <summary>
        ///     Packs a Python <c>str</c> or <c>bytes</c>: <c>int(s)</c> into an integer loop (range-checked like any Python
        ///     int), <c>float(s)</c> into a float loop (rounded once more into float32 / float16), <c>complex(s)</c> into a
        ///     complex loop (bytes decoded as UTF-8 first) — CPython 3.12's exact grammars and texts
        ///     (<see cref="Strings.NDTextParse"/>), quoting the object as <c>'…'</c> / <c>b'…'</c>. NumSharp's decimal parses as
        ///     a float64 and converts like the float64 → decimal cast (the rule its text conversions share).
        /// </summary>
        /// <param name="text">The str, or null when <paramref name="bytes"/> is given.</param>
        /// <param name="bytes">The bytes, or null when <paramref name="text"/> is given.</param>
        /// <param name="loop">The loop dtype (not bool).</param>
        /// <param name="slot">Receives the element.</param>
        /// <exception cref="ValueError">The text is not a valid literal (Python's texts), or an integer literal has more
        ///     than 4300 digits.</exception>
        /// <exception cref="OverflowException">The parsed integer does not fit an integer loop (NumPy's LP64 texts).</exception>
        /// <exception cref="UnicodeDecodeError">Bytes into a complex loop that are not UTF-8.</exception>
        private static unsafe void PackText(string text, byte[] bytes, NPTypeCode loop, Span<byte> slot)
        {
            byte* canon = stackalloc byte[16];
            if (text is not null)
                Strings.NDTextParse.ParseObject(text, loop, canon);
            else
                Strings.NDTextParse.ParseObject(bytes, loop, canon);
            // The canonical value (int64 / uint64 range-checked for the loop, a double, a complex) cast into the loop: exact
            // for an integer, one more rounding for float32 / float16 (NumPy's FLOAT_setitem / npy_double_to_half).
            var canonType = Strings.NDTextParse.CanonicalType(Strings.NDTextParse.FamilyOf(loop));
            fixed (byte* d = slot)
                NDIterCasting.ConvertValue(canon, d, canonType, loop);
        }

        /// <summary>
        ///     Refuses a Python list or tuple the way NumPy's setitem refuses it for the loop's kind.
        /// </summary>
        /// <param name="typeName">The Python type name to report (<c>list</c> or <c>tuple</c>).</param>
        /// <param name="loop">The loop dtype (not bool).</param>
        /// <exception cref="TypeError">An integer loop (int() of a sequence) or a complex loop.</exception>
        /// <exception cref="ValueError">A float or decimal loop.</exception>
        private static void PackSequence(string typeName, NPTypeCode loop)
        {
            if (IsIntegerLoop(loop))
                throw new TypeError($"int() argument must be a string, a bytes-like object or a real number, not '{typeName}'");
            if (loop == NPTypeCode.Complex)
                throw new TypeError($"must be real number, not {typeName}");
            throw new ValueError("setting an array element with a sequence.");
        }

        /// <summary>
        ///     Writes a double into a float, float16 or complex loop element: float32 by the hardware's round-to-nearest-even
        ///     narrowing, float16 by NumPy's <c>npy_double_to_half</c> (one rounding from double), complex as the real part.
        /// </summary>
        /// <param name="d">The value.</param>
        /// <param name="loop">Half, Single, Double or Complex.</param>
        /// <param name="slot">Receives the element.</param>
        /// <exception cref="NotSupportedException"><paramref name="loop"/> is not a float or complex dtype — a caller bug.</exception>
        private static void WriteFloat(double d, NPTypeCode loop, Span<byte> slot)
        {
            switch (loop)
            {
                case NPTypeCode.Double: Unsafe.WriteUnaligned(ref slot[0], d); break;
                case NPTypeCode.Single: Unsafe.WriteUnaligned(ref slot[0], (float)d); break;
                case NPTypeCode.Half: Unsafe.WriteUnaligned(ref slot[0], DirectILKernelGenerator.DoubleToHalfBits(d)); break;
                case NPTypeCode.Complex: Unsafe.WriteUnaligned(ref slot[0], new Complex(d, 0)); break;
                default: throw new NotSupportedException($"ReductionInitial.WriteFloat: {loop} is not a float loop — caller bug.");
            }
        }

        /// <summary>Writes an integer already range-checked for <paramref name="loop"/>.</summary>
        /// <param name="v">The value (inside the loop's range).</param>
        /// <param name="loop">An integer loop dtype.</param>
        /// <param name="slot">Receives the element.</param>
        /// <exception cref="NotSupportedException"><paramref name="loop"/> is not an integer dtype — a caller bug.</exception>
        private static void WriteInteger(BigInteger v, NPTypeCode loop, Span<byte> slot)
        {
            switch (loop)
            {
                case NPTypeCode.SByte: Unsafe.WriteUnaligned(ref slot[0], (sbyte)v); break;
                case NPTypeCode.Byte: Unsafe.WriteUnaligned(ref slot[0], (byte)v); break;
                case NPTypeCode.Int16: Unsafe.WriteUnaligned(ref slot[0], (short)v); break;
                case NPTypeCode.UInt16: Unsafe.WriteUnaligned(ref slot[0], (ushort)v); break;
                case NPTypeCode.Char: Unsafe.WriteUnaligned(ref slot[0], (char)(ushort)v); break;
                case NPTypeCode.Int32: Unsafe.WriteUnaligned(ref slot[0], (int)v); break;
                case NPTypeCode.UInt32: Unsafe.WriteUnaligned(ref slot[0], (uint)v); break;
                case NPTypeCode.Int64: Unsafe.WriteUnaligned(ref slot[0], (long)v); break;
                case NPTypeCode.UInt64: Unsafe.WriteUnaligned(ref slot[0], (ulong)v); break;
                default: throw new NotSupportedException($"ReductionInitial.WriteInteger: {loop} is not an integer loop — caller bug.");
            }
        }

        /// <summary>
        ///     Casts one NumPy scalar into the loop dtype — <c>npy_cast_raw_scalar_item</c>, the unsafe cast NumPy applies to
        ///     a NumPy scalar whose DType the loop's DType does not know as a scalar type.
        /// </summary>
        /// <typeparam name="T">The scalar's element type.</typeparam>
        /// <param name="value">The scalar.</param>
        /// <param name="from">The scalar's dtype.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="slot">Receives the element.</param>
        /// <remarks>The house converter carries NumPy's (MSVC-pinned) unsafe conversions, the same ones every cast tier
        ///     gates, so an out-of-range or NaN float converts to the integer NumPy's cast loop produces.</remarks>
        private static unsafe void CastScalar<T>(T value, NPTypeCode from, NPTypeCode loop, Span<byte> slot) where T : unmanaged
        {
            fixed (byte* d = slot)
                NDIterCasting.ConvertValue((byte*)&value, d, from, loop);
        }

        /// <summary>
        ///     True for NumPy's SIGNED integer dtypes (int8 / int16 / int32 / int64) — the DTypes whose
        ///     <c>is_known_scalar_type</c> accepts every NumPy scalar, so a NumPy scalar initial value goes through their
        ///     range-checked setitem instead of a cast.
        /// </summary>
        /// <param name="t">The loop dtype.</param>
        /// <returns>Whether <paramref name="t"/> is a signed integer dtype.</returns>
        private static bool IsSignedIntegerLoop(NPTypeCode t)
            => t is NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.Int32 or NPTypeCode.Int64;

        /// <summary>True for the dtypes NumPy's integer setitem serves (char included, as NumSharp's uint16).</summary>
        /// <param name="t">The loop dtype.</param>
        /// <returns>Whether <paramref name="t"/> is an integer dtype.</returns>
        private static bool IsIntegerLoop(NPTypeCode t)
            => t is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char
                or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64;

        /// <summary>
        ///     Python's <c>float(str)</c> (CPython 3.12's grammar: Unicode digits and whitespace rewritten, single underscores
        ///     between digits, <c>inf</c> / <c>infinity</c> / <c>nan</c> in any case, <c>-nan</c> with the sign bit set).
        /// </summary>
        /// <param name="s">The text.</param>
        /// <returns>The correctly rounded double.</returns>
        /// <exception cref="ValueError">The text is not a float literal: "could not convert string to float: '…'".</exception>
        internal static unsafe double ParsePythonFloat(string s)
        {
            double value;
            Strings.NDTextParse.ParseObject(s, NPTypeCode.Double, (byte*)&value);
            return value;
        }
    }
}
