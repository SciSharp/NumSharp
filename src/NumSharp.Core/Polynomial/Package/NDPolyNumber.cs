using System;
using System.Collections;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

// =============================================================================
// NDPolyNumber.cs — one numpy.polynomial "value": a Python scalar, a NumPy scalar or an array
// =============================================================================
//
// numpy.polynomial's scalar code (polyutils.mapparms, the {p}line constructors, trimcoef's tolerance,
// mapdomain's offset/scale) never declares a dtype: it runs Python operators on whatever the caller passed,
// and three different arithmetics answer depending on what the operands ARE:
//
//   Python scalar ∘ Python scalar   -> CPython (exact ints, int/int true division, 3.12 complex division,
//                                      ZeroDivisionError on a zero divisor)                   [PyScalar]
//   NumPy scalar  ∘ scalar          -> scalarmath: NEP 50 dtype, the NAIVE complex product,
//                                      IEEE inf/nan on division by zero                       [kernels]
//   anything      ∘ ndarray         -> a ufunc: NEP 50 dtype, the FUSED (simd_cmul) complex
//                                      product; a 0-d result comes back as a NumPy scalar     [kernels / np.*]
//
// PolyNumber carries which of the three a value is (PolyNumberKind) plus its payload, and Binary/Negate/
// NotZero/LessZero dispatch exactly as Python's operator protocol would. The C# boundary mapping follows the
// house NEP 50 convention: bool, every integer primitive, float, double and Complex are PYTHON scalars
// (weak); Half, char and decimal have no Python literal and are NumPy SCALARS (strong); an NDArray is an
// ndarray (a 0-d one included — NumPy's np.asanyarray keeps it an array); a typed C# array is an ndarray of
// its element dtype, while a ValueTuple, object[] (and any other non-dtype array: jagged, NDArray[]) and any other
// IList/IEnumerable are Python sequences, coerced by np.array's nested discovery (see UnmanagedStorage.Coercion.Sequence.cs).
//
// A NumPy scalar's value lives inline (16 raw bytes — every NumSharp dtype fits), so scalarmath chains such
// as mapparms' seven operations never allocate; only an ndarray operand reaches the NDArray ufuncs.
//
// COMPLEX64. NumSharp has one complex dtype (complex128, #569), but NEP 50 makes a complex64 NumPy scalar out of
// a float16/float32 NumPy value and a Python complex (np.float32(a) + 1j), and it stays complex64 against Python
// numbers and narrow NumPy values. Such a scalar is carried as a Complex with float32-exact components and the
// IsComplex64 flag, and its scalarmath runs in float32 (DirectILKernelGenerator.PolyComplex64*: the operands are
// rounded into the loop, every operation rounds) — so a complex128 result built from it (np.array([c64, 1j]),
// a complex64 offset scaling a float64 array) matches NumPy bit for bit, and a complex64 result has NumPy's
// values in the complex128 dtype. A complex64 ARRAY loop (rank >= 1) is not emulated: it runs in complex128.
//
// PLATFORM NOTE (win-amd64, the NumPy build the corpus is generated from): a Python int converted to an
// integer dtype goes through C `long` (32-bit on Windows) for int8..int32/uint8/uint16, `unsigned long` for
// uint32, and `long long` for int64/uint64 — so the OverflowError TEXT for a huge int depends on the dtype
// (NumPy arraytypes.c.src @TYPE@_safe_pyint_setitem). WeakIntToInteger reproduces the Windows texts.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>What a <see cref="PolyNumber"/> is, which decides whose arithmetic applies to it.</summary>
    internal enum PolyNumberKind : byte
    {
        /// <summary>A Python scalar (weak under NEP 50; CPython arithmetic with another Python scalar).</summary>
        Python,
        /// <summary>A NumPy scalar (<c>np.generic</c>: strong dtype, scalarmath).</summary>
        Scalar,
        /// <summary>An ndarray of any rank, 0-d included (strong dtype, ufunc arithmetic).</summary>
        Array,
    }

    /// <summary>Two <see cref="PolyNumber"/>s in one inline buffer — a two-element list without a heap array.</summary>
    [System.Runtime.CompilerServices.InlineArray(2)]
    internal struct PolyNumberPair
    {
        /// <summary>The first element (the rest follow inline).</summary>
        private PolyNumber _e0;
    }

    /// <summary>Sixteen raw bytes — room for one element of any NumSharp dtype (Complex and decimal are 16).</summary>
    [StructLayout(LayoutKind.Sequential, Size = 16)]
    internal struct PolyRaw16
    {
        /// <summary>Low eight bytes.</summary>
        public ulong Lo;
        /// <summary>High eight bytes.</summary>
        public ulong Hi;
    }

    /// <summary>
    ///     One operand of numpy.polynomial's Python-level scalar code — a Python scalar, a NumPy scalar or an
    ///     ndarray — with NumPy 2.4.2 / CPython 3.12's arithmetic between them (see the file header).
    /// </summary>
    /// <remarks>
    ///     Values are immutable. An <see cref="PolyNumberKind.Array"/> value references its NDArray (it does not
    ///     own it: arrays built by the arithmetic are tracked by the caller's <see cref="NDScope"/> like any
    ///     other allocation, so an <c>[NDScoped]</c> facade reclaims every intermediate).
    /// </remarks>
    internal readonly unsafe struct PolyNumber
    {
        /// <summary>Which arithmetic applies.</summary>
        public readonly PolyNumberKind Kind;
        /// <summary>The Python value (<see cref="PolyNumberKind.Python"/> only).</summary>
        public readonly PyScalar Py;
        /// <summary>The dtype of a NumPy scalar or ndarray (<see cref="NPTypeCode.Empty"/> for a Python value).</summary>
        public readonly NPTypeCode Dtype;
        /// <summary>A NumPy scalar's raw element bytes (the first <c>itemsize</c> bytes are meaningful).</summary>
        private readonly PolyRaw16 _raw;
        /// <summary>The ndarray (<see cref="PolyNumberKind.Array"/> only).</summary>
        public readonly NDArray Array;
        /// <summary>
        ///     A NumPy SCALAR of NumPy's complex64 (<see cref="Dtype"/> is <see cref="NPTypeCode.Complex"/>, both
        ///     components float32-exact). NumSharp has no complex64 dtype (#569), but numpy.polynomial's scalar code
        ///     makes such scalars — a float16/float32 NumPy value meeting a Python complex — and their arithmetic is
        ///     NumPy's float32 scalarmath, emulated exactly. Only a scalar can carry it: an ndarray result is complex128.
        /// </summary>
        public readonly bool IsComplex64;

        private PolyNumber(PolyNumberKind kind, in PyScalar py, NPTypeCode dtype, in PolyRaw16 raw, NDArray array, bool complex64 = false)
        {
            Kind = kind; Py = py; Dtype = dtype; _raw = raw; Array = array; IsComplex64 = complex64;
        }

        // Integer range bounds as BigIntegers, built once: comparing a BigInteger with a long/ulong literal converts
        // the literal to a BigInteger on EVERY comparison, and a bound outside the int range allocates its digits.
        /// <summary><c>long.MinValue</c>.</summary>
        private static readonly BigInteger s_longMin = long.MinValue;
        /// <summary><c>long.MaxValue</c>.</summary>
        private static readonly BigInteger s_longMax = long.MaxValue;
        /// <summary><c>ulong.MaxValue</c>.</summary>
        private static readonly BigInteger s_ulongMax = ulong.MaxValue;
        /// <summary><c>uint.MaxValue</c>.</summary>
        private static readonly BigInteger s_uintMax = uint.MaxValue;
        /// <summary><c>decimal.MinValue</c>.</summary>
        private static readonly BigInteger s_decimalMin = new BigInteger(decimal.MinValue);
        /// <summary><c>decimal.MaxValue</c>.</summary>
        private static readonly BigInteger s_decimalMax = new BigInteger(decimal.MaxValue);

        /// <summary>A Python scalar.</summary>
        /// <param name="v">The value.</param>
        /// <returns>The number.</returns>
        public static PolyNumber FromPython(in PyScalar v) => new PolyNumber(PolyNumberKind.Python, v, NPTypeCode.Empty, default, null);

        /// <summary>A NumPy scalar of dtype <paramref name="t"/> whose element bytes are at <paramref name="value"/>.</summary>
        /// <param name="t">The dtype.</param>
        /// <param name="value">Address of one element of <paramref name="t"/> (copied).</param>
        /// <returns>The number.</returns>
        public static PolyNumber FromScalar(NPTypeCode t, void* value)
        {
            PolyRaw16 raw = default;
            Buffer.MemoryCopy(value, &raw, 16, DirectILKernelGenerator.GetTypeSize(t));
            return new PolyNumber(PolyNumberKind.Scalar, default, t, raw, null);
        }

        /// <summary>A NumPy scalar holding the C# value <paramref name="v"/> in its own dtype.</summary>
        /// <typeparam name="T">A NumSharp element type.</typeparam>
        /// <param name="v">The value.</param>
        /// <returns>The number.</returns>
        public static PolyNumber FromScalar<T>(T v) where T : unmanaged => FromScalar(InfoOf<T>.NPTypeCode, &v);

        /// <summary>
        ///     A NumPy complex64 scalar (see <see cref="IsComplex64"/>) whose value is the <see cref="Complex"/> at
        ///     <paramref name="value"/> — the caller guarantees float32-exact components (a complex64 kernel's result).
        /// </summary>
        /// <param name="value">Address of one <see cref="Complex"/> (copied).</param>
        /// <returns>The number.</returns>
        public static PolyNumber FromComplex64(void* value)
        {
            PolyRaw16 raw = default;
            Buffer.MemoryCopy(value, &raw, 16, 16);
            return new PolyNumber(PolyNumberKind.Scalar, default, NPTypeCode.Complex, raw, null, complex64: true);
        }

        /// <summary>
        ///     The NumPy scalar a 0-d ufunc result unwraps to (NumPy returns <c>np.float64(…)</c>, not a 0-d array,
        ///     from any ufunc whose output is 0-d).
        /// </summary>
        /// <param name="zeroDim">A 0-d array.</param>
        /// <returns>The scalar (its value copied out).</returns>
        public static PolyNumber ScalarOf(NDArray zeroDim)
            => FromScalar(zeroDim.typecode, (byte*)zeroDim.Storage.Address + zeroDim.Shape.offset * zeroDim.dtypesize);

        /// <summary>An ndarray (any rank).</summary>
        /// <param name="a">The array (not copied).</param>
        /// <returns>The number.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        public static PolyNumber FromArray(NDArray a)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return new PolyNumber(PolyNumberKind.Array, default, a.typecode, default, a);
        }

        /// <summary>Whether this is a Python value.</summary>
        public bool IsPython => Kind == PolyNumberKind.Python;

        /// <summary>Whether this is an ndarray of rank ≥ 1 (the only kind whose arithmetic produces an array).</summary>
        public bool IsNdArray => Kind == PolyNumberKind.Array && Array.ndim > 0;

        /// <summary>Whether this is a 0-d ndarray: scalar-sized, but a ufunc operand (fused complex product).</summary>
        public bool IsZeroDimArray => Kind == PolyNumberKind.Array && Array.ndim == 0;

        /// <summary>
        ///     Whether this is an ndarray of a string dtype (U, S or StringDType; any rank). Its arithmetic is not a numeric
        ///     loop: the house ufunc must see the operands as they are, to run NumPy's string loop or raise NumPy's no-loop
        ///     TypeError for the pair.
        /// </summary>
        public bool IsStringArray => Kind == PolyNumberKind.Array && Array.typecode == NPTypeCode.String;

        /// <summary>
        ///     Classifies a C# value by the house NEP 50 mapping: bool / integer primitives / BigInteger / float /
        ///     double / Complex are Python scalars; Half / char / decimal are NumPy scalars of their dtype; an
        ///     NDArray is an ndarray; a typed C# array (or <see cref="Memory{T}"/>) of a dtype is converted with
        ///     <see cref="np.asanyarray(object, DType, char)"/> to an ndarray; a Python SEQUENCE — a tuple
        ///     (<see cref="ITuple"/>), <c>object[]</c> or any other non-dtype C# array (jagged, <c>NDArray[]</c>), any other
        ///     enumerable — is coerced by np.array's rules, nested to any depth (<see cref="Backends.UnmanagedStorage.FromSequence"/>). A
        ///     boxed <see cref="PolyNumber"/> is returned as it is: the internal spelling of a NumPy SCALAR (no C# value is
        ///     one — a 0-d NDArray is an ndarray here) that the series classes hand on through an <c>object</c> parameter.
        /// </summary>
        /// <param name="o">The value.</param>
        /// <returns>The number.</returns>
        /// <exception cref="NotSupportedException"><paramref name="o"/> is null or a string — NumPy would build an
        ///     object/str array from it, dtypes NumSharp does not have — or a sequence holding one.</exception>
        /// <exception cref="ValueError">A ragged sequence (NumPy's inhomogeneous-shape text).</exception>
        public static PolyNumber FromObject(object o)
        {
            switch (o)
            {
                case null:
                    throw new NotSupportedException("None has no NumSharp equivalent: NumPy would build an object array, a dtype NumSharp does not have");
                case NDArray nd: return FromArray(nd);
                // A value the package already classified, passed on boxed through an `object` parameter: the series
                // classes compute NumPy SCALARS (mapparms' off/scl, integ's `off + scl*lbnd`, `1./scl`) and hand them to
                // the module functions as NumPy hands them, without a 0-d array — which NumPy would read as an ndarray.
                case PolyNumber p: return p;
                case bool b: return FromPython(PyScalar.Bool(b));
                case sbyte v: return FromPython(PyScalar.Int(v));
                case byte v: return FromPython(PyScalar.Int(v));
                case short v: return FromPython(PyScalar.Int(v));
                case ushort v: return FromPython(PyScalar.Int(v));
                case int v: return FromPython(PyScalar.Int(v));
                case uint v: return FromPython(PyScalar.Int(v));
                case long v: return FromPython(PyScalar.Int(v));
                case ulong v: return FromPython(PyScalar.Int(new BigInteger(v)));
                case BigInteger v: return FromPython(PyScalar.Int(v));
                case float v: return FromPython(PyScalar.Float(v));   // a C# float is a Python float (house rule), widened exactly
                case double v: return FromPython(PyScalar.Float(v));
                case Complex v: return FromPython(PyScalar.Cplx(v));
                case Half v: return FromScalar(v);                    // no Python literal: np.float16
                case char v: return FromScalar(v);
                case decimal v: return FromScalar(v);
                case string:
                    throw new NotSupportedException("a Python str operand makes NumPy build a str/object array, a dtype NumSharp does not have");
                default:
                    // A Python sequence goes through np.array's coercion (np.asanyarray converts only ONE level of an
                    // object[] / tuple, and discovers a C# int as int32 where a Python int is int64); an ndarray-like
                    // (a typed C# array, Memory<T>) converts as a whole.
                    return FromArray(Backends.UnmanagedStorage.IsSequence(o) ? Backends.UnmanagedStorage.FromSequence(o) : np.asanyarray(o));
            }
        }

        /// <summary>
        ///     Whether <see cref="FromObject"/> classifies <paramref name="o"/> as a Python or NumPy SCALAR — a C#
        ///     number — which allocates nothing (every other value is, or converts to, an ndarray, or throws).
        ///     Callers use it to skip an <see cref="NDScope"/> when no intermediate array can appear.
        /// </summary>
        /// <param name="o">The value.</param>
        /// <returns>True for bool, the integer primitives, BigInteger, float, double, Complex, Half, char, decimal, and a
        ///     boxed scalar-kind <see cref="PolyNumber"/> (a Python or NumPy scalar the series classes pass on).</returns>
        public static bool IsScalarValue(object o)
            => o is bool or sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger
                 or float or double or Complex or Half or char or decimal
               || o is PolyNumber { Kind: not PolyNumberKind.Array };

        /// <summary>
        ///     <see cref="FromObject"/> for a statically typed value — the same classification, but a C# primitive,
        ///     <see cref="Complex"/>, <see cref="Half"/>, <c>char</c> or <c>decimal</c> is read without boxing (each
        ///     <c>typeof(T)</c> test folds away when the JIT specializes a value-type <typeparamref name="T"/>).
        ///     Reference types and unlisted value types box into <see cref="FromObject"/>, so the two can never
        ///     disagree.
        /// </summary>
        /// <typeparam name="T">The value's static type (a generic tuple element of a domain argument).</typeparam>
        /// <param name="v">The value.</param>
        /// <returns>The number.</returns>
        /// <exception cref="NotSupportedException">A null or string value (see <see cref="FromObject"/>).</exception>
        public static PolyNumber FromValue<T>(T v)
        {
            // Python scalars (weak): the house mapping of every C# numeric primitive.
            if (typeof(T) == typeof(double)) return FromPython(PyScalar.Float(Unsafe.As<T, double>(ref v)));
            if (typeof(T) == typeof(int)) return FromPython(PyScalar.Int(Unsafe.As<T, int>(ref v)));
            if (typeof(T) == typeof(long)) return FromPython(PyScalar.Int(Unsafe.As<T, long>(ref v)));
            if (typeof(T) == typeof(float)) return FromPython(PyScalar.Float(Unsafe.As<T, float>(ref v)));
            if (typeof(T) == typeof(bool)) return FromPython(PyScalar.Bool(Unsafe.As<T, bool>(ref v)));
            if (typeof(T) == typeof(sbyte)) return FromPython(PyScalar.Int(Unsafe.As<T, sbyte>(ref v)));
            if (typeof(T) == typeof(byte)) return FromPython(PyScalar.Int(Unsafe.As<T, byte>(ref v)));
            if (typeof(T) == typeof(short)) return FromPython(PyScalar.Int(Unsafe.As<T, short>(ref v)));
            if (typeof(T) == typeof(ushort)) return FromPython(PyScalar.Int(Unsafe.As<T, ushort>(ref v)));
            if (typeof(T) == typeof(uint)) return FromPython(PyScalar.Int(Unsafe.As<T, uint>(ref v)));
            if (typeof(T) == typeof(ulong)) return FromPython(PyScalar.Int(new BigInteger(Unsafe.As<T, ulong>(ref v))));
            if (typeof(T) == typeof(Complex)) return FromPython(PyScalar.Cplx(Unsafe.As<T, Complex>(ref v)));
            // NumPy scalars (strong): no Python literal exists for these.
            if (typeof(T) == typeof(Half)) return FromScalar(Unsafe.As<T, Half>(ref v));
            if (typeof(T) == typeof(char)) return FromScalar(Unsafe.As<T, char>(ref v));
            if (typeof(T) == typeof(decimal)) return FromScalar(Unsafe.As<T, decimal>(ref v));
            return FromObject(v);
        }

        // ---------------------------------------------------------------------------------------------
        //  Dtype rules
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NEP 50: the dtype a NumPy value of dtype <paramref name="partner"/> and a weak Python value
        ///     <paramref name="py"/> are computed in. A Python bool keeps any partner's dtype (<c>np.int8(1) + True</c>
        ///     is int8, <c>np.True_ + True</c> is bool); the int/float/complex rules are
        ///     <see cref="PolyTyping.WeakPromote"/>'s.
        /// </summary>
        /// <param name="partner">The strong operand's dtype.</param>
        /// <param name="py">The Python operand.</param>
        /// <returns>The loop dtype.</returns>
        public static NPTypeCode WeakPromote(NPTypeCode partner, in PyScalar py)
            => py.IsBool ? partner : PolyTyping.WeakPromote(partner, py.Kind);

        /// <summary>
        ///     The loop dtype of <c>a op b</c> when at least one operand is a NumPy scalar or an ndarray: NEP 50
        ///     promotion, then true division of an integer (or bool) loop is float64 (NumPy's <c>'dd-&gt;d'</c>).
        /// </summary>
        /// <param name="op">The operator.</param>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <returns>The dtype both operands are converted to and the result has.</returns>
        /// <exception cref="InvalidOperationException">Both operands are Python values (CPython's arithmetic, not a loop).</exception>
        public static NPTypeCode LoopDtype(BinaryOp op, in PolyNumber a, in PolyNumber b)
        {
            NPTypeCode t;
            if (a.IsPython && b.IsPython)
                throw new InvalidOperationException("two Python operands have no NumPy loop");
            else if (a.IsPython)
                t = WeakPromote(b.Dtype, a.Py);
            else if (b.IsPython)
                t = WeakPromote(a.Dtype, b.Py);
            else
                t = PolyTyping.Promote(a.Dtype, b.Dtype);
            if (op == BinaryOp.Divide && PolyTyping.IsIntLike(t))
                t = NPTypeCode.Double;
            return t;
        }

        /// <summary>
        ///     Whether a <see cref="NPTypeCode.Complex"/> loop of <c>a op b</c> is NumPy's COMPLEX64 loop (probed
        ///     2.4.2): a weak Python complex with a float16/float32 partner (<c>np.float32(1) + 1j</c>), a weak Python
        ///     int/float/bool/complex with a complex64 partner, or a complex64 value with a NumPy bool/int8/uint8/
        ///     int16/uint16/float16/float32 (or NumSharp char, a uint16) — every wider partner (int32+, float64,
        ///     complex128) promotes to complex128. Only meaningful when <see cref="LoopDtype"/> is complex.
        /// </summary>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <returns>True when NumPy computes the operation in complex64.</returns>
        public static bool IsComplex64Loop(in PolyNumber a, in PolyNumber b)
        {
            if (a.IsPython)
                return WeakMakesComplex64(b, a.Py);
            if (b.IsPython)
                return WeakMakesComplex64(a, b.Py);
            if (a.IsComplex64)
                return b.IsComplex64 || KeepsComplex64(b.Dtype);
            return b.IsComplex64 && KeepsComplex64(a.Dtype);
        }

        /// <summary>
        ///     <see cref="IsComplex64Loop(in PolyNumber, in PolyNumber)"/> against an operand known only by its dtype —
        ///     an ndarray of rank ≥ 1 or an intermediate array result, which is never a Python value: whether
        ///     <c>a op b</c>, with <paramref name="a"/> a scalar-sized operand and <c>b</c> an array of
        ///     <paramref name="bDtype"/> (complex64 when <paramref name="bIsComplex64"/>), runs NumPy's complex64 loop.
        /// </summary>
        /// <param name="a">The scalar-sized operand (a Python value, a NumPy scalar, or a complex64 carrier).</param>
        /// <param name="bDtype">The array operand's dtype (<see cref="NPTypeCode.Complex"/> for a complex64 array).</param>
        /// <param name="bIsComplex64">The array operand is complex64 (an intermediate NumSharp carries as complex128).</param>
        /// <returns>True when NumPy computes the operation in complex64.</returns>
        public static bool IsComplex64Loop(in PolyNumber a, NPTypeCode bDtype, bool bIsComplex64)
        {
            if (a.IsPython)
                return bIsComplex64 || (a.Py.Kind == PyKind.Complex && bDtype is NPTypeCode.Half or NPTypeCode.Single);
            if (a.IsComplex64)
                return bIsComplex64 || KeepsComplex64(bDtype);
            return bIsComplex64 && KeepsComplex64(a.Dtype);
        }

        /// <summary>NEP 50 for a weak Python value meeting a NumPy value: complex64 stays complex64 whatever the Python
        ///     kind; a Python complex turns a float16/float32 partner into complex64.</summary>
        /// <param name="strong">The NumPy operand.</param>
        /// <param name="py">The Python operand.</param>
        /// <returns>True for a complex64 loop.</returns>
        private static bool WeakMakesComplex64(in PolyNumber strong, in PyScalar py)
            => strong.IsComplex64 || (py.Kind == PyKind.Complex && strong.Dtype is NPTypeCode.Half or NPTypeCode.Single);

        /// <summary><c>np.promote_types(complex64, t) == complex64</c>: the dtypes a complex64 value absorbs.</summary>
        /// <param name="t">The partner dtype.</param>
        /// <returns>True for bool, int8, uint8, int16, uint16, char, float16 and float32.</returns>
        private static bool KeepsComplex64(NPTypeCode t)
            => t is NPTypeCode.Boolean or NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16
                 or NPTypeCode.Char or NPTypeCode.Half or NPTypeCode.Single;

        /// <summary>
        ///     The dtype <c>np.array</c> DISCOVERS for this value as a list element (array coercion, not NEP 50
        ///     promotion): a Python bool is bool, an int int64 — uint64 when it only fits there — a float float64,
        ///     a complex complex128; a NumPy scalar or ndarray keeps its dtype.
        /// </summary>
        /// <returns>The discovered dtype.</returns>
        /// <exception cref="NotSupportedException">A Python int beyond uint64: NumPy builds an OBJECT array
        ///     (<c>np.array([2**64, 1])</c>), a dtype NumSharp does not have.</exception>
        public NPTypeCode DiscoveredDtype()
            => TryDiscoveredDtype(out NPTypeCode t) ? t : throw ObjectArrayRefusal();

        /// <summary>
        ///     <see cref="DiscoveredDtype"/> without the exception: false for a Python int beyond uint64, the one value
        ///     np.array cannot give a numeric dtype (it builds an OBJECT array).
        /// </summary>
        /// <param name="t">The discovered dtype (<see cref="NPTypeCode.Empty"/> when false).</param>
        /// <returns>False for a Python int beyond uint64; true otherwise.</returns>
        /// <remarks>
        ///     Array coercion classifies EVERY leaf of a sequence, and an object series (a Python int past uint64 among its
        ///     terms) is a legitimate input numpy.polynomial computes with — throwing and catching here cost ~1 µs a leaf, half
        ///     of a two-term object series' whole <c>{p}roots</c>. The refusal object is <see cref="ObjectArrayRefusal"/>.
        /// </remarks>
        public bool TryDiscoveredDtype(out NPTypeCode t)
        {
            if (!IsPython)
            {
                t = Dtype;
                return true;
            }
            if (Py.IsBool)
            {
                t = NPTypeCode.Boolean;
                return true;
            }
            switch (Py.Kind)
            {
                case PyKind.Int:
                    if (Py.I >= s_longMin && Py.I <= s_longMax)
                        t = NPTypeCode.Int64;
                    else if (Py.I.Sign >= 0 && Py.I <= s_ulongMax)
                        t = NPTypeCode.UInt64;
                    else
                    {
                        t = NPTypeCode.Empty;
                        return false;
                    }
                    return true;
                case PyKind.Float:
                    t = NPTypeCode.Double;
                    return true;
                default:
                    t = NPTypeCode.Complex;
                    return true;
            }
        }

        /// <summary>
        ///     The NotSupportedException <see cref="DiscoveredDtype"/> raises for a Python int beyond uint64, built but NOT
        ///     thrown — array coercion keeps it as the deferred refusal of an object array and raises it only where NumPy
        ///     would start computing with Python objects.
        /// </summary>
        /// <returns>The exception (its message names the value).</returns>
        public NotSupportedException ObjectArrayRefusal()
            => new NotSupportedException(
                $"Python int {Py.I} fits neither int64 nor uint64: NumPy builds an object array for it, a dtype NumSharp does not have");

        // ---------------------------------------------------------------------------------------------
        //  Value conversion
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Writes this scalar-sized value converted to <paramref name="t"/> into <paramref name="dst"/> — the
        ///     conversion NumPy applies to an operand entering a <paramref name="t"/> loop: a NumPy scalar or 0-d
        ///     array is cast (<c>astype</c>), a Python value converted by <see cref="WriteWeak"/>.
        /// </summary>
        /// <param name="t">Target dtype.</param>
        /// <param name="dst">Destination (one element of <paramref name="t"/>).</param>
        /// <exception cref="InvalidOperationException">An ndarray of rank ≥ 1 (it has no single value).</exception>
        /// <exception cref="OverflowException">A Python int that does not fit (see <see cref="WriteWeak"/>).</exception>
        public void WriteAs(NPTypeCode t, void* dst)
        {
            if (IsPython)
            {
                WriteWeak(Py, t, dst);
                return;
            }
            if (IsNdArray)
                throw new InvalidOperationException("an ndarray of rank >= 1 has no single value");
            PolyRaw16 raw = _raw;
            void* src = &raw;
            if (Kind == PolyNumberKind.Array)
            {
                var a = Array;
                src = (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;
            }
            PolyConstPool.ConvertBuffer(src, Dtype, dst, t, 1);
        }

        /// <summary>
        ///     NumPy's conversion of a WEAK Python value into a <paramref name="t"/> operand (NEP 50): a complex
        ///     stores both parts; an int into an integer dtype is range-checked with NumPy's win-amd64 texts
        ///     (<see cref="WeakIntToInteger"/>); an int into decimal is exact; an int into a float dtype goes
        ///     through CPython's correctly rounded <c>float(int)</c> and then the house cast (NumPy's
        ///     <c>PyFloat_AsDouble</c> + setitem), so float16/float32 see NumPy's double rounding; a float is cast.
        /// </summary>
        /// <param name="v">The Python value.</param>
        /// <param name="t">Target dtype (already NEP 50-resolved, so never narrower than the value's kind).</param>
        /// <param name="dst">Destination (one element of <paramref name="t"/>).</param>
        /// <exception cref="OverflowException">An int out of range for an integer <paramref name="t"/>, too large for
        ///     a float (<c>int too large to convert to float</c>) or for a decimal.</exception>
        public static void WriteWeak(in PyScalar v, NPTypeCode t, void* dst)
        {
            if (v.Kind == PyKind.Complex)
            {
                Complex c = v.C;
                PolyConstPool.ConvertBuffer(&c, NPTypeCode.Complex, dst, t, 1);
                return;
            }
            if (v.Kind == PyKind.Int && (PolyTyping.IsIntLike(t) || t == NPTypeCode.Decimal))
            {
                if (t == NPTypeCode.Decimal)
                {
                    if (v.I < s_decimalMin || v.I > s_decimalMax)
                        throw new OverflowException("Python int too large to convert to a decimal value");
                    *(decimal*)dst = (decimal)v.I;
                    return;
                }
                if (t == NPTypeCode.Boolean)
                {
                    // Only a weak BOOL keeps a bool loop (a weak int makes it int64), so the value is 0/1.
                    *(bool*)dst = !v.I.IsZero;
                    return;
                }
                if (t == NPTypeCode.UInt64 && v.I > s_longMax)
                {
                    ulong u = WeakIntToUnsigned64(v.I);
                    *(ulong*)dst = u;
                    return;
                }
                long value = WeakIntToInteger(v.I, t);
                PolyConstPool.ConvertBuffer(&value, NPTypeCode.Int64, dst, t, 1);
                return;
            }
            double d = v.AsDouble();
            PolyConstPool.ConvertBuffer(&d, NPTypeCode.Double, dst, t, 1);
        }

        /// <summary>
        ///     NumPy's <c>@TYPE@_safe_pyint_setitem</c> for a Python int entering an integer dtype on win-amd64:
        ///     int8 / int16 / int32 / uint8 / uint16 (and NumSharp's char, a uint16) convert through C
        ///     <c>long</c> — 32-bit on Windows — so any |value| ≥ 2³¹ is <c>Python int too large to convert to C
        ///     long</c> before the dtype is looked at; uint32 converts through <c>unsigned long</c> (0..2³²-1),
        ///     retrying a failure as <c>long</c> to detect a negative; int64 / uint64 through <c>long long</c>,
        ///     whose failure reads <c>int too big to convert</c>. A value that converts but does not fit the
        ///     dtype is <c>Python integer {v} out of bounds for {dtype}</c>.
        /// </summary>
        /// <param name="v">The Python int.</param>
        /// <param name="t">An integer dtype (not bool, not uint64 above int64's range — see
        ///     <see cref="WeakIntToUnsigned64"/>).</param>
        /// <returns>The value (in range for <paramref name="t"/>).</returns>
        /// <exception cref="OverflowException">NumPy's OverflowError, with its text.</exception>
        public static long WeakIntToInteger(BigInteger v, NPTypeCode t)
        {
            string OutOfBounds() => $"Python integer {v} out of bounds for {t.AsNumpyDtypeName()}";
            const long LongMin = int.MinValue, LongMax = int.MaxValue;   // C long on win-amd64
            switch (t)
            {
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.Int32:
                case NPTypeCode.Byte:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                {
                    if (v < LongMin || v > LongMax)
                        throw new OverflowException("Python int too large to convert to C long");
                    long x = (long)v;
                    var (lo, hi) = IntegerRange(t);
                    if (x < lo || x > hi)
                        throw new OverflowException(OutOfBounds());
                    return x;
                }
                case NPTypeCode.UInt32:
                {
                    // PyLong_AsUnsignedLong (0..2^32-1); on failure PyLong_AsLong decides between "negative"
                    // (out of bounds) and "too large".
                    if (v.Sign >= 0 && v <= s_uintMax)
                        return (long)v;
                    if (v < LongMin || v > LongMax)
                        throw new OverflowException("Python int too large to convert to C long");
                    throw new OverflowException(OutOfBounds());
                }
                case NPTypeCode.Int64:
                    if (v < s_longMin || v > s_longMax)
                        throw new OverflowException("int too big to convert");
                    return (long)v;
                case NPTypeCode.UInt64:
                    // Values above long.MaxValue go through WeakIntToUnsigned64; here v <= long.MaxValue.
                    if (v >= 0) return (long)v;
                    if (v < s_longMin)
                        throw new OverflowException("int too big to convert");
                    throw new OverflowException(OutOfBounds());
                default:
                    throw new NotSupportedException($"{t} is not an integer dtype");
            }
        }

        /// <summary>The uint64 half of <see cref="WeakIntToInteger"/> for a value above <c>long.MaxValue</c>.</summary>
        /// <param name="v">The Python int (&gt; <c>long.MaxValue</c>).</param>
        /// <returns>The value.</returns>
        /// <exception cref="OverflowException">Beyond 2⁶⁴-1: <c>int too big to convert</c>.</exception>
        private static ulong WeakIntToUnsigned64(BigInteger v)
        {
            if (v > s_ulongMax)
                throw new OverflowException("int too big to convert");
            return (ulong)v;
        }

        /// <summary>The inclusive value range of a narrow integer dtype.</summary>
        /// <param name="t">int8/int16/int32/uint8/uint16/char.</param>
        /// <returns>(min, max).</returns>
        private static (long Lo, long Hi) IntegerRange(NPTypeCode t) => t switch
        {
            NPTypeCode.SByte => (sbyte.MinValue, sbyte.MaxValue),
            NPTypeCode.Int16 => (short.MinValue, short.MaxValue),
            NPTypeCode.Int32 => (int.MinValue, int.MaxValue),
            NPTypeCode.Byte => (byte.MinValue, byte.MaxValue),
            _ => (ushort.MinValue, ushort.MaxValue),   // uint16 and char
        };

        /// <summary>
        ///     This value as an NDArray of dtype <paramref name="t"/> (a 0-d one for a scalar-sized value) — the
        ///     operand a ufunc call receives after NumPy's conversion. An ndarray already of <paramref name="t"/>
        ///     is returned as is.
        /// </summary>
        /// <param name="t">Target dtype.</param>
        /// <returns>The array.</returns>
        /// <exception cref="OverflowException">A Python int that does not fit <paramref name="t"/>.</exception>
        public NDArray AsArrayOf(NPTypeCode t)
        {
            if (Kind == PolyNumberKind.Array && Array.ndim > 0)
                return Array.typecode == t ? Array : Array.astype(t);
            var r = new NDArray(t, Shape.Scalar, false);
            WriteAs(t, (byte*)r.Storage.Address + r.Shape.offset * r.dtypesize);
            return r;
        }

        // ---------------------------------------------------------------------------------------------
        //  Arithmetic
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>a op b</c> as Python evaluates it: CPython for two Python values, scalarmath when both are
        ///     scalar-sized and at least one is a NumPy scalar, a ufunc when an ndarray is involved — a 0-d
        ///     ndarray included, whose ufunc result is a NumPy scalar again. A scalar-sized operation in NumPy's
        ///     complex64 loop (<see cref="IsComplex64Loop"/>) runs in float32 and yields a complex64 scalar; one
        ///     with an ndarray of rank ≥ 1 runs in complex128 (NumSharp has no complex64 arrays, #569). A string array
        ///     operand (any rank) goes to the house ufunc unconverted: NumPy's string loop, or its no-loop TypeError.
        /// </summary>
        /// <param name="op">Add, Subtract, Multiply or Divide (true division).</param>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <returns>The result, of NumPy's kind and dtype.</returns>
        /// <exception cref="TypeError">A bool subtraction (NumPy refuses it), or a string array operand the house
        ///     ufunc has no loop for (NumPy's <c>ufunc 'add' did not contain a loop with signature matching types …</c>).</exception>
        /// <exception cref="DivideByZeroException">Python division by zero (never raised for NumPy values).</exception>
        /// <exception cref="OverflowException">A Python int that does not fit the loop dtype, or CPython's int/float
        ///     overflow.</exception>
        /// <exception cref="IncorrectShapeException">Ndarray operands that do not broadcast.</exception>
        public static PolyNumber Binary(BinaryOp op, in PolyNumber a, in PolyNumber b)
        {
            if (a.IsPython && b.IsPython)
                return FromPython(PyScalar.Apply(op, a.Py, b.Py));

            // Python's operator protocol for `L op R` with a Python L and a NumPy SCALAR R: R's reflected method
            // runs first only when type(R) subclasses type(L) (np.float64 subclasses float, np.complex128
            // complex); otherwise L's own method runs, and it declines every NumPy scalar EXCEPT one — CPython's
            // complex methods accept any float instance, np.float64 included. So `pycomplex op np.float64` is
            // CPython complex arithmetic with a Python-complex result (probed 2.4.2: 1j*np.float64(2) is the
            // Python complex 2j, 1j/np.float64(0) raises ZeroDivisionError).
            if (a.IsPython && a.Py.Kind == PyKind.Complex && b.Kind == PolyNumberKind.Scalar && b.Dtype == NPTypeCode.Double)
            {
                PolyRaw16 raw = b._raw;
                return FromPython(PyScalar.Apply(op, a.Py, PyScalar.Float(*(double*)&raw)));
            }

            // A string array: no numeric loop dtype exists to convert into. The house ufunc gets the operands as they
            // are — a Python value as the 0-d array its C# literal converts to, carrying NumPy's Python-literal mark, so
            // the message names it as NumPy does (dtype('float64') beside U / S, _PyFloatDType beside StringDType) — and
            // runs NumPy's string loop or raises NumPy's `ufunc 'add' did not contain a loop with signature matching
            // types (dtype('float64'), dtype('<U2')) -> None`.
            if (a.IsStringArray || b.IsStringArray)
                return FromArray(HouseUfunc(op, a.AsUfuncOperand(), b.AsUfuncOperand()));

            NPTypeCode t = LoopDtype(op, a, b);
            if (t == NPTypeCode.Boolean && op == BinaryOp.Subtract)
                throw new TypeError("numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, " +
                                    "the `^` operator, or the logical_xor function instead.");

            if (a.IsNdArray || b.IsNdArray)
                return ArrayBinary(op, a, b, t);

            // Scalar-sized operands: scalarmath (naive complex product) unless an operand is a 0-d ndarray, which
            // makes it a ufunc call (fused product) whose 0-d result NumPy hands back as a scalar.
            var product = a.IsZeroDimArray || b.IsZeroDimArray ? PolyComplexProduct.Simd : PolyComplexProduct.Naive;
            PolyRaw16 ra, rb, rr;
            a.WriteAs(t, &ra);
            b.WriteAs(t, &rb);
            if (t == NPTypeCode.Complex && IsComplex64Loop(a, b))
            {
                // NumPy's complex64 loop: both operands enter it rounded to float32 per component, the operation
                // runs in float32, and the result is a complex64 scalar (carried exactly in the Complex).
                *(Complex*)&ra = DirectILKernelGenerator.PolyComplex64Round(*(Complex*)&ra);
                *(Complex*)&rb = DirectILKernelGenerator.PolyComplex64Round(*(Complex*)&rb);
                DirectILKernelGenerator.GetPolyScalarComplex64Kernel(op, product)((byte*)&ra, (byte*)&rb, (byte*)&rr);
                return FromComplex64(&rr);
            }
            DirectILKernelGenerator.GetPolyScalarBinaryKernel(op, t, product)((byte*)&ra, (byte*)&rb, (byte*)&rr);
            return FromScalar(t, &rr);
        }

        /// <summary>
        ///     <c>a op b</c> with an ndarray of rank ≥ 1 involved: both operands converted to the loop dtype
        ///     <paramref name="t"/>, then the house ufunc — except the one product NumPy computes in its
        ///     single-element NpyIter loop (two array operands of different ndim broadcasting to ONE element),
        ///     which is the contracted <c>loop_scalar</c> complex product.
        /// </summary>
        /// <param name="op">The operator.</param>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <param name="t">The loop dtype.</param>
        /// <returns>The array result.</returns>
        private static PolyNumber ArrayBinary(BinaryOp op, in PolyNumber a, in PolyNumber b, NPTypeCode t)
        {
            NDArray x = a.AsArrayOf(t), y = b.AsArrayOf(t);
            using var plain = NDMatrixBoundary.PlainFor(x, y);   // the package computes with plain arrays: no matrix rule in the ufuncs below
            if (op == BinaryOp.Multiply && t == NPTypeCode.Complex && a.IsNdArray && b.IsNdArray
                && x.ndim != y.ndim && x.size == 1 && y.size == 1)
            {
                // NpyIter's single-element loop (output stride 0): CDOUBLE_multiply's contracted loop_scalar.
                var shape = Shape.Adopt(x.ndim > y.ndim ? x.Shape.dimensions : y.Shape.dimensions);
                var r = new NDArray(NPTypeCode.Complex, shape, false);
                Complex p = ILKernelGenerator.PolyLoopScalarComplexMultiply(x.GetAtIndex<Complex>(0), y.GetAtIndex<Complex>(0));
                r.SetAtIndex(p, 0);
                return FromArray(r);
            }
            return FromArray(HouseUfunc(op, x, y));
        }

        /// <summary>
        ///     This value as a house ufunc receives it from Python with no conversion of its own: a Python value as the 0-d
        ///     array its C# literal converts to, marked as a Python literal (<see cref="NDArray.FromPythonOperand"/> — NumPy's
        ///     <c>NPY_ARRAY_WAS_PYTHON_LITERAL</c>), a NumPy scalar as a 0-d array of its dtype, an ndarray as itself.
        /// </summary>
        /// <returns>The operand.</returns>
        /// <exception cref="NotSupportedException">A Python int beyond uint64 (NumPy's object array).</exception>
        private NDArray AsUfuncOperand()
        {
            if (IsPython)
            {
                if (!TryDiscoveredDtype(out _))
                    throw ObjectArrayRefusal();
                return NDArray.FromPythonOperand(ToObject());
            }
            return ToNDArray();
        }

        /// <summary>The house ufunc of a binary operator: <c>np.add</c>, <c>np.subtract</c>, <c>np.multiply</c> or
        ///     <c>np.true_divide</c> (Python's <c>/</c>).</summary>
        /// <param name="op">Add, Subtract, Multiply or Divide.</param>
        /// <param name="x">Left operand.</param>
        /// <param name="y">Right operand.</param>
        /// <returns>The ufunc's result.</returns>
        /// <exception cref="NotSupportedException">Another operator.</exception>
        private static NDArray HouseUfunc(BinaryOp op, NDArray x, NDArray y) => op switch
        {
            BinaryOp.Add => np.add(x, y),
            BinaryOp.Subtract => np.subtract(x, y),
            BinaryOp.Multiply => np.multiply(x, y),
            BinaryOp.Divide => np.true_divide(x, y),
            _ => throw new NotSupportedException(op.ToString()),
        };

        /// <summary>
        ///     <c>-a</c>: CPython negation for a Python value (an int/bool gives an int), NumPy's
        ///     <c>np.negative</c> otherwise (unsigned integers wrap). A NumPy bool refuses.
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <returns>The negation, of the operand's kind.</returns>
        /// <exception cref="TypeError">A NumPy bool scalar or array (NumPy's boolean-negative TypeError).</exception>
        public static PolyNumber Negate(in PolyNumber a)
        {
            if (a.IsPython)
                return FromPython(PyScalar.Negate(a.Py));
            if (a.IsStringArray)
                return FromArray(np.negative(a.Array));   // NumPy's no-loop TypeError for the string dtype, any rank
            if (a.Dtype == NPTypeCode.Boolean)
                throw new TypeError("The numpy boolean negative, the `-` operator, is not supported, use the `~` operator " +
                                    "or the logical_not function instead.");
            if (a.IsNdArray)
                return FromArray(np.negative(a.Array));
            PolyRaw16 ra, rr;
            a.WriteAs(a.Dtype, &ra);
            DirectILKernelGenerator.GetPolyScalarNegateKernel(a.Dtype)((byte*)&ra, (byte*)&rr);
            // A sign flip is exact in any precision: a complex64 scalar negates to a complex64 scalar.
            return a.IsComplex64 ? FromComplex64(&rr) : FromScalar(a.Dtype, &rr);
        }

        /// <summary>
        ///     <c>a ** 2</c> as Python evaluates it — the power with an exact Python int 2, which the weight functions
        ///     write as <c>x**2</c>. A Python value squares with CPython's arithmetic (<see cref="PyScalar.PowTwo"/>). An
        ///     ndarray, 0-d included, takes <c>ndarray.__pow__</c>'s <c>fast_scalar_power</c> (<c>number.c</c>): an exact
        ///     int 2 calls <c>np.square</c> for EVERY dtype, so a bool array squares to int8 and an integer array wraps
        ///     (<c>np.power(bool_array, 2)</c> would be int64), and a 0-d operand's result is a NumPy scalar. A NumPy
        ///     scalar runs scalarmath's power in the NEP 50 dtype of the scalar and the weak int 2.
        /// </summary>
        /// <param name="a">The base.</param>
        /// <returns><c>a ** 2</c>: a Python value for a Python base, an ndarray for an ndarray of rank ≥ 1, a NumPy
        ///     scalar otherwise.</returns>
        /// <exception cref="OverflowException">CPython's float or complex overflow (<see cref="PyScalar.PowTwo"/>).</exception>
        /// <exception cref="NotSupportedException">A complex64 NumPy scalar (its power runs NumPy's complex64 loop,
        ///     which no caller reaches: only an ufunc result can be one, and every weight squares its argument first).</exception>
        /// <remarks>
        ///     <para>The NumPy-scalar branch is scalarmath's <c>@name@_power</c> (<c>scalarmath.c.src</c>) with the exponent
        ///     2 converted into the NEP 50 dtype t:</para>
        ///     <list type="bullet">
        ///         <item><description>float64: <c>npy_pow(x, 2.0)</c>, the C runtime's <c>pow</c> (<see cref="Math.Pow"/>) — NOT
        ///         the product: ucrtbase's <c>pow</c> is not the correctly rounded square (measured: 5,182 of 20M random
        ///         doubles differ from <c>x*x</c>; NumPy's <c>np.float64(2.3315311434106746e+132)**2</c> is <c>pow</c>'s);</description></item>
        ///         <item><description>float32 and float16: <c>npy_powf</c> of the (widened) value (<see cref="MathF.Pow"/>),
        ///         rounded once into t — float16 is <c>half_ctype_power</c>. <c>powf</c> flushes a square below float32's normal
        ///         range to zero where the product rounds to a subnormal (<c>np.float32(2.9465066e-23)**2</c> is 0, its
        ///         <c>np.square</c> 1e-45: 33,574 of 20M random floats differ); for float16 the two agree on every bit
        ///         pattern;</description></item>
        ///         <item><description>complex128: <c>npy_cpow</c>, which returns <c>(+0, +0)</c> for a zero base (either sign of
        ///         either part) and unrolls the exponent 2 into the naive <c>cmul(a, a)</c> — scalarmath's product;</description></item>
        ///         <item><description>the integers (char too) and decimal: <c>@name@_ctype_power</c> squares by repeated
        ///         multiplication in the dtype — the wrapped product. A bool scalar has no scalarmath power: NumPy's generic
        ///         path runs the int64 loop, which the NEP 50 dtype (int64 for a bool and a weak int) gives.</description></item>
        ///     </list>
        /// </remarks>
        public static PolyNumber PowTwo(in PolyNumber a)
        {
            if (a.IsPython)
                return FromPython(PyScalar.PowTwo(a.Py));
            if (a.Kind == PolyNumberKind.Array)
            {
                // fast_scalar_power: the exponent 2 maps to np.square for every array dtype (gh-29388).
                NDArray squared = np.square(a.Array);
                return a.IsNdArray ? FromArray(squared) : ScalarOf(squared);
            }
            if (a.IsComplex64)
                throw new NotSupportedException("a complex64 scalar's power runs NumPy's complex64 loop, which NumSharp does not have (#569)");

            var two = FromPython(PyScalar.Int(2));
            NPTypeCode t = LoopDtype(BinaryOp.Multiply, a, two);
            PolyRaw16 raw, rr;
            a.WriteAs(t, &raw);
            switch (t)
            {
                case NPTypeCode.Double:
                {
                    // npy_pow: the C runtime's pow, not the product (remarks).
                    double r = Math.Pow(*(double*)&raw, 2.0);
                    return FromScalar(NPTypeCode.Double, &r);
                }
                case NPTypeCode.Single:
                case NPTypeCode.Half:
                {
                    // npy_powf of the widened value, rounded once into t by the house cast (npy_float_to_half for float16).
                    float f;
                    PolyConstPool.ConvertBuffer(&raw, t, &f, NPTypeCode.Single, 1);
                    float r = MathF.Pow(f, 2f);
                    PolyConstPool.ConvertBuffer(&r, NPTypeCode.Single, &rr, t, 1);
                    return FromScalar(t, &rr);
                }
                case NPTypeCode.Complex:
                {
                    // npy_cpow: a zero base with a positive real exponent is (+0, +0); otherwise cmul(a, a).
                    Complex z = *(Complex*)&raw;
                    if (z.Real == 0.0 && z.Imaginary == 0.0)
                    {
                        Complex zero = default;
                        return FromScalar(NPTypeCode.Complex, &zero);
                    }
                    break;
                }
            }
            var converted = FromScalar(t, &raw);
            return Binary(BinaryOp.Multiply, converted, converted);
        }

        /// <summary>
        ///     A one-input FLOAT-producing ufunc (<c>np.sqrt</c>, <c>np.exp</c>) applied to this value as NumPy applies it to
        ///     whatever Python passes. An ndarray of rank ≥ 1 goes to the house ufunc and gives an ndarray. Anything
        ///     scalar-sized — a Python value (converted by its discovered dtype: a bool to bool, an int to int64 or uint64, a
        ///     float to float64, a complex to complex128), a NumPy scalar or a 0-d array — runs the loop on one element and
        ///     gives a NumPy SCALAR, because NumPy unwraps a 0-d ufunc result.
        /// </summary>
        /// <param name="op">The ufunc's house operation (<see cref="UnaryOp.Sqrt"/>, <see cref="UnaryOp.Exp"/>, …).</param>
        /// <param name="name">The ufunc's NumPy name, which the object-loop TypeError quotes.</param>
        /// <param name="arrayUfunc">The house ufunc over an NDArray, for an ndarray operand (a static lambda: no closure).</param>
        /// <param name="a">The operand.</param>
        /// <returns>An ndarray for an ndarray operand of rank ≥ 1, a NumPy scalar otherwise.</returns>
        /// <exception cref="TypeError">A Python int beyond int64 and uint64. NumPy converts it to an OBJECT array, whose
        ///     loop calls the element's method of the ufunc's name: <c>loop of ufunc does not support argument 0 of type
        ///     int which has no callable exp method</c>. Or a string operand (the house ufunc's no-loop error).</exception>
        /// <exception cref="NotSupportedException">A complex64 NumPy scalar (NumPy's complex64 loop; see
        ///     <see cref="PowTwo"/>).</exception>
        /// <remarks>
        ///     A scalar-sized operand runs the house's contiguous whole-array kernel of <paramref name="op"/> over ONE element
        ///     held on the stack: the kernel <c>np.sqrt</c> / <c>np.exp</c> run, in the loop dtype NumPy picks
        ///     (<see cref="FloatLoop"/>), without the two 0-d arrays a ufunc call on a scalar costs — the operand and the
        ///     result, ~0.3 µs each, which made a scalar weight several times slower than NumPy's Python arithmetic. With
        ///     IL generation disabled (NativeAOT) the operand goes through the NDArray ufunc instead, the same bits slower.
        /// </remarks>
        public static PolyNumber FloatUfunc(UnaryOp op, string name, Func<NDArray, NDArray> arrayUfunc, in PolyNumber a)
        {
            if (a.IsNdArray || a.IsStringArray)
                return a.IsNdArray ? FromArray(arrayUfunc(a.Array)) : ScalarOf(arrayUfunc(a.Array));
            NPTypeCode tin;
            if (a.IsPython)
            {
                if (!a.TryDiscoveredDtype(out tin))
                    throw new TypeError($"loop of ufunc does not support argument 0 of type {a.Py.TypeName} which has no callable {name} method");
            }
            else
            {
                if (a.IsComplex64)
                    throw new NotSupportedException($"a complex64 scalar's {name} runs NumPy's complex64 loop, which NumSharp does not have (#569)");
                tin = a.Dtype;   // a NumPy scalar or a 0-d array
            }

            if (!DirectILKernelGenerator.Enabled)
                return ScalarOf(arrayUfunc(a.ToNDArray()));
            NPTypeCode tout = FloatLoop(tin);
            var kernel = DirectILKernelGenerator.GetUnaryKernel(new UnaryKernelKey(tin, tout, op, IsContiguous: true));
            PolyRaw16 src, dst;
            a.WriteAs(tin, &src);   // the value in its own dtype (a Python value: its discovered dtype, exactly)
            long one = 1;
            kernel(&src, &dst, &one, &one, 1, 1);
            return FromScalar(tout, &dst);
        }

        /// <summary>
        ///     The loop dtype NumPy picks for a float-producing unary ufunc (<c>sqrt</c>, <c>exp</c>, the trigonometric and
        ///     logarithmic families) with no <c>dtype=</c>: the narrowest float holding the input's integers — bool, int8 and
        ///     uint8 run float16; int16, uint16 (and NumSharp's char, a uint16) float32; the 32- and 64-bit integers float64 —
        ///     and every float, complex and decimal dtype its own. The house engine's rule
        ///     (<c>DefaultEngine.ResolveUnaryFloatReturnType</c>), restated over a dtype code for operands that are not arrays;
        ///     a unit test holds the two equal for every dtype.
        /// </summary>
        /// <param name="t">The input dtype.</param>
        /// <returns>The loop (and result) dtype.</returns>
        public static NPTypeCode FloatLoop(NPTypeCode t) => t switch
        {
            NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte => NPTypeCode.Half,
            NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char => NPTypeCode.Single,
            NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 => NPTypeCode.Double,
            _ => t,
        };

        /// <summary>
        ///     Python's <c>bool(a != 0)</c> — the test <c>{p}line</c> branches on. An ndarray answers through its
        ///     elementwise comparison, whose truth value NumPy defines only for exactly one element.
        /// </summary>
        /// <returns>Whether the value is nonzero (NaN is nonzero).</returns>
        /// <exception cref="ValueError">An ndarray with several elements, or none (NumPy's two truth-value texts).</exception>
        public bool NotZero()
        {
            switch (Kind)
            {
                case PolyNumberKind.Python:
                    return Py.IsNonZero;
                case PolyNumberKind.Scalar:
                {
                    PolyRaw16 raw = _raw;
                    return DirectILKernelGenerator.GetPolyScalarPredicateKernel(Dtype, PolyZeroTest.NotEqual)((byte*)&raw);
                }
                default:
                {
                    var one = SingleElementOf(Array);
                    PolyRaw16 raw;
                    PolyConstPool.ConvertBuffer(one, Dtype, &raw, Dtype, 1);
                    return DirectILKernelGenerator.GetPolyScalarPredicateKernel(Dtype, PolyZeroTest.NotEqual)((byte*)&raw);
                }
            }
        }

        /// <summary>
        ///     Python's <c>bool(a &lt; 0)</c> — trimcoef's tolerance check. A NumPy complex scalar orders
        ///     lexicographically (scalarmath); a Python complex has no order.
        /// </summary>
        /// <returns>Whether the value is negative.</returns>
        /// <exception cref="TypeError">A Python complex.</exception>
        /// <exception cref="ValueError">An ndarray with several elements, or none.</exception>
        public bool LessZero()
        {
            switch (Kind)
            {
                case PolyNumberKind.Python:
                    return Py.IsNegative();
                case PolyNumberKind.Scalar:
                {
                    PolyRaw16 raw = _raw;
                    return DirectILKernelGenerator.GetPolyScalarPredicateKernel(Dtype, PolyZeroTest.Less)((byte*)&raw);
                }
                default:
                {
                    var one = SingleElementOf(Array);
                    PolyRaw16 raw;
                    PolyConstPool.ConvertBuffer(one, Dtype, &raw, Dtype, 1);
                    return DirectILKernelGenerator.GetPolyScalarPredicateKernel(Dtype, PolyZeroTest.Less)((byte*)&raw);
                }
            }
        }

        /// <summary>
        ///     The address of an ndarray's only element, for NumPy's truth value of a comparison result — defined
        ///     for one element only.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <returns>The element's address.</returns>
        /// <exception cref="ValueError">Zero elements (<c>The truth value of an empty array is ambiguous…</c>) or
        ///     several (<c>The truth value of an array with more than one element is ambiguous…</c>).</exception>
        private static void* SingleElementOf(NDArray a)
        {
            CheckSingleElement(a);
            // One element: its offset is the shape's offset (every stride multiplies a zero index).
            return (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;
        }

        /// <summary>
        ///     NumPy's precondition for an array's truth value: exactly one element. Also the whole truth-value test of
        ///     <c>{p}line</c>'s text scale, an all-True comparison result whose truth value only its size decides (a
        ///     <c>StringDType</c> array has no fixed element size, so no element address is computed here).
        /// </summary>
        /// <param name="a">The array.</param>
        /// <exception cref="ValueError">Zero elements (<c>The truth value of an empty array is ambiguous…</c>) or
        ///     several (<c>The truth value of an array with more than one element is ambiguous…</c>).</exception>
        internal static void CheckSingleElement(NDArray a)
        {
            if (a.size == 0)
                throw new ValueError("The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.");
            if (a.size > 1)
                throw new ValueError("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
        }

        // ---------------------------------------------------------------------------------------------
        //  Egress
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     This value as a C# object: a Python float / complex / int / bool as the matching boxed primitive
        ///     (<c>double</c>, <see cref="Complex"/>, <c>long</c> — <c>ulong</c> or <see cref="BigInteger"/> when it
        ///     does not fit — and <c>bool</c>); a NumPy scalar as its dtype's boxed C# value (<c>double</c> for
        ///     float64, <c>float</c> for float32, <see cref="Half"/> for float16, …; a complex64 scalar as the
        ///     <see cref="Complex"/> holding its exact value — NumSharp's one complex type); an ndarray as the NDArray.
        /// </summary>
        /// <returns>The boxed value or the array.</returns>
        public object ToObject()
        {
            switch (Kind)
            {
                case PolyNumberKind.Python:
                    if (Py.IsBool) return !Py.I.IsZero;
                    return Py.Kind switch
                    {
                        PyKind.Float => Py.F,
                        PyKind.Complex => Py.C,
                        _ => Py.I >= s_longMin && Py.I <= s_longMax ? (object)(long)Py.I
                           : Py.I.Sign >= 0 && Py.I <= s_ulongMax ? (ulong)Py.I : (object)Py.I,
                    };
                case PolyNumberKind.Scalar:
                {
                    // The dtype picks the boxed CLR type (an emitted ldobj + box, no per-dtype C#).
                    PolyRaw16 raw = _raw;
                    return DirectILKernelGenerator.GetPolyScalarBoxKernel(Dtype)((byte*)&raw);
                }
                default:
                    return Array;
            }
        }

        /// <summary>
        ///     This value as an NDArray: a 0-d array of <see cref="DiscoveredDtype"/> for a Python value, a 0-d
        ///     array of its dtype for a NumPy scalar, the array itself for an ndarray.
        /// </summary>
        /// <returns>The array.</returns>
        /// <exception cref="NotSupportedException">A Python int beyond uint64.</exception>
        public NDArray ToNDArray()
        {
            if (Kind == PolyNumberKind.Array)
                return Array;
            NPTypeCode t = DiscoveredDtype();
            var r = new NDArray(t, Shape.Scalar, false);
            WriteElement(t, (byte*)r.Storage.Address + r.Shape.offset * r.dtypesize);
            return r;
        }

        /// <summary>
        ///     Writes this scalar-sized value as an element of an array being CREATED with dtype
        ///     <paramref name="t"/> (np.array's setitem): a Python value by its discovered dtype's exact value —
        ///     an int as int64/uint64, never range-checked against <paramref name="t"/> (array creation widened
        ///     <paramref name="t"/> to hold it) — then cast; a NumPy scalar or 0-d array cast.
        /// </summary>
        /// <param name="t">The array's dtype.</param>
        /// <param name="dst">The element's address.</param>
        /// <exception cref="NotSupportedException">A Python int beyond uint64.</exception>
        public void WriteElement(NPTypeCode t, void* dst)
        {
            if (!IsPython)
            {
                WriteAs(t, dst);
                return;
            }
            switch (DiscoveredDtype())
            {
                case NPTypeCode.Boolean:
                {
                    bool b = !Py.I.IsZero;
                    PolyConstPool.ConvertBuffer(&b, NPTypeCode.Boolean, dst, t, 1);
                    return;
                }
                case NPTypeCode.Int64:
                {
                    long v = (long)Py.I;
                    PolyConstPool.ConvertBuffer(&v, NPTypeCode.Int64, dst, t, 1);
                    return;
                }
                case NPTypeCode.UInt64:
                {
                    ulong v = (ulong)Py.I;
                    PolyConstPool.ConvertBuffer(&v, NPTypeCode.UInt64, dst, t, 1);
                    return;
                }
                case NPTypeCode.Double:
                {
                    double v = Py.F;
                    PolyConstPool.ConvertBuffer(&v, NPTypeCode.Double, dst, t, 1);
                    return;
                }
                default:
                {
                    Complex v = Py.C;
                    PolyConstPool.ConvertBuffer(&v, NPTypeCode.Complex, dst, t, 1);
                    return;
                }
            }
        }

        /// <summary>
        ///     <c>np.array([items…])</c>: dtype discovery over the items (<see cref="DiscoveredDtype"/>, promoted
        ///     with <c>np.promote_types</c>), then either a 1-D array of the scalar-sized items or — when every item
        ///     is an ndarray of one shape — their stack along a new leading axis. Any other mix is NumPy's
        ///     inhomogeneous-shape ValueError, whose "detected shape" is the longest common shape prefix.
        /// </summary>
        /// <param name="items">The list elements (at least one).</param>
        /// <returns>The new, owning, C-contiguous array.</returns>
        /// <exception cref="ValueError">The items' shapes are inhomogeneous.</exception>
        /// <exception cref="NotSupportedException">A Python int beyond uint64 (NumPy's object array).</exception>
        public static NDArray MakeArray(params PolyNumber[] items) => MakeArray((ReadOnlySpan<PolyNumber>)items);

        /// <summary><c>np.array([a])</c> (see <see cref="MakeArray(ReadOnlySpan{PolyNumber})"/>), with no list allocation.</summary>
        /// <param name="a">The element.</param>
        /// <returns>The new array.</returns>
        /// <exception cref="NotSupportedException">A Python int beyond uint64.</exception>
        public static NDArray MakeArray(in PolyNumber a) => MakeArray(new ReadOnlySpan<PolyNumber>(in a));

        /// <summary><c>np.array([a, b])</c> (see <see cref="MakeArray(ReadOnlySpan{PolyNumber})"/>), with no list allocation —
        ///     the <c>{p}line</c> case.</summary>
        /// <param name="a">First element.</param>
        /// <param name="b">Second element.</param>
        /// <returns>The new array.</returns>
        /// <exception cref="ValueError">Inhomogeneous shapes.</exception>
        /// <exception cref="NotSupportedException">A Python int beyond uint64.</exception>
        public static NDArray MakeArray(in PolyNumber a, in PolyNumber b)
        {
            var pair = new PolyNumberPair();
            pair[0] = a;
            pair[1] = b;
            return MakeArray((ReadOnlySpan<PolyNumber>)pair);
        }

        /// <summary>
        ///     <c>np.array([items…])</c> over a span (the list itself need not exist): see
        ///     <see cref="MakeArray(PolyNumber[])"/>.
        /// </summary>
        /// <param name="items">The list elements (at least one).</param>
        /// <returns>The new, owning, C-contiguous array.</returns>
        /// <exception cref="ValueError">The items' shapes are inhomogeneous.</exception>
        /// <exception cref="NotSupportedException">A Python int beyond uint64 (NumPy's object array).</exception>
        public static NDArray MakeArray(ReadOnlySpan<PolyNumber> items)
        {
            NPTypeCode t = items[0].DiscoveredDtype();
            for (int i = 1; i < items.Length; i++)
                t = PolyTyping.Promote(t, items[i].DiscoveredDtype());

            bool anyNd = false;
            foreach (var it in items)
                anyNd |= it.IsNdArray;

            if (!anyNd)
            {
                var r = new NDArray(t, new Shape((long)items.Length), false);
                int size = r.dtypesize;
                byte* p = (byte*)r.Storage.Address + r.Shape.offset * size;
                for (int i = 0; i < items.Length; i++)
                    items[i].WriteElement(t, p + (long)i * size);
                return r;
            }

            // Every item must be an ndarray of ONE shape (a 0-d array is shape (), a scalar too).
            long[] first = ShapeOf(items[0]);
            for (int i = 1; i < items.Length; i++)
            {
                long[] s = ShapeOf(items[i]);
                if (!SameDims(first, s))
                    throw InhomogeneousError(items);
            }
            var dims = new long[first.Length + 1];
            dims[0] = items.Length;
            System.Array.Copy(first, 0, dims, 1, first.Length);
            var stacked = new NDArray(t, Shape.Adopt(dims), false);
            for (int i = 0; i < items.Length; i++)
            {
                var src = items[i].Array;
                var dst = stacked[i];   // row view (writes through)
                np.copyto(dst, src.typecode == t ? src : src.astype(t));
            }
            return stacked;
        }

        /// <summary>The shape an item contributes to array discovery (a scalar-sized item is <c>()</c>).</summary>
        /// <param name="p">The item.</param>
        /// <returns>Its dims.</returns>
        private static long[] ShapeOf(in PolyNumber p)
            => p.Kind == PolyNumberKind.Array ? p.Array.Shape.dimensions : System.Array.Empty<long>();

        /// <summary>Elementwise dims equality.</summary>
        /// <param name="a">First dims.</param><param name="b">Second dims.</param>
        /// <returns>True when equal.</returns>
        private static bool SameDims(long[] a, long[] b)
        {
            if (a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i]) return false;
            return true;
        }

        /// <summary>
        ///     NumPy's array-coercion error for items whose shapes disagree: the "detected shape" is the list
        ///     length followed by the longest dims prefix every item shares (probed 2.4.2:
        ///     <c>[zeros((2,3)), zeros((2,4))]</c> → <c>(2, 2)</c>, <c>[zeros((2,3)), 1.0]</c> → <c>(2,)</c>).
        /// </summary>
        /// <param name="items">The list elements.</param>
        /// <returns>The exception to throw.</returns>
        private static ValueError InhomogeneousError(ReadOnlySpan<PolyNumber> items)
        {
            long[] common = ShapeOf(items[0]);
            int len = common.Length;
            for (int i = 1; i < items.Length; i++)
            {
                long[] s = ShapeOf(items[i]);
                int k = 0;
                while (k < len && k < s.Length && s[k] == common[k]) k++;
                len = k;
            }
            var detected = new long[len + 1];
            detected[0] = items.Length;
            System.Array.Copy(common, 0, detected, 1, len);
            string text = detected.Length == 1 ? $"({detected[0]},)" : "(" + string.Join(", ", detected) + ")";
            return new ValueError("setting an array element with a sequence. The requested array has an inhomogeneous shape " +
                                  $"after {detected.Length} dimensions. The detected shape was {text} + inhomogeneous part.");
        }
    }
}
