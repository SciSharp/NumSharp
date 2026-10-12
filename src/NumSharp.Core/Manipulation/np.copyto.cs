using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Copies values from one array to another, broadcasting as necessary: NumPy's <c>np.copyto(dst, src,
        ///     casting='same_kind', where=True)</c>, a statement-for-statement port of <c>array_copyto</c> →
        ///     <c>PyArray_AssignArray</c> (<c>numpy/_core/src/multiarray/array_assign_array.c</c>).
        /// </summary>
        /// <param name="dst">The array into which values are copied (written in place).</param>
        /// <param name="src">The array from which values are copied. It broadcasts to <paramref name="dst"/>'s shape;
        ///     leading axes of extent 1 beyond <paramref name="dst"/>'s rank are stripped first (NumPy's backward
        ///     compatibility rule), so a <c>(1, 3)</c> source fills a <c>(3,)</c> destination. A 0-d array is a SCALAR
        ///     source (NumPy's <c>PyArray_AssignRawScalar</c>): cast once, then written to every selected element. An
        ///     <see cref="NDArray"/> is always NumPy's ndarray (STRONG); a C# number binds
        ///     <see cref="copyto{T}(NDArray, T, string, NDArray)"/> instead, which is NumPy's Python scalar (weak).</param>
        /// <param name="casting">Which casts from <paramref name="src"/>'s dtype to <paramref name="dst"/>'s are allowed:
        ///     <c>"no"</c>, <c>"equiv"</c>, <c>"safe"</c>, <c>"same_kind"</c> (the default) or <c>"unsafe"</c>.</param>
        /// <param name="where">A bool mask broadcast to <paramref name="dst"/>'s shape: elements are written only where it
        ///     is true. <c>null</c> (the default) is NumPy's <c>where=True</c>. A 0-d mask is NumPy's scalar optimization:
        ///     true copies everything, false copies nothing (after the source shape has been checked).</param>
        /// <exception cref="TypeError"><paramref name="casting"/> is <c>null</c>: "casting must be str, not NoneType".</exception>
        /// <exception cref="ValueError"><paramref name="casting"/> is not one of the five names ("casting must be one of
        ///     'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '…')"); <paramref name="dst"/> is read-only
        ///     ("assignment destination is read-only"); <paramref name="src"/> does not broadcast to
        ///     <paramref name="dst"/> ("could not broadcast input array from shape (4,) into shape (3,)"); or
        ///     <paramref name="where"/> does not ("could not broadcast where mask from shape (2,) into shape (3,)") —
        ///     NumPy's ValueError texts.</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array: "Cannot cast array data from
        ///     dtype('int64') to dtype('bool') according to the rule 'safe'" ("scalar" for a 0-d mask) — NumPy's TypeError
        ///     text, under the house type every ufunc's <c>where=</c> check raises.</exception>
        /// <exception cref="InvalidCastException">The cast from <paramref name="src"/>'s dtype to <paramref name="dst"/>'s is
        ///     not allowed under <paramref name="casting"/>: "Cannot cast array data from dtype('float64') to dtype('int32')
        ///     according to the rule 'same_kind'" ("Cannot cast scalar from …" for a 0-d source) — NumPy's TypeError text,
        ///     under the house type of every cast check (astype, concatenate, NDIter).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="dst"/> or <paramref name="src"/> is <c>null</c>.</exception>
        /// <remarks>
        ///     <para><b>NumPy's checks, in NumPy's order</b> (each probed on 2.4.2): the casting name, the where mask's
        ///     dtype, then — for an array source — a redundant self-copy returns at once (the same data, dtype, shape and
        ///     strides: <c>np.copyto(a, a)</c> is legal even when <c>a</c> is read-only), the read-only destination, the
        ///     cast rule, the source's broadcast, the 0-d mask, the mask's broadcast; for a 0-d source: the read-only
        ///     destination, the cast rule, the mask's broadcast.</para>
        ///     <para><b>Overlap.</b> A source that may share memory with the destination is copied first, as NumPy does,
        ///     unless the destination is 1-D with both runs in the same direction; then the copy walks in NumPy's raw order
        ///     (a negative destination stride flipped; backward when the source starts below the destination inside it), so
        ///     overlapping views with different strides give NumPy's bits (<c>np.copyto(a[3:19:2], a[6:14])</c> reads
        ///     elements it has already written, in NumPy and here).</para>
        ///     <para><b>Kernels.</b> The masked copy is a bit copy of the destination's element width over NumPy's prepared
        ///     iteration (<see cref="DirectILKernelGenerator.TryGetMaskedAssignKernel"/>): a SIMD blend for contiguous
        ///     runs, a hoisted value for a broadcast source, a branch-free select otherwise. A source of another dtype is
        ///     cast into the destination's dtype first (one element per distinct source element, laid out in the
        ///     destination's memory order). The unmasked copy is <see cref="NDIter.Copy(UnmanagedStorage, UnmanagedStorage)"/>.</para>
        ///     <para><b>Operand roots.</b> <see cref="NDAnchoredAttribute"/>: the kernels read raw addresses and the copy runs
        ///     storage to storage, so a caller's temporary <paramref name="src"/>, <paramref name="where"/> or even
        ///     <paramref name="dst"/> (<c>np.copyto(np.zeros(n), x)</c>) would otherwise be collectable mid-copy and its
        ///     buffer pooled under the kernel. The weaver roots all three until the call returns; no scope is opened (this
        ///     is a hot internal primitive, and it builds no array it reads after its last use).</para>
        /// </remarks>
        [NDAnchored]
        public static void copyto(NDArray dst, NDArray src, string casting = "same_kind", NDArray @where = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(dst, src, @where);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // npy_parse_arguments runs PyArray_CastingConverter before anything reads dst or src.
            NPY_CASTING rule = CopytoCastingConverter(casting);

            if (dst is null)
                throw new ArgumentNullException(nameof(dst));
            if (src is null)
                throw new ArgumentNullException(nameof(src));

            // array_copyto converts the mask (PyArray_FromAny with a bool descr: the 'safe' rule) before
            // PyArray_AssignArray, so a bad mask wins over a read-only destination and over a bad cast.
            DefaultEngine.ValidateWhereMask(@where);

            CopytoAssignArray(dst, src, @where, rule);
        }

        /// <summary>
        ///     <c>np.copyto(dst, src)</c> with a C# scalar source — NumPy's Python scalar. A C# integer, <c>float</c>,
        ///     <c>double</c> or <see cref="Complex"/> is WEAK (NEP 50): it takes the destination's dtype within its kind, so
        ///     <c>np.copyto(uint8_array, 0)</c> and <c>np.copyto(float32_array, 0.1, casting: "safe")</c> are legal, and an
        ///     integer that does not fit the destination raises instead of wrapping. A <c>bool</c>, <see cref="Half"/>,
        ///     <c>char</c> or <c>decimal</c> is STRONG (a Python bool, a NumPy scalar): a 0-d array of its own dtype.
        /// </summary>
        /// <typeparam name="T">The C# scalar type. Pass an <see cref="NDArray"/> (0-d or not) for NumPy's ndarray
        ///     semantics: <c>NDArray.Scalar(1000L)</c> into int8 wraps to -24 like <c>np.array(1000)</c>, while the literal
        ///     <c>1000</c> raises like the Python int.</typeparam>
        /// <param name="dst">The array into which the value is copied.</param>
        /// <param name="src">The value.</param>
        /// <param name="casting">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</param>
        /// <param name="where">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</param>
        /// <exception cref="OverflowException">A weak integer does not fit an integer destination: "Python integer 1000 out
        ///     of bounds for int8" (NumPy's OverflowError, raised before the mask is checked); a <c>ulong</c> past
        ///     <see cref="long.MaxValue"/> into an integer other than uint32/uint64: "Python int too large to convert to C
        ///     long" (the LP64 text NumSharp pins).</exception>
        /// <exception cref="TypeError"><paramref name="casting"/> is <c>"equiv"</c> and the destination's dtype is not the
        ///     scalar's own default (int64, float64, complex128): "cannot cast Python int to int32 under the casting rule
        ///     'equiv'" — NumPy's text; or <paramref name="casting"/> is <c>null</c>.</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is an unmanaged type with no NumPy counterpart
        ///     (for example <see cref="DateTime"/>).</exception>
        /// <exception cref="ValueError">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</exception>
        /// <exception cref="ArgumentException">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</exception>
        /// <exception cref="InvalidCastException">The scalar's resolved dtype cannot be cast to the destination's under
        ///     <paramref name="casting"/>: a float into an integer destination ("Cannot cast scalar from dtype('float64') to
        ///     dtype('int32') according to the rule 'same_kind'"), an int into a bool destination (dtype('int64')), a complex
        ///     into a real one.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="dst"/> is <c>null</c>.</exception>
        /// <remarks>
        ///     <para>NumPy's <c>array_copyto</c>: <c>npy_mark_tmp_array_if_pyscalar</c> marks the scalar weak,
        ///     <c>npy_find_descr_for_scalar</c> picks the dtype it is stored in — the common DType of the scalar's abstract
        ///     class (<c>_PyLongDType</c>, <c>_PyFloatDType</c>, <c>_PyComplexDType</c>) and the destination's class: the
        ///     destination's own dtype for a number of a kind it can hold, int64 for an int into bool, float64 for a float
        ///     into an integer or bool — and <c>npy_update_operand_for_scalar</c> stores it there with the element setter
        ///     (<c>PyArray_Pack</c>: range-checked, never wrapped). The 0-d value is then assigned like any 0-d source.</para>
        ///     <para>NumSharp has one complex width, so a complex into a float32 destination is resolved (and reported) as
        ///     complex128 where NumPy says complex64.</para>
        ///     <para><b>Operand roots.</b> <see cref="NDAnchoredAttribute"/>, as on
        ///     <see cref="copyto(NDArray, NDArray, string, NDArray)"/>: the destination and the mask stay reachable while
        ///     the fill kernel writes and reads them, even when the caller passed them as temporaries.</para>
        /// </remarks>
        [NDAnchored]
        public static unsafe void copyto<T>(NDArray dst, T src, string casting = "same_kind", NDArray @where = null)
            where T : unmanaged
        {
            using var plain = NDMatrixBoundary.PlainFor(dst, @where);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            NPY_CASTING rule = CopytoCastingConverter(casting);
            if (dst is null)
                throw new ArgumentNullException(nameof(dst));

            // A C# char into a fixed-width string array is one character of TEXT (setitem reads it the same way), not
            // NumSharp's numeric Char dtype: NumPy's np.copyto(u, 'x').
            if (typeof(T) == typeof(char) && dst.typecode == NPTypeCode.String)
            {
                copyto(dst, ((char)(object)src).ToString(), casting, @where);
                return;
            }

            if (typeof(T) == typeof(bool) || typeof(T) == typeof(Half) || typeof(T) == typeof(char) || typeof(T) == typeof(decimal))
            {
                // A Python bool is not marked weak (npy_mark_tmp_array_if_pyscalar requires an integer array), and Half /
                // char / decimal are NumPy scalars: each is a 0-d array of its own dtype, assigned like one.
                byte* slot = stackalloc byte[16];
                *(T*)slot = src;
                DefaultEngine.ValidateWhereMask(@where);
                CopytoAssignRawScalar(dst, slot, InfoOf<T>.NPTypeCode, @where, rule);
                return;
            }

            PyScalarKind kind;
            if (typeof(T) == typeof(sbyte) || typeof(T) == typeof(byte) || typeof(T) == typeof(short) || typeof(T) == typeof(ushort)
                || typeof(T) == typeof(int) || typeof(T) == typeof(uint) || typeof(T) == typeof(long) || typeof(T) == typeof(ulong))
                kind = PyScalarKind.Int;
            else if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
                kind = PyScalarKind.Float;
            else if (typeof(T) == typeof(Complex))
                kind = PyScalarKind.Complex;
            else
                throw new NotSupportedException($"np.copyto: a {typeof(T).Name} source has no NumPy dtype.");

            CopytoPythonScalar(dst, src, kind, rule, @where);
        }

        /// <summary>
        ///     <c>np.copyto(dst, src)</c> with an arbitrary-size integer source — NumPy's Python int of any magnitude
        ///     (weak, range-checked against the destination). See <see cref="copyto{T}(NDArray, T, string, NDArray)"/>.
        /// </summary>
        /// <param name="dst">The array into which the value is copied.</param>
        /// <param name="src">The integer.</param>
        /// <param name="casting">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</param>
        /// <param name="where">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</param>
        /// <exception cref="OverflowException">The integer does not fit an integer destination (NumPy's OverflowError texts),
        ///     or is past the double range for a float destination ("int too large to convert to float").</exception>
        /// <exception cref="TypeError">See <see cref="copyto{T}(NDArray, T, string, NDArray)"/>.</exception>
        /// <exception cref="ValueError">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</exception>
        /// <exception cref="ArgumentException">See <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</exception>
        /// <exception cref="InvalidCastException">See <see cref="copyto{T}(NDArray, T, string, NDArray)"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="dst"/> is <c>null</c>.</exception>
        /// <remarks><see cref="NDAnchoredAttribute"/>: the destination and the mask stay reachable while the fill kernel
        ///     runs, as on <see cref="copyto(NDArray, NDArray, string, NDArray)"/>.</remarks>
        [NDAnchored]
        public static void copyto(NDArray dst, BigInteger src, string casting = "same_kind", NDArray @where = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(dst, @where);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            NPY_CASTING rule = CopytoCastingConverter(casting);
            if (dst is null)
                throw new ArgumentNullException(nameof(dst));
            CopytoPythonScalar(dst, src, PyScalarKind.Int, rule, @where);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_CastingConverter</c> for <c>np.copyto</c>'s <c>casting</c> argument.
        /// </summary>
        /// <param name="casting">The caller's casting name.</param>
        /// <returns>The casting level.</returns>
        /// <exception cref="TypeError"><paramref name="casting"/> is <c>null</c> ("casting must be str, not NoneType").</exception>
        /// <exception cref="ValueError">An unknown name (NumPy's "casting must be one of …" text).</exception>
        private static NPY_CASTING CopytoCastingConverter(string casting)
        {
            if (casting is null)
                throw new TypeError("casting must be str, not NoneType");
            return DTypeCasting.ParseCasting(casting);
        }

        /// <summary>
        ///     The weak-scalar half of <c>array_copyto</c>: resolve the dtype NumPy stores a Python scalar in, store it
        ///     there (range-checked), check the mask, then assign the 0-d value.
        /// </summary>
        /// <param name="dst">The destination (not <c>null</c>).</param>
        /// <param name="value">The boxed C# number (an integer primitive, <see cref="BigInteger"/>, <c>float</c>,
        ///     <c>double</c> or <see cref="Complex"/>).</param>
        /// <param name="kind">Which Python scalar it stands for.</param>
        /// <param name="casting">The casting level.</param>
        /// <param name="where">The caller's mask, or <c>null</c>.</param>
        /// <exception cref="TypeError">The 'equiv' rule with a dtype other than the scalar's default.</exception>
        /// <exception cref="OverflowException">An integer that does not fit an integer destination.</exception>
        private static unsafe void CopytoPythonScalar(NDArray dst, object value, PyScalarKind kind, NPY_CASTING casting, NDArray @where)
        {
            // The dtype PyArray_FromAny gives the scalar on its own: int64 for an int (uint64 past int64, an object
            // array — no NumSharp dtype, NPTypeCode.Empty here — past uint64), float64, complex128.
            NPTypeCode original = kind switch
            {
                PyScalarKind.Float => NPTypeCode.Double,
                PyScalarKind.Complex => NPTypeCode.Complex,
                _ => PyIntDefaultType(value),
            };

            NPTypeCode descr = CopytoFindDescrForScalar(original, kind, dst.typecode);

            // npy_update_operand_for_scalar: a dtype other than the scalar's own cannot be reached under 'equiv'.
            if (casting == NPY_CASTING.NPY_EQUIV_CASTING && descr != original)
            {
                string pyType = kind == PyScalarKind.Int ? "int" : kind == PyScalarKind.Float ? "float" : "complex";
                throw new TypeError($"cannot cast Python {pyType} to {descr.AsNumpyDtypeName()} under the casting rule 'equiv'");
            }

            // PyArray_SETITEM into a new 0-d array of that dtype: NumPy's range checks and roundings (PyArray_Pack).
            byte* slot = stackalloc byte[ReductionInitial.SlotBytes];
            ReductionInitial.TryPack(value, descr, new Span<byte>(slot, ReductionInitial.SlotBytes));

            DefaultEngine.ValidateWhereMask(@where);
            CopytoAssignRawScalar(dst, slot, descr, @where, casting);
        }

        /// <summary>
        ///     The element a text scalar becomes in a numeric array through the legacy text cast (each text parsed by the
        ///     target's <c>int()</c> / <c>float()</c> / <c>complex()</c>; bool reads the text's truth), written into
        ///     <paramref name="element"/>.
        /// </summary>
        /// <param name="text">A 0-d text array (not modified).</param>
        /// <param name="dstType">The numeric target dtype.</param>
        /// <param name="element">Receives the element; at least <paramref name="dstType"/>'s item size.</param>
        /// <exception cref="ValueError">The text does not parse into <paramref name="dstType"/>.</exception>
        /// <exception cref="OverflowException">An integer text outside an integer <paramref name="dstType"/>.</exception>
        private static unsafe void CopytoParseTextScalar(NDArray text, NPTypeCode dstType, Span<byte> element)
        {
            // One element: the array cast's own parse (its NumPy-scalar error texts), then the element copied out.
            using var parsed = text.astype((DType)dstType, true, 'K', "unsafe");
            int size = dstType.SizeOf();
            fixed (byte* e = element)
                Buffer.MemoryCopy((byte*)parsed.Address + parsed.Shape.offset * size, e, element.Length, size);
        }

        /// <summary>
        ///     The one element <c>np.copyto(dst, value, casting='unsafe')</c> stores for a SCALAR <paramref name="value"/> —
        ///     the second statement of NumPy's <c>np.full</c> (<c>a = empty(shape, dtype); copyto(a, fill_value,
        ///     casting='unsafe')</c>) and <c>np.full_like</c>, which fill every element with it.
        /// </summary>
        /// <param name="value">The fill value: a boxed C# number, a C# literal's 0-d array (<see cref="NDArray.WasPythonLiteral"/>)
        ///     or another 0-d array.</param>
        /// <param name="dstType">The new array's dtype (numeric).</param>
        /// <param name="element">Receives the element; at least <see cref="ReductionInitial.SlotBytes"/> bytes.</param>
        /// <returns><c>false</c> when <paramref name="value"/> is not a scalar this rule covers (an array with dimensions, a
        ///     list): the caller keeps its own conversion.</returns>
        /// <exception cref="OverflowException">A Python int out of an integer dtype's range ("Python integer 1000 out of
        ///     bounds for int8"; the LP64 texts past the C <c>long</c>), or past the double range into a float; an integer
        ///     text outside an integer dtype.</exception>
        /// <exception cref="ValueError">A text (a C# string or <c>byte[]</c>, or a 0-d text array) that does not parse into
        ///     <paramref name="dstType"/> (NumPy's <c>invalid literal for int() with base 10: np.str_('x')</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="element"/> is smaller than
        ///     <see cref="ReductionInitial.SlotBytes"/> (a caller bug).</exception>
        /// <remarks>
        ///     <para>
        ///     <c>casting='unsafe'</c> lets every cast through, but a Python scalar is still PACKED first: its stored dtype
        ///     is <c>npy_find_descr_for_scalar</c>'s (<see cref="CopytoFindDescrForScalar"/>), so a Python int into an
        ///     integer array is range-checked in that array's own dtype (<c>np.full(3, 1000, dtype=np.int8)</c> raises),
        ///     while a Python float into an integer array is stored as float64 and then CAST
        ///     (<c>np.full(3, 1e300, dtype=np.int8)</c> is NumPy's unsafe float64 → int8 value, no error), and a Python
        ///     complex into a real array keeps its real part. Probed against NumPy 2.4.2.
        ///     </para>
        ///     <para>
        ///     A Python bool, a NumPy scalar (<see cref="Half"/>, <c>char</c>, <c>decimal</c>) and an unmarked 0-d array are
        ///     strong: one unsafe cast from their own dtype.
        ///     </para>
        /// </remarks>
        internal static unsafe bool TryCopytoUnsafeScalarElement(object value, NPTypeCode dstType, Span<byte> element)
        {
            if (element.Length < ReductionInitial.SlotBytes)
                throw new ArgumentException($"The element slot needs {ReductionInitial.SlotBytes} bytes.", nameof(element));

            // A C# literal's 0-d array stands for the Python scalar it was made from.
            if (value is NDArray marked && marked.WasPythonLiteral)
                value = marked.Storage.GetAtIndex(0);

            PyScalarKind kind;
            NPTypeCode strongType;
            byte* strong = stackalloc byte[ReductionInitial.SlotBytes];
            switch (value)
            {
                case sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger:
                    kind = PyScalarKind.Int;
                    break;
                case float or double:
                    kind = PyScalarKind.Float;
                    break;
                case Complex:
                    kind = PyScalarKind.Complex;
                    break;
                case bool b:
                    *(bool*)strong = b;
                    strongType = NPTypeCode.Boolean;
                    goto Strong;
                case Half h:
                    *(Half*)strong = h;
                    strongType = NPTypeCode.Half;
                    goto Strong;
                case char c:
                    *(char*)strong = c;
                    strongType = NPTypeCode.Char;
                    goto Strong;
                case decimal m:
                    *(decimal*)strong = m;
                    strongType = NPTypeCode.Decimal;
                    goto Strong;
                case NDArray nd when nd.ndim == 0 && nd.typecode != NPTypeCode.String:
                    // A 0-d ndarray: PyArray_AssignArray of a 0-d source, one unsafe cast. Its one element sits at the
                    // shape's offset (Address + offset · itemsize covers a contiguous slice and a strided view alike).
                    fixed (byte* e = element)
                        NDIterCasting.ConvertValue((byte*)nd.Address + nd.Shape.offset * nd.dtypesize, e, nd.typecode, dstType);
                    return true;
                case string or byte[]:
                    // A Python str / bytes: array_copyto converts it with np.asarray (a 0-d U / S array — text is never a
                    // weak scalar), then the legacy text cast parses it; its errors quote the NumPy scalar
                    // (np.full(2, 'x', dtype=int) raises "invalid literal for int() with base 10: np.str_('x')", probed).
                    using (var text = value is string s ? NDArray.Scalar(s) : NDArray.Scalar((byte[])value))
                        CopytoParseTextScalar(text, dstType, element);
                    return true;
                case NDArray nd when nd.ndim == 0:
                    // A 0-d text array: the same array cast.
                    CopytoParseTextScalar(nd, dstType, element);
                    return true;
                default:
                    return false;
            }

            // The weak scalar: stored in npy_find_descr_for_scalar's dtype with PyArray_Pack's checks, then cast.
            NPTypeCode original = kind switch
            {
                PyScalarKind.Float => NPTypeCode.Double,
                PyScalarKind.Complex => NPTypeCode.Complex,
                _ => PyIntDefaultType(value),
            };
            NPTypeCode descr = CopytoFindDescrForScalar(original, kind, dstType);
            byte* packed = stackalloc byte[ReductionInitial.SlotBytes];
            ReductionInitial.TryPack(value, descr, new Span<byte>(packed, ReductionInitial.SlotBytes));
            fixed (byte* e = element)
                NDIterCasting.ConvertValue(packed, e, descr, dstType);
            return true;

        Strong:
            fixed (byte* e = element)
                NDIterCasting.ConvertValue(strong, e, strongType, dstType);
            return true;
        }

        /// <summary>The dtype <c>np.asarray</c> gives a Python int: int64, uint64 past int64, else none (an object array).</summary>
        /// <param name="value">The boxed integer.</param>
        /// <returns><see cref="NPTypeCode.Int64"/>, <see cref="NPTypeCode.UInt64"/>, or <see cref="NPTypeCode.Empty"/>
        ///     (NumPy's object dtype, which NumSharp lacks).</returns>
        private static NPTypeCode PyIntDefaultType(object value)
        {
            switch (value)
            {
                case ulong u:
                    return u > long.MaxValue ? NPTypeCode.UInt64 : NPTypeCode.Int64;
                case BigInteger b:
                    if (b >= long.MinValue && b <= long.MaxValue)
                        return NPTypeCode.Int64;
                    if (b.Sign > 0 && b <= ulong.MaxValue)
                        return NPTypeCode.UInt64;
                    return NPTypeCode.Empty;
                default:
                    return NPTypeCode.Int64;
            }
        }

        /// <summary>
        ///     NumPy's <c>npy_find_descr_for_scalar</c>: the dtype a weak Python scalar is stored in before it is assigned
        ///     to a destination of <paramref name="dstType"/>.
        /// </summary>
        /// <param name="original">The scalar's own dtype (<see cref="PyIntDefaultType"/>, float64, complex128).</param>
        /// <param name="kind">The scalar's Python kind.</param>
        /// <param name="dstType">The destination's dtype.</param>
        /// <returns>The original dtype when it already is the destination's class or the common class; otherwise the
        ///     common DType's default descriptor (the destination's dtype for a scalar of a kind it can hold).</returns>
        private static NPTypeCode CopytoFindDescrForScalar(NPTypeCode original, PyScalarKind kind, NPTypeCode dstType)
        {
            // "There is a good chance, descriptors already match..."
            if (original != NPTypeCode.Empty && original == dstType)
                return original;

            DTypeMeta py = kind switch
            {
                PyScalarKind.Float => DTypeRegistry.PyFloat,
                PyScalarKind.Complex => DTypeRegistry.PyComplex,
                _ => DTypeRegistry.PyLong,
            };

            DTypeMeta common;
            try
            {
                common = DTypePromotion.CommonDType(py, DTypeRegistry.FromTypeCode(dstType));
            }
            catch (DTypePromotionError)
            {
                // "This is fine. We simply assume the original descr is viable." (no NumSharp dtype reaches here)
                return original == NPTypeCode.Empty ? NPTypeCode.Int64 : original;
            }

            if (original != NPTypeCode.Empty && common.TypeCode == original)
                return original;
            return common.DefaultDescr().Meta.TypeCode;
        }

        /// <summary>
        ///     <c>PyArray_AssignArray</c>: an array source, NumPy's checks in NumPy's order, then the copy.
        /// </summary>
        /// <param name="dst">The destination.</param>
        /// <param name="src">The source (any rank; 0-d goes to <see cref="CopytoAssignRawScalar"/>).</param>
        /// <param name="where">The validated bool mask, or <c>null</c>.</param>
        /// <param name="casting">The casting level.</param>
        /// <exception cref="ValueError">A read-only destination, or a source / mask that does not broadcast.</exception>
        /// <exception cref="InvalidCastException">The cast is not allowed under <paramref name="casting"/>.</exception>
        [SkipLocalsInit]
        private static unsafe void CopytoAssignArray(NDArray dst, NDArray src, NDArray @where, NPY_CASTING casting)
        {
            // Fixed-width strings: the cast rules are per DESCRIPTOR (the width is part of the dtype) and the copy is the
            // string route — the same checks in NumPy's order, in CopytoAssignStrings.
            if (src.typecode == NPTypeCode.String || dst.typecode == NPTypeCode.String)
            {
                CopytoAssignStrings(dst, src, @where, casting);
                return;
            }

            // "Use array_assign_scalar if 'src' NDIM is 0"
            if (src.ndim == 0)
            {
                CopytoAssignRawScalar(dst, CopytoData(src), src.typecode, @where, casting);
                return;
            }

            // A redundant self-copy (the same data, dtype, shape and strides) returns BEFORE the read-only check, as in
            // NumPy: `a[1000:6000] += x` assigns a view to an identical view.
            if (CopytoData(src) == CopytoData(dst) && src.typecode == dst.typecode && SameLayout(src.Shape, dst.Shape))
                return;

            NumSharpException.ThrowIfNotWriteable(dst.Shape);

            if (!NDIterCasting.CanCast(src.typecode, dst.typecode, casting))
                throw new InvalidCastException(
                    $"Cannot cast array data from dtype('{src.typecode.AsNumpyDtypeName()}') " +
                    $"to dtype('{dst.typecode.AsNumpyDtypeName()}') according to the rule '{DTypeCasting.CastingToString(casting)}'");

            // An overlapping source is copied first unless dst is 1-D with both runs in the same direction (NumPy's rule;
            // the bounds test is NumPy's arrays_overlap).
            int dnd = dst.ndim;
            bool overlapNeedsCopy =
                ((dnd == 1 && dst.Shape.strides[0] * src.Shape.strides[src.ndim - 1] < 0) || dnd > 1)
                && NDMemOverlap.SolveMayShareMemory(src, dst, 0) != MemOverlap.No;

            // Broadcast 'src' to 'dst' for raw iteration (leading unit axes stripped first).
            long[] srcStrides = CopytoBroadcastStrides(dst.Shape.dimensions, src.Shape.dimensions, src.Shape.strides,
                "input array", stripLeadingUnitAxes: true);

            // "optimization: scalar boolean mask"
            if (@where is not null && @where.ndim == 0)
            {
                if (!*(bool*)CopytoData(@where))
                    return;
                @where = null;
            }

            if (@where is null)
            {
                // raw_array_assign_array: NDIter.Copy is the shared port (its overlap rule is NumPy's). It broadcasts
                // the source itself, so only a source with more axes than dst needs its stripped view.
                UnmanagedStorage source = src.ndim > dnd ? StrippedSourceView(src, dnd) : src.Storage;
                NDIter.Copy(dst.Storage, source);
                return;
            }

            long[] maskStrides = CopytoBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides,
                "where mask", stripLeadingUnitAxes: false);

            if (dst.size == 0)
                return;

            // The masked kernel copies bits of dst's width, so a source of another dtype — and an overlapping source that
            // NumPy would copy first — is cast/copied into dst's dtype first. A 1-D same-direction overlap of the same
            // dtype stays in place: the kernel walks it in NumPy's order.
            if (!overlapNeedsCopy && src.typecode == dst.typecode)
            {
                CopytoMaskedRaw(dst, CopytoData(src), srcStrides, CopytoData(@where), maskStrides);
                return;
            }

            // The staged copy holds one element per distinct source element. A small one lives on the stack ([SkipLocalsInit]
            // keeps it unzeroed; every element is written before the masked copy reads it), a larger one is a pooled
            // buffer (warm on the next call, and never a raw allocation — NativeAllocationChokepointTests pins those). No
            // NDArray and no iterator state: at 1K elements those cost more than the cast itself (0.54 of 0.78 µs, measured).
            long count = 1;
            for (int d = 0; d < dnd; d++)
                count *= srcStrides[d] == 0 ? 1 : dst.Shape.dimensions[d];
            long bytes = count * dst.dtypesize;
            IntPtr pooled = IntPtr.Zero;
            byte* buffer;
            if (bytes <= CopytoStageStackBytes)
            {
                byte* onStack = stackalloc byte[(int)bytes];
                buffer = onStack;
            }
            else
            {
                pooled = SizeBucketedBufferPool.Take(bytes);
                buffer = (byte*)pooled;
            }

            try
            {
                CopytoStageInto(dst, src, srcStrides, buffer, count);
                CopytoMaskedRaw(dst, buffer, srcStrides, CopytoData(@where), maskStrides);
            }
            finally
            {
                // Returned on every exit; a cast never throws (NumPy's unsafe element casts), but the masked copy's
                // kernel lookup could.
                if (pooled != IntPtr.Zero)
                    SizeBucketedBufferPool.Return(pooled, bytes);
            }
        }

        /// <summary>
        ///     The largest staged source (bytes) kept on the stack: 16 KB fits any thread's stack with room to spare, and
        ///     covers a 1K-element copy of every dtype (the size where a pool round trip would be a visible share of the call).
        /// </summary>
        private const int CopytoStageStackBytes = 16 * 1024;

        /// <summary>
        ///     <c>PyArray_AssignRawScalar</c>: a 0-d source, cast once into the destination's dtype, written to every
        ///     element (or every element the mask selects).
        /// </summary>
        /// <param name="dst">The destination.</param>
        /// <param name="srcData">The source element.</param>
        /// <param name="srcType">The source element's dtype.</param>
        /// <param name="where">The validated bool mask, or <c>null</c>.</param>
        /// <param name="casting">The casting level.</param>
        /// <exception cref="ValueError">A read-only destination, or a mask that does not broadcast.</exception>
        /// <exception cref="InvalidCastException">The cast is not allowed ("Cannot cast scalar from …").</exception>
        /// <remarks>A string destination receives the number's NumPy scalar <c>str()</c>, truncated to its width
        ///     (<see cref="NDTextConversion.FormatRun"/>).</remarks>
        private static unsafe void CopytoAssignRawScalar(NDArray dst, byte* srcData, NPTypeCode srcType, NDArray @where, NPY_CASTING casting)
        {
            NumSharpException.ThrowIfNotWriteable(dst.Shape);

            // A number into a fixed-width string array: NumPy's descriptor rule decides first (int64 → <U5 is 'same_kind',
            // so 'safe' raises its TypeError text); an allowed cast formats the number ONCE into one element of the
            // destination's dtype (its scalar's str(), truncated to the width — formatting cannot fail, so doing it before
            // the mask's broadcast check is unobservable), and the masked raw walk copies that element's bytes.
            if (dst.typecode == NPTypeCode.String)
            {
                DType from = srcType;
                if (!DTypeCasting.CanCastTypeTo(from, dst.dtype, casting))
                    throw NDStringCasting.InvalidCast(from, dst.dtype, casting, scalar: true);
                long[] textMaskStrides = null;
                if (@where is not null)
                    textMaskStrides = CopytoBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides,
                        "where mask", stripLeadingUnitAxes: false);
                if (dst.size == 0 || (@where is not null && !np.any(@where)))
                    return;
                var slice = dst.Storage.StringSlice;
                byte[] element = new byte[Math.Max(slice.ItemSize, 1)];
                fixed (byte* text = element)
                {
                    NDTextConversion.FormatRun(srcData, 0, srcType, text, slice.ItemSize, slice.IsUnicode, slice.Chars, 1);
                    long[] zeros = new long[dst.ndim];
                    byte selected = 1;
                    if (@where is null)
                        CopytoMaskedRaw(dst, text, zeros, &selected, zeros);
                    else
                        CopytoMaskedRaw(dst, text, zeros, CopytoData(@where), textMaskStrides);
                }
                return;
            }

            if (!NDIterCasting.CanCast(srcType, dst.typecode, casting))
                throw new InvalidCastException(
                    $"Cannot cast scalar from dtype('{srcType.AsNumpyDtypeName()}') " +
                    $"to dtype('{dst.typecode.AsNumpyDtypeName()}') according to the rule '{DTypeCasting.CastingToString(casting)}'");

            int nd = dst.ndim;
            long[] maskStrides = null;
            if (@where is not null)
                maskStrides = CopytoBroadcastStrides(dst.Shape.dimensions, @where.Shape.dimensions, @where.Shape.strides,
                    "where mask", stripLeadingUnitAxes: false);

            if (dst.size == 0)
                return;

            // "Make a copy of the src data if it's a different dtype than 'dst'": the value is cast ONCE (the same bits a
            // per-element cast would give), and the kernel then copies dst-width bits.
            byte* value = stackalloc byte[16];
            NDIterCasting.ConvertValue(srcData, value, srcType, dst.typecode);

            long[] zero = new long[nd];
            byte trueByte = 1;
            if (@where is null)
            {
                CopytoMaskedRaw(dst, value, zero, &trueByte, zero);
                return;
            }

            CopytoMaskedRaw(dst, value, zero, CopytoData(@where), maskStrides);
        }

        /// <summary>
        ///     NumPy's <c>raw_array_wheremasked_assign_array</c>: the prepared raw iteration of (dst, src, mask) and the
        ///     masked bit copy of dst's element width.
        /// </summary>
        /// <param name="dst">The destination (non-empty, writeable).</param>
        /// <param name="srcData">The source's first element, already in dst's dtype.</param>
        /// <param name="srcStrides">The source's ELEMENT strides broadcast to dst's axes (0 where it broadcasts).</param>
        /// <param name="maskData">The mask's first byte.</param>
        /// <param name="maskStrides">The mask's strides broadcast to dst's axes (a bool is one byte: element = byte).</param>
        /// <remarks>
        ///     A constant true mask (<c>maskStrides</c> all 0 over a true byte) makes this NumPy's unmasked scalar fill.
        /// </remarks>
        private static unsafe void CopytoMaskedRaw(NDArray dst, byte* srcData, long[] srcStrides, byte* maskData, long[] maskStrides)
        {
            int nd = dst.ndim;
            int itemsize = dst.dtypesize;
            int slots = nd < 1 ? 1 : nd;

            long* shape = stackalloc long[slots];
            long* sa = stackalloc long[slots];
            long* sb = stackalloc long[slots];
            long* sc = stackalloc long[slots];
            long[] dims = dst.Shape.dimensions, dstStrides = dst.Shape.strides;
            for (int d = 0; d < nd; d++)
            {
                shape[d] = dims[d];
                sa[d] = dstStrides[d] * itemsize;
                sb[d] = srcStrides[d] * itemsize;
                sc[d] = maskStrides[d];
            }

            byte* dstData = CopytoData(dst);
            long* oShape = stackalloc long[slots];
            long* oa = stackalloc long[slots];
            long* ob = stackalloc long[slots];
            long* oc = stackalloc long[slots];
            int pnd = RawArrayIter.PrepareThree(nd, shape, ref dstData, sa, ref srcData, sb, ref maskData, sc, oShape, oa, ob, oc);
            if (oShape[0] == 0)
                return;

            // With more than one prepared axis the operands cannot overlap (an overlapping source of a multi-axis
            // destination was copied first), so the visiting order is not observable: put innermost the axis along which
            // the most operands are contiguous, so a run is a SIMD blend instead of a strided select (an F destination
            // over C operands is otherwise thousands of 2-element runs). NumPy's own choice, dst's smallest stride, wins
            // ties.
            if (pnd > 1 && !CopytoNumPyOrderOnly)
            {
                int best = 0, bestCost = CopytoRunCost(oa[0], ob[0], oc[0], itemsize);
                for (int k = 1; k < pnd; k++)
                {
                    int cost = CopytoRunCost(oa[k], ob[k], oc[k], itemsize);
                    if (cost < bestCost)
                    {
                        best = k;
                        bestCost = cost;
                    }
                }

                if (best != 0)
                {
                    long ts = oShape[best], ta = oa[best], tb = ob[best], tc = oc[best];
                    for (int k = best; k > 0; k--)
                    {
                        oShape[k] = oShape[k - 1];
                        oa[k] = oa[k - 1];
                        ob[k] = ob[k - 1];
                        oc[k] = oc[k - 1];
                    }
                    oShape[0] = ts;
                    oa[0] = ta;
                    ob[0] = tb;
                    oc[0] = tc;
                }
            }

            // "Overlap check for the 1D case": walk from the end when the source starts below the destination and runs
            // into it (higher-dimensional overlaps were copied first).
            if (pnd == 1 && srcData < dstData && srcData + oShape[0] * ob[0] > dstData)
            {
                long last = oShape[0] - 1;
                srcData += last * ob[0];
                dstData += last * oa[0];
                maskData += last * oc[0];
                ob[0] = -ob[0];
                oa[0] = -oa[0];
                oc[0] = -oc[0];
            }

            // The kernel takes the axes in C order (the run last): NumPy's prepared order is innermost first, and its
            // raw loop moves axis 1 fastest, so the reversal reproduces the visiting order.
            long* kShape = stackalloc long[pnd];
            long* ks = stackalloc long[pnd];
            long* kd = stackalloc long[pnd];
            long* km = stackalloc long[pnd];
            for (int k = 0; k < pnd; k++)
            {
                int p = pnd - 1 - k;
                kShape[k] = oShape[p];
                ks[k] = ob[p];
                kd[k] = oa[p];
                km[k] = oc[p];
            }

            // Any element width: a fixed-width string element (S3, U3) runs as lanes of a kernel width.
            DirectILKernelGenerator.MaskedAssignAnyWidth(itemsize, srcData, dstData, maskData, ks, kd, km, kShape, pnd);
        }

        /// <summary>
        ///     Test and benchmark hook: when set, the masked copy walks NumPy's raw order (the destination's smallest
        ///     stride innermost) even where another inner axis would keep more operands contiguous. The bits are the same
        ///     either way (only non-overlapping operands are reordered); only the speed differs.
        /// </summary>
        internal static bool CopytoNumPyOrderOnly;

        /// <summary>
        ///     How many operands a run along one prepared axis would read or write with a stride: the destination and the
        ///     source unless their stride is the element size (a source stride of 0 is a hoisted broadcast, also free),
        ///     the mask unless its stride is 1 or 0.
        /// </summary>
        /// <param name="dstStride">The destination's byte stride along the axis.</param>
        /// <param name="srcStride">The source's byte stride along the axis.</param>
        /// <param name="maskStride">The mask's byte stride along the axis.</param>
        /// <param name="itemsize">The element size.</param>
        /// <returns>0 to 3: the number of strided operands.</returns>
        private static int CopytoRunCost(long dstStride, long srcStride, long maskStride, int itemsize)
            => (dstStride != itemsize ? 1 : 0)
               + (srcStride != itemsize && srcStride != 0 ? 1 : 0)
               + (maskStride != 1 && maskStride != 0 ? 1 : 0);

        /// <summary>
        ///     NumPy's <c>broadcast_strides</c>: the strides an operand of <paramref name="dims"/> /
        ///     <paramref name="strides"/> walks <paramref name="dstDims"/> with (0 on every axis it broadcasts along).
        /// </summary>
        /// <param name="dstDims">The destination's dimensions.</param>
        /// <param name="dims">The operand's dimensions.</param>
        /// <param name="strides">The operand's ELEMENT strides.</param>
        /// <param name="name">NumPy's operand name in the error: <c>"input array"</c> or <c>"where mask"</c>.</param>
        /// <param name="stripLeadingUnitAxes">Whether leading extent-1 axes beyond the destination's rank are dropped first
        ///     (PyArray_AssignArray does it for the source, "as a special case for backwards compatibility"; never for the
        ///     mask).</param>
        /// <returns>A fresh array of <paramref name="dstDims"/>.Length element strides.</returns>
        /// <exception cref="ValueError">"could not broadcast {name} from shape (…) into shape (…)" — the operand's shape
        ///     after the stripping, as NumPy prints it.</exception>
        private static long[] CopytoBroadcastStrides(long[] dstDims, long[] dims, long[] strides, string name, bool stripLeadingUnitAxes)
        {
            int ndim = dstDims.Length;
            int skip = 0;
            int sndim = dims.Length;
            if (stripLeadingUnitAxes)
            {
                while (sndim - skip > ndim && dims[skip] == 1)
                    skip++;
            }

            int ondim = sndim - skip;
            int start = ndim - ondim;
            var result = new long[ndim];
            if (start < 0)
                throw CopytoBroadcastError(name, dims, skip, dstDims);

            // Process from the end to the start (NumPy's loop); a missing leading axis broadcasts with stride 0.
            for (int d = ndim - 1; d >= start; d--)
            {
                long extent = dims[skip + d - start];
                if (extent == 1)
                    result[d] = 0;
                else if (extent != dstDims[d])
                    throw CopytoBroadcastError(name, dims, skip, dstDims);
                else
                    result[d] = strides[skip + d - start];
            }

            return result;
        }

        /// <summary>NumPy's broadcast error text (<c>convert_shape_to_string</c>: <c>(4,)</c>, <c>(2,3)</c>, <c>()</c>).</summary>
        /// <param name="name">"input array" or "where mask".</param>
        /// <param name="dims">The operand's dimensions.</param>
        /// <param name="skip">How many leading axes were stripped (they are not printed).</param>
        /// <param name="dstDims">The destination's dimensions.</param>
        /// <returns>The ValueError to throw.</returns>
        private static ValueError CopytoBroadcastError(string name, long[] dims, int skip, long[] dstDims)
            => new ValueError($"could not broadcast {name} from shape {PyShape(dims, skip)} into shape {PyShape(dstDims, 0)}");

        /// <summary>A Python shape tuple without spaces: <c>()</c>, <c>(4,)</c>, <c>(2,3)</c>.</summary>
        /// <param name="dims">The dimensions.</param>
        /// <param name="skip">Leading entries to leave out.</param>
        /// <returns>The tuple text.</returns>
        private static string PyShape(long[] dims, int skip)
        {
            int n = dims.Length - skip;
            if (n == 0)
                return "()";
            var sb = new System.Text.StringBuilder("(");
            for (int d = skip; d < dims.Length; d++)
            {
                if (d > skip)
                    sb.Append(',');
                sb.Append(dims[d]);
            }
            if (n == 1)
                sb.Append(',');
            return sb.Append(')').ToString();
        }

        /// <summary>
        ///     The source with its leading extent-1 axes beyond <paramref name="ndim"/> dropped — a view (no copy) for
        ///     <see cref="NDIter.Copy(UnmanagedStorage, UnmanagedStorage)"/>, which broadcasts sources of at most the
        ///     destination's rank. The broadcast was already validated, so the stripped rank never exceeds
        ///     <paramref name="ndim"/>.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <param name="ndim">The destination's rank.</param>
        /// <returns>An alias of the source's storage (no counted reference: the caller's <paramref name="src"/> keeps the
        ///     buffer alive for the copy).</returns>
        private static UnmanagedStorage StrippedSourceView(NDArray src, int ndim)
        {
            long[] dims = src.Shape.dimensions, strides = src.Shape.strides;
            int skip = dims.Length - ndim;
            var nd = new long[ndim];
            var ns = new long[ndim];
            Array.Copy(dims, skip, nd, 0, ndim);
            Array.Copy(strides, skip, ns, 0, ndim);
            var shape = new Shape(nd, ns, src.Shape.offset, src.Shape.bufferSize);
            return src.Storage.Alias(ref shape);
        }

        /// <summary>
        ///     The source copied (and cast) into a raw buffer of dst's dtype: one element per DISTINCT source element (an
        ///     axis along which the source does not move — it broadcasts there, or is itself a stride-0 view — keeps one
        ///     element), laid out densely in dst's memory order (NumPy's <c>PyArray_NewLikeArray(dst, NPY_KEEPORDER)</c>
        ///     temporary), so the masked kernel's runs stay contiguous for both. Rewrites <paramref name="srcStrides"/> to the
        ///     buffer's element strides.
        /// </summary>
        /// <param name="dst">The destination: its dimensions and strides give the layout; its data is not touched.</param>
        /// <param name="src">The source (validated to broadcast to dst).</param>
        /// <param name="srcStrides">In: the source's element strides broadcast to dst's axes (0 where the source does not
        ///     move). Out: the buffer's element strides on the same axes (still 0 on those axes).</param>
        /// <param name="buffer">Room for <paramref name="count"/> elements of dst's dtype; every one of them is written.</param>
        /// <param name="count">The number of distinct source elements: the product of dst's extents over the axes along
        ///     which the source moves (at least 1).</param>
        /// <remarks>
        ///     The cast runs through the kernels <see cref="NDIter.Copy(UnmanagedStorage, UnmanagedStorage)"/> picks for a
        ///     cross-dtype copy (contiguous cast kernel, strided cast kernel, then the per-run inner cast), so the staged bits
        ///     are those of the copy it replaces — only the NDArray around the buffer and the iterator state are gone.
        /// </remarks>
        [SkipLocalsInit]
        private static unsafe void CopytoStageInto(NDArray dst, NDArray src, long[] srcStrides, byte* buffer, long count)
        {
            int nd = dst.ndim;
            long[] dims = dst.Shape.dimensions, dstStrides = dst.Shape.strides;
            byte* srcData = CopytoData(src);
            NPTypeCode from = src.typecode, to = dst.typecode;
            int slots = nd < 1 ? 1 : nd;

            // The axes the source moves along, outermost first by dst's |stride| (an insertion sort that only moves past a
            // strictly smaller stride: C order breaks ties, as NPY_KEEPORDER does). A moving axis always has extent > 1:
            // CopytoBroadcastStrides gives an extent-1 axis stride 0.
            int* axes = stackalloc int[slots];
            int k = 0;
            for (int d = 0; d < nd; d++)
            {
                if (srcStrides[d] == 0)
                    continue;
                long key = Math.Abs(dstStrides[d]);
                int j = k++;
                while (j > 0 && Math.Abs(dstStrides[axes[j - 1]]) < key)
                {
                    axes[j] = axes[j - 1];
                    j--;
                }
                axes[j] = d;
            }

            // The copy's axes in that order, with the buffer dense (innermost stride 1).
            long* shape = stackalloc long[slots];
            long* sStr = stackalloc long[slots];
            long* bStr = stackalloc long[slots];
            long dense = 1;
            for (int j = k - 1; j >= 0; j--)
            {
                int d = axes[j];
                shape[j] = dims[d];
                sStr[j] = srcStrides[d];
                bStr[j] = dense;
                dense *= dims[d];
            }

            // The masked copy reads the buffer through dst's axes: the moving axes take the buffer's strides, the others
            // keep 0. Done before the copy below, which no longer needs the caller's array.
            for (int j = 0; j < k; j++)
                srcStrides[axes[j]] = bStr[j];

            // Coalesce adjacent axes the source walks as one run (the buffer is dense, so the source decides), so a
            // C-ordered source over a C-ordered dst is ONE contiguous cast whatever its rank.
            int m = 0;
            for (int j = 0; j < k; j++)
            {
                if (m > 0 && sStr[m - 1] == sStr[j] * shape[j])
                {
                    shape[m - 1] *= shape[j];
                    sStr[m - 1] = sStr[j];
                    bStr[m - 1] = bStr[j];
                }
                else
                {
                    shape[m] = shape[j];
                    sStr[m] = sStr[j];
                    bStr[m] = bStr[j];
                    m++;
                }
            }

            if (from == to)
            {
                // A same-dtype temporary (an overlapping source NumPy copies first) is a raw bit copy, as NumPy's
                // raw_array_assign_array makes it: a bool byte other than 0/1 (a frombuffer view) stays what it is, where
                // a cast would normalize it. One run is a memcpy; otherwise the masked-assign walker under a constant true
                // mask, which copies dst-width bits run by run.
                int itemsize = dst.dtypesize;
                if (m == 0 || (m == 1 && sStr[0] == 1))
                {
                    long bytes = count * itemsize;
                    Buffer.MemoryCopy(srcData, buffer, bytes, bytes);
                    return;
                }

                long* sb = stackalloc long[m];
                long* bb = stackalloc long[m];
                long* mb = stackalloc long[m];
                for (int j = 0; j < m; j++)
                {
                    sb[j] = sStr[j] * itemsize;
                    bb[j] = bStr[j] * itemsize;
                    mb[j] = 0;
                }

                byte trueByte = 1;
                DirectILKernelGenerator.MaskedAssignAnyWidth(itemsize, srcData, buffer, &trueByte, sb, bb, mb, shape, m);
                return;
            }

            // One distinct element (the source moves along no axis): NDIter.Copy's scalar-broadcast conversion.
            if (m == 0)
            {
                NDIterCasting.ConvertValue(srcData, buffer, from, to);
                return;
            }

            // A contiguous run: the contiguous SIMD cast kernel.
            if (m == 1 && sStr[0] == 1)
            {
                var contiguous = DirectILKernelGenerator.TryGetCastKernel(from, to);
                if (contiguous != null)
                {
                    contiguous(srcData, buffer, count);
                    return;
                }
            }

            var strided = DirectILKernelGenerator.TryGetStridedCastKernel(from, to);
            if (strided != null)
            {
                strided(srcData, buffer, sStr, bStr, shape, m);
                return;
            }

            // Pairs without a strided kernel (a bool source, IL generation off): the per-run inner cast, every pair.
            NDIterCasting.CopyStridedToStridedWithCast(srcData, sStr, from, buffer, bStr, to, shape, m, count);
        }

        /// <summary>The address of an array's first element (<c>PyArray_DATA</c>).</summary>
        /// <param name="a">The array.</param>
        /// <returns><c>Storage.Address + offset * itemsize</c>.</returns>
        private static unsafe byte* CopytoData(NDArray a) => (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;

        /// <summary>Whether two shapes have the same rank, dimensions and strides (<c>PyArray_CompareLists</c> twice).</summary>
        /// <param name="a">The first shape.</param>
        /// <param name="b">The second shape.</param>
        /// <returns><c>true</c> when the layouts are identical.</returns>
        private static bool SameLayout(Shape a, Shape b)
        {
            long[] ad = a.dimensions, bd = b.dimensions, @as = a.strides, bs = b.strides;
            if (ad.Length != bd.Length)
                return false;
            for (int d = 0; d < ad.Length; d++)
                if (ad[d] != bd[d] || @as[d] != bs[d])
                    return false;
            return true;
        }
    }
}
