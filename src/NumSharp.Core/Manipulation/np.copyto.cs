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
        ///     <see cref="copyto{T}(NDArray, T, string, NDArray)"/> instead, which is NumPy's Python scalar (weak). A null
        ///     (<c>(NDArray)null</c>) is Python's <c>None</c>: NumPy's 0-d OBJECT array, castable only under
        ///     <c>casting='unsafe'</c>, where it stores NaN / <c>nan+nanj</c> / <c>False</c> / <c>'None'</c> / a
        ///     <c>StringDType</c> NA object, or raises <c>int(None)</c>'s TypeError for an integer destination.</param>
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
        ///     under the house type of every cast check (astype, concatenate, NDIter); a <c>None</c> source under any rule but
        ///     <c>'unsafe'</c> ("Cannot cast scalar from dtype('O') to …").</exception>
        /// <exception cref="TypeError">A <c>None</c> source (<paramref name="src"/> null) under <c>casting='unsafe'</c> into
        ///     an integer or <c>decimal</c> destination with an element to convert: Python's <c>int(None)</c>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="dst"/> is <c>null</c>.</exception>
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

            // array_copyto converts the mask (PyArray_FromAny with a bool descr: the 'safe' rule) before
            // PyArray_AssignArray, so a bad mask wins over a read-only destination and over a bad cast.
            DefaultEngine.ValidateWhereMask(@where);

            // A null source is Python's None, which PyArray_FromAny makes a 0-d OBJECT array (CopytoNone).
            if (src is null)
            {
                UnmanagedStorage.CopytoNone(dst, @where, rule);
                return;
            }

            UnmanagedStorage.CopytoAssignArray(dst, src, @where, rule);
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
                UnmanagedStorage.CopytoAssignRawScalar(dst, slot, InfoOf<T>.NPTypeCode, @where, rule);
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

            UnmanagedStorage.CopytoPythonScalar(dst, src, kind, rule, @where);
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
            UnmanagedStorage.CopytoPythonScalar(dst, src, PyScalarKind.Int, rule, @where);
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
    }
}
