using System;
using System.Collections;
using System.Numerics;
using System.Reflection;
using NumSharp.Backends;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's scalar constructors (<c>scalartypes.c.src</c>: <c>@name@_arrtype_new</c>, <c>bool_arrtype_new</c>), the
    ///     bodies of the class calls in <see cref="NDScalarClassCalls"/> (<c>np.float64(1.5)</c>, <c>np.int8("12")</c>,
    ///     <c>np.complex128(1, 2)</c>, <c>np.str_(5)</c>). Each returns NumSharp's NumPy scalar - a read-only 0-d array made by
    ///     <see cref="NDArray.PyArray_Return"/> - or, where NumPy returns an ndarray (a sequence or an array argument), the
    ///     array.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>NumPy's structure, kept.</b> <c>str_</c> and <c>bytes_</c> first try the Python base class's constructor
    ///     (<c>str(x)</c>, <c>bytes(x)</c>) and fall back to NumPy's conversion only when that raises and exactly one argument
    ///     was given; <c>bool_</c> answers <c>True</c> / <c>False</c> itself; the complex classes (not <c>clongdouble</c>)
    ///     take a second argument, building <c>complex(real, imag)</c> from two real numbers; every other class converts its
    ///     one argument with <c>PyArray_FromAny(x, descr, 0, 0, NPY_ARRAY_FORCECAST)</c> - a C# value packed by the dtype's
    ///     setitem (NEP 50's range check: <c>np.int8(300)</c> raises OverflowError), an array or a NumPy scalar cast
    ///     unsafely (<c>np.int8(np.int64(300))</c> is 44), a sequence converted element by element - and hands a 0-d result
    ///     back as the scalar.
    ///     </para>
    ///     <para>
    ///     <c>float64</c> also tries Python's <c>float(x)</c> first in NumPy. For every value of the C# boundary map that step
    ///     gives the same scalar, or raises the same error, as the conversion it falls back to (<c>float()</c> and the float64
    ///     setitem share CPython's parser and its errors; a 0-d array's <c>__float__</c> is its cast; an array with
    ///     dimensions fails <c>float()</c> and comes back as the array), so it runs the shared conversion directly.
    ///     </para>
    ///     <para>
    ///     <b>No argument versus null.</b> The class calls declare their arguments <c>[Optional] object</c>, so an omitted
    ///     argument arrives as <see cref="Missing.Value"/>: NumPy's no-argument call, the dtype's zero (<c>''</c> / <c>b''</c>
    ///     for the text classes, False for bool). A null is Python's <c>None</c>, a real argument (<c>np.float64(None)</c> is
    ///     nan, <c>np.int8(None)</c> raises TypeError).
    ///     </para>
    /// </remarks>
    internal static class ArrTypeNew
    {
        /// <summary>
        ///     <c>@name@_arrtype_new</c> for a class with one optional argument (the integer, float, <c>clongdouble</c>
        ///     classes): the zero for no argument, else <c>PyArray_FromAny(x, descr, FORCECAST)</c> with a 0-d result returned
        ///     as the scalar.
        /// </summary>
        /// <param name="descr">The class's descriptor (its <see cref="np"/> field: <c>np.int8</c>, <c>np.longdouble</c>).</param>
        /// <param name="x">The argument, or <see cref="Missing.Value"/> when the caller omitted it; null is <c>None</c>.</param>
        /// <returns>The read-only 0-d scalar of <paramref name="descr"/>, or the ndarray a sequence or an array argument
        ///     converts to (the argument itself when it is an array that already satisfies the request, as in NumPy:
        ///     <c>np.float64(a) is a</c>).</returns>
        /// <exception cref="OverflowError">A C# integer (or a float's integer part) outside an integer class's range
        ///     (<c>Python integer 300 out of bounds for int8</c>), a float infinity into an integer class, or a Python int past
        ///     the double range into a float class (<c>int too large to convert to float</c>).</exception>
        /// <exception cref="ValueError">A float NaN into an integer class (<c>cannot convert float NaN to integer</c>), or a
        ///     text that does not parse as the class's number (<c>invalid literal for int() with base 10: '1.5'</c>,
        ///     <c>could not convert string to float: 'abc'</c>).</exception>
        /// <exception cref="TypeError"><c>None</c> or a complex into an integer class, or a complex into a float class
        ///     (CPython's <c>int()</c> / <c>float()</c> texts).</exception>
        /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
        internal static NDArray Generic(DType descr, object x)
        {
            if (x is Missing)
                return Zero(descr);
            return ToScalarOrArray(FromAnyForceCast(x, descr));
        }

        /// <summary>
        ///     <c>bool_arrtype_new</c>: no argument is False, a C# bool is itself, anything else is
        ///     <c>PyArray_FROM_OTF(x, NPY_BOOL, NPY_ARRAY_FORCECAST)</c> - its truth for a scalar (<c>np.bool_('')</c> is
        ///     False, <c>np.bool_('False')</c> True, <c>np.bool_(None)</c> False), the bool array of a sequence.
        /// </summary>
        /// <param name="x">The argument, or <see cref="Missing.Value"/>; null is <c>None</c>.</param>
        /// <returns>The read-only 0-d bool scalar, or the bool ndarray of a sequence or an array with dimensions.</returns>
        /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
        internal static NDArray Bool(object x)
        {
            // `if (obj == NULL) RETURN_FALSE; if (obj == Py_False) ...; if (obj == Py_True) ...`: the two singletons.
            if (x is Missing)
                return NDArray.Scalar(false);
            if (x is bool b)
                return NDArray.Scalar(b);
            return ToScalarOrArray(FromAnyForceCast(x, np.@bool));
        }

        /// <summary>
        ///     <c>@name@_arrtype_new</c> of <c>cdouble</c> (<c>np.complex128</c>, <c>np.cdouble</c>): NumPy's <c>"|OO"</c>
        ///     parse, where a second argument builds <c>complex(real, imag)</c> and is accepted only when both arguments are
        ///     Python numbers and neither is a complex.
        /// </summary>
        /// <param name="descr">The class's descriptor (<c>np.complex128</c>).</param>
        /// <param name="real">The first argument, or <see cref="Missing.Value"/>; null is <c>None</c>.</param>
        /// <param name="imag">The second argument, or <see cref="Missing.Value"/>.</param>
        /// <returns>The read-only 0-d complex scalar, or the ndarray a one-argument sequence or array converts to.</returns>
        /// <exception cref="TypeError">Two arguments that are not both real numbers (<c>Could not convert arguments into a
        ///     complex scalar. '((1+1j), 2)' given.</c>), or <paramref name="imag"/> named without <paramref name="real"/>
        ///     (NumPy's arguments are positional-only: <c>'imag' is an invalid keyword argument for this function</c>).</exception>
        /// <exception cref="OverflowError">A Python int past the double range as either part (<c>int too large to convert to
        ///     float</c>).</exception>
        /// <exception cref="ValueError">A text that does not parse as a complex (<c>complex() arg is a malformed
        ///     string</c>).</exception>
        /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
        internal static NDArray CDouble(DType descr, object real, object imag)
        {
            if (imag is Missing)
                return Generic(descr, real);
            // A C# named argument can give imag alone; NumPy's parameters are positional-only, so that is its keyword error.
            if (real is Missing)
                throw new TypeError("'imag' is an invalid keyword argument for this function");

            // `PyNumber_Check(real) && PyNumber_Check(imag) && !(PyComplex_Check(real) || PyComplex_Check(imag))`, else
            // NumPy's parse error quoting the argument tuple's repr.
            if (!IsRealNumberForComplex(real) || !IsRealNumberForComplex(imag))
                throw new TypeError($"Could not convert arguments into a complex scalar. '({NDPyObject.Repr(real)}, {NDPyObject.Repr(imag)})' given.");

            // `PyObject_CallObject(&PyComplex_Type, args)`: complex(real, imag) = real + imag*1j over the parts' float()
            // values; a 0-d complex ARRAY passes the checks above (an ndarray is no Python complex) and contributes its own
            // imaginary part, as CPython's complex_new adds cr.imag to ci.real.
            Complex r = PyComplexOf(real);
            Complex i = PyComplexOf(imag);
            return Generic(descr, new Complex(r.Real - i.Imaginary, r.Imaginary + i.Real));
        }

        /// <summary>
        ///     <c>@name@_arrtype_new</c> of <c>unicode</c> (<c>np.str_</c>): Python's <c>str(object, encoding, errors)</c>
        ///     first, which never fails for one argument under the boundary map (<c>np.str_(5)</c> is <c>'5'</c>,
        ///     <c>np.str_([1, 2])</c> is <c>'[1, 2]'</c>, <c>np.str_(None)</c> is <c>'None'</c>), its text returned as the
        ///     <c>&lt;U{len}&gt;</c> scalar.
        /// </summary>
        /// <param name="obj">The object, or <see cref="Missing.Value"/> (<c>''</c>); null is <c>None</c>.</param>
        /// <param name="encoding">The codec <paramref name="obj"/>'s bytes are decoded with, or null; given (or
        ///     <paramref name="errors"/> given), <paramref name="obj"/> must be bytes-like.</param>
        /// <param name="errors">The decoding error handler, or null for <c>strict</c>.</param>
        /// <returns>The read-only 0-d <c>U</c> scalar of the text.</returns>
        /// <exception cref="TypeError">A codec argument with an <paramref name="obj"/> that is not bytes-like (<c>decoding
        ///     str is not supported</c>, <c>decoding to str: need a bytes-like object, int found</c>).</exception>
        /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode under <c>strict</c>.</exception>
        /// <exception cref="LookupError">An unknown codec or error handler.</exception>
        /// <exception cref="ValueError">A Python int of more than 4300 digits (CPython's <c>str()</c> limit).</exception>
        internal static NDArray Unicode(object obj, string encoding, string errors)
        {
            string text = encoding is null && errors is null
                ? obj is Missing ? string.Empty : NDPyObject.Str(obj)
                : DecodeForStr(obj, encoding, errors);
            // `return from_superclass`: the str itself is the np.str_ scalar, its width its length.
            return Backends.UnmanagedStorage.FromAnyScalar(text, null).PyArray_Return();
        }

        /// <summary>
        ///     <c>@name@_arrtype_new</c> of <c>string</c> (<c>np.bytes_</c>): Python's <c>bytes(source, encoding, errors)</c>
        ///     first - a count of zero bytes for an integer (<c>np.bytes_(5)</c> is five NULs, <c>|S5</c>), the bytes of a
        ///     list of 0..255 integers, a <c>byte[]</c> copied, an array's raw buffer -, and when that raises for a single
        ///     argument, NumPy's conversion <c>PyArray_FromAny(x, 'S', FORCECAST)</c> (<c>np.bytes_("a")</c> is <c>b'a'</c>,
        ///     <c>np.bytes_(1.5)</c> <c>b'1.5'</c>, <c>np.bytes_(-1)</c> <c>b'-1'</c>).
        /// </summary>
        /// <param name="source">The source, or <see cref="Missing.Value"/> (<c>b''</c>); null is <c>None</c>.</param>
        /// <param name="encoding">The codec a str <paramref name="source"/> is encoded with, or null.</param>
        /// <param name="errors">The encoding error handler, or null for <c>strict</c>.</param>
        /// <returns>The read-only 0-d <c>S</c> scalar, or the <c>S</c> ndarray a sequence the fallback converts gives
        ///     (<c>np.bytes_([1, 300])</c> is <c>[b'1', b'300']</c>).</returns>
        /// <exception cref="TypeError">A codec argument without a str source (<c>encoding without a string
        ///     argument</c>).</exception>
        /// <exception cref="UnicodeEncodeError">Text the codec (ASCII for the fallback) cannot encode.</exception>
        /// <exception cref="LookupError">An unknown codec or error handler.</exception>
        /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
        internal static NDArray String(object source, string encoding, string errors)
        {
            if (encoding is not null || errors is not null)
                return BytesScalar(EncodeForBytes(source, encoding, errors));
            if (source is Missing)
                return BytesScalar(Array.Empty<byte>());
            if (TryPyBytes(source, out byte[] bytes))
                return BytesScalar(bytes);
            // `if (from_superclass == NULL) { ... PyErr_Clear(); }`: one positional argument, so NumPy's conversion runs.
            return ToScalarOrArray(FromAnyForceCast(source, np.bytes_));
        }

        // ---------------------------------------------------------------------------------------------------------------
        // The shared steps.
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>PyArray_Scalar(NULL, typecode, NULL)</c> followed by the <c>memset</c> to zero: the class's zero scalar.
        /// </summary>
        /// <param name="descr">The class's descriptor.</param>
        /// <returns>The read-only 0-d zero of <paramref name="descr"/>.</returns>
        private static NDArray Zero(DType descr) => np.zeros(Shape.Scalar, descr).PyArray_Return();

        /// <summary>
        ///     NumPy's <c>if (PyArray_NDIM(arr) &gt; 0) return arr;</c> then <c>PyArray_ToScalar</c>: a 0-d result becomes the
        ///     scalar, any other is the ndarray.
        /// </summary>
        /// <param name="arr">The conversion's result: fresh when 0-d (<see cref="FromAnyForceCast"/> guarantees it).</param>
        /// <returns>The scalar, or <paramref name="arr"/>.</returns>
        private static NDArray ToScalarOrArray(NDArray arr) => arr.ndim > 0 ? arr : arr.PyArray_Return();

        /// <summary>
        ///     <c>PyArray_FromAny(x, descr, 0, 0, NPY_ARRAY_FORCECAST)</c> over the C# boundary map: a Python value (a C#
        ///     number, bool, text, None) is packed by the dtype's setitem (<see cref="UnmanagedStorage.FromAnyScalar"/>: NEP 50's
        ///     range check, CPython's parsers, None's setitem), a NumPy scalar (an <see cref="NDArray"/>, a <see cref="Half"/>,
        ///     NumSharp's <see cref="decimal"/>) or an array is cast unsafely, a C# array or collection (a Python list)
        ///     converted by <see cref="np.asanyarray(in object, DType, string)"/>.
        /// </summary>
        /// <param name="x">The argument (not <see cref="Missing.Value"/>); null is <c>None</c>.</param>
        /// <param name="descr">The requested descriptor.</param>
        /// <returns>A NEW array for every 0-d result (never the caller's 0-d array: <see cref="NDArray.PyArray_Return"/>
        ///     changes its argument in place); for an argument with dimensions, the argument itself when it satisfies the
        ///     request (NumPy returns the same object), else its cast.</returns>
        /// <exception cref="OverflowError">See <see cref="Generic"/>.</exception>
        /// <exception cref="ValueError">See <see cref="Generic"/>.</exception>
        /// <exception cref="TypeError">See <see cref="Generic"/>.</exception>
        /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
        private static NDArray FromAnyForceCast(object x, DType descr)
        {
            if (x is NDArray nd)
            {
                // A 0-d argument (a NumPy scalar or a 0-d array) is CAST into a fresh 0-d array: FORCECAST is an unsafe cast,
                // and the result must be fresh for PyArray_Return. An argument with dimensions passes through the array
                // conversion, which returns it as itself when its dtype already satisfies the request.
                return nd.ndim == 0 ? nd.astype(descr, copy: true) : np.asanyarray(nd, descr);
            }

            // A NumPy scalar of the boundary map (Half is np.float16; NumSharp's decimal stands with them) is discovered as its
            // own dtype and CAST by PyArray_Pack, unlike a Python number: np.int8(np.float16(300)) is 44, where the int8
            // setitem of a store would range-check it.
            if (x is Half or decimal)
                return NDArray.WritableScalar(x).astype(descr, copy: true);

            // One Python value: PyArray_Pack into a new 0-d array of the descriptor (the dtype's setitem).
            if (x is null || NDArray.IsStoreScalar(x) || x is string || x is byte[])
                return UnmanagedStorage.FromAnyScalar(x, descr);

            // A Python list, tuple or other sequence: discovered and stored element by element.
            return np.asanyarray(x, descr);
        }

        /// <summary>
        ///     Python's <c>PyNumber_Check(x) &amp;&amp; !PyComplex_Check(x)</c> under the boundary map: what NumPy's two-argument
        ///     complex parse accepts as each part - a C# bool, integer, <see cref="BigInteger"/>, float, double, decimal or
        ///     <see cref="Half"/>, and any <see cref="NDArray"/> that is not a complex SCALAR (an ndarray implements the number
        ///     protocol; NumPy's complex128 scalar IS a Python complex).
        /// </summary>
        /// <param name="value">One argument.</param>
        /// <returns>True when NumPy's check accepts it.</returns>
        private static bool IsRealNumberForComplex(object value) => value switch
        {
            bool or sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger => true,
            float or double or decimal or Half => true,
            NDArray nd => !(nd.IsNumPyScalar && nd.typecode == NPTypeCode.Complex),
            _ => false,
        };

        /// <summary>
        ///     The value <c>complex(real, imag)</c> reads from one accepted part: its float() as the real part, or a 0-d complex
        ///     array's own value.
        /// </summary>
        /// <param name="value">One part (accepted by <see cref="IsRealNumberForComplex"/>).</param>
        /// <returns>The part as a complex number.</returns>
        /// <exception cref="OverflowError">A Python int past the double range (<c>int too large to convert to
        ///     float</c>).</exception>
        /// <exception cref="TypeError">An array with dimensions (CPython's <c>float()</c> of an ndarray: only 0-d arrays
        ///     convert).</exception>
        private static Complex PyComplexOf(object value)
        {
            switch (value)
            {
                case bool b:
                    return b ? 1.0 : 0.0;
                case BigInteger big:
                    double d = (double)big;
                    if (double.IsInfinity(d))
                        throw new OverflowError("int too large to convert to float");
                    return d;
                case Half h:
                    return (double)h;
                case NDArray nd:
                    if (nd.ndim != 0)
                        throw new TypeError("only 0-dimensional arrays can be converted to Python scalars");
                    return nd.typecode == NPTypeCode.Complex ? nd.GetComplex() : Convert.ToDouble(nd.GetAtIndex(0), System.Globalization.CultureInfo.InvariantCulture);
                default:
                    return Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        /// <summary>
        ///     Python's single-argument <c>bytes(x)</c> under the boundary map, for <see cref="String"/>'s first step.
        /// </summary>
        /// <param name="x">The source (not <see cref="Missing.Value"/>).</param>
        /// <param name="bytes">The bytes <c>bytes(x)</c> returns, when it succeeds.</param>
        /// <returns>
        ///     True when CPython's <c>bytes(x)</c> succeeds: a <c>byte[]</c> (copied); a non-negative integer, a bool included
        ///     (that many zero bytes; a NumPy integer scalar or 0-d integer array through <c>__index__</c>); a NumPy scalar or
        ///     array of another dtype (its raw buffer, C order: <c>np.bytes_(np.float64(2.5))</c> is its 8 bytes); a list of
        ///     integers in 0..255. False where CPython raises (a negative count, a str, a float, None, a list with another
        ///     item): <see cref="String"/> then runs NumPy's conversion.
        /// </returns>
        private static bool TryPyBytes(object x, out byte[] bytes)
        {
            bytes = null;
            switch (x)
            {
                case byte[] b:
                    bytes = (byte[])b.Clone();
                    return true;
                case bool flag:
                    // A Python bool is an int: bytes(True) is one zero byte.
                    bytes = new byte[flag ? 1 : 0];
                    return true;
                case sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger:
                    return TryZeroBytes(ToBigInteger(x), out bytes);
                case Half or decimal:
                    // NumPy scalars (np.float16; NumSharp's decimal) expose their buffer: the value's raw bytes.
                    bytes = NDArray.Scalar(x).tobytes();
                    return true;
                case NDArray nd:
                    // A 0-d integer array or scalar has __index__ (a count); every other array exports its buffer, the C-order
                    // bytes. np.bool_ has no __index__ in NumPy 2, so a bool array is its buffer too.
                    if (nd.ndim == 0 && IsIntegerCode(nd.typecode))
                        return TryZeroBytes(ToBigInteger(nd.GetAtIndex(0)), out bytes);
                    bytes = nd.tobytes();
                    return true;
                case string or char or null or float or double or Complex:
                    // CPython raises: string argument without an encoding; cannot convert 'float' / 'NoneType' object to bytes.
                    return false;
                case IEnumerable items:
                    // An iterable of ints, each in range(0, 256); any other item raises (ValueError / TypeError).
                    var list = new System.Collections.Generic.List<byte>();
                    foreach (object item in items)
                    {
                        if (item is bool or sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger)
                        {
                            BigInteger v = item is bool bit ? (bit ? 1 : 0) : ToBigInteger(item);
                            if (v < 0 || v > 255)
                                return false;
                            list.Add((byte)v);
                        }
                        else
                            return false;
                    }
                    bytes = list.ToArray();
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        ///     CPython's <c>bytes(n)</c> of an integer count: <paramref name="count"/> zero bytes, refused (so the caller
        ///     falls back) for a negative count (<c>negative count</c>) and for one past what a .NET array holds (CPython would
        ///     allocate it, and NumPy's scalar of it overflows its 32-bit itemsize).
        /// </summary>
        /// <param name="count">The count.</param>
        /// <param name="bytes">The zero bytes.</param>
        /// <returns>True when the count is in 0..<see cref="Array.MaxLength"/>.</returns>
        private static bool TryZeroBytes(BigInteger count, out byte[] bytes)
        {
            bytes = null;
            if (count < 0 || count > Array.MaxLength)
                return false;
            bytes = new byte[(int)count];
            return true;
        }

        /// <summary>A boxed C# integer (or <see cref="BigInteger"/>) as a <see cref="BigInteger"/>.</summary>
        /// <param name="value">The integer.</param>
        /// <returns>Its value.</returns>
        private static BigInteger ToBigInteger(object value) => value switch
        {
            BigInteger big => big,
            ulong u => u,
            _ => Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture),
        };

        /// <summary>Whether a type code is one of NumPy's integer kinds (NumSharp's char counts, a uint16).</summary>
        /// <param name="code">The type code.</param>
        /// <returns>True for the signed and unsigned integers and char.</returns>
        private static bool IsIntegerCode(NPTypeCode code) => code is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16
            or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Char;

        /// <summary>The read-only 0-d <c>S</c> scalar of <paramref name="bytes"/>, as wide as they are.</summary>
        /// <param name="bytes">The bytes (not null).</param>
        /// <returns>The scalar.</returns>
        private static NDArray BytesScalar(byte[] bytes) => UnmanagedStorage.FromAnyScalar(bytes, null).PyArray_Return();

        /// <summary>
        ///     Python's <c>str(obj, encoding, errors)</c>: <paramref name="obj"/> must be bytes-like (a <c>byte[]</c>, an
        ///     <c>S</c> array's bytes, or any array's raw buffer) and is decoded.
        /// </summary>
        /// <param name="obj">The object, or <see cref="Missing.Value"/> (CPython's default <c>b''</c>).</param>
        /// <param name="encoding">The codec, or null for UTF-8.</param>
        /// <param name="errors">The error handler, or null for <c>strict</c>.</param>
        /// <returns>The decoded text.</returns>
        /// <exception cref="TypeError"><paramref name="obj"/> is not bytes-like.</exception>
        /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode under <c>strict</c>.</exception>
        /// <exception cref="LookupError">An unknown codec or error handler.</exception>
        private static string DecodeForStr(object obj, string encoding, string errors)
        {
            byte[] raw = obj switch
            {
                Missing => Array.Empty<byte>(),
                byte[] b => b,
                NDArray nd when nd.typecode == NPTypeCode.String && nd.dtype.@char == 'S' && nd.ndim == 0 => (byte[])nd.Storage.GetAtIndex(0),
                NDArray nd => nd.tobytes(),
                string or char => throw new TypeError("decoding str is not supported"),
                _ => throw new TypeError($"decoding to str: need a bytes-like object, {NDPyObject.TypeName(obj)} found"),
            };
            // CPython's bytes.decode through np.strings.decode, the codecs NumSharp implements (a 0-d S scalar in, a 0-d U out).
            using var source = NDArray.Scalar(raw);
            using var decoded = np.strings.decode(source, encoding ?? "utf-8", errors);
            return (string)decoded.Storage.GetAtIndex(0);
        }

        /// <summary>
        ///     Python's <c>bytes(source, encoding, errors)</c>: <paramref name="source"/> must be a str and is encoded.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="encoding">The codec, or null (CPython then raises: a codec name is required with errors=).</param>
        /// <param name="errors">The error handler, or null for <c>strict</c>.</param>
        /// <returns>The encoded bytes.</returns>
        /// <exception cref="TypeError"><paramref name="source"/> is not a str (<c>encoding without a string argument</c>),
        ///     or no encoding was given (<c>string argument without an encoding</c>).</exception>
        /// <exception cref="UnicodeEncodeError">Text the codec cannot encode under <c>strict</c>.</exception>
        /// <exception cref="LookupError">An unknown codec or error handler.</exception>
        private static byte[] EncodeForBytes(object source, string encoding, string errors)
        {
            string text = source switch
            {
                string s => s,
                char c => c.ToString(),
                _ => throw new TypeError("encoding without a string argument"),
            };
            if (encoding is null)
                throw new TypeError("string argument without an encoding");
            using var unicode = NDArray.Scalar(text);
            using var encoded = np.strings.encode(unicode, encoding, errors);
            return (byte[])encoded.Storage.GetAtIndex(0);
        }
    }
}
