using System;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Returns True if cast between data types can occur according to the casting rule — NumPy's
        /// <c>np.can_cast(from_, to, casting='safe')</c> over two descriptors (<c>PyArray_CanCastTypeTo</c>, NEP 43):
        /// the <see cref="CastingImpl"/> registered for the two DType CLASSES answers, so byte order and datetime units
        /// take part (<c>can_cast('&gt;i4', 'i4', 'equiv')</c> is True and <c>'no'</c> is False;
        /// <c>can_cast('M8[D]', 'M8[s]', 'safe')</c> is True and the reverse is False).
        /// </summary>
        /// <param name="from">Data type to cast from (any spelling that converts to <see cref="DType"/>).</param>
        /// <param name="to">Data type to cast to.</param>
        /// <param name="casting">
        /// Controls what kind of data casting may occur (case-sensitive, as in NumPy):
        /// - "no" means the data types should not be cast at all.
        /// - "equiv" means only byte-order changes are allowed.
        /// - "safe" means only casts which can preserve values are allowed.
        /// - "same_kind" means only safe casts or casts within a kind (int to int, float to float) are allowed.
        /// - "unsafe" means any data conversions may be done.
        /// </param>
        /// <returns>True if cast can occur according to the casting rule.</returns>
        /// <exception cref="TypeError">
        ///     <paramref name="to"/> or <paramref name="from"/> is null (NumPy's <c>None</c>: "did not understand one of the
        ///     types; 'None' not accepted"), or <paramref name="casting"/> is null ("casting must be str, not NoneType"). NumPy
        ///     converts <c>to</c> and <c>casting</c> while it parses the arguments and <c>from_</c> after, so the checks run
        ///     in that order.
        /// </exception>
        /// <exception cref="ValueError"><c>casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '…')</c> — verbatim NumPy.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.can_cast.html</remarks>
        public static bool can_cast(DType from, DType to, string casting = "safe")
        {
            if (to is null) throw new TypeError("did not understand one of the types; 'None' not accepted");
            var rule = DTypeCasting.ParseCasting(casting);
            if (from is null) throw new TypeError("did not understand one of the types; 'None' not accepted");
            return DTypeCasting.CanCastTypeTo(from, to, rule);
        }

        /// <summary>
        /// <see cref="can_cast(DType, DType, string)"/> for a dtype STRING source (<c>np.can_cast("i4", "i8")</c>). Exists so that a
        /// string first argument binds the dtype grammar rather than NumSharp's string→<see cref="NDArray"/> (character array) conversion.
        /// </summary>
        /// <param name="from">The dtype string to cast from; null is NumPy's <c>None</c>.</param>
        /// <param name="to">Data type to cast to.</param>
        /// <param name="casting">The casting rule (see <see cref="can_cast(DType, DType, string)"/>).</param>
        /// <returns>True if cast can occur according to the casting rule.</returns>
        /// <exception cref="TypeError">
        ///     A null argument (NumPy's texts, in NumPy's order: <c>to</c>, <c>casting</c>, then <c>from_</c>), or a
        ///     <paramref name="from"/> string NumPy cannot parse: NumPy replaces the parser's error with <c>did not understand
        ///     one of the types; 'None' not accepted</c> for <c>from_</c> (a bad <paramref name="to"/> string keeps the
        ///     parser's own error, raised while the argument converts).
        /// </exception>
        /// <exception cref="ValueError">An unknown casting name.</exception>
        /// <exception cref="NotSupportedException"><paramref name="from"/> is a valid NumPy dtype NumSharp does not implement (complex64, structured, ...).</exception>
        public static bool can_cast(string from, DType to, string casting = "safe")
        {
            if (to is null) throw new TypeError("did not understand one of the types; 'None' not accepted");
            var rule = DTypeCasting.ParseCasting(casting);
            if (from is null) throw new TypeError("did not understand one of the types; 'None' not accepted");
            DType fromDescr;
            try
            {
                fromDescr = dtype(from);
            }
            catch (Exception e) when (e is TypeError || e is ValueError)
            {
                // array_can_cast_safely converts from_ with PyArray_DescrConverter2 and then REPLACES whatever error the
                // converter set (PyErr_SetString), so "data type 'x' not understood", a removed alias and a bad datetime
                // unit or divisor all read as this text. A NotSupportedException is not one of NumPy's errors (NumPy
                // accepts the dtype) and passes through.
                throw new TypeError("did not understand one of the types; 'None' not accepted");
            }
            return DTypeCasting.CanCastTypeTo(fromDescr, to, rule);
        }

        /// <summary>
        /// Returns True if cast between data types can occur according to the casting rule (the <see cref="NPY_CASTING"/> spelling).
        /// </summary>
        /// <param name="from">Data type to cast from.</param>
        /// <param name="to">Data type to cast to.</param>
        /// <param name="casting">The casting level.</param>
        /// <returns>True if cast can occur according to the casting rule.</returns>
        /// <exception cref="TypeError"><paramref name="to"/> or <paramref name="from"/> is null (NumPy's "did not understand one of the types; 'None' not accepted").</exception>
        public static bool can_cast(DType from, DType to, NPY_CASTING casting)
        {
            if (to is null) throw new TypeError("did not understand one of the types; 'None' not accepted");
            if (from is null) throw new TypeError("did not understand one of the types; 'None' not accepted");
            return DTypeCasting.CanCastTypeTo(from, to, casting);
        }

        /// <summary>
        /// Returns True if cast between data types can occur according to the casting rule.
        /// </summary>
        /// <param name="from">Data type to cast from.</param>
        /// <param name="to">Data type to cast to.</param>
        /// <param name="casting">
        /// Controls what kind of data casting may occur:
        /// - "no" means the data types should not be cast at all.
        /// - "equiv" means only byte-order changes are allowed.
        /// - "safe" means only casts which can preserve values are allowed.
        /// - "same_kind" means only safe casts or casts within a kind (int to int, float to float) are allowed.
        /// - "unsafe" means any data conversions may be done.
        /// </param>
        /// <returns>True if cast can occur according to the casting rule.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.can_cast.html
        ///
        /// The <see cref="NPTypeCode"/> spelling of <see cref="can_cast(DType, DType, string)"/>: the same NEP 43 engine,
        /// whose answers for the storage-backed types are the rules this overload always applied
        /// (<c>safe ⇔ promote(from, to) == to</c>; <c>same_kind</c> ⇔ safe or the source kind orders at or below the
        /// destination kind).
        /// </remarks>
        /// <example>
        /// <code>
        /// np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64)           // True
        /// np.can_cast(NPTypeCode.Int64, NPTypeCode.Int32)           // False
        /// np.can_cast(NPTypeCode.Int32, NPTypeCode.Single, "same_kind")  // True (int -> float is allowed)
        /// np.can_cast(NPTypeCode.Int32, NPTypeCode.Int16, "unsafe") // True
        /// </code>
        /// </example>
        public static bool can_cast(NPTypeCode from, NPTypeCode to, string casting = "safe")
        {
            var rule = DTypeCasting.ParseCasting(casting);
            if (from == to)
                return true;

            return DTypeCasting.CanCastTypeTo(DType.From(from), DType.From(to), rule);
        }

        /// <summary>
        /// Returns True if cast between data types can occur according to the casting rule.
        /// </summary>
        /// <param name="from">CLR type to cast from.</param>
        /// <param name="to">CLR type to cast to.</param>
        /// <param name="casting">Controls what kind of data casting may occur.</param>
        /// <returns>True if cast can occur according to the casting rule.</returns>
        public static bool can_cast(Type from, Type to, string casting = "safe")
        {
            return can_cast(from.GetTypeCode(), to.GetTypeCode(), casting);
        }

        /// <summary>
        ///     NumPy's <c>np.can_cast(from_, to, casting='safe')</c> for a source whose static type is <see cref="object"/>:
        ///     <c>array_can_cast_safely</c>'s dispatch on <c>from_</c>, statement for statement — an array by its own
        ///     descriptor (<c>PyArray_CanCastArrayTo</c>), a NumPy scalar by its dtype, a Python number REFUSED (NEP 50), and
        ///     anything else converted as a dtype specifier.
        /// </summary>
        /// <param name="value">The source: an <see cref="NDArray"/>; a dtype spelling (<see cref="DType"/>,
        ///     <see cref="NPTypeCode"/>, <see cref="Type"/>, <see cref="NDScalarType"/>, a dtype string); a NumPy scalar (a
        ///     boxed <see cref="Half"/> or <c>char</c>, the C# values NumSharp treats as NumPy scalars); or a C# number, which
        ///     NumPy 2 refuses.</param>
        /// <param name="to">Data type to cast to; <see cref="NPTypeCode.Empty"/> is NumPy's <c>None</c>.</param>
        /// <param name="casting">The casting rule (see <see cref="can_cast(DType, DType, string)"/>).</param>
        /// <returns>True if the source's dtype casts to <paramref name="to"/> under <paramref name="casting"/>.</returns>
        /// <exception cref="TypeError">A boxed C# number (<c>bool</c>, an integer primitive, <c>float</c>, <c>double</c>,
        ///     <see cref="System.Numerics.Complex"/>): NumPy 2's refusal, <c>can_cast() does not support Python ints, floats,
        ///     and complex because the result used to depend on the value.</c> (two lines, see
        ///     <see cref="PythonNumberRefusal"/>). A null source, a null <paramref name="to"/>, or an object that names no
        ///     dtype (a C# array, a list): <c>did not understand one of the types; 'None' not accepted</c>.</exception>
        /// <exception cref="ValueError">An unknown <paramref name="casting"/> name, raised before the source is looked at (NumPy
        ///     parses it with the arguments).</exception>
        /// <exception cref="NotSupportedException"><paramref name="to"/> is <see cref="NPTypeCode.String"/> for an array
        ///     source (see <see cref="can_cast(NDArray, NPTypeCode, string)"/>).</exception>
        /// <remarks>
        ///     API-parity batch 9. NumSharp kept eleven typed value overloads (<c>can_cast(int value, ...)</c> and its twins)
        ///     with a value table, "does 300 fit int8", which NumPy 2 replaced by the refusal; the table also had holes
        ///     (<c>can_cast(1, NPTypeCode.SByte)</c> was False). They went, so a bare C# number now binds
        ///     <see cref="can_cast(NDArray, NPTypeCode, string)"/> through NDArray's implicit conversion, which refuses the
        ///     literal-marked 0-d array the same way; this form refuses the boxed number. The literal <c>0</c> also converts
        ///     to <see cref="NPTypeCode"/>, so <c>np.can_cast(0, NPTypeCode.Byte)</c> is CS0121 at compile time. A
        ///     <c>decimal</c> has no NumPy analog: it is NumSharp's decimal dtype, judged by the dtype rule.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.can_cast.html
        /// </remarks>
        public static bool can_cast(object value, NPTypeCode to, string casting = "safe")
        {
            // npy_parse_arguments converts `to` and `casting` before array_can_cast_safely looks at from_, so their errors
            // win over the source's (an unknown casting name is the ValueError even for a Python number).
            DType toDescr = to; // NPTypeCode.Empty converts to null, NumPy's None
            if (toDescr is null)
                throw new TypeError("did not understand one of the types; 'None' not accepted");
            DTypeCasting.ParseCasting(casting);
            switch (value)
            {
                // `if (PyArray_Check(from_obj))`: the array's own descriptor; a literal-marked 0-d array refuses there.
                case NDArray array:
                    return can_cast(array, to, casting);
                // `PyArray_IsPythonNumber(from_obj)`: NEP 50's refusal. NumPy tests NumPy scalars first, but no C# primitive
                // below is one: NumSharp spells a NumPy scalar as a 0-d array (NDArray.Scalar) or as Half / char.
                case bool or sbyte or byte or short or ushort or int or uint or long or ulong or float or double or System.Numerics.Complex:
                    throw PythonNumberRefusal();
                // `PyArray_IsScalar(from_obj, Generic)`: the C# values NumSharp's implicit conversions mark WasNumPyScalar
                // are NumPy scalars, judged by their dtype.
                case Half:
                    return can_cast(NPTypeCode.Half, to, casting);
                case char:
                    return can_cast(NPTypeCode.Char, to, casting);
                // No NumPy analog: NumSharp's decimal dtype (its implicit conversion leaves the 0-d array unmarked too).
                case decimal:
                    return can_cast(NPTypeCode.Decimal, to, casting);
                // `PyArray_DescrConverter2(from_obj, &d1)`: a dtype specifier in any of NumSharp's spellings.
                case DType descr:
                    return can_cast(descr, toDescr, casting);
                case NPTypeCode code when code != NPTypeCode.Empty:
                    return can_cast(code, to, casting);
                case Type type:
                    return can_cast(DType.From(type), toDescr, casting);
                case NDScalarType scalarType:
                    return can_cast(scalarType.ToDescriptor(), toDescr, casting);
                case string spec:
                    return can_cast(spec, toDescr, casting);
                // A null, a C# array or list, or any other object names no dtype: NumPy replaces the converter's error with
                // this text (PyErr_SetString after PyArray_DescrConverter2 fails or yields NULL).
                default:
                    throw new TypeError("did not understand one of the types; 'None' not accepted");
            }
        }

        /// <summary>
        /// Returns True if cast from array dtype can occur according to the casting rule.
        /// </summary>
        /// <param name="from">Array to cast from: arrays are judged by their dtype, never by value (NEP 50).</param>
        /// <param name="to">Data type to cast to.</param>
        /// <param name="casting">Controls what kind of data casting may occur.</param>
        /// <returns>True if the array can be cast to the target type.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="from"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="from"/> is the 0-d array a bare C# number became (a
        ///     <c>bool</c>, an integer primitive, <c>float</c>, <c>double</c> or <see cref="System.Numerics.Complex"/>
        ///     literal, which NumSharp's implicit conversions mark as NumPy's Python scalar): NumPy 2's NEP 50 refusal
        ///     (<see cref="PythonNumberRefusal"/>). <c>np.array(1)</c>, <c>NDArray.Scalar(1)</c> and a <see cref="Half"/>
        ///     keep the dtype rule, as NumPy's 0-d arrays and NumPy scalars do.</exception>
        /// <exception cref="ValueError"><paramref name="casting"/> is not one of NumPy's five casting names.</exception>
        /// <exception cref="NotSupportedException"><paramref name="to"/> is <see cref="NPTypeCode.String"/>: a storage
        ///     lane names neither the string class nor its width (pass a <see cref="DType"/> such as <c>np.dtype("U5")</c>).</exception>
        /// <remarks>Since API-parity batch 9 this is the overload a bare C# number binds (<c>np.can_cast(1,
        ///     NPTypeCode.SByte)</c>, through NDArray's implicit conversion), so it carries the refusal the removed typed value
        ///     overloads never made.</remarks>
        public static bool can_cast(NDArray from, NPTypeCode to, string casting = "safe")
        {
            if (from is null) throw new ArgumentNullException(nameof(from));
            if (from.WasPythonLiteral || from.WasPythonBool)
            {
                // NumPy parses `casting` with the arguments, before array_can_cast_safely looks at from_.
                DTypeCasting.ParseCasting(casting);
                throw PythonNumberRefusal();
            }
            // Arrays use type-based casting, not value-based. A string array is judged by its DESCRIPTOR (class and
            // width): its storage lane alone cannot say whether <U2 reaches float64 or <U9.
            if (from.typecode == NPTypeCode.String)
                return can_cast(from.dtype, DType.From(to), casting);
            return can_cast(from.GetTypeCode, to, casting);
        }

        /// <summary>
        /// Returns True if cast from the array's dtype can occur according to the casting rule (NumPy's array form of
        /// <c>can_cast</c>; arrays are never inspected by value under NEP 50).
        /// </summary>
        /// <param name="from">The array whose dtype is cast, by its own descriptor (for a string array: class and width);
        ///     null is NumPy's <c>None</c>.</param>
        /// <param name="to">Data type to cast to.</param>
        /// <param name="casting">The casting rule (see <see cref="can_cast(DType, DType, string)"/>).</param>
        /// <returns>True if the array's dtype can be cast to <paramref name="to"/> under <paramref name="casting"/>.</returns>
        /// <exception cref="TypeError">A null argument (NumPy's texts, in NumPy's order: <c>to</c>, <c>casting</c>, then
        ///     <c>from_</c>), or <paramref name="from"/> is the 0-d array a bare C# number became: NumPy 2's NEP 50 refusal
        ///     (<see cref="PythonNumberRefusal"/>), after the <c>to</c> and <c>casting</c> checks.</exception>
        /// <exception cref="ValueError"><paramref name="casting"/> is not one of NumPy's five casting names.</exception>
        public static bool can_cast(NDArray from, DType to, string casting = "safe")
        {
            if (from is not null && (from.WasPythonLiteral || from.WasPythonBool))
            {
                // `to` and `casting` convert while NumPy parses the arguments, before from_ is looked at.
                if (to is null) throw new TypeError("did not understand one of the types; 'None' not accepted");
                DTypeCasting.ParseCasting(casting);
                throw PythonNumberRefusal();
            }
            // The array's own descriptor, never DType.From(typecode): the String lane carries no width. A null array is
            // NumPy's None, which the descriptor overload refuses with NumPy's text, after the checks NumPy makes first.
            return can_cast(from?.dtype, to, casting);
        }

        /// <summary>
        ///     NumPy 2's refusal of a Python number in <c>np.can_cast</c> (<c>array_can_cast_safely</c>, NEP 50): the result
        ///     used to depend on the value (<c>can_cast(300, int8)</c>), which NEP 50 removed.
        /// </summary>
        /// <returns>The <see cref="TypeError"/> with NumPy's two-line text, for the caller to throw.</returns>
        internal static TypeError PythonNumberRefusal()
            => new TypeError("can_cast() does not support Python ints, floats, and complex because the result used to depend on the value.\n"
                             + "This change was part of adopting NEP 50, we may explicitly allow them again in the future.");
    }
}
