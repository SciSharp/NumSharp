using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     True for the 0-d array that a C# number literal becomes through an implicit conversion operator: NumPy's
        ///     <c>NPY_ARRAY_WAS_PYTHON_LITERAL</c> mark (<c>NPY_ARRAY_WAS_PYTHON_INT</c> / <c>_FLOAT</c> / <c>_COMPLEX</c>,
        ///     set by <c>npy_mark_tmp_array_if_pyscalar</c> on the temporary array of a Python scalar argument).
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     The C# boundary map makes the integer primitives, <c>float</c>, <c>double</c> and
        ///     <see cref="System.Numerics.Complex"/> NumPy's Python <c>int</c> / <c>float</c> / <c>complex</c>. A store
        ///     reads a Python scalar differently from an array: <c>a[0] = 1000</c> on an int8 array raises
        ///     <c>OverflowError: Python integer 1000 out of bounds for int8</c> (<c>PyArray_Pack</c>), while
        ///     <c>a[0] = np.array(1000)</c> casts it to -24. An <see cref="NDArray"/> parameter receives both as a 0-d
        ///     array, so this mark is the only way to tell them apart.
        ///     </para>
        ///     <para>
        ///     Only the store paths read the mark (<see cref="PackForStore"/>): the indexer setters, <c>np.put</c>,
        ///     <c>np.putmask</c>, <c>np.place</c>, <c>np.put_along_axis</c>, <c>np.asarray</c> with a dtype, and every
        ///     <c>object</c>-typed value parameter through <see cref="ReductionInitial.TryPack"/>. The ufuncs and the
        ///     comparisons keep their own rule, which reads EVERY 0-d operand as a Python scalar (np.clip's rule,
        ///     <c>[Misaligned]</c>): library code builds its scalar operands with <see cref="Scalar{T}(T)"/> and relies on
        ///     that rule, so switching them to the mark would change their result dtypes. Two readers besides the stores:
        ///     <see cref="Nep50Operand"/> (a function that promotes by NumPy's <c>result_type</c>), and the reduce prologue
        ///     through <see cref="WasScalarArgument"/>, which refuses a C# literal given as <c>out=</c>.
        ///     </para>
        ///     <para>
        ///     The mark belongs to this one object, never to its storage: a view, a copy, a reshape or an operation result
        ///     is a new array without it — as NumPy's flag does not survive <c>np.asarray(x).reshape(1)</c>. A
        ///     <c>bool</c> literal is not marked (NumPy marks only an exact <c>int</c> / <c>float</c> / <c>complex</c>;
        ///     a bool stores like its cast anyway). <see cref="Half"/> and <c>char</c> are NumPy scalars in the boundary
        ///     map and carry <see cref="WasNumPyScalar"/> instead; <c>decimal</c> (no NumPy analog) carries neither.
        ///     </para>
        /// </remarks>
        internal bool WasPythonLiteral { get; private set; }

        /// <summary>
        ///     True for the 0-d array that a C# <see cref="Half"/> or <c>char</c> value becomes through an implicit
        ///     conversion operator: a NumPy SCALAR in the boundary map (<c>np.float16</c>, <c>np.uint16</c>), which a store
        ///     packs by <c>PyArray_Pack</c>'s NumPy-scalar rule (<see cref="ReductionInitial.PackNumPyScalar"/>).
        /// </summary>
        /// <remarks>
        ///     A NumPy scalar stores differently from both a Python scalar and a 0-d array: only a SIGNED integer DType
        ///     knows every NumPy scalar and int()-converts it with the range check (<c>a[0] = np.float16(300)</c> on int8
        ///     raises "Python integer 300 out of bounds for int8"), every other DType casts it
        ///     (<c>np.float16(300)</c> into uint8 is 44). A 0-d ARRAY of the same value is always cast. Like
        ///     <see cref="WasPythonLiteral"/>, the mark belongs to this one object; the store paths read it, and the reduce
        ///     prologue through <see cref="WasScalarArgument"/> (a NumPy scalar is not an <c>out=</c> array).
        /// </remarks>
        internal bool WasNumPyScalar { get; private set; }

        /// <summary>
        ///     True for the 0-d text array that a C# <see cref="string"/> becomes through the implicit conversion operator: a
        ///     Python <c>str</c> in the boundary map. A store packs it the way NumPy's <c>PyArray_Pack</c> packs a Python str —
        ///     parsed by the destination's setitem (<c>int()</c> / <c>float()</c> / <c>complex()</c>), its errors quoting the
        ///     str itself (<c>invalid literal for int() with base 10: 'x'</c>) — where a 0-d text ARRAY
        ///     (<c>np.array("x")</c>) is cast like any array and its errors quote the NumPy scalar (<c>np.str_('x')</c>).
        /// </summary>
        /// <remarks>
        ///     NumPy's own flag (<c>NPY_ARRAY_WAS_PYTHON_LITERAL</c>) marks only int / float / complex, because it serves
        ///     NEP 50 promotion; a Python str never takes part in that. This mark serves the store paths
        ///     (<see cref="PackForStore"/>, <see cref="ReductionInitial.TryPack"/>), which in NumPy receive the str object
        ///     itself, and the reduce prologue through <see cref="WasScalarArgument"/> (a str is not an <c>out=</c> array).
        ///     Like <see cref="WasPythonLiteral"/>, it belongs to this one object: a view or a copy is not marked.
        /// </remarks>
        internal bool WasPythonStr { get; private set; }

        /// <summary>
        ///     True for the 0-d <c>S</c> array that <c>numpy.char</c>'s text rule makes of a C# <c>byte[]</c>
        ///     (<see cref="CharModule.TextOperand"/>): a Python <c>bytes</c> object in that module's boundary map.
        /// </summary>
        /// <remarks>
        ///     A Python <c>bytes</c> object has no <c>dtype</c> attribute, and <c>np.strings.replace</c> reads that attribute
        ///     (<c>getattr(old, 'dtype', None)</c>): an <c>old</c> / <c>new</c> without one is cast to the text's dtype
        ///     first, so <c>np.char.array(['ab']).replace(b'a', 'b')</c> replaces in <c>U</c> text where an <c>S</c> ARRAY
        ///     argument meets NumPy's no-loop TypeError. The other <c>np.strings</c> functions cast their text arguments
        ///     whatever they are and do not read the mark. Only <see cref="PythonBytesLiteral"/> sets it, and like the other
        ///     marks it belongs to this one object.
        /// </remarks>
        internal bool WasPythonBytes { get; private set; }

        /// <summary>
        ///     True for the 0-d bool array that a C# <c>bool</c> becomes through the implicit conversion operator: a Python
        ///     <c>bool</c> in the boundary map.
        /// </summary>
        /// <remarks>
        ///     Two readers. The NEP 55 <c>StringDType</c> store (<see cref="PackForVStringStore"/>): NumPy's
        ///     <c>stringdtype_setitem</c> stores <c>str(True)</c> under coercion and REFUSES a bool under <c>coerce=False</c>,
        ///     where a cast of a 0-d bool array stores <c>'True'</c> in both. Every other store packs a bool like its cast,
        ///     which is why <see cref="WasPythonLiteral"/> (NumPy's flag marks only int / float / complex) stays false for a
        ///     bool. And the reduce prologue, through <see cref="WasScalarArgument"/>: a Python bool is not an <c>out=</c>
        ///     array, so <c>np.max(a, 1, true)</c> — the bool landing in NumPy's third positional parameter, <c>out</c> —
        ///     raises NumPy's TypeError instead of reducing into a 0-d bool. Like the other marks, it belongs to this one
        ///     object.
        /// </remarks>
        internal bool WasPythonBool { get; private set; }

        /// <summary>
        ///     True for the 0-d array that a C# scalar became through an implicit conversion operator — a Python <c>int</c> /
        ///     <c>float</c> / <c>complex</c> (<see cref="WasPythonLiteral"/>), <c>bool</c> (<see cref="WasPythonBool"/>) or
        ///     <c>str</c> (<see cref="WasPythonStr"/>), or a NumPy scalar (<see cref="WasNumPyScalar"/>: <see cref="Half"/>,
        ///     <c>char</c>): a value NumPy receives as a scalar OBJECT, never as an ndarray.
        /// </summary>
        /// <remarks>
        ///     NumPy refuses such a value where it requires an array: a reduction's <c>out=</c> takes only an ndarray
        ///     (<c>_set_out_array</c>: "return arrays must be of ArrayType"), so <c>np.max(a, 1, True)</c> raises there. In C#
        ///     the same call converts the literal to a 0-d array, and only this mark tells it from a 0-d array the caller built
        ///     (<c>np.array(true)</c> is a valid out for a flat reduction). A <c>decimal</c> carries no mark (it has no NumPy
        ///     analog), so it is read as an array. Like the marks it combines, it belongs to this one object: a view or a copy
        ///     of the array is not marked.
        /// </remarks>
        internal bool WasScalarArgument => WasPythonLiteral || WasPythonBool || WasPythonStr || WasNumPyScalar;

        /// <summary>
        ///     The C# text a <see cref="WasPythonStr"/> array was made from, exactly (trailing NULs included); null for any other
        ///     array. The array's 0-d <c>U</c> element reads WITHOUT trailing NULs (<c>U</c>'s storage rule), so a store into a
        ///     NEP 55 <c>StringDType</c> array — which keeps them, as NumPy's setitem stores the Python str itself — reads this.
        /// </summary>
        internal string PythonStrText { get; private set; }

        /// <summary>
        ///     The operand NEP 50 promotion (<see cref="DTypePromotion.ResultType"/>) reads for this array: the boxed C#
        ///     literal a <see cref="WasPythonLiteral"/> array was made from — a WEAK Python <c>int</c> / <c>float</c> /
        ///     <c>complex</c> that adopts the other operands' dtype —, else the array itself, which is STRONG (a 0-d array
        ///     included, as in NumPy 2.x).
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     This is how a function that promotes by NumPy's <c>result_type</c> reads the mark NumPy sets on the temporary
        ///     array of a Python scalar argument (<c>NPY_ARRAY_WAS_PYTHON_LITERAL</c>). It matters most next to text: a Python
        ///     number and a string array have no common DType, so <c>np.where(c, u, 1)</c> raises NumPy's
        ///     <c>DTypePromotionError</c> (<c>The DType &lt;class 'numpy.dtypes.StrDType'&gt; could not be promoted by
        ///     &lt;class 'numpy.dtypes._PyLongDType'&gt;</c>), where the 0-d int32 array of the literal, read as an array,
        ///     promotes to <c>&lt;U11</c>.
        ///     </para>
        ///     <para>
        ///     A bool, str, <see cref="Half"/> or <c>char</c> mark is not weak (NumPy's flag marks only int / float /
        ///     complex): those arrays are returned as themselves, and promote as the strong dtype they hold.
        ///     </para>
        /// </remarks>
        internal object Nep50Operand => WasPythonLiteral ? Storage.GetAtIndex(0) : this;

        /// <summary>
        ///     The 0-d array a boxed C# value stands for as a Python argument, marked the way its implicit conversion to
        ///     <see cref="NDArray"/> marks it: an integer primitive, <c>float</c>, <c>double</c> or
        ///     <see cref="System.Numerics.Complex"/> is a <see cref="WasPythonLiteral"/> array (a weak Python scalar), a
        ///     <c>bool</c> a <see cref="WasPythonBool"/> array, a <see cref="string"/> a <see cref="WasPythonStr"/> array, a
        ///     <see cref="Half"/> or <c>char</c> a <see cref="WasNumPyScalar"/> array. An <see cref="NDArray"/> is returned as
        ///     it is; anything else (a <c>decimal</c>, a C# array or collection) is <c>np.asanyarray(value)</c>.
        /// </summary>
        /// <param name="value">The boxed argument (an <c>object</c> parameter's value).</param>
        /// <returns>The array (a new one, except an <see cref="NDArray"/> argument).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        /// <exception cref="NotSupportedException"><paramref name="value"/> has no NumSharp array form.</exception>
        /// <remarks>
        ///     An <c>object</c> overload must give a boxed literal the same meaning the <see cref="NDArray"/> overload gives
        ///     the literal itself: <c>np.where(c, u, (object)1)</c> and <c>np.where(c, u, 1)</c> are one NumPy call. The
        ///     unmarked <c>np.asanyarray(1)</c> is a strong 0-d array, which a promotion by marks reads as a NumPy array.
        /// </remarks>
        internal static NDArray FromPythonOperand(object value) => value switch
        {
            null => throw new ArgumentNullException(nameof(value)),
            NDArray nd => nd,
            sbyte v => PythonLiteral(v),
            byte v => PythonLiteral(v),
            short v => PythonLiteral(v),
            ushort v => PythonLiteral(v),
            int v => PythonLiteral(v),
            uint v => PythonLiteral(v),
            long v => PythonLiteral(v),
            ulong v => PythonLiteral(v),
            float v => PythonLiteral(v),
            double v => PythonLiteral(v),
            System.Numerics.Complex v => PythonLiteral(v),
            bool v => PythonBoolLiteral(v),
            string v => PythonStrLiteral(v),
            Half v => NumPyScalarLiteral(v),
            char v => NumPyScalarLiteral(v),
            // np.asanyarray(value)'s statement: an NDArray is the arm above, so the class boundaries never apply.
            _ => asanyarray(value, null, 'K'),
        };

        /// <summary>
        ///     Creates the 0-d bool array of a C# <c>bool</c> literal, marked <see cref="WasPythonBool"/>. The implicit
        ///     <c>bool → NDArray</c> conversion calls it.
        /// </summary>
        /// <param name="value">The literal.</param>
        /// <returns>A new 0-d bool array holding <paramref name="value"/>.</returns>
        internal static NDArray PythonBoolLiteral(bool value)
        {
            var nd = Scalar(value);
            nd.WasPythonBool = true;
            return nd;
        }

        /// <summary>
        ///     Creates the 0-d text array of a C# string literal (NumPy's <c>np.array(str)</c>: <c>&lt;U{len}</c>), marked
        ///     <see cref="WasPythonStr"/>. The implicit <c>string → NDArray</c> conversion calls it.
        /// </summary>
        /// <param name="value">The text (not null).</param>
        /// <returns>A new 0-d <c>U</c> array holding <paramref name="value"/>.</returns>
        internal static NDArray PythonStrLiteral(string value)
        {
            var nd = Backends.UnmanagedStorage.FromAnyScalar(value, null);
            nd.WasPythonStr = true;
            nd.PythonStrText = value;
            return nd;
        }

        /// <summary>
        ///     Creates the 0-d <c>S</c> array of a Python <c>bytes</c> value (NumPy's <c>np.asanyarray(b'ab')</c>:
        ///     <c>|S2</c>, and <c>|S1</c> for <c>b''</c>), marked <see cref="WasPythonBytes"/>. <c>numpy.char</c>'s text
        ///     rule calls it for a C# <c>byte[]</c>.
        /// </summary>
        /// <param name="value">The bytes (not null).</param>
        /// <returns>A new 0-d <c>S</c> array holding <paramref name="value"/>.</returns>
        internal static NDArray PythonBytesLiteral(byte[] value)
        {
            var nd = Scalar(value);
            nd.WasPythonBytes = true;
            return nd;
        }

        /// <summary>
        ///     Creates the 0-d array of a C# <see cref="Half"/> or <c>char</c> value, marked <see cref="WasNumPyScalar"/>.
        ///     The implicit conversion operators call it.
        /// </summary>
        /// <typeparam name="T"><see cref="Half"/> or <c>char</c>.</typeparam>
        /// <param name="value">The value.</param>
        /// <returns>A new 0-d array of <typeparamref name="T"/>'s dtype holding <paramref name="value"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NDArray NumPyScalarLiteral<T>(T value) where T : unmanaged
        {
            var nd = Scalar(value);
            nd.WasNumPyScalar = true;
            return nd;
        }

        /// <summary>
        ///     Creates the 0-d array of a C# number literal that stands for a Python <c>int</c>, <c>float</c> or
        ///     <c>complex</c>, marked <see cref="WasPythonLiteral"/>. The implicit conversion operators call it.
        /// </summary>
        /// <typeparam name="T">An integer primitive, <c>float</c>, <c>double</c> or <see cref="System.Numerics.Complex"/>.</typeparam>
        /// <param name="value">The literal.</param>
        /// <returns>A new 0-d array of <typeparamref name="T"/>'s dtype holding <paramref name="value"/>.</returns>
        /// <remarks>Use <see cref="Scalar{T}(T)"/> (or <c>np.array(value)</c>) for a 0-d array that must store like an
        ///     ndarray (cast, never range-checked).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NDArray PythonLiteral<T>(T value) where T : unmanaged
        {
            var nd = Scalar(value);
            nd.WasPythonLiteral = true;
            return nd;
        }

        /// <summary>
        ///     The value an assignment stores into an array of <paramref name="destination"/>: a C# literal
        ///     (<see cref="WasPythonLiteral"/>) is packed the way NumPy's <c>PyArray_Pack</c> packs a Python scalar, a C#
        ///     <see cref="Half"/> / <c>char</c> value (<see cref="WasNumPyScalar"/>) the way it packs a NumPy scalar; any
        ///     other value is returned unchanged, for the caller's usual cast.
        /// </summary>
        /// <param name="value">The value of the assignment; <c>null</c> is Python's <c>None</c>: a NUMERIC destination
        ///     receives None's setitem value (<see cref="NoneForStore"/>: NaN, <c>nan+nanj</c>, <c>False</c>, or int()'s
        ///     TypeError), a string destination gets <c>null</c> back for its text route, which stores <c>str(None)</c> or
        ///     its NA object itself (<see cref="PackForStore(NDArray, DType, bool)"/> resolves it for any dtype).</param>
        /// <param name="destination">The destination's dtype.</param>
        /// <param name="pyArrayPackRoute">
        ///     True when NumPy reaches the element through <c>PyArray_Pack</c> directly — an integer index, a view assignment
        ///     (slices, ellipsis), <c>fill</c>, the flat iterator's single index —; false for the <c>PyArray_FromAny</c>
        ///     routes (fancy and mask assignment, <c>np.put</c>, <c>np.put_along_axis</c>, <c>np.insert</c>, the flat
        ///     iterator's slices). Only a NumPy-scalar mark depends on it: NumPy int()-converts a NumPy scalar into a SIGNED
        ///     integer on the first kind of route (<c>a[0] = np.float16(300)</c> on int8 raises) and CASTS it like a 0-d
        ///     array on the second (<c>a[[0]] = np.float16(300)</c> stores 44), probed against NumPy 2.4.2. A Python
        ///     literal packs the same way on every route.
        /// </param>
        /// <returns>
        ///     <paramref name="value"/> itself when it is not a marked literal, when <paramref name="destination"/> is its
        ///     own dtype (the literal is exact there) or a string dtype (the text routes convert numbers themselves);
        ///     otherwise a NEW 0-d array of <paramref name="destination"/> holding the packed element. The caller disposes
        ///     a new array (<c>!ReferenceEquals(result, value)</c>), or creates it inside an <c>[NDScoped]</c> member.
        /// </returns>
        /// <exception cref="OverflowException">A Python int out of the destination's integer range ("Python integer 1000
        ///     out of bounds for int8"; past the C <c>long</c> the LP64 text "Python int too large to convert to C long"), a
        ///     float infinity into an integer ("cannot convert float infinity to integer"), or an int past the double range
        ///     into a float ("int too large to convert to float").</exception>
        /// <exception cref="ValueError">A float NaN into an integer: "cannot convert float NaN to integer".</exception>
        /// <exception cref="TypeError">A complex into a real or integer destination: Python's <c>float()</c> /
        ///     <c>int()</c> texts; <c>None</c> (a null <paramref name="value"/>) into an integer or <c>decimal</c>
        ///     destination.</exception>
        /// <remarks>
        ///     <para>
        ///     A NumPy-scalar mark packs by <see cref="ReductionInitial.PackNumPyScalar"/> instead: int()-converted with the
        ///     range check into a SIGNED integer destination, cast into every other.
        ///     </para>
        ///     <para>
        ///     A null <paramref name="value"/> is converted HERE, at the point each caller packs a literal, which is where
        ///     NumPy converts <c>None</c> too: after an integer index's bounds check, before a fancy index's, and for an
        ///     empty selection (<c>a[5:] = None</c> on an integer array raises).
        ///     </para>
        ///     <para>
        ///     The conversion is <see cref="ReductionInitial.TryPack"/>, NumPy's element setter: an int is range-checked
        ///     (never wrapped), a float is truncated toward zero into an integer and then range-checked, an int reaches
        ///     float16 / float32 through the correctly rounded double (two roundings past 2^53, as in NumPy), a bool
        ///     destination takes the value's truth.
        ///     </para>
        ///     <para>
        ///     NumPy packs at different points per index kind, and the callers follow it: after the bounds check of an
        ///     integer index (<c>a[5] = 1000</c> on a 3-element array is the IndexError), but BEFORE the bounds check of a
        ///     fancy index (<c>a[[5]] = 1000</c> is the OverflowError) and for an empty selection too (<c>a[5:] = 1000</c>
        ///     raises).
        ///     </para>
        /// </remarks>
        internal static unsafe NDArray PackForStore(NDArray value, NPTypeCode destination, bool pyArrayPackRoute)
        {
            // None into a number: NumPy's setitem of None (NaN, nan+nanj, False, or int()'s TypeError). A string destination's
            // None is its text route's business (str(None), or a StringDType NA object), so null goes back to it.
            if (value is null)
                return destination == NPTypeCode.String || destination == NPTypeCode.Empty ? null : NoneForStore(destination);

            // A NumPy scalar on a PyArray_FromAny route is cast like any 0-d array: the caller's cast applies. A Python str
            // into a string destination is the text route's own business (it converts text itself).
            if (!(value.WasPythonLiteral || value.WasPythonStr || (value.WasNumPyScalar && pyArrayPackRoute))
                || destination == value.typecode || destination == NPTypeCode.String || destination == NPTypeCode.Empty)
                return value;

            // The literal's own C# value (a boxed int / long / ulong / double / float / Complex, a string, or a Half /
            // char), packed by NumPy's rules for the Python or NumPy scalar it stands for. A marked array is 0-d by
            // construction, so element 0 is the literal.
            byte* slot = stackalloc byte[ReductionInitial.SlotBytes];
            if (value.WasPythonLiteral || value.WasPythonStr)
                ReductionInitial.TryPack(value.Storage.GetAtIndex(0), destination, new Span<byte>(slot, ReductionInitial.SlotBytes));
            else
                ReductionInitial.PackNumPyScalar(value.Storage.GetAtIndex(0), destination, new Span<byte>(slot, ReductionInitial.SlotBytes));

            var packed = new NDArray(destination, Shape.Scalar, false);
            // (System.Runtime.CompilerServices.Unsafe is shadowed by NDArray.Unsafe inside this class.)
            Buffer.MemoryCopy(slot, (byte*)packed.Address, packed.dtypesize, packed.dtypesize);
            return packed;
        }

        /// <summary>
        ///     <see cref="PackForStore(NDArray, NPTypeCode, bool)"/> for a destination known by its full descriptor, which
        ///     also resolves <c>None</c> for a text destination: a null <paramref name="value"/> becomes
        ///     <see cref="NoneForStore"/> of <paramref name="destination"/> (<c>'None'</c> truncated to an <c>S</c> /
        ///     <c>U</c> width, a <c>StringDType</c>'s NA object or <c>'None'</c>, NaN / <c>nan+nanj</c> / <c>False</c> for
        ///     numbers), any other value packs as the typecode overload packs it.
        /// </summary>
        /// <param name="value">The value of the assignment; <c>null</c> is Python's <c>None</c>.</param>
        /// <param name="destination">The destination array's descriptor.</param>
        /// <param name="pyArrayPackRoute">See <see cref="PackForStore(NDArray, NPTypeCode, bool)"/>.</param>
        /// <returns><paramref name="value"/> itself, or a NEW 0-d array the caller disposes
        ///     (<c>!ReferenceEquals(result, value)</c>) or creates inside an <c>[NDScoped]</c> member.</returns>
        /// <exception cref="TypeError"><c>None</c> into an integer or <c>decimal</c> destination, or the literal packing
        ///     errors of the typecode overload.</exception>
        /// <exception cref="ValueError"><c>None</c> into a <c>StringDType(coerce=False)</c> destination, or the literal
        ///     packing errors of the typecode overload.</exception>
        /// <exception cref="OverflowException">See <see cref="PackForStore(NDArray, NPTypeCode, bool)"/>.</exception>
        internal static NDArray PackForStore(NDArray value, DType destination, bool pyArrayPackRoute)
            => value is null ? NoneForStore(destination) : PackForStore(value, destination.typecode, pyArrayPackRoute);

        /// <summary>
        ///     Python's <c>None</c> as the value of a store into an array of <paramref name="descr"/>: a NEW 0-d array of
        ///     exactly <paramref name="descr"/> holding what NumPy's setitem stores for <c>None</c> — the conversion
        ///     <c>PyArray_FromAny(None, descr, FORCECAST)</c> / <c>PyArray_Pack(descr, item, None)</c> performs on every store
        ///     route.
        /// </summary>
        /// <param name="descr">The destination's descriptor.</param>
        /// <returns>The 0-d value: NaN (the POSITIVE quiet NaN, NumPy's <c>NPY_NAN</c>) for a float dtype, <c>nan+nanj</c>
        ///     for complex, <c>False</c> for bool, <c>str(None)</c> = <c>'None'</c> encoded and truncated to an <c>S</c> /
        ///     <c>U</c> width (<c>'No'</c> for <c>&lt;U2</c>; an unsized class discovers <c>&lt;U4</c> / <c>|S4</c>), and
        ///     for a <c>StringDType</c> its NA element when its NA object is <c>None</c>, else <c>'None'</c>. The caller
        ///     disposes it or creates it inside an <c>[NDScoped]</c> member.</returns>
        /// <exception cref="TypeError">An integer dtype (<c>char</c> included) — Python's <c>int(None)</c> — or a
        ///     <c>decimal</c> one (<see cref="ReductionInitial.PackNone"/>).</exception>
        /// <exception cref="ValueError">A <c>StringDType(coerce=False)</c> descriptor (<c>StringDType only allows string data
        ///     when string coercion is disabled.</c>).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="descr"/> is null.</exception>
        /// <remarks>
        ///     The caller decides WHEN it converts: at NumPy's conversion point of its route, so the errors keep NumPy's
        ///     order (<c>np.put(int_array, [10], None)</c> raises the TypeError before the index's bounds are checked,
        ///     <c>np.insert(int_array, 10, None)</c> the IndexError first). <c>np.copyto</c> is the one route whose None is an
        ///     OBJECT scalar cast per element (<c>casting='unsafe'</c> only), and it calls this only where NumPy casts.
        /// </remarks>
        internal static unsafe NDArray NoneForStore(DType descr)
        {
            if (descr is null)
                throw new ArgumentNullException(nameof(descr));

            // StringDType: stringdtype_setitem of None (the NA object when it is None, 'None' under coercion, refused under
            // coerce=False).
            if (descr.Meta is VStringDTypeMeta)
                return PackForVStringStore(null, descr);
            // S / U: STRING_setitem / UNICODE_setitem take any object by its str(), so 'None', encoded and truncated to the
            // width (an unsized class discovers its width from the text, as np.asarray(None, dtype='U') is <U4).
            if (descr.Meta is StringDTypeMeta)
                return Backends.UnmanagedStorage.FromAnyScalar(null, descr);

            NPTypeCode tc = descr.typecode;
            byte* slot = stackalloc byte[ReductionInitial.SlotBytes];
            ReductionInitial.PackNone(tc, new Span<byte>(slot, ReductionInitial.SlotBytes));
            var none = new NDArray(tc, Shape.Scalar, false);
            // (System.Runtime.CompilerServices.Unsafe is shadowed by NDArray.Unsafe inside this class.)
            Buffer.MemoryCopy(slot, (byte*)none.Address, none.dtypesize, none.dtypesize);
            return none;
        }

        /// <summary>
        ///     The refusal for <c>None</c> where NumPy would DISCOVER the dtype from it: NumPy builds an object array
        ///     (<c>np.full(2, None)</c>, <c>np.asarray(None)</c>), and NumSharp has no object dtype.
        /// </summary>
        /// <param name="function">The NumPy function, as the message names it (<c>"np.full"</c>).</param>
        /// <returns>The exception to throw.</returns>
        internal static NotSupportedException NoneWithoutDtype(string function)
            => new NotSupportedException(
                $"{function} with None and no dtype builds an object array in NumPy, a dtype NumSharp does not have. Pass a " +
                "dtype: a float or complex dtype stores NaN, bool stores False, a text dtype stores 'None'.");

        /// <summary>
        ///     The value of a store into a NEP 55 <c>StringDType</c> array: a C# literal (NumPy's Python scalar or Python
        ///     <c>str</c>) is stored the way NumPy's <c>stringdtype_setitem</c> stores a Python object — the descriptor's NA
        ///     object packs an NA element, anything else becomes its <c>str()</c> when the descriptor coerces (a
        ///     <c>coerce=False</c> descriptor refuses a non-text value) — into a NEW 0-d array of <paramref name="descr"/>; any
        ///     other value (an array, which NumPy CASTS) is returned unchanged.
        /// </summary>
        /// <param name="value">The value of the assignment; null is Python's <c>None</c> (the NA element of a descriptor whose
        ///     NA object is <c>None</c>, <c>'None'</c> under coercion, refused under <c>coerce=False</c>).</param>
        /// <param name="descr">The destination's <c>StringDType</c> descriptor.</param>
        /// <returns><paramref name="value"/> itself, or a new 0-d array the caller disposes
        ///     (<c>!ReferenceEquals(result, value)</c>).</returns>
        /// <exception cref="ValueError">A non-text literal (a bool and <c>None</c> included) into a <c>coerce=False</c>
        ///     descriptor (<c>StringDType only allows string data when string coercion is disabled.</c>).</exception>
        /// <exception cref="UnicodeEncodeError">A text literal with a lone surrogate.</exception>
        /// <remarks>
        ///     NumPy reaches this conversion for EVERY scalar store into a <c>StringDType</c> array — an element, a slice, a
        ///     fancy or a mask assignment (<c>a[[0]] = 5</c> under <c>coerce=False</c> raises, probed 2.4.2) — while an
        ///     ndarray value takes the cast (<c>a[:] = np.array([1, 2])</c> under <c>coerce=False</c> stores <c>'1'</c>,
        ///     <c>'2'</c>). NumSharp's 0-d arrays that are not C# literals (NumPy scalars included) take the cast route.
        /// </remarks>
        internal static NDArray PackForVStringStore(NDArray value, DType descr)
        {
            // A C# null in an NDArray-typed store is Python's None; a marked literal (number, bool, str) is the Python
            // scalar it stands for. Both go through stringdtype_setitem; any other array is the caller's cast.
            if (value is not null && !(value.WasPythonLiteral || value.WasPythonStr || value.WasPythonBool))
                return value;
            var packed = new NDArray(descr, Shape.Scalar, true);
            try
            {
                // A str literal stores its ORIGINAL C# text: its 0-d U array drops trailing NULs on read ('a\0' reads 'a'),
                // which a StringDType element keeps (NumPy's setitem stores the Python str itself).
                object element = value is null ? null : value.WasPythonStr ? value.PythonStrText : value.Storage.GetAtIndex(0);
                packed.Storage.VStringSlice.SetIndex(0, element);
            }
            catch
            {
                packed.Dispose();
                throw;
            }
            return packed;
        }

        /// <summary>
        ///     True for a boxed C# number that the boundary map makes a Python scalar (an integer primitive,
        ///     <see cref="System.Numerics.BigInteger"/>, <c>bool</c>, <c>float</c>, <c>double</c>,
        ///     <see cref="System.Numerics.Complex"/>) or a NumPy scalar (<see cref="Half"/>, <c>char</c>, <c>decimal</c>) —
        ///     the values <see cref="AsStoreValue"/> packs instead of converting them to an array first.
        /// </summary>
        /// <param name="value">The boxed value.</param>
        /// <returns>Whether <paramref name="value"/> is one number.</returns>
        internal static bool IsStoreScalar(object value)
            => value is bool or sbyte or byte or short or ushort or int or uint or long or ulong
                or System.Numerics.BigInteger or float or double or System.Numerics.Complex or Half or char or decimal;

        /// <summary>
        ///     The value of a store given through an <c>object</c> parameter, converted the way NumPy's
        ///     <c>PyArray_FromAny(value, dtype, NPY_ARRAY_FORCECAST)</c> converts it: a C# number is packed into a NEW 0-d
        ///     array of <paramref name="destination"/> by <c>PyArray_Pack</c>'s rules for its kind; an <see cref="NDArray"/>
        ///     goes through <see cref="PackForStore"/> (a C# literal's 0-d array is packed, any other array returned as is);
        ///     anything else (a C# array or collection — a Python list —, text) is <c>np.asanyarray(value)</c>, for the
        ///     caller's own array cast.
        /// </summary>
        /// <param name="value">The value of the store; <c>null</c> is Python's <c>None</c>, which a NUMERIC destination
        ///     stores as <see cref="NoneForStore"/> gives it (NaN, <c>nan+nanj</c>, <c>False</c>, or int()'s TypeError).</param>
        /// <param name="destination">The destination's dtype.</param>
        /// <returns>A NEW array, except an <see cref="NDArray"/> that <see cref="PackForStore"/> returns unchanged. The
        ///     caller disposes a new array or creates it inside an <c>[NDScoped]</c> member.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null and <paramref name="destination"/> is a
        ///     string dtype: <c>None</c> into text needs the destination's full descriptor (its width, its NA object), so
        ///     such a caller converts it with <see cref="NoneForStore"/> instead.</exception>
        /// <exception cref="TypeError"><c>None</c> into an integer or <c>decimal</c> destination.</exception>
        /// <exception cref="OverflowException">See <see cref="PackForStore"/>; also a NumPy scalar (<see cref="Half"/>,
        ///     <c>char</c>) whose int() does not fit a SIGNED integer destination.</exception>
        /// <exception cref="ValueError">See <see cref="PackForStore"/>.</exception>
        /// <exception cref="TypeError">See <see cref="PackForStore"/>.</exception>
        /// <remarks>
        ///     <c>np.insert</c> (<c>array(values, dtype=arr.dtype)</c>), <c>np.fill_diagonal</c> (<c>a.flat[...] = val</c>)
        ///     and <c>np.put</c>'s scalar overload (<c>PyArray_FromAny(values, dtype, FORCECAST)</c>) convert their value this
        ///     way. <c>PyArray_FromAny</c> packs a Python scalar (an integer primitive, <see cref="System.Numerics.BigInteger"/>,
        ///     <c>bool</c>, <c>float</c>, <c>double</c>, <see cref="System.Numerics.Complex"/>) but CASTS a NumPy scalar like a
        ///     0-d array (<see cref="Half"/>, <c>char</c>, and NumSharp's <c>decimal</c>: <c>np.insert(int8, 1,
        ///     np.float16(300))</c> stores 44, probed). A text value or a string destination is left to the text routes.
        /// </remarks>
        internal static unsafe NDArray AsStoreValue(object value, NPTypeCode destination)
        {
            if (value is null)
            {
                // None into a number is its setitem value; a text destination's caller holds the descriptor this needs.
                if (destination == NPTypeCode.String || destination == NPTypeCode.Empty)
                    throw new ArgumentNullException(nameof(value), "None into a text array is converted with the array's descriptor (NDArray.NoneForStore).");
                return NoneForStore(destination);
            }
            if (value is NDArray nd)
                return PackForStore(nd, destination, pyArrayPackRoute: false);
            // A Python str into a NUMERIC destination: PyArray_FromAny(str, dtype, FORCECAST) runs the dtype's setitem on
            // the str itself (PyArray_Pack: int() / float() / complex(), a bool takes the str's truth), so its errors quote
            // the str ('1.5'), where converting it to a U array first and casting that would quote the array element
            // (np.str_('1.5')). The values are the same either way.
            bool numericText = value is string && destination != NPTypeCode.String && destination != NPTypeCode.Empty;
            // np.asanyarray(value)'s statement (an NDArray value returned above, so no class boundary applies).
            if (!numericText && (!IsStoreScalar(value) || destination == NPTypeCode.String || destination == NPTypeCode.Empty))
                return asanyarray(value, null, 'K');

            byte* slot = stackalloc byte[ReductionInitial.SlotBytes];
            if (value is Half or char or decimal)
                ReductionInitial.CastScalarValue(value, destination, new Span<byte>(slot, ReductionInitial.SlotBytes));
            else
                ReductionInitial.TryPack(value, destination, new Span<byte>(slot, ReductionInitial.SlotBytes));
            var packed = new NDArray(destination, Shape.Scalar, false);
            Buffer.MemoryCopy(slot, (byte*)packed.Address, packed.dtypesize, packed.dtypesize);
            return packed;
        }
    }
}
