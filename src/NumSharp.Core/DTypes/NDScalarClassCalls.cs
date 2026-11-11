using System.Runtime.InteropServices;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's class calls (<c>np.float32(1.5f)</c>, <c>np.int8()</c>, <c>np.complex128(1, 2)</c>, <c>np.str_(5)</c>)
    ///     as
    ///     C# 14 static extension members of <see cref="np"/>, so each concrete scalar-class name is both a VALUE - the
    ///     <see cref="DType"/> field <c>np.float32</c>, unchanged - and a CALL that constructs NumPy's scalar. Reach them
    ///     through
    ///     <c>np.</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>How the two meanings coexist.</b> C# invokes only a method group or a delegate-typed value, and
    ///     <see cref="np"/>
    ///     cannot declare a method beside its same-named field (CS0102). When a name is INVOKED, C#'s member lookup drops
    ///     the
    ///     non-invocable members first (ECMA-334 12.5): for <c>np.float32(x)</c> that leaves no member of <see cref="np"/>,
    ///     so
    ///     the lookup moves on to the extension members and binds the one here. Without parentheses, <c>np.float32</c>
    ///     finds
    ///     the field: the same <see cref="DType"/> object as always, every member intact.
    ///     </para>
    ///     <para>
    ///     <b>What a call returns.</b> NumSharp's NumPy scalar: a read-only 0-d <see cref="NDArray"/> of the class's dtype
    ///     (<see cref="NDArray.PyArray_Return"/>). Where NumPy's constructor returns an ndarray - a list or an array
    ///     argument (<c>np.float32(new object[] { 1, 2 })</c>) - the call returns that array. The conversion is NumPy's
    ///     constructor (<see cref="ArrTypeNew"/>): a C# number is NumPy's Python scalar, packed with NEP 50's range check
    ///     (<c>np.int8(300)</c> raises <see cref="OverflowError"/>); an <see cref="NDArray"/> (a NumPy scalar or an array)
    ///     is
    ///     cast (<c>np.uint8(np.int64(300))</c> is 44); a <see cref="string"/> is parsed (<c>np.int8("12")</c>); omitting
    ///     the
    ///     argument gives the zero, and null is Python's <c>None</c> (<c>np.float64(null)</c> is nan).
    ///     </para>
    ///     <para>
    ///     <b>Who can call it.</b> The call needs C# 14 at the CALL SITE (the .NET 10 SDK, for any target framework, net8.0
    ///     included). A C# 12 consumer compiles every value use and gets CS9202 on the call; it writes
    ///     <c>NDScalarClassCalls.float32(x)</c> instead.
    ///     </para>
    ///     <para>
    ///     <b>Footguns, all compile-time.</b> Under <c>using static NumSharp.np;</c> a bare <c>float32(x)</c> is CS1955:
    ///     write
    ///     <c>np.float32(x)</c>, or import this class (<c>using static NumSharp.NDScalarClassCalls;</c>). A method-group
    ///     conversion (<c>Func&lt;object, NDArray&gt; f = np.float32;</c>) finds the field (CS0029): use a lambda. A
    ///     <c>dynamic</c> argument is CS1973 (extension members are not dispatched dynamically): cast it to
    ///     <see cref="object"/>.
    ///     </para>
    ///     <para>
    ///     Only NumPy's 36 class names with a NumSharp dtype have a call. The abstract classes (<c>np.floating</c>, ...)
    ///     cannot be instantiated in NumPy either; the classes without a NumSharp dtype (<c>complex64</c>,
    ///     <c>datetime64</c>, <c>void</c>, <c>object_</c>) come with their dtypes; NumPy 1.x's removed aliases
    ///     (<c>np.float_</c>, <c>np.bool8</c>, ...) and NumSharp's own dtypes (<c>np.@decimal</c>) have none.
    ///     </para>
    /// </remarks>
    public static class NDScalarClassCalls
    {
        extension(np)
        {
            /// <summary>
            ///     <c>np.bool(x)</c>, the class call of <c>numpy.bool</c>: False for no argument, a C# bool as itself,
            ///     anything else by its truth (<c>np.bool("")</c> is False, <c>np.bool("False")</c> True,
            ///     <c>np.bool(null)</c> False); a sequence gives the bool array.
            ///     NumPy 2's name of the bool class.
            ///     Written without parentheses, <c>np.bool</c> is still the <see cref="DType"/> field
            ///     <see cref="np.@bool"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray @bool([Optional] object x) => ArrTypeNew.Bool(x);

            /// <summary>
            ///     <c>np.bool_(x)</c>, the class call of <c>numpy.bool</c>: False for no argument, a C# bool as itself,
            ///     anything else by its truth (<c>np.bool_("")</c> is False, <c>np.bool_("False")</c> True,
            ///     <c>np.bool_(null)</c> False); a sequence gives the bool array.
            ///     NumPy's historical spelling of the bool class, the same class as <c>np.bool</c>.
            ///     Written without parentheses, <c>np.bool_</c> is still the <see cref="DType"/> field
            ///     <see cref="np.bool_"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray bool_([Optional] object x) => ArrTypeNew.Bool(x);

            /// <summary>
            ///     <c>np.int8(x)</c>, the class call of NumPy's <c>int8</c>: 0 for no argument; a C# integer is packed with
            ///     NumPy's range check, a float truncated toward zero first (<c>np.int8(3.9)</c> is 3), a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely; a
            ///     sequence gives the array.
            ///     Written without parentheses, <c>np.int8</c> is still the <see cref="DType"/> field
            ///     <see cref="np.int8"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray int8([Optional] object x) => ArrTypeNew.Generic(np.int8, x);

            /// <summary>
            ///     <c>np.int16(x)</c>, the class call of NumPy's <c>int16</c>: 0 for no argument; a C# integer is packed
            ///     with NumPy's range check, a float truncated toward zero first (<c>np.int16(3.9)</c> is 3), a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely; a
            ///     sequence gives the array.
            ///     Written without parentheses, <c>np.int16</c> is still the <see cref="DType"/> field
            ///     <see cref="np.int16"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray int16([Optional] object x) => ArrTypeNew.Generic(np.int16, x);

            /// <summary>
            ///     <c>np.int32(x)</c>, the class call of NumPy's <c>int32</c>: 0 for no argument; a C# integer is packed
            ///     with NumPy's range check, a float truncated toward zero first (<c>np.int32(3.9)</c> is 3), a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely; a
            ///     sequence gives the array.
            ///     Written without parentheses, <c>np.int32</c> is still the <see cref="DType"/> field
            ///     <see cref="np.int32"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray int32([Optional] object x) => ArrTypeNew.Generic(np.int32, x);

            /// <summary>
            ///     <c>np.int64(x)</c>, the class call of NumPy's <c>int64</c>: 0 for no argument; a C# integer is packed
            ///     with NumPy's range check, a float truncated toward zero first (<c>np.int64(3.9)</c> is 3), a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely; a
            ///     sequence gives the array.
            ///     Written without parentheses, <c>np.int64</c> is still the <see cref="DType"/> field
            ///     <see cref="np.int64"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray int64([Optional] object x) => ArrTypeNew.Generic(np.int64, x);

            /// <summary>
            ///     <c>np.byte(x)</c>, the class call of NumPy's <c>byte</c>: 0 for no argument; a C# integer is packed with
            ///     NumPy's range check, a float truncated toward zero first (<c>np.byte(3.9)</c> is 3), a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely; a
            ///     sequence gives the array.
            ///     NumPy's <c>np.byte</c> is the C <c>char</c> class, <c>numpy.int8</c>.
            ///     Written without parentheses, <c>np.byte</c> is still the <see cref="DType"/> field
            ///     <see cref="np.@byte"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray @byte([Optional] object x) => ArrTypeNew.Generic(np.@byte, x);

            /// <summary>
            ///     <c>np.short(x)</c>, the class call of NumPy's <c>short</c>: 0 for no argument; a C# integer is packed
            ///     with NumPy's range check, a float truncated toward zero first (<c>np.short(3.9)</c> is 3), a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely; a
            ///     sequence gives the array.
            ///     The C <c>short</c> class, <c>numpy.int16</c>.
            ///     Written without parentheses, <c>np.short</c> is still the <see cref="DType"/> field
            ///     <see cref="np.@short"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray @short([Optional] object x) => ArrTypeNew.Generic(np.@short, x);

            /// <summary>
            ///     <c>np.intc(x)</c>, the class call of NumPy's <c>intc</c>: 0 for no argument; a C# integer is packed with
            ///     NumPy's range check, a float truncated toward zero first (<c>np.intc(3.9)</c> is 3), a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely; a
            ///     sequence gives the array.
            ///     The C <c>int</c> class, <c>numpy.intc</c> (int32 storage).
            ///     Written without parentheses, <c>np.intc</c> is still the <see cref="DType"/> field
            ///     <see cref="np.intc"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray intc([Optional] object x) => ArrTypeNew.Generic(np.intc, x);

            /// <summary>
            ///     <c>np.int_(x)</c>, the class call of NumPy's <c>int_</c>: 0 for no argument; a C# integer is packed with
            ///     NumPy's range check, a float truncated toward zero first (<c>np.int_(3.9)</c> is 3), a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely; a
            ///     sequence gives the array.
            ///     NumPy 2's default integer, <c>numpy.int64</c> (<c>intp</c>).
            ///     Written without parentheses, <c>np.int_</c> is still the <see cref="DType"/> field
            ///     <see cref="np.int_"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray int_([Optional] object x) => ArrTypeNew.Generic(np.int_, x);

            /// <summary>
            ///     <c>np.long(x)</c>, the class call of NumPy's <c>long</c>: 0 for no argument; a C# integer is packed with
            ///     NumPy's range check, a float truncated toward zero first (<c>np.long(3.9)</c> is 3), a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely; a
            ///     sequence gives the array.
            ///     The C <c>long</c> class: int32 on Windows, int64 on Linux and macOS, as the host's NumPy.
            ///     Written without parentheses, <c>np.long</c> is still the <see cref="DType"/> field
            ///     <see cref="np.@long"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray @long([Optional] object x) => ArrTypeNew.Generic(np.@long, x);

            /// <summary>
            ///     <c>np.longlong(x)</c>, the class call of NumPy's <c>longlong</c>: 0 for no argument; a C# integer is
            ///     packed with NumPy's range check, a float truncated toward zero first (<c>np.longlong(3.9)</c> is 3), a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely; a
            ///     sequence gives the array.
            ///     The C <c>long long</c> class (int64 storage).
            ///     Written without parentheses, <c>np.longlong</c> is still the <see cref="DType"/> field
            ///     <see cref="np.longlong"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray longlong([Optional] object x) => ArrTypeNew.Generic(np.longlong, x);

            /// <summary>
            ///     <c>np.intp(x)</c>, the class call of NumPy's <c>intp</c>: 0 for no argument; a C# integer is packed with
            ///     NumPy's range check, a float truncated toward zero first (<c>np.intp(3.9)</c> is 3), a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely; a
            ///     sequence gives the array.
            ///     The pointer-sized integer, <c>numpy.int64</c> on a 64-bit host.
            ///     Written without parentheses, <c>np.intp</c> is still the <see cref="DType"/> field
            ///     <see cref="np.intp"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray intp([Optional] object x) => ArrTypeNew.Generic(np.intp, x);

            /// <summary>
            ///     <c>np.uint8(x)</c>, the class call of NumPy's <c>uint8</c>: 0 for no argument; a C# integer is packed
            ///     with NumPy's range check (<c>np.uint8(-1)</c> raises), a float truncated toward zero first, a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely
            ///     (<c>np.uint8(np.int64(-1))</c> wraps); a sequence gives the array.
            ///     Written without parentheses, <c>np.uint8</c> is still the <see cref="DType"/> field
            ///     <see cref="np.uint8"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray uint8([Optional] object x) => ArrTypeNew.Generic(np.uint8, x);

            /// <summary>
            ///     <c>np.uint16(x)</c>, the class call of NumPy's <c>uint16</c>: 0 for no argument; a C# integer is packed
            ///     with NumPy's range check (<c>np.uint16(-1)</c> raises), a float truncated toward zero first, a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely
            ///     (<c>np.uint16(np.int64(-1))</c> wraps); a sequence gives the array.
            ///     Written without parentheses, <c>np.uint16</c> is still the <see cref="DType"/> field
            ///     <see cref="np.uint16"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray uint16([Optional] object x) => ArrTypeNew.Generic(np.uint16, x);

            /// <summary>
            ///     <c>np.uint32(x)</c>, the class call of NumPy's <c>uint32</c>: 0 for no argument; a C# integer is packed
            ///     with NumPy's range check (<c>np.uint32(-1)</c> raises), a float truncated toward zero first, a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely
            ///     (<c>np.uint32(np.int64(-1))</c> wraps); a sequence gives the array.
            ///     Written without parentheses, <c>np.uint32</c> is still the <see cref="DType"/> field
            ///     <see cref="np.uint32"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray uint32([Optional] object x) => ArrTypeNew.Generic(np.uint32, x);

            /// <summary>
            ///     <c>np.uint64(x)</c>, the class call of NumPy's <c>uint64</c>: 0 for no argument; a C# integer is packed
            ///     with NumPy's range check (<c>np.uint64(-1)</c> raises), a float truncated toward zero first, a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely
            ///     (<c>np.uint64(np.int64(-1))</c> wraps); a sequence gives the array.
            ///     Written without parentheses, <c>np.uint64</c> is still the <see cref="DType"/> field
            ///     <see cref="np.uint64"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray uint64([Optional] object x) => ArrTypeNew.Generic(np.uint64, x);

            /// <summary>
            ///     <c>np.ubyte(x)</c>, the class call of NumPy's <c>ubyte</c>: 0 for no argument; a C# integer is packed
            ///     with NumPy's range check (<c>np.ubyte(-1)</c> raises), a float truncated toward zero first, a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely
            ///     (<c>np.ubyte(np.int64(-1))</c> wraps); a sequence gives the array.
            ///     The C <c>unsigned char</c> class, <c>numpy.uint8</c>.
            ///     Written without parentheses, <c>np.ubyte</c> is still the <see cref="DType"/> field
            ///     <see cref="np.ubyte"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray ubyte([Optional] object x) => ArrTypeNew.Generic(np.ubyte, x);

            /// <summary>
            ///     <c>np.ushort(x)</c>, the class call of NumPy's <c>ushort</c>: 0 for no argument; a C# integer is packed
            ///     with NumPy's range check (<c>np.ushort(-1)</c> raises), a float truncated toward zero first, a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely
            ///     (<c>np.ushort(np.int64(-1))</c> wraps); a sequence gives the array.
            ///     The C <c>unsigned short</c> class, <c>numpy.uint16</c>.
            ///     Written without parentheses, <c>np.ushort</c> is still the <see cref="DType"/> field
            ///     <see cref="np.@ushort"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray @ushort([Optional] object x) => ArrTypeNew.Generic(np.@ushort, x);

            /// <summary>
            ///     <c>np.uintc(x)</c>, the class call of NumPy's <c>uintc</c>: 0 for no argument; a C# integer is packed
            ///     with NumPy's range check (<c>np.uintc(-1)</c> raises), a float truncated toward zero first, a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely
            ///     (<c>np.uintc(np.int64(-1))</c> wraps); a sequence gives the array.
            ///     The C <c>unsigned int</c> class, <c>numpy.uintc</c> (uint32 storage).
            ///     Written without parentheses, <c>np.uintc</c> is still the <see cref="DType"/> field
            ///     <see cref="np.uintc"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray uintc([Optional] object x) => ArrTypeNew.Generic(np.uintc, x);

            /// <summary>
            ///     <c>np.uint(x)</c>, the class call of NumPy's <c>uint</c>: 0 for no argument; a C# integer is packed with
            ///     NumPy's range check (<c>np.uint(-1)</c> raises), a float truncated toward zero first, a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely
            ///     (<c>np.uint(np.int64(-1))</c> wraps); a sequence gives the array.
            ///     NumPy 2's unsigned pointer-sized integer, <c>numpy.uint64</c> on a 64-bit host (<c>uintp</c>).
            ///     Written without parentheses, <c>np.uint</c> is still the <see cref="DType"/> field
            ///     <see cref="np.@uint"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray @uint([Optional] object x) => ArrTypeNew.Generic(np.@uint, x);

            /// <summary>
            ///     <c>np.ulong(x)</c>, the class call of NumPy's <c>ulong</c>: 0 for no argument; a C# integer is packed
            ///     with NumPy's range check (<c>np.ulong(-1)</c> raises), a float truncated toward zero first, a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely
            ///     (<c>np.ulong(np.int64(-1))</c> wraps); a sequence gives the array.
            ///     The C <c>unsigned long</c> class: uint32 on Windows, uint64 on Linux and macOS.
            ///     Written without parentheses, <c>np.ulong</c> is still the <see cref="DType"/> field
            ///     <see cref="np.@ulong"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray @ulong([Optional] object x) => ArrTypeNew.Generic(np.@ulong, x);

            /// <summary>
            ///     <c>np.ulonglong(x)</c>, the class call of NumPy's <c>ulonglong</c>: 0 for no argument; a C# integer is
            ///     packed with NumPy's range check (<c>np.ulonglong(-1)</c> raises), a float truncated toward zero first, a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely
            ///     (<c>np.ulonglong(np.int64(-1))</c> wraps); a sequence gives the array.
            ///     The C <c>unsigned long long</c> class (uint64 storage).
            ///     Written without parentheses, <c>np.ulonglong</c> is still the <see cref="DType"/> field
            ///     <see cref="np.ulonglong"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray ulonglong([Optional] object x) => ArrTypeNew.Generic(np.ulonglong, x);

            /// <summary>
            ///     <c>np.uintp(x)</c>, the class call of NumPy's <c>uintp</c>: 0 for no argument; a C# integer is packed
            ///     with NumPy's range check (<c>np.uintp(-1)</c> raises), a float truncated toward zero first, a
            ///     <see cref="string"/> parsed by Python's <c>int()</c>; a NumPy scalar or an array is cast unsafely
            ///     (<c>np.uintp(np.int64(-1))</c> wraps); a sequence gives the array.
            ///     The unsigned pointer-sized integer, <c>numpy.uint64</c> on a 64-bit host.
            ///     Written without parentheses, <c>np.uintp</c> is still the <see cref="DType"/> field
            ///     <see cref="np.uintp"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A C# integer, or the integer part of a float or text, out of the class's
            ///     range (<c>Python integer 300 out of bounds for int8</c>); a float infinity.</exception>
            /// <exception cref="ValueError">A float NaN (<c>cannot convert float NaN to integer</c>); a text that is no
            ///     integer literal (<c>invalid literal for int() with base 10: '1.5'</c>).</exception>
            /// <exception cref="TypeError">null (<c>None</c>) or a complex: CPython's <c>int()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray uintp([Optional] object x) => ArrTypeNew.Generic(np.uintp, x);

            /// <summary>
            ///     <c>np.float16(x)</c>, the class call of NumPy's <c>float16</c>: 0.0 for no argument, nan for null
            ///     (<c>None</c>); a C# number is converted (rounded to the class's precision), a <see cref="string"/>
            ///     parsed by Python's <c>float()</c> (<c>"1e3"</c>, <c>"nan"</c>, <c>" 7 "</c>); a NumPy scalar or an array
            ///     is cast; a sequence gives the array.
            ///     Written without parentheses, <c>np.float16</c> is still the <see cref="DType"/> field
            ///     <see cref="np.float16"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A Python int past the double range (<c>int too large to convert to
            ///     float</c>).</exception>
            /// <exception cref="ValueError">A text that is no float literal (<c>could not convert string to float:
            ///     'abc'</c>).</exception>
            /// <exception cref="TypeError">A complex: CPython's <c>float()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray float16([Optional] object x) => ArrTypeNew.Generic(np.float16, x);

            /// <summary>
            ///     <c>np.half(x)</c>, the class call of NumPy's <c>half</c>: 0.0 for no argument, nan for null
            ///     (<c>None</c>); a C# number is converted (rounded to the class's precision), a <see cref="string"/>
            ///     parsed by Python's <c>float()</c> (<c>"1e3"</c>, <c>"nan"</c>, <c>" 7 "</c>); a NumPy scalar or an array
            ///     is cast; a sequence gives the array.
            ///     The C <c>half</c> class, <c>numpy.float16</c>.
            ///     Written without parentheses, <c>np.half</c> is still the <see cref="DType"/> field
            ///     <see cref="np.half"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A Python int past the double range (<c>int too large to convert to
            ///     float</c>).</exception>
            /// <exception cref="ValueError">A text that is no float literal (<c>could not convert string to float:
            ///     'abc'</c>).</exception>
            /// <exception cref="TypeError">A complex: CPython's <c>float()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray half([Optional] object x) => ArrTypeNew.Generic(np.half, x);

            /// <summary>
            ///     <c>np.float32(x)</c>, the class call of NumPy's <c>float32</c>: 0.0 for no argument, nan for null
            ///     (<c>None</c>); a C# number is converted (rounded to the class's precision), a <see cref="string"/>
            ///     parsed by Python's <c>float()</c> (<c>"1e3"</c>, <c>"nan"</c>, <c>" 7 "</c>); a NumPy scalar or an array
            ///     is cast; a sequence gives the array.
            ///     Written without parentheses, <c>np.float32</c> is still the <see cref="DType"/> field
            ///     <see cref="np.float32"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A Python int past the double range (<c>int too large to convert to
            ///     float</c>).</exception>
            /// <exception cref="ValueError">A text that is no float literal (<c>could not convert string to float:
            ///     'abc'</c>).</exception>
            /// <exception cref="TypeError">A complex: CPython's <c>float()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray float32([Optional] object x) => ArrTypeNew.Generic(np.float32, x);

            /// <summary>
            ///     <c>np.single(x)</c>, the class call of NumPy's <c>single</c>: 0.0 for no argument, nan for null
            ///     (<c>None</c>); a C# number is converted (rounded to the class's precision), a <see cref="string"/>
            ///     parsed by Python's <c>float()</c> (<c>"1e3"</c>, <c>"nan"</c>, <c>" 7 "</c>); a NumPy scalar or an array
            ///     is cast; a sequence gives the array.
            ///     The C <c>float</c> class, <c>numpy.float32</c>.
            ///     Written without parentheses, <c>np.single</c> is still the <see cref="DType"/> field
            ///     <see cref="np.single"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A Python int past the double range (<c>int too large to convert to
            ///     float</c>).</exception>
            /// <exception cref="ValueError">A text that is no float literal (<c>could not convert string to float:
            ///     'abc'</c>).</exception>
            /// <exception cref="TypeError">A complex: CPython's <c>float()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray single([Optional] object x) => ArrTypeNew.Generic(np.single, x);

            /// <summary>
            ///     <c>np.float64(x)</c>, the class call of NumPy's <c>float64</c>: 0.0 for no argument, nan for null
            ///     (<c>None</c>); a C# number is converted (rounded to the class's precision), a <see cref="string"/>
            ///     parsed by Python's <c>float()</c> (<c>"1e3"</c>, <c>"nan"</c>, <c>" 7 "</c>); a NumPy scalar or an array
            ///     is cast; a sequence gives the array.
            ///     Written without parentheses, <c>np.float64</c> is still the <see cref="DType"/> field
            ///     <see cref="np.float64"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A Python int past the double range (<c>int too large to convert to
            ///     float</c>).</exception>
            /// <exception cref="ValueError">A text that is no float literal (<c>could not convert string to float:
            ///     'abc'</c>).</exception>
            /// <exception cref="TypeError">A complex: CPython's <c>float()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray float64([Optional] object x) => ArrTypeNew.Generic(np.float64, x);

            /// <summary>
            ///     <c>np.double(x)</c>, the class call of NumPy's <c>double</c>: 0.0 for no argument, nan for null
            ///     (<c>None</c>); a C# number is converted (rounded to the class's precision), a <see cref="string"/>
            ///     parsed by Python's <c>float()</c> (<c>"1e3"</c>, <c>"nan"</c>, <c>" 7 "</c>); a NumPy scalar or an array
            ///     is cast; a sequence gives the array.
            ///     The C <c>double</c> class, <c>numpy.float64</c>.
            ///     Written without parentheses, <c>np.double</c> is still the <see cref="DType"/> field
            ///     <see cref="np.@double"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A Python int past the double range (<c>int too large to convert to
            ///     float</c>).</exception>
            /// <exception cref="ValueError">A text that is no float literal (<c>could not convert string to float:
            ///     'abc'</c>).</exception>
            /// <exception cref="TypeError">A complex: CPython's <c>float()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray @double([Optional] object x) => ArrTypeNew.Generic(np.@double, x);

            /// <summary>
            ///     <c>np.longdouble(x)</c>, the class call of NumPy's <c>longdouble</c>: 0.0 for no argument, nan for null
            ///     (<c>None</c>); a C# number is converted (rounded to the class's precision), a <see cref="string"/>
            ///     parsed by Python's <c>float()</c> (<c>"1e3"</c>, <c>"nan"</c>, <c>" 7 "</c>); a NumPy scalar or an array
            ///     is cast; a sequence gives the array.
            ///     The C <c>long double</c> class; NumSharp's long double is the 8-byte double on every host (Windows
            ///     NumPy's too).
            ///     Written without parentheses, <c>np.longdouble</c> is still the <see cref="DType"/> field
            ///     <see cref="np.longdouble"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A Python int past the double range (<c>int too large to convert to
            ///     float</c>).</exception>
            /// <exception cref="ValueError">A text that is no float literal (<c>could not convert string to float:
            ///     'abc'</c>).</exception>
            /// <exception cref="TypeError">A complex: CPython's <c>float()</c> refusal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray longdouble([Optional] object x) => ArrTypeNew.Generic(np.longdouble, x);

            /// <summary>
            ///     <c>np.complex128(real, imag)</c>, the class call of NumPy's <c>complex128</c>: 0j for no argument,
            ///     nan+nanj for null; one argument is converted like <c>complex(x)</c> (a <see cref="string"/> by Python's
            ///     <c>complex()</c>, <c>"1+2j"</c>); two real numbers make <c>complex(real, imag)</c>.
            ///     Written without parentheses, <c>np.complex128</c> is still the <see cref="DType"/> field
            ///     <see cref="np.complex128"/>.
            /// </summary>
            /// <param name="real">The value, or the real part when <paramref name="imag"/> is given: a C# number,
            ///     <see cref="System.Numerics.Complex"/> or <see cref="string"/>, an <see cref="NDArray"/>, a C# array.
            ///     Omitting it is the zero; null is <c>None</c>.</param>
            /// <param name="imag">The imaginary part, a real number (or omitted): NumPy's second positional
            ///     argument.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument (one argument only).</returns>
            /// <exception cref="TypeError">Two arguments that are not both real numbers (<c>Could not convert arguments
            ///     into a complex scalar. '((1+1j), 2)' given.</c>), or <paramref name="imag"/> named without
            ///     <paramref name="real"/>.</exception>
            /// <exception cref="OverflowError">A Python int past the double range as a part.</exception>
            /// <exception cref="ValueError">A text that is no complex literal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray complex128([Optional] object real, [Optional] object imag) =>
                ArrTypeNew.CDouble(np.complex128, real, imag);

            /// <summary>
            ///     <c>np.cdouble(real, imag)</c>, the class call of NumPy's <c>complex128</c>: 0j for no argument, nan+nanj
            ///     for null; one argument is converted like <c>complex(x)</c> (a <see cref="string"/> by Python's
            ///     <c>complex()</c>, <c>"1+2j"</c>); two real numbers make <c>complex(real, imag)</c>.
            ///     The C <c>double complex</c> class, <c>numpy.complex128</c>.
            ///     Written without parentheses, <c>np.cdouble</c> is still the <see cref="DType"/> field
            ///     <see cref="np.cdouble"/>.
            /// </summary>
            /// <param name="real">The value, or the real part when <paramref name="imag"/> is given: a C# number,
            ///     <see cref="System.Numerics.Complex"/> or <see cref="string"/>, an <see cref="NDArray"/>, a C# array.
            ///     Omitting it is the zero; null is <c>None</c>.</param>
            /// <param name="imag">The imaginary part, a real number (or omitted): NumPy's second positional
            ///     argument.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument (one argument only).</returns>
            /// <exception cref="TypeError">Two arguments that are not both real numbers (<c>Could not convert arguments
            ///     into a complex scalar. '((1+1j), 2)' given.</c>), or <paramref name="imag"/> named without
            ///     <paramref name="real"/>.</exception>
            /// <exception cref="OverflowError">A Python int past the double range as a part.</exception>
            /// <exception cref="ValueError">A text that is no complex literal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray cdouble([Optional] object real, [Optional] object imag) =>
                ArrTypeNew.CDouble(np.cdouble, real, imag);

            /// <summary>
            ///     <c>np.clongdouble(x)</c>, the class call of NumPy's <c>clongdouble</c>: 0j for no argument, nan+nanj for
            ///     null; a C# number or <see cref="System.Numerics.Complex"/> is converted, a <see cref="string"/> parsed
            ///     by Python's <c>complex()</c>; a sequence gives the array.
            ///     The C <c>long double complex</c> class (complex128 storage). Its NumPy constructor takes one argument,
            ///     unlike <c>complex128</c>'s two.
            ///     Written without parentheses, <c>np.clongdouble</c> is still the <see cref="DType"/> field
            ///     <see cref="np.clongdouble"/>.
            /// </summary>
            /// <param name="x">The value to convert: a C# number, bool or <see cref="string"/>, an <see cref="NDArray"/>
            ///     (a NumPy scalar or an array), or a C# array (a Python list). Omitting it is NumPy's no-argument call
            ///     (the zero), which differs from passing null, NumPy's <c>None</c>.</param>
            /// <returns>The read-only 0-d scalar of the class (NumSharp's NumPy scalar), or the ndarray of a sequence or an
            ///     array argument.</returns>
            /// <exception cref="OverflowError">A Python int past the double range.</exception>
            /// <exception cref="ValueError">A text that is no complex literal.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray clongdouble([Optional] object x) => ArrTypeNew.Generic(np.clongdouble, x);

            /// <summary>
            ///     <c>np.str_(object, encoding, errors)</c>, the class call of <c>numpy.str_</c>: Python's
            ///     <c>str(object)</c> as a <c>&lt;U{len}&gt;</c> scalar - <c>''</c> for no argument, <c>'5'</c> for 5,
            ///     <c>'None'</c> for null, <c>'[1, 2]'</c> for a list (a C# array), <c>"b'ab'"</c> for a <c>byte[]</c> -
            ///     or, with an <paramref name="encoding"/>, the bytes of <paramref name="object"/> decoded.
            ///     Written without parentheses, <c>np.str_</c> is still the <see cref="DType"/> field
            ///     <see cref="np.str_"/>.
            /// </summary>
            /// <param name="object">The object whose <c>str()</c> the scalar holds (omitted: <c>''</c>; null:
            ///     <c>'None'</c>), or, with a codec argument, the bytes to decode (a <c>byte[]</c>, an <c>S</c>
            ///     array).</param>
            /// <param name="encoding">The codec <paramref name="object"/> is decoded with (Python's <c>encoding=</c>),
            ///     or null.</param>
            /// <param name="errors">The decoding error handler (Python's <c>errors=</c>), or null for
            ///     <c>strict</c>.</param>
            /// <returns>The read-only 0-d <c>U</c> scalar of the text, as wide as the text.</returns>
            /// <exception cref="TypeError">An encoding or errors argument with an <paramref name="object"/> that is not
            ///     bytes-like (<c>decoding str is not supported</c>).</exception>
            /// <exception cref="UnicodeDecodeError">Bytes the codec cannot decode under <c>strict</c>.</exception>
            /// <exception cref="LookupError">An unknown codec or error handler.</exception>
            /// <exception cref="ValueError">A Python int of more than 4300 digits (CPython's <c>str()</c>
            ///     limit).</exception>
            public static NDArray str_([Optional] object @object, string encoding = null, string errors = null) =>
                ArrTypeNew.Unicode(@object, encoding, errors);

            /// <summary>
            ///     <c>np.bytes_(source, encoding, errors)</c>, the class call of <c>numpy.bytes_</c>: Python's
            ///     <c>bytes(source)</c> as an <c>S</c> scalar - <c>b''</c> for no argument, five NULs for 5 (a count), the
            ///     bytes of a <c>byte[]</c> or of a list of 0..255 integers, an array's raw buffer - and, where Python's
            ///     <c>bytes()</c> raises, NumPy's conversion (<c>np.bytes_("a")</c> is <c>b'a'</c>, <c>np.bytes_(1.5)</c>
            ///     <c>b'1.5'</c>, <c>np.bytes_(-1)</c> <c>b'-1'</c>); with an <paramref name="encoding"/>, the str
            ///     <paramref name="source"/> encoded.
            ///     Written without parentheses, <c>np.bytes_</c> is still the <see cref="DType"/> field
            ///     <see cref="np.bytes_"/>.
            /// </summary>
            /// <param name="source">The source (omitted: <c>b''</c>; null: <c>b'None'</c>): a count, a
            ///     <c>byte[]</c>, a list of 0..255 integers, an array, or any value NumPy's conversion writes as text; with
            ///     a
            ///     codec argument, the str to encode.</param>
            /// <param name="encoding">The codec a str <paramref name="source"/> is encoded with (Python's
            ///     <c>encoding=</c>), or null.</param>
            /// <param name="errors">The encoding error handler (Python's <c>errors=</c>), or null for
            ///     <c>strict</c>.</param>
            /// <returns>The read-only 0-d <c>S</c> scalar, or the <c>S</c> ndarray of a sequence NumPy's conversion
            ///     converts (<c>np.bytes_(new object[] { 1, 300 })</c> is <c>[b'1', b'300']</c>).</returns>
            /// <exception cref="TypeError">An encoding or errors argument without a str <paramref name="source"/>
            ///     (<c>encoding without a string argument</c>).</exception>
            /// <exception cref="UnicodeEncodeError">Text the codec (ASCII for NumPy's conversion) cannot
            ///     encode.</exception>
            /// <exception cref="LookupError">An unknown codec or error handler.</exception>
            /// <exception cref="NotSupportedException">A value of a type the array conversion does not read.</exception>
            public static NDArray bytes_([Optional] object source, string encoding = null, string errors = null) =>
                ArrTypeNew.String(source, encoding, errors);
        }
    }
}
