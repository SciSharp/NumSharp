using System;

namespace NumSharp
{
    public partial class np
    {
        // NumPy 2.4.2, numpy/_core/numerictypes.py:
        //
        //     def issubdtype(arg1, arg2):
        //         if not issubclass_(arg1, generic):
        //             arg1 = dtype(arg1).type
        //         if not issubclass_(arg2, generic):
        //             arg2 = dtype(arg2).type
        //         return issubclass(arg1, arg2)
        //
        // A scalar CLASS (np.integer, np.float64.type) is used as given; anything else is converted to a dtype and replaced
        // by its scalar class; then the subclass relation of NumPy's scalar hierarchy decides. NumSharp spells an abstract
        // class as an NDScalarType (np.integer) and a concrete class by its descriptor (np.float64), so every overload below
        // reduces both arguments to NDScalarType objects (ScalarClassOf) and asks NDScalarType.IsSubclassOf — one rule for
        // every spelling, which is what keeps them consistent with each other.

        /// <summary>
        ///     Returns True if the first argument is a scalar class lower than or equal to the second in NumPy's type
        ///     hierarchy — <c>np.issubdtype(arg1, arg2)</c> for two scalar classes (<c>np.issubdtype(np.integer, np.number)</c>
        ///     is True, <c>np.issubdtype(np.number, np.integer)</c> False).
        /// </summary>
        /// <param name="arg1">The class tested; a <see cref="DType"/> also converts here (its scalar class). Null is NumPy's <c>None</c>, i.e. float64.</param>
        /// <param name="arg2">The class tested against; null is NumPy's <c>None</c>, i.e. float64.</param>
        /// <returns>True when <paramref name="arg2"/> is <paramref name="arg1"/> or one of its base classes.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.issubdtype.html — the abstract classes are
        ///     <see cref="generic"/>, <see cref="number"/>, <see cref="integer"/>, <see cref="signedinteger"/>,
        ///     <see cref="unsignedinteger"/>, <see cref="inexact"/>, <see cref="floating"/>, <see cref="complexfloating"/>,
        ///     <see cref="flexible"/> and <see cref="character"/>. bool is directly under generic (not a number), datetime64
        ///     directly under generic, timedelta64 under signedinteger.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.issubdtype(np.floating, np.inexact)   // True
        /// np.issubdtype(np.integer, np.floating)   // False
        /// </code>
        /// </example>
        public static bool issubdtype(NDScalarType arg1, NDScalarType arg2)
            => ScalarClassOf(arg1).IsSubclassOf(ScalarClassOf(arg2));

        /// <summary>
        ///     Returns True if the dtype <paramref name="arg1"/> is under the scalar class <paramref name="arg2"/> — NumPy's
        ///     most common type test, <c>np.issubdtype(a.dtype, np.integer)</c>. Every dtype spelling binds here: a
        ///     <see cref="DType"/>, a C# <see cref="Type"/> (<c>typeof(int)</c>), an <see cref="NPTypeCode"/> and a dtype
        ///     string (<c>"i4"</c>) all convert to <see cref="DType"/> implicitly.
        /// </summary>
        /// <param name="arg1">The dtype tested (NumPy's <c>dtype(arg1).type</c> is the class compared); null is NumPy's <c>None</c>, i.e. float64.</param>
        /// <param name="arg2">The scalar class, usually an abstract one (<see cref="integer"/>, <see cref="floating"/>, ...); null is float64.</param>
        /// <returns>True when <paramref name="arg1"/>'s scalar class is <paramref name="arg2"/> or derives from it.</returns>
        /// <exception cref="NotSupportedException">A dtype string that is not one (<c>data type 'x' not understood</c>), raised while converting <paramref name="arg1"/>.</exception>
        /// <example>
        /// <code>
        /// np.issubdtype(np.int32, np.integer)             // True
        /// np.issubdtype(np.float32, np.floating)          // True
        /// np.issubdtype(np.bool_, np.number)              // False: bool is not a number in NumPy 2.x
        /// np.issubdtype("m8[s]", np.signedinteger)        // True: timedelta64 is a signed integer
        /// </code>
        /// </example>
        public static bool issubdtype(DType arg1, NDScalarType arg2)
            => ScalarClassOf(arg1).IsSubclassOf(ScalarClassOf(arg2));

        /// <summary>
        ///     Returns True if the scalar class <paramref name="arg1"/> is under the dtype <paramref name="arg2"/>'s class:
        ///     <c>np.issubdtype(np.integer, "i4")</c> is False, since a concrete class has no subclasses but itself.
        /// </summary>
        /// <param name="arg1">The scalar class tested; null is NumPy's <c>None</c>, i.e. float64.</param>
        /// <param name="arg2">The dtype (any spelling: a <see cref="DType"/>, <see cref="Type"/>, <see cref="NPTypeCode"/> or dtype string); null is float64.</param>
        /// <returns>True when <paramref name="arg1"/> is <paramref name="arg2"/>'s scalar class or derives from it.</returns>
        /// <exception cref="NotSupportedException">A dtype string that is not one, raised while converting <paramref name="arg2"/>.</exception>
        public static bool issubdtype(NDScalarType arg1, DType arg2)
            => ScalarClassOf(arg1).IsSubclassOf(ScalarClassOf(arg2));

        /// <summary>
        ///     Returns True if the scalar class of <paramref name="arg1"/> is the scalar class of <paramref name="arg2"/> or a
        ///     subclass of it — NumPy's <c>issubdtype(dtype1, dtype2)</c> over two DESCRIPTORS, which compares their
        ///     <c>.type</c>: instances of one class compare equal whatever their parameters
        ///     (<c>issubdtype('M8[s]', 'M8[ns]')</c> is True, as is <c>issubdtype('&gt;i4', 'i4')</c>), and distinct concrete
        ///     classes are never related (<c>issubdtype(int32, int64)</c> is False).
        /// </summary>
        /// <param name="arg1">The dtype tested; null is NumPy's <c>None</c>, i.e. float64.</param>
        /// <param name="arg2">The dtype tested against; null is float64.</param>
        /// <returns>True when both descriptors belong to the same dtype class.</returns>
        public static bool issubdtype(DType arg1, DType arg2)
            => ScalarClassOf(arg1).IsSubclassOf(ScalarClassOf(arg2));

        /// <summary>
        ///     <see cref="issubdtype(DType, DType)"/> with the second argument a NumPy dtype STRING, parsed by
        ///     <see cref="dtype(string, bool, bool, System.Collections.Generic.IReadOnlyDictionary{string, object})"/> as NumPy
        ///     parses it: <c>np.issubdtype(np.float64, "f8")</c> and <c>np.issubdtype(np.float64, "float")</c> are True,
        ///     <c>np.issubdtype(np.float32, "float")</c> is False (<c>"float"</c> names float64, not the floating family).
        /// </summary>
        /// <param name="arg1">The dtype tested; null is NumPy's <c>None</c>, i.e. float64.</param>
        /// <param name="arg2">A dtype string; null is float64.</param>
        /// <returns>True when both name the same dtype class.</returns>
        /// <exception cref="NotSupportedException">
        ///     <paramref name="arg2"/> is not a dtype string (<c>data type 'integer' not understood</c>, NumPy's text for its
        ///     <c>TypeError</c>). The abstract classes are objects, not strings: <see cref="integer"/>, <see cref="floating"/>.
        /// </exception>
        /// <remarks>
        ///     BREAKING (NumPy 2.x parity): NumSharp used to read this string as an abstract category name
        ///     (<c>"integer"</c>, <c>"floating"</c>, <c>"signed"</c>, <c>"float"</c>), which NumPy rejects or reads as a dtype —
        ///     so <c>issubdtype(np.float32, "float")</c> answered True where NumPy answers False. Spell the category as the
        ///     class: <c>np.issubdtype(x, np.floating)</c>.
        /// </remarks>
        public static bool issubdtype(DType arg1, string arg2)
            => ScalarClassOf(arg1).IsSubclassOf(ScalarClassOf(arg2));

        /// <summary>
        ///     <see cref="issubdtype(DType, string)"/> for a dtype STRING first argument (<c>np.issubdtype("i4", "i8")</c>).
        ///     Exists so that a string first argument binds the dtype grammar rather than NumSharp's string → <see cref="NDArray"/>
        ///     (character array) conversion.
        /// </summary>
        /// <param name="arg1">A dtype string; null is NumPy's <c>None</c>, i.e. float64.</param>
        /// <param name="arg2">A dtype string; null is float64.</param>
        /// <returns>True when both name the same dtype class.</returns>
        /// <exception cref="NotSupportedException">Either string is not a dtype string.</exception>
        public static bool issubdtype(string arg1, string arg2)
            => ScalarClassOf(arg1).IsSubclassOf(ScalarClassOf(arg2));

        /// <summary><see cref="issubdtype(DType, string)"/> for an <see cref="NPTypeCode"/> first argument.</summary>
        /// <param name="arg1">The storage code tested.</param>
        /// <param name="arg2">A dtype string; null is NumPy's <c>None</c>, i.e. float64.</param>
        /// <returns>True when both name the same dtype class.</returns>
        /// <exception cref="NotSupportedException"><paramref name="arg2"/> is not a dtype string, or <paramref name="arg1"/> has no dtype class.</exception>
        public static bool issubdtype(NPTypeCode arg1, string arg2)
            => ScalarClassOf(DType.From(arg1)).IsSubclassOf(ScalarClassOf(arg2));

        /// <summary>
        ///     Returns True if the two storage codes name the same dtype class: <c>issubdtype(int32, int32)</c> is True,
        ///     <c>issubdtype(int32, int64)</c> False — concrete classes are never subclasses of one another.
        /// </summary>
        /// <param name="arg1">The storage code tested.</param>
        /// <param name="arg2">The storage code tested against.</param>
        /// <returns>True when the codes are equal.</returns>
        /// <remarks>For the abstract families use the class objects: <c>np.issubdtype(NPTypeCode.Int32, np.integer)</c>.</remarks>
        public static bool issubdtype(NPTypeCode arg1, NPTypeCode arg2)
        {
            // Concrete type comparison: only same type matches
            // This matches NumPy: np.issubdtype(np.int32, np.int64) returns False
            return arg1 == arg2;
        }

        /// <summary><see cref="issubdtype(DType, string)"/> for a C# element <see cref="Type"/> first argument (<c>typeof(int)</c> is int32).</summary>
        /// <param name="arg1">The C# element type tested; null is NumPy's <c>None</c>, i.e. float64.</param>
        /// <param name="arg2">A dtype string; null is float64.</param>
        /// <returns>True when both name the same dtype class.</returns>
        /// <exception cref="NotSupportedException"><paramref name="arg2"/> is not a dtype string, or <paramref name="arg1"/> is no dtype's element type.</exception>
        public static bool issubdtype(Type arg1, string arg2)
            => ScalarClassOf(arg1 is null ? null : DType.From(arg1)).IsSubclassOf(ScalarClassOf(arg2));

        /// <summary>
        ///     NumSharp convenience: <see cref="issubdtype(DType, string)"/> over the array's dtype
        ///     (<c>np.issubdtype(arr, "f8")</c> is <c>np.issubdtype(arr.dtype, "f8")</c>). NumPy itself refuses an array
        ///     argument (<c>Cannot construct a dtype from an array</c>); write <c>arr.dtype</c> in ported code.
        /// </summary>
        /// <param name="arr">The array whose dtype is tested.</param>
        /// <param name="arg2">A dtype string; null is NumPy's <c>None</c>, i.e. float64.</param>
        /// <returns>True when the array's dtype and <paramref name="arg2"/> name the same dtype class.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> is null.</exception>
        /// <exception cref="NotSupportedException"><paramref name="arg2"/> is not a dtype string.</exception>
        public static bool issubdtype(NDArray arr, string arg2)
        {
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));
            return issubdtype(arr.dtype, arg2);
        }

        /// <summary>
        ///     Returns True if the two C# element types are the same dtype's: <c>issubdtype(typeof(int), typeof(long))</c> is
        ///     False, as NumPy's <c>issubdtype(np.int32, np.int64)</c> is.
        /// </summary>
        /// <param name="arg1">The first C# element type.</param>
        /// <param name="arg2">The second C# element type.</param>
        /// <returns>True when both map to the same storage code.</returns>
        public static bool issubdtype(Type arg1, Type arg2)
        {
            return issubdtype(arg1.GetTypeCode(), arg2.GetTypeCode());
        }

        /// <summary>
        ///     The scalar class an <c>issubdtype</c> argument stands for: a NumPy class itself; for a Python built-in class
        ///     the class NumPy reads it as (<c>if not issubclass_(arg, generic): arg = dtype(arg).type</c>, so <c>float</c>
        ///     is <c>numpy.float64</c> and <c>np.issubdtype(bool, int)</c> compares <c>numpy.bool</c> with
        ///     <c>numpy.int64</c>); float64's for null — NumPy's <c>None</c>, which <c>dtype(None)</c> turns into float64
        ///     (<c>np.issubdtype(None, np.floating)</c> is True).
        /// </summary>
        /// <param name="sctype">The scalar class, or null.</param>
        /// <returns>The class to compare.</returns>
        private static NDScalarType ScalarClassOf(NDScalarType sctype) => sctype is null ? NDScalarType.Of(DTypeRegistry.Double) : sctype.NumPyClass;

        /// <summary>The scalar class of a descriptor (NumPy's <c>dtype(arg).type</c>), float64's for null (NumPy's <c>None</c>).</summary>
        /// <param name="dtype">The descriptor, or null.</param>
        /// <returns>The class to compare.</returns>
        private static NDScalarType ScalarClassOf(DType dtype) => NDScalarType.Of((dtype ?? DType.Double).Meta);

        /// <summary>The scalar class of a dtype string (NumPy's <c>dtype(arg).type</c>), float64's for null (NumPy's <c>None</c>).</summary>
        /// <param name="dtype">The dtype string, or null.</param>
        /// <returns>The class to compare.</returns>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> is not a dtype string (<c>data type 'x' not understood</c>).</exception>
        private static NDScalarType ScalarClassOf(string dtype) => NDScalarType.Of((dtype is null ? DType.Double : np.dtype(dtype)).Meta);
    }
}
