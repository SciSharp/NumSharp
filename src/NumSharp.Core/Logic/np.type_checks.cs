using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Determines whether the given object represents a scalar dtype.
        /// </summary>
        /// <param name="rep">The object to check.</param>
        /// <returns>True if rep represents a scalar dtype.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.issctype.html
        /// </remarks>
        /// <example>
        /// <code>
        /// np.issctype(typeof(int))       // True
        /// np.issctype(NPTypeCode.Int32)  // True
        /// np.issctype(typeof(NDArray))   // False
        /// </code>
        /// </example>
        public static bool issctype(object rep)
        {
            if (rep == null)
                return false;

            return rep switch
            {
                NPTypeCode tc => tc != NPTypeCode.Empty && tc != NPTypeCode.String,
                Type t => t.GetTypeCode() is var tc && tc != NPTypeCode.Empty && tc != NPTypeCode.String,
                _ => false
            };
        }

        /// <summary>
        /// Returns True if the descriptor is of a specified category — NumPy 2.x's <c>np.isdtype(dtype, kind)</c> over a
        /// <see cref="DType"/>: <paramref name="kind"/> is one of <c>"bool"</c>, <c>"signed integer"</c>,
        /// <c>"unsigned integer"</c>, <c>"integral"</c>, <c>"real floating"</c>, <c>"complex floating"</c>, <c>"numeric"</c>
        /// (case-sensitive, as in NumPy). The datetime classes belong to none of them (NumPy builds the categories from
        /// <c>sctypes</c>, which exclude <c>datetime64</c>/<c>timedelta64</c>).
        /// <para>
        /// This is the ONE entry for every dtype spelling: a C# <see cref="Type"/> (<c>typeof(int)</c>), an
        /// <see cref="NPTypeCode"/> (<c>NPTypeCode.Int32</c>) or a dtype string all convert implicitly to <see cref="DType"/>
        /// (the earlier <c>NPTypeCode</c>/<c>Type</c> twins accepted <c>issubdtype</c>'s looser vocabulary — <c>"floating"</c>,
        /// <c>"integer"</c> — which NumPy's <c>isdtype</c> rejects; they were folded into this overload).
        /// </para>
        /// </summary>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.isdtype.html — a NumPy 2.0+ function.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.isdtype(NPTypeCode.Int32, "integral")      // True
        /// np.isdtype(typeof(double), "real floating")   // True
        /// np.isdtype(np.int32, "numeric")               // True
        /// </code>
        /// </example>
        /// <exception cref="ValueError"><c>kind argument is a string, but '…' is not a known kind name.</c> — verbatim NumPy.</exception>
        /// <exception cref="TypeError"><paramref name="dtype"/> is a <c>StringDType</c> descriptor (checked FIRST, as NumPy
        ///     preprocesses the dtype before any kind): <c>dtype argument must be a NumPy dtype, but it is a &lt;class
        ///     'numpy.dtypes.StringDType'&gt;.</c></exception>
        public static bool isdtype(DType dtype, string kind)
        {
            if (dtype is null) throw new ArgumentNullException(nameof(dtype));
            ThrowIfNotNumPyDType(dtype);
            CheckIsdtypeKind(kind);

            if (!dtype.Meta.HasStorage)
                return false; // datetime64 / timedelta64: not in NumPy's sctypes categories
            return NPTypeHierarchy.IsSubType(dtype.typecode, kind);
        }

        /// <summary>
        ///     Validates one <c>np.isdtype</c> kind NAME the way NumPy's loop over the kinds does (case-sensitive, before any
        ///     membership is tested), so every overload rejects the same strings with the same texts.
        /// </summary>
        /// <param name="kind">The kind name.</param>
        /// <exception cref="TypeError"><paramref name="kind"/> is null (NumPy's <c>None</c> is neither a dtype nor a string).</exception>
        /// <exception cref="ValueError"><c>kind argument is a string, but '…' is not a known kind name.</c> — verbatim NumPy.</exception>
        private static void CheckIsdtypeKind(string kind)
        {
            if (kind is null) throw new TypeError("kind argument must be comprised of NumPy dtypes or strings only, but is a <class 'NoneType'>.");
            switch (kind)
            {
                case "bool":
                case "signed integer":
                case "unsigned integer":
                case "integral":
                case "real floating":
                case "complex floating":
                case "numeric":
                    break;
                default:
                    throw new ValueError($"kind argument is a string, but '{kind}' is not a known kind name.");
            }
        }

        /// <summary>
        ///     NumPy's <c>np.isdtype(dtype, kind)</c> with a SCALAR CLASS as <paramref name="dtype"/>: a concrete class
        ///     answers as its descriptor does (<c>np.isdtype((NDScalarType)np.int8, "integral")</c> is True), an abstract one
        ///     belongs to none of the named kinds (<c>np.isdtype(np.integer, "integral")</c> is False: NumPy builds each kind
        ///     from the CONCRETE classes of <c>sctypes</c>).
        /// </summary>
        /// <param name="dtype">The scalar class.</param>
        /// <param name="kind">One of <c>"bool"</c>, <c>"signed integer"</c>, <c>"unsigned integer"</c>, <c>"integral"</c>, <c>"real floating"</c>, <c>"complex floating"</c>, <c>"numeric"</c>.</param>
        /// <returns>True when the class is one of the kind's concrete classes.</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> or <paramref name="kind"/> is null (NumPy's texts for <c>None</c>).</exception>
        /// <exception cref="ValueError">An unknown kind name — verbatim NumPy.</exception>
        public static bool isdtype(NDScalarType dtype, string kind)
        {
            CheckIsdtypeDtype(dtype);
            CheckIsdtypeKind(kind);
            // An abstract class is in no kind's set; only then does the concrete class need its descriptor.
            return !dtype.IsAbstract && isdtype(dtype.ToDescriptor(), kind);
        }

        /// <summary>NumPy's tuple form of <see cref="isdtype(NDScalarType, string)"/>: true when the class is in ANY of the kinds (every kind is validated first, as NumPy does).</summary>
        /// <param name="dtype">The scalar class.</param>
        /// <param name="kinds">The kind names.</param>
        /// <returns>True when the class belongs to at least one kind.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="kinds"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="dtype"/> or a kind is null.</exception>
        /// <exception cref="ValueError">An unknown kind name.</exception>
        public static bool isdtype(NDScalarType dtype, params string[] kinds)
        {
            CheckIsdtypeDtype(dtype);
            if (kinds is null) throw new ArgumentNullException(nameof(kinds));
            bool result = false;
            foreach (var kind in kinds)
                result |= isdtype(dtype, kind); // every kind is validated, as NumPy processes the whole tuple first
            return result;
        }

        /// <summary>
        ///     NumPy's <c>np.isdtype(dtype, kind)</c> with a scalar class as <paramref name="kind"/>: true only when the dtype
        ///     IS that class. NumPy compares classes by identity here, NOT by subclass — <c>np.isdtype(np.int64,
        ///     np.integer)</c> is False; use <see cref="issubdtype(DType, NDScalarType)"/> for the hierarchy.
        /// </summary>
        /// <param name="dtype">The dtype (any spelling: a <see cref="DType"/>, <see cref="Type"/>, <see cref="NPTypeCode"/> or dtype string).</param>
        /// <param name="kind">The scalar class.</param>
        /// <returns>True when the dtype's scalar class is <paramref name="kind"/> itself.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="kind"/> is null or Python's built-in <c>str</c>, or <paramref name="dtype"/> is a <c>StringDType</c> descriptor (NumPy's <c>_preprocess_dtype</c> texts).</exception>
        public static bool isdtype(DType dtype, NDScalarType kind)
        {
            if (dtype is null) throw new ArgumentNullException(nameof(dtype));
            ThrowIfNotNumPyDType(dtype);
            CheckIsdtypeKindClass(kind);
            return ReferenceEquals((NDScalarType)dtype, kind);
        }

        /// <summary>
        ///     NumPy's <c>np.isdtype(dtype, kind)</c> over two scalar classes: identity (<c>np.isdtype(np.integer,
        ///     np.integer)</c> is True, <c>np.isdtype(np.integer, np.number)</c> False).
        /// </summary>
        /// <param name="dtype">The scalar class tested.</param>
        /// <param name="kind">The scalar class compared with.</param>
        /// <returns>True when both are the same class.</returns>
        /// <exception cref="TypeError">Either argument is null or Python's built-in <c>str</c> (NumPy's texts).</exception>
        public static bool isdtype(NDScalarType dtype, NDScalarType kind)
        {
            CheckIsdtypeDtype(dtype);
            CheckIsdtypeKindClass(kind);
            return ReferenceEquals(dtype, kind);
        }

        /// <summary>
        ///     NumPy's <c>np.isdtype(dtype, kind)</c> with a scalar class as <paramref name="dtype"/> and a dtype as
        ///     <paramref name="kind"/>: true when the class is the dtype's class (<c>np.isdtype((NDScalarType)np.int64,
        ///     np.int64)</c>); an abstract class never is one.
        /// </summary>
        /// <param name="dtype">The scalar class tested.</param>
        /// <param name="kind">The dtype (any spelling).</param>
        /// <returns>True when <paramref name="dtype"/> is <paramref name="kind"/>'s scalar class.</returns>
        /// <exception cref="TypeError">Either argument is null, <paramref name="dtype"/> is Python's built-in <c>str</c>, or <paramref name="kind"/> is a <c>StringDType</c> descriptor (NumPy's texts).</exception>
        public static bool isdtype(NDScalarType dtype, DType kind)
        {
            CheckIsdtypeDtype(dtype);
            if (kind is null) throw new TypeError("kind argument must be comprised of NumPy dtypes or strings only, but is a <class 'NoneType'>.");
            if (!kind.Meta._legacy)
                throw new TypeError($"kind argument must be comprised of NumPy dtypes or strings only, but is a {kind.Meta}.");
            return ReferenceEquals(dtype, (NDScalarType)kind);
        }

        /// <summary>
        ///     NumPy's tuple form <c>np.isdtype(dtype, (kind, ...))</c> with MIXED kinds — names, dtypes and scalar classes
        ///     (<c>np.isdtype(x, np.int64, "real floating")</c>): true when the dtype is in any of them. A name selects its
        ///     concrete classes, a dtype or a class matches by identity; every kind is validated before the answer, as NumPy
        ///     processes the whole tuple first.
        /// </summary>
        /// <param name="dtype">The dtype (any spelling).</param>
        /// <param name="kinds">Each a kind name (<see cref="string"/>), a <see cref="DType"/>, an <see cref="NDScalarType"/>, a C# element <see cref="Type"/> or an <see cref="NPTypeCode"/>.</param>
        /// <returns>True when the dtype belongs to at least one kind.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> or <paramref name="kinds"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="dtype"/> is a <c>StringDType</c> descriptor (checked first), or a kind is a <c>StringDType</c> descriptor, Python's built-in <c>str</c>, of another type, or null: <c>kind argument must be comprised of NumPy dtypes or strings only, but is a &lt;class 'int'&gt;.</c> (NumPy's text, Python's class name).</exception>
        /// <exception cref="ValueError">An unknown kind name.</exception>
        public static bool isdtype(DType dtype, params object[] kinds)
        {
            if (dtype is null) throw new ArgumentNullException(nameof(dtype));
            ThrowIfNotNumPyDType(dtype); // before any kind, as NumPy preprocesses the dtype first
            return IsdtypeOfKinds(dtype, kinds);
        }

        /// <summary><see cref="isdtype(DType, object[])"/> with a scalar class as <paramref name="dtype"/>: an abstract class matches only itself, never a kind name.</summary>
        /// <param name="dtype">The scalar class.</param>
        /// <param name="kinds">Each a kind name, a <see cref="DType"/>, an <see cref="NDScalarType"/>, a C# element <see cref="Type"/> or an <see cref="NPTypeCode"/>.</param>
        /// <returns>True when the class belongs to at least one kind.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="kinds"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="dtype"/> is null, or a kind of another type.</exception>
        /// <exception cref="ValueError">An unknown kind name.</exception>
        public static bool isdtype(NDScalarType dtype, params object[] kinds)
        {
            CheckIsdtypeDtype(dtype);
            return IsdtypeOfKinds(dtype, kinds);
        }

        /// <summary>
        ///     The body of NumPy's <c>isdtype</c> over a tuple of mixed kinds: every kind is resolved (and refused) in order,
        ///     then the class is tested for membership — so a later invalid kind raises even when an earlier one matched.
        /// </summary>
        /// <param name="sctype">The scalar class of the tested dtype.</param>
        /// <param name="kinds">The kinds.</param>
        /// <returns>True when the class belongs to at least one kind.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="kinds"/> is null.</exception>
        /// <exception cref="TypeError">A kind that is neither a name, a dtype spelling nor a scalar class.</exception>
        /// <exception cref="ValueError">An unknown kind name.</exception>
        private static bool IsdtypeOfKinds(NDScalarType sctype, object[] kinds)
        {
            if (kinds is null) throw new ArgumentNullException(nameof(kinds));
            bool result = false;
            foreach (var kind in kinds)
            {
                switch (kind)
                {
                    case string name:
                        CheckIsdtypeKind(name);
                        // A named kind holds concrete classes only; an abstract class is in none of them.
                        result |= !sctype.IsAbstract && isdtype(sctype.ToDescriptor(), name);
                        break;
                    case NDScalarType cls:
                        CheckIsdtypeKindClass(cls);
                        result |= ReferenceEquals(sctype, cls);
                        break;
                    case DType d:
                        // A StringDType descriptor's class is Python's str, which NumPy refuses as a kind (its type's name).
                        if (!d.Meta._legacy)
                            throw new TypeError($"kind argument must be comprised of NumPy dtypes or strings only, but is a {d.Meta}.");
                        result |= ReferenceEquals(sctype, (NDScalarType)d);
                        break;
                    case NPTypeCode code:
                        result |= ReferenceEquals(sctype, (NDScalarType)DType.From(code));
                        break;
                    case Type t:
                        // NumSharp's dtype spelling of a class: a C# element type (typeof(double)). A type no dtype stores is
                        // not a NumPy dtype, which is NumPy's TypeError for a Python class it does not know.
                        var element = NDScalarType.OfElementType(t);
                        if (element is null)
                            throw new TypeError("kind argument must be comprised of NumPy dtypes or strings only, but is a <class 'type'>.");
                        result |= ReferenceEquals(sctype, element);
                        break;
                    default:
                        throw new TypeError($"kind argument must be comprised of NumPy dtypes or strings only, but is a <class '{(kind is null ? "NoneType" : NumSharp.Backends.Unmanaged.NDBuffer.PythonTypeName(kind))}'>.");
                }
            }

            return result;
        }

        /// <summary>
        ///     NumPy's <c>_preprocess_dtype</c> refusal for a scalar class that is not one of NumPy's: null (Python's
        ///     <c>None</c>) or a Python built-in class (the class of a <c>StringDType</c> descriptor, Python's <c>str</c>).
        /// </summary>
        /// <param name="dtype">The <c>dtype</c> argument of <c>np.isdtype</c>.</param>
        /// <exception cref="TypeError"><paramref name="dtype"/> is null or a built-in class — NumPy's text.</exception>
        private static void CheckIsdtypeDtype(NDScalarType dtype)
        {
            if (dtype is null) throw new TypeError("dtype argument must be a NumPy dtype, but it is a <class 'NoneType'>.");
            // NumPy accepts only its own scalar classes (allTypes): Python's built-in str, the class of a StringDType
            // descriptor, is a class but not a NumPy dtype, and NumPy names the class's type, <class 'type'>.
            if (dtype.__module__ == "builtins")
                throw new TypeError("dtype argument must be a NumPy dtype, but it is a <class 'type'>.");
        }

        /// <summary>NumPy's refusal of a null class, or a Python built-in class, among <c>np.isdtype</c>'s kinds.</summary>
        /// <param name="kind">The kind.</param>
        /// <exception cref="TypeError"><paramref name="kind"/> is null or a built-in class — NumPy's text.</exception>
        private static void CheckIsdtypeKindClass(NDScalarType kind)
        {
            if (kind is null) throw new TypeError("kind argument must be comprised of NumPy dtypes or strings only, but is a <class 'NoneType'>.");
            // The same _preprocess_dtype refusal for a kind: Python's built-in str is not one of NumPy's scalar classes.
            if (kind.__module__ == "builtins")
                throw new TypeError("kind argument must be comprised of NumPy dtypes or strings only, but is a <class 'type'>.");
        }

        /// <summary>
        /// Returns True if the descriptor is of any of the specified categories (NumPy's tuple form).
        /// </summary>
        /// <param name="dtype">The descriptor to test.</param>
        /// <param name="kinds">The category names (see <see cref="isdtype(DType, string)"/>).</param>
        /// <returns>True when <paramref name="dtype"/> is under any of the categories.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> or <paramref name="kinds"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="dtype"/> is a <c>StringDType</c> descriptor (checked first), or an
        ///     entry is null.</exception>
        /// <exception cref="ValueError">An entry is not a known kind name.</exception>
        public static bool isdtype(DType dtype, params string[] kinds)
        {
            if (dtype is null) throw new ArgumentNullException(nameof(dtype));
            ThrowIfNotNumPyDType(dtype); // before any kind, an empty tuple included
            if (kinds is null) throw new ArgumentNullException(nameof(kinds));
            bool result = false;
            foreach (var kind in kinds)
                result |= isdtype(dtype, kind); // every kind is validated, as NumPy processes the whole tuple first
            return result;
        }

        /// <summary>
        /// Returns True if the two descriptors are instances of the same DType class — NumPy's <c>np.isdtype(dtype, other_dtype)</c>
        /// compares the scalar TYPES, so <c>isdtype('M8[s]', 'M8[ns]')</c> is True while <c>isdtype(float32, float64)</c> is False.
        /// </summary>
        /// <param name="dtype">The descriptor to test.</param>
        /// <param name="kind">The descriptor whose class counts as a match.</param>
        /// <returns>True when both are instances of one class.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="kind"/> is null, or either is a <c>StringDType</c> descriptor
        ///     (NumPy's <c>_preprocess_dtype</c> accepts only the builtin scalar types: <c>dtype argument must be a NumPy
        ///     dtype, but it is a …</c> for the dtype, checked first; <c>kind argument must be comprised of NumPy dtypes or
        ///     strings only, but is a …</c> for the kind).</exception>
        public static bool isdtype(DType dtype, DType kind)
        {
            if (dtype is null) throw new ArgumentNullException(nameof(dtype));
            ThrowIfNotNumPyDType(dtype);
            if (kind is null) throw new TypeError("kind argument must be comprised of NumPy dtypes or strings only, but is a <class 'NoneType'>.");
            if (!kind.Meta._legacy)
                throw new TypeError($"kind argument must be comprised of NumPy dtypes or strings only, but is a {kind.Meta}.");
            return ReferenceEquals(dtype.Meta, kind.Meta);
        }

        /// <summary>
        ///     NumPy's <c>_preprocess_dtype</c> refusal for the dtype argument of <c>np.isdtype</c>: only a dtype whose scalar
        ///     type is one of NumPy's builtin scalar types is accepted, and a <c>StringDType</c> descriptor's type is Python's
        ///     own <c>str</c>.
        /// </summary>
        /// <param name="dtype">The dtype argument (not null).</param>
        /// <exception cref="TypeError">A non-legacy (<c>StringDType</c>) descriptor: <c>dtype argument must be a NumPy
        ///     dtype, but it is a &lt;class 'numpy.dtypes.StringDType'&gt;.</c></exception>
        private static void ThrowIfNotNumPyDType(DType dtype)
        {
            if (!dtype.Meta._legacy)
                throw new TypeError($"dtype argument must be a NumPy dtype, but it is a {dtype.Meta}.");
        }

        /// <summary>
        /// Returns True if the descriptor is an instance of ANY of the given dtypes' classes — NumPy's tuple form
        /// <c>np.isdtype(dtype, (np.str_, np.bytes_))</c>. Each entry is compared by class, like
        /// <see cref="isdtype(DType, DType)"/>, so the string LENGTH or the byte order never matter.
        /// </summary>
        /// <param name="dtype">The descriptor to test.</param>
        /// <param name="kinds">The dtypes whose classes count as a match.</param>
        /// <returns>True when <paramref name="dtype"/>'s class is the class of any entry.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> or <paramref name="kinds"/> is null.</exception>
        /// <exception cref="TypeError">An entry is null (NumPy: <c>kind argument must be comprised of NumPy dtypes or strings only</c>),
        ///     or <paramref name="dtype"/> (checked first) or an entry is a <c>StringDType</c> descriptor.</exception>
        public static bool isdtype(DType dtype, params DType[] kinds)
        {
            if (dtype is null) throw new ArgumentNullException(nameof(dtype));
            ThrowIfNotNumPyDType(dtype); // before any kind, an empty tuple included
            if (kinds is null) throw new ArgumentNullException(nameof(kinds));
            bool result = false;
            foreach (var kind in kinds)
                result |= isdtype(dtype, kind); // every entry is validated, as NumPy processes the whole tuple first
            return result;
        }

        /// <summary>
        /// <see cref="isdtype(DType, string)"/> for a dtype STRING (<c>np.isdtype("i4", "integral")</c>). Exists so that a
        /// string first argument binds the dtype grammar rather than NumSharp's string→<see cref="NDArray"/> conversion.
        /// </summary>
        public static bool isdtype(string dtype, string kind)
        {
            if (dtype is null) throw new ArgumentNullException(nameof(dtype));
            return isdtype(np.dtype(dtype), kind);
        }

        /// <summary><see cref="isdtype(DType, string[])"/> for a dtype STRING.</summary>
        public static bool isdtype(string dtype, params string[] kinds)
        {
            if (dtype is null) throw new ArgumentNullException(nameof(dtype));
            return isdtype(np.dtype(dtype), kinds);
        }

        /// <summary>
        /// Returns True if the array's dtype is of a specified category (NumSharp convenience: NumPy's <c>isdtype</c> takes
        /// only a dtype, so this is <c>np.isdtype(arr.dtype, kind)</c>).
        /// </summary>
        /// <param name="arr">The NDArray to check.</param>
        /// <param name="kind">The dtype category.</param>
        /// <returns>True if array's dtype belongs to the specified category.</returns>
        /// <exception cref="ArgumentNullException">Thrown if arr is null.</exception>
        public static bool isdtype(NDArray arr, string kind)
        {
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));
            return isdtype(arr.dtype, kind);
        }

        /// <summary>
        /// Returns True if the array's dtype is of any of the specified categories.
        /// </summary>
        /// <param name="arr">The NDArray to check.</param>
        /// <param name="kinds">Array of dtype categories to check.</param>
        /// <returns>True if array's dtype belongs to any of the specified categories.</returns>
        /// <exception cref="ArgumentNullException">Thrown if arr is null.</exception>
        public static bool isdtype(NDArray arr, string[] kinds)
        {
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));
            return isdtype(arr.dtype, kinds);
        }

        /// <summary>
        /// Determine if a class is a subclass of a second class.
        /// </summary>
        /// <param name="arg1">The dtype to check.</param>
        /// <param name="arg2">The dtype to compare against.</param>
        /// <returns>True if arg1 is a subtype of arg2.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.issubsctype.html
        /// </remarks>
        public static bool issubsctype(NPTypeCode arg1, NPTypeCode arg2)
        {
            return issubdtype(arg1, arg2);
        }

        /// <summary>
        /// Return the string representation of a scalar dtype.
        /// </summary>
        /// <param name="sctype">A scalar type.</param>
        /// <returns>The character code for the type.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.sctype2char.html
        ///
        /// Character codes (NumPy):
        /// '?' - boolean
        /// 'b' - int8 (signed byte)
        /// 'B' - uint8 (unsigned byte)
        /// 'h' - int16 (short)
        /// 'H' - uint16 (unsigned short)
        /// 'i' - int32
        /// 'I' - uint32
        /// 'q' - int64
        /// 'Q' - uint64
        /// 'e' - float16 (Half)
        /// 'f' - float32
        /// 'd' - float64
        /// 'D' - complex128
        /// </remarks>
        /// <example>
        /// <code>
        /// np.sctype2char(NPTypeCode.Int32)   // 'i'
        /// np.sctype2char(NPTypeCode.Double)  // 'd'
        /// </code>
        /// </example>
        public static char sctype2char(NPTypeCode sctype)
        {
            return sctype switch
            {
                NPTypeCode.Boolean => '?',
                NPTypeCode.Byte => 'B',
                NPTypeCode.SByte => 'b',
                NPTypeCode.Int16 => 'h',
                NPTypeCode.UInt16 => 'H',
                NPTypeCode.Int32 => 'i',
                NPTypeCode.UInt32 => 'I',
                NPTypeCode.Int64 => 'q',
                NPTypeCode.UInt64 => 'Q',
                NPTypeCode.Char => 'H',  // Char treated as uint16
                NPTypeCode.Half => 'e',
                NPTypeCode.Single => 'f',
                NPTypeCode.Double => 'd',
                NPTypeCode.Decimal => 'd',  // Closest approximation
                NPTypeCode.Complex => 'D',
                _ => '?'
            };
        }

        /// <summary>
        /// Return the scalar type of highest precision of the same kind as the input.
        /// </summary>
        /// <param name="t">The input scalar type.</param>
        /// <returns>The highest precision type of the same kind.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.maximum_sctype.html
        ///
        /// Uses NPTypeHierarchy for consistent type categorization across all typing functions.
        /// </remarks>
        /// <example>
        /// <code>
        /// np.maximum_sctype(NPTypeCode.Int32)   // Int64
        /// np.maximum_sctype(NPTypeCode.Single)  // Double (or Decimal)
        /// </code>
        /// </example>
        public static DType maximum_sctype(NPTypeCode t)
        {
            return NPTypeHierarchy.GetMaximumType(t);
        }
    }
}
