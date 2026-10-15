using System;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    /// <summary>
    ///     The instance parameters of a NEP 55 <c>StringDType</c> descriptor (<see cref="VStringDTypeMeta"/>) — NumPy's
    ///     <c>PyArray_StringDTypeObject</c> fields <c>na_object</c> and <c>coerce</c>, plus the values NumPy derives from the
    ///     missing-data object once, when the descriptor is built (<c>new_stringdtype_instance</c>): whether it is a string
    ///     (<c>has_string_na</c>) or NaN-like (<c>has_nan_na</c>), its <c>str()</c> (<c>na_name</c>) and, for a string,
    ///     the text a missing value reads as (<c>default_string</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>NumPy's names.</b> Each member that ports a field of <c>PyArray_StringDTypeObject</c>
    ///     (<c>numpy/_core/include/numpy/ndarraytypes.h</c>) carries the field's name. Not ported: <c>base</c> (the
    ///     <c>PyArray_Descr</c> header, which is the <see cref="DType"/> that holds these parameters), <c>array_owned</c>
    ///     and <c>allocator</c> (NumPy gives every array its own descriptor copy holding the string allocator; NumSharp keeps
    ///     the allocator on the array's buffer, <c>VStringAllocator</c>, so a descriptor is never owned by an array).
    ///     <see cref="HasNAObject"/> has no field of its own: NumPy reads "absent" as a null <c>na_object</c> pointer.
    ///     </para>
    ///     <para>
    ///     <b>Immutable, shared.</b> A descriptor never changes its parameters, so the derived fields are computed exactly
    ///     once and every later reader (the cast loops, the element stores of P1.5 / P1.6) reads plain fields instead of
    ///     re-running Python protocols per element — NumPy caches them on the descriptor for the same reason.
    ///     </para>
    ///     <para>
    ///     <b>"Not given" is not <c>None</c>.</b> NumPy distinguishes a descriptor without a missing-data object (the
    ///     attribute does not exist) from one whose missing-data object is <c>None</c>. <see cref="HasNAObject"/> carries
    ///     that bit; <see cref="na_object"/> is <see langword="null"/> both for "absent" and for <c>None</c>.
    ///     </para>
    ///     <para>
    ///     <b>A <see cref="char"/> is stored as a one-character string.</b> Every NumSharp string store reads a C#
    ///     <see cref="char"/> as Python's one-character <c>str</c>, so a char missing-data object is a STRING missing-data
    ///     object here, normalised to a <see cref="string"/> so that equality, hashing and <c>repr()</c> treat
    ///     <c>'?'</c> and <c>"?"</c> as the one Python object they stand for.
    ///     </para>
    /// </remarks>
    internal sealed class VStringParameters
    {
        /// <summary>The parameters of <c>StringDType()</c>: no missing-data object, coercion on.</summary>
        internal static readonly VStringParameters Default = new VStringParameters(false, null, true);

        /// <summary>
        ///     Builds the parameters and derives NumPy's cached fields from the missing-data object.
        /// </summary>
        /// <param name="hasNAObject">True when a missing-data object was given (even <c>None</c>).</param>
        /// <param name="naObject">The missing-data object (ignored when <paramref name="hasNAObject"/> is false); a
        ///     <see cref="char"/> is normalised to a one-character <see cref="string"/>.</param>
        /// <param name="coerce">NumPy's <c>coerce</c>: whether a non-<c>str</c> value is stored as its <c>str()</c>.</param>
        /// <exception cref="UnicodeEncodeError">The missing-data object is a string holding a lone surrogate, or its
        ///     <c>str()</c> does: NumPy stores both as UTF-8 (<c>PyUnicode_AsUTF8AndSize</c>) and raises CPython's text,
        ///     <c>'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed</c>.</exception>
        /// <exception cref="ValueError">The missing-data object is a Python int (<see cref="System.Numerics.BigInteger"/>) of
        ///     more than 4300 decimal digits: NumPy takes its <c>str()</c> for <c>na_name</c> here, and CPython refuses it
        ///     (<c>Exceeds the limit (4300 digits) for integer string conversion; ...</c>), so
        ///     <c>StringDType(na_object=10**4300)</c> cannot be built in either library.</exception>
        internal VStringParameters(bool hasNAObject, object naObject, bool coerce)
        {
            HasNAObject = hasNAObject;
            na_object = hasNAObject ? (naObject is char c ? c.ToString() : naObject) : null;
            // The parameter carries the field's name (NumPy's spelling), so the field is written through `this.`.
            this.coerce = coerce;
            if (!hasNAObject)
            {
                na_name = string.Empty;
                default_string = string.Empty;
                return;
            }
            // first check for a string
            if (na_object is string s)
            {
                has_string_na = true;
                // NumPy keeps default_string as UTF-8 (PyUnicode_AsUTF8AndSize), so a lone surrogate fails the
                // descriptor's construction with CPython's encode error; the count is all the check needs.
                NDUtf8.GetByteCount(s);
                default_string = s;
            }
            else
            {
                // treat as nan-like if != comparison returns a object whose truth value raises an error (pd.NA) or a
                // truthy value (e.g. a NaN-like object)
                has_nan_na = NDPyObject.IsNaNLike(na_object);
                default_string = string.Empty;
            }
            na_name = NDPyObject.Str(na_object);
            // na_name is UTF-8 in NumPy too: an object whose str() holds a lone surrogate fails here (a string was
            // already checked above; its str() is itself).
            if (!has_string_na)
                NDUtf8.GetByteCount(na_name);
        }

        /// <summary>True when the descriptor has a missing-data object (NumPy: the <c>na_object</c> attribute exists).</summary>
        internal bool HasNAObject { get; }

        /// <summary>
        ///     NumPy's <c>na_object</c>: the missing-data object (<see langword="null"/> for <c>None</c> and when there is
        ///     none — <see cref="HasNAObject"/> tells the two apart).
        /// </summary>
        internal object na_object { get; }

        /// <summary>NumPy's <c>coerce</c>: a non-<c>str</c> value is stored as its <c>str()</c> (true) or refused (false).</summary>
        internal bool coerce { get; }

        /// <summary>NumPy's <c>has_nan_na</c>: the missing-data object is NaN-like (<c>na != na</c> is truthy or raises).</summary>
        internal bool has_nan_na { get; }

        /// <summary>NumPy's <c>has_string_na</c>: the missing-data object is a <c>str</c>.</summary>
        internal bool has_string_na { get; }

        /// <summary>NumPy's <c>na_name</c>: the missing-data object's <c>str()</c> (<c>'None'</c>, <c>'nan'</c>, <c>'NA'</c>);
        ///     empty when there is none.</summary>
        internal string na_name { get; }

        /// <summary>NumPy's <c>default_string</c>: for a string missing-data object, that string (what a missing value
        ///     reads as in a string operation); otherwise empty.</summary>
        internal string default_string { get; }

        /// <summary>
        ///     NumPy's <c>_eq_comparison</c>: equal when both coerce alike and their missing-data objects are equal by
        ///     <c>na_eq_cmp</c> (two absent objects are equal; an absent one never equals a present one, not even <c>None</c>).
        /// </summary>
        /// <param name="a">One parameter set (null for a descriptor of another class).</param>
        /// <param name="b">The other.</param>
        /// <returns>True when two descriptors with these parameters are equal.</returns>
        internal static bool AreEqual(VStringParameters a, VStringParameters b)
        {
            if (ReferenceEquals(a, b))
                return true;
            if (a is null || b is null)
                return false;
            if (a.coerce != b.coerce)
                return false;
            if (a.HasNAObject != b.HasNAObject)
                return false; // na_eq_cmp(NULL, x) is 0 for any present x
            return !a.HasNAObject || NDPyObject.Equal(a.na_object, b.na_object);
        }

        /// <summary>
        ///     NumPy's <c>stringdtype_compatible_na</c>: two missing-data objects are compatible when at most one is present or
        ///     both are equal; the common instance then takes the present one (the first when both are).
        /// </summary>
        /// <param name="a">The first descriptor's parameters.</param>
        /// <param name="b">The second descriptor's parameters.</param>
        /// <param name="outHas">Whether the common instance has a missing-data object.</param>
        /// <param name="outNA">The common instance's missing-data object.</param>
        /// <returns>False when both are present and unequal (NumPy raises).</returns>
        internal static bool CompatibleNA(VStringParameters a, VStringParameters b, out bool outHas, out object outNA)
        {
            if (a.HasNAObject && b.HasNAObject && !NDPyObject.Equal(a.na_object, b.na_object))
            {
                outHas = false;
                outNA = null;
                return false;
            }
            outHas = a.HasNAObject || b.HasNAObject;
            outNA = a.HasNAObject ? a.na_object : b.na_object;
            return true;
        }

        /// <summary>A hash consistent with <see cref="AreEqual"/> (NumPy hashes the tuple <c>(coerce, na_object)</c>).</summary>
        /// <returns>The hash.</returns>
        internal int Hash() => HasNAObject ? HashCode.Combine(coerce, NDPyObject.Hash(na_object)) : HashCode.Combine(coerce);

        /// <summary>
        ///     NumPy's <c>stringdtype_repr</c> — both the descriptor's <c>repr()</c> and its <c>str()</c>:
        ///     <c>StringDType()</c>, <c>StringDType(na_object=None)</c>, <c>StringDType(coerce=False)</c>,
        ///     <c>StringDType(na_object='NA', coerce=False)</c>.
        /// </summary>
        /// <returns>The representation.</returns>
        internal string Repr()
        {
            if (HasNAObject && !coerce)
                return "StringDType(na_object=" + NDPyObject.Repr(na_object) + ", coerce=False)";
            if (HasNAObject)
                return "StringDType(na_object=" + NDPyObject.Repr(na_object) + ")";
            return coerce ? "StringDType()" : "StringDType(coerce=False)";
        }
    }
}
