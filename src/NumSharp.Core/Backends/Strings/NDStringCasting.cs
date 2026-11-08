using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The descriptor-level casting rules of the fixed-width string lane: <c>ndarray.astype</c> when either side is a
    ///     string dtype (NumPy's <c>array_astype</c> over DESCRIPTORS — the width is part of the dtype, so the type-code cast
    ///     cannot express it) and NumPy's invalid-cast error (<c>npy_set_invalid_cast_error</c>), shared with
    ///     <c>np.copyto</c>.
    /// </summary>
    /// <remarks>
    ///     A cast between a string dtype and a number (or a <c>datetime64</c>/<c>timedelta64</c>) is checked against the
    ///     casting rule exactly like NumPy — so a refused rule raises NumPy's TypeError text — and an allowed one formats the
    ///     numbers as text (their scalar <c>str()</c>) or parses the text (<c>int()</c> / <c>float()</c> / <c>complex()</c>,
    ///     <see cref="NDTextConversion"/>), with NumPy's errors for text that does not parse.
    /// </remarks>
    internal static class NDStringCasting
    {
        /// <summary>
        ///     <c>a.astype(dtype, order, casting, copy)</c> with a string array or a string target dtype — NumPy's
        ///     <c>array_astype</c>, statement for statement: an UNSIZED target (<c>'U'</c>, <c>'S'</c>, <c>np.str_</c>) is
        ///     adapted to the array (<c>PyArray_AdaptDescriptorToArray</c>: <c>U3 → 'U'</c> is <c>&lt;U3</c>, <c>S3 → 'U'</c> is
        ///     <c>&lt;U3</c>, <c>int64 → 'U'</c> is <c>&lt;U21</c>); <c>copy=false</c> returns the array itself when the layout
        ///     satisfies <paramref name="order"/> and the cast is a no-cast VIEW (same class, width and byte order); otherwise
        ///     the casting rule is checked (<c>PyArray_CanCastArrayTo</c>) and a fresh array of the adapted dtype is filled by
        ///     the string copy route (<c>PyArray_NewLikeArray</c> + <c>PyArray_CopyInto</c>).
        /// </summary>
        /// <param name="src">The array (string or numeric).</param>
        /// <param name="dtype">The requested dtype (a string dtype, or any dtype when <paramref name="src"/> is a string array).</param>
        /// <param name="copy">False to return <paramref name="src"/> itself when no conversion is needed.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c> (either case): the result's memory layout.</param>
        /// <param name="casting">The casting rule; null or empty is NumSharp's astype default, <c>'unsafe'</c>. NumPy 2.4's
        ///     <c>'same_value'</c> is accepted as NumPy's astype accepts it, and — as in NumPy — no string cast satisfies it.</param>
        /// <returns><paramref name="src"/> itself, or a new array of the adapted dtype.</returns>
        /// <exception cref="ValueError"><paramref name="casting"/> is not one of NumPy's astype names (<c>casting must be one of
        ///     'no', 'equiv', 'safe', 'same_kind', 'unsafe', 'same_value' (got '…')</c>).</exception>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of C/F/A/K (<c>order must be one of 'C', 'F',
        ///     'A', or 'K' (got '…')</c>), reported before an unknown casting name.</exception>
        /// <exception cref="InvalidCastException">The cast is not allowed under <paramref name="casting"/> — NumPy's
        ///     TypeError text, <c>Cannot cast array data from dtype('&lt;U5') to dtype('&lt;U3') according to the rule 'safe'</c>
        ///     (<c>Cannot cast scalar from …</c> for a 0-d array).</exception>
        /// <exception cref="UnicodeEncodeError"><c>U → S</c> meets a code point at or above 128.</exception>
        /// <exception cref="UnicodeDecodeError"><c>S → U</c> meets a byte at or above <c>0x80</c>.</exception>
        /// <exception cref="SystemError"><c>U → S</c>, <c>U → bool</c> or <c>U → number</c> meets a unit above
        ///     <c>U+10FFFF</c>.</exception>
        /// <exception cref="ValueError">Text → number meets a text that is not a number of the target's kind (NumPy's
        ///     <c>invalid literal for int() with base 10: np.str_('x')</c>, <c>could not convert string to float: ...</c>,
        ///     <c>complex() arg is a malformed string</c>) or an integer text of more than 4300 digits — the FIRST such
        ///     element in NumPy's raw assignment order.</exception>
        /// <exception cref="OverflowException">Text → integer meets a value outside the target (NumPy's LP64 texts:
        ///     <c>Python integer 300 out of bounds for int8</c>, <c>Python int too large to convert to C long</c>).</exception>
        public static NDArray AsType(NDArray src, DType dtype, bool copy, char order, string casting)
            => AsType(src, dtype, copy, order, casting, beforeAllocate: null);

        /// <summary>
        ///     <see cref="AsType(NDArray, DType, bool, char, string)"/> with a hook that runs where NumPy allocates the
        ///     result: after the adapted dtype, the <c>copy=False</c> shortcut and the casting rule, before any element is
        ///     converted. An ndarray subclass whose <c>__array_finalize__</c> refuses a dtype (<c>numpy.char.chararray</c>)
        ///     raises there, so its error comes after the casting rule's and before a parse error of the conversion.
        /// </summary>
        /// <param name="src">The array (string or numeric).</param>
        /// <param name="dtype">The requested dtype.</param>
        /// <param name="copy">False to return <paramref name="src"/> itself when no conversion is needed (the hook does not
        ///     run then: NumPy allocates nothing).</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <param name="casting">The casting rule; null or empty is <c>'unsafe'</c>.</param>
        /// <param name="beforeAllocate">Receives the ADAPTED result dtype (an unsized request resolved to its width) right
        ///     before the result is allocated; it may throw. <c>null</c> runs nothing.</param>
        /// <returns><paramref name="src"/> itself, or a new array of the adapted dtype.</returns>
        /// <exception cref="ValueError">An unknown casting name, or whatever <paramref name="beforeAllocate"/> throws.</exception>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of C/F/A/K (<c>order must be one of 'C', 'F',
        ///     'A', or 'K' (got '…')</c>), reported before an unknown casting name.</exception>
        /// <exception cref="InvalidCastException">The cast is not allowed under <paramref name="casting"/>.</exception>
        /// <exception cref="UnicodeEncodeError"><c>U → S</c> meets a code point at or above 128.</exception>
        /// <exception cref="UnicodeDecodeError"><c>S → U</c> meets a byte at or above <c>0x80</c>.</exception>
        /// <exception cref="SystemError">A unit above <c>U+10FFFF</c> where the cast decodes text.</exception>
        /// <exception cref="OverflowException">Text → integer meets a value outside the target.</exception>
        internal static NDArray AsType(NDArray src, DType dtype, bool copy, char order, string casting, Action<DType> beforeAllocate)
        {
            // npy_parse_arguments runs astype's converters in its parameter order - dtype, order, casting - before the dtype
            // is adapted: an unknown order is reported before an unknown casting name. astype's casting converter also takes
            // NumPy 2.4's 'same_value', which no string cast implements (every one fails the rule below).
            OrderResolver.Resolve(order, src.Shape); // validates the order character (C/F/A/K)
            var rule = DTypeCasting.ParseCastingSameValue(casting, out bool sameValue);

            var from = src.dtype;
            var to = DTypePromotion.AdaptDescriptorToArray(from, dtype);

            // "If the memory layout matches and, data types are equivalent, ... and if the cast says view are possible,
            // we can skip the copy." — PyArray_SafeCast(dtype, self->descr, NO_CASTING) with a view offset.
            if (!copy && OrderSatisfied(order, src.Shape))
            {
                // StringDType: "views are only legal between descriptors that share allocators (e.g. the same object)"
                // (string_to_string_resolve_descriptors) — the array's own descriptor instance, never an equal fresh one.
                if (from.Meta is VStringDTypeMeta || to.Meta is VStringDTypeMeta)
                {
                    if (ReferenceEquals(from, to))
                        return src;
                }
                else
                {
                    var safety = DTypeCasting.GetCastInfo(to, from, null, out long viewOffset);
                    if (safety == NPY_CASTING.NPY_NO_CASTING && viewOffset != ArrayMethod.NoView)
                        return src;
                }
            }

            // 'same_value' needs a cast loop that checks every value round-trips; the string casts (and the casts between
            // text and numbers) have none, so NumPy refuses them all under that rule — the no-cast shortcut above included.
            if (sameValue)
                throw new InvalidCastException(
                    $"Cannot cast {(src.ndim == 0 ? "scalar" : "array data")} from {from.ToString(true)} to {to.ToString(true)} " +
                    $"according to the rule '{DTypeCasting.SameValueName}'");

            if (!DTypeCasting.CanCastTypeTo(from, to, rule))
                throw DTypeCasting.InvalidCastError(from, to, rule, src.ndim == 0);

            // PyArray_NewLikeArray allocates the result here, as the source's subclass: a subclass's finalize rule
            // (chararray's 'Can only create a chararray from string data.') raises before any element converts.
            beforeAllocate?.Invoke(to);

            // Text to bool is a string's truth value (NumPy's legacy string-to-bool cast: a non-empty string is True, a U
            // unit above U+10FFFF raises its SystemError), laid out by the requested order like any astype result.
            if (from.Meta is StringDTypeMeta && to.typecode == NPTypeCode.Boolean)
                return NDStringTruth.Truth(src, NDStringTruth.Flavor.Cast, order);

            // Every other pair — text to text, numbers to text, text to numbers — is a fresh array of the adapted dtype
            // filled by the string copy route (NumPy's PyArray_NewLikeArray + PyArray_CopyInto): numbers are formatted as
            // their scalar's str(), texts parsed by int() / float() / complex() (NDTextConversion). An empty array
            // allocates the result and converts nothing (np.zeros(0, 'U2').astype(np.float64) is an empty float64 array).
            return NDIter.CopyAs(to, src, order);
        }

        /// <summary>
        ///     The layout half of astype's <c>copy=False</c> shortcut: <c>'K'</c> always, <c>'A'</c> when the array is C- or
        ///     F-contiguous, <c>'C'</c> when it is C-contiguous, <c>'F'</c> when it is F-contiguous.
        /// </summary>
        /// <param name="order">The (validated) order character.</param>
        /// <param name="shape">The array's shape.</param>
        /// <returns>True when the array's own layout satisfies the requested order.</returns>
        internal static bool OrderSatisfied(char order, Shape shape)
        {
            switch (order)
            {
                case 'K':
                case 'k':
                    return true;
                case 'A':
                case 'a':
                    return shape.IsContiguous || shape.IsFContiguous;
                case 'C':
                case 'c':
                    return shape.IsContiguous;
                default:
                    return shape.IsFContiguous;
            }
        }
    }
}
