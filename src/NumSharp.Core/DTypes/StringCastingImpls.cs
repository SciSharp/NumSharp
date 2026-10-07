using System;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     The cast between two descriptors of ONE string class (<c>S → S</c> or <c>U → U</c>) — NumPy's
    ///     <c>string_to_string_cast</c> (<c>convert_datatype.c</c>, <c>string_to_string_resolve_descriptors</c>): a longer
    ///     destination is <c>safe</c> (zero padding, never a view), a shorter one <c>same_kind</c> (truncation; a view at
    ///     offset 0 when the byte order matches), an equal one a no-op view (<c>no</c>) or a <c>U</c> byte swap
    ///     (<c>equiv</c>).
    /// </summary>
    /// <remarks>
    ///     The class-level minimal safety is <c>unsafe</c> ("must resolve"), so <c>can_cast</c> for any stricter rule always
    ///     asks <see cref="ResolveDescriptors"/>. The data movement itself is NumSharp's unit-view copy (truncate or
    ///     zero-pad per element over the <c>uint8</c>/<c>uint32</c> lanes), not a loop handed out by this object.
    /// </remarks>
    public sealed class StringToStringCastingImpl : CastingImpl
    {
        /// <summary>Builds the within-class cast of <paramref name="meta"/>.</summary>
        /// <param name="meta">The string class (<c>S</c> or <c>U</c>) on both sides.</param>
        internal StringToStringCastingImpl(StringDTypeMeta meta)
            : base("string_to_string_cast", NPY_CASTING.NPY_UNSAFE_CASTING,
                NDArrayMethodFlags.REQUIRES_PYAPI | NDArrayMethodFlags.NO_FLOATINGPOINT_ERRORS | NDArrayMethodFlags.SUPPORTS_UNALIGNED,
                meta, meta)
        {
        }

        /// <summary>
        ///     NumPy's <c>string_to_string_resolve_descriptors</c>: the input is used as given; a missing output is the
        ///     input made canonical (same length). The safety is decided by the two item sizes and byte orders.
        /// </summary>
        /// <param name="givenDescrs">The source descriptor and the requested destination descriptor (null = "same length").</param>
        /// <param name="loopDescrs">Receives the descriptors the copy runs on.</param>
        /// <param name="viewOffset">0 when the destination can be a view of the source (it fits and the byte order
        /// matches), else <see cref="ArrayMethod.NoView"/>.</param>
        /// <returns><c>safe</c> (longer), <c>same_kind</c> (shorter), <c>no</c> (identical layout) or <c>equiv</c> (a byte swap).</returns>
        public override NPY_CASTING ResolveDescriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            loopDescrs[0] = givenDescrs[0];
            loopDescrs[1] = givenDescrs[1] ?? From.EnsureCanonical(loopDescrs[0]);
            viewOffset = NoView;

            if (loopDescrs[0].itemsize < loopDescrs[1].itemsize)
            {
                /* New string is longer: safe but cannot be a view */
                return NPY_CASTING.NPY_SAFE_CASTING;
            }

            /* New string fits into old: if the byte-order matches can be a view */
            bool notSwapped = loopDescrs[0].isnative == loopDescrs[1].isnative;
            if (notSwapped)
                viewOffset = 0;

            if (loopDescrs[0].itemsize > loopDescrs[1].itemsize)
                return NPY_CASTING.NPY_SAME_KIND_CASTING;
            /* The strings have the same length: */
            return notSwapped ? NPY_CASTING.NPY_NO_CASTING : NPY_CASTING.NPY_EQUIV_CASTING;
        }
    }

    /// <summary>
    ///     A cast INTO a string class from any other non-datetime class — a numeric class, or the other string class —
    ///     NumPy's <c>legacy_cast_to_string</c> (<c>add_other_to_and_from_string_cast</c> +
    ///     <c>cast_to_string_resolve_descriptors</c>). Its class-level safety is <c>same_kind</c> ("safe when the
    ///     destination is long enough") for every source but <c>U</c>, and <c>unsafe</c> for <c>U → S</c> (a code point
    ///     above 127 cannot be stored in a byte string).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The length a source needs is NumPy's <c>REQUIRED_STR_LEN</c> estimate, in characters: 5 for <c>bool</c>
    ///     (<c>False</c>); <c>3/5/10/20</c> for 1/2/4/8-byte unsigned integers and one more for the sign of a signed one;
    ///     32 for every float; 64 for complex; the source's own length for <c>S</c> and <c>U</c>. NumSharp's two extra
    ///     numeric classes take the numeric class they masquerade as: <c>char</c> sizes as <c>uint16</c> (5) and
    ///     <c>decimal</c> as a float (32, enough for the 31 characters of <c>-0.0000000000000000000000000001</c>).
    ///     </para>
    ///     <para>
    ///     Only the descriptor resolution exists in this phase: the strided loops that format a number as text arrive with
    ///     the computational phase. The <c>S ↔ U</c> casts run on NumSharp's unit-view kernels (ASCII-checked).
    ///     </para>
    /// </remarks>
    public sealed class ToStringCastingImpl : CastingImpl
    {
        /// <summary>Builds the cast from <paramref name="from"/> into the string class <paramref name="to"/>.</summary>
        /// <param name="from">The source class: a numeric storage class or the OTHER string class.</param>
        /// <param name="to">The destination string class.</param>
        internal ToStringCastingImpl(DTypeMeta from, StringDTypeMeta to)
            : base("legacy_cast_to_string",
                /* Almost everything can be same-kind cast to string (except unicode) */
                from is StringDTypeMeta fromString && fromString.IsUnicode ? NPY_CASTING.NPY_UNSAFE_CASTING : NPY_CASTING.NPY_SAME_KIND_CASTING,
                NDArrayMethodFlags.REQUIRES_PYAPI | NDArrayMethodFlags.NO_FLOATINGPOINT_ERRORS, from, to)
        {
        }

        /// <summary>
        ///     NumPy's <c>REQUIRED_STR_LEN</c> table — the decimal digits of the largest value of an unsigned integer of
        ///     each byte width (index = item size): <c>uint8</c> 255 → 3, <c>uint16</c> 65535 → 5, <c>uint32</c> → 10,
        ///     <c>uint64</c> → 20.
        /// </summary>
        private static readonly long[] RequiredStrLen = { 0, 3, 5, 10, 10, 20, 20, 20, 20 };

        /// <summary>
        ///     The characters a value of <paramref name="from"/> needs as text — the <c>switch</c> at the top of
        ///     <c>cast_to_string_resolve_descriptors</c>.
        /// </summary>
        /// <param name="from">The (canonicalised or given) source descriptor.</param>
        /// <returns>The length estimate in characters.</returns>
        /// <exception cref="SystemError">A source class with no string rule (NumPy's <c>Impossible cast to string path requested.</c>).</exception>
        internal static long RequiredLength(DType from)
        {
            switch (from.kind)
            {
                case 'b':
                    /* 5 chars needed for cast to 'True' or 'False' */
                    return 5;
                case 'u':
                    return RequiredStrLen[from.itemsize];
                case 'i':
                    /* Add character for sign symbol */
                    return RequiredStrLen[from.itemsize] + 1;
                case 'f':
                    return 32; // half, float, double (and NumSharp's decimal)
                case 'c':
                    return 2 * 32;
                case 'S':
                    return from.itemsize;
                case 'U':
                    return from.itemsize / 4;
                default:
                    throw new SystemError("Impossible cast to string path requested.");
            }
        }

        /// <summary>
        ///     NumPy's <c>cast_to_string_resolve_descriptors</c>: the destination length is the source's
        ///     <see cref="RequiredLength"/> when none is given; a given destination is used (canonicalised) whatever its length
        ///     — the cast is then <c>safe</c> when it is long enough and <c>same_kind</c> otherwise (always <c>unsafe</c> for
        ///     <c>U → S</c>).
        /// </summary>
        /// <param name="givenDescrs">The source descriptor and the requested destination (null = "pick the length").</param>
        /// <param name="loopDescrs">Receives the descriptors the cast runs on.</param>
        /// <param name="viewOffset">Always <see cref="ArrayMethod.NoView"/>: a conversion to text is never a view.</param>
        /// <returns>The cast safety.</returns>
        /// <exception cref="TypeError">A <c>U</c> destination longer than a descriptor can describe
        /// (<c>string of length N is too large to store inside array.</c>).</exception>
        public override NPY_CASTING ResolveDescriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            viewOffset = NoView;
            var toMeta = (StringDTypeMeta)To;
            long size = RequiredLength(givenDescrs[0]);
            if (toMeta.IsUnicode)
            {
                if (size > int.MaxValue / 4)
                    throw new TypeError($"string of length {size} is too large to store inside array.");
                size *= 4;
            }

            /* The legacy loop can handle mismatching itemsizes */
            loopDescrs[1] = givenDescrs[1] == null
                ? new DType(toMeta, DType.NativeByteOrder, null, (int)size, toMeta.TypeChar)
                : To.EnsureCanonical(givenDescrs[1]);
            /* Set the input one as well (late for easier error management) */
            loopDescrs[0] = From.EnsureCanonical(givenDescrs[0]);

            if (Casting == NPY_CASTING.NPY_UNSAFE_CASTING)
                return Casting; // U -> S: unsafe whatever the lengths

            return loopDescrs[1].itemsize >= size ? NPY_CASTING.NPY_SAFE_CASTING : NPY_CASTING.NPY_SAME_KIND_CASTING;
        }
    }

    /// <summary>
    ///     The cast from <c>datetime64</c> / <c>timedelta64</c> into a string class — NumPy's
    ///     <c>time_to_string_resolve_descriptors</c> (<c>datetime.c</c>): always <c>unsafe</c>; a missing destination is
    ///     the ISO 8601 width of the unit for <c>datetime64</c> (<c>NpyDatetime_GetDatetimeISO8601StrLen</c>) and 21
    ///     characters for <c>timedelta64</c>.
    /// </summary>
    /// <remarks>
    ///     Descriptor level only (<c>can_cast</c>, <c>promote_types</c> answers): the datetime storage and its text
    ///     formatting are Stage C of <c>docs/plans/dtype-system.md</c>.
    /// </remarks>
    public sealed class TimeToStringCastingImpl : CastingImpl
    {
        /// <summary>Builds the cast from a datetime class into a string class.</summary>
        /// <param name="from">The <c>datetime64</c> or <c>timedelta64</c> class.</param>
        /// <param name="to">The destination string class.</param>
        internal TimeToStringCastingImpl(DatetimeDTypeMeta from, StringDTypeMeta to)
            : base(from.IsTimedelta ? "timedelta_to_string_cast" : "datetime_to_string_cast", NPY_CASTING.NPY_UNSAFE_CASTING,
                from.IsTimedelta ? NDArrayMethodFlags.REQUIRES_PYAPI : NDArrayMethodFlags.SUPPORTS_UNALIGNED | NDArrayMethodFlags.REQUIRES_PYAPI,
                from, to)
        {
        }

        /// <summary>
        ///     NumPy's <c>NpyDatetime_GetDatetimeISO8601StrLen(local=0, base)</c>: the characters (plus the C string's NUL)
        ///     an ISO 8601 rendering of a <paramref name="unit"/> datetime needs — the 21-character 64-bit year, then
        ///     <c>-MM</c>, <c>-DD</c>, <c>THH</c>, <c>:MM</c>, <c>:SS</c>, <c>.mmm</c> and three more digits per finer unit, and a
        ///     <c>Z</c> for the time units. The generic unit only ever prints <c>NaT</c> (4).
        /// </summary>
        /// <param name="unit">The datetime unit.</param>
        /// <returns>The width in characters.</returns>
        internal static int DatetimeIsoStrLen(NPY_DATETIMEUNIT unit)
        {
            if (unit == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
                return 4; /* Generic units can only be used to represent NaT */

            int len = 0;
            // The C source is a fall-through switch from the finest unit down to years; each step adds its field.
            if (unit >= NPY_DATETIMEUNIT.NPY_FR_as) len += 3; /* "###" */
            if (unit >= NPY_DATETIMEUNIT.NPY_FR_fs) len += 3; /* "###" */
            if (unit >= NPY_DATETIMEUNIT.NPY_FR_ps) len += 3; /* "###" */
            if (unit >= NPY_DATETIMEUNIT.NPY_FR_ns) len += 3; /* "###" */
            if (unit >= NPY_DATETIMEUNIT.NPY_FR_us) len += 3; /* "###" */
            if (unit >= NPY_DATETIMEUNIT.NPY_FR_ms) len += 4; /* ".###" */
            if (unit >= NPY_DATETIMEUNIT.NPY_FR_s) len += 3;  /* ":##" */
            if (unit >= NPY_DATETIMEUNIT.NPY_FR_m) len += 3;  /* ":##" */
            if (unit >= NPY_DATETIMEUNIT.NPY_FR_h) len += 3;  /* "T##" */
            if (unit >= NPY_DATETIMEUNIT.NPY_FR_W) len += 3;  /* "-##" (weeks and days print a day) */
            if (unit >= NPY_DATETIMEUNIT.NPY_FR_M) len += 3;  /* "-##" */
            len += 21;                                         /* 64-bit year */

            if (unit >= NPY_DATETIMEUNIT.NPY_FR_h)
                len += 1; /* "Z" */
            len += 1;     /* NULL terminator */
            return len;
        }

        /// <summary>
        ///     NumPy's <c>time_to_string_resolve_descriptors</c>: a given destination is kept as is for <c>datetime64</c>
        ///     (NumPy does not check the length here); otherwise — and always for <c>timedelta64</c> — a fresh descriptor of
        ///     the ISO width (datetime) or 21 characters (timedelta) is chosen.
        /// </summary>
        /// <param name="givenDescrs">The datetime source and the requested string destination (may be null).</param>
        /// <param name="loopDescrs">Receives the descriptors the cast runs on.</param>
        /// <param name="viewOffset">Always <see cref="ArrayMethod.NoView"/>.</param>
        /// <returns>Always <c>unsafe</c>.</returns>
        public override NPY_CASTING ResolveDescriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            viewOffset = NoView;
            var toMeta = (StringDTypeMeta)To;
            if (givenDescrs[1] != null && !((DatetimeDTypeMeta)From).IsTimedelta)
            {
                /*
                 * At the time of writing, NumPy does not check the length here,
                 * but will error if filling fails.
                 */
                loopDescrs[1] = givenDescrs[1];
            }
            else
            {
                /* Find the correct string length, possibly based on the unit */
                int size = ((DatetimeDTypeMeta)From).IsTimedelta
                    ? 21 /* This is arguably missing space for the unit */
                    : DatetimeIsoStrLen(givenDescrs[0].DatetimeMetadata.Value.Base);
                if (toMeta.IsUnicode)
                    size *= 4;
                loopDescrs[1] = new DType(toMeta, DType.NativeByteOrder, null, size, toMeta.TypeChar);
            }

            loopDescrs[0] = From.EnsureCanonical(givenDescrs[0]);
            return NPY_CASTING.NPY_UNSAFE_CASTING;
        }
    }

    /// <summary>
    ///     The cast from a string class into <c>datetime64</c> — NumPy's <c>string_to_datetime_cast_resolve_descriptors</c>:
    ///     always <c>unsafe</c>; a missing destination is the generic unit (which NumPy notes "doesn't actually work, and
    ///     will error during the cast"). Descriptor level only until Stage C.
    /// </summary>
    public sealed class StringToDatetimeCastingImpl : CastingImpl
    {
        /// <summary>Builds the cast from a string class into <c>datetime64</c>.</summary>
        /// <param name="from">The source string class.</param>
        /// <param name="to">The <c>datetime64</c> class.</param>
        internal StringToDatetimeCastingImpl(StringDTypeMeta from, DatetimeDTypeMeta to)
            : base("string_to_datetime_cast", NPY_CASTING.NPY_UNSAFE_CASTING,
                from.IsUnicode ? NDArrayMethodFlags.SUPPORTS_UNALIGNED | NDArrayMethodFlags.REQUIRES_PYAPI : NDArrayMethodFlags.SUPPORTS_UNALIGNED,
                from, to)
        {
        }

        /// <summary>
        ///     The destination is the given descriptor made canonical, or the class default (generic unit); the source is
        ///     used as given ("We currently support byte-swapping, so any (unicode) string is OK").
        /// </summary>
        /// <param name="givenDescrs">The string source and the requested datetime destination (may be null).</param>
        /// <param name="loopDescrs">Receives the descriptors the cast runs on.</param>
        /// <param name="viewOffset">Always <see cref="ArrayMethod.NoView"/>.</param>
        /// <returns>Always <c>unsafe</c>.</returns>
        public override NPY_CASTING ResolveDescriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            viewOffset = NoView;
            loopDescrs[1] = givenDescrs[1] == null ? To.DefaultDescr() : To.EnsureCanonical(givenDescrs[1]);
            loopDescrs[0] = givenDescrs[0];
            return NPY_CASTING.NPY_UNSAFE_CASTING;
        }
    }
}
