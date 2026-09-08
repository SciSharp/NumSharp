using System;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     The NEP 43 casting engine over descriptors — the ports of NumPy's <c>convert_datatype.c</c> entry points:
    ///     <see cref="GetCastingImpl"/> (<c>PyArray_GetCastingImpl</c>), <see cref="GetCastInfo"/>
    ///     (<c>PyArray_GetCastInfo</c>), <see cref="CheckCastSafety"/> (<c>PyArray_CheckCastSafety</c>),
    ///     <see cref="CanCastTypeTo"/> (<c>PyArray_CanCastTypeTo</c> — the engine behind <c>np.can_cast</c>),
    ///     <see cref="EquivTypes"/> (<c>PyArray_EquivTypes</c>), <see cref="MinCastSafety"/> and the casting-string
    ///     parser. Every question is answered by the <see cref="CastingImpl"/> registered for the CLASS pair, so a new
    ///     dtype joins <c>np.can_cast</c> by registering its casts — no table edit.
    /// </summary>
    /// <remarks>
    ///     For the storage-backed builtins the answers are exactly those <c>np.can_cast(NPTypeCode, NPTypeCode)</c> always
    ///     gave (the <see cref="BuiltinCastingImpl"/> encodes the same table + kind-order rules), which the
    ///     <c>dtype_text</c> differential-fuzz tier gates. What is new is everything a table cannot express: byte order
    ///     (<c>can_cast('&gt;i4', 'i4', 'equiv')</c> is True, <c>'no'</c> is False) and the datetime unit rules.
    /// </remarks>
    public static class DTypeCasting
    {
        /// <summary>
        ///     NumPy's <c>PyArray_MinCastSafety</c>: the LESS safe of two cast levels (the enum is ordered
        ///     <c>no &lt; equiv &lt; safe &lt; same_kind &lt; unsafe</c>, so the larger value wins).
        /// </summary>
        public static NPY_CASTING MinCastSafety(NPY_CASTING casting1, NPY_CASTING casting2)
            => casting1 > casting2 ? casting1 : casting2;

        /// <summary>The casting implementation from one class to another, or null when no cast exists — <c>PyArray_GetCastingImpl</c>.</summary>
        public static CastingImpl GetCastingImpl(DTypeMeta from, DTypeMeta to)
        {
            if (from == null) throw new ArgumentNullException(nameof(from));
            if (to == null) throw new ArgumentNullException(nameof(to));
            return from.GetCastingImpl(to);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_GetCastInfo</c>: the cast safety between two descriptors (<paramref name="to"/> may be
        ///     null to ask about the class <paramref name="toMeta"/> — "cast to whatever instance you pick"), plus the
        ///     byte offset at which the cast is a plain view (<see cref="ArrayMethod.NoView"/> when it is not). Null when
        ///     the cast is impossible (NumPy returns -1).
        /// </summary>
        public static NPY_CASTING? GetCastInfo(DType from, DType to, DTypeMeta toMeta, out long viewOffset)
        {
            if (from == null) throw new ArgumentNullException(nameof(from));
            if (to != null)
                toMeta = to.Meta;
            if (toMeta == null) throw new ArgumentNullException(nameof(toMeta));

            viewOffset = ArrayMethod.NoView;
            var impl = from.Meta.GetCastingImpl(toMeta);
            if (impl == null)
                return null;
            return GetCastSafetyFromCastingImpl(impl, from, to, out viewOffset);
        }

        /// <summary>
        ///     NumPy's <c>_get_cast_safety_from_castingimpl</c>: run <see cref="ArrayMethod.ResolveDescriptors"/> and, when the
        ///     loop descriptors it picked differ from the given ones (a byte swap folded in, a canonicalisation), fold the
        ///     safety of THOSE adaptations in too, so a multi-step cast reports its weakest step.
        /// </summary>
        private static NPY_CASTING? GetCastSafetyFromCastingImpl(CastingImpl impl, DType from, DType to, out long viewOffset)
        {
            var given = new[] { from, to };
            var loop = new DType[2];
            NPY_CASTING casting;
            try
            {
                casting = impl.ResolveDescriptors(given, loop, out viewOffset);
            }
            catch (TypeError)
            {
                viewOffset = ArrayMethod.NoView;
                return null;
            }

            /* The returned descriptors may not match, requiring a second check */
            if (!SameDescr(loop[0], given[0]))
            {
                var fromCasting = GetCastInfo(given[0], loop[0], null, out long fromOffset);
                if (fromCasting == null)
                    return null;
                casting = MinCastSafety(casting, fromCasting.Value);
                if (fromOffset != viewOffset)
                    viewOffset = ArrayMethod.NoView; /* `view_offset` differs: The multi-step cast cannot be a view. */
            }
            if (given[1] != null && !SameDescr(loop[1], given[1]))
            {
                var toCasting = GetCastInfo(given[1], loop[1], null, out long toOffset);
                if (toCasting == null)
                    return null;
                casting = MinCastSafety(casting, toCasting.Value);
                if (toOffset != viewOffset)
                    viewOffset = ArrayMethod.NoView;
            }
            return casting;
        }

        private static bool SameDescr(DType a, DType b) => ReferenceEquals(a, b) || (a is not null && a.Equals(b));

        /// <summary>
        ///     NumPy's <c>PyArray_CheckCastSafety</c>: whether casting <paramref name="from"/> to <paramref name="to"/> (or to
        ///     the class <paramref name="toMeta"/> when <paramref name="to"/> is null) satisfies <paramref name="casting"/>.
        ///     Short-circuits on the implementation's own minimal safety and only resolves descriptors when the request is
        ///     stricter than that.
        /// </summary>
        public static bool CheckCastSafety(NPY_CASTING casting, DType from, DType to, DTypeMeta toMeta)
        {
            if (from == null) throw new ArgumentNullException(nameof(from));
            if (to != null)
                toMeta = to.Meta;
            if (toMeta == null) throw new ArgumentNullException(nameof(toMeta));

            var impl = from.Meta.GetCastingImpl(toMeta);
            if (impl == null)
                return false;

            if (MinCastSafety(impl.Casting, casting) == casting)
                return true; /* No need to check using `castingimpl.resolve_descriptors()` */

            var safety = GetCastSafetyFromCastingImpl(impl, from, to, out _);
            if (safety == null)
                return false;
            /* If casting is the smaller (or equal) safety we match */
            return MinCastSafety(safety.Value, casting) == casting;
        }

        /// <summary>
        ///     NumPy's <c>PyArray_CanCastTypeTo</c> — the engine behind <c>np.can_cast(from, to, casting)</c> for two
        ///     descriptors. An impossible cast is False, never an error.
        /// </summary>
        public static bool CanCastTypeTo(DType from, DType to, NPY_CASTING casting)
        {
            if (from == null) throw new ArgumentNullException(nameof(from));
            if (to == null) throw new ArgumentNullException(nameof(to));
            return CheckCastSafety(casting, from, to, to.Meta);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_EquivTypes</c>: two descriptors are equivalent when casting between them is a
        ///     <c>no</c>-cast (same class and byte order; for datetimes also an exact metric-prefix fold such as
        ///     <c>M8[1000ms] → M8[s]</c>, which is why NumPy's <c>==</c> is not symmetric there — <see cref="DType.Equals(DType)"/>
        ///     is structural instead).
        /// </summary>
        public static bool EquivTypes(DType type1, DType type2)
        {
            if (type1 == null) throw new ArgumentNullException(nameof(type1));
            if (type2 == null) throw new ArgumentNullException(nameof(type2));
            if (ReferenceEquals(type1, type2))
                return true;
            var safety = GetCastInfo(type1, type2, null, out _);
            if (safety == null)
                return false;
            /* If casting is "no casting" this dtypes are considered equivalent. */
            return MinCastSafety(safety.Value, NPY_CASTING.NPY_NO_CASTING) == NPY_CASTING.NPY_NO_CASTING;
        }

        /// <summary>
        ///     NumPy's <c>PyArray_CastingConverter</c>: the case-sensitive casting strings.
        /// </summary>
        /// <exception cref="ValueError"><c>casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '…')</c>.</exception>
        public static NPY_CASTING ParseCasting(string casting)
        {
            switch (casting)
            {
                case "no": return NPY_CASTING.NPY_NO_CASTING;
                case "equiv": return NPY_CASTING.NPY_EQUIV_CASTING;
                case "safe": return NPY_CASTING.NPY_SAFE_CASTING;
                case "same_kind": return NPY_CASTING.NPY_SAME_KIND_CASTING;
                case "unsafe": return NPY_CASTING.NPY_UNSAFE_CASTING;
                default:
                    throw new ValueError($"casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '{casting}')");
            }
        }

        /// <summary>NumPy's <c>npy_casting_to_string</c>: the casting string of a level.</summary>
        public static string CastingToString(NPY_CASTING casting)
        {
            switch (casting)
            {
                case NPY_CASTING.NPY_NO_CASTING: return "no";
                case NPY_CASTING.NPY_EQUIV_CASTING: return "equiv";
                case NPY_CASTING.NPY_SAFE_CASTING: return "safe";
                case NPY_CASTING.NPY_SAME_KIND_CASTING: return "same_kind";
                case NPY_CASTING.NPY_UNSAFE_CASTING: return "unsafe";
                default: return "<unknown>";
            }
        }

        /// <summary>
        ///     NumPy's <c>dtype_kind_to_ordering</c>: the kind ordering behind <c>same_kind</c> casting —
        ///     <c>b(0) &lt; u(1) &lt; i(2) &lt; f(4) &lt; c(5) &lt; S(6) &lt; U(7) &lt; V(8) &lt; O(9)</c>; datetime kinds do not fit
        ///     the hierarchy (-1).
        /// </summary>
        public static int KindToOrdering(char kind)
        {
            switch (kind)
            {
                case 'b': return 0;
                case 'u': return 1;
                case 'i': return 2;
                case 'f': return 4;
                case 'c': return 5;
                case 'S':
                case 'a': return 6;
                case 'U': return 7;
                case 'V': return 8;
                case 'O': return 9;
                default: return -1;
            }
        }
    }
}
