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
        ///     NumPy's <c>_get_cast_safety_from_castingimpl</c>: run <see cref="ArrayMethod.resolve_descriptors"/> and, when the
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
                casting = impl.resolve_descriptors(given, loop, out viewOffset);
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
        ///     NumPy's <c>PyArray_GetDTypeTransferFunction</c> (<c>dtype_transfer.c</c>: <c>define_cast_for_descrs</c> over
        ///     <c>init_cast_info</c>): the strided loop that casts <paramref name="srcDtype"/> into <paramref name="dstDtype"/>,
        ///     from the casting implementation registered for the CLASS pair - its <see cref="ArrayMethod.resolve_descriptors"/>
        ///     and its <see cref="ArrayMethod.get_strided_loop"/>. Every route that converts element by element over its own walk
        ///     asks this instead of choosing a converter by the dtypes' names: the raw assignment, the iterator's buffer
        ///     transfers, the reductions over text.
        /// </summary>
        /// <param name="aligned">True when both operands are aligned (NumSharp's loops read and write unaligned elements
        ///     too, so NumPy's wrapping of a method without <c>SUPPORTS_UNALIGNED</c> is never needed).</param>
        /// <param name="srcStride">The source's byte stride (of the first call; a loop may assume it fixed).</param>
        /// <param name="dstStride">The destination's byte stride.</param>
        /// <param name="srcDtype">The source descriptor.</param>
        /// <param name="dstDtype">The destination descriptor.</param>
        /// <param name="moveReferences">NumPy's <c>move_references</c> (the source's references may be moved, not copied);
        ///     NumSharp's <c>StringDType</c> loops always duplicate, so it is accepted and has no effect.</param>
        /// <param name="outFlags">Receives the loop's flags (NumPy's <c>out_flags</c>).</param>
        /// <param name="srcAllocator">The string allocator of a <c>StringDType</c> source's cells, else null (NumPy keeps it
        ///     on the descriptor; NumSharp on the cells' buffer, so the caller passes it).</param>
        /// <param name="dstAllocator">The string allocator of a <c>StringDType</c> destination's cells, else null.</param>
        /// <returns>The resolved cast (NumPy's <c>NPY_cast_info</c>).</returns>
        /// <exception cref="ArgumentNullException">A descriptor is null.</exception>
        /// <exception cref="TypeError">No cast is registered for the class pair (NumPy's <c>Cannot cast data from X to
        ///     Y.</c>), or the impl's descriptor resolution refuses the pair.</exception>
        /// <exception cref="NotSupportedException">The resolved descriptors differ from the given ones (a byte swap or another
        ///     wrapping step NumPy would chain: NumSharp's storage is native, so no storage reaches it), or the impl has no
        ///     loop (a datetime cast before Stage C).</exception>
        internal static NDCastInfo GetDTypeTransferFunction(bool aligned, long srcStride, long dstStride, DType srcDtype,
            DType dstDtype, bool moveReferences, out NDArrayMethodFlags outFlags,
            Backends.Strings.VStringAllocator srcAllocator = null, Backends.Strings.VStringAllocator dstAllocator = null)
        {
            if (srcDtype == null) throw new ArgumentNullException(nameof(srcDtype));
            if (dstDtype == null) throw new ArgumentNullException(nameof(dstDtype));

            // init_cast_info: the class pair's implementation, or NumPy's TypeError for a pair without one.
            var impl = srcDtype.Meta.GetCastingImpl(dstDtype.Meta)
                       ?? throw new TypeError($"Cannot cast data from {srcDtype} to {dstDtype}.");
            var given = new[] { srcDtype, dstDtype };
            var loopDescrs = new DType[2];
            // A refusing resolver raises its own TypeError; it is NumPy's error too (init_cast_info passes it on).
            impl.resolve_descriptors(given, loopDescrs, out _);

            // define_cast_for_descrs: a loop descriptor other than the given one needs a wrapping cast into or out of it (a
            // byte swap, a canonicalisation), which NumPy chains as a multi-step cast. Every storage descriptor is native and
            // every storage class's resolver returns the given descriptors, so NumSharp never needs one: refuse loudly
            // instead of running a loop on the wrong descriptor.
            if (!SameDescr(loopDescrs[0], srcDtype) || !SameDescr(loopDescrs[1], dstDtype))
                throw new NotSupportedException(
                    $"The cast from {srcDtype.ToString(true)} to {dstDtype.ToString(true)} resolves to {loopDescrs[0].ToString(true)} -> " +
                    $"{loopDescrs[1].ToString(true)}: a multi-step cast, which NumSharp's native storage never needs.");

            // The allocators ride on the context only when a StringDType operand has one (NumSharp's addition to NumPy's
            // context, see ArrayMethodContext.Allocators).
            var allocators = srcAllocator is null && dstAllocator is null
                ? null
                : new[] { srcAllocator, dstAllocator };
            var context = new ArrayMethodContext(impl, loopDescrs, allocators);
            var func = impl.get_strided_loop(context, aligned, new[] { srcStride, dstStride }, out outFlags);
            return new NDCastInfo(func, context);
        }

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

            if (MinCastSafety(impl.casting, casting) == casting)
                return true; /* No need to check using `castingimpl.resolve_descriptors()` */

            var safety = GetCastSafetyFromCastingImpl(impl, from, to, out _);
            if (safety == null)
                return false;
            /* If casting is the smaller (or equal) safety we match */
            return MinCastSafety(safety.Value, casting) == casting;
        }

        /// <summary>
        ///     NumPy's <c>PyArray_CanCastTypeTo</c> — the engine behind <c>np.can_cast(from, to, casting)</c> for two
        ///     descriptors. An impossible cast is False, never an error. An UNSIZED string target (<c>S0</c>/<c>U0</c>, i.e.
        ///     <c>np.dtype('U')</c>) stands for its CLASS — "cast to whatever length you need" — so
        ///     <c>can_cast('i4', 'U', 'safe')</c> is True although <c>can_cast('i4', 'U10', 'safe')</c> is False, and
        ///     <c>can_cast('U5', 'U', 'no')</c> is True.
        /// </summary>
        /// <param name="from">The source descriptor.</param>
        /// <param name="to">The target descriptor (an unsized string descriptor means its class).</param>
        /// <param name="casting">The rule the cast must satisfy.</param>
        /// <returns>True when the cast satisfies <paramref name="casting"/>.</returns>
        /// <exception cref="ArgumentNullException">Either descriptor is null.</exception>
        public static bool CanCastTypeTo(DType from, DType to, NPY_CASTING casting)
        {
            if (from == null) throw new ArgumentNullException(nameof(from));
            if (to == null) throw new ArgumentNullException(nameof(to));
            var toMeta = to.Meta;
            /*
             * The `to = NULL` branch, which considers "S0" to be "flexible"
             * should probably be deprecated.
             */
            if (to.itemsize == 0 && toMeta is StringDTypeMeta)
                to = null; /* consider mainly S0 and U0 as S and U */
            return CheckCastSafety(casting, from, to, toMeta);
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
        ///     NumPy's <c>PyArray_CastingConverter</c> (<c>string_converter_helper</c> + <c>casting_parser</c>): the
        ///     case-sensitive casting strings.
        /// </summary>
        /// <param name="casting">The casting name.</param>
        /// <returns>The casting level.</returns>
        /// <exception cref="TypeError"><paramref name="casting"/> is null: NumPy's <c>casting must be str, not NoneType</c>.</exception>
        /// <exception cref="ValueError">
        ///     Another string: <c>casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got …)</c>, the string
        ///     quoted by Python's <c>repr</c> (NumPy's <c>%R</c>), so <c>"it's"</c> shows as <c>(got "it's")</c>.
        /// </exception>
        public static NPY_CASTING ParseCasting(string casting)
        {
            switch (casting)
            {
                case "no": return NPY_CASTING.NPY_NO_CASTING;
                case "equiv": return NPY_CASTING.NPY_EQUIV_CASTING;
                case "safe": return NPY_CASTING.NPY_SAFE_CASTING;
                case "same_kind": return NPY_CASTING.NPY_SAME_KIND_CASTING;
                case "unsafe": return NPY_CASTING.NPY_UNSAFE_CASTING;
                case null:
                    // string_converter_helper's non-str branch: "%s must be str, not %s" with the object's type name.
                    throw new TypeError("casting must be str, not NoneType");
                default:
                    throw new ValueError($"casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got {NumSharp.IO.PyLiteral.Repr(casting)})");
            }
        }

        /// <summary>NumPy 2.4's value-preserving casting rule: a name only <c>ndarray.astype</c>'s converter accepts.</summary>
        internal const string SameValueName = "same_value";

        /// <summary>
        ///     NumPy's <c>PyArray_CastingConverterSameValue</c> (<c>conversion_utils.c</c>), <c>ndarray.astype</c>'s casting
        ///     converter: <see cref="ParseCasting"/>'s five names plus NumPy 2.4's <c>'same_value'</c>, whose error text lists
        ///     it too. <c>npy_parse_arguments</c> runs it right after the order converter, before astype adapts the dtype,
        ///     takes the <c>copy=False</c> shortcut or checks the rule — so an unknown name raises even for a cast that needs
        ///     nothing.
        /// </summary>
        /// <param name="casting">The caller's casting name; null or empty is NumSharp's astype default, <c>'unsafe'</c>.</param>
        /// <param name="sameValue">True when the name is <c>'same_value'</c>. NumSharp's <see cref="NPY_CASTING"/> has no
        ///     member for it (its value carries NumPy's <c>NPY_SAME_VALUE_CASTING_FLAG</c>), so the caller reads this flag
        ///     and the returned level is <see cref="NPY_CASTING.NPY_UNSAFE_CASTING"/>, the rule NumPy's flag rides on.</param>
        /// <returns>The casting level.</returns>
        /// <exception cref="ValueError"><c>casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe',
        ///     'same_value' (got '…')</c>.</exception>
        internal static NPY_CASTING ParseCastingSameValue(string casting, out bool sameValue)
        {
            sameValue = casting == SameValueName;
            if (string.IsNullOrEmpty(casting) || sameValue)
                return NPY_CASTING.NPY_UNSAFE_CASTING;
            switch (casting)
            {
                case "no":
                case "equiv":
                case "safe":
                case "same_kind":
                case "unsafe":
                    return ParseCasting(casting);
                default:
                    throw new ValueError(
                        $"casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe', '{SameValueName}' (got '{casting}')");
            }
        }

        /// <summary>
        ///     NumPy's <c>npy_set_invalid_cast_error</c> (<c>convert_datatype.c</c>), the error of a cast the rule refuses:
        ///     <c>Cannot cast array data from dtype('&lt;U5') to dtype('&lt;U3') according to the rule 'safe'</c>, or
        ///     <c>Cannot cast scalar from …</c> for a 0-d source, as the house <see cref="InvalidCastException"/> (NumPy's
        ///     TypeError). Every route that checks a descriptor-level rule before it casts raises it: astype, the assignment,
        ///     the item-selection values, np.concatenate's string route.
        /// </summary>
        /// <param name="from">The source descriptor.</param>
        /// <param name="to">The target descriptor.</param>
        /// <param name="casting">The rule that refused the cast.</param>
        /// <param name="scalar">True for a 0-d source (NumPy's "scalar" wording).</param>
        /// <returns>The exception (not thrown: the caller throws it, so the compiler sees the control flow end).</returns>
        internal static InvalidCastException InvalidCastError(DType from, DType to, NPY_CASTING casting, bool scalar)
            => new InvalidCastException(
                $"Cannot cast {(scalar ? "scalar" : "array data")} from {from.ToString(true)} to {to.ToString(true)} " +
                $"according to the rule '{CastingToString(casting)}'");

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
