using System;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     The cast between two storage-backed builtin classes (NumPy's <c>add_numeric_cast</c>, <c>convert_datatype.c</c>).
    ///     Its <see cref="ArrayMethod.Casting"/> is computed the way NumPy computes <c>spec.casting</c>: the same class →
    ///     <c>equiv</c> (a copy or byte swap); <c>safe</c> when the frozen promotion table says
    ///     <c>promote(from, to) == to</c> (NumPy's <c>_npy_can_cast_safely_table</c>); otherwise <c>same_kind</c> when the
    ///     source kind orders at or below the destination kind (<c>b &lt; u &lt; i &lt; f &lt; c</c>), else <c>unsafe</c>.
    ///     These are exactly the rules <c>np.can_cast(NPTypeCode, NPTypeCode)</c> always applied, so the builtin answers
    ///     are unchanged; the class pair now owns them.
    /// </summary>
    /// <remarks>
    ///     One deliberate departure: NumPy also calls two DIFFERENT classes with the same kind and size equivalent
    ///     (<c>intc</c> vs <c>long</c> on Windows), which would make NumSharp's <c>Char</c> and <c>UInt16</c> an
    ///     <c>equiv</c> pair. They are distinct storage types with distinct kernels, so only the identical class is
    ///     <c>equiv</c> here; <c>Char ↔ UInt16</c> is <c>safe</c> one way and <c>same_kind</c> the other, as before.
    ///     The loop is not provided by this object — the engine drives its IL cast kernels directly (Stage A).
    /// </remarks>
    public sealed class BuiltinCastingImpl : CastingImpl
    {
        internal BuiltinCastingImpl(LegacyBuiltinDTypeMeta from, LegacyBuiltinDTypeMeta to)
            : base(ReferenceEquals(from, to) ? "numeric_copy_or_byteswap" : "numeric_cast", NPY_CASTING.NPY_UNSAFE_CASTING /* see Casting */,
                NDArrayMethodFlags.SUPPORTS_UNALIGNED | NDArrayMethodFlags.NO_FLOATINGPOINT_ERRORS, from, to)
        {
            _from = from;
            _to = to;
        }

        private readonly LegacyBuiltinDTypeMeta _from;
        private readonly LegacyBuiltinDTypeMeta _to;
        private NPY_CASTING? _casting;

        /// <summary>
        ///     The minimal safety of this numeric cast — NumPy's <c>add_numeric_cast</c> value (<c>equiv</c> within a class,
        ///     <c>safe</c> when the promotion table maps <c>(from, to)</c> back onto <c>to</c>, <c>same_kind</c> along the
        ///     <c>b &lt; u &lt; i &lt; f &lt; c</c> kind order, else <c>unsafe</c>). Computed on FIRST READ, not at registration:
        ///     the promotion table is <c>np._nptypemap_arr_arr</c>, built by <c>np</c>'s static constructor, and <c>np</c>'s
        ///     field initializers (<c>np.float64</c> is a <see cref="DType"/>) are what first construct
        ///     <see cref="DTypeRegistry"/> — so reading the table while the registry registers the 15×15 impls would observe
        ///     it before it exists. The value is a constant, so the lazy evaluation is unobservable (a benign race at worst
        ///     computes it twice).
        /// </summary>
        public override NPY_CASTING Casting => _casting ??= ComputeCasting(_from, _to);

        private static NPY_CASTING ComputeCasting(LegacyBuiltinDTypeMeta from, LegacyBuiltinDTypeMeta to)
        {
            if (ReferenceEquals(from, to))
                return NPY_CASTING.NPY_EQUIV_CASTING;
            if (np._nptypemap_arr_arr.TryGetValue((from.TypeCode, to.TypeCode), out var promoted) && promoted == to.TypeCode)
                return NPY_CASTING.NPY_SAFE_CASTING;
            if (DTypeCasting.KindToOrdering(from.Kind) <= DTypeCasting.KindToOrdering(to.Kind))
                return NPY_CASTING.NPY_SAME_KIND_CASTING;
            return NPY_CASTING.NPY_UNSAFE_CASTING;
        }

        /// <inheritdoc/>
        public override NPY_CASTING ResolveDescriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            if (IsWithinDType)
                return LegacySameDTypeResolveDescriptors(this, givenDescrs, loopDescrs, out viewOffset);
            return SimpleCastResolveDescriptors(this, givenDescrs, loopDescrs, out viewOffset);
        }

        /// <summary>
        ///     NumPy's <c>legacy_same_dtype_resolve_descriptors</c>: two descriptors of one class differ only in byte order,
        ///     so the cast is a no-op view when both are (non-)swapped and an <c>equiv</c> byte swap otherwise.
        /// </summary>
        internal static NPY_CASTING LegacySameDTypeResolveDescriptors(CastingImpl impl, DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            loopDescrs[0] = givenDescrs[0];
            loopDescrs[1] = givenDescrs[1] ?? impl.From.EnsureCanonical(givenDescrs[0]);

            /*
             * Legacy dtypes (except datetime) only have byte-order and elsize as
             * storage parameters.
             */
            if (loopDescrs[0].isnative == loopDescrs[1].isnative)
            {
                viewOffset = 0;
                return NPY_CASTING.NPY_NO_CASTING;
            }
            viewOffset = NoView;
            return NPY_CASTING.NPY_EQUIV_CASTING;
        }

        /// <summary>
        ///     NumPy's <c>simple_cast_resolve_descriptors</c> for two different non-parametric classes: canonicalise the
        ///     inputs, default the output, and report the method's own casting level.
        /// </summary>
        internal static NPY_CASTING SimpleCastResolveDescriptors(CastingImpl impl, DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            loopDescrs[0] = impl.From.EnsureCanonical(givenDescrs[0]);
            loopDescrs[1] = givenDescrs[1] != null ? impl.To.EnsureCanonical(givenDescrs[1]) : impl.To.DefaultDescr();

            viewOffset = NoView;
            if (impl.Casting != NPY_CASTING.NPY_NO_CASTING)
                return impl.Casting;
            if (loopDescrs[0].isnative == loopDescrs[1].isnative)
            {
                viewOffset = 0;
                return NPY_CASTING.NPY_NO_CASTING;
            }
            return NPY_CASTING.NPY_EQUIV_CASTING;
        }
    }

    /// <summary>
    ///     The within-class cast of <c>datetime64</c> / <c>timedelta64</c> — a unit conversion. Port of NumPy's
    ///     <c>time_to_time_resolve_descriptors</c> (<c>datetime.c</c>): identical metadata (or an exact 10³ᵏ metric-prefix
    ///     fold such as <c>[1000ms] → [s]</c>) is a no-op view / <c>equiv</c> byte swap; a generic source is
    ///     <c>safe</c>; a generic destination or, for timedelta, a jump across the years-months barrier is <c>unsafe</c>;
    ///     towards a finer unit that divides exactly is <c>safe</c>, anything else <c>same_kind</c>.
    /// </summary>
    public sealed class TimeToTimeCastingImpl : CastingImpl
    {
        internal TimeToTimeCastingImpl(DatetimeDTypeMeta meta)
            : base("datetime_casts", NPY_CASTING.NPY_UNSAFE_CASTING, NDArrayMethodFlags.SUPPORTS_UNALIGNED, meta, meta)
        {
        }

        /// <inheritdoc/>
        public override NPY_CASTING ResolveDescriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            /* This is a within-dtype cast, which currently must handle byteswapping */
            loopDescrs[0] = givenDescrs[0];
            loopDescrs[1] = givenDescrs[1] ?? From.EnsureCanonical(givenDescrs[0]);
            viewOffset = NoView;

            bool isTimedelta = ((DatetimeDTypeMeta)From).IsTimedelta;

            if (ReferenceEquals(givenDescrs[0], givenDescrs[1]))
            {
                viewOffset = 0;
                return NPY_CASTING.NPY_NO_CASTING;
            }

            bool byteorderMayAllowView = loopDescrs[0].isnative == loopDescrs[1].isnative;

            var meta1 = loopDescrs[0].DatetimeMetadata.Value;
            var meta2 = loopDescrs[1].DatetimeMetadata.Value;

            if ((meta1.Base == meta2.Base && meta1.Num == meta2.Num) ||
                // handle some common metric prefix conversions
                // 1000 fold conversions
                (meta2.Base >= NPY_DATETIMEUNIT.NPY_FR_s && (int)meta1.Base - (int)meta2.Base == 1
                                                          && meta2.Num != 0 && meta1.Num / meta2.Num == 1000) ||
                // 10^6 fold conversions
                (meta2.Base >= NPY_DATETIMEUNIT.NPY_FR_s && (int)meta1.Base - (int)meta2.Base == 2
                                                          && meta2.Num != 0 && meta1.Num / meta2.Num == 1000000) ||
                // 10^9 fold conversions
                (meta2.Base >= NPY_DATETIMEUNIT.NPY_FR_s && (int)meta1.Base - (int)meta2.Base == 3
                                                          && meta2.Num != 0 && meta1.Num / meta2.Num == 1000000000))
            {
                if (byteorderMayAllowView)
                {
                    viewOffset = 0;
                    return NPY_CASTING.NPY_NO_CASTING;
                }
                return NPY_CASTING.NPY_EQUIV_CASTING;
            }
            if (meta1.Base == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
            {
                if (byteorderMayAllowView)
                    viewOffset = 0;
                return NPY_CASTING.NPY_SAFE_CASTING;
            }
            if (meta2.Base == NPY_DATETIMEUNIT.NPY_FR_GENERIC)
            {
                /* TODO: This is actually an invalid cast (casting will error) */
                return NPY_CASTING.NPY_UNSAFE_CASTING;
            }
            if (isTimedelta && (
                    /* jump between time units and date units is unsafe for timedelta */
                    (meta1.Base <= NPY_DATETIMEUNIT.NPY_FR_M && meta2.Base > NPY_DATETIMEUNIT.NPY_FR_M) ||
                    (meta1.Base > NPY_DATETIMEUNIT.NPY_FR_M && meta2.Base <= NPY_DATETIMEUNIT.NPY_FR_M)))
            {
                return NPY_CASTING.NPY_UNSAFE_CASTING;
            }
            if (meta1.Base <= meta2.Base)
            {
                /* Casting to a more precise unit is currently considered safe */
                if (DatetimeMetaData.Divides(meta1, meta2, isTimedelta))
                {
                    /* If it divides, we consider it to be a safe cast */
                    return NPY_CASTING.NPY_SAFE_CASTING;
                }
                return NPY_CASTING.NPY_SAME_KIND_CASTING;
            }
            return NPY_CASTING.NPY_SAME_KIND_CASTING;
        }
    }

    /// <summary>
    ///     The cast between <c>datetime64</c> and <c>timedelta64</c> (either direction) — NumPy's
    ///     <c>datetime_to_timedelta_resolve_descriptors</c>: the destination inherits the SOURCE's unit when none is
    ///     given (which is how <c>promote_types</c> turns <c>m8[Y]</c> into <c>M8[Y]</c> before taking the GCD), and the
    ///     cast is always <c>unsafe</c>.
    /// </summary>
    public sealed class DatetimeTimedeltaCastingImpl : CastingImpl
    {
        internal DatetimeTimedeltaCastingImpl(DatetimeDTypeMeta from, DatetimeDTypeMeta to)
            : base(from.IsTimedelta ? "timedelta_and_datetime_cast" : "datetime_to_timedelta_cast",
                NPY_CASTING.NPY_UNSAFE_CASTING, NDArrayMethodFlags.None, from, to)
        {
        }

        /// <inheritdoc/>
        public override NPY_CASTING ResolveDescriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            loopDescrs[0] = From.EnsureCanonical(givenDescrs[0]);
            loopDescrs[1] = givenDescrs[1] == null
                ? ((DatetimeDTypeMeta)To).Descr(givenDescrs[0].DatetimeMetadata.Value)
                : To.EnsureCanonical(givenDescrs[1]);
            viewOffset = NoView;
            /*
             * Mostly NPY_UNSAFE_CASTING is not true, the cast will fail.
             * TODO: Once ufuncs use dtype specific promotion rules,
             *       this is likely unnecessary
             */
            return NPY_CASTING.NPY_UNSAFE_CASTING;
        }
    }

    /// <summary>
    ///     A cast whose safety is fixed per class pair with the plain descriptor resolution — NumPy's
    ///     <c>PyArray_AddLegacyWrapping_CastingImpl</c>, used for the numeric ↔ datetime/timedelta casts: every cast
    ///     to or from <c>datetime64</c> is <c>unsafe</c>; an integer or bool casts <c>safe</c>ly to <c>timedelta64</c>
    ///     (a 64-bit unsigned one only <c>same_kind</c>), a float or complex only <c>unsafe</c>ly.
    /// </summary>
    public sealed class LegacyWrappingCastingImpl : CastingImpl
    {
        internal LegacyWrappingCastingImpl(DTypeMeta from, DTypeMeta to, NPY_CASTING casting)
            : base("legacy_cast", casting, NDArrayMethodFlags.None, from, to)
        {
        }

        /// <inheritdoc/>
        public override NPY_CASTING ResolveDescriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            if (IsWithinDType)
                return BuiltinCastingImpl.LegacySameDTypeResolveDescriptors(this, givenDescrs, loopDescrs, out viewOffset);
            return BuiltinCastingImpl.SimpleCastResolveDescriptors(this, givenDescrs, loopDescrs, out viewOffset);
        }
    }
}
