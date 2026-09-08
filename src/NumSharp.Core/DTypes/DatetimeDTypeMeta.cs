using System;

namespace NumSharp
{
    /// <summary>
    ///     The DType classes of <c>datetime64</c> (<c>np.dtypes.DateTime64DType</c>, type number 21, kind/char
    ///     <c>'M'</c>) and <c>timedelta64</c> (<c>np.dtypes.TimeDelta64DType</c>, 22, <c>'m'</c>) — NumSharp's first
    ///     PARAMETRIC classes: the unit metadata (<see cref="DatetimeMetaData"/>) lives on the descriptor INSTANCE, so
    ///     <c>M8[ns]</c> and <c>M8[s]</c> are two descriptors of one class, and promotion runs
    ///     <see cref="CommonInstance"/> (the unit GCD) instead of returning a class default.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Stage A is DESCRIPTOR-level: parsing (<c>np.dtype("M8[10ns]")</c>), formatting, equality,
    ///     <c>np.datetime_data</c>, GCD promotion and the unit casting rules. The class has no storage yet
    ///     (<see cref="DTypeMeta.TypeCode"/> is <see cref="NPTypeCode.Empty"/>, <see cref="DTypeMeta.ScalarType"/> is
    ///     null); allocating an array with a datetime descriptor raises <see cref="NotSupportedException"/>. Stage C adds
    ///     the int64 storage lane, NaT, ISO-8601 and the arithmetic loops (<c>docs/plans/datetime64.md</c>).
    ///     </para>
    ///     <para>
    ///     The slots are NumPy's (<c>dtypemeta.c</c>): <c>default_descr</c> hands out a COPY of the generic-unit
    ///     descriptor (NumPy's <c>datetime_and_timedelta_default_descr</c> — which is why <c>np.dtype('M8').isbuiltin</c> is
    ///     0), <c>common_dtype</c> is <c>datetime_common_dtype</c> (datetime absorbs timedelta; then the builtin table,
    ///     which lets timedelta absorb bool and the integers narrower than uint64 and refuses everything else), and
    ///     <c>common_instance</c> is <c>datetime_type_promotion</c> (the metadata GCD, strict about the non-linear
    ///     years/months for timedelta operands and relaxed for datetime ones).
    ///     </para>
    /// </remarks>
    public sealed class DatetimeDTypeMeta : DTypeMeta
    {
        internal DatetimeDTypeMeta(bool isTimedelta)
            : base(isTimedelta ? "TimeDelta64DType" : "DateTime64DType", "numpy.dtypes",
                typeNum: isTimedelta ? 22 : 21, scalarType: null, scalarName: isTimedelta ? "timedelta64" : "datetime64",
                NPTypeCode.Empty, kind: isTimedelta ? 'm' : 'M', typeChar: isTimedelta ? 'm' : 'M', itemSize: 8, alignment: 8,
                DTypeFlags.Legacy | DTypeFlags.Parametric)
        {
            IsTimedelta = isTimedelta;
        }

        /// <summary>True for <c>timedelta64</c> (a duration), false for <c>datetime64</c> (a moment).</summary>
        public bool IsTimedelta { get; }

        /// <summary>NumPy's <c>create_datetime_dtype</c>: a native descriptor of this class carrying <paramref name="meta"/>.</summary>
        public DType Descr(DatetimeMetaData meta) => new DType(this, DType.NativeByteOrder, meta);

        /// <summary>NumPy's <c>create_datetime_dtype_with_unit</c>: a native descriptor with unit <paramref name="unit"/> and multiplier 1.</summary>
        public DType Descr(NPY_DATETIMEUNIT unit) => Descr(new DatetimeMetaData(unit, 1));

        /// <summary>A fresh generic-unit descriptor (NumPy's <c>datetime_and_timedelta_default_descr</c> copies the singleton).</summary>
        public override DType DefaultDescr() => Descr(DatetimeMetaData.Generic);

        /// <inheritdoc/>
        public override DType EnsureCanonical(DType descr)
        {
            if (descr == null)
                throw new ArgumentNullException(nameof(descr));
            if (!ReferenceEquals(descr.Meta, this))
                throw new ArgumentException($"{descr.ToString(true)} is not an instance of {this}", nameof(descr));
            return descr.isnative ? descr : descr.WithByteOrder(DType.NativeByteOrder);
        }

        /// <summary>NumPy's <c>datetime_common_dtype</c>: datetime absorbs timedelta; otherwise the builtin rule.</summary>
        public override DTypeMeta CommonDType(DTypeMeta other)
        {
            if (other == null)
                throw new ArgumentNullException(nameof(other));
            /*
             * Timedelta/datetime shouldn't actually promote at all.  That they
             * currently do means that we need additional hacks in the comparison
             * type resolver.  For comparisons we have to make sure we reject it
             * nicely in order to return an array of True/False values.
             */
            if (!IsTimedelta && other is DatetimeDTypeMeta o && o.IsTimedelta)
                return this;
            return DTypePromotion.DefaultBuiltinCommonDType(this, other);
        }

        /// <summary>
        ///     NumPy's <c>datetime_type_promotion</c>: the result is <c>datetime64</c> if either operand is one, else
        ///     <c>timedelta64</c>, with the metadata GCD — strict about years/months for a timedelta operand, relaxed for
        ///     a datetime one. (By the time this runs, <see cref="DTypePromotion.PromoteTypes"/> has already cast both
        ///     operands to THIS class — a timedelta cast to datetime keeps its unit but loses the strictness, which is why
        ///     <c>promote_types('m8[Y]', 'M8[D]')</c> is <c>M8[D]</c> while <c>promote_types('m8[Y]', 'm8[D]')</c> raises.)
        /// </summary>
        public override DType CommonInstance(DType descr1, DType descr2)
        {
            if (descr1 == null) throw new ArgumentNullException(nameof(descr1));
            if (descr2 == null) throw new ArgumentNullException(nameof(descr2));
            if (descr1.Meta is not DatetimeDTypeMeta m1 || descr2.Meta is not DatetimeDTypeMeta m2)
                throw new TypeError($"cannot promote {descr1.ToString(true)} and {descr2.ToString(true)} as datetime metadata");

            bool isDatetime = !m1.IsTimedelta || !m2.IsTimedelta;
            var target = isDatetime ? DTypeRegistry.DateTime64 : DTypeRegistry.TimeDelta64;
            var gcd = DatetimeMetaData.GreatestCommonDivisor(descr1.DatetimeMetadata.Value, descr2.DatetimeMetadata.Value,
                strictWithNonlinearUnits1: m1.IsTimedelta, strictWithNonlinearUnits2: m2.IsTimedelta);
            return target.Descr(gcd);
        }
    }
}
