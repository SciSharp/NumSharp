using System;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>StringDType → StringDType</c> cast (<c>string_to_string_resolve_descriptors</c> in
    ///     <c>stringdtype/casts.cpp</c>): no casting at all between descriptors, except from a descriptor WITH a missing-data
    ///     object to one WITHOUT, which is UNSAFE (the missing values have nowhere to go).
    /// </summary>
    /// <remarks>
    ///     The method's minimal safety is <c>unsafe</c> (the spec's level), so <c>np.can_cast</c> always resolves the
    ///     descriptors for the stricter rules — which is how <c>can_cast(T, T, 'no')</c> is True while
    ///     <c>can_cast(StringDType(na_object=None), StringDType(), 'safe')</c> is False. NumPy allows a view only between
    ///     descriptors that share a string allocator (the same descriptor object); NumSharp's allocator belongs to the
    ///     buffer, so the same object is the test here too.
    /// </remarks>
    public sealed class VStringToVStringCastingImpl : CastingImpl
    {
        /// <summary>Builds the within-class cast. Called once by <see cref="DTypeRegistry"/>.</summary>
        /// <param name="meta">The <c>StringDType</c> class.</param>
        internal VStringToVStringCastingImpl(VStringDTypeMeta meta)
            : base("cast_StringDType_to_StringDType", NPY_CASTING.NPY_UNSAFE_CASTING, NDArrayMethodFlags.SUPPORTS_UNALIGNED, meta, meta)
        {
        }

        /// <summary>
        ///     Resolves the loop descriptors: the source as given, the destination as given (or the source's own parameters
        ///     when it is absent — NumPy's <c>stringdtype_finalize_descr</c>).
        /// </summary>
        /// <param name="givenDescrs">The source descriptor and the optional destination descriptor.</param>
        /// <param name="loopDescrs">Receives the loop descriptors.</param>
        /// <param name="viewOffset">0 when both loop descriptors are the same object (a view is legal), else <see cref="ArrayMethod.NoView"/>.</param>
        /// <returns><c>unsafe</c> from an NA-holding descriptor to one without, else <c>no</c> casting.</returns>
        public override NPY_CASTING resolve_descriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            loopDescrs[1] = givenDescrs[1] ?? givenDescrs[0];
            loopDescrs[0] = givenDescrs[0];
            viewOffset = NoView;

            var descr0 = loopDescrs[0].VString;
            var descr1 = loopDescrs[1].VString;
            if (descr0.HasNAObject && !descr1.HasNAObject)
            {
                // Cast from a dtype with an NA to one without, so it's a lossy unsafe cast. The other way around is still
                // safe, because there isn't an NA in the source to lossily convert.
                return NPY_CASTING.NPY_UNSAFE_CASTING;
            }

            // views are only legal between descriptors that share allocators (e.g. the same object)
            if (ReferenceEquals(loopDescrs[0], loopDescrs[1]))
                viewOffset = 0;
            return NPY_CASTING.NPY_NO_CASTING;
        }
        /// <summary>
        ///     The cast's strided loop (<see cref="VStringCastLoop.Get"/>): NumPy's <c>stringdtype/casts.cpp</c> loop of the
        ///     pair, the allocators of both sides locked per call.
        /// </summary>
        /// <param name="context">The resolved descriptors and the <c>StringDType</c> sides' allocators
        ///     (<see cref="ArrayMethodContext.Allocators"/>).</param>
        /// <param name="aligned">Unused: the loop reads and writes unaligned cells and elements.</param>
        /// <param name="strides">Unused: the loop takes the strides of each call.</param>
        /// <param name="flags">Receives the method's flags.</param>
        /// <returns>The loop: [0] the source run, [1] the destination run, byte strides.</returns>
        /// <exception cref="ArgumentException">A <c>StringDType</c> side without its allocator in the context.</exception>
        /// <exception cref="NotSupportedException">The other class has no storage (datetime64 / timedelta64 before
        ///     Stage C).</exception>
        public override NDInnerLoopFunc get_strided_loop(ArrayMethodContext context, bool aligned, long[] strides, out NDArrayMethodFlags flags)
            => VStringCastLoop.Get(this, context, out flags);
    }

    /// <summary>
    ///     NumPy's casts INTO <c>StringDType</c> from another class (<c>any_to_string_resolve_descriptors&lt;safety&gt;</c>): the
    ///     destination is the given descriptor or a fresh <c>StringDType()</c>, and the safety is fixed per source class —
    ///     <c>safe</c> from numbers, bool and the datetime pair, <c>same_kind</c> from <c>U</c> and <c>S</c>.
    /// </summary>
    /// <remarks>
    ///     Two levels per cast, as NumPy registers them: the method's MINIMAL safety (<see cref="ArrayMethod.casting"/>,
    ///     the spec's level, which <c>np.can_cast</c> checks first) and the safety <see cref="resolve_descriptors"/> returns.
    ///     They differ for bool: the spec says <c>same_kind</c>, the resolver <c>safe</c>, so <c>can_cast(bool, T, 'safe')</c>
    ///     is True only after the resolver runs.
    /// </remarks>
    public sealed class AnyToVStringCastingImpl : CastingImpl
    {
        private readonly NPY_CASTING _resolveSafety;

        /// <summary>Builds one cast into <c>StringDType</c>. Called by <see cref="DTypeRegistry"/>.</summary>
        /// <param name="from">The source class.</param>
        /// <param name="to">The <c>StringDType</c> class.</param>
        /// <param name="minimalCasting">The method's minimal safety (NumPy's spec <c>casting</c>).</param>
        /// <param name="resolveSafety">The safety the resolver returns (NumPy's template argument).</param>
        internal AnyToVStringCastingImpl(DTypeMeta from, VStringDTypeMeta to, NPY_CASTING minimalCasting, NPY_CASTING resolveSafety)
            : base("cast_" + from.Name.Replace("DType", string.Empty) + "_to_StringDType", minimalCasting,
                NDArrayMethodFlags.NO_FLOATINGPOINT_ERRORS, from, to)
        {
            _resolveSafety = resolveSafety;
        }

        /// <summary>
        ///     Resolves the loop descriptors: the source as given, the destination as given or a fresh <c>StringDType()</c>
        ///     (NumPy's <c>new_stringdtype_instance(NULL, 1)</c>).
        /// </summary>
        /// <param name="givenDescrs">The source descriptor and the optional destination descriptor.</param>
        /// <param name="loopDescrs">Receives the loop descriptors.</param>
        /// <param name="viewOffset">Always <see cref="ArrayMethod.NoView"/> (a conversion is never a view).</param>
        /// <returns>The cast's fixed safety.</returns>
        public override NPY_CASTING resolve_descriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            loopDescrs[1] = givenDescrs[1] ?? To.default_descr();
            loopDescrs[0] = givenDescrs[0];
            viewOffset = NoView;
            return _resolveSafety;
        }
        /// <summary>
        ///     The cast's strided loop (<see cref="VStringCastLoop.Get"/>): NumPy's <c>stringdtype/casts.cpp</c> loop of the
        ///     pair, the allocators of both sides locked per call.
        /// </summary>
        /// <param name="context">The resolved descriptors and the <c>StringDType</c> sides' allocators
        ///     (<see cref="ArrayMethodContext.Allocators"/>).</param>
        /// <param name="aligned">Unused: the loop reads and writes unaligned cells and elements.</param>
        /// <param name="strides">Unused: the loop takes the strides of each call.</param>
        /// <param name="flags">Receives the method's flags.</param>
        /// <returns>The loop: [0] the source run, [1] the destination run, byte strides.</returns>
        /// <exception cref="ArgumentException">A <c>StringDType</c> side without its allocator in the context.</exception>
        /// <exception cref="NotSupportedException">The other class has no storage (datetime64 / timedelta64 before
        ///     Stage C).</exception>
        public override NDInnerLoopFunc get_strided_loop(ArrayMethodContext context, bool aligned, long[] strides, out NDArrayMethodFlags flags)
            => VStringCastLoop.Get(this, context, out flags);
    }

    /// <summary>
    ///     NumPy's casts from <c>StringDType</c> to the fixed-width string classes (<c>string_to_fixed_width_resolve_descriptors</c>):
    ///     <c>same_kind</c>, and only to a SIZED destination — there is no way to know the width the strings need.
    /// </summary>
    public sealed class VStringToFixedWidthCastingImpl : CastingImpl
    {
        /// <summary>Builds the cast to <c>U</c> or <c>S</c>. Called by <see cref="DTypeRegistry"/>.</summary>
        /// <param name="from">The <c>StringDType</c> class.</param>
        /// <param name="to">The fixed-width class (<c>StrDType</c> / <c>BytesDType</c>).</param>
        internal VStringToFixedWidthCastingImpl(VStringDTypeMeta from, StringDTypeMeta to)
            : base("cast_StringDType_to_" + to.Name.Replace("DType", string.Empty), NPY_CASTING.NPY_SAME_KIND_CASTING,
                NDArrayMethodFlags.NO_FLOATINGPOINT_ERRORS, from, to)
        {
        }

        /// <summary>
        ///     Resolves the loop descriptors: the destination must be given (a sized <c>U</c> / <c>S</c>).
        /// </summary>
        /// <param name="givenDescrs">The source descriptor and the destination descriptor.</param>
        /// <param name="loopDescrs">Receives the loop descriptors.</param>
        /// <param name="viewOffset">Always <see cref="ArrayMethod.NoView"/>.</param>
        /// <returns><c>same_kind</c>.</returns>
        /// <exception cref="TypeError">No destination descriptor: NumPy's
        ///     <c>Casting from StringDType to a fixed-width dtype with an unspecified size is not currently supported, specify an explicit size for the output dtype instead.</c>
        ///     (<c>astype('U')</c> reports it as <c>cannot cast dtype StringDType() to &lt;class 'numpy.dtypes.StrDType'&gt;.</c>,
        ///     the wrapper of <see cref="DTypePromotion.CastDescrToDType"/>).</exception>
        public override NPY_CASTING resolve_descriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            viewOffset = NoView;
            if (givenDescrs[1] == null)
            {
                // currently there's no way to determine the correct output size, so set an error and bail
                throw new TypeError(
                    "Casting from StringDType to a fixed-width dtype with an unspecified size is not currently supported, " +
                    "specify an explicit size for the output dtype instead.");
            }
            loopDescrs[1] = givenDescrs[1];
            loopDescrs[0] = givenDescrs[0];
            return NPY_CASTING.NPY_SAME_KIND_CASTING;
        }
        /// <summary>
        ///     The cast's strided loop (<see cref="VStringCastLoop.Get"/>): NumPy's <c>stringdtype/casts.cpp</c> loop of the
        ///     pair, the allocators of both sides locked per call.
        /// </summary>
        /// <param name="context">The resolved descriptors and the <c>StringDType</c> sides' allocators
        ///     (<see cref="ArrayMethodContext.Allocators"/>).</param>
        /// <param name="aligned">Unused: the loop reads and writes unaligned cells and elements.</param>
        /// <param name="strides">Unused: the loop takes the strides of each call.</param>
        /// <param name="flags">Receives the method's flags.</param>
        /// <returns>The loop: [0] the source run, [1] the destination run, byte strides.</returns>
        /// <exception cref="ArgumentException">A <c>StringDType</c> side without its allocator in the context.</exception>
        /// <exception cref="NotSupportedException">The other class has no storage (datetime64 / timedelta64 before
        ///     Stage C).</exception>
        public override NDInnerLoopFunc get_strided_loop(ArrayMethodContext context, bool aligned, long[] strides, out NDArrayMethodFlags flags)
            => VStringCastLoop.Get(this, context, out flags);
    }

    /// <summary>
    ///     NumPy's casts from <c>StringDType</c> to numbers, bool and the datetime pair (<c>string_to_int / _float /
    ///     _bool / _datetime_timedelta_resolve_descriptors</c>): the destination is the given descriptor or the class
    ///     default, and the resolved safety is always <c>unsafe</c> (parsing can fail or lose a missing value).
    /// </summary>
    /// <remarks>
    ///     The minimal safety is the spec's: <c>unsafe</c> for numbers and the datetime pair, <c>same_kind</c> for bool —
    ///     which is why <c>can_cast(T, bool, 'same_kind')</c> is True although the resolver says <c>unsafe</c>. A datetime
    ///     destination needs a unit: NumPy refuses an absent descriptor with
    ///     <c>Casting from StringDType to datetimes without a unit is not currently supported</c>.
    /// </remarks>
    public sealed class VStringToNumberCastingImpl : CastingImpl
    {
        /// <summary>Builds one cast out of <c>StringDType</c>. Called by <see cref="DTypeRegistry"/>.</summary>
        /// <param name="from">The <c>StringDType</c> class.</param>
        /// <param name="to">The destination class (a number class, bool, datetime64 or timedelta64).</param>
        /// <param name="minimalCasting">The method's minimal safety (NumPy's spec <c>casting</c>).</param>
        internal VStringToNumberCastingImpl(VStringDTypeMeta from, DTypeMeta to, NPY_CASTING minimalCasting)
            : base("cast_StringDType_to_" + to.Name.Replace("DType", string.Empty), minimalCasting,
                NDArrayMethodFlags.REQUIRES_PYAPI, from, to)
        {
        }

        /// <summary>
        ///     Resolves the loop descriptors: the destination as given, or the class's default descriptor (a datetime
        ///     destination must be given).
        /// </summary>
        /// <param name="givenDescrs">The source descriptor and the optional destination descriptor.</param>
        /// <param name="loopDescrs">Receives the loop descriptors.</param>
        /// <param name="viewOffset">Always <see cref="ArrayMethod.NoView"/>.</param>
        /// <returns><c>unsafe</c>.</returns>
        /// <exception cref="TypeError">A datetime64 / timedelta64 destination without a descriptor (NumPy's
        ///     <c>Casting from StringDType to datetimes without a unit is not currently supported</c>).</exception>
        public override NPY_CASTING resolve_descriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
        {
            viewOffset = NoView;
            if (givenDescrs[1] == null)
            {
                if (To is DatetimeDTypeMeta)
                    throw new TypeError("Casting from StringDType to datetimes without a unit is not currently supported");
                loopDescrs[1] = To.default_descr();
            }
            else
            {
                loopDescrs[1] = givenDescrs[1];
            }
            loopDescrs[0] = givenDescrs[0];
            return NPY_CASTING.NPY_UNSAFE_CASTING;
        }
        /// <summary>
        ///     The cast's strided loop (<see cref="VStringCastLoop.Get"/>): NumPy's <c>stringdtype/casts.cpp</c> loop of the
        ///     pair, the allocators of both sides locked per call.
        /// </summary>
        /// <param name="context">The resolved descriptors and the <c>StringDType</c> sides' allocators
        ///     (<see cref="ArrayMethodContext.Allocators"/>).</param>
        /// <param name="aligned">Unused: the loop reads and writes unaligned cells and elements.</param>
        /// <param name="strides">Unused: the loop takes the strides of each call.</param>
        /// <param name="flags">Receives the method's flags.</param>
        /// <returns>The loop: [0] the source run, [1] the destination run, byte strides.</returns>
        /// <exception cref="ArgumentException">A <c>StringDType</c> side without its allocator in the context.</exception>
        /// <exception cref="NotSupportedException">The other class has no storage (datetime64 / timedelta64 before
        ///     Stage C).</exception>
        public override NDInnerLoopFunc get_strided_loop(ArrayMethodContext context, bool aligned, long[] strides, out NDArrayMethodFlags flags)
            => VStringCastLoop.Get(this, context, out flags);
    }

    /// <summary>
    ///     The strided loop of every <c>StringDType</c> cast (the four casting impls of this file): NumPy's
    ///     <c>stringdtype/casts.cpp</c> loop of the pair (<see cref="Backends.Strings.NDVStringCasting.TransferRun"/>:
    ///     <c>string_to_string</c> between two <c>StringDType</c> descriptors, the encode / decode of <c>U</c> and <c>S</c>, bool,
    ///     the numbers' <c>str()</c> and their parse), with the allocators of both sides locked for each call, as NumPy's loops
    ///     acquire them (<c>NpyString_acquire_allocators</c>).
    /// </summary>
    /// <remarks>
    ///     The lock is a re-entrant monitor, so a caller that already holds an allocator for a whole walk nests safely. The
    ///     allocators come from the context (NumSharp keeps them on the cells' buffers, NumPy on the descriptors).
    /// </remarks>
    internal static class VStringCastLoop
    {
        /// <summary>Builds the loop of <paramref name="impl"/> for <paramref name="context"/>.</summary>
        /// <param name="impl">The <c>StringDType</c> cast (its flags are the loop's).</param>
        /// <param name="context">The resolved descriptors and the allocators of the <c>StringDType</c> sides.</param>
        /// <param name="flags">Receives the method's flags.</param>
        /// <returns>The loop: [0] the source run, [1] the destination run, byte strides.</returns>
        /// <exception cref="ArgumentException">A <c>StringDType</c> side without its allocator in the context.</exception>
        /// <exception cref="NotSupportedException">The other class has no storage (datetime64 / timedelta64 before
        ///     Stage C).</exception>
        internal static unsafe NDInnerLoopFunc Get(CastingImpl impl, ArrayMethodContext context, out NDArrayMethodFlags flags)
        {
            flags = impl.flags;
            var from = context.descriptors[0];
            var to = context.descriptors[1];
            var fromAllocator = context.Allocators?[0];
            var toAllocator = context.Allocators?[1];
            var run = Backends.Strings.NDVStringCasting.TransferRun(from, fromAllocator, to, toAllocator);
            return (ptrs, st, count, aux) =>
            {
                Acquire(fromAllocator, toAllocator);
                try
                {
                    run((byte*)ptrs[0], st[0], (byte*)ptrs[1], st[1], count);
                }
                finally
                {
                    Release(fromAllocator, toAllocator);
                }
            };
        }

        /// <summary>Locks the allocators that exist (one, or both in creation order: no deadlock between two casts).</summary>
        /// <param name="a">The source side's allocator, or null.</param>
        /// <param name="b">The destination side's allocator, or null.</param>
        private static void Acquire(Backends.Strings.VStringAllocator a, Backends.Strings.VStringAllocator b)
        {
            if (a is not null && b is not null)
                Backends.Strings.VStringAllocator.acquire_allocators(a, b);
            else
                (a ?? b)?.acquire_allocator();
        }

        /// <summary>Releases what <see cref="Acquire"/> took.</summary>
        /// <param name="a">The source side's allocator, or null.</param>
        /// <param name="b">The destination side's allocator, or null.</param>
        private static void Release(Backends.Strings.VStringAllocator a, Backends.Strings.VStringAllocator b)
        {
            if (a is not null && b is not null)
                Backends.Strings.VStringAllocator.release_allocators(a, b);
            else
                (a ?? b)?.release_allocator();
        }
    }
}
