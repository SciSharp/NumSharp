using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Iteration;

// =============================================================================
// NDReduceLikeResolution.cs — NumPy's loop resolution for add / multiply reductions, ported
// =============================================================================
//
// A ufunc reduction (np.add.reduce, np.multiply.accumulate, and every function NumPy builds on them: np.trace, np.cumsum,
// np.cumprod, numpy.matrix's sum / prod / mean / var / std) picks its inner loop with reducelike_promote_and_resolve
// (numpy/_core/src/umath/ufunc_object.c), which runs the ufunc dispatcher on the operation DTypes (out, input, out):
//
//   promote_and_get_ufuncimpl → promote_and_get_info_and_ufuncimpl (dispatching.cpp)
//     1. resolve_implementation_info: the best registered loop or promoter whose DTypes match;
//     2. a promoter's new DTypes, recursively (the StringDType promoters, the reduction special path);
//     3. the legacy type resolver (PyUFunc_AdditionTypeResolver / PyUFunc_MultiplicationTypeResolver), which may ignore
//        the dtype= request and MUTATE the signature (legacy_promote_using_legacy_type_resolver).
//   resolve_descriptors → the loop's descriptors (PyArray_CastDescrToDType per operand, then the loop's own rule);
//   the equivalence checks ("the resolved dtypes are not compatible with ...") and the unsafe casting check.
//
// When text meets numbers this decides everything a caller sees: whether np.trace(U, dtype='f8', out=U6) concatenates
// (a refusal: the string add loop's widths grow), whether np.matrix(i8).prod(dtype='f8', out=U6) repeats the first
// count's text (it does: the legacy multiplication resolver sees a string out and an integer input and ignores the
// request), which error text a T operand gets. Every rule below is NumPy 2.4.2's source, statement for statement.
//
// One NumPy artifact is NOT reproduced: the dispatch CACHE. NumPy stores the loop a call resolved under the call's
// operation DTypes, and a later call with the same key reuses it even where a fresh resolution would differ
// (np.add.reduce(u, dtype='f8', out=U6) raises in a fresh process, but computes after np.add.reduce(u, dtype='f8',
// out=f8_array) cached (Float64DType, StrDType, Float64DType) -> ddd). NumSharp always answers as a fresh NumPy process
// does: the cache is meant to be transparent, and a history-dependent answer is not an API anyone can rely on.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The loop families of NumPy's <c>add</c> and <c>multiply</c> a reduction can resolve to — which decides how the
    ///     reduction computes once its descriptors passed NumPy's checks.
    /// </summary>
    internal enum ReduceLoopKind
    {
        /// <summary>
        ///     A legacy numeric loop <c>(X, X, X)</c> (bool's logical or / and, the integers, the floats, complex, and
        ///     NumSharp's Decimal and Char): text operands are CAST to <c>X</c> (a parse in, a <c>str()</c> out) around it.
        /// </summary>
        Numeric,

        /// <summary>The fixed-width string add loop <c>(S, S, S)</c> / <c>(U, U, U)</c> (concatenation).</summary>
        StringAdd,

        /// <summary>
        ///     The fixed-width string repeat loop: <c>(S|U, int64, S|U)</c> with the count second, or
        ///     <c>(int64, S|U, S|U)</c> with the count first (<see cref="ReduceLikeResolution.CountFirst"/>).
        /// </summary>
        StringRepeat,

        /// <summary>The <c>StringDType</c> add loop <c>(T, T, T)</c> (concatenation).</summary>
        VStringAdd,

        /// <summary>
        ///     The <c>StringDType</c> repeat loop: <c>(T, int64|uint64, T)</c>, or <c>(int64|uint64, T, T)</c> with the count
        ///     first.
        /// </summary>
        VStringRepeat,
    }

    /// <summary>
    ///     The loop and descriptors NumPy's <c>reducelike_promote_and_resolve</c> picked for an <c>add</c> / <c>multiply</c>
    ///     reduction or accumulation (after its equivalence and casting checks passed).
    /// </summary>
    internal sealed class ReduceLikeResolution
    {
        /// <summary>Creates the outcome of a resolution.</summary>
        /// <param name="kind">The loop family.</param>
        /// <param name="loopDTypes">The loop's registered DTypes (out-slot, input, output).</param>
        /// <param name="descriptors">The resolved descriptors, with NumPy's <c>out_descrs[0] = out_descrs[2]</c> applied.</param>
        internal ReduceLikeResolution(ReduceLoopKind kind, DTypeMeta[] loopDTypes, DType[] descriptors)
        {
            Kind = kind;
            LoopDTypes = loopDTypes;
            Descriptors = descriptors;
        }

        /// <summary>The loop family (decides the computation).</summary>
        internal ReduceLoopKind Kind { get; }

        /// <summary>The loop's DTypes: slot 0 is the accumulator (the out), slot 1 the reduced input, slot 2 the output.</summary>
        internal DTypeMeta[] LoopDTypes { get; }

        /// <summary>
        ///     The descriptors the loop runs with: <c>[0]</c> and <c>[2]</c> are the accumulator's (NumPy replaces the first by
        ///     the output's once they are equivalent), <c>[1]</c> the reduced input's — what the input is cast to.
        /// </summary>
        internal DType[] Descriptors { get; }

        /// <summary>True for a repeat loop whose COUNT is the first operand (<c>(int64, S, S)</c>, <c>(int64, T, T)</c>).</summary>
        internal bool CountFirst => Kind is ReduceLoopKind.StringRepeat or ReduceLoopKind.VStringRepeat
                                    && LoopDTypes[0] is not StringDTypeMeta && LoopDTypes[0] is not VStringDTypeMeta;
    }

    /// <summary>
    ///     NumPy 2.4.2's loop resolution for a reduction or accumulation of <c>add</c> or <c>multiply</c>, ported (the file
    ///     header names the C functions): the dispatcher over the loops and promoters both ufuncs register, the legacy type
    ///     resolvers, the descriptor resolution of each loop family, and the checks that follow.
    /// </summary>
    /// <remarks>
    ///     The registered loops are modelled as NumPy registers them, in its order: every legacy loop of the ufunc's type list
    ///     (NumPy adds all of them when it creates the ufunc), then the fixed-width string loops (<c>init_string_ufuncs</c>),
    ///     then the <c>StringDType</c> loops and promoters (<c>init_stringdtype_ufuncs</c>). NumSharp's Decimal and Char lanes
    ///     join the legacy list as homogeneous loops: NumSharp computes them, NumPy has no such dtypes. Datetime and object
    ///     loops are left out: no NumSharp array can hold them.
    /// </remarks>
    internal static class NDReduceLikeResolution
    {
        /// <summary>
        ///     NumPy's type numbers of the flexible and special classes the legacy resolvers branch on (<c>ndarraytypes.h</c>:
        ///     <c>NPY_BOOL</c>, <c>NPY_OBJECT</c>, <c>NPY_STRING</c> (bytes), <c>NPY_UNICODE</c>, <c>NPY_VOID</c>). The resolvers
        ///     test these NUMBERS, not the classes: a fixed-width string operand is recognized by <c>type_num</c>, which is why
        ///     two <c>U</c> operands of different widths still take the string add loop.
        /// </summary>
        private const int NPY_BOOL = 0, NPY_OBJECT = 17, NPY_STRING = 18, NPY_UNICODE = 19, NPY_VOID = 20;

        /// <summary>
        ///     NumPy's type numbers of the datetime pair (<c>NPY_DATETIME</c>, <c>NPY_TIMEDELTA</c>: the uniform resolver's
        ///     <c>PyTypeNum_ISDATETIME</c> test) and the count of legacy builtin types (<c>NPY_NTYPES_LEGACY</c>): a type number
        ///     at or past it belongs to a user or new-style class (<c>StringDType</c>, NumSharp's Decimal and Char), which the
        ///     uniform resolver treats like object and hands to the default resolver.
        /// </summary>
        private const int NPY_DATETIME = 21, NPY_TIMEDELTA = 22, NPY_NTYPES_LEGACY = 24;

        /// <summary>The marker of a "not specified" slot in a legacy type tuple (NumPy's <c>NPY_NOTYPE</c>).</summary>
        private const int NPY_NOTYPE = -1;

        /// <summary>
        ///     The two promoters <c>add</c> / <c>multiply</c> register (the C functions of <c>stringdtype_ufuncs.cpp</c>).
        /// </summary>
        private enum PromoterKind
        {
            /// <summary>Not a promoter: a loop.</summary>
            None,

            /// <summary><c>all_strings_promoter</c>: <c>StringDType</c> with <c>StrDType</c> operands of <c>add</c>.</summary>
            AllStrings,

            /// <summary><c>string_multiply_promoter</c>: <c>StringDType</c> with an integer count of <c>multiply</c>.</summary>
            StringMultiply,
        }

        /// <summary>
        ///     One entry of a ufunc's loop list (NumPy's <c>ufunc._loops</c>): a loop or a promoter and the DTypes it was
        ///     registered for.
        /// </summary>
        private sealed class Entry
        {
            /// <summary>Creates an entry.</summary>
            /// <param name="dtypes">The registered DTypes per slot; null marks NumPy's abstract integer DType
            ///     (<c>PyArray_IntAbstractDType</c>) in that slot.</param>
            /// <param name="kind">The loop family (unused for a promoter).</param>
            /// <param name="promoter">The promoter, or <see cref="PromoterKind.None"/> for a loop.</param>
            internal Entry(DTypeMeta[] dtypes, ReduceLoopKind kind, PromoterKind promoter = PromoterKind.None)
            {
                DTypes = dtypes;
                Kind = kind;
                Promoter = promoter;
            }

            /// <summary>The registered DTypes; a null slot is the abstract integer DType.</summary>
            internal DTypeMeta[] DTypes { get; }

            /// <summary>The loop family.</summary>
            internal ReduceLoopKind Kind { get; }

            /// <summary>The promoter, or <see cref="PromoterKind.None"/>.</summary>
            internal PromoterKind Promoter { get; }

            /// <summary>True for a promoter entry.</summary>
            internal bool IsPromoter => Promoter != PromoterKind.None;

            /// <summary>True when slot <paramref name="i"/> is the abstract integer DType.</summary>
            /// <param name="i">The slot.</param>
            /// <returns>Whether the slot is abstract.</returns>
            internal bool IsAbstract(int i) => DTypes[i] is null;
        }

        /// <summary>The legacy homogeneous loop classes in NumPy's type-list order (see <see cref="BuildLegacyTypes"/>).</summary>
        private static readonly DTypeMeta[] s_legacyTypes = BuildLegacyTypes();

        /// <summary><c>np.add._loops</c> as NumPy registers it (legacy loops, fixed-width string loops, StringDType).</summary>
        private static readonly Entry[] s_add = BuildAdd();

        /// <summary><c>np.multiply._loops</c> as NumPy registers it.</summary>
        private static readonly Entry[] s_multiply = BuildMultiply();

        /// <summary>
        ///     The classes of the ufunc's legacy loop list in NumPy's order — <c>"?bBhHiIlLqQefdgFDG"</c>, resolved through
        ///     the host's C <c>long</c> model (<see cref="DTypeRegistry.FromTypeNum"/>: on Windows <c>'i'</c> is the
        ///     lane-sharing <c>IntDType</c> and <c>'l'</c> the int32 class; on Linux <c>'l'</c> is the int64 class and
        ///     <c>'q'</c> the lane-sharing <c>LongLongDType</c>) — then NumSharp's Decimal and Char. NumSharp has no
        ///     complex64 (<c>'F'</c>), which is skipped.
        /// </summary>
        /// <returns>The classes, each once.</returns>
        private static DTypeMeta[] BuildLegacyTypes()
        {
            var nums = new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 23, 11, 12, 13, 14, 15, 16 };
            var list = new List<DTypeMeta>();
            foreach (int n in nums)
            {
                var meta = DTypeRegistry.FromTypeNum(n);
                if (meta is not null && meta.HasStorage && !list.Contains(meta))
                    list.Add(meta);
            }
            list.Add(DTypeRegistry.Decimal);
            list.Add(DTypeRegistry.Char);
            return list.ToArray();
        }

        /// <summary>The homogeneous legacy loops <c>(X, X, X)</c> of <see cref="s_legacyTypes"/>.</summary>
        /// <returns>One entry per class.</returns>
        private static IEnumerable<Entry> LegacyLoops()
            => s_legacyTypes.Select(m => new Entry(new[] { m, m, m }, ReduceLoopKind.Numeric));

        /// <summary>
        ///     <c>np.add</c>'s loop list: the legacy loops, then <c>init_string_ufuncs</c>' <c>(S, S, S)</c> and
        ///     <c>(U, U, U)</c>, then <c>init_stringdtype_ufuncs</c>' <c>(T, T, T)</c> loop and its three
        ///     <c>all_strings_promoter</c> registrations <c>(T, U, T)</c>, <c>(U, T, T)</c>, <c>(U, U, T)</c>.
        /// </summary>
        /// <returns>The entries in registration order.</returns>
        private static Entry[] BuildAdd()
        {
            var S = DTypeRegistry.Bytes;
            var U = DTypeRegistry.Str;
            var T = DTypeRegistry.VString;
            var list = new List<Entry>(LegacyLoops())
            {
                new Entry(new DTypeMeta[] { S, S, S }, ReduceLoopKind.StringAdd),
                new Entry(new DTypeMeta[] { U, U, U }, ReduceLoopKind.StringAdd),
                new Entry(new DTypeMeta[] { T, T, T }, ReduceLoopKind.VStringAdd),
                new Entry(new DTypeMeta[] { T, U, T }, ReduceLoopKind.VStringAdd, PromoterKind.AllStrings),
                new Entry(new DTypeMeta[] { U, T, T }, ReduceLoopKind.VStringAdd, PromoterKind.AllStrings),
                new Entry(new DTypeMeta[] { U, U, T }, ReduceLoopKind.VStringAdd, PromoterKind.AllStrings),
            };
            return list.ToArray();
        }

        /// <summary>
        ///     <c>np.multiply</c>'s loop list: the legacy loops, then the fixed-width repeat loops <c>(S, int64, S)</c>,
        ///     <c>(U, int64, U)</c>, <c>(int64, S, S)</c>, <c>(int64, U, U)</c>, then <c>INIT_MULTIPLY(Int64)</c> and
        ///     <c>INIT_MULTIPLY(UInt64)</c> (the <c>StringDType</c> repeat loops, each count on either side) and the two
        ///     <c>string_multiply_promoter</c> registrations with the abstract integer DType.
        /// </summary>
        /// <returns>The entries in registration order.</returns>
        private static Entry[] BuildMultiply()
        {
            var S = DTypeRegistry.Bytes;
            var U = DTypeRegistry.Str;
            var T = DTypeRegistry.VString;
            var I64 = DTypeRegistry.Int64;
            var U64 = DTypeRegistry.UInt64;
            var list = new List<Entry>(LegacyLoops())
            {
                new Entry(new DTypeMeta[] { S, I64, S }, ReduceLoopKind.StringRepeat),
                new Entry(new DTypeMeta[] { U, I64, U }, ReduceLoopKind.StringRepeat),
                new Entry(new DTypeMeta[] { I64, S, S }, ReduceLoopKind.StringRepeat),
                new Entry(new DTypeMeta[] { I64, U, U }, ReduceLoopKind.StringRepeat),
                new Entry(new DTypeMeta[] { T, I64, T }, ReduceLoopKind.VStringRepeat),
                new Entry(new DTypeMeta[] { I64, T, T }, ReduceLoopKind.VStringRepeat),
                new Entry(new DTypeMeta[] { T, U64, T }, ReduceLoopKind.VStringRepeat),
                new Entry(new DTypeMeta[] { U64, T, T }, ReduceLoopKind.VStringRepeat),
                new Entry(new DTypeMeta[] { T, null, T }, ReduceLoopKind.VStringRepeat, PromoterKind.StringMultiply),
                new Entry(new DTypeMeta[] { null, T, T }, ReduceLoopKind.VStringRepeat, PromoterKind.StringMultiply),
            };
            return list.ToArray();
        }

        /// <summary>
        ///     NumPy's <c>reducelike_promote_and_resolve</c> for <c>add</c> / <c>multiply</c>: the loop and the descriptors of
        ///     a reduction (or accumulation) of an array of <paramref name="arr"/> into an out of <paramref name="out"/> under
        ///     the <c>dtype=</c> class <paramref name="requested"/>.
        /// </summary>
        /// <param name="ufunc"><c>"add"</c> or <c>"multiply"</c>.</param>
        /// <param name="accumulate">True for <c>ufunc.accumulate</c> (its inputs must also be equivalent to the output).</param>
        /// <param name="arr">The reduced input's descriptor.</param>
        /// <param name="out">The out's descriptor, or null without <c>out=</c>.</param>
        /// <param name="requested">The <c>dtype=</c> class (a legacy class: <c>_get_dtype</c> refused a new-style instance
        ///     before), or null.</param>
        /// <returns>The loop and its descriptors.</returns>
        /// <exception cref="TypeError">NumPy's refusals, verbatim: no loop (<c>ufunc 'add' did not contain a loop with
        ///     signature matching types ...</c>, the operation DTypes as the dispatcher held them; or the operand DESCRIPTORS
        ///     when the legacy uniform resolver met a string operand without a request), the legacy resolvers' texts
        ///     (<c>No loop matching the specified signature and casting was found for ufunc add</c>, <c>ufunc 'add' not
        ///     supported for the input types, ... casting rule ''safe''</c>, <c>ufunc 'add' cannot use operands with types
        ///     dtype('S6') and dtype('&lt;U1')</c>), a cast to the loop's class that cannot be sized (<c>cannot cast dtype
        ///     StringDType() to &lt;class 'numpy.dtypes.StrDType'&gt;.</c>), the string loops' own rules (<c>The 'out' kwarg is
        ///     necessary ...</c>, an addition result too long, two incompatible NA objects), and the equivalence check
        ///     (<c>the resolved dtypes are not compatible with add.reduce. Resolved (dtype('&lt;U6'), dtype('&lt;U1'),
        ///     dtype('&lt;U7'))</c>).</exception>
        /// <exception cref="ArgumentException">An operand cannot be cast to its loop descriptor at all (NumPy's
        ///     <c>Cannot cast ufunc ... with casting rule 'unsafe'</c>; the house type of that UFuncTypeError).</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="ufunc"/> is neither add nor multiply.</exception>
        internal static ReduceLikeResolution Resolve(string ufunc, bool accumulate, DType arr, DType @out, DTypeMeta requested)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var table = ufunc switch
            {
                "add" => s_add,
                "multiply" => s_multiply,
                _ => throw new ArgumentOutOfRangeException(nameof(ufunc), ufunc, "Only add and multiply reductions are ported here."),
            };

            var signature = new DTypeMeta[3];
            signature[0] = requested;
            // Without dtype= and out=, add / multiply reduce booleans and small integers in the platform integer (NumPy's
            // "TODO: should be handled by a promoter" block).
            if (requested is null && @out is null)
            {
                int tn = arr.num;
                if (tn == NPY_BOOL)
                    signature[0] = DTypeRegistry.Int64;
                else if (IsIntegerNum(tn) && arr.itemsize < 8)
                    signature[0] = arr.kind == 'u' ? DTypeRegistry.UInt64 : DTypeRegistry.Int64;
            }
            signature[2] = signature[0];

            // NumPy passes the out (or the input, when there is no out) as the first operand: the legacy resolvers cannot
            // handle a NULL first operand.
            var ops = new[] { @out ?? arr, arr, @out };
            var opDTypes = new[] { @out?.Meta, arr.Meta, @out?.Meta };
            var loop = PromoteAndGetUfuncImpl(table, ufunc, ops, signature, opDTypes, ensureReduceCompatible: true);

            var descrs = ResolveDescriptors(loop, ops, signature);
            // The first operand and the output are the same array, so their descriptors must be equivalent; an accumulation
            // also feeds every output back as an input.
            if (!DTypeCasting.EquivTypes(descrs[0], descrs[2]) || accumulate && !DTypeCasting.EquivTypes(descrs[0], descrs[1]))
                throw new TypeError(
                    $"the resolved dtypes are not compatible with {ufunc}.{(accumulate ? "accumulate" : "reduce")}. " +
                    $"Resolved ({descrs[0].ToString(true)}, {descrs[1].ToString(true)}, {descrs[2].ToString(true)})");
            // After the check NumPy enforces the out's descriptor (needed by the string dtypes).
            descrs[0] = descrs[2];
            ValidateCasting(ufunc, ops, descrs, NPY_CASTING.NPY_UNSAFE_CASTING);
            return new ReduceLikeResolution(loop.Kind, (DTypeMeta[])signature.Clone(), descrs);
        }

        // ---------------------------------------------------------------------------------------------
        // The dispatcher (dispatching.cpp)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>promote_and_get_ufuncimpl</c>: the operation DTypes are overridden by the signature, the dispatcher finds a
        ///     loop, a loop that is not reduce-compatible (its first and last DTypes differ) is retried with its output DType
        ///     forced into the first slot, and the signature is filled in from the loop.
        /// </summary>
        /// <param name="table">The ufunc's loop list.</param>
        /// <param name="ufunc">The ufunc's name (error texts).</param>
        /// <param name="ops">The operands' descriptors (out or input, input, out or null).</param>
        /// <param name="signature">The fixed DTypes; filled in from the loop (and possibly mutated by the legacy resolver).</param>
        /// <param name="opDTypes">The operation DTypes; overridden in place, as NumPy's array is (its error texts read it).</param>
        /// <param name="ensureReduceCompatible">True on the first call of a reduction.</param>
        /// <returns>The loop entry.</returns>
        /// <exception cref="TypeError">No loop (the no-loop text over <paramref name="opDTypes"/>), or a resolver's refusal.</exception>
        private static Entry PromoteAndGetUfuncImpl(Entry[] table, string ufunc, DType[] ops, DTypeMeta[] signature,
            DTypeMeta[] opDTypes, bool ensureReduceCompatible)
        {
            bool legacyPossible = true;
            for (int i = 0; i < 3; i++)
            {
                if (signature[i] is not null)
                    opDTypes[i] = signature[i];
                else if (i >= 2)
                    // Outputs not in the signature are ignored by the dispatcher.
                    opDTypes[i] = null;
                // A non-legacy DType (StringDType) rules the legacy resolvers out (an array is never a Python literal here).
                if (opDTypes[i] is not null && !opDTypes[i]._legacy)
                    legacyPossible = false;
            }

            Entry info;
            try
            {
                info = PromoteAndGetInfo(table, ufunc, ops, signature, opDTypes, legacyPossible);
            }
            catch (DTypePromotionError e)
            {
                // A promotion failure means no loop: NumPy re-raises its no-loop error, chained to the cause.
                throw NoLoop(ufunc, opDTypes, e);
            }
            if (info is null)
                throw NoLoop(ufunc, opDTypes);

            // Only the logical ufuncs really reach this: a loop whose input and output DTypes differ cannot reduce, so the
            // result DType is forced into the first slot and the lookup repeats.
            if (ensureReduceCompatible && signature[0] is null && !ReferenceEquals(info.DTypes[0], info.DTypes[2]))
            {
                signature[0] = info.DTypes[2];
                return PromoteAndGetUfuncImpl(table, ufunc, ops, signature, opDTypes, ensureReduceCompatible: false);
            }

            for (int i = 0; i < 3; i++)
            {
                if (signature[i] is null)
                    signature[i] = info.DTypes[i];
                else if (!ReferenceEquals(signature[i], info.DTypes[i]))
                    // A forced signature the loop does not honor (NumPy guards its cache with this check).
                    throw NoLoop(ufunc, opDTypes);
            }
            return info;
        }

        /// <summary>
        ///     <c>promote_and_get_info_and_ufuncimpl</c> without NumPy's cache (see the file header): the best registered
        ///     loop; else a promoter's DTypes (or, for a reduction without a first DType, NumPy's reduction special path),
        ///     recursively; else the legacy type resolver, whose result DTypes are looked up again.
        /// </summary>
        /// <param name="table">The loop list.</param>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="ops">The operands' descriptors.</param>
        /// <param name="signature">The signature (the legacy resolver may mutate it).</param>
        /// <param name="opDTypes">The operation DTypes of this lookup.</param>
        /// <param name="legacyPossible">Whether the legacy resolver may run.</param>
        /// <returns>The loop entry, or null when none is found.</returns>
        /// <exception cref="TypeError">A legacy resolver's refusal.</exception>
        /// <exception cref="RuntimeError">Two promoters match equally well (NumPy's internal error; cannot happen with the
        ///     modelled tables).</exception>
        private static Entry PromoteAndGetInfo(Entry[] table, string ufunc, DType[] ops, DTypeMeta[] signature,
            DTypeMeta[] opDTypes, bool legacyPossible)
        {
            var info = ResolveImplementationInfo(table, opDTypes, onlyPromoters: false);
            if (info is not null && !info.IsPromoter)
                return info;

            // A promoter, or the reduction special path when the first DType is unknown.
            if (info is not null || opDTypes[0] is null)
            {
                var promoted = CallPromoterAndRecurse(table, ufunc, info, opDTypes, signature, ops);
                if (promoted is not null)
                    return promoted;
            }

            if (!legacyPossible)
                return null;

            var newDTypes = LegacyPromote(ufunc, ops, signature);
            // NumPy registers every legacy loop when it creates the ufunc, so the lookup of the legacy result finds it (the
            // "loop_exists" fallback of the C code serves ufuncs that add loops later).
            return PromoteAndGetInfo(table, ufunc, ops, signature, newDTypes, legacyPossible: false);
        }

        /// <summary>
        ///     <c>resolve_implementation_info</c>: the best entry whose DTypes match the operation DTypes. A slot matches when
        ///     the operation DType is unknown for an output, equals the entry's DType, or subclasses the entry's abstract DType.
        ///     Among several matches a concrete DType beats an abstract one, an exact match beats another concrete one, and
        ///     the inputs decide before the output; an ambiguity is retried with the promoters alone.
        /// </summary>
        /// <param name="table">The loop list.</param>
        /// <param name="opDTypes">The operation DTypes.</param>
        /// <param name="onlyPromoters">True on the retry after an ambiguity.</param>
        /// <returns>The entry, or null.</returns>
        /// <exception cref="RuntimeError">Two promoters match equally well.</exception>
        /// <exception cref="NotImplementedException">Two different abstract DTypes compete for a slot (NumPy's
        ///     NotImplementedError; cannot happen with the modelled tables).</exception>
        private static Entry ResolveImplementationInfo(Entry[] table, DTypeMeta[] opDTypes, bool onlyPromoters)
        {
            Entry best = null;
            foreach (var entry in table)
            {
                if (onlyPromoters && !entry.IsPromoter)
                    continue;

                bool matches = true;
                for (int i = 0; i < 3 && matches; i++)
                {
                    var given = opDTypes[i];
                    if (given is null)
                    {
                        // An unspecified output always matches; an unspecified first input matches only an entry registered
                        // with None there (a reduction pattern none of these ufuncs registers).
                        matches = i >= 2;
                        continue;
                    }
                    if (entry.IsAbstract(i))
                        matches = IsIntegerAbstractSubclass(given);
                    else
                        matches = ReferenceEquals(given, entry.DTypes[i]);
                }
                if (!matches)
                    continue;

                if (best is not null)
                {
                    int currentBest = -1; // -1 neither, 0 the current best, 1 the new entry
                    for (int i = 0; i < 3; i++)
                    {
                        // The inputs prefer one loop; the output has lower priority.
                        if (i == 2 && currentBest != -1)
                            break;
                        bool prevAbstract = best.IsAbstract(i), newAbstract = entry.IsAbstract(i);
                        bool same = prevAbstract ? newAbstract : !newAbstract && ReferenceEquals(best.DTypes[i], entry.DTypes[i]);
                        if (same || opDTypes[i] is null)
                            continue;
                        int better;
                        if (!prevAbstract && !newAbstract)
                            // Two concrete DTypes: ambiguous unless one matches exactly.
                            better = ReferenceEquals(best.DTypes[i], opDTypes[i]) ? 0 : ReferenceEquals(entry.DTypes[i], opDTypes[i]) ? 1 : -1;
                        else if (!prevAbstract)
                            better = 0;
                        else if (!newAbstract)
                            better = 1;
                        else
                            throw new NotImplementedException(
                                "deciding which one of two abstract dtypes is a better match is not yet implemented.  " +
                                "This will pick the better (or bail) in the future.");
                        if (better == -1)
                            continue;
                        if (currentBest != -1 && currentBest != better)
                        {
                            currentBest = -1;
                            break;
                        }
                        currentBest = better;
                    }

                    if (currentBest == -1)
                    {
                        // Promoters are designed to disambiguate: retry with them alone.
                        if (!onlyPromoters)
                            return ResolveImplementationInfo(table, opDTypes, onlyPromoters: true);
                        throw new RuntimeError(
                            $"Could not find a loop for the inputs:\n    ({string.Join(", ", opDTypes.Select(Name))})\n" +
                            "The two promoters matched the input equally well.  Promoters must be designed to be unambiguous.  " +
                            "NOTE: This indicates an error in NumPy or an extending library and should be reported.");
                    }
                    if (currentBest == 0)
                        continue;
                }
                best = entry;
            }
            return best;
        }

        /// <summary>
        ///     <c>call_promoter_and_recurse</c>: the promoter's new operation DTypes (or, without a promoter, the reduction
        ///     special path <c>(input, input, output)</c>) looked up again without the legacy resolver. A promoter that
        ///     changes nothing, or refuses, finds nothing.
        /// </summary>
        /// <param name="table">The loop list.</param>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="info">The matched promoter, or null for the reduction special path.</param>
        /// <param name="opDTypes">The operation DTypes.</param>
        /// <param name="signature">The signature.</param>
        /// <param name="ops">The operands' descriptors.</param>
        /// <returns>The loop the recursion found, or null.</returns>
        private static Entry CallPromoterAndRecurse(Entry[] table, string ufunc, Entry info, DTypeMeta[] opDTypes,
            DTypeMeta[] signature, DType[] ops)
        {
            DTypeMeta[] newDTypes;
            if (info is not null)
            {
                newDTypes = RunPromoter(info.Promoter, opDTypes, signature);
                // A refusal without an error (all_strings_promoter's forced-unicode branch) finds nothing.
                if (newDTypes is null)
                    return null;
                // An unchanged tuple would recurse forever.
                bool changed = false;
                for (int i = 0; i < 3; i++)
                    changed |= !ReferenceEquals(newDTypes[i], opDTypes[i]);
                if (!changed)
                    return null;
            }
            else
            {
                newDTypes = new[] { opDTypes[1], opDTypes[1], opDTypes[2] };
            }
            return PromoteAndGetInfo(table, ufunc, ops, signature, newDTypes, legacyPossible: false);
        }

        /// <summary>
        ///     Runs one of the promoters: <c>all_strings_promoter</c> (only <c>StrDType</c> operands — reachable through a
        ///     <c>dtype=</c> request — use the unicode loop; a forced unicode signature over a <c>StringDType</c> operand
        ///     refuses; otherwise everything is <c>StringDType</c>) or <c>string_multiply_promoter</c> (a fixed input DType
        ///     stays, an integer other than int64 / uint64 becomes int64, an unknown input is <c>StringDType</c>; a known
        ///     output stays, an unknown one is <c>StringDType</c>).
        /// </summary>
        /// <param name="promoter">The promoter.</param>
        /// <param name="opDTypes">The operation DTypes.</param>
        /// <param name="signature">The signature.</param>
        /// <returns>The new operation DTypes, or null for the refusal.</returns>
        private static DTypeMeta[] RunPromoter(PromoterKind promoter, DTypeMeta[] opDTypes, DTypeMeta[] signature)
        {
            var T = DTypeRegistry.VString;
            var U = DTypeRegistry.Str;
            if (promoter == PromoterKind.AllStrings)
            {
                if (!ReferenceEquals(opDTypes[0], T) && !ReferenceEquals(opDTypes[1], T) && !ReferenceEquals(opDTypes[2], T))
                    return new DTypeMeta[] { U, U, U };
                if (ReferenceEquals(signature[0], U) && ReferenceEquals(signature[1], U) && ReferenceEquals(signature[2], U))
                    return null;
                return new DTypeMeta[] { T, T, T };
            }

            var result = new DTypeMeta[3];
            for (int i = 0; i < 2; i++)
            {
                if (signature[i] is not null)
                    result[i] = signature[i];
                else if (IsPromotedToInt64(opDTypes[i]))
                    result[i] = DTypeRegistry.Int64;
                else
                    result[i] = opDTypes[i] ?? T;
            }
            result[2] = opDTypes[2] ?? T;
            return result;
        }

        // ---------------------------------------------------------------------------------------------
        // The legacy type resolvers (ufunc_type_resolution.c)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>legacy_promote_using_legacy_type_resolver</c>: the ufunc's legacy resolver runs on the operands (unsafe
        ///     casting) with the signature as a type tuple; its result DTypes are the new operation DTypes, and a signature
        ///     DType the resolver did not honor is REPLACED by the resolver's (NumPy's "signature is mutated" branch: the
        ///     string resolvers ignore the type tuple).
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="ops">The operands' descriptors.</param>
        /// <param name="signature">The signature, mutated where the resolver ignored it.</param>
        /// <returns>The new operation DTypes.</returns>
        /// <exception cref="TypeError">The resolver's refusal.</exception>
        private static DTypeMeta[] LegacyPromote(string ufunc, DType[] ops, DTypeMeta[] signature)
        {
            var typeTup = MakeNewTypeTup(signature);
            var outDescrs = ufunc == "add" ? AdditionTypeResolver(ufunc, ops, typeTup) : MultiplicationTypeResolver(ufunc, ops, typeTup);
            var newDTypes = new DTypeMeta[3];
            for (int i = 0; i < 3; i++)
                newDTypes[i] = outDescrs[i].Meta;
            for (int i = 0; i < 3; i++)
                if (signature[i] is not null && !ReferenceEquals(signature[i], newDTypes[i]))
                    signature[i] = newDTypes[i];
            return newDTypes;
        }

        /// <summary>
        ///     <c>_make_new_typetup</c>: the signature as the legacy resolvers' type tuple — each fixed DType's singleton
        ///     descriptor, null for an unfixed slot; null altogether when nothing is fixed.
        /// </summary>
        /// <param name="signature">The signature.</param>
        /// <returns>The type tuple, or null.</returns>
        /// <exception cref="RuntimeError">A non-legacy or abstract DType is fixed (NumPy's internal error).</exception>
        private static DType[] MakeNewTypeTup(DTypeMeta[] signature)
        {
            if (signature.All(s => s is null))
                return null;
            var tup = new DType[3];
            for (int i = 0; i < 3; i++)
            {
                if (signature[i] is null)
                    continue;
                if (!signature[i]._legacy || signature[i]._abstract)
                    throw new RuntimeError("Internal NumPy error: new DType in signature not yet supported. (This should be unreachable code!)");
                tup[i] = signature[i].singleton;
            }
            return tup;
        }

        /// <summary>
        ///     <c>PyUFunc_AdditionTypeResolver</c> (strings and the uniform rule; NumSharp has no datetime arrays): two
        ///     operands of ONE string type use that string loop whatever the type tuple asks; a fixed-width bytes operand
        ///     with a unicode one is refused; everything else is the uniform resolver.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="ops">The operands' descriptors.</param>
        /// <param name="typeTup">The type tuple, or null.</param>
        /// <returns>The three descriptors (only their classes matter to the caller).</returns>
        /// <exception cref="TypeError">NumPy's binary refusal (<c>ufunc 'add' cannot use operands with types dtype('S6') and
        ///     dtype('&lt;U1')</c>) or the uniform resolver's.</exception>
        private static DType[] AdditionTypeResolver(string ufunc, DType[] ops, DType[] typeTup)
        {
            int t1 = ops[0].num, t2 = ops[1].num;
            if (!IsDatetimeNum(t1) && !IsDatetimeNum(t2) && !(IsStringNum(t1) && IsStringNum(t2)))
                return SimpleUniformOperationTypeResolver(ufunc, ops, typeTup);
            if (t1 == t2)
            {
                // "This is wrong, but only the DType matters here": the unsized descriptor of the string type.
                var d = DTypeRegistry.FromTypeNum(t1).singleton;
                return new[] { d, d, d };
            }
            throw BinaryResolutionError(ufunc, ops[0], ops[1]);
        }

        /// <summary>
        ///     <c>PyUFunc_MultiplicationTypeResolver</c> (strings and the uniform rule): a string operand with an INTEGER one
        ///     uses the repeat loop of that string type (the string side keeps its descriptor, the count is int64) whatever
        ///     the type tuple asks; everything else is the uniform resolver.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="ops">The operands' descriptors.</param>
        /// <param name="typeTup">The type tuple, or null.</param>
        /// <returns>The three descriptors.</returns>
        /// <exception cref="TypeError">The uniform resolver's refusal.</exception>
        private static DType[] MultiplicationTypeResolver(string ufunc, DType[] ops, DType[] typeTup)
        {
            int t1 = ops[0].num, t2 = ops[1].num;
            bool repeat = IsStringNum(t1) && IsIntegerNum(t2) || IsIntegerNum(t1) && IsStringNum(t2);
            if (!IsDatetimeNum(t1) && !IsDatetimeNum(t2) && !repeat)
                return SimpleUniformOperationTypeResolver(ufunc, ops, typeTup);
            if (!repeat)
                throw BinaryResolutionError(ufunc, ops[0], ops[1]);
            var count = DTypeRegistry.Int64.singleton;
            if (IsStringNum(t1))
            {
                var s = Canonical(ops[0]);
                return new[] { s, count, s };
            }
            var t = Canonical(ops[1]);
            return new[] { count, t, t };
        }

        /// <summary>
        ///     <c>PyUFunc_SimpleUniformOperationTypeResolver</c>: a custom or object input goes to the default resolver;
        ///     without a type tuple a flexible (string) input has no loop (NumPy names the operand DESCRIPTORS), else the
        ///     inputs' result type is the loop type; with a type tuple whose fixed slots name one descriptor (the outputs
        ///     fixed), that descriptor is the loop type (the first input's own when its type matches); every operand then
        ///     takes the loop type, and the casting is validated.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="ops">The operands' descriptors.</param>
        /// <param name="typeTup">The type tuple, or null.</param>
        /// <returns>The three descriptors.</returns>
        /// <exception cref="TypeError">No loop for a flexible operand, or the default resolver's refusal.</exception>
        /// <exception cref="ArgumentException">An operand cannot be cast to the loop type at all.</exception>
        private static DType[] SimpleUniformOperationTypeResolver(string ufunc, DType[] ops, DType[] typeTup)
        {
            if (IsCustomOrObject(ops[0]) || IsCustomOrObject(ops[1]))
                return DefaultTypeResolver(ufunc, ops, typeTup);

            DType loop;
            if (typeTup is null)
            {
                // "We never supported flexible dtypes here": a string input has no loop, named by the descriptors.
                if (IsFlexibleNum(ops[0].num) || IsFlexibleNum(ops[1].num))
                    throw NDStringUfuncs.VStringNoLoop(ufunc, new[] { ops[0].ToString(true), ops[1].ToString(true) });
                loop = DTypePromotion.PromoteTypes(ops[0], ops[1]);
            }
            else
            {
                // All outputs must be fixed and every fixed slot the same descriptor; otherwise the default resolver.
                DType descr = null;
                for (int i = 0; i < 3; i++)
                {
                    var item = typeTup[i];
                    if (item is null)
                    {
                        if (i < 2)
                            continue;
                        descr = null;
                        break;
                    }
                    if (descr is not null && !ReferenceEquals(descr, item))
                    {
                        descr = null;
                        break;
                    }
                    descr = item;
                }
                if (descr is null)
                    return DefaultTypeResolver(ufunc, ops, typeTup);
                // Prefer the input descriptor if it matches (preserves metadata).
                if (descr.num == ops[0].num)
                    descr = ops[0];
                loop = Canonical(descr);
            }

            var result = new[] { loop, loop, loop };
            ValidateCasting(ufunc, ops, result, NPY_CASTING.NPY_UNSAFE_CASTING);
            return result;
        }

        /// <summary>
        ///     <c>PyUFunc_DefaultTypeResolver</c>: inputs must cast SAFELY to a loop (outputs unsafely); without a type tuple
        ///     the first loop of the type list that fits (<c>linear_search_type_resolver</c>), with one the first loop that
        ///     honors its fixed slots (<c>type_tuple_type_resolver</c>).
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="ops">The operands' descriptors.</param>
        /// <param name="typeTup">The type tuple, or null.</param>
        /// <returns>The three descriptors.</returns>
        /// <exception cref="TypeError">The search found no loop.</exception>
        private static DType[] DefaultTypeResolver(string ufunc, DType[] ops, DType[] typeTup)
        {
            var inputCasting = NPY_CASTING.NPY_SAFE_CASTING;
            var casting = NPY_CASTING.NPY_UNSAFE_CASTING;
            return typeTup is null
                ? LinearSearchTypeResolver(ufunc, ops, inputCasting, casting)
                : TypeTupleTypeResolver(ufunc, ops, typeTup, inputCasting, casting);
        }

        /// <summary>
        ///     <c>linear_search_type_resolver</c>: the first loop of the type list every input casts to under
        ///     <paramref name="inputCasting"/> and whose output casts to the given out under <paramref name="outputCasting"/>.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="ops">The operands' descriptors.</param>
        /// <param name="inputCasting">The input rule (safe).</param>
        /// <param name="outputCasting">The output rule.</param>
        /// <returns>The loop's descriptors.</returns>
        /// <exception cref="TypeError">No loop: <c>ufunc 'add' output (typecode 'd') could not be coerced to provided output
        ///     parameter (typecode 'U') according to the casting rule ''unsafe''</c> when a loop's inputs fit but no output
        ///     did, else <c>ufunc 'add' not supported for the input types, and the inputs could not be safely coerced to any
        ///     supported types according to the casting rule ''safe''</c> (NumPy's doubled quotes:
        ///     <c>npy_casting_to_string</c> already quotes the rule).</exception>
        private static DType[] LinearSearchTypeResolver(string ufunc, DType[] ops, NPY_CASTING inputCasting, NPY_CASTING outputCasting)
        {
            bool noCastableOutput = false;
            char errSrc = '-', errDst = '-';
            foreach (var meta in s_legacyTypes)
            {
                var types = new[] { meta.type_num, meta.type_num, meta.type_num };
                if (LoopMatches(ops, inputCasting, outputCasting, types, ref noCastableOutput, ref errSrc, ref errDst))
                    return SetUfuncLoopDataTypes(ops, types);
            }
            if (noCastableOutput)
                throw new TypeError(
                    $"ufunc '{ufunc}' output (typecode '{errSrc}') could not be coerced to provided output parameter (typecode " +
                    $"'{errDst}') according to the casting rule '{CastingToString(outputCasting)}'");
            throw new TypeError(
                $"ufunc '{ufunc}' not supported for the input types, and the inputs could not be safely coerced to any supported " +
                $"types according to the casting rule '{CastingToString(inputCasting)}'");
        }

        /// <summary>
        ///     <c>type_tuple_type_resolver</c>: the first loop that equals every fixed slot of the type tuple and accepts the
        ///     other operands; failing that, the unfixed inputs take the (homogeneous) output type and the search repeats.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="ops">The operands' descriptors.</param>
        /// <param name="typeTup">The type tuple.</param>
        /// <param name="inputCasting">The input rule (safe).</param>
        /// <param name="casting">The rule of the final check (unsafe).</param>
        /// <returns>The loop's descriptors.</returns>
        /// <exception cref="TypeError"><c>No loop matching the specified signature and casting was found for ufunc add</c>.</exception>
        /// <exception cref="ArgumentException">An operand cannot be cast to the found loop at all.</exception>
        private static DType[] TypeTupleTypeResolver(string ufunc, DType[] ops, DType[] typeTup, NPY_CASTING inputCasting, NPY_CASTING casting)
        {
            var specified = new int[3];
            for (int i = 0; i < 3; i++)
                specified[i] = typeTup[i]?.num ?? NPY_NOTYPE;

            var res = TypeTupleTypeResolverCore(ufunc, ops, inputCasting, casting, specified);
            if (res is not null)
                return res;

            // dtype=X becomes the signature (None, None, X); a reduction passes (X, None, X): the unspecified inputs take the
            // homogeneous output type (which often casts unsafely — reductions are not refused for it).
            int homogeneous = specified[2];
            if (homogeneous != NPY_NOTYPE)
            {
                for (int i = 0; i < 2; i++)
                    if (specified[i] == NPY_NOTYPE)
                        specified[i] = homogeneous;
                res = TypeTupleTypeResolverCore(ufunc, ops, inputCasting, casting, specified);
                if (res is not null)
                    return res;
            }
            throw new TypeError($"No loop matching the specified signature and casting was found for ufunc {ufunc}");
        }

        /// <summary>
        ///     <c>type_tuple_type_resolver_core</c>: the search over the type list with the fixed slots taken as matched.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="ops">The operands' descriptors.</param>
        /// <param name="inputCasting">The input rule.</param>
        /// <param name="casting">The output rule and the final check's.</param>
        /// <param name="specified">The fixed type numbers, <see cref="NPY_NOTYPE"/> for the free ones.</param>
        /// <returns>The loop's descriptors, or null when no loop fits.</returns>
        /// <exception cref="ArgumentException">An operand cannot be cast to the found loop at all.</exception>
        private static DType[] TypeTupleTypeResolverCore(string ufunc, DType[] ops, NPY_CASTING inputCasting, NPY_CASTING casting,
            int[] specified)
        {
            bool noCastableOutput = false;
            char errSrc = '-', errDst = '-';
            foreach (var meta in s_legacyTypes)
            {
                var orig = new[] { meta.type_num, meta.type_num, meta.type_num };
                var types = new int[3];
                bool fits = true;
                for (int j = 0; j < 3 && fits; j++)
                {
                    if (specified[j] == NPY_NOTYPE)
                    {
                        types[j] = orig[j];
                        continue;
                    }
                    fits = orig[j] == specified[j];
                    // A fixed slot needs no further check.
                    types[j] = NPY_NOTYPE;
                }
                if (!fits)
                    continue;
                if (!LoopMatches(ops, inputCasting, casting, types, ref noCastableOutput, ref errSrc, ref errDst))
                    continue;
                var result = SetUfuncLoopDataTypes(ops, orig);
                ValidateCasting(ufunc, ops, result, casting);
                return result;
            }
            return null;
        }

        /// <summary>
        ///     <c>ufunc_loop_matches</c>: every input with a type to check casts to it under <paramref name="inputCasting"/>,
        ///     then the loop's output casts to the given out under <paramref name="outputCasting"/> (the first output that does
        ///     not is remembered for the error text).
        /// </summary>
        /// <param name="ops">The operands' descriptors.</param>
        /// <param name="inputCasting">The input rule.</param>
        /// <param name="outputCasting">The output rule.</param>
        /// <param name="types">The loop's type numbers; <see cref="NPY_NOTYPE"/> skips a slot.</param>
        /// <param name="noCastableOutput">Set when an output did not cast.</param>
        /// <param name="errSrc">The loop output's type character, for the error text.</param>
        /// <param name="errDst">The given out's type character, for the error text.</param>
        /// <returns>True when the loop fits.</returns>
        private static bool LoopMatches(DType[] ops, NPY_CASTING inputCasting, NPY_CASTING outputCasting, int[] types,
            ref bool noCastableOutput, ref char errSrc, ref char errDst)
        {
            for (int i = 0; i < 2; i++)
            {
                if (types[i] == NPY_NOTYPE)
                    continue;
                if (types[i] == NPY_OBJECT)
                    return false;
                var tmp = DTypeRegistry.FromTypeNum(types[i]).singleton;
                if (!DTypeCasting.CanCastTypeTo(ops[i], tmp, inputCasting))
                    return false;
            }
            if (types[2] != NPY_NOTYPE && ops[2] is not null)
            {
                var tmp = DTypeRegistry.FromTypeNum(types[2]).singleton;
                if (!DTypeCasting.CanCastTypeTo(tmp, ops[2], outputCasting))
                {
                    if (!noCastableOutput)
                    {
                        noCastableOutput = true;
                        errSrc = tmp.@char;
                        errDst = ops[2].@char;
                    }
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        ///     <c>set_ufunc_loop_data_types</c>: an operand whose type number equals the loop's keeps its own (canonical)
        ///     descriptor; an output may also take the first operand's; anything else is the type's default descriptor.
        /// </summary>
        /// <param name="ops">The operands' descriptors.</param>
        /// <param name="types">The loop's type numbers.</param>
        /// <returns>The loop's descriptors.</returns>
        private static DType[] SetUfuncLoopDataTypes(DType[] ops, int[] types)
        {
            var result = new DType[3];
            for (int i = 0; i < 3; i++)
            {
                if (ops[i] is not null && ops[i].num == types[i])
                    result[i] = Canonical(ops[i]);
                else if (i >= 2 && ops[0] is not null && ops[0].num == types[i])
                    result[i] = Canonical(ops[0]);
                else
                    result[i] = DTypeRegistry.FromTypeNum(types[i]).singleton;
            }
            return result;
        }

        /// <summary>
        ///     <c>PyUFunc_ValidateCasting</c>: each input casts to its loop descriptor and the loop's output casts to the out
        ///     under <paramref name="casting"/>.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="ops">The operands' descriptors.</param>
        /// <param name="descrs">The loop's descriptors.</param>
        /// <param name="casting">The rule.</param>
        /// <exception cref="ArgumentException">NumPy's UFuncTypeError texts (<c>Cannot cast ufunc 'add' input 0 from ... to
        ///     ... with casting rule 'unsafe'</c>, <c>Cannot cast ufunc 'add' output from ... to ... with casting rule
        ///     'unsafe'</c>), in the house type of an input-cast refusal.</exception>
        private static void ValidateCasting(string ufunc, DType[] ops, DType[] descrs, NPY_CASTING casting)
        {
            string rule = DTypeCasting.CastingToString(casting);
            for (int i = 0; i < 2; i++)
                if (!DTypeCasting.CanCastTypeTo(ops[i], descrs[i], casting))
                    throw new ArgumentException(
                        $"Cannot cast ufunc '{ufunc}' input {i} from {ops[i].ToString(true)} to {descrs[i].ToString(true)} with casting rule '{rule}'");
            if (ops[2] is not null && !DTypeCasting.CanCastTypeTo(descrs[2], ops[2], casting))
                throw new ArgumentException(
                    $"Cannot cast ufunc '{ufunc}' output from {descrs[2].ToString(true)} to {ops[2].ToString(true)} with casting rule '{rule}'");
        }

        // ---------------------------------------------------------------------------------------------
        // resolve_descriptors (ufunc_object.c) and the loops' own rules
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>resolve_descriptors</c>: every operand is first made an instance of its loop DType
        ///     (<see cref="DTypePromotion.CastDescrToDType"/>: a string class sizes a number's text, a <c>StringDType</c> out
        ///     into a fixed-width class cannot be sized), then the loop family's own rule picks the descriptors.
        /// </summary>
        /// <param name="loop">The loop entry.</param>
        /// <param name="ops">The operands' descriptors.</param>
        /// <param name="signature">The filled signature (the loop's DTypes).</param>
        /// <returns>The loop's descriptors.</returns>
        /// <exception cref="TypeError"><c>cannot cast dtype ... to ...</c>, or the loop rule's refusal.</exception>
        private static DType[] ResolveDescriptors(Entry loop, DType[] ops, DTypeMeta[] signature)
        {
            var given = new DType[3];
            for (int i = 0; i < 3; i++)
                given[i] = ops[i] is null ? null : DTypePromotion.CastDescrToDType(ops[i], signature[i]);

            switch (loop.Kind)
            {
                case ReduceLoopKind.Numeric:
                    return SimpleLegacyResolveDescriptors(signature, given);

                case ReduceLoopKind.StringAdd:
                {
                    // string_addition_resolve_descriptors: the out's descriptor is ignored; the result is as wide as both
                    // inputs together.
                    long resultSize = (long)given[0].itemsize + given[1].itemsize;
                    if (resultSize > int.MaxValue)
                    {
                        long length = given[0].@char == 'U' ? resultSize / 4 : resultSize;
                        throw new TypeError($"addition result string of length {length} is too large to store inside array.");
                    }
                    var l0 = Canonical(given[0]);
                    var l1 = Canonical(given[1]);
                    var cls = (StringDTypeMeta)l0.Meta;
                    return new[] { l0, l1, cls.Descr((l0.itemsize + l1.itemsize) / cls.UnitSize) };
                }

                case ReduceLoopKind.StringRepeat:
                    // string_multiply_resolve_descriptors.
                    if (given[2] is null)
                        throw new TypeError(
                            "The 'out' kwarg is necessary when using the string multiply ufunc directly. Use numpy.strings.multiply " +
                            "to multiply strings without specifying 'out'.");
                    return new[] { Canonical(given[0]), Canonical(given[1]), Canonical(given[2]) };

                case ReduceLoopKind.VStringAdd:
                {
                    // binary_resolve_descriptors: the inputs' NA objects must agree; an absent out is a new StringDType
                    // with the inputs' NA object and the FIRST input's coerce flag (NumPy reads descr1 twice).
                    var p0 = given[0].VString;
                    var p1 = given[1].VString;
                    if (!VStringParameters.CompatibleNA(p0, p1, out bool hasNA, out object na))
                        throw new TypeError(
                            $"Cannot find a compatible null string value for null strings '{NDPyObject.Repr(p0.na_object)}' and " +
                            $"'{NDPyObject.Repr(p1.na_object)}'");
                    var outDescr = given[2] ?? new DType(DTypeRegistry.VString, new VStringParameters(hasNA, na, p0.coerce));
                    return new[] { given[0], given[1], outDescr };
                }

                case ReduceLoopKind.VStringRepeat:
                {
                    // multiply_resolve_descriptors: an absent out copies the StringDType operand's parameters.
                    var text = signature[0] is VStringDTypeMeta ? given[0] : given[1];
                    var outDescr = given[2] ?? new DType(DTypeRegistry.VString,
                        new VStringParameters(text.VString.HasNAObject, text.VString.na_object, text.VString.coerce));
                    return new[] { given[0], given[1], outDescr };
                }

                default:
                    throw new InvalidOperationException($"Unknown loop kind {loop.Kind}.");
            }
        }

        /// <summary>
        ///     <c>simple_legacy_resolve_descriptors</c> for a binary loop: with a given output whose DType is the first's (a
        ///     possible reduction) the output's canonical descriptor serves the first operand too (and the second when its DType
        ///     is also the output's); otherwise each given descriptor is canonicalized and a missing one is the first's (same
        ///     DType) or the DType's default.
        /// </summary>
        /// <param name="dtypes">The loop's DTypes.</param>
        /// <param name="given">The operands cast to the loop DTypes (null where absent).</param>
        /// <returns>The descriptors.</returns>
        private static DType[] SimpleLegacyResolveDescriptors(DTypeMeta[] dtypes, DType[] given)
        {
            var result = new DType[3];
            if (given[2] is not null && ReferenceEquals(dtypes[0], dtypes[2]))
            {
                result[2] = Canonical(given[2]);
                result[0] = result[2];
                result[1] = ReferenceEquals(dtypes[1], dtypes[2]) ? result[2] : Canonical(given[1]);
                return result;
            }
            for (int i = 0; i < 3; i++)
            {
                if (given[i] is not null)
                    result[i] = Canonical(given[i]);
                else if (i > 0 && ReferenceEquals(dtypes[i], dtypes[0]))
                    result[i] = result[0];
                else
                    result[i] = dtypes[i].default_descr();
            }
            return result;
        }

        // ---------------------------------------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's no-loop refusal over the dispatcher's operation DTypes: <c>ufunc 'add' did not contain a loop with
        ///     signature matching types (A, B) -&gt; C</c> (an unknown slot prints <c>None</c>).
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="opDTypes">The operation DTypes as the dispatcher held them.</param>
        /// <param name="cause">The promotion error the refusal replaces, or null.</param>
        /// <returns>The exception (the house type of NumPy's UFuncTypeError).</returns>
        private static TypeError NoLoop(string ufunc, DTypeMeta[] opDTypes, Exception cause = null)
        {
            var e = NDStringUfuncs.VStringNoLoop(ufunc, new[] { Name(opDTypes[0]), Name(opDTypes[1]) }, Name(opDTypes[2]));
            return cause is null ? e : new TypeError(e.Message, cause);
        }

        /// <summary>A DType's repr for an error text, or <c>None</c>.</summary>
        /// <param name="meta">The DType, or null.</param>
        /// <returns>The text.</returns>
        private static string Name(DTypeMeta meta) => meta?.ToString() ?? "None";

        /// <summary>
        ///     NumPy's <c>_UFuncBinaryResolutionError</c>: <c>ufunc 'add' cannot use operands with types dtype('S6') and
        ///     dtype('&lt;U1')</c>.
        /// </summary>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <param name="a">The first operand's descriptor.</param>
        /// <param name="b">The second operand's descriptor.</param>
        /// <returns>The exception.</returns>
        private static TypeError BinaryResolutionError(string ufunc, DType a, DType b)
            => new TypeError($"ufunc '{ufunc}' cannot use operands with types {a.ToString(true)} and {b.ToString(true)}");

        /// <summary><c>NPY_DT_CALL_ensure_canonical</c>.</summary>
        /// <param name="d">The descriptor.</param>
        /// <returns>Its canonical form.</returns>
        private static DType Canonical(DType d) => d.Meta.ensure_canonical(d);

        /// <summary><c>npy_casting_to_string</c> WITHOUT its quotes (the callers' formats add them, as NumPy's do).</summary>
        /// <param name="casting">The rule.</param>
        /// <returns><c>'safe'</c> and so on, quoted (NumPy's helper returns the quotes, its formats add another pair).</returns>
        private static string CastingToString(NPY_CASTING casting) => $"'{DTypeCasting.CastingToString(casting)}'";

        /// <summary><c>PyTypeNum_ISINTEGER</c>: the signed and unsigned integer type numbers (not bool).</summary>
        /// <param name="t">The type number.</param>
        /// <returns>True for 1..10.</returns>
        private static bool IsIntegerNum(int t) => t >= 1 && t <= 10;

        /// <summary><c>PyTypeNum_ISSTRING</c>: fixed-width bytes or unicode (NOT <c>StringDType</c>).</summary>
        /// <param name="t">The type number.</param>
        /// <returns>True for 18 and 19.</returns>
        private static bool IsStringNum(int t) => t == NPY_STRING || t == NPY_UNICODE;

        /// <summary><c>PyTypeNum_ISDATETIME</c>.</summary>
        /// <param name="t">The type number.</param>
        /// <returns>True for datetime64 and timedelta64.</returns>
        private static bool IsDatetimeNum(int t) => t == NPY_DATETIME || t == NPY_TIMEDELTA;

        /// <summary><c>PyTypeNum_ISFLEXIBLE</c>: bytes, unicode and void.</summary>
        /// <param name="t">The type number.</param>
        /// <returns>True for 18..20.</returns>
        private static bool IsFlexibleNum(int t) => t >= NPY_STRING && t <= NPY_VOID;

        /// <summary>
        ///     The uniform resolver's custom-or-object test: a type number past the legacy range (<c>StringDType</c>,
        ///     NumSharp's Decimal and Char) or object.
        /// </summary>
        /// <param name="d">The descriptor.</param>
        /// <returns>True for a custom or object type.</returns>
        private static bool IsCustomOrObject(DType d) => d.num >= NPY_NTYPES_LEGACY || d.num == NPY_OBJECT;

        /// <summary>
        ///     Whether a DType subclasses NumPy's abstract integer DType (<c>_IntegerAbstractDType</c>): the signed and
        ///     unsigned integer classes, the lane-sharing C-type ones included; not bool, not NumSharp's Char.
        /// </summary>
        /// <param name="meta">The DType.</param>
        /// <returns>True for an integer class.</returns>
        private static bool IsIntegerAbstractSubclass(DTypeMeta meta) => meta._legacy && IsIntegerNum(meta.type_num);

        /// <summary>
        ///     <c>string_multiply_promoter</c>'s <c>is_integer_dtype</c>: an integer class other than int64 and uint64 (which
        ///     have their own <c>StringDType</c> loops).
        /// </summary>
        /// <param name="meta">The DType, or null.</param>
        /// <returns>True when the promoter turns it into int64.</returns>
        private static bool IsPromotedToInt64(DTypeMeta meta)
            => meta is not null && IsIntegerAbstractSubclass(meta)
               && !ReferenceEquals(meta, DTypeRegistry.Int64) && !ReferenceEquals(meta, DTypeRegistry.UInt64);
    }
}
