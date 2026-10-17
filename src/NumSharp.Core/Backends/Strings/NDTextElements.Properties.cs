using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

// =============================================================================
// NDTextElements.Properties.cs — the bodies of NumPy's string predicates over S / U (strings phase 2, P2.6; plan D14)
// =============================================================================
//
// NUMPY'S LOOPS (string_ufuncs.cpp string_unary_loop + string_buffer.h)
//   isalpha / isdigit / isspace / isalnum / isdecimal / isnumeric: len = num_codepoints() (one past the last nonzero unit:
//   trailing NULs never count, an embedded NUL is a unit like any other); False for len 0; else every unit of [0, len)
//   has the property. islower: False for len 0; False at the first upper or titlecase unit; True when some unit is lower.
//   isupper mirrors it. istitle: an upper / titlecase unit must not follow a cased unit, a lowercase unit must follow one,
//   and some unit must be cased. The S loop (ENCODING::ASCII) tests NumPyOS_ascii_* (a byte at or past 0x80 has no
//   property; whitespace is space, \t \n \v \f \r) and has no isdecimal / isnumeric loop; the U loop tests CPython's
//   Py_UNICODE_IS* (whitespace adds U+001C..U+001F and the non-ASCII spaces; a unit past U+10FFFF has CPython's record 0,
//   no property).
//
// HOW AN ELEMENT IS COMPUTED HERE
//   Lane masks (bit i of a ulong for unit i, NDTextElements.Lanes.cs) for elements of up to 64 units: per 32-byte chunk the
//   unit classes the predicate needs (two masks: the class, and the "bad" class of islower / isupper / istitle), the
//   nonzero units (the length), then a few bit operations — no branch on the data:
//     every-unit predicates   len > 0 and (class & low(len)) == low(len)
//     islower / isupper       (bad & low(len)) == 0 and (class & low(len)) != 0
//     istitle                 cased = U | L, prev = cased << 1 (the unit before is cased), bad = (U & prev) | (L & ~prev):
//                             (bad & low(len)) == 0 and (cased & low(len)) != 0   (U = upper or title, L = lower)
//   S and the ASCII text of U compute the classes with range compares. U text with a unit at or past 0x80 reads every
//   unit's flags word from PyUnicodeData's BMP table with one vpgatherdd per eight units (a unit past U+FFFF from the
//   two-stage tables, scalar). THE STICKY ESCALATION (plan D14): a call starts on the compare path; the first element
//   with a non-ASCII unit sets the call's state word (the kernel's auxdata) and that element and every later one take the
//   gather path, which is right for ASCII units too — no per-element branch that mispredicts on mixed data.
//   Wider elements stream 32-byte chunks with the same masks and a one-bit carry (istitle's "previous unit cased").
//
// PORTABILITY (plan D21)
//   The mask and stream paths run where AVX2 is supported (a JIT constant); elsewhere every element runs NumPy's unit
//   loops over PyUnicodeData (scalar), with the same results. The unit tests compare the paths on every width class.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static unsafe partial class NDTextElements
    {
        /// <summary>The predicate <c>isalpha</c> (every unit alphabetic).</summary>
        internal const int IsAlphaKind = 0;

        /// <summary>The predicate <c>isdigit</c> (every unit a digit).</summary>
        internal const int IsDigitKind = 1;

        /// <summary>The predicate <c>isspace</c> (every unit whitespace).</summary>
        internal const int IsSpaceKind = 2;

        /// <summary>The predicate <c>isalnum</c> (every unit alphabetic or numeric).</summary>
        internal const int IsAlnumKind = 3;

        /// <summary>The predicate <c>islower</c> (no upper / titlecase unit, some lowercase unit).</summary>
        internal const int IsLowerKind = 4;

        /// <summary>The predicate <c>isupper</c> (no lower / titlecase unit, some uppercase unit).</summary>
        internal const int IsUpperKind = 5;

        /// <summary>The predicate <c>istitle</c> (title-cased words, some cased unit).</summary>
        internal const int IsTitleKind = 6;

        /// <summary>The predicate <c>isdecimal</c> (every unit a decimal digit; no <c>S</c> loop).</summary>
        internal const int IsDecimalKind = 7;

        /// <summary>The predicate <c>isnumeric</c> (every unit numeric; no <c>S</c> loop).</summary>
        internal const int IsNumericKind = 8;

        /// <summary>NumPy's ufunc name of a predicate kind.</summary>
        /// <param name="kind">The kind (<see cref="IsAlphaKind"/> ... <see cref="IsNumericKind"/>).</param>
        /// <returns>The ufunc name.</returns>
        /// <exception cref="ArgumentOutOfRangeException">An unknown kind.</exception>
        internal static string PredicateName(int kind) => kind switch
        {
            IsAlphaKind => "isalpha",
            IsDigitKind => "isdigit",
            IsSpaceKind => "isspace",
            IsAlnumKind => "isalnum",
            IsLowerKind => "islower",
            IsUpperKind => "isupper",
            IsTitleKind => "istitle",
            IsDecimalKind => "isdecimal",
            IsNumericKind => "isnumeric",
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Not a predicate kind."),
        };

        /// <summary>Whether a kind tests EVERY unit for one class (the six predicates other than the case ones).</summary>
        /// <param name="kind">The kind.</param>
        /// <returns>True for isalpha / isdigit / isspace / isalnum / isdecimal / isnumeric.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsEveryUnitKind(int kind) => kind is not (IsLowerKind or IsUpperKind or IsTitleKind);

        /// <summary>
        ///     The flags a kind's first mask reads in a <see cref="PyUnicodeData"/> word: the class of an every-unit kind, the
        ///     lowercase of <c>islower</c>, the uppercase of <c>isupper</c>, upper or title of <c>istitle</c>.
        /// </summary>
        /// <param name="kind">The kind.</param>
        /// <returns>The flag mask.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint FirstFlags(int kind) => kind switch
        {
            IsAlphaKind => PyUnicodeData.AlphaMask,
            IsDigitKind => PyUnicodeData.DigitMask,
            IsSpaceKind => PyUnicodeData.SpaceMask,
            IsAlnumKind => PyUnicodeData.AlnumMask,
            IsLowerKind => PyUnicodeData.LowerMask,
            IsUpperKind => PyUnicodeData.UpperMask,
            IsTitleKind => PyUnicodeData.UpperMask | PyUnicodeData.TitleMask,
            IsDecimalKind => PyUnicodeData.DecimalMask,
            _ => PyUnicodeData.NumericMask,
        };

        /// <summary>
        ///     The flags a kind's second mask reads: the units that make <c>islower</c> False (upper or title), <c>isupper</c>
        ///     False (lower or title), and the lowercase units of <c>istitle</c>; none for the every-unit kinds.
        /// </summary>
        /// <param name="kind">The kind.</param>
        /// <returns>The flag mask.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint SecondFlags(int kind) => kind switch
        {
            IsLowerKind => PyUnicodeData.UpperMask | PyUnicodeData.TitleMask,
            IsUpperKind => PyUnicodeData.LowerMask | PyUnicodeData.TitleMask,
            IsTitleKind => PyUnicodeData.LowerMask,
            _ => 0u,
        };

        // =============================================================================================================
        // The element body
        // =============================================================================================================

        /// <summary>
        ///     The auxdata of a predicate kernel over <c>S</c> / <c>U</c>: the call's escalation word (plan D14) and the BMP
        ///     flags table. One block for the whole call, so a later NDIter chunk starts where the earlier one left off.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct KindState
        {
            /// <summary>0 while every <c>U</c> element of the call so far was ASCII (the compare path), 1 once one was not
            ///     (every later element takes the gather path). Unused for <c>S</c>.</summary>
            internal long Escalated;

            /// <summary>
            ///     <see cref="PyUnicodeData.BmpTable"/>, read once by the driver: a kernel that read the static itself would
            ///     re-test the class's initialization and reload the array at every gather (a <c>DynamicMethod</c> is
            ///     compiled without knowing that the class is initialized).
            /// </summary>
            internal uint* Flags;
        }

        /// <summary>A <see cref="KindState"/> for a call: not escalated, the BMP table read once.</summary>
        /// <returns>The state.</returns>
        internal static KindState NewKindState() => new KindState { Escalated = 0, Flags = PyUnicodeData.BmpTable };

        /// <summary>
        ///     One element of NumPy's <c>isalpha</c> ... <c>isnumeric</c> loop over <c>S</c> (ASCII rules) or <c>U</c> (CPython's
        ///     Unicode rules): the lane-mask form for up to 64 units on an AVX2 host, else the stream / scalar forms.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c>, <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="p">The element's first byte.</param>
        /// <param name="width">The element width in bytes (a JIT constant in the IL shell, so every width test folds).</param>
        /// <param name="kind">The predicate (<see cref="IsAlphaKind"/> ...; a constant in the IL shell).</param>
        /// <param name="state">The call's state (the kernel's auxdata): the escalation word, updated, and the flags
        ///     table.</param>
        /// <returns>The predicate's value.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static bool IsKind<TUnit>(byte* p, int width, int kind, KindState* state) where TUnit : unmanaged
        {
            if (MasksFit<TUnit>(width))
                return IsKindLanes<TUnit>(p, width, kind, state);
            return IsKindLong<TUnit>(p, width, kind, state);
        }

        /// <summary>
        ///     <see cref="IsKind{TUnit}"/> for an element of up to 64 units on an AVX2 host: the masks of every chunk, then the
        ///     predicate's bit test. A <c>U</c> element is first computed with the ASCII compares (unless the call has
        ///     escalated); a unit at or past 0x80 escalates the call and recomputes the element through the gather. An
        ///     element with a unit past U+FFFF runs NumPy's unit loop (<see cref="IsKindScalarCold{TUnit}"/>).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="p">The element.</param>
        /// <param name="width">Its width in bytes (<see cref="MasksFit{TUnit}"/> holds).</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="state">The call's state.</param>
        /// <returns>The predicate's value.</returns>
        /// <remarks>
        ///     The rare element leaves the vector path as a whole, through a call that returns a bool: a vector result merged
        ///     from a rare call and the common gather made the JIT keep every chunk's flags in memory and lay the gather out
        ///     of line (22 ns per 32-unit element measured, against 16 for the gather alone).
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsKindLanes<TUnit>(byte* p, int width, int kind, KindState* state) where TUnit : unmanaged
        {
            int step;
            bool result;
            if (typeof(TUnit) == typeof(byte))
                return KindElement<TUnit>(p, width, kind, PathByteAscii, null, out _);
            uint* flags = state->Flags;
            if (state->Escalated == 0)
            {
                result = KindElement<TUnit>(p, width, kind, PathUnicodeAscii, flags, out step);
                if (step != StepEscalate)
                    return result;
                // The first non-ASCII element of the call: escalate (the rest of the call reads flags words) and recompute
                // this element the same way.
                state->Escalated = 1;
            }
            result = KindElement<TUnit>(p, width, kind, PathGather, flags, out step);
            return step == StepAstral ? IsKindScalarCold<TUnit>(p, width, kind) : result;
        }

        /// <summary>A chunk step's verdict (<see cref="KindStep{TUnit}"/>): go on with the next chunk.</summary>
        private const int StepNext = 0;

        /// <summary>A chunk step's verdict: a unit of this chunk already makes the element's predicate False.</summary>
        private const int StepFalse = 1;

        /// <summary>A chunk step's verdict: a <c>U</c> chunk on <see cref="PathUnicodeAscii"/> holds a unit at or past
        ///     0x80, so its ASCII classes mean nothing and the element must be recomputed through the gather.</summary>
        private const int StepEscalate = 2;

        /// <summary>A chunk step's verdict: a <c>U</c> chunk on <see cref="PathGather"/> holds a unit past U+FFFF, which the
        ///     BMP table does not cover: the element is computed by NumPy's unit loop.</summary>
        private const int StepAstral = 3;

        /// <summary>
        ///     The predicate of an element of up to 64 units on one chunk path: the chunks in increasing lane order, each
        ///     OR-ed into the element's masks (a last chunk overlapping the one before ends at the element's end; the overlap
        ///     repeats the same bits), then the predicate's bit test. An element of more than one chunk stops at the first
        ///     chunk that decides it False — NumPy's loops stop at the first failing unit, and a wide <c>U</c> element costs
        ///     a gather per chunk (see <see cref="KindStep{TUnit}"/>). Written out per chunk position, so a constant width
        ///     compiles to straight-line code.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="p">The element.</param>
        /// <param name="width">Its width in bytes (1 to 64 units).</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="path">The chunk path (<see cref="PathByteAscii"/> ...).</param>
        /// <param name="flags">The BMP flags table (read on <see cref="PathGather"/> only).</param>
        /// <param name="step">Receives the last chunk step's verdict: <see cref="StepEscalate"/> or
        ///     <see cref="StepAstral"/> make the returned value meaningless.</param>
        /// <returns>The predicate's value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool KindElement<TUnit>(byte* p, int width, int kind, int path, uint* flags, out int step) where TUnit : unmanaged
        {
            ulong m1 = 0, m2 = 0, nonzero = 0;
            if (width <= 32)
            {
                // One chunk: the final bit test decides; an early test would only add a branch.
                step = KindStep<TUnit>(LoadPadded32(p, width), kind, path, flags, 0, early: false, ref m1, ref m2, ref nonzero);
            }
            else
            {
                int shiftPerByte = sizeof(TUnit) == 1 ? 0 : 2;
                step = KindStep<TUnit>(Vector256.Load(p), kind, path, flags, 0, early: true, ref m1, ref m2, ref nonzero);
                if (step == StepNext && width >= 64)
                    step = KindStep<TUnit>(Vector256.Load(p + 32), kind, path, flags, 32 >> shiftPerByte, false, ref m1, ref m2, ref nonzero);
                if (step == StepNext && width >= 96)
                    step = KindStep<TUnit>(Vector256.Load(p + 64), kind, path, flags, 64 >> shiftPerByte, false, ref m1, ref m2, ref nonzero);
                if (step == StepNext && width >= 128)
                    step = KindStep<TUnit>(Vector256.Load(p + 96), kind, path, flags, 96 >> shiftPerByte, false, ref m1, ref m2, ref nonzero);
                if (step == StepNext && width >= 160)
                    step = KindStep<TUnit>(Vector256.Load(p + 128), kind, path, flags, 128 >> shiftPerByte, false, ref m1, ref m2, ref nonzero);
                if (step == StepNext && width >= 192)
                    step = KindStep<TUnit>(Vector256.Load(p + 160), kind, path, flags, 160 >> shiftPerByte, false, ref m1, ref m2, ref nonzero);
                if (step == StepNext && width >= 224)
                    step = KindStep<TUnit>(Vector256.Load(p + 192), kind, path, flags, 192 >> shiftPerByte, false, ref m1, ref m2, ref nonzero);
                if (step == StepNext && width >= 256)
                    step = KindStep<TUnit>(Vector256.Load(p + 224), kind, path, flags, 224 >> shiftPerByte, false, ref m1, ref m2, ref nonzero);
                if (step == StepNext && (width & 31) != 0)
                    step = KindStep<TUnit>(Vector256.Load(p + width - 32), kind, path, flags, (width - 32) >> shiftPerByte, false,
                        ref m1, ref m2, ref nonzero);
            }
            if (step != StepNext)
                return false;
            return EvaluateKind(kind, m1, m2, LowLanes(64 - BitOperations.LeadingZeroCount(nonzero)));
        }

        /// <summary>
        ///     One 32-byte chunk of <see cref="KindElement{TUnit}"/>: its class bits OR-ed into the element's masks at lane
        ///     <paramref name="lane"/>, and, under <paramref name="early"/>, whether the chunk already makes the predicate
        ///     False.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="v">The chunk.</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="path">The chunk path.</param>
        /// <param name="flags">The BMP flags table (<see cref="PathGather"/>).</param>
        /// <param name="lane">The chunk's first unit within the element.</param>
        /// <param name="early">Test the chunk's verdict (a constant: false for a one-chunk element).</param>
        /// <param name="m1">The first mask.</param>
        /// <param name="m2">The second mask.</param>
        /// <param name="nonzero">The nonzero units.</param>
        /// <returns><see cref="StepNext"/>, <see cref="StepFalse"/>, <see cref="StepEscalate"/> or
        ///     <see cref="StepAstral"/>.</returns>
        /// <remarks>
        ///     <para>What decides False without the element's length (a unit past it is zero and has no class):</para>
        ///     <list type="bullet">
        ///         <item>an every-unit predicate: a NONZERO unit without the class (every nonzero unit is below the length,
        ///         one past the last nonzero unit);</item>
        ///         <item>islower / isupper: a unit of the bad class (a cased unit, so nonzero);</item>
        ///         <item>istitle: a bad unit by the rule of the file header, computed on the accumulated masks, so the unit
        ///         before the chunk's first one is the previous chunk's last.</item>
        ///     </list>
        ///     <para>The test of a later chunk never runs once one of these holds, and the final bit test of
        ///     <see cref="EvaluateKind"/> stays exact without them (an embedded NUL fails an every-unit predicate there).
        ///     On <see cref="PathUnicodeAscii"/> the escalation is tested first: an ASCII class of a non-ASCII unit is
        ///     not its class. On <see cref="PathGather"/> a unit past U+FFFF is tested before the gather (one
        ///     <c>vptest</c>): the table covers the BMP only.</para>
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int KindStep<TUnit>(Vector256<byte> v, int kind, int path, uint* flags, int lane, bool early, ref ulong m1,
                                           ref ulong m2, ref ulong nonzero) where TUnit : unmanaged
        {
            if (typeof(TUnit) == typeof(uint) && path == PathGather && HasAstral(v.AsUInt32()))
                return StepAstral;
            ChunkClasses<TUnit>(v, kind, path, flags, out uint c1, out uint c2, out uint nz, out uint hi);
            if (path == PathUnicodeAscii && hi != 0)
                return StepEscalate;
            m1 |= (ulong)c1 << lane;
            m2 |= (ulong)c2 << lane;
            nonzero |= (ulong)nz << lane;
            if (!early)
                return StepNext;
            if (IsEveryUnitKind(kind))
                return (nz & ~c1) != 0 ? StepFalse : StepNext;
            if (kind != IsTitleKind)
                return c2 != 0 ? StepFalse : StepNext;
            ulong cased = m1 | m2;
            ulong prev = cased << 1;
            ulong chunk = (sizeof(TUnit) == 1 ? 0xFFFF_FFFFUL : 0xFFUL) << lane;
            return (((m1 & prev) | (m2 & ~prev)) & chunk) != 0 ? StepFalse : StepNext;
        }

        /// <summary>
        ///     The predicate from the masks of an element's first <c>len</c> units (<paramref name="valid"/> = their lanes):
        ///     NumPy's every-unit, islower / isupper and istitle rules as bit tests (see the file header).
        /// </summary>
        /// <param name="kind">The predicate.</param>
        /// <param name="m1">The first mask (the class; lower / upper / upper-or-title).</param>
        /// <param name="m2">The second mask (the bad class; istitle's lowercase).</param>
        /// <param name="valid">The lanes of the units below the element's length.</param>
        /// <returns>The predicate's value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool EvaluateKind(int kind, ulong m1, ulong m2, ulong valid)
        {
            if (kind == IsTitleKind)
            {
                ulong cased = m1 | m2;
                ulong prev = cased << 1;
                ulong bad = (m1 & prev) | (m2 & ~prev);
                return ((bad & valid) == 0) & ((cased & valid) != 0);
            }
            if (kind is IsLowerKind or IsUpperKind)
                return ((m2 & valid) == 0) & ((m1 & valid) != 0);
            return (valid != 0) & ((m1 & valid) == valid);
        }

        /// <summary>The chunk path of <see cref="ChunkClasses{TUnit}"/>: <c>S</c>'s ASCII compares.</summary>
        private const int PathByteAscii = 0;

        /// <summary>The chunk path of <see cref="ChunkClasses{TUnit}"/>: compares with Unicode's rules for ASCII units (<c>U</c>,
        ///     and <c>StringDType</c>'s ASCII text).</summary>
        private const int PathUnicodeAscii = 1;

        /// <summary>The chunk path of <see cref="ChunkClasses{TUnit}"/>: <c>U</c> units' flags words through the gather.</summary>
        private const int PathGather = 2;

        /// <summary>
        ///     The class bits of one 32-byte chunk: <c>S</c> — 32 byte lanes; <c>U</c> — 8 unit lanes. The ASCII paths are
        ///     range compares (NumPyOS_ascii_* for <c>S</c>; Python's rules for ASCII code points otherwise: whitespace also
        ///     U+001C..U+001F, the digits are decimal and numeric too); the gather path reads each unit's
        ///     <see cref="PyUnicodeData"/> flags word.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="v">The chunk.</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="path">The chunk path.</param>
        /// <param name="flags">The BMP flags table (<see cref="PathGather"/>: every unit of the chunk is below
        ///     U+10000, which the caller tested).</param>
        /// <param name="c1">Receives the first class's lanes.</param>
        /// <param name="c2">Receives the second class's lanes.</param>
        /// <param name="nz">Receives the nonzero lanes.</param>
        /// <param name="hi">Receives the lanes at or past 0x80 (<see cref="PathUnicodeAscii"/>; 0 otherwise).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ChunkClasses<TUnit>(Vector256<byte> v, int kind, int path, uint* flags, out uint c1, out uint c2,
                                                out uint nz, out uint hi) where TUnit : unmanaged
        {
            if (typeof(TUnit) == typeof(byte))
            {
                ByteClasses(v, kind, unicodeSpace: path != PathByteAscii, out c1, out c2);
                nz = ~Vector256.Equals(v, Vector256<byte>.Zero).ExtractMostSignificantBits();
                hi = path == PathUnicodeAscii ? v.ExtractMostSignificantBits() : 0u;
                return;
            }
            var u = v.AsUInt32();
            nz = ~Vector256.Equals(u, Vector256<uint>.Zero).ExtractMostSignificantBits() & 0xFFu;
            if (path == PathGather)
            {
                var words = GatherFlags(u, flags);
                c1 = ~Vector256.Equals(words & Vector256.Create(FirstFlags(kind)), Vector256<uint>.Zero).ExtractMostSignificantBits() & 0xFFu;
                c2 = SecondFlags(kind) == 0
                    ? 0u
                    : ~Vector256.Equals(words & Vector256.Create(SecondFlags(kind)), Vector256<uint>.Zero).ExtractMostSignificantBits() & 0xFFu;
                hi = 0;
                return;
            }
            UnitAsciiClasses(u, kind, out c1, out c2);
            hi = ~Vector256.Equals(u & Vector256.Create(0xFFFFFF80u), Vector256<uint>.Zero).ExtractMostSignificantBits() & 0xFFu;
        }

        /// <summary>
        ///     The ASCII class bits of 32 byte lanes: letters, digits, whitespace (space and <c>\t</c>–<c>\r</c>, plus
        ///     U+001C..U+001F under <paramref name="unicodeSpace"/>), lower / upper. A byte at or past 0x80 has no class.
        /// </summary>
        /// <param name="v">The chunk.</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="unicodeSpace">True for Python's whitespace rule (a <c>StringDType</c> text), false for NumPy's
        ///     <c>S</c> rule.</param>
        /// <param name="c1">Receives the first class's lanes.</param>
        /// <param name="c2">Receives the second class's lanes.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ByteClasses(Vector256<byte> v, int kind, bool unicodeSpace, out uint c1, out uint c2)
        {
            var lower = Vector256.LessThanOrEqual(v - Vector256.Create((byte)'a'), Vector256.Create((byte)25));
            var upper = Vector256.LessThanOrEqual(v - Vector256.Create((byte)'A'), Vector256.Create((byte)25));
            c2 = 0;
            switch (kind)
            {
                case IsAlphaKind:
                    c1 = (lower | upper).ExtractMostSignificantBits();
                    return;
                case IsDigitKind:
                case IsDecimalKind:
                case IsNumericKind:
                    c1 = Vector256.LessThanOrEqual(v - Vector256.Create((byte)'0'), Vector256.Create((byte)9)).ExtractMostSignificantBits();
                    return;
                case IsSpaceKind:
                {
                    var ws = Vector256.Equals(v, Vector256.Create((byte)0x20))
                             | Vector256.LessThanOrEqual(v - Vector256.Create((byte)0x09), Vector256.Create((byte)4));
                    if (unicodeSpace)
                        ws |= Vector256.LessThanOrEqual(v - Vector256.Create((byte)0x1C), Vector256.Create((byte)3));
                    c1 = ws.ExtractMostSignificantBits();
                    return;
                }
                case IsAlnumKind:
                    c1 = (lower | upper | Vector256.LessThanOrEqual(v - Vector256.Create((byte)'0'), Vector256.Create((byte)9)))
                        .ExtractMostSignificantBits();
                    return;
                case IsLowerKind:
                    c1 = lower.ExtractMostSignificantBits();
                    c2 = upper.ExtractMostSignificantBits();
                    return;
                default: // isupper; istitle (its first class is upper, its second lower: no ASCII titlecase letter)
                    c1 = upper.ExtractMostSignificantBits();
                    c2 = lower.ExtractMostSignificantBits();
                    return;
            }
        }

        /// <summary>The ASCII class bits of 8 <c>U</c> lanes (Python's rules for code points below 0x80).</summary>
        /// <param name="u">The units (the class bits of a unit at or past 0x80 are meaningless: the caller escalates).</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="c1">Receives the first class's lanes.</param>
        /// <param name="c2">Receives the second class's lanes.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void UnitAsciiClasses(Vector256<uint> u, int kind, out uint c1, out uint c2)
        {
            var lower = Vector256.LessThanOrEqual(u - Vector256.Create((uint)'a'), Vector256.Create(25u));
            var upper = Vector256.LessThanOrEqual(u - Vector256.Create((uint)'A'), Vector256.Create(25u));
            c2 = 0;
            switch (kind)
            {
                case IsAlphaKind:
                    c1 = (lower | upper).ExtractMostSignificantBits();
                    return;
                case IsDigitKind:
                case IsDecimalKind:
                case IsNumericKind:
                    c1 = Vector256.LessThanOrEqual(u - Vector256.Create((uint)'0'), Vector256.Create(9u)).ExtractMostSignificantBits();
                    return;
                case IsSpaceKind:
                    c1 = (Vector256.Equals(u, Vector256.Create(0x20u))
                          | Vector256.LessThanOrEqual(u - Vector256.Create(0x09u), Vector256.Create(4u))
                          | Vector256.LessThanOrEqual(u - Vector256.Create(0x1Cu), Vector256.Create(3u))).ExtractMostSignificantBits();
                    return;
                case IsAlnumKind:
                    c1 = (lower | upper | Vector256.LessThanOrEqual(u - Vector256.Create((uint)'0'), Vector256.Create(9u)))
                        .ExtractMostSignificantBits();
                    return;
                case IsLowerKind:
                    c1 = lower.ExtractMostSignificantBits();
                    c2 = upper.ExtractMostSignificantBits();
                    return;
                default:
                    c1 = upper.ExtractMostSignificantBits();
                    c2 = lower.ExtractMostSignificantBits();
                    return;
            }
        }

        /// <summary>The <see cref="PyUnicodeData"/> flags words of 8 code units below U+10000: one <c>vpgatherdd</c> from
        ///     the BMP table.</summary>
        /// <param name="u">The units (<c>U</c> elements' units, or code points decoded from UTF-8); the caller tested that
        ///     every one is below U+10000 (a larger index would read past the table).</param>
        /// <param name="flags">The BMP table (<see cref="PyUnicodeData.BmpTable"/>).</param>
        /// <returns>Each lane's word: the flags in the low half, the record index in the high half, which a caller masks away
        ///     (every flag tested here is below bit 16).</returns>
        /// <remarks>Measured on 32-unit elements: an in-place clamp of the astral lanes (compare, andnot, mask extract) cost
        ///     2.5 ns of 16, so the callers test a chunk with one <c>vptest</c> and send an element with an astral unit to the
        ///     unit loop as a whole.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> GatherFlags(Vector256<uint> u, uint* flags)
            => Avx2.GatherVector256((int*)flags, u.AsInt32(), 4).AsUInt32();

        /// <summary>Whether a vector of code units has one past U+FFFF (one <c>vptest</c>).</summary>
        /// <param name="u">The units.</param>
        /// <returns>True when a lane has a bit at or past bit 16.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool HasAstral(Vector256<uint> u) => !Avx.TestZ(u, Vector256.Create(0xFFFF_0000u));

        /// <summary>
        ///     NumPy's unit loop for an <c>S</c> / <c>U</c> element that the vector path sent away (a <c>U</c> unit past
        ///     U+FFFF): <see cref="IsKindScalar{TUnit}"/>, kept out of the kernels.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="p">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="kind">The predicate.</param>
        /// <returns>The predicate's value.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static bool IsKindScalarCold<TUnit>(byte* p, int width, int kind) where TUnit : unmanaged
            => IsKindScalar<TUnit>(p, width, kind);

        // =============================================================================================================
        // Wide elements, hosts without AVX2
        // =============================================================================================================

        /// <summary>
        ///     <see cref="IsKind{TUnit}"/> for an element past 64 units, or on a host without AVX2: the chunk stream (AVX2), else
        ///     NumPy's unit loops (<see cref="IsKindScalar{TUnit}"/>).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="p">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="state">The call's state (the escalation word, updated; the flags table).</param>
        /// <returns>The predicate's value.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static bool IsKindLong<TUnit>(byte* p, int width, int kind, KindState* state) where TUnit : unmanaged
        {
            if (Avx2.IsSupported && width >= 32)
            {
                long len = Units<TUnit>(p, width);
                if (len == 0)
                    return false;
                int path = typeof(TUnit) == typeof(byte) ? PathByteAscii : state->Escalated != 0 ? PathGather : PathUnicodeAscii;
                return IsKindStream<TUnit>(p, width / sizeof(TUnit), len, kind, path, state);
            }
            return IsKindScalar<TUnit>(p, width, kind);
        }

        /// <summary>
        ///     The chunk stream of a long text: 32-byte chunks over its first <paramref name="len"/> units (the last one
        ///     overlapping the one before so it ends at the text's end, its masks shifted down past the overlap), each folded
        ///     into the predicate's running state — the every-unit AND, the bad / cased ORs, and istitle's carry (whether the
        ///     unit before the chunk is cased) — stopping at the first chunk that decides False, as NumPy's loops stop at the
        ///     first failing unit. On <see cref="PathUnicodeAscii"/> a <c>U</c> chunk with a non-ASCII unit escalates the call
        ///     and takes the gather path from that chunk on (the gather is right for ASCII units too); a unit past U+FFFF
        ///     sends the text to NumPy's unit loop.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="p">The text's first unit.</param>
        /// <param name="units">The units readable from <paramref name="p"/> (at least one chunk: 32 bytes).</param>
        /// <param name="len">The units the predicate reads (one past the last nonzero unit; 1 to <paramref name="units"/>).</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="path">The starting chunk path.</param>
        /// <param name="state">The call's state (null for text already known to be ASCII, read only through the byte
        ///     compares).</param>
        /// <returns>The predicate's value.</returns>
        private static bool IsKindStream<TUnit>(byte* p, long units, long len, int kind, int path, KindState* state) where TUnit : unmanaged
        {
            int lanes = 32 / sizeof(TUnit);
            uint* flags = state != null ? state->Flags : null;
            bool any = false;
            uint carry = 0;
            for (long s = 0; s < len; s += lanes)
            {
                // The chunk of units [s, s + lanes), or the last chunk of the text shifted to start at unit s.
                long first = s + lanes <= units ? s : units - lanes;
                int shift = (int)(s - first);
                var v = Vector256.Load(p + first * sizeof(TUnit));
                if (typeof(TUnit) == typeof(uint) && path == PathGather && HasAstral(v.AsUInt32()))
                    return IsKindScalar<TUnit>(p, (int)(units * sizeof(TUnit)), kind);
                ChunkClasses<TUnit>(v, kind, path, flags, out uint c1, out uint c2, out _, out uint hi);
                if (typeof(TUnit) == typeof(uint) && hi >> shift != 0)
                {
                    // A non-ASCII unit in this chunk's new lanes: escalate and read this chunk's flags words.
                    if (state != null)
                        state->Escalated = 1;
                    path = PathGather;
                    if (HasAstral(v.AsUInt32()))
                        return IsKindScalar<TUnit>(p, (int)(units * sizeof(TUnit)), kind);
                    ChunkClasses<TUnit>(v, kind, path, flags, out c1, out c2, out _, out _);
                }
                c1 >>= shift;
                c2 >>= shift;
                uint valid = (uint)LowLanes(Math.Min(lanes - shift, len - s));
                if (kind == IsTitleKind)
                {
                    uint cased = c1 | c2;
                    uint prev = (cased << 1) | carry;
                    if ((((c1 & prev) | (c2 & ~prev)) & valid) != 0)
                        return false;
                    any |= (cased & valid) != 0;
                    carry = (cased >> (lanes - shift - 1)) & 1;
                }
                else if (kind is IsLowerKind or IsUpperKind)
                {
                    if ((c2 & valid) != 0)
                        return false;
                    any |= (c1 & valid) != 0;
                }
                else if ((c1 & valid) != valid)
                {
                    return false;
                }
            }
            return IsEveryUnitKind(kind) || any;
        }

        // =============================================================================================================
        // UTF-8 text (StringDType)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>Buffer&lt;ENCODING::UTF8&gt;</c> predicate of a <c>StringDType</c> text: its code points without the
        ///     trailing NUL bytes (<c>num_codepoints_for_utf8_bytes</c> ignores them; an embedded NUL counts), False when there
        ///     are none, Python's Unicode rules per code point.
        /// </summary>
        /// <param name="s">The text (valid UTF-8, as every <c>StringDType</c> text is).</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="flags">The BMP flags table (<see cref="PyUnicodeData.BmpTable"/>, read once per call by the
        ///     driver).</param>
        /// <returns>The predicate's value.</returns>
        /// <remarks>
        ///     <para>The forms on an AVX2 host: up to 16 bytes, one vector padded with zeros (<see cref="IsKindUtf8Window"/>);
        ///     up to 32 bytes of ASCII, the halves <c>[0, 16)</c> and <c>[size - 16, size)</c> through the byte class masks,
        ///     the second half's bits shifted to its place (a byte of the overlap has the same class in both); longer ASCII
        ///     text, the byte chunk stream; text with a byte at or past 0x80, the UTF-8 windows
        ///     (<see cref="IsKindUtf8Windows"/>). Without AVX2, or for text with a character past U+FFFF, NumPy's loops
        ///     (<see cref="IsKindUtf8Scalar"/>).</para>
        ///     <para>Never reads past <c>[s, s + size)</c>: an arena or heap text can end where mapped memory ends. That is why
        ///     the 17–32 byte form loads two halves instead of one padded 32-byte vector, whose shift by a run-time byte count
        ///     has no instruction (<c>psrldq</c> takes an immediate) and became a call to a software fallback: 18.7 ns of a
        ///     StringDType text of 16 to 32 bytes went there.</para>
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsKindUtf8(byte* s, long size, int kind, uint* flags)
        {
            while (size > 0 && s[size - 1] == 0)
                size--;
            if (size == 0)
                return false;
            if (!Avx2.IsSupported)
                return IsKindUtf8Scalar(s, size, kind);
            if (size <= 16)
                return IsKindUtf8Window(LoadPadded16(s, (int)size), (int)size, kind, flags);
            if (size <= 32)
            {
                var lo = Vector128.Load(s);
                var hi = Vector128.Load(s + size - 16);
                if ((lo | hi).ExtractMostSignificantBits() == 0)
                {
                    ByteClasses(Vector256.Create(lo, hi), kind, unicodeSpace: true, out uint b1, out uint b2);
                    int up = (int)size - 16;
                    ulong c1 = (b1 & 0xFFFFu) | ((ulong)(b1 >> 16) << up);
                    ulong c2 = (b2 & 0xFFFFu) | ((ulong)(b2 >> 16) << up);
                    return EvaluateKind(kind, c1, c2, LowLanes(size));
                }
                return IsKindUtf8Windows(s, size, kind, flags);
            }
            return IsAscii(s, size)
                ? IsKindStream<byte>(s, size, size, kind, PathUnicodeAscii, null)
                : IsKindUtf8Windows(s, size, kind, flags);
        }

        /// <summary>
        ///     The predicate of a text of at most 16 bytes held in one vector (a <c>StringDType</c> short cell, or a short
        ///     text read into one): the byte class masks when every byte is below 0x80, else one UTF-8 window.
        /// </summary>
        /// <param name="v">The text's bytes, zero past them.</param>
        /// <param name="end">One past its last nonzero byte (0 to 16): NumPy's code points end there, an embedded NUL
        ///     before it counts.</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="flags">The BMP flags table.</param>
        /// <returns>The predicate's value (False for <paramref name="end"/> 0).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsKindUtf8Window(Vector128<byte> v, int end, int kind, uint* flags)
        {
            ulong valid = LowLanes(end);
            if (v.ExtractMostSignificantBits() == 0)
            {
                ByteClasses(v.ToVector256(), kind, unicodeSpace: true, out uint c1, out uint c2);
                return EvaluateKind(kind, c1, c2, valid);
            }
            // The bytes a character starting at position 15 reads past the window are zero (the text ends in it).
            if (!Utf8WindowClasses(v, Sse2.ShiftRightLogical128BitLane(v, 8), kind, flags, out uint w1, out uint w2))
                return IsKindUtf8VectorCold(v, end, kind);
            // One window: the verdict from bit tests, with no branch on the text (a verdict that splits the data, as
            // islower's on mixed-case text does, mispredicted on half the elements: 30 ns against isalpha's 15).
            uint cont = Vector128.LessThan(v.AsSByte(), Vector128.Create((sbyte)-64)).ExtractMostSignificantBits();
            uint starts = ~cont & (uint)valid;
            if (IsEveryUnitKind(kind))
                return (starts != 0) & ((w1 & starts) == starts);
            if (kind != IsTitleKind)
                return ((w2 & starts) == 0) & ((w1 & starts) != 0);
            uint cased = (w1 | w2) & starts;
            uint prev = SpreadOverContinuations(cased, cont, 0) << 1;
            return ((((w1 & prev) | (w2 & ~prev)) & starts) == 0) & (cased != 0);
        }

        /// <summary>
        ///     istitle's "this byte's character is cased" over a UTF-8 window: the cased bit of each character, set at its
        ///     first byte, copied onto its continuation bytes, with <paramref name="carry"/> standing in for byte -1 (the
        ///     previous window's last character).
        /// </summary>
        /// <param name="cased">The cased characters, at their first bytes.</param>
        /// <param name="cont">The continuation bytes.</param>
        /// <param name="carry">1 when the character holding the byte before the window is cased.</param>
        /// <returns>The cased bit at every byte of each character; shifted left by one (OR the carry) it tells, at a
        ///     character's first byte, whether the previous character is cased.</returns>
        /// <remarks>
        ///     Two copies reach every continuation byte a window can hold: a window reaches this point only when it has no
        ///     character past U+FFFF (<see cref="Utf8WindowClasses"/> sends those to the scalar loop), so a character is at
        ///     most three bytes long. A character that began in the previous window reaches its continuation bytes here
        ///     through <paramref name="carry"/>, which each copy ORs in at byte 0.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint SpreadOverContinuations(uint cased, uint cont, uint carry)
        {
            uint spread = cased;
            spread |= ((spread << 1) | carry) & cont;
            spread |= ((spread << 1) | carry) & cont;
            return spread;
        }

        /// <summary>
        ///     NumPy's UTF-8 loop over a text held in one vector that has a character past U+FFFF (the windows read the BMP
        ///     table only): the vector stored and decoded one character at a time. Rare, so kept out of the kernels.
        /// </summary>
        /// <param name="v">The text's bytes, zero past them.</param>
        /// <param name="end">One past its last nonzero byte (1 to 16).</param>
        /// <param name="kind">The predicate.</param>
        /// <returns>The predicate's value.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static bool IsKindUtf8VectorCold(Vector128<byte> v, int end, int kind)
        {
            byte* text = stackalloc byte[16];
            v.Store(text);
            return IsKindUtf8Scalar(text, end, kind);
        }

        /// <summary>
        ///     The predicate of a text of more than 16 bytes with a byte at or past 0x80: 16-byte UTF-8 windows in order,
        ///     read in place while a window's lookahead stays inside the text, then the last 8 to 23 bytes copied into a
        ///     zeroed buffer and read as one or two windows. Stops at the first window that decides False; a character past
        ///     U+FFFF sends the text to NumPy's loop.
        /// </summary>
        /// <param name="s">The text (valid UTF-8).</param>
        /// <param name="size">Its byte size without the trailing NULs (more than 16).</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="flags">The BMP flags table.</param>
        /// <returns>The predicate's value.</returns>
        /// <remarks>Not inlined: the kernels inline the short forms of <see cref="IsKindUtf8"/>, and a long non-ASCII text
        ///     amortizes the call.</remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        [SkipLocalsInit]
        private static bool IsKindUtf8Windows(byte* s, long size, int kind, uint* flags)
        {
            bool any = false;
            uint carry = 0;
            long o = 0;
            int step = StepNext;
            // A window reads bytes [o, o + 16) and its lookahead [o + 8, o + 24) (a character starting at o + 15 ends by
            // o + 18): in place while that stays inside the text.
            for (; step == StepNext && o + 24 <= size; o += 16)
                step = Utf8WindowStep(Vector128.Load(s + o), Vector128.Load(s + o + 8), kind, flags, 0xFFFFu, ref any, ref carry);
            if (step == StepNext)
            {
                // The rest, 8 to 23 bytes (size > 16, and the loop stopped at the first o with o + 24 > size): copied with
                // two overlapping moves that read only inside the text, zeros after it up to the second window's lookahead.
                long rest = size - o;
                byte* t = stackalloc byte[48];
                Vector256<byte>.Zero.Store(t);
                Vector128<byte>.Zero.Store(t + 32);
                if (rest >= 16)
                {
                    Vector128.Load(s + o).Store(t);
                    Vector128.Load(s + o + rest - 16).Store(t + rest - 16);
                }
                else
                {
                    Unsafe.WriteUnaligned(t, Unsafe.ReadUnaligned<ulong>(s + o));
                    Unsafe.WriteUnaligned(t + rest - 8, Unsafe.ReadUnaligned<ulong>(s + o + rest - 8));
                }
                step = Utf8WindowStep(Vector128.Load(t), Vector128.Load(t + 8), kind, flags, (uint)LowLanes(Math.Min(rest, 16)),
                    ref any, ref carry);
                if (step == StepNext && rest > 16)
                    step = Utf8WindowStep(Vector128.Load(t + 16), Vector128.Load(t + 24), kind, flags, (uint)LowLanes(rest - 16),
                        ref any, ref carry);
            }
            if (step == StepAstral)
                return IsKindUtf8Scalar(s, size, kind);
            return step == StepNext && (IsEveryUnitKind(kind) || any);
        }

        /// <summary>
        ///     One 16-byte UTF-8 window of a predicate: the class bits of the characters starting at its byte positions
        ///     (<see cref="Utf8WindowClasses"/>), tested at the positions that start a character, with the state carried from
        ///     the window before.
        /// </summary>
        /// <param name="v">The window's bytes, zero past the text.</param>
        /// <param name="ahead">Bytes 8 to 23 from the window's start, zero past the text.</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="flags">The BMP flags table.</param>
        /// <param name="valid">The window's positions inside the text (a run from bit 0).</param>
        /// <param name="any">Set when a character has the class (islower / isupper) or is cased (istitle): their True needs
        ///     one; every-unit predicates leave it alone (a text that is not empty starts a character at byte 0).</param>
        /// <param name="carry">istitle: 1 when the character holding the previous window's last byte is cased; updated for
        ///     the next window.</param>
        /// <returns><see cref="StepNext"/>, <see cref="StepFalse"/> (the window makes the predicate False), or
        ///     <see cref="StepAstral"/> (a character past U+FFFF: the window's classes are not computed).</returns>
        /// <remarks>
        ///     istitle needs "the previous character is cased" at each character's first byte. The cased bit of each character
        ///     is spread over its continuation bytes (a character has at most three), so the bit of the byte before a first
        ///     byte is the previous character's; <paramref name="carry"/> stands in for byte -1.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Utf8WindowStep(Vector128<byte> v, Vector128<byte> ahead, int kind, uint* flags, uint valid, ref bool any,
                                          ref uint carry)
        {
            if (!Utf8WindowClasses(v, ahead, kind, flags, out uint c1, out uint c2))
                return StepAstral;
            // A continuation byte is 0x80..0xBF: below -64 as a signed byte. The other positions inside the text start a
            // character (an embedded NUL included: code point 0, no class).
            uint cont = Vector128.LessThan(v.AsSByte(), Vector128.Create((sbyte)-64)).ExtractMostSignificantBits();
            uint starts = ~cont & valid;
            if (IsEveryUnitKind(kind))
                return (c1 & starts) != starts ? StepFalse : StepNext;
            if (kind != IsTitleKind)
            {
                any |= (c1 & starts) != 0;
                return (c2 & starts) != 0 ? StepFalse : StepNext;
            }
            uint cased = (c1 | c2) & starts;
            uint spread = SpreadOverContinuations(cased, cont, carry);
            uint prev = (spread << 1) | carry;
            carry = (spread >> 15) & 1;
            any |= cased != 0;
            return (((c1 & prev) | (c2 & ~prev)) & starts) != 0 ? StepFalse : StepNext;
        }

        /// <summary>
        ///     The class bits of the characters that would start at the 16 byte positions of a UTF-8 window: each position's
        ///     bytes <c>b0 b1 b2 b3</c> shuffled into a 32-bit lane (<c>vpshufb</c>, two registers of 8 positions), decoded
        ///     without a branch (<see cref="Utf8CodePoints"/>), and their <see cref="PyUnicodeData"/> flags gathered. A
        ///     continuation byte's position gets the bits of a meaningless code point below 0x40; the caller tests only the
        ///     positions that start a character.
        /// </summary>
        /// <param name="v">The window's bytes, zero past the text.</param>
        /// <param name="ahead">Bytes 8 to 23 from the window's start, zero past the text.</param>
        /// <param name="kind">The predicate.</param>
        /// <param name="flags">The BMP flags table.</param>
        /// <param name="c1">Receives the first class's positions.</param>
        /// <param name="c2">Receives the second class's positions.</param>
        /// <returns>False when a position decodes past U+FFFF (a 4-byte character: the table covers the BMP only; the
        ///     outputs are then meaningless).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Utf8WindowClasses(Vector128<byte> v, Vector128<byte> ahead, int kind, uint* flags, out uint c1,
                                              out uint c2)
        {
            // vpshufb works within each 128-bit half. Positions 0-7: both halves hold the window (positions 4-7 read up to
            // byte 10). Positions 8-15: the low half holds the window (positions 8-11 read up to byte 14), the high half
            // bytes 8-23 (positions 12-15 read its bytes 4-10, the window's 12-18).
            var w0 = Avx2.Shuffle(Vector256.Create(v, v), Vector256.Create(
                (byte)3, 2, 1, 0, 4, 3, 2, 1, 5, 4, 3, 2, 6, 5, 4, 3,
                7, 6, 5, 4, 8, 7, 6, 5, 9, 8, 7, 6, 10, 9, 8, 7)).AsUInt32();
            var w1 = Avx2.Shuffle(Vector256.Create(v, ahead), Vector256.Create(
                (byte)11, 10, 9, 8, 12, 11, 10, 9, 13, 12, 11, 10, 14, 13, 12, 11,
                7, 6, 5, 4, 8, 7, 6, 5, 9, 8, 7, 6, 10, 9, 8, 7)).AsUInt32();
            var cp0 = Utf8CodePoints(w0);
            var cp1 = Utf8CodePoints(w1);
            if (HasAstral(cp0 | cp1))
            {
                c1 = c2 = 0;
                return false;
            }
            var f0 = GatherFlags(cp0, flags);
            var f1 = GatherFlags(cp1, flags);
            c1 = FlagPositions(f0, f1, FirstFlags(kind));
            c2 = SecondFlags(kind) == 0 ? 0u : FlagPositions(f0, f1, SecondFlags(kind));
            return true;
        }

        /// <summary>
        ///     <c>utf8_char_to_ucs4_code</c> in 8 lanes at once: lane i holds the bytes of a character that would start
        ///     there, its lead byte in bits 24-31; the result is its code point (a continuation byte as the lead reads as
        ///     its low 7 bits, below 0x40).
        /// </summary>
        /// <param name="w">The lanes: bytes <c>b0 b1 b2 b3</c> from bit 31 down.</param>
        /// <returns>The code points.</returns>
        /// <remarks>A lead byte 110xxxxx / 1110xxxx / 11110xxx carries 5 / 4 / 3 payload bits (0x7F shifted right by 2 / 3
        ///     / 4) and each following byte 6; the steps are blended by the lead's length, so no lane branches.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> Utf8CodePoints(Vector256<uint> w)
        {
            var lead = Vector256.ShiftRightLogical(w, 24);
            var two = Vector256.GreaterThan(lead.AsInt32(), Vector256.Create(0xBF)).AsUInt32();
            var three = Vector256.GreaterThan(lead.AsInt32(), Vector256.Create(0xDF)).AsUInt32();
            var four = Vector256.GreaterThan(lead.AsInt32(), Vector256.Create(0xEF)).AsUInt32();
            var shift = (two & Vector256.Create(2u)) + (three & Vector256.Create(1u)) + (four & Vector256.Create(1u));
            var tail = Vector256.Create(0x3Fu);
            var cp = lead & Avx2.ShiftRightLogicalVariable(Vector256.Create(0x7Fu), shift);
            cp = Vector256.ConditionalSelect(two, Vector256.ShiftLeft(cp, 6) | (Vector256.ShiftRightLogical(w, 16) & tail), cp);
            cp = Vector256.ConditionalSelect(three, Vector256.ShiftLeft(cp, 6) | (Vector256.ShiftRightLogical(w, 8) & tail), cp);
            cp = Vector256.ConditionalSelect(four, Vector256.ShiftLeft(cp, 6) | (w & tail), cp);
            return cp;
        }

        /// <summary>The positions of 16 flags words (two registers of 8) that have any of <paramref name="flags"/>.</summary>
        /// <param name="f0">The words of positions 0-7.</param>
        /// <param name="f1">The words of positions 8-15.</param>
        /// <param name="flags">The <see cref="PyUnicodeData"/> flags tested (all below bit 16).</param>
        /// <returns>Bit i set when position i has one of them.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint FlagPositions(Vector256<uint> f0, Vector256<uint> f1, uint flags)
        {
            var m = Vector256.Create(flags);
            uint lo = ~Vector256.Equals(f0 & m, Vector256<uint>.Zero).ExtractMostSignificantBits() & 0xFFu;
            uint hi = ~Vector256.Equals(f1 & m, Vector256<uint>.Zero).ExtractMostSignificantBits() & 0xFFu;
            return lo | (hi << 8);
        }

        /// <summary>Whether every byte of a text is below 0x80 (32-byte chunks, the last one overlapping).</summary>
        /// <param name="s">The text.</param>
        /// <param name="size">Its byte size (at least 32).</param>
        /// <returns>True for ASCII text.</returns>
        private static bool IsAscii(byte* s, long size)
        {
            var acc = Vector256<byte>.Zero;
            long i = 0;
            for (; i + 32 <= size; i += 32)
                acc |= Vector256.Load(s + i);
            if (i < size)
                acc |= Vector256.Load(s + size - 32);
            return acc.ExtractMostSignificantBits() == 0;
        }

        /// <summary>
        ///     NumPy's UTF-8 loops literally: the code points decoded one at a time (<c>utf8_char_to_ucs4_code</c>), the rules
        ///     of <see cref="IsKindScalar{TUnit}"/> over <see cref="PyUnicodeData"/>.
        /// </summary>
        /// <param name="s">The text (valid UTF-8).</param>
        /// <param name="size">Its byte size without the trailing NULs (at least 1).</param>
        /// <param name="kind">The predicate.</param>
        /// <returns>The predicate's value.</returns>
        internal static bool IsKindUtf8Scalar(byte* s, long size, int kind)
        {
            byte* end = s + size;
            if (IsEveryUnitKind(kind))
            {
                uint need = FirstFlags(kind);
                for (byte* c = s; c < end;)
                {
                    uint cp = DecodeUtf8(c, out int n);
                    if ((PyUnicodeData.Flags(cp) & need) == 0)
                        return false;
                    c += n;
                }
                return true;
            }
            if (kind == IsTitleKind)
            {
                bool cased = false, previousIsCased = false;
                for (byte* c = s; c < end;)
                {
                    int f = PyUnicodeData.Flags(DecodeUtf8(c, out int n));
                    c += n;
                    if ((f & (PyUnicodeData.UpperMask | PyUnicodeData.TitleMask)) != 0)
                    {
                        if (previousIsCased)
                            return false;
                        previousIsCased = true;
                        cased = true;
                    }
                    else if ((f & PyUnicodeData.LowerMask) != 0)
                    {
                        if (!previousIsCased)
                            return false;
                        cased = true;
                    }
                    else
                    {
                        previousIsCased = false;
                    }
                }
                return cased;
            }
            bool seen = false;
            uint first = FirstFlags(kind), second = SecondFlags(kind);
            for (byte* c = s; c < end;)
            {
                int f = PyUnicodeData.Flags(DecodeUtf8(c, out int n));
                c += n;
                if ((f & second) != 0)
                    return false;
                if ((f & first) != 0)
                    seen = true;
            }
            return seen;
        }

        /// <summary>NumPy's <c>utf8_char_to_ucs4_code</c>: the code point at a lead byte and its byte count.</summary>
        /// <param name="c">The lead byte (valid UTF-8).</param>
        /// <param name="bytes">Receives the character's byte count (1–4).</param>
        /// <returns>The code point.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static uint DecodeUtf8(byte* c, out int bytes)
        {
            uint b0 = c[0];
            if (b0 < 0x80)
            {
                bytes = 1;
                return b0;
            }
            if (b0 < 0xE0)
            {
                bytes = 2;
                return ((b0 & 0x1F) << 6) | (c[1] & 0x3Fu);
            }
            if (b0 < 0xF0)
            {
                bytes = 3;
                return ((b0 & 0x0F) << 12) | ((c[1] & 0x3Fu) << 6) | (c[2] & 0x3Fu);
            }
            bytes = 4;
            return ((b0 & 0x07) << 18) | ((c[1] & 0x3Fu) << 12) | ((c[2] & 0x3Fu) << 6) | (c[3] & 0x3Fu);
        }

        /// <summary>
        ///     NumPy's loops literally (<c>Buffer::unary_loop</c>, <c>islower</c>, <c>isupper</c>, <c>istitle</c>), one unit at a
        ///     time: the portable twin of the vector forms, and their reference in the tests.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c> (ASCII rules), <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <param name="p">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="kind">The predicate.</param>
        /// <returns>The predicate's value.</returns>
        internal static bool IsKindScalar<TUnit>(byte* p, int width, int kind) where TUnit : unmanaged
        {
            long len = UnitsScalar<TUnit>(p, width);
            if (len == 0)
                return false;
            if (IsEveryUnitKind(kind))
            {
                for (long i = 0; i < len; i++)
                    if (!UnitHas<TUnit>(Unit<TUnit>(p, i), FirstFlags(kind)))
                        return false;
                return true;
            }
            if (kind == IsTitleKind)
            {
                bool cased = false, previousIsCased = false;
                for (long i = 0; i < len; i++)
                {
                    uint u = Unit<TUnit>(p, i);
                    if (UnitHas<TUnit>(u, PyUnicodeData.UpperMask | PyUnicodeData.TitleMask))
                    {
                        if (previousIsCased)
                            return false;
                        previousIsCased = true;
                        cased = true;
                    }
                    else if (UnitHas<TUnit>(u, PyUnicodeData.LowerMask))
                    {
                        if (!previousIsCased)
                            return false;
                        cased = true;
                    }
                    else
                    {
                        previousIsCased = false;
                    }
                }
                return cased;
            }
            // islower / isupper: a unit of the opposite case (or titlecase) ends it False; some unit of the case.
            bool seen = false;
            for (long i = 0; i < len; i++)
            {
                uint u = Unit<TUnit>(p, i);
                if (UnitHas<TUnit>(u, SecondFlags(kind)))
                    return false;
                if (UnitHas<TUnit>(u, FirstFlags(kind)))
                    seen = true;
            }
            return seen;
        }

        /// <summary>Unit <paramref name="i"/> of an element.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="p">The element.</param>
        /// <param name="i">The unit index.</param>
        /// <returns>The unit's value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Unit<TUnit>(byte* p, long i) where TUnit : unmanaged
            => typeof(TUnit) == typeof(byte) ? p[i] : ((uint*)p)[i];

        /// <summary>
        ///     Whether a unit has any of the flags: <c>S</c> through NumPy's ASCII tests (NumPyOS_ascii_*: a byte at or past
        ///     0x80 has none; whitespace without U+001C..U+001F; the digits are the only digits), <c>U</c> through
        ///     <see cref="PyUnicodeData"/>.
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="u">The unit.</param>
        /// <param name="flags">The <see cref="PyUnicodeData"/> flags tested.</param>
        /// <returns>True when the unit has one of them.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool UnitHas<TUnit>(uint u, uint flags) where TUnit : unmanaged
        {
            if (typeof(TUnit) == typeof(uint))
                return (PyUnicodeData.Flags(u) & flags) != 0;
            uint f = 0;
            if (u - 'a' <= 25)
                f = PyUnicodeData.AlphaMask | PyUnicodeData.LowerMask;
            else if (u - 'A' <= 25)
                f = PyUnicodeData.AlphaMask | PyUnicodeData.UpperMask;
            else if (u - '0' <= 9)
                f = PyUnicodeData.DigitMask | PyUnicodeData.DecimalMask | PyUnicodeData.NumericMask;
            else if (u == ' ' || u - 0x09 <= 4)
                f = PyUnicodeData.SpaceMask;
            return (f & flags) != 0;
        }

        /// <summary>NumPy's <c>num_codepoints</c> of an <c>S</c> / <c>U</c> element, one unit at a time from the end.</summary>
        /// <typeparam name="TUnit"><see cref="byte"/> or <see cref="uint"/>.</typeparam>
        /// <param name="p">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <returns>One past the last nonzero unit.</returns>
        private static long UnitsScalar<TUnit>(byte* p, int width) where TUnit : unmanaged
        {
            long n = width / sizeof(TUnit);
            while (n > 0 && Unit<TUnit>(p, n - 1) == 0)
                n--;
            return n;
        }
    }
}
