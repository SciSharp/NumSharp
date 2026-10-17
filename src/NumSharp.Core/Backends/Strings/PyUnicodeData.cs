using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

// =============================================================================
// PyUnicodeData.cs — CPython 3.12's Unicode type records as NumSharp's string code reads them (strings phase 2, P2.A / D9)
// =============================================================================
//
// WHAT THIS IS
//   The runtime half of tools/gen_unicode_tables.py: per code point the flags of CPython's _PyUnicode_TypeRecord that
//   NumPy's string ufuncs and CPython's str methods test (Py_UNICODE_ISALPHA ... _PyUnicode_IsCaseIgnorable), the decimal
//   value (Py_UNICODE_TODECIMAL) and the full case mappings (_PyUnicode_ToUpperFull / ToLowerFull / ToTitleFull), read
//   from the generated two-stage tables (PyUnicodeData.Tables.cs).
//
// WHY PINNED TABLES (strings plan D9)
//   NumPy 2.4.2's oracle runs on CPython 3.12, Unicode 15.0.0; .NET's own tables follow the running runtime (.NET 10 reads
//   Unicode 16.0, which added letters, digits and case pairs). The tables here are what CPython 3.12 answers, so a
//   NumSharp result never depends on the .NET it runs on.
//
// THE LOOKUPS
//   * Record(cp): the record index — Index2[(Index1[cp >> Shift] << Shift) + (cp & mask)], record 0 (U+0000's: no property,
//     maps to itself) for a unit past U+10FFFF, CPython's gettyperecord.
//   * The BMP table (strings plan D14): one 32-bit word per BMP code point, the flags in the low half and the record in
//     the high half, built once from the two-stage tables. It is what a vector gather reads (one vpgatherdd per eight
//     code units) and what the scalar paths read for a BMP code point (one load instead of three).
//   * The pin: ComputeTableHash recomputes the generator's SHA-256 over every code point's data from these lookups, so a
//     table that decodes differently from what the generator recorded fails its test.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     CPython 3.12's Unicode type records (Unicode 15.0.0): the flags, decimal values and full case mappings NumPy's
    ///     string ufuncs and CPython's <c>str</c> methods read, from the generated tables (see the file header).
    /// </summary>
    /// <remarks>
    ///     A code unit past U+10FFFF (a <c>U</c> element can hold any 32-bit value) has CPython's record 0: no property,
    ///     mapped to itself. Lone surrogates (U+D800–U+DFFF) are code points of category <c>Cs</c>: no property, not printable.
    /// </remarks>
    internal static unsafe partial class PyUnicodeData
    {
        /// <summary><c>ALPHA_MASK</c>: <c>Py_UNICODE_ISALPHA</c> (categories Lm, Lt, Lu, Ll, Lo).</summary>
        internal const int AlphaMask = 1 << 0;

        /// <summary><c>DECIMAL_MASK</c>: <c>Py_UNICODE_ISDECIMAL</c> (a decimal digit: its value is <see cref="DecimalValue"/>).</summary>
        internal const int DecimalMask = 1 << 1;

        /// <summary><c>DIGIT_MASK</c>: <c>Py_UNICODE_ISDIGIT</c> (decimal digits and digits such as superscripts).</summary>
        internal const int DigitMask = 1 << 2;

        /// <summary><c>NUMERIC_MASK</c>: <c>Py_UNICODE_ISNUMERIC</c> (any numeric value: fractions, Roman numerals, CJK numerals).</summary>
        internal const int NumericMask = 1 << 3;

        /// <summary><c>SPACE_MASK</c>: <c>Py_UNICODE_ISSPACE</c> (category Zs or bidirectional class WS / B / S).</summary>
        internal const int SpaceMask = 1 << 4;

        /// <summary><c>LOWER_MASK</c>: <c>Py_UNICODE_ISLOWER</c> (the derived Lowercase property).</summary>
        internal const int LowerMask = 1 << 5;

        /// <summary><c>UPPER_MASK</c>: <c>Py_UNICODE_ISUPPER</c> (the derived Uppercase property).</summary>
        internal const int UpperMask = 1 << 6;

        /// <summary><c>TITLE_MASK</c>: <c>Py_UNICODE_ISTITLE</c> (category Lt: <c>ǅ</c>, <c>ᾈ</c>, ...).</summary>
        internal const int TitleMask = 1 << 7;

        /// <summary><c>CASED_MASK</c>: <c>_PyUnicode_IsCased</c> (Lowercase, Uppercase or Lt): <c>str.title</c>'s state and
        ///     the final-sigma rule read it.</summary>
        internal const int CasedMask = 1 << 8;

        /// <summary><c>CASE_IGNORABLE_MASK</c>: <c>_PyUnicode_IsCaseIgnorable</c>, which the final-sigma rule skips.</summary>
        internal const int CaseIgnorableMask = 1 << 9;

        /// <summary><c>PRINTABLE_MASK</c>: <c>Py_UNICODE_ISPRINTABLE</c> (not a C* / Z* category, except the ASCII space),
        ///     the characters <c>repr()</c> leaves unescaped.</summary>
        internal const int PrintableMask = 1 << 10;

        /// <summary><c>EXTENDED_CASE_MASK</c>: the record's case mappings are offsets into the extended table (a mapping of
        ///     more than one code point), not deltas. A layout bit, not a property.</summary>
        internal const int ExtendedCaseMask = 1 << 11;

        /// <summary>The flags of <see cref="IsAlnum"/>: <c>Py_UNICODE_ISALNUM</c> is ALPHA, DECIMAL, DIGIT or NUMERIC.</summary>
        internal const int AlnumMask = AlphaMask | DecimalMask | DigitMask | NumericMask;

        /// <summary>The longest full case mapping of one code point (<c>ﬃ</c> uppercases to <c>FFI</c>).</summary>
        internal const int MaxMapping = 3;

        /// <summary>The first code unit past Unicode: CPython reads record 0 for it and every larger unit.</summary>
        internal const uint UnicodeEnd = 0x110000;

        /// <summary>
        ///     The BMP table: for code point c below 0x10000, <c>flags | record &lt;&lt; 16</c>. A pinned array, so its address
        ///     (<see cref="BmpTable"/>) is stable for the process.
        /// </summary>
        private static readonly uint[] s_bmp = BuildBmp();

        /// <summary>
        ///     The BMP table's first entry: <c>BmpTable[c]</c> is <c>flags | record &lt;&lt; 16</c> for c below 0x10000 (what
        ///     a vector gather reads). Valid for the process's lifetime.
        /// </summary>
        internal static uint* BmpTable
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (uint*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(s_bmp));
        }

        /// <summary>
        ///     The three mapping tables of the records side by side — <c>[upper | lower | title]</c>, <see cref="RecordCount"/>
        ///     entries each — as one pinned array, so a lookup picks its mapping by the index alone (record, plus
        ///     <see cref="CaseDeltaStride"/> for lower, twice that for title): a vector gather chooses swapcase's upper or lower
        ///     mapping per lane in one gather instead of two, and the scalar case map chooses without a data-dependent branch.
        ///     (title's vector form still gathers both of its mappings: choosing by the index would make its gather wait for the
        ///     previous chunk's carry.)
        /// </summary>
        /// <remarks>For a record without <see cref="ExtendedCaseMask"/> an entry is the delta to add to the code point; WITH
        ///     the flag it is an extended-table entry, so a gather reader tests the flag first.</remarks>
        private static readonly int[] s_caseDeltas = BuildCaseDeltas();

        /// <summary>The length of each mapping table in <see cref="CaseDeltas"/> (the number of records): the lower table starts
        ///     at this index, the title table at twice it.</summary>
        internal const int CaseDeltaStride = RecordCount;

        /// <summary>
        ///     The first entry of the side-by-side mapping tables (<c>[upper | lower | title]</c>, see <see cref="s_caseDeltas"/>):
        ///     <c>CaseDeltas[rec]</c> is the uppercase delta, <c>CaseDeltas[CaseDeltaStride + rec]</c> the lowercase one,
        ///     <c>CaseDeltas[2 * CaseDeltaStride + rec]</c> the titlecase one. Valid for the process's lifetime (a pinned array).
        /// </summary>
        internal static int* CaseDeltas
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (int*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(s_caseDeltas));
        }

        /// <summary>Builds <see cref="s_caseDeltas"/> from the three record tables (once, at type initialization).</summary>
        /// <returns>The pinned array.</returns>
        private static int[] BuildCaseDeltas()
        {
            int n = RecordUpper.Length;
            var table = GC.AllocateUninitializedArray<int>(3 * n, pinned: true);
            RecordUpper.CopyTo(table.AsSpan(0, n));
            RecordLower.CopyTo(table.AsSpan(n, n));
            RecordTitle.CopyTo(table.AsSpan(2 * n, n));
            return table;
        }

        /// <summary>Builds the BMP table from the two-stage tables (once, at type initialization; about 0.3 ms).</summary>
        /// <returns>The pinned table.</returns>
        private static uint[] BuildBmp()
        {
            var table = GC.AllocateUninitializedArray<uint>(0x10000, pinned: true);
            for (int c = 0; c < 0x10000; c++)
            {
                int rec = RecordSlow((uint)c);
                table[c] = RecordFlags[rec] | ((uint)rec << 16);
            }
            return table;
        }

        /// <summary>The record of a code point through the two-stage tables (record 0 past U+10FFFF).</summary>
        /// <param name="cp">The code unit (any value).</param>
        /// <returns>The record index.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int RecordSlow(uint cp)
        {
            if (cp >= UnicodeEnd)
                return 0;
            int block = Index1[(int)(cp >> Shift)];
            return Index2[(block << Shift) + (int)(cp & ((1u << Shift) - 1))];
        }

        /// <summary>The record of a code point: the BMP table's high half below 0x10000, the two-stage tables above.</summary>
        /// <param name="cp">The code unit (any value; past U+10FFFF is record 0).</param>
        /// <returns>The record index.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int Record(uint cp) => cp < 0x10000 ? (int)(BmpTable[cp] >> 16) : RecordSlow(cp);

        /// <summary>The flags of a code point (<see cref="AlphaMask"/> ... <see cref="ExtendedCaseMask"/>).</summary>
        /// <param name="cp">The code unit (any value; past U+10FFFF has none).</param>
        /// <returns>The flags.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int Flags(uint cp) => cp < 0x10000 ? (int)(BmpTable[cp] & 0xFFFF) : RecordFlags[RecordSlow(cp)];

        /// <summary><c>Py_UNICODE_TODECIMAL</c>: the decimal value (0–9) of a decimal digit, else -1.</summary>
        /// <param name="cp">The code unit (any value).</param>
        /// <returns>0–9, or -1.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int DecimalValue(uint cp) => RecordDecimal[Record(cp)];

        /// <summary><c>Py_UNICODE_ISSPACE</c>.</summary>
        /// <param name="cp">The code unit.</param>
        /// <returns>True for whitespace.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsSpace(uint cp) => (Flags(cp) & SpaceMask) != 0;

        /// <summary><c>Py_UNICODE_ISALNUM</c>: ALPHA, DECIMAL, DIGIT or NUMERIC.</summary>
        /// <param name="cp">The code unit.</param>
        /// <returns>True for an alphanumeric character.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsAlnum(uint cp) => (Flags(cp) & AlnumMask) != 0;

        /// <summary><c>Py_UNICODE_ISPRINTABLE</c>: the characters <c>repr()</c> leaves unescaped.</summary>
        /// <param name="cp">The code unit (past U+10FFFF is not printable).</param>
        /// <returns>True for a printable character.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsPrintable(uint cp) => (Flags(cp) & PrintableMask) != 0;

        /// <summary><c>_PyUnicode_IsCased</c>.</summary>
        /// <param name="cp">The code unit.</param>
        /// <returns>True for a cased character.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsCased(uint cp) => (Flags(cp) & CasedMask) != 0;

        /// <summary><c>_PyUnicode_IsCaseIgnorable</c>.</summary>
        /// <param name="cp">The code unit.</param>
        /// <returns>True for a case-ignorable character.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsCaseIgnorable(uint cp) => (Flags(cp) & CaseIgnorableMask) != 0;

        /// <summary><c>_PyUnicode_ToUpperFull</c>: the code point's full uppercase mapping.</summary>
        /// <param name="cp">The code unit (past U+10FFFF maps to itself).</param>
        /// <param name="dst">Room for <see cref="MaxMapping"/> code points.</param>
        /// <returns>The mapping's length (1–3).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int ToUpperFull(uint cp, uint* dst) => MapFull(cp, Record(cp), RecordUpper, dst);

        /// <summary><c>_PyUnicode_ToLowerFull</c>: the code point's full lowercase mapping (U+03A3 maps to U+03C3; the
        ///     final-sigma rule is the caller's, CPython's <c>lower_ucs4</c>).</summary>
        /// <param name="cp">The code unit.</param>
        /// <param name="dst">Room for <see cref="MaxMapping"/> code points.</param>
        /// <returns>The mapping's length (1–3).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int ToLowerFull(uint cp, uint* dst) => MapFull(cp, Record(cp), RecordLower, dst);

        /// <summary><c>_PyUnicode_ToTitleFull</c>: the code point's full titlecase mapping.</summary>
        /// <param name="cp">The code unit.</param>
        /// <param name="dst">Room for <see cref="MaxMapping"/> code points.</param>
        /// <returns>The mapping's length (1–3).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int ToTitleFull(uint cp, uint* dst) => MapFull(cp, Record(cp), RecordTitle, dst);

        /// <summary>One full mapping of a record: the delta applied (one code point), or the extended entry copied.</summary>
        /// <param name="cp">The code unit.</param>
        /// <param name="rec">Its record.</param>
        /// <param name="map">The record table of the mapping (upper, lower or title).</param>
        /// <param name="dst">Room for <see cref="MaxMapping"/> code points.</param>
        /// <returns>The mapping's length.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int MapFull(uint cp, int rec, ReadOnlySpan<int> map, uint* dst)
        {
            int v = map[rec];
            if ((RecordFlags[rec] & ExtendedCaseMask) == 0)
            {
                // A delta: wraps like CPython's (Py_UCS4)(ch + delta) for the units past Unicode, whose delta is 0.
                dst[0] = unchecked(cp + (uint)v);
                return 1;
            }
            int offset = v & 0xFFFFFF, length = v >> 24;
            var ext = ExtendedCase;
            for (int i = 0; i < length; i++)
                dst[i] = (uint)ext[offset + i];
            return length;
        }

        /// <summary>
        ///     The SHA-256 the generator recorded (<see cref="TableHash"/>), recomputed from the runtime lookups: per code
        ///     point 0 .. 0x10FFFF the flags (uint16, little-endian, <see cref="ExtendedCaseMask"/> cleared), the decimal
        ///     value (int8) and the upper, lower and title mappings, each as its length (uint8) then its code points (int32
        ///     little-endian).
        /// </summary>
        /// <returns>The lowercase hex digest.</returns>
        /// <remarks>For the pin test; walks 1,114,112 code points (about 0.1 s).</remarks>
        internal static string ComputeTableHash()
        {
            using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            var buffer = new byte[1 << 16];
            int fill = 0;
            uint* mapped = stackalloc uint[MaxMapping];
            for (uint cp = 0; cp < UnicodeEnd; cp++)
            {
                if (fill > buffer.Length - 64)
                {
                    sha.AppendData(buffer, 0, fill);
                    fill = 0;
                }
                ushort flags = (ushort)(Flags(cp) & ~ExtendedCaseMask);
                buffer[fill++] = (byte)flags;
                buffer[fill++] = (byte)(flags >> 8);
                buffer[fill++] = (byte)(sbyte)DecimalValue(cp);
                for (int m = 0; m < 3; m++)
                {
                    int n = m == 0 ? ToUpperFull(cp, mapped) : m == 1 ? ToLowerFull(cp, mapped) : ToTitleFull(cp, mapped);
                    buffer[fill++] = (byte)n;
                    for (int i = 0; i < n; i++)
                    {
                        uint v = mapped[i];
                        buffer[fill++] = (byte)v;
                        buffer[fill++] = (byte)(v >> 8);
                        buffer[fill++] = (byte)(v >> 16);
                        buffer[fill++] = (byte)(v >> 24);
                    }
                }
            }
            sha.AppendData(buffer, 0, fill);
            return Convert.ToHexString(sha.GetHashAndReset()).ToLowerInvariant();
        }
    }
}
