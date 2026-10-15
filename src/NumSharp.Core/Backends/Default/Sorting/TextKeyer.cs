using System;
using NumSharp.Backends.Kernels;

// =============================================================================
// TextKeyer.cs — the order-preserving 64-bit prefix keys of fixed-width text (S / U), for every text algorithm
// =============================================================================
//
// A key packs the first code units an element holds from a given unit on, big-endian, so that unsigned key order is
// NumPy's order on that part of the strings (STRING_compare / UNICODE_compare: unsigned code units, the zero padding read
// like any unit). Two elements whose keys differ are ordered by their keys; two whose keys are equal agree on the units
// the keys cover and are decided by the units after them (or are equal, when the keys cover the elements to their ends:
// the keys are EXACT).
//
// THE PACKINGS (ILKernelGenerator.GetTextSortKeyKernel)
//   S: 8 bytes. U: 8 units of 8 bits (every unit below 2^8), else 4 of 16 bits, 3 of 21 bits (every code point), 2 of 32
//   bits (any unit a view can hold). The keyer first runs the 8-bit packing of U, which ORs the units it reads; when a unit
//   does not fit 8 bits it re-keys with the narrowest packing that holds every unit it saw. ASCII and Latin-1 text costs
//   one pass. Keys of different calls are comparable only when the calls used one packing: a caller that compares keys of
//   two arrays (np.searchsorted's haystack and needles) chooses the packing from both (ProbeUnits8 + KeysWith).
//
// USERS
//   TextSort (the sorts), TextArgReduce (argmax / argmin), TextSearch (searchsorted). Each call owns one keyer: the
//   kernels it resolves are cached in the keyer, so a call that keys many blocks resolves each kernel once.
// =============================================================================

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    ///     The prefix keys of one text dtype's elements (see the file header): adaptive (<see cref="ComputeKeys"/>) or with a
    ///     packing the caller chose for several arrays (<see cref="ProbeUnits8"/>, <see cref="PackingFor"/>,
    ///     <see cref="KeysWith"/>).
    /// </summary>
    internal sealed unsafe class TextKeyer
    {
        /// <summary>The code-unit size: 1 for <c>S</c>, 4 for <c>U</c>.</summary>
        internal readonly int Unit;

        /// <summary>The element width in code units (the units after the last one a key covers are the tail).</summary>
        internal readonly int Units;

        /// <summary>The kernels this keyer used, by (bits class, units, OR flag); filled on first use.</summary>
        private readonly TextSortKeyFunc[] _kernels = new TextSortKeyFunc[4 * 9 * 2];

        /// <summary>Creates a keyer for elements of <paramref name="units"/> code units of <paramref name="unit"/> bytes.</summary>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="units">The element width in code units (0 for a zero-width dtype, which is never keyed).</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="unit"/> is neither 1 nor 4, or
        ///     <paramref name="units"/> is negative.</exception>
        internal TextKeyer(int unit, int units)
        {
            if (unit != 1 && unit != 4)
                throw new ArgumentOutOfRangeException(nameof(unit), unit, "a text code unit is 1 (S) or 4 (U) bytes");
            if (units < 0)
                throw new ArgumentOutOfRangeException(nameof(units), units, "a width cannot be negative");
            Unit = unit;
            Units = units;
        }

        /// <summary>
        ///     Keys the selected elements from unit <paramref name="fromUnit"/> on, choosing the packing from the units they
        ///     hold: <c>S</c> takes the next 8 bytes; <c>U</c> first takes 8 units of 8 bits (collecting their OR), and when a
        ///     unit does not fit 8 bits re-keys with the widest packing its units allow (16, 21 or 32 bits).
        /// </summary>
        /// <param name="text">Element 0 (the element of index 0, or of slot 0 when <paramref name="index"/> is null).</param>
        /// <param name="stride">The byte distance between consecutive elements (any sign).</param>
        /// <param name="index">The element of each key slot, or null for the identity.</param>
        /// <param name="count">The number of keys.</param>
        /// <param name="fromUnit">The first unit keyed (below <see cref="Units"/>).</param>
        /// <param name="keys">The key slots, <paramref name="count"/> of them.</param>
        /// <returns>The units the keys cover (fewer than the packing holds when the element ends first): the keys are
        ///     exact when <paramref name="fromUnit"/> plus this reaches <see cref="Units"/>.</returns>
        /// <remarks>Keys of two calls are comparable only when both chose one packing; this method chooses per call (the
        ///     single-array users key one line or block at a time and compare keys within it).</remarks>
        internal int ComputeKeys(byte* text, long stride, long* index, long count, int fromUnit, ulong* keys)
        {
            int rem = Units - fromUnit;
            byte* first = text + (long)fromUnit * Unit;
            if (Unit == 1)
            {
                int r = Math.Min(8, rem);
                Kernel(8, r, false)(first, stride, index, count, keys, null);
                return r;
            }

            uint* ors = stackalloc uint[4];
            int r8 = Math.Min(8, rem);
            Kernel(8, r8, true)(first, stride, index, count, keys, ors);
            int bits = PackingFor(ors, rem, out int units);
            if (bits == 8)
                return r8;
            Kernel(bits, units, false)(first, stride, index, count, keys, null);
            return units;
        }

        /// <summary>
        ///     The 8-bit pass of <c>U</c> over the selected elements: writes their 8-bit keys (valid only when every unit fits
        ///     8 bits) and ORs the units read into <paramref name="ors"/> (accumulated: the caller zeroes it before the first
        ///     array, so the OR of several arrays decides one packing for all of them). For <c>S</c>, the 8-byte keys, final.
        /// </summary>
        /// <param name="text">Element 0.</param>
        /// <param name="stride">The byte distance between consecutive elements.</param>
        /// <param name="index">The element of each key slot, or null for the identity.</param>
        /// <param name="count">The number of keys.</param>
        /// <param name="fromUnit">The first unit keyed (below <see cref="Units"/>).</param>
        /// <param name="keys">The key slots.</param>
        /// <param name="ors">Four accumulators (units 0-1, 2, 3, 4-7 of the keyed part), OR-ed into; unused for <c>S</c>.</param>
        /// <returns>The units the 8-bit keys cover.</returns>
        internal int ProbeUnits8(byte* text, long stride, long* index, long count, int fromUnit, ulong* keys, uint* ors)
        {
            int rem = Units - fromUnit;
            byte* first = text + (long)fromUnit * Unit;
            int r8 = Math.Min(8, rem);
            if (Unit == 1)
            {
                Kernel(8, r8, false)(first, stride, index, count, keys, null);
                return r8;
            }
            uint* local = stackalloc uint[4];
            Kernel(8, r8, true)(first, stride, index, count, keys, local);
            for (int g = 0; g < 4; g++)
                ors[g] |= local[g];
            return r8;
        }

        /// <summary>
        ///     The packing that orders units of the given ORs: 8 bits when every unit fits, else 16, 21 or 32 bits — the
        ///     narrowest whose width holds every unit among the ones it would key.
        /// </summary>
        /// <param name="ors">The OR of units 0-1, 2, 3 and 4-7 of the keyed part (from the 8-bit pass).</param>
        /// <param name="rem">The units left to key in an element (bounds the units of the packing).</param>
        /// <param name="units">Receives the units a key of the chosen packing covers.</param>
        /// <returns>The bits per unit: 8, 16, 21 or 32.</returns>
        internal static int PackingFor(uint* ors, int rem, out int units)
        {
            uint or3 = ors[0] | ors[1];   // units 0-2
            uint or4 = or3 | ors[2];      // units 0-3
            if ((or4 | ors[3]) < 0x100u)
            {
                units = Math.Min(8, rem);
                return 8;
            }
            if (or4 < 0x10000u)
            {
                units = Math.Min(4, rem);
                return 16;
            }
            if (or3 < 0x200000u)
            {
                units = Math.Min(3, rem);
                return 21;
            }
            units = Math.Min(2, rem);
            return 32;
        }

        /// <summary>
        ///     Keys the selected elements from unit <paramref name="fromUnit"/> on with a packing the caller chose
        ///     (<see cref="PackingFor"/>; <c>S</c> always 8 bits), so that keys of several arrays compare.
        /// </summary>
        /// <param name="bits">The bits per unit: 8, 16, 21 or 32 (8 for <c>S</c>).</param>
        /// <param name="text">Element 0.</param>
        /// <param name="stride">The byte distance between consecutive elements.</param>
        /// <param name="index">The element of each key slot, or null for the identity.</param>
        /// <param name="count">The number of keys.</param>
        /// <param name="fromUnit">The first unit keyed (below <see cref="Units"/>).</param>
        /// <param name="keys">The key slots.</param>
        /// <returns>The units the keys cover.</returns>
        internal int KeysWith(int bits, byte* text, long stride, long* index, long count, int fromUnit, ulong* keys)
        {
            int rem = Units - fromUnit;
            int units = Math.Min(64 / bits, rem);
            Kernel(bits, units, false)(text + (long)fromUnit * Unit, stride, index, count, keys, null);
            return units;
        }

        /// <summary>The prefix-key kernel of one packing: the emitted one, or the managed twin on a host without dynamic code.</summary>
        /// <param name="bits">The bits per unit (8, 16, 21 or 32).</param>
        /// <param name="units">The units per key.</param>
        /// <param name="collectOr">Whether the kernel collects the unit OR.</param>
        /// <returns>The kernel.</returns>
        internal TextSortKeyFunc Kernel(int bits, int units, bool collectOr)
        {
            int bitsClass = bits switch { 8 => 0, 16 => 1, 21 => 2, _ => 3 };
            int slot = ((bitsClass * 9) + units) * 2 + (collectOr ? 1 : 0);
            var kernel = _kernels[slot];
            if (kernel is null)
            {
                kernel = ILKernelGenerator.GetTextSortKeyKernel(Unit, units, bits, collectOr);
                if (kernel is null)
                {
                    int unit = Unit;
                    kernel = (text, stride, index, count, keys, unitOr) =>
                        ILKernelGenerator.TextSortKeysManaged(unit, units, bits, collectOr, text, stride, index, count, keys, unitOr);
                }
                _kernels[slot] = kernel;
            }
            return kernel;
        }
    }
}
