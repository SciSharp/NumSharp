using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

// =============================================================================
// Dragon4.BigInt.cs — NumPy's BigInt arithmetic and its Dragon4 digit loop over it (numpy/_core/src/multiarray/
// dragon4.c, NumPy 2.4.2)
// =============================================================================
//
// WHY. Dragon4 runs exact rational arithmetic: the value and its rounding margins over a common scale, one quotient digit
// per step. NumPy keeps those numbers in a mutable fixed-capacity BigInt and changes them in place, so a digit costs a few
// passes over a few dozen machine words and no allocation. System.Numerics.BigInteger is immutable: every multiply, add
// and division allocates a new number, which made the general path 1.1-3x SLOWER than NumPy's C (a 60-digit exact cut
// 4.2 us against 1.5, 1e300 to 20 places 145 us against 49). This file is NumPy's BigInt, in place, on 64-bit blocks.
//
// WHAT DIFFERS FROM NumPy's BigInt, and why the digits cannot change:
//   * 64-bit blocks (NumPy: 32-bit with 64-bit intermediates), with Math.BigMul for the 128-bit products. Half the blocks,
//     so half the passes.
//   * The division's normalization targets bit 59 of the divisor's top block (NumPy: bit 27 of a 32-bit block): the
//     same rule, B - 5 for a block of B bits, so the top block lies in [8, (2^64 - 1) / 10] and a numerator times 10
//     never outgrows the denominator. The shift scales the value, the scale and both margins by one power of two, so
//     every comparison and every quotient is the same; the estimate dividend.top / (divisor.top + 1) undershoots the
//     true quotient by at most one, as in NumPy, and the one correction step finishes it.
//   * Powers of ten come from a table of exact powers (built once), not NumPy's 10^(8*2^k) squares: one multiplication.
//   * Once the lower margin reaches the scale in unique mode, low and high are true for every later digit (the remainder
//     stays below the scale and the margins only grow), so the loop stops carrying the margins and keeps NumPy's exit
//     test with low = high = true: the digits of NumPy's loop, without its margin arithmetic (min_digits past the
//     shortest digits made NumPy's margins grow by a factor of ten per digit).
//   * Where the margins no longer matter (exact mode, and unique mode once they saturate) and at least 19 more digits
//     come before the cutoff, 19 digits come out of ONE division: with the value V < 10*scale, the next 19 digits are
//     the 19-digit decimal form of floor(V * 10^18 / scale) and the remainder is V * 10^18 mod scale — exactly what 19
//     of NumPy's quotient-digit steps produce. The division is Knuth's algorithm D for one 64-bit quotient word over the
//     scale shifted to its top bit (two passes over the blocks per 19 digits instead of 38). In exact mode NumPy stops
//     at the first digit after which the remainder is zero: a zero remainder at the end of a chunk is that case, and the
//     digit is the chunk's last non-zero one (a remainder that turns zero after a digit makes that digit non-zero).
// Every other statement is NumPy's. The System.Numerics loop (GenerateDigitsReference in Dragon4.Port.cs) stays as an
// independent reference: Dragon4PortTests holds this loop to it, and both to the shortest-digits fast path.
//
// =============================================================================

namespace NumSharp.Backends.Printing
{
    internal static partial class Dragon4
    {
        /// <summary>
        ///     Test hook: when true, every request on THIS thread that reaches the general digit loop runs the
        ///     System.Numerics reference (<see cref="GenerateDigitsReference"/>) instead of the block arithmetic of
        ///     <see cref="GenerateDigitsBlocks"/>. Production code never sets it; <c>Dragon4PortTests</c> uses it to hold
        ///     the block loop to an independent exact computation.
        /// </summary>
        [ThreadStatic]
        internal static bool UseReferenceDigits;

        /// <summary>The five big integers of one thread's digit loop (NumPy's <c>Dragon4_Scratch.bigints</c>).</summary>
        [ThreadStatic]
        private static BigIntScratch t_bigints;

        /// <summary>
        ///     NumPy's Dragon4 loop over <see cref="BigInt"/> (see the file header): the digits of the value into
        ///     <paramref name="buffer"/> at <paramref name="start"/>, the same digits <see cref="GenerateDigitsReference"/>
        ///     computes, statement for statement in NumPy's order.
        /// </summary>
        /// <param name="value">The decomposed value (non-zero mantissa).</param>
        /// <param name="digitMode">Unique or exact digits.</param>
        /// <param name="cutoffMode">What the cutoffs count.</param>
        /// <param name="cutoffMax">The cutoff (-1 for none).</param>
        /// <param name="cutoffMin">The minimum (-1 for none).</param>
        /// <param name="buffer">The output scratch.</param>
        /// <param name="start">Where the digits start.</param>
        /// <param name="bufferSize">The most digits that may be written.</param>
        /// <param name="printExponent">Receives the first digit's exponent.</param>
        /// <returns>The digit count.</returns>
        private static int GenerateDigitsBlocks(in Dragon4Value value, DigitMode digitMode, CutoffMode cutoffMode,
            int cutoffMax, int cutoffMin, char[] buffer, int start, int bufferSize, out int printExponent)
        {
            BigIntScratch s = t_bigints ??= new BigIntScratch();
            BigInt scaledValue = s.Value, scale = s.Scale, scaledMarginLow = s.MarginLow, temp = s.Temp;
            int cur = start;
            int exponent = value.Exponent;
            ulong mantissa = value.Mantissa;
            bool isEven = (mantissa & 1) == 0;
            bool unequal = value.HasUnequalMargins;
            // NumPy's scaledMarginHigh pointer: the low margin itself with equal margins, else its own number (twice the
            // low one — recomputed after every change of the low one, as NumPy does).
            BigInt scaledMarginHigh = unequal ? s.MarginHigh : scaledMarginLow;

            // value = scaledValue / scale; the margins are half the distance to the neighbouring floats on the same scale.
            if (unequal)
            {
                if (exponent > 0)
                {
                    scaledValue.SetUInt64(mantissa);
                    scaledValue.ShiftLeft(exponent + 2);
                    scale.SetUInt64(4);
                    scaledMarginLow.SetPow2(exponent);
                    scaledMarginHigh.SetPow2(exponent + 1);
                }
                else
                {
                    scaledValue.SetUInt64(mantissa);
                    scaledValue.ShiftLeft(2);
                    scale.SetPow2(-exponent + 2);
                    scaledMarginLow.SetUInt64(1);
                    scaledMarginHigh.SetUInt64(2);
                }
            }
            else
            {
                if (exponent > 0)
                {
                    scaledValue.SetUInt64(mantissa);
                    scaledValue.ShiftLeft(exponent + 1);
                    scale.SetUInt64(2);
                    scaledMarginLow.SetPow2(exponent);
                }
                else
                {
                    scaledValue.SetUInt64(mantissa);
                    scaledValue.ShiftLeft(1);
                    scale.SetPow2(-exponent + 1);
                    scaledMarginLow.SetUInt64(1);
                }
            }

            // NumPy's estimate of floor(log10(v)) + 1, low by one at most.
            const double log10_2 = 0.30102999566398119521373889472449;
            int digitExponent = (int)Math.Ceiling((double)(value.MantissaBit + exponent) * log10_2 - 0.69);

            // A fractional cutoff above the first digit: start at the cutoff (the digit there is then rounded).
            if (cutoffMax >= 0 && cutoffMode == CutoffMode.FractionLength && digitExponent <= -cutoffMax)
                digitExponent = -cutoffMax + 1;

            // Divide the value by 10^digitExponent: a positive exponent multiplies the scale, a negative one the value and
            // the margins.
            if (digitExponent > 0)
            {
                BigInt.MultiplyByPow10(scale, digitExponent, temp);
            }
            else if (digitExponent < 0)
            {
                BigInt.MultiplyByPow10(scaledValue, -digitExponent, temp);
                BigInt.MultiplyByPow10(scaledMarginLow, -digitExponent, temp);
                if (unequal)
                    BigInt.Multiply2(scaledMarginHigh, scaledMarginLow);
            }

            // value >= 1: the estimate was low — take the next exponent without the first multiplication.
            if (BigInt.Compare(scaledValue, scale) >= 0)
            {
                digitExponent++;
            }
            else
            {
                scaledValue.Multiply10();
                scaledMarginLow.Multiply10();
                if (unequal)
                    BigInt.Multiply2(scaledMarginHigh, scaledMarginLow);
            }

            // The exponent of the last digit to print (the buffer bounds it), and of the last forced digit.
            int cutoffMaxExponent = digitExponent - bufferSize;
            if (cutoffMax >= 0)
            {
                int desired = cutoffMode == CutoffMode.TotalLength ? digitExponent - cutoffMax : -cutoffMax;
                if (desired > cutoffMaxExponent)
                    cutoffMaxExponent = desired;
            }
            int cutoffMinExponent = digitExponent;
            if (cutoffMin >= 0)
            {
                int desired = cutoffMode == CutoffMode.TotalLength ? digitExponent - cutoffMin : -cutoffMin;
                if (desired < cutoffMinExponent)
                    cutoffMinExponent = desired;
            }

            printExponent = digitExponent - 1;

            // Normalize for the division: the scale's top block into [8, (2^64 - 1) / 10] with its highest bit at 59 (see
            // the file header), every number shifted by the same amount.
            ulong hiBlock = scale.Top;
            if (hiBlock < 8 || hiBlock > 1844674407370955161UL)
            {
                int hiBlockLog2 = 63 - BitOperations.LeadingZeroCount(hiBlock);
                int shift = (64 + 59 - hiBlockLog2) % 64;
                scale.ShiftLeft(shift);
                scaledValue.ShiftLeft(shift);
                scaledMarginLow.ShiftLeft(shift);
                if (unequal)
                    BigInt.Multiply2(scaledMarginHigh, scaledMarginLow);
            }

            bool low, high;
            int outputDigit;
            if (digitMode == DigitMode.Unique)
            {
                // Print until the digits distinguish the value from its neighbours (and the minimum is reached), or until
                // the cutoff.
                bool saturated = false;
                for (;;)
                {
                    digitExponent--;
                    outputDigit = BigInt.DivideWithRemainderMaxQuotient9(scaledValue, scale);
                    BigInt.Add(temp, scaledValue, scaledMarginHigh);
                    int cmp = BigInt.Compare(scaledValue, scaledMarginLow);
                    low = isEven ? cmp <= 0 : cmp < 0;
                    cmp = BigInt.Compare(temp, scale);
                    high = isEven ? cmp >= 0 : cmp > 0;
                    if (((low | high) & (digitExponent <= cutoffMinExponent)) | (digitExponent == cutoffMaxExponent))
                        break;
                    buffer[cur++] = (char)('0' + outputDigit);
                    scaledValue.Multiply10();
                    // The lower margin at or above the scale: every later remainder is below it and every later upper
                    // bound above the scale, so low and high stay true — drop the margins (see the file header).
                    if (low & high && BigInt.Compare(scaledMarginLow, scale) >= 0)
                    {
                        saturated = true;
                        break;
                    }
                    scaledMarginLow.Multiply10();
                    if (unequal)
                        BigInt.Multiply2(scaledMarginHigh, scaledMarginLow);
                }

                if (saturated)
                {
                    // NumPy's loop with low = high = true: it stops at the minimum or at the cutoff. The margins are free
                    // now, so they hold the chunk step's numbers (19 digits a division while 19 more fit before either
                    // stop; a zero remainder does not stop this loop, its chunks are then zeros).
                    low = high = true;
                    int chunkFloor = Math.Max(cutoffMinExponent, cutoffMaxExponent);
                    ChunkDigits(scaledValue, scale, s.MarginLow, s.MarginHigh, buffer, ref cur, ref digitExponent, chunkFloor,
                        stopAtZero: false, out _);
                    for (;;)
                    {
                        digitExponent--;
                        outputDigit = BigInt.DivideWithRemainderMaxQuotient9(scaledValue, scale);
                        if ((digitExponent <= cutoffMinExponent) | (digitExponent == cutoffMaxExponent))
                            break;
                        buffer[cur++] = (char)('0' + outputDigit);
                        scaledValue.Multiply10();
                    }
                }
            }
            else
            {
                // Print until the remaining value is zero or until the cutoff. The margins are unused in exact mode, so
                // they hold the chunk step's numbers (19 digits a division while 19 more fit before the cutoff).
                low = false;
                high = false;
                if (ChunkDigits(scaledValue, scale, scaledMarginLow, s.MarginHigh, buffer, ref cur, ref digitExponent,
                        cutoffMaxExponent, stopAtZero: true, out int lastDigit))
                {
                    // The remainder turned zero inside a chunk: NumPy's loop would have broken at that chunk's last
                    // non-zero digit, which the round-off below then stores as it is (a zero remainder rounds down).
                    outputDigit = lastDigit;
                }
                else
                {
                    for (;;)
                    {
                        digitExponent--;
                        outputDigit = BigInt.DivideWithRemainderMaxQuotient9(scaledValue, scale);
                        if (scaledValue.IsZero | (digitExponent == cutoffMaxExponent))
                            break;
                        buffer[cur++] = (char)('0' + outputDigit);
                        scaledValue.Multiply10();
                    }
                }
            }

            // Round the last digit: down when only the lower neighbour is close, else by comparing the remainder with half
            // a digit (2 * remainder against the scale), a tie to the even digit.
            bool roundDown = low;
            if (low == high)
            {
                scaledValue.Multiply2InPlace();
                int compare = BigInt.Compare(scaledValue, scale);
                roundDown = compare < 0;
                if (compare == 0)
                    roundDown = (outputDigit & 1) == 0;
            }

            if (roundDown)
            {
                buffer[cur++] = (char)('0' + outputDigit);
            }
            else if (outputDigit == 9)
            {
                // The carry: drop trailing 9s and increment the first other digit, or print a 1 one exponent higher.
                for (;;)
                {
                    if (cur == start)
                    {
                        buffer[cur++] = '1';
                        printExponent += 1;
                        break;
                    }
                    --cur;
                    if (buffer[cur] != '9')
                    {
                        buffer[cur] += (char)1;
                        ++cur;
                        break;
                    }
                }
            }
            else
            {
                buffer[cur++] = (char)('0' + outputDigit + 1);
            }

            return cur - start;
        }

        /// <summary>The digits one chunk step produces (the most a 64-bit quotient below 10^19 holds).</summary>
        private const int DigitsPerChunk = 19;

        /// <summary>10^18: the first chunk's multiplier (the value enters the loop already one digit ahead).</summary>
        private const ulong Pow10_18 = 1_000_000_000_000_000_000UL;

        /// <summary>10^19: every later chunk's multiplier.</summary>
        private const ulong Pow10_19 = 10_000_000_000_000_000_000UL;

        /// <summary>
        ///     The chunk step of the margin-free loops (see the file header): while <see cref="DigitsPerChunk"/> more digits
        ///     come before <paramref name="floorExponent"/>, writes the next 19 digits of the value with ONE division and
        ///     moves <paramref name="digitExponent"/> past them; then hands the loop back its value in its own scale
        ///     (the remainder times ten, as after a stored digit).
        /// </summary>
        /// <param name="value">The loop's value (below ten times the scale); on return, the value the digit loop continues
        ///     from — or zero after a stop.</param>
        /// <param name="scale">The scale (normalized for the digit loop; not changed).</param>
        /// <param name="n">Scratch: the chunk numerator (the value shifted and multiplied up).</param>
        /// <param name="d">Scratch: the scale shifted to its top bit (the division's normalized divisor).</param>
        /// <param name="buffer">The digit buffer.</param>
        /// <param name="cur">The write position; advanced past the stored digits.</param>
        /// <param name="digitExponent">The loop's digit exponent; decremented by 19 per chunk.</param>
        /// <param name="floorExponent">The exponent the chunked digits must stay above (the loop's stop: the cutoff, or
        ///     the higher of the minimum and the cutoff).</param>
        /// <param name="stopAtZero">Exact mode: stop when the remainder turns zero (NumPy's loop breaks there).</param>
        /// <param name="lastDigit">After a stop: the chunk's last non-zero digit, NOT stored (the caller's round-off
        ///     stores it); otherwise 0.</param>
        /// <returns>True when the remainder turned zero inside a chunk and <paramref name="stopAtZero"/> asked to stop
        ///     there: the digits end at <paramref name="lastDigit"/>.</returns>
        private static bool ChunkDigits(BigInt value, BigInt scale, BigInt n, BigInt d, char[] buffer, ref int cur,
            ref int digitExponent, int floorExponent, bool stopAtZero, out int lastDigit)
        {
            lastDigit = 0;
            if (digitExponent - DigitsPerChunk <= floorExponent || value.IsZero)
                return false;

            // The divisor normalized to bit 63 (Knuth's precondition), the value scaled by the same power of two: the
            // quotient is unchanged and the remainder comes out scaled by it.
            int shift = BitOperations.LeadingZeroCount(scale.Top);
            d.CopyFrom(scale);
            d.ShiftLeft(shift);
            n.CopyFrom(value);
            n.ShiftLeft(shift);
            n.MultiplySmall(Pow10_18);   // value * 10^18 < 10^19 * scale: the quotient is the next 19 digits
            for (;;)
            {
                ulong q = BigInt.DivideNormalized(n, d);
                // The 19 digits of q, leading zeros included, written from the last.
                for (int i = cur + DigitsPerChunk - 1; i >= cur; i--)
                {
                    ulong t = q / 10;
                    buffer[i] = (char)('0' + (int)(q - t * 10));
                    q = t;
                }
                digitExponent -= DigitsPerChunk;
                if (stopAtZero && n.IsZero)
                {
                    // NumPy's loop stops at the first digit after which the remainder is zero: the chunk's last non-zero
                    // digit (the digits after it are the zeros of a zero remainder).
                    int end = cur + DigitsPerChunk;
                    while (buffer[end - 1] == '0')
                        end--;
                    lastDigit = buffer[end - 1] - '0';
                    cur = end - 1;
                    value.SetUInt64(0);
                    return true;
                }
                cur += DigitsPerChunk;
                if (digitExponent - DigitsPerChunk > floorExponent)
                {
                    n.MultiplySmall(Pow10_19);   // remainder * 10^19 < 10^19 * scale: the next 19 digits
                    continue;
                }
                // Back to the digit loop: the remainder in the loop's scale, times ten (as after a stored digit).
                value.CopyFrom(n);
                value.ShiftRight(shift);
                value.Multiply10();
                return false;
            }
        }

        /// <summary>One thread's big integers (allocated once per thread, reused by every call).</summary>
        private sealed class BigIntScratch
        {
            /// <summary>The value (NumPy's <c>scaledValue</c>; the remainder once the loop runs).</summary>
            public readonly BigInt Value = new();

            /// <summary>The common denominator (NumPy's <c>scale</c>).</summary>
            public readonly BigInt Scale = new();

            /// <summary>Half the distance to the next value below (NumPy's <c>scaledMarginLow</c>).</summary>
            public readonly BigInt MarginLow = new();

            /// <summary>Half the distance to the next value above, with unequal margins (NumPy's <c>optionalMarginHigh</c>).</summary>
            public readonly BigInt MarginHigh = new();

            /// <summary>The products and the value's upper bound (NumPy's <c>temp1</c>).</summary>
            public readonly BigInt Temp = new();
        }

        /// <summary>
        ///     NumPy's <c>BigInt</c>: an unsigned integer of up to <see cref="Capacity"/> 64-bit blocks, least significant
        ///     first, changed in place (see the file header). <see cref="Length"/> counts the blocks in use; the top one is
        ///     never zero, and zero has length 0.
        /// </summary>
        /// <remarks>
        ///     The blocks live in a pinned array (the pinned object heap), so <see cref="Blocks"/> stays valid for the
        ///     object's lifetime and the loops run on raw pointers. Capacity: binary64's largest intermediate is about 1150
        ///     bits (5e-324 times 10^323, normalized), 18 blocks; 40 leave room, and every growing operation checks it.
        /// </remarks>
        internal sealed unsafe class BigInt
        {
            /// <summary>The block capacity (2560 bits).</summary>
            internal const int Capacity = 40;

            /// <summary>The pinned storage behind <see cref="Blocks"/> (kept alive by this reference).</summary>
            private readonly ulong[] _storage = GC.AllocateUninitializedArray<ulong>(Capacity, pinned: true);

            /// <summary>The blocks, least significant first (a pointer into the pinned storage).</summary>
            internal readonly ulong* Blocks;

            /// <summary>The number of blocks in use.</summary>
            internal int Length;

            /// <summary>Creates a zero.</summary>
            internal BigInt()
            {
                Blocks = (ulong*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_storage));
            }

            /// <summary>Whether the number is zero.</summary>
            internal bool IsZero => Length == 0;

            /// <summary>The most significant block (0 for zero).</summary>
            internal ulong Top => Length == 0 ? 0 : Blocks[Length - 1];

            /// <summary>Sets the number to <paramref name="v"/> (NumPy's <c>BigInt_Set_uint64</c>).</summary>
            /// <param name="v">The value.</param>
            internal void SetUInt64(ulong v)
            {
                if (v == 0)
                {
                    Length = 0;
                    return;
                }
                Blocks[0] = v;
                Length = 1;
            }

            /// <summary>Sets the number to 2^<paramref name="exponent"/> (NumPy's <c>BigInt_Pow2</c>).</summary>
            /// <param name="exponent">The exponent (non-negative).</param>
            /// <exception cref="InvalidOperationException">The power does not fit the capacity (never for binary64).</exception>
            internal void SetPow2(int exponent)
            {
                int blockIndex = exponent >> 6;
                EnsureCapacity(blockIndex + 1);
                for (int i = 0; i < blockIndex; i++)
                    Blocks[i] = 0;
                Blocks[blockIndex] = 1UL << (exponent & 63);
                Length = blockIndex + 1;
            }

            /// <summary>
            ///     Shifts the number left by <paramref name="shift"/> bits in place (NumPy's <c>BigInt_ShiftLeft</c>:
            ///     blocks processed from the top, so the shift can run in place).
            /// </summary>
            /// <param name="shift">The bit count (non-negative).</param>
            /// <exception cref="InvalidOperationException">The result does not fit the capacity.</exception>
            internal void ShiftLeft(int shift)
            {
                if (Length == 0 || shift == 0)
                    return;
                int shiftBlocks = shift >> 6;
                int shiftBits = shift & 63;
                if (shiftBits == 0)
                {
                    EnsureCapacity(Length + shiftBlocks);
                    for (int i = Length - 1; i >= 0; i--)
                        Blocks[i + shiftBlocks] = Blocks[i];
                    for (int i = 0; i < shiftBlocks; i++)
                        Blocks[i] = 0;
                    Length += shiftBlocks;
                    return;
                }

                int inBlockIdx = Length - 1;
                int outBlockIdx = Length + shiftBlocks;
                EnsureCapacity(outBlockIdx + 1);
                int lowBitsShift = 64 - shiftBits;
                ulong highBits = 0;
                ulong block = Blocks[inBlockIdx];
                ulong lowBits = block >> lowBitsShift;
                Length = outBlockIdx + 1;
                while (inBlockIdx > 0)
                {
                    Blocks[outBlockIdx] = highBits | lowBits;
                    highBits = block << shiftBits;
                    --inBlockIdx;
                    --outBlockIdx;
                    block = Blocks[inBlockIdx];
                    lowBits = block >> lowBitsShift;
                }
                Blocks[outBlockIdx] = highBits | lowBits;
                Blocks[outBlockIdx - 1] = block << shiftBits;
                for (int i = 0; i < shiftBlocks; i++)
                    Blocks[i] = 0;
                // The top block may hold none of the shifted bits.
                if (Blocks[Length - 1] == 0)
                    --Length;
            }

            /// <summary>Copies another number into this one (NumPy's <c>BigInt_Copy</c>).</summary>
            /// <param name="other">The source (a different object).</param>
            internal void CopyFrom(BigInt other)
            {
                Buffer.MemoryCopy(other.Blocks, Blocks, Capacity * sizeof(ulong), other.Length * sizeof(ulong));
                Length = other.Length;
            }

            /// <summary>Shifts the number right by <paramref name="shift"/> bits in place, dropping the bits shifted out.</summary>
            /// <param name="shift">The bit count, 0 to 63.</param>
            internal void ShiftRight(int shift)
            {
                if (shift == 0 || Length == 0)
                    return;
                ulong* b = Blocks;
                int n = Length;
                for (int i = 0; i < n - 1; i++)
                    b[i] = (b[i] >> shift) | (b[i + 1] << (64 - shift));
                b[n - 1] >>= shift;
                if (b[n - 1] == 0)
                    --Length;
            }

            /// <summary>Multiplies the number by <paramref name="m"/> in place (one pass).</summary>
            /// <param name="m">The multiplier.</param>
            /// <exception cref="InvalidOperationException">The result does not fit the capacity.</exception>
            internal void MultiplySmall(ulong m)
            {
                ulong carry = 0;
                ulong* b = Blocks;
                for (int i = 0, n = Length; i < n; i++)
                {
                    ulong hi = Math.BigMul(b[i], m, out ulong lo);
                    lo += carry;
                    hi += lo < carry ? 1UL : 0UL;
                    b[i] = lo;
                    carry = hi;
                }
                if (carry != 0)
                {
                    EnsureCapacity(Length + 1);
                    b[Length++] = carry;
                }
            }

            /// <summary>
            ///     Knuth's algorithm D for ONE quotient word: divides <paramref name="n"/> by the normalized
            ///     <paramref name="d"/> (top bit set) when the quotient fits in 64 bits, leaving the remainder in
            ///     <paramref name="n"/>.
            /// </summary>
            /// <param name="n">The dividend (below 2^64 times the divisor, so at most one block longer than it), replaced
            ///     by the remainder.</param>
            /// <param name="d">The divisor: its top block's highest bit set.</param>
            /// <returns>The quotient.</returns>
            /// <remarks>
            ///     The estimate from the dividend's top two blocks over the divisor's top block, refined with the next
            ///     block of each, is the quotient or one above it; a negative difference after the multiply-subtract
            ///     means one above — add the divisor back. (A one-block divisor has no second block: the estimate is
            ///     then exact.)
            /// </remarks>
            internal static ulong DivideNormalized(BigInt n, BigInt d)
            {
                int L = d.Length, nl = n.Length;
                Debug.Assert(L > 0 && (d.Top >> 63) == 1, "Dragon4: the chunk divisor is not normalized");
                Debug.Assert(nl <= L + 1, "Dragon4: the chunk quotient does not fit in 64 bits");
                if (nl < L)
                    return 0;
                ulong* np = n.Blocks, dp = d.Blocks;
                ulong n2 = nl > L ? np[L] : 0;
                ulong n1 = np[L - 1];
                ulong d1 = dp[L - 1];
                ulong qhat, rhat;
                bool rhatOverflow = false;
                if (n2 >= d1)
                {
                    // The two-block estimate would not fit: 2^64 - 1 is an upper bound of the quotient.
                    qhat = ulong.MaxValue;
                    UInt128 rem = (((UInt128)n2 << 64) | n1) - (UInt128)qhat * d1;
                    rhatOverflow = (rem >> 64) != 0;
                    rhat = (ulong)rem;
                }
                else
                {
                    UInt128 num = ((UInt128)n2 << 64) | n1;
                    qhat = (ulong)(num / d1);
                    rhat = (ulong)(num - (UInt128)qhat * d1);
                }
                if (L >= 2 && !rhatOverflow)
                {
                    ulong d0 = dp[L - 2], n0 = np[L - 2];
                    // qhat * d0 > rhat * 2^64 + n0: one too big — at most twice.
                    while ((UInt128)qhat * d0 > (((UInt128)rhat << 64) | n0))
                    {
                        qhat--;
                        ulong old = rhat;
                        rhat += d1;
                        if (rhat < old)
                            break;   // rhat passed 2^64: the test cannot fail any more
                    }
                }

                // n -= qhat * d over the divisor's blocks, then the dividend's top block.
                ulong borrow = 0, carry = 0;
                for (int i = 0; i < L; i++)
                {
                    ulong hi = Math.BigMul(dp[i], qhat, out ulong lo);
                    lo += carry;
                    hi += lo < carry ? 1UL : 0UL;
                    carry = hi;
                    ulong x = np[i];
                    ulong diff = x - lo;
                    ulong b1 = x < lo ? 1UL : 0UL;
                    ulong diff2 = diff - borrow;
                    ulong b2 = diff < borrow ? 1UL : 0UL;
                    np[i] = diff2;
                    borrow = b1 | b2;
                }
                ulong t1 = n2 - carry;
                bool neg = n2 < carry;
                ulong t2 = t1 - borrow;
                neg |= t1 < borrow;
                if (neg)
                {
                    // One too big: add the divisor back (the carry out cancels the negative top).
                    qhat--;
                    ulong c = 0;
                    for (int i = 0; i < L; i++)
                    {
                        ulong a = np[i];
                        ulong sum = a + dp[i];
                        ulong c1 = sum < a ? 1UL : 0UL;
                        ulong sum2 = sum + c;
                        ulong c2 = sum2 < sum ? 1UL : 0UL;
                        np[i] = sum2;
                        c = c1 + c2;
                    }
                }
                else
                {
                    Debug.Assert(t2 == 0, "Dragon4: the chunk remainder is not below the divisor");
                }
                int length = L;
                while (length > 0 && np[length - 1] == 0)
                    --length;
                n.Length = length;
                return qhat;
            }

            /// <summary>Test helper: sets the number from blocks, least significant first (leading zero blocks dropped).</summary>
            /// <param name="blocks">The blocks.</param>
            /// <exception cref="InvalidOperationException">More blocks than the capacity.</exception>
            internal void SetBlocks(ReadOnlySpan<ulong> blocks)
            {
                EnsureCapacity(blocks.Length);
                for (int i = 0; i < blocks.Length; i++)
                    Blocks[i] = blocks[i];
                int length = blocks.Length;
                while (length > 0 && Blocks[length - 1] == 0)
                    --length;
                Length = length;
            }

            /// <summary>Test helper: the number as a <see cref="BigInteger"/>.</summary>
            /// <returns>The value.</returns>
            internal BigInteger ToBigInteger()
            {
                var bytes = new byte[Length * 8 + 1];
                for (int i = 0; i < Length; i++)
                    BitConverter.TryWriteBytes(bytes.AsSpan(i * 8, 8), Blocks[i]);
                return new BigInteger(bytes);   // little-endian, the extra zero byte keeps it positive
            }

            /// <summary>Multiplies the number by 10 in place (NumPy's <c>BigInt_Multiply10</c>).</summary>
            /// <exception cref="InvalidOperationException">The result does not fit the capacity.</exception>
            internal void Multiply10()
            {
                ulong carry = 0;
                ulong* b = Blocks;
                for (int i = 0, n = Length; i < n; i++)
                {
                    ulong hi = Math.BigMul(b[i], 10UL, out ulong lo);
                    lo += carry;
                    hi += lo < carry ? 1UL : 0UL;
                    b[i] = lo;
                    carry = hi;
                }
                if (carry != 0)
                {
                    EnsureCapacity(Length + 1);
                    b[Length++] = carry;
                }
            }

            /// <summary>Doubles the number in place (NumPy's <c>BigInt_Multiply2_inplace</c>).</summary>
            /// <exception cref="InvalidOperationException">The result does not fit the capacity.</exception>
            internal void Multiply2InPlace()
            {
                ulong carry = 0;
                ulong* b = Blocks;
                for (int i = 0, n = Length; i < n; i++)
                {
                    ulong cur = b[i];
                    b[i] = (cur << 1) | carry;
                    carry = cur >> 63;
                }
                if (carry != 0)
                {
                    EnsureCapacity(Length + 1);
                    b[Length++] = carry;
                }
            }

            /// <summary>Sets <paramref name="result"/> to twice <paramref name="input"/> (NumPy's <c>BigInt_Multiply2</c>).</summary>
            /// <param name="result">The result (a different object).</param>
            /// <param name="input">The number to double.</param>
            /// <exception cref="InvalidOperationException">The result does not fit the capacity.</exception>
            internal static void Multiply2(BigInt result, BigInt input)
            {
                ulong carry = 0;
                ulong* r = result.Blocks, a = input.Blocks;
                int n = input.Length;
                for (int i = 0; i < n; i++)
                {
                    ulong cur = a[i];
                    r[i] = (cur << 1) | carry;
                    carry = cur >> 63;
                }
                if (carry != 0)
                {
                    result.EnsureCapacity(n + 1);
                    r[n] = carry;
                    result.Length = n + 1;
                }
                else
                {
                    result.Length = n;
                }
            }

            /// <summary>Sets <paramref name="result"/> to <paramref name="lhs"/> + <paramref name="rhs"/> (NumPy's <c>BigInt_Add</c>).</summary>
            /// <param name="result">The result (a different object than either operand).</param>
            /// <param name="lhs">The first addend.</param>
            /// <param name="rhs">The second addend.</param>
            /// <exception cref="InvalidOperationException">The result does not fit the capacity.</exception>
            internal static void Add(BigInt result, BigInt lhs, BigInt rhs)
            {
                BigInt large = lhs, small = rhs;
                if (lhs.Length < rhs.Length)
                {
                    small = lhs;
                    large = rhs;
                }
                ulong* r = result.Blocks, l = large.Blocks, sm = small.Blocks;
                int nl = large.Length, ns = small.Length;
                ulong carry = 0;
                int i = 0;
                for (; i < ns; i++)
                {
                    ulong a = l[i];
                    ulong sum = a + sm[i];
                    ulong c1 = sum < a ? 1UL : 0UL;
                    ulong sum2 = sum + carry;
                    ulong c2 = sum2 < sum ? 1UL : 0UL;
                    r[i] = sum2;
                    carry = c1 + c2;
                }
                for (; i < nl; i++)
                {
                    ulong a = l[i];
                    ulong sum = a + carry;
                    carry = sum < a ? 1UL : 0UL;
                    r[i] = sum;
                }
                if (carry != 0)
                {
                    result.EnsureCapacity(nl + 1);
                    r[nl] = 1;
                    result.Length = nl + 1;
                }
                else
                {
                    result.Length = nl;
                }
            }

            /// <summary>Compares two numbers (NumPy's <c>BigInt_Compare</c>).</summary>
            /// <param name="lhs">The first number.</param>
            /// <param name="rhs">The second number.</param>
            /// <returns>Negative, zero or positive as <paramref name="lhs"/> is below, equal to or above <paramref name="rhs"/>.</returns>
            internal static int Compare(BigInt lhs, BigInt rhs)
            {
                int lengthDiff = lhs.Length - rhs.Length;
                if (lengthDiff != 0)
                    return lengthDiff;
                ulong* a = lhs.Blocks, b = rhs.Blocks;
                for (int i = lhs.Length - 1; i >= 0; i--)
                {
                    if (a[i] == b[i])
                        continue;
                    return a[i] > b[i] ? 1 : -1;
                }
                return 0;
            }

            /// <summary>
            ///     Multiplies <paramref name="x"/> by 10^<paramref name="exponent"/> in place (NumPy's
            ///     <c>BigInt_MultiplyPow10</c> and the multiplication by <c>BigInt_Pow10</c>), through the table of exact
            ///     powers and <paramref name="temp"/>.
            /// </summary>
            /// <param name="x">The number.</param>
            /// <param name="exponent">The power of ten (non-negative; at most <see cref="Pow10Table.MaxExponent"/>).</param>
            /// <param name="temp">Scratch (a different object).</param>
            /// <exception cref="InvalidOperationException">The power is outside the table, or the product does not fit
            ///     the capacity (neither happens for binary16 / 32 / 64).</exception>
            internal static void MultiplyByPow10(BigInt x, int exponent, BigInt temp)
            {
                if (exponent == 0 || x.Length == 0)
                    return;
                if ((uint)exponent > Pow10Table.MaxExponent)
                    throw new InvalidOperationException($"Dragon4: 10^{exponent} is outside the power table");
                ulong[] p = Pow10Table.Powers[exponent];
                fixed (ulong* pb = p)
                    MultiplyInto(temp, x.Blocks, x.Length, pb, p.Length);
                x.EnsureCapacity(temp.Length);
                Buffer.MemoryCopy(temp.Blocks, x.Blocks, Capacity * sizeof(ulong), temp.Length * sizeof(ulong));
                x.Length = temp.Length;
            }

            /// <summary>Sets <paramref name="result"/> to the product of two block runs (NumPy's <c>BigInt_Multiply</c>:
            ///     long multiplication of the larger operand by each block of the smaller).</summary>
            /// <param name="result">The result (not aliasing either run).</param>
            /// <param name="a">The first run.</param>
            /// <param name="na">Its length.</param>
            /// <param name="b">The second run.</param>
            /// <param name="nb">Its length.</param>
            /// <exception cref="InvalidOperationException">The product does not fit the capacity.</exception>
            private static void MultiplyInto(BigInt result, ulong* a, int na, ulong* b, int nb)
            {
                ulong* large = a, small = b;
                int nl = na, ns = nb;
                if (na < nb)
                {
                    large = b;
                    small = a;
                    nl = nb;
                    ns = na;
                }
                int maxResultLen = nl + ns;
                result.EnsureCapacity(maxResultLen);
                ulong* r = result.Blocks;
                for (int i = 0; i < maxResultLen; i++)
                    r[i] = 0;
                for (int j = 0; j < ns; j++)
                {
                    ulong multiplier = small[j];
                    if (multiplier == 0)
                        continue;
                    ulong carry = 0;
                    ulong* rc = r + j;
                    for (int i = 0; i < nl; i++)
                    {
                        // rc[i] + large[i] * multiplier + carry fits in 128 bits.
                        ulong hi = Math.BigMul(large[i], multiplier, out ulong lo);
                        lo += rc[i];
                        hi += lo < rc[i] ? 1UL : 0UL;
                        lo += carry;
                        hi += lo < carry ? 1UL : 0UL;
                        rc[i] = lo;
                        carry = hi;
                    }
                    rc[nl] = carry;
                }
                result.Length = maxResultLen > 0 && r[maxResultLen - 1] == 0 ? maxResultLen - 1 : maxResultLen;
            }

            /// <summary>
            ///     NumPy's <c>BigInt_DivideWithRemainder_MaxQuotient9</c>: divides <paramref name="dividend"/> by
            ///     <paramref name="divisor"/> where the quotient is known to be at most 9, leaving the remainder in the
            ///     dividend.
            /// </summary>
            /// <param name="dividend">The dividend, replaced by the remainder. Its length is at most the divisor's (the
            ///     loop's invariant, which the normalization keeps).</param>
            /// <param name="divisor">The divisor, normalized (its top block in [8, (2^64 - 1) / 10]).</param>
            /// <returns>The quotient digit, 0 to 9.</returns>
            internal static int DivideWithRemainderMaxQuotient9(BigInt dividend, BigInt divisor)
            {
                int length = divisor.Length;
                // A shorter dividend is smaller: the quotient is zero and the dividend is the remainder.
                if (dividend.Length < length)
                    return 0;
                Debug.Assert(dividend.Length == length, "Dragon4: the dividend outgrew the divisor");
                ulong* dv = dividend.Blocks, ds = divisor.Blocks;

                // The estimate from the top blocks matches the quotient or undershoots it by one.
                ulong quotient = dv[length - 1] / (ds[length - 1] + 1);
                Debug.Assert(quotient <= 9);
                if (quotient != 0)
                {
                    // dividend -= divisor * quotient
                    ulong borrow = 0, carry = 0;
                    for (int i = 0; i < length; i++)
                    {
                        ulong hi = Math.BigMul(ds[i], quotient, out ulong lo);
                        lo += carry;
                        hi += lo < carry ? 1UL : 0UL;
                        carry = hi;
                        ulong d = dv[i];
                        ulong diff = d - lo;
                        ulong b1 = d < lo ? 1UL : 0UL;
                        ulong diff2 = diff - borrow;
                        ulong b2 = diff < borrow ? 1UL : 0UL;
                        dv[i] = diff2;
                        borrow = b1 | b2;
                    }
                    while (length > 0 && dv[length - 1] == 0)
                        --length;
                    dividend.Length = length;
                }

                // Still at or above the divisor: the estimate was one short — subtract one more divisor.
                if (Compare(dividend, divisor) >= 0)
                {
                    ++quotient;
                    ulong borrow = 0;
                    int n = divisor.Length;
                    for (int i = 0; i < n; i++)
                    {
                        ulong d = dv[i], s = ds[i];
                        ulong diff = d - s;
                        ulong b1 = d < s ? 1UL : 0UL;
                        ulong diff2 = diff - borrow;
                        ulong b2 = diff < borrow ? 1UL : 0UL;
                        dv[i] = diff2;
                        borrow = b1 | b2;
                    }
                    length = n;
                    while (length > 0 && dv[length - 1] == 0)
                        --length;
                    dividend.Length = length;
                }
                return (int)quotient;
            }

            /// <summary>Refuses a length beyond the capacity (the blocks would be written past the pinned storage).</summary>
            /// <param name="length">The length about to be used.</param>
            /// <exception cref="InvalidOperationException">The length exceeds <see cref="Capacity"/>.</exception>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private void EnsureCapacity(int length)
            {
                if (length > Capacity)
                    throw new InvalidOperationException($"Dragon4: a {length}-block intermediate exceeds the {Capacity}-block capacity");
            }
        }

        /// <summary>The exact powers of ten 10^0 .. 10^<see cref="MaxExponent"/> as 64-bit block runs (built on first use).</summary>
        private static class Pow10Table
        {
            /// <summary>The largest power in the table: binary64 needs 10^324 at most (5e-324 times 10^323, and its
            ///     rounding neighbour), binary16 / 32 far less.</summary>
            internal const int MaxExponent = 350;

            /// <summary>10^k for k in [0, <see cref="MaxExponent"/>], least significant block first, no zero top block.</summary>
            internal static readonly ulong[][] Powers = Build();

            /// <summary>Builds the table by repeated multiplication by ten.</summary>
            /// <returns>The table.</returns>
            private static ulong[][] Build()
            {
                var table = new ulong[MaxExponent + 1][];
                var cur = new ulong[] { 1 };
                table[0] = cur;
                for (int k = 1; k <= MaxExponent; k++)
                {
                    var next = new ulong[cur.Length + 1];
                    ulong carry = 0;
                    for (int i = 0; i < cur.Length; i++)
                    {
                        ulong hi = Math.BigMul(cur[i], 10UL, out ulong lo);
                        lo += carry;
                        hi += lo < carry ? 1UL : 0UL;
                        next[i] = lo;
                        carry = hi;
                    }
                    next[cur.Length] = carry;
                    cur = carry == 0 ? next[..cur.Length] : next;
                    table[k] = cur;
                }
                return table;
            }
        }
    }
}
