using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Integer power helpers matching NumPy's <c>@TYPE@_power</c> loop in <c>loops.c.src</c>.
    /// Uses repeated-squaring with native dtype wraparound (e.g. <c>uint8 ** 8 = 0</c>).
    ///
    /// These helpers assume the exponent is non-negative. NumPy raises
    /// <c>ValueError("Integers to negative integer powers are not allowed.")</c> for any
    /// negative integer exponent, regardless of base value; the caller is responsible
    /// for that pre-check (see <c>DefaultEngine.Power</c>).
    /// </summary>
    public static class NpyIntegerPower
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static sbyte PowSByte(sbyte a, sbyte b)
        {
            sbyte r = 1, x = a;
            long e = b;
            unchecked
            {
                while (e > 0)
                {
                    if ((e & 1) == 1) r = (sbyte)(r * x);
                    e >>= 1;
                    if (e > 0) x = (sbyte)(x * x);
                }
            }
            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static byte PowByte(byte a, byte b)
        {
            byte r = 1, x = a;
            long e = b;
            unchecked
            {
                while (e > 0)
                {
                    if ((e & 1) == 1) r = (byte)(r * x);
                    e >>= 1;
                    if (e > 0) x = (byte)(x * x);
                }
            }
            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static short PowInt16(short a, short b)
        {
            short r = 1, x = a;
            long e = b;
            unchecked
            {
                while (e > 0)
                {
                    if ((e & 1) == 1) r = (short)(r * x);
                    e >>= 1;
                    if (e > 0) x = (short)(x * x);
                }
            }
            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ushort PowUInt16(ushort a, ushort b)
        {
            ushort r = 1, x = a;
            long e = b;
            unchecked
            {
                while (e > 0)
                {
                    if ((e & 1) == 1) r = (ushort)(r * x);
                    e >>= 1;
                    if (e > 0) x = (ushort)(x * x);
                }
            }
            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static char PowChar(char a, char b)
        {
            char r = (char)1;
            char x = a;
            long e = b;
            unchecked
            {
                while (e > 0)
                {
                    if ((e & 1) == 1) r = (char)(r * x);
                    e >>= 1;
                    if (e > 0) x = (char)(x * x);
                }
            }
            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static int PowInt32(int a, int b)
        {
            int r = 1, x = a;
            long e = b;
            unchecked
            {
                while (e > 0)
                {
                    if ((e & 1) == 1) r = r * x;
                    e >>= 1;
                    if (e > 0) x = x * x;
                }
            }
            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static uint PowUInt32(uint a, uint b)
        {
            uint r = 1, x = a;
            long e = b;
            unchecked
            {
                while (e > 0)
                {
                    if ((e & 1) == 1) r = r * x;
                    e >>= 1;
                    if (e > 0) x = x * x;
                }
            }
            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static long PowInt64(long a, long b)
        {
            long r = 1, x = a, e = b;
            unchecked
            {
                while (e > 0)
                {
                    if ((e & 1) == 1) r = r * x;
                    e >>= 1;
                    if (e > 0) x = x * x;
                }
            }
            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ulong PowUInt64(ulong a, ulong b)
        {
            ulong r = 1, x = a, e = b;
            unchecked
            {
                while (e > 0)
                {
                    if ((e & 1) == 1) r = r * x;
                    e >>= 1;
                    if (e > 0) x = x * x;
                }
            }
            return r;
        }
    }
}
