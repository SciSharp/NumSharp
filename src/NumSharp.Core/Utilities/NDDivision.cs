using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// floor-division and remainder helpers matching NumPy's <c>floor_div_@TYPE@</c>
    /// (<c>loops_arithmetic.dispatch.c.src</c>), integer <c>remainder</c>
    /// (<c>loops_modulo.dispatch.c.src</c>), and the floating-point
    /// <c>npy_floor_divide@c@</c> / <c>npy_remainder@c@</c> (Python-divmod port in
    /// <c>npy_math_internal.h.src</c>).
    ///
    /// Semantics replicated exactly:
    /// <list type="bullet">
    /// <item>Integer divide/modulo by zero returns <c>0</c> (NumPy raises a RuntimeWarning but
    /// yields 0, never throwing — C#'s <see cref="DivideByZeroException"/> must not surface).</item>
    /// <item>Signed integer floor-division rounds toward negative infinity (Python <c>//</c>),
    /// not toward zero like C# <c>/</c>; <c>MIN // -1</c> wraps to <c>MIN</c> (overflow), matching
    /// NumPy's <c>npy_set_floatstatus_overflow(); return NPY_MIN</c>.</item>
    /// <item>Signed integer remainder uses the floored (Python) sign convention: the result has the
    /// sign of the divisor; <c>MIN % -1 == 0</c>.</item>
    /// <item>Float floor-division/modulo follow CPython's <c>divmod</c> (fmod, sign-fixup,
    /// snap-to-nearest-integer), so <c>a // 0.0</c> is <c>±inf</c>/<c>nan</c> (not forced NaN) and
    /// edge cases like <c>0.7 // 0.1 == 6.0</c> and <c>-2.0 // inf == -1.0</c> match.</item>
    /// </list>
    /// </summary>
    public static class NDDivision
    {
        // ----------------------------------------------------------------------------------------
        // Signed integers — floor division (round toward -inf), divide-by-zero -> 0.
        // Sub-int types (sbyte/short) compute in the int domain (C# widens operands), so the
        // hardware MIN/-1 #DE trap cannot fire; the narrowing cast reproduces NumPy's overflow wrap.
        // int/long guard d == -1 explicitly: n / -1 == -n (wraps MIN -> MIN) and the floor fix-up is
        // a no-op there, which also dodges the .NET OverflowException on MIN / -1.
        // ----------------------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static sbyte FloorDivSByte(sbyte n, sbyte d)
        {
            if (d == 0) return 0;
            int r = n / d;
            if (((n > 0) != (d > 0)) && (r * d != n)) r--;
            return unchecked((sbyte)r);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static short FloorDivInt16(short n, short d)
        {
            if (d == 0) return 0;
            int r = n / d;
            if (((n > 0) != (d > 0)) && (r * d != n)) r--;
            return unchecked((short)r);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static int FloorDivInt32(int n, int d)
        {
            if (d == 0) return 0;
            if (d == -1) return unchecked(-n);
            int r = n / d;
            if (((n > 0) != (d > 0)) && (r * d != n)) r--;
            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static long FloorDivInt64(long n, long d)
        {
            if (d == 0) return 0;
            if (d == -1) return unchecked(-n);
            long r = n / d;
            if (((n > 0) != (d > 0)) && (r * d != n)) r--;
            return r;
        }

        // ----------------------------------------------------------------------------------------
        // Unsigned integers — floor division == truncating division, divide-by-zero -> 0.
        // ----------------------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static byte FloorDivByte(byte n, byte d) => d == 0 ? (byte)0 : (byte)(n / d);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ushort FloorDivUInt16(ushort n, ushort d) => d == 0 ? (ushort)0 : (ushort)(n / d);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static char FloorDivChar(char n, char d) => d == 0 ? (char)0 : (char)(n / d);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static uint FloorDivUInt32(uint n, uint d) => d == 0 ? 0u : n / d;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ulong FloorDivUInt64(ulong n, ulong d) => d == 0 ? 0ul : n / d;

        // ----------------------------------------------------------------------------------------
        // Signed integers — remainder (floored / Python sign convention), divide-by-zero -> 0.
        // The result takes the divisor's sign; d == -1 short-circuits to 0 (true for all n and
        // avoids the MIN % -1 trap).
        // ----------------------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static sbyte RemSByte(sbyte n, sbyte d)
        {
            if (d == 0) return 0;
            int r = n % d;
            if (r != 0 && ((n > 0) != (d > 0))) r += d;
            return unchecked((sbyte)r);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static short RemInt16(short n, short d)
        {
            if (d == 0) return 0;
            int r = n % d;
            if (r != 0 && ((n > 0) != (d > 0))) r += d;
            return unchecked((short)r);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static int RemInt32(int n, int d)
        {
            if (d == 0) return 0;
            if (d == -1) return 0;
            int r = n % d;
            if (r != 0 && ((n > 0) != (d > 0))) r += d;
            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static long RemInt64(long n, long d)
        {
            if (d == 0) return 0;
            if (d == -1) return 0;
            long r = n % d;
            if (r != 0 && ((n > 0) != (d > 0))) r += d;
            return r;
        }

        // ----------------------------------------------------------------------------------------
        // Unsigned integers — remainder == C# remainder, divide-by-zero -> 0.
        // ----------------------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static byte RemByte(byte n, byte d) => d == 0 ? (byte)0 : (byte)(n % d);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ushort RemUInt16(ushort n, ushort d) => d == 0 ? (ushort)0 : (ushort)(n % d);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static char RemChar(char n, char d) => d == 0 ? (char)0 : (char)(n % d);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static uint RemUInt32(uint n, uint d) => d == 0 ? 0u : n % d;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ulong RemUInt64(ulong n, ulong d) => d == 0 ? 0ul : n % d;

        // ----------------------------------------------------------------------------------------
        // Floating point — CPython divmod port (npy_divmod@c@). b == 0 returns a / b (±inf or nan),
        // never a forced NaN. C# float/double '%' is C fmod (truncated remainder), matching npy_fmod.
        // ----------------------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static double FloorDivDouble(double a, double b)
        {
            if (b == 0.0) return a / b;
            return DivmodDouble(a, b, out _);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static double RemDouble(double a, double b)
        {
            if (b == 0.0) return a % b; // fmod -> nan
            DivmodDouble(a, b, out double mod);
            return mod;
        }

        private static double DivmodDouble(double a, double b, out double modulus)
        {
            double mod = a % b; // fmod

            // a - mod should be very nearly an integer multiple of b
            double div = (a - mod) / b;

            // adjust fmod result to conform to Python's floored convention
            if (mod != 0.0)
            {
                if ((b < 0.0) != (mod < 0.0))
                {
                    mod += b;
                    div -= 1.0;
                }
            }
            else
            {
                // ensure correct sign of a zero remainder
                mod = Math.CopySign(0.0, b);
            }

            // snap quotient to nearest integral value
            double floordiv;
            if (div != 0.0)
            {
                floordiv = Math.Floor(div);
                if (div - floordiv > 0.5)
                    floordiv += 1.0;
            }
            else
            {
                floordiv = Math.CopySign(0.0, a / b);
            }

            modulus = mod;
            return floordiv;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static float FloorDivSingle(float a, float b)
        {
            if (b == 0f) return a / b;
            return DivmodSingle(a, b, out _);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static float RemSingle(float a, float b)
        {
            if (b == 0f) return a % b; // fmodf -> nan
            DivmodSingle(a, b, out float mod);
            return mod;
        }

        private static float DivmodSingle(float a, float b, out float modulus)
        {
            float mod = a % b; // fmodf

            float div = (a - mod) / b;

            if (mod != 0f)
            {
                if ((b < 0f) != (mod < 0f))
                {
                    mod += b;
                    div -= 1f;
                }
            }
            else
            {
                mod = MathF.CopySign(0f, b);
            }

            float floordiv;
            if (div != 0f)
            {
                floordiv = MathF.Floor(div);
                if (div - floordiv > 0.5f)
                    floordiv += 1f;
            }
            else
            {
                floordiv = MathF.CopySign(0f, a / b);
            }

            modulus = mod;
            return floordiv;
        }
    }
}
