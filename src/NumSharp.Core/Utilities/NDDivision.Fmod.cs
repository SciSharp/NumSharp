using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// C-style floating remainder helpers for <c>np.fmod</c> (<c>loops_modulo.dispatch.c.src</c>
    /// integer <c>@TYPE@_fmod</c> and the float <c>npy_fmod@c@</c> alias). Unlike
    /// <see cref="NDDivision"/>'s <c>Rem*</c> family (floored / Python sign convention, sign of the
    /// DIVISOR), <c>fmod</c> uses TRUNCATED division so the result takes the sign of the DIVIDEND.
    ///
    /// Semantics replicated exactly (probed against NumPy 2.4.2):
    /// <list type="bullet">
    /// <item>Integer <c>fmod</c> by zero returns <c>0</c> (NumPy yields 0, never throwing); the signed
    /// <c>MIN % -1</c> overflow case also returns <c>0</c> (matching NumPy's
    /// <c>DIVIDEBYZERO_OVERFLOW_CHECK</c> guard).</item>
    /// <item>Integer <c>fmod</c> is C# truncated <c>%</c> (sign of dividend): <c>fmod(-7,3) == -1</c>,
    /// <c>fmod(7,-3) == 1</c>.</item>
    /// <item>Float <c>fmod</c> is C# <c>a % b</c> — bit-identical to C <c>fmod</c>/<c>fmodf</c> including
    /// <c>fmod(x,0) == NaN</c>, <c>fmod(±inf,y) == NaN</c>, <c>fmod(x,±inf) == x</c>, and the signed-zero
    /// result <c>fmod(-6,3) == -0.0</c> (verified 2026-09-12).</item>
    /// </list>
    /// Sub-int (sbyte/short) compute in the int domain (C# widens the operands), so the hardware
    /// <c>MIN/-1</c> #DE trap cannot fire and the narrowing cast reproduces NumPy's wrap; int/long
    /// short-circuit <c>d == -1</c> to dodge the .NET <see cref="OverflowException"/> on <c>MIN % -1</c>.
    /// </summary>
    public static partial class NDDivision
    {
        // ----- Signed integers -----
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static sbyte FmodSByte(sbyte n, sbyte d) => d == 0 ? (sbyte)0 : unchecked((sbyte)(n % d));

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static short FmodInt16(short n, short d) => d == 0 ? (short)0 : unchecked((short)(n % d));

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static int FmodInt32(int n, int d)
        {
            if (d == 0) return 0;
            if (d == -1) return 0; // n % -1 == 0 for all n; also dodges the MIN % -1 OverflowException
            return n % d;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static long FmodInt64(long n, long d)
        {
            if (d == 0) return 0;
            if (d == -1) return 0;
            return n % d;
        }

        // ----- Unsigned integers (fmod == C# remainder; sign is moot) -----
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static byte FmodByte(byte n, byte d) => d == 0 ? (byte)0 : (byte)(n % d);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ushort FmodUInt16(ushort n, ushort d) => d == 0 ? (ushort)0 : (ushort)(n % d);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static char FmodChar(char n, char d) => d == 0 ? (char)0 : (char)(n % d);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static uint FmodUInt32(uint n, uint d) => d == 0 ? 0u : n % d;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ulong FmodUInt64(ulong n, ulong d) => d == 0 ? 0ul : n % d;

        // ----- Floating point (C# '%' IS C fmod/fmodf — sign of dividend, incl. all specials) -----
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static float FmodSingle(float a, float b) => a % b;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static double FmodDouble(double a, double b) => a % b;

        // ----- Decimal (NumSharp extension; no NumPy analog) -----
        // Truncated remainder = a - Truncate(a/b)*b. decimal '%' already truncates toward zero, so
        // this matches; b == 0 throws DivideByZeroException like every other decimal division path.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static decimal FmodDecimal(decimal a, decimal b) => a % b;
    }
}
