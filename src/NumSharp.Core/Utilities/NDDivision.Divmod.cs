using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Fused divmod helpers for <c>np.divmod</c> — each returns the floored quotient and writes the
    /// floored remainder to the <c>out</c> parameter in ONE pass, matching NumPy's <c>@TYPE@_divmod</c>
    /// (<c>loops_modulo.dispatch.c.src</c>) and the float <c>npy_divmod@c@</c>. By construction the two
    /// outputs equal <c>(FloorDiv*(n,d), Rem*(n,d))</c> element-for-element (the very definition of
    /// <c>np.divmod</c>: <c>(a // b, a % b)</c>), so the fused kernel is bit-identical to composing
    /// <c>np.floor_divide</c> with <c>np.remainder</c>.
    ///
    /// Integer helpers use the ONE-IDIV form <c>q = n / d; r = n - q*d</c> (truncated), then the
    /// floored fix-up <c>if (r != 0 &amp;&amp; sign(n) != sign(d)) { q--; r += d; }</c>. Computing
    /// <c>r</c> as <c>n - q*d</c> rather than a second <c>n % d</c> avoids a second hardware division
    /// (idiv already yields the quotient), which roughly HALVES the integer kernel cost — verified
    /// bit-identical to the two-idiv form over 2M random inputs incl. <c>MIN/-1</c> and <c>÷0</c>.
    /// <c>|q*d| &lt;= |n|</c> always, so <c>n - q*d</c> never overflows.
    ///
    /// Divide-by-zero yields (0, 0) for integers (NumPy: quotient 0, remainder 0, sets the FPE flag it
    /// does not throw on); the signed <c>MIN/-1</c> overflow yields (MIN, 0) (NumPy sets the overflow
    /// flag and returns NPY_MIN). Float <c>b == 0</c> yields (a/b, a%b) = (±inf/nan, nan). The float and
    /// single cores live in <see cref="NDDivision"/> (<c>DivmodDouble</c>/<c>DivmodSingle</c>).
    /// </summary>
    public static partial class NDDivision
    {
        // ----- Signed integers (sub-int widen to the int domain; int/long guard d==-1) -----
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static sbyte DivmodSByte(sbyte n, sbyte d, out sbyte mod)
        {
            if (d == 0) { mod = 0; return 0; }
            int q = n / d;             // int domain: sbyte.MIN / -1 == 128 (no #DE), narrows to -128
            int r = n - q * d;
            if (r != 0 && ((n > 0) != (d > 0))) { r += d; q--; }
            mod = unchecked((sbyte)r);
            return unchecked((sbyte)q);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static short DivmodInt16(short n, short d, out short mod)
        {
            if (d == 0) { mod = 0; return 0; }
            int q = n / d;
            int r = n - q * d;
            if (r != 0 && ((n > 0) != (d > 0))) { r += d; q--; }
            mod = unchecked((short)r);
            return unchecked((short)q);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static int DivmodInt32(int n, int d, out int mod)
        {
            if (d == 0) { mod = 0; return 0; }
            if (d == -1) { mod = 0; return unchecked(-n); } // MIN/-1 -> (MIN, 0); dodges the OverflowException
            int q = n / d;
            int r = n - q * d;
            if (r != 0 && ((n > 0) != (d > 0))) { mod = r + d; return q - 1; }
            mod = r;
            return q;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static long DivmodInt64(long n, long d, out long mod)
        {
            if (d == 0) { mod = 0; return 0; }
            if (d == -1) { mod = 0; return unchecked(-n); }
            long q = n / d;
            long r = n - q * d;
            if (r != 0 && ((n > 0) != (d > 0))) { mod = r + d; return q - 1; }
            mod = r;
            return q;
        }

        // ----- Unsigned integers (floored == truncated; no sign fix-up) -----
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static byte DivmodByte(byte n, byte d, out byte mod)
        {
            if (d == 0) { mod = 0; return 0; }
            int q = n / d;
            mod = (byte)(n - q * d);
            return (byte)q;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ushort DivmodUInt16(ushort n, ushort d, out ushort mod)
        {
            if (d == 0) { mod = 0; return 0; }
            int q = n / d;
            mod = (ushort)(n - q * d);
            return (ushort)q;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static char DivmodChar(char n, char d, out char mod)
        {
            if (d == 0) { mod = (char)0; return (char)0; }
            int q = n / d;
            mod = (char)(n - q * d);
            return (char)q;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static uint DivmodUInt32(uint n, uint d, out uint mod)
        {
            if (d == 0) { mod = 0u; return 0u; }
            uint q = n / d;
            mod = n - q * d;
            return q;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ulong DivmodUInt64(ulong n, ulong d, out ulong mod)
        {
            if (d == 0) { mod = 0ul; return 0ul; }
            ulong q = n / d;
            mod = n - q * d;
            return q;
        }

        // ----- Half (NumPy HALF_divmod: compute in float32, astype 'e'->'f') -----
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Half DivmodHalf(Half a, Half b, out Half mod)
        {
            float floordiv = DivmodSingle((float)a, (float)b, out float m);
            mod = (Half)m;
            return (Half)floordiv;
        }

        // ----- Decimal (NumSharp extension; a - floor(a/b)*b, matching the decimal mod/floor_divide path) -----
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static decimal DivmodDecimal(decimal a, decimal b, out decimal mod)
        {
            decimal floordiv = Math.Floor(a / b);
            mod = a - floordiv * b;
            return floordiv;
        }
    }
}
