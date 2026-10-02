using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Greatest-common-divisor / lowest-common-multiple element helpers for <c>np.gcd</c> and
    /// <c>np.lcm</c> — a line-for-line port of NumPy 2.4.2's <c>npy_gcd@c@</c> / <c>npy_lcm@c@</c>
    /// (<c>numpy/_core/src/npymath/npy_math_internal.h.src</c>) driving the integer <c>@TYPE@_gcd</c> /
    /// <c>@TYPE@_lcm</c> loops (<c>loops.c.src</c>). One helper per NumSharp integer dtype, mirroring the
    /// <see cref="NDDivision"/> <c>Fmod*</c> layout, so the IL kernels resolve a single <c>call</c> per
    /// element (<c>DirectILKernelGenerator.EmitGcdLcmOperation</c>). gcd/lcm have integer loops ONLY —
    /// bool/float/complex/decimal are rejected at the <c>np.*</c> boundary before any kernel runs, so no
    /// float helper exists here.
    ///
    /// <para><b>Algorithm (NumPy verbatim).</b> The unsigned helpers are the core:
    /// <c>gcd(a,b)</c> is the iterative Euclidean remainder loop <c>while(a!=0){c=a; a=b%a; b=c;} return b;</c>
    /// and <c>lcm(a,b) = gcd==0 ? 0 : a/gcd*b</c> (divide-BEFORE-multiply, to widen the exactly-divisible
    /// factor out first — the product can still overflow the dtype and WRAPS, matching NumPy). The signed
    /// helpers delegate to the unsigned core on the operands' magnitudes exactly as NumPy does
    /// (<c>npy_gcd(a,b) = npy_gcdu(a&lt;0?-a:a, b&lt;0?-b:b)</c>), then reinterpret the unsigned result back to
    /// the signed type.</para>
    ///
    /// <para><b>Two consequences that look like bugs but are NumPy-exact</b> (probed against 2.4.2, so do
    /// NOT "fix" them): the result of a signed gcd/lcm can be <b>negative</b> when the unsigned magnitude
    /// wraps the signed range — e.g. <c>gcd(int8 -128, -128) == -128</c> (the magnitude 128 has no positive
    /// int8), <c>lcm(int32.MinValue, 1) == int32.MinValue</c>; and an overflowing lcm product <b>wraps</b>
    /// silently — e.g. <c>lcm(int16 21000, 14000) == -23536</c>, <c>lcm(uint32.MaxValue, MaxValue-1) == 2</c>.
    /// The <c>|MIN|</c> magnitude is obtained by the C idiom <c>(U)(unchecked(-x))</c>: negating the signed
    /// minimum wraps back to itself, and its bit pattern reinterpreted as unsigned IS <c>2^(width-1)</c>
    /// (the true magnitude) — which is why every negate/cast below is <see cref="unchecked"/>.</para>
    ///
    /// <para><b>Why the 8/16-bit helpers may compute in their own width rather than promoting to int32</b>
    /// (as NumPy's <c>BYTE_gcd</c> does via C integer promotion): for gcd, all Euclidean intermediates are
    /// bounded by <c>max(|a|,|b|)</c> and so never exceed the width; for lcm, <c>|a|/gcd</c> is exact and
    /// both it and <c>|b|</c> are &lt; <c>2^width</c>, and modular multiplication is width-invariant
    /// (<c>(x*y) mod 2^w == ((x mod 2^w)*(y mod 2^w)) mod 2^w</c>), so narrowing the int32-domain product
    /// gives the identical wrapped byte/short result. C# already widens <c>byte</c>/<c>ushort</c>/<c>char</c>
    /// operands to <c>int</c> for <c>%</c>/<c>*</c>, so the narrowing cast on store closes the equivalence.</para>
    /// </summary>
    public static partial class NDGcdLcm
    {
        // ============================ GCD ============================
        // ----- Unsigned integers (the Euclidean core) -----

        /// <summary>Euclidean GCD of two <see cref="byte"/> magnitudes (np.gcd uint8 loop). Returns 0 only when both inputs are 0.</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The greatest common divisor, in <c>[0, 255]</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static byte GcdByte(byte a, byte b)
        {
            // Iterative Euclid: a shrinks to b%a each step until it hits 0; b then holds the gcd.
            while (a != 0) { byte c = a; a = (byte)(b % a); b = c; }
            return b;
        }

        /// <summary>Euclidean GCD of two <see cref="ushort"/> magnitudes (np.gcd uint16 loop).</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The greatest common divisor, in <c>[0, 65535]</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ushort GcdUInt16(ushort a, ushort b)
        {
            while (a != 0) { ushort c = a; a = (ushort)(b % a); b = c; }
            return b;
        }

        /// <summary>Euclidean GCD of two <see cref="char"/> code-unit magnitudes (NumSharp Char extension; treated as uint16).</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The greatest common divisor as a <see cref="char"/> code unit.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static char GcdChar(char a, char b)
        {
            while (a != 0) { char c = a; a = (char)(b % a); b = c; }
            return b;
        }

        /// <summary>Euclidean GCD of two <see cref="uint"/> magnitudes (np.gcd uint32 loop).</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The greatest common divisor.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static uint GcdUInt32(uint a, uint b)
        {
            while (a != 0) { uint c = a; a = b % a; b = c; }
            return b;
        }

        /// <summary>Euclidean GCD of two <see cref="ulong"/> magnitudes (np.gcd uint64 loop).</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The greatest common divisor.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ulong GcdUInt64(ulong a, ulong b)
        {
            while (a != 0) { ulong c = a; a = b % a; b = c; }
            return b;
        }

        // ----- Signed integers (magnitude -> unsigned core -> reinterpret) -----

        /// <summary>Signed GCD (np.gcd int8 loop): <c>GcdByte(|a|,|b|)</c> reinterpreted to <see cref="sbyte"/>.</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The gcd; <b>negative</b> only for <c>gcd(-128,-128) == -128</c> (magnitude 128 wraps int8).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static sbyte GcdSByte(sbyte a, sbyte b)
            => unchecked((sbyte)GcdByte((byte)(a < 0 ? -a : a), (byte)(b < 0 ? -b : b)));

        /// <summary>Signed GCD (np.gcd int16 loop): <c>GcdUInt16(|a|,|b|)</c> reinterpreted to <see cref="short"/>.</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The gcd; negative only when the magnitude wraps int16 (e.g. <c>gcd(-32768,-32768)</c>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static short GcdInt16(short a, short b)
            => unchecked((short)GcdUInt16((ushort)(a < 0 ? -a : a), (ushort)(b < 0 ? -b : b)));

        /// <summary>Signed GCD (np.gcd int32 loop): <c>GcdUInt32(|a|,|b|)</c> reinterpreted to <see cref="int"/>.</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The gcd; negative only when the magnitude wraps int32 (e.g. <c>gcd(int.MinValue, int.MinValue)</c>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static int GcdInt32(int a, int b)
            => unchecked((int)GcdUInt32((uint)(a < 0 ? -a : a), (uint)(b < 0 ? -b : b)));

        /// <summary>Signed GCD (np.gcd int64 loop): <c>GcdUInt64(|a|,|b|)</c> reinterpreted to <see cref="long"/>.</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The gcd; negative only when the magnitude wraps int64 (e.g. <c>gcd(long.MinValue, long.MinValue)</c>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static long GcdInt64(long a, long b)
            => unchecked((long)GcdUInt64((ulong)(a < 0 ? -a : a), (ulong)(b < 0 ? -b : b)));

        // ============================ LCM ============================
        // ----- Unsigned integers (the core: gcd==0 ? 0 : a/gcd*b, wrapping) -----

        /// <summary>Lowest common multiple of two <see cref="byte"/> magnitudes (np.lcm uint8 loop). Wraps mod 256 on overflow.</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The lcm (0 if either operand is 0).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static byte LcmByte(byte a, byte b)
        {
            // gcd==0 means a==b==0; NumPy returns 0 rather than dividing by zero.
            byte g = GcdByte(a, b);
            return g == 0 ? (byte)0 : unchecked((byte)(a / g * b));
        }

        /// <summary>Lowest common multiple of two <see cref="ushort"/> magnitudes (np.lcm uint16 loop). Wraps mod 65536 on overflow.</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The lcm (0 if either operand is 0).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ushort LcmUInt16(ushort a, ushort b)
        {
            ushort g = GcdUInt16(a, b);
            return g == 0 ? (ushort)0 : unchecked((ushort)(a / g * b));
        }

        /// <summary>Lowest common multiple of two <see cref="char"/> code-unit magnitudes (NumSharp Char extension). Wraps mod 65536.</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The lcm as a <see cref="char"/> code unit (0 if either operand is 0).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static char LcmChar(char a, char b)
        {
            char g = GcdChar(a, b);
            return g == 0 ? (char)0 : unchecked((char)(a / g * b));
        }

        /// <summary>Lowest common multiple of two <see cref="uint"/> magnitudes (np.lcm uint32 loop). Wraps mod 2^32 on overflow.</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The lcm (0 if either operand is 0).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static uint LcmUInt32(uint a, uint b)
        {
            uint g = GcdUInt32(a, b);
            return g == 0 ? 0u : unchecked(a / g * b);
        }

        /// <summary>Lowest common multiple of two <see cref="ulong"/> magnitudes (np.lcm uint64 loop). Wraps mod 2^64 on overflow.</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The lcm (0 if either operand is 0).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ulong LcmUInt64(ulong a, ulong b)
        {
            ulong g = GcdUInt64(a, b);
            return g == 0 ? 0ul : unchecked(a / g * b);
        }

        // ----- Signed integers (magnitude -> unsigned core -> reinterpret) -----

        /// <summary>Signed LCM (np.lcm int8 loop): <c>LcmByte(|a|,|b|)</c> reinterpreted to <see cref="sbyte"/> (wraps).</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The lcm; may be negative when the (wrapping) magnitude product exceeds sbyte range.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static sbyte LcmSByte(sbyte a, sbyte b)
            => unchecked((sbyte)LcmByte((byte)(a < 0 ? -a : a), (byte)(b < 0 ? -b : b)));

        /// <summary>Signed LCM (np.lcm int16 loop): <c>LcmUInt16(|a|,|b|)</c> reinterpreted to <see cref="short"/> (wraps).</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The lcm; may be negative when the (wrapping) magnitude product exceeds short range.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static short LcmInt16(short a, short b)
            => unchecked((short)LcmUInt16((ushort)(a < 0 ? -a : a), (ushort)(b < 0 ? -b : b)));

        /// <summary>Signed LCM (np.lcm int32 loop): <c>LcmUInt32(|a|,|b|)</c> reinterpreted to <see cref="int"/> (wraps).</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The lcm; may be negative when the (wrapping) magnitude product exceeds int range (e.g. <c>lcm(int.MinValue, 1)</c>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static int LcmInt32(int a, int b)
            => unchecked((int)LcmUInt32((uint)(a < 0 ? -a : a), (uint)(b < 0 ? -b : b)));

        /// <summary>Signed LCM (np.lcm int64 loop): <c>LcmUInt64(|a|,|b|)</c> reinterpreted to <see cref="long"/> (wraps).</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The lcm; may be negative when the (wrapping) magnitude product exceeds long range.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static long LcmInt64(long a, long b)
            => unchecked((long)LcmUInt64((ulong)(a < 0 ? -a : a), (ulong)(b < 0 ? -b : b)));
    }
}
