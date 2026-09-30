using System;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// ILKernelGenerator.Polynomial.Lanes.cs — how every dtype lives in a vector chain
// =============================================================================
//
// A vector chain carries W points per value; W is the LOOP dtype's lane count at 256 bits on an AVX2
// host (float64 4, float32 8, float16 8 — as float32 lanes on the f16 grid, complex128 2). Every OTHER
// dtype a per-point value can have — x's dtype, the per-point coefficients' dtype, int64 for a bool x's
// Python-int products, the peeled steps' coefficient dtype — must then hold exactly W lanes too, and
// convert lane-for-lane (exactly, as NumPy's loop-input cast does) into its promotion target:
//
//   dtype                 W = 2 (complex loop)          W = 4 (float64 loop)     W = 8 (float32 / float16 loop)
//   complex128            Vector256<double> [re,im]x2   —                        —
//   float64/int64/uint64  Vector128<T>                  Vector256<T>             —
//   float32/int32/uint32  Vector128<T>, 2 lanes live    Vector128<T>             Vector256<float>
//   float16               8 float32 lanes, 2 live       8 float32 lanes, 4 live  8 float32 lanes
//   bool/int8/uint8/      Vector128<int>, 2 live        Vector128<int>           Vector256<int>
//   int16/uint16/char     (int32 containers: the value sign/zero-extended, re-wrapped to the dtype's
//                          width after every op — NumPy's int8 arithmetic wraps at 8 bits)
//
// A "live" count below the container's lane count means the load reads EXACTLY W elements (never past
// the chain's last point) and the spare lanes hold zeros that no store ever reads. Every conversion
// listed by LaneConvertible is exact for all inputs, or rounds exactly once as NumPy's cast does: the
// int64/uint64 ones are .NET's ConvertToDouble (vcvtuqq2pd with AVX-512DQ, else an exact hi/lo split and
// ONE rounding add). The scalar tail rounds once too: EmitConvertTo's uint64 -> float64 goes through
// Converts' round-once helper, because .NET 8's own cast rounds twice above 2^63 (ebe18b7c). So a
// vector chain is bit-identical, lane for lane, to the scalar chain the same kernel runs for its tail.
// Gated at the float64 ties above 2^63, on both TFMs, by the review probe's T1 block
// (docs/plans/numpy-polynomial-review.md).
//
// COMPLEX LANES
// -------------
// Multiply is NumPy's array multiply simd_cmul, instruction for instruction: vfmaddsub(a_re, b, a_im*swap(b))
// — the fused a_re*b products with the pre-rounded a_im products as the addend, exactly the per-element
// NDComplexMath.Multiply. Divide is CDOUBLE_divide's Smith algorithm with its three branches selected per
// lane by blends; a SHARED divisor (the steps' `/nd`) is prepared once per block (rat, scl, branch) so
// the per-point work is two multiplies and an add/subtract — no division per point, where NumPy runs two.
//
// Hosts without AVX2 keep the house vector width and run vector chains only when every per-point value
// already has the loop dtype (the generic Vector<T> kinds); every other block runs scalar chains, which
// produce the same bits.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     A complex128 divisor shared by all points of a block, prepared once: Smith's <c>rat</c> and
    ///     <c>scl</c> broadcast to every lane and the branch <c>CDOUBLE_divide</c> takes for it.
    /// </summary>
    internal readonly struct PolyCDivShared
    {
        /// <summary>Branch 0 (<c>|br| &gt;= |bi|</c>) and 1 (else): <c>rat</c> in every lane. Branch 2 (zero
        ///     divisor): <c>[|br|, |bi|, |br|, |bi|]</c>, the component-wise divisors NumPy uses.</summary>
        public readonly Vector256<double> Rat;
        /// <summary>Branches 0/1: <c>scl = 1/(den + num*rat)</c> in every lane.</summary>
        public readonly Vector256<double> Scl;
        /// <summary>0, 1 or 2 — see <see cref="Rat"/>.</summary>
        public readonly int Mode;

        /// <summary>Creates a prepared divisor.</summary>
        /// <param name="rat">Rat lanes.</param><param name="scl">Scl lanes.</param><param name="mode">Branch.</param>
        public PolyCDivShared(Vector256<double> rat, Vector256<double> scl, int mode) { Rat = rat; Scl = scl; Mode = mode; }
    }

    /// <summary>
    ///     The lane-level primitives the polynomial vector chains call (all AggressiveInlining — the JIT
    ///     folds them into the kernel): exact partial loads, int-width wrap-around, the exact lane conversions,
    ///     and NumPy's complex multiply/divide on two complex values per <c>Vector256&lt;double&gt;</c>.
    ///     x86 only: every caller is gated on <see cref="PolyLanes.MixedLanes"/> (AVX2; FMA for complex).
    /// </summary>
    internal static unsafe class PolyLaneOps
    {
        // ---------------------------------------------------------------- int32 containers (narrow ints)

        /// <summary>Sign/zero-extends the low lanes of <paramref name="raw"/> (reinterpreted as
        ///     <typeparamref name="T"/>) to int32 — pmovsx/pmovzx. bool and byte zero-extend.</summary>
        /// <typeparam name="T">bool, sbyte, byte, short, ushort or char.</typeparam>
        /// <param name="raw">The loaded bytes.</param><returns>4 int32 lanes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<int> Widen128<T>(Vector128<byte> raw)
        {
            if (typeof(T) == typeof(sbyte)) return Sse41.ConvertToVector128Int32(raw.AsSByte());
            if (typeof(T) == typeof(short)) return Sse41.ConvertToVector128Int32(raw.AsInt16());
            if (typeof(T) == typeof(ushort) || typeof(T) == typeof(char)) return Sse41.ConvertToVector128Int32(raw.AsUInt16());
            return Sse41.ConvertToVector128Int32(raw);
        }

        /// <summary>Loads exactly 2 elements of <typeparamref name="T"/> into int32 lanes 0-1 (2-3 are zero).</summary>
        /// <typeparam name="T">A 1- or 2-byte integer-like type.</typeparam>
        /// <param name="p">Address of the first element.</param><returns>The container.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<int> NarrowLoad2<T>(byte* p) where T : unmanaged
            => Unsafe.SizeOf<T>() == 1
                ? Widen128<T>(Vector128.CreateScalar((uint)Unsafe.ReadUnaligned<ushort>(p)).AsByte())
                : Widen128<T>(Vector128.CreateScalar(Unsafe.ReadUnaligned<uint>(p)).AsByte());

        /// <summary>Loads exactly 4 elements of <typeparamref name="T"/> into 4 int32 lanes.</summary>
        /// <typeparam name="T">A 1- or 2-byte integer-like type.</typeparam>
        /// <param name="p">Address of the first element.</param><returns>The container.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<int> NarrowLoad4<T>(byte* p) where T : unmanaged
            => Unsafe.SizeOf<T>() == 1
                ? Widen128<T>(Vector128.CreateScalar(Unsafe.ReadUnaligned<uint>(p)).AsByte())
                : Widen128<T>(Vector128.CreateScalar(Unsafe.ReadUnaligned<ulong>(p)).AsByte());

        /// <summary>Loads exactly 8 elements of <typeparamref name="T"/> into 8 int32 lanes.</summary>
        /// <typeparam name="T">A 1- or 2-byte integer-like type.</typeparam>
        /// <param name="p">Address of the first element.</param><returns>The container.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> NarrowLoad8<T>(byte* p) where T : unmanaged
        {
            var raw = Unsafe.SizeOf<T>() == 1
                ? Vector128.CreateScalar(Unsafe.ReadUnaligned<ulong>(p)).AsByte()
                : Vector128.Load(p);
            if (typeof(T) == typeof(sbyte)) return Avx2.ConvertToVector256Int32(raw.AsSByte());
            if (typeof(T) == typeof(short)) return Avx2.ConvertToVector256Int32(raw.AsInt16());
            if (typeof(T) == typeof(ushort) || typeof(T) == typeof(char)) return Avx2.ConvertToVector256Int32(raw.AsUInt16());
            return Avx2.ConvertToVector256Int32(raw);
        }

        /// <summary>Re-wraps int32 lanes to <typeparamref name="T"/>'s width after an op — NumPy's integer
        ///     arithmetic wraps modulo 2^bits (the low bits of an int32 add/sub/mul are exact).</summary>
        /// <typeparam name="T">sbyte, byte, short, ushort or char.</typeparam>
        /// <param name="v">Lanes.</param><returns>The wrapped lanes, sign/zero-extended again.</returns>
        /// <exception cref="NotSupportedException">bool: NumPy never does bool arithmetic here (a bool x meets
        ///     Python ints as int64).</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<int> Wrap128<T>(Vector128<int> v)
        {
            if (typeof(T) == typeof(sbyte)) return Sse2.ShiftRightArithmetic(Sse2.ShiftLeftLogical(v, 24), 24);
            if (typeof(T) == typeof(short)) return Sse2.ShiftRightArithmetic(Sse2.ShiftLeftLogical(v, 16), 16);
            if (typeof(T) == typeof(byte)) return Sse2.And(v, Vector128.Create(0xFF));
            if (typeof(T) == typeof(ushort) || typeof(T) == typeof(char)) return Sse2.And(v, Vector128.Create(0xFFFF));
            throw new NotSupportedException($"no {typeof(T).Name} lane arithmetic");
        }

        /// <summary>The 256-bit <see cref="Wrap128{T}"/>.</summary>
        /// <typeparam name="T">sbyte, byte, short, ushort or char.</typeparam>
        /// <param name="v">Lanes.</param><returns>The wrapped lanes.</returns>
        /// <exception cref="NotSupportedException">bool.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Wrap256<T>(Vector256<int> v)
        {
            if (typeof(T) == typeof(sbyte)) return Avx2.ShiftRightArithmetic(Avx2.ShiftLeftLogical(v, 24), 24);
            if (typeof(T) == typeof(short)) return Avx2.ShiftRightArithmetic(Avx2.ShiftLeftLogical(v, 16), 16);
            if (typeof(T) == typeof(byte)) return Avx2.And(v, Vector256.Create(0xFF));
            if (typeof(T) == typeof(ushort) || typeof(T) == typeof(char)) return Avx2.And(v, Vector256.Create(0xFFFF));
            throw new NotSupportedException($"no {typeof(T).Name} lane arithmetic");
        }

        // ---------------------------------------------------------------- partial loads

        /// <summary>Loads exactly 2 elements of a 4-byte <typeparamref name="T"/> into lanes 0-1 (2-3 zero).</summary>
        /// <typeparam name="T">float, int or uint.</typeparam>
        /// <param name="p">Address.</param><returns>The container.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<T> PackedLoad2<T>(byte* p) where T : struct
            => Vector128.CreateScalar(Unsafe.ReadUnaligned<ulong>(p)).As<ulong, T>();

        /// <summary>Loads exactly <paramref name="w"/> float16 values (2, 4 or 8) as exact float32 lanes; the
        ///     spare lanes are +0.</summary>
        /// <param name="p">Address.</param><param name="w">Element count (a JIT constant at every call site).</param>
        /// <returns>8 float32 lanes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> HalfLoad(byte* p, int w)
        {
            var raw = w switch
            {
                8 => Vector128.Load((ushort*)p),
                4 => Vector128.CreateScalar(Unsafe.ReadUnaligned<ulong>(p)).AsUInt16(),
                _ => Vector128.CreateScalar(Unsafe.ReadUnaligned<uint>(p)).AsUInt16(),
            };
            return DirectILKernelGenerator.HalfWiden8V(raw);
        }

        /// <summary>Loads 8 float16 values (16 bytes) as exact float32 lanes: the gang load one widen of which
        ///     serves 8/w adjacent chains (see <c>PolyLoadCoef</c>).</summary>
        /// <param name="p">Address.</param><returns>8 float32 lanes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> HalfLoad8(byte* p) => HalfLoad(p, 8);

        /// <summary>The 2-live-lane float16 load (<see cref="HalfLoad"/> with w = 2).</summary>
        /// <param name="p">Address.</param><returns>8 float32 lanes, 2 live.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> HalfLoad2(byte* p) => HalfLoad(p, 2);

        /// <summary>The 4-live-lane float16 load (<see cref="HalfLoad"/> with w = 4).</summary>
        /// <param name="p">Address.</param><returns>8 float32 lanes, 4 live.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> HalfLoad4(byte* p) => HalfLoad(p, 4);

        // ---------------------------------------------------------------- exact conversions

        /// <summary>uint32 → float64, 4 lanes: <c>double(int32(u ^ 2^31)) + 2^31</c> — both steps exact, since
        ///     every uint32 is a double.</summary>
        /// <param name="v">Lanes.</param><returns>The doubles.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> U32ToF64x4(Vector128<uint> v)
            => Avx.Add(Avx.ConvertToVector256Double(Sse2.Xor(v, Vector128.Create(0x80000000u)).AsInt32()), Vector256.Create(2147483648.0));

        /// <summary>uint32 → float64, 2 lanes (the low lanes of the container); exact like <see cref="U32ToF64x4"/>.</summary>
        /// <param name="v">Lanes (0-1 live).</param><returns>The doubles.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<double> U32ToF64x2(Vector128<uint> v)
            => Sse2.Add(Sse2.ConvertToVector128Double(Sse2.Xor(v, Vector128.Create(0x80000000u)).AsInt32()), Vector128.Create(2147483648.0));

        /// <summary>Live float16 lanes (as float32) → float64: 4 lanes.</summary>
        /// <param name="v">8 float32 lanes, 0-3 live.</param><returns>The doubles (exact).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> F32LowToF64x4(Vector256<float> v) => Avx.ConvertToVector256Double(v.GetLower());

        /// <summary>Live float16 lanes (as float32) → float64: 2 lanes.</summary>
        /// <param name="v">8 float32 lanes, 0-1 live.</param><returns>The doubles (exact).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector128<double> F32LowToF64x2(Vector256<float> v) => Sse2.ConvertToVector128Double(v.GetLower());

        /// <summary>float64 → complex128 (NumPy's cast: imaginary +0.0), 2 values: <c>[x0, 0, x1, 0]</c>.</summary>
        /// <param name="v">Two doubles.</param><returns>Two complex values.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> F64ToC128x2(Vector128<double> v)
            => Avx2.Permute4x64(Vector256.Create(v, Vector128<double>.Zero), 0b11_01_10_00);

        // ---------------------------------------------------------------- complex128, 2 per Vector256<double>

        /// <summary><c>[re, im, re, im]</c>.</summary>
        /// <param name="c">The value.</param><returns>Two copies.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> CBroadcast(Complex c) => Vector256.Create(c.Real, c.Imaginary, c.Real, c.Imaginary);

        /// <summary>
        ///     NumPy's array complex multiply <c>simd_cmul</c>: <c>vfmaddsub(a_re, b, a_im * swap(b))</c> — real
        ///     <c>fma(a_re, b_re, -(a_im*b_im))</c>, imaginary <c>fma(a_re, b_im, a_im*b_re)</c>, per value
        ///     bit-identical to <see cref="NumSharp.Utilities.NDComplexMath.Multiply"/>. Requires FMA.
        /// </summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>The products.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> CMul(Vector256<double> a, Vector256<double> b)
            => Fma.MultiplyAddSubtract(Avx.Permute(a, 0b0000), b, Avx.Multiply(Avx.Permute(a, 0b1111), Avx.Permute(b, 0b0101)));

        /// <summary>
        ///     <c>CDOUBLE_divide</c> (Smith's algorithm) per value, its three branches selected by blends: for
        ///     <c>|br| &gt;= |bi|</c> <c>rat = bi/br, scl = 1/(br + bi*rat)</c>, <c>((ar + ai*rat)*scl, (ai - ar*rat)*scl)</c>;
        ///     otherwise <c>rat = br/bi, scl = 1/(bi + br*rat)</c>, <c>((ar*rat + ai)*scl, (ai*rat - ar)*scl)</c>;
        ///     a zero divisor divides component-wise by <c>|br|</c>/<c>|bi|</c>. Same ops, same operand order,
        ///     un-fused — bit-identical to the scalar port for every non-NaN result.
        /// </summary>
        /// <param name="a">Dividends.</param><param name="b">Divisors.</param><returns>The quotients.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> CDiv(Vector256<double> a, Vector256<double> b)
        {
            var bAbs = Avx.AndNot(Vector256.Create(-0.0), b);
            var bSwap = Avx.Permute(b, 0b0101);
            // NaN compares false -> the second branch, as C's `in2r_abs >= in2i_abs` does.
            var ge = Avx.Permute(Avx.Compare(bAbs, Avx.Permute(bAbs, 0b0101), FloatComparisonMode.OrderedGreaterThanOrEqualNonSignaling), 0b0000);
            var num = Avx.BlendVariable(b, bSwap, ge);            // even lane: ge ? bi : br
            var den = Avx.BlendVariable(bSwap, b, ge);            // even lane: ge ? br : bi
            var rat = Avx.Permute(Avx.Divide(num, den), 0b0000);
            var scl = Avx.Permute(Avx.Divide(Vector256.Create(1.0), Avx.Add(den, Avx.Multiply(num, rat))), 0b0000);
            var q = Avx.Multiply(a, rat);                         // [ar*rat, ai*rat]
            var numA = Avx.Permute(Avx.AddSubtract(Avx.Permute(a, 0b0101), q), 0b0101);   // [ar + ai*rat, ai - ar*rat]
            var numB = Avx.Permute(Avx.AddSubtract(Avx.Permute(q, 0b0101), a), 0b0101);   // [ar*rat + ai, ai*rat - ar]
            var r = Avx.Multiply(Avx.BlendVariable(numB, numA, ge), scl);
            var zero = Avx.Compare(bAbs, Vector256<double>.Zero, FloatComparisonMode.OrderedEqualNonSignaling);
            var both = Avx.And(zero, Avx.Permute(zero, 0b0101));
            // The zero-divisor branch is rare: pay its division only when a lane takes it.
            return Avx.MoveMask(both) == 0 ? r : Avx.BlendVariable(r, Avx.Divide(a, bAbs), both);
        }

        /// <summary>Prepares a divisor shared by every point of a block: the branch, <c>rat</c> and <c>scl</c>
        ///     exactly as <c>CDOUBLE_divide</c> computes them per element (so the per-point results match).</summary>
        /// <param name="b">The divisor.</param><returns>The prepared divisor.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static PolyCDivShared CDivPrep(Complex b)
        {
            double br = b.Real, bi = b.Imaginary, brAbs = Math.Abs(br), biAbs = Math.Abs(bi);
            if (brAbs >= biAbs)
            {
                if (brAbs == 0.0 && biAbs == 0.0)
                    return new PolyCDivShared(Vector256.Create(brAbs, biAbs, brAbs, biAbs), default, 2);
                double rat = bi / br;
                return new PolyCDivShared(Vector256.Create(rat), Vector256.Create(1.0 / (br + bi * rat)), 0);
            }
            double rat2 = br / bi;
            return new PolyCDivShared(Vector256.Create(rat2), Vector256.Create(1.0 / (bi + br * rat2)), 1);
        }

        /// <summary>Divides two complex values by a prepared shared divisor (<see cref="CDivPrep"/>): the
        ///     per-point half of Smith's algorithm — no division unless the divisor is zero.</summary>
        /// <param name="a">Dividends.</param><param name="p">The prepared divisor.</param><returns>The quotients.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> CDivBy(Vector256<double> a, in PolyCDivShared p)
        {
            if (p.Mode == 0)
                return Avx.Multiply(Avx.Permute(Avx.AddSubtract(Avx.Permute(a, 0b0101), Avx.Multiply(a, p.Rat)), 0b0101), p.Scl);
            if (p.Mode == 1)
                return Avx.Multiply(Avx.Permute(Avx.AddSubtract(Avx.Permute(Avx.Multiply(a, p.Rat), 0b0101), a), 0b0101), p.Scl);
            return Avx.Divide(a, p.Rat);
        }

        /// <summary>
        ///     NumPy's <c>HALF_divide(h, 2)</c> of 8 float16 values held as exact float32 lanes (<see cref="PolyHalfLaneKind"/>),
        ///     returned the same way: the exact half (a float32 multiply by 0.5 — f16 halves never leave float32's normal
        ///     range) rounded to the float16 grid with ties to even, as <c>npy_float_to_half</c> rounds it. Only a half below
        ///     2^-14 (float16's subnormal range) can leave the grid, by one bit; it is rounded to a multiple of 2^-24 by the
        ///     magic-number trick — <c>0.75 + |y|</c> stays in [0.5, 1), whose float32 ulp IS 2^-24, and 0.75 is an even
        ///     multiple of it, so the add rounds |y| to the grid with ties to even and the subtraction is exact — with the sign
        ///     ORed back so a tie to zero stays -0. Infinities and NaNs pass through the multiply (a NaN quieted, NumPy's
        ///     payload rule for one NaN operand). One multiply and five cheap ops, where a narrow + widen round trip per
        ///     halving (the lane kind's generic <c>Bin</c>) made the fused chebmulx kernel's float16 pass 5x slower.
        /// </summary>
        /// <param name="x">8 float16 values as float32 lanes.</param>
        /// <returns>The 8 halves, on the float16 grid.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> HalfHalveOnGrid(Vector256<float> x)
        {
            var y = Avx.Multiply(x, Vector256.Create(0.5f));
            var sign = Vector256.Create(-0.0f);
            var abs = Avx.AndNot(sign, y);
            // NaN compares false: a NaN keeps y.
            var sub = Avx.Compare(abs, Vector256.Create(6.103515625e-05f), FloatComparisonMode.OrderedLessThanNonSignaling);
            var magic = Vector256.Create(0.75f);
            var rounded = Avx.Or(Avx.Subtract(Avx.Add(abs, magic), magic), Avx.And(y, sign));
            return Avx.BlendVariable(y, rounded, sub);
        }

        // ---------------------------------------------------------------- the calculus kernel's widened c *= scl
        //
        // numpy.polynomial's `c *= scl` with a strong scalar that PROMOTES the series (np.float64 on float32, an int16 or
        // float32 scalar on float16, an int32+ or float64 one on float16) runs in the WIDER loop dtype and casts back:
        // DOUBLE_multiply / FLOAT_multiply over the widened series, then the output cast. Both loops take the FIRST
        // operand's NaN (vmulpd / vmulps(in1, in2)) when both are NaN — the series' — which is imposed explicitly: RyuJIT
        // may swap a commutative multiply's operands. The DirectILKernelGenerator.PolyCalculus load / scale stages call
        // these per 8 series lanes (float32, or float16 as exact float32 values on the f16 grid); the *Scalar twins are
        // the stages' scalar tails, the same bits for one value.

        /// <summary>
        ///     <c>c *= s</c> of 8 float32 lanes in a float64 loop: each lane widened exactly, multiplied in float64, rounded
        ///     once back to float32 (NumPy's output cast) — a NaN lane of <paramref name="c"/> stays <c>quiet(c)</c> whatever
        ///     <paramref name="s"/> is.
        /// </summary>
        /// <param name="c">8 float32 series lanes.</param><param name="s">The float64 scale in every lane.</param>
        /// <returns>The 8 scaled float32 lanes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> F32ScaleF64(Vector256<float> c, Vector256<double> s)
        {
            var lo = Avx.ConvertToVector128Single(Avx.Multiply(Avx.ConvertToVector256Double(c.GetLower()), s));
            var hi = Avx.ConvertToVector128Single(Avx.Multiply(Avx.ConvertToVector256Double(c.GetUpper()), s));
            var r = Vector256.Create(lo, hi);
            // quiet(c) is exactly what cvtps2pd -> vmulpd -> cvtpd2ps makes of a NaN c (all 23 payload bits survive).
            return Avx.BlendVariable(r, Avx.Or(c, Vector256.Create(0x00400000).AsSingle()), Avx.CompareUnordered(c, c));
        }

        /// <summary>One value of <see cref="F32ScaleF64"/> (the scalar tail): the same bits.</summary>
        /// <param name="c">The series value.</param><param name="s">The float64 scale.</param><returns>The scaled value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float F32ScaleF64Scalar(float c, double s)
        {
            if (float.IsNaN(c))
                return BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(c) | 0x00400000);
            return (float)((double)c * s);   // a NaN s: quiet(s), narrowed as cvtsd2ss narrows it
        }

        /// <summary>
        ///     <c>c *= s</c> of 8 float16 lanes (exact float32 values) in a float64 loop: widened, multiplied in float64, and
        ///     rounded ONCE to float16 as NumPy's cast (<c>npy_double_to_half</c>) rounds a double. A plain f64 → f32 → f16
        ///     chain would double-round the products that land just past a float16 tie, so each product is first rounded to
        ///     ODD at float32 precision (<see cref="RoundToOddF32"/>), which the RTNE narrow then rounds exactly as the direct
        ///     f64 → f16 rounding would (float32's 24 bits ≥ float16's 11 + 2). NaN products take <c>cvtpd2ps</c>'s quiet NaN
        ///     (the payload's top bits, as <c>npy_double_to_half</c> keeps them) and a NaN c wins over a NaN s, quieted — the
        ///     float64 loop's first-operand rule, imposed explicitly.
        /// </summary>
        /// <param name="c">8 float16 series lanes as float32.</param><param name="s">The float64 scale in every lane.</param>
        /// <returns>The 8 scaled lanes, on the float16 grid.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> HalfScaleF64(Vector256<float> c, Vector256<double> s)
        {
            var lo = Avx.Multiply(Avx.ConvertToVector256Double(c.GetLower()), s);
            var hi = Avx.Multiply(Avx.ConvertToVector256Double(c.GetUpper()), s);
            var f = Vector256.Create(RoundToOddF32(lo), RoundToOddF32(hi));
            // A NaN product: RoundToOddF32 may have cleared its payload to an inf; cvtpd2ps keeps the payload's top bits.
            var fn = Vector256.Create(Avx.ConvertToVector128Single(lo), Avx.ConvertToVector128Single(hi));
            f = Avx.BlendVariable(f, fn, Avx.CompareUnordered(fn, fn));
            var r = DirectILKernelGenerator.HalfWiden8V(DirectILKernelGenerator.HalfNarrow8V(f));
            return Avx.BlendVariable(r, Avx.Or(c, Vector256.Create(0x00400000).AsSingle()), Avx.CompareUnordered(c, c));
        }

        /// <summary>
        ///     4 doubles rounded to ODD at float32 precision: the low 29 mantissa bits truncated (toward zero — the bits are
        ///     sign-magnitude) and the last kept bit set when any truncated bit was, then converted to float32 — EXACTLY for
        ///     every value in float32's normal range (24 significant bits). Values past it convert to ±inf and values below it
        ///     to a subnormal, neither of which any float16 result can tell apart from the true round-to-odd (they are ±inf /
        ///     ±0 in float16). NaN lanes come out as garbage (possibly inf): callers blend them.
        /// </summary>
        /// <param name="d">4 doubles.</param><returns>4 floats.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> RoundToOddF32(Vector256<double> d)
        {
            var mask = Vector256.Create(0x1FFFFFFFUL);
            var bits = d.AsUInt64();
            var exact = Avx2.CompareEqual(Avx2.And(bits, mask), Vector256<ulong>.Zero);
            var kept = Avx2.AndNot(mask, bits);
            var odd = Avx2.AndNot(exact, Vector256.Create(0x20000000UL));
            return Avx.ConvertToVector128Single(Avx2.Or(kept, odd).AsDouble());
        }

        /// <summary>
        ///     One float16 value (as an exact float32, its NaN payload in bits 13..22) times a float64 scale in NumPy's
        ///     float64 loop, as float16 bits: <c>npy_double_to_half(c * s)</c> — a NaN c wins, quieted (cvtss2sd quiets it,
        ///     as NumPy's multiply does); a NaN s alone gives quiet(s).
        /// </summary>
        /// <param name="c">The series value.</param><param name="s">The scale.</param><returns>The float16 bits.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ushort HalfScaleF64Bits(float c, double s)
        {
            double c64 = c;
            return DirectILKernelGenerator.DoubleToHalfBits(double.IsNaN(c64) ? c64 : c64 * s);
        }

        /// <summary>One value of <see cref="HalfScaleF64"/> (the scalar tail): the same bits.</summary>
        /// <param name="c">The series value.</param><param name="s">The float64 scale.</param><returns>The scaled value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Half HalfScaleF64Scalar(Half c, double s)
            => BitConverter.UInt16BitsToHalf(HalfScaleF64Bits(DirectILKernelGenerator.HalfToFloatScalarExact(BitConverter.HalfToUInt16Bits(c)), s));

        /// <summary>
        ///     <c>c *= s</c> of 8 float16 lanes in a float32 loop (an int16 / uint16 / char / float32 scale): the float32
        ///     product of the lanes and the EXACT float32 scale (not rounded to float16, as a float16 loop would round it),
        ///     then NumPy's cast back — one RTNE narrow — with a NaN c winning, quieted.
        /// </summary>
        /// <param name="c">8 float16 series lanes as float32.</param><param name="s">The float32 scale in every lane.</param>
        /// <returns>The 8 scaled lanes, on the float16 grid.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<float> HalfScaleF32(Vector256<float> c, Vector256<float> s)
        {
            var r = DirectILKernelGenerator.HalfWiden8V(DirectILKernelGenerator.HalfNarrow8V(Avx.Multiply(c, s)));
            // A widened f16 NaN has its payload in float bits 13..22: | 0x00400000 is the half quiet bit 0x0200.
            return Avx.BlendVariable(r, Avx.Or(c, Vector256.Create(0x00400000).AsSingle()), Avx.CompareUnordered(c, c));
        }

        /// <summary>One value of <see cref="HalfScaleF32"/> (the scalar tail): the same bits.</summary>
        /// <param name="c">The series value.</param><param name="s">The float32 scale.</param><returns>The scaled value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Half HalfScaleF32Scalar(Half c, float s)
        {
            ushort bits = BitConverter.HalfToUInt16Bits(c);
            if ((bits & 0x7fff) > 0x7c00)
                return BitConverter.UInt16BitsToHalf((ushort)(bits | 0x0200));   // quiet(c)
            return BitConverter.UInt16BitsToHalf(DirectILKernelGenerator.SingleToHalfBits(DirectILKernelGenerator.HalfToFloatScalarExact(bits) * s));
        }

        // ---------------------------------------------------------------- reflection handles

        /// <summary>A public static method of this class by name — fails at type load, never at emission.</summary>
        /// <param name="name">The method name (unique in this class).</param><returns>The method.</returns>
        /// <exception cref="MissingMethodException">No such method (a rename broke the handle).</exception>
        private static MethodInfo M(string name) => typeof(PolyLaneOps).GetMethod(name, BindingFlags.Public | BindingFlags.Static)
                                                     ?? throw new MissingMethodException(nameof(PolyLaneOps), name);

        /// <summary>Reflection handles of the primitives above, resolved once at type load (the emitters
        ///     emit direct calls to them; generic ones are closed per element type at emission).</summary>
        internal static readonly MethodInfo s_narrowLoad2 = M(nameof(NarrowLoad2)), s_narrowLoad4 = M(nameof(NarrowLoad4)),
            s_narrowLoad8 = M(nameof(NarrowLoad8)), s_wrap128 = M(nameof(Wrap128)), s_wrap256 = M(nameof(Wrap256)),
            s_packedLoad2 = M(nameof(PackedLoad2)), s_halfLoad2 = M(nameof(HalfLoad2)), s_halfLoad4 = M(nameof(HalfLoad4)),
            s_u32ToF64x4 = M(nameof(U32ToF64x4)), s_u32ToF64x2 = M(nameof(U32ToF64x2)),
            s_f32LowToF64x4 = M(nameof(F32LowToF64x4)), s_f32LowToF64x2 = M(nameof(F32LowToF64x2)),
            s_f64ToC128x2 = M(nameof(F64ToC128x2)), s_cBroadcast = M(nameof(CBroadcast)), s_cMul = M(nameof(CMul)),
            s_cDiv = M(nameof(CDiv)), s_cDivPrep = M(nameof(CDivPrep)), s_cDivBy = M(nameof(CDivBy)),
            s_halfHalveOnGrid = M(nameof(HalfHalveOnGrid)),
            s_halfLoad8 = M(nameof(HalfLoad8)),
            s_f32ScaleF64 = M(nameof(F32ScaleF64)), s_f32ScaleF64Scalar = M(nameof(F32ScaleF64Scalar)),
            s_halfScaleF64 = M(nameof(HalfScaleF64)), s_halfScaleF64Scalar = M(nameof(HalfScaleF64Scalar)),
            s_halfScaleF32 = M(nameof(HalfScaleF32)), s_halfScaleF32Scalar = M(nameof(HalfScaleF32Scalar));

        /// <summary><c>Avx2.Permute4x64(Vector256&lt;double&gt;, byte)</c> — the 64-bit-lane shuffle the gang load
        ///     emits with a constant control.</summary>
        internal static readonly MethodInfo s_permute4x64 = typeof(Avx2).GetMethod(nameof(Avx2.Permute4x64),
            new[] { typeof(Vector256<double>), typeof(byte) }) ?? throw new MissingMethodException(nameof(Avx2), nameof(Avx2.Permute4x64));
    }

    /// <summary>
    ///     bool / int8 / uint8 / int16 / uint16 / char values in a vector chain: one int32 lane per point
    ///     (<c>Vector128&lt;int&gt;</c> for W ≤ 4, <c>Vector256&lt;int&gt;</c> for W = 8), holding the value
    ///     sign- or zero-extended, re-wrapped to the dtype's width after every op (see the file header).
    /// </summary>
    internal sealed class PolyNarrowIntLaneKind : PolyValueKind
    {
        private readonly NPTypeCode _logical;
        private readonly int _w;
        private readonly Type _elem;

        /// <summary>Creates the kind.</summary>
        /// <param name="logical">The narrow dtype.</param><param name="w">Lanes (2, 4 or 8).</param>
        /// <exception cref="NotSupportedException">Another lane count.</exception>
        public PolyNarrowIntLaneKind(NPTypeCode logical, int w) : base(true, NPTypeCode.Int32, w == 8 ? 256 : 128)
        {
            if (w is not (2 or 4 or 8)) throw new NotSupportedException($"no {w}-lane {logical} kind");
            _logical = logical; _w = w; _elem = DirectILKernelGenerator.GetClrType(logical);
        }

        /// <inheritdoc/>
        public override void Bin(ILGenerator il, BinaryOp op, PolyComplexProduct product = PolyComplexProduct.Simd)
        {
            if (op is not (BinaryOp.Add or BinaryOp.Subtract or BinaryOp.Multiply))
                throw new NotSupportedException($"{op} on {_logical} lanes");   // integer true division is float64
            PolyLanes.EmitVecOperator(il, op, LocalType);
            il.EmitCall(OpCodes.Call, (Bits == 256 ? PolyLaneOps.s_wrap256 : PolyLaneOps.s_wrap128).MakeGenericMethod(_elem), null);
        }

        /// <inheritdoc/>
        public override void Load(ILGenerator il)
        {
            var m = _w switch { 2 => PolyLaneOps.s_narrowLoad2, 4 => PolyLaneOps.s_narrowLoad4, _ => PolyLaneOps.s_narrowLoad8 };
            il.EmitCall(OpCodes.Call, m.MakeGenericMethod(_elem), null);
        }

        /// <inheritdoc/>
        /// <remarks>The IL stack already holds the value sign/zero-extended to int32 (ldloc of a narrow local).</remarks>
        public override void BroadcastFromScalar(ILGenerator il) =>
            il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(Bits, typeof(int)), null);

        /// <inheritdoc/>
        /// <exception cref="NotSupportedException">Always: a narrow integer is never a polynomial result.</exception>
        public override void StoreValueFirst(ILGenerator il) => throw new NotSupportedException($"{_logical} lanes are never stored");

        /// <inheritdoc/>
        public override int Lanes => _w;
    }

    /// <summary>float32 / int32 / uint32 values at W = 2 (a complex128 loop): a <c>Vector128&lt;T&gt;</c> with
    ///     lanes 0-1 live, loaded 8 bytes at a time.</summary>
    internal sealed class PolyPackedLaneKind : PolyValueKind
    {
        /// <summary>Creates the kind.</summary>
        /// <param name="t">float32, int32 or uint32.</param>
        public PolyPackedLaneKind(NPTypeCode t) : base(true, t, 128) { }

        /// <inheritdoc/>
        public override void Load(ILGenerator il) =>
            il.EmitCall(OpCodes.Call, PolyLaneOps.s_packedLoad2.MakeGenericMethod(Clr), null);

        /// <inheritdoc/>
        /// <exception cref="NotSupportedException">Always: a W = 2 kernel stores complex128.</exception>
        public override void StoreValueFirst(ILGenerator il) => throw new NotSupportedException($"{T} W=2 lanes are never stored");

        /// <inheritdoc/>
        public override int Lanes => 2;
    }

    /// <summary>complex128 values, two per <c>Vector256&lt;double&gt;</c> as <c>[re0, im0, re1, im1]</c>, with
    ///     NumPy's array multiply and divide (see the file header).</summary>
    internal sealed class PolyComplexLaneKind : PolyValueKind
    {
        /// <summary>Creates the kind (locals are <c>Vector256&lt;double&gt;</c>).</summary>
        public PolyComplexLaneKind() : base(true, NPTypeCode.Double, 256) { }

        /// <inheritdoc/>
        public override void Bin(ILGenerator il, BinaryOp op, PolyComplexProduct product = PolyComplexProduct.Simd)
        {
            switch (op)
            {
                case BinaryOp.Add:
                case BinaryOp.Subtract:
                    PolyLanes.EmitVecOperator(il, op, LocalType);   // component-wise, as NumPy's CDOUBLE add/subtract
                    break;
                case BinaryOp.Multiply: il.EmitCall(OpCodes.Call, PolyLaneOps.s_cMul, null); break;
                case BinaryOp.Divide: il.EmitCall(OpCodes.Call, PolyLaneOps.s_cDiv, null); break;
                default: throw new NotSupportedException($"{op} on complex128 lanes");
            }
        }

        /// <inheritdoc/>
        public override void BroadcastFromScalar(ILGenerator il) => il.EmitCall(OpCodes.Call, PolyLaneOps.s_cBroadcast, null);

        /// <inheritdoc/>
        public override int Lanes => 2;
    }

    /// <summary>Lane-level facts: vector operators, the lane kinds, the exact lane conversions and the vector-block gate.</summary>
    internal static class PolyLanes
    {
        /// <summary>The house exact float16 → float32 widen of 8 lanes (NaN payloads kept).</summary>
        internal static readonly MethodInfo s_halfWiden8V = typeof(DirectILKernelGenerator).GetMethod(nameof(DirectILKernelGenerator.HalfWiden8V),
            BindingFlags.NonPublic | BindingFlags.Static) ?? throw new MissingMethodException(nameof(DirectILKernelGenerator.HalfWiden8V));
        /// <summary>The house RTNE float32 → float16 narrow of 8 lanes.</summary>
        internal static readonly MethodInfo s_halfNarrow8V = typeof(DirectILKernelGenerator).GetMethod(nameof(DirectILKernelGenerator.HalfNarrow8V),
            BindingFlags.NonPublic | BindingFlags.Static) ?? throw new MissingMethodException(nameof(DirectILKernelGenerator.HalfNarrow8V));
        // The exact x86 lane conversions EmitLaneConvert emits (cvtps2pd/cvtdq2pd at 4 and 2 lanes, cvtdq2ps,
        // pmovsxdq at 4 and 2 lanes).
        private static readonly MethodInfo s_cvtps2pd = typeof(Avx).GetMethod(nameof(Avx.ConvertToVector256Double), new[] { typeof(Vector128<float>) });
        private static readonly MethodInfo s_cvtdq2pd = typeof(Avx).GetMethod(nameof(Avx.ConvertToVector256Double), new[] { typeof(Vector128<int>) });
        private static readonly MethodInfo s_cvtps2pd2 = typeof(Sse2).GetMethod(nameof(Sse2.ConvertToVector128Double), new[] { typeof(Vector128<float>) });
        private static readonly MethodInfo s_cvtdq2pd2 = typeof(Sse2).GetMethod(nameof(Sse2.ConvertToVector128Double), new[] { typeof(Vector128<int>) });
        private static readonly MethodInfo s_cvtdq2ps = typeof(Avx).GetMethod(nameof(Avx.ConvertToVector256Single), new[] { typeof(Vector256<int>) });
        private static readonly MethodInfo s_pmovsxdq4 = typeof(Avx2).GetMethod(nameof(Avx2.ConvertToVector256Int64), new[] { typeof(Vector128<int>) });
        private static readonly MethodInfo s_pmovsxdq2 = typeof(Sse41).GetMethod(nameof(Sse41.ConvertToVector128Int64), new[] { typeof(Vector128<int>) });

        /// <summary>
        ///     The host supports the mixed-dtype lane kinds (x86 AVX2): vector chains then run at 256 bits for
        ///     every loop dtype, and every per-point dtype may join them. Elsewhere only same-dtype chains are
        ///     vectorized, at the house width.
        /// </summary>
        public static bool MixedLanes => Avx2.IsSupported;

        /// <summary>[a, b] → [a op b] through the vector type's own operator (exact IEEE lane-wise for floats,
        ///     wrapping for integers — NumPy's loops' semantics).</summary>
        /// <param name="il">The generator.</param>
        /// <param name="op">Add/Subtract/Multiply/Divide.</param>
        /// <param name="vt">The vector type.</param>
        /// <exception cref="NotSupportedException">Another op.</exception>
        /// <exception cref="MissingMethodException">The vector type lacks that operator.</exception>
        public static void EmitVecOperator(ILGenerator il, BinaryOp op, Type vt)
        {
            string name = op switch
            {
                BinaryOp.Add => "op_Addition", BinaryOp.Subtract => "op_Subtraction",
                BinaryOp.Multiply => "op_Multiply", BinaryOp.Divide => "op_Division",
                _ => throw new NotSupportedException(op.ToString()),
            };
            il.EmitCall(OpCodes.Call, vt.GetMethod(name, new[] { vt, vt }) ?? throw new MissingMethodException(vt.Name, name), null);
        }

        /// <summary>Whether <paramref name="t"/> is one of the int32-container dtypes.</summary>
        /// <param name="t">The dtype.</param><returns>True for bool/int8/uint8/int16/uint16/char.</returns>
        private static bool IsNarrowInt(NPTypeCode t) => t is NPTypeCode.Boolean or NPTypeCode.SByte or NPTypeCode.Byte
            or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char;

        /// <summary>Lanes of the loop dtype in vector chains: 256 bits on a mixed-lane host (complex128 2,
        ///     float16 8 — as float32 lanes), the house width for float32/float64 elsewhere.</summary>
        /// <param name="tl">Loop dtype.</param><returns>Lanes.</returns>
        public static int LoopLanes(NPTypeCode tl) => tl switch
        {
            NPTypeCode.Half => 8,
            NPTypeCode.Complex => 2,
            _ => MixedLanes ? 32 / DirectILKernelGenerator.GetTypeSize(tl) : DirectILKernelGenerator.GetVectorCount(tl),
        };

        /// <summary>Whether values of <paramref name="t"/> can live at <paramref name="w"/> lanes (see the file
        ///     header's table). Hosts without mixed lanes only have the generic same-width vectors.</summary>
        /// <param name="t">The dtype.</param><param name="w">Lanes.</param><returns>True when a lane kind exists.</returns>
        public static bool HasLaneKind(NPTypeCode t, int w)
        {
            if (!MixedLanes)
                return t is NPTypeCode.Single or NPTypeCode.Double && w * DirectILKernelGenerator.GetTypeSize(t) * 8 is 128 or 256 or 512;
            return t switch
            {
                NPTypeCode.Complex => w == 2 && Fma.IsSupported,
                NPTypeCode.Double or NPTypeCode.Int64 or NPTypeCode.UInt64 => w is 2 or 4,
                NPTypeCode.Single => w is 2 or 4 or 8,
                NPTypeCode.Int32 or NPTypeCode.UInt32 => w is 2 or 4,
                NPTypeCode.Half => w is 2 or 4 or 8,
                _ => IsNarrowInt(t) && w is 2 or 4 or 8,
            };
        }

        /// <summary>Creates the lane kind of <paramref name="t"/> at <paramref name="w"/> lanes.</summary>
        /// <param name="t">The dtype.</param><param name="w">Lanes.</param><returns>The kind.</returns>
        /// <exception cref="NotSupportedException">No kind (<see cref="HasLaneKind"/> is false).</exception>
        public static PolyValueKind CreateLaneKind(NPTypeCode t, int w)
        {
            if (!HasLaneKind(t, w)) throw new NotSupportedException($"no {w}-lane {t} values");
            if (t == NPTypeCode.Half) return new PolyHalfLaneKind(w);
            if (t == NPTypeCode.Complex) return new PolyComplexLaneKind();
            if (IsNarrowInt(t)) return new PolyNarrowIntLaneKind(t, w);
            if (w == 2 && DirectILKernelGenerator.GetTypeSize(t) == 4) return new PolyPackedLaneKind(t);
            return new PolyValueKind(true, t, w * 8 * DirectILKernelGenerator.GetTypeSize(t));
        }

        /// <summary>
        ///     Whether a per-point value of <paramref name="from"/> converts lane-for-lane into
        ///     <paramref name="to"/> at <paramref name="w"/> lanes — the promotions a polynomial step can
        ///     perform: every real dtype into float64 and complex128, the dtypes float32 holds exactly into
        ///     float32, int8/uint8/bool into float16, and bool into int64 (a bool x times a Python int).
        ///     Every listed conversion is exact, as NumPy's loop-input cast is for these pairs.
        /// </summary>
        /// <param name="from">Per-point dtype.</param><param name="to">Target dtype.</param><param name="w">Lanes.</param>
        /// <returns>True when <see cref="EmitLaneConvert"/> handles the pair.</returns>
        public static bool LaneConvertible(NPTypeCode from, NPTypeCode to, int w)
        {
            if (from == to) return HasLaneKind(from, w);
            if (!MixedLanes || !HasLaneKind(from, w) || !HasLaneKind(to, w)) return false;
            return to switch
            {
                NPTypeCode.Complex => from != NPTypeCode.Decimal && LaneConvertible(from, NPTypeCode.Double, w),
                NPTypeCode.Double => from is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Int32 or NPTypeCode.UInt32
                                     or NPTypeCode.Int64 or NPTypeCode.UInt64 || IsNarrowInt(from),
                NPTypeCode.Single => w == 8 && (from == NPTypeCode.Half || IsNarrowInt(from)),
                NPTypeCode.Half => w == 8 && from is NPTypeCode.Boolean or NPTypeCode.SByte or NPTypeCode.Byte,
                NPTypeCode.Int64 => from == NPTypeCode.Boolean,
                _ => false,
            };
        }

        /// <summary>[lane value of from] → [lane value of to] (see <see cref="LaneConvertible"/>).</summary>
        /// <param name="il">Generator.</param><param name="from">Source dtype.</param><param name="to">Target dtype.</param>
        /// <param name="w">Lanes.</param>
        /// <exception cref="NotSupportedException">A pair the gate rejects (the gate and the emitter disagree).</exception>
        public static void EmitLaneConvert(ILGenerator il, NPTypeCode from, NPTypeCode to, int w)
        {
            if (from == to) return;
            if (!LaneConvertible(from, to, w)) throw new NotSupportedException($"no {w}-lane conversion {from} -> {to}");
            if (to == NPTypeCode.Complex)
            {
                EmitLaneConvert(il, from, NPTypeCode.Double, w);
                il.EmitCall(OpCodes.Call, PolyLaneOps.s_f64ToC128x2, null);
                return;
            }
            MethodInfo m = (from, to) switch
            {
                // float16 lanes already are exact float32 values; the int32 containers convert exactly
                // (|v| < 2^16 is a float32, and an int8/uint8/bool value is a float16 too).
                (NPTypeCode.Half, NPTypeCode.Single) => null,
                (_, NPTypeCode.Single) or (_, NPTypeCode.Half) => s_cvtdq2ps,
                (NPTypeCode.Half, NPTypeCode.Double) => w == 4 ? PolyLaneOps.s_f32LowToF64x4 : PolyLaneOps.s_f32LowToF64x2,
                (NPTypeCode.Single, NPTypeCode.Double) => w == 4 ? s_cvtps2pd : s_cvtps2pd2,
                (NPTypeCode.UInt32, NPTypeCode.Double) => w == 4 ? PolyLaneOps.s_u32ToF64x4 : PolyLaneOps.s_u32ToF64x2,
                (NPTypeCode.Int64 or NPTypeCode.UInt64, NPTypeCode.Double) =>
                    (w == 4 ? typeof(Vector256) : typeof(Vector128)).GetMethod(nameof(Vector256.ConvertToDouble),
                        new[] { (w == 4 ? typeof(Vector256<>) : typeof(Vector128<>)).MakeGenericType(DirectILKernelGenerator.GetClrType(from)) }),
                (_, NPTypeCode.Double) => w == 4 ? s_cvtdq2pd : s_cvtdq2pd2,   // int32 and the int32 containers
                (NPTypeCode.Boolean, NPTypeCode.Int64) => w == 4 ? s_pmovsxdq4 : s_pmovsxdq2,
                _ => throw new NotSupportedException($"no {w}-lane conversion {from} -> {to}"),
            };
            if (m is not null) il.EmitCall(OpCodes.Call, m, null);
        }

        /// <summary>
        ///     Whether a block may run vector chains: the loop dtype has a lane kind (float16/float32/float64,
        ///     and complex128 on FMA hosts), and every PER-POINT source — x when it is per point, the
        ///     coefficients when every point reads its own series — has one and converts into the loop dtype.
        ///     Shared values never limit the gate: they are scalars in their own dtype, converted and broadcast
        ///     where they meet a per-point value. The values derived from the sources (x's dtype for
        ///     <c>2*x</c>, int64 for a bool x's Python-int products, the peeled steps' coefficient dtype) are
        ///     covered by the same table; a combination this gate missed would fail the kernel's vector
        ///     emission, which then falls back to scalar chains (<see cref="ILKernelGenerator"/>'s compile).
        /// </summary>
        /// <param name="tx">x dtype.</param><param name="xPerPoint">x is a per-point operand.</param>
        /// <param name="tc">Coefficient dtype.</param><param name="perLaneCoef">Every point reads its own series.</param>
        /// <param name="tl">Loop (result) dtype.</param>
        /// <returns>True when vector chains apply.</returns>
        public static bool VectorBlockOk(NPTypeCode tx, bool xPerPoint, NPTypeCode tc, bool perLaneCoef, NPTypeCode tl)
        {
            if (!MixedLanes)
                return tl is NPTypeCode.Single or NPTypeCode.Double && DirectILKernelGenerator.CanUseSimd(tl)
                       && (!xPerPoint || tx == tl) && (!perLaneCoef || tc == tl);
            if (tl is not (NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex)) return false;
            int w = LoopLanes(tl);
            if (!HasLaneKind(tl, w)) return false;
            return (!xPerPoint || LaneConvertible(tx, tl, w)) && (!perLaneCoef || LaneConvertible(tc, tl, w));
        }
    }
}
