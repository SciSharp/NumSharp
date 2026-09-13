using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Fused single-pass "are every corresponding element equal?" reduction — the hot-path engine
    ///     behind <see cref="np.array_equal(NDArray,NDArray,bool)"/> and
    ///     <see cref="np.array_equiv(NDArray,NDArray)"/> for the common EXACT-SHAPE, SAME-DTYPE,
    ///     dense-contiguous case.
    ///
    ///     <para>The library default computes all-equal as <c>np.all(a == b)</c> — a full boolean temp
    ///     (size × 1 byte) written by the comparison kernel and then re-read by the reduction. This class
    ///     fuses both into ONE streaming pass with NO intermediate allocation, and — unlike either NumPy
    ///     or the composition — <b>early-exits at the first differing vector</b>, so a mismatch near the
    ///     front returns almost instantly instead of scanning the whole array. Sibling of
    ///     <see cref="FiniteScan"/>; same design (generic <see cref="Vector{T}"/> body, JIT bakes the
    ///     V128/V256/V512 width per instantiation) and the same <c>Address + offset·itemsize</c> base rule
    ///     for contiguous slices.</para>
    ///
    ///     <para><b>Value vs byte comparison — this is load-bearing.</b> For the integer family (bool /
    ///     signed+unsigned integers / char) value-equality IS byte-equality, so the whole buffer is scanned
    ///     as raw bytes through one <see cref="Vector{Byte}"/> path (32 bytes/vector on AVX2 = the same
    ///     memory throughput a per-dtype vector would give). For <b>float / double / complex</b> byte
    ///     comparison would be WRONG twice over — two NaNs share a bit pattern yet must compare UNEQUAL
    ///     (NumPy's default <c>equal_nan=False</c>: NaN ≠ NaN), and <c>-0.0</c> vs <c>+0.0</c> differ in
    ///     bits yet must compare EQUAL — so those dtypes use IEEE value comparison
    ///     (<see cref="Vector.EqualsAll{T}(Vector{T},Vector{T})"/>, which yields false for a NaN lane and
    ///     true for signed-zero), complex128 by reinterpreting each element as two interleaved doubles.</para>
    ///
    ///     <para><b>Not handled here (the caller falls back to the <c>np.all(a == b)</c> composition):</b>
    ///     differing shapes (broadcasting), mixed dtypes, non-dense or mixed C/F layouts, <see cref="Half"/>
    ///     (no <c>Vector&lt;Half&gt;</c> arithmetic, and a raw-bits scan would mis-handle NaN), and
    ///     <see cref="decimal"/> (no <c>Vector&lt;decimal&gt;</c>). Those are the rarer cases; the
    ///     composition already computes them correctly.</para>
    /// </summary>
    internal static unsafe class EqualityScan
    {
        /// <summary>
        ///     Attempts the fused all-equal fast path. Returns true (with <paramref name="allEqual"/> set)
        ///     only when both operands are eligible: identical shape, identical dtype, both dense in the
        ///     SAME contiguity (both C-contiguous or both F-contiguous, so a single linear walk compares
        ///     corresponding elements), and a dtype the fused scan supports. Returns false — leaving
        ///     <paramref name="allEqual"/> unspecified — for every other case, signalling the caller to use
        ///     the <c>np.all(a == b)</c> composition (which handles broadcasting, mixed dtypes, strided
        ///     views, Half and Decimal correctly).
        /// </summary>
        /// <param name="a">First operand (non-null).</param>
        /// <param name="b">Second operand (non-null).</param>
        /// <param name="allEqual">Set to the all-equal result when the method returns true.</param>
        /// <returns>True if the fast path handled the comparison; false to fall back to the composition.</returns>
        internal static bool TryAllEqual(NDArray a, NDArray b, out bool allEqual)
        {
            allEqual = false;

            // Same dtype only. Mixed dtypes need the comparison kernel's per-element NEP50 promotion
            // (e.g. int64 vs float64 past 2^53), which a same-width byte/value scan cannot reproduce.
            if (a.typecode != b.typecode)
                return false;

            // Same shape only — this fast path never broadcasts. array_equiv routes a genuine broadcast
            // (different shapes) to the composition; array_equal already rejected differing shapes.
            var sa = a.Shape;
            var sb = b.Shape;
            if (!SameDimensions(sa, sb))
                return false;

            // Both C-contiguous OR both F-contiguous: only then does one linear pass over each buffer
            // line up element-for-element. A C operand against an F operand of the same shape would
            // compare transposed positions, so that mix falls back.
            bool bothC = sa.IsContiguous && sb.IsContiguous;
            bool bothF = !bothC && sa.IsFContiguous && sb.IsFContiguous;
            if (!bothC && !bothF)
                return false;

            // Half needs value semantics with no Vector<Half>; Decimal has no Vector<decimal>. Both are
            // rare for an equality test — hand them to the composition rather than grow a scalar branch here.
            if (a.typecode == NPTypeCode.Half || a.typecode == NPTypeCode.Decimal)
                return false;

            allEqual = AreAllEqualDense(a, b);
            return true;
        }

        /// <summary>
        ///     Runs the fused scan on two eligibility-checked operands (same shape, same dtype, dense in a
        ///     shared contiguity, not Half/Decimal). Dispatches float/double/complex to the IEEE value scan
        ///     and every integer-family dtype to the raw-byte scan.
        /// </summary>
        private static bool AreAllEqualDense(NDArray a, NDArray b)
        {
            long n = a.size;
            if (n == 0)
                return true; // all() of an empty comparison is True (vacuous), matching array_equal([], []).

            int isz = a.dtypesize;
            // Logical element zero of a contiguous slice lives at Address + offset·itemsize (the documented
            // base rule; a contiguous slice re-seats Address and keeps a non-zero Shape.offset).
            byte* pa = (byte*)a.Address + sa_offset(a) * (long)isz;
            byte* pb = (byte*)b.Address + sa_offset(b) * (long)isz;

            switch (a.typecode)
            {
                case NPTypeCode.Single:
                    return AllEqualValue((float*)pa, (float*)pb, n);
                case NPTypeCode.Double:
                    return AllEqualValue((double*)pa, (double*)pb, n);
                case NPTypeCode.Complex:
                    // A contiguous Complex[n] is 2·n interleaved doubles (re, im); per-component IEEE
                    // equality gives NumPy's complex == (NaN component ⇒ unequal, ±0 components equal).
                    return AllEqualValue((double*)pa, (double*)pb, n * 2);
                default:
                    // Integer family + bool + char: value-equality == byte-equality, so scan the whole
                    // buffer (n·itemsize bytes) as one Vector<byte> stream.
                    return AllEqualValue(pa, pb, n * (long)isz);
            }
        }

        // Shape.offset is internal to the View layer; read it through the public accessor path the rest of
        // the kernels use. Kept as a tiny helper so the two call sites stay readable.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long sa_offset(NDArray a) => a.Shape.offset;

        /// <summary>
        ///     Contiguous IEEE/value equality scan, 4× unrolled with a single-vector remainder and a scalar
        ///     tail. Generic over <typeparamref name="T"/> so one body serves every element width — the JIT
        ///     bakes the vector width per instantiation. <see cref="Vector.EqualsAll{T}(Vector{T},Vector{T})"/>
        ///     is per-lane <c>==</c>: for float/double a NaN lane yields false (so NaN ≠ NaN, NumPy's
        ///     default) and a <c>-0.0</c>/<c>+0.0</c> pair yields true; for the byte scan it is plain
        ///     equality. The per-group early-exit returns at the first differing 4-vector block.
        /// </summary>
        /// <typeparam name="T">Element type (byte for the integer family, float/double for the value scans).</typeparam>
        /// <param name="a">First operand's logical element zero.</param>
        /// <param name="b">Second operand's logical element zero.</param>
        /// <param name="n">Element count to compare (bytes for the integer-family byte scan).</param>
        /// <returns>True iff every element compares equal.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static bool AllEqualValue<T>(T* a, T* b, long n) where T : unmanaged, INumber<T>
        {
            long i = 0;

            if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
            {
                int w = Vector<T>.Count;
                long step = (long)w * 4;
                if (n >= step)
                {
                    long end = n - step;
                    for (; i <= end; i += step)
                    {
                        // Load 4 independent vector pairs, reduce to 4 all-lanes-equal flags, and branch
                        // once. The pairs are independent so the CPU pipelines the compares; combining the
                        // flags with & keeps a single (well-predicted, always-taken while equal) branch.
                        bool e0 = Vector.EqualsAll(Vector.Load(a + i), Vector.Load(b + i));
                        bool e1 = Vector.EqualsAll(Vector.Load(a + i + w), Vector.Load(b + i + w));
                        bool e2 = Vector.EqualsAll(Vector.Load(a + i + 2 * w), Vector.Load(b + i + 2 * w));
                        bool e3 = Vector.EqualsAll(Vector.Load(a + i + 3 * w), Vector.Load(b + i + 3 * w));
                        if (!(e0 & e1 & e2 & e3))
                            return false; // some lane in this block differs → not all equal
                    }
                }

                // One-vector remainder (still SIMD) before the scalar tail.
                for (; i <= n - w; i += w)
                {
                    if (!Vector.EqualsAll(Vector.Load(a + i), Vector.Load(b + i)))
                        return false;
                }
            }

            for (; i < n; i++)
            {
                // T's own != — IEEE for float/double (NaN != NaN is true → returns false; -0.0 != 0.0 is
                // false → continues), plain value inequality for the byte scan.
                if (a[i] != b[i])
                    return false;
            }
            return true;
        }

        /// <summary>
        ///     Dimension-array equality (same rank and same extents). Cheaper and clearer than building a
        ///     Shape comparison; strides/flags are irrelevant here because eligibility already pins the
        ///     contiguity.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool SameDimensions(Shape a, Shape b)
        {
            if (a.NDim != b.NDim)
                return false;
            var da = a.dimensions;
            var db = b.dimensions;
            for (int d = 0; d < da.Length; d++)
                if (da[d] != db[d])
                    return false;
            return true;
        }
    }
}
