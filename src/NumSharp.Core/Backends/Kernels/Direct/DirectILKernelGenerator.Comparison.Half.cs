using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.Comparison.Half.cs — float16 comparisons
// (==, !=, <, <=, >, >=) — bit-level, conversion-free
// =============================================================================
//
// f16 comparison is pure bit logic: NumPy's HALF loops call the NaN-GUARDED
// npy_half comparators (halffloat.cpp: npy_half_lt → Half::operator<, which is
// `!(isnan || isnan) && raw sign-magnitude compare`), all scalar C — measured
// 414 µs per 100K `less` on 2.4.2. Over the proven order map
//     key = bits ^ (0x8000 | (bits >>ARITH 15))
// every comparison reduces to mask algebra on ushort lanes (probed 2.4.2:
// less(-0,+0)=False, equal(-0,+0)=True, le/ge(-0,+0)=True, NaN compares False
// except != which is True):
//
//     eq = (bitsEqual | bothZero) & ~anyNaN          ne = ~eq
//     le = (leKeys | bothZero)   & ~anyNaN           gt = ~(leKeys | bothZero) & ~anyNaN
//     ge = (geKeys | bothZero)   & ~anyNaN           lt = ~(geKeys | bothZero) & ~anyNaN
//
// with geKeys/leKeys via pmaxuw/pminuw + compare-equal, and bothZero repairing
// the -0/+0 seam where the key order disagrees with IEEE equality. Two 16-lane
// mask vectors pack to 32 canonical 0/1 bool bytes per store (PackSignedSaturate
// + Permute4x64(0xD8) + And 0x01). Serves the (Half,Half) comparison keys for
// SimdFull and both scalar-broadcast paths (the scalar operand's key/NaN masks
// hoist); SimdChunk/General keep the emitted scalar IL (Half operators — same
// semantics, per-element).
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// Returns the static bit-level kernel for a (Half,Half) comparison key, or null
        /// when the path is not served (falls to emitted IL). The lambdas close over the
        /// op; the comparison cache stores one delegate per key, so the capture is one-time.
        /// </summary>
        internal static unsafe ComparisonKernel TryGetHalfComparisonKernel(ComparisonOp op, ExecutionPath path)
        {
            switch (path)
            {
                case ExecutionPath.SimdFull:
                    return (l, r, res, ls, rs, sh, nd, n) =>
                        HalfCompareFull((ushort*)l, (ushort*)r, (byte*)res, n, op);
                case ExecutionPath.SimdScalarRight:
                    return (l, r, res, ls, rs, sh, nd, n) =>
                        HalfCompareScalarRight((ushort*)l, *(ushort*)r, (byte*)res, n, op);
                case ExecutionPath.SimdScalarLeft:
                    return (l, r, res, ls, rs, sh, nd, n) =>
                        HalfCompareScalarLeft(*(ushort*)l, (ushort*)r, (byte*)res, n, op);
            }
            return null;
        }

        /// <summary>Per-lane comparison mask from the shared primitives (all loop-invariant branches).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<short> HalfCompareMask(
            Vector256<short> bitsEq, Vector256<short> bothZero, Vector256<short> anyNaN,
            Vector256<ushort> ka, Vector256<ushort> kb, ComparisonOp op)
        {
            switch (op)
            {
                case ComparisonOp.Equal:
                    return Avx2.AndNot(anyNaN, Avx2.Or(bitsEq, bothZero));
                case ComparisonOp.NotEqual:
                    // ~eq — NaN anywhere → true
                    return Avx2.Or(anyNaN, Avx2.AndNot(Avx2.Or(bitsEq, bothZero), Vector256<short>.AllBitsSet));
                case ComparisonOp.LessEqual:
                {
                    var leKeys = Avx2.CompareEqual(Avx2.Min(ka, kb), ka).AsInt16();
                    return Avx2.AndNot(anyNaN, Avx2.Or(leKeys, bothZero));
                }
                case ComparisonOp.Greater:
                {
                    var leKeys = Avx2.CompareEqual(Avx2.Min(ka, kb), ka).AsInt16();
                    return Avx2.AndNot(anyNaN, Avx2.AndNot(Avx2.Or(leKeys, bothZero), Vector256<short>.AllBitsSet));
                }
                case ComparisonOp.GreaterEqual:
                {
                    var geKeys = Avx2.CompareEqual(Avx2.Max(ka, kb), ka).AsInt16();
                    return Avx2.AndNot(anyNaN, Avx2.Or(geKeys, bothZero));
                }
                default: // ComparisonOp.Less
                {
                    var geKeys = Avx2.CompareEqual(Avx2.Max(ka, kb), ka).AsInt16();
                    return Avx2.AndNot(anyNaN, Avx2.AndNot(Avx2.Or(geKeys, bothZero), Vector256<short>.AllBitsSet));
                }
            }
        }

        /// <summary>Pack two 16-lane short masks into 32 canonical 0/1 bool bytes.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StoreHalfCompareMasks(byte* pr, Vector256<short> m0, Vector256<short> m1)
        {
            var packed = Avx2.Permute4x64(Avx2.PackSignedSaturate(m0, m1).AsInt64(), 0xD8).AsByte();
            Avx.Store(pr, Avx2.And(packed, Vector256.Create((byte)1)));
        }

        /// <summary>Both operands streamed (SimdFull).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfCompareFull(ushort* pa, ushort* pb, byte* pr, long n, ComparisonOp op)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m7fff = Vector256.Create((short)0x7FFF);
                var vinf = Vector256.Create((short)0x7C00);
                var m8000 = Vector256.Create(unchecked((short)0x8000));
                for (; i + 32 <= n; i += 32)
                {
                    var a0 = Avx.LoadVector256((short*)(pa + i));
                    var b0 = Avx.LoadVector256((short*)(pb + i));
                    var a1 = Avx.LoadVector256((short*)(pa + i + 16));
                    var b1 = Avx.LoadVector256((short*)(pb + i + 16));
                    var m0 = HalfCompareLane(a0, b0, m7fff, vinf, m8000, op);
                    var m1 = HalfCompareLane(a1, b1, m7fff, vinf, m8000, op);
                    StoreHalfCompareMasks(pr + i, m0, m1);
                }
            }
            for (; i < n; i++)
                pr[i] = HalfCompareBits(pa[i], pb[i], op);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<short> HalfCompareLane(
            Vector256<short> a, Vector256<short> b,
            Vector256<short> m7fff, Vector256<short> vinf, Vector256<short> m8000, ComparisonOp op)
        {
            var magA = Avx2.And(a, m7fff);
            var magB = Avx2.And(b, m7fff);
            var anyNaN = Avx2.Or(Avx2.CompareGreaterThan(magA, vinf), Avx2.CompareGreaterThan(magB, vinf));
            var bitsEq = Avx2.CompareEqual(a, b);
            var bothZero = Avx2.CompareEqual(Avx2.Or(magA, magB), Vector256<short>.Zero);
            var ka = HalfKeyV(a, m8000);
            var kb = HalfKeyV(b, m8000);
            return HalfCompareMask(bitsEq, bothZero, anyNaN, ka, kb, op);
        }

        /// <summary>lhs streamed, rhs one broadcast scalar (in2) — its masks hoist.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfCompareScalarRight(ushort* pa, ushort b, byte* pr, long n, ComparisonOp op)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m7fff = Vector256.Create((short)0x7FFF);
                var vinf = Vector256.Create((short)0x7C00);
                var m8000 = Vector256.Create(unchecked((short)0x8000));
                var vb = Vector256.Create(unchecked((short)b));
                var magB = Vector256.Create((short)(b & 0x7FFF));
                var nanB = (b & 0x7FFF) > 0x7C00 ? Vector256<short>.AllBitsSet : Vector256<short>.Zero;
                var kb = Vector256.Create(HalfKeyScalar(b));
                for (; i + 32 <= n; i += 32)
                {
                    var m0 = LaneVsScalar(Avx.LoadVector256((short*)(pa + i)), vb, magB, nanB, kb, m7fff, vinf, m8000, op, scalarIsRhs: true);
                    var m1 = LaneVsScalar(Avx.LoadVector256((short*)(pa + i + 16)), vb, magB, nanB, kb, m7fff, vinf, m8000, op, scalarIsRhs: true);
                    StoreHalfCompareMasks(pr + i, m0, m1);
                }
            }
            for (; i < n; i++)
                pr[i] = HalfCompareBits(pa[i], b, op);
        }

        /// <summary>lhs one broadcast scalar (in1), rhs streamed.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfCompareScalarLeft(ushort a, ushort* pb, byte* pr, long n, ComparisonOp op)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m7fff = Vector256.Create((short)0x7FFF);
                var vinf = Vector256.Create((short)0x7C00);
                var m8000 = Vector256.Create(unchecked((short)0x8000));
                var va = Vector256.Create(unchecked((short)a));
                var magA = Vector256.Create((short)(a & 0x7FFF));
                var nanA = (a & 0x7FFF) > 0x7C00 ? Vector256<short>.AllBitsSet : Vector256<short>.Zero;
                var ka = Vector256.Create(HalfKeyScalar(a));
                for (; i + 32 <= n; i += 32)
                {
                    var m0 = LaneVsScalar(Avx.LoadVector256((short*)(pb + i)), va, magA, nanA, ka, m7fff, vinf, m8000, op, scalarIsRhs: false);
                    var m1 = LaneVsScalar(Avx.LoadVector256((short*)(pb + i + 16)), va, magA, nanA, ka, m7fff, vinf, m8000, op, scalarIsRhs: false);
                    StoreHalfCompareMasks(pr + i, m0, m1);
                }
            }
            for (; i < n; i++)
                pr[i] = HalfCompareBits(a, pb[i], op);
        }

        /// <summary>
        /// One streamed lane vector vs the hoisted scalar-side masks. <paramref name="scalarIsRhs"/>
        /// keeps operand ROLES straight (Less/Greater are not symmetric).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<short> LaneVsScalar(
            Vector256<short> v, Vector256<short> vs,
            Vector256<short> magS, Vector256<short> nanS, Vector256<ushort> ks,
            Vector256<short> m7fff, Vector256<short> vinf, Vector256<short> m8000,
            ComparisonOp op, bool scalarIsRhs)
        {
            var magV = Avx2.And(v, m7fff);
            var anyNaN = Avx2.Or(Avx2.CompareGreaterThan(magV, vinf), nanS);
            var bitsEq = Avx2.CompareEqual(v, vs);
            var bothZero = Avx2.CompareEqual(Avx2.Or(magV, magS), Vector256<short>.Zero);
            var kv = HalfKeyV(v, m8000);
            return scalarIsRhs
                ? HalfCompareMask(bitsEq, bothZero, anyNaN, kv, ks, op)
                : HalfCompareMask(bitsEq, bothZero, anyNaN, ks, kv, op);
        }

        /// <summary>Scalar lane comparison — the exact NumPy npy_half comparator on bits (a is in1).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static byte HalfCompareBits(ushort a, ushort b, ComparisonOp op)
        {
            bool anyNaN = (a & 0x7FFF) > 0x7C00 || (b & 0x7FFF) > 0x7C00;
            bool bothZero = ((a | b) & 0x7FFF) == 0;
            switch (op)
            {
                case ComparisonOp.Equal:
                    return (byte)(!anyNaN && (a == b || bothZero) ? 1 : 0);
                case ComparisonOp.NotEqual:
                    return (byte)(anyNaN || !(a == b || bothZero) ? 1 : 0);
            }
            if (anyNaN) return 0;
            ushort ka = HalfKeyScalar(a), kb = HalfKeyScalar(b);
            switch (op)
            {
                case ComparisonOp.Less: return (byte)(!bothZero && ka < kb ? 1 : 0);
                case ComparisonOp.LessEqual: return (byte)(bothZero || ka <= kb ? 1 : 0);
                case ComparisonOp.Greater: return (byte)(!bothZero && ka > kb ? 1 : 0);
                default: return (byte)(bothZero || ka >= kb ? 1 : 0); // GreaterEqual
            }
        }
    }
}
