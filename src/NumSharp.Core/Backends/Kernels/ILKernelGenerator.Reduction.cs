using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Iteration;

// =============================================================================
// ILKernelGenerator.Reduction.cs — per-chunk axis-reduction kernels (NDIter-driven)
// =============================================================================
//
// MODEL
// -----
// These kernels implement the inner loop of a 2-operand REDUCE iterator built by
// NDIterRef.NewReduce ([input, output], REDUCE_OK | EXTERNAL_LOOP). The iterator
// owns the outer walk and advances the operand pointers between calls; each kernel
// invocation folds ONE contiguous inner stripe into the output:
//
//     do { inner(dataptrs, strides, count, aux); } while (iternext(iter));
//
// Two output modes per call, distinguished by the output byte stride:
//   • outStride == 0  → PINNED: the reduced axis is innermost. Fold the whole
//     `count`-element stripe into the single output slot (read it, accumulate,
//     write back once).
//   • outStride != 0  → SLAB: a kept axis is innermost. Each inner element targets
//     a distinct output slot; fold `in[c]` into `out[c]`. The same slab is
//     revisited across outer iterations, so the output MUST be pre-seeded with the
//     reduction identity (DefaultEngine seeds it via SeedReduceIdentity).
//
// COMPLEX (Phase 1)
// -----------------
// System.Numerics.Complex is two contiguous doubles [real, imaginary]. A complex
// Sum slab-fold is therefore a plain f64 elementwise add over 2·count lanes, and a
// complex Sum pinned-reduce is a 2-lane (re,im) f64 accumulator — both ride the
// Vector256<double> machinery. Prod uses scalar complex multiply (no SIMD form);
// Min/Max use lexicographic (Real, Imaginary) pick with NaN-first-wins, matching
// DirectILKernelGenerator's existing ComplexLexPick semantics exactly.
//
// This is the migration TARGET model (see ILKernelGenerator.cs). The legacy
// whole-array complex path lives in DirectILKernelGenerator.Reduction.Axis.cs and
// is retired per-dtype as families move here.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        /// <summary>
        /// Cache key for a per-chunk reduction kernel: operation + input dtype +
        /// accumulator (output) dtype. Layout is handled at runtime inside the
        /// kernel body (pinned vs slab, contiguous vs strided inner loop), so it is
        /// NOT part of the key.
        /// </summary>
        public readonly record struct ReduceKernelKey(ReductionOp Op, NPTypeCode InType, NPTypeCode AccType);

        internal static readonly ConcurrentDictionary<ReduceKernelKey, NDInnerLoopFunc> _reduceCache = new();

        /// <summary>
        /// Returns the cached per-chunk reduction kernel for the given
        /// (op, input, accumulator) triple, or null when no NDIter-driven kernel
        /// exists yet (caller falls back to the DirectILKernelGenerator path).
        /// The returned delegate matches <see cref="NDInnerLoopFunc"/>; hand it to
        /// an iterator built by <see cref="NDIterRef.NewReduce(NDArray, NDArray, int, NDIterGlobalFlags)"/>.
        /// </summary>
        public static NDInnerLoopFunc GetReduceInnerLoop(ReduceKernelKey key) =>
            _reduceCache.GetOrAdd(key, CreateReduceInnerLoop);

        private static unsafe NDInnerLoopFunc CreateReduceInnerLoop(ReduceKernelKey key)
        {
            // Complex: dedicated double-pair kernels (SIMD sum, lex min/max).
            if (key.InType == NPTypeCode.Complex && key.AccType == NPTypeCode.Complex)
            {
                // Sum: prefer the IL-emitted Vector128 pairwise kernel (bit-exact NumPy
                // complex128; the hand-written ComplexSumKernel is a flat accumulator that
                // diverges in the low bits). Fall back to it only when emission is unavailable.
                if (key.Op == ReductionOp.Sum)
                {
                    var emitted = TryEmitPairwiseSumKernel(NPTypeCode.Complex);
                    if (emitted != null) return emitted;
                }
                return key.Op switch
                {
                    ReductionOp.Sum  => ComplexSumKernel,
                    ReductionOp.Prod => ComplexProdKernel,
                    ReductionOp.Min  => ComplexMinKernel,
                    ReductionOp.Max  => ComplexMaxKernel,
                    _ => null
                };
            }

            // Half mean / Decimal: generic INumber scalar kernels fed CONTIGUOUS stripes by the
            // per-chunk iterator (no cache-hostile column gather like the legacy coordinate walk).
            //   - Half MEAN accumulates in Double (input Half → Double); ReduceMean casts back to
            //     Half. (Half sum/prod/min/max stay on the Direct path — see UseNDIterReduce:
            //     their Half-accumulator serial chain can't beat it on the pinned axis.)
            //   - Decimal accumulates in full-precision Decimal (no NumPy reference type).
            //     CreateTypedReduceKernel can build a same-type Half kernel too; it just isn't
            //     routed here today.
            if (key.InType == NPTypeCode.Half && key.AccType == NPTypeCode.Double)
                return key.Op == ReductionOp.Sum ? CreateTypedReduceKernel<Half, double>(ReductionOp.Sum) : null;
            if (key.InType == NPTypeCode.Decimal && key.AccType == NPTypeCode.Decimal)
                return CreateTypedReduceKernel<decimal, decimal>(key.Op);

            // Phase 6 — numeric same-type Sum/Mean (PairwiseFold, bit-for-bit NumPy
            // pairwise_sum) AND Min/Max (SimdMinMaxSameType, NaN-propagating). Double AND
            // float32 route here so f64/f32 reductions ride the stride-ordered NDIter reduce
            // path — SIMD on EVERY layout (transpose/F/strided/negcol/broadcast), fixing the
            // C-contiguity-gated Direct axis kernel's collapse on those layouts. Integer
            // widening sums and Prod stay on the Direct path (CreateSimdReduceKernel returns
            // null for those → caller falls back).
            if (key.InType == key.AccType && (key.InType == NPTypeCode.Double || key.InType == NPTypeCode.Single))
            {
                // Sum: prefer the IL-EMITTED SIMD pairwise kernel — bit-for-bit identical to
                // NumPy's pairwise_sum yet width-native SIMD (the scalar PairwiseFold below
                // forfeits vectorization; see ILKernelGenerator.Reduction.Pairwise.cs). Falls
                // back to the generic scalar fold only if emission is unavailable (IL disabled).
                if (key.Op == ReductionOp.Sum)
                {
                    var emitted = TryEmitPairwiseSumKernel(key.InType);
                    if (emitted != null) return emitted;
                }
                // Min/Max: width-native SIMD with explicit per-lane NaN propagation
                // (Vector256.Min/Max alone do not propagate NaN). Same-type, no widening.
                else if (key.Op == ReductionOp.Min)
                    return key.InType == NPTypeCode.Double ? (NDInnerLoopFunc)SimdMinKernel<double> : SimdMinKernel<float>;
                else if (key.Op == ReductionOp.Max)
                    return key.InType == NPTypeCode.Double ? (NDInnerLoopFunc)SimdMaxKernel<double> : SimdMaxKernel<float>;
                if (key.InType == NPTypeCode.Double) return CreateSimdReduceKernel<double>(key.Op);
                return CreateSimdReduceKernel<float>(key.Op);
            }

            return null;
        }

        /// <summary>
        /// Builds a scalar per-chunk reduce kernel over arbitrary numeric
        /// <typeparamref name="TIn"/> → <typeparamref name="TAccum"/> via .NET generic math.
        /// The JIT monomorphizes one tight body per (TIn,TAccum,op); CreateTruncating folds to
        /// a reinterpret/convert and the arithmetic to native ops. Min/Max propagate NaN
        /// (NumPy parity) via <c>TAccum.IsNaN</c>; Decimal has no NaN so those checks fold away.
        /// </summary>
        private static unsafe NDInnerLoopFunc CreateTypedReduceKernel<TIn, TAccum>(ReductionOp op)
            where TIn : unmanaged, INumberBase<TIn>
            where TAccum : unmanaged, INumber<TAccum>
        {
            return op switch
            {
                ReductionOp.Sum  => TypedSumKernel<TIn, TAccum>,
                ReductionOp.Prod => TypedProdKernel<TIn, TAccum>,
                ReductionOp.Min  => TypedMinKernel<TIn, TAccum>,
                ReductionOp.Max  => TypedMaxKernel<TIn, TAccum>,
                _ => null
            };
        }

        // =====================================================================
        // Generic same-type SUM kernels (Phase 6 — numeric: double/float/…)
        // =====================================================================
        //
        // For TIn == TAccum (no NEP50 widening: float/double Sum/Mean) one generic
        // body monomorphizes per dtype. The two reduce orientations map exactly onto
        // NumPy's two summation behaviors (loops_arithm_fp.dispatch.c.src reduce
        // branch + loops_utils.h.src pairwise_sum):
        //   • PINNED (reduced axis is the contiguous inner loop): NumPy folds the
        //     stripe with pairwise_sum. We do the same — PairwiseFold below is ported
        //     1:1 and is BIT-FOR-BIT identical to np.add.reduce for float/double
        //     (validated).
        //   • SLAB (a kept axis is the contiguous inner loop; reduced axis is outer):
        //     NumPy accumulates rows sequentially (out[c] += in[c]); so do we (the
        //     Vector256 streaming add), which already bit-matches NumPy on that
        //     orientation. Accuracy is orientation-dependent — exactly like NumPy.
        // NaN propagates naturally (a NaN element makes the whole reduction NaN).
        //
        // The pairwise PINNED path is what makes float32 sum/mean SAFE to route
        // (its earlier exclusion was due to a flat 8-accumulator divergence of ~24;
        // pairwise removes that — exact parity). NDIter supplies the substrate:
        // NewReduce orders the contiguous axis innermost (→ PINNED for a contiguous
        // reduced axis), EXTERNAL_LOOP delivers the full stripe as (ptr, count,
        // stride), and the +0 identity seed + slot-accumulate reproduce NumPy's
        // `*acc += pairwise_sum(...)` (so buffered/chunked stripes stay correct).

        private const int PairwiseBlock = 128; // NumPy PW_BLOCKSIZE

        /// <summary>
        /// NumPy's pairwise_sum (loops_utils.h.src) ported 1:1; <paramref name="stride"/>
        /// is in ELEMENTS. n&lt;8 naive (seed -0 to preserve signed zero); n≤128 eight
        /// independent accumulators unrolled by 8 (+ software prefetch on the contiguous
        /// path) then the exact tree-combine; n&gt;128 split (kept a multiple of 8) and
        /// recurse. Bit-for-bit identical to <c>np.add.reduce</c> for float/double.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe T PairwiseFold<T>(T* a, long n, long stride)
            where T : unmanaged, INumber<T>
        {
            if (n < 8)
            {
                T res = -T.Zero; // preserve -0.0 (NumPy: summing only -0 must stay -0)
                for (long i = 0; i < n; i++) res += a[i * stride];
                return res;
            }
            if (n <= PairwiseBlock)
            {
                // 8 independent accumulators (NumPy's r[0..7]); they break the FP-add
                // dependency chain for ILP. The JIT schedules the 8 independent adds; for the
                // routed axis case the stripe recurses into many small leaves so the cost is
                // recursion + combine, not this inner loop (an explicit Vector256 lane form
                // measured identical, so we keep the simpler scalar body). Stride is in elements.
                T r0 = a[0],          r1 = a[stride],     r2 = a[2 * stride], r3 = a[3 * stride],
                  r4 = a[4 * stride], r5 = a[5 * stride], r6 = a[6 * stride], r7 = a[7 * stride];
                long i;
                bool pf = stride == 1 && Sse.IsSupported;
                long ahead = 512 / sizeof(T);
                for (i = 8; i < n - (n % 8); i += 8)
                {
                    if (pf) Sse.Prefetch0((void*)(a + i + ahead)); // NPY_PREFETCH equivalent
                    r0 += a[(i + 0) * stride]; r1 += a[(i + 1) * stride];
                    r2 += a[(i + 2) * stride]; r3 += a[(i + 3) * stride];
                    r4 += a[(i + 4) * stride]; r5 += a[(i + 5) * stride];
                    r6 += a[(i + 6) * stride]; r7 += a[(i + 7) * stride];
                }
                T res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
                for (; i < n; i++) res += a[i * stride];
                return res;
            }
            long n2 = n / 2; n2 -= n2 % 8;
            return PairwiseFold(a, n2, stride) + PairwiseFold(a + n2 * stride, n - n2, stride);
        }

        /// <summary>
        /// Builds a generic same-type (<typeparamref name="T"/> == accumulator) per-chunk
        /// reduce kernel. Currently only Sum is emitted (PairwiseSumSameType); Min/Max/Prod
        /// return null so the caller falls back to the Direct path until they are migrated.
        /// </summary>
        private static unsafe NDInnerLoopFunc CreateSimdReduceKernel<T>(ReductionOp op)
            where T : unmanaged, INumber<T>
        {
            return op switch
            {
                ReductionOp.Sum => PairwiseSumSameType<T>,
                _ => null
            };
        }

        /// <summary>
        /// Same-type sum, dual mode. PINNED (reduced axis inner): pairwise-fold the stripe
        /// and accumulate into the slot — bit-for-bit NumPy parity. SLAB (kept axis inner):
        /// a 4×-unrolled Vector256 elementwise add of the contiguous row into the output
        /// accumulator row (sequential over the outer reduced axis, matching NumPy). Strided
        /// inner loops fall to the scalar/pairwise-with-stride path. NaN propagates naturally.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void PairwiseSumSameType<T>(void** dataptrs, long* strides, long count, void* auxdata)
            where T : unmanaged, INumber<T>
        {
            byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
            byte* outp = (byte*)dataptrs[1]; long outS = strides[1];
            int sz = sizeof(T);

            if (outS == 0)
            {
                // PINNED: pairwise-fold the stripe into the slot. The slot is seeded to the
                // +0 identity, so a single stripe gives *out = pairwise(stripe) == NumPy;
                // chunked/buffered stripes accumulate (NumPy's `*acc += pairwise_sum(...)`).
                *(T*)outp += PairwiseFold((T*)inp, count, inS / sz);
            }
            else
            {
                // SLAB: out[c] += in[c].
                if (inS == sz && outS == sz)
                {
                    T* id = (T*)inp; T* od = (T*)outp; long i = 0; int W = Vector256<T>.Count;
                    if (Vector256.IsHardwareAccelerated)
                    {
                        long lim = count - count % (W * 4);
                        for (; i < lim; i += W * 4)
                        {
                            Vector256.Store(Vector256.Add(Vector256.Load(od + i),         Vector256.Load(id + i)),         od + i);
                            Vector256.Store(Vector256.Add(Vector256.Load(od + i + W),     Vector256.Load(id + i + W)),     od + i + W);
                            Vector256.Store(Vector256.Add(Vector256.Load(od + i + W * 2), Vector256.Load(id + i + W * 2)), od + i + W * 2);
                            Vector256.Store(Vector256.Add(Vector256.Load(od + i + W * 3), Vector256.Load(id + i + W * 3)), od + i + W * 3);
                        }
                        for (; i + W <= count; i += W)
                            Vector256.Store(Vector256.Add(Vector256.Load(od + i), Vector256.Load(id + i)), od + i);
                    }
                    for (; i < count; i++) od[i] += id[i];
                }
                else
                {
                    for (long k = 0; k < count; k++) *(T*)(outp + k * outS) += *(T*)(inp + k * inS);
                }
            }
        }

        /// <summary>
        /// Pre-fill <paramref name="output"/> with the reduction identity for
        /// <paramref name="op"/> before driving a REDUCE iterator. Required because
        /// the per-chunk kernels fold into the existing output slot(s). Writes
        /// through <see cref="NDArray.SetAtIndex(object, long)"/> so any output
        /// layout (contiguous fresh alloc or user-supplied view) is honored.
        /// </summary>
        public static void SeedReduceIdentity(NDArray output, ReductionOp op)
        {
            long n = output.size;
            if (n == 0) return;

            NPTypeCode tc = output.GetTypeCode;
            if (tc == NPTypeCode.Complex)
            {
                System.Numerics.Complex id = op switch
                {
                    ReductionOp.Sum  => System.Numerics.Complex.Zero,
                    ReductionOp.Prod => System.Numerics.Complex.One,
                    // (±inf, ±inf) so the first finite element displaces the identity
                    // under lexicographic comparison — mirrors GetIdentityValueTyped<Complex>.
                    // (GetIdentity's GetMaxValue gives (+inf,0), which is NOT the lex-largest.)
                    ReductionOp.Min  => new System.Numerics.Complex(double.PositiveInfinity, double.PositiveInfinity),
                    ReductionOp.Max  => new System.Numerics.Complex(double.NegativeInfinity, double.NegativeInfinity),
                    _ => throw new NotSupportedException($"SeedReduceIdentity: op {op} unsupported for Complex")
                };
                for (long i = 0; i < n; i++) output.SetAtIndex(id, i);
                return;
            }

            // Scalar numerics (Half/Double/Decimal/...): the shared identity table is correct
            // for total-ordering min/max (min seed = +inf/MaxValue, max seed = -inf/MinValue).
            object scalarId = op.GetIdentity(tc);
            for (long i = 0; i < n; i++) output.SetAtIndex(scalarId, i);
        }

        /// <summary>
        /// Divide every element of <paramref name="output"/> by <paramref name="count"/> in
        /// place — the post-pass that turns an accumulated axis Sum into a Mean. For Complex
        /// this divides both components by the real count (NumPy: <c>mean = sum / n</c>),
        /// which is exactly what the legacy <c>MeanAxisComplex</c> did per element but without
        /// its per-output-row NDArray allocation. Writes through
        /// <see cref="NDArray.SetAtIndex(object, long)"/> so any output layout is honored.
        /// </summary>
        public static void MeanDivideByCount(NDArray output, long count)
        {
            long n = output.size;
            if (n == 0 || count == 0) return;

            NPTypeCode tc = output.GetTypeCode;
            switch (tc)
            {
                case NPTypeCode.Complex:
                {
                    double d = count;
                    for (long i = 0; i < n; i++)
                    {
                        var c = (System.Numerics.Complex)output.GetAtIndex(i);
                        output.SetAtIndex(new System.Numerics.Complex(c.Real / d, c.Imaginary / d), i);
                    }
                    return;
                }
                case NPTypeCode.Double: // Half mean accumulates here (Half→Double); ReduceMean casts back.
                {
                    double d = count;
                    for (long i = 0; i < n; i++)
                        output.SetAtIndex((double)output.GetAtIndex(i) / d, i);
                    return;
                }
                case NPTypeCode.Single: // mean(float32) stays float32 (NumPy: f32 sum / count → f32).
                {
                    float d = count;
                    for (long i = 0; i < n; i++)
                        output.SetAtIndex((float)output.GetAtIndex(i) / d, i);
                    return;
                }
                case NPTypeCode.Decimal:
                {
                    decimal d = count;
                    for (long i = 0; i < n; i++)
                        output.SetAtIndex((decimal)output.GetAtIndex(i) / d, i);
                    return;
                }
                default:
                    throw new NotSupportedException($"MeanDivideByCount not implemented for {tc}");
            }
        }

        // =====================================================================
        // Complex kernels
        // =====================================================================

        private const long ComplexBytes = 16; // sizeof(System.Numerics.Complex)

        // out += sum(in)  — double-pair SIMD for the contiguous fast path.
        private static unsafe void ComplexSumKernel(void** dataptrs, long* strides, long count, void* auxdata)
        {
            byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
            byte* outp = (byte*)dataptrs[1]; long outS = strides[1];

            if (outS == 0)
            {
                // PINNED: fold the whole stripe into one (re,im) slot.
                double* o = (double*)outp;
                double re = o[0], im = o[1];

                if (inS == ComplexBytes)
                {
                    double* d = (double*)inp;
                    long n2 = count * 2;
                    long i = 0;
                    if (Vector256.IsHardwareAccelerated && n2 >= 4)
                    {
                        var acc = Vector256<double>.Zero;
                        for (; i + 4 <= n2; i += 4)
                            acc = Vector256.Add(acc, Vector256.Load(d + i));
                        double* tmp = stackalloc double[4];
                        Vector256.Store(acc, tmp);
                        re += tmp[0] + tmp[2];
                        im += tmp[1] + tmp[3];
                    }
                    for (; i < n2; i += 2) { re += d[i]; im += d[i + 1]; }
                }
                else
                {
                    for (long k = 0; k < count; k++)
                    {
                        double* c = (double*)(inp + k * inS);
                        re += c[0]; im += c[1];
                    }
                }

                o[0] = re; o[1] = im;
                return;
            }

            // SLAB: out[c] += in[c].
            if (inS == ComplexBytes && outS == ComplexBytes)
            {
                double* id = (double*)inp;
                double* od = (double*)outp;
                long n2 = count * 2;
                long i = 0;
                if (Vector256.IsHardwareAccelerated)
                    for (; i + 4 <= n2; i += 4)
                        Vector256.Store(Vector256.Add(Vector256.Load(od + i), Vector256.Load(id + i)), od + i);
                for (; i < n2; i++) od[i] += id[i];
            }
            else
            {
                for (long k = 0; k < count; k++)
                {
                    double* c = (double*)(inp + k * inS);
                    double* o = (double*)(outp + k * outS);
                    o[0] += c[0]; o[1] += c[1];
                }
            }
        }

        // out *= prod(in)  — scalar complex multiply (no SIMD form for complex mul).
        // Inlined naive (ac-bd, ad+bc) on raw doubles: System.Numerics.Complex's
        // operator* carries .NET's infinity-rescue branch (Smith-style) and doesn't
        // inline through `*=` here, which made prod ~5 ns/elem. The naive formula is
        // also what NumPy uses, so this is closer parity on inf/0 mixes too.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ComplexProdKernel(void** dataptrs, long* strides, long count, void* auxdata)
        {
            byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
            byte* outp = (byte*)dataptrs[1]; long outS = strides[1];

            if (outS == 0)
            {
                double* o = (double*)outp;
                double pr = o[0], pi = o[1];
                for (long k = 0; k < count; k++)
                {
                    double* c = (double*)(inp + k * inS);
                    double br = c[0], bi = c[1];
                    double nr = pr * br - pi * bi;
                    pi = pr * bi + pi * br;
                    pr = nr;
                }
                o[0] = pr; o[1] = pi;
            }
            else
            {
                for (long k = 0; k < count; k++)
                {
                    double* c = (double*)(inp + k * inS);
                    double* o = (double*)(outp + k * outS);
                    double ar = o[0], ai = o[1], br = c[0], bi = c[1];
                    double nr = ar * br - ai * bi;
                    o[1] = ar * bi + ai * br;
                    o[0] = nr;
                }
            }
        }

        private static unsafe void ComplexMinKernel(void** dataptrs, long* strides, long count, void* auxdata)
            => ComplexMinMax(dataptrs, strides, count, pickGreater: false);

        private static unsafe void ComplexMaxKernel(void** dataptrs, long* strides, long count, void* auxdata)
            => ComplexMinMax(dataptrs, strides, count, pickGreater: true);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ComplexMinMax(void** dataptrs, long* strides, long count, bool pickGreater)
        {
            byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
            byte* outp = (byte*)dataptrs[1]; long outS = strides[1];

            if (outS == 0)
            {
                double* o = (double*)outp;
                double ar = o[0], ai = o[1];
                for (long k = 0; k < count; k++)
                {
                    double* c = (double*)(inp + k * inS);
                    LexFold(ref ar, ref ai, c[0], c[1], pickGreater);
                }
                o[0] = ar; o[1] = ai;
            }
            else
            {
                for (long k = 0; k < count; k++)
                {
                    double* c = (double*)(inp + k * inS);
                    double* o = (double*)(outp + k * outS);
                    double ar = o[0], ai = o[1];
                    LexFold(ref ar, ref ai, c[0], c[1], pickGreater);
                    o[0] = ar; o[1] = ai;
                }
            }
        }

        /// <summary>
        /// NumPy-parity Complex Min/Max fold on raw (re,im) doubles: a NaN-containing
        /// accumulator wins (stays), otherwise the incoming value replaces it when it
        /// is lexicographically more extreme on (Real, Imaginary). Same semantics as
        /// DirectILKernelGenerator.ComplexLexPick (the legacy path) — NaN tested via
        /// the branch-light <c>x != x</c> idiom.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static void LexFold(ref double ar, ref double ai, double br, double bi, bool pickGreater)
        {
            if (ar != ar || ai != ai) return;           // accumulator is NaN → keep it
            if (br != br || bi != bi) { ar = br; ai = bi; return; } // incoming NaN → take it
            bool aGreater = ar > br || (ar == br && ai > bi);
            bool takeA = pickGreater ? aGreater : !aGreater;
            if (!takeA) { ar = br; ai = bi; }
        }

        // =====================================================================
        // Generic scalar kernels (Half / Decimal — and any INumber numeric)
        // =====================================================================

        /// <summary>
        /// Read one input element and promote to the accumulator type. When TIn==TAccum the
        /// <c>typeof</c> test is a JIT constant and this folds to a plain reinterpret read —
        /// avoiding a per-element CreateTruncating static-virtual call on the hot same-type
        /// path (the difference between Half pinned-reduce at ~parity vs +30%).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe TAccum ConvIn<TIn, TAccum>(byte* p)
            where TIn : unmanaged, INumberBase<TIn>
            where TAccum : unmanaged, INumber<TAccum>
            => typeof(TIn) == typeof(TAccum) ? *(TAccum*)p : TAccum.CreateTruncating(*(TIn*)p);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void TypedSumKernel<TIn, TAccum>(void** dataptrs, long* strides, long count, void* auxdata)
            where TIn : unmanaged, INumberBase<TIn>
            where TAccum : unmanaged, INumber<TAccum>
        {
            byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
            byte* outp = (byte*)dataptrs[1]; long outS = strides[1];
            if (outS == 0)
            {
                TAccum acc = *(TAccum*)outp;
                for (long k = 0; k < count; k++) acc += ConvIn<TIn, TAccum>(inp + k * inS);
                *(TAccum*)outp = acc;
            }
            else
            {
                for (long k = 0; k < count; k++)
                {
                    TAccum* o = (TAccum*)(outp + k * outS);
                    *o += ConvIn<TIn, TAccum>(inp + k * inS);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void TypedProdKernel<TIn, TAccum>(void** dataptrs, long* strides, long count, void* auxdata)
            where TIn : unmanaged, INumberBase<TIn>
            where TAccum : unmanaged, INumber<TAccum>
        {
            byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
            byte* outp = (byte*)dataptrs[1]; long outS = strides[1];
            if (outS == 0)
            {
                TAccum acc = *(TAccum*)outp;
                for (long k = 0; k < count; k++) acc *= ConvIn<TIn, TAccum>(inp + k * inS);
                *(TAccum*)outp = acc;
            }
            else
            {
                for (long k = 0; k < count; k++)
                {
                    TAccum* o = (TAccum*)(outp + k * outS);
                    *o *= ConvIn<TIn, TAccum>(inp + k * inS);
                }
            }
        }

        private static unsafe void TypedMinKernel<TIn, TAccum>(void** dataptrs, long* strides, long count, void* auxdata)
            where TIn : unmanaged, INumberBase<TIn>
            where TAccum : unmanaged, INumber<TAccum>
            => TypedMinMax<TIn, TAccum>(dataptrs, strides, count, pickGreater: false);

        private static unsafe void TypedMaxKernel<TIn, TAccum>(void** dataptrs, long* strides, long count, void* auxdata)
            where TIn : unmanaged, INumberBase<TIn>
            where TAccum : unmanaged, INumber<TAccum>
            => TypedMinMax<TIn, TAccum>(dataptrs, strides, count, pickGreater: true);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void TypedMinMax<TIn, TAccum>(void** dataptrs, long* strides, long count, bool pickGreater)
            where TIn : unmanaged, INumberBase<TIn>
            where TAccum : unmanaged, INumber<TAccum>
        {
            byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
            byte* outp = (byte*)dataptrs[1]; long outS = strides[1];
            if (outS == 0)
            {
                TAccum acc = *(TAccum*)outp;
                for (long k = 0; k < count; k++)
                    acc = MinMaxFold(acc, ConvIn<TIn, TAccum>(inp + k * inS), pickGreater);
                *(TAccum*)outp = acc;
            }
            else
            {
                for (long k = 0; k < count; k++)
                {
                    TAccum* o = (TAccum*)(outp + k * outS);
                    *o = MinMaxFold(*o, ConvIn<TIn, TAccum>(inp + k * inS), pickGreater);
                }
            }
        }

        /// <summary>
        /// NaN-propagating min/max fold (NumPy parity): a NaN accumulator wins (stays); an
        /// incoming NaN replaces a finite accumulator. For types without NaN (Decimal) the
        /// <c>IsNaN</c> calls fold to false and this is a plain comparison.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static TAccum MinMaxFold<TAccum>(TAccum acc, TAccum v, bool pickGreater)
            where TAccum : unmanaged, INumber<TAccum>
        {
            if (TAccum.IsNaN(acc)) return acc;
            if (TAccum.IsNaN(v)) return v;
            bool takeV = pickGreater ? (v > acc) : (v < acc);
            return takeV ? v : acc;
        }

        // =====================================================================
        // SIMD same-type Min/Max (Phase 6 — double/float). Dual-mode, NaN-propagating
        // (NumPy parity). Mirrors PairwiseSumSameType so f64/f32 Min/Max ride the
        // stride-ordered NDIter reduce path at SIMD speed on EVERY layout
        // (transpose / F / strided / negcol / broadcast) instead of falling to the
        // C-contiguity-gated Direct axis kernel (which collapses to a cache-hostile
        // coordinate walk on negative-stride / broadcast inputs — measured 6–17×
        // slower than NumPy). The scalar tail / strided fallback reuse MinMaxFold so
        // they are bit-identical to the Half/Decimal/Complex TypedMinMax path.
        //
        // NaN handling: Vector256.Min/Max do NOT propagate NaN (x86 MINPD/MAXPD return
        // the SECOND operand when either input is NaN), so each step blends with
        // ConditionalSelect to keep the actual NaN OPERAND (accumulator first, then
        // input) — reproducing np.minimum/np.maximum's "NaN wins" AND preserving the
        // input NaN's bit pattern/sign exactly like the scalar MinMaxFold (NumPy
        // propagates the input NaN, not .NET's negative double.NaN). The non-NaN result
        // is order-independent (min/max associative+commutative), so the lane-parallel
        // fold matches the sequential scalar fold for all finite/inf inputs.
        // =====================================================================

        private static unsafe void SimdMinKernel<T>(void** dataptrs, long* strides, long count, void* auxdata)
            where T : unmanaged, IFloatingPointIeee754<T>
            => SimdMinMaxSameType<T>(dataptrs, strides, count, pickGreater: false);

        private static unsafe void SimdMaxKernel<T>(void** dataptrs, long* strides, long count, void* auxdata)
            where T : unmanaged, IFloatingPointIeee754<T>
            => SimdMinMaxSameType<T>(dataptrs, strides, count, pickGreater: true);

        // Per-lane NaN-propagating min/max: keep acc's NaN if acc is NaN, else input's
        // NaN if input is NaN, else the plain Vector256 min/max. Mirrors MinMaxFold lane-wise
        // (and therefore preserves the input NaN bits NumPy propagates).
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static Vector256<T> NaNFoldVec<T>(Vector256<T> acc, Vector256<T> v, bool pickGreater)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            // m is consumed only in lanes where NEITHER operand is NaN (the ConditionalSelect
            // chain overrides NaN lanes), so the raw VMINPD/VMAXPD (no NaN fixup) is exact here.
            var m = pickGreater ? RawMax256(acc, v) : RawMin256(acc, v);
            var accNaN = ~Vector256.Equals(acc, acc);   // acc != acc ⇒ NaN lane
            var vNaN = ~Vector256.Equals(v, v);
            return Vector256.ConditionalSelect(accNaN, acc, Vector256.ConditionalSelect(vNaN, v, m));
        }

        // Raw x86 Avx.Min/Max for float/double — a single VMINPD/VMAXPD WITHOUT the net9+ JIT
        // NaN-propagation fixup (an extra compare+blend) that Vector256.Min/Max carry (~2x the
        // raw instruction). SAFE in this file because every caller either tracks a separate
        // finite mask + cold NaN scan (SimdMinMaxSameType) or overrides NaN lanes via
        // ConditionalSelect (NaNFoldVec), so the raw op's NaN-dropping is intended. Non-x86
        // (ARM) falls back to Vector256.Min/Max (no x86 fixup there). typeof(T)/IsSupported
        // are JIT constants, so each specialization compiles to the single intrinsic.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static Vector256<T> RawMin256<T>(Vector256<T> a, Vector256<T> b) where T : unmanaged
        {
            if (Avx.IsSupported)
            {
                if (typeof(T) == typeof(double)) return Avx.Min(a.AsDouble(), b.AsDouble()).As<double, T>();
                if (typeof(T) == typeof(float)) return Avx.Min(a.AsSingle(), b.AsSingle()).As<float, T>();
            }
            return Vector256.Min(a, b);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static Vector256<T> RawMax256<T>(Vector256<T> a, Vector256<T> b) where T : unmanaged
        {
            if (Avx.IsSupported)
            {
                if (typeof(T) == typeof(double)) return Avx.Max(a.AsDouble(), b.AsDouble()).As<double, T>();
                if (typeof(T) == typeof(float)) return Avx.Max(a.AsSingle(), b.AsSingle()).As<float, T>();
            }
            return Vector256.Max(a, b);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SimdMinMaxSameType<T>(void** dataptrs, long* strides, long count, bool pickGreater)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
            byte* outp = (byte*)dataptrs[1]; long outS = strides[1];
            int sz = sizeof(T);
            int W = Vector256<T>.Count;

            if (outS == 0)
            {
                // PINNED: fold the contiguous stripe into the single slot. The slot is
                // seeded ±inf (SeedReduceIdentity) and accumulates across buffered chunks.
                T cur = *(T*)outp;
                if (T.IsNaN(cur)) return;            // NaN already won — stays NaN
                T* id = (T*)inp;

                if (inS == sz && Vector256.IsHardwareAccelerated && count >= W)
                {
                    // Hot loop: 4×-unrolled PLAIN min/max (1 op/lane, like the sum SLAB) plus a
                    // cheap per-lane finite mask. Vector256.Min/Max don't propagate NaN, but a
                    // NaN in a min/max stripe is rare; the finite mask flags it and a cold scalar
                    // scan then recomputes the exact result (the first NaN's bits — NumPy/scalar
                    // parity). This keeps the common (no-NaN) path at sum-class throughput; the
                    // earlier NaN-correct-every-step fold (NaNFoldVec) was ~5 ops/lane → 2–3×
                    // slower than the Direct whole-array kernel on contiguous reduced axes.
                    var a0 = Vector256.Create(cur); var a1 = a0; var a2 = a0; var a3 = a0;
                    var fin = Vector256<T>.AllBitsSet;
                    long i = 0, lim = count - count % (W * 4);
                    for (; i < lim; i += W * 4)
                    {
                        var v0 = Vector256.Load(id + i);
                        var v1 = Vector256.Load(id + i + W);
                        var v2 = Vector256.Load(id + i + 2 * W);
                        var v3 = Vector256.Load(id + i + 3 * W);
                        if (pickGreater)
                        { a0 = RawMax256(a0, v0); a1 = RawMax256(a1, v1); a2 = RawMax256(a2, v2); a3 = RawMax256(a3, v3); }
                        else
                        { a0 = RawMin256(a0, v0); a1 = RawMin256(a1, v1); a2 = RawMin256(a2, v2); a3 = RawMin256(a3, v3); }
                        fin &= Vector256.Equals(v0, v0) & Vector256.Equals(v1, v1) & Vector256.Equals(v2, v2) & Vector256.Equals(v3, v3);
                    }
                    var va = pickGreater
                        ? RawMax256(RawMax256(a0, a1), RawMax256(a2, a3))
                        : RawMin256(RawMin256(a0, a1), RawMin256(a2, a3));
                    for (; i + W <= count; i += W)
                    {
                        var v = Vector256.Load(id + i);
                        va = pickGreater ? RawMax256(va, v) : RawMin256(va, v);
                        fin &= Vector256.Equals(v, v);
                    }
                    // fin lane = all-ones ⇒ that lane was finite throughout; MSB-extract → bit set.
                    bool anyNaN = Vector256.ExtractMostSignificantBits(fin) != (uint)((1 << W) - 1);
                    T acc = va.GetElement(0);
                    for (int l = 1; l < W; l++) acc = MinMaxFold(acc, va.GetElement(l), pickGreater);
                    for (; i < count; i++)
                    {
                        T x = id[i];
                        if (T.IsNaN(x)) anyNaN = true;
                        acc = MinMaxFold(acc, x, pickGreater);
                    }
                    // Cold path: a NaN was present ⇒ the reduction is NaN. Propagate the FIRST
                    // input NaN's exact bits, matching the sequential scalar fold and np.minimum.
                    if (anyNaN)
                        for (long k = 0; k < count; k++) { T x = id[k]; if (T.IsNaN(x)) { acc = x; break; } }
                    *(T*)outp = acc;
                }
                else
                {
                    T acc = cur;
                    for (long k = 0; k < count; k++)
                        acc = MinMaxFold(acc, *(T*)(inp + k * inS), pickGreater);
                    *(T*)outp = acc;
                }
            }
            else
            {
                // SLAB: out[c] = minmax(out[c], in[c]); the slab is revisited across outer
                // iterations (seeded ±inf), so fold in place.
                if (inS == sz && outS == sz && Vector256.IsHardwareAccelerated)
                {
                    T* id = (T*)inp; T* od = (T*)outp;
                    long i = 0, lim = count - count % W;
                    for (; i < lim; i += W)
                        Vector256.Store(NaNFoldVec(Vector256.Load(od + i), Vector256.Load(id + i), pickGreater), od + i);
                    for (; i < count; i++) od[i] = MinMaxFold(od[i], id[i], pickGreater);
                }
                else
                {
                    for (long k = 0; k < count; k++)
                    {
                        T* o = (T*)(outp + k * outS);
                        *o = MinMaxFold(*o, *(T*)(inp + k * inS), pickGreater);
                    }
                }
            }
        }
    }
}
