using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Iteration;

// =============================================================================
// ILKernelGenerator.Reduction.cs — per-chunk axis-reduction kernels (NpyIter-driven)
// =============================================================================
//
// MODEL
// -----
// These kernels implement the inner loop of a 2-operand REDUCE iterator built by
// NpyIterRef.NewReduce ([input, output], REDUCE_OK | EXTERNAL_LOOP). The iterator
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

        private static readonly ConcurrentDictionary<ReduceKernelKey, NpyInnerLoopFunc> _reduceCache = new();

        /// <summary>
        /// Returns the cached per-chunk reduction kernel for the given
        /// (op, input, accumulator) triple, or null when no NpyIter-driven kernel
        /// exists yet (caller falls back to the DirectILKernelGenerator path).
        /// The returned delegate matches <see cref="NpyInnerLoopFunc"/>; hand it to
        /// an iterator built by <see cref="NpyIterRef.NewReduce(NDArray, NDArray, int, NpyIterGlobalFlags)"/>.
        /// </summary>
        public static NpyInnerLoopFunc GetReduceInnerLoop(ReduceKernelKey key) =>
            _reduceCache.GetOrAdd(key, CreateReduceInnerLoop);

        private static unsafe NpyInnerLoopFunc CreateReduceInnerLoop(ReduceKernelKey key)
        {
            // Complex: dedicated double-pair kernels (SIMD sum, lex min/max).
            if (key.InType == NPTypeCode.Complex && key.AccType == NPTypeCode.Complex)
            {
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
            //     Half. (Half sum/prod/min/max stay on the Direct path — see UseNpyIterReduce:
            //     their Half-accumulator serial chain can't beat it on the pinned axis.)
            //   - Decimal accumulates in full-precision Decimal (no NumPy reference type).
            //     CreateTypedReduceKernel can build a same-type Half kernel too; it just isn't
            //     routed here today.
            if (key.InType == NPTypeCode.Half && key.AccType == NPTypeCode.Double)
                return key.Op == ReductionOp.Sum ? CreateTypedReduceKernel<Half, double>(ReductionOp.Sum) : null;
            if (key.InType == NPTypeCode.Decimal && key.AccType == NPTypeCode.Decimal)
                return CreateTypedReduceKernel<decimal, decimal>(key.Op);

            // Phase 6 — numeric same-type Sum (and Mean, via the Sum kernel + MeanDivideByCount).
            // Double AND float32 now route here: the PINNED path uses PairwiseFold, which is
            // bit-for-bit identical to NumPy's pairwise_sum, so float32 is exact (its earlier
            // exclusion was a flat-accumulator divergence the pairwise leaf removes). Integer
            // widening sums and Min/Max/Prod stay on the Direct path (CreateSimdReduceKernel
            // returns null for those → caller falls back).
            if (key.InType == key.AccType)
            {
                if (key.InType == NPTypeCode.Double) return CreateSimdReduceKernel<double>(key.Op);
                if (key.InType == NPTypeCode.Single) return CreateSimdReduceKernel<float>(key.Op);
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
        private static unsafe NpyInnerLoopFunc CreateTypedReduceKernel<TIn, TAccum>(ReductionOp op)
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
        //     (validated, benchmark/poc/pairwise_parity.{cs,py}).
        //   • SLAB (a kept axis is the contiguous inner loop; reduced axis is outer):
        //     NumPy accumulates rows sequentially (out[c] += in[c]); so do we (the
        //     Vector256 streaming add), which already bit-matches NumPy on that
        //     orientation. Accuracy is orientation-dependent — exactly like NumPy.
        // NaN propagates naturally (a NaN element makes the whole reduction NaN).
        //
        // The pairwise PINNED path is what makes float32 sum/mean SAFE to route
        // (its earlier exclusion was due to a flat 8-accumulator divergence of ~24;
        // pairwise removes that — exact parity). NpyIter supplies the substrate:
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
        private static unsafe NpyInnerLoopFunc CreateSimdReduceKernel<T>(ReductionOp op)
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
    }
}
