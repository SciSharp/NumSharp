using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
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

            // Phase 6 — numeric same-type SIMD. Double Sum (and Mean, which routes through
            // the Sum kernel + MeanDivideByCount) only for now; float32/integers deferred
            // (float needs pairwise to match NumPy; ints need widening accumulators).
            if (key.InType == NPTypeCode.Double && key.AccType == NPTypeCode.Double)
                return CreateSimdReduceKernel<double>(key.Op);

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
        // Generic SIMD same-type kernels (Phase 6 — numeric: double/float/int…)
        // =====================================================================
        //
        // For TIn == TAccum (no NEP50 widening: float/double Sum, and Min/Max on
        // every numeric type) one generic Vector256<T> body monomorphizes to the
        // same machine code the legacy DirectILKernelGenerator emits per dtype.
        // The proof harness (benchmark/poc/phase6_*.cs) measured this at Direct
        // parity at scale (10M double sum axis1 1.01×, axis0 1.09×) once the
        // kernel carries [MethodImpl(AggressiveOptimization)] — without it the
        // delegate target JITs at tier-0 and looks ~10× slower (a measurement
        // artifact, NOT the per-chunk model). Sum is NaN-natural (a NaN element
        // makes the whole reduction NaN, matching NumPy); accumulation order is
        // identity for C-contiguous input (same as Direct), so values match.
        //
        // SCOPE TODAY: Double Sum/Mean only (routed in CreateReduceInnerLoop +
        // UseNpyIterReduce). float32 sum is deliberately NOT routed here — its
        // simple 8-way accumulation diverges from NumPy's pairwise summation by
        // more than float tolerance (proof: 10M axis1 maxdiff ≈ 24). Min/Max and
        // the widening integer sums are later migration steps.

        /// <summary>
        /// Builds a generic same-type (<typeparamref name="T"/> == accumulator) SIMD
        /// per-chunk reduce kernel. The JIT monomorphizes one tight Vector256&lt;T&gt;
        /// body per (T, op). Currently only Sum is emitted; Min/Max/Prod return null
        /// (caller falls back to the Direct path) until they are migrated.
        /// </summary>
        private static unsafe NpyInnerLoopFunc CreateSimdReduceKernel<T>(ReductionOp op)
            where T : unmanaged, INumber<T>
        {
            return op switch
            {
                ReductionOp.Sum => SimdSumSameType<T>,
                _ => null
            };
        }

        /// <summary>
        /// Same-type SIMD sum: PINNED folds a contiguous stripe into one slot with an
        /// 8-accumulator horizontal reduce; SLAB does a 4×-unrolled Vector256 elementwise
        /// add of the contiguous row into the output accumulator row. Strided inner loops
        /// (non-unit element stride) fall to the scalar tail. NaN propagates naturally.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SimdSumSameType<T>(void** dataptrs, long* strides, long count, void* auxdata)
            where T : unmanaged, INumber<T>
        {
            byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
            byte* outp = (byte*)dataptrs[1]; long outS = strides[1];
            int sz = sizeof(T);

            if (outS == 0)
            {
                // PINNED: horizontal sum of the contiguous stripe into one slot.
                T acc = *(T*)outp;
                if (inS == sz)
                {
                    T* d = (T*)inp; long i = 0; int W = Vector256<T>.Count;
                    if (Vector256.IsHardwareAccelerated && count >= W * 8)
                    {
                        Vector256<T> a0 = Vector256<T>.Zero, a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
                        long lim = count - count % (W * 8);
                        for (; i < lim; i += W * 8)
                        {
                            a0 = Vector256.Add(a0, Vector256.Load(d + i));
                            a1 = Vector256.Add(a1, Vector256.Load(d + i + W));
                            a2 = Vector256.Add(a2, Vector256.Load(d + i + W * 2));
                            a3 = Vector256.Add(a3, Vector256.Load(d + i + W * 3));
                            a4 = Vector256.Add(a4, Vector256.Load(d + i + W * 4));
                            a5 = Vector256.Add(a5, Vector256.Load(d + i + W * 5));
                            a6 = Vector256.Add(a6, Vector256.Load(d + i + W * 6));
                            a7 = Vector256.Add(a7, Vector256.Load(d + i + W * 7));
                        }
                        a0 = Vector256.Add(Vector256.Add(Vector256.Add(a0, a1), Vector256.Add(a2, a3)),
                                           Vector256.Add(Vector256.Add(a4, a5), Vector256.Add(a6, a7)));
                        acc += Vector256.Sum(a0);
                    }
                    for (; i < count; i++) acc += d[i];
                }
                else
                {
                    for (long k = 0; k < count; k++) acc += *(T*)(inp + k * inS);
                }
                *(T*)outp = acc;
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
