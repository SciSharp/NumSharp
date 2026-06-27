using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Backends.Iteration;
using Unsafe = System.Runtime.CompilerServices.Unsafe;

// =============================================================================
// DirectILKernelGenerator.WeightedSum.cs — fused weighted-sum kernel for np.average.
//
// RESPONSIBILITY
//   - WeightedSumKernelKey            — cache key (dtype only; layout handled
//                                       at runtime inside the kernel body)
//   - GetWeightedSumIterKernel(key)   — single user-facing entry point;
//                                       cached per dtype, returns the IL-emitted
//                                       NDInnerLoopFunc that np.average
//                                       hands to NDIter.ForEach
//   - CreateWeightedSumIterKernel     — factory; the only place that switches
//                                       on NPTypeCode (one-time per dtype,
//                                       result cached forever in
//                                       _weightedSumCache)
//   - WeightedSumIterKernelBody<T>    — single generic helper that handles all
//                                       NDIter shapes: pinned-output (reduce
//                                       axis is innermost) and scatter-output
//                                       (reduce axis is outer). Uses Vector256<T>
//                                       SIMD via the existing AddOp<T>/MulOp<T>
//                                       op-tag struct generics, falls back to
//                                       AddScalar<T>/MulScalar<T> for the
//                                       remainder + non-contig + non-SIMD paths.
//
// CALL SHAPE
//   Operands: [a, w, num_out, scl_out]
//     a, w           — READONLY,  pre-cast to result dtype by np.average
//     num_out, scl_out — READWRITE, pre-zeroed by np.average
//   Iterator flags : REDUCE_OK | EXTERNAL_LOOP
//   op_axes        : a/w = identity, num/scl = -1 in reduction axes (axis=None
//                    means all -1 → both outputs are 0-D scalars pinned via
//                    stride==0). One ForEach call per output slot; count is
//                    the innermost-axis size after coalescing.
//
//   Kernel body sees the four dataptrs and four strides. When num/scl strides
//   are both 0 the reduction axis is innermost and we run the tight SIMD
//   accumulation; otherwise we fall back to per-element scatter with the
//   running totals already in the pre-zeroed output slots.
//
// SUPPORTED DTYPES
//   The body is generic over T : unmanaged. SIMD specialization activates
//   automatically for T = float/double/int*/uint* via Vector256<T>.IsSupported.
//   For Half/Complex/Decimal/Bool/Char the factory returns null and np.average
//   falls back to the existing `aCast * wgtCast → sum` path.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        public readonly record struct WeightedSumKernelKey(NPTypeCode Dtype);

        internal static readonly ConcurrentDictionary<WeightedSumKernelKey, NDInnerLoopFunc?> _weightedSumCache = new();

        /// <summary>
        /// Returns the cached IL-emitted weighted-sum kernel for the given dtype,
        /// or null if the dtype isn't supported (Bool/Char/Half/Complex/Decimal).
        /// The kernel signature matches NDInnerLoopFunc; pass it to
        /// NDIter.ForEach over the 4-operand [a, w, num_out, scl_out] iterator.
        /// </summary>
        public static NDInnerLoopFunc? GetWeightedSumIterKernel(WeightedSumKernelKey key) =>
            _weightedSumCache.GetOrAdd(key, CreateWeightedSumIterKernel);

        private static NDInnerLoopFunc? CreateWeightedSumIterKernel(WeightedSumKernelKey key) =>
            key.Dtype switch
            {
                NPTypeCode.Single  => CreateWeightedSumIterKernelTyped<float>(),
                NPTypeCode.Double  => CreateWeightedSumIterKernelTyped<double>(),
                NPTypeCode.Byte    => CreateWeightedSumIterKernelTyped<byte>(),
                NPTypeCode.SByte   => CreateWeightedSumIterKernelTyped<sbyte>(),
                NPTypeCode.Int16   => CreateWeightedSumIterKernelTyped<short>(),
                NPTypeCode.UInt16  => CreateWeightedSumIterKernelTyped<ushort>(),
                NPTypeCode.Int32   => CreateWeightedSumIterKernelTyped<int>(),
                NPTypeCode.UInt32  => CreateWeightedSumIterKernelTyped<uint>(),
                NPTypeCode.Int64   => CreateWeightedSumIterKernelTyped<long>(),
                NPTypeCode.UInt64  => CreateWeightedSumIterKernelTyped<ulong>(),
                _                  => null
            };

        private static unsafe NDInnerLoopFunc CreateWeightedSumIterKernelTyped<T>() where T : unmanaged
        {
            // Closure over T captures the generic specialization. The JIT compiles
            // one specialized closure per T, just like AxisReductionSimdHelper<T>.
            return (void** dataptrs, long* strides, long count, void* _) =>
            {
                WeightedSumIterKernelBody<T>(dataptrs, strides, count);
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void WeightedSumIterKernelBody<T>(void** dataptrs, long* strides, long count)
            where T : unmanaged
        {
            byte* ap = (byte*)dataptrs[0];
            byte* wp = (byte*)dataptrs[1];
            byte* nP = (byte*)dataptrs[2];
            byte* sP = (byte*)dataptrs[3];
            long aStride = strides[0];
            long wStride = strides[1];
            long nStride = strides[2];
            long sStride = strides[3];

            // Pinned-output fast path: reduction axis is innermost (or axis=None),
            // so num/scl pointers stay on a single slot for the whole inner stripe.
            // We accumulate into locals then write back once. JIT-folded SIMD
            // branch handles contig input.
            if (nStride == 0 && sStride == 0)
            {
                T num = *(T*)nP;
                T scl = *(T*)sP;
                WeightedSumPinned<T>(ap, wp, ref num, ref scl, aStride, wStride, count);
                *(T*)nP = num;
                *(T*)sP = scl;
                return;
            }

            // Scatter path: each inner element targets a different output slot
            // (reduction axis is outer; iterator visits the inner non-reduce axis
            // with stride != 0 on outputs). Outputs are pre-zeroed so `+=` runs.
            // Per-T inner loops via typeof(T)== chains so the JIT picks one
            // primitive arithmetic body — AddScalar/MulScalar would box every
            // operand and burn the GC on million-element scatters.
            WeightedSumScatter<T>(ap, wp, nP, sP, aStride, wStride, nStride, sStride, count);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void WeightedSumScatter<T>(
            byte* ap, byte* wp, byte* nP, byte* sP,
            long aStride, long wStride, long nStride, long sStride, long count)
            where T : unmanaged
        {
            if (typeof(T) == typeof(double))
            {
                for (long i = 0; i < count; i++)
                {
                    double av = *(double*)(ap + i * aStride);
                    double wv = *(double*)(wp + i * wStride);
                    *(double*)(nP + i * nStride) += av * wv;
                    *(double*)(sP + i * sStride) += wv;
                }
                return;
            }
            if (typeof(T) == typeof(float))
            {
                for (long i = 0; i < count; i++)
                {
                    float av = *(float*)(ap + i * aStride);
                    float wv = *(float*)(wp + i * wStride);
                    *(float*)(nP + i * nStride) += av * wv;
                    *(float*)(sP + i * sStride) += wv;
                }
                return;
            }
            if (typeof(T) == typeof(int))
            {
                for (long i = 0; i < count; i++)
                {
                    int av = *(int*)(ap + i * aStride);
                    int wv = *(int*)(wp + i * wStride);
                    *(int*)(nP + i * nStride) += av * wv;
                    *(int*)(sP + i * sStride) += wv;
                }
                return;
            }
            if (typeof(T) == typeof(long))
            {
                for (long i = 0; i < count; i++)
                {
                    long av = *(long*)(ap + i * aStride);
                    long wv = *(long*)(wp + i * wStride);
                    *(long*)(nP + i * nStride) += av * wv;
                    *(long*)(sP + i * sStride) += wv;
                }
                return;
            }
            // Generic fallback (boxes — only used for rarely-hit dtypes since
            // the kernel cache rejects unsupported dtypes upstream).
            for (long i = 0; i < count; i++)
            {
                T av = *(T*)(ap + i * aStride);
                T wv = *(T*)(wp + i * wStride);
                T* nSlot = (T*)(nP + i * nStride);
                T* sSlot = (T*)(sP + i * sStride);
                *nSlot = AddScalar(*nSlot, MulScalar(av, wv));
                *sSlot = AddScalar(*sSlot, wv);
            }
        }

        // Pinned-output accumulator. Picks SIMD when both strides are element-size,
        // scalar otherwise. The SIMD branch uses 4-way unrolled Vector256<T> with
        // independent accumulators (breaks the FMA dep chain — same pattern as
        // AxisReductionInnermostTyped<T, TOp>).
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void WeightedSumPinned<T>(
            byte* ap, byte* wp, ref T num, ref T scl, long aStride, long wStride, long count)
            where T : unmanaged
        {
            if (aStride == sizeof(T) && wStride == sizeof(T) &&
                Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported &&
                count >= Vector256<T>.Count * 4)
            {
                T* a = (T*)ap;
                T* w = (T*)wp;
                int vCount = Vector256<T>.Count;
                long step = vCount * 4;

                var vN0 = Vector256<T>.Zero; var vN1 = Vector256<T>.Zero;
                var vN2 = Vector256<T>.Zero; var vN3 = Vector256<T>.Zero;
                var vS0 = Vector256<T>.Zero; var vS1 = Vector256<T>.Zero;
                var vS2 = Vector256<T>.Zero; var vS3 = Vector256<T>.Zero;

                long i = 0;
                for (; i + step <= count; i += step)
                {
                    var a0 = Vector256.Load(a + i);            var w0 = Vector256.Load(w + i);
                    var a1 = Vector256.Load(a + i + vCount);   var w1 = Vector256.Load(w + i + vCount);
                    var a2 = Vector256.Load(a + i + vCount*2); var w2 = Vector256.Load(w + i + vCount*2);
                    var a3 = Vector256.Load(a + i + vCount*3); var w3 = Vector256.Load(w + i + vCount*3);
                    vN0 = Vector256.Add(vN0, Vector256.Multiply(a0, w0)); vS0 = Vector256.Add(vS0, w0);
                    vN1 = Vector256.Add(vN1, Vector256.Multiply(a1, w1)); vS1 = Vector256.Add(vS1, w1);
                    vN2 = Vector256.Add(vN2, Vector256.Multiply(a2, w2)); vS2 = Vector256.Add(vS2, w2);
                    vN3 = Vector256.Add(vN3, Vector256.Multiply(a3, w3)); vS3 = Vector256.Add(vS3, w3);
                }

                // Single-vector remainder
                for (; i + vCount <= count; i += vCount)
                {
                    var av = Vector256.Load(a + i);
                    var wv = Vector256.Load(w + i);
                    vN0 = Vector256.Add(vN0, Vector256.Multiply(av, wv));
                    vS0 = Vector256.Add(vS0, wv);
                }

                // Tree merge
                var vN = Vector256.Add(Vector256.Add(vN0, vN1), Vector256.Add(vN2, vN3));
                var vS = Vector256.Add(Vector256.Add(vS0, vS1), Vector256.Add(vS2, vS3));
                num = AddScalar(num, Vector256.Sum(vN));
                scl = AddScalar(scl, Vector256.Sum(vS));

                // Scalar tail
                for (; i < count; i++)
                {
                    T wv = w[i];
                    num = AddScalar(num, MulScalar(a[i], wv));
                    scl = AddScalar(scl, wv);
                }
                return;
            }

            // Scalar / non-contig pinned-output path.
            if (aStride == sizeof(T) && wStride == sizeof(T))
            {
                T* a = (T*)ap;
                T* w = (T*)wp;
                for (long i = 0; i < count; i++)
                {
                    T wv = w[i];
                    num = AddScalar(num, MulScalar(a[i], wv));
                    scl = AddScalar(scl, wv);
                }
            }
            else
            {
                for (long i = 0; i < count; i++)
                {
                    T av = *(T*)(ap + i * aStride);
                    T wv = *(T*)(wp + i * wStride);
                    num = AddScalar(num, MulScalar(av, wv));
                    scl = AddScalar(scl, wv);
                }
            }
        }
    }
}
