using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;

// =============================================================================
// ILKernelGenerator.Scan.cs — NpyIter-driven cumulative-sum (accumulate) kernels
// =============================================================================
//
// MODEL
// -----
// np.cumsum is np.add.accumulate. NumPy drives it with an nditer built over
// [output, input], KEEPORDER + MULTI_INDEX, then NpyIter_RemoveAxis(scanAxis) +
// NpyIter_RemoveMultiIndex(): the iterator walks every axis EXCEPT the scan axis,
// and a strided inner loop runs the running sum along the (removed) scan axis per
// outer position (numpy/_core/src/umath/ufunc_object.c : PyUFunc_Accumulate).
//
// We mirror that exactly. The engine (Default.Reduction.CumAdd.AccumulateAxis)
// builds the iterator, removes the scan axis, enables EXTERNAL_LOOP and drives it
// with ForEach. Each per-chunk call hands us a contiguous stripe of `count`
// scan-line starts over the innermost REMAINING axis (strides[0]/strides[1] are
// that axis' byte strides); the scan axis byte strides + length ride in auxdata.
//
//   for each of `count` outer positions:
//       acc = in[0];  out[0] = acc                    // NumPy memmove-first
//       for i in 1..axisLen:  acc += in[i];  out[i] = acc
//
// ONE generic body (ScanAddWiden<TIn,TAccum>) covers all 14 INumber accumulator
// dtypes via .NET generic math + ConvIn (NEP50 widening folds to a reinterpret on
// the same-type path, a CreateTruncating widen otherwise); Complex (no INumber)
// has a dedicated double-pair kernel. This REPLACES the ~2,500-line per-dtype
// AxisCumSum* tree in DirectILKernelGenerator.Scan.cs with no behavioural change
// other than the NumPy-aligned output layout the engine now allocates.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        /// <summary>
        /// Scan-axis geometry passed to a cumulative inner-loop kernel via
        /// <c>auxdata</c>. Byte strides + element count for the (removed) scan axis;
        /// the iterator supplies the per-operand byte strides for the remaining
        /// innermost axis through the kernel's <c>strides</c> argument.
        /// </summary>
        public struct ScanAxisAux
        {
            /// <summary>Input byte stride along the scan axis (may be negative for reversed views).</summary>
            public long InByteStride;
            /// <summary>Output byte stride along the scan axis.</summary>
            public long OutByteStride;
            /// <summary>Scan-axis length (number of elements accumulated per line).</summary>
            public long AxisLen;
        }

        /// <summary>
        /// Cache of per-chunk cumulative-sum kernels keyed by (input dtype,
        /// accumulator dtype). Construction (reflection for the generic widen body)
        /// happens once per pair; subsequent lookups are a dictionary hit.
        /// </summary>
        private static readonly ConcurrentDictionary<(NPTypeCode In, NPTypeCode Acc), NpyInnerLoopFunc> _cumSumCache = new();

        /// <summary>
        /// Returns the NpyIter-driven cumulative-sum inner loop for
        /// <paramref name="inType"/> → <paramref name="accType"/>. The returned
        /// delegate matches <see cref="NpyInnerLoopFunc"/>; drive it with an
        /// iterator whose scan axis has been removed (see
        /// <c>DefaultEngine.AccumulateAxis</c>), passing a pointer to a
        /// <see cref="ScanAxisAux"/> as the kernel's auxdata.
        /// </summary>
        public static NpyInnerLoopFunc GetCumSumInnerLoop(NPTypeCode inType, NPTypeCode accType)
            => _cumSumCache.GetOrAdd((inType, accType), static k => CreateCumSumInnerLoop(k.In, k.Acc));

        private static unsafe NpyInnerLoopFunc CreateCumSumInnerLoop(NPTypeCode inType, NPTypeCode accType)
        {
            // Complex accumulator: dedicated double-pair kernel (Complex is not INumber).
            // cumsum into Complex from a non-Complex source is not supported (matches the
            // legacy CumSumWithConversionGeneral, which threw for a Complex target).
            if (accType == NPTypeCode.Complex)
            {
                if (inType != NPTypeCode.Complex)
                    throw new NotSupportedException($"cumsum to Complex from {inType} is not supported");
                return ScanAddComplex;
            }

            // Char has no INumberBase implementation but is a bit-identical 2-byte unsigned;
            // read it as ushort so the generic body monomorphizes.
            Type clrIn = inType == NPTypeCode.Char ? typeof(ushort) : DirectILKernelGenerator.GetClrType(inType);
            Type clrAcc = DirectILKernelGenerator.GetClrType(accType);

            var mi = typeof(ILKernelGenerator)
                .GetMethod(nameof(ScanAddWiden), BindingFlags.NonPublic | BindingFlags.Static)!
                .MakeGenericMethod(clrIn, clrAcc);
            return (NpyInnerLoopFunc)mi.CreateDelegate(typeof(NpyInnerLoopFunc));
        }

        /// <summary>
        /// Generic cumulative sum over <typeparamref name="TIn"/> → <typeparamref name="TAccum"/>.
        /// One JIT-monomorphized body per (TIn,TAccum): <see cref="ConvIn{TIn,TAccum}"/> folds to a
        /// reinterpret read on the same-type path and a CreateTruncating widen on the NEP50 path,
        /// the running add is a native op. The first element is copied (NumPy's memmove-first), so
        /// signed-zero / NaN of <c>out[0]</c> match np.add.accumulate exactly.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ScanAddWiden<TIn, TAccum>(void** dataptrs, long* strides, long count, void* auxdata)
            where TIn : unmanaged, INumberBase<TIn>
            where TAccum : unmanaged, INumber<TAccum>
        {
            byte* inBase = (byte*)dataptrs[0]; long inInner = strides[0];
            byte* outBase = (byte*)dataptrs[1]; long outInner = strides[1];
            var aux = (ScanAxisAux*)auxdata;
            long inAxis = aux->InByteStride, outAxis = aux->OutByteStride, n = aux->AxisLen;

            for (long c = 0; c < count; c++)
            {
                byte* ip = inBase + c * inInner;
                byte* op = outBase + c * outInner;
                if (n <= 0) continue;

                TAccum acc = ConvIn<TIn, TAccum>(ip);   // out[0] = (TAccum)in[0]
                *(TAccum*)op = acc;
                for (long i = 1; i < n; i++)
                {
                    acc += ConvIn<TIn, TAccum>(ip + i * inAxis);
                    *(TAccum*)(op + i * outAxis) = acc;
                }
            }
        }

        /// <summary>
        /// Complex cumulative sum accumulated as a (re,im) double pair — bit-identical to
        /// np.add.accumulate on complex128 (sequential add). Mirrors <see cref="ComplexSumKernel"/>'s
        /// raw-double accumulation; first element copied (NumPy memmove-first).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ScanAddComplex(void** dataptrs, long* strides, long count, void* auxdata)
        {
            byte* inBase = (byte*)dataptrs[0]; long inInner = strides[0];
            byte* outBase = (byte*)dataptrs[1]; long outInner = strides[1];
            var aux = (ScanAxisAux*)auxdata;
            long inAxis = aux->InByteStride, outAxis = aux->OutByteStride, n = aux->AxisLen;

            for (long c = 0; c < count; c++)
            {
                byte* ip = inBase + c * inInner;
                byte* op = outBase + c * outInner;
                if (n <= 0) continue;

                double* f = (double*)ip;
                double re = f[0], im = f[1];
                double* o0 = (double*)op;
                o0[0] = re; o0[1] = im;
                for (long i = 1; i < n; i++)
                {
                    double* ci = (double*)(ip + i * inAxis);
                    re += ci[0]; im += ci[1];
                    double* oi = (double*)(op + i * outAxis);
                    oi[0] = re; oi[1] = im;
                }
            }
        }
    }
}
