using System;

namespace NumSharp.Backends.Iteration
{
    // =========================================================================
    // Fused weighted-sum kernels for np.average.
    //
    // Eliminates the broadcast `a * w` temp + axis-sum bottleneck in the
    // weighted path. Two execution shapes:
    //
    //   • Scalar (axis=None): ExecuteReducing<TKernel, WeightedSumAccum*>
    //     over 2 operands [a, w]. The (SumAW, SumW) pair lives in TAccum
    //     so a single pass yields both num and scl.
    //
    //   • Axis-specific: ExecuteGeneric<TKernel> over 4 operands
    //     [a, w, num_out, scl_out] with EXTERNAL_LOOP + REDUCE_OK and
    //     op_axes marking the reduction axes as -1 for num/scl. The kernel
    //     reads/writes the current output slot in place — output is
    //     pre-zeroed by the caller. BUFFER is intentionally NOT used because
    //     BufferedReduce double-iterates over our 3-operand setup; EXLOOP
    //     gives the kernel the inner reduce axis as one count per outer slot.
    //
    // Only float and double are specialized — that covers >95% of weighted
    // np.average use cases because the result dtype promotion rule forces
    // float64 for int+int, int+float, etc. Half/Complex/Decimal fall back
    // to the slower `aCast * wgtCast → sum` path in np.average.cs.
    // =========================================================================

    public struct WeightedSumAccumDouble
    {
        public double SumAW;
        public double SumW;
    }

    public struct WeightedSumAccumFloat
    {
        public float SumAW;
        public float SumW;
    }

    // -------------------------------------------------------------------------
    // Scalar (axis=None): 2 operands [a, w] → (SumAW, SumW) accumulator.
    // -------------------------------------------------------------------------

    public readonly struct WeightedSumScalarDoubleKernel : INpyReducingInnerLoop<WeightedSumAccumDouble>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref WeightedSumAccumDouble accum)
        {
            byte* ap = (byte*)dataptrs[0];
            byte* wp = (byte*)dataptrs[1];
            long aStride = strides[0];
            long wStride = strides[1];
            double sumAW = accum.SumAW;
            double sumW = accum.SumW;

            if (aStride == sizeof(double) && wStride == sizeof(double))
            {
                double* a = (double*)ap;
                double* w = (double*)wp;
                for (long i = 0; i < count; i++)
                {
                    double wv = w[i];
                    sumAW += a[i] * wv;
                    sumW += wv;
                }
            }
            else
            {
                for (long i = 0; i < count; i++)
                {
                    double wv = *(double*)(wp + i * wStride);
                    sumAW += *(double*)(ap + i * aStride) * wv;
                    sumW += wv;
                }
            }

            accum.SumAW = sumAW;
            accum.SumW = sumW;
            return true;
        }
    }

    public readonly struct WeightedSumScalarFloatKernel : INpyReducingInnerLoop<WeightedSumAccumFloat>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref WeightedSumAccumFloat accum)
        {
            byte* ap = (byte*)dataptrs[0];
            byte* wp = (byte*)dataptrs[1];
            long aStride = strides[0];
            long wStride = strides[1];
            float sumAW = accum.SumAW;
            float sumW = accum.SumW;

            if (aStride == sizeof(float) && wStride == sizeof(float))
            {
                float* a = (float*)ap;
                float* w = (float*)wp;
                for (long i = 0; i < count; i++)
                {
                    float wv = w[i];
                    sumAW += a[i] * wv;
                    sumW += wv;
                }
            }
            else
            {
                for (long i = 0; i < count; i++)
                {
                    float wv = *(float*)(wp + i * wStride);
                    sumAW += *(float*)(ap + i * aStride) * wv;
                    sumW += wv;
                }
            }

            accum.SumAW = sumAW;
            accum.SumW = sumW;
            return true;
        }
    }

    // -------------------------------------------------------------------------
    // Axis-specific: 4 operands [a, w, num_out, scl_out] driven by ExecuteGeneric
    // under EXTERNAL_LOOP. The iterator pins num_out/scl_out via stride==0 over
    // the reduction axis so each kernel call sees a single output slot for the
    // whole inner-axis stripe.
    // -------------------------------------------------------------------------

    public readonly struct WeightedSumAxisDoubleKernel : INpyInnerLoop
    {
        public unsafe void Execute(void** dataptrs, long* strides, long count)
        {
            byte* ap = (byte*)dataptrs[0];
            byte* wp = (byte*)dataptrs[1];
            byte* nP = (byte*)dataptrs[2];
            byte* sP = (byte*)dataptrs[3];
            long aStride = strides[0];
            long wStride = strides[1];
            long nStride = strides[2];
            long sStride = strides[3];

            // Pinned-output fast path: NpyIter set outStride=0 because the
            // reduction axis is innermost. Single load/store per call; JIT
            // can autovectorize the tight accumulation.
            if (nStride == 0 && sStride == 0)
            {
                double num = *(double*)nP;
                double scl = *(double*)sP;
                if (aStride == sizeof(double) && wStride == sizeof(double))
                {
                    double* a = (double*)ap;
                    double* w = (double*)wp;
                    for (long i = 0; i < count; i++)
                    {
                        double wv = w[i];
                        num += a[i] * wv;
                        scl += wv;
                    }
                }
                else
                {
                    for (long i = 0; i < count; i++)
                    {
                        double wv = *(double*)(wp + i * wStride);
                        num += *(double*)(ap + i * aStride) * wv;
                        scl += wv;
                    }
                }
                *(double*)nP = num;
                *(double*)sP = scl;
                return;
            }

            // Scatter path: each inner element targets a different output slot
            // (reduction axis is outer, walked across multiple kernel calls).
            // Outputs are pre-zeroed so `*outP += x` accumulates correctly.
            for (long i = 0; i < count; i++)
            {
                double av = *(double*)(ap + i * aStride);
                double wv = *(double*)(wp + i * wStride);
                *(double*)(nP + i * nStride) += av * wv;
                *(double*)(sP + i * sStride) += wv;
            }
        }
    }

    public readonly struct WeightedSumAxisFloatKernel : INpyInnerLoop
    {
        public unsafe void Execute(void** dataptrs, long* strides, long count)
        {
            byte* ap = (byte*)dataptrs[0];
            byte* wp = (byte*)dataptrs[1];
            byte* nP = (byte*)dataptrs[2];
            byte* sP = (byte*)dataptrs[3];
            long aStride = strides[0];
            long wStride = strides[1];
            long nStride = strides[2];
            long sStride = strides[3];

            if (nStride == 0 && sStride == 0)
            {
                float num = *(float*)nP;
                float scl = *(float*)sP;
                if (aStride == sizeof(float) && wStride == sizeof(float))
                {
                    float* a = (float*)ap;
                    float* w = (float*)wp;
                    for (long i = 0; i < count; i++)
                    {
                        float wv = w[i];
                        num += a[i] * wv;
                        scl += wv;
                    }
                }
                else
                {
                    for (long i = 0; i < count; i++)
                    {
                        float wv = *(float*)(wp + i * wStride);
                        num += *(float*)(ap + i * aStride) * wv;
                        scl += wv;
                    }
                }
                *(float*)nP = num;
                *(float*)sP = scl;
                return;
            }

            for (long i = 0; i < count; i++)
            {
                float av = *(float*)(ap + i * aStride);
                float wv = *(float*)(wp + i * wStride);
                *(float*)(nP + i * nStride) += av * wv;
                *(float*)(sP + i * sStride) += wv;
            }
        }
    }
}
