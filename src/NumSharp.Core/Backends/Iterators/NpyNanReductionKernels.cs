using System;

namespace NumSharp.Backends.Iteration
{
    // =========================================================================
    // NaN-Aware Reduction Kernels
    //
    // These struct kernels implement INpyReducingInnerLoop<TAccum> and drive
    // scalar (axis=None) NaN reductions through NpyIterRef.ExecuteReducing.
    // Layout-aware: work for contiguous, sliced, broadcast, and transposed
    // arrays because NpyIter produces per-inner-loop byte strides.
    //
    // Call pattern:
    //     using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);
    //     var result = iter.ExecuteReducing<NanSumFloatKernel, float>(default, 0f);
    // =========================================================================

    // -------------------------------------------------------------------------
    // Accumulators
    // -------------------------------------------------------------------------

    /// <summary>
    /// Accumulator for nanmean: running sum and count of non-NaN elements.
    /// </summary>
    public struct NanMeanAccumulator
    {
        public double Sum;
        public long Count;
    }

    /// <summary>
    /// Accumulator for NanMin/NanMax: running extremum plus a flag indicating
    /// whether any non-NaN element has been seen. Returns NaN if all elements
    /// were NaN.
    /// </summary>
    public struct NanMinMaxFloatAccumulator
    {
        public float Value;
        public bool Found;
    }

    /// <summary>
    /// Accumulator for NanMin/NanMax on double arrays.
    /// </summary>
    public struct NanMinMaxDoubleAccumulator
    {
        public double Value;
        public bool Found;
    }

    // -------------------------------------------------------------------------
    // NanSum kernels — skip NaN, accumulate the rest
    // -------------------------------------------------------------------------

    public readonly struct NanSumFloatKernel : INpyReducingInnerLoop<float>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref float sum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            for (long i = 0; i < count; i++)
            {
                float val = *(float*)(p + i * stride);
                if (!float.IsNaN(val))
                    sum += val;
            }
            return true;
        }
    }

    public readonly struct NanSumDoubleKernel : INpyReducingInnerLoop<double>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref double sum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            for (long i = 0; i < count; i++)
            {
                double val = *(double*)(p + i * stride);
                if (!double.IsNaN(val))
                    sum += val;
            }
            return true;
        }
    }

    // -------------------------------------------------------------------------
    // NanProd kernels — skip NaN, multiply the rest
    // -------------------------------------------------------------------------

    public readonly struct NanProdFloatKernel : INpyReducingInnerLoop<float>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref float prod)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            for (long i = 0; i < count; i++)
            {
                float val = *(float*)(p + i * stride);
                if (!float.IsNaN(val))
                    prod *= val;
            }
            return true;
        }
    }

    public readonly struct NanProdDoubleKernel : INpyReducingInnerLoop<double>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref double prod)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            for (long i = 0; i < count; i++)
            {
                double val = *(double*)(p + i * stride);
                if (!double.IsNaN(val))
                    prod *= val;
            }
            return true;
        }
    }

    // -------------------------------------------------------------------------
    // NanMin kernels — skip NaN, track minimum
    // -------------------------------------------------------------------------

    public readonly struct NanMinFloatKernel : INpyReducingInnerLoop<NanMinMaxFloatAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref NanMinMaxFloatAccumulator accum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            float minVal = accum.Value;
            bool found = accum.Found;
            for (long i = 0; i < count; i++)
            {
                float val = *(float*)(p + i * stride);
                if (!float.IsNaN(val))
                {
                    if (!found || val < minVal)
                        minVal = val;
                    found = true;
                }
            }
            accum.Value = minVal;
            accum.Found = found;
            return true;
        }
    }

    public readonly struct NanMinDoubleKernel : INpyReducingInnerLoop<NanMinMaxDoubleAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref NanMinMaxDoubleAccumulator accum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double minVal = accum.Value;
            bool found = accum.Found;
            for (long i = 0; i < count; i++)
            {
                double val = *(double*)(p + i * stride);
                if (!double.IsNaN(val))
                {
                    if (!found || val < minVal)
                        minVal = val;
                    found = true;
                }
            }
            accum.Value = minVal;
            accum.Found = found;
            return true;
        }
    }

    // -------------------------------------------------------------------------
    // NanMax kernels — skip NaN, track maximum
    // -------------------------------------------------------------------------

    public readonly struct NanMaxFloatKernel : INpyReducingInnerLoop<NanMinMaxFloatAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref NanMinMaxFloatAccumulator accum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            float maxVal = accum.Value;
            bool found = accum.Found;
            for (long i = 0; i < count; i++)
            {
                float val = *(float*)(p + i * stride);
                if (!float.IsNaN(val))
                {
                    if (!found || val > maxVal)
                        maxVal = val;
                    found = true;
                }
            }
            accum.Value = maxVal;
            accum.Found = found;
            return true;
        }
    }

    public readonly struct NanMaxDoubleKernel : INpyReducingInnerLoop<NanMinMaxDoubleAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref NanMinMaxDoubleAccumulator accum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double maxVal = accum.Value;
            bool found = accum.Found;
            for (long i = 0; i < count; i++)
            {
                double val = *(double*)(p + i * stride);
                if (!double.IsNaN(val))
                {
                    if (!found || val > maxVal)
                        maxVal = val;
                    found = true;
                }
            }
            accum.Value = maxVal;
            accum.Found = found;
            return true;
        }
    }

    // -------------------------------------------------------------------------
    // NanMean — first pass: sum + count of non-NaN values.
    // Caller computes mean = sum / count at end.
    // -------------------------------------------------------------------------

    public readonly struct NanMeanFloatKernel : INpyReducingInnerLoop<NanMeanAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref NanMeanAccumulator accum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double sum = accum.Sum;
            long n = accum.Count;
            for (long i = 0; i < count; i++)
            {
                float val = *(float*)(p + i * stride);
                if (!float.IsNaN(val))
                {
                    sum += val;
                    n++;
                }
            }
            accum.Sum = sum;
            accum.Count = n;
            return true;
        }
    }

    public readonly struct NanMeanDoubleKernel : INpyReducingInnerLoop<NanMeanAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref NanMeanAccumulator accum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double sum = accum.Sum;
            long n = accum.Count;
            for (long i = 0; i < count; i++)
            {
                double val = *(double*)(p + i * stride);
                if (!double.IsNaN(val))
                {
                    sum += val;
                    n++;
                }
            }
            accum.Sum = sum;
            accum.Count = n;
            return true;
        }
    }

    // -------------------------------------------------------------------------
    // NanVar/NanStd — second pass: sum of squared deviations from a known mean.
    // Kernel holds the mean from the first pass. Caller divides by (count - ddof)
    // and optionally takes sqrt for std.
    //
    // NOTE: Two-pass (not Welford) to preserve numerical behavior of the legacy
    // AsIterator path exactly.
    // -------------------------------------------------------------------------

    public struct NanSquaredDeviationFloatKernel : INpyReducingInnerLoop<double>
    {
        private readonly double _mean;

        public NanSquaredDeviationFloatKernel(double mean)
        {
            _mean = mean;
        }

        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref double sumSq)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double mean = _mean;
            double total = sumSq;
            for (long i = 0; i < count; i++)
            {
                float val = *(float*)(p + i * stride);
                if (!float.IsNaN(val))
                {
                    double diff = val - mean;
                    total += diff * diff;
                }
            }
            sumSq = total;
            return true;
        }
    }

    public struct NanSquaredDeviationDoubleKernel : INpyReducingInnerLoop<double>
    {
        private readonly double _mean;

        public NanSquaredDeviationDoubleKernel(double mean)
        {
            _mean = mean;
        }

        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref double sumSq)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double mean = _mean;
            double total = sumSq;
            for (long i = 0; i < count; i++)
            {
                double val = *(double*)(p + i * stride);
                if (!double.IsNaN(val))
                {
                    double diff = val - mean;
                    total += diff * diff;
                }
            }
            sumSq = total;
            return true;
        }
    }
}
