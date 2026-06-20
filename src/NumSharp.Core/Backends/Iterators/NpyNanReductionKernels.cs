using System;
using System.Numerics;

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

    // =========================================================================
    // Half NaN kernels — widen each element to double (the precision NumSharp's
    // f16 reductions already use); the caller narrows the result back to Half.
    // Mirror the Float/Double kernels above. NaN = Half.IsNaN(val).
    // =========================================================================

    public readonly struct NanSumHalfKernel : INpyReducingInnerLoop<double>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref double sum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double acc = sum;
            for (long i = 0; i < count; i++)
            {
                double val = (double)*(Half*)(p + i * stride);
                if (!double.IsNaN(val))
                    acc += val;
            }
            sum = acc;
            return true;
        }
    }

    public readonly struct NanProdHalfKernel : INpyReducingInnerLoop<double>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref double prod)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double acc = prod;
            for (long i = 0; i < count; i++)
            {
                double val = (double)*(Half*)(p + i * stride);
                if (!double.IsNaN(val))
                    acc *= val;
            }
            prod = acc;
            return true;
        }
    }

    public readonly struct NanMinHalfKernel : INpyReducingInnerLoop<NanMinMaxDoubleAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref NanMinMaxDoubleAccumulator accum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double minVal = accum.Value;
            bool found = accum.Found;
            for (long i = 0; i < count; i++)
            {
                double val = (double)*(Half*)(p + i * stride);
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

    public readonly struct NanMaxHalfKernel : INpyReducingInnerLoop<NanMinMaxDoubleAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref NanMinMaxDoubleAccumulator accum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double maxVal = accum.Value;
            bool found = accum.Found;
            for (long i = 0; i < count; i++)
            {
                double val = (double)*(Half*)(p + i * stride);
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

    public readonly struct NanMeanHalfKernel : INpyReducingInnerLoop<NanMeanAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref NanMeanAccumulator accum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double sum = accum.Sum;
            long n = accum.Count;
            for (long i = 0; i < count; i++)
            {
                double val = (double)*(Half*)(p + i * stride);
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

    public struct NanSquaredDeviationHalfKernel : INpyReducingInnerLoop<double>
    {
        private readonly double _mean;

        public NanSquaredDeviationHalfKernel(double mean)
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
                double val = (double)*(Half*)(p + i * stride);
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

    // =========================================================================
    // Complex NaN kernels — "NaN" = real OR imaginary is NaN (NumPy parity).
    // nansum keeps the Complex sum; nanmean tracks Complex sum + non-NaN count;
    // nanvar/nanstd's second pass accumulates |z - mean|² as a double.
    // =========================================================================

    /// <summary>nanmean accumulator for Complex: running Complex sum and non-NaN count.</summary>
    public struct NanMeanComplexAccumulator
    {
        public Complex Sum;
        public long Count;
    }

    public readonly struct NanSumComplexKernel : INpyReducingInnerLoop<Complex>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref Complex sum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            Complex acc = sum;
            for (long i = 0; i < count; i++)
            {
                Complex val = *(Complex*)(p + i * stride);
                if (!double.IsNaN(val.Real) && !double.IsNaN(val.Imaginary))
                    acc += val;
            }
            sum = acc;
            return true;
        }
    }

    public readonly struct NanMeanComplexKernel : INpyReducingInnerLoop<NanMeanComplexAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref NanMeanComplexAccumulator accum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            Complex sum = accum.Sum;
            long n = accum.Count;
            for (long i = 0; i < count; i++)
            {
                Complex val = *(Complex*)(p + i * stride);
                if (!double.IsNaN(val.Real) && !double.IsNaN(val.Imaginary))
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

    public struct NanSquaredDeviationComplexKernel : INpyReducingInnerLoop<double>
    {
        private readonly double _meanR;
        private readonly double _meanI;

        public NanSquaredDeviationComplexKernel(double meanR, double meanI)
        {
            _meanR = meanR;
            _meanI = meanI;
        }

        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref double sumSq)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double meanR = _meanR, meanI = _meanI;
            double total = sumSq;
            for (long i = 0; i < count; i++)
            {
                Complex val = *(Complex*)(p + i * stride);
                if (!double.IsNaN(val.Real) && !double.IsNaN(val.Imaginary))
                {
                    double dR = val.Real - meanR;
                    double dI = val.Imaginary - meanI;
                    total += dR * dR + dI * dI;
                }
            }
            sumSq = total;
            return true;
        }
    }
}
