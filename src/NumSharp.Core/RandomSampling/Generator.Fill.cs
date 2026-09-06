using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        // Random output is always a fresh C-contiguous owning array; these fill it sequentially
        // (RNG draws carry a strict data dependency, exactly as NumPy fills a contiguous buffer
        // then reshapes).

        private unsafe NDArray FillDoubleDist(Shape shape, Func<double> sampler)
        {
            var ret = new NDArray(typeof(double), shape, false);
            if (shape.size == 0)
                return ret;
            var p = (double*)ret.Address;
            long n = shape.size;
            for (long i = 0; i < n; i++)
                p[i] = sampler();
            return ret;
        }

        private unsafe NDArray FillFloatDist(Shape shape, Func<float> sampler)
        {
            var ret = new NDArray(typeof(float), shape, false);
            if (shape.size == 0)
                return ret;
            var p = (float*)ret.Address;
            long n = shape.size;
            for (long i = 0; i < n; i++)
                p[i] = sampler();
            return ret;
        }

        private unsafe void FillDoubleDistInto(NDArray outArr, Func<double> sampler)
        {
            long n = outArr.size;
            if (n == 0)
                return;
            var p = (double*)outArr.Address;
            for (long i = 0; i < n; i++)
                p[i] = sampler();
        }

        private unsafe void FillFloatDistInto(NDArray outArr, Func<float> sampler)
        {
            long n = outArr.size;
            if (n == 0)
                return;
            var p = (float*)outArr.Address;
            for (long i = 0; i < n; i++)
                p[i] = sampler();
        }
    }
}
