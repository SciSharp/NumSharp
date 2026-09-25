using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        // Random output is always a fresh C-contiguous owning array; these fill it sequentially
        // (RNG draws carry a strict data dependency through the engine state, exactly as NumPy fills a
        // contiguous buffer with `for i in range(n): out[i] = f(state, ...)`). Every caller holds the
        // bit generator's lock for the whole fill.

        /// <summary>Allocates <paramref name="shape"/> as float64 and fills it with successive <paramref name="sampler"/> draws.</summary>
        /// <param name="shape">The output shape (0-d and empty shapes allowed).</param>
        /// <param name="sampler">The per-element draw; called exactly <c>shape.size</c> times, in order.</param>
        /// <returns>A fresh C-contiguous float64 array.</returns>
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

        /// <summary>Allocates <paramref name="shape"/> as float32 and fills it with successive <paramref name="sampler"/> draws.</summary>
        /// <param name="shape">The output shape (0-d and empty shapes allowed).</param>
        /// <param name="sampler">The per-element draw; called exactly <c>shape.size</c> times, in order.</param>
        /// <returns>A fresh C-contiguous float32 array.</returns>
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

        /// <summary>Fills a validated float64 <c>out</c> in memory order with successive <paramref name="sampler"/> draws.</summary>
        /// <param name="outArr">The destination (contiguous and writeable — already validated).</param>
        /// <param name="sampler">The per-element draw; called exactly <c>outArr.size</c> times.</param>
        /// <remarks>
        ///     NumPy writes <c>PyArray_DATA(out)</c> sequentially, i.e. in MEMORY order, so the start is the
        ///     storage base plus the view's element offset (the documented rule for any view — the bare
        ///     storage base would ignore a view's offset).
        /// </remarks>
        private unsafe void FillDoubleDistInto(NDArray outArr, Func<double> sampler)
        {
            long n = outArr.size;
            if (n == 0)
                return;
            var p = (double*)(outArr.Storage.Address + outArr.Shape.offset * sizeof(double));
            for (long i = 0; i < n; i++)
                p[i] = sampler();
        }

        /// <summary>Fills a validated float32 <c>out</c> in memory order with successive <paramref name="sampler"/> draws.</summary>
        /// <param name="outArr">The destination (contiguous and writeable — already validated).</param>
        /// <param name="sampler">The per-element draw; called exactly <c>outArr.size</c> times.</param>
        /// <remarks>Memory-order write from the offset-adjusted base, as in <see cref="FillDoubleDistInto"/>.</remarks>
        private unsafe void FillFloatDistInto(NDArray outArr, Func<float> sampler)
        {
            long n = outArr.size;
            if (n == 0)
                return;
            var p = (float*)(outArr.Storage.Address + outArr.Shape.offset * sizeof(float));
            for (long i = 0; i < n; i++)
                p[i] = sampler();
        }
    }
}
