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

        // ---- read-ahead fills of the ziggurat samplers (see DrawBuffer.cs for why they keep NumPy's stream) ----
        //
        // The four ziggurat fills below run the accept-outright step INLINE over locals (the read position, the refill
        // bound): the draw buffer's fields would otherwise live on the stack — it is passed by reference to the
        // continuation — and be reloaded and re-stored around every output. Only the rare continuation (a rejected
        // candidate: ~0.7% normal, ~1.1% exponential) resumes a DrawBuffer over the same scratch, owing the same count.
        // Every refill draws at most the outputs still owed, so the stream and the engine state stay NumPy's.

        /// <summary>Fills <paramref name="n"/> standard normals (NumPy's <c>random_standard_normal_fill</c>) from a read-ahead buffer.</summary>
        /// <param name="p">The destination.</param>
        /// <param name="n">The number of draws.</param>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        [System.Runtime.CompilerServices.SkipLocalsInit] // the read-ahead scratch is always written before it is read
        private unsafe void FillStandardNormal(double* p, long n)
        {
            const int cap = DrawBuffer64.Capacity;
            ulong* storage = stackalloc ulong[cap];
            int pos = 0, avail = 0;
            for (long i = 0; i < n; i++)
            {
                if (pos == avail)
                {
                    avail = (int)Math.Min(cap, n - i);
                    _bitGenerator.FillUInt64(storage, avail);
                    pos = 0;
                }
                if (NormalAccept(storage[pos++], out int idx, out ulong rabs, out double x))
                {
                    p[i] = x;
                    continue;
                }
                var src = new DrawBuffer64(_bitGenerator, storage, cap, pos, avail, n - i);
                p[i] = StandardNormalUnlikely(ref src, idx, rabs, x);
                pos = src.Position;
                avail = src.Available;
            }
        }

        /// <summary>Fills <paramref name="n"/> float32 standard normals (NumPy's <c>random_standard_normal_fill_f</c>) from a read-ahead buffer.</summary>
        /// <param name="p">The destination.</param>
        /// <param name="n">The number of draws.</param>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        [System.Runtime.CompilerServices.SkipLocalsInit] // the read-ahead scratch is always written before it is read
        private unsafe void FillStandardNormalF(float* p, long n)
        {
            const int cap = DrawBuffer32.Capacity;
            uint* storage = stackalloc uint[cap];
            int pos = 0, avail = 0;
            for (long i = 0; i < n; i++)
            {
                if (pos == avail)
                {
                    avail = (int)Math.Min(cap, n - i);
                    _bitGenerator.FillUInt32(storage, avail);
                    pos = 0;
                }
                if (NormalAcceptF(storage[pos++], out int idx, out uint rabs, out float x))
                {
                    p[i] = x;
                    continue;
                }
                var src = new DrawBuffer32(_bitGenerator, storage, cap, pos, avail, n - i);
                p[i] = StandardNormalUnlikelyF(ref src, idx, rabs, x);
                pos = src.Position;
                avail = src.Available;
            }
        }

        /// <summary>Fills <paramref name="n"/> standard exponentials (NumPy's <c>random_standard_exponential_fill</c>) from a read-ahead buffer.</summary>
        /// <param name="p">The destination.</param>
        /// <param name="n">The number of draws.</param>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        [System.Runtime.CompilerServices.SkipLocalsInit] // the read-ahead scratch is always written before it is read
        private unsafe void FillStandardExponential(double* p, long n)
        {
            const int cap = DrawBuffer64.Capacity;
            ulong* storage = stackalloc ulong[cap];
            int pos = 0, avail = 0;
            for (long i = 0; i < n; i++)
            {
                if (pos == avail)
                {
                    avail = (int)Math.Min(cap, n - i);
                    _bitGenerator.FillUInt64(storage, avail);
                    pos = 0;
                }
                if (ExponentialAccept(storage[pos++], out int idx, out double x))
                {
                    p[i] = x;
                    continue;
                }
                var src = new DrawBuffer64(_bitGenerator, storage, cap, pos, avail, n - i);
                p[i] = StandardExponentialUnlikely(ref src, idx, x);
                pos = src.Position;
                avail = src.Available;
            }
        }

        /// <summary>Fills <paramref name="n"/> float32 standard exponentials (NumPy's <c>random_standard_exponential_fill_f</c>) from a read-ahead buffer.</summary>
        /// <param name="p">The destination.</param>
        /// <param name="n">The number of draws.</param>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        [System.Runtime.CompilerServices.SkipLocalsInit] // the read-ahead scratch is always written before it is read
        private unsafe void FillStandardExponentialF(float* p, long n)
        {
            const int cap = DrawBuffer32.Capacity;
            uint* storage = stackalloc uint[cap];
            int pos = 0, avail = 0;
            for (long i = 0; i < n; i++)
            {
                if (pos == avail)
                {
                    avail = (int)Math.Min(cap, n - i);
                    _bitGenerator.FillUInt32(storage, avail);
                    pos = 0;
                }
                if (ExponentialAcceptF(storage[pos++], out int idx, out float x))
                {
                    p[i] = x;
                    continue;
                }
                var src = new DrawBuffer32(_bitGenerator, storage, cap, pos, avail, n - i);
                p[i] = StandardExponentialUnlikelyF(ref src, idx, x);
                pos = src.Position;
                avail = src.Available;
            }
        }

        /// <summary>The memory-order start of a validated contiguous <c>out</c> (storage base plus the view's element offset).</summary>
        /// <param name="outArr">The destination.</param>
        /// <returns>The first element's address.</returns>
        private static unsafe byte* OutStart(NDArray outArr) => (byte*)outArr.Storage.Address + outArr.Shape.offset * outArr.dtypesize;

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
