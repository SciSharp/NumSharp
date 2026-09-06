using System;

namespace NumSharp
{
    /// <summary>
    ///     The modern NumPy random number container returned by <c>np.random.default_rng</c>.
    /// </summary>
    /// <remarks>
    ///     Port of NumPy 2.4.2's <c>numpy.random.Generator</c> (<c>numpy/random/_generator.pyx</c>).
    ///     Unlike the legacy <see cref="NumPyRandom"/> (<c>RandomState</c>, MT19937 + polar-method
    ///     normal / inverse-CDF exponential / masked bounded integers), <see cref="Generator"/> draws
    ///     from a <see cref="PCG64"/> bit generator and uses NumPy's newer algorithms — ziggurat
    ///     normal/exponential and Lemire bounded integers — so its stream matches
    ///     <c>default_rng(seed)</c> bit-for-bit, not <c>RandomState</c>.
    /// </remarks>
    public sealed partial class Generator
    {
        private readonly BitGenerator _bitGenerator;

        /// <summary>Constructs a Generator over the given bit generator.</summary>
        public Generator(BitGenerator bitGenerator)
        {
            _bitGenerator = bitGenerator ?? throw new ArgumentNullException(nameof(bitGenerator));
        }

        /// <summary>The bit generator supplying this Generator's stream.</summary>
        public BitGenerator bit_generator => _bitGenerator;

        /// <inheritdoc/>
        public override string ToString() => $"Generator({_bitGenerator.Name})";

        // ---- shared output helpers (random output is always a fresh C-contiguous owning array) ----

        /// <summary>True when <paramref name="size"/> denotes "no size" (NumPy's <c>size=None</c>).</summary>
        private static bool IsNoSize(Shape size) => size.IsEmpty || size.IsScalar;

        private static unsafe NDArray FillDoubles(Shape shape, BitGenerator bg)
        {
            var ret = new NDArray(typeof(double), shape, false);
            if (shape.size == 0)
                return ret;
            var p = (double*)ret.Address;
            long n = shape.size;
            for (long i = 0; i < n; i++)
                p[i] = bg.NextDouble();
            return ret;
        }

        private static unsafe NDArray FillFloats(Shape shape, BitGenerator bg)
        {
            var ret = new NDArray(typeof(float), shape, false);
            if (shape.size == 0)
                return ret;
            var p = (float*)ret.Address;
            long n = shape.size;
            for (long i = 0; i < n; i++)
                p[i] = bg.NextFloat();
            return ret;
        }

        /// <summary>
        ///     Return random floats in the half-open interval <c>[0.0, 1.0)</c>.
        /// </summary>
        /// <param name="size">Output shape. If default/scalar a single value is returned.</param>
        /// <param name="dtype">Desired dtype — only <c>float64</c> (default) and <c>float32</c> are supported.</param>
        /// <param name="out">Optional output array to place the result in.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.random.html</remarks>
        public NDArray random(Shape size = default, Type dtype = null, NDArray @out = null)
        {
            dtype = dtype ?? typeof(double);
            NPTypeCode tc = ResolveFloatDtype(dtype, "random");

            if (@out is not null)
            {
                ValidateOut(@out, size, tc, "random");
                FillRandomInto(@out, tc);
                return @out;
            }

            if (IsNoSize(size))
            {
                // NumPy's float_fill/double_fill return a Python float (float64) for size=None even
                // when dtype=float32 — the float32 draw is widened to double for the scalar return
                // (_common.pyx float_fill: `random_func(state, 1, &out_val); return out_val`, the C
                // float auto-converting to a Python float). Only the sized/out= paths stay float32.
                if (tc == NPTypeCode.Single)
                    return NDArray.Scalar((double)_bitGenerator.NextFloat());
                return NDArray.Scalar(_bitGenerator.NextDouble());
            }

            return tc == NPTypeCode.Single ? FillFloats(size, _bitGenerator) : FillDoubles(size, _bitGenerator);
        }

        private static NPTypeCode ResolveFloatDtype(Type dtype, string name)
        {
            var tc = dtype.GetTypeCode();
            if (tc != NPTypeCode.Double && tc != NPTypeCode.Single)
                throw new TypeError($"Unsupported dtype dtype('{tc.AsNumpyDtypeName()}') for {name}");
            return tc;
        }

        private unsafe void FillRandomInto(NDArray @out, NPTypeCode tc)
        {
            long n = @out.size;
            if (n == 0)
                return;
            if (tc == NPTypeCode.Single)
            {
                var p = (float*)@out.Address;
                for (long i = 0; i < n; i++) p[i] = _bitGenerator.NextFloat();
            }
            else
            {
                var p = (double*)@out.Address;
                for (long i = 0; i < n; i++) p[i] = _bitGenerator.NextDouble();
            }
        }

        /// <summary>
        ///     Validates an <c>out=</c> array against the requested size and loop dtype (NumPy semantics).
        /// </summary>
        /// <remarks>
        ///     Mirrors NumPy's <c>check_output</c> (<c>_common.pyx</c>, <c>require_c_array=False</c>) —
        ///     the checks run in NumPy's order (contiguity/writability → dtype → shape) and use its
        ///     verbatim messages, and a C- OR F-contiguous writable array is accepted (the random fills
        ///     write the raw buffer sequentially, so an F-contiguous <c>out</c> receives the same bytes
        ///     NumPy would write).
        /// </remarks>
        private static void ValidateOut(NDArray @out, Shape size, NPTypeCode loopType, string name)
        {
            if (!((@out.Shape.IsContiguous || @out.Shape.IsFContiguous) && @out.Shape.IsWriteable))
                throw new ValueError("Supplied output array must be contiguous, writable, aligned, and in machine byte-order.");
            if (@out.GetTypeCode != loopType)
                throw new TypeError($"Supplied output array has the wrong type. Expected {loopType.AsNumpyDtypeName()}, got {@out.GetTypeCode.AsNumpyDtypeName()}");
            if (!IsNoSize(size) && !size.Equals(@out.Shape))
                throw new ValueError("size must match out.shape when used together");
        }
    }
}
