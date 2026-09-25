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
    ///     from any <see cref="BitGenerator"/> — <see cref="PCG64"/> by default, or <see cref="PCG64DXSM"/>,
    ///     <see cref="Philox"/>, <see cref="SFC64"/>, <see cref="MT19937"/> — and uses NumPy's newer algorithms —
    ///     ziggurat normal/exponential and Lemire bounded integers — so its stream matches
    ///     <c>np.random.Generator(bit_generator)</c> (and <c>default_rng(seed)</c>) bit-for-bit, not <c>RandomState</c>.
    ///     <para>
    ///     Every public drawing method holds the bit generator's <see cref="BitGenerator.@lock"/> while it
    ///     consumes the stream, so one generator (or several over the same bit generator) can be shared
    ///     between threads with NumPy's guarantee: each call is atomic with respect to the stream.
    ///     </para>
    /// </remarks>
    public sealed partial class Generator
    {
        private readonly BitGenerator _bitGenerator;

        /// <summary>Constructs a Generator over the given bit generator.</summary>
        /// <param name="bitGenerator">The bit generator supplying the stream; its lock is shared with every other consumer.</param>
        /// <exception cref="ArgumentNullException"><paramref name="bitGenerator"/> is null.</exception>
        public Generator(BitGenerator bitGenerator)
        {
            _bitGenerator = bitGenerator ?? throw new ArgumentNullException(nameof(bitGenerator));
        }

        /// <summary>The bit generator supplying this Generator's stream.</summary>
        public BitGenerator bit_generator => _bitGenerator;

        /// <summary>
        ///     Create new independent child generators (NumPy's <c>Generator.spawn</c>): each wraps a child of
        ///     <see cref="bit_generator"/>'s seed sequence, via <see cref="BitGenerator.spawn"/> — the recommended way to
        ///     hand non-overlapping streams to parallel workers.
        /// </summary>
        /// <param name="n_children">The number of children.</param>
        /// <returns>The children, each a <see cref="Generator"/> over a fresh bit generator of the same type.</returns>
        /// <exception cref="TypeError">The bit generator's seed sequence cannot spawn (a legacy-seeded <see cref="MT19937"/>, a
        /// keyed <see cref="Philox"/>): <c>The underlying SeedSequence does not implement spawning.</c></exception>
        /// <exception cref="OverflowException">The seed sequence's child count would leave its uint32 range (e.g. a negative count).</exception>
        /// <remarks>
        ///     Spawning advances the seed sequence's <see cref="SeedSequence.n_children_spawned"/>, so it never repeats a
        ///     child; it does NOT consume this generator's stream (the children are seeded, not drawn).
        /// </remarks>
        public Generator[] spawn(int n_children)
        {
            BitGenerator[] children = _bitGenerator.spawn(n_children);
            var result = new Generator[children.Length];
            for (int i = 0; i < children.Length; i++)
                result[i] = new Generator(children[i]);
            return result;
        }

        /// <summary>NumPy's <c>str(Generator)</c>: the class name and the bit generator's name.</summary>
        /// <returns><c>Generator(&lt;bit generator name&gt;)</c>, e.g. <c>Generator(PCG64)</c>.</returns>
        public override string ToString() => $"Generator({_bitGenerator.Name})";

        // ---- shared output helpers (random output is always a fresh C-contiguous owning array) ----

        /// <summary>
        ///     True when <paramref name="size"/> is NumPy's <c>size=None</c> — the uninitialized
        ///     <c>default(Shape)</c>, which asks for a single SCALAR draw.
        /// </summary>
        /// <param name="size">The size argument as received.</param>
        /// <returns>True only for the default (unset) shape.</returns>
        /// <remarks>
        ///     <c>size=()</c> (<see cref="Shape.Scalar"/>) is deliberately NOT "no size": NumPy fills a 0-d
        ///     ARRAY for it (<c>np.empty(())</c>), which keeps a float32 dtype, while <c>size=None</c> returns a
        ///     Python scalar and widens a float32 draw to float64. Treating the two alike made
        ///     <c>random((), dtype=float32)</c> return float64. Both still consume exactly one draw.
        /// </remarks>
        private static bool IsNoSize(Shape size) => size.IsEmpty;

        /// <summary>Allocates <paramref name="shape"/> as float64 and fills it with <c>next_double</c> draws.</summary>
        /// <param name="shape">The output shape (may be 0-d or empty).</param>
        /// <param name="bg">The bit generator to draw from; the caller holds its lock.</param>
        /// <returns>A fresh C-contiguous float64 array.</returns>
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

        /// <summary>Allocates <paramref name="shape"/> as float32 and fills it with <c>next_float</c> draws.</summary>
        /// <param name="shape">The output shape (may be 0-d or empty).</param>
        /// <param name="bg">The bit generator to draw from; the caller holds its lock.</param>
        /// <returns>A fresh C-contiguous float32 array.</returns>
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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value; <c>()</c> returns a 0-d array.</param>
        /// <param name="dtype">Desired dtype — only native <c>float64</c> (default) and <c>float32</c> are supported.</param>
        /// <param name="out">Optional C- or F-contiguous output array to fill in memory order; returned when given.</param>
        /// <returns>The draws; a float64 0-d array for <c>size=None</c> (NumPy widens a float32 scalar draw to a Python float).</returns>
        /// <exception cref="TypeError"><paramref name="dtype"/> is not a native float32/float64, or <paramref name="out"/> has the wrong dtype.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is not contiguous/writeable, or its shape disagrees with <paramref name="size"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.random.html</remarks>
        public NDArray random(Shape size = default, DType dtype = null, NDArray @out = null)
        {
            dtype ??= DType.Double;
            NPTypeCode tc = ResolveFloatDtype(dtype, "random");

            lock (_bitGenerator.@lock)
            {
                if (@out is not null)
                {
                    ValidateOut(@out, size, tc, requireCContiguous: false);
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
        }

        /// <summary>
        ///     Resolves the float dtype of <c>random</c>/<c>standard_normal</c>/<c>standard_exponential</c>/
        ///     <c>standard_gamma</c> the way NumPy's <c>_dtype == np.float64 / np.float32</c> dispatch does.
        /// </summary>
        /// <param name="dtype">The requested dtype.</param>
        /// <param name="name">The method name used in the error text.</param>
        /// <returns><see cref="NPTypeCode.Double"/> or <see cref="NPTypeCode.Single"/>.</returns>
        /// <exception cref="TypeError">The dtype is not a NATIVE float64/float32 — NumPy's equality test fails for a
        /// byte-swapped descriptor too, so <c>'&gt;f8'</c> is reported with its full repr (<c>dtype('&gt;f8')</c>).</exception>
        private static NPTypeCode ResolveFloatDtype(DType dtype, string name)
        {
            var tc = dtype.GetTypeCode();
            if (dtype.isnative && (tc == NPTypeCode.Double || tc == NPTypeCode.Single))
                return tc;
            throw new TypeError($"Unsupported dtype {dtype.ToString(true)} for {name}");
        }

        /// <summary>Fills a validated <c>out</c> array in memory order with <c>[0, 1)</c> draws.</summary>
        /// <param name="out">The destination (contiguous, writeable, matching dtype — already validated).</param>
        /// <param name="tc">The loop dtype (float64 or float32).</param>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        private unsafe void FillRandomInto(NDArray @out, NPTypeCode tc)
        {
            long n = @out.size;
            if (n == 0)
                return;
            // NumPy fills PyArray_DATA(out) sequentially, i.e. in MEMORY order — for an F-contiguous out
            // that is column-major, which is why the base-plus-offset pointer (not a logical walk) is used.
            if (tc == NPTypeCode.Single)
            {
                var p = (float*)(@out.Storage.Address + @out.Shape.offset * sizeof(float));
                for (long i = 0; i < n; i++) p[i] = _bitGenerator.NextFloat();
            }
            else
            {
                var p = (double*)(@out.Storage.Address + @out.Shape.offset * sizeof(double));
                for (long i = 0; i < n; i++) p[i] = _bitGenerator.NextDouble();
            }
        }

        /// <summary>
        ///     Validates an <c>out=</c> array against the requested size and loop dtype (NumPy's
        ///     <c>check_output</c> in <c>_common.pyx</c>).
        /// </summary>
        /// <param name="out">The user-supplied output array.</param>
        /// <param name="size">The size argument (<c>None</c> skips the shape check; <c>()</c> requires a 0-d out).</param>
        /// <param name="loopType">The dtype the fill writes.</param>
        /// <param name="requireCContiguous">
        ///     NumPy's <c>require_c_array</c>: false for the plain fills (<c>double_fill</c>/<c>float_fill</c> —
        ///     C- or F-contiguous accepted, written in memory order), true for the distribution path
        ///     (<c>cont</c>/<c>cont_f</c>, used by <c>standard_gamma</c>) which only accepts a C-contiguous out.
        /// </param>
        /// <exception cref="ValueError">The out is not (C-)contiguous or not writeable, or its shape disagrees with <paramref name="size"/>.</exception>
        /// <exception cref="TypeError">The out's dtype is not <paramref name="loopType"/>.</exception>
        /// <remarks>
        ///     The checks run in NumPy's order (contiguity/writability → dtype → shape) with its verbatim
        ///     messages, so a call that is wrong in several ways reports the same first error NumPy does.
        /// </remarks>
        private static void ValidateOut(NDArray @out, Shape size, NPTypeCode loopType, bool requireCContiguous)
        {
            bool layoutOk = @out.Shape.IsContiguous || (!requireCContiguous && @out.Shape.IsFContiguous);
            if (!(layoutOk && @out.Shape.IsWriteable))
            {
                string req = requireCContiguous ? "C-" : "";
                throw new ValueError($"Supplied output array must be {req}contiguous, writable, aligned, and in machine byte-order.");
            }
            if (@out.GetTypeCode != loopType)
                throw new TypeError($"Supplied output array has the wrong type. Expected {loopType.AsNumpyDtypeName()}, got {@out.GetTypeCode.AsNumpyDtypeName()}");
            if (!IsNoSize(size) && !size.Equals(@out.Shape))
                throw new ValueError("size must match out.shape when used together");
        }

        /// <summary>
        ///     NumPy's <c>CONS_NON_NEGATIVE</c> constraint (<c>_common.pyx check_constraint</c>):
        ///     <c>not isnan(v) and signbit(v)</c> raises <c>ValueError("&lt;name&gt; &lt; 0")</c>.
        /// </summary>
        /// <param name="value">The parameter value.</param>
        /// <param name="name">The parameter name used in the message (e.g. <c>scale</c>, <c>high - low</c>).</param>
        /// <exception cref="ValueError"><paramref name="value"/> has its sign bit set and is not NaN.</exception>
        /// <remarks>
        ///     The test is the SIGN BIT, not <c>value &lt; 0</c>: <c>-0.0</c> is rejected (a plain comparison
        ///     let it through and drew a value), while a NaN of either sign passes and propagates into the draw.
        /// </remarks>
        private static void CheckNonNegative(double value, string name)
        {
            if (!double.IsNaN(value) && double.IsNegative(value))
                throw new ValueError($"{name} < 0");
        }
    }
}
