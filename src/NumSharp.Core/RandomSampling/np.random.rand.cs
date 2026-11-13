namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Random values in a given shape.
        /// </summary>
        /// <param name="shape">Dimensions of the returned array (d0, d1, ..., dn); none — or a null array, the port of
        ///     Python's <c>None</c> — is NumPy's <c>rand()</c>: one draw, returned 0-d.</param>
        /// <returns>Random values.</returns>
        /// <exception cref="ValueError">A dimension is negative.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        ///     <br/>
        ///     Create an array of the given shape and populate it with random samples
        ///     from a uniform distribution over [0, 1).
        ///     <br/>
        ///     NumPy signature: <c>rand(d0, d1, ..., dn)</c>, loose dimensions only - NumPy's <c>rand((2, 3))</c> is a
        ///     TypeError (<c>'tuple' object cannot be interpreted as an integer</c>), so since API-parity batch 6 there is no
        ///     <see cref="Shape"/> overload either: <c>rand(2, 3)</c>, or <see cref="random_sample(Shape?)"/> for a shape
        ///     value.
        /// </remarks>
        public NDArray rand(params long[] shape)
        {
            // A null params array only arises from an explicit null: no dimensions at all, like the empty call (it used
            // to dereference the null and throw NullReferenceException).
            if (shape is null || shape.Length == 0)
                lock (randomizer.@lock)
                    return NDArray.WritableScalar(randomizer.NextDouble());
            return FillUniform(new Shape(shape));
        }

        /// <summary>
        ///     The [0, 1) fill behind <see cref="rand(long[])"/>, <see cref="random_sample(Shape?)"/> and
        ///     np.random.choice's uniform draws: a fresh C-contiguous float64 array of <paramref name="shape"/>, filled in
        ///     memory (= C) order under the bit generator's lock, as NumPy's <c>double_fill</c> does.
        /// </summary>
        /// <param name="shape">The dimensions; <c>default</c> (no dimensions at all) is one draw, returned 0-d, as
        ///     <c>rand()</c> is.</param>
        /// <returns>The draws.</returns>
        /// <exception cref="ValueError">A dimension is negative, or the array is too big to address.</exception>
        /// <exception cref="MemoryError">A valid shape that cannot be allocated (NumPy's <c>MemoryError</c>).</exception>
        internal NDArray FillUniform(Shape shape)
        {
            // default(Shape) carries no dimensions to allocate from; it is rand() — the single-draw path.
            if (shape.IsEmpty)
                return rand();

            // A fresh C-contiguous array of the requested dimensions: a view's Shape (strides/offset, or a dense F order)
            // must not leak into the allocation, and NumPy's random_sample always fills a new C-order array. AsClean keeps
            // a shape that already is that layout, so the common call allocates no second Shape.
            NDArray ret = new NDArray(typeof(double), shape.AsClean(), false);

            // Handle empty arrays (any dimension is 0)
            if (ret.size == 0)
                return ret;

            unsafe
            {
                // One bulk fill in memory (= C) order, holding the engine lock as NumPy's double_fill does.
                lock (randomizer.@lock)
                    randomizer.FillDouble((double*)ret.Address, ret.size);
            }

            return ret;
        }

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0): NumPy's <c>random_sample(size=None)</c>, ONE size
        ///     argument (an int or a tuple), never loose dimensions.
        /// </summary>
        /// <param name="size">Output shape: <c>null</c> (NumPy's <c>None</c>, the default) is one draw, returned 0-d;
        ///     <see cref="Shape.Scalar"/> is <c>size=()</c> (a 0-d array); an int, an <c>int[]</c> / <c>long[]</c> or a tuple
        ///     literal converts to the array's dimensions.</param>
        /// <returns>The draws, one per element in C order.</returns>
        /// <exception cref="ValueError">A dimension is negative, or the array is too big to address.</exception>
        /// <exception cref="MemoryError">A valid size that cannot be allocated (NumPy's <c>MemoryError</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_sample.html
        ///     <br/>
        ///     Results are from the "continuous uniform" distribution over the stated interval. To sample Unif[a, b), b > a,
        ///     multiply the output by (b-a) and add a.
        ///     <br/>
        ///     One signature since API-parity batch 6: the <c>params long[]</c> form went, as NumPy's
        ///     <c>random_sample(2, 3)</c> is a TypeError (<c>random_sample() takes at most 1 positional argument (2
        ///     given)</c>) - write <c>random_sample((2, 3))</c>.
        /// </remarks>
        public NDArray random_sample(Shape? size = null)
            // null and default(Shape) are None (one draw); anything else, Shape.Scalar included, is the requested array.
            => size is not { } given || given.IsEmpty ? rand() : FillUniform(given);

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0): NumPy's <c>random(size=None)</c>, an alias of
        ///     <see cref="random_sample(Shape?)"/> (the same stream, the same one size argument).
        /// </summary>
        /// <param name="size">Output shape: <c>null</c> (NumPy's <c>None</c>, the default) is one draw, returned 0-d.</param>
        /// <returns>Array of random floats.</returns>
        /// <exception cref="ValueError">A dimension is negative, or the array is too big to address.</exception>
        /// <exception cref="MemoryError">A valid size that cannot be allocated.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/random/generated/numpy.random.random.html</remarks>
        public NDArray random(Shape? size = null) => random_sample(size);

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0) — NumPy's legacy module-level alias
        ///     <c>np.random.ranf</c> of <see cref="random_sample(Shape?)"/>.
        /// </summary>
        /// <param name="size">Output shape: <c>null</c> (NumPy's <c>None</c>, the default) is one draw, returned 0-d.</param>
        /// <returns>Array of random floats of shape size (0-d when no size is given).</returns>
        /// <exception cref="ValueError">A dimension is negative.</exception>
        /// <exception cref="MemoryError">A valid size that cannot be allocated.</exception>
        /// <remarks>
        ///     NumPy defines <c>ranf</c> (and <see cref="sample"/>) on the <c>np.random</c> module only, not on
        ///     <c>RandomState</c>; NumSharp's <c>np.random</c> IS a <see cref="NumPyRandom"/>, so the alias lives here and is
        ///     also reachable from other instances. NumPy's <c>ranf(*args, **kwargs)</c> forwards to random_sample, whose one
        ///     size argument it therefore takes. Same stream as <see cref="random_sample(Shape?)"/>.
        /// </remarks>
        public NDArray ranf(Shape? size = null) => random_sample(size);

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0) — NumPy's legacy module-level alias
        ///     <c>np.random.sample</c> of <see cref="random_sample(Shape?)"/>.
        /// </summary>
        /// <param name="size">Output shape: <c>null</c> (NumPy's <c>None</c>, the default) is one draw, returned 0-d.</param>
        /// <returns>Array of random floats of shape size (0-d when no size is given).</returns>
        /// <exception cref="ValueError">A dimension is negative.</exception>
        /// <exception cref="MemoryError">A valid size that cannot be allocated.</exception>
        /// <remarks>Module-level in NumPy (see <see cref="ranf"/>); forwards to random_sample's one size argument; same
        ///     stream as <see cref="random_sample(Shape?)"/>.</remarks>
        public NDArray sample(Shape? size = null) => random_sample(size);
    }
}
