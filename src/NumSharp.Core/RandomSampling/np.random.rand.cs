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
        ///     NumPy signature: rand(d0, d1, ..., dn) where d0..dn are dimension sizes.
        /// </remarks>
        public NDArray rand(params long[] shape)
        {
            // A null params array only arises from an explicit null: no dimensions at all, like the empty call (it used
            // to dereference the null and throw NullReferenceException).
            if (shape is null || shape.Length == 0)
                lock (randomizer.@lock)
                    return NDArray.Scalar(randomizer.NextDouble());
            return rand(new Shape(shape));
        }

        /// <summary>
        ///     Random values in a given shape.
        /// </summary>
        /// <param name="shape">Shape of the returned array; <c>default</c> (no dimensions at all — NumPy's <c>rand()</c>)
        ///     draws a single value, returned 0-d, exactly as <see cref="rand(long[])"/> with no dimensions.</param>
        /// <returns>Random values.</returns>
        /// <exception cref="ValueError">A dimension is negative.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        ///     <br/>
        ///     Create an array of the given shape and populate it with random samples
        ///     from a uniform distribution over [0, 1).
        ///     <br/>
        ///     NumPy's <c>rand</c> takes loose dimensions (<c>rand(d0, …, dn)</c>); this overload spreads a shape into
        ///     them, so <see cref="Shape.Scalar"/> and <c>default</c> are both <c>rand()</c> — one draw.
        /// </remarks>
        public NDArray rand(Shape shape)
        {
            // default(Shape) carries no dimensions to allocate from; it is rand() — the single-draw path.
            if (shape.IsEmpty)
                return rand();

            // A fresh C-contiguous array of the requested dimensions: a view's Shape (strides/offset) must not leak
            // into the allocation, and NumPy's random_sample always fills a new C-order array.
            NDArray ret = new NDArray(typeof(double), shape.IsEmpty ? shape : new Shape(shape.dimensions), false);

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
        ///     Return random floats in the half-open interval [0.0, 1.0).
        /// </summary>
        /// <param name="size">Output shape; none, or a null array (Python's <c>None</c>), is <c>size=None</c> — one draw, 0-d.</param>
        /// <returns>Array of random floats of shape size.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_sample.html
        ///     <br/>
        ///     Results are from the "continuous uniform" distribution over the stated interval.
        ///     To sample Unif[a, b), b > a, multiply the output by (b-a) and add a.
        /// </remarks>
        public NDArray random_sample(params long[] size) => rand(size);

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0).
        /// </summary>
        /// <param name="size">Output shape; none, or a null array (Python's <c>None</c>), is <c>size=None</c> — one draw, 0-d.</param>
        /// <returns>Array of random floats.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.random.html
        ///     <br/>
        ///     Alias for random_sample.
        /// </remarks>
        public NDArray random(params long[] size) => random_sample(size);

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0) — NumPy's legacy module-level alias
        ///     <c>np.random.ranf</c> of <see cref="random_sample"/>.
        /// </summary>
        /// <param name="size">Output shape (none, or a null array — Python's <c>None</c> — for a single value).</param>
        /// <returns>Array of random floats of shape size (0-d when no size is given).</returns>
        /// <exception cref="ValueError">A dimension is negative.</exception>
        /// <remarks>
        ///     NumPy defines <c>ranf</c> (and <see cref="sample"/>) on the <c>np.random</c> module only, not on
        ///     <c>RandomState</c>; NumSharp's <c>np.random</c> IS a <see cref="NumPyRandom"/>, so the alias lives here and is
        ///     also reachable from other instances. Same stream as <see cref="random_sample"/>.
        /// </remarks>
        public NDArray ranf(params long[] size) => random_sample(size);

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0) — NumPy's legacy module-level alias
        ///     <c>np.random.sample</c> of <see cref="random_sample"/>.
        /// </summary>
        /// <param name="size">Output shape (none, or a null array — Python's <c>None</c> — for a single value).</param>
        /// <returns>Array of random floats of shape size (0-d when no size is given).</returns>
        /// <exception cref="ValueError">A dimension is negative.</exception>
        /// <remarks>Module-level in NumPy (see <see cref="ranf"/>); same stream as <see cref="random_sample"/>.</remarks>
        public NDArray sample(params long[] size) => random_sample(size);

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0) — NumPy's <c>random_sample(size)</c> with the size
        ///     given as ONE shape argument, the way NumPy takes it (<c>size=(2, 3)</c> is <c>new Shape(2, 3)</c>).
        /// </summary>
        /// <param name="size">Output shape: <c>default</c> is NumPy's <c>size=None</c> (a single value, returned 0-d);
        ///     <see cref="Shape.Scalar"/> is <c>size=()</c> (a 0-d array); anything else the array's dimensions.</param>
        /// <returns>The draws, one per element in C order (the stream <see cref="random_sample(long[])"/> consumes).</returns>
        /// <exception cref="ValueError">A dimension is negative, or the array is too big to address.</exception>
        /// <exception cref="OutOfMemoryException">A valid size that cannot be allocated (NumPy's <c>MemoryError</c>).</exception>
        /// <remarks>
        ///     NumPy's <c>random_sample(size=None)</c> takes one <c>size</c> — an int or a tuple — not the loose dimensions
        ///     <see cref="rand(long[])"/> takes; the <c>params long[]</c> overload is NumSharp's convenience spelling of the
        ///     same call, this one its NumPy shape (an <c>int[]</c>/<c>long[]</c> tuple converts to <see cref="Shape"/>).
        /// </remarks>
        public NDArray random_sample(Shape size) => size.IsEmpty ? random_sample() : rand(size);

        /// <summary>NumPy's <c>random(size)</c> (an alias of <see cref="random_sample(Shape)"/>) with the size as one shape.</summary>
        /// <param name="size">Output shape (see <see cref="random_sample(Shape)"/>).</param>
        /// <returns>The draws.</returns>
        /// <exception cref="ValueError">A dimension is negative, or the array is too big to address.</exception>
        /// <exception cref="OutOfMemoryException">A valid size that cannot be allocated.</exception>
        public NDArray random(Shape size) => random_sample(size);

        /// <summary>NumPy's module-level <c>ranf(size)</c> (an alias of <see cref="random_sample(Shape)"/>) with the size as one shape.</summary>
        /// <param name="size">Output shape (see <see cref="random_sample(Shape)"/>).</param>
        /// <returns>The draws.</returns>
        /// <exception cref="ValueError">A dimension is negative, or the array is too big to address.</exception>
        /// <exception cref="OutOfMemoryException">A valid size that cannot be allocated.</exception>
        public NDArray ranf(Shape size) => random_sample(size);

        /// <summary>NumPy's module-level <c>sample(size)</c> (an alias of <see cref="random_sample(Shape)"/>) with the size as one shape.</summary>
        /// <param name="size">Output shape (see <see cref="random_sample(Shape)"/>).</param>
        /// <returns>The draws.</returns>
        /// <exception cref="ValueError">A dimension is negative, or the array is too big to address.</exception>
        /// <exception cref="OutOfMemoryException">A valid size that cannot be allocated.</exception>
        public NDArray sample(Shape size) => random_sample(size);
    }
}
