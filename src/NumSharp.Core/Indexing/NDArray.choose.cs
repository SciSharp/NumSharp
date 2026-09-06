using System.Numerics;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Use this index array to construct a new array from a set of choices. Refer to
        ///     <see cref="np.choose(NDArray,object[],NDArray,string)"/> for full documentation.
        /// </summary>
        /// <param name="choices">
        ///     The choice arrays (each an <see cref="NDArray"/> or a boxed C# scalar). This array
        ///     supplies the indices <c>[0, n-1]</c> into them.
        /// </param>
        /// <param name="out">Optional destination array whose shape equals the broadcast result shape.</param>
        /// <param name="mode">Out-of-bounds behaviour: <c>"raise"</c> (default), <c>"wrap"</c> or <c>"clip"</c>.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.choose.html</remarks>
        public NDArray choose(object[] choices, NDArray @out = null, string mode = "raise")
            => np.choose(this, choices, @out, mode);

        /// <summary>
        ///     Use this index array to choose from <paramref name="choices"/> (the common case — every
        ///     choice is an <see cref="NDArray"/>). Refer to
        ///     <see cref="np.choose(NDArray,NDArray[],NDArray,string)"/> for full documentation.
        /// </summary>
        public NDArray choose(NDArray[] choices, NDArray @out = null, string mode = "raise")
            => np.choose(this, choices, @out, mode);

        /// <summary>
        ///     Use this index array to choose from a single <paramref name="choices"/> array whose
        ///     outermost dimension is the sequence. Refer to
        ///     <see cref="np.choose(NDArray,NDArray,NDArray,string)"/> for full documentation.
        /// </summary>
        public NDArray choose(NDArray choices, NDArray @out = null, string mode = "raise")
            => np.choose(this, choices, @out, mode);
    }
}
