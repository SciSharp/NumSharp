using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp {
    public static partial class np
    {
        /// <summary>
        /// Test whether all array elements evaluate to True.
        /// </summary>
        /// <param name="nd"></param>
        /// <returns></returns>
        public static bool all(NDArray nd)
            => BackendFactory.GetEngine().All(nd);

        /// <summary>
        /// Test whether all array elements along a given axis evaluate to True.
        /// </summary>
        /// <param name="nd"></param>
        /// <param name="axis"></param>
        /// <returns>Returns an array of bools</returns>
        public static NDArray<bool> all(NDArray nd, int axis)
            => BackendFactory.GetEngine().All(nd, axis);
    }
}