namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Translates slice expressions to concatenation along the SECOND axis.
        /// </summary>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.c_</c>. Short-hand for <c>np.r_["-1,2,0", …]</c>: entries are
        ///     upgraded to at least 2-D with the 1s POST-pended (so a 1-D entry becomes a COLUMN vector,
        ///     <c>(N,)</c> → <c>(N, 1)</c>) and then stacked along their last axis. Entries that are already
        ///     2-D or higher pass through untouched.
        ///     <para>
        ///     Every leading-directive, slice-expression and weak-scalar rule of <see cref="RClass"/> applies
        ///     here too — <c>np.c_["0:3", "3:6"]</c> is <c>[[0 3] [1 4] [2 5]]</c>. See
        ///     <see cref="AxisConcatenator"/> for how a Python slice literal is spelled in C#.
        ///     </para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.c_.html
        /// </remarks>
        /// <example>
        /// <code>
        /// np.c_[np.array(new[] {1, 2, 3}), np.array(new[] {4, 5, 6})];  // [[1 4] [2 5] [3 6]]
        /// np.c_[np.array(new[,] {{1, 2, 3}}), 0, 0, np.array(new[,] {{4, 5, 6}})];  // [[1 2 3 0 0 4 5 6]]
        /// </code>
        /// </example>
        public sealed class CClass : AxisConcatenator
        {
            internal CClass() : base(-1, ndmin: 2, trans1d: 0)
            {
            }
        }

        /// <summary>
        ///     Builds arrays by stacking columns — see <see cref="CClass"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.c_.html</remarks>
        public static CClass c_ { get; } = new CClass();
    }
}
