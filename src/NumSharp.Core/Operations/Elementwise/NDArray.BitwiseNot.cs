namespace NumSharp
{
    /// <summary>
    ///     Bitwise NOT (invert) operator for NDArray.
    ///     Delegates to np.invert for element-wise bit inversion.
    /// </summary>
    public partial class NDArray
    {
        /// <summary>
        /// Element-wise bitwise NOT (invert) operation.
        /// For boolean arrays: logical NOT (~True = False, ~False = True).
        /// For integer arrays: bitwise NOT (~0 = -1, ~1 = -2, etc.).
        /// </summary>
        /// <remarks>
        /// Matches NumPy's ~ operator behavior:
        /// - Boolean: ~arr is equivalent to np.logical_not(arr)
        /// - Integer: ~arr is equivalent to np.invert(arr)
        /// </remarks>
        public static NDArray operator ~(NDArray x)
        {
            return np.invert(x);
        }
    }
}
