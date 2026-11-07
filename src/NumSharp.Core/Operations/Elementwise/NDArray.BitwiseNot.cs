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
            // An operator keeps the plain-array meaning for a subclass object seen through an NDArray reference: np.invert
            // gets its plain view (handed the object itself it would take its subclass path and wrap the result). The lease
            // releases that view on return (np.invert's result is a new array); a plain x is passed through untouched.
            using var plain = NDSubclass.LeasePlainView(x);
            return np.invert(plain.Array);
        }
    }
}
