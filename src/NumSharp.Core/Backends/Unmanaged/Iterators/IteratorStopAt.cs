namespace NumSharp.Backends.Unmanaged {
    public enum IteratorStopAt
    {
        /// <summary>
        ///     Stop when rhs has reached the end / has no next, lhs will *NOT* be set to auto-reset
        /// </summary>
        OnlyRhs,

        /// <summary>
        ///     Stop when rhs has reached the end / has no next, lhs will be set to auto-reset
        /// </summary>
        Rhs,

        /// <summary>
        ///     Stop when lhs has reached the end / has no next, rhs will be set to auto-reset
        /// </summary>
        Lhs,

        /// <summary>
        ///     Stop when any reach the end / has no next, none is set to auto-reset. This is the default.
        /// </summary>
        Any,

        /// <summary>
        ///     Stop when both has reach the end / has no next, none is set to auto-reset
        /// </summary>
        Both,
    }
}