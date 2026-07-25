using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     A nicer way to build up index tuples for arrays — the type behind <see cref="s_"/> and
        ///     <see cref="index_exp"/>. Use those two instances rather than constructing this directly.
        /// </summary>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.lib.index_tricks.IndexExpression</c>. For any index combination,
        ///     <c>a[indices]</c> equals <c>a[np.index_exp[indices]]</c>, but <c>np.s_[…]</c> can be stored in
        ///     a variable, passed around and reused.
        ///     <para>
        ///     <b>C# divergence — <c>s_</c> and <c>index_exp</c> return the same thing.</b> NumPy's two
        ///     instances differ only in <c>maketuple</c>: <c>np.s_[2::2]</c> yields the bare
        ///     <c>slice(2, None, 2)</c> while <c>np.index_exp[2::2]</c> wraps it as <c>(slice(2, None, 2),)</c>.
        ///     C# has no such distinction at the index site — <see cref="NDArray"/>'s indexer is
        ///     <c>this[params Slice[]]</c>, so a lone <see cref="Slice"/> and a one-element
        ///     <see cref="Slice"/>[] are literally the same call. Both instances therefore return
        ///     <see cref="Slice"/>[]; the pair is kept so NumPy code ports across verbatim.
        ///     </para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.s_.html
        /// </remarks>
        public sealed class IndexExpression
        {
            /// <summary>
            ///     True for <see cref="index_exp"/> (NumPy always returns a tuple), false for
            ///     <see cref="s_"/>. Kept to mirror NumPy's constructor; it has no observable effect in C#
            ///     because a <see cref="Slice"/>[] is already how NumSharp spells an index tuple.
            /// </summary>
            public bool maketuple { get; }

            internal IndexExpression(bool maketuple) => this.maketuple = maketuple;

            /// <summary>
            ///     Parses Python index notation — comma-separated slices, indices, <c>...</c> and
            ///     <c>newaxis</c> — into the index tuple NumSharp's array indexer consumes.
            /// </summary>
            /// <param name="item">e.g. <c>"2::2"</c>, <c>":, 0"</c>, <c>"..., ::-1"</c>, <c>"np.newaxis, 3:"</c>.</param>
            /// <exception cref="ArgumentException">The notation is malformed.</exception>
            public Slice[] this[string item] => Slice.ParseSlices(item);

            /// <summary>
            ///     Builds an index tuple out of already-constructed <see cref="Slice"/> objects — the
            ///     programmatic spelling of the string form.
            /// </summary>
            public Slice[] this[params Slice[] item] => item ?? Array.Empty<Slice>();
        }

        /// <summary>
        ///     Builds reusable index tuples: <c>np.s_["2::2"]</c> returns the index itself rather than
        ///     applying it, so it can be stored and handed to <c>arr[…]</c> later.
        /// </summary>
        /// <example>
        /// <code>
        /// var every2nd = np.s_["2::2"];
        /// np.array(new[] {0, 1, 2, 3, 4})[every2nd];  // array([2, 4])
        /// </code>
        /// </example>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.s_.html</remarks>
        public static IndexExpression s_ { get; } = new IndexExpression(maketuple: false);

        /// <summary>
        ///     NumPy's always-a-tuple twin of <see cref="s_"/>. Identical in C# — see
        ///     <see cref="IndexExpression"/> for why.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.index_exp.html</remarks>
        public static IndexExpression index_exp { get; } = new IndexExpression(maketuple: true);
    }
}
