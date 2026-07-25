using System;
using System.Collections;

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
        ///     <b>The indexer surface mirrors <see cref="NDArray"/>'s.</b> NumPy's <c>s_</c> passes ANY
        ///     index object straight through — <c>np.s_[[1, 2]]</c> is the list, <c>np.s_[..., mask]</c>
        ///     the tuple — so anything writable as <c>arr[…]</c> must be writable as <c>np.s_[…]</c> too.
        ///     The three overloads below cover the same vocabulary <c>NDArray</c>'s three public indexers
        ///     do, and each returns the array type that indexer consumes: basic expressions (slices,
        ///     slice strings, integers) come back as <see cref="Slice"/>[], while anything advanced —
        ///     a fancy index array, a boolean mask, or a basic/advanced mix — comes back as
        ///     <c>object[]</c>. Either can be handed straight to <c>arr[…]</c>.
        ///     </para>
        ///     <para>
        ///     <b>C# divergence — <c>s_</c> and <c>index_exp</c> return the same thing.</b> NumPy's two
        ///     instances differ only in <c>maketuple</c>: <c>np.s_[2::2]</c> yields the bare
        ///     <c>slice(2, None, 2)</c> while <c>np.index_exp[2::2]</c> wraps it as <c>(slice(2, None, 2),)</c>.
        ///     C# has no such distinction at the index site — <c>NDArray</c>'s indexers are
        ///     <c>this[params Slice[]]</c> / <c>this[params object[]]</c>, so a lone <see cref="Slice"/>
        ///     and a one-element <see cref="Slice"/>[] are literally the same call. The pair is kept so
        ///     NumPy code ports across verbatim.
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
            ///     programmatic spelling of the string form. Integer and string literals convert
            ///     implicitly, so <c>np.s_[0]</c> and <c>np.s_["1:3", "::2"]</c> land here too.
            /// </summary>
            public Slice[] this[params Slice[] item] => item ?? Array.Empty<Slice>();

            /// <summary>
            ///     The ADVANCED-indexing form: captures any index expression <see cref="NDArray"/>'s
            ///     <c>this[params object[]]</c> accepts, so everything writable as <c>arr[…]</c> is
            ///     also writable as <c>np.s_[…]</c> and stored for later.
            /// </summary>
            /// <param name="item">
            ///     Any mix of <see cref="Slice"/>, slice-notation <see cref="string"/>, integer,
            ///     <see cref="NDArray"/> (integer fancy index or boolean mask), <c>int[]</c> /
            ///     <c>long[]</c>, <c>bool[]</c> (and rectangular <c>bool[,]</c>), <see cref="IList"/>
            ///     sequences, and <c>ITuple</c>s.
            /// </param>
            /// <remarks>
            ///     Overload resolution keeps the basic forms on the <see cref="Slice"/>[] overload —
            ///     a lone <c>Slice</c>, an <c>int</c> (which converts implicitly to <c>Slice</c>) and a
            ///     slice <c>string</c> all still return <see cref="Slice"/>[]. This overload takes over
            ///     the moment an entry cannot be a <c>Slice</c>: a fancy index array, a mask, or a
            ///     basic/advanced mix such as <c>np.s_[Slice.All, np.array(new[] {0, 2})]</c>. Both
            ///     return types feed <c>arr[…]</c> directly, including a
            ///     <see cref="Generic.NDArray{T}"/> (its own indexers do not hide the inherited
            ///     <c>this[params object[]]</c>).
            /// </remarks>
            public object[] this[params object[] item] => item ?? Array.Empty<object>();
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
