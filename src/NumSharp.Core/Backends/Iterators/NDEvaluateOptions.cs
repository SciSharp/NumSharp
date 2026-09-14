namespace NumSharp.Backends.Iteration
{
    /// <summary>
    ///     The resolved <c>np.evaluate</c> keyword bundle (<c>dtype=</c> / <c>casting=</c> /
    ///     <c>order=</c>) threaded from the <see cref="np.evaluate(NDExpr, NDArray, DType, string, char)"/>
    ///     API boundary down to the engine, so the (already growing) <c>Evaluate</c> overloads gain ONE
    ///     parameter instead of three and every future keyword rides the same struct.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         The struct is deliberately built so that <c>default(NDEvaluateOptions)</c> reproduces
    ///         np.evaluate's pre-4.5 behaviour EXACTLY — no result cast, a <c>same_kind</c> out= rule, and
    ///         today's <c>'K'</c> layout heuristic. That is why the private fields are a NULLABLE casting
    ///         (raw <c>default(NPY_CASTING)</c> is <c>NPY_NO_CASTING</c>, the wrong default) and a NUL order
    ///         char rather than the enum/char values directly: every internal <c>EvaluateCore</c> call that
    ///         materialises a sub-tree passes <c>default</c> and MUST see the natural dtype/layout, never a
    ///         user override.
    ///     </para>
    ///     <para>
    ///         The API validates and RESOLVES each keyword before constructing this (a bad casting string or
    ///         order char raises at the boundary), so the engine only ever sees a legal, resolved bundle.
    ///     </para>
    /// </remarks>
    public readonly struct NDEvaluateOptions
    {
        /// <summary>
        ///     The requested result dtype (an implicit root cast of the whole expression), or <c>null</c>
        ///     for the tree's natural NEP50 result type. Mutually exclusive with a caller <c>out=</c>
        ///     (which already fixes the result dtype) — the API rejects both together.
        /// </summary>
        public readonly NPTypeCode? Dtype;

        /// <summary>The out= cast rule, or <c>null</c> to mean <see cref="NPY_CASTING.NPY_SAME_KIND_CASTING"/> (NumPy's ufunc default). Read through <see cref="Casting"/>.</summary>
        private readonly NPY_CASTING? _casting;

        /// <summary>The requested fresh-result memory order ('C'/'F'/'A'/'K'), or <c>'\0'</c> to mean <c>'K'</c> (today's layout heuristic). Read through <see cref="Order"/>.</summary>
        private readonly char _order;

        /// <summary>
        ///     Build a resolved keyword bundle. All three arguments are already validated by the API — an
        ///     unset value is expressed as <c>null</c> / <c>'\0'</c> so <c>default</c> stays the no-override
        ///     behaviour.
        /// </summary>
        /// <param name="dtype">The requested result dtype, or <c>null</c> for the natural result type.</param>
        /// <param name="casting">The out= cast rule, or <c>null</c> for <c>same_kind</c>.</param>
        /// <param name="order">The fresh-result order char, or <c>'\0'</c> for <c>'K'</c>.</param>
        public NDEvaluateOptions(NPTypeCode? dtype, NPY_CASTING? casting, char order)
        {
            Dtype = dtype;
            _casting = casting;
            _order = order;
        }

        /// <summary>True when a result dtype was requested (the implicit root cast is active).</summary>
        public bool HasDtype => Dtype.HasValue;

        /// <summary>The effective out= cast rule — the request, or <see cref="NPY_CASTING.NPY_SAME_KIND_CASTING"/> when unset.</summary>
        public NPY_CASTING Casting => _casting ?? NPY_CASTING.NPY_SAME_KIND_CASTING;

        /// <summary>The effective fresh-result order char — the request, or <c>'K'</c> when unset.</summary>
        public char Order => _order == '\0' ? 'K' : _order;

        /// <summary>
        ///     True when the user asked for a NON-default layout (anything but <c>'K'</c>). Used to reject
        ///     <c>order=</c> on a reduction tree, whose result layout the reduction fixes (NumPy's
        ///     reductions carry no <c>order=</c> parameter). A bare or explicit <c>'K'</c> is not "explicit"
        ///     — it is the no-op default, so it never trips the reduction guard.
        /// </summary>
        public bool HasExplicitOrder => _order != '\0' && _order != 'K' && _order != 'k';
    }
}
