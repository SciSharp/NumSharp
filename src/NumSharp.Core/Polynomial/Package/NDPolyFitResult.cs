using NumSharp.Backends;

namespace NumSharp
{
    /// <summary>
    ///     The result of a <c>numpy.polynomial</c> least-squares fit (<c>polyfit</c>, <c>chebfit</c>, <c>legfit</c>,
    ///     <c>lagfit</c>, <c>hermfit</c>, <c>hermefit</c>): the fitted coefficient array ITSELF — an <see cref="NDArray"/>, so
    ///     it indexes, computes and passes to every function (<c>chebval(x, chebfit(x, y, 3))</c>) exactly like NumPy's
    ///     ndarray — that also carries NumPy's <c>full=True</c> statistics.
    /// </summary>
    /// <remarks>
    ///     <para>NumPy returns the coefficient ndarray, or with <c>full=True</c> the tuple <c>(c, [resids, rank, sv,
    ///     rcond])</c>. C# cannot change a method's return type with a flag, so the one result stands for both:
    ///     <c>NDArray c = P.polyfit(x, y, 3)</c> and <c>var c = P.polyfit(x, y, 3)</c> are the coefficients, and
    ///     <c>var (c, (resids, rank, sv, rcond)) = P.polyfit(x, y, 3, full: true)</c> deconstructs as NumPy's tuple (or
    ///     <c>var (c, info) = …</c>). Without <c>full</c> the four statistics are null and deconstruction is refused, as
    ///     NumPy's ndarray does not unpack into <c>(c, info)</c> either.</para>
    ///     <para><b>Lifetime.</b> Disposing the result releases the coefficients only. The statistics are separate arrays the
    ///     result owns: no <see cref="NDScope"/> tracks them (a scope that yields the result would yield only the coefficient
    ///     array, and would reclaim scope-tracked statistics under it), so they are released by the garbage collector, or by
    ///     disposing them explicitly.</para>
    ///     <para>Its memory is NumPy's: for a scalar degree the coefficients are a VIEW of the computed quotient (OWNDATA
    ///     false), for a degree list a new array that owns its data.</para>
    /// </remarks>
    public sealed class NDPolyFitResult : NDArray
    {
        /// <summary>Wraps the coefficient storage and the statistics.</summary>
        /// <param name="storage">The coefficients' storage (an alias for a view, or the owner's storage taken over).</param>
        /// <param name="residuals">lstsq's residuals, or null without <c>full</c>.</param>
        /// <param name="rank">lstsq's rank (a 0-d int32, NumPy's np.intc), or null without <c>full</c>.</param>
        /// <param name="singularValues">lstsq's singular values, or null without <c>full</c>.</param>
        /// <param name="rcond">The rcond used, or null without <c>full</c>.</param>
        internal NDPolyFitResult(UnmanagedStorage storage, NDArray residuals, NDArray rank, NDArray singularValues, NDArray rcond)
            : base(storage)
        {
            this.residuals = residuals;
            this.rank = rank;
            singular_values = singularValues;
            this.rcond = rcond;
        }

        /// <summary>
        ///     The sum of squared residuals of the fit, per data set (NumPy's <c>resids</c>): EMPTY unless the scaled design
        ///     matrix has full rank and more points than coefficients. Null unless the fit ran with <c>full: true</c>.
        /// </summary>
        public NDArray residuals { get; }

        /// <summary>The numerical rank of the scaled design matrix (a 0-d int32); null unless <c>full: true</c>.</summary>
        public NDArray rank { get; }

        /// <summary>The singular values of the scaled design matrix; null unless <c>full: true</c>.</summary>
        public NDArray singular_values { get; }

        /// <summary>
        ///     The rcond the fit used: the caller's value (an NDArray as given, a number as the 0-d array of its dtype), or the
        ///     default <c>len(x)*eps</c> of x's float dtype. Null unless <c>full: true</c>.
        /// </summary>
        public NDArray rcond { get; }

        /// <summary>
        ///     NumPy's <c>c, [resids, rank, sv, rcond] = …fit(…, full=True)</c>: the coefficients and the four statistics.
        /// </summary>
        /// <param name="c">The coefficients (this array).</param>
        /// <param name="info">(residuals, rank, singular values, rcond) — NumPy's list, in its order.</param>
        /// <exception cref="ValueError">The fit ran without <c>full</c> (NumPy's result is the bare coefficient array, which
        ///     does not unpack into <c>(c, info)</c>).</exception>
        public void Deconstruct(out NDArray c, out (NDArray residuals, NDArray rank, NDArray singular_values, NDArray rcond) info)
        {
            if (residuals is null)
                throw new ValueError("the fit ran without full=True: its result is the coefficient array alone");
            c = this;
            info = (residuals, rank, singular_values, rcond);
        }
    }
}
