// NumPy's window functions (numpy/lib/_function_base_impl.py: bartlett, blackman, hamming, hanning, kaiser) as
// DefaultEngine's members. The facades are Math/np.windows.cs; the cephes i0 kernel kaiser applies per element stays on np
// beside np.i0's (np._i0Double, its scalar routine np.BesselI0 in np.windows.cs).
//
// Every window is a 1-D float64 taper of length M. NumPy forces the output to at least float64 by threading M through
// np.array([0.0, M]); M is a double (NumPy's _FloatLike_co), so a non-integer M reproduces NumPy's float-M behaviour: the
// length is len(arange(1-M, M, 2)), which can be a fractional count, and kaiser can end in a NaN once |(n-alpha)/alpha| > 1.
//
// Shared edge cases (the cosine windows bartlett / blackman / hamming / hanning):
//   - M < 1  -> an empty (0,) float64 array;
//   - M == 1 -> ones(1,) float64.
// kaiser has NO M < 1 guard: arange(0, M) is empty for M <= 0 but one element for 0 < M < 1.
//
// FUSION: NumPy builds each window as a chain of ufuncs over n = arange(1 - M, M, 2) (arange(0, M) for kaiser),
// materializing about six intermediate arrays. Here only n is materialized, and the whole elementwise transform is ONE
// pass of this engine's Evaluate (np.evaluate's member, its default options), so the buffer is read and written once. The
// fused tree is built in NumPy's exact operation order, which makes the result BIT-IDENTICAL to NumPy's unfused chain
// (verified across 152 (M, beta) cases and the differential-fuzz oracle): float64 add / subtract / multiply / divide are
// IEEE-exact, and float64 cos is the same scalar Math.Cos the CRT and NumPy both call. Measured NPY/NS about 1.8x-9x at
// 100K / 10M; at 1K a window sits at NumSharp's per-op NDIter-setup floor (window generation is compute-once).
//
// The M-dependent values (M-1, alpha, beta, i0(beta)) ride as 0-d float64 OPERANDS, never as literals: a literal is baked
// into the fused kernel's IL (and its cache key), so every distinct M - the one parameter a window is called with - would
// JIT its own kernel (about 1 ms) and hold its own program-cache entry. The operand form serves every M with ONE kernel per
// window kind; the arithmetic is the same float64 arithmetic on a value loaded from memory instead of an IL constant, so the
// result is bit-identical (a 0-d float64 operand promotes exactly like a weak float literal meeting a float64 array). The
// fixed cost of a stride-0 operand (about 0.25 us, NDIter-level) is the price; the JIT it replaces is 4000x it.

using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped /* the arange intermediate `n` and the 0-d M-1 are reclaimed; the fused result is yielded */]
        public override NDArray Bartlett(double M)
        {
            if (M < 1) return WindowEmpty();
            if (M == 1) return WindowOnes();

            // np.arange(1 - M, M, 2, float64)'s body (its device check has nothing to refuse without a device).
            var n = UnmanagedStorage.ArangeObj(1.0 - M, M, 2.0, np.float64);
            var e = (NDExpr)n;
            // M-1 as a 0-d operand (see the header): one kernel for every M. The same instance on both branches dedups to
            // one iterator stream.
            var m1 = (NDExpr)NewScalar(M - 1);
            return Evaluate(NDExpr.Where(NDExpr.LessEqual(e, 0.0),
                                         1.0 + e / m1,
                                         1.0 - e / m1));
        }

        /// <inheritdoc />
        [NDScoped]
        public override NDArray Blackman(double M)
        {
            if (M < 1) return WindowEmpty();
            if (M == 1) return WindowOnes();

            var n = UnmanagedStorage.ArangeObj(1.0 - M, M, 2.0, np.float64);
            // Operation order preserved verbatim from NumPy for bit-parity:
            //   pi * n / (M - 1)     == (pi * n) / (M - 1)        (left to right)
            //   2.0 * pi * n / (M-1) == ((2.0 * pi) * n) / (M-1)
            // M-1 as a 0-d operand (see the header): one kernel for every M.
            var m1 = (NDExpr)NewScalar(M - 1);
            var arg = np.pi * (NDExpr)n / m1;
            var arg2 = 2.0 * np.pi * (NDExpr)n / m1;
            return Evaluate(0.42 + 0.5 * NDExpr.Cos(arg) + 0.08 * NDExpr.Cos(arg2));
        }

        /// <inheritdoc />
        public override NDArray Hamming(double M) => CosineWindow(M, 0.54, 0.46);

        /// <inheritdoc />
        public override NDArray Hanning(double M) => CosineWindow(M, 0.5, 0.5);

        /// <inheritdoc />
        [NDScoped]
        public override NDArray Kaiser(double M, double beta)
        {
            // NumPy has NO `M < 1` guard here (unlike the cosine windows): kaiser uses n = arange(0, M), which is EMPTY for
            // M <= 0 (an empty result) but ONE element for 0 < M < 1 (a computed, non-empty window). Only M == 1 is
            // special-cased, to avoid the alpha == 0 division. Reproduced verbatim from NumPy 2.4.2.
            if (M == 1) return WindowOnes();

            var n = UnmanagedStorage.ArangeObj(0.0, M, 1.0, np.float64);
            // alpha, beta and i0(beta) as 0-d operands (see the header): one kernel for every (M, beta) pair instead of one
            // JIT per pair. `alpha` appears twice and dedups to one stream.
            var alpha = (NDExpr)NewScalar((M - 1) / 2.0);
            NDArray betaScalar = NewScalar(beta);
            var betaOp = (NDExpr)betaScalar;
            // i0(float(beta)), NumPy's own statement: this engine's I0 over the 0-d beta runs the cephes double kernel the
            // fused pass below applies per element, so the denominator holds the same bits a scalar call of it gives.
            var denom = (NDExpr)I0(betaScalar);
            var inner = ((NDExpr)n - alpha) / alpha;
            // beta * sqrt(1 - ((n-alpha)/alpha)**2), then i0(.) / i0(beta), per element. np._i0Double is np.i0's float64
            // kernel delegate: NDExpr.Call keys its compiled kernel by the delegate's identity, so a stable field serves one
            // cached kernel (a method group would make a new delegate, and a new kernel, on every call).
            var arg = betaOp * NDExpr.Sqrt(1.0 - NDExpr.Power(inner, 2.0));
            return Evaluate(NDExpr.Call(np._i0Double, arg) / denom);
        }

        /// <summary>
        ///     The raised-cosine form <c>a0 + a1*cos(pi*n/(M-1))</c> hanning and hamming share, over
        ///     <c>n = arange(1-M, M, 2)</c>.
        /// </summary>
        /// <param name="M">The number of points (NumPy's float M; the cosine windows' guards apply).</param>
        /// <param name="a0">The constant term (0.5 for hanning, 0.54 for hamming).</param>
        /// <param name="a1">The cosine's weight (0.5 for hanning, 0.46 for hamming).</param>
        /// <returns>A new 1-D float64 array.</returns>
        /// <exception cref="ValueError"><paramref name="M"/> is NaN, +infinity or too large for arange to size.</exception>
        /// <remarks>The weights are C# constants baked into the fused kernel: one kernel per (a0, a1) pair - one for hanning
        ///     and one for hamming - serves every M, since M itself rides as a 0-d operand.</remarks>
        [NDScoped]
        private NDArray CosineWindow(double M, double a0, double a1)
        {
            if (M < 1) return WindowEmpty();
            if (M == 1) return WindowOnes();

            var n = UnmanagedStorage.ArangeObj(1.0 - M, M, 2.0, np.float64);
            // M-1 as a 0-d operand (see the header).
            var m1 = (NDExpr)NewScalar(M - 1);
            return Evaluate(a0 + a1 * NDExpr.Cos(np.pi * (NDExpr)n / m1));
        }

        /// <summary>NumPy's <c>array([], dtype=float64)</c>: the result of a cosine window with <c>M &lt; 1</c>, np.zeros of the
        ///     shape <c>(0,)</c>.</summary>
        /// <returns>A new empty float64 array.</returns>
        private NDArray WindowEmpty() => NewArray(np.float64, new Shape(new int[] { 0 }), true);

        /// <summary>NumPy's <c>ones(1, dtype=float64)</c>: the result of every window with <c>M == 1</c>.</summary>
        /// <returns>A new one-element float64 array holding 1.0.</returns>
        private static NDArray WindowOnes() => NDArray.ones(new Shape(new int[] { 1 }), np.float64);
    }
}
