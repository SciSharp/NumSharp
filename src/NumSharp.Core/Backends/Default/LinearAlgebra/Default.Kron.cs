// NumPy's np.kron (numpy/lib/_shape_base_impl.py) as DefaultEngine's member: the rank padding, the interleaved
// broadcast multiply and its tile fast path (np.multiply's statement, this engine's Multiply; np.broadcast_to's, the
// array object's NDArray.broadcast_to).

using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        /// <remarks>
        ///     <para>Scoped by <c>[NDScoped]</c>: the interleaved reshapes, the broadcast views, the tile path's two
        ///     materialized operands and the product that the final reshape views are released on return, and that
        ///     reshape is yielded. The tile path no longer adds hand-written <c>using</c> declarations on top.</para>
        /// </remarks>
        [NDScoped]
        public override NDArray Kron(NDArray a, NDArray b)
        {
            int nda = a.ndim, ndb = b.ndim;

            // NumPy promotes a to ndmin=b.ndim, so post-promotion nda >= ndb and the scalar shortcut
            // `if (nda == 0 || ndb == 0): return multiply(a, b)` reduces to "b is 0-d": the Kronecker
            // product with a scalar is just the (broadcasting) element-wise product. This also covers
            // the both-scalar case (0-d * 0-d -> 0-d).
            if (ndb == 0 || nda == 0)
                return Multiply(a, b);

            int nd = Math.Max(nda, ndb);

            // Equalise ranks by prepending size-1 axes to the smaller shape. NumPy pads `a` up front via
            // ndmin and pads `b`'s shape in the `bs` computation; padding both to `nd` is the same net
            // shape and keeps the code symmetric.
            var as_ = new long[nd];
            var bs = new long[nd];
            for (int i = 0; i < nd; i++) { as_[i] = 1; bs[i] = 1; }
            for (int i = 0; i < nda; i++) as_[nd - nda + i] = a.Shape.dimensions[i];
            for (int i = 0; i < ndb; i++) bs[nd - ndb + i] = b.Shape.dimensions[i];

            // Interleave: a-dims get a trailing size-1 axis (odd positions), b-dims a leading size-1 axis
            // (even positions). Broadcasting the two interleaved views to (r0,s0,r1,s1,...) and multiplying
            // realises kron[...] = a[i]*b[j]; the collapsed shape (rt*st) is a view over that result.
            var aInter = new long[2 * nd];
            var bInter = new long[2 * nd];
            var outShape = new long[nd];
            long outSize = 1;
            for (int i = 0; i < nd; i++)
            {
                aInter[2 * i] = as_[i]; aInter[2 * i + 1] = 1;
                bInter[2 * i] = 1;      bInter[2 * i + 1] = bs[i];
                outShape[i] = as_[i] * bs[i];
                outSize *= outShape[i];
            }

            // Fast-path selection. The DIRECT broadcast-multiply fills the output in one pass, but its
            // innermost SIMD run is the innermost coalesced contiguous stretch of the interleaved layout
            // (b's trailing dims). When that run is short, the many tiny runs make the single pass slower
            // than materialising both operands to full C-contiguous arrays (repeat-a and tile-b) and
            // running ONE SimdFull multiply — three contiguous passes that beat the short-run single pass.
            // Measured crossover is around run 5-6; the TILE path is taken for short runs on non-trivial
            // outputs (e.g. the common kron(big, 2x2) tensor-product pattern). Both paths are bit-identical.
            //
            // Upper bound: tile writes ~3x the output's bytes (two materialised operands + the product),
            // which pays only while that traffic stays cache-resident. Past ~64 MB of result the 3x RAM
            // traffic loses to direct's single (short-run) pass, so tile is confined to that window. The
            // bound is in BYTES via the wider operand's item size so wide dtypes (complex, f64) fall back
            // to direct earlier than narrow ones. (Very large short-run products are then memory/overhead
            // bound and land nearer parity — still faster than NumPy, just below the tile speedups.)
            //
            // Dtype gate: tile's whole advantage is that its final multiply is SimdFull. For the
            // scalar-path dtypes (Half/Decimal/Complex have no vector multiply) that advantage is absent,
            // so tile is just direct's scalar multiply plus two extra copy passes — strictly worse. Those
            // route to direct.
            long run = InnerRun(as_, bs, nd);
            bool simd = IsSimdMultiplyType(a.typecode) && IsSimdMultiplyType(b.typecode);
            // The item sizes are read only for the SIMD dtypes: a StringDType operand has no fixed item size (its cells are
            // 16-byte handles), and text takes the direct path whatever its size.
            bool useTile = simd && run <= 5 && outSize >= 8192 && outSize * Math.Max(a.dtypesize, b.dtypesize) <= (64L << 20);

            if (useTile)
            {
                var bcast = new long[2 * nd];
                for (int i = 0; i < nd; i++) { bcast[2 * i] = as_[i]; bcast[2 * i + 1] = bs[i]; }
                var bcShape = Shape.Adopt(bcast);

                // repeat-a and tile-b, both materialised C-contiguous at the interleaved shape, then one
                // SimdFull multiply. The final reshape to (rt*st) is a view over the contiguous product.
                // Both operands (and the reshape/broadcast views they came from) are released by this
                // method's scope as soon as it returns — the same moment a `using` would release them.
                var aRep = NDArray.broadcast_to(a.reshape(Shape.Adopt(aInter)), bcShape).copy();
                var bTiled = NDArray.broadcast_to(b.reshape(Shape.Adopt(bInter)), bcShape).copy();
                return Multiply(aRep, bTiled).reshape(Shape.Adopt(outShape));
            }

            // Direct broadcast-multiply. For contiguous inputs the two reshapes are pure views, so this is
            // a single kernel pass; a non-contiguous operand is materialised C-order by reshape (as NumPy
            // does), keeping values correct for every layout.
            return Multiply(a.reshape(Shape.Adopt(aInter)), b.reshape(Shape.Adopt(bInter))).reshape(Shape.Adopt(outShape));
        }

        /// <summary>
        ///     True for dtypes whose element-wise multiply has a vectorised (SimdFull) path. The
        ///     scalar-path trio Half/Decimal/Complex return false — for them the tile fast-path's SimdFull
        ///     multiply is no faster than direct's scalar multiply, so its extra copy passes only cost. Text
        ///     (fixed-width or StringDType) returns false too: its multiply is the string repetition loop or
        ///     NumPy's refusal, never SimdFull, and a StringDType operand has no fixed item size to bound.
        /// </summary>
        /// <param name="tc">The operand's type code.</param>
        /// <returns>True when the tile path's final multiply would run vectorised.</returns>
        private static bool IsSimdMultiplyType(NPTypeCode tc)
            => tc != NPTypeCode.Half && tc != NPTypeCode.Decimal && tc != NPTypeCode.Complex && tc != NPTypeCode.String;

        /// <summary>
        ///     Length of the innermost coalesced contiguous run of the interleaved (a0,b0,a1,b1,...) layout
        ///     used by the direct broadcast-multiply — i.e. the effective inner SIMD run. Walks the output
        ///     axes innermost-first (b_{nd-1}, a_{nd-1}, b_{nd-2}, ...), accumulating the run while it stays
        ///     within one operand's contiguous data and stopping at the first size&gt;1 axis of the other
        ///     operand. Size-1 axes are transparent, so an operand whose trailing dim is 1 correctly yields
        ///     the OTHER operand's dim as the run.
        /// </summary>
        /// <param name="as_">The first operand's dimensions, padded to <paramref name="nd"/>.</param>
        /// <param name="bs">The second operand's dimensions, padded to <paramref name="nd"/>.</param>
        /// <param name="nd">The common rank.</param>
        /// <returns>The number of elements of the innermost contiguous run.</returns>
        private static long InnerRun(long[] as_, long[] bs, int nd)
        {
            long run = 1;
            int active = 0; // 0 = none yet, 1 = a, 2 = b
            for (int k = nd - 1; k >= 0; k--)
            {
                // b_k is inner of the (a_k, b_k) pair in the interleaved layout.
                if (bs[k] > 1)
                {
                    if (active == 1) return run;
                    active = 2; run *= bs[k];
                }
                if (as_[k] > 1)
                {
                    if (active == 2) return run;
                    active = 1; run *= as_[k];
                }
            }
            return run;
        }
    }
}
