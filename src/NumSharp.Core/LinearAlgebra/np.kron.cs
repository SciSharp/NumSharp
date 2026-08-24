using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Kronecker product of two arrays.
        ///     <para>
        ///     Computes the Kronecker product, a composite array made of blocks of the second array
        ///     scaled by the first. If <c>a.shape = (r0,r1,...,rN)</c> and <c>b.shape = (s0,s1,...,sN)</c>
        ///     the result has shape <c>(r0*s0, r1*s1, ..., rN*sN)</c> with
        ///     <c>kron(a,b)[k0,...,kN] = a[i0,...,iN] * b[j0,...,jN]</c> where <c>kt = it*st + jt</c>.
        ///     </para>
        ///     The number of dimensions of <paramref name="a"/> and <paramref name="b"/> need not match —
        ///     the smaller is treated as if prepended with size-1 axes (NumPy's <c>ndmin</c> behaviour).
        /// </summary>
        /// <param name="a">First input array.</param>
        /// <param name="b">Second input array.</param>
        /// <returns>
        ///     A fresh, writeable, C-contiguous array. The result dtype follows NumPy's <c>multiply</c>
        ///     promotion (NEP50). If <paramref name="b"/> is 0-d the result is the element-wise
        ///     <c>a * b</c> (which is itself the degenerate Kronecker product).
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.kron.html</remarks>
        [NDScoped]
        public static NDArray kron(NDArray a, NDArray b)
        {
            int nda = a.ndim, ndb = b.ndim;

            // NumPy promotes a to ndmin=b.ndim, so post-promotion nda >= ndb and the scalar shortcut
            // `if (nda == 0 || ndb == 0): return multiply(a, b)` reduces to "b is 0-d": the Kronecker
            // product with a scalar is just the (broadcasting) element-wise product. This also covers
            // the both-scalar case (0-d * 0-d -> 0-d).
            if (ndb == 0 || nda == 0)
                return multiply(a, b);

            int nd = Math.Max(nda, ndb);

            // Equalise ranks by prepending size-1 axes to the smaller shape. NumPy pads `a` up front via
            // ndmin and pads `b`'s shape in the `bs` computation; padding both to `nd` is the same net
            // shape and keeps the code symmetric.
            var as_ = new long[nd];
            var bs = new long[nd];
            for (int i = 0; i < nd; i++) { as_[i] = 1; bs[i] = 1; }
            for (int i = 0; i < nda; i++) as_[nd - nda + i] = a.shape[i];
            for (int i = 0; i < ndb; i++) bs[nd - ndb + i] = b.shape[i];

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
            long maxItem = Math.Max(a.dtypesize, b.dtypesize);
            bool simd = IsSimdMultiplyType(a.typecode) && IsSimdMultiplyType(b.typecode);
            bool useTile = simd && run <= 5 && outSize >= 8192 && outSize * maxItem <= (64L << 20);

            if (useTile)
            {
                var bcast = new long[2 * nd];
                for (int i = 0; i < nd; i++) { bcast[2 * i] = as_[i]; bcast[2 * i + 1] = bs[i]; }
                var bcShape = new Shape(bcast);

                // repeat-a and tile-b, both materialised C-contiguous at the interleaved shape, then one
                // SimdFull multiply. The final reshape to (rt*st) is a view over the contiguous product.
                using var aRep = broadcast_to(a.reshape(aInter), bcShape).copy();
                using var bTiled = broadcast_to(b.reshape(bInter), bcShape).copy();
                return multiply(aRep, bTiled).reshape(outShape);
            }

            // Direct broadcast-multiply. For contiguous inputs the two reshapes are pure views, so this is
            // a single kernel pass; a non-contiguous operand is materialised C-order by reshape (as NumPy
            // does), keeping values correct for every layout.
            return multiply(a.reshape(aInter), b.reshape(bInter)).reshape(outShape);
        }

        /// <summary>
        ///     True for dtypes whose element-wise multiply has a vectorised (SimdFull) path. The
        ///     scalar-path trio Half/Decimal/Complex return false — for them the tile fast-path's SimdFull
        ///     multiply is no faster than direct's scalar multiply, so its extra copy passes only cost.
        /// </summary>
        private static bool IsSimdMultiplyType(NPTypeCode tc)
            => tc != NPTypeCode.Half && tc != NPTypeCode.Decimal && tc != NPTypeCode.Complex;

        /// <summary>
        ///     Length of the innermost coalesced contiguous run of the interleaved (a0,b0,a1,b1,...) layout
        ///     used by the direct broadcast-multiply — i.e. the effective inner SIMD run. Walks the output
        ///     axes innermost-first (b_{nd-1}, a_{nd-1}, b_{nd-2}, ...), accumulating the run while it stays
        ///     within one operand's contiguous data and stopping at the first size&gt;1 axis of the other
        ///     operand. Size-1 axes are transparent, so an operand whose trailing dim is 1 correctly yields
        ///     the OTHER operand's dim as the run.
        /// </summary>
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
