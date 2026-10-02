using System;
using System.Collections.Generic;
using System.IO;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Catalogue entries for the np.ma members the masked corpus does not carry (creation helpers,
    ///     callable-taking wrappers, fill-value plumbing, mask-structure helpers, predicates — the
    ///     value oracle's <c>MaSiblingOwned</c> set) and for every <see cref="NDMaskedArray"/> instance
    ///     method (the corpus drives the MODULE functions; the instance spellings are their own entry
    ///     points), plus the DSL-object properties whose indexer is the real API (<c>np.mgrid</c>,
    ///     <c>np.ogrid</c>, <c>np.s_</c>, <c>np.index_exp</c>, <c>np.ma.mr_</c>).
    /// </summary>
    internal static partial class LeakCatalogue
    {
        /// <summary>np.ma module members outside the masked corpus, and the DSL indexers.</summary>
        /// <param name="l">The entry list.</param>
        private static void AddMaskedModule(List<LeakCase> l)
        {
            // ---- DSL objects (the property returns the object; its indexer does the work) ----
            E(l, "np.mgrid", "dense 2-D", f => (NDArray)np.mgrid["0:3", "0:4"]);
            E(l, "np.ogrid", "open 2-D", f => (NDArray[])np.ogrid["0:3", "0:4"]);
            E(l, "np.s_", "capture", f => Box(np.s_["1:3", "::2"].Length));
            E(l, "np.index_exp", "capture", f => Box(np.index_exp["::2"].Length));
            E(l, "np.ma.mr_", "masked concatenate", f => np.ma.mr_[f.MV, f.MV]);

            // ---- creation (the plain np twins are corpus-gated; these wrap them) ----
            E(l, "np.ma.arange", "int stop", f => np.ma.arange(10));
            E(l, "np.ma.arange", "double range", f => np.ma.arange(0.0, 2.0, 0.5));
            E(l, "np.ma.empty", "shape", f => np.ma.empty(new Shape(3, 2)));
            E(l, "np.ma.empty_like", "masked", f => np.ma.empty_like(f.MA));
            E(l, "np.ma.ones", "shape", f => np.ma.ones(new Shape(3, 2)));
            E(l, "np.ma.ones_like", "masked", f => np.ma.ones_like(f.MA));
            E(l, "np.ma.zeros", "shape", f => np.ma.zeros(new Shape(3, 2)));
            E(l, "np.ma.zeros_like", "masked", f => np.ma.zeros_like(f.MA));
            E(l, "np.ma.identity", "n", f => np.ma.identity(3));
            E(l, "np.ma.indices", "int[] dims", f => np.ma.indices(new[] { 2, 3 }));
            E(l, "np.ma.frombuffer", "bytes", f => np.ma.frombuffer(f.NpyBytes, np.uint8, count: 16));
            E(l, "np.ma.fromfunction", "i + j", f => np.ma.fromfunction((i, j) => i + j, new Shape(3, 4)));
            E(l, "np.ma.diagflat", "masked vector", f => np.ma.diagflat(f.MV0_4));

            // ---- callable-taking wrappers (callbacks return library-created views: no user temps) ----
            // A callback's result is the CALLER's under the apply_* contract (never disposed by the method), so an
            // ALLOCATING callback (np.ma.sum) would strand every superseded per-axis reduction by design. The
            // apply_over_axes entries therefore reduce with keepdims-shaped SLICES — views that take no buffer —
            // which drives the method's own re-expansion/supersede bookkeeping without any user temporaries.
            E(l, "np.ma.apply_along_axis", "identity func1d", f => np.ma.apply_along_axis(r => r, 1, f.MA));
            E(l, "np.ma.apply_over_axes", "int axis", f => np.ma.apply_over_axes((x, ax) => (NDMaskedArray)x[ax == 0 ? "0:1" : ":, 0:1"], f.MA, 0));
            E(l, "np.ma.apply_over_axes", "int[] axes", f => np.ma.apply_over_axes((x, ax) => (NDMaskedArray)x[ax == 0 ? "0:1" : ":, 0:1"], f.MA, new[] { 0, 1 }));

            // ---- stacking / splitting / shape ----
            E(l, "np.ma.column_stack", "two vectors", f => np.ma.column_stack(f.MV0_5, f.MV5_10));
            E(l, "np.ma.dstack", "two matrices", f => np.ma.dstack(f.MA, f.MB));
            E(l, "np.ma.hsplit", "sections", f => np.ma.hsplit(f.MA, 2));
            E(l, "np.ma.hsplit", "indices", f => np.ma.hsplit(f.MA, new[] { 1, 3 }));
            E(l, "np.ma.moveaxis", "swap", f => np.ma.moveaxis(f.MA, 0, 1));
            E(l, "np.ma.resize", "grow", f => np.ma.resize(f.MA, new Shape(4, 4)));
            E(l, "np.ma.ndim", "masked", f => Box(np.ma.ndim(f.MA)));
            E(l, "np.ma.shape", "masked", f => np.ma.shape(f.MA));
            E(l, "np.ma.size", "masked", f => Box(np.ma.size(f.MA)));

            // ---- mask structure / compression ----
            E(l, "np.ma.clump_masked", "1-D", f => np.ma.clump_masked(f.MV));
            E(l, "np.ma.clump_unmasked", "1-D", f => np.ma.clump_unmasked(f.MV));
            E(l, "np.ma.flatnotmasked_contiguous", "1-D", f => np.ma.flatnotmasked_contiguous(f.MV));
            E(l, "np.ma.flatnotmasked_edges", "1-D", f => np.ma.flatnotmasked_edges(f.MV));
            E(l, "np.ma.notmasked_contiguous", "axis", f => np.ma.notmasked_contiguous(f.MA, 1));
            E(l, "np.ma.notmasked_edges", "axis", f => np.ma.notmasked_edges(f.MA, 1));
            E(l, "np.ma.compress_nd", "all axes", f => np.ma.compress_nd(f.MA));
            E(l, "np.ma.compress_rowcols", "rows", f => np.ma.compress_rowcols(f.MA, 0));
            E(l, "np.ma.nonzero", "masked", f => np.ma.nonzero(f.MA));
            E(l, "np.ma.ndenumerate", "walk unmasked", f =>
            {
                long n = 0;
                foreach (var _ in np.ma.ndenumerate(f.MA))
                    n++;
                return Box(n);
            });
            E(l, "np.ma.getmask", "masked + nomask", f => new object[] { np.ma.getmask(f.MA), np.ma.getmask(f.MAn) });
            E(l, "np.ma.ids", "addresses", f => Box(np.ma.ids(f.MA)));

            // ---- mask mutation helpers (on arrays the entry owns) ----
            E(l, "np.ma.harden_mask", "owned copy", f => np.ma.harden_mask(f.MA.copy()));
            E(l, "np.ma.soften_mask", "owned copy", f => np.ma.soften_mask(f.MA.copy()));
            E(l, "np.ma.shrink_mask", "owned copy", f => np.ma.shrink_mask(f.MA.copy()));
            E(l, "np.ma.putmask", "owned copy", f =>
            {
                var m = f.MA.copy();
                np.ma.putmask(m, f.MA._mask, 9.0);
                return m;
            });
            E(l, "np.ma.set_fill_value", "owned copy", f =>
            {
                var m = f.MA.copy();
                np.ma.set_fill_value(m, -1.0);
                return m;
            });
            E(l, "np.ma.masked_object", "value", f => np.ma.masked_object(f.M, 1.75));

            // ---- fill values / predicates ----
            E(l, "np.ma.common_fill_value", "pair", f => np.ma.common_fill_value(f.MA, f.MB));
            E(l, "np.ma.default_fill_value", "dtype", f => np.ma.default_fill_value(np.float64));
            E(l, "np.ma.maximum_fill_value", "masked", f => np.ma.maximum_fill_value(f.MA));
            E(l, "np.ma.minimum_fill_value", "masked", f => np.ma.minimum_fill_value(f.MA));
            E(l, "np.ma.isMA", "masked", f => Box(np.ma.isMA(f.MA)));
            E(l, "np.ma.isMaskedArray", "masked", f => Box(np.ma.isMaskedArray(f.MA)));
            E(l, "np.ma.isarray", "masked", f => Box(np.ma.isarray(f.MA)));
            E(l, "np.ma.is_mask", "bool array", f => Box(np.ma.is_mask(f.B)));

            // ---- statistics / products on masked operands ----
            E(l, "np.ma.convolve", "propagate mask", f => np.ma.convolve(f.MV0_10, f.MV10_13));
            E(l, "np.ma.correlate", "valid", f => np.ma.correlate(f.MV0_10, f.MV10_13));
            E(l, "np.ma.corrcoef", "rows", f => np.ma.corrcoef(f.MA));
            E(l, "np.ma.cov", "rows", f => np.ma.cov(f.MA));
            E(l, "np.ma.polyfit", "deg 1", f =>
            {
                var r = np.ma.polyfit(f.MV0_10, f.MV10_20, 1);
                return new object[] { r.coeffs, r.residuals, r.rank, r.singular_values, r.covariance };
            }, backend: true);
        }

        /// <summary>Every NDMaskedArray instance method (plus its indexer), each on the fixtures or an owned copy.</summary>
        /// <param name="l">The entry list.</param>
        private static void AddMaskedArray(List<LeakCase> l)
        {
            E(l, "NDMaskedArray.Item", "get element + slice", f => new object[] { f.MA[1, 2], f.MA["1:"] });
            E(l, "NDMaskedArray.Item", "set masked + value on an owned copy", f =>
            {
                var m = f.MA.copy();
                m[0, 0] = np.ma.masked;
                m[1, 1] = 5.0;
                return m;
            });
            E(l, "NDMaskedArray.ToString", "str + repr", f => f.MA.ToString() + f.MA.ToString(true));
            E(l, "NDMaskedArray.all", "axis", f => f.MA.all(0));
            E(l, "NDMaskedArray.any", "axis", f => f.MA.any(1));
            E(l, "NDMaskedArray.anom", "flat", f => f.MA.anom());
            E(l, "NDMaskedArray.argmax", "flat", f => f.MA.argmax());
            E(l, "NDMaskedArray.argmin", "axis", f => f.MA.argmin(1));
            E(l, "NDMaskedArray.argpartition", "kth", f => f.MV.argpartition(3));
            E(l, "NDMaskedArray.argsort", "last axis", f => f.MA.argsort());
            E(l, "NDMaskedArray.astype", "float32", f => f.MA.astype(np.float32));
            E(l, "NDMaskedArray.byteswap", "copy", f => f.MA.byteswap());
            E(l, "NDMaskedArray.choose", "index array", f => f.MI.choose(new object[] { f.M, f.M }, "clip"));
            E(l, "NDMaskedArray.clip", "bounds", f => f.MA.clip(1.0, 10.0));
            E(l, "NDMaskedArray.compress", "condition", f => f.MV.compress(f.B));
            E(l, "NDMaskedArray.compressed", "1-D", f => f.MA.compressed());
            E(l, "NDMaskedArray.conj", "copy", f => f.MA.conj());
            E(l, "NDMaskedArray.conjugate", "copy", f => f.MA.conjugate());
            E(l, "NDMaskedArray.copy", "copy", f => f.MA.copy());
            E(l, "NDMaskedArray.count", "axis", f => f.MA.count(0));
            E(l, "NDMaskedArray.cumprod", "axis", f => f.MA.cumprod(1));
            E(l, "NDMaskedArray.cumsum", "flat", f => f.MA.cumsum());
            E(l, "NDMaskedArray.diagonal", "main", f => f.MA.diagonal());
            E(l, "NDMaskedArray.dot", "matrix", f => f.MA.dot(f.MBT));
            E(l, "NDMaskedArray.fill", "owned copy", f =>
            {
                var m = f.MA.copy();
                m.fill(0.5);
                return m;
            });
            E(l, "NDMaskedArray.filled", "default + value", f => new object[] { f.MA.filled(), f.MA.filled(-1.0) });
            E(l, "NDMaskedArray.flatten", "C", f => f.MA.flatten());
            E(l, "NDMaskedArray.harden_mask", "owned copy", f => f.MA.copy().harden_mask());
            E(l, "NDMaskedArray.ids", "addresses", f => Box(f.MA.ids()));
            E(l, "NDMaskedArray.iscontiguous", "flag", f => Box(f.MA.iscontiguous()));
            E(l, "NDMaskedArray.item", "size-1 sub-array view", f =>
            {
                // A slice index keeps the result a (1, 1) masked VIEW (an element index would already
                // reduce to the bare scalar), so item() runs on a real NDMaskedArray.
                var e = (NDMaskedArray)f.MA["1:2, 3:4"];
                return new object[] { e, e.item() };
            });
            E(l, "NDMaskedArray.max", "axis", f => f.MA.max(0));
            E(l, "NDMaskedArray.mean", "flat", f => f.MA.mean());
            E(l, "NDMaskedArray.min", "axis keepdims", f => f.MA.min(1, null, true));
            E(l, "NDMaskedArray.nonzero", "masked", f => f.MA.nonzero());
            E(l, "NDMaskedArray.partition", "owned copy", f =>
            {
                var m = f.MV.copy();
                m.partition(4);
                return m;
            });
            E(l, "NDMaskedArray.prod", "flat", f => f.MA.prod());
            E(l, "NDMaskedArray.product", "axis", f => f.MA.product(0));
            E(l, "NDMaskedArray.ptp", "axis", f => f.MA.ptp(1));
            E(l, "NDMaskedArray.put", "owned copy", f =>
            {
                var m = f.MV.copy();
                m.put(f.Idx, 3.0);
                return m;
            });
            E(l, "NDMaskedArray.putmask", "owned copy", f =>
            {
                var m = f.MA.copy();
                m.putmask(f.MA._mask, 7.0);
                return m;
            });
            E(l, "NDMaskedArray.ravel", "C", f => f.MA.ravel());
            E(l, "NDMaskedArray.repeat", "axis", f => f.MA.repeat(2, 0));
            E(l, "NDMaskedArray.reshape", "4x3", f => f.MA.reshape(4, 3));
            T(l, "NDMaskedArray.resize", "raises by design (NumPy: a masked array does not own its data)", f =>
            {
                var m = f.MA.copy();
                try
                {
                    m.resize(new Shape(2, 6), refcheck: false);
                    return m;
                }
                finally
                {
                    UndisposedIntermediateTests.DisposeAny(m, f.Keep);   // the entry's own copy — released on the throw
                }
            });
            E(l, "NDMaskedArray.round", "decimals", f => f.MA.round(1));
            E(l, "NDMaskedArray.searchsorted", "sorted masked", f =>
            {
                using var sorted = np.arange(10.0);
                var m = np.ma.masked_array(sorted);
                return m.searchsorted(f.V3);
            });
            E(l, "NDMaskedArray.shrink_mask", "owned copy", f => f.MA.copy().shrink_mask());
            E(l, "NDMaskedArray.soften_mask", "owned copy", f => f.MA.copy().soften_mask());
            E(l, "NDMaskedArray.sort", "last axis", f => f.MA.sort());
            E(l, "NDMaskedArray.squeeze", "no-op", f => f.MA.squeeze());
            E(l, "NDMaskedArray.std", "axis ddof", f => f.MA.std(0, null, 1));
            E(l, "NDMaskedArray.sum", "axis", f => f.MA.sum(1));
            E(l, "NDMaskedArray.swapaxes", "0,1", f => f.MA.swapaxes(0, 1));
            E(l, "NDMaskedArray.take", "flat indices", f => f.MA.take(f.Idx));
            E(l, "NDMaskedArray.tobytes", "fill", f => f.MA.tobytes(0.0));
            T(l, "NDMaskedArray.tofile", "raises by design (NumPy: MaskedArray.tofile() not implemented yet)", f =>
            {
                f.MA.tofile(Path.Combine(f.Dir, "ma.bin"));
                return null;
            });
            E(l, "NDMaskedArray.tolist", "nested", f => f.MA.tolist());
            E(l, "NDMaskedArray.trace", "main", f => f.MA.trace());
            E(l, "NDMaskedArray.transpose", "default", f => f.MA.transpose());
            E(l, "NDMaskedArray.unshare_mask", "owned copy", f =>
            {
                var m = f.MA.copy();
                // unshare_mask REPLACES the mask with a private copy and deliberately never disposes the superseded
                // one (another masked array may still read it). This entry built the copy, so it is that mask's
                // only holder and releases it itself — exactly what the member's documentation tells a caller to do.
                var superseded = m._mask;
                m.unshare_mask();
                superseded?.Dispose();
                return m;
            });
            E(l, "NDMaskedArray.var", "flat", f => f.MA.var());
            E(l, "NDMaskedArray.view", "same dtype", f => f.MA.view());
        }
    }
}
