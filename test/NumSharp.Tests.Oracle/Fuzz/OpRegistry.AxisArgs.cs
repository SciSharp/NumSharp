using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The <c>axis_arg</c> op of the axis_args tier (gen_oracle.py <c>gen_axis_args</c>): every axis-taking function,
    ///     called with each axis value of its operand's rank, in range and out of it, and with repeated / out-of-range /
    ///     empty tuples where the function takes one. <c>params.fn</c> names the NumPy spelling (<c>sum</c>,
    ///     <c>ndarray.argmax</c>, <c>ma.sum</c>, <c>linalg.norm</c>, <c>fft.fft</c>) and the replay calls the NumSharp twin
    ///     of exactly that spelling, because the module function, the ndarray method and the numpy.ma function each have
    ///     their own axis code in both libraries.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The axis parameters are named as NumPy names them: <c>axis</c> (an integer, or a list for a tuple axis),
    ///     <c>axis1</c> / <c>axis2</c>, <c>source</c> / <c>destination</c>, <c>start</c>, <c>axisa</c> / <c>axisb</c> /
    ///     <c>axisc</c> and <c>axes</c>. Whether an <c>axis</c> is a tuple is read from the JSON value itself, so one fn key
    ///     reaches both the integer and the tuple overload.
    ///     </para>
    ///     <para>
    ///     A numpy.ma result is compared through its data: the operands carry no mask, so no slot is masked. The masked
    ///     array's own mask and stored fill are released here, because the leak gate replays this registry and counts
    ///     every pooled buffer a case leaves behind.
    ///     </para>
    /// </remarks>
    public static partial class OpRegistry
    {
        /// <summary>
        ///     Replays one array-result <c>axis_arg</c> case: the NumSharp call named by <c>params.fn</c> with the case's
        ///     axis parameters.
        /// </summary>
        /// <param name="p">The case params: <c>fn</c> and the axis parameters NumPy's call took.</param>
        /// <param name="ops">The reconstructed operands, in the order the generator passed them.</param>
        /// <returns>The function's result as one array (a scalar result wrapped 0-d, a numpy.ma result's data).</returns>
        /// <exception cref="NotSupportedException">The fn key has no replay here (a corpus/registry drift).</exception>
        /// <remarks>Any exception the NumSharp call raises propagates: the error cases compare its type and text.</remarks>
        internal static NDArray AxisArgFromCorpus(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            string fn = p["fn"].GetString();
            bool tuple = AxisArgIsTuple(p, "axis");
            int ax = !tuple && p.ContainsKey("axis") ? p["axis"].GetInt32() : 0;
            int[] axt = tuple ? ParseIntArray(p["axis"]) : null;
            NDArray a = ops.Length > 0 ? ops[0] : null;

            // The stacking family's empty operand list: NumPy's "need at least one array to …" ValueError.
            if (p.TryGetValue("operands", out var operandsKind) && operandsKind.GetString() == "empty")
                return AxisArgEmptyStack(fn);

            // numpy.ma over a MASKED operand: the masked code's own axis handling (the cases are out-of-range axes).
            if (p.TryGetValue("mask", out var maskKind))
                return AxisArgMasked(fn, maskKind.GetString(), a, ax);

            // np.pad's dict pad_width: its keys are axes NumPy uses as list indices.
            if (fn == "pad")
                return np.pad(a, AxisArgPadWidths(p["pad_width"]), p["mode"].GetString());

            switch (fn)
            {
                // ---- the reductions (NumPy's ufunc reduce or its Python wrappers) ----
                case "sum": return np.sum(a, ax);
                case "prod": return np.prod(a, ax);
                case "mean": return np.mean(a, ax);
                case "std": return np.std(a, ax);
                case "var": return np.var(a, ax);
                case "max": return np.max(a, ax);
                case "min": return np.min(a, ax);
                case "amax": return np.amax(a, ax);
                case "amin": return np.amin(a, ax);
                case "ptp": return tuple ? np.ptp(a, axt) : np.ptp(a, ax);
                case "all": return tuple ? np.all(a, axt) : np.all(a, ax);
                case "any": return tuple ? np.any(a, axt) : np.any(a, ax);
                case "argmax": return np.argmax(a, ax);
                case "argmin": return np.argmin(a, ax);
                case "nanargmax": return np.nanargmax(a, ax);
                case "nanargmin": return np.nanargmin(a, ax);
                case "cumsum": return np.cumsum(a, ax);
                case "cumprod": return np.cumprod(a, ax);
                case "nancumsum": return np.nancumsum(a, ax);
                case "nancumprod": return np.nancumprod(a, ax);
                case "nansum": return np.nansum(a, ax);
                case "nanprod": return np.nanprod(a, ax);
                case "nanmean": return np.nanmean(a, ax);
                case "nanstd": return np.nanstd(a, ax);
                case "nanvar": return np.nanvar(a, ax);
                case "nanmax": return np.nanmax(a, ax);
                case "nanmin": return np.nanmin(a, ax);
                case "median": return tuple ? np.median(a, axt) : np.median(a, ax);
                case "nanmedian": return tuple ? np.nanmedian(a, axt) : np.nanmedian(a, ax);
                case "percentile": return tuple ? np.percentile(a, 50, axt) : np.percentile(a, 50, ax);
                case "nanpercentile": return tuple ? np.nanpercentile(a, 50, axt) : np.nanpercentile(a, 50, ax);
                case "quantile": return tuple ? np.quantile(a, 0.5, axt) : np.quantile(a, 0.5, ax);
                case "nanquantile": return tuple ? np.nanquantile(a, 0.5, axt) : np.nanquantile(a, 0.5, ax);
                case "average": return tuple ? np.average(a, axt) : np.average(a, ax);
                case "count_nonzero": return np.count_nonzero(a, ax);

                // ---- sorting, selection, the per-axis manipulation functions ----
                case "sort": return np.sort(a, ax);
                case "argsort": return np.argsort(a, ax);
                case "partition": return np.partition(a, 1, ax);
                case "argpartition": return np.argpartition(a, 1, ax);
                case "take": return np.take(a, ops[1], ax);
                case "take_along_axis": return np.take_along_axis(a, ops[1], ax);
                case "put_along_axis":
                {
                    // NumPy's put_along_axis returns None; the corpus holds the mutated copy (released if the call
                    // raises — the copy is the registry's, not the library's).
                    var copy = a.copy();
                    using var seven = NDArray.Scalar(7.0);
                    return MutatedCopyOrRelease(copy, c => np.put_along_axis(c, ops[1], seven, ax));
                }
                case "compress": return np.compress(ops[0], ops[1], ax);
                case "repeat": return np.repeat(a, 2, ax);
                case "delete": return np.delete(a, 0, ax);
                case "insert": return np.insert(a, 0, 7.0, ax);
                case "append": return np.append(a, a, ax);
                case "unique": return np.unique(a, axis: ax);
                case "flip": return tuple ? np.flip(a, axt) : np.flip(a, ax);
                case "roll": return np.roll(a, 1, ax);
                case "rollaxis":
                    return np.rollaxis(a, ax, p.TryGetValue("start", out var st) ? st.GetInt32() : 0);
                case "concatenate": return np.concatenate(new[] { a, a }, ax);
                case "concat": return np.concat(new[] { a, a }, ax);
                case "diff": return np.diff(a, 1, ax);
                case "trapezoid": return np.trapezoid(a, axis: ax);
                case "unwrap": return np.unwrap(a, axis: ax);
                case "apply_along_axis":
                    return WithCallbackResults(made => np.apply_along_axis(x => made.Keep(np.sum(x)), ax, a));
                case "apply_over_axes":
                    return WithCallbackResults(made => tuple
                        ? np.apply_over_axes((x, axis) => made.Keep(np.sum(x, axis, true)), a, axt)
                        : np.apply_over_axes((x, axis) => made.Keep(np.sum(x, axis, true)), a, ax));
                case "trim_zeros": return tuple ? np.trim_zeros(a, "fb", axt) : np.trim_zeros(a, "fb", ax);
                case "packbits": return np.packbits(a, ax);
                case "unpackbits": return np.unpackbits(a, ax);
                case "size": return NDArray.Scalar(np.size(a, ax));
                case "lexsort": return np.lexsort(new[] { a }, ax);
                case "expand_dims": return tuple ? np.expand_dims(a, axt) : np.expand_dims(a, ax);
                case "stack": return np.stack(new[] { a, a }, ax);
                case "squeeze": return np.squeeze(a, ax);
                case "swapaxes": return np.swapaxes(a, p["axis1"].GetInt32(), p["axis2"].GetInt32());
                case "moveaxis":
                {
                    // The four overloads (int / int[] for each of source and destination), picked by the JSON kinds.
                    bool st2 = AxisArgIsTuple(p, "source"), dt2 = AxisArgIsTuple(p, "destination");
                    if (st2 && dt2) return np.moveaxis(a, ParseIntArray(p["source"]), ParseIntArray(p["destination"]));
                    if (st2) return np.moveaxis(a, ParseIntArray(p["source"]), p["destination"].GetInt32());
                    if (dt2) return np.moveaxis(a, p["source"].GetInt32(), ParseIntArray(p["destination"]));
                    return np.moveaxis(a, p["source"].GetInt32(), p["destination"].GetInt32());
                }
                case "diagonal": return np.diagonal(a, 0, p["axis1"].GetInt32(), p["axis2"].GetInt32());
                case "trace": return np.trace(a, 0, p["axis1"].GetInt32(), p["axis2"].GetInt32());
                case "transpose": return np.transpose(a, ParseIntArray(p["axes"]));
                case "permute_dims": return np.permute_dims(a, ParseIntArray(p["axes"]));
                case "rot90": return np.rot90(a, 1, ParseIntArray(p["axes"]));
                case "cross":
                    return np.cross(ops[0], ops[1],
                        p.TryGetValue("axisa", out var xa) ? xa.GetInt32() : -1,
                        p.TryGetValue("axisb", out var xb) ? xb.GetInt32() : -1,
                        p.TryGetValue("axisc", out var xc) ? xc.GetInt32() : -1,
                        p.TryGetValue("axis", out var xx) ? xx.GetInt32() : (int?)null);

                // ---- np.linalg / the gufunc vecdot ----
                case "linalg.norm": return tuple ? np.linalg.norm(a, null, axt) : np.linalg.norm(a, null, ax);
                case "linalg.vector_norm":
                    return tuple ? np.linalg.vector_norm(a, axis: axt) : np.linalg.vector_norm(a, axis: ax);
                case "linalg.vecdot": return np.linalg.vecdot(ops[0], ops[1], ax);
                case "linalg.cross": return np.linalg.cross(ops[0], ops[1], ax);
                case "vecdot": return np.vecdot(ops[0], ops[1], axis: ax);

                // ---- np.fft ----
                case "fft.fft": return np.fft.fft(a, axis: ax);
                case "fft.ifft": return np.fft.ifft(a, axis: ax);
                case "fft.rfft": return np.fft.rfft(a, axis: ax);
                case "fft.irfft": return np.fft.irfft(a, axis: ax);
                case "fft.hfft": return np.fft.hfft(a, axis: ax);
                case "fft.ihfft": return np.fft.ihfft(a, axis: ax);
                case "fft.fftshift": return tuple ? np.fft.fftshift(a, axes: axt) : np.fft.fftshift(a, axes: ax);
                case "fft.ifftshift": return tuple ? np.fft.ifftshift(a, axes: axt) : np.fft.ifftshift(a, axes: ax);
                case "linalg.tensorsolve": return np.linalg.tensorsolve(a, ops[1], axt);
                case "fft.fftn": return np.fft.fftn(a, axes: axt);
                case "fft.ifftn": return np.fft.ifftn(a, axes: axt);
                case "fft.rfftn": return np.fft.rfftn(a, axes: axt);
                case "fft.irfftn": return np.fft.irfftn(a, axes: axt);
                case "fft.fft2": return np.fft.fft2(a, axes: ParseIntArray(p["axes"]));
                case "fft.ifft2": return np.fft.ifft2(a, axes: ParseIntArray(p["axes"]));
                case "fft.rfft2": return np.fft.rfft2(a, axes: ParseIntArray(p["axes"]));
                case "fft.irfft2": return np.fft.irfft2(a, axes: ParseIntArray(p["axes"]));

                // ---- the ndarray methods ----
                case "ndarray.argmax": return a.argmax(ax);
                case "ndarray.argmin": return a.argmin(ax);
                case "ndarray.cumsum": return a.cumsum(ax);
                case "ndarray.cumprod": return a.cumprod(ax);
                case "ndarray.sort":
                    // NumPy's ndarray.sort sorts in place and returns None; the corpus holds the sorted copy.
                    return MutatedCopyOrRelease(a.copy(), c => c.sort(ax));
                case "ndarray.argsort": return a.argsort(ax);
                case "ndarray.partition":
                    // In place like ndarray.sort: the corpus holds the partitioned copy.
                    return MutatedCopyOrRelease(a.copy(), c => c.partition(1, ax));
                case "ndarray.argpartition": return a.argpartition(1, ax);
                case "ndarray.take": return a.take(ops[1], ax);
                case "ndarray.compress": return ops[1].compress(ops[0], ax);
                case "ndarray.repeat": return a.repeat(2, ax);
                case "ndarray.squeeze": return a.squeeze(ax);
                case "ndarray.swapaxes": return a.swapaxes(p["axis1"].GetInt32(), p["axis2"].GetInt32());
                case "ndarray.diagonal": return a.diagonal(0, p["axis1"].GetInt32(), p["axis2"].GetInt32());
                case "ndarray.trace": return a.trace(0, p["axis1"].GetInt32(), p["axis2"].GetInt32());
                case "ndarray.max": return a.max(ax);
                case "ndarray.min": return a.min(ax);
                case "ndarray.sum": return a.sum(ax);
                case "ndarray.prod": return a.prod(ax);
                case "ndarray.mean": return a.mean(ax);
                case "ndarray.std": return a.std(ax);
                case "ndarray.var": return a.var(ax);
                case "ndarray.all": return a.all(ax);
                case "ndarray.any": return a.any(ax);

                // ---- numpy.ma (compared through the data; see the remarks) ----
                case "ma.sum": return AxisArgMaData(np.ma.sum(a, ax), ops);
                case "ma.prod": return AxisArgMaData(np.ma.prod(a, ax), ops);
                case "ma.mean": return AxisArgMaData(np.ma.mean(a, ax), ops);
                case "ma.min": return AxisArgMaData(np.ma.min(a, ax), ops);
                case "ma.max": return AxisArgMaData(np.ma.max(a, ax), ops);
                case "ma.ptp": return AxisArgMaData(np.ma.ptp(a, ax), ops);
                case "ma.std": return AxisArgMaData(np.ma.std(a, ax), ops);
                case "ma.var": return AxisArgMaData(np.ma.var(a, ax), ops);
                case "ma.count": return np.ma.count(a, ax);
                case "ma.cumsum": return AxisArgMaData(np.ma.cumsum(a, ax), ops);
                case "ma.cumprod": return AxisArgMaData(np.ma.cumprod(a, ax), ops);
                case "ma.argmin": return np.ma.argmin(a, ax);
                case "ma.argmax": return np.ma.argmax(a, ax);
                case "ma.all": return AxisArgMaData(np.ma.all(a, ax), ops);
                case "ma.any": return AxisArgMaData(np.ma.any(a, ax), ops);
                case "ma.anom": return AxisArgMaData(np.ma.anom(a, ax), ops);
                case "ma.repeat": return AxisArgMaData(np.ma.repeat(a, 2, ax), ops);
                case "ma.take": return AxisArgMaData(np.ma.take(a, ops[1], ax), ops);
                case "ma.compress": return AxisArgMaData(np.ma.compress(ops[0], ops[1], ax), ops);
                case "ma.sort": return AxisArgMaData(np.ma.sort(a, ax), ops);
                case "ma.argsort": return np.ma.argsort(a, ax);
                case "ma.concatenate": return AxisArgMaData(np.ma.concatenate(new object[] { a, a }, ax), ops);
                case "ma.diff": return AxisArgMaData(np.ma.diff(a, 1, ax), ops);
                case "ma.append": return AxisArgMaData(np.ma.append(a, a, ax), ops);
                case "ma.count_masked": return np.ma.count_masked(np.ma.array(a), ax);
                case "ma.average": return AxisArgMaData(np.ma.average(a, ax), ops);
                case "ma.median": return AxisArgMaData(np.ma.median(a, ax), ops);
                case "ma.apply_along_axis":
                    return WithCallbackResults(made =>
                        AxisArgMaData(np.ma.apply_along_axis(x => made.KeepMasked(np.ma.sum(x)), ax, a), ops));
                case "ma.squeeze": return AxisArgMaData(np.ma.squeeze(a, ax), ops);
                case "ma.compress_nd": return np.ma.compress_nd(a, ax);
                case "ma.mask_rowcols": return AxisArgMaData(np.ma.mask_rowcols(a, ax), ops);
                case "ma.compress_rowcols": return np.ma.compress_rowcols(a, ax);
                case "ma.expand_dims": return AxisArgMaData(np.ma.expand_dims(a, ax), ops);
                case "ma.stack": return AxisArgMaData(np.ma.stack(new object[] { a, a }, ax), ops);
                case "ma.swapaxes":
                    return AxisArgMaData(np.ma.swapaxes(a, p["axis1"].GetInt32(), p["axis2"].GetInt32()), ops);
                case "ma.diagonal":
                    return AxisArgMaData(np.ma.diagonal(a, 0, p["axis1"].GetInt32(), p["axis2"].GetInt32()), ops);
                case "ma.trace": return np.ma.trace(a, 0, p["axis1"].GetInt32(), p["axis2"].GetInt32());
                case "ma.transpose": return AxisArgMaData(np.ma.transpose(a, ParseIntArray(p["axes"])), ops);

                default:
                    throw new NotSupportedException($"axis_arg fn '{fn}' not wired in OpRegistry.AxisArgs");
            }
        }

        /// <summary>
        ///     Replays one tuple-result <c>axis_arg</c> case: <c>np.split</c> / <c>np.array_split</c> /
        ///     <c>np.unstack</c>, and <c>np.gradient</c>, whose single-axis array result the generator records as a
        ///     one-slot tuple so the case kind does not depend on the axis.
        /// </summary>
        /// <param name="p">The case params: <c>fn</c> and <c>axis</c> (an integer, or a list for np.gradient).</param>
        /// <param name="ops">The reconstructed operand.</param>
        /// <returns>The function's arrays, in NumPy's order.</returns>
        /// <exception cref="NotSupportedException">The fn key has no tuple replay here.</exception>
        internal static NDArray[] AxisArgTupleFromCorpus(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            string fn = p["fn"].GetString();
            NDArray a = ops[0];
            switch (fn)
            {
                case "split": return np.split(a, 1, p["axis"].GetInt32());
                case "array_split": return np.array_split(a, 1, p["axis"].GetInt32());
                case "unstack": return np.unstack(a, p["axis"].GetInt32());
                case "gradient":
                {
                    var r = AxisArgIsTuple(p, "axis")
                        ? np.gradient(a, (object)null, ParseIntArray(p["axis"]))
                        : np.gradient(a, axis: p["axis"].GetInt32());
                    return (NDArray[])r;
                }
                default:
                    throw new NotSupportedException($"axis_arg tuple fn '{fn}' not wired in OpRegistry.AxisArgs");
            }
        }

        /// <summary>
        ///     A numpy.ma function over a MASKED copy of <paramref name="a"/> (params.mask <c>"every3"</c>: the flat
        ///     positions divisible by 3 masked, as the generator builds it), so the masked code's axis handling runs.
        /// </summary>
        /// <param name="fn">The <c>ma.</c> function name.</param>
        /// <param name="maskKind">The mask recipe (only <c>"every3"</c>).</param>
        /// <param name="a">The data operand.</param>
        /// <param name="ax">The axis.</param>
        /// <returns>The function's result through its data (NumPy raises for every case of this kind).</returns>
        /// <exception cref="NotSupportedException">An unknown mask recipe or fn key (a corpus/registry drift).</exception>
        private static NDArray AxisArgMasked(string fn, string maskKind, NDArray a, int ax)
        {
            if (maskKind != "every3")
                throw new NotSupportedException($"axis_arg mask '{maskKind}' not wired in OpRegistry.AxisArgs");

            // np.arange(a.size).reshape(a.shape) % 3 == 0, the generator's mask.
            using var flat = np.arange(a.size);
            using var positions = flat.reshape(a.Shape.dimensions);
            using var rem = positions % 3;
            using var mask = rem == 0;
            var m = np.ma.array(a, mask);
            try
            {
                return AxisArgMaskedCall(fn, m, ax, a);
            }
            finally
            {
                // The masked operand is the registry's: its data / mask / fill go back to the pool whether the call
                // returned or raised (the operand itself, a, is never released here).
                ReleaseMaskedParts(m, new[] { a });
            }
        }

        /// <summary>The numpy.ma call of one masked <c>axis_arg</c> case.</summary>
        /// <param name="fn">The <c>ma.</c> function name.</param>
        /// <param name="m">The masked operand.</param>
        /// <param name="ax">The axis.</param>
        /// <param name="a">The data operand <paramref name="m"/> was built over (the case's operand 0).</param>
        /// <returns>The result through its data.</returns>
        /// <exception cref="NotSupportedException">The fn key has no masked replay here.</exception>
        /// <remarks>
        ///     A result may SHARE the masked operand's arrays: NumPy's <c>mask_rowcols</c> returns its input itself when it
        ///     masks nothing more, and <c>np.ma.array(a, mask)</c> keeps <paramref name="a"/> as its data. The result's parts
        ///     are therefore released with the operand's arrays kept; releasing them with an empty keep set disposed the
        ///     case's operand 0 itself (found by the ownership gate: <c>axis_arg/ma.mask_rowcols</c>, three cases).
        /// </remarks>
        private static NDArray AxisArgMaskedCall(string fn, NDMaskedArray m, int ax, NDArray a)
        {
            var keep = new[] { a, m._data, m._mask, m._fill_value };
            switch (fn)
            {
                case "ma.sum": return AxisArgMaData(np.ma.sum(m, ax), keep);
                case "ma.prod": return AxisArgMaData(np.ma.prod(m, ax), keep);
                case "ma.mean": return AxisArgMaData(np.ma.mean(m, ax), keep);
                case "ma.min": return AxisArgMaData(np.ma.min(m, ax), keep);
                case "ma.max": return AxisArgMaData(np.ma.max(m, ax), keep);
                case "ma.ptp": return AxisArgMaData(np.ma.ptp(m, ax), keep);
                case "ma.std": return AxisArgMaData(np.ma.std(m, ax), keep);
                case "ma.var": return AxisArgMaData(np.ma.var(m, ax), keep);
                case "ma.count": return np.ma.count(m, ax);
                case "ma.cumsum": return AxisArgMaData(np.ma.cumsum(m, ax), keep);
                case "ma.cumprod": return AxisArgMaData(np.ma.cumprod(m, ax), keep);
                case "ma.argmin": return np.ma.argmin(m, ax);
                case "ma.argmax": return np.ma.argmax(m, ax);
                case "ma.all": return AxisArgMaData(np.ma.all(m, ax), keep);
                case "ma.any": return AxisArgMaData(np.ma.any(m, ax), keep);
                case "ma.anom": return AxisArgMaData(np.ma.anom(m, ax), keep);
                case "ma.repeat": return AxisArgMaData(np.ma.repeat(m, 2, ax), keep);
                case "ma.sort": return AxisArgMaData(np.ma.sort(m, ax), keep);
                case "ma.argsort": return np.ma.argsort(m, ax);
                case "ma.concatenate": return AxisArgMaData(np.ma.concatenate(new object[] { m, m }, ax), keep);
                case "ma.diff": return AxisArgMaData(np.ma.diff(m, 1, ax), keep);
                case "ma.append": return AxisArgMaData(np.ma.append(m, m, ax), keep);
                case "ma.count_masked": return np.ma.count_masked(m, ax);
                case "ma.average": return AxisArgMaData(np.ma.average(m, ax), keep);
                case "ma.median": return AxisArgMaData(np.ma.median(m, ax), keep);
                case "ma.apply_along_axis":
                    return WithCallbackResults(made =>
                        AxisArgMaData(np.ma.apply_along_axis(x => made.KeepMasked(np.ma.sum(x)), ax, m), keep));
                case "ma.compress_nd": return np.ma.compress_nd(m, ax);
                case "ma.mask_rowcols": return AxisArgMaData(np.ma.mask_rowcols(m, ax), keep);
                case "ma.compress_rowcols": return np.ma.compress_rowcols(m, ax);
                default:
                    throw new NotSupportedException($"axis_arg masked fn '{fn}' not wired in OpRegistry.AxisArgs");
            }
        }

        /// <summary>
        ///     Runs an in-place mutator on the registry's own copy and returns the copy, releasing it when the mutator
        ///     raises: on an error path the copy is a registry temporary, which the leak gate would otherwise count as the
        ///     library's.
        /// </summary>
        /// <param name="copy">The copy to mutate (owned by the caller of this helper).</param>
        /// <param name="mutate">The in-place call.</param>
        /// <returns><paramref name="copy"/>, mutated.</returns>
        /// <remarks>Any exception the mutator raises propagates after the copy is released.</remarks>
        private static NDArray MutatedCopyOrRelease(NDArray copy, Action<NDArray> mutate)
        {
            try
            {
                mutate(copy);
                return copy;
            }
            catch
            {
                copy.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     The arrays a case's callback allocated (one per np.apply_along_axis slice, one per np.apply_over_axes
        ///     step). np.apply_along_axis documents that the callback's results stay the CALLER's — it copies them and never
        ///     disposes one — so the registry, the caller here, releases them after the call; any of them the result itself
        ///     holds is kept.
        /// </summary>
        private sealed class CallbackResults
        {
            /// <summary>The plain arrays the callback returned.</summary>
            internal readonly List<NDArray> Arrays = new List<NDArray>();

            /// <summary>The masked arrays the callback returned.</summary>
            internal readonly List<NDMaskedArray> Masked = new List<NDMaskedArray>();

            /// <summary>Records one plain callback result and hands it back.</summary>
            /// <param name="r">The callback's result.</param>
            /// <returns><paramref name="r"/>.</returns>
            internal NDArray Keep(NDArray r)
            {
                Arrays.Add(r);
                return r;
            }

            /// <summary>Records one masked callback result and hands it back.</summary>
            /// <param name="r">The callback's result.</param>
            /// <returns><paramref name="r"/>.</returns>
            internal NDMaskedArray KeepMasked(NDMaskedArray r)
            {
                Masked.Add(r);
                return r;
            }
        }

        /// <summary>
        ///     Runs a callback-taking call and releases the callback's results afterwards, whether the call returned or
        ///     raised — except a result the returned array IS (np.apply_over_axes returns its last step).
        /// </summary>
        /// <param name="call">The call; it registers every callback result through the <see cref="CallbackResults"/>.</param>
        /// <returns>The call's result.</returns>
        /// <remarks>Any exception the call raises propagates after the release.</remarks>
        private static NDArray WithCallbackResults(Func<CallbackResults, NDArray> call)
        {
            var made = new CallbackResults();
            NDArray result = null;
            try
            {
                result = call(made);
                return result;
            }
            finally
            {
                foreach (var r in made.Arrays)
                    if (!ReferenceEquals(r, result))
                        r.Dispose();
                foreach (var m in made.Masked)
                    ReleaseMaskedParts(m, result is null ? Array.Empty<NDArray>() : new[] { result });
            }
        }

        /// <summary>
        ///     Releases a masked array's data, mask and stored fill, skipping the process singletons (<c>np.ma.nomask</c>,
        ///     the masked constant's data and mask) and every array in <paramref name="keep"/>.
        /// </summary>
        /// <param name="m">The masked array.</param>
        /// <param name="keep">Arrays that must survive (an operand, the case's result).</param>
        private static void ReleaseMaskedParts(NDMaskedArray m, NDArray[] keep)
        {
            if (m is null || m is NDMaskedConstant)
                return;
            foreach (var part in new[] { m._data, m._mask, m._fill_value })
            {
                if (part is null || ReferenceEquals(part, np.ma.nomask) || ReferenceEquals(part, np.ma.NDMasked._data) ||
                    ReferenceEquals(part, np.ma.NDMasked._mask) || keep.Any(o => ReferenceEquals(o, part)))
                    continue;
                part.Dispose();
            }
        }

        /// <summary>
        ///     Rebuilds a dict <c>pad_width</c> from the corpus encoding (<c>[{"k": axis, "t": kind, "v": value}, …]</c>, in
        ///     the Python dict's order) as the C# values the house map gives NumPy's: a Python int a C# int, a bool a bool,
        ///     a tuple a <see cref="ValueTuple{T1,T2}"/>, <c>"array"</c> a typed <c>long[]</c> (an ndarray), <c>"huge"</c> a
        ///     <see cref="System.Numerics.BigInteger"/> past int64.
        /// </summary>
        /// <param name="items">The encoded items.</param>
        /// <returns>The dictionary, in the encoded order (a <see cref="Dictionary{TKey,TValue}"/> enumerates in insertion
        ///     order when nothing is removed, as Python's dict does).</returns>
        /// <exception cref="NotSupportedException">An item kind the encoding does not define (a corpus/registry drift).</exception>
        private static Dictionary<int, object> AxisArgPadWidths(JsonElement items)
        {
            var widths = new Dictionary<int, object>();
            foreach (var item in items.EnumerateArray())
            {
                int key = item.GetProperty("k").GetInt32();
                var v = item.GetProperty("v");
                widths[key] = item.GetProperty("t").GetString() switch
                {
                    "int" => (object)v.GetInt32(),
                    "bool" => v.GetBoolean(),
                    "tuple" => (PadItem(v[0]), PadItem(v[1])),
                    "tuple3" => (PadItem(v[0]), PadItem(v[1]), PadItem(v[2])),
                    "float" => v.GetDouble(),
                    "str" => v.GetString(),
                    "none" => null,
                    "array" => v.EnumerateArray().Select(e => e.GetInt64()).ToArray(),
                    "huge" => System.Numerics.BigInteger.Parse(v.GetString(), System.Globalization.CultureInfo.InvariantCulture),
                    string other => throw new NotSupportedException($"axis_arg pad_width item kind '{other}'"),
                    null => throw new NotSupportedException("axis_arg pad_width item without a kind"),
                };
            }
            return widths;

            // A tuple item: a JSON bool is a Python bool, a JSON number a Python int.
            static object PadItem(JsonElement e) => e.ValueKind is JsonValueKind.True or JsonValueKind.False
                ? e.GetBoolean()
                : e.GetInt32();
        }

        /// <summary>Whether the case's <paramref name="key"/> parameter is a tuple (a JSON list) rather than an integer.</summary>
        /// <param name="p">The case params.</param>
        /// <param name="key">The parameter name.</param>
        /// <returns><c>true</c> for a list, <c>false</c> for an integer or an absent key.</returns>
        private static bool AxisArgIsTuple(IReadOnlyDictionary<string, JsonElement> p, string key) =>
            p.TryGetValue(key, out var e) && e.ValueKind == JsonValueKind.Array;

        /// <summary>
        ///     The stacking family over an EMPTY operand list, by NumPy spelling — the validation NumPy runs before it
        ///     reads any operand ("need at least one array to concatenate" / "need at least one array to stack").
        /// </summary>
        /// <param name="fn">The NumPy function name.</param>
        /// <returns>Never in NumPy's answer (every one raises); a result if NumSharp fails to.</returns>
        /// <exception cref="NotSupportedException">The fn key has no empty-list replay here.</exception>
        private static NDArray AxisArgEmptyStack(string fn)
        {
            switch (fn)
            {
                case "stack": return np.stack(new NDArray[0]);
                case "vstack": return np.vstack();
                case "hstack": return np.hstack();
                case "dstack": return np.dstack();
                case "concatenate": return np.concatenate(new NDArray[0]);
                case "column_stack": return np.column_stack();
                case "ma.stack": return AxisArgMaData(np.ma.stack(new object[0]), Array.Empty<NDArray>());
                case "ma.concatenate": return AxisArgMaData(np.ma.concatenate(new object[0]), Array.Empty<NDArray>());
                case "ma.vstack": return AxisArgMaData(np.ma.vstack(new object[0]), Array.Empty<NDArray>());
                case "ma.hstack": return AxisArgMaData(np.ma.hstack(new object[0]), Array.Empty<NDArray>());
                case "ma.dstack": return AxisArgMaData(np.ma.dstack(new object[0]), Array.Empty<NDArray>());
                default:
                    throw new NotSupportedException($"axis_arg empty-list fn '{fn}' not wired in OpRegistry.AxisArgs");
            }
        }

        /// <summary>
        ///     A numpy.ma result as the corpus compares it: a copy of its data, with the masked array's own data, mask
        ///     and stored fill released.
        /// </summary>
        /// <param name="m">The masked result.</param>
        /// <param name="ops">The case operands, never released here (a result may hand one back as its data).</param>
        /// <returns>A fresh array holding the result's data.</returns>
        /// <remarks>
        ///     The copy is what makes the release safe: the returned array owns its buffer, so releasing the masked
        ///     array's arrays cannot free it. The process singletons (<c>np.ma.nomask</c>, the masked constant's data and
        ///     mask) are never released — disposing one would free a slot the whole process shares.
        /// </remarks>
        private static NDArray AxisArgMaData(NDMaskedArray m, NDArray[] ops)
        {
            var data = m._data.copy();
            ReleaseMaskedParts(m, ops);
            return data;
        }
    }
}
