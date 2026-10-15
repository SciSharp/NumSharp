using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using NumSharp;
using NumSharp.Tests.Fuzz.RandomApi;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The C# half of the array-OBJECT oracle (<c>test/oracle/gen_viewflags_oracle.py</c>): the vehicle keys
    ///     <c>viewflags</c> (a returned array or tuple), <c>outflags</c> (a call with <c>out=</c>) and <c>shapearg</c> (a
    ///     shape-taking call, its shape in a named C# form). <c>params.fn</c> names the NumPy API; every fn here makes the
    ///     SAME call its twin in the generator makes in NumPy, in NumSharp's spelling.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why one dispatcher per vehicle key.</b> The ordinary registry often composes calls around an API (a tuple
    ///     slot picked by <c>which</c>, a partition sampled at its kth positions, a scalar wrapped back into a 0-d array),
    ///     and a composed result has the HARNESS's flags, not the API's. These tiers compare the result's memory facet —
    ///     flags, strides, view / copy / identity against each operand (<c>FuzzCorpusTests.CompareMem</c>) — so every call
    ///     below returns the API's own result object untouched.
    ///     </para>
    ///     <para>
    ///     <b>Operands.</b> Reconstructed by <see cref="FuzzCorpus.Reconstruct"/>, owners as owners (OWNDATA set) and views
    ///     as views, read-only where NumPy's were — so <c>np.asarray(a)</c> returning <c>a</c> itself, or a view of a
    ///     read-only owner staying read-only, is NumSharp's behavior and not the harness's.
    ///     </para>
    /// </remarks>
    public static partial class OpRegistry
    {
        /// <summary>
        ///     Replays one single-array <c>viewflags</c> case: the API <c>params.fn</c> names, with its params.
        /// </summary>
        /// <param name="p">The case params (<c>fn</c> plus the API's own: <c>order</c>, <c>shape</c>, <c>axis</c>, ...).</param>
        /// <param name="ops">The reconstructed operands (none for the creation functions).</param>
        /// <returns>NumSharp's result object itself — a view, a copy, an operand, or a fresh array.</returns>
        /// <exception cref="NotSupportedException">An fn the tier does not know (a generator / registry drift: the tier
        ///     must fail loudly, never skip).</exception>
        internal static NDArray ApplyViewFlags(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            string fn = p["fn"].GetString();
            NDArray a = ops.Length > 0 ? ops[0] : null;
            switch (fn)
            {
                // ---- reshaping views ----
                case "ravel": return p.ContainsKey("order") ? np.ravel(a, VfChar(p, "order")) : np.ravel(a);
                case "ndarray.ravel": return a.ravel(VfChar(p, "order"));
                case "ndarray.flatten": return a.flatten(VfChar(p, "order"));
                case "reshape": return np.reshape(a, VfLongs(p["shape"]));
                case "ndarray.reshape": return a.reshape(new Shape(VfLongs(p["shape"])), VfChar(p, "order"));
                case "squeeze": return VfOptInt(p, "axis") is int sqAx ? np.squeeze(a, sqAx) : np.squeeze(a);
                case "ndarray.squeeze": return a.squeeze(VfOptInt(p, "axis"));
                case "expand_dims": return np.expand_dims(a, p["axis"].GetInt32());
                case "atleast_1d": return np.atleast_1d(a);
                case "atleast_2d": return np.atleast_2d(a);
                case "atleast_3d": return np.atleast_3d(a);
                case "broadcast_to": return np.broadcast_to(a, new Shape(VfLongs(p["shape"])));

                // ---- axis permutations ----
                case "transpose": return np.transpose(a, p.ContainsKey("axes") ? VfInts(p["axes"]) : null);
                case "ndarray.transpose": return p.ContainsKey("axes") ? a.transpose(VfInts(p["axes"])) : a.transpose();
                case "ndarray.T": return a.T;
                case "ndarray.mT": return a.mT;
                case "matrix_transpose": return np.matrix_transpose(a);
                case "permute_dims": return np.permute_dims(a, p.ContainsKey("axes") ? VfInts(p["axes"]) : null);
                case "swapaxes": return np.swapaxes(a, p["a1"].GetInt32(), p["a2"].GetInt32());
                case "ndarray.swapaxes": return a.swapaxes(p["a1"].GetInt32(), p["a2"].GetInt32());
                case "moveaxis": return np.moveaxis(a, p["src"].GetInt32(), p["dst"].GetInt32());
                case "rollaxis": return np.rollaxis(a, p["axis"].GetInt32(), p["start"].GetInt32());
                case "flip":
                    return p.ContainsKey("axes") ? np.flip(a, VfInts(p["axes"])) : np.flip(a, VfOptInt(p, "axis"));
                case "fliplr": return np.fliplr(a);
                case "flipud": return np.flipud(a);
                case "rot90": return np.rot90(a, p["k"].GetInt32());

                // ---- diagonal / complex parts ----
                case "diagonal": return np.diagonal(a, p["offset"].GetInt32(), p["axis1"].GetInt32(), p["axis2"].GetInt32());
                case "ndarray.diagonal": return a.diagonal(p["offset"].GetInt32(), p["axis1"].GetInt32(), p["axis2"].GetInt32());
                case "diag": return np.diag(a, p["k"].GetInt32());
                case "real": return np.real(a);
                case "imag": return np.imag(a);
                case "ndarray.real": return a.real;
                case "ndarray.imag": return a.imag;
                case "conj": return np.conj(a);
                case "conjugate": return np.conjugate(a);
                case "ndarray.conj": return a.conj();
                case "ndarray.conjugate": return a.conjugate();
                case "trim_zeros": return np.trim_zeros(a, p["trim"].GetString());

                // ---- conversions: identity or copy ----
                case "asarray":
                    return np.asarray(a, VfDType(p), p.ContainsKey("order") ? VfChar(p, "order") : 'K', VfOptBool(p, "copy"));
                case "asanyarray": return np.asanyarray((object)a, VfDType(p));
                case "ascontiguousarray": return np.ascontiguousarray(a, VfDType(p));
                case "asfortranarray": return np.asfortranarray(a, VfDType(p));
                case "asmatrix": return np.asmatrix(a);
                case "array": return np.array(a, p["copy"].GetBoolean());
                case "copy": return np.copy(a, p.ContainsKey("order") ? VfChar(p, "order") : 'K');
                case "ndarray.copy": return a.copy(p.ContainsKey("order") ? VfChar(p, "order") : 'C');
                case "require":
                    return np.require(a, null, p["requirements"].EnumerateArray().Select(e => e.GetString()).ToArray());
                case "asarray_chkfinite": return np.asarray_chkfinite(a);
                case "ndarray.astype":
                    return a.astype(VfDType(p), p["copy"].GetBoolean(), p.ContainsKey("order") ? VfChar(p, "order") : 'K');
                case "ndarray.view": return p.ContainsKey("dtype") ? a.view(VfDType(p)) : a.view();
                case "ndarray.getfield": return a.getfield(VfDType(p), p["offset"].GetInt32());
                case "ndarray.byteswap": return a.byteswap();
                case "nan_to_num": return np.nan_to_num(a, p["copy"].GetBoolean());
                case "real_if_close": return np.real_if_close(a, p["tol"].GetDouble());

                // ---- indexing ----
                case "getitem": return a[VfIndexKey(p["index"], ops)];

                // ---- creation (no operand) ----
                case "linspace":
                    return np.linspace(p["start"].GetDouble(), p["stop"].GetDouble(), p["num"].GetInt32(),
                                       p["endpoint"].GetBoolean(), VfDType(p));
                case "logspace":
                    return np.logspace(p["start"].GetDouble(), p["stop"].GetDouble(), p["num"].GetInt32(),
                                       p["endpoint"].GetBoolean(), p["base"].GetDouble(), VfDType(p));
                case "geomspace":
                    return np.geomspace(p["start"].GetDouble(), p["stop"].GetDouble(), p["num"].GetInt32(),
                                        p["endpoint"].GetBoolean(), VfDType(p));
                case "arange":
                    return p.ContainsKey("dtype")
                        ? np.arange(p["start"].GetDouble(), p["stop"].GetDouble(), p["step"].GetDouble(), VfDType(p))
                        : np.arange(p["start"].GetDouble(), p["stop"].GetDouble(), p["step"].GetDouble());

                // ---- einsum ----
                case "einsum": return np.einsum(p["subscripts"].GetString(), ops);

                // ---- the copy family ----
                case "take": return np.take(a, ops[1], VfOptInt(p, "axis"));
                case "compress": return np.compress(ops[0], ops[1], VfOptInt(p, "axis"));
                case "repeat": return np.repeat(a, p["repeats"].GetInt32(), VfOptInt(p, "axis"));
                case "tile": return np.tile(a, VfLongs(p["reps"]));
                case "resize": return np.resize(a, new Shape(VfLongs(p["shape"])));
                case "roll": return np.roll(a, p["shift"].GetInt32(), VfOptInt(p, "axis"));
                case "delete": return np.delete(a, p["obj"].GetInt32(), VfOptInt(p, "axis"));
                case "insert": return np.insert(a, p["obj"].GetInt32(), ops[1], VfOptInt(p, "axis"));
                case "append": return np.append(a, ops[1], VfOptInt(p, "axis"));
                case "sort": return np.sort(a, p.ContainsKey("axis") ? VfOptInt(p, "axis") : -1);
                case "argsort": return np.argsort(a, p.ContainsKey("axis") ? VfOptInt(p, "axis") : -1);
                case "unique": return np.unique(a);
                case "pad": return np.pad(a, p["width"].GetInt32());
                case "argwhere": return np.argwhere(a);
                case "flatnonzero": return np.flatnonzero(a);
                case "diff": return np.diff(a, p["n"].GetInt32(), p["axis"].GetInt32());
                case "cumsum": return np.cumsum(a, VfOptInt(p, "axis"));
                case "cumprod": return np.cumprod(a, VfOptInt(p, "axis"));
                case "concatenate": return np.concatenate((ops[0], ops[1]), p["axis"].GetInt32());
                case "stack": return np.stack(new[] { ops[0], ops[1] }, p["axis"].GetInt32());
                case "hstack": return np.hstack(ops[0], ops[1]);
                case "vstack": return np.vstack(ops[0], ops[1]);
                case "where3": return np.where(ops[0], ops[1], ops[2]);
                case "clip": return np.clip(ops[0], ops[1], ops[2]);
                case "round": return np.round_(a, p["decimals"].GetInt32());
                case "matmul": return np.matmul(ops[0], ops[1]);
                case "dot": return np.dot(ops[0], ops[1]);
                case "outer": return np.outer(ops[0], ops[1]);
                case "kron": return np.kron(ops[0], ops[1]);
                case "inner": return np.inner(ops[0], ops[1]);

                // ---- like-creators (K order by default) ----
                case "zeros_like": return p.ContainsKey("order") ? np.zeros_like(a, null, VfChar(p, "order")) : np.zeros_like(a);
                case "ones_like": return p.ContainsKey("order") ? np.ones_like(a, null, VfChar(p, "order")) : np.ones_like(a);
                case "full_like": return p.ContainsKey("order") ? np.full_like(a, 2, null, VfChar(p, "order")) : np.full_like(a, 2);
                case "empty_like":
                {
                    // Uninitialized: the generator fills NumPy's result with 0 the same way.
                    var r = p.ContainsKey("order") ? np.empty_like(a, null, default, VfChar(p, "order")) : np.empty_like(a);
                    r.fill(0);
                    return r;
                }

                // ---- ufuncs ----
                case "negative": return np.negative(a);
                case "sqrt": return np.sqrt(a);
                case "absolute": return np.absolute(a);
                case "isnan": return np.isnan(a);
                case "square": return np.square(a);
                case "logical_not": return np.logical_not(a);
                case "sign": return np.sign(a);
                case "floor": return np.floor(a);
                case "positive": return np.positive(a);
                case "invert": return np.invert(a);
                case "exp": return np.exp(a);
                case "add": return np.add(ops[0], ops[1]);
                case "subtract": return np.subtract(ops[0], ops[1]);
                case "multiply": return np.multiply(ops[0], ops[1]);
                case "maximum": return np.maximum(ops[0], ops[1]);
                case "equal": return np.equal(ops[0], ops[1]);
                case "less": return np.less(ops[0], ops[1]);
                case "logical_and": return np.logical_and(ops[0], ops[1]);
                case "bitwise_and": return np.bitwise_and(ops[0], ops[1]);
                case "power": return np.power(ops[0], ops[1]);
                case "arctan2": return np.arctan2(ops[0], ops[1]);
                case "floor_divide": return np.floor_divide(ops[0], ops[1]);

                // ---- reductions ----
                case "ptp": return np.ptp(a, VfOptInt(p, "axis"), null, VfKeepdims(p));
                case "median": return np.median(a, VfOptInt(p, "axis"), keepdims: VfKeepdims(p));
                case "average": return np.average(a, VfOptInt(p, "axis"), null, VfKeepdims(p));
                case "sum": case "prod": case "mean": case "std": case "var": case "max": case "min":
                case "all": case "any": case "nansum": case "nanmax": case "argmax": case "argmin":
                    return ApplyReduce(fn, VfOptInt(p, "axis"), VfKeepdims(p), a);

                // ---- V10: triangles, partitions, nan-reductions, the FFT family, creators, ndarray methods ----
                case "tril": return np.tril(a, p["k"].GetInt32());
                case "triu": return np.triu(a, p["k"].GetInt32());
                case "diagflat": return np.diagflat(a, p["k"].GetInt32());
                case "partition": return np.partition(a, p["kth"].GetInt32(), VfOptInt(p, "axis"));
                case "argpartition": return np.argpartition(a, p["kth"].GetInt32(), VfOptInt(p, "axis"));
                case "nanmean": case "nanstd": case "nanvar": case "nanmin": case "nanprod":
                case "nanargmax": case "nanargmin": case "nanmedian":
                    return ApplyReduce(fn, VfOptInt(p, "axis"), VfKeepdims(p), a);
                case "count_nonzero": return np.count_nonzero(a, p["axis"].GetInt32(), VfKeepdims(p));
                case "fft.fft": return np.fft.fft(a, null, p["axis"].GetInt32());
                case "fft.ifft": return np.fft.ifft(a, null, p["axis"].GetInt32());
                case "fft.rfft": return np.fft.rfft(a, null, p["axis"].GetInt32());
                case "fft.irfft": return np.fft.irfft(a, null, p["axis"].GetInt32());
                case "fft.hfft": return np.fft.hfft(a, null, p["axis"].GetInt32());
                case "fft.ihfft": return np.fft.ihfft(a, null, p["axis"].GetInt32());
                case "fft.fft2": return np.fft.fft2(a);
                case "fft.ifft2": return np.fft.ifft2(a);
                case "fft.fftn": return np.fft.fftn(a);
                case "fft.ifftn": return np.fft.ifftn(a);
                case "fft.rfft2": return np.fft.rfft2(a);
                case "fft.rfftn": return np.fft.rfftn(a);
                case "fft.irfft2": return np.fft.irfft2(a);
                case "fft.irfftn": return np.fft.irfftn(a);
                case "fft.fftshift": return p.ContainsKey("axes") ? np.fft.fftshift(a, VfInts(p["axes"])) : np.fft.fftshift(a);
                case "fft.ifftshift": return p.ContainsKey("axes") ? np.fft.ifftshift(a, VfInts(p["axes"])) : np.fft.ifftshift(a);
                case "eye": return np.eye(p["N"].GetInt32(), VfOptInt(p, "M"), p["k"].GetInt32(), null, VfChar(p, "order"));
                case "identity": return np.identity(p["n"].GetInt32());
                case "tri": return np.tri(p["N"].GetInt32(), VfOptInt(p, "M"), p["k"].GetInt32());
                // The slice texts are handed over verbatim: NumSharp's grid DSL parses "start:stop:step" itself.
                case "mgrid": return np.mgrid[p["slices"].EnumerateArray().Select(e => (object)e.GetString()).ToArray()];
                // NumSharp's ndarray methods spell axis=None with an axis-less overload, and NumPy's scalar result of a
                // flat reduction is gated on that overload's own result object.
                case "ndarray.sum":
                    return VfOptInt(p, "axis") is int sAx ? a.sum(sAx, VfKeepdims(p)) : a.sum();
                case "ndarray.mean":
                    return VfOptInt(p, "axis") is int mAx ? a.mean(mAx, VfKeepdims(p)) : a.mean();
                case "ndarray.std":
                    return VfOptInt(p, "axis") is int sdAx ? a.std(sdAx, VfKeepdims(p)) : a.std(VfKeepdims(p));
                case "ndarray.var":
                    return VfOptInt(p, "axis") is int vAx ? a.var(vAx, VfKeepdims(p)) : a.var(VfKeepdims(p));
                case "ndarray.prod": return a.prod(VfOptInt(p, "axis"), null, VfKeepdims(p));
                case "ndarray.argmax": return a.argmax(p["axis"].GetInt32(), VfKeepdims(p));
                case "ndarray.argmin": return a.argmin(p["axis"].GetInt32(), VfKeepdims(p));
                case "ndarray.argsort": return a.argsort(VfOptInt(p, "axis"));
                case "ndarray.argpartition": return a.argpartition(p["kth"].GetInt32(), VfOptInt(p, "axis"));
                case "ndarray.repeat": return a.repeat(p["repeats"].GetInt32(), VfOptInt(p, "axis"));
                case "ndarray.searchsorted": return ops[0].searchsorted(ops[1], p["side"].GetString());
                case "ndarray.dot": return ops[0].dot(ops[1]);

                default:
                    throw new NotSupportedException($"viewflags: unknown fn '{fn}' (generator / registry drift)");
            }
        }

        /// <summary>
        ///     Replays one tuple-result <c>viewflags</c> case.
        /// </summary>
        /// <param name="p">The case params.</param>
        /// <param name="ops">The reconstructed operands.</param>
        /// <returns>The API's result slots, in NumPy's order, as the API returned them.</returns>
        /// <exception cref="NotSupportedException">An fn the tier does not know.</exception>
        internal static NDArray[] ApplyViewFlagsTuple(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            string fn = p["fn"].GetString();
            NDArray a = ops.Length > 0 ? ops[0] : null;
            switch (fn)
            {
                case "split": return np.split(a, p["sections"].GetInt32(), p["axis"].GetInt32());
                case "array_split": return np.array_split(a, p["sections"].GetInt32(), p["axis"].GetInt32());
                case "hsplit": return np.hsplit(a, p["sections"].GetInt32());
                case "vsplit": return np.vsplit(a, p["sections"].GetInt32());
                case "dsplit": return np.dsplit(a, p["sections"].GetInt32());
                case "unstack": return np.unstack(a, p["axis"].GetInt32());
                case "ix_": return np.ix_(ops.Cast<object>().ToArray());
                case "meshgrid":
                    return np.meshgrid(ops[0], ops[1], p["indexing"].GetString(), p["sparse"].GetBoolean(), p["copy"].GetBoolean());
                case "broadcast_arrays": return np.broadcast_arrays(ops);
                case "nonzero": return np.nonzero(a);
                case "where1": return np.where(a);
                case "unravel_index":
                    return np.unravel_index(a, VfInts(p["shape"]), VfChar(p, "order"));
                // ---- V10: the open grid and the index-tuple generators (diag_indices: ONE array in every slot) ----
                case "ogrid": return np.ogrid[p["slices"].EnumerateArray().Select(e => (object)e.GetString()).ToArray()];
                case "diag_indices": return np.diag_indices(p["n"].GetInt32(), p["ndim"].GetInt32());
                case "tril_indices": return np.tril_indices(p["n"].GetInt32(), p["k"].GetInt32());
                case "triu_indices": return np.triu_indices(p["n"].GetInt32(), p["k"].GetInt32());
                case "ndarray.nonzero": return a.nonzero();
                default:
                    throw new NotSupportedException($"viewflags: unknown tuple fn '{fn}' (generator / registry drift)");
            }
        }

        /// <summary>
        ///     Replays one <c>outflags</c> case: the API called with <c>out=</c> (operand <c>params.out</c>).
        /// </summary>
        /// <param name="p">The case params (<c>fn</c>, <c>out</c>, and the API's own).</param>
        /// <param name="ops">The reconstructed operands: the inputs, then the out array (its prior contents included).</param>
        /// <returns>[the returned array, the whole base buffer behind the out array] — the out_where two-slot contract;
        ///     the facet of slot 0 says whether the call returned the out array itself.</returns>
        /// <exception cref="NotSupportedException">An fn the tier does not know.</exception>
        internal static NDArray[] ApplyOutFlags(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            string fn = p["fn"].GetString();
            int oi = p["out"].GetInt32();
            NDArray o = ops[oi];
            NDArray a = ops[0];
            int? axis = VfOptInt(p, "axis");
            NDArray r = fn switch
            {
                "add" => np.add(ops[0], ops[1], o),
                "subtract" => np.subtract(ops[0], ops[1], o),
                "multiply" => np.multiply(ops[0], ops[1], o),
                "maximum" => np.maximum(ops[0], ops[1], o),
                "equal" => np.equal(ops[0], ops[1], o),
                "logical_and" => np.logical_and(ops[0], ops[1], o),
                "bitwise_and" => np.bitwise_and(ops[0], ops[1], o),
                "arctan2" => np.arctan2(ops[0], ops[1], o),
                "negative" => np.negative(a, o),
                "sqrt" => np.sqrt(a, o),
                "absolute" => np.absolute(a, o),
                "isnan" => np.isnan(a, o),
                "sign" => np.sign(a, o),
                "invert" => np.invert(a, o),
                "square" => np.square(a, o),
                "floor" => np.floor(a, o),
                "conjugate" => np.conjugate(a, o),
                "positive" => np.positive(a, o),
                "ptp" => np.ptp(a, axis, o),
                "median" => np.median(a, axis, o),
                "nanmedian" => np.nanmedian(a, axis, o),
                "percentile" => np.percentile(a, p["q"].GetDouble(), axis, o),
                "quantile" => np.quantile(a, p["q"].GetDouble(), axis, o),
                "nanpercentile" => np.nanpercentile(a, p["q"].GetDouble(), axis, o),
                "nanquantile" => np.nanquantile(a, p["q"].GetDouble(), axis, o),
                "cumsum" => np.cumsum(a, axis, null, o),
                "cumprod" => np.cumprod(a, axis, null, o),
                "nancumsum" => np.nancumsum(a, axis, null, o),
                "nancumprod" => np.nancumprod(a, axis, null, o),
                "nanargmax" => np.nanargmax(a, axis, o),
                "nanargmin" => np.nanargmin(a, axis, o),
                "all" => np.all(a, axis, o),
                "any" => np.any(a, axis, o),
                "max" => np.max(a, axis: axis, @out: o),
                "min" => np.min(a, axis: axis, @out: o),
                "trace" => np.trace(a, @out: o),
                "clip" => np.clip(ops[0], ops[1], ops[2], o),
                "round" => np.round_(a, p["decimals"].GetInt32(), o),
                "take" => np.take(ops[0], ops[1], axis, o),
                "compress" => np.compress(ops[0], ops[1], axis, o),
                "choose" => np.choose(ops[0], new[] { ops[1], ops[2] }, o),
                "concatenate" => np.concatenate(new[] { ops[0], ops[1] }, p["axis"].GetInt32(), o),
                "dot" => np.dot(ops[0], ops[1], o),
                "matmul" => np.matmul(ops[0], ops[1], o),
                "vecdot" => np.vecdot(ops[0], ops[1], @out: o),
                "matvec" => np.matvec(ops[0], ops[1], o),
                "vecmat" => np.vecmat(ops[0], ops[1], o),
                "outer" => np.outer(ops[0], ops[1], o),
                "einsum" => np.einsum(p["subscripts"].GetString(), ops.Take(oi).ToArray(), @out: o),
                "fft" => np.fft.fft(a, null, -1, null, o),
                // ---- the generic unary ufunc rows (gen_viewflags_oracle.OUT_UNARY_UFUNCS) ----
                "abs" => np.abs(a, o),
                "fabs" => np.fabs(a, o),
                "cbrt" => np.cbrt(a, o),
                "reciprocal" => np.reciprocal(a, o),
                "ceil" => np.ceil(a, o),
                "trunc" => np.trunc(a, o),
                "rint" => np.rint(a, o),
                "fix" => np.fix(a, o),
                "sin" => np.sin(a, o),
                "cos" => np.cos(a, o),
                "tan" => np.tan(a, o),
                "arcsin" => np.arcsin(a, o),
                "arccos" => np.arccos(a, o),
                "arctan" => np.arctan(a, o),
                "sinh" => np.sinh(a, o),
                "cosh" => np.cosh(a, o),
                "tanh" => np.tanh(a, o),
                "arcsinh" => np.arcsinh(a, o),
                "arccosh" => np.arccosh(a, o),
                "arctanh" => np.arctanh(a, o),
                "asinh" => np.asinh(a, o),
                "acosh" => np.acosh(a, o),
                "atanh" => np.atanh(a, o),
                "exp" => np.exp(a, o),
                "exp2" => np.exp2(a, o),
                "expm1" => np.expm1(a, o),
                "log" => np.log(a, o),
                "log2" => np.log2(a, o),
                "log10" => np.log10(a, o),
                "log1p" => np.log1p(a, o),
                "deg2rad" => np.deg2rad(a, o),
                "rad2deg" => np.rad2deg(a, o),
                "degrees" => np.degrees(a, o),
                "radians" => np.radians(a, o),
                "spacing" => np.spacing(a, o),
                "isinf" => np.isinf(a, o),
                "isfinite" => np.isfinite(a, o),
                "isposinf" => np.isposinf(a, o),
                "isneginf" => np.isneginf(a, o),
                "signbit" => np.signbit(a, o),
                "conj" => np.conj(a, o),
                "logical_not" => np.logical_not(a, o),
                "bitwise_not" => np.bitwise_not(a, o),
                "bitwise_count" => np.bitwise_count(a, o),
                // ---- the generic binary ufunc rows (gen_viewflags_oracle.OUT_BINARY_UFUNCS) ----
                "divide" => np.divide(ops[0], ops[1], o),
                "true_divide" => np.true_divide(ops[0], ops[1], o),
                "floor_divide" => np.floor_divide(ops[0], ops[1], o),
                "mod" => np.mod(ops[0], ops[1], o),
                "remainder" => np.remainder(ops[0], ops[1], o),
                "fmod" => np.fmod(ops[0], ops[1], o),
                "power" => np.power(ops[0], ops[1], o),
                "float_power" => np.float_power(ops[0], ops[1], o),
                "minimum" => np.minimum(ops[0], ops[1], o),
                "fmax" => np.fmax(ops[0], ops[1], o),
                "fmin" => np.fmin(ops[0], ops[1], o),
                "hypot" => np.hypot(ops[0], ops[1], o),
                "copysign" => np.copysign(ops[0], ops[1], o),
                "nextafter" => np.nextafter(ops[0], ops[1], o),
                "logaddexp" => np.logaddexp(ops[0], ops[1], o),
                "logaddexp2" => np.logaddexp2(ops[0], ops[1], o),
                "heaviside" => np.heaviside(ops[0], ops[1], o),
                "not_equal" => np.not_equal(ops[0], ops[1], o),
                "less" => np.less(ops[0], ops[1], o),
                "less_equal" => np.less_equal(ops[0], ops[1], o),
                "greater" => np.greater(ops[0], ops[1], o),
                "greater_equal" => np.greater_equal(ops[0], ops[1], o),
                "logical_or" => np.logical_or(ops[0], ops[1], o),
                "logical_xor" => np.logical_xor(ops[0], ops[1], o),
                "bitwise_or" => np.bitwise_or(ops[0], ops[1], o),
                "bitwise_xor" => np.bitwise_xor(ops[0], ops[1], o),
                "gcd" => np.gcd(ops[0], ops[1], o),
                "lcm" => np.lcm(ops[0], ops[1], o),
                "left_shift" => np.left_shift(ops[0], ops[1], o),
                "right_shift" => np.right_shift(ops[0], ops[1], o),
                "ldexp" => np.ldexp(ops[0], ops[1], o),
                // ---- aliases, the FFT family, multi_dot ----
                "around" => np.around(a, p["decimals"].GetInt32(), o),
                "concat" => np.concat(new[] { ops[0], ops[1] }, p["axis"].GetInt32(), o),
                "multi_dot" => np.linalg.multi_dot(ops.Take(oi).ToArray(), o),
                "ifft" => np.fft.ifft(a, null, -1, null, o),
                "rfft" => np.fft.rfft(a, null, -1, null, o),
                "irfft" => np.fft.irfft(a, null, -1, null, o),
                "hfft" => np.fft.hfft(a, null, -1, null, o),
                "ihfft" => np.fft.ihfft(a, null, -1, null, o),
                "fft2" => np.fft.fft2(a, null, null, null, o),
                "ifft2" => np.fft.ifft2(a, null, null, null, o),
                "fftn" => np.fft.fftn(a, null, null, null, o),
                "ifftn" => np.fft.ifftn(a, null, null, null, o),
                // ---- the ndarray methods with out= (NumPy's ndarray.compress takes the array as its receiver) ----
                "ndarray.all" => a.all(axis, o),
                "ndarray.any" => a.any(axis, o),
                "ndarray.max" => a.max(axis, o),
                "ndarray.min" => a.min(axis, o),
                "ndarray.cumsum" => a.cumsum(axis, null, o),
                "ndarray.cumprod" => a.cumprod(axis, null, o),
                "ndarray.clip" => ops[0].clip(ops[1], ops[2], o),
                "ndarray.round" => a.round(p["decimals"].GetInt32(), o),
                "ndarray.take" => ops[0].take(ops[1], axis, o),
                "ndarray.compress" => ops[1].compress(ops[0], axis, o),
                "ndarray.choose" => ops[0].choose(new[] { ops[1], ops[2] }, o),
                "ndarray.trace" => a.trace(@out: o),
                _ => throw new NotSupportedException($"outflags: unknown fn '{fn}' (generator / registry drift)"),
            };
            return new[] { r, BaseBuffer(o) };
        }

        /// <summary>
        ///     Replays one <c>shapearg</c> case: the shape-taking API called with <c>params.dims</c> handed over in the C#
        ///     form <c>params.form</c> names (<c>Shape</c>, <c>long[]</c>, <c>int[]</c>, <c>int</c>, or <c>view:&lt;layout&gt;</c>
        ///     — a view's Shape of the same dimensions, built by <see cref="ViewSizeLayouts.Build"/>).
        /// </summary>
        /// <param name="p">The case params (<c>fn</c>, <c>dims</c>, <c>form</c>, optional <c>dtype</c>).</param>
        /// <param name="ops">The operand the method or function works on (none for the creators).</param>
        /// <returns>The API's result (single-array fns).</returns>
        /// <exception cref="NotSupportedException">An fn / form pair the tier does not know.</exception>
        internal static NDArray ApplyShapeArg(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            string fn = p["fn"].GetString();
            string form = p["form"].GetString();
            long[] dims = VfLongs(p["dims"]);
            DType dt = VfDType(p);
            NDArray a = ops.Length > 0 ? ops[0] : null;

            // The three shape spellings every overload set takes: a Shape (fresh, or a view's), a long[], an int[].
            Shape S() => form.StartsWith("view:", StringComparison.Ordinal)
                ? ViewSizeLayouts.Build(dims, form.Substring("view:".Length))
                : new Shape(dims);
            int[] I() => dims.Select(d => checked((int)d)).ToArray();
            bool isShape = form == "Shape" || form.StartsWith("view:", StringComparison.Ordinal);

            switch (fn)
            {
                case "zeros":
                case "ones":
                case "empty":
                {
                    // The array / int forms take no dtype: the generator emits them for float64 only (NumPy's default
                    // dtype, which those overloads allocate), so the call is the bare one.
                    NDArray r;
                    if (isShape)
                        r = fn == "zeros" ? np.zeros(S(), dt) : fn == "ones" ? np.ones(S(), dt) : np.empty(S(), dt);
                    else if (form == "long[]")
                        r = fn == "zeros" ? np.zeros(dims) : fn == "ones" ? np.ones(dims) : np.empty(dims);
                    else if (form == "int[]")
                        r = fn == "zeros" ? np.zeros(I()) : fn == "ones" ? np.ones(I()) : np.empty(I());
                    else if (form == "int")
                        r = fn == "zeros" ? np.zeros(I()[0]) : fn == "ones" ? np.ones(I()[0]) : np.empty(I()[0]);
                    else
                        throw new NotSupportedException($"shapearg: form '{form}' of {fn}");
                    if (fn == "empty")
                        r.fill(0);
                    return r;
                }
                case "full":
                    // The array forms take no dtype: the fill value's C# type decides it, so the value is passed as the
                    // dtype's own scalar (3 converted), which is NumPy's np.full(shape, 3, dtype) answer.
                    if (isShape)
                        return np.full(S(), 3, dt);
                    object fill = Convert.ChangeType(3, dt.type, System.Globalization.CultureInfo.InvariantCulture);
                    return form == "long[]" ? np.full(dims, fill)
                         : form == "int[]" ? np.full(I(), fill)
                         : throw new NotSupportedException($"shapearg: form '{form}' of full");
                case "reshape":
                    return isShape ? np.reshape(a, S())
                         : form == "long[]" ? np.reshape(a, dims)
                         : form == "int[]" ? np.reshape(a, I())
                         : throw new NotSupportedException($"shapearg: form '{form}' of reshape");
                case "ndarray.reshape":
                    return isShape ? a.reshape(S())
                         : form == "long[]" ? a.reshape(dims)
                         : form == "int[]" ? a.reshape(I())
                         : throw new NotSupportedException($"shapearg: form '{form}' of ndarray.reshape");
                case "broadcast_to":
                    return np.broadcast_to(a, S());
                case "resize":
                    return np.resize(a, S());
                case "tile":
                    return form == "long[]" ? np.tile(a, dims) : np.tile(a, I());
                case "indices":
                    return isShape ? np.indices(S(), dt)
                         : form == "long[]" ? np.indices(dims, dt)
                         : np.indices(I(), dt);
                case "empty_like":
                {
                    var r = np.empty_like(a, null, S());
                    r.fill(0);
                    return r;
                }
                default:
                    throw new NotSupportedException($"shapearg: unknown fn '{fn}' (generator / registry drift)");
            }
        }

        /// <summary>Replays one tuple-result <c>shapearg</c> case (<c>indices_sparse</c>).</summary>
        /// <param name="p">The case params (<c>fn</c>, <c>dims</c>, <c>form</c>, <c>dtype</c>).</param>
        /// <param name="ops">Unused (the open grid has no operand).</param>
        /// <returns>The open-grid arrays, in NumPy's order.</returns>
        /// <exception cref="NotSupportedException">An fn the tier does not know.</exception>
        internal static NDArray[] ApplyShapeArgTuple(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            string fn = p["fn"].GetString();
            string form = p["form"].GetString();
            long[] dims = VfLongs(p["dims"]);
            DType dt = VfDType(p);
            if (fn != "indices_sparse")
                throw new NotSupportedException($"shapearg: unknown tuple fn '{fn}' (generator / registry drift)");
            if (form.StartsWith("view:", StringComparison.Ordinal))
                return np.indices_sparse(ViewSizeLayouts.Build(dims, form.Substring("view:".Length)), dt);
            return form switch
            {
                "Shape" => np.indices_sparse(new Shape(dims), dt),
                "long[]" => np.indices_sparse(dims, dt),
                "int[]" => np.indices_sparse(dims.Select(d => checked((int)d)).ToArray(), dt),
                _ => throw new NotSupportedException($"shapearg: form '{form}' of indices_sparse"),
            };
        }

        // ---- param readers (Vf-prefixed: the partial class shares one namespace with the other registry files) ----

        /// <summary>A one-character order parameter (<c>"C"</c> / <c>"F"</c> / <c>"A"</c> / <c>"K"</c>).</summary>
        /// <param name="p">The params.</param>
        /// <param name="key">The parameter name.</param>
        /// <returns>The order character.</returns>
        /// <exception cref="KeyNotFoundException">The parameter is absent (callers check first when it is optional).</exception>
        private static char VfChar(IReadOnlyDictionary<string, JsonElement> p, string key) => p[key].GetString()[0];

        /// <summary>An optional int parameter: absent or JSON null is NumPy's <c>None</c>.</summary>
        /// <param name="p">The params.</param>
        /// <param name="key">The parameter name.</param>
        /// <returns>The value, or null.</returns>
        private static int? VfOptInt(IReadOnlyDictionary<string, JsonElement> p, string key)
            => p.TryGetValue(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : null;

        /// <summary>An optional bool parameter: absent or JSON null is NumPy's <c>None</c>.</summary>
        /// <param name="p">The params.</param>
        /// <param name="key">The parameter name.</param>
        /// <returns>The value, or null.</returns>
        private static bool? VfOptBool(IReadOnlyDictionary<string, JsonElement> p, string key)
            => p.TryGetValue(key, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : null;

        /// <summary>The <c>keepdims</c> flag; absent means false.</summary>
        /// <param name="p">The params.</param>
        /// <returns>The flag.</returns>
        private static bool VfKeepdims(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("keepdims", out var v) && v.ValueKind == JsonValueKind.True;

        /// <summary>The optional <c>dtype</c> parameter (a NumPy dtype name) as NumSharp's descriptor; null when absent.</summary>
        /// <param name="p">The params.</param>
        /// <returns>The dtype, or null for NumPy's <c>dtype=None</c>.</returns>
        /// <exception cref="NotSupportedException">The name has no NumSharp type code.</exception>
        private static DType VfDType(IReadOnlyDictionary<string, JsonElement> p)
            => p.TryGetValue("dtype", out var v) && v.ValueKind == JsonValueKind.String ? FuzzCorpus.DtypeToTC(v.GetString()) : null;

        /// <summary>A JSON integer array as <c>long[]</c>.</summary>
        /// <param name="e">The array.</param>
        /// <returns>The values.</returns>
        private static long[] VfLongs(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt64()).ToArray();

        /// <summary>A JSON integer array as <c>int[]</c>.</summary>
        /// <param name="e">The array.</param>
        /// <returns>The values.</returns>
        private static int[] VfInts(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt32()).ToArray();

        /// <summary>
        ///     Rebuilds the index key of a <c>getitem</c> case from its tokens (<c>gen_viewflags_oracle._index_key</c>):
        ///     <c>{"i": n}</c> an integer, <c>{"s": [start, stop, step]}</c> a slice, <c>{"e": 1}</c> the ellipsis,
        ///     <c>{"n": 1}</c> a new axis, <c>{"op": k}</c> operand k (a fancy index or a boolean mask).
        /// </summary>
        /// <param name="tokens">The token array.</param>
        /// <param name="ops">The case's operands.</param>
        /// <returns>The key, for <see cref="NDArray"/>'s <c>this[params object[]]</c> indexer.</returns>
        /// <exception cref="NotSupportedException">An unknown token.</exception>
        private static object[] VfIndexKey(JsonElement tokens, NDArray[] ops)
        {
            var key = new List<object>();
            foreach (var t in tokens.EnumerateArray())
            {
                if (t.TryGetProperty("i", out var i))
                    key.Add(i.GetInt32());
                else if (t.TryGetProperty("s", out var s))
                {
                    long? Part(int k) => s[k].ValueKind == JsonValueKind.Null ? null : s[k].GetInt64();
                    key.Add(new Slice(Part(0), Part(1), Part(2) ?? 1));
                }
                else if (t.TryGetProperty("e", out _))
                    key.Add(Slice.Ellipsis);
                else if (t.TryGetProperty("n", out _))
                    key.Add(Slice.NewAxis);
                else if (t.TryGetProperty("op", out var op))
                    key.Add(ops[op.GetInt32()]);
                else
                    throw new NotSupportedException($"viewflags getitem: unknown index token {t.GetRawText()}");
            }
            return key.ToArray();
        }
    }
}
