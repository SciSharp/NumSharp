using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    // =====================================================================================
    //  np.vectorize / np.frompyfunc — apply an arbitrary C# function element-wise (or per
    //  gufunc core sub-array) across broadcast arrays.
    //
    //  Port of NumPy 2.x numpy.vectorize (numpy/lib/_function_base_impl.py) and numpy.frompyfunc
    //  (numpy/_core/src/umath/umathmodule.c). NumPy's own structure, followed here:
    //
    //    * ELEMENT-WISE (no signature): broadcast the inputs, call pyfunc on each element-tuple,
    //      collect the results. NumPy builds a `frompyfunc` OBJECT-array ufunc and casts the
    //      result to `otypes`; we instead compose over NDExpr.Call + np.evaluate — the SAME fused
    //      single-pass NDIter kernel the library already uses (docs point vectorize users here).
    //      This is not merely faithful, it is far faster: a C# delegate call is ~1-2 ns where
    //      NumPy pays ~50-100 ns of CPython call overhead PER ELEMENT, so even this straight
    //      composition outruns NumPy 10-100x — the delegate dispatch, not the plumbing, is the
    //      cost, exactly as NumPy documents ("provided primarily for convenience, not performance;
    //      the implementation is essentially a for loop").
    //
    //    * SIGNATURE / gufunc ((m,n),(n)->(m)): call pyfunc on core SUB-ARRAYS, iterating the
    //      broadcast shape with np.ndindex — a straight composition over broadcast_to / GetData /
    //      copyto (the apply_along_axis pattern), a move-for-move port of NumPy's
    //      _vectorize_call_with_signature. The per-slice call is coarse (one per broadcast index,
    //      not per element), so there is no data-parallel kernel to emit.
    //
    //  THREE deliberate C# adaptations of NumPy's model, each a consequence of static typing
    //  (all documented on the members that expose them):
    //
    //    1. The OUTPUT DTYPE IS THE DELEGATE'S RETURN TYPE. NumPy has no static return type, so it
    //       must call pyfunc once on the first element JUST to learn the output dtype (and demands
    //       `otypes` when the input is empty, since there is no first element to probe). C# knows
    //       the return type at compile time, so `otypes` is only ever an OVERRIDE (a trailing cast),
    //       and an empty input works with no `otypes` at all — a strict improvement over NumPy,
    //       whose "cannot call `vectorize` on size 0 inputs unless `otypes` is set" error we do not
    //       reproduce for the element-wise path.
    //
    //    2. `excluded` is not a parameter. NumPy's `excluded` marks args passed through unvectorized;
    //       in C# a captured variable (closure) does exactly that with no ceremony — write the
    //       fixed argument into the lambda body. `doc`/`cache` likewise have no C# meaning (the
    //       return type is known, so there is no nout to cache and no docstring to carry).
    //
    //    3. `frompyfunc` produces TYPED arrays, not object arrays. NumSharp has no object dtype, so
    //       the returned callable applies the delegate with the delegate's own typed output rather
    //       than NumPy's PyObject result. Its NumPy-only surface (ufunc.reduce/accumulate/identity)
    //       has no NumSharp analog and is out of scope; `nin`/`nout` are VALIDATED against the
    //       delegate (which carries the truth) rather than defining it as they do in Python.
    // =====================================================================================

    /// <summary>
    ///     A reusable vectorized wrapper around a C# delegate — the object <see cref="np.vectorize(Delegate, DType[])"/>
    ///     and <see cref="np.frompyfunc(Delegate, int, int)"/> return, mirroring NumPy's <c>vectorize</c> class.
    ///     Build it once and call it many times: an element-wise instance re-uses one compiled fused kernel across
    ///     calls (the delegate identity is stable, so <see cref="np.evaluate(NDExpr, NDArray)"/>'s structural cache
    ///     reuses it), so repeated invocation is cheap.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Two modes, fixed at construction by whether a gufunc <c>signature</c> was supplied:
    ///         <b>element-wise</b> (no signature) broadcasts the inputs and applies the delegate per element via a
    ///         single fused <see cref="NDExpr.Call(Delegate, NDExpr[])"/> pass; <b>signature</b> mode applies the
    ///         delegate to core sub-arrays, iterating the broadcast shape (one call per index, not per element).
    ///     </para>
    ///     <para>
    ///         <b>Footgun — multi-output element-wise calls the delegate once PER OUTPUT.</b> A delegate returning a
    ///         <see cref="ValueTuple"/> of N components is evaluated as N independent fused passes (an
    ///         expression-built adapter per component), so the user delegate runs N times per element rather than
    ///         once. This is identical in result to NumPy for a PURE function — which <c>vectorize</c> already
    ///         assumes (NumPy itself calls the function an extra time merely to infer the output dtype) — but a
    ///         delegate with side effects will observe the extra calls. Prefer <see cref="np.nditer{T}(NDArray, bool, char)"/>
    ///         for single-call multi-output over a hot, impure, or expensive function.
    ///     </para>
    ///     <para>
    ///         <b>Ownership (signature mode).</b> The core sub-array views handed to the delegate and every array
    ///         the delegate returns stay the CALLER's: each result is copied into the output and never disposed,
    ///         because a delegate may return an array it keeps using (the core view itself, a held constant). A
    ///         delegate that ALLOCATES a fresh result per core slice therefore leaves those to a GC — run the call
    ///         inside an <see cref="NDScope"/> (yielding the outputs) when deterministic release matters. The
    ///         wrapper's own broadcast views are released before the call returns.
    ///     </para>
    /// </remarks>
    public sealed class Vectorized
    {
        // The wrapped user function. Held as a field (stable identity) so element-wise calls hit the
        // NDExpr structural kernel cache instead of recompiling a kernel per call.
        private readonly Delegate _pyfunc;

        // Output dtype override(s), or null to use the delegate's own return type(s). Length == nout
        // when non-null (validated at construction).
        private readonly DType[] _otypes;

        // The verbatim gufunc signature string, or null for element-wise mode.
        private readonly string _signature;

        // Parsed signature core dimensions (per input / per output), null in element-wise mode.
        private readonly string[][] _inputCoreDims;
        private readonly string[][] _outputCoreDims;

        // Arity of the delegate (number of input arrays consumed).
        private readonly int _nin;

        // Number of outputs. Element-wise: the delegate's return component count (1, or the tuple
        // arity). Signature: the signature's output-argument count (which the delegate's return must
        // match at each call).
        private readonly int _nout;

        // The delegate's parameter CLR types (element-wise dtype targets); the return component CLR
        // types (one entry, or the ValueTuple's members). Used to build multi-output adapters.
        private readonly Type[] _paramTypes;
        private readonly Type[] _returnComponentTypes;

        // Lazily built one-per-output adapters for multi-output element-wise mode: adapter k invokes
        // _pyfunc and returns its k-th tuple component, so NDExpr.Call can drive it as a scalar
        // delegate. Null until first multi-output element-wise call. See the class remarks footgun.
        private Delegate[] _componentAdapters;

        // Signature-mode per-slice invoker, built once at construction. A typed direct call for the
        // common Func<NDArray,...,NDArray> single-output shapes (avoids DynamicInvoke's reflection
        // cost — measured ~0.3-1 µs/slice, the dominant signature-mode overhead); a DynamicInvoke +
        // NormalizeSignatureResult fallback for multi-output (tuple / NDArray[]) returns. Null in
        // element-wise mode.
        private readonly Func<NDArray[], NDArray[]> _signatureInvoker;

        /// <summary>The number of input arrays this callable consumes (the delegate's parameter count).</summary>
        public int nin => _nin;

        /// <summary>
        ///     The number of output arrays this callable produces. For an element-wise callable this is the
        ///     delegate's return component count (1, or the arity of the returned <see cref="ValueTuple"/>); for a
        ///     signature callable it is the signature's output-argument count. Use <see cref="CallMany"/> when this
        ///     exceeds 1.
        /// </summary>
        public int nout => _nout;

        /// <summary>The gufunc signature string this callable was built with, or <c>null</c> for element-wise mode.</summary>
        public string signature => _signature;

        /// <summary>
        ///     Builds the wrapper. Not called directly — reach it through <see cref="np.vectorize(Delegate, DType[])"/>,
        ///     the typed <c>np.vectorize</c> overloads, or <see cref="np.frompyfunc(Delegate, int, int)"/>.
        /// </summary>
        /// <param name="pyfunc">The user function to vectorize.</param>
        /// <param name="otypes">
        ///     Output dtype override, one per output, or <c>null</c> to use the delegate's own return type(s).
        ///     Unlike NumPy this is only ever an override cast (C# knows the return type), so it is rarely needed.
        /// </param>
        /// <param name="signature">A gufunc signature (e.g. <c>"(m,n),(n)->(m)"</c>), or <c>null</c> for element-wise mode.</param>
        /// <exception cref="ArgumentNullException"><paramref name="pyfunc"/> is null.</exception>
        /// <exception cref="ArgumentException">
        ///     A parameter of <paramref name="pyfunc"/> is <see cref="NDArray"/> but no <paramref name="signature"/>
        ///     was given (element-wise mode requires scalar element types), or a return component is not one of the
        ///     15 supported dtypes in element-wise mode.
        /// </exception>
        /// <exception cref="ValueError">
        ///     <paramref name="signature"/> is not a valid gufunc signature, its input-argument count does not match
        ///     the delegate's arity, or <paramref name="otypes"/> length does not match the output count.
        /// </exception>
        internal Vectorized(Delegate pyfunc, DType[] otypes, string signature)
        {
            _pyfunc = pyfunc ?? throw new ArgumentNullException(nameof(pyfunc));

            MethodInfo mi = pyfunc.Method;
            _paramTypes = mi.GetParameters().Select(p => p.ParameterType).ToArray();
            _nin = _paramTypes.Length;
            _returnComponentTypes = ReturnComponentTypes(mi.ReturnType);

            _signature = signature;
            if (signature != null)
            {
                // Signature mode: the delegate operates on NDArray core sub-arrays and the OUTPUT
                // count comes from the signature, not the delegate's return (which may be a runtime
                // NDArray[] whose length is only known per call). Parse first so a malformed
                // signature fails before any input inspection (NumPy's construction order).
                (_inputCoreDims, _outputCoreDims) = ParseGufuncSignature(signature);
                _nout = _outputCoreDims.Length;

                if (_inputCoreDims.Length != _nin)
                    throw new ValueError(
                        $"vectorize signature has {_inputCoreDims.Length} input argument(s) but the delegate takes {_nin}");

                _signatureInvoker = BuildSignatureInvoker();
            }
            else
            {
                // Element-wise mode: nout is the delegate's return component count, and every param
                // AND return component must be a supported scalar dtype (NDExpr.Call would reject an
                // NDArray element type, but the error there is opaque — give a directed one here).
                _nout = _returnComponentTypes.Length;

                foreach (Type pt in _paramTypes)
                    if (pt == typeof(NDArray))
                        throw new ArgumentException(
                            "vectorize: an NDArray parameter means the delegate takes whole sub-arrays — pass a gufunc " +
                            "signature (e.g. \"(n)->()\") to use signature mode. Element-wise mode needs scalar element types.",
                            nameof(pyfunc));

                foreach (Type rt in _returnComponentTypes)
                    if (rt.GetTypeCode() == NPTypeCode.Empty || rt == typeof(NDArray))
                        throw new ArgumentException(
                            $"vectorize: return type {rt.Name} is not one of the 15 supported element dtypes; " +
                            "a delegate returning NDArray needs a gufunc signature.",
                            nameof(pyfunc));
            }

            if (otypes != null)
            {
                if (otypes.Length != _nout)
                    throw new ValueError(
                        $"vectorize: otypes has {otypes.Length} entr(y/ies) but the function has {_nout} output(s)");
                _otypes = otypes;
            }
        }

        /// <summary>
        ///     Applies the wrapper and returns the single output array. The common case: element-wise application
        ///     over broadcast <paramref name="args"/>, or one gufunc pass in signature mode.
        /// </summary>
        /// <param name="args">
        ///     The input arrays, one per <see cref="nin"/>. C# scalars convert implicitly to 0-d arrays, so
        ///     <c>vf.Call(a, 2)</c> broadcasts the scalar as NumPy's <c>vfunc(a, 2)</c> does.
        /// </param>
        /// <returns>
        ///     The result array. In element-wise mode its dtype is the delegate's return type (or <c>otypes[0]</c>
        ///     if given) and its shape is the broadcast of the inputs; in signature mode its shape is the broadcast
        ///     of the non-core dimensions plus the output core dimensions.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="args"/> or any element is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="args"/> length is not <see cref="nin"/>.</exception>
        /// <exception cref="InvalidOperationException">
        ///     This callable produces more than one output — use <see cref="CallMany"/> instead.
        /// </exception>
        /// <exception cref="IncorrectShapeException">The inputs do not broadcast together (message lists every operand shape, as NumPy does).</exception>
        public NDArray Call(params NDArray[] args)
        {
            NDArray[] outputs = CallCore(args);
            if (outputs.Length != 1)
                throw new InvalidOperationException(
                    $"vectorize: this function has {outputs.Length} outputs; use CallMany(...) to receive all of them.");
            return outputs[0];
        }

        /// <summary>
        ///     Applies the wrapper and returns ALL outputs, in the delegate's tuple / signature order. Use this for
        ///     a multi-output callable (<see cref="nout"/> &gt; 1); it also works for a single-output callable
        ///     (returns a length-1 array).
        /// </summary>
        /// <param name="args">The input arrays, one per <see cref="nin"/> (C# scalars convert implicitly to 0-d arrays).</param>
        /// <returns>
        ///     One array per output. <b>Element-wise multi-output invokes the delegate once per output</b> (see the
        ///     class remarks) — identical to NumPy for a pure function.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="args"/> or any element is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="args"/> length is not <see cref="nin"/>.</exception>
        /// <exception cref="IncorrectShapeException">The inputs do not broadcast together.</exception>
        /// <exception cref="ValueError">Signature mode: the delegate returned a different number of outputs than the signature declares.</exception>
        public NDArray[] CallMany(params NDArray[] args) => CallCore(args);

        // Validates arity and dispatches to the element-wise or signature path.
        private NDArray[] CallCore(NDArray[] args)
        {
            if (args is null) throw new ArgumentNullException(nameof(args));
            for (int i = 0; i < args.Length; i++)
                if (args[i] is null)
                    throw new ArgumentNullException(nameof(args), $"argument {i} is null");
            if (args.Length != _nin)
                throw new ArgumentException(
                    $"vectorize: expected {_nin} input array(s), got {args.Length}.", nameof(args));

            return _signature != null ? CallWithSignature(args) : CallElementwise(args);
        }

        // ---------------------------------------------------------------------------------
        //  Element-wise mode — the fused NDExpr.Call fast path.
        // ---------------------------------------------------------------------------------
        private NDArray[] CallElementwise(NDArray[] args)
        {
            var leaves = new NDExpr[args.Length];
            for (int i = 0; i < args.Length; i++)
                leaves[i] = NDExpr.Arr(args[i]);

            if (_nout == 1)
            {
                // One fused pass: broadcast + per-element delegate + typed store, kernel cached by
                // structure. otypes (if any) is a trailing cast, matching NumPy's asarray(out, dtype).
                NDArray r = np.evaluate(NDExpr.Call(_pyfunc, leaves));
                if (_otypes != null)
                    r = r.astype(_otypes[0]);
                return new[] { r };
            }

            // Multi-output: one fused pass per output component. Each adapter invokes _pyfunc and
            // returns Item(k+1), so NDExpr.Call drives it as an ordinary scalar delegate.
            EnsureAdapters();
            var results = new NDArray[_nout];
            for (int k = 0; k < _nout; k++)
            {
                var kLeaves = new NDExpr[args.Length];
                for (int i = 0; i < args.Length; i++)
                    kLeaves[i] = NDExpr.Arr(args[i]);
                NDArray rk = np.evaluate(NDExpr.Call(_componentAdapters[k], kLeaves));
                if (_otypes != null)
                    rk = rk.astype(_otypes[k]);
                results[k] = rk;
            }
            return results;
        }

        // Builds the per-component adapter delegates once (idempotent). Adapter k is an ORDINARY C#
        // closure `(a0, a1, ...) => extractK(_pyfunc(a0, a1, ...))` produced by the generic AdaptN
        // helpers below. This shape is load-bearing: NDExpr.Call reads a delegate's arity and dtypes
        // off its Method, and a compiled Expression.Lambda would expose a hidden Closure-first static
        // Method (arity off by one, param dtypes shifted). An ordinary closure's Method is a clean
        // instance method whose parameters ARE exactly (a0, a1, ...), which NDExpr.Call drives
        // correctly and fast (a Kind.Delegate direct Invoke per element, no reflection in the loop).
        // The reflection here (MakeGenericMethod) runs ONCE, at first multi-output call.
        private void EnsureAdapters()
        {
            if (_componentAdapters != null) return;

            MethodInfo adaptOpen = typeof(Vectorized).GetMethod(
                "Adapt" + _paramTypes.Length, BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new NotSupportedException(
                    $"vectorize: multi-output element-wise is supported for arities 1-4, not {_paramTypes.Length}. " +
                    "Use a gufunc signature, or split the outputs into separate single-output calls.");

            var adapters = new Delegate[_nout];
            for (int k = 0; k < _nout; k++)
            {
                // extractK: (returnTuple t) => t.Item{k+1}. Only ever invoked INSIDE the ordinary
                // closure (never handed to NDExpr.Call), so its own compiled-Expression Method shape
                // is irrelevant — the field access is pure, capturing nothing.
                Type componentType = _returnComponentTypes[k];
                ParameterExpression t = Expression.Parameter(_pyfunc.Method.ReturnType, "t");
                Delegate extract = Expression.Lambda(
                    Expression.GetFuncType(_pyfunc.Method.ReturnType, componentType),
                    Expression.Field(t, "Item" + (k + 1)), t).Compile();

                // AdaptN<...>(pyfunc, extractK) -> the ordinary-closure adapter (typed Func).
                Type[] typeArgs = _paramTypes.Append(_pyfunc.Method.ReturnType).Append(componentType).ToArray();
                adapters[k] = (Delegate)adaptOpen.MakeGenericMethod(typeArgs).Invoke(null, new object[] { _pyfunc, extract });
            }
            _componentAdapters = adapters;
        }

        // The AdaptN helpers: wrap a tuple-returning delegate + a component extractor into an ordinary
        // C# closure of the component's own signature. The closure (a display-class instance method)
        // is what makes NDExpr.Call read the correct arity/dtypes — see EnsureAdapters.
        private static Func<T1, TRk> Adapt1<T1, TTuple, TRk>(Func<T1, TTuple> f, Func<TTuple, TRk> ex)
            => a => ex(f(a));
        private static Func<T1, T2, TRk> Adapt2<T1, T2, TTuple, TRk>(Func<T1, T2, TTuple> f, Func<TTuple, TRk> ex)
            => (a, b) => ex(f(a, b));
        private static Func<T1, T2, T3, TRk> Adapt3<T1, T2, T3, TTuple, TRk>(Func<T1, T2, T3, TTuple> f, Func<TTuple, TRk> ex)
            => (a, b, c) => ex(f(a, b, c));
        private static Func<T1, T2, T3, T4, TRk> Adapt4<T1, T2, T3, T4, TTuple, TRk>(Func<T1, T2, T3, T4, TTuple> f, Func<TTuple, TRk> ex)
            => (a, b, c, d) => ex(f(a, b, c, d));

        // ---------------------------------------------------------------------------------
        //  Signature (gufunc) mode — port of NumPy's _vectorize_call_with_signature.
        // ---------------------------------------------------------------------------------
        private NDArray[] CallWithSignature(NDArray[] args)
        {
            // 1. Broadcast + core dimensions: split each arg into leading (broadcast) and trailing
            //    (core) dims, record core-dim sizes, broadcast the leading parts together.
            (long[] broadcastShape, Dictionary<string, long> dimSizes) = ParseInputDimensions(args, _inputCoreDims);

            // 2. Broadcast each input to broadcastShape + its own core shape.
            long[][] inputShapes = CalculateShapes(broadcastShape, dimSizes, _inputCoreDims);
            var bargs = new NDArray[args.Length];
            try
            {
                for (int i = 0; i < args.Length; i++)
                    bargs[i] = np.broadcast_to(args[i], new Shape(inputShapes[i]));

                NDArray[] outputs = null;
                // Byte size of one broadcast-index slot per output (product of that output's core dims x
                // itemsize), computed once the outputs exist. ndindex walks broadcastShape in C-order and
                // each output is laid out broadcastShape ++ coreShape (also C-order), so the k-th slice is
                // exactly the k-th contiguous slot — WriteSignatureSlot blits straight into it, skipping
                // copyto's per-call broadcast/cast setup (measured the dominant per-slice cost). This is
                // the same contiguous-slot trick NumPy's own apply_along_axis uses.
                long[] slotBytes = null;
                long k = 0;   // C-order slice counter == flat index over the broadcast dimensions

                // Reused per-iteration scratch: the index buffer (filled from the allocation-free AsSpans
                // stream — GetData never retains it) and the core-args array (a container of transient
                // views). Reusing both keeps the hot loop from allocating a long[] and an NDArray[] per
                // slice; only the input sub-array views themselves are unavoidably fresh each step.
                long[] idxBuf = broadcastShape.Length == 0 ? Array.Empty<long>() : new long[broadcastShape.Length];
                var coreArgs = new NDArray[bargs.Length];

                // 3. Iterate the broadcast index space; each step hands the delegate the core sub-arrays.
                foreach (ReadOnlySpan<long> index in np.ndindex(broadcastShape).AsSpans())
                {
                    index.CopyTo(idxBuf);
                    for (int i = 0; i < bargs.Length; i++)
                        coreArgs[i] = bargs[i].GetData(idxBuf);   // index the leading dims → core sub-array

                    NDArray[] results = _signatureInvoker(coreArgs);
                    if (results.Length != _nout)
                        throw new ValueError(
                            $"wrong number of outputs from pyfunc: expected {_nout}, got {results.Length}");

                    // The FIRST result fixes any new output core-dim sizes (e.g. k in (n),(m)->(k)) and,
                    // with otypes absent, the output dtypes — so allocate the outputs here, once.
                    if (outputs is null)
                    {
                        for (int j = 0; j < _nout; j++)
                            UpdateDimSizes(dimSizes, results[j], _outputCoreDims[j]);
                        outputs = CreateArrays(broadcastShape, dimSizes, _outputCoreDims, _otypes, results);

                        slotBytes = new long[_nout];
                        for (int j = 0; j < _nout; j++)
                        {
                            long coreSize = 1;
                            foreach (string d in _outputCoreDims[j]) coreSize *= dimSizes[d];
                            slotBytes[j] = coreSize * outputs[j].dtypesize;
                        }
                    }

                    for (int j = 0; j < _nout; j++)
                        WriteSignatureSlot(outputs[j], idxBuf, k, results[j], slotBytes[j]);

                    k++;
                }

                // 4. Never called (a broadcast dimension was 0): NumPy needs otypes to know the dtype and
                //    cannot invent an unknown output core dim without running the function.
                if (outputs is null)
                {
                    if (_otypes is null)
                        throw new ValueError(
                            "cannot call `vectorize` on size 0 inputs unless `otypes` is set");
                    foreach (string[] dims in _outputCoreDims)
                        foreach (string dim in dims)
                            if (!dimSizes.ContainsKey(dim))
                                throw new ValueError(
                                    "cannot call `vectorize` with a signature including new output dimensions on size 0 inputs");
                    outputs = CreateArrays(broadcastShape, dimSizes, _outputCoreDims, _otypes, results: null);
                }

                return outputs;
            }
            finally
            {
                // The broadcast views are this method's own (broadcast_to always returns a NEW view) and are
                // never handed to the delegate — the per-slice core views are re-wrapped from them — so they
                // are released on every path, the delegate throwing included. The core views and whatever the
                // delegate returns stay the caller's (see the class remarks on ownership).
                foreach (var b in bargs)
                    b?.Dispose();
            }
        }

        // Writes one gufunc result into its C-order slot. The blit fast path handles the common case
        // (result matches the slot verbatim — same dtype, contiguous, exact core size, output fits a
        // Span): a raw byte copy into the k-th contiguous block, no copyto. Everything else — a
        // differently-shaped result NumPy would broadcast, a strided/cast result, or an output too
        // large for a 32-bit Span — falls back to NumPy's `output[index] = result` (broadcast + cast).
        private static void WriteSignatureSlot(NDArray output, long[] index, long k, NDArray result, long slotBytes)
        {
            if (output.nbytes <= int.MaxValue
                && result.dtype == output.dtype
                && result.Shape.IsContiguous
                && result.size * result.dtypesize == slotBytes)
            {
                Span<byte> dst = output.Unsafe.Bytes();
                result.Unsafe.ReadOnlyBytes().CopyTo(dst.Slice((int)(k * slotBytes), (int)slotBytes));
                return;
            }
            np.copyto(output.GetData(index), result, casting: "unsafe");
        }

        // ---- signature helpers (ports of the NumPy _function_base_impl.py helpers) ----

        // Port of _parse_input_dimensions: records/validates core-dim sizes and broadcasts the
        // leading (non-core) shapes together into the common broadcast shape.
        private static (long[] broadcastShape, Dictionary<string, long> dimSizes) ParseInputDimensions(
            NDArray[] args, string[][] inputCoreDims)
        {
            var dimSizes = new Dictionary<string, long>();
            var leadingShapes = new Shape[args.Length];
            for (int i = 0; i < args.Length; i++)
            {
                UpdateDimSizes(dimSizes, args[i], inputCoreDims[i]);
                long[] dims = args[i].Shape.dimensions;
                int lead = dims.Length - inputCoreDims[i].Length;   // # of broadcast (non-core) dims
                var leading = new long[lead];
                Array.Copy(dims, 0, leading, 0, lead);
                leadingShapes[i] = new Shape(leading);
            }
            // broadcast_shapes is NumPy's _broadcast_shape over the non-core leading shapes. A 0-d
            // result (all inputs entirely core) exposes a null/empty dimension array — normalize to
            // an empty long[] so np.ndindex() yields exactly one (empty) index, matching NumPy.
            long[] broadcastShape = np.broadcast_shapes(leadingShapes).dimensions ?? Array.Empty<long>();
            return (broadcastShape, dimSizes);
        }

        // Port of _update_dim_sizes: reads the trailing core dims off `arg`, checking each named
        // dimension is consistent with sizes already seen.
        private static void UpdateDimSizes(Dictionary<string, long> dimSizes, NDArray arg, string[] coreDims)
        {
            if (coreDims.Length == 0) return;

            long[] dims = arg.Shape.dimensions;
            if (dims.Length < coreDims.Length)
                throw new ValueError(
                    $"{dims.Length}-dimensional argument does not have enough dimensions for all core dimensions ({string.Join(", ", coreDims)})");

            int start = dims.Length - coreDims.Length;
            for (int d = 0; d < coreDims.Length; d++)
            {
                long size = dims[start + d];
                string name = coreDims[d];
                if (dimSizes.TryGetValue(name, out long existing))
                {
                    if (size != existing)
                        throw new ValueError(
                            $"inconsistent size for core dimension '{name}': {size} vs {existing}");
                }
                else
                {
                    dimSizes[name] = size;
                }
            }
        }

        // Port of _calculate_shapes: broadcastShape ++ [dimSizes[d] for d in coreDims], per operand.
        private static long[][] CalculateShapes(long[] broadcastShape, Dictionary<string, long> dimSizes, string[][] coreDimsList)
        {
            var shapes = new long[coreDimsList.Length][];
            for (int i = 0; i < coreDimsList.Length; i++)
            {
                string[] core = coreDimsList[i];
                var shape = new long[broadcastShape.Length + core.Length];
                Array.Copy(broadcastShape, 0, shape, 0, broadcastShape.Length);
                for (int d = 0; d < core.Length; d++)
                    shape[broadcastShape.Length + d] = dimSizes[core[d]];
                shapes[i] = shape;
            }
            return shapes;
        }

        // Port of _create_arrays: one output per output-core-dim group. With `results` given, an
        // absent otype takes that result's dtype (empty_like); otherwise otypes must supply them.
        private static NDArray[] CreateArrays(long[] broadcastShape, Dictionary<string, long> dimSizes,
            string[][] outputCoreDims, DType[] dtypes, NDArray[] results)
        {
            long[][] shapes = CalculateShapes(broadcastShape, dimSizes, outputCoreDims);
            var arrays = new NDArray[shapes.Length];
            for (int j = 0; j < shapes.Length; j++)
            {
                DType dtype = dtypes?[j];
                arrays[j] = results != null
                    ? np.empty_like(results[j], dtype, new Shape(shapes[j]))
                    : np.empty(new Shape(shapes[j]), dtype);   // size-0 path: dtype is otypes[j], never null
            }
            return arrays;
        }

        // Builds the per-slice invoker once (see the field's remarks). The typed cases skip
        // DynamicInvoke entirely; the fallback boxes the args and normalizes an arbitrary return.
        private Func<NDArray[], NDArray[]> BuildSignatureInvoker()
        {
            switch (_pyfunc)
            {
                case Func<NDArray, NDArray> f:
                    return a => new[] { f(a[0]) };
                case Func<NDArray, NDArray, NDArray> f:
                    return a => new[] { f(a[0], a[1]) };
                case Func<NDArray, NDArray, NDArray, NDArray> f:
                    return a => new[] { f(a[0], a[1], a[2]) };
                default:
                    // Multi-output (tuple / NDArray[]) or an arity beyond the typed cases: reflect.
                    return a =>
                    {
                        var boxed = new object[a.Length];
                        for (int i = 0; i < a.Length; i++)
                            boxed[i] = a[i];
                        return NormalizeSignatureResult(_pyfunc.DynamicInvoke(boxed));
                    };
            }
        }

        // Normalizes a signature-mode delegate result (NDArray, NDArray[], or a ValueTuple of
        // NDArrays) into a flat NDArray[] so the loop can index each output uniformly.
        private static NDArray[] NormalizeSignatureResult(object raw)
        {
            switch (raw)
            {
                case null:
                    throw new ValueError("vectorize: pyfunc returned null");
                case NDArray single:
                    return new[] { single };
                case NDArray[] many:
                    return many;
                case ITuple tuple:
                    var arr = new NDArray[tuple.Length];
                    for (int i = 0; i < tuple.Length; i++)
                        arr[i] = tuple[i] as NDArray
                            ?? throw new ValueError($"vectorize: pyfunc tuple element {i} is not an NDArray");
                    return arr;
                default:
                    throw new ValueError(
                        $"vectorize: pyfunc returned {raw.GetType().Name}; signature-mode functions must return NDArray, NDArray[], or a tuple of NDArray");
            }
        }

        // ---- gufunc signature grammar (port of _parse_gufunc_signature + its regexes) ----

        // NumPy's _DIMENSION_NAME=\w+, _CORE_DIMENSION_LIST=(?:\w+(?:,\w+)*)?, _ARGUMENT=\(list\),
        // _ARGUMENT_LIST=arg(?:,arg)*, _SIGNATURE=^args->args$. Whitespace is stripped before matching.
        private const string CoreDimList = @"(?:\w+(?:,\w+)*)?";
        private static readonly Regex SignatureRegex = new(
            $@"^\({CoreDimList}\)(?:,\({CoreDimList}\))*->\({CoreDimList}\)(?:,\({CoreDimList}\))*$",
            RegexOptions.Compiled);
        private static readonly Regex ArgumentRegex = new(@"\([^)]*\)", RegexOptions.Compiled);
        private static readonly Regex DimNameRegex = new(@"\w+", RegexOptions.Compiled);

        /// <summary>
        ///     Parses a gufunc signature such as <c>"(m,n),(n)->(m)"</c> into per-input and per-output core
        ///     dimension-name lists — the port of NumPy's <c>_parse_gufunc_signature</c>.
        /// </summary>
        /// <param name="signature">The signature string; whitespace anywhere is ignored.</param>
        /// <returns>A pair of jagged arrays: the input arguments' core dims and the output arguments' core dims.</returns>
        /// <exception cref="ValueError">The string is not a valid gufunc signature.</exception>
        private static (string[][] inputs, string[][] outputs) ParseGufuncSignature(string signature)
        {
            string s = Regex.Replace(signature, @"\s+", "");
            if (!SignatureRegex.IsMatch(s))
                throw new ValueError($"not a valid gufunc signature: {signature}");

            int arrow = s.IndexOf("->", StringComparison.Ordinal);
            return (ParseArgList(s.Substring(0, arrow)), ParseArgList(s.Substring(arrow + 2)));

            static string[][] ParseArgList(string argList)
            {
                MatchCollection argMatches = ArgumentRegex.Matches(argList);
                var res = new string[argMatches.Count][];
                for (int i = 0; i < argMatches.Count; i++)
                {
                    MatchCollection dims = DimNameRegex.Matches(argMatches[i].Value);
                    var names = new string[dims.Count];
                    for (int d = 0; d < dims.Count; d++)
                        names[d] = dims[d].Value;
                    res[i] = names;
                }
                return res;
            }
        }

        // Splits a delegate return type into its output component CLR types: the whole type for a
        // scalar return, or the ValueTuple's members for a tuple return (arity 2-7).
        private static Type[] ReturnComponentTypes(Type returnType)
        {
            if (returnType.IsGenericType)
            {
                Type def = returnType.GetGenericTypeDefinition();
                if (def == typeof(ValueTuple<,>) || def == typeof(ValueTuple<,,>) ||
                    def == typeof(ValueTuple<,,,>) || def == typeof(ValueTuple<,,,,>) ||
                    def == typeof(ValueTuple<,,,,,>) || def == typeof(ValueTuple<,,,,,,>))
                    return returnType.GetGenericArguments();
            }
            return new[] { returnType };
        }
    }

    public static partial class np
    {
        // =================================================================================
        //  np.vectorize — element-wise (typed delegate → Func) fast-path overloads.
        //
        //  Each returns a Func<NDArray,...,NDArray> so the result reads exactly like NumPy's
        //  vfunc(a, b): the returned delegate wraps ONE reusable Vectorized (built here), so
        //  repeated calls reuse a single compiled fused kernel. otypes (rarely needed in C#,
        //  since the return type IS the dtype) casts the result.
        // =================================================================================

        /// <summary>
        ///     Vectorizes a one-argument scalar function into a broadcasting element-wise operation. The returned
        ///     delegate reads like NumPy's <c>vfunc(a)</c>: apply it to an array of any layout and it returns a fresh
        ///     array of <typeparamref name="TR"/> whose shape is the input's.
        /// </summary>
        /// <typeparam name="T1">The element type the function consumes; inputs are cast to it (NEP50 at the edge).</typeparam>
        /// <typeparam name="TR">The element type the function returns; the result array's dtype (unless <paramref name="otypes"/> overrides).</typeparam>
        /// <param name="pyfunc">The per-element function.</param>
        /// <param name="otypes">Optional single output-dtype override (a trailing cast); usually unnecessary since <typeparamref name="TR"/> already fixes the dtype.</param>
        /// <returns>A delegate applying <paramref name="pyfunc"/> element-wise across a broadcast input.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="pyfunc"/> is null.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="TR"/> is not one of the 15 supported dtypes.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html</remarks>
        public static Func<NDArray, NDArray> vectorize<T1, TR>(Func<T1, TR> pyfunc, DType[] otypes = null)
        {
            var v = new Vectorized(pyfunc, otypes, signature: null);
            return a => v.Call(a);
        }

        /// <summary>
        ///     Vectorizes a two-argument scalar function into a broadcasting element-wise operation. The two inputs
        ///     broadcast together (NumPy rules); the returned delegate reads like <c>vfunc(a, b)</c>.
        /// </summary>
        /// <typeparam name="T1">First argument's element type.</typeparam>
        /// <typeparam name="T2">Second argument's element type.</typeparam>
        /// <typeparam name="TR">Return element type (the result dtype unless <paramref name="otypes"/> overrides).</typeparam>
        /// <param name="pyfunc">The per-element function.</param>
        /// <param name="otypes">Optional single output-dtype override.</param>
        /// <returns>A delegate applying <paramref name="pyfunc"/> element-wise across the broadcast of its two inputs.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="pyfunc"/> is null.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="TR"/> is not one of the 15 supported dtypes.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html</remarks>
        public static Func<NDArray, NDArray, NDArray> vectorize<T1, T2, TR>(Func<T1, T2, TR> pyfunc, DType[] otypes = null)
        {
            var v = new Vectorized(pyfunc, otypes, signature: null);
            return (a, b) => v.Call(a, b);
        }

        /// <summary>
        ///     Vectorizes a three-argument scalar function into a broadcasting element-wise operation (all three
        ///     inputs broadcast together).
        /// </summary>
        /// <typeparam name="T1">First argument's element type.</typeparam>
        /// <typeparam name="T2">Second argument's element type.</typeparam>
        /// <typeparam name="T3">Third argument's element type.</typeparam>
        /// <typeparam name="TR">Return element type.</typeparam>
        /// <param name="pyfunc">The per-element function.</param>
        /// <param name="otypes">Optional single output-dtype override.</param>
        /// <returns>A delegate applying <paramref name="pyfunc"/> element-wise across the broadcast of its three inputs.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="pyfunc"/> is null.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="TR"/> is not one of the 15 supported dtypes.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html</remarks>
        public static Func<NDArray, NDArray, NDArray, NDArray> vectorize<T1, T2, T3, TR>(Func<T1, T2, T3, TR> pyfunc, DType[] otypes = null)
        {
            var v = new Vectorized(pyfunc, otypes, signature: null);
            return (a, b, c) => v.Call(a, b, c);
        }

        /// <summary>
        ///     Vectorizes a four-argument scalar function into a broadcasting element-wise operation (all four
        ///     inputs broadcast together) — the widest element-wise arity with a typed <c>Func</c> return. For more
        ///     inputs, build a <see cref="Vectorized"/> via <see cref="vectorize(Delegate, DType[])"/> and call <c>.Call</c>.
        /// </summary>
        /// <typeparam name="T1">First argument's element type.</typeparam>
        /// <typeparam name="T2">Second argument's element type.</typeparam>
        /// <typeparam name="T3">Third argument's element type.</typeparam>
        /// <typeparam name="T4">Fourth argument's element type.</typeparam>
        /// <typeparam name="TR">Return element type.</typeparam>
        /// <param name="pyfunc">The per-element function.</param>
        /// <param name="otypes">Optional single output-dtype override.</param>
        /// <returns>A delegate applying <paramref name="pyfunc"/> element-wise across the broadcast of its four inputs.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="pyfunc"/> is null.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="TR"/> is not one of the 15 supported dtypes.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html</remarks>
        public static Func<NDArray, NDArray, NDArray, NDArray, NDArray> vectorize<T1, T2, T3, T4, TR>(Func<T1, T2, T3, T4, TR> pyfunc, DType[] otypes = null)
        {
            var v = new Vectorized(pyfunc, otypes, signature: null);
            return (a, b, c, d) => v.Call(a, b, c, d);
        }

        // =================================================================================
        //  np.vectorize — multi-output element-wise (tuple-returning delegate → Vectorized).
        //
        //  A ValueTuple return binds these (a constructed generic 2nd type arg is more specific
        //  than the plain TR of the single-output overloads), so `np.vectorize(x => (x+1, x-1))`
        //  yields a Vectorized whose .CallMany returns both outputs. See the class footgun: each
        //  output is one fused pass, so the delegate runs nout times per element.
        // =================================================================================

        /// <summary>Vectorizes a one-argument function returning TWO components; call the result's <see cref="Vectorized.CallMany"/> for both outputs.</summary>
        /// <typeparam name="T1">Argument element type.</typeparam>
        /// <typeparam name="TR1">First output element type.</typeparam>
        /// <typeparam name="TR2">Second output element type.</typeparam>
        /// <param name="pyfunc">The per-element function returning a 2-tuple.</param>
        /// <param name="otypes">Optional per-output dtype overrides (length 2 if given).</param>
        /// <returns>A <see cref="Vectorized"/> producing two arrays.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="pyfunc"/> is null.</exception>
        /// <exception cref="ArgumentException">A return component is not a supported dtype.</exception>
        /// <exception cref="ValueError"><paramref name="otypes"/> length is not 2.</exception>
        public static Vectorized vectorize<T1, TR1, TR2>(Func<T1, (TR1, TR2)> pyfunc, DType[] otypes = null)
            => new Vectorized(pyfunc, otypes, signature: null);

        /// <summary>Vectorizes a one-argument function returning THREE components; call the result's <see cref="Vectorized.CallMany"/> for all outputs.</summary>
        /// <typeparam name="T1">Argument element type.</typeparam>
        /// <typeparam name="TR1">First output element type.</typeparam>
        /// <typeparam name="TR2">Second output element type.</typeparam>
        /// <typeparam name="TR3">Third output element type.</typeparam>
        /// <param name="pyfunc">The per-element function returning a 3-tuple.</param>
        /// <param name="otypes">Optional per-output dtype overrides (length 3 if given).</param>
        /// <returns>A <see cref="Vectorized"/> producing three arrays.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="pyfunc"/> is null.</exception>
        /// <exception cref="ArgumentException">A return component is not a supported dtype.</exception>
        /// <exception cref="ValueError"><paramref name="otypes"/> length is not 3.</exception>
        public static Vectorized vectorize<T1, TR1, TR2, TR3>(Func<T1, (TR1, TR2, TR3)> pyfunc, DType[] otypes = null)
            => new Vectorized(pyfunc, otypes, signature: null);

        /// <summary>Vectorizes a two-argument function returning TWO components; the inputs broadcast together and <see cref="Vectorized.CallMany"/> returns both outputs.</summary>
        /// <typeparam name="T1">First argument element type.</typeparam>
        /// <typeparam name="T2">Second argument element type.</typeparam>
        /// <typeparam name="TR1">First output element type.</typeparam>
        /// <typeparam name="TR2">Second output element type.</typeparam>
        /// <param name="pyfunc">The per-element function returning a 2-tuple.</param>
        /// <param name="otypes">Optional per-output dtype overrides (length 2 if given).</param>
        /// <returns>A <see cref="Vectorized"/> producing two arrays.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="pyfunc"/> is null.</exception>
        /// <exception cref="ArgumentException">A return component is not a supported dtype.</exception>
        /// <exception cref="ValueError"><paramref name="otypes"/> length is not 2.</exception>
        public static Vectorized vectorize<T1, T2, TR1, TR2>(Func<T1, T2, (TR1, TR2)> pyfunc, DType[] otypes = null)
            => new Vectorized(pyfunc, otypes, signature: null);

        // =================================================================================
        //  np.vectorize — signature (gufunc) mode: delegate over NDArray core sub-arrays.
        //  The required `signature` string disambiguates these from the element-wise overloads.
        // =================================================================================

        /// <summary>
        ///     Builds a signature (gufunc) vectorizer over a one-argument function that consumes and returns
        ///     <see cref="NDArray"/> core sub-arrays — e.g. <c>signature: "(n)->()"</c> to reduce each 1-D slice.
        ///     Call <see cref="Vectorized.Call"/> (single output) or <see cref="Vectorized.CallMany"/> (a tuple / array return).
        /// </summary>
        /// <param name="pyfunc">Applied to each core sub-array; receives an <see cref="NDArray"/> of the input core shape.</param>
        /// <param name="signature">The gufunc signature (input side must have exactly one argument to match the arity).</param>
        /// <param name="otypes">Optional per-output dtype overrides; when absent the first result's dtype is used.</param>
        /// <returns>A <see cref="Vectorized"/> in signature mode.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="pyfunc"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="signature"/> is malformed or its input arity is not 1.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html</remarks>
        public static Vectorized vectorize(Func<NDArray, NDArray> pyfunc, string signature, DType[] otypes = null)
            => new Vectorized(pyfunc, otypes, signature);

        /// <summary>Builds a signature (gufunc) vectorizer over a two-argument function on <see cref="NDArray"/> core sub-arrays (e.g. <c>"(m,n),(n)->(m)"</c>).</summary>
        /// <param name="pyfunc">Applied to each pair of core sub-arrays.</param>
        /// <param name="signature">The gufunc signature (input side must have exactly two arguments).</param>
        /// <param name="otypes">Optional per-output dtype overrides.</param>
        /// <returns>A <see cref="Vectorized"/> in signature mode.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="pyfunc"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="signature"/> is malformed or its input arity is not 2.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html</remarks>
        public static Vectorized vectorize(Func<NDArray, NDArray, NDArray> pyfunc, string signature, DType[] otypes = null)
            => new Vectorized(pyfunc, otypes, signature);

        /// <summary>Builds a signature (gufunc) vectorizer over a three-argument function on <see cref="NDArray"/> core sub-arrays.</summary>
        /// <param name="pyfunc">Applied to each triple of core sub-arrays.</param>
        /// <param name="signature">The gufunc signature (input side must have exactly three arguments).</param>
        /// <param name="otypes">Optional per-output dtype overrides.</param>
        /// <returns>A <see cref="Vectorized"/> in signature mode.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="pyfunc"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="signature"/> is malformed or its input arity is not 3.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html</remarks>
        public static Vectorized vectorize(Func<NDArray, NDArray, NDArray, NDArray> pyfunc, string signature, DType[] otypes = null)
            => new Vectorized(pyfunc, otypes, signature);

        // =================================================================================
        //  np.vectorize — the general object form (any Delegate). Reachable by casting a
        //  concrete delegate to Delegate, e.g. np.vectorize((Delegate)f, otypes) — the escape
        //  hatch for arities/output-counts the typed overloads above do not spell.
        // =================================================================================

        /// <summary>
        ///     Builds a <see cref="Vectorized"/> from any delegate — the general element-wise (no signature) form.
        ///     Reach it by casting a concrete delegate to <see cref="Delegate"/> (<c>np.vectorize((Delegate)f)</c>);
        ///     a bare lambda binds the typed overloads instead. Supports any arity and multi-output (a
        ///     <see cref="ValueTuple"/> return) via <see cref="Vectorized.CallMany"/>.
        /// </summary>
        /// <param name="pyfunc">The per-element function (any supported arity; may return a tuple for multi-output).</param>
        /// <param name="otypes">Optional per-output dtype overrides (length must equal the output count).</param>
        /// <returns>A <see cref="Vectorized"/> in element-wise mode.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="pyfunc"/> is null.</exception>
        /// <exception cref="ArgumentException">A parameter is <see cref="NDArray"/> (needs a signature) or a return component is not a supported dtype.</exception>
        /// <exception cref="ValueError"><paramref name="otypes"/> length does not match the output count.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html</remarks>
        public static Vectorized vectorize(Delegate pyfunc, DType[] otypes = null)
            => new Vectorized(pyfunc, otypes, signature: null);

        /// <summary>
        ///     Builds a signature-mode <see cref="Vectorized"/> from any delegate — the general gufunc form for
        ///     arities or multi-output shapes the typed signature overloads do not spell. Reach it by casting to
        ///     <see cref="Delegate"/>.
        /// </summary>
        /// <param name="pyfunc">Applied to core sub-arrays; must take/return <see cref="NDArray"/> (a tuple / array for multi-output).</param>
        /// <param name="signature">The gufunc signature; its input arity must equal the delegate's parameter count.</param>
        /// <param name="otypes">Optional per-output dtype overrides.</param>
        /// <returns>A <see cref="Vectorized"/> in signature mode.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="pyfunc"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="signature"/> is malformed or its input arity mismatches the delegate.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html</remarks>
        public static Vectorized vectorize(Delegate pyfunc, string signature, DType[] otypes = null)
            => new Vectorized(pyfunc, otypes, signature);

        // =================================================================================
        //  np.frompyfunc — the typed-sibling ufunc factory.
        // =================================================================================

        /// <summary>
        ///     Builds a broadcasting callable from an arbitrary delegate — NumSharp's typed analog of NumPy's
        ///     <c>frompyfunc(func, nin, nout)</c>. <b>Divergence:</b> NumPy's <c>frompyfunc</c> returns a ufunc over
        ///     OBJECT arrays; NumSharp has no object dtype, so this returns a <see cref="Vectorized"/> that produces
        ///     TYPED arrays of the delegate's own return type(s). The ufunc-only surface (<c>reduce</c>/<c>accumulate</c>/
        ///     <c>identity</c>) has no NumSharp analog and is not provided.
        /// </summary>
        /// <param name="func">The scalar function to broadcast (any supported arity; may return a tuple for multi-output).</param>
        /// <param name="nin">Declared input count. <b>Validated</b> against the delegate's parameter count (C# knows the arity), unlike NumPy where it DEFINES the count.</param>
        /// <param name="nout">Declared output count. Validated against the delegate's return (1, or its <see cref="ValueTuple"/> arity).</param>
        /// <returns>A <see cref="Vectorized"/> applying <paramref name="func"/> element-wise; use <see cref="Vectorized.Call"/> or <see cref="Vectorized.CallMany"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="func"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="nin"/> or <paramref name="nout"/> disagrees with the delegate's actual arity / return-component count.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html</remarks>
        public static Vectorized frompyfunc(Delegate func, int nin, int nout)
        {
            var v = new Vectorized(func, otypes: null, signature: null);
            if (v.nin != nin)
                throw new ValueError($"frompyfunc: nin ({nin}) does not match the delegate's parameter count ({v.nin})");
            if (v.nout != nout)
                throw new ValueError($"frompyfunc: nout ({nout}) does not match the delegate's return-component count ({v.nout})");
            return v;
        }
    }
}
