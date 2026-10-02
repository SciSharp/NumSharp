using System;
using System.Numerics;
using NumSharp.Backends.Iteration;   // NDExprTypeRules (weak-scalar overflow check)
using NumSharp.Backends.Kernels;     // DirectILKernelGenerator (fused scalar fast path)
using NumSharp.Generic;              // NDArray<bool>
using NumSharp.Utilities;            // InfoOf

namespace NumSharp
{
    public static partial class np
    {
        // =====================================================================
        //  np.piecewise — evaluate a piecewise-defined function.
        //
        //  Port of NumPy 2.x numpy.piecewise (numpy/lib/_function_base_impl.py):
        //
        //      x = asanyarray(x)
        //      n2 = len(funclist)
        //      ... promote a scalar / bare-array condlist to a list ...
        //      condlist = asarray(condlist, dtype=bool);  n = len(condlist)
        //      if n == n2 - 1:                       # one extra function = default
        //          condelse = ~np.any(condlist, axis=0, keepdims=True)
        //          condlist = concatenate([condlist, condelse]);  n += 1
        //      elif n != n2:
        //          raise ValueError(...)
        //      y = zeros_like(x)                     # RESULT DTYPE == x's dtype
        //      for cond, func in zip(condlist, funclist):
        //          if not callable(func):
        //              y[cond] = func                # scalar / array assignment
        //          else:
        //              vals = x[cond]
        //              if vals.size > 0:             # the func is NOT called on an empty slice
        //                  y[cond] = func(vals, *args, **kw)
        //      return y
        //
        //  NumSharp keeps NumPy's exact structure — a composition, not a new
        //  kernel — because that is what numpy.piecewise itself is: it has no C
        //  loop of its own, it drives zeros_like + boolean fancy get/set (which in
        //  NumSharp are the SIMD IL-kernel-backed BooleanMask / BooleanMaskSet).
        //  So the perf comes for free from those primitives, and every dtype /
        //  layout they support (all 15 dtypes; C / F / strided / transposed /
        //  reversed / broadcast x) is supported here too.
        //
        //  Three behaviours that differ from the sibling np.select and are easy to
        //  get wrong (all probed against NumPy 2.4.2):
        //    * OUTPUT DTYPE is x's dtype (zeros_like), NOT result_type — so a float
        //      scalar func into an int x truncates (piecewise(int32, [c], [1.9]) -> 1).
        //    * LAST true condition WINS (a forward overwrite loop), the OPPOSITE of
        //      select's first-wins.
        //    * a scalar func is assigned as a weak python scalar: a float truncates
        //      into an integer target, an out-of-range integer raises OverflowError,
        //      a complex into a non-complex target raises TypeError.
        // =====================================================================

        /// <summary>
        ///     Evaluate a piecewise-defined function. Wherever <paramref name="condlist"/>[i] is true,
        ///     the output takes <paramref name="funclist"/>[i] applied to (or, for a scalar entry, equal
        ///     to) <paramref name="x"/>. When several conditions overlap the LAST true one wins
        ///     (a forward overwrite — the opposite of <see cref="select"/>'s first-wins). Positions no
        ///     condition covers keep the value 0, unless one extra function is supplied as the default.
        /// </summary>
        /// <param name="x">
        ///     The input domain. The result has <paramref name="x"/>'s shape AND dtype
        ///     (<see cref="zeros_like"/>), so a float-valued function stored into an integer
        ///     <paramref name="x"/> truncates toward zero, exactly as NumPy's <c>y[cond] = …</c> does.
        /// </param>
        /// <param name="condlist">
        ///     The boolean conditions, one per function (a non-bool array is read by nonzero, matching
        ///     NumPy's <c>asarray(condlist, dtype=bool)</c>). Each condition should have
        ///     <paramref name="x"/>'s shape.
        /// </param>
        /// <param name="funclist">
        ///     One entry per condition, or one extra as the default. Each entry is either a
        ///     <b>callable</b> — a <see cref="Func{NDArray, NDArray}"/> called as <c>f(x[cond])</c>, or a
        ///     <see cref="Func{NDArray, TArgs, NDArray}"/> (<c>Func&lt;NDArray, object[], NDArray&gt;</c>)
        ///     called as <c>f(x[cond], args)</c> — or a <b>constant</b>: a C# scalar (a weak python
        ///     scalar: a float truncates into an integer target, an out-of-range integer raises, a
        ///     complex into a non-complex target raises) or an <see cref="NDArray"/> (broadcast / placed
        ///     into the selected slots, unsafe-cast to <paramref name="x"/>'s dtype). If
        ///     <c>funclist.Length == condlist.Length + 1</c> the extra entry is the default, used where
        ///     every condition is false.
        /// </param>
        /// <param name="args">
        ///     Extra positional arguments forwarded to every <c>Func&lt;NDArray, object[], NDArray&gt;</c>
        ///     entry (NumPy's <c>*args</c>). A plain <c>Func&lt;NDArray, NDArray&gt;</c> entry ignores them.
        /// </param>
        /// <returns>A fresh array of <paramref name="x"/>'s shape and dtype.</returns>
        /// <exception cref="ValueError">
        ///     <c>funclist.Length</c> is neither <c>condlist.Length</c> nor <c>condlist.Length + 1</c>.
        /// </exception>
        /// <exception cref="IndexError"><paramref name="condlist"/> is empty (NumPy accesses condlist[0]).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.piecewise.html</remarks>
        public static NDArray piecewise(NDArray x, NDArray[] condlist, object[] funclist, params object[] args)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (condlist is null) throw new ArgumentNullException(nameof(condlist));
            if (funclist is null) throw new ArgumentNullException(nameof(funclist));

            // NumPy's promotion inspects condlist[0]; an empty list raises there, before the
            // function-count check ever runs.
            if (condlist.Length == 0)
                throw new IndexError("list index out of range");

            return PiecewiseCore(x, condlist, funclist, args);
        }

        /// <summary>
        ///     <see cref="piecewise(NDArray, NDArray[], object[], object[])"/> with a single bare
        ///     condition array instead of a list — NumPy's undocumented promotion. A <b>2-D+</b> array
        ///     is split along axis 0 into one condition per row (a pre-stacked
        ///     <c>[c0, c1]</c>); a <b>1-D</b> array is treated as ONE whole condition when
        ///     <paramref name="x"/> is not 0-d, or split element-by-element when <paramref name="x"/> IS
        ///     0-d (matching NumPy's <c>condlist[0]</c> inspection exactly).
        /// </summary>
        public static NDArray piecewise(NDArray x, NDArray condlist, object[] funclist, params object[] args)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (condlist is null) throw new ArgumentNullException(nameof(condlist));
            if (funclist is null) throw new ArgumentNullException(nameof(funclist));

            NDArray[] promoted = PromoteBareCondlist(x, condlist);
            try
            {
                return PiecewiseCore(x, promoted, funclist, args);
            }
            finally
            {
                // PromoteBareCondlist hands back either the caller's own array (one whole condition) or
                // row views it indexed out of it. The views belong to this call: each holds a reference
                // on the caller's buffer, which would otherwise stay pinned until the finalizer ran.
                // The caller's own instance is never disposed here.
                foreach (var c in promoted)
                    if (!ReferenceEquals(c, condlist))
                        c.Dispose();
            }
        }

        /// <summary>
        ///     <see cref="piecewise(NDArray, NDArray[], object[], object[])"/> with a single scalar
        ///     boolean condition — NumPy's <c>isscalar(condlist)</c> branch. The one function is applied
        ///     to all of <paramref name="x"/> when <paramref name="condlist"/> is true, and nowhere when
        ///     it is false.
        /// </summary>
        public static NDArray piecewise(NDArray x, bool condlist, object[] funclist, params object[] args)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (funclist is null) throw new ArgumentNullException(nameof(funclist));

            // isscalar(condlist) => condlist = [condlist] => a single 0-d bool condition (all / nothing).
            // The 0-d wrapper is this call's own temporary (the result never refers to it), so release it
            // on the way out instead of leaving its buffer to the finalizer.
            using var cond = NDArray.Scalar<bool>(condlist);
            return PiecewiseCore(x, new[] { cond }, funclist, args);
        }

        /// <summary>
        ///     Reproduce NumPy's condlist promotion for a bare (non-list) ndarray:
        ///     <code>
        ///     if isscalar(condlist) or (not isinstance(condlist[0], (list, ndarray)) and x.ndim != 0):
        ///         condlist = [condlist]
        ///     </code>
        ///     For a bare ndarray, <c>condlist[0]</c> is a scalar iff ndim == 1 and an ndarray iff
        ///     ndim &gt;= 2; a 0-d array cannot be indexed at all. So: ndim == 0 raises; ndim == 1 with a
        ///     non-0-d x is ONE condition; otherwise (ndim &gt;= 2, or ndim == 1 with a 0-d x) axis 0 is
        ///     the condition axis and each row is a condition.
        /// </summary>
        private static NDArray[] PromoteBareCondlist(NDArray x, NDArray condlist)
        {
            if (condlist.ndim == 0)
                throw new IndexError("too many indices for array: array is 0-dimensional, but 1 were indexed");

            // A 1-D condition over a non-scalar x is the single whole-array condition (condlist[0] is a
            // scalar element and x.ndim != 0, so NumPy wraps it in a one-element list).
            if (condlist.ndim == 1 && x.ndim != 0)
                return new[] { condlist };

            // Otherwise axis 0 is the condition axis. NumPy's condlist[0] access raises on an empty
            // first axis before any wrap decision is made.
            long count = condlist.shape[0];
            if (count == 0)
                throw new IndexError("index 0 is out of bounds for axis 0 with size 0");

            var conds = new NDArray[count];
            for (int i = 0; i < count; i++)
                conds[i] = condlist[i];   // row i — a 0-d scalar view for a 1-D condlist, else a sub-array
            return conds;
        }

        /// <summary>
        ///     The shared piecewise engine: validate the function count, append the default condition if
        ///     one extra function was given, then <see cref="zeros_like"/> the output and overwrite it
        ///     condition-by-condition in forward order (so the last true condition wins).
        /// </summary>
        /// <param name="x">The input domain; the result takes its shape and dtype.</param>
        /// <param name="condlist">The conditions (non-empty). Read only — never disposed here; the caller owns them.</param>
        /// <param name="funclist">One function per condition, or one more (the default).</param>
        /// <param name="args">Extra arguments for <c>Func&lt;NDArray, object[], NDArray&gt;</c> entries.</param>
        /// <returns>A fresh array of <paramref name="x"/>'s shape and dtype, owned by the caller.</returns>
        /// <exception cref="ValueError">The function count is neither <c>condlist.Length</c> nor one more.</exception>
        /// <exception cref="OverflowException">A scalar function does not fit <paramref name="x"/>'s integer dtype.</exception>
        /// <exception cref="TypeError">A complex scalar function targets a non-complex <paramref name="x"/>.</exception>
        /// <remarks>
        /// Every temporary this creates (the bool condition aliases, the default's "otherwise" mask, the
        /// coerced scalar constants, and the result itself when a later step throws) is disposed before
        /// it returns, so a call leaves no buffer for the finalizer to reclaim.
        /// </remarks>
        private static NDArray PiecewiseCore(NDArray x, NDArray[] condlist, object[] funclist, object[] args)
        {
            int n = condlist.Length;    // >= 1 — every caller guarantees a non-empty condition list
            int n2 = funclist.Length;

            // The function count must equal the condition count, or exceed it by exactly one (the extra
            // function is the default). NumPy's message reports the condition count.
            bool hasDefault = n == n2 - 1;
            if (!hasDefault && n != n2)
                throw new ValueError(
                    $"with {n} condition(s), either {n} or {n + 1} functions are expected");

            // Ownership: every conds[i] and condElse is created HERE (a fresh MakeGeneric alias or a
            // converted copy, never the caller's instance), and the finally releases them all. Before,
            // they were dropped live, keeping a reference on their buffers until the finalizer ran;
            // the FuzzMatrix scope audit (Corpus_AllOps_LeaveNoUndisposedIntermediates) counted 1-3
            // escaped buffers per call. `y` is released only if we throw before handing it out. The
            // result of a user callable is NOT ours and is never disposed (see ApplyCallable).
            var conds = new NDArray<bool>[n];
            NDArray<bool> condElse = null;
            NDArray y = null;
            bool handedOut = false;
            try
            {
                // asarray(condlist, dtype=bool): comparison masks are already bool (a cheap alias); an
                // int/float condition converts by nonzero.
                for (int i = 0; i < n; i++)
                    conds[i] = ToBoolCondition(condlist[i]);

                // Fused single-pass fast path for the SCALAR-funclist contiguous case (the common signum
                // pattern). One IL kernel reads each condition once and writes the result once, avoiding the
                // zeros_like fill, the per-condition BooleanMaskSet (popcount + materialize + scatter), AND —
                // for the default case — the ~any(condlist) computation (the kernel seeds each element with
                // the default scalar and overlays the real conditions forward instead). Declines for
                // callables, NDArray funcs, non-contiguous x/conditions, or a non-SIMD dtype — those take
                // the composition below, which already outruns NumPy at scale.
                if (TryPiecewiseScalarFused(x, conds, funclist, hasDefault, out var fused))
                {
                    handedOut = true;
                    return fused;
                }

                // Composition (NumPy's own structure). One extra function => it is the default, evaluated
                // where NO condition is true: condelse = ~np.any(stack(condlist), axis=0), i.e.
                // ~(c0 | c1 | … | c_{n-1}).
                NDArray<bool>[] effConds;
                if (hasDefault)
                {
                    effConds = new NDArray<bool>[n + 1];
                    Array.Copy(conds, effConds, n);
                    effConds[n] = condElse = ComputeElseCondition(conds, n);
                }
                else
                {
                    effConds = conds;
                }

                // The output shares x's shape AND dtype (order 'K' matches x's layout), then starts at 0.
                y = zeros_like(x);

                for (int i = 0; i < effConds.Length; i++)
                {
                    NDArray<bool> cond = effConds[i];
                    object func = funclist[i];

                    switch (func)
                    {
                        // A callable f(x[cond]) — evaluated only over the selected values, exactly as NumPy
                        // does, and (matching its `vals.size > 0` guard) NOT called on an empty selection.
                        case Func<NDArray, NDArray> f1:
                            ApplyCallable(x, y, cond, f1, null, args);
                            break;
                        case Func<NDArray, object[], NDArray> f2:
                            ApplyCallable(x, y, cond, null, f2, args);
                            break;

                        // Anything else is a constant "function" (a scalar or an array), assigned into the
                        // selected slots — the `not callable(func)` branch of NumPy's loop.
                        default:
                            AssignConstant(y, cond, func);
                            break;
                    }
                }

                handedOut = true;
                return y;
            }
            finally
            {
                foreach (var c in conds)
                    c?.Dispose();
                condElse?.Dispose();
                // A validation error (e.g. an out-of-range weak scalar) or a throwing callable abandons
                // the half-written result; the caller never sees it, so it is ours to release.
                if (!handedOut)
                    y?.Dispose();
            }
        }

        /// <summary>
        ///     Evaluate a callable over the selected values and scatter the result back:
        ///     <c>vals = x[cond]; if vals.size &gt; 0: y[cond] = func(vals[, args])</c>. Reads
        ///     <paramref name="x"/> through its own strides in logical C-order (any layout) and writes
        ///     <paramref name="y"/> in the same C-order, so the k-th produced value lands at the k-th
        ///     selected coordinate. The result broadcasts / places into the selection and unsafe-casts to
        ///     <paramref name="y"/>'s dtype (a wrong-length result raises the NumPy boolean-assignment
        ///     ValueError). Exactly one of <paramref name="f1"/>/<paramref name="f2"/> is non-null.
        /// </summary>
        private static void ApplyCallable(NDArray x, NDArray y, NDArray<bool> cond,
            Func<NDArray, NDArray> f1, Func<NDArray, object[], NDArray> f2, object[] args)
        {
            NDArray vals = x[cond];
            if (vals.size == 0)
                return;   // NumPy skips calling func on an empty slice.

            NDArray result = f1 != null ? f1(vals) : f2(vals, args);
            y[cond] = result;
        }

        /// <summary>
        ///     Assign a constant "function" (the <c>not callable(func)</c> branch of NumPy's loop) into
        ///     the selected slots. An <see cref="NDArray"/> is a strong value (broadcast / placed,
        ///     unsafe-cast). A C# scalar is a weak python scalar: a float truncates into an integer
        ///     target, an out-of-range integer raises <see cref="OverflowException"/>, and a complex into
        ///     a non-complex target raises <see cref="TypeError"/> — matching NumPy's <c>y[cond] = scalar</c>.
        /// </summary>
        private static void AssignConstant(NDArray y, NDArray<bool> cond, object func)
        {
            switch (func)
            {
                case null:
                    // A null "function" is a caller bug (NumPy would build an object array — a dtype
                    // NumSharp lacks). The same treatment np.select gives a null choice.
                    throw new ArgumentNullException(nameof(func), "funclist entries must not be null.");

                case NDArray arr:
                    // Strong: broadcast (size 1) or place (exact length) into the selection, unsafe-cast
                    // to y's dtype — exactly NumPy's `y[cond] = ndarray`.
                    y[cond] = arr;
                    return;

                default:
                    // A weak python scalar. Validate it the way NumPy's scalar boolean assignment does,
                    // then let the boolean-mask setter cast it into y's dtype (float -> int truncates,
                    // an in-range int is exact). The 0-d wrapper is released once its value is scattered.
                    // Disposing is safe ONLY because `func` is never an NDArray here (that case returned
                    // above): asanyarray hands back an EXISTING array for NDArray input, but for every
                    // other input CoerceScalarConstant mints a fresh one.
                    using (var scalar = CoerceScalarConstant(func, y.typecode))
                        y[cond] = scalar;
                    return;
            }
        }

        /// <summary>
        ///     Validate a weak (python-scalar) constant against <paramref name="target"/> the way NumPy's
        ///     <c>y[cond] = scalar</c> does and wrap it as a 0-d <see cref="NDArray"/>. Into a non-bool
        ///     integer target: a float is truncated toward zero and the truncated value must fit
        ///     (<c>127.9 -> 127</c> stores, <c>128.0</c> raises, NaN/inf raise), an integer must fit its
        ///     inclusive range (else <see cref="OverflowException"/> "Python integer N out of bounds for
        ///     …", reusing the NEP50 machinery). A complex into a non-complex target raises
        ///     <see cref="TypeError"/> (NumPy: <c>float() argument … not 'complex'</c>). A bool / float /
        ///     complex / decimal target needs no range check (the setter's cast handles it).
        /// </summary>
        private static NDArray CoerceScalarConstant(object value, NPTypeCode target)
        {
            bool integerTarget = target is NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16
                or NPTypeCode.UInt16 or NPTypeCode.Char or NPTypeCode.Int32 or NPTypeCode.UInt32
                or NPTypeCode.Int64 or NPTypeCode.UInt64;

            switch (value)
            {
                case bool:
                    return NDArray.Scalar(value);   // 0/1 always fits an integer/bool target

                case sbyte or byte or short or ushort or int or uint or long or char:
                    if (integerTarget)
                        NDExprTypeRules.CheckIntLiteralFits(Convert.ToInt64(value), target);
                    return NDArray.Scalar(value);

                case ulong u:
                    if (integerTarget)
                    {
                        if (target == NPTypeCode.UInt64) { /* any ulong fits uint64 */ }
                        else if (u <= long.MaxValue) NDExprTypeRules.CheckIntLiteralFits((long)u, target);
                        else throw new OverflowException(
                            $"Python integer {u} out of bounds for {target.AsNumpyDtypeName()}");
                    }
                    return NDArray.Scalar(value);

                case Half or float or double:
                    if (integerTarget)
                        CheckFloatFitsInteger(value is Half h ? (double)h : Convert.ToDouble(value), target);
                    return NDArray.Scalar(value);

                case Complex when target != NPTypeCode.Complex:
                    // NumPy raises this exact TypeError coercing a python complex into a real array.
                    throw new TypeError("float() argument must be a string or a real number, not 'complex'");

                default:
                    // Complex-into-complex, decimal, and any other array_like: let asanyarray + the
                    // setter's cast handle it (no weak-scalar range check applies).
                    return asanyarray(value);
            }
        }

        /// <summary>
        ///     NumPy's float-scalar-into-integer rule: truncate toward zero and require the truncated
        ///     value to fit <paramref name="target"/> (NaN/inf never do). Reuses the same
        ///     <see cref="NDExprTypeRules.CheckIntLiteralFits"/> that names the truncated integer in the
        ///     message; only the out-of-<c>long</c> tails are handled locally.
        /// </summary>
        private static void CheckFloatFitsInteger(double d, NPTypeCode target)
        {
            if (double.IsNaN(d) || double.IsInfinity(d))
                throw new OverflowException(
                    $"cannot assign non-finite float {d} to integer dtype {target.AsNumpyDtypeName()}");

            double t = Math.Truncate(d);
            if (t >= -9223372036854775808.0 && t < 9223372036854775808.0)
            {
                NDExprTypeRules.CheckIntLiteralFits((long)t, target);
                return;
            }
            if (target == NPTypeCode.UInt64 && t >= 0.0 && t <= 18446744073709551615.0)
                return;   // fits uint64 but not long
            throw new OverflowException(
                $"Python integer {t} out of bounds for {target.AsNumpyDtypeName()}");
        }

        /// <summary>
        ///     The fused single-pass fast path for the SCALAR-funclist case. Fires only when every
        ///     function is a C# scalar constant, <paramref name="x"/> and every effective condition are
        ///     C-contiguous offset-0 at <paramref name="x"/>'s shape, and <paramref name="x"/>'s dtype is
        ///     SIMD-eligible (the 1/2/4/8-byte numerics). Coerces the scalars into <paramref name="x"/>'s
        ///     dtype (the same weak-scalar validation the composition uses — a float truncates, an
        ///     out-of-range integer raises, a complex-into-real raises) and delegates to one IL kernel
        ///     (<see cref="DirectILKernelGenerator.GetPiecewiseScalarKernel"/>) that seeds each element
        ///     with 0 and overlays each condition's scalar forward, so the last true condition wins.
        ///     Returns false (leaving <paramref name="result"/> null) for any callable / NDArray func,
        ///     non-contiguous or broadcast operand, or non-SIMD dtype, so the caller uses the composition.
        /// </summary>
        private static unsafe bool TryPiecewiseScalarFused(NDArray x, NDArray<bool>[] conds, object[] funclist, bool hasDefault, out NDArray result)
        {
            result = null;

            if (!DirectILKernelGenerator.Enabled)
                return false;

            NPTypeCode dtype = x.typecode;
            if (!DirectILKernelGenerator.PiecewiseScalarKernelSupportsDtype(dtype))
                return false;

            long size = x.size;
            if (size == 0)
                return false;   // empty — the composition handles it trivially.

            // The kernel writes a fresh C-contiguous result and reads each condition as a full element
            // mask, so x and every condition must be C-contiguous, offset 0, at exactly x's shape.
            if (!(x.Shape.IsContiguous && x.Shape.offset == 0))
                return false;

            int n = conds.Length;   // number of real conditions (the default, if any, is the seed)
            for (int k = 0; k < n; k++)
            {
                var c = conds[k];
                if (!(c.Shape.IsContiguous && c.Shape.offset == 0)) return false;
                if (!PiecewiseShapeEquals(c.Shape, x.Shape)) return false;
            }

            // Every function must be a plain C# scalar constant. A callable, an NDArray func, or a null
            // declines to the composition (which handles callables and array funcs).
            for (int i = 0; i < funclist.Length; i++)
            {
                object f = funclist[i];
                if (f is null || f is NDArray
                    || f is Func<NDArray, NDArray> || f is Func<NDArray, object[], NDArray>)
                    return false;
            }

            var kernel = DirectILKernelGenerator.GetPiecewiseScalarKernel(dtype, n);
            if (kernel == null)
                return false;

            int elemSize = InfoOf.GetSize(dtype);

            // Coerce the n condition scalars PLUS the seed into a small STACK buffer of x's dtype (n+1
            // slots; funcVals[n] is the seed). The seed is the default function where one was given, else
            // 0 (piecewise's zeros_like default). This is the SAME weak-scalar validation the
            // composition's AssignConstant applies — an out-of-range integer / complex-into-real raises
            // here, before any kernel work — and it folds the default in WITHOUT the ~any(condlist) pass.
            byte* fvals = stackalloc byte[checked((n + 1) * elemSize)];
            for (int k = 0; k < n; k++)
                WriteCoercedScalar(fvals + k * elemSize, funclist[k], dtype, elemSize);
            object seed = hasDefault ? funclist[n] : (object)0;   // funclist[n] is the default
            WriteCoercedScalar(fvals + n * elemSize, seed, dtype, elemSize);

            var res = new NDArray(dtype, new Shape((long[])x.Shape.dimensions.Clone()), false);

            bool** condPtrs = stackalloc bool*[n];
            for (int k = 0; k < n; k++)
                condPtrs[k] = (bool*)conds[k].Storage.Address + conds[k].Shape.offset;
            void* resPtr = (byte*)res.Storage.Address + res.Shape.offset * elemSize;

            kernel(condPtrs, fvals, resPtr, size);

            GC.KeepAlive(conds);
            result = res;
            return true;
        }

        /// <summary>
        ///     Validate a weak scalar (<see cref="CoerceScalarConstant"/>) and copy its cast-to-dtype
        ///     bytes into <paramref name="dst"/>. Shared by the fused kernel's condition-scalar and seed
        ///     writes.
        /// </summary>
        /// <param name="dst">Destination for exactly one <paramref name="dtype"/> element.</param>
        /// <param name="value">A plain C# scalar constant (the fused path admits no NDArray or callable).</param>
        /// <param name="dtype">The target dtype (x's dtype).</param>
        /// <param name="elemSize">The byte width of <paramref name="dtype"/>.</param>
        /// <exception cref="OverflowException">An integer target cannot hold the (truncated) value.</exception>
        /// <exception cref="TypeError">A complex value targets a non-complex dtype.</exception>
        private static unsafe void WriteCoercedScalar(byte* dst, object value, NPTypeCode dtype, int elemSize)
        {
            // Both 0-d temporaries are this call's own: CoerceScalarConstant mints a fresh wrapper (value
            // is never an NDArray here) and astype(copy: true) always allocates. Once the bytes are copied
            // out, nothing refers to them, so release both instead of leaving them to the finalizer.
            using var scalar = CoerceScalarConstant(value, dtype);   // validated 0-d
            using var coerced = scalar.astype(dtype);                 // cast to x's dtype
            Buffer.MemoryCopy((byte*)coerced.Storage.Address + coerced.Shape.offset * elemSize,
                dst, elemSize, elemSize);
        }

        /// <summary>True when two shapes have identical dimensions.</summary>
        private static bool PiecewiseShapeEquals(Shape a, Shape b)
        {
            var da = a.dimensions;
            var db = b.dimensions;
            if (da.Length != db.Length) return false;
            for (int i = 0; i < da.Length; i++)
                if (da[i] != db[i]) return false;
            return true;
        }

        /// <summary>Convert a condition to <see cref="NDArray{Boolean}"/> (nonzero for a non-bool array).</summary>
        /// <param name="cond">A caller-owned condition of any dtype; never disposed or mutated here.</param>
        /// <returns>
        /// A NEW object the caller owns and must dispose: a <see cref="NDArray.MakeGeneric{T}"/> alias
        /// (its own reference on <paramref name="cond"/>'s buffer) for a bool condition, or the sole
        /// owner of a fresh nonzero-converted buffer otherwise. Never <paramref name="cond"/> itself,
        /// so disposing the result can never free the caller's array.
        /// </returns>
        private static NDArray<bool> ToBoolCondition(NDArray cond)
        {
            if (cond.typecode == NPTypeCode.Boolean)
                return cond.MakeGeneric<bool>();

            // The conversion's own wrapper is released as soon as the alias exists: the alias holds its
            // own reference, so it becomes the buffer's only owner (previously the wrapper was dropped
            // live and pinned the buffer until the finalizer ran).
            using var converted = cond.astype(NPTypeCode.Boolean);
            return converted.MakeGeneric<bool>();
        }

        /// <summary>
        ///     The "otherwise" condition — true wherever none of the <paramref name="n"/> conditions is
        ///     (NumPy's <c>~np.any(condlist, axis=0)</c>, i.e. <c>logical_not(c0 | c1 | … | c_{n-1})</c>).
        ///     Uses <see cref="logical_not"/> (the ufunc) rather than the <c>!</c> operator: the ufunc is
        ///     layout-correct for every operand, whereas the operator mis-maps a non-C-contiguous input.
        /// </summary>
        /// <param name="conds">The bool conditions; read only (the caller owns and disposes them).</param>
        /// <param name="n">How many leading entries of <paramref name="conds"/> to fold (≥ 1).</param>
        /// <returns>A fresh bool mask the caller owns and must dispose.</returns>
        private static NDArray<bool> ComputeElseCondition(NDArray<bool>[] conds, int n)
        {
            // OR-fold. Each partial OR is a fresh array this method made, so it is released once the next
            // one exists. conds[0] seeds the fold but belongs to the caller, so it is never disposed here.
            NDArray anyTrue = conds[0];
            for (int i = 1; i < n; i++)
            {
                NDArray next = anyTrue | conds[i];
                if (!ReferenceEquals(anyTrue, conds[0]))
                    anyTrue.Dispose();
                anyTrue = next;
            }

            // The alias outlives the `using`: it holds its own reference, so disposing notTrue (and the
            // last partial OR) leaves the returned mask the sole owner of its buffer.
            using var notTrue = logical_not(anyTrue);
            if (!ReferenceEquals(anyTrue, conds[0]))
                anyTrue.Dispose();
            return notTrue.MakeGeneric<bool>();
        }
    }
}
