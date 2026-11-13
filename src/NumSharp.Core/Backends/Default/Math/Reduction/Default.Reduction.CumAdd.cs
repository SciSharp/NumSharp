using System;
using System.Numerics;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's <c>np.cumsum</c> (<c>np.add.accumulate</c>), the whole call: the running sums along
        ///     <paramref name="axis_"/> (or along the flattened array when it is null), stored into <paramref name="out"/>
        ///     when one is given.
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="axis_">The accumulated axis (negative values count from the end), or null for the flattened array. A
        ///     0-d input is treated as 1-D (NumPy's <c>PyArray_CheckAxis</c>), so only 0 and -1 address it.</param>
        /// <param name="dtype">The accumulator dtype (<c>dtype=</c>), or null: then the loop is
        ///     <c>promote_types(out.dtype, arr.dtype)</c> when <paramref name="out"/> is given (no NEP 50 widening:
        ///     <see cref="AccumulateLoop"/>), NEP 50's accumulator of the input otherwise.</param>
        /// <param name="out">NumPy's <c>out=</c>: an array of the result's shape (any layout and dtype; the result is stored
        ///     with unsafe casting), or null for a fresh result.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise a fresh array of the accumulator dtype: the input's
        ///     shape with an axis, 1-D for the flattened form (a 0-d input gives shape (1,)).</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range — checked before every shortcut (empty,
        ///     0-d and one-element inputs included).</exception>
        /// <exception cref="TypeError">Fixed-width text (input or <paramref name="out"/>), or a string
        ///     <paramref name="dtype"/>, where NumPy's loop resolution refuses: its verbatim text. A <c>StringDType</c> input
        ///     computes (the concatenations), and a numeric loop over text with an <paramref name="out"/> accumulates in the
        ///     loop dtype with the text cast around it.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only (<c>output array is read-only</c>, checked
        ///     first), or text that a numeric loop must parse does not parse.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have the result's shape: the text of the
        ///     statement that refuses it (parity QA-31, <see cref="MisfitOutError"/>): NpyIter's <c>operands could not be
        ///     broadcast together with remapped shapes [original-&gt;remapped]: (4,3)-&gt;(4,3) (3,4)-&gt;(3,4) </c> (or its
        ///     <c>op_axes</c> / non-broadcastable-output siblings) on NumPy's iterator path, an input of more than one
        ///     dimension among its triggers; otherwise — an out the iterator accepts, the direct 1-D path, an empty input or
        ///     out included — <c>provided out is the wrong size for the accumulation.</c> (NumPy 2.5 compares the dimensions on
        ///     every path before the empty exits; 2.4 wrote into a longer or broadcast out, parity QA-54).</exception>
        /// <remarks>
        ///     <para>The statements are np.cumsum's, in its order (moved from <c>APIs/np.cumsum.cs</c> by the np-layer dtype
        ///     plan's P0 pilot): the StringDType route, then add.accumulate's loop resolution on (out, arr) for text (an
        ///     <paramref name="out"/> or a string <paramref name="dtype"/> decides it), then the accumulation
        ///     (<see cref="CumSumCore"/>) and the <paramref name="out"/> store (<see cref="WriteScanToOut"/>).</para>
        ///     <para><see cref="CumSumCore"/> repeats the StringDType check without <paramref name="out"/>, as the engine did
        ///     before the move, so a StringDType input with a numeric <paramref name="out"/> still concatenates and then
        ///     casts the text into it.</para>
        /// </remarks>
        public override NDArray ReduceCumAdd(NDArray arr, int? axis_, DType dtype = null, NDArray @out = null)
        {
            // StringDType: add.accumulate computes (strings plan D18) once NumPy's checks passed (the axis, the dtype= and
            // out= loop resolution); the out receives the accumulation itself, or its cast into an S / U out.
            if (Strings.NDStringUfuncs.VStringReduces("cumsum", arr, axis_, dtype, @out))
                return Strings.NDVStringReduction.CumSum(arr, axis_, @out);
            // Fixed-width text (input or out=) or a string dtype= request: add.accumulate's loop resolution on (out, arr)
            // under the request answers after the axis (NDReduceLikeResolution). A U input into a StringDType out concatenates
            // (the StringDType add promoter pairs StringDType with StrDType; the input is cast to the default StringDType());
            // a numeric loop over a text operand accumulates in the loop dtype with the text cast around it (the out's text
            // must parse first, as NumPy's UPDATEIFCOPY copy reads it); everything else is NumPy's refusal, and without out=
            // or a request the guard in CumSumCore answers (ReduceError).
            switch (Strings.NDStringUfuncs.ReduceIntoOutRoute("cumsum", arr, axis_, @out, dtype, out var resolution))
            {
                case Strings.NDStringUfuncs.TextOutReduce.Concatenate:
                    return Strings.NDVStringReduction.CumSumOfCast(arr, axis_, @out);
                case Strings.NDStringUfuncs.TextOutReduce.NumericCast:
                    return Strings.NDTextNumericReduction.Accumulate("add", arr, axis_, @out, resolution.Descriptors[0]);
            }
            // np.cumsum == np.add.accumulate. With an axis argument it preserves the source memory layout (KEEPORDER);
            // CumSumCore allocates the output in that layout directly. axis=None ravels in C-order to a 1-D result.
            // With out= and no dtype= the loop is promote_types(out, arr), without NEP 50's widening (AccumulateLoop).
            // The loop also decides which NumPy statement refuses an out of the wrong shape (parity QA-31).
            DType loop = AccumulateLoop(arr, dtype, @out);
            return WriteScanToOut(CumSumCore(arr, axis_, loop), @out, arr, axis_, loop);
        }

        /// <summary>
        ///     The accumulation of <see cref="ReduceCumAdd"/> without its <c>out=</c> prologue: the running sums along
        ///     <paramref name="axis_"/>, or along the flattened array when it is null.
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="axis_">The accumulated axis (negative values count from the end), or null for the flattened array. A
        ///     0-d input is treated as 1-D (NumPy's <c>PyArray_CheckAxis</c>), so only 0 and -1 address it.</param>
        /// <param name="dtype">The accumulator dtype (<c>dtype=</c>), or null for NEP 50's accumulator of the input.</param>
        /// <returns>A fresh array of the accumulator dtype: the input's shape with an axis, 1-D for the flattened form (a 0-d
        ///     input gives shape (1,)).</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range — checked before every shortcut (empty,
        ///     0-d and one-element inputs included).</exception>
        /// <exception cref="TypeError">Fixed-width text, or a string <paramref name="dtype"/>: NumPy's refusal (its verbatim
        ///     text). A <c>StringDType</c> input computes (the concatenations).</exception>
        private unsafe NDArray CumSumCore(NDArray arr, int? axis_, DType dtype)
        {
            // Text, or a string dtype= request: NumPy's cumsum checks its arguments first (the axis, and the dtype= class where
            // NumPy reads it), then refuses with its verbatim text (fixed width) or computes on the text (StringDType).
            // StringDType: NumPy's add.accumulate computes (strings plan D18) once its checks passed.
            if (Strings.NDStringUfuncs.VStringReduces("cumsum", arr, axis_, dtype))
                return Strings.NDVStringReduction.CumSum(arr, axis_, null);
            Strings.NDStringUfuncs.ThrowIfReduce("cumsum", arr, axis_, dtype);
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            // A bool LOOP (dtype=bool) is NumPy's BOOL_add, logical or, over the input's truth values
            // (Default.Reduction.BoolLoop.cs): the scans have no bool loop, and the input's own dtype gets a NaN or an
            // imaginary part wrong.
            if (typeCode == NPTypeCode.Boolean)
                return LogicalAccumulate(arr, axis_, logicalAnd: false);
            // NumPy: cumsum on boolean arrays treats True as 1 and False as 0, returning int64
            // Convert boolean input to int64 to match NumPy behavior
            if (arr.GetTypeCode == NPTypeCode.Boolean)
            {
                // In NumPy's ufunc layout of the bool input, not astype's K copy: NumPy casts in the accumulate
                // iterator's buffers, which never vote, so a broadcast bool input still accumulates into C (parity
                // QA-36; the matrix tier's broadcast cases).
                var int64Arr = CastInUfuncLayout(arr, NPTypeCode.Int64);
                var cum = CumSumCore(int64Arr, axis_, (typeCode ?? NPTypeCode.Int64).AsType());
                int64Arr.Dispose();
                return cum;
            }

            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])
            var shape = arr.Shape;
            var retTypeCode = typeCode ?? (arr.GetTypeCode.GetAccumulatingType());

            // Validate the axis UP FRONT, before any trivial shortcut — NumPy raises AxisError first:
            // cumsum([5], axis=-3) and cumsum(0d, axis=1) are errors, not no-ops. A 0-d array is
            // treated as 1-D for cumsum, so the valid range is [-nd, nd) with nd = max(ndim, 1).
            int axis = 0;
            if (axis_ != null)
                axis = Shape.check_and_adjust_axis(Math.Max(arr.ndim, 1), axis_.Value);

            // Empty: cumsum returns a FRESH array of the accumulator dtype — NEP50 widening applies
            // even when empty (cumsum(empty int32) is int64). axis=None ravels to 1-D (NumPy shape
            // (0,)); with an axis the shape is preserved.
            if (shape.IsEmpty || shape.size == 0)
                return NewArray(retTypeCode, axis_ == null ? Shape.Vector(shape.size) : new Shape(shape.dimensions, copy: false), false);

            // 0-d scalar or single-element 1-D: cumsum is the value itself, promoted to the
            // accumulator dtype and shaped 1-D — cumsum NEVER returns 0-d (NumPy: cumsum(0-d) -> (1,),
            // cumsum([x]) -> [x] int64). Previously these returned the input dtype (NEP50 skip bug).
            if (shape.IsScalar || (shape.size == 1 && shape.dimensions.Length == 1))
            {
                var single = Cast(arr, retTypeCode, copy: true);
                if (single.ndim != 1)
                    single.Storage.Reshape(Shape.Vector(1));   // 0-d -> (1,)
                return single;
            }

            if (axis_ == null)
            {
                // axis=None ravels in C-order to a 1-D result; cumsum never collapses to 0-d.
                return cumsum_elementwise(arr, typeCode);
            }

            if (shape[axis] == 1)
                // axis of length 1: values unchanged, but promoted to the accumulator dtype + copied
                // (NumPy: cumsum([[5]], axis=0) -> int64). Was an un-promoted view-copy (NEP50 skip). Laid out as the
                // accumulate's iterator allocates it (LengthOneAxisScan), not as astype's K copy.
                return LengthOneAxisScan(arr, retTypeCode);

            // NumPy-aligned accumulate: np.cumsum IS np.add.accumulate. NumPy allocates the
            // output through the iterator with KEEPORDER, so its memory layout follows the
            // source (C-contig source -> C output, F-contig -> F, a permuted source -> its own
            // axis order: AccumulateOutputLayout, parity QA-36). This is what fixes the
            // long-standing C-only output (the old post-hoc copy('F') in np.cumsum) and removes
            // the per-dtype AxisCumSum* tree in favor of one NDIter-driven generic kernel.
            Shape layout = AccumulateOutputLayout(shape);
            // A scan axis that is not innermost in memory over a dense operand of the accumulator's own dtype runs by whole
            // rows (TryScanRows): the same per-position additions, streamed instead of one strided walk per position.
            if (TryScanRows(arr, axis, retTypeCode, layout, multiply: false, out var rows))
                return rows;
            // A complex128 accumulator over another dtype: the axis loops have no converting complex kernel (the
            // complex add loop reads complex operands only, and the legacy kernels refuse the conversion), so NumPy's
            // buffered cast of the input to the loop dtype is done up front, in the input's own memory order. The
            // result's layout still follows the original operand: `order` was resolved from its shape above.
            NDArray complexInput = retTypeCode == NPTypeCode.Complex && arr.GetTypeCode != NPTypeCode.Complex
                ? arr.astype(NPTypeCode.Complex)
                : null;
            if (complexInput is not null)
                arr = complexInput;
            try
            {
                try
                {
                    return AccumulateAxis(arr, axis, retTypeCode, layout);
                }
                catch (Exception ex) when (ex is not OutOfMemoryException)
                {
                    // Defensive fallback to the legacy whole-array axis path. Only reachable if the
                    // NDIter accumulate construction fails for an exotic view; the legacy path
                    // allocates a C-contiguous output, so relay it into NumPy's layout when that is not C.
                    System.Diagnostics.Debug.WriteLine($"[cumsum] AccumulateAxis fallback: {ex.GetType().Name}: {ex.Message}");
                    var ret = NewArray(retTypeCode, new Shape(shape.dimensions, copy: false), false);
                    if (DirectILKernelGenerator.Enabled && !shape.IsBroadcasted && shape.IsContiguous && shape.offset == 0)
                    {
                        bool innerAxisContiguous = (axis == arr.ndim - 1) && (arr.Shape.strides[axis] == 1);
                        var key = new CumulativeAxisKernelKey(arr.GetTypeCode, retTypeCode, ReductionOp.CumSum, innerAxisContiguous);
                        var kernel = DirectILKernelGenerator.TryGetCumulativeAxisKernel(key);
                        if (kernel != null)
                        {
                            fixed (long* inputStrides = arr.Shape.strides)
                            fixed (long* shapePtr = arr.Shape.dimensions)
                                kernel((void*)arr.Address, (void*)ret.Address, inputStrides, shapePtr, axis, arr.ndim, arr.size);
                            return IntoScanLayout(ret, layout);
                        }
                    }
                    var fb = ExecuteAxisCumSumFallback(arr, ret, axis);
                    return IntoScanLayout(fb, layout);
                }
            }
            finally
            {
                // The accumulation wrote a fresh result, so the cast copy is no longer needed.
                complexInput?.Dispose();
            }
        }

        /// <summary>
        ///     The layout NumPy's <c>PyUFunc_Accumulate</c> gives a fresh accumulation of an array of more than one axis:
        ///     its outer iterator (KEEPORDER over every axis, the scan axis included) allocates the output, so the output
        ///     takes <c>npyiter_find_best_axis_ordering</c>'s order of the input alone — the input's memory order (parity
        ///     QA-36: a permuted input keeps it, a reversed axis is laid out forwards, an F input is F).
        /// </summary>
        /// <param name="input">The accumulated array's shape (any layout; broadcast axes abstain).</param>
        /// <returns>A dense, offset-0 layout of the input's dims: the K layout, or C when that is NumPy's answer.</returns>
        /// <remarks>Shared by np.cumsum and np.cumprod (np.nancumsum / np.nancumprod reach them after the NaN fill). A
        ///     1-D accumulation has one layout, so it is C; ranks past 64 sort only their longer axes
        ///     (<see cref="TryKOrderPlan"/>), so a deep input stays linear.</remarks>
        private static Shape AccumulateOutputLayout(in Shape input)
            => TryKOrderPlan(input.dimensions, new[] { input }, out var layout, out _)
                ? layout
                : new Shape(input.dimensions, copy: false);

        /// <summary>
        ///     A scan along an axis of length 1: the values unchanged in the accumulator dtype, in a fresh array laid out as
        ///     NumPy's accumulate allocates its output (<see cref="AccumulateOutputLayout"/>: the iterator's K order of the
        ///     operand, where an extent-1 axis abstains and lands outermost), not as astype's K copy, which keeps a plain
        ///     F array's strides.
        /// </summary>
        /// <param name="arr">The operand (any layout, at least one element).</param>
        /// <param name="accumulator">The scan's accumulator dtype (NEP 50: narrow signed integers int64, unsigned uint64,
        ///     floats kept; or <c>dtype=</c>'s).</param>
        /// <returns>A fresh array the caller owns.</returns>
        /// <remarks>The cast is unsafe, as astype's: every accumulator is reachable from the input by NumPy's own scan
        ///     loop, which validated it before this shortcut.</remarks>
        private NDArray LengthOneAxisScan(NDArray arr, NPTypeCode accumulator)
        {
            var result = NewArray(accumulator, AccumulateOutputLayout(arr.Shape), false);
            try
            {
                UnmanagedStorage.Copyto(result, arr, NPY_CASTING.NPY_UNSAFE_CASTING);
            }
            catch
            {
                // The copy failed: the fresh array must not outlive the call.
                result.Dispose();
                throw;
            }
            return result;
        }

        /// <summary>
        ///     Moves a C-ordered scan result into <paramref name="layout"/> when that is not C: the defensive fallbacks of
        ///     the scans write C, and a fresh array in NumPy's layout receives their values.
        /// </summary>
        /// <param name="c">The C-ordered result. Consumed: returned itself, or disposed after its values are copied out.</param>
        /// <param name="layout">The target layout (<see cref="AccumulateOutputLayout"/>'s).</param>
        /// <returns><paramref name="c"/> when the layout is C, else a fresh array laid out as <paramref name="layout"/>.</returns>
        /// <remarks>Only the fallbacks pay this copy: the main routes write the layout directly
        ///     (<see cref="AccumulateAxis"/>) or relabel a transposed C pass (<see cref="NDArray.AdoptLayout"/>).</remarks>
        private static NDArray IntoScanLayout(NDArray c, Shape layout)
        {
            if (layout.IsContiguous || c.ndim < 2)
                return c;
            var o = new NDArray(c.typecode, layout, false);
            try
            {
                UnmanagedStorage.Copyto(o, c, NPY_CASTING.NPY_NO_CASTING);
            }
            catch
            {
                // The copy failed: the fresh array must not outlive the call; the caller still sees the error.
                o.Dispose();
                throw;
            }
            finally
            {
                // The C result is consumed either way (its values now live in o, or the call is failing).
                c.Dispose();
            }
            return o;
        }

        /// <summary>
        /// NumPy-aligned axis cumulative sum — the np.add.accumulate structure
        /// (numpy/_core/src/umath/ufunc_object.c : PyUFunc_Accumulate). Builds a 2-operand
        /// [input, output] iterator (KEEPORDER, MULTI_INDEX), removes the scan axis
        /// (<see cref="NDIterRef.RemoveAxis"/>) so the iterator walks every OTHER axis, then
        /// drives a single generic running-sum kernel along the removed axis per outer position.
        /// Handles every layout (contiguous / strided / transposed / broadcast / reversed) through
        /// the iterator and honors the chosen <paramref name="layout"/> exactly (the
        /// kernel reads the output's own scan-axis stride), replacing the legacy per-dtype tree.
        /// </summary>
        /// <param name="input">The array, in the form the kernel reads (a complex accumulator's up-front cast done).</param>
        /// <param name="axis">The scan axis, already normalized.</param>
        /// <param name="retType">The accumulator dtype.</param>
        /// <param name="layout">The output layout (<see cref="AccumulateOutputLayout"/>: NumPy's K order of the input).</param>
        /// <returns>A fresh array laid out as <paramref name="layout"/>.</returns>
        /// <exception cref="Exception">Whatever the iterator's construction raises for a view it cannot walk: the caller
        ///     catches it and takes the legacy fallback.</exception>
        private unsafe NDArray AccumulateAxis(NDArray input, int axis, NPTypeCode retType, Shape layout)
        {
            var shape = input.Shape;
            var ret = NewArray(retType, layout, false);

            // Scan-axis geometry (bytes). Negative for reversed views — the iterator base points
            // at scan-axis index 0 and the kernel walks forward in logical order regardless.
            var aux = new ILKernelGenerator.ScanAxisAux
            {
                InByteStride = input.Shape.strides[axis] * input.dtypesize,
                OutByteStride = ret.Shape.strides[axis] * ret.dtypesize,
                AxisLen = shape[axis],
            };

            var kernel = ILKernelGenerator.GetCumSumInnerLoop(input.GetTypeCode, retType);

            // Construct in C-order (a FORCED order → identity axis permutation, no negative-stride
            // flip): NumSharp's RemoveAxis is index-based (unlike NumPy's perm-mapped one), so the
            // iterator's internal axes MUST match logical axes for RemoveAxis(axis) to drop the
            // scan axis. A KEEPORDER construction would permute axes for F-dominant strides and
            // remove the wrong one. RemoveMultiIndex below re-applies KEEPORDER to the REMAINING
            // axes for cache-friendly traversal; iteration order over kept axes can't affect the
            // result (each scan line is independent), and the scan axis is walked explicitly.
            var opFlags = new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READWRITE };
            using (var iter = NDIterRef.AdvancedNew(
                2,
                new[] { input, ret },
                NDIterGlobalFlags.MULTI_INDEX,
                NPY_ORDER.NPY_CORDER,
                NPY_CASTING.NPY_NO_CASTING,
                opFlags))
            {
                iter.RemoveAxis(axis);        // iterator now walks every axis except the scan axis
                iter.RemoveMultiIndex();      // reorder remaining axes by stride (KEEPORDER) + coalesce
                iter.EnableExternalLoop();    // deliver the innermost remaining axis as a stripe
                iter.ForEach(kernel, &aux);
            }

            return ret;
        }

        /// <summary>
        /// Fallback axis cumsum on the new axis iterator path.
        /// </summary>
        private unsafe NDArray ExecuteAxisCumSumFallback(NDArray inputArr, NDArray ret, int axis)
        {
            var retType = ret.GetTypeCode;

            if (inputArr.GetTypeCode != retType)
                inputArr = Cast(inputArr, retType, copy: true);

            NpFunc.Invoke(retType, CumSumAxisDispatch<int>, inputArr.Storage, ret.Storage, axis);

            return ret;
        }

        private static void CumSumAxisDispatch<T>(UnmanagedStorage input, UnmanagedStorage output, int axis) where T : unmanaged, IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>
            => NDAxisIter.ExecuteSameType<T, CumSumAxisKernel<T>>(input, output, axis);

        private static unsafe void CumSumInPlace<T>(nint addr, long size) where T : unmanaged, IAdditionOperators<T, T, T>
        {
            var p = (T*)addr;
            T sum = default;
            for (long i = 0; i < size; i++)
            {
                sum += p[i];
                p[i] = sum;
            }
        }

        /// <summary>
        ///     The flat cumulative sum of <paramref name="arr"/> (<see cref="cumsum_elementwise"/>), cast to
        ///     <paramref name="typeCode"/> when the accumulation came back in another dtype.
        /// </summary>
        /// <typeparam name="T">The element type the caller dispatched on (kept for the call sites' generic form).</typeparam>
        /// <param name="arr">The array.</param>
        /// <param name="typeCode">The requested result dtype, or null for the accumulation's.</param>
        /// <returns>The 1-D running sums.</returns>
        public NDArray CumSumElementwise<T>(NDArray arr, NPTypeCode? typeCode) where T : unmanaged
        {
            var ret = cumsum_elementwise(arr, typeCode);
            return typeCode.HasValue && typeCode.Value != ret.typecode ? ret.astype(typeCode.Value) : ret;
        }

        protected unsafe NDArray cumsum_elementwise(NDArray arr, NPTypeCode? typeCode)
            => ScanElementwiseFlat(arr, typeCode, ReductionOp.CumSum);

        /// <summary>
        /// Flat (axis=None) cumulative scan shared by cumsum and cumprod.
        /// </summary>
        /// <param name="arr">The input, any layout. A 0-d or one-element 1-D input is returned cast (or cloned).</param>
        /// <param name="typeCode">The accumulator dtype (<c>dtype=</c>), or null for NEP 50's accumulator of the input.</param>
        /// <param name="op"><see cref="ReductionOp.CumSum"/> or <see cref="ReductionOp.CumProd"/>.</param>
        /// <returns>A fresh 1-D array of the accumulator dtype holding the running sums or products in C order.</returns>
        /// <remarks>
        /// The IL scan kernel walks the input in C-order via coordinate decode
        /// (<c>EmitScanStridedLoop</c>), so it consumes strided / transposed / sliced /
        /// reversed views directly — there is no need to materialize a contiguous copy first
        /// (the rejected anti-pattern). The combine emits <c>il Add/Mul</c>, which is invalid
        /// for Half/Complex struct arithmetic; those have a valid IL path only when
        /// same-type-contiguous (a C# helper). So for a non-contiguous Half/Complex, a Half/Complex
        /// accumulator over another input dtype (or when IL is disabled, or decimal-cumprod which the
        /// kernel does not emit) we materialize a single C-order copy — the very ravel NumPy itself
        /// performs — cast it to the accumulator dtype, and run the scalar scan.
        /// </remarks>
        private unsafe NDArray ScanElementwiseFlat(NDArray arr, NPTypeCode? typeCode, ReductionOp op)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();

            var retType = typeCode ?? (arr.GetTypeCode.GetAccumulatingType());

            if (DirectILKernelGenerator.Enabled)
            {
                // Half/Complex accumulators cannot be combined with il Add/Mul: the only valid kernel for them is the
                // same-type contiguous one (a C# helper). A CONVERTING kernel into them emits il add / mul on the
                // struct, which the JIT rejects for float16 (InvalidProgramException) and runs over the raw bits for
                // complex128, reading 16 bytes per input element: wrong values, and past the end of the input buffer
                // (an access violation when it ends at unmapped memory). Those scans take the fallback, which casts the
                // input to the accumulator dtype first (NumPy's buffered cast to the loop dtype) and accumulates there.
                bool combineEmittable = retType != NPTypeCode.Half && retType != NPTypeCode.Complex;
                if (combineEmittable || (arr.Shape.IsContiguous && arr.GetTypeCode == retType))
                {
                    var key = new CumulativeKernelKey(arr.GetTypeCode, retType, op, IsContiguous: arr.Shape.IsContiguous);
                    var kernel = DirectILKernelGenerator.TryGetCumulativeKernel(key);
                    if (kernel != null)
                    {
                        var ret = NewArray(retType, Shape.Vector(arr.size), true);
                        // Strided / reversed / 2-D-sliced views keep their base in Shape.offset
                        // (simple contiguous slices bake it into Address, leaving offset==0); the
                        // kernel reads from a raw base, so fold the offset in here — same base
                        // math as DefaultEngine.ReductionOp.cs.
                        byte* baseAddr = (byte*)arr.Address + arr.Shape.offset * arr.dtypesize;
                        fixed (long* strides = arr.Shape.strides)
                        fixed (long* shape = arr.Shape.dimensions)
                        {
                            kernel((void*)baseAddr, (void*)ret.Address, strides, shape, arr.ndim, arr.size);
                        }
                        return ret;
                    }
                }
            }

            // Fallback: IL disabled, non-contiguous Half/Complex, a Half/Complex accumulator over another
            // dtype, or decimal-cumprod — materialize a C-order copy (NumPy's ravel), then the contiguous
            // scalar scan casts it to the accumulator dtype and accumulates.
            if (!arr.Shape.IsContiguous)
                arr = arr.copy();
            return op == ReductionOp.CumSum
                ? cumsum_elementwise_fallback(arr, retType)
                : cumprod_elementwise_fallback(arr, retType);
        }

        /// <summary>
        /// Fallback element-wise cumsum for contiguous input.
        /// </summary>
        private unsafe NDArray cumsum_elementwise_fallback(NDArray arr, NPTypeCode retType)
        {
            if (!arr.Shape.IsContiguous)
                throw new InvalidOperationException("cumsum_elementwise_fallback requires contiguous input.");

            var linearInput = arr.reshape(Shape.Vector(arr.size));
            var converted = linearInput.typecode == retType
                ? linearInput.Clone()
                : Cast(linearInput, retType, copy: true);

            NpFunc.Invoke(retType, CumSumInPlace<int>, (nint)converted.Address, converted.size);

            return converted;
        }
    }
}
