using System;
using System.Numerics;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override unsafe NDArray ReduceCumAdd(NDArray arr, int? axis_, NPTypeCode? typeCode = null)
        {
            // NumPy: cumsum on boolean arrays treats True as 1 and False as 0, returning int64
            // Convert boolean input to int64 to match NumPy behavior
            if (arr.GetTypeCode == NPTypeCode.Boolean)
            {
                var int64Arr = arr.astype(NPTypeCode.Int64, copy: true);
                return ReduceCumAdd(int64Arr, axis_, typeCode ?? NPTypeCode.Int64);
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
            {
                int nd = Math.Max(arr.ndim, 1);
                axis = axis_.Value;
                if (axis < 0) axis += nd;
                if (axis < 0 || axis >= nd)
                    throw new ArgumentOutOfRangeException(nameof(axis_),
                        $"axis {axis_.Value} is out of bounds for array of dimension {nd}");
            }

            // Empty: cumsum returns a FRESH array of the accumulator dtype — NEP50 widening applies
            // even when empty (cumsum(empty int32) is int64). axis=None ravels to 1-D (NumPy shape
            // (0,)); with an axis the shape is preserved.
            if (shape.IsEmpty || shape.size == 0)
                return new NDArray(retTypeCode,
                    axis_ == null ? Shape.Vector((int)shape.size) : new Shape(shape.dimensions), false);

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
                // (NumPy: cumsum([[5]], axis=0) -> int64). Was an un-promoted view-copy (NEP50 skip).
                return Cast(arr, retTypeCode, copy: true);

            // NumPy-aligned accumulate: np.cumsum IS np.add.accumulate. NumPy allocates the
            // output through the iterator with KEEPORDER, so its memory layout follows the
            // source (C-contig source -> C output, F-contig -> F). NumSharp models two physical
            // layouts, so 'K' resolves to C or F via OrderResolver. This is what fixes the
            // long-standing C-only output (the old post-hoc copy('F') in np.cumsum) and removes
            // the per-dtype AxisCumSum* tree in favor of one NDIter-driven generic kernel.
            char order = OrderResolver.Resolve('K', shape);
            try
            {
                return AccumulateAxis(arr, axis, retTypeCode, order);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                // Defensive fallback to the legacy whole-array axis path. Only reachable if the
                // NDIter accumulate construction fails for an exotic view; the legacy path
                // allocates a C-contiguous output, so relay it to F when the source asked for it.
                System.Diagnostics.Debug.WriteLine($"[cumsum] AccumulateAxis fallback: {ex.GetType().Name}: {ex.Message}");
                var ret = new NDArray(retTypeCode, new Shape(shape.dimensions), false);
                if (DirectILKernelGenerator.Enabled && !shape.IsBroadcasted && shape.IsContiguous && shape.offset == 0)
                {
                    bool innerAxisContiguous = (axis == arr.ndim - 1) && (arr.strides[axis] == 1);
                    var key = new CumulativeAxisKernelKey(arr.GetTypeCode, retTypeCode, ReductionOp.CumSum, innerAxisContiguous);
                    var kernel = DirectILKernelGenerator.TryGetCumulativeAxisKernel(key);
                    if (kernel != null)
                    {
                        fixed (long* inputStrides = arr.strides)
                        fixed (long* shapePtr = arr.shape)
                            kernel((void*)arr.Address, (void*)ret.Address, inputStrides, shapePtr, axis, arr.ndim, arr.size);
                        return order == 'F' && ret.Shape.NDim > 1 ? ret.copy('F') : ret;
                    }
                }
                var fb = ExecuteAxisCumSumFallback(arr, ret, axis);
                return order == 'F' && fb.Shape.NDim > 1 ? fb.copy('F') : fb;
            }
        }

        /// <summary>
        /// NumPy-aligned axis cumulative sum — the np.add.accumulate structure
        /// (numpy/_core/src/umath/ufunc_object.c : PyUFunc_Accumulate). Builds a 2-operand
        /// [input, output] iterator (KEEPORDER, MULTI_INDEX), removes the scan axis
        /// (<see cref="NDIterRef.RemoveAxis"/>) so the iterator walks every OTHER axis, then
        /// drives a single generic running-sum kernel along the removed axis per outer position.
        /// Handles every layout (contiguous / strided / transposed / broadcast / reversed) through
        /// the iterator and honors the chosen <paramref name="order"/> output layout exactly (the
        /// kernel reads the output's own scan-axis stride), replacing the legacy per-dtype tree.
        /// </summary>
        private unsafe NDArray AccumulateAxis(NDArray input, int axis, NPTypeCode retType, char order)
        {
            var shape = input.Shape;
            int ndim = input.ndim;

            long[] dims = new long[ndim];
            for (int i = 0; i < ndim; i++) dims[i] = shape.dimensions[i];
            var ret = new NDArray(retType, new Shape(dims, order), false);

            // Scan-axis geometry (bytes). Negative for reversed views — the iterator base points
            // at scan-axis index 0 and the kernel walks forward in logical order regardless.
            var aux = new ILKernelGenerator.ScanAxisAux
            {
                InByteStride = input.strides[axis] * input.dtypesize,
                OutByteStride = ret.strides[axis] * ret.dtypesize,
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

        public NDArray CumSumElementwise<T>(NDArray arr, NPTypeCode? typeCode) where T : unmanaged
        {
            var ret = cumsum_elementwise(arr, typeCode);
            return typeCode.HasValue && typeCode.Value != ret.typecode ? ret.astype(typeCode.Value, true) : ret;
        }

        protected unsafe NDArray cumsum_elementwise(NDArray arr, NPTypeCode? typeCode)
            => ScanElementwiseFlat(arr, typeCode, ReductionOp.CumSum);

        /// <summary>
        /// Flat (axis=None) cumulative scan shared by cumsum and cumprod.
        /// </summary>
        /// <remarks>
        /// The IL scan kernel walks the input in C-order via coordinate decode
        /// (<c>EmitScanStridedLoop</c>), so it consumes strided / transposed / sliced /
        /// reversed views directly — there is no need to materialize a contiguous copy first
        /// (the rejected anti-pattern). The combine emits <c>il Add/Mul</c>, which is invalid
        /// for Half/Complex struct arithmetic; those have a valid IL path only when
        /// same-type-contiguous (a C# helper). So for a non-contiguous Half/Complex (or when IL
        /// is disabled, or decimal-cumprod which the kernel does not emit) we materialize a
        /// single C-order copy — the very ravel NumPy itself performs — and run the scalar scan.
        /// </remarks>
        private unsafe NDArray ScanElementwiseFlat(NDArray arr, NPTypeCode? typeCode, ReductionOp op)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();

            var retType = typeCode ?? (arr.GetTypeCode.GetAccumulatingType());

            if (DirectILKernelGenerator.Enabled)
            {
                // Half/Complex accumulators cannot be combined with il Add/Mul except via the
                // same-type-contiguous C# helper, so only let them reach the kernel contiguous.
                bool combineEmittable = retType != NPTypeCode.Half && retType != NPTypeCode.Complex;
                if (arr.Shape.IsContiguous || combineEmittable)
                {
                    var key = new CumulativeKernelKey(arr.GetTypeCode, retType, op, IsContiguous: arr.Shape.IsContiguous);
                    var kernel = DirectILKernelGenerator.TryGetCumulativeKernel(key);
                    if (kernel != null)
                    {
                        var ret = new NDArray(retType, Shape.Vector(arr.size));
                        // Strided / reversed / 2-D-sliced views keep their base in Shape.offset
                        // (simple contiguous slices bake it into Address, leaving offset==0); the
                        // kernel reads from a raw base, so fold the offset in here — same base
                        // math as DefaultEngine.ReductionOp.cs.
                        byte* baseAddr = (byte*)arr.Address + arr.Shape.offset * arr.dtypesize;
                        fixed (long* strides = arr.strides)
                        fixed (long* shape = arr.shape)
                        {
                            kernel((void*)baseAddr, (void*)ret.Address, strides, shape, arr.ndim, arr.size);
                        }
                        return ret;
                    }
                }
            }

            // Fallback: IL disabled, or non-contiguous Half/Complex/decimal-cumprod — materialize
            // a C-order copy (NumPy's ravel) then run the contiguous scalar scan.
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
