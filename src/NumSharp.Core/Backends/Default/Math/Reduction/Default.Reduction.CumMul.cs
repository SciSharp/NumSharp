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
        ///     NumPy's <c>np.cumprod</c> (<c>np.multiply.accumulate</c>), the whole call: the running products along
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
        /// <exception cref="TypeError">Text, or a string <paramref name="dtype"/>: NumPy's refusal (its verbatim text; no
        ///     multiply loop accumulates text). A numeric loop over text with an <paramref name="out"/> accumulates in the
        ///     loop dtype with the text cast around it.</exception>
        /// <exception cref="ValueError">A numeric <paramref name="dtype"/> over text that does not parse as that number, or a
        ///     read-only <paramref name="out"/> (<c>output array is read-only</c>).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have the result's shape (NumPy's
        ///     <c>provided out is the wrong size for the accumulation.</c>).</exception>
        /// <remarks>
        ///     The statements are np.cumprod's, in its order (moved from <c>APIs/np.cumprod.cs</c> by the np-layer dtype
        ///     plan's P0 pilot): multiply.accumulate's loop resolution on (out, arr) answers FIRST for text or a string
        ///     request, then a numeric <paramref name="dtype"/> over text parses the text, then NumPy's refusal for what is
        ///     left, then the accumulation (<see cref="CumProdCore"/>) and the <paramref name="out"/> store
        ///     (<see cref="WriteScanToOut"/>).
        /// </remarks>
        public override NDArray ReduceCumMul(NDArray arr, int? axis_, DType dtype = null, NDArray @out = null)
        {
            // Text (input or out=) or a string dtype= request: multiply.accumulate's loop resolution on (out, arr) under the
            // request answers FIRST (NDReduceLikeResolution) — an out decides whether the text is parsed at all (an integer
            // out with text makes NumPy's legacy resolver pick the string repeat loop, which an accumulation refuses). A
            // numeric loop over a text operand with out= accumulates in the loop dtype with the text cast around it.
            if (Strings.NDStringUfuncs.ReduceIntoOutRoute("cumprod", arr, axis_, @out, dtype, out var resolution)
                == Strings.NDStringUfuncs.TextOutReduce.NumericCast)
                return Strings.NDTextNumericReduction.Accumulate("multiply", arr, axis_, @out, resolution.Descriptors[0]);
            // Without out= the text input is parsed into the requested number type; the parse is a temporary of this call,
            // released when it returns (the accumulation always allocates its own result, so nothing aliases it). Without
            // out= or a request the guard below answers (ReduceError).
            NDArray parsed = Strings.NDStringUfuncs.ParseForNumericReduce("cumprod", arr, axis_, dtype);
            try
            {
                if (parsed is not null)
                    arr = parsed;
                Strings.NDStringUfuncs.ThrowIfReduce("cumprod", arr, axis_, @out, dtype);
                // With out= and no dtype= the loop is promote_types(out, arr), without NEP 50's widening (AccumulateLoop).
                return WriteScanToOut(CumProdCore(arr, axis_, AccumulateLoop(arr, dtype, @out)), @out);
            }
            finally
            {
                parsed?.Dispose();
            }
        }

        /// <summary>
        ///     The accumulation of <see cref="ReduceCumMul"/> without its <c>out=</c> prologue: the running products along
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
        /// <exception cref="TypeError">Text, or a string <paramref name="dtype"/>: NumPy's refusal (its verbatim text; no
        ///     multiply loop accumulates text).</exception>
        /// <exception cref="ValueError">A numeric <paramref name="dtype"/> over text that does not parse as that number.</exception>
        private unsafe NDArray CumProdCore(NDArray arr, int? axis_, DType dtype)
        {
            // Text, or a string dtype= request: NumPy's cumprod checks its arguments first (the axis, and the dtype= class where
            // NumPy reads it), then refuses with its verbatim text.
            // A numeric dtype= over text: NumPy casts the text to that number type first (a parse), then reduces the numbers.
            if (Strings.NDStringUfuncs.ParseForNumericReduce("cumprod", arr, axis_, dtype) is { } parsed)
                return Strings.NDStringUfuncs.ReduceParsed(parsed, p => CumProdCore(p, axis_, dtype));
            Strings.NDStringUfuncs.ThrowIfReduce("cumprod", arr, axis_, dtype);
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            // A bool LOOP (dtype=bool) is NumPy's BOOL_multiply, logical and, over the input's truth values
            // (Default.Reduction.BoolLoop.cs): the scans have no bool loop, and the input's own dtype gets a NaN or an
            // imaginary part wrong.
            if (typeCode == NPTypeCode.Boolean)
                return LogicalAccumulate(arr, axis_, logicalAnd: true);
            // NumPy: cumprod on boolean arrays treats True as 1 and False as 0, returning int64
            // Convert boolean input to int64 to match NumPy behavior
            if (arr.GetTypeCode == NPTypeCode.Boolean)
            {
                var int64Arr = arr.astype(NPTypeCode.Int64, copy: true);
                var cum = CumProdCore(int64Arr, axis_, (typeCode ?? NPTypeCode.Int64).AsType());
                int64Arr.Dispose();
                return cum;
            }

            var shape = arr.Shape;

            // PyArray_CheckAxis, before any shortcut: the axis addresses rank max(ndim, 1) (a 0-d input ravels to 1-D), so
            // cumprod([5], axis=-3) and cumprod(0-d, axis=1) are NumPy's AxisError, not the value itself. The shortcuts below
            // used to run first, and the later wrap-then-test loop accepted -ndim - 1 (reducing the last axis).
            int axisChecked = axis_ is null ? 0 : Shape.check_and_adjust_axis(Math.Max(arr.ndim, 1), axis_.Value);

            // Empty: cumprod returns a FRESH array of the accumulator dtype (NEP50 widening applies
            // even when empty — cumprod(empty int32) is int64, like cumsum). axis=None ravels to a
            // 1-D (0,) result; with an axis the shape is preserved. Returning `arr` unchanged (as
            // before) kept the source dtype AND, for a multi-dim source, the un-raveled shape —
            // e.g. cumprod((0,3)) gave (0,3) where NumPy gives (0,). Mirrors ReduceCumAdd.
            if (shape.IsEmpty || shape.size == 0)
            {
                var emptyRetType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();

                return new NDArray(emptyRetType,
                    axis_ == null ? Shape.Vector(shape.size) : Shape.Adopt(shape.dimensions), false);
            }

            // 0-d scalar or single-element 1-D: cumprod is the value itself, promoted to the
            // accumulator dtype and shaped 1-D — cumprod NEVER returns 0-d (NumPy: cumprod(0-d) -> (1,),
            // cumprod([x]) -> [x] int64). Mirrors ReduceCumAdd; previously this returned the input dtype
            // AND a 0-d shape for a 0-d input (NEP50-skip + missing (1,) reshape).
            if (shape.IsScalar || (shape.size == 1 && shape.dimensions.Length == 1))
            {
                var single = Cast(arr, typeCode ?? arr.GetTypeCode.GetAccumulatingType(), copy: true);
                if (single.ndim != 1)
                    single.Storage.Reshape(Shape.Vector(1));   // 0-d -> (1,)
                return single;
            }

            if (axis_ == null)
            {
                var r = cumprod_elementwise(arr, typeCode);
                if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1)
                    r.Storage.Reshape(Shape.Scalar);
                return r;
            }

            var axis = axisChecked;

            if (shape[axis] == 1)
            {
                //if the given div axis is 1 - cumprod is just the value itself
                //Return a copy to avoid sharing memory with the original (NumPy behavior)
                return arr.copy();
            }

            // For broadcast arrays, we iterate over the input (which has stride=0 for broadcast dims)
            // but write to a contiguous output array.
            NDArray inputArr = arr;

            // Create output with CONTIGUOUS strides even if input is broadcast.
            var outputShape = Shape.Adopt(shape.dimensions);  // Fresh contiguous shape
            var retTypeCode = typeCode ?? (inputArr.GetTypeCode.GetAccumulatingType());
            var ret = new NDArray(retTypeCode, outputShape, false);

            // Fast path: use IL-generated axis kernel when available
            // Note: We only use the IL kernel for contiguous arrays without offset, as it doesn't
            // handle negative strides or offset-based views correctly.
            if (DirectILKernelGenerator.Enabled && !shape.IsBroadcasted && shape.IsContiguous && shape.offset == 0)
            {
                bool innerAxisContiguous = (axis == arr.ndim - 1) && (arr.Shape.strides[axis] == 1);
                var key = new CumulativeAxisKernelKey(inputArr.GetTypeCode, retTypeCode, ReductionOp.CumProd, innerAxisContiguous);
                var kernel = DirectILKernelGenerator.TryGetCumulativeAxisKernel(key);
                if (kernel != null)
                {
                    fixed (long* inputStrides = arr.Shape.strides)
                    fixed (long* shapePtr = arr.Shape.dimensions)
                    {
                        kernel((void*)arr.Address, (void*)ret.Address, inputStrides, shapePtr, axis, arr.ndim, arr.size);
                    }
                    return ret;
                }
            }

            // Fallback: iterator-based axis cumprod (handles broadcast, non-contiguous, edge cases)
            return ExecuteAxisCumProdFallback(inputArr, ret, axis);
        }

        /// <summary>
        /// Fallback axis cumprod on the new axis iterator path.
        /// </summary>
        private unsafe NDArray ExecuteAxisCumProdFallback(NDArray inputArr, NDArray ret, int axis)
        {
            var retType = ret.GetTypeCode;

            if (inputArr.GetTypeCode != retType)
                inputArr = Cast(inputArr, retType, copy: true);

            NpFunc.Invoke(retType, CumProdAxisDispatch<int>, inputArr.Storage, ret.Storage, axis);

            return ret;
        }

        protected unsafe NDArray cumprod_elementwise(NDArray arr, NPTypeCode? typeCode)
            => ScanElementwiseFlat(arr, typeCode, ReductionOp.CumProd);

        /// <summary>
        /// Fallback element-wise cumprod for contiguous input.
        /// </summary>
        private unsafe NDArray cumprod_elementwise_fallback(NDArray arr, NPTypeCode retType)
        {
            if (!arr.Shape.IsContiguous)
                throw new InvalidOperationException("cumprod_elementwise_fallback requires contiguous input.");

            var linearInput = arr.reshape(Shape.Vector(arr.size));
            var converted = linearInput.typecode == retType
                ? linearInput.Clone()
                : Cast(linearInput, retType, copy: true);

            NpFunc.Invoke(retType, CumProdInPlace<int>, (nint)converted.Address, converted.size);

            return converted;
        }

        private static void CumProdAxisDispatch<T>(UnmanagedStorage input, UnmanagedStorage output, int axis) where T : unmanaged, IMultiplyOperators<T, T, T>, IMultiplicativeIdentity<T, T>
            => NDAxisIter.ExecuteSameType<T, CumProdAxisKernel<T>>(input, output, axis);

        private static unsafe void CumProdInPlace<T>(nint addr, long size) where T : unmanaged, IMultiplyOperators<T, T, T>, IMultiplicativeIdentity<T, T>
        {
            var p = (T*)addr;
            T product = T.MultiplicativeIdentity;
            for (long i = 0; i < size; i++)
            {
                product *= p[i];
                p[i] = product;
            }
        }
    }
}
