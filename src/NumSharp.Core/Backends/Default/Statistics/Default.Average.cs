using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

// =============================================================================
// Default.Average.cs - DefaultEngine's np.average (numpy/lib/_function_base_impl.py: average, _weights_are_valid)
// =============================================================================
//
// DefaultEngine's override of TensorEngine.Average (TensorEngine.FunctionBase.cs): NumPy's statements after the facades'
// checks - the axis tuple (normalize_axis_tuple), the mean without weights and the element count it stands for, and with
// weights _weights_are_valid's reshape, the result dtype, the weight sum and its zero check, the weighted sum and the
// division, and `returned`'s weight sum in the average's shape; over text, NumPy's statements up to the reduction that
// refuses it. With weights it runs NumPy's composition statement by statement (the weight sum, its zero check, the
// products' sum, the division), so every sum is np.sum's add.reduce; the products' sum streams through np.evaluate where
// that gives the same bits (SumOfProducts). Until the np-layer plan's step 2 for this family a fused weighted-sum kernel
// computed both sums in one pass in its own SIMD order, a few ULP from NumPy's pairwise sums.
//
// History: the np-layer dtype plan's P2 moved these statements from the public np.average / np.average_returned into the
// internal operation class AverageEngine; its phase PE (E1) made that class this DefaultEngine partial. The facades keep
// the numpy.matrix boundary and the null check. Its phase P6 made each np.* call of the statements (np.mean, np.sum,
// np.multiply, np.true_divide, np.broadcast_to, np.evaluate) the member its facade dispatches to, and moved the file
// from beside the facades to Backends/Default/Statistics/.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.Average"/>: <c>average(a, axis, weights, returned, keepdims)</c> after
        ///     the facades' checks: the average and, for <paramref name="returned"/>, the sum of weights in the average's
        ///     shape.
        /// </summary>
        /// <param name="a">The operand (not null): any dtype and layout; text takes NumPy's text statements.</param>
        /// <param name="axis">The axes as given (a negative axis counts from the end), or null for every axis.</param>
        /// <param name="weights">The weights, or null for the plain mean: of <paramref name="a"/>'s shape, or of the
        ///     reduced axes' extents in the order of <paramref name="axis"/>.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <param name="returned">Also compute the sum of weights (NumPy's <c>returned=True</c>); false leaves the pair's
        ///     second element null.</param>
        /// <returns>The average and the sum of weights (null unless <paramref name="returned"/>): arrays the caller owns,
        ///     0-d for a reduction over every axis without keepdims.</returns>
        /// <exception cref="AxisError">An axis out of bounds (<c>axis: axis 9 is out of bounds for array of dimension
        ///     2</c>: NumPy names the argument).</exception>
        /// <exception cref="ValueError">An axis that appears twice (<c>repeated axis in `axis` argument</c>); weights not
        ///     of the reduced axes' shape (<c>Shape of weights must be consistent with shape of a along specified
        ///     axis.</c>).</exception>
        /// <exception cref="DivideByZeroException">Weights that sum to zero (<c>Weights sum to zero, can't be
        ///     normalized</c>), or a mean without weights whose result is empty (<c>division by zero</c>: NumPy's
        ///     <c>a.size / avg.size</c>).</exception>
        /// <exception cref="TypeError">Weights of another shape without an axis (<c>Axis must be specified when shapes of a
        ///     and weights differ.</c>); the weight sum's refusal of a text result dtype (U / S weights beside a number, a
        ///     fixed-width text operand: <c>The `dtype` and `signature` arguments to ufuncs only select the general
        ///     DType ...</c>; U weights beside StringDType: <c>Cannot pass a new user DType instance ...</c>); a text
        ///     operand without weights: the mean's add.reduce refusal.</exception>
        /// <exception cref="DTypePromotionError">Weights and an operand with no common DType: StringDType beside a number
        ///     or beside S.</exception>
        /// <exception cref="InvalidOperationException">Not reached: a weight sum in a text result dtype that NumPy refuses
        ///     but that computed (a wiring error inside NumSharp).</exception>
        /// <remarks>NumPy's parameter order (<c>returned</c> before the keyword-only <c>keepdims</c>); the body owns a
        ///     boundary scope, so only the returned arrays survive the call.</remarks>
        public override (NDArray avg, NDArray sumOfWeights) Average(
            NDArray a, int[] axis = null, NDArray weights = null, bool returned = false, bool keepdims = false)
        {
            // Boundary scope: reclaims the weight view, the products and the weighted sum, the
            // count and broadcast temps; each tuple component is yielded individually (a dropped
            // scl on the returned:false paths is reclaimed).
            using var scope = NDScope.Open();

            // Text: NumPy's average normalizes the axis tuple first (normalize_axis_tuple with argname "axis": "axis: axis 9
            // is out of bounds ...", "repeated axis in `axis` argument"), then a.mean (add.reduce) refuses text — or, with
            // weights, sums them with result_type(a, weights), a SIZED string descriptor ufuncs refuse by class.
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
            {
                // StringDType: the axis tuple normalized (average's argument name), then a.mean(axis) — or, with weights,
                // result_type and the weights' sum — computing until NumPy's statement refuses (NDVStringStatistics.Average).
                Backends.Strings.NDStringUfuncs.ValidateReduceAxes("average", a.ndim, axis);
                var avg = Backends.Strings.NDVStringStatistics.Average(a, Backends.Strings.NDStringUfuncs.ReducedAxes(a, axis),
                    axisGiven: axis is not null, weights, keepdims);
                if (!returned)
                    return (scope.Returns(avg), null);
                // scl = avg_as_array.dtype.type(a.size / avg_as_array.size): only a NaN-like NA mean (np.float64) returns,
                // always 0-d (an axis or keepdims meets the divide's refusal first), so the count is NumPy's float64 SCALAR
                // (NumSharp's stand-in: a read-only 0-d owner) and needs no broadcast.
                var sclValue = NDArray.Scalar((double)a.size / avg.size).MarkReductionScalar();
                return (scope.Returns(avg), scope.Returns(sclValue));
            }
            if (Backends.Strings.NDStringUfuncs.IsString(a))
            {
                // U / S: the axis tuple, the weight-free mean's refusal and the weights' shape checks first. Weights of a
                // valid shape go on to the weighted statements below, whose result_type and weight sum refuse the text
                // result dtype exactly as they do beside a numeric operand with text weights.
                var refusal = Backends.Strings.NDStringUfuncs.AverageAnswer(a, axis, weights);
                if (refusal is not null)
                    throw refusal;
            }

            int ndim = a.ndim;
            int[] normalizedAxis = NormalizeAxisTuple(axis, ndim);

            if (weights is null)
            {
                NDArray avg = MeanWithAxes(a, normalizedAxis, keepdims);
                // NumPy computes `scl = avg.dtype.type(a.size / avg.size)` unconditionally
                // (before the returned check), so 0/0 raises ZeroDivisionError even when
                // returned=False — see lib/_function_base_impl.average lines 576-578.
                if (avg.size == 0)
                    throw new DivideByZeroException("division by zero");

                if (!returned) return (scope.Returns(avg), null);

                // scl = avg_as_array.dtype.type(a.size / avg_as_array.size): a NumPy SCALAR of the mean's dtype (NumSharp's
                // stand-in: a read-only 0-d owner); `np.broadcast_to(scl, avg_as_array.shape).copy()` then makes it a fresh
                // array wherever the mean has dimensions.
                double count = (double)a.size / avg.size;
                NDArray scl = NDArray.Scalar(count).astype(avg.typecode).MarkReductionScalar();
                if (!scl.Shape.dimensions.SequenceEqual(avg.Shape.dimensions))
                    scl = NDArray.BroadcastTo(scl, avg.Shape).copy();
                return (scope.Returns(avg), scope.Returns(scl));
            }

            NDArray wgt = WeightsAreValid(weights, a, normalizedAxis);

            // result_dtype: NumPy's result_type over the DESCRIPTORS (with 'f8' for an integer or bool operand), so text
            // weights promote as NumPy promotes them - U / S beside a number to a sized string, StringDType beside a number
            // to nothing (DTypePromotionError) - instead of meeting a numeric table that has no text entry.
            DType resultDescr = ComputeResultDtype(a.dtype, wgt.dtype);
            if (Backends.Strings.NDStringUfuncs.IsTextRequest(resultDescr))
            {
                // scl = wgt.sum(axis=axis, dtype=result_dtype, ...): a text result_dtype is a descriptor INSTANCE, which a
                // ufunc's dtype= refuses ("only select the general DType" for a sized string). The refusal does not
                // depend on the axes (normalized and valid by now), so the sum is asked without them.
                using var refused = wgt.TensorEngine.Sum(wgt, null, resultDescr, false);
                throw new InvalidOperationException("np.average's weight sum in a text result dtype computed where NumPy refuses it.");
            }
            NPTypeCode resultDtype = resultDescr.GetTypeCode();

            // scl = wgt.sum(axis=axis, dtype=result_dtype, **keepdims_kw): the weights summed in the result dtype, each
            // element cast by the reduction's own loop (a cast input is summed one np.getbufsize() fill at a time), not a
            // cast copy summed in one piece, whose pairwise order differs past one fill. (NumPy's scl / avg are named
            // weightSum / average here: the branches above declare their own scl and avg.)
            NDArray weightSum = SumWithAxes(wgt, normalizedAxis, resultDtype, keepdims);
            // if np.any(scl == 0.0): raise ZeroDivisionError - before the product is formed.
            if (HasZero(weightSum))
                throw new DivideByZeroException("Weights sum to zero, can't be normalized");

            // avg = avg_as_array = np.multiply(a, wgt, dtype=result_dtype).sum(axis, **keepdims_kw) / scl
            NDArray weightedSum = SumOfProducts(a, wgt, resultDtype, normalizedAxis, keepdims);
            NDArray average = Quotient(weightedSum, weightSum);

            if (!returned) return (scope.Returns(average), null);
            // if scl.shape != avg_as_array.shape: scl = np.broadcast_to(scl, avg_as_array.shape).copy()
            if (!weightSum.Shape.dimensions.SequenceEqual(average.Shape.dimensions))
                weightSum = NDArray.BroadcastTo(weightSum, average.Shape).copy();
            return (scope.Returns(average), scope.Returns(weightSum));
        }

        /// <summary>
        ///     NumPy's weighted sum, <c>np.multiply(a, wgt, dtype=result_dtype).sum(axis, **keepdims_kw)</c>: the products in
        ///     the result dtype (each operand cast into it first), laid out as NumPy's ufunc lays them out, then np.sum over
        ///     them in that memory order.
        /// </summary>
        /// <param name="a">The operand (any numeric dtype and layout).</param>
        /// <param name="wgt">The weights, already of <paramref name="a"/>'s dimensions or broadcastable to them
        ///     (<see cref="WeightsAreValid"/>'s view).</param>
        /// <param name="resultDtype">The result dtype: the products' and the sum's loop.</param>
        /// <param name="axes">The normalized reduced axes, or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <returns>The weighted sum, a fresh array (0-d for a reduction over every axis without keepdims).</returns>
        /// <remarks>
        ///     <para>
        ///     Where the operands have the same dimensions and one memory order (both C- or both F-contiguous), the result
        ///     dtype is float32, float64 or complex128 and one axis or every axis is reduced, np.evaluate streams the products
        ///     into np.sum's pairwise schedule without the product buffer (<c>Sum(Cast(a) * Cast(w), axis)</c>): the same
        ///     products, rounded in the result dtype, added in the same order, so the same bits. Checked over 11,520
        ///     operand pairs (every input dtype pair with such a result, 8 shapes, C and F order, every axis, NaN payloads
        ///     of both signs, infinities and -0.0): all identical, once np.evaluate's short streamed rows kept the running
        ///     sum's NaN as np.sum's fold does (ff8a2fcf). It is 1.3x-3.3x faster than forming the product (one read of
        ///     each operand instead of a write and a read of the product), measured at (100, 1000) and (1000, 1000).
        ///     </para>
        ///     <para>
        ///     Everything else forms the product: a broadcast weight vector (1-D weights along the axis, a stride-0 operand
        ///     np.evaluate would materialize anyway), operands in different orders (the product's layout then follows
        ///     NumPy's vote), float16 (np.evaluate's float16 sum is not NumPy's HALF loop), an axis tuple of two axes or
        ///     more, a 0-d operand, and hosts without dynamic code.
        ///     </para>
        /// </remarks>
        private static NDArray SumOfProducts(NDArray a, NDArray wgt, NPTypeCode resultDtype, int[] axes, bool keepdims)
        {
            bool flat = axes is null || axes.Length == a.ndim;
            if (DirectILKernelGenerator.Enabled && a.ndim > 0
                && (resultDtype == NPTypeCode.Single || resultDtype == NPTypeCode.Double || resultDtype == NPTypeCode.Complex)
                && (flat || axes.Length == 1)
                && a.Shape.dimensions.SequenceEqual(wgt.Shape.dimensions)
                && ((a.Shape.IsContiguous && wgt.Shape.IsContiguous) || (a.Shape.IsFContiguous && wgt.Shape.IsFContiguous)))
            {
                // A Cast node only where an operand is not of the result dtype: np.multiply(dtype=) casts each input into
                // the loop first, exactly what the node does element by element.
                NDExpr x = a.typecode == resultDtype ? NDExpr.Arr(a) : NDExpr.Cast(NDExpr.Arr(a), resultDtype);
                NDExpr y = wgt.typecode == resultDtype ? NDExpr.Arr(wgt) : NDExpr.Cast(NDExpr.Arr(wgt), resultDtype);
                NDExpr products = x * y;
                // np.evaluate's engine is its expression's first array's: a's.
                return a.TensorEngine.Evaluate(
                    flat ? NDExpr.Sum(products, keepdims) : NDExpr.Sum(products, axes[0], keepdims));
            }

            NDArray product = UfuncResultBuffer(resultDtype, a, wgt);
            a.TensorEngine.Multiply(a, wgt, resultDtype, product, null);
            return SumWithAxes(product, axes, resultDtype, keepdims);
        }

        /// <summary>
        ///     A fresh, uninitialized buffer for the result of a ufunc over <paramref name="x"/> and <paramref name="y"/> (<paramref name="y"/>
        ///     broadcastable to <paramref name="x"/>), laid out as NumPy's iterator allocates it (<see cref="Shape.NpyIterOutputShape"/>:
        ///     each operand votes its memory order and abstains on a stride-0 or length-1 axis, C order wins a conflict). The
        ///     layout is observable: a later sum walks this memory order, and the result keeps it.
        /// </summary>
        /// <param name="dtype">The result dtype.</param>
        /// <param name="x">The operand whose dimensions the result takes.</param>
        /// <param name="y">The other operand, broadcastable to <paramref name="x"/>.</param>
        /// <returns>An uninitialized array of <paramref name="x"/>'s dimensions in NumPy's layout.</returns>
        private static NDArray UfuncResultBuffer(NPTypeCode dtype, NDArray x, NDArray y)
        {
            // A private copy of the dimensions: NpyIterOutputShape adopts the array it is handed.
            var dims = x.ndim == 0 ? Array.Empty<long>() : (long[])x.Shape.dimensions.Clone();
            return new NDArray(dtype, Shape.NpyIterOutputShape(dims, new[] { x.Shape, y.Shape }), false);
        }

        /// <summary>
        ///     NumPy's <c>num / scl</c> of the weighted average: for two scalars (a reduction over every axis without
        ///     keepdims) NumPy's scalar division, the same arithmetic as the ufunc loop (CDOUBLE_divide is called by the
        ///     complex scalars, IEEE division by the real ones, float32 arithmetic by float16), whose result is a scalar
        ///     (NumSharp's stand-in: a read-only 0-d owner); otherwise <c>true_divide</c>'s result laid out as its ufunc
        ///     lays it out.
        /// </summary>
        /// <param name="num">The weighted sum (the result dtype).</param>
        /// <param name="scl">The weight sum (the result dtype; broadcastable to <paramref name="num"/>: its kept axes are
        ///     <paramref name="num"/>'s or 1).</param>
        /// <returns>The average: a read-only 0-d scalar, or a fresh array of <paramref name="num"/>'s dimensions.</returns>
        private static NDArray Quotient(NDArray num, NDArray scl)
        {
            if (num.ndim == 0 && scl.ndim == 0)
                return num.TensorEngine.Divide(num, scl, null, null, null).MarkReductionScalar();
            var r = UfuncResultBuffer(num.GetTypeCode, num, scl);
            num.TensorEngine.Divide(num, scl, null, r, null);
            return r;
        }

        /// <summary>
        ///     np.average's first statement for an axis: <c>normalize_axis_tuple(axis, a.ndim, argname="axis")</c>.
        /// </summary>
        /// <param name="axis">The axes as given, or null for every axis.</param>
        /// <param name="ndim">The input's rank.</param>
        /// <returns>The normalized axes in input order, or null for null.</returns>
        /// <exception cref="AxisError">An axis is out of range: "axis: axis … is out of bounds for array of dimension …"
        ///     (NumPy names the argument).</exception>
        /// <exception cref="ValueError">An axis appears twice: "repeated axis in `axis` argument".</exception>
        private static int[] NormalizeAxisTuple(int[] axis, int ndim)
        {
            if (axis is null) return null;
            return Backends.DefaultEngine.normalize_axis_tuple(axis, ndim, "axis");
        }

        /// <summary>
        ///     <c>a.mean(axis, keepdims)</c> over normalized axes, NumPy's <c>_mean</c>: np.mean for none, one or every axis,
        ///     and for any other set (two axes or more, or the empty tuple) ONE add.reduce over the whole set divided by the
        ///     count of its elements (<see cref="Backends.DefaultEngine.MeanOverAxes"/>).
        /// </summary>
        /// <param name="a">The numeric operand.</param>
        /// <param name="axes">The normalized axes, or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <returns>The mean: a fresh array laid out as NumPy's reduction allocates it (the operand's axis order), or a
        ///     read-only 0-d scalar for every axis without keepdims. Over the empty tuple every element is added to the
        ///     identity and divided by 1 in the mean's dtype: a float64 copy of an integer operand, a -0.0 made +0.0. A decimal
        ///     operand over axes that hold no element gives float64 NaN, as np.mean answers decimal.</returns>
        /// <exception cref="OverflowException">A decimal sum leaves decimal's range.</exception>
        private static NDArray MeanWithAxes(NDArray a, int[] axes, bool keepdims)
        {
            // One axis is np.mean along it - tested first: a 1-D operand's (0,) keeps the one-output axis route, which
            // follows NumPy's schedule where the flat route still declines (a broadcast operand). None and a tuple of every
            // axis are one reduction in NumPy (PyUFunc_GenericReduction reads None as every axis).
            if (axes is not null && axes.Length == 1) return a.TensorEngine.Mean(a, axes[0], null, keepdims);
            if (axes is null || axes.Length == a.ndim) return a.TensorEngine.Mean(a, null, null, keepdims);
            // Any other set: ONE reduction over the whole set. A mean of single-axis means rounds after each axis, and the
            // empty tuple is still a sum (of one element per output) and a division.
            return Backends.DefaultEngine.MeanOverAxes(a, axes, keepdims);
        }

        /// <summary>
        ///     <c>a.sum(axis, dtype=dtype, keepdims)</c> over normalized axes: np.sum for none, one or every axis, and for any
        ///     other set (two axes or more, or the empty tuple) ONE add.reduce over the whole set
        ///     (<see cref="Backends.DefaultEngine.ReduceAddOverAxes"/>: NumPy's iterator schedule, from the identity).
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <param name="axes">The normalized axes, or null for every axis.</param>
        /// <param name="dtype">The loop dtype.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <returns>The sum: a fresh array (never <paramref name="a"/> itself, the empty tuple included: each element is
        ///     added to the identity), or a read-only 0-d scalar for every axis without keepdims.</returns>
        /// <exception cref="NotSupportedException">A set of two axes or more, or the empty tuple, in a loop other than
        ///     float16, float32, float64, complex128 or decimal (the weighted fallback reaches float16, complex128 and
        ///     decimal).</exception>
        /// <exception cref="OverflowException">A decimal sum leaves decimal's range.</exception>
        private static NDArray SumWithAxes(NDArray a, int[] axes, NPTypeCode dtype, bool keepdims)
        {
            // One axis is np.sum along it (tested first, as in MeanWithAxes); None and a tuple of every axis are one
            // reduction (the flat sum).
            if (axes is not null && axes.Length == 1)
                return a.TensorEngine.Sum(a, axes[0], dtype, keepdims);
            if (axes is null || axes.Length == a.ndim)
                return a.TensorEngine.Sum(a, null, dtype, keepdims);
            // Any other set: ONE reduction over the whole set (a sum of single-axis sums rounds after each axis; the empty
            // tuple adds each element to the identity, it does not return the operand).
            return Backends.DefaultEngine.ReduceAddOverAxes(a, axes, keepdims, dtype);
        }

        /// <summary>
        ///     NumPy's <c>_weights_are_valid</c>: weights of the operand's shape as they are; otherwise the weights of the
        ///     reduced axes (of shape <c>tuple(a.shape[ax] for ax in axis)</c>), transposed by <c>np.argsort(axis)</c> and
        ///     reshaped so they broadcast against the operand (the reduced axes' extents, 1 elsewhere).
        /// </summary>
        /// <param name="weights">The weights (not null).</param>
        /// <param name="a">The operand.</param>
        /// <param name="axis">The normalized axes, or null for every axis.</param>
        /// <returns><paramref name="weights"/> itself, or a view of it in the operand's rank.</returns>
        /// <exception cref="TypeError">Weights of another shape without an axis (<c>Axis must be specified when shapes of a
        ///     and weights differ.</c>; NumPy's TypeError).</exception>
        /// <exception cref="ValueError">Weights not of the reduced axes' shape (<c>Shape of weights must be consistent with
        ///     shape of a along specified axis.</c>). An empty axis tuple is an axis given: weights of another shape then
        ///     meet this check, not the TypeError.</exception>
        private static NDArray WeightsAreValid(NDArray weights, NDArray a, int[] axis)
        {
            NDArray wgt = weights;
            if (a.Shape.dimensions.SequenceEqual(wgt.Shape.dimensions))
                return wgt;

            // NumPy's _weights_are_valid: `if axis is None: raise TypeError(...)` — the axis tuple as given (an empty
            // tuple is not None), then the shape of the reduced axes.
            if (axis is null)
                throw new TypeError("Axis must be specified when shapes of a and weights differ.");

            long[] expected = new long[axis.Length];
            for (int i = 0; i < axis.Length; i++)
                expected[i] = a.Shape.dimensions[axis[i]];

            if (!wgt.Shape.dimensions.SequenceEqual(expected))
                throw new ValueError("Shape of weights must be consistent with shape of a along specified axis.");

            // wgt = wgt.transpose(np.argsort(axis))
            if (axis.Length > 1)
            {
                int[] perm = AverageArgSort(axis);
                bool identity = true;
                for (int i = 0; i < perm.Length; i++)
                    if (perm[i] != i) { identity = false; break; }
                if (!identity)
                    wgt = wgt.transpose(perm);
            }

            int ndim = a.ndim;
            long[] newShape = new long[ndim];
            for (int i = 0; i < ndim; i++)
                newShape[i] = Array.IndexOf(axis, i) >= 0 ? a.Shape.dimensions[i] : 1L;
            wgt = wgt.reshape(Shape.Adopt(newShape));
            return wgt;
        }

        /// <summary><c>np.argsort</c> of a short int array (the axis tuple): the positions in ascending value order.</summary>
        /// <param name="arr">The values (distinct: a normalized axis tuple).</param>
        /// <returns>The permutation that sorts <paramref name="arr"/>.</returns>
        private static int[] AverageArgSort(int[] arr)
        {
            int[] idx = new int[arr.Length];
            for (int i = 0; i < arr.Length; i++) idx[i] = i;
            Array.Sort(idx, (x, y) => arr[x].CompareTo(arr[y]));
            return idx;
        }

        /// <summary>
        ///     The weighted average's result dtype, NumPy's statement: <c>np.result_type(a.dtype, wgt.dtype, 'f8')</c> for an
        ///     integer or bool operand, else <c>np.result_type(a.dtype, wgt.dtype)</c> - the promotion over descriptors, so a
        ///     text descriptor promotes as NumPy promotes it.
        /// </summary>
        /// <param name="aType">The operand's descriptor.</param>
        /// <param name="wType">The weights' descriptor.</param>
        /// <returns>The result descriptor: a numeric one, or a text one (U / S weights beside a number give a sized
        ///     string, U / S beside StringDType give StringDType), which the caller's weight sum then refuses.</returns>
        /// <exception cref="DTypePromotionError">No common DType: StringDType beside a number or beside S (NumPy's text,
        ///     listing every DType of the sequence: <c>(Int32DType, StringDType, Float64DType)</c> for an int32
        ///     operand).</exception>
        private static DType ComputeResultDtype(DType aType, DType wType)
            => IsIntegralOrBool(aType.GetTypeCode())
                ? DTypePromotion.ResultType(aType, wType, np.float64)
                : DTypePromotion.ResultType(aType, wType);

        /// <summary>NumPy's <c>issubclass(a.dtype.type, (np.integer, np.bool))</c>, with char as NumSharp's integer.</summary>
        /// <param name="t">The dtype.</param>
        /// <returns>True for bool, the integer dtypes and char.</returns>
        private static bool IsIntegralOrBool(NPTypeCode t)
        {
            switch (t)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        ///     NumPy's <c>np.any(scl == 0.0)</c> for a weight sum of any dtype: the zero of its dtype, the equality and
        ///     np.any (false for an empty sum). Works for float16, complex and decimal, which have no
        ///     <c>Convert.ToDouble</c>.
        /// </summary>
        /// <param name="scl">The weight sum.</param>
        /// <returns>True when some element is zero.</returns>
        [NDScoped] // bool boundary: the zero scalar, its astype and the == mask are all reclaimed here
        private static bool HasZero(NDArray scl)
        {
            if (scl.size == 0) return false;
            NDArray zero = NDArray.Scalar(0).astype(scl.typecode);
            NDArray isZero = scl == zero;
            return isZero.TensorEngine.Any(isZero);
        }
    }
}
