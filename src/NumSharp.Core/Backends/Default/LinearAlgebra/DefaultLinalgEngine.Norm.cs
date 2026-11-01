using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>x.astype(float)</c> of a text operand, the float conversion of an integer / bool one,
        ///     the fast path of the default order over the whole array (raveled in memory order, <c>sqrt(x.dot(x))</c>, and for
        ///     a complex operand <c>sqrt(x_real.dot(x_real) + x_imag.dot(x_imag))</c>), then the one-axis
        ///     (<see cref="NormVectorBranch"/>) or two-axis (<see cref="NormMatrixBranch"/>) branch. Every main-namespace
        ///     call of those statements (ravel, real, imag, dot, sqrt, sum, abs, amax, amin, not_equal, power, reciprocal,
        ///     evaluate, transpose, moveaxis, reshape, ascontiguousarray, add, the allocations) is the member its facade
        ///     dispatches to, on the plain operands np.linalg.norm's class boundary leaves (a numpy.matrix and a
        ///     numpy.char.chararray operand are computed plain inside its region), and each nested norm or svdvals is this
        ///     engine's own member.
        /// </remarks>
        [NDScoped]
        public override NDArray Norm(NDArray x, object ord = null, int[] axis = null, bool keepdims = false)
        {
            // NumPy's norm converts a non-inexact input with x.astype(float): a text array is parsed (each element by
            // float(), the first failing one raising NumPy's text), and the norm runs on the float64 numbers.
            using var parsed = Backends.Strings.NDStringUfuncs.IsText(x) ? x.astype(np.float64, true, 'K', "unsafe") : null;
            if (parsed is not null)
                x = parsed;
            var a = ToFloating(x);
            int nd = a.ndim;

            // NumPy's fast path: with no axis and the default order, the whole array is one
            // vector — and it reaches it through `x.dot(x)`, so this mirrors that call rather
            // than composing a reduction with a different summation order.
            if (axis is null &&
                (ord is null
                 || (IsText(ord, out var word) && (word == "f" || word == "fro") && nd == 2)
                 || (IsNumber(ord, out double two) && two == 2d && nd == 1)))
            {
                // `x = x.ravel(order='K')`: memory order, so an F-ordered or transposed operand is summed in the order
                // NumPy's dot reads it (a view where the memory is one run, as NumPy's). np.ravel's member.
                var flat = NDArray.ravel(a, 'K');
                NDArray squared;
                if (a.typecode == NPTypeCode.Complex)
                {
                    // `sqnorm = x_real.dot(x_real) + x_imag.dot(x_imag)`: two real dot products over the strided parts
                    // (a BLAS ddot of stride two, as NumPy's) - not the reduction of the complex squares the axis forms
                    // take, which rounds differently. Each dot of two vectors is a NumPy scalar, so `+` is scalarmath,
                    // which keeps the FIRST operand's NaN where the array ufunc keeps the second's: PolyNumber's
                    // scalar add, not np.add of two 0-d arrays. The parts are np.real / np.imag's members, each dot
                    // np.dot's (CoreDot), the real part's first.
                    var re = NDArray.real_(flat);
                    var im = NDArray.imag_(flat);
                    squared = PolyNumber.Binary(Backends.Kernels.BinaryOp.Add, PolyNumber.ScalarOf(CoreDot(re, re, null)),
                        PolyNumber.ScalarOf(CoreDot(im, im, null))).ToNDArray();
                }
                else
                {
                    squared = CoreDot(flat, flat, null);
                }
                var whole = squared.TensorEngine.Sqrt(squared);   // np.sqrt's member
                // NumPy's `ret = ret.reshape(ndim * [1])`.
                return keepdims ? ReshapeLikeNumPy(whole, Ones(nd)) : whole;
            }

            int[] axes = axis ?? AllAxes(nd);

            // NumPy's one-axis branch hands the axis to its reductions as it is (they check it), so a string order is
            // refused before any axis check: NormVectorBranch normalizes where those reductions would.
            if (axes.Length == 1)
                return NormVectorBranch(a, ord, axes[0], keepdims);
            if (axes.Length == 2)
                return NormMatrixBranch(a, ord, axes, nd, keepdims);

            throw new ValueError("Improper number of dimensions to norm.");
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's tuple branch: the reduced axes moved to the front and merged into one (np.transpose's member and the
        ///     array object's reshape), then norm's one-axis branch over axis 0 (through this engine's <see cref="Norm"/>,
        ///     NumPy's <c>norm(x, ...)</c>, so a subclass that overrides it computes vector_norm too); without axes, the
        ///     norm of the flattened operand (np.ravel's member).
        /// </remarks>
        [NDScoped]
        public override NDArray VectorNorm(NDArray x, int[] axis = null, bool keepdims = false, object ord = null)
        {
            object order = ord ?? 2;
            if (axis is not null)
            {
                int nd = x.ndim;
                var normalized = Shape.normalize_axis_tuple(axis, nd);

                // rest = the axes not reduced, in order; newshape = axis + rest (the raw axes: transpose accepts them).
                var isReduced = new bool[nd];
                foreach (int ax in normalized)
                    isReduced[ax] = true;
                var permutation = new int[nd];
                var mergedShape = new long[1 + nd - normalized.Length];
                long mergedLength = 1;
                for (int i = 0; i < axis.Length; i++)
                {
                    permutation[i] = axis[i];
                    mergedLength *= x.Shape.dimensions[normalized[i]];
                }
                mergedShape[0] = mergedLength;
                for (int d = 0, k = axis.Length, r = 1; d < nd; d++)
                {
                    if (isReduced[d])
                        continue;
                    permutation[k++] = d;
                    mergedShape[r++] = x.Shape.dimensions[d];
                }

                var merged = x.TensorEngine.Transpose(x, permutation).reshape(Shape.Adopt(mergedShape));
                var res = Norm(merged, order, new[] {0}, false);
                if (!keepdims)
                    return res;

                // keepdims: the input's shape with every reduced axis set to 1 (NumPy reshapes: after the merge it
                // cannot reuse norm's own keepdims).
                var kept = (long[])x.Shape.dimensions.Clone();
                foreach (int ax in normalized)
                    kept[ax] = 1;
                return ReshapeLikeNumPy(res, kept);
            }

            // Flatten first, then reduce — and restore the rank afterwards when keepdims asks,
            // because the ravel has already erased it. np.ravel's member (order 'C'), then this engine's Norm.
            var flattened = Norm(NDArray.ravel(ToFloating(x), 'C'), order, new[] {0}, false);
            return keepdims ? ReshapeLikeNumPy(flattened, Ones(x.ndim)) : flattened;
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     <para>
        ///     NumPy's <c>res = norm(x, axis=_axis, ord=ord)</c> (through this engine's <see cref="Norm"/>, without keepdims)
        ///     and, for keepdims, <c>res.reshape(tuple(shape))</c> with the reduced axis of the input's shape set to 1.
        ///     </para>
        ///     <para>
        ///     NumPy's vector_norm never hands keepdims to norm ("We can't reuse np.linalg.norm(keepdims) because of the
        ///     reshape hacks above"), so its keepdims result is a reshape of the reduced array, not the reduction's own
        ///     keepdims array: a view that does not own its data.
        ///     </para>
        /// </remarks>
        [NDScoped]
        public override NDArray VectorNorm(NDArray x, int axis, bool keepdims = false, object ord = null)
        {
            var res = Norm(x, ord ?? 2, new[] {axis}, false);
            if (!keepdims)
                return res;

            // NumPy's `_axis = normalize_axis_tuple(axis, len(shape))` over the input's shape (taken before anything was
            // computed; norm already accepted the axis, so this cannot fail), then `shape[i] = 1` and the reshape.
            var shape = (long[])x.Shape.dimensions.Clone();
            shape[NormalizeAxis(axis, x.ndim)] = 1;
            return ReshapeLikeNumPy(res, shape);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's one statement <c>return norm(x, axis=(-2, -1), keepdims=keepdims, ord=ord)</c>, through this engine's
        ///     <see cref="Norm"/> (so a subclass that overrides it computes matrix_norm too), the null order standing for
        ///     NumPy's default <c>"fro"</c>. It opens no scope: the call is the whole body and its result is returned as it
        ///     is.
        /// </remarks>
        public override NDArray MatrixNorm(NDArray x, bool keepdims = false, object ord = null)
            => Norm(x, ord ?? "fro", new[] {-2, -1}, keepdims);

        /// <summary>
        ///     NumPy's <c>r.reshape(shape)</c> of a norm result, which may be one of NumPy's scalars: an array's reshape is a
        ///     view; a scalar's is a view of the fresh writeable array NumPy makes from it (WRITEABLE, not OWNDATA), and its
        ///     reshape to <c>()</c> is the scalar itself.
        /// </summary>
        /// <param name="r">The result: an array, or a 0-d array standing for NumPy's scalar (read-only when it is the
        ///     scalar stand-in of a reduction or a gufunc).</param>
        /// <param name="shape">The new shape (the same number of elements; a fresh array this call adopts).</param>
        /// <returns>A view of <paramref name="r"/> (of a copy, for a 0-d <paramref name="r"/>), or <paramref name="r"/>
        ///     itself for a 0-d result reshaped to <c>()</c>.</returns>
        /// <remarks>
        ///     Every 0-d norm result stands for NumPy's scalar: NumPy's reductions and ufuncs return scalars for 0-d results.
        ///     A view of NumSharp's read-only scalar stand-in would stay read-only, where NumPy's reshape of a scalar is
        ///     writeable, so the view is taken of a copy (a 0-d copy costs one element). The reshape is the array object's,
        ///     np.reshape's member, over a plain norm result.
        /// </remarks>
        [NDScopedCovered /* only callers: the [NDScoped] norm bodies and NormMatrixBranch under them; the copy lives on as the view's base */]
        private static NDArray ReshapeLikeNumPy(NDArray r, long[] shape)
        {
            if (r.ndim == 0)
            {
                if (shape.Length == 0)
                    return r;
                r = r.copy();
            }
            return r.reshape(Shape.Adopt(shape));
        }

        /// <summary>
        ///     NumPy's one-axis branch of <c>norm</c>: the vector order <paramref name="ord"/> over <paramref name="axis"/>.
        ///     The inf order reduces with <c>initial=0</c>, so an empty vector's inf-norm is 0.0 where -inf still raises. Any
        ///     other number is NumPy's general order, which keeps a float16 / float32 operand's dtype and raises a whole
        ///     vector's 0-d sum with scalarmath's power (<see cref="ScalarPower"/>).
        /// </summary>
        /// <param name="a">The floating operand.</param>
        /// <param name="ord">The order: null / 2, a number, or +-inf (text orders are refused).</param>
        /// <param name="axis">The axis as given (negative values count from the end): NumPy checks it in the reduction a
        ///     numeric order runs, so a text order is refused whatever the axis.</param>
        /// <param name="keepdims">Keep the axis as length 1.</param>
        /// <returns>The reduced norm.</returns>
        /// <exception cref="ValueError">A text order (<c>Invalid norm order '...' for vectors</c>), or -inf of an empty
        ///     axis (NumPy's empty-min error).</exception>
        /// <exception cref="TypeError">An order that is neither null, a number nor a string.</exception>
        /// <exception cref="AxisError">A numeric order with an axis out of range (NumPy's reductions' error).</exception>
        [NDScopedCovered] // always dispatched from the [NDScoped] Norm body; rides its scope
        private static NDArray NormVectorBranch(NDArray a, object ord, int axis, bool keepdims)
        {
            if (ord is null || (IsNumber(ord, out double n) && n == 2d))
            {
                // NumPy's add.reduce checks the axis here.
                axis = NormalizeAxis(axis, a.ndim);
                // NumPy's special case: sqrt(add.reduce((x.conj() * x).real, axis)). For a real float32 / float64
                // operand the product is x * x, and np.evaluate streams Sum(x * x, axis) in add.reduce's own schedule:
                // the same bits as the eager product + np.sum (each product rounded, then summed in NumPy's order),
                // with no N-element temporary (a (10000, 10) slab: 45 us against 152 us for the eager pair). x * x
                // also keeps a negative NaN's sign as NumPy's x.conj() * x does, where |x| * |x| cleared it. float16
                // and complex keep the eager form (np.evaluate's float16 sum is not NumPy's HALF schedule, and a
                // complex |x|^2 is the real part of NumPy's fused complex product, AbsSquared), as does a host without
                // dynamic code (no fused kernels).
                // np.evaluate's member with its default options (the expression's first leaf, a, names the engine, as
                // np.evaluate's resolution did), np.sum's and np.sqrt's members.
                if (Backends.Kernels.DirectILKernelGenerator.Enabled && a.typecode is NPTypeCode.Double or NPTypeCode.Single)
                {
                    var streamed = a.TensorEngine.Evaluate(NDExpr.Sum((NDExpr)a * a, axis, keepdims));
                    return streamed.TensorEngine.Sqrt(streamed);
                }
                var squares = AbsSquared(a);
                var summed = squares.TensorEngine.Sum(squares, axis, null, keepdims);
                return summed.TensorEngine.Sqrt(summed);
            }

            if (IsText(ord, out var word))
                throw new ValueError($"Invalid norm order '{word}' for vectors");

            double order = AsNumber(ord);
            // Every remaining branch is a reduction over the axis, which checks it: after the order's refusal above.
            axis = NormalizeAxis(axis, a.ndim);

            // NumPy 2.x reduces with initial=0 (`abs(x).max(axis=axis, keepdims=keepdims, initial=0)`): an EMPTY
            // vector's inf-norm is 0.0, not the empty-max ValueError. Every |x| is >= 0, so a non-empty answer is
            // unchanged (a NaN still propagates). np.abs' member, then np.amax's / np.amin's with the keywords given.
            if (double.IsPositiveInfinity(order))
            {
                var magnitudes = a.TensorEngine.Abs(a);
                return magnitudes.TensorEngine.ReduceAMax(magnitudes, axis, keepdims, null, null, 0, null);
            }
            if (double.IsNegativeInfinity(order))
            {
                var magnitudes = a.TensorEngine.Abs(a);
                return magnitudes.TensorEngine.ReduceAMin(magnitudes, axis, keepdims, null, null, null, null);
            }
            if (order == 0d)
            {
                // NumPy counts non-zeros in `x.real.dtype`, NOT `x.dtype` — so a COMPLEX operand's
                // zero-"norm" is float64, not complex. For every real dtype the real component's
                // dtype is the dtype itself, so only Complex needs the substitution. np.not_equal's member, then
                // np.sum's.
                var countType = a.typecode == NPTypeCode.Complex ? NPTypeCode.Double : a.typecode;
                var counted = a.TensorEngine.NotEqual(a, NDArray.Scalar(0)).astype(countType);
                return counted.TensorEngine.Sum(counted, axis, null, keepdims);
            }
            if (order == 1d)
            {
                var magnitudes = a.TensorEngine.Abs(a);
                return magnitudes.TensorEngine.Sum(magnitudes, axis, null, keepdims);
            }

            // decimal has no NumPy dtype, so no NumPy statement to follow: it keeps NumSharp's own three steps, each the
            // member np.abs / np.power / np.sum dispatch to.
            if (a.typecode == NPTypeCode.Decimal)
            {
                var magnitudes = a.TensorEngine.Abs(a);
                var raised = magnitudes.TensorEngine.Power(magnitudes, NDArray.Scalar(order));
                var reduced = raised.TensorEngine.Sum(raised, axis, null, keepdims);
                return reduced.TensorEngine.Power(reduced, NDArray.Scalar(1d / order));
            }

            // NumPy's general order, statement for statement:
            //     absx = abs(x); absx **= ord
            //     ret = add.reduce(absx, axis=axis, keepdims=keepdims)
            //     ret **= reciprocal(ord, dtype=ret.dtype)
            // `absx **= ord` is the ufunc in place with a Python scalar, which NEP 50 keeps weak: the loop is absx's own
            // float dtype (float16 / float32 stay), and np.power's zero-exponent-step fast path decides the orders 0.5
            // (sqrt) and -1 (the reciprocal) where NumPy's does. Each statement is its function's member: np.abs',
            // np.power's with out= (the facade's (x1, x2, dtype, out, where) call), np.sum's.
            var absx = a.TensorEngine.Abs(a);
            absx.TensorEngine.Power(absx, NDArray.Scalar(order), null, absx, null);
            var ret = absx.TensorEngine.Sum(absx, axis, null, keepdims);
            // `reciprocal(ord, dtype=ret.dtype)`: ord converted to ret's dtype, then 1 / ord IN that dtype - float32's
            // 1 / 1.1 is one ULP from the float64 quotient rounded to float32, which the last power then carries.
            // np.reciprocal's member, dtype= passed as the facade passes it.
            var ordinal = NDArray.Scalar(order);
            var recip = ordinal.TensorEngine.Reciprocal(ordinal, ret.dtype, null, null);
            if (ret.ndim > 0)
            {
                // An array's `**=` is the ufunc in place: the 0-d exponent takes np.power's fast path where NumPy's does
                // (1 / 0.5 = 2 squares, 1 / -1 = -1 takes the reciprocal).
                ret.TensorEngine.Power(ret, recip, null, ret, null);
                return ret;
            }

            // A 0-d ret is a NumPy scalar (a whole vector reduced without keepdims), and a scalar's `**=` rebinds the name
            // to `ret ** recip`: scalarmath's power, which has no fast path.
            return ScalarPower(ret, recip);
        }

        /// <summary>
        ///     NumPy's <c>ret **= recip</c> when <c>ret</c> is one of NumPy's scalars (the 0-d reduction of a whole vector):
        ///     Python rebinds <c>ret</c> to <c>ret ** recip</c>, scalarmath's power - <c>npy_pow</c> (the C runtime's
        ///     <c>pow</c>, <see cref="Math.Pow"/>) for float64, <c>npy_powf</c> (<see cref="MathF.Pow"/>) for float32, and
        ///     <c>half_ctype_power</c> for float16 (<c>npy_powf</c> of the operands widened to float32, rounded once back to
        ///     float16).
        /// </summary>
        /// <param name="ret">The 0-d float16 / float32 / float64 reduction result.</param>
        /// <param name="recip">The 0-d exponent, of <paramref name="ret"/>'s dtype.</param>
        /// <returns>A new 0-d array of <paramref name="ret"/>'s dtype standing for NumPy's scalar (read-only).</returns>
        /// <remarks>
        ///     np.power on the two 0-d arrays would run NumPy's ARRAY loop, whose fast path squares for the exponent 2 (the
        ///     order 0.5), and <c>pow(x, 2.0)</c> is not always <c>x * x</c>: for x = 757.7617475757002 pow gives
        ///     <c>0x1.185f5bb7003c2p+19</c> and the product <c>...3c1p+19</c>, so NumPy's <c>norm(v, 0.5)</c> and
        ///     <c>norm(v, 0.5, keepdims=True)</c> of <c>v = [x * x]</c> differ in the last bit.
        /// </remarks>
        [NDScopedCovered] // only caller: NormVectorBranch under the [NDScoped] Norm body; the widened / narrowed copies ride its scope
        private static NDArray ScalarPower(NDArray ret, NDArray recip)
        {
            switch (ret.typecode)
            {
                case NPTypeCode.Double:
                    return NDArray.Scalar(Math.Pow(ret.GetDouble(), recip.GetDouble())).MarkReductionScalar();
                case NPTypeCode.Single:
                    return NDArray.Scalar(MathF.Pow(ret.GetSingle(), recip.GetSingle())).MarkReductionScalar();
                default:
                    // float16, the one other dtype |x| reduces to here (a complex operand's |x| is float64, decimal never
                    // arrives): the house casts are npy_half_to_float and npy_float_to_half bit for bit.
                    float b = ret.astype(np.float32).GetSingle();
                    float e = recip.astype(np.float32).GetSingle();
                    return NDArray.Scalar(MathF.Pow(b, e)).astype(np.float16).MarkReductionScalar();
            }
        }

        /// <summary>
        ///     NumPy's two-axis branch of <c>norm</c>: the matrix order <paramref name="ord"/> over the axis pair. The 1
        ///     and inf orders reduce with <c>initial=0</c> (an empty matrix gives 0.0); -1 and -inf keep the empty-min
        ///     error; 2, -2 and "nuc" go through the singular values.
        /// </summary>
        /// <param name="a">The floating operand.</param>
        /// <param name="ord">The order: null / "fro" / "f", "nuc", or a number among 1, -1, 2, -2, +-inf.</param>
        /// <param name="axes">The (row, column) axes as given.</param>
        /// <param name="nd">The operand's rank.</param>
        /// <param name="keepdims">Put both reduced axes back as length 1.</param>
        /// <returns>The reduced norm.</returns>
        /// <exception cref="ValueError">Duplicate axes, an invalid matrix order, or a min order of an empty
        ///     matrix.</exception>
        /// <exception cref="AxisError">An axis out of range.</exception>
        /// <exception cref="TypeError">An order that is neither null, a number nor a string.</exception>
        /// <exception cref="OpenBlasMissingBackendException">An order 2, -2 or "nuc" over a non-empty matrix with no matrix backend.</exception>
        /// <remarks>An instance member because the 2 / -2 / nuclear orders take this engine's own singular values
        ///     (<see cref="SingularValueNorm"/>, through <see cref="Svdvals"/>).</remarks>
        [NDScopedCovered /* always dispatched from the [NDScoped] Norm body; rides its scope */]
        private NDArray NormMatrixBranch(NDArray a, object ord, int[] axes, int nd, bool keepdims)
        {
            int rowAxis = NormalizeAxis(axes[0], nd);
            int colAxis = NormalizeAxis(axes[1], nd);
            if (rowAxis == colAxis)
                throw new ValueError("Duplicate axes given.");

            NDArray result;

            if (IsText(ord, out var word))
            {
                switch (word)
                {
                    case "f":
                    case "fro":
                        result = FrobeniusOverBoth(a, rowAxis, colAxis);
                        break;
                    case "nuc":
                        result = SingularValueNorm(a, rowAxis, colAxis, "nuc");
                        break;
                    default:
                        throw new ValueError("Invalid norm order for matrices.");
                }
            }
            else if (ord is null)
            {
                result = FrobeniusOverBoth(a, rowAxis, colAxis);
            }
            else
            {
                double order = AsNumber(ord);

                // NumPy reduces one axis FIRST and then shifts the surviving axis index down if
                // the reduced one sat before it — so these pairs are not interchangeable. The two
                // max forms reduce with initial=0 like NumPy 2.x (`.max(axis=..., initial=0)`): an
                // EMPTY matrix's 1-norm and inf-norm are 0.0; the min forms keep NumPy's empty-min
                // error.
                if (order == 2d || order == -2d)
                    result = SingularValueNorm(a, rowAxis, colAxis, order == 2d ? "2" : "-2");
                else if (order == 1d)
                    result = ExtremeOfAxisSums(a, rowAxis, Shift(colAxis, rowAxis), largest: true);
                else if (order == -1d)
                    result = ExtremeOfAxisSums(a, rowAxis, Shift(colAxis, rowAxis), largest: false);
                else if (double.IsPositiveInfinity(order))
                    result = ExtremeOfAxisSums(a, colAxis, Shift(rowAxis, colAxis), largest: true);
                else if (double.IsNegativeInfinity(order))
                    result = ExtremeOfAxisSums(a, colAxis, Shift(rowAxis, colAxis), largest: false);
                else
                    throw new ValueError("Invalid norm order for matrices.");
            }

            if (!keepdims)
                return result;

            // NumPy's `ret_shape = list(x.shape); ret_shape[axis[0]] = 1; ret_shape[axis[1]] = 1;
            // ret = ret.reshape(ret_shape)`: the two reduced axes back as length 1, a view of the result (of the array a
            // scalar result becomes).
            var retShape = (long[])a.Shape.dimensions.Clone();
            retShape[rowAxis] = 1;
            retShape[colAxis] = 1;
            return ReshapeLikeNumPy(result, retShape);
        }

        /// <summary>
        ///     NumPy's 1 / -1 / inf / -inf matrix orders, <c>add.reduce(abs(x), axis=sumAxis).max(axis=extremeAxis,
        ///     initial=0)</c> or <c>.min(axis=extremeAxis)</c>: the absolute values summed along one matrix axis, then the
        ///     largest or smallest of those sums along the other.
        /// </summary>
        /// <param name="a">The floating operand.</param>
        /// <param name="sumAxis">The normalized axis summed first (the rows for 1 / -1, the columns for inf / -inf).</param>
        /// <param name="extremeAxis">The other matrix axis, as numbered after <paramref name="sumAxis"/> was reduced away
        ///     (<see cref="Shift"/>).</param>
        /// <param name="largest">True for the maximum with <c>initial=0</c> (the 1 and inf orders: an empty matrix gives
        ///     0.0); false for the minimum, which keeps NumPy's empty-min error.</param>
        /// <returns>The reduced norm, a fresh array.</returns>
        /// <exception cref="ValueError">The minimum of an empty axis (<c>zero-size array to reduction operation minimum
        ///     which has no identity</c>).</exception>
        /// <remarks>Each step is the member its main-namespace function dispatches to, in the statement's order:
        ///     np.abs', np.sum's (no dtype, no keepdims), then np.amax's / np.amin's with only <c>initial</c> set for the
        ///     maximum.</remarks>
        [NDScopedCovered /* only reached via NormMatrixBranch, itself under the [NDScoped] Norm scope */]
        private static NDArray ExtremeOfAxisSums(NDArray a, int sumAxis, int extremeAxis, bool largest)
        {
            var magnitudes = a.TensorEngine.Abs(a);
            var sums = magnitudes.TensorEngine.Sum(magnitudes, sumAxis, null, false);
            return largest
                ? sums.TensorEngine.ReduceAMax(sums, extremeAxis, false, null, null, 0, null)
                : sums.TensorEngine.ReduceAMin(sums, extremeAxis, false, null, null, null, null);
        }

        /// <summary>
        ///     NumPy's Frobenius norm over an axis pair, <c>sqrt(add.reduce((x.conj() * x).real, axis=axis))</c>: ONE
        ///     reduction over both axes, summed in the order NumPy's reduction iterator visits the squares, which are laid
        ///     out in the operand's K order. Two one-axis sums add the same squares in another order, so their last bits
        ///     can differ from NumPy's.
        /// </summary>
        /// <param name="a">The floating operand (NumPy's <c>x</c> after its float conversion: NumSharp's cast keeps the
        ///     K layout, as NumPy's <c>astype</c> does, so the squares' order can be read from it).</param>
        /// <param name="rowAxis">The normalized row axis.</param>
        /// <param name="colAxis">The normalized column axis (not <paramref name="rowAxis"/>).</param>
        /// <returns>A fresh array: the norm, both axes reduced, its kept axes laid out as NumPy lays out the reduction's
        ///     result (in the operand's memory order) and owning its buffer.</returns>
        /// <remarks>
        ///     <para><b>The order.</b> <c>(x.conj() * x).real</c> is a ufunc result, laid out in <c>x</c>'s K order
        ///     (<see cref="Shape.NpyIterOutputShape"/>), and the reduction iterator walks it in that order. So the squares
        ///     are computed over <c>np.transpose(a, perm)</c>, the operand transposed into its memory order, which makes
        ///     them C-contiguous and the walk a C-order walk (<see cref="SumAxisPairOfCOrderSquares"/>). The result's axes
        ///     are then in that order too, which is where NumPy allocates them: they are relabelled back to the operand's
        ///     axis order (<see cref="NDArray.AdoptAxisPermutation"/>), so the answer keeps NumPy's layout and owns its
        ///     buffer, as the <c>sqrt</c> NumPy returns does.</para>
        ///     <para><b>Not modelled.</b> An empty operand keeps the two one-axis sums: every sum is the identity's 0 (or
        ///     the result holds no element), whatever the order.</para>
        ///     <para>The sums, the root, the transpose and the contiguous copy are the members np.sum, np.sqrt,
        ///     np.transpose and np.ascontiguousarray dispatch to.</para>
        /// </remarks>
        [NDScopedCovered /* only reached via NormMatrixBranch, itself under the [NDScoped] Norm scope */]
        private static NDArray FrobeniusOverBoth(NDArray a, int rowAxis, int colAxis)
        {
            if (a.size == 0)
            {
                var squares = AbsSquared(a);
                var once = squares.TensorEngine.Sum(squares, rowAxis, null, false);
                var twice = once.TensorEngine.Sum(once, Shift(colAxis, rowAxis), null, false);
                return twice.TensorEngine.Sqrt(twice);
            }

            int nd = a.ndim;
            // The squares' layout: the ufunc vote of the operand alone (the complex product's two operands, x.conj() and
            // x, are in one order), outermost axis first.
            var layout = a.Shape;
            int[] perm = Shape.NpyIterOutputShape((long[])layout.dimensions.Clone(), new ReadOnlySpan<Shape>(in layout))
                .DenseAxisOrder();
            bool identity = true;
            for (int i = 0; i < nd && identity; i++)
                identity = perm[i] == i;

            var inOrder = identity ? a : a.TensorEngine.Transpose(a, perm);
            int pr = identity ? rowAxis : Array.IndexOf(perm, rowAxis);
            int pc = identity ? colAxis : Array.IndexOf(perm, colAxis);

            var squared = AbsSquared(inOrder);
            // AbsSquared allocates a fresh C-ordered result over a C-ordered operand; should a kernel ever hand back
            // another layout, a C copy keeps the walk below a C-order walk (the values do not depend on the layout).
            if (!squared.Shape.IsContiguous)
                squared = NDArray.ascontiguousarray(squared, null);

            var sums = SumAxisPairOfCOrderSquares(squared, Math.Min(pr, pc), Math.Max(pr, pc));
            var root = sums.TensorEngine.Sqrt(sums);
            if (identity || root.ndim < 2)
                return root;   // the kept axes are already in the operand's order (one or none is in any order)

            // root's axes are the kept axes in perm order; answer axis i is the i-th kept axis in the operand's order.
            var keptInOrder = new int[nd - 2];
            for (int i = 0, j = 0; i < nd; i++)
                if (perm[i] != rowAxis && perm[i] != colAxis)
                    keptInOrder[j++] = perm[i];
            var permutation = new int[nd - 2];
            for (int ax = 0, i = 0; ax < nd; ax++)
                if (ax != rowAxis && ax != colAxis)
                    permutation[i++] = Array.IndexOf(keptInOrder, ax);
            return NDArray.AdoptAxisPermutation(root, permutation);
        }

        /// <summary>
        ///     NumPy's <c>add.reduce</c> over two axes of a C-contiguous array of squares, in the order its reduction
        ///     iterator adds them, built from one-axis sums - the member <see cref="np.sum(NDArray, int?, DType, NDArray, bool, object, NDArray)"/> dispatches to
        ///     (each NumPy's own add.reduce over one axis, bit for bit).
        /// </summary>
        /// <param name="s">The squares: C-contiguous, non-empty, a float dtype or decimal.</param>
        /// <param name="lo">The smaller reduced axis.</param>
        /// <param name="hi">The larger reduced axis.</param>
        /// <returns>The sums, of <paramref name="s"/>'s dims without <paramref name="lo"/> and <paramref name="hi"/>, C
        ///     order: a fresh array, or a view of <paramref name="s"/> when both reduced axes have extent 1.</returns>
        /// <remarks>
        ///     <para><b>The iterator.</b> NumPy reduces with one NpyIter over the result (stride 0 along the reduced axes)
        ///     and the squares. A C-contiguous operand is walked in C order; extent-1 axes are dropped; adjacent axes merge
        ///     when every operand's strides chain, which for two adjacent REDUCED axes they always do here (and a reduced
        ///     axis never merges with a kept one: the result's stride 0 does not chain with its non-zero stride). The
        ///     innermost axis is the ufunc loop: a reduced run there is one call, <c>out += pairwise_sum(run)</c>; a kept one
        ///     adds element by element into the result. So, over the axes of extent &gt; 1:</para>
        ///     <list type="bullet">
        ///         <item>two reduced axes next to each other are ONE axis of their product's extent (a reshape);</item>
        ///         <item>otherwise, with the inner reduced axis innermost, each run is pairwise-summed and the runs are
        ///         added in order of the outer reduced axis: the inner axis's sum, then the outer one's;</item>
        ///         <item>otherwise every reduced element is added into its result in turn, outer reduced index first
        ///         (one sequential chain): the two axes moved to the front, copied C-contiguous, merged, and summed along
        ///         the merged leading axis, whose one-axis sum is that sequential chain;</item>
        ///         <item>a reduced axis of extent 1 adds one element to the identity, which leaves a square as it is
        ///         (a square is never <c>-0.0</c>, and <c>0 + NaN</c> keeps the NaN), so it drops out.</item>
        ///     </list>
        /// </remarks>
        [NDScopedCovered /* only caller: FrobeniusOverBoth, under the [NDScoped] Norm scope; the views and copies ride it */]
        private static NDArray SumAxisPairOfCOrderSquares(NDArray s, int lo, int hi)
        {
            long[] d = s.Shape.dimensions;
            int n = d.Length;
            var target = new long[n - 2];
            for (int i = 0, j = 0; i < n; i++)
                if (i != lo && i != hi)
                    target[j++] = d[i];

            // The axes NumPy's iterator keeps: those of extent > 1 (a reshape of a C-contiguous array is a view).
            int big = 0;
            for (int i = 0; i < n; i++)
                if (d[i] > 1)
                    big++;
            var squeezed = new long[big];
            int qLo = -1, qHi = -1;
            for (int i = 0, j = 0; i < n; i++)
            {
                if (d[i] <= 1)
                    continue;
                if (i == lo)
                    qLo = j;
                else if (i == hi)
                    qHi = j;
                squeezed[j++] = d[i];
            }

            var y = big == n ? s : s.reshape(squeezed);
            NDArray r;
            if (qLo < 0 && qHi < 0)
            {
                r = y;
            }
            else if (qLo < 0 || qHi < 0)
            {
                r = AxisSum(y, qLo < 0 ? qHi : qLo);
            }
            else if (qHi == qLo + 1)
            {
                var merged = new long[big - 1];
                for (int i = 0, j = 0; i < big; i++)
                {
                    if (i == qHi)
                        continue;
                    merged[j++] = i == qLo ? squeezed[qLo] * squeezed[qHi] : squeezed[i];
                }
                r = AxisSum(y.reshape(merged), qLo);
            }
            else if (qHi == big - 1)
            {
                r = y.typecode == NPTypeCode.Half ? HalfRunSumsAddedInTurn(y, qLo, qHi) : AxisSum(AxisSum(y, qHi), qLo);
            }
            else
            {
                // np.ascontiguousarray(np.moveaxis(...)): np.moveaxis' member, then the array object's contiguous copy.
                var front = NDArray.ascontiguousarray(y.TensorEngine.MoveAxis(y, new[] {qLo, qHi}, new[] {0, 1}), null);
                var chain = new long[big - 1];
                chain[0] = squeezed[qLo] * squeezed[qHi];
                for (int i = 0, j = 1; i < big; i++)
                    if (i != qLo && i != qHi)
                        chain[j++] = squeezed[i];
                r = AxisSum(front.reshape(chain), 0);
            }

            // Put the kept axes of extent 1 back (a view of the C-ordered sums).
            return r.ndim == target.Length ? r : r.reshape(target);
        }

        /// <summary>
        ///     <c>np.sum(x, axis)</c>'s member: <see cref="TensorEngine.Sum"/> of <paramref name="x"/>'s engine over one
        ///     axis, no dtype, no keepdims - NumPy's own add.reduce over that axis, bit for bit.
        /// </summary>
        /// <param name="x">The operand (plain; its engine computes the sum).</param>
        /// <param name="axis">The axis to reduce; a negative value counts from the end, as np.sum's does.</param>
        /// <returns>A fresh array without <paramref name="axis"/>, of the sum's NEP 50 dtype (the operand's own for
        ///     every float dtype these callers pass).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="x"/>.</exception>
        /// <remarks>The one-axis sum the axis-pair schedules and the nuclear norm repeat, kept as one call so each
        ///     statement reads as the np.sum it stands for.</remarks>
        private static NDArray AxisSum(NDArray x, int axis) => x.TensorEngine.Sum(x, axis, null, false);

        /// <summary>
        ///     NumPy's float16 reduction over two reduced axes that do not merge, the inner one innermost: HALF_add's
        ///     reduce loop widens the running result to float32, adds the run's pairwise sum (computed in float32) and
        ///     rounds back to float16 ONCE per run, so a run's own sum is never rounded to float16.
        /// </summary>
        /// <param name="y">The float16 squares, C-contiguous, every axis of extent &gt; 1.</param>
        /// <param name="qLo">The outer reduced axis (a kept axis lies between it and <paramref name="qHi"/>).</param>
        /// <param name="qHi">The inner reduced axis, the last one.</param>
        /// <returns>A fresh float16 array of <paramref name="y"/>'s dims without the two reduced axes, C order.</returns>
        /// <remarks>
        ///     Two float16 one-axis sums would round each run's sum to float16 and then round again on every addition,
        ///     which differs in the last bit. So the run sums are taken over a float32 copy (the widening is exact, and a
        ///     float32 run's pairwise sum is HALF_pairwise_sum's: the same tree of float32 additions over the widened
        ///     elements, at any run length), and then added into the float16 result one run at a time in the order of
        ///     the outer reduced index, each addition in float32 and stored with one rounding (np.add's float32 loop
        ///     into a float16 out). The other float dtypes need none of this: their run sum is already in the
        ///     result's precision.
        ///     <para>The sum, the axis move, the allocation and the additions are the members np.sum, np.moveaxis,
        ///     np.zeros and np.add (with out=) dispatch to.</para>
        /// </remarks>
        [NDScopedCovered /* only caller: SumAxisPairOfCOrderSquares, under the [NDScoped] Norm scope */]
        private static NDArray HalfRunSumsAddedInTurn(NDArray y, int qLo, int qHi)
        {
            var runs = AxisSum(y.astype(np.float32), qHi);
            // runs keeps qLo where it was (qHi was after it); its slabs along qLo are the per-run sums to add in turn.
            var slabs = runs.TensorEngine.MoveAxis(runs, new[] {qLo}, new[] {0});
            // np.zeros(shape, float16)'s own allocation: the zero-filled accumulator.
            var acc = y.TensorEngine.NewArray(NPTypeCode.Half, Shape.Adopt((long[])slabs.Shape.dimensions[1..]), true);
            for (long i = 0; i < slabs.Shape.dimensions[0]; i++)
                acc.TensorEngine.Add(acc, slabs[i], null, acc, null);
            return acc;
        }

        /// <summary>An axis index after a LOWER-numbered axis has been reduced away.</summary>
        /// <param name="axis">The surviving axis, as numbered before the reduction.</param>
        /// <param name="removed">The reduced axis.</param>
        /// <returns><paramref name="axis"/> one lower when <paramref name="removed"/> sat before it, else unchanged.</returns>
        private static int Shift(int axis, int removed) => axis > removed ? axis - 1 : axis;

        /// <summary>
        ///     NumPy's <c>_multi_svd_norm</c>: the singular values over the axis pair, reduced by the largest (2, with
        ///     <c>initial=0</c>), the smallest (-2) or their sum ("nuc").
        /// </summary>
        /// <param name="a">The floating operand.</param>
        /// <param name="rowAxis">The normalized row axis.</param>
        /// <param name="colAxis">The normalized column axis.</param>
        /// <param name="which">"2", "-2" or "nuc".</param>
        /// <returns>The reduced norm.</returns>
        /// <exception cref="OpenBlasMissingBackendException">No matrix backend computes the singular values of a non-empty operand.</exception>
        /// <exception cref="ValueError">"-2" of an empty matrix (the empty-min error).</exception>
        /// <remarks>An instance member because the singular values are this engine's <see cref="Svdvals"/> (NumPy's
        ///     <c>svd(y, compute_uv=False)</c>, so a subclass that overrides it computes the 2 / -2 / nuclear norms too);
        ///     the axis move and the reductions are the members np.moveaxis, np.amax, np.amin and np.sum dispatch
        ///     to.</remarks>
        [NDScopedCovered /* only reached via NormMatrixBranch, itself under the [NDScoped] Norm scope */]
        private NDArray SingularValueNorm(NDArray a, int rowAxis, int colAxis, string which)
        {
            // NumPy's _multi_svd_norm: move the two matrix axes to the end, take the singular values
            // over them, then reduce — 2 → largest, -2 → smallest, 'nuc' → sum of the singulars.
            // NumPy calls `_multi_svd_norm(x, row, col, amax, 0)`: the largest singular value of an
            // EMPTY matrix is 0.0 (singular values are >= 0, so a non-empty answer is unchanged).
            // 'nuc' passes initial=0 to the sum, which the add identity already is.
            var y = a.TensorEngine.MoveAxis(a, new[] {rowAxis, colAxis}, new[] {-2, -1});
            var s = Svdvals(y);
            return which switch
            {
                "2" => s.TensorEngine.ReduceAMax(s, -1, false, null, null, 0, null),
                "-2" => s.TensorEngine.ReduceAMin(s, -1, false, null, null, null, null),
                _ => AxisSum(s, -1), // "nuc"
            };
        }

        /// <summary>
        ///     NumPy computes norms in floating point — a bool or integer operand is cast up
        ///     before anything else happens, which is why <c>norm(arange(3))</c> is float64.
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <returns><paramref name="x"/> itself for a float16 / float32 / float64 / decimal / complex operand, else its
        ///     float64 cast.</returns>
        private static NDArray ToFloating(NDArray x)
            => x.typecode switch
            {
                NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double
                    or NPTypeCode.Decimal or NPTypeCode.Complex => x,
                _ => x.TensorEngine.Cast(x, typeof(double), copy: false)
            };

        /// <summary>Every axis of a rank, in order.</summary>
        /// <param name="nd">The rank.</param>
        /// <returns><c>[0, 1, ..., nd - 1]</c>.</returns>
        private static int[] AllAxes(int nd)
        {
            var all = new int[nd];
            for (int i = 0; i < nd; i++)
                all[i] = i;
            return all;
        }

        /// <summary>A shape of length-1 axes: keepdims after a whole-array reduction.</summary>
        /// <param name="nd">The rank.</param>
        /// <returns><paramref name="nd"/> ones, a fresh array (<see cref="ReshapeLikeNumPy"/> adopts it).</returns>
        private static long[] Ones(int nd)
        {
            var ones = new long[nd];
            for (int i = 0; i < nd; i++)
                ones[i] = 1;
            return ones;
        }

        /// <summary>
        ///     NumPy's <c>normalize_axis_index</c> for the norm family (norm's matrix branch, and the reduce its vector
        ///     branch runs through, which takes the axis as a TUPLE - so a 0-d input has no axis 0 here).
        /// </summary>
        /// <param name="axis">The axis; negative values count from the end.</param>
        /// <param name="nd">The operand's rank.</param>
        /// <returns>The non-negative axis.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        private static int NormalizeAxis(int axis, int nd) => Shape.check_and_adjust_axis(nd, axis);

        /// <summary>
        ///     Python's <c>x.shape[axis]</c>: the tuple index np.linalg.cross reads BEFORE it normalizes anything, so an
        ///     axis out of range is the IndexError Python raises there, not an AxisError.
        /// </summary>
        /// <param name="axis">The axis; negative values count from the end.</param>
        /// <param name="nd">The operand's rank.</param>
        /// <returns>The non-negative axis.</returns>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range").</exception>
        private static int ShapeIndex(int axis, int nd)
        {
            if (axis < -nd || axis >= nd)
                throw new IndexError("tuple index out of range");
            return axis < 0 ? axis + nd : axis;
        }

        /// <summary>Whether an order is a name (a C# string: NumPy's str order).</summary>
        /// <param name="ord">The order argument.</param>
        /// <param name="word">The name, or null.</param>
        /// <returns>True when <paramref name="ord"/> is a string.</returns>
        private static bool IsText(object ord, out string word)
        {
            word = ord as string;
            return word is not null;
        }

        /// <summary>Whether an order is a number, and its value.</summary>
        /// <param name="ord">The order argument.</param>
        /// <param name="value">The order as a double, or 0.</param>
        /// <returns>False for null and for a string; true otherwise (with the converted value).</returns>
        /// <exception cref="TypeError">An order that does not convert to a double.</exception>
        private static bool IsNumber(object ord, out double value)
        {
            value = 0;
            if (ord is null || ord is string)
                return false;
            value = AsNumber(ord);
            return true;
        }

        /// <summary>An order's value as a double.</summary>
        /// <param name="ord">The order argument (not a string).</param>
        /// <returns>The converted value.</returns>
        /// <exception cref="TypeError">An order that does not convert: <c>'ord' must be None, a number or one of 'fro',
        ///     'f', 'nuc'</c>.</exception>
        private static double AsNumber(object ord)
        {
            try
            {
                return Convert.ToDouble(ord, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception e) when (e is FormatException or InvalidCastException or OverflowException)
            {
                throw new TypeError("'ord' must be None, a number or one of 'fro', 'f', 'nuc'", e);
            }
        }
    }
}
