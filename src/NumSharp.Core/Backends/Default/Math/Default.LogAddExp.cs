using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =====================================================================
        // np.logaddexp / np.logaddexp2 / np.nextafter / np.copysign / np.hypot / np.heaviside
        //
        // Float-tier binary ufuncs whose loop signatures (ee->e, ff->f, dd->d, gg->g) and dtype
        // promotion are IDENTICAL to arctan2, so the promotion / path-classification / kernel-dispatch
        // machinery is shared with Default.ATan2.cs (PromoteATan2Binary, ClassifyATan2Path,
        // ExecuteATan2Kernel, ConvertToDouble/ConvertToDecimal). The only op-specific piece is the
        // scalar kernel, which the MixedTypeKernel resolves through EmitScalarOperation ->
        // GetLogAddNextMethod -> NDLogAddExpMath (hypot -> NDHypotMath, heaviside -> NDHeavisideMath).
        //
        // Two members carry a C# SIMD fast path for the contiguous / scalar-broadcast float32/float64
        // cases (NumPy has no SIMD for either, so these beat it several-fold, bit-identically): hypot
        // (correctly-rounded Borges kernel) and heaviside (branchless compare+select step). heaviside is
        // the one NON-commutative member here — x1 selects the branch and x2 is only the x1==0 fill — so
        // its fast path keeps the operands in order (unlike hypot's, which may swap a scalar operand).
        // =====================================================================

        public override NDArray LogAddExp(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null)
            => ExecuteFloatTierBinary(x1, x2, BinaryOp.LogAddExp, dtype?.GetTypeCode(), @out, where);

        public override NDArray LogAddExp2(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null)
            => ExecuteFloatTierBinary(x1, x2, BinaryOp.LogAddExp2, dtype?.GetTypeCode(), @out, where);

        public override NDArray NextAfter(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null)
            => ExecuteFloatTierBinary(x1, x2, BinaryOp.NextAfter, dtype?.GetTypeCode(), @out, where);

        public override NDArray CopySign(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null)
            => ExecuteFloatTierBinary(x1, x2, BinaryOp.CopySign, dtype?.GetTypeCode(), @out, where);

        public override NDArray Hypot(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null)
            => ExecuteFloatTierBinary(x1, x2, BinaryOp.Hypot, dtype?.GetTypeCode(), @out, where);

        public override NDArray Heaviside(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null)
            => ExecuteFloatTierBinary(x1, x2, BinaryOp.Heaviside, dtype?.GetTypeCode(), @out, where);

        // Valid loop dtypes for these ufuncs: the float family (NumPy ee/ff/dd + NumSharp's decimal
        // extension standing in for gg). Everything else -> NumPy's "No loop matching" TypeError.
        private static bool IsFloatTierDtype(NPTypeCode t) =>
            t == NPTypeCode.Half || t == NPTypeCode.Single || t == NPTypeCode.Double || t == NPTypeCode.Decimal;

        private NDArray ExecuteFloatTierBinary(NDArray x1, NDArray x2, BinaryOp op,
            NPTypeCode? typeCode, NDArray @out, NDArray where)
        {
            // NumPy validation order: where parse -> input coercion -> loop resolution -> out.
            ValidateWhereMask(where);

            // Complex has no loop (float-only ufuncs). NumPy: TypeError "ufunc '<name>' not supported
            // for the input types ... casting rule ''safe''" (probed 2.4.2).
            if (x1.GetTypeCode == NPTypeCode.Complex || x2.GetTypeCode == NPTypeCode.Complex)
                throw new IncorrectTypeException(
                    $"ufunc '{UfuncName(op)}' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to the casting rule ''safe''");

            // Only float-family dtype= requests select a loop; int/bool/complex/char raise the no-loop
            // error (probed verbatim, incl. dtype=complex128).
            if (typeCode.HasValue && !IsFloatTierDtype(typeCode.Value))
                throw new IncorrectTypeException(
                    $"No loop matching the specified signature and casting was found for ufunc {UfuncName(op)}");

            // out=/where= ride the shared binary Into-path (same compiled kernels).
            if (@out is not null || where is not null)
            {
                var loopType = typeCode ?? PromoteATan2Binary(x1, x2);
                return ExecuteBinaryUfuncInto(x1, x2, op,
                    x1.GetTypeCode, x2.GetTypeCode, loopType, @out, where);
            }

            if (x1.size == 0)
                return x1.Clone();
            if (x2.size == 0)
                return x2.Clone();

            return ExecuteFloatTierBinaryPlain(x1, x2, op, typeCode);
        }

        /// <summary>
        /// The Direct MixedTypeKernel route (broadcast, allocate, path-classify, dispatch), shared by
        /// logaddexp / logaddexp2 / nextafter. Mirrors <see cref="ExecuteATan2Op"/> but parameterized
        /// by op — the two differ only in the BinaryOp carried into the kernel key and the scalar fold.
        /// </summary>
        private unsafe NDArray ExecuteFloatTierBinaryPlain(NDArray x1, NDArray x2, BinaryOp op, NPTypeCode? typeCode)
        {
            var t1 = x1.GetTypeCode;
            var t2 = x2.GetTypeCode;
            NPTypeCode resultType = typeCode ?? PromoteATan2Binary(x1, x2);

            if (x1.Shape.IsScalar && x2.Shape.IsScalar)
                return ExecuteFloatTierScalarScalar(x1, x2, op, t1, t2, resultType);

            var (s1, s2) = Broadcast(x1.Shape, x2.Shape);
            var resultShape = s1.Clean();
            var result = new NDArray(resultType, resultShape, false);

            ExecutionPath path;
            fixed (long* aStrides = s1.strides)
            fixed (long* bStrides = s2.strides)
            fixed (long* shape = resultShape.dimensions)
            {
                path = ClassifyATan2Path(aStrides, bStrides, shape, resultShape.NDim);
            }

            // np.hypot has no SIMD in NumPy (a scalar BINARY_LOOP), so the correctly-rounded Borges kernel
            // vectorizes cleanly and beats it several-fold. Take the C# SIMD driver for the contiguous /
            // scalar-broadcast, same-dtype (no per-element cast) float32/float64 cases — bit-identical to
            // the IL scalar kernel; every other layout/dtype falls through to it. (arctan2/logaddexp/… have
            // no such fast path: they route through NDIter/scalar libm and stay on the IL kernel.)
            if (op == BinaryOp.Hypot && NDHypotMath.SimdAvailable
                && (resultType == NPTypeCode.Single || resultType == NPTypeCode.Double)
                && t1 == resultType && t2 == resultType
                && TryExecuteHypotSimd(x1, x2, result, s1, s2, path, resultType))
                return result;

            // np.heaviside is likewise a scalar BINARY_LOOP in NumPy; its step is pure compare+select, so
            // the branchless SIMD driver (NDHeavisideMath) is bit-identical to the IL scalar kernel and
            // beats NumPy. Same gate shape as hypot — contiguous / scalar-broadcast, same-dtype (no
            // per-element cast), float32/float64 — every other layout/dtype falls through to the IL kernel.
            if (op == BinaryOp.Heaviside && NDHeavisideMath.SimdAvailable
                && (resultType == NPTypeCode.Single || resultType == NPTypeCode.Double)
                && t1 == resultType && t2 == resultType
                && TryExecuteHeavisideSimd(x1, x2, result, s1, s2, path, resultType))
                return result;

            var key = new MixedTypeKernelKey(t1, t2, resultType, op, path);
            var kernel = DirectILKernelGenerator.GetMixedTypeKernel(key);
            if (kernel == null)
                throw new NotSupportedException(
                    $"IL kernel not available for {UfuncName(op)}({t1}, {t2}) -> {resultType}.");

            ExecuteATan2Kernel(kernel, x1, x2, result, s1, s2);
            return result;
        }

        /// <summary>
        /// The SIMD fast path for <c>np.hypot</c> (see the call site). Handles the contiguous
        /// (<see cref="ExecutionPath.SimdFull"/>) and scalar-broadcast
        /// (<see cref="ExecutionPath.SimdScalarLeft"/>/<see cref="ExecutionPath.SimdScalarRight"/>) cases;
        /// returns false for strided/chunked/general layouts so they take the IL scalar kernel. hypot is
        /// commutative, so a scalar-broadcast operand becomes the second argument regardless of side.
        /// Logical-start pointers follow the house rule <c>Address + Shape.offset·itemsize</c>.
        /// </summary>
        private static unsafe bool TryExecuteHypotSimd(NDArray x1, NDArray x2, NDArray result,
            Shape s1, Shape s2, ExecutionPath path, NPTypeCode resultType)
        {
            long n = result.size;
            int elem = result.dtypesize;
            bool f64 = resultType == NPTypeCode.Double;

            switch (path)
            {
                case ExecutionPath.SimdFull:
                {
                    byte* p1 = (byte*)x1.Address + s1.offset * elem;
                    byte* p2 = (byte*)x2.Address + s2.offset * elem;
                    if (f64) NDHypotMath.HypotContiguousF64((double*)p1, (double*)p2, (double*)result.Address, n);
                    else NDHypotMath.HypotContiguousF32((float*)p1, (float*)p2, (float*)result.Address, n);
                    return true;
                }
                case ExecutionPath.SimdScalarRight:   // x2 scalar-broadcast, x1 contiguous
                {
                    byte* p1 = (byte*)x1.Address + s1.offset * elem;
                    double sc = ConvertToDouble(x2, x2.GetTypeCode);
                    if (f64) NDHypotMath.HypotScalarF64((double*)p1, sc, (double*)result.Address, n);
                    else NDHypotMath.HypotScalarF32((float*)p1, (float)sc, (float*)result.Address, n);
                    return true;
                }
                case ExecutionPath.SimdScalarLeft:    // x1 scalar-broadcast, x2 contiguous
                {
                    byte* p2 = (byte*)x2.Address + s2.offset * elem;
                    double sc = ConvertToDouble(x1, x1.GetTypeCode);
                    if (f64) NDHypotMath.HypotScalarF64((double*)p2, sc, (double*)result.Address, n);
                    else NDHypotMath.HypotScalarF32((float*)p2, (float)sc, (float*)result.Address, n);
                    return true;
                }
                default:
                    return false;
            }
        }

        /// <summary>
        /// The SIMD fast path for <c>np.heaviside</c> (see the call site). Handles the contiguous
        /// (<see cref="ExecutionPath.SimdFull"/>) and scalar-broadcast
        /// (<see cref="ExecutionPath.SimdScalarLeft"/>/<see cref="ExecutionPath.SimdScalarRight"/>) cases;
        /// returns false for strided/chunked/general layouts so they take the IL scalar kernel.
        /// <para>Unlike hypot, heaviside is NOT commutative: <paramref name="x1"/> is the step argument
        /// (selects the branch) and <paramref name="x2"/> is the <c>x1 == 0</c> fill. So the two
        /// scalar-broadcast paths call DIFFERENT drivers — <c>SimdScalarRight</c> (x2 scalar) is the common
        /// <c>np.heaviside(arr, 0.5)</c> shape (fill broadcast), while <c>SimdScalarLeft</c> (x1 scalar)
        /// collapses to a fill-or-copy since one step argument picks a single branch for the whole array.</para>
        /// Logical-start pointers follow the house rule <c>Address + Shape.offset·itemsize</c>.
        /// </summary>
        /// <param name="x1">The step-argument operand.</param>
        /// <param name="x2">The <c>x1 == 0</c> fill operand.</param>
        /// <param name="result">The pre-allocated output (contiguous, loop dtype).</param>
        /// <param name="s1">Broadcast shape of <paramref name="x1"/> against the result.</param>
        /// <param name="s2">Broadcast shape of <paramref name="x2"/> against the result.</param>
        /// <param name="path">The classified execution path (only the three SIMD-eligible ones are served).</param>
        /// <param name="resultType">The loop dtype (Single or Double; the caller gates this).</param>
        /// <returns>True if the SIMD driver handled the layout; false to fall through to the IL kernel.</returns>
        private static unsafe bool TryExecuteHeavisideSimd(NDArray x1, NDArray x2, NDArray result,
            Shape s1, Shape s2, ExecutionPath path, NPTypeCode resultType)
        {
            long n = result.size;
            int elem = result.dtypesize;
            bool f64 = resultType == NPTypeCode.Double;

            switch (path)
            {
                case ExecutionPath.SimdFull:   // both operands contiguous
                {
                    byte* p1 = (byte*)x1.Address + s1.offset * elem;
                    byte* p2 = (byte*)x2.Address + s2.offset * elem;
                    if (f64) NDHeavisideMath.HeavisideContiguousF64((double*)p1, (double*)p2, (double*)result.Address, n);
                    else NDHeavisideMath.HeavisideContiguousF32((float*)p1, (float*)p2, (float*)result.Address, n);
                    return true;
                }
                case ExecutionPath.SimdScalarRight:   // x2 (fill) scalar-broadcast, x1 (step) contiguous
                {
                    byte* p1 = (byte*)x1.Address + s1.offset * elem;
                    double h0 = ConvertToDouble(x2, x2.GetTypeCode);
                    if (f64) NDHeavisideMath.HeavisideH0ScalarF64((double*)p1, h0, (double*)result.Address, n);
                    else NDHeavisideMath.HeavisideH0ScalarF32((float*)p1, (float)h0, (float*)result.Address, n);
                    return true;
                }
                case ExecutionPath.SimdScalarLeft:    // x1 (step) scalar-broadcast, x2 (fill) contiguous
                {
                    byte* p2 = (byte*)x2.Address + s2.offset * elem;
                    double xv = ConvertToDouble(x1, x1.GetTypeCode);
                    if (f64) NDHeavisideMath.HeavisideXScalarF64(xv, (double*)p2, (double*)result.Address, n);
                    else NDHeavisideMath.HeavisideXScalarF32((float)xv, (float*)p2, (float*)result.Address, n);
                    return true;
                }
                default:
                    return false;
            }
        }

        /// <summary>Fold two 0-d operands at the loop dtype (scalar-if-both-scalar, NumPy parity).</summary>
        private static NDArray ExecuteFloatTierScalarScalar(NDArray x1, NDArray x2, BinaryOp op,
            NPTypeCode t1, NPTypeCode t2, NPTypeCode resultType)
        {
            if (resultType == NPTypeCode.Decimal)
            {
                decimal a = ConvertToDecimal(x1, t1), b = ConvertToDecimal(x2, t2);
                decimal r = op switch
                {
                    BinaryOp.LogAddExp => NDLogAddExpMath.LogAddExpDecimal(a, b),
                    BinaryOp.LogAddExp2 => NDLogAddExpMath.LogAddExp2Decimal(a, b),
                    BinaryOp.NextAfter => NDLogAddExpMath.NextAfterDecimal(a, b),
                    BinaryOp.Hypot => NDHypotMath.HypotDecimal(a, b),
                    BinaryOp.Heaviside => NDHeavisideMath.HeavisideDecimal(a, b),
                    _ => NDLogAddExpMath.CopySignDecimal(a, b),
                };
                return NDArray.Scalar(r);
            }

            double xd = ConvertToDouble(x1, t1), yd = ConvertToDouble(x2, t2);
            switch (resultType)
            {
                case NPTypeCode.Half:
                {
                    float xf = (float)xd, yf = (float)yd;
                    Half r = op switch
                    {
                        BinaryOp.LogAddExp => NDLogAddExpMath.LogAddExpHalf((Half)xf, (Half)yf),
                        BinaryOp.LogAddExp2 => NDLogAddExpMath.LogAddExp2Half((Half)xf, (Half)yf),
                        BinaryOp.NextAfter => NDLogAddExpMath.NextAfterHalf((Half)xf, (Half)yf),
                        BinaryOp.Hypot => NDHypotMath.HypotHalf((Half)xf, (Half)yf),
                        // heaviside's f16 loop is astype e->f: (Half)xf/(Half)yf ARE the widen-to-float32-then
                        // -narrow-back operands NumPy uses, so the x1==0 fill h0 keeps its bits (NaN sign / -0.0
                        // survive the round-trip) and a NaN x1 yields the canonical half NaN — matching NumPy.
                        BinaryOp.Heaviside => NDHeavisideMath.HeavisideHalf((Half)xf, (Half)yf),
                        _ => NDLogAddExpMath.CopySignHalf((Half)xf, (Half)yf),
                    };
                    return NDArray.Scalar(r);
                }
                case NPTypeCode.Single:
                {
                    float xf = (float)xd, yf = (float)yd;
                    float r = op switch
                    {
                        BinaryOp.LogAddExp => NDLogAddExpMath.LogAddExpF(xf, yf),
                        BinaryOp.LogAddExp2 => NDLogAddExpMath.LogAddExp2F(xf, yf),
                        BinaryOp.NextAfter => NDLogAddExpMath.NextAfterF(xf, yf),
                        BinaryOp.Hypot => NDHypotMath.HypotF(xf, yf),
                        BinaryOp.Heaviside => NDHeavisideMath.HeavisideF(xf, yf),
                        _ => NDLogAddExpMath.CopySignF(xf, yf),
                    };
                    return NDArray.Scalar(r);
                }
                default:
                {
                    double r = op switch
                    {
                        BinaryOp.LogAddExp => NDLogAddExpMath.LogAddExp(xd, yd),
                        BinaryOp.LogAddExp2 => NDLogAddExpMath.LogAddExp2(xd, yd),
                        BinaryOp.NextAfter => NDLogAddExpMath.NextAfter(xd, yd),
                        BinaryOp.Hypot => NDHypotMath.Hypot(xd, yd),
                        BinaryOp.Heaviside => NDHeavisideMath.Heaviside(xd, yd),
                        _ => NDLogAddExpMath.CopySign(xd, yd),
                    };
                    return NDArray.Scalar(r);
                }
            }
        }
    }
}
