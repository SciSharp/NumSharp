using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =====================================================================
        // np.logaddexp / np.logaddexp2 / np.nextafter
        //
        // Float-tier binary ufuncs whose loop signatures (ee->e, ff->f, dd->d, gg->g) and dtype
        // promotion are IDENTICAL to arctan2, so the promotion / path-classification / kernel-dispatch
        // machinery is shared with Default.ATan2.cs (PromoteATan2Binary, ClassifyATan2Path,
        // ExecuteATan2Kernel, ConvertToDouble/ConvertToDecimal). The only op-specific piece is the
        // scalar kernel, which the MixedTypeKernel resolves through EmitScalarOperation ->
        // ResolveLogAddNextHelper -> NDLogAddExpMath.
        // =====================================================================

        public override NDArray LogAddExp(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null)
            => ExecuteFloatTierBinary(x1, x2, BinaryOp.LogAddExp, dtype?.GetTypeCode(), @out, where);

        public override NDArray LogAddExp2(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null)
            => ExecuteFloatTierBinary(x1, x2, BinaryOp.LogAddExp2, dtype?.GetTypeCode(), @out, where);

        public override NDArray NextAfter(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null)
            => ExecuteFloatTierBinary(x1, x2, BinaryOp.NextAfter, dtype?.GetTypeCode(), @out, where);

        public override NDArray CopySign(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null)
            => ExecuteFloatTierBinary(x1, x2, BinaryOp.CopySign, dtype?.GetTypeCode(), @out, where);

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
                var loopType = typeCode ?? PromoteATan2Binary(x1.GetTypeCode, x2.GetTypeCode);
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
            NPTypeCode resultType = typeCode ?? PromoteATan2Binary(t1, t2);

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

            var key = new MixedTypeKernelKey(t1, t2, resultType, op, path);
            var kernel = DirectILKernelGenerator.GetMixedTypeKernel(key);
            if (kernel == null)
                throw new NotSupportedException(
                    $"IL kernel not available for {UfuncName(op)}({t1}, {t2}) -> {resultType}.");

            ExecuteATan2Kernel(kernel, x1, x2, result, s1, s2);
            return result;
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
                        _ => NDLogAddExpMath.CopySign(xd, yd),
                    };
                    return NDArray.Scalar(r);
                }
            }
        }
    }
}
