using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Invert(NDArray nd, Type dtype) => Invert(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise bitwise NOT using IL-generated kernels.
        /// For integers: computes ~x (ones complement).
        /// For booleans: computes logical NOT (NumPy behavior).
        /// </summary>
        public override NDArray Invert(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy validation order (probed 2.4.2): where parse -> loop
            // resolution -> input cast -> out cast -> shape.
            ValidateWhereMask(where);

            var inputType = nd.typecode;
            var loopType = typeCode ?? inputType;
            bool inputIntLike = inputType == NPTypeCode.Boolean || NDExprTypeRules.IsIntegerKind(inputType);
            bool loopIntLike = loopType == NPTypeCode.Boolean || NDExprTypeRules.IsIntegerKind(loopType);

            if (!loopIntLike)
            {
                // invert has bool+integer loops only. Which error depends on
                // what broke it (both probed verbatim): an int-like INPUT with
                // a float dtype= has no such loop signature; a float INPUT
                // without dtype= cannot coerce at all.
                if (typeCode.HasValue && inputIntLike)
                    throw new IncorrectTypeException(
                        "No loop matching the specified signature and casting was found for ufunc invert");
                throw new TypeError(
                    "ufunc 'invert' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to the casting rule ''safe''");
            }

            // dtype= selects the loop; the input must reach it same_kind
            // (probed: invert(f8, dtype=i4) raises the INPUT cast error;
            // invert(bool, dtype=i4) runs the int32 bitwise loop -> [-2, -1]).
            if (typeCode.HasValue)
                ValidateUnaryInputCast(inputType, typeCode.Value, "invert");

            // NumPy treats boolean invert as logical NOT, not bitwise NOT
            // ~True = False, ~False = True (not ~1 = 0xFE)
            if (loopType == NPTypeCode.Boolean)
            {
                return ExecuteUnaryOp(nd, UnaryOp.LogicalNot, NPTypeCode.Boolean, @out, where);
            }

            // For integer types: use bitwise NOT (~x)
            return ExecuteUnaryOp(nd, UnaryOp.BitwiseNot, loopType, @out, where);
        }
    }
}
