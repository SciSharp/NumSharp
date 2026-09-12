using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// np.divmod — the fused two-output ufunc <c>(floor_divide(lhs, rhs), remainder(lhs, rhs))</c>,
        /// element-wise. Item1 is the floored quotient, Item2 the floored remainder (divisor's sign).
        ///
        /// Two routes, both bit-identical to NumPy 2.4.2:
        /// <list type="bullet">
        /// <item>When <paramref name="out"/> (a 2-tuple) or <paramref name="where"/> is supplied, or an
        /// explicit <paramref name="dtype"/> loop is requested, compose the two already-validated
        /// single-output ufuncs — <see cref="FloorDivide"/> + <see cref="Mod"/> — which carry the full
        /// out=/where=/dtype= machinery. divmod(a,b) IS (a // b, a % b), so this is exact.</item>
        /// <item>Otherwise run the FUSED single-pass IL kernel (<see cref="DirectILKernelGenerator.GetDivModKernel"/>),
        /// which computes both outputs from one <see cref="Utilities.NDDivision"/> Divmod* call per
        /// element over contiguous, promoted operands — measurably faster than composing (no second
        /// pass, one idiv for integers).</item>
        /// </list>
        /// Complex raises (NumPy has no complex divmod loop; matches <see cref="Mod"/>'s refusal).
        /// </summary>
        public override (NDArray Quotient, NDArray Remainder) DivMod(
            NDArray lhs, NDArray rhs, DType dtype = null,
            (NDArray Quotient, NDArray Remainder) @out = default, NDArray where = null)
        {
            // Promotion mirrors ExecuteBinaryOp's Mod/FloorDivide path: result_type, then the
            // no-bool-loop bump (True divmod True -> int8). Complex is unsupported (NumPy TypeError).
            var resultType = np._FindCommonType(lhs, rhs);
            if (resultType == NPTypeCode.Complex ||
                lhs.GetTypeCode == NPTypeCode.Complex || rhs.GetTypeCode == NPTypeCode.Complex)
                throw new NotSupportedException("Operation DivMod not supported for Complex");

            bool hasOut = @out.Quotient is not null || @out.Remainder is not null;

            // Compose route — reuses the fully-validated FloorDivide/Mod (out=/where=/dtype=,
            // promotion, error taxonomy). Used whenever out=/where=/dtype= is present. Values are
            // identical to the fused kernel (divmod == (a // b, a % b)).
            if (hasOut || where is not null || dtype is not null)
            {
                var qOut = FloorDivide(lhs, rhs, dtype, @out.Quotient, where);
                var rOut = Mod(lhs, rhs, dtype, @out.Remainder, where);
                return (qOut, rOut);
            }

            if (resultType == NPTypeCode.Boolean)
                resultType = NPTypeCode.SByte;

            // Broadcast to the common result shape.
            var (leftShape, rightShape) = Broadcast(lhs.Shape, rhs.Shape);
            var cleanShape = leftShape.Clean();

            // Scalars (0-d) and empties: the fused contiguous kernel wants ndim >= 1 with elements.
            // Route the rare 0-d / empty cases through the compose path, which handles them exactly.
            var kernel = cleanShape.NDim >= 1 && cleanShape.size > 0
                ? DirectILKernelGenerator.GetDivModKernel(resultType)
                : null;
            if (kernel is null)
            {
                var qOut = FloorDivide(lhs, rhs);
                var rOut = Mod(lhs, rhs);
                return (qOut, rOut);
            }

            // Materialize both operands as C-contiguous arrays of resultType (reads through any
            // broadcast/strided/mixed-dtype layout). ascontiguousarray is a no-op view when the
            // operand is already C-contiguous at resultType — so the common same-dtype contiguous
            // case pays no copy. (np.modf uses the same materialize-then-fused-kernel pattern.)
            var aC = MaterializeContiguous(lhs, cleanShape, resultType);
            var bC = MaterializeContiguous(rhs, cleanShape, resultType);

            var q = new NDArray(resultType, cleanShape, fillZeros: false);
            var r = new NDArray(resultType, cleanShape, fillZeros: false);

            unsafe
            {
                kernel((void*)aC.Address, (void*)bC.Address, (void*)q.Address, (void*)r.Address, cleanShape.size);
            }

            // Reclaim the materialized operands. MaterializeContiguous returns the ORIGINAL lhs/rhs
            // (passthrough, no copy) when they are already C-contiguous resultType arrays; otherwise a
            // fresh broadcast view or ascontiguousarray copy. Dispose only the ones we minted (a fresh
            // view/copy) — never the caller's lhs/rhs — so the pooled buffer returns promptly instead
            // of falling to the finalizer (the zero-leak UndisposedIntermediate gate).
            if (!ReferenceEquals(aC, lhs)) aC.Dispose();
            if (!ReferenceEquals(bC, rhs)) bC.Dispose();

            // NumPy layout preservation: when every non-scalar operand is strictly F-contiguous the
            // outputs are F-contiguous too (mirrors ExecuteBinaryOp / np.modf). Values are already
            // correct C-order; copy('F') relabels them column-major.
            if (AreAllOperandsStrictFContig(lhs, rhs, cleanShape))
            {
                var qf = q.copy('F'); q.Dispose();
                var rf = r.copy('F'); r.Dispose();
                return (qf, rf);
            }

            return (q, r);
        }

        /// <summary>
        /// Return <paramref name="x"/> broadcast to <paramref name="shape"/> as a C-contiguous array of
        /// <paramref name="rt"/>. A no-op passthrough (no copy) when <paramref name="x"/> already IS a
        /// C-contiguous, offset-0, non-broadcast array of <paramref name="rt"/> with that shape.
        /// </summary>
        private NDArray MaterializeContiguous(NDArray x, Shape shape, NPTypeCode rt)
        {
            var v = x.Shape.Equals(shape) && !x.Shape.IsBroadcasted ? x : np.broadcast_to(x, shape);
            if (v.GetTypeCode == rt && v.Shape.IsContiguous && v.Shape.offset == 0 && !v.Shape.IsBroadcasted)
                return v;
            // asarray(order='C') materializes a C-contiguous resultType copy through any layout,
            // preserving the (ndim>=1) shape — unlike ascontiguousarray it does not reshape 0-d,
            // but this path only runs for ndim>=1 (0-d routes to the compose fallback above).
            return np.ascontiguousarray(v, rt);
        }
    }
}
