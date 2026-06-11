using System;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    /// ufunc <c>out=</c> / <c>where=</c> parameter support (roadmap Wave 2.1).
    ///
    /// Semantics verified against NumPy 2.4.2 (probed, see the Wave 2.1 tests):
    ///   • <c>out</c> participates in broadcasting but may not itself be
    ///     stretched: inputs broadcast UP to a bigger out (add((4,),(4,),
    ///     out=(2,4)) repeats rows); a smaller/mismatched out raises.
    ///   • The loop dtype is resolved from the INPUTS (NEP50); out only has to
    ///     be reachable by a same_kind cast from that result dtype:
    ///     add(f64,f64,out=i32) → UFuncTypeError-equivalent; out=f32/i16 fine.
    ///   • The same object passed as <c>out</c> is returned (reference identity).
    ///   • <c>where</c> must be bool (NumPy casts the mask with the 'safe' rule:
    ///     only bool→bool passes); it broadcasts over the operands AND
    ///     participates in the output shape; out slots where the mask is False
    ///     keep their prior contents (or stay uninitialized for a fresh result —
    ///     NumPy warns 'where' without 'out'; values are unobservable garbage).
    ///   • out aliasing an input is well-defined: COPY_IF_OVERLAP forces a
    ///     write-back temporary exactly like NumPy's ufunc layer.
    ///
    /// Execution: the masked inner loop mirrors NumPy's ufunc machinery
    /// (ufunc_object.c:2190-2226 — wheremask appended as op[nop], outputs
    /// flagged NPY_ITER_WRITEMASKED, trivial loop disabled): the mask rides the
    /// iterator as a trailing ARRAYMASK operand and <see cref="NpyIterRef.ForEach"/>
    /// decomposes each inner chunk into mask-true runs, invoking the unmasked
    /// kernel per run. A dtype-mismatched out additionally engages the
    /// Wave-4 windowed buffer machinery (kernel writes the loop dtype into the
    /// out operand's buffer; the flush casts — and, under WRITEMASKED, masks).
    /// </summary>
    public partial class DefaultEngine
    {
        // Call-invariant flag arrays for the out=/where= iterator configs
        // (Wave 2.2) -- allocated once, reused every call.
        private static readonly NpyIterPerOpFlags[] s_ufuncBinaryOutFlags =
        {
            NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.NO_BROADCAST | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
        };

        private static readonly NpyIterPerOpFlags[] s_ufuncBinaryOutMaskedFlags =
        {
            NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.WRITEMASKED | NpyIterPerOpFlags.NO_BROADCAST | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
        };

        private static readonly NpyIterPerOpFlags[] s_ufuncUnaryOutFlags =
        {
            NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.NO_BROADCAST | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
        };

        private static readonly NpyIterPerOpFlags[] s_ufuncUnaryOutMaskedFlags =
        {
            NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.WRITEMASKED | NpyIterPerOpFlags.NO_BROADCAST | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.ARRAYMASK,
        };

        // =====================================================================
        // NumPy ufunc names (error-text parity). NumPy 2.4.2: np.mod is the
        // 'remainder' ufunc, np.divide/np.true_divide are 'divide',
        // np.abs is 'absolute', np.negative is 'negative'.
        // =====================================================================

        private static string UfuncName(BinaryOp op) => op switch
        {
            BinaryOp.Add => "add",
            BinaryOp.Subtract => "subtract",
            BinaryOp.Multiply => "multiply",
            BinaryOp.Divide => "divide",
            BinaryOp.Mod => "remainder",
            BinaryOp.Power => "power",
            BinaryOp.FloorDivide => "floor_divide",
            BinaryOp.BitwiseAnd => "bitwise_and",
            BinaryOp.BitwiseOr => "bitwise_or",
            BinaryOp.BitwiseXor => "bitwise_xor",
            BinaryOp.ATan2 => "arctan2",
            _ => op.ToString().ToLowerInvariant(),
        };

        private static string UfuncName(ComparisonOp op) => op switch
        {
            ComparisonOp.Equal => "equal",
            ComparisonOp.NotEqual => "not_equal",
            ComparisonOp.Less => "less",
            ComparisonOp.LessEqual => "less_equal",
            ComparisonOp.Greater => "greater",
            ComparisonOp.GreaterEqual => "greater_equal",
            _ => op.ToString().ToLowerInvariant(),
        };

        private static string UfuncName(UnaryOp op) => op switch
        {
            UnaryOp.Sqrt => "sqrt",
            UnaryOp.Abs => "absolute",
            UnaryOp.Negate => "negative",
            UnaryOp.Exp => "exp",
            UnaryOp.Log => "log",
            UnaryOp.Sin => "sin",
            UnaryOp.Cos => "cos",
            UnaryOp.Tan => "tan",
            UnaryOp.Square => "square",
            UnaryOp.Round => "rint",
            UnaryOp.Truncate => "trunc",
            UnaryOp.ASin => "arcsin",
            UnaryOp.ACos => "arccos",
            UnaryOp.ATan => "arctan",
            UnaryOp.BitwiseNot => "invert",
            UnaryOp.LogicalNot => "logical_not",
            UnaryOp.Positive => "positive",
            _ => op.ToString().ToLowerInvariant(),
        };

        /// <summary>
        /// NumPy tuple repr for shapes in ufunc error texts: () / (4,) / (2,4).
        /// </summary>
        private static string NumPyShapeRepr(Shape s)
        {
            var dims = s.dimensions;
            if (dims.Length == 0) return "()";
            if (dims.Length == 1) return $"({dims[0]},)";
            return $"({string.Join(",", dims)})";
        }

        /// <summary>
        /// NumPy requires the where mask to be exactly bool — its converter
        /// casts with the 'safe' rule, and only bool→bool passes
        /// (_wheremask_converter, ufunc_object.c:580). Error text verbatim.
        /// </summary>
        private static void ValidateWhereMask(NDArray? where)
        {
            if (where is null || where.typecode == NPTypeCode.Boolean)
                return;
            throw new ArgumentException(
                $"Cannot cast array data from dtype('{where.typecode.AsNumpyDtypeName()}') " +
                "to dtype('bool') according to the rule 'safe'");
        }

        /// <summary>
        /// out must be reachable from the loop result dtype by a same_kind cast
        /// (NumPy's default ufunc casting rule). Error text verbatim
        /// (UFuncTypeError in NumPy).
        /// </summary>
        private static void ValidateOutCast(NPTypeCode resultType, NPTypeCode outType, string ufuncName)
        {
            if (resultType == outType)
                return;
            if (NpyIterCasting.CanCast(resultType, outType, NPY_CASTING.NPY_SAME_KIND_CASTING))
                return;
            throw new ArgumentException(
                $"Cannot cast ufunc '{ufuncName}' output from " +
                $"dtype('{resultType.AsNumpyDtypeName()}') to " +
                $"dtype('{outType.AsNumpyDtypeName()}') with casting rule 'same_kind'");
        }

        /// <summary>
        /// An explicit dtype= request makes the loop run in that dtype, so each
        /// input must be reachable from its own dtype by a same_kind cast —
        /// NumPy's loop resolution rule (UFuncTypeError, probed 2.4.2:
        /// negative(f64, dtype=i32) → "Cannot cast ufunc 'negative' input from
        /// dtype('float64') to dtype('int32') with casting rule 'same_kind'").
        /// The unary error names no input index; the binary one does
        /// ("input 0" / "input 1", probed via floor_divide).
        /// </summary>
        private static void ValidateUnaryInputCast(NPTypeCode inputType, NPTypeCode loopType, string ufuncName)
        {
            if (inputType == loopType)
                return;
            if (NpyIterCasting.CanCast(inputType, loopType, NPY_CASTING.NPY_SAME_KIND_CASTING))
                return;
            throw new ArgumentException(
                $"Cannot cast ufunc '{ufuncName}' input from " +
                $"dtype('{inputType.AsNumpyDtypeName()}') to " +
                $"dtype('{loopType.AsNumpyDtypeName()}') with casting rule 'same_kind'");
        }

        /// <inheritdoc cref="ValidateUnaryInputCast"/>
        private static void ValidateBinaryInputCasts(NPTypeCode lhsType, NPTypeCode rhsType, NPTypeCode loopType, string ufuncName)
        {
            if (lhsType != loopType && !NpyIterCasting.CanCast(lhsType, loopType, NPY_CASTING.NPY_SAME_KIND_CASTING))
                throw new ArgumentException(
                    $"Cannot cast ufunc '{ufuncName}' input 0 from " +
                    $"dtype('{lhsType.AsNumpyDtypeName()}') to " +
                    $"dtype('{loopType.AsNumpyDtypeName()}') with casting rule 'same_kind'");
            if (rhsType != loopType && !NpyIterCasting.CanCast(rhsType, loopType, NPY_CASTING.NPY_SAME_KIND_CASTING))
                throw new ArgumentException(
                    $"Cannot cast ufunc '{ufuncName}' input 1 from " +
                    $"dtype('{rhsType.AsNumpyDtypeName()}') to " +
                    $"dtype('{loopType.AsNumpyDtypeName()}') with casting rule 'same_kind'");
        }

        /// <summary>
        /// Resolve the iteration (= output) shape for a ufunc call with
        /// optional out/where, with NumPy's exact error texts:
        ///   • out joins the broadcast: inputs may broadcast UP to out's shape.
        ///   • out itself may never be stretched ("non-broadcastable output
        ///     operand with shape (1,) doesn't match the broadcast shape (4,)").
        ///   • an out incompatible with the inputs raises "operands could not
        ///     be broadcast together with shapes (4,) (4,) (5,) " (NumPy lists
        ///     every operand shape, trailing space included).
        ///   • where broadcasts with everything and, when out is absent,
        ///     participates in the output shape (verified: add((4,),(4,),
        ///     where=(2,4)-mask) returns shape (2,4)).
        /// </summary>
        private static Shape ResolveUfuncIterationShape(
            Shape inputBroadcast, NDArray[] inputs, NDArray? @out, NDArray? where)
        {
            Shape full = inputBroadcast;

            if (@out is not null)
            {
                Shape withOut;
                try
                {
                    withOut = Shape.ResolveReturnShape(full, @out.Shape);
                }
                catch (Exception)
                {
                    string shapes = string.Empty;
                    foreach (var inp in inputs)
                        shapes += NumPyShapeRepr(inp.Shape) + " ";
                    shapes += NumPyShapeRepr(@out.Shape) + " ";
                    throw new ArgumentException(
                        $"operands could not be broadcast together with shapes {shapes}");
                }

                // out can absorb the inputs but may not be stretched itself.
                if (!withOut.Equals(@out.Shape))
                {
                    throw new ArgumentException(
                        $"non-broadcastable output operand with shape {NumPyShapeRepr(@out.Shape)} " +
                        $"doesn't match the broadcast shape {NumPyShapeRepr(withOut)}");
                }

                full = withOut;
            }

            if (where is not null)
            {
                Shape withWhere;
                try
                {
                    withWhere = Shape.ResolveReturnShape(full, where.Shape);
                }
                catch (Exception)
                {
                    string shapes = string.Empty;
                    foreach (var inp in inputs)
                        shapes += NumPyShapeRepr(inp.Shape) + " ";
                    if (@out is not null)
                        shapes += NumPyShapeRepr(@out.Shape) + " ";
                    shapes += NumPyShapeRepr(where.Shape) + " ";
                    throw new ArgumentException(
                        $"operands could not be broadcast together with shapes {shapes}");
                }

                if (@out is not null && !withWhere.Equals(full))
                {
                    // The mask would stretch the provided out — same
                    // non-broadcastable-output rule as above.
                    throw new ArgumentException(
                        $"non-broadcastable output operand with shape {NumPyShapeRepr(@out.Shape)} " +
                        $"doesn't match the broadcast shape {NumPyShapeRepr(withWhere)}");
                }

                full = withWhere;
            }

            return full;
        }

        // =====================================================================
        // Binary ufunc with out/where
        // =====================================================================

        /// <summary>
        /// Run <c>op(lhs, rhs)</c> into <paramref name="@out"/> (or a fresh
        /// uninitialized result when only <paramref name="where"/> was given),
        /// optionally write-masked by <paramref name="where"/>. The kernel
        /// bodies and cache keys are identical to
        /// <see cref="TryExecuteBinaryOpViaNpyIter"/> — the routes share
        /// compiled kernels; only the iterator wiring differs (provided
        /// output operand, optional trailing ARRAYMASK, buffered cast when
        /// out's dtype differs from the loop dtype).
        /// </summary>
        private unsafe NDArray ExecuteBinaryUfuncInto(
            NDArray lhs, NDArray rhs, BinaryOp op,
            NPTypeCode lhsType, NPTypeCode rhsType, NPTypeCode resultType,
            NDArray? @out, NDArray? where)
        {
            ValidateWhereMask(where);

            string name = UfuncName(op);
            if (@out is not null)
                ValidateOutCast(resultType, @out.typecode, name);

            // Inputs broadcast first (their own incompatibility raises the
            // pre-existing broadcast error), then out/where join per NumPy.
            var (leftShape, rightShape) = Broadcast(lhs.Shape, rhs.Shape);
            var iterShape = ResolveUfuncIterationShape(
                leftShape.Clean(), new[] { lhs, rhs }, @out, where);

            // NumPy: 'where' without 'out' leaves unmasked slots uninitialized
            // (it warns; values are unobservable). fillZeros:false matches.
            var target = @out ?? new NDArray(resultType, iterShape.Clean(), false);

            if (target.size == 0)
                return target;

            // Kernel bodies — exactly the Tier-3B route's (shared cache).
            bool sameDtype = lhsType == rhsType && lhsType == resultType;
            bool simdViable = sameDtype
                              && DirectILKernelGenerator.CanUseSimd(resultType)
                              && DirectILKernelGenerator.CanUseSimdForOp(op);

            Action<ILGenerator> scalarBody;
            if (sameDtype)
            {
                scalarBody = il => DirectILKernelGenerator.EmitScalarOperation(il, op, resultType);
            }
            else
            {
                NPTypeCode capLhs = lhsType, capRhs = rhsType, capRes = resultType;
                BinaryOp capOp = op;
                scalarBody = il => EmitMixedScalarBody(il, capLhs, capRhs, capRes, capOp);
            }

            Action<ILGenerator>? vectorBody = simdViable
                ? il => DirectILKernelGenerator.EmitVectorOperation(il, op, resultType)
                : null;

            string cacheKey = $"npy_binop_{op}_{lhsType}_{rhsType}_{resultType}";

            // Iterator config. A dtype-mismatched out becomes a CAST operand:
            // the kernel writes the loop dtype into its buffer and the windowed
            // flush casts to the array (Wave 4 machinery); casting was already
            // validated as same_kind above, so the iterator runs UNSAFE exactly
            // like NumPy (loop dtypes are resolved by then; ufunc_object.c
            // passes the user casting to the iterator the same way).
            bool outNeedsCast = target.typecode != resultType;
            var globalFlags = NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.COPY_IF_OVERLAP;
            var casting = NPY_CASTING.NPY_SAFE_CASTING;
            if (outNeedsCast)
            {
                globalFlags |= NpyIterGlobalFlags.BUFFERED
                             | NpyIterGlobalFlags.GROWINNER
                             | NpyIterGlobalFlags.DELAY_BUFALLOC;
                casting = NPY_CASTING.NPY_UNSAFE_CASTING;
            }

            const NpyIterPerOpFlags Elw = NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;

            if (where is null)
            {
                NPTypeCode[]? opDtypes = outNeedsCast
                    ? new[] { lhsType, rhsType, resultType }
                    : null;

                using var iter = NpyIterRef.MultiNew(
                    3, new[] { lhs, rhs, target },
                    globalFlags, NPY_ORDER.NPY_CORDER, casting,
                    s_ufuncBinaryOutFlags,
                    opDtypes);

                iter.ExecuteElementWiseBinary(lhsType, rhsType, resultType, scalarBody, vectorBody, cacheKey);
            }
            else
            {
                // NumPy ufunc masked execution (ufunc_object.c:2190-2226):
                // wheremask appended as op[nop], outputs WRITEMASKED, the
                // masked inner loop runs the unmasked kernel per mask-true run
                // (ForEach's masked driver).
                NPTypeCode[]? opDtypes = outNeedsCast
                    ? new[] { lhsType, rhsType, resultType, NPTypeCode.Empty }
                    : null;

                using var iter = NpyIterRef.MultiNew(
                    4, new[] { lhs, rhs, target, where },
                    globalFlags, NPY_ORDER.NPY_CORDER, casting,
                    s_ufuncBinaryOutMaskedFlags,
                    opDtypes);

                iter.ExecuteElementWise(
                    new[] { lhsType, rhsType, resultType }, scalarBody, vectorBody, cacheKey);
            }

            return target;
        }

        // =====================================================================
        // Comparison ufunc with out/where
        // =====================================================================

        /// <summary>
        /// Run a comparison ufunc into <paramref name="@out"/> (or a fresh
        /// bool result when only <paramref name="where"/> was given). The loop
        /// dtype is Boolean: the kernel compares at result_type(lhs, rhs)
        /// INSIDE the body (fused per-element converts — the Wave-4 measured
        /// winner for cheap ops) and emits bool; a non-bool out is a CAST
        /// operand handled by the windowed flush. bool casts same_kind to
        /// EVERY numeric dtype (probed: less(f8,f8,out=complex128) works,
        /// True→1), so <see cref="ValidateOutCast"/> is structural here.
        /// The scalar body and npy_cmp_* cache key are shared with
        /// <see cref="TryExecuteComparisonOpViaNpyIter"/> (same compiled
        /// kernels) and the flag arrays are the binary configs (identical
        /// operand layout) — no new statics. No F-layout post-step: NumPy
        /// returns the provided out untouched (reference identity).
        /// </summary>
        private unsafe NDArray ExecuteComparisonUfuncInto(
            NDArray lhs, NDArray rhs, ComparisonOp op,
            NPTypeCode lhsType, NPTypeCode rhsType,
            NDArray? @out, NDArray? where)
        {
            ValidateWhereMask(where);

            string name = UfuncName(op);
            if (@out is not null)
                ValidateOutCast(NPTypeCode.Boolean, @out.typecode, name);

            var (leftShape, _) = Broadcast(lhs.Shape, rhs.Shape);
            var iterShape = ResolveUfuncIterationShape(
                leftShape.Clean(), new[] { lhs, rhs }, @out, where);

            // 'where' without 'out': unmasked slots stay uninitialized
            // (NumPy warns; values are unobservable garbage).
            var target = @out ?? new NDArray(NPTypeCode.Boolean, iterShape.Clean(), false);

            if (target.size == 0)
                return target;

            // Comparison computes at the NumPy common dtype inside the kernel
            // (probed A3: greater(i8 2^53+1, f8 2^53) → False, equal → True —
            // both operands cast to f64 first).
            var comparisonType = lhsType == rhsType
                ? lhsType
                : np._FindCommonScalarType(lhsType, rhsType);

            NPTypeCode capLhs = lhsType, capRhs = rhsType, capCmp = comparisonType;
            ComparisonOp capOp = op;
            Action<ILGenerator> scalarBody = il =>
            {
                if (capLhs == capRhs && capLhs == capCmp)
                {
                    // Same-dtype fast path — no convert pass.
                    DirectILKernelGenerator.EmitComparisonOperation(il, capOp, capCmp);
                }
                else
                {
                    var locRhs = il.DeclareLocal(DirectILKernelGenerator.GetClrType(capRhs));
                    il.Emit(OpCodes.Stloc, locRhs);
                    if (capLhs != capCmp)
                        DirectILKernelGenerator.EmitConvertTo(il, capLhs, capCmp);
                    il.Emit(OpCodes.Ldloc, locRhs);
                    if (capRhs != capCmp)
                        DirectILKernelGenerator.EmitConvertTo(il, capRhs, capCmp);
                    DirectILKernelGenerator.EmitComparisonOperation(il, capOp, capCmp);
                }
            };

            // Vector body intentionally null: bool output breaks the Tier-3B
            // same-dtype invariant (unchanged from the no-out route).
            string cacheKey = $"npy_cmp_{op}_{lhsType}_{rhsType}";

            bool outNeedsCast = target.typecode != NPTypeCode.Boolean;
            var globalFlags = NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.COPY_IF_OVERLAP;
            var casting = NPY_CASTING.NPY_SAFE_CASTING;
            if (outNeedsCast)
            {
                globalFlags |= NpyIterGlobalFlags.BUFFERED
                             | NpyIterGlobalFlags.GROWINNER
                             | NpyIterGlobalFlags.DELAY_BUFALLOC;
                casting = NPY_CASTING.NPY_UNSAFE_CASTING;
            }

            if (where is null)
            {
                NPTypeCode[]? opDtypes = outNeedsCast
                    ? new[] { lhsType, rhsType, NPTypeCode.Boolean }
                    : null;

                using var iter = NpyIterRef.MultiNew(
                    3, new[] { lhs, rhs, target },
                    globalFlags, NPY_ORDER.NPY_CORDER, casting,
                    s_ufuncBinaryOutFlags,
                    opDtypes);

                iter.ExecuteElementWiseBinary(lhsType, rhsType, NPTypeCode.Boolean, scalarBody, null, cacheKey);
            }
            else
            {
                NPTypeCode[]? opDtypes = outNeedsCast
                    ? new[] { lhsType, rhsType, NPTypeCode.Boolean, NPTypeCode.Empty }
                    : null;

                using var iter = NpyIterRef.MultiNew(
                    4, new[] { lhs, rhs, target, where },
                    globalFlags, NPY_ORDER.NPY_CORDER, casting,
                    s_ufuncBinaryOutMaskedFlags,
                    opDtypes);

                iter.ExecuteElementWise(
                    new[] { lhsType, rhsType, NPTypeCode.Boolean }, scalarBody, null, cacheKey);
            }

            return target;
        }

        // =====================================================================
        // Unary ufunc with out/where
        // =====================================================================

        /// <summary>
        /// Run <c>op(nd)</c> into <paramref name="@out"/> (or a fresh
        /// uninitialized result when only <paramref name="where"/> was given),
        /// optionally write-masked. Mirrors
        /// <see cref="TryExecuteUnaryOpViaNpyIter"/>'s body construction —
        /// including the promoting buffered-cast configuration (Wave 4: the
        /// input is cast to the compute dtype in buffer windows and the
        /// same-dtype SIMD body runs over the buffer), which composes with a
        /// provided out: the out operand simply requests the compute dtype too
        /// and the flush casts (and masks) on write-back.
        /// </summary>
        private unsafe NDArray ExecuteUnaryUfuncInto(
            NDArray nd, UnaryOp op,
            NPTypeCode inputType, NPTypeCode outputType,
            NDArray? @out, NDArray? where)
        {
            ValidateWhereMask(where);

            string name = UfuncName(op);
            if (@out is not null)
                ValidateOutCast(outputType, @out.typecode, name);

            var iterShape = ResolveUfuncIterationShape(
                nd.Shape.Clean(), new[] { nd }, @out, where);

            var target = @out ?? new NDArray(outputType, iterShape.Clean(), false);

            if (target.size == 0)
                return target;

            // Body construction — same decision tree as TryExecuteUnaryOpViaNpyIter.
            var key = new UnaryKernelKey(inputType, outputType, op, IsContiguous: true);
            bool simdViable = DirectILKernelGenerator.CanUseUnarySimd(key);

            bool bufferedPromoting = inputType != outputType
                && !IsUnaryPredicateOp(op)
                && !(op == UnaryOp.Abs && inputType == NPTypeCode.Complex)
                && DirectILKernelGenerator.CanUseUnarySimd(
                       new UnaryKernelKey(outputType, outputType, op, IsContiguous: true))
                && DirectILKernelGenerator.TryGetCastKernel(inputType, outputType) != null;

            NPTypeCode capIn = inputType, capOut = outputType;
            UnaryOp capOp = op;
            Action<ILGenerator> scalarBody;
            Action<ILGenerator>? vectorBody;
            string cacheKey;
            if (bufferedPromoting)
            {
                scalarBody = il => DirectILKernelGenerator.EmitUnaryScalarOperation(il, capOp, capOut);
                vectorBody = il => DirectILKernelGenerator.EmitUnaryVectorOperation(il, capOp, capOut);
                cacheKey = $"npy_unop_{op}_{outputType}_{outputType}";
            }
            else
            {
                scalarBody = il =>
                {
                    if (IsUnaryPredicateOp(capOp))
                    {
                        DirectILKernelGenerator.EmitUnaryScalarOperation(il, capOp, capIn);
                    }
                    else if (capOp == UnaryOp.Abs && capIn == NPTypeCode.Complex)
                    {
                        il.EmitCall(OpCodes.Call, s_complexAbs, null);
                        if (capOut != NPTypeCode.Double)
                            DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Double, capOut);
                    }
                    else
                    {
                        if (capIn != capOut)
                            DirectILKernelGenerator.EmitConvertTo(il, capIn, capOut);
                        DirectILKernelGenerator.EmitUnaryScalarOperation(il, capOp, capOut);
                    }
                };
                vectorBody = simdViable
                    ? il => DirectILKernelGenerator.EmitUnaryVectorOperation(il, capOp, capIn)
                    : null;
                cacheKey = $"npy_unop_{op}_{inputType}_{outputType}";
            }

            // Buffering engages when the input promotes (bufferedPromoting) or
            // the out dtype differs from the compute dtype — both are CAST
            // operands handled by the windowed machinery.
            bool outNeedsCast = target.typecode != outputType;
            var globalFlags = NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.COPY_IF_OVERLAP;
            var casting = NPY_CASTING.NPY_SAFE_CASTING;
            NPTypeCode kernelIn = inputType;
            NPTypeCode[]? opDtypes = null;
            if (bufferedPromoting || outNeedsCast)
            {
                globalFlags |= NpyIterGlobalFlags.BUFFERED
                             | NpyIterGlobalFlags.GROWINNER
                             | NpyIterGlobalFlags.DELAY_BUFALLOC;
                casting = NPY_CASTING.NPY_UNSAFE_CASTING;
                NPTypeCode inRequest = bufferedPromoting ? outputType : inputType;
                kernelIn = bufferedPromoting ? outputType : inputType;
                opDtypes = where is null
                    ? new[] { inRequest, outputType }
                    : new[] { inRequest, outputType, NPTypeCode.Empty };
            }

            const NpyIterPerOpFlags Elw = NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;

            if (where is null)
            {
                using var iter = NpyIterRef.MultiNew(
                    2, new[] { nd, target },
                    globalFlags, NPY_ORDER.NPY_CORDER, casting,
                    s_ufuncUnaryOutFlags,
                    opDtypes);

                iter.ExecuteElementWiseUnary(kernelIn, outputType, scalarBody, vectorBody, cacheKey);
            }
            else
            {
                using var iter = NpyIterRef.MultiNew(
                    3, new[] { nd, target, where },
                    globalFlags, NPY_ORDER.NPY_CORDER, casting,
                    s_ufuncUnaryOutMaskedFlags,
                    opDtypes);

                iter.ExecuteElementWise(
                    new[] { kernelIn, outputType }, scalarBody, vectorBody, cacheKey);
            }

            return target;
        }
    }
}
