using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDExpr.Typing.cs — NumPy result_type pass over expression trees (Wave 6.1)
// =============================================================================
//
// Resolves every node of an NDExpr tree to the dtype NumPy 2.x would give the
// equivalent ufunc call, so a fused kernel computes EACH NODE at its own dtype
// and converts at node edges — bit-matching the unfused NumPy sequence
// (e.g. (i4*i4)+f8 wraps the multiply in int32 BEFORE promoting to float64).
//
// Rules below are pinned to NumPy 2.4.2 probes (see NDEvaluateTests):
//   • strong-strong promotion: NEP50. bool < ints < floats < complex; mixed
//     int/float promotes the int to its tier float (i8→f16, i16→f32, i32+→f64)
//     and takes the larger float — so int32+float32 → float64.
//   • weak python-scalar semantics for constants: an int literal adopts the
//     other operand's dtype (i4+2→i4, u1+2→u1, bool+2→i64, OverflowError when
//     it doesn't fit); a float literal forces float but adopts float width
//     (f2+2.5→f2, i4+2.5→f8).
//   • true_divide: int/bool common → float64 (even i8/i8).
//   • power: plain promotion, except bool,bool → int8; negative integer
//     exponent CONSTANTS raise like NumPy ("Integers to negative integer
//     powers are not allowed."). Negative values inside exponent ARRAYS are
//     not detected (NumPy raises per-element at runtime; the fused kernel
//     computes a wrapped/garbage value instead — documented divergence).
//   • arctan2: promotion, then int/bool common → its tier float (i1→f16).
//   • unary math (sqrt/exp/log/trig/…): tier-float promotion
//     (bool/i8/u8→f16, i16/u16→f32, i32+→f64, floats preserved).
//   • comparisons / logical_not / isnan-family → bool.
//   • where: result_type of the branches; the condition is nonzero-tested at
//     its own dtype.
//
// The pass writes node→dtype into a reference-keyed side table consumed by
// the emission methods in NDExpr.cs (ctx.TypeOf). Legacy Compile() passes no
// table, and every node falls back to the single output dtype — emission is
// unchanged for existing Tier-3C users.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Weak-scalar kind of a typing result (NEP50: python int/float literals
    /// adopt the dtype of the array they meet instead of forcing promotion).
    /// </summary>
    internal enum NDExprWeak : byte
    {
        None = 0,
        Int = 1,
        Float = 2,
    }

    /// <summary>
    /// Result of typing one node: either a strong dtype (already recorded in
    /// the node-type table) or a weak scalar awaiting adoption by its parent.
    /// </summary>
    internal readonly struct NDExprTypeInfo
    {
        public readonly NPTypeCode Code;
        public readonly NDExprWeak Weak;

        private NDExprTypeInfo(NPTypeCode code, NDExprWeak weak)
        {
            Code = code;
            Weak = weak;
        }

        public bool IsWeak => Weak != NDExprWeak.None;

        /// <summary>NEP50 default when a weak scalar meets no array: int→int64, float→float64.</summary>
        public NPTypeCode DefaultCode => Weak == NDExprWeak.Int ? NPTypeCode.Int64 : NPTypeCode.Double;

        public static NDExprTypeInfo Strong(NPTypeCode code) => new(code, NDExprWeak.None);
        public static readonly NDExprTypeInfo WeakInt = new(NPTypeCode.Int64, NDExprWeak.Int);
        public static readonly NDExprTypeInfo WeakFloat = new(NPTypeCode.Double, NDExprWeak.Float);
    }

    /// <summary>
    /// NumPy 2.x (NEP50) dtype-promotion rules used by the expression typing
    /// pass. Self-contained over all 15 NPTypeCodes (Decimal/Complex follow
    /// NumSharp's conventions: Complex absorbs everything, Decimal absorbs
    /// every non-complex).
    /// </summary>
    internal static class NDExprTypeRules
    {
        internal static bool IsIntegerKind(NPTypeCode t)
            => t == NPTypeCode.Byte || t == NPTypeCode.SByte ||
               t == NPTypeCode.Int16 || t == NPTypeCode.UInt16 || t == NPTypeCode.Char ||
               t == NPTypeCode.Int32 || t == NPTypeCode.UInt32 ||
               t == NPTypeCode.Int64 || t == NPTypeCode.UInt64;

        internal static bool IsFloatKind(NPTypeCode t)
            => t == NPTypeCode.Half || t == NPTypeCode.Single || t == NPTypeCode.Double;

        internal static bool IsSignedInteger(NPTypeCode t)
            => t == NPTypeCode.SByte || t == NPTypeCode.Int16 ||
               t == NPTypeCode.Int32 || t == NPTypeCode.Int64;

        private static int SizeOf(NPTypeCode t) => DirectILKernelGenerator.GetTypeSize(t);

        /// <summary>
        /// The narrowest float whose range covers the integer dtype — NumPy's
        /// tier used both for unary float-promoting ufuncs and for mixed
        /// int/float binary promotion: bool/i8/u8 → f16, i16/u16 → f32,
        /// i32/u32/i64/u64 → f64.
        /// </summary>
        internal static NPTypeCode FloatTier(NPTypeCode intLike) => intLike switch
        {
            NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte => NPTypeCode.Half,
            NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char => NPTypeCode.Single,
            _ => NPTypeCode.Double,
        };

        /// <summary>
        /// NEP50 array-array promotion over the 15 NumSharp dtypes.
        /// </summary>
        internal static NPTypeCode PromoteStrong(NPTypeCode a, NPTypeCode b)
        {
            if (a == b)
                return a;

            if (a == NPTypeCode.Complex || b == NPTypeCode.Complex)
                return NPTypeCode.Complex;
            if (a == NPTypeCode.Decimal || b == NPTypeCode.Decimal)
                return NPTypeCode.Decimal;

            // Char rides as uint16 for promotion purposes (NumSharp-only dtype).
            if (a == NPTypeCode.Char) a = NPTypeCode.UInt16;
            if (b == NPTypeCode.Char) b = NPTypeCode.UInt16;
            if (a == b)
                return a;

            // bool is the weakest kind: promotes to the other operand.
            if (a == NPTypeCode.Boolean) return b;
            if (b == NPTypeCode.Boolean) return a;

            bool af = IsFloatKind(a), bf = IsFloatKind(b);
            if (af && bf)
                return SizeOf(a) >= SizeOf(b) ? a : b;

            if (af || bf)
            {
                // Mixed int/float: the int promotes to its tier float, then the
                // larger float wins — int32+float32 → float64 (NEP50).
                NPTypeCode f = af ? a : b;
                NPTypeCode tier = FloatTier(af ? b : a);
                return SizeOf(tier) >= SizeOf(f) ? tier : f;
            }

            // int + int
            bool asg = IsSignedInteger(a), bsg = IsSignedInteger(b);
            int sa = SizeOf(a), sb = SizeOf(b);
            if (asg == bsg)
                return sa >= sb ? a : b;

            // Mixed signedness: the signed type wins if strictly larger,
            // otherwise the next signed size up (uint64+signed → float64).
            NPTypeCode signed;
            int ssize, usize;
            if (asg)
            {
                signed = a;
                ssize = sa;
                usize = sb;
            }
            else
            {
                signed = b;
                ssize = sb;
                usize = sa;
            }

            if (ssize > usize)
                return signed;
            return usize switch
            {
                1 => NPTypeCode.Int16,
                2 => NPTypeCode.Int32,
                4 => NPTypeCode.Int64,
                _ => NPTypeCode.Double,
            };
        }

        /// <summary>
        /// Promotion of two typing results, applying NEP50 weak-scalar rules
        /// when either side is a literal: weak int adopts the strong dtype
        /// (bool → int64); weak float forces float kind but adopts float width
        /// (ints/bool → float64, f2 stays f2). Two weak operands resolve to the
        /// NEP50 defaults (a ufunc call materializes them — np.add(2,3) → i64).
        /// </summary>
        internal static NPTypeCode PromoteMixed(in NDExprTypeInfo l, in NDExprTypeInfo r)
        {
            if (!l.IsWeak && !r.IsWeak)
                return PromoteStrong(l.Code, r.Code);

            if (l.IsWeak && r.IsWeak)
                return (l.Weak == NDExprWeak.Int && r.Weak == NDExprWeak.Int)
                    ? NPTypeCode.Int64
                    : NPTypeCode.Double;

            var weak = l.IsWeak ? l : r;
            var strong = l.IsWeak ? r.Code : l.Code;

            if (weak.Weak == NDExprWeak.Int)
                return strong == NPTypeCode.Boolean ? NPTypeCode.Int64 : strong;

            // weak float
            if (IsFloatKind(strong) || strong == NPTypeCode.Decimal || strong == NPTypeCode.Complex)
                return strong;
            return NPTypeCode.Double; // bool / any integer + float literal → f64
        }

        /// <summary>
        /// Output dtype of a float-producing unary ufunc (sqrt/exp/log/trig/…):
        /// the tier float for int/bool inputs, the input dtype for floats,
        /// Decimal/Complex preserved (NumSharp extension).
        /// </summary>
        internal static NPTypeCode UnaryFloatResult(NPTypeCode input)
        {
            if (input == NPTypeCode.Decimal || input == NPTypeCode.Complex || IsFloatKind(input))
                return input;
            return FloatTier(input);
        }

        /// <summary>
        /// NEP50 value check when an integer literal adopts an integer dtype —
        /// NumPy raises OverflowError ("Python integer 300 out of bounds for
        /// uint8") instead of wrapping.
        /// </summary>
        internal static void CheckIntLiteralFits(long value, NPTypeCode adopted)
        {
            (long min, ulong max) = adopted switch
            {
                NPTypeCode.Boolean => (0L, 1UL),
                NPTypeCode.Byte => (0L, (ulong)byte.MaxValue),
                NPTypeCode.SByte => ((long)sbyte.MinValue, (ulong)sbyte.MaxValue),
                NPTypeCode.Int16 => ((long)short.MinValue, (ulong)short.MaxValue),
                NPTypeCode.UInt16 => (0L, (ulong)ushort.MaxValue),
                NPTypeCode.Char => (0L, (ulong)char.MaxValue),
                NPTypeCode.Int32 => ((long)int.MinValue, (ulong)int.MaxValue),
                NPTypeCode.UInt32 => (0L, (ulong)uint.MaxValue),
                NPTypeCode.Int64 => (long.MinValue, (ulong)long.MaxValue),
                NPTypeCode.UInt64 => (0L, ulong.MaxValue),
                _ => (long.MinValue, ulong.MaxValue), // float adoptions never overflow-check
            };

            bool fits = value >= min && (value < 0 || (ulong)value <= max);
            if (!fits)
                throw new OverflowException(
                    $"Python integer {value} out of bounds for {adopted.AsNumpyDtypeName()}");
        }
    }

    public abstract partial class NDExpr
    {
        /// <summary>
        /// Resolve this node's NumPy result_type given the operand dtypes.
        /// Strong results are recorded into <paramref name="nodeTypes"/>;
        /// weak literals return without recording and are adopted by their
        /// parent via <see cref="AdoptWeakType"/>.
        /// </summary>
        internal abstract NDExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NDExpr, NPTypeCode> nodeTypes);

        /// <summary>
        /// Assign the dtype a weak literal adopts (NEP50). Only meaningful for
        /// <see cref="ConstNode"/>; any other node typing weak is a pass bug.
        /// </summary>
        internal virtual void AdoptWeakType(NPTypeCode adopted, Dictionary<NDExpr, NPTypeCode> nodeTypes)
            => throw new InvalidOperationException(
                $"{GetType().Name} cannot be weak-typed — typing pass bug.");

        /// <summary>
        /// Record a child's resolved dtype, adopting it at
        /// <paramref name="adoption"/> when the child is a weak literal.
        /// Returns the dtype the child will leave on the stack.
        /// </summary>
        private protected static NPTypeCode ResolveChild(
            NDExpr child, in NDExprTypeInfo info, NPTypeCode adoption,
            Dictionary<NDExpr, NPTypeCode> nodeTypes)
        {
            if (!info.IsWeak)
                return info.Code;
            child.AdoptWeakType(adoption, nodeTypes);
            return adoption;
        }

        /// <summary>
        /// Run the NumPy typing pass and return the tree's resolved result
        /// dtype plus the per-node dtype table (consumed by typed emission).
        /// </summary>
        internal NPTypeCode ResolveNumPyTypes(
            NPTypeCode[] inputTypes, out Dictionary<NDExpr, NPTypeCode> nodeTypes)
        {
            nodeTypes = new Dictionary<NDExpr, NPTypeCode>(ReferenceEqualityComparer.Instance);
            var rootInfo = InferType(inputTypes, nodeTypes);
            if (rootInfo.IsWeak)
            {
                // Constant-only tree: NEP50 defaults (np.evaluate rejects
                // array-free trees earlier, but raw CompileNumPy callers may
                // type one).
                AdoptWeakType(rootInfo.DefaultCode, nodeTypes);
                return rootInfo.DefaultCode;
            }

            return rootInfo.Code;
        }

        /// <summary>
        /// Compile the tree with NumPy 2.x result_type semantics: every node
        /// computes at its own NEP50-resolved dtype, conversions happen at
        /// node edges, and the kernel reads each operand at its NATIVE dtype
        /// (no iterator-side casting needed). <paramref name="resolvedType"/>
        /// receives the tree's result dtype — the output operand's dtype.
        /// SIMD engages iff the whole tree resolves to one SIMD-capable dtype.
        /// </summary>
        public NDInnerLoopFunc CompileNumPy(
            NPTypeCode[] inputTypes, out NPTypeCode resolvedType, string? cacheKey = null)
        {
            if (inputTypes is null) throw new ArgumentNullException(nameof(inputTypes));

            var resolved = ResolveNumPyTypes(inputTypes, out var nodeTypes);
            resolvedType = resolved;
            int nIn = inputTypes.Length;

            bool homogeneous = true;
            for (int i = 0; i < nIn && homogeneous; i++)
                homogeneous = inputTypes[i] == resolved;
            if (homogeneous)
            {
                foreach (var kv in nodeTypes)
                {
                    if (kv.Value != resolved)
                    {
                        homogeneous = false;
                        break;
                    }
                }
            }

            // homogeneous => every operand and every node type equals `resolved`, so the tree (if
            // it vectorizes at all) does so at that single type. SupportsSimdAt(resolved) refines
            // the structural SupportsSimd with type/runtime capability — e.g. it keeps integer
            // rounding and pre-.NET-9 Round/Truncate on the scalar path instead of emitting a
            // Vector{N} method the BCL has no overload for.
            bool wantSimd = homogeneous && SupportsSimdAt(resolved);

            Action<ILGenerator> scalarBody = il =>
            {
                var scalarLocals = new LocalBuilder[nIn];
                for (int i = nIn - 1; i >= 0; i--)
                {
                    scalarLocals[i] = il.DeclareLocal(DirectILKernelGenerator.GetClrType(inputTypes[i]));
                    il.Emit(OpCodes.Stloc, scalarLocals[i]);
                }
                var ctx = new NDExprCompileContext(inputTypes, resolved, scalarLocals, vectorMode: false, nodeTypes);
                EmitScalar(il, ctx);
            };

            Action<ILGenerator>? vectorBody = null;
            if (wantSimd)
            {
                vectorBody = il =>
                {
                    var vectorLocals = new LocalBuilder[nIn];
                    var vecType = DirectILKernelGenerator.GetVectorType(DirectILKernelGenerator.GetClrType(inputTypes[0]));
                    for (int i = nIn - 1; i >= 0; i--)
                    {
                        vectorLocals[i] = il.DeclareLocal(vecType);
                        il.Emit(OpCodes.Stloc, vectorLocals[i]);
                    }
                    var ctx = new NDExprCompileContext(inputTypes, resolved, vectorLocals, vectorMode: true, nodeTypes);
                    EmitVector(il, ctx);
                };
            }

            var operandTypes = new NPTypeCode[nIn + 1];
            Array.Copy(inputTypes, operandTypes, nIn);
            operandTypes[nIn] = resolved;

            // Distinct cache namespace from legacy Compile — same signature,
            // different emission contract.
            string key = (cacheKey ?? DeriveCacheKey(inputTypes, resolved)) + "|np";
            return DirectILKernelGenerator.CompileInnerLoop(operandTypes, scalarBody, vectorBody, key);
        }
    }

    public sealed partial class InputNode
    {
        internal override NDExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NDExpr, NPTypeCode> nodeTypes)
        {
            if (_index >= inputTypes.Length)
                throw new InvalidOperationException(
                    $"Input({_index}) out of range; typing provided {inputTypes.Length} inputs.");
            nodeTypes[this] = inputTypes[_index];
            return NDExprTypeInfo.Strong(inputTypes[_index]);
        }
    }

    public sealed partial class ConstNode
    {
        internal override NDExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NDExpr, NPTypeCode> nodeTypes)
            => _isIntegerLiteral ? NDExprTypeInfo.WeakInt : NDExprTypeInfo.WeakFloat;

        internal override void AdoptWeakType(NPTypeCode adopted, Dictionary<NDExpr, NPTypeCode> nodeTypes)
        {
            if (_isIntegerLiteral && (NDExprTypeRules.IsIntegerKind(adopted) || adopted == NPTypeCode.Boolean))
                NDExprTypeRules.CheckIntLiteralFits(_valueInt, adopted);
            nodeTypes[this] = adopted;
        }
    }

    public sealed partial class BinaryNode
    {
        internal override NDExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NDExpr, NPTypeCode> nodeTypes)
        {
            var lt = _left.InferType(inputTypes, nodeTypes);
            var rt = _right.InferType(inputTypes, nodeTypes);
            var common = NDExprTypeRules.PromoteMixed(lt, rt);
            bool intish = common == NPTypeCode.Boolean || NDExprTypeRules.IsIntegerKind(common);

            NPTypeCode result = _op switch
            {
                // true_divide: integer/bool inputs always produce float64
                // (NumPy's TrueDivision resolver — even int8/int8 → f64).
                BinaryOp.Divide when intish => NPTypeCode.Double,

                // power/remainder/floor_divide have no bool loop; bool inputs
                // fall to the int8 loop (probed: power(?,?)→i8,
                // remainder(?,?)→i8, floor_divide(?,?)→i8).
                BinaryOp.Power or BinaryOp.Mod or BinaryOp.FloorDivide
                    when common == NPTypeCode.Boolean => NPTypeCode.SByte,

                // arctan2 is float-only: int/bool promote to their tier float
                // (i1→f16, i2→f32, i4+→f64) — unlike divide's flat f64.
                BinaryOp.ATan2 when intish => NDExprTypeRules.FloatTier(common),

                BinaryOp.Subtract when common == NPTypeCode.Boolean
                    => throw new NotSupportedException(
                        "numpy boolean subtract, the `-` operator, is not supported, " +
                        "use the bitwise_xor, the `^` operator, or the logical_xor function instead."),

                BinaryOp.BitwiseAnd or BinaryOp.BitwiseOr or BinaryOp.BitwiseXor
                    when !intish && common != NPTypeCode.Boolean
                    => throw new NotSupportedException(
                        $"ufunc '{BitwiseName(_op)}' not supported for the input types, and the inputs " +
                        "could not be safely coerced to any supported types according to the casting rule ''safe''"),

                _ => common,
            };

            // NumPy raises for negative integer exponents; a literal exponent
            // is checkable at typing time (array exponents are runtime-only —
            // see the class doc divergence note).
            if (_op == BinaryOp.Power && NDExprTypeRules.IsIntegerKind(result) &&
                _right is ConstNode { IsIntegerLiteral: true, IntegerValue: < 0 })
                throw new ArgumentException("Integers to negative integer powers are not allowed.");

            ResolveChild(_left, lt, result, nodeTypes);
            ResolveChild(_right, rt, result, nodeTypes);
            nodeTypes[this] = result;
            return NDExprTypeInfo.Strong(result);
        }

        private static string BitwiseName(BinaryOp op) => op switch
        {
            BinaryOp.BitwiseAnd => "bitwise_and",
            BinaryOp.BitwiseOr => "bitwise_or",
            _ => "bitwise_xor",
        };
    }

    public sealed partial class UnaryNode
    {
        private static bool IsFloatPromoting(UnaryOp op) => op switch
        {
            UnaryOp.Sqrt or UnaryOp.Cbrt or
            UnaryOp.Exp or UnaryOp.Exp2 or UnaryOp.Expm1 or
            UnaryOp.Log or UnaryOp.Log2 or UnaryOp.Log10 or UnaryOp.Log1p or
            UnaryOp.Sin or UnaryOp.Cos or UnaryOp.Tan or
            UnaryOp.Sinh or UnaryOp.Cosh or UnaryOp.Tanh or
            UnaryOp.ASin or UnaryOp.ACos or UnaryOp.ATan or
            UnaryOp.Deg2Rad or UnaryOp.Rad2Deg => true,
            _ => false,
        };

        internal override NDExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NDExpr, NPTypeCode> nodeTypes)
        {
            var ct = _child.InferType(inputTypes, nodeTypes);
            var childType = ct.IsWeak ? ct.DefaultCode : ct.Code;

            NPTypeCode result;
            if (IsFloatPromoting(_op))
            {
                result = NDExprTypeRules.UnaryFloatResult(childType);
            }
            else if (IsPredicateResult(_op) || _op == UnaryOp.LogicalNot)
            {
                result = NPTypeCode.Boolean;
            }
            else if (_op == UnaryOp.BitwiseNot &&
                     !NDExprTypeRules.IsIntegerKind(childType) && childType != NPTypeCode.Boolean)
            {
                // NumPy: invert has no float/complex loop.
                throw new NotSupportedException(
                    "ufunc 'invert' not supported for the input types, and the inputs could not " +
                    "be safely coerced to any supported types according to the casting rule ''safe''");
            }
            else if (childType == NPTypeCode.Boolean)
            {
                // NumPy's bool quirks for dtype-preserving ufuncs:
                result = _op switch
                {
                    UnaryOp.Negate => throw new NotSupportedException(
                        "The numpy boolean negative, the `-` operator, is not supported, " +
                        "use the `~` operator or the logical_not function instead."),
                    UnaryOp.Sign => throw new NotSupportedException(
                        "ufunc 'sign' did not contain a loop with signature matching types " +
                        "<class 'numpy.dtypes.BoolDType'> -> None"),
                    UnaryOp.Square or UnaryOp.Reciprocal => NPTypeCode.SByte,
                    _ => NPTypeCode.Boolean, // abs/floor/ceil/trunc/round/invert preserve bool
                };
            }
            else
            {
                result = childType; // dtype-preserving (abs/negative/sign/square/…)
            }

            ResolveChild(_child, ct, childType, nodeTypes);
            nodeTypes[this] = result;
            return NDExprTypeInfo.Strong(result);
        }
    }

    public sealed partial class ComparisonNode
    {
        internal override NDExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NDExpr, NPTypeCode> nodeTypes)
        {
            var lt = _left.InferType(inputTypes, nodeTypes);
            var rt = _right.InferType(inputTypes, nodeTypes);
            var common = NDExprTypeRules.PromoteMixed(lt, rt);

            ResolveChild(_left, lt, common, nodeTypes);
            ResolveChild(_right, rt, common, nodeTypes);
            nodeTypes[this] = NPTypeCode.Boolean;
            return NDExprTypeInfo.Strong(NPTypeCode.Boolean);
        }
    }

    public sealed partial class MinMaxNode
    {
        internal override NDExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NDExpr, NPTypeCode> nodeTypes)
        {
            var lt = _left.InferType(inputTypes, nodeTypes);
            var rt = _right.InferType(inputTypes, nodeTypes);
            var common = NDExprTypeRules.PromoteMixed(lt, rt);

            ResolveChild(_left, lt, common, nodeTypes);
            ResolveChild(_right, rt, common, nodeTypes);
            nodeTypes[this] = common;
            return NDExprTypeInfo.Strong(common);
        }
    }

    public sealed partial class WhereNode
    {
        internal override NDExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NDExpr, NPTypeCode> nodeTypes)
        {
            var condT = _cond.InferType(inputTypes, nodeTypes);
            ResolveChild(_cond, condT, condT.IsWeak ? condT.DefaultCode : condT.Code, nodeTypes);

            var at = _a.InferType(inputTypes, nodeTypes);
            var bt = _b.InferType(inputTypes, nodeTypes);
            var common = NDExprTypeRules.PromoteMixed(at, bt);

            ResolveChild(_a, at, common, nodeTypes);
            ResolveChild(_b, bt, common, nodeTypes);
            nodeTypes[this] = common;
            return NDExprTypeInfo.Strong(common);
        }
    }

    public sealed partial class CallNode
    {
        internal override NDExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NDExpr, NPTypeCode> nodeTypes)
        {
            for (int i = 0; i < _args.Length; i++)
            {
                var at = _args[i].InferType(inputTypes, nodeTypes);
                // Weak literals adopt the parameter dtype directly; strong
                // args keep their own dtype (emission converts at the edge).
                ResolveChild(_args[i], at, _paramCodes[i], nodeTypes);
            }

            nodeTypes[this] = _returnCode;
            return NDExprTypeInfo.Strong(_returnCode);
        }
    }
}
