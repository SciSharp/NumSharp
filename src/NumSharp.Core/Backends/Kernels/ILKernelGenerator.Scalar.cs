using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Scalar.cs - Scalar Kernel Delegates
// =============================================================================
//
// RESPONSIBILITY:
//   - Unary scalar kernels: Func<TInput, TOutput>
//   - Binary scalar kernels: Func<TLhs, TRhs, TResult>
//   - Used for single-value operations in broadcasting
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public sealed partial class ILKernelGenerator
    {
        #region Scalar Kernel Generation

        /// <summary>
        /// Cache for unary scalar kernels.
        /// Key: UnaryScalarKernelKey (InputType, OutputType, Op)
        /// Value: Delegate (Func&lt;TInput, TOutput&gt;)
        /// </summary>
        private static readonly ConcurrentDictionary<UnaryScalarKernelKey, Delegate> _unaryScalarCache = new();

        /// <summary>
        /// Cache for binary scalar kernels.
        /// Key: BinaryScalarKernelKey (LhsType, RhsType, ResultType, Op)
        /// Value: Delegate (Func&lt;TLhs, TRhs, TResult&gt;)
        /// </summary>
        private static readonly ConcurrentDictionary<BinaryScalarKernelKey, Delegate> _binaryScalarCache = new();

        /// <summary>
        /// Number of unary scalar kernels in cache.
        /// </summary>
        public static int UnaryScalarCachedCount => _unaryScalarCache.Count;

        /// <summary>
        /// Number of binary scalar kernels in cache.
        /// </summary>
        public static int BinaryScalarCachedCount => _binaryScalarCache.Count;

        /// <summary>
        /// Clear the scalar kernel caches.
        /// </summary>
        public static void ClearScalar()
        {
            _unaryScalarCache.Clear();
            _binaryScalarCache.Clear();
        }

        /// <summary>
        /// Get or generate an IL-based unary scalar delegate.
        /// Returns a Func&lt;TInput, TOutput&gt; delegate.
        /// </summary>
        public static Delegate GetUnaryScalarDelegate(UnaryScalarKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            return _unaryScalarCache.GetOrAdd(key, GenerateUnaryScalarDelegate);
        }

        /// <summary>
        /// Get or generate an IL-based binary scalar delegate.
        /// Returns a Func&lt;TLhs, TRhs, TResult&gt; delegate.
        /// </summary>
        public static Delegate GetBinaryScalarDelegate(BinaryScalarKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            return _binaryScalarCache.GetOrAdd(key, GenerateBinaryScalarDelegate);
        }

        /// <summary>
        /// Generate an IL-based unary scalar delegate.
        /// Creates a Func&lt;TInput, TOutput&gt; that performs the operation.
        /// </summary>
        private static Delegate GenerateUnaryScalarDelegate(UnaryScalarKernelKey key)
        {
            var inputClr = GetClrType(key.InputType);
            var outputClr = GetClrType(key.OutputType);

            // Create DynamicMethod: TOutput Method(TInput input)
            var dm = new DynamicMethod(
                name: $"ScalarUnary_{key}",
                returnType: outputClr,
                parameterTypes: new[] { inputClr },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // Load input argument
            il.Emit(OpCodes.Ldarg_0);

            // For predicate operations (IsFinite, IsNan, IsInf), operate on INPUT type
            // and the operation itself produces bool. For other ops, convert first.
            if (IsPredicateOp(key.Op))
            {
                // Perform operation on input type - produces bool
                EmitUnaryScalarOperation(il, key.Op, key.InputType);
            }
            else
            {
                // Convert to output type if different
                EmitConvertTo(il, key.InputType, key.OutputType);
                // Perform the unary operation (result is on stack)
                EmitUnaryScalarOperation(il, key.Op, key.OutputType);
            }

            // Return
            il.Emit(OpCodes.Ret);

            // Create typed Func<TInput, TOutput>
            var funcType = typeof(Func<,>).MakeGenericType(inputClr, outputClr);
            return dm.CreateDelegate(funcType);
        }

        /// <summary>
        /// Generate an IL-based binary scalar delegate.
        /// Creates a Func&lt;TLhs, TRhs, TResult&gt; that performs the operation.
        /// </summary>
        private static Delegate GenerateBinaryScalarDelegate(BinaryScalarKernelKey key)
        {
            var lhsClr = GetClrType(key.LhsType);
            var rhsClr = GetClrType(key.RhsType);
            var resultClr = GetClrType(key.ResultType);

            // Create DynamicMethod: TResult Method(TLhs lhs, TRhs rhs)
            var dm = new DynamicMethod(
                name: $"ScalarBinary_{key}",
                returnType: resultClr,
                parameterTypes: new[] { lhsClr, rhsClr },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // Load lhs, convert to result type
            il.Emit(OpCodes.Ldarg_0);
            EmitConvertTo(il, key.LhsType, key.ResultType);

            // Load rhs, convert to result type
            il.Emit(OpCodes.Ldarg_1);
            EmitConvertTo(il, key.RhsType, key.ResultType);

            // Perform binary operation
            EmitScalarOperation(il, key.Op, key.ResultType);

            // Return
            il.Emit(OpCodes.Ret);

            // Create typed Func<TLhs, TRhs, TResult>
            var funcType = typeof(Func<,,>).MakeGenericType(lhsClr, rhsClr, resultClr);
            return dm.CreateDelegate(funcType);
        }

        #endregion
    }
}
