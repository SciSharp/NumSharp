using System;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using NumSharp.Utilities;

// =============================================================================
// ILKernelGenerator - IL-based SIMD kernel generation using DynamicMethod
// =============================================================================
//
// ARCHITECTURE OVERVIEW
// ---------------------
// This class generates high-performance kernels at runtime using IL emission.
// The JIT compiler can then optimize these kernels with full SIMD support (V128/V256/V512).
// Kernels are cached by operation key to avoid repeated IL generation.
//
// FLOW: Caller (DefaultEngine, np.*, NDArray ops)
//         -> Requests kernel via Get*Kernel() or *Helper() methods
//         -> ILKernelGenerator checks cache, generates IL if needed
//         -> Returns delegate that caller invokes with array pointers
//
// DESIGN: Singleton pattern with IKernelProvider interface implementation.
// Use ILKernelGenerator.Instance for instance methods, or static facades for
// backward compatibility with existing code.
//
// EXCEPTION HANDLING
// ------------------
// All TryGet*Kernel() methods use catch-all exception handling that returns null.
// This is intentional graceful degradation: if IL generation fails for any reason
// (unsupported type, reflection error, invalid IL sequence), the caller receives
// null and falls back to an alternative code path (typically scalar loops or
// throwing NotSupportedException with a descriptive message).
//
// This pattern exists in 14 locations across the partial class files:
//   - Binary.cs: TryGenerateContiguousKernel
//   - MixedType.cs: TryGetMixedTypeKernel
//   - MatMul.cs: GenerateMatMulKernelIL
//   - Unary.cs: TryGetUnaryKernel
//   - Shift.cs: GetShiftScalarKernel, GetShiftArrayKernel
//   - Scan.cs: TryGetCumulativeKernel, TryGetCumulativeAxisKernel
//   - Reduction.cs: TryGetTypedElementReductionKernel
//   - Comparison.cs: TryGetComparisonKernel
//   - ILKernelGenerator.cs: IKernelProvider interface implementations
//
// =============================================================================
// PARTIAL CLASS FILES
// =============================================================================
//
// ILKernelGenerator.cs (THIS FILE)
//   OWNERSHIP: Core infrastructure - foundation for all other partial files
//   RESPONSIBILITY:
//     - Singleton instance and IKernelProvider interface implementation
//     - Global state: Enabled flag, VectorBits/VectorBytes (detected at startup)
//     - Type mapping: NPTypeCode <-> CLR Type <-> Vector type conversions
//     - Shared IL emission primitives used by all other partials
//   DEPENDENCIES: None (other partials depend on this)
//   KEY MEMBERS:
//     - Instance - singleton for IKernelProvider access
//     - Enabled, VectorBits, VectorBytes - runtime SIMD capability
//     - GetVectorContainerType(), GetVectorType() - V128/V256/V512 type selection
//     - GetTypeSize(), GetClrType(), CanUseSimd(), IsUnsigned() - type utilities
//     - EmitLoadIndirect(), EmitStoreIndirect() - memory access IL
//     - EmitConvertTo(), EmitScalarOperation() - type conversion and scalar ops
//     - EmitVectorLoad/Store/Create/Operation() - SIMD operations
//
// ILKernelGenerator.Binary.cs
//   OWNERSHIP: Same-type binary operations on contiguous arrays (fast path)
//   RESPONSIBILITY:
//     - Optimized kernels when both operands have identical type and layout
//     - SIMD loop + scalar tail for Add, Sub, Mul, Div
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for same-type contiguous operations
//   KEY MEMBERS:
//     - ContiguousKernel<T> delegate, _contiguousKernelCache
//     - GetContiguousKernel<T>()
//     - Generic helpers: IsSimdSupported<T>(), EmitLoadIndirect<T>(), etc.
//
// ILKernelGenerator.MixedType.cs
//   OWNERSHIP: Mixed-type binary operations with type promotion
//   RESPONSIBILITY:
//     - Handles all binary ops where operand types may differ
//     - Generates path-specific kernels based on stride patterns
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for general binary operations
//   KEY MEMBERS:
//     - MixedTypeKernel delegate, _mixedTypeCache
//     - GetMixedTypeKernel(), TryGetMixedTypeKernel()
//     - Path generators: GenerateSimdFullKernel(), GenerateGeneralKernel(), etc.
//     - Loop emitters: EmitScalarFullLoop(), EmitSimdFullLoop(), EmitGeneralLoop()
//
// ILKernelGenerator.Unary.cs
//   OWNERSHIP: Unary element-wise operations
//   RESPONSIBILITY:
//     - Math functions: Negate, Abs, Sqrt, Sin, Cos, Exp, Log, Sign, Floor, Ceil, etc.
//     - Scalar delegate generation for single-value operations (Func<TIn,TOut>)
//     - Binary scalar delegates for mixed-type scalar operations
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for unary ops; scalar delegates used in broadcasting
//   KEY MEMBERS:
//     - UnaryKernel delegate, _unaryCache, _unaryScalarCache, _binaryScalarCache
//     - GetUnaryKernel(), GetUnaryScalarDelegate(), GetBinaryScalarDelegate()
//     - EmitUnaryScalarOperation(), EmitMathCall(), EmitSignCall()
//
// ILKernelGenerator.Comparison.cs
//   OWNERSHIP: Comparison operations returning boolean arrays
//   RESPONSIBILITY:
//     - Element-wise comparisons: ==, !=, <, >, <=, >=
//     - SIMD comparison with efficient mask-to-bool extraction
//     - Scalar comparison delegates for single-value operations
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by NDArray comparison operators (==, !=, <, >, etc.)
//   KEY MEMBERS:
//     - ComparisonKernel delegate, _comparisonCache, _comparisonScalarCache
//     - GetComparisonKernel(), GetComparisonScalarDelegate()
//     - EmitVectorComparison(), EmitMaskToBoolExtraction()
//
// ILKernelGenerator.Reduction.cs
//   OWNERSHIP: Reduction operations and specialized SIMD helpers
//   RESPONSIBILITY:
//     - Reductions: Sum, Prod, Min, Max, Mean, ArgMax, ArgMin, All, Any
//     - SIMD helpers called DIRECTLY by other NumSharp code (not just via kernels):
//       * All/Any with early-exit optimization
//       * ArgMax/ArgMin with SIMD two-pass (find value, then find index)
//       * NonZero for finding non-zero indices
//       * Boolean masking: CountTrue, CopyMaskedElements
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Kernels called by DefaultEngine; helpers called directly by np.all/any/nonzero/masking
//   KEY MEMBERS:
//     - TypedElementReductionKernel<T> delegate, _elementReductionCache
//     - GetTypedElementReductionKernel<T>()
//     - AllSimdHelper<T>(), AnySimdHelper<T>() - early-exit boolean reductions
//     - ArgMaxSimdHelper<T>(), ArgMinSimdHelper<T>() - index-tracking reductions
//     - NonZeroSimdHelper<T>(), ConvertFlatIndicesToCoordinates()
//     - CountTrueSimdHelper(), CopyMaskedElementsHelper<T>()
//     - EmitTreeReduction(), EmitVectorHorizontalReduction()
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Generates IL-based SIMD kernels using DynamicMethod.
    /// These kernels provide ~10-15% speedup over the C# reference implementations
    /// by allowing the JIT to inline Vector256 operations more aggressively.
    ///
    /// Implements <see cref="IKernelProvider"/> for unified kernel access.
    /// Use <see cref="Instance"/> for interface-based access, or static methods
    /// for backward compatibility.
    /// </summary>
    public sealed partial class ILKernelGenerator : IKernelProvider
    {
        #region Singleton and IKernelProvider Implementation

        /// <summary>
        /// Singleton instance for IKernelProvider interface access.
        /// </summary>
        public static readonly ILKernelGenerator Instance = new();

        /// <summary>
        /// Private constructor to enforce singleton pattern.
        /// </summary>
        private ILKernelGenerator() { }

        /// <summary>
        /// Provider name for diagnostics.
        /// </summary>
        public string Name => "IL";

        /// <summary>
        /// Whether IL generation is enabled. Can be disabled for debugging.
        /// </summary>
        public static bool Enabled { get; set; } = true;

        // IKernelProvider.Enabled - instance property delegates to static
        bool IKernelProvider.Enabled
        {
            get => Enabled;
            set => Enabled = value;
        }

        /// <summary>
        /// Detected vector width at startup: 512, 256, 128, or 0 (no SIMD).
        /// </summary>
        public static readonly int VectorBits =
            Vector512.IsHardwareAccelerated ? 512 :
            Vector256.IsHardwareAccelerated ? 256 :
            Vector128.IsHardwareAccelerated ? 128 : 0;

        // IKernelProvider.VectorBits - instance property delegates to static
        int IKernelProvider.VectorBits => VectorBits;

        /// <summary>
        /// Number of bytes per vector register.
        /// </summary>
        public static readonly int VectorBytes = VectorBits / 8;

        #endregion

        #region Cached MethodInfo Lookups

        /// <summary>
        /// Pre-cached MethodInfo references for frequently used reflection calls.
        /// Caching these avoids repeated GetMethod() lookups during kernel generation.
        /// </summary>
        private static class CachedMethods
        {
            // Math methods (double versions)
            public static readonly MethodInfo MathPow = typeof(Math).GetMethod(nameof(Math.Pow), new[] { typeof(double), typeof(double) })!;
            public static readonly MethodInfo MathFloor = typeof(Math).GetMethod(nameof(Math.Floor), new[] { typeof(double) })!;
            public static readonly MethodInfo MathAtan2 = typeof(Math).GetMethod(nameof(Math.Atan2), new[] { typeof(double), typeof(double) })!;

            // Decimal conversion methods (to decimal)
            public static readonly MethodInfo DecimalImplicitFromInt = typeof(decimal).GetMethod("op_Implicit", new[] { typeof(int) })!;
            public static readonly MethodInfo DecimalImplicitFromByte = typeof(decimal).GetMethod("op_Implicit", new[] { typeof(byte) })!;
            public static readonly MethodInfo DecimalImplicitFromShort = typeof(decimal).GetMethod("op_Implicit", new[] { typeof(short) })!;
            public static readonly MethodInfo DecimalImplicitFromUShort = typeof(decimal).GetMethod("op_Implicit", new[] { typeof(ushort) })!;
            public static readonly MethodInfo DecimalImplicitFromUInt = typeof(decimal).GetMethod("op_Implicit", new[] { typeof(uint) })!;
            public static readonly MethodInfo DecimalImplicitFromLong = typeof(decimal).GetMethod("op_Implicit", new[] { typeof(long) })!;
            public static readonly MethodInfo DecimalImplicitFromULong = typeof(decimal).GetMethod("op_Implicit", new[] { typeof(ulong) })!;
            public static readonly MethodInfo DecimalExplicitFromFloat = typeof(decimal).GetMethod("op_Explicit", new[] { typeof(float) })!;
            public static readonly MethodInfo DecimalExplicitFromDouble = typeof(decimal).GetMethod("op_Explicit", new[] { typeof(double) })!;

            // Decimal conversion methods (from decimal)
            public static readonly MethodInfo DecimalToByte = typeof(decimal).GetMethod("ToByte", new[] { typeof(decimal) })!;
            public static readonly MethodInfo DecimalToInt16 = typeof(decimal).GetMethod("ToInt16", new[] { typeof(decimal) })!;
            public static readonly MethodInfo DecimalToUInt16 = typeof(decimal).GetMethod("ToUInt16", new[] { typeof(decimal) })!;
            public static readonly MethodInfo DecimalToInt32 = typeof(decimal).GetMethod("ToInt32", new[] { typeof(decimal) })!;
            public static readonly MethodInfo DecimalToUInt32 = typeof(decimal).GetMethod("ToUInt32", new[] { typeof(decimal) })!;
            public static readonly MethodInfo DecimalToInt64 = typeof(decimal).GetMethod("ToInt64", new[] { typeof(decimal) })!;
            public static readonly MethodInfo DecimalToUInt64 = typeof(decimal).GetMethod("ToUInt64", new[] { typeof(decimal) })!;
            public static readonly MethodInfo DecimalToSingle = typeof(decimal).GetMethod("ToSingle", new[] { typeof(decimal) })!;
            public static readonly MethodInfo DecimalToDouble = typeof(decimal).GetMethod("ToDouble", new[] { typeof(decimal) })!;

            // Decimal operator methods
            public static readonly MethodInfo DecimalOpAddition = typeof(decimal).GetMethod("op_Addition",
                BindingFlags.Public | BindingFlags.Static, null, new[] { typeof(decimal), typeof(decimal) }, null)!;
            public static readonly MethodInfo DecimalOpSubtraction = typeof(decimal).GetMethod("op_Subtraction",
                BindingFlags.Public | BindingFlags.Static, null, new[] { typeof(decimal), typeof(decimal) }, null)!;
            public static readonly MethodInfo DecimalOpMultiply = typeof(decimal).GetMethod("op_Multiply",
                BindingFlags.Public | BindingFlags.Static, null, new[] { typeof(decimal), typeof(decimal) }, null)!;
            public static readonly MethodInfo DecimalOpDivision = typeof(decimal).GetMethod("op_Division",
                BindingFlags.Public | BindingFlags.Static, null, new[] { typeof(decimal), typeof(decimal) }, null)!;
            public static readonly MethodInfo DecimalFloor = typeof(decimal).GetMethod(nameof(decimal.Floor),
                BindingFlags.Public | BindingFlags.Static, null, new[] { typeof(decimal) }, null)!;

            // DecimalMath.DecimalEx methods
            public static readonly MethodInfo DecimalExPow = typeof(DecimalMath.DecimalEx).GetMethod(
                nameof(DecimalMath.DecimalEx.Pow), BindingFlags.Public | BindingFlags.Static, null,
                new[] { typeof(decimal), typeof(decimal) }, null)!;
            public static readonly MethodInfo DecimalExATan2 = typeof(DecimalMath.DecimalEx).GetMethod(
                nameof(DecimalMath.DecimalEx.ATan2), BindingFlags.Public | BindingFlags.Static, null,
                new[] { typeof(decimal), typeof(decimal) }, null)!;
        }

        #endregion

        /// <summary>
        /// Get the Vector container type (Vector128, Vector256, or Vector512).
        /// </summary>
        internal static Type GetVectorContainerType() => VectorBits switch
        {
            512 => typeof(Vector512),
            256 => typeof(Vector256),
            128 => typeof(Vector128),
            _ => throw new NotSupportedException("No SIMD support")
        };

        /// <summary>
        /// Get the Vector{Width}&lt;T&gt; generic type.
        /// </summary>
        internal static Type GetVectorType(Type elementType) => VectorBits switch
        {
            512 => typeof(Vector512<>).MakeGenericType(elementType),
            256 => typeof(Vector256<>).MakeGenericType(elementType),
            128 => typeof(Vector128<>).MakeGenericType(elementType),
            _ => throw new NotSupportedException("No SIMD support")
        };

        #region NPTypeCode-Based IL Helpers

        /// <summary>
        /// Get size in bytes for NPTypeCode.
        /// </summary>
        internal static int GetTypeSize(NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => 1,
                NPTypeCode.Byte => 1,
                NPTypeCode.Int16 => 2,
                NPTypeCode.UInt16 => 2,
                NPTypeCode.Int32 => 4,
                NPTypeCode.UInt32 => 4,
                NPTypeCode.Int64 => 8,
                NPTypeCode.UInt64 => 8,
                NPTypeCode.Char => 2,
                NPTypeCode.Single => 4,
                NPTypeCode.Double => 8,
                NPTypeCode.Decimal => 16,
                _ => throw new NotSupportedException($"Type {type} not supported")
            };
        }

        /// <summary>
        /// Get CLR Type for NPTypeCode.
        /// </summary>
        internal static Type GetClrType(NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => typeof(bool),
                NPTypeCode.Byte => typeof(byte),
                NPTypeCode.Int16 => typeof(short),
                NPTypeCode.UInt16 => typeof(ushort),
                NPTypeCode.Int32 => typeof(int),
                NPTypeCode.UInt32 => typeof(uint),
                NPTypeCode.Int64 => typeof(long),
                NPTypeCode.UInt64 => typeof(ulong),
                NPTypeCode.Char => typeof(char),
                NPTypeCode.Single => typeof(float),
                NPTypeCode.Double => typeof(double),
                NPTypeCode.Decimal => typeof(decimal),
                _ => throw new NotSupportedException($"Type {type} not supported")
            };
        }

        /// <summary>
        /// Check if type supports SIMD operations (V128/V256/V512).
        /// </summary>
        internal static bool CanUseSimd(NPTypeCode type)
        {
            if (VectorBits == 0) return false;  // No SIMD hardware

            return type switch
            {
                NPTypeCode.Byte => true,
                NPTypeCode.Int16 or NPTypeCode.UInt16 => true,
                NPTypeCode.Int32 or NPTypeCode.UInt32 => true,
                NPTypeCode.Int64 or NPTypeCode.UInt64 => true,
                NPTypeCode.Single or NPTypeCode.Double => true,
                _ => false  // Boolean, Char, Decimal
            };
        }

        /// <summary>
        /// Get vector element count for type (adapts to V128/V256/V512).
        /// </summary>
        internal static int GetVectorCount(NPTypeCode type)
        {
            if (VectorBits == 0) return 1;  // Scalar fallback
            return VectorBytes / GetTypeSize(type);
        }

        /// <summary>
        /// Emit load indirect for NPTypeCode.
        /// </summary>
        internal static void EmitLoadIndirect(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                    il.Emit(OpCodes.Ldind_U1);
                    break;
                case NPTypeCode.Int16:
                    il.Emit(OpCodes.Ldind_I2);
                    break;
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    il.Emit(OpCodes.Ldind_U2);
                    break;
                case NPTypeCode.Int32:
                    il.Emit(OpCodes.Ldind_I4);
                    break;
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Ldind_U4);
                    break;
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Ldind_I8);
                    break;
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Ldind_R4);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Ldind_R8);
                    break;
                case NPTypeCode.Decimal:
                    il.Emit(OpCodes.Ldobj, typeof(decimal));
                    break;
                default:
                    throw new NotSupportedException($"Type {type} not supported for ldind");
            }
        }

        /// <summary>
        /// Emit store indirect for NPTypeCode.
        /// </summary>
        internal static void EmitStoreIndirect(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                    il.Emit(OpCodes.Stind_I1);
                    break;
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    il.Emit(OpCodes.Stind_I2);
                    break;
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Stind_I4);
                    break;
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Stind_I8);
                    break;
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Stind_R4);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Stind_R8);
                    break;
                case NPTypeCode.Decimal:
                    il.Emit(OpCodes.Stobj, typeof(decimal));
                    break;
                default:
                    throw new NotSupportedException($"Type {type} not supported for stind");
            }
        }

        /// <summary>
        /// Emit type conversion from source to target type.
        /// </summary>
        internal static void EmitConvertTo(ILGenerator il, NPTypeCode from, NPTypeCode to)
        {
            if (from == to)
                return; // No conversion needed

            // Special case: decimal conversions require method calls
            if (from == NPTypeCode.Decimal || to == NPTypeCode.Decimal)
            {
                EmitDecimalConversion(il, from, to);
                return;
            }

            // For numeric types, use conv.* opcodes
            switch (to)
            {
                case NPTypeCode.Boolean:
                    // Convert to bool: != 0
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Cgt_Un);
                    break;
                case NPTypeCode.Byte:
                    il.Emit(OpCodes.Conv_U1);
                    break;
                case NPTypeCode.Int16:
                    il.Emit(OpCodes.Conv_I2);
                    break;
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    il.Emit(OpCodes.Conv_U2);
                    break;
                case NPTypeCode.Int32:
                    il.Emit(OpCodes.Conv_I4);
                    break;
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Conv_U4);
                    break;
                case NPTypeCode.Int64:
                    if (IsUnsigned(from))
                        il.Emit(OpCodes.Conv_U8);
                    else
                        il.Emit(OpCodes.Conv_I8);
                    break;
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Conv_U8);
                    break;
                case NPTypeCode.Single:
                    if (IsUnsigned(from))
                        il.Emit(OpCodes.Conv_R_Un);
                    il.Emit(OpCodes.Conv_R4);
                    break;
                case NPTypeCode.Double:
                    if (IsUnsigned(from))
                        il.Emit(OpCodes.Conv_R_Un);
                    il.Emit(OpCodes.Conv_R8);
                    break;
                default:
                    throw new NotSupportedException($"Conversion to {to} not supported");
            }
        }

        /// <summary>
        /// Emit decimal-specific conversions.
        /// </summary>
        private static void EmitDecimalConversion(ILGenerator il, NPTypeCode from, NPTypeCode to)
        {
            if (to == NPTypeCode.Decimal)
            {
                // Convert to decimal - need to handle bool/char first
                if (from == NPTypeCode.Boolean)
                {
                    // bool -> int -> decimal
                    il.Emit(OpCodes.Conv_I4);
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalImplicitFromInt, null);
                    return;
                }
                if (from == NPTypeCode.Char)
                {
                    // char -> int -> decimal
                    il.Emit(OpCodes.Conv_I4);
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalImplicitFromInt, null);
                    return;
                }

                var method = from switch
                {
                    NPTypeCode.Byte => CachedMethods.DecimalImplicitFromByte,
                    NPTypeCode.Int16 => CachedMethods.DecimalImplicitFromShort,
                    NPTypeCode.UInt16 => CachedMethods.DecimalImplicitFromUShort,
                    NPTypeCode.Int32 => CachedMethods.DecimalImplicitFromInt,
                    NPTypeCode.UInt32 => CachedMethods.DecimalImplicitFromUInt,
                    NPTypeCode.Int64 => CachedMethods.DecimalImplicitFromLong,
                    NPTypeCode.UInt64 => CachedMethods.DecimalImplicitFromULong,
                    NPTypeCode.Single => CachedMethods.DecimalExplicitFromFloat,
                    NPTypeCode.Double => CachedMethods.DecimalExplicitFromDouble,
                    _ => throw new NotSupportedException($"Cannot convert {from} to decimal")
                };
                il.EmitCall(OpCodes.Call, method, null);
            }
            else
            {
                // Convert from decimal - need to handle bool/char
                if (to == NPTypeCode.Boolean)
                {
                    // decimal -> int -> bool (compare with 0)
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalToInt32, null);
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Cgt_Un);
                    return;
                }
                if (to == NPTypeCode.Char)
                {
                    // decimal -> int -> char
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalToInt32, null);
                    il.Emit(OpCodes.Conv_U2);
                    return;
                }

                var method = to switch
                {
                    NPTypeCode.Byte => CachedMethods.DecimalToByte,
                    NPTypeCode.Int16 => CachedMethods.DecimalToInt16,
                    NPTypeCode.UInt16 => CachedMethods.DecimalToUInt16,
                    NPTypeCode.Int32 => CachedMethods.DecimalToInt32,
                    NPTypeCode.UInt32 => CachedMethods.DecimalToUInt32,
                    NPTypeCode.Int64 => CachedMethods.DecimalToInt64,
                    NPTypeCode.UInt64 => CachedMethods.DecimalToUInt64,
                    NPTypeCode.Single => CachedMethods.DecimalToSingle,
                    NPTypeCode.Double => CachedMethods.DecimalToDouble,
                    _ => throw new NotSupportedException($"Cannot convert decimal to {to}")
                };
                il.EmitCall(OpCodes.Call, method, null);
            }
        }

        /// <summary>
        /// Check if type is unsigned.
        /// </summary>
        internal static bool IsUnsigned(NPTypeCode type)
        {
            return type == NPTypeCode.Byte || type == NPTypeCode.UInt16 ||
                   type == NPTypeCode.UInt32 || type == NPTypeCode.UInt64 ||
                   type == NPTypeCode.Char;
        }

        /// <summary>
        /// Emit scalar operation for NPTypeCode.
        /// </summary>
        internal static void EmitScalarOperation(ILGenerator il, BinaryOp op, NPTypeCode resultType)
        {
            // Special handling for decimal (uses operator methods)
            if (resultType == NPTypeCode.Decimal)
            {
                EmitDecimalOperation(il, op);
                return;
            }

            // Special handling for Power - requires Math.Pow call
            if (op == BinaryOp.Power)
            {
                EmitPowerOperation(il, resultType);
                return;
            }

            // Special handling for FloorDivide - division followed by floor for floats
            if (op == BinaryOp.FloorDivide)
            {
                EmitFloorDivideOperation(il, resultType);
                return;
            }

            // Special handling for Mod - NumPy uses floored division semantics (Python %)
            // Result sign matches divisor sign, not dividend sign (unlike C# %)
            if (op == BinaryOp.Mod)
            {
                EmitModOperation(il, resultType);
                return;
            }

            // Special handling for ATan2 - requires Math.Atan2 call
            if (op == BinaryOp.ATan2)
            {
                EmitATan2Operation(il, resultType);
                return;
            }

            // Special handling for boolean
            if (resultType == NPTypeCode.Boolean)
            {
                // For bool, only meaningful ops are probably logical, but we'll support arithmetic
                // Treat as byte arithmetic
            }

            var opcode = op switch
            {
                BinaryOp.Add => OpCodes.Add,
                BinaryOp.Subtract => OpCodes.Sub,
                BinaryOp.Multiply => OpCodes.Mul,
                BinaryOp.Divide => IsUnsigned(resultType) ? OpCodes.Div_Un : OpCodes.Div,
                BinaryOp.BitwiseAnd => OpCodes.And,
                BinaryOp.BitwiseOr => OpCodes.Or,
                BinaryOp.BitwiseXor => OpCodes.Xor,
                _ => throw new NotSupportedException($"Operation {op} not supported")
            };

            il.Emit(opcode);
        }

        /// <summary>
        /// Emit Power operation using Math.Pow.
        /// Stack: [base, exponent] -> [result]
        /// </summary>
        private static void EmitPowerOperation(ILGenerator il, NPTypeCode resultType)
        {
            // Math.Pow(double, double) -> double
            // We need to convert both operands to double, call Math.Pow, then convert back

            // Stack has: base (resultType), exponent (resultType)
            // We need to convert both to double for Math.Pow

            // Store exponent temporarily
            var locExp = il.DeclareLocal(GetClrType(resultType));
            il.Emit(OpCodes.Stloc, locExp);

            // Convert base to double
            if (resultType != NPTypeCode.Double)
            {
                if (IsUnsigned(resultType))
                    il.Emit(OpCodes.Conv_R_Un);
                il.Emit(OpCodes.Conv_R8);
            }

            // Load and convert exponent to double
            il.Emit(OpCodes.Ldloc, locExp);
            if (resultType != NPTypeCode.Double)
            {
                if (IsUnsigned(resultType))
                    il.Emit(OpCodes.Conv_R_Un);
                il.Emit(OpCodes.Conv_R8);
            }

            // Call Math.Pow(double, double)
            il.EmitCall(OpCodes.Call, CachedMethods.MathPow, null);

            // Convert result back to target type
            if (resultType != NPTypeCode.Double)
            {
                EmitConvertFromDouble(il, resultType);
            }
        }

        /// <summary>
        /// Emit FloorDivide operation.
        /// NumPy floor_divide always floors toward negative infinity.
        /// For floats: divide then Math.Floor.
        /// For unsigned integers: regular division (same as floor for positive).
        /// For signed integers: correct floor division toward negative infinity.
        /// Stack: [dividend, divisor] -> [result]
        /// </summary>
        private static void EmitFloorDivideOperation(ILGenerator il, NPTypeCode resultType)
        {
            // For floating-point types, divide then floor
            if (resultType == NPTypeCode.Single || resultType == NPTypeCode.Double)
            {
                il.Emit(OpCodes.Div);

                if (resultType == NPTypeCode.Single)
                {
                    il.Emit(OpCodes.Conv_R8);
                    il.EmitCall(OpCodes.Call, CachedMethods.MathFloor, null);
                    il.Emit(OpCodes.Conv_R4);
                }
                else
                {
                    il.EmitCall(OpCodes.Call, CachedMethods.MathFloor, null);
                }
            }
            else if (IsUnsigned(resultType))
            {
                // Unsigned integers: floor = regular division
                il.Emit(OpCodes.Div_Un);
            }
            else
            {
                // Signed integers: need true floor division (toward negative infinity)
                // NumPy: floor_divide(-7, 3) = -3, not -2
                // Approach: convert to double, divide, floor, convert back
                // Stack on entry: [dividend, divisor]

                // Store divisor first (it's on top)
                var locDivisor = il.DeclareLocal(typeof(long));
                il.Emit(OpCodes.Conv_I8);  // Convert to long for storage
                il.Emit(OpCodes.Stloc, locDivisor);

                // Convert dividend to double
                il.Emit(OpCodes.Conv_R8);
                var locDividendDouble = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locDividendDouble);

                // Convert divisor to double
                il.Emit(OpCodes.Ldloc, locDivisor);
                il.Emit(OpCodes.Conv_R8);

                // Load dividend and divisor as doubles
                var locDivisorDouble = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locDivisorDouble);
                il.Emit(OpCodes.Ldloc, locDividendDouble);
                il.Emit(OpCodes.Ldloc, locDivisorDouble);

                // Divide and floor
                il.Emit(OpCodes.Div);
                il.EmitCall(OpCodes.Call, CachedMethods.MathFloor, null);

                // Convert back to result type
                EmitConvertFromDouble(il, resultType);
            }
        }

        /// <summary>
        /// Emit Mod operation using NumPy/Python floored division semantics.
        /// NumPy: result = a - floor(a / b) * b  (result sign matches divisor sign)
        /// C#:    result = a - trunc(a / b) * b  (result sign matches dividend sign)
        /// Stack: [dividend, divisor] -> [result]
        /// </summary>
        private static void EmitModOperation(ILGenerator il, NPTypeCode resultType)
        {
            // For unsigned types, C# remainder is equivalent to floored modulo
            if (IsUnsigned(resultType))
            {
                il.Emit(OpCodes.Rem_Un);
                return;
            }

            // For floating-point types: result = a - floor(a / b) * b
            if (resultType == NPTypeCode.Single || resultType == NPTypeCode.Double)
            {
                // Stack: [a, b]
                // We need to compute: a - floor(a / b) * b

                var locDivisor = il.DeclareLocal(resultType == NPTypeCode.Single ? typeof(float) : typeof(double));
                var locDividend = il.DeclareLocal(resultType == NPTypeCode.Single ? typeof(float) : typeof(double));

                // Store divisor (b)
                il.Emit(OpCodes.Stloc, locDivisor);
                // Store dividend (a)
                il.Emit(OpCodes.Stloc, locDividend);

                // Load a for final subtraction
                il.Emit(OpCodes.Ldloc, locDividend);

                // Compute floor(a / b) * b
                il.Emit(OpCodes.Ldloc, locDividend);
                il.Emit(OpCodes.Ldloc, locDivisor);
                il.Emit(OpCodes.Div);

                // Call Math.Floor
                if (resultType == NPTypeCode.Single)
                {
                    il.Emit(OpCodes.Conv_R8);  // Math.Floor takes double
                    il.EmitCall(OpCodes.Call, CachedMethods.MathFloor, null);
                    il.Emit(OpCodes.Conv_R4);  // Convert back to float
                }
                else
                {
                    il.EmitCall(OpCodes.Call, CachedMethods.MathFloor, null);
                }

                // Multiply by b
                il.Emit(OpCodes.Ldloc, locDivisor);
                il.Emit(OpCodes.Mul);

                // Subtract: a - floor(a/b)*b
                il.Emit(OpCodes.Sub);
                return;
            }

            // For signed integer types, compute: a - floor(a / b) * b
            // Using double arithmetic for correctness

            // Stack: [a, b]
            var locDivisorInt = il.DeclareLocal(typeof(long));
            var locDividendInt = il.DeclareLocal(typeof(long));

            // Widen to long for consistency
            il.Emit(OpCodes.Conv_I8);  // Convert b to long
            il.Emit(OpCodes.Stloc, locDivisorInt);
            il.Emit(OpCodes.Conv_I8);  // Convert a to long
            il.Emit(OpCodes.Stloc, locDividendInt);

            // Load a for final subtraction
            il.Emit(OpCodes.Ldloc, locDividendInt);

            // Compute floor(a / b) * b using double arithmetic
            il.Emit(OpCodes.Ldloc, locDividendInt);
            il.Emit(OpCodes.Conv_R8);  // Convert a to double
            il.Emit(OpCodes.Ldloc, locDivisorInt);
            il.Emit(OpCodes.Conv_R8);  // Convert b to double
            il.Emit(OpCodes.Div);

            // Floor
            il.EmitCall(OpCodes.Call, CachedMethods.MathFloor, null);

            // Convert back to long and multiply by b
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Ldloc, locDivisorInt);
            il.Emit(OpCodes.Mul);

            // Subtract: a - floor(a/b)*b (result is long)
            il.Emit(OpCodes.Sub);

            // Convert to result type
            switch (resultType)
            {
                case NPTypeCode.Int16:
                    il.Emit(OpCodes.Conv_I2);
                    break;
                case NPTypeCode.Int32:
                    il.Emit(OpCodes.Conv_I4);
                    break;
                // Int64 needs no conversion
            }
        }

        /// <summary>
        /// Emit ATan2 operation using Math.Atan2.
        /// NumPy arctan2(y, x) computes the angle in radians between the positive x-axis
        /// and the point (x, y), with the correct quadrant determination.
        /// Stack: [y, x] -> [result] (angle in radians, range [-pi, pi])
        /// </summary>
        private static void EmitATan2Operation(ILGenerator il, NPTypeCode resultType)
        {
            // Math.Atan2(double y, double x) -> double
            // Stack has: y (resultType), x (resultType)
            // We need to convert both to double for Math.Atan2

            // Store x temporarily
            var locX = il.DeclareLocal(GetClrType(resultType));
            il.Emit(OpCodes.Stloc, locX);

            // Convert y to double
            if (resultType != NPTypeCode.Double)
            {
                if (IsUnsigned(resultType))
                    il.Emit(OpCodes.Conv_R_Un);
                il.Emit(OpCodes.Conv_R8);
            }

            // Load and convert x to double
            il.Emit(OpCodes.Ldloc, locX);
            if (resultType != NPTypeCode.Double)
            {
                if (IsUnsigned(resultType))
                    il.Emit(OpCodes.Conv_R_Un);
                il.Emit(OpCodes.Conv_R8);
            }

            // Call Math.Atan2(double y, double x)
            il.EmitCall(OpCodes.Call, CachedMethods.MathAtan2, null);

            // Convert result back to target type
            if (resultType != NPTypeCode.Double)
            {
                EmitConvertFromDouble(il, resultType);
            }
        }

        /// <summary>
        /// Emit conversion from double to target type.
        /// </summary>
        private static void EmitConvertFromDouble(ILGenerator il, NPTypeCode targetType)
        {
            switch (targetType)
            {
                case NPTypeCode.Byte:
                    il.Emit(OpCodes.Conv_U1);
                    break;
                case NPTypeCode.Int16:
                    il.Emit(OpCodes.Conv_I2);
                    break;
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    il.Emit(OpCodes.Conv_U2);
                    break;
                case NPTypeCode.Int32:
                    il.Emit(OpCodes.Conv_I4);
                    break;
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Conv_U4);
                    break;
                case NPTypeCode.Int64:
                    il.Emit(OpCodes.Conv_I8);
                    break;
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Conv_U8);
                    break;
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Conv_R4);
                    break;
                case NPTypeCode.Boolean:
                    // double -> bool: != 0
                    il.Emit(OpCodes.Ldc_R8, 0.0);
                    il.Emit(OpCodes.Ceq);
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Ceq);
                    break;
                // NPTypeCode.Double needs no conversion
            }
        }

        /// <summary>
        /// Emit decimal-specific operation using operator methods.
        /// </summary>
        private static void EmitDecimalOperation(ILGenerator il, BinaryOp op)
        {
            // Bitwise operations not supported for decimal
            if (op == BinaryOp.BitwiseAnd || op == BinaryOp.BitwiseOr || op == BinaryOp.BitwiseXor)
                throw new NotSupportedException($"Bitwise operation {op} not supported for decimal type");

            // Power for decimal uses DecimalEx.Pow
            if (op == BinaryOp.Power)
            {
                il.EmitCall(OpCodes.Call, CachedMethods.DecimalExPow, null);
                return;
            }

            // FloorDivide for decimal: divide then floor toward negative infinity
            if (op == BinaryOp.FloorDivide)
            {
                // Stack: [dividend, divisor]
                // For NumPy semantics: floor(a/b), not truncate
                var locDivisor = il.DeclareLocal(typeof(decimal));
                var locDividend = il.DeclareLocal(typeof(decimal));
                il.Emit(OpCodes.Stloc, locDivisor);
                il.Emit(OpCodes.Stloc, locDividend);

                // Compute a / b
                il.Emit(OpCodes.Ldloc, locDividend);
                il.Emit(OpCodes.Ldloc, locDivisor);
                il.EmitCall(OpCodes.Call, CachedMethods.DecimalOpDivision, null);

                // Call decimal.Floor for floored division toward negative infinity
                il.EmitCall(OpCodes.Call, CachedMethods.DecimalFloor, null);
                return;
            }

            // Mod for decimal: NumPy floored modulo semantics
            // result = a - floor(a / b) * b
            if (op == BinaryOp.Mod)
            {
                // Stack: [dividend, divisor]
                var locDivisor = il.DeclareLocal(typeof(decimal));
                var locDividend = il.DeclareLocal(typeof(decimal));
                il.Emit(OpCodes.Stloc, locDivisor);
                il.Emit(OpCodes.Stloc, locDividend);

                // Load a for final subtraction
                il.Emit(OpCodes.Ldloc, locDividend);

                // Compute floor(a / b)
                il.Emit(OpCodes.Ldloc, locDividend);
                il.Emit(OpCodes.Ldloc, locDivisor);
                il.EmitCall(OpCodes.Call, CachedMethods.DecimalOpDivision, null);
                il.EmitCall(OpCodes.Call, CachedMethods.DecimalFloor, null);

                // Multiply by b
                il.Emit(OpCodes.Ldloc, locDivisor);
                il.EmitCall(OpCodes.Call, CachedMethods.DecimalOpMultiply, null);

                // Subtract: a - floor(a/b)*b
                il.EmitCall(OpCodes.Call, CachedMethods.DecimalOpSubtraction, null);
                return;
            }

            // ATan2 for decimal uses DecimalEx.ATan2
            if (op == BinaryOp.ATan2)
            {
                il.EmitCall(OpCodes.Call, CachedMethods.DecimalExATan2, null);
                return;
            }

            var method = op switch
            {
                BinaryOp.Add => CachedMethods.DecimalOpAddition,
                BinaryOp.Subtract => CachedMethods.DecimalOpSubtraction,
                BinaryOp.Multiply => CachedMethods.DecimalOpMultiply,
                BinaryOp.Divide => CachedMethods.DecimalOpDivision,
                _ => throw new NotSupportedException($"Operation {op} not supported for decimal")
            };

            il.EmitCall(OpCodes.Call, method, null);
        }

        /// <summary>
        /// Emit Vector.Load for NPTypeCode (adapts to V128/V256/V512).
        /// </summary>
        internal static void EmitVectorLoad(ILGenerator il, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);

            var loadMethod = containerType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Load" && m.IsGenericMethod &&
                            m.GetParameters().Length == 1 &&
                            m.GetParameters()[0].ParameterType.IsPointer)
                .MakeGenericMethod(clrType);

            il.EmitCall(OpCodes.Call, loadMethod, null);
        }

        /// <summary>
        /// Emit Vector.Create for NPTypeCode (broadcasts scalar to all vector elements).
        /// Stack must have scalar value on top; result is Vector on stack.
        /// </summary>
        internal static void EmitVectorCreate(ILGenerator il, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);

            var createMethod = containerType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Create" && m.IsGenericMethod &&
                            m.GetParameters().Length == 1 &&
                            !m.GetParameters()[0].ParameterType.IsPointer)
                .MakeGenericMethod(clrType);

            il.EmitCall(OpCodes.Call, createMethod, null);
        }

        /// <summary>
        /// Emit Vector.Store for NPTypeCode (adapts to V128/V256/V512).
        /// </summary>
        internal static void EmitVectorStore(ILGenerator il, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);

            var storeMethod = containerType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Store" && m.IsGenericMethod &&
                            m.GetParameters().Length == 2 &&
                            m.GetParameters()[0].ParameterType.IsGenericType)
                .MakeGenericMethod(clrType);

            il.EmitCall(OpCodes.Call, storeMethod, null);
        }

        /// <summary>
        /// Emit Vector operation for NPTypeCode (adapts to V128/V256/V512).
        /// </summary>
        internal static void EmitVectorOperation(ILGenerator il, BinaryOp op, NPTypeCode type)
        {
            var clrType = GetClrType(type);
            var vectorType = GetVectorType(clrType);
            var containerType = GetVectorContainerType();

            // Bitwise operations use static methods on Vector256/Vector128 container
            if (op == BinaryOp.BitwiseAnd || op == BinaryOp.BitwiseOr || op == BinaryOp.BitwiseXor)
            {
                string methodName = op switch
                {
                    BinaryOp.BitwiseAnd => "BitwiseAnd",
                    BinaryOp.BitwiseOr => "BitwiseOr",
                    BinaryOp.BitwiseXor => "Xor",
                    _ => throw new NotSupportedException()
                };

                var opMethod = containerType
                    .GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .First(m => m.Name == methodName && m.IsGenericMethod &&
                                m.GetParameters().Length == 2)
                    .MakeGenericMethod(clrType);

                il.EmitCall(OpCodes.Call, opMethod, null);
                return;
            }

            // Arithmetic operations use operator overloads on Vector256<T>/Vector128<T>
            string operatorName = op switch
            {
                BinaryOp.Add => "op_Addition",
                BinaryOp.Subtract => "op_Subtraction",
                BinaryOp.Multiply => "op_Multiply",
                BinaryOp.Divide => "op_Division",
                _ => throw new NotSupportedException($"Operation {op} not supported for SIMD")
            };

            var operatorMethod = vectorType.GetMethod(operatorName,
                BindingFlags.Public | BindingFlags.Static,
                null, new[] { vectorType, vectorType }, null);

            il.EmitCall(OpCodes.Call, operatorMethod!, null);
        }

        #endregion

        #region IKernelProvider Interface Implementation

        /// <summary>
        /// Check if type supports SIMD operations (IKernelProvider interface).
        /// </summary>
        bool IKernelProvider.CanUseSimd(NPTypeCode type) => CanUseSimd(type);

        /// <summary>
        /// Get contiguous same-type binary kernel (IKernelProvider interface).
        /// Delegates to static GetContiguousKernel method in Binary partial.
        /// </summary>
        ContiguousKernel<T>? IKernelProvider.GetContiguousKernel<T>(BinaryOp op)
            => GetContiguousKernel<T>(op);

        /// <summary>
        /// Get mixed-type binary kernel (IKernelProvider interface).
        /// Delegates to static TryGetMixedTypeKernel method in MixedType partial.
        /// </summary>
        MixedTypeKernel? IKernelProvider.GetMixedTypeKernel(MixedTypeKernelKey key)
            => TryGetMixedTypeKernel(key);

        /// <summary>
        /// Get unary kernel (IKernelProvider interface).
        /// Delegates to static TryGetUnaryKernel method in Unary partial.
        /// </summary>
        UnaryKernel? IKernelProvider.GetUnaryKernel(UnaryKernelKey key)
            => TryGetUnaryKernel(key);

        /// <summary>
        /// Get unary scalar function (IKernelProvider interface).
        /// </summary>
        UnaryScalar<TIn, TOut>? IKernelProvider.GetUnaryScalar<TIn, TOut>(UnaryOp op)
        {
            var key = new UnaryScalarKernelKey(
                InfoOf<TIn>.NPTypeCode,
                InfoOf<TOut>.NPTypeCode,
                op
            );
            try
            {
                var del = GetUnaryScalarDelegate(key);
                return del as UnaryScalar<TIn, TOut>;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Get element reduction kernel (IKernelProvider interface).
        /// </summary>
        TypedElementReductionKernel<TResult>? IKernelProvider.GetElementReductionKernel<TResult>(ElementReductionKernelKey key)
            => TryGetTypedElementReductionKernel<TResult>(key);

        /// <summary>
        /// Get axis reduction kernel (IKernelProvider interface).
        /// </summary>
        AxisReductionKernel? IKernelProvider.GetAxisReductionKernel(AxisReductionKernelKey key)
            => TryGetAxisReductionKernel(key);

        /// <summary>
        /// Get comparison kernel (IKernelProvider interface).
        /// </summary>
        ComparisonKernel? IKernelProvider.GetComparisonKernel(ComparisonKernelKey key)
            => TryGetComparisonKernel(key);

        /// <summary>
        /// Get comparison scalar function (IKernelProvider interface).
        /// </summary>
        ComparisonScalar<TLhs, TRhs>? IKernelProvider.GetComparisonScalar<TLhs, TRhs>(ComparisonOp op)
        {
            var key = new ComparisonScalarKernelKey(
                InfoOf<TLhs>.NPTypeCode,
                InfoOf<TRhs>.NPTypeCode,
                op
            );
            try
            {
                var del = GetComparisonScalarDelegate(key);
                return del as ComparisonScalar<TLhs, TRhs>;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Get unary scalar delegate with runtime type dispatch (IKernelProvider interface).
        /// </summary>
        Delegate? IKernelProvider.GetUnaryScalarDelegate(UnaryScalarKernelKey key)
        {
            try { return GetUnaryScalarDelegate(key); }
            catch { return null; }
        }

        /// <summary>
        /// Get binary scalar delegate with runtime type dispatch (IKernelProvider interface).
        /// </summary>
        Delegate? IKernelProvider.GetBinaryScalarDelegate(BinaryScalarKernelKey key)
        {
            try { return GetBinaryScalarDelegate(key); }
            catch { return null; }
        }

        /// <summary>
        /// Get comparison scalar delegate with runtime type dispatch (IKernelProvider interface).
        /// </summary>
        Delegate? IKernelProvider.GetComparisonScalarDelegate(ComparisonScalarKernelKey key)
        {
            try { return GetComparisonScalarDelegate(key); }
            catch { return null; }
        }

        #endregion
    }
}
