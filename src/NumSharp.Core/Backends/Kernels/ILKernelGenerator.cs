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
//     - GetContiguousKernel<T>(), GenerateUnifiedKernel<T>()
//     - Generic helpers: IsSimdSupported<T>(), EmitLoadIndirect<T>(), etc.
//
// ILKernelGenerator.MixedType.cs
//   OWNERSHIP: Mixed-type binary operations with type promotion
//   RESPONSIBILITY:
//     - Handles all binary ops where operand types may differ
//     - Generates path-specific kernels based on stride patterns
//     - Owns ClearAll() which clears ALL caches across all partials
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for general binary operations
//   KEY MEMBERS:
//     - MixedTypeKernel delegate, _mixedTypeCache
//     - GetMixedTypeKernel(), TryGetMixedTypeKernel(), ClearAll()
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
//     - GetTypedElementReductionKernel<T>(), ClearReduction()
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
                    il.EmitCall(OpCodes.Call, typeof(decimal).GetMethod("op_Implicit", new[] { typeof(int) })!, null);
                    return;
                }
                if (from == NPTypeCode.Char)
                {
                    // char -> int -> decimal
                    il.Emit(OpCodes.Conv_I4);
                    il.EmitCall(OpCodes.Call, typeof(decimal).GetMethod("op_Implicit", new[] { typeof(int) })!, null);
                    return;
                }

                var method = from switch
                {
                    NPTypeCode.Byte => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(byte) }),
                    NPTypeCode.Int16 => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(short) }),
                    NPTypeCode.UInt16 => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(ushort) }),
                    NPTypeCode.Int32 => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(int) }),
                    NPTypeCode.UInt32 => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(uint) }),
                    NPTypeCode.Int64 => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(long) }),
                    NPTypeCode.UInt64 => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(ulong) }),
                    NPTypeCode.Single => typeof(decimal).GetMethod("op_Explicit", new[] { typeof(float) }),
                    NPTypeCode.Double => typeof(decimal).GetMethod("op_Explicit", new[] { typeof(double) }),
                    _ => throw new NotSupportedException($"Cannot convert {from} to decimal")
                };
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else
            {
                // Convert from decimal - need to handle bool/char
                if (to == NPTypeCode.Boolean)
                {
                    // decimal -> int -> bool (compare with 0)
                    il.EmitCall(OpCodes.Call, typeof(decimal).GetMethod("ToInt32", new[] { typeof(decimal) })!, null);
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Cgt_Un);
                    return;
                }
                if (to == NPTypeCode.Char)
                {
                    // decimal -> int -> char
                    il.EmitCall(OpCodes.Call, typeof(decimal).GetMethod("ToInt32", new[] { typeof(decimal) })!, null);
                    il.Emit(OpCodes.Conv_U2);
                    return;
                }

                var method = to switch
                {
                    NPTypeCode.Byte => typeof(decimal).GetMethod("ToByte", new[] { typeof(decimal) }),
                    NPTypeCode.Int16 => typeof(decimal).GetMethod("ToInt16", new[] { typeof(decimal) }),
                    NPTypeCode.UInt16 => typeof(decimal).GetMethod("ToUInt16", new[] { typeof(decimal) }),
                    NPTypeCode.Int32 => typeof(decimal).GetMethod("ToInt32", new[] { typeof(decimal) }),
                    NPTypeCode.UInt32 => typeof(decimal).GetMethod("ToUInt32", new[] { typeof(decimal) }),
                    NPTypeCode.Int64 => typeof(decimal).GetMethod("ToInt64", new[] { typeof(decimal) }),
                    NPTypeCode.UInt64 => typeof(decimal).GetMethod("ToUInt64", new[] { typeof(decimal) }),
                    NPTypeCode.Single => typeof(decimal).GetMethod("ToSingle", new[] { typeof(decimal) }),
                    NPTypeCode.Double => typeof(decimal).GetMethod("ToDouble", new[] { typeof(decimal) }),
                    _ => throw new NotSupportedException($"Cannot convert decimal to {to}")
                };
                il.EmitCall(OpCodes.Call, method!, null);
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
                BinaryOp.Mod => IsUnsigned(resultType) ? OpCodes.Rem_Un : OpCodes.Rem,
                BinaryOp.BitwiseAnd => OpCodes.And,
                BinaryOp.BitwiseOr => OpCodes.Or,
                BinaryOp.BitwiseXor => OpCodes.Xor,
                _ => throw new NotSupportedException($"Operation {op} not supported")
            };

            il.Emit(opcode);
        }

        /// <summary>
        /// Emit decimal-specific operation using operator methods.
        /// </summary>
        private static void EmitDecimalOperation(ILGenerator il, BinaryOp op)
        {
            // Bitwise operations not supported for decimal
            if (op == BinaryOp.BitwiseAnd || op == BinaryOp.BitwiseOr || op == BinaryOp.BitwiseXor)
                throw new NotSupportedException($"Bitwise operation {op} not supported for decimal type");

            var methodName = op switch
            {
                BinaryOp.Add => "op_Addition",
                BinaryOp.Subtract => "op_Subtraction",
                BinaryOp.Multiply => "op_Multiply",
                BinaryOp.Divide => "op_Division",
                BinaryOp.Mod => "op_Modulus",
                _ => throw new NotSupportedException($"Operation {op} not supported for decimal")
            };

            var method = typeof(decimal).GetMethod(
                methodName,
                BindingFlags.Public | BindingFlags.Static,
                null,
                new[] { typeof(decimal), typeof(decimal) },
                null
            );

            il.EmitCall(OpCodes.Call, method!, null);
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

        /// <summary>
        /// Clear all cached kernels (IKernelProvider interface).
        /// </summary>
        void IKernelProvider.Clear() => ClearAll();

        /// <summary>
        /// Total number of cached kernels across all caches.
        /// </summary>
        int IKernelProvider.CacheCount => GetTotalCacheCount();

        /// <summary>
        /// Get total count of all cached kernels.
        /// </summary>
        private static int GetTotalCacheCount()
        {
            // Sum counts from all partial class caches
            int count = 0;
            count += CachedCount;           // Binary: _contiguousKernelCache
            count += MixedTypeCachedCount;  // MixedType: _mixedTypeCache
            count += UnaryCachedCount;      // Unary: _unaryCache
            count += UnaryScalarCachedCount; // Unary: _unaryScalarCache
            count += ComparisonCachedCount; // Comparison: _comparisonCache
            count += ComparisonScalarCachedCount; // Comparison: _comparisonScalarCache
            count += ElementReductionCachedCount;  // Reduction: _elementReductionCache
            return count;
        }

        #endregion
    }
}
