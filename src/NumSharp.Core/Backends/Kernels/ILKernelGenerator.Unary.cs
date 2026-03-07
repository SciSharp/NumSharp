using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;

// =============================================================================
// ILKernelGenerator - IL-based SIMD kernel generation using DynamicMethod
// =============================================================================
//
// ARCHITECTURE OVERVIEW
// ---------------------
// This partial class generates high-performance kernels at runtime using IL emission.
// The JIT compiler can then optimize these kernels with full SIMD support (V128/V256/V512).
// Kernels are cached by operation key to avoid repeated IL generation.
//
// FLOW: Caller (DefaultEngine, np.*, NDArray ops)
//         -> Requests kernel via Get*Kernel() or *Helper() methods
//         -> ILKernelGenerator checks cache, generates IL if needed
//         -> Returns delegate that caller invokes with array pointers
//
// =============================================================================
// PARTIAL CLASS FILES
// =============================================================================
//
// ILKernelGenerator.cs
//   OWNERSHIP: Core infrastructure - foundation for all other partial files
//   RESPONSIBILITY:
//     - Global state: Enabled flag, VectorBits/VectorBytes (detected at startup)
//     - Type mapping: NPTypeCode <-> CLR Type <-> Vector type conversions
//     - Shared IL emission primitives used by all other partials
//   DEPENDENCIES: None (other partials depend on this)
//
// ILKernelGenerator.Binary.cs
//   OWNERSHIP: Same-type binary operations on contiguous arrays (fast path)
//   RESPONSIBILITY:
//     - Optimized kernels when both operands have identical type and layout
//     - SIMD loop + scalar tail for Add, Sub, Mul, Div
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for same-type contiguous operations
//
// ILKernelGenerator.MixedType.cs
//   OWNERSHIP: Mixed-type binary operations with type promotion
//   RESPONSIBILITY:
//     - Handles all binary ops where operand types may differ
//     - Generates path-specific kernels based on stride patterns
//     - Owns ClearAll() which clears ALL caches across all partials
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for general binary operations
//
// ILKernelGenerator.Unary.cs (THIS FILE)
//   OWNERSHIP: Unary element-wise operations and scalar delegates
//   RESPONSIBILITY:
//     - Array kernels for unary math: Negate, Abs, Sqrt, Sin, Cos, Exp, Log,
//       Sign, Floor, Ceil, Round, Tan, Sinh, Cosh, Tanh, ASin, ACos, ATan,
//       Exp2, Expm1, Log2, Log10, Log1p
//     - SIMD support for Negate, Abs, Sqrt, Floor, Ceil, Round on float/double
//     - Scalar delegates (Func<TIn, TOut>) for single-value operations
//     - Binary scalar delegates (Func<TLhs, TRhs, TResult>) for mixed-type scalars
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW:
//     - Array kernels: Called by DefaultEngine for np.sqrt, np.sin, etc.
//     - Scalar delegates: Used internally for broadcasting and element access
//   KEY MEMBERS:
//     - UnaryKernel delegate, _unaryCache - array operations
//     - _unaryScalarCache - Func<TIn, TOut> for scalar unary ops
//     - _binaryScalarCache - Func<TLhs, TRhs, TResult> for scalar binary ops
//     - GetUnaryKernel(), TryGetUnaryKernel(), ClearUnary()
//     - GetUnaryScalarDelegate(), GetBinaryScalarDelegate()
//     - EmitUnaryScalarOperation() - central dispatcher for all unary ops
//     - EmitMathCall(), EmitSignCall() - Math/MathF function emission
//     - EmitUnarySimdLoop(), EmitUnaryScalarLoop(), EmitUnaryStridedLoop()
//
// ILKernelGenerator.Comparison.cs
//   OWNERSHIP: Comparison operations returning boolean arrays
//   RESPONSIBILITY:
//     - Element-wise comparisons: ==, !=, <, >, <=, >=
//     - SIMD comparison with efficient mask-to-bool extraction
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by NDArray comparison operators
//
// ILKernelGenerator.Reduction.cs
//   OWNERSHIP: Reduction operations and specialized SIMD helpers
//   RESPONSIBILITY:
//     - Reductions: Sum, Prod, Min, Max, Mean, ArgMax, ArgMin, All, Any
//     - SIMD helpers called directly by np.all/any/nonzero/masking
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Kernels called by DefaultEngine; helpers called directly by np.*
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public sealed partial class ILKernelGenerator
    {
        #region Unary Kernel Generation

        /// <summary>
        /// Cache for unary kernels.
        /// Key: UnaryKernelKey (InputType, OutputType, Op, IsContiguous)
        /// </summary>
        private static readonly ConcurrentDictionary<UnaryKernelKey, UnaryKernel> _unaryCache = new();

        /// <summary>
        /// Number of unary kernels in cache.
        /// </summary>
        public static int UnaryCachedCount => _unaryCache.Count;

        /// <summary>
        /// Get or generate a unary kernel for the specified key.
        /// </summary>
        public static UnaryKernel GetUnaryKernel(UnaryKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            return _unaryCache.GetOrAdd(key, GenerateUnaryKernel);
        }

        /// <summary>
        /// Try to get or generate a unary kernel. Returns null if generation fails.
        /// </summary>
        public static UnaryKernel? TryGetUnaryKernel(UnaryKernelKey key)
        {
            if (!Enabled)
                return null;

            try
            {
                return _unaryCache.GetOrAdd(key, GenerateUnaryKernel);
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Clear the unary kernel cache.
        /// </summary>
        public static void ClearUnary() => _unaryCache.Clear();

        /// <summary>
        /// Generate a unary kernel for the specified key.
        /// </summary>
        private static UnaryKernel GenerateUnaryKernel(UnaryKernelKey key)
        {
            // UnaryKernel signature:
            // void(void* input, void* output, int* strides, int* shape, int ndim, int totalSize)
            var dm = new DynamicMethod(
                name: $"Unary_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*),
                    typeof(int*), typeof(int*),
                    typeof(int), typeof(int)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            int inputSize = GetTypeSize(key.InputType);
            int outputSize = GetTypeSize(key.OutputType);

            if (key.IsContiguous)
            {
                // Check if we can use SIMD for this operation
                bool canSimd = CanUseUnarySimd(key);
                if (canSimd)
                {
                    EmitUnarySimdLoop(il, key, inputSize, outputSize);
                }
                else
                {
                    EmitUnaryScalarLoop(il, key, inputSize, outputSize);
                }
            }
            else
            {
                EmitUnaryStridedLoop(il, key, inputSize, outputSize);
            }

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<UnaryKernel>();
        }

        /// <summary>
        /// Check if SIMD can be used for this unary operation.
        /// </summary>
        private static bool CanUseUnarySimd(UnaryKernelKey key)
        {
            // SIMD only for same-type operations on float/double
            if (!key.IsSameType)
                return false;

            // Only float and double have good SIMD support for unary ops
            if (key.InputType != NPTypeCode.Single && key.InputType != NPTypeCode.Double)
                return false;

            // Only certain operations have SIMD support
            return key.Op == UnaryOp.Negate || key.Op == UnaryOp.Abs || key.Op == UnaryOp.Sqrt ||
                   key.Op == UnaryOp.Floor || key.Op == UnaryOp.Ceil || key.Op == UnaryOp.Round;
        }

        /// <summary>
        /// Emit SIMD loop for contiguous unary operations.
        /// </summary>
        private static void EmitUnarySimdLoop(ILGenerator il, UnaryKernelKey key,
            int inputSize, int outputSize)
        {
            int vectorCount = GetVectorCount(key.InputType);

            var locI = il.DeclareLocal(typeof(int)); // loop counter
            var locVectorEnd = il.DeclareLocal(typeof(int)); // totalSize - vectorCount

            var lblSimdLoop = il.DefineLabel();
            var lblSimdLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();

            // vectorEnd = totalSize - vectorCount
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            // === SIMD LOOP ===
            il.MarkLabel(lblSimdLoop);

            // if (i > vectorEnd) goto SimdLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblSimdLoopEnd);

            // Load input vector
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.InputType);

            // Vector operation
            EmitUnaryVectorOperation(il, key.Op, key.InputType);

            // Store result vector
            il.Emit(OpCodes.Ldarg_1); // output
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, outputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorStore(il, key.OutputType);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblSimdLoop);
            il.MarkLabel(lblSimdLoopEnd);

            // === TAIL LOOP ===
            il.MarkLabel(lblTailLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Bge, lblTailLoopEnd);

            // output[i] = op(input[i])
            il.Emit(OpCodes.Ldarg_1); // output
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, outputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);
            EmitConvertTo(il, key.InputType, key.OutputType);

            EmitUnaryScalarOperation(il, key.Op, key.OutputType);
            EmitStoreIndirect(il, key.OutputType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTailLoop);
            il.MarkLabel(lblTailLoopEnd);
        }

        /// <summary>
        /// Emit scalar loop for contiguous unary operations (no SIMD).
        /// </summary>
        private static void EmitUnaryScalarLoop(ILGenerator il, UnaryKernelKey key,
            int inputSize, int outputSize)
        {
            // Args: void* input (0), void* output (1),
            //       int* strides (2), int* shape (3),
            //       int ndim (4), int totalSize (5)

            var locI = il.DeclareLocal(typeof(int)); // loop counter

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // output[i] = op(input[i])
            // Load output address
            il.Emit(OpCodes.Ldarg_1); // output
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, outputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            // Load input[i] and convert to output type
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);
            EmitConvertTo(il, key.InputType, key.OutputType);

            // Perform operation
            EmitUnaryScalarOperation(il, key.Op, key.OutputType);

            // Store result
            EmitStoreIndirect(il, key.OutputType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Emit strided loop for non-contiguous unary operations.
        /// Uses coordinate-based iteration.
        /// </summary>
        private static void EmitUnaryStridedLoop(ILGenerator il, UnaryKernelKey key,
            int inputSize, int outputSize)
        {
            // Args: void* input (0), void* output (1),
            //       int* strides (2), int* shape (3),
            //       int ndim (4), int totalSize (5)

            var locI = il.DeclareLocal(typeof(int)); // linear index
            var locD = il.DeclareLocal(typeof(int)); // dimension counter
            var locInputOffset = il.DeclareLocal(typeof(int)); // input offset
            var locCoord = il.DeclareLocal(typeof(int)); // current coordinate
            var locIdx = il.DeclareLocal(typeof(int)); // temp for coordinate calculation

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();
            var lblDimLoop = il.DefineLabel();
            var lblDimLoopEnd = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            // Main loop
            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // Calculate inputOffset from linear index
            // inputOffset = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locInputOffset);

            // idx = i (for coordinate calculation)
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Stloc, locIdx);

            // For each dimension (right to left): coord = idx % shape[d], idx /= shape[d]
            // d = ndim - 1
            il.Emit(OpCodes.Ldarg_S, (byte)4); // ndim
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);

            il.MarkLabel(lblDimLoop);

            // if (d < 0) goto DimLoopEnd
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Blt, lblDimLoopEnd);

            // coord = idx % shape[d]
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg_3); // shape
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_4); // sizeof(int)
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I4);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stloc, locCoord);

            // idx /= shape[d]
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg_3); // shape
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I4);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locIdx);

            // inputOffset += coord * strides[d]
            il.Emit(OpCodes.Ldloc, locInputOffset);
            il.Emit(OpCodes.Ldloc, locCoord);
            il.Emit(OpCodes.Ldarg_2); // strides
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locInputOffset);

            // d--
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);

            il.Emit(OpCodes.Br, lblDimLoop);
            il.MarkLabel(lblDimLoopEnd);

            // Now compute: output[i] = op(input[inputOffset])
            // Load output address (contiguous output)
            il.Emit(OpCodes.Ldarg_1); // output
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, outputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            // Load input[inputOffset]
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locInputOffset);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);
            EmitConvertTo(il, key.InputType, key.OutputType);

            // Operation
            EmitUnaryScalarOperation(il, key.Op, key.OutputType);

            // Store
            EmitStoreIndirect(il, key.OutputType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Emit unary scalar operation.
        /// </summary>
        internal static void EmitUnaryScalarOperation(ILGenerator il, UnaryOp op, NPTypeCode type)
        {
            // Special handling for decimal
            if (type == NPTypeCode.Decimal)
            {
                EmitUnaryDecimalOperation(il, op);
                return;
            }

            switch (op)
            {
                case UnaryOp.Negate:
                    // For unsigned types, use two's complement: ~val + 1
                    // For signed types, use neg opcode
                    if (IsUnsigned(type))
                    {
                        // ~val + 1 = two's complement negation
                        il.Emit(OpCodes.Not);
                        il.Emit(OpCodes.Ldc_I4_1);
                        // Need to widen to correct type before add
                        if (type == NPTypeCode.UInt64)
                        {
                            il.Emit(OpCodes.Conv_U8);
                        }
                        il.Emit(OpCodes.Add);
                    }
                    else
                    {
                        il.Emit(OpCodes.Neg);
                    }
                    break;

                case UnaryOp.Abs:
                    EmitMathCall(il, "Abs", type);
                    break;

                case UnaryOp.Sqrt:
                    EmitMathCall(il, "Sqrt", type);
                    break;

                case UnaryOp.Exp:
                    EmitMathCall(il, "Exp", type);
                    break;

                case UnaryOp.Log:
                    EmitMathCall(il, "Log", type);
                    break;

                case UnaryOp.Sin:
                    EmitMathCall(il, "Sin", type);
                    break;

                case UnaryOp.Cos:
                    EmitMathCall(il, "Cos", type);
                    break;

                case UnaryOp.Tan:
                    EmitMathCall(il, "Tan", type);
                    break;

                case UnaryOp.Sinh:
                    EmitMathCall(il, "Sinh", type);
                    break;

                case UnaryOp.Cosh:
                    EmitMathCall(il, "Cosh", type);
                    break;

                case UnaryOp.Tanh:
                    EmitMathCall(il, "Tanh", type);
                    break;

                case UnaryOp.ASin:
                    EmitMathCall(il, "Asin", type);
                    break;

                case UnaryOp.ACos:
                    EmitMathCall(il, "Acos", type);
                    break;

                case UnaryOp.ATan:
                    EmitMathCall(il, "Atan", type);
                    break;

                case UnaryOp.Exp2:
                    // Use Math.Pow(2, x) since Math.Exp2 may not be available
                    EmitExp2Call(il, type);
                    break;

                case UnaryOp.Expm1:
                    // exp(x) - 1: call Exp then subtract 1
                    EmitMathCall(il, "Exp", type);
                    EmitSubtractOne(il, type);
                    break;

                case UnaryOp.Log2:
                    EmitMathCall(il, "Log2", type);
                    break;

                case UnaryOp.Log10:
                    EmitMathCall(il, "Log10", type);
                    break;

                case UnaryOp.Log1p:
                    // log(1 + x): add 1 then call Log
                    EmitAddOne(il, type);
                    EmitMathCall(il, "Log", type);
                    break;

                case UnaryOp.Sign:
                    EmitSignCall(il, type);
                    break;

                case UnaryOp.Ceil:
                    EmitMathCall(il, "Ceiling", type);
                    break;

                case UnaryOp.Floor:
                    EmitMathCall(il, "Floor", type);
                    break;

                case UnaryOp.Round:
                    EmitMathCall(il, "Round", type);
                    break;

                default:
                    throw new NotSupportedException($"Unary operation {op} not supported");
            }
        }

        /// <summary>
        /// Emit call to Math.X method with appropriate overload.
        /// </summary>
        private static void EmitMathCall(ILGenerator il, string methodName, NPTypeCode type)
        {
            MethodInfo? method;

            if (type == NPTypeCode.Single)
            {
                // Use MathF for float
                method = typeof(MathF).GetMethod(methodName, new[] { typeof(float) });
            }
            else if (type == NPTypeCode.Double)
            {
                // Use Math for double
                method = typeof(Math).GetMethod(methodName, new[] { typeof(double) });
            }
            else
            {
                // For integer types, convert to double, call Math, convert back
                // Stack has: value (as output type)
                // Need to: conv to double, call Math.X, conv back

                // Convert to double first
                EmitConvertToDouble(il, type);

                // Call Math.X(double)
                method = typeof(Math).GetMethod(methodName, new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);

                // Convert back to target type
                EmitConvertFromDouble(il, type);
                return;
            }

            il.EmitCall(OpCodes.Call, method!, null);
        }

        /// <summary>
        /// Convert stack value to double.
        /// </summary>
        private static void EmitConvertToDouble(ILGenerator il, NPTypeCode from)
        {
            if (from == NPTypeCode.Double)
                return;

            if (IsUnsigned(from))
                il.Emit(OpCodes.Conv_R_Un);
            il.Emit(OpCodes.Conv_R8);
        }

        /// <summary>
        /// Convert double on stack to target type.
        /// </summary>
        private static void EmitConvertFromDouble(ILGenerator il, NPTypeCode to)
        {
            if (to == NPTypeCode.Double)
                return;

            switch (to)
            {
                case NPTypeCode.Boolean:
                    il.Emit(OpCodes.Ldc_R8, 0.0);
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
                    il.Emit(OpCodes.Conv_I8);
                    break;
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Conv_U8);
                    break;
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Conv_R4);
                    break;
                default:
                    throw new NotSupportedException($"Conversion to {to} not supported");
            }
        }

        /// <summary>
        /// Emit 2^x calculation using Math.Pow(2, x).
        /// </summary>
        private static void EmitExp2Call(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                // For float: convert to double, call Pow, convert back
                il.Emit(OpCodes.Conv_R8);
                il.Emit(OpCodes.Ldc_R8, 2.0);
                // Stack: [exponent, base] - but Pow expects (base, exponent)
                // Need to swap them
                var locExp = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locExp);  // Save exponent
                // Now push base then exponent
                il.Emit(OpCodes.Ldc_R8, 2.0);
                il.Emit(OpCodes.Ldloc, locExp);
                il.EmitCall(OpCodes.Call, typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) })!, null);
                il.Emit(OpCodes.Conv_R4);
            }
            else if (type == NPTypeCode.Double)
            {
                // For double: just call Pow
                var locExp = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locExp);  // Save exponent
                il.Emit(OpCodes.Ldc_R8, 2.0);
                il.Emit(OpCodes.Ldloc, locExp);
                il.EmitCall(OpCodes.Call, typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) })!, null);
            }
            else
            {
                // For integer types: convert to double, call Pow, convert back
                EmitConvertToDouble(il, type);
                var locExp = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locExp);  // Save exponent
                il.Emit(OpCodes.Ldc_R8, 2.0);
                il.Emit(OpCodes.Ldloc, locExp);
                il.EmitCall(OpCodes.Call, typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) })!, null);
                EmitConvertFromDouble(il, type);
            }
        }

        /// <summary>
        /// Emit subtraction of 1 from the value on stack.
        /// Used for expm1 = exp(x) - 1.
        /// </summary>
        private static void EmitSubtractOne(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Ldc_R4, 1.0f);
                    il.Emit(OpCodes.Sub);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Ldc_R8, 1.0);
                    il.Emit(OpCodes.Sub);
                    break;
                default:
                    // For integer types, value is already double from math call
                    il.Emit(OpCodes.Ldc_R8, 1.0);
                    il.Emit(OpCodes.Sub);
                    break;
            }
        }

        /// <summary>
        /// Emit addition of 1 to the value on stack.
        /// Used for log1p = log(1 + x).
        /// </summary>
        private static void EmitAddOne(ILGenerator il, NPTypeCode type)
        {
            // Convert to appropriate float type first, then add 1
            if (type == NPTypeCode.Single)
            {
                il.Emit(OpCodes.Ldc_R4, 1.0f);
                il.Emit(OpCodes.Add);
            }
            else if (type == NPTypeCode.Double)
            {
                il.Emit(OpCodes.Ldc_R8, 1.0);
                il.Emit(OpCodes.Add);
            }
            else
            {
                // For integer types, convert to double first, then add 1
                // The conversion to double will happen in EmitMathCall
                EmitConvertToDouble(il, type);
                il.Emit(OpCodes.Ldc_R8, 1.0);
                il.Emit(OpCodes.Add);
            }
        }

        /// <summary>
        /// Emit Math.Sign call with proper type conversion.
        /// Math.Sign returns int, so we need to convert back to target type.
        /// NumPy: sign(NaN) returns NaN, but .NET Math.Sign throws ArithmeticException.
        /// We check for NaN first and return it directly.
        /// </summary>
        private static void EmitSignCall(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                // NumPy: sign(NaN) = NaN. .NET MathF.Sign(NaN) throws.
                // Check for NaN first: if (float.IsNaN(x)) return x; else return MathF.Sign(x);
                var lblNotNaN = il.DefineLabel();
                var lblEnd = il.DefineLabel();

                il.Emit(OpCodes.Dup);  // duplicate for NaN check
                il.EmitCall(OpCodes.Call, typeof(float).GetMethod("IsNaN", new[] { typeof(float) })!, null);
                il.Emit(OpCodes.Brfalse, lblNotNaN);

                // Is NaN - value is already on stack, jump to end
                il.Emit(OpCodes.Br, lblEnd);

                il.MarkLabel(lblNotNaN);
                // Not NaN - call MathF.Sign
                var method = typeof(MathF).GetMethod("Sign", new[] { typeof(float) });
                il.EmitCall(OpCodes.Call, method!, null);
                il.Emit(OpCodes.Conv_R4);

                il.MarkLabel(lblEnd);
            }
            else if (type == NPTypeCode.Double)
            {
                // NumPy: sign(NaN) = NaN. .NET Math.Sign(NaN) throws.
                // Check for NaN first: if (double.IsNaN(x)) return x; else return Math.Sign(x);
                var lblNotNaN = il.DefineLabel();
                var lblEnd = il.DefineLabel();

                il.Emit(OpCodes.Dup);  // duplicate for NaN check
                il.EmitCall(OpCodes.Call, typeof(double).GetMethod("IsNaN", new[] { typeof(double) })!, null);
                il.Emit(OpCodes.Brfalse, lblNotNaN);

                // Is NaN - value is already on stack, jump to end
                il.Emit(OpCodes.Br, lblEnd);

                il.MarkLabel(lblNotNaN);
                // Not NaN - call Math.Sign
                var method = typeof(Math).GetMethod("Sign", new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);
                il.Emit(OpCodes.Conv_R8);

                il.MarkLabel(lblEnd);
            }
            else if (type == NPTypeCode.Decimal)
            {
                // Decimal has its own Sign method that returns int
                var method = typeof(Math).GetMethod("Sign", new[] { typeof(decimal) });
                il.EmitCall(OpCodes.Call, method!, null);
                // Convert int to decimal
                il.EmitCall(OpCodes.Call, typeof(decimal).GetMethod("op_Implicit", new[] { typeof(int) })!, null);
            }
            else
            {
                // For integer types: convert to double, call Math.Sign, convert back
                EmitConvertToDouble(il, type);
                var method = typeof(Math).GetMethod("Sign", new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);
                // Convert int result back to target type
                EmitConvertFromInt(il, type);
            }
        }

        /// <summary>
        /// Convert int on stack to target type.
        /// </summary>
        private static void EmitConvertFromInt(ILGenerator il, NPTypeCode to)
        {
            switch (to)
            {
                case NPTypeCode.Boolean:
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
                    // Already int, no conversion needed
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
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Conv_R8);
                    break;
                default:
                    throw new NotSupportedException($"Conversion from int to {to} not supported");
            }
        }

        /// <summary>
        /// Emit unary operation for decimal type.
        /// </summary>
        private static void EmitUnaryDecimalOperation(ILGenerator il, UnaryOp op)
        {
            switch (op)
            {
                case UnaryOp.Negate:
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("op_UnaryNegation", new[] { typeof(decimal) })!,
                        null);
                    break;

                case UnaryOp.Abs:
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Abs", new[] { typeof(decimal) })!,
                        null);
                    break;

                case UnaryOp.Sign:
                    // Math.Sign(decimal) returns int, convert back to decimal
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Sign", new[] { typeof(decimal) })!,
                        null);
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("op_Implicit", new[] { typeof(int) })!,
                        null);
                    break;

                case UnaryOp.Ceil:
                    // Math.Ceiling has decimal overload
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Ceiling", new[] { typeof(decimal) })!,
                        null);
                    break;

                case UnaryOp.Floor:
                    // Math.Floor has decimal overload
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Floor", new[] { typeof(decimal) })!,
                        null);
                    break;

                case UnaryOp.Round:
                    // Math.Round has decimal overload
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Round", new[] { typeof(decimal) })!,
                        null);
                    break;

                case UnaryOp.Sqrt:
                case UnaryOp.Exp:
                case UnaryOp.Log:
                case UnaryOp.Sin:
                case UnaryOp.Cos:
                case UnaryOp.Tan:
                case UnaryOp.Sinh:
                case UnaryOp.Cosh:
                case UnaryOp.Tanh:
                case UnaryOp.ASin:
                case UnaryOp.ACos:
                case UnaryOp.ATan:
                case UnaryOp.Log2:
                case UnaryOp.Log10:
                    // Convert to double, perform operation, convert back
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("ToDouble", new[] { typeof(decimal) })!,
                        null);

                    string mathMethod = op switch
                    {
                        UnaryOp.Sqrt => "Sqrt",
                        UnaryOp.Exp => "Exp",
                        UnaryOp.Log => "Log",
                        UnaryOp.Sin => "Sin",
                        UnaryOp.Cos => "Cos",
                        UnaryOp.Tan => "Tan",
                        UnaryOp.Sinh => "Sinh",
                        UnaryOp.Cosh => "Cosh",
                        UnaryOp.Tanh => "Tanh",
                        UnaryOp.ASin => "Asin",
                        UnaryOp.ACos => "Acos",
                        UnaryOp.ATan => "Atan",
                        UnaryOp.Log2 => "Log2",
                        UnaryOp.Log10 => "Log10",
                        _ => throw new NotSupportedException()
                    };

                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod(mathMethod, new[] { typeof(double) })!,
                        null);

                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("op_Explicit", new[] { typeof(double) })!,
                        null);
                    break;

                case UnaryOp.Exp2:
                    // 2^x for decimal: convert to double, use Math.Pow, convert back
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("ToDouble", new[] { typeof(decimal) })!,
                        null);
                    // Stack: [exponent (double)] - need to call Pow(2, exponent)
                    var locExpDec = il.DeclareLocal(typeof(double));
                    il.Emit(OpCodes.Stloc, locExpDec);
                    il.Emit(OpCodes.Ldc_R8, 2.0);
                    il.Emit(OpCodes.Ldloc, locExpDec);
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) })!,
                        null);
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("op_Explicit", new[] { typeof(double) })!,
                        null);
                    break;

                case UnaryOp.Expm1:
                    // exp(x) - 1 for decimal
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("ToDouble", new[] { typeof(decimal) })!,
                        null);
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Exp", new[] { typeof(double) })!,
                        null);
                    il.Emit(OpCodes.Ldc_R8, 1.0);
                    il.Emit(OpCodes.Sub);
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("op_Explicit", new[] { typeof(double) })!,
                        null);
                    break;

                case UnaryOp.Log1p:
                    // log(1 + x) for decimal
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("ToDouble", new[] { typeof(decimal) })!,
                        null);
                    il.Emit(OpCodes.Ldc_R8, 1.0);
                    il.Emit(OpCodes.Add);
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Log", new[] { typeof(double) })!,
                        null);
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("op_Explicit", new[] { typeof(double) })!,
                        null);
                    break;

                default:
                    throw new NotSupportedException($"Unary operation {op} not supported for decimal");
            }
        }

        /// <summary>
        /// Emit Vector unary operation (adapts to V128/V256/V512).
        /// </summary>
        private static void EmitUnaryVectorOperation(ILGenerator il, UnaryOp op, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);
            var vectorType = GetVectorType(clrType);

            string methodName = op switch
            {
                UnaryOp.Negate => "op_UnaryNegation",
                UnaryOp.Abs => "Abs",
                UnaryOp.Sqrt => "Sqrt",
                UnaryOp.Floor => "Floor",
                UnaryOp.Ceil => "Ceiling",  // Vector uses "Ceiling" not "Ceil"
                UnaryOp.Round => "Round",
                _ => throw new NotSupportedException($"SIMD operation {op} not supported")
            };

            MethodInfo? method;

            if (op == UnaryOp.Negate)
            {
                // Negation is an operator on Vector<T>
                method = vectorType.GetMethod(methodName, BindingFlags.Public | BindingFlags.Static,
                    null, new[] { vectorType }, null);
            }
            else if (op == UnaryOp.Floor || op == UnaryOp.Ceil || op == UnaryOp.Round)
            {
                // Floor/Ceiling/Round are NOT generic - they're overloaded for specific types
                // Use the single-parameter overload (default MidpointRounding.ToEven for Round)
                method = containerType.GetMethod(methodName, BindingFlags.Public | BindingFlags.Static,
                    null, new[] { vectorType }, null);
            }
            else
            {
                // Abs, Sqrt are generic static methods on Vector container
                method = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Where(m => m.Name == methodName && m.IsGenericMethod && m.GetParameters().Length == 1)
                    .Select(m => m.MakeGenericMethod(clrType))
                    .FirstOrDefault(m => m.GetParameters()[0].ParameterType == vectorType);
            }

            if (method == null)
                throw new InvalidOperationException($"Could not find {methodName} for {vectorType.Name}");

            il.EmitCall(OpCodes.Call, method, null);
        }

        #endregion

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

            // Convert to output type if different
            EmitConvertTo(il, key.InputType, key.OutputType);

            // Perform the unary operation (result is on stack)
            EmitUnaryScalarOperation(il, key.Op, key.OutputType);

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
