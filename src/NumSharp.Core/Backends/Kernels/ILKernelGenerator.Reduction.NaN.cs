using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Reduction.NaN.cs - IL-Generated NaN Reduction Kernels
// =============================================================================
//
// RESPONSIBILITY:
//   - IL-generated element-wise NaN reductions (NanMean, NanVar, NanStd)
//   - SIMD loop emission with NaN masking via Equals(vec, vec)
//   - Sum and count tracking for NaN-aware statistics
//   - Two-pass variance algorithm (mean, then squared differences)
//
// KEY SIMD PATTERN:
//   nanMask = Equals(vec, vec)   // True for non-NaN, false for NaN
//   cleaned = BitwiseAnd(vec, nanMask)  // Zero out NaN values
//   sumVec += cleaned
//   countVec += BitwiseAnd(oneVec, nanMask)  // Count non-NaN elements
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public sealed partial class ILKernelGenerator
    {
        #region NaN Element Reduction IL Generation

        /// <summary>
        /// Cache for NaN element reduction kernels.
        /// </summary>
        private static readonly ConcurrentDictionary<ElementReductionKernelKey, Delegate> _nanElementReductionCache = new();

        /// <summary>
        /// Number of NaN element reduction kernels in cache.
        /// </summary>
        public static int NanElementReductionCachedCount => _nanElementReductionCache.Count;

        /// <summary>
        /// Try to get an IL-generated NaN element reduction kernel.
        /// Only supports float and double types (NaN is only defined for floating-point).
        /// </summary>
        public static TypedElementReductionKernel<TResult>? TryGetNanElementReductionKernel<TResult>(ElementReductionKernelKey key)
            where TResult : unmanaged
        {
            if (!Enabled)
                return null;

            // Only NaN operations
            if (key.Op != ReductionOp.NanSum && key.Op != ReductionOp.NanProd &&
                key.Op != ReductionOp.NanMin && key.Op != ReductionOp.NanMax &&
                key.Op != ReductionOp.NanMean && key.Op != ReductionOp.NanVar &&
                key.Op != ReductionOp.NanStd)
            {
                return null;
            }

            // NaN is only defined for float and double
            if (key.InputType != NPTypeCode.Single && key.InputType != NPTypeCode.Double)
            {
                return null;
            }

            try
            {
                var kernel = _nanElementReductionCache.GetOrAdd(key, GenerateNanElementReductionKernel<TResult>);
                return (TypedElementReductionKernel<TResult>)kernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryGetNanElementReductionKernel<{typeof(TResult).Name}>({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Generate an IL-based NaN element reduction kernel.
        /// </summary>
        private static Delegate GenerateNanElementReductionKernel<TResult>(ElementReductionKernelKey key)
            where TResult : unmanaged
        {
            // TypedElementReductionKernel<TResult> signature:
            // TResult(void* input, long* strides, long* shape, int ndim, long totalSize)
            var dm = new DynamicMethod(
                name: $"NanElemReduce_{key}",
                returnType: typeof(TResult),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(long*), typeof(long*), typeof(int), typeof(long)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();
            int inputSize = GetTypeSize(key.InputType);

            if (key.IsContiguous)
            {
                // NanMean, NanVar, NanStd need sum+count tracking
                if (key.Op == ReductionOp.NanMean || key.Op == ReductionOp.NanVar || key.Op == ReductionOp.NanStd)
                {
                    EmitNanStatSimdLoop(il, key, inputSize);
                }
                else
                {
                    // NanSum, NanProd, NanMin, NanMax use standard masking
                    EmitNanReductionSimdLoop(il, key, inputSize);
                }
            }
            else
            {
                // Strided path - use scalar loop
                if (key.Op == ReductionOp.NanMean || key.Op == ReductionOp.NanVar || key.Op == ReductionOp.NanStd)
                {
                    EmitNanStatStridedLoop(il, key, inputSize);
                }
                else
                {
                    EmitNanReductionStridedLoop(il, key, inputSize);
                }
            }

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<TypedElementReductionKernel<TResult>>();
        }

        #endregion

        #region NaN Statistics SIMD Loop Emission (NanMean, NanVar, NanStd)

        /// <summary>
        /// Emit IL for NaN statistics SIMD loop (NanMean, NanVar, NanStd).
        /// Uses sum and count tracking with NaN masking.
        /// For NanVar/NanStd, uses two-pass algorithm.
        /// </summary>
        private static void EmitNanStatSimdLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            // Args: void* input (0), long* strides (1), long* shape (2), int ndim (3), long totalSize (4)

            var clrType = GetClrType(key.InputType);
            var vectorType = GetVectorType(clrType);
            int vectorCount = GetVectorCount(key.InputType);

            // Locals
            var locI = il.DeclareLocal(typeof(long));           // Loop counter
            var locVectorEnd = il.DeclareLocal(typeof(long));   // totalSize - vectorCount
            var locSum = il.DeclareLocal(clrType);              // Scalar sum accumulator
            var locCount = il.DeclareLocal(clrType);            // Scalar count accumulator
            var locSumVec = il.DeclareLocal(vectorType);        // Vector sum accumulator
            var locCountVec = il.DeclareLocal(vectorType);      // Vector count accumulator
            var locOneVec = il.DeclareLocal(vectorType);        // Vector of 1.0s
            var locVec = il.DeclareLocal(vectorType);           // Loaded vector
            var locNanMask = il.DeclareLocal(vectorType);       // NaN mask result

            // For NanVar/NanStd: additional locals
            LocalBuilder? locMean = null;
            LocalBuilder? locSqDiffSum = null;
            LocalBuilder? locSqDiffVec = null;
            LocalBuilder? locMeanVec = null;
            LocalBuilder? locDiffVec = null;

            if (key.Op == ReductionOp.NanVar || key.Op == ReductionOp.NanStd)
            {
                locMean = il.DeclareLocal(clrType);
                locSqDiffSum = il.DeclareLocal(clrType);
                locSqDiffVec = il.DeclareLocal(vectorType);
                locMeanVec = il.DeclareLocal(vectorType);
                locDiffVec = il.DeclareLocal(vectorType);
            }

            var lblSimdLoop = il.DefineLabel();
            var lblSimdLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();
            var lblAllNaN = il.DefineLabel();
            var lblPass2SimdLoop = key.Op == ReductionOp.NanVar || key.Op == ReductionOp.NanStd ? il.DefineLabel() : default;
            var lblPass2SimdLoopEnd = key.Op == ReductionOp.NanVar || key.Op == ReductionOp.NanStd ? il.DefineLabel() : default;
            var lblPass2TailLoop = key.Op == ReductionOp.NanVar || key.Op == ReductionOp.NanStd ? il.DefineLabel() : default;
            var lblPass2TailLoopEnd = key.Op == ReductionOp.NanVar || key.Op == ReductionOp.NanStd ? il.DefineLabel() : default;

            // === Initialize ===
            // sumVec = Vector.Zero
            EmitLoadVectorZero(il, key.InputType);
            il.Emit(OpCodes.Stloc, locSumVec);

            // countVec = Vector.Zero
            EmitLoadVectorZero(il, key.InputType);
            il.Emit(OpCodes.Stloc, locCountVec);

            // oneVec = Vector.Create(1.0)
            EmitLoadOne(il, key.InputType);
            EmitVectorCreate(il, key.InputType);
            il.Emit(OpCodes.Stloc, locOneVec);

            // vectorEnd = totalSize - vectorCount
            il.Emit(OpCodes.Ldarg_S, (byte)4); // totalSize
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // === PASS 1: SIMD Loop for sum and count ===
            il.MarkLabel(lblSimdLoop);

            // if (i > vectorEnd) goto SimdLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblSimdLoopEnd);

            // vec = Vector.Load(input + i * inputSize)
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.InputType);
            il.Emit(OpCodes.Stloc, locVec);

            // nanMask = Equals(vec, vec)  -- true for non-NaN
            il.Emit(OpCodes.Ldloc, locVec);
            il.Emit(OpCodes.Ldloc, locVec);
            EmitVectorEquals(il, key.InputType);
            EmitVectorAsType(il, key.InputType); // Convert mask to same type
            il.Emit(OpCodes.Stloc, locNanMask);

            // cleaned = BitwiseAnd(vec, nanMask) -- zeros out NaN
            il.Emit(OpCodes.Ldloc, locVec);
            il.Emit(OpCodes.Ldloc, locNanMask);
            EmitVectorBitwiseAnd(il, key.InputType);

            // sumVec = Add(sumVec, cleaned)
            il.Emit(OpCodes.Ldloc, locSumVec);
            EmitVectorAdd(il, key.InputType);
            il.Emit(OpCodes.Stloc, locSumVec);

            // countMask = BitwiseAnd(oneVec, nanMask)
            il.Emit(OpCodes.Ldloc, locOneVec);
            il.Emit(OpCodes.Ldloc, locNanMask);
            EmitVectorBitwiseAnd(il, key.InputType);

            // countVec = Add(countVec, countMask)
            il.Emit(OpCodes.Ldloc, locCountVec);
            EmitVectorAdd(il, key.InputType);
            il.Emit(OpCodes.Stloc, locCountVec);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblSimdLoop);
            il.MarkLabel(lblSimdLoopEnd);

            // === Horizontal reduction ===
            // sum = Vector.Sum(sumVec)
            il.Emit(OpCodes.Ldloc, locSumVec);
            EmitVectorSum(il, key.InputType);
            il.Emit(OpCodes.Stloc, locSum);

            // count = Vector.Sum(countVec)
            il.Emit(OpCodes.Ldloc, locCountVec);
            EmitVectorSum(il, key.InputType);
            il.Emit(OpCodes.Stloc, locCount);

            // === PASS 1: Scalar tail loop ===
            il.MarkLabel(lblTailLoop);

            // if (i >= totalSize) goto TailLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Bge, lblTailLoopEnd);

            // Load input[i]
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);

            // Duplicate for IsNaN check
            il.Emit(OpCodes.Dup);

            // Check if NaN (x != x for NaN)
            il.Emit(OpCodes.Dup);
            var lblIsNaN = il.DefineLabel();
            var lblNotNaN = il.DefineLabel();

            // if (val != val) goto IsNaN (NaN case)
            if (key.InputType == NPTypeCode.Single)
            {
                il.Emit(OpCodes.Bne_Un, lblIsNaN);
            }
            else
            {
                il.Emit(OpCodes.Bne_Un, lblIsNaN);
            }

            // Not NaN: sum += val, count += 1
            il.Emit(OpCodes.Ldloc, locSum);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSum);

            il.Emit(OpCodes.Ldloc, locCount);
            EmitLoadOne(il, key.InputType);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locCount);
            il.Emit(OpCodes.Br, lblNotNaN);

            // IsNaN: pop the duplicated value
            il.MarkLabel(lblIsNaN);
            il.Emit(OpCodes.Pop);

            il.MarkLabel(lblNotNaN);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTailLoop);
            il.MarkLabel(lblTailLoopEnd);

            // === Check if all NaN ===
            // if (count == 0) return NaN
            il.Emit(OpCodes.Ldloc, locCount);
            EmitLoadZero(il, key.InputType);
            il.Emit(OpCodes.Beq, lblAllNaN);

            // For NanMean: return sum / count
            if (key.Op == ReductionOp.NanMean)
            {
                il.Emit(OpCodes.Ldloc, locSum);
                il.Emit(OpCodes.Ldloc, locCount);
                il.Emit(OpCodes.Div);
                il.Emit(OpCodes.Ret);

                // AllNaN path
                il.MarkLabel(lblAllNaN);
                EmitLoadNaN(il, key.InputType);
                return;
            }

            // For NanVar/NanStd: compute mean, then pass 2 for squared differences
            // mean = sum / count
            il.Emit(OpCodes.Ldloc, locSum);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locMean!);

            // === PASS 2: Squared differences ===
            // sqDiffVec = Vector.Zero
            EmitLoadVectorZero(il, key.InputType);
            il.Emit(OpCodes.Stloc, locSqDiffVec!);

            // meanVec = Vector.Create(mean)
            il.Emit(OpCodes.Ldloc, locMean!);
            EmitVectorCreate(il, key.InputType);
            il.Emit(OpCodes.Stloc, locMeanVec!);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // === PASS 2: SIMD Loop ===
            il.MarkLabel(lblPass2SimdLoop);

            // if (i > vectorEnd) goto Pass2SimdLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblPass2SimdLoopEnd);

            // vec = Vector.Load(input + i * inputSize)
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.InputType);
            il.Emit(OpCodes.Stloc, locVec);

            // nanMask = Equals(vec, vec)
            il.Emit(OpCodes.Ldloc, locVec);
            il.Emit(OpCodes.Ldloc, locVec);
            EmitVectorEquals(il, key.InputType);
            EmitVectorAsType(il, key.InputType);
            il.Emit(OpCodes.Stloc, locNanMask);

            // diff = Subtract(vec, meanVec)
            il.Emit(OpCodes.Ldloc, locVec);
            il.Emit(OpCodes.Ldloc, locMeanVec!);
            EmitVectorSubtract(il, key.InputType);
            il.Emit(OpCodes.Stloc, locDiffVec!);

            // sqDiff = Multiply(diff, diff)
            il.Emit(OpCodes.Ldloc, locDiffVec!);
            il.Emit(OpCodes.Ldloc, locDiffVec!);
            EmitVectorMultiply(il, key.InputType);

            // cleanedSqDiff = BitwiseAnd(sqDiff, nanMask)
            il.Emit(OpCodes.Ldloc, locNanMask);
            EmitVectorBitwiseAnd(il, key.InputType);

            // sqDiffVec = Add(sqDiffVec, cleanedSqDiff)
            il.Emit(OpCodes.Ldloc, locSqDiffVec!);
            EmitVectorAdd(il, key.InputType);
            il.Emit(OpCodes.Stloc, locSqDiffVec!);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblPass2SimdLoop);
            il.MarkLabel(lblPass2SimdLoopEnd);

            // sqDiffSum = Vector.Sum(sqDiffVec)
            il.Emit(OpCodes.Ldloc, locSqDiffVec!);
            EmitVectorSum(il, key.InputType);
            il.Emit(OpCodes.Stloc, locSqDiffSum!);

            // === PASS 2: Scalar tail loop ===
            il.MarkLabel(lblPass2TailLoop);

            // if (i >= totalSize) goto Pass2TailLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Bge, lblPass2TailLoopEnd);

            // Load input[i]
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);

            // Duplicate for IsNaN check
            il.Emit(OpCodes.Dup);

            // Check if NaN
            il.Emit(OpCodes.Dup);
            var lblIsNaN2 = il.DefineLabel();
            var lblNotNaN2 = il.DefineLabel();
            il.Emit(OpCodes.Bne_Un, lblIsNaN2);

            // Not NaN: diff = val - mean; sqDiffSum += diff * diff
            il.Emit(OpCodes.Ldloc, locMean!);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Ldloc, locSqDiffSum!);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSqDiffSum!);
            il.Emit(OpCodes.Br, lblNotNaN2);

            // IsNaN: pop
            il.MarkLabel(lblIsNaN2);
            il.Emit(OpCodes.Pop);

            il.MarkLabel(lblNotNaN2);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblPass2TailLoop);
            il.MarkLabel(lblPass2TailLoopEnd);

            // variance = sqDiffSum / count
            il.Emit(OpCodes.Ldloc, locSqDiffSum!);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Div);

            // For NanStd: return sqrt(variance)
            if (key.Op == ReductionOp.NanStd)
            {
                EmitSqrt(il, key.InputType);
            }

            il.Emit(OpCodes.Ret);

            // AllNaN path
            il.MarkLabel(lblAllNaN);
            EmitLoadNaN(il, key.InputType);
        }

        /// <summary>
        /// Emit IL for NaN reduction SIMD loop (NanSum, NanProd, NanMin, NanMax).
        /// </summary>
        private static void EmitNanReductionSimdLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            // Similar to EmitReductionSimdLoop but with NaN masking
            var clrType = GetClrType(key.InputType);
            var vectorType = GetVectorType(clrType);
            int vectorCount = GetVectorCount(key.InputType);

            var locI = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));
            var locAccum = il.DeclareLocal(clrType);
            var locVecAccum = il.DeclareLocal(vectorType);
            var locVec = il.DeclareLocal(vectorType);
            var locNanMask = il.DeclareLocal(vectorType);
            var locIdentityVec = il.DeclareLocal(vectorType);

            var lblSimdLoop = il.DefineLabel();
            var lblSimdLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();

            // Get identity value for the operation
            var nanIdentity = GetNanReductionIdentity(key.Op, key.InputType);

            // Initialize vector accumulator with identity
            EmitLoadConstant(il, nanIdentity, key.InputType);
            EmitVectorCreate(il, key.InputType);
            il.Emit(OpCodes.Stloc, locVecAccum);

            // Store identity vector for ConditionalSelect
            EmitLoadConstant(il, nanIdentity, key.InputType);
            EmitVectorCreate(il, key.InputType);
            il.Emit(OpCodes.Stloc, locIdentityVec);

            // vectorEnd = totalSize - vectorCount
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // === SIMD Loop ===
            il.MarkLabel(lblSimdLoop);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblSimdLoopEnd);

            // Load vector
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.InputType);
            il.Emit(OpCodes.Stloc, locVec);

            // Create NaN mask
            il.Emit(OpCodes.Ldloc, locVec);
            il.Emit(OpCodes.Ldloc, locVec);
            EmitVectorEquals(il, key.InputType);
            il.Emit(OpCodes.Stloc, locNanMask);

            // For Sum: use BitwiseAnd to zero NaN; for others use ConditionalSelect
            if (key.Op == ReductionOp.NanSum)
            {
                il.Emit(OpCodes.Ldloc, locVec);
                il.Emit(OpCodes.Ldloc, locNanMask);
                EmitVectorAsType(il, key.InputType);
                EmitVectorBitwiseAnd(il, key.InputType);
            }
            else
            {
                // ConditionalSelect(mask, vec, identity)
                il.Emit(OpCodes.Ldloc, locNanMask);
                il.Emit(OpCodes.Ldloc, locVec);
                il.Emit(OpCodes.Ldloc, locIdentityVec);
                EmitVectorConditionalSelect(il, key.InputType);
            }

            // Combine with accumulator
            il.Emit(OpCodes.Ldloc, locVecAccum);
            EmitVectorBinaryReductionOp(il, GetBaseOp(key.Op), key.InputType);
            il.Emit(OpCodes.Stloc, locVecAccum);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblSimdLoop);
            il.MarkLabel(lblSimdLoopEnd);

            // Horizontal reduction
            il.Emit(OpCodes.Ldloc, locVecAccum);
            EmitVectorHorizontalReduction(il, GetBaseOp(key.Op), key.InputType);
            il.Emit(OpCodes.Stloc, locAccum);

            // === Scalar tail ===
            il.MarkLabel(lblTailLoop);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Bge, lblTailLoopEnd);

            // Load input[i]
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);

            // Check NaN and combine
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Dup);
            var lblIsNaN = il.DefineLabel();
            var lblNotNaN = il.DefineLabel();
            il.Emit(OpCodes.Bne_Un, lblIsNaN);

            // Not NaN: combine
            il.Emit(OpCodes.Ldloc, locAccum);
            EmitReductionCombine(il, GetBaseOp(key.Op), key.InputType);
            il.Emit(OpCodes.Stloc, locAccum);
            il.Emit(OpCodes.Br, lblNotNaN);

            il.MarkLabel(lblIsNaN);
            il.Emit(OpCodes.Pop);

            il.MarkLabel(lblNotNaN);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTailLoop);
            il.MarkLabel(lblTailLoopEnd);

            // Return accumulator
            il.Emit(OpCodes.Ldloc, locAccum);
        }

        /// <summary>
        /// Emit IL for strided NaN stat reduction.
        /// </summary>
        private static void EmitNanStatStridedLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            // For strided, use scalar loop with coordinate calculation
            // This is similar to EmitReductionStridedLoop but with NaN handling and count tracking
            EmitNanReductionStridedLoop(il, key, inputSize); // For now, delegate to simpler implementation
        }

        /// <summary>
        /// Emit IL for strided NaN reduction (scalar loop with coordinate calculation).
        /// </summary>
        private static void EmitNanReductionStridedLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            // Similar to EmitReductionStridedLoop but with NaN handling
            // For brevity, use scalar loop
            var locI = il.DeclareLocal(typeof(long));
            var locOffset = il.DeclareLocal(typeof(long));
            var locD = il.DeclareLocal(typeof(int));
            var locCoord = il.DeclareLocal(typeof(long));
            var locIdx = il.DeclareLocal(typeof(long));
            var locAccum = il.DeclareLocal(GetClrType(key.AccumulatorType));
            var locCount = il.DeclareLocal(GetClrType(key.AccumulatorType));

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();
            var lblDimLoop = il.DefineLabel();
            var lblDimLoopEnd = il.DefineLabel();

            // Initialize accumulator
            if (key.Op == ReductionOp.NanMean || key.Op == ReductionOp.NanVar || key.Op == ReductionOp.NanStd)
            {
                EmitLoadZero(il, key.AccumulatorType);
                il.Emit(OpCodes.Stloc, locAccum);
                EmitLoadZero(il, key.AccumulatorType);
                il.Emit(OpCodes.Stloc, locCount);
            }
            else
            {
                var identity = GetNanReductionIdentity(key.Op, key.AccumulatorType);
                EmitLoadConstant(il, identity, key.AccumulatorType);
                il.Emit(OpCodes.Stloc, locAccum);
            }

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // Calculate offset from linear index (same as EmitReductionStridedLoop)
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locOffset);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Stloc, locIdx);

            // d = ndim - 1
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);

            il.MarkLabel(lblDimLoop);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Blt, lblDimLoopEnd);

            // coord = idx % shape[d]
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stloc, locCoord);

            // idx /= shape[d]
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locIdx);

            // offset += coord * strides[d]
            il.Emit(OpCodes.Ldloc, locOffset);
            il.Emit(OpCodes.Ldloc, locCoord);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOffset);

            // d--
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);

            il.Emit(OpCodes.Br, lblDimLoop);
            il.MarkLabel(lblDimLoopEnd);

            // Load input[offset]
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locOffset);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);
            EmitConvertTo(il, key.InputType, key.AccumulatorType);

            // Check NaN
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Dup);
            var lblIsNaN = il.DefineLabel();
            var lblNotNaN = il.DefineLabel();
            il.Emit(OpCodes.Bne_Un, lblIsNaN);

            // Not NaN: combine
            il.Emit(OpCodes.Ldloc, locAccum);
            EmitReductionCombine(il, GetBaseOp(key.Op), key.AccumulatorType);
            il.Emit(OpCodes.Stloc, locAccum);

            if (key.Op == ReductionOp.NanMean || key.Op == ReductionOp.NanVar || key.Op == ReductionOp.NanStd)
            {
                il.Emit(OpCodes.Ldloc, locCount);
                EmitLoadOne(il, key.AccumulatorType);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locCount);
            }

            il.Emit(OpCodes.Br, lblNotNaN);

            il.MarkLabel(lblIsNaN);
            il.Emit(OpCodes.Pop);

            il.MarkLabel(lblNotNaN);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);

            // For NanMean: divide by count
            if (key.Op == ReductionOp.NanMean)
            {
                var lblAllNaN = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, locCount);
                EmitLoadZero(il, key.AccumulatorType);
                il.Emit(OpCodes.Beq, lblAllNaN);

                il.Emit(OpCodes.Ldloc, locAccum);
                il.Emit(OpCodes.Ldloc, locCount);
                il.Emit(OpCodes.Div);
                il.Emit(OpCodes.Ret);

                il.MarkLabel(lblAllNaN);
                EmitLoadNaN(il, key.AccumulatorType);
            }
            else
            {
                il.Emit(OpCodes.Ldloc, locAccum);
            }
        }

        #endregion

        #region NaN IL Helpers

        /// <summary>
        /// Get identity value for NaN reduction operations.
        /// </summary>
        private static object GetNanReductionIdentity(ReductionOp op, NPTypeCode type)
        {
            return op switch
            {
                ReductionOp.NanSum => type == NPTypeCode.Single ? 0f : 0.0,
                ReductionOp.NanProd => type == NPTypeCode.Single ? 1f : 1.0,
                ReductionOp.NanMin => type == NPTypeCode.Single ? float.PositiveInfinity : double.PositiveInfinity,
                ReductionOp.NanMax => type == NPTypeCode.Single ? float.NegativeInfinity : double.NegativeInfinity,
                _ => type == NPTypeCode.Single ? 0f : 0.0
            };
        }

        /// <summary>
        /// Convert NaN operation to base operation for SIMD combine.
        /// </summary>
        private static ReductionOp GetBaseOp(ReductionOp op)
        {
            return op switch
            {
                ReductionOp.NanSum => ReductionOp.Sum,
                ReductionOp.NanProd => ReductionOp.Prod,
                ReductionOp.NanMin => ReductionOp.Min,
                ReductionOp.NanMax => ReductionOp.Max,
                ReductionOp.NanMean => ReductionOp.Sum,
                ReductionOp.NanVar => ReductionOp.Sum,
                ReductionOp.NanStd => ReductionOp.Sum,
                _ => op
            };
        }

        /// <summary>
        /// Emit Vector.Zero load for NPTypeCode.
        /// </summary>
        private static void EmitLoadVectorZero(ILGenerator il, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);
            var vectorType = GetVectorType(clrType);

            var zeroField = vectorType.GetField("Zero", BindingFlags.Public | BindingFlags.Static);
            if (zeroField != null)
            {
                il.Emit(OpCodes.Ldsfld, zeroField);
            }
            else
            {
                // Fallback: create from scalar zero
                EmitLoadZero(il, type);
                EmitVectorCreate(il, type);
            }
        }

        /// <summary>
        /// Emit NaN constant for the specified type.
        /// </summary>
        private static void EmitLoadNaN(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                il.Emit(OpCodes.Ldc_R4, float.NaN);
            }
            else
            {
                il.Emit(OpCodes.Ldc_R8, double.NaN);
            }
        }

        /// <summary>
        /// Emit a typed constant value.
        /// </summary>
        private static void EmitLoadConstant(ILGenerator il, object value, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                il.Emit(OpCodes.Ldc_R4, Convert.ToSingle(value));
            }
            else if (type == NPTypeCode.Double)
            {
                il.Emit(OpCodes.Ldc_R8, Convert.ToDouble(value));
            }
            else
            {
                throw new NotSupportedException($"EmitLoadConstant not supported for {type}");
            }
        }

        /// <summary>
        /// Emit Vector.Equals(vec1, vec2) for floating-point comparison.
        /// Returns a vector mask where each element is all 1s (true) or all 0s (false).
        /// </summary>
        private static void EmitVectorEquals(ILGenerator il, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);

            var method = containerType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Equals" && m.IsGenericMethod &&
                            m.GetParameters().Length == 2)
                .MakeGenericMethod(clrType);

            il.EmitCall(OpCodes.Call, method, null);
        }

        /// <summary>
        /// Emit Vector.BitwiseAnd(vec1, vec2).
        /// </summary>
        private static void EmitVectorBitwiseAnd(ILGenerator il, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);

            var method = containerType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "BitwiseAnd" && m.IsGenericMethod &&
                            m.GetParameters().Length == 2)
                .MakeGenericMethod(clrType);

            il.EmitCall(OpCodes.Call, method, null);
        }

        /// <summary>
        /// Emit Vector.Add(vec1, vec2).
        /// </summary>
        private static void EmitVectorAdd(ILGenerator il, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);

            var method = containerType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Add" && m.IsGenericMethod &&
                            m.GetParameters().Length == 2)
                .MakeGenericMethod(clrType);

            il.EmitCall(OpCodes.Call, method, null);
        }

        /// <summary>
        /// Emit Vector.Subtract(vec1, vec2).
        /// </summary>
        private static void EmitVectorSubtract(ILGenerator il, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);

            var method = containerType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Subtract" && m.IsGenericMethod &&
                            m.GetParameters().Length == 2)
                .MakeGenericMethod(clrType);

            il.EmitCall(OpCodes.Call, method, null);
        }

        /// <summary>
        /// Emit Vector.Multiply(vec1, vec2).
        /// </summary>
        private static void EmitVectorMultiply(ILGenerator il, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);

            var method = containerType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Multiply" && m.IsGenericMethod &&
                            m.GetParameters().Length == 2)
                .MakeGenericMethod(clrType);

            il.EmitCall(OpCodes.Call, method, null);
        }

        /// <summary>
        /// Emit Vector.Sum(vec) for horizontal reduction.
        /// </summary>
        private static void EmitVectorSum(ILGenerator il, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);

            var method = containerType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Sum" && m.IsGenericMethod &&
                            m.GetParameters().Length == 1)
                .MakeGenericMethod(clrType);

            il.EmitCall(OpCodes.Call, method, null);
        }

        /// <summary>
        /// Emit Vector.ConditionalSelect(mask, ifTrue, ifFalse).
        /// </summary>
        private static void EmitVectorConditionalSelect(ILGenerator il, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);

            var method = containerType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "ConditionalSelect" && m.IsGenericMethod &&
                            m.GetParameters().Length == 3)
                .MakeGenericMethod(clrType);

            il.EmitCall(OpCodes.Call, method, null);
        }

        /// <summary>
        /// Emit vector.AsT() to reinterpret comparison mask as the element type.
        /// </summary>
        private static void EmitVectorAsType(ILGenerator il, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);
            var vectorType = GetVectorType(clrType);

            string methodName = type == NPTypeCode.Single ? "AsSingle" : "AsDouble";

            var method = containerType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == methodName && m.IsGenericMethod &&
                            m.GetParameters().Length == 1)
                .MakeGenericMethod(clrType);

            il.EmitCall(OpCodes.Call, method, null);
        }

        /// <summary>
        /// Emit Math.Sqrt or MathF.Sqrt depending on type.
        /// </summary>
        private static void EmitSqrt(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                var method = typeof(MathF).GetMethod("Sqrt", new[] { typeof(float) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else
            {
                var method = typeof(Math).GetMethod("Sqrt", new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
        }

        #endregion
    }
}
