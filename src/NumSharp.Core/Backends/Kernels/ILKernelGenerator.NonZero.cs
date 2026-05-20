using System;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.NonZero.cs - IL-emitted nonzero kernels for boolean masks
// =============================================================================
//
// RESPONSIBILITY:
//   Replace the slow scalar coord-iter path in np.nonzero / np.where(cond) for
//   contiguous boolean arrays. Closes the 8x-241x gap to NumPy.
//
// KERNELS (all DynamicMethod-emitted at first use, cached as delegates):
//   * IsAllZeroBoolKernel    : (byte* mask, long size) -> bool
//       SIMD any-true prescan via Vector*.EqualsAll. Branches out of the loop
//       on the first non-zero chunk. NumPy's nonzero does this exact short-
//       circuit, which is why all-false 10M elements is 0.2ms there.
//
//   * NonZeroFlatBoolKernel  : (byte* mask, long* outBuf, long size) -> long
//       SIMD bit-scan index materialization. For each SIMD chunk:
//         1. Load V<byte> chunk
//         2. Equals(chunk, Zero) -> mask
//         3. ExtractMostSignificantBits -> bits
//         4. nonZeroBits = ~bits   (V256: full 32, V128: masked to low 16)
//         5. while (nonZeroBits != 0):
//              pos = BitOperations.TrailingZeroCount(nonZeroBits)
//              outBuf[outIdx++] = i + pos
//              nonZeroBits &= nonZeroBits - 1
//       Scalar tail handles the residual bytes. Returns outIdx (count written).
//
// CHOSEN SIMD WIDTH:
//   ILKernelGenerator.VectorBits is detected once at startup. The kernel emits
//   a single specialized loop (no V128/V256 if/else branches at runtime — the
//   JIT'd kernel only contains the chosen path).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted prescan: returns true iff every byte in the mask is zero.
    /// Used by np.where(cond) to short-circuit when the condition is entirely
    /// false (no index array materialization needed — NumPy parity).
    /// </summary>
    public unsafe delegate bool IsAllZeroBoolKernel(byte* mask, long size);

    /// <summary>
    /// IL-emitted SIMD popcount: returns the count of non-zero bytes in the mask.
    /// Used to pre-size the output buffer for the bit-scan kernel — avoids the
    /// pathological "allocate max-size temp" case for dense masks (10M all-true
    /// elements would otherwise allocate 80MB just to discard it).
    /// </summary>
    public unsafe delegate long NonZeroCountBoolKernel(byte* mask, long size);

    /// <summary>
    /// IL-emitted bit-scan: writes flat indices of non-zero bytes into outBuf
    /// (must be pre-sized to at least <c>count</c> longs — caller obtains the
    /// count via <see cref="NonZeroCountBoolKernel"/>). Returns the number of
    /// indices written.
    /// </summary>
    public unsafe delegate long NonZeroFlatBoolKernel(byte* mask, long* outBuf, long size);

    public static partial class ILKernelGenerator
    {
        #region Cached delegates (lazy init)

        private static IsAllZeroBoolKernel _isAllZeroBoolKernel;
        private static NonZeroCountBoolKernel _nonZeroCountBoolKernel;
        private static NonZeroFlatBoolKernel _nonZeroFlatBoolKernel;

        /// <summary>
        /// Returns the IL-emitted any-true prescan for boolean masks. The kernel
        /// is built lazily on first call and cached for the process lifetime.
        /// Returns <c>null</c> when SIMD is unavailable (VectorBits == 0) — caller
        /// falls back to scalar logic.
        /// </summary>
        public static IsAllZeroBoolKernel GetIsAllZeroBoolKernel()
        {
            if (!Enabled || VectorBits == 0)
                return null;

            var cached = _isAllZeroBoolKernel;
            if (cached != null)
                return cached;

            try
            {
                var kernel = GenerateIsAllZeroBoolKernelIL();
                System.Threading.Interlocked.CompareExchange(ref _isAllZeroBoolKernel, kernel, null);
                return _isAllZeroBoolKernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetIsAllZeroBoolKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Returns the IL-emitted popcount kernel. Lazy-built, cached.
        /// </summary>
        public static NonZeroCountBoolKernel GetNonZeroCountBoolKernel()
        {
            if (!Enabled || VectorBits == 0)
                return null;

            var cached = _nonZeroCountBoolKernel;
            if (cached != null)
                return cached;

            try
            {
                var kernel = GenerateNonZeroCountBoolKernelIL();
                System.Threading.Interlocked.CompareExchange(ref _nonZeroCountBoolKernel, kernel, null);
                return _nonZeroCountBoolKernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetNonZeroCountBoolKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Returns the IL-emitted bit-scan flat-index collector. Lazy-built, cached.
        /// Returns <c>null</c> when SIMD is unavailable; caller falls back to scalar.
        /// </summary>
        public static NonZeroFlatBoolKernel GetNonZeroFlatBoolKernel()
        {
            if (!Enabled || VectorBits == 0)
                return null;

            var cached = _nonZeroFlatBoolKernel;
            if (cached != null)
                return cached;

            try
            {
                var kernel = GenerateNonZeroFlatBoolKernelIL();
                System.Threading.Interlocked.CompareExchange(ref _nonZeroFlatBoolKernel, kernel, null);
                return _nonZeroFlatBoolKernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetNonZeroFlatBoolKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        #endregion

        #region IsAllZeroBoolKernel emission

        /// <summary>
        /// Emits:
        /// <code>
        /// long i = 0;
        /// long vectorEnd = size - chunkBytes;
        /// while (i &lt;= vectorEnd) {
        ///     if (!V&lt;byte&gt;.EqualsAll(V&lt;byte&gt;.Load(mask + i), V&lt;byte&gt;.Zero))
        ///         return false;
        ///     i += chunkBytes;
        /// }
        /// while (i &lt; size) {
        ///     if (mask[i] != 0) return false;
        ///     i++;
        /// }
        /// return true;
        /// </code>
        /// </summary>
        private static IsAllZeroBoolKernel GenerateIsAllZeroBoolKernelIL()
        {
            // V512<byte>.EqualsAll exists on .NET 8+; the EqualsAll helper just calls
            // the underlying generic. Use the detected width directly.
            int simdBits = VectorBits;
            int chunkBytes = simdBits / 8;

            var dm = new DynamicMethod(
                name: $"IL_IsAllZeroBool_V{simdBits}",
                returnType: typeof(bool),
                parameterTypes: new[] { typeof(byte*), typeof(long) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locI = il.DeclareLocal(typeof(long));
            var locVecEnd = il.DeclareLocal(typeof(long));

            var lblSimdHead = il.DefineLabel();
            var lblSimdEnd = il.DefineLabel();
            var lblScalarHead = il.DefineLabel();
            var lblFoundNonZero = il.DefineLabel();
            var lblAllZero = il.DefineLabel();

            // i = 0;
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // vectorEnd = size - chunkBytes;
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, (long)chunkBytes);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVecEnd);

            // ---- SIMD loop ----
            il.MarkLabel(lblSimdHead);

            // if (i > vectorEnd) goto SimdEnd;
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVecEnd);
            il.Emit(OpCodes.Bgt, lblSimdEnd);

            // chunk = V<byte>.Load(mask + i);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.EmitCall(OpCodes.Call, GetVectorLoadMethod(simdBits, typeof(byte)), null);

            // zero = V<byte>.Zero;
            var zeroProp = VType(simdBits, typeof(byte)).GetProperty("Zero",
                BindingFlags.Public | BindingFlags.Static)
                ?? throw new InvalidOperationException($"Vector{simdBits}<byte>.Zero not found");
            il.EmitCall(OpCodes.Call, zeroProp.GetGetMethod(), null);

            // EqualsAll(chunk, zero) -> bool
            il.EmitCall(OpCodes.Call, GetEqualsAllMethodNZ(simdBits, typeof(byte)), null);
            il.Emit(OpCodes.Brfalse, lblFoundNonZero);

            // i += chunkBytes;
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)chunkBytes);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblSimdHead);

            il.MarkLabel(lblSimdEnd);

            // ---- Scalar tail ----
            il.MarkLabel(lblScalarHead);

            // if (i >= size) goto AllZero;
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Bge, lblAllZero);

            // if (mask[i] != 0) goto FoundNonZero;
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brtrue, lblFoundNonZero);

            // i++;
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblScalarHead);

            // ---- Returns ----
            il.MarkLabel(lblFoundNonZero);
            il.Emit(OpCodes.Ldc_I4_0);  // false
            il.Emit(OpCodes.Ret);

            il.MarkLabel(lblAllZero);
            il.Emit(OpCodes.Ldc_I4_1);  // true
            il.Emit(OpCodes.Ret);

            return (IsAllZeroBoolKernel)dm.CreateDelegate(typeof(IsAllZeroBoolKernel));
        }

        #endregion

        #region NonZeroCountBoolKernel emission

        /// <summary>
        /// Emits a SIMD popcount over the mask bytes. For each chunk:
        /// <code>
        /// cmp = V&lt;byte&gt;.Equals(V&lt;byte&gt;.Load(mask + i), Zero);
        /// uint bits = ExtractMostSignificantBits(cmp);
        /// count += PopCount(~bits &amp; chunkMask);   // popcount of "is non-zero" lanes
        /// </code>
        /// Scalar tail counts the residual bytes. PopCount of inverted MSB-bits is the
        /// same as counting non-zero bytes in the chunk.
        /// </summary>
        private static NonZeroCountBoolKernel GenerateNonZeroCountBoolKernelIL()
        {
            int simdBits = VectorBits >= 256 ? 256 : 128;
            int chunkBytes = simdBits / 8;
            uint chunkMask = simdBits == 256 ? 0xFFFFFFFFu : 0x0000FFFFu;

            var dm = new DynamicMethod(
                name: $"IL_NonZeroCountBool_V{simdBits}",
                returnType: typeof(long),
                parameterTypes: new[] { typeof(byte*), typeof(long) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locI = il.DeclareLocal(typeof(long));
            var locVecEnd = il.DeclareLocal(typeof(long));
            var locCount = il.DeclareLocal(typeof(long));

            var lblSimdHead = il.DefineLabel();
            var lblSimdEnd = il.DefineLabel();
            var lblScalarHead = il.DefineLabel();
            var lblScalarEnd = il.DefineLabel();
            var lblScalarSkip = il.DefineLabel();

            // i = 0; count = 0;
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locCount);

            // vectorEnd = size - chunkBytes;
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, (long)chunkBytes);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVecEnd);

            // ---- SIMD loop ----
            il.MarkLabel(lblSimdHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVecEnd);
            il.Emit(OpCodes.Bgt, lblSimdEnd);

            // chunk = V<byte>.Load(mask + i);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.EmitCall(OpCodes.Call, GetVectorLoadMethod(simdBits, typeof(byte)), null);

            // zero
            var zeroProp = VType(simdBits, typeof(byte)).GetProperty("Zero",
                BindingFlags.Public | BindingFlags.Static)
                ?? throw new InvalidOperationException($"Vector{simdBits}<byte>.Zero not found");
            il.EmitCall(OpCodes.Call, zeroProp.GetGetMethod(), null);

            // cmp = Equals(chunk, zero)
            il.EmitCall(OpCodes.Call, GetVectorEqualsMethodNZ(simdBits, typeof(byte)), null);

            // bits = ExtractMostSignificantBits(cmp)
            il.EmitCall(OpCodes.Call, GetExtractMostSignificantBitsMethodNZ(simdBits, typeof(byte)), null);

            // ~bits & chunkMask
            il.Emit(OpCodes.Not);
            il.Emit(OpCodes.Ldc_I4, unchecked((int)chunkMask));
            il.Emit(OpCodes.And);

            // PopCount(uint)
            il.EmitCall(OpCodes.Call, BitOpsPopCountUInt32, null);

            // count += (long)PopCount(...);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locCount);

            // i += chunkBytes;
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)chunkBytes);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblSimdHead);

            il.MarkLabel(lblSimdEnd);

            // ---- Scalar tail ----
            il.MarkLabel(lblScalarHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Bge, lblScalarEnd);

            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblScalarSkip);

            // count++;
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locCount);

            il.MarkLabel(lblScalarSkip);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblScalarHead);

            il.MarkLabel(lblScalarEnd);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Ret);

            return (NonZeroCountBoolKernel)dm.CreateDelegate(typeof(NonZeroCountBoolKernel));
        }

        #endregion

        #region NonZeroFlatBoolKernel emission

        /// <summary>
        /// Emits a SIMD bit-scan that materializes flat indices of non-zero bytes
        /// from <c>mask</c> into <c>outBuf</c>. Returns the number written.
        ///
        /// Layout per SIMD chunk (chunkBytes = simdBits/8 elements):
        /// <code>
        /// chunk = V&lt;byte&gt;.Load(mask + i);
        /// cmp = V&lt;byte&gt;.Equals(chunk, Zero);
        /// uint bits = V&lt;byte&gt;.ExtractMostSignificantBits(cmp);
        /// uint nz = ~bits &amp; chunkMask;          // chunkMask handles V128's 16-bit case
        /// while (nz != 0) {
        ///     int pos = BitOperations.TrailingZeroCount(nz);
        ///     outBuf[outIdx++] = i + pos;
        ///     nz &amp;= nz - 1;
        /// }
        /// i += chunkBytes;
        /// </code>
        /// Scalar tail walks the residual bytes one at a time.
        /// </summary>
        private static NonZeroFlatBoolKernel GenerateNonZeroFlatBoolKernelIL()
        {
            // The bit-scan uses ExtractMostSignificantBits which returns a uint with up to
            // 32 valid lanes. V256<byte>.Count = 32 (full 32 bits), V128<byte>.Count = 16
            // (low 16 bits valid). V512<byte>.ExtractMostSignificantBits returns ulong —
            // for simplicity we cap at V256 here. V512 hardware will still benefit because
            // the byte-load throughput is generally the bottleneck, not the per-chunk
            // mask extraction.
            int simdBits = VectorBits >= 256 ? 256 : 128;
            int chunkBytes = simdBits / 8;
            uint chunkMask = simdBits == 256 ? 0xFFFFFFFFu : 0x0000FFFFu;

            var dm = new DynamicMethod(
                name: $"IL_NonZeroFlatBool_V{simdBits}",
                returnType: typeof(long),
                parameterTypes: new[] { typeof(byte*), typeof(long*), typeof(long) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locI = il.DeclareLocal(typeof(long));        // mask byte index
            var locOut = il.DeclareLocal(typeof(long));      // outBuf write index
            var locVecEnd = il.DeclareLocal(typeof(long));   // vector loop bound
            var locNz = il.DeclareLocal(typeof(uint));       // inverted bits
            var locPos = il.DeclareLocal(typeof(int));       // trailing zero count

            var lblSimdHead = il.DefineLabel();
            var lblSimdEnd = il.DefineLabel();
            var lblBitHead = il.DefineLabel();
            var lblBitEnd = il.DefineLabel();
            var lblScalarHead = il.DefineLabel();
            var lblScalarEnd = il.DefineLabel();
            var lblScalarSkip = il.DefineLabel();

            // i = 0; out = 0;
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locOut);

            // vectorEnd = size - chunkBytes;
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, (long)chunkBytes);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVecEnd);

            // ---- SIMD outer loop ----
            il.MarkLabel(lblSimdHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVecEnd);
            il.Emit(OpCodes.Bgt, lblSimdEnd);

            // chunk = V<byte>.Load(mask + i);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.EmitCall(OpCodes.Call, GetVectorLoadMethod(simdBits, typeof(byte)), null);

            // zero = V<byte>.Zero;
            var zeroProp = VType(simdBits, typeof(byte)).GetProperty("Zero",
                BindingFlags.Public | BindingFlags.Static)
                ?? throw new InvalidOperationException($"Vector{simdBits}<byte>.Zero not found");
            il.EmitCall(OpCodes.Call, zeroProp.GetGetMethod(), null);

            // cmp = Equals(chunk, zero);
            il.EmitCall(OpCodes.Call, GetVectorEqualsMethodNZ(simdBits, typeof(byte)), null);

            // bits = ExtractMostSignificantBits(cmp); (uint)
            il.EmitCall(OpCodes.Call, GetExtractMostSignificantBitsMethodNZ(simdBits, typeof(byte)), null);

            // nz = ~bits & chunkMask;
            il.Emit(OpCodes.Not);
            il.Emit(OpCodes.Ldc_I4, unchecked((int)chunkMask));
            il.Emit(OpCodes.And);
            il.Emit(OpCodes.Stloc, locNz);

            // ---- Inner bit-scan loop ----
            il.MarkLabel(lblBitHead);
            il.Emit(OpCodes.Ldloc, locNz);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Beq, lblBitEnd);

            // pos = BitOperations.TrailingZeroCount(nz);
            il.Emit(OpCodes.Ldloc, locNz);
            il.EmitCall(OpCodes.Call, BitOpsTrailingZeroCountUInt32, null);
            il.Emit(OpCodes.Stloc, locPos);

            // outBuf[out] = i + pos;
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locOut);
            il.Emit(OpCodes.Ldc_I4_8);  // sizeof(long) = 8
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locPos);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stind_I8);

            // out++;
            il.Emit(OpCodes.Ldloc, locOut);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOut);

            // nz &= nz - 1;
            il.Emit(OpCodes.Ldloc, locNz);
            il.Emit(OpCodes.Ldloc, locNz);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.And);
            il.Emit(OpCodes.Stloc, locNz);

            il.Emit(OpCodes.Br, lblBitHead);

            il.MarkLabel(lblBitEnd);

            // i += chunkBytes;
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)chunkBytes);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblSimdHead);

            il.MarkLabel(lblSimdEnd);

            // ---- Scalar tail ----
            il.MarkLabel(lblScalarHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblScalarEnd);

            // if (mask[i] == 0) skip;
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblScalarSkip);

            // outBuf[out++] = i;
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locOut);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Stind_I8);

            il.Emit(OpCodes.Ldloc, locOut);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOut);

            il.MarkLabel(lblScalarSkip);
            // i++;
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblScalarHead);

            il.MarkLabel(lblScalarEnd);
            il.Emit(OpCodes.Ldloc, locOut);
            il.Emit(OpCodes.Ret);

            return (NonZeroFlatBoolKernel)dm.CreateDelegate(typeof(NonZeroFlatBoolKernel));
        }

        #endregion

        #region SIMD reflection helpers (suffixed NZ to avoid clashing with existing helpers
        //         that have file-private visibility in Cast.cs / Cast.Masked.cs)

        private static MethodInfo GetEqualsAllMethodNZ(int simdBits, Type elementType)
        {
            return System.Linq.Enumerable.First(
                ContainerType(simdBits).GetMethods(BindingFlags.Public | BindingFlags.Static),
                m => m.Name == "EqualsAll" && m.IsGenericMethod &&
                     m.GetParameters().Length == 2 &&
                     m.GetGenericArguments().Length == 1 &&
                     m.ReturnType == typeof(bool))
                .MakeGenericMethod(elementType);
        }

        private static MethodInfo GetVectorEqualsMethodNZ(int simdBits, Type elementType)
        {
            // V<T>.Equals(V<T>, V<T>) -> V<T>   (NOT the bool-returning instance Equals — we want the static op)
            var vt = VType(simdBits, elementType);
            return System.Linq.Enumerable.First(
                ContainerType(simdBits).GetMethods(BindingFlags.Public | BindingFlags.Static),
                m => m.Name == "Equals" && m.IsGenericMethod &&
                     m.GetParameters().Length == 2 &&
                     m.GetGenericArguments().Length == 1 &&
                     m.ReturnType.IsGenericType)
                .MakeGenericMethod(elementType);
        }

        private static MethodInfo GetExtractMostSignificantBitsMethodNZ(int simdBits, Type elementType)
        {
            return System.Linq.Enumerable.First(
                ContainerType(simdBits).GetMethods(BindingFlags.Public | BindingFlags.Static),
                m => m.Name == "ExtractMostSignificantBits" && m.IsGenericMethod &&
                     m.GetParameters().Length == 1 &&
                     m.GetGenericArguments().Length == 1)
                .MakeGenericMethod(elementType);
        }

        private static readonly MethodInfo BitOpsTrailingZeroCountUInt32 =
            typeof(BitOperations).GetMethod(nameof(BitOperations.TrailingZeroCount),
                BindingFlags.Public | BindingFlags.Static,
                binder: null,
                types: new[] { typeof(uint) },
                modifiers: null)
            ?? throw new InvalidOperationException("BitOperations.TrailingZeroCount(uint) not found");

        private static readonly MethodInfo BitOpsPopCountUInt32 =
            typeof(BitOperations).GetMethod(nameof(BitOperations.PopCount),
                BindingFlags.Public | BindingFlags.Static,
                binder: null,
                types: new[] { typeof(uint) },
                modifiers: null)
            ?? throw new InvalidOperationException("BitOperations.PopCount(uint) not found");

        #endregion
    }
}
