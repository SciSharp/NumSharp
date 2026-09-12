using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;

namespace NumSharp.Backends.Kernels
{
    // ============================ Fused divmod (np.divmod) ============================
    // np.divmod is a single nout=2 ufunc: divmod(a, b) == (a // b, a % b), computed in ONE pass.
    // Composing np.floor_divide with np.remainder reads a and b twice and does the divmod work
    // twice (two idiv for integers, two npy_divmod calls for floats) — measured BELOW NumPy's own
    // divmod. This fused kernel computes both outputs from a SINGLE NDDivision.Divmod* call per
    // element (the CPython-divmod core that also backs FloorDiv*/Rem*, so the two outputs are
    // bit-identical to the composition), over pre-materialized contiguous same-dtype operands.
    //
    // Integer Divmod* uses the ONE-IDIV form (q = n/d; r = n - q*d), avoiding the second hardware
    // division a naive (n/d, n%d) pays — roughly halving the integer kernel cost. divmod is
    // scalar-only (no Vector divide/remainder), so the loop is a 4x-unrolled scalar loop for ILP.
    //
    // Layout contract (enforced by DefaultEngine.DivMod): a, b, q, r are C-contiguous buffers of
    // the SAME resultType and length n. Broadcasting / casting / non-contiguous inputs are resolved
    // into contiguous resultType operands before this kernel runs (the np.modf pattern).
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        ///     Fused divmod kernel over contiguous same-dtype buffers: for each i,
        ///     <c>q[i] = floor(a[i]/b[i])</c> and <c>r[i] = a[i] - q[i]*b[i]</c> (floored),
        ///     both produced by a single <see cref="Utilities.NDDivision"/> Divmod* call.
        /// </summary>
        public unsafe delegate void DivModKernel(void* a, void* b, void* q, void* r, long n);

        /// <summary>Cache of fused divmod kernels keyed by (contiguous) element dtype.</summary>
        internal static readonly ConcurrentDictionary<NPTypeCode, DivModKernel> _divModCache = new();

        /// <summary>
        ///     Get or generate the fused divmod kernel for <paramref name="dt"/>. Returns <c>null</c>
        ///     when IL generation is disabled or the dtype has no Divmod* helper (Boolean — never
        ///     reaches here, promoted to int8; Complex — divmod is unsupported), so the caller falls
        ///     back to composing floor_divide + remainder.
        /// </summary>
        public static DivModKernel GetDivModKernel(NPTypeCode dt)
        {
            if (!Enabled)
                return null;
            if (GetDivmodMethod(dt) == null)
                return null;
            try
            {
                return _divModCache.GetOrAdd(dt, GenerateDivModKernel);
            }
            catch (NotSupportedException)
            {
                return null;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(
                    $"[ILKernel] GetDivModKernel({dt}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     Resolve the <see cref="Utilities.NDDivision"/> fused divmod helper
        ///     (<c>Divmod*(T a, T b, out T mod) -> T</c>) for <paramref name="dt"/>.
        /// </summary>
        private static MethodInfo GetDivmodMethod(NPTypeCode dt) => dt switch
        {
            NPTypeCode.SByte => CachedMethods.DivmodSByte,
            NPTypeCode.Byte => CachedMethods.DivmodByte,
            NPTypeCode.Int16 => CachedMethods.DivmodInt16,
            NPTypeCode.UInt16 => CachedMethods.DivmodUInt16,
            NPTypeCode.Char => CachedMethods.DivmodChar,
            NPTypeCode.Int32 => CachedMethods.DivmodInt32,
            NPTypeCode.UInt32 => CachedMethods.DivmodUInt32,
            NPTypeCode.Int64 => CachedMethods.DivmodInt64,
            NPTypeCode.UInt64 => CachedMethods.DivmodUInt64,
            NPTypeCode.Half => CachedMethods.DivmodHalf,
            NPTypeCode.Single => CachedMethods.DivmodSingle,
            NPTypeCode.Double => CachedMethods.DivmodDouble,
            NPTypeCode.Decimal => CachedMethods.DivmodDecimal,
            _ => null
        };

        private static DivModKernel GenerateDivModKernel(NPTypeCode dt)
        {
            var helper = GetDivmodMethod(dt)
                ?? throw new NotSupportedException($"divmod not supported for dtype {dt}");
            int elemSize = GetTypeSize(dt);
            Type clr = GetClrType(dt);

            var dm = new DynamicMethod(
                name: $"DivMod_{dt}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), // a  (dividend)
                    typeof(void*), // b  (divisor)
                    typeof(void*), // q  (floored quotient out)
                    typeof(void*), // r  (floored remainder out)
                    typeof(long)   // n  (element count)
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            var locMod = il.DeclareLocal(clr);       // scratch for the Divmod* out-parameter
            var locI = il.DeclareLocal(typeof(long)); // element index

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // 4x-unrolled body: process while i <= n - 4
            var lblUnroll = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var locUnrollEnd = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldarg_S, (byte)4); // n
            il.Emit(OpCodes.Ldc_I8, 4L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            il.MarkLabel(lblUnroll);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);
            for (int u = 0; u < 4; u++)
                EmitDivModStep(il, dt, elemSize, helper, locMod, locI, u);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 4L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblUnrollEnd);

            // scalar tail: while i < n
            var lblTail = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();
            il.MarkLabel(lblTail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)4); // n
            il.Emit(OpCodes.Bge, lblTailEnd);
            EmitDivModStep(il, dt, elemSize, helper, locMod, locI, 0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblTail);
            il.MarkLabel(lblTailEnd);

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<DivModKernel>();
        }

        /// <summary>
        ///     Emit one element at offset (i + <paramref name="off"/>):
        ///     <c>q[k] = Divmod(a[k], b[k], out mod); r[k] = mod;</c>.
        /// </summary>
        private static void EmitDivModStep(
            ILGenerator il, NPTypeCode dt, int elemSize, MethodInfo helper,
            LocalBuilder locMod, LocalBuilder locI, int off)
        {
            // --- store the quotient: push q address, compute value, stind ---
            // q addr = (byte*)arg2 + (i+off)*elemSize
            il.Emit(OpCodes.Ldarg_2);
            EmitDivModByteOffset(il, locI, off, elemSize);
            il.Emit(OpCodes.Add);

            // a[k], b[k]
            il.Emit(OpCodes.Ldarg_0);
            EmitDivModByteOffset(il, locI, off, elemSize);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, dt);
            il.Emit(OpCodes.Ldarg_1);
            EmitDivModByteOffset(il, locI, off, elemSize);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, dt);

            // &mod, call Divmod(a, b, out mod) -> floordiv
            il.Emit(OpCodes.Ldloca_S, locMod);
            il.EmitCall(OpCodes.Call, helper, null);

            // stack: [qAddr, floordiv] -> store
            EmitStoreIndirect(il, dt);

            // --- store the remainder: r addr, mod value, stind ---
            il.Emit(OpCodes.Ldarg_3);
            EmitDivModByteOffset(il, locI, off, elemSize);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locMod);
            EmitStoreIndirect(il, dt);
        }

        /// <summary>Push <c>(i + off) * elemSize</c> as a native int (byte offset from a base pointer).</summary>
        private static void EmitDivModByteOffset(ILGenerator il, LocalBuilder locI, int off, int elemSize)
        {
            il.Emit(OpCodes.Ldloc, locI);
            if (off != 0)
            {
                il.Emit(OpCodes.Ldc_I8, (long)off);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
        }
    }
}
