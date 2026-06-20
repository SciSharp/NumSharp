using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using NumSharp.Utilities;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        // =====================================================================
        // Scalar IL cast kernels — the per-element inner loop for cross-dtype
        // casts that the SIMD StridedCastKernel rejects (anything touching
        // Boolean/Char/Half/Decimal/Complex, plus float->int wrapping pairs).
        //
        // The kernel emits a DIRECT `call Converts.To{Dst}(srcValue)` per element
        // — the JIT inlines the conversion, so it runs at hand-written direct-call
        // speed. Probed (4M elems): a DynamicMethod call-loop is 0.99-1.00x of a
        // direct static-call loop, while the Func<TIn,TOut> delegate it replaces is
        // 1.1-4.8x SLOWER (the lighter the conversion, the more the delegate
        // dominates — bool/char widening were the worst, ~3-5x). Converts.{To X}
        // is the same bit-exact, NumPy-faithful table FindConverter bound to, so
        // semantics are unchanged; only the indirection is gone.
        //
        // Addressing is supplied by the caller (NpyIterCasting): this loop just
        // walks ONE inner run, advancing each pointer by its per-element BYTE
        // stride (which may be 0 for broadcast, negative for reversed, or any
        // multiple for strided). The caller's incremental-coord outer walk calls
        // it once per inner run.
        // =====================================================================

        /// <summary>
        /// Convert <paramref name="count"/> elements src→dst, advancing each pointer
        /// by its byte stride per element. One direct <c>call Converts.To{Dst}</c> body.
        /// </summary>
        public unsafe delegate void InnerCastLoop(
            void* src, long srcStrideBytes, void* dst, long dstStrideBytes, long count);

        private static readonly ConcurrentDictionary<CastKernelKey, InnerCastLoop> _innerCastCache = new();

        /// <summary>Number of cached scalar inner-cast kernels (diagnostics).</summary>
        public static int InnerCastCachedCount => _innerCastCache.Count;

        /// <summary>
        /// Get or emit the scalar inner-loop cast kernel for the pair. Non-null for every
        /// dtype pair (all 225 <c>Converts.To{Dst}({Src})</c> methods exist); returns null
        /// only if IL generation is disabled or a method unexpectedly fails to resolve.
        /// </summary>
        public static InnerCastLoop TryGetInnerCastKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (!Enabled) return null;

            var key = new CastKernelKey(srcType, dstType);
            if (_innerCastCache.TryGetValue(key, out var existing)) return existing;

            try
            {
                var kernel = GenerateInnerCastKernel(srcType, dstType);
                return kernel == null ? null : _innerCastCache.GetOrAdd(key, kernel);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryGetInnerCastKernel({srcType}, {dstType}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        private static InnerCastLoop GenerateInnerCastKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            MethodInfo conv = GetConvertsMethod(srcType, dstType);
            if (conv == null)
                return null;

            var dm = new DynamicMethod(
                $"InnerCast_{srcType}To{dstType}",
                typeof(void),
                new[] { typeof(void*), typeof(long), typeof(void*), typeof(long), typeof(long) },
                typeof(DirectILKernelGenerator).Module,
                skipVisibility: true);

            var il = dm.GetILGenerator();
            var sp = il.DeclareLocal(typeof(byte*)); // src cursor
            var dp = il.DeclareLocal(typeof(byte*)); // dst cursor
            var i = il.DeclareLocal(typeof(long));   // element counter
            var top = il.DefineLabel();
            var cond = il.DefineLabel();

            // sp = src; dp = dst; i = 0;
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Stloc, sp);
            il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Stloc, dp);
            il.Emit(OpCodes.Ldc_I4_0); il.Emit(OpCodes.Conv_I8); il.Emit(OpCodes.Stloc, i);
            il.Emit(OpCodes.Br, cond);

            il.MarkLabel(top);
            // *(Dst*)dp = Converts.To{Dst}( *(Src*)sp );
            il.Emit(OpCodes.Ldloc, dp);        // dest address (for the store)
            il.Emit(OpCodes.Ldloc, sp);        // src address
            EmitLoadIndirect(il, srcType);     // load *(Src*)sp  (int32-on-stack for sub-word; struct via ldobj for Half/dec/Complex)
            il.Emit(OpCodes.Call, conv);       // Converts.To{Dst}(value)  — JIT inlines
            EmitStoreIndirect(il, dstType);    // *(Dst*)dp = converted

            // sp += srcStrideBytes;
            il.Emit(OpCodes.Ldloc, sp); il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, sp);
            // dp += dstStrideBytes;
            il.Emit(OpCodes.Ldloc, dp); il.Emit(OpCodes.Ldarg_3); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, dp);
            // i++;
            il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I4_1); il.Emit(OpCodes.Conv_I8); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, i);

            il.MarkLabel(cond);
            il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldarg_S, (byte)4); il.Emit(OpCodes.Blt, top); // i < count (signed; both >= 0)
            il.Emit(OpCodes.Ret);

            return (InnerCastLoop)dm.CreateDelegate(typeof(InnerCastLoop));
        }

        /// <summary>
        /// Resolve <c>Converts.To{Dst}({Src})</c> — the typed, NumPy-bit-exact conversion the
        /// emitted loop calls directly. All 15×15 overloads exist (verified).
        /// </summary>
        private static MethodInfo GetConvertsMethod(NPTypeCode srcType, NPTypeCode dstType)
        {
            return typeof(Converts).GetMethod(
                "To" + ConvertsSuffix(dstType),
                BindingFlags.Public | BindingFlags.Static,
                new[] { GetClrType(srcType) });
        }

        private static string ConvertsSuffix(NPTypeCode t) => t switch
        {
            NPTypeCode.Boolean => "Boolean",
            NPTypeCode.Byte => "Byte",
            NPTypeCode.SByte => "SByte",
            NPTypeCode.Int16 => "Int16",
            NPTypeCode.UInt16 => "UInt16",
            NPTypeCode.Int32 => "Int32",
            NPTypeCode.UInt32 => "UInt32",
            NPTypeCode.Int64 => "Int64",
            NPTypeCode.UInt64 => "UInt64",
            NPTypeCode.Char => "Char",
            NPTypeCode.Half => "Half",
            NPTypeCode.Single => "Single",
            NPTypeCode.Double => "Double",
            NPTypeCode.Decimal => "Decimal",
            NPTypeCode.Complex => "Complex",
            _ => throw new NotSupportedException(t.ToString())
        };
    }
}
