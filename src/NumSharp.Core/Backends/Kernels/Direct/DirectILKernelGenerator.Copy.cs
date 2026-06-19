using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        private static readonly ConcurrentDictionary<CopyKernelKey, CopyKernel> _copyKernelCache = new();

        public static CopyKernel GetCopyKernel(CopyKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            return _copyKernelCache.GetOrAdd(key, GenerateCopyKernel);
        }

        public static CopyKernel? TryGetCopyKernel(CopyKernelKey key)
        {
            if (!Enabled)
                return null;

            try
            {
                return _copyKernelCache.GetOrAdd(key, GenerateCopyKernel);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryGetCopyKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        private static CopyKernel GenerateCopyKernel(CopyKernelKey key)
        {
            var dm = new DynamicMethod(
                name: $"Copy_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*),
                    typeof(long*), typeof(long*), typeof(long*),
                    typeof(int), typeof(long)
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            switch (key.Path)
            {
                case CopyExecutionPath.Contiguous:
                    EmitContiguousCopy(il, GetTypeSize(key.Type));
                    break;
                case CopyExecutionPath.General:
                    EmitGeneralCopyHelperCall(il, key.Type);
                    break;
                default:
                    throw new NotSupportedException($"Copy path {key.Path} is not supported.");
            }

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<CopyKernel>();
        }

        private static void EmitContiguousCopy(ILGenerator il, int elementSize)
        {
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldarg_S, (byte)6);
            il.Emit(OpCodes.Ldc_I8, (long)elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_U);
            il.Emit(OpCodes.Cpblk);
        }

        private static void EmitGeneralCopyHelperCall(ILGenerator il, NPTypeCode type)
        {
            var genericHelper = GetGenericHelper(nameof(CopyGeneralSameType), GetClrType(type));

            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Ldarg_S, (byte)5);
            il.Emit(OpCodes.Ldarg_S, (byte)6);
            il.EmitCall(OpCodes.Call, genericHelper, null);
        }

        /// <summary>
        /// Same-dtype strided/broadcast copy fallback. Used for the dtypes the SIMD
        /// <c>StridedCastKernel</c> rejects (Char/Half/Decimal/Complex/Boolean — no
        /// <see cref="System.Numerics.Vector{T}"/> conversion lanes); same-dtype copy
        /// piggybacks on the cast kernel and inherits that rejection even though it
        /// needs no conversion at all — it is pure byte movement.
        ///
        /// The innermost axis defines a contiguous run when BOTH sides have unit
        /// element-stride there; that run is moved in one shot with
        /// <see cref="Buffer.MemoryCopy"/> (optimal for same-dtype bytes, and just as
        /// fast for the Vector-less dtypes as for any other). The outer axes are
        /// advanced by incremental stride add + carry — NO per-element div/mod, which
        /// was the old pathology that made non-contiguous Char/Half/Complex clones
        /// ~16-33x slower than NumPy's typed strided copy.
        /// </summary>
        private static unsafe void CopyGeneralSameType<T>(
            void* src,
            void* dst,
            long* srcStrides,
            long* dstStrides,
            long* shape,
            int ndim,
            long totalSize)
            where T : unmanaged
        {
            if (totalSize == 0)
                return;

            var srcPtr = (T*)src;
            var dstPtr = (T*)dst;

            // Scalar / 0-d: a single element, no axes to walk.
            if (ndim == 0)
            {
                dstPtr[0] = srcPtr[0];
                return;
            }

            int elemSize = sizeof(T);
            int last = ndim - 1;
            long inner = shape[last];
            long srcInner = srcStrides[last];
            long dstInner = dstStrides[last];
            // Unit element-stride on both sides => the inner run is a flat block.
            // A reversed (negative-stride) or broadcast (stride-0) inner axis is not
            // contiguous, so it correctly drops to the scalar inner walk below.
            bool innerContig = srcInner == 1 && dstInner == 1;
            long innerBytes = inner * (long)elemSize;

            // outerCount == product(shape[0..last-1]); exact since totalSize is the
            // product of every dim and inner > 0 whenever totalSize > 0.
            long outerCount = totalSize / inner;

            // Current coordinate per axis (only the outer axes 0..last-1 are advanced).
            long* coords = stackalloc long[ndim];
            for (int a = 0; a < ndim; a++)
                coords[a] = 0;

            long srcBase = 0, dstBase = 0;
            for (long r = 0; r < outerCount; r++)
            {
                if (innerContig)
                {
                    Buffer.MemoryCopy(srcPtr + srcBase, dstPtr + dstBase, innerBytes, innerBytes);
                }
                else
                {
                    long s = srcBase, d = dstBase;
                    for (long i = 0; i < inner; i++)
                    {
                        dstPtr[d] = srcPtr[s];
                        s += srcInner;
                        d += dstInner;
                    }
                }

                // Advance the outer coordinate by adding strides, carrying on overflow.
                // Does nothing when ndim == 1 (last - 1 < 0) — the single run above
                // already covered the whole axis.
                for (int a = last - 1; a >= 0; a--)
                {
                    srcBase += srcStrides[a];
                    dstBase += dstStrides[a];
                    if (++coords[a] < shape[a])
                        break;
                    coords[a] = 0;
                    srcBase -= srcStrides[a] * shape[a];
                    dstBase -= dstStrides[a] * shape[a];
                }
            }
        }
    }
}
