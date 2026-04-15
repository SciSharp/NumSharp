using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
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
                owner: typeof(ILKernelGenerator),
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
            var helperMethod = typeof(ILKernelGenerator).GetMethod(
                nameof(CopyGeneralSameType),
                BindingFlags.NonPublic | BindingFlags.Static)!;

            var genericHelper = helperMethod.MakeGenericMethod(GetClrType(type));

            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Ldarg_S, (byte)5);
            il.Emit(OpCodes.Ldarg_S, (byte)6);
            il.EmitCall(OpCodes.Call, genericHelper, null);
        }

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

            for (long i = 0; i < totalSize; i++)
            {
                long srcOffset = 0;
                long dstOffset = 0;
                long idx = i;

                for (int axis = ndim - 1; axis >= 0; axis--)
                {
                    long dim = shape[axis];
                    long coord = idx % dim;
                    idx /= dim;

                    srcOffset += coord * srcStrides[axis];
                    dstOffset += coord * dstStrides[axis];
                }

                dstPtr[dstOffset] = srcPtr[srcOffset];
            }
        }
    }
}
