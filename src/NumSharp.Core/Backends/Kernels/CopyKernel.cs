using System;

namespace NumSharp.Backends.Kernels
{
    public enum CopyExecutionPath
    {
        Contiguous,
        General
    }

    public readonly record struct CopyKernelKey(
        NPTypeCode Type,
        CopyExecutionPath Path
    )
    {
        public override string ToString() => $"{Type}_{Path}";
    }

    public unsafe delegate void CopyKernel(
        void* src,
        void* dst,
        long* srcStrides,
        long* dstStrides,
        long* shape,
        int ndim,
        long totalSize
    );
}
