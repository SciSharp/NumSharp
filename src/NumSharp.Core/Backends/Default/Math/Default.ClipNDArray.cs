using System;
using System.Linq;
using System.Numerics;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // ClipNDArray - Clip with array-valued min/max bounds
        // =============================================================================
        //
        // This implements np.clip(a, min_array, max_array) where min and/or max are
        // NDArrays (possibly broadcast) rather than scalar values.
        //
        // NumPy behavior:
        // - result[i] = min(max(a[i], min[i]), max[i])
        // - When min[i] > max[i] at any position, result is max[i]
        // - NaN in bounds array: propagates to result (IEEE comparison semantics)
        // - min/max arrays are broadcast to match input shape
        //
        // Implementation strategy:
        // 1. Broadcast min/max arrays to match input shape
        // 2. Create output array (copy of input with requested dtype)
        // 3. If all arrays are contiguous, use SIMD-optimized IL kernel path
        // 4. Otherwise, fall back to iterator-based element-wise processing
        //
        // =============================================================================

        public override NDArray ClipNDArray(NDArray lhs, NDArray min, NDArray max, Type dtype, NDArray @out = null)
            => ClipNDArray(lhs, min, max, dtype?.GetTypeCode(), @out);

        // Returns `bound` ready to feed the array-bound kernels: same shape as
        // `targetShape`, same dtype as `outType`, contiguous from offset 0.
        // Skips the broadcast + clone when the bound already satisfies all
        // three (the common case where users pass pre-shaped arrays).
        private static NDArray PrepareBound(NDArray bound, Shape targetShape, NPTypeCode outType)
        {
            if (bound is null) return null;
            if (bound.GetTypeCode == outType
                && bound.Shape.Equals(targetShape)
                && bound.Shape.IsContiguous
                && bound.Shape.Offset == 0)
            {
                return bound;
            }
            return np.broadcast_to(bound, targetShape).astype(outType);
        }

        // Promotion rule for a single bound. A 0-d bound of the same kind
        // (int / float / complex / bool / decimal) as `outType` is treated as a
        // "weak" scalar — output dtype stays unchanged. Array bounds or
        // cross-kind scalars promote via np.result_type, matching ufunc rules.
        private static NPTypeCode PromoteClipBound(NPTypeCode outType, NDArray bound)
        {
            if (bound is null) return outType;
            if (bound.ndim == 0 && IsSameKind(outType, bound.typecode))
                return outType;
            return np.result_type(outType, bound.typecode);
        }

        private static bool IsSameKind(NPTypeCode a, NPTypeCode b)
        {
            int ga = a.GetGroup();
            int gb = b.GetGroup();
            // 0 = Byte/Char (unsigned 1-byte int-like), 1 = signed int, 2 = unsigned int — all integer kind.
            bool aInt = ga >= 0 && ga <= 2;
            bool bInt = gb >= 0 && gb <= 2;
            if (aInt && bInt) return true;
            return ga == gb;
        }

        public override NDArray ClipNDArray(NDArray lhs, NDArray min, NDArray max, NPTypeCode? typeCode = null, NDArray @out = null)
        {
            // Determine the natural output dtype:
            //   explicit `dtype=` wins; otherwise apply NEP 50 weak-scalar
            //   promotion consistent with the binary-op engine — a 0-d bound
            //   of the same kind (int/float/complex) as `lhs` is treated as
            //   a "weak" scalar and does not promote, mirroring how NumPy
            //   handles Python int/float literals (`np.clip(uint8, 50, 75)`
            //   stays uint8). Cross-kind or array bounds promote per
            //   `np.result_type`.
            NPTypeCode outType;
            if (typeCode.HasValue)
            {
                outType = typeCode.Value;
            }
            else
            {
                outType = lhs.typecode;
                outType = PromoteClipBound(outType, min);
                outType = PromoteClipBound(outType, max);
            }

            // Validate @out up front (shape, writeable, dtype) — NumPy raises
            // _UFuncOutputCastingError when the destination dtype can't take the
            // promoted result; we surface the same constraint as ArgumentException.
            if (@out is not null)
            {
                NumSharpException.ThrowIfNotWriteable(@out.Shape);
                if (@out.Shape != lhs.Shape)
                    throw new ArgumentException($"@out's shape ({@out.Shape}) must match lhs's shape ({lhs.Shape}).'");
                if (@out.GetTypeCode != outType)
                    throw new ArgumentException(
                        $"Cannot cast ufunc 'clip' output from dtype('{outType.AsNumpyDtypeName()}') to dtype('{@out.GetTypeCode.AsNumpyDtypeName()}') with casting rule 'same_kind'.");
            }

            if (lhs.size == 0)
            {
                if (@out is not null) return @out;
                return Cast(lhs, outType, copy: true);
            }

            // If both bounds are null, just return a copy at the requested dtype.
            if (min is null && max is null)
            {
                if (@out is not null)
                {
                    np.copyto(@out, Cast(lhs, outType, copy: false));
                    return @out;
                }
                return Cast(lhs, outType, copy: true);
            }

            bool minIsScalar = min is null || min.ndim == 0;
            bool maxIsScalar = max is null || max.ndim == 0;

            // Fastest path: scalar bounds + contiguous same-dtype lhs + fresh
            // output. Use the fused copy+clip kernel — one pass over memory
            // (1R src + 1W dst) instead of two (Cast does 1R+1W, then in-place
            // clip does another 1R+1W). Halves memory bandwidth, brings
            // `np.clip(a, lo, hi)` close to NumPy parity.
            bool canFuse = minIsScalar && maxIsScalar
                        && @out is null
                        && lhs.GetTypeCode == outType
                        && lhs.Shape.IsContiguous
                        && lhs.Shape.Offset == 0;
            if (canFuse)
                return ClipNDArrayFusedScalarBounds(lhs, min, max, outType);

            // Materialize output buffer at outType, copying the (possibly promoted) input.
            if (@out is null)
            {
                @out = Cast(lhs, outType, copy: true);
            }
            else
            {
                // @out's dtype already validated to equal outType above.
                np.copyto(@out, Cast(lhs, outType, copy: false));
            }

            // Fast path: 0-d (scalar) bounds — skip the broadcast + astype
            // materialization (which previously allocated and wrote two
            // `lhs.size`-element bound arrays per call) and call the scalar
            // SIMD kernels (`ClipUnified` / `ClipMinUnified` / `ClipMaxUnified`)
            // directly. Reached here when @out was supplied or lhs needed
            // casting (so the fused single-pass path doesn't apply).
            if (minIsScalar && maxIsScalar)
                return ClipNDArrayScalarBounds(@out, min, max, outType);

            // Slow path: at least one bound is an array — broadcast & cast.
            // PrepareBound skips the broadcast_to + astype clone when the bound
            // is already same-shape, same-dtype, and contiguous, which avoids
            // a `len`-sized memory write per bound (the dominant overhead in
            // the array-bounds path for matched-dtype inputs).
            var _min = PrepareBound(min, lhs.Shape, outType);
            var _max = PrepareBound(max, lhs.Shape, outType);

            var len = @out.size;

            // Check if we can use the contiguous SIMD path for array bounds.
            // All participating arrays must be contiguous with zero offset
            bool canUseFastPath = @out.Shape.IsContiguous && @out.Shape.Offset == 0;
            if (!(_min is null) && canUseFastPath)
                canUseFastPath = _min.Shape.IsContiguous && _min.Shape.Offset == 0;
            if (!(_max is null) && canUseFastPath)
                canUseFastPath = _max.Shape.IsContiguous && _max.Shape.Offset == 0;

            if (canUseFastPath)
                return ClipNDArrayContiguous(@out, _min, _max, len);
            else
                return ClipNDArrayGeneral(@out, _min, _max, len);
        }

        /// <summary>
        /// Fastest path: contiguous lhs + scalar bounds + no @out + no cast.
        /// Allocates a fresh output buffer and runs the fused CopyAndClip
        /// kernel — single memory pass (1R src + 1W dst) instead of the
        /// classic Cast-then-clip pattern (2R + 2W).
        /// </summary>
        private unsafe NDArray ClipNDArrayFusedScalarBounds(NDArray lhs, NDArray min, NDArray max, NPTypeCode outType)
        {
            long len = lhs.size;

            // Allocate destination at outType (==lhs.typecode here, by canFuse check).
            // np.empty avoids the redundant initial fill that np.zeros does.
            var @out = np.empty(lhs.Shape, outType);

            // Cast bounds to outType. 0-d astype is O(1).
            var minCast = min is null ? null : min.astype(outType);
            var maxCast = max is null ? null : max.astype(outType);

            switch (outType)
            {
                case NPTypeCode.Byte:
                    FusedClipScalar<byte>(lhs, @out, minCast, maxCast, len,
                        (s, d, n, lo, hi) => ILKernelGenerator.CopyAndClip(s, d, n, lo, hi),
                        (s, d, n, lo)     => ILKernelGenerator.CopyAndClipMin(s, d, n, lo),
                        (s, d, n, hi)     => ILKernelGenerator.CopyAndClipMax(s, d, n, hi));
                    return @out;
                case NPTypeCode.SByte:
                    FusedClipScalar<sbyte>(lhs, @out, minCast, maxCast, len,
                        (s, d, n, lo, hi) => ILKernelGenerator.CopyAndClip(s, d, n, lo, hi),
                        (s, d, n, lo)     => ILKernelGenerator.CopyAndClipMin(s, d, n, lo),
                        (s, d, n, hi)     => ILKernelGenerator.CopyAndClipMax(s, d, n, hi));
                    return @out;
                case NPTypeCode.Int16:
                    FusedClipScalar<short>(lhs, @out, minCast, maxCast, len,
                        (s, d, n, lo, hi) => ILKernelGenerator.CopyAndClip(s, d, n, lo, hi),
                        (s, d, n, lo)     => ILKernelGenerator.CopyAndClipMin(s, d, n, lo),
                        (s, d, n, hi)     => ILKernelGenerator.CopyAndClipMax(s, d, n, hi));
                    return @out;
                case NPTypeCode.UInt16:
                    FusedClipScalar<ushort>(lhs, @out, minCast, maxCast, len,
                        (s, d, n, lo, hi) => ILKernelGenerator.CopyAndClip(s, d, n, lo, hi),
                        (s, d, n, lo)     => ILKernelGenerator.CopyAndClipMin(s, d, n, lo),
                        (s, d, n, hi)     => ILKernelGenerator.CopyAndClipMax(s, d, n, hi));
                    return @out;
                case NPTypeCode.Int32:
                    FusedClipScalar<int>(lhs, @out, minCast, maxCast, len,
                        (s, d, n, lo, hi) => ILKernelGenerator.CopyAndClip(s, d, n, lo, hi),
                        (s, d, n, lo)     => ILKernelGenerator.CopyAndClipMin(s, d, n, lo),
                        (s, d, n, hi)     => ILKernelGenerator.CopyAndClipMax(s, d, n, hi));
                    return @out;
                case NPTypeCode.UInt32:
                    FusedClipScalar<uint>(lhs, @out, minCast, maxCast, len,
                        (s, d, n, lo, hi) => ILKernelGenerator.CopyAndClip(s, d, n, lo, hi),
                        (s, d, n, lo)     => ILKernelGenerator.CopyAndClipMin(s, d, n, lo),
                        (s, d, n, hi)     => ILKernelGenerator.CopyAndClipMax(s, d, n, hi));
                    return @out;
                case NPTypeCode.Int64:
                    FusedClipScalar<long>(lhs, @out, minCast, maxCast, len,
                        (s, d, n, lo, hi) => ILKernelGenerator.CopyAndClip(s, d, n, lo, hi),
                        (s, d, n, lo)     => ILKernelGenerator.CopyAndClipMin(s, d, n, lo),
                        (s, d, n, hi)     => ILKernelGenerator.CopyAndClipMax(s, d, n, hi));
                    return @out;
                case NPTypeCode.UInt64:
                    FusedClipScalar<ulong>(lhs, @out, minCast, maxCast, len,
                        (s, d, n, lo, hi) => ILKernelGenerator.CopyAndClip(s, d, n, lo, hi),
                        (s, d, n, lo)     => ILKernelGenerator.CopyAndClipMin(s, d, n, lo),
                        (s, d, n, hi)     => ILKernelGenerator.CopyAndClipMax(s, d, n, hi));
                    return @out;
                case NPTypeCode.Single:
                    FusedClipScalar<float>(lhs, @out, minCast, maxCast, len,
                        (s, d, n, lo, hi) => ILKernelGenerator.CopyAndClip(s, d, n, lo, hi),
                        (s, d, n, lo)     => ILKernelGenerator.CopyAndClipMin(s, d, n, lo),
                        (s, d, n, hi)     => ILKernelGenerator.CopyAndClipMax(s, d, n, hi));
                    return @out;
                case NPTypeCode.Double:
                    FusedClipScalar<double>(lhs, @out, minCast, maxCast, len,
                        (s, d, n, lo, hi) => ILKernelGenerator.CopyAndClip(s, d, n, lo, hi),
                        (s, d, n, lo)     => ILKernelGenerator.CopyAndClipMin(s, d, n, lo),
                        (s, d, n, hi)     => ILKernelGenerator.CopyAndClipMax(s, d, n, hi));
                    return @out;
                case NPTypeCode.Char:
                    FusedClipScalar<char>(lhs, @out, minCast, maxCast, len,
                        (s, d, n, lo, hi) => ILKernelGenerator.CopyAndClip(s, d, n, lo, hi),
                        (s, d, n, lo)     => ILKernelGenerator.CopyAndClipMin(s, d, n, lo),
                        (s, d, n, hi)     => ILKernelGenerator.CopyAndClipMax(s, d, n, hi));
                    return @out;
                case NPTypeCode.Decimal:
                    FusedClipScalar<decimal>(lhs, @out, minCast, maxCast, len,
                        (s, d, n, lo, hi) => ILKernelGenerator.CopyAndClip(s, d, n, lo, hi),
                        (s, d, n, lo)     => ILKernelGenerator.CopyAndClipMin(s, d, n, lo),
                        (s, d, n, hi)     => ILKernelGenerator.CopyAndClipMax(s, d, n, hi));
                    return @out;
                case NPTypeCode.Half:
                case NPTypeCode.Complex:
                    // No fused kernel for these — fall back to allocate-then-clip.
                    var copy = Cast(lhs, outType, copy: true);
                    return ClipNDArrayScalarBounds(copy, min, max, outType);
                default:
                    throw new NotSupportedException($"ClipNDArray not supported for dtype {outType}");
            }
        }

        private static unsafe void FusedClipScalar<T>(
            NDArray src, NDArray dst, NDArray minCast, NDArray maxCast, long len,
            FusedBoth<T> both, FusedMin<T> minOnly, FusedMax<T> maxOnly)
            where T : unmanaged
        {
            var s = (T*)src.Address;
            var d = (T*)dst.Address;
            if (minCast is not null && maxCast is not null)
                both(s, d, len, *(T*)minCast.Address, *(T*)maxCast.Address);
            else if (minCast is not null)
                minOnly(s, d, len, *(T*)minCast.Address);
            else
                maxOnly(s, d, len, *(T*)maxCast.Address);
        }

        private unsafe delegate void FusedBoth<T>(T* s, T* d, long n, T lo, T hi) where T : unmanaged;
        private unsafe delegate void FusedMin<T>(T* s, T* d, long n, T lo) where T : unmanaged;
        private unsafe delegate void FusedMax<T>(T* s, T* d, long n, T hi) where T : unmanaged;

        /// <summary>
        /// Fast path when min and max are both 0-d scalars (or one is null).
        /// Dispatches to the scalar-bound SIMD kernels (`ClipUnified` /
        /// `ClipMinUnified` / `ClipMaxUnified`) which broadcast the scalar
        /// internally within the vector loop — avoiding the cost of
        /// materializing a `len`-sized bound array.
        ///
        /// Handles strided/sliced `@out` via the *Unified kernels (which
        /// switch to the strided variant when `@out.Shape.IsContiguous` is
        /// false).
        /// </summary>
        private unsafe NDArray ClipNDArrayScalarBounds(NDArray @out, NDArray min, NDArray max, NPTypeCode outType)
        {
            long len = @out.size;

            // Convert scalar bounds to outType. `astype` on a 0-d array is O(1).
            var minCast = min is null ? null : min.astype(outType);
            var maxCast = max is null ? null : max.astype(outType);

            switch (outType)
            {
                case NPTypeCode.Byte:
                    ClipScalar<byte>(@out, minCast, maxCast, len,
                        (d, s, lo, hi, sh) => ILKernelGenerator.ClipUnified(d, s, lo, hi, sh),
                        (d, s, lo,     sh) => ILKernelGenerator.ClipMinUnified(d, s, lo, sh),
                        (d, s,     hi, sh) => ILKernelGenerator.ClipMaxUnified(d, s, hi, sh));
                    return @out;
                case NPTypeCode.SByte:
                    ClipScalar<sbyte>(@out, minCast, maxCast, len,
                        (d, s, lo, hi, sh) => ILKernelGenerator.ClipUnified(d, s, lo, hi, sh),
                        (d, s, lo,     sh) => ILKernelGenerator.ClipMinUnified(d, s, lo, sh),
                        (d, s,     hi, sh) => ILKernelGenerator.ClipMaxUnified(d, s, hi, sh));
                    return @out;
                case NPTypeCode.Int16:
                    ClipScalar<short>(@out, minCast, maxCast, len,
                        (d, s, lo, hi, sh) => ILKernelGenerator.ClipUnified(d, s, lo, hi, sh),
                        (d, s, lo,     sh) => ILKernelGenerator.ClipMinUnified(d, s, lo, sh),
                        (d, s,     hi, sh) => ILKernelGenerator.ClipMaxUnified(d, s, hi, sh));
                    return @out;
                case NPTypeCode.UInt16:
                    ClipScalar<ushort>(@out, minCast, maxCast, len,
                        (d, s, lo, hi, sh) => ILKernelGenerator.ClipUnified(d, s, lo, hi, sh),
                        (d, s, lo,     sh) => ILKernelGenerator.ClipMinUnified(d, s, lo, sh),
                        (d, s,     hi, sh) => ILKernelGenerator.ClipMaxUnified(d, s, hi, sh));
                    return @out;
                case NPTypeCode.Int32:
                    ClipScalar<int>(@out, minCast, maxCast, len,
                        (d, s, lo, hi, sh) => ILKernelGenerator.ClipUnified(d, s, lo, hi, sh),
                        (d, s, lo,     sh) => ILKernelGenerator.ClipMinUnified(d, s, lo, sh),
                        (d, s,     hi, sh) => ILKernelGenerator.ClipMaxUnified(d, s, hi, sh));
                    return @out;
                case NPTypeCode.UInt32:
                    ClipScalar<uint>(@out, minCast, maxCast, len,
                        (d, s, lo, hi, sh) => ILKernelGenerator.ClipUnified(d, s, lo, hi, sh),
                        (d, s, lo,     sh) => ILKernelGenerator.ClipMinUnified(d, s, lo, sh),
                        (d, s,     hi, sh) => ILKernelGenerator.ClipMaxUnified(d, s, hi, sh));
                    return @out;
                case NPTypeCode.Int64:
                    ClipScalar<long>(@out, minCast, maxCast, len,
                        (d, s, lo, hi, sh) => ILKernelGenerator.ClipUnified(d, s, lo, hi, sh),
                        (d, s, lo,     sh) => ILKernelGenerator.ClipMinUnified(d, s, lo, sh),
                        (d, s,     hi, sh) => ILKernelGenerator.ClipMaxUnified(d, s, hi, sh));
                    return @out;
                case NPTypeCode.UInt64:
                    ClipScalar<ulong>(@out, minCast, maxCast, len,
                        (d, s, lo, hi, sh) => ILKernelGenerator.ClipUnified(d, s, lo, hi, sh),
                        (d, s, lo,     sh) => ILKernelGenerator.ClipMinUnified(d, s, lo, sh),
                        (d, s,     hi, sh) => ILKernelGenerator.ClipMaxUnified(d, s, hi, sh));
                    return @out;
                case NPTypeCode.Single:
                    ClipScalar<float>(@out, minCast, maxCast, len,
                        (d, s, lo, hi, sh) => ILKernelGenerator.ClipUnified(d, s, lo, hi, sh),
                        (d, s, lo,     sh) => ILKernelGenerator.ClipMinUnified(d, s, lo, sh),
                        (d, s,     hi, sh) => ILKernelGenerator.ClipMaxUnified(d, s, hi, sh));
                    return @out;
                case NPTypeCode.Double:
                    ClipScalar<double>(@out, minCast, maxCast, len,
                        (d, s, lo, hi, sh) => ILKernelGenerator.ClipUnified(d, s, lo, hi, sh),
                        (d, s, lo,     sh) => ILKernelGenerator.ClipMinUnified(d, s, lo, sh),
                        (d, s,     hi, sh) => ILKernelGenerator.ClipMaxUnified(d, s, hi, sh));
                    return @out;
                case NPTypeCode.Decimal:
                    ClipScalar<decimal>(@out, minCast, maxCast, len,
                        (d, s, lo, hi, sh) => ILKernelGenerator.ClipUnified(d, s, lo, hi, sh),
                        (d, s, lo,     sh) => ILKernelGenerator.ClipMinUnified(d, s, lo, sh),
                        (d, s,     hi, sh) => ILKernelGenerator.ClipMaxUnified(d, s, hi, sh));
                    return @out;
                case NPTypeCode.Char:
                    ClipScalar<char>(@out, minCast, maxCast, len,
                        (d, s, lo, hi, sh) => ILKernelGenerator.ClipUnified(d, s, lo, hi, sh),
                        (d, s, lo,     sh) => ILKernelGenerator.ClipMinUnified(d, s, lo, sh),
                        (d, s,     hi, sh) => ILKernelGenerator.ClipMaxUnified(d, s, hi, sh));
                    return @out;
                case NPTypeCode.Half:
                    // Half is not IComparable<Half> in some BCL versions; fall
                    // back to the array-bound path (cheap broadcast for 0-d).
                    return ClipNDArrayScalarBoundsFallback(@out, min, max, outType);
                case NPTypeCode.Complex:
                    // Complex has lex ordering with NaN propagation — keep the
                    // existing array-bound path which already implements the
                    // exact semantics.
                    return ClipNDArrayScalarBoundsFallback(@out, min, max, outType);
                default:
                    throw new NotSupportedException($"ClipNDArray not supported for dtype {outType}");
            }
        }

        // Helper extracts the scalar value(s) from 0-d NDArrays and dispatches
        // to the appropriate Unified kernel. Inlined-by-JIT delegate avoids
        // duplicating the dispatch logic 12 times.
        private static unsafe void ClipScalar<T>(
            NDArray @out, NDArray minCast, NDArray maxCast, long len,
            BothBoundsKernel<T> both,
            MinKernel<T> minOnly,
            MaxKernel<T> maxOnly)
            where T : unmanaged
        {
            var dst = (T*)@out.Address;
            if (minCast is not null && maxCast is not null)
            {
                T lo = *(T*)minCast.Address;
                T hi = *(T*)maxCast.Address;
                both(dst, len, lo, hi, @out.Shape);
            }
            else if (minCast is not null)
            {
                T lo = *(T*)minCast.Address;
                minOnly(dst, len, lo, @out.Shape);
            }
            else
            {
                T hi = *(T*)maxCast.Address;
                maxOnly(dst, len, hi, @out.Shape);
            }
        }

        private unsafe delegate void BothBoundsKernel<T>(T* data, long size, T lo, T hi, Shape shape) where T : unmanaged;
        private unsafe delegate void MinKernel<T>(T* data, long size, T lo, Shape shape) where T : unmanaged;
        private unsafe delegate void MaxKernel<T>(T* data, long size, T hi, Shape shape) where T : unmanaged;

        // Fallback for dtypes (Half, Complex) whose scalar SIMD kernels aren't
        // wired. Materializes the 0-d bound by broadcast — still much cheaper
        // than the full slow path because `np.broadcast_to(0-d, shape)` returns
        // a stride-0 view and `astype` only allocates `1` element.
        private NDArray ClipNDArrayScalarBoundsFallback(NDArray @out, NDArray min, NDArray max, NPTypeCode outType)
        {
            var _min = min is null ? null : np.broadcast_to(min, @out.Shape).astype(outType);
            var _max = max is null ? null : np.broadcast_to(max, @out.Shape).astype(outType);

            long len = @out.size;
            bool canUseFastPath = @out.Shape.IsContiguous && @out.Shape.Offset == 0;
            if (!(_min is null) && canUseFastPath)
                canUseFastPath = _min.Shape.IsContiguous && _min.Shape.Offset == 0;
            if (!(_max is null) && canUseFastPath)
                canUseFastPath = _max.Shape.IsContiguous && _max.Shape.Offset == 0;

            return canUseFastPath
                ? ClipNDArrayContiguous(@out, _min, _max, len)
                : ClipNDArrayGeneral(@out, _min, _max, len);
        }

        /// <summary>
        /// Fast path for contiguous arrays - uses IL kernel with SIMD support.
        /// </summary>
        private unsafe NDArray ClipNDArrayContiguous(NDArray @out, NDArray min, NDArray max, long len)
        {
            if (!(min is null) && !(max is null))
            {
                // Both bounds - use ClipArrayBounds
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ILKernelGenerator.ClipArrayBounds((byte*)@out.Address, (byte*)min.Address, (byte*)max.Address, len);
                        return @out;
                    case NPTypeCode.SByte:
                        ILKernelGenerator.ClipArrayBounds((sbyte*)@out.Address, (sbyte*)min.Address, (sbyte*)max.Address, len);
                        return @out;
                    case NPTypeCode.Int16:
                        ILKernelGenerator.ClipArrayBounds((short*)@out.Address, (short*)min.Address, (short*)max.Address, len);
                        return @out;
                    case NPTypeCode.UInt16:
                        ILKernelGenerator.ClipArrayBounds((ushort*)@out.Address, (ushort*)min.Address, (ushort*)max.Address, len);
                        return @out;
                    case NPTypeCode.Int32:
                        ILKernelGenerator.ClipArrayBounds((int*)@out.Address, (int*)min.Address, (int*)max.Address, len);
                        return @out;
                    case NPTypeCode.UInt32:
                        ILKernelGenerator.ClipArrayBounds((uint*)@out.Address, (uint*)min.Address, (uint*)max.Address, len);
                        return @out;
                    case NPTypeCode.Int64:
                        ILKernelGenerator.ClipArrayBounds((long*)@out.Address, (long*)min.Address, (long*)max.Address, len);
                        return @out;
                    case NPTypeCode.UInt64:
                        ILKernelGenerator.ClipArrayBounds((ulong*)@out.Address, (ulong*)min.Address, (ulong*)max.Address, len);
                        return @out;
                    case NPTypeCode.Single:
                        ILKernelGenerator.ClipArrayBounds((float*)@out.Address, (float*)min.Address, (float*)max.Address, len);
                        return @out;
                    case NPTypeCode.Double:
                        ILKernelGenerator.ClipArrayBounds((double*)@out.Address, (double*)min.Address, (double*)max.Address, len);
                        return @out;
                    case NPTypeCode.Decimal:
                        ClipArrayBoundsDecimal((decimal*)@out.Address, (decimal*)min.Address, (decimal*)max.Address, len);
                        return @out;
                    case NPTypeCode.Char:
                        ClipArrayBoundsChar((char*)@out.Address, (char*)min.Address, (char*)max.Address, len);
                        return @out;
                    case NPTypeCode.Half:
                        ClipArrayBoundsHalf((Half*)@out.Address, (Half*)min.Address, (Half*)max.Address, len);
                        return @out;
                    case NPTypeCode.Complex:
                        ClipArrayBoundsComplex((Complex*)@out.Address, (Complex*)min.Address, (Complex*)max.Address, len);
                        return @out;
                    default:
                        throw new NotSupportedException($"ClipNDArray not supported for dtype {@out.GetTypeCode}");
                }
            }
            else if (!(min is null))
            {
                // Min only - use ClipArrayMin
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ILKernelGenerator.ClipArrayMin((byte*)@out.Address, (byte*)min.Address, len);
                        return @out;
                    case NPTypeCode.SByte:
                        ILKernelGenerator.ClipArrayMin((sbyte*)@out.Address, (sbyte*)min.Address, len);
                        return @out;
                    case NPTypeCode.Int16:
                        ILKernelGenerator.ClipArrayMin((short*)@out.Address, (short*)min.Address, len);
                        return @out;
                    case NPTypeCode.UInt16:
                        ILKernelGenerator.ClipArrayMin((ushort*)@out.Address, (ushort*)min.Address, len);
                        return @out;
                    case NPTypeCode.Int32:
                        ILKernelGenerator.ClipArrayMin((int*)@out.Address, (int*)min.Address, len);
                        return @out;
                    case NPTypeCode.UInt32:
                        ILKernelGenerator.ClipArrayMin((uint*)@out.Address, (uint*)min.Address, len);
                        return @out;
                    case NPTypeCode.Int64:
                        ILKernelGenerator.ClipArrayMin((long*)@out.Address, (long*)min.Address, len);
                        return @out;
                    case NPTypeCode.UInt64:
                        ILKernelGenerator.ClipArrayMin((ulong*)@out.Address, (ulong*)min.Address, len);
                        return @out;
                    case NPTypeCode.Single:
                        ILKernelGenerator.ClipArrayMin((float*)@out.Address, (float*)min.Address, len);
                        return @out;
                    case NPTypeCode.Double:
                        ILKernelGenerator.ClipArrayMin((double*)@out.Address, (double*)min.Address, len);
                        return @out;
                    case NPTypeCode.Decimal:
                        ClipArrayMinDecimal((decimal*)@out.Address, (decimal*)min.Address, len);
                        return @out;
                    case NPTypeCode.Char:
                        ClipArrayMinChar((char*)@out.Address, (char*)min.Address, len);
                        return @out;
                    case NPTypeCode.Half:
                        ClipArrayMinHalf((Half*)@out.Address, (Half*)min.Address, len);
                        return @out;
                    case NPTypeCode.Complex:
                        ClipArrayMinComplex((Complex*)@out.Address, (Complex*)min.Address, len);
                        return @out;
                    default:
                        throw new NotSupportedException($"ClipNDArray not supported for dtype {@out.GetTypeCode}");
                }
            }
            else // max is not null
            {
                // Max only - use ClipArrayMax
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ILKernelGenerator.ClipArrayMax((byte*)@out.Address, (byte*)max.Address, len);
                        return @out;
                    case NPTypeCode.SByte:
                        ILKernelGenerator.ClipArrayMax((sbyte*)@out.Address, (sbyte*)max.Address, len);
                        return @out;
                    case NPTypeCode.Int16:
                        ILKernelGenerator.ClipArrayMax((short*)@out.Address, (short*)max.Address, len);
                        return @out;
                    case NPTypeCode.UInt16:
                        ILKernelGenerator.ClipArrayMax((ushort*)@out.Address, (ushort*)max.Address, len);
                        return @out;
                    case NPTypeCode.Int32:
                        ILKernelGenerator.ClipArrayMax((int*)@out.Address, (int*)max.Address, len);
                        return @out;
                    case NPTypeCode.UInt32:
                        ILKernelGenerator.ClipArrayMax((uint*)@out.Address, (uint*)max.Address, len);
                        return @out;
                    case NPTypeCode.Int64:
                        ILKernelGenerator.ClipArrayMax((long*)@out.Address, (long*)max.Address, len);
                        return @out;
                    case NPTypeCode.UInt64:
                        ILKernelGenerator.ClipArrayMax((ulong*)@out.Address, (ulong*)max.Address, len);
                        return @out;
                    case NPTypeCode.Single:
                        ILKernelGenerator.ClipArrayMax((float*)@out.Address, (float*)max.Address, len);
                        return @out;
                    case NPTypeCode.Double:
                        ILKernelGenerator.ClipArrayMax((double*)@out.Address, (double*)max.Address, len);
                        return @out;
                    case NPTypeCode.Decimal:
                        ClipArrayMaxDecimal((decimal*)@out.Address, (decimal*)max.Address, len);
                        return @out;
                    case NPTypeCode.Char:
                        ClipArrayMaxChar((char*)@out.Address, (char*)max.Address, len);
                        return @out;
                    case NPTypeCode.Half:
                        ClipArrayMaxHalf((Half*)@out.Address, (Half*)max.Address, len);
                        return @out;
                    case NPTypeCode.Complex:
                        ClipArrayMaxComplex((Complex*)@out.Address, (Complex*)max.Address, len);
                        return @out;
                    default:
                        throw new NotSupportedException($"ClipNDArray not supported for dtype {@out.GetTypeCode}");
                }
            }
        }

        /// <summary>
        /// General path for non-contiguous/broadcast arrays - uses GetAtIndex for element access.
        /// </summary>
        private unsafe NDArray ClipNDArrayGeneral(NDArray @out, NDArray min, NDArray max, long len)
        {
            if (!(min is null) && !(max is null))
            {
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ClipNDArrayGeneralCore<byte>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.SByte:
                        ClipNDArrayGeneralCore<sbyte>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Int16:
                        ClipNDArrayGeneralCore<short>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.UInt16:
                        ClipNDArrayGeneralCore<ushort>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Int32:
                        ClipNDArrayGeneralCore<int>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.UInt32:
                        ClipNDArrayGeneralCore<uint>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Int64:
                        ClipNDArrayGeneralCore<long>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.UInt64:
                        ClipNDArrayGeneralCore<ulong>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Single:
                        ClipNDArrayGeneralCore<float>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Double:
                        ClipNDArrayGeneralCore<double>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Decimal:
                        ClipNDArrayGeneralCore<decimal>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Char:
                        ClipNDArrayGeneralCore<char>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Half:
                        ClipNDArrayGeneralCoreHalf(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Complex:
                        ClipNDArrayGeneralCoreComplex(@out, min, max, len);
                        return @out;
                    default:
                        throw new NotSupportedException($"ClipNDArray not supported for dtype {@out.GetTypeCode}");
                }
            }
            else if (!(min is null))
            {
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ClipNDArrayMinGeneralCore<byte>(@out, min, len);
                        return @out;
                    case NPTypeCode.SByte:
                        ClipNDArrayMinGeneralCore<sbyte>(@out, min, len);
                        return @out;
                    case NPTypeCode.Int16:
                        ClipNDArrayMinGeneralCore<short>(@out, min, len);
                        return @out;
                    case NPTypeCode.UInt16:
                        ClipNDArrayMinGeneralCore<ushort>(@out, min, len);
                        return @out;
                    case NPTypeCode.Int32:
                        ClipNDArrayMinGeneralCore<int>(@out, min, len);
                        return @out;
                    case NPTypeCode.UInt32:
                        ClipNDArrayMinGeneralCore<uint>(@out, min, len);
                        return @out;
                    case NPTypeCode.Int64:
                        ClipNDArrayMinGeneralCore<long>(@out, min, len);
                        return @out;
                    case NPTypeCode.UInt64:
                        ClipNDArrayMinGeneralCore<ulong>(@out, min, len);
                        return @out;
                    case NPTypeCode.Single:
                        ClipNDArrayMinGeneralCore<float>(@out, min, len);
                        return @out;
                    case NPTypeCode.Double:
                        ClipNDArrayMinGeneralCore<double>(@out, min, len);
                        return @out;
                    case NPTypeCode.Decimal:
                        ClipNDArrayMinGeneralCore<decimal>(@out, min, len);
                        return @out;
                    case NPTypeCode.Char:
                        ClipNDArrayMinGeneralCore<char>(@out, min, len);
                        return @out;
                    case NPTypeCode.Half:
                        ClipNDArrayMinGeneralCoreHalf(@out, min, len);
                        return @out;
                    case NPTypeCode.Complex:
                        ClipNDArrayMinGeneralCoreComplex(@out, min, len);
                        return @out;
                    default:
                        throw new NotSupportedException($"ClipNDArray not supported for dtype {@out.GetTypeCode}");
                }
            }
            else // max is not null
            {
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ClipNDArrayMaxGeneralCore<byte>(@out, max, len);
                        return @out;
                    case NPTypeCode.SByte:
                        ClipNDArrayMaxGeneralCore<sbyte>(@out, max, len);
                        return @out;
                    case NPTypeCode.Int16:
                        ClipNDArrayMaxGeneralCore<short>(@out, max, len);
                        return @out;
                    case NPTypeCode.UInt16:
                        ClipNDArrayMaxGeneralCore<ushort>(@out, max, len);
                        return @out;
                    case NPTypeCode.Int32:
                        ClipNDArrayMaxGeneralCore<int>(@out, max, len);
                        return @out;
                    case NPTypeCode.UInt32:
                        ClipNDArrayMaxGeneralCore<uint>(@out, max, len);
                        return @out;
                    case NPTypeCode.Int64:
                        ClipNDArrayMaxGeneralCore<long>(@out, max, len);
                        return @out;
                    case NPTypeCode.UInt64:
                        ClipNDArrayMaxGeneralCore<ulong>(@out, max, len);
                        return @out;
                    case NPTypeCode.Single:
                        ClipNDArrayMaxGeneralCore<float>(@out, max, len);
                        return @out;
                    case NPTypeCode.Double:
                        ClipNDArrayMaxGeneralCore<double>(@out, max, len);
                        return @out;
                    case NPTypeCode.Decimal:
                        ClipNDArrayMaxGeneralCore<decimal>(@out, max, len);
                        return @out;
                    case NPTypeCode.Char:
                        ClipNDArrayMaxGeneralCore<char>(@out, max, len);
                        return @out;
                    case NPTypeCode.Half:
                        ClipNDArrayMaxGeneralCoreHalf(@out, max, len);
                        return @out;
                    case NPTypeCode.Complex:
                        ClipNDArrayMaxGeneralCoreComplex(@out, max, len);
                        return @out;
                    default:
                        throw new NotSupportedException($"ClipNDArray not supported for dtype {@out.GetTypeCode}");
                }
            }
        }

        #region General Path Core Methods

        private static unsafe void ClipNDArrayGeneralCore<T>(NDArray @out, NDArray min, NDArray max, long len)
            where T : unmanaged, IComparable<T>
        {
            // Use specialized implementations for float/double to handle NaN correctly
            if (typeof(T) == typeof(float))
            {
                ClipNDArrayGeneralCoreFloat(@out, min, max, len);
                return;
            }
            if (typeof(T) == typeof(double))
            {
                ClipNDArrayGeneralCoreDouble(@out, min, max, len);
                return;
            }

            var outAddr = (T*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ChangeType<T>(min.GetAtIndex(i));
                var maxVal = Converts.ChangeType<T>(max.GetAtIndex(i));

                // NumPy semantics: min(max(val, minVal), maxVal)
                if (val.CompareTo(minVal) < 0)
                    val = minVal;
                if (val.CompareTo(maxVal) > 0)
                    val = maxVal;
                outAddr[outOffset] = val;
            }
        }

        private static unsafe void ClipNDArrayMinGeneralCore<T>(NDArray @out, NDArray min, long len)
            where T : unmanaged, IComparable<T>
        {
            // Use specialized implementations for float/double to handle NaN correctly
            if (typeof(T) == typeof(float))
            {
                ClipNDArrayMinGeneralCoreFloat(@out, min, len);
                return;
            }
            if (typeof(T) == typeof(double))
            {
                ClipNDArrayMinGeneralCoreDouble(@out, min, len);
                return;
            }

            var outAddr = (T*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ChangeType<T>(min.GetAtIndex(i));

                if (val.CompareTo(minVal) < 0)
                    outAddr[outOffset] = minVal;
            }
        }

        private static unsafe void ClipNDArrayMaxGeneralCore<T>(NDArray @out, NDArray max, long len)
            where T : unmanaged, IComparable<T>
        {
            // Use specialized implementations for float/double to handle NaN correctly
            if (typeof(T) == typeof(float))
            {
                ClipNDArrayMaxGeneralCoreFloat(@out, max, len);
                return;
            }
            if (typeof(T) == typeof(double))
            {
                ClipNDArrayMaxGeneralCoreDouble(@out, max, len);
                return;
            }

            var outAddr = (T*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var maxVal = Converts.ChangeType<T>(max.GetAtIndex(i));

                if (val.CompareTo(maxVal) > 0)
                    outAddr[outOffset] = maxVal;
            }
        }

        #region Floating-Point General Path (NaN-aware)

        // These use Math.Max/Min which properly propagate NaN per IEEE semantics

        private static unsafe void ClipNDArrayGeneralCoreFloat(NDArray @out, NDArray min, NDArray max, long len)
        {
            var outAddr = (float*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToSingle(min.GetAtIndex(i));
                var maxVal = Converts.ToSingle(max.GetAtIndex(i));
                outAddr[outOffset] = Math.Min(Math.Max(val, minVal), maxVal);
            }
        }

        private static unsafe void ClipNDArrayGeneralCoreDouble(NDArray @out, NDArray min, NDArray max, long len)
        {
            var outAddr = (double*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToDouble(min.GetAtIndex(i));
                var maxVal = Converts.ToDouble(max.GetAtIndex(i));
                outAddr[outOffset] = Math.Min(Math.Max(val, minVal), maxVal);
            }
        }

        private static unsafe void ClipNDArrayMinGeneralCoreFloat(NDArray @out, NDArray min, long len)
        {
            var outAddr = (float*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToSingle(min.GetAtIndex(i));
                outAddr[outOffset] = Math.Max(val, minVal);
            }
        }

        private static unsafe void ClipNDArrayMinGeneralCoreDouble(NDArray @out, NDArray min, long len)
        {
            var outAddr = (double*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToDouble(min.GetAtIndex(i));
                outAddr[outOffset] = Math.Max(val, minVal);
            }
        }

        private static unsafe void ClipNDArrayMaxGeneralCoreFloat(NDArray @out, NDArray max, long len)
        {
            var outAddr = (float*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var maxVal = Converts.ToSingle(max.GetAtIndex(i));
                outAddr[outOffset] = Math.Min(val, maxVal);
            }
        }

        private static unsafe void ClipNDArrayMaxGeneralCoreDouble(NDArray @out, NDArray max, long len)
        {
            var outAddr = (double*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var maxVal = Converts.ToDouble(max.GetAtIndex(i));
                outAddr[outOffset] = Math.Min(val, maxVal);
            }
        }

        #endregion

        #endregion

        #region Scalar Fallbacks for Non-SIMD Types (Decimal, Char) - Array Bounds

        private static unsafe void ClipArrayBoundsDecimal(decimal* output, decimal* minArr, decimal* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
            {
                var val = output[i];
                if (val < minArr[i]) val = minArr[i];
                if (val > maxArr[i]) val = maxArr[i];
                output[i] = val;
            }
        }

        private static unsafe void ClipArrayMinDecimal(decimal* output, decimal* minArr, long size)
        {
            for (long i = 0; i < size; i++)
                if (output[i] < minArr[i]) output[i] = minArr[i];
        }

        private static unsafe void ClipArrayMaxDecimal(decimal* output, decimal* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
                if (output[i] > maxArr[i]) output[i] = maxArr[i];
        }

        private static unsafe void ClipArrayBoundsChar(char* output, char* minArr, char* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
            {
                var val = output[i];
                if (val < minArr[i]) val = minArr[i];
                if (val > maxArr[i]) val = maxArr[i];
                output[i] = val;
            }
        }

        private static unsafe void ClipArrayMinChar(char* output, char* minArr, long size)
        {
            for (long i = 0; i < size; i++)
                if (output[i] < minArr[i]) output[i] = minArr[i];
        }

        private static unsafe void ClipArrayMaxChar(char* output, char* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
                if (output[i] > maxArr[i]) output[i] = maxArr[i];
        }

        #endregion

        #region Half Clip (NaN-aware, matches NumPy float16 semantics)

        // NumPy parity for floating point: NaN propagates. If either operand is NaN, result is NaN.
        // Half doesn't have Math.Max/Min — we route through NaN-aware helpers.

        private static Half HalfMaxNaN(Half a, Half b)
        {
            // Matches NumPy np.maximum / clip-min: if either is NaN, result is NaN.
            if (Half.IsNaN(a) || Half.IsNaN(b)) return Half.NaN;
            return a > b ? a : b;
        }

        private static Half HalfMinNaN(Half a, Half b)
        {
            if (Half.IsNaN(a) || Half.IsNaN(b)) return Half.NaN;
            return a < b ? a : b;
        }

        private static unsafe void ClipArrayBoundsHalf(Half* output, Half* minArr, Half* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
                output[i] = HalfMinNaN(HalfMaxNaN(output[i], minArr[i]), maxArr[i]);
        }

        private static unsafe void ClipArrayMinHalf(Half* output, Half* minArr, long size)
        {
            for (long i = 0; i < size; i++)
                output[i] = HalfMaxNaN(output[i], minArr[i]);
        }

        private static unsafe void ClipArrayMaxHalf(Half* output, Half* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
                output[i] = HalfMinNaN(output[i], maxArr[i]);
        }

        private static unsafe void ClipNDArrayGeneralCoreHalf(NDArray @out, NDArray min, NDArray max, long len)
        {
            var outAddr = (Half*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToHalf(min.GetAtIndex(i));
                var maxVal = Converts.ToHalf(max.GetAtIndex(i));
                outAddr[outOffset] = HalfMinNaN(HalfMaxNaN(val, minVal), maxVal);
            }
        }

        private static unsafe void ClipNDArrayMinGeneralCoreHalf(NDArray @out, NDArray min, long len)
        {
            var outAddr = (Half*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToHalf(min.GetAtIndex(i));
                outAddr[outOffset] = HalfMaxNaN(val, minVal);
            }
        }

        private static unsafe void ClipNDArrayMaxGeneralCoreHalf(NDArray @out, NDArray max, long len)
        {
            var outAddr = (Half*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var maxVal = Converts.ToHalf(max.GetAtIndex(i));
                outAddr[outOffset] = HalfMinNaN(val, maxVal);
            }
        }

        #endregion

        #region Complex Clip (lex ordering, NaN propagation)

        // NumPy parity for complex: np.maximum/minimum use lex ordering on (real, imag).
        // "NaN-containing" = double.IsNaN(Real) || double.IsNaN(Imaginary).
        // NaN propagation: if either operand is NaN-containing, return it (first wins when both NaN).
        // For clip-min (≡ max(val, minBound)): passes the larger; if either is NaN, returns "val"
        // then "minBound" rule — doesn't matter which since both paths return the NaN-carrier.

        private static bool ComplexIsNaN(Complex z)
            => double.IsNaN(z.Real) || double.IsNaN(z.Imaginary);

        private static bool ComplexLexGreater(Complex a, Complex b)
        {
            // a > b lex: a.real > b.real OR (a.real == b.real AND a.imag > b.imag)
            if (a.Real > b.Real) return true;
            if (a.Real < b.Real) return false;
            return a.Imaginary > b.Imaginary;
        }

        private static Complex ComplexMaxNaN(Complex a, Complex b)
        {
            // NumPy: first NaN wins. If a is NaN-containing, return a regardless of b.
            if (ComplexIsNaN(a)) return a;
            if (ComplexIsNaN(b)) return b;
            return ComplexLexGreater(a, b) ? a : b;
        }

        private static Complex ComplexMinNaN(Complex a, Complex b)
        {
            if (ComplexIsNaN(a)) return a;
            if (ComplexIsNaN(b)) return b;
            return ComplexLexGreater(a, b) ? b : a;
        }

        private static unsafe void ClipArrayBoundsComplex(Complex* output, Complex* minArr, Complex* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
                output[i] = ComplexMinNaN(ComplexMaxNaN(output[i], minArr[i]), maxArr[i]);
        }

        private static unsafe void ClipArrayMinComplex(Complex* output, Complex* minArr, long size)
        {
            for (long i = 0; i < size; i++)
                output[i] = ComplexMaxNaN(output[i], minArr[i]);
        }

        private static unsafe void ClipArrayMaxComplex(Complex* output, Complex* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
                output[i] = ComplexMinNaN(output[i], maxArr[i]);
        }

        private static unsafe void ClipNDArrayGeneralCoreComplex(NDArray @out, NDArray min, NDArray max, long len)
        {
            var outAddr = (Complex*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToComplex(min.GetAtIndex(i));
                var maxVal = Converts.ToComplex(max.GetAtIndex(i));
                outAddr[outOffset] = ComplexMinNaN(ComplexMaxNaN(val, minVal), maxVal);
            }
        }

        private static unsafe void ClipNDArrayMinGeneralCoreComplex(NDArray @out, NDArray min, long len)
        {
            var outAddr = (Complex*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToComplex(min.GetAtIndex(i));
                outAddr[outOffset] = ComplexMaxNaN(val, minVal);
            }
        }

        private static unsafe void ClipNDArrayMaxGeneralCoreComplex(NDArray @out, NDArray max, long len)
        {
            var outAddr = (Complex*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var maxVal = Converts.ToComplex(max.GetAtIndex(i));
                outAddr[outOffset] = ComplexMinNaN(val, maxVal);
            }
        }

        #endregion
    }
}
