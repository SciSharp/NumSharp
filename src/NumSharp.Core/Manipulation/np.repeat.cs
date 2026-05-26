using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Repeat each element of an array after themselves.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="repeats">The number of repetitions for each element.</param>
        /// <param name="axis">Axis along which to repeat values. <c>null</c> (NumPy <c>None</c>) flattens the input and returns a flat array.</param>
        /// <returns>Output array which has the same shape as <paramref name="a"/>, except along <paramref name="axis"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html</remarks>
        public static NDArray repeat(NDArray a, int repeats, int? axis = null)
            => repeat(a, (long)repeats, axis);

        /// <summary>
        ///     Repeat each element of an array after themselves.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="repeats">The number of repetitions for each element.</param>
        /// <param name="axis">Axis along which to repeat values. <c>null</c> (NumPy <c>None</c>) flattens the input.</param>
        /// <returns>Output array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html</remarks>
        public static NDArray repeat(NDArray a, long repeats, int? axis = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (repeats < 0)
                throw new ArgumentException("repeats may not contain negative values");

            if (axis is null)
                return RepeatScalarFlat(a, repeats);

            return RepeatScalarAlongAxis(a, repeats, axis.Value);
        }

        /// <summary>
        ///     Repeat elements of an array with per-element repeat counts. Mirrors NumPy
        ///     <c>np.repeat(a, repeats, axis)</c>: scalar / size-1 <paramref name="repeats"/> broadcasts to
        ///     every element along the (flattened or selected) axis; otherwise the length must match.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="repeats">Repeat counts. Either a 0-d/size-1 array (broadcast) or a 1-D array of length equal to <c>a.size</c> (axis=None) or <c>a.shape[axis]</c>.</param>
        /// <param name="axis">Axis along which to repeat. <c>null</c> flattens the input.</param>
        /// <returns>A new array with elements repeated according to <paramref name="repeats"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html</remarks>
        public static NDArray repeat(NDArray a, NDArray repeats, int? axis = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (repeats is null)
                throw new ArgumentNullException(nameof(repeats));

            // NumPy parity: repeats must be safely castable to int64 — reject float/complex/uint64.
            if (!IsSafeToInt64(repeats.GetTypeCode))
                throw new TypeError($"Cannot cast array data from dtype('{repeats.GetTypeCode.AsNumpyDtypeName()}') to dtype('int64') according to the rule 'safe'");

            if (axis is null)
                return RepeatPerElementFlat(a, repeats);

            return RepeatPerElementAlongAxis(a, repeats, axis.Value);
        }

        /// <summary>
        ///     Repeat a scalar value.
        /// </summary>
        /// <param name="a">Input scalar.</param>
        /// <param name="repeats">The number of repetitions.</param>
        /// <returns>A 1-D array with the scalar repeated.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html</remarks>
        public static unsafe NDArray repeat<T>(T a, int repeats) where T : unmanaged
            => repeat(a, (long)repeats);

        /// <summary>
        ///     Repeat a scalar value.
        /// </summary>
        /// <param name="a">Input scalar.</param>
        /// <param name="repeats">The number of repetitions.</param>
        /// <returns>A 1-D array with the scalar repeated.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.repeat.html</remarks>
        public static unsafe NDArray repeat<T>(T a, long repeats) where T : unmanaged
        {
            if (repeats < 0)
                throw new ArgumentException("repeats may not contain negative values");

            if (repeats == 0)
                return new NDArray(InfoOf<T>.NPTypeCode, Shape.Vector(0));

            var ret = new NDArray(InfoOf<T>.NPTypeCode, Shape.Vector(repeats));
            var dst = (T*)ret.Address;
            for (long j = 0; j < repeats; j++)
                dst[j] = a;
            return ret;
        }

        // ============== axis=None (flatten) paths ==============

        private static unsafe NDArray RepeatScalarFlat(NDArray a, long repeats)
        {
            if (a.size == 0 || repeats == 0)
                return new NDArray(a.GetTypeCode, Shape.Vector(0));

            // ravel() returns a C-contig view (no copy when already contig) or a fresh contig copy.
            NDArray src = a.ravel();
            long total = src.size * repeats;
            var ret = new NDArray(a.GetTypeCode, Shape.Vector(total));

            // Degenerate 3-loop: n_outer=1, n=size, chunk=elsize.
            var kernel = DirectILKernelGenerator.GetRepeatBroadcastKernel(a.dtypesize);
            kernel(
                src: (byte*)src.Address,
                dst: (byte*)ret.Address,
                n_outer: 1,
                n: src.size,
                count: repeats);

            return ret;
        }

        private static unsafe NDArray RepeatPerElementFlat(NDArray a, NDArray repeats)
        {
            NDArray src = a.ravel();
            NDArray repFlat = repeats.ravel();

            // NumPy: scalar (0-d) or size-1 repeats broadcasts to a.size; anything else must match exactly.
            bool broadcast = repFlat.size == 1 || repeats.ndim == 0;
            if (!broadcast && src.size != repFlat.size)
                throw new ArgumentException(
                    $"operands could not be broadcast together with shape ({src.size},) ({repFlat.size},)");

            if (src.size == 0)
                return new NDArray(a.GetTypeCode, Shape.Vector(0));

            long[] counts;
            long broadcastVal;
            long total = ComputeCounts(repFlat, broadcast, src.size, out counts, out broadcastVal);

            if (total == 0)
                return new NDArray(a.GetTypeCode, Shape.Vector(0));

            var ret = new NDArray(a.GetTypeCode, Shape.Vector(total));

            if (broadcast)
            {
                var kernel = DirectILKernelGenerator.GetRepeatBroadcastKernel(a.dtypesize);
                kernel(
                    src: (byte*)src.Address,
                    dst: (byte*)ret.Address,
                    n_outer: 1,
                    n: src.size,
                    count: broadcastVal);
            }
            else
            {
                var kernel = DirectILKernelGenerator.GetRepeatPerJKernel(a.dtypesize);
                fixed (long* pCounts = counts)
                {
                    kernel(
                        src: (byte*)src.Address,
                        dst: (byte*)ret.Address,
                        n_outer: 1,
                        n: src.size,
                        counts: pCounts);
                }
            }

            return ret;
        }

        // ============== axis-aware paths ==============

        private static unsafe NDArray RepeatScalarAlongAxis(NDArray a, long repeats, int axis)
        {
            int ndim = a.ndim;

            // NumPy: 0-d input with axis=0/-1 is silently promoted to a 1-d, size-1 array.
            if (ndim == 0)
            {
                if (axis != 0 && axis != -1)
                    throw new AxisError(axis, ndim);
                a = a.reshape(1);
                ndim = 1;
            }

            int normalizedAxis = NormalizeAxis(axis, ndim);

            // NumPy's PyArray_CheckAxis(... CARRAY) makes the operand C-contig — the
            // chunked memcpy reads a logically-rectangular slab of inner dims.
            NDArray src = a.Shape.IsContiguous ? a : np.ascontiguousarray(a);

            long[] inDims = src.shape;
            long n = inDims[normalizedAxis];
            long total = n * repeats;

            long[] outDims = (long[])inDims.Clone();
            outDims[normalizedAxis] = total;
            var ret = new NDArray(a.GetTypeCode, new Shape(outDims));

            if (src.size == 0 || total == 0)
                return ret;

            ComputeAxisGeometry(inDims, normalizedAxis, out long n_outer, out long nel);
            int chunkBytes = checked((int)(nel * a.dtypesize));

            var kernel = DirectILKernelGenerator.GetRepeatBroadcastKernel(chunkBytes);
            kernel(
                src: (byte*)src.Address,
                dst: (byte*)ret.Address,
                n_outer: n_outer,
                n: n,
                count: repeats);

            return ret;
        }

        private static unsafe NDArray RepeatPerElementAlongAxis(NDArray a, NDArray repeats, int axis)
        {
            int ndim = a.ndim;
            if (ndim == 0)
            {
                if (axis != 0 && axis != -1)
                    throw new AxisError(axis, ndim);
                a = a.reshape(1);
                ndim = 1;
            }

            int normalizedAxis = NormalizeAxis(axis, ndim);

            NDArray src = a.Shape.IsContiguous ? a : np.ascontiguousarray(a);
            NDArray repFlat = repeats.ravel();

            long[] inDims = src.shape;
            long n = inDims[normalizedAxis];

            // NumPy parity: scalar (0-d) or size-1 repeats broadcasts along the axis; otherwise the
            // size must match the axis length exactly.
            bool broadcast = repFlat.size == 1 || repeats.ndim == 0;
            if (!broadcast && repFlat.size != n)
                throw new ArgumentException(
                    $"operands could not be broadcast together with shape ({n},) ({repFlat.size},)");

            long[] counts;
            long broadcastVal;
            long total = ComputeCounts(repFlat, broadcast, n, out counts, out broadcastVal);

            long[] outDims = (long[])inDims.Clone();
            outDims[normalizedAxis] = total;
            var ret = new NDArray(a.GetTypeCode, new Shape(outDims));

            if (src.size == 0 || total == 0)
                return ret;

            ComputeAxisGeometry(inDims, normalizedAxis, out long n_outer, out long nel);
            int chunkBytes = checked((int)(nel * a.dtypesize));

            if (broadcast)
            {
                var kernel = DirectILKernelGenerator.GetRepeatBroadcastKernel(chunkBytes);
                kernel(
                    src: (byte*)src.Address,
                    dst: (byte*)ret.Address,
                    n_outer: n_outer,
                    n: n,
                    count: broadcastVal);
            }
            else
            {
                var kernel = DirectILKernelGenerator.GetRepeatPerJKernel(chunkBytes);
                fixed (long* pCounts = counts)
                {
                    kernel(
                        src: (byte*)src.Address,
                        dst: (byte*)ret.Address,
                        n_outer: n_outer,
                        n: n,
                        counts: pCounts);
                }
            }

            return ret;
        }

        // ============== helpers ==============

        private static int NormalizeAxis(int axis, int ndim)
        {
            int original = axis;
            if (axis < 0)
                axis += ndim;
            if (axis < 0 || axis >= ndim)
                throw new AxisError(original, ndim);
            return axis;
        }

        private static void ComputeAxisGeometry(long[] dims, int axis, out long n_outer, out long nel)
        {
            n_outer = 1;
            for (int i = 0; i < axis; i++) n_outer *= dims[i];
            nel = 1;
            for (int i = axis + 1; i < dims.Length; i++) nel *= dims[i];
        }

        // Materializes the per-j repeat counts as a long[] and validates non-negative.
        // Returns the total output size along the axis.
        private static long ComputeCounts(NDArray repFlat, bool broadcast, long n, out long[] counts, out long broadcastVal)
        {
            if (broadcast)
            {
                broadcastVal = Converts.ToInt64(repFlat.GetAtIndex(0));
                if (broadcastVal < 0)
                    throw new ArgumentException("repeats may not contain negative values");
                counts = null;
                return broadcastVal * n;
            }

            broadcastVal = 0;
            counts = new long[n];
            long total = 0;
            for (long j = 0; j < n; j++)
            {
                long c = Converts.ToInt64(repFlat.GetAtIndex(j));
                if (c < 0)
                    throw new ArgumentException("repeats may not contain negative values");
                counts[j] = c;
                total += c;
            }
            return total;
        }

        /// <summary>
        ///     NumPy "safe" casting check for the repeats dtype (target int64).
        ///     Integers that fit in int64 + boolean pass; uint64/float/complex/decimal reject.
        /// </summary>
        private static bool IsSafeToInt64(NPTypeCode code)
        {
            switch (code)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.Char:
                    return true;
                default:
                    return false;
            }
        }
    }
}
