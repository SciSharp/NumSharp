using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        protected static IEnumerable<object> ExpandEllipsis(object[] ndarrays, int ndim)
        {
            // count dimensions without counting ellipsis or newaxis
            var count = ndarrays.OfType<Slice>().Count(slice => !(slice.IsNewAxis || slice.IsEllipsis));

            // expand 
            for (int i = 0; i < ndarrays.Length; i++)
            {
                var obj = ndarrays[i];

                if (obj is Slice slice && slice.IsEllipsis)
                {
                    for (int j = 0; j < ndim - count; j++)
                        yield return Slice.All;
                    continue;
                }

                yield return obj;
            }
        }

        /// <summary>
        ///     Converts a slice to indices for the special case where slices are mixed with NDArrays in this[...]
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="slice"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        [MethodImpl(Inline)]
        protected internal static NDArray<long> GetIndicesFromSlice(Shape shape, Slice slice, int axis)
        {
            return GetIndicesFromSlice(shape.dimensions, slice, axis);
        }

        /// <summary>
        ///     Converts a slice to indices for the special case where slices are mixed with NDArrays in this[...]
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="slice"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        [MethodImpl(Inline)]
        protected internal static NDArray<long> GetIndicesFromSlice(long[] shape, Slice slice, int axis)
        {
            var dim = shape[axis];
            var slice_def = slice.ToSliceDef(dim); // this resolves negative slice indices
            // Use long overload of np.arange for int64 indexing support
            return np.arange(slice_def.Start, slice_def.Start + slice_def.Step * slice_def.Count, slice.Step).MakeGeneric<long>();
        }

        /// <summary>
        ///     Normalizes an index array for fancy indexing.
        ///     NumPy accepts all integer types (int8/16/32/64, uint8/16/32/64) for indexing.
        ///     Non-integer types (float, decimal, char, bool) raise IndexError.
        ///     We keep Int32/Int64 as-is; other integer types are converted to Int64.
        /// </summary>
        /// <param name="indices">The index array to normalize.</param>
        /// <returns>The normalized index array (Int32 or Int64).</returns>
        /// <exception cref="IndexOutOfRangeException">When the index array is not an integer type.</exception>
        [MethodImpl(Inline)]
        protected static NDArray NormalizeIndexArray(NDArray indices)
        {
            var tc = indices.typecode;

            // Int32 and Int64 are the native types for PrepareIndexGetters - keep as-is
            if (tc == NPTypeCode.Int32 || tc == NPTypeCode.Int64)
                return indices;

            // Other integer types: convert to Int64 (widest signed integer)
            // This matches NumPy which accepts all integer types for indexing
            if (tc == NPTypeCode.Byte || tc == NPTypeCode.Int16 || tc == NPTypeCode.UInt16 ||
                tc == NPTypeCode.UInt32 || tc == NPTypeCode.UInt64)
                return indices.astype(NPTypeCode.Int64, copy: true);

            // Non-integer types are not valid for indexing (matches NumPy behavior)
            throw new IndexOutOfRangeException(
                $"arrays used as indices must be of integer type, got {indices.dtype.Name}");
        }

        /// <summary>
        ///     Generates index getter function based on given <paramref name="indices"/>.
        /// </summary>
        /// <param name="srcShape">The shape to get indice from</param>
        /// <param name="indices">The indices trying to index.</param>
        protected static unsafe Func<long, long>[] PrepareIndexGetters(Shape srcShape, NDArray[] indices)
        {
            var indexGetters = new Func<long, long>[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                var idxs = indices[i];
                var dimensionSize = srcShape[i];

                if (idxs is null)
                {
                    if (idxs.Shape.IsContiguous)
                    {
                        indexGetters[i] = idx =>
                        {
                            return idx;
                        };
                    }
                    else
                    {
                        //we are basically flatten the shape.
                        var flatSrcShape = new Shape(srcShape.size);
                        indexGetters[i] = idx => flatSrcShape.GetOffset_1D(idx);
                    }
                }
                else
                {
                    // Handle both int32 and int64 index arrays
                    if (idxs.typecode == NPTypeCode.Int64)
                    {
                        var idxAddr = (long*)idxs.Address;
                        if (idxs.Shape.IsContiguous)
                        {
                            indexGetters[i] = idx =>
                            {
                                var val = idxAddr[idx];
                                if (val < 0)
                                    return dimensionSize + val;
                                return val;
                            };
                        }
                        else
                        {
                            idxs = idxs.flat;
                            var idxShape = idxs.Shape;
                            indexGetters[i] = idx =>
                            {
                                var val = idxAddr[idxShape.GetOffset_1D(idx)];
                                if (val < 0)
                                    return dimensionSize + val;
                                return val;
                            };
                        }
                    }
                    else
                    {
                        // Assume int32 for backward compatibility
                        var idxAddr = (int*)idxs.Address;
                        if (idxs.Shape.IsContiguous)
                        {
                            indexGetters[i] = idx =>
                            {
                                var val = idxAddr[idx];
                                if (val < 0)
                                    return dimensionSize + val;
                                return val;
                            };
                        }
                        else
                        {
                            idxs = idxs.flat;
                            var idxShape = idxs.Shape;
                            indexGetters[i] = idx =>
                            {
                                var val = idxAddr[idxShape.GetOffset_1D(idx)];
                                if (val < 0)
                                    return dimensionSize + val;
                                return val;
                            };
                        }
                    }
                }
            }

            return indexGetters;
        }
    }
}
