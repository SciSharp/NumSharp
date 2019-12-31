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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected internal static NDArray<int> GetIndicesFromSlice(Shape shape, Slice slice, int axis)
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected internal static NDArray<int> GetIndicesFromSlice(int[] shape, Slice slice, int axis)
        {
            var dim = shape[axis];
            var slice_def = slice.ToSliceDef(dim); // this resolves negative slice indices
            return np.arange(slice_def.Start, slice_def.Start + slice_def.Step * slice_def.Count, slice.Step).MakeGeneric<int>();
        }

        /// <summary>
        ///     Generates index getter function based on given <paramref name="indices"/>.
        /// </summary>
        /// <param name="srcShape">The shape to get indice from</param>
        /// <param name="indices">The indices trying to index.</param>
        protected static unsafe Func<int, int>[] PrepareIndexGetters(Shape srcShape, NDArray[] indices)
        {
            var indexGetters = new Func<int, int>[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                var idxs = indices[i];
                var dimensionSize = srcShape[i];
                var idxAddr = (int*)idxs.Address;

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
                        if (srcShape.IsBroadcasted)
                        {
                            if (srcShape.IsSliced)
                                flatSrcShape.ViewInfo = new ViewInfo() {ParentShape = srcShape.BroadcastInfo.OriginalShape, Slices = null};
                        }
                        else if (srcShape.IsSliced)
                            // Set up the new shape (of reshaped slice) to recursively represent a shape within a sliced shape
                            flatSrcShape.ViewInfo = new ViewInfo() {ParentShape = srcShape, Slices = null};

                        indexGetters[i] = flatSrcShape.GetOffset_1D;
                    }
                }
                else
                {
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
                        Func<int, int> offset = idxs.Shape.GetOffset_1D;
                        indexGetters[i] = idx =>
                        {
                            var val = idxAddr[offset(idx)];
                            if (val < 0)
                                return dimensionSize + val;
                            return val;
                        };
                    }
                }
            }

            return indexGetters;
        }
    }
}
