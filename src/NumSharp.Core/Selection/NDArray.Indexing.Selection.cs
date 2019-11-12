using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Threading.Tasks;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        private static unsafe NDArray retrieve_indices(NDArray src, NDArray[] indices)
        {
#if _REGEN
            #region Compute
		    switch (src.typecode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: return retrieve_indices<#2>(src.MakeGeneric<#2>(), nds);
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute

            switch (src.typecode)
            {
                case NPTypeCode.Boolean: return retrieve_indices<bool>(src.MakeGeneric<bool>(), indices);
                case NPTypeCode.Byte: return retrieve_indices<byte>(src.MakeGeneric<byte>(), indices);
                case NPTypeCode.Int16: return retrieve_indices<short>(src.MakeGeneric<short>(), indices);
                case NPTypeCode.UInt16: return retrieve_indices<ushort>(src.MakeGeneric<ushort>(), indices);
                case NPTypeCode.Int32: return retrieve_indices<int>(src.MakeGeneric<int>(), indices);
                case NPTypeCode.UInt32: return retrieve_indices<uint>(src.MakeGeneric<uint>(), indices);
                case NPTypeCode.Int64: return retrieve_indices<long>(src.MakeGeneric<long>(), indices);
                case NPTypeCode.UInt64: return retrieve_indices<ulong>(src.MakeGeneric<ulong>(), indices);
                case NPTypeCode.Char: return retrieve_indices<char>(src.MakeGeneric<char>(), indices);
                case NPTypeCode.Double: return retrieve_indices<double>(src.MakeGeneric<double>(), indices);
                case NPTypeCode.Single: return retrieve_indices<float>(src.MakeGeneric<float>(), indices);
                case NPTypeCode.Decimal: return retrieve_indices<decimal>(src.MakeGeneric<decimal>(), indices);
                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif
        }

        //TODO: this
        private static unsafe NDArray set_indices(NDArray src, NDArray[] nds, NDArray dst)
        {
#if _REGEN
            #region Compute
		    switch (src.typecode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: return retrieve_indices<#2>(src.MakeGeneric<#2>(), nds);
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute

            switch (src.typecode)
            {
                //TODO: ....
                case NPTypeCode.Boolean: return retrieve_indices<bool>(src.MakeGeneric<bool>(), nds);
                case NPTypeCode.Byte: return retrieve_indices<byte>(src.MakeGeneric<byte>(), nds);
                case NPTypeCode.Int16: return retrieve_indices<short>(src.MakeGeneric<short>(), nds);
                case NPTypeCode.UInt16: return retrieve_indices<ushort>(src.MakeGeneric<ushort>(), nds);
                case NPTypeCode.Int32: return retrieve_indices<int>(src.MakeGeneric<int>(), nds);
                case NPTypeCode.UInt32: return retrieve_indices<uint>(src.MakeGeneric<uint>(), nds);
                case NPTypeCode.Int64: return retrieve_indices<long>(src.MakeGeneric<long>(), nds);
                case NPTypeCode.UInt64: return retrieve_indices<ulong>(src.MakeGeneric<ulong>(), nds);
                case NPTypeCode.Char: return retrieve_indices<char>(src.MakeGeneric<char>(), nds);
                case NPTypeCode.Double: return retrieve_indices<double>(src.MakeGeneric<double>(), nds);
                case NPTypeCode.Single: return retrieve_indices<float>(src.MakeGeneric<float>(), nds);
                case NPTypeCode.Decimal: return retrieve_indices<decimal>(src.MakeGeneric<decimal>(), nds);
                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif
        }

        private static unsafe NDArray<T> retrieve_indices<T>(NDArray<T> source, NDArray[] indices) where T : unmanaged
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            if (indices == null)
                throw new ArgumentNullException(nameof(indices));
            if (indices.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(indices));

            if (source.Shape.IsScalar)
                source = source.reshape(1);

            int[] retShape = null, subShape = null;

            int indicesSize = indices[0].size;
            var srcShape = source.Shape;
            var ndsCount = indices.Length;
            bool isSubshaped = ndsCount != source.ndim;

            //preprocess indices -----------------------------------------------------------------------------------------------
            //handle non-flat indices and detect if broadcasting required
            if (indices.Length == 1)
            {
                //fast-lane for 1-d.
                var idxs = indices[0];

                //TODO what does this check even test if (nd.shape[0] > source.shape[0])
                //TODO what does this check even test     throw new ArgumentOutOfRangeException($"index {nd.size - 1} is out of bounds for axis 0 with size {nd.shape[0]}");

                if (idxs.Shape.IsEmpty)
                    return new NDArray<T>();

                //resolve retShape
                if (!isSubshaped)
                {
                    retShape = (int[])idxs.shape.Clone();
                }
                else
                {
                    retShape = new int[idxs.ndim + srcShape.NDim - ndsCount];
                    for (int i = 0; i < idxs.ndim; i++)
                        retShape[i] = idxs.shape[i];


                    subShape = new int[srcShape.NDim - ndsCount];
                    for (int dst_i = idxs.ndim, src_i = ndsCount, i = 0; src_i < srcShape.NDim; dst_i++, src_i++, i++)
                    {
                        retShape[dst_i] = srcShape[src_i];
                        subShape[i] = srcShape[src_i];
                    }
                }

                //handle non-flat index
                if (idxs.ndim != 1)
                    idxs = idxs.flat;

                //handle non-int32 index
                if (idxs.typecode != NPTypeCode.Int32)
                    idxs = idxs.astype(NPTypeCode.Int32, true);

                indices[0] = idxs;
            }
            else
            {
                bool broadcastRequired = false;
                for (int i = 0; i < indices.Length; i++)
                {
                    var nd = indices[i];

                    if (nd.Shape.IsEmpty)
                        return new NDArray<T>();

                    //test for broadcasting requirement
                    if (nd.size != indicesSize)
                        broadcastRequired = true;

                    //handle non-int32 index
                    if (nd.typecode != NPTypeCode.Int32)
                        indices[i] = nd.astype(NPTypeCode.Int32);
                }

                //handle broadcasting
                if (broadcastRequired)
                {
                    indices = np.broadcast_arrays(indices);
                    indicesSize = indices[0].size;
                }

                //resolve retShape
                var idxs = indices[0];
                if (!isSubshaped)
                {
                    retShape = (int[])idxs.shape.Clone();
                }
                else
                {
                    retShape = new int[idxs.ndim + srcShape.NDim - ndsCount];
                    for (int i = 0; i < idxs.ndim; i++)
                        retShape[i] = idxs.shape[i];


                    subShape = new int[srcShape.NDim - ndsCount];
                    for (int dst_i = idxs.ndim, src_i = ndsCount, i = 0; src_i < srcShape.NDim; dst_i++, src_i++, i++)
                    {
                        retShape[dst_i] = srcShape[src_i];
                        subShape[i] = srcShape[src_i];
                    }
                }

                //handle non-flat shapes post (possibly) broadcasted
                for (int i = 0; i < indices.Length; i++)
                {
                    var nd = indices[i];
                    if (nd.ndim != 1)
                        indices[i] = nd = nd.flat;
                }
            }

            //by now all indices are flat, relative indices, might be subshaped, might be non-linear ---------------
            //we flatten to linear absolute points -----------------------------------------------------------------
            var computedOffsets = new NDArray<int>(Shape.Vector(indicesSize), false);
            var computedAddr = computedOffsets.Address;

            //prepare indices getters
            var indexGetters = PrepareIndexGetters(srcShape, indices);

            //figure out the largest possible abosulte offset
            int largestOffset;
            if (srcShape.IsContiguous)
                largestOffset = source.size;
            else
            {
                var largestIndices = (int[])source.shape.Clone();
                for (int i = 0; i < largestIndices.Length; i++)
                    largestIndices[i] = largestIndices[i] - 1;

                largestOffset = srcShape.GetOffset(largestIndices);
            }

            //compute coordinates
            if (indices.Length > 1)
            {
                Parallel.For(0, indicesSize, i =>
                {
                    var index = stackalloc int[ndsCount];

                    for (int ndIdx = 0; ndIdx < ndsCount; ndIdx++) //todo optimize this loop with unmanaged address.
                        index[ndIdx] = indexGetters[ndIdx](i); //replace with memory access or iterators

                    if ((computedAddr[i] = srcShape.GetOffset(index, ndsCount)) > largestOffset)
                        throw new IndexOutOfRangeException($"Index [{string.Join(", ", new Span<int>(index, ndsCount).ToArray())}] exceeds given NDArray's bounds. NDArray is shaped {srcShape}.");
                });
            }
            else
            {
                Func<int, int> srcOffset = srcShape.GetOffset_1D;
                var getter = indexGetters[0];
                Parallel.For(0, indicesSize, i =>
                {
                    if ((computedAddr[i] = srcOffset(getter(i))) > largestOffset)
                        throw new IndexOutOfRangeException($"Index [{getter(i)}] exceeds given NDArray's bounds. NDArray is shaped {srcShape}.");
                });
            }

            //based on recently made `computedOffsets` we retreive data -----------------------------------------

            if (!isSubshaped)
            {
                var idxAddr = computedOffsets.Address;
                var srcAddr = source.Address;
                var dst = new NDArray<T>(Shape.Vector(computedOffsets.size), false);
                T* dstAddr = dst.Address;
                //indices point to a scalar
                Parallel.For(0, dst.size, i => *(dstAddr + i) = *(srcAddr + *(idxAddr + i))); //TODO linear might be faster. bench it.

                if (retShape != null)
                    return dst.reshape_unsafe(retShape);

                return dst;
            }
            else
            {
                //indices point to an ndarray
                if (source.Shape.IsContiguous)
                    return retriever_indices_nd(source, computedOffsets, ndsCount, retShape: retShape, subShape: subShape);
                return retriever_indices_nd_nonlinear(source, indices, ndsCount, retShape: retShape, subShape: subShape);
            }
        }

        /// <summary>
        ///     Accepts collapsed 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="offsets"></param>
        /// <param name="retShape"></param>
        /// <param name="absolute">Is the given <paramref name="offsets"/> already point to the offset of <paramref name="src"/>.</param>
        /// <returns></returns>
        private static unsafe NDArray<T> retriever_indices_nd<T>(NDArray<T> src, NDArray<int> offsets, int ndsCount, int[] retShape, int[] subShape) where T : unmanaged
        {
            //facts:
            //indices are always offsetted to 
            Debug.Assert(offsets.ndim == 1);
            Debug.Assert(retShape != null);

            //handle pointers pointing to subshape
            var subShapeSize = 1;
            for (int i = 0; i < subShape.Length; i++)
                subShapeSize *= subShape[i];

            int* offsetAddr = offsets.Address;
            var offsetsSize = offsets.size;
            T* srcAddr = src.Address;

            var dst = new NDArray<T>(retShape, false);
            T* dstAddr = dst.Address;
            int copySize = subShapeSize * InfoOf<T>.Size;

            for (int i = 0; i < offsetsSize; i++) //TODO: Parallel for
            {
                Buffer.MemoryCopy(srcAddr + *(offsetAddr + i), dstAddr + i * subShapeSize, copySize, copySize);
            }

            return dst;
        }

        /// <summary>
        ///     Accepts collapsed 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="offsets"></param>
        /// <param name="retShape"></param>
        /// <param name="absolute">Is the given <paramref name="offsets"/> already point to the offset of <paramref name="source"/>.</param>
        /// <returns></returns>
        [SuppressMessage("ReSharper", "SuggestVarOrType_Elsewhere")]
        private static unsafe NDArray<T> retriever_indices_nd_nonlinear<T>(NDArray<T> source, NDArray[] indices, int ndsCount, int[] retShape, int[] subShape) where T : unmanaged
        {
            //facts:
            //indices are always offsetted to 
            //handle pointers pointing to subshape
            var subShapeNDim = subShape.Length;

            var size = indices[0].size; //first is ok because they are broadcasted t oeac
            T* srcAddr = source.Address;

            var dst = new NDArray(InfoOf<T>.NPTypeCode, retShape, false);
            T* dstAddr = (T*)dst.Address;

            var srcDims = indices.Length;
            var indexGetters = PrepareIndexGetters(source.Shape, indices);
            //compute coordinates
            Parallel.For(0, size, i => //TODO: make parallel.for
            {
                int* index = stackalloc int[srcDims];

                //load indices
                //index[0] = i;
                for (int k = 0; k < srcDims; k++)
                    index[k] = indexGetters[k](i); //replace with memory access or iterators
#if DEBUG
                var from = source[index, srcDims];
                var to = dst[i];

                //assign
                dst[i] = from;
#else
                dst[i] = source[index, srcDims];
#endif
            });

            return dst.MakeGeneric<T>();
        }

        private static unsafe Func<int, int>[] PrepareIndexGetters(Shape srcShape, NDArray[] indices)
        {
            var indexGetters = new Func<int, int>[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                var idxs = indices[i];
                var dimensionSize = srcShape[i];
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

            return indexGetters;
        }
    }
}
