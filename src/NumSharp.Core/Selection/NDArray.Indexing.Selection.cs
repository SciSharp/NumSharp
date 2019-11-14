using System;
using System.Linq;
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
        private static unsafe NDArray retrieve_indices(NDArray src, NDArray[] indices, NDArray @out)
        {
#if _REGEN
            #region Compute
		    switch (src.typecode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: return retrieve_indices<#2>(src.MakeGeneric<#2>(), indices, @out);
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute
		    switch (src.typecode)
		    {
			    case NPTypeCode.Boolean: return retrieve_indices<bool>(src.MakeGeneric<bool>(), indices, @out);
			    case NPTypeCode.Byte: return retrieve_indices<byte>(src.MakeGeneric<byte>(), indices, @out);
			    case NPTypeCode.Int16: return retrieve_indices<short>(src.MakeGeneric<short>(), indices, @out);
			    case NPTypeCode.UInt16: return retrieve_indices<ushort>(src.MakeGeneric<ushort>(), indices, @out);
			    case NPTypeCode.Int32: return retrieve_indices<int>(src.MakeGeneric<int>(), indices, @out);
			    case NPTypeCode.UInt32: return retrieve_indices<uint>(src.MakeGeneric<uint>(), indices, @out);
			    case NPTypeCode.Int64: return retrieve_indices<long>(src.MakeGeneric<long>(), indices, @out);
			    case NPTypeCode.UInt64: return retrieve_indices<ulong>(src.MakeGeneric<ulong>(), indices, @out);
			    case NPTypeCode.Char: return retrieve_indices<char>(src.MakeGeneric<char>(), indices, @out);
			    case NPTypeCode.Double: return retrieve_indices<double>(src.MakeGeneric<double>(), indices, @out);
			    case NPTypeCode.Single: return retrieve_indices<float>(src.MakeGeneric<float>(), indices, @out);
			    case NPTypeCode.Decimal: return retrieve_indices<decimal>(src.MakeGeneric<decimal>(), indices, @out);
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif
        }

        //TODO: this
        private static unsafe NDArray set_indices(NDArray src, NDArray[] indices, NDArray dst)
        {
#if _REGEN
            #region Compute
		    switch (src.typecode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: return retrieve_indices<#2>(src.MakeGeneric<#2>(), indices, dst);
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute
            switch (src.typecode)
		    {
			    case NPTypeCode.Boolean: return retrieve_indices<bool>(src.MakeGeneric<bool>(), indices, dst);
			    case NPTypeCode.Byte: return retrieve_indices<byte>(src.MakeGeneric<byte>(), indices, dst);
			    case NPTypeCode.Int16: return retrieve_indices<short>(src.MakeGeneric<short>(), indices, dst);
			    case NPTypeCode.UInt16: return retrieve_indices<ushort>(src.MakeGeneric<ushort>(), indices, dst);
			    case NPTypeCode.Int32: return retrieve_indices<int>(src.MakeGeneric<int>(), indices, dst);
			    case NPTypeCode.UInt32: return retrieve_indices<uint>(src.MakeGeneric<uint>(), indices, dst);
			    case NPTypeCode.Int64: return retrieve_indices<long>(src.MakeGeneric<long>(), indices, dst);
			    case NPTypeCode.UInt64: return retrieve_indices<ulong>(src.MakeGeneric<ulong>(), indices, dst);
			    case NPTypeCode.Char: return retrieve_indices<char>(src.MakeGeneric<char>(), indices, dst);
			    case NPTypeCode.Double: return retrieve_indices<double>(src.MakeGeneric<double>(), indices, dst);
			    case NPTypeCode.Single: return retrieve_indices<float>(src.MakeGeneric<float>(), indices, dst);
			    case NPTypeCode.Decimal: return retrieve_indices<decimal>(src.MakeGeneric<decimal>(), indices, dst);
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif
        }

        private static unsafe NDArray<T> retrieve_indices<T>(NDArray<T> source, NDArray[] indices, NDArray @out) where T : unmanaged
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
            NDArray idxs;
            int[] indicesImpliedShape = null;
            //preprocess indices -----------------------------------------------------------------------------------------------
            //handle non-flat indices and detect if broadcasting required
            if (indices.Length == 1)
            {
                //fast-lane for 1-d.
                idxs = indices[0];
                //TODO what does this check even test if (nd.shape[0] > source.shape[0])
                //TODO what does this check even test     throw new ArgumentOutOfRangeException($"index {nd.size - 1} is out of bounds for axis 0 with size {nd.shape[0]}");

                if (idxs.Shape.IsEmpty)
                    return new NDArray<T>();

                //handle non-flat index
                if (idxs.ndim != 1)
                {
                    indicesImpliedShape = idxs.shape;
                    idxs = idxs.flat;
                }

                //handle non-int32 index
                if (idxs.typecode != NPTypeCode.Int32)
                    idxs = idxs.astype(NPTypeCode.Int32, true);

                indices[0] = idxs;
            }
            else
            {
                idxs = indices[0];
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
                               
                //handle non-flat shapes post (possibly) broadcasted
                for (int i = 0; i < indices.Length; i++)
                {
                    var nd = indices[i];
                    if (nd.ndim != 1) {
                        indicesImpliedShape = nd.shape;
                        indices[i] = nd = nd.flat;
                    }
                }
            }

            //resolve retShape
            if (!isSubshaped)
            {
                retShape = indicesImpliedShape ?? (int[])idxs.shape.Clone();
            }
            else
            {
                if (indicesImpliedShape == null)
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
                } else
                {

                    retShape = indicesImpliedShape;

                    subShape = new int[srcShape.NDim - ndsCount];
                    for (int src_i = ndsCount, i = 0; src_i < srcShape.NDim; src_i++, i++)
                    {
                        subShape[i] = srcShape[src_i];
                    }

                    if (isSubshaped)
                        retShape = Arrays.Concat(indicesImpliedShape, subShape);
                }
            }

            //when -----------------------------------------
            //indices point to an ndarray
            if (isSubshaped && (!source.Shape.IsContiguous || (!(@out is null) && !@out.Shape.IsContiguous)))
                return retriever_indices_nd_nonlinear(source, indices, ndsCount, retShape: retShape, subShape: subShape, @out);

            //by now all indices are flat, relative indices, might be subshaped, might be non-linear ---------------
            //we flatten to linear absolute points -----------------------------------------------------------------
            var computedOffsets = new NDArray<int>(Shape.Vector(indicesSize), false);
            var computedAddr = computedOffsets.Address;

            //prepare indices getters
            var indexGetters = PrepareIndexGetters(srcShape, indices);

            //figure out the largest possible abosulte offset
            int largestOffset;
            if (srcShape.IsContiguous)
                largestOffset = source.size - 1;
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
                    return dst.reshape(retShape);

                return dst;
            }
            else
            {
                //non linear is handled before calculating computedOffsets
                return retriever_indices_nd(source, computedOffsets, indices, ndsCount, retShape: retShape, subShape: subShape, @out);
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
        private static unsafe NDArray<T> retriever_indices_nd<T>(NDArray<T> src, NDArray<int> offsets, NDArray[] indices, int ndsCount, int[] retShape, int[] subShape, NDArray @out) where T : unmanaged
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

            NDArray dst;
            if (@out is null)
                dst = new NDArray<T>(retShape, false);
            else
            {
                //compare computed retShape vs given @out
                if (!retShape.SequenceEqual(@out.shape))
                    throw new ArgumentException($"Given @out NDArray is expected to be shaped [{string.Join(", ", retShape)}] but is instead [{string.Join(", ", @out.shape)}]");
                if (@out.dtype != typeof(T))
                    throw new ArgumentException($"Given @out NDArray is expected to be dtype '{typeof(T).Name}' but is instead '{@out.dtype.Name}'");

                dst = @out;
            }

            T* dstAddr = (T*)dst.Address;
            int copySize = subShapeSize * InfoOf<T>.Size;

            Parallel.For(0, offsetsSize, i =>       
                Buffer.MemoryCopy(srcAddr + *(offsetAddr + i), dstAddr + i * subShapeSize, copySize, copySize));

            return dst.MakeGeneric<T>();
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
        private static unsafe NDArray<T> retriever_indices_nd_nonlinear<T>(NDArray<T> source, NDArray[] indices, int ndsCount, int[] retShape, int[] subShape, NDArray @out) where T : unmanaged
        {
            //facts:
            //indices are always offsetted to 
            //handle pointers pointing to subshape
            var subShapeNDim = subShape.Length;

            var size = indices[0].size; //first is ok because they are broadcasted t oeac
            T* srcAddr = source.Address;

            NDArray dst;
            if (@out is null)
                dst = new NDArray<T>(retShape, false);
            else
            {
                //compare computed retShape vs given @out
                if (!retShape.SequenceEqual(@out.shape))
                    throw new ArgumentException($"Given @out NDArray is expected to be shaped [{string.Join(", ", retShape)}] but is instead [{string.Join(", ", @out.shape)}]");
                if (@out.dtype != typeof(T))
                    throw new ArgumentException($"Given @out NDArray is expected to be dtype '{typeof(T).Name}' but is instead '{@out.dtype.Name}'");

                dst = @out;
            }

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
