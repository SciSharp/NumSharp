using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Threading.Tasks;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Used to perform selection based on indices, equivalent to nd[NDArray[]].
        /// </summary>
        /// <param name="@out">Alternative output array in which to place the result. It must have the same shape as the expected output and be of dtype <see cref="Int32"/>.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.17.0/user/basics.indexing.html</remarks>
        /// <exception cref="IndexOutOfRangeException">When one of the indices exceeds limits.</exception>
        /// <exception cref="ArgumentException">indices must be of Int type (byte, u/short, u/int, u/long).</exception>
        public NDArray GetIndices(NDArray @out, NDArray[] indices)
        {
            return FetchIndices(this, indices, @out, true);
        }

        private NDArray FetchIndices(object[] indicesObjects)
        {
            var indicesLen = indicesObjects.Length;
            if (indicesLen == 1)
            {
                switch (indicesObjects[0])
                {
                    case NDArray nd:
                        return FetchIndices(this, new NDArray[] {nd}, null, true);
                    case int i:
                        return new NDArray(Storage.GetData(i));
                    case bool boolean:
                        if (boolean == false)
                            return new NDArray(dtype); //return empty

                        return np.expand_dims(this, 0); //equivalent to [np.newaxis]

                    case int[] coords:
                        return GetData(coords);
                    case NDArray[] nds:
                        return this[nds];
                    case object[] objs:
                        return this[objs];
                    case string slicesStr:
                        return new NDArray(Storage.GetView(Slice.ParseSlices(slicesStr)));
                    case null: throw new ArgumentNullException($"The 1th dimension in given indices is null.");
                    //no default
                }
            }

            int ints = 0;
            int bools = 0;
            bool foundSlices = false;
            for (var i = 0; i < indicesObjects.Length; i++)
            {
                switch (indicesObjects[i])
                {
                    case NDArray _:
                    case int[] _:
                        goto _NDArrayFound;
                    case int _:
                        ints++;
                        continue;
                    case bool @bool:
                        bools++;
                        continue;
                    case string _:
                    case Slice _:
                        continue;
                    case null: throw new ArgumentNullException($"The {i}th dimension in given indices is null.");
                    default: throw new ArgumentException($"Unsupported indexing type: '{(indicesObjects[i]?.GetType()?.Name ?? "null")}'");
                }
            }

            //handle all ints
            if (ints == indicesLen)
                return new NDArray(Storage.GetData(indicesObjects.Cast<int>().ToArray()));

            //handle all booleans
            if (bools == indicesLen)
                return this[np.array(indicesObjects.Cast<bool>().ToArray(), false).MakeGeneric<bool>()];

            Slice[] slices;
            //handle regular slices
            try
            {
                slices = indicesObjects.Select(x =>
                {
                    switch (x)
                    {
                        case Slice o: return o;
                        case int o: return Slice.Index(o);
                        case string o: return new Slice(o);
                        case bool o: return o ? Slice.NewAxis : throw new NumSharpException("false bool detected"); //TODO: verify this
                        case IConvertible o: return Slice.Index((int)o.ToInt32(CultureInfo.InvariantCulture));
                        default: throw new ArgumentException($"Unsupported slice type: '{(x?.GetType()?.Name ?? "null")}'");
                    }
                }).ToArray();
            }
            catch (NumSharpException e) when (e.Message.Contains("false bool detected"))
            {
                //handle rare case of false bool
                return new NDArray(dtype);
            }

            return new NDArray(Storage.GetView(slices));

//handle complex ndarrays indexing
            _NDArrayFound:
            var @this = this;
            var indices = new List<NDArray>();
            bool foundNewAxis = false;
            int countNewAxes = 0;
            //handle ndarray indexing
            bool hasCustomExpandedSlice = false; //use for premature slicing detection
            for (int i = 0; i < indicesLen; i++)
            {
                var idx = indicesObjects[i];
                _recuse:
                switch (idx)
                {
                    case Slice o:

                        if (o.IsEllipsis)
                        {
                            indicesObjects = ExpandEllipsis(indicesObjects, @this.ndim).ToArray();
                            //TODO: i think we need to set here indicesLen = indicesObjects.Length
                            continue;
                        }

                        if (o.IsNewAxis)
                        {
                            countNewAxes++;
                            foundNewAxis = true;
                            continue;
                        }

                        hasCustomExpandedSlice = true;
                        indices.Add(GetIndicesFromSlice(@this.Shape.dimensions, o, i - countNewAxes));
                        continue;
                    case int o:
                        indices.Add(NDArray.Scalar<int>(o));
                        continue;
                    case string o:
                        indicesObjects[i] = idx = new Slice(o);

                        goto _recuse;
                    case bool o:
                        if (o)
                        {
                            indicesObjects[i] = idx = Slice.NewAxis;
                            goto _recuse;
                        }
                        else
                            return new NDArray<int>(); //false bool causes nullification of return.
                    case IConvertible o:
                        indices.Add(NDArray.Scalar<int>(o.ToInt32(CultureInfo.InvariantCulture)));
                        continue;
                    case int[] o:
                        indices.Add(np.array(o, copy: false)); //we dont copy, pinning will be freed automatically after we done indexing.
                        continue;
                    case NDArray nd:
                        if (nd.typecode == NPTypeCode.Boolean)
                        {
                            //TODO: mask only specific axis??? find a unit test to check it against.
                            throw new Exception("if (nd.typecode == NPTypeCode.Boolean)");
                        }

                        indices.Add(nd);
                        continue;
                    default: throw new ArgumentException($"Unsupported slice type: '{(idx?.GetType()?.Name ?? "null")}'");
                }
            }

            var indicesArray = indices.ToArray();

            //handle premature slicing when the shapes cant be broadcasted together
            if (hasCustomExpandedSlice && !np.are_broadcastable(indicesArray))
            {
                var ndim = indicesObjects.Length;
                var prematureSlices = new Slice[ndim];
                var dims = @this.shape;
                for (int i = 0; i < ndim; i++)
                {
                    if (indicesObjects[i] is Slice slice)
                    {
                        prematureSlices[i] = slice;
                        //todo: we might need this in the future indicesObjects[i] = Slice.All;
                    }
                    else
                    {
                        prematureSlices[i] = Slice.All;
                    }
                }

                @this = @this[prematureSlices];

                //updated premature axes
                dims = @this.shape;
                for (int i = 0; i < ndim; i++)
                {
                    if (prematureSlices[i] != Slice.All)
                    {
                        indicesArray[i] = GetIndicesFromSlice(dims, Slice.All, i);
                    }
                }
            }

            //TODO: we can use a slice as null indice instead of expanding it, then we use PrepareIndexGetters to actually simulate that.
            var ret = FetchIndices(@this, indicesArray, null, !(indicesObjects[0] is int));

            if (foundNewAxis)
            {
                var targettedAxis = indices.Count - 1;
                var axisOffset = this.ndim - targettedAxis;
                var retShape = ret.Shape;
                for (int i = 0; i < indicesLen; i++)
                {
                    if (!(indicesObjects[i] is Slice slc) || !slc.IsNewAxis)
                        continue;

                    var axis = Math.Max(0, Math.Min(i - axisOffset, ret.ndim));
                    retShape = retShape.ExpandDimension(axis);
                }

                ret = ret.reshape(retShape);
            }

            return ret;
        }

        protected static NDArray FetchIndices(NDArray src, NDArray[] indices, NDArray @out, bool extraDim)
        {
#if _REGEN1
            #region Compute
		    switch (src.typecode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: return FetchIndices<#2>(src.MakeGeneric<#2>(), indices, @out, extraDim);
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute
		    switch (src.typecode)
		    {
			    case NPTypeCode.Boolean: return FetchIndices(src.MakeGeneric<bool>(), indices, @out, extraDim);
			    case NPTypeCode.Byte: return FetchIndices(src.MakeGeneric<byte>(), indices, @out, extraDim);
                case NPTypeCode.Char: return FetchIndices(src.MakeGeneric<char>(), indices, @out, extraDim);
                case NPTypeCode.Int32: return FetchIndices(src.MakeGeneric<int>(), indices, @out, extraDim);
			    case NPTypeCode.Int64: return FetchIndices(src.MakeGeneric<long>(), indices, @out, extraDim);
			    case NPTypeCode.Single: return FetchIndices(src.MakeGeneric<float>(), indices, @out, extraDim);
			    case NPTypeCode.Double: return FetchIndices(src.MakeGeneric<double>(), indices, @out, extraDim);
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#endif
        }

        protected static unsafe NDArray<T> FetchIndices<T>(NDArray<T> source, NDArray[] indices, NDArray @out, bool extraDim) where T : unmanaged
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            if (indices == null)
                throw new ArgumentNullException(nameof(indices));

            if (indices.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(indices));

            if (source.Shape.IsScalar)
                source = source.reshape(1);

            if (source.Shape.IsBroadcasted)
                source = np.copy(source).MakeGeneric<T>();

            int[] retShape = null, subShape = null;

            int indicesSize = indices.Max(nd => nd.size);
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
                if (idxs.Shape.IsEmpty)
                    return new NDArray<T>();

                //handle non-flat index
                if (idxs.ndim != 1)
                {
                    indicesImpliedShape = idxs.shape;
                    if (indicesImpliedShape.Length == 0)
                        indicesImpliedShape = new int[] {1};

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
                    idxs = indices[0];
                }

                //handle non-flat shapes post (possibly) broadcasted
                for (int i = 0; i < indices.Length; i++)
                {
                    var nd = indices[i];
                    if (nd.ndim != 1)
                    {
                        indicesImpliedShape = nd.shape;
                        if (indicesImpliedShape.Length == 0)
                            indicesImpliedShape = new int[] {1};

                        indices[i] = nd = np.atleast_1d(nd).flat;
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
                }
                else
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
            {
                var ret = FetchIndicesNDNonLinear(source, indices, ndsCount, retShape: retShape, subShape: subShape, @out);
                return ret;
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
                var ret = FetchIndicesND(source, computedOffsets, indices, ndsCount, retShape: retShape, subShape: subShape, @out);

                return ret;
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
        protected static unsafe NDArray<T> FetchIndicesND<T>(NDArray<T> src, NDArray<int> offsets, NDArray[] indices, int ndsCount, int[] retShape, int[] subShape, NDArray @out) where T : unmanaged
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
        protected static unsafe NDArray<T> FetchIndicesNDNonLinear<T>(NDArray<T> source, NDArray[] indices, int ndsCount, int[] retShape, int[] subShape, NDArray @out) where T : unmanaged
        {
            //facts:
            //indices are always offsetted to 
            //handle pointers pointing to subshape
            var subShapeNDim = subShape.Length;

            var size = indices[0].size; //first is ok because they are broadcasted t oeac
            T* srcAddr = source.Address;

            NDArray ret;
            if (@out is null)
                ret = new NDArray<T>(subShape, false);
            else
            {
                //compare computed retShape vs given @out
                if (!retShape.SequenceEqual(@out.shape))
                    throw new ArgumentException($"Given @out NDArray is expected to be shaped [{string.Join(", ", retShape)}] but is instead [{string.Join(", ", @out.shape)}]");
                if (@out.dtype != typeof(T))
                    throw new ArgumentException($"Given @out NDArray is expected to be dtype '{typeof(T).Name}' but is instead '{@out.dtype.Name}'");

                ret = @out;
            }

            T* dstAddr = (T*)ret.Address;

            var srcDims = indices.Length;
            var indexGetters = PrepareIndexGetters(source.Shape, indices);

            //compute coordinates
            //for (int i = 0; i < size; i++)
            Parallel.For(0, size, i => //TODO: make parallel.for
            {
                int* index = stackalloc int[srcDims];

                //load indices
                //index[0] = i;
                for (int k = 0; k < srcDims; k++)
                    index[k] = indexGetters[k](i); //replace with memory access or iterators
#if DEBUG
                var from = source[index, srcDims];
                var to = ret[i];

                //assign
                ret[i] = from;
#else
                ret[i] = source[index, srcDims];
#endif
            });
            //}

            return ret.flat.reshape(retShape).MakeGeneric<T>();
        }
    }
}
