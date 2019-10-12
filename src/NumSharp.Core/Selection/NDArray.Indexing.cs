using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Used to perform selection based on a boolean mask.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.17.0/user/basics.indexing.html</remarks>
        /// <exception cref="IndexOutOfRangeException">When one of the indices exceeds limits.</exception>
        /// <exception cref="ArgumentException">indices must be of Int type (byte, u/short, u/int, u/long).</exception>
        public NDArray this[NDArray<bool> mask]
        {
            get => _extract_indices(new[] {np.nonzero(mask)}, false, null);
            set
            {
                throw new NotImplementedException("Setter is not implemnted yet");
            }
        }

        /// <summary>
        ///     Used to perform selection based on a selection indices.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.17.0/user/basics.indexing.html</remarks>
        /// <exception cref="IndexOutOfRangeException">When one of the indices exceeds limits.</exception>
        /// <exception cref="ArgumentException">indices must be of Int type (byte, u/short, u/int, u/long).</exception>
        public NDArray this[params NDArray<int>[] selection]
        {
            get => _extract_indices(selection.Select(array => (NDArray)array).ToArray(), false, null);
            set
            {
                throw new NotImplementedException("Setter is not implemnted yet");
            }
        }

        /// <summary>
        ///     Used to perform selection based on given indices.
        /// </summary>
        /// <param name="dims">The pointer to the dimensions</param>
        /// <param name="ndims">The count of ints in <paramref name="dims"/></param>
        public unsafe NDArray this[int* dims, int ndims]
        {
            get => new NDArray(Storage.GetData(dims, ndims));
            set => Storage.GetData(dims, ndims).SetData(value);
        }

        /// <summary>
        /// Perform slicing, index extraction, masking and indexing all at the same time with mixed index objects
        /// </summary>
        /// <param name="indices_or_slices"></param>
        /// <returns></returns>
        public NDArray this[params object[] indices_or_slices]
        {
            get
            {
                var indicesLen = indices_or_slices.Length;
                if (indicesLen == 1 && indices_or_slices[0] is int[] coordinates)
                {
                    return GetData(coordinates);
                }

                int ints = 0;
                int bools = 0;
                foreach (var o in indices_or_slices)
                {
                    switch (o)
                    {
                        case NDArray _:
                            goto _foundNDArray;
                        case int _:
                            ints++;
                            break;
                        case bool @bool:
                            bools++;
                            break;
                        case string _:
                        case Slice _:
                            goto _slice;

                        default: throw new ArgumentException($"Unsupported indexing type: '{(o?.GetType()?.Name ?? "null")}'");
                    }
                }

                if (ints == indicesLen)
                    return new NDArray(Storage.GetData(indices_or_slices.Cast<int>().ToArray()));

                if (bools == indicesLen)
                {
                    if (indicesLen != 1)
                        return this[np.array(indices_or_slices.Cast<bool>()).MakeGeneric<bool>()];

                    var @bool = (bool)indices_or_slices[0];

                    if (!@bool)
                        return new NDArray(NPTypeCode.Int32); //return empty

                    return np.expand_dims(this, 0); //equivalent to [np.newaxis]
                }

                _slice:
                if (indicesLen == 1 && indices_or_slices[0] is string slicesStr)
                    return new NDArray(Storage.GetView(Slice.ParseSlices(slicesStr)));

                var slices = indices_or_slices.Select(x =>
                {
                    switch (x)
                    {
                        case Slice o: return o;
                        case int o: return Slice.Index(o);
                        case string o: return new Slice(o);
                        default: throw new ArgumentException($"Unsupported slice type: '{(x?.GetType()?.Name ?? "null")}'");
                    }
                }).ToArray();

                return new NDArray(Storage.GetView(slices));

                _foundNDArray:
                return _extract_indices(indices_or_slices.Select(nd => (NDArray)nd).ToArray(), false, null);
            }
            set
            {
                var indicesLen = indices_or_slices.Length;
                if (indicesLen  == 1 && indices_or_slices[0] is int[] coordinates)
                {
                    Storage.GetData(coordinates).SetData(value);
                    return;
                }

                int ints = 0;
                int bools = 0;
                for (int i = 0; i < indicesLen; i++)
                {
                    var o = indices_or_slices[i];
                    switch (o)
                    {
                        case NDArray _:
                            goto _foundNDArray;
                        case int _:
                            ints++;
                            break;
                        case string _:
                        case Slice _:
                            goto _slice;
                        case bool @bool:
                            bools++;
                            break;
                        default: throw new ArgumentException($"Unsupported indexing type: '{(o?.GetType()?.Name ?? "null")}'");
                    }
                }

                if (ints == indicesLen)
                {
                    Storage.SetData(value, indices_or_slices.Cast<int>().ToArray());
                    return;
                }

                if (bools == indicesLen)
                {
                    if (indicesLen != 1) ;
                    //TODO: setter version of return this[np.array(indices_or_slices.Cast<bool>()).MakeGeneric<bool>()];

                    var @bool = (bool)indices_or_slices[0];

                    if (!@bool)
                        return;

                    np.expand_dims(this, 0).SetData(value); //equivalent to [np.newaxis]
                    return;
                }

                _slice:
                if (indicesLen == 1 && indices_or_slices[0] is string slicesStr)
                {
                    Storage.GetView(Slice.ParseSlices(slicesStr)).SetData(value);
                    return;
                }

                var slices = indices_or_slices.Select(x =>
                {
                    switch (x)
                    {
                        case Slice o: return o;
                        case int o: return Slice.Index(o);
                        case string o: return new Slice(o);
                        default: throw new ArgumentException($"Unsupported slice type: '{(x?.GetType()?.Name ?? "null")}'");
                    }
                }).ToArray();

                Storage.GetView(slices).SetData(value);
                return;
                _foundNDArray:
                throw new NotSupportedException();
                //TODO: setter version of return retrieve_indices(this, indices_or_slices.Select(nd => (NDArray)nd).ToArray());
                ;
            }
        }

        /// <summary>
        ///     Used to perform selection based on indices, equivalent to nd[NDArray[]].
        /// </summary>
        /// <param name="@out">Alternative output array in which to place the result. It must have the same shape as the expected output and be of dtype <see cref="Int32"/>.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.17.0/user/basics.indexing.html</remarks>
        /// <exception cref="IndexOutOfRangeException">When one of the indices exceeds limits.</exception>
        /// <exception cref="ArgumentException">indices must be of Int type (byte, u/short, u/int, u/long).</exception>
        public NDArray GetIndices(NDArray @out, params NDArray[] indices)
        {
            return _extract_indices(indices, false, @out);
        }

        private NDArray _extract_indices(NDArray[] mindices, bool isCollapsed, NDArray @out)
        {
            if (mindices == null)
                throw new ArgumentNullException(nameof(mindices));

            //when mindices is single and 
            if (mindices.Length == 1 && (mindices[0].ndim >= this.ndim || isCollapsed))
            {
                //element-wise if ndims are equal or src is flat
                return _index_elemntwise(mindices[0], @out);

                //otherwise returns like GetNDArrays() but specific indexes so we falldown
            }

            if (mindices.Length > ndim)
                throw new ArgumentException($"There are more mindices ({mindices.Length}) than dimensions ({ndim}) in current array.");

            //handle broadcasting if required
            var indicesShape = mindices[0].Shape;
            for (int i = 1; i < mindices.Length; i++)
            {
                if (indicesShape != mindices[i].Shape)
                {
                    mindices = DefaultEngine.Broadcast(mindices);
                    break;
                }
            }

            indicesShape = mindices[0].Shape.Clean(); //we clean it later incase we broadcast mindices[0] in the code above

            //invalidate @out
            if (!(@out is null))
            {
                if (indicesShape != @out.Shape)
                    throw new IncorrectShapeException($"shapes {indicesShape} is not compatible with given @out array's shape {@out.Shape} for indexing nd[NDArray].");

                if (@out.typecode != NPTypeCode.Int32)
                    throw new IncorrectTypeException("Unable to index nd[NDArray] when the @out specified dtype is not int32.");
            }

            //handle when mindicies[i] is not int array.
            for (int i = 0; i < mindices.Length; i++)
            {
                if (mindices[i].typecode != NPTypeCode.Int32)
                    mindices[i] = new NDArray(mindices[i].Storage.Cast<int>());
            }

            //case 1 multidim eq to ndim, meaning every index points to a scalar array
            if (mindices.Length == ndim)
            {
                //this is multidims, collapse them into singledim
                var collapsed = new NDArray(NPTypeCode.Int32, indicesShape, false);
                var iter = new NDCoordinatesIncrementor(ref indicesShape);
                var ndims = mindices.Length;
                var collapsedIndex = new int[ndims];
                var individualIndex = iter.Index;
                Func<int[], int> getOffset = Shape.GetOffset;
                do
                {
                    for (int i = 0; i < ndims; i++)
                    {
                        collapsedIndex[i] = (int)mindices[i].GetInt32(individualIndex);
                    }

                    collapsed[individualIndex] = getOffset(collapsedIndex);
                } while (iter.Next() != null);

                return _extract_indices(new NDArray[] {collapsed}, true, @out);
            }

            //case 2 return is not scalar collection but axis iteration
            {
                var flat_mindices = new NDArray[mindices.Length];
                for (var i = 0; i < mindices.Length; i++) flat_mindices[i] = mindices[i];

                NDArray ret = @out;
                if (ret is null)
                {
                    var (retShape, _) = Shape.GetSubshape(new int[indicesShape.dimensions.Length]);
                    var dims = indicesShape.dimensions.Concat(retShape.dimensions).ToArray();
                    ret = new NDArray(typecode, dims, false); //retshape is already clean
                }

                var iter = new NDCoordinatesIncrementor(ref indicesShape);
                var ndims = mindices.Length;
                var collapsedIndex = new int[ndims];
                var individualIndex = iter.Index;
                //TODO when mindicies.length == 1 we can really optimize it
                do
                {
                    for (int i = 0; i < ndims; i++)
                        collapsedIndex[i] = (int)mindices[i].GetInt32(individualIndex);

                    ret[individualIndex] = this[collapsedIndex];
                } while (iter.Next() != null);

                return ret;
            }
        }

        public NDArray this[string slice]
        {
            get => new NDArray(Storage.GetView(Slice.ParseSlices(slice)));
            set => Storage.GetView(Slice.ParseSlices(slice)).SetData(value);
        }

        //TODO! masking with boolean array
        // example:
        //
        // a = np.arange(12);
        // b=a[a > 6];
        //
        //public NDArray this[NDArray<bool> booleanArray]
        //{
        //    get
        //    {
        //        if (!Enumerable.SequenceEqual(shape, booleanArray.shape))
        //        {
        //            throw new IncorrectShapeException();
        //        }

        //        var boolDotNetArray = booleanArray.Data<bool>();

        //        switch (dtype.Name)
        //        {
        //            case "Int32":
        //                {
        //                    var nd = new List<int>();

        //                    for (int idx = 0; idx < boolDotNetArray.Length; idx++)
        //                    {
        //                        if (boolDotNetArray[idx])
        //                        {
        //                            nd.Add(Data<int>(booleanArray.Storage.Shape.GetDimIndexOutShape(idx)));
        //                        }
        //                    }

        //                    return new NDArray(nd.ToArray(), nd.Count);
        //                }
        //            case "Double":
        //                {
        //                    var nd = new List<double>();

        //                    for (int idx = 0; idx < boolDotNetArray.Length; idx++)
        //                    {
        //                        if (boolDotNetArray[idx])
        //                        {
        //                            nd.Add(Data<double>(booleanArray.Storage.Shape.GetDimIndexOutShape(idx)));
        //                        }
        //                    }

        //                    return new NDArray(nd.ToArray(), nd.Count);
        //                }
        //        }

        //        throw new NotImplementedException("");

        //    }
        //    set
        //    {
        //        if (!Enumerable.SequenceEqual(shape, booleanArray.shape))
        //        {
        //            throw new IncorrectShapeException();
        //        }

        //        object scalarObj = value.Storage.GetData().GetValue(0);

        //        bool[] boolDotNetArray = booleanArray.Storage.GetData() as bool[];

        //        int elementsAmount = booleanArray.size;

        //        for (int idx = 0; idx < elementsAmount; idx++)
        //        {
        //            if (boolDotNetArray[idx])
        //            {
        //                int[] indexes = booleanArray.Storage.Shape.GetDimIndexOutShape(idx);
        //                Array.SetValue(scalarObj, Storage.Shape.GetOffset(slice, indexes));
        //            }
        //        }

        //    }
        //}

        [MethodImpl((MethodImplOptions)512)]
        private NDArray _index_elemntwise(NDArray indices, NDArray @out = null)
        {
            //verify the indices dtype is Int.
            var grp = indices.typecode.GetGroup();
            if (grp != 1 && grp != 2 && indices.typecode != NPTypeCode.Byte)
                throw new ArgumentException("indices must be of Int type.", nameof(indices));

            //invalidate @out
            if (!(@out is null))
            {
                if (indices.Shape != @out.Shape) throw new IncorrectShapeException($"shapes {indices.Shape} is not compatible with given @out array's shape {@out.Shape} for indexing element-wise nd[NDArray].");

                if (@out.typecode != NPTypeCode.Int32)
                    throw new IncorrectTypeException("Unable to index element-wise nd[NDArray] when the @out specified dtype is not int32.");
            }

            var ret = @out ?? new NDArray(dtype, indices.Shape.Clean(), false);

            // ReSharper disable once LocalVariableHidesMember
            var size = this.size;
            var src = this.flat;

            if (this.ndim == 1)
            {
                var dst = ret.flat;
#if _REGEN
                #region Compute
		        switch (typecode)
		        {
			        %foreach supported_dtypes,supported_dtypes_lowercase%
			        case NPTypeCode.#1:
			        {
				        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = new int[1];
                        var indexset = new int[1];
                        while (hasNextIndex()) {
                            index[0] = nextIndex();
                            dst.Set#1(src.Get#1(index), indexset);
                            indexset[0]++;
                        }
                        break;
			        }
			        %
			        default:
				        throw new NotSupportedException();
		        }
                #endregion
#else

                #region Compute

                switch (typecode)
                {
                    case NPTypeCode.Boolean:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = new int[1];
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            index[0] = nextIndex();
                            dst.SetBoolean(src.GetBoolean(index), indexset);
                            indexset[0]++;
                        }

                        break;
                    }

                    case NPTypeCode.Byte:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = new int[1];
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            index[0] = nextIndex();
                            dst.SetByte(src.GetByte(index), indexset);
                            indexset[0]++;
                        }

                        break;
                    }

                    case NPTypeCode.Int16:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = new int[1];
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            index[0] = nextIndex();
                            dst.SetInt16(src.GetInt16(index), indexset);
                            indexset[0]++;
                        }

                        break;
                    }

                    case NPTypeCode.UInt16:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = new int[1];
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            index[0] = nextIndex();
                            dst.SetUInt16(src.GetUInt16(index), indexset);
                            indexset[0]++;
                        }

                        break;
                    }

                    case NPTypeCode.Int32:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = new int[1];
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            index[0] = nextIndex();
                            dst.SetInt32(src.GetInt32(index), indexset);
                            indexset[0]++;
                        }

                        break;
                    }

                    case NPTypeCode.UInt32:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = new int[1];
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            index[0] = nextIndex();
                            dst.SetUInt32(src.GetUInt32(index), indexset);
                            indexset[0]++;
                        }

                        break;
                    }

                    case NPTypeCode.Int64:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = new int[1];
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            index[0] = nextIndex();
                            dst.SetInt64(src.GetInt64(index), indexset);
                            indexset[0]++;
                        }

                        break;
                    }

                    case NPTypeCode.UInt64:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = new int[1];
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            index[0] = nextIndex();
                            dst.SetUInt64(src.GetUInt64(index), indexset);
                            indexset[0]++;
                        }

                        break;
                    }

                    case NPTypeCode.Char:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = new int[1];
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            index[0] = nextIndex();
                            dst.SetChar(src.GetChar(index), indexset);
                            indexset[0]++;
                        }

                        break;
                    }

                    case NPTypeCode.Double:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = new int[1];
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            index[0] = nextIndex();
                            dst.SetDouble(src.GetDouble(index), indexset);
                            indexset[0]++;
                        }

                        break;
                    }

                    case NPTypeCode.Single:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = new int[1];
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            index[0] = nextIndex();
                            dst.SetSingle(src.GetSingle(index), indexset);
                            indexset[0]++;
                        }

                        break;
                    }

                    case NPTypeCode.Decimal:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = new int[1];
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            index[0] = nextIndex();
                            dst.SetDecimal(src.GetDecimal(index), indexset);
                            indexset[0]++;
                        }

                        break;
                    }

                    default:
                        throw new NotSupportedException();
                }

                #endregion

#endif
            }
            else
            {
                var retIncr = new NDCoordinatesIncrementor(ret.shape);
#if _REGEN
                #region Compute
		        switch (typecode)
		        {
			        %foreach supported_dtypes,supported_dtypes_lowercase%
			        case NPTypeCode.#1:
			        {
				        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = incr.Index;
                        var indexset = new int[1];
                        while (hasNextIndex()) {
                            ret.Set#1(src.Get#1(nextIndex()), index);
                            indexset[0]++;
                            incr.Next();
                        }
                        break;
			        }
			        %
			        default:
				        throw new NotSupportedException();
		        }
                #endregion
#else

                #region Compute

                switch (typecode)
                {
                    case NPTypeCode.Boolean:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = retIncr.Index;
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            ret.SetBoolean(src.GetBoolean(nextIndex()), index);
                            indexset[0]++;
                            retIncr.Next();
                        }

                        break;
                    }

                    case NPTypeCode.Byte:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = retIncr.Index;
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            ret.SetByte(src.GetByte(nextIndex()), index);
                            indexset[0]++;
                            retIncr.Next();
                        }

                        break;
                    }

                    case NPTypeCode.Int16:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = retIncr.Index;
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            ret.SetInt16(src.GetInt16(nextIndex()), index);
                            indexset[0]++;
                            retIncr.Next();
                        }

                        break;
                    }

                    case NPTypeCode.UInt16:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = retIncr.Index;
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            ret.SetUInt16(src.GetUInt16(nextIndex()), index);
                            indexset[0]++;
                            retIncr.Next();
                        }

                        break;
                    }

                    case NPTypeCode.Int32:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = retIncr.Index;
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            ret.SetInt32(src.GetInt32(nextIndex()), index);
                            indexset[0]++;
                            retIncr.Next();
                        }

                        break;
                    }

                    case NPTypeCode.UInt32:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = retIncr.Index;
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            ret.SetUInt32(src.GetUInt32(nextIndex()), index);
                            indexset[0]++;
                            retIncr.Next();
                        }

                        break;
                    }

                    case NPTypeCode.Int64:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = retIncr.Index;
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            ret.SetInt64(src.GetInt64(nextIndex()), index);
                            indexset[0]++;
                            retIncr.Next();
                        }

                        break;
                    }

                    case NPTypeCode.UInt64:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = retIncr.Index;
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            ret.SetUInt64(src.GetUInt64(nextIndex()), index);
                            indexset[0]++;
                            retIncr.Next();
                        }

                        break;
                    }

                    case NPTypeCode.Char:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = retIncr.Index;
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            ret.SetChar(src.GetChar(nextIndex()), index);
                            indexset[0]++;
                            retIncr.Next();
                        }

                        break;
                    }

                    case NPTypeCode.Double:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = retIncr.Index;
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            ret.SetDouble(src.GetDouble(nextIndex()), index);
                            indexset[0]++;
                            retIncr.Next();
                        }

                        break;
                    }

                    case NPTypeCode.Single:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = retIncr.Index;
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            ret.SetSingle(src.GetSingle(nextIndex()), index);
                            indexset[0]++;
                            retIncr.Next();
                        }

                        break;
                    }

                    case NPTypeCode.Decimal:
                    {
                        var data = indices.AsIterator<int>(); //iterator handles cast internal if required
                        var hasNextIndex = data.HasNext;
                        var nextIndex = data.MoveNext;
                        var _nextIndexClosure = nextIndex;
                        //handle cases of negative index
                        nextIndex = () =>
                        {
                            var nextIndexValue = _nextIndexClosure();
                            if (nextIndexValue >= size)
                                throw new IndexOutOfRangeException($"index {nextIndexValue} out of bounds 0<=index<{size}");

                            return nextIndexValue < 0 ? size + nextIndexValue : nextIndexValue;
                        };

                        var index = retIncr.Index;
                        var indexset = new int[1];
                        while (hasNextIndex())
                        {
                            ret.SetDecimal(src.GetDecimal(nextIndex()), index);
                            indexset[0]++;
                            retIncr.Next();
                        }

                        break;
                    }

                    default:
                        throw new NotSupportedException();
                }

                #endregion

#endif
            }

            if (indices.ndim != ret.ndim)
                ret.Storage.SetShapeUnsafe(indices.Shape);

            return ret;
        }
    }
}
