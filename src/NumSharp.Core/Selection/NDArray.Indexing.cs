using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
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
        ///     Used to perform selection based on a selection indices.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.17.0/user/basics.indexing.html</remarks>
        /// <exception cref="IndexOutOfRangeException">When one of the indices exceeds limits.</exception>
        /// <exception cref="ArgumentException">indices must be of Int type (byte, u/short, u/int, u/long).</exception>
        public NDArray this[params NDArray<int>[] selection]
        {
            get => retrieve_indices(this, selection.Select(array => (NDArray)array).ToArray(), null);
            set
            {
                throw new NotImplementedException("Setter is not implemnted yet");
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
            return retrieve_indices(this, indices, @out);
        }

        /// <summary>
        /// Perform slicing, index extraction, masking and indexing all at the same time with mixed index objects
        /// </summary>
        /// <param name="indicesObjects"></param>
        /// <returns></returns>
        public NDArray this[params object[] indicesObjects]
        {
            get
            {
                return retrieve_indices(indicesObjects);
            }
            set
            {
                var indicesLen = indicesObjects.Length;
                if (indicesLen == 1 && indicesObjects[0] is int[] coordinates)
                {
                    Storage.GetData(coordinates).SetData(value);
                    return;
                }

                int ints = 0;
                int bools = 0;
                for (int i = 0; i < indicesLen; i++)
                {
                    var o = indicesObjects[i];
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
                    Storage.SetData(value, indicesObjects.Cast<int>().ToArray());
                    return;
                }

                if (bools == indicesLen)
                {
                    if (indicesLen != 1) ;
                    //TODO: setter version of return this[np.array(indices_or_slices.Cast<bool>()).MakeGeneric<bool>()];

                    var @bool = (bool)indicesObjects[0];

                    if (!@bool)
                        return;

                    np.expand_dims(this, 0).SetData(value); //equivalent to [np.newaxis]
                    return;
                }

                _slice:
                if (indicesLen == 1 && indicesObjects[0] is string slicesStr)
                {
                    Storage.GetView(Slice.ParseSlices(slicesStr)).SetData(value);
                    return;
                }

                var slices = indicesObjects.Select(x =>
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

        private NDArray retrieve_indices(object[] indicesObjects)
        {
            var indicesLen = indicesObjects.Length;
            if (indicesLen == 1) 
            {
                switch (indicesObjects[0])
                {
                    case NDArray nd:
                        return retrieve_indices(this, new NDArray[] { nd }, null);
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
                            indicesObjects = ExpandEllipsis(indicesObjects).ToArray();
                            //TODO: i think we need to set here indicesLen = indicesObjects.Length
                            continue;
                        }

                        if (o.IsNewAxis)
                        {
                            //TODO:
                            //TODO: currently we just swallow them.
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
                        } else
                            return new NDArray<int>(); //false bool causes nullification of return.
                        continue;
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
                    } else
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
            var ret = retrieve_indices(@this, indicesArray, null);

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

        private IEnumerable<object> ExpandEllipsis(object[] ndarrays)
        {
            // count dimensions without counting ellipsis or newaxis
            var count = ndarrays.OfType<Slice>().Count(slice=>!(slice.IsNewAxis || slice.IsEllipsis));

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
        internal static NDArray<int> GetIndicesFromSlice(Shape shape, Slice slice, int axis)
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
        internal static NDArray<int> GetIndicesFromSlice(int[] shape, Slice slice, int axis)
        {
            var dim = shape[axis];
            var slice_def = slice.ToSliceDef(dim); // this resolves negative slice indices
            return np.arange(slice_def.Start, slice_def.Start + slice_def.Step * slice_def.Count, slice.Step).MakeGeneric<int>();
        }

        /// <summary>
        ///     Slice the array with Python slice notation like this: ":, 2:7:1, ..., np.newaxis"
        /// </summary>
        /// <param name="slice">A string containing slice notations for every dimension, delimited by comma</param>
        /// <returns>A sliced view</returns>
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
