using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class ViewStorage : IStorage
    {
        protected readonly IStorage _data;
        protected Slice[] _slices;
        protected Shape _nonReducedShape;

        public ViewStorage(IStorage dataStorage, string slice_notation) : this(dataStorage, Slice.ParseSlices(slice_notation)) { }

        public ViewStorage(IStorage dataStorage, params Slice[] slices)
        {
            if (dataStorage == null)
                throw new ArgumentException("dataStorage must not be null");
            if (slices == null)
                throw new ArgumentException("slices must not be null");
            if (slices.Length == 0)
            {
                // if now slices are given the view returns all of all dimensions
                slices = new Slice[dataStorage.Shape.NDim];
                for (int dim = 0; dim < dataStorage.Shape.NDim; dim++)
                    slices[dim] = Slice.All();
            }

            _data = (IStorage) dataStorage; //all types must inheriet IInternalStorage
            _slices = slices;
            Engine = dataStorage.Engine;
            EnsureValidSlicingDefinitions();
        }

        public bool SupportsSpan => false;

        public Type DType => _data.DType;

        /// <summary>
        ///     The engine that was used to create this <see cref="IStorage"/>.
        /// </summary>
        public ITensorEngine Engine { get; } //initialized in constructors (from Storage.TensorEngine).

        /// <summary>
        ///     The <see cref="NPTypeCode"/> of <see cref="IStorage.DType"/>.
        /// </summary>
        public NPTypeCode TypeCode => _data.TypeCode;

        /// <summary>
        ///     The size in bytes of a single value of <see cref="IStorage.DType"/>
        /// </summary>
        /// <remarks>Computed by <see cref="Marshal.SizeOf(object)"/></remarks>
        public int DTypeSize => _data.DTypeSize;

        /// <summary>
        /// storage shape for outside representation
        /// </summary>
        /// <value>Numpy's equivalent to np.shape</value>
        public Shape Shape { get; protected set; }

        /// <summary>
        ///     The current slice this <see cref="IStorage"/> instance currently represent.
        /// </summary>
        public Slice Slice { get; set; }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Array GetData() => CloneData();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T[] GetData<T>() => CloneData<T>();

        /// <summary>
        ///     Attempts to cast internal storage to an array of type <typeparamref name="T"/> and returns the result.
        /// </summary>
        /// <typeparam name="T">The type </typeparam>
        public T[] AsArray<T>() => _data.AsArray<T>();

        public T GetData<T>(params int[] indices) => _data.GetData<T>(TransformIndices(indices, _slices));

        public Span<T> GetSpanData<T>(Slice slice, params int[] indices)
        {
            throw new NotImplementedException("span of a view is not supported as it typically is not contiguous memory");
        }

        /// <summary>
        /// Set 1 single value to internal storage and keep dtype
        /// </summary>
        /// <param name="value"></param>
        /// <param name="indexes"></param>
        public void SetData(object value, params int[] indexes)
        {
            _data.SetData(value, TransformIndices(indexes, _slices));
        }

        public void SetData<T>(T value, params int[] indexes)
        {
            _data.SetData(value, TransformIndices(indexes, _slices));
        }

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data storage and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <remarks>Does not copy values.</remarks>
        public void ReplaceData(NDArray values)
        {
            var sliceDims = new int[_slices.Length];
            var sliceStartIdxs = new int[_slices.Length];
            var sliceStopIdxs = new int[_slices.Length];
            var sliceSteps = new int[_slices.Length];
            for (var dimIdx = 0; dimIdx < _slices.Length; dimIdx++)
            {
                sliceDims[dimIdx] = Shape.Dimensions[dimIdx];
                sliceStartIdxs[dimIdx] = 0;
                sliceStopIdxs[dimIdx] = sliceDims[dimIdx] - 1;
                sliceSteps[dimIdx] = _slices[dimIdx].GetAbsStep();
            }

            var storageStartIdxs = TransformIndices(sliceStartIdxs, _slices);
            var storageStopIdxs = TransformIndices(sliceStopIdxs, _slices);
            var storageSteps = sliceSteps;
            var storageIndexes = storageStartIdxs;
            var tLength = values.len;
            for (var tIdx = 0; tIdx < tLength; tIdx++)
            {
                //todo! switch values type and do a fast replace.
                _data.SetData(values.GetValue(tIdx), storageIndexes);
                IncrementIndexes(storageIndexes, storageStartIdxs, storageStopIdxs, storageSteps);
            }
        }

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data source and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        public void ReplaceData(Array values)
        {
            var sliceDims = new int[_slices.Length];
            var sliceStartIdxs = new int[_slices.Length];
            var sliceStopIdxs = new int[_slices.Length];
            var sliceSteps = new int[_slices.Length];
            for (var dimIdx = 0; dimIdx < _slices.Length; dimIdx++)
            {
                sliceDims[dimIdx] = Shape.Dimensions[dimIdx];
                sliceStartIdxs[dimIdx] = 0;
                sliceStopIdxs[dimIdx] = sliceDims[dimIdx] - 1;
                sliceSteps[dimIdx] = _slices[dimIdx].GetAbsStep();
            }

            var storageStartIdxs = TransformIndices(sliceStartIdxs, _slices);
            var storageStopIdxs = TransformIndices(sliceStopIdxs, _slices);
            var storageSteps = sliceSteps;
            var storageIndexes = storageStartIdxs;
            var tLength = values.Length;
            for (var tIdx = 0; tIdx < tLength; tIdx++)
            {
                //todo! switch values type and do a fast replace.
                _data.SetData(values.GetValue(tIdx), storageIndexes);
                IncrementIndexes(storageIndexes, storageStartIdxs, storageStopIdxs, storageSteps);
            }
        }

        public void Reshape(params int[] dimensions)
        {
            //Shape = new Shape(dimensions);
            //TODO: how to reshape a view?
        }

        public Span<T> View<T>(Slice slice = null)
        {
            throw new NotImplementedException();
        }

        private int[] TransformIndices(int[] coords, Slice[] slices)
        {
            var sliced_coords = new int[slices.Length];
            if (coords.Length < slices.Length)
            {
                // special case indexing into dimenionality reduced slice
                // the user of this view doesn't know the dims have been reduced so we have to augment the indices accordingly
                int coord_index = 0;
                for (int i = 0; i < slices.Length; i++)
                {
                    var slice = slices[i];
                    if (slice.IsIndex)
                    {
                        sliced_coords[i] = slice.Start.Value;
                        continue;
                    }

                    var idx = coord_index < coords.Length ? coords[coord_index] : 0;
                    coord_index++;
                    sliced_coords[i] = TransformIndex(idx, slice);
                }

                return sliced_coords;
            }

            // normal case
            for (int i = 0; i < coords.Length; i++)
            {
                var idx = coords[i];
                var slice = slices[i];
                sliced_coords[i] = TransformIndex(idx, slice);
            }

            return sliced_coords;
        }

        private int TransformIndex(int idx, Slice slice)
        {
            var start = slice.Step > 0 ? slice.Start.Value : Math.Max(0, slice.Stop.Value - 1);
            //var stop = slice.Step > 0 ? slice.Stop.Value : Math.Max(0, slice.Start.Value - 1);
            return start + idx * slice.Step;
        }


        private void IncrementIndexes(int[] storageIndexes, int[] storageStartIdxs, int[] storageStopIdxs, int[] storageSteps)
        {
            var dims = storageIndexes.Length;
            for (var dimIdx = dims - 1; dimIdx >= 0; dimIdx--)
            {
                storageIndexes[dimIdx] += storageSteps[dimIdx];
                if (storageIndexes[dimIdx] > storageStopIdxs[dimIdx])
                {
                    storageIndexes[dimIdx] = storageStartIdxs[dimIdx];
                }
                else
                {
                    break;
                }
            }
        }

        private void EnsureValidSlicingDefinitions()
        {
            // we need to be working with the original shape here because Slicing changes the own shape!
            var shape = _data.Shape;
            // we need at least one slice per dimension in order to correctly handle multi-dimensional arrays, if not given, extend by Slice.All() which returns the whole dimension
            var temp_slices = _slices;
            if (_slices == null)
                temp_slices = new Slice[0];
            _slices = new Slice[shape.NDim];
            for (int dim = 0; dim < shape.NDim; dim++)
            {
                if (temp_slices.Length > dim)
                    _slices[dim] = temp_slices[dim] ?? Slice.All(); // <-- allow to pass null for Slice.All()
                else
                    _slices[dim] = Slice.All();
            }

            for (int dim = 0; dim < shape.NDim; dim++)
            {
                var slice = _slices[dim];
                var size = shape.Dimensions[dim];
                if (slice.IsIndex)
                {
                    // special case: reduce this dimension
                    if (slice.Start < 0 || slice.Start >= size)
                        throw new IndexOutOfRangeException($"Index {slice.Start} is out of bounds for axis {dim} with size {size}");
                }

                slice.Start = Math.Max(0, slice.Start ?? 0);
                slice.Stop = Math.Min(size, slice.Stop ?? size);
            }

            // internal shape contains axis with only 1 element that will be reduced in public shape.
            _nonReducedShape = _data.Shape.Slice(_slices, reduce: false);
            Shape = _data.Shape.Slice(_slices, reduce: true);
        }


        #region Getters

        /// <summary>
        ///     Retrieves value of type <see cref="NDArray"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="NDArray"/></exception>
        public NDArray GetNDArray(params int[] indices)
        {
            return _data.GetNDArray(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="Complex"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="Complex"/></exception>
        public Complex GetComplex(params int[] indices)
        {
            return _data.GetComplex(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="bool"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="bool"/></exception>
        public bool GetBoolean(params int[] indices)
        {
            return _data.GetBoolean(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="byte"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="byte"/></exception>
        public byte GetByte(params int[] indices)
        {
            return _data.GetByte(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="short"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="short"/></exception>
        public short GetInt16(params int[] indices)
        {
            return _data.GetInt16(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="ushort"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ushort"/></exception>
        public ushort GetUInt16(params int[] indices)
        {
            return _data.GetUInt16(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="int"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="int"/></exception>
        public int GetInt32(params int[] indices)
        {
            return _data.GetInt32(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="uint"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="uint"/></exception>
        public uint GetUInt32(params int[] indices)
        {
            return _data.GetUInt32(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="long"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="long"/></exception>
        public long GetInt64(params int[] indices)
        {
            return _data.GetInt64(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="ulong"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ulong"/></exception>
        public ulong GetUInt64(params int[] indices)
        {
            return _data.GetUInt64(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="char"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="char"/></exception>
        public char GetChar(params int[] indices)
        {
            return _data.GetChar(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="double"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="double"/></exception>
        public double GetDouble(params int[] indices)
        {
            return _data.GetDouble(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="float"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="float"/></exception>
        public float GetSingle(params int[] indices)
        {
            return _data.GetSingle(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="decimal"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="decimal"/></exception>
        public decimal GetDecimal(params int[] indices)
        {
            return _data.GetDecimal(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of type <see cref="string"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="string"/></exception>
        public string GetString(params int[] indices)
        {
            return _data.GetString(TransformIndices(indices, _slices));
        }

        /// <summary>
        ///     Retrieves value of unspecified type (will figure using <see cref="IStorage.DType"/>).
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="IStorage.DType"/> is not <see cref="object"/></exception>
        public object GetValue(params int[] indices)
        {
            return _data.GetValue(TransformIndices(indices, _slices));
        }

        #endregion


        public ViewStorage Clone()
        {
            return new ViewStorage(_data, _slices) { };
        }

        #region Explicit Implementation

        void IStorage.ReplaceData(Array values, Type dtype)
        {
            throw new NotSupportedException("Unable to replace data of a view.");
        }

        object ICloneable.Clone()
        {
            return Clone();
        }

        void IStorage.Allocate(Shape shape, Type dtype)
        {
            // not sure if View should be allowed to Allocate
            throw new NotSupportedException("ViewStorage can not perform allocation.");
        }

        /// <summary>
        ///     Set an Array to internal storage, cast it to new dtype and if necessary change dtype  
        /// </summary>
        /// <param name="values"></param>
        /// <param name="typeCode"></param>
        /// <remarks>Does not copy values unless cast is necessary and doesn't change shape.</remarks>
        void IStorage.ReplaceData(Array values, NPTypeCode typeCode)
        {
            throw new NotSupportedException();
        }
        void IStorage.Allocate(Array values)
        {
            // not sure if View should be allowed to Allocate
            throw new NotSupportedException("ViewStorage can not perform allocation.");
        }

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        /// <param name="shape">The shape of given array</param>
        void IStorage.Allocate(Array values, Shape shape)
        {
            throw new NotSupportedException("ViewStorage can not perform allocation.");
        }

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        void IStorage.Allocate<T>(T[] values)
        {
            throw new NotSupportedException("ViewStorage can not perform allocation.");
        }

        #endregion

        #region ToString

        public override string ToString()
        {
            var s = new StringBuilder();
            PrettyPrint(s, false);
            return s.ToString();
        }

        public string ToString(bool flat)
        {
            var s = new StringBuilder();
            PrettyPrint(s, flat);
            return s.ToString();
        }

        private void PrettyPrint(StringBuilder s, bool flat = false)
        {
            if (Shape.Dimensions.Length == 0)
            {
                s.Append($"{GetValue(0)}");
                return;
            }

            if (Shape.Dimensions.Length == 1)
            {
                s.Append("[");
                s.Append(string.Join(", ", this.GetData().OfType<object>().Select(x => x == null ? "null" : x.ToString())));
                s.Append("]");
                return;
            }

            var last_dim = Shape.Dimensions.Last();
            var slices = new Slice[Shape.Dimensions.Length];
            s.Append("[");
            for (int i = 0; i < last_dim; i++)
            {
                slices[0] = Slice.Index(i);
                var n_minus_one_dim_slice = new ViewStorage(this, slices);
                n_minus_one_dim_slice.PrettyPrint(s, flat);
                if (i < last_dim - 1)
                {
                    s.Append(", ");
                    if (!flat)
                        s.AppendLine();
                }
            }

            s.Append("]");
        }

        #endregion

    }
}
