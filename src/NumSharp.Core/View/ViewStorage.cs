using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public class ViewStorage : IStorage
    {

        public ViewStorage(IStorage dataStorage, string slice_notation) : this(dataStorage, Slice.ParseSlices(slice_notation))
        {
        }

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
            _data = dataStorage;
            _slices = slices;
            EnsureValidSlicingDefinitions();
        }

        private IStorage _data = null;
        private Slice[] _slices = null;
        private Shape internal_shape;

        private void EnsureValidSlicingDefinitions()
        {
            // we need to be working with the original shape here because Slicing changes the own shape!
            var shape = _data.Shape;
            // we need at least one slice per dimension in order to correctly handle multi-dimensional arrays, if not given, extend by Slice.All() which returns the whole dimension
            var temp_slices = _slices;
            if (_slices == null)
                temp_slices=new Slice[0];
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
            internal_shape = _data.Shape.Slice(_slices, reduce:false);
            Shape = _data.Shape.Slice(_slices, reduce:true);
        }

        public object Clone()
        {
            return new ViewStorage(_data, _slices);
        }

        public Type DType => _data.DType;
        public int DTypeSize => _data.DTypeSize;
        public Slice Slice { get; set; } // <--- this is not doing anything in View! 
        public Shape Shape { get; private set; }


        public void Allocate(Shape shape, Type dtype = null)
        {
            // not sure if View should be allowed to Allocate
            throw new NotImplementedException();
        }

        public void Allocate(Array values)
        {
            // not sure if View should be allowed to Allocate
            throw new NotImplementedException();
        }

        public Array GetData()
        {
            // since the view is a subset of the data we have to copy here
            int size = internal_shape.Size;
            var data = AllocateArray(size, _data.DType);
            // the algorithm is split into 1-D and N-D because for 1-D we need not go through coordinate transformation
            if (_slices.Length == 1)
            {
                for (var i = 0; i < size; i++)
                    data.SetValue(GetValue(i), i);
            }
            else
            {
                for (var i = 0; i < size; i++)
                    data.SetValue(GetValue(internal_shape.GetDimIndexOutShape(i)), i);
            }
            return data;
        }

        private object GetValue(params int[] idx)
        {
            switch (DType.Name)
            {
                //case "Byte":
                //    return GetByte(idx);
                case "Boolean":
                    return GetBoolean(idx);
                case "Int16":
                    return GetInt16(idx);
                case "Int32":
                    return GetInt32(idx);
                case "Int64":
                    return GetInt64(idx);
                //case "UInt32":
                //    return GetUInt32(idx);
                case "Single":
                    return GetSingle(idx);
                case "Double":
                    return GetDouble(idx);
                case "Decimal":
                    return GetDecimal(idx);
                case "String":
                    return GetString(idx);
                //case "Object":
                //    return GetObject(idx);
                case "NDArray":
                    return GetNDArray(idx);
                default:
                    throw new NotImplementedException($"GetValue {DType.Name}");
            }
        }


        private Array AllocateArray(int size, Type dataDType)
        {
            switch (dataDType.Name)
            {
                case "Byte":
                    return new byte[size];
                case "Boolean":
                    return new bool[size];
                case "Int16":
                    return new short[size];
                case "Int32":
                    return new int[size];
                case "Int64":
                    return new long[size];
                case "UInt32":
                    return new uint[size];
                case "Single":
                    return new float[size];
                case "Double":
                    return new double[size];
                case "Decimal":
                    return new decimal[size];
                case "String":
                    return new string[size];
                case "Object":
                    return new object[size];
                case "NDArray":
                    return new NDArray[size];
                default:
                    throw new NotImplementedException($"AllocateArray {dataDType.Name}");
            }
        }

        public Array CloneData()
        {
            throw new NotImplementedException("Cloning the data is not supported in view");
        }

        public T[] GetData<T>()
        {
            return GetData() as T[];
        }

        public T GetData<T>(params int[] indices)
        {
            return _data.GetData<T>(TransformIndices(indices, _slices));
        }

        public bool SupportsSpan => false;

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

        public Span<T> GetSpanData<T>(Slice slice, params int[] indices)
        {
            throw new NotImplementedException("span of a view is not supported as it typically is not contiguous memory");
        }

        public bool GetBoolean(params int[] indices)
        {
            return _data.GetBoolean(TransformIndices(indices, _slices));
        }

        public short GetInt16(params int[] indices)
        {
            return _data.GetInt16(TransformIndices(indices, _slices));
        }

        public int GetInt32(params int[] indices)
        {
            return _data.GetInt32(TransformIndices(indices, _slices));
        }

        public long GetInt64(params int[] indices)
        {
            return _data.GetInt64(TransformIndices(indices, _slices));
        }

        public float GetSingle(params int[] indices)
        {
            return _data.GetSingle(TransformIndices(indices, _slices));
        }

        public double GetDouble(params int[] indices)
        {
            return _data.GetDouble(TransformIndices(indices, _slices));
        }

        public decimal GetDecimal(params int[] indices)
        {
            return _data.GetDecimal(TransformIndices(indices, _slices));
        }

        public string GetString(params int[] indices)
        {
            return _data.GetString(TransformIndices(indices, _slices));
        }

        public NDArray GetNDArray(params int[] indices)
        {
            return _data.GetNDArray(TransformIndices(indices, _slices));
        }

        public void SetData(Array values)
        {
            throw new NotImplementedException();
        }

        public void SetData<T>(Array values)
        {
            throw new NotImplementedException();
        }

        public void SetData<T>(T value, params int[] indexes)
        {
            _data.SetData<T>(value, TransformIndices(indexes, _slices));
        }

        public void SetData(Array values, Type dtype)
        {
            throw new NotImplementedException();
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


        public override string ToString()
        {
            var s = new StringBuilder();
            PrettyPrint(s);
            return s.ToString();
        }

        public string ToString(bool flat = false)
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
                var n_minus_one_dim_slice = new ViewStorage( this, slices);
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
    }
}
