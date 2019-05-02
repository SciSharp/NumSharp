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
            if (dataStorage==null)
                throw  new ArgumentException("dataStorage must not be null");
            if (slices == null)
                throw new ArgumentException("slices must not be null");
            if (slices.Length == 0)
                throw new ArgumentException("slices must contain at least one slice");
            _data = dataStorage;
            _slices = slices;
            EnsureSliceStartStop();
        }

        private IStorage _data = null;
        private Slice[] _slices =  null;

        private void EnsureSliceStartStop()
        {
            if (_slices.Length == 1)
            {
                _slices[0].Start = Math.Max(0, _slices[0].Start ?? 0);
                var size = _data.Shape.Size;
                _slices[0].Stop = Math.Min(size, _slices[0].Stop ?? size);                    
            }
            else
            {
                throw new NotImplementedException("Multi-Dim slicing");
            }
        }

        public object Clone()
        {
            return new ViewStorage(_data, _slices);
        }

        public Type DType => _data.DType;
        public int DTypeSize => _data.DTypeSize;
        public Slice Slice { get; set; } // <--- this is not doing anything in View! 
        public Shape Shape { get { return _data.Shape.Slice(_slices); } }

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
            if (_slices.Length == 1)
            {
                var slice = _slices[0];
                int size = Shape.Size; 
                var data = AllocateArray(size, _data.DType);
                for (var i = 0; i < size; i++)
                    data.SetValue(GetValue(i), i);
                return data;
            }
            else
            {
                throw new NotImplementedException("Multi-Dim slicing");
            }
        }

        private object GetValue(int idx)
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
            throw  new NotImplementedException("Cloning the data is not supported in view");            
        }

        public T[] GetData<T>()
        {
            return GetData() as T[];
        }

        public T GetData<T>(params int[] indices)
        {
            return _data.GetData<T>(TransformIndices(indices, _slices));
        }

        // todo move to Shape when implementing of multi-dim slicing
        private int[] TransformIndices(int[] indices, Slice[] slices)
        {
            // transform in-place, for performance reasons
            for (int i = 0; i < indices.Length; i++)
            {
                var idx = indices[i];
                indices[i] = TransformIndex(idx, slices);
            }
            return indices;
        }

        // todo move to Shape when implementing of multi-dim slicing
        private int TransformIndex(int idx, Slice[] slices)
        {
            if (slices.Length == 1)
            {
                var slice = slices[0];
                var start = slice.Step > 0 ? slice.Start.Value : Math.Max(0,  slice.Stop.Value-1);
                //var stop = slice.Step > 0 ? slice.Stop.Value : Math.Max(0, slice.Start.Value - 1);
                return start + idx * slice.Step;
            }
            else 
                throw new NotImplementedException("Multi-Dim slicing");
        }

        public Span<T> GetSpanData<T>(params int[] indices)
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
    }
}
