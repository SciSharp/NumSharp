using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NumSharp.Backends
{
    /// <summary>
    /// Storage
    ///
    /// Responsible for :
    ///
    ///  - store data type, elements, Shape
    ///  - offers methods for accessing elements depending on shape
    ///  - offers methods for casting elements
    ///  - offers methods for change tensor order
    ///  - GetData always return reference object to the true storage
    ///  - GetData<T> and SetData<T> change dtype and cast storage
    ///  - CloneData always create a clone of storage and return this as reference object
    ///  - CloneData<T> clone storage and cast this clone 
    ///     
    /// </summary>
    public class TypedArrayStorage : IStorage
    {
        protected bool[] _arrayBoolean;
        protected byte[] _arrayByte;
        protected char[] _arrayChar;
        protected short[] _arrayInt16;
        protected ushort[] _arrayUInt16;
        protected int[] _arrayInt32;
        protected uint[] _arrayUInt32;
        protected long[] _arrayInt64;
        protected float[] _arraySingle;
        protected double[] _arrayDouble;
        protected decimal[] _arrayDecimal;
        protected Complex[] _arrayComplex;
        protected string[] _arrayString;
        protected NDArray[] _arrayNDArray;
        protected object[] _arrayObject;

        protected Type _DType;
        protected Shape _Shape;
        protected Slice _slice;

        /// <summary>
        /// Flag current storage order
        /// This flag will be different from Shape.Order when order is changed 
        /// </summary>
        private string order = "C";

        protected Array _ChangeTypeOfArray(Array arrayVar, Type to_dtype)
        {
            if (to_dtype == arrayVar.GetType().GetElementType()) return arrayVar;
            
            Array newValues = null;

            switch (Type.GetTypeCode(to_dtype)) 
            {
                case TypeCode.Byte:
                    {
                        switch (Type.GetTypeCode(_DType))
                        {
                            case TypeCode.Boolean:
                                newValues = Array.ConvertAll(_arrayBoolean, x => Convert.ToByte(x));
                                break;
                            case TypeCode.Int32:
                                newValues = Array.ConvertAll(_arrayInt32, x => Convert.ToByte(x));
                                break;
                        }
                    }
                    break;
                case TypeCode.UInt16:
                    {
                        switch (Type.GetTypeCode(_DType))
                        {
                            case TypeCode.Byte:
                                newValues = Array.ConvertAll(_arrayByte, x => Convert.ToUInt16(x));
                                break;
                            case TypeCode.Int32:
                                newValues = Array.ConvertAll(_arrayInt32, x => Convert.ToUInt16(x));
                                break;
                        }
                    }
                    break;
                case TypeCode.Int32 :
                    switch (Type.GetTypeCode(_DType))
                    {
                        case TypeCode.Byte:
                            newValues = Array.ConvertAll(_arrayByte, x => Convert.ToInt32(x));
                            break;
                        case TypeCode.Int64:
                            newValues = Array.ConvertAll(_arrayInt64, x => Convert.ToInt32(x));
                            break;
                    }
                    break;
                case TypeCode.Int64 :
                    switch (Type.GetTypeCode(_DType))
                    {
                        case TypeCode.Int32:
                            newValues = Array.ConvertAll(_arrayInt32, x => Convert.ToInt64(x));
                            break;
                    }
                    break;
                case TypeCode.Single:
                    switch (Type.GetTypeCode(_DType))
                    {
                        case TypeCode.Byte:
                            newValues = Array.ConvertAll(_arrayByte, x => Convert.ToSingle(x));
                            break;
                        case TypeCode.Single:
                            newValues = Array.ConvertAll(_arraySingle, x => Convert.ToSingle(x));
                            break;
                        case TypeCode.Double:
                            newValues = Array.ConvertAll(_arrayDouble, x => Convert.ToSingle(x));
                            break;
                    }
                    break;
                case TypeCode.Double:
                    switch (Type.GetTypeCode(_DType))
                    {
                        case TypeCode.Byte:
                            newValues = Array.ConvertAll(_arrayByte, x => Convert.ToDouble(x));
                            break;
                        case TypeCode.Single:
                            newValues = Array.ConvertAll(_arraySingle, x => Convert.ToDouble(x));
                            break;
                        case TypeCode.Double:
                            newValues = Array.ConvertAll(_arrayDouble, x => Convert.ToDouble(x));
                            break;
                    }
                    break;
                case TypeCode.Decimal:
                    newValues = Array.ConvertAll(_arrayDecimal, x => Convert.ToDecimal(x));
                    break;
            }

            if(newValues == null)
                throw new NotImplementedException($"_ChangeTypeOfArray from {_DType.Name} to {to_dtype.Name}");

            _DType = to_dtype;

            return newValues;
        }

        /// <summary>
        /// Data Type of stored elements
        /// </summary>
        /// <value>numpys equal dtype</value>
        public Type DType {get {return _DType;}}

        public int DTypeSize
        {
            get
            {
                if(_DType == typeof(string))
                {
                    return 0;
                }
                else
                {
                    return Marshal.SizeOf(_DType);
                }
            }
        }
        /// <summary>
        /// storage shape for outside representation
        /// </summary>
        /// <value>numpys equal shape</value>
        public Shape Shape => _Shape;

        public Slice Slice { get; set; }

        public TypedArrayStorage(Type dtype)
        {
            _DType = dtype;
            _Shape = new Shape(0);
        }

        public TypedArrayStorage(double[] values)
        {
            _DType = typeof(double);
            _Shape = new Shape(values.Length);
            _arrayDouble = values;
        }

        public TypedArrayStorage(int[] values)
        {
            _DType = typeof(int);
            _Shape = new Shape(values.Length);
            _arrayInt32 = values;
        }

        public TypedArrayStorage(object[] values)
        {
            _DType = values.GetType().GetElementType();
            _Shape = new Shape(values.Length);
            _arrayObject = values;
        }

        /// <summary>
        /// Allocate memory by dtype, shape, tensororder (default column wise)
        /// </summary>
        /// <param name="dtype">storage data type</param>
        /// <param name="shape">storage data shape</param>
        public void Allocate(Shape shape, Type dtype = null)
        {
            _Shape = shape;

            if (dtype != null)
                _DType = dtype;

            switch (_DType.Name)
            {
                case "Byte":
                    _arrayByte = new byte[shape.Size];
                    break;
                case "Boolean":
                    _arrayBoolean = new bool[shape.Size];
                    break;
                case "Int16":
                    _arrayInt16 = new short[shape.Size];
                    break;
                case "UInt16":
                    _arrayUInt16 = new ushort[shape.Size];
                    break;
                case "Int32":
                    _arrayInt32 = new int[shape.Size];
                    break;
                case "Int64":
                    _arrayInt64 = new long[shape.Size];
                    break;
                case "UInt32":
                    _arrayUInt32 = new uint[shape.Size];
                    break;
                case "Single":
                    _arraySingle = new float[shape.Size];
                    break;
                case "Double":
                    _arrayDouble = new double[shape.Size];
                    break;
                case "Decimal":
                    _arrayDecimal = new decimal[shape.Size];
                    break;
                case "String":
                    _arrayString = new string[shape.Size];
                    break;
                case "Object":
                    _arrayObject = new object[shape.Size];
                    break;
                case "NDArray":
                    _arrayNDArray = new NDArray[shape.Size];
                    break;
                default:
                    throw new NotImplementedException($"Allocate {_DType.Name}");
            }
        }

        /// <summary>
        /// Allocate memory by Array and tensororder and deduce shape and dtype (default column wise)
        /// </summary>
        /// <param name="values">elements to store</param>
        public void Allocate(Array values)
        {
            int[] dim = new int[values.Rank];
            for (int idx = 0; idx < dim.Length;idx++)
                dim[idx] = values.GetLength(idx);
            
            _Shape = new Shape(dim);
            Type elementType = values.GetType();
            while (elementType.IsArray)
                elementType = elementType.GetElementType();
            
            _DType = elementType;
        }

        public void Allocate<T>(T[] values)
        {
            int[] dim = new int[values.Rank];
            for (int idx = 0; idx < dim.Length; idx++)
                dim[idx] = values.GetLength(idx);

            _Shape = new Shape(dim);
            Type elementType = values.GetType();
            while (elementType.IsArray)
                elementType = elementType.GetElementType();

            _DType = elementType;
        }

        /// <summary>
        /// Get reference to internal data storage
        /// </summary>
        /// <returns>reference to internal storage as System.Array</returns>
        public Array GetData()
        {
            /*if(order != Shape.Order)
            {
                if (Shape.Order == "F")
                {
                    var s = new Shape(Shape.Dimensions);
                    s.ChangeTensorLayout("C");

                    switch (Type.GetTypeCode(DType))
                    {
                        case TypeCode.Int32:
                            {
                                var array = new int[Shape.Size];
                                switch (Shape.NDim)
                                {
                                    case 2:
                                        for (int row = 0; row < Shape[0]; row++)
                                            for (int col = 0; col < Shape[1]; col++)
                                                array[s.GetIndexInShape(row, col)] = _arrayInt32[Shape.GetIndexInShape(row, col)];
                                        _arrayInt32 = array;
                                        break;
                                    default:
                                        throw new NotImplementedException("Array GetData() Order F Changed.");
                                }
                            }
                            break;
                        case TypeCode.Single:
                            {
                                var array = new float[Shape.Size];
                                switch (Shape.NDim)
                                {
                                    case 2:
                                        for (int row = 0; row < Shape[0]; row++)
                                            for (int col = 0; col < Shape[1]; col++)
                                                array[s.GetIndexInShape(row, col)] = _arraySingle[Shape.GetIndexInShape(row, col)];
                                        _arraySingle = array;
                                        break;
                                    default:
                                        throw new NotImplementedException("Array GetData() Order F Changed.");
                                }
                            }
                            break;
                    }

                    Shape.ChangeTensorLayout("C");
                    order = Shape.Order;
                }
                else if (Shape.Order == "C")
                {
                    throw new NotImplementedException("Array GetData() Order C Changed.");
                }

                
            }*/

            switch (DType.Name)
            {
                case "Byte":
                    return _arrayByte;
                case "Boolean":
                    return _arrayBoolean;
                case "Int16":
                    return _arrayInt16;
                case "UInt16":
                    return _arrayUInt16;
                case "Int32":
                    return _arrayInt32;
                case "Int64":
                    return _arrayInt64;
                case "UInt32":
                    return _arrayUInt32;
                case "Single":
                    return _arraySingle;
                case "Double":
                    return _arrayDouble;
                case "String":
                    return _arrayString;
                case "NDArray":
                    return _arrayNDArray;
            }

            throw new NotImplementedException($"GetData {DType.Name}");
        }

        /// <summary>
        /// Clone internal storage and get reference to it
        /// </summary>
        /// <returns>reference to cloned storage as System.Array</returns>
        public Array CloneData()
        {
            return GetData().Clone() as Array;
        }

        /// <summary>
        /// Get reference to internal data storage and cast elements to new dtype
        /// </summary>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>reference to internal (casted) storage as T[]</returns>
        public T[] GetData<T>()
        {
            if (typeof(T).Name != _DType.Name)
            {
                Console.WriteLine($"Warning: GetData {typeof(T).Name} is not {_DType.Name} of storage.");
                SetData(_ChangeTypeOfArray(GetData(), typeof(T)));
            }

            return GetData() as T[];
        }

        /// <summary>
        /// Get all elements from cloned storage as T[] and cast dtype
        /// </summary>
        /// <typeparam name="T">cloned storgae dtype</typeparam>
        /// <returns>reference to cloned storage as T[]</returns>
        public T[] CloneData<T>()
        {
            var puffer = (Array) this.GetData().Clone();
            puffer = _ChangeTypeOfArray(puffer, typeof(T));

            return puffer as T[];
        }

        /// <summary>
        /// Get single value from internal storage as type T and cast dtype to T
        /// </summary>
        /// <param name="indexes">indexes</param>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>element from internal storage</returns>
        public T GetData<T>(params int[] indexes)
        {
            T[] values = GetData<T>();

            return values[Shape.GetIndexInShape(Slice, indexes)];
        }

        public bool SupportsSpan => true;

        public bool GetBoolean(params int[] indexes)
            => _arrayBoolean[Shape.GetIndexInShape(Slice, indexes)];

        public byte GetByte(params int[] indexes)
            => _arrayByte[Shape.GetIndexInShape(Slice, indexes)];

        public short GetInt16(params int[] indexes)
            => _arrayInt16[Shape.GetIndexInShape(Slice, indexes)];

        public ushort GetUInt16(params int[] indexes)
            => _arrayUInt16[Shape.GetIndexInShape(Slice, indexes)];

        public int GetInt32(params int[] indexes) 
            => _arrayInt32[Shape.GetIndexInShape(Slice, indexes)];

        public long GetInt64(params int[] indexes) 
            => _arrayInt64[Shape.GetIndexInShape(Slice, indexes)];

        public float GetSingle(params int[] indexes) 
            => _arraySingle[Shape.GetIndexInShape(Slice, indexes)];

        public double GetDouble(params int[] indexes) 
            => _arrayDouble[Shape.GetIndexInShape(Slice, indexes)];

        public decimal GetDecimal(params int[] indexes) 
            => _arrayDecimal[Shape.GetIndexInShape(Slice, indexes)];

        public string GetString(params int[] indexes)
            => _arrayString[Shape.GetIndexInShape(Slice, indexes)];

        public NDArray GetNDArray(params int[] indexes) 
            => _arrayNDArray[Shape.GetIndexInShape(Slice, indexes)];

        /// <summary>
        /// Set an array to internal storage and keep dtype
        /// </summary>
        /// <param name="values"></param>
        public void SetData(Array values)
        {
            if (_DType != values.GetType().GetElementType())
            {
                //_values = _ChangeTypeOfArray(values, _DType);
            }
            else
            {
                switch (DType.Name)
                {
                    case "Boolean":
                        _arrayBoolean = values as bool[];
                        break;
                    case "Byte":
                        _arrayByte = values as byte[];
                        break;
                    case "Int16":
                        _arrayInt16 = values as short[];
                        break;
                    case "UInt16":
                        _arrayUInt16 = values as ushort[];
                        break;
                    case "Int32":
                        _arrayInt32 = values as int[];
                        break;
                    case "UInt32":
                        _arrayUInt32 = values as uint[];
                        break;
                    case "Int64":
                        _arrayInt64 = values as long[];
                        break;
                    case "Single":
                        _arraySingle = values as float[];
                        break;
                    case "Double":
                        _arrayDouble = values as double[];
                        break;
                    case "Decimal":
                        _arrayDecimal = values as decimal[];
                        break;
                    case "String":
                        _arrayString = values as string[];
                        break;
                    case "NDArray":
                        _arrayNDArray = values as NDArray[];
                        break;
                    case "Object":
                        _arrayObject = values as object[];
                        break;
                    default:
                        throw new NotImplementedException($"SetData {DType.Name}");
                }
            }
        }

        /// <summary>
        /// Set 1 single value to internal storage and keep dtype
        /// </summary>
        /// <param name="value"></param>
        /// <param name="indice"></param>
        public void SetData<T>(T value, params int[] indice)
        {
            int idx = _Shape.GetIndexInShape(Slice, indice);
            switch (value)
            {
                case bool val:
                    _arrayBoolean[idx] = val;
                    break;
                case bool[] values:
                    if (indice.Length == 0)
                        _arrayBoolean = values;
                    else
                        _arrayBoolean.SetValue(values, idx);
                    break;
                case byte val:
                    _arrayByte[idx] = val;
                    break;
                case byte[] values:
                    if (indice.Length == 0)
                        _arrayByte = values;
                    else
                        _arrayByte.SetValue(values, idx);
                    break;
                case short val:
                    _arrayInt16[idx] = val;
                    break;
                case short[] values:
                    if (value.GetType() == typeof(ushort[]))
                    {
                        if (indice.Length == 0)
                            _arrayUInt16 = value as ushort[];
                        else
                            _arrayUInt16.SetValue(values, idx);
                        break;
                    }
                    else
                    {
                        if (indice.Length == 0)
                            _arrayInt16 = values;
                        else
                            _arrayInt16.SetValue(values, idx);
                    }
                    break;
                case ushort val:
                    _arrayUInt16[idx] = val;
                    break;
                case ushort[] values:
                    if (indice.Length == 0)
                        _arrayUInt16 = values;
                    else
                        _arrayUInt16.SetValue(values, idx);
                    break;
                case int val:
                    _arrayInt32[idx] = val;
                    break;
                case int[] values:
                    if (indice.Length == 0)
                        _arrayInt32 = values;
                    else
                        _arrayInt32.SetValue(values, idx);
                    break;
                case long val:
                    _arrayInt64[idx] = val;
                    break;
                case long[] values:
                    if (indice.Length == 0)
                        _arrayInt64 = values;
                    else
                        _arrayInt64.SetValue(values, idx);
                    break;
                case float val:
                    _arraySingle[idx] = val;
                    break;
                case float[] values:
                    if (indice.Length == 0)
                        _arraySingle = values;
                    else
                        _arraySingle.SetValue(values, idx);
                    break;
                case double val:
                    _arrayDouble[idx] = val;
                    break;
                case double[] values:
                    if (indice.Length == 0)
                        _arrayDouble = values;
                    else
                        _arrayDouble.SetValue(values, idx);
                    break;
                case string[] values:
                    if (indice.Length == 0)
                        _arrayString = values;
                    else
                        _arrayString.SetValue(values, idx);
                    break;
                case NDArray nd:
                    int offset = 0;
                    if (Shape.NDim == 1)
                        offset = idx + (Slice is null ? 0 : Slice.Start.Value);
                    else
                        offset = idx + (Slice is null ? 0 : Slice.Start.Value) * Shape.Strides[0];
                    if (nd.ndim == 0)
                    {
                        // we are setting a scalar
                        switch (Type.GetTypeCode(nd.dtype))
                        {
                            case TypeCode.Boolean:
                                _arrayBoolean[offset]=nd;
                                break;
                            case TypeCode.Int16:
                                _arrayInt16[offset]=nd;
                                break;
                            case TypeCode.Int32:
                                _arrayInt32[offset] = nd;
                                break;
                            case TypeCode.Int64:
                                _arrayInt64[offset]=nd;
                                break;
                            case TypeCode.Single:
                                _arraySingle[offset]=nd;
                                break;
                            case TypeCode.Double:
                                _arrayDouble[offset]=nd;
                                break;
                            case TypeCode.Decimal:
                                _arrayDecimal[offset]=nd;
                                break;
                            case TypeCode.String:
                                _arrayString[offset]=nd;
                                break;
                            default:
                                throw new NotImplementedException($"SetData<T>(T value, Shape indexes)");
                        }
                        return;
                    }
                    // we are setting a range
                    switch (Type.GetTypeCode(nd.dtype))
                    {
                        case TypeCode.Boolean:
                            nd.Data<bool>().AsSpan().CopyTo(_arrayBoolean.AsSpan(offset));
                            break;
                        case TypeCode.Int16:
                            nd.Data<short>().AsSpan().CopyTo(_arrayInt16.AsSpan(offset));
                            break;
                        case TypeCode.UInt16:
                            nd.Data<ushort>().AsSpan().CopyTo(_arrayUInt16.AsSpan(offset));
                            break;
                        case TypeCode.Int32:
                            nd.Data<int>().AsSpan().CopyTo(_arrayInt32.AsSpan(offset));
                            break;
                        case TypeCode.Int64:
                            nd.Data<long>().AsSpan().CopyTo(_arrayInt64.AsSpan(offset));
                            break;
                        case TypeCode.Single:
                            nd.Data<float>().AsSpan().CopyTo(_arraySingle.AsSpan(offset));
                            break;
                        case TypeCode.Double:
                            nd.Data<double>().AsSpan().CopyTo(_arrayDouble.AsSpan(offset));
                            break;
                        case TypeCode.Decimal:
                            nd.Data<decimal>().AsSpan().CopyTo(_arrayDecimal.AsSpan(offset));
                            break;
                        case TypeCode.String:
                            nd.Data<string>().AsSpan().CopyTo(_arrayString.AsSpan(offset));
                            break;
                        default:
                            throw new NotImplementedException($"SetData<T>(T value, Shape indexes)");
                    }
                    break;
                case NDArray[] nd:
                    _arrayNDArray = nd;
                    break;
                default:
                    throw new NotImplementedException($"SetData<T>(T value, Shape indexes)");
            }
                
        }

        /// <summary>
        /// Set a 1D Array of type T to internal storage and cast dtype
        /// </summary>
        /// <param name="values"></param>
        /// <typeparam name="T"></typeparam>
        public void SetData<T>(Array values)
        {
            SetData(values, typeof(T));
        }

        /// <summary>
        /// Set an Array to internal storage, cast it to new dtype and change dtype  
        /// </summary>
        /// <param name="values"></param>
        /// <param name="dtype"></param>
        public void SetData(Array values, Type dtype)
        {
            SetData(_ChangeTypeOfArray(values, dtype));
        } 

        public void SetNewShape(params int[] dimensions)
        {
            _Shape = new Shape(dimensions);
        }

        public void Reshape(params int[] dimensions)
        {
            _Shape = new Shape(dimensions);
        }

        public object Clone()
        {
            var puffer = new ArrayStorage(_DType);
            puffer.Allocate(new Shape(_Shape.Dimensions));
            puffer.SetData((Array)GetData().Clone());

            return puffer;
        }

        public Span<T> View<T>(Slice slice = null)
        {
            if (slice is null)
                slice = Slice;

            if (slice is null)
            {
                return GetData<T>().AsSpan();
            }
            else
            {
                var shape = Shape.GetShape(Shape.Dimensions, axis: 0);
                var offset = Shape.GetSize(shape);
                return GetData<T>().AsSpan(slice.Start.Value * offset, slice.Length.Value * offset);
            }
        }

        public Span<T> GetSpanData<T>(Slice slice, params int[] indice)
        {
            int stride = Shape.NDim == 0 ? 1 : Shape.Strides[indice.Length - 1];
            int idx = Shape.GetIndexInShape(Slice, indice);
            int offset = idx + (Slice is null ? 0 : Slice.Start.Value) * Shape.Strides[0];

            return GetData<T>().AsSpan(offset, stride);
        }
    }
}
