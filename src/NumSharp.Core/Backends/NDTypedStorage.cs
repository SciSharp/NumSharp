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
    public class NDTypedStorage : IStorage
    {
        protected bool[] _arrayBoolean;
        protected byte[] _arrayByte;
        protected char[] _arrayChar;
        protected short[] _arrayInt16;
        protected int[] _arrayInt32;
        protected uint[] _arrayUInt32;
        protected long[] _arrayInt64;
        protected float[] _arraySingle;
        protected double[] _arrayDouble;
        protected decimal[] _arrayDecimal;
        protected Complex[] _arrayComplex;
        protected string[] _arrayString;
        protected object[] _arrayObject;

        protected Type _DType;
        protected Shape _Shape;
        
        protected Array _ChangeTypeOfArray(Array arrayVar, Type dtype)
        {
            if (dtype == arrayVar.GetType().GetElementType()) return arrayVar;
            
            Array newValues = null;

            switch (Type.GetTypeCode(dtype)) 
            {
                case TypeCode.Int32 :
                    switch (Type.GetTypeCode(_DType))
                    {
                        case TypeCode.Byte:
                            newValues = Array.ConvertAll(_arrayByte, x => Convert.ToInt32(x));
                            break;
                    }
                    break;
                case TypeCode.Int64 :
                    newValues = Array.ConvertAll(_arrayInt64, x => Convert.ToInt64(x));
                    break;
                case TypeCode.Single:
                    switch (Type.GetTypeCode(_DType))
                    {
                        case TypeCode.Byte:
                            newValues = Array.ConvertAll(_arrayByte, x => Convert.ToSingle(x));
                            break;
                        case TypeCode.Double:
                            newValues = Array.ConvertAll(_arrayDouble, x => Convert.ToSingle(x));
                            break;
                    }
                    break;
                case TypeCode.Double:
                    newValues = Array.ConvertAll(_arrayByte, x => Convert.ToDouble(x));
                    break;
                case TypeCode.Decimal:
                    newValues = Array.ConvertAll(_arrayDecimal, x => Convert.ToDecimal(x));
                    break;
            }

            if(newValues == null)
                throw new NotImplementedException($"_ChangeTypeOfArray from {_DType.Name} to {dtype.Name}");

            _DType = dtype;

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
        public Shape Shape {get {return _Shape;}}

        public NDTypedStorage(Type dtype)
        {
            _DType = dtype;
            _Shape = new Shape(0);
        }

        public NDTypedStorage(double[] values)
        {
            _DType = typeof(double);
            _Shape = new Shape(values.Length);
            _arrayDouble = values;
        }

        public NDTypedStorage(object[] values)
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
        /// Get Back Storage with Columnwise tensor Layout
        /// By this method the layout is changed if layout is not columnwise
        /// </summary>
        /// <returns>reference to storage (transformed or not)</returns>
        public IStorage GetColumWiseStorage()
        {
            //if ( _TensorLayout != 2 )
                //this._ChangeRowToColumnLayout();
            
            return this;
        }

        /// <summary>
        /// Get reference to internal data storage
        /// </summary>
        /// <returns>reference to internal storage as System.Array</returns>
        public Array GetData()
        {
            switch (DType.Name)
            {
                case "Byte":
                    return _arrayByte;
                case "Boolean":
                    return _arrayBoolean;
                case "Int16":
                    return _arrayInt16;
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
        /// <param name="dtype">new storage data type</param>
        /// <returns>reference to internal (casted) storage as System.Array </returns>
        public Array GetData(Type dtype)
        {
            var methods = this.GetType().GetMethods().Where(x => x.Name.Equals("GetData") && x.IsGenericMethod && x.ReturnType.Name.Equals("T[]"));
            var genMethods = methods.First().MakeGenericMethod(dtype);

            return (Array) genMethods.Invoke(this,null);
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
                throw new Exception($"GetData {typeof(T).Name} is not {_DType.Name} of storage.");
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
        /// Get single value from internal storage and do not cast dtype
        /// </summary>
        /// <param name="indexes">indexes</param>
        /// <returns>element from internal storage</returns>
        public NDArray GetData(params int[] indexes)
        {
            if (indexes.Length == Shape.NDim ||
                Shape.Dimensions.Last() == 1)
            {
                switch (DType.Name)
                {
                    case "Boolean":
                        return _arrayBoolean[Shape.GetIndexInShape(indexes)];
                    case "Int16":
                        return _arrayInt16[Shape.GetIndexInShape(indexes)];
                    case "Int32":
                        return _arrayInt32[Shape.GetIndexInShape(indexes)];
                    case "Int64":
                        return _arrayInt64[Shape.GetIndexInShape(indexes)];
                    case "Single":
                        return _arraySingle[Shape.GetIndexInShape(indexes)];
                    case "Double":
                        return _arrayDouble[Shape.GetIndexInShape(indexes)];
                    case "Decimal":
                        return _arrayDecimal[Shape.GetIndexInShape(indexes)];
                    case "String":
                        return _arrayString[Shape.GetIndexInShape(indexes)];
                }
            }
            else if (indexes.Length == Shape.NDim - 1)
            {
                var offset = new int[Shape.NDim];
                for (int i = 0; i < Shape.NDim - 1; i++)
                    offset[i] = indexes[i];

                var nd = new NDArray(DType, Shape.Dimensions[Shape.NDim - 1]);
                var data = GetData();
                for (int i = 0; i < Shape.Dimensions[Shape.NDim - 1]; i++)
                {
                    offset[offset.Length - 1] = i;
                    nd.SetData(data.GetValue(Shape.GetIndexInShape(offset)), i);
                }

                return nd;
            }
            // 3 Dim
            else if (indexes.Length == Shape.NDim - 2)
            {
                var offset = new int[Shape.NDim];
                var nd = new NDArray(DType, new int[]{ Shape.Dimensions[Shape.NDim - 2] , Shape.Dimensions[Shape.NDim - 1] });
                var data = GetData();
                for (int i = 0; i < Shape.Dimensions[Shape.NDim - 2]; i++)
                {
                    for (int j = 0; j < Shape.Dimensions[Shape.NDim - 1]; j++)
                    {
                        offset[0] = 0;
                        offset[1] = i;
                        offset[2] = j;
                        nd.SetData(data.GetValue(Shape.GetIndexInShape(offset)), i, j);
                    }
                }

                return nd;
            }

            throw new Exception("NDStorage.GetData");
        }

        /// <summary>
        /// Get single value from internal storage as type T and cast dtype to T
        /// </summary>
        /// <param name="indexes">indexes</param>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>element from internal storage</returns>
        public T GetData<T>(params int[] indexes)
        {
            T[] values = GetData() as T[];

            return values[Shape.GetIndexInShape(indexes)];
        }

        public int GetInt32(params int[] indexes)
        {
            return _arrayInt32[Shape.GetIndexInShape(indexes)];
        }

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
        /// <param name="indexes"></param>
        public void SetData<T>(T value, params int[] indexes)
        {
            int idx = _Shape.GetIndexInShape(indexes);
            switch (value)
            {
                case bool val:
                    _arrayBoolean[idx] = val;
                    break;
                case bool[] values:
                    if (indexes.Length == 0)
                        _arrayBoolean = values;
                    else
                        _arrayBoolean.SetValue(values, idx);
                    break;
                case byte val:
                    _arrayByte[idx] = val;
                    break;
                case byte[] values:
                    if (indexes.Length == 0)
                        _arrayByte = values;
                    else
                        _arrayByte.SetValue(values, idx);
                    break;
                case short val:
                    _arrayInt16[idx] = val;
                    break;
                case short[] values:
                    if (indexes.Length == 0)
                        _arrayInt16 = values;
                    else
                        _arrayInt16.SetValue(values, idx);
                    break;
                case int val:
                    _arrayInt32[idx] = val;
                    break;
                case int[] values:
                    if (indexes.Length == 0)
                        _arrayInt32 = values;
                    else
                        _arrayInt32.SetValue(values, idx);
                    break;
                case long val:
                    _arrayInt64[idx] = val;
                    break;
                case long[] values:
                    if (indexes.Length == 0)
                        _arrayInt64 = values;
                    else
                        _arrayInt64.SetValue(values, idx);
                    break;
                case float val:
                    _arraySingle[idx] = val;
                    break;
                case float[] values:
                    if (indexes.Length == 0)
                        _arraySingle = values;
                    else
                        _arraySingle.SetValue(values, idx);
                    break;
                case double val:
                    _arrayDouble[idx] = val;
                    break;
                case double[] values:
                    if (indexes.Length == 0)
                        _arrayDouble = values;
                    else
                        _arrayDouble.SetValue(values, idx);
                    break;
                case string[] values:
                    if (indexes.Length == 0)
                        _arrayString = values;
                    else
                        _arrayString.SetValue(values, idx);
                    break;
                case NDArray nd:
                    switch(nd.dtype.Name)
                    {
                        case "Boolean":
                            _arrayBoolean.SetValue(nd.Data<bool>(0), idx);
                            break;
                        case "Int16":
                            _arrayInt16.SetValue(nd.Data<short>(0), idx);
                            break;
                        case "Int32":
                            _arrayInt32.SetValue(nd.Data<int>(0), idx);
                            break;
                        case "Int64":
                            _arrayInt64.SetValue(nd.Data<long>(0), idx);
                            break;
                        case "Single":
                            _arraySingle.SetValue(nd.Data<float>(0), idx);
                            break;
                        case "Double":
                            _arrayDouble.SetValue(nd.Data<double>(0), idx);
                            break;
                        case "Decimal":
                            _arrayDecimal.SetValue(nd.Data<decimal>(0), idx);
                            break;
                        default:
                            throw new NotImplementedException($"SetData<T>(T value, Shape indexes)");
                    }
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
            var puffer = new NDStorage(_DType);
            puffer.Allocate(new Shape(_Shape.Dimensions));
            puffer.SetData((Array)GetData(_DType).Clone());

            return puffer;
        }
    }
}
