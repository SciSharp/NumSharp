using System;
using System.Linq;
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
    public class ArrayStorage : IStorage
    {
        /// <summary>
        /// storage, low performance when element-wise access
        /// will refactor this seperate into dedicate typed 1-d array
        /// </summary>
        protected Array _values;

        protected Type _DType;
        protected Shape _Shape;
        
        protected Array _ChangeTypeOfArray(Array arrayVar, Type dtype)
        {
            if (dtype == arrayVar.GetType().GetElementType()) return arrayVar;

            _DType = dtype;
            Array newValues = null;

            switch (Type.GetTypeCode(dtype)) 
            {
                case TypeCode.Double : 
                {
                    newValues = new double[arrayVar.Length];
                    for(int idx = 0;idx < arrayVar.Length;idx++)
                        newValues.SetValue(Convert.ToDouble(arrayVar.GetValue(idx)),idx);
                    break;
                }
                case TypeCode.Single : 
                {
                    newValues = new float[arrayVar.Length];
                    for(int idx = 0;idx < arrayVar.Length;idx++)
                        newValues.SetValue(Convert.ToSingle(arrayVar.GetValue(idx)),idx);
                    break;
                }
                case TypeCode.Decimal : 
                {
                    newValues = new Decimal[arrayVar.Length];
                    for(int idx = 0;idx < arrayVar.Length;idx++)
                        newValues.SetValue(Convert.ToDecimal(arrayVar.GetValue(idx)),idx);
                    break;
                }
                case TypeCode.Int32 : 
                {
                    newValues = new int[arrayVar.Length];
                    for(int idx = 0;idx < arrayVar.Length;idx++)
                        newValues.SetValue(Convert.ToInt32(arrayVar.GetValue(idx)),idx);
                    break;
                }
                case TypeCode.Int64 :
                {
                    newValues = new Int64[arrayVar.Length];
                    for(int idx = 0;idx < arrayVar.Length;idx++)
                        newValues.SetValue(Convert.ToInt64(arrayVar.GetValue(idx)),idx);
                    break;
                }
                case TypeCode.Object : 
                {
                    if( dtype == typeof(System.Numerics.Complex) )
                    {
                        newValues = new System.Numerics.Complex[arrayVar.Length];
                        for(int idx = 0;idx < arrayVar.Length;idx++)
                            newValues.SetValue(new System.Numerics.Complex((double)arrayVar.GetValue(idx),0),idx);
                        break;
                    }
                    /*else if ( dtype == typeof(System.Numerics.Quaternion) )
                    {
                        newValues = new System.Numerics.Quaternion[arrayVar.Length];
                        for(int idx = 0;idx < arrayVar.Length;idx++)
                            newValues.SetValue(new System.Numerics.Quaternion(new System.Numerics.Vector3(0,0,0) , (float)arrayVar.GetValue(idx)),idx);
                        break;
                    }*/
                    else 
                    {
                        newValues = new object[arrayVar.Length];
                        for(int idx = 0;idx < arrayVar.Length;idx++)
                            newValues.SetValue(arrayVar.GetValue(idx),idx);
                        break;
                    }
                    
                }
                default :
                    newValues = arrayVar;
                    break;
            }

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

        public ArrayStorage(Type dtype)
        {
            _DType = dtype;
            _values = Array.CreateInstance(dtype, 1);
            _Shape = new Shape(1);
        }

        public ArrayStorage(double[] values)
        {
            _DType = typeof(double);
            _Shape = new Shape(values.Length);
            _values = values;
        }

        public ArrayStorage(object[] values)
        {
            _DType = values.GetType().GetElementType();
            _Shape = new Shape(values.Length);
            _values = values;
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
            _values = Array.CreateInstance(_DType, shape.Size);
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
            return _values;
        }

        public int GetInt32(params int[] indexes)
        {
            var data = _values as int[];
            return data[Shape.GetIndexInShape(indexes)];
        }

        public float GetSingle(params int[] indexes)
        {
            var data = _values as float[];
            return data[Shape.GetIndexInShape(indexes)];
        }

        /// <summary>
        /// Clone internal storage and get reference to it
        /// </summary>
        /// <returns>reference to cloned storage as System.Array</returns>
        public Array CloneData()
        {
            return (Array) _values.Clone();
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
        /// Clone internal storage and cast elements to new dtype
        /// </summary>
        /// <param name="dtype">cloned storage data type</param>
        /// <returns>reference to cloned storage as System.Array</returns>
        public Array CloneData(Type dtype)
        {
            return (Array) this.GetData().Clone();
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
                
            return _values as T[];
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
                        return GetData<bool>(indexes);
                    case "Int16":
                        return GetData<short>(indexes);
                    case "Int32":
                        return GetData<int>(indexes);
                    case "Int64":
                        return GetData<long>(indexes);
                    case "Single":
                        return GetData<float>(indexes);
                    case "Double":
                        return GetData<double>(indexes);
                    case "Decimal":
                        return GetData<decimal>(indexes);
                    case "String":
                        return GetData<string>(indexes);
                }
            }
            else if (indexes.Length == Shape.NDim - 1)
            {
                var offset = new int[Shape.NDim];
                for (int i = 0; i < Shape.NDim - 1; i++)
                    offset[i] = indexes[i];

                var nd = new NDArray(DType, Shape.Dimensions[Shape.NDim - 1]);
                for (int i = 0; i < Shape.Dimensions[Shape.NDim - 1]; i++)
                {
                    offset[offset.Length - 1] = i;
                    nd.SetData(_values.GetValue(Shape.GetIndexInShape(offset)), i);
                }

                return nd;
            }
            // 3 Dim
            else if (indexes.Length == Shape.NDim - 2)
            {
                var offset = new int[Shape.NDim];
                var nd = new NDArray(DType, new int[]{ Shape.Dimensions[Shape.NDim - 2] , Shape.Dimensions[Shape.NDim - 1] });
                
                for (int i = 0; i < Shape.Dimensions[Shape.NDim - 2]; i++)
                {
                    for (int j = 0; j < Shape.Dimensions[Shape.NDim - 1]; j++)
                    {
                        offset[0] = 0;
                        offset[1] = i;
                        offset[2] = j;
                        nd.SetData(_values.GetValue(Shape.GetIndexInShape(offset)), i, j);
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

        /// <summary>
        /// Set an array to internal storage and keep dtype
        /// </summary>
        /// <param name="values"></param>
        public void SetData(Array values)
        {
            if (_DType != values.GetType().GetElementType())
            {
                _values = _ChangeTypeOfArray(values, _DType);
            }
            else
            {
                _values = values;
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
                    _values.SetValue(val, idx);
                    break;
                case bool[] values:
                    if (indexes.Length == 0)
                        _values = values;
                    else
                        _values.SetValue(values, idx);
                    break;
                case byte val:
                    _values.SetValue(val, idx);
                    break;
                case byte[] values:
                    if (indexes.Length == 0)
                        _values = values;
                    else
                        _values.SetValue(values, idx);
                    break;
                case int val:
                    _values.SetValue(val, idx);
                    break;
                case int[] values:
                    if (indexes.Length == 0)
                        _values = values;
                    else
                        _values.SetValue(values, idx);
                    break;
                case long val:
                    _values.SetValue(val, idx);
                    break;
                case long[] values:
                    if (indexes.Length == 0)
                        _values = values;
                    else
                        _values.SetValue(values, idx);
                    break;
                case float val:
                    _values.SetValue(val, idx);
                    break;
                case float[] values:
                    if (indexes.Length == 0)
                        _values = values;
                    else
                        _values.SetValue(values, idx);
                    break;
                case double val:
                    _values.SetValue(val, idx);
                    break;
                case double[] values:
                    if (indexes.Length == 0)
                        _values = values;
                    else
                        _values.SetValue(values, idx);
                    break;
                case NDArray nd:
                    switch(nd.dtype.Name)
                    {
                        case "Boolean":
                            _values.SetValue(nd.Data<bool>(0), idx);
                            break;
                        case "Int16":
                            _values.SetValue(nd.Data<short>(0), idx);
                            break;
                        case "Int32":
                            _values.SetValue(nd.Data<int>(0), idx);
                            break;
                        case "Int64":
                            _values.SetValue(nd.Data<long>(0), idx);
                            break;
                        case "Single":
                            _values.SetValue(nd.Data<float>(0), idx);
                            break;
                        case "Double":
                            _values.SetValue(nd.Data<double>(0), idx);
                            break;
                        case "Decimal":
                            _values.SetValue(nd.Data<decimal>(0), idx);
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
            _values = _ChangeTypeOfArray(values, typeof(T));
        }

        /// <summary>
        /// Set an Array to internal storage, cast it to new dtype and change dtype  
        /// </summary>
        /// <param name="values"></param>
        /// <param name="dtype"></param>
        public void SetData(Array values, Type dtype)
        {
            _values = _ChangeTypeOfArray(values, dtype);
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
            puffer.SetData((Array)_values.Clone());

            return puffer;
        }
    }
}
