using NumSharp;
using NumSharp.Backends;
using System;

namespace NumSharp
{
    /// <summary>
    /// An abstract interface as design basis for the true Storage
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
    public interface IStorage : ICloneable
    {
        /// <summary>
        /// Data Type of stored elements
        /// </summary>
        /// <value>numpys equal dtype</value>
        Type DType { get; }

        int DTypeSize { get; }

        /// <summary>
        /// storage shape for outside representation
        /// </summary>
        /// <value>numpys equal shape</value>
        Shape Shape {get;}

        /// <summary>
        /// Allocate memory by dtype, shape, tensororder (default column wise)
        /// </summary>
        /// <param name="dtype">storage data type</param>
        /// <param name="shape">storage data shape</param>
        void Allocate(Type dtype, Shape shape);
        /// <summary>
        /// Allocate memory by Array and tensororder and deduce shape and dtype (default column wise)
        /// </summary>
        /// <param name="values">elements to store</param>
        void Allocate(Array values);
        /// <summary>
        /// Get Back Storage with Columnwise tensor Layout
        /// By this method the layout is changed if layout is not columnwise
        /// </summary>
        /// <returns>reference to storage (transformed or not)</returns>
        //IStorage GetColumWiseStorage();
        /// <summary>
        /// Get Back Storage with row wise tensor Layout
        /// By this method the layout is changed if layout is not row wise
        /// </summary>
        /// <returns>reference to storage (transformed or not)</returns>
        //IStorage GetRowWiseStorage();
        /// <summary>
        /// Get reference to internal data storage
        /// </summary>
        /// <returns>reference to internal storage as System.Array</returns>
        Array GetData();
        /// <summary>
        /// Clone internal storage and get reference to it
        /// </summary>
        /// <returns>reference to cloned storage as System.Array</returns>
        Array CloneData();
        /// <summary>
        /// Get reference to internal data storage and cast elements to new dtype
        /// </summary>
        /// <param name="dtype">new storage data type</param>
        /// <returns>reference to internal (casted) storage as System.Array </returns>
        Array GetData(Type dtype);
        /// <summary>
        /// Clone internal storage and cast elements to new dtype
        /// </summary>
        /// <param name="dtype">cloned storage data type</param>
        /// <returns>reference to cloned storage as System.Array</returns>
        Array CloneData(Type dtype);
        /// <summary>
        /// Get reference to internal data storage and cast elements to new dtype
        /// </summary>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>reference to internal (casted) storage as T[]</returns>
        T[] GetData<T>();
        /// <summary>
        /// Get all elements from cloned storage as T[] and cast dtype
        /// </summary>
        /// <typeparam name="T">cloned storgae dtype</typeparam>
        /// <returns>reference to cloned storage as T[]</returns>
        T[] CloneData<T>();
        /// <summary>
        /// Get single value from internal storage and do not cast dtype
        /// </summary>
        /// <param name="indexes">indexes</param>
        /// <returns>element from internal storage</returns>
        object GetData(params int[] indexes);
        /// <summary>
        /// Get single value from internal storage as type T and cast dtype to T
        /// </summary>
        /// <param name="indexes">indexes</param>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>element from internal storage</returns>
        T GetData<T>(params int[] indexes);
        /// <summary>
        /// Set an array to internal storage and keep dtype
        /// </summary>
        /// <param name="values"></param>
        void SetData(Array values);

        void SetData<T>(Array values);

        // void SetData<T>(T value, int offset);

        /// <summary>
        /// Set 1 single value to internal storage and keep dtype
        /// </summary>
        /// <param name="value"></param>
        /// <param name="indexes"></param>
        void SetData(object value, params int[] indexes);
        /// <summary>
        /// Set an Array to internal storage, cast it to new dtype and change dtype  
        /// </summary>
        /// <param name="values"></param>
        /// <param name="dtype"></param>
        void SetData(Array values, Type dtype);
        /// <summary>
        /// Change dtype of elements
        /// </summary>
        /// <param name="dtype">new storage data type</param>
        /// <returns>sucess or not</returns>
        void ChangeDataType(Type dtype);   

        void Reshape(params int[] dimensions);
    }
}