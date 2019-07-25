using NumSharp;
using NumSharp.Backends;
using System;
using System.Numerics;
using System.Runtime.InteropServices;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    /// <summary>
    ///     Serves as a typed storage for an array.
    /// </summary>
    /// <remarks>
    ///     Responsible for :<br></br>
    ///      - store data type, elements, Shape<br></br>
    ///      - offers methods for accessing elements depending on shape<br></br>
    ///      - offers methods for casting elements<br></br>
    ///      - offers methods for change tensor order<br></br>
    ///      - GetData always return reference object to the true storage<br></br>
    ///      - GetData{T} and SetData{T} change dtype and cast storage<br></br>
    ///      - CloneData always create a clone of storage and return this as reference object<br></br>
    ///      - CloneData{T} clone storage and cast this clone <br></br>
    /// </remarks>
    public interface IStorage : ICloneable
    {
        /// <summary>
        ///     The data type of internal storage array.
        /// </summary>
        /// <value>numpys equal dtype</value>
        /// <remarks>Has to be compliant with <see cref="NPTypeCode"/>.</remarks>
        Type DType { get; }

        /// <summary>
        ///     The <see cref="NPTypeCode"/> of <see cref="DType"/>.
        /// </summary>
        NPTypeCode TypeCode { get; }

        /// <summary>
        ///     The size in bytes of a single value of <see cref="DType"/>
        /// </summary>
        /// <remarks>Computed by <see cref="Marshal.SizeOf(object)"/></remarks>
        int DTypeSize { get; }

        /// <summary>
        /// storage shape for outside representation
        /// </summary>
        /// <value>Numpy's equivalent to np.shape</value>
        Shape Shape { get; }

        /// <summary>
        ///     The engine that was used to create this <see cref="IStorage"/>.
        /// </summary>
        TensorEngine Engine { get; }

        /// <summary>
        ///     Allocates a new <see cref="Array"/> into memory.
        /// </summary>
        /// <param name="dtype">The type of the Array, if null <see cref="DType"/> is used.</param>
        /// <param name="shape">The shape of the array.</param>
        void Allocate(Shape shape, Type dtype = null);

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        void Allocate(Array values);

        /// <summary>
        ///     Assign this <see cref="ArraySlice{T}"/> as the internal array storage and assign <see cref="shape"/> to it.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <param name="shape">The shape of the array.</param>
        /// <param name="copy">Should perform a copy of <paramref name="values"/></param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        void Allocate<T>(ArraySlice<T> values, Shape shape, bool copy = false) where T : unmanaged;

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <param name="shape">The shape of the array.</param>
        /// <param name="copy">Should perform a copy of <paramref name="values"/></param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        void Allocate(IArraySlice values, Shape shape, bool copy = false);

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        /// <param name="shape">The shape of given array</param>
        void Allocate(Array values, Shape shape);

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        void Allocate<T>(T[] values) where T : unmanaged;

        /// <summary>
        ///     Get reference to internal data storage
        /// </summary>
        /// <returns>reference to internal storage as System.Array</returns>
        IArraySlice GetData();

        /// <summary>
        ///     Clone internal storage and returns reference to it
        /// </summary>
        /// <returns>reference to cloned storage as System.Array</returns>
        IArraySlice CloneData();

        /// <summary>
        ///     Get reference to internal data storage and if necessary: copy and cast elements to new dtype of <typeparamref name="T"/>.
        /// </summary>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>reference to internal (casted) storage as T[]</returns>
        ArraySlice<T> GetData<T>() where T : unmanaged;

        /// <summary>
        ///     Get all elements from cloned storage as T[] and cast dtype
        /// </summary>
        /// <typeparam name="T">cloned storgae dtype</typeparam>
        /// <returns>reference to cloned storage as T[]</returns>
        ArraySlice<T> CloneData<T>() where T : unmanaged;

        /// <summary>
        ///     Get single value from internal storage as type T and cast dtype to T
        /// </summary>
        /// <param name="indices">indices</param>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>element from internal storage</returns>
        /// <exception cref="NullReferenceException">When <typeparamref name="T"/> does not equal to <see cref="DType"/></exception>
        T GetData<T>(params int[] indices) where T : unmanaged;

        /// <summary>
        ///     Set a single value at given <see cref="indices"/>.
        /// </summary>
        /// <param name="value">The value to set</param>
        /// <param name="indices">The </param>
        /// <remarks>
        ///     Does not change internal storage data type.<br></br>
        ///     If <paramref name="value"/> does not match <see cref="DType"/>, <paramref name="value"/> will be converted.
        /// </remarks>
        unsafe void SetData<T>(T value, params int[] indices) where T : unmanaged;

        /// <summary>
        ///     Set a <see cref="IArraySlice"/> at given <see cref="indices"/>.
        /// </summary>
        /// <param name="value">The value to set</param>
        /// <param name="indices">The </param>
        /// <remarks>
        ///     Does not change internal storage data type.<br></br>
        ///     If <paramref name="value"/> does not match <see cref="DType"/>, <paramref name="value"/> will be converted.
        /// </remarks>
        void SetData(IArraySlice value, params int[] indices);

        /// <summary>
        ///     Set a <see cref="NDArray"/> at given <see cref="indices"/>.
        /// </summary>
        /// <param name="value">The value to set</param>
        /// <param name="indices">The </param>
        /// <remarks>
        ///     Does not change internal storage data type.<br></br>
        ///     If <paramref name="value"/> does not match <see cref="DType"/>, <paramref name="value"/> will be converted.
        /// </remarks>
        void SetData(NDArray value, params int[] indices);

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data source and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <remarks>Does not copy values and doesn't change shape.</remarks>
        void ReplaceData(Array values);

        /// <summary>
        ///     Sets <see cref="values"/> as the internal data source and changes the internal storage data type to <see cref="values"/> type.
        /// </summary>
        /// <param name="values"></param>
        /// <remarks>Does not copy values and doesn't change shape.</remarks>
        void ReplaceData(IArraySlice values);

        void ReplaceData(IArraySlice values, Type dtype);

        /// <summary>
        ///     Set an Array to internal storage, cast it to new dtype and if necessary change dtype  
        /// </summary>
        /// <param name="values"></param>
        /// <param name="dtype"></param>
        /// <remarks>Does not copy values unless cast is necessary and doesn't change shape.</remarks>
        void ReplaceData(Array values, Type dtype);

        /// <summary>
        ///     Set an Array to internal storage, cast it to new dtype and if necessary change dtype  
        /// </summary>
        /// <param name="values"></param>
        /// <param name="typeCode"></param>
        /// <remarks>Does not copy values unless cast is necessary and doesn't change shape.</remarks>
        void ReplaceData(Array values, NPTypeCode typeCode);

        /// <summary>
        ///     Sets <see cref="nd"/> as the internal data storage and changes the internal storage data type to <see cref="nd"/> type.
        /// </summary>
        /// <param name="nd"></param>
        /// <remarks>Does not copy values and does change shape and dtype.</remarks>
        void ReplaceData(NDArray nd);

        /// <summary>
        ///     Reshapes current internal storage.
        /// </summary>
        /// <param name="dimensions"></param>
        void Reshape(params int[] dimensions);

        #region Get Methods

#if _REGEN
        %foreach supported_currently_supported,supported_currently_supported_lowercase%

        /// <summary>
        ///     Retrieves value of type <see cref="#2"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="#2"/></exception>
        @#1 Get#1(params int[] indices);
        %
#else
        /// <summary>
        ///     Retrieves value of type <see cref="byte"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="byte"/></exception>
        @Byte GetByte(params int[] indices);

        /// <summary>
        ///     Retrieves value of type <see cref="short"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="short"/></exception>
        @Int16 GetInt16(params int[] indices);

        /// <summary>
        ///     Retrieves value of type <see cref="ushort"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ushort"/></exception>
        @UInt16 GetUInt16(params int[] indices);

        /// <summary>
        ///     Retrieves value of type <see cref="int"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="int"/></exception>
        @Int32 GetInt32(params int[] indices);

        /// <summary>
        ///     Retrieves value of type <see cref="uint"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="uint"/></exception>
        @UInt32 GetUInt32(params int[] indices);

        /// <summary>
        ///     Retrieves value of type <see cref="long"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="long"/></exception>
        @Int64 GetInt64(params int[] indices);

        /// <summary>
        ///     Retrieves value of type <see cref="ulong"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="ulong"/></exception>
        @UInt64 GetUInt64(params int[] indices);

        /// <summary>
        ///     Retrieves value of type <see cref="char"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="char"/></exception>
        @Char GetChar(params int[] indices);

        /// <summary>
        ///     Retrieves value of type <see cref="double"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="double"/></exception>
        @Double GetDouble(params int[] indices);

        /// <summary>
        ///     Retrieves value of type <see cref="float"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="float"/></exception>
        @Single GetSingle(params int[] indices);

        /// <summary>
        ///     Retrieves value of type <see cref="decimal"/> from internal storage..
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="decimal"/></exception>
        @Decimal GetDecimal(params int[] indices);
#endif

        /// <summary>
        ///     Retrieves value of unspecified type (will figure using <see cref="IStorage.DType"/>).
        /// </summary>
        /// <param name="indices">The shape's indices to get.</param>
        /// <returns></returns>
        /// <exception cref="NullReferenceException">When <see cref="IStorage.DType"/> is not <see cref="object"/></exception>
        object GetValue(params int[] indices);

        #endregion
    }
}
