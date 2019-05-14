
#if NOT_USED


    public static class ConverterExtensions
    {
        public static NDArray<Int64> NDArray<T>(this NDArray<Int32> nd)
        {
            return null;
        }
    }
// Untyped NDArray slice = slice not implemented


using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NumSharp.Backends
{
    public static class StorageSetup
    {
        public static Type[] KnownIConvertibleStorageElementTypes = new[] {
            typeof(Byte),
            typeof(SByte),
            typeof(UInt32),
            typeof(Int32),
            typeof(UInt16),
            typeof(Int16),
            typeof(Single),
            typeof(Double),
            typeof(Decimal),
            typeof(Char),
            typeof(String),

            // Types not supporting IConvertible
            //Complex
            //NDArray
            //Object
        };
    }

    public abstract class ArrayTypeConverter
    {
        static ArrayTypeConverter()
        {
            foreach (var eType in StorageSetup.KnownIConvertibleStorageElementTypes)
            {
                factoryDict[eType] = CreateConverter(eType, eType);
            }
            // factoryDict[typeof(System.Numerics.Complex] = _ => new ComplexArrayTypeConverter()
        }

        public static Dictionary<Type, Func<ArrayTypeConverter>> factoryDict = new Dictionary<Type, Func<ArrayTypeConverter>>();
        //public abstract Array ToInt64Array(Array aIn);

        public static ArrayTypeConverter CreateInstance(Type inType, Type outType)
        {
            return factoryDict[inType].Invoke();
        }

        private static Func<ArrayTypeConverter> CreateConverter(Type inType, Type outType)
        {
            // This function should only be called during class initialization because it is slow.
            //Crate a Dummy object
            Type constructedType = typeof(ArrayTypeConverter<,>).MakeGenericType(new Type[] { inType, outType });
            var constructorParameters = new object[] { };
            var o = Activator.CreateInstance(constructedType, constructorParameters);
            var g = o as ArrayTypeConverter;
            // Get the Factory Func    
            return g;
        }

    }


    public class ArrayTypeConverter<TIn, TOut> : ArrayTypeConverter
        where TIn : IComparable, IComparable<int>, IEquatable<int>, IFormattable, IConvertible
        ///where TOut : IComparable, IComparable<int>, IEquatable<int>, IFormattable, IConvertible
    {

        public Func<Array, Array> GetConverter(Type tIn, Type tOut)
        {
            switch (Type.GetTypeCode(tOut))
            {
                case TypeCode.Int64:
                    return ToInt64Array;
            }
            ...
            throw new ArgumentException("Type to of wrong type", nameof(tOut));
        }

        private Array ToInt64Array(Array arrayToConvert)
        {
            if (arrayToConvert is TIn[] aIn)
            {
                var arrayLength = arrayToConvert.Length;
                var aOut = new Int64[arrayLength];
                for (var idx = 0; idx < arrayLength; idx++)
                {
                    aOut[idx] = Convert.ToInt64(aIn[idx]);
                }
                return aOut;
            }
            throw new ArgumentException("Array to of wrong type", nameof(arrayToConvert));
        }
    }

    public abstract class GenericArraySliceStorage 
    {
        //private IStorage Storage;

        public static Dictionary<Type, Func<Slice[], GenericArraySliceStorage>> factoryDict = new Dictionary<Type, Func<Slice[], GenericArraySliceStorage>>();
        public abstract Func<Slice[], GenericArraySliceStorage> CreateFactory();



        protected GenericArraySliceStorage()
        {
            //Storage
        }

        static GenericArraySliceStorage()
        {
            foreach (var eType in StorageSetup.KnownIConvertibleStorageElementTypes)
            {
                factoryDict[eType] = CreateInstanceFactory(eType);
            }
            // factoryDict[typeof(System.Numerics.Complex] = _ => new ComplexArraySliceStorage()
        }

        public static GenericArraySliceStorage CreateInstance(Type elementType, Slice[] slices)
        {
            return factoryDict[elementType].Invoke(slices);
        }

        private static Func<Slice[], GenericArraySliceStorage> CreateInstanceFactory(Type elemenType)
        {
            // This function should only be called during class initialization because it is slow.
            //Crate a Dummy object
            Type constructedType = typeof(GenericArraySliceStorage<>).MakeGenericType(new Type[] { elemenType });
            var constructorParameters = new object[] { };
            var o = Activator.CreateInstance(constructedType, constructorParameters);
            var g = o as GenericArraySliceStorage;
            // Get the Factory Func    
            return g.CreateFactory();
        }
    }

    public class GenericArraySliceStorage<T> : GenericArraySliceStorage, IStorage where T : IComparable, IComparable<int>, IEquatable<int>, IFormattable, IConvertible
    {
        public override Func<Slice[], GenericArraySliceStorage> CreateFactory()
        {
            return slices => new GenericArraySliceStorage<T>(slices);
        }

        public GenericArraySliceStorage(Slice[] slices)
        {

        }

        private T[] values;

        /// <summary>
        /// Data Type of stored elements
        /// </summary>
        /// <value>numpys equal dtype</value>
        public Type DType => typeof(T);

        public int DTypeSize => Marshal.SizeOf(default(T));

        Slice Slice { get; set; }

        /// <summary>
        /// storage shape for outside representation
        /// </summary>
        /// <value>numpys equal shape</value>
        public Shape Shape { get; private set;  }

        /// <summary>
        /// Allocate memory by dtype, shape, tensororder (default column wise)
        /// </summary>
        /// <param name="dtype">storage data type</param>
        /// <param name="shape">storage data shape</param>
        public void Allocate(Shape shape, Type dtype = null)
        {
            if (dtype != null && typeof(T) != dtype)
            {
                throw new ArgumentException("You can not allocate storage of different type after initialization", nameof(dtype));
            }
            Shape = shape;
            values = new T[shape.Size];
        }
        /// <summary>
        /// Allocate memory by Array and tensororder and deduce shape and dtype (default column wise)
        /// </summary>
        /// <param name="sampleArray">elements to store</param>
        public void Allocate(Array sampleArray)
        {
            int[] dim = new int[sampleArray.Rank];
            for (int idx = 0; idx < dim.Length; idx++)
                dim[idx] = sampleArray.GetLength(idx);

            Shape = new Shape(dim);
        }

        /// <summary>
        /// Get reference to internal data storage
        /// </summary>
        /// <returns>reference to internal storage as System.Array</returns>
        Array GetData() => values;
        /// <summary>
        /// Clone internal storage and get reference to it
        /// </summary>
        /// <returns>reference to cloned storage as System.Array</returns>
        Array CloneData()
        {
            return values.Clone() as Array;
        }

        /// <summary>
        /// Get reference to internal data storage and cast elements to new dtype
        /// </summary>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>reference to internal (casted) storage as T[]</returns>
        TResult[] GetData<TResult>()
        {

            var arrayLength = Shape.Size;
            var result = new TResult[arrayLength];
            for (var idx = 0; idx < arrayLength; idx++)
            {
                result[idx] = Convert.
            }
        }

        /// <summary>
        /// Get single value from internal storage as type T and cast dtype to T
        /// </summary>
        /// <param name="indice">indexes</param>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>element from internal storage</returns>
        TResult GetData<TResult>(params int[] indice);

        public bool SupportsSpan => false;
        Span<TResult> GetSpanData<TResult>(Slice slice, params int[] indice)// where TResult = T;
        {
            throw new NotSupportedException("GetSpanData is not supported");
        }

        public bool GetBoolean(params int[] indice)
        {
            if (typeof(T) == typeof(bool))
            {

            } else {
                throw new NotSupportedException("GetBoolean only supported on NDArray<boolean>")
            }
        };
        short GetInt16(params int[] indice);
        int GetInt32(params int[] indice);
        long GetInt64(params int[] indice);
        float GetSingle(params int[] indice);
        double GetDouble(params int[] indice);
        decimal GetDecimal(params int[] indice);
        string GetString(params int[] indice);
        NDArray GetNDArray(params int[] indice);

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
        void SetData<T>(T value, params int[] indexes);

        /// <summary>
        /// Set an Array to internal storage, cast it to new dtype and change dtype  
        /// </summary>
        /// <param name="values"></param>
        /// <param name="dtype"></param>
        void SetData(Array values, Type dtype);

        void Reshape(params int[] dimensions);

        Span<T> View<T>(Slice slice = null);
    }

}
#endif
