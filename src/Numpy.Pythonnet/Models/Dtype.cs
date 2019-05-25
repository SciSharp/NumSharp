using System;
using System.Collections.Generic;
using System.Text;
using Python.Runtime;

namespace Numpy
{

    public partial class Dtype : PythonObject
    {
        public Dtype(PyObject pyobj) : base(pyobj)
        {
        }

        public Dtype(Dtype t) : base((PyObject)t.PyObject)
        {
        }

    }


    public static class DtypeExtensions
    {
        public static Dtype GetDtype(this object obj)
        {
            switch (obj)
            {
                case bool o: return np.bool8;
                case sbyte o: return np.int8;
                case byte o: return np.uint8;
                case short o: return np.int16;
                case ushort o: return np.uint16;
                case int o: return np.int32;
                case uint o: return np.uint32;
                case long o: return np.int64;
                case ulong o: return np.uint64;
                case float o: return np.float32;
                case double o: return np.float64;
                case bool[] o: return np.bool8;
                case byte[] o: return np.@byte;
                case short[] o: return np.int16;
                case int[] o: return np.int32;
                case long[] o: return np.int64;
                case float[] o: return np.float32;
                case double[] o: return np.float64;
                case bool[,] o: return np.bool8;
                case byte[,] o: return np.uint8;
                case short[,] o: return np.int16;
                case int[,] o: return np.int32;
                case long[,] o: return np.int64;
                case float[,] o: return np.float32;
                case double[,] o: return np.float64;
                case bool[,,] o: return np.bool8;
                case byte[,,] o: return np.uint8;
                case short[,,] o: return np.int16;
                case int[,,] o: return np.int32;
                case long[,,] o: return np.int64;
                case float[,,] o: return np.float32;
                case double[,,] o: return np.float64;
                default: throw new ArgumentException("Can not convert type of given object to dtype: " + obj.GetType());
            }
        }

        //public static dtype ToDtype(this Type t)
        //{
        //    if (t == typeof(byte)) return dtype.UInt8;
        //    if (t == typeof(short)) return dtype.Int16;
        //    if (t == typeof(int)) return dtype.Int32;
        //    if (t == typeof(long)) return dtype.Int64;
        //    if (t == typeof(float)) return dtype.Float32;
        //    if (t == typeof(double)) return dtype.Float64;
        //    throw new ArgumentException("Can not convert given type to dtype: " + t);
        //}
    }
}
