using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Numpy;
using Numpy.Models;
using Python.Runtime;

namespace Numpy
{
    /// <summary>
    /// Manual type conversions
    /// </summary>
    public partial class NumPy
    {

        public NDarray array(NDarray @object, Dtype dtype = null, bool? copy = null, string order = null, bool? subok = null, int? ndmin = null)
        {
            var args = ToTuple(new object[]
            {
                @object,
            });
            var kwargs = new PyDict();
            if (dtype != null) kwargs["dtype"] = ToPython(dtype);
            if (copy != null) kwargs["copy"] = ToPython(copy);
            if (order != null) kwargs["order"] = ToPython(order);
            if (subok != null) kwargs["subok"] = ToPython(subok);
            if (ndmin != null) kwargs["ndmin"] = ToPython(ndmin);
            dynamic py = self.InvokeMethod("array", args, kwargs);
            return ToCsharp<NDarray>(py);
        }

        public NDarray<T> array<T>(T[] @object, Dtype dtype = null, bool? copy = null, string order = null, bool? subok = null, int? ndmin = null)
        {
            var type = @object.GetDtype();
            //if (dtype != null && !type.Equals( dtype))
            //    throw new NotImplementedException("Type of the array is different from specified dtype. Data conversion is not supported (yet)");
            var ndarray = this.empty(new Shape(@object.Length), dtype: dtype??type, order:order); // todo: check out the other parameters
            long ptr = ndarray.PyObject.ctypes.data;
            switch ((object)@object)
            {
                case byte[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case short[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case int[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case long[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case float[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case double[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case bool[] a:
                    var bytes = a.Select(x => (byte)(x ? 1 : 0)).ToArray();
                    Marshal.Copy(bytes, 0, new IntPtr(ptr), a.Length);
                    break;
            }
            return new NDarray<T>(ndarray);
        }

        public NDarray<T> array<T>(T[,] @object, Dtype dtype = null, bool? copy = null, string order = null, bool? subok = null, int? ndmin = null)
        {
            var d1_array = @object.Cast<T>().ToArray();
            var type = d1_array.GetDtype();
            //if (dtype != null && type != dtype)
            //    throw new NotImplementedException("Type of the array is different from specified dtype. Data conversion is not supported (yet)");
            var ndarray = this.empty(new Shape(@object.GetLength(0), @object.GetLength(1)), dtype: dtype??type, order: order); // todo: check out the other parameters
            long ptr = ndarray.PyObject.ctypes.data;
            switch ((object)d1_array)
            {
                case byte[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case short[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case int[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case long[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case float[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case double[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case bool[] a:
                    var bytes = a.Select(x => (byte) (x ? 1 : 0)).ToArray();
                    Marshal.Copy(bytes, 0, new IntPtr(ptr), a.Length);
                    break;
            }
            return new NDarray<T>(ndarray);
        }

        public NDarray<T> array<T>(T[,,] @object, Dtype dtype = null, bool? copy = null, string order = null, bool? subok = null, int? ndmin = null)
        {
            var d1_array = @object.Cast<T>().ToArray();
            var type = d1_array.GetDtype();
            //if (dtype != null && type != dtype)
            //    throw new NotImplementedException("Type of the array is different from specified dtype. Data conversion is not supported (yet)");
            var ndarray = this.empty(new Shape(@object.GetLength(0), @object.GetLength(1), @object.GetLength(2)), dtype: dtype ?? type, order: order); // todo: check out the other parameters
            long ptr = ndarray.PyObject.ctypes.data;
            switch ((object)d1_array)
            {
                case byte[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case short[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case int[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case long[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case float[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case double[] a: Marshal.Copy(a, 0, new IntPtr(ptr), a.Length); break;
                case bool[] a:
                    var bytes = a.Select(x => (byte)(x ? 1 : 0)).ToArray();
                    Marshal.Copy(bytes, 0, new IntPtr(ptr), a.Length);
                    break;
            }
            return new NDarray<T>(ndarray);
        }

        public NDarray array(string[] obj, int? itemsize = null, bool? copy = null, bool? unicode = null, string order = null)
        {
            var args = ToTuple(obj);
            var kwargs = new PyDict();
            if (itemsize != null) kwargs["itemsize"] = ToPython(itemsize);
            if (copy != null) kwargs["copy"] = ToPython(copy);
            if (unicode != null) kwargs["unicode"] = ToPython(unicode);
            if (order != null) kwargs["order"] = ToPython(order);
            dynamic py = self.InvokeMethod("array", args, kwargs);
            return ToCsharp<NDarray>(py);
        }

        /// <summary>
        /// Create a 0d array from scalar
        /// </summary>
        /// <param name="scalar"></param>
        /// <param name="dtype"></param>
        /// <returns></returns>
        public NDarray asarray(ValueType scalar, Dtype dtype = null)
        {
            var __self__ = self;
            var pyargs = ToTuple(new object[]
            {
                scalar,
            });
            var kwargs = new PyDict();
            if (dtype != null) kwargs["dtype"] = ToPython(dtype);
            dynamic py = __self__.InvokeMethod("asarray", pyargs, kwargs);
            return ToCsharp<NDarray>(py);
        }

        /// <summary>
        /// Convert an array of size 1 to its scalar equivalent.
        /// </summary>
        /// <returns>
        /// Scalar representation of a. The output data type is the same type
        /// returned by the input’s item method.
        /// </returns>
        public T asscalar<T>(NDarray a) => self.InvokeMethod("asscalar", a.PyObject).As<T>();
       

    }
}
