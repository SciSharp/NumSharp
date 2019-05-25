using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Numpy.Models;
using Python.Runtime;


namespace Numpy
{
    public partial class NDarray : PythonObject
    {
        // these are manual overrides of functions or properties that can not be automatically generated

        public NDarray(PyObject pyobj) : base(pyobj)
        {
        }

        public NDarray(NDarray t) : base((PyObject) t.PyObject)
        {
        }

        /// <summary>
        /// Returns a copy of the array data
        /// </summary>
        public T[] GetData<T>()
        {
            // note: this implementation works only for device CPU
            long ptr = PyObject.ctypes.data;
            int size = PyObject.size;
            object array = null;
            if (typeof(T) == typeof(byte)) array = new byte[size];
            else if (typeof(T) == typeof(short)) array = new short[size];
            else if (typeof(T) == typeof(int)) array = new int[size];
            else if (typeof(T) == typeof(long)) array = new long[size];
            else if (typeof(T) == typeof(float)) array = new float[size];
            else if (typeof(T) == typeof(double)) array = new double[size];
            else
                throw new InvalidOperationException(
                    "Can not copy the data with data type due to limitations of Marshal.Copy: " + typeof(T).Name);
            switch (array)
            {
                case byte[] a:
                    Marshal.Copy(new IntPtr(ptr), a, 0, a.Length);
                    break;
                case short[] a:
                    Marshal.Copy(new IntPtr(ptr), a, 0, a.Length);
                    break;
                case int[] a:
                    Marshal.Copy(new IntPtr(ptr), a, 0, a.Length);
                    break;
                case long[] a:
                    Marshal.Copy(new IntPtr(ptr), a, 0, a.Length);
                    break;
                case float[] a:
                    Marshal.Copy(new IntPtr(ptr), a, 0, a.Length);
                    break;
                case double[] a:
                    Marshal.Copy(new IntPtr(ptr), a, 0, a.Length);
                    break;
            }

            return (T[]) array;
        }

        /// <summary>
        /// Information about the memory layout of the array.
        /// </summary>
        public Flags flags => new Flags(self.GetAttr("flags")); // TODO: implement Flags

        /// <summary>
        /// Tuple of array dimensions.
        /// </summary>
        public Shape shape => new Shape( self.GetAttr("shape").As<int[]>());

        /// <summary>
        /// Tuple of bytes to step in each dimension when traversing an array.
        /// </summary>
        public int[] strides => self.GetAttr("strides").As<int[]>();

        /// <summary>
        /// Number of array dimensions.
        /// </summary>
        public int ndim => self.GetAttr("ndim").As<int>();

        /// <summary>
        /// Python buffer object pointing to the start of the array’s data.
        /// </summary>
        public PyObject data => self.GetAttr("data");

        /// <summary>
        /// Number of elements in the array.
        /// </summary>
        public int size => self.GetAttr("size").As<int>();

        /// <summary>
        /// Length of one array element in bytes.
        /// </summary>
        public int itemsize => self.GetAttr("itemsize").As<int>();

        /// <summary>
        /// Total bytes consumed by the elements of the array.
        /// </summary>
        public int nbytes => self.GetAttr("nbytes").As<int>();

        /// <summary>
        /// Base object if memory is from some other object.
        /// </summary>
        public NDarray @base
        {
            get
            {
                PyObject base_obj = self.GetAttr("base");
                if (base_obj.IsNone())
                    return null;
                return new NDarray(base_obj);
            }
        }

        /// <summary>
        /// Data-type of the array’s elements.
        /// </summary>
        public Dtype dtype => new Dtype(self.GetAttr("dtype"));

        /// <summary>
        /// Same as self.transpose(), except that self is returned if self.ndim &lt; 2.
        /// </summary>
        public NDarray T => new NDarray(self.GetAttr("T"));

        ///// <summary>
        ///// The real part of the array.
        ///// </summary>
        //public NDarray real => new NDarray(self.GetAttr("real"));

        ///// <summary>
        ///// The imaginary part of the array.
        ///// </summary>
        //public NDarray imag => new NDarray(self.GetAttr("imag"));

        /// <summary>
        /// A 1-D iterator over the array.
        /// </summary>
        public PyObject flat => self.GetAttr("flat"); // todo: wrap and support usecases

        /// <summary>
        /// An object to simplify the interaction of the array with the ctypes module.
        /// </summary>
        public PyObject ctypes => self.GetAttr("ctypes"); // TODO: wrap ctypes


        /// <summary>
        /// Length of the array (same as size)
        /// </summary>
        public int len => self.InvokeMethod("__len__").As<int>();

        /// <summary>
        /// Insert scalar into an array (scalar is cast to array’s dtype, if possible)
        /// 
        /// There must be at least 1 argument, and define the last argument
        /// as item.  Then, a.itemset(*args) is equivalent to but faster
        /// than a[args] = item.  The item should be a scalar value and args
        /// must select a single item in the array a.
        /// 
        /// Notes
        /// 
        /// Compared to indexing syntax, itemset provides some speed increase
        /// for placing a scalar into a particular location in an ndarray,
        /// if you must do this.  However, generally this is discouraged:
        /// among other problems, it complicates the appearance of the code.
        /// Also, when using itemset (and item) inside a loop, be sure
        /// to assign the methods to a local variable to avoid the attribute
        /// look-up at each loop iteration.
        /// </summary>
        /// <param name="args">
        /// If one argument: a scalar, only used in case a is of size 1.
        /// If two arguments: the last argument is the value to be set
        /// and must be a scalar, the first argument specifies a single array
        /// element location. It is either an int or a tuple.
        /// </param>
        public void itemset(params object[] args)
        {
            var pyargs = ToTuple(args);
            var kwargs = new PyDict();
            dynamic py = self.InvokeMethod("itemset", pyargs, kwargs);
        }

        /// <summary>
        /// Construct Python bytes containing the raw data bytes in the array.
        /// 
        /// Constructs Python bytes showing a copy of the raw contents of
        /// data memory. The bytes object can be produced in either ‘C’ or ‘Fortran’,
        /// or ‘Any’ order (the default is ‘C’-order). ‘Any’ order means C-order
        /// unless the F_CONTIGUOUS flag in the array is set, in which case it
        /// means ‘Fortran’ order.
        /// 
        /// This function is a compatibility alias for tobytes. Despite its name it returns bytes not strings.
        /// </summary>
        /// <param name="order">
        /// Order of the data for multidimensional arrays:
        /// C, Fortran, or the same as for the original array.
        /// </param>
        /// <returns>
        /// Python bytes exhibiting a copy of a’s raw data.
        /// </returns>
        public byte[] tostring(string order = null)
        {
            return tobytes();
        }

        /// <summary>
        /// Construct Python bytes containing the raw data bytes in the array.
        /// 
        /// Constructs Python bytes showing a copy of the raw contents of
        /// data memory. The bytes object can be produced in either ‘C’ or ‘Fortran’,
        /// or ‘Any’ order (the default is ‘C’-order). ‘Any’ order means C-order
        /// unless the F_CONTIGUOUS flag in the array is set, in which case it
        /// means ‘Fortran’ order.
        /// </summary>
        /// <param name="order">
        /// Order of the data for multidimensional arrays:
        /// C, Fortran, or the same as for the original array.
        /// </param>
        /// <returns>
        /// Python bytes exhibiting a copy of a’s raw data.
        /// </returns>
        public byte[] tobytes(string order = null)
        {
            throw new NotImplementedException("TODO: this needs to be implemented with Marshal.Copy");
            var pyargs = ToTuple(new object[]
            {
            });
            var kwargs = new PyDict();
            if (order != null) kwargs["order"] = ToPython(order);
            dynamic py = self.InvokeMethod("tobytes", pyargs, kwargs);
            return ToCsharp<byte[]>(py);
        }

        /// <summary>
        /// New view of array with the same data.
        /// 
        /// Notes
        /// 
        /// a.view() is used two different ways:
        /// 
        /// a.view(some_dtype) or a.view(dtype=some_dtype) constructs a view
        /// of the array’s memory with a different data-type.  This can cause a
        /// reinterpretation of the bytes of memory.
        /// 
        /// a.view(ndarray_subclass) or a.view(type=ndarray_subclass) just
        /// returns an instance of ndarray_subclass that looks at the same array
        /// (same shape, dtype, etc.)  This does not cause a reinterpretation of the
        /// memory.
        /// 
        /// For a.view(some_dtype), if some_dtype has a different number of
        /// bytes per entry than the previous dtype (for example, converting a
        /// regular array to a structured array), then the behavior of the view
        /// cannot be predicted just from the superficial appearance of a (shown
        /// by print(a)). It also depends on exactly how a is stored in
        /// memory. Therefore if a is C-ordered versus fortran-ordered, versus
        /// defined as a slice or transpose, etc., the view may give different
        /// results.
        /// </summary>
        /// <param name="dtype">
        /// Data-type descriptor of the returned view, e.g., float32 or int16. The
        /// default, None, results in the view having the same data-type as a.
        /// This argument can also be specified as an ndarray sub-class, which
        /// then specifies the type of the returned object (this is equivalent to
        /// setting the type parameter).
        /// </param>
        /// <param name="type">
        /// Type of the returned view, e.g., ndarray or matrix.  Again, the
        /// default None results in type preservation.
        /// </param>
        public void view(Dtype dtype = null, Type type = null)
        {
            throw new NotImplementedException("Get python type 'ndarray' and 'matrix' and substitute them for the given .NET type");
            var pyargs = ToTuple(new object[]
            {
            });
            var kwargs = new PyDict();
            if (dtype != null) kwargs["dtype"] = ToPython(dtype);
            if (type != null) kwargs["type"] = ToPython(type);
            dynamic py = self.InvokeMethod("view", pyargs, kwargs);
        }

        /// <summary>
        /// Change shape and size of array in-place.
        /// 
        /// Notes
        /// 
        /// This reallocates space for the data area if necessary.
        /// 
        /// Only contiguous arrays (data elements consecutive in memory) can be
        /// resized.
        /// 
        /// The purpose of the reference count check is to make sure you
        /// do not use this array as a buffer for another Python object and then
        /// reallocate the memory. However, reference counts can increase in
        /// other ways so if you are sure that you have not shared the memory
        /// for this array with another Python object, then you may safely set
        /// refcheck to False.
        /// </summary>
        /// <param name="new_shape">
        /// Shape of resized array.
        /// </param>
        /// <param name="refcheck">
        /// If False, reference count will not be checked. Default is True.
        /// </param>
        public void resize(Shape new_shape, bool? refcheck = null)
        {
            var pyargs = ToTuple(new object[]
            {
                new_shape,
            });
            var kwargs = new PyDict();
            if (refcheck != null) kwargs["refcheck"] = ToPython(refcheck);
            dynamic py = self.InvokeMethod("resize", pyargs, kwargs);
        }

        /// <summary>
        /// Gives a new shape to an array without changing its data.
        /// 
        /// Notes
        /// 
        /// It is not always possible to change the shape of an array without
        /// copying the data. If you want an error to be raised when the data is copied,
        /// you should assign the new shape to the shape attribute of the array:
        /// 
        /// The order keyword gives the index ordering both for fetching the values
        /// from a, and then placing the values into the output array.
        /// For example, let’s say you have an array:
        /// 
        /// You can think of reshaping as first raveling the array (using the given
        /// index order), then inserting the elements from the raveled array into the
        /// new array using the same kind of index ordering as was used for the
        /// raveling.
        /// </summary>
        /// <param name="newshape">
        /// The new shape should be compatible with the original shape. If
        /// an integer, then the result will be a 1-D array of that length.
        /// One shape dimension can be -1. In this case, the value is
        /// inferred from the length of the array and remaining dimensions.
        /// </param>
        /// <returns>
        /// This will be a new view object if possible; otherwise, it will
        /// be a copy.  Note there is no guarantee of the memory layout (C- or
        /// Fortran- contiguous) of the returned array.
        /// </returns>
        public NDarray reshape(params int[] newshape)
        {
            //auto-generated code, do not change
            var @this = this;
            return NumPy.Instance.reshape(@this, new Shape(newshape));
        }

        /// <summary>
        /// returns the 'array([ .... ])'-representation known from the console
        /// </summary>
        public string repr => self.InvokeMethod("__repr__").As<string>();

        /// <summary>
        /// returns the '[ .... ]'-representation
        /// </summary>
        public string str => self.InvokeMethod("__str__").As<string>();

        public NDarray this[string slicing_notation]
        {
            get
            {                
                var tuple=new PyTuple(Slice.ParseSlices(slicing_notation).Select(s =>
                {
                    if (s.IsIndex)
                        return new PyInt(s.Start.Value);
                    else
                        return s.ToPython();
                }).ToArray());
                return new NDarray(this.PyObject[tuple]);
            }
        }

        public new NDarray this[params int[] coords]
        {
            get
            {
                var tuple = ToTuple(coords);
                return new NDarray(this.PyObject[tuple]);
            }
        }

        public NDarray this[params NDarray[] indices]
        {
            get
            {
                var tuple = new PyTuple(indices.Select(a => (PyObject)a.PyObject).ToArray());
                return new NDarray(this.PyObject[tuple]);
            }
        }

        public new NDarray this[params object[] arrays_slices_or_indices]
        {
            get
            {
                var pyobjs = arrays_slices_or_indices.Select<object ,PyObject>(x =>
                {
                    switch (x)
                    {
                        case int i: return new PyInt(i);
                        case NDarray a: return a.PyObject;
                        case string s: return new Slice(s).ToPython();
                        default: return ToPython(x);
                    }
                }).ToArray();
                var tuple = new PyTuple(pyobjs);
                return new NDarray(this.PyObject[tuple]);
            }
        }

        /// <summary>
        /// Convert an array of size 1 to its scalar equivalent.
        /// </summary>
        /// <returns>
        /// Scalar representation of a. The output data type is the same type
        /// returned by the input’s item method.
        /// </returns>
        public T asscalar<T>()
        {
            return NumPy.Instance.asscalar<T>(this);
        }
    }

    public class NDarray<T> : NDarray
    {
        public NDarray(NDarray t) : base(t)
        {
        }

        public NDarray(PyObject pyobject) : base(pyobject)
        {
        }

        /// <summary>
        /// Returns a copy of the array data
        /// </summary>
        public T[] GetData()
        {
            return base.GetData<T>();
        }

        public new NDarray<T> this[string slicing_notation]
        {
            get
            {
                var tuple = new PyTuple(Slice.ParseSlices(slicing_notation).Select(s =>
                {
                    if (s.IsIndex)
                        return new PyInt(s.Start.Value);
                    else
                        return s.ToPython();
                }).ToArray());
                return new NDarray<T>(this.PyObject[tuple]);
            }
        }

        public new NDarray this[params int[] coords]
        {
            get
            {
                var tuple = ToTuple(coords);
                return new NDarray<T>(this.PyObject[tuple]);
            }
        }

        public new NDarray this[params NDarray[] indices]
        {
            get
            {
                var tuple = new PyTuple(indices.Select(a => (PyObject)a.PyObject).ToArray());
                return new NDarray<T>(this.PyObject[tuple]);
            }
        }

        public new NDarray this[params object[] arrays_slices_or_indices]
        {
            get
            {
                var pyobjs = arrays_slices_or_indices.Select<object, PyObject>(x =>
                {
                    switch (x)
                    {
                        case int i: return new PyInt(i);
                        case NDarray a: return a.PyObject;
                        case string s: return new Slice(s).ToPython();
                        default: return ToPython(x);
                    }
                }).ToArray();
                var tuple = new PyTuple(pyobjs);
                return new NDarray(this.PyObject[tuple]);
            }
        }
    }
}
