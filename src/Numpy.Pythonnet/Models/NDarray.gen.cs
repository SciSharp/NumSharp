using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Python.Runtime;
using Python.Included;
using Numpy.Models;

namespace Numpy
{
    public partial class NDarray
    {
        
        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// 
        /// Notes
        /// 
        /// When the data type of a is longdouble or clongdouble, item() returns
        /// a scalar array object because there is no available Python scalar that
        /// would not lose information. Void arrays return a buffer object for item(),
        /// unless fields are defined, in which case a tuple is returned.
        /// 
        /// item is very similar to a[args], except, instead of an array scalar,
        /// a standard Python scalar is returned. This can be useful for speeding up
        /// access to elements of the array and doing arithmetic on elements of the
        /// array using Python’s optimized math.
        /// </summary>
        /// <returns>
        /// A copy of the specified element of the array as a suitable
        /// Python scalar
        /// </returns>
        public T item<T>(params int[] args)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
                args,
            });
            var kwargs=new PyDict();
            dynamic py = self.InvokeMethod("item", pyargs, kwargs);
            return ToCsharp<T>(py);
        }
        
        /// <summary>
        /// Return the array as a (possibly nested) list.
        /// 
        /// Return a copy of the array data as a (nested) Python list.
        /// Data items are converted to the nearest compatible Python type.
        /// 
        /// Notes
        /// 
        /// The array may be recreated, a = np.array(a.tolist()).
        /// </summary>
        /// <returns>
        /// The possibly nested list of array elements.
        /// </returns>
        public List<T> tolist<T>()
        {
            //auto-generated code, do not change
            dynamic py = self.InvokeMethod("tolist");
            return ToCsharp<List<T>>(py);
        }
        
        /// <summary>
        /// Write array to a file as text or binary (default).
        /// 
        /// Data is always written in ‘C’ order, independent of the order of a.
        /// The data produced by this method can be recovered using the function
        /// fromfile().
        /// 
        /// Notes
        /// 
        /// This is a convenience function for quick storage of array data.
        /// Information on endianness and precision is lost, so this method is not a
        /// good choice for files intended to archive data or transport data between
        /// machines with different endianness. Some of these problems can be overcome
        /// by outputting the data as text files, at the expense of speed and file
        /// size.
        /// 
        /// When fid is a file object, array contents are directly written to the
        /// file, bypassing the file object’s write method. As a result, tofile
        /// cannot be used with files objects supporting compression (e.g., GzipFile)
        /// or file-like objects that do not support fileno() (e.g., BytesIO).
        /// </summary>
        /// <param name="fid">
        /// An open file object, or a string containing a filename.
        /// </param>
        /// <param name="sep">
        /// Separator between array items for text output.
        /// If “” (empty), a binary file is written, equivalent to
        /// file.write(a.tobytes()).
        /// </param>
        /// <param name="format">
        /// Format string for text file output.
        /// Each entry in the array is formatted to text by first converting
        /// it to the closest Python type, and then using “format” % item.
        /// </param>
        public void tofile(string fid, string sep, string format)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
                fid,
                sep,
                format,
            });
            var kwargs=new PyDict();
            dynamic py = self.InvokeMethod("tofile", pyargs, kwargs);
        }
        
        /// <summary>
        /// Dump a pickle of the array to the specified file.
        /// The array can be read back with pickle.load or numpy.load.
        /// </summary>
        /// <param name="file">
        /// A string naming the dump file.
        /// </param>
        public void dump(string file)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
                file,
            });
            var kwargs=new PyDict();
            dynamic py = self.InvokeMethod("dump", pyargs, kwargs);
        }
        
        /// <summary>
        /// Returns the pickle of the array as a string.
        /// pickle.loads or numpy.loads will convert the string back to an array.
        /// </summary>
        public void dumps()
        {
            //auto-generated code, do not change
            dynamic py = self.InvokeMethod("dumps");
        }
        
        /// <summary>
        /// Copy of the array, cast to a specified type.
        /// 
        /// Notes
        /// 
        /// Starting in NumPy 1.9, astype method now returns an error if the string
        /// dtype to cast to is not long enough in ‘safe’ casting mode to hold the max
        /// value of integer/float array that is being casted. Previously the casting
        /// was allowed even if the result was truncated.
        /// </summary>
        /// <param name="dtype">
        /// Typecode or data-type to which the array is cast.
        /// </param>
        /// <param name="order">
        /// Controls the memory layout order of the result.
        /// ‘C’ means C order, ‘F’ means Fortran order, ‘A’
        /// means ‘F’ order if all the arrays are Fortran contiguous,
        /// ‘C’ order otherwise, and ‘K’ means as close to the
        /// order the array elements appear in memory as possible.
        /// Default is ‘K’.
        /// </param>
        /// <param name="casting">
        /// Controls what kind of data casting may occur. Defaults to ‘unsafe’
        /// for backwards compatibility.
        /// </param>
        /// <param name="subok">
        /// If True, then sub-classes will be passed-through (default), otherwise
        /// the returned array will be forced to be a base-class array.
        /// </param>
        /// <param name="copy">
        /// By default, astype always returns a newly allocated array. If this
        /// is set to false, and the dtype, order, and subok
        /// requirements are satisfied, the input array is returned instead
        /// of a copy.
        /// </param>
        /// <returns>
        /// Unless copy is False and the other conditions for returning the input
        /// array are satisfied (see description for copy input parameter), arr_t
        /// is a new array of the same shape as the input array, with dtype, order
        /// given by dtype, order.
        /// </returns>
        public NDarray astype(Dtype dtype, string order = null, string casting = null, bool? subok = null, bool? copy = null)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
                dtype,
            });
            var kwargs=new PyDict();
            if (order!=null) kwargs["order"]=ToPython(order);
            if (casting!=null) kwargs["casting"]=ToPython(casting);
            if (subok!=null) kwargs["subok"]=ToPython(subok);
            if (copy!=null) kwargs["copy"]=ToPython(copy);
            dynamic py = self.InvokeMethod("astype", pyargs, kwargs);
            return ToCsharp<NDarray>(py);
        }
        
        /// <summary>
        /// Swap the bytes of the array elements
        /// 
        /// Toggle between low-endian and big-endian data representation by
        /// returning a byteswapped array, optionally swapped in-place.
        /// </summary>
        /// <param name="inplace">
        /// If True, swap bytes in-place, default is False.
        /// </param>
        /// <returns>
        /// The byteswapped array. If inplace is True, this is
        /// a view to self.
        /// </returns>
        public NDarray byteswap(bool? inplace = null)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
            });
            var kwargs=new PyDict();
            if (inplace!=null) kwargs["inplace"]=ToPython(inplace);
            dynamic py = self.InvokeMethod("byteswap", pyargs, kwargs);
            return ToCsharp<NDarray>(py);
        }
        
        /// <summary>
        /// Return a copy of the array.
        /// </summary>
        /// <param name="order">
        /// Controls the memory layout of the copy. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        /// ‘C’ otherwise. ‘K’ means match the layout of a as closely
        /// as possible. (Note that this function and numpy.copy are very
        /// similar, but have different default values for their order=
        /// arguments.)
        /// </param>
        public void copy(string order = null)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
            });
            var kwargs=new PyDict();
            if (order!=null) kwargs["order"]=ToPython(order);
            dynamic py = self.InvokeMethod("copy", pyargs, kwargs);
        }
        
        /// <summary>
        /// Returns a field of the given array as a certain type.
        /// 
        /// A field is a view of the array data with a given data-type. The values in
        /// the view are determined by the given type and the offset into the current
        /// array in bytes. The offset needs to be such that the view dtype fits in the
        /// array dtype; for example an array of dtype complex128 has 16-byte elements.
        /// If taking a view with a 32-bit integer (4 bytes), the offset needs to be
        /// between 0 and 12 bytes.
        /// </summary>
        /// <param name="dtype">
        /// The data type of the view. The dtype size of the view can not be larger
        /// than that of the array itself.
        /// </param>
        /// <param name="offset">
        /// Number of bytes to skip before beginning the element view.
        /// </param>
        public void getfield(Dtype dtype, int offset)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
                dtype,
                offset,
            });
            var kwargs=new PyDict();
            dynamic py = self.InvokeMethod("getfield", pyargs, kwargs);
        }
        
        /// <summary>
        /// Set array flags WRITEABLE, ALIGNED, (WRITEBACKIFCOPY and UPDATEIFCOPY),
        /// respectively.
        /// 
        /// These Boolean-valued flags affect how numpy interprets the memory
        /// area used by a (see Notes below). The ALIGNED flag can only
        /// be set to True if the data is actually aligned according to the type.
        /// The WRITEBACKIFCOPY and (deprecated) UPDATEIFCOPY flags can never be set
        /// to True. The flag WRITEABLE can only be set to True if the array owns its
        /// own memory, or the ultimate owner of the memory exposes a writeable buffer
        /// interface, or is a string. (The exception for string is made so that
        /// unpickling can be done without copying memory.)
        /// 
        /// Notes
        /// 
        /// Array flags provide information about how the memory area used
        /// for the array is to be interpreted. There are 7 Boolean flags
        /// in use, only four of which can be changed by the user:
        /// WRITEBACKIFCOPY, UPDATEIFCOPY, WRITEABLE, and ALIGNED.
        /// 
        /// WRITEABLE (W) the data area can be written to;
        /// 
        /// ALIGNED (A) the data and strides are aligned appropriately for the hardware
        /// (as determined by the compiler);
        /// 
        /// UPDATEIFCOPY (U) (deprecated), replaced by WRITEBACKIFCOPY;
        /// 
        /// WRITEBACKIFCOPY (X) this array is a copy of some other array (referenced
        /// by .base). When the C-API function PyArray_ResolveWritebackIfCopy is
        /// called, the base array will be updated with the contents of this array.
        /// 
        /// All flags can be accessed using the single (upper case) letter as well
        /// as the full name.
        /// </summary>
        /// <param name="write">
        /// Describes whether or not a can be written to.
        /// </param>
        /// <param name="align">
        /// Describes whether or not a is aligned properly for its type.
        /// </param>
        /// <param name="uic">
        /// Describes whether or not a is a copy of another “base” array.
        /// </param>
        public void setflags(bool? write = null, bool? align = null, bool? uic = null)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
            });
            var kwargs=new PyDict();
            if (write!=null) kwargs["write"]=ToPython(write);
            if (align!=null) kwargs["align"]=ToPython(align);
            if (uic!=null) kwargs["uic"]=ToPython(uic);
            dynamic py = self.InvokeMethod("setflags", pyargs, kwargs);
        }
        
        /// <summary>
        /// Fill the array with a scalar value.
        /// </summary>
        /// <param name="@value">
        /// All elements of a will be assigned this value.
        /// </param>
        public void fill(ValueType @value)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
                @value,
            });
            var kwargs=new PyDict();
            dynamic py = self.InvokeMethod("fill", pyargs, kwargs);
        }
        
        /// <summary>
        /// Returns a view of the array with axes transposed.
        /// 
        /// For a 1-D array, this has no effect. (To change between column and
        /// row vectors, first cast the 1-D array into a matrix object.)
        /// For a 2-D array, this is the usual matrix transpose.
        /// For an n-D array, if axes are given, their order indicates how the
        /// axes are permuted (see Examples). If axes are not provided and
        /// a.shape = (i[0], i[1], ... i[n-2], i[n-1]), then
        /// a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0]).
        /// </summary>
        /// <returns>
        /// View of a, with axes suitably permuted.
        /// </returns>
        public NDarray transpose(int[] axes = null)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
                axes,
            });
            var kwargs=new PyDict();
            dynamic py = self.InvokeMethod("transpose", pyargs, kwargs);
            return ToCsharp<NDarray>(py);
        }
        
        /// <summary>
        /// Return a copy of the array collapsed into one dimension.
        /// </summary>
        /// <param name="order">
        /// ‘C’ means to flatten in row-major (C-style) order.
        /// ‘F’ means to flatten in column-major (Fortran-
        /// style) order. ‘A’ means to flatten in column-major
        /// order if a is Fortran contiguous in memory,
        /// row-major order otherwise. ‘K’ means to flatten
        /// a in the order the elements occur in memory.
        /// The default is ‘C’.
        /// </param>
        /// <returns>
        /// A copy of the input array, flattened to one dimension.
        /// </returns>
        public NDarray flatten(string order = null)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
            });
            var kwargs=new PyDict();
            if (order!=null) kwargs["order"]=ToPython(order);
            dynamic py = self.InvokeMethod("flatten", pyargs, kwargs);
            return ToCsharp<NDarray>(py);
        }
        
        /// <summary>
        /// Sort an array, in-place.
        /// 
        /// Notes
        /// 
        /// See sort for notes on the different sorting algorithms.
        /// </summary>
        /// <param name="axis">
        /// Axis along which to sort. Default is -1, which means sort along the
        /// last axis.
        /// </param>
        /// <param name="kind">
        /// Sorting algorithm. Default is ‘quicksort’.
        /// </param>
        /// <param name="order">
        /// When a is an array with fields defined, this argument specifies
        /// which fields to compare first, second, etc.  A single field can
        /// be specified as a string, and not all fields need be specified,
        /// but unspecified fields will still be used, in the order in which
        /// they come up in the dtype, to break ties.
        /// </param>
        public void sort(int? axis = null, string kind = null, string order = null)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
            });
            var kwargs=new PyDict();
            if (axis!=null) kwargs["axis"]=ToPython(axis);
            if (kind!=null) kwargs["kind"]=ToPython(kind);
            if (order!=null) kwargs["order"]=ToPython(order);
            dynamic py = self.InvokeMethod("sort", pyargs, kwargs);
        }
        
        /// <summary>
        /// Rearranges the elements in the array in such a way that the value of the
        /// element in kth position is in the position it would be in a sorted array.
        /// All elements smaller than the kth element are moved before this element and
        /// all equal or greater are moved behind it. The ordering of the elements in
        /// the two partitions is undefined.
        /// 
        /// Notes
        /// 
        /// See np.partition for notes on the different algorithms.
        /// </summary>
        /// <param name="kth">
        /// Element index to partition by. The kth element value will be in its
        /// final sorted position and all smaller elements will be moved before it
        /// and all equal or greater elements behind it.
        /// The order of all elements in the partitions is undefined.
        /// If provided with a sequence of kth it will partition all elements
        /// indexed by kth of them into their sorted position at once.
        /// </param>
        /// <param name="axis">
        /// Axis along which to sort. Default is -1, which means sort along the
        /// last axis.
        /// </param>
        /// <param name="kind">
        /// Selection algorithm. Default is ‘introselect’.
        /// </param>
        /// <param name="order">
        /// When a is an array with fields defined, this argument specifies
        /// which fields to compare first, second, etc. A single field can
        /// be specified as a string, and not all fields need to be specified,
        /// but unspecified fields will still be used, in the order in which
        /// they come up in the dtype, to break ties.
        /// </param>
        public void partition(int[] kth, int? axis = null, string kind = null, string order = null)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
                kth,
            });
            var kwargs=new PyDict();
            if (axis!=null) kwargs["axis"]=ToPython(axis);
            if (kind!=null) kwargs["kind"]=ToPython(kind);
            if (order!=null) kwargs["order"]=ToPython(order);
            dynamic py = self.InvokeMethod("partition", pyargs, kwargs);
        }
        
        /// <summary>
        /// For unpickling.
        /// 
        /// The state argument must be a sequence that contains the following
        /// elements:
        /// </summary>
        /// <param name="version">
        /// optional pickle version. If omitted defaults to 0.
        /// </param>
        /// <param name="rawdata">
        /// a binary string with the data (or a list if ‘a’ is an object array)
        /// </param>
        public void __setstate__(int version, Shape shape, Dtype dtype, bool isFortran, string rawdata)
        {
            //auto-generated code, do not change
            var pyargs=ToTuple(new object[]
            {
                version,
                shape,
                dtype,
                isFortran,
                rawdata,
            });
            var kwargs=new PyDict();
            dynamic py = self.InvokeMethod("__setstate__", pyargs, kwargs);
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
        /// <param name="order">
        /// Read the elements of a using this index order, and place the
        /// elements into the reshaped array using this index order.  ‘C’
        /// means to read / write the elements using C-like index order,
        /// with the last axis index changing fastest, back to the first
        /// axis index changing slowest. ‘F’ means to read / write the
        /// elements using Fortran-like index order, with the first index
        /// changing fastest, and the last index changing slowest. Note that
        /// the ‘C’ and ‘F’ options take no account of the memory layout of
        /// the underlying array, and only refer to the order of indexing.
        /// ‘A’ means to read / write the elements in Fortran-like index
        /// order if a is Fortran contiguous in memory, C-like order
        /// otherwise.
        /// </param>
        /// <returns>
        /// This will be a new view object if possible; otherwise, it will
        /// be a copy.  Note there is no guarantee of the memory layout (C- or
        /// Fortran- contiguous) of the returned array.
        /// </returns>
        public NDarray reshape(Shape newshape, string order = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.reshape(@this, newshape, order:order);
        }
        
        /// <summary>
        /// Return a contiguous flattened array.
        /// 
        /// A 1-D array, containing the elements of the input, is returned.  A copy is
        /// made only if needed.
        /// 
        /// As of NumPy 1.10, the returned array will have the same type as the input
        /// array. (for example, a masked array will be returned for a masked array
        /// input)
        /// 
        /// Notes
        /// 
        /// In row-major, C-style order, in two dimensions, the row index
        /// varies the slowest, and the column index the quickest.  This can
        /// be generalized to multiple dimensions, where row-major order
        /// implies that the index along the first axis varies slowest, and
        /// the index along the last quickest.  The opposite holds for
        /// column-major, Fortran-style index ordering.
        /// 
        /// When a view is desired in as many cases as possible, arr.reshape(-1)
        /// may be preferable.
        /// </summary>
        /// <param name="order">
        /// The elements of a are read using this index order. ‘C’ means
        /// to index the elements in row-major, C-style order,
        /// with the last axis index changing fastest, back to the first
        /// axis index changing slowest.  ‘F’ means to index the elements
        /// in column-major, Fortran-style order, with the
        /// first index changing fastest, and the last index changing
        /// slowest. Note that the ‘C’ and ‘F’ options take no account of
        /// the memory layout of the underlying array, and only refer to
        /// the order of axis indexing.  ‘A’ means to read the elements in
        /// Fortran-like index order if a is Fortran contiguous in
        /// memory, C-like order otherwise.  ‘K’ means to read the
        /// elements in the order they occur in memory, except for
        /// reversing the data when strides are negative.  By default, ‘C’
        /// index order is used.
        /// </param>
        /// <returns>
        /// y is an array of the same subtype as a, with shape (a.size,).
        /// Note that matrices are special cased for backward compatibility, if a
        /// is a matrix, then y is a 1-D ndarray.
        /// </returns>
        public NDarray ravel(string order = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.ravel(@this, order:order);
        }
        
        /// <summary>
        /// Move axes of an array to new positions.
        /// 
        /// Other axes remain in their original order.
        /// </summary>
        /// <param name="source">
        /// Original positions of the axes to move. These must be unique.
        /// </param>
        /// <param name="destination">
        /// Destination positions for each of the original axes. These must also be
        /// unique.
        /// </param>
        /// <returns>
        /// Array with moved axes. This array is a view of the input array.
        /// </returns>
        public NDarray moveaxis(int[] source, int[] destination)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.moveaxis(@this, source, destination);
        }
        
        /// <summary>
        /// Roll the specified axis backwards, until it lies in a given position.
        /// 
        /// This function continues to be supported for backward compatibility, but you
        /// should prefer moveaxis. The moveaxis function was added in NumPy
        /// 1.11.
        /// </summary>
        /// <param name="axis">
        /// The axis to roll backwards.  The positions of the other axes do not
        /// change relative to one another.
        /// </param>
        /// <param name="start">
        /// The axis is rolled until it lies before this position.  The default,
        /// 0, results in a “complete” roll.
        /// </param>
        /// <returns>
        /// For NumPy &gt;= 1.10.0 a view of a is always returned. For earlier
        /// NumPy versions a view of a is returned only if the order of the
        /// axes is changed, otherwise the input array is returned.
        /// </returns>
        public NDarray rollaxis(int axis, int? start = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.rollaxis(@this, axis, start:start);
        }
        
        /// <summary>
        /// Interchange two axes of an array.
        /// </summary>
        /// <param name="axis1">
        /// First axis.
        /// </param>
        /// <param name="axis2">
        /// Second axis.
        /// </param>
        /// <returns>
        /// For NumPy &gt;= 1.10.0, if a is an ndarray, then a view of a is
        /// returned; otherwise a new array is created. For earlier NumPy
        /// versions a view of a is returned only if the order of the
        /// axes is changed, otherwise the input array is returned.
        /// </returns>
        public NDarray swapaxes(int axis1, int axis2)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.swapaxes(@this, axis1, axis2);
        }
        
        /// <summary>
        /// Broadcast an array to a new shape.
        /// 
        /// Notes
        /// </summary>
        /// <param name="shape">
        /// The shape of the desired array.
        /// </param>
        /// <param name="subok">
        /// If True, then sub-classes will be passed-through, otherwise
        /// the returned array will be forced to be a base-class array (default).
        /// </param>
        /// <returns>
        /// A readonly view on the original array with the given shape. It is
        /// typically not contiguous. Furthermore, more than one element of a
        /// broadcasted array may refer to a single memory location.
        /// </returns>
        public NDarray broadcast_to(Shape shape, bool? subok = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.broadcast_to(@this, shape, subok:subok);
        }
        
        /// <summary>
        /// Expand the shape of an array.
        /// 
        /// Insert a new axis that will appear at the axis position in the expanded
        /// array shape.
        /// </summary>
        /// <param name="axis">
        /// Position in the expanded axes where the new axis is placed.
        /// </param>
        /// <returns>
        /// Output array. The number of dimensions is one greater than that of
        /// the input array.
        /// </returns>
        public NDarray expand_dims(int axis)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.expand_dims(@this, axis);
        }
        
        /// <summary>
        /// Remove single-dimensional entries from the shape of an array.
        /// </summary>
        /// <param name="axis">
        /// Selects a subset of the single-dimensional entries in the
        /// shape. If an axis is selected with shape entry greater than
        /// one, an error is raised.
        /// </param>
        /// <returns>
        /// The input array, but with all or a subset of the
        /// dimensions of length 1 removed. This is always a itself
        /// or a view into a.
        /// </returns>
        public NDarray squeeze(int[] axis = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.squeeze(@this, axis:axis);
        }
        
        /// <summary>
        /// Return an array converted to a float type.
        /// </summary>
        /// <param name="dtype">
        /// Float type code to coerce input array a.  If dtype is one of the
        /// ‘int’ dtypes, it is replaced with float64.
        /// </param>
        /// <returns>
        /// The input a as a float ndarray.
        /// </returns>
        public NDarray asfarray(Dtype dtype = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.asfarray(@this, dtype:dtype);
        }
        
        /// <summary>
        /// Return an array (ndim &gt;= 1) laid out in Fortran order in memory.
        /// </summary>
        /// <param name="dtype">
        /// By default, the data-type is inferred from the input data.
        /// </param>
        /// <returns>
        /// The input a in Fortran, or column-major, order.
        /// </returns>
        public NDarray asfortranarray(Dtype dtype = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.asfortranarray(@this, dtype:dtype);
        }
        
        /// <summary>
        /// Convert the input to an array, checking for NaNs or Infs.
        /// </summary>
        /// <param name="dtype">
        /// By default, the data-type is inferred from the input data.
        /// </param>
        /// <param name="order">
        /// Whether to use row-major (C-style) or
        /// column-major (Fortran-style) memory representation.
        /// Defaults to ‘C’.
        /// </param>
        /// <returns>
        /// Array interpretation of a.  No copy is performed if the input
        /// is already an ndarray.  If a is a subclass of ndarray, a base
        /// class ndarray is returned.
        /// </returns>
        public NDarray asarray_chkfinite(Dtype dtype = null, string order = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.asarray_chkfinite(@this, dtype:dtype, order:order);
        }
        
        /// <summary>
        /// Convert an array of size 1 to its scalar equivalent.
        /// </summary>
        /// <returns>
        /// Scalar representation of a. The output data type is the same type
        /// returned by the input’s item method.
        /// </returns>
        public ValueType asscalar()
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.asscalar(@this);
        }
        
        /// <summary>
        /// Return an ndarray of the provided type that satisfies requirements.
        /// 
        /// This function is useful to be sure that an array with the correct flags
        /// is returned for passing to compiled code (perhaps through ctypes).
        /// 
        /// Notes
        /// 
        /// The returned array will be guaranteed to have the listed requirements
        /// by making a copy if needed.
        /// </summary>
        /// <param name="dtype">
        /// The required data-type. If None preserve the current dtype. If your
        /// application requires the data to be in native byteorder, include
        /// a byteorder specification as a part of the dtype specification.
        /// </param>
        /// <param name="requirements">
        /// The requirements list can be any of the following
        /// </param>
        public NDarray require(Dtype dtype, string[] requirements)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.require(@this, dtype, requirements);
        }
        
        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        /// <param name="indices_or_sections">
        /// If indices_or_sections is an integer, N, the array will be divided
        /// into N equal arrays along axis.  If such a split is not possible,
        /// an error is raised.
        /// 
        /// If indices_or_sections is a 1-D array of sorted integers, the entries
        /// indicate where along axis the array is split.  For example,
        /// [2, 3] would, for axis=0, result in
        /// 
        /// If an index exceeds the dimension of the array along axis,
        /// an empty sub-array is returned correspondingly.
        /// </param>
        /// <param name="axis">
        /// The axis along which to split, default is 0.
        /// </param>
        /// <returns>
        /// A list of sub-arrays.
        /// </returns>
        public NDarray[] split(int[] indices_or_sections, int? axis = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.split(@this, indices_or_sections, axis:axis);
        }
        
        /// <summary>
        /// Construct an array by repeating A the number of times given by reps.
        /// 
        /// If reps has length d, the result will have dimension of
        /// max(d, A.ndim).
        /// 
        /// If A.ndim &lt; d, A is promoted to be d-dimensional by prepending new
        /// axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
        /// or shape (1, 1, 3) for 3-D replication. If this is not the desired
        /// behavior, promote A to d-dimensions manually before calling this
        /// function.
        /// 
        /// If A.ndim &gt; d, reps is promoted to A.ndim by pre-pending 1’s to it.
        /// Thus for an A of shape (2, 3, 4, 5), a reps of (2, 2) is treated as
        /// (1, 1, 2, 2).
        /// 
        /// Note : Although tile may be used for broadcasting, it is strongly
        /// recommended to use numpy’s broadcasting operations and functions.
        /// </summary>
        /// <param name="reps">
        /// The number of repetitions of A along each axis.
        /// </param>
        /// <returns>
        /// The tiled output array.
        /// </returns>
        public NDarray tile(NDarray reps)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.tile(@this, reps);
        }
        
        /// <summary>
        /// Repeat elements of an array.
        /// </summary>
        /// <param name="repeats">
        /// The number of repetitions for each element.  repeats is broadcasted
        /// to fit the shape of the given axis.
        /// </param>
        /// <param name="axis">
        /// The axis along which to repeat values.  By default, use the
        /// flattened input array, and return a flat output array.
        /// </param>
        /// <returns>
        /// Output array which has the same shape as a, except along
        /// the given axis.
        /// </returns>
        public NDarray repeat(int[] repeats, int? axis = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.repeat(@this, repeats, axis:axis);
        }
        
        /// <summary>
        /// Return a new array with sub-arrays along an axis deleted. For a one
        /// dimensional array, this returns those entries not returned by
        /// arr[obj].
        /// 
        /// Notes
        /// 
        /// Often it is preferable to use a boolean mask. For example:
        /// 
        /// Is equivalent to np.delete(arr, [0,2,4], axis=0), but allows further
        /// use of mask.
        /// </summary>
        /// <param name="obj">
        /// Indicate which sub-arrays to remove.
        /// </param>
        /// <param name="axis">
        /// The axis along which to delete the subarray defined by obj.
        /// If axis is None, obj is applied to the flattened array.
        /// </param>
        /// <returns>
        /// A copy of arr with the elements specified by obj removed. Note
        /// that delete does not occur in-place. If axis is None, out is
        /// a flattened array.
        /// </returns>
        public NDarray delete(Slice obj, int? axis = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.delete(@this, obj, axis:axis);
        }
        
        /// <summary>
        /// Insert values along the given axis before the given indices.
        /// 
        /// Notes
        /// 
        /// Note that for higher dimensional inserts obj=0 behaves very different
        /// from obj=[0] just like arr[:,0,:] = values is different from
        /// arr[:,[0],:] = values.
        /// </summary>
        /// <param name="obj">
        /// Object that defines the index or indices before which values is
        /// inserted.
        /// 
        /// Support for multiple insertions when obj is a single scalar or a
        /// sequence with one element (similar to calling insert multiple
        /// times).
        /// </param>
        /// <param name="values">
        /// Values to insert into arr. If the type of values is different
        /// from that of arr, values is converted to the type of arr.
        /// values should be shaped so that arr[...,obj,...] = values
        /// is legal.
        /// </param>
        /// <param name="axis">
        /// Axis along which to insert values.  If axis is None then arr
        /// is flattened first.
        /// </param>
        /// <returns>
        /// A copy of arr with values inserted.  Note that insert
        /// does not occur in-place: a new array is returned. If
        /// axis is None, out is a flattened array.
        /// </returns>
        public NDarray insert(int obj, NDarray values, int? axis = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.insert(@this, obj, values, axis:axis);
        }
        
        /// <summary>
        /// Insert values along the given axis before the given indices.
        /// 
        /// Notes
        /// 
        /// Note that for higher dimensional inserts obj=0 behaves very different
        /// from obj=[0] just like arr[:,0,:] = values is different from
        /// arr[:,[0],:] = values.
        /// </summary>
        /// <param name="obj">
        /// Object that defines the index or indices before which values is
        /// inserted.
        /// 
        /// Support for multiple insertions when obj is a single scalar or a
        /// sequence with one element (similar to calling insert multiple
        /// times).
        /// </param>
        /// <param name="values">
        /// Values to insert into arr. If the type of values is different
        /// from that of arr, values is converted to the type of arr.
        /// values should be shaped so that arr[...,obj,...] = values
        /// is legal.
        /// </param>
        /// <param name="axis">
        /// Axis along which to insert values.  If axis is None then arr
        /// is flattened first.
        /// </param>
        /// <returns>
        /// A copy of arr with values inserted.  Note that insert
        /// does not occur in-place: a new array is returned. If
        /// axis is None, out is a flattened array.
        /// </returns>
        public NDarray<T> insert<T>(int obj, T[] values, int? axis = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.insert(@this, obj, values, axis:axis);
        }
        
        /// <summary>
        /// Insert values along the given axis before the given indices.
        /// 
        /// Notes
        /// 
        /// Note that for higher dimensional inserts obj=0 behaves very different
        /// from obj=[0] just like arr[:,0,:] = values is different from
        /// arr[:,[0],:] = values.
        /// </summary>
        /// <param name="obj">
        /// Object that defines the index or indices before which values is
        /// inserted.
        /// 
        /// Support for multiple insertions when obj is a single scalar or a
        /// sequence with one element (similar to calling insert multiple
        /// times).
        /// </param>
        /// <param name="values">
        /// Values to insert into arr. If the type of values is different
        /// from that of arr, values is converted to the type of arr.
        /// values should be shaped so that arr[...,obj,...] = values
        /// is legal.
        /// </param>
        /// <param name="axis">
        /// Axis along which to insert values.  If axis is None then arr
        /// is flattened first.
        /// </param>
        /// <returns>
        /// A copy of arr with values inserted.  Note that insert
        /// does not occur in-place: a new array is returned. If
        /// axis is None, out is a flattened array.
        /// </returns>
        public NDarray<T> insert<T>(int obj, T[,] values, int? axis = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.insert(@this, obj, values, axis:axis);
        }
        
        /// <summary>
        /// Append values to the end of an array.
        /// </summary>
        /// <param name="values">
        /// These values are appended to a copy of arr.  It must be of the
        /// correct shape (the same shape as arr, excluding axis).  If
        /// axis is not specified, values can be any shape and will be
        /// flattened before use.
        /// </param>
        /// <param name="axis">
        /// The axis along which values are appended.  If axis is not
        /// given, both arr and values are flattened before use.
        /// </param>
        /// <returns>
        /// A copy of arr with values appended to axis.  Note that
        /// append does not occur in-place: a new array is allocated and
        /// filled.  If axis is None, out is a flattened array.
        /// </returns>
        public NDarray append(NDarray values, int? axis = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.append(@this, values, axis:axis);
        }
        
        /// <summary>
        /// Append values to the end of an array.
        /// </summary>
        /// <param name="values">
        /// These values are appended to a copy of arr.  It must be of the
        /// correct shape (the same shape as arr, excluding axis).  If
        /// axis is not specified, values can be any shape and will be
        /// flattened before use.
        /// </param>
        /// <param name="axis">
        /// The axis along which values are appended.  If axis is not
        /// given, both arr and values are flattened before use.
        /// </param>
        /// <returns>
        /// A copy of arr with values appended to axis.  Note that
        /// append does not occur in-place: a new array is allocated and
        /// filled.  If axis is None, out is a flattened array.
        /// </returns>
        public NDarray<T> append<T>(T[] values, int? axis = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.append(@this, values, axis:axis);
        }
        
        /// <summary>
        /// Append values to the end of an array.
        /// </summary>
        /// <param name="values">
        /// These values are appended to a copy of arr.  It must be of the
        /// correct shape (the same shape as arr, excluding axis).  If
        /// axis is not specified, values can be any shape and will be
        /// flattened before use.
        /// </param>
        /// <param name="axis">
        /// The axis along which values are appended.  If axis is not
        /// given, both arr and values are flattened before use.
        /// </param>
        /// <returns>
        /// A copy of arr with values appended to axis.  Note that
        /// append does not occur in-place: a new array is allocated and
        /// filled.  If axis is None, out is a flattened array.
        /// </returns>
        public NDarray<T> append<T>(T[,] values, int? axis = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.append(@this, values, axis:axis);
        }
        
        /// <summary>
        /// Trim the leading and/or trailing zeros from a 1-D array or sequence.
        /// </summary>
        /// <param name="trim">
        /// A string with ‘f’ representing trim from front and ‘b’ to trim from
        /// back. Default is ‘fb’, trim zeros from both front and back of the
        /// array.
        /// </param>
        /// <returns>
        /// The result of trimming the input. The input data type is preserved.
        /// </returns>
        public NDarray trim_zeros(string trim = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.trim_zeros(@this, trim:trim);
        }
        
        /// <summary>
        /// Find the unique elements of an array.
        /// 
        /// Returns the sorted unique elements of an array. There are three optional
        /// outputs in addition to the unique elements:
        /// 
        /// Notes
        /// 
        /// When an axis is specified the subarrays indexed by the axis are sorted.
        /// This is done by making the specified axis the first dimension of the array
        /// and then flattening the subarrays in C order. The flattened subarrays are
        /// then viewed as a structured type with each element given a label, with the
        /// effect that we end up with a 1-D array of structured types that can be
        /// treated in the same way as any other 1-D array. The result is that the
        /// flattened subarrays are sorted in lexicographic order starting with the
        /// first element.
        /// </summary>
        /// <param name="return_index">
        /// If True, also return the indices of ar (along the specified axis,
        /// if provided, or in the flattened array) that result in the unique array.
        /// </param>
        /// <param name="return_inverse">
        /// If True, also return the indices of the unique array (for the specified
        /// axis, if provided) that can be used to reconstruct ar.
        /// </param>
        /// <param name="return_counts">
        /// If True, also return the number of times each unique item appears
        /// in ar.
        /// </param>
        /// <param name="axis">
        /// The axis to operate on. If None, ar will be flattened. If an integer,
        /// the subarrays indexed by the given axis will be flattened and treated
        /// as the elements of a 1-D array with the dimension of the given axis,
        /// see the notes for more details.  Object arrays or structured arrays
        /// that contain objects are not supported if the axis kwarg is used. The
        /// default is None.
        /// </param>
        /// <returns>
        /// A tuple of:
        /// unique
        /// The sorted unique values.
        /// unique_indices
        /// The indices of the first occurrences of the unique values in the
        /// original array. Only provided if return_index is True.
        /// unique_inverse
        /// The indices to reconstruct the original array from the
        /// unique array. Only provided if return_inverse is True.
        /// unique_counts
        /// The number of times each of the unique values comes up in the
        /// original array. Only provided if return_counts is True.
        /// </returns>
        // Error generating delaration: unique
        // Message: Return tuple
        /*
           at CodeMinion.Core.CodeGenerator.GenerateReturnType(Declaration decl) in D:\dev\CodeMinion\src\CodeMinion.Core\CodeGenerator.cs:line 235
   at CodeMinion.Core.CodeGenerator.GenerateApiFunction(Declaration decl, CodeWriter s) in D:\dev\CodeMinion\src\CodeMinion.Core\CodeGenerator.cs:line 62
   at CodeMinion.Core.CodeGenerator.<>c__DisplayClass55_0.<GenerateDynamicApi>b__1() in D:\dev\CodeMinion\src\CodeMinion.Core\CodeGenerator.cs:line 484
        ----------------------------
        Declaration JSON:
        {
  "Arguments": [
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "return_index",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, also return the indices of ar (along the specified axis,\nif provided, or in the flattened array) that result in the unique array.",
      "ConvertToSharpType": null,
      "Position": 1,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "return_inverse",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, also return the indices of the unique array (for the specified\naxis, if provided) that can be used to reconstruct ar.",
      "ConvertToSharpType": null,
      "Position": 2,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "return_counts",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, also return the number of times each unique item appears\nin ar.",
      "ConvertToSharpType": null,
      "Position": 3,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "axis",
      "Type": "int or None",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The axis to operate on. If None, ar will be flattened. If an integer,\nthe subarrays indexed by the given axis will be flattened and treated\nas the elements of a 1-D array with the dimension of the given axis,\nsee the notes for more details.  Object arrays or structured arrays\nthat contain objects are not supported if the axis kwarg is used. The\ndefault is None.",
      "ConvertToSharpType": null,
      "Position": 4,
      "IsReturnValue": false
    }
  ],
  "Generics": null,
  "ForwardToStaticImpl": "NumPy.Instance",
  "Name": "unique",
  "ClassName": "numpy",
  "Returns": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "unique",
      "Type": "NDarray",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The sorted unique values.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "unique_indices",
      "Type": "NDarray",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The indices of the first occurrences of the unique values in the\noriginal array. Only provided if return_index is True.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "unique_inverse",
      "Type": "NDarray",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The indices to reconstruct the original array from the\nunique array. Only provided if return_inverse is True.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "unique_counts",
      "Type": "NDarray",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The number of times each of the unique values comes up in the\noriginal array. Only provided if return_counts is True.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    }
  ],
  "IsDeprecated": false,
  "ManualOverride": false,
  "CommentOut": false,
  "DebuggerBreak": false,
  "Description": "Find the unique elements of an array.\r\n\r\nReturns the sorted unique elements of an array. There are three optional\noutputs in addition to the unique elements:\r\n\r\nNotes\r\n\r\nWhen an axis is specified the subarrays indexed by the axis are sorted.\nThis is done by making the specified axis the first dimension of the array\nand then flattening the subarrays in C order. The flattened subarrays are\nthen viewed as a structured type with each element given a label, with the\neffect that we end up with a 1-D array of structured types that can be\ntreated in the same way as any other 1-D array. The result is that the\nflattened subarrays are sorted in lexicographic order starting with the\nfirst element."
}
        */
        /// <summary>
        /// Reverse the order of elements in an array along the given axis.
        /// 
        /// The shape of the array is preserved, but the elements are reordered.
        /// 
        /// Notes
        /// 
        /// flip(m, 0) is equivalent to flipud(m).
        /// 
        /// flip(m, 1) is equivalent to fliplr(m).
        /// 
        /// flip(m, n) corresponds to m[...,::-1,...] with ::-1 at position n.
        /// 
        /// flip(m) corresponds to m[::-1,::-1,...,::-1] with ::-1 at all
        /// positions.
        /// 
        /// flip(m, (0, 1)) corresponds to m[::-1,::-1,...] with ::-1 at
        /// position 0 and position 1.
        /// </summary>
        /// <param name="axis">
        /// Axis or axes along which to flip over. The default,
        /// axis=None, will flip over all of the axes of the input array.
        /// If axis is negative it counts from the last to the first axis.
        /// 
        /// If axis is a tuple of ints, flipping is performed on all of the axes
        /// specified in the tuple.
        /// </param>
        /// <returns>
        /// A view of m with the entries of axis reversed.  Since a view is
        /// returned, this operation is done in constant time.
        /// </returns>
        public NDarray flip(int[] axis = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.flip(@this, axis:axis);
        }
        
        /// <summary>
        /// Flip array in the left/right direction.
        /// 
        /// Flip the entries in each row in the left/right direction.
        /// Columns are preserved, but appear in a different order than before.
        /// 
        /// Notes
        /// 
        /// Equivalent to m[:,::-1]. Requires the array to be at least 2-D.
        /// </summary>
        /// <returns>
        /// A view of m with the columns reversed.  Since a view
        /// is returned, this operation is .
        /// </returns>
        public NDarray fliplr()
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.fliplr(@this);
        }
        
        /// <summary>
        /// Flip array in the up/down direction.
        /// 
        /// Flip the entries in each column in the up/down direction.
        /// Rows are preserved, but appear in a different order than before.
        /// 
        /// Notes
        /// 
        /// Equivalent to m[::-1,...].
        /// Does not require the array to be two-dimensional.
        /// </summary>
        /// <returns>
        /// A view of m with the rows reversed.  Since a view is
        /// returned, this operation is .
        /// </returns>
        public NDarray flipud()
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.flipud(@this);
        }
        
        /// <summary>
        /// Roll array elements along a given axis.
        /// 
        /// Elements that roll beyond the last position are re-introduced at
        /// the first.
        /// 
        /// Notes
        /// 
        /// Supports rolling over multiple dimensions simultaneously.
        /// </summary>
        /// <param name="shift">
        /// The number of places by which elements are shifted.  If a tuple,
        /// then axis must be a tuple of the same size, and each of the
        /// given axes is shifted by the corresponding number.  If an int
        /// while axis is a tuple of ints, then the same value is used for
        /// all given axes.
        /// </param>
        /// <param name="axis">
        /// Axis or axes along which elements are shifted.  By default, the
        /// array is flattened before shifting, after which the original
        /// shape is restored.
        /// </param>
        /// <returns>
        /// Output array, with the same shape as a.
        /// </returns>
        public NDarray roll(int[] shift, int[] axis = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.roll(@this, shift, axis:axis);
        }
        
        /// <summary>
        /// Rotate an array by 90 degrees in the plane specified by axes.
        /// 
        /// Rotation direction is from the first towards the second axis.
        /// 
        /// Notes
        /// 
        /// rot90(m, k=1, axes=(1,0)) is the reverse of rot90(m, k=1, axes=(0,1))
        /// rot90(m, k=1, axes=(1,0)) is equivalent to rot90(m, k=-1, axes=(0,1))
        /// </summary>
        /// <param name="k">
        /// Number of times the array is rotated by 90 degrees.
        /// </param>
        /// <param name="axes">
        /// The array is rotated in the plane defined by the axes.
        /// Axes must be different.
        /// </param>
        /// <returns>
        /// A rotated view of m.
        /// </returns>
        public NDarray rot90(int k, int[] axes = null)
        {
            //auto-generated code, do not change
            var @this=this;
            return NumPy.Instance.rot90(@this, k, axes);
        }
        
    }
}
