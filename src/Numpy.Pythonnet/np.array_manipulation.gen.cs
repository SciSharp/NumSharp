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
    public static partial class np
    {
        
        /// <summary>
        /// Copies values from one array to another, broadcasting as necessary.
        /// 
        /// Raises a TypeError if the casting rule is violated, and if
        /// where is provided, it selects which elements to copy.
        /// </summary>
        /// <param name="dst">
        /// The array into which values are copied.
        /// </param>
        /// <param name="src">
        /// The array from which values are copied.
        /// </param>
        /// <param name="casting">
        /// Controls what kind of data casting may occur when copying.
        /// </param>
        /// <param name="@where">
        /// A boolean array which is broadcasted to match the dimensions
        /// of dst, and selects elements to copy from src to dst
        /// wherever it contains the value True.
        /// </param>
        public static void copyto(NDarray dst, NDarray src, string casting = null, NDarray @where = null)
            => NumPy.Instance.copyto(dst, src, casting:casting, @where:@where);
        
        /// <summary>
        /// Copies values from one array to another, broadcasting as necessary.
        /// 
        /// Raises a TypeError if the casting rule is violated, and if
        /// where is provided, it selects which elements to copy.
        /// </summary>
        /// <param name="dst">
        /// The array into which values are copied.
        /// </param>
        /// <param name="src">
        /// The array from which values are copied.
        /// </param>
        /// <param name="casting">
        /// Controls what kind of data casting may occur when copying.
        /// </param>
        /// <param name="@where">
        /// A boolean array which is broadcasted to match the dimensions
        /// of dst, and selects elements to copy from src to dst
        /// wherever it contains the value True.
        /// </param>
        public static void copyto<T>(NDarray dst, T[] src, string casting = null, NDarray @where = null)
            => NumPy.Instance.copyto(dst, src, casting:casting, @where:@where);
        
        /// <summary>
        /// Copies values from one array to another, broadcasting as necessary.
        /// 
        /// Raises a TypeError if the casting rule is violated, and if
        /// where is provided, it selects which elements to copy.
        /// </summary>
        /// <param name="dst">
        /// The array into which values are copied.
        /// </param>
        /// <param name="src">
        /// The array from which values are copied.
        /// </param>
        /// <param name="casting">
        /// Controls what kind of data casting may occur when copying.
        /// </param>
        /// <param name="@where">
        /// A boolean array which is broadcasted to match the dimensions
        /// of dst, and selects elements to copy from src to dst
        /// wherever it contains the value True.
        /// </param>
        public static void copyto<T>(NDarray dst, T[,] src, string casting = null, NDarray @where = null)
            => NumPy.Instance.copyto(dst, src, casting:casting, @where:@where);
        
        /// <summary>
        /// Copies values from one array to another, broadcasting as necessary.
        /// 
        /// Raises a TypeError if the casting rule is violated, and if
        /// where is provided, it selects which elements to copy.
        /// </summary>
        /// <param name="dst">
        /// The array into which values are copied.
        /// </param>
        /// <param name="src">
        /// The array from which values are copied.
        /// </param>
        /// <param name="casting">
        /// Controls what kind of data casting may occur when copying.
        /// </param>
        /// <param name="@where">
        /// A boolean array which is broadcasted to match the dimensions
        /// of dst, and selects elements to copy from src to dst
        /// wherever it contains the value True.
        /// </param>
        public static void copyto(NDarray dst, NDarray src, string casting = null, bool[] @where = null)
            => NumPy.Instance.copyto(dst, src, casting:casting, @where:@where);
        
        /// <summary>
        /// Copies values from one array to another, broadcasting as necessary.
        /// 
        /// Raises a TypeError if the casting rule is violated, and if
        /// where is provided, it selects which elements to copy.
        /// </summary>
        /// <param name="dst">
        /// The array into which values are copied.
        /// </param>
        /// <param name="src">
        /// The array from which values are copied.
        /// </param>
        /// <param name="casting">
        /// Controls what kind of data casting may occur when copying.
        /// </param>
        /// <param name="@where">
        /// A boolean array which is broadcasted to match the dimensions
        /// of dst, and selects elements to copy from src to dst
        /// wherever it contains the value True.
        /// </param>
        public static void copyto<T>(NDarray dst, T[] src, string casting = null, bool[] @where = null)
            => NumPy.Instance.copyto(dst, src, casting:casting, @where:@where);
        
        /// <summary>
        /// Copies values from one array to another, broadcasting as necessary.
        /// 
        /// Raises a TypeError if the casting rule is violated, and if
        /// where is provided, it selects which elements to copy.
        /// </summary>
        /// <param name="dst">
        /// The array into which values are copied.
        /// </param>
        /// <param name="src">
        /// The array from which values are copied.
        /// </param>
        /// <param name="casting">
        /// Controls what kind of data casting may occur when copying.
        /// </param>
        /// <param name="@where">
        /// A boolean array which is broadcasted to match the dimensions
        /// of dst, and selects elements to copy from src to dst
        /// wherever it contains the value True.
        /// </param>
        public static void copyto<T>(NDarray dst, T[,] src, string casting = null, bool[] @where = null)
            => NumPy.Instance.copyto(dst, src, casting:casting, @where:@where);
        
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
        /// <param name="a">
        /// Array to be reshaped.
        /// </param>
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
        public static NDarray reshape(NDarray a, Shape newshape, string order = null)
            => NumPy.Instance.reshape(a, newshape, order:order);
        
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
        /// <param name="a">
        /// Input array.  The elements in a are read in the order specified by
        /// order, and packed as a 1-D array.
        /// </param>
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
        public static NDarray ravel(NDarray a, string order = null)
            => NumPy.Instance.ravel(a, order:order);
        
        /// <summary>
        /// Move axes of an array to new positions.
        /// 
        /// Other axes remain in their original order.
        /// </summary>
        /// <param name="a">
        /// The array whose axes should be reordered.
        /// </param>
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
        public static NDarray moveaxis(NDarray a, int[] source, int[] destination)
            => NumPy.Instance.moveaxis(a, source, destination);
        
        /// <summary>
        /// Roll the specified axis backwards, until it lies in a given position.
        /// 
        /// This function continues to be supported for backward compatibility, but you
        /// should prefer moveaxis. The moveaxis function was added in NumPy
        /// 1.11.
        /// </summary>
        /// <param name="a">
        /// Input array.
        /// </param>
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
        public static NDarray rollaxis(NDarray a, int axis, int? start = null)
            => NumPy.Instance.rollaxis(a, axis, start:start);
        
        /// <summary>
        /// Interchange two axes of an array.
        /// </summary>
        /// <param name="a">
        /// Input array.
        /// </param>
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
        public static NDarray swapaxes(NDarray a, int axis1, int axis2)
            => NumPy.Instance.swapaxes(a, axis1, axis2);
        
        /// <summary>
        /// Permute the dimensions of an array.
        /// 
        /// Notes
        /// 
        /// Use transpose(a, argsort(axes)) to invert the transposition of tensors
        /// when using the axes keyword argument.
        /// 
        /// Transposing a 1-D array returns an unchanged view of the original array.
        /// </summary>
        /// <param name="a">
        /// Input array.
        /// </param>
        /// <param name="axes">
        /// By default, reverse the dimensions, otherwise permute the axes
        /// according to the values given.
        /// </param>
        /// <returns>
        /// a with its axes permuted.  A view is returned whenever
        /// possible.
        /// </returns>
        public static NDarray transpose(NDarray a, int[] axes = null)
            => NumPy.Instance.transpose(a, axes:axes);
        
        /// <summary>
        /// Convert inputs to arrays with at least one dimension.
        /// 
        /// Scalar inputs are converted to 1-dimensional arrays, whilst
        /// higher-dimensional inputs are preserved.
        /// </summary>
        /// <param name="arys">
        /// One or more input arrays.
        /// </param>
        /// <returns>
        /// An array, or list of arrays, each with a.ndim &gt;= 1.
        /// Copies are made only if necessary.
        /// </returns>
        public static NDarray atleast_1d(params NDarray[] arys)
            => NumPy.Instance.atleast_1d(arys);
        
        /// <summary>
        /// View inputs as arrays with at least two dimensions.
        /// </summary>
        /// <param name="arys">
        /// One or more array-like sequences.  Non-array inputs are converted
        /// to arrays.  Arrays that already have two or more dimensions are
        /// preserved.
        /// </param>
        /// <returns>
        /// An array, or list of arrays, each with a.ndim &gt;= 2.
        /// Copies are avoided where possible, and views with two or more
        /// dimensions are returned.
        /// </returns>
        public static NDarray atleast_2d(params NDarray[] arys)
            => NumPy.Instance.atleast_2d(arys);
        
        /// <summary>
        /// View inputs as arrays with at least three dimensions.
        /// </summary>
        /// <param name="arys">
        /// One or more array-like sequences.  Non-array inputs are converted to
        /// arrays.  Arrays that already have three or more dimensions are
        /// preserved.
        /// </param>
        /// <returns>
        /// An array, or list of arrays, each with a.ndim &gt;= 3.  Copies are
        /// avoided where possible, and views with three or more dimensions are
        /// returned.  For example, a 1-D array of shape (N,) becomes a view
        /// of shape (1, N, 1), and a 2-D array of shape (M, N) becomes a
        /// view of shape (M, N, 1).
        /// </returns>
        public static NDarray atleast_3d(params NDarray[] arys)
            => NumPy.Instance.atleast_3d(arys);
        
        /// <summary>
        /// Broadcast an array to a new shape.
        /// 
        /// Notes
        /// </summary>
        /// <param name="array">
        /// The array to broadcast.
        /// </param>
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
        public static NDarray broadcast_to(NDarray array, Shape shape, bool? subok = null)
            => NumPy.Instance.broadcast_to(array, shape, subok:subok);
        
        /// <summary>
        /// Broadcast any number of arrays against each other.
        /// </summary>
        /// <param name="args">
        /// The arrays to broadcast.
        /// </param>
        /// <param name="subok">
        /// If True, then sub-classes will be passed-through, otherwise
        /// the returned arrays will be forced to be a base-class array (default).
        /// </param>
        /// <returns>
        /// These arrays are views on the original arrays.  They are typically
        /// not contiguous.  Furthermore, more than one element of a
        /// broadcasted array may refer to a single memory location.  If you
        /// need to write to the arrays, make copies first.
        /// </returns>
        public static NDarray[] broadcast_arrays(NDarray[] args, bool? subok = null)
            => NumPy.Instance.broadcast_arrays(args, subok:subok);
        
        /// <summary>
        /// Expand the shape of an array.
        /// 
        /// Insert a new axis that will appear at the axis position in the expanded
        /// array shape.
        /// </summary>
        /// <param name="a">
        /// Input array.
        /// </param>
        /// <param name="axis">
        /// Position in the expanded axes where the new axis is placed.
        /// </param>
        /// <returns>
        /// Output array. The number of dimensions is one greater than that of
        /// the input array.
        /// </returns>
        public static NDarray expand_dims(NDarray a, int axis)
            => NumPy.Instance.expand_dims(a, axis);
        
        /// <summary>
        /// Remove single-dimensional entries from the shape of an array.
        /// </summary>
        /// <param name="a">
        /// Input data.
        /// </param>
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
        public static NDarray squeeze(NDarray a, int[] axis = null)
            => NumPy.Instance.squeeze(a, axis:axis);
        
        /// <summary>
        /// Return an array converted to a float type.
        /// </summary>
        /// <param name="a">
        /// The input array.
        /// </param>
        /// <param name="dtype">
        /// Float type code to coerce input array a.  If dtype is one of the
        /// ‘int’ dtypes, it is replaced with float64.
        /// </param>
        /// <returns>
        /// The input a as a float ndarray.
        /// </returns>
        public static NDarray asfarray(NDarray a, Dtype dtype = null)
            => NumPy.Instance.asfarray(a, dtype:dtype);
        
        /// <summary>
        /// Return an array converted to a float type.
        /// </summary>
        /// <param name="a">
        /// The input array.
        /// </param>
        /// <param name="dtype">
        /// Float type code to coerce input array a.  If dtype is one of the
        /// ‘int’ dtypes, it is replaced with float64.
        /// </param>
        /// <returns>
        /// The input a as a float ndarray.
        /// </returns>
        public static NDarray<T> asfarray<T>(T[] a, Dtype dtype = null)
            => NumPy.Instance.asfarray(a, dtype:dtype);
        
        /// <summary>
        /// Return an array converted to a float type.
        /// </summary>
        /// <param name="a">
        /// The input array.
        /// </param>
        /// <param name="dtype">
        /// Float type code to coerce input array a.  If dtype is one of the
        /// ‘int’ dtypes, it is replaced with float64.
        /// </param>
        /// <returns>
        /// The input a as a float ndarray.
        /// </returns>
        public static NDarray<T> asfarray<T>(T[,] a, Dtype dtype = null)
            => NumPy.Instance.asfarray(a, dtype:dtype);
        
        /// <summary>
        /// Return an array (ndim &gt;= 1) laid out in Fortran order in memory.
        /// </summary>
        /// <param name="a">
        /// Input array.
        /// </param>
        /// <param name="dtype">
        /// By default, the data-type is inferred from the input data.
        /// </param>
        /// <returns>
        /// The input a in Fortran, or column-major, order.
        /// </returns>
        public static NDarray asfortranarray(NDarray a, Dtype dtype = null)
            => NumPy.Instance.asfortranarray(a, dtype:dtype);
        
        /// <summary>
        /// Return an array (ndim &gt;= 1) laid out in Fortran order in memory.
        /// </summary>
        /// <param name="a">
        /// Input array.
        /// </param>
        /// <param name="dtype">
        /// By default, the data-type is inferred from the input data.
        /// </param>
        /// <returns>
        /// The input a in Fortran, or column-major, order.
        /// </returns>
        public static NDarray<T> asfortranarray<T>(T[] a, Dtype dtype = null)
            => NumPy.Instance.asfortranarray(a, dtype:dtype);
        
        /// <summary>
        /// Return an array (ndim &gt;= 1) laid out in Fortran order in memory.
        /// </summary>
        /// <param name="a">
        /// Input array.
        /// </param>
        /// <param name="dtype">
        /// By default, the data-type is inferred from the input data.
        /// </param>
        /// <returns>
        /// The input a in Fortran, or column-major, order.
        /// </returns>
        public static NDarray<T> asfortranarray<T>(T[,] a, Dtype dtype = null)
            => NumPy.Instance.asfortranarray(a, dtype:dtype);
        
        /// <summary>
        /// Convert the input to an array, checking for NaNs or Infs.
        /// </summary>
        /// <param name="a">
        /// Input data, in any form that can be converted to an array.  This
        /// includes lists, lists of tuples, tuples, tuples of tuples, tuples
        /// of lists and ndarrays.  Success requires no NaNs or Infs.
        /// </param>
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
        public static NDarray asarray_chkfinite(NDarray a, Dtype dtype = null, string order = null)
            => NumPy.Instance.asarray_chkfinite(a, dtype:dtype, order:order);
        
        /// <summary>
        /// Convert the input to an array, checking for NaNs or Infs.
        /// </summary>
        /// <param name="a">
        /// Input data, in any form that can be converted to an array.  This
        /// includes lists, lists of tuples, tuples, tuples of tuples, tuples
        /// of lists and ndarrays.  Success requires no NaNs or Infs.
        /// </param>
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
        public static NDarray<T> asarray_chkfinite<T>(T[] a, Dtype dtype = null, string order = null)
            => NumPy.Instance.asarray_chkfinite(a, dtype:dtype, order:order);
        
        /// <summary>
        /// Convert the input to an array, checking for NaNs or Infs.
        /// </summary>
        /// <param name="a">
        /// Input data, in any form that can be converted to an array.  This
        /// includes lists, lists of tuples, tuples, tuples of tuples, tuples
        /// of lists and ndarrays.  Success requires no NaNs or Infs.
        /// </param>
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
        public static NDarray<T> asarray_chkfinite<T>(T[,] a, Dtype dtype = null, string order = null)
            => NumPy.Instance.asarray_chkfinite(a, dtype:dtype, order:order);
        
        /// <summary>
        /// Convert an array of size 1 to its scalar equivalent.
        /// </summary>
        /// <param name="a">
        /// Input array of size 1.
        /// </param>
        /// <returns>
        /// Scalar representation of a. The output data type is the same type
        /// returned by the input’s item method.
        /// </returns>
        public static ValueType asscalar(NDarray a)
            => NumPy.Instance.asscalar(a);
        
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
        /// <param name="a">
        /// The object to be converted to a type-and-requirement-satisfying array.
        /// </param>
        /// <param name="dtype">
        /// The required data-type. If None preserve the current dtype. If your
        /// application requires the data to be in native byteorder, include
        /// a byteorder specification as a part of the dtype specification.
        /// </param>
        /// <param name="requirements">
        /// The requirements list can be any of the following
        /// </param>
        public static NDarray require(NDarray a, Dtype dtype, string[] requirements)
            => NumPy.Instance.require(a, dtype, requirements);
        
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
        /// <param name="a">
        /// The object to be converted to a type-and-requirement-satisfying array.
        /// </param>
        /// <param name="dtype">
        /// The required data-type. If None preserve the current dtype. If your
        /// application requires the data to be in native byteorder, include
        /// a byteorder specification as a part of the dtype specification.
        /// </param>
        /// <param name="requirements">
        /// The requirements list can be any of the following
        /// </param>
        public static NDarray<T> require<T>(T[] a, Dtype dtype, string[] requirements)
            => NumPy.Instance.require(a, dtype, requirements);
        
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
        /// <param name="a">
        /// The object to be converted to a type-and-requirement-satisfying array.
        /// </param>
        /// <param name="dtype">
        /// The required data-type. If None preserve the current dtype. If your
        /// application requires the data to be in native byteorder, include
        /// a byteorder specification as a part of the dtype specification.
        /// </param>
        /// <param name="requirements">
        /// The requirements list can be any of the following
        /// </param>
        public static NDarray<T> require<T>(T[,] a, Dtype dtype, string[] requirements)
            => NumPy.Instance.require(a, dtype, requirements);
        
        /// <summary>
        /// Join a sequence of arrays along an existing axis.
        /// 
        /// Notes
        /// 
        /// When one or more of the arrays to be concatenated is a MaskedArray,
        /// this function will return a MaskedArray object instead of an ndarray,
        /// but the input masks are not preserved. In cases where a MaskedArray
        /// is expected as input, use the ma.concatenate function from the masked
        /// array module instead.
        /// </summary>
        /// <param name="arys">
        /// The arrays must have the same shape, except in the dimension
        /// corresponding to axis (the first, by default).
        /// </param>
        /// <param name="axis">
        /// The axis along which the arrays will be joined.  If axis is None,
        /// arrays are flattened before use.  Default is 0.
        /// </param>
        /// <param name="@out">
        /// If provided, the destination to place the result. The shape must be
        /// correct, matching that of what concatenate would have returned if no
        /// out argument were specified.
        /// </param>
        /// <returns>
        /// The concatenated array.
        /// </returns>
        public static NDarray concatenate(NDarray[] arys, int? axis = null, NDarray @out = null)
            => NumPy.Instance.concatenate(arys, axis:axis, @out:@out);
        
        /// <summary>
        /// Join a sequence of arrays along a new axis.
        /// 
        /// The axis parameter specifies the index of the new axis in the dimensions
        /// of the result. For example, if axis=0 it will be the first dimension
        /// and if axis=-1 it will be the last dimension.
        /// </summary>
        /// <param name="arrays">
        /// Each array must have the same shape.
        /// </param>
        /// <param name="axis">
        /// The axis in the result array along which the input arrays are stacked.
        /// </param>
        /// <param name="@out">
        /// If provided, the destination to place the result. The shape must be
        /// correct, matching that of what stack would have returned if no
        /// out argument were specified.
        /// </param>
        /// <returns>
        /// The stacked array has one more dimension than the input arrays.
        /// </returns>
        public static NDarray stack(NDarray[] arrays, int? axis = null, NDarray @out = null)
            => NumPy.Instance.stack(arrays, axis:axis, @out:@out);
        
        /// <summary>
        /// Stack 1-D arrays as columns into a 2-D array.
        /// 
        /// Take a sequence of 1-D arrays and stack them as columns
        /// to make a single 2-D array. 2-D arrays are stacked as-is,
        /// just like with hstack.  1-D arrays are turned into 2-D columns
        /// first.
        /// </summary>
        /// <param name="tup">
        /// Arrays to stack. All of them must have the same first dimension.
        /// </param>
        /// <returns>
        /// The array formed by stacking the given arrays.
        /// </returns>
        public static NDarray column_stack(NDarray[] tup)
            => NumPy.Instance.column_stack(tup);
        
        /// <summary>
        /// Stack arrays in sequence depth wise (along third axis).
        /// 
        /// This is equivalent to concatenation along the third axis after 2-D arrays
        /// of shape (M,N) have been reshaped to (M,N,1) and 1-D arrays of shape
        /// (N,) have been reshaped to (1,N,1). Rebuilds arrays divided by
        /// dsplit.
        /// 
        /// This function makes most sense for arrays with up to 3 dimensions. For
        /// instance, for pixel-data with a height (first axis), width (second axis),
        /// and r/g/b channels (third axis). The functions concatenate, stack and
        /// block provide more general stacking and concatenation operations.
        /// </summary>
        /// <param name="tup">
        /// The arrays must have the same shape along all but the third axis.
        /// 1-D or 2-D arrays must have the same shape.
        /// </param>
        /// <returns>
        /// The array formed by stacking the given arrays, will be at least 3-D.
        /// </returns>
        public static NDarray dstack(NDarray[] tup)
            => NumPy.Instance.dstack(tup);
        
        /// <summary>
        /// Stack arrays in sequence horizontally (column wise).
        /// 
        /// This is equivalent to concatenation along the second axis, except for 1-D
        /// arrays where it concatenates along the first axis. Rebuilds arrays divided
        /// by hsplit.
        /// 
        /// This function makes most sense for arrays with up to 3 dimensions. For
        /// instance, for pixel-data with a height (first axis), width (second axis),
        /// and r/g/b channels (third axis). The functions concatenate, stack and
        /// block provide more general stacking and concatenation operations.
        /// </summary>
        /// <param name="tup">
        /// The arrays must have the same shape along all but the second axis,
        /// except 1-D arrays which can be any length.
        /// </param>
        /// <returns>
        /// The array formed by stacking the given arrays.
        /// </returns>
        public static NDarray hstack(NDarray[] tup)
            => NumPy.Instance.hstack(tup);
        
        /// <summary>
        /// Stack arrays in sequence vertically (row wise).
        /// 
        /// This is equivalent to concatenation along the first axis after 1-D arrays
        /// of shape (N,) have been reshaped to (1,N). Rebuilds arrays divided by
        /// vsplit.
        /// 
        /// This function makes most sense for arrays with up to 3 dimensions. For
        /// instance, for pixel-data with a height (first axis), width (second axis),
        /// and r/g/b channels (third axis). The functions concatenate, stack and
        /// block provide more general stacking and concatenation operations.
        /// </summary>
        /// <param name="tup">
        /// The arrays must have the same shape along all but the first axis.
        /// 1-D arrays must have the same length.
        /// </param>
        /// <returns>
        /// The array formed by stacking the given arrays, will be at least 2-D.
        /// </returns>
        public static NDarray vstack(NDarray[] tup)
            => NumPy.Instance.vstack(tup);
        
        /*
        /// <summary>
        /// Assemble an nd-array from nested lists of blocks.
        /// 
        /// Blocks in the innermost lists are concatenated (see concatenate) along
        /// the last dimension (-1), then these are concatenated along the
        /// second-last dimension (-2), and so on until the outermost list is reached.
        /// 
        /// Blocks can be of any dimension, but will not be broadcasted using the normal
        /// rules. Instead, leading axes of size 1 are inserted, to make block.ndim
        /// the same for all blocks. This is primarily useful for working with scalars,
        /// and means that code like np.block([v, 1]) is valid, where
        /// v.ndim == 1.
        /// 
        /// When the nested list is two levels deep, this allows block matrices to be
        /// constructed from their components.
        /// 
        /// Notes
        /// 
        /// When called with only scalars, np.block is equivalent to an ndarray
        /// call. So np.block([[1, 2], [3, 4]]) is equivalent to
        /// np.array([[1, 2], [3, 4]]).
        /// 
        /// This function does not enforce that the blocks lie on a fixed grid.
        /// np.block([[a, b], [c, d]]) is not restricted to arrays of the form:
        /// 
        /// But is also allowed to produce, for some a, b, c, d:
        /// 
        /// Since concatenation happens along the last axis first, block is _not_
        /// capable of producing the following directly:
        /// 
        /// Matlab’s “square bracket stacking”, [A, B, ...; p, q, ...], is
        /// equivalent to np.block([[A, B, ...], [p, q, ...]]).
        /// </summary>
        /// <param name="arrays">
        /// If passed a single ndarray or scalar (a nested list of depth 0), this
        /// is returned unmodified (and not copied).
        /// 
        /// Elements shapes must match along the appropriate axes (without
        /// broadcasting), but leading 1s will be prepended to the shape as
        /// necessary to make the dimensions match.
        /// </param>
        /// <returns>
        /// The array assembled from the given blocks.
        /// 
        /// The dimensionality of the output is equal to the greatest of:
        /// * the dimensionality of all the inputs
        /// * the depth to which the input list is nested
        /// </returns>
        public static NDarray block(nested list of array_like or scalars (but not tuples) arrays)
            => NumPy.Instance.block(arrays);
        */
        
        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        /// <param name="ary">
        /// Array to be divided into sub-arrays.
        /// </param>
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
        public static NDarray[] split(NDarray ary, int[] indices_or_sections, int? axis = null)
            => NumPy.Instance.split(ary, indices_or_sections, axis:axis);
        
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
        /// <param name="A">
        /// The input array.
        /// </param>
        /// <param name="reps">
        /// The number of repetitions of A along each axis.
        /// </param>
        /// <returns>
        /// The tiled output array.
        /// </returns>
        public static NDarray tile(NDarray A, NDarray reps)
            => NumPy.Instance.tile(A, reps);
        
        /// <summary>
        /// Repeat elements of an array.
        /// </summary>
        /// <param name="a">
        /// Input array.
        /// </param>
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
        public static NDarray repeat(NDarray a, int[] repeats, int? axis = null)
            => NumPy.Instance.repeat(a, repeats, axis:axis);
        
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
        /// <param name="arr">
        /// Input array.
        /// </param>
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
        public static NDarray delete(NDarray arr, Slice obj, int? axis = null)
            => NumPy.Instance.delete(arr, obj, axis:axis);
        
        /// <summary>
        /// Insert values along the given axis before the given indices.
        /// 
        /// Notes
        /// 
        /// Note that for higher dimensional inserts obj=0 behaves very different
        /// from obj=[0] just like arr[:,0,:] = values is different from
        /// arr[:,[0],:] = values.
        /// </summary>
        /// <param name="arr">
        /// Input array.
        /// </param>
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
        public static NDarray insert(NDarray arr, int obj, NDarray values, int? axis = null)
            => NumPy.Instance.insert(arr, obj, values, axis:axis);
        
        /// <summary>
        /// Insert values along the given axis before the given indices.
        /// 
        /// Notes
        /// 
        /// Note that for higher dimensional inserts obj=0 behaves very different
        /// from obj=[0] just like arr[:,0,:] = values is different from
        /// arr[:,[0],:] = values.
        /// </summary>
        /// <param name="arr">
        /// Input array.
        /// </param>
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
        public static NDarray<T> insert<T>(NDarray arr, int obj, T[] values, int? axis = null)
            => NumPy.Instance.insert(arr, obj, values, axis:axis);
        
        /// <summary>
        /// Insert values along the given axis before the given indices.
        /// 
        /// Notes
        /// 
        /// Note that for higher dimensional inserts obj=0 behaves very different
        /// from obj=[0] just like arr[:,0,:] = values is different from
        /// arr[:,[0],:] = values.
        /// </summary>
        /// <param name="arr">
        /// Input array.
        /// </param>
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
        public static NDarray<T> insert<T>(NDarray arr, int obj, T[,] values, int? axis = null)
            => NumPy.Instance.insert(arr, obj, values, axis:axis);
        
        /// <summary>
        /// Append values to the end of an array.
        /// </summary>
        /// <param name="arr">
        /// Values are appended to a copy of this array.
        /// </param>
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
        public static NDarray append(NDarray arr, NDarray values, int? axis = null)
            => NumPy.Instance.append(arr, values, axis:axis);
        
        /// <summary>
        /// Append values to the end of an array.
        /// </summary>
        /// <param name="arr">
        /// Values are appended to a copy of this array.
        /// </param>
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
        public static NDarray<T> append<T>(NDarray arr, T[] values, int? axis = null)
            => NumPy.Instance.append(arr, values, axis:axis);
        
        /// <summary>
        /// Append values to the end of an array.
        /// </summary>
        /// <param name="arr">
        /// Values are appended to a copy of this array.
        /// </param>
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
        public static NDarray<T> append<T>(NDarray arr, T[,] values, int? axis = null)
            => NumPy.Instance.append(arr, values, axis:axis);
        
        /// <summary>
        /// Return a new array with the specified shape.
        /// 
        /// If the new array is larger than the original array, then the new
        /// array is filled with repeated copies of a.  Note that this behavior
        /// is different from a.resize(new_shape) which fills with zeros instead
        /// of repeated copies of a.
        /// 
        /// Notes
        /// 
        /// Warning: This functionality does not consider axes separately,
        /// i.e. it does not apply interpolation/extrapolation.
        /// It fills the return array with the required number of elements, taken
        /// from a as they are laid out in memory, disregarding strides and axes.
        /// (This is in case the new shape is smaller. For larger, see above.)
        /// This functionality is therefore not suitable to resize images,
        /// or data where each axis represents a separate and distinct entity.
        /// </summary>
        /// <param name="a">
        /// Array to be resized.
        /// </param>
        /// <param name="new_shape">
        /// Shape of resized array.
        /// </param>
        /// <returns>
        /// The new array is formed from the data in the old array, repeated
        /// if necessary to fill out the required number of elements.  The
        /// data are repeated in the order that they are stored in memory.
        /// </returns>
        public static NDarray resize(NDarray a, Shape new_shape)
            => NumPy.Instance.resize(a, new_shape);
        
        /// <summary>
        /// Trim the leading and/or trailing zeros from a 1-D array or sequence.
        /// </summary>
        /// <param name="filt">
        /// Input array.
        /// </param>
        /// <param name="trim">
        /// A string with ‘f’ representing trim from front and ‘b’ to trim from
        /// back. Default is ‘fb’, trim zeros from both front and back of the
        /// array.
        /// </param>
        /// <returns>
        /// The result of trimming the input. The input data type is preserved.
        /// </returns>
        public static NDarray trim_zeros(NDarray filt, string trim = null)
            => NumPy.Instance.trim_zeros(filt, trim:trim);
        
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
        /// <param name="ar">
        /// Input array. Unless axis is specified, this will be flattened if it
        /// is not already 1-D.
        /// </param>
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
   at CodeMinion.Core.CodeGenerator.GenerateStaticApiRedirection(StaticApi api, Declaration decl, CodeWriter s) in D:\dev\CodeMinion\src\CodeMinion.Core\CodeGenerator.cs:line 105
   at CodeMinion.Core.CodeGenerator.<>c__DisplayClass52_0.<GenerateStaticApi>b__1() in D:\dev\CodeMinion\src\CodeMinion.Core\CodeGenerator.cs:line 405
        ----------------------------
        Declaration JSON:
        {
  "Arguments": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "ar",
      "Type": "NDarray",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "Input array. Unless axis is specified, this will be flattened if it\nis not already 1-D.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": false
    },
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
  "ForwardToStaticImpl": null,
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
        /// <param name="ar">
        /// Input array. Unless axis is specified, this will be flattened if it
        /// is not already 1-D.
        /// </param>
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
   at CodeMinion.Core.CodeGenerator.GenerateStaticApiRedirection(StaticApi api, Declaration decl, CodeWriter s) in D:\dev\CodeMinion\src\CodeMinion.Core\CodeGenerator.cs:line 105
   at CodeMinion.Core.CodeGenerator.<>c__DisplayClass52_0.<GenerateStaticApi>b__1() in D:\dev\CodeMinion\src\CodeMinion.Core\CodeGenerator.cs:line 405
        ----------------------------
        Declaration JSON:
        {
  "Arguments": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "ar",
      "Type": "T[]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "Input array. Unless axis is specified, this will be flattened if it\nis not already 1-D.",
      "ConvertToSharpType": "NDarray",
      "Position": 0,
      "IsReturnValue": false
    },
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
  "Generics": [
    "T"
  ],
  "ForwardToStaticImpl": null,
  "Name": "unique",
  "ClassName": "numpy",
  "Returns": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "unique",
      "Type": "NDarray<T>",
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
        /// <param name="ar">
        /// Input array. Unless axis is specified, this will be flattened if it
        /// is not already 1-D.
        /// </param>
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
   at CodeMinion.Core.CodeGenerator.GenerateStaticApiRedirection(StaticApi api, Declaration decl, CodeWriter s) in D:\dev\CodeMinion\src\CodeMinion.Core\CodeGenerator.cs:line 105
   at CodeMinion.Core.CodeGenerator.<>c__DisplayClass52_0.<GenerateStaticApi>b__1() in D:\dev\CodeMinion\src\CodeMinion.Core\CodeGenerator.cs:line 405
        ----------------------------
        Declaration JSON:
        {
  "Arguments": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "ar",
      "Type": "T[,]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "Input array. Unless axis is specified, this will be flattened if it\nis not already 1-D.",
      "ConvertToSharpType": "NDarray",
      "Position": 0,
      "IsReturnValue": false
    },
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
  "Generics": [
    "T"
  ],
  "ForwardToStaticImpl": null,
  "Name": "unique",
  "ClassName": "numpy",
  "Returns": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "unique",
      "Type": "NDarray<T>",
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
        /// <param name="m">
        /// Input array.
        /// </param>
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
        public static NDarray flip(NDarray m, int[] axis = null)
            => NumPy.Instance.flip(m, axis:axis);
        
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
        /// <param name="m">
        /// Input array, must be at least 2-D.
        /// </param>
        /// <returns>
        /// A view of m with the columns reversed.  Since a view
        /// is returned, this operation is .
        /// </returns>
        public static NDarray fliplr(NDarray m)
            => NumPy.Instance.fliplr(m);
        
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
        /// <param name="m">
        /// Input array.
        /// </param>
        /// <returns>
        /// A view of m with the rows reversed.  Since a view is
        /// returned, this operation is .
        /// </returns>
        public static NDarray flipud(NDarray m)
            => NumPy.Instance.flipud(m);
        
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
        /// <param name="a">
        /// Input array.
        /// </param>
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
        public static NDarray roll(NDarray a, int[] shift, int[] axis = null)
            => NumPy.Instance.roll(a, shift, axis:axis);
        
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
        /// <param name="m">
        /// Array of two or more dimensions.
        /// </param>
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
        public static NDarray rot90(NDarray m, int k, int[] axes = null)
            => NumPy.Instance.rot90(m, k, axes);
        
        
    }
}
