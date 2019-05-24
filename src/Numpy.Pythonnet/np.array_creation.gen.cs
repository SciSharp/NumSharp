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
        /// Return a new array of given shape and type, without initializing entries.
        /// 
        /// Notes
        /// 
        /// empty, unlike zeros, does not set the array values to zero,
        /// and may therefore be marginally faster.  On the other hand, it requires
        /// the user to manually set all the values in the array, and should be
        /// used with caution.
        /// </summary>
        /// <param name="shape">
        /// Shape of the empty array, e.g., (2, 3) or 2.
        /// </param>
        /// <param name="dtype">
        /// Desired output data-type for the array, e.g, numpy.int8. Default is
        /// numpy.float64.
        /// </param>
        /// <param name="order">
        /// Whether to store multi-dimensional data in row-major
        /// (C-style) or column-major (Fortran-style) order in
        /// memory.
        /// </param>
        /// <returns>
        /// Array of uninitialized (arbitrary) data of the given shape, dtype, and
        /// order.  Object arrays will be initialized to None.
        /// </returns>
        public static NDarray empty(Shape shape, Dtype dtype = null, string order = null)
            => NumPy.Instance.empty(shape, dtype:dtype, order:order);
        
        /// <summary>
        /// Return a new array with the same shape and type as a given array.
        /// 
        /// Notes
        /// 
        /// This function does not initialize the returned array; to do that use
        /// zeros_like or ones_like instead.  It may be marginally faster than
        /// the functions that do set the array values.
        /// </summary>
        /// <param name="prototype">
        /// The shape and data-type of prototype define these same attributes
        /// of the returned array.
        /// </param>
        /// <param name="dtype">
        /// Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        /// Overrides the memory layout of the result. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if prototype is Fortran
        /// contiguous, ‘C’ otherwise. ‘K’ means match the layout of prototype
        /// as closely as possible.
        /// </param>
        /// <param name="subok">
        /// If True, then the newly created array will use the sub-class
        /// type of ‘a’, otherwise it will be a base-class array. Defaults
        /// to True.
        /// </param>
        /// <returns>
        /// Array of uninitialized (arbitrary) data with the same
        /// shape and type as prototype.
        /// </returns>
        public static NDarray empty_like(NDarray prototype, Dtype dtype = null, string order = null, bool? subok = null)
            => NumPy.Instance.empty_like(prototype, dtype:dtype, order:order, subok:subok);
        
        /// <summary>
        /// Return a new array with the same shape and type as a given array.
        /// 
        /// Notes
        /// 
        /// This function does not initialize the returned array; to do that use
        /// zeros_like or ones_like instead.  It may be marginally faster than
        /// the functions that do set the array values.
        /// </summary>
        /// <param name="prototype">
        /// The shape and data-type of prototype define these same attributes
        /// of the returned array.
        /// </param>
        /// <param name="dtype">
        /// Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        /// Overrides the memory layout of the result. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if prototype is Fortran
        /// contiguous, ‘C’ otherwise. ‘K’ means match the layout of prototype
        /// as closely as possible.
        /// </param>
        /// <param name="subok">
        /// If True, then the newly created array will use the sub-class
        /// type of ‘a’, otherwise it will be a base-class array. Defaults
        /// to True.
        /// </param>
        /// <returns>
        /// Array of uninitialized (arbitrary) data with the same
        /// shape and type as prototype.
        /// </returns>
        public static NDarray<T> empty_like<T>(T[] prototype, Dtype dtype = null, string order = null, bool? subok = null)
            => NumPy.Instance.empty_like(prototype, dtype:dtype, order:order, subok:subok);
        
        /// <summary>
        /// Return a new array with the same shape and type as a given array.
        /// 
        /// Notes
        /// 
        /// This function does not initialize the returned array; to do that use
        /// zeros_like or ones_like instead.  It may be marginally faster than
        /// the functions that do set the array values.
        /// </summary>
        /// <param name="prototype">
        /// The shape and data-type of prototype define these same attributes
        /// of the returned array.
        /// </param>
        /// <param name="dtype">
        /// Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        /// Overrides the memory layout of the result. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if prototype is Fortran
        /// contiguous, ‘C’ otherwise. ‘K’ means match the layout of prototype
        /// as closely as possible.
        /// </param>
        /// <param name="subok">
        /// If True, then the newly created array will use the sub-class
        /// type of ‘a’, otherwise it will be a base-class array. Defaults
        /// to True.
        /// </param>
        /// <returns>
        /// Array of uninitialized (arbitrary) data with the same
        /// shape and type as prototype.
        /// </returns>
        public static NDarray<T> empty_like<T>(T[,] prototype, Dtype dtype = null, string order = null, bool? subok = null)
            => NumPy.Instance.empty_like(prototype, dtype:dtype, order:order, subok:subok);
        
        /// <summary>
        /// Return a 2-D array with ones on the diagonal and zeros elsewhere.
        /// </summary>
        /// <param name="N">
        /// Number of rows in the output.
        /// </param>
        /// <param name="M">
        /// Number of columns in the output. If None, defaults to N.
        /// </param>
        /// <param name="k">
        /// Index of the diagonal: 0 (the default) refers to the main diagonal,
        /// a positive value refers to an upper diagonal, and a negative value
        /// to a lower diagonal.
        /// </param>
        /// <param name="dtype">
        /// Data-type of the returned array.
        /// </param>
        /// <param name="order">
        /// Whether the output should be stored in row-major (C-style) or
        /// column-major (Fortran-style) order in memory.
        /// </param>
        /// <returns>
        /// An array where all elements are equal to zero, except for the k-th
        /// diagonal, whose values are equal to one.
        /// </returns>
        public static NDarray eye(int N, int? M = null, int? k = null, Dtype dtype = null, string order = null)
            => NumPy.Instance.eye(N, M:M, k:k, dtype:dtype, order:order);
        
        /// <summary>
        /// Return the identity array.
        /// 
        /// The identity array is a square array with ones on
        /// the main diagonal.
        /// </summary>
        /// <param name="n">
        /// Number of rows (and columns) in n x n output.
        /// </param>
        /// <param name="dtype">
        /// Data-type of the output.  Defaults to float.
        /// </param>
        /// <returns>
        /// n x n array with its main diagonal set to one,
        /// and all other elements 0.
        /// </returns>
        public static NDarray identity(int n, Dtype dtype = null)
            => NumPy.Instance.identity(n, dtype:dtype);
        
        /// <summary>
        /// Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shape">
        /// Shape of the new array, e.g., (2, 3) or 2.
        /// </param>
        /// <param name="dtype">
        /// The desired data-type for the array, e.g., numpy.int8.  Default is
        /// numpy.float64.
        /// </param>
        /// <param name="order">
        /// Whether to store multi-dimensional data in row-major
        /// (C-style) or column-major (Fortran-style) order in
        /// memory.
        /// </param>
        /// <returns>
        /// Array of ones with the given shape, dtype, and order.
        /// </returns>
        public static NDarray ones(Shape shape, Dtype dtype = null, string order = null)
            => NumPy.Instance.ones(shape, dtype:dtype, order:order);
        
        /// <summary>
        /// Return an array of ones with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        /// The shape and data-type of a define these same attributes of
        /// the returned array.
        /// </param>
        /// <param name="dtype">
        /// Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        /// Overrides the memory layout of the result. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        /// ‘C’ otherwise. ‘K’ means match the layout of a as closely
        /// as possible.
        /// </param>
        /// <param name="subok">
        /// If True, then the newly created array will use the sub-class
        /// type of ‘a’, otherwise it will be a base-class array. Defaults
        /// to True.
        /// </param>
        /// <returns>
        /// Array of ones with the same shape and type as a.
        /// </returns>
        public static NDarray ones_like(NDarray a, Dtype dtype = null, string order = null, bool? subok = null)
            => NumPy.Instance.ones_like(a, dtype:dtype, order:order, subok:subok);
        
        /// <summary>
        /// Return an array of ones with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        /// The shape and data-type of a define these same attributes of
        /// the returned array.
        /// </param>
        /// <param name="dtype">
        /// Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        /// Overrides the memory layout of the result. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        /// ‘C’ otherwise. ‘K’ means match the layout of a as closely
        /// as possible.
        /// </param>
        /// <param name="subok">
        /// If True, then the newly created array will use the sub-class
        /// type of ‘a’, otherwise it will be a base-class array. Defaults
        /// to True.
        /// </param>
        /// <returns>
        /// Array of ones with the same shape and type as a.
        /// </returns>
        public static NDarray<T> ones_like<T>(T[] a, Dtype dtype = null, string order = null, bool? subok = null)
            => NumPy.Instance.ones_like(a, dtype:dtype, order:order, subok:subok);
        
        /// <summary>
        /// Return an array of ones with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        /// The shape and data-type of a define these same attributes of
        /// the returned array.
        /// </param>
        /// <param name="dtype">
        /// Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        /// Overrides the memory layout of the result. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        /// ‘C’ otherwise. ‘K’ means match the layout of a as closely
        /// as possible.
        /// </param>
        /// <param name="subok">
        /// If True, then the newly created array will use the sub-class
        /// type of ‘a’, otherwise it will be a base-class array. Defaults
        /// to True.
        /// </param>
        /// <returns>
        /// Array of ones with the same shape and type as a.
        /// </returns>
        public static NDarray<T> ones_like<T>(T[,] a, Dtype dtype = null, string order = null, bool? subok = null)
            => NumPy.Instance.ones_like(a, dtype:dtype, order:order, subok:subok);
        
        /// <summary>
        /// Return a new array of given shape and type, filled with zeros.
        /// </summary>
        /// <param name="shape">
        /// Shape of the new array, e.g., (2, 3) or 2.
        /// </param>
        /// <param name="dtype">
        /// The desired data-type for the array, e.g., numpy.int8.  Default is
        /// numpy.float64.
        /// </param>
        /// <param name="order">
        /// Whether to store multi-dimensional data in row-major
        /// (C-style) or column-major (Fortran-style) order in
        /// memory.
        /// </param>
        /// <returns>
        /// Array of zeros with the given shape, dtype, and order.
        /// </returns>
        public static NDarray zeros(Shape shape, Dtype dtype = null, string order = null)
            => NumPy.Instance.zeros(shape, dtype:dtype, order:order);
        
        /// <summary>
        /// Return an array of zeros with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        /// The shape and data-type of a define these same attributes of
        /// the returned array.
        /// </param>
        /// <param name="dtype">
        /// Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        /// Overrides the memory layout of the result. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        /// ‘C’ otherwise. ‘K’ means match the layout of a as closely
        /// as possible.
        /// </param>
        /// <param name="subok">
        /// If True, then the newly created array will use the sub-class
        /// type of ‘a’, otherwise it will be a base-class array. Defaults
        /// to True.
        /// </param>
        /// <returns>
        /// Array of zeros with the same shape and type as a.
        /// </returns>
        public static NDarray zeros_like(NDarray a, Dtype dtype = null, string order = null, bool? subok = null)
            => NumPy.Instance.zeros_like(a, dtype:dtype, order:order, subok:subok);
        
        /// <summary>
        /// Return an array of zeros with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        /// The shape and data-type of a define these same attributes of
        /// the returned array.
        /// </param>
        /// <param name="dtype">
        /// Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        /// Overrides the memory layout of the result. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        /// ‘C’ otherwise. ‘K’ means match the layout of a as closely
        /// as possible.
        /// </param>
        /// <param name="subok">
        /// If True, then the newly created array will use the sub-class
        /// type of ‘a’, otherwise it will be a base-class array. Defaults
        /// to True.
        /// </param>
        /// <returns>
        /// Array of zeros with the same shape and type as a.
        /// </returns>
        public static NDarray<T> zeros_like<T>(T[] a, Dtype dtype = null, string order = null, bool? subok = null)
            => NumPy.Instance.zeros_like(a, dtype:dtype, order:order, subok:subok);
        
        /// <summary>
        /// Return an array of zeros with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        /// The shape and data-type of a define these same attributes of
        /// the returned array.
        /// </param>
        /// <param name="dtype">
        /// Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        /// Overrides the memory layout of the result. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        /// ‘C’ otherwise. ‘K’ means match the layout of a as closely
        /// as possible.
        /// </param>
        /// <param name="subok">
        /// If True, then the newly created array will use the sub-class
        /// type of ‘a’, otherwise it will be a base-class array. Defaults
        /// to True.
        /// </param>
        /// <returns>
        /// Array of zeros with the same shape and type as a.
        /// </returns>
        public static NDarray<T> zeros_like<T>(T[,] a, Dtype dtype = null, string order = null, bool? subok = null)
            => NumPy.Instance.zeros_like(a, dtype:dtype, order:order, subok:subok);
        
        /// <summary>
        /// Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="shape">
        /// Shape of the new array, e.g., (2, 3) or 2.
        /// </param>
        /// <param name="fill_value">
        /// Fill value.
        /// </param>
        /// <param name="order">
        /// Whether to store multidimensional data in C- or Fortran-contiguous
        /// (row- or column-wise) order in memory.
        /// </param>
        /// <returns>
        /// Array of fill_value with the given shape, dtype, and order.
        /// </returns>
        public static NDarray full(Shape shape, ValueType fill_value, Dtype dtype = null, string order = null)
            => NumPy.Instance.full(shape, fill_value, dtype:dtype, order:order);
        
        /// <summary>
        /// Return a full array with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        /// The shape and data-type of a define these same attributes of
        /// the returned array.
        /// </param>
        /// <param name="fill_value">
        /// Fill value.
        /// </param>
        /// <param name="dtype">
        /// Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        /// Overrides the memory layout of the result. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        /// ‘C’ otherwise. ‘K’ means match the layout of a as closely
        /// as possible.
        /// </param>
        /// <param name="subok">
        /// If True, then the newly created array will use the sub-class
        /// type of ‘a’, otherwise it will be a base-class array. Defaults
        /// to True.
        /// </param>
        /// <returns>
        /// Array of fill_value with the same shape and type as a.
        /// </returns>
        public static NDarray full_like(NDarray a, ValueType fill_value, Dtype dtype = null, string order = null, bool? subok = null)
            => NumPy.Instance.full_like(a, fill_value, dtype:dtype, order:order, subok:subok);
        
        /// <summary>
        /// Return a full array with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        /// The shape and data-type of a define these same attributes of
        /// the returned array.
        /// </param>
        /// <param name="fill_value">
        /// Fill value.
        /// </param>
        /// <param name="dtype">
        /// Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        /// Overrides the memory layout of the result. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        /// ‘C’ otherwise. ‘K’ means match the layout of a as closely
        /// as possible.
        /// </param>
        /// <param name="subok">
        /// If True, then the newly created array will use the sub-class
        /// type of ‘a’, otherwise it will be a base-class array. Defaults
        /// to True.
        /// </param>
        /// <returns>
        /// Array of fill_value with the same shape and type as a.
        /// </returns>
        public static NDarray<T> full_like<T>(T[] a, ValueType fill_value, Dtype dtype = null, string order = null, bool? subok = null)
            => NumPy.Instance.full_like(a, fill_value, dtype:dtype, order:order, subok:subok);
        
        /// <summary>
        /// Return a full array with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        /// The shape and data-type of a define these same attributes of
        /// the returned array.
        /// </param>
        /// <param name="fill_value">
        /// Fill value.
        /// </param>
        /// <param name="dtype">
        /// Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        /// Overrides the memory layout of the result. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        /// ‘C’ otherwise. ‘K’ means match the layout of a as closely
        /// as possible.
        /// </param>
        /// <param name="subok">
        /// If True, then the newly created array will use the sub-class
        /// type of ‘a’, otherwise it will be a base-class array. Defaults
        /// to True.
        /// </param>
        /// <returns>
        /// Array of fill_value with the same shape and type as a.
        /// </returns>
        public static NDarray<T> full_like<T>(T[,] a, ValueType fill_value, Dtype dtype = null, string order = null, bool? subok = null)
            => NumPy.Instance.full_like(a, fill_value, dtype:dtype, order:order, subok:subok);
        
        /// <summary>
        /// Create an array.
        /// 
        /// Notes
        /// 
        /// When order is ‘A’ and object is an array in neither ‘C’ nor ‘F’ order,
        /// and a copy is forced by a change in dtype, then the order of the result is
        /// not necessarily ‘C’ as expected. This is likely a bug.
        /// </summary>
        /// <param name="@object">
        /// An array, any object exposing the array interface, an object whose
        /// __array__ method returns an array, or any (nested) sequence.
        /// </param>
        /// <param name="dtype">
        /// The desired data-type for the array.  If not given, then the type will
        /// be determined as the minimum type required to hold the objects in the
        /// sequence.  This argument can only be used to ‘upcast’ the array.  For
        /// downcasting, use the .astype(t) method.
        /// </param>
        /// <param name="copy">
        /// If true (default), then the object is copied.  Otherwise, a copy will
        /// only be made if __array__ returns a copy, if obj is a nested sequence,
        /// or if a copy is needed to satisfy any of the other requirements
        /// (dtype, order, etc.).
        /// </param>
        /// <param name="order">
        /// Specify the memory layout of the array. If object is not an array, the
        /// newly created array will be in C order (row major) unless ‘F’ is
        /// specified, in which case it will be in Fortran order (column major).
        /// If object is an array the following holds.
        /// 
        /// When copy=False and a copy is made for other reasons, the result is
        /// the same as if copy=True, with some exceptions for A, see the
        /// Notes section. The default order is ‘K’.
        /// </param>
        /// <param name="subok">
        /// If True, then sub-classes will be passed-through, otherwise
        /// the returned array will be forced to be a base-class array (default).
        /// </param>
        /// <param name="ndmin">
        /// Specifies the minimum number of dimensions that the resulting
        /// array should have.  Ones will be pre-pended to the shape as
        /// needed to meet this requirement.
        /// </param>
        /// <returns>
        /// An array object satisfying the specified requirements.
        /// </returns>
        public static NDarray array(NDarray @object, Dtype dtype = null, bool? copy = null, string order = null, bool? subok = null, int? ndmin = null)
            => NumPy.Instance.array(@object, dtype:dtype, copy:copy, order:order, subok:subok, ndmin:ndmin);
        
        /// <summary>
        /// Create an array.
        /// 
        /// Notes
        /// 
        /// When order is ‘A’ and object is an array in neither ‘C’ nor ‘F’ order,
        /// and a copy is forced by a change in dtype, then the order of the result is
        /// not necessarily ‘C’ as expected. This is likely a bug.
        /// </summary>
        /// <param name="@object">
        /// An array, any object exposing the array interface, an object whose
        /// __array__ method returns an array, or any (nested) sequence.
        /// </param>
        /// <param name="dtype">
        /// The desired data-type for the array.  If not given, then the type will
        /// be determined as the minimum type required to hold the objects in the
        /// sequence.  This argument can only be used to ‘upcast’ the array.  For
        /// downcasting, use the .astype(t) method.
        /// </param>
        /// <param name="copy">
        /// If true (default), then the object is copied.  Otherwise, a copy will
        /// only be made if __array__ returns a copy, if obj is a nested sequence,
        /// or if a copy is needed to satisfy any of the other requirements
        /// (dtype, order, etc.).
        /// </param>
        /// <param name="order">
        /// Specify the memory layout of the array. If object is not an array, the
        /// newly created array will be in C order (row major) unless ‘F’ is
        /// specified, in which case it will be in Fortran order (column major).
        /// If object is an array the following holds.
        /// 
        /// When copy=False and a copy is made for other reasons, the result is
        /// the same as if copy=True, with some exceptions for A, see the
        /// Notes section. The default order is ‘K’.
        /// </param>
        /// <param name="subok">
        /// If True, then sub-classes will be passed-through, otherwise
        /// the returned array will be forced to be a base-class array (default).
        /// </param>
        /// <param name="ndmin">
        /// Specifies the minimum number of dimensions that the resulting
        /// array should have.  Ones will be pre-pended to the shape as
        /// needed to meet this requirement.
        /// </param>
        /// <returns>
        /// An array object satisfying the specified requirements.
        /// </returns>
        public static NDarray<T> array<T>(T[] @object, Dtype dtype = null, bool? copy = null, string order = null, bool? subok = null, int? ndmin = null)
            => NumPy.Instance.array(@object, dtype:dtype, copy:copy, order:order, subok:subok, ndmin:ndmin);
        
        /// <summary>
        /// Create an array.
        /// 
        /// Notes
        /// 
        /// When order is ‘A’ and object is an array in neither ‘C’ nor ‘F’ order,
        /// and a copy is forced by a change in dtype, then the order of the result is
        /// not necessarily ‘C’ as expected. This is likely a bug.
        /// </summary>
        /// <param name="@object">
        /// An array, any object exposing the array interface, an object whose
        /// __array__ method returns an array, or any (nested) sequence.
        /// </param>
        /// <param name="dtype">
        /// The desired data-type for the array.  If not given, then the type will
        /// be determined as the minimum type required to hold the objects in the
        /// sequence.  This argument can only be used to ‘upcast’ the array.  For
        /// downcasting, use the .astype(t) method.
        /// </param>
        /// <param name="copy">
        /// If true (default), then the object is copied.  Otherwise, a copy will
        /// only be made if __array__ returns a copy, if obj is a nested sequence,
        /// or if a copy is needed to satisfy any of the other requirements
        /// (dtype, order, etc.).
        /// </param>
        /// <param name="order">
        /// Specify the memory layout of the array. If object is not an array, the
        /// newly created array will be in C order (row major) unless ‘F’ is
        /// specified, in which case it will be in Fortran order (column major).
        /// If object is an array the following holds.
        /// 
        /// When copy=False and a copy is made for other reasons, the result is
        /// the same as if copy=True, with some exceptions for A, see the
        /// Notes section. The default order is ‘K’.
        /// </param>
        /// <param name="subok">
        /// If True, then sub-classes will be passed-through, otherwise
        /// the returned array will be forced to be a base-class array (default).
        /// </param>
        /// <param name="ndmin">
        /// Specifies the minimum number of dimensions that the resulting
        /// array should have.  Ones will be pre-pended to the shape as
        /// needed to meet this requirement.
        /// </param>
        /// <returns>
        /// An array object satisfying the specified requirements.
        /// </returns>
        public static NDarray<T> array<T>(T[,] @object, Dtype dtype = null, bool? copy = null, string order = null, bool? subok = null, int? ndmin = null)
            => NumPy.Instance.array(@object, dtype:dtype, copy:copy, order:order, subok:subok, ndmin:ndmin);
        
        /// <summary>
        /// Convert the input to an array.
        /// </summary>
        /// <param name="a">
        /// Input data, in any form that can be converted to an array.  This
        /// includes lists, lists of tuples, tuples, tuples of tuples, tuples
        /// of lists and ndarrays.
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
        /// is already an ndarray with matching dtype and order.  If a is a
        /// subclass of ndarray, a base class ndarray is returned.
        /// </returns>
        public static NDarray asarray(NDarray a, Dtype dtype = null, string order = null)
            => NumPy.Instance.asarray(a, dtype:dtype, order:order);
        
        /// <summary>
        /// Convert the input to an array.
        /// </summary>
        /// <param name="a">
        /// Input data, in any form that can be converted to an array.  This
        /// includes lists, lists of tuples, tuples, tuples of tuples, tuples
        /// of lists and ndarrays.
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
        /// is already an ndarray with matching dtype and order.  If a is a
        /// subclass of ndarray, a base class ndarray is returned.
        /// </returns>
        public static NDarray<T> asarray<T>(T[] a, Dtype dtype = null, string order = null)
            => NumPy.Instance.asarray(a, dtype:dtype, order:order);
        
        /// <summary>
        /// Convert the input to an array.
        /// </summary>
        /// <param name="a">
        /// Input data, in any form that can be converted to an array.  This
        /// includes lists, lists of tuples, tuples, tuples of tuples, tuples
        /// of lists and ndarrays.
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
        /// is already an ndarray with matching dtype and order.  If a is a
        /// subclass of ndarray, a base class ndarray is returned.
        /// </returns>
        public static NDarray<T> asarray<T>(T[,] a, Dtype dtype = null, string order = null)
            => NumPy.Instance.asarray(a, dtype:dtype, order:order);
        
        /// <summary>
        /// Convert the input to an ndarray, but pass ndarray subclasses through.
        /// </summary>
        /// <param name="a">
        /// Input data, in any form that can be converted to an array.  This
        /// includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        /// tuples of lists, and ndarrays.
        /// </param>
        /// <param name="dtype">
        /// By default, the data-type is inferred from the input data.
        /// </param>
        /// <param name="order">
        /// Whether to use row-major (C-style) or column-major
        /// (Fortran-style) memory representation.  Defaults to ‘C’.
        /// </param>
        /// <returns>
        /// Array interpretation of a.  If a is an ndarray or a subclass
        /// of ndarray, it is returned as-is and no copy is performed.
        /// </returns>
        public static NDarray asanyarray(NDarray a, Dtype dtype = null, string order = null)
            => NumPy.Instance.asanyarray(a, dtype:dtype, order:order);
        
        /// <summary>
        /// Convert the input to an ndarray, but pass ndarray subclasses through.
        /// </summary>
        /// <param name="a">
        /// Input data, in any form that can be converted to an array.  This
        /// includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        /// tuples of lists, and ndarrays.
        /// </param>
        /// <param name="dtype">
        /// By default, the data-type is inferred from the input data.
        /// </param>
        /// <param name="order">
        /// Whether to use row-major (C-style) or column-major
        /// (Fortran-style) memory representation.  Defaults to ‘C’.
        /// </param>
        /// <returns>
        /// Array interpretation of a.  If a is an ndarray or a subclass
        /// of ndarray, it is returned as-is and no copy is performed.
        /// </returns>
        public static NDarray<T> asanyarray<T>(T[] a, Dtype dtype = null, string order = null)
            => NumPy.Instance.asanyarray(a, dtype:dtype, order:order);
        
        /// <summary>
        /// Convert the input to an ndarray, but pass ndarray subclasses through.
        /// </summary>
        /// <param name="a">
        /// Input data, in any form that can be converted to an array.  This
        /// includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        /// tuples of lists, and ndarrays.
        /// </param>
        /// <param name="dtype">
        /// By default, the data-type is inferred from the input data.
        /// </param>
        /// <param name="order">
        /// Whether to use row-major (C-style) or column-major
        /// (Fortran-style) memory representation.  Defaults to ‘C’.
        /// </param>
        /// <returns>
        /// Array interpretation of a.  If a is an ndarray or a subclass
        /// of ndarray, it is returned as-is and no copy is performed.
        /// </returns>
        public static NDarray<T> asanyarray<T>(T[,] a, Dtype dtype = null, string order = null)
            => NumPy.Instance.asanyarray(a, dtype:dtype, order:order);
        
        /// <summary>
        /// Return a contiguous array (ndim &gt;= 1) in memory (C order).
        /// </summary>
        /// <param name="a">
        /// Input array.
        /// </param>
        /// <param name="dtype">
        /// Data-type of returned array.
        /// </param>
        /// <returns>
        /// Contiguous array of same shape and content as a, with type dtype
        /// if specified.
        /// </returns>
        public static NDarray ascontiguousarray(NDarray a, Dtype dtype = null)
            => NumPy.Instance.ascontiguousarray(a, dtype:dtype);
        
        /// <summary>
        /// Return a contiguous array (ndim &gt;= 1) in memory (C order).
        /// </summary>
        /// <param name="a">
        /// Input array.
        /// </param>
        /// <param name="dtype">
        /// Data-type of returned array.
        /// </param>
        /// <returns>
        /// Contiguous array of same shape and content as a, with type dtype
        /// if specified.
        /// </returns>
        public static NDarray<T> ascontiguousarray<T>(T[] a, Dtype dtype = null)
            => NumPy.Instance.ascontiguousarray(a, dtype:dtype);
        
        /// <summary>
        /// Return a contiguous array (ndim &gt;= 1) in memory (C order).
        /// </summary>
        /// <param name="a">
        /// Input array.
        /// </param>
        /// <param name="dtype">
        /// Data-type of returned array.
        /// </param>
        /// <returns>
        /// Contiguous array of same shape and content as a, with type dtype
        /// if specified.
        /// </returns>
        public static NDarray<T> ascontiguousarray<T>(T[,] a, Dtype dtype = null)
            => NumPy.Instance.ascontiguousarray(a, dtype:dtype);
        
        /// <summary>
        /// Interpret the input as a matrix.
        /// 
        /// Unlike matrix, asmatrix does not make a copy if the input is already
        /// a matrix or an ndarray.  Equivalent to matrix(data, copy=False).
        /// </summary>
        /// <param name="data">
        /// Input data.
        /// </param>
        /// <param name="dtype">
        /// Data-type of the output matrix.
        /// </param>
        /// <returns>
        /// data interpreted as a matrix.
        /// </returns>
        public static Matrix asmatrix(NDarray data, Dtype dtype)
            => NumPy.Instance.asmatrix(data, dtype);
        
        /// <summary>
        /// Interpret the input as a matrix.
        /// 
        /// Unlike matrix, asmatrix does not make a copy if the input is already
        /// a matrix or an ndarray.  Equivalent to matrix(data, copy=False).
        /// </summary>
        /// <param name="data">
        /// Input data.
        /// </param>
        /// <param name="dtype">
        /// Data-type of the output matrix.
        /// </param>
        /// <returns>
        /// data interpreted as a matrix.
        /// </returns>
        public static Matrix asmatrix<T>(T[] data, Dtype dtype)
            => NumPy.Instance.asmatrix(data, dtype);
        
        /// <summary>
        /// Interpret the input as a matrix.
        /// 
        /// Unlike matrix, asmatrix does not make a copy if the input is already
        /// a matrix or an ndarray.  Equivalent to matrix(data, copy=False).
        /// </summary>
        /// <param name="data">
        /// Input data.
        /// </param>
        /// <param name="dtype">
        /// Data-type of the output matrix.
        /// </param>
        /// <returns>
        /// data interpreted as a matrix.
        /// </returns>
        public static Matrix asmatrix<T>(T[,] data, Dtype dtype)
            => NumPy.Instance.asmatrix(data, dtype);
        
        /// <summary>
        /// Return an array copy of the given object.
        /// 
        /// Notes
        /// 
        /// This is equivalent to:
        /// </summary>
        /// <param name="a">
        /// Input data.
        /// </param>
        /// <param name="order">
        /// Controls the memory layout of the copy. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        /// ‘C’ otherwise. ‘K’ means match the layout of a as closely
        /// as possible. (Note that this function and ndarray.copy are very
        /// similar, but have different default values for their order=
        /// arguments.)
        /// </param>
        /// <returns>
        /// Array interpretation of a.
        /// </returns>
        public static NDarray copy(NDarray a, string order = null)
            => NumPy.Instance.copy(a, order:order);
        
        /// <summary>
        /// Return an array copy of the given object.
        /// 
        /// Notes
        /// 
        /// This is equivalent to:
        /// </summary>
        /// <param name="a">
        /// Input data.
        /// </param>
        /// <param name="order">
        /// Controls the memory layout of the copy. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        /// ‘C’ otherwise. ‘K’ means match the layout of a as closely
        /// as possible. (Note that this function and ndarray.copy are very
        /// similar, but have different default values for their order=
        /// arguments.)
        /// </param>
        /// <returns>
        /// Array interpretation of a.
        /// </returns>
        public static NDarray<T> copy<T>(T[] a, string order = null)
            => NumPy.Instance.copy(a, order:order);
        
        /// <summary>
        /// Return an array copy of the given object.
        /// 
        /// Notes
        /// 
        /// This is equivalent to:
        /// </summary>
        /// <param name="a">
        /// Input data.
        /// </param>
        /// <param name="order">
        /// Controls the memory layout of the copy. ‘C’ means C-order,
        /// ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        /// ‘C’ otherwise. ‘K’ means match the layout of a as closely
        /// as possible. (Note that this function and ndarray.copy are very
        /// similar, but have different default values for their order=
        /// arguments.)
        /// </param>
        /// <returns>
        /// Array interpretation of a.
        /// </returns>
        public static NDarray<T> copy<T>(T[,] a, string order = null)
            => NumPy.Instance.copy(a, order:order);
        
        /*
        /// <summary>
        /// Interpret a buffer as a 1-dimensional array.
        /// 
        /// Notes
        /// 
        /// If the buffer has data that is not in machine byte-order, this should
        /// be specified as part of the data-type, e.g.:
        /// 
        /// The data of the resulting array will not be byteswapped, but will be
        /// interpreted correctly.
        /// </summary>
        /// <param name="buffer">
        /// An object that exposes the buffer interface.
        /// </param>
        /// <param name="dtype">
        /// Data-type of the returned array; default: float.
        /// </param>
        /// <param name="count">
        /// Number of items to read. -1 means all data in the buffer.
        /// </param>
        /// <param name="offset">
        /// Start reading the buffer from this offset (in bytes); default: 0.
        /// </param>
        public static void frombuffer(buffer_like buffer, Dtype dtype = null, int? count = null, int? offset = null)
            => NumPy.Instance.frombuffer(buffer, dtype:dtype, count:count, offset:offset);
        */
        
        /// <summary>
        /// Construct an array from data in a text or binary file.
        /// 
        /// A highly efficient way of reading binary data with a known data-type,
        /// as well as parsing simply formatted text files.  Data written using the
        /// tofile method can be read using this function.
        /// 
        /// Notes
        /// 
        /// Do not rely on the combination of tofile and fromfile for
        /// data storage, as the binary files generated are are not platform
        /// independent.  In particular, no byte-order or data-type information is
        /// saved.  Data can be stored in the platform independent .npy format
        /// using save and load instead.
        /// </summary>
        /// <param name="file">
        /// Open file object or filename.
        /// </param>
        /// <param name="dtype">
        /// Data type of the returned array.
        /// For binary files, it is used to determine the size and byte-order
        /// of the items in the file.
        /// </param>
        /// <param name="count">
        /// Number of items to read. -1 means all items (i.e., the complete
        /// file).
        /// </param>
        /// <param name="sep">
        /// Separator between items if file is a text file.
        /// Empty (“”) separator means the file should be treated as binary.
        /// Spaces (” “) in the separator match zero or more whitespace characters.
        /// A separator consisting only of spaces must match at least one
        /// whitespace.
        /// </param>
        public static void fromfile(string file, Dtype dtype, int count, string sep)
            => NumPy.Instance.fromfile(file, dtype, count, sep);
        
        /// <summary>
        /// Construct an array by executing a function over each coordinate.
        /// 
        /// The resulting array therefore has a value fn(x, y, z) at
        /// coordinate (x, y, z).
        /// 
        /// Notes
        /// 
        /// Keywords other than dtype are passed to function.
        /// </summary>
        /// <param name="function">
        /// The function is called with N parameters, where N is the rank of
        /// shape.  Each parameter represents the coordinates of the array
        /// varying along a specific axis.  For example, if shape
        /// were (2, 2), then the parameters would be
        /// array([[0, 0], [1, 1]]) and array([[0, 1], [0, 1]])
        /// </param>
        /// <param name="shape">
        /// Shape of the output array, which also determines the shape of
        /// the coordinate arrays passed to function.
        /// </param>
        /// <param name="dtype">
        /// Data-type of the coordinate arrays passed to function.
        /// By default, dtype is float.
        /// </param>
        /// <returns>
        /// The result of the call to function is passed back directly.
        /// Therefore the shape of fromfunction is completely determined by
        /// function.  If function returns a scalar value, the shape of
        /// fromfunction would not match the shape parameter.
        /// </returns>
        public static object fromfunction(Delegate function, Shape shape, Dtype dtype = null)
            => NumPy.Instance.fromfunction(function, shape, dtype:dtype);
        
        /// <summary>
        /// Create a new 1-dimensional array from an iterable object.
        /// 
        /// Notes
        /// 
        /// Specify count to improve performance.  It allows fromiter to
        /// pre-allocate the output array, instead of resizing it on demand.
        /// </summary>
        /// <param name="iterable">
        /// An iterable object providing data for the array.
        /// </param>
        /// <param name="dtype">
        /// The data-type of the returned array.
        /// </param>
        /// <param name="count">
        /// The number of items to read from iterable.  The default is -1,
        /// which means all data is read.
        /// </param>
        /// <returns>
        /// The output array.
        /// </returns>
        public static NDarray<T> fromiter<T>(IEnumerable<T> iterable, Dtype dtype, int? count = null)
            => NumPy.Instance.fromiter(iterable, dtype, count:count);
        
        /// <summary>
        /// A new 1-D array initialized from text data in a string.
        /// </summary>
        /// <param name="@string">
        /// A string containing the data.
        /// </param>
        /// <param name="dtype">
        /// The data type of the array; default: float.  For binary input data,
        /// the data must be in exactly this format.
        /// </param>
        /// <param name="count">
        /// Read this number of dtype elements from the data.  If this is
        /// negative (the default), the count will be determined from the
        /// length of the data.
        /// </param>
        /// <param name="sep">
        /// The string separating numbers in the data; extra whitespace between
        /// elements is also ignored.
        /// </param>
        /// <returns>
        /// The constructed array.
        /// </returns>
        public static NDarray fromstring(string @string, Dtype dtype = null, int? count = null, string sep = null)
            => NumPy.Instance.fromstring(@string, dtype:dtype, count:count, sep:sep);
        
        /// <summary>
        /// Load data from a text file.
        /// 
        /// Each row in the text file must have the same number of values.
        /// 
        /// Notes
        /// 
        /// This function aims to be a fast reader for simply formatted files.  The
        /// genfromtxt function provides more sophisticated handling of, e.g.,
        /// lines with missing values.
        /// 
        /// The strings produced by the Python float.hex method can be used as
        /// input for floats.
        /// </summary>
        /// <param name="fname">
        /// File, filename, or generator to read.  If the filename extension is
        /// .gz or .bz2, the file is first decompressed. Note that
        /// generators should return byte strings for Python 3k.
        /// </param>
        /// <param name="dtype">
        /// Data-type of the resulting array; default: float.  If this is a
        /// structured data-type, the resulting array will be 1-dimensional, and
        /// each row will be interpreted as an element of the array.  In this
        /// case, the number of columns used must match the number of fields in
        /// the data-type.
        /// </param>
        /// <param name="comments">
        /// The characters or list of characters used to indicate the start of a
        /// comment. None implies no comments. For backwards compatibility, byte
        /// strings will be decoded as ‘latin1’. The default is ‘#’.
        /// </param>
        /// <param name="delimiter">
        /// The string used to separate values. For backwards compatibility, byte
        /// strings will be decoded as ‘latin1’. The default is whitespace.
        /// </param>
        /// <param name="converters">
        /// A dictionary mapping column number to a function that will parse the
        /// column string into the desired value.  E.g., if column 0 is a date
        /// string: converters = {0: datestr2num}.  Converters can also be
        /// used to provide a default value for missing data (but see also
        /// genfromtxt): converters = {3: lambda s: float(s.strip() or 0)}.
        /// Default: None.
        /// </param>
        /// <param name="skiprows">
        /// Skip the first skiprows lines; default: 0.
        /// </param>
        /// <param name="usecols">
        /// Which columns to read, with 0 being the first. For example,
        /// usecols = (1,4,5) will extract the 2nd, 5th and 6th columns.
        /// The default, None, results in all columns being read.
        /// </param>
        /// <param name="unpack">
        /// If True, the returned array is transposed, so that arguments may be
        /// unpacked using x, y, z = loadtxt(...).  When used with a structured
        /// data-type, arrays are returned for each field.  Default is False.
        /// </param>
        /// <param name="ndmin">
        /// The returned array will have at least ndmin dimensions.
        /// Otherwise mono-dimensional axes will be squeezed.
        /// Legal values: 0 (default), 1 or 2.
        /// </param>
        /// <param name="encoding">
        /// Encoding used to decode the inputfile. Does not apply to input streams.
        /// The special value ‘bytes’ enables backward compatibility workarounds
        /// that ensures you receive byte arrays as results if possible and passes
        /// ‘latin1’ encoded strings to converters. Override this value to receive
        /// unicode arrays and pass strings as input to converters.  If set to None
        /// the system default is used. The default value is ‘bytes’.
        /// </param>
        /// <param name="max_rows">
        /// Read max_rows lines of content after skiprows lines. The default
        /// is to read all the lines.
        /// </param>
        /// <returns>
        /// Data read from the text file.
        /// </returns>
        public static NDarray loadtxt(string fname, Dtype dtype = null, string[] comments = null, string delimiter = null, Hashtable converters = null, int? skiprows = null, int[] usecols = null, bool? unpack = null, int? ndmin = null, string encoding = null, int? max_rows = null)
            => NumPy.Instance.loadtxt(fname, dtype:dtype, comments:comments, delimiter:delimiter, converters:converters, skiprows:skiprows, usecols:usecols, unpack:unpack, ndmin:ndmin, encoding:encoding, max_rows:max_rows);
        
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        /// Start of interval.  The interval includes this value.  The default
        /// start value is 0.
        /// </param>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDarray arange(byte start, byte stop, byte step = 1, Dtype dtype = null)
            => NumPy.Instance.arange(start, stop, step:step, dtype:dtype);
        
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDarray arange(byte stop, byte step = 1, Dtype dtype = null)
            => NumPy.Instance.arange(stop, step:step, dtype:dtype);
        
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        /// Start of interval.  The interval includes this value.  The default
        /// start value is 0.
        /// </param>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDarray arange(short start, short stop, short step = 1, Dtype dtype = null)
            => NumPy.Instance.arange(start, stop, step:step, dtype:dtype);
        
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDarray arange(short stop, short step = 1, Dtype dtype = null)
            => NumPy.Instance.arange(stop, step:step, dtype:dtype);
        
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        /// Start of interval.  The interval includes this value.  The default
        /// start value is 0.
        /// </param>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDarray arange(int start, int stop, int step = 1, Dtype dtype = null)
            => NumPy.Instance.arange(start, stop, step:step, dtype:dtype);
        
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDarray arange(int stop, int step = 1, Dtype dtype = null)
            => NumPy.Instance.arange(stop, step:step, dtype:dtype);
        
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        /// Start of interval.  The interval includes this value.  The default
        /// start value is 0.
        /// </param>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDarray arange(long start, long stop, long step = 1, Dtype dtype = null)
            => NumPy.Instance.arange(start, stop, step:step, dtype:dtype);
        
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDarray arange(long stop, long step = 1, Dtype dtype = null)
            => NumPy.Instance.arange(stop, step:step, dtype:dtype);
        
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        /// Start of interval.  The interval includes this value.  The default
        /// start value is 0.
        /// </param>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDarray arange(float start, float stop, float step = 1, Dtype dtype = null)
            => NumPy.Instance.arange(start, stop, step:step, dtype:dtype);
        
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDarray arange(float stop, float step = 1, Dtype dtype = null)
            => NumPy.Instance.arange(stop, step:step, dtype:dtype);
        
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        /// Start of interval.  The interval includes this value.  The default
        /// start value is 0.
        /// </param>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDarray arange(double start, double stop, double step = 1, Dtype dtype = null)
            => NumPy.Instance.arange(start, stop, step:step, dtype:dtype);
        
        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// 
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// For integer arguments the function is equivalent to the Python built-in
        /// range function, but returns an ndarray rather than a list.
        /// 
        /// When using a non-integer step, such as 0.1, the results will often not
        /// be consistent.  It is better to use numpy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        /// End of interval.  The interval does not include this value, except
        /// in some cases where step is not an integer and floating point
        /// round-off affects the length of out.
        /// </param>
        /// <param name="step">
        /// Spacing between values.  For any output out, this is the distance
        /// between two adjacent values, out[i+1] - out[i].  The default
        /// step size is 1.  If step is specified as a position argument,
        /// start must also be given.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <returns>
        /// Array of evenly spaced values.
        /// 
        /// For floating point arguments, the length of the result is
        /// ceil((stop - start)/step).  Because of floating point overflow,
        /// this rule may result in the last element of out being greater
        /// than stop.
        /// </returns>
        public static NDarray arange(double stop, double step = 1, Dtype dtype = null)
            => NumPy.Instance.arange(stop, step:step, dtype:dtype);
        
        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// 
        /// Returns num evenly spaced samples, calculated over the
        /// interval [start, stop].
        /// 
        /// The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">
        /// The starting value of the sequence.
        /// </param>
        /// <param name="stop">
        /// The end value of the sequence, unless endpoint is set to False.
        /// In that case, the sequence consists of all but the last of num + 1
        /// evenly spaced samples, so that stop is excluded.  Note that the step
        /// size changes when endpoint is False.
        /// </param>
        /// <param name="num">
        /// Number of samples to generate. Default is 50. Must be non-negative.
        /// </param>
        /// <param name="endpoint">
        /// If True, stop is the last sample. Otherwise, it is not included.
        /// Default is True.
        /// </param>
        /// <param name="retstep">
        /// If True, return (samples, step), where step is the spacing
        /// between samples.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <param name="axis">
        /// The axis in the result to store the samples.  Relevant only if start
        /// or stop are array-like.  By default (0), the samples will be along a
        /// new axis inserted at the beginning. Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        /// A tuple of:
        /// samples
        /// There are num equally spaced samples in the closed interval
        /// [start, stop] or the half-open interval [start, stop)
        /// (depending on whether endpoint is True or False).
        /// step
        /// Only returned if retstep is True
        /// 
        /// Size of spacing between samples.
        /// </returns>
        // Error generating delaration: linspace
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
      "Name": "start",
      "Type": "NDarray",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The starting value of the sequence.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": false
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "stop",
      "Type": "NDarray",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The end value of the sequence, unless endpoint is set to False.\nIn that case, the sequence consists of all but the last of num + 1\nevenly spaced samples, so that stop is excluded.  Note that the step\nsize changes when endpoint is False.",
      "ConvertToSharpType": null,
      "Position": 1,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "num",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "Number of samples to generate. Default is 50. Must be non-negative.",
      "ConvertToSharpType": null,
      "Position": 2,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "endpoint",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, stop is the last sample. Otherwise, it is not included.\nDefault is True.",
      "ConvertToSharpType": null,
      "Position": 3,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "retstep",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, return (samples, step), where step is the spacing\nbetween samples.",
      "ConvertToSharpType": null,
      "Position": 4,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "dtype",
      "Type": "Dtype",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The type of the output array.  If dtype is not given, infer the data\ntype from the other input arguments.",
      "ConvertToSharpType": null,
      "Position": 5,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "axis",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The axis in the result to store the samples.  Relevant only if start\nor stop are array-like.  By default (0), the samples will be along a\nnew axis inserted at the beginning. Use -1 to get an axis at the end.",
      "ConvertToSharpType": null,
      "Position": 6,
      "IsReturnValue": false
    }
  ],
  "Generics": null,
  "ForwardToStaticImpl": null,
  "Name": "linspace",
  "ClassName": "numpy",
  "Returns": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "samples",
      "Type": "NDarray",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "There are num equally spaced samples in the closed interval\n[start, stop] or the half-open interval [start, stop)\n(depending on whether endpoint is True or False).",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "step",
      "Type": "float",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "Only returned if retstep is True\r\n\r\nSize of spacing between samples.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    }
  ],
  "IsDeprecated": false,
  "ManualOverride": false,
  "CommentOut": false,
  "DebuggerBreak": false,
  "Description": "Return evenly spaced numbers over a specified interval.\r\n\r\nReturns num evenly spaced samples, calculated over the\ninterval [start, stop].\r\n\r\nThe endpoint of the interval can optionally be excluded."
}
        */
        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// 
        /// Returns num evenly spaced samples, calculated over the
        /// interval [start, stop].
        /// 
        /// The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">
        /// The starting value of the sequence.
        /// </param>
        /// <param name="stop">
        /// The end value of the sequence, unless endpoint is set to False.
        /// In that case, the sequence consists of all but the last of num + 1
        /// evenly spaced samples, so that stop is excluded.  Note that the step
        /// size changes when endpoint is False.
        /// </param>
        /// <param name="num">
        /// Number of samples to generate. Default is 50. Must be non-negative.
        /// </param>
        /// <param name="endpoint">
        /// If True, stop is the last sample. Otherwise, it is not included.
        /// Default is True.
        /// </param>
        /// <param name="retstep">
        /// If True, return (samples, step), where step is the spacing
        /// between samples.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <param name="axis">
        /// The axis in the result to store the samples.  Relevant only if start
        /// or stop are array-like.  By default (0), the samples will be along a
        /// new axis inserted at the beginning. Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        /// A tuple of:
        /// samples
        /// There are num equally spaced samples in the closed interval
        /// [start, stop] or the half-open interval [start, stop)
        /// (depending on whether endpoint is True or False).
        /// step
        /// Only returned if retstep is True
        /// 
        /// Size of spacing between samples.
        /// </returns>
        // Error generating delaration: linspace
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
      "Name": "start",
      "Type": "T[]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The starting value of the sequence.",
      "ConvertToSharpType": "NDarray",
      "Position": 0,
      "IsReturnValue": false
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "stop",
      "Type": "array_like",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The end value of the sequence, unless endpoint is set to False.\nIn that case, the sequence consists of all but the last of num + 1\nevenly spaced samples, so that stop is excluded.  Note that the step\nsize changes when endpoint is False.",
      "ConvertToSharpType": null,
      "Position": 1,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "num",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "Number of samples to generate. Default is 50. Must be non-negative.",
      "ConvertToSharpType": null,
      "Position": 2,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "endpoint",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, stop is the last sample. Otherwise, it is not included.\nDefault is True.",
      "ConvertToSharpType": null,
      "Position": 3,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "retstep",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, return (samples, step), where step is the spacing\nbetween samples.",
      "ConvertToSharpType": null,
      "Position": 4,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "dtype",
      "Type": "Dtype",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The type of the output array.  If dtype is not given, infer the data\ntype from the other input arguments.",
      "ConvertToSharpType": null,
      "Position": 5,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "axis",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The axis in the result to store the samples.  Relevant only if start\nor stop are array-like.  By default (0), the samples will be along a\nnew axis inserted at the beginning. Use -1 to get an axis at the end.",
      "ConvertToSharpType": null,
      "Position": 6,
      "IsReturnValue": false
    }
  ],
  "Generics": [
    "T"
  ],
  "ForwardToStaticImpl": null,
  "Name": "linspace",
  "ClassName": "numpy",
  "Returns": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "samples",
      "Type": "NDarray<T>",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "There are num equally spaced samples in the closed interval\n[start, stop] or the half-open interval [start, stop)\n(depending on whether endpoint is True or False).",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "step",
      "Type": "float",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "Only returned if retstep is True\r\n\r\nSize of spacing between samples.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    }
  ],
  "IsDeprecated": false,
  "ManualOverride": false,
  "CommentOut": false,
  "DebuggerBreak": false,
  "Description": "Return evenly spaced numbers over a specified interval.\r\n\r\nReturns num evenly spaced samples, calculated over the\ninterval [start, stop].\r\n\r\nThe endpoint of the interval can optionally be excluded."
}
        */
        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// 
        /// Returns num evenly spaced samples, calculated over the
        /// interval [start, stop].
        /// 
        /// The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">
        /// The starting value of the sequence.
        /// </param>
        /// <param name="stop">
        /// The end value of the sequence, unless endpoint is set to False.
        /// In that case, the sequence consists of all but the last of num + 1
        /// evenly spaced samples, so that stop is excluded.  Note that the step
        /// size changes when endpoint is False.
        /// </param>
        /// <param name="num">
        /// Number of samples to generate. Default is 50. Must be non-negative.
        /// </param>
        /// <param name="endpoint">
        /// If True, stop is the last sample. Otherwise, it is not included.
        /// Default is True.
        /// </param>
        /// <param name="retstep">
        /// If True, return (samples, step), where step is the spacing
        /// between samples.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <param name="axis">
        /// The axis in the result to store the samples.  Relevant only if start
        /// or stop are array-like.  By default (0), the samples will be along a
        /// new axis inserted at the beginning. Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        /// A tuple of:
        /// samples
        /// There are num equally spaced samples in the closed interval
        /// [start, stop] or the half-open interval [start, stop)
        /// (depending on whether endpoint is True or False).
        /// step
        /// Only returned if retstep is True
        /// 
        /// Size of spacing between samples.
        /// </returns>
        // Error generating delaration: linspace
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
      "Name": "start",
      "Type": "T[,]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The starting value of the sequence.",
      "ConvertToSharpType": "NDarray",
      "Position": 0,
      "IsReturnValue": false
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "stop",
      "Type": "array_like",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The end value of the sequence, unless endpoint is set to False.\nIn that case, the sequence consists of all but the last of num + 1\nevenly spaced samples, so that stop is excluded.  Note that the step\nsize changes when endpoint is False.",
      "ConvertToSharpType": null,
      "Position": 1,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "num",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "Number of samples to generate. Default is 50. Must be non-negative.",
      "ConvertToSharpType": null,
      "Position": 2,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "endpoint",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, stop is the last sample. Otherwise, it is not included.\nDefault is True.",
      "ConvertToSharpType": null,
      "Position": 3,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "retstep",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, return (samples, step), where step is the spacing\nbetween samples.",
      "ConvertToSharpType": null,
      "Position": 4,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "dtype",
      "Type": "Dtype",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The type of the output array.  If dtype is not given, infer the data\ntype from the other input arguments.",
      "ConvertToSharpType": null,
      "Position": 5,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "axis",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The axis in the result to store the samples.  Relevant only if start\nor stop are array-like.  By default (0), the samples will be along a\nnew axis inserted at the beginning. Use -1 to get an axis at the end.",
      "ConvertToSharpType": null,
      "Position": 6,
      "IsReturnValue": false
    }
  ],
  "Generics": [
    "T"
  ],
  "ForwardToStaticImpl": null,
  "Name": "linspace",
  "ClassName": "numpy",
  "Returns": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "samples",
      "Type": "NDarray<T>",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "There are num equally spaced samples in the closed interval\n[start, stop] or the half-open interval [start, stop)\n(depending on whether endpoint is True or False).",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "step",
      "Type": "float",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "Only returned if retstep is True\r\n\r\nSize of spacing between samples.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    }
  ],
  "IsDeprecated": false,
  "ManualOverride": false,
  "CommentOut": false,
  "DebuggerBreak": false,
  "Description": "Return evenly spaced numbers over a specified interval.\r\n\r\nReturns num evenly spaced samples, calculated over the\ninterval [start, stop].\r\n\r\nThe endpoint of the interval can optionally be excluded."
}
        */
        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// 
        /// Returns num evenly spaced samples, calculated over the
        /// interval [start, stop].
        /// 
        /// The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">
        /// The starting value of the sequence.
        /// </param>
        /// <param name="stop">
        /// The end value of the sequence, unless endpoint is set to False.
        /// In that case, the sequence consists of all but the last of num + 1
        /// evenly spaced samples, so that stop is excluded.  Note that the step
        /// size changes when endpoint is False.
        /// </param>
        /// <param name="num">
        /// Number of samples to generate. Default is 50. Must be non-negative.
        /// </param>
        /// <param name="endpoint">
        /// If True, stop is the last sample. Otherwise, it is not included.
        /// Default is True.
        /// </param>
        /// <param name="retstep">
        /// If True, return (samples, step), where step is the spacing
        /// between samples.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <param name="axis">
        /// The axis in the result to store the samples.  Relevant only if start
        /// or stop are array-like.  By default (0), the samples will be along a
        /// new axis inserted at the beginning. Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        /// A tuple of:
        /// samples
        /// There are num equally spaced samples in the closed interval
        /// [start, stop] or the half-open interval [start, stop)
        /// (depending on whether endpoint is True or False).
        /// step
        /// Only returned if retstep is True
        /// 
        /// Size of spacing between samples.
        /// </returns>
        // Error generating delaration: linspace
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
      "Name": "start",
      "Type": "NDarray",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The starting value of the sequence.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": false
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "stop",
      "Type": "T[]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The end value of the sequence, unless endpoint is set to False.\nIn that case, the sequence consists of all but the last of num + 1\nevenly spaced samples, so that stop is excluded.  Note that the step\nsize changes when endpoint is False.",
      "ConvertToSharpType": "NDarray",
      "Position": 1,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "num",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "Number of samples to generate. Default is 50. Must be non-negative.",
      "ConvertToSharpType": null,
      "Position": 2,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "endpoint",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, stop is the last sample. Otherwise, it is not included.\nDefault is True.",
      "ConvertToSharpType": null,
      "Position": 3,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "retstep",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, return (samples, step), where step is the spacing\nbetween samples.",
      "ConvertToSharpType": null,
      "Position": 4,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "dtype",
      "Type": "Dtype",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The type of the output array.  If dtype is not given, infer the data\ntype from the other input arguments.",
      "ConvertToSharpType": null,
      "Position": 5,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "axis",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The axis in the result to store the samples.  Relevant only if start\nor stop are array-like.  By default (0), the samples will be along a\nnew axis inserted at the beginning. Use -1 to get an axis at the end.",
      "ConvertToSharpType": null,
      "Position": 6,
      "IsReturnValue": false
    }
  ],
  "Generics": [
    "T"
  ],
  "ForwardToStaticImpl": null,
  "Name": "linspace",
  "ClassName": "numpy",
  "Returns": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "samples",
      "Type": "NDarray<T>",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "There are num equally spaced samples in the closed interval\n[start, stop] or the half-open interval [start, stop)\n(depending on whether endpoint is True or False).",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "step",
      "Type": "float",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "Only returned if retstep is True\r\n\r\nSize of spacing between samples.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    }
  ],
  "IsDeprecated": false,
  "ManualOverride": false,
  "CommentOut": false,
  "DebuggerBreak": false,
  "Description": "Return evenly spaced numbers over a specified interval.\r\n\r\nReturns num evenly spaced samples, calculated over the\ninterval [start, stop].\r\n\r\nThe endpoint of the interval can optionally be excluded."
}
        */
        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// 
        /// Returns num evenly spaced samples, calculated over the
        /// interval [start, stop].
        /// 
        /// The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">
        /// The starting value of the sequence.
        /// </param>
        /// <param name="stop">
        /// The end value of the sequence, unless endpoint is set to False.
        /// In that case, the sequence consists of all but the last of num + 1
        /// evenly spaced samples, so that stop is excluded.  Note that the step
        /// size changes when endpoint is False.
        /// </param>
        /// <param name="num">
        /// Number of samples to generate. Default is 50. Must be non-negative.
        /// </param>
        /// <param name="endpoint">
        /// If True, stop is the last sample. Otherwise, it is not included.
        /// Default is True.
        /// </param>
        /// <param name="retstep">
        /// If True, return (samples, step), where step is the spacing
        /// between samples.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <param name="axis">
        /// The axis in the result to store the samples.  Relevant only if start
        /// or stop are array-like.  By default (0), the samples will be along a
        /// new axis inserted at the beginning. Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        /// A tuple of:
        /// samples
        /// There are num equally spaced samples in the closed interval
        /// [start, stop] or the half-open interval [start, stop)
        /// (depending on whether endpoint is True or False).
        /// step
        /// Only returned if retstep is True
        /// 
        /// Size of spacing between samples.
        /// </returns>
        // Error generating delaration: linspace
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
      "Name": "start",
      "Type": "NDarray",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The starting value of the sequence.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": false
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "stop",
      "Type": "T[,]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The end value of the sequence, unless endpoint is set to False.\nIn that case, the sequence consists of all but the last of num + 1\nevenly spaced samples, so that stop is excluded.  Note that the step\nsize changes when endpoint is False.",
      "ConvertToSharpType": "NDarray",
      "Position": 1,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "num",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "Number of samples to generate. Default is 50. Must be non-negative.",
      "ConvertToSharpType": null,
      "Position": 2,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "endpoint",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, stop is the last sample. Otherwise, it is not included.\nDefault is True.",
      "ConvertToSharpType": null,
      "Position": 3,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "retstep",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, return (samples, step), where step is the spacing\nbetween samples.",
      "ConvertToSharpType": null,
      "Position": 4,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "dtype",
      "Type": "Dtype",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The type of the output array.  If dtype is not given, infer the data\ntype from the other input arguments.",
      "ConvertToSharpType": null,
      "Position": 5,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "axis",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The axis in the result to store the samples.  Relevant only if start\nor stop are array-like.  By default (0), the samples will be along a\nnew axis inserted at the beginning. Use -1 to get an axis at the end.",
      "ConvertToSharpType": null,
      "Position": 6,
      "IsReturnValue": false
    }
  ],
  "Generics": [
    "T"
  ],
  "ForwardToStaticImpl": null,
  "Name": "linspace",
  "ClassName": "numpy",
  "Returns": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "samples",
      "Type": "NDarray<T>",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "There are num equally spaced samples in the closed interval\n[start, stop] or the half-open interval [start, stop)\n(depending on whether endpoint is True or False).",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "step",
      "Type": "float",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "Only returned if retstep is True\r\n\r\nSize of spacing between samples.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    }
  ],
  "IsDeprecated": false,
  "ManualOverride": false,
  "CommentOut": false,
  "DebuggerBreak": false,
  "Description": "Return evenly spaced numbers over a specified interval.\r\n\r\nReturns num evenly spaced samples, calculated over the\ninterval [start, stop].\r\n\r\nThe endpoint of the interval can optionally be excluded."
}
        */
        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// 
        /// Returns num evenly spaced samples, calculated over the
        /// interval [start, stop].
        /// 
        /// The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">
        /// The starting value of the sequence.
        /// </param>
        /// <param name="stop">
        /// The end value of the sequence, unless endpoint is set to False.
        /// In that case, the sequence consists of all but the last of num + 1
        /// evenly spaced samples, so that stop is excluded.  Note that the step
        /// size changes when endpoint is False.
        /// </param>
        /// <param name="num">
        /// Number of samples to generate. Default is 50. Must be non-negative.
        /// </param>
        /// <param name="endpoint">
        /// If True, stop is the last sample. Otherwise, it is not included.
        /// Default is True.
        /// </param>
        /// <param name="retstep">
        /// If True, return (samples, step), where step is the spacing
        /// between samples.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <param name="axis">
        /// The axis in the result to store the samples.  Relevant only if start
        /// or stop are array-like.  By default (0), the samples will be along a
        /// new axis inserted at the beginning. Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        /// A tuple of:
        /// samples
        /// There are num equally spaced samples in the closed interval
        /// [start, stop] or the half-open interval [start, stop)
        /// (depending on whether endpoint is True or False).
        /// step
        /// Only returned if retstep is True
        /// 
        /// Size of spacing between samples.
        /// </returns>
        // Error generating delaration: linspace
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
      "Name": "start",
      "Type": "T[]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The starting value of the sequence.",
      "ConvertToSharpType": "NDarray",
      "Position": 0,
      "IsReturnValue": false
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "stop",
      "Type": "T[]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The end value of the sequence, unless endpoint is set to False.\nIn that case, the sequence consists of all but the last of num + 1\nevenly spaced samples, so that stop is excluded.  Note that the step\nsize changes when endpoint is False.",
      "ConvertToSharpType": "NDarray",
      "Position": 1,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "num",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "Number of samples to generate. Default is 50. Must be non-negative.",
      "ConvertToSharpType": null,
      "Position": 2,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "endpoint",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, stop is the last sample. Otherwise, it is not included.\nDefault is True.",
      "ConvertToSharpType": null,
      "Position": 3,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "retstep",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, return (samples, step), where step is the spacing\nbetween samples.",
      "ConvertToSharpType": null,
      "Position": 4,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "dtype",
      "Type": "Dtype",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The type of the output array.  If dtype is not given, infer the data\ntype from the other input arguments.",
      "ConvertToSharpType": null,
      "Position": 5,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "axis",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The axis in the result to store the samples.  Relevant only if start\nor stop are array-like.  By default (0), the samples will be along a\nnew axis inserted at the beginning. Use -1 to get an axis at the end.",
      "ConvertToSharpType": null,
      "Position": 6,
      "IsReturnValue": false
    }
  ],
  "Generics": [
    "T"
  ],
  "ForwardToStaticImpl": null,
  "Name": "linspace",
  "ClassName": "numpy",
  "Returns": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "samples",
      "Type": "NDarray<T>",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "There are num equally spaced samples in the closed interval\n[start, stop] or the half-open interval [start, stop)\n(depending on whether endpoint is True or False).",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "step",
      "Type": "float",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "Only returned if retstep is True\r\n\r\nSize of spacing between samples.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    }
  ],
  "IsDeprecated": false,
  "ManualOverride": false,
  "CommentOut": false,
  "DebuggerBreak": false,
  "Description": "Return evenly spaced numbers over a specified interval.\r\n\r\nReturns num evenly spaced samples, calculated over the\ninterval [start, stop].\r\n\r\nThe endpoint of the interval can optionally be excluded."
}
        */
        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// 
        /// Returns num evenly spaced samples, calculated over the
        /// interval [start, stop].
        /// 
        /// The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">
        /// The starting value of the sequence.
        /// </param>
        /// <param name="stop">
        /// The end value of the sequence, unless endpoint is set to False.
        /// In that case, the sequence consists of all but the last of num + 1
        /// evenly spaced samples, so that stop is excluded.  Note that the step
        /// size changes when endpoint is False.
        /// </param>
        /// <param name="num">
        /// Number of samples to generate. Default is 50. Must be non-negative.
        /// </param>
        /// <param name="endpoint">
        /// If True, stop is the last sample. Otherwise, it is not included.
        /// Default is True.
        /// </param>
        /// <param name="retstep">
        /// If True, return (samples, step), where step is the spacing
        /// between samples.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <param name="axis">
        /// The axis in the result to store the samples.  Relevant only if start
        /// or stop are array-like.  By default (0), the samples will be along a
        /// new axis inserted at the beginning. Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        /// A tuple of:
        /// samples
        /// There are num equally spaced samples in the closed interval
        /// [start, stop] or the half-open interval [start, stop)
        /// (depending on whether endpoint is True or False).
        /// step
        /// Only returned if retstep is True
        /// 
        /// Size of spacing between samples.
        /// </returns>
        // Error generating delaration: linspace
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
      "Name": "start",
      "Type": "T[]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The starting value of the sequence.",
      "ConvertToSharpType": "NDarray",
      "Position": 0,
      "IsReturnValue": false
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "stop",
      "Type": "T[,]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The end value of the sequence, unless endpoint is set to False.\nIn that case, the sequence consists of all but the last of num + 1\nevenly spaced samples, so that stop is excluded.  Note that the step\nsize changes when endpoint is False.",
      "ConvertToSharpType": "NDarray",
      "Position": 1,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "num",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "Number of samples to generate. Default is 50. Must be non-negative.",
      "ConvertToSharpType": null,
      "Position": 2,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "endpoint",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, stop is the last sample. Otherwise, it is not included.\nDefault is True.",
      "ConvertToSharpType": null,
      "Position": 3,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "retstep",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, return (samples, step), where step is the spacing\nbetween samples.",
      "ConvertToSharpType": null,
      "Position": 4,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "dtype",
      "Type": "Dtype",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The type of the output array.  If dtype is not given, infer the data\ntype from the other input arguments.",
      "ConvertToSharpType": null,
      "Position": 5,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "axis",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The axis in the result to store the samples.  Relevant only if start\nor stop are array-like.  By default (0), the samples will be along a\nnew axis inserted at the beginning. Use -1 to get an axis at the end.",
      "ConvertToSharpType": null,
      "Position": 6,
      "IsReturnValue": false
    }
  ],
  "Generics": [
    "T"
  ],
  "ForwardToStaticImpl": null,
  "Name": "linspace",
  "ClassName": "numpy",
  "Returns": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "samples",
      "Type": "NDarray<T>",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "There are num equally spaced samples in the closed interval\n[start, stop] or the half-open interval [start, stop)\n(depending on whether endpoint is True or False).",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "step",
      "Type": "float",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "Only returned if retstep is True\r\n\r\nSize of spacing between samples.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    }
  ],
  "IsDeprecated": false,
  "ManualOverride": false,
  "CommentOut": false,
  "DebuggerBreak": false,
  "Description": "Return evenly spaced numbers over a specified interval.\r\n\r\nReturns num evenly spaced samples, calculated over the\ninterval [start, stop].\r\n\r\nThe endpoint of the interval can optionally be excluded."
}
        */
        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// 
        /// Returns num evenly spaced samples, calculated over the
        /// interval [start, stop].
        /// 
        /// The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">
        /// The starting value of the sequence.
        /// </param>
        /// <param name="stop">
        /// The end value of the sequence, unless endpoint is set to False.
        /// In that case, the sequence consists of all but the last of num + 1
        /// evenly spaced samples, so that stop is excluded.  Note that the step
        /// size changes when endpoint is False.
        /// </param>
        /// <param name="num">
        /// Number of samples to generate. Default is 50. Must be non-negative.
        /// </param>
        /// <param name="endpoint">
        /// If True, stop is the last sample. Otherwise, it is not included.
        /// Default is True.
        /// </param>
        /// <param name="retstep">
        /// If True, return (samples, step), where step is the spacing
        /// between samples.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <param name="axis">
        /// The axis in the result to store the samples.  Relevant only if start
        /// or stop are array-like.  By default (0), the samples will be along a
        /// new axis inserted at the beginning. Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        /// A tuple of:
        /// samples
        /// There are num equally spaced samples in the closed interval
        /// [start, stop] or the half-open interval [start, stop)
        /// (depending on whether endpoint is True or False).
        /// step
        /// Only returned if retstep is True
        /// 
        /// Size of spacing between samples.
        /// </returns>
        // Error generating delaration: linspace
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
      "Name": "start",
      "Type": "T[,]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The starting value of the sequence.",
      "ConvertToSharpType": "NDarray",
      "Position": 0,
      "IsReturnValue": false
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "stop",
      "Type": "T[]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The end value of the sequence, unless endpoint is set to False.\nIn that case, the sequence consists of all but the last of num + 1\nevenly spaced samples, so that stop is excluded.  Note that the step\nsize changes when endpoint is False.",
      "ConvertToSharpType": "NDarray",
      "Position": 1,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "num",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "Number of samples to generate. Default is 50. Must be non-negative.",
      "ConvertToSharpType": null,
      "Position": 2,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "endpoint",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, stop is the last sample. Otherwise, it is not included.\nDefault is True.",
      "ConvertToSharpType": null,
      "Position": 3,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "retstep",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, return (samples, step), where step is the spacing\nbetween samples.",
      "ConvertToSharpType": null,
      "Position": 4,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "dtype",
      "Type": "Dtype",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The type of the output array.  If dtype is not given, infer the data\ntype from the other input arguments.",
      "ConvertToSharpType": null,
      "Position": 5,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "axis",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The axis in the result to store the samples.  Relevant only if start\nor stop are array-like.  By default (0), the samples will be along a\nnew axis inserted at the beginning. Use -1 to get an axis at the end.",
      "ConvertToSharpType": null,
      "Position": 6,
      "IsReturnValue": false
    }
  ],
  "Generics": [
    "T"
  ],
  "ForwardToStaticImpl": null,
  "Name": "linspace",
  "ClassName": "numpy",
  "Returns": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "samples",
      "Type": "NDarray<T>",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "There are num equally spaced samples in the closed interval\n[start, stop] or the half-open interval [start, stop)\n(depending on whether endpoint is True or False).",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "step",
      "Type": "float",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "Only returned if retstep is True\r\n\r\nSize of spacing between samples.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    }
  ],
  "IsDeprecated": false,
  "ManualOverride": false,
  "CommentOut": false,
  "DebuggerBreak": false,
  "Description": "Return evenly spaced numbers over a specified interval.\r\n\r\nReturns num evenly spaced samples, calculated over the\ninterval [start, stop].\r\n\r\nThe endpoint of the interval can optionally be excluded."
}
        */
        /// <summary>
        /// Return evenly spaced numbers over a specified interval.
        /// 
        /// Returns num evenly spaced samples, calculated over the
        /// interval [start, stop].
        /// 
        /// The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">
        /// The starting value of the sequence.
        /// </param>
        /// <param name="stop">
        /// The end value of the sequence, unless endpoint is set to False.
        /// In that case, the sequence consists of all but the last of num + 1
        /// evenly spaced samples, so that stop is excluded.  Note that the step
        /// size changes when endpoint is False.
        /// </param>
        /// <param name="num">
        /// Number of samples to generate. Default is 50. Must be non-negative.
        /// </param>
        /// <param name="endpoint">
        /// If True, stop is the last sample. Otherwise, it is not included.
        /// Default is True.
        /// </param>
        /// <param name="retstep">
        /// If True, return (samples, step), where step is the spacing
        /// between samples.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <param name="axis">
        /// The axis in the result to store the samples.  Relevant only if start
        /// or stop are array-like.  By default (0), the samples will be along a
        /// new axis inserted at the beginning. Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        /// A tuple of:
        /// samples
        /// There are num equally spaced samples in the closed interval
        /// [start, stop] or the half-open interval [start, stop)
        /// (depending on whether endpoint is True or False).
        /// step
        /// Only returned if retstep is True
        /// 
        /// Size of spacing between samples.
        /// </returns>
        // Error generating delaration: linspace
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
      "Name": "start",
      "Type": "T[,]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The starting value of the sequence.",
      "ConvertToSharpType": "NDarray",
      "Position": 0,
      "IsReturnValue": false
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "stop",
      "Type": "T[,]",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "The end value of the sequence, unless endpoint is set to False.\nIn that case, the sequence consists of all but the last of num + 1\nevenly spaced samples, so that stop is excluded.  Note that the step\nsize changes when endpoint is False.",
      "ConvertToSharpType": "NDarray",
      "Position": 1,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "num",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "Number of samples to generate. Default is 50. Must be non-negative.",
      "ConvertToSharpType": null,
      "Position": 2,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "endpoint",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, stop is the last sample. Otherwise, it is not included.\nDefault is True.",
      "ConvertToSharpType": null,
      "Position": 3,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "retstep",
      "Type": "bool",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "If True, return (samples, step), where step is the spacing\nbetween samples.",
      "ConvertToSharpType": null,
      "Position": 4,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "dtype",
      "Type": "Dtype",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The type of the output array.  If dtype is not given, infer the data\ntype from the other input arguments.",
      "ConvertToSharpType": null,
      "Position": 5,
      "IsReturnValue": false
    },
    {
      "IsNullable": true,
      "IsValueType": false,
      "Name": "axis",
      "Type": "int",
      "DefaultValue": null,
      "IsNamedArg": true,
      "Description": "The axis in the result to store the samples.  Relevant only if start\nor stop are array-like.  By default (0), the samples will be along a\nnew axis inserted at the beginning. Use -1 to get an axis at the end.",
      "ConvertToSharpType": null,
      "Position": 6,
      "IsReturnValue": false
    }
  ],
  "Generics": [
    "T"
  ],
  "ForwardToStaticImpl": null,
  "Name": "linspace",
  "ClassName": "numpy",
  "Returns": [
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "samples",
      "Type": "NDarray<T>",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "There are num equally spaced samples in the closed interval\n[start, stop] or the half-open interval [start, stop)\n(depending on whether endpoint is True or False).",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    },
    {
      "IsNullable": false,
      "IsValueType": false,
      "Name": "step",
      "Type": "float",
      "DefaultValue": null,
      "IsNamedArg": false,
      "Description": "Only returned if retstep is True\r\n\r\nSize of spacing between samples.",
      "ConvertToSharpType": null,
      "Position": 0,
      "IsReturnValue": true
    }
  ],
  "IsDeprecated": false,
  "ManualOverride": false,
  "CommentOut": false,
  "DebuggerBreak": false,
  "Description": "Return evenly spaced numbers over a specified interval.\r\n\r\nReturns num evenly spaced samples, calculated over the\ninterval [start, stop].\r\n\r\nThe endpoint of the interval can optionally be excluded."
}
        */
        /// <summary>
        /// Return numbers spaced evenly on a log scale.
        /// 
        /// In linear space, the sequence starts at base ** start
        /// (base to the power of start) and ends with base ** stop
        /// (see endpoint below).
        /// 
        /// Notes
        /// 
        /// Logspace is equivalent to the code
        /// </summary>
        /// <param name="start">
        /// base ** start is the starting value of the sequence.
        /// </param>
        /// <param name="stop">
        /// base ** stop is the final value of the sequence, unless endpoint
        /// is False.  In that case, num + 1 values are spaced over the
        /// interval in log-space, of which all but the last (a sequence of
        /// length num) are returned.
        /// </param>
        /// <param name="num">
        /// Number of samples to generate.  Default is 50.
        /// </param>
        /// <param name="endpoint">
        /// If true, stop is the last sample. Otherwise, it is not included.
        /// Default is True.
        /// </param>
        /// <param name="@base">
        /// The base of the log space. The step size between the elements in
        /// ln(samples) / ln(base) (or log_base(samples)) is uniform.
        /// Default is 10.0.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <param name="axis">
        /// The axis in the result to store the samples.  Relevant only if start
        /// or stop are array-like.  By default (0), the samples will be along a
        /// new axis inserted at the beginning. Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        /// num samples, equally spaced on a log scale.
        /// </returns>
        public static NDarray logspace(NDarray start, NDarray stop, int? num = null, bool? endpoint = null, float? @base = null, Dtype dtype = null, int? axis = null)
            => NumPy.Instance.logspace(start, stop, num:num, endpoint:endpoint, @base:@base, dtype:dtype, axis:axis);
        
        /// <summary>
        /// Return numbers spaced evenly on a log scale (a geometric progression).
        /// 
        /// This is similar to logspace, but with endpoints specified directly.
        /// Each output sample is a constant multiple of the previous.
        /// 
        /// Notes
        /// 
        /// If the inputs or dtype are complex, the output will follow a logarithmic
        /// spiral in the complex plane.  (There are an infinite number of spirals
        /// passing through two points; the output will follow the shortest such path.)
        /// </summary>
        /// <param name="start">
        /// The starting value of the sequence.
        /// </param>
        /// <param name="stop">
        /// The final value of the sequence, unless endpoint is False.
        /// In that case, num + 1 values are spaced over the
        /// interval in log-space, of which all but the last (a sequence of
        /// length num) are returned.
        /// </param>
        /// <param name="num">
        /// Number of samples to generate.  Default is 50.
        /// </param>
        /// <param name="endpoint">
        /// If true, stop is the last sample. Otherwise, it is not included.
        /// Default is True.
        /// </param>
        /// <param name="dtype">
        /// The type of the output array.  If dtype is not given, infer the data
        /// type from the other input arguments.
        /// </param>
        /// <param name="axis">
        /// The axis in the result to store the samples.  Relevant only if start
        /// or stop are array-like.  By default (0), the samples will be along a
        /// new axis inserted at the beginning. Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        /// num samples, equally spaced on a log scale.
        /// </returns>
        public static NDarray geomspace(NDarray start, NDarray stop, int? num = null, bool? endpoint = null, Dtype dtype = null, int? axis = null)
            => NumPy.Instance.geomspace(start, stop, num:num, endpoint:endpoint, dtype:dtype, axis:axis);
        
        /*
        /// <summary>
        /// Return coordinate matrices from coordinate vectors.
        /// 
        /// Make N-D coordinate arrays for vectorized evaluations of
        /// N-D scalar/vector fields over N-D grids, given
        /// one-dimensional coordinate arrays x1, x2,…, xn.
        /// 
        /// Notes
        /// 
        /// This function supports both indexing conventions through the indexing
        /// keyword argument.  Giving the string ‘ij’ returns a meshgrid with
        /// matrix indexing, while ‘xy’ returns a meshgrid with Cartesian indexing.
        /// In the 2-D case with inputs of length M and N, the outputs are of shape
        /// (N, M) for ‘xy’ indexing and (M, N) for ‘ij’ indexing.  In the 3-D case
        /// with inputs of length M, N and P, outputs are of shape (N, M, P) for
        /// ‘xy’ indexing and (M, N, P) for ‘ij’ indexing.  The difference is
        /// illustrated by the following code snippet:
        /// 
        /// In the 1-D and 0-D case, the indexing and sparse keywords have no effect.
        /// </summary>
        /// <param name="x1, x2,…, xn">
        /// 1-D arrays representing the coordinates of a grid.
        /// </param>
        /// <param name="indexing">
        /// Cartesian (‘xy’, default) or matrix (‘ij’) indexing of output.
        /// See Notes for more details.
        /// </param>
        /// <param name="sparse">
        /// If True a sparse grid is returned in order to conserve memory.
        /// Default is False.
        /// </param>
        /// <param name="copy">
        /// If False, a view into the original arrays are returned in order to
        /// conserve memory.  Default is True.  Please note that
        /// sparse=False, copy=False will likely return non-contiguous
        /// arrays.  Furthermore, more than one element of a broadcast array
        /// may refer to a single memory location.  If you need to write to the
        /// arrays, make copies first.
        /// </param>
        /// <returns>
        /// For vectors x1, x2,…, ‘xn’ with lengths Ni=len(xi) ,
        /// return (N1, N2, N3,...Nn) shaped arrays if indexing=’ij’
        /// or (N2, N1, N3,...Nn) shaped arrays if indexing=’xy’
        /// with the elements of xi repeated to fill the matrix along
        /// the first dimension for x1, the second for x2 and so on.
        /// </returns>
        public static NDarray meshgrid(NDarray x1, x2,…, xn, string indexing = null, bool? sparse = null, bool? copy = null)
            => NumPy.Instance.meshgrid(x1, x2,…, xn, indexing:indexing, sparse:sparse, copy:copy);
        */
        
        /*
        /// <summary>
        /// Return coordinate matrices from coordinate vectors.
        /// 
        /// Make N-D coordinate arrays for vectorized evaluations of
        /// N-D scalar/vector fields over N-D grids, given
        /// one-dimensional coordinate arrays x1, x2,…, xn.
        /// 
        /// Notes
        /// 
        /// This function supports both indexing conventions through the indexing
        /// keyword argument.  Giving the string ‘ij’ returns a meshgrid with
        /// matrix indexing, while ‘xy’ returns a meshgrid with Cartesian indexing.
        /// In the 2-D case with inputs of length M and N, the outputs are of shape
        /// (N, M) for ‘xy’ indexing and (M, N) for ‘ij’ indexing.  In the 3-D case
        /// with inputs of length M, N and P, outputs are of shape (N, M, P) for
        /// ‘xy’ indexing and (M, N, P) for ‘ij’ indexing.  The difference is
        /// illustrated by the following code snippet:
        /// 
        /// In the 1-D and 0-D case, the indexing and sparse keywords have no effect.
        /// </summary>
        /// <param name="x1, x2,…, xn">
        /// 1-D arrays representing the coordinates of a grid.
        /// </param>
        /// <param name="indexing">
        /// Cartesian (‘xy’, default) or matrix (‘ij’) indexing of output.
        /// See Notes for more details.
        /// </param>
        /// <param name="sparse">
        /// If True a sparse grid is returned in order to conserve memory.
        /// Default is False.
        /// </param>
        /// <param name="copy">
        /// If False, a view into the original arrays are returned in order to
        /// conserve memory.  Default is True.  Please note that
        /// sparse=False, copy=False will likely return non-contiguous
        /// arrays.  Furthermore, more than one element of a broadcast array
        /// may refer to a single memory location.  If you need to write to the
        /// arrays, make copies first.
        /// </param>
        /// <returns>
        /// For vectors x1, x2,…, ‘xn’ with lengths Ni=len(xi) ,
        /// return (N1, N2, N3,...Nn) shaped arrays if indexing=’ij’
        /// or (N2, N1, N3,...Nn) shaped arrays if indexing=’xy’
        /// with the elements of xi repeated to fill the matrix along
        /// the first dimension for x1, the second for x2 and so on.
        /// </returns>
        public static NDarray<T> meshgrid<T>(T[] x1, x2,…, xn, string indexing = null, bool? sparse = null, bool? copy = null)
            => NumPy.Instance.meshgrid(x1, x2,…, xn, indexing:indexing, sparse:sparse, copy:copy);
        */
        
        /*
        /// <summary>
        /// Return coordinate matrices from coordinate vectors.
        /// 
        /// Make N-D coordinate arrays for vectorized evaluations of
        /// N-D scalar/vector fields over N-D grids, given
        /// one-dimensional coordinate arrays x1, x2,…, xn.
        /// 
        /// Notes
        /// 
        /// This function supports both indexing conventions through the indexing
        /// keyword argument.  Giving the string ‘ij’ returns a meshgrid with
        /// matrix indexing, while ‘xy’ returns a meshgrid with Cartesian indexing.
        /// In the 2-D case with inputs of length M and N, the outputs are of shape
        /// (N, M) for ‘xy’ indexing and (M, N) for ‘ij’ indexing.  In the 3-D case
        /// with inputs of length M, N and P, outputs are of shape (N, M, P) for
        /// ‘xy’ indexing and (M, N, P) for ‘ij’ indexing.  The difference is
        /// illustrated by the following code snippet:
        /// 
        /// In the 1-D and 0-D case, the indexing and sparse keywords have no effect.
        /// </summary>
        /// <param name="x1, x2,…, xn">
        /// 1-D arrays representing the coordinates of a grid.
        /// </param>
        /// <param name="indexing">
        /// Cartesian (‘xy’, default) or matrix (‘ij’) indexing of output.
        /// See Notes for more details.
        /// </param>
        /// <param name="sparse">
        /// If True a sparse grid is returned in order to conserve memory.
        /// Default is False.
        /// </param>
        /// <param name="copy">
        /// If False, a view into the original arrays are returned in order to
        /// conserve memory.  Default is True.  Please note that
        /// sparse=False, copy=False will likely return non-contiguous
        /// arrays.  Furthermore, more than one element of a broadcast array
        /// may refer to a single memory location.  If you need to write to the
        /// arrays, make copies first.
        /// </param>
        /// <returns>
        /// For vectors x1, x2,…, ‘xn’ with lengths Ni=len(xi) ,
        /// return (N1, N2, N3,...Nn) shaped arrays if indexing=’ij’
        /// or (N2, N1, N3,...Nn) shaped arrays if indexing=’xy’
        /// with the elements of xi repeated to fill the matrix along
        /// the first dimension for x1, the second for x2 and so on.
        /// </returns>
        public static NDarray<T> meshgrid<T>(T[,] x1, x2,…, xn, string indexing = null, bool? sparse = null, bool? copy = null)
            => NumPy.Instance.meshgrid(x1, x2,…, xn, indexing:indexing, sparse:sparse, copy:copy);
        */
        
        /// <summary>
        /// Extract a diagonal or construct a diagonal array.
        /// 
        /// See the more detailed documentation for numpy.diagonal if you use this
        /// function to extract a diagonal and wish to write to the resulting array;
        /// whether it returns a copy or a view depends on what version of numpy you
        /// are using.
        /// </summary>
        /// <param name="v">
        /// If v is a 2-D array, return a copy of its k-th diagonal.
        /// If v is a 1-D array, return a 2-D array with v on the k-th
        /// diagonal.
        /// </param>
        /// <param name="k">
        /// Diagonal in question. The default is 0. Use k&gt;0 for diagonals
        /// above the main diagonal, and k&lt;0 for diagonals below the main
        /// diagonal.
        /// </param>
        /// <returns>
        /// The extracted diagonal or constructed diagonal array.
        /// </returns>
        public static NDarray diag(NDarray v, int? k = null)
            => NumPy.Instance.diag(v, k:k);
        
        /// <summary>
        /// Extract a diagonal or construct a diagonal array.
        /// 
        /// See the more detailed documentation for numpy.diagonal if you use this
        /// function to extract a diagonal and wish to write to the resulting array;
        /// whether it returns a copy or a view depends on what version of numpy you
        /// are using.
        /// </summary>
        /// <param name="v">
        /// If v is a 2-D array, return a copy of its k-th diagonal.
        /// If v is a 1-D array, return a 2-D array with v on the k-th
        /// diagonal.
        /// </param>
        /// <param name="k">
        /// Diagonal in question. The default is 0. Use k&gt;0 for diagonals
        /// above the main diagonal, and k&lt;0 for diagonals below the main
        /// diagonal.
        /// </param>
        /// <returns>
        /// The extracted diagonal or constructed diagonal array.
        /// </returns>
        public static NDarray<T> diag<T>(T[] v, int? k = null)
            => NumPy.Instance.diag(v, k:k);
        
        /// <summary>
        /// Extract a diagonal or construct a diagonal array.
        /// 
        /// See the more detailed documentation for numpy.diagonal if you use this
        /// function to extract a diagonal and wish to write to the resulting array;
        /// whether it returns a copy or a view depends on what version of numpy you
        /// are using.
        /// </summary>
        /// <param name="v">
        /// If v is a 2-D array, return a copy of its k-th diagonal.
        /// If v is a 1-D array, return a 2-D array with v on the k-th
        /// diagonal.
        /// </param>
        /// <param name="k">
        /// Diagonal in question. The default is 0. Use k&gt;0 for diagonals
        /// above the main diagonal, and k&lt;0 for diagonals below the main
        /// diagonal.
        /// </param>
        /// <returns>
        /// The extracted diagonal or constructed diagonal array.
        /// </returns>
        public static NDarray<T> diag<T>(T[,] v, int? k = null)
            => NumPy.Instance.diag(v, k:k);
        
        /// <summary>
        /// Create a two-dimensional array with the flattened input as a diagonal.
        /// </summary>
        /// <param name="v">
        /// Input data, which is flattened and set as the k-th
        /// diagonal of the output.
        /// </param>
        /// <param name="k">
        /// Diagonal to set; 0, the default, corresponds to the “main” diagonal,
        /// a positive (negative) k giving the number of the diagonal above
        /// (below) the main.
        /// </param>
        /// <returns>
        /// The 2-D output array.
        /// </returns>
        public static NDarray diagflat(NDarray v, int? k = null)
            => NumPy.Instance.diagflat(v, k:k);
        
        /// <summary>
        /// Create a two-dimensional array with the flattened input as a diagonal.
        /// </summary>
        /// <param name="v">
        /// Input data, which is flattened and set as the k-th
        /// diagonal of the output.
        /// </param>
        /// <param name="k">
        /// Diagonal to set; 0, the default, corresponds to the “main” diagonal,
        /// a positive (negative) k giving the number of the diagonal above
        /// (below) the main.
        /// </param>
        /// <returns>
        /// The 2-D output array.
        /// </returns>
        public static NDarray<T> diagflat<T>(T[] v, int? k = null)
            => NumPy.Instance.diagflat(v, k:k);
        
        /// <summary>
        /// Create a two-dimensional array with the flattened input as a diagonal.
        /// </summary>
        /// <param name="v">
        /// Input data, which is flattened and set as the k-th
        /// diagonal of the output.
        /// </param>
        /// <param name="k">
        /// Diagonal to set; 0, the default, corresponds to the “main” diagonal,
        /// a positive (negative) k giving the number of the diagonal above
        /// (below) the main.
        /// </param>
        /// <returns>
        /// The 2-D output array.
        /// </returns>
        public static NDarray<T> diagflat<T>(T[,] v, int? k = null)
            => NumPy.Instance.diagflat(v, k:k);
        
        /// <summary>
        /// An array with ones at and below the given diagonal and zeros elsewhere.
        /// </summary>
        /// <param name="N">
        /// Number of rows in the array.
        /// </param>
        /// <param name="M">
        /// Number of columns in the array.
        /// By default, M is taken equal to N.
        /// </param>
        /// <param name="k">
        /// The sub-diagonal at and below which the array is filled.
        /// k = 0 is the main diagonal, while k &lt; 0 is below it,
        /// and k &gt; 0 is above.  The default is 0.
        /// </param>
        /// <param name="dtype">
        /// Data type of the returned array.  The default is float.
        /// </param>
        /// <returns>
        /// Array with its lower triangle filled with ones and zero elsewhere;
        /// in other words T[i,j] == 1 for i &lt;= j + k, 0 otherwise.
        /// </returns>
        public static NDarray tri(int N, int? M = null, int? k = null, Dtype dtype = null)
            => NumPy.Instance.tri(N, M:M, k:k, dtype:dtype);
        
        /// <summary>
        /// Lower triangle of an array.
        /// 
        /// Return a copy of an array with elements above the k-th diagonal zeroed.
        /// </summary>
        /// <param name="m">
        /// Input array.
        /// </param>
        /// <param name="k">
        /// Diagonal above which to zero elements.  k = 0 (the default) is the
        /// main diagonal, k &lt; 0 is below it and k &gt; 0 is above.
        /// </param>
        /// <returns>
        /// Lower triangle of m, of same shape and data-type as m.
        /// </returns>
        public static NDarray tril(NDarray m, int? k = null)
            => NumPy.Instance.tril(m, k:k);
        
        /// <summary>
        /// Lower triangle of an array.
        /// 
        /// Return a copy of an array with elements above the k-th diagonal zeroed.
        /// </summary>
        /// <param name="m">
        /// Input array.
        /// </param>
        /// <param name="k">
        /// Diagonal above which to zero elements.  k = 0 (the default) is the
        /// main diagonal, k &lt; 0 is below it and k &gt; 0 is above.
        /// </param>
        /// <returns>
        /// Lower triangle of m, of same shape and data-type as m.
        /// </returns>
        public static NDarray<T> tril<T>(T[] m, int? k = null)
            => NumPy.Instance.tril(m, k:k);
        
        /// <summary>
        /// Lower triangle of an array.
        /// 
        /// Return a copy of an array with elements above the k-th diagonal zeroed.
        /// </summary>
        /// <param name="m">
        /// Input array.
        /// </param>
        /// <param name="k">
        /// Diagonal above which to zero elements.  k = 0 (the default) is the
        /// main diagonal, k &lt; 0 is below it and k &gt; 0 is above.
        /// </param>
        /// <returns>
        /// Lower triangle of m, of same shape and data-type as m.
        /// </returns>
        public static NDarray<T> tril<T>(T[,] m, int? k = null)
            => NumPy.Instance.tril(m, k:k);
        
        /// <summary>
        /// Generate a Vandermonde matrix.
        /// 
        /// The columns of the output matrix are powers of the input vector. The
        /// order of the powers is determined by the increasing boolean argument.
        /// Specifically, when increasing is False, the i-th output column is
        /// the input vector raised element-wise to the power of N - i - 1. Such
        /// a matrix with a geometric progression in each row is named for Alexandre-
        /// Theophile Vandermonde.
        /// </summary>
        /// <param name="x">
        /// 1-D input array.
        /// </param>
        /// <param name="N">
        /// Number of columns in the output.  If N is not specified, a square
        /// array is returned (N = len(x)).
        /// </param>
        /// <param name="increasing">
        /// Order of the powers of the columns.  If True, the powers increase
        /// from left to right, if False (the default) they are reversed.
        /// </param>
        /// <returns>
        /// Vandermonde matrix.  If increasing is False, the first column is
        /// x^(N-1), the second x^(N-2) and so forth. If increasing is
        /// True, the columns are x^0, x^1, ..., x^(N-1).
        /// </returns>
        public static NDarray vander(NDarray x, int? N = null, bool? increasing = null)
            => NumPy.Instance.vander(x, N:N, increasing:increasing);
        
        /// <summary>
        /// Generate a Vandermonde matrix.
        /// 
        /// The columns of the output matrix are powers of the input vector. The
        /// order of the powers is determined by the increasing boolean argument.
        /// Specifically, when increasing is False, the i-th output column is
        /// the input vector raised element-wise to the power of N - i - 1. Such
        /// a matrix with a geometric progression in each row is named for Alexandre-
        /// Theophile Vandermonde.
        /// </summary>
        /// <param name="x">
        /// 1-D input array.
        /// </param>
        /// <param name="N">
        /// Number of columns in the output.  If N is not specified, a square
        /// array is returned (N = len(x)).
        /// </param>
        /// <param name="increasing">
        /// Order of the powers of the columns.  If True, the powers increase
        /// from left to right, if False (the default) they are reversed.
        /// </param>
        /// <returns>
        /// Vandermonde matrix.  If increasing is False, the first column is
        /// x^(N-1), the second x^(N-2) and so forth. If increasing is
        /// True, the columns are x^0, x^1, ..., x^(N-1).
        /// </returns>
        public static NDarray<T> vander<T>(T[] x, int? N = null, bool? increasing = null)
            => NumPy.Instance.vander(x, N:N, increasing:increasing);
        
        /// <summary>
        /// Generate a Vandermonde matrix.
        /// 
        /// The columns of the output matrix are powers of the input vector. The
        /// order of the powers is determined by the increasing boolean argument.
        /// Specifically, when increasing is False, the i-th output column is
        /// the input vector raised element-wise to the power of N - i - 1. Such
        /// a matrix with a geometric progression in each row is named for Alexandre-
        /// Theophile Vandermonde.
        /// </summary>
        /// <param name="x">
        /// 1-D input array.
        /// </param>
        /// <param name="N">
        /// Number of columns in the output.  If N is not specified, a square
        /// array is returned (N = len(x)).
        /// </param>
        /// <param name="increasing">
        /// Order of the powers of the columns.  If True, the powers increase
        /// from left to right, if False (the default) they are reversed.
        /// </param>
        /// <returns>
        /// Vandermonde matrix.  If increasing is False, the first column is
        /// x^(N-1), the second x^(N-2) and so forth. If increasing is
        /// True, the columns are x^0, x^1, ..., x^(N-1).
        /// </returns>
        public static NDarray<T> vander<T>(T[,] x, int? N = null, bool? increasing = null)
            => NumPy.Instance.vander(x, N:N, increasing:increasing);
        
        /*
        /// <summary>
        /// Interpret the input as a matrix.
        /// 
        /// Unlike matrix, asmatrix does not make a copy if the input is already
        /// a matrix or an ndarray.  Equivalent to matrix(data, copy=False).
        /// </summary>
        /// <param name="data">
        /// Input data.
        /// </param>
        /// <param name="dtype">
        /// Data-type of the output matrix.
        /// </param>
        /// <returns>
        /// data interpreted as a matrix.
        /// </returns>
        public static Matrix mat(NDarray data, Dtype dtype)
            => NumPy.Instance.mat(data, dtype);
        */
        
        /*
        /// <summary>
        /// Interpret the input as a matrix.
        /// 
        /// Unlike matrix, asmatrix does not make a copy if the input is already
        /// a matrix or an ndarray.  Equivalent to matrix(data, copy=False).
        /// </summary>
        /// <param name="data">
        /// Input data.
        /// </param>
        /// <param name="dtype">
        /// Data-type of the output matrix.
        /// </param>
        /// <returns>
        /// data interpreted as a matrix.
        /// </returns>
        public static Matrix mat<T>(T[] data, Dtype dtype)
            => NumPy.Instance.mat(data, dtype);
        */
        
        /*
        /// <summary>
        /// Interpret the input as a matrix.
        /// 
        /// Unlike matrix, asmatrix does not make a copy if the input is already
        /// a matrix or an ndarray.  Equivalent to matrix(data, copy=False).
        /// </summary>
        /// <param name="data">
        /// Input data.
        /// </param>
        /// <param name="dtype">
        /// Data-type of the output matrix.
        /// </param>
        /// <returns>
        /// data interpreted as a matrix.
        /// </returns>
        public static Matrix mat<T>(T[,] data, Dtype dtype)
            => NumPy.Instance.mat(data, dtype);
        */
        
        /*
        /// <summary>
        /// Build a matrix object from a string, nested sequence, or array.
        /// </summary>
        /// <param name="obj">
        /// Input data. If a string, variables in the current scope may be
        /// referenced by name.
        /// </param>
        /// <param name="ldict">
        /// A dictionary that replaces local operands in current frame.
        /// Ignored if obj is not a string or gdict is None.
        /// </param>
        /// <param name="gdict">
        /// A dictionary that replaces global operands in current frame.
        /// Ignored if obj is not a string.
        /// </param>
        /// <returns>
        /// Returns a matrix object, which is a specialized 2-D array.
        /// </returns>
        public static Matrix bmat(string obj, Hashtable ldict = null, Hashtable gdict = null)
            => NumPy.Instance.bmat(obj, ldict:ldict, gdict:gdict);
        */
        
        /*
        /// <summary>
        /// Build a matrix object from a string, nested sequence, or array.
        /// </summary>
        /// <param name="obj">
        /// Input data. If a string, variables in the current scope may be
        /// referenced by name.
        /// </param>
        /// <param name="ldict">
        /// A dictionary that replaces local operands in current frame.
        /// Ignored if obj is not a string or gdict is None.
        /// </param>
        /// <param name="gdict">
        /// A dictionary that replaces global operands in current frame.
        /// Ignored if obj is not a string.
        /// </param>
        /// <returns>
        /// Returns a matrix object, which is a specialized 2-D array.
        /// </returns>
        public static matrix<T> bmat<T>(T[] obj, Hashtable ldict = null, Hashtable gdict = null)
            => NumPy.Instance.bmat(obj, ldict:ldict, gdict:gdict);
        */
        
        
    }
}
