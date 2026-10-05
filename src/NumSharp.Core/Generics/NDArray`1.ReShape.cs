using System;
using System.Diagnostics.CodeAnalysis;

namespace NumSharp.Generic
{
    public partial class NDArray<TDType>
    {
        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newShape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public new NDArray<TDType> reshape(Shape newShape)
        {
            return reshape(ref newShape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newShape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public new NDArray<TDType> reshape(ref Shape newShape)
        {
            var ret = Storage.Alias();
            ret.Reshape(ref newShape, false);
            return new NDArray<TDType>(ret) { TensorEngine = TensorEngine };
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="shape">The new shape should be compatible with the original shape. If an integer, then the result will be a 
        /// 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array 
        /// and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the 
        /// memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public new NDArray<TDType> reshape(int[] shape)
        {
            return reshape(Shape.Adopt(Shape.ComputeLongShape(shape)));
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="shape">The new shape should be compatible with the original shape. If an integer, then the result will be a
        /// 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array
        /// and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the
        /// memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public new NDArray<TDType> reshape(params long[] shape)
        {
            var ret = Storage.Alias();
            ret.Reshape(shape, false);
            return new NDArray<TDType>(ret) { TensorEngine = TensorEngine };
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data, keeping the typed wrapper. This is the overload
        ///     inline values bind (<c>a.reshape(2, 3)</c>): the call allocates only the dimension array the new shape keeps.
        /// </summary>
        /// <param name="shape">The new dimensions; one may be negative (inferred from the size).</param>
        /// <returns>An alias of this array's storage with the new shape (a copy first when this array is not
        ///     contiguous).</returns>
        /// <exception cref="IncorrectShapeException">The size does not match: NumPy's
        ///     <c>cannot reshape array of size {n} into shape {s}</c>.</exception>
        /// <exception cref="ValueError">More than one dimension is negative.</exception>
        /// <remarks>
        ///     Hides the untyped <see cref="NDArray.reshape(ReadOnlySpan{long})"/> so a typed array stays typed. An array
        ///     argument binds <see cref="reshape(long[])"/> instead, which copies it (bug 2d).
        /// </remarks>
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public new NDArray<TDType> reshape(params ReadOnlySpan<long> shape)
        {
            var newShape = new Shape(shape);
            return reshape(ref newShape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newshape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public new NDArray<TDType> reshape_unsafe(Shape newshape)
        {
            return reshape_unsafe(ref newshape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newshape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public new NDArray<TDType> reshape_unsafe(ref Shape newShape)
        {
            var ret = Storage.Alias();
            ret.Reshape(ref newShape, true);
            return new NDArray<TDType>(ret) { TensorEngine = TensorEngine };
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="shape">The new shape should be compatible with the original shape. If an integer, then the result will be a 
        /// 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array 
        /// and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the 
        /// memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public new NDArray<TDType> reshape_unsafe(int[] shape)
        {
            return reshape_unsafe(Shape.Adopt(Shape.ComputeLongShape(shape)));
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="shape">The new shape should be compatible with the original shape. If an integer, then the result will be a
        /// 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array
        /// and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the
        /// memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html</remarks>
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public new NDArray<TDType> reshape_unsafe(params long[] shape)
        {
            var ret = Storage.Alias();
            ret.Reshape(shape, true);
            return new NDArray<TDType>(ret) { TensorEngine = TensorEngine };
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data and without the size check, keeping the typed
        ///     wrapper. This is the overload inline values bind (<c>a.reshape_unsafe(2, 3)</c>): the call allocates only
        ///     the dimension array the new shape keeps.
        /// </summary>
        /// <param name="shape">The new dimensions; one may be negative (inferred from the size).</param>
        /// <returns>An alias of this array's storage with the new shape (a copy first when this array is not
        ///     contiguous).</returns>
        /// <remarks>
        ///     Unsafe because the element count is not validated against the buffer: a shape larger than the buffer reads
        ///     past it. Hides the untyped <see cref="NDArray.reshape_unsafe(ReadOnlySpan{long})"/>; an array argument binds
        ///     <see cref="reshape_unsafe(long[])"/> instead, which copies it.
        /// </remarks>
        [SuppressMessage("ReSharper", "ParameterHidesMember")]
        public new NDArray<TDType> reshape_unsafe(params ReadOnlySpan<long> shape)
        {
            var newShape = new Shape(shape);
            return reshape_unsafe(ref newShape);
        }
    }
}
