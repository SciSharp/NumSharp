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
        /// <exception cref="IncorrectShapeException">The size does not match: NumPy's
        ///     <c>cannot reshape array of size {n} into shape {s}</c>.</exception>
        /// <remarks>
        ///     <para>Only the DIMENSIONS of <paramref name="newShape"/> are a request, as in NumPy, whose reshape takes a
        ///     shape tuple. A <see cref="Shape"/> taken from a view (<c>x[":, :, :4"].Shape</c>), a transposed or a
        ///     broadcast array also carries that array's strides, offset and buffer size. Until 2026-10-10 this
        ///     overload installed them over THIS array's buffer, so the result addressed elements outside it:
        ///     <c>np.zeros(40, bool).AsGeneric&lt;bool&gt;().reshape(np.arange(80).reshape(5, 2, 8)[":, :, :4"].Shape)</c>
        ///     had strides (16, 8, 1) over 40 elements and read up to element 75 (other heap memory; an access
        ///     violation when the buffer ends where mapped memory ends). The untyped
        ///     <see cref="NDArray.reshape(Shape)"/> never did. <see cref="Shape.AsClean"/> keeps a shape that already
        ///     is a plain dimension list as it is, so <c>a.reshape(2, 3)</c> and <c>a.reshape(new Shape(2, 3))</c> cost
        ///     what they did.</para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</para>
        /// </remarks>
        public new NDArray<TDType> reshape(ref Shape newShape)
        {
            // Dimensions only (see the remarks): the layout of the result comes from this array, never from the argument.
            var requested = newShape.AsClean();
            var ret = Storage.Alias();
            ret.Reshape(ref requested, false);
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
