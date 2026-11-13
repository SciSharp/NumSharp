using System;
using System.Diagnostics.CodeAnalysis;

namespace NumSharp.Generic
{
    public partial class NDArray<TDType>
    {
        /// <summary>
        ///     Gives a new shape to a typed array without changing its data, reading the elements in the specified index
        ///     order — NumPy's <c>a.reshape(shape, /, *, order='C', copy=None)</c>, keeping the typed wrapper.
        /// </summary>
        /// <param name="shape">The new shape (one dimension may be -1, inferred): an int, a tuple, an <c>int[]</c> /
        ///     <c>long[]</c>, a C# collection expression or a <see cref="Shape"/>. Only its DIMENSIONS are a request, as in
        ///     NumPy, whose reshape takes a shape tuple: a <see cref="Shape"/> taken from a view also carries that view's
        ///     strides, offset and buffer size, and none of them reaches the result.</param>
        /// <param name="order">'C' (default), 'F', or 'A' ('F' when this array is F-contiguous and not C-contiguous); 'K' is
        ///     NumPy's <c>ValueError("order 'K' is not permitted for reshaping")</c>.</param>
        /// <param name="copy">NumPy's <c>copy</c> keyword: null copies only when needed, true always copies, false raises
        ///     <c>Unable to avoid creating a copy while reshaping.</c> when no view can express the reshape.</param>
        /// <returns>A typed view whenever the reshape can be expressed over the existing strides, otherwise a typed view
        ///     over an internal copy (see <see cref="NDArray.reshape(Shape, char, bool?)"/>).</returns>
        /// <exception cref="IncorrectShapeException">The size does not match: NumPy's
        ///     <c>cannot reshape array of size {n} into shape {s}</c>.</exception>
        /// <exception cref="ValueError">'K', more than one -1, or the refusal of <paramref name="copy"/> false.</exception>
        /// <exception cref="ArgumentException">An order letter outside NumPy's vocabulary.</exception>
        /// <remarks>
        ///     <para>API-parity batch 9: this replaces the typed <c>(Shape)</c>, <c>(ref Shape)</c> and <c>(int[])</c> forms,
        ///     which had no <paramref name="order"/> or <paramref name="copy"/>, so <c>t.reshape(s, order: 'F')</c> fell
        ///     through to the untyped method and returned an untyped array. The default spelling (C order, copy None) keeps
        ///     the typed alias engine the <c>(Shape)</c> form always ran, at its cost (the loose-int forms' allocation, pinned
        ///     by <c>Shape.CallerArrays.Tests</c>); an order or copy request runs the untyped reshape engine (NumPy's
        ///     <c>_reshape_with_copy_arg</c>, so a non-contiguous source gives NumPy's strided view where one exists) and
        ///     re-wraps the result typed.</para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</para>
        /// </remarks>
        public new NDArray<TDType> reshape(Shape shape, char order = 'C', bool? copy = null)
        {
            // The default spelling: the typed alias engine, no untyped detour (an extra storage alias and wrapper per call).
            if (copy is null && (order == 'C' || order == 'c'))
                return ReshapeDims(shape);
            // The untyped result is a fresh alias only this frame holds: MakeGeneric gives the typed wrapper its own storage
            // alias (never the untyped object's, which the using releases), so no reference outlives its owner.
            using var untyped = base.reshape(shape, order, copy);
            return untyped.MakeGeneric<TDType>();
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
            return ReshapeDims(new Shape(shape));
        }

        /// <summary>
        ///     The loose-integer reshape of a typed array (the body of the <c>(ref Shape)</c> form API-parity batch 9
        ///     removed): an alias of this array's storage relabelled with the requested dimensions, a copy first when this
        ///     array is not contiguous.
        /// </summary>
        /// <param name="newShape">The requested dimensions (one may be -1); only its dimensions are read.</param>
        /// <returns>A typed alias of this array's storage with the new shape.</returns>
        /// <exception cref="IncorrectShapeException">The size does not match: NumPy's
        ///     <c>cannot reshape array of size {n} into shape {s}</c>.</exception>
        /// <remarks>Only the DIMENSIONS of <paramref name="newShape"/> are a request. A <see cref="Shape"/> taken from a view
        ///     also carries that view's strides, offset and buffer size; until 2026-10-10 the typed reshape installed them over
        ///     THIS array's buffer, so the result addressed elements outside it. <see cref="Shape.AsClean"/> keeps a shape that
        ///     already is a plain dimension list as it is, so <c>a.reshape(2, 3)</c> costs what it did.</remarks>
        private NDArray<TDType> ReshapeDims(Shape newShape)
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
            return reshape_unsafe(new Shape(Shape.ComputeLongShape(shape), copy: false));
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
