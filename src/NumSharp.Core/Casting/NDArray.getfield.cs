using System;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Returns a field of the array as a certain dtype — a VIEW whose elements are the
        ///     <paramref name="offset"/>-th byte(s) of each of this array's elements, reinterpreted as
        ///     <paramref name="dtype"/>. Port of NumPy's <c>ndarray.getfield(dtype, offset=0)</c>.
        /// </summary>
        /// <param name="dtype">
        ///     The dtype to read the field as. Its itemsize must be ≤ this array's itemsize.
        /// </param>
        /// <param name="offset">
        ///     Byte offset of the field within each element. Must be in <c>[0, itemsize − newItemsize]</c>.
        /// </param>
        /// <returns>
        ///     A byte-reinterpreting VIEW that SHARES memory with this array (writes through, unless this
        ///     array is read-only). Its shape equals this array's shape; the byte-strides are preserved, so
        ///     a narrower field of a contiguous array is a strided view.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.getfield.html
        ///     <para>
        ///     Unlike <see cref="view(Type)"/> — which rescales the last axis so the whole buffer is
        ///     re-tiled into the new dtype — <c>getfield</c> keeps the layout and reads a sub-slice of each
        ///     element's bytes. For a <c>complex128</c> array, <c>getfield(float64, 0)</c> is the real part
        ///     and <c>getfield(float64, 8)</c> is the imaginary part; for an <c>int32</c> array,
        ///     <c>getfield(int16, 0)</c> / <c>getfield(int16, 2)</c> are the low / high halves.
        ///     </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">
        ///     <c>new type is larger than original type</c>, <c>offset is negative</c>, or
        ///     <c>new type plus offset is larger than original type</c> — the verbatim NumPy texts.
        /// </exception>
        public NDArray getfield(Type dtype, int offset = 0)
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));

            return getfield(dtype.GetTypeCode(), offset);
        }

        /// <summary>
        ///     Returns a field of the array as a certain dtype. See <see cref="getfield(Type,int)"/>.
        /// </summary>
        /// <param name="dtype">The dtype to read the field as (its itemsize must be ≤ this array's itemsize).</param>
        /// <param name="offset">Byte offset of the field within each element.</param>
        public NDArray getfield(NPTypeCode dtype, int offset = 0)
        {
            return new NDArray(Storage.GetFieldAlias(dtype, offset)) { TensorEngine = TensorEngine };
        }

        /// <summary>
        ///     Returns a field of the array as a certain dtype — the typed generic form of
        ///     <see cref="getfield(Type,int)"/>.
        /// </summary>
        /// <typeparam name="T">The field dtype (its itemsize must be ≤ this array's itemsize).</typeparam>
        /// <param name="offset">Byte offset of the field within each element.</param>
        public NDArray<T> getfield<T>(int offset = 0) where T : unmanaged
            => new NDArray(Storage.GetFieldAlias<T>(offset)) { TensorEngine = TensorEngine }.AsGeneric<T>();

        /// <summary>
        ///     Puts a value into a specified place in a field defined by a dtype — writes <paramref name="value"/>
        ///     (cast to <paramref name="dtype"/>) into the <paramref name="offset"/>-th byte(s) of each element,
        ///     leaving the rest of each element untouched. In place. Port of NumPy's
        ///     <c>ndarray.setfield(val, dtype, offset=0)</c>.
        /// </summary>
        /// <param name="value">
        ///     Value(s) to place into the field. A scalar is broadcast; an array must broadcast to this array's
        ///     shape. Cast to <paramref name="dtype"/> with NumPy's assignment ('unsafe') rule (float → int
        ///     truncates toward zero).
        /// </param>
        /// <param name="dtype">The dtype of the field being set (its itemsize must be ≤ this array's itemsize).</param>
        /// <param name="offset">Byte offset of the field within each element.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setfield.html
        ///     <para>
        ///     <c>setfield</c> is <c>getfield</c> plus an assignment: it builds the same byte-reinterpreting
        ///     field view and copies <paramref name="value"/> into it. Writeability is checked FIRST (before
        ///     the dtype/offset validation), matching NumPy's <c>PyArray_SetField</c>.
        ///     </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> or <paramref name="value"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">
        ///     <c>assignment destination is read-only</c> (this array is not writeable), or one of
        ///     <c>getfield</c>'s three dtype/offset <see cref="ValueError"/>s.
        /// </exception>
        public void setfield(object value, Type dtype, int offset = 0)
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));

            setfield(value, dtype.GetTypeCode(), offset);
        }

        /// <summary>
        ///     Puts a value into a specified place in a field defined by a dtype. See
        ///     <see cref="setfield(object,Type,int)"/>.
        /// </summary>
        /// <param name="value">Value(s) to place into the field (scalar broadcast; array must broadcast).</param>
        /// <param name="dtype">The dtype of the field being set.</param>
        /// <param name="offset">Byte offset of the field within each element.</param>
        public void setfield(object value, NPTypeCode dtype, int offset = 0)
        {
            // NumPy checks writeability of SELF first (PyArray_FailUnlessWriteable), before building the
            // field view and validating the dtype/offset. Raise the exact ValueError NumPy raises.
            if (!Shape.IsWriteable)
                throw new ValueError("assignment destination is read-only");

            if (value is null)
                throw new ArgumentNullException(nameof(value));

            // Build the byte-reinterpreting field view (validates dtype/offset), then copy value into it.
            NDArray view = getfield(dtype, offset);

            // Convert value → NDArray. Genuine scalars build a 0-d array of their natural dtype (broadcast
            // by copyto); arrays / collections / Half / Complex go through asanyarray — mirroring how
            // fill_diagonal coerces its object-typed value.
            NDArray src = value as NDArray;
            if (src is null)
            {
                src = (value is IConvertible && !(value is string))
                    ? NDArray.Scalar(value)
                    : np.asanyarray(value);
            }

            // NumPy's PyArray_CopyObject == assignment ('unsafe' cast, broadcasting). The view is writeable
            // (self is), so copyto's own read-only guard is a no-op here.
            np.copyto(view, src, casting: "unsafe");
        }
    }
}
