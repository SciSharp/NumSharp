using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Change shape and size of this array <b>in-place</b>.
        ///     <para>
        ///     If the new array is larger than the original array, the new array is filled with
        ///     <b>zeros</b> (note: this differs from <see cref="np.resize(NDArray, Shape)"/> which
        ///     fills with repeated copies). If smaller, the data is truncated (in C-order for
        ///     C-contiguous arrays, memory-order for F-contiguous ones).
        ///     </para>
        ///     Multi-argument form: <c>a.resize(2, 3)</c>. A no-argument call <c>a.resize()</c> is a
        ///     no-op (matches NumPy's <c>a.resize()</c> / <c>a.resize(None)</c>).
        /// </summary>
        /// <param name="new_shape">Shape of resized array (one value per dimension).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.resize.html</remarks>
        /// <exception cref="IncorrectShapeException">
        ///     If this array is not single-segment (contiguous); if growing/shrinking an array that
        ///     does not own its data or is referenced by another array; or if a dimension is negative.
        /// </exception>
        public void resize(params long[] new_shape)
        {
            // NumPy: a.resize() and a.resize(None) return None (no-op). C# collapses both to an
            // empty/null params array. Explicit scalar (a.resize(())) is reachable via the Shape
            // overload with a 0-d shape.
            if (new_shape == null || new_shape.Length == 0)
                return;

            // The public Shape constructor copies the caller's array (bug 2d); inline values bind the span overload
            // below and skip that copy.
            resize(new Shape(new_shape), refcheck: true);
        }

        /// <summary>
        ///     Change shape and size of this array <b>in-place</b> (see <see cref="resize(long[])"/> for the fill and
        ///     truncate rules). This is the overload inline values bind (<c>a.resize(2, 3)</c>): the call allocates only
        ///     the dimension array the new shape keeps.
        /// </summary>
        /// <param name="new_shape">Shape of resized array (one value per dimension). Empty is a no-op, as NumPy's
        ///     <c>a.resize()</c>.</param>
        /// <exception cref="IncorrectShapeException">
        ///     If this array is not single-segment (contiguous); if growing/shrinking an array that does not own its data
        ///     or is referenced by another array; or if a dimension is negative.
        /// </exception>
        /// <exception cref="OutOfMemoryException">The new element count overflows (NumPy's <c>MemoryError</c>).</exception>
        /// <remarks>
        ///     <c>a.resize()</c> binds this overload. An array argument binds <see cref="resize(long[])"/> instead, which
        ///     copies it. https://numpy.org/doc/stable/reference/generated/numpy.ndarray.resize.html
        /// </remarks>
        public void resize(params ReadOnlySpan<long> new_shape)
        {
            // NumPy's a.resize() is a no-op; C# sends it here as an empty span (the array overload's empty/null rule).
            if (new_shape.IsEmpty)
                return;

            resize(new Shape(new_shape), refcheck: true);
        }

        /// <summary>
        ///     Change shape and size of this array <b>in-place</b>.
        ///     <para>Primary overload — see <see cref="resize(long[])"/> for the fill/truncate semantics.</para>
        /// </summary>
        /// <param name="new_shape">Shape of resized array. A 0-d shape resizes to a scalar.</param>
        /// <param name="refcheck">
        ///     If <c>true</c> (default), reference counting is used to check that this array's buffer
        ///     is not shared with another array before resizing (when the total size changes).
        ///     Set to <c>false</c> to skip that check.
        /// </param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.resize.html</remarks>
        /// <exception cref="IncorrectShapeException">
        ///     If this array is not single-segment (contiguous); if growing/shrinking an array that
        ///     does not own its data or (with <paramref name="refcheck"/>) is referenced by another
        ///     array; or if a dimension is negative.
        /// </exception>
        public void resize(Shape new_shape, bool refcheck = true)
        {
            UnmanagedStorage.Resize(this, new_shape, refcheck);
        }
    }
}
