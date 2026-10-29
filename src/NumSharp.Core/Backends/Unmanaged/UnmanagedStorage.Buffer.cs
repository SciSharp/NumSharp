using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        /// <summary>
        ///     A storage that lays <paramref name="layout"/> over <paramref name="slice"/>, memory it does not own, with
        ///     <paramref name="baseStorage"/> as its base — NumPy's <c>PyArray_NewFromDescr_int</c> with a data pointer and a
        ///     base object, the storage of <c>np.ndarray(shape, buffer=...)</c>'s result.
        /// </summary>
        /// <param name="slice">The elements: the buffer array's own slice (when the dtype and element grid match), or a
        ///     forwarding alias over the span the array occupies (<see cref="ArraySlice.WrapShared(NPTypeCode, void*, long, IArraySlice)"/>).
        ///     Its element type is the storage's dtype.</param>
        /// <param name="layout">The dimensions, element strides, offset and flags, as given. It is not validated: the
        ///     caller has proved that every element it addresses lies inside <paramref name="slice"/>.</param>
        /// <param name="baseStorage">The array's base (NumPy's collapsed <c>base</c>): the buffer array's owner, or the
        ///     storage that stands for a foreign buffer.</param>
        /// <returns>A storage that does not own its data (no <c>OWNDATA</c>), with the flags of
        ///     <paramref name="layout"/>.</returns>
        /// <remarks>
        ///     Unlike <see cref="Alias(Shape)"/>, this neither inherits writeability nor forces <c>ALIGNED</c>: the caller
        ///     states <c>WRITEABLE</c> in <paramref name="layout"/> (NumPy's rule for a buffer is "writeable exactly when the
        ///     buffer is", which a stride-0 axis does not change), and <c>flags.aligned</c> reads the memory's real
        ///     alignment (<see cref="IsDataAligned"/>). One storage is built, not a chain of reinterpreting aliases.
        /// </remarks>
        internal static UnmanagedStorage OverBuffer(IArraySlice slice, Shape layout, UnmanagedStorage baseStorage)
        {
            var r = new UnmanagedStorage();
            r._shape = layout;
            // The type code first: SetInternalArray dispatches on it to fill the typed-slice union.
            r._typecode = slice.TypeCode;
            r._dtype = r._typecode.AsType();
            r.SetInternalArray(slice);
            // Count is the slice's capacity, as SetShapeUnsafe sets it for a strided or broadcast layout.
            r.Count = layout.bufferSize;
            r._baseStorage = baseStorage;
            r.OnReshaped(); // a buffer view never owns its memory (NumPy: data passed in clears OWNDATA)
            return r;
        }
    }
}
