using System;
using System.Runtime.InteropServices;
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
        public unsafe void resize(Shape new_shape, bool refcheck = true)
        {
            var shape = this.Shape;

            // 1. Single-segment (contiguous) requirement — always enforced first, exactly like
            //    NumPy's PyArray_ISONESEGMENT gate. 0-d arrays are trivially single-segment.
            if (shape.NDim != 0 && !shape.IsContiguous && !shape.IsFContiguous)
                throw new IncorrectShapeException("resize only works on single-segment arrays");

            // 2. Validate dimensions and compute the new element count. NumPy stops at the first
            //    zero dim (so a negative dim following a zero is NOT reported) and rejects negatives
            //    otherwise, with this exact message (distinct from np.resize's wording).
            var newDims = new_shape.dimensions ?? System.Array.Empty<long>();
            long newSize = 1;
            for (int i = 0; i < newDims.Length; i++)
            {
                if (newDims[i] == 0) { newSize = 0; break; }
                if (newDims[i] < 0)
                    throw new IncorrectShapeException("negative dimensions not allowed");
                newSize *= newDims[i];
            }

            long itemsize = this.dtypesize;
            long oldSize = this.size;
            long oldBytes = oldSize * itemsize;
            long newBytes = newSize * itemsize;

            // The physical memory layout to preserve: F only when strictly F-contiguous (and not C),
            // matching NumPy's _array_fill_strides which keys off the array's current flags. 1-D and
            // 0-d arrays are C.
            char order = (shape.IsFContiguous && !shape.IsContiguous) ? 'F' : 'C';

            if (oldBytes != newBytes)
            {
                // 3. Reallocation is required. Ownership and reference checks fire ONLY here (a
                //    same-size resize is a pure reshape and skips them, exactly like NumPy).

                // a. Must own its data (not a slice/reshape/transpose view).
                if (this.Storage.IsView)
                    throw new IncorrectShapeException("cannot resize this array: it does not own its data");

                // b. Must not be shared with another array (unless the caller opts out via refcheck).
                if (refcheck && !this.Storage.InternalArray.IsUniquelyReferenced)
                    throw new IncorrectShapeException(
                        "cannot resize an array that references or is referenced\n" +
                        "by another array in this way.\n" +
                        "Use the np.resize function or refcheck=False");

                // Allocate the fresh buffer (uninitialized), copy the surviving prefix of the raw
                // memory (dtype-agnostic — resize operates on the contiguous byte buffer, which is
                // why an F-contiguous grow re-labels the same bytes with the new column-major
                // strides), then zero any grown tail.
                var freshStrides = order == 'F' ? FortranStrides(newDims) : ContiguousStrides(newDims);
                var fresh = new NDArray(this.typecode, new Shape(newDims, freshStrides), fillZeros: false);

                byte* src = this.Storage.Address + shape.offset * itemsize;
                byte* dst = fresh.Storage.Address;
                long copyBytes = Math.Min(oldBytes, newBytes);
                if (copyBytes > 0)
                    Buffer.MemoryCopy(src, dst, newBytes, copyBytes);
                if (newBytes > copyBytes)
                    NativeMemory.Clear(dst + copyBytes, (nuint)(newBytes - copyBytes));

                // ARC-correct in-place swap: take a reference on the fresh buffer for `this`, drop
                // `this`'s reference to the old buffer (freed here when uniquely owned; kept alive
                // for any other sharer under refcheck:false), then discard the fresh wrapper so the
                // net effect is `this` solely owning the new buffer.
                var newStorage = fresh.Storage;
                newStorage.InternalArray.TryAddRef();
                this.Storage.InternalArray?.Release();
                this.Storage = newStorage;
                fresh.Dispose();
            }
            else
            {
                // 4. Same total byte size → reshape in place. Preserve the physical order and any
                //    view offset/buffer, so a same-size resize of an F-contiguous array (or a
                //    single-segment slice) relabels the existing memory just as NumPy does.
                var strides = order == 'F' ? FortranStrides(newDims) : ContiguousStrides(newDims);
                var target = new Shape(newDims, strides, shape.offset, shape.bufferSize);
                this.Storage.SetShapeUnsafe(ref target);
            }
        }

        /// <summary>Row-major (C-order) strides for <paramref name="dims"/>.</summary>
        private static long[] ContiguousStrides(long[] dims)
        {
            var strides = new long[dims.Length];
            long acc = 1;
            for (int i = dims.Length - 1; i >= 0; i--)
            {
                strides[i] = acc;
                acc *= dims[i];
            }
            return strides;
        }

        /// <summary>Column-major (F-order) strides for <paramref name="dims"/>.</summary>
        private static long[] FortranStrides(long[] dims)
        {
            var strides = new long[dims.Length];
            long acc = 1;
            for (int i = 0; i < dims.Length; i++)
            {
                strides[i] = acc;
                acc *= dims[i];
            }
            return strides;
        }
    }
}
