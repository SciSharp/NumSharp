using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        #region Shaping

        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        public void Reshape(params long[] dimensions)
        {
            if (dimensions == null || dimensions.Length == 0)
                throw new ArgumentException(nameof(dimensions));

            Reshape(dimensions, false);
        }

        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        /// <exception cref="ArgumentException">If <paramref name="dimensions"/>'s size == 0</exception>
        public void Reshape(long[] dimensions, bool @unsafe)
        {
            if (dimensions == null)
                throw new ArgumentNullException(nameof(dimensions));

            // NumPy behavior: non-contiguous arrays require a copy when reshaping
            // because Shape.Reshape creates standard C-order strides that don't
            // account for the original non-contiguous memory layout
            if (!_shape.IsContiguous)
            {
                // Make a contiguous copy of the data
                var contiguousData = CloneData();
                var cleanShape = _shape.Clean();
                SetInternalArray(contiguousData);
                _shape = cleanShape;
            }

            SetShapeUnsafe(_shape.Reshape(dimensions, @unsafe));
        }

        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        /// <exception cref="ArgumentException">If <paramref name="newShape"/>'s size == 0</exception>
        [MethodImpl(OptimizeAndInline)]
        public void Reshape(Shape newShape, bool @unsafe = false)
        {
            Reshape(ref newShape, @unsafe);
        }

        /// <summary>
        ///     Changes the shape representing this storage.
        /// </summary>
        /// <exception cref="IncorrectShapeException">If shape's size mismatches current shape size.</exception>
        /// <exception cref="ArgumentException">If <paramref name="newShape"/>'s size == 0</exception>
        [MethodImpl(OptimizeAndInline)]
        public void Reshape(ref Shape newShape, bool @unsafe = false)
        {
            // NumPy behavior: non-contiguous arrays require a copy when reshaping
            // because Shape.Reshape creates standard C-order strides that don't
            // account for the original non-contiguous memory layout
            if (!_shape.IsContiguous)
            {
                // Make a contiguous copy of the data
                var contiguousData = CloneData();
                var cleanShape = _shape.Clean();
                SetInternalArray(contiguousData);
                _shape = cleanShape;
            }

            this.SetShapeUnsafe(_shape.Reshape(newShape, @unsafe));
        }

        /// <summary>
        ///     Set the shape of this storage without checking if sizes match.
        /// </summary>
        /// <remarks>Used during broadcasting</remarks>
        protected internal void SetShapeUnsafe(Shape shape)
        {
            SetShapeUnsafe(ref shape);
        }

        /// <summary>
        ///     Set the shape of this storage without checking if sizes match.
        /// </summary>
        /// <remarks>Used during broadcasting. Uses bufferSize (not size) because
        /// broadcast shapes have logical size != physical buffer size.</remarks>
        protected internal void SetShapeUnsafe(ref Shape shape)
        {
            _shape = shape;
            // Use bufferSize for Count because broadcast shapes have stride=0 dimensions
            // where logical size (e.g., 5x5=25) exceeds physical buffer (e.g., 5 elements).
            // This ensures ArraySlice bounds checks use the actual buffer capacity.
            Count = _shape.bufferSize;
            // The incoming shape is usually freshly built (OWNDATA bit clear) — reconcile with this
            // storage's actual ownership so an in-place reshape/setflags does not strip an owner's bit
            // (NumPy: in-place shape/flag mutation never changes OWNDATA).
            OnReshaped();
        }

        protected internal void ExpandDimension(int axis)
        {
            _shape = _shape.ExpandDimension(axis);
            OnReshaped(); // rebuilt shape — keep OWNDATA mirroring this storage's ownership
        }

        #endregion

        #region Resize

        /// <summary>
        ///     NumPy's <c>PyArray_Resize</c> (<c>shape.c</c>), the body of <c>ndarray.resize</c>: <paramref name="self"/>
        ///     takes <paramref name="new_shape"/> in place. A changed byte size reallocates the buffer, copies the surviving
        ///     prefix of the raw memory and zero-fills a grown tail; the same byte size relabels the existing memory. The
        ///     memory order kept is F only for a strictly F-contiguous array.
        /// </summary>
        /// <param name="self">The array resized; its <c>Storage</c> is replaced when the byte size changes.</param>
        /// <param name="new_shape">The new shape: only its dimensions are read. A 0-d shape resizes to a scalar.</param>
        /// <param name="refcheck">Refuse to reallocate a buffer another array shares (NumPy's reference check).</param>
        /// <exception cref="IncorrectShapeException"><paramref name="self"/> is not single-segment; a dimension is
        ///     negative (checked up to the first zero dimension, as NumPy); or the byte size changes and
        ///     <paramref name="self"/> does not own its data or, under <paramref name="refcheck"/>, shares it.</exception>
        /// <exception cref="OutOfMemoryException">The new element or byte count overflows (NumPy's
        ///     <c>MemoryError</c>).</exception>
        /// <remarks>
        ///     <para>
        ///     The validation is <c>PyArray_Resize_int</c>'s, not the creation guard's: it stops at the first zero
        ///     dimension, so <c>(0, -1)</c> is a legal target, and the buffer is allocated by element count and then
        ///     relabelled, as NumPy mallocs <c>newnbytes</c> before it writes the dimensions.
        ///     </para>
        ///     <para>
        ///     The old buffer loses <paramref name="self"/>'s reference here (freed when nothing else holds it); a
        ///     <c>StringDType</c> buffer's surviving cells are duplicated into the new buffer's allocator, never copied as
        ///     raw bytes.
        ///     </para>
        /// </remarks>
        internal static unsafe void Resize(NDArray self, Shape new_shape, bool refcheck)
        {
            var shape = self.Shape;

            // 1. Single-segment (contiguous) requirement — always enforced first, exactly like
            //    NumPy's PyArray_ISONESEGMENT gate. 0-d arrays are trivially single-segment.
            if (shape.NDim != 0 && !shape.IsContiguous && !shape.IsFContiguous)
                throw new IncorrectShapeException("resize only works on single-segment arrays");

            // 2. Validate dimensions and compute the new element count. NumPy stops at the first
            //    zero dim (so a negative dim following a zero is NOT reported) and rejects negatives
            //    otherwise, with this exact message (distinct from np.resize's wording).
            var newDims = new_shape.dimensions ?? System.Array.Empty<long>();
            // The descriptor's item size (a StringDType cell is 16 bytes; its cells are moved through the allocators below,
            // never as raw bytes).
            long itemsize = self.Storage.ItemBytes;
            long newSize = 1;
            for (int i = 0; i < newDims.Length; i++)
            {
                if (newDims[i] == 0) { newSize = 0; break; }
                if (newDims[i] < 0)
                    throw new IncorrectShapeException("negative dimensions not allowed");

                // NumPy guards this product with npy_mul_sizes_with_overflow and answers a wrapped
                // one with PyErr_NoMemory — a MemoryError, NOT the ValueError that array CREATION
                // raises for the same arithmetic (probed: a.resize((2**62,2**62)) is MemoryError,
                // np.zeros((2**62,2**62)) is "array is too big"). Without the check the product
                // wraps to a small (often zero) count and the array is relabelled over a buffer
                // that was never allocated for it.
                if (newDims[i] > long.MaxValue / newSize)
                    throw new OutOfMemoryException();
                newSize *= newDims[i];
            }

            long oldSize = self.size;
            long oldBytes = oldSize * itemsize;
            if (itemsize > 0 && newSize > long.MaxValue / itemsize)
                throw new OutOfMemoryException();
            long newBytes = newSize * itemsize;

            // The physical memory layout to preserve: F only when strictly F-contiguous (and not C),
            // matching NumPy's _array_fill_strides which keys off the array's current flags. 1-D and
            // 0-d arrays are C.
            char order = (shape.IsFContiguous && !shape.IsContiguous) ? 'F' : 'C';

            if (oldBytes != newBytes)
            {
                // 3. Reallocation is required. Ownership and reference checks fire ONLY here (a
                //    same-size resize is a pure reshape and skips them, exactly like NumPy).

                // a. Must own its data — not a slice/reshape/transpose view, and not an externally-based
                //    array either (a memmap's memory belongs to the file mapping; NumPy's memmap.resize
                //    refuses with the same text, and reallocating a mapped region would be fatal).
                //    Note NumPy checks OWNERSHIP only, not WRITEABLE: a read-only OWNER resizes fine.
                if (!self.Storage.OwnsData)
                    throw new IncorrectShapeException("cannot resize this array: it does not own its data");

                // b. Must not be shared with another array (unless the caller opts out via refcheck).
                if (refcheck && !self.Storage.InternalArray.IsUniquelyReferenced)
                    throw new IncorrectShapeException(
                        "cannot resize an array that references or is referenced\n" +
                        "by another array in this way.\n" +
                        "Use the np.resize function or refcheck=False");

                // Allocate the fresh buffer (uninitialized), copy the surviving prefix of the raw
                // memory (dtype-agnostic — resize operates on the contiguous byte buffer, which is
                // why an F-contiguous grow re-labels the same bytes with the new column-major
                // strides), then zero any grown tail.
                var freshStrides = order == 'F' ? FortranStrides(newDims) : ContiguousStrides(newDims);

                // Allocate by ELEMENT COUNT and relabel, rather than by the requested dimensions.
                // The validation above is PyArray_Resize_int's, which is a DIFFERENT loop from the
                // creation-time one AllocationGuard enforces: resize stops at the first zero dim,
                // so `(0, -1)` is a legal resize target while `np.zeros((0, -1))` is not. NumPy has
                // the same split — it mallocs newnbytes and only then writes the dimensions.
                // The DESCRIPTOR allocates: a fixed-width string's width lives on it, not on its storage lane (an S5
                // buffer grows by 5 bytes an element, and the zeroed tail reads b'' / '', NumPy's grown elements).
                var fresh = new NDArray(self.dtype, Shape.Adopt(new[] { newSize }), fillZeros: false);

                byte* src = self.Storage.Address + shape.offset * itemsize;
                byte* dst = fresh.Storage.Address;
                long copyBytes = Math.Min(oldBytes, newBytes);
                if (self.Storage.IsVString)
                {
                    // A StringDType buffer: the surviving cells are DUPLICATED into the fresh buffer's own allocator (a raw
                    // copy would leave arena offsets and heap pointers that only the old allocator can read, and the old
                    // buffer's clear loop would free those heap strings). The grown tail stays zero: '' cells, NumPy's
                    // zero-filled new elements. The truncated cells' strings are freed with the old buffer. NumPy reallocates
                    // the cells and keeps its allocator, so its arena offsets survive; here the copy's arena is compact (the
                    // values are the same, the raw cells of tobytes() can differ for strings longer than 15 bytes).
                    long cells = copyBytes / itemsize;
                    var inAllocator = self.Storage.VStringSlice.Allocator;
                    var outAllocator = fresh.Storage.VStringSlice.Allocator;
                    Backends.Strings.VStringAllocator.acquire_allocators(inAllocator, outAllocator);
                    try
                    {
                        Backends.Strings.NDVStringCopy.CopyRun(src, itemsize, dst, itemsize, cells, inAllocator, outAllocator, null);
                    }
                    catch
                    {
                        fresh.Dispose(); // its clear loop frees the strings duplicated so far
                        throw;
                    }
                    finally
                    {
                        Backends.Strings.VStringAllocator.release_allocators(inAllocator, outAllocator);
                    }
                }
                else
                {
                    if (copyBytes > 0)
                        Buffer.MemoryCopy(src, dst, newBytes, copyBytes);
                    if (newBytes > copyBytes)
                        NativeMemory.Clear(dst + copyBytes, (nuint)(newBytes - copyBytes));
                }

                // ARC-correct in-place swap: take a reference on the fresh buffer for `self`, drop
                // `self`'s reference to the old buffer (freed here when uniquely owned; kept alive
                // for any other sharer under refcheck:false), then discard the fresh wrapper so the
                // net effect is `self` solely owning the new buffer.
                var newStorage = fresh.Storage;
                var freshShape = Shape.Adopt(newDims, freshStrides);
                newStorage.SetShapeUnsafe(ref freshShape);
                newStorage.InternalArray.TryAddRef();
                self.Storage.InternalArray?.Release();
                self.Storage = newStorage;
                fresh.Dispose();
            }
            else
            {
                // 4. Same total byte size → reshape in place. Preserve the physical order and any
                //    view offset/buffer, so a same-size resize of an F-contiguous array (or a
                //    single-segment slice) relabels the existing memory just as NumPy does.
                var strides = order == 'F' ? FortranStrides(newDims) : ContiguousStrides(newDims);
                var target = new Shape(newDims, strides, shape.offset, shape.bufferSize);
                self.Storage.SetShapeUnsafe(ref target);
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

        #endregion
    }
}
