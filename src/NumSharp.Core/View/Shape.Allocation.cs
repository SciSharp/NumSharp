using System;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    public partial struct Shape
    {
        // The layout a FRESH allocation installs — NumPy's PyArray_NewFromDescr_int with no data pointer
        // (numpy/_core/src/multiarray/ctors.c). NumPy only ever hands that constructor strides it computed for the
        // buffer it is about to allocate: _array_fill_strides' C (or F) strides for a dims-only request (np.empty,
        // np.zeros, np.ones, np.full, every random sampler's np.empty(size)), or PyArray_NewLikeArray's KEEPORDER
        // permutation of them (copy / astype / *_like with order='K'). Caller strides reach a fresh buffer only through
        // ndarray.__new__, and that path validates them first (PyArray_CheckStrides: "strides is incompatible with
        // shape of requested array and size of buffer").
        //
        // A NumSharp Shape carries a view's strides, offset and buffer size next to its dimensions, so the natural port
        // np.empty(x.Shape) of NumPy's np.empty(x.shape) hands the allocator the VIEW's addressing. Installed over a
        // buffer of `size` elements it reads and writes past the end: for x = arange(20)["::4"] (strides 4) logical
        // element i sits at i * 4 of a 5-element buffer. This file is the one place that decides which layouts a fresh
        // buffer may keep.

        /// <summary>
        ///     The layout to install over a FRESH, owned buffer of exactly <see cref="size"/> elements allocated for this
        ///     shape's dimensions. Every allocating entry point (<see cref="Backends.UnmanagedStorage.Allocate(Shape, DType, bool)"/>
        ///     and its overloads, <c>np.ones</c>, <c>np.full</c>) routes its shape through here, so a shape read off a view
        ///     (<c>np.empty(x.Shape)</c>, <c>rng.integers(0, 100, x.Shape)</c>) can no longer address outside the new
        ///     buffer.
        /// </summary>
        /// <returns>
        ///     This shape when it already describes a valid fresh layout — offset 0, a buffer size equal to its element
        ///     count, and strides that visit every element of <c>[0, size)</c> exactly once with no negative step (C, F,
        ///     or a KEEPORDER permutation such as <see cref="KeepOrder"/> builds) — with <see cref="ArrayFlags.WRITEABLE"/>
        ///     set. Any other shape (a stepped, offset, reversed or broadcast view, a contiguous slice whose buffer is
        ///     larger than the slice) is replaced by <see cref="Clean"/>: its dimensions laid out in C order, NumPy's
        ///     answer for a dims-only request. An uninitialized shape (<see cref="IsEmpty"/>) is returned unchanged, so
        ///     each caller keeps its own error for it.
        /// </returns>
        /// <remarks>
        ///     <para>
        ///     <b>Why a dense layout is kept, not re-laid out in C order.</b> NumSharp builds the KEEPORDER output of
        ///     <c>copy(order='K')</c>, <c>astype</c>, the ufunc paths (<see cref="NpyIterOutputShape"/>) and
        ///     <c>concatenate</c> (<see cref="ConcatOutputShape"/>) as an F or permuted Shape and allocates it through
        ///     this funnel, exactly as NumPy hands <c>PyArray_NewLikeArray</c>'s strides to its constructor. A dense
        ///     Shape read off a view of a whole array (<c>x.T.Shape</c>) is indistinguishable from such a request, so it
        ///     keeps its layout too. That is memory-safe; callers that need NumPy's dims-only C layout regardless (the
        ///     random samplers' <c>size</c>) call <see cref="AsClean"/> themselves.
        ///     </para>
        ///     <para>
        ///     <b>Why the buffer size and offset must match.</b> A fresh buffer has no parent: an offset or a larger
        ///     buffer size are a view's bookkeeping, and code that trusts them on a fresh array reads past it
        ///     (<c>AliasComplexLane</c> sizes its float64 lane by <see cref="bufferSize"/>). A contiguous slice such as
        ///     <c>arange(20)[":5"]</c> has the C layout of its dimensions but reports a buffer of 20, so it is re-laid out.
        ///     </para>
        ///     <para>
        ///     <b>Why the result is always writeable.</b> The buffer is brand-new owned memory, so it is writeable even
        ///     when the shape was borrowed from a read-only source — an elementwise result whose shape is the read-only
        ///     operand's shape (<c>mmap_r + 1</c>) must not inherit the operand's flag, or it diverges from NumPy. Views
        ///     never come through here: they alias, which correctly inherits read-only.
        ///     </para>
        ///     <para>
        ///     <b>Cost.</b> The check every allocation pays is three comparisons against cached fields and flags. Only a
        ///     shape that is neither C- nor F-contiguous runs the permutation check (<see cref="IsDensePermutation"/>), and
        ///     only a rejected shape allocates (its cloned dimensions and C strides).
        ///     </para>
        ///     <para>
        ///     The returned shape SHARES this shape's dimension and stride arrays when it is kept, as the allocators always
        ///     did; <see cref="Clean"/> clones them.
        ///     </para>
        /// </remarks>
        internal readonly Shape ForFreshAllocation()
        {
            // default(Shape) has no dimensions to lay out; the allocators raise their own error for it.
            if (IsEmpty)
                return this;

            // A fresh layout starts at element 0 of a buffer exactly `size` long (0 = unset, the BufferSize convention)
            // and steps densely: C/F are the cached flags (never set on a broadcast of size > 0), anything else must be
            // a dense permutation. The `||` keeps the permutation walk off the hot path.
            bool fresh = offset == 0
                         && (bufferSize == size || bufferSize == 0)
                         && ((_flags & (int)(ArrayFlags.C_CONTIGUOUS | ArrayFlags.F_CONTIGUOUS)) != 0 || IsDensePermutation());
            if (!fresh)
                return Clean();

            return IsWriteable ? this : WithFlags(flagsToSet: ArrayFlags.WRITEABLE);
        }

        /// <summary>
        ///     NumPy's <c>np.empty(size)</c> layout for a shape used only for its DIMENSIONS — a random sampler's <c>size</c>,
        ///     which NumPy receives as a tuple: the C layout of the dimensions over a buffer of exactly <see cref="size"/>
        ///     elements. Use it where the result must be C-ordered whatever layout the caller's shape carried (a sampler fills
        ///     in memory order, so even a dense F layout would transpose the draws); use <see cref="ForFreshAllocation"/> where a
        ///     dense layout is a deliberate request.
        /// </summary>
        /// <returns>
        ///     This shape when it already is that layout — offset 0, a buffer size equal to its element count (or unset), and
        ///     exactly the C strides of its dimensions — so the common call allocates nothing; otherwise <see cref="Clean"/>.
        ///     An uninitialized shape (<see cref="IsEmpty"/>, NumPy's <c>size=None</c>) is returned unchanged: it names no
        ///     dimensions, and the callers branch on it first.
        /// </returns>
        /// <remarks>
        ///     <para>
        ///     The <c>as</c> prefix follows NumPy's <c>asarray</c>/<c>array</c> pair: <see cref="Clean"/> always builds a new
        ///     shape (cloned dimensions, fresh strides), this returns its input when the input already conforms. A kept shape
        ///     therefore shares its dimension and stride arrays with the caller's, exactly as passing the shape straight to an
        ///     allocator always did.
        ///     </para>
        ///     <para>
        ///     The stride comparison is exact, not the C-contiguity flag: the flag skips extent-1 axes and is set for every
        ///     empty shape, so a stepped empty view (strides <c>(0, 2)</c>) or a size-1 axis with a stray stride would pass it
        ///     and leave those strides on a result NumPy lays out as <c>np.empty(size)</c>.
        ///     </para>
        /// </remarks>
        internal readonly Shape AsClean()
        {
            // size=None has no dimensions to lay out; every caller branches on it before allocating.
            if (IsEmpty)
                return this;

            // The view baggage first (O(1)), then the strides against the C strides of the dimensions, innermost first:
            // the same running product ComputeContiguousStrides builds, so an empty extent compares equal too.
            if (offset == 0 && (bufferSize == size || bufferSize == 0))
            {
                bool c = true;
                long expected = 1;
                for (int i = dimensions.Length - 1; i >= 0 && c; i--)
                {
                    c = strides[i] == expected;
                    expected *= dimensions[i];
                }

                if (c)
                    return this;
            }

            return Clean();
        }

        /// <summary>
        ///     The layout NumPy's <c>ndarray(shape, buffer=values, order=order)</c> installs over a CALLER'S buffer: this
        ///     shape's dimensions with C strides (F strides when <paramref name="order"/> is <c>'F'</c>), offset 0, a buffer
        ///     size equal to the element count, writeable. The public wrap constructors <c>new NDArray(Array, Shape, char)</c>
        ///     and <c>new NDArray(IArraySlice, Shape, char)</c> route their <c>shape</c> argument through here, so the strides,
        ///     offset, buffer size and flags a view's shape carries (<c>x.Shape</c> of <c>a[::4]</c>, <c>a[::-1]</c>,
        ///     <c>a.T</c> or a broadcast) never address the caller's values.
        /// </summary>
        /// <param name="order">The PHYSICAL order: <c>'F'</c> for column-major, anything else for row-major. Callers resolve
        ///     the user's char first with <see cref="OrderResolver.ResolveBufferOrder"/>.</param>
        /// <returns>
        ///     This shape when it already is exactly that layout and writeable, so the common call allocates nothing; the
        ///     same layout with <see cref="ArrayFlags.WRITEABLE"/> set when only the flag differs; otherwise a new shape of
        ///     the same dimensions in that layout (its dimension array is a copy, so it shares nothing with this shape). An
        ///     uninitialized shape (<see cref="IsEmpty"/>) is returned unchanged: the wrap constructors replace it with the
        ///     values' own shape before they call here.
        /// </returns>
        /// <remarks>
        ///     <para>
        ///     <b>NumPy's rule.</b> <c>array_new</c> (<c>numpy/_core/src/multiarray/arrayobject.c</c>) reads <c>shape</c> as a
        ///     sequence of dimensions and, with a buffer and no <c>strides</c>, lets <c>PyArray_NewFromDescr_int</c> compute
        ///     them with <c>_array_fill_strides</c>: C order, or F order for <c>order='F'</c>. Probed against NumPy 2.4.2:
        ///     <c>np.ndarray(a[::4].shape, buffer=v)</c> reads <c>v</c> densely, <c>np.ndarray(a[::-1].shape, buffer=v)</c> reads
        ///     it forwards, and <c>np.ndarray(a.T.shape, buffer=v)</c> reads it in C order.
        ///     </para>
        ///     <para>
        ///     <b>Why not <see cref="ForFreshAllocation"/>.</b> That rule keeps a dense F or permuted layout, because NumSharp
        ///     allocates its KEEPORDER outputs that way. A wrap is a dimensions-only request, so even a dense transposed
        ///     layout is re-laid out in the requested order, as NumPy does.
        ///     </para>
        ///     <para>
        ///     <b>Why the strides are compared exactly.</b> The C and F flags skip extent-1 axes and are set on every empty
        ///     shape, so a stepped empty view (strides <c>(0, 1)</c> for dims <c>(0, 3)</c>) would pass them and keep strides
        ///     NumPy does not produce (it gives <c>(3, 1)</c>).
        ///     </para>
        ///     <para>
        ///     <b>Why the result is always writeable.</b> The buffer is the caller's own memory, so the read-only flag of the
        ///     shape's source array (a broadcast, a read-only memmap) is not the new array's (NumPy: writeable).
        ///     </para>
        /// </remarks>
        internal readonly Shape ForWrappedBuffer(char order)
        {
            // default(Shape) has no dimensions; the wrap constructors replace it with the values' own shape first.
            if (IsEmpty)
                return this;

            bool fortran = order == 'F';
            if (offset == 0 && (bufferSize == size || bufferSize == 0) && HasExactContiguousStrides(fortran))
                return IsWriteable ? this : WithFlags(flagsToSet: ArrayFlags.WRITEABLE);

            // Clean() clones the dimensions and computes C strides; the F constructor gets a clone too, so the new layout
            // never shares a dimension array with the caller's shape.
            return fortran ? new Shape((long[])dimensions.Clone(), 'F') : Clean();
        }

        /// <summary>
        ///     Whether the strides are exactly the C strides of the dimensions (or the F strides when
        ///     <paramref name="fortran"/> is true), axis by axis, in NumSharp's running-product convention (the one
        ///     <c>ComputeContiguousStrides</c> and <c>ComputeFContiguousStrides</c> use, where an empty extent zeroes the
        ///     strides outside it).
        /// </summary>
        /// <param name="fortran">True to compare against the F strides, false against the C strides.</param>
        /// <returns>True when every axis has exactly the expected stride, including extent-1 and empty axes, which the
        ///     cached C/F flags skip.</returns>
        private readonly bool HasExactContiguousStrides(bool fortran)
        {
            var dims = dimensions;
            var strd = strides;
            if (strd == null || strd.Length != dims.Length)
                return false;

            long expected = 1;
            if (fortran)
            {
                for (int i = 0; i < dims.Length; i++)
                {
                    if (strd[i] != expected)
                        return false;
                    expected *= dims[i];
                }
            }
            else
            {
                for (int i = dims.Length - 1; i >= 0; i--)
                {
                    if (strd[i] != expected)
                        return false;
                    expected *= dims[i];
                }
            }

            return true;
        }

        /// <summary>
        ///     Whether the strides visit every element of <c>[0, size)</c> exactly once with no negative step, in SOME axis
        ///     order — the neither-C-nor-F layouts NumPy's KEEPORDER allocates (<c>a.transpose(1, 0, 2).copy(order='K')</c>
        ///     keeps the transposed stride order). Use it only on a shape whose offset is 0; it does not read
        ///     <see cref="offset"/>.
        /// </summary>
        /// <returns>
        ///     True when, ignoring extent-1 axes, the strides sorted ascending are exactly <c>1, d0, d0*d1, …</c> of their
        ///     own extents; false for a gap (a stepped view), an overlap (a stride-0 broadcast, two axes on one stride), a
        ///     negative stride (a reversed view, which also needs an offset), an empty extent (the caller's flags already
        ///     decided those), or more than 64 axes of extent above 1 (no valid allocation has them: each doubles the
        ///     element count at least).
        /// </returns>
        /// <remarks>
        ///     NoInlining keeps this stack scratch and loop out of <see cref="ForFreshAllocation"/>, which every allocation
        ///     runs and which only reaches here for the rare permuted layout.
        /// </remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private readonly bool IsDensePermutation()
        {
            var dims = dimensions;
            var strd = strides;
            if (dims == null || strd == null || strd.Length != dims.Length)
                return false;

            // The axes that move the address, sorted by stride ascending as they are found (insertion sort: at most 64).
            Span<int> axes = stackalloc int[64];
            int count = 0;
            for (int i = 0; i < dims.Length; i++)
            {
                long extent = dims[i];
                // An extent-1 axis is only ever read at index 0, so its stride never moves the address (NumPy's flags
                // skip it the same way); a ndmin-padded shape is mostly such axes.
                if (extent == 1)
                    continue;
                long stride = strd[i];
                if (extent <= 0 || stride <= 0 || count == axes.Length)
                    return false;

                int j = count++;
                while (j > 0 && strd[axes[j - 1]] > stride)
                {
                    axes[j] = axes[j - 1];
                    j--;
                }

                axes[j] = i;
            }

            // Dense means each stride is the element count of everything inside it: a larger stride leaves a gap, an
            // equal one overlaps the previous axis.
            long expected = 1;
            for (int j = 0; j < count; j++)
            {
                int axis = axes[j];
                if (strd[axis] != expected)
                    return false;
                expected *= dims[axis];
            }

            return true;
        }
    }
}
