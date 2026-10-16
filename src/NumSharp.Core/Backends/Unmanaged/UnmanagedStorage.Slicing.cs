using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        #region Slicing

        [MethodImpl(Inline)]
        public UnmanagedStorage GetView(string slicing_notation) => GetView(Slice.ParseSlices(slicing_notation));

        [MethodImpl(Inline)]
        public UnmanagedStorage GetView(params Slice[] slices)
        {
            if (slices == null)
                throw new ArgumentNullException(nameof(slices));
            // deal with ellipsis and newaxis if any before continuing into GetViewInternal
            int ellipsis_count = 0;
            int newaxis_count = 0;
            foreach (var slice in slices)
            {
                if (slice.IsEllipsis)
                    ellipsis_count++;
                if (slice.IsNewAxis)
                    newaxis_count++;
            }

            // Too many indices: count the AXIS-CONSUMING slices (everything except newaxis
            // and ellipsis). NumPy raises IndexError when this exceeds the array's rank — an
            // ellipsis only ABSORBS axes, so it can never rescue an already-over-full index
            // (A[:, :, :] / A[..., :, :, :] on a 2-D array both raise). Previously the extra
            // slices were silently dropped by Shape.Slice (it iterates only over NDim).
            int consuming = slices.Length - newaxis_count - ellipsis_count;
            if (consuming > Shape.NDim)
                throw new IndexError($"too many indices for array: array is {Shape.NDim}-dimensional, but {consuming} were indexed");

            // deal with ellipsis
            if (ellipsis_count > 1)
                throw new ArgumentException("IndexError: an index can only have a single ellipsis ('...')");
            else if (ellipsis_count == 1)
                slices = ExpandEllipsis(slices).ToArray();

            // NumPy's prepare_index checks every integer index against the axis it consumes before any view exists:
            // outside [-dim, dim) is "index 5 is out of bounds for axis 0 with size 3", the axis counted in THIS array
            // (a new axis consumes none, so the check runs before the newaxis expansion below renumbers the axes).
            // Shape.Slice's own check knows no axis number (its ArgumentException reached a strided view's notation
            // index) and let index 0 of an empty axis through (a[..., 0] of a (2, 0) array returned shape (2,)).
            int sourceAxis = 0;
            foreach (var slice in slices)
            {
                if (slice.IsNewAxis)
                    continue;
                if (slice.IsIndex)
                {
                    long index = slice.Start ?? 0;
                    long dim = Shape.dimensions[sourceAxis];
                    if (index < -dim || index >= dim)
                        throw new IndexError($"index {index} is out of bounds for axis {sourceAxis} with size {dim}");
                }
                sourceAxis++;
            }

            // deal with newaxis
            if (newaxis_count > 0)
            {
                var view = this;
                for (var axis = 0; axis < slices.Length; axis++)
                {
                    var slice = slices[axis];
                    if (slice.IsNewAxis)
                    {
                        slices[axis] = Slice.All;
                        view = view.Alias(view.Shape.ExpandDimension(axis));
                    }
                }

                return view.GetViewInternal(slices);
            }

            // slicing without newaxis
            return GetViewInternal(slices);
        }

        [MethodImpl(Inline)]
        [SuppressMessage("ReSharper", "PossibleInvalidOperationException")]
        private UnmanagedStorage GetViewInternal(params Slice[] slices)
        {
            // NOTE: GetViewInternal can not deal with Slice.Ellipsis or Slice.NewAxis!
            // For contiguous shapes, if all slices are indices, we can use GetData to get
            // a direct memory slice. For non-contiguous shapes (stepped, transposed), we
            // must go through the full slice path to create a proper view.
            if (_shape.IsContiguous)
            {
                // Every slice an integer — or all but a trailing ':' (incase it is a trailing :, e.g. [2,2, :] in a shape
                // (3,3,5,5) -> (5,5)) — is a direct memory slice. The integers are counted first, so the index array is
                // allocated only when this case is taken (a slice view used to allocate it too, then jump past it).
                int leadingIndices = 0;
                while (leadingIndices < slices.Length && slices[leadingIndices].IsIndex)
                    leadingIndices++;
                if (leadingIndices == slices.Length || (leadingIndices == slices.Length - 1 && slices[leadingIndices] == Slice.All))
                {
                    var indices = new long[leadingIndices];
                    for (var i = 0; i < leadingIndices; i++)
                        indices[i] = slices[i].Start.Value;
                    return GetData(indices);
                }
            }

            //perform a regular slicing

            // In case the slices selected are all ":" - just return an alias
            if (AllFullSlices(slices))
                return Alias();

            // Broadcast arrays: DO NOT materialize! Shape.Slice correctly handles stride=0.
            // The old code materialized broadcast data, losing stride=0 information.
            // NumPy preserves stride=0 when slicing broadcast views.
            // Example: broadcast_to([1,2,3], (4,3)) with strides (0,1)
            //   [:, 0] should give shape (4,) with stride (0,) - NOT materialized stride (3,)
            //
            // NOTE: Cumsum and other axis reductions need this to work correctly with
            // broadcast arrays without requiring memory-expensive .copy() workarounds.

            var slicedShape = _shape.Slice(slices);

            // Handle empty slices (e.g., a[100:200] on a 10-element array)
            // NumPy returns an empty array with shape (0,) or similar
            if (slicedShape.size == 0)
            {
                // Create empty storage with correct dtype and shape. It is still a VIEW of the parent
                // (NumPy: owndata=False, base set — there are no elements, so the fresh zero-length
                // buffer shares nothing, but the base chain carries owndata and the setflags(write=True)
                // base-writeability answer).
                // A string lane has no width of its own (the descriptor carries it), so its empty view keeps the
                // parent's descriptor through a zero-length slice of the same layout (fixed-width or StringDType).
                var emptySlice = AllocateLike(0, false);
                var emptyStorage = new UnmanagedStorage();
                emptyStorage._Allocate(slicedShape, emptySlice);
                emptyStorage._baseStorage = _baseStorage ?? this;
                // A lane-sharing descriptor (longdouble, Windows' intc) is not on the buffer: the empty view carries it.
                emptyStorage.AdoptLaneSharingDescr(_descr);
                emptyStorage.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL ⟹ !OWNDATA)
                return emptyStorage;
            }

            // NumPy-aligned optimization: For contiguous slices, slice the InternalArray directly
            // and create a fresh shape with offset=0. This matches NumPy's data pointer adjustment
            // for contiguous views and makes IsSliced=false for contiguous slices.
            if (slicedShape.IsContiguous && slicedShape.offset > 0)
            {
                // Create a fresh contiguous shape (no offset)
                var freshShape = Shape.Adopt(slicedShape.dimensions);
                // A view of a non-writeable array (broadcast, or a read-only 'r' memmap) must stay
                // non-writeable — the fresh shape above defaults WRITEABLE back to true, which would let
                // a write reach read-only memory (a hard segfault on memory-mapped read-only pages, and
                // a silent NumPy-divergence for broadcasts). Shape.Slice already cleared it on
                // slicedShape; carry that through.
                if (!slicedShape.IsWriteable)
                    freshShape = freshShape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                var view = new UnmanagedStorage(InternalArray.Slice(slicedShape.offset, slicedShape.size), freshShape);
                view._baseStorage = _baseStorage ?? this;
                // A lane-sharing descriptor (longdouble, Windows' intc) is not on the buffer: the new view carries it.
                view.AdoptLaneSharingDescr(_descr);
                view.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL ⟹ !OWNDATA)
                return view;
            }

            // Non-contiguous slices: create an alias with the sliced shape.
            // The slicedShape contains the correct offset and strides computed by Shape.Slice().
            // Views with non-zero offset or non-standard strides use coordinate-based access.
            return Alias(slicedShape);
        }

        /// <summary>
        ///     Whether every slice equals <see cref="Slice.All"/> (<c>Equals(Slice.All, s)</c>: a <see cref="Slice"/> with no
        ///     start, no stop and step 1) — the index that selects the whole array, whose view is a plain alias.
        /// </summary>
        /// <param name="slices">The slices (no ellipsis or new axis left).</param>
        /// <returns>True when each one is a full slice; true for no slices.</returns>
        private static bool AllFullSlices(Slice[] slices)
        {
            foreach (var s in slices)
                if (!Equals(Slice.All, s))
                    return false;
            return true;
        }

        private IEnumerable<Slice> ExpandEllipsis(Slice[] slices)
        {
            // count dimensions without counting ellipsis or newaxis
            var count = 0;
            foreach (var slice in slices)
            {
                if (slice.IsNewAxis || slice.IsEllipsis)
                    continue;
                count++;
            }

            // expand 
            foreach (var slice in slices)
            {
                if (slice.IsEllipsis)
                {
                    for (int i = 0; i < Shape.NDim - count; i++)
                        yield return Slice.All;
                    continue;
                }

                yield return slice;
            }
        }

        #endregion
    }
}
