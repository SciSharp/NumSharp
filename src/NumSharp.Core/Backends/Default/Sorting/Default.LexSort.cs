using System;
using NumSharp.Backends.Sorting;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>PyArray_LexSort</c>, as NumSharp composes it: the key-list checks in NumPy's order (an empty list,
        ///     a null key, keys of different shapes, the axis, which a 0-d key lets through as -1 or 0), then one stable
        ///     argsort pass per key from the first to the last, each re-sorting the running permutation, so the last key
        ///     decides first and the earlier keys break its ties. A StringDType key re-sorts the permutation in place with
        ///     NumPy's own pass (<see cref="VStringSort.LexPass"/>). Moved from the public <c>np.lexsort</c> by the np-layer
        ///     dtype plan's phase PE (E5), statement for statement; the public function keeps its class boundaries, and
        ///     this file sits beside it while it calls np.argsort and np.take_along_axis (ratchet R2).
        /// </remarks>
        [NDScoped]
        public override NDArray LexSort(NDArray[] keys, int axis = -1)
        {
            if (keys is null || keys.Length == 0)
                throw new TypeError("need sequence of keys with len > 0 in lexsort");
            for (int i = 0; i < keys.Length; i++)
                if (keys[i] is null)
                    throw new ArgumentNullException(nameof(keys), $"lexsort key {i} is null");

            var first = keys[0];
            for (int i = 1; i < keys.Length; i++)
            {
                if (keys[i].ndim != first.ndim)
                    throw new ValueError("all keys need to be the same shape");
                for (int d = 0; d < first.ndim; d++)
                    if (keys[i].Shape.dimensions[d] != first.Shape.dimensions[d])
                        throw new ValueError("all keys need to be the same shape");
            }

            int nd = first.ndim;
            int ax = axis;
            if (nd == 0 && (axis == 0 || axis == -1))
            {
                // NumPy lets axis={-1,0} slip through for 0-d keys (backwards compatibility).
            }
            else
            {
                ax = AxisSort.NormalizeAxis(axis, nd);
            }

            if (nd == 0)
                return NewScalar(0L);
            if (first.size <= 1)
                return NewArray(NPTypeCode.Int64, Shape.Adopt(first.Shape.dimensions), true); // [0] for one element, [] for none

            // First pass over an identity permutation IS the first key's stable argsort;
            // every later key re-sorts the permutation it receives (stably), so the LAST
            // key decides first and earlier keys only break its ties — NumPy's ordering.
            // Every pass asks for the STABLE kind: NumPy's PyArray_LexSort runs each key's NPY_STABLESORT argsort, and a
            // text key's default kind would be NumPy's introsort, whose order of equal strings is not the stable one (a
            // numeric key's argsort is one stable order for every kind).
            // A StringDType key has no legacy stable argsort: NumPy runs npy_atimsort with the dtype's compare over the
            // running index array, whose run detection reverses adjacent NaN-like NAs — VStringSort.LexPass re-sorts the
            // permutation in place exactly so (the first key's pass starts from the identity, as NumPy's index array does).
            NDArray perm;
            if (Backends.Strings.NDStringUfuncs.IsVString(keys[0]))
            {
                perm = IdentityPermutation(first, ax);
                VStringSort.LexPass(keys[0], perm, ax);
            }
            else
                perm = ArgSort(keys[0], ax, kind: "stable");
            for (int j = 1; j < keys.Length; j++)
            {
                if (Backends.Strings.NDStringUfuncs.IsVString(keys[j]))
                {
                    VStringSort.LexPass(keys[j], perm, ax);
                    continue;
                }
                var g = ArgSort(NDArray.take_along_axis(keys[j], perm, ax), ax, kind: "stable");
                perm = NDArray.take_along_axis(perm, g, ax);
            }
            return perm;
        }

        /// <summary>
        ///     NumPy's initial lexsort index array (<c>iptr[i] = i</c> along the axis): a fresh, writeable, C-contiguous int64
        ///     array of <paramref name="like"/>'s dimensions holding each element's index along <paramref name="axis"/>.
        /// </summary>
        /// <param name="like">The keys' shape donor (at least one dimension).</param>
        /// <param name="axis">The normalised axis.</param>
        /// <returns>The identity permutation of every line.</returns>
        private static NDArray IdentityPermutation(NDArray like, int axis)
        {
            var dims = like.Shape.dimensions;
            var lineShape = new long[dims.Length];
            for (int d = 0; d < dims.Length; d++)
                lineShape[d] = d == axis ? dims[d] : 1;
            // arange along the axis, broadcast over the other axes and materialized (the pass rewrites it in place).
            using var line = UnmanagedStorage.ArangeObj(0L, dims[axis], 1L, null).reshape(Shape.Adopt(lineShape));
            using var broadcast = NDArray.broadcast_to(line, Shape.Adopt((long[])dims.Clone()));
            return broadcast.copy();
        }
    }
}
