using System;
using System.Diagnostics.CodeAnalysis;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newShape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape(Shape newShape)
        {
            return ReshapeCore(newShape, 'C');
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data, reading the elements in the
        ///     specified index order.
        /// </summary>
        /// <param name="newShape">The new shape (one dimension may be -1 — inferred, any order).</param>
        /// <param name="order">
        ///     Read/write index order for the reshape.
        ///     'C' (default) - row-major, 'F' - column-major,
        ///     'A' - 'F' when the source is F-contiguous and NOT C-contiguous, else 'C';
        ///     'K' raises NumPy's <c>ValueError("order 'K' is not permitted for reshaping")</c>.
        /// </param>
        /// <returns>
        ///     A VIEW whenever the reshape can be expressed over the existing strides
        ///     (contiguous-in-order relabel, or NumPy's <c>_attempt_nocopy_reshape</c> grouping —
        ///     which can yield a non-contiguous strided view); otherwise a view over an INTERNAL
        ///     copy taken in <paramref name="order"/> (so the result reports owndata=False either
        ///     way, exactly like NumPy's reshape).
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape(Shape newShape, char order)
        {
            return ReshapeCore(newShape, order);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newShape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape(ref Shape newShape)
        {
            return ReshapeCore(newShape, 'C');
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
        public NDArray reshape(int[] shape)
        {
            return reshape(Shape.ComputeLongShape(shape));
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
        public NDArray reshape(params long[] shape)
        {
            return ReshapeCore(new Shape(shape), 'C');
        }

        /// <summary>
        ///     The reshape engine — a port of NumPy's <c>_reshape_with_copy_arg</c>
        ///     (numpy/_core/src/multiarray/shape.c, <c>NPY_COPY_IF_NEEDED</c>), route for route:
        ///     <list type="number">
        ///     <item>order 'A' resolves via <c>PyArray_ISFORTRAN</c> (F-and-not-C); 'K' raises
        ///     NumPy's ValueError verbatim.</item>
        ///     <item>Same dims (checked BEFORE -1 resolution) → <c>PyArray_View</c>: a full alias
        ///     keeping shape, strides, offset, and flags (a same-shape reshape of a strided view
        ///     stays that strided view).</item>
        ///     <item><c>_fix_unknown_dimension</c> + size validation with NumPy's verbatim error
        ///     texts (delegated to the existing <see cref="Shape.Reshape(Shape, bool)"/> port).</item>
        ///     <item>Contiguous in the requested order (or size ≤ 1 / size 0, which NumPy always
        ///     flags contiguous) → relabel the same buffer window as a view.</item>
        ///     <item><see cref="Shape.TryNocopyReshape"/> → a (possibly non-contiguous) strided
        ///     VIEW sharing this array's memory — <c>arange(24).reshape(3,8)[:, ::2].reshape(2,6)</c>
        ///     is the byte-stride (96,16) view NumPy returns, and a reversed 1-D input splits to
        ///     negative-stride 2-D.</item>
        ///     <item>Otherwise copy in <paramref name="order"/> and return a view OVER that
        ///     internal copy — NumPy's reshape result reports owndata=False on the copy path too
        ///     (its base is the internal copy), e.g. reshape(order:'F') of a C source.</item>
        ///     </list>
        /// </summary>
        private NDArray ReshapeCore(Shape requestedShape, char order)
        {
            var src = this.Shape;

            // The uninitialized-shape sentinel has no dims/flags to reason about — keep the
            // legacy alias+relabel route for it verbatim.
            if (src.IsEmpty)
            {
                var legacy = Storage.Alias();
                legacy.Reshape(ref requestedShape, false);
                return new NDArray(legacy) { TensorEngine = TensorEngine };
            }

            // ---- order resolution (_reshape_with_copy_arg's head) ----
            if (order == 'A' || order == 'a')
                order = src.IsFContiguous && !src.IsContiguous ? 'F' : 'C'; // PyArray_ISFORTRAN
            else if (order == 'K' || order == 'k')
                throw new ValueError("order 'K' is not permitted for reshaping"); // NumPy verbatim
            else if (order == 'c')
                order = 'C';
            else if (order == 'f')
                order = 'F';
            else if (order != 'C' && order != 'F')
                OrderResolver.Resolve(order, src); // throws the house order-vocabulary ArgumentException

            // ---- quick same-shape check (BEFORE -1 resolution) → a plain full view ----
            var reqDims = requestedShape.dimensions ?? System.Array.Empty<long>();
            if (reqDims.Length == src.NDim)
            {
                bool same = true;
                for (int i = 0; i < reqDims.Length && same; i++)
                    same = src.dimensions[i] == reqDims[i];
                if (same)
                    return new NDArray(Storage.Alias()) { TensorEngine = TensorEngine };
            }

            // ---- -1 resolution + size validation (NumPy's verbatim texts, already ported) ----
            var resolvedDims = src.Reshape(requestedShape, @unsafe: false).dimensions;

            // ---- contiguous in the requested order: relabel the same buffer window ----
            // size ≤ 1 and size 0 arrays are always contiguous in NumPy's flag model, so they
            // relabel unconditionally (this is also what keeps TryNocopyReshape's non-zero-size
            // precondition honest).
            bool contigInOrder = order == 'F' ? src.IsFContiguous : src.IsContiguous;
            if (contigInOrder || src.size <= 1)
                return MakeReshapeView(this, resolvedDims, order);

            // ---- _attempt_nocopy_reshape: express the reshape over the existing strides ----
            if (src.TryNocopyReshape(resolvedDims, order == 'F', out var nocopyStrides))
            {
                var viewShape = new Shape((long[])resolvedDims.Clone(), nocopyStrides, src.offset,
                    src.bufferSize > 0 ? src.bufferSize : src.size);
                return new NDArray(Storage.Alias(viewShape)) { TensorEngine = TensorEngine };
            }

            // ---- copy in the requested order, then hand back a VIEW of that internal copy ----
            var copy = NDIter.CopyAs(this.typecode, this, order, TensorEngine);
            return MakeReshapeView(copy, resolvedDims, order);
        }

        /// <summary>
        ///     Relabel <paramref name="source"/>'s buffer window as <paramref name="dims"/> with
        ///     dense strides in <paramref name="order"/> — the "interpret the contiguous buffer
        ///     correctly" step of NumPy's reshape. Always an alias: offset and bufferSize carry
        ///     over (a C-contiguous split child reshapes in place at its own offset), writeability
        ///     is inherited, and the result reports owndata=False whether the source is the
        ///     original array or reshape's internal copy.
        /// </summary>
        private static NDArray MakeReshapeView(NDArray source, long[] dims, char order)
        {
            var srcShape = source.Shape;
            var dimsClone = (long[])dims.Clone();
            var viewShape = new Shape(dimsClone, Shape.ContiguousStridesFor(dimsClone, order == 'F'),
                srcShape.offset, srcShape.bufferSize > 0 ? srcShape.bufferSize : srcShape.size);
            return new NDArray(source.Storage.Alias(viewShape)) { TensorEngine = source.TensorEngine };
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newshape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape_unsafe(Shape newshape)
        {
            return reshape_unsafe(ref newshape);
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        /// </summary>
        /// <param name="newshape">The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.</param>
        /// <returns>This will be a new view object if possible; otherwise, it will be a copy. Note there is no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.reshape.html</remarks>
        public NDArray reshape_unsafe(ref Shape newshape)
        {
            var ret = Storage.Alias();
            ret.Reshape(ref newshape, true);
            return new NDArray(ret) { TensorEngine = TensorEngine };
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
        public NDArray reshape_unsafe(int[] shape)
        {
            return reshape_unsafe(Shape.ComputeLongShape(shape));
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
        public NDArray reshape_unsafe(params long[] shape)
        {
            var ret = Storage.Alias();
            ret.Reshape(shape, true);
            return new NDArray(ret) { TensorEngine = TensorEngine };
        }
    }
}
