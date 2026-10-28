// The array-object members TensorEngine shipped in 0.70.0: the storage factory, CreateNDArray, the transposing
// views, the cast and the boolean-mask indexers, plus Diagonal (added since, unreleased; it leaves the engine in
// the plan's E7). They stay for compatibility. The array object's operations belong to NDArray,
// UnmanagedStorage and Shape, and no new member of this kind is added (docs/plans/np-layer-dtype-logic.md, E7).

using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     Get storage for given <paramref name="dtype"/>.
        /// </summary>
        public abstract UnmanagedStorage GetStorage(Type dtype);

        /// <summary>
        ///     Get storage described by the dtype DESCRIPTOR <paramref name="descr"/> (NumPy's <c>PyArray_NewFromDescr</c>
        ///     entry — the descriptor becomes the array's <c>dtype</c>, its class supplies the storage lane). The default
        ///     routes through <see cref="GetStorage(Type)"/>, which is exact for every builtin (one canonical descriptor
        ///     per lane); an engine that carries parametric descriptors overrides it.
        /// </summary>
        public virtual UnmanagedStorage GetStorage(DType descr)
        {
            if (descr is null)
                throw new ArgumentNullException(nameof(descr));
            return GetStorage((Type)descr);
        }

        public abstract NDArray CreateNDArray(Shape shape, DType dtype = null, Array buffer = null, char order = 'C');
        public abstract NDArray CreateNDArray(Shape shape, DType dtype = null, IArraySlice buffer = null, char order = 'C');
        public abstract NDArray Transpose(NDArray nd, int[] premute = null);
        public abstract NDArray SwapAxes(NDArray nd, int axis1, int axis2);
        /// <summary>
        ///     NumPy's <c>np.moveaxis</c>: the axes of <paramref name="source"/> moved to the positions of
        ///     <paramref name="destination"/>, the other axes keeping their order (a view).
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="source">The axes to move (negative values count from the end).</param>
        /// <param name="destination">Their new positions.</param>
        /// <returns>A view with the permuted axes.</returns>
        /// <exception cref="AxisError">An axis is out of range ("source: …" / "destination: …").</exception>
        /// <exception cref="ValueError">An argument repeats an axis, or the two have different lengths.</exception>
        public abstract NDArray MoveAxis(NDArray nd, int[] source, int[] destination);
        public abstract NDArray RollAxis(NDArray nd, int axis, int start = 0);

        /// <summary>
        ///     NumPy's <c>np.diagonal(a, offset, axis1, axis2)</c> (<c>PyArray_Diagonal</c>): a READ-ONLY view of the
        ///     diagonals of the 2-D sub-arrays spanned by <paramref name="axis1"/> and <paramref name="axis2"/>, appended as
        ///     the last axis; no element is read or copied.
        /// </summary>
        /// <param name="a">The array (at least 2-D).</param>
        /// <param name="offset">The diagonal's offset from the main one: positive above it (along
        ///     <paramref name="axis2"/>), negative below (along <paramref name="axis1"/>).</param>
        /// <param name="axis1">The first axis of the 2-D sub-arrays (negative counts from the end).</param>
        /// <param name="axis2">The second axis of the 2-D sub-arrays.</param>
        /// <returns>A read-only view sharing <paramref name="a"/>'s storage: <paramref name="a"/>'s shape without the two
        ///     axes, the diagonal (of length 0 when the offset leaves none) as the last axis.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="a"/> has fewer than two dimensions ("diag requires an array of at
        ///     least two dimensions"), or the two axes are the same axis ("axis1 and axis2 cannot be the same").</exception>
        /// <exception cref="AxisError">An axis is out of range: "axis1: axis …" / "axis2: axis …" (axis1 checked
        ///     first).</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, P1b): the public <c>np.diagonal</c> applies the class
        ///     boundaries and calls this, and <see cref="Trace"/> reads its diagonal through it.</remarks>
        public abstract NDArray Diagonal(NDArray a, int offset = 0, int axis1 = 0, int axis2 = 1);
        public abstract NDArray Cast(NDArray x, Type dtype, bool copy);

        // Boolean masking
        public abstract NDArray BooleanMask(NDArray arr, NDArray mask);

        public abstract void BooleanMaskSet(NDArray arr, NDArray mask, NDArray value);
    }
}
