using System;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    ///     Broadcasting operations for Shape.
    /// </summary>
    /// <remarks>
    ///     Based on https://numpy.org/doc/stable-1.16.1/user/basics.broadcasting.html
    ///     These are pure shape operations - they compute output shapes without touching array data.
    /// </remarks>
    public partial struct Shape
    {
        /// <summary>
        ///     Resolves the output shape when broadcasting two shapes together.
        /// </summary>
        /// <param name="leftShape">First shape to broadcast.</param>
        /// <param name="rightShape">Second shape to broadcast.</param>
        /// <returns>The resulting broadcast shape.</returns>
        /// <exception cref="IncorrectShapeException">If shapes cannot be broadcast together.</exception>
        public static Shape ResolveReturnShape(Shape leftShape, Shape rightShape)
        {
            if (leftShape == rightShape)
                return leftShape;

            // A 0-d left shape takes the right one; so does a one-element 1-d left shape, but only against a right shape
            // of at least one axis: (1,) broadcast with () is (1,) — the RANK is the larger one's, so returning the 0-d
            // right shape there dropped the axis (np.add(x1, x2, out=o, where=False) of one-element arrays refused its out).
            if (leftShape.IsScalar || leftShape.NDim == 1 && leftShape[0] == 1 && rightShape.NDim >= 1)
                return rightShape;

            // Reached only with a left shape of at least one axis, so a 0-d or one-element 1-d right shape leaves it as is.

            if (rightShape.IsScalar || rightShape.NDim == 1 && rightShape[0] == 1)
                return leftShape;

            int i, nd, k;
            long tmp;

            nd = Math.Max(leftShape.NDim, rightShape.NDim);

            // This is the shared shape aka the target broadcast
            var mit = new long[nd];

            // Discover the broadcast shape in each dimension
            for (i = 0; i < nd; i++)
            {
                mit[i] = 1;

                // This prepends 1 to shapes not already equal to nd
                k = i + leftShape.NDim - nd;
                if (k >= 0)
                {
                    tmp = leftShape.dimensions[k];
                    if (tmp == 1)
                    {
                        goto _continue;
                    }

                    if (mit[i] == 1)
                    {
                        mit[i] = tmp;
                    }
                    else if (mit[i] != tmp)
                    {
                        throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape");
                    }
                }

                _continue:
                // This prepends 1 to shapes not already equal to nd
                k = i + rightShape.NDim - nd;
                if (k >= 0)
                {
                    tmp = rightShape.dimensions[k];
                    if (tmp == 1)
                    {
                        continue;
                    }

                    if (mit[i] == 1)
                    {
                        mit[i] = tmp;
                    }
                    else if (mit[i] != tmp)
                    {
                        throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape");
                    }
                }
            }

            // We don't need to mark it as IsBroadcasted because technically, it has not broadcasted strides.
            return Shape.Adopt(mit); // mit is built here, so the shape takes it as is
        }

        /// <summary>
        ///     Resolves the output shape when broadcasting multiple shapes together.
        /// </summary>
        /// <param name="shapes">Shapes to broadcast.</param>
        /// <returns>The resulting broadcast shape.</returns>
        /// <exception cref="IncorrectShapeException">If shapes cannot be broadcast together.</exception>
        public static Shape ResolveReturnShape(params Shape[] shapes)
        {
            if (shapes.Length == 0)
                return default;

            if (shapes.Length == 1)
                return shapes[0].Clean();

            Shape mit;
            int i, nd, k;
            long tmp;

            int len = shapes.Length;

            // Discover the broadcast number of dimensions
            // Gets the largest ndim of all iterators
            nd = 0;
            for (i = 0; i < len; i++)
                nd = Math.Max(nd, shapes[i].NDim);

            // This is the shared shape aka the target broadcast
            mit = Shape.Empty(nd);

            // Discover the broadcast shape in each dimension
            for (i = 0; i < nd; i++)
            {
                mit.dimensions[i] = 1;
                for (int targetIndex = 0; targetIndex < len; targetIndex++)
                {
                    Shape target = shapes[targetIndex];
                    // This prepends 1 to shapes not already equal to nd
                    k = i + target.NDim - nd;
                    if (k >= 0)
                    {
                        tmp = target.dimensions[k];
                        if (tmp == 1)
                        {
                            continue;
                        }

                        if (mit.dimensions[i] == 1)
                        {
                            mit.dimensions[i] = tmp;
                        }
                        else if (mit.dimensions[i] != tmp)
                        {
                            throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape");
                        }
                    }
                }
            }

            // We don't need to mark it as IsBroadcasted because technically, it has not broadcasted strides.
            return mit.Clean();
        }

        /// <summary>
        ///     Broadcasts multiple shapes and returns the broadcasted shapes with computed strides.
        /// </summary>
        /// <param name="shapes">Shapes to broadcast.</param>
        /// <returns>Array of broadcasted shapes with stride=0 where dimensions were stretched.</returns>
        /// <exception cref="IncorrectShapeException">If shapes cannot be broadcast together.</exception>
        public static Shape[] Broadcast(params Shape[] shapes)
        {
            if (shapes.Length == 0)
                return Array.Empty<Shape>();

            if (shapes.Length == 1)
                return new Shape[] { shapes[0].Clean() };

            int i, nd, k, j;
            long tmp;

            var ret = new Shape[shapes.Length];
            int len = shapes.Length;

            tmp = 0;
            // Discover the broadcast number of dimensions
            nd = 0;
            for (i = 0; i < len; i++)
                nd = Math.Max(nd, shapes[i].NDim);

            // Use temporary array for broadcast dimensions (not Shape.Empty)
            var mitDims = new long[nd];

            // Discover the broadcast shape in each dimension
            for (i = 0; i < nd; i++)
            {
                mitDims[i] = 1;
                for (int targetIndex = 0; targetIndex < len; targetIndex++)
                {
                    Shape target = shapes[targetIndex];
                    k = i + target.NDim - nd;
                    if (k >= 0)
                    {
                        tmp = target.dimensions[k];
                        if (tmp == 1)
                            continue;

                        if (mitDims[i] == 1)
                            mitDims[i] = tmp;
                        else if (mitDims[i] != tmp)
                            throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape");
                    }
                }
            }

            // Create broadcast shapes for each input
            for (i = 0; i < len; i++)
            {
                Shape ogiter = shapes[i];
                int ogNd = ogiter.NDim;

                // Compute broadcast strides
                var broadcastStrides = new long[nd];
                for (j = 0; j < nd; j++)
                {
                    k = j + ogNd - nd;
                    if ((k < 0) || ogiter.dimensions[k] != mitDims[j])
                        broadcastStrides[j] = 0;
                    else
                        broadcastStrides[j] = ogiter.strides[k];
                }

                // Create immutable shape via constructor
                long bufSize = ogiter.bufferSize > 0 ? ogiter.bufferSize : ogiter.size;
                ret[i] = new Shape(
                    (long[])mitDims.Clone(),
                    broadcastStrides,
                    ogiter.offset,
                    bufSize
                );
            }

            return ret;
        }

        /// <summary>
        ///     Broadcasts two shapes and returns the broadcasted shapes with computed strides.
        /// </summary>
        /// <param name="leftShape">First shape to broadcast.</param>
        /// <param name="rightShape">Second shape to broadcast.</param>
        /// <returns>Tuple of (LeftBroadcastedShape, RightBroadcastedShape).</returns>
        /// <exception cref="IncorrectShapeException">If shapes cannot be broadcast together.</exception>
        public static (Shape LeftShape, Shape RightShape) Broadcast(Shape leftShape, Shape rightShape)
        {
            // Fast path: IDENTICAL shapes broadcast to themselves. The shape hash is only a cheap
            // pre-filter — it COLLIDES across dims that differ but hash alike (notably a 0-length
            // axis vs a size-1 axis: (1,1) and (0,1) both hash to the same value). Confirm the
            // dimensions are actually equal before short-circuiting, else e.g. broadcast_to((1,1),
            // (0,1)) wrongly returned (1,1) instead of stretching the size-1 axis to the 0 axis.
            if (leftShape._hashCode != 0 && leftShape._hashCode == rightShape._hashCode
                && leftShape.NDim == rightShape.NDim)
            {
                bool sameDims = true;
                for (int d = 0; d < leftShape.NDim; d++)
                    if (leftShape.dimensions[d] != rightShape.dimensions[d]) { sameDims = false; break; }
                if (sameDims)
                    return (leftShape, rightShape);
            }

            int i, nd, k, j;
            long tmp;

            // Is left a scalar / size-1 — broadcast to right's shape with zero strides. Guard on
            // rightShape.NDim >= leftShape.NDim: a 1-D [1] broadcast against a 0-D scalar must keep
            // its rank (result [1], not []), so the size-1 collapse only applies when the other
            // operand has at least as many dimensions. NumPy: result ndim == max(ndims).
            if ((leftShape.IsScalar || (leftShape.NDim == 1 && leftShape.size == 1)) && rightShape.NDim >= leftShape.NDim)
            {
                var zeroStrides = new long[rightShape.NDim];
                long leftBufSize = leftShape.bufferSize > 0 ? leftShape.bufferSize : leftShape.size;
                var left = new Shape(
                    (long[])rightShape.dimensions.Clone(),
                    zeroStrides,
                    leftShape.offset,
                    leftBufSize
                );
                return (left, rightShape);
            }

            // Is right a scalar / size-1 — broadcast to left's shape with zero strides. Symmetric
            // guard: leftShape.NDim >= rightShape.NDim so a 0-D right against a 1-D [1] left keeps
            // rank (result [1]).
            if ((rightShape.IsScalar || (rightShape.NDim == 1 && rightShape.size == 1)) && leftShape.NDim >= rightShape.NDim)
            {
                var zeroStrides = new long[leftShape.NDim];
                long rightBufSize = rightShape.bufferSize > 0 ? rightShape.bufferSize : rightShape.size;
                var right = new Shape(
                    (long[])leftShape.dimensions.Clone(),
                    zeroStrides,
                    rightShape.offset,
                    rightBufSize
                );
                return (leftShape, right);
            }

            // General case: compute broadcast shape
            nd = Math.Max(rightShape.NDim, leftShape.NDim);

            // Compute broadcast dimensions into temporary array
            var mitDims = new long[nd];
            for (i = 0; i < nd; i++)
            {
                mitDims[i] = 1;

                k = i + leftShape.NDim - nd;
                if (k >= 0)
                {
                    tmp = leftShape.dimensions[k];
                    if (tmp != 1)
                    {
                        if (mitDims[i] == 1)
                            mitDims[i] = tmp;
                        else if (mitDims[i] != tmp)
                            throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape");
                    }
                }

                k = i + rightShape.NDim - nd;
                if (k >= 0)
                {
                    tmp = rightShape.dimensions[k];
                    if (tmp != 1)
                    {
                        if (mitDims[i] == 1)
                            mitDims[i] = tmp;
                        else if (mitDims[i] != tmp)
                            throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape");
                    }
                }
            }

            // Compute left broadcast strides
            var leftStrides = new long[nd];
            for (j = 0; j < nd; j++)
            {
                k = j + leftShape.NDim - nd;
                if ((k < 0) || leftShape.dimensions[k] != mitDims[j])
                    leftStrides[j] = 0;
                else
                    leftStrides[j] = leftShape.strides[k];
            }

            // Compute right broadcast strides
            var rightStrides = new long[nd];
            for (j = 0; j < nd; j++)
            {
                k = j + rightShape.NDim - nd;
                if ((k < 0) || rightShape.dimensions[k] != mitDims[j])
                    rightStrides[j] = 0;
                else
                    rightStrides[j] = rightShape.strides[k];
            }

            // Create immutable shapes via constructors
            long leftBufSize2 = leftShape.bufferSize > 0 ? leftShape.bufferSize : leftShape.size;
            long rightBufSize2 = rightShape.bufferSize > 0 ? rightShape.bufferSize : rightShape.size;

            var leftResult = new Shape(mitDims, leftStrides, leftShape.offset, leftBufSize2);
            var rightResult = new Shape((long[])mitDims.Clone(), rightStrides, rightShape.offset, rightBufSize2);

            return (leftResult, rightResult);
        }

        /// <summary>
        ///     np.broadcast_to's layout: <paramref name="from"/> stretched to <paramref name="against"/>'s dimensions, after
        ///     np.broadcast_to's check that only the source's length-1 axes stretch. It is the whole answer of the shape forms
        ///     of np.broadcast_to, and the view's layout of <see cref="NDArray.broadcast_to(UnmanagedStorage, Shape)"/>; the
        ///     engine calls it where it validates or lays out a copy's broadcast source (NDIter, the text copies).
        /// </summary>
        /// <param name="from">The source's shape: its dimensions are checked, its strides and offset become the result's.</param>
        /// <param name="against">The target; only its dimensions are read.</param>
        /// <returns><paramref name="against"/>'s dimensions over <paramref name="from"/>'s offset and buffer: the source's
        ///     stride on every axis that keeps its length, 0 on a stretched length-1 axis and on every prepended axis. When the
        ///     dimensions already agree it is <paramref name="from"/> itself (its strides and flags, WRITEABLE included); the
        ///     array form clears WRITEABLE on the result, this one does not.</returns>
        /// <exception cref="IncorrectShapeException"><paramref name="from"/> has more axes than <paramref name="against"/>,
        ///     or an axis of a length other than 1 differs from the target's (<c>operands could not be broadcast together with
        ///     remapped shapes [original->remapped]: (2,3) and requested shape (4,3)</c>), checked before any shape is
        ///     built.</exception>
        internal static Shape BroadcastTo(Shape from, Shape against)
        {
            ValidateBroadcastTo(from, against);
            return Broadcast(from, against).LeftShape;
        }

        /// <summary>
        ///     The check of <see cref="BroadcastTo"/>: <paramref name="from"/> broadcasts to <paramref name="target"/> one way,
        ///     as NumPy's broadcast_to does. Only a length-1 axis of the source stretches; the target never does, so a source
        ///     with more axes, or with an axis longer than the target's, is refused where a two-way broadcast would accept it.
        /// </summary>
        /// <param name="from">The source's shape; only its dimensions are read.</param>
        /// <param name="target">The requested shape; only its dimensions are read.</param>
        /// <exception cref="IncorrectShapeException">The source does not broadcast to the target (the text names both
        ///     shapes' dimensions).</exception>
        private static void ValidateBroadcastTo(Shape from, Shape target)
        {
            if (from.NDim > target.NDim)
                throw new IncorrectShapeException(
                    $"operands could not be broadcast together with remapped shapes " +
                    $"[original->remapped]: ({string.Join(",", from.dimensions)}) and requested shape ({string.Join(",", target.dimensions)})");

            // Right-align: iterate from the rightmost dimension
            for (int i = 0; i < from.NDim; i++)
            {
                long fromDim = from.dimensions[from.NDim - 1 - i];
                long targetDim = target.dimensions[target.NDim - 1 - i];

                if (fromDim != 1 && fromDim != targetDim)
                    throw new IncorrectShapeException(
                        $"operands could not be broadcast together with remapped shapes " +
                        $"[original->remapped]: ({string.Join(",", from.dimensions)}) and requested shape ({string.Join(",", target.dimensions)})");
            }
        }

        /// <summary>
        ///     Checks if the given shapes can be broadcast together.
        /// </summary>
        /// <param name="shapes">Shapes to check.</param>
        /// <returns>True if all shapes are broadcastable, false otherwise.</returns>
        public static unsafe bool AreBroadcastable(params Shape[] shapes)
        {
            if (shapes.Length <= 1)
                return true;

            int i, nd, k;
            long tmp;

            int len = shapes.Length;

            // Discover the broadcast number of dimensions
            // Gets the largest ndim of all iterators
            nd = 0;
            for (i = 0; i < len; i++)
                nd = Math.Max(nd, shapes[i].NDim);

            // This is the shared shape aka the target broadcast
            var mit = stackalloc long[nd];

            // Discover the broadcast shape in each dimension
            for (i = 0; i < nd; i++)
            {
                mit[i] = 1;
                for (int targetIndex = 0; targetIndex < len; targetIndex++)
                {
                    Shape target = shapes[targetIndex];
                    // This prepends 1 to shapes not already equal to nd
                    k = i + target.NDim - nd;
                    if (k >= 0)
                    {
                        tmp = target.dimensions[k];
                        if (tmp == 1)
                        {
                            continue;
                        }

                        if (mit[i] == 1)
                        {
                            mit[i] = tmp;
                        }
                        else if (mit[i] != tmp)
                        {
                            return false;
                        }
                    }
                }
            }

            return true;
        }

        /// <summary>
        ///     Checks if the given dimension arrays can be broadcast together.
        /// </summary>
        /// <param name="shapes">Dimension arrays to check.</param>
        /// <returns>True if all shapes are broadcastable, false otherwise.</returns>
        public static unsafe bool AreBroadcastable(params int[][] shapes)
        {
            if (shapes.Length <= 1)
                return true;

            int i, nd, k, tmp;

            int len = shapes.Length;

            tmp = 0;
            // Discover the broadcast number of dimensions
            // Gets the largest ndim of all iterators
            nd = 0;
            for (i = 0; i < len; i++)
                nd = Math.Max(nd, shapes[i].Length);

            // This is the shared shape aka the target broadcast
            var mit = stackalloc int[nd];

            // Discover the broadcast shape in each dimension
            for (i = 0; i < nd; i++)
            {
                mit[i] = 1;
                for (int targetIndex = 0; targetIndex < len; targetIndex++)
                {
                    int[] target = shapes[targetIndex];
                    // This prepends 1 to shapes not already equal to nd
                    k = i + target.Length - nd;
                    if (k >= 0)
                    {
                        tmp = target[k];
                        if (tmp == 1)
                        {
                            continue;
                        }

                        if (mit[i] == 1)
                        {
                            mit[i] = tmp;
                        }
                        else if (mit[i] != tmp)
                        {
                            return false;
                        }
                    }
                }
            }

            return true;
        }

        /// <summary>
        ///     NumPy's <c>NPY_MAXARGS</c>: the most operands one <c>np.broadcast</c> call takes, hence the size of the first
        ///     group <see cref="BroadcastShape"/> broadcasts (each later group holds one fewer, the shape found so far).
        /// </summary>
        private const int MaxBroadcastArgs = 64;

        /// <summary>
        ///     NumPy's <c>_broadcast_shape(*args)</c> (<c>lib/_stride_tricks_impl.py</c>): the shape <c>np.broadcast</c>'s
        ///     multi-iterator gives the operands, with NumPy's mismatch text naming the two operands that disagree. Use it
        ///     where NumPy broadcasts arguments through <c>np.broadcast_arrays</c> or <c>np.broadcast_shapes</c> (np.select's two
        ///     groups) and the error text is part of the contract; <see cref="ResolveReturnShape(Shape[])"/> computes the same
        ///     shape with a generic text.
        /// </summary>
        /// <param name="shapes">The operand shapes in argument order; only their dimensions are read.</param>
        /// <returns>A C-ordered shape of the broadcast dimensions; the scalar shape when there is no operand or every operand is
        ///     0-d.</returns>
        /// <exception cref="IncorrectShapeException">Two operands disagree on an axis: NumPy's ValueError text, verbatim
        ///     (<c>shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (7,) and
        ///     arg 1 with shape (3, 4).</c>), under the house exception type of a broadcast failure.</exception>
        /// <exception cref="ValueError">The broadcast shape's element count does not fit a 64-bit integer
        ///     (<c>broadcast dimensions too large.</c>).</exception>
        /// <remarks>
        ///     <para>
        ///     The scan is <c>PyArray_Broadcast</c>'s (<c>iterators.c</c>): axis by axis from the first, the operands in order
        ///     within each axis, so a mismatch names the operand that first set the axis's length and the first one that
        ///     disagrees with it, on the LEFTMOST axis that has a conflict. A scan that walks operand by operand instead
        ///     reports another pair when two axes conflict: <c>np.broadcast_shapes((1,4), (1,5), (2,1), (3,1))</c> names args 2
        ///     and 3 in NumPy, not 0 and 1.
        ///     </para>
        ///     <para>
        ///     <c>np.broadcast</c> takes at most 64 operands, so NumPy broadcasts the first 64, then each further group of 63
        ///     together with the shape found so far (<c>broadcast_to(0, b.shape)</c>), which is that call's argument 0: past 64
        ///     operands a mismatch numbers the operands within their group, as NumPy's text does
        ///     (<c>Mismatch is between arg 0 with shape (4,) and arg 3 with shape (5,).</c> for the 67th operand).
        ///     </para>
        /// </remarks>
        internal static Shape BroadcastShape(params Shape[] shapes)
        {
            // b = np.broadcast(*args[:64])
            int count = shapes?.Length ?? 0;
            long[] dims = MultiIterDims(null, shapes, 0, Math.Min(count, MaxBroadcastArgs));
            // for pos in range(64, len(args), 63): b = broadcast_to(0, b.shape); b = np.broadcast(b, *args[pos:pos + 63])
            for (int pos = MaxBroadcastArgs; pos < count; pos += MaxBroadcastArgs - 1)
                dims = MultiIterDims(dims, shapes, pos, Math.Min(count, pos + MaxBroadcastArgs - 1));
            return Shape.Adopt(dims);
        }

        /// <summary>
        ///     One <c>np.broadcast</c> call of <see cref="BroadcastShape"/>: <c>PyArray_Broadcast</c>'s scan over an optional
        ///     leading operand (the dimensions the previous group broadcast to, this call's argument 0) followed by
        ///     <c>shapes[from..to)</c>, then its overflow check of the element count.
        /// </summary>
        /// <param name="lead">The previous group's broadcast dimensions, or null for the first group.</param>
        /// <param name="shapes">All the operand shapes (null only when the group is empty).</param>
        /// <param name="from">The first operand of this group.</param>
        /// <param name="to">One past the last operand of this group.</param>
        /// <returns>This call's broadcast dimensions (a new array).</returns>
        /// <exception cref="IncorrectShapeException">See <see cref="BroadcastShape"/>.</exception>
        /// <exception cref="ValueError">See <see cref="BroadcastShape"/>.</exception>
        private static long[] MultiIterDims(long[] lead, Shape[] shapes, int from, int to)
        {
            // The argument numbers of this call: the lead (when there is one) is 0, shapes[from] the next.
            int first = lead is null ? 0 : 1;
            int count = first + (to - from);

            // Discover the broadcast number of dimensions.
            int nd = lead?.Length ?? 0;
            for (int j = from; j < to; j++)
                nd = Math.Max(nd, shapes[j].NDim);

            var result = new long[nd];
            // PyArray_Broadcast's src_iter: the operand that set the current axis's length (every axis starts at 1, so a
            // conflict on an axis always follows the operand that set that axis).
            int src = -1;
            for (int i = 0; i < nd; i++)
            {
                result[i] = 1;
                for (int j = 0; j < count; j++)
                {
                    long[] d = j < first ? lead : shapes[from + j - first].dimensions;
                    // This prepends 1 to shapes shorter than nd (right alignment).
                    int k = i + d.Length - nd;
                    if (k < 0)
                        continue;
                    long tmp = d[k];
                    if (tmp == 1)
                        continue;
                    if (result[i] == 1)
                    {
                        result[i] = tmp;
                        src = j;
                    }
                    else if (result[i] != tmp)
                    {
                        // set_shape_mismatch_exception: both operands' WHOLE shapes, as Python tuple reprs.
                        long[] s = src < first ? lead : shapes[from + src - first].dimensions;
                        throw new IncorrectShapeException(
                            "shape mismatch: objects cannot be broadcast to a single shape.  " +
                            $"Mismatch is between arg {src} with shape {PythonTupleRepr(s)} and arg {j} with shape {PythonTupleRepr(d)}.");
                    }
                }
            }

            // PyArray_OverflowMultiplyList over the broadcast dimensions, left to right: a zero length ends the product (no
            // overflow past it), a product past the 64-bit range is the error.
            long product = 1;
            for (int i = 0; i < nd; i++)
            {
                if (result[i] == 0)
                    break;
                if (product > long.MaxValue / result[i])
                    throw new ValueError("broadcast dimensions too large.");
                product *= result[i];
            }

            return result;
        }

        /// <summary>A shape tuple as Python's <c>repr</c> writes it: <c>()</c>, <c>(4,)</c>, <c>(2, 3)</c> (a space after each
        ///     comma, unlike <see cref="PythonTuple"/>).</summary>
        /// <param name="dims">The dimensions (not null).</param>
        /// <returns>The tuple text.</returns>
        private static string PythonTupleRepr(long[] dims)
        {
            if (dims.Length == 0)
                return "()";
            if (dims.Length == 1)
                return "(" + dims[0] + ",)";
            var sb = new System.Text.StringBuilder("(");
            for (int i = 0; i < dims.Length; i++)
            {
                if (i > 0)
                    sb.Append(", ");
                sb.Append(dims[i]);
            }
            return sb.Append(')').ToString();
        }
    }
}
