using System;
using System.Linq;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <remarks>Based on https://numpy.org/doc/stable/user/basics.broadcasting.html </remarks>
        public static Shape ResolveReturnShape(Shape leftShape, Shape rightShape)
        {
            if (leftShape == rightShape)
                return leftShape;

            if (leftShape.IsScalar || leftShape.NDim == 1 && leftShape[0] == 1)
                return rightShape;

            if (rightShape.IsScalar || rightShape.NDim == 1 && rightShape[0] == 1)
                return leftShape;

            //PyArrayMultiIterObject *mit
            int i, nd, k;
            int tmp;

            nd = Math.Max(leftShape.NDim, rightShape.NDim);

            //this is the shared shape aka the target broadcast
            var mit = new int[nd];

            /* Discover the broadcast shape in each dimension */
            for (i = 0; i < nd; i++)
            {
                mit[i] = 1;

                /* This prepends 1 to shapes not already equal to nd */
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
                /* This prepends 1 to shapes not already equal to nd */
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

            //we dont need to mark it as IsBroadcasted because techincally, it has not broadcasted strides.
            return mit; //implicit cast
        }

        /// <remarks>Based on https://numpy.org/doc/stable/user/basics.broadcasting.html </remarks>
        public static Shape ResolveReturnShape(params Shape[] shapes)
        {
            if (shapes.Length == 0)
                return default;

            if (shapes.Length == 1)
                return shapes[0].Clean();

            Shape mit;
            int i, nd, k, tmp;

            int len = shapes.Length;

            /* Discover the broadcast number of dimensions */
            //Gets the largest ndim of all iterators
            nd = 0;
            for (i = 0; i < len; i++)
                nd = Math.Max(nd, shapes[i].NDim);

            //this is the shared shape aka the target broadcast
            mit = Shape.Empty(nd);

            /* Discover the broadcast shape in each dimension */
            for (i = 0; i < nd; i++)
            {
                mit.dimensions[i] = 1;
                for (int targetIndex = 0; targetIndex < len; targetIndex++)
                {
                    Shape target = shapes[targetIndex];
                    /* This prepends 1 to shapes not already equal to nd */
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

            //we dont need to mark it as IsBroadcasted because techincally, it has not broadcasted strides.
            return mit.Clean();
        }

        /// <remarks>Based on https://numpy.org/doc/stable/user/basics.broadcasting.html </remarks>
        public static Shape ResolveReturnShape(params NDArray[] shapes)
        {
            if (shapes.Length == 0)
                return default;

            if (shapes.Length == 1)
                return shapes[0].Shape.Clean();

            Shape mit;
            int i, nd, k, tmp;

            int len = shapes.Length;

            /* Discover the broadcast number of dimensions */
            //Gets the largest ndim of all iterators
            nd = 0;
            for (i = 0; i < len; i++)
                nd = Math.Max(nd, shapes[i].ndim);


            //this is the shared shape aka the target broadcast
            mit = Shape.Empty(nd);

            /* Discover the broadcast shape in each dimension */
            for (i = 0; i < nd; i++)
            {
                mit.dimensions[i] = 1;
                for (int targetIndex = 0; targetIndex < len; targetIndex++)
                {
                    Shape target = shapes[targetIndex].Shape;
                    /* This prepends 1 to shapes not already equal to nd */
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

            //we dont need to mark it as IsBroadcasted because techincally, it has not broadcasted strides.
            return mit.Clean();
        }

        /// <remarks>Based on https://numpy.org/doc/stable/user/basics.broadcasting.html </remarks>
        public static Shape[] Broadcast(params Shape[] shapes)
        {
            if (shapes.Length == 0)
                return Array.Empty<Shape>();

            if (shapes.Length == 1)
                return new Shape[] {shapes[0].Clean()};

            int i, nd, k, j, tmp;

            var ret = new Shape[shapes.Length];
            int len = shapes.Length;

            tmp = 0;
            // Discover the broadcast number of dimensions
            nd = 0;
            for (i = 0; i < len; i++)
                nd = Math.Max(nd, shapes[i].NDim);

            // Use temporary array for broadcast dimensions (not Shape.Empty)
            var mitDims = new int[nd];

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
                var broadcastStrides = new int[nd];
                for (j = 0; j < nd; j++)
                {
                    k = j + ogNd - nd;
                    if ((k < 0) || ogiter.dimensions[k] != mitDims[j])
                        broadcastStrides[j] = 0;
                    else
                        broadcastStrides[j] = ogiter.strides[k];
                }

                // Create immutable shape via constructor
                int bufSize = ogiter.bufferSize > 0 ? ogiter.bufferSize : ogiter.size;
                ret[i] = new Shape(
                    (int[])mitDims.Clone(),
                    broadcastStrides,
                    ogiter.offset,
                    bufSize
                );
            }

            return ret;
        }

        //private static readonly int[][] _zeros = new int[][] {new int[0], new int[] {0}, new int[] {0, 0}, new int[] {0, 0, 0}, new int[] {0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},};

        /// <remarks>Based on https://numpy.org/doc/stable/user/basics.broadcasting.html </remarks>
        public static (Shape LeftShape, Shape RightShape) Broadcast(Shape leftShape, Shape rightShape)
        {
            if (leftShape._hashCode != 0 && leftShape._hashCode == rightShape._hashCode)
                return (leftShape, rightShape);

            int i, nd, k, j, tmp;

            // Is left a scalar - broadcast to right's shape with zero strides
            if (leftShape.IsScalar || leftShape.NDim == 1 && leftShape.size == 1)
            {
                var zeroStrides = new int[rightShape.NDim];
                int leftBufSize = leftShape.bufferSize > 0 ? leftShape.bufferSize : leftShape.size;
                var left = new Shape(
                    (int[])rightShape.dimensions.Clone(),
                    zeroStrides,
                    leftShape.offset,
                    leftBufSize
                );
                return (left, rightShape);
            }

            // Is right a scalar - broadcast to left's shape with zero strides
            if (rightShape.IsScalar || rightShape.NDim == 1 && rightShape.size == 1)
            {
                var zeroStrides = new int[leftShape.NDim];
                int rightBufSize = rightShape.bufferSize > 0 ? rightShape.bufferSize : rightShape.size;
                var right = new Shape(
                    (int[])leftShape.dimensions.Clone(),
                    zeroStrides,
                    rightShape.offset,
                    rightBufSize
                );
                return (leftShape, right);
            }

            // General case: compute broadcast shape
            tmp = 0;
            nd = Math.Max(rightShape.NDim, leftShape.NDim);

            // Compute broadcast dimensions into temporary array
            var mitDims = new int[nd];
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
            var leftStrides = new int[nd];
            for (j = 0; j < nd; j++)
            {
                k = j + leftShape.NDim - nd;
                if ((k < 0) || leftShape.dimensions[k] != mitDims[j])
                    leftStrides[j] = 0;
                else
                    leftStrides[j] = leftShape.strides[k];
            }

            // Compute right broadcast strides
            var rightStrides = new int[nd];
            for (j = 0; j < nd; j++)
            {
                k = j + rightShape.NDim - nd;
                if ((k < 0) || rightShape.dimensions[k] != mitDims[j])
                    rightStrides[j] = 0;
                else
                    rightStrides[j] = rightShape.strides[k];
            }

            // Create immutable shapes via constructors
            int leftBufSize2 = leftShape.bufferSize > 0 ? leftShape.bufferSize : leftShape.size;
            int rightBufSize2 = rightShape.bufferSize > 0 ? rightShape.bufferSize : rightShape.size;

            var leftResult = new Shape(mitDims, leftStrides, leftShape.offset, leftBufSize2);
            var rightResult = new Shape((int[])mitDims.Clone(), rightStrides, rightShape.offset, rightBufSize2);

            return (leftResult, rightResult);
        }

        /// <remarks>Based on https://numpy.org/doc/stable/user/basics.broadcasting.html </remarks>
        public static NDArray[] Broadcast(params NDArray[] arrays)
        {
            if (arrays.Length == 0)
                return Array.Empty<NDArray>();

            if (arrays.Length == 1)
                return arrays;

            var shapes = Broadcast(arrays.Select(r => r.Shape).ToArray());

            for (int i = 0; i < arrays.Length; i++)
                arrays[i] = new NDArray(arrays[i].Storage, shapes[i]);

            return arrays;
        }

        /// <remarks>Based on https://numpy.org/doc/stable/user/basics.broadcasting.html </remarks>
        public unsafe static bool AreBroadcastable(params Shape[] shapes)
        {
            if (shapes.Length <= 1)
                return true;

            int i, nd, k, j, tmp;

            int len = shapes.Length;

            tmp = 0;
            /* Discover the broadcast number of dimensions */
            //Gets the largest ndim of all iterators
            nd = 0;
            for (i = 0; i < len; i++)
                nd = Math.Max(nd, shapes[i].NDim);

            //this is the shared shape aka the target broadcast
            var mit = stackalloc int[nd];

            /* Discover the broadcast shape in each dimension */
            for (i = 0; i < nd; i++)
            {
                mit[i] = 1;
                for (int targetIndex = 0; targetIndex < len; targetIndex++)
                {
                    Shape target = shapes[targetIndex];
                    /* This prepends 1 to shapes not already equal to nd */
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

        /// <remarks>Based on https://numpy.org/doc/stable/user/basics.broadcasting.html </remarks>
        public unsafe static bool AreBroadcastable(params int[][] shapes)
        {
            if (shapes.Length <= 1)
                return true;

            int i, nd, k, j, tmp;

            int len = shapes.Length;

            tmp = 0;
            /* Discover the broadcast number of dimensions */
            //Gets the largest ndim of all iterators
            nd = 0;
            for (i = 0; i < len; i++)
                nd = Math.Max(nd, shapes[i].Length);

            //this is the shared shape aka the target broadcast
            var mit = stackalloc int[nd];

            /* Discover the broadcast shape in each dimension */
            for (i = 0; i < nd; i++)
            {
                mit[i] = 1;
                for (int targetIndex = 0; targetIndex < len; targetIndex++)
                {
                    int[] target = shapes[targetIndex];
                    /* This prepends 1 to shapes not already equal to nd */
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

        /// <remarks>Based on https://numpy.org/doc/stable/user/basics.broadcasting.html </remarks>
        public static bool AreBroadcastable(params NDArray[] arrays)
        {
            if (arrays.Length <= 1)
                return true;

            return AreBroadcastable(arrays.Select(r => r.shape).ToArray());
        }
    }
}
