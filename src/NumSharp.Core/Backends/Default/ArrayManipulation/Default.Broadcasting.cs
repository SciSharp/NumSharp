using System;
using System.Linq;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <remarks>Based on https://docs.scipy.org/doc/numpy-1.16.1/user/basics.broadcasting.html </remarks>
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

        /// <remarks>Based on https://docs.scipy.org/doc/numpy-1.16.1/user/basics.broadcasting.html </remarks>
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

        /// <remarks>Based on https://docs.scipy.org/doc/numpy-1.16.1/user/basics.broadcasting.html </remarks>
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

        /// <remarks>Based on https://docs.scipy.org/doc/numpy-1.16.1/user/basics.broadcasting.html </remarks>
        public static Shape[] Broadcast(params Shape[] shapes)
        {
            if (shapes.Length == 0)
                return Array.Empty<Shape>();

            if (shapes.Length == 1)
                return new Shape[] {shapes[0].Clean()};

            Shape mit;
            int i, nd, k, j, tmp;

            var ret = new Shape[shapes.Length];
            int len = shapes.Length;

            tmp = 0;
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

            for (i = 0; i < len; i++)
            {
                Shape ogiter = shapes[i];
                // When re-broadcasting, track the root original shape
                Shape ogOriginal = ogiter.IsBroadcasted ? ogiter.BroadcastInfo.OriginalShape : ogiter;
                ret[i] = mit.Clean();
                ref Shape it = ref ret[i];
                nd = ogiter.NDim;
                it.BroadcastInfo = new BroadcastInfo(ogOriginal);
                // Set ViewInfo for sliced inputs so GetOffset resolves slice strides correctly
                if (ogOriginal.IsSliced)
                    it.ViewInfo = new ViewInfo() {ParentShape = ogOriginal, Slices = null};
                for (j = 0; j < mit.NDim; j++)
                {
                    //it->dims_m1[j] = mit.dimensions[j] - 1;
                    k = j + nd - mit.NDim;
                    /*
                     * If this dimension was added or shape of
                     * underlying array was 1
                     */
                    if ((k < 0) ||
                        ogiter.dimensions[k] != mit.dimensions[j])
                    {
                        it.strides[j] = 0;
                    }
                    else
                    {
                        it.strides[j] = ogiter.strides[k];
                    }

                    //it.backstrides[j] = it.strides[j] * (it.dimensions[j] - 1);
                    //if (j > 0)
                    //    it.factors[mit.NDim - j - 1] = it.factors[mit.NDim - j] * mit.dimensions[mit.NDim - j];
                }

                it.ComputeHashcode();
            }

            return ret;
        }

        //private static readonly int[][] _zeros = new int[][] {new int[0], new int[] {0}, new int[] {0, 0}, new int[] {0, 0, 0}, new int[] {0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},};

        /// <remarks>Based on https://docs.scipy.org/doc/numpy-1.16.1/user/basics.broadcasting.html </remarks>
        public static (Shape LeftShape, Shape RightShape) Broadcast(Shape leftShape, Shape rightShape)
        {
            if (leftShape._hashCode != 0 && leftShape._hashCode == rightShape._hashCode)
                return (leftShape, rightShape);

            // When re-broadcasting an already-broadcast shape, resolve to the original
            // pre-broadcast shape for BroadcastInfo tracking. The broadcast shape's dimensions
            // and strides are used as-is for the new broadcast computation — stride=0 dims
            // from the prior broadcast naturally propagate.
            Shape leftOriginal = leftShape.IsBroadcasted ? leftShape.BroadcastInfo.OriginalShape : leftShape;
            Shape rightOriginal = rightShape.IsBroadcasted ? rightShape.BroadcastInfo.OriginalShape : rightShape;

            Shape left, right, mit;
            int i, nd, k, j, tmp;

            //is left a scalar
            if (leftShape.IsScalar || leftShape.NDim == 1 && leftShape.size == 1)
            {
                left = rightShape; //copy right
                left.strides = new int[left.strides.Length]; //zero strides
                left.BroadcastInfo = new BroadcastInfo(leftOriginal);
                return (left, rightShape);
            }
            //is right a scalar
            else if (rightShape.IsScalar || rightShape.NDim == 1 && rightShape.size == 1)
            {
                right = leftShape; //copy left
                right.strides = new int[right.strides.Length]; //zero strides
                right.BroadcastInfo = new BroadcastInfo(rightOriginal);
                return (leftShape, right);
            }
            else
            {
                tmp = 0;
                /* Discover the broadcast number of dimensions */
                //Gets the largest ndim of all iterators
                nd = Math.Max(rightShape.NDim, leftShape.NDim);

                //this is the shared shape aka the target broadcast
                mit = Shape.Empty(nd);

                /* Discover the broadcast shape in each dimension */
                for (i = 0; i < nd; i++)
                {
                    mit.dimensions[i] = 1;

                    /* This prepends 1 to shapes not already equal to nd */
                    k = i + leftShape.NDim - nd;
                    if (k >= 0)
                    {
                        tmp = leftShape.dimensions[k];
                        if (tmp == 1)
                        {
                            goto _continue;
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


                left = new Shape(mit.dimensions) {BroadcastInfo = new BroadcastInfo(leftOriginal)};
                right = new Shape(mit.dimensions) {BroadcastInfo = new BroadcastInfo(rightOriginal)};
                if (leftOriginal.IsSliced)
                    left.ViewInfo = new ViewInfo() {ParentShape = leftOriginal, Slices = null};
                if (rightOriginal.IsSliced)
                    right.ViewInfo = new ViewInfo() {ParentShape = rightOriginal, Slices = null};

                //left.ViewInfo = leftShape.ViewInfo;
                //right.ViewInfo = rightShape.ViewInfo;
            }

            //if (nd != 0)
            //{
            //    it->factors[mit.nd - 1] = 1;
            //}
            for (j = 0; j < mit.NDim; j++)
            {
                //it->dims_m1[j] = mit.dimensions[j] - 1;
                k = j + leftShape.NDim - mit.NDim;
                /*
                 * If this dimension was added or shape of
                 * underlying array was 1
                 */
                if ((k < 0) ||
                    leftShape.dimensions[k] != mit.dimensions[j])
                {
                    left.strides[j] = 0;
                }
                else
                {
                    left.strides[j] = leftShape.strides[k];
                }

                //it.backstrides[j] = it.strides[j] * (it.dimensions[j] - 1);
                //if (j > 0)
                //    it.factors[mit.NDim - j - 1] = it.factors[mit.NDim - j] * mit.dimensions[mit.NDim - j];
            }

            //if (nd != 0)
            //{
            //    it->factors[mit.nd - 1] = 1;
            //}
            for (j = 0; j < mit.NDim; j++)
            {
                //it->dims_m1[j] = mit.dimensions[j] - 1;
                k = j + rightShape.NDim - mit.NDim;
                /*
                 * If this dimension was added or shape of
                 * underlying array was 1
                 */
                if ((k < 0) ||
                    rightShape.dimensions[k] != mit.dimensions[j])
                {
                    right.strides[j] = 0;
                }
                else
                {
                    right.strides[j] = rightShape.strides[k];
                }

                //it.backstrides[j] = it.strides[j] * (it.dimensions[j] - 1);
                //if (j > 0)
                //    it.factors[mit.NDim - j - 1] = it.factors[mit.NDim - j] * mit.dimensions[mit.NDim - j];
            }

            left.ComputeHashcode();
            right.ComputeHashcode();

            return (left, right);
        }

        /// <remarks>Based on https://docs.scipy.org/doc/numpy-1.16.1/user/basics.broadcasting.html </remarks>
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

        /// <remarks>Based on https://docs.scipy.org/doc/numpy-1.16.1/user/basics.broadcasting.html </remarks>
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

        /// <remarks>Based on https://docs.scipy.org/doc/numpy-1.16.1/user/basics.broadcasting.html </remarks>
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

        /// <remarks>Based on https://docs.scipy.org/doc/numpy-1.16.1/user/basics.broadcasting.html </remarks>
        public static bool AreBroadcastable(params NDArray[] arrays)
        {
            if (arrays.Length <= 1)
                return true;

            return AreBroadcastable(arrays.Select(r => r.shape).ToArray());
        }
    }
}
