﻿using System;

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
                    tmp = leftShape.Dimensions[k];
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
                        throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape"); //TODO mismatch
                    }
                }

                _continue:
                /* This prepends 1 to shapes not already equal to nd */
                k = i + rightShape.NDim - nd;
                if (k >= 0)
                {
                    tmp = rightShape.Dimensions[k];
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
                        throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape"); //TODO mismatch
                    }
                }
            }

            return mit; //implicit cast
        }

        /// <remarks>Based on https://docs.scipy.org/doc/numpy-1.16.1/user/basics.broadcasting.html </remarks>
        public static Shape[] Broadcast(params Shape[] shapes)
        {
            if (shapes.Length == 0)
                return Array.Empty<Shape>();

            if (shapes.Length == 1)
                return new Shape[] {new Shape(shapes[0].dimensions),};

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
                mit.Dimensions[i] = 1;
                for (int targetIndex = 0; targetIndex < len; targetIndex++)
                {
                    Shape target = shapes[targetIndex];
                    /* This prepends 1 to shapes not already equal to nd */
                    k = i + target.NDim - nd;
                    if (k >= 0)
                    {
                        tmp = target.Dimensions[k];
                        if (tmp == 1)
                        {
                            continue;
                        }

                        if (mit.Dimensions[i] == 1)
                        {
                            mit.Dimensions[i] = tmp;
                        }
                        else if (mit.Dimensions[i] != tmp)
                        {
                            throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape"); //TODO mismatch
                        }
                    }
                }
            }

            for (i = 0; i < len; i++)
            {
                Shape ogiter = shapes[i];
                ret[i] = new Shape(mit);
                ref Shape it = ref ret[i];
                nd = ogiter.NDim;
                it.size = tmp;
                //if (nd != 0)
                //{
                //    it->factors[mit.nd - 1] = 1;
                //}
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
                        it.layout = 'C';
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
            }

            return ret;
        }

        /// <remarks>Based on https://docs.scipy.org/doc/numpy-1.16.1/user/basics.broadcasting.html </remarks>
        public static (Shape LeftShape, Shape RightShape) Broadcast(Shape leftShape, Shape rightShape)
        {
            if (leftShape == rightShape)
                return (leftShape, rightShape);

            Shape left, right, mit, it;
            int i, nd, k, j, tmp;

            //is left a scalar
            if (leftShape.IsScalar || leftShape.NDim == 1 && leftShape.size == 1)
            {
                left = rightShape; //copy right
                left.strides = new int[left.strides.Length]; //zero strides
                return (left, rightShape); //TODO! cache up to 16 dims zeros[ndim] returning an array filled with zeros.
            }
            //is right a scalar
            else if (rightShape.IsScalar || rightShape.NDim == 1 && rightShape.size == 1)
            {
                right = leftShape; //copy left
                right.strides = new int[right.strides.Length]; //zero strides
                return (leftShape, right); //TODO! cache up to 16 dims zeros[ndim] returning an array filled with zeros.
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
                    mit.Dimensions[i] = 1;

                    /* This prepends 1 to shapes not already equal to nd */
                    k = i + leftShape.NDim - nd;
                    if (k >= 0)
                    {
                        tmp = leftShape.Dimensions[k];
                        if (tmp == 1)
                        {
                            goto _continue;
                        }

                        if (mit.Dimensions[i] == 1)
                        {
                            mit.Dimensions[i] = tmp;
                        }
                        else if (mit.Dimensions[i] != tmp)
                        {
                            throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape"); //TODO mismatch
                        }
                    }

                    _continue:
/* This prepends 1 to shapes not already equal to nd */
                    k = i + rightShape.NDim - nd;
                    if (k >= 0)
                    {
                        tmp = rightShape.Dimensions[k];
                        if (tmp == 1)
                        {
                            continue;
                        }

                        if (mit.Dimensions[i] == 1)
                        {
                            mit.Dimensions[i] = tmp;
                        }
                        else if (mit.Dimensions[i] != tmp)
                        {
                            throw new IncorrectShapeException("shape mismatch: objects cannot be broadcast to a single shape"); //TODO mismatch
                        }
                    }
                }


                left = new Shape(mit.dimensions);
                right = new Shape(mit.dimensions);
            }

            i = 1;
            Shape ogiter = leftShape;
            it = left;
            nd = ogiter.NDim;
            it.size = tmp;
            //if (nd != 0)
            //{
            //    it->factors[mit.nd - 1] = 1;
            //}
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
                    it.layout = 'C';
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

            ogiter = rightShape;
            it = right;
            nd = ogiter.NDim;
            it.size = tmp;
            //if (nd != 0)
            //{
            //    it->factors[mit.nd - 1] = 1;
            //}
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
                    it.layout = 'C';
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

            return (left, right);
        }
    }
}