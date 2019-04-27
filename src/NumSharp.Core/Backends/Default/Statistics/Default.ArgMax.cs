using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Extensions;

namespace NumSharp.Backends
{
    public abstract partial class DefaultEngine
    {
        public NDArray ArgMax(NDArray nd, int axis = -1)
        {
            if (axis == -1)
                return nd.argmax();
            else
                return ArgMaxByAxis(nd, axis: axis);
        }

        private NDArray ArgMaxByAxis(NDArray nd, int axis)
        {
            var shape = Shape.GetShape(nd.shape, axis: axis);
            var nd2 = new NDArray(np.int32, shape);

            switch (shape.Length)
            {
                case 1: // 2 dimension
                    switch (Type.GetTypeCode(nd.dtype))
                    {
                        case TypeCode.Int32:
                            nd2.Array = ArgMax2dInt32(nd, shape, axis);
                            break;
                        case TypeCode.Single:
                            nd2.Array = ArgMax2dSingle(nd, shape, axis);
                            break;
                        default:
                            throw new NotImplementedException($"ArgMaxByAxis for {nd.dtype.Name}");
                    }

                    break;
            }

            return nd2;
        }

        private int[] ArgMax2dInt32(NDArray nd, int[] shape, int axis)
        {
            int i = 0;
            var size = Shape.GetSize(shape);
            var data = new int[size];
            for (int d0 = 0; d0 < shape[0]; d0++)
            {
                int arg = 0;
                int max = 0;
                int next = 0;

                for (int d1 = 0; d1 < nd.shape[axis]; d1++)
                {
                    next = axis == 0 ? nd.GetInt32(d1, d0) : nd.GetInt32(d0, d1);
                    if (next > max)
                    {
                        arg = d1;
                        max = next;
                    }
                }

                data[i++] = arg;
            }

            return data;
        }

        private int[] ArgMax2dSingle(NDArray nd, int[] shape, int axis)
        {
            int i = 0;
            var size = Shape.GetSize(shape);
            var data = new int[size];
            for (int d0 = 0; d0 < shape[0]; d0++)
            {
                int arg = 0;
                float max = 0;
                float next = 0;

                for (int d1 = 0; d1 < nd.shape[axis]; d1++)
                {
                    next = axis == 0 ? nd.GetSingle(d1, d0) : nd.GetSingle(d0, d1);
                    if (next > max)
                    {
                        arg = d1;
                        max = next;
                    }
                }

                data[i++] = arg;
            }

            return data;
        }
    }

}
