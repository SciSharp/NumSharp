using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Stack arrays in sequence vertically (row wise).<br></br>
        ///     This is equivalent to concatenation along the first axis after 1-D arrays of shape(N,) have been reshaped to(1, N). Rebuilds arrays divided by vsplit.
        /// </summary>
        /// <typeparam name="T">The type dtype to return.</typeparam>
        /// <param name="tup">The arrays must have the same shape along all but the first axis. 1-D arrays must have the same length.</param>
        /// <returns>The array formed by stacking the given arrays, will be at least 2-D.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html</remarks>
        public static NDArray vstack<T>(params NDArray[] tup) where T : unmanaged
        {
            if (tup == null || tup.Length == 0)
                throw new Exception("Input arrays can not be empty");

            NDArray ret = new NDArray(typeof(T), Shape.Vector(tup.Sum(n => n.size)));
            Shape commonShape = tup[0].shape;
            unsafe
            {
                var retaddr = (byte*)ret.Address;
                int accoffset = 0;
                foreach (NDArray curr in tup)
                {
                    if (commonShape != curr.Shape)
                        throw new Exception("Arrays mush have same shapes");
                    var data = curr.Storage.GetData<T>();
                    var len = data.Count * data.ItemLength;
                    Buffer.MemoryCopy(data.Address, (retaddr + accoffset), len, len);
                    accoffset += len;
                }
            }

            if (commonShape.NDim == 1) //all dims are equal across all ndarrays
            {
                ret.Storage.Reshape(Shape.Matrix(tup.Length, commonShape[0]));
            }
            else
            {
                int[] shapes = commonShape.dimensions;
                shapes[0] *= tup.Length;
                ret.Storage.Reshape(new Shape(shapes));
            }

            return ret;
        }

        /// <summary>
        ///     Stack arrays in sequence vertically (row wise).<br></br>
        ///     This is equivalent to concatenation along the first axis after 1-D arrays of shape(N,) have been reshaped to(1, N). Rebuilds arrays divided by vsplit.
        /// </summary>
        /// <typeparam name="T">The type dtype to return.</typeparam>
        /// <param name="tup">The arrays must have the same shape along all but the first axis. 1-D arrays must have the same length.</param>
        /// <returns>The array formed by stacking the given arrays, will be at least 2-D.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html</remarks>
        public static NDArray vstack(params NDArray[] tup)
        {
            var common_type = np._FindCommonType(tup);
            if (tup == null || tup.Length == 0)
                throw new Exception("Input arrays can not be empty");

            NDArray ret = new NDArray(common_type, Shape.Vector(tup.Sum(n => n.size)));
            Shape commonShape = tup[0].shape;
            unsafe
            {
                var retaddr = (byte*)ret.Address;
                int accBytesOffset = 0;
                foreach (NDArray curr in tup)
                {
                    if (commonShape != curr.Shape)
                        throw new Exception("Arrays mush have same shapes");
                    var data = curr.Storage.InternalArray;
                    var bytesLen = data.Count * data.ItemLength;
                    if (data.TypeCode != common_type)
                        data.CastTo(ret.Array, bytesOffset: accBytesOffset);
                    else
                        Buffer.MemoryCopy(data.Address, (retaddr + accBytesOffset), bytesLen, bytesLen);
                    accBytesOffset += bytesLen;
                }
            }

            if (commonShape.NDim == 1) //all dims are equal across all ndarrays
            {
                ret.Storage.Reshape(Shape.Matrix(tup.Length, commonShape[0]));
            }
            else
            {
                int[] shapes = commonShape.dimensions;
                shapes[0] *= tup.Length;
                ret.Storage.Reshape(new Shape(shapes));
            }

            return ret;
        }

    }
}
