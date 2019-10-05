using System;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test whether all array elements evaluate to True.
        /// </summary>
        /// <param name="nd"></param>
        /// <returns></returns>
        public override bool All(NDArray nd)
        {
            if (nd.GetTypeCode != NPTypeCode.Boolean)
            {
                throw new NotSupportedException("DefaultEngine.All supports only boolean dtype."); //TODO!
            }
            unsafe
            {
                var addr = (bool*)nd.Address;
                var shape = nd.Shape;
                var incr = new NDCoordinatesIncrementor(shape.dimensions);
                do
                {
                    if (!(*(addr + shape.GetOffset(incr.Index))))
                        return false;

                } while (incr.Next() != null);

                return true;
            }
        }

        /// <summary>
        /// Test whether all array elements along a given axis evaluate to True.
        /// </summary>
        /// <param name="nd"></param>
        /// <param name="axis"></param>
        /// <returns>Returns an array of bools</returns>
        public override NDArray<bool> All(NDArray nd, int axis)
        {
            throw new NotImplementedException($"np.all axis {axis}"); //TODO! 
        }
    }
}
