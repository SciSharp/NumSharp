using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// NumPy-like where: If only condition is given, return tuple of indices where condition is True.
        /// If x and y are given, return elements chosen from x or y depending on condition.
        /// Supports NDArray, bool[], IEnumerable&lt;bool&gt;, Array, IEnumerable, etc.
        /// </summary>
        public static NDArray where(NDArray condition, NDArray x, NDArray y)
        {
            var broadcasted = np.broadcast_arrays(condition, x, y);
            var broadcasted_condition = broadcasted[0];
            var broadcasted_x = broadcasted[1];
            var broadcasted_y = broadcasted[2];

            var cond = broadcasted_condition.MakeGeneric<bool>();
            var result = new NDArray(broadcasted_x.dtype, broadcasted_x.shape);
            var condSpan = cond.GetData().AsSpan<bool>();
            var xArr = broadcasted_x.GetData();
            var yArr = broadcasted_y.GetData();
            var resArr = result.GetData();

            for (int i = 0; i < condSpan.Length; i++)
            {
                resArr.SetIndex(i, condSpan[i] ? xArr.GetIndex(i) : yArr.GetIndex(i));
            }
            return result;
        }

        public static NDArray where(NDArray condition, NDArray x, NDArray y, Type outType)
        {
            return where(condition, x.astype(outType), y.astype(outType));
        }

        public static NDArray where(NDArray condition, NDArray x, NDArray y, NPTypeCode typeCode)
        {
            return where(condition, x.astype(typeCode), y.astype(typeCode));
        }

        public static NDArray[] where(NDArray condition)
        {
            return condition.TensorEngine.NonZero(condition);
        }

        public static NDArray[] where(bool[] condition)
        {
            return where(np.array(condition));
        }

        public static NDArray[] where(IEnumerable<bool> condition)
        {
            return where(np.array(condition.ToArray()));
        }

        public static NDArray where(bool[] condition, Array x, Array y)
        {
            return where(np.array(condition), np.array(x), np.array(y));
        }

        public static NDArray where(IEnumerable<bool> condition, IEnumerable x, IEnumerable y)
        {
            return where(np.array(condition.ToArray()), np.array(x.Cast<object>().ToArray()), np.array(y.Cast<object>().ToArray()));
        }
    }
}
