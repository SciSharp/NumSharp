using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        private static readonly int[] __one = new int[] {1};

        public static NDArray expand_dims(NDArray a, int axis)
        {
            //test if the ndarray is empty.
            if (a.size == 0)
                return a;

            var shape = np.array(a.shape);
            var left = shape[$":{axis}"].Array.Cast<int>();
            var right = shape[$"{axis}:"].Array.Cast<int>();

            return a.reshape(_combineEnumerables(left, __one, right).ToArray());
        }

        private static IEnumerable<T> _combineEnumerables<T>(params IEnumerable<T>[] arrays)
        {
            foreach (var arr in arrays)
            {
                foreach (var val in arr)
                {
                    yield return val;
                }
            }
        }
    }
}
