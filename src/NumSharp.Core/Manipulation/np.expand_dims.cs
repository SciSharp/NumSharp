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

            //we create an nd-array of the shape and then slice/split it on axis index.
            var shape = np.array(a.shape);
            var left = shape[$":{axis}"].Cast<int>();
            var right = shape[$"{axis}:"].Cast<int>();

            //then we append a 1 dim between the slice/split.
            return new NDArray(a.Storage.Alias(new Shape(_combineEnumerables(left, __one, right).ToArray())));

            IEnumerable<T> _combineEnumerables<T>(params IEnumerable<T>[] arrays)
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
}
