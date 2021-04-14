using System.Numerics;

namespace NumSharp.Backends
{
    /// <summary>
    /// SIMD Tensor Engine implemented in pure micro-optimized C#.
    /// </summary>
    public partial class SimdEngine
    {
        public static T[] Subtract<T>(T[] left, T right) where T : unmanaged
        {
            var offset = Vector<T>.Count;
            var v2 = new Vector<T>(right);
            var result = new T[left.Length];

            int i = 0;
            for (i = 0; i < left.Length - offset; i += offset)
            {
                var v1 = new Vector<T>(left, i);
                (v1 - v2).CopyTo(result, i);
            }

            var move = offset - (left.Length - i);
            var v3 = new Vector<T>(left, i - move);
            (v3 - v2).CopyTo(result, i - move);
            //remaining items
            /*for (; i < left.Length; ++i)
            {
                result[i] = left[i] - right;
            }*/

            return result;
        }

        public static T[] Subtract<T>(T left, T[] right) where T : unmanaged
        {
            var offset = Vector<T>.Count;
            var v1 = new Vector<T>(left);
            var result = new T[right.Length];
            int i = 0;
            for (i = 0; i < right.Length - offset; i += offset)
            {
                var v2 = new Vector<T>(right, i);
                (v1 - v2).CopyTo(result, i);
            }

            var move = offset - (right.Length - i);
            var v3 = new Vector<T>(right, i - move);
            (v1 - v3).CopyTo(result, i - move);

            //remaining items
            /*for (; i < right.Length; ++i)
            {
                result[i] = left - right[i];
            }*/

            return result;
        }

        public static T[] Subtract<T>(T[] left, T[] right) where T : unmanaged
        {
            var offset = Vector<T>.Count;
            var result = new T[left.Length];
            int i = 0;
            for (i = 0; i < left.Length - offset; i += offset)
            {
                var v1 = new Vector<T>(left, i);
                var v2 = new Vector<T>(right, i);
                (v1 - v2).CopyTo(result, i);
            }

            var move = offset - (left.Length - i);
            var v3 = new Vector<T>(left, i - move);
            var v4 = new Vector<T>(right, i - move);
            (v3 - v4).CopyTo(result, i - move);

            //remaining items
            /*for (; i < left.Length; ++i)
            {
                result[i] = left[i] - right[i];
            }*/

            return result;
        }
    }
}
