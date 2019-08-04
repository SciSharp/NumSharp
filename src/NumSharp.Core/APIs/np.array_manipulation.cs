using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray ndarray(Shape shape, Type dtype = null, Array buffer = null, char order = 'F')
            => BackendFactory.GetEngine().CreateNDArray(shape, dtype: dtype, buffer: buffer, order: order);

        /// <summary>
        /// Roll array elements along a given axis.
        /// 
        /// Elements that roll beyond the last position are re-introduced at the first.
        /// </summary>
        public static int roll(NDArray nd, int shift, int axis = -1)
            => (int) nd.roll(shift, axis);

        /// <summary>
        /// Returns a view of the array with axes transposed.
        /// For a 1-D array this has no effect, as a transposed vector is simply the same vector.To convert a 1-D array into a 2D column vector, an 
        /// additional dimension must be added. np.atleast2d(a).T achieves this, as does a[:, np.newaxis]. 
        /// For a 2-D array, this is a standard matrix transpose. For an n-D array, if axes are given, their 
        /// order indicates how the axes are permuted (see Examples). If axes are not provided 
        /// and a.shape = (i[0], i[1], ...i[n - 2], i[n - 1]), then a.transpose().shape = (i[n - 1], i[n - 2], ...i[1], i[0]).
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="axes">None or no argument: reverses the order of the axes. 
        /// Tuple of ints: i in the j-th place in the tuple means a’s i-th axis becomes a.transpose()’s j-th axis.</param>
        /// <returns>View of a, with axes suitably permuted</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.transpose.html?highlight=transpose#numpy.ndarray.transpose</remarks>
        public static NDArray transpose(NDArray x, int[] axes = null)
            => x.TensorEngine.Transpose(x, axes: axes);

        /// <summary>
        /// Find the unique elements of an array.
        /// 
        /// Returns the sorted unique elements of an array.There are three optional outputs in addition to the unique elements:
        /// * the indices of the input array that give the unique values
        /// * the indices of the unique array that reconstruct the input array
        /// * the number of times each unique value comes up in the input array
        /// </summary>
        public static NDArray unique<T>(NDArray a)
            => a.unique<T>();
    }
}
