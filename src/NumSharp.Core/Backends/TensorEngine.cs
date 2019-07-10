using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public abstract class TensorEngine
    {
        #region Allocation

        /// <summary>
        ///     Get storage for given <paramref name="dtype"/>.
        /// </summary>
        public abstract UnmanagedStorage GetStorage(Type dtype);

        /// <summary>
        ///     Get storage for given <paramref name="typeCode"/>.
        /// </summary>
        public abstract UnmanagedStorage GetStorage(NPTypeCode typeCode);

        #endregion

        #region Math

        public abstract NDArray Add(NDArray x, NDArray y);
        public abstract NDArray Dot(NDArray x, NDArray y);
        public abstract NDArray Divide(NDArray x, NDArray y);
        public abstract NDArray Log(NDArray nd);
        public abstract NDArray MatMul(NDArray x, NDArray y);
        public abstract NDArray Mean(NDArray x, int axis = -1);
        public abstract NDArray Multiply(NDArray x, NDArray y);
        public abstract NDArray Power(NDArray x, ValueType y);
        public abstract NDArray Sub(NDArray x, NDArray y);
        public abstract NDArray Sum(NDArray x, int? axis = null);
        public abstract NDArray Negate(NDArray x);

        #endregion

        #region Logic

        public abstract bool All(NDArray nd);
        public abstract NDArray<bool> All(NDArray nd, int axis);
        public abstract bool AllClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false);
        public abstract NDArray<bool> IsClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false);
        public abstract NDArray<bool> IsFinite(NDArray a);
        public abstract NDArray<bool> IsNan(NDArray a);

        #endregion

        #region Array Manipulation

        public abstract NDArray CreateNDArray(Shape shape, Type dtype = null, Array buffer = null, char order = 'C');
        public abstract NDArray CreateNDArray(Shape shape, Type dtype = null, IArraySlice buffer = null, char order = 'C');
        public abstract NDArray Transpose(NDArray nd, int[] axes = null);
        public abstract NDArray Cast(NDArray x, Type dtype, bool copy);

        #endregion

        #region Sorting, searching, counting

        public abstract NDArray ArgMax(NDArray nd, int axis = -1);

        #endregion
    }
}
