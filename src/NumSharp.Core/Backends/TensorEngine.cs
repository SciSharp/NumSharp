using System;
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

        #region Reduction

        public abstract NDArray ReduceAdd(in NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null, NDArray @out = null);
        public abstract NDArray ReduceCumAdd(in NDArray arr, int? axis_, NPTypeCode? typeCode = null);
        public abstract NDArray ReduceMean(in NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null);

        #endregion

        public abstract NDArray Add(in NDArray lhs, in NDArray rhs);
        public abstract NDArray Subtract(in NDArray lhs, in NDArray rhs);
        public abstract NDArray Multiply(NDArray lhs, NDArray rhs);
        public abstract NDArray Divide(in NDArray lhs, in NDArray rhs);
        public abstract NDArray Mod(in NDArray lhs, in NDArray rhs);

        public abstract NDArray Mean(in NDArray nd, int? axis = null, NPTypeCode? typeCode = null, bool keepdims = false);
        public abstract NDArray Mean(in NDArray nd, int axis, Type dtype, bool keepdims = false);
        public abstract NDArray Power(in NDArray lhs, in ValueType rhs, Type type);
        public abstract NDArray Power(in NDArray lhs, in ValueType rhs, NPTypeCode? typeCode = null);
        public abstract NDArray Sum(in NDArray nd, int? axis = null, NPTypeCode? typeCode = null, bool keepdims = false);
        public abstract NDArray Sum(in NDArray nd, int axis, Type dtype, bool keepdims = false);
        public abstract NDArray Negate(in NDArray nd);

        public abstract NDArray Dot(in NDArray x, in NDArray y);
        public abstract NDArray Matmul(NDArray lhs, NDArray rhs);

        public abstract NDArray Abs(in NDArray nd, Type dtype);
        public abstract NDArray Abs(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Sqrt(in NDArray nd, Type dtype);
        public abstract NDArray Sqrt(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Log(in NDArray nd, Type dtype);
        public abstract NDArray Log(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Log2(in NDArray nd, Type dtype);
        public abstract NDArray Log2(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Log10(in NDArray nd, Type dtype);
        public abstract NDArray Log10(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Log1p(in NDArray nd, Type dtype);
        public abstract NDArray Log1p(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Exp(in NDArray nd, Type dtype);
        public abstract NDArray Exp(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Exp2(in NDArray nd, Type dtype);
        public abstract NDArray Exp2(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Expm1(in NDArray nd, Type dtype);
        public abstract NDArray Expm1(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Tan(in NDArray nd, Type dtype);
        public abstract NDArray Tan(in NDArray nd, NPTypeCode? typeCod = null);
        public abstract NDArray Sin(in NDArray nd, Type dtype);
        public abstract NDArray Sin(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Cos(in NDArray nd, Type dtype);
        public abstract NDArray Cos(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Sign(in NDArray nd, Type dtype);
        public abstract NDArray Sign(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Floor(in NDArray nd, Type dtype);
        public abstract NDArray Floor(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Ceil(in NDArray nd, Type dtype);
        public abstract NDArray Ceil(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Round(in NDArray nd, Type dtype);
        public abstract NDArray Round(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract (NDArray Fractional, NDArray Intergral) ModF(in NDArray nd, Type dtype);
        public abstract (NDArray Fractional, NDArray Intergral) ModF(in NDArray nd, NPTypeCode? typeCode = null);
       
        public abstract NDArray Tanh(in NDArray nd, Type dtype);
        public abstract NDArray Tanh(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Cosh(in NDArray nd, Type dtype);
        public abstract NDArray Cosh(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Sinh(in NDArray nd, Type dtype);
        public abstract NDArray Sinh(in NDArray nd, NPTypeCode? typeCode = null);

        public abstract NDArray ATan(in NDArray nd, Type dtype);
        public abstract NDArray ATan(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray ATan2(in NDArray y, in NDArray x, Type dtype);
        public abstract NDArray ATan2(in NDArray y, in NDArray x, NPTypeCode? typeCode = null);
        public abstract NDArray ACos(in NDArray nd, Type dtype);
        public abstract NDArray ACos(in NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray ASin(in NDArray nd, Type dtype);
        public abstract NDArray ASin(in NDArray nd, NPTypeCode? typeCode = null);

        public abstract NDArray Clip(in NDArray lhs, in ValueType min, in ValueType max, Type dtype);
        public abstract NDArray Clip(in NDArray lhs, in ValueType min, in ValueType max, NPTypeCode? typeCode = null);
        public abstract NDArray ClipNDArray(in NDArray lhs, in NDArray min, in NDArray max, Type dtype, NDArray @out = null);
        public abstract NDArray ClipNDArray(in NDArray lhs, in NDArray min, in NDArray max, NPTypeCode ? typeCode = null, NDArray @out = null);

        #endregion

        #region Logic

        public abstract NDArray<bool> Compare(in NDArray lhs, in NDArray rhs);
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
        public abstract NDArray Transpose(in NDArray nd, int[] premute = null);
        public abstract NDArray SwapAxes(in NDArray nd, int axis1, int axis2);
        public abstract NDArray MoveAxis(in NDArray nd, int[] source, int[] destinition);
        public abstract NDArray RollAxis(in NDArray nd, int axis, int start = 0);
        public abstract NDArray Cast(NDArray x, Type dtype, bool copy);
        public abstract NDArray Cast(NDArray x, NPTypeCode dtype, bool copy);

        #endregion

        #region Sorting, searching, counting

        #region Reduction

        public abstract NDArray ReduceAMax(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null);
        public abstract NDArray ReduceAMin(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null);
        public abstract NDArray ReduceArgMax(NDArray arr, int? axis_);
        public abstract NDArray ReduceArgMin(NDArray arr, int? axis_);
        public abstract NDArray ReduceProduct(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null);
        public abstract NDArray ReduceStd(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, NPTypeCode? typeCode = null);
        public abstract NDArray ReduceVar(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, NPTypeCode? typeCode = null);

        #endregion

        public abstract NDArray ArgMax(in NDArray a);
        public abstract NDArray ArgMax(in NDArray a, int axis);

        public abstract NDArray ArgMin(in NDArray a);
        public abstract NDArray ArgMin(in NDArray a, int axis);

        public abstract NDArray AMax(in NDArray nd, int? axis = null, NPTypeCode? typeCode = null, bool keepdims = false);
        public abstract NDArray AMax(in NDArray nd, int axis, Type dtype, bool keepdims = false);

        public abstract NDArray AMin(in NDArray nd, int? axis = null, NPTypeCode? typeCode = null, bool keepdims = false);
        public abstract NDArray AMin(in NDArray nd, int axis, Type dtype, bool keepdims = false);

        #endregion


        #region Indexing

        public abstract NDArray<int>[] NonZero(in NDArray a);

        #endregion
    }
}
