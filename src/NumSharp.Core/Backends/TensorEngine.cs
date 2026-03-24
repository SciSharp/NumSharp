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

        public abstract NDArray ReduceAdd(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null, NDArray @out = null);
        public abstract NDArray ReduceCumAdd(NDArray arr, int? axis_, NPTypeCode? typeCode = null);
        public abstract NDArray ReduceCumMul(NDArray arr, int? axis_, NPTypeCode? typeCode = null);
        public abstract NDArray ReduceMean(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null);

        // NaN-aware reductions
        public abstract NDArray NanSum(NDArray a, int? axis = null, bool keepdims = false);
        public abstract NDArray NanProd(NDArray a, int? axis = null, bool keepdims = false);
        public abstract NDArray NanMin(NDArray a, int? axis = null, bool keepdims = false);
        public abstract NDArray NanMax(NDArray a, int? axis = null, bool keepdims = false);

        #endregion

        public abstract NDArray Add(NDArray lhs, NDArray rhs);
        public abstract NDArray Subtract(NDArray lhs, NDArray rhs);
        public abstract NDArray Multiply(NDArray lhs, NDArray rhs);
        public abstract NDArray Divide(NDArray lhs, NDArray rhs);
        public abstract NDArray Mod(NDArray lhs, NDArray rhs);

        public abstract NDArray Mean(NDArray nd, int? axis = null, NPTypeCode? typeCode = null, bool keepdims = false);
        public abstract NDArray Mean(NDArray nd, int axis, Type dtype, bool keepdims = false);
        public abstract NDArray Power(NDArray lhs, ValueType rhs, Type type);
        public abstract NDArray Power(NDArray lhs, ValueType rhs, NPTypeCode? typeCode = null);
        public abstract NDArray Power(NDArray lhs, NDArray rhs, Type type);
        public abstract NDArray Power(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null);
        public abstract NDArray FloorDivide(NDArray lhs, NDArray rhs, Type dtype);
        public abstract NDArray FloorDivide(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null);
        public abstract NDArray FloorDivide(NDArray lhs, ValueType rhs, Type dtype);
        public abstract NDArray FloorDivide(NDArray lhs, ValueType rhs, NPTypeCode? typeCode = null);
        public abstract NDArray Sum(NDArray nd, int? axis = null, NPTypeCode? typeCode = null, bool keepdims = false);
        public abstract NDArray Sum(NDArray nd, int axis, Type dtype, bool keepdims = false);
        public abstract NDArray Negate(NDArray nd);

        public abstract NDArray Dot(NDArray x, NDArray y);
        public abstract NDArray Matmul(NDArray lhs, NDArray rhs);

        public abstract NDArray Abs(NDArray nd, Type dtype);
        public abstract NDArray Abs(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Sqrt(NDArray nd, Type dtype);
        public abstract NDArray Sqrt(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Log(NDArray nd, Type dtype);
        public abstract NDArray Log(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Log2(NDArray nd, Type dtype);
        public abstract NDArray Log2(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Log10(NDArray nd, Type dtype);
        public abstract NDArray Log10(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Log1p(NDArray nd, Type dtype);
        public abstract NDArray Log1p(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Exp(NDArray nd, Type dtype);
        public abstract NDArray Exp(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Exp2(NDArray nd, Type dtype);
        public abstract NDArray Exp2(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Expm1(NDArray nd, Type dtype);
        public abstract NDArray Expm1(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Tan(NDArray nd, Type dtype);
        public abstract NDArray Tan(NDArray nd, NPTypeCode? typeCod = null);
        public abstract NDArray Sin(NDArray nd, Type dtype);
        public abstract NDArray Sin(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Cos(NDArray nd, Type dtype);
        public abstract NDArray Cos(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Sign(NDArray nd, Type dtype);
        public abstract NDArray Sign(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Floor(NDArray nd, Type dtype);
        public abstract NDArray Floor(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Ceil(NDArray nd, Type dtype);
        public abstract NDArray Ceil(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Round(NDArray nd, Type dtype);
        public abstract NDArray Round(NDArray nd, int decimals, Type dtype);
        public abstract NDArray Round(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Round(NDArray nd, int decimals, NPTypeCode? typeCode = null);
        public abstract NDArray Truncate(NDArray nd, Type dtype);
        public abstract NDArray Truncate(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Reciprocal(NDArray nd, Type dtype);
        public abstract NDArray Reciprocal(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Square(NDArray nd, Type dtype);
        public abstract NDArray Square(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Deg2Rad(NDArray nd, Type dtype);
        public abstract NDArray Deg2Rad(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Rad2Deg(NDArray nd, Type dtype);
        public abstract NDArray Rad2Deg(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Invert(NDArray nd, Type dtype);
        public abstract NDArray Invert(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Cbrt(NDArray nd, Type dtype);
        public abstract NDArray Cbrt(NDArray nd, NPTypeCode? typeCode = null);
        public abstract (NDArray Fractional, NDArray Intergral) ModF(NDArray nd, Type dtype);
        public abstract (NDArray Fractional, NDArray Intergral) ModF(NDArray nd, NPTypeCode? typeCode = null);
       
        public abstract NDArray Tanh(NDArray nd, Type dtype);
        public abstract NDArray Tanh(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Cosh(NDArray nd, Type dtype);
        public abstract NDArray Cosh(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray Sinh(NDArray nd, Type dtype);
        public abstract NDArray Sinh(NDArray nd, NPTypeCode? typeCode = null);

        public abstract NDArray ATan(NDArray nd, Type dtype);
        public abstract NDArray ATan(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray ATan2(NDArray y, NDArray x, Type dtype);
        public abstract NDArray ATan2(NDArray y, NDArray x, NPTypeCode? typeCode = null);
        public abstract NDArray ACos(NDArray nd, Type dtype);
        public abstract NDArray ACos(NDArray nd, NPTypeCode? typeCode = null);
        public abstract NDArray ASin(NDArray nd, Type dtype);
        public abstract NDArray ASin(NDArray nd, NPTypeCode? typeCode = null);

        public abstract NDArray Clip(NDArray lhs, ValueType min, ValueType max, Type dtype);
        public abstract NDArray Clip(NDArray lhs, ValueType min, ValueType max, NPTypeCode? typeCode = null);
        public abstract NDArray ClipNDArray(NDArray lhs, NDArray min, NDArray max, Type dtype, NDArray @out = null);
        public abstract NDArray ClipNDArray(NDArray lhs, NDArray min, NDArray max, NPTypeCode ? typeCode = null, NDArray @out = null);

        #endregion

        #region Logic

        // Comparison operations - all return NDArray<bool>
        public abstract NDArray<bool> Compare(NDArray lhs, NDArray rhs);  // Equal
        public abstract NDArray<bool> NotEqual(NDArray lhs, NDArray rhs);
        public abstract NDArray<bool> Less(NDArray lhs, NDArray rhs);
        public abstract NDArray<bool> LessEqual(NDArray lhs, NDArray rhs);
        public abstract NDArray<bool> Greater(NDArray lhs, NDArray rhs);
        public abstract NDArray<bool> GreaterEqual(NDArray lhs, NDArray rhs);

        // Bitwise operations
        public abstract NDArray BitwiseAnd(NDArray lhs, NDArray rhs);
        public abstract NDArray BitwiseOr(NDArray lhs, NDArray rhs);
        public abstract NDArray BitwiseXor(NDArray lhs, NDArray rhs);

        // Bit shift operations (integer types only)
        public abstract NDArray LeftShift(NDArray lhs, NDArray rhs);
        public abstract NDArray LeftShift(NDArray lhs, ValueType rhs);
        public abstract NDArray RightShift(NDArray lhs, NDArray rhs);
        public abstract NDArray RightShift(NDArray lhs, ValueType rhs);

        public abstract bool All(NDArray nd);
        public abstract NDArray<bool> All(NDArray nd, int axis);
        public abstract bool Any(NDArray nd);
        public abstract NDArray<bool> Any(NDArray nd, int axis);
        public abstract bool AllClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false);
        public abstract NDArray<bool> IsClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false);
        public abstract NDArray<bool> IsFinite(NDArray a);
        public abstract NDArray<bool> IsNan(NDArray a);
        public abstract NDArray<bool> IsInf(NDArray a);

        #endregion

        #region Array Manipulation

        public abstract NDArray CreateNDArray(Shape shape, Type dtype = null, Array buffer = null, char order = 'C');
        public abstract NDArray CreateNDArray(Shape shape, Type dtype = null, IArraySlice buffer = null, char order = 'C');
        public abstract NDArray Transpose(NDArray nd, int[] premute = null);
        public abstract NDArray SwapAxes(NDArray nd, int axis1, int axis2);
        public abstract NDArray MoveAxis(NDArray nd, int[] source, int[] destinition);
        public abstract NDArray RollAxis(NDArray nd, int axis, int start = 0);
        public abstract NDArray Cast(NDArray x, Type dtype, bool copy);
        public abstract NDArray Cast(NDArray x, NPTypeCode dtype, bool copy);

        #endregion

        #region Sorting, searching, counting

        #region Reduction

        public abstract NDArray ReduceAMax(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null);
        public abstract NDArray ReduceAMin(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null);
        public abstract NDArray ReduceArgMax(NDArray arr, int? axis_, bool keepdims = false);
        public abstract NDArray ReduceArgMin(NDArray arr, int? axis_, bool keepdims = false);
        public abstract NDArray ReduceProduct(NDArray arr, int? axis_, bool keepdims = false, NPTypeCode? typeCode = null);
        public abstract NDArray ReduceStd(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, NPTypeCode? typeCode = null);
        public abstract NDArray ReduceVar(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, NPTypeCode? typeCode = null);

        #endregion

        public abstract NDArray ArgMax(NDArray a);
        public abstract NDArray ArgMax(NDArray a, int axis, bool keepdims = false);

        public abstract NDArray ArgMin(NDArray a);
        public abstract NDArray ArgMin(NDArray a, int axis, bool keepdims = false);

        public abstract NDArray AMax(NDArray nd, int? axis = null, NPTypeCode? typeCode = null, bool keepdims = false);
        public abstract NDArray AMax(NDArray nd, int axis, Type dtype, bool keepdims = false);

        public abstract NDArray AMin(NDArray nd, int? axis = null, NPTypeCode? typeCode = null, bool keepdims = false);
        public abstract NDArray AMin(NDArray nd, int axis, Type dtype, bool keepdims = false);

        #endregion


        #region Indexing

        public abstract NDArray<int>[] NonZero(NDArray a);

        public abstract long CountNonZero(NDArray a);

        public abstract NDArray CountNonZero(NDArray a, int axis, bool keepdims = false);

        // Boolean masking
        public abstract NDArray BooleanMask(NDArray arr, NDArray mask);

        #endregion
    }
}
