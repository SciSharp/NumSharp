using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
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

        // Binary arithmetic ufuncs — house parameter order (inputs, typeCode,
        // out, where). typeCode is NumPy's ufunc dtype=: it selects the LOOP
        // (computation runs in that dtype; inputs must be same_kind-castable
        // to it; out is validated against it). Divide is float-only — an
        // integer/bool dtype= raises NumPy's no-loop TypeError.
        public abstract NDArray Add(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Subtract(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Multiply(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Divide(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Mod(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);

        public abstract NDArray Mean(NDArray nd, int? axis = null, NPTypeCode? typeCode = null, bool keepdims = false);
        public abstract NDArray Mean(NDArray nd, int axis, Type dtype, bool keepdims = false);
        public abstract NDArray Power(NDArray lhs, NDArray rhs, Type type);
        public abstract NDArray Power(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray FloorDivide(NDArray lhs, NDArray rhs, Type dtype);
        public abstract NDArray FloorDivide(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Sum(NDArray nd, int? axis = null, NPTypeCode? typeCode = null, bool keepdims = false);
        public abstract NDArray Sum(NDArray nd, int axis, Type dtype, bool keepdims = false);
        public abstract NDArray Negate(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     NumPy 'positive' — identity at every numeric dtype (no bool loop).
        ///     typeCode selects the loop (positive(i32, dtype=f64) widens; a bool
        ///     loop request raises NumPy's did-not-contain-a-loop TypeError).
        /// </summary>
        public abstract NDArray Positive(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);

        public abstract NDArray Dot(NDArray x, NDArray y);
        public abstract NDArray Matmul(NDArray lhs, NDArray rhs);

        public abstract NDArray Abs(NDArray nd, Type dtype);
        public abstract NDArray Abs(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Sqrt(NDArray nd, Type dtype);
        public abstract NDArray Sqrt(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Log(NDArray nd, Type dtype);
        public abstract NDArray Log(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Log2(NDArray nd, Type dtype);
        public abstract NDArray Log2(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Log10(NDArray nd, Type dtype);
        public abstract NDArray Log10(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Log1p(NDArray nd, Type dtype);
        public abstract NDArray Log1p(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Exp(NDArray nd, Type dtype);
        public abstract NDArray Exp(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Exp2(NDArray nd, Type dtype);
        public abstract NDArray Exp2(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Expm1(NDArray nd, Type dtype);
        public abstract NDArray Expm1(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Tan(NDArray nd, Type dtype);
        public abstract NDArray Tan(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Sin(NDArray nd, Type dtype);
        public abstract NDArray Sin(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Cos(NDArray nd, Type dtype);
        public abstract NDArray Cos(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Sign(NDArray nd, Type dtype);
        public abstract NDArray Sign(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Floor(NDArray nd, Type dtype);
        public abstract NDArray Floor(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Ceil(NDArray nd, Type dtype);
        public abstract NDArray Ceil(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Round(NDArray nd, Type dtype);
        public abstract NDArray Round(NDArray nd, int decimals, Type dtype);
        public abstract NDArray Round(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Round(NDArray nd, int decimals, NPTypeCode? typeCode = null, NDArray @out = null);
        public abstract NDArray Truncate(NDArray nd, Type dtype);
        public abstract NDArray Truncate(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Reciprocal(NDArray nd, Type dtype);
        public abstract NDArray Reciprocal(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Square(NDArray nd, Type dtype);
        public abstract NDArray Square(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Deg2Rad(NDArray nd, Type dtype);
        public abstract NDArray Deg2Rad(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Rad2Deg(NDArray nd, Type dtype);
        public abstract NDArray Rad2Deg(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Invert(NDArray nd, Type dtype);
        public abstract NDArray Invert(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Cbrt(NDArray nd, Type dtype);
        public abstract NDArray Cbrt(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract (NDArray Fractional, NDArray Intergral) ModF(NDArray nd, Type dtype);
        public abstract (NDArray Fractional, NDArray Intergral) ModF(NDArray nd, NPTypeCode? typeCode = null);
       
        public abstract NDArray Tanh(NDArray nd, Type dtype);
        public abstract NDArray Tanh(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Cosh(NDArray nd, Type dtype);
        public abstract NDArray Cosh(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Sinh(NDArray nd, Type dtype);
        public abstract NDArray Sinh(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);

        public abstract NDArray ATan(NDArray nd, Type dtype);
        public abstract NDArray ATan(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray ATan2(NDArray y, NDArray x, Type dtype);
        public abstract NDArray ATan2(NDArray y, NDArray x, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray ACos(NDArray nd, Type dtype);
        public abstract NDArray ACos(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray ASin(NDArray nd, Type dtype);
        public abstract NDArray ASin(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);

        public abstract NDArray ClipNDArray(NDArray lhs, NDArray min, NDArray max, Type dtype, NDArray @out = null);
        public abstract NDArray ClipNDArray(NDArray lhs, NDArray min, NDArray max, NPTypeCode? typeCode = null, NDArray @out = null);

        /// <summary>
        ///     Fused evaluation of an <see cref="NpyExpr"/> tree in one iterator
        ///     pass (np.evaluate, roadmap Wave 6.1). Virtual with a
        ///     NotSupported default so alternative engines opt in explicitly.
        /// </summary>
        public virtual NDArray Evaluate(NpyExpr expr, NDArray @out = null)
            => throw new NotSupportedException($"{GetType().Name} does not support fused expression evaluation.");

        /// <summary>
        ///     Fused evaluation against an explicit operand list
        ///     (<see cref="NpyExpr.Input"/> leaves reference operands by position).
        /// </summary>
        public virtual NDArray Evaluate(NpyExpr expr, NDArray[] operands, NDArray @out = null)
            => throw new NotSupportedException($"{GetType().Name} does not support fused expression evaluation.");

        #endregion

        #region Logic

        // Comparison ufuncs — ONE NumPy-shaped member each (no bare/out split),
        // house parameter order (typeCode, out, where) like the unary family.
        // The loop output is bool: typeCode may only request Boolean (NumPy
        // raises the no-loop TypeError for anything else — dtype= is a
        // validate-only parameter on comparisons); with out= the provided
        // array is returned as-is (any numeric dtype — bool casts same_kind
        // to all of them), so the static return type is plain NDArray.
        // CONTRACT: a plain call (out == null && where == null) must return an
        // NDArray<bool> instance — the C# comparison operators rely on it for
        // their zero-alloc AsGeneric<bool>() typed sugar.
        public abstract NDArray Compare(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);  // Equal
        public abstract NDArray NotEqual(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Less(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray LessEqual(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Greater(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray GreaterEqual(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);

        // Bitwise operations — typeCode (ufunc dtype=) selects the loop among
        // the bool/integer loops; float/complex/decimal requests raise NumPy's
        // no-loop TypeError (the bitwise family has no such loops).
        public abstract NDArray BitwiseAnd(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray BitwiseOr(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray BitwiseXor(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);

        // Bit shift operations (integer types only)
        public abstract NDArray LeftShift(NDArray lhs, NDArray rhs);
        public abstract NDArray RightShift(NDArray lhs, NDArray rhs);

        public abstract bool All(NDArray nd);
        public abstract NDArray<bool> All(NDArray nd, int axis);
        public abstract bool Any(NDArray nd);
        public abstract NDArray<bool> Any(NDArray nd, int axis);
        public abstract bool AllClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false);
        public abstract NDArray<bool> IsClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false);
        // Predicate ufuncs — same single-member rule as the comparisons above
        // (bool loop, validate-only typeCode, plain-NDArray return, plain
        // calls must return an NDArray<bool> instance).
        public abstract NDArray IsFinite(NDArray a, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray IsNan(NDArray a, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray IsInf(NDArray a, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null);

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

        public abstract NDArray<long>[] NonZero(NDArray a);

        public abstract NDArray<long> FlatNonZero(NDArray a);

        public abstract NDArray Argwhere(NDArray a);

        public abstract long CountNonZero(NDArray a);

        public abstract NDArray CountNonZero(NDArray a, int axis, bool keepdims = false);

        // Boolean masking
        public abstract NDArray BooleanMask(NDArray arr, NDArray mask);

        public abstract void BooleanMaskSet(NDArray arr, NDArray mask, NDArray value);

        #endregion
    }
}
