using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     An external BLAS this engine delegates its matrix products to, or null (the default)
        ///     to compute them with NumSharp's own managed SIMD kernels.
        /// </summary>
        /// <remarks>
        ///     NumSharp.Core is 100 % managed C# and never assigns this. It is set by an optional
        ///     package — <c>NumSharp.Interop.OpenBLAS</c> does it from a <c>[ModuleInitializer]</c>, so
        ///     referencing that package is the whole opt-in — and can be cleared again by assigning
        ///     null. A backend only answers for what it implements (see
        ///     <see cref="IBlasBackend"/>); everything else falls back to the managed kernels.
        ///     <para>
        ///     Engines are cached singletons (<see cref="BackendFactory"/>), so assigning this
        ///     affects every array that resolves to the engine — including ones already created.
        ///     </para>
        /// </remarks>
        public IBlasBackend Blas { get; set; }

        #region Allocation

        /// <summary>
        ///     Get storage for given <paramref name="dtype"/>.
        /// </summary>
        public abstract UnmanagedStorage GetStorage(Type dtype);

        /// <summary>
        ///     Get storage described by the dtype DESCRIPTOR <paramref name="descr"/> (NumPy's <c>PyArray_NewFromDescr</c>
        ///     entry — the descriptor becomes the array's <c>dtype</c>, its class supplies the storage lane). The default
        ///     routes through <see cref="GetStorage(Type)"/>, which is exact for every builtin (one canonical descriptor
        ///     per lane); an engine that carries parametric descriptors overrides it.
        /// </summary>
        public virtual UnmanagedStorage GetStorage(DType descr)
        {
            if (descr is null)
                throw new ArgumentNullException(nameof(descr));
            return GetStorage((Type)descr);
        }

        #endregion

        #region Math

        #region Reduction

        public abstract NDArray ReduceAdd(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null, NDArray @out = null);
        /// <summary>
        ///     NumPy's <c>np.cumsum</c> (<c>np.add.accumulate</c>), the whole call: the running sums along
        ///     <paramref name="axis_"/> (the flattened array when null), stored into <paramref name="out"/> when given.
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="axis_">The accumulated axis (negative counts from the end), or null for the flattened array.</param>
        /// <param name="dtype">The loop / result dtype (<c>dtype=</c>), or null for NEP 50's accumulator of the input.</param>
        /// <param name="out">NumPy's <c>out=</c>, an array of the result's shape that receives it with unsafe casting, or
        ///     null for a fresh result.</param>
        /// <returns><paramref name="out"/> itself when given, else a fresh array.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range.</exception>
        /// <exception cref="TypeError">A text operand or request NumPy's loop resolution refuses.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>, or text a numeric loop cannot parse.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <remarks>
        ///     The engine owns NumPy's whole call, dtype decisions included (the np-layer dtype plan): the public
        ///     <c>np.cumsum</c> only applies the matrix boundary and calls this. <paramref name="out"/> was added to the
        ///     contract in 0.71.0 (a breaking change for a subclass, decision D2 of that plan).
        /// </remarks>
        public abstract NDArray ReduceCumAdd(NDArray arr, int? axis_, DType dtype = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.cumprod</c> (<c>np.multiply.accumulate</c>), the whole call: the running products along
        ///     <paramref name="axis_"/> (the flattened array when null), stored into <paramref name="out"/> when given.
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="axis_">The accumulated axis (negative counts from the end), or null for the flattened array.</param>
        /// <param name="dtype">The loop / result dtype (<c>dtype=</c>), or null for NEP 50's accumulator of the input.</param>
        /// <param name="out">NumPy's <c>out=</c>, an array of the result's shape that receives it with unsafe casting, or
        ///     null for a fresh result.</param>
        /// <returns><paramref name="out"/> itself when given, else a fresh array.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range.</exception>
        /// <exception cref="TypeError">A text operand or request NumPy's loop resolution refuses.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>, or text a numeric dtype cannot parse.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <remarks>
        ///     The engine owns NumPy's whole call, dtype decisions included (the np-layer dtype plan): the public
        ///     <c>np.cumprod</c> only applies the matrix boundary and calls this. <paramref name="out"/> was added to the
        ///     contract in 0.71.0 (a breaking change for a subclass, decision D2 of that plan).
        /// </remarks>
        public abstract NDArray ReduceCumMul(NDArray arr, int? axis_, DType dtype = null, NDArray @out = null);
        public abstract NDArray ReduceMean(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null);

        // NaN-aware reductions
        public abstract NDArray NanSum(NDArray a, int? axis = null, bool keepdims = false);
        public abstract NDArray NanProd(NDArray a, int? axis = null, bool keepdims = false);
        public abstract NDArray NanMin(NDArray a, int? axis = null, bool keepdims = false);
        public abstract NDArray NanMax(NDArray a, int? axis = null, bool keepdims = false);

        #endregion

        // Binary arithmetic ufuncs — house parameter order (inputs, dtype,
        // out, where). dtype is NumPy's ufunc dtype=: it selects the LOOP
        // (computation runs in that dtype; inputs must be same_kind-castable
        // to it; out is validated against it). Divide is float-only — an
        // integer/bool dtype= raises NumPy's no-loop TypeError.
        public abstract NDArray Add(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Subtract(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Multiply(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Divide(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Mod(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     np.fmod — C-style truncated remainder (result takes the DIVIDEND's sign), the
        ///     complement of <see cref="Mod"/> (floored, divisor's sign). Same NEP50 promotion as
        ///     Mod (integer stays integer, bool -> int8) and the same (inputs, dtype, out, where)
        ///     house order.
        /// </summary>
        public abstract NDArray Fmod(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     np.gcd — element-wise greatest common divisor of |lhs| and |lhs|. INTEGER-ONLY: a
        ///     bool/float/complex/decimal operand (or a uint64+signed pair promoting to float64) has no
        ///     loop and raises NumPy's no-loop <see cref="TypeError"/>. Uniform NEP50 promotion
        ///     (both operands + output share one integer dtype); a negative result is possible only where
        ///     the magnitude wraps the signed range (gcd(-128,-128)==-128). Same (inputs, dtype, out, where)
        ///     house order as the rest of the binary ufunc surface.
        /// </summary>
        /// <param name="lhs">First input array.</param>
        /// <param name="rhs">Second input array.</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=); must name an integer loop.</param>
        /// <param name="out">Output location (NumPy ufunc out=); returned as-is when supplied.</param>
        /// <param name="where">Boolean mask (NumPy ufunc where=); masked-off elements keep their prior value.</param>
        /// <returns>The element-wise gcd, in the promoted integer dtype.</returns>
        public abstract NDArray Gcd(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     np.lcm — element-wise lowest common multiple of |lhs| and |rhs| (0 when either is 0). Shares
        ///     every rule with <see cref="Gcd"/> (integer-only loops, uniform NEP50 promotion, same house
        ///     order); the product |lhs|/gcd*|rhs| WRAPS the dtype on overflow, matching NumPy.
        /// </summary>
        /// <param name="lhs">First input array.</param>
        /// <param name="rhs">Second input array.</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=); must name an integer loop.</param>
        /// <param name="out">Output location (NumPy ufunc out=); returned as-is when supplied.</param>
        /// <param name="where">Boolean mask (NumPy ufunc where=); masked-off elements keep their prior value.</param>
        /// <returns>The element-wise lcm, in the promoted integer dtype.</returns>
        public abstract NDArray Lcm(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     np.divmod — the fused two-output ufunc returning
        ///     <c>(floor_divide(lhs, rhs), remainder(lhs, rhs))</c> element-wise in a single pass.
        ///     Item1 is the floored quotient, Item2 the floored remainder (divisor's sign). Same NEP50
        ///     promotion as <see cref="Mod"/>/<see cref="FloorDivide"/>. When <paramref name="out"/>
        ///     (a 2-tuple) or <paramref name="where"/> is supplied the op composes the two validated
        ///     single-output ufuncs; otherwise it runs the fused kernel.
        /// </summary>
        public abstract (NDArray Quotient, NDArray Remainder) DivMod(NDArray lhs, NDArray rhs, DType dtype = null, (NDArray Quotient, NDArray Remainder) @out = default, NDArray where = null);

        // Element-wise min/max ufuncs (np.maximum / np.minimum / np.fmax / np.fmin).
        // Maximum/Minimum PROPAGATE NaN (a NaN operand wins); FMax/FMin IGNORE NaN
        // (the non-NaN operand wins). Same (inputs, dtype, out, where) house order.
        public abstract NDArray Maximum(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Minimum(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray FMax(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray FMin(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);

        public abstract NDArray Mean(NDArray nd, int? axis = null, DType dtype = null, bool keepdims = false);
        public abstract NDArray Power(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     Element-wise power with a MINIMUM precision of float64 (np.float_power). Unlike
        ///     <see cref="Power"/>, float_power has ONLY the float64 (<c>dd->d</c>) and complex128
        ///     (<c>DD->D</c>) loops, so every real input (bool/int/float16/float32/decimal/char)
        ///     promotes to float64 and a complex operand promotes to complex128 — the result is
        ///     therefore always an inexact float. That absence of an integer loop is exactly why
        ///     float_power does NOT raise Power's "Integers to negative integer powers are not
        ///     allowed" (<c>float_power(2, -1) == 0.5</c>). The computation itself is bit-identical
        ///     to <see cref="Power"/> on those two loops.
        /// </summary>
        /// <param name="lhs">The bases (any dtype; promoted to the float loop).</param>
        /// <param name="rhs">The exponents (any dtype; promoted to the float loop).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc <c>dtype=</c>). Only float64 or
        ///     complex128 select a loop; any other request raises NumPy's "No loop matching the
        ///     specified signature and casting was found for ufunc float_power".</param>
        /// <param name="out">Destination (NumPy ufunc <c>out=</c>): joins the broadcast without being
        ///     stretched, must be same_kind-castable from the float loop dtype; returned as-is.</param>
        /// <param name="where">Boolean write-mask (NumPy ufunc <c>where=</c>): only mask-true slots are
        ///     computed/written; false slots keep the prior <paramref name="out"/> contents.</param>
        /// <returns>The bases raised to the exponents at float64/complex128 precision (or
        ///     <paramref name="out"/> when supplied).</returns>
        /// <exception cref="IncorrectTypeException">A <paramref name="dtype"/> other than float64 or
        ///     complex128 was requested (no such loop exists).</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is non-bool; a complex
        ///     input cannot same_kind-cast to a requested float64 loop; <paramref name="out"/> is
        ///     read-only or not same_kind-castable from the loop dtype; or the shapes do not broadcast.</exception>
        public abstract NDArray FloatPower(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray FloorDivide(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Sum(NDArray nd, int? axis = null, DType dtype = null, bool keepdims = false);
        public abstract NDArray Negate(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     NumPy 'positive' — identity at every numeric dtype (no bool loop).
        ///     dtype selects the loop (positive(i32, dtype=f64) widens; a bool
        ///     loop request raises NumPy's did-not-contain-a-loop TypeError).
        /// </summary>
        public abstract NDArray Positive(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     NumPy 'conjugate' / 'conj' — the complex conjugate, element-wise. Identity at every
        ///     real dtype (values unchanged, dtype preserved); for Complex it flips the sign of the
        ///     imaginary part. dtype selects the loop; out=/where= follow the ufunc contract.
        /// </summary>
        public abstract NDArray Conjugate(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);

        public abstract NDArray Dot(NDArray x, NDArray y);
        public abstract NDArray Matmul(NDArray lhs, NDArray rhs);

        public abstract NDArray Abs(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        // np.fabs — the FLOAT-ONLY absolute value: promotes int/bool to float (NEP50 tier) and has NO
        // complex loop (unlike Abs, which preserves int and maps complex→magnitude). The operation IS
        // Abs on the float loops (clear the IEEE sign bit); Fabs differs only in dtype resolution + the
        // complex rejection, so the engine reuses the SIMD UnaryOp.Abs kernel.
        public abstract NDArray Fabs(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Sqrt(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Log(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Log2(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Log10(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Log1p(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Exp(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Exp2(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Expm1(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Tan(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Sin(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Cos(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Sign(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Floor(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Ceil(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Round(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Round(NDArray nd, int decimals, DType dtype = null, NDArray @out = null);
        public abstract NDArray Truncate(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Rint(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Reciprocal(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Square(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Deg2Rad(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Rad2Deg(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Invert(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        // np.bitwise_count — set-bit count of |x|; integer/bool input, uint8 output (dtype= accepts only uint8).
        public abstract NDArray BitwiseCount(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Cbrt(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        // np.modf — the two-output (fractional, integral) ufunc. dtype selects the float loop (e/f/d/g;
        // int/bool/char promote to the narrowest float per width — bool/i8/u8->f16, i16/u16/char->f32,
        // i32+->f64; complex has no loop); outFrac/outIntegral are the two ufunc outputs (either may be
        // null to auto-allocate); where masks BOTH outputs.
        public abstract (NDArray Fractional, NDArray Integral) ModF(NDArray nd, DType dtype = null,
            NDArray outFrac = null, NDArray outIntegral = null, NDArray where = null);
        // np.frexp — decompose x into (mantissa in [0.5,1), int32 exponent) with x == mantissa * 2^exponent
        // (the two-output inverse of Ldexp). The mantissa carries x's float tier; the exponent is always int32.
        // out1/out2 are the ufunc's two output operands (NumPy's positional out1,out2 / out=(o1,o2)); where= masks
        // both (masked-off out slots keep prior contents). Each out takes a same_kind cast from its loop dtype.
        public abstract (NDArray Mantissa, NDArray Exponent) Frexp(NDArray x, NDArray out1 = null, NDArray out2 = null, NDArray where = null);
        // np.ldexp — compose x1 * 2^x2 (the inverse of Frexp). x1 is the float mantissa (int/bool promote to a
        // float tier), x2 an INTEGER exponent that does NOT widen x1's dtype; the result is x1's float tier.
        // out=/where= follow the single-output ufunc contract (same_kind cast into out, masked write).
        public abstract NDArray Ldexp(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null);

        public abstract NDArray Tanh(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Cosh(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Sinh(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);

        public abstract NDArray ATan(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray ATan2(NDArray y, NDArray x, DType dtype = null, NDArray @out = null, NDArray where = null);
        // np.logaddexp / np.logaddexp2 / np.nextafter / np.copysign / np.hypot — float-tier binary
        // ufuncs (same promotion as ATan2).
        public abstract NDArray LogAddExp(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray LogAddExp2(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray NextAfter(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray CopySign(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Hypot(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null);
        // np.heaviside — float-tier binary ufunc (same promotion as ATan2/Hypot); NOT commutative, and the
        // only family member with a branchless SIMD fast path (step is compare+select, no libm).
        public abstract NDArray Heaviside(NDArray x1, NDArray x2, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray ACos(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray ASin(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);

        public abstract NDArray ASinh(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray ACosh(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray ATanh(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);

        // np.spacing — distance to the adjacent representable value away from zero (one ULP). Float-only
        // unary ufunc (ee/ff/dd loops + NumSharp's decimal extension); complex has NO loop.
        public abstract NDArray Spacing(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        /// np.clip's engine entry (NumPy's <c>_methods._clip</c> plus the routed ufunc): drops an unclipping
        /// Python-int bound, routes to positive / minimum / maximum / clip, and broadcasts the array, the bounds,
        /// <paramref name="out"/> and <paramref name="where"/> together.
        /// </summary>
        /// <param name="lhs">The array to clip.</param>
        /// <param name="min">The lower bound, or null for none (a 0-d bound is a scalar; a 0-d integer bound is a Python int).</param>
        /// <param name="max">The upper bound, or null for none.</param>
        /// <param name="dtype">The loop dtype (NumPy's <c>dtype=</c>), or null for the promoted one.</param>
        /// <param name="out">The out= array, or null for a fresh result.</param>
        /// <param name="where">The bool where= mask, or null for everywhere.</param>
        /// <returns><paramref name="out"/>, or a fresh array in NumPy's ufunc output layout.</returns>
        /// <exception cref="ValueError">A read-only out, or operands that do not broadcast / an out that would be stretched.</exception>
        /// <exception cref="ArgumentException">A non-bool where, or a dtype= / out= cast NumPy's same_kind rule refuses.</exception>
        /// <exception cref="OverflowException">A kept Python-int bound does not fit an integer loop dtype.</exception>
        /// <exception cref="TypeError">No bound is left and the loop would be bool (positive has no bool loop).</exception>
        public abstract NDArray ClipNDArray(NDArray lhs, NDArray min, NDArray max, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     Fused evaluation of an <see cref="NDExpr"/> tree in one iterator
        ///     pass (np.evaluate, roadmap Wave 6.1). Virtual with a
        ///     NotSupported default so alternative engines opt in explicitly.
        /// </summary>
        /// <param name="options">Resolved <c>dtype=</c>/<c>casting=</c>/<c>order=</c> keywords (plan P4.5); <c>default</c> = no override.</param>
        public virtual NDArray Evaluate(NDExpr expr, NDArray @out = null, in NDEvaluateOptions options = default)
            => throw new NotSupportedException($"{GetType().Name} does not support fused expression evaluation.");

        /// <summary>
        ///     Fused evaluation against an explicit operand list
        ///     (<see cref="NDExpr.Input"/> leaves reference operands by position).
        /// </summary>
        /// <param name="options">Resolved <c>dtype=</c>/<c>casting=</c>/<c>order=</c> keywords (plan P4.5); <c>default</c> = no override.</param>
        public virtual NDArray Evaluate(NDExpr expr, NDArray[] operands, NDArray @out = null, in NDEvaluateOptions options = default)
            => throw new NotSupportedException($"{GetType().Name} does not support fused expression evaluation.");

        /// <summary>
        ///     Fused evaluation of an already-compiled program (<see cref="CompiledExpression"/>):
        ///     the per-call work only — iteration shape, result allocation, iterator. The two
        ///     <c>Evaluate(NDExpr, …)</c> members above resolve the program from the tree's cache and
        ///     land here.
        /// </summary>
        /// <param name="options">Resolved <c>dtype=</c>/<c>casting=</c>/<c>order=</c> keywords (plan P4.5); <c>default</c> = no override.</param>
        internal virtual NDArray Evaluate(NDExprProgram program, NDArray[] operands, NDArray @out, in NDEvaluateOptions options = default)
            => throw new NotSupportedException($"{GetType().Name} does not support fused expression evaluation.");

        #endregion

        #region Logic

        // Comparison ufuncs — ONE NumPy-shaped member each (no bare/out split),
        // house parameter order (dtype, out, where) like the unary family.
        // The loop output is bool: dtype may only request Boolean (NumPy
        // raises the no-loop TypeError for anything else — dtype= is a
        // validate-only parameter on comparisons); with out= the provided
        // array is returned as-is (any numeric dtype — bool casts same_kind
        // to all of them), so the static return type is plain NDArray.
        // CONTRACT: a plain call (out == null && where == null) must return an
        // NDArray<bool> instance — the C# comparison operators rely on it for
        // their zero-alloc AsGeneric<bool>() typed sugar.
        public abstract NDArray Compare(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);  // Equal
        public abstract NDArray NotEqual(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Less(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray LessEqual(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Greater(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray GreaterEqual(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);

        // Bitwise operations — dtype (ufunc dtype=) selects the loop among
        // the bool/integer loops; float/complex/decimal requests raise NumPy's
        // no-loop TypeError (the bitwise family has no such loops).
        public abstract NDArray BitwiseAnd(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray BitwiseOr(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray BitwiseXor(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);

        // Bit shift operations (integer types only). dtype (ufunc dtype=) selects the loop among the
        // bool/integer loops; a float/complex/decimal request raises NumPy's no-loop TypeError (shifts
        // have no such loops). out/where follow the standard ufunc contract (same_kind cast into out,
        // masked write). A uint64×signed pair promotes to float64 (no loop) and raises "not supported".
        public abstract NDArray LeftShift(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray RightShift(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     NumPy's <c>np.all(a)</c> over every element: <c>logical_and.reduce</c> of the elements' truth values (a text
        ///     element reads true when non-empty), True for an empty array (the identity).
        /// </summary>
        /// <param name="nd">The array (any dtype, any layout).</param>
        /// <returns>Whether every element is true.</returns>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF (NumPy's bool cast of the
        ///     text).</exception>
        public abstract bool All(NDArray nd);

        /// <summary>
        ///     <c>np.all</c> along one axis, or over every axis when <paramref name="axis"/> is null: the boolean reduction
        ///     without <c>out=</c> or <c>where=</c>, a fresh array the caller may wrap as NumPy's scalar.
        /// </summary>
        /// <param name="nd">The array (any dtype, any layout).</param>
        /// <param name="axis">The reduced axis (negative counts from the end; a 0-d array accepts 0 and -1), or null for
        ///     every axis.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <returns>The bool result, laid out as NumPy's reduction allocates it. Over every axis it is a fresh, writeable
        ///     owner (a 0-d array without <paramref name="keepdims"/>); the public <c>np.all</c> overloads decide whether a
        ///     0-d result becomes NumPy's read-only scalar.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="nd"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF.</exception>
        /// <remarks><paramref name="keepdims"/> and the null axis were added to the contract in 0.71.0 (the np-layer dtype
        ///     plan's P1b, decision D2: widened in place; the member took one <c>int</c> axis before).</remarks>
        public abstract NDArray<bool> All(NDArray nd, int? axis, bool keepdims = false);

        /// <summary>
        ///     <c>np.all</c> over a tuple of axes: every axis checked, then a repeat refused, then the boolean reduction
        ///     (an EMPTY tuple reduces nothing and returns the truth values in the input's shape).
        /// </summary>
        /// <param name="nd">The array (any dtype, any layout).</param>
        /// <param name="axis">The reduced axes (negative values count from the end).</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <returns>The bool result, a fresh array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="nd"/> or <paramref name="axis"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of range (the first one, in the tuple's order).</exception>
        /// <exception cref="ValueError">An axis appears twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF.</exception>
        /// <remarks>Abstract since 0.71.0 (the np-layer dtype plan's P1b): <see cref="DefaultEngine"/> had it as its own
        ///     member, which the public <c>np.all</c> reached by a type test.</remarks>
        public abstract NDArray<bool> All(NDArray nd, int[] axis, bool keepdims = false);

        /// <summary>
        ///     NumPy's <c>np.all(a, axis, out, keepdims, *, where)</c>, the whole call: <c>logical_and.reduce(a, axis, bool,
        ///     out, keepdims=keepdims, where=where)</c> — the reduce prologue in NumPy's order, the text truth route, the masked
        ///     reduction and the <c>out=</c> store.
        /// </summary>
        /// <param name="a">The array (any dtype, any layout).</param>
        /// <param name="axis">The reduced axis, or null for every axis (NumPy's <c>axis=None</c>).</param>
        /// <param name="out">NumPy's <c>out=</c>: an array of the result's shape and any dtype that receives the bool result
        ///     with unsafe casting, or null for a fresh result.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="where">NumPy's <c>where=</c>: a bool mask that broadcasts to <paramref name="a"/>'s shape (masked-off
        ///     elements count as True, the identity), or null for no mask.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise a fresh bool array, read-only when 0-d (NumPy's
        ///     scalar).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only (<c>output array is read-only</c>, checked
        ///     first), has the wrong rank, or cannot hold the result; or <paramref name="where"/> has more axes than
        ///     <paramref name="a"/> or would make it stretch (NumPy's reduction-iterator texts).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array (NumPy's TypeError text: the
        ///     mask must cast to bool under the <c>safe</c> rule); or <paramref name="out"/> does not have the result's
        ///     shape.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast
        ///     against <paramref name="a"/>.</exception>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF.</exception>
        /// <remarks>
        ///     The engine owns NumPy's whole call, dtype decisions included (the np-layer dtype plan, P1b): the public
        ///     <c>np.all</c> only applies the matrix boundary and calls this. New in 0.71.0. A bare <c>null</c> axis is
        ///     ambiguous with the tuple member below: pass a typed <c>int?</c>.
        /// </remarks>
        public abstract NDArray All(NDArray a, int? axis, NDArray @out, bool keepdims = false, NDArray where = null);

        /// <summary>
        ///     NumPy's <c>np.all</c> with a tuple <paramref name="axis"/>, the whole call: the reduce prologue (every axis,
        ///     then a repeat), the text truth route, the masked reduction and the <c>out=</c> store.
        /// </summary>
        /// <param name="a">The array (any dtype, any layout).</param>
        /// <param name="axis">The reduced axes (an empty tuple reduces nothing).</param>
        /// <param name="out">NumPy's <c>out=</c>, or null for a fresh result.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="where">NumPy's <c>where=</c> bool mask, or null.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise the fresh bool array, read-only when 0-d.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="axis"/> is null.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>, a mask with more axes than
        ///     <paramref name="a"/>, or an axis that appears twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array, or <paramref name="out"/> does
        ///     not have the result's shape.</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast
        ///     against <paramref name="a"/>.</exception>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, P1b).</remarks>
        public abstract NDArray All(NDArray a, int[] axis, NDArray @out, bool keepdims = false, NDArray where = null);

        /// <summary>
        ///     NumPy's <c>np.any(a)</c> over every element: <c>logical_or.reduce</c> of the elements' truth values (a text
        ///     element reads true when non-empty), False for an empty array (the identity).
        /// </summary>
        /// <param name="nd">The array (any dtype, any layout).</param>
        /// <returns>Whether any element is true.</returns>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF (NumPy's bool cast of the
        ///     text).</exception>
        public abstract bool Any(NDArray nd);

        /// <summary>
        ///     <c>np.any</c> along one axis, or over every axis when <paramref name="axis"/> is null: the boolean reduction
        ///     without <c>out=</c> or <c>where=</c>, a fresh array the caller may wrap as NumPy's scalar.
        /// </summary>
        /// <param name="nd">The array (any dtype, any layout).</param>
        /// <param name="axis">The reduced axis (negative counts from the end; a 0-d array accepts 0 and -1), or null for
        ///     every axis.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <returns>The bool result, laid out as NumPy's reduction allocates it. Over every axis it is a fresh, writeable
        ///     owner (a 0-d array without <paramref name="keepdims"/>); the public <c>np.any</c> overloads decide whether a
        ///     0-d result becomes NumPy's read-only scalar.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="nd"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF.</exception>
        /// <remarks><paramref name="keepdims"/> and the null axis were added to the contract in 0.71.0 (the np-layer dtype
        ///     plan's P1b, decision D2: widened in place; the member took one <c>int</c> axis before).</remarks>
        public abstract NDArray<bool> Any(NDArray nd, int? axis, bool keepdims = false);

        /// <summary>
        ///     <c>np.any</c> over a tuple of axes: every axis checked, then a repeat refused, then the boolean reduction
        ///     (an EMPTY tuple reduces nothing and returns the truth values in the input's shape).
        /// </summary>
        /// <param name="nd">The array (any dtype, any layout).</param>
        /// <param name="axis">The reduced axes (negative values count from the end).</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <returns>The bool result, a fresh array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="nd"/> or <paramref name="axis"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of range (the first one, in the tuple's order).</exception>
        /// <exception cref="ValueError">An axis appears twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF.</exception>
        /// <remarks>Abstract since 0.71.0 (the np-layer dtype plan's P1b): <see cref="DefaultEngine"/> had it as its own
        ///     member, which the public <c>np.any</c> reached by a type test.</remarks>
        public abstract NDArray<bool> Any(NDArray nd, int[] axis, bool keepdims = false);

        /// <summary>
        ///     NumPy's <c>np.any(a, axis, out, keepdims, *, where)</c>, the whole call: <c>logical_or.reduce(a, axis, bool,
        ///     out, keepdims=keepdims, where=where)</c> — the reduce prologue in NumPy's order, the text truth route, the masked
        ///     reduction and the <c>out=</c> store.
        /// </summary>
        /// <param name="a">The array (any dtype, any layout).</param>
        /// <param name="axis">The reduced axis, or null for every axis (NumPy's <c>axis=None</c>).</param>
        /// <param name="out">NumPy's <c>out=</c>: an array of the result's shape and any dtype that receives the bool result
        ///     with unsafe casting, or null for a fresh result.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="where">NumPy's <c>where=</c>: a bool mask that broadcasts to <paramref name="a"/>'s shape (masked-off
        ///     elements count as False, the identity), or null for no mask.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise a fresh bool array, read-only when 0-d (NumPy's
        ///     scalar).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only (<c>output array is read-only</c>, checked
        ///     first), has the wrong rank, or cannot hold the result; or <paramref name="where"/> has more axes than
        ///     <paramref name="a"/> or would make it stretch (NumPy's reduction-iterator texts).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array (NumPy's TypeError text: the
        ///     mask must cast to bool under the <c>safe</c> rule); or <paramref name="out"/> does not have the result's
        ///     shape.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast
        ///     against <paramref name="a"/>.</exception>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF.</exception>
        /// <remarks>
        ///     The engine owns NumPy's whole call, dtype decisions included (the np-layer dtype plan, P1b): the public
        ///     <c>np.any</c> only applies the matrix boundary and calls this. New in 0.71.0. A bare <c>null</c> axis is
        ///     ambiguous with the tuple member below: pass a typed <c>int?</c>.
        /// </remarks>
        public abstract NDArray Any(NDArray a, int? axis, NDArray @out, bool keepdims = false, NDArray where = null);

        /// <summary>
        ///     NumPy's <c>np.any</c> with a tuple <paramref name="axis"/>, the whole call: the reduce prologue (every axis,
        ///     then a repeat), the text truth route, the masked reduction and the <c>out=</c> store.
        /// </summary>
        /// <param name="a">The array (any dtype, any layout).</param>
        /// <param name="axis">The reduced axes (an empty tuple reduces nothing).</param>
        /// <param name="out">NumPy's <c>out=</c>, or null for a fresh result.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="where">NumPy's <c>where=</c> bool mask, or null.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise the fresh bool array, read-only when 0-d.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="axis"/> is null.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>, a mask with more axes than
        ///     <paramref name="a"/>, or an axis that appears twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array, or <paramref name="out"/> does
        ///     not have the result's shape.</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast
        ///     against <paramref name="a"/>.</exception>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, P1b).</remarks>
        public abstract NDArray Any(NDArray a, int[] axis, NDArray @out, bool keepdims = false, NDArray where = null);
        public abstract bool AllClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false);
        public abstract NDArray<bool> IsClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false);
        // Predicate ufuncs — same single-member rule as the comparisons above
        // (bool loop, validate-only dtype, plain-NDArray return, plain
        // calls must return an NDArray<bool> instance).
        public abstract NDArray IsFinite(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray IsNan(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray IsInf(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);
        // np.isposinf / np.isneginf (numpy/lib/_ufunclike_impl.py). Not ufuncs — no
        // where/dtype; the single settable side is out (positional in NumPy). Complex
        // input is rejected at the np.* layer (NumPy's signbit raises), so the engine
        // predicate never sees Complex.
        public abstract NDArray IsPosInf(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray IsNegInf(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);
        // np.signbit — a full ufunc (unlike isposinf/isneginf): out=/where=/dtype= all honoured, same
        // single-member predicate rule as isnan/isinf (bool loop, validate-only dtype, plain-NDArray
        // return, a plain call returns an NDArray<bool> instance). Complex has no loop (rejected here).
        public abstract NDArray SignBit(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);

        #endregion

        #region Array Manipulation

        public abstract NDArray CreateNDArray(Shape shape, DType dtype = null, Array buffer = null, char order = 'C');
        public abstract NDArray CreateNDArray(Shape shape, DType dtype = null, IArraySlice buffer = null, char order = 'C');
        public abstract NDArray Transpose(NDArray nd, int[] premute = null);
        public abstract NDArray SwapAxes(NDArray nd, int axis1, int axis2);
        /// <summary>
        ///     NumPy's <c>np.moveaxis</c>: the axes of <paramref name="source"/> moved to the positions of
        ///     <paramref name="destination"/>, the other axes keeping their order (a view).
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="source">The axes to move (negative values count from the end).</param>
        /// <param name="destination">Their new positions.</param>
        /// <returns>A view with the permuted axes.</returns>
        /// <exception cref="AxisError">An axis is out of range ("source: …" / "destination: …").</exception>
        /// <exception cref="ValueError">An argument repeats an axis, or the two have different lengths.</exception>
        public abstract NDArray MoveAxis(NDArray nd, int[] source, int[] destination);
        public abstract NDArray RollAxis(NDArray nd, int axis, int start = 0);
        public abstract NDArray Cast(NDArray x, Type dtype, bool copy);

        #endregion

        #region Sorting, searching, counting

        #region Reduction

        public abstract NDArray ReduceAMax(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null);
        public abstract NDArray ReduceAMin(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null);
        public abstract NDArray ReduceArgMax(NDArray arr, int? axis_, bool keepdims = false);
        public abstract NDArray ReduceArgMin(NDArray arr, int? axis_, bool keepdims = false);
        public abstract NDArray ReduceProduct(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null);
        public abstract NDArray ReduceStd(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, DType dtype = null);
        public abstract NDArray ReduceVar(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, DType dtype = null);

        #endregion

        public abstract NDArray ArgMax(NDArray a);
        public abstract NDArray ArgMax(NDArray a, int axis, bool keepdims = false);

        public abstract NDArray ArgMin(NDArray a);
        public abstract NDArray ArgMin(NDArray a, int axis, bool keepdims = false);

        public abstract NDArray AMax(NDArray nd, int? axis = null, DType dtype = null, bool keepdims = false);

        public abstract NDArray AMin(NDArray nd, int? axis = null, DType dtype = null, bool keepdims = false);

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
