// The ufuncs of NumPy's _multiarray_umath (numpy/_core/src/umath): the element-wise arithmetic, comparison, logic,
// bitwise and transcendental operations, the predicates, the two-output ufuncs and the matmul family's gufuncs.
// Each member takes NumPy's ufunc keywords in the house order (inputs, dtype, out, where), and DefaultEngine
// overrides it. The reductions and scans behind np.sum, np.max, np.cumsum, ... implement ndarray methods and sit
// in TensorEngine.Methods.cs.

using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
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

        public abstract NDArray Negate(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     NumPy 'positive' — identity at every numeric dtype (no bool loop).
        ///     dtype selects the loop (positive(i32, dtype=f64) widens; a bool
        ///     loop request raises NumPy's did-not-contain-a-loop TypeError).
        /// </summary>
        public abstract NDArray Positive(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     NumPy 'conjugate' / 'conj' — the complex conjugate, element-wise. Identity at every
        ///     real dtype with a loop (values unchanged, dtype preserved); bool has none and resolves to
        ///     the int8 loop when no dtype is given (an int8 result of 0 / 1); for Complex it flips the
        ///     sign of the imaginary part. dtype selects the loop; out=/where= follow the ufunc contract.
        /// </summary>
        /// <param name="nd">The input.</param>
        /// <param name="dtype">The loop dtype, or null for the input's (int8 for bool).</param>
        /// <param name="out">An array to write the result into and return, or null.</param>
        /// <param name="where">A bool mask of the elements to compute, or null for all.</param>
        /// <returns>The conjugate, or <paramref name="out"/>.</returns>
        /// <exception cref="TypeError">A text input, or a <paramref name="dtype"/> / <paramref name="out"/> the loop
        ///     cannot reach (NumPy's ufunc texts).</exception>
        public abstract NDArray Conjugate(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);

        public abstract NDArray Matmul(NDArray lhs, NDArray rhs);

        /// <summary>
        ///     NumPy's <c>np.matmul(x1, x2, out, axes, axis, keepdims, dtype, casting, order)</c> — the gufunc
        ///     <c>(n?,k),(k,m?)-&gt;(n?,m?)</c> — the whole call: the argument checks (the order and casting names,
        ///     <c>axis</c> with <c>axes</c>, the <c>dtype=</c> class, the loop refusal of text, <c>keepdims</c> /
        ///     <c>axis</c>, a read-only <c>out=</c>), the core axes <c>axes=</c> names, the loop <c>dtype=</c> selects (each
        ///     input's cast checked under <paramref name="casting"/>), the product (<see cref="Matmul(NDArray, NDArray)"/>),
        ///     the result's core axes placed, the layout <paramref name="order"/> asks for, and the delivery into
        ///     <paramref name="out"/> (its cast and its core dimensions checked first).
        /// </summary>
        /// <param name="x1">The left operand, at least 1-D: a 1-D operand drops its optional core dimension <c>n?</c>.</param>
        /// <param name="x2">The right operand, at least 1-D: a 1-D operand drops <c>m?</c>.</param>
        /// <param name="out">Where the product is delivered, or null. Any layout; takes a cast of the product under
        ///     <paramref name="casting"/>; its extra leading loop axes receive copies of the product.</param>
        /// <param name="axes">The axes carrying each operand's core dimensions (<c>{x1, x2, out}</c>: two for an operand
        ///     of two or more dimensions, one for a 1-D one; all three entries are required), or null.</param>
        /// <param name="axis">Refused for any value (matmul's three core dimensions are distinct), or null.</param>
        /// <param name="keepdims">Refused for any value, false included (the output has core dimensions); null when the
        ///     caller did not pass it.</param>
        /// <param name="dtype">The loop dtype, or null (the product's own).</param>
        /// <param name="casting">The rule the input-to-loop and product-to-out casts are checked under: <c>no</c>,
        ///     <c>equiv</c>, <c>safe</c>, <c>same_kind</c> or <c>unsafe</c>.</param>
        /// <param name="order">The result's layout: <c>'C'</c>, <c>'F'</c>, <c>'A'</c> (F when both inputs are
        ///     F-contiguous, else C) or <c>'K'</c> (the product's own).</param>
        /// <returns>The product, or <paramref name="out"/> itself holding it.</returns>
        /// <exception cref="ValueError">An order or casting name NumPy refuses; an <c>axes=</c> list without three
        ///     entries, or with a repeated axis; a read-only <paramref name="out"/>; an <paramref name="out"/> whose core
        ///     dimension differs from the product's, or whose loop axes the product does not broadcast into.</exception>
        /// <exception cref="TypeError"><paramref name="axis"/> with <paramref name="axes"/>; a <paramref name="dtype"/>
        ///     with details beyond its class; text operands or a text <paramref name="dtype"/> (no matmul loop);
        ///     <paramref name="keepdims"/> or <paramref name="axis"/> given.</exception>
        /// <exception cref="AxisError">An <c>axes=</c> entry whose length is not its operand's core rank, or an axis out of
        ///     range.</exception>
        /// <exception cref="ArgumentException">A cast <paramref name="casting"/> refuses (NumPy's UFuncTypeError
        ///     text).</exception>
        /// <exception cref="IncorrectShapeException">Core dimensions that do not match, or loop axes that do not
        ///     broadcast (the product's texts).</exception>
        /// <exception cref="InvalidOperationException">A 0-d operand.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> is a class without storage yet.</exception>
        /// <exception cref="NullReferenceException"><paramref name="x2"/> is null.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, P2): the public <c>np.matmul</c> keeps the matrix boundary and
        ///     calls this; <see cref="Matmul(NDArray, NDArray)"/> stays the product kernel the call ends in.</remarks>
        public abstract NDArray Matmul(NDArray x1, NDArray x2, NDArray @out, int[][] axes, int? axis, bool? keepdims,
            DType dtype, string casting, char order);

        /// <summary>
        ///     NumPy's <c>np.vecdot(x1, x2, out, axes, axis, keepdims, dtype, casting, order)</c> — the gufunc <c>(n),(n)-&gt;()</c>,
        ///     <paramref name="x1"/> conjugated — the whole call: the argument checks, the core axis (<c>axes=</c> or
        ///     <c>axis=</c>, both operands' moved last), the core sizes, the loop <paramref name="dtype"/> selects, the
        ///     product (<see cref="Vecdot(NDArray, NDArray)"/>), the kept axis under <paramref name="keepdims"/>, and the
        ///     delivery into <paramref name="out"/>.
        /// </summary>
        /// <param name="x1">The first operand, at least 1-D; conjugated when complex.</param>
        /// <param name="x2">The second operand, at least 1-D.</param>
        /// <param name="out">Where the answer is delivered (its cast checked under <paramref name="casting"/>), or
        ///     null.</param>
        /// <param name="axes">The core axis of each operand (<c>{x1, x2}</c>, or <c>{x1, x2, out}</c> where the output
        ///     entry is empty, or holds the kept axis under <paramref name="keepdims"/>), or null.</param>
        /// <param name="axis">The core axis of both operands, or null (the last).</param>
        /// <param name="keepdims">Whether the contracted axis stays in the result with length 1.</param>
        /// <param name="dtype">The loop dtype, or null.</param>
        /// <param name="casting">The rule the input-to-loop and answer-to-out casts are checked under: <c>no</c>,
        ///     <c>equiv</c>, <c>safe</c>, <c>same_kind</c> (NumPy's default) or <c>unsafe</c>.</param>
        /// <param name="order">The layout of an allocated answer: <c>'K'</c> (NumPy's default: the loop axes in the
        ///     inputs' memory order), <c>'C'</c>, <c>'F'</c>, or <c>'A'</c> (F when both inputs are F-contiguous, else C);
        ///     ignored with <paramref name="out"/>. Lowercase letters are accepted.</param>
        /// <returns>The vector dot products, or <paramref name="out"/> itself holding them.</returns>
        /// <exception cref="TypeError"><paramref name="axis"/> with <paramref name="axes"/>; a <paramref name="dtype"/>
        ///     with details beyond its class; text operands or a text <paramref name="dtype"/> (no vecdot loop).</exception>
        /// <exception cref="ValueError">A 0-d operand; core axes of different lengths; an <c>axes=</c> list of the wrong
        ///     length; a read-only <paramref name="out"/>, or one the answer does not broadcast into.</exception>
        /// <exception cref="AxisError">An <c>axes=</c> entry of the wrong length, or an axis out of range.</exception>
        /// <exception cref="ArgumentException">An input cannot be cast to the loop, or the answer into
        ///     <paramref name="out"/>, under <paramref name="casting"/> (NumPy's UFuncTypeError text).</exception>
        /// <exception cref="ValueError">A <paramref name="casting"/> or <paramref name="order"/> name NumPy refuses
        ///     (<c>casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '…')</c>, <c>order must be one
        ///     of 'C', 'F', 'A', or 'K' (got '…')</c>), checked casting first.</exception>
        /// <exception cref="TypeError">A null <paramref name="casting"/> (<c>casting must be str, not
        ///     NoneType</c>).</exception>
        /// <exception cref="IncorrectShapeException">Loop axes that do not broadcast.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> is a class without storage yet.</exception>
        /// <exception cref="NullReferenceException"><paramref name="x2"/> is null.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, P2): the public <c>np.vecdot</c> keeps the matrix boundary and
        ///     calls this.</remarks>
        public abstract NDArray Vecdot(NDArray x1, NDArray x2, NDArray @out, int[][] axes, int? axis, bool keepdims,
            DType dtype, string casting, char order);

        /// <summary>
        ///     NumPy's <c>np.matvec(x1, x2, out, axes, axis, keepdims, dtype, casting, order)</c> — the gufunc <c>(m,n),(n)-&gt;(m)</c> —
        ///     the whole call: the argument checks, the core axes <c>axes=</c> names, the core sizes, the loop
        ///     <paramref name="dtype"/> selects, the product (<see cref="Matvec(NDArray, NDArray)"/>), the result's core axis
        ///     placed, and the delivery into <paramref name="out"/>.
        /// </summary>
        /// <param name="x1">The matrix operand, at least 2-D.</param>
        /// <param name="x2">The vector operand, at least 1-D; not conjugated.</param>
        /// <param name="out">Where the answer is delivered (its cast checked under <paramref name="casting"/>), or
        ///     null.</param>
        /// <param name="axes">The core axes per operand (<c>{(m,n), (n), (m)}</c>, all three entries), or null.</param>
        /// <param name="axis">Refused for any value (the two core dimensions are distinct), or null.</param>
        /// <param name="keepdims">Refused when true.</param>
        /// <param name="dtype">The loop dtype, or null.</param>
        /// <param name="casting">The rule the input-to-loop and answer-to-out casts are checked under: <c>no</c>,
        ///     <c>equiv</c>, <c>safe</c>, <c>same_kind</c> (NumPy's default) or <c>unsafe</c>.</param>
        /// <param name="order">The layout of an allocated answer: <c>'K'</c> (NumPy's default: the loop axes in the
        ///     inputs' memory order), <c>'C'</c>, <c>'F'</c>, or <c>'A'</c> (F when both inputs are F-contiguous, else C);
        ///     ignored with <paramref name="out"/>. Lowercase letters are accepted.</param>
        /// <returns>The matrix-vector products, or <paramref name="out"/> itself holding them.</returns>
        /// <exception cref="TypeError"><paramref name="axis"/> with <paramref name="axes"/>, or either given; a
        ///     <paramref name="dtype"/> with details beyond its class; text operands or a text <paramref name="dtype"/>;
        ///     <paramref name="keepdims"/>.</exception>
        /// <exception cref="ValueError">An operand with too few dimensions; core axes of different lengths; an
        ///     <c>axes=</c> list without three entries; a read-only <paramref name="out"/>, or one the answer does not
        ///     broadcast into.</exception>
        /// <exception cref="AxisError">An <c>axes=</c> entry of the wrong length, or an axis out of range.</exception>
        /// <exception cref="ArgumentException">An input cannot be cast to the loop, or the answer into
        ///     <paramref name="out"/>, under <paramref name="casting"/> (NumPy's UFuncTypeError text).</exception>
        /// <exception cref="ValueError">A <paramref name="casting"/> or <paramref name="order"/> name NumPy refuses
        ///     (<c>casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '…')</c>, <c>order must be one
        ///     of 'C', 'F', 'A', or 'K' (got '…')</c>), checked casting first.</exception>
        /// <exception cref="TypeError">A null <paramref name="casting"/> (<c>casting must be str, not
        ///     NoneType</c>).</exception>
        /// <exception cref="IncorrectShapeException">Loop axes that do not broadcast (the product's text).</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> is a class without storage yet.</exception>
        /// <exception cref="NullReferenceException"><paramref name="x2"/> is null.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, P2): the public <c>np.matvec</c> keeps the matrix boundary and
        ///     calls this.</remarks>
        public abstract NDArray Matvec(NDArray x1, NDArray x2, NDArray @out, int[][] axes, int? axis, bool keepdims,
            DType dtype, string casting, char order);

        /// <summary>
        ///     NumPy's <c>np.vecmat(x1, x2, out, axes, axis, keepdims, dtype, casting, order)</c> — the gufunc <c>(n),(n,m)-&gt;(m)</c>,
        ///     <paramref name="x1"/> conjugated — the whole call: the argument checks, the core axes <c>axes=</c> names,
        ///     the core sizes, the loop <paramref name="dtype"/> selects, the product
        ///     (<see cref="Vecmat(NDArray, NDArray)"/>), the result's core axis placed, and the delivery into
        ///     <paramref name="out"/>.
        /// </summary>
        /// <param name="x1">The vector operand, at least 1-D; conjugated when complex.</param>
        /// <param name="x2">The matrix operand, at least 2-D.</param>
        /// <param name="out">Where the answer is delivered (its cast checked under <paramref name="casting"/>), or
        ///     null.</param>
        /// <param name="axes">The core axes per operand (<c>{(n), (n,m), (m)}</c>, all three entries), or null.</param>
        /// <param name="axis">Refused for any value (the two core dimensions are distinct), or null.</param>
        /// <param name="keepdims">Refused when true.</param>
        /// <param name="dtype">The loop dtype, or null.</param>
        /// <param name="casting">The rule the input-to-loop and answer-to-out casts are checked under: <c>no</c>,
        ///     <c>equiv</c>, <c>safe</c>, <c>same_kind</c> (NumPy's default) or <c>unsafe</c>.</param>
        /// <param name="order">The layout of an allocated answer: <c>'K'</c> (NumPy's default: the loop axes in the
        ///     inputs' memory order), <c>'C'</c>, <c>'F'</c>, or <c>'A'</c> (F when both inputs are F-contiguous, else C);
        ///     ignored with <paramref name="out"/>. Lowercase letters are accepted.</param>
        /// <returns>The vector-matrix products, or <paramref name="out"/> itself holding them.</returns>
        /// <exception cref="TypeError"><paramref name="axis"/> with <paramref name="axes"/>, or either given; a
        ///     <paramref name="dtype"/> with details beyond its class; text operands or a text <paramref name="dtype"/>;
        ///     <paramref name="keepdims"/>.</exception>
        /// <exception cref="ValueError">An operand with too few dimensions; core axes of different lengths; an
        ///     <c>axes=</c> list without three entries; a read-only <paramref name="out"/>, or one the answer does not
        ///     broadcast into.</exception>
        /// <exception cref="AxisError">An <c>axes=</c> entry of the wrong length, or an axis out of range.</exception>
        /// <exception cref="ArgumentException">An input cannot be cast to the loop, or the answer into
        ///     <paramref name="out"/>, under <paramref name="casting"/> (NumPy's UFuncTypeError text).</exception>
        /// <exception cref="ValueError">A <paramref name="casting"/> or <paramref name="order"/> name NumPy refuses
        ///     (<c>casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '…')</c>, <c>order must be one
        ///     of 'C', 'F', 'A', or 'K' (got '…')</c>), checked casting first.</exception>
        /// <exception cref="TypeError">A null <paramref name="casting"/> (<c>casting must be str, not
        ///     NoneType</c>).</exception>
        /// <exception cref="IncorrectShapeException">Loop axes that do not broadcast (the product's text).</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> is a class without storage yet.</exception>
        /// <exception cref="NullReferenceException"><paramref name="x2"/> is null.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, P2): the public <c>np.vecmat</c> keeps the matrix boundary and
        ///     calls this.</remarks>
        public abstract NDArray Vecmat(NDArray x1, NDArray x2, NDArray @out, int[][] axes, int? axis, bool keepdims,
            DType dtype, string casting, char order);

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

        public abstract NDArray Truncate(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Rint(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Reciprocal(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Square(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Deg2Rad(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Rad2Deg(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     NumPy's <c>np.radians</c> ufunc: the same conversion as <see cref="Deg2Rad"/> (degrees to radians), but a ufunc of its
        ///     own, so a refusal (text, an operand type with no loop) names <c>radians</c>.
        /// </summary>
        /// <param name="nd">The angles.</param>
        /// <param name="dtype">The loop dtype (<c>dtype=</c>), or null for the operand's float tier.</param>
        /// <param name="out">The array to write into (<c>out=</c>), or null.</param>
        /// <param name="where">The bool mask of the positions to compute (<c>where=</c>), or null for all.</param>
        /// <returns>The converted angles: <paramref name="out"/> when given, else a new array of the float tier (float16
        ///     for bool / int8 / uint8, float32 for int16 / uint16, float64 for wider integers; floats keep theirs).</returns>
        /// <exception cref="TypeError">A text or complex operand (no radians loop), with NumPy's texts.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>, or a <paramref name="dtype"/> the
        ///     operand cannot reach.</exception>
        public abstract NDArray Radians(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     NumPy's <c>np.degrees</c> ufunc: the same conversion as <see cref="Rad2Deg"/> (radians to degrees), but a ufunc of its
        ///     own, so a refusal (text, an operand type with no loop) names <c>degrees</c>.
        /// </summary>
        /// <param name="nd">The angles.</param>
        /// <param name="dtype">The loop dtype (<c>dtype=</c>), or null for the operand's float tier.</param>
        /// <param name="out">The array to write into (<c>out=</c>), or null.</param>
        /// <param name="where">The bool mask of the positions to compute (<c>where=</c>), or null for all.</param>
        /// <returns>The converted angles: <paramref name="out"/> when given, else a new array of the float tier (float16
        ///     for bool / int8 / uint8, float32 for int16 / uint16, float64 for wider integers; floats keep theirs).</returns>
        /// <exception cref="TypeError">A text or complex operand (no degrees loop), with NumPy's texts.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>, or a <paramref name="dtype"/> the
        ///     operand cannot reach.</exception>
        public abstract NDArray Degrees(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
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

        /// <summary>
        ///     NumPy's <c>logical_and</c> ufunc, the whole call: the element-wise AND of the operands' truth values, NumPy's
        ///     <c>logical_and(x1, x2, /, out=None, *, where=True, dtype=None)</c>. A text operand reads true when
        ///     non-empty (NumPy's string loop computes over the operands' truth arrays).
        /// </summary>
        /// <param name="lhs">The first operand (any dtype, text included; any layout).</param>
        /// <param name="rhs">The second operand; broadcasts against <paramref name="lhs"/>.</param>
        /// <param name="dtype">The <c>dtype=</c> loop request: the logical loops are bool-output only, so it validates and
        ///     selects nothing; a non-bool request raises NumPy's no-loop error.</param>
        /// <param name="out">A location into which the result is stored (any numeric dtype: bool casts same_kind to all,
        ///     True to 1), or null; returned itself when given.</param>
        /// <param name="where">A bool mask: only its true elements are computed; masked-off slots of <paramref name="out"/>
        ///     keep their contents (of a fresh result they are unspecified, as in NumPy).</param>
        /// <returns>A bool array of the broadcast shape, or <paramref name="out"/>.</returns>
        /// <exception cref="IncorrectTypeException">A non-bool <paramref name="dtype"/> (<c>No loop matching the specified
        ///     signature and casting was found for ufunc logical_and</c>).</exception>
        /// <exception cref="TypeError">A detailed or non-legacy <paramref name="dtype"/> descriptor, or a text call NumPy's
        ///     string loop refuses.</exception>
        /// <exception cref="SystemError">A <c>U</c> operand holding a unit above U+10FFFF (NumPy's bool cast of the
        ///     text).</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="ArgumentException">A <paramref name="where"/> that is not bool, an <paramref name="out"/> the
        ///     bool result cannot be cast into, or a mask or <paramref name="out"/> that does not broadcast with the
        ///     operands (NumPy's ValueError text).</exception>
        /// <exception cref="IncorrectShapeException">Operands that do not broadcast together.</exception>
        /// <remarks>NumPy's loop is C (<c>@TYPE@_logical_and</c>), so the call is an engine member, as every ufunc's is; the
        ///     numpy.matrix and numpy.char.chararray boundaries stay with <c>np.logical_and</c>.</remarks>
        public abstract NDArray LogicalAnd(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     NumPy's <c>logical_or</c> ufunc, the whole call: the element-wise OR of the operands' truth values, NumPy's
        ///     <c>logical_or(x1, x2, /, out=None, *, where=True, dtype=None)</c>. A text operand reads true when non-empty.
        /// </summary>
        /// <param name="lhs">The first operand (any dtype, text included; any layout).</param>
        /// <param name="rhs">The second operand; broadcasts against <paramref name="lhs"/>.</param>
        /// <param name="dtype">The <c>dtype=</c> loop request: validate-only (bool loops); a non-bool request raises NumPy's
        ///     no-loop error.</param>
        /// <param name="out">A location into which the result is stored (any numeric dtype), or null; returned itself when
        ///     given.</param>
        /// <param name="where">A bool mask: only its true elements are computed; masked-off slots of <paramref name="out"/>
        ///     keep their contents.</param>
        /// <returns>A bool array of the broadcast shape, or <paramref name="out"/>.</returns>
        /// <exception cref="IncorrectTypeException">A non-bool <paramref name="dtype"/> (<c>No loop matching the specified
        ///     signature and casting was found for ufunc logical_or</c>).</exception>
        /// <exception cref="TypeError">A detailed or non-legacy <paramref name="dtype"/> descriptor, or a text call NumPy's
        ///     string loop refuses.</exception>
        /// <exception cref="SystemError">A <c>U</c> operand holding a unit above U+10FFFF.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="ArgumentException">A <paramref name="where"/> that is not bool, an <paramref name="out"/> the
        ///     bool result cannot be cast into, or a mask or <paramref name="out"/> that does not broadcast with the
        ///     operands (NumPy's ValueError text).</exception>
        /// <exception cref="IncorrectShapeException">Operands that do not broadcast together.</exception>
        public abstract NDArray LogicalOr(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     NumPy's <c>logical_xor</c> ufunc, the whole call: the element-wise XOR of the operands' truth values, NumPy's
        ///     <c>logical_xor(x1, x2, /, out=None, *, where=True, dtype=None)</c>. A text operand reads true when non-empty.
        /// </summary>
        /// <param name="lhs">The first operand (any dtype, text included; any layout).</param>
        /// <param name="rhs">The second operand; broadcasts against <paramref name="lhs"/>.</param>
        /// <param name="dtype">The <c>dtype=</c> loop request: validate-only (bool loops); a non-bool request raises NumPy's
        ///     no-loop error.</param>
        /// <param name="out">A location into which the result is stored (any numeric dtype), or null; returned itself when
        ///     given.</param>
        /// <param name="where">A bool mask: only its true elements are computed; masked-off slots of <paramref name="out"/>
        ///     keep their contents.</param>
        /// <returns>A bool array of the broadcast shape, or <paramref name="out"/>.</returns>
        /// <exception cref="IncorrectTypeException">A non-bool <paramref name="dtype"/> (<c>No loop matching the specified
        ///     signature and casting was found for ufunc logical_xor</c>).</exception>
        /// <exception cref="TypeError">A detailed or non-legacy <paramref name="dtype"/> descriptor, or a text call NumPy's
        ///     string loop refuses.</exception>
        /// <exception cref="SystemError">A <c>U</c> operand holding a unit above U+10FFFF.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="ArgumentException">A <paramref name="where"/> that is not bool, an <paramref name="out"/> the
        ///     bool result cannot be cast into, or a mask or <paramref name="out"/> that does not broadcast with the
        ///     operands (NumPy's ValueError text).</exception>
        /// <exception cref="IncorrectShapeException">Operands that do not broadcast together.</exception>
        public abstract NDArray LogicalXor(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     NumPy's <c>logical_not</c> ufunc, the whole call: the element-wise NOT of the operand's truth value (zero is
        ///     True, nonzero False; a NaN is truthy), NumPy's <c>logical_not(x, /, out=None, *, where=True, dtype=None)</c>.
        ///     Unlike the binary three it has no string loop in NumPy, so a text operand is refused.
        /// </summary>
        /// <param name="nd">The operand (any numeric dtype; any layout).</param>
        /// <param name="dtype">The <c>dtype=</c> loop request: validate-only (bool loops); a non-bool request raises NumPy's
        ///     no-loop error.</param>
        /// <param name="out">A location into which the result is stored (any numeric dtype), or null; returned itself when
        ///     given.</param>
        /// <param name="where">A bool mask: only its true elements are computed; masked-off slots of <paramref name="out"/>
        ///     keep their contents.</param>
        /// <returns>A bool array of the operand's shape (broadcast with the mask and <paramref name="out"/>), or
        ///     <paramref name="out"/>.</returns>
        /// <exception cref="IncorrectTypeException">A non-bool <paramref name="dtype"/> (<c>No loop matching the specified
        ///     signature and casting was found for ufunc logical_not</c>), for a text operand too: the loops are bool-output
        ///     only.</exception>
        /// <exception cref="TypeError">A text operand without <paramref name="dtype"/> (NumPy's no-loop refusal; a
        ///     StringDType operand's text names the requested output class), or a detailed or non-legacy
        ///     <paramref name="dtype"/> descriptor.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="ArgumentException">A <paramref name="where"/> that is not bool, an <paramref name="out"/> the
        ///     bool result cannot be cast into, a fixed-width text operand under <c>dtype=bool</c> (the input cast NumPy
        ///     refuses, its UFuncTypeError text), or a mask or <paramref name="out"/> that does not broadcast with the
        ///     operand (NumPy's ValueError text).</exception>
        public abstract NDArray LogicalNot(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);

        // Bit shift operations (integer types only). dtype (ufunc dtype=) selects the loop among the
        // bool/integer loops; a float/complex/decimal request raises NumPy's no-loop TypeError (shifts
        // have no such loops). out/where follow the standard ufunc contract (same_kind cast into out,
        // masked write). A uint64×signed pair promotes to float64 (no loop) and raises "not supported".
        public abstract NDArray LeftShift(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray RightShift(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null);

        // Predicate ufuncs — same single-member rule as the comparisons above
        // (bool loop, validate-only dtype, plain-NDArray return, plain
        // calls must return an NDArray<bool> instance).
        public abstract NDArray IsFinite(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray IsNan(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray IsInf(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);

        // np.signbit — a full ufunc (unlike isposinf/isneginf): out=/where=/dtype= all honoured, same
        // single-member predicate rule as isnan/isinf (bool loop, validate-only dtype, plain-NDArray
        // return, a plain call returns an NDArray<bool> instance). Complex has no loop (rejected here).
        public abstract NDArray SignBit(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);
    }
}
