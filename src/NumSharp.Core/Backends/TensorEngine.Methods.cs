// The ndarray methods that compute, which numpy/_core/fromnumeric.py wraps as np.sum, np.prod, np.mean, np.std,
// np.var, np.max, np.min, np.argmax, np.argmin, np.all, np.any, np.cumsum, np.cumprod, np.searchsorted,
// np.nonzero, np.trace, np.clip, np.round, np.sort, np.argsort, np.partition and np.argpartition (NumPy implements
// them in _core/_methods.py and in C over the ufunc reductions and the sort functions). The ndarray methods of the
// array object (reshape, take, put, ...) are not engine members.

using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        public abstract NDArray ReduceAdd(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null, NDArray @out = null);
        /// <summary>
        ///     NumPy's <c>np.cumsum</c> (<c>np.add.accumulate</c>), the whole call: the running sums along
        ///     <paramref name="axis_"/> (the flattened array when null), stored into <paramref name="out"/> when given.
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="axis_">The accumulated axis (negative counts from the end), or null for the flattened array.</param>
        /// <param name="dtype">The loop / result dtype (<c>dtype=</c>), or null: then the loop is
        ///     <c>promote_types(out.dtype, arr.dtype)</c> when <paramref name="out"/> is given (no NEP 50 widening), NEP 50's
        ///     accumulator of the input otherwise.</param>
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
        /// <param name="dtype">The loop / result dtype (<c>dtype=</c>), or null: then the loop is
        ///     <c>promote_types(out.dtype, arr.dtype)</c> when <paramref name="out"/> is given (no NEP 50 widening), NEP 50's
        ///     accumulator of the input otherwise.</param>
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

        /// <summary>
        ///     NumPy's <c>np.mean</c> (<c>_methods._mean</c>), the whole call: the add reduction divided by the count of the
        ///     reduced elements, stored into <paramref name="out"/> when given.
        /// </summary>
        /// <param name="arr">The array to average (any dtype, any layout).</param>
        /// <param name="axis_">The reduced axis (negative counts from the end), or null for every axis. Unlike a ufunc
        ///     reduction, a 0-d array takes no axis.</param>
        /// <param name="keepdims">Whether the reduced axes stay in the result with length 1.</param>
        /// <param name="dtype">The loop of the sum (<c>dtype=</c>), or null: float64 for a bool or integer input, float32 for
        ///     float16 (with a float16 result), the input's own dtype otherwise.</param>
        /// <param name="out">NumPy's <c>out=</c>, an array of the result's shape, or null for a fresh result. NumPy stores the
        ///     SUM into it (in <c>promote_types(out, arr)</c> when <paramref name="dtype"/> is null and the input is not a bool,
        ///     integer or float16 one) and then divides it in place, both with an unsafe cast: an integer out holds truncated
        ///     values.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise a fresh array, NumPy's read-only scalar for a full
        ///     reduction.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range (every axis is, for a 0-d input).</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape (NumPy's
        ///     reduce prologue texts), or text a numeric loop must parse does not parse.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not broadcast against the input.</exception>
        /// <exception cref="TypeError">A text operand, out or request NumPy's loop resolution refuses.</exception>
        /// <remarks>
        ///     The engine owns NumPy's whole call, dtype decisions included (the np-layer dtype plan). <paramref name="out"/>
        ///     was added to the contract in 0.71.0 (P1d, decision D2: a breaking change for a subclass); numpy.matrix's
        ///     <c>mean(out=)</c> runs through it.
        /// </remarks>
        public abstract NDArray ReduceMean(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null, NDArray @out = null);

        public abstract NDArray Mean(NDArray nd, int? axis = null, DType dtype = null, bool keepdims = false);

        public abstract NDArray Sum(NDArray nd, int? axis = null, DType dtype = null, bool keepdims = false);

        public abstract NDArray Round(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null);
        public abstract NDArray Round(NDArray nd, int decimals, DType dtype = null, NDArray @out = null);

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

        /// <summary>
        ///     NumPy's <c>np.max</c> / <c>np.amax</c> (<c>np.maximum.reduce</c>) over <paramref name="axis_"/>, or over every
        ///     axis when it is null, with <c>ufunc.reduce</c>'s <c>dtype=</c> and NumPy's <c>out=</c>, <c>initial=</c> and
        ///     <c>where=</c>: the whole call of <c>np.max</c> and <c>ndarray.max</c>.
        /// </summary>
        /// <param name="arr">The array (any dtype; <c>StringDType</c> computes, fixed-width text is refused).</param>
        /// <param name="axis_">The reduced axis (negative counts from the end), or null for every axis. A 0-d array accepts
        ///     0 and -1 and nothing else.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="dtype">The loop dtype (<c>dtype=</c>, NumPy's <c>np.maximum.reduce</c> keyword), or null for the
        ///     input's. <c>np.max</c> itself has none.</param>
        /// <param name="out">NumPy's <c>out=</c>: the array that receives the result (any dtype: the loop is
        ///     <c>promote_types(out, arr)</c> and its result is cast in), or null for a fresh result.</param>
        /// <param name="initial">NumPy's <c>initial=</c>: the value every output starts from, packed into the loop dtype as
        ///     NumPy's <c>PyArray_Pack</c> does (a C# integer or float is a Python int or float, range-checked, never
        ///     wrapped), or null for none.</param>
        /// <param name="where">NumPy's <c>where=</c>: a bool mask broadcast against <paramref name="arr"/>, only the selected
        ///     elements reduce; it needs <paramref name="initial"/>. Null, or a 0-d true mask, is no mask.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise a fresh array, a read-only 0-d one (NumPy's scalar)
        ///     for a full reduction without keepdims.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range.</exception>
        /// <exception cref="ArgumentException">An empty reduction without <paramref name="initial"/> (<c>zero-size array to
        ///     reduction operation maximum which has no identity</c>, NumPy's ValueError); a non-bool mask (NumPy's TypeError
        ///     text).</exception>
        /// <exception cref="TypeError">Fixed-width text, a string <paramref name="dtype"/>, or a complex
        ///     <paramref name="initial"/> for a real loop.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> together with <paramref name="out"/>,
        ///     <paramref name="initial"/> or <paramref name="where"/> (NumPy's <c>ufunc.reduce</c> takes them together;
        ///     NumSharp's keyword path computes in <c>promote_types(out, arr)</c> only).</exception>
        /// <exception cref="ValueError">With a keyword: a read-only or misshapen <paramref name="out"/>, a mask with more axes
        ///     than <paramref name="arr"/>, a mask without <paramref name="initial"/> or an empty reduced axis without it
        ///     (NumPy's no-identity texts), an <paramref name="initial"/> the loop cannot hold (a NaN into an integer loop, a
        ///     sequence).</exception>
        /// <exception cref="OverflowException">An integer <paramref name="initial"/> past the loop's range.</exception>
        /// <exception cref="IncorrectShapeException">The mask or <paramref name="out"/> does not broadcast.</exception>
        /// <remarks>Widened with <paramref name="out"/>, <paramref name="initial"/> and <paramref name="where"/> in 0.71.0
        ///     (the np-layer dtype plan, phase PE, E5: D2): the keyword path was the static
        ///     <c>DefaultEngine.MaxMinNumPy</c>, which the public functions called past the engine.</remarks>
        public abstract NDArray ReduceAMax(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null,
            NDArray @out = null, object initial = null, NDArray @where = null);

        /// <summary>
        ///     NumPy's <c>np.min</c> / <c>np.amin</c> (<c>np.minimum.reduce</c>) over <paramref name="axis_"/>, or over every
        ///     axis when it is null, with <c>ufunc.reduce</c>'s <c>dtype=</c> and NumPy's <c>out=</c>, <c>initial=</c> and
        ///     <c>where=</c>: the whole call of <c>np.min</c> and <c>ndarray.min</c>.
        /// </summary>
        /// <param name="arr">The array (any dtype; <c>StringDType</c> computes, fixed-width text is refused).</param>
        /// <param name="axis_">The reduced axis (negative counts from the end), or null for every axis. A 0-d array accepts
        ///     0 and -1 and nothing else.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="dtype">The loop dtype (<c>dtype=</c>, NumPy's <c>np.minimum.reduce</c> keyword), or null for the
        ///     input's. <c>np.min</c> itself has none.</param>
        /// <param name="out">NumPy's <c>out=</c>: the array that receives the result (any dtype: the loop is
        ///     <c>promote_types(out, arr)</c> and its result is cast in), or null for a fresh result.</param>
        /// <param name="initial">NumPy's <c>initial=</c>: the value every output starts from, packed into the loop dtype as
        ///     NumPy's <c>PyArray_Pack</c> does (a C# integer or float is a Python int or float, range-checked, never
        ///     wrapped), or null for none.</param>
        /// <param name="where">NumPy's <c>where=</c>: a bool mask broadcast against <paramref name="arr"/>, only the selected
        ///     elements reduce; it needs <paramref name="initial"/>. Null, or a 0-d true mask, is no mask.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise a fresh array, a read-only 0-d one (NumPy's scalar)
        ///     for a full reduction without keepdims.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range.</exception>
        /// <exception cref="ArgumentException">An empty reduction without <paramref name="initial"/> (<c>zero-size array to
        ///     reduction operation minimum which has no identity</c>, NumPy's ValueError); a non-bool mask (NumPy's TypeError
        ///     text).</exception>
        /// <exception cref="TypeError">Fixed-width text, a string <paramref name="dtype"/>, or a complex
        ///     <paramref name="initial"/> for a real loop.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> together with <paramref name="out"/>,
        ///     <paramref name="initial"/> or <paramref name="where"/> (NumPy's <c>ufunc.reduce</c> takes them together;
        ///     NumSharp's keyword path computes in <c>promote_types(out, arr)</c> only).</exception>
        /// <exception cref="ValueError">With a keyword: a read-only or misshapen <paramref name="out"/>, a mask with more axes
        ///     than <paramref name="arr"/>, a mask without <paramref name="initial"/> or an empty reduced axis without it
        ///     (NumPy's no-identity texts), an <paramref name="initial"/> the loop cannot hold (a NaN into an integer loop, a
        ///     sequence).</exception>
        /// <exception cref="OverflowException">An integer <paramref name="initial"/> past the loop's range.</exception>
        /// <exception cref="IncorrectShapeException">The mask or <paramref name="out"/> does not broadcast.</exception>
        /// <remarks>Widened with <paramref name="out"/>, <paramref name="initial"/> and <paramref name="where"/> in 0.71.0
        ///     (the np-layer dtype plan, phase PE, E5: D2): the keyword path was the static
        ///     <c>DefaultEngine.MaxMinNumPy</c>, which the public functions called past the engine.</remarks>
        public abstract NDArray ReduceAMin(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null,
            NDArray @out = null, object initial = null, NDArray @where = null);
        public abstract NDArray ReduceArgMax(NDArray arr, int? axis_, bool keepdims = false);
        public abstract NDArray ReduceArgMin(NDArray arr, int? axis_, bool keepdims = false);
        /// <summary>
        ///     NumPy's <c>np.prod</c> (<c>np.multiply.reduce</c>), the whole call: the product over <paramref name="axis_"/>
        ///     (every axis when null), stored into <paramref name="out"/> when given.
        /// </summary>
        /// <param name="arr">The array (any dtype, any layout).</param>
        /// <param name="axis_">The reduced axis (negative counts from the end), or null for every axis. A 0-d input accepts
        ///     0 and -1 (ufunc.reduce's backward-compatibility rule).</param>
        /// <param name="keepdims">Whether the reduced axes stay in the result with length 1.</param>
        /// <param name="dtype">The loop dtype (<c>dtype=</c>), or null: NEP 50's accumulator of the input without
        ///     <paramref name="out"/>, <c>promote_types(out, arr)</c> with it (no small-integer widening then).</param>
        /// <param name="out">NumPy's <c>out=</c>, an array of the result's shape that receives it with unsafe casting, or
        ///     null for a fresh result.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise a fresh array, NumPy's read-only scalar for a full
        ///     reduction.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape (NumPy's
        ///     reduce prologue texts), or text a numeric loop must parse does not parse.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not broadcast against the input.</exception>
        /// <exception cref="TypeError">Text, or a string request: NumPy's refusal (no multiply loop reduces text into itself;
        ///     integer counts into a text out repeat).</exception>
        /// <remarks>
        ///     <paramref name="out"/> was added to the contract in 0.71.0 (the np-layer dtype plan's P1d, decision D2: a
        ///     breaking change for a subclass); numpy.matrix's <c>prod(out=)</c> runs through it.
        /// </remarks>
        public abstract NDArray ReduceProduct(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.std</c> (<c>_methods._std</c>), the whole call: the square root of <see cref="ReduceVar"/>, stored
        ///     into <paramref name="out"/> when given.
        /// </summary>
        /// <param name="arr">The operand (any dtype, any layout).</param>
        /// <param name="axis_">The reduced axis (negative counts from the end), or null for every axis.</param>
        /// <param name="keepdims">Whether the reduced axes stay in the result with length 1.</param>
        /// <param name="ddof">Delta degrees of freedom (null is 0): the divisor is <c>max(n - ddof, 0)</c>.</param>
        /// <param name="dtype">The loop of both sums (an integer or bool input defaults to float64, every other input to its
        ///     own dtype).</param>
        /// <param name="out">NumPy's <c>out=</c>: a FLOAT or complex array of the result's shape, or null. NumPy hands it to
        ///     the variance's last sum (whose loop it joins), divides it in place and takes the square root in place.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise a fresh array, a read-only 0-d scalar when the result
        ///     has no dimensions.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is outside the operand (a 0-d operand included).</exception>
        /// <exception cref="TypeError">A bool input with <c>dtype=bool</c>, or text NumPy's loop resolution refuses.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape.</exception>
        /// <exception cref="ArgumentException">An integer or bool <paramref name="dtype"/> (with a result that has dimensions)
        ///     or <paramref name="out"/>: NumPy's <c>sqrt</c> output-cast error under <c>same_kind</c>.</exception>
        /// <remarks>
        ///     <paramref name="out"/> was added to the contract in 0.71.0 (the np-layer dtype plan's P1d, decision D2);
        ///     numpy.matrix's <c>std(out=)</c> runs through it.
        /// </remarks>
        public abstract NDArray ReduceStd(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, DType dtype = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.var</c> (<c>_methods._var</c>), the whole call: the mean of the squared deviations from the mean,
        ///     divided by <c>max(n - ddof, 0)</c>, stored into <paramref name="out"/> when given.
        /// </summary>
        /// <param name="arr">The operand (any dtype, any layout).</param>
        /// <param name="axis_">The reduced axis (negative counts from the end), or null for every axis.</param>
        /// <param name="keepdims">Whether the reduced axes stay in the result with length 1.</param>
        /// <param name="ddof">Delta degrees of freedom (null is 0).</param>
        /// <param name="dtype">The loop of both sums (an integer or bool input defaults to float64, every other input to its
        ///     own dtype); it decides the result dtype.</param>
        /// <param name="out">NumPy's <c>out=</c>, an array of the result's shape, or null. NumPy hands it to its LAST sum only
        ///     (the squared deviations', whose loop is <c>promote_types(out, x)</c> without <paramref name="dtype"/>), then
        ///     divides it in place.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise a fresh array, a read-only 0-d scalar when the result
        ///     has no dimensions.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is outside the operand (a 0-d operand included).</exception>
        /// <exception cref="TypeError">A bool input with <c>dtype=bool</c>, a text out for a float or complex input, or text
        ///     NumPy's loop resolution refuses.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or does not have the result's shape.</exception>
        /// <remarks>
        ///     <paramref name="out"/> was added to the contract in 0.71.0 (the np-layer dtype plan's P1d, decision D2);
        ///     numpy.matrix's <c>var(out=)</c> runs through it.
        /// </remarks>
        public abstract NDArray ReduceVar(NDArray arr, int? axis_, bool keepdims = false, int? ddof = null, DType dtype = null, NDArray @out = null);

        public abstract NDArray ArgMax(NDArray a);
        public abstract NDArray ArgMax(NDArray a, int axis, bool keepdims = false);

        public abstract NDArray ArgMin(NDArray a);
        public abstract NDArray ArgMin(NDArray a, int axis, bool keepdims = false);

        public abstract NDArray AMax(NDArray nd, int? axis = null, DType dtype = null, bool keepdims = false);

        public abstract NDArray AMin(NDArray nd, int? axis = null, DType dtype = null, bool keepdims = false);

        /// <summary>
        ///     NumPy's <c>np.searchsorted(a, v, side, sorter)</c> (<c>PyArray_SearchSorted</c>) for a Python int key: the
        ///     index at which <paramref name="v"/> would be inserted into the sorted 1-D <paramref name="a"/> to keep it
        ///     sorted, the whole call (NumPy's argument checks in its order, then the search in the operands' common dtype).
        /// </summary>
        /// <param name="a">The haystack: 1-D, sorted ascending unless <paramref name="sorter"/> is given (an unsorted one
        ///     gets NumPy's bisection's answer, not an error). Any dtype, text included; any layout.</param>
        /// <param name="v">The key, read as NumPy's Python int (a strong int32 descriptor: the search runs in
        ///     <c>promote_types(a, int32)</c>, never casting the key down to <paramref name="a"/>'s dtype).</param>
        /// <param name="side"><c>"left"</c> (the first suitable index) or <c>"right"</c> (the last).</param>
        /// <param name="sorter">Indices that sort <paramref name="a"/> (typically its argsort), or null: 1-D, integer, of
        ///     <paramref name="a"/>'s size.</param>
        /// <returns>The insertion index (0 to <c>a.size</c>).</returns>
        /// <exception cref="ArgumentException"><paramref name="side"/> is not <c>"left"</c> / <c>"right"</c> (NumPy's two
        ///     ValueError texts).</exception>
        /// <exception cref="ValueError"><paramref name="a"/> is not 1-D, a sorter of uint64, a sorter whose size differs
        ///     from <paramref name="a"/>'s, or a sorter index outside <paramref name="a"/> that the search reaches.</exception>
        /// <exception cref="TypeError">A sorter that is not 1-D (<c>could not parse sorter argument</c>) or not integer
        ///     (<c>sorter must only contain integers</c>), or a StringDType haystack whose first comparison with the int
        ///     Python refuses (<c>'&lt;' not supported between instances of 'str' and 'int'</c>).</exception>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, P2): the public <c>np.searchsorted</c> opens the matrix boundary's
        ///     plain region and calls this. A member of its own, not the array form over a 0-d key: the key stays on the
        ///     stack, about 15x cheaper per call.</remarks>
        public abstract long SearchSorted(NDArray a, int v, string side = "left", NDArray sorter = null);

        /// <summary>
        ///     NumPy's <c>np.searchsorted(a, v, side, sorter)</c> (<c>PyArray_SearchSorted</c>) for a Python float key: the
        ///     index at which <paramref name="v"/> would be inserted into the sorted 1-D <paramref name="a"/> to keep it
        ///     sorted, the whole call.
        /// </summary>
        /// <param name="a">The haystack: 1-D, sorted ascending unless <paramref name="sorter"/> is given. Any dtype, text
        ///     included; any layout.</param>
        /// <param name="v">The key, read as NumPy's Python float (the search runs in <c>promote_types(a, float64)</c>: a
        ///     float32 haystack is searched in float64, as NumPy does).</param>
        /// <param name="side"><c>"left"</c> (the first suitable index) or <c>"right"</c> (the last).</param>
        /// <param name="sorter">Indices that sort <paramref name="a"/>, or null: 1-D, integer, of <paramref name="a"/>'s
        ///     size.</param>
        /// <returns>The insertion index (0 to <c>a.size</c>).</returns>
        /// <exception cref="ArgumentException"><paramref name="side"/> is not <c>"left"</c> / <c>"right"</c>.</exception>
        /// <exception cref="ValueError">See <see cref="SearchSorted(NDArray, int, string, NDArray)"/>.</exception>
        /// <exception cref="TypeError">A sorter refusal (see the int member), or a StringDType haystack whose first
        ///     comparison with the float Python refuses (<c>'&lt;' not supported between instances of 'str' and
        ///     'float'</c>).</exception>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, P2): the public <c>np.searchsorted</c> opens the matrix boundary's
        ///     plain region and calls this.</remarks>
        public abstract long SearchSorted(NDArray a, double v, string side = "left", NDArray sorter = null);

        /// <summary>
        ///     NumPy's <c>np.searchsorted(a, v, side, sorter)</c> (<c>PyArray_SearchSorted</c>): the indices at which the
        ///     elements of <paramref name="v"/> would be inserted into the sorted 1-D <paramref name="a"/> to keep it sorted,
        ///     the whole call — the descriptor both operands are converted to (numbers: their common dtype; text: the common
        ///     string dtype, a number formatted; StringDType without a common descriptor: Python objects), NumPy's checks in
        ///     its order, and its binary search, its monotonic carry and its sorter checks included.
        /// </summary>
        /// <param name="a">The haystack: 1-D, sorted ascending unless <paramref name="sorter"/> is given. Any dtype, text
        ///     included; any layout.</param>
        /// <param name="v">The keys: any dtype, shape and layout (a 0-d array for one key).</param>
        /// <param name="side"><c>"left"</c> (the first suitable index) or <c>"right"</c> (the last).</param>
        /// <param name="sorter">Indices that sort <paramref name="a"/>, or null: 1-D, integer, of <paramref name="a"/>'s
        ///     size.</param>
        /// <returns>A fresh int64 array of <paramref name="v"/>'s shape (0-d for a 0-d key).</returns>
        /// <exception cref="ArgumentException"><paramref name="side"/> is not <c>"left"</c> / <c>"right"</c>.</exception>
        /// <exception cref="ValueError">See <see cref="SearchSorted(NDArray, int, string, NDArray)"/>; or a StringDType
        ///     comparison that touched a None-like NA.</exception>
        /// <exception cref="TypeError">A sorter refusal; a U array converted to StringDType (NumPy's 'safe' cast refusal);
        ///     the first comparison Python refuses on the object path (<c>'&lt;' not supported between instances of 'str'
        ///     and 'int'</c>).</exception>
        /// <exception cref="UnicodeDecodeError">A text search converts bytes that are not ASCII to the common <c>U</c>
        ///     dtype.</exception>
        /// <exception cref="DTypePromotionError">Text with an operand that has no common dtype with it.</exception>
        /// <exception cref="NullReferenceException"><paramref name="a"/> or <paramref name="v"/> is null.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, P2): the public <c>np.searchsorted</c> opens the matrix boundary's
        ///     plain region and calls this.</remarks>
        public abstract NDArray SearchSorted(NDArray a, NDArray v, string side = "left", NDArray sorter = null);

        public abstract NDArray<long>[] NonZero(NDArray a);

        /// <summary>
        ///     NumPy's <c>np.trace(a, offset, axis1, axis2, dtype, out)</c> (<c>PyArray_Trace</c>), the whole call: the sums
        ///     along the diagonals <see cref="np.diagonal(NDArray, int, int, int)"/> returns, <c>add.reduce</c> over its
        ///     last axis with
        ///     <paramref name="dtype"/> and <paramref name="out"/>.
        /// </summary>
        /// <param name="a">The array (at least 2-D; any dtype, text included).</param>
        /// <param name="offset">The diagonal's offset from the main one.</param>
        /// <param name="axis1">The first axis of the 2-D sub-arrays.</param>
        /// <param name="axis2">The second axis of the 2-D sub-arrays.</param>
        /// <param name="dtype">The sum's dtype (<c>dtype=</c>), or null: then the loop is
        ///     <c>promote_types(out.dtype, a.dtype)</c> when <paramref name="out"/> is given (no NEP 50 widening), NEP 50's
        ///     accumulator of <paramref name="a"/> otherwise (signed integers and bool to int64, unsigned integers to
        ///     uint64).</param>
        /// <param name="out">NumPy's <c>out=</c>, an array of the result's shape that receives it with unsafe casting, or
        ///     null for a fresh result: the add reduction's out operand, checked by the reduce prologue (read-only first,
        ///     then its rank, then its length).</param>
        /// <returns><paramref name="out"/> itself when given; otherwise the sums: a read-only 0-d result (NumPy's scalar)
        ///     for 2-D <paramref name="a"/>, <paramref name="a"/>'s shape without the two axes otherwise.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><see cref="np.diagonal(NDArray, int, int, int)"/>'s refusals; a StringDType diagonal that cannot be reduced
        ///     (empty: no identity; an NA under the None-like rule); a read-only <paramref name="out"/> ("output array is
        ///     read-only") or one of the wrong rank ("output parameter for reduction operation add has the wrong number of
        ///     dimensions: ...").</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="TypeError">NumPy's loop resolution refuses text: a text operand with another out class, a
        ///     text out for numbers, a string <paramref name="dtype"/>.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/>'s length does not broadcast against the
        ///     diagonal (NumPy's "operands could not be broadcast together with remapped shapes" text).</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, P1b): the public <c>np.trace</c> opens the matrix boundary's
        ///     plain region and calls this.</remarks>
        public abstract NDArray Trace(NDArray a, int offset = 0, int axis1 = 0, int axis2 = 1, DType dtype = null,
            NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.ptp</c> (<c>_methods._ptp</c>): the range, <c>maximum.reduce(a, axis) -
        ///     minimum.reduce(a, axis)</c>, in <paramref name="a"/>'s own dtype, so an integer range wraps exactly as NumPy's
        ///     does (<c>ptp(int8 [-128, 127])</c> is -1).
        /// </summary>
        /// <param name="a">
        ///     The array, a plain one (<c>np.ptp</c> decides the <c>numpy.matrix</c> class first). Numbers and StringDType;
        ///     fixed-width text is refused as NumPy's <c>maximum.reduce</c> refuses it.
        /// </param>
        /// <param name="axis">
        ///     The reduced axis (negative counts from the end; a 0-d array accepts 0 and -1), or null for every axis.
        /// </param>
        /// <param name="out">
        ///     NumPy's <c>out=</c>: an array of the result's shape that receives the range and is returned, or null for a
        ///     fresh result. NumPy parses it before anything else, so a read-only out fails first.
        /// </param>
        /// <param name="keepdims">Whether the reduced axis stays with length one.</param>
        /// <returns>
        ///     <paramref name="out"/> itself when given; otherwise a fresh array, read-only when 0-d (NumPy's scalar).
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">Fixed-width text, or a text out: NumPy's maximum.reduce has no such loop.</exception>
        /// <exception cref="ValueError">
        ///     A read-only out, an empty reduction (NumPy's maximum has no identity), or a StringDType subtraction NumPy
        ///     refuses.
        /// </exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <remarks>
        ///     NumPy writes ptp in Python (<c>numpy/_core/_methods.py</c>); DefaultEngine's override runs those statements
        ///     over <see cref="ReduceAMax"/> and <see cref="ReduceAMin"/>. An override replaces <c>np.ptp</c> for every
        ///     array whose engine it is, and the tuple-axis overload re-enters this one for a single axis or for every
        ///     axis. New in 0.71.0 (the np-layer dtype plan, phase PE); it was an internal static of DefaultEngine.
        /// </remarks>
        public abstract NDArray Ptp(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false);

        /// <summary>
        ///     NumPy's <c>np.ptp(a, axis=(...))</c>: the range over the axes of a tuple, with <c>maximum.reduce</c>'s tuple
        ///     checks first (every axis in range, then no axis twice).
        /// </summary>
        /// <param name="a">The array, a plain one (the facade decides the <c>numpy.matrix</c> class first).</param>
        /// <param name="axis">
        ///     The reduced axes (negative values count from the end), or null for every axis. An empty tuple reduces
        ///     nothing: the result is <c>a - a</c>.
        /// </param>
        /// <param name="out">NumPy's <c>out=</c>, or null for a fresh result.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise the range, dtype preserved.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of range (a 0-d input has no axis for a tuple to name).</exception>
        /// <exception cref="ValueError">An axis appears twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <exception cref="TypeError">Fixed-width text, or a text out (NumPy's loop refusal).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> has the wrong shape.</exception>
        /// <remarks>
        ///     A tuple of one axis, or of every axis, re-enters <see cref="Ptp(NDArray, int?, NDArray, bool)"/> through
        ///     virtual dispatch, so an override of that member sees those calls too. New in 0.71.0 (the np-layer dtype
        ///     plan, phase PE).
        /// </remarks>
        public abstract NDArray Ptp(NDArray a, int[] axis, NDArray @out = null, bool keepdims = false);

        /// <summary>
        ///     NumPy's <c>np.sort(a, axis, kind, order, stable)</c> (<c>fromnumeric.sort</c> over <c>PyArray_Sort</c>), the
        ///     whole call: a sorted copy of <paramref name="a"/>, which is only read. Use <see cref="SortInPlace"/> for
        ///     <c>ndarray.sort</c>.
        /// </summary>
        /// <param name="a">The array (any dtype: text sorts by code points, <c>StringDType</c> in NumPy's NA order).</param>
        /// <param name="axis">The sorted axis (negative counts from the end), or null for the C-order flattening.</param>
        /// <param name="kind">NumPy's <c>kind=</c>, read from its first character (<c>q</c>uicksort, <c>h</c>eapsort,
        ///     <c>m</c>ergesort or <c>s</c>table, any case), or null for the default. It changes the answer only where NumPy's
        ///     does: the stable kind's float value sort keeps every NaN's bits, and StringDType refuses the heapsort kind.</param>
        /// <param name="order">NumPy's <c>order=</c>, a field name or a list of them, or null: every NumSharp dtype has no
        ///     fields, so any value is refused.</param>
        /// <param name="stable">NumPy's keyword-only <c>stable=</c>: true asks for the stable kind; any non-null value is
        ///     refused together with <paramref name="kind"/>.</param>
        /// <returns>A fresh array of <paramref name="a"/>'s dtype laid out as <c>copy(order='K')</c> lays out
        ///     <paramref name="a"/> (an F input stays F), or the sorted flattening for a null axis.</returns>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/> (<c>sort kind must be one of 'quick', 'heap', or
        ///     'stable' (got 'x')</c>), <paramref name="kind"/> together with <paramref name="stable"/>, or an
        ///     <paramref name="order"/>, all before the array is read; a <c>StringDType</c> line of two or more elements
        ///     holding an NA whose NA object is neither NaN-like nor a string.</exception>
        /// <exception cref="RuntimeError">A <c>StringDType</c> array with the heapsort kind (<c>unsupported sort
        ///     kind</c>).</exception>
        /// <exception cref="AxisError">The axis is out of range (a 0-d array refuses every axis: NumPy's sort does not ravel
        ///     it first).</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, phase PE, E5): the public <c>np.sort</c> applies its class
        ///     boundaries and calls this; DefaultEngine sorts with its AxisSort kernels.</remarks>
        public abstract NDArray Sort(NDArray a, int? axis = -1, string kind = null, object order = null, bool? stable = null);

        /// <summary>
        ///     NumPy's <c>ndarray.sort(axis, kind, order, stable)</c> (<c>array_sort</c> over <c>PyArray_Sort</c>), the whole
        ///     call: sorts <paramref name="a"/> IN PLACE along the axis, in NumPy's order of checks (the arguments, the axis,
        ///     writeability, then the sort). Use <see cref="Sort"/> for a sorted copy.
        /// </summary>
        /// <param name="a">The array sorted in place (any dtype).</param>
        /// <param name="axis">The sorted axis (negative counts from the end), or null: NumSharp's extension that sorts a
        ///     contiguous array's flattening in place.</param>
        /// <param name="kind">NumPy's <c>kind=</c>, read from its first character (<c>q</c>uicksort, <c>h</c>eapsort,
        ///     <c>m</c>ergesort or <c>s</c>table, any case), or null for the default. It changes the answer only where NumPy's
        ///     does: the stable kind's float value sort keeps every NaN's bits, and StringDType refuses the heapsort kind.</param>
        /// <param name="order">NumPy's <c>order=</c>, a field name or a list of them, or null: every NumSharp dtype has no
        ///     fields, so any value is refused.</param>
        /// <param name="stable">NumPy's keyword-only <c>stable=</c>: true asks for the stable kind; any non-null value is
        ///     refused together with <paramref name="kind"/>.</param>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/> (<c>sort kind must be one of 'quick', 'heap', or
        ///     'stable' (got 'x')</c>), <paramref name="kind"/> together with <paramref name="stable"/>, or an
        ///     <paramref name="order"/>, all before the array is read; a read-only
        ///     array (<c>sort array is read-only</c>); a <c>StringDType</c> line of two or more elements
        ///     holding an NA whose NA object is neither NaN-like nor a string.</exception>
        /// <exception cref="RuntimeError">A <c>StringDType</c> array with the heapsort kind (<c>unsupported sort
        ///     kind</c>).</exception>
        /// <exception cref="AxisError">The axis is out of range.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, phase PE, E5): <c>NDArray.sort</c> calls this.</remarks>
        public abstract void SortInPlace(NDArray a, int? axis = -1, string kind = null, object order = null, bool? stable = null);

        /// <summary>
        ///     NumPy's <c>np.argsort(a, axis, kind, order, stable)</c> and <c>ndarray.argsort</c> (<c>PyArray_ArgSort</c>),
        ///     the whole call: the int64 indices that sort <paramref name="a"/> along the axis.
        /// </summary>
        /// <param name="a">The array (only read). A 0-d array ravels to one element first, as NumPy's
        ///     <c>PyArray_CheckAxis</c> does: its answer is <c>[0]</c>, and an axis other than 0 / -1 reports dimension 1.</param>
        /// <param name="axis">The sorted axis (negative counts from the end), or null for the C-order flattening.</param>
        /// <param name="kind">NumPy's <c>kind=</c>, read from its first character (<c>q</c>uicksort, <c>h</c>eapsort,
        ///     <c>m</c>ergesort or <c>s</c>table, any case), or null for the default. It changes the answer only where NumPy's
        ///     does: the stable kind's float value sort keeps every NaN's bits, and StringDType refuses the heapsort kind.</param>
        /// <param name="order">NumPy's <c>order=</c>, a field name or a list of them, or null: every NumSharp dtype has no
        ///     fields, so any value is refused.</param>
        /// <param name="stable">NumPy's keyword-only <c>stable=</c>: true asks for the stable kind; any non-null value is
        ///     refused together with <paramref name="kind"/>.</param>
        /// <returns>A fresh, C-contiguous int64 array of <paramref name="a"/>'s shape (one-dimensional for a null axis).
        ///     Ties keep their order for every numeric kind (NumSharp's numeric argsort is one stable order); text follows
        ///     NumPy's introsort for the default kind.</returns>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/> (<c>sort kind must be one of 'quick', 'heap', or
        ///     'stable' (got 'x')</c>), <paramref name="kind"/> together with <paramref name="stable"/>, or an
        ///     <paramref name="order"/>, all before the array is read; a <c>StringDType</c> line of two or more elements
        ///     holding an NA whose NA object is neither NaN-like nor a string.</exception>
        /// <exception cref="RuntimeError">A <c>StringDType</c> array with the heapsort kind (<c>unsupported sort
        ///     kind</c>), before the axis: NumPy resolves the argsort loop first.</exception>
        /// <exception cref="AxisError">The axis is out of range.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, phase PE, E5): <c>np.argsort</c> and <c>NDArray.argsort</c>
        ///     call this.</remarks>
        public abstract NDArray ArgSort(NDArray a, int? axis = -1, string kind = null, object order = null, bool? stable = null);

        /// <summary>
        ///     NumPy's <c>np.partition(a, kth, axis, kind, order)</c> (<c>fromnumeric.partition</c> over
        ///     <c>PyArray_Partition</c>), the whole call: a copy of <paramref name="a"/> in which every
        ///     <paramref name="kth"/> element is where a sorted copy holds it, with no larger element before it and no
        ///     smaller one after it along the axis (introselect). Use <see cref="PartitionInPlace"/> for
        ///     <c>ndarray.partition</c>.
        /// </summary>
        /// <param name="a">The array (only read). Text has no partition function, so NumPy sorts each line.</param>
        /// <param name="kth">NumPy's array_like of positions along the axis: a C# <see cref="int"/>, an <see cref="int"/>[],
        ///     an integer index array, or any other object, converted with <c>np.asanyarray</c>. A negative position counts
        ///     from the end; every listed position ends where a sorted copy holds its element.</param>
        /// <param name="axis">The axis (negative counts from the end), or null for the flattened array.</param>
        /// <param name="kind">NumPy's <c>kind=</c>: only <c>"introselect"</c> (the default) or null.</param>
        /// <param name="order">NumPy's <c>order=</c>, a field name, or null: every NumSharp dtype has no fields, so any
        ///     value is refused.</param>
        /// <returns>A fresh array of <paramref name="a"/>'s dtype laid out as <c>copy(order='K')</c> lays out
        ///     <paramref name="a"/>, or the partitioned flattening for a null axis. Float NaNs end each line with their own
        ///     bits; the arrangement between the listed positions is the introselect's own.</returns>
        /// <exception cref="ValueError">A <paramref name="kind"/> other than <c>"introselect"</c> (<c>select kind must be
        ///     'introselect' (got 'x')</c>) or an <paramref name="order"/>, before the array is read; a kth index array of
        ///     more than one dimension (<c>object too deep for desired array</c>, before the axis) or of bool dtype
        ///     (<c>Booleans unacceptable as partition index</c>); a position out of bounds (<c>kth(=5) out of bounds
        ///     (5)</c>, skipped for an empty array).</exception>
        /// <exception cref="TypeError">A kth index array of a non-integer dtype (<c>Partition index must be
        ///     integer</c>).</exception>
        /// <exception cref="AxisError">The axis is out of range.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="kth"/> is null, at NumPy's kth stage (after the axis:
        ///     NumPy's kth=None raises its TypeError there).</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, phase PE, E5): the public <c>np.partition</c> applies its class
        ///     boundaries and calls this; DefaultEngine partitions with its AxisPartition kernels.</remarks>
        public abstract NDArray Partition(NDArray a, object kth, int? axis = -1, string kind = "introselect", string order = null);

        /// <summary>
        ///     NumPy's <c>ndarray.partition(kth, axis, kind, order)</c> (<c>array_partition</c> over <c>PyArray_Partition</c>),
        ///     the whole call: partitions <paramref name="a"/> IN PLACE along the axis (see <see cref="Partition"/>), in
        ///     NumPy's order of checks (the kind, the order, the axis, writeability, then the positions).
        /// </summary>
        /// <param name="a">The array partitioned in place.</param>
        /// <param name="kth">NumPy's array_like of positions along the axis: a C# <see cref="int"/>, an <see cref="int"/>[],
        ///     an integer index array, or any other object, converted with <c>np.asanyarray</c>. A negative position counts
        ///     from the end; every listed position ends where a sorted copy holds its element.</param>
        /// <param name="axis">The axis (negative counts from the end), or null: NumSharp's extension that partitions a
        ///     contiguous array's flattening in place.</param>
        /// <param name="kind">NumPy's <c>kind=</c>: only <c>"introselect"</c> (the default) or null.</param>
        /// <param name="order">NumPy's <c>order=</c>, a field name, or null: every NumSharp dtype has no fields, so any
        ///     value is refused.</param>
        /// <exception cref="ValueError">A <paramref name="kind"/> other than <c>"introselect"</c> (<c>select kind must be
        ///     'introselect' (got 'x')</c>) or an <paramref name="order"/>, before the array is read; a kth index array of
        ///     more than one dimension (<c>object too deep for desired array</c>, before the axis) or of bool dtype
        ///     (<c>Booleans unacceptable as partition index</c>); a read-only array (<c>partition array is
        ///     read-only</c>); a position out of bounds (<c>kth(=5) out of bounds
        ///     (5)</c>, skipped for an empty array).</exception>
        /// <exception cref="TypeError">A kth index array of a non-integer dtype (<c>Partition index must be
        ///     integer</c>).</exception>
        /// <exception cref="AxisError">The axis is out of range.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="kth"/> is null, at NumPy's kth stage (after the axis:
        ///     NumPy's kth=None raises its TypeError there).</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, phase PE, E5): <c>NDArray.partition</c> calls this.</remarks>
        public abstract void PartitionInPlace(NDArray a, object kth, int? axis = -1, string kind = "introselect", string order = null);

        /// <summary>
        ///     NumPy's <c>np.argpartition(a, kth, axis, kind, order)</c> and <c>ndarray.argpartition</c>
        ///     (<c>PyArray_ArgPartition</c>), the whole call: the int64 indices that partition <paramref name="a"/> along the
        ///     axis (see <see cref="Partition"/>).
        /// </summary>
        /// <param name="a">The array (only read). A 0-d array ravels to one element first, as NumPy's
        ///     <c>PyArray_CheckAxis</c> does.</param>
        /// <param name="kth">NumPy's array_like of positions along the axis: a C# <see cref="int"/>, an <see cref="int"/>[],
        ///     an integer index array, or any other object, converted with <c>np.asanyarray</c>. A negative position counts
        ///     from the end; every listed position ends where a sorted copy holds its element.</param>
        /// <param name="axis">The axis (negative counts from the end), or null for the flattened array.</param>
        /// <param name="kind">NumPy's <c>kind=</c>: only <c>"introselect"</c> (the default) or null.</param>
        /// <param name="order">NumPy's <c>order=</c>, a field name, or null: every NumSharp dtype has no fields, so any
        ///     value is refused.</param>
        /// <returns>A fresh, C-contiguous int64 array of <paramref name="a"/>'s shape (one-dimensional for a null axis):
        ///     the element at each listed position is where a sorted copy holds it. Text answers with the default argsort
        ///     (NumPy's generic introsort fallback).</returns>
        /// <exception cref="ValueError">A <paramref name="kind"/> other than <c>"introselect"</c> (<c>select kind must be
        ///     'introselect' (got 'x')</c>) or an <paramref name="order"/>, before the array is read; a kth index array of
        ///     more than one dimension (<c>object too deep for desired array</c>, before the axis) or of bool dtype
        ///     (<c>Booleans unacceptable as partition index</c>); a position out of bounds (<c>kth(=5) out of bounds
        ///     (5)</c>, skipped for an empty array).</exception>
        /// <exception cref="TypeError">A kth index array of a non-integer dtype (<c>Partition index must be
        ///     integer</c>).</exception>
        /// <exception cref="AxisError">The axis is out of range.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="kth"/> is null, at NumPy's kth stage (after the axis:
        ///     NumPy's kth=None raises its TypeError there).</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, phase PE, E5): <c>np.argpartition</c> and
        ///     <c>NDArray.argpartition</c> call this.</remarks>
        public abstract NDArray ArgPartition(NDArray a, object kth, int? axis = -1, string kind = "introselect", string order = null);
    }
}
