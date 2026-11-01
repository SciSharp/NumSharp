using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

// =============================================================================
// Default.Reduction.AddMul.Keywords.cs — np.sum / np.prod with NumPy's initial= and where= (np-layer dtype plan, P10
// item 1, decision D10; plan docs/plans/api-parameter-parity.md, items B1 and B2)
// =============================================================================
//
// np.sum(a, axis, dtype, out, keepdims, initial, where) IS np.add.reduce(a, axis, dtype, out, keepdims=, initial=, where=)
// (numpy/_core/fromnumeric.py::_wrapreduction; ndarray.sum is _methods._sum, the same call), and np.prod is
// np.multiply.reduce. A call with either keyword runs PyUFunc_GenericReduction / PyUFunc_ReduceWrapper's statements, in
// NumPy's order (numpy/_core/src/umath/ufunc_object.c, reduction.c):
//
//   1. the parsed arguments: a C# scalar passed as out ("return arrays must be of ArrayType"), a read-only out, a mask that
//      is not bool (ReductionArguments.ValidateParsedArguments), then the axis (a 0-d input takes 0 / -1 as every axis);
//   2. the LOOP dtype (reducelike_promote_and_resolve): dtype=, else promote_types(out, a) with an out — no small-integer
//      widening —, else NEP 50's accumulator (bool and the signed integers narrower than 64 bits run int64, the unsigned
//      ones uint64); text resolves here (StringDType add concatenates, everything else is NumPy's refusal);
//   3. the iterator's shape checks over [out, input, where] (out's rank, the broadcast, a mask with more axes);
//   4. initial= packed into the loop dtype (PyArray_Pack: ReductionInitial — a Python int range-checked, a float truncated
//      into an integer loop, a 0-d array cast), or the loop's IDENTITY (add 0, multiply 1; the bool loops False / True);
//   5. the result — the out (any dtype: values reach it with an unsafe cast) or a fresh array laid out in the iteration
//      order — is SET to that value (raw_array_assign_scalar), and NumPy's iterator folds every selected element in
//      (NumPyReduceIterator: the axis order, fills, cores, masked runs, a cast out's round trip through its own dtype at
//      every fill). An empty reduction, or an output no mask selects, keeps the value.
//
// The loops are NumPy's own, in both of the iterator's call forms:
//   * add: the inexact reduce form `*out += pairwise_sum(run)` and the elementwise form with NumPy's NaN operand order
//     (NumPyAddLoops); modular sums for the integers (any order gives the same bits); logical OR for bool (BOOL_add);
//   * multiply: the reduce form is SEQUENTIAL (`io1 *= in2`: loops_arithm_fp's PW = 0 for multiply), so the RUNNING
//     value's NaN survives (probed 2.4.2: np.prod([p1, p2]) keeps p1's payload, initial=p1 over [p2] too); the elementwise
//     form keeps the SECOND operand's NaN (np.prod of rows [p1], [p2] along axis 0 keeps p2's); HALF_multiply accumulates a
//     call in float32 and narrows once; CDOUBLE_multiply's reduce form is MSVC's contracted loop_scalar (the output stride
//     is 0) and its elementwise form simd_cmul; logical AND for bool (BOOL_multiply); modular products for the integers.
//
// A call without either keyword keeps the engine member's own routes (ReduceAdd / ReduceProduct, out= included).
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Whether <paramref name="where"/> is Python's <c>where=True</c>: a 0-d bool array holding true, the array a C#
        ///     <c>true</c> literal converts to. NumPy treats the literal as no mask at all; an <c>np.True_</c> mask gives the
        ///     same result in every reduction that has an identity (add, multiply), so both read as no mask there.
        /// </summary>
        /// <param name="where">The caller's mask, or null.</param>
        /// <returns>True for a 0-d bool true mask; false for null, a mask with dimensions, a false or a non-bool one (the
        ///     reduce prologue still checks a non-bool 0-d mask's dtype).</returns>
        internal static bool IsTrueScalarMask(NDArray where)
            => where is not null && where.ndim == 0 && where.GetTypeCode == NPTypeCode.Boolean && where.GetAtIndex<bool>(0);

        /// <summary>
        ///     <c>np.sum</c> / <c>np.prod</c> with NumPy's <c>initial=</c> and <c>where=</c>: <c>ufunc.reduce</c> of
        ///     <c>add</c> / <c>multiply</c> through NumPy's reduction iterator (see the file header for the statement order).
        /// </summary>
        /// <param name="arr">The array to reduce, any layout.</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="dtype">The loop dtype (<c>dtype=</c>), or null for NumPy's resolution (file header, step 2).</param>
        /// <param name="out">The array to receive the result (any dtype: the loop's values are cast into it), or null.</param>
        /// <param name="initial">The start value of every output (<see cref="ReductionInitial"/>'s conversion), or null for
        ///     the loop's identity.</param>
        /// <param name="where">A bool mask broadcast against <paramref name="arr"/>: only selected elements reduce. Null for no
        ///     mask (the engine members turn a 0-d true mask into null before calling).</param>
        /// <param name="multiply">True for <c>multiply.reduce</c> (<c>np.prod</c>), false for <c>add.reduce</c>
        ///     (<c>np.sum</c>).</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array in the loop dtype, a read-only 0-d array
        ///     (NumPy's scalar) for a reduction over every axis without keepdims.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> is null.</exception>
        /// <exception cref="TypeError">A C# scalar as <paramref name="out"/>; NumPy's text refusals; an
        ///     <paramref name="initial"/> of a type the loop cannot take (a complex into a real loop).</exception>
        /// <exception cref="ValueError">A read-only or misshapen out, a mask with more axes than <paramref name="arr"/>; an
        ///     unconvertible <paramref name="initial"/> (a NaN into an integer loop, a sequence).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array (NumPy's TypeError text).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException">The mask or the out does not broadcast against <paramref name="arr"/>.</exception>
        /// <exception cref="OverflowException"><paramref name="initial"/> does not fit an integer loop; a decimal result
        ///     leaves decimal's range.</exception>
        /// <remarks><see cref="NDAnchoredAttribute"/>: the iterator reads raw addresses of the input, the mask and the out, and
        ///     the engine members reach this method without a scope of their own; the weaver roots the operands until it
        ///     returns, so a caller's temporary operand or mask cannot be collected under the loop.</remarks>
        [NDAnchored]
        internal unsafe NDArray AddMulNumPy(NDArray arr, int? axis, bool keepdims, DType dtype, NDArray @out, object initial,
            NDArray @where, bool multiply)
        {
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));
            string op = multiply ? "prod" : "sum";
            string ufunc = multiply ? "multiply" : "add";

            // 1. The parsed arguments in NumPy's order (an out that is not an array, a read-only out, the mask's dtype), then
            // the axis.
            ReductionArguments.ValidateParsedArguments(@out, @where);
            bool[] flags = ReductionArguments.AxisFlags(arr.ndim, axis);

            // 2 (text). The loop resolution answers for text before the iterator's shape checks: StringDType's add
            // concatenates (initial= packed into its descriptor), every other text loop is NumPy's refusal.
            if (Strings.NDStringUfuncs.IsText(arr) || Strings.NDStringUfuncs.IsText(@out) || Strings.NDStringUfuncs.IsTextRequest(dtype))
                return AddMulTextNumPy(op, arr, axis, keepdims, dtype, @out, initial, @where);

            // 2. The loop: dtype=, else promote_types(out, arr) (no NEP 50 widening once out is given), else the accumulator.
            NPTypeCode inT = arr.GetTypeCode;
            NPTypeCode loop = UfuncLoopCode(dtype)
                              ?? (@out is not null ? DTypePromotion.PromoteTypes(@out.GetTypeCode, inT) : inT.GetAccumulatingType());

            // 3. The iterator's shape checks over [out, input, where].
            ReductionArguments.ValidateOperands(arr, flags, keepdims, @out, @where, ufunc);

            // 4. initial= packed into the loop (after every shape error, as PyUFunc_ReduceWrapper packs it), else the identity.
            byte* seed = stackalloc byte[ReductionInitial.SlotBytes];
            var slot = new Span<byte>(seed, ReductionInitial.SlotBytes);
            if (!ReductionInitial.TryPack(initial, loop, slot))
                WriteIdentity(loop, multiply, slot);

            // 5. The result set to the start value, then the iterator over [out, input, where].
            return AddMulThroughIterator(arr, flags, keepdims, @out, @where, loop, seed, multiply);
        }

        /// <summary>
        ///     The identity of NumPy's add / multiply loop in <paramref name="loop"/>: 0 / 1 in the loop's type (+0.0 for an
        ///     inexact loop, 0+0j / 1+0j for complex), False / True for the bool loops (logical OR / AND).
        /// </summary>
        /// <param name="loop">The loop dtype (one of the 15).</param>
        /// <param name="multiply">True for multiply's identity 1, false for add's 0.</param>
        /// <param name="slot">Receives the element (at least <see cref="ReductionInitial.SlotBytes"/> bytes).</param>
        /// <exception cref="NotSupportedException"><paramref name="loop"/> is not one of the 15 dtypes.</exception>
        private static void WriteIdentity(NPTypeCode loop, bool multiply, Span<byte> slot)
        {
            slot.Clear();
            if (!multiply)
                return;   // every add identity is all-zero bits (+0.0, 0, False, 0m, 0+0j)
            switch (loop)
            {
                case NPTypeCode.Boolean: slot[0] = 1; break;
                case NPTypeCode.SByte: slot[0] = 1; break;
                case NPTypeCode.Byte: slot[0] = 1; break;
                case NPTypeCode.Int16: Unsafe.WriteUnaligned(ref slot[0], (short)1); break;
                case NPTypeCode.UInt16: Unsafe.WriteUnaligned(ref slot[0], (ushort)1); break;
                case NPTypeCode.Char: Unsafe.WriteUnaligned(ref slot[0], (char)1); break;
                case NPTypeCode.Int32: Unsafe.WriteUnaligned(ref slot[0], 1); break;
                case NPTypeCode.UInt32: Unsafe.WriteUnaligned(ref slot[0], 1u); break;
                case NPTypeCode.Int64: Unsafe.WriteUnaligned(ref slot[0], 1L); break;
                case NPTypeCode.UInt64: Unsafe.WriteUnaligned(ref slot[0], 1UL); break;
                case NPTypeCode.Half: Unsafe.WriteUnaligned(ref slot[0], (Half)1); break;
                case NPTypeCode.Single: Unsafe.WriteUnaligned(ref slot[0], 1f); break;
                case NPTypeCode.Double: Unsafe.WriteUnaligned(ref slot[0], 1d); break;
                case NPTypeCode.Decimal: Unsafe.WriteUnaligned(ref slot[0], 1m); break;
                case NPTypeCode.Complex: Unsafe.WriteUnaligned(ref slot[0], System.Numerics.Complex.One); break;
                default: throw new NotSupportedException($"multiply.reduce: dtype {loop} has no identity here.");
            }
        }

        /// <summary>
        ///     The text form of <see cref="AddMulNumPy"/>: a <c>StringDType</c> input under <c>add</c> concatenates from the
        ///     packed <c>initial=</c> over the selected elements (<see cref="Strings.NDVStringReduction.SumNumPyKeywords"/>); a
        ///     numeric <c>dtype=</c> over text parses the text first and reduces the numbers with the same keywords; every
        ///     other text call is NumPy's refusal.
        /// </summary>
        /// <param name="op">The NumPy function (<c>"sum"</c> or <c>"prod"</c>).</param>
        /// <param name="arr">The array.</param>
        /// <param name="axis">The axis (already validated), or null.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="out">The out, or null.</param>
        /// <param name="initial">The start value, or null.</param>
        /// <param name="where">The mask, or null.</param>
        /// <returns>The result (<paramref name="out"/> when given).</returns>
        /// <exception cref="TypeError">NumPy's loop refusal (no text loop for multiply, fixed-width text, an out of another
        ///     class).</exception>
        /// <exception cref="ValueError">A mask without <paramref name="initial"/> over <c>StringDType</c> (its add has no
        ///     identity), an empty reduction without it, or a text that does not parse into a numeric loop.</exception>
        /// <exception cref="NotSupportedException">A text OUT under a numeric loop with these keywords: NumPy casts the running
        ///     value through the text at every fill, which the keyword path does not model.</exception>
        private NDArray AddMulTextNumPy(string op, NDArray arr, int? axis, bool keepdims, DType dtype, NDArray @out,
            object initial, NDArray @where)
        {
            // StringDType's add: its loop resolution, then PyUFunc_ReduceWrapper's checks and the concatenation.
            if (op == "sum" && dtype is null && Strings.NDStringUfuncs.IsVString(arr))
                return Strings.NDVStringReduction.SumNumPyKeywords(arr, axis, @out, keepdims, initial, @where);
            // A numeric dtype= over a text input (no text out): NumPy's iterator casts (parses) every element into the loop
            // before the loop reads it, so the reduction of the parsed array with the same keywords is NumPy's.
            if (!Strings.NDStringUfuncs.IsText(@out)
                && Strings.NDStringUfuncs.ParseForNumericReduce(op, arr, axis, dtype) is { } parsed)
                return Strings.NDStringUfuncs.ReduceParsed(parsed, p => AddMulNumPy(p, axis, keepdims, dtype, @out, initial, @where, op == "prod"));
            // Everything else is NumPy's refusal, raised with its text (the out joins the resolution).
            Strings.NDStringUfuncs.ThrowIfReduce(op, arr, axis, @out, dtype);
            throw new NotSupportedException(
                $"np.{op} with initial= or where= and a text out= is not supported: NumPy round-trips the running value through the out's text at every buffer fill.");
        }

        /// <summary>
        ///     The result of <see cref="AddMulNumPy"/>: set to the start value, then NumPy's reduction iterator over
        ///     <c>[out, input, where]</c> folds every selected element in with the loop dtype's add or multiply loop.
        /// </summary>
        /// <param name="a">The input (numeric).</param>
        /// <param name="flags">The reduced-axis flags.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="out">The out, or null.</param>
        /// <param name="where">The mask, or null.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="seed">The packed start value (the initial or the identity), in the loop dtype.</param>
        /// <param name="multiply">True for multiply.</param>
        /// <returns><paramref name="out"/>, or the fresh result (a read-only 0-d scalar when it has no dimensions).</returns>
        /// <exception cref="OverflowException">A decimal result leaves decimal's range (the fresh result is released first).</exception>
        private unsafe NDArray AddMulThroughIterator(NDArray a, bool[] flags, bool keepdims, NDArray @out, NDArray @where,
            NPTypeCode loop, byte* seed, bool multiply)
        {
            // COPY_IF_OVERLAP: an input or mask that may share memory with the out is read from a copy, as NumPy's iterator
            // does (the out is written while they are still read).
            NDArray input = a, mask = @where;
            NDArray inputCopy = null, maskCopy = null;
            NDArray result = null;
            try
            {
                if (@out is not null)
                {
                    if (NDMemOverlap.SolveMayShareMemory(@out, a, 0) != MemOverlap.No)
                        input = inputCopy = a.copy('K');
                    if (mask is not null && NDMemOverlap.SolveMayShareMemory(@out, mask, 0) != MemOverlap.No)
                        mask = maskCopy = mask.copy('K');
                }

                NPTypeCode inT = input.GetTypeCode;
                bool outCast = @out is not null && @out.GetTypeCode != loop;
                var it = NumPyReduceIterator.Build(input.Shape, flags, keepdims, @out?.Shape, mask?.Shape,
                    outCast, inT != loop, NDIterBufferManager.CurrentBufferSize);

                // The result: the out, or a fresh array in the loop dtype laid out in the iteration order
                // (npyiter_new_temp_array). raw_array_assign_scalar then sets every output to the start value, cast from the
                // loop dtype into the result's (an int8 out of an int16 loop holds the initial wrapped, as NumPy's does).
                result = @out ?? NewArray(loop, it.ResultDims.Length == 0 ? Shape.Scalar : Shape.Adopt(it.ResultDims, it.ResultStrides), false);
                using (var init = NewArray(loop, Shape.Scalar, false))
                {
                    Buffer.MemoryCopy(seed, (byte*)init.Address, loop.SizeOf(), loop.SizeOf());
                    UnmanagedStorage.Copyto(result, init, NPY_CASTING.NPY_UNSAFE_CASTING);
                }

                // Every output starts from a value, so no element is skipped (NumPy's skip_first_count is 0).
                RunAddMul(it, result, input, mask, loop, multiply);
                if (@out is not null)
                    return @out;
                var r = result;
                result = null;   // handed to the caller: not disposed below
                return r.ndim == 0 ? r.MarkReductionScalar() : r;
            }
            finally
            {
                if (@out is null)
                    result?.Dispose();
                inputCopy?.Dispose();
                maskCopy?.Dispose();
            }
        }

        /// <summary>
        ///     Runs the iterator with the loop dtype's NumPy add or multiply loop (see the file header).
        /// </summary>
        /// <param name="it">The iterator.</param>
        /// <param name="result">The result (the out, or the fresh array), already set to the start value.</param>
        /// <param name="input">The input (a copy when it overlapped the out).</param>
        /// <param name="mask">The mask, or null.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="multiply">True for multiply.</param>
        /// <exception cref="NotSupportedException"><paramref name="loop"/> is not one of the 15 dtypes.</exception>
        /// <exception cref="OverflowException">A decimal result leaves decimal's range.</exception>
        private static unsafe void RunAddMul(NumPyReduceIterator it, NDArray result, NDArray input, NDArray mask, NPTypeCode loop,
            bool multiply)
        {
            byte* o = (byte*)result.Address + result.Shape.offset * result.dtypesize;
            byte* x = (byte*)input.Address + input.Shape.offset * input.dtypesize;
            byte* m = mask is null ? null : (byte*)mask.Address + mask.Shape.offset;
            NPTypeCode oT = result.GetTypeCode, xT = input.GetTypeCode;
            switch (loop)
            {
                case NPTypeCode.Double:
                    if (multiply) it.Run(o, oT, x, xT, m, loop, 0, NumPyMultiplyLoops.Float<double>());
                    else it.Run(o, oT, x, xT, m, loop, 0, NumPyAddLoops.Double());
                    break;
                case NPTypeCode.Single:
                    if (multiply) it.Run(o, oT, x, xT, m, loop, 0, NumPyMultiplyLoops.Float<float>());
                    else it.Run(o, oT, x, xT, m, loop, 0, NumPyAddLoops.Single());
                    break;
                case NPTypeCode.Half:
                    if (multiply) it.Run(o, oT, x, xT, m, loop, 0, NumPyMultiplyLoops.Half());
                    else it.Run(o, oT, x, xT, m, loop, 0, NumPyAddLoops.Half());
                    break;
                case NPTypeCode.Complex:
                    if (multiply) it.Run(o, oT, x, xT, m, loop, 0, NumPyMultiplyLoops.Complex());
                    else it.Run(o, oT, x, xT, m, loop, 0, NumPyAddLoops.Complex());
                    break;
                case NPTypeCode.Decimal:
                    if (multiply) it.Run(o, oT, x, xT, m, loop, 0, NumPyMultiplyLoops.Decimal());
                    else it.Run(o, oT, x, xT, m, loop, 0, NumPyAddLoops.Decimal());
                    break;
                case NPTypeCode.Boolean:
                    // BOOL_add IS logical_or and BOOL_multiply logical_and (NumPy aliases them), the loops BOOL_maximum /
                    // BOOL_minimum also are: the early-exit reduce and the SIMD elementwise op.
                    it.Run(o, oT, x, xT, m, loop, 0, NumPyMinMaxLoops.Bool(isMax: !multiply));
                    break;
                case NPTypeCode.SByte: RunInteger<sbyte>(it, o, oT, x, xT, m, loop, multiply); break;
                case NPTypeCode.Byte: RunInteger<byte>(it, o, oT, x, xT, m, loop, multiply); break;
                case NPTypeCode.Int16: RunInteger<short>(it, o, oT, x, xT, m, loop, multiply); break;
                case NPTypeCode.UInt16: RunInteger<ushort>(it, o, oT, x, xT, m, loop, multiply); break;
                case NPTypeCode.Char: RunInteger<char>(it, o, oT, x, xT, m, loop, multiply); break;
                case NPTypeCode.Int32: RunInteger<int>(it, o, oT, x, xT, m, loop, multiply); break;
                case NPTypeCode.UInt32: RunInteger<uint>(it, o, oT, x, xT, m, loop, multiply); break;
                case NPTypeCode.Int64: RunInteger<long>(it, o, oT, x, xT, m, loop, multiply); break;
                case NPTypeCode.UInt64: RunInteger<ulong>(it, o, oT, x, xT, m, loop, multiply); break;
                default:
                    throw new NotSupportedException($"np.sum / np.prod: dtype {loop} is not supported.");
            }
        }

        /// <summary>The integer branch of <see cref="RunAddMul"/>: the modular add or multiply loop of
        ///     <typeparamref name="T"/>.</summary>
        /// <typeparam name="T">The integer loop's element type (NumSharp's char is NumPy's uint16).</typeparam>
        /// <param name="it">The iterator.</param>
        /// <param name="o">The result's logical element 0.</param>
        /// <param name="oT">The result's dtype.</param>
        /// <param name="x">The input's logical element 0.</param>
        /// <param name="xT">The input's dtype.</param>
        /// <param name="m">The mask's logical element 0, or null.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="multiply">True for multiply.</param>
        private static unsafe void RunInteger<T>(NumPyReduceIterator it, byte* o, NPTypeCode oT, byte* x, NPTypeCode xT, byte* m,
            NPTypeCode loop, bool multiply) where T : unmanaged, IBinaryInteger<T>
        {
            if (multiply)
                it.Run(o, oT, x, xT, m, loop, 0, NumPyIntegerAddMulLoops.Multiply<T>());
            else
                it.Run(o, oT, x, xT, m, loop, 0, NumPyIntegerAddMulLoops.Add<T>());
        }
    }

    /// <summary>
    ///     NumPy's integer add and multiply loops in the call forms of <see cref="NumPyReduceIterator.Loop{T}"/>. Integer
    ///     arithmetic wraps (NumPy's C loops, C#'s unchecked operators) and is associative and commutative, so every order of
    ///     the same elements gives the same bits: the forms fold whole calls, masked ones included, in whatever order is
    ///     fastest.
    /// </summary>
    internal static unsafe class NumPyIntegerAddMulLoops
    {
        /// <summary>The add loop (<c>@TYPE@_add</c>): all five call forms.</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<T> Add<T>() where T : unmanaged, IBinaryInteger<T>
            => new(&AddReduce<T>, &AddBinary<T>, &AddMaskedReduce<T>, &AddMaskedBinary<T>, &AddBinaryRows<T>);

        /// <summary>The multiply loop (<c>@TYPE@_multiply</c>): all five call forms.</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<T> Multiply<T>() where T : unmanaged, IBinaryInteger<T>
            => new(&MulReduce<T>, &MulBinary<T>, &MulMaskedReduce<T>, &MulMaskedBinary<T>, &MulBinaryRows<T>);

        /// <summary>The add reduce form: the running value plus every element of the run (wrapping).</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <param name="acc">The out's running value.</param>
        /// <param name="x">The run's first element.</param>
        /// <param name="xs">The run's stride in elements.</param>
        /// <param name="n">The run's element count.</param>
        /// <returns>The out's new value.</returns>
        /// <remarks>A contiguous run of a type <see cref="Vector{T}"/> supports sums a vector at a time (wrapping lanes, then
        ///     a wrapping horizontal sum: the same modular total).</remarks>
        private static T AddReduce<T>(T acc, T* x, long xs, long n) where T : unmanaged, IBinaryInteger<T>
        {
            long i = 0;
            if (xs == 1 && Vector.IsHardwareAccelerated && Vector<T>.IsSupported && n >= 2L * Vector<T>.Count)
            {
                int w = Vector<T>.Count;
                var v0 = Vector<T>.Zero;
                var v1 = Vector<T>.Zero;
                for (; i + 2 * w <= n; i += 2 * w)
                {
                    v0 += Vector.Load(x + i);
                    v1 += Vector.Load(x + i + w);
                }
                acc += Vector.Sum(v0 + v1);
            }
            for (; i < n; i++)
                acc += x[i * xs];
            return acc;
        }

        /// <summary>The add elementwise form: <c>o[i] = o[i] + x[i]</c> (wrapping).</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <param name="o">The out run.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="n">The element count.</param>
        private static void AddBinary<T>(T* o, long os, T* x, long xs, long n) where T : unmanaged, IBinaryInteger<T>
        {
            long i = 0;
            if (os == 1 && xs == 1 && Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
            {
                int w = Vector<T>.Count;
                for (; i + w <= n; i += w)
                    Vector.Store(Vector.Load(o + i) + Vector.Load(x + i), o + i);
            }
            for (; i < n; i++)
                o[i * os] += x[i * xs];
        }

        /// <summary>The add reduce form over a masked call in one piece: the running value plus every selected element.</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <param name="acc">The out's running value.</param>
        /// <param name="x">The input's first element.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="m">The mask's first byte.</param>
        /// <param name="ms">The mask's byte stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The out's new value.</returns>
        private static T AddMaskedReduce<T>(T acc, T* x, long xs, byte* m, long ms, long n) where T : unmanaged, IBinaryInteger<T>
        {
            for (long i = 0; i < n; i++)
                if (m[i * ms] != 0)
                    acc += x[i * xs];
            return acc;
        }

        /// <summary>The add elementwise form over a masked call: <c>o[i] += x[i]</c> where <c>m[i]</c>.</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <param name="o">The out run.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="m">The mask's first byte.</param>
        /// <param name="ms">The mask's byte stride.</param>
        /// <param name="n">The element count.</param>
        private static void AddMaskedBinary<T>(T* o, long os, T* x, long xs, byte* m, long ms, long n) where T : unmanaged, IBinaryInteger<T>
        {
            for (long i = 0; i < n; i++)
                if (m[i * ms] != 0)
                    o[i * os] += x[i * xs];
        }

        /// <summary>The add elementwise form over several consecutive rows into one out row: each out element plus the
        ///     rows' elements at its position.</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <param name="o">The out row.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The first input row.</param>
        /// <param name="xs">The input stride within a row.</param>
        /// <param name="rs">The distance from one input row to the next (0: a broadcast row).</param>
        /// <param name="rows">The row count.</param>
        /// <param name="n">The elements per row.</param>
        private static void AddBinaryRows<T>(T* o, long os, T* x, long xs, long rs, long rows, long n) where T : unmanaged, IBinaryInteger<T>
        {
            for (long r = 0; r < rows; r++)
                AddBinary(o, os, x + r * rs, xs, n);
        }

        /// <summary>The multiply reduce form: the running value times every element of the run (wrapping).</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <param name="acc">The out's running value.</param>
        /// <param name="x">The run's first element.</param>
        /// <param name="xs">The run's stride in elements.</param>
        /// <param name="n">The run's element count.</param>
        /// <returns>The out's new value.</returns>
        /// <remarks>Four independent products (the modular product is order-free), so the multiplies overlap instead of
        ///     waiting on one chain.</remarks>
        private static T MulReduce<T>(T acc, T* x, long xs, long n) where T : unmanaged, IBinaryInteger<T>
        {
            T p0 = T.One, p1 = T.One, p2 = T.One, p3 = T.One;
            long i = 0;
            for (; i + 4 <= n; i += 4)
            {
                p0 *= x[i * xs];
                p1 *= x[(i + 1) * xs];
                p2 *= x[(i + 2) * xs];
                p3 *= x[(i + 3) * xs];
            }
            for (; i < n; i++)
                p0 *= x[i * xs];
            return acc * (p0 * p1 * (p2 * p3));
        }

        /// <summary>The multiply elementwise form: <c>o[i] = o[i] * x[i]</c> (wrapping).</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <param name="o">The out run.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="n">The element count.</param>
        private static void MulBinary<T>(T* o, long os, T* x, long xs, long n) where T : unmanaged, IBinaryInteger<T>
        {
            for (long i = 0; i < n; i++)
                o[i * os] *= x[i * xs];
        }

        /// <summary>The multiply reduce form over a masked call in one piece.</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <param name="acc">The out's running value.</param>
        /// <param name="x">The input's first element.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="m">The mask's first byte.</param>
        /// <param name="ms">The mask's byte stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The out's new value.</returns>
        private static T MulMaskedReduce<T>(T acc, T* x, long xs, byte* m, long ms, long n) where T : unmanaged, IBinaryInteger<T>
        {
            for (long i = 0; i < n; i++)
                if (m[i * ms] != 0)
                    acc *= x[i * xs];
            return acc;
        }

        /// <summary>The multiply elementwise form over a masked call: <c>o[i] *= x[i]</c> where <c>m[i]</c>.</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <param name="o">The out run.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="m">The mask's first byte.</param>
        /// <param name="ms">The mask's byte stride.</param>
        /// <param name="n">The element count.</param>
        private static void MulMaskedBinary<T>(T* o, long os, T* x, long xs, byte* m, long ms, long n) where T : unmanaged, IBinaryInteger<T>
        {
            for (long i = 0; i < n; i++)
                if (m[i * ms] != 0)
                    o[i * os] *= x[i * xs];
        }

        /// <summary>The multiply elementwise form over several consecutive rows into one out row.</summary>
        /// <typeparam name="T">The integer type.</typeparam>
        /// <param name="o">The out row.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The first input row.</param>
        /// <param name="xs">The input stride within a row.</param>
        /// <param name="rs">The distance from one input row to the next (0: a broadcast row).</param>
        /// <param name="rows">The row count.</param>
        /// <param name="n">The elements per row.</param>
        private static void MulBinaryRows<T>(T* o, long os, T* x, long xs, long rs, long rows, long n) where T : unmanaged, IBinaryInteger<T>
        {
            for (long r = 0; r < rows; r++)
                MulBinary(o, os, x + r * rs, xs, n);
        }
    }

    /// <summary>
    ///     NumPy's floating, complex and decimal multiply loops in the call forms of <see cref="NumPyReduceIterator.Loop{T}"/>
    ///     (see the file header for the NaN and complex forms each one keeps).
    /// </summary>
    internal static unsafe class NumPyMultiplyLoops
    {
        /// <summary>FLOAT_multiply / DOUBLE_multiply: the sequential reduce form and the elementwise form.</summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<T> Float<T>() where T : unmanaged, IFloatingPointIeee754<T>
            => new(&FloatReduce<T>, &FloatBinary<T>);

        /// <summary>HALF_multiply over raw float16 bits: float32 arithmetic, the running value narrowed once per call.</summary>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<ushort> Half() => new(&HalfReduce, &HalfBinary);

        /// <summary>CDOUBLE_multiply: MSVC's contracted loop_scalar for the reduce form, simd_cmul for the elementwise
        ///     form.</summary>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<Complex> Complex() => new(&ComplexReduce, &ComplexBinary);

        /// <summary>NumSharp's decimal multiply (no NumPy dtype): sequential decimal products.</summary>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<decimal> Decimal() => new(&DecimalReduce, &DecimalBinary);

        /// <summary>
        ///     The reduce form of FLOAT / DOUBLE_multiply (PW = 0: <c>acc = *out; for: acc *= in2; *out = acc</c>), in the
        ///     run's order. Once the running value is a NaN it stays that NaN: NumPy's compiled loop keeps the running value's
        ///     NaN when both operands are NaN (probed 2.4.2), so the first NaN of the sequence, quieted, is the answer.
        /// </summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        /// <param name="acc">The out's running value.</param>
        /// <param name="x">The run's first element.</param>
        /// <param name="xs">The run's stride in elements.</param>
        /// <param name="n">The run's element count.</param>
        /// <returns>The out's new value.</returns>
        /// <remarks>The NaN test is explicit because RyuJIT may swap a commutative multiply's operands, which would let a
        ///     later NaN's payload replace the running one.</remarks>
        private static T FloatReduce<T>(T acc, T* x, long xs, long n) where T : unmanaged, IFloatingPointIeee754<T>
        {
            for (long i = 0; i < n; i++)
            {
                if (T.IsNaN(acc))
                    return acc + acc;   // quiet (a signalling start value is quieted by its first multiply in NumPy)
                acc *= x[i * xs];
            }
            return acc;
        }

        /// <summary>
        ///     The elementwise form of FLOAT / DOUBLE_multiply, <c>o[i] = o[i] * x[i]</c>, keeping the SECOND operand's NaN
        ///     when both are NaN (NumPy's win-amd64 ufunc rule, <c>NaNOperandOrder.cs</c>).
        /// </summary>
        /// <typeparam name="T"><see cref="float"/> or <see cref="double"/>.</typeparam>
        /// <param name="o">The out run.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="n">The element count.</param>
        /// <remarks>A contiguous call runs a vector loop whose <c>(a &amp; (b == b)) * b</c> leaves the instruction no NaN
        ///     choice; a broadcast input (stride 0) takes the scalar statement, the second operand's NaN in every
        ///     position.</remarks>
        private static void FloatBinary<T>(T* o, long os, T* x, long xs, long n) where T : unmanaged, IFloatingPointIeee754<T>
        {
            long i = 0;
            if (os == 1 && xs == 1 && Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported)
            {
                int w = Vector256<T>.Count;
                for (; i + w <= n; i += w)
                {
                    var b = Vector256.Load(x + i);
                    Vector256.Store((Vector256.Load(o + i) & Vector256.Equals(b, b)) * b, o + i);
                }
            }
            for (; i < n; i++)
            {
                T a = o[i * os], b = x[i * xs];
                T r = a * b;
                if (T.IsNaN(r) && T.IsNaN(b))
                    r = b + b;
                o[i * os] = r;
            }
        }

        /// <summary>HALF_multiply's reduce form: <c>io1 = half_to_float(*out); for: io1 *= half_to_float(in2); *out =
        ///     float_to_half(io1)</c>, the running float32 value's NaN kept as in <see cref="FloatReduce{T}"/>.</summary>
        /// <param name="acc">The out's running value (float16 bits).</param>
        /// <param name="x">The run's first element (float16 bits).</param>
        /// <param name="xs">The run's stride in elements.</param>
        /// <param name="n">The run's element count.</param>
        /// <returns>The out's new value (float16 bits).</returns>
        private static ushort HalfReduce(ushort acc, ushort* x, long xs, long n)
        {
            float io1 = DirectILKernelGenerator.HalfToFloatScalarExact(acc);
            for (long i = 0; i < n; i++)
            {
                if (float.IsNaN(io1))
                {
                    io1 += io1;
                    break;
                }
                io1 *= DirectILKernelGenerator.HalfToFloatScalarExact(x[i * xs]);
            }
            return DirectILKernelGenerator.SingleToHalfBits(io1);
        }

        /// <summary>HALF_multiply's elementwise form: <c>float_to_half(half_to_float(o[i]) * half_to_float(x[i]))</c>, the
        ///     second operand's NaN kept (float16 keeps it on every path).</summary>
        /// <param name="o">The out run (float16 bits).</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run (float16 bits).</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="n">The element count.</param>
        /// <remarks>A contiguous out runs the house SIMD widen-compute-narrow kernel in place
        ///     (<see cref="DirectILKernelGenerator.HalfArithRun"/>, which supports that one aliasing); a strided out keeps the
        ///     scalar statement.</remarks>
        [SkipLocalsInit]
        private static void HalfBinary(ushort* o, long os, ushort* x, long xs, long n)
        {
            if (os == 1 && n > 0)
            {
                // The gather blocks for a strided input; each is written before it is read, so it is never zeroed.
                ushort* stageA = stackalloc ushort[DirectILKernelGenerator.HalfArithGatherBlock];
                ushort* stageB = stackalloc ushort[DirectILKernelGenerator.HalfArithGatherBlock];
                DirectILKernelGenerator.HalfArithRun(o, 1, x, xs, o, n, BinaryOp.Multiply, stageA, stageB);
                return;
            }
            for (long i = 0; i < n; i++)
            {
                float a = DirectILKernelGenerator.HalfToFloatScalarExact(o[i * os]);
                float b = DirectILKernelGenerator.HalfToFloatScalarExact(x[i * xs]);
                float r = a * b;
                if (float.IsNaN(r) && float.IsNaN(b))
                    r = b + b;
                o[i * os] = DirectILKernelGenerator.SingleToHalfBits(r);
            }
        }

        /// <summary>CDOUBLE_multiply's reduce form: NumPy jumps to its <c>loop_scalar</c> when the output stride is 0, which
        ///     MSVC compiles with contracted products (<see cref="ILKernelGenerator.PolyLoopScalarComplexMultiply"/>), one
        ///     element after the other.</summary>
        /// <param name="acc">The out's running value.</param>
        /// <param name="x">The run's first element.</param>
        /// <param name="xs">The run's stride in elements.</param>
        /// <param name="n">The run's element count.</param>
        /// <returns>The out's new value.</returns>
        private static Complex ComplexReduce(Complex acc, Complex* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                acc = ILKernelGenerator.PolyLoopScalarComplexMultiply(acc, x[i * xs]);
            return acc;
        }

        /// <summary>CDOUBLE_multiply's elementwise form: <c>simd_cmul</c> (<see cref="NDComplexMath.Multiply"/>), the form
        ///     NumPy runs for an in-place call (an output that IS the first input is not an overlap there).</summary>
        /// <param name="o">The out run.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="n">The element count.</param>
        private static void ComplexBinary(Complex* o, long os, Complex* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                o[i * os] = NDComplexMath.Multiply(o[i * os], x[i * xs]);
        }

        /// <summary>The decimal reduce form: the run multiplied into the running value one element after the other.</summary>
        /// <param name="acc">The out's running value.</param>
        /// <param name="x">The run's first element.</param>
        /// <param name="xs">The run's stride in elements.</param>
        /// <param name="n">The run's element count.</param>
        /// <returns>The out's new value.</returns>
        /// <exception cref="OverflowException">The product leaves decimal's range.</exception>
        private static decimal DecimalReduce(decimal acc, decimal* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                acc *= x[i * xs];
            return acc;
        }

        /// <summary>The decimal elementwise form: <c>o[i] = o[i] * x[i]</c>.</summary>
        /// <param name="o">The out run.</param>
        /// <param name="os">The out stride in elements.</param>
        /// <param name="x">The input run.</param>
        /// <param name="xs">The input stride in elements.</param>
        /// <param name="n">The element count.</param>
        /// <exception cref="OverflowException">A product leaves decimal's range.</exception>
        private static void DecimalBinary(decimal* o, long os, decimal* x, long xs, long n)
        {
            for (long i = 0; i < n; i++)
                o[i * os] *= x[i * xs];
        }
    }
}
