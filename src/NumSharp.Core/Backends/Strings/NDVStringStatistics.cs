using System;
using System.Numerics;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     NumPy's compound statistics over NEP 55 <c>StringDType</c> text — <c>np.mean</c> / <c>np.average</c> /
    ///     <c>np.nanmean</c> (<c>_mean</c>), <c>np.var</c> / <c>np.std</c> / <c>np.nanvar</c> / <c>np.nanstd</c>
    ///     (<c>_var</c>), <c>np.ptp</c> (<c>_ptp</c>) — statement for statement: they COMPUTE their sum or extremes on the
    ///     text (<see cref="NDVStringReduction"/>) and then meet NumPy's next statement, which refuses in the words the
    ///     operand kinds decide.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumPy's Python layer sees a reduction's result in two forms: an ndarray (an axis left, <c>keepdims</c>, an
    ///     <c>out=</c>) or, for a 0-d result, the Python object the element is — a <c>str</c>, or the descriptor's NA object
    ///     (<c>None</c>, a <c>float</c> NaN, a sentinel string). The next statement then runs a ufunc on arrays
    ///     (<c>true_divide(ret, rcount, out=ret)</c>: <c>ufunc 'divide' did not contain a loop with signature matching types
    ///     (StringDType, Int64DType)</c>), or Python arithmetic on the object: <c>'abc' / np.intp(2)</c> becomes
    ///     <c>np.true_divide</c> of a <c>&lt;U3</c> array (NumPy's legacy refusal <c>ufunc 'divide' not supported for the
    ///     input types, ...</c>), <c>None / np.intp(2)</c> is Python's <c>unsupported operand type(s) for /: 'NoneType' and
    ///     'int'</c>, and a NaN divides into <c>np.float64(nan)</c>, which is the only answer that succeeds.
    ///     </para>
    ///     <para>
    ///     Each statement is reproduced by handing the same operands to NumSharp's ufuncs (whose no-loop and legacy refusals
    ///     carry NumPy's texts) or, for Python's own <c>TypeError</c> on <c>None</c>, by its text. NumSharp has no object
    ///     dtype: an operation NumPy would run on Python objects and that does NOT raise there cannot occur on these paths
    ///     (every <c>None</c> operand raises).
    ///     </para>
    /// </remarks>
    internal static partial class NDVStringStatistics
    {
        // =============================================================================================================
        // mean / average / nanmean
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>_mean</c> over a <c>StringDType</c> array: <c>rcount</c> (the reduced element count, an
        ///     <c>np.intp</c>), <c>umr_sum(arr, axis, dtype, keepdims)</c> — computed, with its loop's checks and NA rules —
        ///     then the division: <c>true_divide(ret, rcount, out=ret)</c> for an ndarray sum, Python's <c>ret / rcount</c>
        ///     for a 0-d one.
        /// </summary>
        /// <param name="op">The NumPy function (<c>"mean"</c>, <c>"average"</c>, <c>"nanmean"</c>) for the loop's checks.</param>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axes">The reduced axes, normalized by the caller (null for every axis; empty for none).</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null (any request has no <c>StringDType</c> loop).</param>
        /// <returns>The only mean that exists: <c>np.float64(nan)</c> (a read-only 0-d float64) for a 0-d sum that is a
        ///     NaN-like NA, or the quotient of another numeric NA object.</returns>
        /// <exception cref="TypeError">The divide's refusal (see the class remarks), or the add loop's (a <c>dtype=</c>
        ///     request, an <c>out</c> of another class).</exception>
        /// <exception cref="ValueError">Two or more reduced axes, an empty reduction, or the add loop's NA refusal.</exception>
        internal static NDArray Mean(string op, NDArray a, int[] axes, bool keepdims, DType dtype)
        {
            if (NDStringUfuncs.VStringReduceAnswer(op, a, axes, dtype) is { } refusal)
                throw refusal;
            long rcount = ReducedCount(a, axes);
            using var ret = SumOver(a, axes, keepdims);
            if (ret is null || ret.ndim > 0)
                // An ndarray sum: `um.true_divide(ret, rcount, out=ret, casting='unsafe')` — StringDType has no divide loop.
                throw DivideRefusal(ret);
            return PyDivide(ret.Storage.GetAtIndex(0), rcount);
        }

        /// <summary>
        ///     NumPy's <c>average</c> over a <c>StringDType</c> array once its axis tuple was normalized: without weights
        ///     <c>a.mean(axis, keepdims)</c> (<see cref="Mean"/>); with weights the shape checks first
        ///     (<c>_weights_are_valid</c>), then the result dtype <c>result_type(a.dtype, wgt.dtype)</c> (no common DType with
        ///     a number), then the weights' sum with that DESCRIPTOR as <c>dtype=</c> — which the ufunc refuses (a descriptor
        ///     instance, not a class).
        /// </summary>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axes">The normalized axes, or null (every axis; also a 0-d operand's tuple, see
        ///     <see cref="NDStringUfuncs.ReducedAxes(NDArray, int[])"/>).</param>
        /// <param name="axisGiven">Whether the caller passed an axis tuple (NumPy's <c>axis is not None</c>): with one, weights
        ///     of another shape meet the shape-of-the-axes check even where <paramref name="axes"/> is null (a 0-d
        ///     operand's tuple), as NumPy's <c>_weights_are_valid</c> tests the tuple, not the reduction.</param>
        /// <param name="weights">The weights, or null.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <returns>The mean (see <see cref="Mean"/>; only a NaN-like NA sum returns).</returns>
        /// <exception cref="DTypePromotionError">Weights of a class <c>StringDType</c> does not promote with.</exception>
        /// <exception cref="TypeError">See <see cref="Mean"/>; weights of another shape without an axis; the weights' sum
        ///     refusing the descriptor.</exception>
        /// <exception cref="ValueError">See <see cref="Mean"/>; weights inconsistent with the axis.</exception>
        internal static NDArray Average(NDArray a, int[] axes, bool axisGiven, NDArray weights, bool keepdims)
        {
            if (weights is null)
                return Mean("average", a, axes, keepdims, null);
            // _weights_are_valid: the shape checks run before anything promotes.
            if (!a.Shape.Equals(weights.Shape) && !SameDims(a, weights))
            {
                if (!axisGiven)
                    throw new TypeError("Axis must be specified when shapes of a and weights differ.");
                // A 0-d operand's tuple reduces every axis (axes is null) yet is an axis given: its weights must have the
                // shape of the tuple's axes — none, so any other shape is inconsistent.
                bool consistent = axes is null ? weights.ndim == 0 : weights.ndim == axes.Length;
                for (int i = 0; consistent && axes is not null && i < axes.Length; i++)
                    consistent = weights.Shape.dimensions[i] == a.Shape.dimensions[axes[i]];
                if (!consistent)
                    throw new ValueError("Shape of weights must be consistent with shape of a along specified axis.");
            }
            // `result_type = np.result_type(a.dtype, wgt.dtype)` (StringDType is not an integer or bool): the SEQUENCE
            // promotion, whose refusal lists every DType ("could not be promoted by ... The full list of DTypes is: ..."),
            // not the pairwise promote_types text.
            DType resultDtype = np.result_type(a.dtype, weights.dtype);
            // `scl = wgt.sum(axis=axis, dtype=result_dtype)`: a dtype= descriptor instance has no loop.
            int? axis = axes is { Length: 1 } ? axes[0] : null;
            using var scl = weights.TensorEngine.Sum(weights, axis, resultDtype, keepdims);
            throw new InvalidOperationException("The weighted average of StringDType text reached a sum NumPy refuses.");
        }

        // =============================================================================================================
        // var / std / nanvar / nanstd
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>_var</c> (and <c>_std</c>, which calls it first) over a <c>StringDType</c> array:
        ///     <c>arrmean = umr_sum(arr, axis, dtype, keepdims=True)</c> — computed — then <c>true_divide(arrmean, rcount,
        ///     out=arrmean)</c> for an ndarray mean, or Python's <c>arrmean / rcount</c> for a 0-d one (a 0-d input), whose
        ///     one successful form (a NaN-like NA) then meets <c>arr - arrmean</c>.
        /// </summary>
        /// <param name="op">The NumPy function (<c>"var"</c>, <c>"std"</c>, <c>"nanvar"</c>, <c>"nanstd"</c>).</param>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axes">The reduced axes, normalized by the caller (null for every axis).</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <returns>Never returns: every form reaches a refusal.</returns>
        /// <exception cref="TypeError">The divide's or the subtract's refusal, or the add loop's.</exception>
        /// <exception cref="ValueError">Two or more reduced axes, an empty reduction, or the add loop's NA refusal.</exception>
        internal static NDArray Var(string op, NDArray a, int[] axes, DType dtype)
        {
            if (NDStringUfuncs.VStringReduceAnswer(op, a, axes, dtype) is { } refusal)
                throw refusal;
            long rcount = ReducedCount(a, axes);
            using var arrmean = SumOver(a, axes, keepdims: true);
            if (arrmean is null || arrmean.ndim > 0)
                throw DivideRefusal(arrmean);
            // A 0-d input: the mean is the Python object `arrmean / rcount`, and `x = asanyarray(arr - arrmean)` follows.
            using var mean = PyDivide(arrmean.Storage.GetAtIndex(0), rcount);
            using var x = a.TensorEngine.Subtract(a, mean);
            throw new InvalidOperationException("np.var of StringDType text reached a subtraction NumPy refuses.");
        }

        // =============================================================================================================
        // ptp
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>_ptp</c> over a <c>StringDType</c> array: <c>subtract(maximum.reduce(a, axis, out=out, keepdims),
        ///     minimum.reduce(a, axis, keepdims), out)</c> — both extremes computed (with their loops' checks and NA rules),
        ///     then the subtraction of what Python holds: ndarrays (<c>StringDType</c> has no subtract loop), or the 0-d
        ///     results as Python objects (two <c>str</c> become <c>&lt;U</c> arrays and meet NumPy's legacy refusal naming both
        ///     descriptors; a NaN-like NA is a <c>float</c>; <c>None</c> raises Python's <c>TypeError</c>).
        /// </summary>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axis">The axis (validated by the caller), or null.</param>
        /// <param name="out">The output, or null (the maximum is written into it, so it is an ndarray operand).</param>
        /// <param name="keepdims">Keep the reduced axis.</param>
        /// <returns><c>np.float64(nan)</c> (a read-only 0-d float64) when both extremes are a NaN-like NA — the one
        ///     subtraction that exists.</returns>
        /// <exception cref="TypeError">The subtract's refusal, or Python's for <c>None</c>.</exception>
        /// <exception cref="ValueError">Two or more reduced axes, an empty reduction, or a comparison with a
        ///     <c>None</c>-like NA.</exception>
        internal static NDArray Ptp(NDArray a, int? axis, NDArray @out, bool keepdims)
        {
            var max = @out is null
                ? NDVStringReduction.MaxMin(true, a, axis, keepdims, null, null, null)
                : NDVStringReduction.MaxMinNumPy("max", a, axis, @out, keepdims, null, null);
            NDArray min = null;
            try
            {
                min = NDVStringReduction.MaxMin(false, a, axis, keepdims, null, null, null);
                // What Python holds: the out (always an ndarray) or an ndarray result as is, a 0-d result as its object.
                return SubtractExtremes(max, @out is not null || max.ndim > 0, min, min.ndim > 0, @out);
            }
            finally
            {
                min?.Dispose();
                if (!ReferenceEquals(max, @out))
                    max.Dispose();
            }
        }

        /// <summary>
        ///     <c>np.ptp(a, axis=())</c> over a <c>StringDType</c> array: <c>maximum.reduce</c> and <c>minimum.reduce</c> over
        ///     NO axis copy every element (the maximum into the out, after the out's shape check), so the subtraction meets
        ///     two ndarrays — or, for a 0-d input without an out, the element's Python object twice.
        /// </summary>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="out">The output, or null.</param>
        /// <returns><c>np.float64(nan)</c> for a 0-d NaN-like NA input, the one subtraction that exists.</returns>
        /// <exception cref="TypeError">The subtract's refusal, or Python's for <c>None</c>.</exception>
        /// <exception cref="ValueError">An out of the wrong rank or length.</exception>
        /// <exception cref="IncorrectShapeException">An out that does not broadcast.</exception>
        internal static NDArray PtpOverNoAxis(NDArray a, NDArray @out)
        {
            if (@out is not null)
            {
                // maximum.reduce(a, axis=(), out=out): the out's shape is the input's; CopyInitialReduceValues copies.
                ReductionArguments.ValidateOperands(a, new bool[a.ndim], false, @out, null, "maximum");
                UnmanagedStorage.Copyto(@out, a, NPY_CASTING.NPY_UNSAFE_CASTING);
            }
            return SubtractExtremes(@out ?? a, @out is not null || a.ndim > 0, a, a.ndim > 0, @out);
        }

        /// <summary>
        ///     <c>um.subtract(max, min, out)</c> of <c>_ptp</c> over what Python holds: two ndarrays meet the dispatcher
        ///     (no subtract loop for <c>StringDType</c>); a 0-d extreme is its Python object — two <c>str</c> become
        ///     <c>&lt;U</c> arrays (NumPy's legacy refusal naming both descriptors), a <c>float</c> NaN-like NA a weak Python
        ///     float, <c>None</c> beside another object Python's <c>TypeError</c> and beside an ndarray an object array that no
        ///     loop pairs with <c>StringDType</c>.
        /// </summary>
        /// <param name="max">The maximum (the out when one was given).</param>
        /// <param name="maxIsArray">Whether Python holds the maximum as an ndarray.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="minIsArray">Whether Python holds the minimum as an ndarray.</param>
        /// <param name="out">The output, or null.</param>
        /// <returns>The difference — only two NaN-like NA objects subtract (a read-only 0-d float64).</returns>
        /// <exception cref="TypeError">The refusal.</exception>
        private static NDArray SubtractExtremes(NDArray max, bool maxIsArray, NDArray min, bool minIsArray, NDArray @out)
        {
            object maxObj = maxIsArray ? null : max.Storage.GetAtIndex(0);
            object minObj = minIsArray ? null : min.Storage.GetAtIndex(0);
            if (!maxIsArray && !minIsArray && (maxObj is null || minObj is null))
                throw new TypeError($"unsupported operand type(s) for -: '{PyTypeName(maxObj)}' and '{PyTypeName(minObj)}'");
            if ((maxIsArray && !minIsArray && minObj is null) || (minIsArray && !maxIsArray && maxObj is null))
                // None beside an ndarray is an object array: no loop pairs it with StringDType.
                throw NDStringUfuncs.VStringNoLoop("subtract", new[]
                {
                    maxIsArray ? NDStringUfuncs.DispatchClass(max) : ObjectDTypeClass,
                    minIsArray ? NDStringUfuncs.DispatchClass(min) : ObjectDTypeClass,
                });
            using var x = maxIsArray ? null : PyOperand(maxObj);
            using var y = minIsArray ? null : PyOperand(minObj);
            NDArray minuend = x ?? max, subtrahend = y ?? min;
            var diff = minuend.TensorEngine.Subtract(minuend, subtrahend, null, @out);
            return @out is null ? diff.MarkReductionScalar() : diff;
        }

        // =============================================================================================================
        // numpy.ma's `dsum * 1.`
        // =============================================================================================================

        /// <summary>
        ///     numpy.ma's masked mean on a 0-d <c>StringDType</c> sum, <c>dsum * 1. / cnt</c> with <c>dsum</c> the Python
        ///     object of the sum (MaskedArray.sum returns the element): a <c>str</c> cannot be repeated by a float (<c>can't
        ///     multiply sequence by non-int of type 'float'</c>), <c>None</c> has no product, and a number (the NaN-like NA)
        ///     multiplies and divides into an <c>np.float64</c>.
        /// </summary>
        /// <param name="dsum">The sum's element (a <see cref="string"/>, null, or the NA object).</param>
        /// <param name="count">The count of unmasked elements (an <c>np.intp</c>).</param>
        /// <returns>The mean: a 0-d float64 (or complex128 for a complex NA object).</returns>
        /// <exception cref="TypeError">Python's refusal of the product.</exception>
        internal static NDArray MaskedMeanOfScalar(object dsum, long count)
        {
            switch (dsum)
            {
                case string:
                    throw new TypeError("can't multiply sequence by non-int of type 'float'");
                case null:
                    throw new TypeError("unsupported operand type(s) for *: 'NoneType' and 'float'");
            }
            if (!TryPyNumber(dsum, out double real, out Complex complex, out bool isComplex))
                throw new TypeError($"unsupported operand type(s) for *: '{PyTypeName(dsum)}' and 'float'");
            // `dsum * 1.` is a Python float (or complex); `/ cnt` with the np.intp count gives the NumPy scalar.
            return isComplex ? NDArray.Scalar(complex * 1.0 / count) : NDArray.Scalar(real * 1.0 / count);
        }

        // =============================================================================================================
        // shared statements
        // =============================================================================================================

        /// <summary>The repr NumPy's dispatcher prints for the object DType class (an object array operand).</summary>
        private const string ObjectDTypeClass = "<class 'numpy.dtypes.ObjectDType'>";

        /// <summary>
        ///     <c>_count_reduce_items</c>: the number of elements each output reduces (the product of the reduced extents; 1
        ///     for no axis), the <c>np.intp</c> the statistics divide by.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="axes">The normalized axes, or null for every axis.</param>
        /// <returns>The count.</returns>
        private static long ReducedCount(NDArray a, int[] axes)
        {
            if (axes is null)
                return a.size;
            long n = 1;
            foreach (int ax in axes)
                n *= a.Shape.dimensions[ax];
            return n;
        }

        /// <summary>
        ///     <c>umr_sum(arr, axis, keepdims=keepdims)</c> once the loop's checks passed: the computed sum along the one axis
        ///     (or flat), or null for a reduction over NO axis of an array with dimensions (an ndarray NumPy only copies, which
        ///     the statistics never read: only its being an ndarray decides the next statement).
        /// </summary>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axes">The normalized axes (at most one: the checks refused more), null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <returns>The sum (owned by the caller), or null as described.</returns>
        private static NDArray SumOver(NDArray a, int[] axes, bool keepdims)
        {
            if (axes is { Length: 0 })
                // add.reduce over no axis copies every element; a 0-d input stays 0-d — its element is the sum's object.
                return a.ndim == 0 ? NDVStringReduction.Sum(a, null, keepdims, null) : null;
            return NDVStringReduction.Sum(a, axes is null ? null : axes[0], keepdims, null);
        }

        /// <summary>
        ///     <c>um.true_divide(ret, rcount, out=ret, casting='unsafe')</c> on an ndarray <c>StringDType</c> sum: the
        ///     dispatcher has no divide loop for (StringDType, Int64DType).
        /// </summary>
        /// <param name="ret">The sum (unused when null: the class names the operands).</param>
        /// <returns>The refusal (not thrown).</returns>
        private static TypeError DivideRefusal(NDArray ret)
            => NDStringUfuncs.VStringNoLoop("divide", new[] { DTypeRegistry.VString.ToString(), DTypeRegistry.FromTypeCode(NPTypeCode.Int64).ToString() });

        /// <summary>
        ///     Python's <c>value / np.intp(rcount)</c> for the object of a 0-d <c>StringDType</c> sum: a <c>str</c> becomes
        ///     <c>np.true_divide</c> of a <c>&lt;U</c> array (NumPy's legacy refusal), <c>None</c> reaches the object loop's
        ///     <c>None / int</c>, a number divides.
        /// </summary>
        /// <param name="value">The element (a <see cref="string"/>, null, or the NA object).</param>
        /// <param name="rcount">The count.</param>
        /// <returns>The quotient: a read-only 0-d float64 (complex128 for a complex NA object).</returns>
        /// <exception cref="TypeError">The refusal.</exception>
        private static NDArray PyDivide(object value, long rcount)
        {
            switch (value)
            {
                case string s:
                {
                    using var text = NDArray.PythonStrLiteral(s);
                    using var count = NDArray.Scalar(rcount);
                    // np.int64.__rtruediv__ hands the str to np.true_divide, whose legacy resolver refuses (U, int64).
                    using var quotient = text.TensorEngine.Divide(text, count);
                    throw new InvalidOperationException("np.true_divide of text and an integer returned.");
                }
                case null:
                    throw new TypeError("unsupported operand type(s) for /: 'NoneType' and 'int'");
            }
            if (!TryPyNumber(value, out double real, out Complex complex, out bool isComplex))
                throw new TypeError($"unsupported operand type(s) for /: '{PyTypeName(value)}' and 'int'");
            return (isComplex ? NDArray.Scalar(complex / rcount) : NDArray.Scalar(real / rcount)).MarkReductionScalar();
        }

        /// <summary>
        ///     The operand a Python object of a 0-d extreme is in <c>um.subtract(max, min)</c>: a <c>str</c> is a Python str
        ///     (<c>&lt;U</c>, or <c>StrDType</c> beside a <c>StringDType</c> array), a <c>float</c> a weak Python float, any
        ///     other number its NumPy scalar.
        /// </summary>
        /// <param name="value">The element (a <see cref="string"/> or a numeric NA object; never null).</param>
        /// <returns>A fresh 0-d operand (owned by the caller).</returns>
        /// <exception cref="TypeError">An NA object of a type NumSharp has no array for.</exception>
        private static NDArray PyOperand(object value)
        {
            switch (value)
            {
                case string s:
                    return NDArray.PythonStrLiteral(s);
                case double d:
                    return NDArray.PythonLiteral(PyFloatValue(d));
                case float f:
                    return NDArray.PythonLiteral(PyFloatValue(f));
                case bool b:
                    return NDArray.PythonBoolLiteral(b);
                case long l:
                    return NDArray.PythonLiteral(l);
                case int i:
                    return NDArray.PythonLiteral((long)i);
                case Complex c:
                    return NDArray.PythonLiteral(c);
            }
            throw new TypeError($"unsupported operand type(s) for -: '{PyTypeName(value)}' and '{PyTypeName(value)}'");
        }

        /// <summary>
        ///     The Python float a C# NA object stands for: .NET's <see cref="double.NaN"/> (whose sign bit is SET) is NumPy's
        ///     <c>float('nan')</c> / <c>np.nan</c>, the POSITIVE quiet NaN <c>0x7ff8000000000000</c> — so a NaN-like NA mean is
        ///     <c>np.float64(nan)</c> with NumPy's bits; any other value (a NaN with another payload included) is itself.
        /// </summary>
        /// <param name="d">The NA object's value.</param>
        /// <returns>The value Python's arithmetic starts from.</returns>
        private static double PyFloatValue(double d)
            => BitConverter.DoubleToInt64Bits(d) == BitConverter.DoubleToInt64Bits(double.NaN)
                ? BitConverter.Int64BitsToDouble(0x7FF8000000000000L)
                : d;

        /// <summary>A Python number's value: an int / bool / float as a double, a complex as itself.</summary>
        /// <param name="value">The object.</param>
        /// <param name="real">Receives the real value.</param>
        /// <param name="complex">Receives the complex value.</param>
        /// <param name="isComplex">Receives whether the object is complex.</param>
        /// <returns>True for a number.</returns>
        private static bool TryPyNumber(object value, out double real, out Complex complex, out bool isComplex)
        {
            isComplex = false;
            complex = default;
            switch (value)
            {
                case double d: real = PyFloatValue(d); return true;
                case float f: real = PyFloatValue(f); return true;
                case long l: real = l; return true;
                case int i: real = i; return true;
                case bool b: real = b ? 1 : 0; return true;
                case BigInteger bi: real = (double)bi; return true;
                case Complex c:
                    real = 0;
                    complex = c;
                    isComplex = true;
                    return true;
            }
            real = 0;
            return false;
        }

        /// <summary>The Python type name of an element object (<c>str</c>, <c>NoneType</c>, <c>float</c>, ...).</summary>
        /// <param name="value">The object.</param>
        /// <returns>The name.</returns>
        private static string PyTypeName(object value) => value switch
        {
            null => "NoneType",
            string => "str",
            double or float => "float",
            bool => "bool",
            long or int or BigInteger => "int",
            Complex => "complex",
            _ => Unmanaged.NDBuffer.PythonTypeName(value),
        };

        /// <summary>Whether two arrays have the same dimensions.</summary>
        /// <param name="x">The first array.</param>
        /// <param name="y">The second array.</param>
        /// <returns>True for equal ranks and extents.</returns>
        private static bool SameDims(NDArray x, NDArray y)
        {
            if (x.ndim != y.ndim)
                return false;
            for (int i = 0; i < x.ndim; i++)
                if (x.Shape.dimensions[i] != y.Shape.dimensions[i])
                    return false;
            return true;
        }
    }
}
