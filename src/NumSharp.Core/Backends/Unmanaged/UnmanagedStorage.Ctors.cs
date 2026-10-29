using System;
using System.Numerics;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        // ---------------------------------------------------------------------------------------------
        // ctors.c (np-layer dtype plan, Appendix D's UnmanagedStorage.Ctors.cs): PyArray_ArangeObj. np.arange's overloads
        // hand it Python ints (long) or Python floats (double); the length comes from Python's arithmetic on them, and the
        // array is filled by its dtype's slots - start and start + step through setitem, the rest through fill - as NumPy
        // fills it. No ufunc runs, so no TensorEngine member is involved.
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The ceiling of <c>(double)NPY_MAX_INTP</c>: <c>2**63</c>, the value the range check of
        ///     <c>_arange_safe_ceil_to_intp</c> rounds <c>NPY_MAX_INTP</c> to.
        /// </summary>
        private const double ArangeMaxIntpAsDouble = 9223372036854775808.0;

        /// <summary>NumPy's text when the length does not fit npy_intp (the OverflowError NumPy re-raises as a ValueError).</summary>
        private const string ArangeMaxSize = "Maximum allowed size exceeded";

        /// <summary>
        ///     NumPy's <c>PyArray_ArangeObj</c> (<c>ctors.c</c>) for Python-float bounds: the length from Python's float
        ///     arithmetic (<c>_calc_length</c>), then the array filled by its dtype's <c>setitem</c> and <c>fill</c> slots.
        /// </summary>
        /// <param name="start">The start, a Python float.</param>
        /// <param name="stop">The stop (excluded), a Python float.</param>
        /// <param name="step">The step, a Python float.</param>
        /// <param name="dtype">The requested dtype, or null for NumPy's discovery from Python floats: float64.</param>
        /// <returns>A fresh 1-D array of the dtype (its native form): <c>start</c>, <c>start + step</c> stored as Python
        ///     floats, then <c>start + i * delta</c> in the dtype's own arithmetic, with <c>delta</c> the difference of the
        ///     first two stored elements.</returns>
        /// <exception cref="NotSupportedException">The dtype's class has no NumSharp storage yet (datetime64 and timedelta64,
        ///     which NumPy hands to <c>datetime_arange</c>).</exception>
        /// <exception cref="TypeError">The dtype has no fill function (<c>S</c>, <c>U</c>, <c>StringDType</c>: <c>arange() not
        ///     supported for inputs with DType &lt;class 'numpy.dtypes.StrDType'&gt;.</c>), or it is bool and the result has
        ///     more than two elements (NumPy's <c>BOOL_fill</c>).</exception>
        /// <exception cref="DivideByZeroException">A zero step (NumPy's <c>ZeroDivisionError: float division by zero</c>).</exception>
        /// <exception cref="ValueError">A NaN length (<c>arange: cannot compute length</c>), a length past npy_intp
        ///     (<c>Maximum allowed size exceeded</c>), or an allocation past the byte limit (<see cref="AllocationGuard"/>'s
        ///     <c>array is too big</c>); a store's own ValueError (a NaN into an integer dtype).</exception>
        /// <exception cref="OverflowException">The first or second element does not fit the dtype (NumPy's
        ///     <c>Python integer 130 out of bounds for int8</c>, an infinity into an integer dtype).</exception>
        internal static NDArray ArangeObj(double start, double stop, double step, DType dtype)
        {
            var descr = ArangeDescr(dtype ?? NumSharp.DType.Double);
            long length = ArangeLength(start, stop, step);
            // _calc_length forms next = start + step only for a positive length; it is stored only for two elements or more.
            return ArangeFill(descr, length, start, length > 1 ? start + step : 0.0);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_ArangeObj</c> (<c>ctors.c</c>) for Python-int bounds: the length from Python's exact integer
        ///     arithmetic and correctly rounded true division (<c>_calc_length</c>), then the array filled by its dtype's
        ///     <c>setitem</c> and <c>fill</c> slots.
        /// </summary>
        /// <param name="start">The start, a Python int.</param>
        /// <param name="stop">The stop (excluded), a Python int.</param>
        /// <param name="step">The step, a Python int.</param>
        /// <param name="dtype">The requested dtype, or null for NumPy's discovery from Python ints within int64: int64.</param>
        /// <returns>A fresh 1-D array of the dtype (its native form): <c>start</c>, <c>start + step</c> stored as Python
        ///     ints, then <c>start + i * delta</c> in the dtype's own arithmetic.</returns>
        /// <exception cref="NotSupportedException">The dtype's class has no NumSharp storage yet.</exception>
        /// <exception cref="TypeError">The dtype has no fill function, or it is bool and the result has more than two
        ///     elements.</exception>
        /// <exception cref="DivideByZeroException">A zero step (NumPy's <c>ZeroDivisionError: division by zero</c>).</exception>
        /// <exception cref="ValueError">A length past npy_intp (<c>Maximum allowed size exceeded</c>), or an allocation past
        ///     the byte limit (<c>array is too big</c>).</exception>
        /// <exception cref="OverflowException">The first or second element does not fit the dtype.</exception>
        internal static NDArray ArangeObj(long start, long stop, long step, DType dtype)
        {
            var descr = ArangeDescr(dtype ?? NumSharp.DType.Int64);
            long length = ArangeLength((BigInteger)start, stop, step);
            // A Python int's sum is exact; with two elements or more it lies between start and stop, so it fits a long.
            return ArangeFill(descr, length, start, length > 1 ? (object)((BigInteger)start + step) : null);
        }

        /// <summary>
        ///     The checks <c>PyArray_ArangeObj</c> makes on the dtype before it computes anything: the datetime pair goes to
        ///     <c>datetime_arange</c> in NumPy (refused here: no storage yet), and a dtype whose class has no fill function
        ///     is a TypeError.
        /// </summary>
        /// <param name="dtype">The requested or discovered dtype.</param>
        /// <returns><paramref name="dtype"/>.</returns>
        /// <exception cref="NotSupportedException">The class has no NumSharp storage yet.</exception>
        /// <exception cref="TypeError">The class has no fill function.</exception>
        private static DType ArangeDescr(DType dtype)
        {
            if (!dtype.Meta.HasStorage)
                dtype.GetTypeCode(); // raises the descriptor layer's "no NumSharp storage yet" refusal
            // NumPy: if (!funcs->fill) "arange() not supported for inputs with DType %S." with the class object's repr.
            if (!dtype.Meta.HasFill)
                throw new TypeError($"arange() not supported for inputs with DType {dtype.Meta}.");
            return dtype;
        }

        /// <summary>
        ///     <c>_calc_length</c> over Python floats: <c>(stop - start) / step</c> in float arithmetic, an underflowed
        ///     quotient of a nonzero difference counted by its sign, else its ceiling as npy_intp.
        /// </summary>
        /// <param name="start">The start.</param>
        /// <param name="stop">The stop.</param>
        /// <param name="step">The step.</param>
        /// <returns>The length; zero or negative means an empty array.</returns>
        /// <exception cref="DivideByZeroException">A zero step (Python's <c>float division by zero</c>).</exception>
        /// <exception cref="ValueError">A NaN quotient, or a ceiling outside npy_intp.</exception>
        private static long ArangeLength(double start, double stop, double step)
        {
            double next = stop - start;
            // Python's comparison: a NaN difference is "nonzero".
            bool nextIsNonzero = next != 0;
            double value = PyScalar.FloatDiv(next, step);
            // NumPy's "underflow and divide-by-inf" check: a zero quotient of a NONZERO difference is one element for a
            // positive zero and none for a negative one.
            if (value == 0 && nextIsNonzero)
                return double.IsNegative(value) ? 0 : 1;
            return ArangeSafeCeilToIntp(value);
        }

        /// <summary>
        ///     <c>_calc_length</c> over Python ints: the exact difference, Python's correctly rounded true division by the
        ///     step, then the same tests as the float form.
        /// </summary>
        /// <param name="start">The start.</param>
        /// <param name="stop">The stop.</param>
        /// <param name="step">The step.</param>
        /// <returns>The length; zero or negative means an empty array.</returns>
        /// <exception cref="DivideByZeroException">A zero step (Python's <c>division by zero</c>).</exception>
        /// <exception cref="ValueError">A quotient past the float range or a ceiling outside npy_intp (NumPy re-raises both
        ///     OverflowErrors as <c>Maximum allowed size exceeded</c>).</exception>
        private static long ArangeLength(BigInteger start, BigInteger stop, BigInteger step)
        {
            BigInteger next = stop - start;
            double value;
            try
            {
                value = PyScalar.IntTrueDivide(next, step);
            }
            catch (OverflowException)
            {
                // PyArray_ArangeObj turns any OverflowError of _calc_length into this ValueError.
                throw new ValueError(ArangeMaxSize);
            }
            if (value == 0 && !next.IsZero)
                return double.IsNegative(value) ? 0 : 1;
            return ArangeSafeCeilToIntp(value);
        }

        /// <summary>
        ///     NumPy's <c>_arange_safe_ceil_to_intp</c>: the ceiling, refused when it is NaN or outside
        ///     <c>[(double)NPY_MIN_INTP, (double)NPY_MAX_INTP]</c>, then cast to npy_intp.
        /// </summary>
        /// <param name="value">The quotient <c>(stop - start) / step</c>.</param>
        /// <returns>The ceiling as a long; <see cref="long.MinValue"/> for a ceiling of exactly <c>2**63</c>.</returns>
        /// <exception cref="ValueError">A NaN (<c>arange: cannot compute length</c>), or a ceiling out of range (NumPy's
        ///     OverflowError, which <c>PyArray_ArangeObj</c> reports as <c>Maximum allowed size exceeded</c>).</exception>
        /// <remarks>
        ///     <c>(double)NPY_MAX_INTP</c> rounds to <c>2**63</c>, so a ceiling of exactly <c>2**63</c> passes the range check,
        ///     and C's <c>(npy_intp)</c> cast of it is <c>INT64_MIN</c> under x86-64's <c>cvttsd2si</c> (the "integer
        ///     indefinite" value): a negative length, which np.arange turns into an empty array (<c>np.arange(0, 2**63)</c> is
        ///     <c>[]</c> in NumPy on x86). .NET 9+ saturates that conversion instead, so the case is decided explicitly.
        /// </remarks>
        private static long ArangeSafeCeilToIntp(double value)
        {
            double ivalue = Math.Ceiling(value);
            // The condition inverted, as NumPy's, so a NaN fails it.
            if (double.IsNaN(ivalue))
                throw new ValueError("arange: cannot compute length");
            if (!(-ArangeMaxIntpAsDouble <= ivalue && ivalue <= ArangeMaxIntpAsDouble))
                throw new ValueError(ArangeMaxSize);
            return ivalue == ArangeMaxIntpAsDouble ? long.MinValue : (long)ivalue;
        }

        /// <summary>
        ///     The rest of <c>PyArray_ArangeObj</c>: allocate the result, store <paramref name="start"/> through the dtype's
        ///     <c>setitem</c>, then <paramref name="next"/>, then <c>fill</c> the remaining elements.
        /// </summary>
        /// <param name="dtype">The dtype (its class has a fill function).</param>
        /// <param name="length">The length; zero or less gives an empty array.</param>
        /// <param name="start">The first element's value (a Python float or int).</param>
        /// <param name="next">The second element's value, <c>start + step</c> (read only when the length is 2 or more).</param>
        /// <returns>The array.</returns>
        /// <exception cref="ValueError">The allocation is past the byte limit, or a store's ValueError.</exception>
        /// <exception cref="OverflowException">A store does not fit the dtype.</exception>
        /// <exception cref="TypeError">A store's TypeError, or bool's fill.</exception>
        /// <remarks>The allocation is released when a store or the fill fails: nothing else holds it yet, as NumPy's
        ///     <c>Py_DECREF(range)</c> on its failure path.</remarks>
        private static unsafe NDArray ArangeFill(DType dtype, long length, object start, object next)
        {
            if (length <= 0)
                length = 0;
            var range = new NDArray(dtype, Shape.Vector(length), false);
            if (length == 0)
                return range;
            try
            {
                // PyArray_DESCR(range): the descriptor the allocation installed (the native form of a byte-swapped request,
                // whose bytes NumSharp stores natively), and the buffer a StringDType class would need for its allocator.
                var descr = range.dtype;
                var meta = descr.Meta;
                var storage = range.Storage;
                byte* data = storage.Address;
                meta.setitem(descr, start, data, storage.InternalArray);
                if (length == 1)
                    return range;
                meta.setitem(descr, next, data + descr.itemsize, storage.InternalArray);
                if (length == 2)
                    return range;
                // HasFill was checked before the length; PyArray_Arange keeps this test for the same answer.
                if (!meta.fill(descr, data, length))
                    throw new ValueError("no fill-function for data-type.");
                return range;
            }
            catch
            {
                range.Dispose();
                throw;
            }
        }
    }
}
