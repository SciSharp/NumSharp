using System;
using System.Numerics;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    ///     The NEP 50 weak Python scalar of the six comparison ufuncs (<c>equal</c>, <c>not_equal</c>, <c>less</c>,
    ///     <c>less_equal</c>, <c>greater</c>, <c>greater_equal</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumPy 2.x promotes a Python <c>int</c>, <c>float</c> or <c>complex</c> as a WEAK scalar: the comparison runs at
    ///     the array's own precision, and the scalar is converted into that precision first. So
    ///     <c>np.array([0.1, 0.5], np.float32) &gt; 0.1</c> is <c>[False, True]</c>: <c>0.1</c> becomes
    ///     <c>np.float32(0.1)</c>, which equals the first element. Comparing in float64 instead, as NumSharp did before,
    ///     gives <c>[True, True]</c>, because <c>(double)0.1f</c> is 0.10000000149… and that is greater than 0.1.
    ///     </para>
    ///     <para>
    ///     The Python scalar is the 0-d array a C# number literal became (<see cref="NDArray.WasPythonLiteral"/>: the integer
    ///     primitives, <c>float</c>, <c>double</c> and <see cref="Complex"/>), the rule NumSharp's arithmetic uses
    ///     (<see cref="np._FindCommonType(NDArray, NDArray)"/>). Every other 0-d operand is a NumPy scalar or 0-d array and
    ///     compares STRONG, as in NumPy 2: an element read, a reduction result, <c>np.array(0.1)</c>
    ///     (<c>np.greater(f32, np.array(0.1))</c> compares in float64 and is <c>[True, True]</c>), <see cref="Half"/>,
    ///     <c>char</c> and <c>decimal</c> values. A literal against a strong 0-d operand is weak too
    ///     (<c>np.float32(0.1) &gt; 0.1</c> is False); two literals keep the strong promotion between their C# dtypes, as in
    ///     NumSharp's arithmetic. numpy.ma keeps its own rule: <c>getdata</c> makes every operand an array, so its
    ///     comparison wrappers widen both operands before they call the ufunc.
    ///     </para>
    ///     <para>
    ///     Only float16 and float32 arrays change values. For every other array dtype the comparison NumSharp already runs
    ///     is NumPy's:
    ///     <list type="bullet">
    ///     <item>An integer or bool array against a Python int compares EXACTLY in NumPy 2.x, in range or not
    ///     (<c>special_integer_comparisons.cpp</c>: <c>int8 &gt; 1000</c> is all False, <c>uint64 &gt; -1</c> all True),
    ///     and the strong promotion NumSharp uses (with the exact int64 / uint64 pair at Decimal) is exact for every
    ///     integer pair. An int that fits an integer array's dtype is still re-packed into that dtype, as NumPy's
    ///     in-range branch casts it: the values are the same, and the comparison then runs the same-dtype vector
    ///     kernel instead of widening every element.</item>
    ///     <item>An integer or bool array against a Python float or complex compares at float64 or complex128 in both
    ///     libraries.</item>
    ///     <item>A float64 or complex128 array holds every Python float, and every Python int rounds into it the same way
    ///     (correct rounding, <c>(double)long</c> / <c>(double)ulong</c>).</item>
    ///     </list>
    ///     </para>
    ///     <para>
    ///     The conversion is NumPy's <c>PyArray_Pack</c> into the loop dtype (<c>MyPyFloat_AsFloat</c> /
    ///     <c>MyPyFloat_AsHalf</c> / <c>CFLOAT_setitem</c> in <c>arraytypes.c.src</c>): a Python int first becomes the
    ///     correctly rounded double (<c>PyNumber_Float</c>), then narrows. That is TWO roundings for an int past 2^53:
    ///     <c>2**60 + 2**36 + 1</c> packs into float32 as 2^60 where a direct conversion gives 2^60 + 2^37. A value past
    ///     the dtype's range becomes infinity (NumPy warns, it does not raise), so <c>np.array([np.inf], np.float16) ==
    ///     70000</c> is True. A Python complex against a float16 or float32 array compares as complex64: each part rounds
    ///     to float32. NumSharp has one complex width, so the comparison stays at complex128 with the parts rounded, which
    ///     is exact (widening a float32 to double keeps every order and equality).
    ///     </para>
    /// </remarks>
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Converts the weak operand of a comparison (a C# literal) into the precision NumPy compares it at, when that
        ///     precision is not the one the operands' own dtypes promote to — see the type remarks.
        /// </summary>
        /// <param name="lhs">The left operand.</param>
        /// <param name="rhs">The right operand.</param>
        /// <param name="scalarIsLhs">True when the returned array replaces <paramref name="lhs"/>, false when it replaces
        ///     <paramref name="rhs"/>; unspecified when the method returns null.</param>
        /// <returns>A new 0-d array that holds the scalar in the comparison's loop dtype (float16, float32, or complex128
        ///     with float32-rounded parts), or in an integer array's own dtype when an integer 0-d operand fits it; null
        ///     when the operands stand as given. The caller owns the array and disposes it after the comparison.</returns>
        /// <exception cref="OverflowException">A bool operand against a Python int at or past 2^63: NumPy's int64 pack
        ///     fails ("Python int too large to convert to C long", the LP64 text).</exception>
        /// <remarks>
        ///     <para>
        ///     The weak operand is a literal-marked 0-d array (<see cref="NDArray.WasPythonLiteral"/>) against a strong one
        ///     (an array, or a strong 0-d array). Without exactly one, only the exact integer re-pack below runs, for a
        ///     strong 0-d integer against an integer array: NumPy compares such a pair in the promoted dtype, exactly, which
        ///     the re-packed same-dtype compare reproduces value for value.
        ///     </para>
        ///     <para>
        ///     The replacement is a same-dtype operand for the array, so the comparison also takes the faster same-dtype
        ///     kernels (no per-element conversion of the array; the scalar-broadcast kernels then compare a whole vector at
        ///     a time). The integer re-pack changes no value: an in-range int compares the same in the array's dtype as in
        ///     the promoted one. The scalar is read once, through a boxed read of its single element; the conversion is per
        ///     call, not per element.
        ///     </para>
        /// </remarks>
        internal NDArray AdoptWeakComparisonScalar(NDArray lhs, NDArray rhs, out bool scalarIsLhs)
        {
            scalarIsLhs = false;
            bool lhsWeak = lhs.WasPythonLiteral, rhsWeak = rhs.WasPythonLiteral;
            if (lhsWeak == rhsWeak)
            {
                // No weak operand, or two C# literals (strong between their C# dtypes, NumSharp's arithmetic rule): the
                // promoted compare stands, except the exact re-pack of a strong 0-d integer against an integer array.
                bool lhsScalar = lhs.ndim == 0, rhsScalar = rhs.ndim == 0;
                if (lhsWeak || lhsScalar == rhsScalar)
                    return null;
                var fitted = RepackFittingIntScalar((lhsScalar ? rhs : lhs).GetTypeCode, lhsScalar ? lhs : rhs);
                scalarIsLhs = lhsScalar;
                return fitted;
            }

            // Exactly one weak operand: the literal; the other (an array, or a strong 0-d array) sets the precision.
            var array = lhsWeak ? rhs : lhs;
            var scalar = lhsWeak ? lhs : rhs;
            var arrayType = array.GetTypeCode;
            var scalarType = scalar.GetTypeCode;
            // A bool array has no exact Python-int comparison (only integer arrays do): NumPy compares it in the int64
            // loop, whose pack fails for an int at or past 2^63 ("Python int too large to convert to C long").
            if (arrayType == NPTypeCode.Boolean && IsPythonIntType(scalarType))
                NDExprTypeRules.CheckPyIntPacksIntoLoop(ReadPythonInt(scalar), NPTypeCode.Boolean);
            // An integer array against a Python int of another width: NumPy casts an int that FITS the array's dtype into
            // that dtype and runs the array's own loop (special_integer_comparisons.cpp, the in-range branch). The values
            // are the ones the exact promoted compare gives, so no result changes; the comparison becomes a same-dtype one,
            // which the scalar-broadcast kernels vectorize (uint8 > 128 compared every byte as an int32 before, one element
            // at a time). An int outside the range keeps the exact promoted compare, which gives NumPy's constant result.
            if (IsPythonIntType(arrayType) && IsPythonIntType(scalarType) && scalarType != arrayType)
            {
                scalarIsLhs = lhsWeak;
                return RepackFittingIntScalar(arrayType, scalar);
            }
            // Only a float16 / float32 array compares narrower than the promotion (see the remarks); a scalar of the
            // array's own dtype is already the loop dtype.
            if ((arrayType != NPTypeCode.Half && arrayType != NPTypeCode.Single) || scalarType == arrayType)
                return null;
            if (!IsPythonScalarType(scalarType))
                return null;

            scalarIsLhs = lhsWeak;
            return PackPythonScalarForFloatLoop(scalar.Storage.GetAtIndex(0), arrayType);
        }

        /// <summary>
        ///     A 0-d integer operand re-packed into an integer array's own dtype when its value fits that dtype: the exact
        ///     promoted comparison's answers, through the same-dtype kernels.
        /// </summary>
        /// <param name="arrayType">The other operand's dtype (the re-pack applies only to the eight integer dtypes).</param>
        /// <param name="scalar">The 0-d operand.</param>
        /// <returns>A new 0-d array of <paramref name="arrayType"/> the caller owns, or null when either dtype is not an
        ///     integer, they are equal, or the value is out of <paramref name="arrayType"/>'s range (the promoted compare
        ///     then gives NumPy's constant answer).</returns>
        private NDArray RepackFittingIntScalar(NPTypeCode arrayType, NDArray scalar)
        {
            var scalarType = scalar.GetTypeCode;
            if (!IsPythonIntType(arrayType) || !IsPythonIntType(scalarType) || scalarType == arrayType)
                return null;
            object boxed = scalar.Storage.GetAtIndex(0);
            var (min, max) = NDExprTypeRules.IntBounds(arrayType);
            Int128 v = boxed is ulong u ? (Int128)u : (Int128)Convert.ToInt64(boxed, System.Globalization.CultureInfo.InvariantCulture);
            if (v < min || v > max)
                return null;
            return NewScalar(boxed, arrayType);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_Pack</c> of a Python scalar into the precision of a float16 / float32 array: the 0-d
        ///     operand a comparison (or numpy.ma's comparison with a Python value) runs against that array.
        /// </summary>
        /// <param name="value">The boxed C# value standing for the Python scalar: a <c>bool</c>, an integer primitive, a
        ///     <see cref="BigInteger"/> (a Python int of any size), a <c>float</c> / <c>double</c>, or a
        ///     <see cref="Complex"/>.</param>
        /// <param name="arrayType">The array's dtype: <see cref="NPTypeCode.Half"/> or <see cref="NPTypeCode.Single"/>.</param>
        /// <returns>A new 0-d array the caller owns: of <paramref name="arrayType"/> for a real value, or complex128 with
        ///     both parts rounded to float32 for a complex one (the complex64 NumPy compares at).</returns>
        /// <exception cref="OverflowException">A <see cref="BigInteger"/> past the double range: CPython's
        ///     <c>float(int)</c> text, "int too large to convert to float".</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="arrayType"/> is neither float16 nor float32 — a
        ///     caller bug.</exception>
        /// <remarks>
        ///     A real value goes through <c>PyNumber_Float</c> then narrows (<c>MyPyFloat_AsFloat</c> /
        ///     <c>MyPyFloat_AsHalf</c>): an int is the correctly rounded double first (<see cref="Converts.ToDouble(object)"/>
        ///     rounds a <c>ulong</c> once on every runtime, <see cref="PythonInt.ToDouble"/> a <see cref="BigInteger"/>), so
        ///     an int past 2^53 is rounded twice, as in NumPy; float16 narrows with NumPy's <c>npy_double_to_half</c>; a value
        ///     past the dtype's range becomes infinity (NumPy warns, it does not raise). A complex value is
        ///     <c>CFLOAT_setitem</c>: each part narrows to float32. The float16 / float32 array widens exactly into
        ///     complex64, so comparing complex128 values with float32 parts IS the complex64 loop (widening keeps every
        ///     order and equality).
        /// </remarks>
        internal static NDArray PackPythonScalarForFloatLoop(object value, NPTypeCode arrayType)
        {
            if (arrayType != NPTypeCode.Half && arrayType != NPTypeCode.Single)
                throw new ArgumentOutOfRangeException(nameof(arrayType), arrayType, "PackPythonScalarForFloatLoop: float16 or float32 only — caller bug.");

            if (value is Complex z)
                return NDArray.Scalar(new Complex((float)z.Real, (float)z.Imaginary));

            double d;
            if (value is BigInteger bi)
            {
                d = PythonInt.ToDouble(bi);
                // PythonInt.ToDouble saturates; CPython's float(int) raises past the double range instead.
                if (double.IsInfinity(d))
                    throw new OverflowException("int too large to convert to float");
            }
            else
            {
                // A Python bool is 0.0 / 1.0, an int its correctly rounded double, a float itself.
                d = Converts.ToDouble(value);
            }

            return arrayType == NPTypeCode.Single
                ? NDArray.Scalar((float)d)
                : NDArray.Scalar(BitConverter.UInt16BitsToHalf(DirectILKernelGenerator.DoubleToHalfBits(d)));
        }

        /// <summary>
        ///     The 0-d integer operand of a float16 / float32 ufunc loop re-packed the way NumPy re-packs a weak Python int
        ///     (<c>npy_update_operand_for_scalar</c>, then <c>PyArray_Pack</c>), or null when the cast the kernels apply gives
        ///     the same value.
        /// </summary>
        /// <param name="operand">A ufunc operand (any shape; only a literal-marked 0-d one,
        ///     <see cref="NDArray.WasPythonLiteral"/>, is a weak Python int), or null.</param>
        /// <param name="loop">The ufunc's final loop dtype.</param>
        /// <returns>A new 0-d array of <paramref name="loop"/> that the caller owns and disposes, or null.</returns>
        /// <remarks>
        ///     NumPy packs a Python int into a float loop through the correctly rounded double, so an int past 2^53 rounds
        ///     TWICE: <c>np.zeros(1, np.float32) + (2**60 + 2**36 + 1)</c> is 2^60, where a direct conversion (the array
        ///     cast, <c>np.array([2**60 + 2**36 + 1]).astype(np.float32)</c>) gives 2^60 + 2^37. Every integer of at most 53
        ///     bits converts exactly into a double, so only an int64 / uint64 operand past 2^53 in magnitude can differ, and
        ///     the common path allocates nothing. A float operand needs no repack: the cast from a double rounds once, like
        ///     the pack. A strong 0-d integer (a NumPy scalar) is CAST into the loop like any array, so it is left alone.
        /// </remarks>
        internal NDArray RepackWeakIntForFloatLoop(NDArray operand, NPTypeCode loop)
        {
            if ((loop != NPTypeCode.Half && loop != NPTypeCode.Single) || operand is null || operand.ndim != 0)
                return null;
            // A strong 0-d integer is cast into the loop like any array: only a C# literal is NumPy's Python int.
            if (!operand.WasPythonLiteral)
                return null;
            var type = operand.GetTypeCode;
            if (type != NPTypeCode.Int64 && type != NPTypeCode.UInt64)
                return null;
            object value = operand.Storage.GetAtIndex(0);
            const long Exact = 1L << 53;
            bool pastExact = type == NPTypeCode.Int64
                ? (long)value is < -Exact or > Exact
                : (ulong)value > Exact;
            return pastExact ? PackPythonScalarForFloatLoop(value, loop) : null;
        }

        /// <summary>
        ///     NumPy's range check of a weak Python int packed into an integer loop: the setitem
        ///     <c>npy_update_operand_for_scalar</c> runs for a Python-literal operand raises instead of wrapping
        ///     (<c>np.int8([1, 2]) + 1000</c> is <c>OverflowError: Python integer 1000 out of bounds for int8</c>).
        /// </summary>
        /// <param name="lhs">The left operand.</param>
        /// <param name="rhs">The right operand.</param>
        /// <param name="loop">The ufunc's final loop dtype.</param>
        /// <param name="loopFromPromotion">True when <paramref name="loop"/> came from promotion (no <c>dtype=</c>): a
        ///     bool array then meets the Python int in NumPy's int64 loop, whatever dtype NumSharp's table names.</param>
        /// <exception cref="OverflowException">The 0-d Python-int operand does not fit the loop dtype (NumPy's texts, LP64:
        ///     <see cref="NDExprTypeRules.CheckPyIntPacksIntoLoop"/>).</exception>
        /// <remarks>
        ///     <para>
        ///     The Python int is a literal-marked 0-d operand (<see cref="NDArray.WasPythonLiteral"/>: the C# integer
        ///     primitives) against a strong one — an array, or a strong 0-d array (<c>np.int8(1) + 1000</c> raises in NumPy
        ///     too). A bool literal is a Python bool (always fits), a <c>char</c> is NumPy's strong uint16 scalar, a strong
        ///     0-d integer (an element read, a reduction result, <c>np.array(1000)</c>) promotes instead of packing, and a
        ///     float loop rounds an int instead of checking it. The kernels used to CAST the operand into the loop, which
        ///     wrapped silently (<c>int8 + 1000</c> gave <c>[-23, -22]</c>).
        ///     </para>
        ///     <para>
        ///     A bool array is the one case where NumPy's loop and NumSharp's table differ: NumPy promotes bool with a
        ///     Python int to its default int64, while the table keeps a C# <c>ulong</c> literal's uint64. The values agree
        ///     below 2^63; at and past it NumPy's int64 pack fails ("Python int too large to convert to C long").
        ///     </para>
        /// </remarks>
        internal static void CheckWeakPyIntOperand(NDArray lhs, NDArray rhs, NPTypeCode loop, bool loopFromPromotion)
        {
            // Exactly one weak operand: two literals promote between their own C# dtypes (nothing to pack), and two strong
            // operands are cast into the loop.
            bool lhsWeak = lhs.WasPythonLiteral;
            if (lhsWeak == rhs.WasPythonLiteral)
                return;
            var scalar = lhsWeak ? lhs : rhs;
            if (!IsPythonIntType(scalar.GetTypeCode))
                return;
            var array = lhsWeak ? rhs : lhs;
            var packLoop = loopFromPromotion && array.GetTypeCode == NPTypeCode.Boolean && NDExprTypeRules.IsIntegerKind(loop)
                ? NPTypeCode.Int64
                : loop;
            if (!NDExprTypeRules.IsIntegerKind(packLoop) && packLoop != NPTypeCode.Boolean)
                return;
            NDExprTypeRules.CheckPyIntPacksIntoLoop(ReadPythonInt(scalar), packLoop);
        }

        /// <summary>Whether a 0-d operand of <paramref name="type"/> stands for a Python int: the eight C# integer
        ///     primitives (not bool, a Python bool; not char, NumPy's strong uint16 scalar).</summary>
        /// <param name="type">The 0-d operand's dtype.</param>
        /// <returns>True for int8 through uint64.</returns>
        private static bool IsPythonIntType(NPTypeCode type)
            => type is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16
                or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64;

        /// <summary>The value of a 0-d integer operand, widened exactly into 128 bits (every C# integer fits).</summary>
        /// <param name="scalar">A 0-d array of one of the eight integer dtypes.</param>
        /// <returns>The value.</returns>
        private static Int128 ReadPythonInt(NDArray scalar)
        {
            object v = scalar.Storage.GetAtIndex(0);
            return v is ulong u ? (Int128)u : (Int128)Convert.ToInt64(v, System.Globalization.CultureInfo.InvariantCulture);
        }

        /// <summary>
        ///     Replaces the weak-int operand of a binary ufunc by its re-packed form for the final float loop
        ///     (<see cref="RepackWeakIntForFloatLoop"/>), when exactly one operand is a literal-marked 0-d array
        ///     (<see cref="NDArray.WasPythonLiteral"/>) and the pack differs from the cast.
        /// </summary>
        /// <param name="lhs">The left operand; replaced when it is the re-packed one.</param>
        /// <param name="rhs">The right operand; replaced when it is the re-packed one.</param>
        /// <param name="loop">The ufunc's final loop dtype.</param>
        /// <returns>The replacement, which the caller disposes after the ufunc ran, or null when nothing changed.</returns>
        /// <remarks>The other operand may be an array or a strong 0-d array (<c>np.float32(0) + (2**60 + 2**36 + 1)</c>
        ///     packs the int as NumPy does); two literals promote between their C# dtypes and are cast.</remarks>
        internal NDArray AdoptWeakIntOperand(ref NDArray lhs, ref NDArray rhs, NPTypeCode loop)
        {
            bool lhsWeak = lhs.WasPythonLiteral;
            // Exactly one weak operand: two literals promote between their C# dtypes and are cast, two strong ones too.
            if (lhsWeak == rhs.WasPythonLiteral)
                return null;
            var packed = RepackWeakIntForFloatLoop(lhsWeak ? lhs : rhs, loop);
            if (packed is null)
                return null;
            if (lhsWeak)
                lhs = packed;
            else
                rhs = packed;
            return packed;
        }

        /// <summary>
        ///     Whether a 0-d operand of <paramref name="type"/> stands for a Python scalar (weak under NEP 50): the C#
        ///     types that have a Python literal. <see cref="Half"/>, <c>char</c>, <c>decimal</c> and text are NumPy
        ///     scalars (or have no NumPy analog) and stay strong.
        /// </summary>
        /// <param name="type">The 0-d operand's dtype.</param>
        /// <returns>True for bool, the eight integer dtypes, float32, float64 and complex128.</returns>
        private static bool IsPythonScalarType(NPTypeCode type)
            => type is NPTypeCode.Boolean or NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16
                or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Single
                or NPTypeCode.Double or NPTypeCode.Complex;

        /// <summary>
        ///     Runs one comparison ufunc after its argument checks: the weak-scalar conversion, then the bare route
        ///     (<see cref="ExecuteComparisonOp"/>) or the <c>out=</c> / <c>where=</c> route
        ///     (<see cref="ExecuteComparisonUfuncInto"/>).
        /// </summary>
        /// <param name="lhs">The left operand.</param>
        /// <param name="rhs">The right operand.</param>
        /// <param name="op">The comparison.</param>
        /// <param name="out">The output array, or null for a fresh result.</param>
        /// <param name="where">The mask, or null for none.</param>
        /// <returns>The bool result (a <c>NDArray&lt;bool&gt;</c> instance on the bare route, <paramref name="out"/>
        ///     itself when one is given).</returns>
        /// <exception cref="IncorrectShapeException">The operands, <paramref name="out"/> or <paramref name="where"/> do
        ///     not broadcast.</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <remarks>The converted scalar is a temporary of this call: it is disposed on every exit, an exception
        ///     included, and never aliases the result.</remarks>
        private NDArray ExecuteComparison(NDArray lhs, NDArray rhs, ComparisonOp op, NDArray @out, NDArray where)
        {
            using var weak = AdoptWeakComparisonScalar(lhs, rhs, out bool scalarIsLhs);
            if (weak is not null)
            {
                if (scalarIsLhs)
                    lhs = weak;
                else
                    rhs = weak;
            }

            if (@out is null && where is null)
                return ExecuteComparisonOp(lhs, rhs, op);
            return ExecuteComparisonUfuncInto(lhs, rhs, op, lhs.GetTypeCode, rhs.GetTypeCode, @out, where);
        }
    }
}
