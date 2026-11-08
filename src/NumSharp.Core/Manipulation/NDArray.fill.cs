using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Fill the array with a scalar value, IN PLACE (NumPy's <c>ndarray.fill</c>). Every element —
        ///     across whatever memory layout this array has (contiguous, F-order, sliced, transposed,
        ///     negative-stride) — is set to <paramref name="value"/> coerced to this array's dtype.
        ///
        ///     <para>
        ///     Coercion follows NumPy's scalar-assignment (NEP50 weak-scalar) rules, probed against
        ///     NumPy 2.4.2. A C# primitive is NumSharp's analog of a Python scalar (weak): assigned to an
        ///     INTEGER dtype it is range-checked — an out-of-bounds value RAISES
        ///     (<c>OverflowException</c> "Python integer 300 out of bounds for int8") rather than wrapping,
        ///     and a float source is TRUNCATED toward zero before the check (<c>3.9</c> stores 3, <c>300.0</c>
        ///     into int8 raises <c>OverflowException</c>; NaN raises <c>ValueError</c> "cannot convert float NaN
        ///     to integer"; ±inf raises <c>OverflowException</c> "cannot convert float infinity to integer"; a
        ///     complex source raises <c>TypeError</c>, exactly as NumPy's setitem runs <c>int()</c>/<c>float()</c>
        ///     on the value). Assigned to a float/complex dtype it casts, saturating to
        ///     ±inf on overflow (<c>float32.fill(1e300)</c> → inf). A 0-d <see cref="NDArray"/> is a STRONG
        ///     scalar and WRAPS on cast (matching NumPy's 0-d array) — except a C# literal's own 0-d array
        ///     (<see cref="NDArray.WasPythonLiteral"/>), which packs as the literal; a higher-rank array is a
        ///     sequence and raises <c>ValueError("setting an array element with a sequence.")</c>. A
        ///     <see cref="Half"/> or <c>char</c> is a NumPy scalar: range-checked into a SIGNED integer dtype,
        ///     cast into every other (<c>uint8.fill((Half)300)</c> is 44, as in NumPy). The whole conversion is
        ///     NumPy's <c>PyArray_Pack</c> (<see cref="ReductionInitial.TryPack"/>), shared with every store path.
        ///     </para>
        ///
        ///     NumPy checks writeability FIRST, then packs the scalar (which may raise) BEFORE touching any
        ///     element — so a read-only destination raises the read-only error even for a bad value, and an
        ///     out-of-range value raises even on an EMPTY array. Both orderings are reproduced.
        /// </summary>
        /// <param name="value">The scalar to fill with. A C# primitive (weak) or a 0-d <see cref="NDArray"/> (strong);
        ///     <c>null</c> is Python's <c>None</c>: NaN for a float array (NumPy's positive <c>NPY_NAN</c>), <c>nan+nanj</c>
        ///     for complex, <c>False</c> for bool, <c>'None'</c> (truncated to the width) for <c>S</c> / <c>U</c>, a
        ///     <c>StringDType</c>'s NA object or <c>'None'</c>.</param>
        /// <exception cref="TypeError"><paramref name="value"/> is null and this is an integer (or <c>decimal</c>) array:
        ///     Python's <c>int(None)</c> — raised even for an EMPTY array, as NumPy packs the scalar first.</exception>
        /// <exception cref="ValueError">This array is read-only (broadcast view / read-only memmap);
        ///     raises NumPy's exact <c>ValueError: assignment destination is read-only</c>.</exception>
        /// <exception cref="OverflowException">A weak integer/float value is out of range for an integer dtype,
        ///     or a ±inf value is assigned to an integer dtype (NumPy's <c>OverflowError</c>).</exception>
        /// <exception cref="ValueError"><paramref name="value"/> is a multi-element array (a sequence), or a NaN
        ///     value is assigned to an integer dtype (NumPy's <c>ValueError</c>).</exception>
        /// <exception cref="TypeError">A complex <paramref name="value"/> is assigned to a real (non-bool integer
        ///     or float) dtype — NumPy funnels it through <c>int()</c>/<c>float()</c>, which reject a complex.</exception>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into a bytes (<c>S</c>) array.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into a str (<c>U</c>) array.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.fill.html</remarks>
        public void fill(object value)
        {
            UnmanagedStorage.FillWithScalar(this, value);
        }
    }
}
