using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.<br></br>
        ///     Matches NumPy 2.x signature: <c>clip(a, a_min=&lt;no value&gt;, a_max=&lt;no value&gt;, out=None, *, min=&lt;no value&gt;, max=&lt;no value&gt;, **kwargs)</c>, with the ufunc keywords <c>dtype</c> and <c>where</c>. Either or both bounds may be <c>null</c> (NumPy's <c>None</c>). The <paramref name="min"/> and <paramref name="max"/> keyword aliases (added in NumPy 2.0) are accepted under NumPy's rules (see the remarks).
        /// </summary>
        /// <param name="a">Array containing elements to clip. It broadcasts with the bounds, <paramref name="out"/> and <paramref name="where"/>.</param>
        /// <param name="a_min">Minimum value. If null, clipping is not performed on lower interval edge. A C# integer literal is NumPy's Python int: on an integer array a value at or below the dtype's minimum is ignored (it cannot clip anything), and a value that does not fit the dtype on the other side raises. An array bound broadcasts against <paramref name="a"/>.</param>
        /// <param name="a_max">Maximum value. If null, clipping is not performed on upper interval edge. A C# integer literal at or above an integer array's dtype maximum is ignored; one below the dtype's minimum raises.</param>
        /// <param name="out">The results will be placed in this array, which is returned as-is. It may be the input array for in-place clipping, and it may overlap an input in any other way: the result is then computed as if from a copy of that input (NumPy's COPY_IF_OVERLAP). <paramref name="out"/> must have the broadcast shape of the operands (it is never stretched), and its dtype must be reachable from the computed dtype by a same_kind cast (NumPy's ufunc rule; e.g. a uint8 clip may write into an int8 or a float64 out).</param>
        /// <param name="dtype">The dtype the computation runs in (NumPy's ufunc <c>dtype=</c>). The inputs are cast to it, and each must reach it by a same_kind cast.</param>
        /// <param name="min">NumPy 2.x keyword alias for <paramref name="a_min"/>; see the remarks for how it combines with the positional bounds.</param>
        /// <param name="max">NumPy 2.x keyword alias for <paramref name="a_max"/>; see the remarks for how it combines with the positional bounds.</param>
        /// <param name="where">NumPy's ufunc <c>where=</c> mask (a bool array, broadcast with the other operands). Only the True positions are computed and written. The other positions of <paramref name="out"/> keep their contents; without <paramref name="out"/> they are uninitialized, as in NumPy.</param>
        /// <returns>An array with the elements of a, but where values &lt; min are replaced with min, and those &gt; max with max (<paramref name="out"/> when given).</returns>
        /// <exception cref="TypeError">A <paramref name="min"/> / <paramref name="max"/> keyword is given while exactly one of <paramref name="a_min"/> / <paramref name="a_max"/> is null: NumPy's "clip() missing 1 required positional argument: 'a_max'" (or 'a_min'). Or neither bound is given (or both were ignored) and the computation would run in bool: NumPy then calls <c>np.positive</c>, which has no bool loop.</exception>
        /// <exception cref="ValueError">A <paramref name="min"/> / <paramref name="max"/> keyword is given together with both positional bounds ("Passing `min` or `max` keyword argument when `a_min` and `a_max` are provided is forbidden."). Or <paramref name="out"/> is read-only (checked before every ufunc argument). Or the operands do not broadcast together, or <paramref name="out"/> would have to be stretched (NumPy's two iterator texts).</exception>
        /// <exception cref="ArgumentException">A <paramref name="where"/> that is not bool ("Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'"); or a <paramref name="dtype"/> the inputs cannot reach, or an <paramref name="out"/> the result cannot be cast into, under the same_kind rule (NumPy's UFuncTypeError texts, naming the routed ufunc).</exception>
        /// <exception cref="OverflowException">An integer bound that is kept does not fit the integer dtype the computation runs in, for example <c>np.clip(int8_array, 1000, 2000)</c>: "Python integer 1000 out of bounds for int8" (NumPy's OverflowError). It is raised before any element is read, so an empty array raises too.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.clip.html
        ///     <para>
        ///     <b>Bound arguments.</b> NumPy tells an omitted argument (<c>&lt;no value&gt;</c>) from an explicit <c>None</c>;
        ///     C# cannot, so a null bound means None until a keyword alias is used:
        ///     </para>
        ///     <list type="bullet">
        ///     <item>No <paramref name="min"/> / <paramref name="max"/> keyword: <paramref name="a_min"/> and <paramref name="a_max"/>
        ///     are the bounds, a null one is None. NumPy raises "missing 1 required positional argument" for
        ///     <c>np.clip(a, 1)</c>; here that call clips from below only, as <c>np.clip(a, 1, None)</c> does.</item>
        ///     <item>A keyword given: a null positional counts as omitted, as it does in NumPy's own call
        ///     <c>np.clip(a, min=2)</c>. Both positionals null → the keywords are the bounds; exactly one null → NumPy's
        ///     TypeError naming it; both given → NumPy's ValueError. A positional passed explicitly as null next to a
        ///     keyword therefore reports the TypeError where NumPy's <c>np.clip(a, 2, None, min=3)</c> reports the
        ///     ValueError.</item>
        ///     </list>
        ///     <para>
        ///     <b>Routing.</b> A port of NumPy's <c>numpy/_core/_methods.py::_clip</c>, the function behind both <c>np.clip</c> and
        ///     <c>ndarray.clip</c>. On an integer array it first drops a Python-int bound at or past the dtype's limit:
        ///     <c>np.clip(uint8_array, 0, 256)</c> returns the array unchanged, and <c>np.clip(uint8_array, -1, 100)</c>
        ///     only clips from above. It then routes the call: no bound left → <c>positive</c> (a copy), one bound →
        ///     <c>minimum</c> / <c>maximum</c>, two → the <c>clip</c> ufunc. The routed name is what the cast errors quote.
        ///     A kept integer bound is packed into the computed dtype under NEP 50, so it raises instead of wrapping.
        ///     The drop rule uses the input array's own range, even when <paramref name="dtype"/> changes the computation:
        ///     <c>np.clip(int8_array, 1000, 2000, dtype: np.int16)</c> is legal and returns all 1000.
        ///     </para>
        ///     <para>
        ///     <b>The ufunc.</b> The routed ufunc broadcasts <paramref name="a"/>, the bounds, <paramref name="out"/> and
        ///     <paramref name="where"/> together; a fresh result takes the memory order of its operands (C order breaks ties),
        ///     as NumPy's iterator allocates it. For two scalar float32/float64 bounds NumPy runs its const fast path
        ///     (<c>x &lt; lo ? lo : x</c>, then <c>x &gt; hi ? hi : x</c>: a signed-zero tie keeps the element, a NaN bound fills the
        ///     result with that NaN's bits); array bounds run <c>MIN(MAX(x, lo), hi)</c> (a tie takes the bound). Which one runs
        ///     for a broadcast or buffered call is NumPy's own per-call decision, reproduced exactly.
        ///     </para>
        ///     <para>
        ///     NumSharp cannot tell a C# literal from a 0-d <see cref="NDArray"/> at this boundary, so every 0-d integer
        ///     bound counts as a Python int. NumPy promotes with a 0-d array or a NumPy scalar instead (<c>np.int64(1000)</c>
        ///     makes an int8 clip int64). The values agree; only that result dtype differs.
        ///     </para>
        ///     <para>
        ///     When <c>|value| &gt;= 2^31</c>, NumPy's overflow text depends on the platform's C <c>long</c>. NumSharp
        ///     uses the Linux/macOS (LP64) text: values that fit a 64-bit long say "Python integer ... out of bounds
        ///     for ...", larger ones say "Python int too large to convert to C long".
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        // No [NDScoped] here: this entry point creates no array of its own (it validates the keyword rule and
        // forwards), and the engine's ClipNDArray is the [NDScoped] boundary that owns every intermediate. A second,
        // empty scope cost an NDScope, its list and the argument capture on every call (~0.2 KB, measured).
        public static NDArray clip(
            NDArray a,
            NDArray a_min = null,
            NDArray a_max = null,
            NDArray @out = null,
            DType dtype = null,
            NDArray min = null,
            NDArray max = null,
            NDArray where = null)
        {
            // A numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary). np.clip is a ufunc call
            // there, so a matrix BOUND decides too (probed 2.4.2: np.clip(ndarray, matrix, None) is a matrix); a matrix
            // out is returned itself by the plain path, and a matrix where= does not change the class.
            if (NDMatrixBoundary.Sees(a, a_min, a_max) || NDMatrixBoundary.Sees(min, max))
                return NDMatrixBoundary.Wrapped((a, a_min, a_max, @out, dtype, min, max, where), static s => clip(s.a, s.a_min, s.a_max, s.@out, s.dtype, s.min, s.max, s.where), @out);
            // ---- numpy/_core/fromnumeric.py::clip, the argument rule. NumPy runs it in Python before the
            //      ufunc sees any argument, so its TypeError / ValueError win over every ufunc error (the
            //      read-only out included). A keyword alias makes a null positional "omitted" (see remarks).
            NDArray lo = a_min, hi = a_max;
            if (min is not null || max is not null)
            {
                if (a_min is null && a_max is null)
                {
                    lo = min;
                    hi = max;
                }
                else if (a_min is null)
                    throw new TypeError("clip() missing 1 required positional argument: 'a_min'");
                else if (a_max is null)
                    throw new TypeError("clip() missing 1 required positional argument: 'a_max'");
                else
                    throw new ValueError("Passing `min` or `max` keyword argument when `a_min` and `a_max` are provided is forbidden.");
            }

            // A read-only out refuses FIRST among the ufunc's arguments — probed 2.4.2: NumPy's clip ufunc
            // reports "output array is read-only" ahead of the where= mask, the Python-int OverflowError and
            // any broadcast/shape error, and even on the no-bounds identity path (clip(a, out=ro) still raises).
            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");

            return a.TensorEngine.ClipNDArray(a, lo, hi, dtype, @out, where);
        }
    }
}
