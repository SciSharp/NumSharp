using System;
using System.Collections;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return an ndarray of the provided type that satisfies requirements — NumPy's
        ///     <c>np.require(a, dtype=None, requirements=None, *, like=None)</c>, one signature.
        /// </summary>
        /// <param name="a">The object to be converted to a type-and-requirement-satisfying array.</param>
        /// <param name="dtype">The required data-type. <c>null</c> preserves the current dtype.</param>
        /// <param name="requirements">
        ///     NumPy's <c>requirements</c>: what NumPy iterates, so a STRING is read character by character (<c>"F"</c> is
        ///     the Fortran flag, <c>"CW"</c> is C plus WRITEABLE, and <c>"F_CONTIGUOUS"</c> is NOT one name — its
        ///     <c>'_'</c> raises NumPy's <c>KeyError: '_'</c>), while a <c>string[]</c>, a <c>List&lt;string&gt;</c> or any
        ///     other sequence of names is read name by name. Each name is one of (any case, NumPy's aliases):
        ///     <list type="bullet">
        ///         <item>'F_CONTIGUOUS' / 'F' / 'FORTRAN' — ensure a Fortran-contiguous array</item>
        ///         <item>'C_CONTIGUOUS' / 'C' / 'CONTIGUOUS' — ensure a C-contiguous array</item>
        ///         <item>'ALIGNED' / 'A' — ensure a data-type aligned array</item>
        ///         <item>'WRITEABLE' / 'W' — ensure a writable array</item>
        ///         <item>'OWNDATA' / 'O' — ensure an array that owns its own data</item>
        ///         <item>'ENSUREARRAY' / 'E' — ensure a base-class array (a <c>numpy.matrix</c> or chararray operand
        ///             gives a plain array)</item>
        ///     </list>
        ///     <c>null</c>, an empty string and an empty collection are NumPy's "no requirement" (<c>asanyarray</c>).
        ///     A C# tuple is not a sequence here: pass an array.
        /// </param>
        /// <param name="like">Reference array for NumPy's array-function dispatch — accepted for signature parity but has no
        ///     observable effect in NumSharp (no array-function dispatch).</param>
        /// <returns>Array with the specified requirements and dtype if given. A copy is made only when needed.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="KeyError">A name that is no flag, with NumPy's text: the upper-cased name, quoted
        ///     (<c>'BOGUS'</c>, <c>'_'</c>), raised at that name — before any later name is looked at.</exception>
        /// <exception cref="AttributeError">An item that is not a string (or a character): NumPy's <c>x.upper()</c> fails
        ///     with <c>'int' object has no attribute 'upper'</c> (<c>'NoneType'</c> for a null item).</exception>
        /// <exception cref="ValueError">Both 'C' and 'F' order are requested (<c>Cannot specify both "C" and "F"
        ///     order</c>), or a <c>numpy.matrix</c> operand whose result cannot be a matrix (<c>shape too large to be a
        ///     matrix.</c>).</exception>
        /// <remarks>
        ///     <para>Port of NumPy 2.x <c>numpy.require</c>: with no requirements this is exactly <c>asanyarray(a, dtype)</c>;
        ///     otherwise the set <c>{POSSIBLE_FLAGS[x.upper()] for x in requirements}</c>, an order ('A' by default, or
        ///     'C' / 'F' when requested), <see cref="asarray(NDArray, DType, char?, bool?, NDArray, string)"/>, then ONE
        ///     copy (in the resolved order) if any of the remaining ALIGNED / WRITEABLE / OWNDATA flags is not already
        ///     satisfied: ALIGNED (false after <c>setflags(align=False)</c> or over a caller's buffer at a misaligning byte
        ///     offset), WRITEABLE (false for broadcast views) and OWNDATA (false for views).</para>
        ///     <para>API-parity batch 10: one signature. The <c>string</c> and <c>string[]</c> overloads became this
        ///     <see cref="IEnumerable"/> parameter, which is what NumPy reads (it iterates <c>requirements</c>); an unknown
        ///     name raises NumPy's <see cref="KeyError"/> instead of the former <c>ValueError("Cannot understand requirement
        ///     'x'")</c>.</para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.require.html
        /// </remarks>
        public static NDArray require(NDArray a, DType dtype = null, IEnumerable requirements = null, NDArray like = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // `like` drives NumPy's __array_function__ protocol; NumSharp has no array-function dispatch, so it is accepted
            // for signature parity but never read.
            _ = like;

            // NumPy's statements (asanyarray for no requirement, else the names, the order, asarray and at most one copy)
            // are the array object's.
            return NDArray.require(a, dtype, requirements);
        }
    }
}
