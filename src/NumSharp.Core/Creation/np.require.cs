using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return an ndarray of the provided type that satisfies requirements.
        /// </summary>
        /// <param name="a">The object to be converted to a type-and-requirement-satisfying array.</param>
        /// <param name="dtype">The required data-type. <c>null</c> preserves the current dtype.</param>
        /// <param name="requirements">
        ///     The requirements list. Each element is one of (case-insensitive, with aliases):
        ///     <list type="bullet">
        ///         <item>'F_CONTIGUOUS' / 'F' / 'FORTRAN' — ensure a Fortran-contiguous array</item>
        ///         <item>'C_CONTIGUOUS' / 'C' / 'CONTIGUOUS' — ensure a C-contiguous array</item>
        ///         <item>'ALIGNED' / 'A' — ensure a data-type aligned array</item>
        ///         <item>'WRITEABLE' / 'W' — ensure a writable array</item>
        ///         <item>'OWNDATA' / 'O' — ensure an array that owns its own data</item>
        ///         <item>'ENSUREARRAY' / 'E' — ensure a base array instead of a subclass (no-op in NumSharp: no ndarray subclasses)</item>
        ///     </list>
        /// </param>
        /// <returns>Array with the specified requirements and dtype if given. A copy is made only when needed.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <param name="like">Reference array for NumPy's array-function dispatch — accepted for signature parity but has no observable effect in NumSharp (no array-subclass dispatch).</param>
        /// <exception cref="ValueError">If both 'C' and 'F' order are requested, or a requirement string is not understood.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.require</c>. With no requirements this is exactly
        ///     <c>asanyarray(a, dtype)</c>. Otherwise it resolves an order ('A' by default, or 'C'/'F' when
        ///     requested), routes through <see cref="asarray(NDArray, DType, char, bool?, NDArray, string)"/>,
        ///     then makes a single copy (in the resolved order) if any of the remaining ALIGNED / WRITEABLE /
        ///     OWNDATA flags is not already satisfied: ALIGNED (false after <c>setflags(align=False)</c> or over
        ///     a caller's buffer at a misaligning byte offset), WRITEABLE (false for broadcast views) and
        ///     OWNDATA (false for views). 'ENSUREARRAY' is accepted and stripped but has no effect — NumSharp has
        ///     no ndarray subclasses to demote.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.require.html
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray require(NDArray a, DType dtype = null, string[] requirements = null, NDArray like = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, dtype, requirements, like), static s => require(s.a, s.dtype, s.requirements, s.like));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.Kept((a, dtype, requirements, like), static s => require(s.a, s.dtype, s.requirements, s.like));
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // `like` drives NumPy's __array_function__ protocol; NumSharp has no array-subclass
            // dispatch, so it is accepted for signature parity but never read.
            _ = like;

            // NumPy's statements (asanyarray for no requirement, else the names, the order, asarray and at most one copy)
            // are the array object's.
            return NDArray.require(a, dtype, requirements);
        }

        /// <summary>
        ///     Single-string requirements overload. Mirrors NumPy's iteration semantics exactly: a string
        ///     is iterated CHARACTER BY CHARACTER, so <c>"F"</c> is the Fortran flag, <c>"CF"</c> requests
        ///     both C and F (and raises), and a multi-character alias such as <c>"F_CONTIGUOUS"</c> is NOT
        ///     one token — its '_' is unrecognized and raises. Pass a single-element array
        ///     (<c>new[]{ "F_CONTIGUOUS" }</c>) to use the full alias.
        /// </summary>
        /// <param name="a">The array to convert.</param>
        /// <param name="dtype">The required data-type, or null to keep <paramref name="a"/>'s.</param>
        /// <param name="requirements">The flag letters, one per character (case-insensitive); null or empty for none.</param>
        /// <returns>Array with the requirements and dtype (see the list overload); a copy only when needed.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError">A character that is no flag letter, or both C and F.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.require.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.
        /// </remarks>
        public static NDArray require(NDArray a, DType dtype, string requirements)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((a, dtype, requirements), static s => require(s.a, s.dtype, s.requirements))
            : NDCharArrayBoundary.Sees(a)   // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            ? NDCharArrayBoundary.Kept((a, dtype, requirements), static s => require(s.a, s.dtype, s.requirements))
            : require(a, dtype, ToCharTokens(requirements));

        /// <summary>Splits a requirements string into one-character tokens (NumPy iterates a string by char).</summary>
        /// <param name="requirements">The string, or null.</param>
        /// <returns>One single-character token per character (a new array), or null for null (no requirement).</returns>
        private static string[] ToCharTokens(string requirements)
        {
            if (requirements == null)
                return null;

            var tokens = new string[requirements.Length];
            for (int i = 0; i < requirements.Length; i++)
                tokens[i] = requirements[i].ToString();
            return tokens;
        }
    }
}
