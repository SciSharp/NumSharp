using System;
using System.Collections.Generic;

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
        /// <returns>Array with the specified requirements and dtype if given. A copy is made only when needed.</returns>
        /// <param name="like">Reference array for NumPy's array-function dispatch — accepted for signature parity but has no observable effect in NumSharp (no array-subclass dispatch).</param>
        /// <exception cref="ValueError">If both 'C' and 'F' order are requested, or a requirement string is not understood.</exception>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.require</c>. With no requirements this is exactly
        ///     <c>asanyarray(a, dtype)</c>. Otherwise it resolves an order ('A' by default, or 'C'/'F' when
        ///     requested), routes through <see cref="asarray(NDArray, Type, char, bool?, NDArray, string)"/>,
        ///     then makes a single copy (in the resolved order) if any of the remaining ALIGNED / WRITEABLE /
        ///     OWNDATA flags is not already satisfied. ALIGNED is always satisfied in NumSharp (managed
        ///     allocations), so only WRITEABLE (false for broadcast views) and OWNDATA (false for views)
        ///     can force a copy. 'ENSUREARRAY' is accepted and stripped but has no effect — NumSharp has no
        ///     ndarray subclasses to demote.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.require.html
        /// </remarks>
        public static NDArray require(NDArray a, Type dtype = null, string[] requirements = null, NDArray like = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // `like` drives NumPy's __array_function__ protocol; NumSharp has no array-subclass
            // dispatch, so it is accepted for signature parity but never read.
            _ = like;

            // `not requirements` in NumPy is true for None AND an empty sequence.
            if (requirements == null || requirements.Length == 0)
                return asanyarray(a, dtype);

            var reqs = new HashSet<char>();
            foreach (var r in requirements)
                reqs.Add(CanonicalizeRequirement(r));

            // 'E' (ENSUREARRAY) demotes subclasses to base ndarray. NumSharp has none, so strip and ignore.
            reqs.Remove('E');

            bool wantC = reqs.Contains('C');
            bool wantF = reqs.Contains('F');
            if (wantC && wantF)
                throw new ValueError("Cannot specify both \"C\" and \"F\" order");

            char order = 'A';
            if (wantF) { order = 'F'; reqs.Remove('F'); }
            else if (wantC) { order = 'C'; reqs.Remove('C'); }

            var arr = asarray(a, dtype, order);

            // Remaining flags can only be A / W / O. NumPy copies (in `order`) on the first unsatisfied
            // one; a copy owns its data, is writeable and aligned, so it satisfies all of them at once.
            foreach (var prop in reqs)
            {
                if (!RequirementSatisfied(arr, prop))
                    return arr.copy(order);
            }
            return arr;
        }

        /// <summary>
        ///     Single-string requirements overload. Mirrors NumPy's iteration semantics exactly: a string
        ///     is iterated CHARACTER BY CHARACTER, so <c>"F"</c> is the Fortran flag, <c>"CF"</c> requests
        ///     both C and F (and raises), and a multi-character alias such as <c>"F_CONTIGUOUS"</c> is NOT
        ///     one token — its '_' is unrecognized and raises. Pass a single-element array
        ///     (<c>new[]{ "F_CONTIGUOUS" }</c>) to use the full alias.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.require.html</remarks>
        public static NDArray require(NDArray a, Type dtype, string requirements)
            => require(a, dtype, ToCharTokens(requirements));

        /// <summary>Convenience overload taking a NumPy-style dtype string (e.g. <c>"float32"</c>).</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.require.html</remarks>
        public static NDArray require(NDArray a, string dtype, string[] requirements = null)
            => require(a, dtype == null ? null : np.dtype(dtype).type, requirements);

        /// <summary>Convenience overload taking a NumPy-style dtype string and a single requirements string.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.require.html</remarks>
        public static NDArray require(NDArray a, string dtype, string requirements)
            => require(a, dtype == null ? null : np.dtype(dtype).type, ToCharTokens(requirements));

        /// <summary>Convenience overload taking <see cref="NPTypeCode"/>.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.require.html</remarks>
        public static NDArray require(NDArray a, NPTypeCode dtype, string[] requirements = null)
            => require(a, dtype == NPTypeCode.Empty ? null : dtype.AsType(), requirements);

        /// <summary>Convenience overload taking <see cref="NPTypeCode"/> and a single requirements string.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.require.html</remarks>
        public static NDArray require(NDArray a, NPTypeCode dtype, string requirements)
            => require(a, dtype == NPTypeCode.Empty ? null : dtype.AsType(), ToCharTokens(requirements));

        /// <summary>Splits a requirements string into one-character tokens (NumPy iterates a string by char).</summary>
        private static string[] ToCharTokens(string requirements)
        {
            if (requirements == null)
                return null;

            var tokens = new string[requirements.Length];
            for (int i = 0; i < requirements.Length; i++)
                tokens[i] = requirements[i].ToString();
            return tokens;
        }

        /// <summary>
        ///     Maps a requirement token to its canonical flag letter (C/F/A/W/O/E), matching NumPy's
        ///     <c>POSSIBLE_FLAGS</c> alias table. Unknown tokens raise (NumPy raises KeyError here).
        /// </summary>
        private static char CanonicalizeRequirement(string requirement)
        {
            switch (requirement?.ToUpperInvariant())
            {
                case "C":
                case "C_CONTIGUOUS":
                case "CONTIGUOUS":
                    return 'C';
                case "F":
                case "F_CONTIGUOUS":
                case "FORTRAN":
                    return 'F';
                case "A":
                case "ALIGNED":
                    return 'A';
                case "W":
                case "WRITEABLE":
                    return 'W';
                case "O":
                case "OWNDATA":
                    return 'O';
                case "E":
                case "ENSUREARRAY":
                    return 'E';
                default:
                    throw new ValueError($"Cannot understand requirement '{requirement}'");
            }
        }

        /// <summary>
        ///     Whether <paramref name="arr"/> already satisfies a single ALIGNED / WRITEABLE / OWNDATA flag.
        ///     ALIGNED is always true for NumSharp's managed allocations; WRITEABLE is false for broadcast
        ///     views; OWNDATA is false for any view (slice / reshape / transpose / broadcast).
        /// </summary>
        private static bool RequirementSatisfied(NDArray arr, char flag)
        {
            switch (flag)
            {
                case 'A':
                    return true; // ALIGNED — always satisfied in NumSharp
                case 'W':
                    return arr.Shape.IsWriteable;
                case 'O':
                    return !arr.Storage.IsView;
                default:
                    return true;
            }
        }
    }
}
