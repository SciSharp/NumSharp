using System.Collections.Generic;

namespace NumSharp
{
    // ============================== _core/_asarray.py ==============================
    // NumPy's numpy/_core/_asarray.py statements as a member of the array object: require (Appendix D's
    // NDArray.CoreAsArray.cs), with its requirement-name table and its flag test. np.require's facades keep their class rules
    // (numpy.matrix, numpy.char.chararray), the null check and the like= argument NumSharp has no dispatch for, and call the
    // member; the single-string overload splits the string into one-character tokens first, as NumPy iterates a str. The
    // statements call the members np.asanyarray and np.asarray dispatch to (NDArray.asanyarray / NDArray.asarray, the body of
    // _array_fromobject_generic), so the file sits under Backends/, beside the array object's class (plan C3).
    public partial class NDArray
    {
        // ---------------------------- np.require ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.require(a, dtype, requirements)</c> after the facade's class rules and null check: no
        ///     requirement is <c>asanyarray(a, dtype)</c>; otherwise the names are canonicalized, ENSUREARRAY dropped, the
        ///     order resolved (<c>'A'</c>, or <c>'C'</c> / <c>'F'</c> when asked), <c>asarray(a, dtype, order)</c> taken,
        ///     and ONE copy in that order made when any of the remaining ALIGNED / WRITEABLE / OWNDATA flags fails.
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <param name="dtype">The required dtype, or null to keep <paramref name="a"/>'s.</param>
        /// <param name="requirements">The requirement names (case-insensitive, NumPy's aliases), or null / empty for none.
        ///     A null entry is an unknown name.</param>
        /// <returns><paramref name="a"/> itself when nothing has to change (the caller must not dispose it as a new array),
        ///     else the conversion, else its copy: a copy owns its data, is writeable and aligned, so it satisfies every
        ///     flag at once.</returns>
        /// <exception cref="ValueError">An unknown name (<c>Cannot understand requirement 'x'</c>; NumPy raises a
        ///     KeyError), both orders (<c>Cannot specify both "C" and "F" order</c>), or the conversion's refusal of
        ///     <paramref name="dtype"/>.</exception>
        /// <remarks>No class boundary runs here: the facade applies its rules (a matrix result wrapped, a chararray operand's
        ///     class kept) before it calls this. ENSUREARRAY is accepted and ignored: the facade's rules decide the class,
        ///     and a plain array is what this returns.</remarks>
        internal static NDArray require(NDArray a, DType dtype, string[] requirements)
        {
            // `not requirements` in NumPy is true for None AND an empty sequence. np.asanyarray(a, dtype)'s statement once
            // its class rules ran: the array object's asanyarray in 'K' order.
            if (requirements == null || requirements.Length == 0)
                return asanyarray(a, dtype, 'K');

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

            // np.asarray(a, dtype, order)'s statement once its class rules and device check ran (copy=None).
            var arr = asarray(a, dtype, order, null);

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
        ///     Maps a requirement token to its canonical flag letter (C/F/A/W/O/E), matching NumPy's
        ///     <c>POSSIBLE_FLAGS</c> alias table. Unknown tokens raise (NumPy raises KeyError here).
        /// </summary>
        /// <param name="requirement">The token, any case; null is unknown.</param>
        /// <returns>The flag letter.</returns>
        /// <exception cref="ValueError">The token is no requirement name (<c>Cannot understand requirement 'x'</c>).</exception>
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
        ///     Whether <paramref name="arr"/> already satisfies a single ALIGNED / WRITEABLE / OWNDATA flag —
        ///     NumPy's <c>arr.flags[prop]</c> test in <c>require</c>. ALIGNED is false after
        ///     <c>setflags(align=False)</c> and over a caller's buffer at a misaligning byte offset; WRITEABLE is
        ///     false for broadcast views; OWNDATA is false for any view (slice / reshape / transpose / broadcast).
        /// </summary>
        /// <param name="arr">The array after the dtype/order conversion.</param>
        /// <param name="flag">'A', 'W' or 'O'.</param>
        /// <returns>True when no copy is needed for this flag.</returns>
        private static bool RequirementSatisfied(NDArray arr, char flag)
        {
            switch (flag)
            {
                case 'A':
                    // The flags' ALIGNED: the shape bit (setflags) and the memory's real alignment.
                    return (arr.Shape.Flags & ArrayFlags.ALIGNED) != 0 && arr.Storage.IsDataAligned;
                case 'W':
                    return arr.Shape.IsWriteable;
                case 'O':
                    return arr.Storage.OwnsData; // false for views AND externally-based arrays (memmaps)
                default:
                    return true;
            }
        }
    }
}
