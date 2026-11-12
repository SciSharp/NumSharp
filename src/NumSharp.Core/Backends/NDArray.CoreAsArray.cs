using System.Collections;
using System.Collections.Generic;
using NumSharp.Backends.Printing;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    // ============================== _core/_asarray.py ==============================
    // NumPy's numpy/_core/_asarray.py statements as a member of the array object: require (Appendix D's
    // NDArray.CoreAsArray.cs), with its requirement-name table and its flag test. np.require's facade keeps its class rules
    // (numpy.matrix, numpy.char.chararray), the null check and the like= argument NumSharp has no dispatch for, and calls the
    // member with the requirements as NumPy receives them (an iterable: a str is iterated by character). The statements call
    // the members np.asanyarray and np.asarray dispatch to (NDArray.asanyarray / NDArray.asarray, the body of
    // _array_fromobject_generic), so the file sits under Backends/, beside the array object's class (plan C3).
    public partial class NDArray
    {
        // ---------------------------- np.require ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.require(a, dtype, requirements)</c> after the facade's class rules and null check: no
        ///     requirement is <c>asanyarray(a, dtype)</c>; otherwise <c>{POSSIBLE_FLAGS[x.upper()] for x in
        ///     requirements}</c>, ENSUREARRAY dropped, the order resolved (<c>'A'</c>, or <c>'C'</c> / <c>'F'</c> when
        ///     asked), <c>asarray(a, dtype, order)</c> taken, and ONE copy in that order made when any of the remaining
        ///     ALIGNED / WRITEABLE / OWNDATA flags fails.
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <param name="dtype">The required dtype, or null to keep <paramref name="a"/>'s.</param>
        /// <param name="requirements">What NumPy iterates: a string (its characters are the names), a sequence of names, or
        ///     null. Python's truthiness decides "no requirement": null, an empty string and an empty collection; any other
        ///     enumerable (a LINQ query, like a Python generator) is truthy and iterated once.</param>
        /// <returns><paramref name="a"/> itself when nothing has to change (the caller must not dispose it as a new array),
        ///     else the conversion, else its copy: a copy owns its data, is writeable and aligned, so it satisfies every
        ///     flag at once.</returns>
        /// <exception cref="KeyError">A name that is no flag: NumPy's text, the upper-cased name quoted
        ///     (<c>'BOGUS'</c>).</exception>
        /// <exception cref="AttributeError">An item that is not a string or a character (<c>'int' object has no attribute
        ///     'upper'</c>).</exception>
        /// <exception cref="ValueError">Both orders (<c>Cannot specify both "C" and "F" order</c>), or the conversion's
        ///     refusal of <paramref name="dtype"/>.</exception>
        /// <remarks>ENSUREARRAY is accepted and ignored here: this computes a plain array, and the class of a subclass
        ///     operand's result is the computation's rule (<see cref="NDSubclassRules.Require"/>: the operand's class, a plain
        ///     array for ENSUREARRAY - NumPy's <c>subok</c>).</remarks>
        [SubclassRule(typeof(NDSubclassRules), nameof(NDSubclassRules.Require))]
        internal static NDArray require(NDArray a, DType dtype, IEnumerable requirements)
        {
            // `if not requirements: return asanyarray(a, dtype=dtype)` - Python's truthiness: None, '' and an empty container
            // are false. np.asanyarray(a, dtype)'s statement once its class rules ran: the array object's asanyarray in 'K'
            // order.
            if (IsFalsyRequirements(requirements))
                return asanyarray(a, dtype, 'K');

            // `{POSSIBLE_FLAGS[x.upper()] for x in requirements}`: ONE pass, so a name raises at its own position (an
            // unknown name before a later non-string item, and the reverse).
            var reqs = new HashSet<char>();
            foreach (var item in requirements)
                reqs.Add(CanonicalizeRequirement(item));

            // 'E' (ENSUREARRAY) is NumPy's subok=False: the subclass rule demotes the result; the computation ignores it.
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
        ///     Python's <c>not requirements</c> for the C# shapes NumPy's parameter takes: null (None), an empty string and an
        ///     empty collection are false. Any other enumerable is TRUE without being looked at, as a Python generator is - so
        ///     a lazy sequence is enumerated exactly once, by the name pass.
        /// </summary>
        /// <param name="requirements">The caller's requirements.</param>
        /// <returns>True for "no requirement" (NumPy's <c>asanyarray</c> branch).</returns>
        private static bool IsFalsyRequirements(IEnumerable requirements)
            => requirements switch
            {
                null => true,
                string text => text.Length == 0,
                ICollection collection => collection.Count == 0,
                _ => false,
            };

        /// <summary>
        ///     NumPy's <c>POSSIBLE_FLAGS[x.upper()]</c> for one item of the requirements: the canonical flag letter (C / F /
        ///     A / W / O / E) of a name or alias, any case.
        /// </summary>
        /// <param name="item">One item: a string (a name), a character (one item of an iterated string), or anything else.</param>
        /// <returns>The flag letter.</returns>
        /// <exception cref="AttributeError">The item is not a string or a character: <c>x.upper()</c> fails with Python's
        ///     text, <c>'int' object has no attribute 'upper'</c> (<c>'NoneType'</c> for null).</exception>
        /// <exception cref="KeyError">The upper-cased name is no key of <c>POSSIBLE_FLAGS</c>: Python's text, the key's repr
        ///     (<c>'BOGUS'</c>).</exception>
        /// <remarks>Upper-casing is <see cref="string.ToUpperInvariant"/>: Python's <c>str.upper</c> on every flag name (they
        ///     are ASCII); a non-ASCII item is an unknown name either way, only its quoted text could differ from Python's
        ///     full case mapping (<c>'ß'.upper()</c> is <c>'SS'</c> in Python).</remarks>
        private static char CanonicalizeRequirement(object item)
        {
            string name = item switch
            {
                string text => text,
                // A C# string enumerates its characters: each one is NumPy's one-character str.
                char c => c.ToString(),
                null => throw new AttributeError("'NoneType' object has no attribute 'upper'"),
                _ => throw new AttributeError($"'{NDBuffer.PythonTypeName(item)}' object has no attribute 'upper'"),
            };
            string upper = name.ToUpperInvariant();
            switch (upper)
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
                    // Python's str(KeyError(key)) is repr(key): the quoted, escaped upper-cased name.
                    throw new KeyError(PyRepr.Str(upper));
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
