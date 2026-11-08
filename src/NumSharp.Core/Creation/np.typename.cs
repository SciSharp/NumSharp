using System;
using System.Collections.Generic;
using NumSharp.IO;   // PyLiteral.Repr — the NumPy-parity Python-repr used for the KeyError message.

namespace NumSharp
{
    public static partial class np
    {
        // ==========================================================================================
        //  np.typename — port of numpy/lib/_type_check_impl.py (typename).
        //
        //  A pure type-CODE -> human-description lookup: NumPy's `typename(char)` is literally
        //  `_namefromtype[char]`, a dict indexed by the single array-protocol type-code CHARACTER
        //  (e.g. 'i', 'f', 'D', plus the two-char 'S1'). There is NO NDArray operand, no dtype loop and
        //  no kernel — it is the type-check sibling of the scalar->string formatters binary_repr/base_repr,
        //  so it is classified "sibling-owned" in the oracle surface guard and gated by its own suite.
        //
        //  Fidelity notes (every point probed against NumPy 2.4.2):
        //    * The 22 keys are VERBATIM NumPy's _namefromtype and are CASE-SENSITIVE — 's' misses, 'S' hits.
        //    * 'S1' -> "character" and 'S' -> "string" are DISTINCT entries (a bare 'S' is the string kind),
        //      so passing the wrong one silently returns a different (still valid) description, as in NumPy.
        //    * 'g'/'G' read "long precision" / "complex long double precision" — NumPy 2.4.2's actual wording,
        //      NOT the older "long double precision" its own (stale) docstring example still prints.
        //    * There is deliberately NO 'e' (half) key — NumPy's typename omits it, so typename("e") raises
        //      exactly as NumSharp does. This is parity, not a gap.
        //    * A miss — an unknown code, an empty or multi-character string, a differently-cased code, or a
        //      null argument (Python's None) — raises KeyError whose message is the Python repr of the
        //      argument ('x', '', None). We route null EXPLICITLY because Dictionary.TryGetValue throws on a
        //      null key instead of reporting a miss, and the null path must still surface as KeyError("None")
        //      like np.typename(None). The repr comes from PyLiteral.Repr, the helper NumPy-parity error text
        //      already uses (correct single/double quote selection + \\ \n \r \t escaping).
        // ==========================================================================================

        /// <summary>
        ///     Returns a human-readable description for an array-protocol data-type CODE (np.typename) — e.g.
        ///     <c>"i"</c> -> <c>"integer"</c>, <c>"D"</c> -> <c>"complex double precision"</c>. This is a
        ///     type-CODE lookup, NOT a dtype-name lookup: it expects the single-character codes
        ///     (<c>?bBhHiIlLqQfdgFDGSUVO</c>) or the two-character <c>"S1"</c>, is CASE-SENSITIVE, and has no
        ///     entry for <c>"e"</c> (half) — matching NumPy, which omits it.
        /// </summary>
        /// <param name="char">
        ///     The array-protocol type code to describe. Footgun: <c>"S1"</c> -> <c>"character"</c> but a bare
        ///     <c>"S"</c> -> <c>"string"</c> (two distinct entries), so the wrong one returns a different valid
        ///     description rather than raising; a differently-cased code (<c>"s"</c>) or a dtype NAME
        ///     (<c>"int32"</c>) is not a code and raises.
        /// </param>
        /// <returns>The description string for <paramref name="char"/>.</returns>
        /// <exception cref="KeyError">
        ///     <paramref name="char"/> is not one of the 22 recognised codes — this covers an unknown code, an
        ///     empty or multi-character string, a differently-cased code, and a <c>null</c> argument. The
        ///     message is the Python <c>repr</c> of the argument (<c>'x'</c>, <c>''</c>, <c>None</c>), matching
        ///     the raw <c>dict</c> KeyError NumPy leaks.
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.typename.html</remarks>
        public static string typename(string @char)
        {
            // NumPy is `return _namefromtype[char]` — a bare dict index, so any miss (and a None key) raises
            // KeyError. Guard null first: Dictionary.TryGetValue throws ArgumentNullException on a null key
            // rather than returning false, and np.typename(None) must surface as KeyError("None"), not an NRE.
            if (@char != null && _namefromtype.TryGetValue(@char, out string name))
                return name;
            throw new KeyError(PyLiteral.Repr(@char));
        }

        /// <summary>
        ///     NumPy's <c>_namefromtype</c> table (numpy/lib/_type_check_impl.py) reproduced verbatim:
        ///     array-protocol type CODE -> description. Ordinal (case-sensitive) is load-bearing for parity —
        ///     <c>'s'</c> must miss while <c>'S'</c> hits — and it is the default string comparer regardless,
        ///     passed explicitly here to document that the case sensitivity is intentional, not incidental.
        /// </summary>
        private static readonly Dictionary<string, string> _namefromtype = new(StringComparer.Ordinal)
        {
            ["S1"] = "character",
            ["?"] = "bool",
            ["b"] = "signed char",
            ["B"] = "unsigned char",
            ["h"] = "short",
            ["H"] = "unsigned short",
            ["i"] = "integer",
            ["I"] = "unsigned integer",
            ["l"] = "long integer",
            ["L"] = "unsigned long integer",
            ["q"] = "long long integer",
            ["Q"] = "unsigned long long integer",
            ["f"] = "single precision",
            ["d"] = "double precision",
            ["g"] = "long precision",
            ["F"] = "complex single precision",
            ["D"] = "complex double precision",
            ["G"] = "complex long double precision",
            ["S"] = "string",
            ["U"] = "unicode",
            ["V"] = "void",
            ["O"] = "object",
        };
    }
}
