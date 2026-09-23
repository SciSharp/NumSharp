using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     Pins <see cref="np.typename(string)"/> against NumPy 2.4.2 (numpy/lib/_type_check_impl.py's
    ///     <c>typename</c> = <c>_namefromtype[char]</c>). This is a type-CODE -> description dict lookup with
    ///     no array operand, so it has no differential-fuzz corpus entry (classified sibling-owned in the
    ///     oracle surface guard) — this suite IS its gate. Every expected string, and every KeyError message,
    ///     was produced by running the matching NumPy 2.4.2 call. Coverage: all 22 codes, the case-sensitivity
    ///     of the lookup, the distinct <c>"S1"</c>/<c>"S"</c> entries, the deliberately-absent <c>"e"</c>
    ///     (half) code, and the KeyError message (Python <c>repr</c>) on every miss — including the quote
    ///     selection, short escapes, non-printable \x/\u/\U escaping, and a null argument.
    /// </summary>
    [TestClass]
    public class TypenameTest
    {
        /// <summary>All 22 recognised codes return NumPy's exact description strings — the whole
        /// <c>_namefromtype</c> table, verified verbatim against NumPy 2.4.2.</summary>
        [TestMethod]
        public void Typename_AllCodes_MatchNumpy()
        {
            Assert.AreEqual("character", np.typename("S1"));
            Assert.AreEqual("bool", np.typename("?"));
            Assert.AreEqual("signed char", np.typename("b"));
            Assert.AreEqual("unsigned char", np.typename("B"));
            Assert.AreEqual("short", np.typename("h"));
            Assert.AreEqual("unsigned short", np.typename("H"));
            Assert.AreEqual("integer", np.typename("i"));
            Assert.AreEqual("unsigned integer", np.typename("I"));
            Assert.AreEqual("long integer", np.typename("l"));
            Assert.AreEqual("unsigned long integer", np.typename("L"));
            Assert.AreEqual("long long integer", np.typename("q"));
            Assert.AreEqual("unsigned long long integer", np.typename("Q"));
            Assert.AreEqual("single precision", np.typename("f"));
            Assert.AreEqual("double precision", np.typename("d"));
            Assert.AreEqual("long precision", np.typename("g"));
            Assert.AreEqual("complex single precision", np.typename("F"));
            Assert.AreEqual("complex double precision", np.typename("D"));
            Assert.AreEqual("complex long double precision", np.typename("G"));
            Assert.AreEqual("string", np.typename("S"));
            Assert.AreEqual("unicode", np.typename("U"));
            Assert.AreEqual("void", np.typename("V"));
            Assert.AreEqual("object", np.typename("O"));
        }

        /// <summary><c>"S1"</c> and a bare <c>"S"</c> are DISTINCT keys (character vs the string kind), a
        /// footgun NumPy exposes verbatim — the wrong one returns a different valid description, not an error.</summary>
        [TestMethod]
        public void Typename_S1_And_S_AreDistinct()
        {
            Assert.AreEqual("character", np.typename("S1"));
            Assert.AreEqual("string", np.typename("S"));
        }

        /// <summary>'g'/'G' carry NumPy 2.4.2's ACTUAL wording ("long precision" / "complex long double
        /// precision"), NOT the older "long double precision" the stale docstring example still prints.</summary>
        [TestMethod]
        public void Typename_LongDouble_Uses_2_4_2_Wording()
        {
            Assert.AreEqual("long precision", np.typename("g"));
            Assert.AreEqual("complex long double precision", np.typename("G"));
        }

        /// <summary>The lookup is CASE-SENSITIVE (ordinal): a lower-cased upper-code (and vice versa) misses.
        /// 's' is not 'S', 'o' is not 'O' — each raises rather than silently mapping.</summary>
        [TestMethod]
        public void Typename_IsCaseSensitive()
        {
            Assert.ThrowsException<KeyError>(() => np.typename("s"));
            Assert.ThrowsException<KeyError>(() => np.typename("o"));
            // The uppercase forms still resolve, proving the miss above is case (not a typo).
            Assert.AreEqual("string", np.typename("S"));
            Assert.AreEqual("object", np.typename("O"));
        }

        /// <summary>There is deliberately no <c>"e"</c> (half) entry — NumPy's typename omits it, so a half
        /// code raises exactly as NumSharp does. This is parity, not a NumSharp gap.</summary>
        [TestMethod]
        public void Typename_Half_Code_Absent_LikeNumpy()
        {
            Assert.ThrowsException<KeyError>(() => np.typename("e"));
        }

        /// <summary>A dtype NAME (not a code) misses — typename takes the single-character array-protocol
        /// codes, so "int32"/"float64"/"i4" all raise KeyError just as NumPy's raw dict lookup does.</summary>
        [TestMethod]
        public void Typename_DtypeNames_Miss()
        {
            Assert.ThrowsException<KeyError>(() => np.typename("int32"));
            Assert.ThrowsException<KeyError>(() => np.typename("float64"));
            Assert.ThrowsException<KeyError>(() => np.typename("i4"));
        }

        /// <summary>The KeyError message is the Python <c>repr</c> of the argument, byte-for-byte with NumPy:
        /// an unknown single code and a multi-character string quote as <c>'x'</c> / <c>'BB'</c>, and an empty
        /// string as <c>''</c>.</summary>
        [TestMethod]
        public void Typename_KeyError_Message_Is_PythonRepr()
        {
            var e1 = Assert.ThrowsException<KeyError>(() => np.typename("x"));
            Assert.AreEqual("'x'", e1.Message);

            var e2 = Assert.ThrowsException<KeyError>(() => np.typename("BB"));
            Assert.AreEqual("'BB'", e2.Message);

            var e3 = Assert.ThrowsException<KeyError>(() => np.typename(""));
            Assert.AreEqual("''", e3.Message);
        }

        /// <summary>The KeyError message reproduces Python <c>repr</c>'s quote selection and short escapes:
        /// a string containing a single quote (and no double quote) is double-quoted, and backslash / newline
        /// / tab / carriage-return use their short escapes — byte-for-byte with NumPy.</summary>
        [TestMethod]
        public void Typename_KeyError_Message_QuotesAndShortEscapes()
        {
            Assert.AreEqual("\"it's\"", Assert.ThrowsException<KeyError>(() => np.typename("it's")).Message);   // switches to double quotes
            Assert.AreEqual("'a\"b'", Assert.ThrowsException<KeyError>(() => np.typename("a\"b")).Message);      // stays single-quoted
            Assert.AreEqual("'\\\\'", Assert.ThrowsException<KeyError>(() => np.typename("\\")).Message);         // backslash -> \\
            Assert.AreEqual("'\\n'", Assert.ThrowsException<KeyError>(() => np.typename("\n")).Message);          // newline -> \n
            Assert.AreEqual("'\\t'", Assert.ThrowsException<KeyError>(() => np.typename("\t")).Message);          // tab -> \t
            Assert.AreEqual("'\\r'", Assert.ThrowsException<KeyError>(() => np.typename("\r")).Message);          // CR -> \r
        }

        /// <summary>NON-printable characters escape as Python <c>repr</c> does — \xNN (&lt;0x100), \uNNNN
        /// (&lt;0x10000), \UNNNNNNNN (supplementary) — while a PRINTABLE non-ASCII character (incl. an emoji)
        /// is rendered verbatim. Pins the exact NumPy 2.4.2 KeyError text for control chars, NBSP, soft
        /// hyphen, zero-width space and a non-printable supplementary code point (all differentially verified).
        /// Inputs use \u/\U C# escapes (never raw control bytes) so the source stays legible and unambiguous.</summary>
        [TestMethod]
        public void Typename_KeyError_Message_EscapesNonPrintables_LikePythonRepr()
        {
            Assert.AreEqual("'\\x00'", Assert.ThrowsException<KeyError>(() => np.typename("\u0000")).Message);      // NUL control
            Assert.AreEqual("'\\x1b'", Assert.ThrowsException<KeyError>(() => np.typename("\u001b")).Message);      // ESC control
            Assert.AreEqual("'\\x7f'", Assert.ThrowsException<KeyError>(() => np.typename("\u007f")).Message);      // DEL
            Assert.AreEqual("'\\xa0'", Assert.ThrowsException<KeyError>(() => np.typename("\u00a0")).Message);      // NBSP (Zs, not the 0x20 exception)
            Assert.AreEqual("'\\xad'", Assert.ThrowsException<KeyError>(() => np.typename("\u00ad")).Message);      // soft hyphen (Cf)
            Assert.AreEqual("'\\u200b'", Assert.ThrowsException<KeyError>(() => np.typename("\u200b")).Message);    // zero-width space (Cf)
            Assert.AreEqual("'\\U000e0001'", Assert.ThrowsException<KeyError>(() => np.typename("\U000E0001")).Message); // TAG (Cf, supplementary) -> \U, lowercase hex
            Assert.AreEqual("'\u00fc'", Assert.ThrowsException<KeyError>(() => np.typename("\u00fc")).Message);     // U+00FC '\u00fc' printable non-ASCII -> verbatim
            Assert.AreEqual("'\U0001F600'", Assert.ThrowsException<KeyError>(() => np.typename("\U0001F600")).Message); // U+1F600 emoji printable -> verbatim
        }

        /// <summary>A null argument is Python's <c>None</c>: it must surface as <c>KeyError("None")</c>
        /// (matching <c>np.typename(None)</c>), never a raw <see cref="System.NullReferenceException"/> or the
        /// <see cref="System.ArgumentNullException"/> a bare dictionary lookup would throw on a null key.</summary>
        [TestMethod]
        public void Typename_Null_Raises_KeyError_None()
        {
            var e = Assert.ThrowsException<KeyError>(() => np.typename(null));
            Assert.AreEqual("None", e.Message);
        }
    }
}
