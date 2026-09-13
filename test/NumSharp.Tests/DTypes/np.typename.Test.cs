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
    ///     (half) code, and the KeyError message (Python <c>repr</c>) on every miss including a null argument.
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
