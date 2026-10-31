using System;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     <see cref="np.mintypecode(string, string, char)"/> — the port of <c>numpy/lib/_type_check_impl.py::mintypecode</c>.
    ///     Every expectation was probed against <c>numpy==2.4.2</c>.
    /// </summary>
    [TestClass]
    public class NpMintypecodeTests
    {
        /// <summary>
        ///     The intersection with the type set, the default for none, <c>'D'</c> for <c>'F'</c> with <c>'d'</c>, else the
        ///     character listed first in <c>'GDFgdfQqLlIiHhBb?'</c>.
        /// </summary>
        [DataTestMethod]
        [DataRow("d", "GDFgdf", 'd')]
        [DataRow("f", "GDFgdf", 'f')]
        [DataRow("F", "GDFgdf", 'F')]
        [DataRow("D", "GDFgdf", 'D')]
        [DataRow("fd", "GDFgdf", 'd')]
        [DataRow("Fd", "GDFgdf", 'D')]
        [DataRow("dF", "GDFgdf", 'D')]
        [DataRow("if", "GDFgdf", 'f')]
        [DataRow("iF", "GDFgdf", 'F')]
        [DataRow("bh", "GDFgdf", 'd')]
        [DataRow("", "GDFgdf", 'd')]
        [DataRow("g", "GDFgdf", 'g')]
        [DataRow("G", "GDFgdf", 'G')]
        [DataRow("gF", "GDFgdf", 'F')]
        [DataRow("Gd", "GDFgdf", 'G')]
        [DataRow("fF", "GDFgdf", 'F')]
        [DataRow("dfFD", "GDFgdf", 'D')]
        [DataRow("Fde", "Fde", 'D')]
        [DataRow("id", "id", 'd')]
        [DataRow("", "x", 'd')]
        public void Answers_AreNumPys(string typechars, string typeset, char expected)
        {
            np.mintypecode(typechars, typeset).Should().Be(expected);
            np.mintypecode(typechars.ToCharArray(), typeset).Should().Be(expected, "the char[] spelling is the same function");
        }

        /// <summary>The default is returned when nothing matches.</summary>
        [TestMethod]
        public void NoMatch_ReturnsTheDefault()
        {
            np.mintypecode("xyz", @default: 'q').Should().Be('q');
        }

        /// <summary>
        ///     NumPy picks the minimum with <c>_typecodes_by_elsize.index</c>; a matching character missing from that string
        ///     (<c>'e'</c>) makes <c>str.index</c> raise "substring not found" — Python's <c>min</c> evaluates the key of
        ///     every element, so it raises even when another element would win. The <c>'F'</c> + <c>'d'</c> rule answers
        ///     first and never reaches the index.
        /// </summary>
        [TestMethod]
        public void AMatchOutsideTheSizedCodes_RaisesNumPysValueError()
        {
            new Action(() => np.mintypecode("e", "efd")).Should().Throw<ValueError>().WithMessage("substring not found");
            new Action(() => np.mintypecode("ef", "efd")).Should().Throw<ValueError>().WithMessage("substring not found");
            np.mintypecode("Fde", "Fde").Should().Be('D');
        }

        /// <summary>
        ///     Python's own errors for None: iterating None (<c>typechars</c>), and <c>t in None</c> (<c>typeset</c>), which
        ///     only happens once there is a character to test.
        /// </summary>
        [TestMethod]
        public void None_Arguments_ArePythonsTypeErrors()
        {
            new Action(() => np.mintypecode((string)null)).Should().Throw<TypeError>().WithMessage("'NoneType' object is not iterable");
            new Action(() => np.mintypecode((char[])null)).Should().Throw<TypeError>().WithMessage("'NoneType' object is not iterable");
            new Action(() => np.mintypecode("d", null)).Should().Throw<TypeError>().WithMessage("argument of type 'NoneType' is not iterable");
            np.mintypecode("", null).Should().Be('d', "no character ever reaches the type set");
        }
    }
}
