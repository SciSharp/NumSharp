using System;
using System.Numerics;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     <see cref="np.mintypecode(object, string, char)"/> — the port of <c>numpy/lib/_type_check_impl.py::mintypecode</c>.
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
            np.mintypecode(new object[0], null).Should().Be('d', "an empty sequence never reaches the type set either");
        }

        /// <summary>
        ///     NumPy's docstring example and the array forms: an array's elements along the first axis each give the array's
        ///     <c>dtype.char</c> (none for an empty first axis, every row of an N-D array the same character, a strided view
        ///     too), and a 0-d array is not iterable.
        /// </summary>
        [TestMethod]
        public void Arrays_GiveTheirCharacterPerElement()
        {
            np.mintypecode(np.array(new[] { new Complex(1.1, 0), new Complex(2, -3) })).Should().Be('D'); // >>> np.mintypecode(x)
            np.mintypecode(np.array(new[] { 1f, 2f })).Should().Be('f');
            np.mintypecode(np.array(new float[0])).Should().Be('d', "an empty first axis gives no character");
            np.mintypecode(np.zeros(new Shape(0, 3))).Should().Be('d');
            np.mintypecode(np.zeros(new Shape(2, 3), np.complex128)).Should().Be('D');
            np.mintypecode(np.arange(10).astype(np.float32)["::3"]).Should().Be('f');
            np.mintypecode(np.array(new[] { true })).Should().Be('d');
            np.mintypecode(np.matrix(np.array(new[,] { { 1f } }))).Should().Be('f');
            new Action(() => np.mintypecode(np.array(1.0))).Should().Throw<TypeError>().WithMessage("iteration over a 0-d array");
            new Action(() => np.mintypecode(np.zeros(new Shape(2), np.float16), "e")).Should().Throw<ValueError>().WithMessage("substring not found");
        }

        /// <summary>
        ///     A 1-D <c>U</c> or <c>StringDType</c> array's elements are Python <c>str</c> in NumPy, so their TEXT is the type
        ///     character (an empty element is <c>asarray("")</c>'s <c>'U'</c>); a chararray's text after its <c>rstrip()</c>;
        ///     an <c>S</c> array's elements are bytes (<c>'S'</c>), and the rows of a 2-D text array are arrays (<c>'U'</c>).
        /// </summary>
        [TestMethod]
        public void TextArrays_GiveTheirElementsText()
        {
            np.mintypecode(np.array(new[] { "d", "f" })).Should().Be('d');
            np.mintypecode(np.array(new[] { "", "f" })).Should().Be('f');
            np.mintypecode(np.array(new[] { "g", "S" })).Should().Be('g');
            np.mintypecode(np.array(new[] { "d", "f" }, dtype: np.dtype("T"))).Should().Be('d');
            np.mintypecode(np.array(new[] { "f", "F" }, dtype: np.dtype("S"))).Should().Be('d', "bytes elements are 'S'");
            np.mintypecode(np.array(new[,] { { "f", "F" } })).Should().Be('d', "a row is an array: 'U'");
            np.mintypecode(np.@char.array(new[] { "d ", "f" })).Should().Be('d');
            np.mintypecode(np.@char.array(new[] { "  ", "f" })).Should().Be('f');
        }

        /// <summary>
        ///     A masked array reads like its data, except that a masked element of a 1-D array is <c>np.ma.masked</c>, a
        ///     float64 constant: <c>'d'</c>. The rows of a 2-D masked array are masked arrays, not the constant.
        /// </summary>
        [TestMethod]
        public void MaskedArrays_ReadMaskedElementsAsTheFloat64Constant()
        {
            var data = np.array(new[] { 1f, 2f });
            np.mintypecode(np.ma.array(data)).Should().Be('f');
            np.mintypecode(np.ma.array(data, mask: np.array(new[] { true, false }))).Should().Be('d');
            np.mintypecode(np.ma.array(data, mask: np.array(new[] { true, true }))).Should().Be('d');
            np.mintypecode(np.ma.array(np.zeros(new Shape(2, 2), np.float32), mask: np.array(new[,] { { true, false }, { false, false } }))).Should().Be('f');
            np.mintypecode(np.ma.array(np.array(new[] { "f", "g" }), mask: np.array(new[] { true, false }))).Should().Be('g');
            np.mintypecode(new object[] { np.ma.masked }).Should().Be('d');
        }

        /// <summary>
        ///     A sequence is read item by item: a string is itself (a C# <c>char</c> the character it spells), an array its
        ///     character, a scalar <c>np.asarray(x).dtype.char</c>; what NumPy keeps in an object array (null, a descriptor,
        ///     a Python int past uint64) is <c>'O'</c>, which no default type set holds. A C# tuple is a Python tuple.
        /// </summary>
        [TestMethod]
        public void Sequences_AreReadItemByItem()
        {
            np.mintypecode(new[] { "d", "f", "S" }).Should().Be('d'); // >>> np.mintypecode(['d', 'f', 'S'])
            np.mintypecode(new object[] { np.zeros(new Shape(2), np.float32), "d" }).Should().Be('d');
            np.mintypecode(new object[] { np.zeros(new Shape(2), np.complex128), np.zeros(new Shape(2)) }).Should().Be('D');
            np.mintypecode(new object[] { 1.5, 2 }).Should().Be('d');
            np.mintypecode(new object[] { 1f }).Should().Be('f');
            np.mintypecode(new object[] { new Complex(0, 1) }).Should().Be('D');
            np.mintypecode(new object[] { new[] { 1f } }).Should().Be('f');
            np.mintypecode(("f", "d")).Should().Be('d');
            np.mintypecode(new[] { "" }).Should().Be('d', "the empty string is asarray('')'s 'U'");
            np.mintypecode(new[] { np.float32, np.float64 }).Should().Be('d', "a descriptor is held in an object array: 'O'");
            np.mintypecode(new object[] { null }).Should().Be('d');
            np.mintypecode(new object[] { BigInteger.Pow(10, 30) }).Should().Be('d');
            new Action(() => np.mintypecode(new object[] { null }, "O")).Should().Throw<ValueError>().WithMessage("substring not found");
            new Action(() => np.mintypecode(new object[] { BigInteger.Pow(10, 30) }, "O")).Should().Throw<ValueError>().WithMessage("substring not found");
            np.mintypecode("\U0001F600d").Should().Be('d', "a surrogate pair is one code point");
        }

        /// <summary>
        ///     NumPy tests a string item with <c>t in typeset</c>, a SUBSTRING test, so a multi-character item that typeset
        ///     contains takes part ('Fg' ranks at 2, beside 'F'); when it would be the answer, a char cannot hold it. In a
        ///     tie with a one-character member NumPy's pick follows Python's hash-randomized set order, and NumSharp takes the
        ///     one-character member, one of NumPy's answers.
        /// </summary>
        [TestMethod]
        public void MultiCharacterItems_FollowNumPysSubstringTest()
        {
            np.mintypecode(new[] { "Fd" }).Should().Be('d', "'Fd' is no substring of 'GDFgdf'");
            np.mintypecode(new[] { "Fg", "F" }).Should().Be('F');
            foreach (var typechars in new object[] { new[] { "Fg" }, new[] { "Fg", "d" }, np.array(new[] { "Fg" }) })
                new Action(() => np.mintypecode(typechars)).Should().Throw<NotSupportedException>().WithMessage("*'Fg'*");
        }

        /// <summary>Anything else is not iterable, with Python's texts (a descriptor names its numpy.dtypes class).</summary>
        [TestMethod]
        public void NonIterables_ArePythonsTypeErrors()
        {
            new Action(() => np.mintypecode(3)).Should().Throw<TypeError>().WithMessage("'int' object is not iterable");
            new Action(() => np.mintypecode(np.float32)).Should().Throw<TypeError>().WithMessage("'numpy.dtypes.Float32DType' object is not iterable");
            new Action(() => np.mintypecode(np.floating)).Should().Throw<TypeError>().WithMessage("'type' object is not iterable");
        }
    }
}
