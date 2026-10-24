using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     The add / multiply reductions in a char LOOP (<c>dtype=char</c>): NumSharp's char is NumPy's uint16 (the oracle's
    ///     proxy), so a char loop is NumPy's uint16 loop, which wraps modulo 2^16 (DefaultEngine.ReduceInCharLoop). Every
    ///     literal is NumPy 2.4.2's answer for the same call with <c>dtype=np.uint16</c> over the same values as uint16.
    /// </summary>
    /// <remarks>
    ///     Before the fix the flat sum and product raised NotSupportedException ("Sum not supported for type Char"), the
    ///     product along a non-innermost axis multiplied through a double and saturated (536 came out as 65535), the flat
    ///     mean summed without wrapping (21823 for 10900), and the flat var raised through its sum.
    /// </remarks>
    [TestClass]
    public class np_reduce_char_loop_tests
    {
        private static readonly DType Char = NPTypeCode.Char;

        /// <summary>[[3, 300, 65535], [100, 2, 65000]] as char: the values the probes used.</summary>
        private static NDArray Values() =>
            np.array(new char[,] { { (char)3, (char)300, (char)65535 }, { (char)100, (char)2, (char)65000 } });

        /// <summary>The values of a char result as integers, in C order.</summary>
        private static int[] Ints(NDArray r) => r.flat.ToArray<char>().Select(c => (int)c).ToArray();

        /// <summary>The flat sum and product wrap in uint16, and come back as NumPy's read-only scalar of the loop dtype.</summary>
        [TestMethod]
        public void FlatSumAndProduct_WrapInTheUInt16Loop()
        {
            using var a = Values();
            using (var s = np.sum(a, Char))
            {
                s.typecode.Should().Be(NPTypeCode.Char);
                s.ndim.Should().Be(0);
                s.Shape.IsWriteable.Should().BeFalse();
                Ints(s).Should().Equal(65404);
            }
            using (var p = np.prod(a, null, Char))
            {
                p.typecode.Should().Be(NPTypeCode.Char);
                Ints(p).Should().Equal(11008);
            }
        }

        /// <summary>
        ///     Along an axis: the product along axis 0 wraps 65535 · 65000 to 536 (it saturated to 65535 through a double),
        ///     the sum and the product along axis 1 wrap too.
        /// </summary>
        [TestMethod]
        public void AxisSumAndProduct_WrapInTheUInt16Loop()
        {
            using var a = Values();
            using (var r = np.sum(a, 0, Char))
                Ints(r).Should().Equal(103, 302, 64999);
            using (var r = np.prod(a, 0, Char))
            {
                r.typecode.Should().Be(NPTypeCode.Char);
                Ints(r).Should().Equal(300, 600, 536);
            }
            using (var r = np.prod(a, 1, Char))
                Ints(r).Should().Equal(64636, 23872);
        }

        /// <summary>
        ///     A non-char input is cast into the loop first (NumPy's buffered cast into uint16: int8 -1 is 65535), and the
        ///     result is char.
        /// </summary>
        [TestMethod]
        public void OtherInput_IsCastIntoTheLoop()
        {
            using var a = np.array(new sbyte[] { -1, 3, 2 });
            using (var s = np.sum(a, Char))
            {
                s.typecode.Should().Be(NPTypeCode.Char);
                Ints(s).Should().Equal(4);
            }
            using (var p = np.prod(a, null, Char))
                Ints(p).Should().Equal(65530);
        }

        /// <summary>
        ///     _mean and _var over the char loop: the sum wraps in uint16, then the float64 quotient is cast back (the flat
        ///     mean summed without wrapping before: 21823), and the variance's sums wrap the same way.
        /// </summary>
        [TestMethod]
        public void MeanAndVar_SumInTheUInt16Loop()
        {
            using var a = Values();
            var engine = a.TensorEngine;
            using (var m = engine.ReduceMean(a, null, false, Char))
            {
                m.typecode.Should().Be(NPTypeCode.Char);
                Ints(m).Should().Equal(10900);
            }
            using (var m = engine.ReduceMean(a, 0, false, Char))
                Ints(m).Should().Equal(51, 151, 32499);
            using (var v = engine.ReduceVar(a, null, false, null, Char))
                Ints(v).Should().Equal(2173);
            using (var v = engine.ReduceVar(a, 1, false, null, Char))
                Ints(v).Should().Equal(19870, 12636);
        }

        /// <summary>
        ///     The result keeps the uint16 route's layout (NumPy's reduction output order): an F-ordered 3-D input reduced
        ///     along axis 1 gives an F-ordered (2, 4) result, owning its data and writeable.
        /// </summary>
        [TestMethod]
        public void Result_KeepsTheReductionLayout()
        {
            using var c = (np.arange(24).reshape(2, 3, 4) * 3000).astype(np.uint16);
            using var f = np.asfortranarray(c);
            using var f3 = f.view(Char);
            using var r = np.sum(f3, 1, Char);
            r.typecode.Should().Be(NPTypeCode.Char);
            r.Shape.IsFContiguous.Should().BeTrue();
            r.Shape.IsContiguous.Should().BeFalse();
            r.@base.Should().BeNull();
            r.Shape.IsWriteable.Should().BeTrue();
            Ints(r).Should().Equal(36000, 45000, 54000, 63000, 12928, 21928, 30928, 39928);
        }

        /// <summary>Empty inputs: the loop's identities (0 for the sum, 1 for the product), as char.</summary>
        [TestMethod]
        public void EmptyInput_GivesTheIdentities()
        {
            using var e = np.zeros(new Shape(0, 3), Char);
            using (var s = np.sum(e, Char))
                Ints(s).Should().Equal(0);
            using (var p = np.prod(e, 0, Char))
            {
                p.typecode.Should().Be(NPTypeCode.Char);
                Ints(p).Should().Equal(1, 1, 1);
            }
        }
    }
}
