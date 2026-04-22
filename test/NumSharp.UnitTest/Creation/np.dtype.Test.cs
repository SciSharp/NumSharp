using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    ///     Core <see cref="np.dtype(string)"/> smoke tests. Full NumPy-parity coverage lives in
    ///     <c>DTypeStringParityTests</c>.
    /// </summary>
    [TestClass]
    public class np_dtype_tests
    {
        [TestMethod]
        public void Case1_ValidForms()
        {
            np.dtype("?").type.Should().Be<bool>();
            np.dtype("i4").type.Should().Be<Int32>();
            np.dtype("i8").type.Should().Be<Int64>();
            np.dtype("f").type.Should().Be<float>();
            np.dtype("f8").type.Should().Be<double>();
            np.dtype("double").type.Should().Be<double>();
        }

        [TestMethod]
        public void Case2_InvalidFormsThrow()
        {
            // NumPy parity: these are not valid dtype strings, NumPy raises TypeError.
            Action act;
            act = () => np.dtype("?64"); act.Should().Throw<NotSupportedException>();
            act = () => np.dtype("d8");  act.Should().Throw<NotSupportedException>();
            act = () => np.dtype("single16"); act.Should().Throw<NotSupportedException>();
            act = () => np.dtype("f16"); act.Should().Throw<NotSupportedException>();
        }
    }
}
