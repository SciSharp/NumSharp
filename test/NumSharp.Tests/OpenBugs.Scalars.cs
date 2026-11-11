using System;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Bug pins found while building NumPy's scalar class calls (<c>np.int8(300)</c>, <c>np.str_(5)</c>): the class call
    ///     converts its argument with <c>PyArray_FromAny(x, descr, FORCECAST)</c>, which for one Python value is the
    ///     dtype's setitem, and <see cref="np.asanyarray(in object, DType, string)"/> - NumPy's conversion of the same value
    ///     - converted a C# scalar with a requested dtype by building the scalar in its own dtype and CASTING it: no NEP 50
    ///     range check, and a fixed text width instead of the width of the value's text. Every expectation is NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class OpenBugsScalarTests : TestClass
    {
        /// <summary>
        ///     <c>np.asanyarray(300, dtype=np.int8)</c> raises NumPy's <c>OverflowError: Python integer 300 out of bounds for
        ///     int8</c> (NEP 50: a Python int is packed with the dtype's setitem, never wrapped). NumSharp cast the int32 scalar
        ///     and returned 44.
        /// </summary>
        [TestMethod, OpenBugs]
        public void AsAnyArray_PythonIntOutOfRange_IntoInt8_RaisesOverflowError()
        {
            Action call = () => np.asanyarray(300, np.int8);
            call.Should().Throw<OverflowError>().WithMessage("Python integer 300 out of bounds for int8");
        }

        /// <summary>
        ///     <c>np.asanyarray(5, dtype=np.str_)</c> is <c>array('5', dtype='&lt;U1')</c>: the unsized <c>U</c> class
        ///     discovers its width from the value's <c>str()</c>. NumSharp cast the int32 scalar to a fixed-width string.
        /// </summary>
        [TestMethod, OpenBugs]
        public void AsAnyArray_PythonInt_IntoUnsizedStr_DiscoversTheWidthOfItsText()
        {
            var r = np.asanyarray(5, np.str_);
            r.dtype.str.Should().Be("<U1");
            r.Storage.GetAtIndex(0).Should().Be("5");
        }
    }
}
