using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.APIs
{
    /// <summary>
    ///     NumPy 2.x parity tests for <c>np.iinfo(int8 / sbyte)</c>. Verified against
    ///     <c>python -c "import numpy as np; i = np.iinfo(np.int8); print(i.bits, i.min, i.max)"</c>.
    /// </summary>
    [TestClass]
    public class NpIInfoNewDtypesTests
    {
        [TestMethod]
        public void IInfo_SByte_Bits() =>
            np.iinfo(NPTypeCode.SByte).bits.Should().Be(8);

        [TestMethod]
        public void IInfo_SByte_Min() =>
            np.iinfo(NPTypeCode.SByte).min.Should().Be(-128);

        [TestMethod]
        public void IInfo_SByte_Max() =>
            np.iinfo(NPTypeCode.SByte).max.Should().Be(127);

        [TestMethod]
        public void IInfo_SByte_Kind() =>
            np.iinfo(NPTypeCode.SByte).kind.Should().Be('i');

        [TestMethod]
        public void IInfo_SByte_Dtype() =>
            np.iinfo(NPTypeCode.SByte).dtype.Should().Be(NPTypeCode.SByte);

        [TestMethod]
        public void IInfo_SByte_MaxUnsigned() =>
            np.iinfo(NPTypeCode.SByte).maxUnsigned.Should().Be(127);

        [TestMethod]
        public void IInfo_SByte_From_Type() =>
            np.iinfo(typeof(sbyte)).bits.Should().Be(8);

        [TestMethod]
        public void IInfo_SByte_From_Generic() =>
            np.iinfo<sbyte>().bits.Should().Be(8);

        [TestMethod]
        public void IInfo_SByte_From_Array()
        {
            var arr = np.array(new sbyte[] { 1, 2, 3 });
            np.iinfo(arr).bits.Should().Be(8);
        }

        [TestMethod]
        public void IInfo_SByte_From_String_int8() =>
            np.iinfo("int8").bits.Should().Be(8);

        [TestMethod]
        public void IInfo_SByte_From_String_sbyte() =>
            np.iinfo("sbyte").bits.Should().Be(8);

        [TestMethod]
        public void IInfo_SByte_From_String_b() =>
            np.iinfo("b").bits.Should().Be(8);

        [TestMethod]
        public void IInfo_SByte_From_String_i1() =>
            np.iinfo("i1").bits.Should().Be(8);

        [TestMethod]
        public void IInfo_Half_Throws()
        {
            // NumPy 2.x: np.iinfo(np.float16) raises ValueError. NumSharp: ArgumentException.
            Action act = () => np.iinfo(NPTypeCode.Half);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void IInfo_Complex_Throws()
        {
            Action act = () => np.iinfo(NPTypeCode.Complex);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void IInfo_SByte_ToString_IncludesCorrectRange()
        {
            // NumPy: "iinfo(min=-128, max=127, dtype=int8)"
            var s = np.iinfo(NPTypeCode.SByte).ToString();
            s.Should().Contain("-128");
            s.Should().Contain("127");
            s.Should().Contain("int8");
        }
    }
}
