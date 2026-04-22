using System;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    ///     Tests for NumPy's platform-dependent integer dtypes. The divergence is specifically
    ///     around the C <c>long</c> type: 32-bit on Windows (MSVC), 64-bit on Linux/Mac LP64 (gcc).
    ///
    ///     Affected NumPy spellings:
    ///     <list type="bullet">
    ///       <item><c>'l'</c>, <c>'L'</c> — single-char codes for signed/unsigned C long</item>
    ///       <item><c>'long'</c>, <c>'ulong'</c> — named forms</item>
    ///     </list>
    ///
    ///     Not affected (always 64-bit on 64-bit platforms in NumPy 2.x):
    ///     <list type="bullet">
    ///       <item><c>'int'</c>, <c>'int_'</c>, <c>'intp'</c> → <c>intp</c> = int64 on 64-bit</item>
    ///       <item><c>'p'</c>, <c>'P'</c> → <c>intptr</c> / <c>uintptr</c></item>
    ///       <item><c>'q'</c>, <c>'Q'</c>, <c>'longlong'</c>, <c>'ulonglong'</c> → always 64-bit</item>
    ///     </list>
    /// </summary>
    [TestClass]
    public class DTypePlatformDivergenceTests
    {
        private static bool IsWindows => RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        private static bool Is64Bit  => IntPtr.Size == 8;

        /// <summary>Expected NumPy int type for C long on the current platform.</summary>
        private static NPTypeCode ExpectedCLong =>
            (IsWindows || !Is64Bit) ? NPTypeCode.Int32 : NPTypeCode.Int64;
        private static NPTypeCode ExpectedCULong =>
            (IsWindows || !Is64Bit) ? NPTypeCode.UInt32 : NPTypeCode.UInt64;
        private static NPTypeCode ExpectedIntp =>
            Is64Bit ? NPTypeCode.Int64 : NPTypeCode.Int32;
        private static NPTypeCode ExpectedUIntp =>
            Is64Bit ? NPTypeCode.UInt64 : NPTypeCode.UInt32;

        // ---------------------------------------------------------------------
        // 'l' and 'L' — C long / unsigned long (platform-dependent)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void SingleChar_l_MatchesPlatformCLong() =>
            np.dtype("l").typecode.Should().Be(ExpectedCLong);

        [TestMethod]
        public void SingleChar_L_MatchesPlatformCULong() =>
            np.dtype("L").typecode.Should().Be(ExpectedCULong);

        [TestMethod]
        public void Named_long_MatchesPlatformCLong() =>
            np.dtype("long").typecode.Should().Be(ExpectedCLong);

        [TestMethod]
        public void Named_ulong_MatchesPlatformCULong() =>
            np.dtype("ulong").typecode.Should().Be(ExpectedCULong);

        // ---------------------------------------------------------------------
        // 'int', 'int_', 'intp', 'p', 'P' — always intp (pointer-sized) in NumPy 2.x
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Named_int_MatchesIntp() =>
            np.dtype("int").typecode.Should().Be(ExpectedIntp);

        [TestMethod]
        public void Named_intUnderscore_MatchesIntp() =>
            np.dtype("int_").typecode.Should().Be(ExpectedIntp);

        [TestMethod]
        public void Named_intp_MatchesPointerSize() =>
            np.dtype("intp").typecode.Should().Be(ExpectedIntp);

        [TestMethod]
        public void Named_uintp_MatchesPointerSize() =>
            np.dtype("uintp").typecode.Should().Be(ExpectedUIntp);

        [TestMethod]
        public void SingleChar_p_MatchesIntp() =>
            np.dtype("p").typecode.Should().Be(ExpectedIntp);

        [TestMethod]
        public void SingleChar_P_MatchesUIntp() =>
            np.dtype("P").typecode.Should().Be(ExpectedUIntp);

        [TestMethod]
        public void Named_uint_MatchesUIntp() =>
            np.dtype("uint").typecode.Should().Be(ExpectedUIntp);

        // ---------------------------------------------------------------------
        // 'q', 'Q', 'longlong', 'ulonglong' — always 64-bit across platforms
        // ---------------------------------------------------------------------

        [TestMethod]
        public void SingleChar_q_AlwaysInt64() =>
            np.dtype("q").typecode.Should().Be(NPTypeCode.Int64);

        [TestMethod]
        public void SingleChar_Q_AlwaysUInt64() =>
            np.dtype("Q").typecode.Should().Be(NPTypeCode.UInt64);

        [TestMethod]
        public void Named_longlong_AlwaysInt64() =>
            np.dtype("longlong").typecode.Should().Be(NPTypeCode.Int64);

        [TestMethod]
        public void Named_ulonglong_AlwaysUInt64() =>
            np.dtype("ulonglong").typecode.Should().Be(NPTypeCode.UInt64);

        // ---------------------------------------------------------------------
        // 'i', 'I' — always 32-bit (NumPy specifies these as fixed int32/uint32)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void SingleChar_i_AlwaysInt32() =>
            np.dtype("i").typecode.Should().Be(NPTypeCode.Int32);

        [TestMethod]
        public void SingleChar_I_AlwaysUInt32() =>
            np.dtype("I").typecode.Should().Be(NPTypeCode.UInt32);

        [TestMethod]
        public void Sized_i4_AlwaysInt32() =>
            np.dtype("i4").typecode.Should().Be(NPTypeCode.Int32);

        [TestMethod]
        public void Sized_u4_AlwaysUInt32() =>
            np.dtype("u4").typecode.Should().Be(NPTypeCode.UInt32);

        // ---------------------------------------------------------------------
        // 'h', 'H' — always 16-bit
        // ---------------------------------------------------------------------

        [TestMethod]
        public void SingleChar_h_AlwaysInt16() =>
            np.dtype("h").typecode.Should().Be(NPTypeCode.Int16);

        [TestMethod]
        public void SingleChar_H_AlwaysUInt16() =>
            np.dtype("H").typecode.Should().Be(NPTypeCode.UInt16);

        // ---------------------------------------------------------------------
        // Consistency: np.int_ direct access aligns with np.dtype("int_")
        // ---------------------------------------------------------------------

        [TestMethod]
        public void NpInt_Consistent_With_DtypeIntUnderscore()
        {
            // NumPy 2.x: np.int_ and np.dtype("int_") both resolve to intp.
            np.int_.Should().Be(np.dtype("int_").type);
        }

        [TestMethod]
        public void NpIntp_Consistent_With_DtypeIntp()
        {
            // np.intp is typeof(nint). On 64-bit, nint has 8 bytes (same as int64).
            IntPtr.Size.Should().Be(Is64Bit ? 8 : 4);
            if (Is64Bit)
                np.dtype("intp").typecode.Should().Be(NPTypeCode.Int64);
        }
    }
}
