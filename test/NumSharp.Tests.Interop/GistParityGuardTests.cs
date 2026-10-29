using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Interop;

[TestClass]
public class GistParityGuardTests : InteropTestBase
{
    [TestMethod]
    public void ParityGuard_RejectsSameBytesWithWrongShapeOrDtype()
    {
        using var actual = np.zeros(2, dtype: np.float64);
        using (Gil())
        {
            using var wrongShape = Scope.Eval("np.zeros((1,2),dtype=np.float64)");
            using var wrongDtype = Scope.Eval("np.zeros(2,dtype=np.int64)");
            Assert.ThrowsException<AssertFailedException>(() => GistParity.AssertExact(actual, wrongShape, "wrong shape"));
            Assert.ThrowsException<AssertFailedException>(() => GistParity.AssertExact(actual, wrongDtype, "wrong dtype"));
        }
    }

    [TestMethod]
    public void ParityGuard_EnforcesExplicitUlpCapOnNegativeValues()
    {
        using var actual = np.array(new[] { System.Math.BitIncrement(-1.0) });
        using (Gil())
        {
            using var expected = Scope.Eval("np.array([-1.],dtype=np.float64)");
            GistParity.AssertUlps(actual, expected, 1, "one adjacent negative float");
            Assert.ThrowsException<AssertFailedException>(() => GistParity.AssertUlps(actual, expected, 0, "no allowance"));
        }
    }

    [TestMethod]
    public void ParityGuard_DoesNotHideDifferentZeroOrNanBits()
    {
        using var negativeZero = np.array(new[] { BitConverter.Int64BitsToDouble(long.MinValue) });
        using var negativeNan = np.array(new[] { BitConverter.Int64BitsToDouble(unchecked((long)0xfff8000000000000)) });
        using (Gil())
        {
            using var zero = Scope.Eval("np.array([0.],dtype=np.float64)");
            using var nan = Scope.Eval("np.array([np.nan],dtype=np.float64)");
            Assert.ThrowsException<AssertFailedException>(() => GistParity.AssertUlps(negativeZero, zero, 10, "signed zero"));
            Assert.ThrowsException<AssertFailedException>(() => GistParity.AssertUlps(negativeNan, nan, 10, "NaN bits"));
        }
    }
}
