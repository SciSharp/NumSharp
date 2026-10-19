using System;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The port of NumPy 2.4.2's reduction loop resolution (<see cref="NDReduceLikeResolution"/>:
    ///     <c>reducelike_promote_and_resolve</c> with its legacy-resolver fallback) for add / multiply reductions and
    ///     accumulations that involve text. Each case pins what fresh NumPy processes report — the loop kind and the descriptors
    ///     where NumPy computes, its error text where it refuses. NumPy's dispatch cache makes the answer depend on the calls a
    ///     process made before (a cached loop is reused for later operand classes), so the expected values were generated one
    ///     fresh process per case, which is the behavior ported.
    /// </summary>
    [TestClass]
    public class ReduceLikeResolutionTests
    {
        private static DType U1 => np.dtype("<U1");

        private static DType U6 => np.dtype("<U6");

        private static DType S1 => np.dtype("S1");

        private static DType S6 => np.dtype("S6");

        private static DType StringDType => np.dtypes.StringDType();

        [TestMethod]
        public void NumericRequest_OverText_PicksTheNumericLoop()
        {
            // add.reduce(u, dtype=float64, out=int64): the request's loop, every descriptor float64 (the text is cast in).
            var r = NDReduceLikeResolution.Resolve("add", false, U1, np.int64, np.float64.Meta);
            r.Kind.Should().Be(ReduceLoopKind.Numeric);
            r.Descriptors.Should().HaveCount(3);
            foreach (var d in r.Descriptors)
                d.Should().Be(np.float64);
        }

        [TestMethod]
        public void TwoStringsOfOneClass_IgnoreTheRequest_AndFailTheEquivalenceCheck()
        {
            // AdditionTypeResolver takes the string add loop for two U operands, whatever dtype= asked for: the output is wider
            // than the first operand, so the reduction is refused with the descriptors it resolved.
            Action act = () => NDReduceLikeResolution.Resolve("add", false, U1, U6, np.float64.Meta);
            act.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U6'), dtype('<U1'), dtype('<U7'))");
        }

        [TestMethod]
        public void MixedStringClasses_AreTheAdditionResolversRefusal()
        {
            Action act = () => NDReduceLikeResolution.Resolve("add", false, U1, S6, np.float64.Meta);
            act.Should().Throw<TypeError>().WithMessage("ufunc 'add' cannot use operands with types dtype('S6') and dtype('<U1')");
        }

        [TestMethod]
        public void IntegerCountsIntoText_TakeTheRepeatLoop_InTheOutsText()
        {
            // multiply.reduce(counts, out=U6): the repeat loop with the count second; the out keeps its own descriptor.
            var r = NDReduceLikeResolution.Resolve("multiply", false, np.int64, U6, null);
            r.Kind.Should().Be(ReduceLoopKind.StringRepeat);
            r.Descriptors[0].Should().Be(U6);
            r.Descriptors[2].Should().Be(U6);
        }

        [TestMethod]
        public void TextRequestWithNumericOut_RepeatsInTheOutCastToTheRequestedClass()
        {
            // multiply.reduce(counts, dtype='U', out=int64): MultiplicationTypeResolver takes the repeat loop, and the int64 out
            // cast to the unicode class reads as U21 (the width of int64's str()).
            var r = NDReduceLikeResolution.Resolve("multiply", false, np.int64, np.int64, np.dtype("U").Meta);
            r.Kind.Should().Be(ReduceLoopKind.StringRepeat);
            r.Descriptors[0].Should().Be(np.dtype("<U21"));
        }

        [TestMethod]
        public void RepeatAsAnAccumulation_IsRefused()
        {
            // An integer out with a bytes input: the repeat loop's output would be text where its first operand is an integer.
            Action act = () => NDReduceLikeResolution.Resolve("multiply", true, S1, np.int64, np.float64.Meta);
            act.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with multiply.accumulate. Resolved (dtype('int64'), dtype('S1'), " +
                "dtype('S21'))");
        }

        [TestMethod]
        public void TextRequestIntoStringDTypeOut_HasNoMatchingLoop()
        {
            // add.reduce(int64, dtype='U', out=StringDType()): the request fixes the first and last slots to StrDType, which
            // the StringDType out cannot be cast to, so the type-tuple search finds no loop.
            Action act = () => NDReduceLikeResolution.Resolve("add", false, np.int64, StringDType, np.dtype("U").Meta);
            act.Should().Throw<TypeError>().WithMessage("No loop matching the specified signature and casting was found for ufunc add");
        }
    }
}
