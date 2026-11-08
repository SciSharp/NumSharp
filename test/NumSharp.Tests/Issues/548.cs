using System;
using System.Collections.Generic;
using System.Runtime.Intrinsics;
using NumSharp.Backends.Kernels;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #548, "[NEP10 NEP38 NEP54] SIMD and Iterator Performance"
    ///     (https://github.com/SciSharp/NumSharp/issues/548): <c>NDIter</c> is a port of NumPy's <c>NpyIter</c> (axis
    ///     coalescing, <c>order='K'</c> result allocation, buffered iteration with casts, NumPy's casting rules), and the
    ///     kernels run on the widest vector the CPU has, detected at startup. Expected values are NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class Issue548
    {
        /// <summary>
        ///     Axis coalescing: <c>nditer</c> with <c>external_loop</c> over a C-contiguous (2, 3, 4) array hands out ONE inner
        ///     loop of 24 elements, as NumPy's does, instead of one per row.
        /// </summary>
        [TestMethod]
        public void Nditer_ExternalLoop_CoalescesAContiguousArray()
        {
            var a = np.arange(24).reshape(2, 3, 4);
            var chunks = new List<long>();
            using (var it = np.nditer(a, flags: new[] { "external_loop" }))
                foreach (var ops in it)
                    chunks.Add(ops[0].size);

            chunks.Should().Equal(24L);
        }

        /// <summary>
        ///     K-order allocation: a fresh ufunc result over a transposed (F-ordered) operand takes the operand's memory
        ///     order, so <c>a.T + 1</c> is F-contiguous, not C-contiguous, with NumPy's values.
        /// </summary>
        [TestMethod]
        public void UfuncResult_TakesKOrder()
        {
            var t = np.arange(6.0).reshape(2, 3).T + 1;

            t.flags.c_contiguous.Should().BeFalse();
            t.flags.f_contiguous.Should().BeTrue();
            t.Should().BeShaped(3, 2).And.BeOfValues(1.0, 4.0, 2.0, 5.0, 3.0, 6.0);
        }

        /// <summary>
        ///     Buffered iteration with a cast: <c>nditer</c> over an int64 array with <c>buffered</c> and
        ///     <c>op_dtypes=[float64]</c> hands out float64 elements 1.0, 2.0, 3.0.
        /// </summary>
        [TestMethod]
        public void Nditer_Buffered_CastsToTheRequestedDtype()
        {
            var values = new List<double>();
            using (var it = np.nditer(np.array(new long[] { 1, 2, 3 }), flags: new[] { "buffered" }, op_dtypes: new[] { np.float64 }))
                foreach (var ops in it)
                {
                    ops[0].typecode.Should().Be(NPTypeCode.Double);
                    values.Add(ops[0].GetDouble());
                }

            values.Should().Equal(1.0, 2.0, 3.0);
        }

        /// <summary>
        ///     NumPy's casting rules: <c>can_cast</c> answers as NumPy 2.4.2 does, and a ufunc whose float64 loop cannot be
        ///     written into an int64 <c>out</c> under <c>same_kind</c> raises NumPy's text.
        /// </summary>
        [TestMethod]
        public void CastingRules_AreNumPys()
        {
            np.can_cast(np.int64, np.int32, "same_kind").Should().BeTrue();
            np.can_cast(np.int64, np.int32, "safe").Should().BeFalse();
            np.can_cast(np.float64, np.int64, "same_kind").Should().BeFalse();

            var xi = np.array(new long[] { 1, 2 });
            new Action(() => np.add(xi, 1.5, @out: xi)).Should().Throw<TypeError>()
                .WithMessage("Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
        }

        /// <summary>
        ///     The vector width is the widest the CPU accelerates (Vector512, else Vector256, else Vector128), detected once at
        ///     startup.
        /// </summary>
        [TestMethod]
        public void VectorWidth_IsDetectedFromTheHardware()
        {
            int expected = Vector512.IsHardwareAccelerated ? 512 :
                Vector256.IsHardwareAccelerated ? 256 :
                Vector128.IsHardwareAccelerated ? 128 : 0;

            DirectILKernelGenerator.VectorBits.Should().Be(expected);
        }
    }
}
