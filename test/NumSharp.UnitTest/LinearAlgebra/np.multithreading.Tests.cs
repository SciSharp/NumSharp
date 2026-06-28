using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    /// np.multithreading(...) global toggle + the parallel 1-D dot path it controls.
    /// Each test resets the flag in a finally so it cannot leak into other tests
    /// (small dots never parallelize, so the blast radius is already nil).
    /// </summary>
    [TestClass]
    public class np_multithreading_test : TestClass
    {
        [TestMethod]
        public void Api_SetsState_AndClampsMaxThreads()
        {
            try
            {
                np.multithreading(true, 4);
                Assert.IsTrue(MultiThread.Enabled);
                Assert.AreEqual(4, MultiThread.MaxThreads);

                np.multithreading(false);
                Assert.IsFalse(MultiThread.Enabled);

                np.multithreading(true, -5);            // clamps to >= 1
                Assert.AreEqual(1, MultiThread.MaxThreads);
            }
            finally { np.multithreading(false); MultiThread.MaxThreads = 8; }
        }

        // Small/medium work stays single-threaded; only large work parallelizes; disabled => always 1.
        [TestMethod]
        public void Gate_OnlyParallelizesLargeWork()
        {
            try
            {
                np.multithreading(true, 8);
                Assert.AreEqual(1, MultiThread.DegreeOfParallelism(1_000));
                Assert.AreEqual(1, MultiThread.DegreeOfParallelism(49_999));   // below MinTotalWork
                Assert.IsTrue(MultiThread.DegreeOfParallelism(1_000_000) > 1); // large -> parallel
            }
            finally { np.multithreading(false); }

            Assert.AreEqual(1, MultiThread.DegreeOfParallelism(1_000_000));    // disabled -> single thread
        }

        [TestMethod]
        public void Gate_RespectsMaxThreads()
        {
            try
            {
                np.multithreading(true, 2);
                Assert.AreEqual(2, MultiThread.DegreeOfParallelism(100_000_000)); // capped by max_threads
            }
            finally { np.multithreading(false); MultiThread.MaxThreads = 8; }
        }

        // The parallel dot must agree with the single-threaded dot (to FP tolerance).
        [TestMethod]
        public void ParallelDot_MatchesSequential_Double()
        {
            var a = np.arange(200_000.0);
            var b = np.arange(200_000.0);
            try
            {
                np.multithreading(false);
                double seq = np.dot(a, b).GetAtIndex<double>(0);

                np.multithreading(true, 8);
                double par = np.dot(a, b).GetAtIndex<double>(0);

                Assert.AreEqual(seq, par, Math.Abs(seq) * 1e-12);
            }
            finally { np.multithreading(false); }
        }

        [TestMethod]
        public void ParallelDot_MatchesSequential_Single()
        {
            var a = np.arange(200_000.0).astype(NPTypeCode.Single);
            var b = np.arange(200_000.0).astype(NPTypeCode.Single);
            try
            {
                np.multithreading(false);
                float seq = np.dot(a, b).GetAtIndex<float>(0);

                np.multithreading(true, 8);
                float par = np.dot(a, b).GetAtIndex<float>(0);

                Assert.AreEqual(seq, par, Math.Abs(seq) * 1e-4f);
            }
            finally { np.multithreading(false); }
        }

        // Exact case: full(2) · full(3) over 200k elements = 1,200,000 regardless of threading.
        [TestMethod]
        public void ParallelDot_ExactValue()
        {
            var a = np.full(new Shape(200_000L), 2.0);
            var b = np.full(new Shape(200_000L), 3.0);
            try
            {
                np.multithreading(true, 8);
                Assert.AreEqual(1_200_000.0, np.dot(a, b).GetAtIndex<double>(0), 1e-6);
            }
            finally { np.multithreading(false); }
        }
    }
}
