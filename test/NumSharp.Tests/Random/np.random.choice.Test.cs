namespace NumSharp.Tests.Random
{
    [TestClass]
    public class NpRandomChoiceTests : TestClass
    {
        [TestMethod]
        public void UniformOneSample()
        {
            // Generate a uniform random sample from np.arange(5) of size 1:
            // This is equivalent to np.random.randint(0,5,1)
            int low = 0;
            int high = 5;
            int nrSamples = 1;

            NDArray actual = np.random.choice(high); // Not specifying size means 1 single value is wanted

            Assert.AreEqual(actual.size, nrSamples, "Unexpected number of elements");

            // Verify that all elements in output are within the range
            for (int i = 0; i < actual.size; i++)
            {
                Assert.IsTrue(System.Convert.ToInt64(actual.GetAtIndex(i)) >= low, "Element was less than expected");
                Assert.IsTrue(System.Convert.ToInt64(actual.GetAtIndex(i)) < high, "Element was greater than expected");
            }
        }

        [TestMethod]
        public void UniformMultipleSample()
        {
            // Generate a uniform random sample from np.arange(5) of size 3:
            // This is equivalent to np.random.randint(0,5,3)
            int low = 0;
            int high = 5;
            int nrSamples = 3;

            NDArray actual = np.random.choice(high, (Shape)nrSamples);

            Assert.AreEqual(actual.size, nrSamples, "Unexpected number of elements");

            // Verify that all elements in output are within the range
            for (int i = 0; i < actual.size; i++)
            {
                Assert.IsTrue(System.Convert.ToInt64(actual.GetAtIndex(i)) >= low, "Element was less than expected");
                Assert.IsTrue(System.Convert.ToInt64(actual.GetAtIndex(i)) < high, "Element was greater than expected");
            }
        }

        [TestMethod]
        public void NonUniformSample()
        {
            // Generate a non-uniform random sample from np.arange(5) of size 3:
            int low = 0;
            int high = 5;
            int nrSamples = 3;
            double[] probabilities = new double[] {0.1, 0, 0.3, 0.6, 0};

            NDArray actual = np.random.choice(5, (Shape) nrSamples, p: probabilities);

            Assert.AreEqual(actual.size, nrSamples, "Unexpected number of elements");

            // Verify that all elements in output are within the range
            for (int i = 0; i < actual.size; i++)
            {
                Assert.IsTrue(System.Convert.ToInt64(actual.GetAtIndex(i)) >= low, "Element was less than expected");
                Assert.IsTrue(System.Convert.ToInt64(actual.GetAtIndex(i)) < high, "Element was greater than expected");
                Assert.IsTrue(System.Convert.ToInt64(actual.GetAtIndex(i)) != 1, "Sampled zero-probability element");
                Assert.IsTrue(System.Convert.ToInt64(actual.GetAtIndex(i)) != 4, "Sampled zero-probability element");
            }
        }

        [TestMethod]
        public void UniformSampleWithoutReplace()
        {
            // replace=False draws distinct members of arange(5) (NumPy: permutation(5)[:3]).
            NDArray actual = np.random.choice(5, (Shape)3, replace: false);
            Assert.AreEqual(3, actual.size, "Unexpected number of elements");
            var seen = new System.Collections.Generic.HashSet<long>();
            for (int i = 0; i < actual.size; i++)
            {
                long v = System.Convert.ToInt64(actual.GetAtIndex(i));
                Assert.IsTrue(v >= 0 && v < 5, "Element out of range");
                Assert.IsTrue(seen.Add(v), "Duplicate element without replacement");
            }
        }

        [TestMethod]
        public void NonUniformSampleWithoutReplace()
        {
            // Exactly three non-zero probabilities and three samples without replacement: the draw must be
            // {0, 2, 3} in some order, never a zero-probability member.
            double[] probabilities = new double[] {0.1, 0, 0.3, 0.6, 0};
            NDArray actual = np.random.choice(5, (Shape)3, replace: false, p: probabilities);
            Assert.AreEqual(3, actual.size, "Unexpected number of elements");
            var got = new System.Collections.Generic.HashSet<long>();
            for (int i = 0; i < actual.size; i++)
                got.Add(System.Convert.ToInt64(actual.GetAtIndex(i)));
            Assert.IsTrue(got.SetEquals(new long[] { 0, 2, 3 }), "Expected exactly the three non-zero-probability members");
        }

        [TestMethod]
        [Ignore("Choice with string arrays not implemented yet")]
        public void StringArraySample1()
        {
            //int nrSamples = 5;

            //NDArray aa_milne_arr = new string[] { "pooh", "rabbit", "piglet", "Christopher" };
            //double[] probabilities = new double[] { 0.5, 0.1, 0.0, 0.3 };

            //NDArray actual = np.random.choice(aa_milne_arr, nrSamples, p: probabilities);

            //Assert.AreEqual(actual.len, nrSamples, "Unexpected number of elements");

            //// Verify that all elements in output are within the possibilities
            //for (int i = 0; i < actual.len; i++)
            //{
            //    Assert.IsTrue((string)actual[i] != (string)aa_milne_arr[2], "Sampled zero-probability element");
            //}
        }

        [TestMethod]
        public void IntegerArraySample()
        {
            int nrSamples = 5;

            NDArray int_arr = new int[] {42, 96, 3, 101};
            double[] probabilities = new double[] {0.5, 0.2, 0.0, 0.3};

            NDArray actual = np.random.choice(int_arr, (Shape)nrSamples, p: probabilities);

            Assert.AreEqual(actual.size, nrSamples, "Unexpected number of elements");

            // Verify that all elements in output are within the possibilities
            for (int i = 0; i < actual.size; i++)
            {
                Assert.IsTrue((int)actual[i] != (int)int_arr[2], "Sampled zero-probability element");
            }
        }
    }
}
