using System;

namespace NumSharp.Tests.Math
{
    [TestClass]
    public class NDArrayNegateTest : TestClass
    {
        [TestMethod]
        public void NegateArray()
        {
            //initialization
            var nd = new NDArray(np.float32, 3);
            nd.ReplaceData(new float[] {1, -2, 3.3f});

            //perform test
            nd = -nd;

            //assertions
            nd.Data<float>().Should().BeEquivalentTo(new float[] {-1, 2, -3.3f});
        }

        [TestMethod]
        public void NegateArray2()
        {
            //initialization
            var nd = new NDArray(np.int32, 3);
            nd.ReplaceData(new int[] {-1, 0, 1});

            //perform test
            nd = -nd;

            //assertions
            nd.Data<int>().Should().BeEquivalentTo(new int[] {1, 0, -1});
        }

        [TestMethod]
        public void NegateArray3()
        {
            //initialization
            var nd = new NDArray(np.uint32, 3);
            nd.ReplaceData(new uint[] {0, 1, 2});

            //perform test
            nd = -nd;
            Console.WriteLine(nd.ToString(false));

            //assertions
            nd.Data<uint>().Should().BeEquivalentTo(new uint[] {0, 4294967295, 4294967294});
        }

        [TestMethod]
        public void NegateArray4()
        {
            //initialization
            var nd = new NDArray(np.uint64, 3);
            nd.ReplaceData(new ulong[] {0, 1, 2});

            //perform test
            nd = -nd;
            Console.WriteLine(nd.ToString(false));

            //assertions
            nd.Data<ulong>().Should().BeEquivalentTo(new ulong[] {0, 18446744073709551615, 18446744073709551614});
        }

        [TestMethod]
        public void AddArray()
        {
            //initialization
            var nd = new NDArray(np.float32, 3);
            var input = new float[] {1, -2, 3.3f};
            nd.ReplaceData(input);

            //perform test
            nd = +nd;

            //assertions
            nd.Data<float>().Should().BeEquivalentTo(new float[] {1, -2, 3.3f});
        }

        [TestMethod]
        public void NegateEmptyArray()
        {
            //initialization
            var nd1 = new NDArray(np.float32, 0);

            //perform test
            var nd2 = -nd1;

            //assertions
            Assert.IsTrue(nd1.size == nd2.size);
        }

        /// <summary>
        ///     The unary <c>+</c> is NumPy's positive ufunc (probed 2.4.2): a bool operand has no loop, empty and 0-d ones
        ///     included (<c>ufunc 'positive' did not contain a loop with signature matching types &lt;class
        ///     'numpy.dtypes.BoolDType'&gt; -&gt; None</c>), and an F-ordered operand gives an F-ordered copy that owns its
        ///     data. Until the np-layer dtype plan's P1c the operator returned <c>x.copy()</c>: a bool copy, C-ordered.
        ///     <c>ndarray.positive()</c> and <c>np.positive</c> answer the same.
        /// </summary>
        [TestMethod]
        public void UnaryPlus_IsNpPositive_BoolRefusedAndLayoutKept()
        {
            const string boolText =
                "ufunc 'positive' did not contain a loop with signature matching types <class 'numpy.dtypes.BoolDType'> -> None";
            using var flags = np.array(new[] { true, false });
            using var none = np.zeros(new Shape(0, 3), np.@bool);
            using var zeroD = np.array(true);
            foreach (var b in new[] { flags, none, zeroD })
            {
                Action plus = () => { using var r = +b; };
                plus.Should().Throw<TypeError>().WithMessage(boolText);
                Action method = () => { using var r = b.positive(); };
                method.Should().Throw<TypeError>().WithMessage(boolText);
            }

            using var basis = np.arange(12).astype(np.int8).reshape(3, 4);
            using var f = np.asfortranarray(basis);
            using (var r = +f)
            {
                r.flags.f_contiguous.Should().BeTrue("NumPy's ufunc output keeps an F-ordered operand's order");
                r.flags.c_contiguous.Should().BeFalse();
                r.flags.owndata.Should().BeTrue();
                ReferenceEquals(r, f).Should().BeFalse();
                r.ToArray<sbyte>().Should().Equal(f.ToArray<sbyte>());
            }
        }
    }
}
