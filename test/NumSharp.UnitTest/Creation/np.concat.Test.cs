using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    ///     np.concat — the NumPy 2.0 Array-API alias of np.concatenate
    ///     (numpy/_core/__init__.py: concat = numeric.concatenate).
    /// </summary>
    [TestClass]
    public class np_concat_test : TestClass
    {
        [TestMethod]
        public void Basic()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 4, 5, 6 });
            var r = np.concat(new[] { a, b });
            r.Data<int>().Should().Equal(1, 2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void TupleForm_And_Axis()
        {
            var a = np.arange(6).reshape(2, 3);
            var b = (np.arange(6) + 6).reshape(2, 3);
            var r = np.concat((a, b), axis: 1);
            r.shape.Should().Equal(2L, 6L);
            r.Data<int>().Should().Equal(0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11);
        }

        [TestMethod]
        public void AxisNull_Flattens()
        {
            var r = np.concat(new[] { np.arange(4).reshape(2, 2), np.arange(2).reshape(1, 2) }, axis: null);
            r.shape.Should().Equal(6L);
            r.Data<int>().Should().Equal(0, 1, 2, 3, 0, 1);
        }

        [TestMethod]
        public void KeywordArguments_MatchConcatenate()
        {
            var a = np.array(new[] { 1, 2 });
            var b = np.array(new[] { 3, 4 });

            // dtype=
            np.concat(new[] { a, b }, dtype: NPTypeCode.Double).typecode.Should().Be(NPTypeCode.Double);

            // out=
            var @out = np.zeros(new Shape(4), NPTypeCode.Int32);
            var r = np.concat(new[] { a, b }, @out: @out);
            ReferenceEquals(r, @out).Should().BeTrue();
            @out.Data<int>().Should().Equal(1, 2, 3, 4);

            // casting validation flows through
            var act = () => np.concat(new[] { a, np.array(new[] { 0.5 }) }, casting: "no");
            act.Should().Throw<InvalidCastException>();
        }

        [TestMethod]
        public void SameResultsAsConcatenate()
        {
            var a = np.arange(12).reshape(3, 4);
            var b = (np.arange(12) + 100).reshape(3, 4);
            ((bool)np.array_equal(np.concat(new[] { a, b }, 0), np.concatenate(new[] { a, b }, 0))).Should().BeTrue();
            ((bool)np.array_equal(np.concat(new[] { a, b }, -1), np.concatenate(new[] { a, b }, -1))).Should().BeTrue();
        }
    }
}
