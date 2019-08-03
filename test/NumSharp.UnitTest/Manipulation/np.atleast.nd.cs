using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class np_atleast_tests
    {
        [TestMethod]
        public void alteast_3d()
        {
            var a = np.atleast_3d(3.0);
            a.Shape.NDim.Should().Be(3);
            a.Shape.Should().BeEquivalentTo(new Shape(1, 1, 1));

            a = np.atleast_3d(NDArray.Scalar(3.0));
            a.Shape.NDim.Should().Be(3);
            a.Shape.Should().BeEquivalentTo(new Shape(1, 1, 1));

            a = np.atleast_3d(np.arange(3));
            a.Shape.NDim.Should().Be(3);
            a.Shape.Should().BeEquivalentTo(new Shape(1, 3, 1));

            a = np.atleast_3d(np.arange(12.0).reshape(4, 3));
            a.Shape.NDim.Should().Be(3);
            a.Shape.Should().BeEquivalentTo(new Shape(4, 3, 1));
        }

        [TestMethod]
        public void alteast_2d()
        {
            var a = np.atleast_2d(3.0);
            a.Shape.NDim.Should().Be(2);
            a.Shape.Should().BeEquivalentTo(new Shape(1, 1));
            
            a = np.atleast_2d(NDArray.Scalar(3.0));
            a.Shape.NDim.Should().Be(2);
            a.Shape.Should().BeEquivalentTo(new Shape(1, 1));

            a = np.atleast_2d(np.arange(3));
            a.Shape.NDim.Should().Be(2);
            a.Shape.Should().BeEquivalentTo(new Shape(1, 3));
        }

        [TestMethod]
        public void alteast_1d()
        {
            var a = np.atleast_1d(3.0);
            a.Shape.NDim.Should().Be(1);
            a.Shape.Should().BeEquivalentTo(new Shape(1));

            a = np.atleast_1d(NDArray.Scalar(3.0));
            a.Shape.NDim.Should().Be(1);
            a.Shape.Should().BeEquivalentTo(new Shape(1));
        }
    }
}
