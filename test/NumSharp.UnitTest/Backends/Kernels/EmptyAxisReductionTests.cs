using System;
using System.Threading.Tasks;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels
{
    /// <summary>
    /// Tests for empty array axis reduction behavior.
    /// NumPy: np.sum(np.zeros((0,3)), axis=0) returns array([0., 0., 0.]) with shape (3,)
    /// </summary>
    public class EmptyAxisReductionTests
    {
        #region Sum Tests

        [TestMethod]
        public async Task Sum_EmptyAxis0_ReturnsZerosWithReducedShape()
        {
            // NumPy: np.sum(np.zeros((0, 3)), axis=0) returns array([0., 0., 0.]) with shape (3,)
            var arr = np.zeros(new Shape(0, 3));
            var result = np.sum(arr, axis: 0);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 3 });
            await Assert.That(result.GetDouble(0)).IsEqualTo(0.0);
            await Assert.That(result.GetDouble(1)).IsEqualTo(0.0);
            await Assert.That(result.GetDouble(2)).IsEqualTo(0.0);
        }

        [TestMethod]
        public async Task Sum_EmptyAxis1_ReturnsEmptyArray()
        {
            // NumPy: np.sum(np.zeros((0, 3)), axis=1) returns array([]) with shape (0,)
            var arr = np.zeros(new Shape(0, 3));
            var result = np.sum(arr, axis: 1);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 0 });
            await Assert.That(result.size).IsEqualTo(0);
        }

        [TestMethod]
        public async Task Sum_EmptyAxis0_Keepdims_ReturnsCorrectShape()
        {
            // NumPy: np.sum(np.zeros((0, 3)), axis=0, keepdims=True) returns shape (1, 3)
            var arr = np.zeros(new Shape(0, 3));
            var result = np.sum(arr, axis: 0, keepdims: true);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 1, 3 });
        }

        [TestMethod]
        public async Task Sum_EmptyAxis1_Keepdims_ReturnsCorrectShape()
        {
            // NumPy: np.sum(np.zeros((0, 3)), axis=1, keepdims=True) returns shape (0, 1)
            var arr = np.zeros(new Shape(0, 3));
            var result = np.sum(arr, axis: 1, keepdims: true);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 0, 1 });
        }

        [TestMethod]
        public async Task Sum_Empty3D_ReturnsCorrectShapes()
        {
            // NumPy: np.sum(np.zeros((2, 0, 4)), axis=1) returns shape (2, 4)
            var arr = np.zeros(new Shape(2, 0, 4));
            var result = np.sum(arr, axis: 1);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 2, 4 });
            await Assert.That(result.size).IsEqualTo(8);
        }

        #endregion

        #region Prod Tests

        [TestMethod]
        public async Task Prod_EmptyAxis0_ReturnsOnesWithReducedShape()
        {
            // NumPy: np.prod(np.zeros((0, 3)), axis=0) returns array([1., 1., 1.]) with shape (3,)
            var arr = np.zeros(new Shape(0, 3));
            var result = np.prod(arr, axis: 0);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 3 });
            await Assert.That(result.GetDouble(0)).IsEqualTo(1.0);
            await Assert.That(result.GetDouble(1)).IsEqualTo(1.0);
            await Assert.That(result.GetDouble(2)).IsEqualTo(1.0);
        }

        [TestMethod]
        public async Task Prod_EmptyAxis1_ReturnsEmptyArray()
        {
            // NumPy: np.prod(np.zeros((0, 3)), axis=1) returns array([]) with shape (0,)
            var arr = np.zeros(new Shape(0, 3));
            var result = np.prod(arr, axis: 1);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 0 });
            await Assert.That(result.size).IsEqualTo(0);
        }

        #endregion

        #region Min/Max Tests

        [TestMethod]
        public async Task Min_EmptyAxis0_ThrowsArgumentException()
        {
            // NumPy: np.min(np.zeros((0, 3)), axis=0) raises ValueError
            var arr = np.zeros(new Shape(0, 3));

            await Assert.That(() => np.amin(arr, axis: 0))
                .Throws<ArgumentException>();
        }

        [TestMethod]
        public async Task Min_EmptyAxis1_ReturnsEmptyArray()
        {
            // NumPy: np.min(np.zeros((0, 3)), axis=1) returns array([]) with shape (0,)
            var arr = np.zeros(new Shape(0, 3));
            var result = np.amin(arr, axis: 1);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 0 });
            await Assert.That(result.size).IsEqualTo(0);
        }

        [TestMethod]
        public async Task Max_EmptyAxis0_ThrowsArgumentException()
        {
            // NumPy: np.max(np.zeros((0, 3)), axis=0) raises ValueError
            var arr = np.zeros(new Shape(0, 3));

            await Assert.That(() => np.amax(arr, axis: 0))
                .Throws<ArgumentException>();
        }

        [TestMethod]
        public async Task Max_EmptyAxis1_ReturnsEmptyArray()
        {
            // NumPy: np.max(np.zeros((0, 3)), axis=1) returns array([]) with shape (0,)
            var arr = np.zeros(new Shape(0, 3));
            var result = np.amax(arr, axis: 1);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 0 });
            await Assert.That(result.size).IsEqualTo(0);
        }

        #endregion

        #region ArgMax/ArgMin Tests

        [TestMethod]
        public async Task ArgMax_EmptyAxis0_ThrowsArgumentException()
        {
            // NumPy: np.argmax(np.zeros((0, 3)), axis=0) raises ValueError
            var arr = np.zeros(new Shape(0, 3));

            await Assert.That(() => np.argmax(arr, axis: 0))
                .Throws<ArgumentException>();
        }

        [TestMethod]
        public async Task ArgMax_EmptyAxis1_ReturnsEmptyArray()
        {
            // NumPy: np.argmax(np.zeros((0, 3)), axis=1) returns array([], dtype=int64) with shape (0,)
            var arr = np.zeros(new Shape(0, 3));
            var result = np.argmax(arr, axis: 1);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 0 });
            await Assert.That(result.size).IsEqualTo(0);
            await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);
        }

        [TestMethod]
        public async Task ArgMin_EmptyAxis0_ThrowsArgumentException()
        {
            // NumPy: np.argmin(np.zeros((0, 3)), axis=0) raises ValueError
            var arr = np.zeros(new Shape(0, 3));

            await Assert.That(() => np.argmin(arr, axis: 0))
                .Throws<ArgumentException>();
        }

        [TestMethod]
        public async Task ArgMin_EmptyAxis1_ReturnsEmptyArray()
        {
            // NumPy: np.argmin(np.zeros((0, 3)), axis=1) returns array([], dtype=int64) with shape (0,)
            var arr = np.zeros(new Shape(0, 3));
            var result = np.argmin(arr, axis: 1);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 0 });
            await Assert.That(result.size).IsEqualTo(0);
            await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);
        }

        #endregion

        #region Mean/Std/Var Tests

        [TestMethod]
        public async Task Mean_EmptyAxis0_ReturnsNaNArray()
        {
            // NumPy: np.mean(np.zeros((0, 3)), axis=0) returns array([nan, nan, nan])
            var arr = np.zeros(new Shape(0, 3));
            var result = np.mean(arr, axis: 0);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 3 });
            await Assert.That(double.IsNaN(result.GetDouble(0))).IsTrue();
            await Assert.That(double.IsNaN(result.GetDouble(1))).IsTrue();
            await Assert.That(double.IsNaN(result.GetDouble(2))).IsTrue();
        }

        [TestMethod]
        public async Task Mean_EmptyAxis1_ReturnsEmptyArray()
        {
            // NumPy: np.mean(np.zeros((0, 3)), axis=1) returns array([]) with shape (0,)
            var arr = np.zeros(new Shape(0, 3));
            var result = np.mean(arr, axis: 1);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 0 });
            await Assert.That(result.size).IsEqualTo(0);
        }

        [TestMethod]
        public async Task Std_EmptyAxis0_ReturnsNaNArray()
        {
            // NumPy: np.std(np.zeros((0, 3)), axis=0) returns array([nan, nan, nan])
            var arr = np.zeros(new Shape(0, 3));
            var result = np.std(arr, axis: 0);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 3 });
            await Assert.That(double.IsNaN(result.GetDouble(0))).IsTrue();
            await Assert.That(double.IsNaN(result.GetDouble(1))).IsTrue();
            await Assert.That(double.IsNaN(result.GetDouble(2))).IsTrue();
        }

        [TestMethod]
        public async Task Var_EmptyAxis0_ReturnsNaNArray()
        {
            // NumPy: np.var(np.zeros((0, 3)), axis=0) returns array([nan, nan, nan])
            var arr = np.zeros(new Shape(0, 3));
            var result = np.var(arr, axis: 0);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 3 });
            await Assert.That(double.IsNaN(result.GetDouble(0))).IsTrue();
            await Assert.That(double.IsNaN(result.GetDouble(1))).IsTrue();
            await Assert.That(double.IsNaN(result.GetDouble(2))).IsTrue();
        }

        #endregion

        #region Edge Cases

        [TestMethod]
        public async Task Sum_Reversed_EmptyAxis()
        {
            // NumPy: np.sum(np.zeros((3, 0)), axis=1) returns array([0., 0., 0.]) with shape (3,)
            var arr = np.zeros(new Shape(3, 0));
            var result = np.sum(arr, axis: 1);

            await Assert.That(result.shape).IsEquivalentTo(new long[] { 3 });
            await Assert.That(result.GetDouble(0)).IsEqualTo(0.0);
            await Assert.That(result.GetDouble(1)).IsEqualTo(0.0);
            await Assert.That(result.GetDouble(2)).IsEqualTo(0.0);
        }

        [TestMethod]
        public async Task Sum_NoAxis_EmptyArray_ReturnsScalar()
        {
            // NumPy: np.sum(np.zeros((0,))) returns 0.0
            var arr = np.zeros(new Shape(0));
            var result = np.sum(arr);

            await Assert.That(result.shape).IsEquivalentTo(Array.Empty<int>());
            await Assert.That((double)result).IsEqualTo(0.0);
        }

        [TestMethod]
        public async Task Prod_NoAxis_EmptyArray_ReturnsOne()
        {
            // NumPy: np.prod(np.zeros((0,))) returns 1.0
            var arr = np.zeros(new Shape(0));
            var result = np.prod(arr);

            await Assert.That((double)result).IsEqualTo(1.0);
        }

        #endregion
    }
}
