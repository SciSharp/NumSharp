using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     np.mgrid — the dense mesh-grid constructor. Every expectation below is real NumPy 2.4.2
    ///     output, transcribed with C#'s string spelling of a Python slice literal
    ///     (<c>np.mgrid[0:5]</c> → <c>np.mgrid["0:5"]</c>). A single slice yields a bare 1-D array;
    ///     several slices yield the stacked <c>(N, *sizes)</c> grid (a single <see cref="NDArray"/>,
    ///     wrapped in <see cref="np.MGridResult"/>).
    /// </summary>
    [TestClass]
    public class np_mgrid_Test
    {
        // ------------------------------------------------------------------ single slice → bare array

        [TestMethod]
        public void MGrid_SingleSlice_IsArange()
        {
            NDArray a = np.mgrid["0:5"];
            a.Should().BeOfType(NPTypeCode.Int64).And.BeShaped(5).And.BeOfValues(0, 1, 2, 3, 4);

            ((NDArray)np.mgrid["0:5:2"]).Should().BeShaped(3).And.BeOfValues(0, 2, 4);
            ((NDArray)np.mgrid["5:0:-1"]).Should().BeShaped(5).And.BeOfValues(5, 4, 3, 2, 1);
        }

        [TestMethod]
        public void MGrid_SingleSlice_FloatAndImaginary()
        {
            ((NDArray)np.mgrid["0.0:5"]).Should().BeOfType(NPTypeCode.Double)
                .And.BeShaped(5).And.BeOfValues(0.0, 1.0, 2.0, 3.0, 4.0);

            // Imaginary step -> linspace, stop inclusive, float64.
            ((NDArray)np.mgrid["-1:1:5j"]).Should().BeOfType(NPTypeCode.Double)
                .And.BeShaped(5).And.BeOfValues(-1.0, -0.5, 0.0, 0.5, 1.0);
        }

        [TestMethod]
        public void MGrid_SingleSlice_MissingStopAndEmpty()
        {
            ((NDArray)np.mgrid["5:"]).Should().BeShaped(5).And.BeOfValues(0, 1, 2, 3, 4);
            ((NDArray)np.mgrid[":"]).Should().BeShaped(0);
            ((NDArray)np.mgrid["5:0"]).Should().BeShaped(0);
        }

        // ------------------------------------------------------------------ multi slice → dense stack

        [TestMethod]
        public void MGrid_TwoSlices_StackedGrid()
        {
            // np.mgrid[0:5, 0:3] -> shape (2, 5, 3), int64.
            NDArray g = np.mgrid["0:5", "0:3"];
            g.Should().BeOfType(NPTypeCode.Int64).And.BeShaped(2, 5, 3)
                .And.BeOfValues(
                    // g[0]: row index broadcast across columns
                    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                    // g[1]: column index broadcast across rows
                    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2);
        }

        [TestMethod]
        public void MGrid_TwoSlices_SubGrids()
        {
            // aa, bb = np.mgrid[0:5, 0:3]  (the classic index-grid pair, now int64)
            var (aa, bb) = np.mgrid["0:5", "0:3"];
            aa.Should().BeShaped(5, 3).And.BeOfValues(
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4);
            bb.Should().BeShaped(5, 3).And.BeOfValues(
                0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2);

            var (cc, dd) = np.mgrid["0:3", "0:5"];
            cc.Should().BeShaped(3, 5).And.BeOfValues(
                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2);
            dd.Should().BeShaped(3, 5).And.BeOfValues(
                0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4);
        }

        [TestMethod]
        public void MGrid_CommaString_SplitsIntoSlices()
        {
            NDArray g = np.mgrid["0:5, 0:3"];
            g.Should().BeShaped(2, 5, 3);
        }

        [TestMethod]
        public void MGrid_ThreeSlices_StackedGrid()
        {
            NDArray g = np.mgrid["0:2", "0:3", "0:4"];
            g.Should().BeShaped(3, 2, 3, 4).And.BeOfType(NPTypeCode.Int64);

            var (i, j, k) = np.mgrid["0:2", "0:3", "0:4"];
            i.Should().BeShaped(2, 3, 4);
            j.Should().BeShaped(2, 3, 4);
            k.Should().BeShaped(2, 3, 4);
        }

        [TestMethod]
        public void MGrid_MultiSlice_SharesOneDtype()
        {
            // A float in ANY slice promotes the WHOLE dense grid to float64.
            NDArray g = np.mgrid["0:2", "0.0:2"];
            g.Should().BeOfType(NPTypeCode.Double).And.BeShaped(2, 2, 2)
                .And.BeOfValues(0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0);

            // Imaginary step also floats the whole grid.
            NDArray h = np.mgrid["-1:1:3j", "0:4"];
            h.Should().BeOfType(NPTypeCode.Double).And.BeShaped(2, 3, 4)
                .And.BeOfValues(
                    -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                    0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0);
        }

        [TestMethod]
        public void MGrid_MultiSlice_EmptyAxis()
        {
            NDArray g = np.mgrid["0:0", "0:3"];
            g.Should().BeShaped(2, 0, 3);
        }

        [TestMethod]
        public void MGrid_SliceObjects_Work()
        {
            NDArray g = np.mgrid[new Slice(0, 5), new Slice(0, 3)];
            g.Should().BeShaped(2, 5, 3);
        }

        // ------------------------------------------------------------------ result surface

        [TestMethod]
        public void MGrid_ImplicitNDArray_IsTheStack()
        {
            NDArray g = np.mgrid["0:5", "0:3"];
            g.ndim.Should().Be(3);
            g[0].Should().BeShaped(5, 3);
            g[1].Should().BeShaped(5, 3);
        }

        [TestMethod]
        public void MGrid_ReconstructsMeshgrid()
        {
            // rows + cols reconstructs the additive grid, like meshgrid broadcasting.
            var (rows, cols) = np.mgrid["0:3", "0:5"];
            NDArray sum = rows + cols;
            sum.Should().BeShaped(3, 5).And.BeOfValues(
                0, 1, 2, 3, 4,
                1, 2, 3, 4, 5,
                2, 3, 4, 5, 6);
        }

        // ------------------------------------------------------------------ errors

        [TestMethod]
        public void MGrid_MultiSlice_MissingStop_Throws()
        {
            new Action(() => { NDArray _ = np.mgrid["0:3", "5:"]; })
                .Should().Throw<ValueError>().WithMessage("mgrid*requires an explicit stop*");
            new Action(() => { NDArray _ = np.mgrid["0:3", "5::3j"]; })
                .Should().Throw<ValueError>().WithMessage("mgrid*requires an explicit stop*");
        }

        [TestMethod]
        public void MGrid_SingleImaginary_MissingStop_Throws()
        {
            new Action(() => { NDArray _ = np.mgrid["5::3j"]; })
                .Should().Throw<ValueError>().WithMessage("*imaginary step requires a stop*");
        }

        [TestMethod]
        public void MGrid_NonSliceEntry_Throws()
        {
            new Action(() => _ = np.mgrid[5]).Should().Throw<ArgumentException>()
                .WithMessage("mgrid*");
            new Action(() => _ = np.mgrid["abc"]).Should().Throw<ArgumentException>()
                .WithMessage("mgrid*");
        }

        private static object _;
    }
}
