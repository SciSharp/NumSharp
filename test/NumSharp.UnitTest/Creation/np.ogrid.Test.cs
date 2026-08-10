using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    ///     np.ogrid — the open-mesh grid constructor. Every expectation below is real NumPy 2.4.2
    ///     output, transcribed with C#'s string spelling of a Python slice literal
    ///     (<c>np.ogrid[0:5]</c> → <c>np.ogrid["0:5"]</c>). A single slice yields a bare array, several
    ///     slices yield an open mesh (a tuple in NumPy, an <see cref="np.OGridResult"/> here).
    /// </summary>
    [TestClass]
    public class np_ogrid_Test
    {
        // ------------------------------------------------------------------ single slice → bare array

        [TestMethod]
        public void OGrid_SingleSlice_IsArange()
        {
            // np.ogrid[0:5] == np.arange(0, 5); the LITERAL decides the dtype.
            NDArray a = np.ogrid["0:5"];
            a.Should().BeOfType(NPTypeCode.Int64).And.BeShaped(5).And.BeOfValues(0, 1, 2, 3, 4);

            NDArray b = np.ogrid["0:5:2"];
            b.Should().BeShaped(3).And.BeOfValues(0, 2, 4);

            NDArray c = np.ogrid["5:0:-1"];
            c.Should().BeShaped(5).And.BeOfValues(5, 4, 3, 2, 1);
        }

        [TestMethod]
        public void OGrid_SingleSlice_FloatLiteralGivesFloat64()
        {
            NDArray a = np.ogrid["0.0:5"];
            a.Should().BeOfType(NPTypeCode.Double).And.BeShaped(5).And.BeOfValues(0.0, 1.0, 2.0, 3.0, 4.0);
        }

        [TestMethod]
        public void OGrid_SingleSlice_MissingStop_IsArangeFromZero()
        {
            // NumPy's except-branch: arange(start, None) == arange(0, start). np.ogrid[5:] -> [0..4].
            NDArray a = np.ogrid["5:"];
            a.Should().BeShaped(5).And.BeOfValues(0, 1, 2, 3, 4);

            NDArray b = np.ogrid["5::2"];
            b.Should().BeShaped(3).And.BeOfValues(0, 2, 4);

            // np.ogrid[:] and np.ogrid[::2] are empty (arange(0, 0)).
            ((NDArray)np.ogrid[":"]).Should().BeShaped(0);
            ((NDArray)np.ogrid["::2"]).Should().BeShaped(0);
        }

        [TestMethod]
        public void OGrid_SingleSlice_EmptyRange_IsEmpty()
        {
            ((NDArray)np.ogrid["5:0"]).Should().BeShaped(0);
            ((NDArray)np.ogrid["0:5:-1"]).Should().BeShaped(0);
        }

        [TestMethod]
        public void OGrid_SingleSlice_ImaginaryStep_IsLinspaceInclusive()
        {
            // np.ogrid[-1:1:5j] -> linspace(-1, 1, 5), stop INCLUSIVE, always float64.
            NDArray a = np.ogrid["-1:1:5j"];
            a.Should().BeOfType(NPTypeCode.Double).And.BeShaped(5)
                .And.BeOfValues(-1.0, -0.5, 0.0, 0.5, 1.0);

            // Interior points are the exact doubles NumPy 2.4.2 produces (fma-free start + i*delta).
            NDArray b = np.ogrid["-1:1:6j"];
            b.Should().BeShaped(6).And.BeOfValues(-1.0, -0.6, -0.19999999999999996,
                0.20000000000000018, 0.6000000000000001, 1.0);

            // Degenerate point counts.
            ((NDArray)np.ogrid["0:1:1j"]).Should().BeShaped(1).And.BeOfValues(0.0);
            ((NDArray)np.ogrid["0:10:0j"]).Should().BeShaped(0);
            ((NDArray)np.ogrid["2:2:3j"]).Should().BeShaped(3).And.BeOfValues(2.0, 2.0, 2.0);

            // Start defaults to 0 for an imaginary step too: np.ogrid[:5:3j] -> [0, 2.5, 5].
            ((NDArray)np.ogrid[":5:3j"]).Should().BeShaped(3).And.BeOfValues(0.0, 2.5, 5.0);
        }

        // ------------------------------------------------------------------ multi slice → open mesh

        [TestMethod]
        public void OGrid_TwoSlices_OpenMesh()
        {
            // np.ogrid[0:3, 0:5] -> (3,1) and (1,5), both int64.
            NDArray[] g = np.ogrid["0:3", "0:5"];
            g.Length.Should().Be(2);
            g[0].Should().BeOfType(NPTypeCode.Int64).And.BeShaped(3, 1).And.BeOfValues(0, 1, 2);
            g[1].Should().BeOfType(NPTypeCode.Int64).And.BeShaped(1, 5).And.BeOfValues(0, 1, 2, 3, 4);
        }

        [TestMethod]
        public void OGrid_CommaString_SplitsIntoSlices()
        {
            // A single string may carry several comma-separated slices.
            NDArray[] g = np.ogrid["0:3, 0:5"];
            g.Length.Should().Be(2);
            g[0].Should().BeShaped(3, 1).And.BeOfValues(0, 1, 2);
            g[1].Should().BeShaped(1, 5).And.BeOfValues(0, 1, 2, 3, 4);
        }

        [TestMethod]
        public void OGrid_ThreeSlices_OpenMesh()
        {
            NDArray[] g = np.ogrid["0:2", "0:3", "0:4"];
            g.Length.Should().Be(3);
            g[0].Should().BeShaped(2, 1, 1).And.BeOfValues(0, 1);
            g[1].Should().BeShaped(1, 3, 1).And.BeOfValues(0, 1, 2);
            g[2].Should().BeShaped(1, 1, 4).And.BeOfValues(0, 1, 2, 3);
        }

        [TestMethod]
        public void OGrid_MultiSlice_SharesOneDtype()
        {
            // A float in ANY slice promotes ALL to float64 (NumPy's single result_type over all bounds).
            NDArray[] a = np.ogrid["0:2", "0.0:2"];
            a[0].Should().BeOfType(NPTypeCode.Double).And.BeShaped(2, 1).And.BeOfValues(0.0, 1.0);
            a[1].Should().BeOfType(NPTypeCode.Double).And.BeShaped(1, 2).And.BeOfValues(0.0, 1.0);

            NDArray[] b = np.ogrid["0:2", "0:3:1.5"];
            b[0].Should().BeOfType(NPTypeCode.Double).And.BeOfValues(0.0, 1.0);
            b[1].Should().BeOfType(NPTypeCode.Double).And.BeOfValues(0.0, 1.5);

            // An imaginary step also forces float64 across the whole mesh.
            NDArray[] c = np.ogrid["-1:1:3j", "0:4"];
            c[0].Should().BeOfType(NPTypeCode.Double).And.BeShaped(3, 1).And.BeOfValues(-1.0, 0.0, 1.0);
            c[1].Should().BeOfType(NPTypeCode.Double).And.BeShaped(1, 4).And.BeOfValues(0.0, 1.0, 2.0, 3.0);
        }

        [TestMethod]
        public void OGrid_MultiSlice_EmptyAxis()
        {
            NDArray[] a = np.ogrid["0:0", "0:3"];
            a[0].Should().BeShaped(0, 1);
            a[1].Should().BeShaped(1, 3).And.BeOfValues(0, 1, 2);

            NDArray[] b = np.ogrid["0:3", "0:0"];
            b[0].Should().BeShaped(3, 1).And.BeOfValues(0, 1, 2);
            b[1].Should().BeShaped(1, 0);
        }

        [TestMethod]
        public void OGrid_MultiSlice_ImaginaryAndFloatValues()
        {
            NDArray[] a = np.ogrid["0:3", "0:6:3j"];
            a[0].Should().BeShaped(3, 1).And.BeOfValues(0.0, 1.0, 2.0);
            a[1].Should().BeShaped(1, 3).And.BeOfValues(0.0, 3.0, 6.0);

            NDArray[] b = np.ogrid["0.5:3.5", "0:2"];
            b[0].Should().BeShaped(3, 1).And.BeOfValues(0.5, 1.5, 2.5);
            b[1].Should().BeShaped(1, 2).And.BeOfValues(0.0, 1.0);
        }

        [TestMethod]
        public void OGrid_SliceObjects_Work()
        {
            NDArray[] g = np.ogrid[new Slice(0, 3), new Slice(0, 5)];
            g[0].Should().BeShaped(3, 1).And.BeOfValues(0, 1, 2);
            g[1].Should().BeShaped(1, 5).And.BeOfValues(0, 1, 2, 3, 4);
        }

        // ------------------------------------------------------------------ result surface

        [TestMethod]
        public void OGrid_Deconstruct()
        {
            var (y, x) = np.ogrid["0:3", "0:5"];
            y.Should().BeShaped(3, 1).And.BeOfValues(0, 1, 2);
            x.Should().BeShaped(1, 5).And.BeOfValues(0, 1, 2, 3, 4);

            var (a, b, c) = np.ogrid["0:2", "0:3", "0:4"];
            a.Should().BeShaped(2, 1, 1);
            b.Should().BeShaped(1, 3, 1);
            c.Should().BeShaped(1, 1, 4);
        }

        [TestMethod]
        public void OGrid_BroadcastsLikeMeshgrid()
        {
            // The whole point: the open mesh broadcasts to a dense grid.
            NDArray[] g = np.ogrid["0:3", "0:5"];
            NDArray sum = g[0] + g[1];
            sum.Should().BeShaped(3, 5)
                .And.BeOfValues(0, 1, 2, 3, 4,
                                1, 2, 3, 4, 5,
                                2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void OGrid_MultiSlice_ImplicitToNDArray_Throws()
        {
            // A multi-slice mesh is ambiguous as a single NDArray.
            new Action(() => { NDArray _ = np.ogrid["0:3", "0:5"]; })
                .Should().Throw<InvalidOperationException>();
        }

        // ------------------------------------------------------------------ errors

        [TestMethod]
        public void OGrid_MultiSlice_MissingStop_Throws()
        {
            // NumPy leaks an AttributeError sizing each axis; NumSharp is explicit.
            new Action(() => { NDArray[] _ = np.ogrid["0:3", "5:"]; })
                .Should().Throw<ValueError>().WithMessage("*requires an explicit stop*");
            new Action(() => { NDArray[] _ = np.ogrid["0:3", "5::2"]; })
                .Should().Throw<ValueError>().WithMessage("*requires an explicit stop*");
            new Action(() => { NDArray[] _ = np.ogrid["0:3", "5::3j"]; })
                .Should().Throw<ValueError>().WithMessage("*requires an explicit stop*");
        }

        [TestMethod]
        public void OGrid_SingleImaginary_MissingStop_Throws()
        {
            new Action(() => { NDArray _ = np.ogrid["5::3j"]; })
                .Should().Throw<ValueError>().WithMessage("*imaginary step requires a stop*");
        }

        [TestMethod]
        public void OGrid_NonSliceEntry_Throws()
        {
            new Action(() => _ = np.ogrid[5]).Should().Throw<ArgumentException>();
            new Action(() => _ = np.ogrid["abc"]).Should().Throw<ArgumentException>();
        }

        private static object _;
    }
}
