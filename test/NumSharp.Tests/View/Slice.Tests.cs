using System;

namespace NumSharp.Tests.View
{
    [TestClass]
    public class SliceTests
    {
        [TestMethod]
        public void SliceNotation()
        {
            // items start through stop-1
            Assert.AreEqual("1:3", new Slice("1:3").ToString());
            Assert.AreEqual("-5:-8", new Slice("-5:-8").ToString());
            Assert.AreEqual("3:4", new Slice(3, 4).ToString());
            Assert.AreEqual("7:8:9", new Slice(7, 8, 9).ToString());
            Assert.AreEqual("7:8:9", new Slice("7:8:9").ToString());

            // items start through the rest of the array
            Assert.AreEqual("1:", new Slice("1:").ToString());
            Assert.AreEqual("1:", new Slice(1).ToString());
            Assert.AreEqual("1:", new Slice(1, null).ToString());
            Assert.AreEqual("1:", new Slice(1, null, 1).ToString());
            Assert.AreEqual("7::9", new Slice(7, null, 9).ToString());
            Assert.AreEqual("7::9", new Slice("7::9").ToString());

            // items from the beginning through stop-1
            Assert.AreEqual(":2", new Slice(":2").ToString());
            Assert.AreEqual(":2", new Slice(null, 2).ToString());
            Assert.AreEqual(":2", new Slice(stop: 2).ToString());
            Assert.AreEqual(":7:9", new Slice(null, 7, 9).ToString());
            Assert.AreEqual(":7:9", new Slice(":7:9").ToString());

            // slice a view of the whole array or matrix
            Assert.AreEqual(":", new Slice(":").ToString());
            Assert.AreEqual(":", new Slice().ToString());
            Assert.AreEqual(":", new Slice(null, null).ToString());
            Assert.AreEqual(":", new Slice(null, null, 1).ToString());

            // step
            Assert.AreEqual("::-1", new Slice("::- 1").ToString());
            Assert.AreEqual("::2", new Slice(step: 2).ToString());
            Assert.AreEqual("::2", new Slice(null, null, 2).ToString());

            // pick exactly one item and reduce the dimension
            Assert.AreEqual("17", new Slice("17").ToString());
            // pick exactly one item but keep the dimension
            Assert.AreEqual("17:18", new Slice("17:18").ToString());


            // equality
            Assert.AreEqual(new Slice("::- 1"), new Slice("::- 1"));
            Assert.AreEqual(new Slice(":"), new Slice(":"));
            Assert.AreEqual(new Slice(":7:9"), new Slice(":7:9"));
            Assert.AreEqual(new Slice(":2"), new Slice(":2"));
            Assert.AreEqual(new Slice("7::9"), new Slice("7::9"));
            Assert.AreEqual(new Slice("7:8:9"), new Slice("7:8:9"));
            Assert.AreEqual(new Slice("17"), new Slice("17"));
            Assert.AreEqual(new Slice("-5:-8"), new Slice("-5:-8"));
            Assert.AreEqual(new Slice("-  5:- 8"), new Slice("-5:-8"));
            Assert.AreEqual(new Slice("+  5:+ 8"), new Slice("+5:8"));
            Assert.AreEqual(new Slice("        5:    8"), new Slice("+5:8"));
            Assert.AreEqual(new Slice("\r\n\t:\t  "), new Slice(":"));
            Assert.AreEqual(new Slice(":\t:\t    \t1"), new Slice(":"));
            Assert.AreEqual(new Slice("  : \t:\t    \t2  "), new Slice("::2"));

            // inequality
            Assert.AreNotEqual(new Slice(":"), new Slice("1:"));
            Assert.AreNotEqual(new Slice(":1"), new Slice("1:"));
            Assert.AreNotEqual(new Slice(":8:9"), new Slice(":7:9"));
            Assert.AreNotEqual(new Slice(":7:8"), new Slice(":7:9"));
            Assert.AreNotEqual(new Slice(":-2"), new Slice(":2"));
            Assert.AreNotEqual(new Slice("7::9"), new Slice("7::19"));
            Assert.AreNotEqual(new Slice("17::9"), new Slice("7::9"));
            Assert.AreNotEqual(new Slice("7:1:9"), new Slice("7::9"));
            Assert.AreNotEqual(new Slice("7:8:9"), new Slice("7:18:9"));
            Assert.AreNotEqual(new Slice("-5:-8"), new Slice("-5:-8:2"));

            // Create functions
            Assert.AreEqual(Slice.All, new Slice(":"));
            Assert.AreEqual(Slice.None, new Slice("0:0"));
            Assert.AreEqual(Slice.Index(17), new Slice("17:18"));
            Assert.AreEqual(Slice.Ellipsis, new Slice("..."));
            Assert.AreEqual(Slice.NewAxis, new Slice("np.newaxis"));

            // invalid values
            Assert.ThrowsException<ArgumentException>(() => new Slice(""));
            Assert.ThrowsException<ArgumentException>(() => new Slice(":::"));
            Assert.ThrowsException<ArgumentException>(() => new Slice("x"));
            Assert.ThrowsException<ArgumentException>(() => new Slice("0.5:"));
            Assert.ThrowsException<ArgumentException>(() => new Slice("0.00008"));
            Assert.ThrowsException<ArgumentException>(() => new Slice("x:y:z"));
            Assert.ThrowsException<ArgumentException>(() => new Slice("209572048752047520934750283947529083475"));
            Assert.ThrowsException<ArgumentException>(() => new Slice("209572048752047520934750283947529083475:"));
            Assert.ThrowsException<ArgumentException>(() => new Slice(":209572048752047520934750283947529083475:2"));
            Assert.ThrowsException<ArgumentException>(() => new Slice("::209572048752047520934750283947529083475"));
            Assert.ThrowsException<ArgumentException>(() => new Slice(".."));
            Assert.ThrowsException<ArgumentException>(() => new Slice("...."));
        }

        [TestMethod]
        public void N_DimensionalSliceNotation()
        {
            var s = "1:3,-5:-8,7:8:9,...,1:,999,:,:1,7::9,:7:9,::-1,-5:-8,5:8,...";
            Assert.AreEqual(s, Slice.FormatSlices(Slice.ParseSlices(s)));
        }

        [TestMethod]
        public void SliceDef()
        {
            // slice sanitation (prerequisite for shape slicing and correct merging!)

            new Slice("0:10").ToSliceDef(10).Should().Be(new SliceDef { Start = 0, Step = 1, Count = 10 });
            new Slice(":").ToSliceDef(10).Should().Be(new SliceDef { Start = 0, Step = 1, Count = 10 });
            new Slice("1:9").ToSliceDef(10).Should().Be(new SliceDef { Start = 1, Step = 1, Count = 8 });
            new Slice("2:3").ToSliceDef(10).Should().Be(new SliceDef { Start = 2, Step = 1, Count = 1 });
            new Slice("3:2").ToSliceDef(10).Should().Be(new SliceDef { Start = 0, Step = 0, Count = 0 });
            new Slice("2:2").ToSliceDef(10).Should().Be(new SliceDef { Start = 0, Step = 0, Count = 0 });
            new Slice("2:2:-1").ToSliceDef(10).Should().Be(new SliceDef { Start = 0, Step = 0, Count = 0 });
            new Slice("-77:77").ToSliceDef(10).Should().Be(new SliceDef { Start = 0, Step = 1, Count = 10 });
            new Slice("77:-77").ToSliceDef(10).Should().Be(new SliceDef { Start = 0, Step = 0, Count = 0 });
            new Slice("77:-77:-1").ToSliceDef(10).Should().Be(new SliceDef { Start = 9, Step = -1, Count = 10 });
            new Slice("::77").ToSliceDef(10).Should().Be(new SliceDef { Start = 0, Step = 77, Count = 1 });
            new Slice("::-77").ToSliceDef(10).Should().Be(new SliceDef { Start = 9, Step = -77, Count = 1 });
            new Slice("::7").ToSliceDef(10).Should().Be(new SliceDef { Start = 0, Step = 7, Count = 2 });
            new Slice("::-7").ToSliceDef(10).Should().Be(new SliceDef { Start = 9, Step = -7, Count = 2 });
            new Slice("::2").ToSliceDef(10).Should().Be(new SliceDef { Start = 0, Step = 2, Count = 5 });
            new Slice("::-2").ToSliceDef(10).Should().Be(new SliceDef { Start = 9, Step = -2, Count = 5 });
            new Slice("::3").ToSliceDef(10).Should().Be(new SliceDef { Start = 0, Step = 3, Count = 4 });
            new Slice("::-3").ToSliceDef(10).Should().Be(new SliceDef { Start = 9, Step = -3, Count = 4 });
            new Slice("10:2:-7").ToSliceDef(10).Should().Be(new SliceDef { Start = 9, Step = -7, Count = 1 });
            new Slice("10:1:-7").ToSliceDef(10).Should().Be(new SliceDef { Start = 9, Step = -7, Count = 2 });
            new Slice("-7::- 1").ToSliceDef(10).Should().Be(new SliceDef { Start = 3, Step = -1, Count = 4 });
            new Slice("9:2:-2").ToSliceDef(10).Should().Be(new SliceDef { Start = 9, Step = -2, Count = 4 });
            new Slice("9:2:-2").ToSliceDef(10).Should().Be(new SliceDef(9, -2, 4));
            new Slice("9:2:-2").ToSliceDef(10).Should().Be(new SliceDef("(9>>-2*4)"));
            new Slice("-77:77:-77").ToSliceDef(10).Should().Be(new SliceDef { Start = 0, Step = 0, Count = 0 });
            new Slice("77:-77:-77").ToSliceDef(10).Should().Be(new SliceDef { Start = 9, Step = -77, Count = 1 });
            new Slice(":-5:-1").ToSliceDef(10).Should().Be(new SliceDef("(9>>-1*4)"));
            new Slice(":-6:-1").ToSliceDef(10).Should().Be(new SliceDef("(9>>-1*5)"));
        }

        /// <summary>
        ///     A zero step is NumPy's <c>ValueError: slice step cannot be zero</c> (<c>PySlice_Unpack</c>) wherever the slice
        ///     is USED — a read through the string parser, a <see cref="Slice"/> object, an <c>np.s_</c> expression, a 2-D
        ///     axis — while building the slice stays legal, as Python's <c>slice(0, 5, 0)</c> / <c>np.s_[::0]</c> are (parity
        ///     QA-01; the step used to resolve to an empty selection, so <c>a[::0]</c> was an empty view).
        /// </summary>
        [TestMethod]
        public void ZeroStep_RaisesWhenIndexing_BuildingIsLegal()
        {
            using var a = np.arange(10);
            using var m = np.arange(6).reshape(2, 3);
            var built = new Slice(0, 5, 0);
            var sexpr = np.s_["::0"];

            built.Step.Should().Be(0);
            sexpr.Length.Should().Be(1);
            foreach (Action read in new Action[]
                     {
                         () => a["::0"].Dispose(),
                         () => a["1:8:0"].Dispose(),
                         () => a[built].Dispose(),
                         () => a[sexpr].Dispose(),
                         () => m[":, ::0"].Dispose(),
                     })
                read.Should().Throw<ValueError>().WithMessage("slice step cannot be zero");
            Action resolve = () => built.ToSliceDef(10);
            resolve.Should().Throw<ValueError>().WithMessage("slice step cannot be zero");
        }

        /// <summary>
        ///     A zero-step WRITE raises the same ValueError before anything is written (it used to return normally and drop
        ///     the assignment silently).
        /// </summary>
        [TestMethod]
        public void ZeroStep_Set_RaisesBeforeWriting()
        {
            using var a = np.arange(10);

            Action act = () => a["1:8:0"] = 5;

            act.Should().Throw<ValueError>().WithMessage("slice step cannot be zero");
            a.ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L);
        }
    }
}
