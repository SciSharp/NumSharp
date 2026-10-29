using System;
using System.Linq;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     np.mgrid / np.ogrid as NumPy's <c>nd_grid.__getitem__</c> statements. A key of one slice takes the except branch
    ///     (<c>arange(start, stop, step)</c>, or for an imaginary step <c>arange(0, n, 1, dtype=typ) * step + start</c>); a key
    ///     of several computes every size (<c>math.ceil((stop - start) / step)</c>, <c>int(abs(step))</c>) in Python's
    ///     arithmetic, the shared <c>result_type</c>, <c>indices</c> / <c>arange</c>, then <c>nn[k] = nn[k] * step + start</c>.
    ///     Every expected value (bit for bit), flag, error type and text is NumPy 2.4.2's, probed on Windows and on Linux:
    ///     the same answers on both but the int64 conversion text, where NumSharp pins Linux's (LP64) one. NumPy's
    ///     AttributeError for a missing stop in a key of several is NumSharp's ValueError (a documented divergence), raised
    ///     at the same point.
    /// </summary>
    [TestClass]
    public class np_mgrid_numpy_port_tests
    {
        private static long[] Bits(NDArray a) => a.ToArray<double>().Select(BitConverter.DoubleToInt64Bits).ToArray();

        private static long[] Bits(params double[] v) => v.Select(BitConverter.DoubleToInt64Bits).ToArray();

        /// <summary>
        ///     An imaginary step of one slice is <c>arange(0, n) * step + start</c>, a fresh array that owns its data — not
        ///     linspace, whose last point is forced onto the stop (and whose result was a view).
        /// </summary>
        [TestMethod]
        public void SingleImaginary_IsArangeTimesStepPlusStart_Owning()
        {
            // np.mgrid[0.3:0.9:7j] -> [0.3, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9000000000000001]
            NDArray a = np.mgrid["0.3:0.9:7j"];
            a.dtype.Should().Be(np.float64);
            a.flags.owndata.Should().BeTrue();
            Bits(a).Should().Equal(Bits(0.3, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9000000000000001));
            NDArray o = np.ogrid["0.3:0.9:7j"];
            o.flags.owndata.Should().BeTrue();
            Bits(o).Should().Equal(Bits(a));

            // np.mgrid[-0.0:5:1j] -> [0.0]: 0 * 1 + -0.0 is +0.0. np.mgrid[5::1j] -> [5.0]: one point needs no stop.
            Bits(np.mgrid["-0.0:5:1j"]).Should().Equal(Bits(0.0));
            Bits(np.mgrid["5::1j"]).Should().Equal(Bits(5.0));
            // np.mgrid[0:1:2.5j] -> [0., 1.] (int(2.5) points); np.mgrid[0:1:-3j] -> [0., 0.5, 1.] (abs(step))
            Bits(np.mgrid["0:1:2.5j"]).Should().Equal(Bits(0.0, 1.0));
            Bits(np.mgrid["0:1:-3j"]).Should().Equal(Bits(0.0, 0.5, 1.0));
        }

        /// <summary>
        ///     A mesh axis is <c>i * step + start</c> in the mesh dtype, not arange's <c>start + i * delta</c> (which drifts
        ///     for a float step), nor linspace for an imaginary step.
        /// </summary>
        [TestMethod]
        public void MeshAxis_IsIndexTimesStepPlusStart()
        {
            // np.mgrid[0.5:1.5:0.1, 0:2][0][:, 0]
            var expected = Bits(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3, 1.4);
            NDArray g = np.mgrid["0.5:1.5:0.1", "0:2"];
            g.Should().BeShaped(2, 10, 2);
            Bits(g[0][":, 0"].copy()).Should().Equal(expected);
            Bits(g[1][":, 1"].copy()).Should().Equal(Enumerable.Repeat(BitConverter.DoubleToInt64Bits(1.0), 10));
            // np.ogrid[0.5:1.5:0.1, 0:2][0].ravel(): the same line, a view (owndata False) of shape (10, 1)
            var (y, x) = np.ogrid["0.5:1.5:0.1", "0:2"];
            y.Should().BeShaped(10, 1);
            y.flags.owndata.Should().BeFalse();
            Bits(y.ravel()).Should().Equal(expected);
            Bits(x.ravel()).Should().Equal(Bits(0.0, 1.0));

            // np.mgrid[1:2:0.1, 0:1][0].ravel()
            Bits(np.mgrid["1:2:0.1", "0:1"][0].ravel()).Should().Equal(
                Bits(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7000000000000002, 1.8, 1.9));
            // np.ogrid[0.3:0.9:7j, 0:2][0] ends at 0.9000000000000001
            NDArray[] o = np.ogrid["0.3:0.9:7j", "0:2"];
            o[0].ravel().GetDouble(6).Should().Be(0.9000000000000001);
        }

        /// <summary>
        ///     A zero real step is Python's ZeroDivisionError from <c>(stop - start) / step</c>: <c>division by zero</c> over
        ///     ints, <c>float division by zero</c> once a float is involved — in the mesh's size loop and in a single slice's
        ///     arange.
        /// </summary>
        [TestMethod]
        public void ZeroStep_IsPythonsDivisionByZero()
        {
            new Action(() => { NDArray _ = np.mgrid["0:3:0", "0:2"]; }).Should().Throw<DivideByZeroException>()
                .WithMessage("division by zero");
            new Action(() => { NDArray[] _ = np.ogrid["0:3:0", "0:2"]; }).Should().Throw<DivideByZeroException>()
                .WithMessage("division by zero");
            new Action(() => { NDArray _ = np.mgrid["0.0:3:0", "0:2"]; }).Should().Throw<DivideByZeroException>()
                .WithMessage("float division by zero");
            new Action(() => { NDArray _ = np.mgrid["0:3:0.0", "0:2"]; }).Should().Throw<DivideByZeroException>()
                .WithMessage("float division by zero");
            new Action(() => { NDArray[] _ = np.ogrid["0.0:3:0", "0:2"]; }).Should().Throw<DivideByZeroException>()
                .WithMessage("float division by zero");
            new Action(() => { NDArray _ = np.mgrid["3:3:0"]; }).Should().Throw<DivideByZeroException>()
                .WithMessage("division by zero");
            new Action(() => { NDArray _ = np.mgrid["0:5:0.0"]; }).Should().Throw<DivideByZeroException>()
                .WithMessage("float division by zero");
        }

        /// <summary>
        ///     An imaginary step of one point needs no stop (NumPy never subtracts it), in a single slice and in a mesh; with
        ///     more points the single slice is Python's TypeError for <c>None - start</c>, and a mesh's missing stop is raised
        ///     in the rescale, after the grid exists (so a negative size wins over it in np.mgrid, not in np.ogrid).
        /// </summary>
        [TestMethod]
        public void ImaginaryStepWithoutStop_FollowsNumPysStatements()
        {
            // np.mgrid[0:2, 5::1j] -> shape (2, 2, 1): [0., 1., 5., 5.]
            NDArray g = np.mgrid["0:2", "5::1j"];
            g.Should().BeShaped(2, 2, 1);
            g.dtype.Should().Be(np.float64);
            Bits(g.ravel()).Should().Equal(Bits(0.0, 1.0, 5.0, 5.0));
            // np.ogrid[0:2, 5::1j] -> ((2, 1) [0., 1.], (1, 1) [5.])
            NDArray[] o = np.ogrid["0:2", "5::1j"];
            o[0].Should().BeShaped(2, 1);
            o[1].Should().BeShaped(1, 1);
            Bits(o[1].ravel()).Should().Equal(Bits(5.0));

            // np.mgrid[5::3j] -> TypeError: unsupported operand type(s) for -: 'NoneType' and 'int' ('float' for 5.5)
            new Action(() => { NDArray _ = np.mgrid["5::3j"]; }).Should().Throw<TypeError>()
                .WithMessage("unsupported operand type(s) for -: 'NoneType' and 'int'");
            new Action(() => { NDArray _ = np.ogrid["5.5::3j"]; }).Should().Throw<TypeError>()
                .WithMessage("unsupported operand type(s) for -: 'NoneType' and 'float'");
            new Action(() => { NDArray _ = np.mgrid["5::0j"]; }).Should().Throw<TypeError>()
                .WithMessage("unsupported operand type(s) for -: 'NoneType' and 'int'");

            // np.mgrid[0:-1, 5::3j] -> ValueError: negative dimensions are not allowed (indices runs before the rescale);
            // np.ogrid[0:-1, 5::3j] -> AttributeError (NumSharp: the missing-stop ValueError of slice 1)
            new Action(() => { NDArray _ = np.mgrid["0:-1", "5::3j"]; }).Should().Throw<ValueError>()
                .WithMessage("negative dimensions are not allowed");
            new Action(() => { NDArray[] _ = np.ogrid["0:-1", "5::3j"]; }).Should().Throw<ValueError>()
                .WithMessage("ogrid*requires an explicit stop for slice 1*");
        }

        /// <summary>
        ///     The size loop runs in key order: a zero step before a missing stop is the ZeroDivisionError, a missing stop
        ///     before a zero step the missing-stop error.
        /// </summary>
        [TestMethod]
        public void SizeLoop_RaisesInKeyOrder()
        {
            new Action(() => { NDArray _ = np.mgrid["0:3:0", "0:"]; }).Should().Throw<DivideByZeroException>()
                .WithMessage("division by zero");
            new Action(() => { NDArray _ = np.mgrid["0:", "0:3:0"]; }).Should().Throw<ValueError>()
                .WithMessage("mgrid*requires an explicit stop for slice 0*");
        }

        /// <summary>
        ///     NaN and infinite sizes are <c>math.ceil</c>'s / <c>int()</c>'s errors; a size past int64 is the dense grid's
        ///     dimension conversion (<c>Maximum allowed dimension exceeded</c>, before a negative dimension is checked) and
        ///     the open grid's arange length (<c>Maximum allowed size exceeded</c>); a huge size within int64 is the
        ///     allocation guard's.
        /// </summary>
        [TestMethod]
        public void NonFiniteAndHugeSizes_RaiseNumPysErrors()
        {
            new Action(() => { NDArray _ = np.mgrid["0:nan", "0:2"]; }).Should().Throw<ValueError>()
                .WithMessage("cannot convert float NaN to integer");
            new Action(() => { NDArray[] _ = np.ogrid["0:nan", "0:2"]; }).Should().Throw<ValueError>()
                .WithMessage("cannot convert float NaN to integer");
            new Action(() => { NDArray _ = np.mgrid["0:Infinity", "0:2"]; }).Should().Throw<OverflowException>()
                .WithMessage("cannot convert float infinity to integer");
            new Action(() => { NDArray _ = np.mgrid["0:1:nanj", "0:2"]; }).Should().Throw<ValueError>()
                .WithMessage("cannot convert float NaN to integer");
            new Action(() => { NDArray _ = np.mgrid["0:1:Infinityj"]; }).Should().Throw<OverflowException>()
                .WithMessage("cannot convert float infinity to integer");
            new Action(() => { NDArray _ = np.mgrid["0:1e300", "0:2"]; }).Should().Throw<ValueError>()
                .WithMessage("Maximum allowed dimension exceeded");
            new Action(() => { NDArray _ = np.mgrid["0:-1", "0:-1e300"]; }).Should().Throw<ValueError>()
                .WithMessage("Maximum allowed dimension exceeded");
            new Action(() => { NDArray[] _ = np.ogrid["0:1e300", "0:2"]; }).Should().Throw<ValueError>()
                .WithMessage("Maximum allowed size exceeded");
            new Action(() => { NDArray _ = np.mgrid["5:7:1e30j"]; }).Should().Throw<ValueError>()
                .WithMessage("Maximum allowed size exceeded");
            new Action(() => { NDArray _ = np.mgrid["0:4e18", "0:2"]; }).Should().Throw<ValueError>()
                .WithMessage("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.");
        }

        /// <summary>
        ///     An underflowed quotient is a size of 0 in a mesh (math.ceil of 0.0), while a single slice's arange counts one
        ///     element for it (arange's underflow rule).
        /// </summary>
        [TestMethod]
        public void UnderflowedQuotient_MeshIsEmpty_SingleSliceHasOneElement()
        {
            np.mgrid["0:1e-300:1e300", "0:2"].Grid.Should().BeShaped(2, 0, 2);
            NDArray[] o = np.ogrid["0:1e-300:1e300", "0:2"];
            o[0].Should().BeShaped(0, 1);
            Bits(np.mgrid["0:1e-300:1e300"]).Should().Equal(Bits(0.0));
            np.mgrid["0:1:Infinity", "0:2"].Grid.Should().BeShaped(2, 0, 2);
        }

        /// <summary>
        ///     Python ints are exact: sizes and int64 values past 2**53 are NumPy's; a float step makes the mesh float64
        ///     (each point rounded once); a stop past int64 only sizes the axis; a start or step past int64 in an int64 mesh
        ///     is NEP 50's conversion error (Linux NumPy's text).
        /// </summary>
        [TestMethod]
        public void PythonInts_AreExact()
        {
            NDArray g = np.mgrid["9007199254740993:9007199254740995", "0:2"];
            g.dtype.Should().Be(np.int64);
            g.ToArray<long>().Should().Equal(9007199254740993L, 9007199254740993L, 9007199254740994L, 9007199254740994L,
                0L, 1L, 0L, 1L);

            NDArray f = np.mgrid["9007199254740993:9007199254740995:1.0", "0:2"];
            f.dtype.Should().Be(np.float64);
            Bits(f.ravel()).Should().Equal(Bits(9007199254740992.0, 9007199254740992.0, 9007199254740992.0,
                9007199254740992.0, 0.0, 1.0, 0.0, 1.0));

            np.mgrid["0:9223372036854775808:4611686018427387904", "0:2"].Grid.ToArray<long>().Should().Equal(
                0L, 0L, 4611686018427387904L, 4611686018427387904L, 0L, 1L, 0L, 1L);
            np.mgrid["-9223372036854775808:-9223372036854775806", "0:1"].Grid.ToArray<long>().Should().Equal(
                long.MinValue, long.MinValue + 1, 0L, 0L);

            new Action(() => { NDArray _ = np.mgrid["9223372036854775808:9223372036854775810", "0:2"]; })
                .Should().Throw<OverflowException>().WithMessage("Python int too large to convert to C long");
            new Action(() => { NDArray[] _ = np.ogrid["9223372036854775808:9223372036854775808", "0:2"]; })
                .Should().Throw<OverflowException>().WithMessage("Python int too large to convert to C long");
        }

        /// <summary>
        ///     A -0.0 start: arange keeps it (a single real slice), a mesh axis adds it to +0.0 coordinates.
        /// </summary>
        [TestMethod]
        public void NegativeZeroStart()
        {
            // np.mgrid[-0.0:5] -> [-0., 1., 2., 3., 4.]
            Bits(np.mgrid["-0.0:5"]).Should().Equal(Bits(-0.0, 1.0, 2.0, 3.0, 4.0));
            // np.mgrid[-0.0:2, 0:1.5][0].ravel() -> [0., 0., 1., 1.]
            Bits(np.mgrid["-0.0:2", "0:1.5"][0].ravel()).Should().Equal(Bits(0.0, 0.0, 1.0, 1.0));
        }
    }
}
