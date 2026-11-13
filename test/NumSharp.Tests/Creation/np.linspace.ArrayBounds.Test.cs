using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.Creation;

/// <summary>
///     <c>np.linspace</c> over NumPy's array_like ends: the computation dtype <c>result_type(start, stop, 1.0)</c> (float32
///     ends compute in float32), array ends with the new sample axis and <c>axis</c>, <c>retstep</c>, NumPy's errors, result
///     ownership and the matrix wrap. Every expectation is NumPy 2.4.2's output for the same call (probed; float bits read
///     through <c>.view(np.uint32)</c> / <c>.view(np.uint16)</c>).
/// </summary>
[TestClass]
public class np_linspace_ArrayBounds_Test
{
    /// <summary>The float32 bits of every element.</summary>
    /// <param name="a">A float32 array.</param>
    /// <returns>The bits, C order.</returns>
    private static uint[] Bits32(NDArray a) => a.ToArray<float>().Select(BitConverter.SingleToUInt32Bits).ToArray();

    /// <summary>
    ///     Gym.NET's N-4 case: <c>np.linspace(np.float32(0.1), np.float32(0.7), 11)</c> computes in float32, and four of its
    ///     eleven samples differ from the float64 samples cast to float32.
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void Float32Ends_ComputeInFloat32()
    {
        using NDArray start = NDArray.Scalar(0.1f), stop = NDArray.Scalar(0.7f);
        using var samples = np.linspace(start, stop, 11);
        samples.typecode.Should().Be(NPTypeCode.Single);
        Bits32(samples).Should().Equal(1036831949u, 1042536202u, 1046562734u, 1049582632u, 1051595898u, 1053609164u,
            1055622429u, 1057300152u, 1058306785u, 1059313418u, 1060320051u);

        using var viaFloat64 = np.linspace(0.1f, 0.7f, 11).astype(np.float32);
        var differ = Enumerable.Range(0, 11).Where(i => Bits32(viaFloat64)[i] != Bits32(samples)[i]).ToArray();
        differ.Should().Equal(3, 4, 5, 6);
    }

    /// <summary>
    ///     A C# number is a WEAK Python number: two of them compute in float64 (a C# <c>float</c> included), beside a
    ///     float32 end it takes float32, and an integer literal stays an exact Python int (no float32 rounding past
    ///     2^24: <c>np.linspace(-20000001, 20000001, 3)</c> keeps both ends).
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void CSharpNumbers_ArePythonNumbers()
    {
        using (var ints = np.linspace(0, 10, 5))
        {
            ints.typecode.Should().Be(NPTypeCode.Double);
            ints.ToArray<double>().Should().Equal(0.0, 2.5, 5.0, 7.5, 10.0);
        }
        using (var floats = np.linspace(0.1f, 0.7f, 3))
            floats.typecode.Should().Be(NPTypeCode.Double);
        using (var exact = np.linspace(-20000001, 20000001, 3))
            exact.ToArray<double>().Should().Equal(-20000001.0, 0.0, 20000001.0);
        using (var open = np.linspace(0, 100000001, 5, endpoint: false))
            open.GetDouble(1).Should().Be(20000000.2);

        // np.linspace(0.1, np.float32(0.7), 4): the Python float start takes float32.
        using NDArray stop = NDArray.Scalar(0.7f);
        using var mixed = np.linspace(0.1, stop, 4);
        mixed.typecode.Should().Be(NPTypeCode.Single);
        Bits32(mixed).Should().Equal(1036831949u, 1050253721u, 1056964607u, 1060320051u);

        // np.linspace(np.float32(0), np.float64(1.5), 4): two strong ends promote to float64.
        using NDArray f32 = NDArray.Scalar(0f), f64 = NDArray.Scalar(1.5);
        using var promoted = np.linspace(f32, f64, 4);
        promoted.typecode.Should().Be(NPTypeCode.Double);
    }

    /// <summary>
    ///     float16 ends compute in float16 (NumPy's bits), and a complex end computes in complex128
    ///     (<c>np.linspace(0, 1+1j, 3)</c> is <c>[0, 0.5+0.5j, 1+1j]</c>).
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void Float16AndComplexEnds_KeepTheirDtype()
    {
        using NDArray h0 = NDArray.Scalar((Half)0.1), h1 = NDArray.Scalar((Half)0.7);
        using var half = np.linspace(h0, h1, 5);
        half.typecode.Should().Be(NPTypeCode.Half);
        half.ToArray<Half>().Select(BitConverter.HalfToUInt16Bits).Should().Equal((ushort)11878, (ushort)13312, (ushort)13926,
            (ushort)14439, (ushort)14746);

        using var complex = np.linspace(0, new Complex(1, 1), 3);
        complex.typecode.Should().Be(NPTypeCode.Complex);
        complex.ToArray<Complex>().Should().Equal(Complex.Zero, new Complex(0.5, 0.5), new Complex(1, 1));
    }

    /// <summary>
    ///     Array ends add the sample axis in front of their broadcast shape; <c>axis</c> moves it (1 and -1 give the same
    ///     (2, 3) result), and NumPy's float32 arithmetic holds per column.
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void ArrayEnds_AddTheSampleAxis()
    {
        using (var columns = np.linspace(np.array(new[] { 0f, 1f }), np.array(new[] { 1f, 3f }), 3))
        {
            columns.shape.Should().Equal(3L, 2L);
            columns.typecode.Should().Be(NPTypeCode.Single);
            columns.ToArray<float>().Should().Equal(0f, 1f, 0.5f, 2f, 1f, 3f);
        }
        using (var broadcast = np.linspace(np.array(new[,] { { 0.0 }, { 1.0 } }), np.array(new[] { 1.0, 2.0, 3.0 }), 2))
        {
            broadcast.shape.Should().Equal(2L, 2L, 3L);
            broadcast.ToArray<double>().Should().Equal(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0);
        }
        // Bool ends subtract in float64 (np.subtract(stop, start, dtype=float64)), never in the refused bool loop.
        using (var flags = np.linspace(np.array(new[] { true, false }), np.array(new[] { false, true }), 3))
        {
            flags.typecode.Should().Be(NPTypeCode.Double);
            flags.ToArray<double>().Should().Equal(1.0, 0.0, 0.5, 0.5, 0.0, 1.0);
        }
        foreach (int axis in new[] { 1, -1 })
        {
            using var rows = np.linspace(np.array(new[] { 0.0, 1.0 }), np.array(new[] { 1.0, 3.0 }), 3, axis: axis);
            rows.shape.Should().Equal(2L, 3L);
            rows.ToArray<double>().Should().Equal(0.0, 0.5, 1.0, 1.0, 2.0, 3.0);
        }
        using var open = np.linspace(np.array(new[] { 0.1f, 1.0f }), np.array(new[] { 0.7f, -2.5f }), 4, endpoint: false, axis: -1);
        open.shape.Should().Equal(2L, 4L);
        Bits32(open).Should().Equal(1036831949u, 1048576000u, 1053609164u, 1057803469u, 1065353216u, 1040187392u,
            3208642560u, 3218079744u);
    }

    /// <summary>
    ///     The step-free and degenerate cases: no samples is (0, 2), one sample is the start, NumPy's gh-5437 order keeps a
    ///     subnormal range spread out (float64 scalars and float32 arrays), an infinite end gives NaN first, bool ends count
    ///     as 1 and 0, and the default count is 50.
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void DegenerateCounts_SubnormalsAndInfinities()
    {
        using (var none = np.linspace(np.array(new[] { 0.0, 1.0 }), np.array(new[] { 1.0, 3.0 }), 0))
            none.shape.Should().Equal(0L, 2L);
        using (var one = np.linspace(np.array(new[] { 0.0, 1.0 }), np.array(new[] { 1.0, 3.0 }), 1))
            one.ToArray<double>().Should().Equal(0.0, 1.0);
        using (var subnormal = np.linspace(0, 5e-323, 4))
            subnormal.ToArray<double>().Should().Equal(0.0, 1.5e-323, 3e-323, 5e-323);
        using (var subnormal32 = np.linspace(np.array(new[] { 0f, 0f }), np.array(new[] { 1e-45f, 1.0f }), 3))
            subnormal32.ToArray<float>().Should().Equal(0f, 0f, 0f, 0.5f, 1e-45f, 1f);
        using (NDArray zero = NDArray.Scalar(0f), inf = NDArray.Scalar(float.PositiveInfinity))
        using (var infinite = np.linspace(zero, inf, 3))
        {
            float[] values = infinite.ToArray<float>();
            float.IsNaN(values[0]).Should().BeTrue();
            values[1].Should().Be(float.PositiveInfinity);
            values[2].Should().Be(float.PositiveInfinity);
        }
        using (NDArray t = NDArray.Scalar(true), f = NDArray.Scalar(false))
        using (var flags = np.linspace(t, f, 3))
            flags.ToArray<double>().Should().Equal(1.0, 0.5, 0.0);
        using (var defaults = np.linspace(0.0, 1.0))
            defaults.size.Should().Be(50);
    }

    /// <summary>
    ///     A requested dtype casts the computed samples: an integer dtype floors first (<c>[-1, 1]</c> for -0.75 / 0.75), bool
    ///     is nonzero, a text dtype formats them.
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void RequestedDtype_FloorsCastsOrFormats()
    {
        using (var floored = np.linspace(np.array(new[] { 0, 0 }), np.array(new[] { -3, 3 }), 4, dtype: np.int32))
        {
            floored.shape.Should().Equal(4L, 2L);
            floored.typecode.Should().Be(NPTypeCode.Int32);
            floored.ToArray<int>().Should().Equal(0, 0, -1, 1, -2, 2, -3, 3);
        }
        using (var flags = np.linspace(0, 1, 3, dtype: np.@bool))
            flags.ToArray<bool>().Should().Equal(false, true, true);
        using NDArray f0 = NDArray.Scalar(0f), f1 = NDArray.Scalar(1f);
        using var text = np.linspace(f0, f1, 3, dtype: "U5");
        text.dtype.str.Should().Be("<U5");
        text.ToString().Should().Be("['0.0' '0.5' '1.0']");
    }

    /// <summary>
    ///     <c>retstep</c> returns NumPy's step with the samples: a 0-d scalar of the computation dtype for scalar ends
    ///     (float32 for float32 ends, float64 under an integer dtype), an array for array or list ends, and a NaN when there
    ///     is no step. Without <c>retstep</c> the result is the samples alone.
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void RetStep_ReturnsNumPysStep()
    {
        var (samples, step) = np.linspace(0, 1, 5, retstep: true);
        samples.ToArray<double>().Should().Equal(0.0, 0.25, 0.5, 0.75, 1.0);
        step.ndim.Should().Be(0);
        step.typecode.Should().Be(NPTypeCode.Double);
        step.GetDouble().Should().Be(0.25);

        using NDArray f0 = NDArray.Scalar(0f), f1 = NDArray.Scalar(1f);
        var single = np.linspace(f0, f1, 5, retstep: true);
        single.step.typecode.Should().Be(NPTypeCode.Single);
        single.step.GetSingle().Should().Be(0.25f);

        var arrays = np.linspace(np.array(new[] { 0.0, 1.0 }), np.array(new[] { 1.0, 3.0 }), 3, retstep: true);
        arrays.samples.shape.Should().Equal(3L, 2L);
        arrays.step.ToArray<double>().Should().Equal(0.5, 1.0);

        var lists = np.linspace(new object[] { 0, 1 }, new object[] { 1, 3 }, 3, retstep: true);
        lists.step.ToArray<double>().Should().Equal(0.5, 1.0);

        double.IsNaN(np.linspace(0, 1, 1, retstep: true).step.GetDouble()).Should().BeTrue();
        double.IsNaN(np.linspace(0, 1, 0, retstep: true).step.GetDouble()).Should().BeTrue();
        np.linspace(0, 1, 1, endpoint: false, retstep: true).step.GetDouble().Should().Be(1.0);

        var integers = np.linspace(0, 10, 3, retstep: true, dtype: np.int32);
        integers.samples.typecode.Should().Be(NPTypeCode.Int32);
        integers.step.typecode.Should().Be(NPTypeCode.Double);
        integers.step.GetDouble().Should().Be(5.0);

        // The positional spelling np.linspace(0, 1, 5, True, True) binds the same overload.
        NDArray positional = np.linspace(0, 1, 5, true, true);
        positional.size.Should().Be(5);

        var bare = np.linspace(0, 1, 5, retstep: false);
        bare.step.Should().BeNull();
        Action unpack = () => { var (_, _) = bare; };
        unpack.Should().Throw<InvalidOperationException>();
    }

    /// <summary>NumPy's errors, in NumPy's order: the count first, then the promotion, the broadcast, the device, the axis.</summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void Errors_AreNumPys()
    {
        using var two = np.array(new[] { 0.0, 1.0 });
        using var three = np.array(new[] { 1.0, 2.0, 3.0 });

        Action negative = () => np.linspace(two, two, -1, device: "gpu");
        negative.Should().ThrowExactly<ValueError>().WithMessage("Number of samples, -1, must be non-negative.");

        Action text = () => np.linspace(np.array(new[] { "a" }), np.array(new[] { "b" }), 3);
        text.Should().ThrowExactly<DTypePromotionError>().WithMessage(
            "The DType <class 'numpy.dtypes._PyFloatDType'> could not be promoted by <class 'numpy.dtypes.StrDType'>. *" +
            "The full list of DTypes is: (<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes._PyFloatDType'>)");

        Action broadcast = () => np.linspace(two, three, 2);
        broadcast.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (3,) (2,) ");

        Action device = () => np.linspace(two, two, 2, device: "gpu");
        device.Should().ThrowExactly<ValueError>().WithMessage("Device not understood. Only \"cpu\" is allowed, but received: gpu");

        Action axis = () => np.linspace(two, two, 3, axis: 3);
        axis.Should().Throw<AxisError>().WithMessage("destination: axis 3 is out of bounds for array of dimension 2*");
        Action scalarAxis = () => np.linspace(0.0, 1.0, 3, axis: 1);
        scalarAxis.Should().Throw<AxisError>().WithMessage("destination: axis 1 is out of bounds for array of dimension 1*");
    }

    /// <summary>
    ///     The result owns its buffer exactly when NumPy's does: array ends and a dtype cast give a fresh array, a moved axis
    ///     a view (F-contiguous), scalar ends the ramp's view.
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void Ownership_FollowsNumPysStatements()
    {
        using var lo = np.array(new[] { 0.0, 1.0 });
        using var hi = np.array(new[] { 1.0, 3.0 });
        using (var fresh = np.linspace(lo, hi, 3))
            (fresh.flags.owndata, fresh.flags.c_contiguous).Should().Be((true, true));
        using (var moved = np.linspace(lo, hi, 3, axis: 1))
            (moved.flags.owndata, moved.flags.f_contiguous, moved.flags.c_contiguous).Should().Be((false, true, false));
        using (var cast = np.linspace(lo, hi, 3, dtype: np.float32))
            cast.flags.owndata.Should().BeTrue();
        using NDArray f0 = NDArray.Scalar(0f), f1 = NDArray.Scalar(1f);
        using var ramp = np.linspace(f0, f1, 3);
        ramp.flags.owndata.Should().BeFalse();
    }

    /// <summary>
    ///     <c>numpy.matrix</c> ends give a matrix, as NumPy's <c>_array_converter.wrap</c> does (its finalize drops the length-1
    ///     axis: (3, 2), and (1, 3) for 1 x 1 ends), and the step of matrix ends is a matrix too.
    /// </summary>
    /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from NumPy.</exception>
    [TestMethod]
    public void MatrixEnds_GiveAMatrix()
    {
        var m0 = np.matrix(new double[,] { { 0, 1 } });
        var m1 = np.matrix(new double[,] { { 1, 2 } });
        NDArray samples = np.linspace(m0, m1, 3);
        samples.Should().BeOfType<NDMatrix>();
        samples.shape.Should().Equal(3L, 2L);
        samples.ToArray<double>().Should().Equal(0.0, 1.0, 0.5, 1.5, 1.0, 2.0);

        var withStep = np.linspace((object)m0, m1, 3, retstep: true);
        withStep.step.Should().BeOfType<NDMatrix>();
        withStep.step.ToArray<double>().Should().Equal(0.5, 0.5);

        NDArray single = np.linspace(np.matrix(new double[,] { { 0 } }), np.matrix(new double[,] { { 1 } }), 3);
        single.Should().BeOfType<NDMatrix>();
        single.shape.Should().Equal(1L, 3L);
    }
}
