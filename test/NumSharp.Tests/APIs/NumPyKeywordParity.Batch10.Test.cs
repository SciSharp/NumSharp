using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     API-parity batch 10 (the owner approved fixes 9 and 10): np.require's one signature over NumPy's iterable
    ///     requirements, and np.geomspace over NumPy's array_like ends. Every expected value, dtype, flag and text is NumPy
    ///     2.4.2's (scratchpad batch10_numpy_evidence*.txt and b10_np4.py), for the call exactly as the C# fix makes it;
    ///     floating values are compared bit for bit (their shortest round-trip text) unless a test says otherwise.
    /// </summary>
    [TestClass]
    public class NumPyKeywordParityBatch10Tests
    {
        /// <summary>Asserts that <paramref name="act"/> throws <typeparamref name="T"/> with NumPy's exact text.</summary>
        /// <typeparam name="T">The exception type NumPy's error maps to.</typeparam>
        /// <param name="act">The call that must fail.</param>
        /// <param name="message">NumPy 2.4.2's text, verbatim.</param>
        /// <exception cref="AssertFailedException">The call does not throw <typeparamref name="T"/> with that text.</exception>
        private static void Throws<T>(Action act, string message) where T : Exception
            => act.Should().Throw<T>().Which.Message.Should().Be(message);

        /// <summary>The values of an array in C order as float64 round-trip texts (Python's <c>repr</c> of each), so a
        ///     comparison sees every bit.</summary>
        /// <param name="a">The array (any real dtype; float16 / float32 widen exactly).</param>
        /// <returns>One text per element.</returns>
        private static string[] Bits(NDArray a)
            => a.astype(np.float64).ToArray<double>().Select(v => double.IsNaN(v) ? "nan" : v.ToString("R", CultureInfo.InvariantCulture)).ToArray();

        /// <summary>NumPy's <c>np.asfortranarray(np.arange(6.0).reshape(2, 3))</c>.</summary>
        /// <returns>A new F-contiguous (2, 3) float64 array.</returns>
        private static NDArray Fortran() => np.asfortranarray(np.arange(6.0).reshape(2, 3));

        // ------------------------------------------------------------------------------------------------------------
        // Fix 9: np.require(a, dtype=None, requirements=None) - requirements is what NumPy iterates.
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>A string is iterated by character, as NumPy iterates a str: <c>"C"</c> and <c>"CW"</c> make the F array
        ///     C-contiguous (NumPy: <c>flags.c_contiguous</c> True for both).</summary>
        [TestMethod]
        public void Require_String_IsReadByCharacter()
        {
            np.require(Fortran(), null, "C").Shape.IsContiguous.Should().BeTrue();
            np.require(Fortran(), null, "CW").Shape.IsContiguous.Should().BeTrue();
            np.require(np.arange(6.0).reshape(2, 3), null, "c").Shape.IsContiguous.Should().BeTrue("the names are upper-cased first");
        }

        /// <summary>A string array, a <see cref="List{T}"/> and any other sequence are read name by name, aliases included
        ///     (NumPy: <c>require(xf, None, ['C', 'W'])</c> and <c>['C_CONTIGUOUS']</c> are C-contiguous; <c>('C', 'O')</c>
        ///     owns its data).</summary>
        [TestMethod]
        public void Require_Sequences_AreReadByName()
        {
            np.require(Fortran(), null, new[] { "C", "W" }).Shape.IsContiguous.Should().BeTrue();
            np.require(Fortran(), null, new[] { "C_CONTIGUOUS" }).Shape.IsContiguous.Should().BeTrue();
            np.require(Fortran(), null, new List<string> { "C", "O" }).flags.owndata.Should().BeTrue();
            var f32 = np.require(Fortran(), np.float32, requirements: "F");
            f32.dtype.Should().Be(np.float32);
            f32.Shape.IsFContiguous.Should().BeTrue();
        }

        /// <summary>An unknown name raises NumPy's <c>KeyError</c> with the upper-cased name quoted - a whole alias inside a
        ///     string is unknown at its <c>'_'</c> (NumPy: <c>KeyError: '_'</c>, <c>'X'</c>, <c>'BOGUS'</c>, <c>'ZZ'</c>).</summary>
        [TestMethod]
        public void Require_UnknownName_IsNumPysKeyError()
        {
            var x = np.arange(6.0).reshape(2, 3);
            Throws<KeyError>(() => np.require(Fortran(), null, "C_CONTIGUOUS"), "'_'");
            Throws<KeyError>(() => np.require(x, null, "x"), "'X'");
            Throws<KeyError>(() => np.require(x, null, new[] { "bogus" }), "'BOGUS'");
            Throws<KeyError>(() => np.require(x, null, new[] { "C", "zz" }), "'ZZ'");
        }

        /// <summary>An item that is not a string fails NumPy's <c>x.upper()</c>: <c>AttributeError: 'NoneType' object has no
        ///     attribute 'upper'</c> / <c>'int' ...</c>; the names are read in one pass, so an unknown name BEFORE a bad item
        ///     is the KeyError (NumPy: <c>['bogus', 5]</c> raises <c>KeyError: 'BOGUS'</c>).</summary>
        [TestMethod]
        public void Require_NonStringItems_AreNumPysAttributeError_InOrder()
        {
            var x = np.arange(6.0).reshape(2, 3);
            Throws<AttributeError>(() => np.require(x, null, new string[] { null }), "'NoneType' object has no attribute 'upper'");
            Throws<AttributeError>(() => np.require(x, null, new object[] { 5 }), "'int' object has no attribute 'upper'");
            Throws<KeyError>(() => np.require(x, null, new object[] { "bogus", 5 }), "'BOGUS'");
        }

        /// <summary>Python's truthiness picks the branch: an empty string is "no requirement" (<c>asanyarray</c>: the F
        ///     array stays F-contiguous, itself), a C/F conflict is NumPy's ValueError, and an empty lazy sequence is truthy
        ///     like a generator (NumPy: <c>require(x, None, iter([])) is x</c>).</summary>
        [TestMethod]
        public void Require_Truthiness_And_OrderConflict()
        {
            var xf = Fortran();
            ReferenceEquals(np.require(xf, null, ""), xf).Should().BeTrue();
            Throws<ValueError>(() => np.require(xf, null, "CF"), "Cannot specify both \"C\" and \"F\" order");
            var x = np.arange(6.0).reshape(2, 3);
            ReferenceEquals(np.require(x, null, Enumerable.Empty<string>()), x).Should().BeTrue();
            ReferenceEquals(np.require(x, requirements: new[] { "W" }), x).Should().BeTrue("an already-writeable array is not copied");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 10: np.geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0) - array_like ends.
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>Array ends give one progression per pair along the new first axis (NumPy:
        ///     <c>geomspace([1, 10], [1000, 10000], 4)</c> is (4, 2) <c>[[1, 10], [10, 100], [100, 1000], [1000, 10000]]</c>,
        ///     owning its data).</summary>
        [TestMethod]
        public void Geomspace_ArrayEnds()
        {
            var r = np.geomspace(new[] { 1.0, 10.0 }, new[] { 1000.0, 10000.0 }, 4);
            r.shape.Should().Equal(4L, 2L);
            r.dtype.Should().Be(np.float64);
            Bits(r).Should().Equal("1", "10", "10", "100", "100", "1000", "1000", "10000");
            r.flags.owndata.Should().BeTrue();
        }

        /// <summary><c>axis</c> moves the sample axis (NumPy's moveaxis view: OWNDATA False): <c>axis=1</c> is (2, 4); a
        ///     2-D end pair with <c>axis=2</c> is (2, 3, 3); <c>axis=-1</c> of 1-D ends is (2, 3).</summary>
        [TestMethod]
        public void Geomspace_Axis()
        {
            var a1 = np.geomspace(new[] { 1.0, 10.0 }, new[] { 1000.0, 10000.0 }, 4, axis: 1);
            a1.shape.Should().Equal(2L, 4L);
            Bits(a1).Should().Equal("1", "10", "100", "1000", "10", "100", "1000", "10000");
            a1.flags.owndata.Should().BeFalse();

            var col = np.array(new double[,] { { 1 }, { 2 } });
            np.geomspace(col, new[] { 10.0, 100.0, 1000.0 }, 3).shape.Should().Equal(3L, 2L, 3L);
            np.geomspace(col, new[] { 10.0, 100.0, 1000.0 }, 3, axis: 2).shape.Should().Equal(2L, 3L, 3L);

            var last = np.geomspace(new[] { 1.0, 2.0 }, new[] { 10.0, 20.0 }, 3, axis: -1);
            Bits(last).Should().Equal("1", "3.1622776601683795", "10", "2", "6.32455532033676", "20");
            last.flags.owndata.Should().BeFalse();
        }

        /// <summary>A number end broadcasts against an array end (NumPy: <c>geomspace(1, [100, 1000], 3)</c> is
        ///     <c>[[1, 1], [10, 31.622776601683793], [100, 1000]]</c>).</summary>
        [TestMethod]
        public void Geomspace_NumberAndArrayEnd_Broadcast()
        {
            var r = np.geomspace(1, new[] { 100.0, 1000.0 }, 3);
            r.shape.Should().Equal(3L, 2L);
            Bits(r).Should().Equal("1", "1", "10", "31.622776601683793", "100", "1000");
        }

        /// <summary>The computation dtype is NumPy's <c>result_type(start, stop, float(num), zeros((), dtype))</c>: float32
        ///     ends compute in float64 by default (<c>[[1, 2], [2.82842712474619, 3.999999999999999], [8, 8]]</c>) but in
        ///     float32 when the dtype asks for it (<c>[[1, 2], [2.8284270763397217, 3.999999761581421], [8, 8]]</c>), and a
        ///     C# number end is a Python number (float64) even with a float32 dtype (<c>geomspace(1.5, 100, 3,
        ///     dtype=float32)</c> is <c>[1.5, 12.247448921203613, 100]</c>).</summary>
        [TestMethod]
        public void Geomspace_ComputationDType()
        {
            var f32 = np.array(new float[] { 1, 2 });
            var wide = np.geomspace(f32, 8, 3);
            wide.dtype.Should().Be(np.float64);
            Bits(wide).Should().Equal("1", "2", "2.82842712474619", "3.999999999999999", "8", "8");

            var narrow = np.geomspace(f32, NDArray.Scalar(8f), 3, dtype: np.float32);
            narrow.dtype.Should().Be(np.float32);
            Bits(narrow).Should().Equal("1", "2", "2.8284270763397217", "3.999999761581421", "8", "8");

            var literal = np.geomspace(1.5, 100, 3, dtype: np.float32);
            literal.dtype.Should().Be(np.float32);
            Bits(literal).Should().Equal("1.5", "12.247448921203613", "100");

            var half = np.geomspace(NDArray.Scalar((Half)1), NDArray.Scalar((Half)8), 4, dtype: np.float16);
            half.dtype.Should().Be(np.float16);
            Bits(half).Should().Equal("1", "2", "4", "8");

            np.geomspace(np.array(new long[] { 1, 2 }), 8, 3).dtype.Should().Be(np.float64);
        }

        /// <summary>Signs and NaN per progression, as NumPy's sign rotation gives them: a negative start against a positive
        ///     stop is NaN inside (NumPy: <c>[[-1, 2], [nan, 6.32455532033676], [10, 20]]</c>), a NaN start makes its whole
        ///     column NaN; an integer dtype truncates (<c>[[1, 2], [3, 6], [10, 20]]</c>).</summary>
        [TestMethod]
        public void Geomspace_SignsNaNAndIntegerDType()
        {
            Bits(np.geomspace(new[] { -1.0, 2.0 }, new[] { 10.0, 20.0 }, 3)).Should().Equal("-1", "2", "nan", "6.32455532033676", "10", "20");
            Bits(np.geomspace(new[] { double.NaN, 2.0 }, new[] { 10.0, 20.0 }, 3)).Should().Equal("nan", "2", "nan", "6.32455532033676", "nan", "20");
            var ints = np.geomspace(new[] { 1.0, 2.0 }, new[] { 10.0, 20.0 }, 3, dtype: np.int32);
            ints.dtype.Should().Be(np.int32);
            ints.ToArray<int>().Should().Equal(1, 2, 3, 6, 10, 20);
        }

        /// <summary>The sample counts at the edges: 0 samples is (0, 2), 1 sample is the start with or without the endpoint,
        ///     a negative count is NumPy's ValueError.</summary>
        [TestMethod]
        public void Geomspace_SampleCountEdges()
        {
            np.geomspace(new[] { 1.0, 2.0 }, new[] { 10.0, 20.0 }, 0).shape.Should().Equal(0L, 2L);
            Bits(np.geomspace(new[] { 1.0, 2.0 }, new[] { 10.0, 20.0 }, 1)).Should().Equal("1", "2");
            Bits(np.geomspace(new[] { 1.0, 2.0 }, new[] { 10.0, 20.0 }, 1, endpoint: false)).Should().Equal("1", "2");
            Throws<ValueError>(() => np.geomspace(new[] { 1.0, 2.0 }, new[] { 10.0, 20.0 }, -1), "Number of samples, -1, must be non-negative.");
        }

        /// <summary>NumPy's refusals over arrays: a zero element anywhere (<c>Geometric sequence cannot include zero</c>),
        ///     ends that do not broadcast (NumPy's ValueError text, NumSharp's IncorrectShapeException), an axis out of range
        ///     for the result (<c>destination: axis 2 is out of bounds for array of dimension 2</c>).</summary>
        [TestMethod]
        public void Geomspace_ArrayRefusals()
        {
            Throws<ValueError>(() => np.geomspace(new[] { 1.0, 0.0 }, new[] { 2.0, 3.0 }), "Geometric sequence cannot include zero");
            new Action(() => np.geomspace(new[] { 1.0, 2.0 }, new[] { 10.0, 20.0, 30.0 }, 3))
                .Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (3,) (2,)*");
            new Action(() => np.geomspace(new[] { 1.0, 2.0 }, new[] { 10.0, 20.0 }, 3, axis: 2))
                .Should().Throw<AxisError>().WithMessage("*destination: axis 2 is out of bounds for array of dimension 2*");
        }

        /// <summary>Number ends keep their batch-9 answers bit for bit (the fused scalar core): NumPy's
        ///     <c>geomspace(1, 1000, 4, endpoint=False)</c>, <c>geomspace(-1000, -1, 4)</c>, the <c>axis=-1</c> view and the
        ///     zero refusal.</summary>
        [TestMethod]
        public void Geomspace_NumberEnds_Unchanged()
        {
            Bits(np.geomspace(1, 1000, 4, endpoint: false)).Should().Equal("1", "5.623413251903491", "31.622776601683793", "177.82794100389228");
            Bits(np.geomspace(-1000, -1, 4)).Should().Equal("-1000", "-100", "-10", "-1");
            var view = np.geomspace(1, 1000, 4, axis: -1);
            Bits(view).Should().Equal("1", "10", "100", "1000");
            view.flags.owndata.Should().BeFalse();
            Throws<ValueError>(() => np.geomspace(0, 1), "Geometric sequence cannot include zero");
            np.geomspace(1, 256, 9, dtype: np.int64).ToArray<long>().Should().Equal(1, 2, 4, 7, 16, 32, 63, 127, 256);
        }

        /// <summary>Complex array ends give one spiral per pair, within the complex-unary envelope of NumPy's values (NumPy:
        ///     <c>geomspace([1, -2+0j], 8, 3)</c> is <c>[[1, -2], [2.82842712474619, -2.4492935982947064e-16+4j], [8,
        ///     8]]</c>; the complex log / power routes are accurate, not NumPy's bits).</summary>
        [TestMethod]
        public void Geomspace_ComplexArrayEnds_WithinEnvelope()
        {
            var r = np.geomspace(np.array(new[] { new Complex(1, 0), new Complex(-2, 0) }), 8, 3);
            r.dtype.Should().Be(np.complex128);
            r.shape.Should().Equal(3L, 2L);
            var values = r.ToArray<Complex>();
            var expected = new[]
            {
                new Complex(1, 0), new Complex(-2, 0), new Complex(2.82842712474619, 0),
                new Complex(-2.4492935982947064e-16, 4), new Complex(8, 0), new Complex(8, 0),
            };
            for (int i = 0; i < expected.Length; i++)
                Complex.Abs(values[i] - expected[i]).Should().BeLessThan(1e-14 * System.Math.Max(1.0, Complex.Abs(expected[i])), $"element {i}");
        }
    }
}
