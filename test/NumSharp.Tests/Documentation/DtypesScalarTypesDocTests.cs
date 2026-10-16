using System;
using System.Linq;
using NumSharp;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for the "Scalar Types and Type Checks" section of <c>docs/website-src/docs/dtypes.md</c>
    ///     (NumPy's scalar classes as <see cref="NDScalarType"/>, the type tests, the type tables and <c>np.typing</c>).
    ///     Each test mirrors one documented snippet and asserts what the page claims, so the page cannot drift from the
    ///     library; the claimed values are NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class DtypesScalarTypesDocTests
    {
        /// <summary>The type tests of the first snippet answer what the page's comments say.</summary>
        [TestMethod]
        public void TypeChecks_AsDocumented()
        {
            using var a = np.array(new[] { 1.0, 2.0, 3.0 });
            np.issubdtype(a.dtype, np.integer).Should().BeFalse();
            np.issubdtype(np.float32, np.floating).Should().BeTrue();
            np.issubdtype(np.bool_, np.number).Should().BeFalse();
            np.issubdtype("m8[s]", np.signedinteger).Should().BeTrue();

            using var mean = a.mean();
            np.floating.__instancecheck__(mean).Should().BeTrue();
            np.integer.__subclasscheck__(a.dtype.type).Should().BeFalse();

            NDScalarType t = a.dtype;
            t.ToString().Should().Be("<class 'numpy.float64'>");
            t.__mro__.Select(c => c.__name__).Should().Equal("float64", "floating", "inexact", "number", "generic", "float", "object");
            np.floating.__subclasses__().Select(c => c.__name__).Should().Equal("float16", "float32", "float64", "decimal");
            ReferenceEquals((DType)t, np.float64).Should().BeTrue();
        }

        /// <summary>The page's built-in and dtype-less classes snippet: Python's classes in the hierarchy, np.complex64 without a dtype.</summary>
        [TestMethod]
        public void BuiltinAndDtypelessClasses_AsDocumented()
        {
            NDScalarType.PyFloat.__subclasscheck__(np.float64).Should().BeTrue();
            NDScalarType.PyFloat.__subclasscheck__(np.float32).Should().BeFalse();
            np.issubdtype(NDScalarType.PyFloat, np.floating).Should().BeTrue();
            np.issubdtype(NDScalarType.PyBool, NDScalarType.PyInt).Should().BeFalse();
            np.issubdtype(np.complex64, np.complexfloating).Should().BeTrue();
            np.finfo(np.complex64).dtype.Should().Be(np.float32);
            new Action(() => np.dtype(np.complex64)).Should().Throw<NotSupportedException>();
        }

        /// <summary>The page's rules: an abstract class is not a dtype, a string is a dtype string, isdtype compares identity, a C# value is a scalar.</summary>
        [TestMethod]
        public void Rules_AsDocumented()
        {
            new Action(() => np.dtype(np.integer)).Should().Throw<TypeError>();
            np.issubdtype(np.float32, "float").Should().BeFalse();
            np.isdtype(np.int64, np.integer).Should().BeFalse();
            np.isdtype(np.int64, "integral").Should().BeTrue();
            np.floating.__instancecheck__(2.5).Should().BeTrue();
            using var arr = np.arange(4);
            using var first = arr[0];
            using var explicitArray = np.array(5);
            np.integer.__instancecheck__(first).Should().BeTrue();
            np.integer.__instancecheck__(explicitArray).Should().BeFalse();
        }

        /// <summary>The type tables and the np.typing snippet.</summary>
        [TestMethod]
        public void TablesAndTyping_AsDocumented()
        {
            np.typecodes["AllFloat"].Should().Be("efdgFDG");
            ReferenceEquals(np.sctypeDict["float64"], np.float64).Should().BeTrue();
            np.True_.Should().BeTrue();
            np.False_.Should().BeFalse();
            ReferenceEquals(np.longdouble, np.float64).Should().BeTrue();

            np.typing.NDArray.MakeGenericType(typeof(double)).Should().Be(typeof(NumSharp.Generic.NDArray<double>));
            np.typing.ArrayLike.Should().Be(typeof(NDArray));
            np.typing.DTypeLike.Should().Be(typeof(DType));

            // static NDArray scale(NDArray x, DType dtype = null) => np.asarray(x, dtype) * 2.0;
            static NDArray scale(NDArray x, DType dtype = null) => np.asarray(x, dtype) * 2.0;
            using var ints = np.array(new[] { 1, 2, 3 });
            using var scaled = scale(ints, np.float64);
            scaled.dtype.Should().Be(np.float64);
            scaled.ToArray<double>().Should().Equal(2.0, 4.0, 6.0);
        }
    }
}
